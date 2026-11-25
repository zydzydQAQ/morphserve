"""Offline unit tests for M2.3/M2.4 (stratifier, downstream scorers, InternVL-U quota) & M2.5 (zero-bias simulator)."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "python"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from morphserve.bench.exactness import (  # noqa: E402
    ClipEditingScorer,
    ExactnessEvaluator,
    OwlV2GenEvalScorer,
    build_m3_active_accuracy_suite,
    resolve_evaluator_model_dir,
    score_mmvet_answer,
)
from morphserve.bench.interleaved_bench import InterleavedSession, InterleavedTurn, WorkloadSuite  # noqa: E402
from morphserve.bench.run_phase2_bagel_main import (  # noqa: E402
    DEFAULT_ABLATION_SESSIONS,
    DEFAULT_MAIN_SESSIONS,
    DEFAULT_SKEW_SESSIONS,
    build_ablation_configs,
    scale_suite_sessions,
    stratify_suite,
)
from morphserve.bench.run_phase2_overnight_deep import (  # noqa: E402
    build_static_split_configs,
    run_overnight_deep,
)
from morphserve.bench.validate_phase2_internvl_u import (  # noqa: E402
    compute_internvlu_memory_quota_overrides,
    run_internvlu_generality_validation,  # noqa: F401
    verify_internvlu_memory_quota,
)
from morphserve.bench.validate_phase2_multinode_2x8 import evaluate_cross_node_routing_check  # noqa: E402
from morphserve.engine import EnginePolicyConfig  # noqa: E402
from morphserve.sim import MeasuredPhysicsProfile, run_scale_and_interconnect_sweep  # noqa: E402
from morphserve.sim.cluster_simulator import (  # noqa: E402
    HardwareInterconnectSpec,
    SimGpuMemoryLedger,
    WorkloadScaleSpec,
    simulate_cluster_point,
)


def _scratch_dir() -> str | None:
    d_root = Path("/mnt/d/Transfer/MORPHSERVE")
    if Path("/mnt/d/Transfer").is_dir():
        d_root.mkdir(parents=True, exist_ok=True)
        return str(d_root)
    return None


class TestPhase2M23M25Offline(unittest.TestCase):
    def test_stratify_suite_preserves_quadrant_balance(self) -> None:
        sessions = []
        quads = ("visual_cot",) * 12 + ("multiturn_edit",) * 12 + ("multi_image_vqa",) * 10 + ("highres_t2i",) * 6
        for i, q in enumerate(quads):
            sessions.append(
                InterleavedSession(
                    session_id=f"s_{i:02d}",
                    quadrant=q,
                    source_dataset="unit",
                    arrival_offset_s=float(i) * 0.5,
                    turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="hello")],
                )
            )
        suite = WorkloadSuite(name="full40", trace_source="BurstGPT+GenTD26", base_qps=1.0, sessions=sessions)
        sliced = stratify_suite(suite, max_sessions=20)
        assert len(sliced.sessions) == 20
        by_q = {}
        for s in sliced.sessions:
            by_q[s.quadrant] = by_q.get(s.quadrant, 0) + 1
        assert by_q == {"visual_cot": 6, "multiturn_edit": 6, "multi_image_vqa": 5, "highres_t2i": 3}

    def test_scale_suite_sessions_and_steady_state_defaults(self) -> None:
        assert DEFAULT_MAIN_SESSIONS == 96
        assert DEFAULT_ABLATION_SESSIONS == 80
        assert DEFAULT_SKEW_SESSIONS == 80
        sig = inspect.signature(run_overnight_deep)
        assert sig.parameters["main_sessions"].default == 96
        assert sig.parameters["skew_sessions"].default == 80

        sessions = []
        quads = ("visual_cot",) * 12 + ("multiturn_edit",) * 12 + ("multi_image_vqa",) * 10 + ("highres_t2i",) * 6
        for i, q in enumerate(quads):
            sessions.append(
                InterleavedSession(
                    session_id=f"s_{i:02d}",
                    quadrant=q,
                    source_dataset="unit",
                    arrival_offset_s=float(i) * 0.8,
                    turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="hello")],
                )
            )
        suite = WorkloadSuite(name="base40", trace_source="BurstGPT+GenTD26", base_qps=1.0, sessions=sessions)
        expanded = scale_suite_sessions(suite, target_sessions=96)
        assert len(expanded.sessions) == 96
        offsets = [s.arrival_offset_s for s in expanded.sessions]
        assert offsets == sorted(offsets)
        assert len({s.session_id for s in expanded.sessions}) == 96

        assert scale_suite_sessions(suite, target_sessions=0) is suite
        down = scale_suite_sessions(suite, target_sessions=20)
        assert len(down.sessions) == 20

        # Real `main_mix_80.json` -> `96`: verify exact Hamilton quadrant quotas, smooth appended `[80..95]` tail
        # interarrivals (no ~50s wraparound gap), and balanced steady-state core `[14:82]` across all four quadrants.
        from collections import Counter

        real_80 = WorkloadSuite.load_json(REPO_ROOT / "data/interleaved-bench-v1/suites/main_mix_80.json")
        real_96 = scale_suite_sessions(real_80, target_sessions=96)
        assert len(real_96.sessions) == 96
        assert Counter(s.quadrant for s in real_96.sessions) == {
            "multiturn_edit": 29,
            "visual_cot": 29,
            "multi_image_vqa": 24,
            "highres_t2i": 14,
        }
        appended_dts = [
            real_96.sessions[i].arrival_offset_s - real_96.sessions[i - 1].arrival_offset_s for i in range(80, 96)
        ]
        assert all(0.0 <= dt <= 15.0 for dt in appended_dts)
        core_counts = Counter(s.quadrant for s in real_96.sessions[14:82])
        assert all(core_counts[q] >= 12 for q in ("multiturn_edit", "visual_cot", "multi_image_vqa", "highres_t2i"))

    def test_build_ablation_configs_ladder(self) -> None:
        ladder = build_ablation_configs(num_gpus=8)
        names = [n for n, _ in ladder]
        assert names == [
            "full_morphserve",
            "no_m4_layer_chunk",
            "no_m3_cow_loopback",
            "no_m2_dual_path",
            "no_m1_tripool_vmm",
        ]
        cfgs = dict(ladder)
        # Strict single-variable Leave-One-Out checks relative to `full_morphserve`:
        assert cfgs["no_m4_layer_chunk"].preemption_mode == "non_preemptive"
        assert cfgs["no_m4_layer_chunk"].use_module3_cow_loopback
        assert cfgs["no_m4_layer_chunk"].routing_mode == "dual_path"
        assert cfgs["no_m4_layer_chunk"].use_tripool_vmm

        assert not cfgs["no_m3_cow_loopback"].use_module3_cow_loopback
        assert cfgs["no_m3_cow_loopback"].preemption_mode == "layer_chunk"
        assert cfgs["no_m3_cow_loopback"].routing_mode == "dual_path"
        assert cfgs["no_m3_cow_loopback"].use_tripool_vmm

        assert cfgs["no_m2_dual_path"].routing_mode == "sticky_least_loaded"
        assert cfgs["no_m2_dual_path"].preemption_mode == "layer_chunk"
        assert cfgs["no_m2_dual_path"].use_module3_cow_loopback
        assert cfgs["no_m2_dual_path"].use_tripool_vmm

        assert not cfgs["no_m1_tripool_vmm"].use_tripool_vmm
        assert cfgs["no_m1_tripool_vmm"].preemption_mode == "layer_chunk"
        assert cfgs["no_m1_tripool_vmm"].use_module3_cow_loopback
        assert cfgs["no_m1_tripool_vmm"].routing_mode == "dual_path"

    def test_exactness_evaluator_mmvet_and_parity(self) -> None:
        assert score_mmvet_answer("The painting is by Vincent van Gogh in 1889.", "van gogh<AND>1889") == 1.0
        assert score_mmvet_answer("It shows four apples.", "4<OR>four") == 1.0
        assert score_mmvet_answer("Unrelated answer", "van gogh<AND>1889") == 0.0
        assert score_mmvet_answer("There are 24 apples on the table.", "4") == 0.0
        assert score_mmvet_answer("Approximation is 3.14.", "3.14") == 1.0
        assert score_mmvet_answer("Population is 1,250,000.", "1250000") == 1.0
        assert score_mmvet_answer("", "") == 0.0
        assert score_mmvet_answer("non-empty response", "") == 1.0

        sessions = [
            InterleavedSession(
                session_id="vqa_0",
                quadrant="multi_image_vqa",
                source_dataset="mmvet",
                turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="who?", expected_answer="van gogh")],
            )
        ]
        ref_out = [{"session_id": "vqa_0", "status": "ok", "ar_texts": ["Vincent van Gogh"], "image_paths_out": []}]
        cand_out = [{"session_id": "vqa_0", "status": "ok", "ar_texts": ["Vincent van Gogh"], "image_paths_out": []}]
        res = ExactnessEvaluator().evaluate_parity_and_quality(sessions, ref_out, cand_out)
        assert res["lossless_pass"] is True
        assert res["ar_greedy_exact_match_pct"] == 100.0
        assert res["mmvet_delta_pct"] == 0.0

    def test_owlv2_and_clip_scorers_offline_and_custom_backends(self) -> None:
        with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmp:
            empty_staging = Path(tmp) / "empty_models"
            empty_staging.mkdir(parents=True, exist_ok=True)
            owl_off = OwlV2GenEvalScorer(allow_hf_download=False, staging_root=empty_staging)
            clip_off = ClipEditingScorer(allow_hf_download=False, staging_root=empty_staging)
            assert owl_off.available is False
            assert owl_off.backend == "unavailable_offline"
            assert clip_off.available is False
            assert clip_off.backend == "unavailable_offline"

            # Half-downloaded directory with only `config.json` (no weight files) must be rejected;
            # once `model.safetensors` exists, `resolve_evaluator_model_dir` resolves it cleanly!
            partial_dir = empty_staging / "owlv2-base-patch16-ensemble"
            partial_dir.mkdir(parents=True, exist_ok=True)
            (partial_dir / "config.json").write_text("{}", encoding="utf-8")
            assert (
                resolve_evaluator_model_dir(
                    partial_dir, "MORPHSERVE_OWLV2_DIR", "google/owlv2-base-patch16-ensemble", staging_root=empty_staging
                )
                is None
            )
            (partial_dir / "model.safetensors").write_bytes(b"00")
            assert (
                resolve_evaluator_model_dir(
                    partial_dir, "MORPHSERVE_OWLV2_DIR", "google/owlv2-base-patch16-ensemble", staging_root=empty_staging
                )
                == partial_dir
            )

            canvas = np.full((128, 128, 3), 220, dtype=np.uint8)
            # Left box: red [10..50, 20..60]; Right box: blue [70..110, 20..60]
            canvas[20:60, 10:50] = np.array([215, 40, 40], dtype=np.uint8)
            canvas[20:60, 70:110] = np.array([40, 90, 220], dtype=np.uint8)
            img = Image.fromarray(canvas)
            assert owl_off.score_image(img, {"tag": "single_object", "include": [{"class": "apple", "count": 1}]})["score"] is None

            def _mock_detector(_pil: Image.Image, queries: list[str]) -> list[dict]:
                out = []
                for q in queries:
                    if "apple" in q:
                        # Two overlapping boxes -> NMS should deduplicate to 1!
                        out.append({"query": q, "score": 0.92, "box": [10.0, 20.0, 50.0, 60.0]})
                        out.append({"query": q, "score": 0.75, "box": [12.0, 21.0, 49.0, 59.0]})
                    if "cup" in q:
                        out.append({"query": q, "score": 0.88, "box": [70.0, 20.0, 110.0, 60.0]})
                return out

            owl = OwlV2GenEvalScorer(detector_fn=_mock_detector)
            assert owl.available is True
            # Counting + exclude + NMS check
            c_ok = owl.score_image(
                img,
                {"tag": "counting", "include": [{"class": "apple", "count": 1}], "exclude": [{"class": "apple", "count": 2}]},
            )
            assert c_ok["score"] == 1.0
            # Color attribute + spatial position (`apple` left of `cup`)
            pos_ok = owl.score_image(
                img,
                {
                    "tag": "position",
                    "include": [
                        {"class": "cup", "count": 1, "color": "blue"},
                        {"class": "apple", "count": 1, "color": "red", "position": ["left of", 0]},
                    ],
                },
            )
            assert pos_ok["score"] == 1.0
            pos_fail = owl.score_image(
                img,
                {
                    "tag": "position",
                    "include": [
                        {"class": "cup", "count": 1},
                        {"class": "apple", "count": 1, "position": ["right of", 0]},
                    ],
                },
            )
            assert pos_fail["score"] == 0.0

            clip_sc = ClipEditingScorer(
                image_embed_fn=lambda im: [float(np.asarray(im).mean()), 1.0, 0.0],
                text_embed_fn=lambda tx: [float(len(tx)), 1.0, 0.0] if tx else [0.0, 0.0, 0.0],
            )
            assert clip_sc.available is True
            edit_sc = clip_sc.score_edit_turn(img, "make the apple green", source_img=img)
            assert edit_sc["clip_t_pct"] > 50.0
            assert abs(edit_sc["clip_i_pct"] - 100.0) < 1e-3
            assert clip_sc.score_edit_turn(img, "")["clip_t_pct"] == 0.0
            assert abs(clip_sc.pair_similarity_pct(img, img) - 100.0) < 1e-3

    def test_owlv2_hf_processor_padding_and_raw_fallback(self) -> None:
        """Verify multi-query unequal-length batch passes `padding`/`truncation` and survives `post_fn` failure."""
        import contextlib
        import types

        class _ArrTensor:
            def __init__(self, arr: np.ndarray) -> None:
                self._arr = np.asarray(arr)

            def to(self, _dev):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self._arr.tolist()

            def item(self):
                return self._arr.item()

            def reshape(self, *sh):
                return _ArrTensor(self._arr.reshape(*sh))

            def max(self, dim: int = -1):
                return _ArrTensor(self._arr.max(axis=dim)), _ArrTensor(self._arr.argmax(axis=dim))

            def __ge__(self, other):
                return _ArrTensor(self._arr >= other)

            def __getitem__(self, idx):
                return _ArrTensor(self._arr[idx])

        fake_torch = types.ModuleType("torch")
        fake_torch.inference_mode = contextlib.nullcontext  # type: ignore[attr-defined]
        fake_torch.is_tensor = lambda x: isinstance(x, _ArrTensor)  # type: ignore[attr-defined]
        fake_torch.tensor = lambda data, device=None: _ArrTensor(np.asarray(data))  # type: ignore[attr-defined]
        fake_torch.sigmoid = lambda t: _ArrTensor(1.0 / (1.0 + np.exp(-t._arr)))  # type: ignore[attr-defined]
        fake_torch.nonzero = lambda mask, as_tuple=False: _ArrTensor(np.argwhere(mask._arr))  # type: ignore[attr-defined]

        class _StrictClipLikeProcessor:
            def __call__(self, *, text, images, padding=False, truncation=False, return_tensors="pt"):
                # Mimic `CLIPTokenizerFast`: unequal-length multi-query batch crashes unless `padding` is enabled!
                lens = [len(q.split()) for q in text[0]]
                if len(set(lens)) > 1 and not padding:
                    raise ValueError("Unable to create tensor, activate truncation and/or padding")
                assert padding in ("max_length", True)
                assert truncation is True
                return {"input_ids": _ArrTensor(np.zeros((1, len(text[0]), 16), dtype=np.int64))}

            def post_process_grounded_object_detection(self, **_kw):
                raise RuntimeError("simulated transformers post_process signature mismatch")

        class _StubOut:
            logits = _ArrTensor(np.array([[[2.5, -3.0], [-4.0, 3.0]]], dtype=np.float32))
            pred_boxes = _ArrTensor(np.array([[[0.25, 0.5, 0.2, 0.2], [0.75, 0.5, 0.2, 0.2]]], dtype=np.float32))

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            owl = OwlV2GenEvalScorer(allow_hf_download=False)
            owl._processor = _StrictClipLikeProcessor()
            owl._model = lambda **_kw: _StubOut()
            owl._device = "cpu"
            owl.backend = "transformers_owlv2"

            img = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8))
            dets = owl.detect_queries(img, ["a photo of a cup", "a photo of a bright red fire hydrant"])
            assert len(dets) == 2
            assert {d["query"] for d in dets} == {"a photo of a cup", "a photo of a bright red fire hydrant"}

    def test_m3_active_accuracy_suite_and_tiered_ssim_exactness(self) -> None:
        bench_root = REPO_ROOT / "data" / "interleaved-bench-v1"
        suite = build_m3_active_accuracy_suite(bench_root)
        by_q: dict[str, list[InterleavedSession]] = {}
        for s in suite.sessions:
            by_q.setdefault(s.quadrant, []).append(s)

        assert len(by_q["visual_cot"]) == 48
        # Visual-CoT GenEval MUST be 2-turn (`ar_only` `<think>` >= 32 tokens -> `dit_t2i` with `geneval_meta`)
        # so `BagelCPTGAdapter._prepare_turn_inner` sees `has_hist == True` and forks `L_shared >= 32`!
        assert all(len(s.turns) == 2 for s in by_q["visual_cot"])
        assert all(
            s.turns[0].modality_plan == "ar_only"
            and len(s.turns[0].prompt.split()) >= 20
            and s.turns[1].modality_plan == "dit_t2i"
            and bool(s.turns[1].geneval_meta)
            for s in by_q["visual_cot"]
        )
        assert len(by_q["multiturn_edit"]) == 45
        assert all(len(s.turns) == 4 for s in by_q["multiturn_edit"])
        assert len(by_q["multi_image_vqa"]) == 48
        assert all(len(s.turns) == 2 for s in by_q["multi_image_vqa"])

        # Verify Tiered SSIM Red Line: Turn-0 requires >=0.99 while chained Turn-1+ requires >=0.96
        with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmp:
            tdir = Path(tmp)
            rng = np.random.default_rng(7)
            base_px = rng.integers(30, 220, size=(64, 64, 3), dtype=np.uint8)
            noisy_px = np.clip(base_px.astype(np.int16) + rng.integers(-22, 23, size=(64, 64, 3)), 0, 255).astype(np.uint8)
            p_ref = tdir / "ref.png"
            p_noisy = tdir / "noisy.png"
            Image.fromarray(base_px).save(p_ref)
            Image.fromarray(noisy_px).save(p_noisy)

            ev = ExactnessEvaluator(
                owlv2_scorer=OwlV2GenEvalScorer(
                    detector_fn=lambda _i, qs: [{"query": q, "score": 0.9, "box": [4, 4, 40, 40]} for q in qs]
                ),
                clip_scorer=ClipEditingScorer(image_embed_fn=lambda _i: [1.0, 0.0], text_embed_fn=lambda _t: [1.0, 0.0]),
            )
            s_edit = [
                InterleavedSession(
                    session_id="e0",
                    quadrant="multiturn_edit",
                    source_dataset="magicbrush",
                    turns=[
                        InterleavedTurn(turn_idx=0, modality_plan="dit_edit", prompt="t0"),
                        InterleavedTurn(turn_idx=1, modality_plan="dit_edit", prompt="t1"),
                    ],
                )
            ]
            r_edit = [{"session_id": "e0", "status": "ok", "ar_texts": [], "image_paths_out": [str(p_ref), str(p_ref)]}]
            c_edit = [{"session_id": "e0", "status": "ok", "ar_texts": [], "image_paths_out": [str(p_ref), str(p_noisy)]}]
            rep_edit = ev.evaluate_parity_and_quality(s_edit, r_edit, c_edit)
            assert 0.96 <= rep_edit["chained_multiturn_min_ssim"] < 0.99
            assert rep_edit["lossless_pass"] is True
            assert rep_edit["clip_t_delta_pct"] == 0.0

            # The exact same noisy image on Turn-0 must fail the stricter 0.99 Turn-0 SSIM gate!
            c_bad_t0 = [{"session_id": "e0", "status": "ok", "ar_texts": [], "image_paths_out": [str(p_noisy), str(p_ref)]}]
            rep_bad = ev.evaluate_parity_and_quality(s_edit, r_edit, c_bad_t0)
            assert rep_bad["turn0_min_ssim"] < 0.99
            assert rep_bad["lossless_pass"] is False

    def test_internvlu_12gib_memory_quota_verification(self) -> None:
        from morphserve.models.internvl_u_adapter import compute_internvlu_memory_plan

        ov, budget = compute_internvlu_memory_quota_overrides(
            "morphserve",
            8,
            "/shared/data/MORPHSERVE/models/InternVL-U",
            constrained_gib=12.0,
            total_gpu_gib=40.0,
        )
        assert ov["super_handle_mib"] == 16
        assert ov["allow_vmm_overflow"] is False
        assert ov["super_pool_mib"] + ov["fine_kv_pool_mib"] == 6688
        assert budget["runtime_combined_budget_mib"] == 12_288.0
        assert abs(ov["gpu_memory_fraction"] - 0.3) < 1e-6

        # On an 80 GiB GPU, `total_gpu_gib=80.0` must produce `0.15` (`12 / 80`), never `0.30` (`24 GiB`)!
        ov_80g, budget_80g = compute_internvlu_memory_quota_overrides(
            "monolithic_dp",
            8,
            "/shared/data/MORPHSERVE/models/InternVL-U",
            constrained_gib=12.0,
            total_gpu_gib=80.0,
        )
        assert abs(ov_80g["gpu_memory_fraction"] - 0.15) < 1e-6
        assert budget_80g["quota_limit_mib"] == 12_288.0

        pol = EnginePolicyConfig.from_preset("morphserve", num_gpus=8, **ov)
        # Verify end-to-end integration with `InternVLUCPTGAdapter.compute_internvlu_memory_plan` (co-paged VLM+Decoder)
        plan = compute_internvlu_memory_plan(
            total_dev_mib=40_960.0,
            free_dev_mib=35_000.0,
            base_pytorch_alloc_mib=260.0,
            mirror_bytes=4_864 * (1024**2),
            policy_super_pool_mib=pol.super_pool_mib,
            policy_fine_kv_pool_mib=pol.fine_kv_pool_mib,
            super_handle_mib=pol.super_handle_mib,
            allow_vmm_overflow=pol.allow_vmm_overflow,
            gpu_memory_fraction=pol.gpu_memory_fraction,
        )
        assert plan["quota_constrained"] is True
        assert plan["allow_overflow"] is False
        assert plan["vmm_physical_pool_mib"] == 6688
        assert plan["pytorch_cap_mib"] + plan["vmm_physical_pool_mib"] <= 12_288.0

        # Explicit decomposed (`pytorch_peak_reserved_mib` + `vmm_physical_pool_mib`) audit
        ok_audit = verify_internvlu_memory_quota(
            {"pytorch_peak_reserved_mib": 4_950.0, "vmm_physical_pool_mib": 6_688.0, "vmm_overflow_creates": 0},
            pol,
            constrained_gib=12.0,
        )
        assert ok_audit["within_quota"] is True
        assert ok_audit["combined_physical_mib"] == 11_638.0

        # Combined `ClusterRunReport.peak_gpu_reserved_mib` audit (no double-counting of VMM pool!)
        rep_audit = verify_internvlu_memory_quota(
            {"peak_gpu_reserved_mib": 11_638.0, "vmm_overflow_creates": 0},
            pol,
            constrained_gib=12.0,
        )
        assert rep_audit["within_quota"] is True
        assert rep_audit["pytorch_peak_reserved_mib"] == 4_950.0
        assert rep_audit["combined_physical_mib"] == 11_638.0

        over_audit = verify_internvlu_memory_quota(
            {"pytorch_peak_reserved_mib": 7_800.0, "vmm_physical_pool_mib": 4_832.0, "vmm_overflow_creates": 0},
            pol,
            constrained_gib=12.0,
        )
        assert over_audit["within_quota"] is False

        # Verify VLM + Decoder co-paging over a single shared NumaMemfdMirror + SimulatedTriPoolBackend
        from morphserve.vmm import ExpertPartitionSpec, LayerExpertPager, NumaMemfdMirror, SimulatedTriPoolBackend

        class _FakeParam:
            def __init__(self, nbytes: int, fill_byte: int) -> None:
                self._nbytes = nbytes
                self.shape = (nbytes // 2,)
                self.dtype = "torch.bfloat16"
                self.raw_bytes = bytes([fill_byte]) * min(nbytes, 4096)
                self.bound_view = None

            def numel(self) -> int:
                return self._nbytes // 2

            def element_size(self) -> int:
                return 2

        class _FakeMod:
            def __init__(self, params: list[tuple[str, _FakeParam]]) -> None:
                self._params = params
                self.blocks = [object(), object()]

            def named_parameters(self):
                return list(self._params)

        class _FakePipe:
            def __init__(self) -> None:
                tied_emb = _FakeParam(32 * (1024**2), 0xA5)
                self.vlm = _FakeMod(
                    [
                        ("embed_tokens.weight", tied_emb),
                        ("lm_head.weight", tied_emb),  # tied parameter deduplicated!
                        ("layers.0.weight", _FakeParam(24 * (1024**2), 0x5A)),
                        ("layers.1.weight", _FakeParam(24 * (1024**2), 0x3C)),
                    ]
                )
                self.generation_decoder = _FakeMod(
                    [
                        ("blocks.0.weight", _FakeParam(28 * (1024**2), 0x11)),
                        ("blocks.1.weight", _FakeParam(28 * (1024**2), 0x22)),
                    ]
                )
                self.vae = None

        fpipe = _FakePipe()
        dec_spec = ExpertPartitionSpec.for_internvlu(fpipe, super_handle_mib=16, wrap_vae=False, install_module_hooks=False)
        vlm_spec = ExpertPartitionSpec.for_internvlu_vlm(fpipe, super_handle_mib=16, install_module_hooks=False)
        assert sum(len(pl.params) for pl in vlm_spec.plans) == 3  # tied `lm_head` deduplicated!

        dec_pager = LayerExpertPager(dec_spec)
        vlm_pager = LayerExpertPager(vlm_spec)
        shared_m = NumaMemfdMirror.create(
            nbytes=dec_pager.total_mirror_bytes + vlm_pager.total_mirror_bytes,
            numa_node=0,
            name="unit_copage_ivlu",
        )
        sim_vmm = SimulatedTriPoolBackend(
            super_pool_mib=288,
            fine_kv_pool_mib=16,
            super_handle_mib=16,
            allow_overflow=False,
            materialize_va_bytes=True,
        )
        try:
            dec_pager.offload_to_shared_mirror(shared_m, populate=True, mirror_offset=0)
            vlm_pager.offload_to_shared_mirror(
                shared_m,
                populate=True,
                mirror_offset=dec_pager.total_mirror_bytes,
            )
            dec_pager.bind_vmm(sim_vmm, initial_alpha=0.0)
            vlm_pager.bind_vmm(sim_vmm, initial_alpha=1.0)
            assert sim_vmm.kv_free_bytes(for_dit=True) > 0

            with vlm_pager.generation_mode(prefetch_all=True):
                assert vlm_pager.resident_alpha() == 1.0
            with dec_pager.generation_mode(prefetch_all=True):
                assert dec_pager.resident_alpha() == 1.0
            # Both unpinned -> transient VAE spike can claim up to full 288 MiB super-pool with zero overflow!
            with sim_vmm.transient_scope():
                ptr = sim_vmm.allocate_transient_mib(272)
                assert ptr != 0
            assert sim_vmm.stats_dict()["overflow_creates"] == 0
        finally:
            vlm_pager.close()
            dec_pager.close()
            sim_vmm.shutdown()
            shared_m.close()

    def test_sim_gpu_memory_ledger_physical_conservation(self) -> None:
        ms_pol = EnginePolicyConfig.from_preset("morphserve", num_gpus=8)
        mono_pol = EnginePolicyConfig.from_preset("monolithic_dp", num_gpus=8)
        ms_ledger = SimGpuMemoryLedger(gpu_id=0, total_hbm_mib=40_960.0, policy=ms_pol, role="hybrid")
        mono_ledger = SimGpuMemoryLedger(gpu_id=0, total_hbm_mib=40_960.0, policy=mono_pol, role="hybrid")

        # Verify ZERO synthetic per-turn fragmentation drift across repeated KV updates
        mono_ledger.upsert_session_kv(99, 512.0)
        first_peak = mono_ledger.peak_used_mib
        for _ in range(50):
            mono_ledger.upsert_session_kv(99, 512.0)
        assert mono_ledger.peak_used_mib == first_peak
        mono_ledger.release_session(99)
        mono_ledger.peak_used_mib = 0.0

        # Under heavy multi-session KV + 1024px VAE spike (~6.5 GiB unscoped vs ~3.6 GiB scoped):
        # MorphServe auto-evicts clean tail expert handles (`expert_evictions > 0`) with zero OOM,
        # whereas unpaged monolithic_dp exceeds 40 GiB HBM cleanly from byte conservation!
        for sid in range(4):
            assert ms_ledger.upsert_session_kv(sid, 1800.0) is True
            mono_ledger.upsert_session_kv(sid, 1800.0)
        ok_ms, swapped_0 = ms_ledger.execute_dit_and_vae(1024, active_sid=0)
        assert ok_ms is True
        assert swapped_0 == 0.0
        assert ms_ledger.expert_evictions > 0
        ok_mono, _ = mono_ledger.execute_dit_and_vae(1024, active_sid=0)
        assert ok_mono is False

        # Under extreme think-time KV co-residency (>11 GiB across 8 sessions), MorphServe swaps idle KV to Host DRAM
        # (`Decision 1/6`) rather than OOMing when all 28 expert layers pin for DiT!
        for sid in range(4, 8):
            assert ms_ledger.upsert_session_kv(sid, 1800.0) is True
        ok_swap, swapped_mib = ms_ledger.execute_dit_and_vae(1024, active_sid=7)
        assert ok_swap is True
        assert swapped_mib > 0.0
        assert ms_ledger.kv_host_swap_outs > 0
        restored_mib = ms_ledger.ensure_session_resident(0)
        assert restored_mib == 1800.0
        assert ms_ledger.kv_host_swap_ins == 1

    def test_cluster_simulator_64gpu_sweep(self) -> None:
        ph = MeasuredPhysicsProfile.from_phase01_results()
        assert len(ph.provenance_files) == 7

        # Verify 1-to-1 `MorphServeClusterEngine` multi-turn `tight_mot` forward-release + post-DiT return-ship physics:
        # Session 0 (`multiturn_edit`, 4 turns) across 2 single-GPU nodes (`gpus_per_node=1`) executes 4 forward ships
        # (`path_a_ships == 4`) plus 3 post-DiT return ships on turns 0..2 (`cross_node_ships == 7`), and marks OOM
        # cleanly without exceeding physical HBM if `und` rank fills during DiT!
        hw_xnode = HardwareInterconnectSpec(
            name="unit_2node_2gpu",
            gpu_hbm_gib=40.0,
            compute_speedup_x=1.0,
            solo_h2d_gbps=10.0,
            agg_h2d_gbps=10.0,
            intra_half_p2p_gbps=10.0,
            inter_socket_p2p_gbps=10.0,
            inter_node_rdma_gbps=5.0,
            gpus_per_node=1,
        )
        r_nccl_1 = simulate_cluster_point("static_nccl_fission", 2, hw_xnode, offered_qps=0.5, num_sessions=1, physics=ph)
        r_shm_1 = simulate_cluster_point("twostage_shm", 2, hw_xnode, offered_qps=0.5, num_sessions=1, physics=ph)
        assert r_nccl_1["oom_sessions"] == 0
        assert r_nccl_1["path_a_ships"] == 4
        assert r_nccl_1["cross_node_ships"] == 7
        assert r_shm_1["oom_sessions"] == 0
        assert r_shm_1["path_a_ships"] == 4
        assert r_shm_1["cross_node_ships"] == 7
        assert r_shm_1["dit_turn_p95_ms"] > r_nccl_1["dit_turn_p95_ms"] + 5000.0

        hw_tight_und = HardwareInterconnectSpec(
            name="unit_tight_und_35g",
            gpu_hbm_gib=35.0,
            compute_speedup_x=1.0,
            solo_h2d_gbps=10.0,
            agg_h2d_gbps=10.0,
            intra_half_p2p_gbps=10.0,
            inter_socket_p2p_gbps=10.0,
            inter_node_rdma_gbps=10.0,
            gpus_per_node=2,
        )
        ws_big_kv = WorkloadScaleSpec(name="unit_big_kv", kv_per_turn_scale=8.0, think_s_range_override=(30.0, 40.0))
        for pol_oom in ("static_nccl_fission", "twostage_shm"):
            r_ret_oom = simulate_cluster_point(
                pol_oom,
                2,
                hw_tight_und,
                offered_qps=15.0,
                num_sessions=40,
                physics=ph,
                workload_scale=ws_big_kv,
                skew_tag="ar80_dit20",
            )
            assert r_ret_oom["oom_sessions"] > 0
            assert r_ret_oom["peak_hbm_mib"] <= 35.0 * 1024.0

        # Non-multiple `num_gpus` across `gpus_per_node=8` (`num_nodes=2`) must index `rdma_free_at` cleanly
        r_12g = simulate_cluster_point(
            "static_nccl_fission",
            12,
            HardwareInterconnectSpec.presets(physics=ph)[0],
            offered_qps=1.0,
            num_sessions=10,
            physics=ph,
        )
        assert r_12g["oom_sessions"] == 0

        # Verify causal discrete-event forward-release & `end_dit` return-ship ordering (`cluster_simulator.py`):
        # Session 0 (`dit_edit` -> `ar_only`) frees `und` GPU 0 across `[dit_ready_ms, end_dit]` so Session 1 arriving
        # at 400ms during Session 0's ~5s DiT executes on `und` GPU 0 with zero false co-residency OOM!
        import morphserve.sim.cluster_simulator as cs_mod

        hw_causal = HardwareInterconnectSpec(
            name="unit_causal_fwd_release_58g",
            gpu_hbm_gib=58.0,
            compute_speedup_x=1.0,
            solo_h2d_gbps=50.0,
            agg_h2d_gbps=50.0,
            intra_half_p2p_gbps=100.0,
            inter_socket_p2p_gbps=100.0,
            inter_node_rdma_gbps=100.0,
            gpus_per_node=2,
        )
        ws_causal = WorkloadScaleSpec(name="unit_causal_kv", kv_per_turn_scale=85.0, think_s_range_override=(60.0, 60.0))
        with mock.patch.object(
            cs_mod,
            "_build_session_stream",
            return_value=[
                (0.0, "multiturn_edit", (("dit_edit", 512), ("ar_only", 512))),
                (400.0, "visual_cot", (("cot_then_dit", 512),)),
            ],
        ):
            r_causal = simulate_cluster_point(
                "static_nccl_fission",
                2,
                hw_causal,
                offered_qps=0.8,
                num_sessions=2,
                physics=ph,
                workload_scale=ws_causal,
            )
        assert r_causal["oom_sessions"] == 0
        assert r_causal["oom_retries"] == 0
        assert r_causal["path_a_ships"] == 2

        with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmp:
            rep = run_scale_and_interconnect_sweep(output_dir=Path(tmp))
        assert rep["status"] == "ok"
        assert rep["zero_manual_bias_constants"] is True
        assert rep["calibration"]["zero_parameter_fitting"] is True
        assert rep["calibration"]["within_8pct_oos_gate"] is True
        assert len(rep["scaling_matrix"]) == 54
        nvlink_64 = [
            r
            for r in rep["scaling_matrix"]
            if r["interconnect"] == "H100_SXM5_NVLink900G_RDMA400G" and r["num_gpus"] == 64
        ]
        by_pol = {r["policy_name"]: r for r in nvlink_64}
        assert by_pol["morphserve"]["joint_slo_goodput_rps"] > by_pol["monolithic_dp"]["joint_slo_goodput_rps"]
        assert by_pol["morphserve"]["crossover_kv_mib"] > 1000.0
        assert set(by_pol["morphserve"]["slo_curve_by_k"].keys()) == {"1.5x", "2.0x", "2.5x", "3.0x"}

        h100_gen = rep["h100_80gb_generality"]
        r1 = [r for r in h100_gen["regime1_7b_qps_latency_curve"] if abs(r["offered_qps"] - 1.9) < 1e-6]
        r1_pol = {r["policy_name"]: r for r in r1}
        assert r1_pol["morphserve"]["joint_slo_goodput_rps"] > r1_pol["monolithic_dp"]["joint_slo_goodput_rps"]
        assert r1_pol["morphserve"]["ar_ttft_p95_ms"] < r1_pol["monolithic_dp"]["ar_ttft_p95_ms"]
        assert r1_pol["morphserve"]["ar_tpot_p95_ms"] < r1_pol["prism_tetriserve_coloc"]["ar_tpot_p95_ms"]

        for reg_key in ("regime2_scaled_56gib_weights", "regime2_long_kv_high_concurrency"):
            rows_8g = [r for r in h100_gen[reg_key] if r["num_gpus"] == 8]
            pol_8g = {r["policy_name"]: r for r in rows_8g}
            assert pol_8g["morphserve"]["oom_sessions"] == 0
            assert pol_8g["morphserve"]["vmm_expert_evictions"] > 0
            assert pol_8g["monolithic_dp"]["oom_sessions"] > 0
            assert pol_8g["morphserve"]["joint_slo_goodput_rps"] > pol_8g["monolithic_dp"]["joint_slo_goodput_rps"]
            assert all(r["peak_hbm_mib"] <= 81_920.0 for r in h100_gen[reg_key])

    def test_build_static_split_configs(self) -> None:
        splits = dict(build_static_split_configs(num_gpus=8))
        assert set(splits.keys()) == {"morphserve_elastic", "static_2und_6gen", "static_4und_4gen", "static_6und_2gen"}
        assert len(splits["static_2und_6gen"].static_und_ranks) == 2
        assert len(splits["static_6und_2gen"].static_und_ranks) == 6
        splits_16 = dict(build_static_split_configs(num_gpus=16))
        assert len(splits_16["static_2und_6gen"].static_und_ranks) == 4
        assert len(splits_16["static_6und_2gen"].static_und_ranks) == 12
        with self.assertRaises(ValueError):
            build_static_split_configs(num_gpus=6)

    def test_multinode_16gpu_cross_node_routing_check(self) -> None:
        for inter_bw in (1.1, 11.5):
            chk = evaluate_cross_node_routing_check(
                world_size=16,
                gpus_per_node=8,
                intra_bw=24.0,
                socket_bw=18.0,
                inter_bw=inter_bw,
            )
            assert chk["routing_pass"] is True
            assert chk["bridge_24mib_decision"]["dst_gpu"] == 8
            assert chk["suppressed_cross_node_1gib_kv_ship"] is True

    def test_osdi27_r2b_regression_fixes(self) -> None:
        from morphserve.engine import InProcessWorkerFleet, MorphServeClusterEngine
        from morphserve.models.base import is_oom_exception
        from morphserve.vmm import (
            ExpertPartitionSpec,
            LayerExpertPager,
            LayerPackPlan,
            NumaMemfdMirror,
            ParamSlice,
            SimulatedTriPoolBackend,
        )

        # 1. `is_oom_exception` classifies cuBLAS/cuDNN/CUDA workspace & resize allocation failures as OOM
        for oom_msg in (
            "RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)",
            "RuntimeError: cuDNN error: CUDNN_STATUS_ALLOC_FAILED",
            "RuntimeError: cudaErrorMemoryAllocation: out of memory",
            "RuntimeError: memory allocation failed",
            'RuntimeError: torch_call_dispatcher( "aten::resize_" ) failed',
            "RuntimeError: ms_vmm_expert_wait_layer(0, 25) failed (rc=-3)",
        ):
            assert is_oom_exception(RuntimeError(oom_msg)) is True
        assert is_oom_exception(RuntimeError("CUDA error: an illegal memory access was encountered")) is False

        # 2. `LayerExpertPager._wrap_vae_callable`: when `vae.encode` runs OUTSIDE `generation_mode`
        # (`was_gen_active == False`, as in `Bagel.forward_cache_update_vae`) and Arena-2 scratchpad
        # evicts tail expert layers because `free_super_handles == 0`, `_wrap_vae_callable` MUST
        # synchronously reload all `[0, prev_res)` layers once Arena-2 releases its handles!
        class _DummyVae:
            def __init__(self, vmm_ref: SimulatedTriPoolBackend) -> None:
                self._vmm = vmm_ref
                self.evicted_during_encode = False

            def encode(self, x: int) -> int:
                # Allocate 32 MiB (2 super-handles) inside transient scope -> forces eviction of tail expert layers!
                ptr = self._vmm.allocate_transient_mib(32)
                assert ptr != 0
                self.evicted_during_encode = self._vmm.expert_resident_count(0) < 4
                return x + 1

            def decode(self, z: int) -> int:
                ptr = self._vmm.allocate_transient_mib(32)
                assert ptr != 0
                return z + 1

        class _RawParam:
            def __init__(self, fill_byte: int) -> None:
                self.raw_bytes = bytes([fill_byte]) * 4096
                self.bound_view = None

        # 4 layers * 16 MiB = 64 MiB fills the entire 64 MiB super-pool (`free_super_handles == 0`)
        sim_vmm = SimulatedTriPoolBackend(
            super_pool_mib=64,
            fine_kv_pool_mib=16,
            super_handle_mib=16,
            allow_overflow=False,
            materialize_va_bytes=True,
        )
        plans = [
            LayerPackPlan(
                layer_idx=i,
                used_bytes=4096,
                params=[
                    ParamSlice(
                        name=f"l{i}.w",
                        offset=0,
                        nbytes=4096,
                        shape=(4096,),
                        dtype_str="uint8",
                        param_ref=_RawParam(0x10 + i),
                    )
                ],
            )
            for i in range(4)
        ]
        vae_mod = _DummyVae(sim_vmm)
        spec = ExpertPartitionSpec(
            name="unit_vae_encode_restore",
            plans=plans,
            layer_stride_bytes=16 * (1024**2),
            hook_modules=[],
            vae_module=vae_mod,
            install_module_hooks=False,
        )
        pager = LayerExpertPager(spec)
        mirror = NumaMemfdMirror.create(nbytes=pager.total_mirror_bytes, numa_node=0, name="unit_vae_restore")
        try:
            pager.offload_to_shared_mirror(mirror, populate=True)
            pager.bind_vmm(sim_vmm, initial_alpha=1.0)
            assert sim_vmm.expert_resident_count(pager.expert_id) == 4
            assert pager._generation_active is False

            out_enc = vae_mod.encode(41)
            assert out_enc == 42
            assert vae_mod.evicted_during_encode is True
            # Synchronous restoration when `was_gen_active == False` guarantees `language_model(mode="gen")`
            # never touches unmapped tail expert VAs!
            assert sim_vmm.expert_resident_count(pager.expert_id) == 4
            tail_view = sim_vmm.expert_layer_uint8_view(pager.expert_id, 3)
            assert bytes(tail_view[:16]) == bytes([0x13]) * 16

            # Conversely, `vae.decode` preserves asymmetric lazy residency (`resident < 4` after decode)
            out_dec = vae_mod.decode(9)
            assert out_dec == 10
            assert sim_vmm.expert_resident_count(pager.expert_id) < 4
        finally:
            pager.close()
            sim_vmm.shutdown()
            mirror.close()

        # 3. `MorphServeClusterEngine` paces `arrival_offset_s == 0.0` suites at `1 / target_qps`
        zero_offset_suite = WorkloadSuite(
            name="unit_zero_offset_pacing",
            trace_source="deterministic",
            base_qps=1.0,
            sessions=[
                InterleavedSession(
                    session_id=f"z_{i}",
                    quadrant="multi_image_vqa",
                    source_dataset="unit",
                    arrival_offset_s=0.0,
                    turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="hi", max_ar_tokens=2)],
                )
                for i in range(4)
            ],
        )
        pol_syn = EnginePolicyConfig.from_preset(
            "morphserve",
            num_gpus=2,
            model_family="synthetic",
            inter_turn_think_s_range=(0.0, 0.0),
        )
        with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmp:
            eng = MorphServeClusterEngine(
                policy=pol_syn,
                model_dir=Path(tmp),
                bench_root=Path(tmp),
                tmp_root=Path(tmp) / "tmp_eng",
                fleet=InProcessWorkerFleet(pol_syn),
            )
            try:
                eng.start()
                rep = eng.run_suite(zero_offset_suite, target_qps=20.0)
                assert rep.completed_sessions == 4
                assert rep.error_sessions == 0
                # 4 sessions paced at 20 QPS -> last session arrives at 3 / 20 = 0.15s (`wall_seconds >= 0.14s`)
                assert rep.wall_seconds >= 0.14

                # 5. Reused `MorphServeClusterEngine` resets `HalfNodeH2DPacer` analytical clock & worker `alpha`
                eng.router.pacer.reserve_slot(0, 500.0, 240_000.0)
                assert eng.router.pacer.predicted_wait_ms(0, 0.0) > 240_000.0
                asyncio.run(eng._rpc(0, {"op": "SET_ALPHA", "alpha": 1.0}))
                assert eng.worker_states[0].expert_alpha == 1.0
                rep2 = eng.run_suite(zero_offset_suite, target_qps=40.0)
                assert rep2.completed_sessions == 4
                assert rep2.error_sessions == 0
                assert eng.router.pacer.predicted_wait_ms(0, 0.0) < 1000.0
                assert abs(eng.worker_states[0].expert_alpha - float(pol_syn.initial_alphas[0])) < 1e-6
            finally:
                eng.shutdown()

        # 4. `_truncate_bagel_cache` atomically rolls back partial-layer OOM appends (`47346 != 47364` fix)
        from morphserve.models.bagel_adapter import _truncate_bagel_cache

        class _PartialNaiveCache:
            def __init__(self, n: int) -> None:
                self.num_layers = n
                self.key_cache = {
                    l: np.zeros((23673 if l < 15 else 23655, 4, 8), dtype=np.uint8) for l in range(n)
                }
                self.value_cache = {
                    l: np.zeros((23673 if l < 15 else 23655, 4, 8), dtype=np.uint8) for l in range(n)
                }

        p_cache = _PartialNaiveCache(28)
        _truncate_bagel_cache(p_cache, 23655)
        assert all(p_cache.key_cache[l].shape[0] == 23655 for l in range(28))
        assert all(p_cache.value_cache[l].shape[0] == 23655 for l in range(28))
        _truncate_bagel_cache(p_cache, 0)
        assert all(p_cache.key_cache[l] is None for l in range(28))

        # 6. `_vit_preprocess_u8_gpu` delegates directly to `TorchvisionBackend` fast processors
        import types
        from morphserve.kernels.gpu_loopback import _vit_preprocess_u8_gpu

        class TorchvisionBackend:
            pass

        class _FakeSiglipFast(TorchvisionBackend):
            def __init__(self) -> None:
                self.calls: list[str] = []

            def __call__(self, *, images, return_tensors="pt", input_data_format="channels_first"):
                self.calls.append(f"{return_tensors}:{input_data_format}")
                return types.SimpleNamespace(pixel_values=[images])

        class _FakeGpuU8:
            device = "cuda:0"

            def to(self, *, device=None, dtype=None):
                return ("tv_fast_exact", device, dtype)

        fake_torch_mod = types.ModuleType("torch")
        fake_torch_mod.float32 = "f32"  # type: ignore[attr-defined]
        fake_nn_mod = types.ModuleType("torch.nn")
        fake_f_mod = types.ModuleType("torch.nn.functional")
        fake_nn_mod.functional = fake_f_mod  # type: ignore[attr-defined]
        fake_torch_mod.nn = fake_nn_mod  # type: ignore[attr-defined]
        with mock.patch.dict(
            sys.modules,
            {"torch": fake_torch_mod, "torch.nn": fake_nn_mod, "torch.nn.functional": fake_f_mod},
        ):
            fast_proc = _FakeSiglipFast()
            out_tv = _vit_preprocess_u8_gpu(_FakeGpuU8(), fast_proc)
        assert fast_proc.calls == ["pt:channels_first"]
        assert out_tv == ("tv_fast_exact", "cuda:0", "f32")

    def test_osdi27_statistical_hygiene_peak_reset_oom_cdf_and_slo_knee(self) -> None:
        """Verify per-suite `reset_peak_memory_stats`, OOM-aware improper CDFs, and `@80%/@90%` SLO QPS interpolation."""
        import ctypes
        import tempfile
        from morphserve._core import call_json, encode_json, get_core_lib
        from morphserve.bench.interleaved_bench import InterleavedSession, InterleavedTurn, WorkloadSuite
        from morphserve.bench.run_phase2_bagel_main import _compute_sustainable_qps_summary
        from morphserve.engine import EnginePolicyConfig, MorphServeClusterEngine
        from morphserve.engine.multinode_fleet import serialize_policy_config
        from morphserve.engine.worker_runtime import InProcessWorkerFleet
        from morphserve.models import SyntheticCPTGAdapter

        # 1. `reset_peak_memory_stats` is invoked before EVERY `run_suite` (including first suite `_suites_completed == 0`)
        reset_calls: list[int] = []

        class _TrackingAdapter(SyntheticCPTGAdapter):
            def reset_peak_memory_stats(self) -> None:
                reset_calls.append(self.rank)

            def memory_and_vmm_stats(self) -> dict[str, object]:
                base = super().memory_and_vmm_stats()
                # Even if legacy `combined_physical_mib` were inflated, `nvml_peak_used_mib` governs physical peak!
                base["combined_physical_mib"] = 44840.0
                base["nvml_peak_used_mib"] = 25420.0
                return base

        pol = EnginePolicyConfig.from_preset("morphserve", num_gpus=2, model_family="synthetic")
        tiny_suite = WorkloadSuite(
            name="hygiene_smoke",
            trace_source="unit",
            base_qps=10.0,
            sessions=[
                InterleavedSession(
                    session_id="hyg_0",
                    quadrant="multi_image_vqa",
                    source_dataset="unit",
                    arrival_offset_s=0.0,
                    turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="hi", max_ar_tokens=4)],
                )
            ],
        )
        with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmp:
            fleet = InProcessWorkerFleet(pol, adapter_factory=lambda r, p: _TrackingAdapter(rank=r, policy=p))
            eng = MorphServeClusterEngine(
                policy=pol,
                model_dir=Path(tmp),
                bench_root=Path(tmp),
                tmp_root=Path(tmp) / "tmp_hyg",
                fleet=fleet,
            )
            try:
                eng.start()
                rep1 = eng.run_suite(tiny_suite, target_qps=10.0)
                assert len(reset_calls) >= 2  # Called across all ranks BEFORE suite 0!
                assert rep1.peak_nvml_used_mib == 25420.0
                assert rep1.peak_gpu_reserved_mib == 25420.0  # Eliminates 44.8 GiB double-counting!
                assert rep1.survival_rate_pct == 100.0
                assert rep1.all_session_e2e_p95_ms is not None
            finally:
                eng.shutdown()

        # 2. Improper all-session CDF places OOM sessions at +inf (`None` when survival < 95%, shifted when >= 95%)
        L = get_core_lib()
        pol_json = encode_json(serialize_policy_config(EnginePolicyConfig.from_preset("monolithic_dp", num_gpus=8)))
        # Case A: 90/100 survive (`survival_rate_pct == 90.0% < 95%`) -> `all_session_e2e_p95_ms` is `None` (`+inf`), P90 is max survivor!
        outcomes_90 = [
            {
                "status": "ok" if i < 90 else "oom",
                "e2e_ms": float((i + 1) * 1000.0),
                "ar_ttfts_ms": [200.0],
                "ar_tpots_ms": [50.0],
                "dit_turns_ms": [float((i + 1) * 500.0)] if i < 90 else [],
                "ar_slo_met": True,
                "dit_slo_met": True,
                "joint_slo_met": (i < 90),
            }
            for i in range(100)
        ]
        agg_90 = call_json(
            L,
            L.ms_ctrl_aggregate_report_json,
            pol_json,
            b"cdf_test_90",
            ctypes.c_double(0.4),
            ctypes.c_double(250.0),
            ctypes.c_double(0.2),
            encode_json(outcomes_90),
            encode_json([{"nvml_peak_used_mib": 39810.0, "pytorch_reserved_mib": 39810.0, "oom_count": 10}]),
        )
        assert agg_90["survival_rate_pct"] == 90.0
        assert agg_90["all_session_e2e_p95_ms"] is None
        assert agg_90["all_session_dit_p95_ms"] is None
        assert abs(float(agg_90["all_session_e2e_p90_ms"]) - 90000.0) < 1e-3

        # 3. `@80%` and `@90%` SLO linear crossing interpolation on `SLO Attainment (%) vs. QPS`
        sweep_rows = [
            {"policy_name": "morphserve", "offered_qps": 0.10, "joint_slo_attainment_pct": 95.0, "steady_state_attainment_pct": 95.0, "total_sessions": 48, "completed_sessions": 48, "oom_sessions": 0, "peak_nvml_used_mib": 34200.0},
            {"policy_name": "morphserve", "offered_qps": 0.20, "joint_slo_attainment_pct": 85.0, "steady_state_attainment_pct": 85.0, "total_sessions": 48, "completed_sessions": 48, "oom_sessions": 0, "peak_nvml_used_mib": 35100.0},
            {"policy_name": "morphserve", "offered_qps": 0.40, "joint_slo_attainment_pct": 65.0, "steady_state_attainment_pct": 65.0, "total_sessions": 96, "completed_sessions": 96, "oom_sessions": 0, "peak_nvml_used_mib": 36400.0},
            {"policy_name": "twostage_shm", "offered_qps": 0.10, "joint_slo_attainment_pct": 62.5, "steady_state_attainment_pct": 58.0, "total_sessions": 48, "completed_sessions": 42, "oom_sessions": 6, "peak_nvml_used_mib": 39800.0},
        ]
        s_qps = _compute_sustainable_qps_summary(sweep_rows)
        assert abs(s_qps["morphserve"]["max_sustainable_qps_90pct"] - 0.15) < 1e-4
        assert abs(s_qps["morphserve"]["max_sustainable_qps_80pct"] - 0.25) < 1e-4
        assert s_qps["twostage_shm"]["max_sustainable_qps_80pct"] == 0.0
        assert s_qps["twostage_shm"]["max_sustainable_qps_90pct"] == 0.0

    def test_cross_node_clock_skew_immunity_pager_fastpath_and_cascade_default(self) -> None:
        """Verify D6 clock-skew immunity, LayerExpertPager all-resident fast-path, and Cascade-CFG E2E default."""
        from morphserve.engine.worker_runtime import WorkerRuntime
        from morphserve.models.base import SyntheticCPTGAdapter
        from morphserve.vmm import ExpertPartitionSpec, LayerExpertPager, LayerPackPlan, NumaMemfdMirror, ParamSlice

        # 1. WorkerRuntime stamps local `_worker_recv_s`, completely ignoring a skewed cross-node `enqueue_time_s`!
        pol = EnginePolicyConfig.from_preset("morphserve", num_gpus=1)
        ad = SyntheticCPTGAdapter(rank=0, policy=pol)
        ad.attach_expert_mirror(None, populate=True)
        ad.bind_runtime_resources()
        acks: list[dict] = []
        rt = WorkerRuntime(rank=0, policy=pol, adapter=ad, peer_mesh=None, ack_sink=acks.append)
        try:
            skewed_remote_ts = time.perf_counter() - 2025.5  # Simulate Node 0 vs Node 1 boot offset!
            rt.enqueue_command(
                {
                    "op": "EXEC_AR",
                    "cmd_id": "skew_1",
                    "ctx_id": "ctx_skew_1",
                    "prompt": "hello",
                    "max_ar_tokens": 4,
                    "enqueue_time_s": skewed_remote_ts,
                }
            )
            cmd = rt._pop_next_main_cmd(timeout=0.1)
            assert cmd is not None and "_worker_recv_s" in cmd
            rt._execute_ar_cmds_and_ack([cmd])
            assert len(acks) == 1 and acks[0]["status"] == "ok"
            # Must be ~12ms (pure worker-local), NOT ~2,025,500ms!
            assert float(acks[0]["ttft_ms"]) < 1000.0, f"cross-node clock skew leaked into ttft_ms={acks[0]['ttft_ms']}"
        finally:
            ad.close()

        # 2. LayerExpertPager all-resident fast-path skips redundant C++ wait_expert_layer calls on Steps 1..N
        sim_vmm = SimulatedTriPoolBackend(
            device=0, super_pool_mib=128, fine_kv_pool_mib=32, super_handle_mib=16, allow_overflow=False
        )
        plans = [
            LayerPackPlan(
                layer_idx=i,
                used_bytes=4096,
                params=[ParamSlice(name=f"l{i}", offset=0, nbytes=4096, shape=(4096,), dtype_str="uint8")],
            )
            for i in range(4)
        ]
        spec = ExpertPartitionSpec(
            name="unit_fastpath",
            plans=plans,
            layer_stride_bytes=16 * (1024**2),
            hook_modules=[],
            vae_module=None,
            install_module_hooks=False,
        )
        pager = LayerExpertPager(spec)
        mirror = NumaMemfdMirror.create(nbytes=pager.total_mirror_bytes, numa_node=0, name="unit_fp_mirror")
        try:
            pager.offload_to_shared_mirror(mirror, populate=False)
            pager.bind_vmm(sim_vmm, initial_alpha=0.5)
            pager.set_wait_gate(lambda: True)
            with pager.generation_mode(prefetch_all=True):
                # Step 0: waits all 4 layers (triggers C++ wait_expert_layer)
                for lidx in range(4):
                    pager.wait_layer_if_needed(lidx, "forward")
                calls_step0 = int(sim_vmm.stats_dict()["expert_wait_calls"])
                assert calls_step0 == 4
                assert pager._all_resident_synced is True
                # Step 1: fast-path skips C++ wait_expert_layer completely while preserving layer_hook_calls!
                for lidx in range(4):
                    pager.wait_layer_if_needed(lidx, "forward")
                assert int(sim_vmm.stats_dict()["expert_wait_calls"]) == calls_step0
                assert pager.layer_hook_calls == 8
        finally:
            pager.close()
            sim_vmm.shutdown()
            mirror.close()

        # 3. Verify `BagelCPTGAdapter` enables `allow_cascade_split=True` for E2E FlashPrefix-CFG
        bagel_src = (REPO_ROOT / "python/morphserve/models/bagel_adapter.py").read_text(encoding="utf-8")
        assert "enable_shared_prefix_cfg(min_shared_tokens=32, allow_cascade_split=True)" in bagel_src


if __name__ == "__main__":
    unittest.main()
