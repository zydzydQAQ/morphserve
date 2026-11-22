"""Offline unit tests for M2.3/M2.4 (stratifier, downstream scorers, InternVL-U quota) & M2.5 (zero-bias simulator)."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
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
        assert cfgs["no_m4_layer_chunk"].preemption_mode == "non_preemptive"
        assert not cfgs["no_m3_cow_loopback"].use_module3_cow_loopback
        assert cfgs["no_m2_dual_path"].routing_mode == "sticky_least_loaded"
        assert not cfgs["no_m1_tripool_vmm"].use_tripool_vmm

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
        assert ov["super_pool_mib"] + ov["fine_kv_pool_mib"] == 4832
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
        # Verify end-to-end integration with `InternVLUCPTGAdapter.compute_internvlu_memory_plan`
        plan = compute_internvlu_memory_plan(
            total_dev_mib=40_960.0,
            free_dev_mib=35_000.0,
            base_pytorch_alloc_mib=5_474.0,
            mirror_bytes=4_480 * (1024**2),
            policy_super_pool_mib=pol.super_pool_mib,
            policy_fine_kv_pool_mib=pol.fine_kv_pool_mib,
            super_handle_mib=pol.super_handle_mib,
            allow_vmm_overflow=pol.allow_vmm_overflow,
            gpu_memory_fraction=pol.gpu_memory_fraction,
        )
        assert plan["quota_constrained"] is True
        assert plan["allow_overflow"] is False
        assert plan["vmm_physical_pool_mib"] == 4832
        assert plan["pytorch_cap_mib"] + plan["vmm_physical_pool_mib"] <= 12_288.0

        # Explicit decomposed (`pytorch_peak_reserved_mib` + `vmm_physical_pool_mib`) audit
        ok_audit = verify_internvlu_memory_quota(
            {"pytorch_peak_reserved_mib": 6_900.0, "vmm_physical_pool_mib": 4_832.0, "vmm_overflow_creates": 0},
            pol,
            constrained_gib=12.0,
        )
        assert ok_audit["within_quota"] is True
        assert ok_audit["combined_physical_mib"] == 11_732.0

        # Combined `ClusterRunReport.peak_gpu_reserved_mib` audit (no double-counting of VMM pool!)
        rep_audit = verify_internvlu_memory_quota(
            {"peak_gpu_reserved_mib": 11_732.0, "vmm_overflow_creates": 0},
            pol,
            constrained_gib=12.0,
        )
        assert rep_audit["within_quota"] is True
        assert rep_audit["pytorch_peak_reserved_mib"] == 6_900.0
        assert rep_audit["combined_physical_mib"] == 11_732.0

        over_audit = verify_internvlu_memory_quota(
            {"pytorch_peak_reserved_mib": 7_800.0, "vmm_physical_pool_mib": 4_832.0, "vmm_overflow_creates": 0},
            pol,
            constrained_gib=12.0,
        )
        assert over_audit["within_quota"] is False

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


if __name__ == "__main__":
    unittest.main()
