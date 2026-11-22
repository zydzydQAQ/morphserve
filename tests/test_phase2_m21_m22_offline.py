"""Offline unit & integration tests for Phase-2 M2.1 (Interleaved-Bench) and M2.2 (MorphServeClusterEngine)."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from morphserve.bench.interleaved_bench import (  # noqa: E402
    InterleavedSession,
    InterleavedTurn,
    WorkloadSuite,
    assign_dual_trace_arrivals,
    compose_weighted_suite,
    load_burstgpt_interarrivals,
    load_gentd26_interarrivals,
    scale_arrivals_to_qps,
)
from morphserve.engine import (  # noqa: E402
    EnginePolicyConfig,
    InProcessWorkerFleet,
    MorphServeClusterEngine,
    VALID_POLICIES,
)
from morphserve.models import SyntheticCPTGAdapter  # noqa: E402


def _scratch_dir() -> str | None:
    d_root = Path("/mnt/d/Transfer/MORPHSERVE")
    if Path("/mnt/d/Transfer").is_dir():
        d_root.mkdir(parents=True, exist_ok=True)
        return str(d_root)
    return None


def _build_mini_interleaved_suite() -> WorkloadSuite:
    return WorkloadSuite(
        name="inproc_mini_4",
        trace_source="unit_test",
        base_qps=4.0,
        sessions=[
            InterleavedSession(
                session_id="s_vcot",
                quadrant="visual_cot",
                source_dataset="unit",
                arrival_offset_s=0.00,
                turns=[
                    InterleavedTurn(turn_idx=0, modality_plan="cot_then_dit", prompt="sketch layout", max_ar_tokens=6, dit_steps=3),
                    InterleavedTurn(turn_idx=1, modality_plan="ar_only", prompt="verify objects", max_ar_tokens=6),
                ],
            ),
            InterleavedSession(
                session_id="s_edit",
                quadrant="multiturn_edit",
                source_dataset="unit",
                arrival_offset_s=0.01,
                turns=[
                    InterleavedTurn(turn_idx=0, modality_plan="dit_t2i", prompt="cabin by lake", dit_steps=3),
                    InterleavedTurn(turn_idx=1, modality_plan="dit_edit", prompt="add autumn leaves", dit_steps=3),
                    InterleavedTurn(turn_idx=2, modality_plan="dit_edit", prompt="make it twilight", dit_steps=3),
                    InterleavedTurn(turn_idx=3, modality_plan="dit_edit", prompt="add snow peaks", dit_steps=3),
                ],
            ),
            InterleavedSession(
                session_id="s_vqa",
                quadrant="multi_image_vqa",
                source_dataset="unit",
                arrival_offset_s=0.02,
                turns=[
                    InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="compare scenes", max_ar_tokens=6),
                ],
            ),
            InterleavedSession(
                session_id="s_hires",
                quadrant="highres_t2i",
                source_dataset="unit",
                arrival_offset_s=0.03,
                turns=[
                    InterleavedTurn(
                        turn_idx=0, modality_plan="dit_t2i", prompt="mountain vista", height=1024, width=1024, dit_steps=3
                    ),
                ],
            ),
        ],
    )


class TestPhase2M21M22Offline(unittest.TestCase):
    def test_trace_parsers_and_qps_scaling(self) -> None:
        with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmp:
            td = Path(tmp)
            burst_csv = td / "burst.csv"
            burst_csv.write_text(
                "Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n"
                "5,ChatGPT,100,20,120,Conversation log\n"
                "15,ChatGPT,100,20,120,Conversation log\n"
                "35,GPT-4,100,20,120,Conversation log\n",
                encoding="utf-8",
            )
            gentd_csv = td / "gentd.csv"
            gentd_csv.write_text(
                "gmt_create,predict_type,predict_status,exec_time_seconds,groupId\n"
                "2024-11-15 16:57:50,TXT_2_IMG,SUCCEED,32.0,G0\n"
                "2024-11-15 16:58:00,TXT_2_IMG,FAILED,1.0,G0\n"
                "2024-11-15 16:58:10,IMG_2_IMG,SUCCEED,40.0,G0\n"
                "2024-11-15 16:58:50,TXT_2_IMG,SUCCEED,20.0,G0\n",
                encoding="utf-8",
            )
            bd = load_burstgpt_interarrivals(burst_csv)
            gd = load_gentd26_interarrivals(gentd_csv)
            self.assertAlmostEqual(sum(bd) / len(bd), 1.0, places=5)
            self.assertAlmostEqual(sum(gd) / len(gd), 1.0, places=5)

            offs = scale_arrivals_to_qps(bd, target_qps=2.0, count=5)
            self.assertEqual(len(offs), 5)
            self.assertEqual(offs[0], 0.0)
            self.assertTrue(all(offs[i] >= offs[i - 1] for i in range(1, len(offs))))

    def test_weighted_suite_composition_and_roundtrip(self) -> None:
        pools = {
            q: [
                InterleavedSession(
                    session_id=f"{q}_0",
                    quadrant=q,
                    source_dataset="test",
                    turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only" if q == "multi_image_vqa" else "dit_t2i", prompt="p")],
                )
            ]
            for q in ("visual_cot", "multiturn_edit", "multi_image_vqa", "highres_t2i")
        }
        suite = compose_weighted_suite(
            "mix_test",
            pools,
            {"visual_cot": 6, "multiturn_edit": 6, "multi_image_vqa": 5, "highres_t2i": 3},
            [0.5, 1.5],
            [0.8, 1.2],
            target_qps=1.0,
        )
        self.assertEqual(len(suite.sessions), 20)
        raw = json.dumps(suite.to_dict())
        loaded = WorkloadSuite.from_dict(json.loads(raw))
        self.assertEqual(len(loaded.sessions), 20)
        ts = [s.arrival_offset_s for s in loaded.sessions]
        self.assertEqual(ts, sorted(ts))

        assign_dual_trace_arrivals(loaded.sessions, [1.0], [1.0], target_qps=2.0)
        self.assertEqual(loaded.sessions[0].arrival_offset_s, 0.0)

    def test_engine_policy_presets_matrix(self) -> None:
        self.assertEqual(len(VALID_POLICIES), 6)
        ms = EnginePolicyConfig.from_preset("morphserve", num_gpus=8)
        self.assertTrue(ms.use_tripool_vmm and ms.use_expert_pager and ms.use_scoped_vae)
        self.assertTrue(ms.use_shared_numa_mirror and ms.use_h2d_pacer and ms.use_module3_cow_loopback)
        self.assertEqual(ms.preemption_mode, "layer_chunk")
        self.assertEqual(ms.routing_mode, "dual_path")
        self.assertEqual(ms.initial_alphas, {g: 1.0 for g in range(8)})

        prism = EnginePolicyConfig.from_preset("prism_tetriserve_coloc", num_gpus=8)
        self.assertTrue(prism.use_tripool_vmm and prism.kv_vmm_only)
        self.assertFalse(prism.use_expert_pager)
        self.assertEqual(prism.preemption_mode, "step_preemptive")
        self.assertLessEqual(prism.super_pool_mib + prism.fine_kv_pool_mib, 4096)

        fission = EnginePolicyConfig.from_preset("static_nccl_fission", num_gpus=8)
        self.assertEqual(fission.static_und_ranks, (0, 1, 4, 5))
        self.assertEqual(fission.static_gen_ranks, (2, 3, 6, 7))

        shm = EnginePolicyConfig.from_preset("twostage_shm", num_gpus=8)
        self.assertEqual(shm.static_und_ranks, (0, 2, 4, 6))
        self.assertEqual(shm.static_gen_ranks, (1, 3, 5, 7))

    def test_inprocess_cluster_engine_all_6_policies_and_internvlu(self) -> None:
        suite = _build_mini_interleaved_suite()
        with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmp:
            td = Path(tmp)
            for preset in VALID_POLICIES:
                base_pol = EnginePolicyConfig.from_preset(preset, num_gpus=4)
                pol = replace(base_pol, inter_turn_think_s_range=(0.0, 0.0))
                fleet = InProcessWorkerFleet(pol)
                engine = MorphServeClusterEngine(
                    policy=pol,
                    model_dir=td,
                    bench_root=td,
                    tmp_root=td / preset,
                    fleet=fleet,
                )
                try:
                    engine.start()
                    rep = engine.run_suite(suite)
                finally:
                    engine.shutdown()
                self.assertEqual(rep.status, "ok", msg=f"preset={preset} failed: {rep.to_dict()}")
                self.assertEqual(rep.completed_sessions, 4)
                self.assertEqual(rep.error_sessions, 0)
                self.assertEqual(rep.oom_sessions, 0)
                self.assertIn("2.0x", rep.slo_curve_by_k)
                if preset == "morphserve":
                    self.assertGreater(rep.path_a_ships + rep.path_b_morphs, 0)

            # Also verify decoupled Bridge MLLM+DiT (`model_family="internvl_u"`) over unified MorphServeClusterEngine
            ivl_pol = replace(
                EnginePolicyConfig.from_preset("morphserve", num_gpus=4, model_family="internvl_u"),
                inter_turn_think_s_range=(0.0, 0.0),
            )
            ivl_engine = MorphServeClusterEngine(
                policy=ivl_pol,
                model_dir=td,
                bench_root=td,
                tmp_root=td / "internvl_u",
                fleet=InProcessWorkerFleet(ivl_pol),
            )
            try:
                ivl_engine.start()
                ivl_rep = ivl_engine.run_suite(suite)
            finally:
                ivl_engine.shutdown()
            self.assertEqual(ivl_rep.status, "ok")
            self.assertEqual(ivl_rep.completed_sessions, 4)

    def test_inprocess_mid_dit_preemption_and_bounded_oom_retry(self) -> None:
        with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmp:
            td = Path(tmp)
            # 1. Single-worker mid-DiT preemption (`dit_step_sleep_s=0.04` ensures AR arrives mid-DiT)
            pol_1g = replace(
                EnginePolicyConfig.from_preset("morphserve", num_gpus=1),
                inter_turn_think_s_range=(0.0, 0.0),
            )
            preempt_suite = WorkloadSuite(
                name="preempt_check",
                trace_source="unit",
                base_qps=10.0,
                sessions=[
                    InterleavedSession(
                        session_id="dit_holder",
                        quadrant="highres_t2i",
                        source_dataset="unit",
                        arrival_offset_s=0.0,
                        turns=[InterleavedTurn(turn_idx=0, modality_plan="dit_t2i", prompt="slow dit", dit_steps=4)],
                    ),
                    InterleavedSession(
                        session_id="ar_urgent",
                        quadrant="multi_image_vqa",
                        source_dataset="unit",
                        arrival_offset_s=0.02,
                        turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="urgent ar", max_ar_tokens=4)],
                    ),
                ],
            )
            fleet_1g = InProcessWorkerFleet(
                pol_1g,
                adapter_factory=lambda r, p: SyntheticCPTGAdapter(r, p, dit_step_sleep_s=0.04),
            )
            eng_1g = MorphServeClusterEngine(pol_1g, td, td, tmp_root=td / "preempt", fleet=fleet_1g)
            try:
                eng_1g.start()
                rep_p = eng_1g.run_suite(preempt_suite)
            finally:
                eng_1g.shutdown()
            self.assertEqual(rep_p.completed_sessions, 2)
            self.assertGreaterEqual(rep_p.preempted_ar_count, 1)

            # 2. Transient rank-0 OOM recovered cleanly on peer rank via bounded 1-retry (`max_oom_retries=1`)
            pol_2g = replace(
                EnginePolicyConfig.from_preset("morphserve", num_gpus=2),
                inter_turn_think_s_range=(0.0, 0.0),
                max_oom_retries=1,
            )
            oom_suite = WorkloadSuite(
                name="oom_retry_check",
                trace_source="unit",
                base_qps=10.0,
                sessions=[
                    InterleavedSession(
                        session_id="s_retry",
                        quadrant="multi_image_vqa",
                        source_dataset="unit",
                        arrival_offset_s=0.0,
                        turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="recover me", max_ar_tokens=4)],
                    )
                ],
            )
            fleet_2g = InProcessWorkerFleet(
                pol_2g,
                adapter_factory=lambda r, p: SyntheticCPTGAdapter(r, p, oom_ranks=(0,)),
            )
            eng_2g = MorphServeClusterEngine(pol_2g, td, td, tmp_root=td / "oom_retry", fleet=fleet_2g)
            try:
                eng_2g.start()
                rep_o = eng_2g.run_suite(oom_suite)
            finally:
                eng_2g.shutdown()
            self.assertEqual(rep_o.completed_sessions, 1)
            self.assertEqual(rep_o.oom_sessions, 0)
            self.assertGreaterEqual(rep_o.oom_retry_count, 1)

    def test_phase2_scripts_syntax_and_py_compile(self) -> None:
        for script_rel in (
            "deploy/kairos/scripts/08_phase2_m21_bench_and_m3gate.sh",
            "deploy/kairos/scripts/09_phase2_m22_engine_smoke_8gpu.sh",
        ):
            script = REPO_ROOT / script_rel
            proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=f"bash -n failed on {script_rel}: {proc.stderr}")

        for py_rel in (
            "src/morphserve/telemetry.py",
            "src/morphserve/tools/stage_bench_sources.py",
            "src/morphserve/bench/exactness.py",
            "src/morphserve/bench/interleaved_bench.py",
            "src/morphserve/bench/validate_phase2_engine_smoke.py",
            "src/morphserve/engine/__init__.py",
            "src/morphserve/engine/config.py",
            "src/morphserve/engine/worker_runtime.py",
            "src/morphserve/engine/worker_proc.py",
            "src/morphserve/engine/cluster_engine.py",
        ):
            p = REPO_ROOT / py_rel
            res = subprocess.run([sys.executable, "-m", "py_compile", str(p)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, msg=f"py_compile failed on {py_rel}: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
