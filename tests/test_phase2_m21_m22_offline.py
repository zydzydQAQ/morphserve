"""Offline unit tests for Phase-2 M2.1 (Interleaved-Bench) and M2.2 (MorphServeClusterEngine presets)."""

from __future__ import annotations

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
from morphserve.engine import EnginePolicyConfig, VALID_POLICIES  # noqa: E402


class TestPhase2M21M22Offline(unittest.TestCase):
    def test_trace_parsers_and_qps_scaling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
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
        # Verify arrival times are non-decreasing
        ts = [s.arrival_offset_s for s in loaded.sessions]
        self.assertEqual(ts, sorted(ts))

        # Verify rescaling helper updates arrivals cleanly
        assign_dual_trace_arrivals(loaded.sessions, [1.0], [1.0], target_qps=2.0)
        self.assertEqual(loaded.sessions[0].arrival_offset_s, 0.0)

    def test_engine_policy_presets_matrix(self) -> None:
        self.assertEqual(len(VALID_POLICIES), 6)
        ms = EnginePolicyConfig.from_preset("morphserve", num_gpus=8)
        self.assertTrue(ms.use_tripool_vmm and ms.use_expert_pager and ms.use_scoped_vae)
        self.assertTrue(ms.use_shared_numa_mirror and ms.use_h2d_pacer and ms.use_module3_cow_loopback)
        self.assertEqual(ms.preemption_mode, "layer_chunk")
        self.assertEqual(ms.routing_mode, "dual_path")

        prism = EnginePolicyConfig.from_preset("prism_tetriserve_coloc", num_gpus=8)
        self.assertTrue(prism.use_tripool_vmm and prism.kv_vmm_only)
        self.assertFalse(prism.use_expert_pager)
        self.assertEqual(prism.preemption_mode, "step_preemptive")
        # Ensure KV-only VMM leaves room for full static 28.15 GiB weights on 40GB A100
        self.assertLessEqual(prism.super_pool_mib + prism.fine_kv_pool_mib, 4096)

        fission = EnginePolicyConfig.from_preset("static_nccl_fission", num_gpus=8)
        self.assertEqual(fission.static_und_ranks, (0, 1, 4, 5))
        self.assertEqual(fission.static_gen_ranks, (2, 3, 6, 7))

        shm = EnginePolicyConfig.from_preset("twostage_shm", num_gpus=8)
        self.assertEqual(shm.static_und_ranks, (0, 2, 4, 6))
        self.assertEqual(shm.static_gen_ranks, (1, 3, 5, 7))

    def test_phase2_scripts_syntax_and_py_compile(self) -> None:
        for script_rel in (
            "deploy/kairos/scripts/08_phase2_m21_bench_and_m3gate.sh",
            "deploy/kairos/scripts/09_phase2_m22_engine_smoke_8gpu.sh",
        ):
            script = REPO_ROOT / script_rel
            proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=f"bash -n failed on {script_rel}: {proc.stderr}")

        for py_rel in (
            "src/morphserve/tools/stage_bench_sources.py",
            "src/morphserve/bench/interleaved_bench.py",
            "src/morphserve/bench/validate_phase2_engine_smoke.py",
            "src/morphserve/engine/__init__.py",
            "src/morphserve/engine/config.py",
            "src/morphserve/engine/worker_proc.py",
            "src/morphserve/engine/cluster_engine.py",
        ):
            p = REPO_ROOT / py_rel
            res = subprocess.run([sys.executable, "-m", "py_compile", str(p)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, msg=f"py_compile failed on {py_rel}: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
