"""Offline unit tests for Module 2 (Topology-Aware Dual-Path Morphing Router)."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from morphserve.router import (  # noqa: E402
    CPTGRequestSpec,
    ClusterTopologyProfile,
    DualPathMorphRouter,
    HalfNodeH2DPacer,
    SharedNumaMemfdMirror,
    WorkerState,
)


MIB = 1024**2
GIB = 1024**3


class TestModule2Offline(unittest.TestCase):
    def test_topology_half_node_contention_matches_job0c(self) -> None:
        topo = ClusterTopologyProfile.a100_pcie_8gpu_default()
        self.assertTrue(topo.same_half(0, 1))
        self.assertFalse(topo.same_half(0, 4))
        self.assertEqual(topo.numa_node_of(2), 0)
        self.assertEqual(topo.numa_node_of(6), 1)
        # Job 0c: solo ~8.75 GB/s, 4-GPU same-half collapses to ~2.58 GB/s
        self.assertAlmostEqual(topo.effective_h2d_gbps(0, active_h2d_same_half=1), 8.75, places=2)
        self.assertLess(topo.effective_h2d_gbps(0, active_h2d_same_half=4), 2.70)
        self.assertGreater(topo.effective_p2p_gbps(0, 1), topo.effective_p2p_gbps(0, 4, concurrent_cross_half=4))

        job0c_path = REPO_ROOT / "results/phase0-job0c/topo-8gpu-report.json"
        if job0c_path.exists():
            parsed = ClusterTopologyProfile.from_job0c_report(job0c_path)
            self.assertEqual(parsed.num_gpus, 8)
            self.assertEqual(len(parsed.half_groups), 2)

    def test_dual_path_crossover_bagel_and_internvl_u(self) -> None:
        topo = ClusterTopologyProfile.a100_pcie_8gpu_default()
        router = DualPathMorphRouter(topo)
        workers = {
            0: WorkerState(gpu_id=0, role="und", expert_alpha=0.50, kv_free_bytes=16 * GIB),
            1: WorkerState(gpu_id=1, role="gen", expert_alpha=1.00, kv_free_bytes=16 * GIB),
            4: WorkerState(gpu_id=4, role="und", expert_alpha=0.50, kv_free_bytes=16 * GIB),
            5: WorkerState(gpu_id=5, role="gen", expert_alpha=1.00, kv_free_bytes=16 * GIB),
        }

        # Case 1: Turn-0 BAGEL short KV (~4 MiB) -> State-Ship to same-half Gen peer GPU 1 (~3.4 ms vs ~470 ms morph)
        short_req = CPTGRequestSpec(
            req_id="bagel-turn0",
            src_gpu=0,
            architecture="tight_mot",
            kv_bytes=4 * MIB,
            dit_step1_ms=287.0,
            interleaved_roundtrip_gamma=1,
        )
        d_short = router.decide(short_req, workers)
        self.assertEqual(d_short.path, "state_ship")
        self.assertEqual(d_short.dst_gpu, 1)

        # Case 2: Multi-turn BAGEL long interleaved KV (1.8 GiB, gamma=2, alpha=0.75 or 1024px alpha=0.5) -> In-Place Morph!
        workers_hot = dict(workers)
        workers_hot[0] = WorkerState(gpu_id=0, role="und", expert_alpha=0.75, kv_free_bytes=16 * GIB)
        long_req = CPTGRequestSpec(
            req_id="bagel-turn4",
            src_gpu=0,
            architecture="tight_mot",
            kv_bytes=1950 * MIB,
            dit_step1_ms=340.0,
            interleaved_roundtrip_gamma=2,
        )
        d_long = router.decide(long_req, workers_hot)
        self.assertEqual(d_long.path, "in_place_morph")
        self.assertEqual(d_long.dst_gpu, 0)

        # Case 3: 1024px high-res DiT even at alpha=0.50 overlaps Step-1 H2D (~1650ms 3-branch Step1) -> In-Place wins
        req_1024 = CPTGRequestSpec(
            req_id="bagel-1024",
            src_gpu=0,
            architecture="tight_mot",
            kv_bytes=1200 * MIB,
            dit_step1_ms=1650.0,
            resolution=1024,
            interleaved_roundtrip_gamma=2,
        )
        d_1024 = router.decide(req_1024, workers)
        self.assertEqual(d_1024.path, "in_place_morph")
        self.assertEqual(d_1024.dst_gpu, 0)

        # Case 4: Bridge MLLM+DiT (InternVL-U, 24 MiB compact bridge tensor) -> ALWAYS State-Ship to same-half GPU 5!
        bridge_req = CPTGRequestSpec(
            req_id="internvl-u-turn4",
            src_gpu=4,
            architecture="bridge_mllm_dit",
            kv_bytes=3 * GIB,  # ignored for bridge!
            bridge_bytes=24 * MIB,
            dit_step1_ms=320.0,
            interleaved_roundtrip_gamma=2,
        )
        d_bridge = router.decide(bridge_req, workers)
        self.assertEqual(d_bridge.path, "state_ship")
        self.assertEqual(d_bridge.dst_gpu, 5)

    def test_half_node_pacer_balances_mixed_burst(self) -> None:
        topo = ClusterTopologyProfile.a100_pcie_8gpu_default()
        pacer = HalfNodeH2DPacer(topo, max_concurrent_per_half=1)
        # Stack H2D work on half 0 -> predicted wait grows, while half 1 stays 0.0!
        w0 = pacer.reserve_slot(0, duration_ms=500.0, now_ms=0.0)
        w1 = pacer.reserve_slot(1, duration_ms=500.0, now_ms=0.0)
        w4 = pacer.reserve_slot(4, duration_ms=500.0, now_ms=0.0)
        self.assertEqual(w0, 0.0)
        self.assertEqual(w1, 500.0)
        self.assertEqual(w4, 0.0)

        router = DualPathMorphRouter(topo, pacer=HalfNodeH2DPacer(topo, max_concurrent_per_half=1))
        workers = {
            g: WorkerState(gpu_id=g, role="gen" if g in (1, 3, 5, 7) else "hybrid", expert_alpha=1.0 if g in (1, 3, 5, 7) else 0.5)
            for g in range(8)
        }
        burst = [
            CPTGRequestSpec(req_id="r0", src_gpu=0, architecture="tight_mot", kv_bytes=8 * MIB, dit_step1_ms=280.0, dit_total_steps=20),
            CPTGRequestSpec(req_id="r1", src_gpu=2, architecture="tight_mot", kv_bytes=8 * MIB, dit_step1_ms=280.0, dit_total_steps=20),
            CPTGRequestSpec(req_id="r2", src_gpu=4, architecture="bridge_mllm_dit", bridge_bytes=24 * MIB, dit_step1_ms=280.0, dit_total_steps=20),
            CPTGRequestSpec(req_id="r3", src_gpu=6, architecture="tight_mot", kv_bytes=2200 * MIB, dit_step1_ms=1650.0, dit_total_steps=10, interleaved_roundtrip_gamma=2),
        ]
        decs = router.route_batch(burst, workers)
        self.assertEqual(len(decs), 4)
        # r0 ships to same-half gen GPU 1, then r1 sees GPU 1 busy (~4.7s queue!) and adapts (ships to GPU 5 or morphs in-place)!
        self.assertEqual((decs[0].path, decs[0].dst_gpu), ("state_ship", 1))
        self.assertEqual(decs[2].path, "state_ship")
        self.assertEqual(decs[3].path, "in_place_morph")

    def test_shared_numa_memfd_proc_fd_shares_same_inode(self) -> None:
        payload = b"MORPHSERVE_SHARED_EXPERT_CHECK"
        m0 = SharedNumaMemfdMirror.create(nbytes=4 * MIB, numa_node=0, name="unit_test_shared_fd", populate_bytes=payload)
        try:
            m1 = SharedNumaMemfdMirror.attach_proc_fd(os.getpid(), m0.fd, nbytes=4 * MIB, numa_node=0)
            try:
                self.assertEqual(m0.inode, m1.inode)
                self.assertEqual(bytes(m1.mm[: len(payload)]), payload)
            finally:
                m1.close()
        finally:
            m0.close()

    def test_module2_syntax_and_imports(self) -> None:
        script = REPO_ROOT / "deploy/kairos/scripts/07_phase1_module2_validate.sh"
        if script.exists():
            proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        for rel in (
            "src/morphserve/router/__init__.py",
            "src/morphserve/router/topology.py",
            "src/morphserve/router/cost_router.py",
            "src/morphserve/router/nccl_state_shipper.py",
        ):
            p = REPO_ROOT / rel
            res = subprocess.run([sys.executable, "-m", "py_compile", str(p)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, msg=f"py_compile failed on {rel}: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
