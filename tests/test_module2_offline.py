"""Offline unit tests for Module 2 (Stateful Topology-Aware DualPathMorphRouter & NumaMemfdMirror)."""

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
    WorkerState,
)
from morphserve.vmm import NumaMemfdMirror  # noqa: E402


MIB = 1024**2
GIB = 1024**3


class TestModule2Offline(unittest.TestCase):
    def test_topology_half_node_contention_matches_job0c(self) -> None:
        topo = ClusterTopologyProfile.a100_pcie_8gpu_default()
        self.assertTrue(topo.same_half(0, 1))
        self.assertFalse(topo.same_half(0, 4))
        self.assertEqual(topo.numa_node_of(2), 0)
        self.assertEqual(topo.numa_node_of(6), 1)
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

        bridge_req = CPTGRequestSpec(
            req_id="internvl-u-turn4",
            src_gpu=4,
            architecture="bridge_mllm_dit",
            kv_bytes=3 * GIB,
            bridge_bytes=24 * MIB,
            dit_step1_ms=320.0,
            interleaved_roundtrip_gamma=2,
        )
        d_bridge = router.decide(bridge_req, workers)
        self.assertEqual(d_bridge.path, "state_ship")
        self.assertEqual(d_bridge.dst_gpu, 5)

    def test_stateful_router_atomic_reservations_and_placement(self) -> None:
        topo = ClusterTopologyProfile.a100_pcie_8gpu_default()
        router = DualPathMorphRouter(topo, pacer=HalfNodeH2DPacer(topo, max_concurrent_per_half=1))
        router.init_workers(
            num_gpus=4,
            role_of=lambda g: "gen" if g in (1, 3) else "hybrid",
            initial_alphas={0: 0.5, 1: 1.0, 2: 0.5, 3: 1.0},
            kv_free_bytes=16 * GIB,
        )
        # Initial placement avoids busy rank 0
        chosen = router.select_initial_gpu(
            num_gpus=4,
            routing_mode="dual_path",
            preemption_mode="layer_chunk",
            static_und_ranks=(),
            modality_plan="ar_only",
            gpu_busy={0: True, 1: False, 2: False, 3: False},
        )
        self.assertNotEqual(chosen, 0)

        # Path-A reservation atomically reserves target rank queue_wait_ms BEFORE async NCCL ship finishes
        req_a = CPTGRequestSpec(
            req_id="atomic-a",
            src_gpu=0,
            architecture="tight_mot",
            kv_bytes=4 * MIB,
            dit_step1_ms=280.0,
            dit_total_steps=20,
        )
        dec_a, mode_a = router.plan_and_reserve_transition(
            routing_mode="dual_path",
            static_und_ranks=(),
            static_gen_ranks=(),
            req=req_a,
            est_dit_ms=4200.0,
            now_ms=100.0,
        )
        self.assertEqual((dec_a.path, dec_a.dst_gpu, mode_a), ("state_ship", 1, "nccl_dual"))
        self.assertGreaterEqual(router.worker_states[1].queue_wait_ms, 4200.0)

        # Path-B reservation: on a 2-GPU pair where GPU 1 just reserved 4200 ms above, OR on 1024px high-res,
        # the router steers away from busy GPU 1 into in-place morph on GPU 2 (1024px) and books HalfNodeH2DPacer!
        req_b = CPTGRequestSpec(
            req_id="atomic-b",
            src_gpu=2,
            architecture="tight_mot",
            kv_bytes=2048 * MIB,
            dit_step1_ms=1650.0,
            dit_total_steps=20,
            resolution=1024,
            interleaved_roundtrip_gamma=2,
        )
        dec_b, mode_b = router.plan_and_reserve_transition(
            routing_mode="dual_path",
            static_und_ranks=(),
            static_gen_ranks=(),
            req=req_b,
            est_dit_ms=4500.0,
            now_ms=100.0,
        )
        self.assertEqual((dec_b.path, dec_b.dst_gpu, mode_b), ("in_place_morph", 2, "none"))
        self.assertGreater(router.pacer.predicted_wait_ms(0, now_ms=100.0), 0.0)

        # Also verify a second short request on GPU 0 (when only GPUs {0,1} exist) sees GPU 1's 4200ms pre-ship
        # reservation and adapts immediately instead of colliding onto GPU 1!
        router2 = DualPathMorphRouter(topo, pacer=HalfNodeH2DPacer(topo, max_concurrent_per_half=1))
        router2.init_workers(
            num_gpus=2,
            role_of=lambda g: "gen" if g == 1 else "hybrid",
            initial_alphas={0: 0.5, 1: 1.0},
            kv_free_bytes=16 * GIB,
        )
        d1, _ = router2.plan_and_reserve_transition(
            routing_mode="dual_path", static_und_ranks=(), static_gen_ranks=(), req=req_a, est_dit_ms=4200.0, now_ms=0.0
        )
        d2, _ = router2.plan_and_reserve_transition(
            routing_mode="dual_path", static_und_ranks=(), static_gen_ranks=(), req=req_a, est_dit_ms=4200.0, now_ms=1.0
        )
        self.assertEqual((d1.path, d1.dst_gpu), ("state_ship", 1))
        self.assertEqual((d2.path, d2.dst_gpu), ("in_place_morph", 0))

        router.release_dit_reservation(1, 4200.0)
        self.assertEqual(router.worker_states[1].queue_wait_ms, 0.0)

    def test_half_node_pacer_balances_mixed_burst(self) -> None:
        topo = ClusterTopologyProfile.a100_pcie_8gpu_default()
        pacer = HalfNodeH2DPacer(topo, max_concurrent_per_half=1)
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
        self.assertEqual((decs[0].path, decs[0].dst_gpu), ("state_ship", 1))
        self.assertEqual(decs[2].path, "state_ship")
        self.assertEqual(decs[3].path, "in_place_morph")

    def test_shared_numa_memfd_proc_fd_shares_same_inode(self) -> None:
        payload = b"MORPHSERVE_SHARED_EXPERT_CHECK"
        m0 = NumaMemfdMirror.create(nbytes=4 * MIB, numa_node=0, name="unit_test_shared_fd", populate_bytes=payload)
        try:
            m1 = NumaMemfdMirror.attach_proc_fd(os.getpid(), m0.fd, nbytes=4 * MIB, numa_node=0)
            try:
                self.assertEqual(m0.inode, m1.inode)
                self.assertEqual(bytes(m1.mm[: len(payload)]), payload)
            finally:
                m1.close()
        finally:
            m0.close()

    def test_realtime_kv_free_bytes_sync_and_inflight_reservation(self) -> None:
        topo = ClusterTopologyProfile.a100_pcie_8gpu_default()
        router = DualPathMorphRouter(topo, pacer=HalfNodeH2DPacer(topo, max_concurrent_per_half=1))
        router.init_workers(
            num_gpus=4,
            role_of=lambda g: "gen" if g in (1, 3) else "hybrid",
            initial_alphas={0: 0.5, 1: 1.0, 2: 0.5, 3: 1.0},
            kv_free_bytes={0: 16 * GIB, 1: 600 * MIB, 2: 16 * GIB, 3: 16 * GIB},
        )
        # First 400 MiB Path-A request fits on GPU 1 (`600 MiB -> 200 MiB` reserved pre-unpack)
        req1 = CPTGRequestSpec(
            req_id="kv_r1",
            src_gpu=0,
            architecture="tight_mot",
            kv_bytes=400 * MIB,
            dit_step1_ms=280.0,
            dit_total_steps=20,
        )
        d1, mode1 = router.plan_and_reserve_transition(
            routing_mode="dual_path",
            static_und_ranks=(),
            static_gen_ranks=(),
            req=req1,
            est_dit_ms=50.0,
            now_ms=0.0,
        )
        self.assertEqual((d1.path, d1.dst_gpu, mode1), ("state_ship", 1, "nccl_dual"))
        self.assertEqual(router.worker_states[1].kv_free_bytes, 200 * MIB)

        # Unrelated ACK arriving on GPU 1 BEFORE `req1` unpacks must NOT erase the 400 MiB in-flight reservation!
        router.update_worker_telemetry(1, expert_alpha=1.0, kv_free_bytes=600 * MIB)
        self.assertEqual(router.worker_states[1].kv_free_bytes, 200 * MIB)

        # Second 400 MiB request sees `200 MiB < 400 MiB` on GPU 1 and steers to GPU 3 (never OOM-stampeding GPU 1!)
        req2 = CPTGRequestSpec(
            req_id="kv_r2",
            src_gpu=0,
            architecture="tight_mot",
            kv_bytes=400 * MIB,
            dit_step1_ms=280.0,
            dit_total_steps=20,
        )
        d2, mode2 = router.plan_and_reserve_transition(
            routing_mode="dual_path",
            static_und_ranks=(),
            static_gen_ranks=(),
            req=req2,
            est_dit_ms=50.0,
            now_ms=1.0,
        )
        self.assertEqual((d2.path, d2.dst_gpu, mode2), ("state_ship", 3, "nccl_dual"))

        # Once `req1` unpacks, worker reports physical `kv_free_bytes = 200 MiB`; releasing reservation avoids double-subtraction!
        router.release_kv_reservation(1, 400 * MIB, latest_kv_free_bytes=200 * MIB)
        self.assertEqual(router.worker_states[1].kv_free_bytes, 200 * MIB)

        # Initial placement avoids critically KV-exhausted rank 0 (`< 128 MiB`)
        router.update_worker_telemetry(0, expert_alpha=0.25, kv_free_bytes=32 * MIB)
        picked = router.select_initial_gpu(
            num_gpus=4,
            routing_mode="dual_path",
            preemption_mode="layer_chunk",
            static_und_ranks=(),
            modality_plan="ar_only",
            gpu_busy={0: False, 1: False, 2: False, 3: False},
        )
        self.assertNotEqual(picked, 0)

    def test_static_fission_cost_based_gen_selection_and_monotonic_telemetry_seq(self) -> None:
        from morphserve.engine import InMemoryByteBus, TcpByteMesh

        topo = ClusterTopologyProfile.a100_pcie_8gpu_default()
        router = DualPathMorphRouter(topo, pacer=HalfNodeH2DPacer(topo, max_concurrent_per_half=1))
        router.init_workers(
            num_gpus=8,
            role_of=lambda g: "und" if g in (0, 1, 4, 5) else "gen",
            initial_alphas={g: (0.0 if g in (0, 1, 4, 5) else 1.0) for g in range(8)},
            kv_free_bytes=16 * GIB,
        )
        # Intra-half `gen` GPUs (2, 3) have 21s queue backlog while Socket-1 `gen` GPU 6 is idle (`0ms`):
        # fair cost-based `select_static_gen_gpu` MUST pick idle GPU 6 (`~3.4ms` ship) rather than waiting 21s!
        router.worker_states[2].queue_wait_ms = 21_000.0
        router.worker_states[3].queue_wait_ms = 21_000.0
        router.worker_states[6].queue_wait_ms = 0.0
        router.worker_states[7].queue_wait_ms = 5_000.0
        chosen_gen = router.select_static_gen_gpu(
            routing_mode="static_nccl_fission",
            static_und_ranks=(0, 1, 4, 5),
            static_gen_ranks=(2, 3, 6, 7),
            src_und_gpu=0,
            required_kv_bytes=4 * MIB,
        )
        self.assertEqual(chosen_gen, 6)

        # Monotonic `telemetry_seq` guard prevents stale out-of-order coroutine ACKs from overwriting newer KV state
        router.update_worker_telemetry(6, expert_alpha=0.75, kv_free_bytes=8 * GIB, telemetry_seq=10)
        router.update_worker_telemetry(6, expert_alpha=1.00, kv_free_bytes=15 * GIB, telemetry_seq=9)  # stale!
        self.assertEqual(router.worker_states[6].kv_free_bytes, 8 * GIB)
        self.assertAlmostEqual(router.worker_states[6].expert_alpha, 0.75)

        # Verify `InMemoryPeerMesh` and `TcpByteMesh` `recv_u8(..., out_tensor=buf)` populate `buf` in-place
        bus = InMemoryByteBus(2)
        m0, m1 = bus.endpoint_for(0), bus.endpoint_for(1)
        m0.send_u8(b"INPLACE_MEM", dst_rank=1)
        buf_mem = m1.alloc_recv_u8(11)
        out_mem, _ = m1.recv_u8(11, src_rank=0, out_tensor=buf_mem)
        self.assertIs(out_mem, buf_mem)
        self.assertEqual(bytes(buf_mem), b"INPLACE_MEM")

        tcp_mesh = TcpByteMesh(2)
        try:
            t0, t1 = tcp_mesh.endpoint_for(0), tcp_mesh.endpoint_for(1)
            t0.send_u8(b"INPLACE_TCP", dst_rank=1)
            buf_tcp = t1.alloc_recv_u8(11)
            out_tcp, _ = t1.recv_u8(11, src_rank=0, out_tensor=buf_tcp)
            self.assertIs(out_tcp, buf_tcp)
            self.assertEqual(bytes(buf_tcp), b"INPLACE_TCP")
        finally:
            tcp_mesh.close()

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
            "src/morphserve/bench/validate_module2_dualpath.py",
        ):
            p = REPO_ROOT / rel
            res = subprocess.run([sys.executable, "-m", "py_compile", str(p)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, msg=f"py_compile failed on {rel}: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
