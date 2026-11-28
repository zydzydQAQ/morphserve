"""TDD tests for Resonator-inspired 3D Performance Atlas, FlashGen-inspired Memory+Slack GPU Queue,
LRU Think-Time KV Swap vs. Recompute ablation, and Mean/P99/HBM-utilization report metrics."""

from __future__ import annotations

import asyncio
import time

from morphserve._core import call_json, encode_json, get_core_lib
from morphserve.engine.cluster_engine import ClusterRunReport, MemorySlackGpuLock
from morphserve.engine.config import EnginePolicyConfig
from morphserve.router import (
    CPTGRequestSpec,
    ClusterTopologyProfile,
    DualPathMorphRouter,
    WorkerState,
)
from morphserve.sim.cluster_simulator import (
    HardwareInterconnectSpec,
    MeasuredPhysicsProfile,
    WorkloadScaleSpec,
    simulate_cluster_point,
)

MIB = 1024**2
GIB = 1024**3


def test_policy_preset_defaults_and_roundtrip_new_modes():
    ms = EnginePolicyConfig.from_preset("morphserve", num_gpus=8)
    mono = EnginePolicyConfig.from_preset("monolithic_dp", num_gpus=8)
    prism = EnginePolicyConfig.from_preset("prism_tetriserve_coloc", num_gpus=8)
    assert ms.cost_model_mode == "pipeline_analytical"
    assert ms.gpu_queue_mode == "memory_slack"
    assert ms.gpu_queue_max_aging_ms == 45000.0
    assert ms.gpu_queue_max_bypasses == 5
    assert ms.kv_eviction_mode == "swap"
    assert mono.cost_model_mode == "legacy_analytical"
    assert mono.gpu_queue_mode == "fcfs"
    assert prism.gpu_queue_mode == "memory_slack"
    assert prism.gpu_queue_max_aging_ms == 45000.0
    assert prism.gpu_queue_max_bypasses == 5
    assert prism.super_pool_mib == 0 and prism.fine_kv_pool_mib == 64

    ov = EnginePolicyConfig.from_preset(
        "morphserve",
        num_gpus=8,
        cost_model_mode="pipeline_analytical",
        gpu_queue_mode="fcfs",
        gpu_queue_max_aging_ms=4500.0,
        gpu_queue_max_bypasses=2,
        kv_eviction_mode="recompute",
    )
    rt = EnginePolicyConfig.from_json(ov.to_json())
    assert rt.cost_model_mode == "pipeline_analytical"
    assert rt.gpu_queue_mode == "fcfs"
    assert rt.gpu_queue_max_aging_ms == 4500.0
    assert rt.gpu_queue_max_bypasses == 2
    assert rt.kv_eviction_mode == "recompute"


def test_router_3d_atlas_vs_legacy_and_colocated_ar_stretch():
    topo = ClusterTopologyProfile.a100_pcie_8gpu_default()
    r_atlas = DualPathMorphRouter(topo, cost_model_mode="atlas_3d")
    r_legacy = DualPathMorphRouter(topo, cost_model_mode="legacy_analytical")

    workers = {
        0: WorkerState(gpu_id=0, expert_alpha=0.0, kv_free_bytes=16 * GIB),
        1: WorkerState(gpu_id=1, expert_alpha=1.0, kv_free_bytes=16 * GIB),
    }
    req_1024 = CPTGRequestSpec(
        req_id="r_1024_cold",
        src_gpu=0,
        architecture="tight_mot",
        model_family="bagel",
        kv_bytes=1600 * MIB,
        resolution=1024,
        interleaved_roundtrip_gamma=2,
    )
    d_atlas = r_atlas.decide(req_1024, workers=workers)
    d_legacy = r_legacy.decide(req_1024, workers=workers)
    assert d_atlas.path == "in_place_morph"
    assert d_atlas.t_morph_ms < 300.0
    assert d_legacy.path == "state_ship"
    assert d_legacy.t_morph_ms > 5.0 * d_atlas.t_morph_ms

    # Post-eviction contention at alpha=0.75 (1024^2): atlas_3d sees ~268 ms stall whereas legacy sees 0 ms.
    workers_pe = {
        0: WorkerState(gpu_id=0, expert_alpha=0.75, kv_free_bytes=16 * GIB, post_evict=True),
        1: WorkerState(gpu_id=1, expert_alpha=1.0, kv_free_bytes=16 * GIB),
    }
    req_pe = CPTGRequestSpec(
        req_id="r_1024_pe",
        src_gpu=0,
        architecture="tight_mot",
        model_family="bagel",
        kv_bytes=120 * MIB,
        resolution=1024,
        interleaved_roundtrip_gamma=1,
    )
    d_pe_atlas = r_atlas.decide(req_pe, workers=workers_pe)
    d_pe_legacy = r_legacy.decide(req_pe, workers=workers_pe)
    assert d_pe_atlas.t_morph_ms > 250.0
    assert d_pe_atlas.path == "state_ship"
    assert d_pe_legacy.t_morph_ms < 20.0

    # 3rd axis: colocated AR decode tasks stretch peer queue_wait_ms in atlas_3d.
    w_idle_ar = {
        0: WorkerState(gpu_id=0, expert_alpha=1.0, queue_wait_ms=1000.0, colocated_ar_b=0),
    }
    w_busy_ar = {
        0: WorkerState(gpu_id=0, expert_alpha=1.0, queue_wait_ms=1000.0, colocated_ar_b=4),
    }
    req_512 = CPTGRequestSpec(req_id="r_coloc", src_gpu=0, resolution=512, kv_bytes=64 * MIB)
    c_idle = r_atlas.decide(req_512, workers=w_idle_ar).t_morph_ms
    c_busy = r_atlas.decide(req_512, workers=w_busy_ar).t_morph_ms
    assert c_busy > c_idle * 1.4

    # Direct C ABI atlas estimator sanity.
    L = get_core_lib()
    est = call_json(
        L,
        L.ms_atlas_estimate_json,
        encode_json(
            {
                "cost_model_mode": "atlas_3d",
                "model_family": "bagel",
                "resolution": 1024,
                "alpha_g": 0.5,
                "is_post_evict": False,
                "colocated_ar_b": 8,
            }
        ),
    )
    assert est["from_atlas"] is True
    assert abs(est["exposed_stall_ms"] - 7.53) < 0.1
    assert est["dit_step_stretch_factor"] > 1.3


def test_memory_slack_gpu_lock_reorders_and_bounds_starvation():
    async def _run() -> tuple[list[str], dict[str, int]]:
        free_bytes = 4 * GIB
        lock = MemorySlackGpuLock(
            gpu_id=0,
            mode="memory_slack",
            max_wait_aging_ms=60_000.0,
            max_bypasses=2,
            free_mem_provider=lambda _g: free_bytes,
        )
        order: list[str] = []
        await lock.acquire()  # Hold GPU initially

        async def _waiter(name: str, mem_b: int, slack_ms: float) -> None:
            now_ms = time.perf_counter() * 1000.0
            await lock.acquire(
                est_mem_bytes=mem_b,
                est_exec_ms=1000.0,
                slo_deadline_ms=now_ms + slack_ms + 1000.0,
                op_priority=1,
            )
            try:
                order.append(name)
                await asyncio.sleep(0)
            finally:
                lock.release()

        # Enqueue heavy (8 GiB > 4 GiB free), tight-fit (1 GiB), tighter-fit (1 GiB), and late loose-fit (2 GiB).
        # Both 1 GiB waiters bypass heavy_8g (bypass_count hits max_bypasses=2), while late med_2g_loose is not
        # starving yet, so bounded anti-starvation MUST promote heavy_8g ahead of med_2g_loose.
        t_heavy = asyncio.create_task(_waiter("heavy_8g", 8 * GIB, 50_000.0))
        await asyncio.sleep(0.005)
        t_tight = asyncio.create_task(_waiter("tight_1g", 1 * GIB, 3_000.0))
        await asyncio.sleep(0.005)
        t_late = asyncio.create_task(_waiter("late_1g_tight", 1 * GIB, 2_000.0))
        await asyncio.sleep(0.005)
        t_med = asyncio.create_task(_waiter("med_2g_loose", 2 * GIB, 20_000.0))
        await asyncio.sleep(0.005)

        lock.release()
        await asyncio.gather(t_heavy, t_med, t_tight, t_late)
        st = lock.stats()
        lock.close()
        return order, st

    order, st = asyncio.run(_run())
    # Tightest fitting waiter runs first, then second tightest; after 2 bypasses, heavy_8g is promoted!
    assert order[0] == "late_1g_tight"
    assert order[1] == "tight_1g"
    assert order[2] == "heavy_8g"
    assert order[3] == "med_2g_loose"
    assert st["memory_bypasses_total"] >= 2
    assert st["starvation_promotions_total"] >= 1


def test_simulator_metrics_and_cluster_report_schema_parity():
    ph = MeasuredPhysicsProfile()
    hw = HardwareInterconnectSpec.presets(ph)[0]
    ws = WorkloadScaleSpec.bagel_7b_standard()
    raw = simulate_cluster_point(
        policy_name="morphserve",
        num_gpus=8,
        hw=hw,
        offered_qps=0.6,
        num_sessions=32,
        seed=2026,
        physics=ph,
        workload_scale=ws,
        policy_overrides={"steady_window_mode": "session_trim", "steady_state_trim_frac": 0.1},
    )
    rep = ClusterRunReport.from_dict(raw)
    assert rep.cost_model_mode == "pipeline_analytical"
    assert rep.gpu_queue_mode == "memory_slack"
    assert rep.kv_eviction_mode == "swap"
    assert rep.ar_ttft_mean_ms > 0.0
    assert rep.ar_tpot_mean_ms > 0.0
    assert rep.dit_turn_mean_ms > 0.0
    assert rep.session_e2e_mean_ms > 0.0
    assert rep.active_jct_mean_ms > 0.0
    assert rep.effective_hbm_util_mean_pct is not None and rep.effective_hbm_util_mean_pct > 0.0
    assert "all_session_e2e_p99_ms" in raw
    assert "all_session_dit_p99_ms" in raw
    assert "all_session_active_jct_p99_ms" in raw
