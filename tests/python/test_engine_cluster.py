"""Seam tests for `MorphServeClusterEngine`, `WorkerFleet`, `DualPathMorphRouter`, `PeerByteTransport`, `EnginePolicyConfig`, and `ClusterSimulator`."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
import threading
import time
from typing import Any

import pytest

from morphserve.engine import (
    ClusterRunReport,
    EnginePolicyConfig,
    InProcessWorkerFleet,
    InterleavedSession,
    InterleavedTurn,
    LocalNodeWorkerHost,
    MAIN_QPS_POLICIES,
    MorphServeClusterEngine,
    MultinodeCoordinatorHub,
    RunWindowOptions,
    STATIC_SPLIT_POLICIES,
    TcpMultinodeWorkerFleet,
    VALID_POLICIES,
    WorkerBudgetPlanner,
    WorkerFleet,
    WorkerRuntime,
    WorkloadSuite,
    build_ablation_configs,
    build_static_split_configs,
    hamilton_apportion,
    run_multinode_worker_daemon,
)
from morphserve.engine.cluster_engine import MemorySlackGpuLock
from morphserve.models import SyntheticCPTGAdapter
from morphserve.router import (
    CPTGRequestSpec,
    ClusterTopologyProfile,
    DualPathMorphRouter,
    OperatorLatencyProfile,
    WorkerState,
    default_operator_profile,
    default_topology_for,
)
from morphserve.runtime import (
    InMemoryByteBus,
    InMemoryPeerMesh,
    TcpByteMesh,
    find_free_tcp_port,
)
from morphserve.sim import (
    HardwareInterconnectSpec,
    MeasuredPhysicsProfile,
    SimGpuMemoryLedger,
    WorkloadScaleSpec,
    simulate_cluster_point,
)


def _make_multiturn_suite(name: str = "unit_multiturn") -> WorkloadSuite:
    return WorkloadSuite(
        name=name,
        trace_source="unit_synthetic",
        base_qps=2.0,
        sessions=[
            InterleavedSession(
                session_id="sess_interleaved_0",
                quadrant="interleaved_co_creation",
                source_dataset="unit",
                arrival_offset_s=0.0,
                turns=[
                    InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="plan alpine cabin", max_ar_tokens=4),
                    InterleavedTurn(turn_idx=1, modality_plan="dit_t2i", prompt="alpine cabin", height=512, width=512, dit_steps=2),
                    InterleavedTurn(
                        turn_idx=2,
                        modality_plan="ar_only",
                        prompt="inspect cabin",
                        max_ar_tokens=4,
                        use_prev_image=True,
                    ),
                    InterleavedTurn(
                        turn_idx=3,
                        modality_plan="dit_edit",
                        prompt="add warm lantern",
                        height=512,
                        width=512,
                        dit_steps=2,
                        use_prev_image=True,
                    ),
                ],
            ),
            InterleavedSession(
                session_id="sess_vqa_1",
                quadrant="multi_image_vqa",
                source_dataset="unit",
                arrival_offset_s=0.02,
                turns=[
                    InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="describe scene", max_ar_tokens=4),
                    InterleavedTurn(turn_idx=1, modality_plan="ar_only", prompt="follow up question", max_ar_tokens=4),
                ],
            ),
            InterleavedSession(
                session_id="sess_cot_2",
                quadrant="visual_cot_highres",
                source_dataset="unit",
                arrival_offset_s=0.04,
                turns=[
                    InterleavedTurn(
                        turn_idx=0,
                        modality_plan="cot_then_dit",
                        prompt="reason and draw lighthouse",
                        max_ar_tokens=4,
                        height=1024,
                        width=1024,
                        dit_steps=2,
                    ),
                ],
            ),
        ],
    )


# ==================== 1. Policy Config, RunWindowOptions, BudgetPlanner & Apportionment ====================


def test_policy_config_presets_serialization_ablations_and_run_window_options(tmp_path: Path) -> None:
    assert len(VALID_POLICIES) == 7
    assert "monolithic_kv_swap" in VALID_POLICIES
    assert len(MAIN_QPS_POLICIES) == 5
    assert "monolithic_kv_swap" in MAIN_QPS_POLICIES

    for pname in VALID_POLICIES + STATIC_SPLIT_POLICIES:
        cfg = EnginePolicyConfig.from_preset(pname, num_gpus=8)
        assert cfg.policy_name == pname
        rebuilt = EnginePolicyConfig.from_dict(cfg.to_dict())
        assert rebuilt == cfg
        rebuilt_json = EnginePolicyConfig.from_json(cfg.to_json())
        assert rebuilt_json == cfg

    # Verify unified SLO & window defaults + `monolithic_kv_swap` fairness preset
    p_ms = EnginePolicyConfig.from_preset("morphserve", num_gpus=8)
    p_mono = EnginePolicyConfig.from_preset("monolithic_dp", num_gpus=8)
    p_mono_swap = EnginePolicyConfig.from_preset("monolithic_kv_swap", num_gpus=8)
    assert p_ms.ar_tpot_slo_ms == pytest.approx(101.40, abs=1e-2)
    assert p_ms.dit_512_slo_ms == pytest.approx(14916.30, abs=1e-2)
    assert p_ms.dit_1024_slo_ms == pytest.approx(57219.60, abs=1e-2)
    assert p_ms.warmup_window_s == pytest.approx(60.0)
    assert p_ms.steady_window_s == pytest.approx(150.0)
    assert p_ms.cooldown_s == pytest.approx(30.0)
    assert p_ms.ar_ttft_slo_for(False, 0, 512) == pytest.approx(250.0, abs=1e-2)
    assert p_ms.ar_ttft_slo_for(True, 1, 512) == pytest.approx(2800.0, abs=1e-2)
    assert p_ms.ar_ttft_slo_for(True, 1, 1024) > p_ms.ar_ttft_slo_for(True, 1, 512)
    assert p_ms.use_idle_kv_swap is True
    assert p_mono.use_idle_kv_swap is False
    assert p_mono_swap.use_idle_kv_swap is True
    assert p_mono_swap.use_tripool_vmm is False
    assert p_mono_swap.use_scoped_vae is False
    assert p_mono_swap.routing_mode == p_mono.routing_mode

    ablations = dict(build_ablation_configs(num_gpus=8))
    assert set(ablations) == {
        "full_morphserve",
        "no_m1_tripool_vmm",
        "no_m1_scoped_vae",
        "no_m2_dual_path",
        "no_m2_pacer_numa",
        "no_m3_cow_loopback",
        "no_m4_layer_chunk",
        "no_queue_sched",
    }
    assert ablations["full_morphserve"].use_tripool_vmm is True
    assert ablations["no_m1_tripool_vmm"].use_tripool_vmm is False
    assert ablations["no_m1_tripool_vmm"].use_expert_pager is False
    assert ablations["no_m1_tripool_vmm"].use_scoped_vae is True
    assert ablations["no_m1_scoped_vae"].use_tripool_vmm is True
    assert ablations["no_m1_scoped_vae"].use_scoped_vae is False
    assert ablations["no_m2_dual_path"].routing_mode == "sticky_least_loaded"
    assert ablations["no_m2_dual_path"].use_h2d_pacer is True
    assert ablations["no_m2_pacer_numa"].routing_mode == "dual_path"
    assert ablations["no_m2_pacer_numa"].use_h2d_pacer is False
    assert ablations["no_m2_pacer_numa"].use_shared_numa_mirror is False
    assert ablations["no_m3_cow_loopback"].use_module3_cow_loopback is False
    assert ablations["no_m4_layer_chunk"].preemption_mode != "layer_chunk"
    assert ablations["no_queue_sched"].gpu_queue_mode == "fcfs"

    splits = dict(build_static_split_configs(num_gpus=8))
    assert len(splits["static_2und_6gen"].static_und_ranks) == 2
    assert len(splits["static_2und_6gen"].static_gen_ranks) == 6
    assert len(splits["static_6und_2gen"].static_und_ranks) == 6
    assert len(splits["static_6und_2gen"].static_gen_ranks) == 2

    # RunWindowOptions must return a modified copy without mutating the source policy.
    base = EnginePolicyConfig.from_preset("morphserve", num_gpus=8)
    orig_ttft = base.ar_ttft_slo_ms
    orig_warmup = base.warmup_window_s
    orig_cooldown = base.cooldown_s
    opts = RunWindowOptions(
        inter_turn_think_s_range=(0.5, 2.5),
        think_time_dist="uniform",
        steady_window_mode="time",
        warmup_window_s=15.0,
        steady_window_s=60.0,
        cooldown_s=20.0,
        slo_scale_k=4.0,
        dit_slo_scale_k=6.0,
        ar_tpot_slo_mode="turn_mean",
    )
    eff = opts.apply_to_policy(base)
    assert base.ar_ttft_slo_ms == orig_ttft
    assert base.warmup_window_s == orig_warmup
    assert base.cooldown_s == orig_cooldown
    assert eff.inter_turn_think_s_range == (0.5, 2.5)
    assert eff.think_time_dist == "uniform"
    assert eff.steady_window_mode == "time"
    assert eff.warmup_window_s == 15.0
    assert eff.steady_window_s == 60.0
    assert eff.cooldown_s == 20.0
    assert eff.ar_tpot_slo_mode == "turn_mean"
    assert eff.slo_scale_k == 4.0
    assert eff.dit_slo_scale_k == 6.0
    assert eff.ar_ttft_slo_ms > orig_ttft or eff.ar_ttft_slo_ms != orig_ttft

    # Public WorkerBudgetPlanner seam
    plan = WorkerBudgetPlanner.compute_internvlu_memory_plan(
        total_dev_mib=40960.0,
        free_dev_mib=36000.0,
        base_pytorch_alloc_mib=4200.0,
        mirror_bytes=4 * (1024**3),
        policy_super_pool_mib=4864,
        policy_fine_kv_pool_mib=512,
        gpu_memory_quota_mib=12288.0,
    )
    assert plan["allow_overflow"] is False
    assert plan["super_pool_mib"] > 0 and plan["pytorch_cap_mib"] > 0
    ov, budget = WorkerBudgetPlanner.compute_internvlu_quota_overrides(
        "morphserve",
        num_gpus=8,
        vendor_dir=tmp_path,
        constrained_gib=12.0,
        total_gpu_gib=40.0,
    )
    assert ov["allow_vmm_overflow"] is False
    assert budget["runtime_combined_budget_mib"] <= budget["quota_limit_mib"] + 1e-3
    pol_ivlu = EnginePolicyConfig.from_preset("morphserve", num_gpus=8, **ov)
    pass_audit = WorkerBudgetPlanner.verify_internvlu_memory_quota(
        {
            "peak_pytorch_reserved_mib": 4500.0,
            "vmm_physical_pool_mib": 6000.0,
            "peak_gpu_reserved_mib": 10500.0,
            "vmm_overflow_creates": 0,
        },
        pol_ivlu,
        constrained_gib=12.0,
    )
    assert pass_audit["within_quota"] is True


def test_hamilton_apportion_invariants() -> None:
    res = hamilton_apportion(10, {"a": 0.34, "b": 0.33, "c": 0.33})
    assert sum(res.values()) == 10
    assert res == {"a": 4, "b": 3, "c": 3}

    capped = hamilton_apportion(10, {"a": 0.8, "b": 0.2}, caps={"a": 5, "b": 10})
    assert capped == {"a": 5, "b": 5}

    assert hamilton_apportion(0, {"a": 1.0, "b": 2.0}) == {"a": 0, "b": 0}
    with pytest.raises(ValueError):
        hamilton_apportion(-1, {"a": 1.0})


# ==================== 2. Topology & DualPathMorphRouter C++ Write-Through ====================


def test_topology_defaults_and_router_operator_profile_readonly_and_telemetry_seq() -> None:
    op_default = default_operator_profile()
    assert op_default.bagel_expert_bytes > 0
    assert op_default.ar_tpot_ms == pytest.approx(50.70, abs=1e-2)
    assert op_default.internvlu_ar_tpot_ms == pytest.approx(44.0, abs=1e-2)
    assert op_default.ar_text_unloaded_ttft_ms == pytest.approx(125.0, abs=1e-2)
    assert op_default.ar_img_512_unloaded_ttft_ms == pytest.approx(1400.0, abs=1e-2)
    slos_bagel = op_default.slo_thresholds_ms(k=2.0, model_family="bagel")
    assert slos_bagel["ar_ttft_slo_text_ms"] == pytest.approx(250.0, abs=1e-2)
    assert slos_bagel["ar_ttft_slo_img512_ms"] == pytest.approx(2800.0, abs=1e-2)
    assert slos_bagel["ar_ttft_slo_img1024_ms"] > slos_bagel["ar_ttft_slo_img512_ms"]
    slos_ivlu = op_default.slo_thresholds_ms(k=2.0, model_family="internvl_u")
    assert slos_ivlu["ar_tpot_slo_ms"] == pytest.approx(88.0, abs=1e-2)

    topo_4 = default_topology_for(4)
    assert topo_4.num_gpus == 4
    topo_8 = default_topology_for(8)
    assert topo_8.num_gpus == 8
    topo_16 = default_topology_for(16, gpus_per_node=8)
    assert topo_16.num_gpus == 16
    assert topo_16.same_node(0, 7) is True
    assert topo_16.same_node(0, 8) is False
    # Inter-node default bandwidth must be canonical 11.5 GB/s (no 1.15 GB/s drift).
    assert topo_16.effective_p2p_gbps(0, 8) >= 10.0

    # Custom OperatorLatencyProfile must be honored by C++ DualPathMorphRouter!
    op_heavy = replace(op_default, bagel_expert_bytes=op_default.bagel_expert_bytes * 2)
    r_default = DualPathMorphRouter(topo_8, operator_profile=op_default)
    r_heavy = DualPathMorphRouter(topo_8, operator_profile=op_heavy)
    try:
        x_default = r_default.analytical_crossover_kv_mib(alpha=0.5, src_gpu=0, dst_gpu=1)
        x_heavy = r_heavy.analytical_crossover_kv_mib(alpha=0.5, src_gpu=0, dst_gpu=1)
        assert x_heavy > x_default

        r_default.init_workers(8, lambda _g: "hybrid", initial_alphas={g: 0.5 for g in range(8)})
        # AR queue reserve & release mutate C++ directly and clamp at 0.
        r_default.reserve_ar_queue(0, 250.0)
        assert abs(r_default.get_worker_state(0).queue_wait_ms - 250.0) < 1e-6
        r_default.release_ar_queue(0, 100.0)
        assert abs(r_default.get_worker_state(0).queue_wait_ms - 150.0) < 1e-6
        r_default.release_ar_queue(0, 999.0)
        assert r_default.get_worker_state(0).queue_wait_ms == 0.0

        # Explicit write-through via set_worker_state_fields + monotonic telemetry_seq enforcement.
        r_default.set_worker_state_fields(1, expert_alpha=1.0, queue_wait_ms=0.0, kv_free_bytes=20 * (1024**3))
        assert r_default.get_worker_state(1).expert_alpha == 1.0
        r_default.update_worker_telemetry(1, kv_free_bytes=12 * (1024**3), telemetry_seq=10)
        assert r_default.get_worker_state(1).kv_free_bytes == 12 * (1024**3)
        # Stale telemetry_seq must be ignored!
        r_default.update_worker_telemetry(1, kv_free_bytes=1 * (1024**3), telemetry_seq=5)
        assert r_default.get_worker_state(1).kv_free_bytes == 12 * (1024**3)
        # Newer telemetry_seq updates C++ state.
        r_default.update_worker_telemetry(1, kv_free_bytes=15 * (1024**3), telemetry_seq=11)
        assert r_default.get_worker_state(1).kv_free_bytes == 15 * (1024**3)

        # Readonly transition plan must NOT mutate worker queue or KV reservations!
        snap_before = r_default.worker_states_snapshot()
        req = CPTGRequestSpec(
            req_id="ro_test",
            src_gpu=0,
            kv_bytes=16 * (1024**2),
            dit_step1_ms=280.0,
            dit_total_steps=20,
            resolution=512,
        )
        dec_ro, ship_ro = r_default.plan_transition_readonly(
            routing_mode="dual_path",
            static_und_ranks=(),
            static_gen_ranks=(),
            req=req,
            est_dit_ms=5000.0,
            now_ms=0.0,
        )
        snap_after = r_default.worker_states_snapshot()
        assert snap_after[dec_ro.dst_gpu].queue_wait_ms == snap_before[dec_ro.dst_gpu].queue_wait_ms
        assert snap_after[dec_ro.dst_gpu].kv_free_bytes == snap_before[dec_ro.dst_gpu].kv_free_bytes

        # Mutating plan_and_reserve must match decision and actually reserve queue wait.
        dec_mut, ship_mut = r_default.plan_and_reserve_transition(
            routing_mode="dual_path",
            static_und_ranks=(),
            static_gen_ranks=(),
            req=req,
            est_dit_ms=5000.0,
            now_ms=0.0,
        )
        assert dec_mut.path == dec_ro.path
        assert dec_mut.dst_gpu == dec_ro.dst_gpu
        assert ship_mut == ship_ro
        assert r_default.get_worker_state(dec_mut.dst_gpu).queue_wait_ms >= snap_before[dec_mut.dst_gpu].queue_wait_ms + 4999.0

        # Fair load-balanced `twostage_shm` gen selection (no rigid 2k -> 2k+1 pairing bottleneck)
        p_shm = EnginePolicyConfig.from_preset("twostage_shm", num_gpus=8)
        shm_workers = {
            g: WorkerState(
                gpu_id=g,
                role=p_shm.role_of(g),
                expert_alpha=1.0 if g in p_shm.static_gen_ranks else 0.0,
                queue_wait_ms=9000.0 if g == p_shm.static_gen_ranks[0] else 0.0,
                kv_free_bytes=16 * (1024**3),
            )
            for g in range(8)
        }
        chosen_gen = r_default.select_static_gen_gpu(
            routing_mode="twostage_shm",
            static_und_ranks=p_shm.static_und_ranks,
            static_gen_ranks=p_shm.static_gen_ranks,
            src_und_gpu=p_shm.static_und_ranks[0],
            workers=shm_workers,
            gpu_busy={p_shm.static_gen_ranks[0]: True},
        )
        assert chosen_gen in p_shm.static_gen_ranks
        assert chosen_gen != p_shm.static_gen_ranks[0]

        # Fair 2x8 multi-node `static_nccl_fission` must prefer same-node gen rank over remote-node gen rank.
        r16 = DualPathMorphRouter(topo_16, operator_profile=op_default)
        try:
            p_fis16 = EnginePolicyConfig.from_preset("static_nccl_fission", num_gpus=16)
            w16 = {
                g: WorkerState(
                    gpu_id=g,
                    role=p_fis16.role_of(g),
                    expert_alpha=1.0 if g in p_fis16.static_gen_ranks else 0.0,
                    queue_wait_ms=1500.0 if topo_16.same_node(0, g) else 0.0,
                    kv_free_bytes=16 * (1024**3),
                )
                for g in range(16)
            }
            chosen_fis16 = r16.select_static_gen_gpu(
                routing_mode="static_nccl_fission",
                static_und_ranks=p_fis16.static_und_ranks,
                static_gen_ranks=p_fis16.static_gen_ranks,
                src_und_gpu=0,
                workers=w16,
                gpu_busy={},
            )
            assert chosen_fis16 in p_fis16.static_gen_ranks
            assert topo_16.same_node(0, chosen_fis16) is True
        finally:
            r16.close()
    finally:
        r_default.close()
        r_heavy.close()


# ==================== 3. Unified PeerByteTransport (InMemory & TCP) ====================


def test_peer_byte_transport_bidirectional_inmemory_and_tcp() -> None:
    port = find_free_tcp_port()
    assert 1024 <= port <= 65535

    payload_01 = bytes((i * 13 + 7) % 256 for i in range(65536))
    payload_10 = bytes((i * 29 + 3) % 256 for i in range(49152))

    # 1. InMemoryByteBus + InMemoryPeerMesh bidirectional exchange + SHM put/pop
    bus = InMemoryByteBus(world_size=2)
    with InMemoryPeerMesh(bus, 0) as ep0, InMemoryPeerMesh(bus, 1) as ep1:
        got: dict[str, bytes] = {}

        def _r0() -> None:
            ep0.send_u8(payload_01, dst_rank=1)
            data, _ = ep0.recv_u8(len(payload_10), src_rank=1)
            got["r0"] = bytes(data)

        def _r1() -> None:
            ep1.send_u8(payload_10, dst_rank=0)
            data, _ = ep1.recv_u8(len(payload_01), src_rank=0)
            got["r1"] = bytes(data)

        t0 = threading.Thread(target=_r0)
        t1 = threading.Thread(target=_r1)
        t0.start()
        t1.start()
        t0.join(timeout=5.0)
        t1.join(timeout=5.0)
        assert not t0.is_alive() and not t1.is_alive()
        assert got["r0"] == payload_10
        assert got["r1"] == payload_01

        bus.put_shm("k0", b"shared-payload")
        assert bus.pop_shm("k0") == b"shared-payload"
    bus.close()

    # 2. Full-duplex TcpByteMesh concurrent bidirectional exchange without deadlock
    with TcpByteMesh(world_size=2, bind_host="127.0.0.1") as mesh:
        ep0 = mesh.endpoint_for(0)
        ep1 = mesh.endpoint_for(1)
        got_tcp: dict[str, bytes] = {}
        errs: list[BaseException] = []

        def _t0_send() -> None:
            try:
                ep0.send_u8(payload_01, dst_rank=1)
            except BaseException as exc:
                errs.append(exc)

        def _t0_recv() -> None:
            try:
                data, _ = ep0.recv_u8(len(payload_10), src_rank=1)
                got_tcp["r0"] = bytes(data)
            except BaseException as exc:
                errs.append(exc)

        def _t1_send() -> None:
            try:
                ep1.send_u8(payload_10, dst_rank=0)
            except BaseException as exc:
                errs.append(exc)

        def _t1_recv() -> None:
            try:
                data, _ = ep1.recv_u8(len(payload_01), src_rank=0)
                got_tcp["r1"] = bytes(data)
            except BaseException as exc:
                errs.append(exc)

        threads = [threading.Thread(target=fn) for fn in (_t0_send, _t0_recv, _t1_send, _t1_recv)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=10.0)
        assert not errs
        assert all(not th.is_alive() for th in threads)
        assert got_tcp["r0"] == payload_10
        assert got_tcp["r1"] == payload_01


# ==================== 4. Deterministic WorkerRuntime Public Step Seam & Clock Skew ====================


def test_worker_runtime_step_once_batching_and_clock_skew_immunity() -> None:
    pol = EnginePolicyConfig.from_preset("morphserve", num_gpus=1, model_family="synthetic")
    adapter = SyntheticCPTGAdapter(rank=0, policy=pol)
    bus = InMemoryByteBus(world_size=1)
    mesh = InMemoryPeerMesh(bus, rank=0)
    acks: list[dict[str, Any]] = []
    rt = WorkerRuntime(rank=0, policy=pol, adapter=adapter, peer_mesh=mesh, ack_sink=acks.append, shm_bus=bus)
    try:
        # Enqueue two concurrent EXEC_AR commands with extreme remote clock skew (+3600s).
        skewed_wall = time.perf_counter() + 3600.0
        rt.enqueue_command(
            {
                "cmd_id": "c1",
                "op": "EXEC_AR",
                "ctx_id": "ctx_a",
                "prompt": "first prompt",
                "max_ar_tokens": 3,
                "enqueued_wall_s": skewed_wall,
            }
        )
        rt.enqueue_command(
            {
                "cmd_id": "c2",
                "op": "EXEC_AR",
                "ctx_id": "ctx_b",
                "prompt": "second prompt",
                "max_ar_tokens": 3,
                "enqueued_wall_s": skewed_wall,
            }
        )
        assert rt.step_once(timeout_s=0.1) is True
        assert {a["cmd_id"] for a in acks} == {"c1", "c2"}
        for ack in acks:
            assert ack["status"] == "ok"
            # Skewed wall clock must NOT inflate TTFT by 3,600,000 ms!
            assert 0.0 <= float(ack["ttft_ms"]) < 500.0
            assert int(ack["kv_free_bytes"]) == adapter.available_kv_bytes()

        # Direct handle_command seam for PREP_DIT -> EXEC_DIT -> GET_STATS -> RELEASE_CTX
        prep = rt.handle_command(
            {"cmd_id": "c3", "op": "PREP_DIT", "ctx_id": "ctx_a", "prompt": "draw", "height": 512, "width": 512}
        )
        assert prep is not None and prep["status"] == "ok"
        dit = rt.handle_command(
            {"cmd_id": "c4", "op": "EXEC_DIT", "ctx_id": "ctx_a", "prompt": "draw", "height": 512, "width": 512, "dit_steps": 2}
        )
        assert dit is not None and dit["status"] == "ok"
        stats = rt.handle_command({"cmd_id": "c5", "op": "GET_STATS"})
        assert stats is not None and stats["status"] == "ok"
        rel = rt.handle_command({"cmd_id": "c6", "op": "RELEASE_CTX", "ctx_id": "ctx_a"})
        assert rel is not None and rel["status"] == "ok"
    finally:
        rt.handle_command({"cmd_id": "c_shutdown", "op": "SHUTDOWN"})
        mesh.close()
        bus.close()


# ==================== 5. MorphServeClusterEngine E2E, RunWindowOptions & OOM Retry ====================


def test_cluster_engine_multiturn_continuity_run_options_and_canonical_report(tmp_path: Path) -> None:
    pol = EnginePolicyConfig.from_preset("morphserve", num_gpus=2, model_family="synthetic", inter_turn_think_s_range=(0.0, 0.0))
    suite = _make_multiturn_suite()
    save_dir = tmp_path / "outputs"
    fleet = InProcessWorkerFleet(pol)
    engine = MorphServeClusterEngine(policy=pol, fleet=fleet, tmp_root=tmp_path / "tmp")
    engine.start()
    try:
        orig_think = engine.policy.inter_turn_think_s_range
        orig_mode = engine.policy.steady_window_mode
        opts = RunWindowOptions(
            inter_turn_think_s_range=(0.0, 0.01),
            think_time_dist="uniform",
            steady_window_mode="session_trim",
            steady_state_trim_frac=0.0,
        )
        rep = engine.run_suite(suite, target_qps=4.0, save_outputs_dir=save_dir, run_options=opts)
        assert isinstance(rep, ClusterRunReport)
        # Running with RunWindowOptions must NEVER mutate engine.policy.
        assert engine.policy.inter_turn_think_s_range == orig_think
        assert engine.policy.steady_window_mode == orig_mode

        assert rep.total_sessions == len(suite.sessions)
        assert rep.completed_sessions == len(suite.sessions)
        assert rep.oom_sessions == 0
        assert rep.error_sessions == 0
        assert all(isinstance(lk, MemorySlackGpuLock) for lk in engine._gpu_locks)
        # Canonical report schema invariants
        assert rep.joint_slo_goodput_rps == rep.steady_joint_slo_goodput_rps
        assert rep.joint_slo_attainment_pct == rep.steady_joint_slo_attainment_pct
        rep_dict = rep.to_dict()
        assert "full_trace_goodput_rps" in rep_dict
        assert "full_trace_joint_slo_attainment_pct" in rep_dict
        assert "vmm_physical_pool_mib" in rep_dict
        assert "kv_host_swap_ins" in rep_dict
        assert "alpha_timeline" in rep_dict
        assert "latency_breakdown_ms" in rep_dict
        assert "nvml_sm_util_mean_pct" in rep_dict
        assert "actual_steady_arrival_qps" in rep_dict
        assert "steady_turn_slo_goodput_rps" in rep_dict
        assert set(rep_dict["latency_breakdown_ms"]).issuperset(
            {"queue_ms", "h2d_weight_ms", "kv_transfer_ms", "vae_ms", "preempt_yield_ms"}
        )
        # Verify multi-turn image outputs were saved cleanly
        assert (save_dir / "sess_interleaved_0_t1.png").is_file()
        assert (save_dir / "sess_interleaved_0_t3.png").is_file()
    finally:
        engine.shutdown()


def test_cluster_engine_oom_retry_and_exhaustion(tmp_path: Path) -> None:
    pol = EnginePolicyConfig.from_preset(
        "morphserve", num_gpus=2, model_family="synthetic", inter_turn_think_s_range=(0.0, 0.0)
    )
    # 1. Rank 0 OOMs transiently while rank 1 succeeds -> session recovers via OOM retry!
    fleet_recover = InProcessWorkerFleet(
        pol,
        adapter_factory=lambda rank, policy: SyntheticCPTGAdapter(
            rank=rank,
            policy=policy,
            oom_ranks=(0,),
        ),
    )
    suite_one = WorkloadSuite(
        name="oom_retry_suite",
        trace_source="unit",
        base_qps=1.0,
        sessions=[
            InterleavedSession(
                session_id="s_oom_0",
                quadrant="multi_image_vqa",
                source_dataset="unit",
                arrival_offset_s=0.0,
                turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="oom_trigger", max_ar_tokens=3)],
            )
        ],
    )
    engine = MorphServeClusterEngine(policy=pol, fleet=fleet_recover, tmp_root=tmp_path / "recover")
    engine.start()
    try:
        rep = engine.run_suite(suite_one, target_qps=1.0)
        assert rep.completed_sessions == 1
        assert rep.oom_sessions == 0
    finally:
        engine.shutdown()

    # 2. Both ranks OOM -> cleanly reported as oom_sessions == 1 with no hang.
    fleet_all_oom = InProcessWorkerFleet(
        pol,
        adapter_factory=lambda rank, policy: SyntheticCPTGAdapter(
            rank=rank,
            policy=policy,
            oom_prompts=("oom_trigger",),
            oom_ranks=(0, 1),
        ),
    )
    engine2 = MorphServeClusterEngine(policy=pol, fleet=fleet_all_oom, tmp_root=tmp_path / "all_oom")
    engine2.start()
    try:
        rep2 = engine2.run_suite(suite_one, target_qps=1.0)
        assert rep2.oom_sessions == 1
        assert rep2.completed_sessions == 0
    finally:
        engine2.shutdown()


# ==================== 6. Worker Fault Fail-Fast & Multinode Fleet End-to-End ====================


class _FaultInjectingFleet(WorkerFleet):
    """Minimal public `WorkerFleet` implementation that triggers `on_worker_fault` on the first AR command."""

    def __init__(self, num_gpus: int = 1) -> None:
        self.num_gpus = int(num_gpus)
        self._on_ack: Callable[[dict[str, Any]], None] | None = None
        self._on_fault: Callable[[list[int], str], None] | None = None

    def start(
        self,
        on_ack: Callable[[dict[str, Any]], None],
        on_worker_fault: Callable[[list[int], str], None] | None = None,
    ) -> dict[str, Any]:
        self._on_ack = on_ack
        self._on_fault = on_worker_fault
        return {
            "host_shmem_delta_gib": 0.0,
            "workers": {
                r: {
                    "role": "hybrid",
                    "resident_alpha": 1.0,
                    "kv_free_bytes": 16 * (1024**3),
                }
                for r in range(self.num_gpus)
            },
        }

    def send(self, rank: int, payload: dict[str, Any]) -> None:
        op = payload.get("op")
        if op in ("RESET_SUITE", "GET_STATS", "RELEASE_CTX"):
            if self._on_ack is not None:
                self._on_ack({"cmd_id": payload.get("cmd_id"), "rank": rank, "status": "ok", "stats": {}})
            return
        # Simulate abrupt worker death on first workload RPC.
        if self._on_fault is not None:
            threading.Thread(
                target=self._on_fault,
                args=([rank], "simulated child SIGKILL"),
                daemon=True,
            ).start()

    def shutdown(self) -> None:
        return None


def test_cluster_engine_worker_fault_fails_fast_without_hanging(tmp_path: Path) -> None:
    pol = EnginePolicyConfig.from_preset("morphserve", num_gpus=1, model_family="synthetic", inter_turn_think_s_range=(0.0, 0.0))
    fleet = _FaultInjectingFleet(num_gpus=1)
    engine = MorphServeClusterEngine(policy=pol, fleet=fleet, tmp_root=tmp_path / "fault")
    engine.start()
    try:
        suite = WorkloadSuite(
            name="fault_suite",
            trace_source="unit",
            base_qps=1.0,
            sessions=[
                InterleavedSession(
                    session_id="s_fault_0",
                    quadrant="multi_image_vqa",
                    source_dataset="unit",
                    arrival_offset_s=0.0,
                    turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="hello", max_ar_tokens=4)],
                )
            ],
        )
        t0 = time.perf_counter()
        rep = engine.run_suite(suite, target_qps=1.0)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0
        assert rep.error_sessions == 1
        assert rep.completed_sessions == 0
    finally:
        engine.shutdown()


def test_local_node_worker_host_and_multinode_tcp_fleet_end_to_end_and_disconnect(tmp_path: Path) -> None:
    # 1. Single-node LocalNodeWorkerHost direct public lifecycle check
    pol1 = EnginePolicyConfig.from_preset("morphserve", num_gpus=2, model_family="synthetic")
    host_acks: list[dict[str, Any]] = []
    host = LocalNodeWorkerHost(
        node_rank=0,
        gpus_per_node=2,
        policy=pol1,
        model_dir=tmp_path,
        bench_root=tmp_path,
        backend="inproc_mem",
        on_ack=host_acks.append,
    )
    host.prepare_until_mesh()
    ready = host.finalize_mesh_and_start()
    try:
        assert len(ready["workers"]) == 2
        assert host.check_worker_liveness() == []
    finally:
        host.shutdown()

    # 2. 2-node x 2-GPU (4 ranks) TcpMultinodeWorkerFleet + run_multinode_worker_daemon end-to-end
    pol4 = EnginePolicyConfig.from_preset(
        "morphserve",
        num_gpus=4,
        model_family="synthetic",
        inter_turn_think_s_range=(0.0, 0.0),
    )
    topo4 = default_topology_for(4, gpus_per_node=2)
    port = find_free_tcp_port()
    hub = MultinodeCoordinatorHub(
        num_nodes=2,
        gpus_per_node=2,
        port=port,
        bind_host="127.0.0.1",
        advertise_host="127.0.0.1",
        backend="inproc_tcp",
        connect_timeout_s=30.0,
    )

    daemon_errs: list[BaseException] = []

    def _run_remote_daemon() -> None:
        try:
            run_multinode_worker_daemon(
                master_addr="127.0.0.1",
                port=hub.bound_port,
                node_rank=1,
                num_nodes=2,
                gpus_per_node=2,
                advertise_host="127.0.0.1",
                connect_timeout_s=30.0,
            )
        except BaseException as exc:
            daemon_errs.append(exc)

    d1 = threading.Thread(target=_run_remote_daemon, daemon=True)
    d1.start()

    fleet = TcpMultinodeWorkerFleet(policy=pol4, model_dir=tmp_path, bench_root=tmp_path, hub=hub)
    engine = MorphServeClusterEngine(policy=pol4, topology=topo4, fleet=fleet, tmp_root=tmp_path / "mn")
    try:
        engine.start()
        rep = engine.run_suite(_make_multiturn_suite("mn_2x2"), target_qps=4.0)
        assert rep.completed_sessions == 3
        assert rep.oom_sessions == 0
        assert rep.error_sessions == 0
    finally:
        engine.shutdown()
        hub.close()
        d1.join(timeout=10.0)
    assert not daemon_errs


# ==================== 7. ClusterSimulator Public Seam & Ledger ====================


def test_cluster_simulator_custom_sessions_full_overrides_and_ledger_stats() -> None:
    ph_zero = MeasuredPhysicsProfile()
    ph_def = MeasuredPhysicsProfile.default()
    assert ph_zero == ph_def
    ph_ov = MeasuredPhysicsProfile(vae_512_ms=ph_def.vae_512_ms + 5.0)
    assert ph_ov.vae_512_ms == ph_def.vae_512_ms + 5.0
    with pytest.raises(TypeError):
        MeasuredPhysicsProfile(nonexistent_physics_field=1.0)

    pol = EnginePolicyConfig.from_preset("morphserve", num_gpus=8)
    with SimGpuMemoryLedger(
        gpu_id=0,
        total_hbm_mib=ph_def.driver_total_hbm_40g_mib,
        policy=pol,
        role="hybrid",
        resident_expert_layers=14,
        physics=ph_def,
    ) as ledger:
        st0 = ledger.stats()
        assert int(st0["free_kv_bytes"]) == ledger.free_kv_bytes()
        assert ledger.upsert_kv(1, 512.0) is True
        ok, _ = ledger.exec_dit_vae(512, active_sid=1)
        assert ok is True
        st1 = ledger.stats()
        assert st1["peak_used_mib"] <= ph_def.driver_total_hbm_40g_mib + 1e-3
        ledger.release_session(1)
        assert ledger.stats()["active_kv_mib"] == 0.0

    hw = HardwareInterconnectSpec.presets(ph_def)[0]
    custom = [
        (0.0, "interleaved_co_creation", (("ar_only", 512), ("dit_t2i", 512))),
        (500.0, "multiturn_edit", (("dit_edit", 512), ("dit_edit", 512))),
    ]
    sim_rep = simulate_cluster_point(
        "morphserve",
        num_gpus=8,
        hw=hw,
        offered_qps=0.5,
        num_sessions=len(custom),
        seed=2026,
        physics=ph_def,
        workload_scale=WorkloadScaleSpec.bagel_7b_standard(),
        custom_sessions=custom,
        policy_overrides={"allow_cascade_cfg_split": True, "steady_state_trim_frac": 0.0},
    )
    assert sim_rep["total_sessions"] == len(custom)
    assert sim_rep["joint_slo_goodput_rps"] == sim_rep["steady_joint_slo_goodput_rps"]
    assert sim_rep["joint_slo_attainment_pct"] == sim_rep["steady_joint_slo_attainment_pct"]


def test_worker_runtime_recv_ready_event_and_normal_dit_requeues_leftover_preempt() -> None:
    """Verify ALLOC_RECV_BUF -> RECV_NCCL passes `ready_event` and EXEC_DIT never orphans polled AR cmds."""
    pol = EnginePolicyConfig.from_preset("morphserve", num_gpus=2, model_family="synthetic")

    class _FakeCudaBuf:
        is_cuda = False

        def __init__(self, n: int) -> None:
            self.n = n

    seen_ready_events: list[Any] = []

    class _MockMesh:
        def alloc_recv_u8(self, nbytes: int) -> Any:
            return bytearray(nbytes)

        def recv_u8(
            self,
            nbytes: int,
            src_rank: int,
            sync_stream: bool = True,
            *,
            out_tensor: Any = None,
            ready_event: Any = None,
            cuda_stream: int | None = None,
        ) -> tuple[Any, float]:
            seen_ready_events.append((out_tensor, ready_event))
            return b"x" * nbytes, 0.1

    class _OrphaningAdapter(SyntheticCPTGAdapter):
        def generate_dit(self, ctx_id: str, cmd: dict[str, Any], poll_preempt_cmds=None, emit_preempt_ack=None):
            if poll_preempt_cmds is not None:
                _ = poll_preempt_cmds()  # Polls cmd but intentionally returns without ACKing it!
            return super().generate_dit(ctx_id, cmd, poll_preempt_cmds=None, emit_preempt_ack=None)

    acks: list[dict[str, Any]] = []
    adapter = _OrphaningAdapter(rank=0, policy=pol)
    rt = WorkerRuntime(rank=0, policy=pol, adapter=adapter, peer_mesh=_MockMesh(), ack_sink=acks.append)
    try:
        # 1. ALLOC_RECV_BUF -> RECV_NCCL unpacks `(buf, alloc_ev)` and forwards `ready_event` into `recv_u8`.
        a_ack = rt._dispatch_xfer_cmd({"cmd_id": "a1", "op": "ALLOC_RECV_BUF", "nbytes": 16})
        assert a_ack is not None and a_ack["status"] == "ok"
        r_ack = rt._dispatch_xfer_cmd(
            {
                "cmd_id": "r1",
                "op": "RECV_NCCL",
                "src_rank": 1,
                "recv_id": a_ack["recv_id"],
                "meta": {"total_bytes": 16, "logical_bytes": 16},
            }
        )
        assert r_ack is not None and r_ack["status"] == "ok"
        assert len(seen_ready_events) == 1
        assert isinstance(seen_ready_events[0][0], bytearray)

        # 2. If `generate_dit` polls a preempted AR command and returns normally without ACKing it,
        # WorkerRuntime defensively re-enqueues it onto `ar_preempt_q` so coordinator never hangs 600s!
        rt.handle_command({"cmd_id": "p0", "op": "PREP_DIT", "ctx_id": "c_dit", "prompt": "draw", "height": 512, "width": 512})
        rt.enqueue_command({"cmd_id": "ar_orphan", "op": "EXEC_AR_PREEMPT", "ctx_id": "c_ar", "prompt": "hi", "max_ar_tokens": 2})
        dit_ack = rt.handle_command({"cmd_id": "d0", "op": "EXEC_DIT", "ctx_id": "c_dit", "prompt": "draw", "height": 512, "width": 512, "dit_steps": 1})
        assert dit_ack is not None and dit_ack["status"] == "ok"
        requeued = rt._poll_preempt_cmds()
        assert [m["cmd_id"] for m in requeued] == ["ar_orphan"]
    finally:
        rt.handle_command({"cmd_id": "sh", "op": "SHUTDOWN"})


def test_cluster_engine_concurrent_ar_waiters_no_livelock_and_batch_before_dit(tmp_path: Path) -> None:
    """Regression test for direct-handoff `MemorySlackGpuLock`: multiple queued `EXEC_AR` turns must
    wake atomically without livelocking, batch concurrently, and drain before the next exclusive DiT."""
    pol = EnginePolicyConfig.from_preset(
        "morphserve",
        num_gpus=1,
        model_family="synthetic",
        preemption_mode="non_preemptive",
        gpu_queue_mode="memory_slack",
    )

    fleet = InProcessWorkerFleet(pol)
    eng = MorphServeClusterEngine(policy=pol, fleet=fleet, tmp_root=tmp_path / "ar_livelock")
    eng.start()

    async def _run() -> tuple[list[dict[str, Any]], int, bool]:
        eng._ensure_async_primitives()
        # Hold GPU 0 lock exclusively as if a DiT is in progress.
        await eng._gpu_locks[0].acquire(est_mem_bytes=0, est_exec_ms=100.0, slo_deadline_ms=0.0, op_priority=1)
        turn = InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="hello", max_ar_tokens=2)

        ar_tasks = [
            asyncio.create_task(
                eng._execute_ar_phase(pol, 0, f"ar_ctx_{i}", turn, prepare_for_dit=False)
            )
            for i in range(3)
        ]
        # Wait until all 3 AR tasks are queued behind GPU 0 lock, then also queue an exclusive DiT waiter.
        await asyncio.sleep(0.02)
        assert eng._gpu_locks[0].has_waiters()

        dit_entered_while_ar_active = {"bad": False}

        async def _dit_waiter() -> None:
            await eng._acquire_exclusive_gpu(0, est_mem_bytes=0, est_exec_ms=100.0, slo_deadline_ms=0.0, op_priority=1)
            try:
                if eng._inflight_ar_on_gpu.get(0, 0) > 0:
                    dit_entered_while_ar_active["bad"] = True
            finally:
                eng._gpu_locks[0].release()

        dit_task = asyncio.create_task(_dit_waiter())
        await asyncio.sleep(0.01)

        # Release initial lock: all 3 AR tasks must complete cleanly with zero livelock!
        eng._gpu_locks[0].release()
        res_list = await asyncio.wait_for(asyncio.gather(*ar_tasks), timeout=5.0)
        await asyncio.wait_for(dit_task, timeout=5.0)
        return [r[0] for r in res_list], eng._inflight_ar_on_gpu.get(0, 0), dit_entered_while_ar_active["bad"]

    try:
        outs, inflight_after, bad_overlap = asyncio.run(_run())
    finally:
        eng.shutdown()
    assert len(outs) == 3
    assert all(o.get("status") == "ok" for o in outs)
    assert inflight_after == 0
    assert bad_overlap is False


def test_cluster_engine_mid_dit_preemption_gate_for_internvlu_regimes() -> None:
    """Mid-DiT preemption is active for BAGEL and InternVL-U Regime A, and disabled in quota-capped Regime B."""
    pol_bagel = EnginePolicyConfig.from_preset("morphserve", num_gpus=2, model_family="bagel")
    pol_ivlu_a = EnginePolicyConfig.from_preset("morphserve", num_gpus=2, model_family="internvl_u")
    pol_ivlu_b_quota = EnginePolicyConfig.from_preset(
        "morphserve", num_gpus=2, model_family="internvl_u", gpu_memory_quota_mib=12288.0
    )
    pol_ivlu_b_frac = EnginePolicyConfig.from_preset(
        "morphserve", num_gpus=2, model_family="internvl_u", gpu_memory_fraction=0.30
    )
    pol_nonpre = EnginePolicyConfig.from_preset(
        "morphserve", num_gpus=2, model_family="internvl_u", preemption_mode="non_preemptive"
    )
    assert MorphServeClusterEngine._supports_mid_dit_preemption(pol_bagel) is True
    assert MorphServeClusterEngine._supports_mid_dit_preemption(pol_ivlu_a) is True
    assert MorphServeClusterEngine._supports_mid_dit_preemption(pol_ivlu_b_quota) is False
    assert MorphServeClusterEngine._supports_mid_dit_preemption(pol_ivlu_b_frac) is False
    assert MorphServeClusterEngine._supports_mid_dit_preemption(pol_nonpre) is False


def test_nccl_transfer_guard_reclaims_dst_headroom_after_probe_oom(tmp_path: Path) -> None:
    """When `ALLOC_RECV_BUF(probe_only=True)` OOMs, `_nccl_transfer_guard` locks dst, runs `RECLAIM_HEADROOM`, and succeeds."""
    pol = EnginePolicyConfig.from_preset(
        "morphserve",
        num_gpus=2,
        model_family="synthetic",
        inter_turn_think_s_range=(0.0, 0.0),
    )
    fleet = InProcessWorkerFleet(pol)
    eng = MorphServeClusterEngine(
        policy=pol,
        fleet=fleet,
        tmp_root=tmp_path / "tmp_nccl_reclaim",
    )
    eng.start()
    ops_on_dst: list[tuple[str, bool]] = []
    reclaimed = {"done": False}
    orig_rpc = eng._rpc

    async def _intercept_rpc(rank: int, payload: dict[str, Any]) -> dict[str, Any]:
        op = str(payload.get("op", ""))
        if rank == 1:
            ops_on_dst.append((op, eng._gpu_locks[1].locked()))
            if op == "ALLOC_RECV_BUF" and payload.get("probe_only", False) and not reclaimed["done"]:
                return {"status": "oom", "probe_only": True, "error": "injected probe oom"}
            if op == "RECLAIM_HEADROOM":
                assert eng._gpu_locks[1].locked() is True
                reclaimed["done"] = True
        return await orig_rpc(rank, payload)

    eng._rpc = _intercept_rpc  # type: ignore[method-assign]

    async def _run() -> dict[str, Any]:
        eng._ensure_async_primitives()
        # Seed a context on rank 0 so PACK_CTX has state to ship.
        await orig_rpc(
            0,
            {
                "op": "EXEC_AR",
                "ctx_id": "ctx_reclaim",
                "prompt": "seed context",
                "image_paths": [],
                "use_prev_image": False,
                "max_ar_tokens": 2,
                "prepare_for_dit": False,
            },
        )
        async with eng._nccl_transfer_guard(
            0,
            1,
            "ctx_reclaim",
            "ctx_reclaim_dst",
            include_cfg_dup=False,
            release_src_ctx=True,
        ) as ship:
            assert eng._gpu_locks[1].locked() is True
            return dict(ship)

    try:
        ship_res = asyncio.run(_run())
    finally:
        eng.shutdown()

    assert ship_res.get("status") == "ok"
    assert reclaimed["done"] is True
    dst_op_names = [op for op, _ in ops_on_dst]
    assert "RECLAIM_HEADROOM" in dst_op_names
    idx_reclaim = dst_op_names.index("RECLAIM_HEADROOM")
    assert dst_op_names[idx_reclaim - 1] == "ALLOC_RECV_BUF"
    assert dst_op_names[idx_reclaim + 1] == "ALLOC_RECV_BUF"
    assert ops_on_dst[idx_reclaim][1] is True


def test_nccl_dual_oom_preserves_src_ctx_and_falls_back_to_in_place_path_b(tmp_path: Path) -> None:
    """When `nccl_dual` Path-A hits `PACK_CTX` or `ALLOC_RECV_BUF` OOM, `src_ctx` stays intact and falls back to Path B."""
    for fail_mode in ("pack_oom", "recv_oom"):
        pol = EnginePolicyConfig.from_preset(
            "morphserve",
            num_gpus=2,
            model_family="synthetic",
            inter_turn_think_s_range=(0.0, 0.0),
        )
        fleet = InProcessWorkerFleet(pol)
        eng = MorphServeClusterEngine(
            policy=pol,
            fleet=fleet,
            tmp_root=tmp_path / f"tmp_nccl_fb_{fail_mode}",
        )
        eng.start()
        seen_ops: list[tuple[int, str, dict[str, Any]]] = []
        orig_rpc = eng._rpc
        orig_plan = eng._plan_turn_transition

        def _force_nccl_dual(ctrl_handle: int, **kwargs: Any) -> dict[str, Any]:
            res = orig_plan(ctrl_handle, **kwargs)
            src_g = int(kwargs["src_gpu"])
            other_g = 1 - src_g
            res["ship_mode"] = "nccl_dual"
            res["dst_gpu"] = other_g
            res["reserved_kv_bytes"] = 0
            res["release_src_on_fwd"] = True
            res["ephemeral_gen_ctx"] = False
            res["needs_return_ship"] = False
            res["dit_ctx_id"] = str(kwargs["base_ctx_id"])
            return res

        async def _intercept_rpc(rank: int, payload: dict[str, Any]) -> dict[str, Any]:
            op = str(payload.get("op", ""))
            seen_ops.append((rank, op, dict(payload)))
            if fail_mode == "pack_oom" and op == "PACK_CTX":
                return {"status": "oom", "error": "injected pack_ctx oom"}
            if fail_mode == "recv_oom" and op == "ALLOC_RECV_BUF":
                return {"status": "oom", "probe_only": bool(payload.get("probe_only", False)), "error": "injected recv oom"}
            return await orig_rpc(rank, payload)

        eng._plan_turn_transition = _force_nccl_dual  # type: ignore[method-assign]
        eng._rpc = _intercept_rpc  # type: ignore[method-assign]
        try:
            suite = WorkloadSuite(
                name=f"fb_{fail_mode}",
                trace_source="unit",
                base_qps=1.0,
                sessions=[
                    InterleavedSession(
                        session_id=f"s_fb_{fail_mode}",
                        quadrant="interleaved_co_creation",
                        source_dataset="unit",
                        arrival_offset_s=0.0,
                        turns=[
                            InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="plan scene", max_ar_tokens=2),
                            InterleavedTurn(
                                turn_idx=1,
                                modality_plan="dit_t2i",
                                prompt="render scene",
                                height=512,
                                width=512,
                                dit_steps=2,
                            ),
                        ],
                    )
                ],
            )
            rep = eng.run_suite(suite, target_qps=1.0)
        finally:
            eng.shutdown()

        assert rep.completed_sessions == 1
        assert rep.oom_sessions == 0
        assert rep.error_sessions == 0
        assert rep.path_a_ships == 0
        assert rep.path_b_morphs == 1
        pack_calls = [p for _, op, p in seen_ops if op == "PACK_CTX"]
        assert pack_calls and pack_calls[0].get("release_after_pack") is False
        if fail_mode == "recv_oom":
            assert any(op == "DROP_PACKED_CTX" for _, op, _ in seen_ops)
            assert any(op == "RECLAIM_HEADROOM" for _, op, _ in seen_ops)


def test_bagel_chunked_stream_views_bit_exact_across_odd_chunk_boundaries() -> None:
    """Verify zero-OOM chunked BAGEL KV view scatter/gather matches pack/unpack bit-exactly across small odd chunks."""
    torch = pytest.importorskip("torch")
    from morphserve.models.bagel_adapter import (
        BagelSessionContext,
        collect_bagel_stream_views,
        copy_chunk_to_views,
        copy_views_to_chunk,
        finalize_bagel_stream_recv,
        prealloc_bagel_stream_recv,
    )

    class _SimpleCache:
        def __init__(self, n_layers: int) -> None:
            self.num_layers = n_layers
            self.key_cache: dict[int, Any] = {i: None for i in range(n_layers)}
            self.value_cache: dict[int, Any] = {i: None for i in range(n_layers)}

    torch.manual_seed(42)
    n_layers = 4
    main_pkv = _SimpleCache(n_layers)
    cfg_img_pkv = _SimpleCache(n_layers)
    for lidx in range(n_layers):
        main_pkv.key_cache[lidx] = torch.randn((19, 4, 32), dtype=torch.bfloat16)
        main_pkv.value_cache[lidx] = torch.randn((19, 4, 32), dtype=torch.bfloat16)
        cfg_img_pkv.key_cache[lidx] = torch.randn((7, 4, 32), dtype=torch.bfloat16)
        cfg_img_pkv.value_cache[lidx] = torch.randn((7, 4, 32), dtype=torch.bfloat16)

    last_u8 = torch.randint(0, 256, (3, 17, 23), dtype=torch.uint8)
    sess = BagelSessionContext(
        main_ctx={"kv_lens": [19], "ropes": [19], "past_key_values": main_pkv},
        cfg_img_ctx={"kv_lens": [7], "ropes": [7], "past_key_values": cfg_img_pkv},
        cfg_text_prefix_len=11,
        last_img_u8=last_u8,
    )
    sess.cfg_text_ctx = {
        "kv_lens": [11],
        "ropes": [11],
        "past_key_values": _SimpleCache(n_layers),
    }
    for lidx in range(n_layers):
        sess.cfg_text_ctx["past_key_values"].key_cache[lidx] = main_pkv.key_cache[lidx][:11]
        sess.cfg_text_ctx["past_key_values"].value_cache[lidx] = main_pkv.value_cache[lidx][:11]

    meta, src_views = collect_bagel_stream_views(sess, include_cfg_text_dup=False, device=torch.device("cpu"))
    total_b = int(meta["total_bytes"])
    assert total_b > 4096

    partial, dst_views = prealloc_bagel_stream_recv(meta, _SimpleCache, _SimpleCache, device=torch.device("cpu"))
    chunk_b = 1537  # Intentionally odd non-aligned chunk size straddling multiple K/V tensors!
    stage = torch.empty((chunk_b,), dtype=torch.uint8)
    off = 0
    while off < total_b:
        n = min(chunk_b, total_b - off)
        assert copy_views_to_chunk(src_views, off, stage[:n]) == n
        assert copy_chunk_to_views(dst_views, off, stage[:n]) == n
        off += n

    restored = finalize_bagel_stream_recv(partial, meta, _SimpleCache, use_cow_rebind=True)
    assert restored.main_ctx is not None and restored.cfg_img_ctx is not None and restored.cfg_text_ctx is not None
    assert restored.last_img_u8 is not None and torch.equal(restored.last_img_u8, last_u8)
    for lidx in range(n_layers):
        assert torch.equal(restored.main_ctx["past_key_values"].key_cache[lidx], main_pkv.key_cache[lidx])
        assert torch.equal(restored.main_ctx["past_key_values"].value_cache[lidx], main_pkv.value_cache[lidx])
        assert torch.equal(restored.cfg_img_ctx["past_key_values"].key_cache[lidx], cfg_img_pkv.key_cache[lidx])
        assert torch.equal(restored.cfg_img_ctx["past_key_values"].value_cache[lidx], cfg_img_pkv.value_cache[lidx])
        assert torch.equal(restored.cfg_text_ctx["past_key_values"].key_cache[lidx], main_pkv.key_cache[lidx][:11])


def test_bridge_mllm_dit_quota_constrained_decouples_vlm_from_decoder_ranks(tmp_path: Path) -> None:
    """Under quota-constrained `bridge_mllm_dit`, Turn > 0 VLM turns re-route to VLM-hot ranks with exact `vlm_history`."""
    pol = EnginePolicyConfig.from_preset(
        "morphserve",
        num_gpus=2,
        model_family="synthetic",
        cptg_arch_override="bridge_mllm_dit",
        gpu_memory_quota_mib=12288.0,
        inter_turn_think_s_range=(0.0, 0.0),
    )
    eng = MorphServeClusterEngine(
        pol,
        tmp_root=tmp_path / "bridge_decouple",
        fleet=InProcessWorkerFleet(pol),
    )
    eng.start()
    eng.router.set_worker_state_fields(0, expert_alpha=0.0, kv_free_bytes=8 * 1024**3)
    eng.router.set_worker_state_fields(1, expert_alpha=1.0, kv_free_bytes=8 * 1024**3)

    vlm_ops: list[tuple[int, str, Any]] = []
    orig_rpc = eng._rpc
    orig_sync = eng._sync_worker_telemetry

    def _pinned_sync(rank: int, ack: dict[str, Any]) -> None:
        patched = dict(ack)
        if "resident_alpha" in patched:
            patched["resident_alpha"] = 0.0 if int(rank) == 0 else 1.0
        orig_sync(rank, patched)

    async def _intercept_rpc(rank: int, payload: dict[str, Any]) -> dict[str, Any]:
        op = str(payload.get("op", ""))
        if op in ("EXEC_AR", "PREP_DIT"):
            vlm_ops.append((int(rank), op, payload.get("vlm_history")))
        return await orig_rpc(rank, payload)

    eng._sync_worker_telemetry = _pinned_sync  # type: ignore[method-assign]
    eng._rpc = _intercept_rpc  # type: ignore[method-assign]
    try:
        suite = WorkloadSuite(
            name="bridge_decouple_2turn",
            trace_source="unit",
            base_qps=1.0,
            sessions=[
                InterleavedSession(
                    session_id="s_bridge_decouple",
                    quadrant="interleaved_co_creation",
                    source_dataset="unit",
                    arrival_offset_s=0.0,
                    turns=[
                        InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="describe apple", max_ar_tokens=2),
                        InterleavedTurn(
                            turn_idx=1,
                            modality_plan="dit_t2i",
                            prompt="draw red apple",
                            height=512,
                            width=512,
                            dit_steps=2,
                        ),
                        InterleavedTurn(turn_idx=2, modality_plan="ar_only", prompt="review image", max_ar_tokens=2),
                    ],
                )
            ],
        )
        rep = eng.run_suite(suite, target_qps=1.0)
    finally:
        eng.shutdown()

    assert rep.completed_sessions == 1
    assert rep.oom_sessions == 0
    assert rep.error_sessions == 0
    # Every VLM op (`EXEC_AR`, `PREP_DIT`) MUST execute on VLM-hot rank 0 (never thrash Decoder-hot rank 1)!
    assert len(vlm_ops) == 3
    assert all(rank == 0 for rank, _, _ in vlm_ops)
    # Turn 2 (`EXEC_AR` after Turn 1 shipped to rank 1) MUST carry Turn 0's exact `vlm_history`!
    assert vlm_ops[2][2] is not None and len(vlm_ops[2][2]) == 1 and vlm_ops[2][2][0][0] == "describe apple"


def test_chunked_nccl_stream_acquires_dst_lock_before_pinning_src_ctx(tmp_path: Path) -> None:
    """Chunked NCCL streaming must acquire `_gpu_locks[dst]` BEFORE sending `PIN_STREAM_CTX` to `src`."""
    pol = EnginePolicyConfig.from_preset(
        "morphserve",
        num_gpus=2,
        model_family="synthetic",
        inter_turn_think_s_range=(0.0, 0.0),
    )
    eng = MorphServeClusterEngine(
        pol,
        tmp_root=tmp_path / "chunked_dst_before_pin",
        fleet=InProcessWorkerFleet(pol),
    )
    eng._force_chunked_stream_for_test = True  # type: ignore[attr-defined]
    eng.start()
    pin_seen_with_dst_locked: list[bool] = []
    orig_rpc = eng._rpc

    async def _intercept_rpc(rank: int, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("op") == "PIN_STREAM_CTX":
            pin_seen_with_dst_locked.append(bool(eng._gpu_locks[1].locked()))
        return await orig_rpc(rank, payload)

    eng._rpc = _intercept_rpc  # type: ignore[method-assign]

    async def _run() -> dict[str, Any]:
        eng._ensure_async_primitives()
        await orig_rpc(
            0,
            {
                "op": "EXEC_AR",
                "ctx_id": "ctx_pin_order",
                "prompt": "seed",
                "image_paths": [],
                "use_prev_image": False,
                "max_ar_tokens": 2,
                "prepare_for_dit": False,
            },
        )
        # Hold dst lock briefly so `_nccl_transfer_guard` must queue on dst BEFORE pinning src.
        await eng._gpu_locks[1].acquire(est_mem_bytes=0, est_exec_ms=10.0, slo_deadline_ms=0.0, op_priority=1)

        async def _release_soon() -> None:
            await asyncio.sleep(0.03)
            # While dst lock is held by another DiT turn, src MUST NOT have received PIN_STREAM_CTX yet!
            assert pin_seen_with_dst_locked == []
            eng._gpu_locks[1].release()

        rel_task = asyncio.create_task(_release_soon())
        async with eng._nccl_transfer_guard(
            0,
            1,
            "ctx_pin_order",
            "ctx_pin_order_dst",
            include_cfg_dup=False,
            est_state_bytes=128 * 1024 * 1024,
        ) as ship:
            out = dict(ship)
        await rel_task
        return out

    try:
        res = asyncio.run(_run())
    finally:
        eng.shutdown()

    assert res.get("status") == "ok"
    assert pin_seen_with_dst_locked == [True]
