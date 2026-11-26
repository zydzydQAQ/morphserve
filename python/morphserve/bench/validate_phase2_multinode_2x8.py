"""Phase-2 Milestone M2.5: Live 2x8 (16-GPU) Cross-Node Fabric Calibration & Distributed E2E Serving."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import socket
import struct
import time
from typing import Any, Callable, Sequence

from morphserve.bench.harness import (
    BenchmarkRunner,
    SuiteRunSpec,
    parse_csv_strings,
    resolve_default_main_suite_path,
    resolve_sources_dir,
)
from morphserve.bench.interleaved_bench import (
    InterleavedSession,
    InterleavedTurn,
    POST_DIT_INSPECT_PROMPT,
    WorkloadSuite,
    load_burstgpt_interarrivals,
    load_gentd26_interarrivals,
)
from morphserve.engine import (
    EnginePolicyConfig,
    MultinodeCoordinatorHub,
    TcpMultinodeWorkerFleet,
    run_multinode_worker_daemon,
)
from morphserve.engine.workload import build_time_window_suite
from morphserve.router import (
    CPTGRequestSpec,
    ClusterTopologyProfile,
    DualPathMorphRouter,
    PcieHalfGroup,
    WorkerState,
)
from morphserve.runtime.peer_transport import DirectNcclMesh
from morphserve.vmm import TriPoolVMM

MIB = 1024**2
GIB = 1024**3


def _recv_exact(conn: socket.socket, nbytes: int) -> bytes:
    """Read exactly `nbytes` from a TCP stream, raising `ConnectionError` on premature EOF."""
    target = int(nbytes)
    if target < 0:
        raise ValueError(f"negative byte length: {target}")
    buf = bytearray()
    while len(buf) < target:
        chunk = conn.recv(target - len(buf))
        if not chunk:
            raise ConnectionError(f"unexpected EOF: expected {target} bytes, received {len(buf)}")
        buf.extend(chunk)
    return bytes(buf)


def _exchange_bytes_tcp(is_master: bool, master_addr: str, master_port: int, payload: bytes = b"") -> bytes:
    """Broadcast `payload` from node rank 0 to node rank 1 over `$MASTER_ADDR:$MASTER_PORT`."""
    if is_master:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.settimeout(180.0)
        srv.bind(("0.0.0.0", master_port))
        srv.listen(4)
        try:
            conn, _ = srv.accept()
            with conn:
                conn.sendall(struct.pack("!I", len(payload)) + payload)
        finally:
            srv.close()
        return payload

    deadline = time.monotonic() + 180.0
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            cli = socket.create_connection((master_addr, master_port), timeout=5.0)
            with cli:
                (nbytes,) = struct.unpack("!I", _recv_exact(cli, 4))
                return _recv_exact(cli, nbytes)
        except Exception as exc:
            last_err = exc
            time.sleep(1.0)
    raise TimeoutError(f"failed to connect to {master_addr}:{master_port}: {last_err}")


def _send_json_to_master(master_addr: str, port: int, data: dict[str, Any]) -> None:
    raw = json.dumps(data).encode("utf-8")
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((master_addr, port), timeout=5.0) as cli:
                cli.sendall(struct.pack("!I", len(raw)) + raw)
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError("failed to send worker summary to master")


def _recv_json_on_master(port: int) -> dict[str, Any]:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.settimeout(180.0)
    srv.bind(("0.0.0.0", port))
    srv.listen(2)
    try:
        conn, _ = srv.accept()
        with conn:
            (nbytes,) = struct.unpack("!I", _recv_exact(conn, 4))
            payload = _recv_exact(conn, nbytes)
            return json.loads(payload.decode("utf-8"))
    finally:
        srv.close()


def _local_gpu_worker(
    local_rank: int,
    node_rank: int,
    gpus_per_node: int,
    world_size: int,
    uid_bytes: bytes,
    res_q: Any,
) -> None:
    import torch

    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")
    global_rank = node_rank * gpus_per_node + local_rank

    vmm = TriPoolVMM(
        device=local_rank,
        super_pool_mib=256,
        fine_kv_pool_mib=128,
        super_handle_mib=64,
        transient_va_gib=2,
        kv_va_gib=2,
    )
    mesh = DirectNcclMesh(
        rank=global_rank,
        world_size=world_size,
        uid_bytes=uid_bytes,
        device_index=local_rank,
    )
    try:
        next_r = (global_rank + 1) % world_size
        prev_r = (global_rank - 1) % world_size
        w_buf = torch.ones((4096,), dtype=torch.uint8, device=dev)
        if global_rank == 0:
            mesh.send_u8(w_buf, next_r)
            mesh.recv_u8(4096, prev_r)
        else:
            mesh.recv_u8(4096, prev_r)
            mesh.send_u8(w_buf, next_r)
        torch.cuda.synchronize(dev)

        measurements: list[dict[str, Any]] = []
        pairs = (
            ("intra_half_pcie", 0, 1),
            ("intra_node_cross_socket", 0, 4 if world_size > 4 else 1),
            ("inter_node_fabric", 0, gpus_per_node if world_size > gpus_per_node else 1),
        )
        sizes_mib = (24, 192, 1024)

        for tier_name, src_r, dst_r in pairs:
            for sz_mib in sizes_mib:
                nbytes = sz_mib * MIB
                iters = 5 if sz_mib <= 192 else 3
                if global_rank == src_r:
                    t_src = torch.randint(0, 255, (nbytes,), dtype=torch.uint8, device=dev)
                    _ = mesh.send_u8(t_src, dst_r)
                    times = [mesh.send_u8(t_src, dst_r) for _ in range(iters)]
                    med_ms = sorted(times)[len(times) // 2]
                    gbps = (nbytes / 1e9) / max(1e-6, med_ms / 1000.0)
                    measurements.append(
                        {
                            "tier": tier_name,
                            "src_global_rank": src_r,
                            "dst_global_rank": dst_r,
                            "payload_mib": sz_mib,
                            "ship_median_ms": round(med_ms, 2),
                            "effective_gbps": round(gbps, 2),
                        }
                    )
                    del t_src
                elif global_rank == dst_r:
                    _, _ = mesh.recv_u8(nbytes, src_r)
                    for _ in range(iters):
                        r_buf, _ = mesh.recv_u8(nbytes, src_r)
                        del r_buf
                torch.cuda.synchronize(dev)

        res_q.put(
            {
                "global_rank": global_rank,
                "local_rank": local_rank,
                "gpu_name": torch.cuda.get_device_name(local_rank),
                "vmm_ok": vmm.stats_dict().get("overflow_creates", 0) == 0,
                "measurements": measurements,
            }
        )
    finally:
        mesh.close()
        vmm.shutdown()


def evaluate_cross_node_routing_check(
    world_size: int,
    gpus_per_node: int,
    intra_bw: float,
    socket_bw: float,
    inter_bw: float,
) -> dict[str, Any]:
    """Build 16-GPU 2-node topology with measured fabric bandwidths & verify Dual-Path routing."""
    groups = [
        PcieHalfGroup(
            group_id=gi,
            numa_node=gi % 2,
            gpu_ids=tuple(range(gi * 4, (gi + 1) * 4)),
            aggregate_h2d_gbps=24.5,
            solo_h2d_gbps=22.9,
        )
        for gi in range(world_size // 4)
    ]
    bw_mat: list[list[float]] = []
    for s in range(world_size):
        row: list[float] = []
        for d in range(world_size):
            if s == d:
                row.append(900.0)
            elif (s // 4) == (d // 4):
                row.append(intra_bw)
            elif (s // gpus_per_node) == (d // gpus_per_node):
                row.append(socket_bw)
            else:
                row.append(inter_bw)
        bw_mat.append(row)

    topo16 = ClusterTopologyProfile(
        num_gpus=world_size,
        gpu_names=["NVIDIA A100-PCIE-40GB"] * world_size,
        half_groups=groups,
        p2p_bw_gbps=bw_mat,
        intra_half_p2p_cap_gbps=max(1.0, intra_bw),
        cross_half_p2p_cap_gbps=max(1.0, socket_bw),
    )
    router16 = DualPathMorphRouter(topo16)

    states = {
        g: WorkerState(
            gpu_id=g,
            role="hybrid" if g == 0 else "gen",
            expert_alpha=0.0 if g == 0 else 1.0,
            queue_wait_ms=0.0 if g in (0, gpus_per_node) else 4500.0,
            kv_free_bytes=20 * GIB,
        )
        for g in range(world_size)
    }
    dec_bridge = router16.decide(
        CPTGRequestSpec(
            req_id="internvlu_bridge_24m",
            src_gpu=0,
            architecture="bridge_mllm_dit",
            bridge_bytes=24 * MIB,
            kv_bytes=24 * MIB,
            dit_step1_ms=190.0,
            dit_total_steps=20,
            resolution=1024,
        ),
        states,
    )
    dec_bagel_1g = router16.decide(
        CPTGRequestSpec(
            req_id="bagel_multiturn_1536m",
            src_gpu=0,
            architecture="tight_mot",
            kv_bytes=1536 * MIB,
            dit_step1_ms=287.0,
            dit_total_steps=20,
            resolution=1024,
            interleaved_roundtrip_gamma=2,
        ),
        states,
    )
    return {
        "bridge_24mib_decision": {
            "path": dec_bridge.path,
            "dst_gpu": dec_bridge.dst_gpu,
            "ship_cost_ms": round(dec_bridge.t_ship_ms, 2),
            "morph_cost_ms": round(dec_bridge.t_morph_ms, 2),
        },
        "tight_mot_1536mib_decision": {
            "path": dec_bagel_1g.path,
            "dst_gpu": dec_bagel_1g.dst_gpu,
            "ship_cost_ms": round(dec_bagel_1g.t_ship_ms, 2),
            "morph_cost_ms": round(dec_bagel_1g.t_morph_ms, 2),
        },
        "suppressed_cross_node_1gib_kv_ship": dec_bagel_1g.path == "in_place_morph",
        "routing_pass": (dec_bridge.path == "state_ship") and (dec_bagel_1g.path == "in_place_morph"),
    }


def run_multinode_spotcheck(output_dir: Path, gpus_per_node: int = 8) -> dict[str, Any]:
    import torch.multiprocessing as mp

    node_rank = int(os.environ.get("RANK", "0"))
    num_nodes = int(os.environ.get("WORLD_SIZE", "2"))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))
    world_size = num_nodes * gpus_per_node

    is_master = node_rank == 0
    raw_uid = DirectNcclMesh.generate_unique_id_bytes() if is_master else b""
    uid_bytes = _exchange_bytes_tcp(is_master, master_addr, master_port + 11, raw_uid)

    ctx = mp.get_context("spawn")
    res_q = ctx.Queue()
    procs = []
    for lr in range(gpus_per_node):
        p = ctx.Process(
            target=_local_gpu_worker,
            args=(lr, node_rank, gpus_per_node, world_size, uid_bytes, res_q),
        )
        p.start()
        procs.append(p)

    local_summaries = [res_q.get(timeout=240.0) for _ in range(gpus_per_node)]
    for p in procs:
        p.join(timeout=30.0)
    local_summaries.sort(key=lambda x: x["global_rank"])

    if not is_master:
        _send_json_to_master(master_addr, master_port + 12, {"node_rank": node_rank, "ranks": local_summaries})
        return {"status": "ok", "node_rank": node_rank}

    output_dir.mkdir(parents=True, exist_ok=True)
    all_ranks = list(local_summaries)
    if num_nodes > 1:
        peer = _recv_json_on_master(master_port + 12)
        all_ranks.extend(peer["ranks"])
    all_ranks.sort(key=lambda x: x["global_rank"])

    r0_meas = next((r["measurements"] for r in all_ranks if r["global_rank"] == 0), [])
    by_tier_1g = {m["tier"]: m for m in r0_meas if m["payload_mib"] == 1024}
    intra_bw = by_tier_1g.get("intra_half_pcie", {}).get("effective_gbps", 24.0)
    socket_bw = by_tier_1g.get("intra_node_cross_socket", {}).get("effective_gbps", 18.0)
    inter_bw = by_tier_1g.get("inter_node_fabric", {}).get("effective_gbps", 11.0)

    routing_chk = evaluate_cross_node_routing_check(
        world_size=world_size,
        gpus_per_node=gpus_per_node,
        intra_bw=intra_bw,
        socket_bw=socket_bw,
        inter_bw=inter_bw,
    )
    all_vmm_ok = all(r["vmm_ok"] for r in all_ranks)
    routing_pass = bool(routing_chk["routing_pass"])
    report = {
        "schema": "morphserve.phase2.m25_multinode_2x8.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_nodes": num_nodes,
        "gpus_per_node": gpus_per_node,
        "world_size": world_size,
        "fabric_measurements": r0_meas,
        "calibrated_gbps": {
            "intra_half_pcie": intra_bw,
            "intra_node_cross_socket": socket_bw,
            "inter_node_fabric": inter_bw,
        },
        "cross_node_routing_verification": routing_chk,
        "ranks_verified": len(all_ranks),
        "status": "ok" if (len(all_ranks) == world_size and all_vmm_ok and routing_pass) else "check_failed",
    }
    out_path = output_dir / "m25-multinode-2x8-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return report


def _build_synthetic_multinode_suite(num_sessions: int, offered_qps: float) -> WorkloadSuite:
    """Build v2 4-quadrant multi-turn synthetic suite (including Q1 bidirectional `interleaved_co_creation`)."""
    sessions: list[InterleavedSession] = []
    dt = 1.0 / max(0.05, float(offered_qps))
    for i in range(max(1, int(num_sessions))):
        off = round((i // 2) * dt * 1.6 + (i % 2) * 0.01, 4)
        mod = i % 4
        if mod == 0:
            sessions.append(
                InterleavedSession(
                    session_id=f"mn_cocreate_{i:03d}",
                    quadrant="interleaved_co_creation",
                    source_dataset="synthetic_2x8_v2",
                    arrival_offset_s=off,
                    turns=[
                        InterleavedTurn(
                            turn_idx=0,
                            modality_plan="ar_only",
                            prompt="plan visual scene layout and lighting",
                            max_ar_tokens=6,
                            use_prev_image=False,
                        ),
                        InterleavedTurn(
                            turn_idx=1,
                            modality_plan="dit_t2i",
                            prompt="minimalist coastal courtyard at dusk",
                            height=512,
                            width=512,
                            dit_steps=3,
                            use_prev_image=False,
                        ),
                        InterleavedTurn(
                            turn_idx=2,
                            modality_plan="ar_only",
                            prompt=POST_DIT_INSPECT_PROMPT,
                            max_ar_tokens=6,
                            use_prev_image=True,
                        ),
                        InterleavedTurn(
                            turn_idx=3,
                            modality_plan="dit_edit",
                            prompt="warm sunset reflection along the stone arches",
                            height=512,
                            width=512,
                            dit_steps=3,
                            use_prev_image=True,
                        ),
                    ],
                )
            )
        elif mod == 1:
            sessions.append(
                InterleavedSession(
                    session_id=f"mn_edit_{i:03d}",
                    quadrant="multiturn_edit",
                    source_dataset="synthetic_2x8_v2",
                    arrival_offset_s=off,
                    turns=[
                        InterleavedTurn(
                            turn_idx=0,
                            modality_plan="dit_t2i",
                            prompt="coastal town",
                            dit_steps=3,
                            use_prev_image=False,
                        ),
                        InterleavedTurn(
                            turn_idx=1,
                            modality_plan="dit_edit",
                            prompt="sunset sky",
                            dit_steps=3,
                            use_prev_image=True,
                        ),
                        InterleavedTurn(
                            turn_idx=2,
                            modality_plan="dit_edit",
                            prompt="harbor lights",
                            dit_steps=3,
                            use_prev_image=True,
                        ),
                        InterleavedTurn(
                            turn_idx=3,
                            modality_plan="dit_edit",
                            prompt="gentle evening fog",
                            dit_steps=3,
                            use_prev_image=True,
                        ),
                    ],
                )
            )
        elif mod == 2:
            sessions.append(
                InterleavedSession(
                    session_id=f"mn_vqa_{i:03d}",
                    quadrant="multi_image_vqa",
                    source_dataset="synthetic_2x8_v2",
                    arrival_offset_s=off,
                    turns=[
                        InterleavedTurn(
                            turn_idx=0,
                            modality_plan="ar_only",
                            prompt="compare architectural features",
                            max_ar_tokens=6,
                            use_prev_image=False,
                        ),
                        InterleavedTurn(
                            turn_idx=1,
                            modality_plan="ar_only",
                            prompt="identify lighting direction in second view",
                            max_ar_tokens=6,
                            use_prev_image=False,
                        ),
                    ],
                )
            )
        else:
            plan = "cot_then_dit" if ((i // 4) % 2 == 0) else "dit_t2i"
            sessions.append(
                InterleavedSession(
                    session_id=f"mn_vcot_hires_{i:03d}",
                    quadrant="visual_cot_highres",
                    source_dataset="synthetic_2x8_v2",
                    arrival_offset_s=off,
                    turns=[
                        InterleavedTurn(
                            turn_idx=0,
                            modality_plan=plan,
                            prompt="alpine ridge under crisp morning light",
                            height=1024,
                            width=1024,
                            dit_steps=3,
                            max_ar_tokens=6,
                            use_prev_image=False,
                        ),
                    ],
                )
            )
    span_s = round(max((float(s.arrival_offset_s) for s in sessions), default=0.0), 3)
    return WorkloadSuite(
        name=f"synthetic_multinode_v2_n{len(sessions)}",
        trace_source="synthetic_burst_v2",
        base_qps=offered_qps,
        sessions=sessions,
        schema="morphserve.interleaved_bench.v2",
        warmup_s=0.0,
        steady_s=span_s,
        arrival_duration_s=span_s,
    )


def _prepare_e2e_suite(
    bench_root: Path,
    suite_json: Path | None,
    num_sessions: int | None,
    offered_qps: float,
    synthetic_e2e: bool,
    *,
    arrival_duration_s: float = 240.0,
    warmup_s: float = 60.0,
    steady_s: float = 180.0,
    min_sessions: int = 40,
) -> WorkloadSuite:
    if synthetic_e2e:
        n_syn = int(num_sessions) if num_sessions is not None else 16
        return _build_synthetic_multinode_suite(num_sessions=n_syn, offered_qps=offered_qps)

    s_path = suite_json if suite_json is not None else resolve_default_main_suite_path(bench_root)
    raw = WorkloadSuite.load_json(s_path)

    sources_dir = resolve_sources_dir(bench_root)
    b_csv = sources_dir / "burstgpt_1_sample.csv"
    g_csv = sources_dir / "gentd26_lora_request_trace.csv"
    burst_d = load_burstgpt_interarrivals(b_csv) if b_csv.is_file() else [0.4, 0.8, 1.8]
    gentd_d = load_gentd26_interarrivals(g_csv) if g_csv.is_file() else [0.6, 1.0, 1.4]

    eff_max = int(num_sessions) if (num_sessions is not None and int(num_sessions) > 0) else None
    eff_min = min(int(min_sessions), eff_max) if eff_max is not None else int(min_sessions)

    suite = build_time_window_suite(
        raw,
        target_qps=float(offered_qps),
        burstgpt_deltas=burst_d,
        gentd26_deltas=gentd_d,
        arrival_duration_s=float(arrival_duration_s),
        min_sessions=eff_min,
        max_sessions=eff_max,
        warmup_s=float(warmup_s),
        steady_s=float(steady_s),
    )
    suite.name = f"{raw.name}_2x8_n{len(suite.sessions)}"
    return suite


def run_multinode_e2e_benchmark(
    output_dir: Path,
    *,
    model_dir: Path = Path("."),
    bench_root: Path = Path("."),
    suite_json: Path | None = None,
    gpus_per_node: int = 8,
    num_sessions: int | None = None,
    offered_qps: float = 1.6,
    arrival_duration_s: float = 240.0,
    warmup_s: float = 60.0,
    steady_s: float = 180.0,
    min_sessions: int = 40,
    policies: Sequence[str] = ("morphserve", "static_nccl_fission", "monolithic_dp"),
    model_family: str = "bagel",
    intra_bw: float = 8.37,
    socket_bw: float = 8.59,
    inter_bw: float = 1.11,
    synthetic_e2e: bool = False,
    coord_port_override: int | None = None,
    spotcheck_summary: dict[str, Any] | None = None,
    engine_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run live 2-node (e.g. 2x8 = 16-GPU) distributed `MorphServeClusterEngine` E2E serving evaluation (v2)."""
    node_rank = int(os.environ.get("RANK", "0"))
    num_nodes = int(os.environ.get("WORLD_SIZE", "2"))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))
    env_coord_port = os.environ.get("MORPHSERVE_COORD_PORT")
    coord_port = (
        int(coord_port_override)
        if coord_port_override is not None
        else (int(env_coord_port) if env_coord_port else (master_port + 21))
    )
    world_size = num_nodes * int(gpus_per_node)
    backend = "inproc_synthetic" if (synthetic_e2e or model_family == "synthetic") else "mp_gpu"
    eff_family = "synthetic" if synthetic_e2e else model_family

    if node_rank >= 1:
        return run_multinode_worker_daemon(
            master_addr=master_addr,
            port=coord_port,
            node_rank=node_rank,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    topo = ClusterTopologyProfile.from_multinode_measurements(
        world_size=world_size,
        gpus_per_node=gpus_per_node,
        intra_bw=intra_bw,
        socket_bw=socket_bw,
        inter_bw=inter_bw,
    )
    suite = _prepare_e2e_suite(
        bench_root=bench_root,
        suite_json=suite_json,
        num_sessions=num_sessions,
        offered_qps=offered_qps,
        synthetic_e2e=(backend == "inproc_synthetic"),
        arrival_duration_s=arrival_duration_s,
        warmup_s=warmup_s,
        steady_s=steady_s,
        min_sessions=min_sessions,
    )
    quadrant_counts: dict[str, int] = defaultdict(int)
    for s in suite.sessions:
        quadrant_counts[s.quadrant] += 1

    hub = MultinodeCoordinatorHub(
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        bind_host="0.0.0.0",
        port=coord_port,
        advertise_host=master_addr,
        backend=backend,
    )
    runner = BenchmarkRunner(
        model_dir=model_dir,
        bench_root=bench_root,
        tmp_root_base=output_dir,
        engine_factory=engine_factory,
        fleet_factory=lambda cfg: TcpMultinodeWorkerFleet(cfg, model_dir=model_dir, bench_root=bench_root, hub=hub),
        topology=topo,
    )
    results: list[dict[str, Any]] = []
    summary_by_policy: dict[str, dict[str, Any]] = {}
    ms_max_inter_ship_bytes = 0
    window_mode = "session_trim" if backend == "inproc_synthetic" else "time"

    try:
        hub.wait_for_remote_nodes()
        for pol_name in policies:
            pol_overrides: dict[str, Any] = {
                "model_family": eff_family,
                "ar_tpot_slo_mode": "turn_p95",
                "steady_window_mode": window_mode,
                "warmup_window_s": float(suite.warmup_s),
                "steady_window_s": float(suite.steady_s),
            }
            if backend == "inproc_synthetic":
                pol_overrides.update(
                    {
                        "inter_turn_think_s_range": (0.0, 0.0),
                        "think_time_dist": "constant",
                        "think_time_median_s": 0.0,
                        "think_time_sigma": 0.0,
                    }
                )
            else:
                pol_overrides.update(
                    {
                        "inter_turn_think_s_range": (2.0, 25.0),
                        "think_time_dist": "lognormal",
                        "think_time_median_s": 6.0,
                        "think_time_sigma": 0.75,
                    }
                )

            cfg = EnginePolicyConfig.from_preset(pol_name, num_gpus=world_size, **pol_overrides)

            def _on_pol(spec: SuiteRunSpec, rep: Any, compact: dict[str, Any], _pol_name: str = pol_name) -> None:
                nonlocal ms_max_inter_ship_bytes
                if _pol_name == "morphserve":
                    ms_max_inter_ship_bytes = max(
                        ms_max_inter_ship_bytes,
                        int(compact.get("max_cross_node_ship_bytes", 0)),
                    )
                    for s_row in rep.sessions:
                        for b in s_row.get("cross_node_ship_bytes_list") or ():
                            ms_max_inter_ship_bytes = max(ms_max_inter_ship_bytes, int(b))

                summary_by_policy[_pol_name] = {
                    "steady_state_window_mode": str(rep.steady_state_window_mode),
                    "steady_window_start_s": float(rep.steady_window_start_s),
                    "steady_window_end_s": float(rep.steady_window_end_s),
                    "steady_state_sessions": int(rep.steady_state_sessions),
                    "steady_oom_sessions": int(rep.steady_oom_sessions),
                    "joint_slo_goodput_rps": float(rep.joint_slo_goodput_rps),
                    "full_trace_goodput_rps": float(rep.full_trace_goodput_rps),
                    "joint_slo_attainment_pct": float(rep.joint_slo_attainment_pct),
                    "steady_state_survival_pct": float(rep.steady_state_survival_pct),
                    "active_jct_p50_ms": float(rep.active_jct_p50_ms),
                    "active_jct_p95_ms": float(rep.active_jct_p95_ms),
                    "active_jct_p99_ms": float(rep.active_jct_p99_ms),
                    "all_session_active_jct_p95_ms": rep.all_session_active_jct_p95_ms,
                    "kv_host_swap_outs": int(rep.kv_host_swap_outs),
                    "kv_host_swap_ins": int(rep.kv_host_swap_ins),
                    "vmm_physical_pool_mib": float(rep.vmm_physical_pool_mib),
                    "per_turn_stats": rep.per_turn_stats,
                    "cross_node_ships": int(rep.cross_node_ships),
                    "max_cross_node_ship_bytes": int(rep.max_cross_node_ship_bytes),
                    "peak_nvml_used_mib": float(rep.peak_nvml_used_mib),
                }
                errs = compact.get("error_samples") or ()
                print(
                    f"[M2.5-E2E-v2] policy={_pol_name} world_size={world_size} n={len(spec.suite.sessions)} qps={offered_qps} "
                    f"win={rep.steady_state_window_mode} steady_goodput={rep.joint_slo_goodput_rps:.3f} "
                    f"steady_slo={rep.joint_slo_attainment_pct}% steady_surv={rep.steady_state_survival_pct}% "
                    f"active_jct_p95={rep.active_jct_p95_ms}ms swaps={rep.kv_host_swap_outs} xnode_ships={rep.cross_node_ships} "
                    f"steady_oom={rep.steady_oom_sessions} oom={rep.oom_sessions} err={rep.error_sessions}"
                    + (f" sample_err={errs[0]!r}" if errs else ""),
                    flush=True,
                )

            results.extend(
                runner.run_policy_sweep(
                    cfg,
                    [SuiteRunSpec(suite=suite, target_qps=float(offered_qps))],
                    tmp_tag=f"_tmp_m25_{pol_name}",
                    max_error_samples=5,
                    attach_init_shmem=False,
                    on_result=_on_pol,
                )
            )
    finally:
        hub.close()

    all_ok = all(r.get("status") == "ok" and int(r.get("error_sessions", 0)) == 0 for r in results)
    suppressed_multigib = ms_max_inter_ship_bytes <= 128 * MIB
    report: dict[str, Any] = {
        "schema": "morphserve.osdi27.m25_multinode_2x8_e2e.v3",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_nodes": num_nodes,
        "gpus_per_node": gpus_per_node,
        "world_size": world_size,
        "backend": backend,
        "model_family": eff_family,
        "offered_qps": float(offered_qps),
        "steady_state_window_mode": window_mode,
        "arrival_duration_s": float(suite.arrival_duration_s),
        "warmup_s": float(suite.warmup_s),
        "steady_s": float(suite.steady_s),
        "num_sessions": len(suite.sessions),
        "quadrant_counts": dict(sorted(quadrant_counts.items())),
        "calibrated_gbps": {
            "intra_half_pcie": intra_bw,
            "intra_node_cross_socket": socket_bw,
            "inter_node_fabric": inter_bw,
        },
        "suppressed_cross_node_multigib_kv": suppressed_multigib,
        "morphserve_max_cross_node_ship_bytes": ms_max_inter_ship_bytes,
        "summary_by_policy": summary_by_policy,
        "results": results,
        "status": "ok" if (all_ok and suppressed_multigib) else "check_failed",
    }
    if spotcheck_summary is not None:
        report["fabric_spotcheck"] = spotcheck_summary

    e2e_path = output_dir / "m25-multinode-2x8-e2e-report.json"
    e2e_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {e2e_path}", flush=True)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MorphServe M2.5 2x8 (16-GPU) Multi-Node Fabric Calibration & Distributed E2E Benchmark (v2)"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("spotcheck", "e2e", "both"), default="spotcheck")
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--model-dir", type=Path, default=Path("."))
    parser.add_argument("--bench-root", type=Path, default=Path("."))
    parser.add_argument("--suite-json", type=Path, default=None)
    parser.add_argument("--num-sessions", type=int, default=None)
    parser.add_argument("--arrival-duration-s", type=float, default=240.0)
    parser.add_argument("--warmup-s", type=float, default=60.0)
    parser.add_argument("--steady-s", type=float, default=180.0)
    parser.add_argument("--min-sessions", type=int, default=40)
    parser.add_argument("--qps", type=float, default=1.6)
    parser.add_argument(
        "--policies",
        type=str,
        default="morphserve,static_nccl_fission,monolithic_dp",
        help="Comma-separated policy presets for E2E multi-node benchmark",
    )
    parser.add_argument("--model-family", choices=("bagel", "internvl_u", "synthetic"), default="bagel")
    parser.add_argument("--intra-bw", type=float, default=8.37)
    parser.add_argument("--socket-bw", type=float, default=8.59)
    parser.add_argument("--inter-bw", type=float, default=1.11)
    parser.add_argument("--coord-port", type=int, default=None)
    parser.add_argument("--synthetic-e2e", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    eff_mode = args.mode
    if args.synthetic_e2e and eff_mode == "spotcheck":
        eff_mode = "e2e"

    spot_rep: dict[str, Any] | None = None
    intra_bw = float(args.intra_bw)
    socket_bw = float(args.socket_bw)
    inter_bw = float(args.inter_bw)

    if eff_mode in ("spotcheck", "both"):
        spot_rep = run_multinode_spotcheck(args.output_dir, gpus_per_node=args.gpus_per_node)
        if eff_mode == "spotcheck":
            return 0 if spot_rep.get("status") == "ok" else 2
        calib = spot_rep.get("calibrated_gbps") or {}
        if calib:
            intra_bw = float(calib.get("intra_half_pcie", intra_bw))
            socket_bw = float(calib.get("intra_node_cross_socket", socket_bw))
            inter_bw = float(calib.get("inter_node_fabric", inter_bw))

    pol_list = tuple(parse_csv_strings(args.policies))
    e2e_rep = run_multinode_e2e_benchmark(
        args.output_dir,
        model_dir=args.model_dir,
        bench_root=args.bench_root,
        suite_json=args.suite_json,
        gpus_per_node=args.gpus_per_node,
        num_sessions=args.num_sessions,
        offered_qps=args.qps,
        arrival_duration_s=args.arrival_duration_s,
        warmup_s=args.warmup_s,
        steady_s=args.steady_s,
        min_sessions=args.min_sessions,
        policies=pol_list,
        model_family=args.model_family,
        intra_bw=intra_bw,
        socket_bw=socket_bw,
        inter_bw=inter_bw,
        synthetic_e2e=args.synthetic_e2e,
        coord_port_override=args.coord_port,
        spotcheck_summary=spot_rep,
    )
    spot_ok = True if spot_rep is None else (spot_rep.get("status") == "ok")
    return 0 if (spot_ok and e2e_rep.get("status") == "ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
