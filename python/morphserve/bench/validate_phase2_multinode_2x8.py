"""Phase-2 Milestone M2.5: Live 2x8 (16-GPU) Cross-Node Fabric Calibration & Distributed E2E Serving."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import math
import os
from pathlib import Path
import socket
import struct
import time
from typing import Any, Sequence

from morphserve.bench.interleaved_bench import (
    InterleavedSession,
    InterleavedTurn,
    WorkloadSuite,
    assign_dual_trace_arrivals,
    load_burstgpt_interarrivals,
    load_gentd26_interarrivals,
)
from morphserve.engine import (
    EnginePolicyConfig,
    MorphServeClusterEngine,
    MultinodeCoordinatorHub,
    TcpMultinodeWorkerFleet,
    run_multinode_worker_daemon,
)
from morphserve.router import (
    CPTGRequestSpec,
    ClusterTopologyProfile,
    DirectNcclMesh,
    DualPathMorphRouter,
    PcieHalfGroup,
    WorkerState,
)
from morphserve.vmm import TriPoolVMM

MIB = 1024**2
GIB = 1024**3


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
                hdr = b""
                while len(hdr) < 4:
                    chunk = cli.recv(4 - len(hdr))
                    if not chunk:
                        raise ConnectionError("short header")
                    hdr += chunk
                (nbytes,) = struct.unpack("!I", hdr)
                buf = bytearray()
                while len(buf) < nbytes:
                    chunk = cli.recv(nbytes - len(buf))
                    if not chunk:
                        raise ConnectionError("short payload")
                    buf.extend(chunk)
                return bytes(buf)
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
            hdr = conn.recv(4)
            (nbytes,) = struct.unpack("!I", hdr)
            buf = bytearray()
            while len(buf) < nbytes:
                buf.extend(conn.recv(nbytes - len(buf)))
            return json.loads(buf.decode("utf-8"))
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
        # Warmup ring/tree across all 16 GPUs
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
        # 3 fabric tiers from global_rank 0:
        #   Tier 1: 0 -> 1 (intra-half PCIe switch)
        #   Tier 2: 0 -> 4 (intra-node cross-socket)
        #   Tier 3: 0 -> 8 (inter-node fabric to Node 1 local_rank 0)
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

    # Scenario: Node 0 GPUs busy except hybrid GPU 0 (alpha=0.50); Node 1 GPU 8 is idle warm Gen (alpha=1.0)
    states = {
        g: WorkerState(
            gpu_id=g,
            role="hybrid" if g == 0 else "gen",
            expert_alpha=0.50 if g == 0 else 1.0,
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
    sessions: list[InterleavedSession] = []
    dt = 1.0 / max(0.05, float(offered_qps))
    for i in range(max(1, int(num_sessions))):
        off = round((i // 2) * dt * 1.6 + (i % 2) * 0.01, 4)
        mod = i % 4
        if mod == 0:
            sessions.append(
                InterleavedSession(
                    session_id=f"mn_vcot_{i:03d}",
                    quadrant="visual_cot",
                    source_dataset="synthetic_2x8",
                    arrival_offset_s=off,
                    turns=[
                        InterleavedTurn(
                            turn_idx=0,
                            modality_plan="cot_then_dit",
                            prompt="plan scene layout",
                            max_ar_tokens=6,
                            dit_steps=3,
                        ),
                        InterleavedTurn(
                            turn_idx=1,
                            modality_plan="ar_only",
                            prompt="verify spatial details",
                            max_ar_tokens=6,
                        ),
                    ],
                )
            )
        elif mod == 1:
            sessions.append(
                InterleavedSession(
                    session_id=f"mn_edit_{i:03d}",
                    quadrant="multiturn_edit",
                    source_dataset="synthetic_2x8",
                    arrival_offset_s=off,
                    turns=[
                        InterleavedTurn(turn_idx=0, modality_plan="dit_t2i", prompt="coastal town", dit_steps=3),
                        InterleavedTurn(turn_idx=1, modality_plan="dit_edit", prompt="sunset sky", dit_steps=3),
                        InterleavedTurn(turn_idx=2, modality_plan="dit_edit", prompt="harbor lights", dit_steps=3),
                    ],
                )
            )
        elif mod == 2:
            sessions.append(
                InterleavedSession(
                    session_id=f"mn_vqa_{i:03d}",
                    quadrant="multi_image_vqa",
                    source_dataset="synthetic_2x8",
                    arrival_offset_s=off,
                    turns=[
                        InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="compare views", max_ar_tokens=6),
                    ],
                )
            )
        else:
            sessions.append(
                InterleavedSession(
                    session_id=f"mn_hires_{i:03d}",
                    quadrant="highres_t2i",
                    source_dataset="synthetic_2x8",
                    arrival_offset_s=off,
                    turns=[
                        InterleavedTurn(
                            turn_idx=0,
                            modality_plan="dit_t2i",
                            prompt="alpine ridge",
                            height=1024,
                            width=1024,
                            dit_steps=3,
                        ),
                    ],
                )
            )
    return WorkloadSuite(
        name=f"synthetic_multinode_n{len(sessions)}",
        trace_source="synthetic_burst",
        base_qps=offered_qps,
        sessions=sessions,
    )


def _resolve_sources_dir(bench_root: Path) -> Path:
    for cand in (bench_root / "sources", bench_root.parent / "bench_sources"):
        if cand.is_dir():
            return cand
    return bench_root / "sources"


def _hamilton_quadrant_counts(
    by_quad: dict[str, list[InterleavedSession]],
    n_tot: int,
    target_sessions: int,
    *,
    cap_to_available: bool,
) -> dict[str, int]:
    quotas: list[tuple[str, int, float]] = []
    used = 0
    for q, items in sorted(by_quad.items()):
        exact = target_sessions * (len(items) / float(max(1, n_tot)))
        cnt = int(math.floor(exact))
        if cap_to_available:
            cnt = min(len(items), cnt)
        quotas.append((q, cnt, exact - math.floor(exact)))
        used += cnt

    quotas.sort(key=lambda x: (-x[2], x[0]))
    counts = {q: c for q, c, _ in quotas}
    rem = target_sessions - used
    idx = 0
    while rem > 0 and quotas:
        q = quotas[idx % len(quotas)][0]
        if (not cap_to_available) or counts[q] < len(by_quad[q]):
            counts[q] += 1
            rem -= 1
        elif all(counts[k] >= len(by_quad[k]) for k in counts):
            break
        idx += 1
    return counts


def _prepare_e2e_suite(
    bench_root: Path,
    suite_json: Path | None,
    num_sessions: int,
    offered_qps: float,
    synthetic_e2e: bool,
) -> tuple[WorkloadSuite, list[float] | None, list[float] | None]:
    if synthetic_e2e:
        return _build_synthetic_multinode_suite(num_sessions=num_sessions, offered_qps=offered_qps), None, None

    s_path = suite_json if suite_json is not None else (bench_root / "suites/main_mix_80.json")
    raw = WorkloadSuite.load_json(s_path)
    n_target = max(1, int(num_sessions))
    by_q: dict[str, list[InterleavedSession]] = defaultdict(list)
    for s in raw.sessions:
        by_q[s.quadrant].append(s)

    if n_target <= len(raw.sessions):
        counts = _hamilton_quadrant_counts(by_q, len(raw.sessions), n_target, cap_to_available=True)
        picked: list[InterleavedSession] = []
        for q, items in sorted(by_q.items()):
            picked.extend(items[: counts[q]])
        picked.sort(key=lambda s: (s.arrival_offset_s, s.session_id))
    else:
        counts = _hamilton_quadrant_counts(by_q, len(raw.sessions), n_target, cap_to_available=False)
        horizon_s = max((float(s.arrival_offset_s) for s in raw.sessions), default=1.0)
        mean_step_s = max(0.5, horizon_s / float(max(1, len(raw.sessions))))
        cycle_span_s = horizon_s + mean_step_s
        picked = []
        for q, items in sorted(by_q.items()):
            need = counts[q]
            n_q = len(items)
            for i in range(need):
                base_s = items[i % n_q]
                rep_idx = i // n_q
                sid = base_s.session_id if rep_idx == 0 else f"{base_s.session_id}_r{rep_idx}"
                picked.append(
                    InterleavedSession(
                        session_id=sid,
                        quadrant=base_s.quadrant,
                        source_dataset=base_s.source_dataset,
                        arrival_offset_s=round(float(base_s.arrival_offset_s) + rep_idx * cycle_span_s, 4),
                        turns=list(base_s.turns),
                    )
                )
        picked.sort(key=lambda s: (s.arrival_offset_s, s.session_id))

    sources_dir = _resolve_sources_dir(bench_root)
    b_csv = sources_dir / "burstgpt_1_sample.csv"
    g_csv = sources_dir / "gentd26_lora_request_trace.csv"
    burst_d = load_burstgpt_interarrivals(b_csv) if b_csv.is_file() else [0.4, 0.8, 1.8]
    gentd_d = load_gentd26_interarrivals(g_csv) if g_csv.is_file() else [0.6, 1.0, 1.4]
    assign_dual_trace_arrivals(picked, burst_d, gentd_d, target_qps=offered_qps)
    suite = WorkloadSuite(
        name=f"{raw.name}_2x8_n{len(picked)}",
        trace_source=raw.trace_source,
        base_qps=offered_qps,
        sessions=picked,
    )
    return suite, burst_d, gentd_d


def run_multinode_e2e_benchmark(
    output_dir: Path,
    *,
    model_dir: Path = Path("."),
    bench_root: Path = Path("."),
    suite_json: Path | None = None,
    gpus_per_node: int = 8,
    num_sessions: int = 160,
    offered_qps: float = 1.6,
    policies: Sequence[str] = ("morphserve", "static_nccl_fission", "monolithic_dp"),
    model_family: str = "bagel",
    intra_bw: float = 8.37,
    socket_bw: float = 8.59,
    inter_bw: float = 1.11,
    synthetic_e2e: bool = False,
    coord_port_override: int | None = None,
    spotcheck_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run live 2-node (e.g. 2x8 = 16-GPU) distributed `MorphServeClusterEngine` E2E serving evaluation."""
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
    suite, burst_d, gentd_d = _prepare_e2e_suite(
        bench_root=bench_root,
        suite_json=suite_json,
        num_sessions=num_sessions,
        offered_qps=offered_qps,
        synthetic_e2e=(backend == "inproc_synthetic"),
    )

    hub = MultinodeCoordinatorHub(
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        bind_host="0.0.0.0",
        port=coord_port,
        advertise_host=master_addr,
        backend=backend,
    )
    results: list[dict[str, Any]] = []
    ms_max_inter_ship_bytes = 0
    try:
        hub.wait_for_remote_nodes()
        for pol_name in policies:
            pol_overrides: dict[str, Any] = {"model_family": eff_family}
            if backend == "inproc_synthetic":
                pol_overrides["inter_turn_think_s_range"] = (0.0, 0.0)
            cfg = EnginePolicyConfig.from_preset(pol_name, num_gpus=world_size, **pol_overrides)
            fleet = TcpMultinodeWorkerFleet(cfg, model_dir=model_dir, bench_root=bench_root, hub=hub)
            tmp_dir = output_dir / f"_tmp_m25_{pol_name}"
            engine = MorphServeClusterEngine(
                policy=cfg,
                model_dir=model_dir,
                bench_root=bench_root,
                tmp_root=tmp_dir,
                fleet=fleet,
                topology=topo,
            )
            try:
                engine.start()
                rep = engine.run_suite(
                    suite,
                    target_qps=offered_qps,
                    burstgpt_deltas=burst_d,
                    gentd26_deltas=gentd_d,
                )
            finally:
                engine.shutdown()
                gc.collect()

            rep_dict = rep.to_dict()
            sess_rows = rep_dict.pop("sessions", [])
            err_samples = [str(s.get("error", ""))[:240] for s in sess_rows if s.get("status") == "error"][:5]
            if err_samples:
                rep_dict["error_samples"] = err_samples
            if pol_name == "morphserve":
                ms_max_inter_ship_bytes = max(
                    ms_max_inter_ship_bytes,
                    int(rep_dict.get("max_cross_node_ship_bytes", 0)),
                )
                for s_row in sess_rows:
                    for b in s_row.get("cross_node_ship_bytes_list") or ():
                        ms_max_inter_ship_bytes = max(ms_max_inter_ship_bytes, int(b))
            print(
                f"[M2.5-E2E] policy={pol_name} world_size={world_size} n={len(suite.sessions)} qps={offered_qps} "
                f"steady_goodput={rep.steady_state_goodput_rps:.3f} steady_slo={rep.steady_state_attainment_pct}% "
                f"xnode_ships={rep.cross_node_ships} oom={rep.oom_sessions} err={rep.error_sessions}"
                + (f" sample_err={err_samples[0]!r}" if err_samples else ""),
                flush=True,
            )
            results.append(rep_dict)
    finally:
        hub.close()

    all_ok = all(r.get("status") == "ok" and int(r.get("error_sessions", 0)) == 0 for r in results)
    suppressed_multigib = ms_max_inter_ship_bytes <= 128 * MIB
    report: dict[str, Any] = {
        "schema": "morphserve.osdi27.m25_multinode_2x8_e2e.v2",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_nodes": num_nodes,
        "gpus_per_node": gpus_per_node,
        "world_size": world_size,
        "backend": backend,
        "model_family": eff_family,
        "offered_qps": offered_qps,
        "num_sessions": len(suite.sessions),
        "calibrated_gbps": {
            "intra_half_pcie": intra_bw,
            "intra_node_cross_socket": socket_bw,
            "inter_node_fabric": inter_bw,
        },
        "suppressed_cross_node_multigib_kv": suppressed_multigib,
        "morphserve_max_cross_node_ship_bytes": ms_max_inter_ship_bytes,
        "results": results,
        "status": "ok" if (all_ok and suppressed_multigib) else "check_failed",
    }
    if spotcheck_summary is not None:
        report["fabric_spotcheck"] = spotcheck_summary

    e2e_path = output_dir / "m25-multinode-2x8-e2e-report.json"
    e2e_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {e2e_path}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MorphServe M2.5 2x8 (16-GPU) Multi-Node Fabric Calibration & Distributed E2E Benchmark"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("spotcheck", "e2e", "both"), default="spotcheck")
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--model-dir", type=Path, default=Path("."))
    parser.add_argument("--bench-root", type=Path, default=Path("."))
    parser.add_argument("--suite-json", type=Path, default=None)
    parser.add_argument("--num-sessions", type=int, default=160)
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
    args = parser.parse_args(argv)

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

    pol_list = tuple(p.strip() for p in str(args.policies).split(",") if p.strip())
    e2e_rep = run_multinode_e2e_benchmark(
        args.output_dir,
        model_dir=args.model_dir,
        bench_root=args.bench_root,
        suite_json=args.suite_json,
        gpus_per_node=args.gpus_per_node,
        num_sessions=args.num_sessions,
        offered_qps=args.qps,
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
