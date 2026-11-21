"""Phase-2 Milestone M2.5: Live 2x8 (16-GPU) Cross-Node Fabric & Dual-Path Routing Spot-Check."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import struct
import time
from typing import Any

from morphserve.router import (
    CPTGRequestSpec,
    ClusterTopologyProfile,
    DirectNcclMesh,
    DualPathMorphRouter,
    HalfNodeGroup,
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

    # Build 16-GPU 2-node topology with live measured bandwidths & verify routing decisions
    groups = tuple(
        HalfNodeGroup(
            group_id=gi,
            numa_node=gi % 2,
            gpu_ids=tuple(range(gi * 4, (gi + 1) * 4)),
            h2d_bandwidth_gbps=22.9,
            p2p_intra_gbps=intra_bw,
        )
        for gi in range(world_size // 4)
    )
    bw_mat: dict[tuple[int, int], float] = {}
    for s in range(world_size):
        for d in range(world_size):
            if s == d:
                bw_mat[(s, d)] = 999.0
            elif (s // 4) == (d // 4):
                bw_mat[(s, d)] = intra_bw
            elif (s // gpus_per_node) == (d // gpus_per_node):
                bw_mat[(s, d)] = socket_bw
            else:
                bw_mat[(s, d)] = inter_bw

    topo16 = ClusterTopologyProfile(
        num_gpus=world_size,
        half_groups=groups,
        p2p_gbps_matrix=bw_mat,
        gen_expert_bytes=int(12.02 * GIB),
    )
    router16 = DualPathMorphRouter(topo16)

    # Scenario: Node 0 GPUs busy except cold hybrid GPU 0 (alpha=0.25); Node 1 GPU 8 is idle warm Gen (alpha=1.0)
    states = {
        g: WorkerState(
            gpu_id=g,
            role="hybrid" if g == 0 else "gen",
            expert_alpha=0.25 if g == 0 else 1.0,
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
            kv_bytes=24 * MIB,
            dit_step1_ms=190.0,
            dit_total_steps=20,
            resolution=1024,
        ),
        states,
    )
    dec_bagel_1g = router16.decide(
        CPTGRequestSpec(
            req_id="bagel_multiturn_1024m",
            src_gpu=0,
            architecture="tight_mot",
            kv_bytes=1024 * MIB,
            dit_step1_ms=287.0,
            dit_total_steps=20,
            resolution=512,
        ),
        states,
    )

    all_vmm_ok = all(r["vmm_ok"] for r in all_ranks)
    routing_pass = (dec_bridge.path == "state_ship") and (dec_bagel_1g.path == "in_place_morph")
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
        "cross_node_routing_verification": {
            "bridge_24mib_decision": {
                "path": dec_bridge.path,
                "dst_gpu": dec_bridge.dst_gpu,
                "ship_cost_ms": round(dec_bridge.ship_cost_ms, 2),
                "morph_cost_ms": round(dec_bridge.morph_cost_ms, 2),
            },
            "tight_mot_1024mib_decision": {
                "path": dec_bagel_1g.path,
                "dst_gpu": dec_bagel_1g.dst_gpu,
                "ship_cost_ms": round(dec_bagel_1g.ship_cost_ms, 2),
                "morph_cost_ms": round(dec_bagel_1g.morph_cost_ms, 2),
            },
            "suppressed_cross_node_1gib_kv_ship": dec_bagel_1g.path == "in_place_morph",
        },
        "ranks_verified": len(all_ranks),
        "status": "ok" if (len(all_ranks) == world_size and all_vmm_ok and routing_pass) else "check_failed",
    }
    out_path = output_dir / "m25-multinode-2x8-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MorphServe M2.5 2x8 (16-GPU) Multi-Node Spot-Check")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    args = parser.parse_args(argv)
    rep = run_multinode_spotcheck(args.output_dir, gpus_per_node=args.gpus_per_node)
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
