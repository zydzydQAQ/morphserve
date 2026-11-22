"""Three-tier PCIe/Socket/RDMA hardware topology profile and bandwidth modeling for MorphServe."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OperatorLatencyProfile:
    """Single calibrated source of truth for solo unloaded A100 operator latencies (`T_unloaded`) & `k×` SLOs."""

    ar_prefill_ms: float = 64.8
    ar_tpot_ms: float = 62.82
    prep_overhead_ms: float = 240.0
    bagel_dit_512_step_ms: float = 287.0
    bagel_dit_1024_step_ms: float = 1040.0
    bagel_vae_512_ms: float = 140.0
    bagel_vae_1024_ms: float = 680.0
    bagel_expert_bytes: int = 12_446 * (1024**2)
    internvlu_dit_512_step_ms: float = 145.0
    internvlu_dit_1024_step_ms: float = 420.0
    internvlu_vae_512_ms: float = 110.0
    internvlu_vae_1024_ms: float = 340.0
    internvlu_expert_bytes: int = 4_350 * (1024**2)

    def step_ms(self, model_family: str, height: int) -> float:
        is_1024 = int(height) >= 1024
        if model_family == "internvl_u":
            return self.internvlu_dit_1024_step_ms if is_1024 else self.internvlu_dit_512_step_ms
        return self.bagel_dit_1024_step_ms if is_1024 else self.bagel_dit_512_step_ms

    def vae_ms(self, model_family: str, height: int) -> float:
        is_1024 = int(height) >= 1024
        if model_family == "internvl_u":
            return self.internvlu_vae_1024_ms if is_1024 else self.internvlu_vae_512_ms
        return self.bagel_vae_1024_ms if is_1024 else self.bagel_vae_512_ms

    def expert_bytes(self, model_family: str) -> int:
        return self.internvlu_expert_bytes if model_family == "internvl_u" else self.bagel_expert_bytes

    def unloaded_dit_turn_ms(self, model_family: str, height: int, steps: int = 20) -> float:
        eff_steps = max(1, int(steps))
        return eff_steps * self.step_ms(model_family, height) + self.vae_ms(model_family, height) + self.prep_overhead_ms

    def dit_turn_slo_ms(self, height: int, k: float = 2.0, model_family: str = "bagel", steps: int = 20) -> float:
        return round(float(k) * self.unloaded_dit_turn_ms(model_family, height, steps=steps), 2)

    def slo_thresholds_ms(self, k: float = 2.0, model_family: str = "bagel") -> dict[str, float]:
        fk = float(k)
        return {
            "slo_scale_k": fk,
            "ar_tpot_slo_ms": round(fk * self.ar_tpot_ms, 2),
            "ar_ttft_slo_ms": round(750.0 * fk, 2),
            "dit_512_slo_ms": self.dit_turn_slo_ms(512, k=fk, model_family=model_family, steps=20),
            "dit_1024_slo_ms": self.dit_turn_slo_ms(1024, k=fk, model_family=model_family, steps=16 if model_family == "internvl_u" else 20),
        }


DEFAULT_OPERATOR_PROFILE = OperatorLatencyProfile()


@dataclass(frozen=True)
class PcieHalfGroup:
    """Represents one PCIe root-complex / NUMA socket group of GPUs sharing H2D uplink."""

    group_id: int
    numa_node: int
    gpu_ids: tuple[int, ...]
    aggregate_h2d_gbps: float = 10.35
    solo_h2d_gbps: float = 8.75


@dataclass
class ClusterTopologyProfile:
    """Three-tier interconnect profile (`intra_half`, `inter_socket`, `inter_node`) across 1..N GPU nodes."""

    num_gpus: int
    gpu_names: list[str]
    half_groups: list[PcieHalfGroup]
    p2p_bw_gbps: list[list[float]]
    intra_half_p2p_cap_gbps: float = 19.67
    cross_half_p2p_cap_gbps: float = 9.98
    inter_node_p2p_cap_gbps: float = 11.5
    gpus_per_node: int = 8
    intra_node_handoff_ms: float = 3.0
    inter_node_handoff_ms: float = 8.0

    def node_of(self, gpu_id: int) -> int:
        return int(gpu_id) // max(1, int(self.gpus_per_node))

    def same_node(self, gpu_a: int, gpu_b: int) -> bool:
        return self.node_of(gpu_a) == self.node_of(gpu_b)

    def half_group_of(self, gpu_id: int) -> PcieHalfGroup:
        for grp in self.half_groups:
            if gpu_id in grp.gpu_ids:
                return grp
        raise KeyError(f"gpu_id={gpu_id} not mapped to any PcieHalfGroup")

    def numa_node_of(self, gpu_id: int) -> int:
        return self.half_group_of(gpu_id).numa_node

    def same_half(self, gpu_a: int, gpu_b: int) -> bool:
        return self.same_node(gpu_a, gpu_b) and (self.half_group_of(gpu_a).group_id == self.half_group_of(gpu_b).group_id)

    def fabric_tier(self, src_gpu: int, dst_gpu: int) -> str:
        if src_gpu == dst_gpu:
            return "self"
        if not self.same_node(src_gpu, dst_gpu):
            return "inter_node"
        if self.same_half(src_gpu, dst_gpu):
            return "intra_half"
        return "inter_socket"

    def handoff_ms(self, src_gpu: int, dst_gpu: int) -> float:
        if src_gpu == dst_gpu:
            return 0.0
        return self.intra_node_handoff_ms if self.same_node(src_gpu, dst_gpu) else self.inter_node_handoff_ms

    def solo_h2d_gbps(self, gpu_id: int) -> float:
        return self.half_group_of(gpu_id).solo_h2d_gbps

    def effective_h2d_gbps(self, gpu_id: int, active_h2d_same_half: int = 1) -> float:
        """Per-GPU effective H2D DMA rate modeling PCIe root uplink saturation on same half-node."""
        grp = self.half_group_of(gpu_id)
        k = max(1, int(active_h2d_same_half))
        return min(grp.solo_h2d_gbps, grp.aggregate_h2d_gbps / float(k))

    def effective_p2p_gbps(
        self,
        src_gpu: int,
        dst_gpu: int,
        concurrent_intra_half: int = 1,
        concurrent_cross_half: int = 1,
        concurrent_inter_node: int = 1,
    ) -> float:
        """Directional P2P/NCCL/RDMA bandwidth across all 3 fabric tiers under concurrency."""
        tier = self.fabric_tier(src_gpu, dst_gpu)
        if tier == "self":
            return 900.0
        raw = float(self.p2p_bw_gbps[src_gpu][dst_gpu])
        if tier == "intra_half":
            cap = self.intra_half_p2p_cap_gbps / float(max(1, concurrent_intra_half))
        elif tier == "inter_socket":
            cap = self.cross_half_p2p_cap_gbps / float(max(1, concurrent_cross_half))
        else:
            cap = self.inter_node_p2p_cap_gbps / float(max(1, concurrent_inter_node))
        return max(0.5, min(raw, cap))

    def peer_candidates(self, src_gpu: int, prefer_same_half: bool = True) -> list[int]:
        peers = [g for g in range(self.num_gpus) if g != src_gpu]
        if prefer_same_half:
            tier_rank = {"intra_half": 0, "inter_socket": 1, "inter_node": 2}
            peers.sort(
                key=lambda g: (
                    tier_rank.get(self.fabric_tier(src_gpu, g), 3),
                    -self.p2p_bw_gbps[src_gpu][g],
                    g,
                )
            )
        return peers

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_gpus": self.num_gpus,
            "gpus_per_node": self.gpus_per_node,
            "gpu_names": list(self.gpu_names),
            "half_groups": [
                {
                    "group_id": g.group_id,
                    "numa_node": g.numa_node,
                    "gpu_ids": list(g.gpu_ids),
                    "aggregate_h2d_gbps": g.aggregate_h2d_gbps,
                    "solo_h2d_gbps": g.solo_h2d_gbps,
                }
                for g in self.half_groups
            ],
            "intra_half_p2p_cap_gbps": self.intra_half_p2p_cap_gbps,
            "cross_half_p2p_cap_gbps": self.cross_half_p2p_cap_gbps,
            "inter_node_p2p_cap_gbps": self.inter_node_p2p_cap_gbps,
            "p2p_bw_gbps": self.p2p_bw_gbps,
        }

    @classmethod
    def a100_pcie_8gpu_default(cls) -> "ClusterTopologyProfile":
        """Measured Job-0c ground-truth profile for 8x NVIDIA A100-PCIE-40GB."""
        p2p = [[0.0 for _ in range(8)] for _ in range(8)]
        for i in range(8):
            for j in range(8):
                if i == j:
                    continue
                same = (i < 4 and j < 4) or (i >= 4 and j >= 4)
                p2p[i][j] = 10.6 if same else 9.55
        return cls(
            num_gpus=8,
            gpu_names=["NVIDIA A100-PCIE-40GB"] * 8,
            half_groups=[
                PcieHalfGroup(group_id=0, numa_node=0, gpu_ids=(0, 1, 2, 3), aggregate_h2d_gbps=10.35, solo_h2d_gbps=8.75),
                PcieHalfGroup(group_id=1, numa_node=1, gpu_ids=(4, 5, 6, 7), aggregate_h2d_gbps=10.40, solo_h2d_gbps=8.75),
            ],
            p2p_bw_gbps=p2p,
            intra_half_p2p_cap_gbps=19.67,
            cross_half_p2p_cap_gbps=9.98,
            inter_node_p2p_cap_gbps=11.5,
            gpus_per_node=8,
        )

    @classmethod
    def from_multinode_measurements(
        cls,
        world_size: int,
        gpus_per_node: int,
        intra_bw: float,
        socket_bw: float,
        inter_bw: float,
        solo_h2d_bw: float = 8.75,
        agg_h2d_bw: float = 10.35,
        gpu_name: str = "NVIDIA A100-PCIE-40GB",
        intra_handoff_ms: float = 3.0,
        inter_handoff_ms: float = 8.0,
    ) -> "ClusterTopologyProfile":
        """Construct a 3-tier multi-node topology profile for N GPUs across `world_size // gpus_per_node` nodes."""
        half_sz = max(1, min(4, gpus_per_node // 2 if gpus_per_node >= 2 else 1))
        n_groups = max(1, (world_size + half_sz - 1) // half_sz)
        groups = [
            PcieHalfGroup(
                group_id=gi,
                numa_node=gi % 2,
                gpu_ids=tuple(range(gi * half_sz, min(world_size, (gi + 1) * half_sz))),
                aggregate_h2d_gbps=agg_h2d_bw,
                solo_h2d_gbps=solo_h2d_bw,
            )
            for gi in range(n_groups)
        ]
        bw_mat: list[list[float]] = []
        for s in range(world_size):
            row: list[float] = []
            for d in range(world_size):
                if s == d:
                    row.append(900.0)
                elif (s // gpus_per_node) != (d // gpus_per_node):
                    row.append(float(inter_bw))
                elif (s // half_sz) == (d // half_sz):
                    row.append(float(intra_bw))
                else:
                    row.append(float(socket_bw))
            bw_mat.append(row)
        return cls(
            num_gpus=world_size,
            gpu_names=[gpu_name] * world_size,
            half_groups=groups,
            p2p_bw_gbps=bw_mat,
            intra_half_p2p_cap_gbps=max(0.5, float(intra_bw)),
            cross_half_p2p_cap_gbps=max(0.5, float(socket_bw)),
            inter_node_p2p_cap_gbps=max(0.5, float(inter_bw)),
            gpus_per_node=gpus_per_node,
            intra_node_handoff_ms=intra_handoff_ms,
            inter_node_handoff_ms=inter_handoff_ms,
        )

    @classmethod
    def from_job0c_report(cls, path: Path) -> "ClusterTopologyProfile":
        """Construct a topology profile from a Phase-0 Job-0c `topo-8gpu-report.json`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        prof = cls.a100_pcie_8gpu_default()
        p2p_mat = data.get("p2p_matrix_gbps") or data.get("p2p_bw_gbps")
        if isinstance(p2p_mat, list) and len(p2p_mat) == prof.num_gpus:
            prof.p2p_bw_gbps = [[float(x or 0.0) for x in row] for row in p2p_mat]
        elif isinstance(data.get("p2p"), dict) and isinstance(data["p2p"].get("bandwidth_gbps"), list):
            mat = data["p2p"]["bandwidth_gbps"]
            if len(mat) == prof.num_gpus:
                prof.p2p_bw_gbps = [[float(x or 0.0) for x in row] for row in mat]

        h2d_info = data.get("h2d_contention") or data.get("numa_h2d") or {}
        if isinstance(h2d_info, dict):
            h0_agg = h2d_info.get("half0_aggregate_gbps") or h2d_info.get("half0_4gpu_agg_gbps")
            h1_agg = h2d_info.get("half1_aggregate_gbps") or h2d_info.get("half1_4gpu_agg_gbps")
            solo_bw = h2d_info.get("solo_gpu0_gbps") or h2d_info.get("memfd_numa0_gbps")
            prof.half_groups = [
                PcieHalfGroup(
                    group_id=0,
                    numa_node=0,
                    gpu_ids=(0, 1, 2, 3),
                    aggregate_h2d_gbps=float(h0_agg) if h0_agg else 10.35,
                    solo_h2d_gbps=float(solo_bw) if solo_bw else 8.75,
                ),
                PcieHalfGroup(
                    group_id=1,
                    numa_node=1,
                    gpu_ids=(4, 5, 6, 7),
                    aggregate_h2d_gbps=float(h1_agg) if h1_agg else 10.40,
                    solo_h2d_gbps=float(solo_bw) if solo_bw else 8.75,
                ),
            ]
        return prof


def _read_gpu_numa_node_sysfs(bus_id: str) -> int:
    parts = bus_id.lower().split(":")
    if len(parts) == 3 and len(parts[0]) > 4:
        norm = f"{parts[0][-4:]}:{parts[1]}:{parts[2]}"
    else:
        norm = bus_id.lower()
    sys_path = Path(f"/sys/bus/pci/devices/{norm}/numa_node")
    try:
        val = int(sys_path.read_text(encoding="utf-8").strip())
        return val if val >= 0 else -1
    except Exception:
        return -1


def probe_live_topology(
    job0c_fallback: Path | None = None,
    quick_p2p_mib: int = 0,
) -> ClusterTopologyProfile:
    """Inspect live CUDA GPUs and PCIe/NUMA grouping cleanly; falls back to calibrated Job-0c defaults."""
    try:
        import torch
    except ImportError:
        return (
            ClusterTopologyProfile.from_job0c_report(job0c_fallback)
            if job0c_fallback and job0c_fallback.exists()
            else ClusterTopologyProfile.a100_pcie_8gpu_default()
        )

    if not torch.cuda.is_available():
        return (
            ClusterTopologyProfile.from_job0c_report(job0c_fallback)
            if job0c_fallback and job0c_fallback.exists()
            else ClusterTopologyProfile.a100_pcie_8gpu_default()
        )

    n = int(torch.cuda.device_count())
    names = [str(torch.cuda.get_device_name(i)) for i in range(n)]
    if n != 8:
        half = max(1, n // 2)
        groups = [PcieHalfGroup(group_id=0, numa_node=0, gpu_ids=tuple(range(half)))]
        if n > half:
            groups.append(PcieHalfGroup(group_id=1, numa_node=1, gpu_ids=tuple(range(half, n))))
        p2p = [[0.0 if i == j else 10.0 for j in range(n)] for i in range(n)]
        return ClusterTopologyProfile(num_gpus=n, gpu_names=names, half_groups=groups, p2p_bw_gbps=p2p, gpus_per_node=max(1, n))

    numa_map: dict[int, list[int]] = {}
    for i in range(n):
        node = -1
        props = torch.cuda.get_device_properties(i)
        bus_id = getattr(props, "pci_bus_id", None)
        if bus_id is not None and hasattr(props, "pci_device_id"):
            dom = getattr(props, "pci_domain_id", 0)
            node = _read_gpu_numa_node_sysfs(f"{dom:04x}:{int(bus_id):02x}:{int(props.pci_device_id):02x}.0")
        if node < 0:
            node = 0 if i < 4 else 1
        numa_map.setdefault(node, []).append(i)

    base = (
        ClusterTopologyProfile.from_job0c_report(job0c_fallback)
        if job0c_fallback and job0c_fallback.exists()
        else ClusterTopologyProfile.a100_pcie_8gpu_default()
    )
    half_groups: list[PcieHalfGroup] = []
    if len(numa_map) < 2:
        half_groups = list(base.half_groups)
    else:
        for gid, node in enumerate(sorted(numa_map.keys())):
            gpus = tuple(sorted(numa_map[node]))
            solo_bw = base.half_groups[gid % len(base.half_groups)].solo_h2d_gbps
            agg_bw = base.half_groups[gid % len(base.half_groups)].aggregate_h2d_gbps
            half_groups.append(
                PcieHalfGroup(group_id=gid, numa_node=node, gpu_ids=gpus, aggregate_h2d_gbps=agg_bw, solo_h2d_gbps=solo_bw)
            )

    p2p_bw = [list(row) for row in base.p2p_bw_gbps]
    if quick_p2p_mib > 0:
        nbytes = quick_p2p_mib * (1024**2)
        nelem = nbytes // 2
        bufs = [torch.empty((nelem,), dtype=torch.bfloat16, device=f"cuda:{i}") for i in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    p2p_bw[i][j] = 0.0
                    continue
                if not torch.cuda.can_device_access_peer(i, j):
                    p2p_bw[i][j] = 5.0
                    continue
                with torch.cuda.device(j):
                    bufs[j].copy_(bufs[i], non_blocking=True)
                    torch.cuda.synchronize(j)
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    for _ in range(4):
                        bufs[j].copy_(bufs[i], non_blocking=True)
                    e1.record()
                    torch.cuda.synchronize(j)
                dt_s = max(e0.elapsed_time(e1) / 1000.0, 1e-6)
                p2p_bw[i][j] = round((4.0 * nbytes / 1e9) / dt_s, 2)
        del bufs
        torch.cuda.empty_cache()

    return ClusterTopologyProfile(
        num_gpus=n,
        gpu_names=names,
        half_groups=half_groups,
        p2p_bw_gbps=p2p_bw,
        intra_half_p2p_cap_gbps=base.intra_half_p2p_cap_gbps,
        cross_half_p2p_cap_gbps=base.cross_half_p2p_cap_gbps,
        inter_node_p2p_cap_gbps=base.inter_node_p2p_cap_gbps,
        gpus_per_node=n,
    )
