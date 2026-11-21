"""Hardware PCIe/NUMA topology profile and analytical bandwidth modeling for Module 2."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


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
    """Measured PCIe/NUMA/P2P bandwidth profile across GPUs on a multi-GPU node."""

    num_gpus: int
    gpu_names: list[str]
    half_groups: list[PcieHalfGroup]
    p2p_bw_gbps: list[list[float]]
    intra_half_p2p_cap_gbps: float = 19.67
    cross_half_p2p_cap_gbps: float = 9.98

    def half_group_of(self, gpu_id: int) -> PcieHalfGroup:
        for grp in self.half_groups:
            if gpu_id in grp.gpu_ids:
                return grp
        raise KeyError(f"gpu_id={gpu_id} not mapped to any PcieHalfGroup")

    def numa_node_of(self, gpu_id: int) -> int:
        return self.half_group_of(gpu_id).numa_node

    def same_half(self, gpu_a: int, gpu_b: int) -> bool:
        return self.half_group_of(gpu_a).group_id == self.half_group_of(gpu_b).group_id

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
    ) -> float:
        """Directional P2P/NCCL bandwidth between `src_gpu` and `dst_gpu` under concurrency."""
        if src_gpu == dst_gpu:
            return 900.0
        raw = float(self.p2p_bw_gbps[src_gpu][dst_gpu])
        if self.same_half(src_gpu, dst_gpu):
            cap = self.intra_half_p2p_cap_gbps / float(max(1, concurrent_intra_half))
        else:
            cap = self.cross_half_p2p_cap_gbps / float(max(1, concurrent_cross_half))
        return max(0.5, min(raw, cap))

    def peer_candidates(self, src_gpu: int, prefer_same_half: bool = True) -> list[int]:
        peers = [g for g in range(self.num_gpus) if g != src_gpu]
        if prefer_same_half:
            peers.sort(
                key=lambda g: (
                    0 if self.same_half(src_gpu, g) else 1,
                    -self.p2p_bw_gbps[src_gpu][g],
                    g,
                )
            )
        return peers

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_gpus": self.num_gpus,
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
    # Normalize "00000000:19:00.0" -> "0000:19:00.0" for /sys/bus/pci/devices lookup
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
        return ClusterTopologyProfile.from_job0c_report(job0c_fallback) if job0c_fallback and job0c_fallback.exists() else ClusterTopologyProfile.a100_pcie_8gpu_default()

    if not torch.cuda.is_available():
        return ClusterTopologyProfile.from_job0c_report(job0c_fallback) if job0c_fallback and job0c_fallback.exists() else ClusterTopologyProfile.a100_pcie_8gpu_default()

    n = int(torch.cuda.device_count())
    names = [str(torch.cuda.get_device_name(i)) for i in range(n)]
    if n != 8:
        half = max(1, n // 2)
        groups = [PcieHalfGroup(group_id=0, numa_node=0, gpu_ids=tuple(range(half)))]
        if n > half:
            groups.append(PcieHalfGroup(group_id=1, numa_node=1, gpu_ids=tuple(range(half, n))))
        p2p = [[0.0 if i == j else 10.0 for j in range(n)] for i in range(n)]
        return ClusterTopologyProfile(num_gpus=n, gpu_names=names, half_groups=groups, p2p_bw_gbps=p2p)

    # Detect sysfs NUMA socket per GPU if available; otherwise map {0..3} -> NUMA 0, {4..7} -> NUMA 1
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
    )
