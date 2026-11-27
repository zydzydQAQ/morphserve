"""Three-tier PCIe/Socket/RDMA hardware topology profile and operator latency profile backed by C++17."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OperatorLatencyProfile:
    """Calibrated solo unloaded operator latencies (`T_unloaded`) and `k×` SLO thresholds fetched from C++."""

    ar_prefill_ms: float
    ar_tpot_ms: float
    prep_overhead_ms: float
    bagel_dit_512_step_ms: float
    bagel_dit_1024_step_ms: float
    bagel_vae_512_ms: float
    bagel_vae_1024_ms: float
    bagel_expert_bytes: int
    internvlu_dit_512_step_ms: float
    internvlu_dit_1024_step_ms: float
    internvlu_vae_512_ms: float
    internvlu_vae_1024_ms: float
    internvlu_expert_bytes: int
    internvlu_ar_tpot_ms: float = 44.0
    ar_text_unloaded_ttft_ms: float = 125.0
    ar_img_512_unloaded_ttft_ms: float = 1400.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ar_prefill_ms": float(self.ar_prefill_ms),
            "ar_tpot_ms": float(self.ar_tpot_ms),
            "internvlu_ar_tpot_ms": float(self.internvlu_ar_tpot_ms),
            "ar_text_unloaded_ttft_ms": float(self.ar_text_unloaded_ttft_ms),
            "ar_img_512_unloaded_ttft_ms": float(self.ar_img_512_unloaded_ttft_ms),
            "prep_overhead_ms": float(self.prep_overhead_ms),
            "bagel_dit_512_step_ms": float(self.bagel_dit_512_step_ms),
            "bagel_dit_1024_step_ms": float(self.bagel_dit_1024_step_ms),
            "bagel_vae_512_ms": float(self.bagel_vae_512_ms),
            "bagel_vae_1024_ms": float(self.bagel_vae_1024_ms),
            "bagel_expert_bytes": int(self.bagel_expert_bytes),
            "internvlu_dit_512_step_ms": float(self.internvlu_dit_512_step_ms),
            "internvlu_dit_1024_step_ms": float(self.internvlu_dit_1024_step_ms),
            "internvlu_vae_512_ms": float(self.internvlu_vae_512_ms),
            "internvlu_vae_1024_ms": float(self.internvlu_vae_1024_ms),
            "internvlu_expert_bytes": int(self.internvlu_expert_bytes),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OperatorLatencyProfile":
        return cls(
            ar_prefill_ms=float(d["ar_prefill_ms"]),
            ar_tpot_ms=float(d["ar_tpot_ms"]),
            internvlu_ar_tpot_ms=float(d.get("internvlu_ar_tpot_ms", 44.0)),
            ar_text_unloaded_ttft_ms=float(d.get("ar_text_unloaded_ttft_ms", 125.0)),
            ar_img_512_unloaded_ttft_ms=float(d.get("ar_img_512_unloaded_ttft_ms", 1400.0)),
            prep_overhead_ms=float(d["prep_overhead_ms"]),
            bagel_dit_512_step_ms=float(d["bagel_dit_512_step_ms"]),
            bagel_dit_1024_step_ms=float(d["bagel_dit_1024_step_ms"]),
            bagel_vae_512_ms=float(d["bagel_vae_512_ms"]),
            bagel_vae_1024_ms=float(d["bagel_vae_1024_ms"]),
            bagel_expert_bytes=int(d["bagel_expert_bytes"]),
            internvlu_dit_512_step_ms=float(d["internvlu_dit_512_step_ms"]),
            internvlu_dit_1024_step_ms=float(d["internvlu_dit_1024_step_ms"]),
            internvlu_vae_512_ms=float(d["internvlu_vae_512_ms"]),
            internvlu_vae_1024_ms=float(d["internvlu_vae_1024_ms"]),
            internvlu_expert_bytes=int(d["internvlu_expert_bytes"]),
        )

    @classmethod
    def default(cls, **overrides: Any) -> "OperatorLatencyProfile":
        """Fetch the canonical default `OperatorLatencyProfile` from C++ with optional field overrides."""
        base = default_operator_profile()
        return replace(base, **overrides) if overrides else base

    def ar_tpot_ms_for(self, model_family: str = "bagel") -> float:
        return self.internvlu_ar_tpot_ms if model_family == "internvl_u" else self.ar_tpot_ms

    def unloaded_ar_ttft_ms(
        self,
        has_image: bool,
        num_images: int = 0,
        image_res: int = 512,
        model_family: str = "bagel",
    ) -> float:
        if (not has_image) and int(num_images) <= 0:
            return float(self.ar_text_unloaded_ttft_ms)
        n_img = max(1, int(num_images))
        res_scale = 2.0 if int(image_res) >= 1024 else 1.0
        fam_scale = 0.70 if model_family == "internvl_u" else 1.0
        per_img_ms = max(0.0, float(self.ar_img_512_unloaded_ttft_ms) - float(self.ar_text_unloaded_ttft_ms))
        return float(self.ar_text_unloaded_ttft_ms) + float(n_img) * per_img_ms * res_scale * fam_scale

    def ar_ttft_slo_for(
        self,
        has_image: bool,
        num_images: int = 0,
        image_res: int = 512,
        k: float = 2.0,
        model_family: str = "bagel",
    ) -> float:
        return round(float(k) * self.unloaded_ar_ttft_ms(has_image, num_images, image_res, model_family), 2)

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
        eff_steps = 16 if model_family == "internvl_u" else 20
        return {
            "slo_scale_k": fk,
            "ar_tpot_slo_ms": round(fk * self.ar_tpot_ms_for(model_family), 2),
            "ar_ttft_slo_ms": round(750.0 * fk, 2),
            "ar_ttft_slo_text_ms": self.ar_ttft_slo_for(False, 0, 512, k=fk, model_family=model_family),
            "ar_ttft_slo_img512_ms": self.ar_ttft_slo_for(True, 1, 512, k=fk, model_family=model_family),
            "ar_ttft_slo_img1024_ms": self.ar_ttft_slo_for(True, 1, 1024, k=fk, model_family=model_family),
            "dit_512_slo_ms": self.dit_turn_slo_ms(512, k=fk, model_family=model_family, steps=eff_steps),
            "dit_1024_slo_ms": self.dit_turn_slo_ms(1024, k=fk, model_family=model_family, steps=eff_steps),
        }


@lru_cache(maxsize=1)
def default_operator_profile() -> OperatorLatencyProfile:
    """Return the cached C++ authoritative default `OperatorLatencyProfile`."""
    from morphserve._core import call_json, get_core_lib

    L = get_core_lib()
    return OperatorLatencyProfile.from_dict(call_json(L, L.ms_topology_default_operator_profile_json))


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
        return self.same_node(gpu_a, gpu_b) and (
            self.half_group_of(gpu_a).group_id == self.half_group_of(gpu_b).group_id
        )

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
            "intra_node_handoff_ms": self.intra_node_handoff_ms,
            "inter_node_handoff_ms": self.inter_node_handoff_ms,
            "p2p_bw_gbps": self.p2p_bw_gbps,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ClusterTopologyProfile":
        return cls(
            num_gpus=int(d["num_gpus"]),
            gpu_names=[str(x) for x in d.get("gpu_names", [])],
            half_groups=[
                PcieHalfGroup(
                    group_id=int(g["group_id"]),
                    numa_node=int(g["numa_node"]),
                    gpu_ids=tuple(int(x) for x in g["gpu_ids"]),
                    aggregate_h2d_gbps=float(g["aggregate_h2d_gbps"]),
                    solo_h2d_gbps=float(g["solo_h2d_gbps"]),
                )
                for g in d.get("half_groups", [])
            ],
            p2p_bw_gbps=[[float(v) for v in row] for row in d.get("p2p_bw_gbps", [])],
            intra_half_p2p_cap_gbps=float(d["intra_half_p2p_cap_gbps"]),
            cross_half_p2p_cap_gbps=float(d["cross_half_p2p_cap_gbps"]),
            inter_node_p2p_cap_gbps=float(d["inter_node_p2p_cap_gbps"]),
            gpus_per_node=int(d.get("gpus_per_node", 8)),
            intra_node_handoff_ms=float(d["intra_node_handoff_ms"]),
            inter_node_handoff_ms=float(d["inter_node_handoff_ms"]),
        )

    @classmethod
    def a100_pcie_8gpu_default(cls) -> "ClusterTopologyProfile":
        """Measured Job-0c ground-truth profile for 8x NVIDIA A100-PCIE-40GB fetched from C++."""
        from morphserve._core import call_json, get_core_lib

        L = get_core_lib()
        return cls.from_dict(call_json(L, L.ms_topology_a100_default_json))

    @classmethod
    def from_multinode_measurements(
        cls,
        world_size: int,
        gpus_per_node: int,
        intra_bw: float,
        socket_bw: float,
        inter_bw: float,
        solo_h2d_bw: float | None = None,
        agg_h2d_bw: float | None = None,
        gpu_name: str = "NVIDIA A100-PCIE-40GB",
        intra_handoff_ms: float | None = None,
        inter_handoff_ms: float | None = None,
    ) -> "ClusterTopologyProfile":
        """Construct a 3-tier multi-node topology profile via C++ `ClusterTopologyProfile`."""
        from morphserve._core import call_json, get_core_lib

        if solo_h2d_bw is None or agg_h2d_bw is None or intra_handoff_ms is None or inter_handoff_ms is None:
            base = cls.a100_pcie_8gpu_default()
            grp0 = base.half_groups[0]
            solo_h2d_bw = grp0.solo_h2d_gbps if solo_h2d_bw is None else solo_h2d_bw
            agg_h2d_bw = grp0.aggregate_h2d_gbps if agg_h2d_bw is None else agg_h2d_bw
            intra_handoff_ms = base.intra_node_handoff_ms if intra_handoff_ms is None else intra_handoff_ms
            inter_handoff_ms = base.inter_node_handoff_ms if inter_handoff_ms is None else inter_handoff_ms

        L = get_core_lib()
        return cls.from_dict(
            call_json(
                L,
                L.ms_topology_multinode_json,
                int(world_size),
                int(gpus_per_node),
                float(intra_bw),
                float(socket_bw),
                float(inter_bw),
                float(solo_h2d_bw),
                float(agg_h2d_bw),
                gpu_name.encode("utf-8"),
                float(intra_handoff_ms),
                float(inter_handoff_ms),
            )
        )

    @classmethod
    def from_job0c_report(cls, path: Path) -> "ClusterTopologyProfile":
        """Construct a topology profile from a Phase-0 Job-0c `topo-8gpu-report.json` via C++."""
        from morphserve._core import call_json, get_core_lib

        L = get_core_lib()
        return cls.from_dict(call_json(L, L.ms_topology_from_job0c_json, str(Path(path)).encode("utf-8")))


def default_topology_for(num_gpus: int, gpus_per_node: int = 8) -> ClusterTopologyProfile:
    """Single canonical default `ClusterTopologyProfile` constructor derived from C++ A100 baselines."""
    n = max(1, int(num_gpus))
    gpn = min(n, max(1, int(gpus_per_node)))
    base = ClusterTopologyProfile.a100_pcie_8gpu_default()
    if n == 8 and gpn == 8:
        return base
    intra_bw = float(base.p2p_bw_gbps[0][1])
    socket_bw = float(base.p2p_bw_gbps[0][4])
    grp0 = base.half_groups[0]
    return ClusterTopologyProfile.from_multinode_measurements(
        world_size=n,
        gpus_per_node=gpn,
        intra_bw=intra_bw,
        socket_bw=socket_bw,
        inter_bw=float(base.inter_node_p2p_cap_gbps),
        solo_h2d_bw=float(grp0.solo_h2d_gbps),
        agg_h2d_bw=float(grp0.aggregate_h2d_gbps),
        gpu_name=base.gpu_names[0] if base.gpu_names else "NVIDIA A100-PCIE-40GB",
        intra_handoff_ms=float(base.intra_node_handoff_ms),
        inter_handoff_ms=float(base.inter_node_handoff_ms),
    )


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
        topo = default_topology_for(n, gpus_per_node=max(1, n))
        topo.gpu_names = names
        return topo

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
        intra_node_handoff_ms=base.intra_node_handoff_ms,
        inter_node_handoff_ms=base.inter_node_handoff_ms,
    )


def __getattr__(name: str) -> Any:
    if name == "DEFAULT_OPERATOR_PROFILE":
        return default_operator_profile()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ClusterTopologyProfile",
    "DEFAULT_OPERATOR_PROFILE",
    "OperatorLatencyProfile",
    "PcieHalfGroup",
    "default_operator_profile",
    "default_topology_for",
    "probe_live_topology",
]
