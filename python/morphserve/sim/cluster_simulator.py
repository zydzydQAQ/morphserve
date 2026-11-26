"""Python wrapper around C++17 `ClusterSimulator` (`morphserve::control::ClusterSimulator`)."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import ctypes
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from morphserve._core import call_json, encode_json, get_core_lib
from morphserve.engine.config import EnginePolicyConfig, STATIC_SPLIT_POLICIES, VALID_POLICIES
from morphserve.router import ClusterTopologyProfile, DualPathMorphRouter

MIB = 1024**2
GIB = 1024**3
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_BENCH_ROOT = REPO_ROOT / "data" / "interleaved-bench-v1"
DEFAULT_SOURCES_DIR = REPO_ROOT / "data" / "bench_sources"


@lru_cache(maxsize=1)
def _cpp_default_physics_dict() -> dict[str, Any]:
    L = get_core_lib()
    return dict(call_json(L, L.ms_physics_from_phase01_json, b""))


@dataclass(frozen=True, init=False)
class MeasuredPhysicsProfile:
    """100% Phase-0/1 empirical hardware & operator constants fetched directly from C++ `ClusterPhysicsProfile`."""

    vmm_64m_acquire_4gib_ms: float
    vmm_solo_h2d_stream_gbps: float
    pcie_solo_h2d_gbps: float
    pcie_half0_agg_h2d_gbps: float
    pcie_half1_agg_h2d_gbps: float
    pcie_intra_half_p2p_gbps: float
    pcie_cross_half_p2p_gbps: float

    vae_512_ms: float
    vae_1024_ms: float
    scoped_vae_512_spike_mib: float
    unscoped_vae_512_reserved_mib: float
    scoped_vae_1024_spike_mib: float
    unscoped_vae_1024_reserved_mib: float
    twostage_shm_base_ms: float
    twostage_shm_ms_per_mib: float

    monolithic_loaded_driver_mib: float
    offloaded_base_driver_mib: float
    driver_total_hbm_40g_mib: float
    num_expert_layers: int
    layer_stride_mib: int
    packed_expert_data_mib: float
    dit_512_20step_base_ms: float
    dit_512_transient_mib: float
    dit_1024_step_ms: float
    vmm_full_expert_map_ms: float

    cpu_pil_prep_only_ms: float
    gpu_inplace_prep_only_ms: float
    stock_loopback_turn_ms: float
    morphserve_loopback_turn_ms: float
    non_cow_kv_mult: float
    turn_edit_512_kv_delta_mib: float
    kv_mib_per_token: float
    stock_dit_512_ms_per_kv_tok: float
    morphserve_dit_512_ms_per_kv_tok: float

    ar_prefill_ms: float
    ar_unloaded_tpot_p50_ms: float
    ar_unloaded_tpot_p99_ms: float
    ar_batched2_step_p50_ms: float
    vmm_ar_kv_step_overhead_ms: float
    dit_512_cfg_step_mean_ms: float
    dit_512_layer_p50_ms: float
    m4_adaptive_chunk_layers: int
    m4_colocated_ar_tpot_p50_ms: float
    step_preempt_ar_tpot_p50_ms: float

    paced_h2d_per_gpu_gbps: float
    unpaced_4gpu_half_h2d_per_gpu_gbps: float
    nccl_intra_half_kv_gbps: float

    kairos_2x8_intra_half_gbps: float
    kairos_2x8_cross_socket_gbps: float
    kairos_2x8_inter_node_gbps: float

    provenance_files: dict[str, str] = field(default_factory=dict)

    def __init__(self, **overrides: Any) -> None:
        merged = dict(_cpp_default_physics_dict())
        merged.update(overrides)
        valid = {f.name for f in fields(self)}
        unknown = set(merged) - valid
        if unknown:
            raise TypeError(f"MeasuredPhysicsProfile() got unexpected keyword argument(s): {sorted(unknown)}")
        for name in valid:
            val = merged[name]
            if name == "provenance_files":
                val = dict(val) if isinstance(val, Mapping) else {}
            object.__setattr__(self, name, val)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeasuredPhysicsProfile":
        return cls(**dict(data))

    @classmethod
    def default(cls, **overrides: Any) -> "MeasuredPhysicsProfile":
        """Fetch canonical default C++ `ClusterPhysicsProfile` values with optional field overrides."""
        return cls(**overrides)

    @classmethod
    def from_phase01_results(cls, results_dir: Path | None = None) -> "MeasuredPhysicsProfile":
        root = DEFAULT_RESULTS_DIR if results_dir is None else Path(results_dir)
        L = get_core_lib()
        dir_arg = str(root).encode("utf-8") if root.is_dir() else b""
        d = call_json(L, L.ms_physics_from_phase01_json, dir_arg)
        return cls.from_dict(d)


@dataclass(frozen=True)
class WorkloadScaleSpec:
    """Model & session-KV scaling configuration for A100-40GB and H100-80GB generality evaluation."""

    name: str = "bagel_7b_standard"
    weight_scale: float = 1.0
    compute_scale: float = 1.0
    kv_per_turn_scale: float = 1.0
    vae_spike_scale: float = 1.0
    extra_edit_turns: int = 0
    think_s_range_override: tuple[float, float] | None = None
    super_pool_mib_80g: int = 28_672
    fine_kv_pool_mib_80g: int = 512

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weight_scale": float(self.weight_scale),
            "compute_scale": float(self.compute_scale),
            "kv_per_turn_scale": float(self.kv_per_turn_scale),
            "vae_spike_scale": float(self.vae_spike_scale),
            "extra_edit_turns": int(self.extra_edit_turns),
            "think_s_range_override": list(self.think_s_range_override)
            if self.think_s_range_override is not None
            else None,
            "super_pool_mib_80g": int(self.super_pool_mib_80g),
            "fine_kv_pool_mib_80g": int(self.fine_kv_pool_mib_80g),
        }

    @classmethod
    def bagel_7b_standard(cls) -> "WorkloadScaleSpec":
        return cls(name="bagel_7b_standard")

    @classmethod
    def frontier_mot_56gib(cls) -> "WorkloadScaleSpec":
        return cls(
            name="frontier_mot_56gib",
            weight_scale=2.0,
            compute_scale=1.85,
            kv_per_turn_scale=2.0,
            vae_spike_scale=1.5,
            extra_edit_turns=1,
            think_s_range_override=(6.0, 14.0),
            super_pool_mib_80g=28_672,
            fine_kv_pool_mib_80g=512,
        )

    @classmethod
    def bagel_7b_long_kv_high_concurrency(cls) -> "WorkloadScaleSpec":
        return cls(
            name="bagel_7b_long_kv_high_concurrency",
            weight_scale=1.0,
            compute_scale=1.0,
            kv_per_turn_scale=2.8,
            vae_spike_scale=1.0,
            extra_edit_turns=6,
            think_s_range_override=(20.0, 38.0),
            super_pool_mib_80g=28_672,
            fine_kv_pool_mib_80g=1_024,
        )


@dataclass(frozen=True)
class HardwareInterconnectSpec:
    """Hardware compute + memory + 3-tier fabric parameters for multi-node cluster simulation."""

    name: str
    gpu_hbm_gib: float
    compute_speedup_x: float
    solo_h2d_gbps: float
    agg_h2d_gbps: float
    intra_half_p2p_gbps: float
    inter_socket_p2p_gbps: float
    inter_node_rdma_gbps: float
    gpus_per_node: int = 8

    @classmethod
    def presets(cls, physics: MeasuredPhysicsProfile | None = None) -> tuple["HardwareInterconnectSpec", ...]:
        ph = MeasuredPhysicsProfile.default() if physics is None else physics
        return (
            cls(
                name="A100_PCIe_Gen4_RDMA100G",
                gpu_hbm_gib=40.0,
                compute_speedup_x=1.0,
                solo_h2d_gbps=ph.pcie_solo_h2d_gbps,
                agg_h2d_gbps=ph.pcie_half0_agg_h2d_gbps,
                intra_half_p2p_gbps=ph.pcie_intra_half_p2p_gbps,
                inter_socket_p2p_gbps=ph.pcie_cross_half_p2p_gbps,
                inter_node_rdma_gbps=11.5,
            ),
            cls(
                name="A100_SXM4_NVLink600G_RDMA200G",
                gpu_hbm_gib=40.0,
                compute_speedup_x=1.08,
                solo_h2d_gbps=22.9,
                agg_h2d_gbps=24.5,
                intra_half_p2p_gbps=265.0,
                inter_socket_p2p_gbps=265.0,
                inter_node_rdma_gbps=23.0,
            ),
            cls(
                name="H100_SXM5_NVLink900G_RDMA400G",
                gpu_hbm_gib=80.0,
                compute_speedup_x=2.25,
                solo_h2d_gbps=52.0,
                agg_h2d_gbps=56.0,
                intra_half_p2p_gbps=410.0,
                inter_socket_p2p_gbps=410.0,
                inter_node_rdma_gbps=46.0,
            ),
        )

    def to_topology(self, num_gpus: int) -> ClusterTopologyProfile:
        return ClusterTopologyProfile.from_multinode_measurements(
            world_size=num_gpus,
            gpus_per_node=self.gpus_per_node,
            intra_bw=self.intra_half_p2p_gbps,
            socket_bw=self.inter_socket_p2p_gbps,
            inter_bw=self.inter_node_rdma_gbps,
            solo_h2d_bw=self.solo_h2d_gbps,
            agg_h2d_bw=self.agg_h2d_gbps,
            gpu_name=self.name,
            intra_handoff_ms=0.35 if self.intra_half_p2p_gbps > 50.0 else 3.0,
            inter_handoff_ms=1.8 if self.inter_node_rdma_gbps > 20.0 else 8.0,
        )


@dataclass
class SimValidationReport:
    """Zero-tuning out-of-sample validation report against live 8-GPU / 16-GPU cluster runs."""

    reference_source: str
    zero_parameter_fitting: bool
    points_checked: int
    goodput_mape_pct: float
    ar_tpot_p95_mape_pct: float
    dit_p95_mape_pct: float
    max_point_error_pct: float
    within_8pct_oos_gate: bool
    within_5pct_gate: bool
    per_point: list[dict[str, Any]]


class SimGpuMemoryLedger:
    """Byte-accurate per-GPU physical HBM ledger backed by C++17 `morphserve::control::SimGpuMemoryLedger`."""

    def __init__(
        self,
        gpu_id: int,
        total_hbm_mib: float,
        policy: EnginePolicyConfig,
        role: str,
        resident_expert_layers: int = 28,
        num_expert_layers: int = 28,
        layer_stride_mib: int = 448,
        physics: MeasuredPhysicsProfile | None = None,
        weight_scale: float = 1.0,
        vae_spike_scale: float = 1.0,
    ) -> None:
        self.gpu_id = int(gpu_id)
        self.total_hbm_mib = float(total_hbm_mib)
        self.policy = policy
        self.role = str(role)
        self.physics = MeasuredPhysicsProfile.default() if physics is None else physics
        self.weight_scale = float(weight_scale)
        self.vae_spike_scale = float(vae_spike_scale)
        self._lib = get_core_lib()

        init_req = {
            "gpu_id": self.gpu_id,
            "total_hbm_mib": self.total_hbm_mib,
            "policy": policy.to_dict(),
            "role": self.role,
            "physics": self.physics.to_dict(),
            "weight_scale": self.weight_scale,
            "vae_spike_scale": self.vae_spike_scale,
            "resident_expert_layers": int(resident_expert_layers),
            "num_expert_layers": int(num_expert_layers),
            "layer_stride_mib": int(layer_stride_mib),
        }
        h = int(self._lib.ms_ledger_create(encode_json(init_req)) or 0)
        if h == 0:
            raise RuntimeError("ms_ledger_create failed")
        self._handle = h

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "SimGpuMemoryLedger":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        h = getattr(self, "_handle", 0)
        if h:
            self._handle = 0
            try:
                self._lib.ms_ledger_destroy(ctypes.c_void_p(h))
            except Exception:
                pass

    _STAT_ATTRS: frozenset[str] = frozenset(
        {
            "resident_expert_layers",
            "expert_alpha",
            "active_kv_mib",
            "super_to_fine_carves",
            "expert_evictions",
            "kv_host_swap_outs",
            "kv_host_swap_ins",
            "peak_used_mib",
        }
    )

    def stats(self) -> dict[str, Any]:
        """Return a single atomic snapshot dictionary of all C++ ledger counters and watermarks."""
        return dict(call_json(self._lib, self._lib.ms_ledger_stats_json, ctypes.c_void_p(self._handle)))

    def free_kv_bytes(self) -> int:
        """Return available physical KV budget in bytes from the authoritative C++ ledger."""
        return int(self.stats()["free_kv_bytes"])

    def __getattr__(self, name: str) -> Any:
        if name in self._STAT_ATTRS:
            return self.stats()[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def set_peak_used_mib(self, val: float) -> None:
        self._lib.ms_ledger_set_peak_used_mib(ctypes.c_void_p(self._handle), ctypes.c_double(float(val)))

    def evict_idle_kv(self, needed_free_mib: float, exclude_sid: int | None = None) -> float:
        return float(
            self._lib.ms_ledger_evict_idle_kv(
                ctypes.c_void_p(self._handle),
                ctypes.c_double(float(needed_free_mib)),
                1 if exclude_sid is not None else 0,
                ctypes.c_int(0 if exclude_sid is None else int(exclude_sid)),
            )
        )

    def ensure_resident(self, sid: int) -> float:
        return float(self._lib.ms_ledger_ensure_resident(ctypes.c_void_p(self._handle), ctypes.c_int(int(sid))))

    def upsert_kv(self, sid: int, logical_kv_mib: float) -> bool:
        return bool(
            self._lib.ms_ledger_upsert_kv(
                ctypes.c_void_p(self._handle),
                ctypes.c_int(int(sid)),
                ctypes.c_double(float(logical_kv_mib)),
            )
        )

    def exec_dit_vae(self, height: int, active_sid: int | None = None) -> tuple[bool, float]:
        swapped = ctypes.c_double(0.0)
        ok = int(
            self._lib.ms_ledger_exec_dit_vae(
                ctypes.c_void_p(self._handle),
                ctypes.c_int(int(height)),
                1 if active_sid is not None else 0,
                ctypes.c_int(0 if active_sid is None else int(active_sid)),
                ctypes.byref(swapped),
            )
        )
        return bool(ok == 1), float(swapped.value)

    def release_session(self, sid: int) -> None:
        self._lib.ms_ledger_release_session(ctypes.c_void_p(self._handle), ctypes.c_int(int(sid)))

    # Canonical aliases for ergonomic simulator callers.
    evict_idle_kv_to_host = evict_idle_kv
    ensure_session_resident = ensure_resident
    upsert_session_kv = upsert_kv
    execute_dit_and_vae = exec_dit_vae


def _crossover_kv_mib(
    hw: HardwareInterconnectSpec,
    physics: MeasuredPhysicsProfile | None = None,
    workload_scale: WorkloadScaleSpec | None = None,
) -> float:
    ph = MeasuredPhysicsProfile.default() if physics is None else physics
    ws = WorkloadScaleSpec.bagel_7b_standard() if workload_scale is None else workload_scale
    topo = hw.to_topology(max(2, hw.gpus_per_node))
    router = DualPathMorphRouter(topo, overlap_efficiency=0.90)
    return router.analytical_crossover_kv_mib(
        alpha=0.50,
        gen_expert_bytes=int(ph.packed_expert_data_mib * ws.weight_scale * MIB),
        dit_step1_ms=(ph.dit_512_cfg_step_mean_ms * ws.compute_scale) / hw.compute_speedup_x,
        src_gpu=0,
        dst_gpu=1,
    )


def _serialize_custom_sessions(custom_sessions: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for item in custom_sessions:
        if is_dataclass(item) and not isinstance(item, type):
            out.append(asdict(item))
        elif isinstance(item, Mapping):
            out.append(dict(item))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            row: list[Any] = [float(item[0]), str(item[1]), [list(t) for t in item[2]]]
            if len(item) > 3:
                row.append(str(item[3]))
            out.append(row)
        else:
            raise ValueError(f"unsupported custom_sessions entry: {item!r}")
    return out


def simulate_cluster_point(
    policy_name: str,
    num_gpus: int,
    hw: HardwareInterconnectSpec,
    offered_qps: float,
    num_sessions: int = 96,
    seed: int = 2026,
    physics: MeasuredPhysicsProfile | None = None,
    workload_scale: WorkloadScaleSpec | None = None,
    skew_tag: str | None = None,
    *,
    policy_overrides: Mapping[str, Any] | None = None,
    custom_sessions: Sequence[Mapping[str, Any] | tuple[Any, ...]] | None = None,
    bench_root: Path | str | None = None,
    sources_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute zero-bias discrete-event simulation natively in C++17 `ClusterSimulator`."""
    if policy_name not in VALID_POLICIES and policy_name not in STATIC_SPLIT_POLICIES:
        raise ValueError(f"unknown policy: {policy_name}")

    ph = MeasuredPhysicsProfile.from_phase01_results() if physics is None else physics
    ws = WorkloadScaleSpec.bagel_7b_standard() if workload_scale is None else workload_scale
    b_root = DEFAULT_BENCH_ROOT if bench_root is None else Path(bench_root)
    s_dir = DEFAULT_SOURCES_DIR if sources_dir is None else Path(sources_dir)

    req: dict[str, Any] = {
        "policy_name": policy_name,
        "num_gpus": int(num_gpus),
        "hw": asdict(hw),
        "offered_qps": float(offered_qps),
        "num_sessions": int(num_sessions),
        "seed": int(seed),
        "physics": ph.to_dict(),
        "workload_scale": ws.to_dict(),
        "skew_tag": "" if skew_tag is None else str(skew_tag),
        "bench_root": str(b_root),
        "sources_dir": str(s_dir),
    }
    if policy_overrides:
        ser_ov: dict[str, Any] = {}
        for k, v in policy_overrides.items():
            if k == "initial_alphas" and isinstance(v, Mapping):
                ser_ov[k] = {str(rk): float(rv) for rk, rv in v.items()}
            elif isinstance(v, tuple):
                ser_ov[k] = list(v)
            elif isinstance(v, Path):
                ser_ov[k] = str(v)
            else:
                ser_ov[k] = v
        req["policy_overrides"] = ser_ov
    if custom_sessions is not None:
        req["custom_sessions"] = _serialize_custom_sessions(custom_sessions)

    L = get_core_lib()
    return call_json(L, L.ms_sim_point_json, encode_json(req))


def evaluate_against_real_runs(
    real_runs: list[dict[str, Any]],
    physics: MeasuredPhysicsProfile | None = None,
    *,
    bench_root: Path | str | None = None,
    sources_dir: Path | str | None = None,
) -> SimValidationReport:
    """Out-of-sample validation with ZERO parameter grid search via C++17 `ClusterSimulator`."""
    ph = MeasuredPhysicsProfile.from_phase01_results() if physics is None else physics
    b_root = DEFAULT_BENCH_ROOT if bench_root is None else Path(bench_root)
    s_dir = DEFAULT_SOURCES_DIR if sources_dir is None else Path(sources_dir)
    L = get_core_lib()
    req = {
        "real_runs": list(real_runs),
        "physics": ph.to_dict(),
        "bench_root": str(b_root),
        "sources_dir": str(s_dir),
    }
    out = call_json(L, L.ms_sim_evaluate_real_json, encode_json(req))
    return SimValidationReport(**out)


def run_h100_80gb_generality_study(
    physics: MeasuredPhysicsProfile | None = None,
    *,
    bench_root: Path | str | None = None,
    sources_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate H100-80GB NVLink generality natively in C++17 `ClusterSimulator`."""
    ph = MeasuredPhysicsProfile.from_phase01_results() if physics is None else physics
    b_root = DEFAULT_BENCH_ROOT if bench_root is None else Path(bench_root)
    s_dir = DEFAULT_SOURCES_DIR if sources_dir is None else Path(sources_dir)
    L = get_core_lib()
    req = {
        "physics": ph.to_dict(),
        "bench_root": str(b_root),
        "sources_dir": str(s_dir),
    }
    return call_json(L, L.ms_sim_h100_study_json, encode_json(req))


def run_scale_and_interconnect_sweep(
    output_dir: Path,
    real_report_path: Path | None = None,
    results_dir: Path | None = None,
    *,
    bench_root: Path | str | None = None,
    sources_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run full 8->16->64 GPU & H100-80GB generality sweep natively in C++17 `ClusterSimulator`."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir = DEFAULT_RESULTS_DIR if results_dir is None else Path(results_dir)
    b_root = DEFAULT_BENCH_ROOT if bench_root is None else Path(bench_root)
    s_dir = DEFAULT_SOURCES_DIR if sources_dir is None else Path(sources_dir)
    L = get_core_lib()
    req = {
        "output_dir": str(out_dir),
        "real_report_path": "" if real_report_path is None else str(real_report_path),
        "results_dir": str(res_dir),
        "bench_root": str(b_root),
        "sources_dir": str(s_dir),
    }
    return call_json(L, L.ms_sim_sweep_json, encode_json(req))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MorphServe Zero-Bias Physics-Driven Cluster Simulator")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-report", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--bench-root", type=Path, default=None)
    parser.add_argument("--sources-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    rep = run_scale_and_interconnect_sweep(
        args.output_dir,
        real_report_path=args.real_report,
        results_dir=args.results_dir,
        bench_root=args.bench_root,
        sources_dir=args.sources_dir,
    )
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
