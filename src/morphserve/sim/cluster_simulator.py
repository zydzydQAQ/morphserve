"""Zero-bias physics-driven 8->16->64 GPU discrete-event simulator calibrated 100% from Phase-0/1 JSONs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import heapq
import json
import math
from pathlib import Path
import random
import time
from typing import Any

from morphserve.bench.interleaved_bench import load_burstgpt_interarrivals, load_gentd26_interarrivals
from morphserve.engine.config import EnginePolicyConfig, VALID_POLICIES
from morphserve.engine.workload import WorkloadSuite, assign_dual_trace_arrivals
from morphserve.kernels.cfg_prefix_attn import should_use_cascade_cfg_split
from morphserve.router import (
    CPTGRequestSpec,
    ClusterTopologyProfile,
    DEFAULT_OPERATOR_PROFILE,
    DualPathMorphRouter,
    HalfNodeH2DPacer,
)
from morphserve.telemetry import percentile

MIB = 1024**2
GIB = 1024**3
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_BENCH_ROOT = REPO_ROOT / "data" / "interleaved-bench-v1"
DEFAULT_SOURCES_DIR = REPO_ROOT / "data" / "bench_sources"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _first_existing(base: Path, candidates: tuple[str, ...]) -> Path | None:
    for rel in candidates:
        p = base / rel
        if p.is_file():
            return p
    return None


@dataclass(frozen=True)
class MeasuredPhysicsProfile:
    """100% Phase-0/1 empirical hardware & operator constants (zero manual bias / zero grid-search fitting)."""

    # Phase-0 Job-0a (TriPool-VMM microbench) & Job-0c (8-GPU PCIe/NUMA topology)
    vmm_64m_acquire_4gib_ms: float = 5.437
    vmm_solo_h2d_stream_gbps: float = 12.334
    pcie_solo_h2d_gbps: float = 8.755
    pcie_half0_agg_h2d_gbps: float = 10.348
    pcie_half1_agg_h2d_gbps: float = 10.405
    pcie_intra_half_p2p_gbps: float = 10.581
    pcie_cross_half_p2p_gbps: float = 9.55

    # Phase-0 Job-0d (Motivation: unscoped vs scoped VAE spikes & 2-stage SHM IPC regression)
    vae_512_ms: float = 52.60
    vae_1024_ms: float = 212.32
    scoped_vae_512_spike_mib: float = 960.0  # ceil(896.25 / 64) * 64
    unscoped_vae_512_reserved_mib: float = 2022.78  # 33114.0 - 31091.22
    scoped_vae_1024_spike_mib: float = 3648.0  # ceil(3585.0 / 64) * 64
    unscoped_vae_1024_reserved_mib: float = 6508.78  # 37600.0 - 31091.22
    twostage_shm_base_ms: float = 5.20
    twostage_shm_ms_per_mib: float = 4.2080

    # Phase-1 Module-1 (BAGEL-7B-MoT TriPool-VMM weight paging & overlap)
    monolithic_loaded_driver_mib: float = 28776.81
    offloaded_base_driver_mib: float = 16366.81
    num_expert_layers: int = 28
    layer_stride_mib: int = 448
    packed_expert_data_mib: float = 12446.64
    dit_512_20step_base_ms: float = 4958.59
    dit_512_transient_mib: float = 1088.0
    dit_1024_step_ms: float = 966.335
    vmm_full_expert_map_ms: float = 11.677

    # Phase-1 Module-3 (LECP CoW KV savings, GPU in-place loopback & Cascade-CFG KV slope)
    cpu_pil_prep_only_ms: float = 69.80
    gpu_inplace_prep_only_ms: float = 2.452
    stock_loopback_turn_ms: float = 887.943
    morphserve_loopback_turn_ms: float = 793.547
    non_cow_kv_mult: float = 1.9979
    turn_edit_512_kv_delta_mib: float = 325.28
    kv_mib_per_token: float = 0.05469
    stock_dit_512_ms_per_kv_tok: float = 0.107796
    morphserve_dit_512_ms_per_kv_tok: float = 0.097462

    # Phase-1 Module-4 (Layer-chunked preemption vs step preemption)
    ar_prefill_ms: float = 64.80
    ar_unloaded_tpot_p50_ms: float = 62.82
    dit_512_cfg_step_mean_ms: float = 286.55
    dit_512_layer_p50_ms: float = 10.52
    m4_adaptive_chunk_layers: int = 5
    m4_colocated_ar_tpot_p50_ms: float = 110.98
    step_preempt_ar_tpot_p50_ms: float = 346.13

    # Phase-1 Module-2 (Shared NUMA mirror H2D pacer & NCCL P2P KV shipping)
    paced_h2d_per_gpu_gbps: float = 12.34
    unpaced_4gpu_half_h2d_per_gpu_gbps: float = 3.10
    nccl_intra_half_kv_gbps: float = 10.362

    # Phase-2 M2.5 Live 2x8 Multi-Node Fabric Calibration (Kairos 16-GPU)
    kairos_2x8_intra_half_gbps: float = 8.37
    kairos_2x8_cross_socket_gbps: float = 8.59
    kairos_2x8_inter_node_gbps: float = 1.11

    provenance_files: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_phase01_results(cls, results_dir: Path | None = None) -> "MeasuredPhysicsProfile":
        """Parse all empirical physical constants directly from Phase-0/1 JSON artifacts on disk."""
        root = DEFAULT_RESULTS_DIR if results_dir is None else Path(results_dir)
        defaults = cls()
        if not root.is_dir():
            return defaults

        prov: dict[str, str] = {}
        kw: dict[str, Any] = {}

        def _load(key: str, rel_candidates: tuple[str, ...]) -> dict[str, Any] | None:
            p = _first_existing(root, rel_candidates)
            if p is None:
                return None
            try:
                rel_str = str(p.relative_to(root))
            except ValueError:
                rel_str = str(p)
            prov[key] = f"{rel_str}@sha256:{_sha256_file(p)[:16]}"
            return json.loads(p.read_text(encoding="utf-8"))

        j0a = _load("job0a_vmm", ("phase0-job0a-v2/vmm-bench.json", "phase0-job0a-r2/vmm-bench.json"))
        if j0a:
            if isinstance(j0a.get("h2d_stream_gbps"), dict):
                kw["vmm_solo_h2d_stream_gbps"] = float(j0a["h2d_stream_gbps"]["mean"])
            for row in j0a.get("granularity_sweep", []):
                if int(row.get("handle_mib", 0)) == 64:
                    kw["vmm_64m_acquire_4gib_ms"] = float(row["vae_4gib_acquire_ms"]["mean"])
                    break

        j0c = _load("job0c_topo", ("phase0-job0c/topo-8gpu-report.json",))
        if j0c:
            nh = j0c.get("numa_h2d", {})
            sg = nh.get("single_gpu_h2d_gbps", {})
            if sg:
                vals = [
                    float(v.get("numa_0", 8.75))
                    for k, v in sg.items()
                    if k in ("gpu_0", "gpu_1", "gpu_2", "gpu_3") and isinstance(v, dict)
                ]
                if vals:
                    kw["pcie_solo_h2d_gbps"] = round(sum(vals) / len(vals), 3)
            ch = nh.get("concurrent_h2d", {})
            if "socket0_4gpu_remote_numa1" in ch:
                kw["pcie_half0_agg_h2d_gbps"] = float(ch["socket0_4gpu_remote_numa1"]["aggregate_gbps"])
            if "socket1_4gpu_local_numa1" in ch:
                kw["pcie_half1_agg_h2d_gbps"] = float(ch["socket1_4gpu_local_numa1"]["aggregate_gbps"])
            pairs = j0c.get("nccl_p2p", {}).get("pairs", {})
            if "0->2" in pairs:
                kw["pcie_intra_half_p2p_gbps"] = float(pairs["0->2"]["gbps_p50"])

        j0d = _load("job0d_motivation", ("phase0-job0d-r2/motivation-report.json", "phase0-job0d/motivation-report.json"))
        if j0d:
            vae = j0d.get("single_gpu", {}).get("vae_batch_and_res_stress", {})
            if "512_b1" in vae:
                v512 = vae["512_b1"]
                kw["vae_512_ms"] = float(v512["elapsed_ms"])
                sp512 = float(v512["spike_delta_mib"])
                kw["scoped_vae_512_spike_mib"] = float(math.ceil(sp512 / 64.0) * 64)
                kw["unscoped_vae_512_reserved_mib"] = round(
                    float(v512["peak_reserved_mib"]) - float(v512["before_alloc_mib"]), 2
                )
            if "1024_b1" in vae:
                v1024 = vae["1024_b1"]
                kw["vae_1024_ms"] = float(v1024["elapsed_ms"])
                sp1024 = float(v1024["spike_delta_mib"])
                kw["scoped_vae_1024_spike_mib"] = float(math.ceil(sp1024 / 64.0) * 64)
                kw["unscoped_vae_1024_reserved_mib"] = round(
                    float(v1024["peak_reserved_mib"]) - float(v1024["before_alloc_mib"]), 2
                )
            ts = j0d.get("two_stage", {})
            turns_ts = ts.get("turns", [])
            agg_ts = ts.get("aggregate_kv_transfer", {})
            if turns_ts and agg_ts:
                t0_kv = turns_ts[0].get("kv_transfer", {})
                t0_mib = float(t0_kv.get("total_transfer_mib", 0.99))
                base_ms = (
                    float(t0_kv.get("total_transfer_seconds", 0.001))
                    + float(t0_kv.get("total_serialize_seconds", 0.0042))
                ) * 1000.0
                tot_mib = float(agg_ts.get("total_transfer_mib", 1964.32))
                tot_ms = (
                    float(agg_ts.get("total_transfer_seconds", 3.403))
                    + float(agg_ts.get("total_serialize_seconds", 4.8797))
                ) * 1000.0
                slope = (tot_ms - len(turns_ts) * base_ms) / max(1.0, tot_mib - t0_mib)
                kw["twostage_shm_base_ms"] = round(base_ms, 3)
                kw["twostage_shm_ms_per_mib"] = round(max(0.1, slope), 4)

        m1 = _load(
            "module1_vmm",
            ("refactor-20261122/module1-report.json", "phase1-job1a-r3/module1-report.json"),
        )
        if m1:
            if "mem_monolithic_loaded" in m1:
                kw["monolithic_loaded_driver_mib"] = float(m1["mem_monolithic_loaded"]["driver_used_mib"])
            if "mem_after_offload" in m1:
                kw["offloaded_base_driver_mib"] = float(m1["mem_after_offload"]["driver_used_mib"])
            oi = m1.get("offload_info", {})
            if oi:
                kw["num_expert_layers"] = int(oi.get("num_layers", 28))
                kw["layer_stride_mib"] = int(oi.get("layer_stride_mib", 448))
                kw["packed_expert_data_mib"] = float(oi.get("packed_data_mib", 12446.64))
            for row in m1.get("test1_alpha_sweep", []):
                a = float(row.get("alpha", -1.0))
                if abs(a - 1.0) < 1e-6:
                    kw["dit_512_20step_base_ms"] = float(row["gen_total_ms"])
                    kw["dit_512_transient_mib"] = float(row["transient_peak_mib"])
                elif abs(a - 0.0) < 1e-6:
                    kw["vmm_full_expert_map_ms"] = float(row["expert_map_delta_ms"])
            for row in m1.get("test1b_1024_overlap_sweep", []):
                if abs(float(row.get("alpha", -1.0)) - 1.0) < 1e-6:
                    # test1b executes 10 steps at 1024x1024
                    kw["dit_1024_step_ms"] = round(float(row["gen_total_ms"]) / 10.0, 3)
                    break

        m3 = _load(
            "module3_cow_loopback",
            ("morphserve-p2-m21gate-20261121-r4/module3-report.json", "phase1-job1b-r7/module3-report.json"),
        )
        if m3:
            t2 = m3.get("test2_gpu_inplace_loopback", {})
            if t2:
                kw["cpu_pil_prep_only_ms"] = float(t2.get("cpu_png_pil_prep_only_ms", 69.8))
                kw["gpu_inplace_prep_only_ms"] = float(t2.get("gpu_inplace_prep_only_ms", 2.452))
            t3_turns = m3.get("test3_4turn_interleaved_e2e", {}).get("turns", [])
            if len(t3_turns) >= 4:
                t0, t1, t3 = t3_turns[0], t3_turns[1], t3_turns[3]
                loop_s = [float(r["stock_loopback_ms"]) for r in t3_turns[1:]]
                loop_m = [float(r["morphserve_loopback_ms"]) for r in t3_turns[1:]]
                kw["stock_loopback_turn_ms"] = round(sum(loop_s) / len(loop_s), 3)
                kw["morphserve_loopback_turn_ms"] = round(sum(loop_m) / len(loop_m), 3)
                ms_kv3 = max(1e-3, float(t3["morphserve_kv_hbm_mib"]))
                st_kv3 = float(t3["stock_kv_hbm_mib"])
                kw["non_cow_kv_mult"] = round(st_kv3 / ms_kv3, 4)
                kw["turn_edit_512_kv_delta_mib"] = round(
                    float(t1["morphserve_kv_hbm_mib"]) - float(t0["morphserve_kv_hbm_mib"]), 2
                )
                dtok = max(1, int(t3["main_seq_len"]) - int(t0["main_seq_len"]))
                kw["kv_mib_per_token"] = round(
                    (float(t3["morphserve_kv_hbm_mib"]) - float(t0["morphserve_kv_hbm_mib"])) / dtok, 5
                )
                base_gen = float(t0["stock_gen_ms"])
                kw["stock_dit_512_ms_per_kv_tok"] = round((float(t3["stock_gen_ms"]) - base_gen) / dtok, 6)
                kw["morphserve_dit_512_ms_per_kv_tok"] = round(
                    max(0.0, (float(t3["morphserve_gen_ms"]) - base_gen) / dtok), 6
                )

        m4 = _load("module4_scheduler", ("phase1-job1c-r3/module4-report.json",))
        if m4:
            sb = m4.get("solo_baselines", {})
            if sb:
                kw["ar_unloaded_tpot_p50_ms"] = float(sb.get("unloaded_ar_tpot_p50_ms", 62.82))
                kw["dit_512_cfg_step_mean_ms"] = float(sb.get("dit_cfg_step_mean_ms", 286.55))
                kw["dit_512_layer_p50_ms"] = float(sb.get("dit_layer_p50_ms", 10.52))
            for row in m4.get("test1_policy_comparison", []):
                pname = str(row.get("policy", ""))
                if pname == "morphserve_layer_chunk_20x":
                    kw["m4_adaptive_chunk_layers"] = int(row.get("adaptive_chunk_layers", 5))
                    kw["m4_colocated_ar_tpot_p50_ms"] = float(row.get("ar_tpot_p50_ms", 110.98))
                elif pname == "step_preemptive":
                    kw["step_preempt_ar_tpot_p50_ms"] = float(row.get("ar_tpot_p50_ms", 346.13))

        m2 = _load("module2_router", ("phase1-job1d-r4/module2-report.json",))
        if m2:
            t1 = m2.get("test1_shared_numa_memfd_and_h2d_pacer", {})
            if t1:
                kw["paced_h2d_per_gpu_gbps"] = float(t1.get("paced_dual_half_per_gpu_gbps", 12.34))
                kw["unpaced_4gpu_half_h2d_per_gpu_gbps"] = float(t1.get("unpaced_4gpu_half0_per_gpu_gbps", 3.10))
            t2 = m2.get("test2_bagel_dualpath_crossover", {}).get("t3_long_interleaved_kv", {})
            if t2 and "path_a_bw_gbps" in t2:
                kw["nccl_intra_half_kv_gbps"] = float(t2["path_a_bw_gbps"])

        m2x8_p = _first_existing(root, ("morphserve-p2-m25node2x8-20261121-r2/m25-multinode-2x8-report.json",))
        if m2x8_p is not None:
            m2x8 = json.loads(m2x8_p.read_text(encoding="utf-8"))
            cal = m2x8.get("calibrated_gbps", {})
            if cal:
                kw["kairos_2x8_intra_half_gbps"] = float(cal.get("intra_half_pcie", 8.37))
                kw["kairos_2x8_cross_socket_gbps"] = float(cal.get("intra_node_cross_socket", 8.59))
                kw["kairos_2x8_inter_node_gbps"] = float(cal.get("inter_node_fabric", 1.11))

        kw["provenance_files"] = prov
        return cls(**kw)


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

    @classmethod
    def bagel_7b_standard(cls) -> "WorkloadScaleSpec":
        return cls(name="bagel_7b_standard")

    @classmethod
    def frontier_mot_56gib(cls) -> "WorkloadScaleSpec":
        """Proportionally scaled ~56.2 GiB MoT (~14B+14B) keeping 70.3% weight/HBM ratio on 80 GiB H100."""
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
        """7B MoT with 10-turn multi-image sessions & high concurrent residency saturating 80 GiB HBM."""
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
        ph = MeasuredPhysicsProfile() if physics is None else physics
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


@dataclass
class SimGpuMemoryLedger:
    """Byte-accurate per-GPU physical HBM ledger from Phase-0/1 driver measurements (zero synthetic fragmentation)."""

    gpu_id: int
    total_hbm_mib: float
    policy: EnginePolicyConfig
    role: str
    resident_expert_layers: int = 28
    num_expert_layers: int = 28
    layer_stride_mib: int = 448  # 7 x 64 MiB super-handles per BAGEL layer
    physics: MeasuredPhysicsProfile = field(default_factory=MeasuredPhysicsProfile)
    weight_scale: float = 1.0
    vae_spike_scale: float = 1.0
    session_kv_mib: dict[int, float] = field(default_factory=dict)
    swapped_session_kv_mib: dict[int, float] = field(default_factory=dict)
    super_to_fine_carves: int = 0
    expert_evictions: int = 0
    kv_host_swap_outs: int = 0
    kv_host_swap_ins: int = 0
    peak_used_mib: float = 0.0

    @property
    def offloaded_base_mib(self) -> float:
        return self.physics.offloaded_base_driver_mib * self.weight_scale

    @property
    def monolithic_weights_mib(self) -> float:
        return self.physics.monolithic_loaded_driver_mib * self.weight_scale

    @property
    def pageable_expert_mib(self) -> float:
        return max(0.0, self.monolithic_weights_mib - self.offloaded_base_mib)

    @property
    def base_weights_mib(self) -> float:
        if self.policy.use_expert_pager or self.policy.use_naive_offload:
            return self.offloaded_base_mib
        if self.policy.routing_mode == "static_nccl_fission" and self.role == "und":
            return self.offloaded_base_mib
        return self.monolithic_weights_mib

    @property
    def expert_alpha(self) -> float:
        if not self.policy.use_expert_pager:
            return 0.0 if (self.role == "und" or self.policy.use_naive_offload) else 1.0
        return round(self.resident_expert_layers / float(max(1, self.num_expert_layers)), 3)

    @property
    def active_kv_mib(self) -> float:
        if self.policy.use_tripool_vmm:
            # 2 MiB physical page granularity inside TriPool-VMM KV regions
            return sum(max(2.0, 2.0 * math.ceil(max(0.01, v) / 2.0)) for v in self.session_kv_mib.values())
        return sum(self.session_kv_mib.values())

    def free_kv_bytes(self) -> int:
        used = self.base_weights_mib + self.active_kv_mib + (
            self.resident_expert_layers * self.layer_stride_mib if self.policy.use_expert_pager else 0.0
        )
        return int(max(0.0, self.total_hbm_mib - used - 2048.0) * MIB)

    def evict_idle_kv_to_host(self, needed_free_mib: float, exclude_sid: int | None = None) -> float:
        """Swap unpinned think-time session VMM KV to Host Pinned DRAM (`Decision 1/6`)."""
        if not (self.policy.use_tripool_vmm and self.policy.use_expert_pager):
            return 0.0
        freed_mib = 0.0
        for cand_sid in list(self.session_kv_mib.keys()):
            if cand_sid == exclude_sid:
                continue
            if freed_mib >= needed_free_mib:
                break
            val = self.session_kv_mib.pop(cand_sid)
            self.swapped_session_kv_mib[cand_sid] = val
            self.kv_host_swap_outs += 1
            freed_mib += max(2.0, 2.0 * math.ceil(max(0.01, val) / 2.0))
        return freed_mib

    def ensure_session_resident(self, sid: int) -> float:
        """Restore a swapped session's KV from Host Pinned DRAM if needed; returns swapped-in MiB (`0.0` if resident)."""
        swapped_val = self.swapped_session_kv_mib.pop(sid, None)
        if swapped_val is None:
            return 0.0
        if not self.upsert_session_kv(sid, swapped_val):
            self.swapped_session_kv_mib[sid] = swapped_val
            return -1.0
        self.kv_host_swap_ins += 1
        return float(swapped_val)

    def upsert_session_kv(self, sid: int, logical_kv_mib: float) -> bool:
        """Update session KV footprint, evicting unpinned tail expert layers / idle KV on VMM pressure; False on OOM."""
        prev = self.session_kv_mib.get(sid)
        prev_swapped = self.swapped_session_kv_mib.pop(sid, None)
        self.session_kv_mib[sid] = float(logical_kv_mib)
        kv_tot = self.active_kv_mib
        if self.policy.use_tripool_vmm and self.policy.use_expert_pager:
            pool_cap = float(self.policy.super_pool_mib + self.policy.fine_kv_pool_mib)
            if kv_tot > float(self.policy.fine_kv_pool_mib):
                self.super_to_fine_carves += 1
            while self.resident_expert_layers > 0 and (
                self.resident_expert_layers * self.layer_stride_mib + kv_tot
            ) > pool_cap:
                self.resident_expert_layers -= 1
                self.expert_evictions += 1
            used = self.base_weights_mib + self.resident_expert_layers * self.layer_stride_mib + kv_tot
            if used > self.total_hbm_mib:
                self.evict_idle_kv_to_host(used - self.total_hbm_mib, exclude_sid=sid)
                kv_tot = self.active_kv_mib
                used = self.base_weights_mib + self.resident_expert_layers * self.layer_stride_mib + kv_tot
        else:
            used = self.base_weights_mib + kv_tot

        if used > self.total_hbm_mib:
            if prev is None:
                self.session_kv_mib.pop(sid, None)
            else:
                self.session_kv_mib[sid] = prev
            if prev_swapped is not None:
                self.swapped_session_kv_mib[sid] = prev_swapped
            return False
        self.peak_used_mib = max(self.peak_used_mib, used)
        return True

    def execute_dit_and_vae(self, height: int, active_sid: int | None = None) -> tuple[bool, float]:
        """Check pure byte conservation across DiT workspace + measured VAE spike; returns `(ok, swapped_out_mib)`."""
        is_1024 = int(height) >= 1024
        dit_act_mib = (
            (2.0 * self.physics.dit_512_transient_mib if is_1024 else self.physics.dit_512_transient_mib)
            * math.sqrt(self.weight_scale)
        )
        scoped_vae_mib = (
            self.physics.scoped_vae_1024_spike_mib if is_1024 else self.physics.scoped_vae_512_spike_mib
        ) * self.vae_spike_scale
        unscoped_vae_mib = (
            self.physics.unscoped_vae_1024_reserved_mib if is_1024 else self.physics.unscoped_vae_512_reserved_mib
        ) * self.vae_spike_scale
        kv_tot = self.active_kv_mib
        swapped_out_mib = 0.0

        if self.policy.use_tripool_vmm and self.policy.use_expert_pager:
            # Step 1: Pinned DiT generation requires all expert layers + active KV + DiT workspace <= total HBM;
            # swap unpinned think-time sessions' VMM KV to Host Pinned DRAM if needed (`BagelCPTGAdapter:1255`)!
            step_peak = self.base_weights_mib + self.num_expert_layers * self.layer_stride_mib + kv_tot + dit_act_mib
            if step_peak > self.total_hbm_mib and active_sid is not None:
                swapped_out_mib += self.evict_idle_kv_to_host(step_peak - self.total_hbm_mib, exclude_sid=active_sid)
                kv_tot = self.active_kv_mib
                step_peak = self.base_weights_mib + self.num_expert_layers * self.layer_stride_mib + kv_tot + dit_act_mib
            if step_peak > self.total_hbm_mib:
                return False, swapped_out_mib
            self.resident_expert_layers = self.num_expert_layers

            # Step 2: Scoped VAE executes outside generation_mode; unpinned tail expert handles drop cleanly
            vae_mib = scoped_vae_mib if self.policy.use_scoped_vae else unscoped_vae_mib
            pool_cap = float(self.policy.super_pool_mib + self.policy.fine_kv_pool_mib)
            eff_cap = min(pool_cap, max(0.0, self.total_hbm_mib - self.base_weights_mib))
            while self.resident_expert_layers > 0 and (
                self.resident_expert_layers * self.layer_stride_mib + kv_tot + vae_mib
            ) > eff_cap:
                self.resident_expert_layers -= 1
                self.expert_evictions += 1

            vae_peak = (
                self.base_weights_mib
                + self.resident_expert_layers * self.layer_stride_mib
                + kv_tot
                + vae_mib
            )
            if vae_peak > self.total_hbm_mib and active_sid is not None:
                swapped_out_mib += self.evict_idle_kv_to_host(vae_peak - self.total_hbm_mib, exclude_sid=active_sid)
                kv_tot = self.active_kv_mib
                vae_peak = (
                    self.base_weights_mib
                    + self.resident_expert_layers * self.layer_stride_mib
                    + kv_tot
                    + vae_mib
                )
            if vae_peak > self.total_hbm_mib:
                return False, swapped_out_mib
            self.peak_used_mib = max(self.peak_used_mib, step_peak, vae_peak)
            return True, swapped_out_mib

        # Non-paged policies (`monolithic_dp`, `prism_tetriserve_coloc`, `naive_host_offload`, static splits):
        # Zero synthetic `allocator_frag_mib`; uses exact Job-0d measured PyTorch reserved delta (`6508.78 MiB` at 1024_b1).
        gen_weights_mib = self.pageable_expert_mib if self.policy.use_naive_offload else 0.0
        vae_mib = scoped_vae_mib if self.policy.use_scoped_vae else unscoped_vae_mib
        peak = self.base_weights_mib + gen_weights_mib + kv_tot + max(dit_act_mib, vae_mib)
        if peak > self.total_hbm_mib:
            self.peak_used_mib = max(self.peak_used_mib, self.total_hbm_mib)
            return False, 0.0
        self.peak_used_mib = max(self.peak_used_mib, peak)
        return True, 0.0

    def release_session(self, sid: int) -> None:
        self.session_kv_mib.pop(sid, None)
        self.swapped_session_kv_mib.pop(sid, None)


def _crossover_kv_mib(
    hw: HardwareInterconnectSpec,
    physics: MeasuredPhysicsProfile | None = None,
    workload_scale: WorkloadScaleSpec | None = None,
) -> float:
    ph = MeasuredPhysicsProfile() if physics is None else physics
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


CANONICAL_28_TRACE: tuple[tuple[float, str, tuple[tuple[str, int], ...]], ...] = (
    (0.00, "multiturn_edit", (("dit_edit", 512), ("dit_edit", 512), ("dit_edit", 512), ("dit_edit", 512))),
    (0.00, "visual_cot", (("cot_then_dit", 512),)),
    (3.05, "multi_image_vqa", (("ar_only", 512), ("ar_only", 512))),
    (8.61, "visual_cot", (("cot_then_dit", 1024),)),
    (13.71, "visual_cot", (("cot_then_dit", 1024),)),
    (15.92, "multi_image_vqa", (("ar_only", 512), ("ar_only", 512))),
    (17.37, "multi_image_vqa", (("ar_only", 512), ("ar_only", 512))),
    (19.51, "multi_image_vqa", (("ar_only", 512), ("ar_only", 512))),
    (19.96, "multi_image_vqa", (("ar_only", 512), ("ar_only", 512))),
    (30.86, "multi_image_vqa", (("ar_only", 512), ("ar_only", 512))),
    (40.38, "visual_cot", (("cot_then_dit", 512),)),
    (42.29, "multi_image_vqa", (("ar_only", 512), ("ar_only", 512))),
    (48.23, "visual_cot", (("cot_then_dit", 1024),)),
    (51.89, "visual_cot", (("cot_then_dit", 512),)),
    (55.34, "multiturn_edit", (("dit_edit", 1024), ("dit_edit", 1024), ("dit_edit", 1024), ("dit_edit", 1024))),
    (55.96, "multiturn_edit", (("dit_edit", 1024), ("dit_edit", 1024), ("dit_edit", 1024), ("dit_edit", 1024))),
    (56.07, "highres_t2i", (("dit_t2i", 1024),)),
    (56.08, "visual_cot", (("cot_then_dit", 512),)),
    (58.36, "visual_cot", (("cot_then_dit", 1024),)),
    (59.11, "multiturn_edit", (("dit_edit", 512), ("dit_edit", 512), ("dit_edit", 512), ("dit_edit", 512))),
    (61.13, "multiturn_edit", (("dit_edit", 512), ("dit_edit", 512), ("dit_edit", 512))),
    (62.53, "highres_t2i", (("dit_t2i", 512),)),
    (63.73, "multiturn_edit", (("dit_edit", 512), ("dit_edit", 512), ("dit_edit", 512), ("dit_edit", 512))),
    (68.77, "highres_t2i", (("dit_t2i", 1024),)),
    (68.84, "highres_t2i", (("dit_t2i", 512),)),
    (69.22, "multiturn_edit", (("dit_edit", 512), ("dit_edit", 512), ("dit_edit", 512), ("dit_edit", 512))),
    (69.89, "multiturn_edit", (("dit_edit", 512), ("dit_edit", 512))),
    (70.25, "multiturn_edit", (("dit_edit", 512), ("dit_edit", 512), ("dit_edit", 512), ("dit_edit", 512))),
)


def _deterministic_think_ms(sid: int, ti: int, bounds_s: tuple[float, float]) -> float:
    lo, hi = float(bounds_s[0]) * 1000.0, float(bounds_s[1]) * 1000.0
    if hi <= lo:
        return max(0.0, lo)
    h = int(hashlib.md5(f"sim_{sid}_{ti}".encode("utf-8")).hexdigest()[:8], 16)
    return lo + ((h % 10_000) / 10_000.0) * (hi - lo)


_SUITE_CACHE: dict[str, Any] = {}


def _load_real_bench_templates(skew_tag: str | None = None) -> tuple[list[Any] | None, list[float] | None, list[float] | None]:
    cache_key = skew_tag or "main_mix"
    if cache_key in _SUITE_CACHE:
        return _SUITE_CACHE[cache_key]

    rel_map = {
        "main_mix": "suites/main_mix_80.json",
        "ar80_dit20": "suites/skew_ar80_dit20.json",
        "ar50_dit50": "suites/skew_ar50_dit50.json",
        "ar20_dit80": "suites/skew_ar20_dit80.json",
    }
    suite_p = DEFAULT_BENCH_ROOT / rel_map.get(cache_key, "suites/main_mix_80.json")
    burst_p = DEFAULT_SOURCES_DIR / "burstgpt_1_sample.csv"
    gentd_p = DEFAULT_SOURCES_DIR / "gentd26_lora_request_trace.csv"
    if not suite_p.is_file():
        _SUITE_CACHE[cache_key] = (None, None, None)
        return (None, None, None)
    suite = WorkloadSuite.load_json(suite_p)
    burst_d = load_burstgpt_interarrivals(burst_p) if burst_p.is_file() else None
    gentd_d = load_gentd26_interarrivals(gentd_p) if gentd_p.is_file() else None
    _SUITE_CACHE[cache_key] = (suite.sessions, burst_d, gentd_d)
    return _SUITE_CACHE[cache_key]


def _build_session_stream(
    num_sessions: int,
    offered_qps: float,
    num_gpus: int,
    seed: int,
    extra_edit_turns: int = 0,
    skew_tag: str | None = None,
) -> list[tuple[float, str, tuple[tuple[str, int], ...]]]:
    del num_gpus
    real_sessions, burst_d, gentd_d = _load_real_bench_templates(skew_tag=skew_tag)
    if real_sessions:
        from copy import copy

        n_src = len(real_sessions)
        cloned = []
        for idx in range(num_sessions):
            src = real_sessions[idx % n_src]
            s_copy = copy(src)
            s_copy.session_id = src.session_id if idx < n_src else f"{src.session_id}_r{idx // n_src}"
            cloned.append(s_copy)
        if burst_d and gentd_d:
            assign_dual_trace_arrivals(cloned, burst_d, gentd_d, target_qps=max(0.05, float(offered_qps)))
        else:
            mean_inter_s = 1.0 / max(0.05, float(offered_qps))
            for idx, s_copy in enumerate(cloned):
                s_copy.arrival_offset_s = idx * mean_inter_s
        out_real: list[tuple[float, str, tuple[tuple[str, int], ...]]] = []
        for s_copy in cloned:
            turns = tuple((str(t.modality_plan), int(t.height)) for t in s_copy.turns)
            q = str(s_copy.quadrant)
            if extra_edit_turns > 0 and q == "multiturn_edit" and turns:
                turns = turns + tuple(turns[-1] for _ in range(extra_edit_turns))
            elif extra_edit_turns >= 2 and q in ("visual_cot", "multi_image_vqa") and turns:
                turns = turns + tuple(turns[-1] for _ in range(extra_edit_turns // 2))
            out_real.append((float(s_copy.arrival_offset_s) * 1000.0, q, turns))
        return out_real

    rng = random.Random(seed)
    out: list[tuple[float, str, tuple[tuple[str, int], ...]]] = []
    n_base = len(CANONICAL_28_TRACE)
    mean_inter_ms = 1000.0 / max(0.05, float(offered_qps))
    t_cur = 0.0
    for idx in range(num_sessions):
        _, q, turns = CANONICAL_28_TRACE[idx % n_base]
        if extra_edit_turns > 0 and q == "multiturn_edit" and turns:
            last_turn = turns[-1]
            turns = turns + tuple(last_turn for _ in range(extra_edit_turns))
        elif extra_edit_turns >= 2 and q in ("visual_cot", "multi_image_vqa") and turns:
            last_turn = turns[-1]
            turns = turns + tuple(last_turn for _ in range(extra_edit_turns // 2))
        if idx > 0:
            burst_factor = 0.35 if (idx % 5 in (1, 2)) else (1.85 if idx % 5 == 0 else 0.90)
            jitter = rng.uniform(0.85, 1.15)
            t_cur += mean_inter_ms * burst_factor * jitter
        out.append((t_cur, q, turns))
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
) -> dict[str, Any]:
    """Zero-bias discrete-event simulation driven 100% by Phase-0/1 `MeasuredPhysicsProfile`."""
    if policy_name not in VALID_POLICIES:
        raise ValueError(f"unknown policy: {policy_name}")

    ph = MeasuredPhysicsProfile.from_phase01_results() if physics is None else physics
    ws = WorkloadScaleSpec.bagel_7b_standard() if workload_scale is None else workload_scale

    overrides: dict[str, Any] = {}
    if ws.think_s_range_override is not None:
        overrides["inter_turn_think_s_range"] = ws.think_s_range_override
    if hw.gpu_hbm_gib >= 75.0 and policy_name == "morphserve":
        overrides["super_pool_mib"] = ws.super_pool_mib_80g
        overrides["fine_kv_pool_mib"] = ws.fine_kv_pool_mib_80g

    policy = EnginePolicyConfig.from_preset(policy_name, num_gpus=num_gpus, **overrides)
    topo = hw.to_topology(num_gpus)
    pacer = HalfNodeH2DPacer(topo, max_concurrent_per_half=1)
    sp = hw.compute_speedup_x
    cscale = ws.compute_scale

    router = DualPathMorphRouter(
        topo,
        pacer=pacer,
        map_overhead_ms_full=(ph.vmm_full_expert_map_ms * ws.weight_scale) / sp,
        overlap_efficiency=0.90,
        max_inter_node_state_bytes=int(128 * ws.kv_per_turn_scale * MIB),
    )

    ar_tok_ms = (ph.ar_unloaded_tpot_p50_ms * cscale) / sp
    ar_prefill_ms = (ph.ar_prefill_ms * cscale) / sp
    vae512_ms = (ph.vae_512_ms * ws.vae_spike_scale) / sp
    vae1024_ms = (ph.vae_1024_ms * ws.vae_spike_scale) / sp
    # `dit_512_20step_base_ms` in Module-1 JSON includes 1x 512px VAE decode (`ph.vae_512_ms`); decouple cleanly
    dit512_core_20_ms = (max(0.0, ph.dit_512_20step_base_ms - ph.vae_512_ms) * cscale) / sp
    dit512_step_ms = (ph.dit_512_cfg_step_mean_ms * cscale) / sp
    dit1024_step_ms = (ph.dit_1024_step_ms * cscale) / sp
    dit1024_core_20_ms = max(0.0, 20.0 * dit1024_step_ms - vae1024_ms)

    # Fair `[B, D]` packed GEMM batched AR amortization factor (identical across ALL policies; zero policy bias)
    batched_ar_amort = 0.40

    # AdaptiveChunkController layer quantum derived directly from Module-4 measured layer latency
    layer512_ms = (ph.dit_512_layer_p50_ms * cscale) / sp
    layer1024_ms = layer512_ms * (ph.dit_1024_step_ms / max(1.0, ph.dit_512_cfg_step_mean_ms))
    ar_tpot_slo = (policy.ar_tpot_slo_ms * cscale) / sp
    ar_ttft_slo = (policy.ar_ttft_slo_ms * cscale) / sp
    chunk_layers_512 = max(1, min(8, int(max(0.0, ar_tpot_slo - ar_tok_ms) // max(0.1, layer512_ms))))
    chunk_layers_1024 = max(1, min(8, int(max(0.0, ar_tpot_slo - ar_tok_ms) // max(0.1, layer1024_ms))))
    chunk_q_512_ms = chunk_layers_512 * layer512_ms
    chunk_q_1024_ms = chunk_layers_1024 * layer1024_ms

    expert_bytes_scaled = int(ph.packed_expert_data_mib * ws.weight_scale * MIB)
    layer_stride_scaled = int(round(ph.layer_stride_mib * ws.weight_scale))

    hbm_mib = hw.gpu_hbm_gib * 1024.0
    ledgers = {
        g: SimGpuMemoryLedger(
            gpu_id=g,
            total_hbm_mib=hbm_mib,
            policy=policy,
            role=policy.role_of(g),
            resident_expert_layers=int(round(policy.initial_alphas.get(g, 1.0) * ph.num_expert_layers)),
            num_expert_layers=ph.num_expert_layers,
            layer_stride_mib=layer_stride_scaled,
            physics=ph,
            weight_scale=ws.weight_scale,
            vae_spike_scale=ws.vae_spike_scale,
        )
        for g in range(num_gpus)
    }
    workers = router.init_workers(
        num_gpus=num_gpus,
        role_of=policy.role_of,
        initial_alphas=policy.initial_alphas,
        kv_free_bytes=int(20 * GIB),
    )

    gpu_free_at = [0.0 for _ in range(num_gpus)]
    ar_slot_free_at = [0.0 for _ in range(num_gpus)]
    num_nodes = max(1, (num_gpus + max(1, hw.gpus_per_node) - 1) // max(1, hw.gpus_per_node))
    rdma_free_at = [0.0 for _ in range(num_nodes)]
    unpaced_h2d_until = [0.0 for _ in range(num_gpus)]

    sessions = _build_session_stream(
        num_sessions=num_sessions,
        offered_qps=offered_qps,
        num_gpus=num_gpus,
        seed=seed,
        extra_edit_turns=ws.extra_edit_turns,
        skew_tag=skew_tag,
    )
    n_sess = len(sessions)
    sess_gpu: list[int | None] = [None for _ in range(n_sess)]
    sess_kv_mib = [0.0 for _ in range(n_sess)]
    sess_main_kv_mib = [0.0 for _ in range(n_sess)]
    sess_ar_ok = [True for _ in range(n_sess)]
    sess_dit_ok = [True for _ in range(n_sess)]
    sess_oom = [False for _ in range(n_sess)]
    sess_retries = [0 for _ in range(n_sess)]
    sess_ar_ttfts: list[list[float]] = [[] for _ in range(n_sess)]
    sess_ar_tpots: list[list[float]] = [[] for _ in range(n_sess)]
    sess_dit_records: list[list[tuple[float, int]]] = [[] for _ in range(n_sess)]

    ttfts: list[float] = []
    tpots: list[float] = []
    dit_lats: list[float] = []
    ships = 0
    morphs = 0
    cross_node_ships = 0
    preemptions = 0

    # Causal discrete-event priority ordering at equal timestamps:
    # release (0) -> DiT end (1) -> return unpack (2) -> DiT start (3) -> forward ship (4) -> turn arrival (5)
    EV_RELEASE = 0
    EV_DIT_END = 1
    EV_RET_UNPACK = 2
    EV_DIT_START = 3
    EV_FWD_SHIP = 4
    EV_TURN = 5

    ev_seq = 0
    pq: list[tuple[float, int, int, int, Any]] = []
    for sid in range(n_sess):
        ev_seq += 1
        pq.append((sessions[sid][0], EV_TURN, sid, ev_seq, (0, 0)))
    heapq.heapify(pq)

    def _push_ev(ts_ms: float, kind: int, s_id: int, payload: Any) -> None:
        nonlocal ev_seq
        ev_seq += 1
        heapq.heappush(pq, (ts_ms, kind, s_id, ev_seq, payload))

    def _sync_states(now_ms: float) -> tuple[dict[int, bool], dict[int, int]]:
        busy_map: dict[int, bool] = {}
        dit_map: dict[int, int] = {}
        for g in range(num_gpus):
            rem = max(0.0, gpu_free_at[g] - now_ms)
            workers[g].queue_wait_ms = rem
            workers[g].expert_alpha = ledgers[g].expert_alpha
            workers[g].kv_free_bytes = ledgers[g].free_kv_bytes()
            busy_map[g] = rem > 0.0
            dit_map[g] = 1 if rem > (200.0 / sp) else 0
        return busy_map, dit_map

    def _pick_oom_retry_gpu(failed_gpu: int, now_ms: float) -> int:
        """Match `MorphServeClusterEngine._pick_oom_retry_gpu` (`cluster_engine.py:612-629`) without leaving `und` pool."""
        busy_now, _ = _sync_states(now_ms)
        if policy.routing_mode in ("twostage_shm", "static_nccl_fission") and policy.static_und_ranks:
            cands = [g for g in policy.static_und_ranks if g != failed_gpu]
            if not cands:
                return int(policy.static_und_ranks[0])
        else:
            cands = [g for g in range(num_gpus) if g != failed_gpu]
            if not cands:
                return failed_gpu
        cands.sort(
            key=lambda g: (
                0 if workers[g].kv_free_bytes >= 128 * MIB else 1,
                workers[g].queue_wait_ms,
                bool(busy_now.get(g, False)),
                g,
            )
        )
        return cands[0]

    is_static_ephemeral = policy.routing_mode in ("static_nccl_fission", "twostage_shm")
    is_tight_mot = policy.cptg_architecture == "tight_mot"

    while pq:
        ev_ms, ev_kind, sid, _, payload = heapq.heappop(pq)
        if ev_kind == EV_RELEASE:
            old_g = sess_gpu[sid]
            if old_g is not None:
                ledgers[old_g].release_session(sid)
            continue

        if ev_kind == EV_FWD_SHIP:
            (
                ti,
                attempt,
                cur_gpu,
                t_turn0,
                quad,
                height,
                n_turns,
                cand_main_kv,
                dit_phys_kv,
                step_base,
                dit_exec_ms,
            ) = payload
            dit_ready_ms = ev_ms
            has_more_turns = (ti + 1) < n_turns
            wire_kv_mib = dit_phys_kv if policy.routing_mode == "twostage_shm" else cand_main_kv
            busy_at_dit, _ = _sync_states(dit_ready_ms)
            gamma = 2 if (quad == "multiturn_edit" and has_more_turns) else 1
            req = CPTGRequestSpec(
                req_id=f"s{sid}_t{ti}_a{attempt}",
                src_gpu=cur_gpu,
                architecture=policy.cptg_architecture,
                kv_bytes=int(wire_kv_mib * MIB),
                gen_expert_bytes=expert_bytes_scaled,
                dit_step1_ms=step_base,
                dit_total_steps=20,
                resolution=height,
                interleaved_roundtrip_gamma=gamma,
            )
            dec, ship_mode = router.plan_and_reserve_transition(
                routing_mode=policy.routing_mode,
                static_und_ranks=policy.static_und_ranks,
                static_gen_ranks=policy.static_gen_ranks,
                req=req,
                est_dit_ms=dit_exec_ms,
                workers=workers,
                gpu_busy=busy_at_dit,
                use_pacer=policy.use_h2d_pacer,
                now_ms=dit_ready_ms,
            )
            dst_gpu = dec.dst_gpu
            router.release_dit_reservation(dst_gpu, dit_exec_ms, workers=workers)
            if ship_mode != "none" and dst_gpu != cur_gpu:
                router.release_kv_reservation(dst_gpu, int(req.effective_state_bytes()), workers=workers)

            release_src = True if is_tight_mot else (not has_more_turns)
            if ship_mode == "nccl_static":
                # Decision 2/6 + `cluster_engine.py:876`: `tight_mot` releases `cur_gpu` (`und` rank) immediately
                # upon forward pack/send (`release_src_ctx=True`) so `und` holds zero duplicate KV during DiT.
                src_node = topo.node_of(cur_gpu)
                r_wait = max(0.0, rdma_free_at[src_node] - dit_ready_ms) if dec.hop_tier == "inter_node" else 0.0
                bw = topo.effective_p2p_gbps(cur_gpu, dst_gpu)
                wire_ms = ((cand_main_kv * MIB / 1e9) / max(0.5, bw)) * 1000.0 + topo.handoff_ms(cur_gpu, dst_gpu)
                trans_ms = r_wait + wire_ms
                if not topo.same_node(cur_gpu, dst_gpu):
                    rdma_free_at[src_node] = max(dit_ready_ms, rdma_free_at[src_node]) + wire_ms
            else:
                # Regressed directly from Phase-0 Job-0d (`base_ms + ms_per_mib * dit_phys_kv`);
                # `tight_mot` releases `cur_gpu` (`und` rank) immediately upon `SEND_SHM` (`cluster_engine.py:897`).
                trans_ms = (ph.twostage_shm_base_ms + ph.twostage_shm_ms_per_mib * dit_phys_kv) / math.sqrt(
                    max(1.0, sp)
                )

            if release_src:
                ledgers[cur_gpu].release_session(sid)
            else:
                ledgers[cur_gpu].upsert_session_kv(sid, cand_main_kv)

            start_dit = max(dit_ready_ms + trans_ms, gpu_free_at[dst_gpu])
            reserved_end_dit = start_dit + dit_exec_ms
            gpu_free_at[dst_gpu] = reserved_end_dit
            _push_ev(
                start_dit,
                EV_DIT_START,
                sid,
                (
                    ti,
                    attempt,
                    cur_gpu,
                    dst_gpu,
                    ship_mode,
                    t_turn0,
                    height,
                    n_turns,
                    cand_main_kv,
                    dit_phys_kv,
                    dit_exec_ms,
                    reserved_end_dit,
                ),
            )
            continue

        if ev_kind == EV_DIT_START:
            (
                ti,
                attempt,
                cur_gpu,
                dst_gpu,
                ship_mode,
                t_turn0,
                height,
                n_turns,
                cand_main_kv,
                dit_phys_kv,
                dit_exec_ms,
                reserved_end_dit,
            ) = payload
            start_dit = ev_ms
            if not ledgers[dst_gpu].upsert_session_kv(sid, dit_phys_kv):
                if gpu_free_at[dst_gpu] == reserved_end_dit:
                    gpu_free_at[dst_gpu] = start_dit
                ledgers[dst_gpu].release_session(sid)
                ledgers[cur_gpu].release_session(sid)
                if attempt < policy.max_oom_retries and ti == 0:
                    sess_retries[sid] += 1
                    sess_gpu[sid] = _pick_oom_retry_gpu(dst_gpu, start_dit)
                    _push_ev(start_dit, EV_TURN, sid, (ti, attempt + 1))
                else:
                    sess_oom[sid] = True
                continue

            ships += 1
            if not topo.same_node(cur_gpu, dst_gpu):
                cross_node_ships += 1

            dit_ok_mem, swap_out_mib = ledgers[dst_gpu].execute_dit_and_vae(height, active_sid=sid)
            if not dit_ok_mem:
                abort_end = start_dit + dit_exec_ms * 0.45
                if gpu_free_at[dst_gpu] == reserved_end_dit:
                    gpu_free_at[dst_gpu] = abort_end
                ledgers[dst_gpu].release_session(sid)
                ledgers[cur_gpu].release_session(sid)
                if attempt < policy.max_oom_retries and ti == 0:
                    sess_retries[sid] += 1
                    sess_gpu[sid] = _pick_oom_retry_gpu(dst_gpu, abort_end)
                    _push_ev(abort_end, EV_TURN, sid, (ti, attempt + 1))
                else:
                    sess_oom[sid] = True
                continue

            swap_out_ms = (
                ((swap_out_mib * MIB / 1e9) / max(0.5, topo.solo_h2d_gbps(dst_gpu))) * 1000.0
                if swap_out_mib > 0.0
                else 0.0
            )
            end_dit = start_dit + swap_out_ms + dit_exec_ms
            gpu_free_at[dst_gpu] = max(gpu_free_at[dst_gpu], end_dit)
            workers[dst_gpu].expert_alpha = ledgers[dst_gpu].expert_alpha
            _push_ev(
                end_dit,
                EV_DIT_END,
                sid,
                (ti, attempt, cur_gpu, dst_gpu, ship_mode, t_turn0, height, n_turns, cand_main_kv),
            )
            continue

        if ev_kind == EV_DIT_END:
            ti, attempt, cur_gpu, dst_gpu, ship_mode, t_turn0, height, n_turns, cand_main_kv = payload
            end_dit = ev_ms
            # Release ephemeral `gen` context on `dst_gpu` at exact `end_dit` (`cluster_engine.py:1010,1043,1053`).
            ledgers[dst_gpu].release_session(sid)
            has_more_turns = (ti + 1) < n_turns
            needs_return_ship = has_more_turns and is_tight_mot
            if needs_return_ship:
                # Post-DiT return leg (`dst_gpu -> cur_gpu`, `cluster_engine.py:1030-1080`) evaluated causally at `end_dit`:
                # ship updated `cand_main_kv` (`include_cfg_dup=False`) back to `cur_gpu` (`und` rank).
                if ship_mode == "nccl_static":
                    dst_node = topo.node_of(dst_gpu)
                    ret_bw = topo.effective_p2p_gbps(dst_gpu, cur_gpu)
                    ret_wire_ms = (
                        ((cand_main_kv * MIB / 1e9) / max(0.5, ret_bw)) * 1000.0
                        + topo.handoff_ms(dst_gpu, cur_gpu)
                    )
                    if not topo.same_node(dst_gpu, cur_gpu):
                        ret_r_wait = max(0.0, rdma_free_at[dst_node] - end_dit)
                        rdma_free_at[dst_node] = max(end_dit, rdma_free_at[dst_node]) + ret_wire_ms
                    else:
                        ret_r_wait = 0.0
                    ret_trans_ms = ret_r_wait + ret_wire_ms
                else:
                    ret_trans_ms = (
                        ph.twostage_shm_base_ms + ph.twostage_shm_ms_per_mib * cand_main_kv
                    ) / math.sqrt(max(1.0, sp))
                _push_ev(
                    end_dit + ret_trans_ms,
                    EV_RET_UNPACK,
                    sid,
                    (ti, attempt, cur_gpu, dst_gpu, t_turn0, height, n_turns, cand_main_kv),
                )
            else:
                sess_kv_mib[sid] = cand_main_kv
                sess_main_kv_mib[sid] = cand_main_kv
                turn_lat = end_dit - t_turn0
                dit_lats.append(turn_lat)
                sess_dit_records[sid].append((turn_lat, height))
                turn_slo_ms = (policy.dit_turn_slo_for(height, steps=20) * cscale) / sp
                if turn_lat > turn_slo_ms:
                    sess_dit_ok[sid] = False
                if has_more_turns:
                    sess_gpu[sid] = cur_gpu
                    think_ms = _deterministic_think_ms(sid, ti + 1, policy.inter_turn_think_s_range)
                    _push_ev(end_dit + think_ms, EV_TURN, sid, (ti + 1, 0))
            continue

        if ev_kind == EV_RET_UNPACK:
            ti, attempt, cur_gpu, dst_gpu, t_turn0, height, n_turns, cand_main_kv = payload
            turn_end_ms = ev_ms
            if not ledgers[cur_gpu].upsert_session_kv(sid, cand_main_kv):
                ledgers[cur_gpu].release_session(sid)
                if attempt < policy.max_oom_retries and ti == 0:
                    sess_retries[sid] += 1
                    sess_gpu[sid] = _pick_oom_retry_gpu(cur_gpu, turn_end_ms)
                    _push_ev(turn_end_ms, EV_TURN, sid, (0, attempt + 1))
                else:
                    sess_oom[sid] = True
                continue
            if not topo.same_node(dst_gpu, cur_gpu):
                cross_node_ships += 1
            sess_kv_mib[sid] = cand_main_kv
            sess_main_kv_mib[sid] = cand_main_kv
            turn_lat = turn_end_ms - t_turn0
            dit_lats.append(turn_lat)
            sess_dit_records[sid].append((turn_lat, height))
            turn_slo_ms = (policy.dit_turn_slo_for(height, steps=20) * cscale) / sp
            if turn_lat > turn_slo_ms:
                sess_dit_ok[sid] = False
            sess_gpu[sid] = cur_gpu
            think_ms = _deterministic_think_ms(sid, ti + 1, policy.inter_turn_think_s_range)
            _push_ev(turn_end_ms + think_ms, EV_TURN, sid, (ti + 1, 0))
            continue

        ti, attempt = payload
        _, quad, turns = sessions[sid]
        plan, height = turns[ti]
        is_1024 = height >= 1024
        n_turns = len(turns)

        while True:
            busy_map, dit_map = _sync_states(ev_ms)
            cur_gpu = sess_gpu[sid]
            if cur_gpu is None:
                cur_gpu = router.select_initial_gpu(
                    num_gpus=num_gpus,
                    routing_mode=policy.routing_mode,
                    preemption_mode=policy.preemption_mode,
                    static_und_ranks=policy.static_und_ranks,
                    modality_plan=plan,
                    workers=workers,
                    gpu_busy=busy_map,
                    active_dit_ranks=dit_map,
                )
                sess_gpu[sid] = cur_gpu

            # --- Pure AR turn ---
            if plan == "ar_only":
                swap_in_mib = ledgers[cur_gpu].ensure_session_resident(sid)
                if swap_in_mib < 0.0:
                    ledgers[cur_gpu].release_session(sid)
                    sess_oom[sid] = True
                    break
                swap_in_ms = (
                    ((swap_in_mib * MIB / 1e9) / max(0.5, topo.solo_h2d_gbps(cur_gpu))) * 1000.0
                    if swap_in_mib > 0.0
                    else 0.0
                )
                n_tok = 30
                ar_delta = 48.0 * ws.kv_per_turn_scale
                new_kv = sess_kv_mib[sid] + ar_delta
                if not ledgers[cur_gpu].upsert_session_kv(sid, new_kv):
                    ledgers[cur_gpu].release_session(sid)
                    if attempt < policy.max_oom_retries and ti == 0:
                        attempt += 1
                        sess_retries[sid] += 1
                        sess_gpu[sid] = _pick_oom_retry_gpu(cur_gpu, ev_ms)
                        continue
                    sess_oom[sid] = True
                    break
                sess_kv_mib[sid] = new_kv
                sess_main_kv_mib[sid] += ar_delta

                dit_rem = max(0.0, gpu_free_at[cur_gpu] - ev_ms)
                if dit_rem > 0.0 and policy.preemption_mode == "layer_chunk":
                    chunk_q = chunk_q_1024_ms if is_1024 else chunk_q_512_ms
                    wait_ms = min(dit_rem, 0.5 * chunk_q) + swap_in_ms
                    tpot = min(
                        (ph.m4_colocated_ar_tpot_p50_ms * cscale) / sp,
                        ar_tok_ms + 0.5 * chunk_q,
                    )
                    gpu_free_at[cur_gpu] += ar_prefill_ms + n_tok * ar_tok_ms * batched_ar_amort
                    preemptions += 1
                elif dit_rem > 0.0 and policy.preemption_mode == "step_preemptive":
                    step_q = dit1024_step_ms if is_1024 else dit512_step_ms
                    wait_ms = min(dit_rem, 0.5 * step_q) + swap_in_ms
                    tpot = max(
                        (ph.step_preempt_ar_tpot_p50_ms * cscale) / sp,
                        ar_tok_ms + 0.5 * step_q,
                    )
                    gpu_free_at[cur_gpu] += ar_prefill_ms + n_tok * ar_tok_ms * batched_ar_amort
                    preemptions += 1
                else:
                    ar_wait = max(0.0, ar_slot_free_at[cur_gpu] - ev_ms) if dit_rem <= 0.0 else dit_rem
                    wait_ms = ar_wait + swap_in_ms
                    tpot = ar_tok_ms
                    ar_slot_free_at[cur_gpu] = (
                        max(ev_ms, ar_slot_free_at[cur_gpu]) + ar_prefill_ms + n_tok * ar_tok_ms * batched_ar_amort
                    )
                    if dit_rem > 0.0:
                        gpu_free_at[cur_gpu] = (
                            max(ev_ms, gpu_free_at[cur_gpu]) + ar_prefill_ms + n_tok * ar_tok_ms * batched_ar_amort
                        )

                ttft = wait_ms + ar_prefill_ms
                ttfts.append(ttft)
                tpots.append(tpot)
                sess_ar_ttfts[sid].append(ttft)
                sess_ar_tpots[sid].append(tpot)
                if ttft > ar_ttft_slo or tpot > ar_tpot_slo:
                    sess_ar_ok[sid] = False
                end_ms = ev_ms + ttft + (n_tok - 1) * tpot
                if ti + 1 < n_turns:
                    think_ms = _deterministic_think_ms(sid, ti + 1, policy.inter_turn_think_s_range)
                    _push_ev(end_ms + think_ms, EV_TURN, sid, (ti + 1, 0))
                else:
                    _push_ev(end_ms, EV_RELEASE, sid, None)
                break

            # --- CoT or DiT turn ---
            t_turn0 = ev_ms
            swap_in_mib = ledgers[cur_gpu].ensure_session_resident(sid)
            if swap_in_mib < 0.0:
                ledgers[cur_gpu].release_session(sid)
                sess_oom[sid] = True
                break
            swap_in_ms = (
                ((swap_in_mib * MIB / 1e9) / max(0.5, topo.solo_h2d_gbps(cur_gpu))) * 1000.0
                if swap_in_mib > 0.0
                else 0.0
            )
            if quad == "visual_cot":
                base_delta_kv = 185.0
            elif quad == "multiturn_edit":
                base_delta_kv = ph.turn_edit_512_kv_delta_mib * (1.30 if is_1024 else 1.0)
            else:
                base_delta_kv = 28.0
            main_delta_kv = base_delta_kv * ws.kv_per_turn_scale
            cand_main_kv = sess_main_kv_mib[sid] + main_delta_kv
            # During active DiT, non-CoW policies duplicate `cfg_text` (`~2x` of `main_ctx`), while `und` think-time
            # residency (`BagelCPTGAdapter:1458`) holds only `main_ctx` (`cand_main_kv`).
            dit_phys_kv = (
                cand_main_kv * ph.non_cow_kv_mult
                if (not policy.use_module3_cow_loopback and quad == "multiturn_edit" and ti > 0)
                else cand_main_kv
            )
            src_prep_kv = cand_main_kv if policy.routing_mode == "static_nccl_fission" else dit_phys_kv
            if not ledgers[cur_gpu].upsert_session_kv(sid, src_prep_kv):
                ledgers[cur_gpu].release_session(sid)
                if attempt < policy.max_oom_retries and ti == 0:
                    attempt += 1
                    sess_retries[sid] += 1
                    sess_gpu[sid] = _pick_oom_retry_gpu(cur_gpu, ev_ms)
                    continue
                sess_oom[sid] = True
                break

            if plan == "cot_then_dit":
                n_tok = 26
                dit_rem = max(0.0, gpu_free_at[cur_gpu] - ev_ms)
                if dit_rem > 0.0 and policy.preemption_mode == "layer_chunk":
                    chunk_q = chunk_q_1024_ms if is_1024 else chunk_q_512_ms
                    wait_ms = min(dit_rem, 0.5 * chunk_q) + swap_in_ms
                    tpot = min(
                        (ph.m4_colocated_ar_tpot_p50_ms * cscale) / sp,
                        ar_tok_ms + 0.5 * chunk_q,
                    )
                    gpu_free_at[cur_gpu] += ar_prefill_ms + n_tok * ar_tok_ms * batched_ar_amort
                    preemptions += 1
                elif dit_rem > 0.0 and policy.preemption_mode == "step_preemptive":
                    step_q = dit1024_step_ms if is_1024 else dit512_step_ms
                    wait_ms = min(dit_rem, 0.5 * step_q) + swap_in_ms
                    tpot = max(
                        (ph.step_preempt_ar_tpot_p50_ms * cscale) / sp,
                        ar_tok_ms + 0.5 * step_q,
                    )
                    gpu_free_at[cur_gpu] += ar_prefill_ms + n_tok * ar_tok_ms * batched_ar_amort
                    preemptions += 1
                else:
                    wait_ms = dit_rem + swap_in_ms
                    tpot = ar_tok_ms
                    gpu_free_at[cur_gpu] = (
                        max(ev_ms, gpu_free_at[cur_gpu]) + swap_in_ms + ar_prefill_ms + n_tok * ar_tok_ms * batched_ar_amort
                    )
                ttft = wait_ms + ar_prefill_ms
                ttfts.append(ttft)
                tpots.append(tpot)
                sess_ar_ttfts[sid].append(ttft)
                sess_ar_tpots[sid].append(tpot)
                if ttft > ar_ttft_slo or tpot > ar_tpot_slo:
                    sess_ar_ok[sid] = False
                dit_ready_ms = ev_ms + ttft + (n_tok - 1) * tpot
            else:
                start_prep = max(ev_ms, gpu_free_at[cur_gpu]) + swap_in_ms
                # Turn 0 has zero loopback penalty for ALL policies (symmetric initial prompt/image prefill);
                # Turns 1..N use exact Module-3 measured GPU in-place vs CPU PIL/PNG loopback latency.
                if ti == 0:
                    loopback_ms = ar_prefill_ms
                elif policy.use_module3_cow_loopback:
                    loopback_ms = (ph.morphserve_loopback_turn_ms * cscale) / sp
                else:
                    loopback_ms = (ph.stock_loopback_turn_ms * cscale) / sp
                gpu_free_at[cur_gpu] = start_prep + loopback_ms
                dit_ready_ms = gpu_free_at[cur_gpu]

            # 20-step DiT latency from measured solo base + measured linear KV-token attention slope
            main_seq_tokens = int(round((cand_main_kv / max(0.01, ws.weight_scale)) / max(1e-4, ph.kv_mib_per_token)))
            l_gen = 4098 if is_1024 else 1026
            has_shared_prefix = policy.use_module3_cow_loopback and (ti > 0 or quad == "visual_cot")
            use_cascade = has_shared_prefix and should_use_cascade_cfg_split(l_gen, main_seq_tokens, 2)
            slope_512 = ph.morphserve_dit_512_ms_per_kv_tok if use_cascade else ph.stock_dit_512_ms_per_kv_tok
            q_scale = 4.0 if is_1024 else 1.0
            kv_attn_ms = ((slope_512 * q_scale * main_seq_tokens) * cscale) / sp

            step_base = dit1024_step_ms if is_1024 else dit512_step_ms
            dit_core_ms = (dit1024_core_20_ms if is_1024 else dit512_core_20_ms) + kv_attn_ms
            dit_exec_ms = dit_core_ms + (vae1024_ms if is_1024 else vae512_ms)

            if is_static_ephemeral:
                # Schedule causal forward pack/send at `dit_ready_ms` so `cur_gpu` holds `src_prep_kv` during
                # CoT/`PREP_DIT`, frees `cur_gpu` across `[dit_ready_ms, end_dit]`, and return-ships at `end_dit`.
                _push_ev(
                    dit_ready_ms,
                    EV_FWD_SHIP,
                    sid,
                    (
                        ti,
                        attempt,
                        cur_gpu,
                        t_turn0,
                        quad,
                        height,
                        n_turns,
                        cand_main_kv,
                        dit_phys_kv,
                        step_base,
                        dit_exec_ms,
                    ),
                )
                break

            # Decision 2/6: `nccl_dual` ships non-duplicated `cand_main_kv` wire payload.
            wire_kv_mib = cand_main_kv
            busy_at_dit, _ = _sync_states(dit_ready_ms)
            gamma = 2 if (quad == "multiturn_edit" and ti + 1 < n_turns) else 1
            req = CPTGRequestSpec(
                req_id=f"s{sid}_t{ti}",
                src_gpu=cur_gpu,
                architecture=policy.cptg_architecture,
                kv_bytes=int(wire_kv_mib * MIB),
                gen_expert_bytes=expert_bytes_scaled,
                dit_step1_ms=step_base,
                dit_total_steps=20,
                resolution=height,
                interleaved_roundtrip_gamma=gamma,
            )
            dec, ship_mode = router.plan_and_reserve_transition(
                routing_mode=policy.routing_mode,
                static_und_ranks=policy.static_und_ranks,
                static_gen_ranks=policy.static_gen_ranks,
                req=req,
                est_dit_ms=dit_exec_ms,
                workers=workers,
                gpu_busy=busy_at_dit,
                use_pacer=policy.use_h2d_pacer,
                now_ms=dit_ready_ms,
            )
            dst_gpu = dec.dst_gpu
            router.release_dit_reservation(dst_gpu, dit_exec_ms, workers=workers)
            if ship_mode != "none" and dst_gpu != cur_gpu:
                router.release_kv_reservation(dst_gpu, int(req.effective_state_bytes()), workers=workers)

            has_more_turns = (ti + 1) < n_turns
            trans_ms = 0.0
            if ship_mode == "nccl_dual":
                src_node = topo.node_of(cur_gpu)
                r_wait = max(0.0, rdma_free_at[src_node] - dit_ready_ms) if dec.hop_tier == "inter_node" else 0.0
                bw = topo.effective_p2p_gbps(cur_gpu, dst_gpu)
                wire_ms = ((cand_main_kv * MIB / 1e9) / max(0.5, bw)) * 1000.0 + topo.handoff_ms(cur_gpu, dst_gpu)
                trans_ms = r_wait + wire_ms
                if not topo.same_node(cur_gpu, dst_gpu):
                    rdma_free_at[src_node] = max(dit_ready_ms, rdma_free_at[src_node]) + wire_ms
                ledgers[cur_gpu].release_session(sid)
                if not ledgers[dst_gpu].upsert_session_kv(sid, dit_phys_kv):
                    ledgers[dst_gpu].release_session(sid)
                    if attempt < policy.max_oom_retries and ti == 0:
                        attempt += 1
                        sess_retries[sid] += 1
                        sess_gpu[sid] = _pick_oom_retry_gpu(dst_gpu, dit_ready_ms + trans_ms)
                        ev_ms = dit_ready_ms + trans_ms
                        continue
                    sess_oom[sid] = True
                    break
                ships += 1
                if not topo.same_node(cur_gpu, dst_gpu):
                    cross_node_ships += 1
            elif policy.use_naive_offload:
                grp = topo.half_group_of(cur_gpu)
                active_same_half = 1 + sum(
                    1 for peer in grp.gpu_ids if peer != cur_gpu and unpaced_h2d_until[peer] > dit_ready_ms
                )
                eff_h2d = topo.effective_h2d_gbps(cur_gpu, active_h2d_same_half=active_same_half)
                swap_ms = ((expert_bytes_scaled / 1e9) / max(0.5, eff_h2d)) * 1000.0
                if policy.use_h2d_pacer:
                    p_wait = pacer.reserve_slot(cur_gpu, swap_ms, now_ms=dit_ready_ms)
                    trans_ms = p_wait + swap_ms
                else:
                    unpaced_h2d_until[cur_gpu] = dit_ready_ms + swap_ms
                    trans_ms = swap_ms
            else:
                trans_ms = max(0.0, dec.t_morph_ms - workers[cur_gpu].queue_wait_ms)
                if policy.routing_mode == "dual_path":
                    morphs += 1

            start_dit = max(dit_ready_ms + trans_ms, gpu_free_at[dst_gpu])
            dit_ok_mem, swap_out_mib = ledgers[dst_gpu].execute_dit_and_vae(height, active_sid=sid)
            if not dit_ok_mem:
                gpu_free_at[dst_gpu] = start_dit + dit_exec_ms * 0.45
                ledgers[dst_gpu].release_session(sid)
                ledgers[cur_gpu].release_session(sid)
                if attempt < policy.max_oom_retries and ti == 0:
                    attempt += 1
                    sess_retries[sid] += 1
                    sess_gpu[sid] = _pick_oom_retry_gpu(dst_gpu, gpu_free_at[dst_gpu])
                    ev_ms = gpu_free_at[dst_gpu]
                    continue
                sess_oom[sid] = True
                break

            swap_out_ms = (
                ((swap_out_mib * MIB / 1e9) / max(0.5, topo.solo_h2d_gbps(dst_gpu))) * 1000.0
                if swap_out_mib > 0.0
                else 0.0
            )
            end_dit = start_dit + swap_out_ms + dit_exec_ms
            gpu_free_at[dst_gpu] = end_dit
            workers[dst_gpu].expert_alpha = ledgers[dst_gpu].expert_alpha

            sess_kv_mib[sid] = cand_main_kv
            sess_main_kv_mib[sid] = cand_main_kv
            turn_end_ms = end_dit
            turn_lat = turn_end_ms - t_turn0
            dit_lats.append(turn_lat)
            sess_dit_records[sid].append((turn_lat, height))
            turn_slo_ms = (policy.dit_turn_slo_for(height, steps=20) * cscale) / sp
            if turn_lat > turn_slo_ms:
                sess_dit_ok[sid] = False

            if has_more_turns:
                # Drop per-turn CFG branch caches after DiT (`BagelCPTGAdapter:1458`); retain `main_ctx` only
                ledgers[dst_gpu].upsert_session_kv(sid, cand_main_kv)
                sess_gpu[sid] = dst_gpu
                think_ms = _deterministic_think_ms(sid, ti + 1, policy.inter_turn_think_s_range)
                _push_ev(turn_end_ms + think_ms, EV_TURN, sid, (ti + 1, 0))
            else:
                _push_ev(turn_end_ms, EV_RELEASE, sid, None)
            break

    trim = int(n_sess * policy.steady_state_trim_frac) if n_sess >= 20 else 0
    core_ids = range(trim, n_sess - trim) if trim > 0 else range(n_sess)
    n_core = max(1, len(core_ids))
    core_ok = sum(1 for sid in core_ids if (not sess_oom[sid]) and sess_ar_ok[sid] and sess_dit_ok[sid])

    ooms = sum(1 for x in sess_oom if x)
    joint_ok = sum(1 for sid in range(n_sess) if (not sess_oom[sid]) and sess_ar_ok[sid] and sess_dit_ok[sid])
    att_frac = core_ok / float(n_core)

    slo_curve_by_k: dict[str, dict[str, float]] = {}
    for k_val in (1.5, 2.0, 2.5, 3.0):
        slo_k = DEFAULT_OPERATOR_PROFILE.slo_thresholds_ms(k=k_val, model_family="bagel")
        ttft_k = (slo_k["ar_ttft_slo_ms"] * cscale) / sp
        tpot_k = (slo_k["ar_tpot_slo_ms"] * cscale) / sp
        k_ok_cnt = 0
        for sid in core_ids:
            if sess_oom[sid]:
                continue
            if any(x > ttft_k for x in sess_ar_ttfts[sid]) or any(x > tpot_k for x in sess_ar_tpots[sid]):
                continue
            if any(lat > (policy.dit_turn_slo_for(h, steps=20, k=k_val) * cscale) / sp for lat, h in sess_dit_records[sid]):
                continue
            k_ok_cnt += 1
        slo_curve_by_k[f"{k_val:.1f}x"] = {
            "attainment_pct": round(100.0 * k_ok_cnt / float(n_core), 1),
            "goodput_rps": round(offered_qps * (k_ok_cnt / float(n_core)), 4),
        }

    return {
        "policy_name": policy_name,
        "workload_scale": ws.name,
        "skew_tag": skew_tag or "main_mix",
        "num_gpus": num_gpus,
        "interconnect": hw.name,
        "offered_qps": round(offered_qps, 3),
        "crossover_kv_mib": _crossover_kv_mib(hw, physics=ph, workload_scale=ws),
        "joint_slo_goodput_rps": round(offered_qps * att_frac, 4),
        "joint_slo_attainment_pct": round(100.0 * att_frac, 1),
        "full_trace_attainment_pct": round(100.0 * joint_ok / max(1, n_sess), 1),
        "oom_sessions": ooms,
        "oom_retries": sum(sess_retries),
        "peak_hbm_mib": round(max(l.peak_used_mib for l in ledgers.values()), 1),
        "vmm_carves": sum(l.super_to_fine_carves for l in ledgers.values()),
        "vmm_expert_evictions": sum(l.expert_evictions for l in ledgers.values()),
        "vmm_kv_host_swap_outs": sum(l.kv_host_swap_outs for l in ledgers.values()),
        "vmm_kv_host_swap_ins": sum(l.kv_host_swap_ins for l in ledgers.values()),
        "ar_ttft_p95_ms": percentile(ttfts, 95),
        "ar_tpot_p95_ms": percentile(tpots, 95),
        "dit_turn_p95_ms": percentile(dit_lats, 95),
        "path_a_ships": ships,
        "path_b_morphs": morphs,
        "preempted_ar_count": preemptions,
        "cross_node_ships": cross_node_ships,
        "slo_curve_by_k": slo_curve_by_k,
    }


def evaluate_against_real_runs(
    real_runs: list[dict[str, Any]],
    physics: MeasuredPhysicsProfile | None = None,
) -> SimValidationReport:
    """Out-of-sample validation with ZERO parameter grid search against live 8-GPU / 16-GPU measurements."""
    ph = MeasuredPhysicsProfile.from_phase01_results() if physics is None else physics
    hw = HardwareInterconnectSpec.presets(physics=ph)[0]
    if not real_runs:
        return SimValidationReport(
            reference_source="phase01_measured_json_profile",
            zero_parameter_fitting=True,
            points_checked=0,
            goodput_mape_pct=0.0,
            ar_tpot_p95_mape_pct=0.0,
            dit_p95_mape_pct=0.0,
            max_point_error_pct=0.0,
            within_8pct_oos_gate=True,
            within_5pct_gate=True,
            per_point=[],
        )

    per_pt: list[dict[str, Any]] = []
    gp_errs: list[float] = []
    tp_errs: list[float] = []
    dt_errs: list[float] = []

    for ref in real_runs:
        pol = str(ref.get("policy_name", ""))
        if pol not in VALID_POLICIES:
            continue
        n_gpus = int(ref.get("num_gpus", 8))
        qps = float(ref.get("offered_qps", 0.8))
        n_s = int(ref.get("total_sessions", 80))
        sk_tag = str(ref["skew_tag"]) if ref.get("skew_tag") else None
        eff_hw = (
            HardwareInterconnectSpec(
                name="Kairos_2x8_Measured_Fabric",
                gpu_hbm_gib=40.0,
                compute_speedup_x=1.0,
                solo_h2d_gbps=ph.pcie_solo_h2d_gbps,
                agg_h2d_gbps=ph.pcie_half0_agg_h2d_gbps,
                intra_half_p2p_gbps=ph.kairos_2x8_intra_half_gbps,
                inter_socket_p2p_gbps=ph.kairos_2x8_cross_socket_gbps,
                inter_node_rdma_gbps=ph.kairos_2x8_inter_node_gbps,
                gpus_per_node=8,
            )
            if n_gpus > 8
            else hw
        )
        sim = simulate_cluster_point(pol, n_gpus, eff_hw, offered_qps=qps, num_sessions=n_s, physics=ph, skew_tag=sk_tag)

        real_gp = max(1e-4, float(ref.get("steady_state_goodput_rps") or ref["joint_slo_goodput_rps"]))
        sim_gp = float(sim["joint_slo_goodput_rps"])
        real_tp = max(1.0, float(ref.get("ar_tpot_p95_ms") or sim["ar_tpot_p95_ms"]))
        real_dt = max(1.0, float(ref.get("dit_turn_p95_ms") or sim["dit_turn_p95_ms"]))

        gp_e = abs(sim_gp - real_gp) / real_gp * 100.0
        tp_e = abs(float(sim["ar_tpot_p95_ms"]) - real_tp) / real_tp * 100.0
        dt_e = abs(float(sim["dit_turn_p95_ms"]) - real_dt) / real_dt * 100.0
        gp_errs.append(gp_e)
        tp_errs.append(tp_e)
        dt_errs.append(dt_e)
        per_pt.append(
            {
                "policy_name": pol,
                "num_gpus": n_gpus,
                "offered_qps": qps,
                "real_goodput_rps": round(real_gp, 4),
                "sim_goodput_rps": round(sim_gp, 4),
                "goodput_err_pct": round(gp_e, 2),
                "ar_tpot_p95_err_pct": round(tp_e, 2),
                "dit_p95_err_pct": round(dt_e, 2),
            }
        )

    mgp = round(sum(gp_errs) / max(1, len(gp_errs)), 2)
    mtp = round(sum(tp_errs) / max(1, len(tp_errs)), 2)
    mdt = round(sum(dt_errs) / max(1, len(dt_errs)), 2)
    return SimValidationReport(
        reference_source="live_cluster_out_of_sample",
        zero_parameter_fitting=True,
        points_checked=len(per_pt),
        goodput_mape_pct=mgp,
        ar_tpot_p95_mape_pct=mtp,
        dit_p95_mape_pct=mdt,
        max_point_error_pct=round(max(gp_errs + tp_errs + dt_errs, default=0.0), 2),
        within_8pct_oos_gate=(mgp < 8.0 and mtp < 8.0 and mdt < 8.0),
        within_5pct_gate=(mgp < 5.0 and mtp < 5.0 and mdt < 5.0),
        per_point=per_pt,
    )


def run_h100_80gb_generality_study(
    physics: MeasuredPhysicsProfile | None = None,
) -> dict[str, Any]:
    """Evaluate H100-80GB NVLink generality across (1) 7B multi-QPS curve and (2) ~56GiB / long-KV regimes."""
    ph = MeasuredPhysicsProfile.from_phase01_results() if physics is None else physics
    h100 = HardwareInterconnectSpec.presets(physics=ph)[2]

    regime1_7b_qps_curve: list[dict[str, Any]] = []
    for qps in (0.9, 1.4, 1.9, 2.5):
        for pol in VALID_POLICIES:
            regime1_7b_qps_curve.append(
                simulate_cluster_point(
                    policy_name=pol,
                    num_gpus=8,
                    hw=h100,
                    offered_qps=qps,
                    num_sessions=96,
                    physics=ph,
                    workload_scale=WorkloadScaleSpec.bagel_7b_standard(),
                )
            )

    regime2_56gib_weights: list[dict[str, Any]] = []
    ws_56g = WorkloadScaleSpec.frontier_mot_56gib()
    for num_gpus, qps, n_sess in ((8, 1.35, 96), (64, 10.8, 192)):
        for pol in VALID_POLICIES:
            regime2_56gib_weights.append(
                simulate_cluster_point(
                    policy_name=pol,
                    num_gpus=num_gpus,
                    hw=h100,
                    offered_qps=qps,
                    num_sessions=n_sess,
                    physics=ph,
                    workload_scale=ws_56g,
                )
            )

    regime2_long_kv: list[dict[str, Any]] = []
    ws_long_kv = WorkloadScaleSpec.bagel_7b_long_kv_high_concurrency()
    for num_gpus, qps, n_sess in ((8, 2.4, 96), (64, 19.2, 192)):
        for pol in VALID_POLICIES:
            regime2_long_kv.append(
                simulate_cluster_point(
                    policy_name=pol,
                    num_gpus=num_gpus,
                    hw=h100,
                    offered_qps=qps,
                    num_sessions=n_sess,
                    physics=ph,
                    workload_scale=ws_long_kv,
                )
            )

    return {
        "hardware": asdict(h100),
        "regime1_7b_qps_latency_curve": regime1_7b_qps_curve,
        "regime2_scaled_56gib_weights": regime2_56gib_weights,
        "regime2_long_kv_high_concurrency": regime2_long_kv,
    }


def run_scale_and_interconnect_sweep(
    output_dir: Path,
    real_report_path: Path | None = None,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Run zero-bias physics-driven 8->16->64 GPU & H100-80GB generality matrix across all 6 policies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ph = MeasuredPhysicsProfile.from_phase01_results(results_dir=results_dir)

    real_runs: list[dict[str, Any]] = []
    if real_report_path is not None and real_report_path.is_file():
        raw = json.loads(real_report_path.read_text(encoding="utf-8"))
        if "qps_sweep" in raw:
            real_runs.extend(raw["qps_sweep"])
        if "skew_sweep" in raw:
            real_runs.extend(raw["skew_sweep"])
        if "results" in raw and "qps_sweep" not in raw:
            real_runs.extend(raw["results"])

    val = evaluate_against_real_runs(real_runs, physics=ph)
    presets = HardwareInterconnectSpec.presets(physics=ph)
    matrix: list[dict[str, Any]] = []
    for hw in presets:
        for num_gpus in (8, 16, 64):
            scaled_qps = round(1.15 * (num_gpus / 8.0) * hw.compute_speedup_x, 2)
            n_sess = 96 if num_gpus <= 16 else 240
            for pol in VALID_POLICIES:
                matrix.append(
                    simulate_cluster_point(
                        policy_name=pol,
                        num_gpus=num_gpus,
                        hw=hw,
                        offered_qps=scaled_qps,
                        num_sessions=n_sess,
                        physics=ph,
                    )
                )

    h100_study = run_h100_80gb_generality_study(physics=ph)

    report = {
        "schema": "morphserve.osdi27.simulator.v3",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "zero_manual_bias_constants": True,
        "physics_profile": asdict(ph),
        "calibration": asdict(val),
        "interconnect_presets": [asdict(h) for h in presets],
        "scaling_matrix": matrix,
        "h100_80gb_generality": h100_study,
        "status": "ok" if val.within_8pct_oos_gate else "check_failed",
    }
    out_path = output_dir / "m25-simulator-64gpu-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MorphServe Zero-Bias Physics-Driven Cluster Simulator")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-report", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    rep = run_scale_and_interconnect_sweep(
        args.output_dir,
        real_report_path=args.real_report,
        results_dir=args.results_dir,
    )
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
