"""Model-agnostic async coordinator managing `WorkerFleet`, stateful `DualPathMorphRouter` & Joint SLOs."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import hashlib
import math
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any

from morphserve._core import call_json, encode_json, get_core_lib
from morphserve.router import (
    CPTGRequestSpec,
    ClusterTopologyProfile,
    DEFAULT_OPERATOR_PROFILE,
    DualPathMorphRouter,
    HalfNodeH2DPacer,
    WorkerState,
)

from .config import EnginePolicyConfig
from .multinode_fleet import serialize_policy_config
from .worker_proc import MpSpawnGpuWorkerFleet
from .worker_runtime import WorkerFleet
from .workload import (
    InterleavedSession,
    InterleavedTurn,
    WorkloadSuite,
    assign_dual_trace_arrivals,
)

MIB = 1024**2
GIB = 1024**3


def _percentile(vals: list[float], p: float) -> float:
    """Linear-interpolated percentile matching C++ `percentile_rounded` without premature rounding."""
    if not vals:
        return 0.0
    s = sorted(float(v) for v in vals)
    idx = float(len(s) - 1) * (float(p) / 100.0)
    lo = int(math.floor(idx))
    hi = min(len(s) - 1, int(math.ceil(idx)))
    if lo == hi:
        return s[lo]
    frac = idx - float(lo)
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _deterministic_think_s(
    session_id: str,
    turn_idx: int,
    bounds_or_policy: tuple[float, float] | EnginePolicyConfig,
    *,
    dist: str | None = None,
    median_s: float | None = None,
    sigma: float | None = None,
) -> float:
    if isinstance(bounds_or_policy, EnginePolicyConfig):
        lo_s = float(bounds_or_policy.inter_turn_think_s_range[0])
        hi_s = float(bounds_or_policy.inter_turn_think_s_range[1])
        eff_dist = str(dist if dist is not None else bounds_or_policy.think_time_dist)
        eff_median = float(median_s if median_s is not None else bounds_or_policy.think_time_median_s)
        eff_sigma = float(sigma if sigma is not None else bounds_or_policy.think_time_sigma)
    else:
        lo_s = float(bounds_or_policy[0])
        hi_s = float(bounds_or_policy[1])
        eff_dist = str(dist if dist is not None else "lognormal")
        eff_median = float(median_s if median_s is not None else 6.0)
        eff_sigma = float(sigma if sigma is not None else 0.75)

    if hi_s <= lo_s:
        return max(0.0, lo_s)

    L = get_core_lib()
    sid_b = str(session_id).encode("utf-8")
    if hasattr(L, "ms_ctrl_deterministic_think_dist_s"):
        return float(
            L.ms_ctrl_deterministic_think_dist_s(
                sid_b,
                int(turn_idx),
                lo_s,
                hi_s,
                eff_dist.encode("utf-8"),
                eff_median,
                eff_sigma,
            )
        )

    # Safe Python fallback when running against an un-rebuilt local `libmorphserve_core.so`.
    eff_lo = max(0.0, lo_s)
    eff_hi = max(eff_lo, hi_s)
    if eff_dist == "constant":
        return max(eff_lo, min(eff_hi, max(0.0, eff_median)))
    if eff_dist == "uniform":
        return float(L.ms_ctrl_deterministic_think_s(sid_b, int(turn_idx), lo_s, hi_s))
    key = f"{session_id}:{int(turn_idx)}"
    h1 = int(hashlib.md5(f"{key}:a".encode("utf-8")).hexdigest()[:8], 16)
    h2 = int(hashlib.md5(f"{key}:b".encode("utf-8")).hexdigest()[:8], 16)
    u1 = max(1e-6, min(1.0 - 1e-6, float(h1) / 4294967296.0))
    u2 = max(1e-6, min(1.0 - 1e-6, float(h2) / 4294967296.0))
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    val = max(1e-6, eff_median) * math.exp(max(0.0, eff_sigma) * z)
    return max(eff_lo, min(eff_hi, val))


@dataclass
class SessionOutcome:
    """Execution outcome and per-class normalized SLO compliance of one Interleaved-Bench session."""

    session_id: str
    quadrant: str
    status: str  # "ok" | "oom" | "error"
    e2e_ms: float
    ar_ttfts_ms: list[float] = field(default_factory=list)
    ar_tpots_ms: list[float] = field(default_factory=list)
    dit_turns_ms: list[float] = field(default_factory=list)
    dit_turn_heights: list[int] = field(default_factory=list)
    dit_turn_steps: list[int] = field(default_factory=list)
    ar_slo_met: bool = True
    dit_slo_met: bool = True
    joint_slo_met: bool = True
    path_a_ships: int = 0
    path_b_morphs: int = 0
    cross_node_ships: int = 0
    cross_node_ship_bytes_list: list[int] = field(default_factory=list)
    ship_bytes_list: list[int] = field(default_factory=list)
    ship_ms_list: list[float] = field(default_factory=list)
    preempted_ar_turns: int = 0
    total_yields: int = 0
    oom_retries: int = 0
    ar_texts: list[str] = field(default_factory=list)
    image_paths_out: list[str] = field(default_factory=list)
    error: str = ""
    arrival_offset_s: float = 0.0
    phase_label: str = ""
    think_time_ms: float = 0.0
    active_jct_ms: float = 0.0
    ar_turn_tpots_ms: list[list[float]] = field(default_factory=list)
    ar_slo_p95_tpot_met: bool = True
    ar_slo_mean_tpot_met: bool = True
    ar_slo_strict_max_tpot_met: bool = True
    turn_records: list[dict[str, Any]] = field(default_factory=list)
    planned_turns: int = 0
    planned_ar_turns: int = 0
    planned_dit_turns: int = 0
    completed_offset_s: float = 0.0
    resident_alpha_sample: float = 1.0


@dataclass
class ClusterRunReport:
    """Aggregated benchmark telemetry for one policy on one suite/QPS point."""

    policy_name: str
    suite_name: str
    num_gpus: int
    offered_qps: float
    wall_seconds: float
    total_sessions: int
    completed_sessions: int
    oom_sessions: int
    error_sessions: int
    joint_slo_goodput_rps: float
    raw_wall_goodput_rps: float
    steady_state_goodput_rps: float
    steady_state_attainment_pct: float
    joint_slo_attainment_pct: float
    ar_slo_attainment_pct: float
    dit_slo_attainment_pct: float
    ar_ttft_p50_ms: float
    ar_ttft_p95_ms: float
    ar_ttft_p99_ms: float
    ar_tpot_p50_ms: float
    ar_tpot_p95_ms: float
    ar_tpot_p99_ms: float
    dit_turn_p50_ms: float
    dit_turn_p95_ms: float
    dit_turn_p99_ms: float
    peak_gpu_reserved_mib: float
    host_shmem_delta_gib: float
    vmm_overflow_creates: int
    vmm_super_to_fine_carves: int
    vmm_expert_evictions_for_kv: int
    vmm_expert_evictions_for_transient: int
    path_a_ships: int
    path_b_morphs: int
    preempted_ar_count: int
    total_yields: int
    worker_oom_count: int
    oom_retry_count: int
    peak_nvml_used_mib: float = 0.0
    peak_pytorch_reserved_mib: float = 0.0
    survival_rate_pct: float = 100.0
    steady_state_survival_pct: float = 100.0
    dit_turn_survival_pct: float = 100.0
    session_e2e_p50_ms: float = 0.0
    session_e2e_p95_ms: float = 0.0
    session_e2e_p99_ms: float = 0.0
    all_session_e2e_p50_ms: float | None = None
    all_session_e2e_p90_ms: float | None = None
    all_session_e2e_p95_ms: float | None = None
    all_session_dit_p50_ms: float | None = None
    all_session_dit_p90_ms: float | None = None
    all_session_dit_p95_ms: float | None = None
    e2e_cdf_points: list[dict[str, Any]] = field(default_factory=list)
    dit_cdf_points: list[dict[str, Any]] = field(default_factory=list)
    cross_node_ships: int = 0
    max_cross_node_ship_bytes: int = 0
    slo_config: dict[str, Any] = field(default_factory=dict)
    slo_curve_by_k: dict[str, dict[str, float]] = field(default_factory=dict)
    slo_curve_by_dit_k: dict[str, dict[str, float]] = field(default_factory=dict)
    status: str = "ok"
    sessions: list[dict[str, Any]] = field(default_factory=list)
    steady_state_window_mode: str = "time"
    steady_window_start_s: float = 0.0
    steady_window_end_s: float = 0.0
    steady_state_sessions: int = 0
    steady_oom_sessions: int = 0
    total_planned_turns: int = 0
    completed_turns: int = 0
    turn_slo_attainment_pct: float = 0.0
    turn_slo_goodput_rps: float = 0.0
    steady_planned_turns: int = 0
    steady_completed_turns: int = 0
    steady_turn_survival_pct: float = 100.0
    steady_turn_slo_attainment_pct: float = 0.0
    steady_turn_ar_slo_attainment_pct: float = 100.0
    steady_turn_dit_slo_attainment_pct: float = 100.0
    steady_turn_slo_goodput_rps: float = 0.0
    steady_turn_goodput_tps: float = 0.0
    effective_slo_throughput_rps: float = 0.0
    steady_ar_slo_attainment_pct: float = 0.0
    steady_ar_slo_mean_pct: float = 0.0
    steady_ar_slo_strict_max_pct: float = 0.0
    steady_dit_slo_attainment_pct: float = 0.0
    active_jct_p50_ms: float = 0.0
    active_jct_p95_ms: float = 0.0
    active_jct_p99_ms: float = 0.0
    all_session_active_jct_p50_ms: float | None = None
    all_session_active_jct_p90_ms: float | None = None
    all_session_active_jct_p95_ms: float | None = None
    active_jct_cdf_points: list[dict[str, Any]] = field(default_factory=list)
    kv_host_swap_outs: int = 0
    per_turn_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    time_series_10s: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MorphServeClusterEngine:
    """Central async coordinator driving `WorkerFleet` + stateful `DualPathMorphRouter`."""

    def __init__(
        self,
        policy: EnginePolicyConfig,
        model_dir: Path = Path("."),
        bench_root: Path = Path("."),
        tmp_root: Path = Path("/tmp/morphserve_engine_tmp"),
        *,
        fleet: WorkerFleet | None = None,
        topology: ClusterTopologyProfile | None = None,
    ) -> None:
        self.policy = policy
        self.model_dir = Path(model_dir)
        self.bench_root = Path(bench_root)
        self.tmp_root = Path(tmp_root)
        self.topo = topology or self._default_topology(policy.num_gpus)
        self.pacer = HalfNodeH2DPacer(self.topo, max_concurrent_per_half=1) if policy.use_h2d_pacer else None
        self.router = DualPathMorphRouter(self.topo, pacer=self.pacer)
        self.fleet: WorkerFleet = fleet or MpSpawnGpuWorkerFleet(policy, self.model_dir, self.bench_root)

        self._pending_futures: dict[str, tuple[int, asyncio.AbstractEventLoop, asyncio.Future]] = {}
        self._fut_lock = threading.Lock()
        self._cmd_seq = 0

        self.worker_states: dict[int, WorkerState] = {}
        self._gpu_locks: list[asyncio.Lock] = []
        self._xfer_locks: list[asyncio.Lock] = []
        self._dst_staging_sems: list[asyncio.Semaphore] = []
        self._inflight_ar_on_gpu: dict[int, int] = {}
        self._ar_drained_events: list[asyncio.Event] = []
        self._active_dit_ranks: dict[int, int] = {}
        self._active_preemptions: dict[int, int] = {}
        self._suites_completed = 0
        self._started = False
        self.host_shmem_delta_gib = 0.0

    @staticmethod
    def _default_topology(num_gpus: int) -> ClusterTopologyProfile:
        if num_gpus == 8:
            return ClusterTopologyProfile.a100_pcie_8gpu_default()
        return ClusterTopologyProfile.from_multinode_measurements(
            world_size=num_gpus,
            gpus_per_node=min(8, max(1, num_gpus)),
            intra_bw=10.6,
            socket_bw=9.55,
            inter_bw=11.5,
            solo_h2d_bw=8.75,
            agg_h2d_bw=10.35,
        )

    def _next_cmd_id(self) -> str:
        self._cmd_seq += 1
        return f"cmd_{self._cmd_seq}"

    def _handle_worker_ack(self, msg: dict[str, Any]) -> None:
        if msg.get("op") == "NODE_DISCONNECTED":
            dead_node = int(msg.get("node_rank", -1))
            gpn = max(1, int(self.topo.gpus_per_node))
            err_payload = {
                "status": "error",
                "error": str(msg.get("error", f"remote node {dead_node} disconnected")),
            }
            victims: list[tuple[asyncio.AbstractEventLoop, asyncio.Future]] = []
            with self._fut_lock:
                for k, (rk, lp, ft) in list(self._pending_futures.items()):
                    if (rk // gpn) == dead_node:
                        self._pending_futures.pop(k, None)
                        victims.append((lp, ft))
            for lp, ft in victims:
                if not lp.is_closed():
                    lp.call_soon_threadsafe(lambda f=ft, m=err_payload: None if f.done() else f.set_result(m))
            return

        cid = msg.get("cmd_id")
        if cid is None:
            return
        with self._fut_lock:
            entry = self._pending_futures.pop(str(cid), None)
        if entry is not None:
            _, loop, fut = entry
            if not loop.is_closed():
                loop.call_soon_threadsafe(lambda f=fut, m=msg: None if f.done() else f.set_result(m))

    def _sync_worker_telemetry(self, rank: int, msg: dict[str, Any]) -> None:
        r = int(rank)
        if r not in self.worker_states:
            return
        alpha = float(msg["resident_alpha"]) if "resident_alpha" in msg else None
        kv_free = int(msg["kv_free_bytes"]) if "kv_free_bytes" in msg else None
        t_seq = int(msg["telemetry_seq"]) if "telemetry_seq" in msg else None
        if alpha is not None or kv_free is not None:
            self.router.update_worker_telemetry(
                r,
                expert_alpha=alpha,
                kv_free_bytes=kv_free,
                telemetry_seq=t_seq,
                workers=self.worker_states,
            )

    def _ensure_async_primitives(self) -> None:
        n_gpus = self.policy.num_gpus
        if len(self._gpu_locks) != n_gpus:
            self._gpu_locks = [asyncio.Lock() for _ in range(n_gpus)]
        if len(self._xfer_locks) != n_gpus:
            self._xfer_locks = [asyncio.Lock() for _ in range(n_gpus)]
        if len(self._dst_staging_sems) != n_gpus:
            self._dst_staging_sems = [asyncio.Semaphore(1) for _ in range(n_gpus)]
        if len(self._ar_drained_events) != n_gpus:
            self._ar_drained_events = []
            for _ in range(n_gpus):
                ev = asyncio.Event()
                ev.set()
                self._ar_drained_events.append(ev)

    def start(self) -> dict[str, Any]:
        if self._started:
            return {}
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        init_summary = self.fleet.start(on_ack=self._handle_worker_ack)
        self.host_shmem_delta_gib = float(init_summary.get("host_shmem_delta_gib", 0.0))

        n = self.policy.num_gpus
        workers_info = init_summary.get("workers") or {}
        free_est = 22 * GIB if self.policy.use_tripool_vmm else 10 * GIB
        alphas = {
            rank: float(
                (workers_info.get(rank) or workers_info.get(str(rank)) or {}).get(
                    "resident_alpha", self.policy.initial_alphas.get(rank, 1.0)
                )
            )
            for rank in range(n)
        }
        kv_frees = {
            rank: int(
                (workers_info.get(rank) or workers_info.get(str(rank)) or {}).get("kv_free_bytes", free_est)
            )
            for rank in range(n)
        }
        self.worker_states = self.router.init_workers(
            num_gpus=n,
            role_of=self.policy.role_of,
            initial_alphas=alphas,
            kv_free_bytes=kv_frees,
        )
        self._active_dit_ranks = {g: 0 for g in range(n)}
        self._active_preemptions = {g: 0 for g in range(n)}
        self._inflight_ar_on_gpu = {g: 0 for g in range(n)}
        self._started = True
        return init_summary

    async def _rpc(self, rank: int, payload: dict[str, Any], timeout_s: float = 600.0) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        cid = payload.get("cmd_id") or self._next_cmd_id()
        payload = dict(payload, cmd_id=cid)
        fut: asyncio.Future = loop.create_future()
        with self._fut_lock:
            self._pending_futures[cid] = (int(rank), loop, fut)
        self.fleet.send(rank, payload)
        try:
            res = await asyncio.wait_for(fut, timeout=timeout_s)
        except Exception:
            with self._fut_lock:
                self._pending_futures.pop(cid, None)
            raise
        self._sync_worker_telemetry(int(rank), res)
        if res.get("status") == "error":
            err_txt = str(res.get("error", ""))
            if "cuda" in err_txt.lower():
                self.router.update_worker_telemetry(
                    int(rank),
                    expert_alpha=0.0,
                    kv_free_bytes=0,
                    workers=self.worker_states,
                )
            raise RuntimeError(
                f"worker rank {rank} error on {payload.get('op')}: {err_txt}\n{res.get('traceback', '')}"
            )
        return res

    def _gpu_busy_map(self) -> dict[int, bool]:
        return {
            g: (
                (g < len(self._gpu_locks) and self._gpu_locks[g].locked())
                or self._inflight_ar_on_gpu.get(g, 0) > 0
            )
            for g in range(self.policy.num_gpus)
        }

    def _pick_initial_gpu(self, modality_plan: str) -> int:
        return self.router.select_initial_gpu(
            num_gpus=self.policy.num_gpus,
            routing_mode=self.policy.routing_mode,
            preemption_mode=self.policy.preemption_mode,
            static_und_ranks=self.policy.static_und_ranks,
            modality_plan=modality_plan,
            workers=self.worker_states,
            gpu_busy=self._gpu_busy_map(),
            active_dit_ranks=self._active_dit_ranks,
            active_preemptions=self._active_preemptions,
            architecture=self.policy.cptg_architecture,
        )

    async def _execute_ar_phase(
        self,
        resident_gpu: int,
        ctx_id: str,
        turn: InterleavedTurn,
        prepare_for_dit: bool,
        use_prev_image: bool = False,
        prev_image_path: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Dispatch AR via mid-DiT preemption OR continuous-batched `EXEC_AR` without single-AR lock serialization."""
        can_preempt = (
            self.policy.preemption_mode != "non_preemptive"
            and (self._gpu_locks[resident_gpu].locked() or self._active_dit_ranks.get(resident_gpu, 0) > 0)
            and self._active_preemptions.get(resident_gpu, 0) < 2
        )
        if can_preempt:
            self._active_preemptions[resident_gpu] += 1
            try:
                ar_res = await self._rpc(
                    resident_gpu,
                    {
                        "op": "EXEC_AR_PREEMPT",
                        "ctx_id": ctx_id,
                        "prompt": turn.prompt,
                        "image_paths": turn.image_paths,
                        "use_prev_image": bool(use_prev_image),
                        "prev_image_path": prev_image_path,
                        "max_ar_tokens": turn.max_ar_tokens,
                        "tpot_slo_ms": self.policy.ar_tpot_slo_ms,
                        "prepare_for_dit": prepare_for_dit,
                    },
                )
            finally:
                self._active_preemptions[resident_gpu] -= 1
            was_mid_dit = bool(ar_res.get("preempted_mid_dit", True)) and ar_res.get("status") == "ok"
            return ar_res, was_mid_dit

        t_enq = time.perf_counter()
        est_ar_ms = DEFAULT_OPERATOR_PROFILE.ar_tpot_ms * max(1, min(turn.max_ar_tokens, 16))
        self.worker_states[resident_gpu].queue_wait_ms += est_ar_ms
        try:
            # Wait only if an exclusive DiT / NCCL operation holds `_gpu_locks`; multiple AR turns
            # dispatch concurrently into `WorkerRuntime` so `decode_ar_batch` batches them!
            while self._gpu_locks[resident_gpu].locked():
                async with self._gpu_locks[resident_gpu]:
                    pass
            # Measure coordinator-side queue wait purely on the Coordinator's monotonic clock and add it
            # to the worker's purely worker-local `ttft_ms` so multi-node `perf_counter()` skew is 0!
            coord_queue_wait_ms = max(0.0, (time.perf_counter() - t_enq) * 1000.0)
            self._inflight_ar_on_gpu[resident_gpu] = self._inflight_ar_on_gpu.get(resident_gpu, 0) + 1
            self._ar_drained_events[resident_gpu].clear()
            try:
                ar_res = await self._rpc(
                    resident_gpu,
                    {
                        "op": "EXEC_AR",
                        "ctx_id": ctx_id,
                        "prompt": turn.prompt,
                        "image_paths": turn.image_paths,
                        "use_prev_image": bool(use_prev_image),
                        "prev_image_path": prev_image_path,
                        "max_ar_tokens": turn.max_ar_tokens,
                        "prepare_for_dit": prepare_for_dit,
                    },
                )
            finally:
                rem = max(0, self._inflight_ar_on_gpu.get(resident_gpu, 1) - 1)
                self._inflight_ar_on_gpu[resident_gpu] = rem
                if rem == 0:
                    self._ar_drained_events[resident_gpu].set()
            if ar_res.get("status") == "ok" and coord_queue_wait_ms > 0.0:
                ar_res = dict(ar_res, ttft_ms=round(float(ar_res.get("ttft_ms", 0.0)) + coord_queue_wait_ms, 2))
            return ar_res, False
        finally:
            self.worker_states[resident_gpu].queue_wait_ms = max(
                0.0, self.worker_states[resident_gpu].queue_wait_ms - est_ar_ms
            )

    async def _acquire_exclusive_gpu(self, gpu: int):
        """Acquire exclusive GPU lock after any non-preempting standalone AR batch finishes draining."""
        await self._gpu_locks[gpu].acquire()
        try:
            await self._ar_drained_events[gpu].wait()
        except BaseException:
            self._gpu_locks[gpu].release()
            raise

    async def _ship_nccl(
        self,
        src_gpu: int,
        dst_gpu: int,
        ctx_id: str,
        new_ctx_id: str,
        include_cfg_dup: bool,
        *,
        release_src_ctx: bool = True,
        defer_unpack: bool = False,
    ) -> dict[str, Any]:
        """Decoupled NCCL P2P transfer: pack on `src_gpu` & release lock immediately, then stream over `xfer_cmd_q`."""
        self._ensure_async_primitives()

        await self._acquire_exclusive_gpu(src_gpu)
        try:
            pack_res = await self._rpc(
                src_gpu,
                {
                    "op": "PACK_CTX",
                    "ctx_id": ctx_id,
                    "include_cfg_dup": include_cfg_dup,
                    "release_after_pack": release_src_ctx,
                },
            )
        finally:
            self._gpu_locks[src_gpu].release()

        if pack_res.get("status") != "ok":
            return dict(pack_res, packed_ok=False)

        packed_id = str(pack_res["packed_id"])
        meta = pack_res["meta"]
        wire_bytes = int(meta["total_bytes"])
        logical_bytes = int(meta.get("logical_bytes", wire_bytes))

        staging_sem = self._dst_staging_sems[dst_gpu]
        await staging_sem.acquire()
        holding_sem = True
        holding_dst_lock = False
        recv_id: str | None = None
        packed_consumed = False
        try:
            alloc_res = await self._rpc(dst_gpu, {"op": "ALLOC_RECV_BUF", "nbytes": wire_bytes, "probe_only": True})
            if alloc_res.get("status") != "ok":
                if self._gpu_locks[dst_gpu].locked() or self._active_dit_ranks.get(dst_gpu, 0) > 0:
                    await self._acquire_exclusive_gpu(dst_gpu)
                    holding_dst_lock = True
                alloc_res = await self._rpc(
                    dst_gpu,
                    {"op": "ALLOC_RECV_BUF", "nbytes": wire_bytes, "probe_only": False},
                )
                if alloc_res.get("status") != "ok":
                    try:
                        await self._rpc(src_gpu, {"op": "DROP_PACKED_CTX", "packed_id": packed_id})
                        packed_consumed = True
                    except Exception:
                        pass
                    return {"status": "oom", "packed_ok": True, "error": alloc_res.get("error", "alloc_recv_oom")}

            recv_id = str(alloc_res["recv_id"])
            lo, hi = (src_gpu, dst_gpu) if src_gpu < dst_gpu else (dst_gpu, src_gpu)
            async with self._xfer_locks[lo]:
                async with self._xfer_locks[hi]:
                    s_res, r_res = await asyncio.gather(
                        self._rpc(
                            src_gpu,
                            {"op": "SEND_NCCL", "packed_id": packed_id, "dst_rank": dst_gpu},
                        ),
                        self._rpc(
                            dst_gpu,
                            {"op": "RECV_NCCL", "src_rank": src_gpu, "recv_id": recv_id, "meta": meta},
                        ),
                    )
                    packed_consumed = True

            if s_res.get("status") != "ok" or r_res.get("status") != "ok":
                try:
                    await self._rpc(dst_gpu, {"op": "DROP_RECV_BUF", "recv_id": recv_id})
                except Exception:
                    pass
                recv_id = None
                return {"status": "oom", "packed_ok": True, "error": "nccl_xfer_oom"}

            ship_ms = max(float(s_res.get("ship_ms", 0.0)), float(r_res.get("ship_ms", 0.0)))
            if defer_unpack:
                out_d = {
                    "status": "ok",
                    "packed_ok": True,
                    "ship_ms": ship_ms,
                    "bytes": logical_bytes,
                    "recv_id": recv_id,
                    "meta": meta,
                    "_holding_staging_sem": True,
                    "_holding_dst_lock": holding_dst_lock,
                }
                holding_sem = False
                holding_dst_lock = False
                recv_id = None
                return out_d

            if not holding_dst_lock:
                await self._acquire_exclusive_gpu(dst_gpu)
                holding_dst_lock = True
            u_res = await self._rpc(
                dst_gpu,
                {"op": "UNPACK_CTX", "recv_id": recv_id, "new_ctx_id": new_ctx_id, "meta": meta},
            )
            recv_id = None
            if u_res.get("status") != "ok":
                return {"status": "oom", "packed_ok": True, "error": u_res.get("error", "unpack_oom")}
            return {
                "status": "ok",
                "packed_ok": True,
                "ship_ms": ship_ms,
                "bytes": logical_bytes,
                "kv_free_bytes": u_res.get("kv_free_bytes"),
                "telemetry_seq": u_res.get("telemetry_seq"),
            }
        finally:
            if not packed_consumed:
                try:
                    await self._rpc(src_gpu, {"op": "DROP_PACKED_CTX", "packed_id": packed_id})
                except Exception:
                    pass
            if recv_id is not None:
                try:
                    await self._rpc(dst_gpu, {"op": "DROP_RECV_BUF", "recv_id": recv_id})
                except Exception:
                    pass
            if holding_dst_lock:
                self._gpu_locks[dst_gpu].release()
            if holding_sem:
                staging_sem.release()

    async def _ship_shm(
        self,
        src_gpu: int,
        dst_gpu: int,
        ctx_id: str,
        new_ctx_id: str,
        *,
        include_cfg_dup: bool = True,
        release_src_ctx: bool = True,
    ) -> dict[str, Any]:
        self._ensure_async_primitives()
        shm_name = f"ms_shm_{os.getpid()}_{src_gpu}_{dst_gpu}_{self._next_cmd_id()}"
        await self._acquire_exclusive_gpu(src_gpu)
        try:
            s_res = await self._rpc(
                src_gpu,
                {
                    "op": "SEND_SHM",
                    "ctx_id": ctx_id,
                    "shm_name": shm_name,
                    "include_cfg_dup": include_cfg_dup,
                    "release_after_send": release_src_ctx,
                },
            )
            if s_res.get("status") != "ok":
                return dict(s_res, sent_ok=False)
        finally:
            self._gpu_locks[src_gpu].release()

        await self._acquire_exclusive_gpu(dst_gpu)
        try:
            r_res = await self._rpc(
                dst_gpu,
                {"op": "RECV_SHM", "shm_name": shm_name, "new_ctx_id": new_ctx_id, "meta": s_res["meta"]},
            )
            r_res["bytes"] = int(s_res.get("bytes", 0))
            r_res["sent_ok"] = True
            return r_res
        finally:
            self._gpu_locks[dst_gpu].release()

    def _build_cptg_spec(
        self,
        req_id: str,
        src_gpu: int,
        state_bytes: int,
        turn: InterleavedTurn,
        gamma: int,
    ) -> CPTGRequestSpec:
        arch = self.policy.cptg_architecture
        fam = "internvl_u" if arch == "bridge_mllm_dit" else "bagel"
        step1_ms = DEFAULT_OPERATOR_PROFILE.step_ms(fam, turn.height)
        exp_bytes = DEFAULT_OPERATOR_PROFILE.expert_bytes(fam)
        if arch == "bridge_mllm_dit":
            eff = max(state_bytes, 64 * 1024)
            return CPTGRequestSpec(
                req_id=req_id,
                src_gpu=src_gpu,
                architecture="bridge_mllm_dit",
                kv_bytes=eff,
                bridge_bytes=eff,
                gen_expert_bytes=exp_bytes,
                dit_step1_ms=step1_ms,
                dit_total_steps=turn.dit_steps,
                resolution=turn.height,
                interleaved_roundtrip_gamma=gamma,
            )
        return CPTGRequestSpec(
            req_id=req_id,
            src_gpu=src_gpu,
            architecture="tight_mot",
            kv_bytes=max(state_bytes, 1 * MIB),
            gen_expert_bytes=exp_bytes,
            dit_step1_ms=step1_ms,
            dit_total_steps=turn.dit_steps,
            resolution=turn.height,
            interleaved_roundtrip_gamma=gamma,
        )

    def _pick_oom_retry_gpu(self, failed_gpu: int, modality_plan: str) -> int:
        del modality_plan
        return self.router.select_oom_retry_gpu(
            num_gpus=self.policy.num_gpus,
            routing_mode=self.policy.routing_mode,
            static_und_ranks=self.policy.static_und_ranks,
            failed_gpu=failed_gpu,
            gpu_busy=self._gpu_busy_map(),
        )

    async def _run_session_coro(
        self,
        sess: InterleavedSession,
        suite_t0: float,
        save_outputs_dir: Path | None = None,
    ) -> SessionOutcome:
        delay = (suite_t0 + sess.arrival_offset_s) - time.perf_counter()
        if delay > 0.0:
            await asyncio.sleep(delay)

        t_sess0 = time.perf_counter()
        resident_gpu: int | None = None
        ctx_id = f"ctx_{sess.session_id}"
        src_ctx_alive = False
        kv_bytes = 0
        prev_tmp_img: str | None = None
        owned_tmp_imgs: list[str] = []

        ar_ttfts: list[float] = []
        ar_tpots: list[float] = []
        ar_turn_tpots: list[list[float]] = []
        dit_turns: list[float] = []
        dit_heights: list[int] = []
        dit_steps_list: list[int] = []
        ar_texts: list[str] = []
        image_paths_out: list[str] = []
        turn_records: list[dict[str, Any]] = []
        path_a_ships = 0
        path_b_morphs = 0
        cross_node_ships = 0
        cross_node_ship_bytes_list: list[int] = []
        ship_bytes_list: list[int] = []
        ship_ms_list: list[float] = []
        preempted_ar_turns = 0
        total_yields = 0
        oom_retries = 0
        think_time_ms = 0.0
        is_tight_mot = self.policy.cptg_architecture == "tight_mot"
        cur_turn_idx = 0
        cur_turn_plan = ""
        cur_turn_height = 0

        def _sample_resident_alpha() -> float:
            if not self.worker_states:
                return 1.0
            return round(
                sum(float(w.expert_alpha) for w in self.worker_states.values()) / float(len(self.worker_states)),
                4,
            )

        planned_turns = len(sess.turns)
        planned_ar_turns = sum(1 for t in sess.turns if t.modality_plan in ("ar_only", "cot_then_dit"))
        planned_dit_turns = sum(1 for t in sess.turns if t.modality_plan != "ar_only")

        def _make_outcome(
            status: str,
            *,
            ar_ok: bool = False,
            ar_p95_ok: bool = False,
            ar_mean_ok: bool = False,
            ar_strict_ok: bool = False,
            dit_ok: bool = False,
            error: str = "",
        ) -> SessionOutcome:
            now_pc = time.perf_counter()
            e2e_ms = round((now_pc - t_sess0) * 1000.0, 2)
            active_jct_ms = round(max(0.0, e2e_ms - think_time_ms), 2)
            completed_offset_s = round(max(0.0, now_pc - suite_t0), 3)
            return SessionOutcome(
                session_id=sess.session_id,
                quadrant=sess.quadrant,
                status=status,
                e2e_ms=e2e_ms,
                ar_ttfts_ms=ar_ttfts,
                ar_tpots_ms=ar_tpots,
                dit_turns_ms=[round(x, 2) for x in dit_turns],
                dit_turn_heights=dit_heights,
                dit_turn_steps=dit_steps_list,
                ar_slo_met=ar_ok,
                dit_slo_met=dit_ok,
                joint_slo_met=(ar_ok and dit_ok) if status == "ok" else False,
                path_a_ships=path_a_ships,
                path_b_morphs=path_b_morphs,
                cross_node_ships=cross_node_ships,
                cross_node_ship_bytes_list=cross_node_ship_bytes_list,
                ship_bytes_list=ship_bytes_list,
                ship_ms_list=[round(x, 3) for x in ship_ms_list],
                preempted_ar_turns=preempted_ar_turns,
                total_yields=total_yields,
                oom_retries=oom_retries,
                ar_texts=ar_texts,
                image_paths_out=image_paths_out,
                error=error,
                arrival_offset_s=float(sess.arrival_offset_s),
                phase_label=str(sess.phase_label),
                think_time_ms=round(think_time_ms, 2),
                active_jct_ms=active_jct_ms,
                ar_turn_tpots_ms=ar_turn_tpots,
                ar_slo_p95_tpot_met=ar_p95_ok,
                ar_slo_mean_tpot_met=ar_mean_ok,
                ar_slo_strict_max_tpot_met=ar_strict_ok,
                turn_records=turn_records,
                planned_turns=planned_turns,
                planned_ar_turns=planned_ar_turns,
                planned_dit_turns=planned_dit_turns,
                completed_offset_s=completed_offset_s,
                resident_alpha_sample=_sample_resident_alpha(),
            )

        try:
            for ti, turn in enumerate(sess.turns):
                cur_turn_idx = int(ti)
                cur_turn_plan = str(turn.modality_plan)
                cur_turn_height = 0 if cur_turn_plan == "ar_only" else int(turn.height)
                if ti > 0:
                    # Realistic user think-time residency (`D1`) holding live KV across turns!
                    think_s = _deterministic_think_s(sess.session_id, ti, self.policy)
                    if think_s > 0.0:
                        t_th0 = time.perf_counter()
                        await asyncio.sleep(think_s)
                        think_time_ms += (time.perf_counter() - t_th0) * 1000.0

                plan = turn.modality_plan
                is_edit_followup = (plan == "dit_edit") and (ti > 0)
                turn_use_prev_image = (ti > 0) and (bool(turn.use_prev_image) or (plan == "dit_edit"))
                has_more_turns = (ti + 1) < len(sess.turns)
                has_more_edits = any(t.modality_plan == "dit_edit" for t in sess.turns[ti + 1 :])
                has_future_prev_image = any(
                    bool(t.use_prev_image) or (t.modality_plan == "dit_edit") for t in sess.turns[ti + 1 :]
                )

                # Bounded OOM retry loop (`D3`: `max_oom_retries = 1` with user arrival clock running!)
                attempt = 0
                while True:
                    if resident_gpu is None:
                        resident_gpu = self._pick_initial_gpu(plan)
                    eff_prev_img = (None if (is_tight_mot and src_ctx_alive) else prev_tmp_img) if turn_use_prev_image else None

                    # ---- Branch 1: Pure AR turn (VQA) ----
                    if plan == "ar_only":
                        ar_res, did_preempt = await self._execute_ar_phase(
                            resident_gpu,
                            ctx_id,
                            turn,
                            prepare_for_dit=False,
                            use_prev_image=turn_use_prev_image,
                            prev_image_path=eff_prev_img,
                        )
                        if ar_res["status"] == "oom":
                            try:
                                await self._rpc(resident_gpu, {"op": "RELEASE_CTX", "ctx_id": ctx_id})
                            except Exception:
                                pass
                            src_ctx_alive = False
                            if attempt < self.policy.max_oom_retries and ti == 0:
                                attempt += 1
                                oom_retries += 1
                                resident_gpu = self._pick_oom_retry_gpu(resident_gpu, plan)
                                ctx_id = f"ctx_{sess.session_id}_r{oom_retries}"
                                continue
                            turn_records.append(
                                {
                                    "turn_idx": int(ti),
                                    "modality_plan": plan,
                                    "status": "oom",
                                    "ar_ttft_ms": None,
                                    "ar_tpot_p95_ms": None,
                                    "ar_tpot_mean_ms": None,
                                    "dit_ms": None,
                                    "height": 0,
                                    "kv_bytes": int(kv_bytes),
                                    "shipped": False,
                                    "morphed": False,
                                }
                            )
                            return _make_outcome("oom", error=ar_res.get("error", "oom"))
                        src_ctx_alive = True
                        if did_preempt:
                            preempted_ar_turns += 1
                        kv_bytes = int(ar_res.get("kv_bytes", kv_bytes))
                        turn_ttft = float(ar_res.get("ttft_ms", 0.0))
                        turn_tpots = [float(x) for x in ar_res.get("tpots_ms", [])]
                        ar_ttfts.append(turn_ttft)
                        ar_tpots.extend(turn_tpots)
                        ar_turn_tpots.append(turn_tpots)
                        ar_texts.append(str(ar_res.get("text", "")))
                        turn_records.append(
                            {
                                "turn_idx": int(ti),
                                "modality_plan": plan,
                                "status": "ok",
                                "ar_ttft_ms": round(turn_ttft, 2),
                                "ar_tpot_p95_ms": round(_percentile(turn_tpots, 95.0), 2) if turn_tpots else None,
                                "ar_tpot_mean_ms": (
                                    round(sum(turn_tpots) / float(len(turn_tpots)), 2) if turn_tpots else None
                                ),
                                "ar_tpot_strict_max_ms": round(max(turn_tpots), 2) if turn_tpots else None,
                                "dit_ms": None,
                                "height": 0,
                                "dit_steps": 0,
                                "kv_bytes": int(kv_bytes),
                                "shipped": False,
                                "morphed": False,
                            }
                        )
                        break

                    # ---- Branch 2: Visual-CoT / DiT Edit / High-Res T2I ----
                    # D3: For `cot_then_dit`, decouple AR CoT timing from DiT stage timing (`t_dit0` starts
                    # strictly when AR CoT completes, covering routing, transfer/morph, queue, denoising, VAE).
                    t_dit0 = time.perf_counter()
                    cot_text_pending: str | None = None
                    turn_ar_ttft: float | None = None
                    turn_ar_tpots: list[float] = []
                    turn_shipped = False
                    turn_morphed = False
                    if plan == "cot_then_dit":
                        cot_res, did_preempt = await self._execute_ar_phase(
                            resident_gpu,
                            ctx_id,
                            turn,
                            prepare_for_dit=True,
                            use_prev_image=turn_use_prev_image,
                            prev_image_path=eff_prev_img,
                        )
                        if cot_res["status"] == "oom":
                            try:
                                await self._rpc(resident_gpu, {"op": "RELEASE_CTX", "ctx_id": ctx_id})
                            except Exception:
                                pass
                            src_ctx_alive = False
                            if attempt < self.policy.max_oom_retries and ti == 0:
                                attempt += 1
                                oom_retries += 1
                                resident_gpu = self._pick_oom_retry_gpu(resident_gpu, plan)
                                ctx_id = f"ctx_{sess.session_id}_r{oom_retries}"
                                continue
                            turn_records.append(
                                {
                                    "turn_idx": int(ti),
                                    "modality_plan": plan,
                                    "status": "oom",
                                    "ar_ttft_ms": None,
                                    "ar_tpot_p95_ms": None,
                                    "ar_tpot_mean_ms": None,
                                    "ar_tpot_strict_max_ms": None,
                                    "dit_ms": None,
                                    "height": int(turn.height),
                                    "dit_steps": int(turn.dit_steps),
                                    "kv_bytes": int(kv_bytes),
                                    "shipped": False,
                                    "morphed": False,
                                }
                            )
                            return _make_outcome("oom", error=cot_res.get("error", "oom"))
                        src_ctx_alive = True
                        if did_preempt:
                            preempted_ar_turns += 1
                        kv_bytes = int(cot_res.get("kv_bytes", kv_bytes))
                        turn_ar_ttft = float(cot_res.get("ttft_ms", 0.0))
                        turn_ar_tpots = [float(x) for x in cot_res.get("tpots_ms", [])]
                        ar_ttfts.append(turn_ar_ttft)
                        ar_tpots.extend(turn_ar_tpots)
                        ar_turn_tpots.append(turn_ar_tpots)
                        cot_text_pending = str(cot_res.get("text", ""))
                        t_dit0 = time.perf_counter()
                    else:
                        est_prep_ms = DEFAULT_OPERATOR_PROFILE.prep_overhead_ms
                        self.worker_states[resident_gpu].queue_wait_ms += est_prep_ms
                        try:
                            await self._acquire_exclusive_gpu(resident_gpu)
                            try:
                                # When `tight_mot` already holds live context (`last_img_u8` / `last_img_pil`),
                                # pass `prev_image_path=None` so `GpuInPlaceLoopback` is never bypassed by disk PNGs!
                                prep_res = await self._rpc(
                                    resident_gpu,
                                    {
                                        "op": "PREP_DIT",
                                        "ctx_id": ctx_id,
                                        "prompt": turn.prompt,
                                        "height": turn.height,
                                        "width": turn.width,
                                        "image_paths": (
                                            turn.image_paths
                                            if (ti == 0)
                                            else ([] if is_edit_followup else turn.image_paths)
                                        ),
                                        "use_prev_image": turn_use_prev_image,
                                        "prev_image_path": eff_prev_img,
                                        "prepare_for_dit": True,
                                    },
                                )
                            finally:
                                self._gpu_locks[resident_gpu].release()
                        finally:
                            self.worker_states[resident_gpu].queue_wait_ms = max(
                                0.0, self.worker_states[resident_gpu].queue_wait_ms - est_prep_ms
                            )
                        if prep_res["status"] == "oom":
                            try:
                                await self._rpc(resident_gpu, {"op": "RELEASE_CTX", "ctx_id": ctx_id})
                            except Exception:
                                pass
                            src_ctx_alive = False
                            if attempt < self.policy.max_oom_retries and ti == 0:
                                attempt += 1
                                oom_retries += 1
                                resident_gpu = self._pick_oom_retry_gpu(resident_gpu, plan)
                                ctx_id = f"ctx_{sess.session_id}_r{oom_retries}"
                                continue
                            turn_records.append(
                                {
                                    "turn_idx": int(ti),
                                    "modality_plan": plan,
                                    "status": "oom",
                                    "ar_ttft_ms": None,
                                    "ar_tpot_p95_ms": None,
                                    "ar_tpot_mean_ms": None,
                                    "dit_ms": None,
                                    "height": int(turn.height),
                                    "kv_bytes": int(kv_bytes),
                                    "shipped": False,
                                    "morphed": False,
                                }
                            )
                            return _make_outcome("oom", error=prep_res.get("error", "oom"))
                        src_ctx_alive = True
                        kv_bytes = int(prep_res.get("kv_bytes", kv_bytes))

                    # Route cross-paradigm transition & atomically reserve dst_gpu queue + KV + H2D pacer BEFORE shipping!
                    gamma = 2 if (plan == "dit_edit" and has_more_edits) else 1
                    req = self._build_cptg_spec(
                        req_id=f"{sess.session_id}_t{ti}_a{attempt}",
                        src_gpu=resident_gpu,
                        state_bytes=kv_bytes,
                        turn=turn,
                        gamma=gamma,
                    )
                    fam = "internvl_u" if self.policy.model_family == "internvl_u" else "bagel"
                    est_dit_ms = DEFAULT_OPERATOR_PROFILE.unloaded_dit_turn_ms(fam, turn.height, steps=turn.dit_steps)
                    now_rel_ms = (time.perf_counter() - suite_t0) * 1000.0
                    dec, ship_mode = self.router.plan_and_reserve_transition(
                        routing_mode=self.policy.routing_mode,
                        static_und_ranks=self.policy.static_und_ranks,
                        static_gen_ranks=self.policy.static_gen_ranks,
                        req=req,
                        est_dit_ms=est_dit_ms,
                        workers=self.worker_states,
                        gpu_busy=self._gpu_busy_map(),
                        use_pacer=self.policy.use_h2d_pacer,
                        now_ms=now_rel_ms,
                    )
                    dst_gpu = dec.dst_gpu
                    src_gpu_before_ship = resident_gpu
                    reserved_kv_b = (
                        int(req.effective_state_bytes()) if (ship_mode != "none" and dst_gpu != resident_gpu) else 0
                    )
                    kv_res_released = reserved_kv_b == 0
                    dit_ctx_id = ctx_id
                    deferred_ship: dict[str, Any] | None = None
                    save_tmp: str | None = None
                    dit_res: dict[str, Any] = {"status": "error", "error": "not_run"}

                    ephemeral_gen_ctx = ship_mode in ("nccl_static", "shm_static")
                    needs_return_ship = ephemeral_gen_ctx and has_more_turns and is_tight_mot

                    try:
                        if ship_mode == "nccl_dual":
                            ship = await self._ship_nccl(
                                resident_gpu,
                                dst_gpu,
                                ctx_id,
                                ctx_id,
                                include_cfg_dup=False,
                                release_src_ctx=True,
                                defer_unpack=True,
                            )
                            if ship.get("packed_ok"):
                                src_ctx_alive = False
                            if ship["status"] != "ok":
                                dit_res = {"status": "oom", "error": ship.get("error", "nccl_ship_oom")}
                            else:
                                deferred_ship = ship
                                dit_ctx_id = ctx_id
                        elif ship_mode == "nccl_static":
                            gen_ctx_id = f"{ctx_id}_gen_t{ti}"
                            release_src = True if is_tight_mot else (not has_more_turns)
                            # Fair strong NCCL fission (`Decision 2/6`): non-duplicated CFG wire payload +
                            # full post-DiT `Text + ViT + VAE` KV continuity across turns!
                            ship = await self._ship_nccl(
                                resident_gpu,
                                dst_gpu,
                                ctx_id,
                                gen_ctx_id,
                                include_cfg_dup=False,
                                release_src_ctx=release_src,
                                defer_unpack=True,
                            )
                            if release_src and ship.get("packed_ok"):
                                src_ctx_alive = False
                            if ship["status"] != "ok":
                                dit_res = {"status": "oom", "error": ship.get("error", "static_ship_oom")}
                            else:
                                deferred_ship = ship
                                dit_ctx_id = gen_ctx_id
                        elif ship_mode == "shm_static":
                            gen_ctx_id = f"{ctx_id}_gen_t{ti}"
                            release_src = True if is_tight_mot else (not has_more_turns)
                            ship = await self._ship_shm(
                                resident_gpu,
                                dst_gpu,
                                ctx_id,
                                gen_ctx_id,
                                include_cfg_dup=True,
                                release_src_ctx=release_src,
                            )
                            if release_src and ship.get("sent_ok"):
                                src_ctx_alive = False
                            self.router.release_kv_reservation(
                                dst_gpu,
                                reserved_kv_b,
                                latest_kv_free_bytes=ship.get("kv_free_bytes"),
                                telemetry_seq=ship.get("telemetry_seq"),
                                workers=self.worker_states,
                            )
                            kv_res_released = True
                            if ship["status"] != "ok":
                                dit_res = {"status": "oom", "error": ship.get("error", "shm_ship_oom")}
                            else:
                                dit_ctx_id = gen_ctx_id
                                path_a_ships += 1
                                turn_shipped = True
                                fwd_b = int(ship.get("bytes", kv_bytes))
                                if not self.topo.same_node(src_gpu_before_ship, dst_gpu):
                                    cross_node_ships += 1
                                    cross_node_ship_bytes_list.append(fwd_b)
                                ship_bytes_list.append(fwd_b)
                        elif self.policy.routing_mode == "dual_path":
                            path_b_morphs += 1
                            turn_morphed = True

                        if dit_res.get("status") != "oom":
                            if save_outputs_dir is not None:
                                save_tmp = str(save_outputs_dir / f"{sess.session_id}_t{ti}.png")
                            elif (
                                has_future_prev_image
                                and (not is_tight_mot)
                                and self.policy.routing_mode in ("twostage_shm", "static_nccl_fission")
                            ):
                                save_tmp = str(self.tmp_root / f"{sess.session_id}_t{ti}.png")
                                owned_tmp_imgs.append(save_tmp)

                            already_locked = bool(deferred_ship and deferred_ship.get("_holding_dst_lock"))
                            holding_sem = bool(deferred_ship and deferred_ship.get("_holding_staging_sem"))
                            if not already_locked:
                                try:
                                    await self._acquire_exclusive_gpu(dst_gpu)
                                except BaseException:
                                    if deferred_ship is not None and deferred_ship.get("recv_id"):
                                        try:
                                            await self._rpc(
                                                dst_gpu, {"op": "DROP_RECV_BUF", "recv_id": deferred_ship["recv_id"]}
                                            )
                                        except Exception:
                                            pass
                                    if holding_sem:
                                        self._dst_staging_sems[dst_gpu].release()
                                    raise

                            try:
                                if deferred_ship is not None:
                                    try:
                                        u_res = await self._rpc(
                                            dst_gpu,
                                            {
                                                "op": "UNPACK_CTX",
                                                "recv_id": deferred_ship["recv_id"],
                                                "new_ctx_id": dit_ctx_id,
                                                "meta": deferred_ship["meta"],
                                            },
                                        )
                                    finally:
                                        if holding_sem:
                                            self._dst_staging_sems[dst_gpu].release()
                                            holding_sem = False
                                    self.router.release_kv_reservation(
                                        dst_gpu,
                                        reserved_kv_b,
                                        latest_kv_free_bytes=u_res.get("kv_free_bytes"),
                                        telemetry_seq=u_res.get("telemetry_seq"),
                                        workers=self.worker_states,
                                    )
                                    kv_res_released = True
                                    if u_res.get("status") != "ok":
                                        dit_res = {"status": "oom", "error": u_res.get("error", "unpack_oom")}
                                    else:
                                        if ship_mode == "nccl_dual":
                                            resident_gpu = dst_gpu
                                            src_ctx_alive = True
                                        path_a_ships += 1
                                        turn_shipped = True
                                        fwd_b = int(deferred_ship.get("bytes", kv_bytes))
                                        if not self.topo.same_node(src_gpu_before_ship, dst_gpu):
                                            cross_node_ships += 1
                                            cross_node_ship_bytes_list.append(fwd_b)
                                        ship_bytes_list.append(fwd_b)
                                        ship_ms_list.append(float(deferred_ship.get("ship_ms", 0.0)))

                                if dit_res.get("status") != "oom":
                                    self._active_dit_ranks[dst_gpu] += 1
                                    try:
                                        dit_res = await self._rpc(
                                            dst_gpu,
                                            {
                                                "op": "EXEC_DIT",
                                                "ctx_id": dit_ctx_id,
                                                "prompt": turn.prompt,
                                                "height": turn.height,
                                                "width": turn.width,
                                                "dit_steps": turn.dit_steps,
                                                "seed": 52 + ti,
                                                "save_tmp_image_path": save_tmp,
                                                "release_after_dit": ephemeral_gen_ctx and (not needs_return_ship),
                                            },
                                        )
                                        if ephemeral_gen_ctx and dit_res.get("status") != "ok":
                                            try:
                                                await self._rpc(dst_gpu, {"op": "RELEASE_CTX", "ctx_id": dit_ctx_id})
                                            except Exception:
                                                pass
                                    finally:
                                        self._active_dit_ranks[dst_gpu] -= 1
                            finally:
                                if holding_sem:
                                    self._dst_staging_sems[dst_gpu].release()
                                self._gpu_locks[dst_gpu].release()
                    finally:
                        if not kv_res_released:
                            self.router.release_kv_reservation(dst_gpu, reserved_kv_b, workers=self.worker_states)
                        self.router.release_dit_reservation(dst_gpu, est_dit_ms, workers=self.worker_states)

                    # Return-ship updated post-DiT `main_ctx` (`Text + ViT + VAE` KV + `last_img`) back to `und` rank
                    # for multi-turn `tight_mot` disaggregated policies (`static_nccl_fission`, `twostage_shm`).
                    if dit_res.get("status") == "ok" and needs_return_ship:
                        ret_kv_b = max(0, int(dit_res.get("kv_bytes", kv_bytes)))
                        self.router.reserve_kv_bytes(resident_gpu, ret_kv_b, workers=self.worker_states)
                        ret_ship: dict[str, Any] = {}
                        try:
                            if ship_mode == "nccl_static":
                                ret_ship = await self._ship_nccl(
                                    dst_gpu,
                                    resident_gpu,
                                    dit_ctx_id,
                                    ctx_id,
                                    include_cfg_dup=False,
                                    release_src_ctx=True,
                                    defer_unpack=False,
                                )
                            else:
                                ret_ship = await self._ship_shm(
                                    dst_gpu,
                                    resident_gpu,
                                    dit_ctx_id,
                                    ctx_id,
                                    include_cfg_dup=False,
                                    release_src_ctx=True,
                                )
                        finally:
                            self.router.release_kv_reservation(
                                resident_gpu,
                                ret_kv_b,
                                latest_kv_free_bytes=ret_ship.get("kv_free_bytes"),
                                telemetry_seq=ret_ship.get("telemetry_seq"),
                                workers=self.worker_states,
                            )
                        if ret_ship.get("status") != "ok":
                            if not ret_ship.get("packed_ok") and not ret_ship.get("sent_ok"):
                                try:
                                    await self._rpc(dst_gpu, {"op": "RELEASE_CTX", "ctx_id": dit_ctx_id})
                                except Exception:
                                    pass
                            dit_res = {"status": "oom", "error": ret_ship.get("error", "return_ship_oom")}
                        else:
                            src_ctx_alive = True
                            ret_b = int(ret_ship.get("bytes", ret_kv_b))
                            kv_bytes = ret_b
                            if not self.topo.same_node(dst_gpu, resident_gpu):
                                cross_node_ships += 1
                                cross_node_ship_bytes_list.append(ret_b)
                            if ship_ms_list and ship_mode == "nccl_static":
                                ship_ms_list[-1] = round(
                                    ship_ms_list[-1] + float(ret_ship.get("ship_ms", 0.0)), 3
                                )

                    if dit_res["status"] == "oom":
                        for g_clean, c_clean in {(resident_gpu, ctx_id), (dst_gpu, dit_ctx_id)}:
                            if g_clean is not None:
                                try:
                                    await self._rpc(g_clean, {"op": "RELEASE_CTX", "ctx_id": c_clean})
                                except Exception:
                                    pass
                        src_ctx_alive = False
                        if attempt < self.policy.max_oom_retries and ti == 0:
                            attempt += 1
                            oom_retries += 1
                            resident_gpu = self._pick_oom_retry_gpu(dst_gpu, plan)
                            ctx_id = f"ctx_{sess.session_id}_r{oom_retries}"
                            # Keep t_sess0 ticking so wasted OOM attempt counts toward latency!
                            continue
                        turn_records.append(
                            {
                                "turn_idx": int(ti),
                                "modality_plan": plan,
                                "status": "oom",
                                "ar_ttft_ms": round(turn_ar_ttft, 2) if turn_ar_ttft is not None else None,
                                "ar_tpot_p95_ms": (
                                    round(_percentile(turn_ar_tpots, 95.0), 2) if turn_ar_tpots else None
                                ),
                                "ar_tpot_mean_ms": (
                                    round(sum(turn_ar_tpots) / float(len(turn_ar_tpots)), 2) if turn_ar_tpots else None
                                ),
                                "ar_tpot_strict_max_ms": round(max(turn_ar_tpots), 2) if turn_ar_tpots else None,
                                "dit_ms": None,
                                "height": int(turn.height),
                                "dit_steps": int(turn.dit_steps),
                                "kv_bytes": int(kv_bytes),
                                "shipped": turn_shipped,
                                "morphed": turn_morphed,
                            }
                        )
                        return _make_outcome("oom", error=dit_res.get("error", "oom"))

                    if cot_text_pending is not None:
                        ar_texts.append(cot_text_pending)
                    self._sync_worker_telemetry(dst_gpu, dit_res)
                    total_yields += int(dit_res.get("yield_count", 0))
                    if not needs_return_ship:
                        kv_bytes = int(dit_res.get("kv_bytes", kv_bytes))
                    if save_tmp is not None:
                        prev_tmp_img = save_tmp
                        if save_outputs_dir is not None:
                            image_paths_out.append(save_tmp)

                    dit_ms = (time.perf_counter() - t_dit0) * 1000.0
                    dit_turns.append(dit_ms)
                    dit_heights.append(int(turn.height))
                    dit_steps_list.append(int(turn.dit_steps))
                    turn_records.append(
                        {
                            "turn_idx": int(ti),
                            "modality_plan": plan,
                            "status": "ok",
                            "ar_ttft_ms": round(turn_ar_ttft, 2) if turn_ar_ttft is not None else None,
                            "ar_tpot_p95_ms": (
                                round(_percentile(turn_ar_tpots, 95.0), 2) if turn_ar_tpots else None
                            ),
                            "ar_tpot_mean_ms": (
                                round(sum(turn_ar_tpots) / float(len(turn_ar_tpots)), 2) if turn_ar_tpots else None
                            ),
                            "ar_tpot_strict_max_ms": round(max(turn_ar_tpots), 2) if turn_ar_tpots else None,
                            "dit_ms": round(dit_ms, 2),
                            "height": int(turn.height),
                            "dit_steps": int(turn.dit_steps),
                            "kv_bytes": int(kv_bytes),
                            "shipped": turn_shipped,
                            "morphed": turn_morphed,
                        }
                    )
                    break

            ttft_ok = all(x <= self.policy.ar_ttft_slo_ms for x in ar_ttfts)
            ar_p95_ok = ttft_ok and all(
                _percentile(t_list, 95.0) <= self.policy.ar_tpot_slo_ms for t_list in ar_turn_tpots if t_list
            )
            ar_mean_ok = ttft_ok and all(
                (sum(t_list) / float(len(t_list))) <= self.policy.ar_tpot_slo_ms
                for t_list in ar_turn_tpots
                if t_list
            )
            ar_strict_ok = ttft_ok and all(x <= self.policy.ar_tpot_slo_ms for x in ar_tpots)
            ar_mode = str(self.policy.ar_tpot_slo_mode)
            if ar_mode == "turn_mean":
                ar_ok = ar_mean_ok
            elif ar_mode in ("token_max", "strict_max"):
                ar_ok = ar_strict_ok
            else:
                ar_ok = ar_p95_ok
            dit_ok = all(
                lat <= self.policy.dit_turn_slo_for(h, steps=st)
                for lat, h, st in zip(dit_turns, dit_heights, dit_steps_list)
            )
            return _make_outcome(
                "ok",
                ar_ok=ar_ok,
                ar_p95_ok=ar_p95_ok,
                ar_mean_ok=ar_mean_ok,
                ar_strict_ok=ar_strict_ok,
                dit_ok=dit_ok,
            )
        except Exception as exc:
            turn_records.append(
                {
                    "turn_idx": int(cur_turn_idx),
                    "modality_plan": cur_turn_plan,
                    "status": "error",
                    "ar_ttft_ms": None,
                    "ar_tpot_p95_ms": None,
                    "ar_tpot_mean_ms": None,
                    "dit_ms": None,
                    "height": int(cur_turn_height),
                    "kv_bytes": int(kv_bytes),
                    "shipped": False,
                    "morphed": False,
                }
            )
            return _make_outcome("error", error=f"{type(exc).__name__}: {exc}")
        finally:
            if resident_gpu is not None and src_ctx_alive:
                try:
                    await self._rpc(resident_gpu, {"op": "RELEASE_CTX", "ctx_id": ctx_id})
                except Exception:
                    pass
            for p_img in owned_tmp_imgs:
                Path(p_img).unlink(missing_ok=True)

    def _evaluate_outcome_at_k(self, out: SessionOutcome, k: float) -> bool:
        if out.status != "ok":
            return False
        fam = "internvl_u" if self.policy.model_family == "internvl_u" else "bagel"
        slo = DEFAULT_OPERATOR_PROFILE.slo_thresholds_ms(k=k, model_family=fam)
        ttft_cap = slo["ar_ttft_slo_ms"]
        tpot_cap = slo["ar_tpot_slo_ms"]
        if not all(x <= ttft_cap for x in out.ar_ttfts_ms):
            return False
        mode = str(self.policy.ar_tpot_slo_mode)
        turn_lists = out.ar_turn_tpots_ms if out.ar_turn_tpots_ms else ([out.ar_tpots_ms] if out.ar_tpots_ms else [])
        if mode in ("token_max", "strict_max"):
            ar_ok = all(x <= tpot_cap for x in out.ar_tpots_ms)
        elif mode == "turn_mean":
            ar_ok = all((sum(t) / float(len(t))) <= tpot_cap for t in turn_lists if t)
        else:
            ar_ok = all(_percentile(t, 95.0) <= tpot_cap for t in turn_lists if t)
        if not ar_ok:
            return False
        return all(
            lat <= self.policy.dit_turn_slo_for(h, steps=st, k=k)
            for lat, h, st in zip(out.dit_turns_ms, out.dit_turn_heights, out.dit_turn_steps)
        )

    async def _run_suite_async(
        self,
        suite: WorkloadSuite,
        target_qps: float | None,
        burstgpt_deltas: list[float] | None,
        gentd26_deltas: list[float] | None,
        save_outputs_dir: Path | None = None,
    ) -> ClusterRunReport:
        n_gpus = self.policy.num_gpus
        self._gpu_locks = [asyncio.Lock() for _ in range(n_gpus)]
        self._xfer_locks = [asyncio.Lock() for _ in range(n_gpus)]
        self._dst_staging_sems = [asyncio.Semaphore(1) for _ in range(n_gpus)]
        self._inflight_ar_on_gpu = {g: 0 for g in range(n_gpus)}
        self._active_dit_ranks = {g: 0 for g in range(n_gpus)}
        self._active_preemptions = {g: 0 for g in range(n_gpus)}
        self._ar_drained_events = []
        for _ in range(n_gpus):
            ev = asyncio.Event()
            ev.set()
            self._ar_drained_events.append(ev)

        # Reset analytical H2D timeline (`now_rel_ms` restarts at 0.0 per suite) and clean reused workers!
        self.router.pacer.reset_analytical_slots()
        if self.pacer is not None and self.pacer is not self.router.pacer:
            self.pacer.reset_analytical_slots()
        # Reset per-suite `reset_peak_memory_stats()` + NVML peak watermark BEFORE every `run_suite`
        # (including `_suites_completed == 0`) to eliminate pre-suite load/warmup double-counting!
        should_reset_alpha = self._suites_completed > 0
        reset_acks = await asyncio.gather(
            *(self._rpc(g, {"op": "RESET_SUITE", "reset_alpha": should_reset_alpha}) for g in range(n_gpus))
        )
        if should_reset_alpha:
            free_est = 22 * GIB if self.policy.use_tripool_vmm else 10 * GIB
            alphas = {
                g: float(reset_acks[g].get("resident_alpha", self.policy.initial_alphas.get(g, 1.0)))
                for g in range(n_gpus)
            }
            kv_frees = {
                g: int(reset_acks[g].get("kv_free_bytes", free_est))
                for g in range(n_gpus)
            }
            self.worker_states = self.router.init_workers(
                num_gpus=n_gpus,
                role_of=self.policy.role_of,
                initial_alphas=alphas,
                kv_free_bytes=kv_frees,
            )
        self._suites_completed += 1

        if save_outputs_dir is not None:
            save_outputs_dir.mkdir(parents=True, exist_ok=True)
        sessions = [
            InterleavedSession(
                session_id=s.session_id,
                quadrant=s.quadrant,
                source_dataset=s.source_dataset,
                arrival_offset_s=s.arrival_offset_s,
                turns=list(s.turns),
                phase_label=s.phase_label,
            )
            for s in suite.sessions
        ]
        eff_qps = float(target_qps if target_qps is not None else suite.base_qps)
        has_phase_schedule = bool(getattr(suite, "phase_schedule", None))
        if not has_phase_schedule:
            if target_qps is not None and burstgpt_deltas and gentd26_deltas:
                assign_dual_trace_arrivals(sessions, burstgpt_deltas, gentd26_deltas, target_qps=eff_qps)
            elif (
                target_qps is not None
                and eff_qps > 0.0
                and len(sessions) > 1
                and all(float(s.arrival_offset_s) == 0.0 for s in sessions)
            ):
                for idx_s, s_item in enumerate(sessions):
                    s_item.arrival_offset_s = round(idx_s / eff_qps, 4)

        t0 = time.perf_counter()
        outcomes = await asyncio.gather(
            *(self._run_session_coro(s, t0, save_outputs_dir=save_outputs_dir) for s in sessions)
        )
        wall_s = max(time.perf_counter() - t0, 1e-6)

        stat_res = await asyncio.gather(*(self._rpc(g, {"op": "GET_STATS"}) for g in range(n_gpus)))
        L = get_core_lib()
        agg = call_json(
            L,
            L.ms_ctrl_aggregate_report_json,
            encode_json(serialize_policy_config(self.policy)),
            str(suite.name).encode("utf-8"),
            float(eff_qps),
            float(wall_s),
            float(self.host_shmem_delta_gib),
            encode_json([asdict(o) for o in outcomes]),
            encode_json(list(stat_res)),
        )
        return ClusterRunReport(**agg)

    def run_suite(
        self,
        suite: WorkloadSuite,
        target_qps: float | None = None,
        burstgpt_deltas: list[float] | None = None,
        gentd26_deltas: list[float] | None = None,
        save_outputs_dir: Path | None = None,
    ) -> ClusterRunReport:
        if not self._started:
            self.start()
        return asyncio.run(
            self._run_suite_async(
                suite=suite,
                target_qps=target_qps,
                burstgpt_deltas=burstgpt_deltas,
                gentd26_deltas=gentd26_deltas,
                save_outputs_dir=save_outputs_dir,
            )
        )

    def shutdown(self) -> None:
        if not self._started:
            return
        self.fleet.shutdown()
        shutil.rmtree(self.tmp_root, ignore_errors=True)
        self._started = False
