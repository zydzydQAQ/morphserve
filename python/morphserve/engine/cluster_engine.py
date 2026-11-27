"""Model-agnostic async coordinator unifying `WorkerFleet`, C++ `MorphControlEngine`, `DualPathMorphRouter` & Joint SLOs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager, contextmanager
import ctypes
from dataclasses import asdict, dataclass, field, fields
import math
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Callable, Iterator

from morphserve._core import call_json, encode_json, get_core_lib, last_error
from morphserve.router import (
    ClusterTopologyProfile,
    DualPathMorphRouter,
    HalfNodeH2DPacer,
    WorkerState,
    default_operator_profile,
    default_topology_for,
)

from .config import EnginePolicyConfig, RunWindowOptions
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
    """Linear-interpolated percentile matching C++ `percentile_raw` for per-turn summaries."""
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
    return float(
        L.ms_ctrl_deterministic_think_dist_s(
            str(session_id).encode("utf-8"),
            int(turn_idx),
            lo_s,
            hi_s,
            eff_dist.encode("utf-8"),
            eff_median,
            eff_sigma,
        )
    )


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
    ar_turn_has_image: list[bool] = field(default_factory=list)
    ar_turn_num_images: list[int] = field(default_factory=list)
    ar_turn_image_res: list[int] = field(default_factory=list)
    queue_ms: float = 0.0
    h2d_weight_ms: float = 0.0
    kv_transfer_ms: float = 0.0
    vae_ms: float = 0.0
    preempt_yield_ms: float = 0.0


@dataclass
class ClusterRunReport:
    """Aggregated canonical benchmark telemetry emitted by C++ `MorphControlEngine::aggregate_cluster_run_report`."""

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
    steady_joint_slo_goodput_rps: float = 0.0
    steady_joint_slo_attainment_pct: float = 0.0
    full_trace_goodput_rps: float = 0.0
    full_trace_joint_slo_attainment_pct: float = 0.0
    full_trace_attainment_pct: float = 0.0
    peak_nvml_used_mib: float = 0.0
    peak_pytorch_reserved_mib: float = 0.0
    vmm_physical_pool_mib: float = 0.0
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
    kv_host_swap_ins: int = 0
    per_turn_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    time_series_10s: list[dict[str, Any]] = field(default_factory=list)
    actual_arrival_qps: float = 0.0
    actual_steady_arrival_qps: float = 0.0
    steady_sessions_with_ar: int = 0
    steady_sessions_with_dit: int = 0
    nvml_sm_util_mean_pct: float | None = None
    nvml_mem_util_mean_pct: float | None = None
    effective_hbm_util_mean_pct: float | None = None
    ar_ttft_mean_ms: float = 0.0
    ar_tpot_mean_ms: float = 0.0
    dit_turn_mean_ms: float = 0.0
    session_e2e_mean_ms: float = 0.0
    active_jct_mean_ms: float = 0.0
    all_session_active_jct_p99_ms: float | None = None
    all_session_e2e_p99_ms: float | None = None
    all_session_dit_p99_ms: float | None = None
    cost_model_mode: str = "atlas_3d"
    gpu_queue_mode: str = "memory_slack"
    kv_eviction_mode: str = "swap"
    queue_reorders_total: int = 0
    queue_memory_bypasses_total: int = 0
    queue_starvation_promotions_total: int = 0
    kv_recompute_events: int = 0
    kv_recompute_evictions: int = 0
    kv_swap_hit_rate_pct: float = 100.0
    kv_swap_in_mean_ms: float = 0.0
    kv_swap_in_p95_ms: float = 0.0
    kv_recompute_total_ms: float = 0.0
    kv_saved_recompute_ms: float = 0.0
    alpha_timeline: list[dict[str, float]] = field(default_factory=list)
    latency_breakdown_ms: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClusterRunReport":
        """Construct from C++ aggregate JSON while resiliently ignoring unknown forward-compatible keys."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class MemorySlackGpuLock:
    """Async per-GPU exclusive lock delegating waiter selection to C++ `GpuWaitQueueScheduler`."""

    def __init__(
        self,
        gpu_id: int,
        *,
        mode: str = "memory_slack",
        max_wait_aging_ms: float = 45000.0,
        max_bypasses: int = 5,
        free_mem_provider: Callable[[int], int] | None = None,
    ) -> None:
        self.gpu_id = int(gpu_id)
        self._free_mem_provider = free_mem_provider
        self._lib = get_core_lib()
        pol_j = {
            "mode": str(mode),
            "max_wait_aging_ms": float(max_wait_aging_ms),
            "max_bypasses": int(max_bypasses),
        }
        h = int(self._lib.ms_gpu_queue_create(encode_json(pol_j)) or 0)
        if h == 0:
            raise RuntimeError(f"ms_gpu_queue_create failed: {last_error(self._lib)}")
        self._handle = h
        self._locked = False
        self._seq = 0
        self._waiters: list[tuple[dict[str, Any], asyncio.Future[None]]] = []

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        h = getattr(self, "_handle", 0)
        if h:
            self._handle = 0
            try:
                self._lib.ms_gpu_queue_destroy(ctypes.c_void_p(h))
            except Exception:
                pass

    def locked(self) -> bool:
        return self._locked

    def has_waiters(self) -> bool:
        return any(not f.done() for _, f in self._waiters)

    def stats(self) -> dict[str, int]:
        if not self._handle:
            return {
                "dispatches_total": 0,
                "reorders_total": 0,
                "memory_bypasses_total": 0,
                "starvation_promotions_total": 0,
            }
        raw = call_json(self._lib, self._lib.ms_gpu_queue_stats_json, ctypes.c_void_p(self._handle))
        return {k: int(v) for k, v in raw.items()}

    async def acquire(
        self,
        *,
        est_mem_bytes: int = 0,
        est_exec_ms: float = 0.0,
        slo_deadline_ms: float = 0.0,
        op_priority: int = 1,
    ) -> bool:
        if not self._locked and not self._waiters:
            self._locked = True
            return True
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._seq += 1
        spec = {
            "waiter_id": int(self._seq),
            "enqueue_ms": float(time.perf_counter() * 1000.0),
            "est_mem_bytes": int(max(0, est_mem_bytes)),
            "est_exec_ms": float(max(0.0, est_exec_ms)),
            "slo_deadline_ms": float(max(0.0, slo_deadline_ms)),
            "op_priority": int(op_priority),
            "bypass_count": 0,
        }
        entry = (spec, fut)
        self._waiters.append(entry)
        try:
            await fut
        except BaseException:
            if not fut.done():
                try:
                    self._waiters.remove(entry)
                except ValueError:
                    pass
            elif not fut.cancelled():
                self.release()
            raise
        return True

    def release(self) -> None:
        if not self._locked:
            raise RuntimeError("MemorySlackGpuLock.release() called on unlocked lock")
        # Purge already-cancelled waiters before querying C++ selector.
        self._waiters = [(s, f) for (s, f) in self._waiters if not f.done()]
        if not self._waiters:
            self._locked = False
            return
        live_free = 0
        if self._free_mem_provider is not None:
            try:
                live_free = max(0, int(self._free_mem_provider(self.gpu_id)))
            except Exception:
                live_free = 0
        req = {
            "waiters": [s for (s, _) in self._waiters],
            "now_ms": float(time.perf_counter() * 1000.0),
            "live_free_mem_bytes": int(live_free),
        }
        resp = call_json(
            self._lib,
            self._lib.ms_gpu_queue_select_json,
            ctypes.c_void_p(self._handle),
            encode_json(req),
        )
        sel = resp.get("selected")
        rem_by_id = {int(w["waiter_id"]): w for w in resp.get("remaining_waiters", [])}
        chosen_wid = int(sel["waiter_id"]) if isinstance(sel, Mapping) else int(self._waiters[0][0]["waiter_id"])
        chosen_fut: asyncio.Future[None] | None = None
        next_waiters: list[tuple[dict[str, Any], asyncio.Future[None]]] = []
        for s, f in self._waiters:
            wid = int(s["waiter_id"])
            if wid == chosen_wid and chosen_fut is None:
                chosen_fut = f
            else:
                if wid in rem_by_id:
                    s["bypass_count"] = int(rem_by_id[wid].get("bypass_count", s.get("bypass_count", 0)))
                next_waiters.append((s, f))
        self._waiters = next_waiters
        if chosen_fut is not None and not chosen_fut.done():
            chosen_fut.set_result(None)
        else:
            self._locked = False

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


class MorphServeClusterEngine:
    """Central async coordinator driving `WorkerFleet` + C++17 `MorphControlEngine` + `DualPathMorphRouter`."""

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
        self.topo = topology or default_topology_for(
            int(policy.num_gpus), gpus_per_node=min(8, max(1, int(policy.num_gpus)))
        )
        self.pacer = HalfNodeH2DPacer(self.topo, max_concurrent_per_half=1) if policy.use_h2d_pacer else None
        self.router = DualPathMorphRouter(
            self.topo,
            pacer=self.pacer,
            cost_model_mode=self.policy.cost_model_mode,
        )
        self.fleet: WorkerFleet = fleet or MpSpawnGpuWorkerFleet(
            policy, self.model_dir, self.bench_root, topology=self.topo
        )
        self._lib = get_core_lib()
        self._ctrl_handle = self._create_ctrl_handle(self.policy)

        self._pending_futures: dict[str, tuple[int, asyncio.AbstractEventLoop, asyncio.Future]] = {}
        self._faulted_ranks: dict[int, str] = {}
        self._fut_lock = threading.Lock()
        self._cmd_seq = 0

        self._gpu_locks: list[MemorySlackGpuLock] = []
        self._xfer_locks: list[asyncio.Lock] = []
        self._dst_staging_sems: list[asyncio.Semaphore] = []
        self._inflight_ar_on_gpu: dict[int, int] = {}
        self._ar_drained_events: list[asyncio.Event] = []
        self._active_dit_ranks: dict[int, int] = {}
        self._active_preemptions: dict[int, int] = {}
        self._suites_completed = 0
        self._started = False
        self.host_shmem_delta_gib = 0.0
        self._image_res_cache: dict[str, int] = {}
        self._pending_bg_releases: dict[tuple[int, str], asyncio.Task[Any]] = {}

    def _probe_image_max_edge(self, rel_path: str, fallback: int = 512) -> int:
        key = str(rel_path)
        cached = self._image_res_cache.get(key)
        if cached is not None:
            return cached
        p = Path(key) if os.path.isabs(key) else (self.bench_root / key)
        res = int(fallback)
        if p.is_file():
            try:
                from PIL import Image

                with Image.open(p) as im:
                    w, h = im.size
                    res = max(int(w), int(h))
            except Exception:
                res = int(fallback)
        if len(self._image_res_cache) < 1024:
            self._image_res_cache[key] = res
        return res

    def __del__(self) -> None:
        h = getattr(self, "_ctrl_handle", 0)
        if h:
            self._ctrl_handle = 0
            try:
                self._lib.ms_ctrl_destroy(ctypes.c_void_p(h))
            except Exception:
                pass

    def _create_ctrl_handle(self, policy: EnginePolicyConfig) -> int:
        h = int(
            self._lib.ms_ctrl_create(
                encode_json(policy.to_dict()),
                encode_json(self.topo.to_dict()),
                ctypes.c_void_p(self.router._handle),
            )
            or 0
        )
        if h == 0:
            raise RuntimeError(f"ms_ctrl_create failed: {last_error(self._lib)}")
        return h

    @contextmanager
    def _scoped_ctrl_handle(self, eff_policy: EnginePolicyConfig) -> Iterator[int]:
        if eff_policy == self.policy:
            yield self._ctrl_handle
            return
        h = self._create_ctrl_handle(eff_policy)
        try:
            yield h
        finally:
            self._lib.ms_ctrl_destroy(ctypes.c_void_p(h))

    @property
    def worker_states(self) -> dict[int, WorkerState]:
        """Authoritative worker state snapshot owned by C++ `DualPathMorphRouter`."""
        return self.router.worker_states_snapshot()

    def _next_cmd_id(self) -> str:
        self._cmd_seq += 1
        return f"cmd_{self._cmd_seq}"

    def _handle_worker_fault(self, ranks: list[int], reason: str) -> None:
        dead = {int(r) for r in ranks}
        if not dead:
            return
        err_payload = {"status": "error", "error": str(reason)}
        victims: list[tuple[asyncio.AbstractEventLoop, asyncio.Future]] = []
        with self._fut_lock:
            for r in dead:
                self._faulted_ranks[r] = str(reason)
            for k, (rk, lp, ft) in list(self._pending_futures.items()):
                if rk in dead:
                    self._pending_futures.pop(k, None)
                    victims.append((lp, ft))
        for lp, ft in victims:
            if not lp.is_closed():
                lp.call_soon_threadsafe(lambda f=ft, m=err_payload: None if f.done() else f.set_result(m))

    def _handle_worker_ack(self, msg: dict[str, Any]) -> None:
        cid = msg.get("cmd_id")
        if cid is None:
            return
        with self._fut_lock:
            entry = self._pending_futures.pop(str(cid), None)
        if entry is not None:
            _, loop, fut = entry
            if not loop.is_closed():
                loop.call_soon_threadsafe(lambda f=fut, m=msg: None if f.done() else f.set_result(m))

    def _sync_worker_telemetry(self, rank: int, msg: Mapping[str, Any]) -> None:
        alpha = float(msg["resident_alpha"]) if "resident_alpha" in msg else None
        kv_free = int(msg["kv_free_bytes"]) if "kv_free_bytes" in msg else None
        t_seq = int(msg["telemetry_seq"]) if "telemetry_seq" in msg else None
        if alpha is not None or kv_free is not None:
            self.router.update_worker_telemetry(
                int(rank),
                expert_alpha=alpha,
                kv_free_bytes=kv_free,
                telemetry_seq=t_seq,
            )

    def _init_router_workers_from_acks(self, workers_info: Mapping[Any, Mapping[str, Any]]) -> None:
        n = int(self.policy.num_gpus)
        free_est = 22 * GIB if self.policy.use_tripool_vmm else 10 * GIB
        alphas: dict[int, float] = {}
        kv_frees: dict[int, int] = {}
        for rank in range(n):
            info = workers_info.get(rank) or workers_info.get(str(rank)) or {}
            alphas[rank] = float(info.get("resident_alpha", self.policy.initial_alphas.get(rank, 1.0)))
            kv_frees[rank] = int(info.get("kv_free_bytes", free_est))
        self.router.init_workers(
            num_gpus=n,
            role_of=self.policy.role_of,
            initial_alphas=alphas,
            kv_free_bytes=kv_frees,
        )

    def _free_mem_bytes_for_gpu(self, gpu: int) -> int:
        try:
            return max(0, int(self.router.get_worker_state(gpu).kv_free_bytes))
        except Exception:
            return 0

    def _build_gpu_locks(self, eff_policy: EnginePolicyConfig) -> list[MemorySlackGpuLock]:
        for lk in self._gpu_locks:
            if hasattr(lk, "close"):
                lk.close()
        return [
            MemorySlackGpuLock(
                g,
                mode=eff_policy.gpu_queue_mode,
                max_wait_aging_ms=eff_policy.gpu_queue_max_aging_ms,
                max_bypasses=eff_policy.gpu_queue_max_bypasses,
                free_mem_provider=self._free_mem_bytes_for_gpu,
            )
            for g in range(eff_policy.num_gpus)
        ]

    def _ensure_async_primitives(self) -> None:
        n_gpus = self.policy.num_gpus
        if len(self._gpu_locks) != n_gpus:
            self._gpu_locks = self._build_gpu_locks(self.policy)
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

    def _schedule_bg_release_ctx(self, rank: int, ctx_id: str) -> None:
        key = (int(rank), str(ctx_id))
        prev = self._pending_bg_releases.get(key)

        async def _do_release() -> None:
            if prev is not None and not prev.done():
                try:
                    await prev
                except Exception:
                    pass
            try:
                await self._rpc(int(rank), {"op": "RELEASE_CTX", "ctx_id": str(ctx_id)})
            except Exception:
                pass
            finally:
                if self._pending_bg_releases.get(key) is task:
                    self._pending_bg_releases.pop(key, None)

        task = asyncio.create_task(_do_release())
        self._pending_bg_releases[key] = task

    async def _await_bg_release_ctx(self, rank: int, ctx_id: str) -> None:
        task = self._pending_bg_releases.get((int(rank), str(ctx_id)))
        if task is not None and not task.done():
            try:
                await task
            except Exception:
                pass

    async def _drain_bg_releases(self) -> None:
        while self._pending_bg_releases:
            tasks = list(self._pending_bg_releases.values())
            await asyncio.gather(*tasks, return_exceptions=True)

    def start(self) -> dict[str, Any]:
        if self._started:
            return {}
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        with self._fut_lock:
            self._faulted_ranks.clear()
        init_summary = self.fleet.start(
            on_ack=self._handle_worker_ack,
            on_worker_fault=self._handle_worker_fault,
        )
        self.host_shmem_delta_gib = float(init_summary.get("host_shmem_delta_gib", 0.0))

        n = self.policy.num_gpus
        self._init_router_workers_from_acks(init_summary.get("workers") or {})
        self._active_dit_ranks = {g: 0 for g in range(n)}
        self._active_preemptions = {g: 0 for g in range(n)}
        self._inflight_ar_on_gpu = {g: 0 for g in range(n)}
        self._started = True
        return init_summary

    async def _rpc(self, rank: int, payload: dict[str, Any], timeout_s: float = 600.0) -> dict[str, Any]:
        r = int(rank)
        with self._fut_lock:
            fault_reason = self._faulted_ranks.get(r)
        if fault_reason is not None:
            raise RuntimeError(f"worker rank {r} faulted prior to {payload.get('op')}: {fault_reason}")

        loop = asyncio.get_running_loop()
        cid = payload.get("cmd_id") or self._next_cmd_id()
        payload = dict(payload, cmd_id=cid)
        fut: asyncio.Future = loop.create_future()
        with self._fut_lock:
            fault_reason = self._faulted_ranks.get(r)
            if fault_reason is not None:
                raise RuntimeError(f"worker rank {r} faulted prior to {payload.get('op')}: {fault_reason}")
            self._pending_futures[cid] = (r, loop, fut)
        self.fleet.send(r, payload)
        try:
            res = await asyncio.wait_for(fut, timeout=timeout_s)
        except Exception:
            with self._fut_lock:
                self._pending_futures.pop(cid, None)
            raise
        self._sync_worker_telemetry(r, res)
        if res.get("status") == "error":
            err_txt = str(res.get("error", ""))
            if "cuda" in err_txt.lower():
                self.router.update_worker_telemetry(r, expert_alpha=0.0, kv_free_bytes=0)
            raise RuntimeError(
                f"worker rank {r} error on {payload.get('op')}: {err_txt}\n{res.get('traceback', '')}"
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

    @staticmethod
    def _supports_mid_dit_preemption(eff_policy: EnginePolicyConfig) -> bool:
        """True when workers can preempt active DiT steps without thrashing co-paged VLM/Decoder banks."""
        if eff_policy.preemption_mode == "non_preemptive":
            return False
        if eff_policy.cptg_architecture == "bridge_mllm_dit":
            raw_quota = getattr(eff_policy, "gpu_memory_quota_mib", None)
            raw_frac = getattr(eff_policy, "gpu_memory_fraction", None)
            if (raw_quota is not None and float(raw_quota) > 0.0) or (
                raw_frac is not None and float(raw_frac) < 0.999
            ):
                return False
        return True

    def _pick_initial_gpu(self, eff_policy: EnginePolicyConfig, modality_plan: str) -> int:
        eff_preempt = (
            eff_policy.preemption_mode if self._supports_mid_dit_preemption(eff_policy) else "non_preemptive"
        )
        return self.router.select_initial_gpu(
            num_gpus=eff_policy.num_gpus,
            routing_mode=eff_policy.routing_mode,
            preemption_mode=eff_preempt,
            static_und_ranks=eff_policy.static_und_ranks,
            modality_plan=modality_plan,
            gpu_busy=self._gpu_busy_map(),
            active_dit_ranks=self._active_dit_ranks,
            active_preemptions=self._active_preemptions,
            max_preemptions_per_gpu=4,
            architecture=eff_policy.cptg_architecture,
        )

    def _pick_oom_retry_gpu(self, eff_policy: EnginePolicyConfig, failed_gpu: int) -> int:
        return self.router.select_oom_retry_gpu(
            num_gpus=eff_policy.num_gpus,
            routing_mode=eff_policy.routing_mode,
            static_und_ranks=eff_policy.static_und_ranks,
            failed_gpu=failed_gpu,
            gpu_busy=self._gpu_busy_map(),
        )

    def _plan_turn_transition(
        self,
        ctrl_handle: int,
        *,
        session_id: str,
        base_ctx_id: str,
        turn_idx: int,
        attempt: int,
        src_gpu: int,
        state_bytes: int,
        modality_plan: str,
        height: int,
        dit_steps: int,
        has_more_turns: bool,
        has_more_edits: bool,
        now_ms: float,
        src_lock_held: bool = False,
    ) -> dict[str, Any]:
        req = {
            "session_id": str(session_id),
            "base_ctx_id": str(base_ctx_id),
            "turn_idx": int(turn_idx),
            "attempt": int(attempt),
            "src_gpu": int(src_gpu),
            "state_bytes": int(state_bytes),
            "modality_plan": str(modality_plan),
            "height": int(height),
            "dit_steps": int(dit_steps),
            "has_more_turns": bool(has_more_turns),
            "has_more_edits": bool(has_more_edits),
            "now_ms": float(now_ms),
            "gpu_busy": {str(k): bool(v) for k, v in self._gpu_busy_map().items()},
            "src_lock_held": bool(src_lock_held),
        }
        return call_json(
            self._lib,
            self._lib.ms_ctrl_plan_turn_transition_json,
            ctypes.c_void_p(ctrl_handle),
            encode_json(req),
        )

    def _evaluate_session_slos(
        self,
        eff_policy: EnginePolicyConfig,
        *,
        ar_ttfts_ms: list[float],
        ar_tpots_ms: list[float],
        ar_turn_tpots_ms: list[list[float]],
        dit_turns_ms: list[float],
        dit_turn_heights: list[int],
        dit_turn_steps: list[int],
        ar_turn_has_image: list[bool] | None = None,
        ar_turn_num_images: list[int] | None = None,
        ar_turn_image_res: list[int] | None = None,
        turn_records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        req: dict[str, Any] = {
            "ar_ttfts_ms": [float(x) for x in ar_ttfts_ms],
            "ar_tpots_ms": [float(x) for x in ar_tpots_ms],
            "ar_turn_tpots_ms": [[float(v) for v in row] for row in ar_turn_tpots_ms],
            "dit_turns_ms": [float(x) for x in dit_turns_ms],
            "dit_heights": [int(h) for h in dit_turn_heights],
            "dit_turn_heights": [int(h) for h in dit_turn_heights],
            "dit_steps": [int(st) for st in dit_turn_steps],
            "dit_turn_steps": [int(st) for st in dit_turn_steps],
            "slo_time_scale": 1.0,
        }
        if ar_turn_has_image is not None:
            req["ar_turn_has_image"] = [bool(x) for x in ar_turn_has_image]
        if ar_turn_num_images is not None:
            req["ar_turn_num_images"] = [int(x) for x in ar_turn_num_images]
        if ar_turn_image_res is not None:
            req["ar_turn_image_res"] = [int(x) for x in ar_turn_image_res]
        if turn_records is not None:
            req["turn_records"] = turn_records
        return call_json(
            self._lib,
            self._lib.ms_ctrl_evaluate_session_slos_json,
            encode_json(eff_policy.to_dict()),
            encode_json(req),
        )

    async def _execute_ar_phase(
        self,
        eff_policy: EnginePolicyConfig,
        resident_gpu: int,
        ctx_id: str,
        turn: InterleavedTurn,
        prepare_for_dit: bool,
        use_prev_image: bool = False,
        prev_image_path: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Dispatch AR via mid-DiT preemption OR continuous-batched `EXEC_AR` without single-AR lock serialization."""
        can_preempt = (
            self._supports_mid_dit_preemption(eff_policy)
            and (self._gpu_locks[resident_gpu].locked() or self._active_dit_ranks.get(resident_gpu, 0) > 0)
            and self._active_preemptions.get(resident_gpu, 0) < 4
        )
        if can_preempt:
            self._active_preemptions[resident_gpu] += 1
            self.router.set_colocated_ar_count(resident_gpu, self._active_preemptions[resident_gpu])
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
                        "tpot_slo_ms": eff_policy.ar_tpot_slo_ms,
                        "prepare_for_dit": prepare_for_dit,
                    },
                )
            finally:
                self._active_preemptions[resident_gpu] = max(0, self._active_preemptions[resident_gpu] - 1)
                self.router.set_colocated_ar_count(resident_gpu, self._active_preemptions[resident_gpu])
            was_mid_dit = bool(ar_res.get("preempted_mid_dit", True)) and ar_res.get("status") == "ok"
            if ar_res.get("status") == "ok":
                ar_res = dict(ar_res, queue_ms=round(float(ar_res.get("queue_ms", 0.0)), 2))
            return ar_res, was_mid_dit

        t_enq = time.perf_counter()
        est_ar_ms = default_operator_profile().ar_tpot_ms * max(1, min(turn.max_ar_tokens, 16))
        self.router.reserve_ar_queue(resident_gpu, est_ar_ms)
        try:
            # Wait only if an exclusive DiT / NCCL operation holds `_gpu_locks`; multiple AR turns
            # register atomically before releasing the direct-handoff `MemorySlackGpuLock` so waiting
            # ARs never livelock and batch concurrently into `WorkerRuntime.decode_ar_batch`.
            ar_has_img = bool(turn.image_paths or use_prev_image)
            ar_num_img = len(turn.image_paths) + (1 if use_prev_image else 0)
            ar_dl_ms = t_enq * 1000.0 + eff_policy.ar_ttft_slo_for(
                has_image=ar_has_img,
                num_images=ar_num_img,
                image_res=max(turn.height, turn.width),
            )
            if self._gpu_locks[resident_gpu].locked():
                await self._gpu_locks[resident_gpu].acquire(
                    est_mem_bytes=0,
                    est_exec_ms=est_ar_ms,
                    slo_deadline_ms=ar_dl_ms,
                    op_priority=0,
                )
                try:
                    coord_queue_wait_ms = max(0.0, (time.perf_counter() - t_enq) * 1000.0)
                    self._inflight_ar_on_gpu[resident_gpu] = self._inflight_ar_on_gpu.get(resident_gpu, 0) + 1
                    self._ar_drained_events[resident_gpu].clear()
                finally:
                    self._gpu_locks[resident_gpu].release()
            else:
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
            if ar_res.get("status") == "ok":
                worker_q_ms = float(ar_res.get("queue_ms", 0.0))
                ar_res = dict(
                    ar_res,
                    ttft_ms=round(float(ar_res.get("ttft_ms", 0.0)) + coord_queue_wait_ms, 2),
                    queue_ms=round(coord_queue_wait_ms + worker_q_ms, 2),
                )
            return ar_res, False
        finally:
            self.router.release_ar_queue(resident_gpu, est_ar_ms)

    async def _acquire_exclusive_gpu(
        self,
        gpu: int,
        *,
        est_mem_bytes: int = 0,
        est_exec_ms: float = 0.0,
        slo_deadline_ms: float = 0.0,
        op_priority: int = 1,
    ) -> None:
        """Acquire exclusive GPU lock via C++ Memory+Slack scheduler after standalone AR drains."""
        await self._gpu_locks[gpu].acquire(
            est_mem_bytes=est_mem_bytes,
            est_exec_ms=est_exec_ms,
            slo_deadline_ms=slo_deadline_ms,
            op_priority=op_priority,
        )
        try:
            await self._ar_drained_events[gpu].wait()
        except BaseException:
            self._gpu_locks[gpu].release()
            raise

    @asynccontextmanager
    async def _nccl_transfer_guard(
        self,
        src_gpu: int,
        dst_gpu: int,
        ctx_id: str,
        new_ctx_id: str,
        include_cfg_dup: bool,
        *,
        release_src_ctx: bool = True,
        defer_src_release: bool = False,
        src_lock_held: bool = False,
        dst_est_mem_bytes: int = 0,
        dst_est_exec_ms: float = 0.0,
        dst_slo_deadline_ms: float = 0.0,
        dst_op_priority: int = 1,
    ):
        """Async context manager executing decoupled NCCL pack->send->recv->unpack while owning lock/semaphore lifetime.

        When `src_lock_held=True`, ownership of the caller's held `_gpu_locks[src_gpu]` transfers here and is
        guaranteed released immediately after `PACK_CTX` completes (zero extra `src_gpu` queue wait).
        When `defer_src_release=True` (used for `nccl_dual`), `PACK_CTX` leaves `ctx_id` intact on `src_gpu`
        until `UNPACK_CTX` succeeds on `dst_gpu`, so any staging OOM preserves `src_gpu` KV for seamless
        in-place Path-B fallback instead of killing the session.
        When `ship["status"] == "ok"`, `_gpu_locks[dst_gpu]` remains exclusively held for the body so callers
        may run `EXEC_DIT` atomically right after `UNPACK_CTX` without releasing/reacquiring the GPU lock.
        All staging semaphores, destination locks, and temporary buffers are guaranteed released on exit.
        """
        src_queue_ms = 0.0
        t_pack0 = 0.0
        pack_release_flag = bool(release_src_ctx and not defer_src_release)
        if src_lock_held:
            try:
                self._ensure_async_primitives()
                t_pack0 = time.perf_counter()
                pack_res = await self._rpc(
                    src_gpu,
                    {
                        "op": "PACK_CTX",
                        "ctx_id": ctx_id,
                        "include_cfg_dup": include_cfg_dup,
                        "release_after_pack": pack_release_flag,
                    },
                )
                pack_ms = max(0.0, (time.perf_counter() - t_pack0) * 1000.0)
            finally:
                self._gpu_locks[src_gpu].release()
        else:
            self._ensure_async_primitives()
            t_src_q0 = time.perf_counter()
            await self._acquire_exclusive_gpu(
                src_gpu,
                est_mem_bytes=0,
                est_exec_ms=default_operator_profile().prep_overhead_ms,
                slo_deadline_ms=dst_slo_deadline_ms,
                op_priority=0,
            )
            src_queue_ms = max(0.0, (time.perf_counter() - t_src_q0) * 1000.0)
            t_pack0 = time.perf_counter()
            try:
                pack_res = await self._rpc(
                    src_gpu,
                    {
                        "op": "PACK_CTX",
                        "ctx_id": ctx_id,
                        "include_cfg_dup": include_cfg_dup,
                        "release_after_pack": pack_release_flag,
                    },
                )
                pack_ms = max(0.0, (time.perf_counter() - t_pack0) * 1000.0)
            finally:
                self._gpu_locks[src_gpu].release()

        if pack_res.get("status") != "ok":
            yield dict(
                pack_res,
                packed_ok=False,
                src_ctx_preserved=True,
                src_ctx_released=False,
            )
            return

        packed_id = str(pack_res["packed_id"])
        meta = pack_res["meta"]
        wire_bytes = int(meta["total_bytes"])
        logical_bytes = int(meta.get("logical_bytes", wire_bytes))

        t_dst_q_acc = 0.0
        staging_sem = self._dst_staging_sems[dst_gpu]
        t_sem0 = time.perf_counter()
        await staging_sem.acquire()
        t_dst_q_acc += max(0.0, (time.perf_counter() - t_sem0) * 1000.0)
        holding_sem = True
        holding_dst_lock = False
        recv_id: str | None = None
        packed_consumed = False
        try:
            alloc_res = await self._rpc(dst_gpu, {"op": "ALLOC_RECV_BUF", "nbytes": wire_bytes, "probe_only": True})
            if alloc_res.get("status") != "ok":
                if not holding_dst_lock:
                    t_dq0 = time.perf_counter()
                    await self._acquire_exclusive_gpu(
                        dst_gpu,
                        est_mem_bytes=dst_est_mem_bytes,
                        est_exec_ms=dst_est_exec_ms,
                        slo_deadline_ms=dst_slo_deadline_ms,
                        op_priority=dst_op_priority,
                    )
                    t_dst_q_acc += max(0.0, (time.perf_counter() - t_dq0) * 1000.0)
                    holding_dst_lock = True
                # Exclusively own `dst_gpu` now: reclaim idle KV + trim VMM overflow + defragment
                # PyTorch caching allocator so the contiguous NCCL recv buffer succeeds without OOM.
                await self._rpc(
                    dst_gpu,
                    {"op": "RECLAIM_HEADROOM", "needed_bytes": int(wire_bytes) + 512 * MIB},
                )
                alloc_res = await self._rpc(
                    dst_gpu,
                    {"op": "ALLOC_RECV_BUF", "nbytes": wire_bytes, "probe_only": False},
                )
                if alloc_res.get("status") != "ok":
                    yield {
                        "status": "oom",
                        "packed_ok": True,
                        "src_ctx_preserved": bool(not pack_release_flag),
                        "src_ctx_released": bool(pack_release_flag),
                        "error": alloc_res.get("error", "alloc_recv_oom"),
                    }
                    return

            recv_id = str(alloc_res["recv_id"])
            lo, hi = (src_gpu, dst_gpu) if src_gpu < dst_gpu else (dst_gpu, src_gpu)
            t_xf0 = time.perf_counter()
            async with self._xfer_locks[lo]:
                async with self._xfer_locks[hi]:
                    s_res, r_res = await asyncio.gather(
                        self._rpc(src_gpu, {"op": "SEND_NCCL", "packed_id": packed_id, "dst_rank": dst_gpu}),
                        self._rpc(dst_gpu, {"op": "RECV_NCCL", "src_rank": src_gpu, "recv_id": recv_id, "meta": meta}),
                    )
                    packed_consumed = True
            xfer_wall_ms = max(0.0, (time.perf_counter() - t_xf0) * 1000.0)

            if s_res.get("status") != "ok" or r_res.get("status") != "ok":
                yield {
                    "status": "oom",
                    "packed_ok": True,
                    "src_ctx_preserved": bool(not pack_release_flag),
                    "src_ctx_released": bool(pack_release_flag),
                    "error": "nccl_xfer_oom",
                }
                return

            ship_ms = max(float(s_res.get("ship_ms", 0.0)), float(r_res.get("ship_ms", 0.0)))
            if not holding_dst_lock:
                t_dq1 = time.perf_counter()
                await self._acquire_exclusive_gpu(
                    dst_gpu,
                    est_mem_bytes=dst_est_mem_bytes,
                    est_exec_ms=dst_est_exec_ms,
                    slo_deadline_ms=dst_slo_deadline_ms,
                    op_priority=dst_op_priority,
                )
                t_dst_q_acc += max(0.0, (time.perf_counter() - t_dq1) * 1000.0)
                holding_dst_lock = True

            await self._await_bg_release_ctx(dst_gpu, new_ctx_id)
            t_unp0 = time.perf_counter()
            u_res = await self._rpc(
                dst_gpu,
                {"op": "UNPACK_CTX", "recv_id": recv_id, "new_ctx_id": new_ctx_id, "meta": meta},
            )
            unpack_ms = max(0.0, (time.perf_counter() - t_unp0) * 1000.0)
            recv_id = None
            staging_sem.release()
            holding_sem = False

            if u_res.get("status") != "ok":
                yield {
                    "status": "oom",
                    "packed_ok": True,
                    "src_ctx_preserved": bool(not pack_release_flag),
                    "src_ctx_released": bool(pack_release_flag),
                    "error": u_res.get("error", "unpack_oom"),
                    "kv_free_bytes": u_res.get("kv_free_bytes"),
                    "telemetry_seq": u_res.get("telemetry_seq"),
                }
                return

            if release_src_ctx and defer_src_release:
                self._schedule_bg_release_ctx(src_gpu, ctx_id)

            yield {
                "status": "ok",
                "packed_ok": True,
                "src_ctx_preserved": False,
                "src_ctx_released": bool(release_src_ctx),
                "ship_ms": ship_ms,
                "kv_transfer_ms": round(pack_ms + max(ship_ms, xfer_wall_ms) + unpack_ms, 2),
                "queue_ms": round(src_queue_ms + t_dst_q_acc, 2),
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
        src_lock_held: bool = False,
        dst_est_mem_bytes: int = 0,
        dst_est_exec_ms: float = 0.0,
        dst_slo_deadline_ms: float = 0.0,
        dst_op_priority: int = 1,
    ) -> dict[str, Any]:
        src_q_ms = 0.0
        if src_lock_held:
            try:
                if not self.topo.same_node(src_gpu, dst_gpu):
                    raise ValueError(
                        f"SHM state shipping requires same-node GPUs, got cross-node transfer {src_gpu}->{dst_gpu} "
                        f"(node {self.topo.node_of(src_gpu)} -> node {self.topo.node_of(dst_gpu)})"
                    )
                self._ensure_async_primitives()
                shm_name = f"ms_shm_{os.getpid()}_{src_gpu}_{dst_gpu}_{self._next_cmd_id()}"
                t_s0 = time.perf_counter()
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
                send_ms = max(0.0, (time.perf_counter() - t_s0) * 1000.0)
                if s_res.get("status") != "ok":
                    return dict(s_res, sent_ok=False)
            finally:
                self._gpu_locks[src_gpu].release()
        else:
            if not self.topo.same_node(src_gpu, dst_gpu):
                raise ValueError(
                    f"SHM state shipping requires same-node GPUs, got cross-node transfer {src_gpu}->{dst_gpu} "
                    f"(node {self.topo.node_of(src_gpu)} -> node {self.topo.node_of(dst_gpu)})"
                )
            self._ensure_async_primitives()
            shm_name = f"ms_shm_{os.getpid()}_{src_gpu}_{dst_gpu}_{self._next_cmd_id()}"
            t_sq0 = time.perf_counter()
            await self._acquire_exclusive_gpu(
                src_gpu,
                est_mem_bytes=0,
                est_exec_ms=default_operator_profile().prep_overhead_ms,
                slo_deadline_ms=dst_slo_deadline_ms,
                op_priority=0,
            )
            src_q_ms = max(0.0, (time.perf_counter() - t_sq0) * 1000.0)
            t_s0 = time.perf_counter()
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
                send_ms = max(0.0, (time.perf_counter() - t_s0) * 1000.0)
                if s_res.get("status") != "ok":
                    return dict(s_res, sent_ok=False)
            finally:
                self._gpu_locks[src_gpu].release()

        t_dq0 = time.perf_counter()
        await self._acquire_exclusive_gpu(
            dst_gpu,
            est_mem_bytes=dst_est_mem_bytes,
            est_exec_ms=dst_est_exec_ms,
            slo_deadline_ms=dst_slo_deadline_ms,
            op_priority=dst_op_priority,
        )
        dst_q_ms = max(0.0, (time.perf_counter() - t_dq0) * 1000.0)
        t_r0 = time.perf_counter()
        try:
            r_res = await self._rpc(
                dst_gpu,
                {"op": "RECV_SHM", "shm_name": shm_name, "new_ctx_id": new_ctx_id, "meta": s_res["meta"]},
            )
            recv_ms = max(0.0, (time.perf_counter() - t_r0) * 1000.0)
            r_res["bytes"] = int(s_res.get("bytes", 0))
            r_res["sent_ok"] = True
            r_res["kv_transfer_ms"] = round(send_ms + recv_ms, 2)
            r_res["queue_ms"] = round(src_q_ms + dst_q_ms, 2)
            return r_res
        finally:
            self._gpu_locks[dst_gpu].release()

    async def _exec_dit_on_locked_gpu(
        self,
        dst_gpu: int,
        dit_ctx_id: str,
        turn: InterleavedTurn,
        turn_idx: int,
        *,
        save_tmp: str | None,
        ephemeral_gen_ctx: bool,
        needs_return_ship: bool,
    ) -> dict[str, Any]:
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
                    "seed": 52 + turn_idx,
                    "save_tmp_image_path": save_tmp,
                    "release_after_dit": ephemeral_gen_ctx and (not needs_return_ship),
                },
            )
            if ephemeral_gen_ctx and dit_res.get("status") != "ok":
                try:
                    await self._rpc(dst_gpu, {"op": "RELEASE_CTX", "ctx_id": dit_ctx_id})
                except Exception:
                    pass
            return dit_res
        finally:
            self._active_dit_ranks[dst_gpu] -= 1

    async def _retry_or_fail_oom(
        self,
        *,
        eff_policy: EnginePolicyConfig,
        sess_id: str,
        turn_idx: int,
        turn: InterleavedTurn,
        attempt: int,
        oom_retries: int,
        failed_gpu: int,
        cleanup_targets: set[tuple[int | None, str]],
        kv_bytes: int,
        turn_records: list[dict[str, Any]],
        ar_ttft_ms: float | None = None,
        ar_tpots_ms: list[float] | None = None,
        shipped: bool = False,
        morphed: bool = False,
        has_image: bool = False,
        num_images: int = 0,
        image_res: int = 0,
        queue_ms: float = 0.0,
        h2d_weight_ms: float = 0.0,
        kv_transfer_ms: float = 0.0,
        vae_ms: float = 0.0,
        preempt_yield_ms: float = 0.0,
    ) -> tuple[bool, int, int, int, str]:
        """Release partial context(s), decide bounded Turn-0 OOM retry vs terminal OOM turn record."""
        for g_clean, c_clean in cleanup_targets:
            if g_clean is not None:
                try:
                    await self._rpc(g_clean, {"op": "RELEASE_CTX", "ctx_id": c_clean})
                except Exception:
                    pass

        if attempt < eff_policy.max_oom_retries and turn_idx == 0:
            next_attempt = attempt + 1
            next_retries = oom_retries + 1
            next_gpu = self._pick_oom_retry_gpu(eff_policy, failed_gpu)
            next_ctx_id = f"ctx_{sess_id}_r{next_retries}"
            return True, next_attempt, next_retries, next_gpu, next_ctx_id

        tpots = ar_tpots_ms or []
        plan = str(turn.modality_plan)
        is_ar_only = plan == "ar_only"
        turn_records.append(
            {
                "turn_idx": int(turn_idx),
                "modality_plan": plan,
                "status": "oom",
                "ar_ttft_ms": round(ar_ttft_ms, 2) if ar_ttft_ms is not None else None,
                "ar_tpot_p95_ms": round(_percentile(tpots, 95.0), 2) if tpots else None,
                "ar_tpot_mean_ms": round(sum(tpots) / float(len(tpots)), 2) if tpots else None,
                "ar_tpot_strict_max_ms": round(max(tpots), 2) if tpots else None,
                "dit_ms": None,
                "height": 0 if is_ar_only else int(turn.height),
                "dit_steps": 0 if is_ar_only else int(turn.dit_steps),
                "kv_bytes": int(kv_bytes),
                "shipped": bool(shipped),
                "morphed": bool(morphed),
                "has_image": bool(has_image),
                "num_images": int(num_images),
                "image_res": int(image_res),
                "queue_ms": round(float(queue_ms), 2),
                "h2d_weight_ms": round(float(h2d_weight_ms), 2),
                "kv_transfer_ms": round(float(kv_transfer_ms), 2),
                "vae_ms": round(float(vae_ms), 2),
                "preempt_yield_ms": round(float(preempt_yield_ms), 2),
            }
        )
        return False, attempt, oom_retries, failed_gpu, ""

    async def _run_session_coro(
        self,
        sess: InterleavedSession,
        suite_t0: float,
        eff_policy: EnginePolicyConfig,
        ctrl_handle: int,
        save_outputs_dir: Path | None = None,
        on_turn_complete: Callable[[], float] | None = None,
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
        last_generated_res = 0

        ar_ttfts: list[float] = []
        ar_tpots: list[float] = []
        ar_turn_tpots: list[list[float]] = []
        ar_turn_has_image: list[bool] = []
        ar_turn_num_images: list[int] = []
        ar_turn_image_res: list[int] = []
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
        sess_queue_ms = 0.0
        sess_h2d_weight_ms = 0.0
        sess_kv_transfer_ms = 0.0
        sess_vae_ms = 0.0
        sess_preempt_yield_ms = 0.0
        is_tight_mot = eff_policy.cptg_architecture == "tight_mot"
        cur_turn_idx = 0
        cur_turn_plan = ""
        cur_turn_height = 0
        cur_turn_has_image = False
        cur_turn_num_images = 0
        cur_turn_image_res = 0

        def _sample_resident_alpha() -> float:
            ws = self.router.worker_states_snapshot()
            if not ws:
                return 1.0
            return round(sum(float(w.expert_alpha) for w in ws.values()) / float(len(ws)), 4)

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
            joint_ok: bool | None = None,
            error: str = "",
        ) -> SessionOutcome:
            now_pc = time.perf_counter()
            e2e_ms = round((now_pc - t_sess0) * 1000.0, 2)
            active_jct_ms = round(max(0.0, e2e_ms - think_time_ms), 2)
            completed_offset_s = round(max(0.0, now_pc - suite_t0), 3)
            eff_joint = (ar_ok and dit_ok) if joint_ok is None else bool(joint_ok)
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
                joint_slo_met=eff_joint if status == "ok" else False,
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
                ar_turn_has_image=ar_turn_has_image,
                ar_turn_num_images=ar_turn_num_images,
                ar_turn_image_res=ar_turn_image_res,
                queue_ms=round(sess_queue_ms, 2),
                h2d_weight_ms=round(sess_h2d_weight_ms, 2),
                kv_transfer_ms=round(sess_kv_transfer_ms, 2),
                vae_ms=round(sess_vae_ms, 2),
                preempt_yield_ms=round(sess_preempt_yield_ms, 2),
            )

        try:
            for ti, turn in enumerate(sess.turns):
                cur_turn_idx = int(ti)
                cur_turn_plan = str(turn.modality_plan)
                cur_turn_height = 0 if cur_turn_plan == "ar_only" else int(turn.height)
                if ti > 0:
                    think_s = _deterministic_think_s(sess.session_id, ti, eff_policy)
                    if think_s > 0.0:
                        t_th0 = time.perf_counter()
                        await asyncio.sleep(think_s)
                        think_time_ms += (time.perf_counter() - t_th0) * 1000.0

                plan = turn.modality_plan
                is_edit_followup = (plan == "dit_edit") and (ti > 0)
                has_prior_dit = any(t.modality_plan != "ar_only" for t in sess.turns[:ti])
                # Preserve Q1 Turn 1 (`dit_edit` after `ar_only` Turn 0): only request `use_prev_image`
                # when explicitly set or when a prior DiT turn actually produced an image, so
                # `bagel_adapter.py`'s `pending_vae_only_paths` handles Turn 1 VAE cleanly.
                turn_use_prev_image = (ti > 0) and (
                    bool(turn.use_prev_image) or (plan == "dit_edit" and has_prior_dit)
                )
                eff_prep_img_paths = (
                    []
                    if (is_edit_followup and turn_use_prev_image)
                    else list(turn.image_paths)
                )
                eff_turn_img_paths = list(turn.image_paths) if plan in ("ar_only", "cot_then_dit") else eff_prep_img_paths
                turn_num_images = len(eff_turn_img_paths) + (1 if turn_use_prev_image else 0)
                turn_has_image = turn_num_images > 0
                turn_image_res = 0
                if turn_has_image:
                    fallback_res = max(int(turn.height), int(turn.width), 512)
                    res_cands: list[int] = []
                    if turn_use_prev_image:
                        res_cands.append(last_generated_res if last_generated_res > 0 else fallback_res)
                    for rp in eff_turn_img_paths:
                        res_cands.append(self._probe_image_max_edge(rp, fallback=fallback_res))
                    turn_image_res = max(res_cands) if res_cands else fallback_res
                cur_turn_has_image = turn_has_image
                cur_turn_num_images = turn_num_images
                cur_turn_image_res = turn_image_res

                has_more_turns = (ti + 1) < len(sess.turns)
                has_more_edits = any(t.modality_plan == "dit_edit" for t in sess.turns[ti + 1 :])
                has_future_prev_image = any(
                    bool(t.use_prev_image) or (t.modality_plan == "dit_edit") for t in sess.turns[ti + 1 :]
                )

                attempt = 0
                while True:
                    if resident_gpu is None:
                        resident_gpu = self._pick_initial_gpu(eff_policy, plan)
                    eff_prev_img = (
                        (None if (is_tight_mot and src_ctx_alive) else prev_tmp_img) if turn_use_prev_image else None
                    )

                    # ---- Branch 1: Pure AR turn (VQA) ----
                    if plan == "ar_only":
                        ar_res, did_preempt = await self._execute_ar_phase(
                            eff_policy,
                            resident_gpu,
                            ctx_id,
                            turn,
                            prepare_for_dit=False,
                            use_prev_image=turn_use_prev_image,
                            prev_image_path=eff_prev_img,
                        )
                        if ar_res["status"] == "oom":
                            src_ctx_alive = False
                            should_retry, attempt, oom_retries, resident_gpu, next_cid = await self._retry_or_fail_oom(
                                eff_policy=eff_policy,
                                sess_id=sess.session_id,
                                turn_idx=ti,
                                turn=turn,
                                attempt=attempt,
                                oom_retries=oom_retries,
                                failed_gpu=resident_gpu,
                                cleanup_targets={(resident_gpu, ctx_id)},
                                kv_bytes=kv_bytes,
                                turn_records=turn_records,
                                has_image=turn_has_image,
                                num_images=turn_num_images,
                                image_res=turn_image_res,
                            )
                            if should_retry:
                                ctx_id = next_cid
                                continue
                            return _make_outcome("oom", error=ar_res.get("error", "oom"))

                        src_ctx_alive = True
                        if did_preempt:
                            preempted_ar_turns += 1
                        kv_bytes = int(ar_res.get("kv_bytes", kv_bytes))
                        turn_ttft = float(ar_res.get("ttft_ms", 0.0))
                        turn_tpots = [float(x) for x in ar_res.get("tpots_ms", [])]
                        turn_q_ms = float(ar_res.get("queue_ms", 0.0))
                        turn_h2d_ms = float(ar_res.get("h2d_weight_ms", 0.0))
                        turn_kv_xfer_ms = float(ar_res.get("kv_transfer_ms", 0.0))
                        turn_vae_ms = float(ar_res.get("vae_ms", 0.0))
                        sess_queue_ms += turn_q_ms
                        sess_h2d_weight_ms += turn_h2d_ms
                        sess_kv_transfer_ms += turn_kv_xfer_ms
                        sess_vae_ms += turn_vae_ms
                        ar_ttfts.append(turn_ttft)
                        ar_tpots.extend(turn_tpots)
                        ar_turn_tpots.append(turn_tpots)
                        ar_turn_has_image.append(turn_has_image)
                        ar_turn_num_images.append(turn_num_images)
                        ar_turn_image_res.append(turn_image_res)
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
                                "has_image": bool(turn_has_image),
                                "num_images": int(turn_num_images),
                                "image_res": int(turn_image_res),
                                "use_prev_image": bool(turn_use_prev_image),
                                "queue_ms": round(turn_q_ms, 2),
                                "h2d_weight_ms": round(turn_h2d_ms, 2),
                                "kv_transfer_ms": round(turn_kv_xfer_ms, 2),
                                "vae_ms": round(turn_vae_ms, 2),
                                "preempt_yield_ms": 0.0,
                            }
                        )
                        if on_turn_complete is not None:
                            on_turn_complete()
                        break

                    # ---- Branch 2: Visual-CoT / DiT Edit / High-Res T2I ----
                    t_dit0 = time.perf_counter()
                    cot_text_pending: str | None = None
                    turn_ar_ttft: float | None = None
                    turn_ar_tpots: list[float] = []
                    turn_shipped = False
                    turn_morphed = False
                    turn_q_ms = 0.0
                    turn_h2d_ms = 0.0
                    turn_kv_xfer_ms = 0.0
                    turn_vae_ms = 0.0
                    turn_yield_ms = 0.0

                    holding_src_lock = False
                    held_src_gpu = resident_gpu
                    if plan == "cot_then_dit":
                        cot_res, did_preempt = await self._execute_ar_phase(
                            eff_policy,
                            resident_gpu,
                            ctx_id,
                            turn,
                            prepare_for_dit=True,
                            use_prev_image=turn_use_prev_image,
                            prev_image_path=eff_prev_img,
                        )
                        if cot_res["status"] == "oom":
                            src_ctx_alive = False
                            should_retry, attempt, oom_retries, resident_gpu, next_cid = await self._retry_or_fail_oom(
                                eff_policy=eff_policy,
                                sess_id=sess.session_id,
                                turn_idx=ti,
                                turn=turn,
                                attempt=attempt,
                                oom_retries=oom_retries,
                                failed_gpu=resident_gpu,
                                cleanup_targets={(resident_gpu, ctx_id)},
                                kv_bytes=kv_bytes,
                                turn_records=turn_records,
                                has_image=turn_has_image,
                                num_images=turn_num_images,
                                image_res=turn_image_res,
                            )
                            if should_retry:
                                ctx_id = next_cid
                                continue
                            return _make_outcome("oom", error=cot_res.get("error", "oom"))

                        src_ctx_alive = True
                        if did_preempt:
                            preempted_ar_turns += 1
                        kv_bytes = int(cot_res.get("kv_bytes", kv_bytes))
                        turn_ar_ttft = float(cot_res.get("ttft_ms", 0.0))
                        turn_ar_tpots = [float(x) for x in cot_res.get("tpots_ms", [])]
                        turn_q_ms += float(cot_res.get("queue_ms", 0.0))
                        turn_h2d_ms += float(cot_res.get("h2d_weight_ms", 0.0))
                        turn_kv_xfer_ms += float(cot_res.get("kv_transfer_ms", 0.0))
                        turn_vae_ms += float(cot_res.get("vae_ms", 0.0))
                        ar_ttfts.append(turn_ar_ttft)
                        ar_tpots.extend(turn_ar_tpots)
                        ar_turn_tpots.append(turn_ar_tpots)
                        ar_turn_has_image.append(turn_has_image)
                        ar_turn_num_images.append(turn_num_images)
                        ar_turn_image_res.append(turn_image_res)
                        cot_text_pending = str(cot_res.get("text", ""))
                        t_dit0 = time.perf_counter()
                    else:
                        est_prep_ms = default_operator_profile().prep_overhead_ms
                        self.router.reserve_ar_queue(resident_gpu, est_prep_ms)
                        try:
                            t_pq0 = time.perf_counter()
                            prep_dl_ms = t_dit0 * 1000.0 + eff_policy.dit_turn_slo_for(turn.height, turn.dit_steps)
                            await self._acquire_exclusive_gpu(
                                resident_gpu,
                                est_mem_bytes=(512 if turn.height >= 1024 else 256) * MIB,
                                est_exec_ms=est_prep_ms,
                                slo_deadline_ms=prep_dl_ms,
                                op_priority=1,
                            )
                            holding_src_lock = True
                            held_src_gpu = resident_gpu
                            turn_q_ms += max(0.0, (time.perf_counter() - t_pq0) * 1000.0)
                            try:
                                prep_res = await self._rpc(
                                    resident_gpu,
                                    {
                                        "op": "PREP_DIT",
                                        "ctx_id": ctx_id,
                                        "prompt": turn.prompt,
                                        "height": turn.height,
                                        "width": turn.width,
                                        "image_paths": eff_prep_img_paths,
                                        "use_prev_image": turn_use_prev_image,
                                        "prev_image_path": eff_prev_img,
                                        "prepare_for_dit": True,
                                    },
                                )
                            except BaseException:
                                holding_src_lock = False
                                self._gpu_locks[held_src_gpu].release()
                                raise
                        finally:
                            self.router.release_ar_queue(resident_gpu, est_prep_ms)

                        if prep_res["status"] == "oom":
                            if holding_src_lock:
                                holding_src_lock = False
                                self._gpu_locks[held_src_gpu].release()
                            src_ctx_alive = False
                            should_retry, attempt, oom_retries, resident_gpu, next_cid = await self._retry_or_fail_oom(
                                eff_policy=eff_policy,
                                sess_id=sess.session_id,
                                turn_idx=ti,
                                turn=turn,
                                attempt=attempt,
                                oom_retries=oom_retries,
                                failed_gpu=resident_gpu,
                                cleanup_targets={(resident_gpu, ctx_id)},
                                kv_bytes=kv_bytes,
                                turn_records=turn_records,
                                has_image=turn_has_image,
                                num_images=turn_num_images,
                                image_res=turn_image_res,
                                queue_ms=turn_q_ms,
                            )
                            if should_retry:
                                ctx_id = next_cid
                                continue
                            return _make_outcome("oom", error=prep_res.get("error", "oom"))
                        src_ctx_alive = True
                        kv_bytes = int(prep_res.get("kv_bytes", kv_bytes))
                        turn_h2d_ms += float(prep_res.get("h2d_weight_ms", 0.0))
                        turn_vae_ms += float(prep_res.get("vae_ms", 0.0))

                    # Plan cross-paradigm transition & atomically reserve router/pacer state in C++ `MorphControlEngine`.
                    try:
                        now_rel_ms = (time.perf_counter() - suite_t0) * 1000.0
                        trans = self._plan_turn_transition(
                            ctrl_handle,
                            session_id=sess.session_id,
                            base_ctx_id=ctx_id,
                            turn_idx=ti,
                            attempt=attempt,
                            src_gpu=resident_gpu,
                            state_bytes=kv_bytes,
                            modality_plan=plan,
                            height=turn.height,
                            dit_steps=turn.dit_steps,
                            has_more_turns=has_more_turns,
                            has_more_edits=has_more_edits,
                            now_ms=now_rel_ms,
                            src_lock_held=bool(holding_src_lock and held_src_gpu == resident_gpu),
                        )
                    except BaseException:
                        if holding_src_lock:
                            holding_src_lock = False
                            self._gpu_locks[held_src_gpu].release()
                        raise
                    ship_mode = str(trans["ship_mode"])
                    dst_gpu = int(trans["dst_gpu"])
                    est_dit_ms = float(trans["est_dit_ms"])
                    reserved_kv_b = int(trans["reserved_kv_bytes"])
                    ephemeral_gen_ctx = bool(trans["ephemeral_gen_ctx"])
                    needs_return_ship = bool(trans["needs_return_ship"])
                    release_src_on_fwd = bool(trans["release_src_on_fwd"])
                    include_cfg_dup = bool(trans["include_cfg_dup_on_wire"])
                    dit_ctx_id = str(trans["dit_ctx_id"])
                    dit_dl_ms = t_dit0 * 1000.0 + eff_policy.dit_turn_slo_for(turn.height, turn.dit_steps)
                    est_dit_mem_b = int(kv_bytes) + ((4096 if turn.height >= 1024 else 1024) * MIB)

                    src_gpu_before_ship = resident_gpu
                    kv_res_released = reserved_kv_b == 0
                    save_tmp: str | None = None
                    if save_outputs_dir is not None:
                        save_tmp = str(save_outputs_dir / f"{sess.session_id}_t{ti}.png")
                    elif (
                        has_future_prev_image
                        and (not is_tight_mot)
                        and eff_policy.routing_mode in ("twostage_shm", "static_nccl_fission")
                    ):
                        save_tmp = str(self.tmp_root / f"{sess.session_id}_t{ti}.png")
                        owned_tmp_imgs.append(save_tmp)

                    dit_res: dict[str, Any] = {"status": "error", "error": "not_run"}
                    reserved_dit_gpu = dst_gpu
                    fallback_in_place = False
                    try:
                        if ship_mode in ("nccl_dual", "nccl_static"):
                            pass_src_lock = holding_src_lock and (held_src_gpu == resident_gpu)
                            if pass_src_lock:
                                holding_src_lock = False
                            async with self._nccl_transfer_guard(
                                resident_gpu,
                                dst_gpu,
                                ctx_id,
                                dit_ctx_id,
                                include_cfg_dup=include_cfg_dup,
                                release_src_ctx=release_src_on_fwd,
                                defer_src_release=(ship_mode == "nccl_dual"),
                                src_lock_held=pass_src_lock,
                                dst_est_mem_bytes=est_dit_mem_b,
                                dst_est_exec_ms=est_dit_ms,
                                dst_slo_deadline_ms=dit_dl_ms,
                                dst_op_priority=1,
                            ) as ship:
                                if release_src_on_fwd and ship.get("src_ctx_released", ship.get("packed_ok")):
                                    src_ctx_alive = False
                                self.router.release_kv_reservation(
                                    dst_gpu,
                                    reserved_kv_b,
                                    latest_kv_free_bytes=ship.get("kv_free_bytes"),
                                    telemetry_seq=ship.get("telemetry_seq"),
                                )
                                kv_res_released = True
                                if ship.get("status") != "ok":
                                    if ship_mode == "nccl_dual" and ship.get("src_ctx_preserved"):
                                        fallback_in_place = True
                                    else:
                                        dit_res = {"status": "oom", "error": ship.get("error", f"{ship_mode}_oom")}
                                else:
                                    if ship_mode == "nccl_dual":
                                        resident_gpu = dst_gpu
                                        src_ctx_alive = True
                                    path_a_ships += 1
                                    turn_shipped = True
                                    fwd_b = int(ship.get("bytes", kv_bytes))
                                    if not self.topo.same_node(src_gpu_before_ship, dst_gpu):
                                        cross_node_ships += 1
                                        cross_node_ship_bytes_list.append(fwd_b)
                                    ship_bytes_list.append(fwd_b)
                                    ship_ms_list.append(float(ship.get("ship_ms", 0.0)))
                                    turn_kv_xfer_ms += float(ship.get("kv_transfer_ms", ship.get("ship_ms", 0.0)))
                                    turn_q_ms += float(ship.get("queue_ms", 0.0))
                                    dit_res = await self._exec_dit_on_locked_gpu(
                                        dst_gpu,
                                        dit_ctx_id,
                                        turn,
                                        ti,
                                        save_tmp=save_tmp,
                                        ephemeral_gen_ctx=ephemeral_gen_ctx,
                                        needs_return_ship=needs_return_ship,
                                    )
                            if fallback_in_place:
                                # Path-A NCCL staging hit OOM while `ctx_id` remained intact on `resident_gpu`:
                                # seamlessly fall back to in-place Path-B execution (0 extra staging HBM).
                                self.router.release_dit_reservation(reserved_dit_gpu, est_dit_ms)
                                dst_gpu = resident_gpu
                                reserved_dit_gpu = dst_gpu
                                self.router.reserve_ar_queue(reserved_dit_gpu, est_dit_ms)
                                dit_ctx_id = ctx_id
                                ephemeral_gen_ctx = False
                                needs_return_ship = False
                                path_b_morphs += 1
                                turn_morphed = True
                                t_dq_fb = time.perf_counter()
                                await self._acquire_exclusive_gpu(
                                    dst_gpu,
                                    est_mem_bytes=est_dit_mem_b,
                                    est_exec_ms=est_dit_ms,
                                    slo_deadline_ms=dit_dl_ms,
                                    op_priority=1,
                                )
                                turn_q_ms += max(0.0, (time.perf_counter() - t_dq_fb) * 1000.0)
                                try:
                                    dit_res = await self._exec_dit_on_locked_gpu(
                                        dst_gpu,
                                        dit_ctx_id,
                                        turn,
                                        ti,
                                        save_tmp=save_tmp,
                                        ephemeral_gen_ctx=False,
                                        needs_return_ship=False,
                                    )
                                finally:
                                    self._gpu_locks[dst_gpu].release()
                        else:
                            if ship_mode == "shm_static":
                                pass_src_lock = holding_src_lock and (held_src_gpu == resident_gpu)
                                if pass_src_lock:
                                    holding_src_lock = False
                                ship = await self._ship_shm(
                                    resident_gpu,
                                    dst_gpu,
                                    ctx_id,
                                    dit_ctx_id,
                                    include_cfg_dup=include_cfg_dup,
                                    release_src_ctx=release_src_on_fwd,
                                    src_lock_held=pass_src_lock,
                                    dst_est_mem_bytes=est_dit_mem_b,
                                    dst_est_exec_ms=est_dit_ms,
                                    dst_slo_deadline_ms=dit_dl_ms,
                                    dst_op_priority=1,
                                )
                                if release_src_on_fwd and ship.get("sent_ok"):
                                    src_ctx_alive = False
                                self.router.release_kv_reservation(
                                    dst_gpu,
                                    reserved_kv_b,
                                    latest_kv_free_bytes=ship.get("kv_free_bytes"),
                                    telemetry_seq=ship.get("telemetry_seq"),
                                )
                                kv_res_released = True
                                if ship.get("status") != "ok":
                                    dit_res = {"status": "oom", "error": ship.get("error", "shm_ship_oom")}
                                else:
                                    path_a_ships += 1
                                    turn_shipped = True
                                    fwd_b = int(ship.get("bytes", kv_bytes))
                                    if not self.topo.same_node(src_gpu_before_ship, dst_gpu):
                                        cross_node_ships += 1
                                        cross_node_ship_bytes_list.append(fwd_b)
                                    ship_bytes_list.append(fwd_b)
                                    turn_kv_xfer_ms += float(ship.get("kv_transfer_ms", 0.0))
                                    turn_q_ms += float(ship.get("queue_ms", 0.0))
                            elif eff_policy.routing_mode == "dual_path":
                                path_b_morphs += 1
                                turn_morphed = True

                            if dit_res.get("status") != "oom":
                                reuse_held_lock = holding_src_lock and (held_src_gpu == dst_gpu)
                                if reuse_held_lock:
                                    is_expired_turn = (dit_dl_ms - (time.perf_counter() * 1000.0) - est_dit_ms) < 0.0
                                    should_yield_to_sched = (
                                        eff_policy.gpu_queue_mode == "memory_slack"
                                        and self._gpu_locks[dst_gpu].has_waiters()
                                        and (
                                            turn.height >= 1024
                                            or is_expired_turn
                                            or est_dit_mem_b > self._free_mem_bytes_for_gpu(dst_gpu)
                                        )
                                    )
                                    if should_yield_to_sched:
                                        holding_src_lock = False
                                        self._gpu_locks[held_src_gpu].release()
                                        reuse_held_lock = False
                                if holding_src_lock and not reuse_held_lock:
                                    holding_src_lock = False
                                    self._gpu_locks[held_src_gpu].release()

                                if reuse_held_lock:
                                    holding_src_lock = False
                                else:
                                    t_dq_dit = time.perf_counter()
                                    await self._acquire_exclusive_gpu(
                                        dst_gpu,
                                        est_mem_bytes=est_dit_mem_b,
                                        est_exec_ms=est_dit_ms,
                                        slo_deadline_ms=dit_dl_ms,
                                        op_priority=1,
                                    )
                                    turn_q_ms += max(0.0, (time.perf_counter() - t_dq_dit) * 1000.0)
                                try:
                                    dit_res = await self._exec_dit_on_locked_gpu(
                                        dst_gpu,
                                        dit_ctx_id,
                                        turn,
                                        ti,
                                        save_tmp=save_tmp,
                                        ephemeral_gen_ctx=ephemeral_gen_ctx,
                                        needs_return_ship=needs_return_ship,
                                    )
                                finally:
                                    self._gpu_locks[dst_gpu].release()
                    finally:
                        if holding_src_lock:
                            holding_src_lock = False
                            self._gpu_locks[held_src_gpu].release()
                        if not kv_res_released:
                            self.router.release_kv_reservation(dst_gpu, reserved_kv_b)
                        self.router.release_dit_reservation(reserved_dit_gpu, est_dit_ms)

                    # Return-ship updated post-DiT `main_ctx` back to `und` rank for multi-turn `tight_mot` fission.
                    if dit_res.get("status") == "ok" and needs_return_ship:
                        ret_kv_b = max(0, int(dit_res.get("kv_bytes", kv_bytes)))
                        self.router.reserve_kv_bytes(resident_gpu, ret_kv_b)
                        ret_ship: dict[str, Any] = {}
                        try:
                            if ship_mode == "nccl_static":
                                async with self._nccl_transfer_guard(
                                    dst_gpu,
                                    resident_gpu,
                                    dit_ctx_id,
                                    ctx_id,
                                    include_cfg_dup=False,
                                    release_src_ctx=True,
                                ) as rs:
                                    ret_ship = dict(rs)
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
                            turn_kv_xfer_ms += float(ret_ship.get("kv_transfer_ms", ret_ship.get("ship_ms", 0.0)))
                            turn_q_ms += float(ret_ship.get("queue_ms", 0.0))
                            if not self.topo.same_node(dst_gpu, resident_gpu):
                                cross_node_ships += 1
                                cross_node_ship_bytes_list.append(ret_b)
                            if ship_ms_list and ship_mode == "nccl_static":
                                ship_ms_list[-1] = round(
                                    ship_ms_list[-1] + float(ret_ship.get("ship_ms", 0.0)), 3
                                )

                    if dit_res["status"] == "oom":
                        src_ctx_alive = False
                        should_retry, attempt, oom_retries, resident_gpu, next_cid = await self._retry_or_fail_oom(
                            eff_policy=eff_policy,
                            sess_id=sess.session_id,
                            turn_idx=ti,
                            turn=turn,
                            attempt=attempt,
                            oom_retries=oom_retries,
                            failed_gpu=dst_gpu,
                            cleanup_targets={(resident_gpu, ctx_id), (dst_gpu, dit_ctx_id)},
                            kv_bytes=kv_bytes,
                            turn_records=turn_records,
                            ar_ttft_ms=turn_ar_ttft,
                            ar_tpots_ms=turn_ar_tpots,
                            shipped=turn_shipped,
                            morphed=turn_morphed,
                            has_image=turn_has_image,
                            num_images=turn_num_images,
                            image_res=turn_image_res,
                            queue_ms=turn_q_ms,
                            h2d_weight_ms=turn_h2d_ms,
                            kv_transfer_ms=turn_kv_xfer_ms,
                            vae_ms=turn_vae_ms,
                            preempt_yield_ms=turn_yield_ms,
                        )
                        if should_retry:
                            ctx_id = next_cid
                            continue
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

                    turn_q_ms += float(dit_res.get("queue_ms", 0.0))
                    turn_h2d_ms += float(dit_res.get("h2d_weight_ms", 0.0))
                    turn_kv_xfer_ms += float(dit_res.get("kv_transfer_ms", 0.0))
                    turn_vae_ms += float(dit_res.get("vae_ms", 0.0))
                    turn_yield_ms += float(dit_res.get("preempt_yield_ms", 0.0))
                    sess_queue_ms += turn_q_ms
                    sess_h2d_weight_ms += turn_h2d_ms
                    sess_kv_transfer_ms += turn_kv_xfer_ms
                    sess_vae_ms += turn_vae_ms
                    sess_preempt_yield_ms += turn_yield_ms

                    last_generated_res = max(int(turn.height), int(turn.width))
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
                            "has_image": bool(turn_has_image),
                            "num_images": int(turn_num_images),
                            "image_res": int(turn_image_res),
                            "use_prev_image": bool(turn_use_prev_image),
                            "queue_ms": round(turn_q_ms, 2),
                            "h2d_weight_ms": round(turn_h2d_ms, 2),
                            "kv_transfer_ms": round(turn_kv_xfer_ms, 2),
                            "vae_ms": round(turn_vae_ms, 2),
                            "preempt_yield_ms": round(turn_yield_ms, 2),
                        }
                    )
                    if on_turn_complete is not None:
                        on_turn_complete()
                    break

            slo_eval = self._evaluate_session_slos(
                eff_policy,
                ar_ttfts_ms=ar_ttfts,
                ar_tpots_ms=ar_tpots,
                ar_turn_tpots_ms=ar_turn_tpots,
                dit_turns_ms=dit_turns,
                dit_turn_heights=dit_heights,
                dit_turn_steps=dit_steps_list,
                ar_turn_has_image=ar_turn_has_image,
                ar_turn_num_images=ar_turn_num_images,
                ar_turn_image_res=ar_turn_image_res,
                turn_records=turn_records,
            )
            return _make_outcome(
                "ok",
                ar_ok=bool(slo_eval["ar_slo_met"]),
                ar_p95_ok=bool(slo_eval["ar_slo_p95_tpot_met"]),
                ar_mean_ok=bool(slo_eval["ar_slo_mean_tpot_met"]),
                ar_strict_ok=bool(slo_eval["ar_slo_strict_max_tpot_met"]),
                dit_ok=bool(slo_eval["dit_slo_met"]),
                joint_ok=bool(slo_eval["joint_slo_met"]),
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
                    "has_image": bool(cur_turn_has_image),
                    "num_images": int(cur_turn_num_images),
                    "image_res": int(cur_turn_image_res),
                    "queue_ms": 0.0,
                    "h2d_weight_ms": 0.0,
                    "kv_transfer_ms": 0.0,
                    "vae_ms": 0.0,
                    "preempt_yield_ms": 0.0,
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

    async def _run_suite_async(
        self,
        suite: WorkloadSuite,
        target_qps: float | None,
        burstgpt_deltas: list[float] | None,
        gentd26_deltas: list[float] | None,
        save_outputs_dir: Path | None = None,
        run_options: RunWindowOptions | None = None,
    ) -> ClusterRunReport:
        eff_policy = run_options.apply_to_policy(self.policy) if run_options is not None else self.policy
        n_gpus = eff_policy.num_gpus
        self._gpu_locks = self._build_gpu_locks(eff_policy)
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

        self.router.pacer.reset_analytical_slots()
        if self.pacer is not None and self.pacer is not self.router.pacer:
            self.pacer.reset_analytical_slots()

        should_reset_alpha = self._suites_completed > 0
        reset_acks = await asyncio.gather(
            *(self._rpc(g, {"op": "RESET_SUITE", "reset_alpha": should_reset_alpha}) for g in range(n_gpus))
        )
        if should_reset_alpha:
            self._init_router_workers_from_acks({g: reset_acks[g] for g in range(n_gpus)})
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

        alpha_timeline: list[dict[str, float]] = []
        with self._scoped_ctrl_handle(eff_policy) as ctrl_handle:
            t0 = time.perf_counter()
            t_run0 = time.monotonic()

            def _record_alpha_sample() -> float:
                ws = self.router.worker_states_snapshot()
                mean_alpha = (
                    round(sum(float(w.expert_alpha) for w in ws.values()) / float(len(ws)), 4)
                    if ws
                    else 1.0
                )
                alpha_timeline.append(
                    {
                        "t_s": round(max(0.0, time.monotonic() - t_run0), 3),
                        "alpha": mean_alpha,
                        "resident_alpha": mean_alpha,
                    }
                )
                return mean_alpha

            async def _alpha_ticker(interval_s: float = 2.0) -> None:
                while True:
                    await asyncio.sleep(interval_s)
                    _record_alpha_sample()

            _record_alpha_sample()
            ticker_task = asyncio.create_task(_alpha_ticker(2.0))
            try:
                outcomes = await asyncio.gather(
                    *(
                        self._run_session_coro(
                            s,
                            t0,
                            eff_policy=eff_policy,
                            ctrl_handle=ctrl_handle,
                            save_outputs_dir=save_outputs_dir,
                            on_turn_complete=_record_alpha_sample,
                        )
                        for s in sessions
                    )
                )
            finally:
                ticker_task.cancel()
                try:
                    await ticker_task
                except asyncio.CancelledError:
                    pass
                await self._drain_bg_releases()
            _record_alpha_sample()
            wall_s = max(time.perf_counter() - t0, 1e-6)

        stat_res: list[dict[str, Any]] = []
        raw_stats = await asyncio.gather(
            *(self._rpc(g, {"op": "GET_STATS"}) for g in range(n_gpus)),
            return_exceptions=True,
        )
        sm_util_samples: list[float] = []
        mem_util_samples: list[float] = []
        hbm_util_samples: list[float] = []
        for item in raw_stats:
            if isinstance(item, Mapping):
                st_d = dict(item)
                sm_val = st_d.get("nvml_sm_util_mean_pct")
                has_sm = isinstance(sm_val, (int, float)) and int(
                    st_d.get("nvml_sm_util_samples", 1 if float(sm_val) > 0.0 else 0)
                ) > 0
                if has_sm:
                    sm_f = round(float(sm_val), 2)
                    st_d["nvml_sm_util_mean_pct"] = sm_f
                    sm_util_samples.append(sm_f)
                else:
                    st_d.pop("nvml_sm_util_mean_pct", None)
                mem_val = st_d.get("nvml_mem_util_mean_pct")
                has_mem = isinstance(mem_val, (int, float)) and int(
                    st_d.get("nvml_mem_util_samples", 1 if float(mem_val) > 0.0 else 0)
                ) > 0
                if has_mem:
                    mem_f = round(float(mem_val), 2)
                    st_d["nvml_mem_util_mean_pct"] = mem_f
                    mem_util_samples.append(mem_f)
                else:
                    st_d.pop("nvml_mem_util_mean_pct", None)
                hbm_val = st_d.get("effective_hbm_util_mean_pct")
                has_hbm = isinstance(hbm_val, (int, float)) and int(
                    st_d.get("effective_hbm_util_samples", 1 if float(hbm_val) > 0.0 else 0)
                ) > 0
                if has_hbm:
                    hbm_f = round(float(hbm_val), 2)
                    st_d["effective_hbm_util_mean_pct"] = hbm_f
                    hbm_util_samples.append(hbm_f)
                else:
                    st_d.pop("effective_hbm_util_mean_pct", None)
                stat_res.append(st_d)
            else:
                stat_res.append({"oom_count": 0})

        cluster_sm_util_pct: float | None = (
            round(sum(sm_util_samples) / float(len(sm_util_samples)), 2) if sm_util_samples else None
        )
        cluster_mem_util_pct: float | None = (
            round(sum(mem_util_samples) / float(len(mem_util_samples)), 2) if mem_util_samples else None
        )
        cluster_hbm_util_pct: float | None = (
            round(sum(hbm_util_samples) / float(len(hbm_util_samples)), 2) if hbm_util_samples else None
        )
        if stat_res:
            stat_res[0]["alpha_timeline"] = alpha_timeline
        else:
            entry: dict[str, Any] = {"oom_count": 0, "alpha_timeline": alpha_timeline}
            if cluster_sm_util_pct is not None:
                entry["nvml_sm_util_mean_pct"] = cluster_sm_util_pct
            if cluster_mem_util_pct is not None:
                entry["nvml_mem_util_mean_pct"] = cluster_mem_util_pct
            if cluster_hbm_util_pct is not None:
                entry["effective_hbm_util_mean_pct"] = cluster_hbm_util_pct
            stat_res.append(entry)

        agg = call_json(
            self._lib,
            self._lib.ms_ctrl_aggregate_report_json,
            encode_json(eff_policy.to_dict()),
            str(suite.name).encode("utf-8"),
            float(eff_qps),
            float(wall_s),
            float(self.host_shmem_delta_gib),
            encode_json([asdict(o) for o in outcomes]),
            encode_json(stat_res),
        )
        if "nvml_sm_util_mean_pct" not in agg or (
            agg.get("nvml_sm_util_mean_pct") is None and cluster_sm_util_pct is not None
        ):
            agg["nvml_sm_util_mean_pct"] = cluster_sm_util_pct
        if "nvml_mem_util_mean_pct" not in agg or (
            agg.get("nvml_mem_util_mean_pct") is None and cluster_mem_util_pct is not None
        ):
            agg["nvml_mem_util_mean_pct"] = cluster_mem_util_pct
        if "effective_hbm_util_mean_pct" not in agg or (
            agg.get("effective_hbm_util_mean_pct") is None and cluster_hbm_util_pct is not None
        ):
            agg["effective_hbm_util_mean_pct"] = cluster_hbm_util_pct
        q_reorders = sum(lk.stats().get("reorders_total", 0) for lk in self._gpu_locks)
        q_bypasses = sum(lk.stats().get("memory_bypasses_total", 0) for lk in self._gpu_locks)
        q_starv = sum(lk.stats().get("starvation_promotions_total", 0) for lk in self._gpu_locks)
        agg["queue_reorders_total"] = int(q_reorders)
        agg["queue_memory_bypasses_total"] = int(q_bypasses)
        agg["queue_starvation_promotions_total"] = int(q_starv)
        if "alpha_timeline" not in agg:
            agg["alpha_timeline"] = alpha_timeline
        return ClusterRunReport.from_dict(agg)

    def run_suite(
        self,
        suite: WorkloadSuite,
        target_qps: float | None = None,
        burstgpt_deltas: list[float] | None = None,
        gentd26_deltas: list[float] | None = None,
        save_outputs_dir: Path | None = None,
        *,
        run_options: RunWindowOptions | None = None,
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
                run_options=run_options,
            )
        )

    def shutdown(self) -> None:
        for lk in self._gpu_locks:
            if hasattr(lk, "close"):
                lk.close()
        self._gpu_locks = []
        if not self._started:
            return
        self.fleet.shutdown()
        shutil.rmtree(self.tmp_root, ignore_errors=True)
        self._started = False


__all__ = [
    "ClusterRunReport",
    "MemorySlackGpuLock",
    "MorphServeClusterEngine",
    "SessionOutcome",
]
