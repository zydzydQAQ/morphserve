"""Python interface to C++17 `DualPathMorphRouter` (`ms_router_*`) and `HalfNodeH2DPacer` (`ms_pacer_*`)."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Callable, Iterator, Mapping, Sequence

from morphserve._core import call_json, encode_json, get_core_lib, last_error

from .topology import ClusterTopologyProfile, DEFAULT_OPERATOR_PROFILE, OperatorLatencyProfile

MIB = 1024**2
GIB = 1024**3


@dataclass
class CPTGRequestSpec:
    """Unified Cross-Paradigm Transition Graph (CPTG) request specification."""

    req_id: str
    src_gpu: int
    architecture: str = "tight_mot"  # "tight_mot" | "bridge_mllm_dit"
    kv_bytes: int = 0
    bridge_bytes: int = 0
    gen_expert_bytes: int = DEFAULT_OPERATOR_PROFILE.bagel_expert_bytes
    dit_step1_ms: float = DEFAULT_OPERATOR_PROFILE.bagel_dit_512_step_ms
    dit_total_steps: int = 20
    resolution: int = 512
    interleaved_roundtrip_gamma: int = 1

    def effective_state_bytes(self) -> int:
        if self.architecture == "bridge_mllm_dit":
            return int(self.bridge_bytes if self.bridge_bytes > 0 else self.kv_bytes)
        if self.architecture == "tight_mot":
            return int(self.kv_bytes)
        raise ValueError(f"unsupported CPTG architecture: {self.architecture}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "req_id": self.req_id,
            "src_gpu": int(self.src_gpu),
            "architecture": self.architecture,
            "kv_bytes": int(self.kv_bytes),
            "bridge_bytes": int(self.bridge_bytes),
            "gen_expert_bytes": int(self.gen_expert_bytes),
            "dit_step1_ms": float(self.dit_step1_ms),
            "dit_total_steps": int(self.dit_total_steps),
            "resolution": int(self.resolution),
            "interleaved_roundtrip_gamma": int(self.interleaved_roundtrip_gamma),
        }


@dataclass
class WorkerState:
    """Instantaneous HBM residency, queue backlog, and H2D state of one GPU worker."""

    gpu_id: int
    role: str = "hybrid"  # "und" | "gen" | "hybrid"
    expert_alpha: float = 1.0
    queue_wait_ms: float = 0.0
    active_h2d_transfers: int = 0
    kv_free_bytes: int = 12 * GIB
    p2p_concurrent_links: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_id": int(self.gpu_id),
            "role": self.role,
            "expert_alpha": float(self.expert_alpha),
            "queue_wait_ms": float(self.queue_wait_ms),
            "active_h2d_transfers": int(self.active_h2d_transfers),
            "kv_free_bytes": int(self.kv_free_bytes),
            "p2p_concurrent_links": int(self.p2p_concurrent_links),
        }


@dataclass(frozen=True)
class RouteDecision:
    """Selected AR->DiT execution path and analytical latency breakdown."""

    req_id: str
    path: str  # "state_ship" | "in_place_morph"
    src_gpu: int
    dst_gpu: int
    t_ship_ms: float
    t_morph_ms: float
    chosen_cost_ms: float
    selected_bw_gbps: float
    effective_state_mib: float
    reason: str
    hop_tier: str = "in_place"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RouteDecision":
        t_morph = d["t_morph_ms"]
        return cls(
            req_id=str(d["req_id"]),
            path=str(d["path"]),
            src_gpu=int(d["src_gpu"]),
            dst_gpu=int(d["dst_gpu"]),
            t_ship_ms=float(d["t_ship_ms"]),
            t_morph_ms=float("inf") if t_morph is None or float(t_morph) > 1e14 else float(t_morph),
            chosen_cost_ms=float(d["chosen_cost_ms"]),
            selected_bw_gbps=float(d["selected_bw_gbps"]),
            effective_state_mib=float(d["effective_state_mib"]),
            reason=str(d["reason"]),
            hop_tier=str(d.get("hop_tier", "in_place")),
        )


class HalfNodeH2DPacer:
    """C++-backed PCIe half-node H2D DMA pacer (`ms_pacer_*`: analytical timeline + thread + POSIX `flock`)."""

    def __init__(
        self,
        topology: ClusterTopologyProfile,
        max_concurrent_per_half: int = 1,
        lock_dir: str | Path = "/tmp/morphserve_h2d_locks",
    ) -> None:
        self.topology = topology
        self.max_concurrent_per_half = max(1, int(max_concurrent_per_half))
        self.lock_dir = Path(lock_dir)
        self._lib = get_core_lib()

        pairs: list[tuple[int, int]] = []
        for grp in topology.half_groups:
            for gid in grp.gpu_ids:
                pairs.append((int(gid), int(grp.group_id)))
        n = len(pairs)
        g_arr = (ctypes.c_int * n)(*(p[0] for p in pairs)) if n > 0 else None
        grp_arr = (ctypes.c_int * n)(*(p[1] for p in pairs)) if n > 0 else None
        h = int(
            self._lib.ms_pacer_create(
                g_arr,
                grp_arr,
                ctypes.c_size_t(n),
                ctypes.c_int(self.max_concurrent_per_half),
                str(self.lock_dir).encode("utf-8"),
            )
            or 0
        )
        if h == 0:
            raise RuntimeError(f"ms_pacer_create failed: {last_error(self._lib)}")
        self._handle = h

    def __del__(self) -> None:
        h = getattr(self, "_handle", 0)
        if h:
            self._handle = 0
            try:
                self._lib.ms_pacer_destroy(ctypes.c_void_p(h))
            except Exception:
                pass

    def predicted_wait_ms(self, gpu_id: int, now_ms: float = 0.0) -> float:
        _ = self.topology.half_group_of(gpu_id)
        return float(
            self._lib.ms_pacer_predicted_wait_ms(
                ctypes.c_void_p(self._handle), ctypes.c_int(int(gpu_id)), ctypes.c_double(float(now_ms))
            )
        )

    def reserve_slot(self, gpu_id: int, duration_ms: float, now_ms: float = 0.0) -> float:
        _ = self.topology.half_group_of(gpu_id)
        return float(
            self._lib.ms_pacer_reserve_slot(
                ctypes.c_void_p(self._handle),
                ctypes.c_int(int(gpu_id)),
                ctypes.c_double(float(duration_ms)),
                ctypes.c_double(float(now_ms)),
            )
        )

    def reset_analytical_slots(self) -> None:
        self._lib.ms_pacer_reset_slots(ctypes.c_void_p(self._handle))

    @contextmanager
    def acquire_thread_slot(self, gpu_id: int) -> Iterator[int]:
        gid = self.topology.half_group_of(gpu_id).group_id
        self._lib.ms_pacer_acquire_thread(ctypes.c_void_p(self._handle), ctypes.c_int(int(gpu_id)))
        try:
            yield gid
        finally:
            self._lib.ms_pacer_release_thread(ctypes.c_void_p(self._handle), ctypes.c_int(int(gpu_id)))

    @contextmanager
    def acquire_process_slot(self, gpu_id: int) -> Iterator[int]:
        _ = self.topology.half_group_of(gpu_id)
        out_gid = ctypes.c_int(-1)
        fd = int(
            self._lib.ms_pacer_acquire_process_flock(
                ctypes.c_void_p(self._handle), ctypes.c_int(int(gpu_id)), ctypes.byref(out_gid)
            )
        )
        if fd < 0:
            raise OSError(f"ms_pacer_acquire_process_flock({gpu_id}) failed: {last_error(self._lib)}")
        try:
            yield int(out_gid.value)
        finally:
            self._lib.ms_pacer_release_process_flock(ctypes.c_int(fd))


class DualPathMorphRouter:
    """Stateful placement, queue reservation, and Dual-Path AR->DiT router backed by C++17 `DualPathMorphRouter`."""

    def __init__(
        self,
        topology: ClusterTopologyProfile,
        pacer: HalfNodeH2DPacer | None = None,
        operator_profile: OperatorLatencyProfile = DEFAULT_OPERATOR_PROFILE,
        state_handoff_ms: float | None = None,
        map_overhead_ms_full: float = 12.0,
        overlap_efficiency: float = 0.90,
        allow_cross_half_state_ship: bool = True,
        allow_inter_node_state_ship: bool = True,
        max_inter_node_state_bytes: int = 128 * MIB,
    ) -> None:
        self.topology = topology
        self.pacer = pacer if pacer is not None else HalfNodeH2DPacer(topology)
        self.operator_profile = operator_profile
        self.state_handoff_ms = state_handoff_ms
        self.map_overhead_ms_full = float(map_overhead_ms_full)
        self.overlap_efficiency = float(overlap_efficiency)
        self.allow_cross_half_state_ship = bool(allow_cross_half_state_ship)
        self.allow_inter_node_state_ship = bool(allow_inter_node_state_ship)
        self.max_inter_node_state_bytes = int(max_inter_node_state_bytes)

        self.worker_states: dict[int, WorkerState] = {}
        self._synced_fields: dict[int, tuple[float, float, int, int, int]] = {}
        self._mu = threading.RLock()
        self._lib = get_core_lib()

        opts = {
            "state_handoff_ms": None if state_handoff_ms is None else float(state_handoff_ms),
            "map_overhead_ms_full": self.map_overhead_ms_full,
            "overlap_efficiency": self.overlap_efficiency,
            "allow_cross_half_state_ship": self.allow_cross_half_state_ship,
            "allow_inter_node_state_ship": self.allow_inter_node_state_ship,
            "max_inter_node_state_bytes": self.max_inter_node_state_bytes,
        }
        h = int(
            self._lib.ms_router_create(
                encode_json(topology.to_dict()),
                ctypes.c_void_p(self.pacer._handle),
                encode_json(opts),
            )
            or 0
        )
        if h == 0:
            raise RuntimeError(f"ms_router_create failed: {last_error(self._lib)}")
        self._handle = h

    def __del__(self) -> None:
        h = getattr(self, "_handle", 0)
        if h:
            self._handle = 0
            try:
                self._lib.ms_router_destroy(ctypes.c_void_p(h))
            except Exception:
                pass

    @staticmethod
    def _snap_tuple(w: WorkerState) -> tuple[float, float, int, int, int]:
        return (
            float(w.expert_alpha),
            float(w.queue_wait_ms),
            int(w.active_h2d_transfers),
            int(w.kv_free_bytes),
            int(w.p2p_concurrent_links),
        )

    def _push_python_mutations_locked(self) -> None:
        for gid, w in self.worker_states.items():
            cur = self._snap_tuple(w)
            prev = self._synced_fields.get(gid)
            if prev == cur:
                continue
            has_a = 1 if (prev is None or cur[0] != prev[0]) else 0
            has_q = 1 if (prev is None or cur[1] != prev[1]) else 0
            has_h = 1 if (prev is None or cur[2] != prev[2]) else 0
            has_k = 1 if (prev is None or cur[3] != prev[3]) else 0
            has_p = 1 if (prev is None or cur[4] != prev[4]) else 0
            self._lib.ms_router_set_worker_fields(
                ctypes.c_void_p(self._handle),
                ctypes.c_int(int(gid)),
                has_a,
                ctypes.c_double(cur[0]),
                has_q,
                ctypes.c_double(cur[1]),
                has_h,
                ctypes.c_int(cur[2]),
                has_k,
                ctypes.c_int64(cur[3]),
                has_p,
                ctypes.c_int(cur[4]),
            )
            self._synced_fields[gid] = cur

    def _pull_cpp_workers_locked(self) -> None:
        obj = call_json(self._lib, self._lib.ms_router_get_workers_json, ctypes.c_void_p(self._handle))
        items = obj.values() if isinstance(obj, Mapping) else obj
        for item in items:
            gid = int(item["gpu_id"])
            w = self.worker_states.get(gid)
            if w is None:
                w = WorkerState(gpu_id=gid, role=str(item.get("role", "hybrid")))
                self.worker_states[gid] = w
            w.role = str(item.get("role", w.role))
            w.expert_alpha = float(item["expert_alpha"])
            w.queue_wait_ms = float(item["queue_wait_ms"])
            w.active_h2d_transfers = int(item["active_h2d_transfers"])
            w.kv_free_bytes = int(item["kv_free_bytes"])
            w.p2p_concurrent_links = int(item["p2p_concurrent_links"])
            self._synced_fields[gid] = self._snap_tuple(w)

    def init_workers(
        self,
        num_gpus: int,
        role_of: Callable[[int], str],
        initial_alphas: Mapping[int, float],
        kv_free_bytes: int | Mapping[int, int] = 16 * GIB,
    ) -> dict[int, WorkerState]:
        with self._mu:
            n = int(num_gpus)
            payload = {
                "num_gpus": n,
                "roles": {str(g): str(role_of(g)) for g in range(n)},
                "initial_alphas": {str(g): float(initial_alphas.get(g, 1.0)) for g in range(n)},
                "kv_free_bytes": {
                    str(g): int(kv_free_bytes.get(g, 16 * GIB))
                    if isinstance(kv_free_bytes, Mapping)
                    else int(kv_free_bytes)
                    for g in range(n)
                },
            }
            obj = call_json(
                self._lib,
                self._lib.ms_router_init_workers_json,
                ctypes.c_void_p(self._handle),
                encode_json(payload),
            )
            self.worker_states.clear()
            self._synced_fields.clear()
            items = obj.values() if isinstance(obj, Mapping) else obj
            for item in items:
                gid = int(item["gpu_id"])
                w = WorkerState(
                    gpu_id=gid,
                    role=str(item["role"]),
                    expert_alpha=float(item["expert_alpha"]),
                    queue_wait_ms=float(item["queue_wait_ms"]),
                    active_h2d_transfers=int(item["active_h2d_transfers"]),
                    kv_free_bytes=int(item["kv_free_bytes"]),
                    p2p_concurrent_links=int(item["p2p_concurrent_links"]),
                )
                self.worker_states[gid] = w
                self._synced_fields[gid] = self._snap_tuple(w)
            return self.worker_states

    def update_worker_telemetry(
        self,
        gpu_id: int,
        *,
        expert_alpha: float | None = None,
        kv_free_bytes: int | None = None,
        telemetry_seq: int | None = None,
        workers: dict[int, WorkerState] | None = None,
    ) -> None:
        gid = int(gpu_id)
        with self._mu:
            if workers is not None and workers is not self.worker_states:
                w = workers.get(gid)
                if w is None:
                    return
                if expert_alpha is not None:
                    w.expert_alpha = float(expert_alpha)
                if kv_free_bytes is not None:
                    w.kv_free_bytes = max(0, int(kv_free_bytes))
                return
            self._push_python_mutations_locked()
            self._lib.ms_router_update_telemetry(
                ctypes.c_void_p(self._handle),
                ctypes.c_int(gid),
                1 if expert_alpha is not None else 0,
                ctypes.c_double(0.0 if expert_alpha is None else float(expert_alpha)),
                1 if kv_free_bytes is not None else 0,
                ctypes.c_int64(0 if kv_free_bytes is None else int(kv_free_bytes)),
                1 if telemetry_seq is not None else 0,
                ctypes.c_int64(0 if telemetry_seq is None else int(telemetry_seq)),
            )
            self._pull_cpp_workers_locked()

    def select_initial_gpu(
        self,
        *,
        num_gpus: int,
        routing_mode: str,
        preemption_mode: str,
        static_und_ranks: Sequence[int],
        modality_plan: str,
        workers: Mapping[int, WorkerState] | None = None,
        gpu_busy: Mapping[int, bool] | None = None,
        active_dit_ranks: Mapping[int, int] | None = None,
        active_preemptions: Mapping[int, int] | None = None,
        max_preemptions_per_gpu: int = 2,
    ) -> int:
        with self._mu:
            req: dict[str, Any] = {
                "num_gpus": int(num_gpus),
                "routing_mode": str(routing_mode),
                "preemption_mode": str(preemption_mode),
                "static_und_ranks": [int(x) for x in static_und_ranks],
                "modality_plan": str(modality_plan),
                "gpu_busy": {str(k): bool(v) for k, v in (gpu_busy or {}).items()},
                "active_dit_ranks": {str(k): int(v) for k, v in (active_dit_ranks or {}).items()},
                "active_preemptions": {str(k): int(v) for k, v in (active_preemptions or {}).items()},
                "max_preemptions_per_gpu": int(max_preemptions_per_gpu),
            }
            if workers is not None and workers is not self.worker_states:
                req["workers"] = {str(g): w.to_dict() for g, w in workers.items()}
            else:
                self._push_python_mutations_locked()
            chosen = int(
                self._lib.ms_router_select_initial_gpu(ctypes.c_void_p(self._handle), encode_json(req))
            )
            if chosen < 0:
                raise RuntimeError(f"ms_router_select_initial_gpu failed: {last_error(self._lib)}")
            return chosen

    def select_static_gen_gpu(
        self,
        *,
        routing_mode: str,
        static_und_ranks: Sequence[int],
        static_gen_ranks: Sequence[int],
        src_und_gpu: int,
        workers: Mapping[int, WorkerState] | None = None,
        gpu_busy: Mapping[int, bool] | None = None,
        required_kv_bytes: int = 64 * MIB,
    ) -> int:
        with self._mu:
            req: dict[str, Any] = {
                "routing_mode": str(routing_mode),
                "static_und_ranks": [int(x) for x in static_und_ranks],
                "static_gen_ranks": [int(x) for x in static_gen_ranks],
                "src_und_gpu": int(src_und_gpu),
                "gpu_busy": {str(k): bool(v) for k, v in (gpu_busy or {}).items()},
                "required_kv_bytes": int(required_kv_bytes),
            }
            if workers is not None and workers is not self.worker_states:
                req["workers"] = {str(g): w.to_dict() for g, w in workers.items()}
            else:
                self._push_python_mutations_locked()
            chosen = int(
                self._lib.ms_router_select_static_gen_gpu(ctypes.c_void_p(self._handle), encode_json(req))
            )
            if chosen < 0:
                raise RuntimeError(f"ms_router_select_static_gen_gpu failed: {last_error(self._lib)}")
            return chosen

    def select_oom_retry_gpu(
        self,
        *,
        num_gpus: int,
        routing_mode: str,
        static_und_ranks: Sequence[int],
        failed_gpu: int,
        gpu_busy: Mapping[int, bool] | None = None,
    ) -> int:
        with self._mu:
            self._push_python_mutations_locked()
            req = {
                "num_gpus": int(num_gpus),
                "routing_mode": str(routing_mode),
                "static_und_ranks": [int(x) for x in static_und_ranks],
                "failed_gpu": int(failed_gpu),
                "gpu_busy": {str(k): bool(v) for k, v in (gpu_busy or {}).items()},
            }
            chosen = int(
                self._lib.ms_router_select_oom_retry_gpu(ctypes.c_void_p(self._handle), encode_json(req))
            )
            return failed_gpu if chosen < 0 else chosen

    def plan_and_reserve_transition(
        self,
        *,
        routing_mode: str,
        static_und_ranks: Sequence[int],
        static_gen_ranks: Sequence[int],
        req: CPTGRequestSpec,
        est_dit_ms: float,
        workers: dict[int, WorkerState] | None = None,
        gpu_busy: Mapping[int, bool] | None = None,
        use_pacer: bool = True,
        now_ms: float = 0.0,
    ) -> tuple[RouteDecision, str]:
        _ = req.effective_state_bytes()
        with self._mu:
            payload: dict[str, Any] = {
                "routing_mode": str(routing_mode),
                "static_und_ranks": [int(x) for x in static_und_ranks],
                "static_gen_ranks": [int(x) for x in static_gen_ranks],
                "req": req.to_dict(),
                "est_dit_ms": float(est_dit_ms),
                "gpu_busy": {str(k): bool(v) for k, v in (gpu_busy or {}).items()},
                "use_pacer": bool(use_pacer),
                "now_ms": float(now_ms),
            }
            ext_workers = workers is not None and workers is not self.worker_states
            if ext_workers:
                assert workers is not None
                payload["workers"] = {str(g): w.to_dict() for g, w in workers.items()}
            else:
                self._push_python_mutations_locked()

            out = call_json(
                self._lib,
                self._lib.ms_router_plan_and_reserve_json,
                ctypes.c_void_p(self._handle),
                encode_json(payload),
            )
            dec = RouteDecision.from_dict(out["decision"])
            ship_mode = str(out["ship_mode"])
            if ext_workers:
                assert workers is not None
                w_out = out.get("workers", {})
                w_items = w_out.values() if isinstance(w_out, Mapping) else w_out
                for item in w_items:
                    gid = int(item["gpu_id"])
                    if gid in workers:
                        workers[gid].kv_free_bytes = int(item["kv_free_bytes"])
                        workers[gid].queue_wait_ms = float(item["queue_wait_ms"])
                        workers[gid].expert_alpha = float(item["expert_alpha"])
            else:
                self._pull_cpp_workers_locked()
            return dec, ship_mode

    def reserve_kv_bytes(
        self,
        gpu_id: int,
        reserved_kv_bytes: int,
        *,
        workers: dict[int, WorkerState] | None = None,
    ) -> None:
        gid = int(gpu_id)
        res_kv = max(0, int(reserved_kv_bytes))
        if res_kv <= 0:
            return
        with self._mu:
            if workers is not None and workers is not self.worker_states:
                if gid in workers:
                    workers[gid].kv_free_bytes = max(0, int(workers[gid].kv_free_bytes) - res_kv)
                return
            self._push_python_mutations_locked()
            self._lib.ms_router_reserve_kv_bytes(
                ctypes.c_void_p(self._handle), ctypes.c_int(gid), ctypes.c_int64(res_kv)
            )
            self._pull_cpp_workers_locked()

    def release_kv_reservation(
        self,
        gpu_id: int,
        reserved_kv_bytes: int,
        *,
        latest_kv_free_bytes: int | None = None,
        telemetry_seq: int | None = None,
        workers: dict[int, WorkerState] | None = None,
    ) -> None:
        gid = int(gpu_id)
        rel = max(0, int(reserved_kv_bytes))
        with self._mu:
            if workers is not None and workers is not self.worker_states:
                if gid in workers:
                    if latest_kv_free_bytes is not None:
                        workers[gid].kv_free_bytes = max(0, int(latest_kv_free_bytes))
                    else:
                        workers[gid].kv_free_bytes = max(0, int(workers[gid].kv_free_bytes) + rel)
                return
            self._push_python_mutations_locked()
            self._lib.ms_router_release_kv_reservation(
                ctypes.c_void_p(self._handle),
                ctypes.c_int(gid),
                ctypes.c_int64(rel),
                1 if latest_kv_free_bytes is not None else 0,
                ctypes.c_int64(0 if latest_kv_free_bytes is None else int(latest_kv_free_bytes)),
                1 if telemetry_seq is not None else 0,
                ctypes.c_int64(0 if telemetry_seq is None else int(telemetry_seq)),
            )
            self._pull_cpp_workers_locked()

    def release_dit_reservation(
        self,
        gpu_id: int,
        est_dit_ms: float,
        workers: dict[int, WorkerState] | None = None,
    ) -> None:
        gid = int(gpu_id)
        with self._mu:
            if workers is not None and workers is not self.worker_states:
                if gid in workers:
                    workers[gid].queue_wait_ms = max(0.0, float(workers[gid].queue_wait_ms) - max(0.0, float(est_dit_ms)))
                return
            self._push_python_mutations_locked()
            self._lib.ms_router_release_dit_reservation(
                ctypes.c_void_p(self._handle), ctypes.c_int(gid), ctypes.c_double(float(est_dit_ms))
            )
            self._pull_cpp_workers_locked()

    def analytical_crossover_kv_mib(
        self,
        alpha: float = 0.50,
        gen_expert_bytes: int = DEFAULT_OPERATOR_PROFILE.bagel_expert_bytes,
        dit_step1_ms: float = DEFAULT_OPERATOR_PROFILE.bagel_dit_512_step_ms,
        src_gpu: int = 0,
        dst_gpu: int = 1,
    ) -> float:
        return float(
            self._lib.ms_router_crossover_kv_mib(
                ctypes.c_void_p(self._handle),
                ctypes.c_double(float(alpha)),
                ctypes.c_int64(int(gen_expert_bytes)),
                ctypes.c_double(float(dit_step1_ms)),
                ctypes.c_int(int(src_gpu)),
                ctypes.c_int(int(dst_gpu)),
            )
        )

    def decide(
        self,
        req: CPTGRequestSpec,
        workers: Mapping[int, WorkerState] | None = None,
        now_ms: float = 0.0,
    ) -> RouteDecision:
        _ = req.effective_state_bytes()
        with self._mu:
            payload: dict[str, Any] = {"req": req.to_dict(), "now_ms": float(now_ms)}
            if workers is not None and workers is not self.worker_states:
                payload["workers"] = {str(g): w.to_dict() for g, w in workers.items()}
            else:
                self._push_python_mutations_locked()
            out = call_json(
                self._lib,
                self._lib.ms_router_decide_json,
                ctypes.c_void_p(self._handle),
                encode_json(payload),
            )
            return RouteDecision.from_dict(out)

    def route_batch(
        self,
        requests: list[CPTGRequestSpec],
        workers: Mapping[int, WorkerState],
        now_ms: float = 0.0,
    ) -> list[RouteDecision]:
        for r in requests:
            _ = r.effective_state_bytes()
        payload = {
            "requests": [r.to_dict() for r in requests],
            "workers": {str(g): w.to_dict() for g, w in workers.items()},
            "now_ms": float(now_ms),
        }
        arr = call_json(
            self._lib,
            self._lib.ms_router_route_batch_json,
            ctypes.c_void_p(self._handle),
            encode_json(payload),
        )
        return [RouteDecision.from_dict(d) for d in arr]
