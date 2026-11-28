"""Python interface to C++17 `DualPathMorphRouter` (`ms_router_*`) and `HalfNodeH2DPacer` (`ms_pacer_*`)."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Callable, Iterator, Mapping, Sequence

from morphserve._core import call_json, encode_json, get_core_lib, last_error

from .topology import ClusterTopologyProfile, OperatorLatencyProfile, default_operator_profile

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
    gen_expert_bytes: int | None = None
    dit_step1_ms: float | None = None
    dit_total_steps: int = 20
    resolution: int = 512
    interleaved_roundtrip_gamma: int = 1
    model_family: str = ""

    def __post_init__(self) -> None:
        if self.gen_expert_bytes is None or self.dit_step1_ms is None:
            prof = default_operator_profile()
            fam = self.model_family or ("internvl_u" if self.architecture == "bridge_mllm_dit" else "bagel")
            if self.gen_expert_bytes is None:
                self.gen_expert_bytes = prof.expert_bytes(fam)
            if self.dit_step1_ms is None:
                self.dit_step1_ms = prof.step_ms(fam, self.resolution)

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
            "gen_expert_bytes": int(self.gen_expert_bytes or 0),
            "dit_step1_ms": float(self.dit_step1_ms or 0.0),
            "dit_total_steps": int(self.dit_total_steps),
            "resolution": int(self.resolution),
            "interleaved_roundtrip_gamma": int(self.interleaved_roundtrip_gamma),
            "model_family": str(self.model_family),
        }


@dataclass
class WorkerState:
    """Instantaneous HBM residency, queue backlog, and H2D state snapshot of one GPU worker."""

    gpu_id: int
    role: str = "hybrid"  # "und" | "gen" | "hybrid"
    expert_alpha: float = 1.0
    queue_wait_ms: float = 0.0
    active_h2d_transfers: int = 0
    kv_free_bytes: int = 12 * GIB
    p2p_concurrent_links: int = 1
    post_evict: bool = False
    colocated_ar_b: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_id": int(self.gpu_id),
            "role": self.role,
            "expert_alpha": float(self.expert_alpha),
            "queue_wait_ms": float(self.queue_wait_ms),
            "active_h2d_transfers": int(self.active_h2d_transfers),
            "kv_free_bytes": int(self.kv_free_bytes),
            "p2p_concurrent_links": int(self.p2p_concurrent_links),
            "post_evict": bool(self.post_evict),
            "colocated_ar_b": int(self.colocated_ar_b),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WorkerState":
        return cls(
            gpu_id=int(d["gpu_id"]),
            role=str(d.get("role", "hybrid")),
            expert_alpha=float(d["expert_alpha"]),
            queue_wait_ms=float(d["queue_wait_ms"]),
            active_h2d_transfers=int(d["active_h2d_transfers"]),
            kv_free_bytes=int(d["kv_free_bytes"]),
            p2p_concurrent_links=int(d["p2p_concurrent_links"]),
            post_evict=bool(d.get("post_evict", False)),
            colocated_ar_b=int(d.get("colocated_ar_b", 0)),
        )


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
        self.close()

    def close(self) -> None:
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
                ctypes.c_void_p(self._handle),
                ctypes.c_int(int(gpu_id)),
                ctypes.c_double(float(now_ms)),
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
                ctypes.c_void_p(self._handle),
                ctypes.c_int(int(gpu_id)),
                ctypes.byref(out_gid),
            )
        )
        if fd < 0:
            raise OSError(f"ms_pacer_acquire_process_flock({gpu_id}) failed: {last_error(self._lib)}")
        try:
            yield int(out_gid.value)
        finally:
            self._lib.ms_pacer_release_process_flock(ctypes.c_int(fd))


class DualPathMorphRouter:
    """Stateful placement, reservation ledger, and Dual-Path AR->DiT router owned directly by C++17."""

    def __init__(
        self,
        topology: ClusterTopologyProfile,
        pacer: HalfNodeH2DPacer | None = None,
        operator_profile: OperatorLatencyProfile | None = None,
        state_handoff_ms: float | None = None,
        map_overhead_ms_full: float = 12.0,
        overlap_efficiency: float = 0.90,
        allow_cross_half_state_ship: bool = True,
        allow_inter_node_state_ship: bool = True,
        max_inter_node_state_bytes: int = 128 * MIB,
        cost_model_mode: str = "pipeline_analytical",
        performance_atlas: Mapping[str, Any] | None = None,
    ) -> None:
        self.topology = topology
        self.pacer = pacer if pacer is not None else HalfNodeH2DPacer(topology)
        self.operator_profile = operator_profile if operator_profile is not None else default_operator_profile()
        self.state_handoff_ms = state_handoff_ms
        self.map_overhead_ms_full = float(map_overhead_ms_full)
        self.overlap_efficiency = float(overlap_efficiency)
        self.allow_cross_half_state_ship = bool(allow_cross_half_state_ship)
        self.allow_inter_node_state_ship = bool(allow_inter_node_state_ship)
        self.max_inter_node_state_bytes = int(max_inter_node_state_bytes)
        self.cost_model_mode = str(cost_model_mode)

        self._mu = threading.RLock()
        self._lib = get_core_lib()

        opts: dict[str, Any] = {
            "operator_profile": self.operator_profile.to_dict(),
            "state_handoff_ms": None if state_handoff_ms is None else float(state_handoff_ms),
            "map_overhead_ms_full": self.map_overhead_ms_full,
            "overlap_efficiency": self.overlap_efficiency,
            "allow_cross_half_state_ship": self.allow_cross_half_state_ship,
            "allow_inter_node_state_ship": self.allow_inter_node_state_ship,
            "max_inter_node_state_bytes": self.max_inter_node_state_bytes,
            "cost_model_mode": self.cost_model_mode,
        }
        if performance_atlas is not None:
            opts["performance_atlas"] = dict(performance_atlas)
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
        self.close()

    def close(self) -> None:
        h = getattr(self, "_handle", 0)
        if h:
            self._handle = 0
            try:
                self._lib.ms_router_destroy(ctypes.c_void_p(h))
            except Exception:
                pass

    @staticmethod
    def _parse_workers_obj(obj: Any) -> dict[int, WorkerState]:
        items = obj.values() if isinstance(obj, Mapping) else obj
        return {int(item["gpu_id"]): WorkerState.from_dict(item) for item in items}

    def worker_states_snapshot(self) -> dict[int, WorkerState]:
        """Return a fresh snapshot of the C++ router's authoritative worker state ledger."""
        with self._mu:
            obj = call_json(self._lib, self._lib.ms_router_get_workers_json, ctypes.c_void_p(self._handle))
            return self._parse_workers_obj(obj)

    @property
    def worker_states(self) -> dict[int, WorkerState]:
        """Read-only snapshot view of the C++ worker state ledger."""
        return self.worker_states_snapshot()

    def get_worker_state(self, gpu_id: int) -> WorkerState:
        """Return the current authoritative `WorkerState` for `gpu_id` from C++."""
        gid = int(gpu_id)
        snap = self.worker_states_snapshot()
        if gid not in snap:
            raise KeyError(f"gpu_id={gid} not initialized in DualPathMorphRouter")
        return snap[gid]

    def set_worker_state_fields(
        self,
        gpu_id: int,
        *,
        expert_alpha: float | None = None,
        queue_wait_ms: float | None = None,
        active_h2d_transfers: int | None = None,
        kv_free_bytes: int | None = None,
        p2p_concurrent_links: int | None = None,
    ) -> None:
        """Explicit write-through mutation of worker state fields directly in C++."""
        with self._mu:
            self._lib.ms_router_set_worker_fields(
                ctypes.c_void_p(self._handle),
                ctypes.c_int(int(gpu_id)),
                1 if expert_alpha is not None else 0,
                ctypes.c_double(0.0 if expert_alpha is None else float(expert_alpha)),
                1 if queue_wait_ms is not None else 0,
                ctypes.c_double(0.0 if queue_wait_ms is None else float(queue_wait_ms)),
                1 if active_h2d_transfers is not None else 0,
                ctypes.c_int(0 if active_h2d_transfers is None else int(active_h2d_transfers)),
                1 if kv_free_bytes is not None else 0,
                ctypes.c_int64(0 if kv_free_bytes is None else int(kv_free_bytes)),
                1 if p2p_concurrent_links is not None else 0,
                ctypes.c_int(0 if p2p_concurrent_links is None else int(p2p_concurrent_links)),
            )

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
            return self._parse_workers_obj(obj)

    def update_worker_telemetry(
        self,
        gpu_id: int,
        *,
        expert_alpha: float | None = None,
        kv_free_bytes: int | None = None,
        telemetry_seq: int | None = None,
    ) -> None:
        """Apply worker telemetry directly to C++ with monotonic `telemetry_seq` and inflight KV reconciliation."""
        with self._mu:
            self._lib.ms_router_update_telemetry(
                ctypes.c_void_p(self._handle),
                ctypes.c_int(int(gpu_id)),
                1 if expert_alpha is not None else 0,
                ctypes.c_double(0.0 if expert_alpha is None else float(expert_alpha)),
                1 if kv_free_bytes is not None else 0,
                ctypes.c_int64(0 if kv_free_bytes is None else int(kv_free_bytes)),
                1 if telemetry_seq is not None else 0,
                ctypes.c_int64(0 if telemetry_seq is None else int(telemetry_seq)),
            )

    def reserve_ar_queue(self, gpu_id: int, est_ar_ms: float) -> None:
        """Reserve estimated AR/prep queue latency on `gpu_id` directly in C++."""
        with self._mu:
            self._lib.ms_router_reserve_ar_queue(
                ctypes.c_void_p(self._handle),
                ctypes.c_int(int(gpu_id)),
                ctypes.c_double(float(est_ar_ms)),
            )

    def release_ar_queue(self, gpu_id: int, est_ar_ms: float) -> None:
        """Release estimated AR/prep queue latency on `gpu_id` directly in C++."""
        with self._mu:
            self._lib.ms_router_release_ar_queue(
                ctypes.c_void_p(self._handle),
                ctypes.c_int(int(gpu_id)),
                ctypes.c_double(float(est_ar_ms)),
            )

    def set_colocated_ar_count(self, gpu_id: int, count: int) -> None:
        """Update active mid-DiT colocated AR count (3rd Performance Atlas axis) on `gpu_id`."""
        with self._mu:
            self._lib.ms_router_set_colocated_ar(
                ctypes.c_void_p(self._handle),
                ctypes.c_int(int(gpu_id)),
                ctypes.c_int(max(0, int(count))),
            )

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
        max_preemptions_per_gpu: int = 4,
        architecture: str = "tight_mot",
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
                "architecture": str(architecture),
            }
            if workers is not None:
                req["workers"] = {str(g): w.to_dict() for g, w in workers.items()}
            chosen = int(self._lib.ms_router_select_initial_gpu(ctypes.c_void_p(self._handle), encode_json(req)))
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
            if workers is not None:
                req["workers"] = {str(g): w.to_dict() for g, w in workers.items()}
            chosen = int(self._lib.ms_router_select_static_gen_gpu(ctypes.c_void_p(self._handle), encode_json(req)))
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
        workers: Mapping[int, WorkerState] | None = None,
        gpu_busy: Mapping[int, bool] | None = None,
    ) -> int:
        with self._mu:
            req: dict[str, Any] = {
                "num_gpus": int(num_gpus),
                "routing_mode": str(routing_mode),
                "static_und_ranks": [int(x) for x in static_und_ranks],
                "failed_gpu": int(failed_gpu),
                "gpu_busy": {str(k): bool(v) for k, v in (gpu_busy or {}).items()},
            }
            if workers is not None:
                req["workers"] = {str(g): w.to_dict() for g, w in workers.items()}
            chosen = int(self._lib.ms_router_select_oom_retry_gpu(ctypes.c_void_p(self._handle), encode_json(req)))
            return int(failed_gpu) if chosen < 0 else chosen

    def plan_transition_readonly(
        self,
        *,
        routing_mode: str,
        static_und_ranks: Sequence[int],
        static_gen_ranks: Sequence[int],
        req: CPTGRequestSpec,
        est_dit_ms: float = 0.0,
        workers: Mapping[int, WorkerState] | None = None,
        gpu_busy: Mapping[int, bool] | None = None,
        now_ms: float = 0.0,
    ) -> tuple[RouteDecision, str]:
        """Pure read-only transition decision with zero pacer or worker-state mutation."""
        _ = req.effective_state_bytes()
        with self._mu:
            payload: dict[str, Any] = {
                "routing_mode": str(routing_mode),
                "static_und_ranks": [int(x) for x in static_und_ranks],
                "static_gen_ranks": [int(x) for x in static_gen_ranks],
                "req": req.to_dict(),
                "est_dit_ms": float(est_dit_ms),
                "gpu_busy": {str(k): bool(v) for k, v in (gpu_busy or {}).items()},
                "now_ms": float(now_ms),
            }
            if workers is not None:
                payload["workers"] = {str(g): w.to_dict() for g, w in workers.items()}
            out = call_json(
                self._lib,
                self._lib.ms_router_plan_transition_readonly_json,
                ctypes.c_void_p(self._handle),
                encode_json(payload),
            )
            return RouteDecision.from_dict(out["decision"]), str(out["ship_mode"])

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
            if workers is not None:
                payload["workers"] = {str(g): w.to_dict() for g, w in workers.items()}
            out = call_json(
                self._lib,
                self._lib.ms_router_plan_and_reserve_json,
                ctypes.c_void_p(self._handle),
                encode_json(payload),
            )
            dec = RouteDecision.from_dict(out["decision"])
            ship_mode = str(out["ship_mode"])
            if workers is not None and "workers" in out:
                w_out = out["workers"]
                w_items = w_out.values() if isinstance(w_out, Mapping) else w_out
                for item in w_items:
                    gid = int(item["gpu_id"])
                    if gid in workers:
                        workers[gid].kv_free_bytes = int(item["kv_free_bytes"])
                        workers[gid].queue_wait_ms = float(item["queue_wait_ms"])
                        workers[gid].expert_alpha = float(item["expert_alpha"])
            return dec, ship_mode

    def reserve_kv_bytes(self, gpu_id: int, reserved_kv_bytes: int) -> None:
        res_kv = max(0, int(reserved_kv_bytes))
        if res_kv <= 0:
            return
        with self._mu:
            self._lib.ms_router_reserve_kv_bytes(
                ctypes.c_void_p(self._handle),
                ctypes.c_int(int(gpu_id)),
                ctypes.c_int64(res_kv),
            )

    def release_kv_reservation(
        self,
        gpu_id: int,
        reserved_kv_bytes: int,
        *,
        latest_kv_free_bytes: int | None = None,
        telemetry_seq: int | None = None,
    ) -> None:
        rel = max(0, int(reserved_kv_bytes))
        with self._mu:
            self._lib.ms_router_release_kv_reservation(
                ctypes.c_void_p(self._handle),
                ctypes.c_int(int(gpu_id)),
                ctypes.c_int64(rel),
                1 if latest_kv_free_bytes is not None else 0,
                ctypes.c_int64(0 if latest_kv_free_bytes is None else int(latest_kv_free_bytes)),
                1 if telemetry_seq is not None else 0,
                ctypes.c_int64(0 if telemetry_seq is None else int(telemetry_seq)),
            )

    def release_dit_reservation(self, gpu_id: int, est_dit_ms: float) -> None:
        with self._mu:
            self._lib.ms_router_release_dit_reservation(
                ctypes.c_void_p(self._handle),
                ctypes.c_int(int(gpu_id)),
                ctypes.c_double(float(est_dit_ms)),
            )

    def analytical_crossover_kv_mib(
        self,
        alpha: float = 0.50,
        gen_expert_bytes: int = 0,
        dit_step1_ms: float = 0.0,
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
            if workers is not None:
                payload["workers"] = {str(g): w.to_dict() for g, w in workers.items()}
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
        workers: Mapping[int, WorkerState] | None = None,
        now_ms: float = 0.0,
    ) -> list[RouteDecision]:
        for r in requests:
            _ = r.effective_state_bytes()
        eff_workers = self.worker_states_snapshot() if workers is None else workers
        payload = {
            "requests": [r.to_dict() for r in requests],
            "workers": {str(g): w.to_dict() for g, w in eff_workers.items()},
            "now_ms": float(now_ms),
        }
        arr = call_json(
            self._lib,
            self._lib.ms_router_route_batch_json,
            ctypes.c_void_p(self._handle),
            encode_json(payload),
        )
        return [RouteDecision.from_dict(d) for d in arr]
