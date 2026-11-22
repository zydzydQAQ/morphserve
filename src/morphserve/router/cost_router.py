"""Stateful Three-Tier PCIe/Socket/RDMA Dual-Path Morphing Router and Half-Node H2D Pacer."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import threading
from typing import Any, Callable, Iterator, Mapping, Sequence

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


class HalfNodeH2DPacer:
    """Serializes heavy expert H2D DMA per PCIe root half-node to avoid uplink collapse."""

    def __init__(
        self,
        topology: ClusterTopologyProfile,
        max_concurrent_per_half: int = 1,
        lock_dir: str | Path = "/tmp/morphserve_h2d_locks",
    ) -> None:
        self.topology = topology
        self.max_concurrent_per_half = max(1, int(max_concurrent_per_half))
        self.lock_dir = Path(lock_dir)
        self._mu = threading.Lock()
        self._busy_until_ms: dict[int, list[float]] = {
            grp.group_id: [0.0] * self.max_concurrent_per_half for grp in topology.half_groups
        }
        self._semaphores: dict[int, threading.Semaphore] = {
            grp.group_id: threading.Semaphore(self.max_concurrent_per_half) for grp in topology.half_groups
        }

    def predicted_wait_ms(self, gpu_id: int, now_ms: float = 0.0) -> float:
        gid = self.topology.half_group_of(gpu_id).group_id
        with self._mu:
            slots = self._busy_until_ms[gid]
            earliest = min(slots)
            return max(0.0, earliest - now_ms)

    def reserve_slot(self, gpu_id: int, duration_ms: float, now_ms: float = 0.0) -> float:
        """Reserve an analytical half-node H2D DMA slot; returns wait ms before DMA starts."""
        gid = self.topology.half_group_of(gpu_id).group_id
        with self._mu:
            slots = self._busy_until_ms[gid]
            idx = min(range(len(slots)), key=lambda i: slots[i])
            start_ms = max(now_ms, slots[idx])
            slots[idx] = start_ms + max(0.0, duration_ms)
            return start_ms - now_ms

    @contextmanager
    def acquire_thread_slot(self, gpu_id: int) -> Iterator[int]:
        gid = self.topology.half_group_of(gpu_id).group_id
        sem = self._semaphores[gid]
        sem.acquire()
        try:
            yield gid
        finally:
            sem.release()

    @contextmanager
    def acquire_process_slot(self, gpu_id: int) -> Iterator[int]:
        """Cross-process flock pacing per PCIe half-node (safe across `mp.spawn` workers)."""
        import fcntl

        gid = self.topology.half_group_of(gpu_id).group_id
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file = self.lock_dir / f"pcie_half_{gid}.lock"
        fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield gid
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


class DualPathMorphRouter:
    """Stateful placement, queue reservation, and Dual-Path AR->DiT router.

    Consolidates initial GPU admission, static-fission rank pairing, atomic pre-ship `queue_wait_ms`
    and `HalfNodeH2DPacer` reservations, and three-tier analytical Path-A (`state_ship`) vs.
    Path-B (`in_place_morph`) cost selection across live `MorphServeClusterEngine` and `cluster_simulator.py`.
    """

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
        self.rr_cursor: int = 0
        self._mu = threading.Lock()

    def init_workers(
        self,
        num_gpus: int,
        role_of: Callable[[int], str],
        initial_alphas: Mapping[int, float],
        kv_free_bytes: int = 16 * GIB,
    ) -> dict[int, WorkerState]:
        """Initialize the router's authoritative `WorkerState` ledger."""
        self.worker_states = {
            g: WorkerState(
                gpu_id=g,
                role=role_of(g),
                expert_alpha=float(initial_alphas.get(g, 1.0)),
                queue_wait_ms=0.0,
                kv_free_bytes=int(kv_free_bytes),
            )
            for g in range(int(num_gpus))
        }
        self.rr_cursor = 0
        return self.worker_states

    # --- Stateful Placement & Reservation Authority (replaces shallow ClusterPolicyCore) ---

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
        """Select initial GPU rank while preserving `e27bce5` queue, preemption-cap, and alpha invariants."""
        w_map = workers if workers is not None else self.worker_states
        busy = gpu_busy or {}
        dit_map = active_dit_ranks or {}
        pre_map = active_preemptions or {}
        n = int(num_gpus)

        with self._mu:
            cur = self.rr_cursor
            if routing_mode in ("twostage_shm", "static_nccl_fission") and static_und_ranks:
                cands = list(static_und_ranks)
                cands.sort(
                    key=lambda g: (
                        w_map[g].queue_wait_ms,
                        bool(busy.get(g, False)),
                        (g - cur) % n,
                    )
                )
                chosen = cands[0]
                self.rr_cursor = (chosen + 1) % n
                return chosen

            if modality_plan == "ar_only":
                idle = [
                    g for g in range(n) if (not bool(busy.get(g, False))) and w_map[g].queue_wait_ms <= 1.0
                ]
                if idle:
                    idle.sort(key=lambda g: (w_map[g].expert_alpha, (g - cur) % n))
                    chosen = idle[0]
                    self.rr_cursor = (chosen + 1) % n
                    return chosen
                if preemption_mode != "non_preemptive":
                    elig = [g for g in range(n) if pre_map.get(g, 0) < max_preemptions_per_gpu]
                    if elig:
                        elig.sort(
                            key=lambda g: (
                                pre_map.get(g, 0),
                                0 if dit_map.get(g, 0) > 0 else 1,
                                w_map[g].expert_alpha,
                                (g - cur) % n,
                            )
                        )
                        chosen = elig[0]
                        self.rr_cursor = (chosen + 1) % n
                        return chosen

            if modality_plan == "cot_then_dit":
                idle = [
                    g for g in range(n) if (not bool(busy.get(g, False))) and w_map[g].queue_wait_ms <= 1.0
                ]
                if idle:
                    idle.sort(key=lambda g: ((g - cur) % n, w_map[g].expert_alpha))
                    chosen = idle[0]
                    self.rr_cursor = (chosen + 1) % n
                    return chosen
                if preemption_mode != "non_preemptive":
                    elig = [g for g in range(n) if pre_map.get(g, 0) < max_preemptions_per_gpu]
                    if elig:
                        elig.sort(
                            key=lambda g: (
                                pre_map.get(g, 0),
                                w_map[g].queue_wait_ms,
                                (g - cur) % n,
                            )
                        )
                        chosen = elig[0]
                        self.rr_cursor = (chosen + 1) % n
                        return chosen

            cands = list(range(n))
            cands.sort(
                key=lambda g: (
                    w_map[g].queue_wait_ms,
                    bool(busy.get(g, False)),
                    -w_map[g].expert_alpha if routing_mode == "dual_path" else 0.0,
                    (g - cur) % n,
                )
            )
            chosen = cands[0]
            self.rr_cursor = (chosen + 1) % n
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
    ) -> int:
        """Select destination generation rank for `twostage_shm` or `static_nccl_fission`."""
        w_map = workers if workers is not None else self.worker_states
        busy = gpu_busy or {}
        if routing_mode == "twostage_shm":
            idx = list(static_und_ranks).index(src_und_gpu)
            return static_gen_ranks[idx % len(static_gen_ranks)]

        tier_order = {"intra_half": 0, "inter_socket": 1, "inter_node": 2}
        cands = list(static_gen_ranks)
        cands.sort(
            key=lambda g: (
                tier_order.get(self.topology.fabric_tier(src_und_gpu, g), 3),
                w_map[g].queue_wait_ms,
                bool(busy.get(g, False)),
                g,
            )
        )
        return cands[0]

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
        """Atomically route AR/Prefill -> DiT AND reserve `queue_wait_ms` + `HalfNodeH2DPacer` before async shipping."""
        w_map = workers if workers is not None else self.worker_states
        with self._mu:
            if routing_mode == "dual_path":
                dec = self.decide(req, w_map, now_ms=now_ms)
                dst_gpu = dec.dst_gpu
                ship_mode = "nccl_dual" if (dec.path == "state_ship" and dst_gpu != req.src_gpu) else "none"
                if dec.path == "in_place_morph" and use_pacer:
                    alpha = w_map[dst_gpu].expert_alpha
                    missing = (1.0 - alpha) * float(req.gen_expert_bytes)
                    if missing > 0.0:
                        bw = max(1.0, dec.selected_bw_gbps)
                        dma_ms = (missing / 1e9) / bw * 1000.0
                        self.pacer.reserve_slot(dst_gpu, dma_ms, now_ms=now_ms + w_map[dst_gpu].queue_wait_ms)
            elif routing_mode in ("static_nccl_fission", "twostage_shm"):
                dst_gpu = self.select_static_gen_gpu(
                    routing_mode=routing_mode,
                    static_und_ranks=static_und_ranks,
                    static_gen_ranks=static_gen_ranks,
                    src_und_gpu=req.src_gpu,
                    workers=w_map,
                    gpu_busy=gpu_busy,
                )
                ship_mode = "nccl_static" if routing_mode == "static_nccl_fission" else "shm_static"
                bw = self.topology.effective_p2p_gbps(req.src_gpu, dst_gpu)
                t_ship = (req.effective_state_bytes() / 1e9) / max(0.5, bw) * 1000.0 + self.topology.handoff_ms(
                    req.src_gpu, dst_gpu
                )
                dec = RouteDecision(
                    req_id=req.req_id,
                    path="state_ship",
                    src_gpu=req.src_gpu,
                    dst_gpu=dst_gpu,
                    t_ship_ms=round(t_ship, 3),
                    t_morph_ms=float("inf"),
                    chosen_cost_ms=round(t_ship, 3),
                    selected_bw_gbps=round(bw, 3),
                    effective_state_mib=round(req.effective_state_bytes() / MIB, 2),
                    reason=f"static {routing_mode} {req.src_gpu}->{dst_gpu}",
                    hop_tier=self.topology.fabric_tier(req.src_gpu, dst_gpu),
                )
            else:
                dst_gpu = req.src_gpu
                ship_mode = "none"
                dec = RouteDecision(
                    req_id=req.req_id,
                    path="in_place_morph",
                    src_gpu=req.src_gpu,
                    dst_gpu=dst_gpu,
                    t_ship_ms=-1.0,
                    t_morph_ms=round(w_map[dst_gpu].queue_wait_ms, 3),
                    chosen_cost_ms=round(w_map[dst_gpu].queue_wait_ms, 3),
                    selected_bw_gbps=self.topology.solo_h2d_gbps(dst_gpu),
                    effective_state_mib=round(req.effective_state_bytes() / MIB, 2),
                    reason=f"sticky {routing_mode} on gpu {dst_gpu}",
                    hop_tier="in_place",
                )

            w_map[dst_gpu].queue_wait_ms += max(0.0, float(est_dit_ms))
            return dec, ship_mode

    def release_dit_reservation(
        self,
        gpu_id: int,
        est_dit_ms: float,
        workers: dict[int, WorkerState] | None = None,
    ) -> None:
        w_map = workers if workers is not None else self.worker_states
        with self._mu:
            w_map[gpu_id].queue_wait_ms = max(0.0, w_map[gpu_id].queue_wait_ms - max(0.0, float(est_dit_ms)))

    # --- Analytical Dual-Path Cost Model ---

    def analytical_crossover_kv_mib(
        self,
        alpha: float = 0.50,
        gen_expert_bytes: int = DEFAULT_OPERATOR_PROFILE.bagel_expert_bytes,
        dit_step1_ms: float = DEFAULT_OPERATOR_PROFILE.bagel_dit_512_step_ms,
        src_gpu: int = 0,
        dst_gpu: int = 1,
    ) -> float:
        """Closed-form intra-half KV crossover size (MiB) where Path A `t_ship` equals Path B `t_morph`."""
        w_src = WorkerState(gpu_id=src_gpu, role="hybrid", expert_alpha=alpha, queue_wait_ms=0.0)
        req_probe = CPTGRequestSpec(
            req_id="crossover_probe",
            src_gpu=src_gpu,
            architecture="tight_mot",
            kv_bytes=0,
            gen_expert_bytes=gen_expert_bytes,
            dit_step1_ms=dit_step1_ms,
        )
        t_morph_ms, _ = self.evaluate_path_b_morph_ms(req_probe, w_src, now_ms=0.0)
        handoff = (
            float(self.state_handoff_ms)
            if self.state_handoff_ms is not None
            else self.topology.handoff_ms(src_gpu, dst_gpu)
        )
        net_budget_ms = max(0.0, t_morph_ms - handoff)
        bw_p2p = self.topology.effective_p2p_gbps(src_gpu, dst_gpu)
        kv_bytes = (net_budget_ms / 1000.0) * bw_p2p * 1e9
        return round(kv_bytes / MIB, 1)

    def evaluate_path_b_morph_ms(
        self,
        req: CPTGRequestSpec,
        src_worker: WorkerState,
        now_ms: float = 0.0,
    ) -> tuple[float, float]:
        """Return `(t_morph_ms, effective_h2d_gbps)` for in-place DiT morph on `src_worker`."""
        alpha = max(0.0, min(1.0, float(src_worker.expert_alpha)))
        missing_bytes = (1.0 - alpha) * float(req.gen_expert_bytes)
        if missing_bytes <= 0.0:
            return float(src_worker.queue_wait_ms), self.topology.solo_h2d_gbps(src_worker.gpu_id)

        h2d_wait_ms = self.pacer.predicted_wait_ms(src_worker.gpu_id, now_ms=now_ms)
        active = 1 if self.pacer.max_concurrent_per_half == 1 else max(1, src_worker.active_h2d_transfers + 1)
        bw_h2d_gbps = self.topology.effective_h2d_gbps(src_worker.gpu_id, active_h2d_same_half=active)
        h2d_xfer_ms = (missing_bytes / 1e9) / bw_h2d_gbps * 1000.0
        map_ms = self.map_overhead_ms_full * (1.0 - alpha)
        resident_compute_ms = self.overlap_efficiency * alpha * float(req.dit_step1_ms)
        exposed_stall_ms = max(0.0, h2d_xfer_ms - resident_compute_ms)
        t_morph = float(src_worker.queue_wait_ms) + h2d_wait_ms + map_ms + exposed_stall_ms
        return round(t_morph, 3), round(bw_h2d_gbps, 3)

    def evaluate_path_a_ship_ms(
        self,
        req: CPTGRequestSpec,
        src_worker: WorkerState,
        dst_worker: WorkerState,
        now_ms: float = 0.0,
    ) -> tuple[float, float]:
        """Return `(t_ship_ms, effective_p2p_gbps)` for shipping CPTG state from `src` to `dst`."""
        state_bytes = float(req.effective_state_bytes())
        if dst_worker.gpu_id == src_worker.gpu_id:
            return float("inf"), 0.0
        if dst_worker.kv_free_bytes < state_bytes:
            return float("inf"), 0.0

        tier = self.topology.fabric_tier(src_worker.gpu_id, dst_worker.gpu_id)
        if tier == "inter_socket" and not self.allow_cross_half_state_ship:
            return float("inf"), 0.0
        if tier == "inter_node":
            if not self.allow_inter_node_state_ship:
                return float("inf"), 0.0
            if state_bytes > float(self.max_inter_node_state_bytes):
                return float("inf"), 0.0

        concurrent = max(1, min(src_worker.p2p_concurrent_links, dst_worker.p2p_concurrent_links))
        bw_p2p_gbps = self.topology.effective_p2p_gbps(
            src_worker.gpu_id,
            dst_worker.gpu_id,
            concurrent_intra_half=concurrent if tier == "intra_half" else 1,
            concurrent_cross_half=concurrent if tier == "inter_socket" else 1,
            concurrent_inter_node=concurrent if tier == "inter_node" else 1,
        )
        gamma = max(1, int(req.interleaved_roundtrip_gamma))
        handoff = (
            float(self.state_handoff_ms)
            if self.state_handoff_ms is not None
            else self.topology.handoff_ms(src_worker.gpu_id, dst_worker.gpu_id)
        )
        ship_xfer_ms = (gamma * state_bytes / 1e9) / bw_p2p_gbps * 1000.0 + handoff

        dst_morph_ms = 0.0
        if dst_worker.expert_alpha < 1.0:
            dst_morph_only, _ = self.evaluate_path_b_morph_ms(
                req,
                WorkerState(
                    gpu_id=dst_worker.gpu_id,
                    role=dst_worker.role,
                    expert_alpha=dst_worker.expert_alpha,
                    queue_wait_ms=0.0,
                    active_h2d_transfers=dst_worker.active_h2d_transfers,
                ),
                now_ms=now_ms,
            )
            dst_morph_ms = dst_morph_only

        t_ship = ship_xfer_ms + float(dst_worker.queue_wait_ms) + dst_morph_ms
        return round(t_ship, 3), round(bw_p2p_gbps, 3)

    def decide(
        self,
        req: CPTGRequestSpec,
        workers: Mapping[int, WorkerState] | None = None,
        now_ms: float = 0.0,
    ) -> RouteDecision:
        """Select optimal AR->DiT execution path via 3-tier topology + residency cost model."""
        w_map = workers if workers is not None else self.worker_states
        src_w = w_map[req.src_gpu]
        t_morph_ms, h2d_bw = self.evaluate_path_b_morph_ms(req, src_w, now_ms=now_ms)

        best_dst = req.src_gpu
        best_ship_ms = float("inf")
        best_p2p_bw = 0.0

        for cand_id in self.topology.peer_candidates(req.src_gpu, prefer_same_half=True):
            if cand_id not in w_map:
                continue
            cand_w = w_map[cand_id]
            t_ship, bw_p2p = self.evaluate_path_a_ship_ms(req, src_w, cand_w, now_ms=now_ms)
            if t_ship < best_ship_ms:
                best_ship_ms = t_ship
                best_dst = cand_id
                best_p2p_bw = bw_p2p

        state_mib = round(req.effective_state_bytes() / MIB, 2)
        if best_ship_ms < t_morph_ms:
            tier = self.topology.fabric_tier(req.src_gpu, best_dst)
            hop_label = "same_half" if tier == "intra_half" else tier
            return RouteDecision(
                req_id=req.req_id,
                path="state_ship",
                src_gpu=req.src_gpu,
                dst_gpu=best_dst,
                t_ship_ms=best_ship_ms,
                t_morph_ms=t_morph_ms,
                chosen_cost_ms=best_ship_ms,
                selected_bw_gbps=best_p2p_bw,
                effective_state_mib=state_mib,
                reason=f"{req.architecture}: {hop_label} state_ship({state_mib}MiB) {best_ship_ms:.1f}ms < morph {t_morph_ms:.1f}ms",
                hop_tier=tier,
            )

        return RouteDecision(
            req_id=req.req_id,
            path="in_place_morph",
            src_gpu=req.src_gpu,
            dst_gpu=req.src_gpu,
            t_ship_ms=best_ship_ms if best_ship_ms < 1e12 else -1.0,
            t_morph_ms=t_morph_ms,
            chosen_cost_ms=t_morph_ms,
            selected_bw_gbps=h2d_bw,
            effective_state_mib=state_mib,
            reason=f"{req.architecture}: in_place_morph(alpha={src_w.expert_alpha:.2f}) {t_morph_ms:.1f}ms <= ship {best_ship_ms:.1f}ms",
            hop_tier="in_place",
        )

    def route_batch(
        self,
        requests: list[CPTGRequestSpec],
        workers: Mapping[int, WorkerState],
        now_ms: float = 0.0,
    ) -> list[RouteDecision]:
        """Greedy online routing for a burst of CPTG transitions, updating queue and H2D reservations."""
        state_copy: dict[int, WorkerState] = {
            gid: WorkerState(**vars(w)) for gid, w in workers.items()
        }
        decisions: list[RouteDecision] = []
        dit_dur_ms = lambda r: float(r.dit_step1_ms * max(1, r.dit_total_steps) * 0.85)
        for req in requests:
            dec = self.decide(req, state_copy, now_ms=now_ms)
            decisions.append(dec)
            dst_w = state_copy[dec.dst_gpu]
            if dec.path == "in_place_morph":
                alpha = dst_w.expert_alpha
                missing = (1.0 - alpha) * float(req.gen_expert_bytes)
                if missing > 0:
                    bw = max(1.0, dec.selected_bw_gbps)
                    dma_ms = (missing / 1e9) / bw * 1000.0
                    self.pacer.reserve_slot(dec.dst_gpu, dma_ms, now_ms=now_ms + dst_w.queue_wait_ms)
                dst_w.expert_alpha = 1.0
            else:
                dst_w.kv_free_bytes = max(0, dst_w.kv_free_bytes - req.effective_state_bytes())
            dst_w.queue_wait_ms += dit_dur_ms(req)
        return decisions
