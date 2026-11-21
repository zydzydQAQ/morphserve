"""PCIe/NUMA topology-aware Dual-Path Morphing Router and Half-Node H2D Pacer."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
import threading
from typing import Any, Iterator

from .topology import ClusterTopologyProfile

MIB = 1024**2
GIB = 1024**3


@dataclass
class CPTGRequestSpec:
    """Unified Cross-Paradigm Transition Graph (CPTG) request specification.

    Supports both tightly-coupled Mixture-of-Transformers models (`architecture="tight_mot"`,
    e.g., BAGEL-7B-MoT / Show-o2 where cross-paradigm state is multi-turn KV cache) and
    bridge-coupled models (`architecture="bridge_mllm_dit"`, e.g., InternVL-U / Ming-Lite-Uni
    where only compact conditioning hidden states cross to the MMDiT head).
    """

    req_id: str
    src_gpu: int
    architecture: str = "tight_mot"
    kv_bytes: int = 0
    bridge_bytes: int = 0
    gen_expert_bytes: int = 12_446 * MIB
    dit_step1_ms: float = 287.0
    dit_total_steps: int = 20
    resolution: int = 512
    interleaved_roundtrip_gamma: int = 1

    def effective_state_bytes(self) -> int:
        if self.architecture == "bridge_mllm_dit":
            return int(self.bridge_bytes)
        if self.architecture == "tight_mot":
            return int(self.kv_bytes)
        raise ValueError(f"unsupported CPTG architecture: {self.architecture}")


@dataclass
class WorkerState:
    """Instantaneous HBM residency, queue backlog, and H2D state of one GPU worker."""

    gpu_id: int
    role: str = "hybrid"  # "und" | "gen" | "hybrid"
    expert_alpha: float = 0.50
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


class HalfNodeH2DPacer:
    """Serializes heavy expert H2D DMA per PCIe root half-node to avoid uplink collapse.

    Job 0c proved 4-GPU simultaneous intra-half H2D collapses per-GPU bandwidth from
    ~8.75 GB/s to 2.58 GB/s (10.35 GB/s shared root limit). Pacing to 1 heavy DMA per
    half-node preserves full ~8.7 GB/s per transfer and enables cross-half parallelism
    across NUMA sockets (~17.4 GB/s aggregate).
    """

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
    """Topology- and cost-aware AR->DiT router selecting between State-Shipping (Path A)
    and TriPool-VMM Step-1 In-Place Morphing (Path B).
    """

    def __init__(
        self,
        topology: ClusterTopologyProfile,
        pacer: HalfNodeH2DPacer | None = None,
        state_handoff_ms: float = 3.0,
        map_overhead_ms_full: float = 12.0,
        overlap_efficiency: float = 0.90,
        allow_cross_half_state_ship: bool = True,
    ) -> None:
        self.topology = topology
        self.pacer = pacer if pacer is not None else HalfNodeH2DPacer(topology)
        self.state_handoff_ms = float(state_handoff_ms)
        self.map_overhead_ms_full = float(map_overhead_ms_full)
        self.overlap_efficiency = float(overlap_efficiency)
        self.allow_cross_half_state_ship = bool(allow_cross_half_state_ship)

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

        # Under H2D pacing, this transfer waits for any busy slot on its half-node,
        # then executes at full solo H2D bandwidth instead of contended ~2.58 GB/s!
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
    ) -> tuple[float, float]:
        """Return `(t_ship_ms, effective_p2p_gbps)` for shipping CPTG state from `src` to `dst`."""
        state_bytes = float(req.effective_state_bytes())
        if dst_worker.gpu_id == src_worker.gpu_id:
            return float("inf"), 0.0
        if dst_worker.kv_free_bytes < state_bytes:
            return float("inf"), 0.0
        same = self.topology.same_half(src_worker.gpu_id, dst_worker.gpu_id)
        if not self.allow_cross_half_state_ship and not same:
            return float("inf"), 0.0

        concurrent = max(1, min(src_worker.p2p_concurrent_links, dst_worker.p2p_concurrent_links))
        bw_p2p_gbps = self.topology.effective_p2p_gbps(
            src_worker.gpu_id,
            dst_worker.gpu_id,
            concurrent_intra_half=concurrent if same else 1,
            concurrent_cross_half=concurrent if not same else 1,
        )
        gamma = max(1, int(req.interleaved_roundtrip_gamma))
        ship_xfer_ms = (gamma * state_bytes / 1e9) / bw_p2p_gbps * 1000.0 + self.state_handoff_ms

        # If destination peer is missing some generation expert weights, include its incremental morph stall!
        dst_morph_ms = 0.0
        if dst_worker.expert_alpha < 1.0:
            dst_morph_only, _ = self.evaluate_path_b_morph_ms(req, WorkerState(
                gpu_id=dst_worker.gpu_id,
                role=dst_worker.role,
                expert_alpha=dst_worker.expert_alpha,
                queue_wait_ms=0.0,
                active_h2d_transfers=dst_worker.active_h2d_transfers,
            ))
            dst_morph_ms = dst_morph_only

        t_ship = ship_xfer_ms + float(dst_worker.queue_wait_ms) + dst_morph_ms
        return round(t_ship, 3), round(bw_p2p_gbps, 3)

    def decide(
        self,
        req: CPTGRequestSpec,
        workers: dict[int, WorkerState],
        now_ms: float = 0.0,
    ) -> RouteDecision:
        """Select optimal AR->DiT execution path via analytical topology + residency cost model."""
        src_w = workers[req.src_gpu]
        t_morph_ms, h2d_bw = self.evaluate_path_b_morph_ms(req, src_w, now_ms=now_ms)

        best_dst = req.src_gpu
        best_ship_ms = float("inf")
        best_p2p_bw = 0.0

        for cand_id in self.topology.peer_candidates(req.src_gpu, prefer_same_half=True):
            if cand_id not in workers:
                continue
            cand_w = workers[cand_id]
            t_ship, bw_p2p = self.evaluate_path_a_ship_ms(req, src_w, cand_w)
            if t_ship < best_ship_ms:
                best_ship_ms = t_ship
                best_dst = cand_id
                best_p2p_bw = bw_p2p

        state_mib = round(req.effective_state_bytes() / MIB, 2)
        if best_ship_ms < t_morph_ms:
            same = self.topology.same_half(req.src_gpu, best_dst)
            hop = "same_half" if same else "cross_half"
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
                reason=f"{req.architecture}: {hop} state_ship({state_mib}MiB) {best_ship_ms:.1f}ms < morph {t_morph_ms:.1f}ms",
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
        )

    def route_batch(
        self,
        requests: list[CPTGRequestSpec],
        workers: dict[int, WorkerState],
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
