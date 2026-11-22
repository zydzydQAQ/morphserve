"""Model-agnostic async coordinator managing `WorkerFleet`, stateful `DualPathMorphRouter` & Joint SLOs."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import hashlib
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any

from morphserve.router import (
    CPTGRequestSpec,
    ClusterTopologyProfile,
    DEFAULT_OPERATOR_PROFILE,
    DualPathMorphRouter,
    HalfNodeH2DPacer,
    WorkerState,
)
from morphserve.telemetry import percentile

from .config import EnginePolicyConfig
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


def _deterministic_think_s(session_id: str, turn_idx: int, bounds: tuple[float, float]) -> float:
    lo, hi = float(bounds[0]), float(bounds[1])
    if hi <= lo:
        return max(0.0, lo)
    h = int(hashlib.md5(f"{session_id}:{turn_idx}".encode("utf-8")).hexdigest()[:8], 16)
    frac = (h % 10_000) / 10_000.0
    return lo + frac * (hi - lo)


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
    peak_pytorch_reserved_mib: float = 0.0
    cross_node_ships: int = 0
    max_cross_node_ship_bytes: int = 0
    slo_curve_by_k: dict[str, dict[str, float]] = field(default_factory=dict)
    status: str = "ok"
    sessions: list[dict[str, Any]] = field(default_factory=list)

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
        if res.get("status") == "error":
            raise RuntimeError(
                f"worker rank {rank} error on {payload.get('op')}: {res.get('error')}\n{res.get('traceback', '')}"
            )
        self._sync_worker_telemetry(int(rank), res)
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
        )

    async def _execute_ar_phase(
        self,
        resident_gpu: int,
        ctx_id: str,
        turn: InterleavedTurn,
        prepare_for_dit: bool,
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
                        "max_ar_tokens": turn.max_ar_tokens,
                        "tpot_slo_ms": self.policy.ar_tpot_slo_ms,
                        "enqueue_time_s": time.perf_counter(),
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
                        "max_ar_tokens": turn.max_ar_tokens,
                        "enqueue_time_s": t_enq,
                        "prepare_for_dit": prepare_for_dit,
                    },
                )
            finally:
                rem = max(0, self._inflight_ar_on_gpu.get(resident_gpu, 1) - 1)
                self._inflight_ar_on_gpu[resident_gpu] = rem
                if rem == 0:
                    self._ar_drained_events[resident_gpu].set()
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
        if self.policy.routing_mode in ("twostage_shm", "static_nccl_fission") and self.policy.static_und_ranks:
            cands = [g for g in self.policy.static_und_ranks if g != failed_gpu]
        else:
            cands = [g for g in range(self.policy.num_gpus) if g != failed_gpu]
        if not cands:
            return failed_gpu
        busy = self._gpu_busy_map()
        cands.sort(
            key=lambda g: (
                0 if self.worker_states[g].kv_free_bytes >= 128 * MIB else 1,
                self.worker_states[g].queue_wait_ms,
                bool(busy.get(g, False)),
                g,
            )
        )
        return cands[0]

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
        dit_turns: list[float] = []
        dit_heights: list[int] = []
        dit_steps_list: list[int] = []
        ar_texts: list[str] = []
        image_paths_out: list[str] = []
        path_a_ships = 0
        path_b_morphs = 0
        cross_node_ships = 0
        cross_node_ship_bytes_list: list[int] = []
        ship_bytes_list: list[int] = []
        ship_ms_list: list[float] = []
        preempted_ar_turns = 0
        total_yields = 0
        oom_retries = 0
        is_tight_mot = self.policy.cptg_architecture == "tight_mot"

        try:
            for ti, turn in enumerate(sess.turns):
                if ti > 0:
                    # Realistic user think-time residency (`D1`) holding live KV across turns!
                    think_s = _deterministic_think_s(sess.session_id, ti, self.policy.inter_turn_think_s_range)
                    if think_s > 0.0:
                        await asyncio.sleep(think_s)

                plan = turn.modality_plan
                is_edit_followup = (plan == "dit_edit") and (ti > 0)
                has_more_turns = (ti + 1) < len(sess.turns)
                has_more_edits = any(t.modality_plan == "dit_edit" for t in sess.turns[ti + 1 :])

                # Bounded OOM retry loop (`D3`: `max_oom_retries = 1` with user arrival clock running!)
                attempt = 0
                while True:
                    if resident_gpu is None:
                        resident_gpu = self._pick_initial_gpu(plan)

                    # ---- Branch 1: Pure AR turn (VQA) ----
                    if plan == "ar_only":
                        ar_res, did_preempt = await self._execute_ar_phase(
                            resident_gpu, ctx_id, turn, prepare_for_dit=False
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
                            return SessionOutcome(
                                session_id=sess.session_id,
                                quadrant=sess.quadrant,
                                status="oom",
                                e2e_ms=round((time.perf_counter() - t_sess0) * 1000.0, 2),
                                ar_ttfts_ms=ar_ttfts,
                                ar_tpots_ms=ar_tpots,
                                dit_turns_ms=dit_turns,
                                dit_turn_heights=dit_heights,
                                dit_turn_steps=dit_steps_list,
                                ar_slo_met=False,
                                dit_slo_met=False,
                                joint_slo_met=False,
                                oom_retries=oom_retries,
                                error=ar_res.get("error", "oom"),
                            )
                        src_ctx_alive = True
                        if did_preempt:
                            preempted_ar_turns += 1
                        kv_bytes = int(ar_res.get("kv_bytes", kv_bytes))
                        ar_ttfts.append(float(ar_res.get("ttft_ms", 0.0)))
                        ar_tpots.extend(float(x) for x in ar_res.get("tpots_ms", []))
                        ar_texts.append(str(ar_res.get("text", "")))
                        break

                    # ---- Branch 2: Visual-CoT / DiT Edit / High-Res T2I ----
                    t_turn0 = time.perf_counter()
                    cot_text_pending: str | None = None
                    if plan == "cot_then_dit":
                        cot_res, did_preempt = await self._execute_ar_phase(
                            resident_gpu, ctx_id, turn, prepare_for_dit=True
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
                            return SessionOutcome(
                                session_id=sess.session_id,
                                quadrant=sess.quadrant,
                                status="oom",
                                e2e_ms=round((time.perf_counter() - t_sess0) * 1000.0, 2),
                                ar_slo_met=False,
                                dit_slo_met=False,
                                joint_slo_met=False,
                                oom_retries=oom_retries,
                                error=cot_res.get("error", "oom"),
                            )
                        src_ctx_alive = True
                        if did_preempt:
                            preempted_ar_turns += 1
                        kv_bytes = int(cot_res.get("kv_bytes", kv_bytes))
                        ar_ttfts.append(float(cot_res.get("ttft_ms", 0.0)))
                        ar_tpots.extend(float(x) for x in cot_res.get("tpots_ms", []))
                        cot_text_pending = str(cot_res.get("text", ""))
                    else:
                        est_prep_ms = DEFAULT_OPERATOR_PROFILE.prep_overhead_ms
                        self.worker_states[resident_gpu].queue_wait_ms += est_prep_ms
                        try:
                            await self._acquire_exclusive_gpu(resident_gpu)
                            try:
                                # When `tight_mot` already holds live context (`last_img_u8` / `last_img_pil`),
                                # pass `prev_image_path=None` so `GpuInPlaceLoopback` is never bypassed by disk PNGs!
                                eff_prev_img = None if (is_tight_mot and src_ctx_alive) else prev_tmp_img
                                prep_res = await self._rpc(
                                    resident_gpu,
                                    {
                                        "op": "PREP_DIT",
                                        "ctx_id": ctx_id,
                                        "prompt": turn.prompt,
                                        "height": turn.height,
                                        "width": turn.width,
                                        "image_paths": turn.image_paths if (ti == 0) else [],
                                        "use_prev_image": is_edit_followup,
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
                            return SessionOutcome(
                                session_id=sess.session_id,
                                quadrant=sess.quadrant,
                                status="oom",
                                e2e_ms=round((time.perf_counter() - t_sess0) * 1000.0, 2),
                                ar_slo_met=False,
                                dit_slo_met=False,
                                joint_slo_met=False,
                                oom_retries=oom_retries,
                                error=prep_res.get("error", "oom"),
                            )
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
                                fwd_b = int(ship.get("bytes", kv_bytes))
                                if not self.topo.same_node(src_gpu_before_ship, dst_gpu):
                                    cross_node_ships += 1
                                    cross_node_ship_bytes_list.append(fwd_b)
                                ship_bytes_list.append(fwd_b)
                        elif self.policy.routing_mode == "dual_path":
                            path_b_morphs += 1

                        if dit_res.get("status") != "oom":
                            if save_outputs_dir is not None:
                                save_tmp = str(save_outputs_dir / f"{sess.session_id}_t{ti}.png")
                            elif (
                                has_more_edits
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
                        return SessionOutcome(
                            session_id=sess.session_id,
                            quadrant=sess.quadrant,
                            status="oom",
                            e2e_ms=round((time.perf_counter() - t_sess0) * 1000.0, 2),
                            ar_ttfts_ms=ar_ttfts,
                            ar_tpots_ms=ar_tpots,
                            dit_turns_ms=dit_turns,
                            dit_turn_heights=dit_heights,
                            dit_turn_steps=dit_steps_list,
                            ar_slo_met=False,
                            dit_slo_met=False,
                            joint_slo_met=False,
                            oom_retries=oom_retries,
                            error=dit_res.get("error", "oom"),
                        )

                    if cot_text_pending is not None:
                        ar_texts.append(cot_text_pending)
                    self._sync_worker_telemetry(dst_gpu, dit_res)
                    total_yields += int(dit_res.get("yield_count", 0))
                    if save_tmp is not None:
                        prev_tmp_img = save_tmp
                        if save_outputs_dir is not None:
                            image_paths_out.append(save_tmp)

                    dit_turns.append((time.perf_counter() - t_turn0) * 1000.0)
                    dit_heights.append(int(turn.height))
                    dit_steps_list.append(int(turn.dit_steps))
                    break

            ar_ok = all(x <= self.policy.ar_ttft_slo_ms for x in ar_ttfts) and all(
                x <= self.policy.ar_tpot_slo_ms for x in ar_tpots
            )
            dit_ok = all(
                lat <= self.policy.dit_turn_slo_for(h, steps=st)
                for lat, h, st in zip(dit_turns, dit_heights, dit_steps_list)
            )
            return SessionOutcome(
                session_id=sess.session_id,
                quadrant=sess.quadrant,
                status="ok",
                e2e_ms=round((time.perf_counter() - t_sess0) * 1000.0, 2),
                ar_ttfts_ms=ar_ttfts,
                ar_tpots_ms=ar_tpots,
                dit_turns_ms=[round(x, 2) for x in dit_turns],
                dit_turn_heights=dit_heights,
                dit_turn_steps=dit_steps_list,
                ar_slo_met=ar_ok,
                dit_slo_met=dit_ok,
                joint_slo_met=(ar_ok and dit_ok),
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
            )
        except Exception as exc:
            return SessionOutcome(
                session_id=sess.session_id,
                quadrant=sess.quadrant,
                status="error",
                e2e_ms=round((time.perf_counter() - t_sess0) * 1000.0, 2),
                ar_ttfts_ms=ar_ttfts,
                ar_tpots_ms=ar_tpots,
                dit_turns_ms=[round(x, 2) for x in dit_turns],
                dit_turn_heights=dit_heights,
                dit_turn_steps=dit_steps_list,
                ar_slo_met=False,
                dit_slo_met=False,
                joint_slo_met=False,
                oom_retries=oom_retries,
                error=f"{type(exc).__name__}: {exc}",
            )
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
        ar_ok = all(x <= slo["ar_ttft_slo_ms"] for x in out.ar_ttfts_ms) and all(
            x <= slo["ar_tpot_slo_ms"] for x in out.ar_tpots_ms
        )
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
        self._ar_drained_events = []
        for _ in range(n_gpus):
            ev = asyncio.Event()
            ev.set()
            self._ar_drained_events.append(ev)

        if save_outputs_dir is not None:
            save_outputs_dir.mkdir(parents=True, exist_ok=True)
        sessions = [
            InterleavedSession(
                session_id=s.session_id,
                quadrant=s.quadrant,
                source_dataset=s.source_dataset,
                arrival_offset_s=s.arrival_offset_s,
                turns=list(s.turns),
            )
            for s in suite.sessions
        ]
        eff_qps = float(target_qps if target_qps is not None else suite.base_qps)
        if target_qps is not None and burstgpt_deltas and gentd26_deltas:
            assign_dual_trace_arrivals(sessions, burstgpt_deltas, gentd26_deltas, target_qps=eff_qps)

        t0 = time.perf_counter()
        outcomes = await asyncio.gather(
            *(self._run_session_coro(s, t0, save_outputs_dir=save_outputs_dir) for s in sessions)
        )
        wall_s = max(time.perf_counter() - t0, 1e-6)

        stat_res = await asyncio.gather(*(self._rpc(g, {"op": "GET_STATS"}) for g in range(n_gpus)))
        peak_res_mib = max(
            (
                float(
                    st["combined_physical_mib"]
                    if st.get("combined_physical_mib") is not None
                    else (st["peak_reserved_mib"] if st.get("peak_reserved_mib") is not None else 0.0)
                )
                for st in stat_res
            ),
            default=0.0,
        )
        peak_pytorch_res_mib = max(
            (
                float(
                    st["pytorch_reserved_mib"]
                    if st.get("pytorch_reserved_mib") is not None
                    else (st["peak_reserved_mib"] if st.get("peak_reserved_mib") is not None else 0.0)
                )
                for st in stat_res
            ),
            default=0.0,
        )
        worker_ooms = sum(int(st.get("oom_count", 0)) for st in stat_res)
        overflow_creates = sum(int((st.get("vmm_stats") or {}).get("overflow_creates", 0)) for st in stat_res)
        carves = sum(int((st.get("vmm_stats") or {}).get("super_to_fine_carves", 0)) for st in stat_res)
        evict_kv = sum(int((st.get("vmm_stats") or {}).get("expert_auto_evictions_for_kv", 0)) for st in stat_res)
        evict_tr = sum(
            int((st.get("vmm_stats") or {}).get("expert_auto_evictions_for_transient", 0)) for st in stat_res
        )

        # Steady-state core window (`D5`: trim first/last `steady_state_trim_frac` when n >= 20)
        n_tot = max(1, len(outcomes))
        trim = int(len(outcomes) * self.policy.steady_state_trim_frac) if len(outcomes) >= 20 else 0
        core_outs = outcomes[trim : len(outcomes) - trim] if trim > 0 else list(outcomes)
        n_core = max(1, len(core_outs))
        core_joint_cnt = sum(1 for o in core_outs if o.status == "ok" and o.joint_slo_met)

        all_ttfts = [x for o in outcomes for x in o.ar_ttfts_ms]
        all_tpots = [x for o in outcomes for x in o.ar_tpots_ms]
        all_dits = [x for o in outcomes for x in o.dit_turns_ms]

        ok_outs = [o for o in outcomes if o.status == "ok"]
        oom_cnt = sum(1 for o in outcomes if o.status == "oom")
        err_cnt = sum(1 for o in outcomes if o.status == "error")
        joint_cnt = sum(1 for o in ok_outs if o.joint_slo_met)
        ar_cnt = sum(1 for o in ok_outs if o.ar_slo_met)
        dit_cnt = sum(1 for o in ok_outs if o.dit_slo_met)

        slo_curve: dict[str, dict[str, float]] = {}
        for k_val in (1.5, 2.0, 2.5, 3.0):
            k_cnt = sum(1 for o in core_outs if self._evaluate_outcome_at_k(o, k_val))
            att_k = round(100.0 * k_cnt / n_core, 1)
            slo_curve[f"{k_val:.1f}x"] = {
                "attainment_pct": att_k,
                "goodput_rps": round(eff_qps * (k_cnt / n_core), 4),
            }

        return ClusterRunReport(
            policy_name=self.policy.policy_name,
            suite_name=suite.name,
            num_gpus=n_gpus,
            offered_qps=round(eff_qps, 3),
            wall_seconds=round(wall_s, 2),
            total_sessions=len(outcomes),
            completed_sessions=len(ok_outs),
            oom_sessions=oom_cnt,
            error_sessions=err_cnt,
            joint_slo_goodput_rps=round(eff_qps * (joint_cnt / n_tot), 4),
            raw_wall_goodput_rps=round(joint_cnt / wall_s, 4),
            steady_state_goodput_rps=round(eff_qps * (core_joint_cnt / n_core), 4),
            steady_state_attainment_pct=round(100.0 * core_joint_cnt / n_core, 1),
            joint_slo_attainment_pct=round(100.0 * joint_cnt / n_tot, 1),
            ar_slo_attainment_pct=round(100.0 * ar_cnt / n_tot, 1),
            dit_slo_attainment_pct=round(100.0 * dit_cnt / n_tot, 1),
            ar_ttft_p50_ms=percentile(all_ttfts, 50),
            ar_ttft_p95_ms=percentile(all_ttfts, 95),
            ar_ttft_p99_ms=percentile(all_ttfts, 99),
            ar_tpot_p50_ms=percentile(all_tpots, 50),
            ar_tpot_p95_ms=percentile(all_tpots, 95),
            ar_tpot_p99_ms=percentile(all_tpots, 99),
            dit_turn_p50_ms=percentile(all_dits, 50),
            dit_turn_p95_ms=percentile(all_dits, 95),
            dit_turn_p99_ms=percentile(all_dits, 99),
            peak_gpu_reserved_mib=round(peak_res_mib, 2),
            peak_pytorch_reserved_mib=round(peak_pytorch_res_mib, 2),
            host_shmem_delta_gib=self.host_shmem_delta_gib,
            vmm_overflow_creates=overflow_creates,
            vmm_super_to_fine_carves=carves,
            vmm_expert_evictions_for_kv=evict_kv,
            vmm_expert_evictions_for_transient=evict_tr,
            path_a_ships=sum(o.path_a_ships for o in outcomes),
            path_b_morphs=sum(o.path_b_morphs for o in outcomes),
            preempted_ar_count=sum(o.preempted_ar_turns for o in outcomes),
            total_yields=sum(o.total_yields for o in outcomes),
            worker_oom_count=worker_ooms,
            oom_retry_count=sum(o.oom_retries for o in outcomes),
            cross_node_ships=sum(o.cross_node_ships for o in outcomes),
            max_cross_node_ship_bytes=max(
                (b for o in outcomes for b in o.cross_node_ship_bytes_list),
                default=0,
            ),
            slo_curve_by_k=slo_curve,
            status="ok" if err_cnt == 0 else "check_failed",
            sessions=[asdict(o) for o in outcomes],
        )

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
