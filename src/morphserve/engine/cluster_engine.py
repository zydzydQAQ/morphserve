"""Async 8-GPU coordinator managing persistent MorphServe workers, Dual-Path routing & Joint SLO metrics."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import math
import os
from pathlib import Path
import queue
import shutil
import threading
import time
from typing import Any

from morphserve.bench.interleaved_bench import (
    InterleavedSession,
    WorkloadSuite,
    assign_dual_trace_arrivals,
)
from morphserve.router import (
    CPTGRequestSpec,
    ClusterTopologyProfile,
    DirectNcclMesh,
    DualPathMorphRouter,
    HalfNodeH2DPacer,
    WorkerState,
)

from .config import EnginePolicyConfig
from .worker_proc import run_worker_process

MIB = 1024**2
GIB = 1024**3


def _read_shmem_kib() -> int:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("Shmem:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = (len(s) - 1) * (p / 100.0)
    lo = int(math.floor(idx))
    hi = min(len(s) - 1, int(math.ceil(idx)))
    if lo == hi:
        return round(float(s[lo]), 2)
    frac = idx - lo
    return round(float(s[lo] * (1.0 - frac) + s[hi] * frac), 2)


@dataclass
class SessionOutcome:
    """Execution outcome and SLO compliance of one Interleaved-Bench session."""

    session_id: str
    quadrant: str
    status: str  # "ok" | "oom" | "error"
    e2e_ms: float
    ar_ttfts_ms: list[float] = field(default_factory=list)
    ar_tpots_ms: list[float] = field(default_factory=list)
    dit_turns_ms: list[float] = field(default_factory=list)
    ar_slo_met: bool = True
    dit_slo_met: bool = True
    joint_slo_met: bool = True
    path_a_ships: int = 0
    path_b_morphs: int = 0
    preempted_ar_turns: int = 0
    total_yields: int = 0
    ar_texts: list[str] = field(default_factory=list)
    image_paths_out: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class ClusterRunReport:
    """Aggregated Phase-2 benchmark telemetry for one policy on one suite/QPS point."""

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
    path_a_ships: int
    path_b_morphs: int
    preempted_ar_count: int
    total_yields: int
    worker_oom_count: int
    status: str = "ok"
    sessions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MorphServeClusterEngine:
    """1 central async coordinator + N persistent GPU worker processes."""

    def __init__(
        self,
        policy: EnginePolicyConfig,
        model_dir: Path,
        bench_root: Path,
        tmp_root: Path = Path("/tmp/morphserve_engine_tmp"),
    ) -> None:
        self.policy = policy
        self.model_dir = Path(model_dir)
        self.bench_root = Path(bench_root)
        self.tmp_root = Path(tmp_root)
        self.topo = ClusterTopologyProfile.a100_pcie_8gpu_default()
        self.pacer = HalfNodeH2DPacer(self.topo, max_concurrent_per_half=1) if policy.use_h2d_pacer else None
        self.router = DualPathMorphRouter(self.topo, pacer=self.pacer)

        self._mp_ctx = None
        self._procs: list[Any] = []
        self._cmd_qs: list[Any] = []
        self._ack_q: Any = None
        self._ack_thread: threading.Thread | None = None
        self._stop_ack = threading.Event()
        self._pending_futures: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Future]] = {}
        self._fut_lock = threading.Lock()
        self._cmd_seq = 0

        self.worker_states: dict[int, WorkerState] = {}
        self._gpu_locks: list[asyncio.Lock] = []
        self._active_dit_ranks: dict[int, int] = {}
        self._active_preemptions: dict[int, int] = {}
        self._active_ar_count: int = 0
        self._rr_cursor = 0
        self._started = False
        self._shmem_before_kib = 0
        self.host_shmem_delta_gib = 0.0

    def _next_cmd_id(self) -> str:
        self._cmd_seq += 1
        return f"cmd_{self._cmd_seq}"

    def _drain_init_acks(self, expected_op: str) -> dict[int, dict[str, Any]]:
        collected: dict[int, dict[str, Any]] = {}
        deadline = time.monotonic() + 300.0
        while len(collected) < self.policy.num_gpus:
            rem = max(0.1, deadline - time.monotonic())
            if rem <= 0.1 and time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {expected_op}; got ranks {sorted(collected.keys())}")
            try:
                msg = self._ack_q.get(timeout=min(2.0, rem))
            except queue.Empty:
                for rank, p in enumerate(self._procs):
                    if not p.is_alive() and rank not in collected:
                        raise RuntimeError(f"worker rank {rank} exited prematurely (exitcode={p.exitcode})")
                continue
            if msg.get("op") == "WORKER_FATAL":
                raise RuntimeError(f"worker rank {msg.get('rank')} fatal: {msg.get('error')}\n{msg.get('traceback', '')}")
            collected[int(msg["rank"])] = msg
        return collected

    def start(self) -> dict[str, Any]:
        import torch
        import torch.multiprocessing as mp

        if self._started:
            return {}
        n = self.policy.num_gpus
        if torch.cuda.is_available() and torch.cuda.device_count() < n:
            raise RuntimeError(f"policy requires {n} GPUs, saw {torch.cuda.device_count()}")

        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self._shmem_before_kib = _read_shmem_kib()
        self._mp_ctx = mp.get_context("spawn")
        self._cmd_qs = [self._mp_ctx.Queue() for _ in range(n)]
        self._ack_q = self._mp_ctx.Queue()
        self._procs = []

        for rank in range(n):
            p = self._mp_ctx.Process(
                target=run_worker_process,
                args=(
                    rank,
                    self.policy,
                    str(self.model_dir),
                    str(self.bench_root),
                    self._cmd_qs[rank],
                    self._ack_q,
                ),
                daemon=False,
            )
            p.start()
            self._procs.append(p)

        if self.policy.use_shared_numa_mirror and self.policy.use_expert_pager:
            init_msgs = self._drain_init_acks("INIT_LEADER_MIRROR/FOLLOWER")
            leaders: dict[str, dict[str, Any]] = {}
            for m in init_msgs.values():
                if m.get("op") == "INIT_LEADER_MIRROR":
                    leaders[str(m["numa_node"])] = {
                        "pid": m["pid"],
                        "fd": m["fd"],
                        "nbytes": m["nbytes"],
                        "inode": m["inode"],
                    }
            for rank in range(n):
                self._cmd_qs[rank].put({"op": "ATTACH_SHARED_MIRROR", "leaders": leaders})

        _ = self._drain_init_acks("READY_FOR_NCCL")
        uid_bytes = DirectNcclMesh.generate_unique_id_bytes() if n >= 2 else None
        for rank in range(n):
            self._cmd_qs[rank].put({"op": "INIT_NCCL_MESH", "uid_bytes": uid_bytes})

        ready_msgs = self._drain_init_acks("WORKER_READY")
        shmem_after_kib = _read_shmem_kib()
        self.host_shmem_delta_gib = round(max(0.0, (shmem_after_kib - self._shmem_before_kib) / (1024**2)), 2)

        self.worker_states = {}
        for rank in range(n):
            rm = ready_msgs[rank]
            free_est = 22 * GIB if self.policy.use_tripool_vmm else 10 * GIB
            self.worker_states[rank] = WorkerState(
                gpu_id=rank,
                role=self.policy.role_of(rank),
                expert_alpha=float(rm.get("resident_alpha", self.policy.initial_alphas.get(rank, 1.0))),
                queue_wait_ms=0.0,
                kv_free_bytes=free_est,
            )

        self._active_dit_ranks = {g: 0 for g in range(n)}
        self._active_preemptions = {g: 0 for g in range(n)}
        self._stop_ack.clear()
        self._ack_thread = threading.Thread(target=self._ack_loop, name="ms_coord_acks", daemon=True)
        self._ack_thread.start()
        self._started = True
        return {
            "num_gpus": n,
            "policy": self.policy.policy_name,
            "host_shmem_delta_gib": self.host_shmem_delta_gib,
            "workers": {r: ready_msgs[r] for r in sorted(ready_msgs.keys())},
        }

    def _ack_loop(self) -> None:
        while not self._stop_ack.is_set():
            try:
                msg = self._ack_q.get(timeout=0.05)
            except queue.Empty:
                continue
            cid = msg.get("cmd_id")
            if cid is None:
                continue
            with self._fut_lock:
                entry = self._pending_futures.pop(str(cid), None)
            if entry is not None:
                loop, fut = entry
                if not loop.is_closed():
                    loop.call_soon_threadsafe(lambda f=fut, m=msg: None if f.done() else f.set_result(m))

    async def _rpc(self, rank: int, payload: dict[str, Any], timeout_s: float = 600.0) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        cid = payload.get("cmd_id") or self._next_cmd_id()
        payload = dict(payload)
        payload["cmd_id"] = cid
        fut: asyncio.Future = loop.create_future()
        with self._fut_lock:
            self._pending_futures[cid] = (loop, fut)
        self._cmd_qs[rank].put(payload)
        try:
            res = await asyncio.wait_for(fut, timeout=timeout_s)
        except Exception:
            with self._fut_lock:
                self._pending_futures.pop(cid, None)
            raise
        if res.get("status") == "error":
            raise RuntimeError(f"worker rank {rank} error on {payload.get('op')}: {res.get('error')}\n{res.get('traceback', '')}")
        return res

    def _pick_initial_gpu(self, modality_plan: str) -> int:
        n = self.policy.num_gpus
        if self.policy.routing_mode in ("twostage_shm", "static_nccl_fission"):
            cands = list(self.policy.static_und_ranks)
            cands.sort(key=lambda g: (self._gpu_locks[g].locked(), self.worker_states[g].queue_wait_ms, (g - self._rr_cursor) % n))
            self._rr_cursor = (cands[0] + 1) % n
            return cands[0]

        if modality_plan in ("ar_only", "cot_then_dit"):
            idle = [g for g in range(n) if not self._gpu_locks[g].locked()]
            if idle:
                idle.sort(key=lambda g: (self.worker_states[g].expert_alpha, (g - self._rr_cursor) % n))
                self._rr_cursor = (idle[0] + 1) % n
                return idle[0]
            # When all GPUs are busy and preemption is allowed, target an active DiT GPU with fewest preemptions!
            if self.policy.preemption_mode != "non_preemptive":
                active_dit = [g for g in range(n) if self._active_dit_ranks.get(g, 0) > 0]
                if active_dit:
                    active_dit.sort(key=lambda g: (self._active_preemptions.get(g, 0), self.worker_states[g].expert_alpha, g))
                    return active_dit[0]

        # DiT-heavy requests prefer higher resident alpha and lower queue wait
        cands = list(range(n))
        cands.sort(
            key=lambda g: (
                self._gpu_locks[g].locked(),
                self.worker_states[g].queue_wait_ms,
                -self.worker_states[g].expert_alpha,
                (g - self._rr_cursor) % n,
            )
        )
        self._rr_cursor = (cands[0] + 1) % n
        return cands[0]

    def _pick_static_gen_gpu(self, src_und_gpu: int) -> int:
        if self.policy.routing_mode == "twostage_shm":
            idx = self.policy.static_und_ranks.index(src_und_gpu)
            return self.policy.static_gen_ranks[idx % len(self.policy.static_gen_ranks)]
        cands = list(self.policy.static_gen_ranks)
        cands.sort(
            key=lambda g: (
                0 if self.topo.same_half(src_und_gpu, g) else 1,
                self._gpu_locks[g].locked(),
                self.worker_states[g].queue_wait_ms,
                g,
            )
        )
        return cands[0]

    async def _ship_nccl(
        self,
        src_gpu: int,
        dst_gpu: int,
        ctx_id: str,
        new_ctx_id: str,
        include_cfg_dup: bool,
    ) -> dict[str, Any]:
        # Canonical lock order prevents AB-BA NCCL deadlocks
        lo, hi = (src_gpu, dst_gpu) if src_gpu < dst_gpu else (dst_gpu, src_gpu)
        async with self._gpu_locks[lo]:
            async with self._gpu_locks[hi]:
                pack_res = await self._rpc(
                    src_gpu,
                    {"op": "PACK_CTX", "ctx_id": ctx_id, "include_cfg_text_dup": include_cfg_dup},
                )
                if pack_res.get("status") != "ok":
                    return pack_res
                send_fut = self._rpc(
                    src_gpu,
                    {"op": "SEND_NCCL", "packed_id": pack_res["packed_id"], "dst_rank": dst_gpu},
                )
                recv_fut = self._rpc(
                    dst_gpu,
                    {"op": "RECV_NCCL", "src_rank": src_gpu, "new_ctx_id": new_ctx_id, "meta": pack_res["meta"]},
                )
                s_res, r_res = await asyncio.gather(send_fut, recv_fut)
                await self._rpc(src_gpu, {"op": "RELEASE_CTX", "ctx_id": ctx_id})
                return {
                    "status": "ok" if (s_res["status"] == "ok" and r_res["status"] == "ok") else "oom",
                    "ship_ms": max(s_res.get("ship_ms", 0.0), r_res.get("ship_ms", 0.0)),
                    "bytes": pack_res["meta"]["total_bytes"],
                }

    async def _ship_shm(self, src_gpu: int, dst_gpu: int, ctx_id: str, new_ctx_id: str) -> dict[str, Any]:
        shm_name = f"ms_shm_{src_gpu}_{dst_gpu}_{self._next_cmd_id()}"
        async with self._gpu_locks[src_gpu]:
            s_res = await self._rpc(src_gpu, {"op": "SEND_SHM", "ctx_id": ctx_id, "shm_name": shm_name})
            if s_res.get("status") != "ok":
                return s_res
            await self._rpc(src_gpu, {"op": "RELEASE_CTX", "ctx_id": ctx_id})
        async with self._gpu_locks[dst_gpu]:
            r_res = await self._rpc(
                dst_gpu,
                {"op": "RECV_SHM", "shm_name": shm_name, "new_ctx_id": new_ctx_id, "meta": s_res["meta"]},
            )
            return r_res

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
        kv_bytes = 0
        prev_tmp_img: str | None = None
        prev_tmp_owned = False

        ar_ttfts: list[float] = []
        ar_tpots: list[float] = []
        dit_turns: list[float] = []
        ar_texts: list[str] = []
        image_paths_out: list[str] = []
        path_a_ships = 0
        path_b_morphs = 0
        preempted_ar_turns = 0
        total_yields = 0

        try:
            for ti, turn in enumerate(sess.turns):
                plan = turn.modality_plan
                is_edit_followup = (plan == "dit_edit") and (ti > 0)
                has_more_edits = any(t.modality_plan == "dit_edit" for t in sess.turns[ti + 1 :])

                if resident_gpu is None:
                    resident_gpu = self._pick_initial_gpu(plan)

                # ---- Branch 1: Pure AR turn (VQA) ----
                if plan == "ar_only":
                    self._active_ar_count += 1
                    try:
                        can_preempt = (
                            self.policy.preemption_mode != "non_preemptive"
                            and self._active_dit_ranks.get(resident_gpu, 0) > 0
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
                                        "prepare_for_dit": False,
                                    },
                                )
                            finally:
                                self._active_preemptions[resident_gpu] -= 1
                            preempted_ar_turns += 1
                        else:
                            t_enq = time.perf_counter()
                            est_ar_ms = 22.0 * max(1, turn.max_ar_tokens)
                            self.worker_states[resident_gpu].queue_wait_ms += est_ar_ms
                            try:
                                async with self._gpu_locks[resident_gpu]:
                                    q_wait_ms = (time.perf_counter() - t_enq) * 1000.0
                                    ar_res = await self._rpc(
                                        resident_gpu,
                                        {
                                            "op": "EXEC_AR",
                                            "ctx_id": ctx_id,
                                            "prompt": turn.prompt,
                                            "image_paths": turn.image_paths,
                                            "max_ar_tokens": turn.max_ar_tokens,
                                            "prepare_for_dit": False,
                                        },
                                    )
                                    ar_res["ttft_ms"] = round(q_wait_ms + float(ar_res.get("ttft_ms", 0.0)), 2)
                            finally:
                                self.worker_states[resident_gpu].queue_wait_ms = max(
                                    0.0, self.worker_states[resident_gpu].queue_wait_ms - est_ar_ms
                                )
                    finally:
                        self._active_ar_count = max(0, self._active_ar_count - 1)

                    if ar_res["status"] == "oom":
                        return SessionOutcome(
                            session_id=sess.session_id,
                            quadrant=sess.quadrant,
                            status="oom",
                            e2e_ms=(time.perf_counter() - t_sess0) * 1000.0,
                            ar_ttfts_ms=ar_ttfts,
                            ar_tpots_ms=ar_tpots,
                            dit_turns_ms=dit_turns,
                            error=ar_res.get("error", "oom"),
                        )
                    kv_bytes = int(ar_res.get("kv_bytes", kv_bytes))
                    ar_ttfts.append(float(ar_res.get("ttft_ms", 0.0)))
                    ar_tpots.extend(float(x) for x in ar_res.get("tpots_ms", []))
                    ar_texts.append(str(ar_res.get("text", "")))
                    continue

                # ---- Branch 2: Visual-CoT / DiT Edit / High-Res T2I ----
                t_turn0 = time.perf_counter()
                if plan == "cot_then_dit":
                    self._active_ar_count += 1
                    try:
                        can_preempt = (
                            self.policy.preemption_mode != "non_preemptive"
                            and self._active_dit_ranks.get(resident_gpu, 0) > 0
                        )
                        if can_preempt:
                            self._active_preemptions[resident_gpu] += 1
                            try:
                                cot_res = await self._rpc(
                                    resident_gpu,
                                    {
                                        "op": "EXEC_AR_PREEMPT",
                                        "ctx_id": ctx_id,
                                        "prompt": turn.prompt,
                                        "image_paths": turn.image_paths,
                                        "max_ar_tokens": turn.max_ar_tokens,
                                        "tpot_slo_ms": self.policy.ar_tpot_slo_ms,
                                        "enqueue_time_s": time.perf_counter(),
                                        "prepare_for_dit": True,
                                    },
                                )
                            finally:
                                self._active_preemptions[resident_gpu] -= 1
                            preempted_ar_turns += 1
                        else:
                            t_enq = time.perf_counter()
                            est_ar_ms = 22.0 * max(1, turn.max_ar_tokens)
                            self.worker_states[resident_gpu].queue_wait_ms += est_ar_ms
                            try:
                                async with self._gpu_locks[resident_gpu]:
                                    q_wait_ms = (time.perf_counter() - t_enq) * 1000.0
                                    cot_res = await self._rpc(
                                        resident_gpu,
                                        {
                                            "op": "EXEC_AR",
                                            "ctx_id": ctx_id,
                                            "prompt": turn.prompt,
                                            "image_paths": turn.image_paths,
                                            "max_ar_tokens": turn.max_ar_tokens,
                                            "prepare_for_dit": True,
                                        },
                                    )
                                    cot_res["ttft_ms"] = round(q_wait_ms + float(cot_res.get("ttft_ms", 0.0)), 2)
                            finally:
                                self.worker_states[resident_gpu].queue_wait_ms = max(
                                    0.0, self.worker_states[resident_gpu].queue_wait_ms - est_ar_ms
                                )
                    finally:
                        self._active_ar_count = max(0, self._active_ar_count - 1)
                    if cot_res["status"] == "oom":
                        return SessionOutcome(
                            session_id=sess.session_id,
                            quadrant=sess.quadrant,
                            status="oom",
                            e2e_ms=(time.perf_counter() - t_sess0) * 1000.0,
                            error=cot_res.get("error", "oom"),
                        )
                    kv_bytes = int(cot_res.get("kv_bytes", kv_bytes))
                    ar_ttfts.append(float(cot_res.get("ttft_ms", 0.0)))
                    ar_tpots.extend(float(x) for x in cot_res.get("tpots_ms", []))
                    ar_texts.append(str(cot_res.get("text", "")))
                else:
                    # dit_edit or dit_t2i prefill
                    async with self._gpu_locks[resident_gpu]:
                        prep_res = await self._rpc(
                            resident_gpu,
                            {
                                "op": "PREP_DIT",
                                "ctx_id": ctx_id,
                                "prompt": turn.prompt,
                                "image_paths": turn.image_paths if (ti == 0) else [],
                                "use_prev_image": is_edit_followup,
                                "prev_image_path": prev_tmp_img,
                                "prepare_for_dit": True,
                            },
                        )
                    if prep_res["status"] == "oom":
                        return SessionOutcome(
                            session_id=sess.session_id,
                            quadrant=sess.quadrant,
                            status="oom",
                            e2e_ms=(time.perf_counter() - t_sess0) * 1000.0,
                            error=prep_res.get("error", "oom"),
                        )
                    kv_bytes = int(prep_res.get("kv_bytes", kv_bytes))

                # Route cross-paradigm transition AR/Prefill -> DiT
                dst_gpu = resident_gpu
                if self.policy.routing_mode == "dual_path":
                    req = CPTGRequestSpec(
                        req_id=f"{sess.session_id}_t{ti}",
                        src_gpu=resident_gpu,
                        architecture="tight_mot",
                        kv_bytes=max(kv_bytes, 1 * MIB),
                        dit_step1_ms=1650.0 if turn.height >= 1024 else 287.0,
                        dit_total_steps=turn.dit_steps,
                        resolution=turn.height,
                        interleaved_roundtrip_gamma=2 if (plan == "dit_edit" and has_more_edits) else 1,
                    )
                    dec = self.router.decide(req, self.worker_states)
                    if dec.path == "state_ship" and dec.dst_gpu != resident_gpu:
                        ship = await self._ship_nccl(
                            resident_gpu, dec.dst_gpu, ctx_id, ctx_id, include_cfg_dup=False
                        )
                        if ship["status"] != "ok":
                            return SessionOutcome(
                                session_id=sess.session_id,
                                quadrant=sess.quadrant,
                                status="oom",
                                e2e_ms=(time.perf_counter() - t_sess0) * 1000.0,
                                error="nccl_ship_oom",
                            )
                        dst_gpu = dec.dst_gpu
                        resident_gpu = dst_gpu
                        path_a_ships += 1
                    else:
                        path_b_morphs += 1
                elif self.policy.routing_mode == "static_nccl_fission":
                    dst_gpu = self._pick_static_gen_gpu(resident_gpu)
                    gen_ctx_id = f"{ctx_id}_gen_t{ti}"
                    ship = await self._ship_nccl(resident_gpu, dst_gpu, ctx_id, gen_ctx_id, include_cfg_dup=True)
                    if ship["status"] != "ok":
                        return SessionOutcome(
                            session_id=sess.session_id,
                            quadrant=sess.quadrant,
                            status="oom",
                            e2e_ms=(time.perf_counter() - t_sess0) * 1000.0,
                            error="static_ship_oom",
                        )
                    ctx_id = gen_ctx_id
                    path_a_ships += 1
                elif self.policy.routing_mode == "twostage_shm":
                    dst_gpu = self._pick_static_gen_gpu(resident_gpu)
                    gen_ctx_id = f"{ctx_id}_gen_t{ti}"
                    ship = await self._ship_shm(resident_gpu, dst_gpu, ctx_id, gen_ctx_id)
                    if ship["status"] != "ok":
                        return SessionOutcome(
                            session_id=sess.session_id,
                            quadrant=sess.quadrant,
                            status="oom",
                            e2e_ms=(time.perf_counter() - t_sess0) * 1000.0,
                            error="shm_ship_oom",
                        )
                    ctx_id = gen_ctx_id
                    path_a_ships += 1

                save_tmp = None
                if save_outputs_dir is not None:
                    save_tmp = str(save_outputs_dir / f"{sess.session_id}_t{ti}.png")
                    prev_tmp_owned = False
                elif has_more_edits and self.policy.routing_mode in ("twostage_shm", "static_nccl_fission"):
                    save_tmp = str(self.tmp_root / f"{sess.session_id}_t{ti}.png")
                    prev_tmp_owned = True

                est_dit_ms = (1500.0 if turn.height >= 1024 else 285.0) * turn.dit_steps
                self.worker_states[dst_gpu].queue_wait_ms += est_dit_ms
                try:
                    async with self._gpu_locks[dst_gpu]:
                        self._active_dit_ranks[dst_gpu] += 1
                        try:
                            dit_res = await self._rpc(
                                dst_gpu,
                                {
                                    "op": "EXEC_DIT",
                                    "ctx_id": ctx_id,
                                    "prompt": turn.prompt,
                                    "height": turn.height,
                                    "width": turn.width,
                                    "dit_steps": turn.dit_steps,
                                    "seed": 52 + ti,
                                    "save_tmp_image_path": save_tmp,
                                },
                            )
                            # Elastic soft-biased alpha rebalance on hybrid ranks (even GPUs)
                            if (
                                dit_res.get("status") == "ok"
                                and self.policy.adaptive_alpha_rebalance
                                and self.policy.use_expert_pager
                                and self.policy.num_gpus >= 2
                                and (dst_gpu % 2 == 0)
                            ):
                                active_dit_tot = sum(self._active_dit_ranks.values()) - 1
                                target_alpha = 0.25 if self._active_ar_count > active_dit_tot else 0.50
                                alpha_res = await self._rpc(dst_gpu, {"op": "SET_ALPHA", "alpha": target_alpha})
                                dit_res["resident_alpha"] = round(
                                    float(alpha_res.get("resident_layers", 14)) / 28.0, 3
                                )
                        finally:
                            self._active_dit_ranks[dst_gpu] -= 1
                finally:
                    self.worker_states[dst_gpu].queue_wait_ms = max(
                        0.0, self.worker_states[dst_gpu].queue_wait_ms - est_dit_ms
                    )

                if dit_res["status"] == "oom":
                    return SessionOutcome(
                        session_id=sess.session_id,
                        quadrant=sess.quadrant,
                        status="oom",
                        e2e_ms=(time.perf_counter() - t_sess0) * 1000.0,
                        error=dit_res.get("error", "oom"),
                    )

                self.worker_states[dst_gpu].expert_alpha = float(
                    dit_res.get("resident_alpha", self.worker_states[dst_gpu].expert_alpha)
                )
                total_yields += int(dit_res.get("yield_count", 0))
                if save_tmp is not None:
                    prev_tmp_img = save_tmp
                    if save_outputs_dir is not None:
                        image_paths_out.append(save_tmp)
                # If static fission sent context away from und_gpu, next turn starts back on an und_gpu with prev_tmp_img
                if self.policy.routing_mode in ("twostage_shm", "static_nccl_fission"):
                    await self._rpc(dst_gpu, {"op": "RELEASE_CTX", "ctx_id": ctx_id})
                    resident_gpu = self._pick_initial_gpu("dit_edit")
                    ctx_id = f"ctx_{sess.session_id}_t{ti + 1}"

                dit_turns.append((time.perf_counter() - t_turn0) * 1000.0)

            ar_ok = all(x <= self.policy.ar_ttft_slo_ms for x in ar_ttfts) and all(
                x <= self.policy.ar_tpot_slo_ms for x in ar_tpots
            )
            dit_ok = all(x <= self.policy.dit_turn_slo_ms for x in dit_turns)
            return SessionOutcome(
                session_id=sess.session_id,
                quadrant=sess.quadrant,
                status="ok",
                e2e_ms=round((time.perf_counter() - t_sess0) * 1000.0, 2),
                ar_ttfts_ms=ar_ttfts,
                ar_tpots_ms=ar_tpots,
                dit_turns_ms=[round(x, 2) for x in dit_turns],
                ar_slo_met=ar_ok,
                dit_slo_met=dit_ok,
                joint_slo_met=(ar_ok and dit_ok),
                path_a_ships=path_a_ships,
                path_b_morphs=path_b_morphs,
                preempted_ar_turns=preempted_ar_turns,
                total_yields=total_yields,
                ar_texts=ar_texts,
                image_paths_out=image_paths_out,
            )
        finally:
            if resident_gpu is not None:
                try:
                    await self._rpc(resident_gpu, {"op": "RELEASE_CTX", "ctx_id": ctx_id})
                except Exception:
                    pass
            if prev_tmp_img and prev_tmp_owned:
                Path(prev_tmp_img).unlink(missing_ok=True)

    async def _run_suite_async(
        self,
        suite: WorkloadSuite,
        target_qps: float | None,
        burstgpt_deltas: list[float] | None,
        gentd26_deltas: list[float] | None,
        save_outputs_dir: Path | None = None,
    ) -> ClusterRunReport:
        self._gpu_locks = [asyncio.Lock() for _ in range(self.policy.num_gpus)]
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

        stat_res = await asyncio.gather(*(self._rpc(g, {"op": "GET_STATS"}) for g in range(self.policy.num_gpus)))
        peak_res_mib = max(float(st.get("peak_reserved_mib", 0.0)) for st in stat_res)
        worker_ooms = sum(int(st.get("oom_count", 0)) for st in stat_res)
        overflow_creates = sum(int((st.get("vmm_stats") or {}).get("overflow_creates", 0)) for st in stat_res)

        all_ttfts = [x for o in outcomes for x in o.ar_ttfts_ms]
        all_tpots = [x for o in outcomes for x in o.ar_tpots_ms]
        all_dits = [x for o in outcomes for x in o.dit_turns_ms]

        completed = [o for o in outcomes if o.status == "ok"]
        oom_sess = sum(1 for o in outcomes if o.status == "oom")
        err_sess = sum(1 for o in outcomes if o.status == "error")
        joint_ok = [o for o in completed if o.joint_slo_met]
        ar_ok_cnt = sum(1 for o in completed if o.ar_slo_met)
        dit_ok_cnt = sum(1 for o in completed if o.dit_slo_met)
        n_tot = max(1, len(outcomes))

        return ClusterRunReport(
            policy_name=self.policy.policy_name,
            suite_name=suite.name,
            num_gpus=self.policy.num_gpus,
            offered_qps=round(eff_qps, 3),
            wall_seconds=round(wall_s, 2),
            total_sessions=len(outcomes),
            completed_sessions=len(completed),
            oom_sessions=oom_sess,
            error_sessions=err_sess,
            joint_slo_goodput_rps=round(len(joint_ok) / wall_s, 4),
            joint_slo_attainment_pct=round(100.0 * len(joint_ok) / n_tot, 1),
            ar_slo_attainment_pct=round(100.0 * ar_ok_cnt / n_tot, 1),
            dit_slo_attainment_pct=round(100.0 * dit_ok_cnt / n_tot, 1),
            ar_ttft_p50_ms=_percentile(all_ttfts, 50),
            ar_ttft_p95_ms=_percentile(all_ttfts, 95),
            ar_ttft_p99_ms=_percentile(all_ttfts, 99),
            ar_tpot_p50_ms=_percentile(all_tpots, 50),
            ar_tpot_p95_ms=_percentile(all_tpots, 95),
            ar_tpot_p99_ms=_percentile(all_tpots, 99),
            dit_turn_p50_ms=_percentile(all_dits, 50),
            dit_turn_p95_ms=_percentile(all_dits, 95),
            dit_turn_p99_ms=_percentile(all_dits, 99),
            peak_gpu_reserved_mib=round(peak_res_mib, 2),
            host_shmem_delta_gib=self.host_shmem_delta_gib,
            vmm_overflow_creates=overflow_creates,
            path_a_ships=sum(o.path_a_ships for o in outcomes),
            path_b_morphs=sum(o.path_b_morphs for o in outcomes),
            preempted_ar_count=sum(o.preempted_ar_turns for o in outcomes),
            total_yields=sum(o.total_yields for o in outcomes),
            worker_oom_count=worker_ooms,
            status="ok" if err_sess == 0 else "check_failed",
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
                suite,
                target_qps,
                burstgpt_deltas,
                gentd26_deltas,
                save_outputs_dir=save_outputs_dir,
            )
        )

    def shutdown(self) -> None:
        if not self._started and not self._procs:
            return
        for rank in range(len(self._cmd_qs)):
            try:
                self._cmd_qs[rank].put({"op": "SHUTDOWN", "cmd_id": f"shut_{rank}"})
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        self._stop_ack.set()
        if self._ack_thread is not None:
            self._ack_thread.join(timeout=2)
            self._ack_thread = None
        self._procs.clear()
        self._cmd_qs.clear()
        self._ack_q = None
        self._started = False
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)
