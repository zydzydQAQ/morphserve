"""MorphServe Module 4: Deterministic Layer-Chunk & Step-Preemptive Unified AR+DiT Scheduler."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass, field
import time
from typing import Any, Callable

from morphserve._core import get_core_lib

from .bagel_interceptor import BagelExecutionInterceptor, _sync_cuda_if_present


@dataclass
class ArDecodeTask:
    """One interactive autoregressive text decoding request managed by the unified scheduler."""

    req_id: str
    kv_cache: Any
    curr_tokens: Any  # LongTensor [1]
    pos_ids: Any  # LongTensor [1]
    max_tokens: int
    end_token_id: int | None
    tpot_slo_ms: float
    arrival_time_s: float = 0.0
    generated_token_ids: list[int] = field(default_factory=list)
    token_timestamps_s: list[float] = field(default_factory=list)
    last_step_time_s: float = 0.0
    finished: bool = False
    _curr_tok_int: int | None = field(default=None, repr=False)

    def next_deadline_s(self) -> float:
        base = self.last_step_time_s if self.last_step_time_s > 0.0 else self.arrival_time_s
        return base + (self.tpot_slo_ms / 1000.0)

    def inter_token_latencies_ms(self) -> list[float]:
        if len(self.token_timestamps_s) < 2:
            return []
        return [
            (self.token_timestamps_s[i] - self.token_timestamps_s[i - 1]) * 1000.0
            for i in range(1, len(self.token_timestamps_s))
        ]

    def ttft_ms(self) -> float:
        if not self.token_timestamps_s:
            return 0.0
        return max(0.0, (self.token_timestamps_s[0] - self.arrival_time_s) * 1000.0)


class UnifiedPreemptiveScheduler:
    """Single-process AR+DiT EDF scheduler (`non_preemptive`, `step_preemptive`, `layer_chunk`).

    Combines immediate `on_task_complete` mid-DiT ACK emission, adaptive `k >= 1` layer-chunk bounds,
    and true `[B, D]` packed GEMM microbatch execution across concurrent active `ArDecodeTask`s.
    """

    def __init__(
        self,
        pipeline: Any,
        mode: str = "layer_chunk",
        tpot_slo_ms: float = 125.6,
        guard_ms: float = 14.0,
        init_layer_ms: float = 10.5,
        init_ar_step_ms: float = 66.0,
        max_ar_tokens_per_yield: int = 1,
        max_microbatch_cap: int = 8,
        pager: Any | None = None,
        external_arrival_poll: Callable[[float], list[ArDecodeTask]] | None = None,
        on_task_complete: Callable[[ArDecodeTask], None] | None = None,
    ) -> None:
        if mode not in ("non_preemptive", "step_preemptive", "layer_chunk"):
            raise ValueError(f"unsupported scheduler mode: {mode}")
        self.pipeline = pipeline
        self.bagel = pipeline.bagel
        self.lm_model = pipeline.bagel.language_model.model
        self.lm_head = pipeline.bagel.language_model.lm_head
        self.mode = mode
        self.tpot_slo_ms = float(tpot_slo_ms)
        self.guard_ms = float(guard_ms)
        self.ema_layer_ms = float(init_layer_ms)
        self.ema_ar_step_ms = float(init_ar_step_ms)
        self.max_ar_tokens_per_yield = int(max_ar_tokens_per_yield)
        self.max_microbatch_cap = max(1, int(max_microbatch_cap))
        self.pager = pager
        self.external_arrival_poll = external_arrival_poll
        self.on_task_complete = on_task_complete

        self.active_ar_tasks: deque[ArDecodeTask] = deque()
        self.completed_ar_tasks: list[ArDecodeTask] = []
        self._pending_arrivals: list[tuple[float, ArDecodeTask]] = []
        self._run_t0_s: float = 0.0
        self._last_sync_s: float = 0.0
        self._queued_gpu_ms: float = 0.0
        self._step0_h2d_streaming: bool = False
        self._layer_calibrated: bool = False

        self._lib = get_core_lib()
        self._edf_handle = int(
            self._lib.ms_edf_create(
                ctypes.c_int(max(1, len(self.lm_model.layers))),
                ctypes.c_double(self.tpot_slo_ms),
                ctypes.c_double(self.guard_ms),
                ctypes.c_double(self.ema_layer_ms),
                ctypes.c_double(self.ema_ar_step_ms),
                ctypes.c_int(self.max_microbatch_cap),
            )
            or 0
        )

        # Telemetry
        self.yield_count: int = 0
        self.ar_steps_executed: int = 0
        self.ar_compute_total_ms: float = 0.0
        self.batched_ar_yields: int = 0
        self.max_microbatch_size: int = 0
        self.layer_times_ms: list[float] = []
        self.step_times_ms: list[float] = []
        self.last_effective_chunk_layers: int = self.compute_adaptive_chunk_layers()
        self.in_ar_yield: bool = False

    def __del__(self) -> None:
        h = getattr(self, "_edf_handle", 0)
        if h:
            self._edf_handle = 0
            try:
                self._lib.ms_edf_destroy(ctypes.c_void_p(h))
            except Exception:
                pass

    def _sync_edf_state(self) -> None:
        if self._edf_handle:
            self._lib.ms_edf_set_state(
                ctypes.c_void_p(self._edf_handle),
                ctypes.c_double(self.tpot_slo_ms),
                ctypes.c_double(self.guard_ms),
                ctypes.c_double(self.ema_layer_ms),
                ctypes.c_double(self.ema_ar_step_ms),
                ctypes.c_int(max(1, int(self.max_microbatch_cap))),
            )

    def reset_telemetry(self, *, preserve_layer_calibration: bool = False) -> None:
        self.yield_count = 0
        self.ar_steps_executed = 0
        self.ar_compute_total_ms = 0.0
        self.batched_ar_yields = 0
        self.max_microbatch_size = 0
        if not preserve_layer_calibration:
            self.layer_times_ms = []
            self._layer_calibrated = False
        self.step_times_ms = []
        self.last_effective_chunk_layers = self.compute_adaptive_chunk_layers()
        self.active_ar_tasks.clear()
        self.completed_ar_tasks.clear()
        self._pending_arrivals.clear()

    @property
    def interceptor(self) -> BagelExecutionInterceptor:
        return BagelExecutionInterceptor.for_pipeline(self.pipeline)

    def begin_session_clock(self) -> None:
        now = time.perf_counter()
        self._run_t0_s = now
        self._last_sync_s = now
        self._queued_gpu_ms = 0.0
        streaming = False
        if self.pager is not None and getattr(self.pager, "vmm", None) is not None:
            res_alpha_fn = getattr(self.pager, "resident_alpha", None)
            if callable(res_alpha_fn):
                streaming = float(res_alpha_fn()) < 0.999
            elif callable(getattr(self.pager.vmm, "expert_resident_count", None)):
                streaming = int(self.pager.vmm.expert_resident_count(self.pager.expert_id)) < int(self.pager.num_layers)
        self._step0_h2d_streaming = streaming
        self.last_effective_chunk_layers = self.compute_adaptive_chunk_layers()

    def schedule_ar_arrival(self, offset_ms: float, task: ArDecodeTask) -> None:
        """Register an AR decode task to arrive `offset_ms` after the mixed run starts."""
        self._pending_arrivals.append((float(offset_ms) / 1000.0, task))
        self._pending_arrivals.sort(key=lambda pair: pair[0])

    def _admit_due_arrivals(self, now_s: float) -> None:
        while self._pending_arrivals and (self._run_t0_s + self._pending_arrivals[0][0]) <= now_s:
            rel_s, task = self._pending_arrivals.pop(0)
            task.arrival_time_s = self._run_t0_s + rel_s
            task.last_step_time_s = task.arrival_time_s
            self.active_ar_tasks.append(task)
        if self.external_arrival_poll is not None and not self.in_ar_yield:
            for task in self.external_arrival_poll(now_s):
                if task.arrival_time_s <= 0.0:
                    task.arrival_time_s = now_s
                if task.last_step_time_s <= 0.0:
                    task.last_step_time_s = task.arrival_time_s
                self.active_ar_tasks.append(task)

    def _earliest_deadline_s(self) -> float | None:
        if not self.active_ar_tasks:
            return None
        return min(t.next_deadline_s() for t in self.active_ar_tasks)

    def compute_adaptive_chunk_layers(self, tpot_slo_ms: float | None = None) -> int:
        """Analytical layer-chunk bound `k >= 1` evaluated by C++ `PreemptEdfController`."""
        self._sync_edf_state()
        ovr = -1.0 if tpot_slo_ms is None else float(tpot_slo_ms)
        return int(self._lib.ms_edf_adaptive_chunk_layers(ctypes.c_void_p(self._edf_handle), ctypes.c_double(ovr)))

    def _should_yield_now(self, now_s: float, next_unit_ms: float) -> bool:
        self._admit_due_arrivals(now_s)
        if not self.active_ar_tasks:
            return False
        earliest = self._earliest_deadline_s()
        assert earliest is not None
        self._sync_edf_state()
        return bool(
            self._lib.ms_edf_should_yield(
                ctypes.c_void_p(self._edf_handle),
                ctypes.c_double(now_s),
                ctypes.c_double(earliest),
                ctypes.c_double(next_unit_ms),
            )
        )

    def _finalize_task_if_done(self, task: ArDecodeTask) -> None:
        if task.finished:
            self.completed_ar_tasks.append(task)
            if self.on_task_complete is not None:
                self.on_task_complete(task)
        else:
            self.active_ar_tasks.append(task)

    def _execute_packed_ar_tasks(self, live_tasks: list[ArDecodeTask], sync_gpu: bool = True) -> int:
        if not live_tasks:
            return 0
        interceptor = BagelExecutionInterceptor.for_pipeline(self.pipeline)
        if self.pager is not None:
            interceptor.bind_pager(self.pager)

        t0 = time.perf_counter()
        self.in_ar_yield = True
        try:
            for task in live_tasks:
                tok_val = (
                    task._curr_tok_int
                    if task._curr_tok_int is not None
                    else (
                        int(task.curr_tokens.reshape(-1)[0].item())
                        if hasattr(task.curr_tokens, "reshape")
                        else int(task.curr_tokens)
                    )
                )
                task.generated_token_ids.append(tok_val)

            next_tokens, updated_caches = interceptor.forward_ar_step_batch(
                [t.curr_tokens for t in live_tasks],
                [t.pos_ids for t in live_tasks],
                [t.kv_cache for t in live_tasks],
            )
            # Single compact D2H copy of [B] int64 token IDs also synchronizes stream completion cleanly
            raw_tok_iter = next_tokens.detach().cpu().tolist() if hasattr(next_tokens, "detach") else list(next_tokens)
            next_ids = [int(x) for x in raw_tok_iter]
            if sync_gpu and not bool(getattr(next_tokens, "is_cuda", False)):
                _sync_cuda_if_present(getattr(self.pipeline, "device", None))
            now_s = time.perf_counter()
            batch_ms = (now_s - t0) * 1000.0

            for idx, task in enumerate(live_tasks):
                nxt_id = next_ids[idx]
                task.kv_cache = updated_caches[idx]
                task.curr_tokens = next_tokens[idx : idx + 1] if hasattr(next_tokens, "ndim") else nxt_id
                task._curr_tok_int = nxt_id
                task.pos_ids = task.pos_ids + 1
                task.token_timestamps_s.append(now_s)
                task.last_step_time_s = now_s
                reached_len = len(task.generated_token_ids) >= task.max_tokens
                hit_eos = task.end_token_id is not None and nxt_id == int(task.end_token_id)
                if reached_len or hit_eos:
                    task.finished = True

            n_served = len(live_tasks)
            if n_served > 1:
                self.batched_ar_yields += 1
            self.max_microbatch_size = max(self.max_microbatch_size, n_served)
            self.ar_steps_executed += n_served
            self.ar_compute_total_ms += batch_ms
            self._sync_edf_state()
            self.ema_ar_step_ms = float(
                self._lib.ms_edf_record_ar_batch(ctypes.c_void_p(self._edf_handle), ctypes.c_double(batch_ms))
            )
            self.last_effective_chunk_layers = self.compute_adaptive_chunk_layers()
            return n_served
        finally:
            self.in_ar_yield = False

    def _run_one_ar_step(self, task: ArDecodeTask, sync_gpu: bool = True) -> None:
        if task.finished:
            return
        self._execute_packed_ar_tasks([task], sync_gpu=sync_gpu)

    def _run_ar_microbatch(self, tasks: list[ArDecodeTask]) -> int:
        """Advance a microbatch of active AR tasks in `[B, D]` packed GEMM yield windows (capped at `max_microbatch_cap`)."""
        live = [t for t in tasks if not t.finished]
        if not live:
            return 0
        if getattr(self._run_one_ar_step, "__func__", None) is not UnifiedPreemptiveScheduler._run_one_ar_step:
            for t in live:
                self._run_one_ar_step(t)  # Supports unit-test overrides cleanly
            return len(live)
        cap = max(1, self.max_microbatch_cap)
        served_total = 0
        for off in range(0, len(live), cap):
            served_total += self._execute_packed_ar_tasks(live[off : off + cap], sync_gpu=True)
        return served_total

    def service_one_step_round(self) -> int:
        """Serve one AR decode token across all active tasks at a DiT step boundary."""
        self._admit_due_arrivals(time.perf_counter())
        batch_len = len(self.active_ar_tasks)
        if batch_len == 0:
            return 0
        batch: list[ArDecodeTask] = []
        for _ in range(batch_len):
            if not self.active_ar_tasks:
                break
            batch.append(self.active_ar_tasks.popleft())
        served = self._run_ar_microbatch(batch)
        for task in batch:
            self._finalize_task_if_done(task)
        self._admit_due_arrivals(time.perf_counter())
        if served > 0:
            self.yield_count += 1
        return served

    def service_due_ar_tasks(self, force_all_urgent: bool = False) -> int:
        """Execute urgent AR decode steps (EDF order, `[B, D]` microbatched) via C++ `PreemptEdfController`."""
        now_s = time.perf_counter()
        self._admit_due_arrivals(now_s)
        if not self.active_ar_tasks:
            return 0
        tasks_snapshot = list(self.active_ar_tasks)
        self.active_ar_tasks.clear()
        n = len(tasks_snapshot)
        dls = (ctypes.c_double * n)(*(t.next_deadline_s() for t in tasks_snapshot))
        due_idx = (ctypes.c_size_t * n)()
        def_idx = (ctypes.c_size_t * n)()
        due_cnt = ctypes.c_size_t(0)
        def_cnt = ctypes.c_size_t(0)
        self._sync_edf_state()
        self._lib.ms_edf_select_microbatch(
            ctypes.c_void_p(self._edf_handle),
            ctypes.c_double(now_s),
            dls,
            ctypes.c_size_t(n),
            1 if force_all_urgent else 0,
            due_idx,
            ctypes.byref(due_cnt),
            def_idx,
            ctypes.byref(def_cnt),
        )
        due = [tasks_snapshot[int(due_idx[i])] for i in range(int(due_cnt.value))]
        for i in range(int(def_cnt.value)):
            self.active_ar_tasks.append(tasks_snapshot[int(def_idx[i])])

        served = self._run_ar_microbatch(due)
        for task in due:
            self._finalize_task_if_done(task)
        self._admit_due_arrivals(time.perf_counter())
        if served > 0:
            self.yield_count += 1
        return served

    def drain_remaining_ar_tasks(self) -> None:
        """Finish any remaining or not-yet-arrived AR tasks after DiT completes via `[B, D]` packed GEMMs."""
        while self._pending_arrivals or self.active_ar_tasks:
            now_s = time.perf_counter()
            self._admit_due_arrivals(now_s)
            if not self.active_ar_tasks and self._pending_arrivals:
                rel_s, task = self._pending_arrivals.pop(0)
                task.arrival_time_s = max(now_s, self._run_t0_s + rel_s)
                task.last_step_time_s = task.arrival_time_s
                self.active_ar_tasks.append(task)
            if self.active_ar_tasks:
                batch = list(self.active_ar_tasks)
                self.active_ar_tasks.clear()
                self._run_ar_microbatch(batch)
                for task in batch:
                    self._finalize_task_if_done(task)
                self._admit_due_arrivals(time.perf_counter())

    def on_gen_layer_complete(self, layer_idx: int, num_layers: int, t_layer_start_s: float) -> None:
        """Invoked by `BagelExecutionInterceptor` after each `mode='gen'` decoder block."""
        dev = getattr(self.pipeline, "device", None)
        if (
            not self._layer_calibrated
            and not self._step0_h2d_streaming
            and len(self.layer_times_ms) < num_layers
            and not self.active_ar_tasks
            and not self._pending_arrivals
        ):
            _sync_cuda_if_present(dev)
            now_s = time.perf_counter()
            dt_l = (now_s - t_layer_start_s) * 1000.0
            self.layer_times_ms.append(dt_l)
            self._sync_edf_state()
            self.ema_layer_ms = float(
                self._lib.ms_edf_record_layer(ctypes.c_void_p(self._edf_handle), ctypes.c_double(dt_l))
            )
            if len(self.layer_times_ms) >= num_layers:
                self._layer_calibrated = True
            self.last_effective_chunk_layers = self.compute_adaptive_chunk_layers()
            self._last_sync_s = now_s
            self._queued_gpu_ms = 0.0
            return

        self._queued_gpu_ms += self.ema_layer_ms
        est_gpu_s = max(time.perf_counter(), self._last_sync_s + self._queued_gpu_ms / 1000.0)
        if self._should_yield_now(est_gpu_s, next_unit_ms=self.ema_layer_ms):
            _sync_cuda_if_present(dev)
            real_now = time.perf_counter()
            self._last_sync_s = real_now
            self._queued_gpu_ms = 0.0
            self._admit_due_arrivals(real_now)
            if self.active_ar_tasks:
                self.service_due_ar_tasks(force_all_urgent=False)
                self._last_sync_s = time.perf_counter()
                self._queued_gpu_ms = 0.0

    def on_dit_step_complete(self, step_start_s: float, ar_compute_before_ms: float) -> None:
        """Invoked by `BagelExecutionInterceptor` after each `bagel.forward` denoising step."""
        dev = getattr(self.pipeline, "device", None)
        _sync_cuda_if_present(dev)
        now_s = time.perf_counter()
        self._last_sync_s = now_s
        self._queued_gpu_ms = 0.0
        self._step0_h2d_streaming = False
        pure_step_ms = (now_s - step_start_s) * 1000.0 - (self.ar_compute_total_ms - ar_compute_before_ms)
        self.step_times_ms.append(pure_step_ms)

        if self.mode in ("step_preemptive", "layer_chunk"):
            self._admit_due_arrivals(now_s)
            if self.active_ar_tasks:
                if self.mode == "step_preemptive":
                    self.service_one_step_round()
                    self._last_sync_s = time.perf_counter()
                    self._queued_gpu_ms = 0.0
                elif self._should_yield_now(now_s, next_unit_ms=1.6 * self.ema_layer_ms):
                    self.service_due_ar_tasks(force_all_urgent=False)
                    self._last_sync_s = time.perf_counter()
                    self._queued_gpu_ms = 0.0

    @contextmanager
    def run_session(self):
        interceptor = BagelExecutionInterceptor.for_pipeline(self.pipeline)
        if self.pager is not None:
            interceptor.bind_pager(self.pager)
        with interceptor.generation_session(scheduler=self):
            yield self
