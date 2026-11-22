"""MorphServe Module 4: Deterministic Layer-Chunk & Step-Preemptive Unified AR+DiT Scheduler."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any, Callable

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

    Restores immediate `on_task_complete` mid-DiT ACK emission (`e27bce5`), A100-calibrated step/layer
    EMAs (`0805fc1`), and batched yield execution across concurrent active `ArDecodeTask`s.
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
        self.pager = pager
        self.external_arrival_poll = external_arrival_poll
        self.on_task_complete = on_task_complete

        self.active_ar_tasks: deque[ArDecodeTask] = deque()
        self.completed_ar_tasks: list[ArDecodeTask] = []
        self._pending_arrivals: list[tuple[float, ArDecodeTask]] = []
        self._run_t0_s: float = 0.0
        self._last_sync_s: float = 0.0
        self._queued_gpu_ms: float = 0.0

        # Telemetry
        self.yield_count: int = 0
        self.ar_steps_executed: int = 0
        self.ar_compute_total_ms: float = 0.0
        self.layer_times_ms: list[float] = []
        self.step_times_ms: list[float] = []
        self.last_effective_chunk_layers: int = self.compute_adaptive_chunk_layers()
        self.in_ar_yield: bool = False

    def reset_telemetry(self) -> None:
        self.yield_count = 0
        self.ar_steps_executed = 0
        self.ar_compute_total_ms = 0.0
        self.layer_times_ms = []
        self.step_times_ms = []
        self.last_effective_chunk_layers = self.compute_adaptive_chunk_layers()
        self.active_ar_tasks.clear()
        self.completed_ar_tasks.clear()
        self._pending_arrivals.clear()

    def begin_session_clock(self) -> None:
        now = time.perf_counter()
        self._run_t0_s = now
        self._last_sync_s = now
        self._queued_gpu_ms = 0.0
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
        """Analytical layer-chunk bound derived from deterministic layer time + AR step time."""
        slo = self.tpot_slo_ms if tpot_slo_ms is None else float(tpot_slo_ms)
        budget_ms = max(0.0, slo - self.ema_ar_step_ms - self.guard_ms)
        num_layers = len(self.lm_model.layers)
        if self.ema_layer_ms <= 0.0:
            return 1
        k = int(budget_ms // self.ema_layer_ms)
        return max(1, min(num_layers, k))

    def _should_yield_now(self, now_s: float, next_unit_ms: float) -> bool:
        self._admit_due_arrivals(now_s)
        if not self.active_ar_tasks:
            return False
        earliest = self._earliest_deadline_s()
        assert earliest is not None
        predicted_finish_ms = (now_s - earliest) * 1000.0 + next_unit_ms + self.ema_ar_step_ms + self.guard_ms
        return predicted_finish_ms >= 0.0

    def _finalize_task_if_done(self, task: ArDecodeTask) -> None:
        if task.finished:
            self.completed_ar_tasks.append(task)
            if self.on_task_complete is not None:
                self.on_task_complete(task)
        else:
            self.active_ar_tasks.append(task)

    def _run_one_ar_step(self, task: ArDecodeTask, sync_gpu: bool = True) -> None:
        import torch

        if task.finished:
            return
        t0 = time.perf_counter()
        self.in_ar_yield = True
        try:
            task.generated_token_ids.append(int(task.curr_tokens[0].item()))
            query_lens = torch.ones_like(task.curr_tokens)
            out = self.bagel.language_model(
                packed_text_ids=task.curr_tokens,
                query_lens=query_lens,
                packed_query_position_ids=task.pos_ids,
                past_key_values=task.kv_cache,
                update_past_key_values=True,
                is_causal=True,
                mode="und",
            )
            task.kv_cache = out.past_key_values
            logits = self.lm_head(out.packed_query_sequence)
            next_tok = torch.argmax(logits, dim=-1)
            task.curr_tokens = next_tok
            task.pos_ids = task.pos_ids + 1
            if sync_gpu:
                _sync_cuda_if_present()
            now_s = time.perf_counter()
            step_ms = (now_s - t0) * 1000.0

            task.token_timestamps_s.append(now_s)
            task.last_step_time_s = now_s
            self.ar_steps_executed += 1
            self.ar_compute_total_ms += step_ms
            self.ema_ar_step_ms = 0.5 * self.ema_ar_step_ms + 0.5 * step_ms
            self.last_effective_chunk_layers = self.compute_adaptive_chunk_layers()

            reached_len = len(task.generated_token_ids) >= task.max_tokens
            hit_eos = task.end_token_id is not None and int(next_tok[0].item()) == int(task.end_token_id)
            if reached_len or hit_eos:
                task.finished = True
        finally:
            self.in_ar_yield = False

    def _run_ar_microbatch(self, tasks: list[ArDecodeTask]) -> int:
        """Advance a microbatch of active AR tasks in one synchronized yield window."""
        if not tasks:
            return 0
        n = len(tasks)
        for idx, t in enumerate(tasks):
            # Only synchronize on the final task of the microbatch when running real steps
            if getattr(self._run_one_ar_step, "__func__", None) is UnifiedPreemptiveScheduler._run_one_ar_step:
                self._run_one_ar_step(t, sync_gpu=(idx == n - 1))
            else:
                self._run_one_ar_step(t)  # Supports unit-test overrides cleanly
        return n

    def service_one_step_round(self) -> int:
        """Serve one AR decode token across all active tasks at a DiT step boundary."""
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
        """Execute urgent AR decode steps (EDF order, microbatched) during a DiT preemption boundary."""
        if not self.active_ar_tasks:
            return 0
        now_s = time.perf_counter()
        ordered = sorted(self.active_ar_tasks, key=lambda t: t.next_deadline_s())
        self.active_ar_tasks.clear()

        due: list[ArDecodeTask] = []
        deferred: list[ArDecodeTask] = []
        max_serve = max(self.max_ar_tokens_per_yield, len(ordered))
        for idx, task in enumerate(ordered):
            slack_ms = (task.next_deadline_s() - now_s) * 1000.0
            if force_all_urgent or idx == 0 or (
                len(due) < max_serve and slack_ms <= (self.ema_ar_step_ms + self.guard_ms)
            ):
                due.append(task)
            else:
                deferred.append(task)

        for d in deferred:
            self.active_ar_tasks.append(d)

        served = self._run_ar_microbatch(due)
        for task in due:
            self._finalize_task_if_done(task)
        self._admit_due_arrivals(time.perf_counter())
        if served > 0:
            self.yield_count += 1
        return served

    def drain_remaining_ar_tasks(self) -> None:
        """Finish any remaining or not-yet-arrived AR tasks after DiT completes."""
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
        if len(self.layer_times_ms) < num_layers and not self.active_ar_tasks and not self._pending_arrivals:
            _sync_cuda_if_present()
            now_s = time.perf_counter()
            dt_l = (now_s - t_layer_start_s) * 1000.0
            self.layer_times_ms.append(dt_l)
            self.ema_layer_ms = 0.7 * self.ema_layer_ms + 0.3 * dt_l
            self.last_effective_chunk_layers = self.compute_adaptive_chunk_layers()
            self._last_sync_s = now_s
            self._queued_gpu_ms = 0.0
            return

        self._queued_gpu_ms += self.ema_layer_ms
        est_gpu_s = max(time.perf_counter(), self._last_sync_s + self._queued_gpu_ms / 1000.0)
        if self._should_yield_now(est_gpu_s, next_unit_ms=self.ema_layer_ms):
            _sync_cuda_if_present()
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
        _sync_cuda_if_present()
        now_s = time.perf_counter()
        self._last_sync_s = now_s
        self._queued_gpu_ms = 0.0
        pure_step_ms = (now_s - step_start_s) * 1000.0 - (self.ar_compute_total_ms - ar_compute_before_ms)
        self.step_times_ms.append(pure_step_ms)

        if self.mode in ("step_preemptive", "layer_chunk"):
            self._admit_due_arrivals(now_s)
            if self.active_ar_tasks:
                if self.mode == "step_preemptive":
                    self.service_one_step_round()
                    self._last_sync_s = time.perf_counter()
                    self._queued_gpu_ms = 0.0
                elif self._should_yield_now(now_s, next_unit_ms=self.ema_layer_ms):
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
