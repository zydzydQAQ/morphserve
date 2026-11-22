"""Offline unit tests for Module 4 (Adaptive Layer-Chunk & Deadline-Driven Preemptive Scheduler)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from morphserve.worker import ArDecodeTask, UnifiedPreemptiveScheduler  # noqa: E402


class _MockModel:
    def __init__(self, num_layers: int = 28):
        self.layers = [object() for _ in range(num_layers)]

    def forward(self, *args, **kwargs):
        return None


class _MockLM:
    def __init__(self):
        self.model = _MockModel(28)
        self.lm_head = object()


class _MockBagel:
    def __init__(self):
        self.language_model = _MockLM()

    def forward(self, *args, **kwargs):
        return None


class _MockPipeline:
    def __init__(self):
        self.bagel = _MockBagel()


class TestModule4Offline(unittest.TestCase):
    def test_calibrated_defaults_and_adaptive_chunk_layers(self) -> None:
        pipe = _MockPipeline()
        default_sched = UnifiedPreemptiveScheduler(pipe)
        self.assertEqual(default_sched.guard_ms, 14.0)
        self.assertEqual(default_sched.ema_ar_step_ms, 66.0)

        sched = UnifiedPreemptiveScheduler(
            pipe, mode="layer_chunk", tpot_slo_ms=60.0, guard_ms=4.0, init_layer_ms=8.5, init_ar_step_ms=20.0
        )
        self.assertEqual(sched.compute_adaptive_chunk_layers(60.0), 4)
        self.assertEqual(sched.compute_adaptive_chunk_layers(100.0), 8)
        sched.ema_layer_ms = 32.0
        self.assertEqual(sched.compute_adaptive_chunk_layers(60.0), 1)

        sched._run_t0_s = 100.0
        task = ArDecodeTask(
            req_id="r1", kv_cache=None, curr_tokens=None, pos_ids=None, max_tokens=16, end_token_id=None, tpot_slo_ms=60.0
        )
        sched.schedule_ar_arrival(500.0, task)
        self.assertFalse(sched._should_yield_now(100.400, next_unit_ms=8.5))
        self.assertFalse(sched._should_yield_now(100.510, next_unit_ms=8.5))
        self.assertTrue(sched._should_yield_now(100.530, next_unit_ms=8.5))

    def test_step_preemptive_serves_at_most_one_token_per_active_task(self) -> None:
        pipe = _MockPipeline()
        sched = UnifiedPreemptiveScheduler(pipe, mode="step_preemptive", tpot_slo_ms=130.0)
        for rid in ("a1", "a2"):
            sched.active_ar_tasks.append(
                ArDecodeTask(
                    req_id=rid,
                    kv_cache=None,
                    curr_tokens=None,
                    pos_ids=None,
                    max_tokens=5,
                    end_token_id=None,
                    tpot_slo_ms=130.0,
                )
            )
        ran: list[str] = []

        def fake_step(task: ArDecodeTask) -> None:
            ran.append(task.req_id)

        sched._run_one_ar_step = fake_step  # type: ignore[method-assign]
        served = sched.service_one_step_round()
        self.assertEqual(served, 2)
        self.assertEqual(ran, ["a1", "a2"])
        self.assertEqual(sched.yield_count, 1)

    def test_mid_dit_on_task_complete_fires_immediately_before_dit_ends(self) -> None:
        pipe = _MockPipeline()
        events: list[str] = []

        sched = UnifiedPreemptiveScheduler(
            pipe,
            mode="layer_chunk",
            tpot_slo_ms=80.0,
            guard_ms=10.0,
            init_layer_ms=10.0,
            init_ar_step_ms=20.0,
            on_task_complete=lambda t: events.append(f"ack:{t.req_id}"),
        )
        now = time.perf_counter()
        for rid in ("ar_fast_1", "ar_fast_2"):
            sched.active_ar_tasks.append(
                ArDecodeTask(
                    req_id=rid,
                    kv_cache=None,
                    curr_tokens=None,
                    pos_ids=None,
                    max_tokens=2,
                    end_token_id=None,
                    tpot_slo_ms=80.0,
                    arrival_time_s=now - 0.090,
                    last_step_time_s=now - 0.090,
                )
            )

        def fake_step(task: ArDecodeTask) -> None:
            task.generated_token_ids.append(len(task.generated_token_ids) + 1)
            task.token_timestamps_s.append(time.perf_counter())
            task.last_step_time_s = task.token_timestamps_s[-1]
            if len(task.generated_token_ids) >= task.max_tokens:
                task.finished = True

        sched._run_one_ar_step = fake_step  # type: ignore[method-assign]
        # Yield 1: both urgent tasks advance 1 token in a single microbatch
        self.assertEqual(sched.service_due_ar_tasks(force_all_urgent=True), 2)
        self.assertEqual(events, [])
        # Yield 2 (still mid-DiT!): both finish token 2 and emit immediate completion callbacks BEFORE DiT ends
        self.assertEqual(sched.service_due_ar_tasks(force_all_urgent=True), 2)
        events.append("dit_finished_later")
        self.assertEqual(events, ["ack:ar_fast_1", "ack:ar_fast_2", "dit_finished_later"])

    def test_module4_script_syntax_and_imports(self) -> None:
        script = REPO_ROOT / "deploy/kairos/scripts/06_phase1_module4_validate.sh"
        proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        for rel in (
            "src/morphserve/worker/__init__.py",
            "src/morphserve/worker/bagel_interceptor.py",
            "src/morphserve/worker/preemptive_scheduler.py",
            "src/morphserve/bench/validate_module4_preemption.py",
        ):
            p = REPO_ROOT / rel
            res = subprocess.run([sys.executable, "-m", "py_compile", str(p)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, msg=f"py_compile failed on {rel}: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
