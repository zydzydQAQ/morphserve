"""Offline unit tests for Module 4 (Adaptive Layer-Chunk & Deadline-Driven Preemptive Scheduler)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
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
    def test_adaptive_chunk_layers_and_deadline_yield_math(self) -> None:
        pipe = _MockPipeline()
        # 512px: t_layer=8.5ms, t_AR=20ms, guard=4ms -> at 60ms SLO budget=36ms -> 4 layers (~34ms)!
        sched = UnifiedPreemptiveScheduler(
            pipe, mode="layer_chunk", tpot_slo_ms=60.0, guard_ms=4.0, init_layer_ms=8.5, init_ar_step_ms=20.0
        )
        self.assertEqual(sched.compute_adaptive_chunk_layers(60.0), 4)
        self.assertEqual(sched.compute_adaptive_chunk_layers(100.0), 8)
        # 1024px: t_layer=32ms -> at 60ms SLO -> 1 layer (~32ms)!
        sched.ema_layer_ms = 32.0
        self.assertEqual(sched.compute_adaptive_chunk_layers(60.0), 1)

        # Verify deadline admission & trigger
        sched._run_t0_s = 100.0
        task = ArDecodeTask(
            req_id="r1", kv_cache=None, curr_tokens=None, pos_ids=None, max_tokens=16, end_token_id=None, tpot_slo_ms=60.0
        )
        sched.schedule_ar_arrival(500.0, task)
        # Before arrival (t=100.4s): no yield
        self.assertFalse(sched._should_yield_now(100.400, next_unit_ms=8.5))
        # Right after arrival (t=100.510s, deadline=100.560s, remaining=50ms > 8.5+20+4=32.5ms): don't yield yet!
        self.assertFalse(sched._should_yield_now(100.510, next_unit_ms=8.5))
        # Approaching deadline (t=100.530s, remaining=30ms <= 32.5ms): YIELD NOW!
        self.assertTrue(sched._should_yield_now(100.530, next_unit_ms=8.5))

    def test_module4_script_syntax_and_imports(self) -> None:
        script = REPO_ROOT / "deploy/kairos/scripts/06_phase1_module4_validate.sh"
        proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        for rel in (
            "src/morphserve/worker/__init__.py",
            "src/morphserve/worker/preemptive_scheduler.py",
            "src/morphserve/bench/validate_module4_preemption.py",
        ):
            p = REPO_ROOT / rel
            res = subprocess.run([sys.executable, "-m", "py_compile", str(p)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, msg=f"py_compile failed on {rel}: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
