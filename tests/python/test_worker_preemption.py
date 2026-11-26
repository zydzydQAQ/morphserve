"""Seam tests for `BagelExecutionInterceptor`, `UnifiedPreemptiveScheduler`, `PreemptionSession`, and `ArDecodeTask`."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from morphserve.models import VmmOutOfMemoryError
from morphserve.worker import (
    ArDecodeTask,
    BagelExecutionInterceptor,
    PreemptionSession,
    UnifiedPreemptiveScheduler,
)


def _make_pure_python_stub_pipeline(num_layers: int = 28) -> Any:
    """Pure-Python stub pipeline (no `torch` import) for C++ EDF scheduler & `PreemptionSession` seam tests."""
    lm_model = SimpleNamespace(layers=[SimpleNamespace() for _ in range(num_layers)], forward=lambda **kw: None)
    lm_outer = SimpleNamespace(model=lm_model, lm_head=lambda x: x)
    bagel = SimpleNamespace(language_model=lm_outer, forward=lambda *a, **kw: None)
    return SimpleNamespace(device="cpu", bagel=bagel)


# ==================== 1. EDF Adaptive Chunking & Slack-Coalesced Microbatching ====================


def test_unified_preemptive_scheduler_edf_adaptive_chunking_and_slack_coalescing() -> None:
    pipe = _make_pure_python_stub_pipeline(num_layers=28)
    executed_steps: list[str] = []

    def _step_exec(task: ArDecodeTask) -> None:
        executed_steps.append(task.req_id)
        task.generated_token_ids.append(len(task.generated_token_ids) + 1)
        if len(task.generated_token_ids) >= task.max_tokens:
            task.finished = True

    sched = UnifiedPreemptiveScheduler(
        pipe,
        mode="layer_chunk",
        tpot_slo_ms=125.6,
        guard_ms=14.0,
        init_layer_ms=10.5,
        init_ar_step_ms=40.0,
        max_microbatch_cap=4,
        step_executor=_step_exec,
    )

    # Adaptive chunk sizing backed live by C++ PreemptEdfController
    k_base = sched.compute_adaptive_chunk_layers()
    assert 1 <= k_base <= 28
    k_tight = sched.compute_adaptive_chunk_layers(tpot_slo_ms=60.0)
    k_loose = sched.compute_adaptive_chunk_layers(tpot_slo_ms=400.0)
    assert k_tight <= k_base <= k_loose

    # Property write-through updates C++ PreemptEdfController live.
    sched.ema_layer_ms = 5.0
    assert abs(sched.ema_layer_ms - 5.0) < 1e-6
    assert sched.compute_adaptive_chunk_layers() >= k_base
    sched.guard_ms = 10.0
    assert abs(sched.guard_ms - 10.0) < 1e-6

    # Slack coalescing: urgent task + near-deadline task coalesce in one microbatch, while
    # a far-future task with ample slack is deferred until due or drained.
    t0 = sched.begin_session_clock()
    urgent = ArDecodeTask(
        req_id="req_urgent",
        kv_cache=None,
        curr_tokens=None,
        pos_ids=None,
        max_tokens=2,
        end_token_id=None,
        tpot_slo_ms=10.0,
        arrival_time_s=t0 - 0.05,
        last_step_time_s=t0 - 0.05,
    )
    near_urgent = ArDecodeTask(
        req_id="req_near",
        kv_cache=None,
        curr_tokens=None,
        pos_ids=None,
        max_tokens=2,
        end_token_id=None,
        tpot_slo_ms=15.0,
        arrival_time_s=t0 - 0.01,
        last_step_time_s=t0 - 0.01,
    )
    slack_task = ArDecodeTask(
        req_id="req_slack",
        kv_cache=None,
        curr_tokens=None,
        pos_ids=None,
        max_tokens=1,
        end_token_id=None,
        tpot_slo_ms=5000.0,
        arrival_time_s=t0 + 10.0,
        last_step_time_s=t0 + 10.0,
    )
    sched.schedule_ar_arrival(0.0, urgent)
    sched.schedule_ar_arrival(0.0, near_urgent)
    sched.schedule_ar_arrival(0.0, slack_task)
    admitted = sched.admit_due_arrivals(t0 + 0.001)
    assert admitted == 3

    n_serviced = sched.service_due_ar_tasks(force_all_urgent=False)
    assert n_serviced == 2
    assert sched.batched_ar_yields >= 1
    assert sched.max_microbatch_size == 2
    assert "req_slack" not in executed_steps

    # Drain completes every remaining token across urgent, near_urgent, and slack_task.
    sched.drain_remaining_ar_tasks()
    assert urgent.finished and len(urgent.generated_token_ids) == 2
    assert near_urgent.finished and len(near_urgent.generated_token_ids) == 2
    assert slack_task.finished and len(slack_task.generated_token_ids) == 1
    assert len(sched.completed_ar_tasks) == 3


# ==================== 2. PreemptionSession Poll, Heavy Deferral, Drain & Abort ====================


def test_preemption_session_heavy_cmd_deferral_drain_and_oom_abort() -> None:
    pipe = _make_pure_python_stub_pipeline(num_layers=8)
    fail_on_step = {"boom": False}

    def _step_exec(task: ArDecodeTask) -> None:
        if fail_on_step["boom"]:
            raise VmmOutOfMemoryError("injected preemption step OOM")
        task.generated_token_ids.append(1)
        if len(task.generated_token_ids) >= task.max_tokens:
            task.finished = True

    sched = UnifiedPreemptiveScheduler(
        pipe,
        mode="layer_chunk",
        tpot_slo_ms=100.0,
        guard_ms=10.0,
        init_layer_ms=8.0,
        init_ar_step_ms=20.0,
        step_executor=_step_exec,
    )

    # Scenario A: lightweight text cmd admitted immediately; heavy image cmd deferred while AR active,
    # then admitted & completed during `drain()`.
    pending_inbox = [
        {"cmd_id": "cmd_text", "ctx_id": "c_text", "max_ar_tokens": 2},
        {"cmd_id": "cmd_heavy", "ctx_id": "c_img", "image_paths": ["photo.png"], "max_ar_tokens": 1},
    ]
    admitted_order: list[str] = []
    completed_ids: list[str] = []

    def _poll() -> list[dict[str, Any]]:
        out = list(pending_inbox)
        pending_inbox.clear()
        return out

    def _admit(cmd: dict[str, Any], now_s: float) -> tuple[ArDecodeTask, str]:
        admitted_order.append(str(cmd["cmd_id"]))
        task = ArDecodeTask(
            req_id=str(cmd["cmd_id"]),
            kv_cache=None,
            curr_tokens=None,
            pos_ids=None,
            max_tokens=int(cmd["max_ar_tokens"]),
            end_token_id=None,
            tpot_slo_ms=100.0,
            arrival_time_s=now_s,
            last_step_time_s=now_s,
        )
        return task, str(cmd["cmd_id"])

    with PreemptionSession(
        sched,
        poll_cmds=_poll,
        admit_cmd=_admit,
        on_complete=lambda _t, h: completed_ids.append(h),
    ) as ps:
        t_now = sched.begin_session_clock()
        # First poll admits ONLY the lightweight text command; heavy image cmd is deferred!
        sched.admit_due_arrivals(t_now)
        assert admitted_order == ["cmd_text"]
        ps.drain(stop_polling_after=False)
        assert admitted_order == ["cmd_text", "cmd_heavy"]
        assert completed_ids == ["cmd_text", "cmd_heavy"]

    assert sched.external_arrival_poll is None
    assert sched.on_task_complete is None

    # Scenario B: OOM during drain cleanly aborts both active and deferred commands!
    pending_inbox.extend(
        [
            {"cmd_id": "cmd_oom_active", "ctx_id": "c1", "max_ar_tokens": 2},
            {"cmd_id": "cmd_oom_deferred", "ctx_id": "c2", "image_paths": ["img2.png"], "max_ar_tokens": 1},
        ]
    )
    aborted_handles: list[tuple[str, BaseException | None]] = []
    aborted_deferred: list[str] = []
    fail_on_step["boom"] = True

    with PreemptionSession(
        sched,
        poll_cmds=_poll,
        admit_cmd=_admit,
        on_complete=lambda _t, h: completed_ids.append(h),
        on_abort=lambda h, exc: aborted_handles.append((h, exc)),
        on_abort_deferred=lambda cmd: aborted_deferred.append(str(cmd["cmd_id"])),
    ):
        sched.admit_due_arrivals(sched.begin_session_clock())
        # Exiting triggers drain -> OOM -> abort_all for active & deferred commands!

    assert len(aborted_handles) == 1
    assert aborted_handles[0][0] == "cmd_oom_active"
    assert isinstance(aborted_handles[0][1], VmmOutOfMemoryError)
    assert aborted_deferred == ["cmd_oom_deferred"]
    assert not sched.active_ar_tasks


# ==================== 3. BagelExecutionInterceptor Hooks, Cache Isolation & Packed `[B, D]` Equivalence ====================


def _build_packed_ar_transformer_stub(torch: Any, *, seed: int = 123, num_layers: int = 2, hidden: int = 16, num_heads: int = 4, num_kv_heads: int = 2):
    """Deterministic CPU Transformer stub supporting `BagelExecutionInterceptor.forward_ar_step_batch` `[B, D]` GEMM."""
    torch.manual_seed(seed)
    head_dim = hidden // num_heads
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    class _StubCache:
        def __init__(self, n_layers: int) -> None:
            self.num_layers = n_layers
            self.key_cache: dict[int, Any] = {i: None for i in range(n_layers)}
            self.value_cache: dict[int, Any] = {i: None for i in range(n_layers)}

    class _RotaryEmb(torch.nn.Module):
        def forward(self, x: Any, pos_ids: Any) -> tuple[Any, Any]:
            # pos_ids: [1, B] -> cos/sin [1, B, head_dim]
            angles = pos_ids.to(torch.float32).unsqueeze(-1) * torch.linspace(0.1, 1.0, head_dim).view(1, 1, -1)
            return torch.cos(angles), torch.sin(angles)

    class _TupleLinear(torch.nn.Module):
        def __init__(self, in_f: int, out_f: int) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(in_f, out_f, bias=False)

        def forward(self, x: Any, *args: Any, **kwargs: Any) -> tuple[Any, None]:
            return self.lin(x), None

    class _StubAttn(torch.nn.Module):
        def __init__(self, layer_idx: int) -> None:
            super().__init__()
            self.layer_idx = layer_idx
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.q_size = q_size
            self.kv_size = kv_size
            self.hidden_size = hidden
            self.qkv_proj = _TupleLinear(hidden, q_size + 2 * kv_size)
            self.o_proj = _TupleLinear(q_size, hidden)
            self.q_norm = torch.nn.LayerNorm(head_dim)
            self.k_norm = torch.nn.LayerNorm(head_dim)

        def rotary_op(self, x: Any, cos_h: Any, sin_h: Any) -> Any:
            # x: [1, B, H, D], cos_h/sin_h: [B, D//2]
            x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
            c = cos_h.unsqueeze(0).unsqueeze(2)
            s = sin_h.unsqueeze(0).unsqueeze(2)
            return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)

        def _forward_und(self, packed_query_sequence, packed_query_position_embeddings, past_key_values, is_causal, update_past_key_values=True):
            return packed_query_sequence, past_key_values

        def _forward_gen(self, packed_query_sequence, query_lens, packed_query_position_embeddings, past_key_values, packed_vae_token_indexes, packed_text_indexes, update_past_key_values=False):
            return packed_query_sequence, past_key_values

    class _StubDecoderLayer(torch.nn.Module):
        def __init__(self, layer_idx: int) -> None:
            super().__init__()
            self.input_layernorm = torch.nn.LayerNorm(hidden)
            self.self_attn = _StubAttn(layer_idx)
            self.post_attention_layernorm = torch.nn.LayerNorm(hidden)
            self.mlp = torch.nn.Sequential(torch.nn.Linear(hidden, hidden, bias=False), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden, bias=False))

    class _StubLMModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(64, hidden)
            self.rotary_emb = _RotaryEmb()
            self.layers = torch.nn.ModuleList([_StubDecoderLayer(i) for i in range(num_layers)])
            self.norm = torch.nn.LayerNorm(hidden)

        def forward(self, **kwargs: Any) -> Any:
            return kwargs

    lm_model = _StubLMModel()
    lm_outer = SimpleNamespace(model=lm_model, lm_head=torch.nn.Linear(hidden, 64, bias=False))
    bagel = SimpleNamespace(language_model=lm_outer, forward=lambda *a, **kw: None)
    pipe = SimpleNamespace(device=torch.device("cpu"), bagel=bagel)
    return pipe, _StubCache


def test_bagel_interceptor_hooks_workspace_isolation_and_packed_ar_equivalence() -> None:
    torch = pytest.importorskip("torch")

    pipe_a, cache_cls = _build_packed_ar_transformer_stub(torch, seed=101)
    pipe_b, _ = _build_packed_ar_transformer_stub(torch, seed=202)

    orig_lm_fwd_a = pipe_a.bagel.language_model.model.forward
    orig_und_a0 = pipe_a.bagel.language_model.model.layers[0].self_attn._forward_und
    orig_gen_a0 = pipe_a.bagel.language_model.model.layers[0].self_attn._forward_gen

    ic_a = BagelExecutionInterceptor.for_pipeline(pipe_a)
    ic_b = BagelExecutionInterceptor.for_pipeline(pipe_b)
    assert ic_a is not ic_b

    # Install + CFG enable + scoped cascade override + clean uninstall restores exact original methods.
    ic_a.install()
    ic_a.enable_shared_prefix_cfg(min_shared_tokens=16, allow_cascade_split=False)
    assert ic_a.cfg_enabled is True
    assert ic_a.cfg_allow_cascade_split is False
    with ic_a.shared_prefix_cfg_scope(allow_cascade_split=True):
        assert ic_a.cfg_allow_cascade_split is True
    assert ic_a.cfg_allow_cascade_split is False

    # Packed [B, D] AR equivalence: B=2 batched logits & updated KV must match B=1 solo execution!
    cache_0_solo = cache_cls(2)
    cache_1_solo = cache_cls(2)
    for lidx in range(2):
        # Distinct prefix KV lengths (3 tokens for req 0, 5 tokens for req 1)
        torch.manual_seed(10 + lidx)
        cache_0_solo.key_cache[lidx] = torch.randn((3, 2, 4), dtype=torch.float32)
        cache_0_solo.value_cache[lidx] = torch.randn((3, 2, 4), dtype=torch.float32)
        cache_1_solo.key_cache[lidx] = torch.randn((5, 2, 4), dtype=torch.float32)
        cache_1_solo.value_cache[lidx] = torch.randn((5, 2, 4), dtype=torch.float32)

    cache_0_batch = deepcopy(cache_0_solo)
    cache_1_batch = deepcopy(cache_1_solo)

    tok0 = torch.tensor([7], dtype=torch.long)
    pos0 = torch.tensor([3], dtype=torch.long)
    tok1 = torch.tensor([19], dtype=torch.long)
    pos1 = torch.tensor([5], dtype=torch.long)

    ic_a.reset_ar_telemetry()
    with torch.inference_mode():
        logits_0_solo, _ = ic_a.forward_ar_step_batch([tok0], [pos0], [cache_0_solo], return_logits=True)
        logits_1_solo, _ = ic_a.forward_ar_step_batch([tok1], [pos1], [cache_1_solo], return_logits=True)
        logits_batched, _ = ic_a.forward_ar_step_batch(
            [tok0, tok1], [pos0, pos1], [cache_0_batch, cache_1_batch], return_logits=True
        )

    assert torch.allclose(logits_batched[0], logits_0_solo[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(logits_batched[1], logits_1_solo[0], atol=1e-5, rtol=1e-5)
    for lidx in range(2):
        assert torch.allclose(cache_0_batch.key_cache[lidx], cache_0_solo.key_cache[lidx], atol=1e-5)
        assert torch.allclose(cache_1_batch.value_cache[lidx], cache_1_solo.value_cache[lidx], atol=1e-5)
    assert ic_a.max_ar_batch_width_seen == 2
    assert ic_a.ar_batched_tokens_total == 4

    # Clearing/uninstalling ic_a must restore pipe_a's methods without impacting ic_b!
    ic_b.install()
    ic_a.clear_workspace_caches()
    ic_a.uninstall()
    assert pipe_a.bagel.language_model.model.forward == orig_lm_fwd_a
    assert pipe_a.bagel.language_model.model.layers[0].self_attn._forward_und == orig_und_a0
    assert pipe_a.bagel.language_model.model.layers[0].self_attn._forward_gen == orig_gen_a0

    with torch.inference_mode():
        logits_b, _ = ic_b.forward_ar_step_batch([tok0], [pos0], [cache_cls(2)], return_logits=True)
    assert tuple(logits_b.shape) == (1, 64)
    ic_b.uninstall()
