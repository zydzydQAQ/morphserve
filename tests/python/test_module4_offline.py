"""Offline unit tests for Module 4 (`[B, D]` Batched AR, Preemption, Host KV Swap, Multi-Turn Pack & InternVL-U Plan)."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import subprocess
import sys
import time
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "python"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from morphserve.engine.config import EnginePolicyConfig  # noqa: E402
from morphserve.kernels import cow_fork_context  # noqa: E402
from morphserve.models import (  # noqa: E402
    SyntheticCPTGAdapter,
    bagel_ctx_nbytes,
    bagel_entry_nbytes,
    compute_internvlu_memory_plan,
    pack_bagel_entry_to_pinned_host,
    pack_bagel_multi_context,
    unpack_bagel_entry_from_pinned_host,
    unpack_bagel_multi_context,
)
from morphserve.models.internvl_u_adapter import _trim_internvlu_generated_tokens  # noqa: E402
from morphserve.vmm import SimulatedTriPoolBackend, VmmNaiveCache  # noqa: E402
from morphserve.worker import (  # noqa: E402
    ArDecodeTask,
    BagelExecutionInterceptor,
    UnifiedPreemptiveScheduler,
    append_or_update_kv_cache,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

MIB = 1024**2


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


class _SimpleNaiveCache:
    def __init__(self, num_layers: int):
        self.key_cache = {i: None for i in range(num_layers)}
        self.value_cache = {i: None for i in range(num_layers)}
        self.key_values_lens = None

    @property
    def num_layers(self) -> int:
        return len(self.key_cache)


if HAS_TORCH:

    class _MiniRMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.linspace(0.8, 1.2, dim))
            self.eps = eps

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            orig_dt = x.dtype
            xf = x.float()
            var = xf.pow(2).mean(dim=-1, keepdim=True)
            return (xf * torch.rsqrt(var + self.eps) * self.weight.float()).to(orig_dt)

    class _MiniRotaryEmb(nn.Module):
        def __init__(self, head_dim: int, theta: float = 10000.0):
            super().__init__()
            inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            inv_exp = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            pos_exp = position_ids[:, None, :].float()
            freqs = (inv_exp @ pos_exp).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    class _MiniRotaryOp(nn.Module):
        def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            # x: (1, S, H, D), cos/sin: (S, D/2)
            d_half = x.shape[-1] // 2
            x1 = x[..., :d_half]
            x2 = x[..., d_half:]
            c = cos.unsqueeze(0).unsqueeze(2)
            s = sin.unsqueeze(0).unsqueeze(2)
            return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

    class _TupleLinear(nn.Module):
        def __init__(self, in_f: int, out_f: int, bias: bool = True):
            super().__init__()
            self.lin = nn.Linear(in_f, out_f, bias=bias)
            self.calls = 0
            self.batch_widths: list[int] = []

        def forward(self, x: torch.Tensor):
            self.calls += 1
            self.batch_widths.append(int(x.shape[0]))
            return self.lin(x), None

    class _MiniDiffusionAttn(nn.Module):
        def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, causal: bool):
            super().__init__()
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.causal = causal

        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, meta=None) -> torch.Tensor:
            out = F.scaled_dot_product_attention(
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
                is_causal=self.causal,
                scale=1.0 / math.sqrt(self.head_dim),
                enable_gqa=(self.num_heads != self.num_kv_heads),
            )
            return out.permute(0, 2, 1, 3)

    class _MiniPackedAttentionMoT(nn.Module):
        """Exact miniature replica of `vllm_omni` `PackedAttentionMoT._forward_und`."""

        def __init__(self, layer_idx: int, hidden_size: int = 64, num_heads: int = 4, num_kv_heads: int = 2):
            super().__init__()
            self.layer_idx = layer_idx
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = hidden_size // num_heads
            self.q_size = num_heads * self.head_dim
            self.kv_size = num_kv_heads * self.head_dim
            self.qkv_proj = _TupleLinear(hidden_size, self.q_size + 2 * self.kv_size, bias=True)
            self.o_proj = _TupleLinear(self.q_size, hidden_size, bias=False)
            self.q_norm = _MiniRMSNorm(self.head_dim)
            self.k_norm = _MiniRMSNorm(self.head_dim)
            self.rotary_op = _MiniRotaryOp()
            self.attn_causal = _MiniDiffusionAttn(num_heads, num_kv_heads, self.head_dim, causal=True)
            self.attn_noncausal = _MiniDiffusionAttn(num_heads, num_kv_heads, self.head_dim, causal=False)
            self.stock_mask_allocs = 0

        def _forward_und(
            self,
            packed_query_sequence: torch.Tensor,
            packed_query_position_embeddings: tuple[torch.Tensor, torch.Tensor],
            past_key_values,
            is_causal: bool,
            update_past_key_values: bool = True,
        ):
            qkv, _ = self.qkv_proj(packed_query_sequence)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
            q = self.q_norm(q.to(torch.float32))
            k = self.k_norm(k.to(torch.float32))
            cos, sin = [x[..., : self.head_dim // 2] for x in packed_query_position_embeddings]
            q = self.rotary_op(q.to(cos.dtype).unsqueeze(0), cos, sin).squeeze(0)
            k = self.rotary_op(k.to(cos.dtype).unsqueeze(0), cos, sin).squeeze(0)
            q = q.to(cos.dtype)
            k = k.to(cos.dtype)
            v = v.to(cos.dtype)

            if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
                ck = past_key_values.key_cache[self.layer_idx]
                cv = past_key_values.value_cache[self.layer_idx]
                full_k = torch.cat([ck, k], dim=0)
                full_v = torch.cat([cv, v], dim=0)
                cache_len = ck.shape[0]
            else:
                full_k, full_v, cache_len = k, v, 0

            if is_causal and cache_len > 0:
                self.stock_mask_allocs += 1
                q_len = q.shape[0]
                k_len = full_k.shape[0]
                aq = torch.arange(q_len, device=q.device).unsqueeze(1)
                ak = torch.arange(k_len, device=q.device).unsqueeze(0)
                mask = ak > (cache_len + aq)
                attn_bias = torch.zeros(q_len, k_len, dtype=q.dtype, device=q.device)
                attn_bias.masked_fill_(mask, float("-inf"))
                attn_out_4d = F.scaled_dot_product_attention(
                    q.unsqueeze(0).permute(0, 2, 1, 3),
                    full_k.unsqueeze(0).permute(0, 2, 1, 3),
                    full_v.unsqueeze(0).permute(0, 2, 1, 3),
                    attn_mask=attn_bias.unsqueeze(0).unsqueeze(0),
                    dropout_p=0.0,
                    is_causal=False,
                    scale=1.0 / (self.head_dim**0.5),
                    enable_gqa=(self.num_heads != self.num_kv_heads),
                )
                attn_out = attn_out_4d.permute(0, 2, 1, 3)
            else:
                attn = self.attn_causal if is_causal else self.attn_noncausal
                attn_out = attn(q.unsqueeze(0), full_k.unsqueeze(0), full_v.unsqueeze(0))

            attn_out = attn_out.squeeze(0).reshape(-1, self.q_size)
            attn_out, _ = self.o_proj(attn_out)
            if update_past_key_values and past_key_values is not None:
                past_key_values.key_cache[self.layer_idx] = full_k
                past_key_values.value_cache[self.layer_idx] = full_v
            return attn_out, past_key_values

        def forward(
            self,
            packed_query_sequence: torch.Tensor,
            query_lens=None,
            packed_query_position_embeddings=None,
            past_key_values=None,
            update_past_key_values: bool = True,
            is_causal: bool = True,
            mode: str = "und",
            **kwargs,
        ):
            return self._forward_und(
                packed_query_sequence,
                packed_query_position_embeddings,
                past_key_values,
                is_causal=is_causal,
                update_past_key_values=update_past_key_values,
            )

    class _MiniMLP(nn.Module):
        def __init__(self, hidden_size: int = 64, inter_size: int = 128):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, inter_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, inter_size, bias=False)
            self.down_proj = nn.Linear(inter_size, hidden_size, bias=False)
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class _MiniDecoderLayer(nn.Module):
        def __init__(self, layer_idx: int, hidden_size: int = 64):
            super().__init__()
            self.layer_idx = layer_idx
            self.input_layernorm = _MiniRMSNorm(hidden_size)
            self.self_attn = _MiniPackedAttentionMoT(layer_idx, hidden_size=hidden_size)
            self.post_attention_layernorm = _MiniRMSNorm(hidden_size)
            self.mlp = _MiniMLP(hidden_size=hidden_size)

        def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states=None,
            query_lens=None,
            packed_query_position_embeddings=None,
            past_key_values=None,
            update_past_key_values: bool = True,
            is_causal: bool = True,
            mode: str = "und",
            **kwargs,
        ):
            res = hidden_states
            h = self.input_layernorm(hidden_states)
            h, past_key_values = self.self_attn(
                packed_query_sequence=h,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                past_key_values=past_key_values,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                mode=mode,
            )
            h = res + h
            res = h
            h = self.post_attention_layernorm(h)
            h = res + self.mlp(h)
            return h, past_key_values

    class _MiniQwen2MoTModel(nn.Module):
        def __init__(self, num_layers: int = 2, vocab_size: int = 96, hidden_size: int = 64):
            super().__init__()
            self.use_moe = True
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([_MiniDecoderLayer(i, hidden_size) for i in range(num_layers)])
            self.norm = _MiniRMSNorm(hidden_size)
            self.rotary_emb = _MiniRotaryEmb(head_dim=hidden_size // 4)

        def forward(
            self,
            packed_query_sequence=None,
            query_lens=None,
            packed_query_position_ids=None,
            past_key_values=None,
            update_past_key_values: bool = True,
            is_causal: bool = True,
            mode: str = "und",
            packed_text_ids=None,
            return_embeddings_only: bool = False,
            **kwargs,
        ):
            from morphserve.worker.bagel_interceptor import _make_navit_output

            if packed_query_sequence is None:
                packed_query_sequence = self.embed_tokens(packed_text_ids)
            if return_embeddings_only:
                return _make_navit_output(packed_query_sequence, past_key_values)
            cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
            pos_emb = (cos.squeeze(0), sin.squeeze(0))
            for lyr in self.layers:
                packed_query_sequence, past_key_values = lyr(
                    hidden_states=packed_query_sequence,
                    query_lens=query_lens,
                    packed_query_position_embeddings=pos_emb,
                    past_key_values=past_key_values,
                    update_past_key_values=update_past_key_values,
                    is_causal=is_causal,
                    mode=mode,
                )
            packed_query_sequence = self.norm(packed_query_sequence)
            return _make_navit_output(packed_query_sequence, past_key_values)

    class _MiniCausalLM(nn.Module):
        def __init__(self, num_layers: int = 2, vocab_size: int = 96, hidden_size: int = 64):
            super().__init__()
            self.model = _MiniQwen2MoTModel(num_layers=num_layers, vocab_size=vocab_size, hidden_size=hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        def forward(self, **kwargs):
            return self.model(**kwargs)

    class _MiniBagelWrapper:
        def __init__(self):
            torch.manual_seed(2027)
            self.language_model = _MiniCausalLM()

        def forward(self, *args, **kwargs):
            return None

    class _MiniPipelineWrapper:
        def __init__(self):
            self.device = "cpu"
            self.bagel = _MiniBagelWrapper()


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

    def test_idle_session_kv_watermark_swap_out_and_available_kv_bytes(self) -> None:
        """Decision 1/6: Verify live `available_kv_bytes()` sync + automatic Host DRAM KV swap-out/swap-in."""
        base_pol = EnginePolicyConfig.from_preset("morphserve", num_gpus=1)
        # Tight 12,544 MiB super pool (exact 28 x 448 MiB experts) + 256 MiB fine pool = 12,800 MiB total, no overflow
        pol = replace(
            base_pol,
            super_pool_mib=12_544,
            fine_kv_pool_mib=256,
            allow_vmm_overflow=False,
        )
        adapter = SyntheticCPTGAdapter(rank=0, policy=pol, max_kv_region_mib=128)
        adapter.attach_expert_mirror(None, populate=True)
        init_info = adapter.bind_runtime_resources()
        try:
            total_pool_b = (12_544 + 256) * MIB
            self.assertEqual(init_info["kv_free_bytes"], total_pool_b)
            self.assertEqual(adapter.available_kv_bytes(), total_pool_b)

            # Prepare 3 multi-image sessions (128 MiB region each = 384 MiB KV):
            # 384 MiB KV + 12,544 MiB pinned DiT experts = 12,928 MiB > 12,800 MiB physical pool!
            adapter.prepare_turn("sess_0", {"prompt": "turn0 img", "image_paths": ["a.png"]})
            free_after_1 = adapter.available_kv_bytes()
            self.assertLess(free_after_1, total_pool_b)

            adapter.prepare_turn("sess_1", {"prompt": "turn1 img", "image_paths": ["b.png"]})
            adapter.prepare_turn("sess_2", {"prompt": "turn2 img", "image_paths": ["c.png"]})

            # Active DiT on `sess_2` pins all 28 expert layers (12,544 MiB) + runs 1,600 MiB scoped VAE:
            # adapter MUST swap out idle session KV (`sess_0`) to Host DRAM so DiT completes with 0 overflow!
            dit_res = adapter.generate_dit("sess_2", {"prompt": "gen", "height": 512, "dit_steps": 2})
            self.assertEqual(dit_res["status"], "ok")
            self.assertIn("kv_free_bytes", dit_res)

            st_mid = adapter.memory_and_vmm_stats()
            # Exactly ONE idle session (`sess_0`, 128 MiB) must be evicted to fit 12,544 MiB experts + 256 MiB KV;
            # `sess_1` MUST stay resident until the explicit high-watermark eviction below!
            self.assertEqual(st_mid["kv_host_swap_outs"], 1)
            self.assertEqual(st_mid["swapped_kv_contexts"], 1)
            self.assertEqual(st_mid["vmm_stats"]["overflow_creates"], 0)
            self.assertEqual(st_mid["vmm_physical_pool_mib"], 12_800.0)
            self.assertGreaterEqual(st_mid["combined_physical_mib"], st_mid["vmm_physical_pool_mib"])

            # Explicit high-watermark eviction swaps out remaining unpinned idle session (`sess_1`) while preserving excluded `sess_2`
            freed_extra = adapter.evict_idle_kv_to_host(total_pool_b - 128 * MIB, exclude_ctx_ids=("sess_2",))
            self.assertEqual(freed_extra, 128 * MIB)
            self.assertGreaterEqual(adapter.available_kv_bytes(), total_pool_b - 128 * MIB)

            # Resuming swapped-out `sess_0` after think-time transparently swaps its KV back in!
            ar_res = adapter.decode_ar("sess_0", max_tokens=4)
            self.assertEqual(ar_res["tokens"], 4)
            self.assertGreaterEqual(adapter.memory_and_vmm_stats()["kv_host_swap_ins"], 1)

            # Releasing all contexts restores full 12,800 MiB KV availability
            for cid in ("sess_0", "sess_1", "sess_2"):
                adapter.release_context(cid)
            self.assertEqual(adapter.available_kv_bytes(), total_pool_b)
            self.assertEqual(adapter.memory_and_vmm_stats()["kv_host_pinned_bytes"], 0)
        finally:
            adapter.close()

    def test_synthetic_multiturn_pack_unpack_no_dup_inflation(self) -> None:
        """Decision 2/6: Verify `include_cfg_dup=True` (`twostage_shm`) wire 2x vs persistent `base_logical_bytes`."""
        pol = EnginePolicyConfig.from_preset("twostage_shm", num_gpus=2)
        ad = SyntheticCPTGAdapter(rank=0, policy=pol)
        ad.bind_runtime_resources()
        try:
            ad.prepare_turn("c0", {"prompt": "t0"})
            b_t1 = ad.prepare_turn("c0", {"prompt": "t1", "image_paths": ["i.png"], "prepare_for_dit": True})
            packed_dup = ad.pack_state("c0", include_cfg_dup=True)
            packed_fair = ad.pack_state("c0", include_cfg_dup=False)
            self.assertEqual(packed_dup.logical_bytes, 2 * b_t1)
            self.assertEqual(packed_fair.logical_bytes, b_t1)

            restored_b = ad.unpack_state("c0_dst", packed_dup.meta, packed_dup.payload)
            self.assertEqual(restored_b, b_t1)
        finally:
            ad.close()

    def test_internvlu_dual_regime_memory_plan_conservation(self) -> None:
        """Decision 5/6: Verify 40/80GB ample `super_pool` (zero premature eviction) & strict `<= 12 GiB` quota."""
        mirror_5gib = 5_120 * MIB  # Measured 20 x 256 MiB InternVL-U generation_decoder mirror

        # 1. Regime A (40 GB A100 unconstrained): super_pool >= mirror + 5632 MiB -> zero transient eviction
        plan_40g = compute_internvlu_memory_plan(
            total_dev_mib=40_338.0,
            free_dev_mib=35_400.0,
            base_pytorch_alloc_mib=4_200.0,
            mirror_bytes=mirror_5gib,
            policy_super_pool_mib=4_864,
            policy_fine_kv_pool_mib=512,
            super_handle_mib=64,
            allow_vmm_overflow=True,
            gpu_memory_fraction=None,
        )
        self.assertFalse(plan_40g["quota_constrained"])
        self.assertGreaterEqual(plan_40g["super_pool_mib"], 5_120 + 5_632)

        # 2. Regime A (80 GB H100 unconstrained)
        plan_80g = compute_internvlu_memory_plan(
            total_dev_mib=81_559.0,
            free_dev_mib=76_500.0,
            base_pytorch_alloc_mib=4_200.0,
            mirror_bytes=mirror_5gib,
            policy_super_pool_mib=4_864,
            policy_fine_kv_pool_mib=512,
        )
        self.assertFalse(plan_80g["quota_constrained"])
        self.assertGreaterEqual(plan_80g["super_pool_mib"], 10_752)

        # 3. Regime B (12 GiB quota on A100-40GB, nominal 40GiB, and H100-80GB):
        #    Total GPU physical budget (PyTorch cap + VMM super + VMM fine) MUST be strictly <= 12 GiB (12,288 MiB)!
        for tot_mib, frac in ((40_338.0, 0.30), (40_960.0, 0.30), (81_559.0, 0.15)):
            plan_12g = compute_internvlu_memory_plan(
                total_dev_mib=tot_mib,
                free_dev_mib=tot_mib - 4_300.0,
                base_pytorch_alloc_mib=4_200.0,
                mirror_bytes=mirror_5gib,
                policy_super_pool_mib=4_864,
                policy_fine_kv_pool_mib=512,
                super_handle_mib=64,
                allow_vmm_overflow=False,
                gpu_memory_fraction=frac,
            )
            self.assertTrue(plan_12g["quota_constrained"])
            self.assertFalse(plan_12g["allow_overflow"])
            self.assertGreaterEqual(plan_12g["super_pool_mib"], 5_120)
            total_phys_mib = plan_12g["pytorch_cap_mib"] + plan_12g["vmm_physical_pool_mib"]
            self.assertLessEqual(total_phys_mib, 12_288.0)
            self.assertLessEqual(
                tot_mib * plan_12g["pytorch_memory_fraction"] + plan_12g["vmm_physical_pool_mib"],
                12_288.0,
            )

        # 4. Regime B pre-discounted startup fraction (`compute_internvlu_memory_quota_overrides`: 16 MiB handle, 4800 MiB super)
        plan_12g_override = compute_internvlu_memory_plan(
            total_dev_mib=40_960.0,
            free_dev_mib=36_600.0,
            base_pytorch_alloc_mib=3_950.0,
            mirror_bytes=4_800 * MIB,
            policy_super_pool_mib=4_800,
            policy_fine_kv_pool_mib=32,
            super_handle_mib=16,
            allow_vmm_overflow=False,
            gpu_memory_fraction=round(8_640.0 / 40_960.0, 6),
        )
        self.assertTrue(plan_12g_override["quota_constrained"])
        self.assertEqual(plan_12g_override["super_pool_mib"], 4_800)
        self.assertEqual(plan_12g_override["vmm_physical_pool_mib"], 4_832)
        self.assertLessEqual(
            plan_12g_override["pytorch_cap_mib"] + plan_12g_override["vmm_physical_pool_mib"],
            12_288.0,
        )

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for exact tensor equivalence verification")
    def test_packed_bd_batched_ar_exact_equivalence_and_vmm_inplace_append(self) -> None:
        """Decision 3/6: Prove `[B, D]` packed GEMM + `VmmNaiveCache` in-place + `Q=1` mask-free exact parity."""
        pipe_stock = _MiniPipelineWrapper()
        pipe_fast = _MiniPipelineWrapper()
        pipe_fast.bagel.language_model.load_state_dict(pipe_stock.bagel.language_model.state_dict())

        interceptor = BagelExecutionInterceptor.for_pipeline(pipe_fast)
        vmm = SimulatedTriPoolBackend(
            device=0, super_pool_mib=64, fine_kv_pool_mib=16, super_handle_mib=64, materialize_va_bytes=True
        )
        try:
            prompts = [
                torch.tensor([3, 11, 25], dtype=torch.long),
                torch.tensor([7, 19, 31, 42, 55, 61], dtype=torch.long),
                torch.tensor([2], dtype=torch.long),
                torch.tensor([5, 8, 13, 21, 34, 50, 63, 77, 81], dtype=torch.long),
            ]
            num_layers = 2
            stock_caches = [_SimpleNaiveCache(num_layers) for _ in prompts]
            vmm_caches = [VmmNaiveCache(num_layers, vmm=vmm, max_seq_len=128) for _ in prompts]

            # 1. Prefill each heterogeneous prompt independently
            with torch.no_grad():
                for b, p_ids in enumerate(prompts):
                    pos = torch.arange(len(p_ids), dtype=torch.long)
                    ql = torch.tensor([len(p_ids)], dtype=torch.int32)
                    pipe_stock.bagel.language_model(
                        packed_text_ids=p_ids,
                        query_lens=ql,
                        packed_query_position_ids=pos,
                        past_key_values=stock_caches[b],
                        update_past_key_values=True,
                        is_causal=True,
                        mode="und",
                    )
                    pipe_fast.bagel.language_model(
                        packed_text_ids=p_ids,
                        query_lens=ql,
                        packed_query_position_ids=pos,
                        past_key_values=vmm_caches[b],
                        update_past_key_values=True,
                        is_causal=True,
                        mode="und",
                    )

            base_ptrs = [int(vmm_caches[b].key_cache[0].data_ptr()) for b in range(4)]

            # Reset projection call counters before multi-step decode
            for lyr in pipe_fast.bagel.language_model.model.layers:
                lyr.self_attn.qkv_proj.calls = 0
                lyr.self_attn.qkv_proj.batch_widths.clear()
                lyr.self_attn.o_proj.calls = 0
                lyr.mlp.calls = 0
                lyr.self_attn.stock_mask_allocs = 0
            interceptor.reset_ar_telemetry()

            # 2. Run 6 AR decode steps: compare 4 serial stock steps vs ONE `[B=4, D]` packed GEMM step per round!
            stock_curr = [torch.tensor([10 + b], dtype=torch.long) for b in range(4)]
            stock_pos = [torch.tensor([len(prompts[b])], dtype=torch.long) for b in range(4)]

            sched = UnifiedPreemptiveScheduler(pipe_fast, mode="layer_chunk", tpot_slo_ms=80.0)
            tasks = [
                ArDecodeTask(
                    req_id=f"req_{b}",
                    kv_cache=vmm_caches[b],
                    curr_tokens=torch.tensor([10 + b], dtype=torch.long),
                    pos_ids=torch.tensor([len(prompts[b])], dtype=torch.long),
                    max_tokens=6,
                    end_token_id=None,
                    tpot_slo_ms=80.0,
                )
                for b in range(4)
            ]
            for tk in tasks:
                sched.schedule_ar_arrival(0.0, tk)

            with torch.no_grad():
                sched.begin_session_clock()
                sched.drain_remaining_ar_tasks()

                stock_hist: list[list[int]] = [[] for _ in range(4)]
                for _step in range(6):
                    for b in range(4):
                        stock_hist[b].append(int(stock_curr[b][0].item()))
                        out_s = pipe_stock.bagel.language_model(
                            packed_text_ids=stock_curr[b],
                            query_lens=torch.ones_like(stock_curr[b]),
                            packed_query_position_ids=stock_pos[b],
                            past_key_values=stock_caches[b],
                            update_past_key_values=True,
                            is_causal=True,
                            mode="und",
                        )
                        logits_s = pipe_stock.bagel.language_model.lm_head(out_s.packed_query_sequence)
                        stock_curr[b] = torch.argmax(logits_s, dim=-1)
                        stock_pos[b] = stock_pos[b] + 1

            # Exact greedy token sequence match across all 4 heterogeneous requests!
            for b in range(4):
                self.assertEqual(tasks[b].generated_token_ids, stock_hist[b])
                # VmmNaiveCache grew strictly in-place at the exact same base VA pointer (zero torch.cat realloc!)
                self.assertEqual(int(vmm_caches[b].key_cache[0].data_ptr()), base_ptrs[b])
                max_kv_diff = float((vmm_caches[b].key_cache[0] - stock_caches[b].key_cache[0]).abs().max().item())
                self.assertLess(max_kv_diff, 1e-5)

            # Weight projections ran ONCE per step (`6` steps total, batch width `4`), NOT `4 * 6 = 24` times!
            lyr0 = pipe_fast.bagel.language_model.model.layers[0]
            self.assertEqual(lyr0.self_attn.qkv_proj.calls, 6)
            self.assertEqual(lyr0.self_attn.qkv_proj.batch_widths, [4] * 6)
            self.assertEqual(lyr0.self_attn.o_proj.calls, 6)
            self.assertEqual(lyr0.mlp.calls, 6)
            self.assertEqual(lyr0.self_attn.stock_mask_allocs, 0)
            self.assertEqual(sched.batched_ar_yields, 6)
            self.assertEqual(sched.max_microbatch_size, 4)
            self.assertEqual(interceptor.ar_vmm_inplace_appends, 6 * 4 * num_layers)
            self.assertEqual(interceptor.ar_q1_fast_attn_calls, 6 * 4 * num_layers)

            # Also verify CoW read-only fork safety under `append_or_update_kv_cache`
            parent_snap = vmm_caches[0].key_cache[0].clone()
            fork_ctx = cow_fork_context({"kv_lens": [9], "ropes": [9], "past_key_values": vmm_caches[0]})
            child_cache = VmmNaiveCache(num_layers, vmm=vmm, max_seq_len=128)
            for lidx in range(num_layers):
                dict.__setitem__(child_cache.key_cache, lidx, fork_ctx["past_key_values"].key_cache[lidx])
                dict.__setitem__(child_cache.value_cache, lidx, fork_ctx["past_key_values"].value_cache[lidx])
            new_k = torch.randn(1, 2, 16)
            new_v = torch.randn(1, 2, 16)
            fk, _, prev_l, used_vmm = append_or_update_kv_cache(child_cache, 0, new_k, new_v)
            self.assertTrue(used_vmm)
            self.assertEqual(prev_l, parent_snap.shape[0])
            self.assertEqual(fk.shape[0], parent_snap.shape[0] + 1)
            self.assertTrue(torch.equal(vmm_caches[0].key_cache[0], parent_snap))

            child_cache.close()
            for c in vmm_caches:
                c.close()
        finally:
            interceptor.uninstall()
            vmm.shutdown()

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for BAGEL multi-turn pack/unpack & host swap tests")
    def test_bagel_multiturn_pack_unpack_and_pinned_host_swap_roundtrip(self) -> None:
        """Decisions 1/6 & 2/6: Verify exact `cfg_text` RoPE/CoW restoration, loopback u8 pack, & Host DRAM swap."""
        vmm = SimulatedTriPoolBackend(
            device=0, super_pool_mib=64, fine_kv_pool_mib=16, super_handle_mib=64, materialize_va_bytes=True
        )
        try:
            main_pkv = VmmNaiveCache(2, vmm=vmm, max_seq_len=128)
            cfg_img_pkv = _SimpleNaiveCache(2)
            for lidx in range(2):
                main_pkv.key_cache[lidx] = torch.randn(48, 2, 16)
                main_pkv.value_cache[lidx] = torch.randn(48, 2, 16)
                cfg_img_pkv.key_cache[lidx] = torch.randn(12, 2, 16)
                cfg_img_pkv.value_cache[lidx] = torch.randn(12, 2, 16)

            # Prefix before Turn-1 prompt was 36 tokens (ropes=[36]), while main_ctx after prompt is 48 (ropes=[48])!
            cfg_text_pkv = _SimpleNaiveCache(2)
            for lidx in range(2):
                cfg_text_pkv.key_cache[lidx] = main_pkv.key_cache[lidx][:36]
                cfg_text_pkv.value_cache[lidx] = main_pkv.value_cache[lidx][:36]
            cfg_text_pkv._cow_shared_prefix_len = 36

            img_u8 = torch.randint(0, 255, (3, 16, 16), dtype=torch.uint8)
            entry = {
                "main_ctx": {"kv_lens": [48], "ropes": [48], "past_key_values": main_pkv},
                "cfg_text_ctx": {"kv_lens": [36], "ropes": [36], "past_key_values": cfg_text_pkv},
                "cfg_img_ctx": {"kv_lens": [12], "ropes": [12], "past_key_values": cfg_img_pkv},
                "cfg_text_prefix_len": 36,
                "pending_vae_paths": ["turn0.png"],
                "pending_vae_u8": True,
                "pending_vae_pil": False,
                "last_img_u8": img_u8,
                "last_img_pil": None,
            }

            # 1. Test `pack_bagel_multi_context(include_cfg_text_dup=False)` -> `unpack_bagel_multi_context`:
            packed = pack_bagel_multi_context(entry, include_cfg_text_dup=False, device="cpu")
            unpacked_cow = unpack_bagel_multi_context(
                packed.meta,
                packed.payload,
                main_cache_factory=lambda n: VmmNaiveCache(n, vmm=vmm, max_seq_len=128),
                aux_cache_factory=_SimpleNaiveCache,
                use_cow_rebind=True,
            )
            self.assertEqual(unpacked_cow["cfg_text_ctx"]["ropes"], [36])
            self.assertEqual(unpacked_cow["cfg_text_ctx"]["kv_lens"], [36])
            self.assertEqual(unpacked_cow["main_ctx"]["ropes"], [48])
            self.assertTrue(unpacked_cow["pending_vae_u8"])
            self.assertEqual(unpacked_cow["pending_vae_paths"], ["turn0.png"])
            self.assertTrue(torch.equal(unpacked_cow["last_img_u8"], img_u8))
            # Zero-copy CoW storage sharing preserved on `morphserve` (`use_cow_rebind=True`)
            self.assertEqual(
                unpacked_cow["cfg_text_ctx"]["past_key_values"].key_cache[0].data_ptr(),
                unpacked_cow["main_ctx"]["past_key_values"].key_cache[0].data_ptr(),
            )

            # Fair `static_nccl_fission` (`use_cow_rebind=False`): single-copy wire payload, independent HBM storage
            unpacked_fission = unpack_bagel_multi_context(
                packed.meta,
                packed.payload,
                main_cache_factory=_SimpleNaiveCache,
                aux_cache_factory=_SimpleNaiveCache,
                use_cow_rebind=False,
            )
            self.assertEqual(unpacked_fission["cfg_text_ctx"]["ropes"], [36])
            self.assertNotEqual(
                unpacked_fission["cfg_text_ctx"]["past_key_values"].key_cache[0].data_ptr(),
                unpacked_fission["main_ctx"]["past_key_values"].key_cache[0].data_ptr(),
            )

            # 2. Test Host DRAM Swap-Out & Swap-In (`pack_bagel_entry_to_pinned_host` / `unpack_bagel_entry_from_pinned_host`)
            ref_main_k0 = main_pkv.key_cache[0].clone()
            ref_img_v1 = cfg_img_pkv.value_cache[1].clone()
            bundle = pack_bagel_entry_to_pinned_host(entry, pin_memory=False)
            # CoW prefix is NOT duplicated in Host DRAM!
            self.assertTrue(bundle["cfg_text_is_prefix_only"])
            self.assertNotIn("cfg_text_ctx", bundle["branches"])

            main_pkv.close()
            self.assertEqual(unpacked_cow["main_ctx"]["past_key_values"].allocated_regions, 4)
            unpacked_cow["main_ctx"]["past_key_values"].close()
            self.assertEqual(vmm.stats_dict()["kv_mapped_pages_current"], 0)

            restored_entry: dict = {}
            unpack_bagel_entry_from_pinned_host(
                bundle,
                restored_entry,
                device="cpu",
                main_cache_factory=lambda n: VmmNaiveCache(n, vmm=vmm, max_seq_len=128),
                aux_cache_factory=_SimpleNaiveCache,
                use_cow_rebind=True,
            )
            self.assertTrue(torch.equal(restored_entry["main_ctx"]["past_key_values"].key_cache[0], ref_main_k0))
            self.assertTrue(torch.equal(restored_entry["cfg_img_ctx"]["past_key_values"].value_cache[1], ref_img_v1))
            self.assertEqual(restored_entry["cfg_text_ctx"]["ropes"], [36])
            self.assertTrue(
                torch.equal(restored_entry["cfg_text_ctx"]["past_key_values"].key_cache[0], ref_main_k0[:36])
            )
            restored_entry["main_ctx"]["past_key_values"].close()
        finally:
            vmm.shutdown()

    def test_scheduler_microbatch_cap_step0_h2d_no_sync_and_calibration_persist(self) -> None:
        """Verify `max_microbatch_cap` bounds burst microbatches and Step-0 H2D streaming skips per-layer calibration."""
        pipe = _MockPipeline()
        sched = UnifiedPreemptiveScheduler(
            pipe,
            mode="layer_chunk",
            tpot_slo_ms=80.0,
            max_microbatch_cap=3,
        )
        chunk_widths: list[int] = []

        def _mock_batch_step(curr_list, _pos_list, kv_list):
            chunk_widths.append(len(curr_list))
            return [99 for _ in curr_list], kv_list

        try:
            sched.interceptor.forward_ar_step_batch = _mock_batch_step
            sched.begin_session_clock()
            for idx in range(7):
                sched.schedule_ar_arrival(
                    0.0,
                    ArDecodeTask(
                        req_id=f"burst_{idx}",
                        kv_cache=object(),
                        curr_tokens=idx,
                        pos_ids=0,
                        max_tokens=1,
                        end_token_id=None,
                        tpot_slo_ms=80.0,
                    ),
                )
            served = sched.service_due_ar_tasks(force_all_urgent=True)
            self.assertEqual(served, 7)
            self.assertEqual(chunk_widths, [3, 3, 1])
            self.assertEqual(sched.max_microbatch_size, 3)

            # Verify Step-0 H2D streaming does NOT record per-layer sync samples on Step 0, then calibrates on Step 1
            class _MockVmm:
                def expert_resident_count(self, _eid: int) -> int:
                    return 14

            class _MockPager:
                def __init__(self):
                    self.vmm = _MockVmm()
                    self.expert_id = 0
                    self.num_layers = 28

            sched.pager = _MockPager()
            sched.reset_telemetry()
            sched.begin_session_clock()
            self.assertTrue(sched._step0_h2d_streaming)
            self.assertFalse(sched._layer_calibrated)

            # Step 0: must NOT calibrate `layer_times_ms` while H2D is streaming (`_step0_h2d_streaming=True`)
            t_s0 = time.perf_counter()
            for lidx in range(28):
                sched.on_gen_layer_complete(lidx, 28, time.perf_counter() - 0.001)
            sched.on_dit_step_complete(t_s0, 0.0)
            self.assertEqual(len(sched.layer_times_ms), 0)
            self.assertFalse(sched._layer_calibrated)
            self.assertFalse(sched._step0_h2d_streaming)

            # Step 1: calibrates `layer_times_ms` cleanly once all layers are resident
            t_s1 = time.perf_counter()
            for lidx in range(28):
                sched.on_gen_layer_complete(lidx, 28, time.perf_counter() - 0.001)
            sched.on_dit_step_complete(t_s1, 0.0)
            self.assertEqual(len(sched.layer_times_ms), 28)
            self.assertTrue(sched._layer_calibrated)

            # `reset_telemetry(preserve_layer_calibration=True)` retains `_layer_calibrated` across DiT turns
            sched.reset_telemetry(preserve_layer_calibration=True)
            self.assertTrue(sched._layer_calibrated)
        finally:
            sched.interceptor.uninstall()

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for swapped VmmNaiveCache & pending_dit_turn tests")
    def test_swapped_vmm_naive_cache_nbytes_no_swapin_and_pending_dit_turn_roundtrip(self) -> None:
        """Verify `bagel_ctx_nbytes`/`bagel_entry_nbytes` never swap in parked sessions & `pending_dit_turn` round-trips."""
        vmm = SimulatedTriPoolBackend(
            device=0, super_pool_mib=64, fine_kv_pool_mib=16, super_handle_mib=64, materialize_va_bytes=True
        )
        try:
            pkv = VmmNaiveCache(2, vmm=vmm, max_seq_len=128)
            for lidx in range(2):
                pkv.key_cache[lidx] = torch.randn(32, 2, 16)
                pkv.value_cache[lidx] = torch.randn(32, 2, 16)
            expected_ctx_b = bagel_ctx_nbytes({"kv_lens": [32], "ropes": [32], "past_key_values": pkv})
            self.assertGreater(expected_ctx_b, 0)

            reclaimed = pkv.swap_out_async()
            self.assertGreater(reclaimed, 0)
            self.assertTrue(pkv.is_swapped_out)
            self.assertEqual(vmm.stats_dict()["kv_mapped_pages_current"], 0)
            self.assertEqual(vmm.stats_dict()["kv_swap_ins_total"], 0)

            entry = {
                "main_ctx": {"kv_lens": [32], "ropes": [32], "past_key_values": pkv},
                "cfg_text_ctx": None,
                "cfg_img_ctx": None,
                "cfg_text_prefix_len": 0,
                "pending_dit_turn": {
                    "use_prev_image": True,
                    "prev_image_path": None,
                    "image_paths": ["ref.png"],
                    "prompt": "Make it snowy.",
                },
                "last_img_u8": torch.zeros((3, 8, 8), dtype=torch.uint8),
                "last_img_pil": None,
            }
            # Querying `bagel_ctx_nbytes` and `bagel_entry_nbytes` MUST NOT trigger `ensure_resident()`!
            swapped_ctx_b = bagel_ctx_nbytes(entry["main_ctx"])
            swapped_ent_b = bagel_entry_nbytes(entry)
            self.assertEqual(swapped_ctx_b, expected_ctx_b)
            self.assertEqual(swapped_ent_b, expected_ctx_b + 3 * 8 * 8)
            self.assertTrue(pkv.is_swapped_out)
            self.assertEqual(vmm.stats_dict()["kv_swap_ins_total"], 0)

            # Swap back in, then verify `pending_dit_turn` round-trips through wire pack/unpack & host swap
            pkv.ensure_resident()
            packed = pack_bagel_multi_context(entry, include_cfg_text_dup=False, device="cpu")
            unpacked = unpack_bagel_multi_context(
                packed.meta,
                packed.payload,
                main_cache_factory=_SimpleNaiveCache,
                aux_cache_factory=_SimpleNaiveCache,
                use_cow_rebind=True,
            )
            self.assertEqual(unpacked["pending_dit_turn"], entry["pending_dit_turn"])

            h_bundle = pack_bagel_entry_to_pinned_host(entry, pin_memory=False)
            restored = unpack_bagel_entry_from_pinned_host(
                h_bundle,
                {},
                device="cpu",
                main_cache_factory=_SimpleNaiveCache,
                aux_cache_factory=_SimpleNaiveCache,
                use_cow_rebind=True,
            )
            self.assertEqual(restored["pending_dit_turn"], entry["pending_dit_turn"])
            pkv.close()
        finally:
            vmm.shutdown()

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for forward_ar_step_batch kwarg & alias tests")
    def test_forward_ar_step_batch_readonly_kv_return_logits_and_telemetry_aliases(self) -> None:
        """Verify `update_past_key_values=False`, `return_logits=True`, length guard, and telemetry aliases."""
        pipe = _MiniPipelineWrapper()
        interceptor = BagelExecutionInterceptor.for_pipeline(pipe)
        try:
            caches = [_SimpleNaiveCache(2), _SimpleNaiveCache(2)]
            with torch.no_grad():
                for b, seq_len in enumerate((4, 7)):
                    toks = torch.arange(1, seq_len + 1, dtype=torch.long)
                    pipe.bagel.language_model(
                        packed_text_ids=toks,
                        query_lens=torch.tensor([seq_len], dtype=torch.int32),
                        packed_query_position_ids=torch.arange(seq_len, dtype=torch.long),
                        past_key_values=caches[b],
                        update_past_key_values=True,
                        is_causal=True,
                        mode="und",
                    )
                interceptor.reset_ar_telemetry()
                logits, out_caches = interceptor.forward_ar_step_batch(
                    [torch.tensor([9], dtype=torch.long), torch.tensor([12], dtype=torch.long)],
                    [torch.tensor([4], dtype=torch.long), torch.tensor([7], dtype=torch.long)],
                    caches,
                    update_past_key_values=False,
                    return_logits=True,
                )
            self.assertEqual(tuple(logits.shape), (2, pipe.bagel.language_model.lm_head.out_features))
            self.assertEqual(logits.dtype, torch.float32)
            # `update_past_key_values=False` MUST leave prior KV lengths untouched (4 and 7)
            self.assertEqual(int(out_caches[0].key_cache[0].shape[0]), 4)
            self.assertEqual(int(out_caches[1].key_cache[0].shape[0]), 7)
            self.assertEqual(interceptor.batched_ar_steps, 1)
            self.assertEqual(interceptor.batched_ar_tokens, 2)
            self.assertEqual(interceptor.mask_free_sdpa_calls, 4)
            self.assertEqual(interceptor.vmm_inplace_kv_appends, 0)

            with self.assertRaises(ValueError):
                interceptor.forward_ar_step_batch(
                    [torch.tensor([1], dtype=torch.long)],
                    [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
                    [caches[0]],
                )
        finally:
            interceptor.uninstall()

    def test_internvlu_trim_generated_tokens_and_high_base_alloc_12gib_conservation(self) -> None:
        """Verify right-padded EOS/PAD trimming in `InternVL-U` batch decode & unconditional `<= 12 GiB` cap."""

        class _MockTok:
            pad_token_id = 0
            eos_token_id = 2
            unk_token_id = 3

            def convert_tokens_to_ids(self, s: str) -> int:
                return {"<|im_end|>": 151645, "<img>": 151665}.get(s, 3)

        tok = _MockTok()
        self.assertEqual(_trim_internvlu_generated_tokens([11, 22, 33, 151645, 0, 0, 0], tok), [11, 22, 33, 151645])
        self.assertEqual(_trim_internvlu_generated_tokens([11, 22, 0, 0], tok), [11, 22])
        self.assertEqual(_trim_internvlu_generated_tokens([0, 0, 0], tok), [0])

        # High base PyTorch alloc (`7,350 MiB`) under `4,832 MiB` VMM override MUST still stay `<= 12,288 MiB`!
        plan_tight = compute_internvlu_memory_plan(
            total_dev_mib=40_338.0,
            free_dev_mib=32_900.0,
            base_pytorch_alloc_mib=7_350.0,
            mirror_bytes=4_800 * MIB,
            policy_super_pool_mib=4_800,
            policy_fine_kv_pool_mib=32,
            super_handle_mib=16,
            allow_vmm_overflow=False,
            gpu_memory_fraction=round(8_640.0 / 40_960.0, 6),
        )
        self.assertTrue(plan_tight["quota_constrained"])
        self.assertLessEqual(plan_tight["pytorch_cap_mib"] + plan_tight["vmm_physical_pool_mib"], 12_288.0)

    def test_module4_script_syntax_and_imports(self) -> None:
        script = REPO_ROOT / "deploy/kairos/scripts/06_phase1_module4_validate.sh"
        proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        for rel in (
            "python/morphserve/worker/__init__.py",
            "python/morphserve/worker/bagel_interceptor.py",
            "python/morphserve/worker/preemptive_scheduler.py",
            "python/morphserve/models/__init__.py",
            "python/morphserve/models/base.py",
            "python/morphserve/models/bagel_adapter.py",
            "python/morphserve/models/internvl_u_adapter.py",
            "python/morphserve/bench/validate_module4_preemption.py",
        ):
            p = REPO_ROOT / rel
            res = subprocess.run([sys.executable, "-m", "py_compile", str(p)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, msg=f"py_compile failed on {rel}: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
