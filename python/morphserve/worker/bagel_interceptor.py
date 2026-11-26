"""Single permanent, mode-aware execution coordinator for `BagelPipeline` (`gen` + packed `[B, D]` `und` AR)."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Sequence

from morphserve.kernels import SharedPrefixCache, shared_prefix_cfg_attention
from morphserve.vmm import VmmNaiveCache


@dataclass
class _FallbackNavitOutput:
    packed_query_sequence: Any
    past_key_values: Any


def _make_navit_output(seq: Any, pkv: Any) -> Any:
    try:
        from vllm_omni.diffusion.models.bagel.bagel_transformer import BaseNavitOutputWithPast

        return BaseNavitOutputWithPast(packed_query_sequence=seq, past_key_values=pkv)
    except ImportError:
        return _FallbackNavitOutput(packed_query_sequence=seq, past_key_values=pkv)


def _sync_cuda_if_present(device: Any = None) -> None:
    """Synchronize the current compute CUDA stream without stalling background VMM H2D/swap streams."""
    try:
        import torch

        if torch.cuda.is_available():
            if device is not None:
                if str(device).startswith("cuda") or getattr(device, "type", None) == "cuda":
                    torch.cuda.current_stream(device).synchronize()
            else:
                torch.cuda.current_stream().synchronize()
    except ImportError:
        pass


_ACTIVE_CFG_INTERCEPTOR: ContextVar["BagelExecutionInterceptor | None"] = ContextVar(
    "_ACTIVE_CFG_INTERCEPTOR", default=None
)
_ORIG_NAIVE_MERGE: Any = None


def _ensure_contextvar_naive_merge_hook() -> None:
    """Install one idempotent ContextVar-dispatched merge hook on `NaiveCache` (no cross-instance clobber)."""
    global _ORIG_NAIVE_MERGE
    if _ORIG_NAIVE_MERGE is not None:
        return
    try:
        from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache
    except ImportError:
        return

    _ORIG_NAIVE_MERGE = NaiveCache.merge

    def _dispatched_merge(caches: Sequence[Any]) -> Any:
        interceptor = _ACTIVE_CFG_INTERCEPTOR.get()
        if interceptor is None or not interceptor._cfg_active or len(caches) < 2:
            return _ORIG_NAIVE_MERGE(caches)
        sp = SharedPrefixCache.from_branch_caches(
            caches,
            min_shared_tokens=interceptor.cfg_min_shared_tokens,
            verify_all_layers=True,
        )
        interceptor.merge_calls += 1
        interceptor.last_shared_prefix_len = sp.shared_prefix_len
        interceptor.max_shared_prefix_len = max(interceptor.max_shared_prefix_len, sp.shared_prefix_len)
        interceptor.last_unique_kv_tokens = sp.unique_kv_tokens
        interceptor.last_naive_kv_tokens = sp.naive_kv_tokens
        interceptor.last_padded_naive_kv_tokens = sp.padded_naive_kv_tokens
        return sp

    NaiveCache.merge = staticmethod(_dispatched_merge)


class BagelExecutionInterceptor:
    """Permanent per-pipeline execution coordinator owning `gen` & packed `[B, D]` `und` fast-paths.

    Consolidates:
      1. `LayerExpertPager` block-level H2D wait gating via `pager.set_wait_gate(...)` (zero private pokes).
      2. Step-0 `HalfNodeH2DPacer` slot release delegated cleanly to `LayerExpertPager.generation_mode(...)`.
      3. `UnifiedPreemptiveScheduler` EDF block-chunk & step-boundary AR yields.
      4. `ContextVar`-scoped `SharedPrefixCache` (`LECP CoW`) + `FlashPrefix-CFG` attention with
         instance-scoped workspace caches (zero module-global tensor caches).
      5. Fair `mode == 'und'` `Q=1` mask-free SDPA + `VmmNaiveCache.append_kv` + `[B, D]` packed GEMM AR batching.
    """

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.bagel = pipeline.bagel
        self.lm_model = pipeline.bagel.language_model.model
        self.device = getattr(pipeline, "device", None)

        self.pager: Any = None
        self._active_scheduler: Any = None
        self._step0_release_cb: Callable[[], None] | None = None
        self._step0_done: bool = False
        self._gen_depth: int = 0
        self._installed: bool = False
        self._orig_lm_forward: Any = None
        self._orig_bagel_forward: Any = None
        self._orig_forward_unds: list[tuple[Any, Any]] = []

        # Shared-prefix CFG hooks & strictly instance-scoped workspace caches
        self._cfg_enabled: bool = False
        self._cfg_active: bool = False
        self._cfg_hooks_installed: bool = False
        self.cfg_min_shared_tokens: int = 32
        self.cfg_allow_cascade_split: bool = True
        self._orig_forward_gens: list[tuple[Any, Any]] = []
        self._direct_ws_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._cu_seqlens_cache: dict[tuple[str, tuple[int, ...]], Any] = {}
        self._attn_meta_cache: dict[tuple[str, tuple[int, ...]], Any] = {}
        self.reset_cfg_telemetry()

        # AR fast-path & batching telemetry
        self.ar_q1_fast_attn_calls: int = 0
        self.ar_vmm_inplace_appends: int = 0
        self.ar_batched_step_calls: int = 0
        self.ar_batched_tokens_total: int = 0
        self.max_ar_batch_width_seen: int = 0

    def reset_ar_telemetry(self) -> None:
        self.ar_q1_fast_attn_calls = 0
        self.ar_vmm_inplace_appends = 0
        self.ar_batched_step_calls = 0
        self.ar_batched_tokens_total = 0
        self.max_ar_batch_width_seen = 0

    def reset_cfg_telemetry(self) -> None:
        self.merge_calls: int = 0
        self.shared_attn_calls: int = 0
        self.cascade_split_calls: int = 0
        self.unpadded_single_branch_calls: int = 0
        self.last_shared_prefix_len: int = 0
        self.max_shared_prefix_len: int = 0
        self.last_unique_kv_tokens: int = 0
        self.last_naive_kv_tokens: int = 0
        self.last_padded_naive_kv_tokens: int = 0

    def clear_workspace_caches(self) -> None:
        """Release instance-scoped attention workspaces and clear `SharedPrefixCache` LECP memo."""
        self._direct_ws_cache.clear()
        self._cu_seqlens_cache.clear()
        self._attn_meta_cache.clear()
        SharedPrefixCache.clear_memo()

    @classmethod
    def for_pipeline(cls, pipeline: Any) -> "BagelExecutionInterceptor":
        existing = getattr(pipeline, "_morphserve_interceptor", None)
        if isinstance(existing, cls):
            existing.install()
            return existing
        inst = cls(pipeline)
        inst.install()
        setattr(pipeline, "_morphserve_interceptor", inst)
        return inst

    @property
    def is_gen_compute_active(self) -> bool:
        sched = self._active_scheduler
        in_ar_yield = bool(sched is not None and getattr(sched, "in_ar_yield", False))
        return (self._gen_depth > 0) and (not in_ar_yield)

    @property
    def cfg_enabled(self) -> bool:
        return self._cfg_enabled

    def bind_pager(self, pager: Any | None) -> None:
        """Bind `LayerExpertPager` so its layer waits are gated automatically on `mode == 'gen'`."""
        self.pager = pager
        if pager is not None and hasattr(pager, "set_wait_gate"):
            pager.set_wait_gate(lambda: self.is_gen_compute_active)

    def _install_cfg_gen_hooks(self) -> None:
        if self._cfg_hooks_installed:
            return
        _ensure_contextvar_naive_merge_hook()
        interceptor = self
        layers = getattr(self.lm_model, "layers", None) or []
        for layer in layers:
            attn_mod = getattr(layer, "self_attn", None)
            orig_fwd_gen = getattr(attn_mod, "_forward_gen", None) if attn_mod is not None else None
            if not callable(orig_fwd_gen):
                continue
            self._orig_forward_gens.append((attn_mod, orig_fwd_gen))

            def _make_fwd_gen(mod: Any, orig_fn: Any):
                def _wrapped_forward_gen(
                    packed_query_sequence: Any,
                    query_lens: Any,
                    packed_query_position_embeddings: Any,
                    past_key_values: Any,
                    packed_vae_token_indexes: Any,
                    packed_text_indexes: Any,
                    update_past_key_values: bool = False,
                ) -> tuple[Any, Any]:
                    active_ic = _ACTIVE_CFG_INTERCEPTOR.get()
                    use_cfg = (
                        (active_ic is interceptor or (active_ic is None and interceptor._cfg_active))
                        and interceptor._cfg_active
                    )
                    if (
                        not use_cfg
                        or not isinstance(past_key_values, SharedPrefixCache)
                        or update_past_key_values
                        or (callable(getattr(mod, "_is_sp_active", None)) and mod._is_sp_active())
                    ):
                        return orig_fn(
                            packed_query_sequence=packed_query_sequence,
                            query_lens=query_lens,
                            packed_query_position_embeddings=packed_query_position_embeddings,
                            past_key_values=past_key_values,
                            packed_vae_token_indexes=packed_vae_token_indexes,
                            packed_text_indexes=packed_text_indexes,
                            update_past_key_values=update_past_key_values,
                        )

                    import torch

                    text_indices = packed_text_indexes
                    vae_indices = packed_vae_token_indexes
                    packed_query_sequence = packed_query_sequence.to(torch.bfloat16)

                    qkv, _ = mod.qkv_proj(packed_query_sequence, text_indices, vae_indices)
                    q, k, v = qkv.split([mod.q_size, mod.kv_size, mod.kv_size], dim=-1)
                    q = q.view(-1, mod.num_heads, mod.head_dim)
                    k = k.view(-1, mod.num_kv_heads, mod.head_dim)
                    v = v.view(-1, mod.num_kv_heads, mod.head_dim)

                    q = mod.q_norm(q.to(torch.float32), text_indices, vae_indices)
                    k = mod.k_norm(k.to(torch.float32), text_indices, vae_indices)

                    cos, sin = [x[..., : mod.head_dim // 2] for x in packed_query_position_embeddings]
                    q = mod.rotary_op(q.to(cos.dtype).unsqueeze(0), cos, sin).squeeze(0)
                    k = mod.rotary_op(k.to(cos.dtype).unsqueeze(0), cos, sin).squeeze(0)

                    q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
                    text_q, text_k, text_v = q[text_indices], k[text_indices], v[text_indices]
                    vae_q, vae_k, vae_v = q[vae_indices], k[vae_indices], v[vae_indices]

                    num_branches = len(query_lens)
                    text_per_branch = text_q.shape[0] // num_branches
                    vae_per_branch = vae_q.shape[0] // num_branches

                    q_list = [
                        torch.cat([t, vx], dim=0)
                        for t, vx in zip(
                            text_q.split([text_per_branch] * num_branches),
                            vae_q.split([vae_per_branch] * num_branches),
                        )
                    ]
                    k_loc_list = [
                        torch.cat([t, vx], dim=0)
                        for t, vx in zip(
                            text_k.split([text_per_branch] * num_branches),
                            vae_k.split([vae_per_branch] * num_branches),
                        )
                    ]
                    v_loc_list = [
                        torch.cat([t, vx], dim=0)
                        for t, vx in zip(
                            text_v.split([text_per_branch] * num_branches),
                            vae_v.split([vae_per_branch] * num_branches),
                        )
                    ]

                    lidx = mod.layer_idx
                    mode_backend = "cascade_cfg" if interceptor.cfg_allow_cascade_split else "lossless_varlen"
                    call_stats: dict[str, int] = {}
                    branch_outs = shared_prefix_cfg_attention(
                        q_list=q_list,
                        k_loc_list=k_loc_list,
                        v_loc_list=v_loc_list,
                        k_prefix=past_key_values.shared_prefix_k[lidx],
                        v_prefix=past_key_values.shared_prefix_v[lidx],
                        shared_branch_ids=past_key_values.shared_branch_ids,
                        k_suffix_list=[past_key_values.branch_suffix_k[b][lidx] for b in range(num_branches)],
                        v_suffix_list=[past_key_values.branch_suffix_v[b][lidx] for b in range(num_branches)],
                        sm_scale=1.0 / math.sqrt(mod.head_dim),
                        attn_noncausal=mod.attn_noncausal,
                        backend=mode_backend,
                        stats_out=call_stats,
                        direct_ws_cache=interceptor._direct_ws_cache,
                        cu_seqlens_cache=interceptor._cu_seqlens_cache,
                        attn_meta_cache=interceptor._attn_meta_cache,
                    )
                    if past_key_values.shared_prefix_len > 0:
                        interceptor.shared_attn_calls += 1
                        interceptor.cascade_split_calls += int(call_stats.get("cascade_split_executed", 0))
                    interceptor.unpadded_single_branch_calls += len(past_key_values.unshared_branch_ids)

                    attn_out = torch.stack(branch_outs, dim=0).reshape(num_branches, -1, mod.q_size)
                    text_attn = attn_out[:, :text_per_branch].reshape(-1, mod.q_size)
                    vae_attn = attn_out[:, text_per_branch:].reshape(-1, mod.q_size)
                    text_len = text_attn.shape[0]

                    local_packed = torch.cat([text_attn, vae_attn], dim=0)
                    local_text_idx = torch.arange(text_len, device=local_packed.device)
                    local_vae_idx = torch.arange(text_len, text_len + vae_attn.shape[0], device=local_packed.device)
                    local_out, _ = mod.o_proj(local_packed, local_text_idx, local_vae_idx)

                    full_output = local_out.new_zeros((packed_query_sequence.shape[0], mod.hidden_size))
                    full_output[packed_text_indexes] = local_out[:text_len]
                    full_output[packed_vae_token_indexes] = local_out[text_len:]
                    return full_output, past_key_values

                return _wrapped_forward_gen

            attn_mod._forward_gen = _make_fwd_gen(attn_mod, orig_fwd_gen)

        self._cfg_hooks_installed = True

    def enable_shared_prefix_cfg(
        self,
        min_shared_tokens: int = 32,
        allow_cascade_split: bool = True,
    ) -> "BagelExecutionInterceptor":
        """Install and configure `SharedPrefixCache` + `FlashPrefix-CFG` hooks directly on this interceptor."""
        self.cfg_min_shared_tokens = int(min_shared_tokens)
        self.cfg_allow_cascade_split = bool(allow_cascade_split)
        self._cfg_enabled = True
        self._install_cfg_gen_hooks()
        return self

    @contextmanager
    def shared_prefix_cfg_scope(
        self,
        *,
        reset_stats: bool = True,
        allow_cascade_split: bool | None = None,
    ):
        """Activate `ContextVar`-scoped `SharedPrefixCache` + `FlashPrefix-CFG` for one generation pass."""
        if not self._cfg_hooks_installed:
            self._install_cfg_gen_hooks()
        if reset_stats:
            self.reset_cfg_telemetry()
        prev_active = self._cfg_active
        prev_split = self.cfg_allow_cascade_split
        if allow_cascade_split is not None:
            self.cfg_allow_cascade_split = bool(allow_cascade_split)
        tok = _ACTIVE_CFG_INTERCEPTOR.set(self)
        self._cfg_active = True
        try:
            yield self
        finally:
            self._cfg_active = prev_active
            self.cfg_allow_cascade_split = prev_split
            _ACTIVE_CFG_INTERCEPTOR.reset(tok)
            SharedPrefixCache.clear_memo()

    @staticmethod
    def _project_und_qkv_rope(
        attn_mod: Any,
        hidden: Any,
        cos_half: Any,
        sin_half: Any,
        *,
        num_rows: int = -1,
    ) -> tuple[Any, Any, Any]:
        """Shared QKV projection, FP32 QK-norm, RoPE, and dtype alignment for `mode == 'und'`."""
        import torch

        qkv, _ = attn_mod.qkv_proj(hidden)
        q, k, v = qkv.split([attn_mod.q_size, attn_mod.kv_size, attn_mod.kv_size], dim=-1)
        q = q.view(num_rows, attn_mod.num_heads, attn_mod.head_dim)
        k = k.view(num_rows, attn_mod.num_kv_heads, attn_mod.head_dim)
        v = v.view(num_rows, attn_mod.num_kv_heads, attn_mod.head_dim)

        q = attn_mod.q_norm(q.to(torch.float32))
        k = attn_mod.k_norm(k.to(torch.float32))

        q = attn_mod.rotary_op(q.to(cos_half.dtype).unsqueeze(0), cos_half, sin_half).squeeze(0)
        k = attn_mod.rotary_op(k.to(cos_half.dtype).unsqueeze(0), cos_half, sin_half).squeeze(0)

        out_dt = v.dtype
        return q.to(out_dt), k.to(out_dt), v

    def _append_cache_kv(
        self,
        cache: Any | None,
        layer_idx: int,
        k_new: Any,
        v_new: Any,
        *,
        update_past_key_values: bool = True,
    ) -> tuple[Any, Any, int]:
        """Delegate KV append to `VmmNaiveCache.append_kv` or stock `NaiveCache` concat cleanly."""
        if cache is None:
            return k_new, v_new, 0

        if isinstance(cache, VmmNaiveCache):
            fk, fv = cache.append_kv(
                layer_idx,
                k_new,
                v_new,
                update_past_key_values=update_past_key_values,
            )
            if update_past_key_values and cache.uses_vmm_backing:
                self.ar_vmm_inplace_appends += 1
            delta = 1 if getattr(k_new, "ndim", 3) == 2 else int(k_new.shape[0])
            prev_len = max(0, int(fk.shape[0]) - delta)
            return fk, fv, prev_len

        import torch

        lidx = int(layer_idx)
        get_k = cache.key_cache.get if isinstance(cache.key_cache, dict) else (lambda i: cache.key_cache[i])
        get_v = cache.value_cache.get if isinstance(cache.value_cache, dict) else (lambda i: cache.value_cache[i])
        ck = get_k(lidx)
        cv = get_v(lidx)
        prev_len = int(ck.shape[0]) if ck is not None else 0
        if ck is not None and prev_len > 0:
            fk = torch.cat([ck, k_new], dim=0)
            fv = torch.cat([cv, v_new], dim=0)
        else:
            fk = k_new
            fv = v_new
        if update_past_key_values:
            cache.key_cache[lidx] = fk
            cache.value_cache[lidx] = fv
        return fk, fv, prev_len

    def _supports_packed_ar_gemm(self) -> bool:
        lm = self.lm_model
        lm_outer = getattr(self.bagel, "language_model", None)
        if not (hasattr(lm, "embed_tokens") and hasattr(lm, "rotary_emb") and hasattr(lm, "norm")):
            return False
        if lm_outer is None or not callable(getattr(lm_outer, "lm_head", None)):
            return False
        layers = getattr(lm, "layers", None)
        if not layers:
            return False
        first = layers[0]
        return (
            hasattr(first, "input_layernorm")
            and hasattr(first, "post_attention_layernorm")
            and hasattr(first, "mlp")
            and hasattr(first, "self_attn")
            and hasattr(first.self_attn, "qkv_proj")
            and hasattr(first.self_attn, "o_proj")
        )

    def forward_ar_step_batch(
        self,
        curr_tokens_list: Sequence[Any],
        pos_ids_list: Sequence[Any],
        kv_caches: Sequence[Any],
        *,
        update_past_key_values: bool = True,
        return_logits: bool = False,
    ) -> tuple[Any, list[Any]]:
        """Execute ONE `[B, D]` packed GEMM causal AR step across `B >= 1` concurrent requests."""
        import torch
        import torch.nn.functional as F

        batch_size = len(kv_caches)
        if batch_size == 0:
            raise ValueError("forward_ar_step_batch requires at least 1 active request")
        if len(curr_tokens_list) != batch_size or len(pos_ids_list) != batch_size:
            raise ValueError("curr_tokens_list, pos_ids_list, and kv_caches must have identical batch length")

        self.install()
        self.ar_batched_step_calls += 1
        self.ar_batched_tokens_total += batch_size
        self.max_ar_batch_width_seen = max(self.max_ar_batch_width_seen, batch_size)
        do_update = bool(update_past_key_values)

        if not self._supports_packed_ar_gemm():
            next_list: list[Any] = []
            logits_list: list[Any] = []
            out_caches: list[Any] = []
            for tok_t, pos_t, cache in zip(curr_tokens_list, pos_ids_list, kv_caches):
                ql = torch.ones_like(tok_t)
                out = self.bagel.language_model(
                    packed_text_ids=tok_t,
                    query_lens=ql,
                    packed_query_position_ids=pos_t,
                    past_key_values=cache,
                    update_past_key_values=do_update,
                    is_causal=True,
                    mode="und",
                )
                logits_b = self.bagel.language_model.lm_head(out.packed_query_sequence)
                logits_list.append(logits_b.reshape(1, -1))
                next_list.append(torch.argmax(logits_b, dim=-1).reshape(1))
                out_caches.append(out.past_key_values)
            if return_logits:
                return torch.cat(logits_list, dim=0), out_caches
            return torch.cat(next_list, dim=0), out_caches

        packed_ids = (
            curr_tokens_list[0].reshape(1)
            if batch_size == 1
            else torch.cat([t.reshape(1) for t in curr_tokens_list], dim=0)
        )
        packed_pos = (
            pos_ids_list[0].reshape(1)
            if batch_size == 1
            else torch.cat([p.reshape(1) for p in pos_ids_list], dim=0)
        )

        lm = self.lm_model
        lm_head = self.bagel.language_model.lm_head
        hidden_states = lm.embed_tokens(packed_ids)  # [B, D]
        cos, sin = lm.rotary_emb(hidden_states, packed_pos.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)

        updated_caches = list(kv_caches)
        if do_update:
            for cache in updated_caches:
                if isinstance(cache, VmmNaiveCache) and not cache.fast_ar_active:
                    cache.set_fast_ar_mode(True, reserve_tokens=128)

        first_half_d = lm.layers[0].self_attn.head_dim // 2
        cos_h0 = cos[..., :first_half_d]
        sin_h0 = sin[..., :first_half_d]

        for layer_idx, decoder_layer in enumerate(lm.layers):
            residual = hidden_states
            normed = decoder_layer.input_layernorm(hidden_states)
            attn_mod = decoder_layer.self_attn

            half_d = attn_mod.head_dim // 2
            cos_h = cos_h0 if half_d == first_half_d else cos[..., :half_d]
            sin_h = sin_h0 if half_d == first_half_d else sin[..., :half_d]
            q, k, v = self._project_und_qkv_rope(attn_mod, normed, cos_h, sin_h, num_rows=batch_size)

            sm_scale = 1.0 / (attn_mod.head_dim**0.5)
            use_gqa = attn_mod.num_heads != attn_mod.num_kv_heads

            if batch_size == 1:
                full_k, full_v, _ = self._append_cache_kv(
                    updated_caches[0], layer_idx, k, v, update_past_key_values=do_update
                )
                self.ar_q1_fast_attn_calls += 1
                attn_4d = F.scaled_dot_product_attention(
                    q.unsqueeze(0).permute(0, 2, 1, 3),
                    full_k.unsqueeze(0).permute(0, 2, 1, 3),
                    full_v.unsqueeze(0).permute(0, 2, 1, 3),
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=sm_scale,
                    enable_gqa=use_gqa,
                )
                attn_packed = attn_4d.permute(0, 2, 1, 3).reshape(1, attn_mod.q_size)
            else:
                attn_packed = torch.empty((batch_size, attn_mod.q_size), dtype=v.dtype, device=q.device)
                for b in range(batch_size):
                    full_kb, full_vb, _ = self._append_cache_kv(
                        updated_caches[b],
                        layer_idx,
                        k[b : b + 1],
                        v[b : b + 1],
                        update_past_key_values=do_update,
                    )
                    ob_4d = F.scaled_dot_product_attention(
                        q[b : b + 1].unsqueeze(0).permute(0, 2, 1, 3),
                        full_kb.unsqueeze(0).permute(0, 2, 1, 3),
                        full_vb.unsqueeze(0).permute(0, 2, 1, 3),
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False,
                        scale=sm_scale,
                        enable_gqa=use_gqa,
                    )
                    attn_packed[b : b + 1] = ob_4d.permute(0, 2, 1, 3).reshape(1, attn_mod.q_size)
                self.ar_q1_fast_attn_calls += batch_size

            attn_proj, _ = attn_mod.o_proj(attn_packed)
            hidden_states = residual + attn_proj

            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = lm.norm(hidden_states)
        logits = lm_head(hidden_states)
        if return_logits:
            return logits, updated_caches
        next_tokens = torch.argmax(logits, dim=-1)
        return next_tokens, updated_caches

    def _install_und_attn_hooks(self) -> None:
        self._orig_forward_unds.clear()
        interceptor = self
        layers = getattr(self.lm_model, "layers", None) or []
        for layer in layers:
            attn_mod = getattr(layer, "self_attn", None)
            orig_und = getattr(attn_mod, "_forward_und", None) if attn_mod is not None else None
            if not callable(orig_und):
                continue
            self._orig_forward_unds.append((attn_mod, orig_und))

            def _make_fast_forward_und(mod: Any):
                def _fast_forward_und(
                    packed_query_sequence: Any,
                    packed_query_position_embeddings: Any,
                    past_key_values: Any,
                    is_causal: bool,
                    update_past_key_values: bool = True,
                ) -> tuple[Any, Any]:
                    import torch
                    import torch.nn.functional as F

                    cos_h, sin_h = [x[..., : mod.head_dim // 2] for x in packed_query_position_embeddings]
                    q, k, v = interceptor._project_und_qkv_rope(
                        mod, packed_query_sequence, cos_h, sin_h, num_rows=-1
                    )
                    full_k, full_v, cache_len = interceptor._append_cache_kv(
                        past_key_values,
                        mod.layer_idx,
                        k,
                        v,
                        update_past_key_values=bool(update_past_key_values),
                    )

                    q_len = int(q.shape[0])
                    if q_len == 1:
                        interceptor.ar_q1_fast_attn_calls += 1
                        attn_out_4d = F.scaled_dot_product_attention(
                            q.unsqueeze(0).permute(0, 2, 1, 3),
                            full_k.unsqueeze(0).permute(0, 2, 1, 3),
                            full_v.unsqueeze(0).permute(0, 2, 1, 3),
                            attn_mask=None,
                            dropout_p=0.0,
                            is_causal=False,
                            scale=1.0 / (mod.head_dim**0.5),
                            enable_gqa=(mod.num_heads != mod.num_kv_heads),
                        )
                        attn_out = attn_out_4d.permute(0, 2, 1, 3)
                    elif is_causal and cache_len > 0:
                        k_len = int(full_k.shape[0])
                        arange_q = torch.arange(q_len, device=q.device).unsqueeze(1)
                        arange_k = torch.arange(k_len, device=q.device).unsqueeze(0)
                        mask = arange_k > (cache_len + arange_q)
                        attn_bias = torch.zeros(q_len, k_len, dtype=q.dtype, device=q.device)
                        attn_bias.masked_fill_(mask, float("-inf"))
                        attn_out_4d = F.scaled_dot_product_attention(
                            q.unsqueeze(0).permute(0, 2, 1, 3),
                            full_k.unsqueeze(0).permute(0, 2, 1, 3),
                            full_v.unsqueeze(0).permute(0, 2, 1, 3),
                            attn_mask=attn_bias.unsqueeze(0).unsqueeze(0),
                            dropout_p=0.0,
                            is_causal=False,
                            scale=1.0 / (mod.head_dim**0.5),
                            enable_gqa=(mod.num_heads != mod.num_kv_heads),
                        )
                        attn_out = attn_out_4d.permute(0, 2, 1, 3)
                    else:
                        attn = mod.attn_causal if is_causal else mod.attn_noncausal
                        attn_out = attn(
                            q.unsqueeze(0),
                            full_k.unsqueeze(0),
                            full_v.unsqueeze(0),
                        )

                    attn_out = attn_out.squeeze(0).reshape(-1, mod.q_size)
                    attn_out, _ = mod.o_proj(attn_out)
                    return attn_out, past_key_values

                return _fast_forward_und

            attn_mod._forward_und = _make_fast_forward_und(attn_mod)

    def install(self) -> None:
        if self._installed:
            return
        interceptor = self
        self._orig_lm_forward = self.lm_model.forward
        self._orig_bagel_forward = self.bagel.forward
        self._install_und_attn_hooks()

        def _intercepted_lm_forward(
            packed_query_sequence=None,
            query_lens=None,
            packed_query_position_ids=None,
            past_key_values=None,
            update_past_key_values=True,
            is_causal=True,
            mode="und",
            packed_vae_token_indexes=None,
            packed_text_indexes=None,
            packed_text_ids=None,
            return_embeddings_only=False,
        ):
            sched = interceptor._active_scheduler
            in_yield = bool(sched is not None and getattr(sched, "in_ar_yield", False))
            if mode != "gen" or return_embeddings_only or in_yield:
                return interceptor._orig_lm_forward(
                    packed_query_sequence=packed_query_sequence,
                    query_lens=query_lens,
                    packed_query_position_ids=packed_query_position_ids,
                    past_key_values=past_key_values,
                    update_past_key_values=update_past_key_values,
                    is_causal=is_causal,
                    mode=mode,
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_text_indexes=packed_text_indexes,
                    packed_text_ids=packed_text_ids,
                    return_embeddings_only=return_embeddings_only,
                )

            interceptor._gen_depth += 1
            try:
                if sched is None or sched.mode != "layer_chunk":
                    return interceptor._orig_lm_forward(
                        packed_query_sequence=packed_query_sequence,
                        query_lens=query_lens,
                        packed_query_position_ids=packed_query_position_ids,
                        past_key_values=past_key_values,
                        update_past_key_values=update_past_key_values,
                        is_causal=is_causal,
                        mode=mode,
                        packed_vae_token_indexes=packed_vae_token_indexes,
                        packed_text_indexes=packed_text_indexes,
                        packed_text_ids=packed_text_ids,
                        return_embeddings_only=return_embeddings_only,
                    )

                lm = interceptor.lm_model
                if packed_query_sequence is None:
                    packed_query_sequence = lm.embed_tokens(packed_text_ids)
                if hasattr(lm, "rotary_emb") and packed_query_position_ids is not None:
                    cos, sin = lm.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
                    packed_query_position_embeddings = (cos.squeeze(0), sin.squeeze(0))
                else:
                    packed_query_position_embeddings = None

                extra_inputs: dict[str, Any] = {}
                if getattr(lm, "use_moe", False):
                    extra_inputs = {
                        "mode": "gen",
                        "packed_vae_token_indexes": packed_vae_token_indexes,
                        "packed_text_indexes": packed_text_indexes,
                    }

                num_layers = len(lm.layers)
                for layer_idx, decoder_layer in enumerate(lm.layers):
                    if interceptor.pager is not None and not interceptor.pager.spec.install_module_hooks:
                        interceptor.pager.wait_layer_if_needed(layer_idx, "interceptor_gen")
                    t_l0 = time.perf_counter()
                    packed_query_sequence, past_key_values = decoder_layer(
                        hidden_states=packed_query_sequence,
                        encoder_hidden_states=None,
                        query_lens=query_lens,
                        packed_query_position_embeddings=packed_query_position_embeddings,
                        past_key_values=past_key_values,
                        update_past_key_values=update_past_key_values,
                        is_causal=is_causal,
                        **extra_inputs,
                    )
                    sched.on_gen_layer_complete(layer_idx=layer_idx, num_layers=num_layers, t_layer_start_s=t_l0)

                if hasattr(lm, "norm"):
                    packed_query_sequence = lm.norm(
                        packed_query_sequence, packed_text_indexes, packed_vae_token_indexes
                    )
                return _make_navit_output(packed_query_sequence, past_key_values)
            finally:
                interceptor._gen_depth -= 1

        def _intercepted_bagel_forward(*args: Any, **kwargs: Any) -> Any:
            sched = interceptor._active_scheduler
            t_s0 = time.perf_counter()
            ar_before_ms = float(sched.ar_compute_total_ms) if sched is not None else 0.0

            interceptor._gen_depth += 1
            try:
                out = interceptor._orig_bagel_forward(*args, **kwargs)
            finally:
                interceptor._gen_depth -= 1

            if interceptor._step0_release_cb is not None and not interceptor._step0_done:
                cb = interceptor._step0_release_cb
                interceptor._step0_done = True
                interceptor._step0_release_cb = None
                cb()

            if sched is not None:
                sched.on_dit_step_complete(step_start_s=t_s0, ar_compute_before_ms=ar_before_ms)
            return out

        self.lm_model.forward = _intercepted_lm_forward
        self.bagel.forward = _intercepted_bagel_forward
        self._installed = True

    def uninstall(self) -> None:
        self._cfg_active = False
        self._cfg_enabled = False
        for attn_mod, orig_gen in self._orig_forward_gens:
            attn_mod._forward_gen = orig_gen
        self._orig_forward_gens.clear()
        self._cfg_hooks_installed = False
        self.clear_workspace_caches()

        for attn_mod, orig_und in self._orig_forward_unds:
            attn_mod._forward_und = orig_und
        self._orig_forward_unds.clear()
        if not self._installed:
            return
        if self._orig_lm_forward is not None:
            self.lm_model.forward = self._orig_lm_forward
            self._orig_lm_forward = None
        if self._orig_bagel_forward is not None:
            self.bagel.forward = self._orig_bagel_forward
            self._orig_bagel_forward = None
        self._installed = False

    @contextmanager
    def generation_session(
        self,
        *,
        pager: Any | None = None,
        use_cfg_patch: bool = True,
        scheduler: Any | None = None,
        pacer: Any | None = None,
        rank: int = 0,
        prefetch_all: bool = True,
        reset_cfg_stats: bool = True,
        exclude_session_ids: Sequence[int] = (),
    ):
        """Coordinate Pager + Pacer + SharedPrefix-CFG + EDF Scheduler over DiT denoising steps."""
        self.install()
        eff_pager = pager if pager is not None else self.pager
        if eff_pager is not None:
            self.bind_pager(eff_pager)

        prev_sched = self._active_scheduler
        prev_cb = self._step0_release_cb
        prev_done = self._step0_done

        with ExitStack() as stack:
            if eff_pager is not None:
                stack.enter_context(
                    eff_pager.generation_mode(
                        prefetch_all=prefetch_all,
                        pacer=pacer,
                        rank=rank,
                        exclude_session_ids=exclude_session_ids,
                    )
                )
                if pacer is not None and callable(getattr(eff_pager, "release_active_pacer", None)):
                    self._step0_release_cb = lambda: eff_pager.release_active_pacer(sync_stream=True)
                    self._step0_done = False
            if use_cfg_patch and self._cfg_enabled:
                stack.enter_context(self.shared_prefix_cfg_scope(reset_stats=reset_cfg_stats))
            if scheduler is not None:
                scheduler.begin_session_clock()
                self._active_scheduler = scheduler
            try:
                yield self
            finally:
                self._active_scheduler = prev_sched
                self._step0_release_cb = prev_cb
                self._step0_done = prev_done
                SharedPrefixCache.clear_memo()
