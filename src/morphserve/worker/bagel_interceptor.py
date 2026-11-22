"""Single permanent, mode-aware execution coordinator for `BagelPipeline` (`gen` + packed `[B, D]` `und` AR)."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import time
from typing import Any, Callable, Sequence

from morphserve.kernels import BagelSharedPrefixPatch, install_shared_prefix_bagel_hooks
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


def _stream_ptr_for_tensor(tensor: Any) -> int:
    try:
        import torch

        if torch.cuda.is_available() and torch.is_tensor(tensor) and tensor.is_cuda:
            return int(torch.cuda.current_stream(tensor.device).cuda_stream)
    except Exception:
        pass
    return 0


def _vmm_append_single_slot(
    pkv: VmmNaiveCache,
    kind: str,
    slot_map: dict[int, Any],
    layer_idx: int,
    new_t: Any,
) -> Any:
    """Append `new_t` (`[add_len, H_kv, D_h]`) in-place into `VmmNaiveCache` Arena-1 VA without `torch.cat`."""
    import torch

    lidx = int(layer_idx)
    prev_t = dict.get(slot_map, lidx)
    if prev_t is None or int(prev_t.shape[0]) == 0:
        slot_map[lidx] = new_t
        return dict.get(slot_map, lidx)

    old_len = int(prev_t.shape[0])
    add_len = int(new_t.shape[0])
    if add_len == 0:
        return prev_t

    new_len = old_len + add_len
    elem_sz = int(new_t.element_size())
    row_bytes = max(1, (int(new_t.numel()) // add_len) * elem_sz)
    total_bytes = new_len * row_bytes
    out_shape = (new_len, *new_t.shape[1:])
    stream_ptr = _stream_ptr_for_tensor(new_t)

    rid = pkv._ensure_region(kind, lidx, total_bytes, row_bytes, stream_ptr)
    mapped_b = pkv.vmm.kv_region_mapped_bytes(rid)
    raw_u8 = pkv.vmm.kv_region_uint8_view(rid, mapped_b)
    key = (kind, lidx)

    if not bool(getattr(pkv.vmm, "is_simulated", False)):
        va_typed = raw_u8[:total_bytes].view(new_t.dtype).reshape(out_shape)
        # Materialize CoW read-only prefix into fresh private region if storage differs
        if int(prev_t.data_ptr()) != int(va_typed.data_ptr()):
            va_typed[:old_len].copy_(prev_t)
        va_typed[old_len:new_len].copy_(new_t)
        pkv._lengths[key] = new_len
        dict.__setitem__(slot_map, lidx, va_typed)
        return va_typed

    if len(raw_u8) >= total_bytes:
        va_typed = torch.frombuffer(raw_u8, dtype=torch.uint8, count=total_bytes).view(new_t.dtype).reshape(out_shape)
        if int(prev_t.data_ptr()) != int(va_typed.data_ptr()):
            va_typed[:old_len].copy_(prev_t.detach().cpu())
        va_typed[old_len:new_len].copy_(new_t.detach().cpu())
        pkv._lengths[key] = new_len
        dict.__setitem__(slot_map, lidx, va_typed)
        return va_typed.to(new_t.device) if new_t.device.type != "cpu" else va_typed

    cat_t = torch.cat([prev_t, new_t], dim=0)
    pkv._lengths[key] = new_len
    dict.__setitem__(slot_map, lidx, cat_t)
    return cat_t


def append_or_update_kv_cache(
    past_key_values: Any | None,
    layer_idx: int,
    k_new: Any,
    v_new: Any,
    *,
    update_past_key_values: bool = True,
) -> tuple[Any, Any, int, bool]:
    """Append `(k_new, v_new)` to `past_key_values` and return `(full_k, full_v, prev_cache_len, used_vmm_inplace)`."""
    import torch

    lidx = int(layer_idx)
    if past_key_values is None:
        return k_new, v_new, 0, False

    if getattr(past_key_values, "_is_swapped", False) and callable(getattr(past_key_values, "ensure_resident", None)):
        past_key_values.ensure_resident()

    get_k = (
        (lambda i: dict.get(past_key_values.key_cache, i))
        if isinstance(past_key_values.key_cache, dict)
        else (lambda i: past_key_values.key_cache[i])
    )
    get_v = (
        (lambda i: dict.get(past_key_values.value_cache, i))
        if isinstance(past_key_values.value_cache, dict)
        else (lambda i: past_key_values.value_cache[i])
    )
    cache_k = get_k(lidx)
    cache_v = get_v(lidx)
    prev_len = int(cache_k.shape[0]) if cache_k is not None else 0

    if not update_past_key_values:
        if cache_k is None:
            return k_new, v_new, 0, False
        return torch.cat([cache_k, k_new], dim=0), torch.cat([cache_v, v_new], dim=0), prev_len, False

    if (
        isinstance(past_key_values, VmmNaiveCache)
        and past_key_values.vmm is not None
        and not getattr(past_key_values, "_closed", False)
    ):
        append_fn = (
            getattr(past_key_values, "append_inplace", None)
            or getattr(past_key_values, "append_layer", None)
            or getattr(past_key_values, "append_kv", None)
        )
        if callable(append_fn):
            fk, fv = append_fn(lidx, k_new, v_new)
            return fk, fv, prev_len, True
        full_k = _vmm_append_single_slot(past_key_values, "k", past_key_values.key_cache, lidx, k_new)
        full_v = _vmm_append_single_slot(past_key_values, "v", past_key_values.value_cache, lidx, v_new)
        return full_k, full_v, prev_len, True

    if cache_k is not None and prev_len > 0:
        full_k = torch.cat([cache_k, k_new], dim=0)
        full_v = torch.cat([cache_v, v_new], dim=0)
    else:
        full_k = k_new
        full_v = v_new
    past_key_values.key_cache[lidx] = full_k
    past_key_values.value_cache[lidx] = full_v
    return full_k, full_v, prev_len, False


class BagelExecutionInterceptor:
    """Permanent per-pipeline execution coordinator owning `gen` & packed `[B, D]` `und` fast-paths.

    Consolidates:
      1. `LayerExpertPager` block-level H2D wait gating via `pager.set_wait_gate(...)` (zero private pokes).
      2. Step-0 `HalfNodeH2DPacer` slot release immediately after Step-0 completes.
      3. `UnifiedPreemptiveScheduler` EDF block-chunk & step-boundary AR yields.
      4. `BagelSharedPrefixPatch` `ContextVar`-scoped `SharedPrefixCache` + `FlashPrefix-CFG` attention.
      5. Fair `mode == 'und'` `Q=1` mask-free SDPA + `VmmNaiveCache` in-place append + `[B, D]` packed GEMM AR batching.
    """

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.bagel = pipeline.bagel
        self.lm_model = pipeline.bagel.language_model.model
        self.device = getattr(pipeline, "device", None)

        self.pager: Any = None
        self.cfg_patch: BagelSharedPrefixPatch | None = None
        self._active_scheduler: Any = None
        self._step0_release_cb: Callable[[], None] | None = None
        self._step0_done: bool = False
        self._gen_depth: int = 0
        self._installed: bool = False
        self._orig_lm_forward: Any = None
        self._orig_bagel_forward: Any = None
        self._orig_forward_unds: list[tuple[Any, Any]] = []

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

    @property
    def batched_ar_steps(self) -> int:
        return self.ar_batched_step_calls

    @property
    def batched_ar_tokens(self) -> int:
        return self.ar_batched_tokens_total

    @property
    def vmm_inplace_kv_appends(self) -> int:
        return self.ar_vmm_inplace_appends

    @property
    def mask_free_sdpa_calls(self) -> int:
        return self.ar_q1_fast_attn_calls

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

    def bind_pager(self, pager: Any | None) -> None:
        """Bind `LayerExpertPager` so its layer waits are gated automatically on `mode == 'gen'`."""
        self.pager = pager
        if pager is not None and hasattr(pager, "set_wait_gate"):
            pager.set_wait_gate(lambda: self.is_gen_compute_active)

    def enable_shared_prefix_cfg(self, min_shared_tokens: int = 32) -> BagelSharedPrefixPatch:
        """Install and bind `BagelSharedPrefixPatch` under this execution coordinator."""
        if self.cfg_patch is None:
            self.cfg_patch = install_shared_prefix_bagel_hooks(self.pipeline, min_shared_tokens=min_shared_tokens)
        return self.cfg_patch

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
        """Execute ONE `[B, D]` packed GEMM causal AR step across `B >= 1` concurrent requests.

        Reads every text projection (`embed_tokens`, `qkv_proj`, `o_proj`, `mlp.gate_up_proj`,
        `mlp.down_proj`, `lm_head`) ONCE for the whole `[B, D]` batch, appends each request's
        new `[1, H_kv, D_h]` K/V slice in-place on `VmmNaiveCache` (or `NaiveCache`), and runs
        mask-free `Q=1` SDPA per request with exact mathematical equivalence to `PackedAttentionMoT._forward_und`.
        """
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
        cos = cos.squeeze(0)  # [B, head_dim]
        sin = sin.squeeze(0)  # [B, head_dim]

        updated_caches = list(kv_caches)
        for layer_idx, decoder_layer in enumerate(lm.layers):
            residual = hidden_states
            normed = decoder_layer.input_layernorm(hidden_states)
            attn_mod = decoder_layer.self_attn

            qkv, _ = attn_mod.qkv_proj(normed)
            q, k, v = qkv.split([attn_mod.q_size, attn_mod.kv_size, attn_mod.kv_size], dim=-1)
            q = q.view(batch_size, attn_mod.num_heads, attn_mod.head_dim)
            k = k.view(batch_size, attn_mod.num_kv_heads, attn_mod.head_dim)
            v = v.view(batch_size, attn_mod.num_kv_heads, attn_mod.head_dim)

            q = attn_mod.q_norm(q.to(torch.float32))
            k = attn_mod.k_norm(k.to(torch.float32))

            half_d = attn_mod.head_dim // 2
            cos_h = cos[..., :half_d]
            sin_h = sin[..., :half_d]
            q = attn_mod.rotary_op(q.to(cos_h.dtype).unsqueeze(0), cos_h, sin_h).squeeze(0)
            k = attn_mod.rotary_op(k.to(cos_h.dtype).unsqueeze(0), cos_h, sin_h).squeeze(0)

            out_dt = v.dtype
            q = q.to(out_dt)
            k = k.to(out_dt)

            sm_scale = 1.0 / (attn_mod.head_dim**0.5)
            use_gqa = attn_mod.num_heads != attn_mod.num_kv_heads

            if batch_size == 1:
                full_k, full_v, _, used_vmm = append_or_update_kv_cache(
                    updated_caches[0], layer_idx, k, v, update_past_key_values=do_update
                )
                if used_vmm:
                    self.ar_vmm_inplace_appends += 1
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
                attn_packed = torch.empty((batch_size, attn_mod.q_size), dtype=out_dt, device=q.device)
                for b in range(batch_size):
                    kb = k[b : b + 1]
                    vb = v[b : b + 1]
                    full_kb, full_vb, _, used_vmm = append_or_update_kv_cache(
                        updated_caches[b], layer_idx, kb, vb, update_past_key_values=do_update
                    )
                    if used_vmm:
                        self.ar_vmm_inplace_appends += 1
                    qb_4d = q[b : b + 1].unsqueeze(0).permute(0, 2, 1, 3)
                    kb_4d = full_kb.unsqueeze(0).permute(0, 2, 1, 3)
                    vb_4d = full_vb.unsqueeze(0).permute(0, 2, 1, 3)
                    ob_4d = F.scaled_dot_product_attention(
                        qb_4d,
                        kb_4d,
                        vb_4d,
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

                    qkv, _ = mod.qkv_proj(packed_query_sequence)
                    q, k, v = qkv.split([mod.q_size, mod.kv_size, mod.kv_size], dim=-1)
                    q = q.view(-1, mod.num_heads, mod.head_dim)
                    k = k.view(-1, mod.num_kv_heads, mod.head_dim)
                    v = v.view(-1, mod.num_kv_heads, mod.head_dim)

                    q = mod.q_norm(q.to(torch.float32))
                    k = mod.k_norm(k.to(torch.float32))

                    cos, sin = [x[..., : mod.head_dim // 2] for x in packed_query_position_embeddings]
                    q = mod.rotary_op(q.to(cos.dtype).unsqueeze(0), cos, sin).squeeze(0)
                    k = mod.rotary_op(k.to(cos.dtype).unsqueeze(0), cos, sin).squeeze(0)

                    out_dt = v.dtype
                    q = q.to(out_dt)
                    k = k.to(out_dt)

                    full_k, full_v, cache_len, used_vmm = append_or_update_kv_cache(
                        past_key_values,
                        mod.layer_idx,
                        k,
                        v,
                        update_past_key_values=bool(update_past_key_values),
                    )
                    if used_vmm:
                        interceptor.ar_vmm_inplace_appends += 1

                    q_len = int(q.shape[0])
                    if q_len == 1:
                        # Single-token decode attends to all [0 .. cache_len] keys with zero mask allocation
                        interceptor.ar_q1_fast_attn_calls += 1
                        q_4d = q.unsqueeze(0).permute(0, 2, 1, 3)
                        k_4d = full_k.unsqueeze(0).permute(0, 2, 1, 3)
                        v_4d = full_v.unsqueeze(0).permute(0, 2, 1, 3)
                        attn_out_4d = F.scaled_dot_product_attention(
                            q_4d,
                            k_4d,
                            v_4d,
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
                        q_4d = q.unsqueeze(0).permute(0, 2, 1, 3)
                        k_4d = full_k.unsqueeze(0).permute(0, 2, 1, 3)
                        v_4d = full_v.unsqueeze(0).permute(0, 2, 1, 3)
                        attn_out_4d = F.scaled_dot_product_attention(
                            q_4d,
                            k_4d,
                            v_4d,
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
                _sync_cuda_if_present(interceptor.device)
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
        if self.cfg_patch is not None:
            try:
                self.cfg_patch.uninstall()
            except Exception:
                pass
            self.cfg_patch = None
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
        cfg_patch: Any | None = None,
        use_cfg_patch: bool = True,
        scheduler: Any | None = None,
        pacer_slot_cm: Any | None = None,
        prefetch_all: bool = True,
        reset_cfg_stats: bool = True,
    ):
        """Coordinate Pager + Pacer + CFG Patch + EDF Scheduler over DiT denoising steps."""
        self.install()
        eff_pager = pager if pager is not None else self.pager
        if eff_pager is not None:
            self.bind_pager(eff_pager)
        eff_cfg = cfg_patch if cfg_patch is not None else (self.cfg_patch if use_cfg_patch else None)

        prev_sched = self._active_scheduler
        prev_cb = self._step0_release_cb
        prev_done = self._step0_done
        pacer_active = {"held": False}

        def _release_pacer_once() -> None:
            if pacer_slot_cm is not None and pacer_active["held"]:
                pacer_active["held"] = False
                pacer_slot_cm.__exit__(None, None, None)

        with ExitStack() as stack:
            if pacer_slot_cm is not None:
                pacer_slot_cm.__enter__()
                pacer_active["held"] = True
                self._step0_release_cb = _release_pacer_once
                self._step0_done = False
            if eff_pager is not None:
                stack.enter_context(eff_pager.generation_mode(prefetch_all=prefetch_all))
            if eff_cfg is not None:
                stack.enter_context(eff_cfg.active(reset_stats=reset_cfg_stats))
            if scheduler is not None:
                scheduler.begin_session_clock()
                self._active_scheduler = scheduler
            try:
                yield self
            finally:
                self._active_scheduler = prev_sched
                self._step0_release_cb = prev_cb
                self._step0_done = prev_done
                _release_pacer_once()
