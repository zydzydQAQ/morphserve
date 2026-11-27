"""Pure stateless arithmetic-intensity-gated `FlashPrefix-CFG` attention & fused Triton LSE merge kernels.

Module 3 Cross-Paradigm State De-Redundancy kernels:
  - Arithmetic-intensity-gated two-pass `FlashPrefix-CFG` attention (`_run_flash_prefix_cfg` +
    fused Triton `_merge_two_lse_fwd`), automatically falling back losslessly to single-copy
    direct-workspace non-causal attention below threshold (`should_use_flash_prefix_split`).
  - Zero module-global tensor workspace caches: callers pass instance-scoped workspace dicts
    explicitly (e.g., owned by `BagelExecutionInterceptor`), or omit them for standalone calls.
"""

from __future__ import annotations

import math
from typing import Any

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _merge_two_lse_fwd(
        Oa,
        Lse_a,
        Ob,
        Lse_b,
        Out,
        soa_r,
        soa_d,
        sob_r,
        sob_d,
        sout_r,
        sout_d,
        sla_m,
        sla_h,
        slb_m,
        slb_h,
        scale_a,
        scale_b,
        N_ROWS,
        H_Q: tl.constexpr,
        BLOCK_R: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """Fused exact online-softmax LSE merge for FlashPrefix-CFG (zero transpose, zero autotune stall)."""
        pid_r = tl.program_id(0)
        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_d = tl.arange(0, HEAD_DIM)
        mask_r = offs_r < N_ROWS

        offs_m = offs_r // H_Q
        offs_h = offs_r - offs_m * H_Q
        la = tl.load(Lse_a + offs_m * sla_m + offs_h * sla_h, mask=mask_r, other=-1.0e20).to(tl.float32) * scale_a
        lb = tl.load(Lse_b + offs_m * slb_m + offs_h * slb_h, mask=mask_r, other=-1.0e20).to(tl.float32) * scale_b
        m = tl.maximum(tl.maximum(la, lb), -1.0e30)
        wa = tl.exp(la - m)
        wb = tl.exp(lb - m)
        inv = 1.0 / (wa + wb)
        ca = (wa * inv)[:, None]
        cb = (wb * inv)[:, None]

        oa = tl.load(Oa + offs_r[:, None] * soa_r + offs_d[None, :] * soa_d, mask=mask_r[:, None], other=0.0).to(tl.float32)
        ob = tl.load(Ob + offs_r[:, None] * sob_r + offs_d[None, :] * sob_d, mask=mask_r[:, None], other=0.0).to(tl.float32)
        merged = (oa * ca + ob * cb).to(Out.dtype.element_ty)
        tl.store(Out + offs_r[:, None] * sout_r + offs_d[None, :] * sout_d, merged, mask=mask_r[:, None])


def shared_prefix_cfg_attention_ref(
    q_list: list[Any],
    k_loc_list: list[Any],
    v_loc_list: list[Any],
    k_prefix: Any | None,
    v_prefix: Any | None,
    shared_branch_ids: list[int],
    k_suffix_list: list[Any | None],
    v_suffix_list: list[Any | None],
    sm_scale: float | None = None,
    fp32_accum: bool = False,
) -> list[Any]:
    """Exact PyTorch SDPA reference (zero padding, per-branch true length)."""
    import torch
    import torch.nn.functional as F

    num_branches = len(q_list)
    h_q = q_list[0].shape[1]
    h_kv = k_loc_list[0].shape[1]
    head_dim = q_list[0].shape[2]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    shared_set = set(shared_branch_ids)
    outs = []
    for b in range(num_branches):
        parts_k = []
        parts_v = []
        if b in shared_set and k_prefix is not None and k_prefix.shape[0] > 0:
            parts_k.append(k_prefix)
            parts_v.append(v_prefix)
        if k_suffix_list[b] is not None and k_suffix_list[b].shape[0] > 0:
            parts_k.append(k_suffix_list[b])
            parts_v.append(v_suffix_list[b])
        parts_k.append(k_loc_list[b])
        parts_v.append(v_loc_list[b])
        kb = torch.cat(parts_k, dim=0)
        vb = torch.cat(parts_v, dim=0)
        qb = q_list[b]
        if fp32_accum:
            g = h_q // h_kv
            kb_f = kb.float().repeat_interleave(g, dim=1) if g > 1 else kb.float()
            vb_f = vb.float().repeat_interleave(g, dim=1) if g > 1 else vb.float()
            ob = F.scaled_dot_product_attention(
                qb.float().unsqueeze(0).permute(0, 2, 1, 3),
                kb_f.unsqueeze(0).permute(0, 2, 1, 3),
                vb_f.unsqueeze(0).permute(0, 2, 1, 3),
                is_causal=False,
                scale=sm_scale,
            )
            outs.append(ob.permute(0, 2, 1, 3).squeeze(0).to(qb.dtype))
        else:
            ob = F.scaled_dot_product_attention(
                qb.unsqueeze(0).permute(0, 2, 1, 3),
                kb.unsqueeze(0).permute(0, 2, 1, 3),
                vb.unsqueeze(0).permute(0, 2, 1, 3),
                is_causal=False,
                scale=sm_scale,
                enable_gqa=(h_q != h_kv),
            )
            outs.append(ob.permute(0, 2, 1, 3).squeeze(0))
    return outs


def _run_unshared_single_branch(
    qb: Any,
    k_suf: Any | None,
    v_suf: Any | None,
    k_loc: Any,
    v_loc: Any,
    sm_scale: float,
    attn_noncausal: Any | None = None,
) -> Any:
    """Fast unpadded single-branch attention for unshared short branches (e.g. `cfg_img`)."""
    import torch
    import torch.nn.functional as F

    if k_suf is not None and k_suf.shape[0] > 0:
        kb = torch.cat([k_suf, k_loc], dim=0)
        vb = torch.cat([v_suf, v_loc], dim=0)
    else:
        kb = k_loc
        vb = v_loc

    if attn_noncausal is not None:
        return attn_noncausal(qb.unsqueeze(0), kb.unsqueeze(0), vb.unsqueeze(0), None).squeeze(0)

    h_q = qb.shape[1]
    h_kv = kb.shape[1]
    out = F.scaled_dot_product_attention(
        qb.unsqueeze(0).permute(0, 2, 1, 3),
        kb.unsqueeze(0).permute(0, 2, 1, 3),
        vb.unsqueeze(0).permute(0, 2, 1, 3),
        is_causal=False,
        scale=sm_scale,
        enable_gqa=(h_q != h_kv),
    )
    return out.permute(0, 2, 1, 3).squeeze(0)


# Hardware / library capability probe cache (holds only function handles & LSE scale metadata, never tensors).
_FA2_BACKEND_CACHE: dict[tuple[str, int, int, int], tuple[str, float, Any, bool]] = {}
_FA2_BACKEND_ERRORS: list[str] = []


def _get_cu_seqlens(
    lens: list[int],
    device: Any,
    cache: dict[tuple[str, tuple[int, ...]], Any] | None = None,
) -> Any:
    import torch

    key = (str(device), tuple(int(x) for x in lens))
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached
    vals = [0]
    for x in lens:
        vals.append(vals[-1] + int(x))
    t = torch.tensor(vals, dtype=torch.int32, device=device)
    if cache is not None and len(cache) < 256:
        cache[key] = t
    return t


def get_fa2_backend_status() -> dict[str, Any]:
    """Return active FlashPrefix-CFG attention backend info and any probe diagnostics."""
    return {
        "backends": {f"{dev}_hq{hq}_hkv{hkv}_d{d}": kind for (dev, hq, hkv, d), (kind, _, _, _) in _FA2_BACKEND_CACHE.items()},
        "has_active_fa2": any(kind in ("vllm_varlen", "flashinfer") for kind, _, _, _ in _FA2_BACKEND_CACHE.values()),
        "probe_errors": list(_FA2_BACKEND_ERRORS[-8:]),
    }


def _merge_two_lse_states(
    o_a: Any,
    lse_a: Any,
    o_b: Any,
    lse_b: Any,
    out: Any | None = None,
    *,
    lse_transposed: bool = False,
    lse_scale: float = 1.0,
) -> Any:
    """Exact log-sum-exp online-softmax merge for `[M, H_q, D]` outputs and `[M, H_q]` or `[H_q, M]` LSEs."""
    import torch

    m_tot, h_q, d = o_a.shape
    n_rows = m_tot * h_q
    oa_2d = o_a.reshape(n_rows, d).contiguous()
    ob_2d = o_b.reshape(n_rows, d).contiguous()
    if lse_a.ndim == 3 and lse_a.shape[0] == 1:
        lse_a = lse_a.squeeze(0)
    if lse_b.ndim == 3 and lse_b.shape[0] == 1:
        lse_b = lse_b.squeeze(0)

    if HAS_TRITON and oa_2d.is_cuda and d in (64, 128) and lse_a.ndim == 2 and lse_b.ndim == 2:
        if lse_transposed:
            sla_m, sla_h = int(lse_a.stride(1)), int(lse_a.stride(0))
            slb_m, slb_h = int(lse_b.stride(1)), int(lse_b.stride(0))
        else:
            sla_m, sla_h = int(lse_a.stride(0)), int(lse_a.stride(1))
            slb_m, slb_h = int(lse_b.stride(0)), int(lse_b.stride(1))
        can_write_out_direct = (
            out is not None
            and out.shape == (m_tot, h_q, d)
            and out.dtype == oa_2d.dtype
            and out.device == oa_2d.device
            and out.is_contiguous()
        )
        out_2d = out.view(n_rows, d) if can_write_out_direct else torch.empty_like(oa_2d)
        block_r = 64 if n_rows >= 256 else 16
        grid = (triton.cdiv(n_rows, block_r),)
        _merge_two_lse_fwd[grid](
            oa_2d,
            lse_a,
            ob_2d,
            lse_b,
            out_2d,
            oa_2d.stride(0),
            oa_2d.stride(1),
            ob_2d.stride(0),
            ob_2d.stride(1),
            out_2d.stride(0),
            out_2d.stride(1),
            sla_m,
            sla_h,
            slb_m,
            slb_h,
            float(lse_scale),
            float(lse_scale),
            n_rows,
            H_Q=h_q,
            BLOCK_R=block_r,
            HEAD_DIM=d,
            num_warps=4,
        )
        res_3d = out_2d.reshape(m_tot, h_q, d)
        if out is not None and not can_write_out_direct:
            out.copy_(res_3d)
            return out
        return res_3d

    la_n = _normalize_lse_shape(lse_a, m_tot, h_q, lse_scale, lse_transposed=lse_transposed)
    lb_n = _normalize_lse_shape(lse_b, m_tot, h_q, lse_scale, lse_transposed=lse_transposed)
    m = torch.clamp_min(torch.maximum(la_n, lb_n), -1.0e30)
    wa = torch.exp(la_n - m)
    wb = torch.exp(lb_n - m)
    inv = 1.0 / (wa + wb)
    merged = (o_a.float() * (wa * inv).unsqueeze(-1) + o_b.float() * (wb * inv).unsqueeze(-1)).to(o_a.dtype)
    if out is not None:
        out.copy_(merged)
        return out
    return merged


def _normalize_lse_shape(
    lse: Any,
    total_q: int,
    h_q: int,
    lse_scale: float = 1.0,
    lse_transposed: bool | None = None,
) -> Any:
    if lse.ndim == 3 and lse.shape[0] == 1:
        lse = lse.squeeze(0)
    if lse.ndim == 2:
        if lse_transposed is True:
            lse = lse.transpose(0, 1).contiguous()
        elif lse_transposed is None and lse.shape == (h_q, total_q) and h_q != total_q:
            lse = lse.transpose(0, 1).contiguous()
    out = lse.float()
    if lse_scale != 1.0:
        out = out * lse_scale
    return out


def _probe_fa2_backend(device: Any, dtype: Any, h_q: int, h_kv: int, head_dim: int) -> tuple[str, float, Any, bool]:
    """Probe and self-verify vllm_flash_attn / flashinfer LSE backends against FP32 ground truth."""
    key = (str(device), h_q, h_kv, head_dim)
    cached = _FA2_BACKEND_CACHE.get(key)
    if cached is not None:
        return cached

    import inspect
    import torch
    import torch.nn.functional as F

    ln2 = 0.6931471805599453
    m_test, n_test = 8, 12
    sm_scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn((m_test, h_q, head_dim), dtype=dtype, device=device) * 0.5
    k = torch.randn((n_test, h_kv, head_dim), dtype=dtype, device=device) * 0.5
    v = torch.randn((n_test, h_kv, head_dim), dtype=dtype, device=device) * 0.5

    g = h_q // h_kv
    kf = k.float().repeat_interleave(g, dim=1) if g > 1 else k.float()
    vf = v.float().repeat_interleave(g, dim=1) if g > 1 else v.float()
    scores = torch.einsum("qhd,khd->hqk", q.float(), kf) * sm_scale
    lse_ref_ln = torch.logsumexp(scores, dim=-1).transpose(0, 1).contiguous()
    out_ref = torch.einsum("hqk,khd->qhd", F.softmax(scores, dim=-1), vf).contiguous()

    for mod_name in ("vllm_omni.diffusion.attention.backends.utils.fa", "flash_attn", "vllm.vllm_flash_attn"):
        try:
            mod = __import__(mod_name, fromlist=["flash_attn_varlen_func"])
            fn = getattr(mod, "flash_attn_varlen_func", None)
            if fn is None:
                continue
            sig = inspect.signature(fn).parameters
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values())
            if "return_softmax_lse" not in sig and not has_var_kw:
                _FA2_BACKEND_ERRORS.append(f"{mod_name}: no return_softmax_lse in signature {list(sig)}")
                continue
            cu_q = torch.tensor([0, m_test], dtype=torch.int32, device=device)
            cu_k = torch.tensor([0, n_test], dtype=torch.int32, device=device)
            res = fn(
                q,
                k,
                v,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=m_test,
                max_seqlen_k=n_test,
                softmax_scale=sm_scale,
                causal=False,
                return_softmax_lse=True,
            )
            if isinstance(res, tuple) and len(res) >= 2:
                out_c, lse_c = res[0], res[1]
                out_err = float((out_c.float() - out_ref).abs().max().item())
                lse_transposed = bool(lse_c.ndim >= 2 and lse_c.shape[-2] == h_q and lse_c.shape[-1] == m_test)
                for scale in (1.0, ln2):
                    lse_n = _normalize_lse_shape(lse_c, m_test, h_q, scale, lse_transposed=lse_transposed)
                    lse_err = float((lse_n - lse_ref_ln).abs().max().item()) if lse_n.shape == lse_ref_ln.shape else 999.0
                    if out_err < 0.05 and lse_err < 0.05:
                        entry = ("vllm_varlen", scale, fn, lse_transposed)
                        _FA2_BACKEND_CACHE[key] = entry
                        return entry
                _FA2_BACKEND_ERRORS.append(f"{mod_name}: mismatch out_err={out_err:.4f} lse_shape={tuple(lse_c.shape)}")
            else:
                _FA2_BACKEND_ERRORS.append(f"{mod_name}: expected tuple>=2, got {type(res).__name__}")
        except Exception as exc:
            _FA2_BACKEND_ERRORS.append(f"{mod_name}: {type(exc).__name__}: {exc}")
            continue

    try:
        from flashinfer import single_prefill_with_kv_cache

        res = single_prefill_with_kv_cache(q, k, v, causal=False, sm_scale=sm_scale, return_lse=True)
        if isinstance(res, tuple) and len(res) >= 2:
            out_c, lse_c = res[0], res[1]
            out_err = float((out_c.float() - out_ref).abs().max().item())
            lse_transposed = bool(lse_c.ndim >= 2 and lse_c.shape[-2] == h_q and lse_c.shape[-1] == m_test)
            for scale in (ln2, 1.0):
                lse_n = _normalize_lse_shape(lse_c, m_test, h_q, scale, lse_transposed=lse_transposed)
                lse_err = float((lse_n - lse_ref_ln).abs().max().item()) if lse_n.shape == lse_ref_ln.shape else 999.0
                if out_err < 0.05 and lse_err < 0.05:
                    entry = ("flashinfer", scale, single_prefill_with_kv_cache, lse_transposed)
                    _FA2_BACKEND_CACHE[key] = entry
                    return entry
            _FA2_BACKEND_ERRORS.append(f"flashinfer: mismatch out_err={out_err:.4f} lse_shape={tuple(lse_c.shape)}")
    except Exception as exc:
        _FA2_BACKEND_ERRORS.append(f"flashinfer: {type(exc).__name__}: {exc}")

    entry = ("none", 1.0, None, False)
    _FA2_BACKEND_CACHE[key] = entry
    return entry


def _run_varlen_fa2_with_lse(
    backend: tuple[str, float, Any, bool],
    q_cat: Any,
    k_cat: Any,
    v_cat: Any,
    q_lens: list[int],
    k_lens: list[int],
    sm_scale: float,
    *,
    cu_seqlens_cache: dict[tuple[str, tuple[int, ...]], Any] | None = None,
    raw_lse: bool = False,
) -> tuple[Any, Any]:
    import torch

    kind, lse_scale, fn, lse_transposed = backend
    h_q = int(q_cat.shape[1])
    total_q = int(q_cat.shape[0])

    if kind == "vllm_varlen":
        cu_q = _get_cu_seqlens(q_lens, q_cat.device, cache=cu_seqlens_cache)
        cu_k = _get_cu_seqlens(k_lens, q_cat.device, cache=cu_seqlens_cache)
        out, lse = fn(
            q_cat,
            k_cat,
            v_cat,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max(k_lens),
            softmax_scale=sm_scale,
            causal=False,
            return_softmax_lse=True,
        )
        if raw_lse:
            return out, lse
        return out, _normalize_lse_shape(lse, total_q, h_q, lse_scale, lse_transposed=lse_transposed)

    if len(q_lens) == 1:
        out, lse = fn(q_cat, k_cat, v_cat, causal=False, sm_scale=sm_scale, return_lse=True)
        if raw_lse:
            return out, lse
        return out, _normalize_lse_shape(lse, total_q, h_q, lse_scale, lse_transposed=lse_transposed)

    outs, lses = [], []
    q_off = 0
    k_off = 0
    for ql, kl in zip(q_lens, k_lens):
        oi, li = fn(
            q_cat[q_off : q_off + ql],
            k_cat[k_off : k_off + kl],
            v_cat[k_off : k_off + kl],
            causal=False,
            sm_scale=sm_scale,
            return_lse=True,
        )
        outs.append(oi)
        lses.append(_normalize_lse_shape(li, ql, h_q, lse_scale, lse_transposed=lse_transposed))
        q_off += ql
        k_off += kl
    return torch.cat(outs, dim=0), torch.cat(lses, dim=0)


def _run_flash_prefix_cfg(
    q_list: list[Any],
    k_loc_list: list[Any],
    v_loc_list: list[Any],
    k_prefix: Any,
    v_prefix: Any,
    shared_branch_ids: list[int],
    k_suffix_list: list[Any | None],
    v_suffix_list: list[Any | None],
    sm_scale: float,
    *,
    cu_seqlens_cache: dict[tuple[str, tuple[int, ...]], Any] | None = None,
) -> list[Any] | None:
    """FlashPrefix-CFG: Single-read FA2 on shared prefix + compact varlen tail + fused zero-transpose Triton LSE merge."""
    q_len, h_q, head_dim = q_list[0].shape
    h_kv = k_loc_list[0].shape[1]
    backend = _probe_fa2_backend(q_list[0].device, q_list[0].dtype, h_q, h_kv, head_dim)
    if backend[0] == "none":
        return None
    import torch

    kind, lse_scale, _, lse_transposed = backend
    use_raw_lse = kind == "vllm_varlen"
    num_branches = len(q_list)
    b_sh = list(shared_branch_ids)
    shared_set = set(b_sh)
    unshared_ids = [b for b in range(num_branches) if b not in shared_set]
    ordered_ids = b_sh + unshared_ids
    n_sh_rows = len(b_sh) * q_len

    # Allocate packed Q once; shared-prefix pass slices a zero-copy view `q_all[:n_sh_rows]`.
    q_all = torch.cat([q_list[b] for b in ordered_ids], dim=0)
    q_shared_cat = q_all[:n_sh_rows]

    # Stage 1: Single HBM read of (k_prefix, v_prefix) for all shared branches packed along Q.
    kp = k_prefix if k_prefix.is_contiguous() else k_prefix.contiguous()
    vp = v_prefix if v_prefix.is_contiguous() else v_prefix.contiguous()
    o_pre_cat, lse_pre_cat = _run_varlen_fa2_with_lse(
        backend,
        q_shared_cat,
        kp,
        vp,
        [n_sh_rows],
        [int(kp.shape[0])],
        sm_scale,
        cu_seqlens_cache=cu_seqlens_cache,
        raw_lse=use_raw_lse,
    )

    # Stage 2: Compact batched unpadded tail pass for all shared + unshared branches (flat single cat).
    tail_k_parts, tail_v_parts, tail_lens = [], [], []
    for b in ordered_ids:
        ks, vs = k_suffix_list[b], v_suffix_list[b]
        kl, vl = k_loc_list[b], v_loc_list[b]
        ln = int(kl.shape[0])
        if ks is not None and vs is not None and ks.shape[0] > 0:
            tail_k_parts.append(ks)
            tail_v_parts.append(vs)
            ln += int(ks.shape[0])
        tail_k_parts.append(kl)
        tail_v_parts.append(vl)
        tail_lens.append(ln)
    k_tail_cat = torch.cat(tail_k_parts, dim=0)
    v_tail_cat = torch.cat(tail_v_parts, dim=0)
    o_tail_all, lse_tail_all = _run_varlen_fa2_with_lse(
        backend,
        q_all,
        k_tail_cat,
        v_tail_cat,
        [q_len] * len(ordered_ids),
        tail_lens,
        sm_scale,
        cu_seqlens_cache=cu_seqlens_cache,
        raw_lse=use_raw_lse,
    )

    # Stage 3: In-place exact FP32 online-softmax LSE merge directly into `o_tail_all[:n_sh_rows]`.
    o_tail_sh = o_tail_all[:n_sh_rows]
    if use_raw_lse and lse_transposed:
        lse_tail_sh = lse_tail_all[..., :n_sh_rows]
        eff_transposed = True
        eff_scale = lse_scale
    elif use_raw_lse:
        lse_tail_sh = lse_tail_all[:n_sh_rows]
        eff_transposed = False
        eff_scale = lse_scale
    else:
        lse_tail_sh = lse_tail_all[:n_sh_rows]
        eff_transposed = False
        eff_scale = 1.0
    o_merged = _merge_two_lse_states(
        o_pre_cat,
        lse_pre_cat,
        o_tail_sh,
        lse_tail_sh,
        out=o_tail_sh,
        lse_transposed=eff_transposed,
        lse_scale=eff_scale,
    )
    outputs: list[Any] = [None] * num_branches
    for idx_b, b in enumerate(b_sh):
        outputs[b] = o_merged[idx_b * q_len : (idx_b + 1) * q_len]
    for idx_u, b in enumerate(unshared_ids):
        off = n_sh_rows + idx_u * q_len
        outputs[b] = o_tail_all[off : off + q_len]
    return outputs


def should_use_cascade_cfg_split(q_len: int, l_shared: int, num_shared_branches: int) -> bool:
    """Gate two-pass FlashPrefix-CFG + fused LSE merge on empirically calibrated arithmetic-intensity payoff."""
    if q_len <= 0 or num_shared_branches < 2 or l_shared < 256:
        return False
    if q_len <= 2048:
        return l_shared >= max(256, int(1.5 * q_len))
    return l_shared >= 4 * q_len


# Canonical FlashPrefix-CFG gate name matching `GLOSSARY.md`
should_use_flash_prefix_split = should_use_cascade_cfg_split


def _get_cached_left_pad_meta(
    lens: tuple[int, ...],
    device: Any,
    cache: dict[tuple[str, tuple[int, ...]], Any] | None = None,
) -> Any:
    """Return cached DiffusionAttentionMetadata(attn_mask) without per-layer mask alloc/fill."""
    import torch

    max_len = max(lens)
    if all(ln == max_len for ln in lens):
        return None
    key = (str(device), lens)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached
    mask = torch.zeros((len(lens), max_len), dtype=torch.bool, device=device)
    for i, ln in enumerate(lens):
        mask[i, max_len - ln :] = True
    try:
        from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata as DiffusionAttentionMetadata

        meta = DiffusionAttentionMetadata(attn_mask=mask)
    except Exception:
        meta = None
    if meta is not None and cache is not None and len(cache) < 256:
        cache[key] = meta
    return meta


def _supports_packed_varlen_attn(attn_noncausal: Any) -> bool:
    """Return True if `attn_noncausal` exposes vLLM-Omni's packed-varlen FlashAttention path."""
    if attn_noncausal is None:
        return False
    impl = getattr(attn_noncausal, "attention", None)
    return callable(getattr(impl, "_forward_varlen_packed", None))


def _run_packed_varlen_noncausal_cfg(
    q_list: list[Any],
    k_loc_list: list[Any],
    v_loc_list: list[Any],
    k_prefix: Any | None,
    v_prefix: Any | None,
    shared_branch_ids: list[int],
    k_suffix_list: list[Any | None],
    v_suffix_list: list[Any | None],
    attn_noncausal: Any,
    *,
    cu_seqlens_cache: dict[tuple[str, tuple[int, ...]], Any] | None = None,
) -> list[Any] | None:
    """Zero-padding packed varlen dispatch through `attn_noncausal`'s native `_forward_varlen_packed`.

    Invokes the exact same `flash_attn_varlen_func` kernel and `(cu_seqlens_q, cu_seqlens_k)`
    as stock `_forward_varlen_masked`, preserving 100% bit-exact parity while eliminating
    left-padding, mask allocation, and `_unpad_input`/`_index_first_axis` copies.
    """
    if not _supports_packed_varlen_attn(attn_noncausal):
        return None
    try:
        from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata as DiffusionAttentionMetadata
    except Exception:
        return None
    import torch

    num_branches = len(q_list)
    q_len = int(q_list[0].shape[0])
    shared_set = set(shared_branch_ids)
    has_pre = k_prefix is not None and v_prefix is not None and int(k_prefix.shape[0]) > 0

    q_all = torch.cat(q_list, dim=0)
    k_parts: list[Any] = []
    v_parts: list[Any] = []
    k_lens: list[int] = []
    for b in range(num_branches):
        ln = 0
        if b in shared_set and has_pre:
            k_parts.append(k_prefix)
            v_parts.append(v_prefix)
            ln += int(k_prefix.shape[0])
        ks, vs = k_suffix_list[b], v_suffix_list[b]
        if ks is not None and vs is not None and int(ks.shape[0]) > 0:
            k_parts.append(ks)
            v_parts.append(vs)
            ln += int(ks.shape[0])
        kl, vl = k_loc_list[b], v_loc_list[b]
        k_parts.append(kl)
        v_parts.append(vl)
        ln += int(kl.shape[0])
        k_lens.append(ln)

    k_all = torch.cat(k_parts, dim=0)
    v_all = torch.cat(v_parts, dim=0)
    cu_q = _get_cu_seqlens([q_len] * num_branches, q_all.device, cache=cu_seqlens_cache)
    cu_k = _get_cu_seqlens(k_lens, q_all.device, cache=cu_seqlens_cache)
    meta = DiffusionAttentionMetadata(
        extra={
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_q": q_len,
            "max_seqlen_k": max(k_lens),
        }
    )
    out = attn_noncausal(q_all.unsqueeze(0), k_all.unsqueeze(0), v_all.unsqueeze(0), meta)
    if out.ndim == 4 and out.shape[0] == 1:
        out = out.squeeze(0)
    return [out[b * q_len : (b + 1) * q_len] for b in range(num_branches)]


def _run_unpadded_varlen_cfg(
    q_list: list[Any],
    k_loc_list: list[Any],
    v_loc_list: list[Any],
    k_prefix: Any | None,
    v_prefix: Any | None,
    shared_branch_ids: list[int],
    k_suffix_list: list[Any | None],
    v_suffix_list: list[Any | None],
    sm_scale: float,
    *,
    cu_seqlens_cache: dict[tuple[str, tuple[int, ...]], Any] | None = None,
) -> list[Any] | None:
    """Single-pass unpadded varlen FA2 for multi-branch CFG below two-pass split threshold (zero left-padding)."""
    q_len, h_q, head_dim = q_list[0].shape
    h_kv = k_loc_list[0].shape[1]
    backend = _probe_fa2_backend(q_list[0].device, q_list[0].dtype, h_q, h_kv, head_dim)
    if backend[0] != "vllm_varlen":
        return None
    import torch

    _, _, fn, _ = backend
    num_branches = len(q_list)
    shared_set = set(shared_branch_ids)
    has_pre = k_prefix is not None and v_prefix is not None and int(k_prefix.shape[0]) > 0

    q_all = torch.cat(q_list, dim=0).contiguous()
    k_parts: list[Any] = []
    v_parts: list[Any] = []
    k_lens: list[int] = []
    for b in range(num_branches):
        ln = 0
        if b in shared_set and has_pre:
            k_parts.append(k_prefix)
            v_parts.append(v_prefix)
            ln += int(k_prefix.shape[0])
        ks, vs = k_suffix_list[b], v_suffix_list[b]
        if ks is not None and vs is not None and int(ks.shape[0]) > 0:
            k_parts.append(ks)
            v_parts.append(vs)
            ln += int(ks.shape[0])
        kl, vl = k_loc_list[b], v_loc_list[b]
        k_parts.append(kl)
        v_parts.append(vl)
        ln += int(kl.shape[0])
        k_lens.append(ln)

    k_all = torch.cat(k_parts, dim=0).contiguous()
    v_all = torch.cat(v_parts, dim=0).contiguous()
    q_lens = [q_len] * num_branches
    cu_q = _get_cu_seqlens(q_lens, q_all.device, cache=cu_seqlens_cache)
    cu_k = _get_cu_seqlens(k_lens, q_all.device, cache=cu_seqlens_cache)
    out = fn(
        q_all,
        k_all,
        v_all,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=q_len,
        max_seqlen_k=max(k_lens),
        softmax_scale=sm_scale,
        causal=False,
        return_softmax_lse=False,
    )
    if isinstance(out, tuple):
        out = out[0]
    return [out[b * q_len : (b + 1) * q_len] for b in range(num_branches)]


def _run_direct_workspace_noncausal_cfg(
    q_list: list[Any],
    k_loc_list: list[Any],
    v_loc_list: list[Any],
    k_prefix: Any | None,
    v_prefix: Any | None,
    shared_branch_ids: list[int],
    k_suffix_list: list[Any | None],
    v_suffix_list: list[Any | None],
    attn_noncausal: Any,
    *,
    direct_ws_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
    attn_meta_cache: dict[tuple[str, tuple[int, ...]], Any] | None = None,
) -> list[Any]:
    """Single-copy direct write into 1 reusable layer workspace + cached mask metadata."""
    import torch

    num_branches = len(q_list)
    q0 = q_list[0]
    q_len, h_q, head_dim = q0.shape
    h_kv = k_loc_list[0].shape[1]
    shared_set = set(shared_branch_ids)
    l_shared = int(k_prefix.shape[0]) if k_prefix is not None else 0

    lens_list: list[int] = []
    for b in range(num_branches):
        ln = int(k_loc_list[b].shape[0])
        if b in shared_set and l_shared > 0 and k_prefix is not None:
            ln += l_shared
        ks = k_suffix_list[b]
        if ks is not None and ks.shape[0] > 0:
            ln += int(ks.shape[0])
        lens_list.append(ln)
    lens = tuple(lens_list)
    max_len = max(lens)

    wkey = (str(q0.device), str(q0.dtype))
    ws = direct_ws_cache.get(wkey) if direct_ws_cache is not None else None
    q_shape = (num_branches, q_len, h_q, head_dim)
    kv_shape = (num_branches, max_len, h_kv, head_dim)
    allocated_fresh = False
    if (
        ws is None
        or ws["q"].shape != q_shape
        or ws["k"].shape[0] != num_branches
        or ws["k"].shape[1] < max_len
        or ws["k"].shape[2:] != (h_kv, head_dim)
    ):
        ws = {
            "q": torch.empty(q_shape, dtype=q0.dtype, device=q0.device),
            "k": torch.empty(kv_shape, dtype=k_loc_list[0].dtype, device=q0.device),
            "v": torch.empty(kv_shape, dtype=v_loc_list[0].dtype, device=q0.device),
            "last_lens": lens,
            "last_max": max_len,
        }
        allocated_fresh = True
        if direct_ws_cache is not None:
            direct_ws_cache[wkey] = ws

    q_buf = ws["q"]
    k_buf = ws["k"][:, :max_len]
    v_buf = ws["v"][:, :max_len]
    need_zero_pad = allocated_fresh or (ws["last_lens"] != lens) or (ws["last_max"] != max_len)
    if need_zero_pad:
        ws["last_lens"] = lens
        ws["last_max"] = max_len

    for b in range(num_branches):
        q_buf[b].copy_(q_list[b])
        pad = max_len - lens[b]
        if need_zero_pad and pad > 0:
            k_buf[b, :pad].zero_()
            v_buf[b, :pad].zero_()
        pos = pad
        if b in shared_set and l_shared > 0 and k_prefix is not None and v_prefix is not None:
            k_buf[b, pos : pos + l_shared].copy_(k_prefix)
            v_buf[b, pos : pos + l_shared].copy_(v_prefix)
            pos += l_shared
        ks, vs = k_suffix_list[b], v_suffix_list[b]
        if ks is not None and vs is not None and ks.shape[0] > 0:
            ls = int(ks.shape[0])
            k_buf[b, pos : pos + ls].copy_(ks)
            v_buf[b, pos : pos + ls].copy_(vs)
            pos += ls
        kl, vl = k_loc_list[b], v_loc_list[b]
        ll = int(kl.shape[0])
        k_buf[b, pos : pos + ll].copy_(kl)
        v_buf[b, pos : pos + ll].copy_(vl)

    meta = _get_cached_left_pad_meta(lens, q0.device, cache=attn_meta_cache)
    o_4d = attn_noncausal(q_buf, k_buf, v_buf, meta)
    return [o_4d[b] for b in range(num_branches)]


def shared_prefix_cfg_attention(
    q_list: list[Any],
    k_loc_list: list[Any],
    v_loc_list: list[Any],
    k_prefix: Any | None,
    v_prefix: Any | None,
    shared_branch_ids: list[int],
    k_suffix_list: list[Any | None],
    v_suffix_list: list[Any | None],
    sm_scale: float | None = None,
    attn_noncausal: Any | None = None,
    backend: str = "cascade_cfg",
    stats_out: dict[str, int] | None = None,
    *,
    direct_ws_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
    cu_seqlens_cache: dict[tuple[str, tuple[int, ...]], Any] | None = None,
    attn_meta_cache: dict[tuple[str, tuple[int, ...]], Any] | None = None,
) -> list[Any]:
    """Dispatch multi-branch CFG attention via arithmetic-intensity-gated FlashPrefix-CFG or direct workspace."""
    import torch

    num_branches = len(q_list)
    q_len, _, head_dim = q_list[0].shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    l_shared = int(k_prefix.shape[0]) if k_prefix is not None else 0

    if q_list[0].is_cuda:
        if backend in ("cascade_cfg", "flash_prefix"):
            if should_use_flash_prefix_split(q_len, l_shared, len(shared_branch_ids)):
                fp_out = _run_flash_prefix_cfg(
                    q_list=q_list,
                    k_loc_list=k_loc_list,
                    v_loc_list=v_loc_list,
                    k_prefix=k_prefix,
                    v_prefix=v_prefix,
                    shared_branch_ids=shared_branch_ids,
                    k_suffix_list=k_suffix_list,
                    v_suffix_list=v_suffix_list,
                    sm_scale=sm_scale,
                    cu_seqlens_cache=cu_seqlens_cache,
                )
                if fp_out is not None:
                    if stats_out is not None:
                        stats_out["cascade_split_executed"] = stats_out.get("cascade_split_executed", 0) + 1
                    return fp_out
            else:
                pv_out = _run_packed_varlen_noncausal_cfg(
                    q_list=q_list,
                    k_loc_list=k_loc_list,
                    v_loc_list=v_loc_list,
                    k_prefix=k_prefix,
                    v_prefix=v_prefix,
                    shared_branch_ids=shared_branch_ids,
                    k_suffix_list=k_suffix_list,
                    v_suffix_list=v_suffix_list,
                    attn_noncausal=attn_noncausal,
                    cu_seqlens_cache=cu_seqlens_cache,
                )
                if pv_out is not None:
                    return pv_out
                if attn_noncausal is None:
                    vp_out = _run_unpadded_varlen_cfg(
                        q_list=q_list,
                        k_loc_list=k_loc_list,
                        v_loc_list=v_loc_list,
                        k_prefix=k_prefix,
                        v_prefix=v_prefix,
                        shared_branch_ids=shared_branch_ids,
                        k_suffix_list=k_suffix_list,
                        v_suffix_list=v_suffix_list,
                        sm_scale=sm_scale,
                        cu_seqlens_cache=cu_seqlens_cache,
                    )
                    if vp_out is not None:
                        return vp_out

        if backend in ("lossless_varlen", "unpadded_varlen"):
            pv_out = _run_packed_varlen_noncausal_cfg(
                q_list=q_list,
                k_loc_list=k_loc_list,
                v_loc_list=v_loc_list,
                k_prefix=k_prefix,
                v_prefix=v_prefix,
                shared_branch_ids=shared_branch_ids,
                k_suffix_list=k_suffix_list,
                v_suffix_list=v_suffix_list,
                attn_noncausal=attn_noncausal,
                cu_seqlens_cache=cu_seqlens_cache,
            )
            if pv_out is not None:
                return pv_out

        if attn_noncausal is not None and backend in ("cascade_cfg", "flash_prefix", "lossless_varlen", "unpadded_varlen"):
            return _run_direct_workspace_noncausal_cfg(
                q_list=q_list,
                k_loc_list=k_loc_list,
                v_loc_list=v_loc_list,
                k_prefix=k_prefix,
                v_prefix=v_prefix,
                shared_branch_ids=shared_branch_ids,
                k_suffix_list=k_suffix_list,
                v_suffix_list=v_suffix_list,
                attn_noncausal=attn_noncausal,
                direct_ws_cache=direct_ws_cache,
                attn_meta_cache=attn_meta_cache,
            )

    outputs: list[Any] = [None] * num_branches
    for b in range(num_branches):
        k_suf, v_suf = k_suffix_list[b], v_suffix_list[b]
        if b in shared_branch_ids and l_shared > 0 and k_prefix is not None and v_prefix is not None:
            k_suf = torch.cat([k_prefix, k_suf], dim=0) if (k_suf is not None and k_suf.shape[0] > 0) else k_prefix
            v_suf = torch.cat([v_prefix, v_suf], dim=0) if (v_suf is not None and v_suf.shape[0] > 0) else v_prefix
        outputs[b] = _run_unshared_single_branch(
            q_list[b], k_suf, v_suf, k_loc_list[b], v_loc_list[b], sm_scale, attn_noncausal=attn_noncausal
        )
    return outputs
