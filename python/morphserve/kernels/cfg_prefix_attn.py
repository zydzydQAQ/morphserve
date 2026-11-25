"""Arithmetic-Intensity-Gated Cascade-CFG Attention, Fused LSE Merge & BAGEL runtime hooks.

Module 3 Cross-Paradigm State De-Redundancy combines:
  1. `LECP CoW` (`SharedPrefixCache`) eliminating ~50% multi-turn CFG KV HBM duplication;
  2. `GpuInPlaceLoopback` eliminating CPU PIL/PNG round-trips across editing/Visual-CoT turns;
  3. Arithmetic-intensity-gated two-pass `Cascade-CFG` attention (`_run_flash_prefix_cfg` +
     fused Triton `_merge_two_lse_fwd`), automatically falling back to lossless unpadded
     attention when `Q` is large relative to `L_shared` (e.g., high-res `1024^2` early turns).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import math
from typing import Any

from .lecp_cache import SharedPrefixCache

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
        N_ROWS,
        BLOCK_R: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """Fused exact online-softmax LSE merge for Cascade-CFG (zero autotune stall)."""
        pid_r = tl.program_id(0)
        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_d = tl.arange(0, HEAD_DIM)
        mask_r = offs_r < N_ROWS

        la = tl.load(Lse_a + offs_r, mask=mask_r, other=-1.0e20)
        lb = tl.load(Lse_b + offs_r, mask=mask_r, other=-1.0e20)
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


_FA2_BACKEND_CACHE: dict[tuple[str, int, int, int], tuple[str, float, Any, bool]] = {}
_FA2_BACKEND_ERRORS: list[str] = []
_CU_SEQLENS_CACHE: dict[tuple[str, tuple[int, ...]], Any] = {}
_ATTN_META_CACHE: dict[tuple[str, tuple[int, ...]], Any] = {}
_DIRECT_WS_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _get_cu_seqlens(lens: list[int], device: Any) -> Any:
    import torch

    key = (str(device), tuple(int(x) for x in lens))
    cached = _CU_SEQLENS_CACHE.get(key)
    if cached is not None:
        return cached
    vals = [0]
    for x in lens:
        vals.append(vals[-1] + int(x))
    t = torch.tensor(vals, dtype=torch.int32, device=device)
    if len(_CU_SEQLENS_CACHE) < 256:
        _CU_SEQLENS_CACHE[key] = t
    return t


def get_fa2_backend_status() -> dict[str, Any]:
    """Return active FlashPrefix-CFG attention backend info and any probe diagnostics."""
    return {
        "backends": {f"{dev}_hq{hq}_hkv{hkv}_d{d}": kind for (dev, hq, hkv, d), (kind, _, _, _) in _FA2_BACKEND_CACHE.items()},
        "has_active_fa2": any(kind in ("vllm_varlen", "flashinfer") for kind, _, _, _ in _FA2_BACKEND_CACHE.values()),
        "probe_errors": list(_FA2_BACKEND_ERRORS[-8:]),
    }


def _merge_two_lse_states(o_a: Any, lse_a: Any, o_b: Any, lse_b: Any, out: Any | None = None) -> Any:
    """Exact log-sum-exp online-softmax merge for `[M, H_q, D]` outputs and `[M, H_q]` ln-LSEs."""
    import torch

    m_tot, h_q, d = o_a.shape
    n_rows = m_tot * h_q
    oa_2d = o_a.reshape(n_rows, d).contiguous()
    ob_2d = o_b.reshape(n_rows, d).contiguous()
    la_1d = lse_a.reshape(n_rows).contiguous()
    lb_1d = lse_b.reshape(n_rows).contiguous()

    if HAS_TRITON and oa_2d.is_cuda and d in (64, 128):
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
            la_1d,
            ob_2d,
            lb_1d,
            out_2d,
            oa_2d.stride(0),
            oa_2d.stride(1),
            ob_2d.stride(0),
            ob_2d.stride(1),
            out_2d.stride(0),
            out_2d.stride(1),
            n_rows,
            BLOCK_R=block_r,
            HEAD_DIM=d,
            num_warps=4,
        )
        res_3d = out_2d.reshape(m_tot, h_q, d)
        if out is not None and not can_write_out_direct:
            out.copy_(res_3d)
            return out
        return res_3d

    m = torch.clamp_min(torch.maximum(lse_a, lse_b), -1.0e30)
    wa = torch.exp(lse_a - m)
    wb = torch.exp(lse_b - m)
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

    for mod_name in ("flash_attn", "vllm.vllm_flash_attn"):
        try:
            mod = __import__(mod_name, fromlist=["flash_attn_varlen_func"])
            fn = getattr(mod, "flash_attn_varlen_func", None)
            if fn is None:
                continue
            sig = inspect.signature(fn).parameters
            if "return_softmax_lse" not in sig:
                _FA2_BACKEND_ERRORS.append(f"{mod_name}: no return_softmax_lse in signature {list(sig)}")
                continue
            cu_q = torch.tensor([0, m_test], dtype=torch.int32, device=device)
            cu_k = torch.tensor([0, n_test], dtype=torch.int32, device=device)
            # vllm.vllm_flash_attn reverses (max_seqlen, cu_seqlens) order vs flash_attn; pass via keywords!
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
) -> tuple[Any, Any]:
    import torch

    kind, lse_scale, fn, lse_transposed = backend
    h_q = int(q_cat.shape[1])
    total_q = int(q_cat.shape[0])

    if kind == "vllm_varlen":
        cu_q = _get_cu_seqlens(q_lens, q_cat.device)
        cu_k = _get_cu_seqlens(k_lens, q_cat.device)
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
        return out, _normalize_lse_shape(lse, total_q, h_q, lse_scale, lse_transposed=lse_transposed)

    if len(q_lens) == 1:
        out, lse = fn(q_cat, k_cat, v_cat, causal=False, sm_scale=sm_scale, return_lse=True)
        return out, _normalize_lse_shape(lse, total_q, h_q, lse_scale, lse_transposed=lse_transposed)

    outs, lses = [], []
    q_off = 0
    k_off = 0
    for ql, kl in zip(q_lens, k_lens):
        oi, li = fn(
            q_cat[q_off : q_off + ql],
            k_cat[k_off : k_off + kl],
            v_cat[k_off : k_off + kl],
            causal=False, sm_scale=sm_scale, return_lse=True,
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
    attn_noncausal: Any | None = None,
) -> list[Any] | None:
    """FlashPrefix-CFG: Single-read FA2 on shared prefix + compact varlen tail + fused Triton LSE merge."""
    q_len, h_q, head_dim = q_list[0].shape
    h_kv = k_loc_list[0].shape[1]
    backend = _probe_fa2_backend(q_list[0].device, q_list[0].dtype, h_q, h_kv, head_dim)
    if backend[0] == "none":
        return None
    import torch

    num_branches = len(q_list)
    b_sh = list(shared_branch_ids)
    shared_set = set(b_sh)
    unshared_ids = [b for b in range(num_branches) if b not in shared_set]
    ordered_ids = b_sh + unshared_ids
    n_sh_rows = len(b_sh) * q_len

    # Allocate packed Q once; shared-prefix pass slices a zero-copy view `q_all[:n_sh_rows]`!
    q_all = torch.cat([q_list[b] for b in ordered_ids], dim=0).contiguous()
    q_shared_cat = q_all[:n_sh_rows]

    # Stage 1: Single HBM read of (k_prefix, v_prefix) for all shared branches packed along Q!
    kp, vp = k_prefix.contiguous(), v_prefix.contiguous()
    o_pre_cat, lse_pre_cat = _run_varlen_fa2_with_lse(
        backend, q_shared_cat, kp, vp, [n_sh_rows], [int(kp.shape[0])], sm_scale
    )

    # Stage 2: Compact batched unpadded tail pass for ALL shared + unshared branches
    tail_ks, tail_vs, tail_lens = [], [], []
    for b in ordered_ids:
        ks, vs = k_suffix_list[b], v_suffix_list[b]
        if ks is not None and ks.shape[0] > 0:
            kb = torch.cat([ks, k_loc_list[b]], dim=0)
            vb = torch.cat([vs, v_loc_list[b]], dim=0)
        else:
            kb, vb = k_loc_list[b], v_loc_list[b]
        tail_ks.append(kb)
        tail_vs.append(vb)
        tail_lens.append(int(kb.shape[0]))
    k_tail_cat = torch.cat(tail_ks, dim=0).contiguous()
    v_tail_cat = torch.cat(tail_vs, dim=0).contiguous()
    o_tail_all, lse_tail_all = _run_varlen_fa2_with_lse(
        backend, q_all, k_tail_cat, v_tail_cat, [q_len] * len(ordered_ids), tail_lens, sm_scale
    )

    # Stage 3: In-place exact FP32 online-softmax LSE merge directly into `o_tail_all[:n_sh_rows]`
    o_tail_sh = o_tail_all[:n_sh_rows]
    o_merged = _merge_two_lse_states(
        o_pre_cat, lse_pre_cat, o_tail_sh, lse_tail_all[:n_sh_rows], out=o_tail_sh
    )
    outputs: list[Any] = [None] * num_branches
    for idx_b, b in enumerate(b_sh):
        outputs[b] = o_merged[idx_b * q_len : (idx_b + 1) * q_len]
    for idx_u, b in enumerate(unshared_ids):
        off = n_sh_rows + idx_u * q_len
        outputs[b] = o_tail_all[off : off + q_len]
    return outputs


def should_use_cascade_cfg_split(q_len: int, l_shared: int, num_shared_branches: int) -> bool:
    """Gate two-pass Cascade-CFG + fused LSE merge on empirically calibrated arithmetic-intensity payoff."""
    if q_len <= 0 or num_shared_branches < 2 or l_shared < 256:
        return False
    if q_len <= 2048:
        return l_shared >= max(256, int(1.5 * q_len))
    return l_shared >= 4 * q_len


# Backward-compatible alias used across Module 3 callers and unit tests
should_use_flash_prefix_split = should_use_cascade_cfg_split


def _get_cached_left_pad_meta(lens: tuple[int, ...], device: Any) -> Any:
    """Return cached DiffusionAttentionMetadata(attn_mask) without per-layer mask alloc/fill."""
    import torch

    max_len = max(lens)
    if all(ln == max_len for ln in lens):
        return None
    key = (str(device), lens)
    cached = _ATTN_META_CACHE.get(key)
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
    if meta is not None and len(_ATTN_META_CACHE) < 256:
        _ATTN_META_CACHE[key] = meta
    return meta


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
    ws = _DIRECT_WS_CACHE.get(wkey)
    q_shape = (num_branches, q_len, h_q, head_dim)
    kv_shape = (num_branches, max_len, h_kv, head_dim)
    if (
        ws is None
        or ws["q"].shape != q_shape
        or ws["k"].shape[0] != num_branches
        or ws["k"].shape[1] < max_len
        or ws["k"].shape[2:] != (h_kv, head_dim)
    ):
        ws = {
            "q": torch.empty(q_shape, dtype=q0.dtype, device=q0.device),
            "k": torch.zeros(kv_shape, dtype=k_loc_list[0].dtype, device=q0.device),
            "v": torch.zeros(kv_shape, dtype=v_loc_list[0].dtype, device=q0.device),
            "last_lens": lens,
            "last_max": max_len,
        }
        _DIRECT_WS_CACHE[wkey] = ws

    q_buf = ws["q"]
    k_buf = ws["k"][:, :max_len]
    v_buf = ws["v"][:, :max_len]
    need_zero_pad = (ws["last_lens"] != lens) or (ws["last_max"] != max_len)
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

    meta = _get_cached_left_pad_meta(lens, q0.device)
    o_4d = attn_noncausal(q_buf, k_buf, v_buf, meta)
    return [o_4d[b] for b in range(num_branches)]


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
    attn_noncausal: Any | None = None,
    prefer_unpadded_varlen: bool = False,
) -> list[Any] | None:
    """High-res / short-prefix lossless CFG attention without LSE-merge bottleneck."""
    import torch

    num_branches = len(q_list)
    q_len, h_q, head_dim = q_list[0].shape
    h_kv = k_loc_list[0].shape[1]
    shared_set = set(shared_branch_ids)
    l_shared = int(k_prefix.shape[0]) if k_prefix is not None else 0

    all_k_parts: list[Any] = []
    all_v_parts: list[Any] = []
    k_lens: list[int] = []
    for b in range(num_branches):
        b_len = 0
        if b in shared_set and l_shared > 0 and k_prefix is not None and v_prefix is not None:
            all_k_parts.append(k_prefix)
            all_v_parts.append(v_prefix)
            b_len += l_shared
        ks, vs = k_suffix_list[b], v_suffix_list[b]
        if ks is not None and vs is not None and ks.shape[0] > 0:
            all_k_parts.append(ks)
            all_v_parts.append(vs)
            b_len += int(ks.shape[0])
        all_k_parts.append(k_loc_list[b])
        all_v_parts.append(v_loc_list[b])
        b_len += int(k_loc_list[b].shape[0])
        k_lens.append(b_len)

    # When `prefer_unpadded_varlen=True` (i.e. Cascade-CFG fast mode) and branch lengths are highly
    # asymmetric (e.g. long multi-turn main/cfg_text vs short cfg_img), prefer unpadded `vllm_varlen`
    # to eliminate wasteful left-padding of the short branch. Keep `attn_noncausal` for explicit
    # bit-exact `lossless_varlen` fallback (`prefer_unpadded_varlen=False`).
    asymmetric_pad = bool(
        prefer_unpadded_varlen and len(k_lens) >= 2 and max(k_lens) >= 1024 and max(k_lens) > 2 * min(k_lens)
    )
    if attn_noncausal is not None and not asymmetric_pad:
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
        )

    backend = _probe_fa2_backend(q_list[0].device, q_list[0].dtype, h_q, h_kv, head_dim)
    kind, _, fn, _ = backend

    if kind == "vllm_varlen":
        q_cat = torch.cat(q_list, dim=0).contiguous()
        k_cat = torch.cat(all_k_parts, dim=0).contiguous()
        v_cat = torch.cat(all_v_parts, dim=0).contiguous()
        cu_q = _get_cu_seqlens([q_len] * num_branches, q_cat.device)
        cu_k = _get_cu_seqlens(k_lens, q_cat.device)
        res = fn(
            q_cat,
            k_cat,
            v_cat,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=q_len,
            max_seqlen_k=max(k_lens),
            softmax_scale=sm_scale,
            causal=False,
            return_softmax_lse=False,
        )
        out_cat = res[0] if isinstance(res, tuple) else res
        return list(out_cat.split(q_len, dim=0))

    if attn_noncausal is not None:
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
        )

    if kind == "flashinfer":
        outs = []
        for b in range(num_branches):
            ks, vs = k_suffix_list[b], v_suffix_list[b]
            if b in shared_set and l_shared > 0 and k_prefix is not None and v_prefix is not None:
                ks = torch.cat([k_prefix, ks], dim=0) if (ks is not None and ks.shape[0] > 0) else k_prefix
                vs = torch.cat([v_prefix, vs], dim=0) if (vs is not None and vs.shape[0] > 0) else v_prefix
            outs.append(
                _run_unshared_single_branch(
                    q_list[b], ks, vs, k_loc_list[b], v_loc_list[b], sm_scale, attn_noncausal=None
                )
            )
        return outs

    return None


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
) -> list[Any]:
    """Dispatch multi-branch CFG attention via arithmetic-intensity-gated Cascade-CFG or lossless varlen."""
    import torch

    num_branches = len(q_list)
    q_len, _, head_dim = q_list[0].shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    l_shared = int(k_prefix.shape[0]) if k_prefix is not None else 0

    if q_list[0].is_cuda:
        if backend in ("cascade_cfg", "flash_prefix") and should_use_cascade_cfg_split(
            q_len, l_shared, len(shared_branch_ids)
        ):
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
            )
            if fp_out is not None:
                if stats_out is not None:
                    stats_out["cascade_split_executed"] = stats_out.get("cascade_split_executed", 0) + 1
                return fp_out

        if backend in ("cascade_cfg", "flash_prefix", "lossless_varlen", "unpadded_varlen"):
            unpadded_out = _run_unpadded_varlen_cfg(
                q_list=q_list,
                k_loc_list=k_loc_list,
                v_loc_list=v_loc_list,
                k_prefix=k_prefix,
                v_prefix=v_prefix,
                shared_branch_ids=shared_branch_ids,
                k_suffix_list=k_suffix_list,
                v_suffix_list=v_suffix_list,
                sm_scale=sm_scale,
                attn_noncausal=attn_noncausal,
                prefer_unpadded_varlen=(backend in ("cascade_cfg", "flash_prefix", "unpadded_varlen")),
            )
            if unpadded_out is not None:
                return unpadded_out

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


_ACTIVE_CFG_PATCH: ContextVar["BagelSharedPrefixPatch | None"] = ContextVar("_ACTIVE_CFG_PATCH", default=None)
_ORIG_NAIVE_MERGE: Any = None


def _ensure_contextvar_naive_merge_hook() -> None:
    """Install a single idempotent ContextVar-dispatched merge hook on `NaiveCache` (no cross-instance clobber)."""
    global _ORIG_NAIVE_MERGE
    if _ORIG_NAIVE_MERGE is not None:
        return
    from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache

    _ORIG_NAIVE_MERGE = NaiveCache.merge

    def _dispatched_merge(caches):
        patch = _ACTIVE_CFG_PATCH.get()
        if patch is None or not patch.enabled or len(caches) < 2:
            return _ORIG_NAIVE_MERGE(caches)
        sp = SharedPrefixCache.from_branch_caches(
            caches, min_shared_tokens=patch.min_shared_tokens, verify_all_layers=True
        )
        patch.merge_calls += 1
        patch.last_shared_prefix_len = sp.shared_prefix_len
        patch.max_shared_prefix_len = max(patch.max_shared_prefix_len, sp.shared_prefix_len)
        patch.last_unique_kv_tokens = sp.unique_kv_tokens
        patch.last_naive_kv_tokens = sp.naive_kv_tokens
        patch.last_padded_naive_kv_tokens = sp.padded_naive_kv_tokens
        return sp

    NaiveCache.merge = staticmethod(_dispatched_merge)


class BagelSharedPrefixPatch:
    """ContextVar-scoped SharedPrefixCache (LECP CoW) + Cascade-CFG attention hook for `BagelPipeline`."""

    def __init__(
        self,
        pipeline: Any,
        min_shared_tokens: int = 32,
        allow_cascade_split: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.bagel = pipeline.bagel
        self.layers = pipeline.bagel.language_model.model.layers
        self.min_shared_tokens = int(min_shared_tokens)
        self.allow_cascade_split = bool(allow_cascade_split)
        self.enabled = False
        self._installed = False
        self._orig_forward_gens: list[tuple[Any, Any]] = []
        self.reset_telemetry()

    def reset_telemetry(self) -> None:
        self.merge_calls = 0
        self.shared_attn_calls = 0
        self.cascade_split_calls = 0
        self.unpadded_single_branch_calls = 0
        self.last_shared_prefix_len = 0
        self.max_shared_prefix_len = 0
        self.last_unique_kv_tokens = 0
        self.last_naive_kv_tokens = 0
        self.last_padded_naive_kv_tokens = 0

    def install(self) -> None:
        if self._installed:
            return
        import torch

        _ensure_contextvar_naive_merge_hook()
        patch = self

        for layer in self.layers:
            attn_mod = layer.self_attn
            orig_fwd_gen = attn_mod._forward_gen
            self._orig_forward_gens.append((attn_mod, orig_fwd_gen))

            def _make_fwd_gen(mod, orig_fn):
                def _wrapped_forward_gen(
                    packed_query_sequence,
                    query_lens,
                    packed_query_position_embeddings,
                    past_key_values,
                    packed_vae_token_indexes,
                    packed_text_indexes,
                    update_past_key_values=False,
                ):
                    active_patch = _ACTIVE_CFG_PATCH.get()
                    use_patch = (active_patch is patch or (active_patch is None and patch.enabled)) and patch.enabled
                    if (
                        not use_patch
                        or not isinstance(past_key_values, SharedPrefixCache)
                        or update_past_key_values
                        or mod._is_sp_active()
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
                    mode_backend = "cascade_cfg" if patch.allow_cascade_split else "lossless_varlen"
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
                    )
                    if past_key_values.shared_prefix_len > 0:
                        patch.shared_attn_calls += 1
                        patch.cascade_split_calls += int(call_stats.get("cascade_split_executed", 0))
                    patch.unpadded_single_branch_calls += len(past_key_values.unshared_branch_ids)

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

        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self.enabled = False
        for attn_mod, orig_fn in self._orig_forward_gens:
            attn_mod._forward_gen = orig_fn
        self._orig_forward_gens.clear()
        _DIRECT_WS_CACHE.clear()
        _CU_SEQLENS_CACHE.clear()
        _ATTN_META_CACHE.clear()
        self._installed = False

    @contextmanager
    def active(self, reset_stats: bool = True, allow_cascade_split: bool | None = None):
        if not self._installed:
            self.install()
        if reset_stats:
            self.reset_telemetry()
        prev_enabled = self.enabled
        prev_split = self.allow_cascade_split
        if allow_cascade_split is not None:
            self.allow_cascade_split = bool(allow_cascade_split)
        tok = _ACTIVE_CFG_PATCH.set(self)
        self.enabled = True
        try:
            yield self
        finally:
            self.enabled = prev_enabled
            self.allow_cascade_split = prev_split
            _ACTIVE_CFG_PATCH.reset(tok)


def install_shared_prefix_bagel_hooks(
    pipeline: Any,
    min_shared_tokens: int = 32,
    allow_cascade_split: bool = True,
) -> BagelSharedPrefixPatch:
    patch = BagelSharedPrefixPatch(
        pipeline,
        min_shared_tokens=min_shared_tokens,
        allow_cascade_split=allow_cascade_split,
    )
    patch.install()
    return patch
