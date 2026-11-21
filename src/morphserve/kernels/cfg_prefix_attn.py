"""Single-HBM-Read Triton Shared-Prefix Multi-Branch CFG Attention & BAGEL runtime hooks."""

from __future__ import annotations

from contextlib import contextmanager
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

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        ],
        key=["Q_LEN", "L_SHARED", "HEAD_DIM"],
    )
    @triton.jit
    def _shared_prefix_cfg2_attn_fwd(
        Q0, Q1, K_pre, V_pre,
        K_suf0, V_suf0, K_suf1, V_suf1,
        K_loc0, V_loc0, K_loc1, V_loc1,
        Out0, Out1,
        sq0_m, sq0_h, sq0_d,
        sq1_m, sq1_h, sq1_d,
        skp_n, skp_h, skp_d,
        svp_n, svp_h, svp_d,
        sks0_n, sks0_h, sks0_d,
        svs0_n, svs0_h, svs0_d,
        sks1_n, sks1_h, sks1_d,
        svs1_n, svs1_h, svs1_d,
        skl0_n, skl0_h, skl0_d,
        svl0_n, svl0_h, svl0_d,
        skl1_n, skl1_h, skl1_d,
        svl1_n, svl1_h, svl1_d,
        so0_m, so0_h, so0_d,
        so1_m, so1_h, so1_d,
        Q_LEN, L_SHARED, L_SUF0, L_SUF1,
        sm_scale_log2,
        GQA_GROUP: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Exact non-causal FlashAttention-2 for 2 CFG branches sharing `L_SHARED` KV tokens.

        Reads every `(K_pre, V_pre)` tile ONCE from HBM and updates BOTH branches in registers,
        then continues online softmax over each branch's private suffix + local `[text, vae]` KV.
        """
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        kv_h = pid_h // GQA_GROUP

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        offs_n = tl.arange(0, BLOCK_N)
        m_mask = offs_m < Q_LEN

        q0 = tl.load(Q0 + offs_m[:, None] * sq0_m + pid_h * sq0_h + offs_d[None, :] * sq0_d, mask=m_mask[:, None], other=0.0)
        q1 = tl.load(Q1 + offs_m[:, None] * sq1_m + pid_h * sq1_h + offs_d[None, :] * sq1_d, mask=m_mask[:, None], other=0.0)

        m0 = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l0 = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc0 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        m1 = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l1 = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc1 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Segment 1: Shared Prefix [0, L_SHARED) — SINGLE HBM LOAD FOR BOTH BRANCHES!
        for start_n in range(0, L_SHARED, BLOCK_N):
            cur_n = start_n + offs_n
            n_mask = cur_n < L_SHARED
            k = tl.load(K_pre + cur_n[:, None] * skp_n + kv_h * skp_h + offs_d[None, :] * skp_d, mask=n_mask[:, None], other=0.0)
            v = tl.load(V_pre + cur_n[:, None] * svp_n + kv_h * svp_h + offs_d[None, :] * svp_d, mask=n_mask[:, None], other=0.0)

            qk0 = tl.where(n_mask[None, :], tl.dot(q0, tl.trans(k)) * sm_scale_log2, float("-inf"))
            m0_new = tl.maximum(m0, tl.max(qk0, axis=1))
            alpha0 = tl.math.exp2(m0 - m0_new)
            p0 = tl.math.exp2(qk0 - m0_new[:, None])
            l0 = l0 * alpha0 + tl.sum(p0, axis=1)
            acc0 = tl.dot(p0.to(v.dtype), v, acc0 * alpha0[:, None])
            m0 = m0_new

            qk1 = tl.where(n_mask[None, :], tl.dot(q1, tl.trans(k)) * sm_scale_log2, float("-inf"))
            m1_new = tl.maximum(m1, tl.max(qk1, axis=1))
            alpha1 = tl.math.exp2(m1 - m1_new)
            p1 = tl.math.exp2(qk1 - m1_new[:, None])
            l1 = l1 * alpha1 + tl.sum(p1, axis=1)
            acc1 = tl.dot(p1.to(v.dtype), v, acc1 * alpha1[:, None])
            m1 = m1_new

        # Segment 2a: Branch 0 private suffix
        for start_n in range(0, L_SUF0, BLOCK_N):
            cur_n = start_n + offs_n
            n_mask = cur_n < L_SUF0
            k = tl.load(K_suf0 + cur_n[:, None] * sks0_n + kv_h * sks0_h + offs_d[None, :] * sks0_d, mask=n_mask[:, None], other=0.0)
            v = tl.load(V_suf0 + cur_n[:, None] * svs0_n + kv_h * svs0_h + offs_d[None, :] * svs0_d, mask=n_mask[:, None], other=0.0)
            qk0 = tl.where(n_mask[None, :], tl.dot(q0, tl.trans(k)) * sm_scale_log2, float("-inf"))
            m0_new = tl.maximum(m0, tl.max(qk0, axis=1))
            alpha0 = tl.math.exp2(m0 - m0_new)
            p0 = tl.math.exp2(qk0 - m0_new[:, None])
            l0 = l0 * alpha0 + tl.sum(p0, axis=1)
            acc0 = tl.dot(p0.to(v.dtype), v, acc0 * alpha0[:, None])
            m0 = m0_new

        # Segment 2b: Branch 1 private suffix
        for start_n in range(0, L_SUF1, BLOCK_N):
            cur_n = start_n + offs_n
            n_mask = cur_n < L_SUF1
            k = tl.load(K_suf1 + cur_n[:, None] * sks1_n + kv_h * sks1_h + offs_d[None, :] * sks1_d, mask=n_mask[:, None], other=0.0)
            v = tl.load(V_suf1 + cur_n[:, None] * svs1_n + kv_h * svs1_h + offs_d[None, :] * svs1_d, mask=n_mask[:, None], other=0.0)
            qk1 = tl.where(n_mask[None, :], tl.dot(q1, tl.trans(k)) * sm_scale_log2, float("-inf"))
            m1_new = tl.maximum(m1, tl.max(qk1, axis=1))
            alpha1 = tl.math.exp2(m1 - m1_new)
            p1 = tl.math.exp2(qk1 - m1_new[:, None])
            l1 = l1 * alpha1 + tl.sum(p1, axis=1)
            acc1 = tl.dot(p1.to(v.dtype), v, acc1 * alpha1[:, None])
            m1 = m1_new

        # Segment 3a: Branch 0 local [text, vae]
        for start_n in range(0, Q_LEN, BLOCK_N):
            cur_n = start_n + offs_n
            n_mask = cur_n < Q_LEN
            k = tl.load(K_loc0 + cur_n[:, None] * skl0_n + kv_h * skl0_h + offs_d[None, :] * skl0_d, mask=n_mask[:, None], other=0.0)
            v = tl.load(V_loc0 + cur_n[:, None] * svl0_n + kv_h * svl0_h + offs_d[None, :] * svl0_d, mask=n_mask[:, None], other=0.0)
            qk0 = tl.where(n_mask[None, :], tl.dot(q0, tl.trans(k)) * sm_scale_log2, float("-inf"))
            m0_new = tl.maximum(m0, tl.max(qk0, axis=1))
            alpha0 = tl.math.exp2(m0 - m0_new)
            p0 = tl.math.exp2(qk0 - m0_new[:, None])
            l0 = l0 * alpha0 + tl.sum(p0, axis=1)
            acc0 = tl.dot(p0.to(v.dtype), v, acc0 * alpha0[:, None])
            m0 = m0_new

        # Segment 3b: Branch 1 local [text, vae]
        for start_n in range(0, Q_LEN, BLOCK_N):
            cur_n = start_n + offs_n
            n_mask = cur_n < Q_LEN
            k = tl.load(K_loc1 + cur_n[:, None] * skl1_n + kv_h * skl1_h + offs_d[None, :] * skl1_d, mask=n_mask[:, None], other=0.0)
            v = tl.load(V_loc1 + cur_n[:, None] * svl1_n + kv_h * svl1_h + offs_d[None, :] * svl1_d, mask=n_mask[:, None], other=0.0)
            qk1 = tl.where(n_mask[None, :], tl.dot(q1, tl.trans(k)) * sm_scale_log2, float("-inf"))
            m1_new = tl.maximum(m1, tl.max(qk1, axis=1))
            alpha1 = tl.math.exp2(m1 - m1_new)
            p1 = tl.math.exp2(qk1 - m1_new[:, None])
            l1 = l1 * alpha1 + tl.sum(p1, axis=1)
            acc1 = tl.dot(p1.to(v.dtype), v, acc1 * alpha1[:, None])
            m1 = m1_new

        tl.store(Out0 + offs_m[:, None] * so0_m + pid_h * so0_h + offs_d[None, :] * so0_d, (acc0 / l0[:, None]).to(Out0.dtype.element_ty), mask=m_mask[:, None])
        tl.store(Out1 + offs_m[:, None] * so1_m + pid_h * so1_h + offs_d[None, :] * so1_d, (acc1 / l1[:, None]).to(Out1.dtype.element_ty), mask=m_mask[:, None])

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        ],
        key=["Q_LEN", "L_SHARED", "HEAD_DIM"],
    )
    @triton.jit
    def _shared_prefix_cfg3_attn_fwd(
        Q0, Q1, Q2, K_pre, V_pre,
        K_suf0, V_suf0, K_suf1, V_suf1, K_suf2, V_suf2,
        K_loc0, V_loc0, K_loc1, V_loc1, K_loc2, V_loc2,
        Out0, Out1, Out2,
        sq0_m, sq0_h, sq0_d, sq1_m, sq1_h, sq1_d, sq2_m, sq2_h, sq2_d,
        skp_n, skp_h, skp_d, svp_n, svp_h, svp_d,
        sks0_n, sks0_h, sks0_d, svs0_n, svs0_h, svs0_d,
        sks1_n, sks1_h, sks1_d, svs1_n, svs1_h, svs1_d,
        sks2_n, sks2_h, sks2_d, svs2_n, svs2_h, svs2_d,
        skl0_n, skl0_h, skl0_d, svl0_n, svl0_h, svl0_d,
        skl1_n, skl1_h, skl1_d, svl1_n, svl1_h, svl1_d,
        skl2_n, skl2_h, skl2_d, svl2_n, svl2_h, svl2_d,
        so0_m, so0_h, so0_d, so1_m, so1_h, so1_d, so2_m, so2_h, so2_d,
        Q_LEN, L_SHARED, L_SUF0, L_SUF1, L_SUF2,
        sm_scale_log2,
        GQA_GROUP: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Exact non-causal FlashAttention-2 for 3 CFG branches sharing `L_SHARED` KV tokens."""
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        kv_h = pid_h // GQA_GROUP

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        offs_n = tl.arange(0, BLOCK_N)
        m_mask = offs_m < Q_LEN

        q0 = tl.load(Q0 + offs_m[:, None] * sq0_m + pid_h * sq0_h + offs_d[None, :] * sq0_d, mask=m_mask[:, None], other=0.0)
        q1 = tl.load(Q1 + offs_m[:, None] * sq1_m + pid_h * sq1_h + offs_d[None, :] * sq1_d, mask=m_mask[:, None], other=0.0)
        q2 = tl.load(Q2 + offs_m[:, None] * sq2_m + pid_h * sq2_h + offs_d[None, :] * sq2_d, mask=m_mask[:, None], other=0.0)

        m0 = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l0 = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc0 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        m1 = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l1 = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc1 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        m2 = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l2 = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc2 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        for start_n in range(0, L_SHARED, BLOCK_N):
            cur_n = start_n + offs_n
            n_mask = cur_n < L_SHARED
            k = tl.load(K_pre + cur_n[:, None] * skp_n + kv_h * skp_h + offs_d[None, :] * skp_d, mask=n_mask[:, None], other=0.0)
            v = tl.load(V_pre + cur_n[:, None] * svp_n + kv_h * svp_h + offs_d[None, :] * svp_d, mask=n_mask[:, None], other=0.0)

            qk0 = tl.where(n_mask[None, :], tl.dot(q0, tl.trans(k)) * sm_scale_log2, float("-inf"))
            m0_n = tl.maximum(m0, tl.max(qk0, axis=1))
            a0 = tl.math.exp2(m0 - m0_n)
            p0 = tl.math.exp2(qk0 - m0_n[:, None])
            l0 = l0 * a0 + tl.sum(p0, axis=1)
            acc0 = tl.dot(p0.to(v.dtype), v, acc0 * a0[:, None])
            m0 = m0_n

            qk1 = tl.where(n_mask[None, :], tl.dot(q1, tl.trans(k)) * sm_scale_log2, float("-inf"))
            m1_n = tl.maximum(m1, tl.max(qk1, axis=1))
            a1 = tl.math.exp2(m1 - m1_n)
            p1 = tl.math.exp2(qk1 - m1_n[:, None])
            l1 = l1 * a1 + tl.sum(p1, axis=1)
            acc1 = tl.dot(p1.to(v.dtype), v, acc1 * a1[:, None])
            m1 = m1_n

            qk2 = tl.where(n_mask[None, :], tl.dot(q2, tl.trans(k)) * sm_scale_log2, float("-inf"))
            m2_n = tl.maximum(m2, tl.max(qk2, axis=1))
            a2 = tl.math.exp2(m2 - m2_n)
            p2 = tl.math.exp2(qk2 - m2_n[:, None])
            l2 = l2 * a2 + tl.sum(p2, axis=1)
            acc2 = tl.dot(p2.to(v.dtype), v, acc2 * a2[:, None])
            m2 = m2_n

        for start_n in range(0, L_SUF0, BLOCK_N):
            cur_n = start_n + offs_n
            n_mask = cur_n < L_SUF0
            k = tl.load(K_suf0 + cur_n[:, None] * sks0_n + kv_h * sks0_h + offs_d[None, :] * sks0_d, mask=n_mask[:, None], other=0.0)
            v = tl.load(V_suf0 + cur_n[:, None] * svs0_n + kv_h * svs0_h + offs_d[None, :] * svs0_d, mask=n_mask[:, None], other=0.0)
            qk0 = tl.where(n_mask[None, :], tl.dot(q0, tl.trans(k)) * sm_scale_log2, float("-inf"))
            m0_n = tl.maximum(m0, tl.max(qk0, axis=1))
            a0 = tl.math.exp2(m0 - m0_n)
            p0 = tl.math.exp2(qk0 - m0_n[:, None])
            l0 = l0 * a0 + tl.sum(p0, axis=1)
            acc0 = tl.dot(p0.to(v.dtype), v, acc0 * a0[:, None])
            m0 = m0_n
        for start_n in range(0, Q_LEN, BLOCK_N):
            cur_n = start_n + offs_n
            n_mask = cur_n < Q_LEN
            k = tl.load(K_loc0 + cur_n[:, None] * skl0_n + kv_h * skl0_h + offs_d[None, :] * skl0_d, mask=n_mask[:, None], other=0.0)
            v = tl.load(V_loc0 + cur_n[:, None] * svl0_n + kv_h * svl0_h + offs_d[None, :] * svl0_d, mask=n_mask[:, None], other=0.0)
            qk0 = tl.where(n_mask[None, :], tl.dot(q0, tl.trans(k)) * sm_scale_log2, float("-inf"))
            m0_n = tl.maximum(m0, tl.max(qk0, axis=1))
            a0 = tl.math.exp2(m0 - m0_n)
            p0 = tl.math.exp2(qk0 - m0_n[:, None])
            l0 = l0 * a0 + tl.sum(p0, axis=1)
            acc0 = tl.dot(p0.to(v.dtype), v, acc0 * a0[:, None])
            m0 = m0_n

        for start_n in range(0, L_SUF1, BLOCK_N):
            cur_n = start_n + offs_n
            n_mask = cur_n < L_SUF1
            k = tl.load(K_suf1 + cur_n[:, None] * sks1_n + kv_h * sks1_h + offs_d[None, :] * sks1_d, mask=n_mask[:, None], other=0.0)
            v = tl.load(V_suf1 + cur_n[:, None] * svs1_n + kv_h * svs1_h + offs_d[None, :] * svs1_d, mask=n_mask[:, None], other=0.0)
            qk1 = tl.where(n_mask[None, :], tl.dot(q1, tl.trans(k)) * sm_scale_log2, float("-inf"))
            m1_n = tl.maximum(m1, tl.max(qk1, axis=1))
            a1 = tl.math.exp2(m1 - m1_n)
            p1 = tl.math.exp2(qk1 - m1_n[:, None])
            l1 = l1 * a1 + tl.sum(p1, axis=1)
            acc1 = tl.dot(p1.to(v.dtype), v, acc1 * a1[:, None])
            m1 = m1_n
        for start_n in range(0, Q_LEN, BLOCK_N):
            cur_n = start_n + offs_n
            n_mask = cur_n < Q_LEN
            k = tl.load(K_loc1 + cur_n[:, None] * skl1_n + kv_h * skl1_h + offs_d[None, :] * skl1_d, mask=n_mask[:, None], other=0.0)
            v = tl.load(V_loc1 + cur_n[:, None] * svl1_n + kv_h * svl1_h + offs_d[None, :] * svl1_d, mask=n_mask[:, None], other=0.0)
            qk1 = tl.where(n_mask[None, :], tl.dot(q1, tl.trans(k)) * sm_scale_log2, float("-inf"))
            m1_n = tl.maximum(m1, tl.max(qk1, axis=1))
            a1 = tl.math.exp2(m1 - m1_n)
            p1 = tl.math.exp2(qk1 - m1_n[:, None])
            l1 = l1 * a1 + tl.sum(p1, axis=1)
            acc1 = tl.dot(p1.to(v.dtype), v, acc1 * a1[:, None])
            m1 = m1_n

        for start_n in range(0, L_SUF2, BLOCK_N):
            cur_n = start_n + offs_n
            n_mask = cur_n < L_SUF2
            k = tl.load(K_suf2 + cur_n[:, None] * sks2_n + kv_h * sks2_h + offs_d[None, :] * sks2_d, mask=n_mask[:, None], other=0.0)
            v = tl.load(V_suf2 + cur_n[:, None] * svs2_n + kv_h * svs2_h + offs_d[None, :] * svs2_d, mask=n_mask[:, None], other=0.0)
            qk2 = tl.where(n_mask[None, :], tl.dot(q2, tl.trans(k)) * sm_scale_log2, float("-inf"))
            m2_n = tl.maximum(m2, tl.max(qk2, axis=1))
            a2 = tl.math.exp2(m2 - m2_n)
            p2 = tl.math.exp2(qk2 - m2_n[:, None])
            l2 = l2 * a2 + tl.sum(p2, axis=1)
            acc2 = tl.dot(p2.to(v.dtype), v, acc2 * a2[:, None])
            m2 = m2_n
        for start_n in range(0, Q_LEN, BLOCK_N):
            cur_n = start_n + offs_n
            n_mask = cur_n < Q_LEN
            k = tl.load(K_loc2 + cur_n[:, None] * skl2_n + kv_h * skl2_h + offs_d[None, :] * skl2_d, mask=n_mask[:, None], other=0.0)
            v = tl.load(V_loc2 + cur_n[:, None] * svl2_n + kv_h * svl2_h + offs_d[None, :] * svl2_d, mask=n_mask[:, None], other=0.0)
            qk2 = tl.where(n_mask[None, :], tl.dot(q2, tl.trans(k)) * sm_scale_log2, float("-inf"))
            m2_n = tl.maximum(m2, tl.max(qk2, axis=1))
            a2 = tl.math.exp2(m2 - m2_n)
            p2 = tl.math.exp2(qk2 - m2_n[:, None])
            l2 = l2 * a2 + tl.sum(p2, axis=1)
            acc2 = tl.dot(p2.to(v.dtype), v, acc2 * a2[:, None])
            m2 = m2_n

        tl.store(Out0 + offs_m[:, None] * so0_m + pid_h * so0_h + offs_d[None, :] * so0_d, (acc0 / l0[:, None]).to(Out0.dtype.element_ty), mask=m_mask[:, None])
        tl.store(Out1 + offs_m[:, None] * so1_m + pid_h * so1_h + offs_d[None, :] * so1_d, (acc1 / l1[:, None]).to(Out1.dtype.element_ty), mask=m_mask[:, None])
        tl.store(Out2 + offs_m[:, None] * so2_m + pid_h * so2_h + offs_d[None, :] * so2_d, (acc2 / l2[:, None]).to(Out2.dtype.element_ty), mask=m_mask[:, None])

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
        """Fused exact online-softmax LSE merge for FlashPrefix-CFG (zero autotune stall)."""
        pid_r = tl.program_id(0)
        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_d = tl.arange(0, HEAD_DIM)
        mask_r = offs_r < N_ROWS

        la = tl.load(Lse_a + offs_r, mask=mask_r, other=-1.0e20)
        lb = tl.load(Lse_b + offs_r, mask=mask_r, other=-1.0e20)
        m = tl.maximum(la, lb)
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
_HIGHRES_STRATEGY_CACHE: dict[tuple[str, int, int, int], str] = {}


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
        out_2d = ob_2d if (out is o_b and ob_2d.is_contiguous()) else torch.empty_like(oa_2d)
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
        return out_2d.reshape(m_tot, h_q, d)

    m = torch.maximum(lse_a, lse_b)
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


def _should_use_flash_prefix_split(q_len: int, l_shared: int, num_shared_branches: int) -> bool:
    """Gate two-pass FlashPrefix-CFG + LSE merge on compute-to-LSE-merge payoff."""
    if num_shared_branches < 2 or l_shared < 256:
        return False
    if q_len <= 2048:
        return l_shared >= max(256, int(1.5 * q_len))
    return l_shared >= 4 * q_len


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


def _fast_left_pad_kv_pair(k_tensors: list[Any], v_tensors: list[Any]) -> tuple[Any, Any, Any]:
    """Pack K/V branch tensors without redundant full-buffer cudaMemsetAsync or mask allocations."""
    import torch

    lens = tuple(int(t.shape[0]) for t in k_tensors)
    max_len = max(lens)
    if all(ln == max_len for ln in lens):
        return torch.stack(k_tensors, dim=0), torch.stack(v_tensors, dim=0), None
    bsz = len(k_tensors)
    shape = (bsz, max_len, *k_tensors[0].shape[1:])
    k_out = torch.empty(shape, dtype=k_tensors[0].dtype, device=k_tensors[0].device)
    v_out = torch.empty(shape, dtype=v_tensors[0].dtype, device=v_tensors[0].device)
    for i, (kt, vt, ln) in enumerate(zip(k_tensors, v_tensors, lens)):
        pad = max_len - ln
        if pad > 0:
            k_out[i, :pad].zero_()
            v_out[i, :pad].zero_()
        k_out[i, pad:] = kt
        v_out[i, pad:] = vt
    meta = _get_cached_left_pad_meta(lens, k_tensors[0].device)
    return k_out, v_out, meta


def _assemble_branch_kv(
    num_branches: int,
    shared_set: set[int],
    l_shared: int,
    k_prefix: Any | None,
    v_prefix: Any | None,
    k_suffix_list: list[Any | None],
    v_suffix_list: list[Any | None],
    k_loc_list: list[Any],
    v_loc_list: list[Any],
) -> tuple[list[Any], list[Any]]:
    import torch

    kb_list: list[Any] = []
    vb_list: list[Any] = []
    for b in range(num_branches):
        k_parts: list[Any] = []
        v_parts: list[Any] = []
        if b in shared_set and l_shared > 0 and k_prefix is not None and v_prefix is not None:
            k_parts.append(k_prefix)
            v_parts.append(v_prefix)
        ks, vs = k_suffix_list[b], v_suffix_list[b]
        if ks is not None and vs is not None and ks.shape[0] > 0:
            k_parts.append(ks)
            v_parts.append(vs)
        k_parts.append(k_loc_list[b])
        v_parts.append(v_loc_list[b])
        kb_list.append(k_parts[0] if len(k_parts) == 1 else torch.cat(k_parts, dim=0))
        vb_list.append(v_parts[0] if len(v_parts) == 1 else torch.cat(v_parts, dim=0))
    return kb_list, vb_list


def _run_clustered_noncausal_cfg(
    q_list: list[Any],
    kb_list: list[Any],
    vb_list: list[Any],
    shared_branch_ids: list[int],
    attn_noncausal: Any,
) -> list[Any]:
    """Batch equal/near-equal shared branches (B=2) while running short unshared `cfg_img` unpadded (B=1)."""
    import torch

    num_branches = len(q_list)
    b_sh = list(shared_branch_ids)
    shared_set = set(b_sh)
    unshared_ids = [b for b in range(num_branches) if b not in shared_set]
    outs: list[Any] = [None] * num_branches

    if len(b_sh) >= 2:
        q_sh = torch.stack([q_list[b] for b in b_sh], dim=0)
        k_sh, v_sh, meta_sh = _fast_left_pad_kv_pair([kb_list[b] for b in b_sh], [vb_list[b] for b in b_sh])
        o_sh = attn_noncausal(q_sh, k_sh, v_sh, meta_sh)
        for idx_b, b in enumerate(b_sh):
            outs[b] = o_sh[idx_b]
    elif len(b_sh) == 1:
        b = b_sh[0]
        outs[b] = attn_noncausal(q_list[b].unsqueeze(0), kb_list[b].unsqueeze(0), vb_list[b].unsqueeze(0), None).squeeze(0)

    for b in unshared_ids:
        outs[b] = attn_noncausal(q_list[b].unsqueeze(0), kb_list[b].unsqueeze(0), vb_list[b].unsqueeze(0), None).squeeze(0)
    return outs


def _run_fast_batched_noncausal_cfg(
    q_list: list[Any],
    kb_list: list[Any],
    vb_list: list[Any],
    attn_noncausal: Any,
) -> list[Any]:
    """Single-launch batched `attn_noncausal` using cached mask metadata and zero-memset empty buffers."""
    import torch

    q_4d = torch.stack(q_list, dim=0)
    k_4d, v_4d, meta = _fast_left_pad_kv_pair(kb_list, vb_list)
    o_4d = attn_noncausal(q_4d, k_4d, v_4d, meta)
    return [o_4d[b] for b in range(len(q_list))]


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
) -> list[Any] | None:
    """High-res / short-prefix lossless CFG attention without LSE-merge bottleneck."""
    import torch

    num_branches = len(q_list)
    q_len, h_q, head_dim = q_list[0].shape
    h_kv = k_loc_list[0].shape[1]
    shared_set = set(shared_branch_ids)
    l_shared = int(k_prefix.shape[0]) if k_prefix is not None else 0

    if attn_noncausal is not None:
        skey = (str(q_list[0].device), q_len, l_shared // 1024, len(shared_branch_ids))
        strat = _HIGHRES_STRATEGY_CACHE.get(skey)
        if strat is None and q_list[0].is_cuda:
            # One-time <15ms micro-calibration per shape bucket during warmup
            cands: dict[str, Any] = {
                "clustered": lambda: _run_clustered_noncausal_cfg(
                    q_list,
                    *_assemble_branch_kv(
                        num_branches, shared_set, l_shared, k_prefix, v_prefix, k_suffix_list, v_suffix_list, k_loc_list, v_loc_list
                    ),
                    shared_branch_ids,
                    attn_noncausal,
                ),
                "fast_batched": lambda: _run_fast_batched_noncausal_cfg(
                    q_list,
                    *_assemble_branch_kv(
                        num_branches, shared_set, l_shared, k_prefix, v_prefix, k_suffix_list, v_suffix_list, k_loc_list, v_loc_list
                    ),
                    attn_noncausal,
                ),
            }
            if l_shared >= 256 and len(shared_branch_ids) >= 2:
                cands["flash_prefix"] = lambda: _run_flash_prefix_cfg(
                    q_list, k_loc_list, v_loc_list, k_prefix, v_prefix, shared_branch_ids, k_suffix_list, v_suffix_list, sm_scale
                )
            best_name = "clustered"
            best_ms = float("inf")
            for name, fn_cand in cands.items():
                out_c = fn_cand()
                if out_c is None:
                    continue
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                for _ in range(3):
                    _ = fn_cand()
                ev1.record()
                torch.cuda.synchronize(q_list[0].device)
                ms = ev0.elapsed_time(ev1) / 3.0
                if ms < best_ms:
                    best_ms = ms
                    best_name = name
            strat = best_name
            if len(_HIGHRES_STRATEGY_CACHE) < 128:
                _HIGHRES_STRATEGY_CACHE[skey] = strat
        elif strat is None:
            strat = "clustered"

        if strat == "flash_prefix":
            fp_out = _run_flash_prefix_cfg(
                q_list, k_loc_list, v_loc_list, k_prefix, v_prefix, shared_branch_ids, k_suffix_list, v_suffix_list, sm_scale
            )
            if fp_out is not None:
                return fp_out
        kb_list, vb_list = _assemble_branch_kv(
            num_branches, shared_set, l_shared, k_prefix, v_prefix, k_suffix_list, v_suffix_list, k_loc_list, v_loc_list
        )
        if strat == "fast_batched":
            return _run_fast_batched_noncausal_cfg(q_list, kb_list, vb_list, attn_noncausal)
        return _run_clustered_noncausal_cfg(q_list, kb_list, vb_list, shared_branch_ids, attn_noncausal)

    backend = _probe_fa2_backend(q_list[0].device, q_list[0].dtype, h_q, h_kv, head_dim)
    kind, _, fn, _ = backend

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
    backend: str = "flash_prefix",
) -> list[Any]:
    """Dispatch multi-branch CFG attention via FlashPrefix-CFG (default) or single-pass Triton kernel."""
    import torch

    num_branches = len(q_list)
    q_len, h_q, head_dim = q_list[0].shape
    h_kv = k_loc_list[0].shape[1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    sm_scale_log2 = sm_scale * 1.4426950408889634
    gqa_group = h_q // h_kv
    l_shared = int(k_prefix.shape[0]) if k_prefix is not None else 0

    # Preferred default: FlashPrefix-CFG when shared prefix compute outweighs LSE merge, else unpadded FA2 varlen.
    if backend == "flash_prefix" and q_list[0].is_cuda:
        if _should_use_flash_prefix_split(q_len, l_shared, len(shared_branch_ids)):
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
                return fp_out
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
        )
        if unpadded_out is not None:
            return unpadded_out

    outputs: list[Any] = [None] * num_branches
    handled: set[int] = set()

    use_triton = (
        HAS_TRITON
        and q_list[0].is_cuda
        and head_dim in (64, 128)
        and l_shared > 0
        and len(shared_branch_ids) in (2, 3)
    )

    def _prep_suf(ks, vs, fb_k, fb_v):
        ln = int(ks.shape[0]) if ks is not None else 0
        return (ks.contiguous(), vs.contiguous(), ln) if ln > 0 else (fb_k, fb_v, 0)

    if use_triton and len(shared_branch_ids) == 2:
        b0, b1 = shared_branch_ids[0], shared_branch_ids[1]
        q0, q1 = q_list[b0].contiguous(), q_list[b1].contiguous()
        kl0, vl0 = k_loc_list[b0].contiguous(), v_loc_list[b0].contiguous()
        kl1, vl1 = k_loc_list[b1].contiguous(), v_loc_list[b1].contiguous()
        kp, vp = k_prefix.contiguous(), v_prefix.contiguous()
        ks0, vs0, l_suf0 = _prep_suf(k_suffix_list[b0], v_suffix_list[b0], kl0, vl0)
        ks1, vs1, l_suf1 = _prep_suf(k_suffix_list[b1], v_suffix_list[b1], kl1, vl1)
        out0, out1 = torch.empty_like(q0), torch.empty_like(q1)
        grid = lambda meta: (triton.cdiv(q_len, meta["BLOCK_M"]), h_q)
        _shared_prefix_cfg2_attn_fwd[grid](
            q0, q1, kp, vp, ks0, vs0, ks1, vs1, kl0, vl0, kl1, vl1, out0, out1,
            q0.stride(0), q0.stride(1), q0.stride(2), q1.stride(0), q1.stride(1), q1.stride(2),
            kp.stride(0), kp.stride(1), kp.stride(2), vp.stride(0), vp.stride(1), vp.stride(2),
            ks0.stride(0), ks0.stride(1), ks0.stride(2), vs0.stride(0), vs0.stride(1), vs0.stride(2),
            ks1.stride(0), ks1.stride(1), ks1.stride(2), vs1.stride(0), vs1.stride(1), vs1.stride(2),
            kl0.stride(0), kl0.stride(1), kl0.stride(2), vl0.stride(0), vl0.stride(1), vl0.stride(2),
            kl1.stride(0), kl1.stride(1), kl1.stride(2), vl1.stride(0), vl1.stride(1), vl1.stride(2),
            out0.stride(0), out0.stride(1), out0.stride(2), out1.stride(0), out1.stride(1), out1.stride(2),
            q_len, l_shared, l_suf0, l_suf1, sm_scale_log2, GQA_GROUP=gqa_group, HEAD_DIM=head_dim,
        )
        outputs[b0], outputs[b1] = out0, out1
        handled.update((b0, b1))

    elif use_triton and len(shared_branch_ids) == 3:
        b0, b1, b2 = shared_branch_ids[0], shared_branch_ids[1], shared_branch_ids[2]
        q0, q1, q2 = q_list[b0].contiguous(), q_list[b1].contiguous(), q_list[b2].contiguous()
        kl0, vl0 = k_loc_list[b0].contiguous(), v_loc_list[b0].contiguous()
        kl1, vl1 = k_loc_list[b1].contiguous(), v_loc_list[b1].contiguous()
        kl2, vl2 = k_loc_list[b2].contiguous(), v_loc_list[b2].contiguous()
        kp, vp = k_prefix.contiguous(), v_prefix.contiguous()
        ks0, vs0, l_suf0 = _prep_suf(k_suffix_list[b0], v_suffix_list[b0], kl0, vl0)
        ks1, vs1, l_suf1 = _prep_suf(k_suffix_list[b1], v_suffix_list[b1], kl1, vl1)
        ks2, vs2, l_suf2 = _prep_suf(k_suffix_list[b2], v_suffix_list[b2], kl2, vl2)
        out0, out1, out2 = torch.empty_like(q0), torch.empty_like(q1), torch.empty_like(q2)
        grid = lambda meta: (triton.cdiv(q_len, meta["BLOCK_M"]), h_q)
        _shared_prefix_cfg3_attn_fwd[grid](
            q0, q1, q2, kp, vp, ks0, vs0, ks1, vs1, ks2, vs2,
            kl0, vl0, kl1, vl1, kl2, vl2, out0, out1, out2,
            q0.stride(0), q0.stride(1), q0.stride(2), q1.stride(0), q1.stride(1), q1.stride(2),
            q2.stride(0), q2.stride(1), q2.stride(2), kp.stride(0), kp.stride(1), kp.stride(2),
            vp.stride(0), vp.stride(1), vp.stride(2), ks0.stride(0), ks0.stride(1), ks0.stride(2),
            vs0.stride(0), vs0.stride(1), vs0.stride(2), ks1.stride(0), ks1.stride(1), ks1.stride(2),
            vs1.stride(0), vs1.stride(1), vs1.stride(2), ks2.stride(0), ks2.stride(1), ks2.stride(2),
            vs2.stride(0), vs2.stride(1), vs2.stride(2), kl0.stride(0), kl0.stride(1), kl0.stride(2),
            vl0.stride(0), vl0.stride(1), vl0.stride(2), kl1.stride(0), kl1.stride(1), kl1.stride(2),
            vl1.stride(0), vl1.stride(1), vl1.stride(2), kl2.stride(0), kl2.stride(1), kl2.stride(2),
            vl2.stride(0), vl2.stride(1), vl2.stride(2), out0.stride(0), out0.stride(1), out0.stride(2),
            out1.stride(0), out1.stride(1), out1.stride(2), out2.stride(0), out2.stride(1), out2.stride(2),
            q_len, l_shared, l_suf0, l_suf1, l_suf2, sm_scale_log2, GQA_GROUP=gqa_group, HEAD_DIM=head_dim,
        )
        outputs[b0], outputs[b1], outputs[b2] = out0, out1, out2
        handled.update((b0, b1, b2))

    for b in range(num_branches):
        if b in handled:
            continue
        k_suf, v_suf = k_suffix_list[b], v_suffix_list[b]
        if b in shared_branch_ids and l_shared > 0:
            k_suf = torch.cat([k_prefix, k_suf], dim=0) if (k_suf is not None and k_suf.shape[0] > 0) else k_prefix
            v_suf = torch.cat([v_prefix, v_suf], dim=0) if (v_suf is not None and v_suf.shape[0] > 0) else v_prefix
        outputs[b] = _run_unshared_single_branch(
            q_list[b], k_suf, v_suf, k_loc_list[b], v_loc_list[b], sm_scale, attn_noncausal=attn_noncausal
        )
    return outputs


class BagelSharedPrefixPatch:
    """Toggleable monkey-patch wiring `SharedPrefixCache` + Triton CFG Attention into a live `BagelPipeline`."""

    def __init__(self, pipeline: Any, min_shared_tokens: int = 32) -> None:
        self.pipeline = pipeline
        self.bagel = pipeline.bagel
        self.layers = pipeline.bagel.language_model.model.layers
        self.min_shared_tokens = int(min_shared_tokens)
        self.enabled = False
        self._installed = False
        self._orig_naive_merge = None
        self._orig_forward_gens: list[tuple[Any, Any]] = []
        self.reset_telemetry()

    def reset_telemetry(self) -> None:
        self.merge_calls = 0
        self.shared_attn_calls = 0
        self.unpadded_single_branch_calls = 0
        self.last_shared_prefix_len = 0
        self.last_unique_kv_tokens = 0
        self.last_naive_kv_tokens = 0
        self.last_padded_naive_kv_tokens = 0

    def install(self) -> None:
        if self._installed:
            return
        import torch
        from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache

        patch = self
        self._orig_naive_merge = NaiveCache.merge

        def _patched_merge(caches):
            if not patch.enabled or len(caches) < 2:
                return patch._orig_naive_merge(caches)
            sp = SharedPrefixCache.from_branch_caches(
                caches, min_shared_tokens=patch.min_shared_tokens, verify_all_layers=True
            )
            patch.merge_calls += 1
            patch.last_shared_prefix_len = sp.shared_prefix_len
            patch.last_unique_kv_tokens = sp.unique_kv_tokens
            patch.last_naive_kv_tokens = sp.naive_kv_tokens
            patch.last_padded_naive_kv_tokens = sp.padded_naive_kv_tokens
            return sp

        NaiveCache.merge = staticmethod(_patched_merge)

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
                    if (
                        not patch.enabled
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
                    )
                    if past_key_values.shared_prefix_len > 0:
                        patch.shared_attn_calls += 1
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
        from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache

        self.enabled = False
        if self._orig_naive_merge is not None:
            NaiveCache.merge = self._orig_naive_merge
            self._orig_naive_merge = None
        for attn_mod, orig_fn in self._orig_forward_gens:
            attn_mod._forward_gen = orig_fn
        self._orig_forward_gens.clear()
        self._installed = False

    @contextmanager
    def active(self, reset_stats: bool = True):
        if not self._installed:
            self.install()
        if reset_stats:
            self.reset_telemetry()
        prev = self.enabled
        self.enabled = True
        try:
            yield self
        finally:
            self.enabled = prev


def install_shared_prefix_bagel_hooks(pipeline: Any, min_shared_tokens: int = 32) -> BagelSharedPrefixPatch:
    patch = BagelSharedPrefixPatch(pipeline, min_shared_tokens=min_shared_tokens)
    patch.install()
    return patch
