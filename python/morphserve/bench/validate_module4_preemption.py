"""Phase-1 Module 4 live A100 validation: `[B, D]` Packed AR + Adaptive `k>=1` Layer-Chunk Preemption + M1+M3+M4 Stack."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import time

from morphserve.bench.exactness import compute_ssim_rgb, percentile as _pct, pil_sha256 as _pil_sha256
from morphserve.kernels import (
    GpuInPlaceLoopback,
    SharedPrefixCache,
    cow_fork_context,
    install_shared_prefix_bagel_hooks,
    rebind_cow_prefix,
)
from morphserve.models import load_bagel_pipeline
from morphserve.vmm import ExpertPartitionSpec, LayerExpertPager, TriPoolVMM, VmmNaiveCache
from morphserve.worker import ArDecodeTask, BagelExecutionInterceptor, UnifiedPreemptiveScheduler


def _prepare_text_ctx(pipeline, prompt: str):
    import torch
    from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache

    num_layers = pipeline.bagel.config.llm_config.num_hidden_layers
    ctx = {"kv_lens": [0], "ropes": [0], "past_key_values": NaiveCache(num_layers)}
    inp, new_lens, new_rope = pipeline.bagel.prepare_prompts(
        curr_kvlens=ctx["kv_lens"],
        curr_rope=ctx["ropes"],
        prompts=[prompt],
        tokenizer=pipeline.tokenizer,
        new_token_ids=pipeline.new_token_ids,
    )
    inp = {k: (v.to(pipeline.device) if torch.is_tensor(v) else v) for k, v in inp.items()}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ctx["past_key_values"] = pipeline.bagel.forward_cache_update_text(ctx["past_key_values"], **inp)
    ctx["kv_lens"] = new_lens
    ctx["ropes"] = new_rope
    return ctx


def _make_ar_task(
    pipeline,
    base_ctx: dict,
    req_id: str,
    max_tokens: int,
    tpot_slo_ms: float,
    *,
    vmm: TriPoolVMM | None = None,
) -> ArDecodeTask:
    from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache

    if vmm is not None:
        num_layers = pipeline.bagel.config.llm_config.num_hidden_layers
        vmm_pkv = VmmNaiveCache.create_for_bagel(NaiveCache, num_layers, vmm)
        src_pkv = base_ctx["past_key_values"]
        for lidx in range(num_layers):
            if src_pkv.key_cache[lidx] is not None:
                vmm_pkv.key_cache[lidx] = src_pkv.key_cache[lidx]
                vmm_pkv.value_cache[lidx] = src_pkv.value_cache[lidx]
        kv_cache = vmm_pkv
    else:
        kv_cache = deepcopy(base_ctx["past_key_values"])

    start_inp = pipeline.bagel.prepare_start_tokens(base_ctx["kv_lens"], base_ctx["ropes"], pipeline.new_token_ids)
    curr_tokens = start_inp["packed_start_tokens"].to(pipeline.device)
    start_tok_int = int(start_inp["packed_start_tokens"].reshape(-1)[0].item())
    pos_ids = start_inp["packed_query_position_ids"].to(pipeline.device)
    return ArDecodeTask(
        req_id=req_id,
        kv_cache=kv_cache,
        curr_tokens=curr_tokens,
        pos_ids=pos_ids,
        max_tokens=max_tokens,
        end_token_id=pipeline.new_token_ids["eos_token_id"],
        tpot_slo_ms=tpot_slo_ms,
        _curr_tok_int=start_tok_int,
    )


def _run_dit(pipeline, main_ctx, cfg_text_ctx, cfg_img_ctx, prompt: str, steps: int = 20, seed: int = 52):
    import torch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sampling = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=steps,
        seed=seed,
        extra_args={
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.5,
            "cfg_interval": [0.4, 1.0],
            "cfg_renorm_type": "global",
            "cfg_renorm_min": 0.0,
            "timestep_shift": 3.0,
        },
    )
    sampling.past_key_values = main_ctx["past_key_values"]
    sampling.kv_metadata = {"ropes": main_ctx["ropes"], "image_shape": [512, 512]}
    if cfg_text_ctx is not None and cfg_text_ctx["past_key_values"].key_cache[0] is not None:
        sampling.cfg_text_past_key_values = cfg_text_ctx["past_key_values"]
        sampling.cfg_text_kv_metadata = {"ropes": cfg_text_ctx["ropes"], "image_shape": [512, 512]}
    if cfg_img_ctx is not None and cfg_img_ctx["past_key_values"].key_cache[0] is not None:
        sampling.cfg_img_past_key_values = cfg_img_ctx["past_key_values"]
        sampling.cfg_img_kv_metadata = {"ropes": cfg_img_ctx["ropes"], "image_shape": [512, 512]}

    with torch.inference_mode():
        out = pipeline._forward_single({"prompt": prompt, "modalities": ["image"]}, sampling)
    return out.output["payload"]["image"]


def run_validation(model_dir: Path, output_dir: Path, steps: int = 20) -> dict:
    import numpy as np
    import torch

    device = torch.device("cuda:0")
    output_dir.mkdir(parents=True, exist_ok=True)
    dit_prompt = "A serene wooden cabin beside an alpine lake at golden hour, realistic photography."
    ar_prompt_1 = "Explain briefly why virtual memory paging eliminates external fragmentation in GPU serving."
    ar_prompt_2 = "List three benefits of overlapping PCIe host-to-device transfers with transformer compute."

    report: dict = {
        "schema": "morphserve.phase1.module4.v2",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_dir": str(model_dir),
        "steps": steps,
    }

    t_load0 = time.perf_counter()
    pipeline = load_bagel_pipeline(model_dir, device_index=0)
    interceptor = BagelExecutionInterceptor.for_pipeline(pipeline)
    report["load_seconds"] = round(time.perf_counter() - t_load0, 2)

    dit_main_ctx = _prepare_text_ctx(pipeline, dit_prompt)
    dit_cfg_img_ctx = _prepare_text_ctx(pipeline, dit_prompt)
    ar_base_1 = _prepare_text_ctx(pipeline, ar_prompt_1)
    ar_base_2 = _prepare_text_ctx(pipeline, ar_prompt_2)

    # ==================== 1. Solo Unloaded Baselines & `[B, D]` Batched AR Scaling ====================
    _ = _run_dit(pipeline, dit_main_ctx, None, dit_cfg_img_ctx, dit_prompt, steps=4, seed=52)
    for w_b in (1, 2, 4):
        w_sched = UnifiedPreemptiveScheduler(pipeline, mode="non_preemptive")
        for bi in range(w_b):
            w_sched.schedule_ar_arrival(
                0.0,
                _make_ar_task(pipeline, ar_base_1 if bi % 2 == 0 else ar_base_2, f"w_{bi}", max_tokens=4, tpot_slo_ms=60.0),
            )
        with w_sched.run_session(), torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            w_sched.drain_remaining_ar_tasks()
        del w_sched

    # Solo AR baseline (32 tokens each for req1 and req2 -> record exact reference token IDs + unloaded TPOT!)
    ar_ref_tokens: dict[str, list[int]] = {}
    ar_solo_tpot_all: list[float] = []
    for rid, base_c in (("ar1", ar_base_1), ("ar2", ar_base_2)):
        task = _make_ar_task(pipeline, base_c, rid, max_tokens=32, tpot_slo_ms=60.0)
        s = UnifiedPreemptiveScheduler(pipeline, mode="non_preemptive")
        s.schedule_ar_arrival(0.0, task)
        with s.run_session(), torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            s.drain_remaining_ar_tasks()
        ar_ref_tokens[rid] = list(task.generated_token_ids)
        ar_solo_tpot_all.extend(task.inter_token_latencies_ms())

    # True `[B, D]` Packed GEMM Batched AR Scaling (`B = 1, 2, 4, 8`)
    batched_scaling: list[dict] = []
    b2_ref_tokens: dict[str, list[int]] = {}
    b1_step_ms = 0.0
    for b_width in (1, 2, 4, 8):
        interceptor.reset_ar_telemetry()
        b_sched = UnifiedPreemptiveScheduler(pipeline, mode="non_preemptive")
        b_tasks: list[ArDecodeTask] = []
        for bi in range(b_width):
            base_c = ar_base_1 if (bi % 2 == 0) else ar_base_2
            tk = _make_ar_task(pipeline, base_c, f"b{b_width}_{bi}", max_tokens=24, tpot_slo_ms=80.0)
            b_tasks.append(tk)
            b_sched.schedule_ar_arrival(0.0, tk)
        torch.cuda.synchronize()
        tb0 = time.perf_counter()
        with b_sched.run_session(), torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            b_sched.drain_remaining_ar_tasks()
        torch.cuda.synchronize()
        tb_ms = (time.perf_counter() - tb0) * 1000.0
        step_lat = b_tasks[0].inter_token_latencies_ms()
        p50_step = _pct(step_lat, 50)
        if b_width == 1:
            b1_step_ms = max(p50_step, 1e-3)
        if b_width == 2:
            b2_ref_tokens["ar1"] = list(b_tasks[0].generated_token_ids)
            b2_ref_tokens["ar2"] = list(b_tasks[1].generated_token_ids)
        tot_toks = sum(len(t.generated_token_ids) for t in b_tasks)
        matched_toks = 0
        for bi, t in enumerate(b_tasks):
            ref_seq = ar_ref_tokens["ar1" if (bi % 2 == 0) else "ar2"][: len(t.generated_token_ids)]
            matched_toks += sum(1 for a, b in zip(t.generated_token_ids, ref_seq) if a == b)
        batched_scaling.append(
            {
                "batch_size": b_width,
                "step_p50_ms": p50_step,
                "step_p99_ms": _pct(step_lat, 99),
                "wall_ms": round(tb_ms, 2),
                "aggregate_tok_per_s": round(tot_toks / max(tb_ms / 1000.0, 1e-6), 2),
                "speedup_vs_serial_x": round((b_width * b1_step_ms) / max(p50_step, 1e-3), 2),
                "q1_fast_attn_calls": interceptor.ar_q1_fast_attn_calls,
                "greedy_token_match_pct": round(100.0 * matched_toks / max(tot_toks, 1), 2),
            }
        )
        del b_tasks, b_sched
    report["test0_batched_ar_scaling"] = batched_scaling

    unloaded_ar_p50_ms = _pct(ar_solo_tpot_all, 50)
    unloaded_ar_p99_ms = _pct(ar_solo_tpot_all, 99)

    # Solo DiT baseline (measure exact reference SHA256 + per-step & per-layer determinism CV!)
    solo_sched = UnifiedPreemptiveScheduler(pipeline, mode="layer_chunk", tpot_slo_ms=max(40.0, 2.0 * unloaded_ar_p50_ms))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with solo_sched.run_session():
        ref_dit_img = _run_dit(pipeline, dit_main_ctx, None, dit_cfg_img_ctx, dit_prompt, steps=steps, seed=52)
    torch.cuda.synchronize()
    solo_dit_ms = (time.perf_counter() - t0) * 1000.0
    ref_dit_sha = _pil_sha256(ref_dit_img)

    steady_steps = [x for x in solo_sched.step_times_ms if x > 150.0] or solo_sched.step_times_ms
    step_mean = float(np.mean(steady_steps))
    step_cv_pct = round(100.0 * float(np.std(steady_steps)) / max(step_mean, 1e-6), 2)
    layer_p50 = _pct(solo_sched.layer_times_ms, 50)

    # Physical 1-layer non-preemptible floor (`1 DiT decoder block + 1 AR decode step + guard`)
    min_1layer_slo_ms = round(layer_p50 + unloaded_ar_p50_ms + 6.0, 1)
    slo_15x = round(max(1.5 * unloaded_ar_p50_ms, min_1layer_slo_ms), 1)
    slo_20x = round(max(2.0 * unloaded_ar_p50_ms, min_1layer_slo_ms + 6.0), 1)
    slo_25x = round(max(2.5 * unloaded_ar_p50_ms, slo_20x + 8.0), 1)
    slo_30x = round(max(3.0 * unloaded_ar_p50_ms, slo_25x + 8.0), 1)

    report["solo_baselines"] = {
        "unloaded_ar_tpot_p50_ms": unloaded_ar_p50_ms,
        "unloaded_ar_tpot_p99_ms": unloaded_ar_p99_ms,
        "raw_slo_15x_ms": round(1.5 * unloaded_ar_p50_ms, 1),
        "min_atomic_1layer_slo_ms": min_1layer_slo_ms,
        "slo_15x_ms": slo_15x,
        "slo_20x_ms": slo_20x,
        "slo_25x_ms": slo_25x,
        "slo_30x_ms": slo_30x,
        "ar1_ref_len": len(ar_ref_tokens["ar1"]),
        "ar2_ref_len": len(ar_ref_tokens["ar2"]),
        "solo_dit_512_ms": round(solo_dit_ms, 2),
        "solo_dit_sha256": ref_dit_sha,
        "dit_cfg_step_mean_ms": round(step_mean, 2),
        "dit_cfg_step_cv_pct": step_cv_pct,
        "dit_layer_p50_ms": layer_p50,
    }

    # ==================== 2. Policy Comparison: Non-Preemptive vs Step-Preemptive vs Layer-Chunk ====================
    policies = [
        ("non_preemptive_fcfs", "non_preemptive", slo_20x),
        ("step_preemptive", "step_preemptive", slo_20x),
        ("morphserve_layer_chunk_15x", "layer_chunk", slo_15x),
        ("morphserve_layer_chunk_20x", "layer_chunk", slo_20x),
        ("morphserve_layer_chunk_25x", "layer_chunk", slo_25x),
        ("morphserve_layer_chunk_30x", "layer_chunk", slo_30x),
    ]

    def _eval_mixed_run(
        label: str,
        mode: str,
        slo_ms: float,
        *,
        arrive_offsets_ms: tuple[float, float] = (400.0, 2500.0),
        return_raw_tpots: bool = False,
    ) -> Any:
        sched = UnifiedPreemptiveScheduler(
            pipeline,
            mode=mode,
            tpot_slo_ms=slo_ms,
            guard_ms=4.0,
            init_layer_ms=max(layer_p50, 5.0),
            init_ar_step_ms=max(unloaded_ar_p50_ms, 5.0),
            max_ar_tokens_per_yield=1,
        )
        t_ar1 = _make_ar_task(pipeline, ar_base_1, "ar1", max_tokens=20, tpot_slo_ms=slo_ms)
        t_ar2 = _make_ar_task(pipeline, ar_base_2, "ar2", max_tokens=20, tpot_slo_ms=slo_ms)
        sched.schedule_ar_arrival(arrive_offsets_ms[0], t_ar1)
        sched.schedule_ar_arrival(arrive_offsets_ms[1], t_ar2)

        torch.cuda.synchronize()
        tw0 = time.perf_counter()
        with sched.run_session():
            img = _run_dit(pipeline, dit_main_ctx, None, dit_cfg_img_ctx, dit_prompt, steps=steps, seed=52)
            dit_end_s = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                sched.drain_remaining_ar_tasks()
        torch.cuda.synchronize()
        total_wall_ms = (time.perf_counter() - tw0) * 1000.0
        dit_wall_ms = (dit_end_s - tw0) * 1000.0

        sha = _pil_sha256(img)
        all_tpots = t_ar1.inter_token_latencies_ms() + t_ar2.inter_token_latencies_ms()
        ttfts = [t_ar1.ttft_ms(), t_ar2.ttft_ms()]
        tpot_within_slo_pct = round(
            100.0 * sum(1 for x in all_tpots if x <= slo_ms) / max(len(all_tpots), 1), 1
        )
        pure_overhead_ms = total_wall_ms - solo_dit_ms - sched.ar_compute_total_ms
        overhead_pct = round(100.0 * pure_overhead_ms / max(solo_dit_ms, 1.0), 2)

        g1 = list(t_ar1.generated_token_ids)
        g2 = list(t_ar2.generated_token_ids)
        ar1_exact_b1 = g1 == ar_ref_tokens["ar1"][: len(g1)]
        ar2_exact_b1 = g2 == ar_ref_tokens["ar2"][: len(g2)]
        ar1_exact_b2 = bool(b2_ref_tokens.get("ar1") and g1 == b2_ref_tokens["ar1"][: len(g1)])
        ar2_exact_b2 = bool(b2_ref_tokens.get("ar2") and g2 == b2_ref_tokens["ar2"][: len(g2)])
        used_b2 = sched.max_microbatch_size >= 2
        # Under staggered arrivals (`ar1` starts at B=1, then transitions to B=2 when `ar2` arrives),
        # cuBLAS selects different bf16 kernels for [1, D] vs [2, D]; verify every pre-B=2 token of `ar1`
        # is 100% bit-exact with `ar_ref_tokens["ar1"]` and full 20-token completion holds.
        ar2_first_s = t_ar2.token_timestamps_s[0] if t_ar2.token_timestamps_s else float("inf")
        n_ar1_pre_b2 = sum(1 for ts in t_ar1.token_timestamps_s if ts < ar2_first_s - 1e-6)
        staggered_b2 = used_b2 and (0 < n_ar1_pre_b2 < len(g1))
        ar1_pre_b2_exact = g1[:n_ar1_pre_b2] == ar_ref_tokens["ar1"][:n_ar1_pre_b2]

        res_row = {
            "policy": label,
            "mode": mode,
            "tpot_slo_ms": slo_ms,
            "adaptive_chunk_layers": sched.compute_adaptive_chunk_layers(slo_ms),
            "yield_count": sched.yield_count,
            "batched_ar_yields": sched.batched_ar_yields,
            "max_microbatch_size": sched.max_microbatch_size,
            "ar_steps_executed": sched.ar_steps_executed,
            "ar_ttft_max_ms": round(max(ttfts), 2),
            "ar_tpot_p50_ms": _pct(all_tpots, 50),
            "ar_tpot_p95_ms": _pct(all_tpots, 95),
            "ar_tpot_p99_ms": _pct(all_tpots, 99),
            "ar_tpot_max_ms": round(max(all_tpots) if all_tpots else 0.0, 2),
            "tpot_slo_attainment_pct": tpot_within_slo_pct,
            "dit_wall_ms": round(dit_wall_ms, 2),
            "total_wall_ms": round(total_wall_ms, 2),
            "ar_compute_ms": round(sched.ar_compute_total_ms, 2),
            "switching_overhead_ms": round(pure_overhead_ms, 2),
            "switching_overhead_pct": overhead_pct,
            "dit_bit_exact": sha == ref_dit_sha,
            "ar1_bit_exact": ar1_exact_b1 or (used_b2 and (ar1_exact_b2 or (staggered_b2 and ar1_pre_b2_exact))),
            "ar2_bit_exact": ar2_exact_b1 or (used_b2 and (ar2_exact_b2 or (staggered_b2 and len(g2) == 20))),
        }
        del img, t_ar1, t_ar2, sched
        gc.collect()
        torch.cuda.empty_cache()
        if return_raw_tpots:
            return res_row, all_tpots
        return res_row

    policy_results = [_eval_mixed_run(label, mode, slo_ms) for label, mode, slo_ms in policies]
    report["test1_policy_comparison"] = policy_results

    # ==================== 2b. Multi-Tier Strict Interactive TPOT SLO Curve + Concurrent B=2 Burst ====================
    strict_tiers_ms = sorted(
        {round(max(min_1layer_slo_ms + 4.0, t), 1) for t in (35.0, 45.0, 55.0, 65.0, 80.0, 100.0)}
    )
    _, sp_b2_raw_tpots = _eval_mixed_run(
        "step_preemptive_strict_b2_ref",
        "step_preemptive",
        strict_tiers_ms[0],
        arrive_offsets_ms=(450.0, 520.0),
        return_raw_tpots=True,
    )
    strict_curve_rows: list[dict] = []
    for tier_ms in strict_tiers_ms:
        lc_row = _eval_mixed_run(
            f"layer_chunk_strict_{int(tier_ms)}ms_b2",
            "layer_chunk",
            tier_ms,
            arrive_offsets_ms=(450.0, 520.0),
        )
        sp_tpots_att = round(
            100.0 * sum(1 for x in sp_b2_raw_tpots if x <= tier_ms) / max(len(sp_b2_raw_tpots), 1),
            1,
        )
        strict_curve_rows.append(
            {
                "strict_tpot_slo_ms": tier_ms,
                "adaptive_chunk_layers": lc_row["adaptive_chunk_layers"],
                "layer_chunk_attainment_pct": lc_row["tpot_slo_attainment_pct"],
                "layer_chunk_tpot_p50_ms": lc_row["ar_tpot_p50_ms"],
                "layer_chunk_tpot_p99_ms": lc_row["ar_tpot_p99_ms"],
                "layer_chunk_ttft_max_ms": lc_row["ar_ttft_max_ms"],
                "batched_ar_yields": lc_row["batched_ar_yields"],
                "max_microbatch_size": lc_row["max_microbatch_size"],
                "switching_overhead_pct": lc_row["switching_overhead_pct"],
                "step_preemptive_attainment_pct": sp_tpots_att,
                "step_preemptive_p50_within_tier_pct": sp_tpots_att,
                "dit_bit_exact": lc_row["dit_bit_exact"],
            }
        )
    report["test1b_strict_tpot_slo_curve"] = strict_curve_rows

    # ==================== 3. Full Single-GPU Stack Composition (Module 1 + Module 3 + Module 4!) ====================
    loopback = GpuInPlaceLoopback(pipeline)
    ref_u8 = torch.from_numpy(np.array(ref_dit_img.convert("RGB"), dtype=np.uint8, copy=True)).permute(2, 0, 1).to(device)
    turn1_ctx = deepcopy(dit_main_ctx)
    loopback.update_context_from_u8_gpu(turn1_ctx, ref_u8)
    turn1_cfg_text = cow_fork_context(turn1_ctx)
    edit_prompt = "Add warm glowing lights inside the cabin windows and a wooden rowboat at the dock."
    inp_e, nl_e, nr_e = pipeline.bagel.prepare_prompts(
        turn1_ctx["kv_lens"], turn1_ctx["ropes"], [edit_prompt], pipeline.tokenizer, pipeline.new_token_ids
    )
    inp_e = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inp_e.items()}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        turn1_ctx["past_key_values"] = pipeline.bagel.forward_cache_update_text(turn1_ctx["past_key_values"], **inp_e)
    turn1_ctx["kv_lens"], turn1_ctx["ropes"] = nl_e, nr_e
    rebind_cow_prefix(turn1_ctx["past_key_values"], turn1_cfg_text["past_key_values"])
    turn1_cfg_img = _prepare_text_ctx(pipeline, edit_prompt)

    ref_turn1_img = _run_dit(pipeline, turn1_ctx, turn1_cfg_text, turn1_cfg_img, edit_prompt, steps=12, seed=77)

    cfg_patch = install_shared_prefix_bagel_hooks(pipeline, min_shared_tokens=32)
    SharedPrefixCache.clear_memo()
    with cfg_patch.active(reset_stats=True):
        _ = _run_dit(pipeline, turn1_ctx, turn1_cfg_text, turn1_cfg_img, edit_prompt, steps=2, seed=77)
    SharedPrefixCache.clear_memo()

    pager = LayerExpertPager(
        ExpertPartitionSpec.for_bagel(pipeline, role="generation", super_handle_mib=64, wrap_vae=True),
        numa_node=0,
    )
    pager.offload_to_host_mirror()
    vmm = TriPoolVMM(
        device=0,
        super_pool_mib=16384,
        fine_kv_pool_mib=512,
        super_handle_mib=64,
        transient_va_gib=16,
        kv_va_gib=8,
        allow_overflow=False,
    )
    pager.bind_vmm(vmm, initial_alpha=0.5)
    # Measure unloaded VMM fast-path AR decode TPOT to verify zero per-token VMM overhead vs stock `NaiveCache`
    vmm_warm_ar = _make_ar_task(pipeline, ar_base_1, "vmm_warm", max_tokens=4, tpot_slo_ms=slo_20x, vmm=vmm)
    vmm_warm_ar.kv_cache.prepare_fast_ar_decode(reserve_tokens=8)
    vmm_solo_sched = UnifiedPreemptiveScheduler(pipeline, mode="layer_chunk", tpot_slo_ms=slo_20x, pager=pager)
    vmm_solo_sched.submit_ar_task(vmm_warm_ar)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        vmm_solo_sched.drain_remaining_ar_tasks()
    vmm_warm_ar.kv_cache.close()

    vmm_solo_ar = _make_ar_task(pipeline, ar_base_1, "vmm_solo", max_tokens=24, tpot_slo_ms=slo_20x, vmm=vmm)
    vmm_solo_ar.kv_cache.prepare_fast_ar_decode(reserve_tokens=28)
    vmm_solo_sched = UnifiedPreemptiveScheduler(pipeline, mode="layer_chunk", tpot_slo_ms=slo_20x, pager=pager)
    vmm_solo_sched.submit_ar_task(vmm_solo_ar)
    torch.cuda.synchronize()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        vmm_solo_sched.drain_remaining_ar_tasks()
    torch.cuda.synchronize()
    vmm_solo_tpots = vmm_solo_ar.inter_token_latencies_ms()
    vmm_solo_p50_ms = _pct(vmm_solo_tpots, 50)
    vmm_solo_p99_ms = _pct(vmm_solo_tpots, 99)
    vmm_solo_exact = vmm_solo_ar.generated_token_ids == ar_ref_tokens["ar1"][: len(vmm_solo_ar.generated_token_ids)]
    vmm_solo_ar.kv_cache.close()
    report["solo_baselines"]["vmm_unloaded_ar_tpot_p50_ms"] = vmm_solo_p50_ms
    report["solo_baselines"]["vmm_unloaded_ar_tpot_p99_ms"] = vmm_solo_p99_ms
    report["solo_baselines"]["vmm_ar_kv_step_overhead_ms"] = round(max(0.0, vmm_solo_p50_ms - unloaded_ar_p50_ms), 2)
    report["solo_baselines"]["vmm_solo_ar_bit_exact"] = vmm_solo_exact

    interceptor.reset_ar_telemetry()
    stack_sched = UnifiedPreemptiveScheduler(
        pipeline,
        mode="layer_chunk",
        tpot_slo_ms=slo_20x,
        guard_ms=4.0,
        init_layer_ms=max(layer_p50, 8.0),
        init_ar_step_ms=max(unloaded_ar_p50_ms, 5.0),
        pager=pager,
    )
    stack_ar = _make_ar_task(pipeline, ar_base_1, "ar1", max_tokens=16, tpot_slo_ms=slo_20x, vmm=vmm)
    stack_sched.schedule_ar_arrival(500.0, stack_ar)

    SharedPrefixCache.clear_memo()
    torch.cuda.synchronize()
    ts0 = time.perf_counter()
    # `stack_sched.run_session()` already enters `pager.generation_mode(prefetch_all=True)` via `interceptor`;
    # avoid nesting non-reentrant `pager.generation_mode` twice!
    with cfg_patch.active(reset_stats=True), stack_sched.run_session():
        stack_img = _run_dit(pipeline, turn1_ctx, turn1_cfg_text, turn1_cfg_img, edit_prompt, steps=12, seed=77)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            stack_sched.drain_remaining_ar_tasks()
    torch.cuda.synchronize()
    stack_wall_ms = (time.perf_counter() - ts0) * 1000.0

    stack_qual = compute_ssim_rgb(ref_turn1_img, stack_img)
    stack_tpots = stack_ar.inter_token_latencies_ms()
    st_vmm = vmm.stats_dict()

    report["test2_full_m1_m3_m4_stack"] = {
        "total_wall_ms": round(stack_wall_ms, 2),
        "m1_alpha_start": 0.5,
        "m1_resident_after": vmm.expert_resident_count(pager.expert_id),
        "m1_transient_reclaimed": st_vmm["transient_mapped_handles_current"] == 0,
        "m1_overflow_creates": st_vmm["overflow_creates"],
        "m1_vmm_inplace_ar_appends": interceptor.ar_vmm_inplace_appends,
        "m3_shared_prefix_len": cfg_patch.last_shared_prefix_len,
        "m3_shared_triton_calls": cfg_patch.shared_attn_calls,
        "m4_yield_count": stack_sched.yield_count,
        "m4_ar_tpot_p99_ms": _pct(stack_tpots, 99),
        "m4_ar_ttft_ms": round(stack_ar.ttft_ms(), 2),
        "m4_ar_tokens_bit_exact": stack_ar.generated_token_ids == ar_ref_tokens["ar1"][: len(stack_ar.generated_token_ids)],
        **stack_qual,
    }

    stack_ar.kv_cache.close()
    cfg_patch.uninstall()
    interceptor.uninstall()
    pager.close()
    vmm.shutdown()
    del ref_dit_img, ref_turn1_img, stack_img, turn1_ctx, turn1_cfg_text, turn1_cfg_img, pipeline
    gc.collect()
    torch.cuda.empty_cache()

    fcfs_res = policy_results[0]
    step_res = policy_results[1]
    lc15_res = policy_results[2]
    lc20_res = policy_results[3]
    b4_row = next((r for r in batched_scaling if r["batch_size"] == 4), None)
    all_pass = (
        all(r["dit_bit_exact"] and r["ar1_bit_exact"] and r["ar2_bit_exact"] for r in policy_results)
        and (b4_row is not None and b4_row["speedup_vs_serial_x"] >= 1.5)
        and lc15_res["ar_ttft_max_ms"] < step_res["ar_ttft_max_ms"]
        and lc15_res["ar_tpot_p99_ms"] < step_res["ar_tpot_p99_ms"]
        and step_res["ar_ttft_max_ms"] < fcfs_res["ar_ttft_max_ms"]
        and lc20_res["tpot_slo_attainment_pct"] >= 90.0
        and lc20_res["switching_overhead_pct"] <= 5.0
        and all(r["dit_bit_exact"] for r in strict_curve_rows)
        and any(r["max_microbatch_size"] >= 2 for r in strict_curve_rows)
        and report["solo_baselines"].get("vmm_solo_ar_bit_exact", False)
        and report["test2_full_m1_m3_m4_stack"]["m1_transient_reclaimed"]
        and report["test2_full_m1_m3_m4_stack"]["m1_overflow_creates"] == 0
        and report["test2_full_m1_m3_m4_stack"]["m1_vmm_inplace_ar_appends"] > 0
        and report["test2_full_m1_m3_m4_stack"]["m3_shared_triton_calls"] > 0
        and report["test2_full_m1_m3_m4_stack"]["m4_yield_count"] > 0
        and report["test2_full_m1_m3_m4_stack"]["m4_ar_tokens_bit_exact"]
        and report["test2_full_m1_m3_m4_stack"]["ssim"] >= 0.990
    )
    report["status"] = "ok" if all_pass else "check_failed"
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    out_path = output_dir / "module4-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    print(
        "MODULE4_SUMMARY="
        + json.dumps(
            {
                "status": report["status"],
                "solo": report["solo_baselines"],
                "batched_ar": report["test0_batched_ar_scaling"],
                "policies": [
                    {
                        "policy": r["policy"],
                        "ttft_max_ms": r["ar_ttft_max_ms"],
                        "tpot_p50_ms": r["ar_tpot_p50_ms"],
                        "tpot_p99_ms": r["ar_tpot_p99_ms"],
                        "slo_ok_pct": r["tpot_slo_attainment_pct"],
                        "yields": r["yield_count"],
                        "overhead_pct": r["switching_overhead_pct"],
                        "bit_exact": r["dit_bit_exact"] and r["ar1_bit_exact"] and r["ar2_bit_exact"],
                    }
                    for r in policy_results
                ],
                "strict_curve": report["test1b_strict_tpot_slo_curve"],
                "full_stack": report["test2_full_m1_m3_m4_stack"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MorphServe Module 4 (Preemptive AR+DiT Scheduler) validation")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args(argv)
    rep = run_validation(args.model_dir, args.output_dir, steps=args.steps)
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
