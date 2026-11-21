"""Phase-1 Module 4 live A100 validation: Deterministic Layer-Chunk Preemptive AR+DiT Scheduler + M1+M3+M4 Stack."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import time

from morphserve.bench.validate_module1_bagel import load_bagel_pipeline
from morphserve.kernels import (
    GpuInPlaceLoopback,
    SharedPrefixCache,
    cow_fork_context,
    install_shared_prefix_bagel_hooks,
    rebind_cow_prefix,
)
from morphserve.vmm import BagelExpertPager, TriPoolVMM
from morphserve.worker import ArDecodeTask, UnifiedPreemptiveScheduler


def _pil_sha256(img) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(img.convert("RGB"), dtype=np.uint8).tobytes()).hexdigest()


def _pct(vals: list[float], p: float) -> float:
    import numpy as np

    if not vals:
        return 0.0
    return round(float(np.percentile(vals, p)), 2)


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


def _make_ar_task(pipeline, base_ctx: dict, req_id: str, max_tokens: int, tpot_slo_ms: float) -> ArDecodeTask:
    import torch

    # Clone KV cache so each run starts from identical clean state
    ctx_copy = deepcopy(base_ctx)
    start_inp = pipeline.bagel.prepare_start_tokens(ctx_copy["kv_lens"], ctx_copy["ropes"], pipeline.new_token_ids)
    curr_tokens = start_inp["packed_start_tokens"].to(pipeline.device)
    pos_ids = start_inp["packed_query_position_ids"].to(pipeline.device)
    return ArDecodeTask(
        req_id=req_id,
        kv_cache=ctx_copy["past_key_values"],
        curr_tokens=curr_tokens,
        pos_ids=pos_ids,
        max_tokens=max_tokens,
        end_token_id=pipeline.new_token_ids["eos_token_id"],
        tpot_slo_ms=tpot_slo_ms,
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
        "schema": "morphserve.phase1.module4.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_dir": str(model_dir),
        "steps": steps,
    }

    t_load0 = time.perf_counter()
    pipeline = load_bagel_pipeline(model_dir, device_index=0)
    report["load_seconds"] = round(time.perf_counter() - t_load0, 2)

    dit_main_ctx = _prepare_text_ctx(pipeline, dit_prompt)
    dit_cfg_img_ctx = _prepare_text_ctx(pipeline, dit_prompt)
    ar_base_1 = _prepare_text_ctx(pipeline, ar_prompt_1)
    ar_base_2 = _prepare_text_ctx(pipeline, ar_prompt_2)

    # ==================== 1. Solo Unloaded Baselines & DiT Determinism Check ====================
    # Warmup DiT + AR once
    _ = _run_dit(pipeline, dit_main_ctx, None, dit_cfg_img_ctx, dit_prompt, steps=4, seed=52)
    _warm_ar = _make_ar_task(pipeline, ar_base_1, "warm", max_tokens=8, tpot_slo_ms=60.0)
    _warm_sched = UnifiedPreemptiveScheduler(pipeline, mode="non_preemptive")
    _warm_sched.schedule_ar_arrival(0.0, _warm_ar)
    with _warm_sched.run_session(), torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _warm_sched.drain_remaining_ar_tasks()
    del _warm_ar, _warm_sched

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

    unloaded_ar_p50_ms = _pct(ar_solo_tpot_all, 50)
    unloaded_ar_p99_ms = _pct(ar_solo_tpot_all, 99)
    slo_15x = round(1.5 * unloaded_ar_p50_ms, 1)
    slo_20x = round(2.0 * unloaded_ar_p50_ms, 1)
    slo_25x = round(2.5 * unloaded_ar_p50_ms, 1)

    # Solo DiT baseline (measure exact reference SHA256 + per-step & per-layer determinism CV!)
    solo_sched = UnifiedPreemptiveScheduler(pipeline, mode="layer_chunk", tpot_slo_ms=slo_20x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with solo_sched.run_session():
        ref_dit_img = _run_dit(pipeline, dit_main_ctx, None, dit_cfg_img_ctx, dit_prompt, steps=steps, seed=52)
    torch.cuda.synchronize()
    solo_dit_ms = (time.perf_counter() - t0) * 1000.0
    ref_dit_sha = _pil_sha256(ref_dit_img)

    # Exclude first 1-2 CFG-window transition steps when computing steady-state CV
    steady_steps = [x for x in solo_sched.step_times_ms if x > 150.0] or solo_sched.step_times_ms
    step_mean = float(np.mean(steady_steps))
    step_cv_pct = round(100.0 * float(np.std(steady_steps)) / max(step_mean, 1e-6), 2)
    layer_p50 = _pct(solo_sched.layer_times_ms, 50)
    report["solo_baselines"] = {
        "unloaded_ar_tpot_p50_ms": unloaded_ar_p50_ms,
        "unloaded_ar_tpot_p99_ms": unloaded_ar_p99_ms,
        "slo_15x_ms": slo_15x,
        "slo_20x_ms": slo_20x,
        "slo_25x_ms": slo_25x,
        "ar1_ref_len": len(ar_ref_tokens["ar1"]),
        "ar2_ref_len": len(ar_ref_tokens["ar2"]),
        "solo_dit_512_ms": round(solo_dit_ms, 2),
        "solo_dit_sha256": ref_dit_sha,
        "dit_cfg_step_mean_ms": round(step_mean, 2),
        "dit_cfg_step_cv_pct": step_cv_pct,
        "dit_layer_p50_ms": layer_p50,
    }

    # ==================== 2. Policy Comparison: Non-Preemptive vs Step-Preemptive vs Layer-Chunk ====================
    # Mixed workload: DiT (512x512, 20 steps, ~4.95s) starts at t=0;
    #   - AR request 1 (20 tokens) arrives at t = 400 ms (mid-DiT!)
    #   - AR request 2 (20 tokens) arrives at t = 2500 ms (mid-DiT!)
    # Anchored at 1.5x / 2.0x / 2.5x unloaded AR TPOT (`unloaded_ar_p50_ms`).
    policies = [
        ("non_preemptive_fcfs", "non_preemptive", slo_20x),
        ("step_preemptive", "step_preemptive", slo_20x),
        ("morphserve_layer_chunk_15x", "layer_chunk", slo_15x),
        ("morphserve_layer_chunk_20x", "layer_chunk", slo_20x),
        ("morphserve_layer_chunk_25x", "layer_chunk", slo_25x),
    ]

    policy_results = []
    for label, mode, slo_ms in policies:
        sched = UnifiedPreemptiveScheduler(
            pipeline,
            mode=mode,
            tpot_slo_ms=slo_ms,
            guard_ms=4.0,
            init_layer_ms=max(layer_p50, 5.0),
            init_ar_step_ms=max(unloaded_ar_p50_ms, 15.0),
            max_ar_tokens_per_yield=1,
        )
        t_ar1 = _make_ar_task(pipeline, ar_base_1, "ar1", max_tokens=20, tpot_slo_ms=slo_ms)
        t_ar2 = _make_ar_task(pipeline, ar_base_2, "ar2", max_tokens=20, tpot_slo_ms=slo_ms)
        sched.schedule_ar_arrival(400.0, t_ar1)
        sched.schedule_ar_arrival(2500.0, t_ar2)

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
        # Include TTFT wait from arrival -> first token as first-token blocking metric!
        ttfts = [t_ar1.ttft_ms(), t_ar2.ttft_ms()]
        tpot_within_slo_pct = round(
            100.0 * sum(1 for x in all_tpots if x <= slo_ms) / max(len(all_tpots), 1), 1
        )
        # Pure preemption switching overhead = (total_wall - solo_dit - ar_compute_total)
        pure_overhead_ms = total_wall_ms - solo_dit_ms - sched.ar_compute_total_ms
        overhead_pct = round(100.0 * pure_overhead_ms / max(solo_dit_ms, 1.0), 2)

        policy_results.append(
            {
                "policy": label,
                "mode": mode,
                "tpot_slo_ms": slo_ms,
                "adaptive_chunk_layers": sched.compute_adaptive_chunk_layers(slo_ms),
                "yield_count": sched.yield_count,
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
                "ar1_bit_exact": t_ar1.generated_token_ids == ar_ref_tokens["ar1"][:20],
                "ar2_bit_exact": t_ar2.generated_token_ids == ar_ref_tokens["ar2"][:20],
            }
        )
        del img, t_ar1, t_ar2, sched
        gc.collect()
        torch.cuda.empty_cache()

    report["test1_policy_comparison"] = policy_results

    # ==================== 3. Full Single-GPU Stack Composition (Module 1 + Module 3 + Module 4!) ====================
    # Run TriPool-VMM (alpha=0.50 streaming + scoped VAE) + SharedPrefixCache Triton CFG + Layer-Chunk Preemption
    # simultaneously on a Turn-1 multi-turn editing context + concurrent AR decode!
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

    # Reference Turn-1 edit image BEFORE attaching Module 1 + 3 + 4
    ref_turn1_img = _run_dit(pipeline, turn1_ctx, turn1_cfg_text, turn1_cfg_img, edit_prompt, steps=12, seed=77)

    # Pre-warm FlashPrefix-CFG & LSE merge kernels before attaching TriPool-VMM + Scheduler
    cfg_patch = install_shared_prefix_bagel_hooks(pipeline, min_shared_tokens=32)
    SharedPrefixCache.clear_memo()
    with cfg_patch.active(reset_stats=True):
        _ = _run_dit(pipeline, turn1_ctx, turn1_cfg_text, turn1_cfg_img, edit_prompt, steps=2, seed=77)
    SharedPrefixCache.clear_memo()

    # Attach Module 1 (TriPool-VMM) + Module 3 (SharedPrefixCache + FlashPrefix CFG) + Module 4 (Layer-Chunk Preemption)!
    pager = BagelExpertPager(pipeline, role="generation", numa_node=0, super_handle_mib=64, wrap_vae=True)
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
    stack_sched = UnifiedPreemptiveScheduler(
        pipeline,
        mode="layer_chunk",
        tpot_slo_ms=slo_20x,
        guard_ms=4.0,
        init_layer_ms=max(layer_p50, 8.0),
        init_ar_step_ms=max(unloaded_ar_p50_ms, 15.0),
        pager=pager,
    )
    stack_ar = _make_ar_task(pipeline, ar_base_1, "ar1", max_tokens=16, tpot_slo_ms=slo_20x)
    stack_sched.schedule_ar_arrival(500.0, stack_ar)

    SharedPrefixCache.clear_memo()
    torch.cuda.synchronize()
    ts0 = time.perf_counter()
    with pager.generation_mode(prefetch_all=True), cfg_patch.active(reset_stats=True), stack_sched.run_session():
        stack_img = _run_dit(pipeline, turn1_ctx, turn1_cfg_text, turn1_cfg_img, edit_prompt, steps=12, seed=77)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            stack_sched.drain_remaining_ar_tasks()
    torch.cuda.synchronize()
    stack_wall_ms = (time.perf_counter() - ts0) * 1000.0

    from morphserve.bench.validate_module3_cfg_loopback import _compute_ssim_rgb

    stack_qual = _compute_ssim_rgb(ref_turn1_img, stack_img)
    stack_tpots = stack_ar.inter_token_latencies_ms()
    st_vmm = vmm.stats_dict()

    report["test2_full_m1_m3_m4_stack"] = {
        "total_wall_ms": round(stack_wall_ms, 2),
        "m1_alpha_start": 0.5,
        "m1_resident_after": vmm.expert_resident_count(pager.expert_id),
        "m1_transient_reclaimed": st_vmm["transient_mapped_handles_current"] == 0,
        "m1_overflow_creates": st_vmm["overflow_creates"],
        "m3_shared_prefix_len": cfg_patch.last_shared_prefix_len,
        "m3_shared_triton_calls": cfg_patch.shared_attn_calls,
        "m4_yield_count": stack_sched.yield_count,
        "m4_ar_tpot_p99_ms": _pct(stack_tpots, 99),
        "m4_ar_ttft_ms": round(stack_ar.ttft_ms(), 2),
        "m4_ar_tokens_bit_exact": stack_ar.generated_token_ids == ar_ref_tokens["ar1"][: len(stack_ar.generated_token_ids)],
        **stack_qual,
    }

    cfg_patch.uninstall()
    pager.close()
    vmm.shutdown()
    del ref_dit_img, ref_turn1_img, stack_img, turn1_ctx, turn1_cfg_text, turn1_cfg_img, pipeline
    gc.collect()
    torch.cuda.empty_cache()

    fcfs_res = policy_results[0]
    step_res = policy_results[1]
    lc15_res = policy_results[2]
    lc20_res = policy_results[3]
    all_pass = (
        all(r["dit_bit_exact"] and r["ar1_bit_exact"] and r["ar2_bit_exact"] for r in policy_results)
        and lc15_res["ar_ttft_max_ms"] < step_res["ar_ttft_max_ms"]
        and lc15_res["ar_tpot_p99_ms"] < step_res["ar_tpot_p99_ms"]
        and step_res["ar_ttft_max_ms"] < fcfs_res["ar_ttft_max_ms"]
        and lc20_res["tpot_slo_attainment_pct"] >= 90.0
        and lc20_res["switching_overhead_pct"] <= 5.0
        and report["test2_full_m1_m3_m4_stack"]["m1_transient_reclaimed"]
        and report["test2_full_m1_m3_m4_stack"]["m1_overflow_creates"] == 0
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
