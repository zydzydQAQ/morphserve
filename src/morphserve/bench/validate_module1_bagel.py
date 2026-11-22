"""Phase-1 Module 1 live A100 validation: TriPool-VMM integrated with real BAGEL-7B-MoT."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
from pathlib import Path
import time

from morphserve.bench.exactness import MIB, gpu_mem_info_mib, pil_sha256, tensor_sha256
from morphserve.models import load_bagel_pipeline
from morphserve.vmm import ExpertPartitionSpec, LayerExpertPager, TriPoolVMM


def _build_turn0_context(pipeline, prompt: str, device):
    import torch
    from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache

    bagel = pipeline.bagel
    num_layers = bagel.config.llm_config.num_hidden_layers

    def make_ctx():
        ctx = {"kv_lens": [0], "ropes": [0], "past_key_values": NaiveCache(num_layers)}
        inputs, kv_lens, ropes = bagel.prepare_prompts(
            curr_kvlens=ctx["kv_lens"],
            curr_rope=ctx["ropes"],
            prompts=[prompt],
            tokenizer=pipeline.tokenizer,
            new_token_ids=pipeline.new_token_ids,
        )
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ctx["past_key_values"] = bagel.forward_cache_update_text(ctx["past_key_values"], **inputs)
        ctx["kv_lens"] = kv_lens
        ctx["ropes"] = ropes
        return ctx

    return make_ctx(), make_ctx()


def _run_bagel_generate(
    pipeline,
    main_ctx,
    cfg_img_ctx,
    prompt: str,
    steps: int = 20,
    seed: int = 52,
    height: int = 512,
    width: int = 512,
    sync_before: bool = True,
):
    import torch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sampling = OmniDiffusionSamplingParams(
        height=height,
        width=width,
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
    sampling.kv_metadata = {"ropes": main_ctx["ropes"], "image_shape": [height, width]}
    sampling.cfg_img_past_key_values = cfg_img_ctx["past_key_values"]
    sampling.cfg_img_kv_metadata = {"ropes": cfg_img_ctx["ropes"], "image_shape": [height, width]}

    if sync_before:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = pipeline._forward_single({"prompt": prompt, "modalities": ["image"]}, sampling)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return out.output["payload"]["image"], elapsed_ms


def run_validation(model_dir: Path, output_dir: Path, steps: int = 20) -> dict:
    import numpy as np
    import torch

    device = torch.device("cuda:0")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = "A serene wooden cabin beside an alpine lake at golden hour, realistic photography."

    report: dict = {
        "schema": "morphserve.phase1.module1.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_dir": str(model_dir),
        "steps": steps,
    }

    # ==================== 0. Unmodified Baseline Reference ====================
    t_load0 = time.perf_counter()
    pipeline = load_bagel_pipeline(model_dir, device_index=0)
    report["load_seconds"] = round(time.perf_counter() - t_load0, 2)
    report["mem_monolithic_loaded"] = gpu_mem_info_mib(0)
    try:
        import shutil

        layer0 = pipeline.bagel.language_model.model.layers[0]
        pipe_file = inspect.getsourcefile(type(pipeline))
        bagel_file = inspect.getsourcefile(type(pipeline.bagel))
        llm_file = inspect.getsourcefile(type(pipeline.bagel.language_model.model))
        ref_dir = output_dir / "vllm_omni_ref"
        ref_dir.mkdir(parents=True, exist_ok=True)
        for src_p in {pipe_file, bagel_file, llm_file}:
            if src_p and Path(src_p).is_file():
                shutil.copy2(src_p, ref_dir / Path(src_p).name)
        report["bagel_introspection"] = {
            "pipeline_file": pipe_file,
            "bagel_file": bagel_file,
            "llm_file": llm_file,
            "layer0_class": type(layer0).__name__,
            "layer0_params": [
                {
                    "name": n,
                    "shape": list(p.shape),
                    "dtype": str(p.dtype),
                    "mib": round(p.numel() * p.element_size() / MIB, 2),
                }
                for n, p in layer0.named_parameters()
            ],
            "layer0_forward_methods": [m for m in dir(layer0) if "forward" in m],
            "pipeline_forward_methods": [m for m in dir(pipeline) if "forward" in m or "diffuse" in m or "step" in m],
        }
    except Exception as exc:
        report["bagel_introspection"] = {"error": str(exc)}

    # Warmup + Reference generation (512x512, 20 steps)
    main_ctx, cfg_img_ctx = _build_turn0_context(pipeline, prompt, device)
    _img_warm, _ = _run_bagel_generate(pipeline, main_ctx, cfg_img_ctx, prompt, steps=steps, seed=52)
    del _img_warm
    ref_img, ref_gen_ms = _run_bagel_generate(pipeline, main_ctx, cfg_img_ctx, prompt, steps=steps, seed=52)
    ref_sha = pil_sha256(ref_img)
    ref_arr = np.asarray(ref_img.convert("RGB"), dtype=np.int16)

    # Reference 1024x1024 batch=2 FLUX VAE decode (fixed latent) for Test 2 bit-exactness!
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    vae_lat_1024_b2 = torch.randn((2, 16, 128, 128), dtype=torch.bfloat16, device=device)
    torch.cuda.synchronize()
    t_vref0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        vae_ref_out = pipeline.vae.decode(vae_lat_1024_b2)
    torch.cuda.synchronize()
    vae_ref_ms = (time.perf_counter() - t_vref0) * 1000.0
    vae_ref_sha = tensor_sha256(vae_ref_out)
    del vae_ref_out
    gc.collect()
    torch.cuda.empty_cache()

    report["baseline_reference"] = {
        "ref_sha256": ref_sha,
        "ref_gen_ms": round(ref_gen_ms, 2),
        "vae_1024_b2_ref_ms": round(vae_ref_ms, 2),
        "vae_1024_b2_ref_sha256": vae_ref_sha,
    }

    # ==================== 1. Offload Gen Expert -> NUMA memfd & Verify HBM Drop ====================
    pager = LayerExpertPager(
        ExpertPartitionSpec.for_bagel(pipeline, role="generation", super_handle_mib=64, wrap_vae=True),
        numa_node=0,
    )
    offload_info = pager.offload_to_host_mirror()
    mem_after_offload = gpu_mem_info_mib(0)
    report["offload_info"] = offload_info
    report["mem_after_offload"] = mem_after_offload

    # ==================== 2. Test 1: Alpha Sweep (1.0 -> 0.75 -> 0.50 -> 0.25 -> 0.0) ====================
    # Super pool = 18 GiB (288 x 64 MiB handles: holds all 196 gen handles + 92 handles = 5.75 GiB free)
    # Fine KV pool = 1 GiB (512 x 2 MiB pages)
    vmm = TriPoolVMM(
        device=0,
        super_pool_mib=18432,
        fine_kv_pool_mib=1024,
        super_handle_mib=64,
        transient_va_gib=16,
        kv_va_gib=16,
        allow_overflow=False,
    )
    bind_info = pager.bind_vmm(vmm, initial_alpha=1.0)
    report["bind_info"] = bind_info
    report["mem_after_vmm_bind"] = gpu_mem_info_mib(0)

    # Also test AR-only prefill while Generation Expert is 100% evicted (alpha=0.0)!
    pager.set_residency_alpha(0.0)
    ar_prefill_res_before = vmm.expert_resident_count(pager.expert_id)
    ar_ctx_main, _ = _build_turn0_context(pipeline, "Describe the alpine lake scene in detail.", device)
    ar_prefill_res_after = vmm.expert_resident_count(pager.expert_id)
    del ar_ctx_main
    report["ar_isolation_check"] = {
        "gen_resident_before_ar_prefill": ar_prefill_res_before,
        "gen_resident_after_ar_prefill": ar_prefill_res_after,
        "zero_gen_faults_in_ar_mode": ar_prefill_res_before == 0 and ar_prefill_res_after == 0,
    }

    alpha_results = []
    # Measure Step-1 overlap across alpha = 1.0, 0.75, 0.50, 0.25, 0.0 (512x512, 20 steps)
    for alpha in (1.0, 0.75, 0.50, 0.25, 0.0):
        res_before = pager.set_residency_alpha(alpha)
        st_before = vmm.stats_dict()

        # Sync BEFORE generation_mode so prefetch_expert_async overlaps with Step-1 compute cleanly!
        torch.cuda.synchronize()
        t_wall0 = time.perf_counter()
        with pager.generation_mode(prefetch_all=True):
            img, _ = _run_bagel_generate(
                pipeline, main_ctx, cfg_img_ctx, prompt, steps=steps, seed=52, sync_before=False
            )
        gen_ms = (time.perf_counter() - t_wall0) * 1000.0

        st_after = vmm.stats_dict()
        sha = pil_sha256(img)
        arr = np.asarray(img.convert("RGB"), dtype=np.int16)
        max_px_diff = int(np.abs(arr - ref_arr).max())
        streamed_mib = round((st_after["expert_h2d_bytes_total"] - st_before["expert_h2d_bytes_total"]) / MIB, 2)
        alpha_results.append(
            {
                "alpha": alpha,
                "resident_layers_before": res_before,
                "resident_layers_after": vmm.expert_resident_count(pager.expert_id),
                "missing_layers_streamed": pager.num_layers - res_before,
                "streamed_h2d_mib": streamed_mib,
                "gen_total_ms": round(gen_ms, 2),
                "delta_vs_ref_ms": round(gen_ms - ref_gen_ms, 2),
                "bit_exact": sha == ref_sha and max_px_diff == 0,
                "sha256": sha,
                "max_pixel_diff": max_px_diff,
                "transient_reclaimed": st_after["transient_mapped_handles_current"] == 0,
                "transient_peak_mib": round(
                    st_after["transient_mapped_handles_peak"] * st_after["super_handle_bytes"] / MIB, 2
                ),
                "expert_map_delta_ms": round((st_after["expert_map_us"] - st_before["expert_map_us"]) / 1000.0, 3),
                "expert_wait_stalls_delta": st_after["expert_wait_stalls"] - st_before["expert_wait_stalls"],
                "overflow_creates": st_after["overflow_creates"],
            }
        )
        del img

    # Compute exact overlap efficiency relative to alpha=1.0
    t_alpha1 = alpha_results[0]["gen_total_ms"]
    for item in alpha_results:
        item["morph_stall_vs_alpha1_ms"] = round(item["gen_total_ms"] - t_alpha1, 2)
        # Theoretical un-overlapped H2D time at measured 12.33 GB/s
        raw_h2d_ms = (item["streamed_h2d_mib"] * MIB / 1e9) / 12.33 * 1000.0
        item["theoretical_unoverlapped_h2d_ms"] = round(raw_h2d_ms, 2)
        item["hidden_by_overlap_ms"] = round(max(0.0, raw_h2d_ms - max(0.0, item["morph_stall_vs_alpha1_ms"])), 2)

    report["test1_alpha_sweep"] = alpha_results

    # Also measure high-res 1024x1024 Step-1 overlap (10 steps: ~4x larger DiT compute horizon!)
    # Warm up 1024x1024 Triton/cuDNN kernels once at alpha=1.0 so the sweep is jitter-free.
    pager.set_residency_alpha(1.0)
    with pager.generation_mode(prefetch_all=False):
        _h_warm, _ = _run_bagel_generate(
            pipeline, main_ctx, cfg_img_ctx, prompt, steps=2, seed=52, height=1024, width=1024, sync_before=True
        )
    del _h_warm
    hires_results = []
    hires_ref_sha = None
    for alpha in (1.0, 0.75, 0.50, 0.0):
        res_before = pager.set_residency_alpha(alpha)
        st_before = vmm.stats_dict()
        torch.cuda.synchronize()
        t_wall0 = time.perf_counter()
        with pager.generation_mode(prefetch_all=True):
            h_img, _ = _run_bagel_generate(
                pipeline,
                main_ctx,
                cfg_img_ctx,
                prompt,
                steps=10,
                seed=52,
                height=1024,
                width=1024,
                sync_before=False,
            )
        h_ms = (time.perf_counter() - t_wall0) * 1000.0
        st_after = vmm.stats_dict()
        h_sha = pil_sha256(h_img)
        if hires_ref_sha is None:
            hires_ref_sha = h_sha
        streamed_mib = round((st_after["expert_h2d_bytes_total"] - st_before["expert_h2d_bytes_total"]) / MIB, 2)
        hires_results.append(
            {
                "alpha": alpha,
                "resident_layers_before": res_before,
                "streamed_h2d_mib": streamed_mib,
                "gen_total_ms": round(h_ms, 2),
                "bit_exact_vs_alpha1": h_sha == hires_ref_sha,
                "expert_map_delta_ms": round((st_after["expert_map_us"] - st_before["expert_map_us"]) / 1000.0, 3),
                "transient_reclaimed": st_after["transient_mapped_handles_current"] == 0,
                "overflow_creates": st_after["overflow_creates"],
            }
        )
        del h_img
    t_hires1 = hires_results[0]["gen_total_ms"]
    for item in hires_results:
        item["morph_stall_vs_alpha1_ms"] = round(item["gen_total_ms"] - t_hires1, 2)
        raw_h2d_ms = (item["streamed_h2d_mib"] * MIB / 1e9) / 12.33 * 1000.0
        item["theoretical_unoverlapped_h2d_ms"] = round(raw_h2d_ms, 2)
        item["hidden_by_overlap_ms"] = round(max(0.0, raw_h2d_ms - max(0.0, item["morph_stall_vs_alpha1_ms"])), 2)
    report["test1b_1024_overlap_sweep"] = hires_results
    report["layer_hook_telemetry"] = {
        "layer_hook_calls": pager.layer_hook_calls,
        "hook_methods_triggered": sorted(pager.hook_methods_triggered),
    }

    # ==================== 3. Test 4 (Before Teardown): Persistent KV 2 MiB Pages + Elastic Carve ====================
    kv_slots = []
    # Allocate 540 x 2 MiB pages (= 1080 MiB > 1024 MiB initial fine pool -> forces 1 super->fine carve!)
    pager.set_residency_alpha(0.5)
    st_kv0 = vmm.stats_dict()
    for _ in range(540):
        kv_slots.append(vmm.kv_alloc_page())
    # Verify real GPU write/read on carved page
    t_last = vmm.kv_page_tensor(kv_slots[-1])
    t_last.fill_(0x4B)
    torch.cuda.synchronize()
    kv_readback_ok = bool(int(t_last[0].item()) == 0x4B and int(t_last[-1].item()) == 0x4B)
    del t_last
    st_kv_peak = vmm.stats_dict()
    for slot in kv_slots:
        vmm.kv_free_page(slot, torch.cuda.current_stream(device).cuda_stream)
    st_kv_after = vmm.stats_dict()
    report["test4_kv_arena_and_carve"] = {
        "allocated_2mb_pages": len(kv_slots),
        "super_to_fine_carves_triggered": st_kv_peak["super_to_fine_carves"] - st_kv0["super_to_fine_carves"],
        "kv_readback_verified": kv_readback_ok,
        "kv_pages_reclaimed": st_kv_after["kv_mapped_pages_current"] == 0,
        "overflow_creates": st_kv_after["overflow_creates"],
    }

    # Tear down first VMM cleanly before tight-pool pressure proof
    pager.unbind_vmm_keep_mirror()
    vmm.shutdown()
    gc.collect()
    torch.cuda.empty_cache()

    # ==================== 4. Test 2: Tight 14 GiB Pool + 7.0 GiB 1024x1024 b=2 VAE Spike (Zero-OOM Proof!) ====================
    # Super pool = 14 GiB (224 x 64 MiB handles, allow_overflow=False!):
    #   All 28 Gen Expert layers occupy 196 handles (12.25 GiB) -> only 28 handles (1.75 GiB) free!
    #   Firing 1024x1024 b=2 FLUX VAE decode (+7.0 GiB = ~112 handles) MUST auto-evict clean tail expert layers!
    vmm_tight = TriPoolVMM(
        device=0,
        super_pool_mib=14336,
        fine_kv_pool_mib=256,
        super_handle_mib=64,
        transient_va_gib=16,
        kv_va_gib=4,
        allow_overflow=False,
    )
    pager.bind_vmm(vmm_tight, initial_alpha=1.0)
    res_before_vae = vmm_tight.expert_resident_count(pager.expert_id)
    st_t0 = vmm_tight.stats_dict()

    torch.cuda.synchronize()
    t_vt0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        vae_vmm_out = pipeline.vae.decode(vae_lat_1024_b2)
    torch.cuda.synchronize()
    vae_vmm_ms = (time.perf_counter() - t_vt0) * 1000.0
    vae_vmm_sha = tensor_sha256(vae_vmm_out)
    del vae_vmm_out

    st_t1 = vmm_tight.stats_dict()
    res_after_vae = vmm_tight.expert_resident_count(pager.expert_id)

    # Now verify instant recovery: DiT generation immediately after auto-eviction streams back ONLY dropped tail layers!
    torch.cuda.synchronize()
    t_rec0 = time.perf_counter()
    with pager.generation_mode(prefetch_all=True):
        post_img, _ = _run_bagel_generate(
            pipeline, main_ctx, cfg_img_ctx, prompt, steps=steps, seed=52, sync_before=False
        )
    post_gen_ms = (time.perf_counter() - t_rec0) * 1000.0
    post_sha = pil_sha256(post_img)
    del post_img

    report["test2_tight_pool_7gib_vae_pressure"] = {
        "super_pool_mib": 14336,
        "allow_overflow": False,
        "resident_layers_before_vae": res_before_vae,
        "resident_layers_after_vae": res_after_vae,
        "auto_evicted_expert_layers": st_t1["expert_auto_evictions_for_transient"]
        - st_t0["expert_auto_evictions_for_transient"],
        "transient_peak_mib": round(
            st_t1["transient_mapped_handles_peak"] * st_t1["super_handle_bytes"] / MIB, 2
        ),
        "transient_all_reclaimed": st_t1["transient_mapped_handles_current"] == 0,
        "overflow_creates": st_t1["overflow_creates"],
        "vae_1024_b2_vmm_ms": round(vae_vmm_ms, 2),
        "vae_1024_b2_ref_ms": round(vae_ref_ms, 2),
        "vae_1024_b2_bit_exact": vae_vmm_sha == vae_ref_sha,
        "vae_1024_b2_vmm_sha256": vae_vmm_sha,
        "post_eviction_dit_recovery_ms": round(post_gen_ms, 2),
        "post_eviction_dit_bit_exact": post_sha == ref_sha,
        "final_stats": vmm_tight.stats_dict(),
    }

    pager.close()
    vmm_tight.shutdown()
    del vae_lat_1024_b2, main_ctx, cfg_img_ctx, pipeline
    gc.collect()
    torch.cuda.empty_cache()

    all_pass = (
        report["ar_isolation_check"]["zero_gen_faults_in_ar_mode"]
        and report["layer_hook_telemetry"]["layer_hook_calls"] > 0
        and all(x["bit_exact"] and x["transient_reclaimed"] and x["overflow_creates"] == 0 for x in alpha_results)
        and all(x["bit_exact_vs_alpha1"] and x["transient_reclaimed"] and x["overflow_creates"] == 0 for x in hires_results)
        and report["test4_kv_arena_and_carve"]["kv_readback_verified"]
        and report["test4_kv_arena_and_carve"]["kv_pages_reclaimed"]
        and report["test4_kv_arena_and_carve"]["super_to_fine_carves_triggered"] >= 1
        and report["test2_tight_pool_7gib_vae_pressure"]["auto_evicted_expert_layers"] > 0
        and report["test2_tight_pool_7gib_vae_pressure"]["transient_all_reclaimed"]
        and report["test2_tight_pool_7gib_vae_pressure"]["overflow_creates"] == 0
        and report["test2_tight_pool_7gib_vae_pressure"]["vae_1024_b2_bit_exact"]
        and report["test2_tight_pool_7gib_vae_pressure"]["post_eviction_dit_bit_exact"]
    )
    report["status"] = "ok" if all_pass else "check_failed"

    out_path = output_dir / "module1-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    print(
        "MODULE1_SUMMARY="
        + json.dumps(
            {
                "status": report["status"],
                "ref_gen_ms": report["baseline_reference"]["ref_gen_ms"],
                "alpha_sweep": [
                    {
                        "alpha": x["alpha"],
                        "gen_ms": x["gen_total_ms"],
                        "stall_ms": x["morph_stall_vs_alpha1_ms"],
                        "raw_h2d_ms": x["theoretical_unoverlapped_h2d_ms"],
                        "bit_exact": x["bit_exact"],
                    }
                    for x in alpha_results
                ],
                "tight_7gib_vae": {
                    "auto_evicted_layers": report["test2_tight_pool_7gib_vae_pressure"]["auto_evicted_expert_layers"],
                    "peak_mib": report["test2_tight_pool_7gib_vae_pressure"]["transient_peak_mib"],
                    "vae_bit_exact": report["test2_tight_pool_7gib_vae_pressure"]["vae_1024_b2_bit_exact"],
                    "post_dit_bit_exact": report["test2_tight_pool_7gib_vae_pressure"]["post_eviction_dit_bit_exact"],
                    "overflow": report["test2_tight_pool_7gib_vae_pressure"]["overflow_creates"],
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MorphServe Module 1 (TriPool-VMM + BAGEL-7B-MoT) validation")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args(argv)
    rep = run_validation(args.model_dir, args.output_dir, steps=args.steps)
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
