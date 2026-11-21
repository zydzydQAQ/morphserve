"""Phase-1 Module 3 live A100 validation: Shared-Prefix Triton CFG Attention + LECP CoW + GPU In-Place Loopback."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import io
import json
import math
from pathlib import Path
import time

from morphserve.bench.validate_module1_bagel import load_bagel_pipeline
from morphserve.kernels import (
    GpuInPlaceLoopback,
    SharedPrefixCache,
    cow_fork_context,
    decode_latent_to_u8_gpu,
    install_shared_prefix_bagel_hooks,
    prepare_vae_images_gpu,
    prepare_vit_images_gpu,
    rebind_cow_prefix,
    shared_prefix_cfg_attention,
    shared_prefix_cfg_attention_ref,
)

MIB = 1024**2


def _compute_ssim_rgb(img_a, img_b) -> dict[str, float]:
    import numpy as np

    a = np.asarray(img_a.convert("RGB"), dtype=np.float64)
    b = np.asarray(img_b.convert("RGB"), dtype=np.float64)
    diff = np.abs(a - b)
    mae = float(diff.mean())
    max_diff = float(diff.max())
    mse = float(((a - b) ** 2).mean())
    psnr = 99.99 if mse == 0.0 else float(10.0 * math.log10((255.0**2) / mse))

    # Standard global + 8x8 windowed SSIM on luminance/RGB
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    H, W, C = a.shape
    bh, bw = H // 8, W // 8
    a_blk = a[: bh * 8, : bw * 8].reshape(bh, 8, bw, 8, C).transpose(0, 2, 4, 1, 3).reshape(-1, 64)
    b_blk = b[: bh * 8, : bw * 8].reshape(bh, 8, bw, 8, C).transpose(0, 2, 4, 1, 3).reshape(-1, 64)
    mu_a = a_blk.mean(axis=1)
    mu_b = b_blk.mean(axis=1)
    var_a = a_blk.var(axis=1)
    var_b = b_blk.var(axis=1)
    cov = ((a_blk - mu_a[:, None]) * (b_blk - mu_b[:, None])).mean(axis=1)
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return {
        "ssim": round(float(ssim_map.mean()), 6),
        "psnr_db": round(psnr, 2),
        "mae_u8": round(mae, 4),
        "max_abs_diff_u8": int(max_diff),
    }


def _left_pad_stack_stock(tensors):
    """Exact replica of stock vllm_omni `left_pad_stack` (materializes padded 4D KV + dense mask)."""
    import torch

    max_len = max(t.shape[0] for t in tensors)
    if all(t.shape[0] == max_len for t in tensors):
        return torch.stack(tensors, dim=0), None
    bsz = len(tensors)
    out = tensors[0].new_zeros((bsz, max_len, *tensors[0].shape[1:]))
    mask = torch.zeros((bsz, max_len), dtype=torch.bool, device=tensors[0].device)
    for i, t in enumerate(tensors):
        ln = t.shape[0]
        out[i, max_len - ln :] = t
        mask[i, max_len - ln :] = True
    return out, mask


def bench_triton_cfg_attention_micro(device, attn_noncausal=None) -> list[dict]:
    """Microbenchmark + exactness check of Triton Single-HBM-Read CFG Attention vs real vLLM-Omni FlashAttn left_pad_stack."""
    import torch
    import torch.nn.functional as F
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata as DiffusionAttentionMetadata

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    h_q, h_kv, d = 28, 4, 128
    sm_scale = 1.0 / math.sqrt(d)

    results = []
    # Sweep realistic BAGEL-7B-MoT multi-turn shapes:
    #   Branch 0 (main): L_shared + suffix + Q local
    #   Branch 1 (cfg_text): L_shared + 0 suffix + Q local
    #   Branch 2 (cfg_img): text-only + Q local
    configs = [
        ("512px_turn1_2k", 1026, 2048, 24, 0, 24),
        ("512px_turn2_4k", 1026, 4096, 24, 0, 24),
        ("512px_turn4_8k", 1026, 8192, 32, 0, 32),
        ("512px_turn8_16k", 1026, 16384, 32, 0, 32),
        ("1024px_turn2_8k", 4098, 8192, 32, 0, 32),
    ]

    with torch.inference_mode():
        for label, q_len, l_shared, l_suf0, l_suf1, l_img in configs:
            q_list = [torch.randn((q_len, h_q, d), dtype=torch.bfloat16, device=device) * 0.5 for _ in range(3)]
            k_loc = [torch.randn((q_len, h_kv, d), dtype=torch.bfloat16, device=device) * 0.5 for _ in range(3)]
            v_loc = [torch.randn((q_len, h_kv, d), dtype=torch.bfloat16, device=device) * 0.5 for _ in range(3)]

            k_pre = torch.randn((l_shared, h_kv, d), dtype=torch.bfloat16, device=device) * 0.5
            v_pre = torch.randn((l_shared, h_kv, d), dtype=torch.bfloat16, device=device) * 0.5
            k_suf0 = torch.randn((l_suf0, h_kv, d), dtype=torch.bfloat16, device=device) * 0.5
            v_suf0 = torch.randn((l_suf0, h_kv, d), dtype=torch.bfloat16, device=device) * 0.5
            k_suf1 = None if l_suf1 == 0 else torch.randn((l_suf1, h_kv, d), dtype=torch.bfloat16, device=device) * 0.5
            v_suf1 = None if l_suf1 == 0 else torch.randn((l_suf1, h_kv, d), dtype=torch.bfloat16, device=device) * 0.5
            k_img = torch.randn((l_img, h_kv, d), dtype=torch.bfloat16, device=device) * 0.5
            v_img = torch.randn((l_img, h_kv, d), dtype=torch.bfloat16, device=device) * 0.5

            k_suf_list = [k_suf0, k_suf1, k_img]
            v_suf_list = [v_suf0, v_suf1, v_img]

            # Pre-built stock branch caches (simulating NaiveCache copies held by main, cfg_text, cfg_img)
            ck0 = torch.cat([k_pre, k_suf0], dim=0)
            cv0 = torch.cat([v_pre, v_suf0], dim=0)
            ck1 = k_pre.clone() if k_suf1 is None else torch.cat([k_pre, k_suf1], dim=0)
            cv1 = v_pre.clone() if v_suf1 is None else torch.cat([v_pre, v_suf1], dim=0)
            ck2, cv2 = k_img, v_img

            def run_stock_layer():
                q_4d = torch.stack(q_list, dim=0)  # (3, Q, H_q, D)
                kb = [
                    torch.cat([ck0, k_loc[0]], dim=0),
                    torch.cat([ck1, k_loc[1]], dim=0),
                    torch.cat([ck2, k_loc[2]], dim=0),
                ]
                vb = [
                    torch.cat([cv0, v_loc[0]], dim=0),
                    torch.cat([cv1, v_loc[1]], dim=0),
                    torch.cat([cv2, v_loc[2]], dim=0),
                ]
                k_4d, mask = _left_pad_stack_stock(kb)
                v_4d, _ = _left_pad_stack_stock(vb)
                if attn_noncausal is not None:
                    meta = DiffusionAttentionMetadata(attn_mask=mask) if mask is not None else None
                    return attn_noncausal(q_4d, k_4d, v_4d, meta)
                # Fallback if attn_noncausal not supplied: run unmasked per-branch FlashAttn/SDPA
                outs = []
                for i in range(3):
                    ob = F.scaled_dot_product_attention(
                        q_list[i].unsqueeze(0).permute(0, 2, 1, 3),
                        kb[i].unsqueeze(0).permute(0, 2, 1, 3),
                        vb[i].unsqueeze(0).permute(0, 2, 1, 3),
                        is_causal=False,
                        scale=sm_scale,
                        enable_gqa=True,
                    )
                    outs.append(ob.permute(0, 2, 1, 3).squeeze(0))
                return torch.stack(outs, dim=0)

            def run_morphserve_layer():
                return shared_prefix_cfg_attention(
                    q_list=q_list,
                    k_loc_list=k_loc,
                    v_loc_list=v_loc,
                    k_prefix=k_pre,
                    v_prefix=v_pre,
                    shared_branch_ids=[0, 1],
                    k_suffix_list=k_suf_list,
                    v_suffix_list=v_suf_list,
                    sm_scale=sm_scale,
                    attn_noncausal=attn_noncausal,
                )

            # Exactness vs FP32 reference
            ref_fp32 = shared_prefix_cfg_attention_ref(
                q_list, k_loc, v_loc, k_pre, v_pre, [0, 1], k_suf_list, v_suf_list, sm_scale=sm_scale, fp32_accum=True
            )
            out_ms = run_morphserve_layer()
            out_stock = run_stock_layer()

            max_err = 0.0
            mean_err = 0.0
            min_cos = 1.0
            stock_max_err = 0.0
            for b in range(3):
                diff = (out_ms[b].float() - ref_fp32[b].float()).abs()
                max_err = max(max_err, float(diff.max().item()))
                mean_err += float(diff.mean().item()) / 3.0
                cos = float(
                    F.cosine_similarity(out_ms[b].float().reshape(-1), ref_fp32[b].float().reshape(-1), dim=0).item()
                )
                min_cos = min(min_cos, cos)
                sdiff = (out_stock[b].float() - ref_fp32[b].float()).abs()
                stock_max_err = max(stock_max_err, float(sdiff.max().item()))

            # Warmup (triggers Triton autotune on first call!)
            for _ in range(5):
                _ = run_stock_layer()
                _ = run_morphserve_layer()
            torch.cuda.synchronize()

            iters = 20
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
            for _ in range(iters):
                _ = run_stock_layer()
            ev1.record()
            torch.cuda.synchronize()
            stock_ms = ev0.elapsed_time(ev1) / iters

            ev0.record()
            for _ in range(iters):
                _ = run_morphserve_layer()
            ev1.record()
            torch.cuda.synchronize()
            ms_ms = ev0.elapsed_time(ev1) / iters

            stock_kv_tokens_read = 3 * (l_shared + l_suf0 + q_len)
            ms_kv_tokens_read = l_shared + l_suf0 + l_suf1 + l_img + 3 * q_len
            results.append(
                {
                    "case": label,
                    "q_len": q_len,
                    "l_shared": l_shared,
                    "stock_ms": round(stock_ms, 3),
                    "morphserve_ms": round(ms_ms, 3),
                    "speedup": round(stock_ms / max(ms_ms, 1e-6), 2),
                    "kv_hbm_read_reduction_x": round(stock_kv_tokens_read / ms_kv_tokens_read, 2),
                    "max_abs_err_vs_fp32": round(max_err, 5),
                    "stock_max_abs_err_vs_fp32": round(stock_max_err, 5),
                    "mean_abs_err_vs_fp32": round(mean_err, 6),
                    "min_cosine_sim": round(min_cos, 7),
                    "exact_pass": max_err <= 0.02 and mean_err <= 0.002 and min_cos >= 0.9999,
                }
            )
            del (
                q_list,
                k_loc,
                v_loc,
                k_pre,
                v_pre,
                k_suf0,
                v_suf0,
                k_img,
                v_img,
                ck0,
                cv0,
                ck1,
                cv1,
                ref_fp32,
                out_ms,
                out_stock,
            )
            torch.cuda.empty_cache()

    return results


def _run_bagel_generate_3branch(
    pipeline,
    main_ctx,
    cfg_text_ctx,
    cfg_img_ctx,
    prompt: str,
    steps: int = 20,
    seed: int = 52,
    height: int = 512,
    width: int = 512,
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
    if cfg_text_ctx is not None and cfg_text_ctx["past_key_values"].key_cache[0] is not None:
        sampling.cfg_text_past_key_values = cfg_text_ctx["past_key_values"]
        sampling.cfg_text_kv_metadata = {"ropes": cfg_text_ctx["ropes"], "image_shape": [height, width]}
    if cfg_img_ctx is not None and cfg_img_ctx["past_key_values"].key_cache[0] is not None:
        sampling.cfg_img_past_key_values = cfg_img_ctx["past_key_values"]
        sampling.cfg_img_kv_metadata = {"ropes": cfg_img_ctx["ropes"], "image_shape": [height, width]}

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = pipeline._forward_single({"prompt": prompt, "modalities": ["image"]}, sampling)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return out.output["payload"]["image"], elapsed_ms


def _update_text_ctx(pipeline, ctx: dict, prompt: str) -> None:
    import torch

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


def _update_image_ctx_cpu_pil(pipeline, ctx: dict, pil_img) -> float:
    """Stock CPU PIL/PNG round-trip + VAE/ViT KV update; returns CPU loopback overhead ms."""
    import numpy as np
    from PIL import Image
    import torch

    t_cpu0 = time.perf_counter()
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    reloaded = Image.open(buf).convert("RGB")

    def vae_transforms(img):
        arr = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
        return arr.permute(2, 0, 1)

    def vit_transforms(img):
        return pipeline.image_processor(images=img, return_tensors="pt").pixel_values[0]

    vae_inp, new_lens, new_rope = pipeline.bagel.prepare_vae_images(
        curr_kvlens=ctx["kv_lens"],
        curr_rope=ctx["ropes"],
        images=[reloaded],
        transforms=vae_transforms,
        new_token_ids=pipeline.new_token_ids,
    )
    vae_inp = {k: (v.to(pipeline.device) if torch.is_tensor(v) else v) for k, v in vae_inp.items()}
    vit_inp, new_lens_vit, new_rope_vit = pipeline.bagel.prepare_vit_images(
        curr_kvlens=new_lens,
        curr_rope=new_rope,
        images=[reloaded],
        transforms=vit_transforms,
        new_token_ids=pipeline.new_token_ids,
    )
    vit_inp = {k: (v.to(pipeline.device) if torch.is_tensor(v) else v) for k, v in vit_inp.items()}
    cpu_prep_ms = (time.perf_counter() - t_cpu0) * 1000.0

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ctx["past_key_values"] = pipeline.bagel.forward_cache_update_vae(
            pipeline.vae, ctx["past_key_values"], **vae_inp
        )
        ctx["kv_lens"] = new_lens
        ctx["ropes"] = new_rope
        ctx["past_key_values"] = pipeline.bagel.forward_cache_update_vit(ctx["past_key_values"], **vit_inp)
        ctx["kv_lens"] = new_lens_vit
        ctx["ropes"] = new_rope_vit
    return cpu_prep_ms


def _generate_ar_text(pipeline, ctx: dict, max_tokens: int = 32) -> str:
    import torch

    ctx_copy = cow_fork_context(ctx)
    start_inp = pipeline.bagel.prepare_start_tokens(ctx_copy["kv_lens"], ctx_copy["ropes"], pipeline.new_token_ids)
    start_inp = {k: (v.to(pipeline.device) if torch.is_tensor(v) else v) for k, v in start_inp.items()}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        tok_ids = pipeline.bagel.generate_text(
            past_key_values=ctx_copy["past_key_values"],
            max_length=max_tokens,
            do_sample=False,
            temperature=0.0,
            end_token_id=pipeline.new_token_ids["eos_token_id"],
            **start_inp,
        )
    return pipeline.tokenizer.decode(tok_ids[:, 0].tolist())


def _kv_cache_unique_bytes(caches: list) -> int:
    seen_ptrs: set[int] = set()
    total = 0
    for c in caches:
        for l in range(c.num_layers):
            for t in (c.key_cache[l], c.value_cache[l]):
                if t is None:
                    continue
                ptr = int(t.data_ptr())
                if ptr not in seen_ptrs:
                    seen_ptrs.add(ptr)
                    total += int(t.numel() * t.element_size())
    return total


def run_validation(model_dir: Path, output_dir: Path, steps: int = 20) -> dict:
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache

    device = torch.device("cuda:0")
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "schema": "morphserve.phase1.module3.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_dir": str(model_dir),
        "steps": steps,
    }

    # ==================== Load Real BAGEL-7B-MoT ====================
    t_load0 = time.perf_counter()
    pipeline = load_bagel_pipeline(model_dir, device_index=0)
    report["load_seconds"] = round(time.perf_counter() - t_load0, 2)
    num_layers = pipeline.bagel.config.llm_config.num_hidden_layers
    loopback = GpuInPlaceLoopback(pipeline)
    cfg_patch = install_shared_prefix_bagel_hooks(pipeline, min_shared_tokens=32)

    # ==================== Test 1: Triton Shared-Prefix CFG Attention Microbenchmark ====================
    # Benchmark directly against BAGEL's real FlashAttention-2 DiffusionAttention (`attn_noncausal`)!
    attn_noncausal = pipeline.bagel.language_model.model.layers[0].self_attn.attn_noncausal
    micro_res = bench_triton_cfg_attention_micro(device, attn_noncausal=attn_noncausal)
    report["test1_triton_cfg_microbench"] = micro_res
    gc.collect()
    torch.cuda.empty_cache()

    # ==================== Test 2: GPU In-Place Loopback vs CPU PIL/PNG Loopback ====================
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    # Use a real latent shape matching 512x512 BAGEL output (`1024 x 64`)
    lat_512 = torch.randn((1024, 64), dtype=torch.bfloat16, device=device) * 0.5
    image_shape = (512, 512)

    # Warmup VAE decode
    with torch.inference_mode():
        _ = decode_latent_to_u8_gpu(pipeline.bagel, pipeline.vae, lat_512, image_shape)
    torch.cuda.synchronize()

    # CPU PIL/PNG reference path
    torch.cuda.synchronize()
    t_cpu_full0 = time.perf_counter()
    with torch.inference_mode():
        pil_ref = pipeline._decode_image_from_latent(pipeline.bagel, pipeline.vae, lat_512, image_shape)
    buf = io.BytesIO()
    pil_ref.save(buf, format="PNG")
    buf.seek(0)
    pil_reloaded = Image.open(buf).convert("RGB")
    vae_cpu_t = (torch.from_numpy(np.array(pil_reloaded)).float() / 127.5 - 1.0).permute(2, 0, 1).to(device)
    vit_cpu_inp, _, _ = pipeline.bagel.prepare_vit_images(
        [0], [0], [pil_reloaded], lambda im: pipeline.image_processor(images=im, return_tensors="pt").pixel_values[0], pipeline.new_token_ids
    )
    vit_cpu_tokens = vit_cpu_inp["packed_vit_tokens"].to(device)
    torch.cuda.synchronize()
    cpu_loopback_ms = (time.perf_counter() - t_cpu_full0) * 1000.0

    # GPU In-Place Loopback path (measure steady-state over 5 runs)
    gpu_times = []
    img_u8_gpu = None
    vae_gpu_inp = None
    vit_gpu_inp = None
    for _ in range(5):
        torch.cuda.synchronize()
        t_g0 = time.perf_counter()
        with torch.inference_mode():
            img_u8_gpu = decode_latent_to_u8_gpu(pipeline.bagel, pipeline.vae, lat_512, image_shape)
            vae_gpu_inp, _, _ = prepare_vae_images_gpu(pipeline.bagel, [img_u8_gpu], [0], [0], pipeline.new_token_ids)
            vit_gpu_inp, _, _ = prepare_vit_images_gpu(
                pipeline.bagel, pipeline.image_processor, [img_u8_gpu], [0], [0], pipeline.new_token_ids
            )
        torch.cuda.synchronize()
        gpu_times.append((time.perf_counter() - t_g0) * 1000.0)

    # Pure non-VAE prep overhead comparison (PNG+PIL+CPU->GPU vs GPU u8->VAE/ViT tensors)
    torch.cuda.synchronize()
    t_prep_g0 = time.perf_counter()
    with torch.inference_mode():
        vae_gpu_inp, _, _ = prepare_vae_images_gpu(pipeline.bagel, [img_u8_gpu], [0], [0], pipeline.new_token_ids)
        vit_gpu_inp, _, _ = prepare_vit_images_gpu(
            pipeline.bagel, pipeline.image_processor, [img_u8_gpu], [0], [0], pipeline.new_token_ids
        )
    torch.cuda.synchronize()
    gpu_prep_only_ms = (time.perf_counter() - t_prep_g0) * 1000.0

    t_prep_c0 = time.perf_counter()
    buf2 = io.BytesIO()
    pil_ref.save(buf2, format="PNG")
    buf2.seek(0)
    pil_r2 = Image.open(buf2).convert("RGB")
    _ = (torch.from_numpy(np.array(pil_r2)).float() / 127.5 - 1.0).permute(2, 0, 1).to(device)
    _ = pipeline.bagel.prepare_vit_images(
        [0], [0], [pil_r2], lambda im: pipeline.image_processor(images=im, return_tensors="pt").pixel_values[0], pipeline.new_token_ids
    )
    torch.cuda.synchronize()
    cpu_prep_only_ms = (time.perf_counter() - t_prep_c0) * 1000.0

    vae_diff_max = float((vae_gpu_inp["padded_images"][0] - vae_cpu_t).abs().max().item())
    vit_gpu_tokens = vit_gpu_inp["packed_vit_tokens"]
    vit_diff_max = float((vit_gpu_tokens - vit_cpu_tokens).abs().max().item())
    vit_cos = float(
        F.cosine_similarity(vit_gpu_tokens.float().reshape(-1), vit_cpu_tokens.float().reshape(-1), dim=0).item()
    )

    # Compare AR text decode after CPU loopback vs GPU loopback
    ctx_ar_cpu = {"kv_lens": [0], "ropes": [0], "past_key_values": NaiveCache(num_layers)}
    _update_image_ctx_cpu_pil(pipeline, ctx_ar_cpu, pil_ref)
    _update_text_ctx(pipeline, ctx_ar_cpu, "Describe the colors and textures in this image concisely.")
    ar_text_cpu = _generate_ar_text(pipeline, ctx_ar_cpu, max_tokens=24)

    ctx_ar_gpu = {"kv_lens": [0], "ropes": [0], "past_key_values": NaiveCache(num_layers)}
    loopback.update_context_from_u8_gpu(ctx_ar_gpu, img_u8_gpu)
    _update_text_ctx(pipeline, ctx_ar_gpu, "Describe the colors and textures in this image concisely.")
    ar_text_gpu = _generate_ar_text(pipeline, ctx_ar_gpu, max_tokens=24)

    ip_dict = pipeline.image_processor.to_dict() if hasattr(pipeline.image_processor, "to_dict") else {}
    report["test2_gpu_inplace_loopback"] = {
        "image_processor_class": type(pipeline.image_processor).__name__,
        "image_processor_cfg": {
            "do_resize": ip_dict.get("do_resize"),
            "size": ip_dict.get("size"),
            "do_rescale": ip_dict.get("do_rescale"),
            "do_normalize": ip_dict.get("do_normalize"),
            "image_mean": ip_dict.get("image_mean"),
            "image_std": ip_dict.get("image_std"),
        },
        "cpu_decode_plus_prep_ms": round(cpu_loopback_ms, 2),
        "gpu_decode_plus_prep_p50_ms": round(float(np.median(gpu_times)), 2),
        "cpu_png_pil_prep_only_ms": round(cpu_prep_only_ms, 2),
        "gpu_inplace_prep_only_ms": round(gpu_prep_only_ms, 3),
        "prep_speedup_x": round(cpu_prep_only_ms / max(gpu_prep_only_ms, 1e-4), 1),
        "vae_input_max_abs_diff": vae_diff_max,
        "vae_input_bit_exact": vae_diff_max == 0.0,
        "vit_tokens_max_abs_diff": round(vit_diff_max, 6),
        "vit_tokens_cosine_sim": round(vit_cos, 7),
        "ar_text_cpu": ar_text_cpu,
        "ar_text_gpu": ar_text_gpu,
        "ar_text_exact_match": ar_text_cpu == ar_text_gpu,
    }
    del ctx_ar_cpu, ctx_ar_gpu, lat_512, img_u8_gpu, vae_gpu_inp, vit_gpu_inp, vae_cpu_t, vit_cpu_tokens
    gc.collect()
    torch.cuda.empty_cache()

    # ==================== Test 3: End-to-End 4-Turn Interleaved Visual-CoT (Stock vs MorphServe M3) ====================
    prompts = [
        "A serene wooden cabin beside an alpine lake at golden hour, realistic photography.",
        "Add warm glowing lights inside the cabin windows and a wooden rowboat at the dock.",
        "Change the season to crisp autumn with golden maple trees reflecting in the lake.",
        "Make it twilight with a starry night sky and soft mist rising over the water.",
    ]

    # Run identical 4-turn editing session under:
    #   (A) Stock vLLM-Omni: deepcopy(main) -> cfg_text + NaiveCache.merge + left_pad_stack + CPU PNG loopback
    #   (B) MorphServe Module 3: CoW fork + SharedPrefixCache + Single-Read Triton CFG Attention + GPU In-Place Loopback
    def run_4turn_session(use_morphserve: bool) -> tuple[list[dict], list]:
        SharedPrefixCache.clear_memo()
        main_ctx = {"kv_lens": [0], "ropes": [0], "past_key_values": NaiveCache(num_layers)}
        turn_logs = []
        imgs = []
        for turn_idx, p in enumerate(prompts):
            loopback_ms = 0.0
            if turn_idx > 0:
                if use_morphserve:
                    # Simulate GPU tensor already on GPU after previous turn's VAE decode!
                    prev_u8 = torch.from_numpy(np.asarray(imgs[-1].convert("RGB"), dtype=np.uint8)).permute(2, 0, 1).to(device)
                    torch.cuda.synchronize()
                    t_lb0 = time.perf_counter()
                    loopback.update_context_from_u8_gpu(main_ctx, prev_u8)
                    torch.cuda.synchronize()
                    loopback_ms = (time.perf_counter() - t_lb0) * 1000.0
                else:
                    torch.cuda.synchronize()
                    t_lb0 = time.perf_counter()
                    _ = _update_image_ctx_cpu_pil(pipeline, main_ctx, imgs[-1])
                    torch.cuda.synchronize()
                    loopback_ms = (time.perf_counter() - t_lb0) * 1000.0

            # Fork cfg_text BEFORE appending this turn's prompt
            t_fork0 = time.perf_counter()
            if use_morphserve:
                cfg_text_ctx = cow_fork_context(main_ctx)
            else:
                cfg_text_ctx = deepcopy(main_ctx)
            torch.cuda.synchronize()
            fork_ms = (time.perf_counter() - t_fork0) * 1000.0

            # Append this turn's prompt to main
            _update_text_ctx(pipeline, main_ctx, p)
            if use_morphserve:
                # Zero-copy CoW rebind: cfg_text views main[:L_shared], freeing old pre-append KV immediately!
                rebind_cow_prefix(main_ctx["past_key_values"], cfg_text_ctx["past_key_values"])

            # cfg_img: text-only branch (no image history, matches pipeline_bagel.py:866)
            cfg_img_ctx = {"kv_lens": [0], "ropes": [0], "past_key_values": NaiveCache(num_layers)}
            _update_text_ctx(pipeline, cfg_img_ctx, p)

            kv_unique_mib = round(
                _kv_cache_unique_bytes(
                    [main_ctx["past_key_values"], cfg_text_ctx["past_key_values"], cfg_img_ctx["past_key_values"]]
                )
                / MIB,
                2,
            )

            if use_morphserve:
                with cfg_patch.active(reset_stats=True):
                    img, gen_ms = _run_bagel_generate_3branch(
                        pipeline, main_ctx, cfg_text_ctx, cfg_img_ctx, p, steps=steps, seed=52 + turn_idx
                    )
                shared_len = cfg_patch.last_shared_prefix_len
                shared_calls = cfg_patch.shared_attn_calls
                naive_toks = cfg_patch.last_naive_kv_tokens
                padded_toks = cfg_patch.last_padded_naive_kv_tokens
                uniq_toks = cfg_patch.last_unique_kv_tokens
            else:
                cfg_patch.enabled = False
                img, gen_ms = _run_bagel_generate_3branch(
                    pipeline, main_ctx, cfg_text_ctx, cfg_img_ctx, p, steps=steps, seed=52 + turn_idx
                )
                shared_len = 0
                shared_calls = 0
                l0 = main_ctx["past_key_values"].seq_lens
                l1 = cfg_text_ctx["past_key_values"].seq_lens
                l2 = cfg_img_ctx["past_key_values"].seq_lens
                naive_toks = l0 + l1 + l2
                padded_toks = max(l0, l1, l2) * 3
                uniq_toks = naive_toks

            imgs.append(img)
            turn_logs.append(
                {
                    "turn": turn_idx,
                    "main_seq_len": main_ctx["past_key_values"].seq_lens,
                    "cfg_text_seq_len": cfg_text_ctx["past_key_values"].seq_lens,
                    "cfg_img_seq_len": cfg_img_ctx["past_key_values"].seq_lens,
                    "kv_unique_hbm_mib": kv_unique_mib,
                    "fork_copy_ms": round(fork_ms, 2),
                    "loopback_total_ms": round(loopback_ms, 2),
                    "gen_ms": round(gen_ms, 2),
                    "shared_prefix_len": shared_len,
                    "shared_triton_attn_calls": shared_calls,
                    "unique_kv_tokens": uniq_toks,
                    "naive_kv_tokens": naive_toks,
                    "padded_naive_kv_tokens": padded_toks,
                }
            )
            del cfg_text_ctx, cfg_img_ctx
            torch.cuda.empty_cache()
        del main_ctx
        gc.collect()
        torch.cuda.empty_cache()
        return turn_logs, imgs

    # Warmup once on Turn 0 so stock & morphserve both measure steady-state GPU clocks
    _w_ctx = {"kv_lens": [0], "ropes": [0], "past_key_values": NaiveCache(num_layers)}
    _update_text_ctx(pipeline, _w_ctx, prompts[0])
    _, _ = _run_bagel_generate_3branch(pipeline, _w_ctx, deepcopy(_w_ctx), deepcopy(_w_ctx), prompts[0], steps=4, seed=1)
    del _w_ctx
    torch.cuda.empty_cache()

    stock_turns, stock_imgs = run_4turn_session(use_morphserve=False)
    ms_turns, ms_imgs = run_4turn_session(use_morphserve=True)

    turn_comparisons = []
    for st, mt, simg, mimg in zip(stock_turns, ms_turns, stock_imgs, ms_imgs):
        qual = _compute_ssim_rgb(simg, mimg)
        turn_comparisons.append(
            {
                "turn": st["turn"],
                "main_seq_len": st["main_seq_len"],
                "shared_prefix_len": mt["shared_prefix_len"],
                "stock_kv_hbm_mib": st["kv_unique_hbm_mib"],
                "morphserve_kv_hbm_mib": mt["kv_unique_hbm_mib"],
                "kv_hbm_saved_pct": round(
                    100.0 * (st["kv_unique_hbm_mib"] - mt["kv_unique_hbm_mib"]) / max(st["kv_unique_hbm_mib"], 1e-6),
                    1,
                ),
                "stock_fork_ms": st["fork_copy_ms"],
                "morphserve_fork_ms": mt["fork_copy_ms"],
                "stock_loopback_ms": st["loopback_total_ms"],
                "morphserve_loopback_ms": mt["loopback_total_ms"],
                "stock_gen_ms": st["gen_ms"],
                "morphserve_gen_ms": mt["gen_ms"],
                "gen_speedup_x": round(st["gen_ms"] / max(mt["gen_ms"], 1e-6), 3),
                "shared_triton_attn_calls": mt["shared_triton_attn_calls"],
                **qual,
            }
        )

    report["test3_4turn_interleaved_e2e"] = {
        "turns": turn_comparisons,
        "total_stock_gen_ms": round(sum(t["stock_gen_ms"] for t in turn_comparisons), 2),
        "total_morphserve_gen_ms": round(sum(t["morphserve_gen_ms"] for t in turn_comparisons), 2),
        "total_stock_loopback_ms": round(sum(t["stock_loopback_ms"] for t in turn_comparisons), 2),
        "total_morphserve_loopback_ms": round(sum(t["morphserve_loopback_ms"] for t in turn_comparisons), 2),
        "min_ssim_across_4turns": min(t["ssim"] for t in turn_comparisons),
    }

    cfg_patch.uninstall()
    del stock_imgs, ms_imgs, pipeline
    gc.collect()
    torch.cuda.empty_cache()

    all_pass = (
        all(x["exact_pass"] for x in micro_res)
        and report["test2_gpu_inplace_loopback"]["vae_input_bit_exact"]
        and report["test2_gpu_inplace_loopback"]["vit_tokens_cosine_sim"] >= 0.9999
        and report["test2_gpu_inplace_loopback"]["ar_text_exact_match"]
        and all(t["shared_triton_attn_calls"] > 0 for t in turn_comparisons if t["turn"] >= 1)
        and all(t["ssim"] >= 0.999 for t in turn_comparisons)
        and all(t["kv_hbm_saved_pct"] >= 40.0 for t in turn_comparisons if t["turn"] >= 1)
    )
    report["status"] = "ok" if all_pass else "check_failed"

    out_path = output_dir / "module3-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    print(
        "MODULE3_SUMMARY="
        + json.dumps(
            {
                "status": report["status"],
                "micro": [
                    {
                        "case": x["case"],
                        "stock_ms": x["stock_ms"],
                        "ms_ms": x["morphserve_ms"],
                        "speedup": x["speedup"],
                        "max_err": x["max_abs_err_vs_fp32"],
                    }
                    for x in micro_res
                ],
                "loopback": {
                    "cpu_prep_ms": report["test2_gpu_inplace_loopback"]["cpu_png_pil_prep_only_ms"],
                    "gpu_prep_ms": report["test2_gpu_inplace_loopback"]["gpu_inplace_prep_only_ms"],
                    "vae_bit_exact": report["test2_gpu_inplace_loopback"]["vae_input_bit_exact"],
                    "ar_match": report["test2_gpu_inplace_loopback"]["ar_text_exact_match"],
                },
                "e2e_4turn": [
                    {
                        "turn": t["turn"],
                        "seq": t["main_seq_len"],
                        "kv_saved_pct": t["kv_hbm_saved_pct"],
                        "stock_gen_ms": t["stock_gen_ms"],
                        "ms_gen_ms": t["morphserve_gen_ms"],
                        "ssim": t["ssim"],
                    }
                    for t in turn_comparisons
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MorphServe Module 3 (Shared-Prefix CFG + GPU Loopback) validation")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args(argv)
    rep = run_validation(args.model_dir, args.output_dir, steps=args.steps)
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
