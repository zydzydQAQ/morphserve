"""Phase-0 Job 0d: BAGEL-7B-MoT weight layout, 1-GPU VAE spike/DiT timing, and 2-GPU vLLM-Omni KV stall profiling."""

from __future__ import annotations

import argparse
import base64
import collections
import gc
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request


MIB = 1024**2
GIB = 1024**3


def _classify_bagel_param(name: str) -> tuple[str, int | None]:
    layer_match = re.search(r"layers\.(\d+)\.", name)
    layer_idx = int(layer_match.group(1)) if layer_match else None
    if name.startswith(("vae", "vae_model", "ae.")):
        return "vae", None
    if "vit" in name or "vision" in name or "connector" in name:
        return "vit_and_connector", None
    if "_moe_gen" in name or "time_embedder" in name or "vae2llm" in name or "llm2vae" in name or "latent_pos_embed" in name:
        if "self_attn" in name:
            return "generation_attn", layer_idx
        if "mlp" in name:
            return "generation_mlp", layer_idx
        if layer_idx is not None:
            return "generation_norm_other", layer_idx
        return "generation_io_embed", None
    if "self_attn" in name:
        return "understanding_attn", layer_idx
    if "mlp" in name:
        return "understanding_mlp", layer_idx
    if layer_idx is not None:
        return "understanding_norm_other", layer_idx
    if "embed_tokens" in name or "lm_head" in name or "norm" in name:
        return "understanding_io_embed", None
    return "other", layer_idx


def profile_weights_and_configs(model_dir: Path) -> dict:
    from safetensors import safe_open

    category_bytes: dict[str, int] = collections.defaultdict(int)
    category_params: dict[str, int] = collections.defaultdict(int)
    per_layer_bytes: dict[str, dict[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    dtype_bytes: dict[str, int] = collections.defaultdict(int)
    sample_names: dict[str, list[str]] = collections.defaultdict(list)
    files_scanned = []

    dtype_sizes = {
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "F64": 8,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
    }

    for st_path in sorted(model_dir.rglob("*.safetensors")):
        rel = str(st_path.relative_to(model_dir))
        file_bytes = 0
        with safe_open(str(st_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                sl = f.get_slice(key)
                shape = sl.get_shape()
                dtype = sl.get_dtype()
                numel = math.prod(shape) if shape else 1
                nbytes = numel * dtype_sizes.get(dtype, 2)
                cat, layer = _classify_bagel_param(key)
                if rel.startswith("ae.") or "ae.safetensors" in rel:
                    cat, layer = "vae", None
                category_bytes[cat] += nbytes
                category_params[cat] += numel
                dtype_bytes[dtype] += nbytes
                if layer is not None:
                    per_layer_bytes[cat][layer] += nbytes
                if len(sample_names[cat]) < 6:
                    sample_names[cat].append(f"{rel}:{key} {list(shape)} {dtype}")
                file_bytes += nbytes
        files_scanned.append({"file": rel, "tensor_mib": round(file_bytes / MIB, 2)})

    total_bytes = sum(category_bytes.values())
    und_total = sum(v for k, v in category_bytes.items() if k.startswith("understanding"))
    gen_total = sum(v for k, v in category_bytes.items() if k.startswith("generation"))
    vit_total = category_bytes.get("vit_and_connector", 0)
    vae_total = category_bytes.get("vae", 0)

    # Summarize per-layer sizes for generation expert & understanding expert
    num_layers = max(
        [max(d.keys(), default=-1) for d in per_layer_bytes.values()] + [-1]
    ) + 1
    gen_layer_mib = []
    und_layer_mib = []
    for lyr in range(num_layers):
        g_b = (
            per_layer_bytes["generation_attn"].get(lyr, 0)
            + per_layer_bytes["generation_mlp"].get(lyr, 0)
            + per_layer_bytes["generation_norm_other"].get(lyr, 0)
        )
        u_b = (
            per_layer_bytes["understanding_attn"].get(lyr, 0)
            + per_layer_bytes["understanding_mlp"].get(lyr, 0)
            + per_layer_bytes["understanding_norm_other"].get(lyr, 0)
        )
        gen_layer_mib.append(round(g_b / MIB, 2))
        und_layer_mib.append(round(u_b / MIB, 2))

    # Dump vllm_omni built-in BAGEL YAML deploy configs
    builtin_configs = {}
    try:
        import vllm_omni

        root = Path(vllm_omni.__file__).resolve().parent
        for rel_cfg in ("deploy/bagel.yaml", "deploy/bagel_single_stage.yaml", "deploy/bagel_think.yaml"):
            p = root / rel_cfg
            if p.is_file():
                builtin_configs[rel_cfg] = p.read_text(encoding="utf-8")
    except Exception as exc:
        builtin_configs["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "model_dir": str(model_dir),
        "files_scanned": files_scanned,
        "total_gib": round(total_bytes / GIB, 3),
        "understanding_expert_gib": round(und_total / GIB, 3),
        "generation_expert_gib": round(gen_total / GIB, 3),
        "vit_and_connector_gib": round(vit_total / GIB, 3),
        "vae_gib": round(vae_total / GIB, 3),
        "category_mib": {k: round(v / MIB, 2) for k, v in sorted(category_bytes.items())},
        "category_params_m": {k: round(v / 1e6, 2) for k, v in sorted(category_params.items())},
        "dtype_gib": {k: round(v / GIB, 3) for k, v in sorted(dtype_bytes.items())},
        "num_layers": num_layers,
        "per_layer_generation_mib_summary": {
            "min": min(gen_layer_mib) if gen_layer_mib else 0,
            "median": statistics.median(gen_layer_mib) if gen_layer_mib else 0,
            "max": max(gen_layer_mib) if gen_layer_mib else 0,
        },
        "per_layer_understanding_mib_summary": {
            "min": min(und_layer_mib) if und_layer_mib else 0,
            "median": statistics.median(und_layer_mib) if und_layer_mib else 0,
            "max": max(und_layer_mib) if und_layer_mib else 0,
        },
        "sample_names": dict(sample_names),
        "vllm_omni_builtin_bagel_yamls": builtin_configs,
    }


def _mem_mib() -> dict:
    import torch

    return {
        "allocated_mib": round(torch.cuda.memory_allocated() / MIB, 2),
        "reserved_mib": round(torch.cuda.memory_reserved() / MIB, 2),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / MIB, 2),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 2),
    }


def profile_single_gpu_bagel(model_dir: Path, steps: int = 20, width: int = 512, height: int = 512) -> dict:
    import numpy as np
    from PIL import Image
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.config.load import LoadConfig
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache
    from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    current_omni_platform.set_device(device)

    t_load0 = time.perf_counter()
    od_config = OmniDiffusionConfig(
        model=str(model_dir),
        model_class_name="BagelPipeline",
        dtype=torch.bfloat16,
        num_gpus=1,
        distributed_executor_backend="uni",
        trust_remote_code=True,
        enforce_eager=True,
    )
    vllm_config = create_diffusion_vllm_config(device, od_config)
    current_omni_platform.init_diffusion_worker_vllm_config(vllm_config)
    loader = DiffusersPipelineLoader(LoadConfig(), od_config)
    with (
        set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config),
        set_current_vllm_config(vllm_config),
    ):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file:///tmp/morphserve-p0-mot-{time.time_ns()}",
            local_rank=0,
        )
        current_omni_platform.set_device(device)
        pc = od_config.parallel_config
        initialize_model_parallel(
            data_parallel_size=pc.data_parallel_size,
            cfg_parallel_size=pc.cfg_parallel_size,
            sequence_parallel_size=pc.sequence_parallel_size,
            ulysses_degree=pc.ulysses_degree,
            ring_degree=pc.ring_degree,
            allgather_degree=pc.allgather_degree,
            tensor_parallel_size=pc.tensor_parallel_size,
            pipeline_parallel_size=pc.pipeline_parallel_size,
            fully_shard_degree=pc.hsdp_shard_size if pc.use_hsdp else 1,
            enable_expert_parallel=pc.enable_expert_parallel,
            use_hsdp=pc.use_hsdp,
        )
        pipeline = loader.load_model(load_device="cuda:0")
    pipeline.eval()
    torch.cuda.synchronize()
    load_sec = time.perf_counter() - t_load0
    mem_after_load = _mem_mib()

    # Instrument VAE decode & encode + DiT forward_flow calls to measure exact spikes and step CV!
    vae_events: list[dict] = []
    orig_vae_decode = pipeline.vae.decode
    orig_vae_encode = pipeline.vae.encode

    def wrapped_vae_decode(*args, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        before = _mem_mib()
        t0 = time.perf_counter()
        out = orig_vae_decode(*args, **kwargs)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        after = _mem_mib()
        vae_events.append(
            {
                "op": "vae_decode",
                "elapsed_ms": round(dt_ms, 2),
                "before_alloc_mib": before["allocated_mib"],
                "peak_alloc_mib": after["peak_allocated_mib"],
                "spike_delta_mib": round(after["peak_allocated_mib"] - before["allocated_mib"], 2),
                "after_alloc_mib": after["allocated_mib"],
                "reserved_mib": after["reserved_mib"],
                "input_shape": list(args[0].shape) if args and torch.is_tensor(args[0]) else None,
            }
        )
        return out

    def wrapped_vae_encode(*args, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        before = _mem_mib()
        t0 = time.perf_counter()
        out = orig_vae_encode(*args, **kwargs)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        after = _mem_mib()
        vae_events.append(
            {
                "op": "vae_encode",
                "elapsed_ms": round(dt_ms, 2),
                "before_alloc_mib": before["allocated_mib"],
                "peak_alloc_mib": after["peak_allocated_mib"],
                "spike_delta_mib": round(after["peak_allocated_mib"] - before["allocated_mib"], 2),
                "after_alloc_mib": after["allocated_mib"],
                "reserved_mib": after["reserved_mib"],
                "input_shape": list(args[0].shape) if args and torch.is_tensor(args[0]) else None,
            }
        )
        return out

    pipeline.vae.decode = wrapped_vae_decode
    pipeline.vae.encode = wrapped_vae_encode

    # Instrument BAGEL exact DiT step & per-CFG-branch methods discovered on vllm-omni 0.30.0
    step_call_ms: list[float] = []
    branch_call_ms: list[float] = []
    bagel = pipeline.bagel
    flow_method_name = None
    for cand in ("predict_noise_maybe_with_cfg", "predict_noise_with_multi_branch_cfg", "predict_noise"):
        if hasattr(bagel, cand):
            flow_method_name = cand
            break
    if flow_method_name is not None:
        orig_flow = getattr(bagel, flow_method_name)

        def wrapped_flow(*args, **kwargs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = orig_flow(*args, **kwargs)
            torch.cuda.synchronize()
            step_call_ms.append(round((time.perf_counter() - t0) * 1000.0, 3))
            return res

        setattr(bagel, flow_method_name, wrapped_flow)

    if hasattr(bagel, "forward_single_branch"):
        orig_branch = bagel.forward_single_branch

        def wrapped_branch(*args, **kwargs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = orig_branch(*args, **kwargs)
            torch.cuda.synchronize()
            branch_call_ms.append(round((time.perf_counter() - t0) * 1000.0, 3))
            return res

        bagel.forward_single_branch = wrapped_branch

    lm_forward_ms = branch_call_ms

    # Inspect BAGEL methods for AR text decode capability
    bagel_methods = sorted(m for m in dir(bagel) if not m.startswith("_"))

    def make_empty_ctx():
        return {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(bagel.config.llm_config.num_hidden_layers),
        }

    def move_dev(d):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}

    def update_text(ctx, prompt: str):
        inputs, kv_lens, ropes = bagel.prepare_prompts(
            curr_kvlens=ctx["kv_lens"],
            curr_rope=ctx["ropes"],
            prompts=[prompt],
            tokenizer=pipeline.tokenizer,
            new_token_ids=pipeline.new_token_ids,
        )
        inputs = move_dev(inputs)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ctx["past_key_values"] = bagel.forward_cache_update_text(ctx["past_key_values"], **inputs)
        ctx["kv_lens"] = kv_lens
        ctx["ropes"] = ropes

    def update_image(ctx, img: Image.Image):
        def vae_transform(v):
            arr = torch.from_numpy(np.array(v.convert("RGB"))).float() / 127.5 - 1.0
            return arr.permute(2, 0, 1)

        def vit_transform(v):
            return pipeline.image_processor(images=v, return_tensors="pt").pixel_values[0]

        inputs, kv_lens, ropes = bagel.prepare_vae_images(
            curr_kvlens=ctx["kv_lens"],
            curr_rope=ctx["ropes"],
            images=[img],
            transforms=vae_transform,
            new_token_ids=pipeline.new_token_ids,
        )
        inputs = move_dev(inputs)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ctx["past_key_values"] = bagel.forward_cache_update_vae(
                pipeline.vae, ctx["past_key_values"], **inputs
            )
        ctx["kv_lens"] = kv_lens
        ctx["ropes"] = ropes

        inputs, kv_lens, ropes = bagel.prepare_vit_images(
            curr_kvlens=ctx["kv_lens"],
            curr_rope=ctx["ropes"],
            images=[img],
            transforms=vit_transform,
            new_token_ids=pipeline.new_token_ids,
        )
        inputs = move_dev(inputs)
        for k in ("packed_indexes", "packed_key_value_indexes", "key_values_lens"):
            inputs.pop(k, None)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ctx["past_key_values"] = bagel.forward_cache_update_vit(ctx["past_key_values"], **inputs)
        ctx["kv_lens"] = kv_lens
        ctx["ropes"] = ropes

    def kv_bytes(ctx) -> int:
        cache = ctx["past_key_values"]
        seen = {}
        for lyr in range(cache.num_layers):
            for t in (cache.key_cache[lyr], cache.value_cache[lyr]):
                if t is not None:
                    st = t.untyped_storage()
                    seen[(st.data_ptr(), st.nbytes())] = st.nbytes()
        return sum(seen.values())

    # 4-turn interleaved Visual-CoT / editing sequence
    prompts = [
        "A serene wooden cabin beside an alpine lake at golden hour, realistic photography.",
        "Add a small red canoe floating near the wooden dock on the calm lake.",
        "Change the season to early autumn with golden birch trees along the shoreline.",
        "Add soft mist rising over the water and warm glowing light inside the cabin windows.",
    ]

    main_ctx = make_empty_ctx()
    cfg_img_ctx = make_empty_ctx()
    turn_records = []
    generated_images: list[Image.Image] = []

    for turn_idx, prompt in enumerate(prompts):
        vae_start_idx = len(vae_events)
        lm_start_idx = len(lm_forward_ms)
        flow_start_idx = len(step_call_ms)

        t_turn0 = time.perf_counter()
        t_prefill0 = time.perf_counter()
        if generated_images:
            # Measure CPU PNG loopback cost vs pure update_image
            buf = io.BytesIO()
            t_png0 = time.perf_counter()
            generated_images[-1].save(buf, format="PNG")
            buf.seek(0)
            reloaded = Image.open(buf).convert("RGB")
            reloaded.load()
            png_loopback_ms = (time.perf_counter() - t_png0) * 1000.0
            update_image(main_ctx, reloaded)
        else:
            png_loopback_ms = 0.0

        # Snapshot cfg_text prefix BEFORE appending turn text (exact BAGEL 3-way CFG prefix sharing!)
        cfg_text_ctx = {
            "kv_lens": list(main_ctx["kv_lens"]),
            "ropes": list(main_ctx["ropes"]),
            "past_key_values": NaiveCache(bagel.config.llm_config.num_hidden_layers),
        }
        cfg_text_ctx["past_key_values"].key_cache = main_ctx["past_key_values"].key_cache.copy()
        cfg_text_ctx["past_key_values"].value_cache = main_ctx["past_key_values"].value_cache.copy()

        update_text(main_ctx, prompt)
        update_text(cfg_img_ctx, prompt)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t_prefill0) * 1000.0

        # Measure real AR text generation TPOT on cloned NaiveCache (32 tokens)
        ar_decode_info = None
        if hasattr(bagel, "generate_text"):
            try:
                import inspect as _inspect

                sig = str(_inspect.signature(bagel.generate_text))
                ar_decode_info = {"signature": sig}
                if turn_idx in (0, 3):
                    tmp_kv = NaiveCache(bagel.config.llm_config.num_hidden_layers)
                    for lyr in range(tmp_kv.num_layers):
                        k = main_ctx["past_key_values"].key_cache[lyr]
                        v = main_ctx["past_key_values"].value_cache[lyr]
                        tmp_kv.key_cache[lyr] = None if k is None else k.clone()
                        tmp_kv.value_cache[lyr] = None if v is None else v.clone()
                    ar_inputs, _, _ = bagel.prepare_start_tokens(
                        curr_kvlens=list(main_ctx["kv_lens"]),
                        curr_rope=list(main_ctx["ropes"]),
                        new_token_ids=pipeline.new_token_ids,
                    )
                    ar_inputs = move_dev(ar_inputs)
                    torch.cuda.synchronize()
                    t_ar0 = time.perf_counter()
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        gen_ids = bagel.generate_text(
                            past_key_values=tmp_kv,
                            max_length=32,
                            do_sample=False,
                            end_token_id=pipeline.new_token_ids["eos_token_id"],
                            **ar_inputs,
                        )
                    torch.cuda.synchronize()
                    ar_ms = (time.perf_counter() - t_ar0) * 1000.0
                    n_tok = int(gen_ids.numel()) if torch.is_tensor(gen_ids) else len(gen_ids)
                    ar_decode_info.update(
                        {
                            "tokens": n_tok,
                            "total_ms": round(ar_ms, 2),
                            "tpot_ms": round(ar_ms / max(1, n_tok), 2),
                        }
                    )
                    del tmp_kv, ar_inputs, gen_ids
            except Exception as exc:
                ar_decode_info = {"error": f"{type(exc).__name__}: {exc}"}

        # Run DiT denoising + FLUX VAE decode
        torch.cuda.reset_peak_memory_stats(device)
        mem_pre_gen = _mem_mib()
        sampling = OmniDiffusionSamplingParams(
            height=height,
            width=width,
            num_inference_steps=steps,
            seed=52 + turn_idx,
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
        sampling.kv_metadata = {"ropes": main_ctx["ropes"], "image_shape": [width, height]}
        sampling.cfg_img_past_key_values = cfg_img_ctx["past_key_values"]
        sampling.cfg_img_kv_metadata = {"ropes": cfg_img_ctx["ropes"], "image_shape": [width, height]}
        if turn_idx > 0:
            sampling.cfg_text_past_key_values = cfg_text_ctx["past_key_values"]
            sampling.cfg_text_kv_metadata = {"ropes": cfg_text_ctx["ropes"], "image_shape": [width, height]}

        t_gen0 = time.perf_counter()
        with torch.inference_mode():
            out = pipeline._forward_single({"prompt": prompt, "modalities": ["image"]}, sampling)
        torch.cuda.synchronize()
        gen_ms = (time.perf_counter() - t_gen0) * 1000.0
        mem_post_gen = _mem_mib()

        img = out.output["payload"]["image"]
        generated_images.append(img)

        turn_vae = vae_events[vae_start_idx:]
        turn_lm_ms = lm_forward_ms[lm_start_idx:]
        turn_flow_ms = step_call_ms[flow_start_idx:]
        step_cv = (
            round(statistics.stdev(turn_flow_ms[1:]) / statistics.mean(turn_flow_ms[1:]), 4)
            if len(turn_flow_ms) >= 5 and statistics.mean(turn_flow_ms[1:]) > 0
            else None
        )
        branch_cv = (
            round(statistics.stdev(turn_lm_ms[3:]) / statistics.mean(turn_lm_ms[3:]), 4)
            if len(turn_lm_ms) >= 6 and statistics.mean(turn_lm_ms[3:]) > 0
            else None
        )

        turn_records.append(
            {
                "turn": turn_idx,
                "main_seq_len": main_ctx["past_key_values"].seq_lens,
                "cfg_img_seq_len": cfg_img_ctx["past_key_values"].seq_lens,
                "cfg_text_seq_len": cfg_text_ctx["past_key_values"].seq_lens,
                "main_kv_mib": round(kv_bytes(main_ctx) / MIB, 2),
                "cfg_img_kv_mib": round(kv_bytes(cfg_img_ctx) / MIB, 2),
                "png_cpu_loopback_ms": round(png_loopback_ms, 2),
                "prefill_ms": round(prefill_ms, 2),
                "gen_total_ms": round(gen_ms, 2),
                "turn_total_ms": round((time.perf_counter() - t_turn0) * 1000.0, 2),
                "mem_pre_gen": mem_pre_gen,
                "mem_post_gen": mem_post_gen,
                "vae_events": turn_vae,
                "dit_step_call_count": len(turn_flow_ms),
                "dit_step_ms_all": turn_flow_ms,
                "dit_step_p50_ms": round(statistics.median(turn_flow_ms), 2) if turn_flow_ms else None,
                "dit_step_cv_steady": step_cv,
                "cfg_branch_call_count": len(turn_lm_ms),
                "cfg_branch_ms_sample": turn_lm_ms[:15],
                "cfg_branch_p50_ms": round(statistics.median(turn_lm_ms), 2) if turn_lm_ms else None,
                "cfg_branch_cv_steady": branch_cv,
                "ar_decode_info": ar_decode_info,
            }
        )

    # Probe 1024x1024 FLUX VAE decode spike + batch=2 static OOM wall on 40GB A100!
    vae_stress = {}
    for label, (b, h_lat, w_lat) in (
        ("512_b1", (1, 64, 64)),
        ("512_b2", (2, 64, 64)),
        ("1024_b1", (1, 128, 128)),
        ("1024_b2", (2, 128, 128)),
    ):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before = _mem_mib()
        try:
            lat = torch.randn((b, 16, h_lat, w_lat), dtype=torch.bfloat16, device=device)
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                dec = orig_vae_decode(lat)
            torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            after = _mem_mib()
            vae_stress[label] = {
                "oom": False,
                "elapsed_ms": round(dt_ms, 2),
                "before_alloc_mib": before["allocated_mib"],
                "peak_alloc_mib": after["peak_allocated_mib"],
                "spike_delta_mib": round(after["peak_allocated_mib"] - before["allocated_mib"], 2),
                "peak_reserved_mib": after["peak_reserved_mib"],
                "out_shape": list(dec.shape),
            }
            del dec, lat
        except torch.cuda.OutOfMemoryError as exc:
            vae_stress[label] = {
                "oom": True,
                "before_alloc_mib": before["allocated_mib"],
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / MIB, 2),
                "error": str(exc)[:300],
            }
        gc.collect()
        torch.cuda.empty_cache()

    del main_ctx, cfg_img_ctx, pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "status": "ok",
        "load_seconds": round(load_sec, 2),
        "mem_after_load": mem_after_load,
        "flow_method_hooked": flow_method_name,
        "bagel_methods_head": bagel_methods[:60],
        "turns": turn_records,
        "vae_batch_and_res_stress": vae_stress,
    }


def _image_to_data_url(img) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _find_data_image(value):
    if isinstance(value, dict):
        for k, v in value.items():
            if k == "url" and isinstance(v, str) and v.startswith("data:image/"):
                return v
            if k == "b64_json" and isinstance(v, str):
                return "data:image/png;base64," + v
            found = _find_data_image(v)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_data_image(item)
            if found:
                return found
    return None


def _parse_kv_transfer_log(text: str) -> dict:
    transfers = []
    for req_id, byte_count, sec in re.findall(
        r"KV transfer OK: (\S+?), (\d+) bytes across [^,]+, ([0-9.]+)s", text
    ):
        transfers.append({"request_id": req_id, "bytes": int(byte_count), "seconds": float(sec)})
    serializes = []
    for req_id, ms in re.findall(r"KV cache serialized for (\S+) in ([0-9.]+) ms", text):
        serializes.append({"request_id": req_id, "ms": float(ms)})
    return {
        "transfer_count": len(transfers),
        "total_transfer_mib": round(sum(x["bytes"] for x in transfers) / MIB, 2),
        "total_transfer_seconds": round(sum(x["seconds"] for x in transfers), 4),
        "total_serialize_seconds": round(sum(x["ms"] for x in serializes) / 1000.0, 4),
        "transfers": transfers,
        "serializes": serializes,
    }


def profile_two_stage_vllm_omni(
    model_dir: Path,
    deploy_config: Path,
    log_path: Path,
    steps: int = 20,
    width: int = 512,
    height: int = 512,
) -> dict:
    from PIL import Image

    port = 8191
    cmd = [
        "vllm",
        "serve",
        str(model_dir),
        "--omni",
        "--port",
        str(port),
        "--deploy-config",
        str(deploy_config),
        "--omni-master-address",
        "127.0.0.1",
        "--omni-master-port",
        "18191",
    ]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_LOGGING_LEVEL": "INFO",
            "MORPHSERVE_ENABLE_BAGEL_MM_ADAPTER": "1",
            "MORPHSERVE_MAX_IMAGE_ITEMS": "16",
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        # Wait up to 900s for healthy server
        deadline = time.monotonic() + 900.0
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:])
                return {"status": "server_crashed", "returncode": proc.returncode, "log_tail": tail}
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(3)
        if not ready:
            return {"status": "startup_timeout"}

        prompts = [
            "A serene wooden cabin beside an alpine lake at golden hour, realistic photography.",
            "Add a small red canoe floating near the wooden dock on the calm lake.",
            "Change the season to early autumn with golden birch trees along the shoreline.",
            "Add soft mist rising over the water and warm glowing light inside the cabin windows.",
        ]
        turns = []
        messages: list[dict] = []
        last_img_url = None
        for idx, prompt in enumerate(prompts):
            pos = log_path.stat().st_size
            content = [{"type": "text", "text": f"<|im_start|>{prompt}<|im_end|>"}]
            if last_img_url is not None:
                content.append({"type": "image_url", "image_url": {"url": last_img_url}})
            messages.append({"role": "user", "content": content})

            payload = {
                "messages": messages,
                "modalities": ["image"],
                "height": height,
                "width": width,
                "num_inference_steps": steps,
                "seed": 52 + idx,
                "cfg_text_scale": 4.0,
                "cfg_img_scale": 1.5,
                "cfg_interval": [0.4, 1.0],
                "cfg_renorm_type": "global",
                "cfg_renorm_min": 0.0,
                "timestep_shift": 3.0,
            }
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    body = json.load(resp)
            except urllib.error.HTTPError as err:
                err_txt = err.read().decode("utf-8", "replace")[:2000]
                turns.append({"turn": idx, "error": f"HTTP {err.code}: {err_txt}"})
                break
            elapsed = time.perf_counter() - t0
            data_url = _find_data_image(body)
            if not data_url:
                turns.append({"turn": idx, "error": "no image in response"})
                break
            raw = base64.b64decode(data_url.split(",", 1)[1])
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            img.load()
            last_img_url = _image_to_data_url(img)

            # Read new log chunk for exact KV transfer stats
            with log_path.open("r", encoding="utf-8", errors="replace") as rh:
                rh.seek(pos)
                chunk = rh.read()
            kv_stats = _parse_kv_transfer_log(chunk)
            turns.append(
                {
                    "turn": idx,
                    "e2e_seconds": round(elapsed, 3),
                    "usage": body.get("usage"),
                    "kv_transfer": kv_stats,
                }
            )
            # Keep assistant turn marker minimal for multi-turn chaining
            messages.append({"role": "assistant", "content": [{"type": "text", "text": "Generated image."}]})

        full_kv = _parse_kv_transfer_log(log_path.read_text(encoding="utf-8", errors="replace"))
        return {"status": "ok", "deploy_config": str(deploy_config), "turns": turns, "aggregate_kv_transfer": full_kv}
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=20)
        log_fh.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MorphServe Phase-0 BAGEL-7B-MoT Motivation Profiler")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deploy-config", type=Path, default=None)
    parser.add_argument("--modes", nargs="+", default=["weights", "single_gpu", "two_stage"])
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "schema": "morphserve.phase0.motivation.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_dir": str(args.model_dir),
    }
    if "weights" in args.modes:
        report["weights_and_configs"] = profile_weights_and_configs(args.model_dir)
    if "single_gpu" in args.modes:
        report["single_gpu"] = profile_single_gpu_bagel(args.model_dir, steps=args.steps)
    if "two_stage" in args.modes:
        cfg = args.deploy_config
        if cfg is None:
            raise ValueError("--deploy-config required for two_stage mode")
        report["two_stage"] = profile_two_stage_vllm_omni(
            args.model_dir,
            cfg,
            args.output_dir / "vllm-omni-2gpu.log",
            steps=args.steps,
        )

    out_path = args.output_dir / "motivation-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
