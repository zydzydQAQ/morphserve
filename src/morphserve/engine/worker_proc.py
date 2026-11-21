"""Persistent per-GPU worker process for MorphServeClusterEngine."""

from __future__ import annotations

from copy import deepcopy
import gc
import io
from multiprocessing import shared_memory
import os
from pathlib import Path
import queue
import threading
import time
import traceback
from typing import Any

from morphserve.bench.validate_module1_bagel import load_bagel_pipeline
from morphserve.kernels import (
    GpuInPlaceLoopback,
    SharedPrefixCache,
    cow_fork_context,
    decode_latent_to_u8_gpu,
    install_shared_prefix_bagel_hooks,
    rebind_cow_prefix,
)
from morphserve.router import (
    ClusterTopologyProfile,
    DirectNcclMesh,
    HalfNodeH2DPacer,
    SharedNumaMemfdMirror,
)
from morphserve.vmm import BagelExpertPager, TriPoolVMM
from morphserve.vmm.bagel_pager import is_generation_param_name
from morphserve.worker import ArDecodeTask, UnifiedPreemptiveScheduler

from .config import EnginePolicyConfig

MIB = 1024**2


def _ctx_nbytes(ctx: dict[str, Any] | None) -> int:
    if ctx is None:
        return 0
    pkv = ctx.get("past_key_values")
    if pkv is None:
        return 0
    num_layers = int(pkv.num_layers)
    get_k = pkv.key_cache.get if isinstance(pkv.key_cache, dict) else (lambda i: pkv.key_cache[i])
    get_v = pkv.value_cache.get if isinstance(pkv.value_cache, dict) else (lambda i: pkv.value_cache[i])
    seen: dict[tuple[int, int], int] = {}
    for l in range(num_layers):
        for t in (get_k(l), get_v(l)):
            if t is not None:
                st = t.untyped_storage()
                seen[(st.data_ptr(), st.nbytes())] = int(st.nbytes())
    return sum(seen.values())


def _entry_nbytes(entry: dict[str, Any]) -> int:
    return (
        _ctx_nbytes(entry.get("main_ctx"))
        + _ctx_nbytes(entry.get("cfg_text_ctx"))
        + _ctx_nbytes(entry.get("cfg_img_ctx"))
    )


def _new_empty_ctx(num_layers: int, naive_cache_cls: Any) -> dict[str, Any]:
    return {"kv_lens": [0], "ropes": [0], "past_key_values": naive_cache_cls(num_layers)}


def _slice_prefix_ctx(main_ctx: dict[str, Any], prefix_len: int, naive_cache_cls: Any) -> dict[str, Any] | None:
    if prefix_len <= 0:
        return None
    pkv = main_ctx["past_key_values"]
    num_layers = int(pkv.num_layers)
    sub = naive_cache_cls(num_layers)
    for l in range(num_layers):
        k = pkv.key_cache[l]
        v = pkv.value_cache[l]
        sub.key_cache[l] = None if k is None else k[:prefix_len]
        sub.value_cache[l] = None if v is None else v[:prefix_len]
    sub._cow_shared_prefix_len = prefix_len
    return {
        "kv_lens": [prefix_len],
        "ropes": list(main_ctx["ropes"]),
        "past_key_values": sub,
    }


def _pack_multi_ctx(
    main_ctx: dict[str, Any],
    cfg_text_ctx: dict[str, Any] | None,
    cfg_img_ctx: dict[str, Any] | None,
    cfg_text_prefix_len: int,
    include_cfg_text_dup: bool,
    device: Any,
    pending_vae_paths: list[str] | None = None,
) -> tuple[dict[str, Any], Any]:
    import torch

    named_items: list[tuple[str, dict[str, Any]]] = [("main", main_ctx)]
    if include_cfg_text_dup and cfg_text_ctx is not None and cfg_text_ctx["past_key_values"].key_cache[0] is not None:
        named_items.append(("cfg_text", cfg_text_ctx))
    if cfg_img_ctx is not None and cfg_img_ctx["past_key_values"].key_cache[0] is not None:
        named_items.append(("cfg_img", cfg_img_ctx))

    sections: dict[str, Any] = {}
    flat_chunks: list[Any] = []
    offset_bytes = 0
    for name, c in named_items:
        m, fl = DirectNcclMesh.pack_bagel_context(c, device=device)
        nbytes = int(fl.numel())
        sections[name] = {"meta": m, "offset": offset_bytes, "nbytes": nbytes}
        if nbytes > 0:
            flat_chunks.append(fl)
            offset_bytes += nbytes

    flat = torch.cat(flat_chunks, dim=0) if flat_chunks else torch.empty((0,), dtype=torch.uint8, device=device)
    meta = {
        "total_bytes": int(flat.numel()),
        "cfg_text_prefix_len": int(cfg_text_prefix_len),
        "include_cfg_text_dup": bool(include_cfg_text_dup),
        "pending_vae_paths": list(pending_vae_paths or []),
        "sections": sections,
    }
    return meta, flat


def _unpack_multi_ctx(
    meta: dict[str, Any],
    flat_u8: Any,
    naive_cache_cls: Any,
    use_cow_rebind: bool,
) -> dict[str, Any]:
    sections = meta["sections"]
    m_sec = sections["main"]
    main_ctx = DirectNcclMesh.unpack_bagel_context(
        m_sec["meta"],
        flat_u8[m_sec["offset"] : m_sec["offset"] + m_sec["nbytes"]],
        naive_cache_cls,
    )
    cfg_text_ctx = None
    if "cfg_text" in sections:
        ct_sec = sections["cfg_text"]
        cfg_text_ctx = DirectNcclMesh.unpack_bagel_context(
            ct_sec["meta"],
            flat_u8[ct_sec["offset"] : ct_sec["offset"] + ct_sec["nbytes"]],
            naive_cache_cls,
        )
        if use_cow_rebind:
            rebind_cow_prefix(main_ctx["past_key_values"], cfg_text_ctx["past_key_values"])
    elif int(meta.get("cfg_text_prefix_len", 0)) > 0:
        cfg_text_ctx = _slice_prefix_ctx(main_ctx, int(meta["cfg_text_prefix_len"]), naive_cache_cls)

    cfg_img_ctx = None
    if "cfg_img" in sections:
        ci_sec = sections["cfg_img"]
        cfg_img_ctx = DirectNcclMesh.unpack_bagel_context(
            ci_sec["meta"],
            flat_u8[ci_sec["offset"] : ci_sec["offset"] + ci_sec["nbytes"]],
            naive_cache_cls,
        )

    return {
        "main_ctx": main_ctx,
        "cfg_text_ctx": cfg_text_ctx,
        "cfg_img_ctx": cfg_img_ctx,
        "cfg_text_prefix_len": int(meta.get("cfg_text_prefix_len", 0)),
        "pending_vae_paths": list(meta.get("pending_vae_paths") or []),
        "last_img_u8": None,
        "last_img_pil": None,
    }


def run_worker_process(
    rank: int,
    policy: EnginePolicyConfig,
    model_dir_str: str,
    bench_root_str: str,
    cmd_q: Any,
    ack_q: Any,
) -> None:
    """Main entry point for each persistent GPU worker process."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import numpy as np
    from PIL import Image
    import torch

    torch.cuda.set_device(rank)
    from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    device = torch.device(f"cuda:{rank}")
    bench_root = Path(bench_root_str)
    model_dir = Path(model_dir_str)
    topo = ClusterTopologyProfile.a100_pcie_8gpu_default()
    numa_node = topo.numa_node_of(rank)
    role = policy.role_of(rank)

    pipeline = None
    pager: BagelExpertPager | None = None
    vmm: TriPoolVMM | None = None
    shared_mirror: SharedNumaMemfdMirror | None = None
    nccl_mesh: DirectNcclMesh | None = None
    cfg_patch = None
    sched: UnifiedPreemptiveScheduler | None = None
    loopback: GpuInPlaceLoopback | None = None
    pacer = HalfNodeH2DPacer(topo, max_concurrent_per_half=1) if policy.use_h2d_pacer else None

    cpu_naive_weights: dict[str, Any] = {}
    img_cache: dict[str, Image.Image] = {}
    ctx_store: dict[str, dict[str, Any]] = {}
    packed_store: dict[str, Any] = {}
    kv_vmm_pages: dict[str, list[int]] = {}
    oom_count = 0
    max_reserved_mib = 0.0

    # Background inbox separates urgent mid-DiT AR preemption commands from normal sequential commands
    main_cmd_q: queue.Queue = queue.Queue()
    ar_preempt_q: queue.Queue = queue.Queue()
    stop_inbox = threading.Event()

    def _inbox_loop() -> None:
        while not stop_inbox.is_set():
            try:
                msg = cmd_q.get(timeout=0.05)
            except queue.Empty:
                continue
            if msg is None:
                continue
            if msg.get("op") == "EXEC_AR_PREEMPT":
                ar_preempt_q.put(msg)
            else:
                main_cmd_q.put(msg)
            if msg.get("op") == "SHUTDOWN":
                break

    inbox_thread = threading.Thread(target=_inbox_loop, name=f"ms_inbox_r{rank}", daemon=True)
    inbox_thread.start()

    def _load_pil(rel_path: str) -> Image.Image:
        im = img_cache.get(rel_path)
        if im is None:
            p = bench_root / rel_path if not os.path.isabs(rel_path) else Path(rel_path)
            with Image.open(p) as raw:
                im = raw.convert("RGB")
                im.load()
            if len(img_cache) < 512:
                img_cache[rel_path] = im
        return im

    def _update_mem_hwm() -> None:
        nonlocal max_reserved_mib
        res = float(torch.cuda.memory_reserved(device) / MIB)
        max_reserved_mib = max(max_reserved_mib, res)

    def _alloc_kv_vmm_if_needed(ctx_id: str, ctx: dict[str, Any]) -> None:
        if vmm is None or not policy.kv_vmm_only:
            return
        needed_pages = max(1, (_ctx_nbytes(ctx) + 2 * MIB - 1) // (2 * MIB))
        held = kv_vmm_pages.setdefault(ctx_id, [])
        while len(held) < needed_pages:
            try:
                held.append(vmm.kv_alloc_page())
            except RuntimeError:
                break

    def _free_kv_vmm(ctx_id: str) -> None:
        if vmm is None:
            return
        pages = kv_vmm_pages.pop(ctx_id, [])
        if pages:
            stream_ptr = int(torch.cuda.current_stream(device).cuda_stream)
            for slot in pages:
                try:
                    vmm.kv_free_page(slot, stream_ptr=stream_ptr)
                except Exception:
                    pass

    def _update_text_ctx(ctx: dict[str, Any], prompt: str) -> None:
        clean = prompt.removeprefix("<|im_start|>").removesuffix("<|im_end|>")
        inp, nl, nr = pipeline.bagel.prepare_prompts(
            curr_kvlens=ctx["kv_lens"],
            curr_rope=ctx["ropes"],
            prompts=[clean],
            tokenizer=pipeline.tokenizer,
            new_token_ids=pipeline.new_token_ids,
        )
        inp = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inp.items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ctx["past_key_values"] = pipeline.bagel.forward_cache_update_text(ctx["past_key_values"], **inp)
        ctx["kv_lens"], ctx["ropes"] = nl, nr

    def _ensure_gen_weights_for_vae() -> bool:
        if pager is not None and vmm is not None:
            if vmm.expert_resident_count(pager.expert_id) < pager.num_layers:
                vmm.load_expert_layers_sync(pager.expert_id, 0, pager.num_layers)
            return False
        if policy.use_naive_offload and cpu_naive_weights:
            for layer in pipeline.bagel.language_model.model.layers:
                for pname, p in layer.named_parameters():
                    key = f"{id(layer)}.{pname}"
                    if key in cpu_naive_weights and p.numel() == 0:
                        p.data = cpu_naive_weights[key].to(device)
            torch.cuda.synchronize(device)
            return True
        return False

    def _drop_naive_gen_weights() -> None:
        if policy.use_naive_offload and cpu_naive_weights:
            for layer in pipeline.bagel.language_model.model.layers:
                for pname, p in layer.named_parameters():
                    key = f"{id(layer)}.{pname}"
                    if key in cpu_naive_weights:
                        p.data = torch.empty(0, dtype=p.dtype, device=device)
            torch.cuda.empty_cache()

    def _update_image_ctx(
        ctx: dict[str, Any],
        img_or_u8: Any,
        update_vae: bool = True,
        update_vit: bool = True,
    ) -> None:
        swapped = _ensure_gen_weights_for_vae() if update_vae else False
        try:
            if torch.is_tensor(img_or_u8) and loopback is not None:
                loopback.update_context_from_u8_gpu(ctx, img_or_u8, update_vae=update_vae, update_vit=update_vit)
                return
            pil_img = img_or_u8

            def vae_tf(v):
                arr = torch.from_numpy(np.array(v.convert("RGB"), copy=True)).float() / 127.5 - 1.0
                return arr.permute(2, 0, 1)

            def vit_tf(v):
                return pipeline.image_processor(images=v, return_tensors="pt").pixel_values[0]

            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if update_vae:
                    vi, nl, nr = pipeline.bagel.prepare_vae_images(
                        curr_kvlens=ctx["kv_lens"],
                        curr_rope=ctx["ropes"],
                        images=[pil_img],
                        transforms=vae_tf,
                        new_token_ids=pipeline.new_token_ids,
                    )
                    vi = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in vi.items()}
                    ctx["past_key_values"] = pipeline.bagel.forward_cache_update_vae(
                        pipeline.vae, ctx["past_key_values"], **vi
                    )
                    ctx["kv_lens"], ctx["ropes"] = nl, nr
                if update_vit:
                    vti, nlv, nrv = pipeline.bagel.prepare_vit_images(
                        curr_kvlens=ctx["kv_lens"],
                        curr_rope=ctx["ropes"],
                        images=[pil_img],
                        transforms=vit_tf,
                        new_token_ids=pipeline.new_token_ids,
                    )
                    vti = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in vti.items()}
                    for k in ("packed_indexes", "packed_key_value_indexes", "key_values_lens"):
                        vti.pop(k, None)
                    ctx["past_key_values"] = pipeline.bagel.forward_cache_update_vit(ctx["past_key_values"], **vti)
                    ctx["kv_lens"], ctx["ropes"] = nlv, nrv
        finally:
            if swapped:
                _drop_naive_gen_weights()

    def _run_ar_direct(ctx: dict[str, Any], max_tokens: int) -> dict[str, Any]:
        start_inp = pipeline.bagel.prepare_start_tokens(ctx["kv_lens"], ctx["ropes"], pipeline.new_token_ids)
        curr_tokens = start_inp["packed_start_tokens"].to(device)
        pos_ids = start_inp["packed_query_position_ids"].to(device)
        end_id = pipeline.new_token_ids["eos_token_id"]

        prev_guard = None
        if pager is not None:
            prev_guard = pager._active_guard
            pager._active_guard = False

        gen_ids: list[int] = []
        stamps: list[float] = []
        t0 = time.perf_counter()
        try:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for _ in range(max_tokens):
                    gen_ids.append(int(curr_tokens[0].item()))
                    ql = torch.ones_like(curr_tokens)
                    out = pipeline.bagel.language_model(
                        packed_text_ids=curr_tokens,
                        query_lens=ql,
                        packed_query_position_ids=pos_ids,
                        past_key_values=ctx["past_key_values"],
                        update_past_key_values=True,
                        is_causal=True,
                        mode="und",
                    )
                    ctx["past_key_values"] = out.past_key_values
                    logits = pipeline.bagel.language_model.lm_head(out.packed_query_sequence)
                    next_tok = torch.argmax(logits, dim=-1)
                    curr_tokens = next_tok
                    pos_ids = pos_ids + 1
                    torch.cuda.synchronize(device)
                    stamps.append(time.perf_counter())
                    if end_id is not None and int(next_tok[0].item()) == int(end_id):
                        break
        finally:
            if pager is not None and prev_guard is not None:
                pager._active_guard = prev_guard

        n_tok = len(gen_ids)
        ctx["kv_lens"] = [kl + n_tok for kl in ctx["kv_lens"]]
        ctx["ropes"] = [r + n_tok for r in ctx["ropes"]]
        ttft_ms = (stamps[0] - t0) * 1000.0 if stamps else 0.0
        tpots_ms = [(stamps[i] - stamps[i - 1]) * 1000.0 for i in range(1, len(stamps))]
        decoded = pipeline.tokenizer.decode(gen_ids) if gen_ids else ""
        return {
            "text": decoded.split("<|im_end|>")[0].split("<|im_start|>")[-1],
            "tokens": n_tok,
            "ttft_ms": round(ttft_ms, 2),
            "tpots_ms": [round(x, 2) for x in tpots_ms],
        }

    def _prepare_entry_for_cmd(cmd: dict[str, Any]) -> dict[str, Any]:
        ctx_id = cmd["ctx_id"]
        num_layers = pipeline.bagel.config.llm_config.num_hidden_layers
        entry = ctx_store.get(ctx_id)
        if entry is None:
            entry = {
                "main_ctx": _new_empty_ctx(num_layers, NaiveCache),
                "cfg_text_ctx": None,
                "cfg_img_ctx": None,
                "cfg_text_prefix_len": 0,
                "last_img_u8": None,
                "last_img_pil": None,
            }
            ctx_store[ctx_id] = entry

        entry.setdefault("pending_vae_paths", [])
        main_ctx = entry["main_ctx"]
        prep_dit = bool(cmd.get("prepare_for_dit", False))
        # Pure VQA (`prep_dit=False`) only needs ViT (`mode='und'`).
        # On static `und` ranks (which dropped `gen` weights), defer VAE latent encode to the `gen` rank!
        can_run_vae_here = prep_dit and not (policy.routing_mode == "static_nccl_fission" and role == "und")

        if cmd.get("use_prev_image"):
            if cmd.get("prev_image_path"):
                p_prev = str(cmd["prev_image_path"])
                _update_image_ctx(main_ctx, _load_pil(p_prev), update_vae=can_run_vae_here, update_vit=True)
                if prep_dit and not can_run_vae_here:
                    entry["pending_vae_paths"].append(p_prev)
            elif policy.use_module3_cow_loopback and entry.get("last_img_u8") is not None:
                _update_image_ctx(main_ctx, entry["last_img_u8"], update_vae=can_run_vae_here, update_vit=True)
            elif entry.get("last_img_pil") is not None:
                prev_pil = entry["last_img_pil"]
                if not policy.use_module3_cow_loopback:
                    buf = io.BytesIO()
                    prev_pil.save(buf, format="PNG")
                    buf.seek(0)
                    prev_pil = Image.open(buf).convert("RGB")
                    prev_pil.load()
                _update_image_ctx(main_ctx, prev_pil, update_vae=can_run_vae_here, update_vit=True)

        for rel_img in cmd.get("image_paths", []):
            _update_image_ctx(main_ctx, _load_pil(rel_img), update_vae=can_run_vae_here, update_vit=True)
            if prep_dit and not can_run_vae_here:
                entry["pending_vae_paths"].append(str(rel_img))

        prompt = cmd["prompt"]
        if prep_dit:
            has_hist = main_ctx["past_key_values"].key_cache[0] is not None
            if has_hist:
                cfg_text_ctx = (
                    cow_fork_context(main_ctx)
                    if policy.use_module3_cow_loopback
                    else deepcopy(main_ctx)
                )
                entry["cfg_text_prefix_len"] = int(main_ctx["past_key_values"].key_cache[0].shape[0])
            else:
                cfg_text_ctx = None
                entry["cfg_text_prefix_len"] = 0

            _update_text_ctx(main_ctx, prompt)
            if cfg_text_ctx is not None and policy.use_module3_cow_loopback:
                rebind_cow_prefix(main_ctx["past_key_values"], cfg_text_ctx["past_key_values"])

            cfg_img_ctx = _new_empty_ctx(num_layers, NaiveCache)
            _update_text_ctx(cfg_img_ctx, prompt)
            entry["cfg_text_ctx"] = cfg_text_ctx
            entry["cfg_img_ctx"] = cfg_img_ctx
        else:
            _update_text_ctx(main_ctx, prompt)

        _alloc_kv_vmm_if_needed(ctx_id, main_ctx)
        return entry

    def _run_dit_exec(cmd: dict[str, Any]) -> dict[str, Any]:
        ctx_id = cmd["ctx_id"]
        entry = ctx_store[ctx_id]
        main_ctx = entry["main_ctx"]
        cfg_text_ctx = entry.get("cfg_text_ctx")
        cfg_img_ctx = entry.get("cfg_img_ctx")
        for pend_img in entry.pop("pending_vae_paths", []) or []:
            pil_v = _load_pil(pend_img)
            _update_image_ctx(main_ctx, pil_v, update_vae=True, update_vit=False)
            if cfg_text_ctx is not None:
                _update_image_ctx(cfg_text_ctx, pil_v, update_vae=True, update_vit=False)
        height = int(cmd.get("height", 512))
        width = int(cmd.get("width", 512))
        steps = int(cmd.get("dit_steps", 20))
        seed = int(cmd.get("seed", 52))
        prompt = cmd.get("prompt", "")
        want_pil = bool(cmd.get("return_pil", False))
        save_tmp_path = cmd.get("save_tmp_image_path")

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

        # Naive offload baseline: copy Generation Expert synchronously from unpinned CPU memory
        if policy.use_naive_offload and cpu_naive_weights:
            for layer in pipeline.bagel.language_model.model.layers:
                for pname, p in layer.named_parameters():
                    key = f"{id(layer)}.{pname}"
                    if key in cpu_naive_weights:
                        p.data = cpu_naive_weights[key].to(device)
            torch.cuda.synchronize(device)

        # Hook Step-0 only on HalfNodeH2DPacer so flock releases immediately once Step-1 prefetch completes
        pacer_cm = None
        orig_fwd_for_pacer = None
        if (
            pacer is not None
            and pager is not None
            and vmm is not None
            and vmm.expert_resident_count(pager.expert_id) < pager.num_layers
        ):
            pacer_cm = pacer.acquire_process_slot(rank)
            pacer_cm.__enter__()
            orig_fwd_for_pacer = pipeline.bagel.forward
            step0_released = {"done": False}

            def _pacer_step0_wrap(*a, **kw):
                res = orig_fwd_for_pacer(*a, **kw)
                if not step0_released["done"] and pacer_cm is not None:
                    torch.cuda.synchronize(device)
                    pacer_cm.__exit__(None, None, None)
                    step0_released["done"] = True
                return res

            pipeline.bagel.forward = _pacer_step0_wrap

        # Active mid-DiT preemption tracking for injected EXEC_AR_PREEMPT messages
        inflight_ar_meta: dict[int, tuple[str, str, dict[str, Any], float]] = {}

        def _poll_external_ar(now_s: float) -> list[ArDecodeTask]:
            tasks: list[ArDecodeTask] = []
            prev_g = None
            if pager is not None:
                prev_g = pager._active_guard
                pager._active_guard = False
            try:
                while True:
                    try:
                        ar_msg = ar_preempt_q.get_nowait()
                    except queue.Empty:
                        break
                    ar_entry = _prepare_entry_for_cmd(ar_msg)
                    ar_ctx = ar_entry["main_ctx"]
                    st = pipeline.bagel.prepare_start_tokens(
                        ar_ctx["kv_lens"], ar_ctx["ropes"], pipeline.new_token_ids
                    )
                    task = ArDecodeTask(
                        req_id=str(ar_msg["cmd_id"]),
                        kv_cache=ar_ctx["past_key_values"],
                        curr_tokens=st["packed_start_tokens"].to(device),
                        pos_ids=st["packed_query_position_ids"].to(device),
                        max_tokens=int(ar_msg.get("max_ar_tokens", 32)),
                        end_token_id=pipeline.new_token_ids["eos_token_id"],
                        tpot_slo_ms=float(ar_msg.get("tpot_slo_ms", policy.ar_tpot_slo_ms)),
                        arrival_time_s=float(ar_msg.get("enqueue_time_s", now_s)),
                    )
                    task.last_step_time_s = task.arrival_time_s
                    inflight_ar_meta[id(task)] = (
                        str(ar_msg["cmd_id"]),
                        str(ar_msg["ctx_id"]),
                        ar_entry,
                        float(ar_msg.get("enqueue_time_s", now_s)),
                    )
                    tasks.append(task)
            finally:
                if pager is not None and prev_g is not None:
                    pager._active_guard = prev_g
            return tasks

        def _flush_completed_ar() -> None:
            if sched is None:
                return
            while sched.completed_ar_tasks:
                t = sched.completed_ar_tasks.pop(0)
                meta_tuple = inflight_ar_meta.pop(id(t), None)
                if meta_tuple is None:
                    continue
                c_id, ar_ctx_id, ar_entry, _ = meta_tuple
                ar_ctx = ar_entry["main_ctx"]
                ar_ctx["past_key_values"] = t.kv_cache
                n_tok = len(t.generated_token_ids)
                ar_ctx["kv_lens"] = [kl + n_tok for kl in ar_ctx["kv_lens"]]
                ar_ctx["ropes"] = [r + n_tok for r in ar_ctx["ropes"]]
                decoded = pipeline.tokenizer.decode(t.generated_token_ids) if t.generated_token_ids else ""
                ack_q.put(
                    {
                        "cmd_id": c_id,
                        "rank": rank,
                        "status": "ok",
                        "ctx_id": ar_ctx_id,
                        "text": decoded.split("<|im_end|>")[0].split("<|im_start|>")[-1],
                        "tokens": n_tok,
                        "ttft_ms": round(t.ttft_ms(), 2),
                        "tpots_ms": [round(x, 2) for x in t.inter_token_latencies_ms()],
                        "kv_bytes": _entry_nbytes(ar_entry),
                        "preempted_mid_dit": True,
                    }
                )

        if sched is not None:
            sched.reset_telemetry()
            sched.external_arrival_poll = _poll_external_ar

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                prep = pipeline._forward_single(
                    {"prompt": prompt, "modalities": ["image"]},
                    sampling,
                    prepare_only=True,
                )
                gp = prep["gen_params"]
                img_shape = prep["image_shape"]

                cm_pager = pager.generation_mode(prefetch_all=True) if pager is not None else None
                cm_cfg = cfg_patch.active(reset_stats=True) if cfg_patch is not None else None
                cm_sched = sched.run_session() if sched is not None else None

                if cm_pager is not None:
                    cm_pager.__enter__()
                if cm_cfg is not None:
                    cm_cfg.__enter__()
                if cm_sched is not None:
                    cm_sched.__enter__()
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        latents, *_ = pipeline.bagel.generate_image(
                            past_key_values=prep["gen_context"]["past_key_values"],
                            cfg_text_past_key_values=prep["cfg_text_context"]["past_key_values"],
                            cfg_img_past_key_values=prep["cfg_img_context"]["past_key_values"],
                            num_timesteps=gp.num_timesteps,
                            timestep_shift=gp.timestep_shift,
                            cfg_text_scale=gp.cfg_text_scale,
                            cfg_img_scale=gp.cfg_img_scale,
                            cfg_interval=gp.cfg_interval,
                            cfg_renorm_min=gp.cfg_renorm_min,
                            cfg_renorm_type=gp.cfg_renorm_type,
                            **prep["generation_input"],
                            cfg_text_packed_position_ids=prep["cfg_text_packed_position_ids"],
                            cfg_img_packed_position_ids=prep["cfg_img_packed_position_ids"],
                            return_trajectory_latents=False,
                            scheduler=pipeline.scheduler,
                            scheduler_kwargs=pipeline.scheduler_kwargs,
                        )
                    if policy.use_module3_cow_loopback:
                        img_u8 = decode_latent_to_u8_gpu(pipeline.bagel, pipeline.vae, latents[0], img_shape)
                        entry["last_img_u8"] = img_u8
                        if want_pil or save_tmp_path:
                            pil_img = Image.fromarray(
                                img_u8.permute(1, 2, 0).contiguous().cpu().numpy()
                            )
                            entry["last_img_pil"] = pil_img
                            if save_tmp_path:
                                Path(save_tmp_path).parent.mkdir(parents=True, exist_ok=True)
                                pil_img.save(save_tmp_path, format="PNG")
                    else:
                        pil_img = pipeline._decode_image_from_latent(
                            pipeline.bagel, pipeline.vae, latents[0], img_shape
                        )
                        entry["last_img_pil"] = pil_img
                        if save_tmp_path:
                            Path(save_tmp_path).parent.mkdir(parents=True, exist_ok=True)
                            pil_img.save(save_tmp_path, format="PNG")
                finally:
                    if cm_sched is not None:
                        try:
                            _flush_completed_ar()
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                sched.drain_remaining_ar_tasks()
                            _flush_completed_ar()
                        except torch.cuda.OutOfMemoryError as ar_oom:
                            oom_count += 1
                            torch.cuda.empty_cache()
                            for c_id, _, _, _ in list(inflight_ar_meta.values()):
                                ack_q.put({"cmd_id": c_id, "rank": rank, "status": "oom", "error": str(ar_oom)[:200]})
                            inflight_ar_meta.clear()
                        except Exception as ar_exc:
                            for c_id, _, _, _ in list(inflight_ar_meta.values()):
                                ack_q.put(
                                    {
                                        "cmd_id": c_id,
                                        "rank": rank,
                                        "status": "error",
                                        "error": f"{type(ar_exc).__name__}: {ar_exc}",
                                    }
                                )
                            inflight_ar_meta.clear()
                        finally:
                            for c_id, _, _, _ in list(inflight_ar_meta.values()):
                                ack_q.put({"cmd_id": c_id, "rank": rank, "status": "oom", "error": "aborted mid-dit"})
                            inflight_ar_meta.clear()
                            cm_sched.__exit__(None, None, None)
                    if cm_cfg is not None:
                        cm_cfg.__exit__(None, None, None)
                    if cm_pager is not None:
                        cm_pager.__exit__(None, None, None)
        finally:
            if orig_fwd_for_pacer is not None:
                pipeline.bagel.forward = orig_fwd_for_pacer
            if pacer_cm is not None:
                try:
                    pacer_cm.__exit__(None, None, None)
                except Exception:
                    pass
            if sched is not None:
                sched.external_arrival_poll = None

        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Naive offload baseline: drop GPU generation weights immediately after DiT
        if policy.use_naive_offload and cpu_naive_weights:
            for layer in pipeline.bagel.language_model.model.layers:
                for pname, p in layer.named_parameters():
                    key = f"{id(layer)}.{pname}"
                    if key in cpu_naive_weights:
                        p.data = torch.empty(0, dtype=p.dtype, device=device)
            torch.cuda.empty_cache()

        _update_mem_hwm()
        res_alpha = (
            vmm.expert_resident_count(pager.expert_id) / float(pager.num_layers)
            if (pager is not None and vmm is not None)
            else policy.initial_alphas.get(rank, 1.0)
        )
        return {
            "status": "ok",
            "ctx_id": ctx_id,
            "elapsed_ms": round(elapsed_ms, 2),
            "kv_bytes": _entry_nbytes(entry),
            "resident_alpha": round(res_alpha, 3),
            "yield_count": sched.yield_count if sched is not None else 0,
        }

    # -------------------- Startup & Model Init --------------------
    try:
        pipeline = load_bagel_pipeline(model_dir, device_index=rank)
        loopback = GpuInPlaceLoopback(pipeline)

        # Static fission Und ranks drop unused generation weights; Gen ranks retain full MoT for text_indices
        if policy.routing_mode == "static_nccl_fission" and role == "und":
            for layer in pipeline.bagel.language_model.model.layers:
                for pname, p in layer.named_parameters():
                    if is_generation_param_name(pname):
                        p.data = torch.empty(0, dtype=p.dtype, device=device)
            gc.collect()
            torch.cuda.empty_cache()

        if policy.use_naive_offload:
            for layer in pipeline.bagel.language_model.model.layers:
                for pname, p in layer.named_parameters():
                    if is_generation_param_name(pname):
                        key = f"{id(layer)}.{pname}"
                        cpu_naive_weights[key] = p.detach().contiguous().cpu()
                        p.data = torch.empty(0, dtype=p.dtype, device=device)
            gc.collect()
            torch.cuda.empty_cache()

        if policy.use_expert_pager:
            pager = BagelExpertPager(
                pipeline,
                role="generation",
                numa_node=numa_node,
                super_handle_mib=policy.super_handle_mib,
                wrap_vae=policy.use_scoped_vae,
            )
            if policy.use_shared_numa_mirror:
                grp = topo.half_group_of(rank)
                is_leader = rank == grp.gpu_ids[0]
                if is_leader:
                    shared_mirror = SharedNumaMemfdMirror.create(
                        nbytes=pager.total_mirror_bytes,
                        numa_node=numa_node,
                        name=f"ms_eng_numa{numa_node}_{os.getpid()}",
                    )
                    pager.offload_to_shared_mirror(shared_mirror, populate=True)
                    ack_q.put(
                        {
                            "op": "INIT_LEADER_MIRROR",
                            "rank": rank,
                            "numa_node": numa_node,
                            "pid": os.getpid(),
                            "fd": shared_mirror.fd,
                            "nbytes": shared_mirror.nbytes,
                            "inode": shared_mirror.inode,
                        }
                    )
                else:
                    ack_q.put({"op": "INIT_FOLLOWER_READY", "rank": rank, "numa_node": numa_node})

                # Wait for coordinator to broadcast leader fd table
                attach_msg = main_cmd_q.get()
                if attach_msg.get("op") != "ATTACH_SHARED_MIRROR":
                    raise RuntimeError(f"unexpected init message: {attach_msg}")
                if not is_leader:
                    info = attach_msg["leaders"][str(numa_node)]
                    shared_mirror = SharedNumaMemfdMirror.attach_proc_fd(
                        leader_pid=int(info["pid"]),
                        leader_fd=int(info["fd"]),
                        nbytes=int(info["nbytes"]),
                        numa_node=numa_node,
                    )
                    pager.offload_to_shared_mirror(shared_mirror, populate=False)
            else:
                pager.offload_to_host_mirror()

        if policy.use_tripool_vmm:
            vmm = TriPoolVMM(
                device=rank,
                super_pool_mib=policy.super_pool_mib,
                fine_kv_pool_mib=policy.fine_kv_pool_mib,
                super_handle_mib=policy.super_handle_mib,
                transient_va_gib=policy.transient_va_gib,
                kv_va_gib=policy.kv_va_gib,
                allow_overflow=policy.allow_vmm_overflow,
            )
            if pager is not None:
                pager.bind_vmm(vmm, initial_alpha=policy.initial_alphas.get(rank, 0.50))

        # Receive NCCL unique ID from coordinator and initialize full-mesh DirectNcclMesh
        ack_q.put({"op": "READY_FOR_NCCL", "rank": rank})
        nccl_msg = main_cmd_q.get()
        if nccl_msg.get("op") != "INIT_NCCL_MESH":
            raise RuntimeError(f"expected INIT_NCCL_MESH, got {nccl_msg}")
        if nccl_msg.get("uid_bytes") is not None and policy.num_gpus >= 2:
            nccl_mesh = DirectNcclMesh(
                rank=rank,
                world_size=policy.num_gpus,
                uid_bytes=nccl_msg["uid_bytes"],
                device_index=rank,
            )

        if policy.use_module3_cow_loopback:
            cfg_patch = install_shared_prefix_bagel_hooks(pipeline, min_shared_tokens=32)
        if policy.preemption_mode != "non_preemptive":
            sched = UnifiedPreemptiveScheduler(
                pipeline,
                mode=policy.preemption_mode,
                tpot_slo_ms=policy.ar_tpot_slo_ms,
                guard_ms=4.0,
                init_layer_ms=10.5,
                init_ar_step_ms=22.0,
            )

        # Warm up kernels & scheduler on non-gen / non-und capabilities safely
        warm_num_layers = pipeline.bagel.config.llm_config.num_hidden_layers
        if role != "gen":
            w_ctx = _new_empty_ctx(warm_num_layers, NaiveCache)
            _update_text_ctx(w_ctx, "Warmup prompt.")
            _ = _run_ar_direct(w_ctx, max_tokens=2)
            del w_ctx
        if role != "und" and not policy.use_naive_offload:
            w_main = _new_empty_ctx(warm_num_layers, NaiveCache)
            w_img = _new_empty_ctx(warm_num_layers, NaiveCache)
            _update_text_ctx(w_main, "Warmup.")
            _update_text_ctx(w_img, "Warmup.")
            ctx_store["__warm__"] = {
                "main_ctx": w_main,
                "cfg_text_ctx": None,
                "cfg_img_ctx": w_img,
                "cfg_text_prefix_len": 0,
                "last_img_u8": None,
                "last_img_pil": None,
            }
            _ = _run_dit_exec({"ctx_id": "__warm__", "height": 512, "width": 512, "dit_steps": 2, "seed": 1, "prompt": "Warmup."})
            ctx_store.pop("__warm__", None)
            if pager is not None:
                pager.set_residency_alpha(policy.initial_alphas.get(rank, 0.50))

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        max_reserved_mib = float(torch.cuda.memory_reserved(device) / MIB)
        ack_q.put(
            {
                "op": "WORKER_READY",
                "rank": rank,
                "resident_alpha": (
                    vmm.expert_resident_count(pager.expert_id) / float(pager.num_layers)
                    if (pager is not None and vmm is not None)
                    else policy.initial_alphas.get(rank, 1.0)
                ),
                "reserved_mib": round(max_reserved_mib, 2),
                "mirror_inode": shared_mirror.inode if shared_mirror is not None else None,
            }
        )

        # -------------------- Main Command Loop --------------------
        while True:
            # First drain any leftover ar_preempt_q requests that arrived right as DiT finished
            if not ar_preempt_q.empty():
                ar_msg = ar_preempt_q.get_nowait()
                try:
                    ar_entry = _prepare_entry_for_cmd(ar_msg)
                    ar_res = _run_ar_direct(ar_entry["main_ctx"], int(ar_msg.get("max_ar_tokens", 32)))
                    _update_mem_hwm()
                    ack_q.put(
                        {
                            "cmd_id": ar_msg["cmd_id"],
                            "rank": rank,
                            "status": "ok",
                            "ctx_id": ar_msg["ctx_id"],
                            "kv_bytes": _entry_nbytes(ar_entry),
                            "preempted_mid_dit": False,
                            **ar_res,
                        }
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    oom_count += 1
                    torch.cuda.empty_cache()
                    ack_q.put({"cmd_id": ar_msg["cmd_id"], "rank": rank, "status": "oom", "error": str(exc)[:200]})
                except Exception as exc:
                    ack_q.put(
                        {
                            "cmd_id": ar_msg["cmd_id"],
                            "rank": rank,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                continue

            try:
                cmd = main_cmd_q.get(timeout=0.02)
            except queue.Empty:
                continue

            op = cmd.get("op")
            cmd_id = cmd.get("cmd_id")
            if op == "SHUTDOWN":
                ack_q.put({"cmd_id": cmd_id, "rank": rank, "status": "ok"})
                break

            try:
                if op == "EXEC_AR":
                    entry = _prepare_entry_for_cmd(cmd)
                    ar_res = _run_ar_direct(entry["main_ctx"], int(cmd.get("max_ar_tokens", 32)))
                    _update_mem_hwm()
                    ack_q.put(
                        {
                            "cmd_id": cmd_id,
                            "rank": rank,
                            "status": "ok",
                            "ctx_id": cmd["ctx_id"],
                            "kv_bytes": _entry_nbytes(entry),
                            **ar_res,
                        }
                    )
                elif op == "PREP_DIT":
                    entry = _prepare_entry_for_cmd(cmd)
                    _update_mem_hwm()
                    ack_q.put(
                        {
                            "cmd_id": cmd_id,
                            "rank": rank,
                            "status": "ok",
                            "ctx_id": cmd["ctx_id"],
                            "kv_bytes": _entry_nbytes(entry),
                        }
                    )
                elif op == "EXEC_DIT":
                    dit_res = _run_dit_exec(cmd)
                    ack_q.put({"cmd_id": cmd_id, "rank": rank, **dit_res})
                elif op == "PACK_CTX":
                    ctx_id = cmd["ctx_id"]
                    entry = ctx_store[ctx_id]
                    meta, flat = _pack_multi_ctx(
                        entry["main_ctx"],
                        entry.get("cfg_text_ctx"),
                        entry.get("cfg_img_ctx"),
                        entry.get("cfg_text_prefix_len", 0),
                        include_cfg_text_dup=bool(cmd.get("include_cfg_text_dup", False)),
                        device=device,
                        pending_vae_paths=entry.get("pending_vae_paths"),
                    )
                    packed_id = f"{ctx_id}__{cmd_id}"
                    packed_store[packed_id] = flat
                    ack_q.put({"cmd_id": cmd_id, "rank": rank, "status": "ok", "packed_id": packed_id, "meta": meta})
                elif op == "SEND_NCCL":
                    assert nccl_mesh is not None
                    flat = packed_store.pop(cmd["packed_id"])
                    ship_ms = nccl_mesh.send_u8(flat, dst_rank=int(cmd["dst_rank"]))
                    nbytes = int(flat.numel())
                    del flat
                    ack_q.put({"cmd_id": cmd_id, "rank": rank, "status": "ok", "ship_ms": ship_ms, "bytes": nbytes})
                elif op == "RECV_NCCL":
                    assert nccl_mesh is not None
                    meta = cmd["meta"]
                    flat, ship_ms = nccl_mesh.recv_u8(int(meta["total_bytes"]), src_rank=int(cmd["src_rank"]))
                    entry = _unpack_multi_ctx(
                        meta, flat, NaiveCache, use_cow_rebind=policy.use_module3_cow_loopback
                    )
                    del flat
                    new_ctx_id = cmd["new_ctx_id"]
                    ctx_store[new_ctx_id] = entry
                    _alloc_kv_vmm_if_needed(new_ctx_id, entry["main_ctx"])
                    _update_mem_hwm()
                    ack_q.put(
                        {
                            "cmd_id": cmd_id,
                            "rank": rank,
                            "status": "ok",
                            "ship_ms": ship_ms,
                            "bytes": int(meta["total_bytes"]),
                        }
                    )
                elif op == "SEND_SHM":
                    entry = ctx_store[cmd["ctx_id"]]
                    t_s0 = time.perf_counter()
                    meta, flat = _pack_multi_ctx(
                        entry["main_ctx"],
                        entry.get("cfg_text_ctx"),
                        entry.get("cfg_img_ctx"),
                        entry.get("cfg_text_prefix_len", 0),
                        include_cfg_text_dup=True,
                        device=device,
                        pending_vae_paths=entry.get("pending_vae_paths"),
                    )
                    cpu_bytes = flat.cpu().numpy().tobytes()
                    del flat
                    shm_name = cmd["shm_name"]
                    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=max(1, len(cpu_bytes)))
                    if cpu_bytes:
                        shm.buf[: len(cpu_bytes)] = cpu_bytes
                    shm.close()
                    ser_ms = (time.perf_counter() - t_s0) * 1000.0
                    ack_q.put(
                        {
                            "cmd_id": cmd_id,
                            "rank": rank,
                            "status": "ok",
                            "shm_name": shm_name,
                            "meta": meta,
                            "serialize_ms": round(ser_ms, 2),
                            "bytes": len(cpu_bytes),
                        }
                    )
                elif op == "RECV_SHM":
                    t_r0 = time.perf_counter()
                    shm = shared_memory.SharedMemory(name=cmd["shm_name"], create=False)
                    try:
                        raw = bytes(shm.buf[: int(cmd["meta"]["total_bytes"])])
                    finally:
                        shm.close()
                        try:
                            shm.unlink()
                        except FileNotFoundError:
                            pass
                    flat = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
                    entry = _unpack_multi_ctx(cmd["meta"], flat, NaiveCache, use_cow_rebind=False)
                    del flat
                    ctx_store[cmd["new_ctx_id"]] = entry
                    _update_mem_hwm()
                    ack_q.put(
                        {
                            "cmd_id": cmd_id,
                            "rank": rank,
                            "status": "ok",
                            "recv_ms": round((time.perf_counter() - t_r0) * 1000.0, 2),
                        }
                    )
                elif op == "SET_ALPHA":
                    if pager is not None and vmm is not None:
                        cnt = pager.set_residency_alpha(float(cmd["alpha"]))
                        ack_q.put({"cmd_id": cmd_id, "rank": rank, "status": "ok", "resident_layers": cnt})
                    else:
                        ack_q.put({"cmd_id": cmd_id, "rank": rank, "status": "ok"})
                elif op == "RELEASE_CTX":
                    cid = cmd["ctx_id"]
                    ctx_store.pop(cid, None)
                    _free_kv_vmm(cid)
                    torch.cuda.empty_cache()
                    ack_q.put({"cmd_id": cmd_id, "rank": rank, "status": "ok"})
                elif op == "GET_STATS":
                    _update_mem_hwm()
                    vmm_stats = vmm.stats_dict() if vmm is not None else {}
                    ack_q.put(
                        {
                            "cmd_id": cmd_id,
                            "rank": rank,
                            "status": "ok",
                            "peak_reserved_mib": round(max_reserved_mib, 2),
                            "oom_count": oom_count,
                            "vmm_stats": vmm_stats,
                            "active_contexts": len(ctx_store),
                        }
                    )
                else:
                    ack_q.put({"cmd_id": cmd_id, "rank": rank, "status": "error", "error": f"unknown op: {op}"})
            except torch.cuda.OutOfMemoryError as exc:
                oom_count += 1
                gc.collect()
                torch.cuda.empty_cache()
                ack_q.put({"cmd_id": cmd_id, "rank": rank, "status": "oom", "error": str(exc)[:300]})
            except Exception as exc:
                ack_q.put(
                    {
                        "cmd_id": cmd_id,
                        "rank": rank,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-1200:],
                    }
                )
    except Exception as exc:
        ack_q.put(
            {
                "op": "WORKER_FATAL",
                "rank": rank,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-1500:],
            }
        )
    finally:
        stop_inbox.set()
        try:
            inbox_thread.join(timeout=1.0)
        except Exception:
            pass
        if cfg_patch is not None:
            try:
                cfg_patch.uninstall()
            except Exception:
                pass
        if nccl_mesh is not None:
            nccl_mesh.close()
        if pager is not None:
            pager.close()
        if vmm is not None:
            vmm.shutdown()
        if shared_mirror is not None:
            shared_mirror.close()
        ctx_store.clear()
        packed_store.clear()
        cpu_naive_weights.clear()
        gc.collect()
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
