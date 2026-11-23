"""Deep `BagelCPTGAdapter` (`tight_mot`) integrating `VmmNaiveCache`, `LayerExpertPager`, and `BagelExecutionInterceptor`."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import gc
import io
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

from morphserve.kernels import (
    GpuInPlaceLoopback,
    decode_latent_to_u8_gpu,
    rebind_cow_prefix,
)
from morphserve.vmm import (
    ExpertPartitionSpec,
    LayerExpertPager,
    NaiveHostOffloader,
    NumaMemfdMirror,
    TriPoolVMM,
    VmmNaiveCache,
    drop_partition_weights,
)
from morphserve.worker import ArDecodeTask, BagelExecutionInterceptor, UnifiedPreemptiveScheduler

from .base import CPTGModelAdapter, PackedTransferState, compute_vmm_available_kv_bytes, is_oom_exception

MIB = 1024**2
GIB = 1024**3


def apply_vllm_omni_mm_limit() -> None:
    from vllm_omni.model_executor.models.bagel.bagel import OmniBagelProcessingInfo

    max_images = int(os.environ.get("MORPHSERVE_MAX_IMAGE_ITEMS", "16"))
    if max_images < 1:
        raise ValueError("MORPHSERVE_MAX_IMAGE_ITEMS must be positive")

    def get_supported_mm_limits(self):  # noqa: ARG001
        return {"image": max_images, "img2img": 1}

    OmniBagelProcessingInfo.get_supported_mm_limits = get_supported_mm_limits
    print(
        json.dumps({"morphserve_bagel_mm_adapter": True, "supported_mm_limits": {"image": max_images, "img2img": 1}}),
        flush=True,
    )


def load_bagel_pipeline(model_dir: Path, device_index: int = 0) -> Any:
    """Canonical BAGEL-7B-MoT single-GPU `vllm_omni` pipeline loader."""
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
    from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
    from vllm_omni.platforms import current_omni_platform

    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    current_omni_platform.set_device(device)
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
            distributed_init_method=f"file:///tmp/morphserve-bagel-{os.getpid()}-{time.time_ns()}",
            local_rank=device_index,
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
        pipeline = loader.load_model(load_device=f"cuda:{device_index}")
    pipeline.eval()
    return pipeline


def _tensor_logical_bytes(t: Any) -> int:
    if t is None:
        return 0
    return int(t.numel() * t.element_size())


def _peek_cache_slot(slot_map: Any, layer_idx: int) -> Any:
    """Read `slot_map[layer_idx]` without triggering `VmmNaiveCache.ensure_resident()` or LRU mutation."""
    if isinstance(slot_map, dict):
        return dict.get(slot_map, layer_idx)
    return slot_map[layer_idx]


def _truncate_bagel_cache(pkv: Any, target_seq_len: int) -> None:
    """Roll back a partially updated `NaiveCache` or `VmmNaiveCache` so all layers match `target_seq_len`."""
    if pkv is None:
        return
    if callable(getattr(pkv, "truncate", None)):
        pkv.truncate(target_seq_len)
        return
    target = max(0, int(target_seq_len))
    num_layers = int(pkv.num_layers)
    for l in range(num_layers):
        for slot_map in (pkv.key_cache, pkv.value_cache):
            cur = _peek_cache_slot(slot_map, l)
            if target == 0:
                slot_map[l] = None
            elif cur is not None and int(cur.shape[0]) > target:
                slot_map[l] = cur[:target]


def bagel_ctx_nbytes(ctx: dict[str, Any] | None) -> int:
    """Compute deduplicated logical GPU bytes occupied by one BAGEL KV context."""
    if ctx is None:
        return 0
    pkv = ctx.get("past_key_values")
    if pkv is None:
        return 0
    if bool(getattr(pkv, "is_swapped_out", False)) or bool(getattr(pkv, "_is_swapped", False)):
        return int(getattr(pkv, "swapped_bytes", 0))
    num_layers = int(pkv.num_layers)
    seen: dict[int, int] = {}
    for l in range(num_layers):
        for t in (_peek_cache_slot(pkv.key_cache, l), _peek_cache_slot(pkv.value_cache, l)):
            if t is not None and hasattr(t, "data_ptr"):
                ptr = int(t.data_ptr())
                seen[ptr] = max(seen.get(ptr, 0), _tensor_logical_bytes(t))
    return sum(seen.values())


def bagel_entry_nbytes(entry: dict[str, Any]) -> int:
    """Compute deduplicated logical bytes across `main`, `cfg_text`, and `cfg_img` branches (or swapped host bundle)."""
    swapped = entry.get("swapped_host_bundle")
    if swapped is not None:
        return int(swapped.get("total_bytes", 0))
    swapped_vmm_b = 0
    seen: dict[int, int] = {}
    for key in ("main_ctx", "cfg_text_ctx", "cfg_img_ctx"):
        ctx = entry.get(key)
        if ctx is None or ctx.get("past_key_values") is None:
            continue
        pkv = ctx["past_key_values"]
        if bool(getattr(pkv, "is_swapped_out", False)) or bool(getattr(pkv, "_is_swapped", False)):
            swapped_vmm_b += int(getattr(pkv, "swapped_bytes", 0))
            continue
        num_layers = int(pkv.num_layers)
        for l in range(num_layers):
            for t in (_peek_cache_slot(pkv.key_cache, l), _peek_cache_slot(pkv.value_cache, l)):
                if t is not None and hasattr(t, "data_ptr"):
                    ptr = int(t.data_ptr())
                    seen[ptr] = max(seen.get(ptr, 0), _tensor_logical_bytes(t))
    u8 = entry.get("last_img_u8")
    img_b = _tensor_logical_bytes(u8) if u8 is not None else 0
    return swapped_vmm_b + sum(seen.values()) + img_b


def pack_bagel_kv_context(ctx: dict[str, Any], device: Any = None) -> tuple[dict[str, Any], Any]:
    """Pack one BAGEL `NaiveCache` / `VmmNaiveCache` context into `(meta_dict, flat_u8_cuda_tensor)`."""
    import torch

    pkv = ctx["past_key_values"]
    num_layers = int(pkv.num_layers)
    get_k = pkv.key_cache.get if isinstance(pkv.key_cache, dict) else (lambda i: pkv.key_cache[i])
    get_v = pkv.value_cache.get if isinstance(pkv.value_cache, dict) else (lambda i: pkv.value_cache[i])
    sample_k = next((get_k(l) for l in range(num_layers) if get_k(l) is not None), None)
    if device is None:
        device = sample_k.device if sample_k is not None else torch.device(f"cuda:{torch.cuda.current_device()}")

    layer_specs: list[list[Any] | None] = []
    chunks: list[Any] = []
    for l in range(num_layers):
        k = get_k(l)
        v = get_v(l)
        if k is None or v is None:
            layer_specs.append(None)
            continue
        layer_specs.append([list(k.shape), list(v.shape), str(k.dtype).replace("torch.", "")])
        chunks.append(k.to(device).contiguous().view(torch.uint8).reshape(-1))
        chunks.append(v.to(device).contiguous().view(torch.uint8).reshape(-1))
    flat = torch.cat(chunks, dim=0) if chunks else torch.empty((0,), dtype=torch.uint8, device=device)
    meta = {
        "num_layers": num_layers,
        "kv_lens": list(ctx["kv_lens"]),
        "ropes": list(ctx["ropes"]),
        "total_bytes": int(flat.numel()),
        "layers": layer_specs,
    }
    return meta, flat


def unpack_bagel_kv_context(
    meta: dict[str, Any],
    flat_u8: Any,
    cache_factory: Callable[[int], Any],
) -> dict[str, Any]:
    """Reconstruct one BAGEL KV context from `(meta_dict, flat_u8_cuda_tensor)` into `cache_factory(num_layers)`."""
    import torch

    num_layers = int(meta["num_layers"])
    pkv = cache_factory(num_layers)
    off = 0
    for l, spec in enumerate(meta["layers"]):
        if spec is None:
            continue
        k_shape, v_shape, dt_name = spec
        dtype = getattr(torch, dt_name)
        elem_sz = int(torch.empty((), dtype=dtype).element_size())
        k_bytes = int(torch.Size(k_shape).numel() * elem_sz)
        v_bytes = int(torch.Size(v_shape).numel() * elem_sz)
        pkv.key_cache[l] = flat_u8[off : off + k_bytes].view(dtype).reshape(k_shape).clone()
        off += k_bytes
        pkv.value_cache[l] = flat_u8[off : off + v_bytes].view(dtype).reshape(v_shape).clone()
        off += v_bytes
    return {"kv_lens": list(meta["kv_lens"]), "ropes": list(meta["ropes"]), "past_key_values": pkv}


def _slice_prefix_ctx(
    main_ctx: dict[str, Any],
    prefix_len: int,
    naive_cache_cls: Any,
    *,
    ropes: list[int] | None = None,
    kv_lens: list[int] | None = None,
    clone_slices: bool = False,
) -> dict[str, Any] | None:
    if prefix_len <= 0:
        return None
    pkv = main_ctx["past_key_values"]
    num_layers = int(pkv.num_layers)
    sub = naive_cache_cls(num_layers)
    for l in range(num_layers):
        k = pkv.key_cache[l]
        v = pkv.value_cache[l]
        if k is None:
            sub.key_cache[l] = None
        else:
            ks = k[:prefix_len]
            sub.key_cache[l] = ks.clone() if clone_slices else ks
        if v is None:
            sub.value_cache[l] = None
        else:
            vs = v[:prefix_len]
            sub.value_cache[l] = vs.clone() if clone_slices else vs
    if not clone_slices:
        sub._cow_shared_prefix_len = prefix_len
    return {
        "kv_lens": list(kv_lens) if kv_lens is not None else [prefix_len],
        "ropes": list(ropes) if ropes is not None else list(main_ctx["ropes"]),
        "past_key_values": sub,
    }


def pack_bagel_multi_context(
    entry: dict[str, Any],
    include_cfg_text_dup: bool,
    device: Any,
) -> PackedTransferState:
    """Pack multi-turn BAGEL session (`main_ctx`, CFG branches/metadata, and loopback image state)."""
    import numpy as np
    import torch

    main_ctx = entry["main_ctx"]
    cfg_text_ctx = entry.get("cfg_text_ctx")
    cfg_img_ctx = entry.get("cfg_img_ctx")
    cfg_text_prefix_len = int(entry.get("cfg_text_prefix_len", 0))
    pending_vae_paths = list(entry.get("pending_vae_paths") or [])
    pending_vae_u8 = bool(entry.get("pending_vae_u8", False))
    pending_vae_pil = bool(entry.get("pending_vae_pil", False))
    pending_dit_turn = dict(entry["pending_dit_turn"]) if entry.get("pending_dit_turn") is not None else None

    has_cfg_text = cfg_text_ctx is not None and cfg_text_ctx["past_key_values"].key_cache[0] is not None
    cfg_text_ropes = list(cfg_text_ctx["ropes"]) if has_cfg_text else None
    cfg_text_kv_lens = list(cfg_text_ctx["kv_lens"]) if has_cfg_text else None

    named_items: list[tuple[str, dict[str, Any]]] = [("main", main_ctx)]
    if include_cfg_text_dup and has_cfg_text:
        named_items.append(("cfg_text", cfg_text_ctx))
    if cfg_img_ctx is not None and cfg_img_ctx["past_key_values"].key_cache[0] is not None:
        named_items.append(("cfg_img", cfg_img_ctx))

    sections: dict[str, Any] = {}
    flat_chunks: list[Any] = []
    offset_bytes = 0
    for name, c in named_items:
        m, fl = pack_bagel_kv_context(c, device=device)
        nbytes = int(fl.numel())
        sections[name] = {"meta": m, "offset": offset_bytes, "nbytes": nbytes}
        if nbytes > 0:
            flat_chunks.append(fl)
            offset_bytes += nbytes

    image_sections: dict[str, Any] = {}
    last_u8 = entry.get("last_img_u8")
    last_pil = entry.get("last_img_pil")
    if last_u8 is not None and torch.is_tensor(last_u8):
        u8_c = last_u8.to(device=device, dtype=torch.uint8).contiguous()
        nb = int(u8_c.numel())
        image_sections["last_img_u8"] = {
            "shape": list(u8_c.shape),
            "offset": offset_bytes,
            "nbytes": nb,
        }
        if nb > 0:
            flat_chunks.append(u8_c.reshape(-1))
            offset_bytes += nb
    elif last_pil is not None:
        arr = np.array(last_pil.convert("RGB"), dtype=np.uint8, copy=True)
        pil_u8 = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device=device, dtype=torch.uint8)
        nb = int(pil_u8.numel())
        image_sections["last_img_pil_u8"] = {
            "shape": list(pil_u8.shape),
            "offset": offset_bytes,
            "nbytes": nb,
        }
        if nb > 0:
            flat_chunks.append(pil_u8.reshape(-1))
            offset_bytes += nb

    flat = torch.cat(flat_chunks, dim=0) if flat_chunks else torch.empty((0,), dtype=torch.uint8, device=device)
    total_b = int(flat.numel())
    meta = {
        "total_bytes": total_b,
        "logical_bytes": total_b,
        "cfg_text_prefix_len": cfg_text_prefix_len,
        "cfg_text_ropes": cfg_text_ropes,
        "cfg_text_kv_lens": cfg_text_kv_lens,
        "include_cfg_text_dup": bool(include_cfg_text_dup),
        "pending_vae_paths": pending_vae_paths,
        "pending_vae_u8": pending_vae_u8,
        "pending_vae_pil": pending_vae_pil,
        "pending_dit_turn": pending_dit_turn,
        "sections": sections,
        "image_sections": image_sections,
    }
    return PackedTransferState(meta=meta, payload=flat, logical_bytes=total_b)


def unpack_bagel_multi_context(
    meta: dict[str, Any],
    flat_u8: Any,
    main_cache_factory: Callable[[int], Any],
    aux_cache_factory: Callable[[int], Any],
    use_cow_rebind: bool,
) -> dict[str, Any]:
    """Unpack multi-turn BAGEL session entry, restoring exact `cfg_text` RoPE/prefix and loopback image state."""
    sections = meta["sections"]
    m_sec = sections["main"]
    main_ctx = unpack_bagel_kv_context(
        m_sec["meta"],
        flat_u8[m_sec["offset"] : m_sec["offset"] + m_sec["nbytes"]],
        main_cache_factory,
    )
    cfg_text_ctx = None
    if "cfg_text" in sections:
        ct_sec = sections["cfg_text"]
        cfg_text_ctx = unpack_bagel_kv_context(
            ct_sec["meta"],
            flat_u8[ct_sec["offset"] : ct_sec["offset"] + ct_sec["nbytes"]],
            aux_cache_factory,
        )
        if use_cow_rebind:
            rebind_cow_prefix(main_ctx["past_key_values"], cfg_text_ctx["past_key_values"])
    elif int(meta.get("cfg_text_prefix_len", 0)) > 0:
        cfg_text_ctx = _slice_prefix_ctx(
            main_ctx,
            int(meta["cfg_text_prefix_len"]),
            aux_cache_factory,
            ropes=meta.get("cfg_text_ropes"),
            kv_lens=meta.get("cfg_text_kv_lens"),
            clone_slices=(not use_cow_rebind),
        )

    cfg_img_ctx = None
    if "cfg_img" in sections:
        ci_sec = sections["cfg_img"]
        cfg_img_ctx = unpack_bagel_kv_context(
            ci_sec["meta"],
            flat_u8[ci_sec["offset"] : ci_sec["offset"] + ci_sec["nbytes"]],
            aux_cache_factory,
        )

    last_img_u8 = None
    last_img_pil = None
    img_secs = meta.get("image_sections") or {}
    if "last_img_u8" in img_secs:
        isec = img_secs["last_img_u8"]
        off, nb = int(isec["offset"]), int(isec["nbytes"])
        last_img_u8 = flat_u8[off : off + nb].reshape(isec["shape"]).clone()
    elif "last_img_pil_u8" in img_secs:
        from PIL import Image

        isec = img_secs["last_img_pil_u8"]
        off, nb = int(isec["offset"]), int(isec["nbytes"])
        chw = flat_u8[off : off + nb].reshape(isec["shape"]).detach().cpu().numpy()
        last_img_pil = Image.fromarray(chw.transpose(1, 2, 0))

    return {
        "main_ctx": main_ctx,
        "cfg_text_ctx": cfg_text_ctx,
        "cfg_img_ctx": cfg_img_ctx,
        "cfg_text_prefix_len": int(meta.get("cfg_text_prefix_len", 0)),
        "pending_vae_paths": list(meta.get("pending_vae_paths") or []),
        "pending_vae_u8": bool(meta.get("pending_vae_u8", False)),
        "pending_vae_pil": bool(meta.get("pending_vae_pil", False)),
        "pending_dit_turn": dict(meta["pending_dit_turn"]) if meta.get("pending_dit_turn") is not None else None,
        "last_img_u8": last_img_u8,
        "last_img_pil": last_img_pil,
    }


def _close_ctx_vmm_if_any(ctx: dict[str, Any] | None) -> None:
    if ctx is None:
        return
    pkv = ctx.get("past_key_values")
    if pkv is not None and hasattr(pkv, "close"):
        pkv.close()


def _serialize_ctx_for_host_swap(ctx: dict[str, Any]) -> tuple[dict[str, Any], list[Any], int]:
    pkv = ctx["past_key_values"]
    num_layers = int(pkv.num_layers)
    get_k = pkv.key_cache.get if isinstance(pkv.key_cache, dict) else (lambda i: pkv.key_cache[i])
    get_v = pkv.value_cache.get if isinstance(pkv.value_cache, dict) else (lambda i: pkv.value_cache[i])
    layers: list[dict[str, Any] | None] = []
    tensors: list[Any] = []
    total_b = 0
    for l in range(num_layers):
        k = get_k(l)
        v = get_v(l)
        if k is None or v is None:
            layers.append(None)
            continue
        kb = _tensor_logical_bytes(k)
        vb = _tensor_logical_bytes(v)
        layers.append(
            {
                "k_shape": list(k.shape),
                "v_shape": list(v.shape),
                "dtype": str(k.dtype).replace("torch.", ""),
                "k_bytes": kb,
                "v_bytes": vb,
            }
        )
        tensors.append(k)
        tensors.append(v)
        total_b += kb + vb
    meta = {
        "num_layers": num_layers,
        "kv_lens": list(ctx["kv_lens"]),
        "ropes": list(ctx["ropes"]),
        "layers": layers,
    }
    return meta, tensors, total_b


def pack_bagel_entry_to_pinned_host(
    entry: dict[str, Any],
    *,
    pin_memory: bool | None = None,
) -> dict[str, Any]:
    """Copy an idle BAGEL session's GPU KV into a contiguous Host Pinned DRAM buffer (`cudaHostAlloc`)."""
    import torch

    main_ctx = entry["main_ctx"]
    cfg_text_ctx = entry.get("cfg_text_ctx")
    cfg_img_ctx = entry.get("cfg_img_ctx")
    prefix_len = int(entry.get("cfg_text_prefix_len", 0))

    cfg_text_k0 = _peek_cache_slot(cfg_text_ctx["past_key_values"].key_cache, 0) if cfg_text_ctx is not None else None
    has_cfg_text = cfg_text_k0 is not None
    cfg_text_seq = int(cfg_text_k0.shape[0]) if has_cfg_text else 0
    cfg_text_is_prefix_only = bool(has_cfg_text and prefix_len > 0 and cfg_text_seq == prefix_len)

    branch_items: list[tuple[str, dict[str, Any]]] = [("main_ctx", main_ctx)]
    if has_cfg_text and not cfg_text_is_prefix_only:
        branch_items.append(("cfg_text_ctx", cfg_text_ctx))
    if cfg_img_ctx is not None and _peek_cache_slot(cfg_img_ctx["past_key_values"].key_cache, 0) is not None:
        branch_items.append(("cfg_img_ctx", cfg_img_ctx))

    branch_specs: dict[str, Any] = {}
    src_tensors: list[Any] = []
    total_bytes = 0
    any_cuda = False
    cuda_dev: Any = None
    for bname, bctx in branch_items:
        bmeta, btensors, bbytes = _serialize_ctx_for_host_swap(bctx)
        branch_specs[bname] = {"meta": bmeta, "offset": total_bytes, "nbytes": bbytes}
        total_bytes += bbytes
        for t in btensors:
            src_tensors.append(t)
            if torch.is_tensor(t) and t.is_cuda:
                any_cuda = True
                cuda_dev = t.device

    use_pin = (torch.cuda.is_available() and any_cuda) if pin_memory is None else bool(pin_memory)
    host_u8 = torch.empty((max(1, total_bytes),), dtype=torch.uint8, device="cpu", pin_memory=use_pin)
    cursor = 0
    for t in src_tensors:
        src_u8 = t.detach().contiguous().view(torch.uint8).reshape(-1)
        nb = int(src_u8.numel())
        if nb > 0:
            host_u8[cursor : cursor + nb].copy_(src_u8, non_blocking=use_pin)
            cursor += nb
    if any_cuda and cuda_dev is not None:
        torch.cuda.current_stream(cuda_dev).synchronize()

    return {
        "host_u8": host_u8[:total_bytes],
        "pinned": use_pin,
        "total_bytes": total_bytes,
        "branches": branch_specs,
        "cfg_text_is_prefix_only": cfg_text_is_prefix_only,
        "cfg_text_prefix_len": prefix_len,
        "cfg_text_ropes": list(cfg_text_ctx["ropes"]) if has_cfg_text else None,
        "cfg_text_kv_lens": list(cfg_text_ctx["kv_lens"]) if has_cfg_text else None,
        "pending_dit_turn": dict(entry["pending_dit_turn"]) if entry.get("pending_dit_turn") is not None else None,
    }


def unpack_bagel_entry_from_pinned_host(
    bundle: dict[str, Any],
    entry: dict[str, Any],
    *,
    device: Any,
    main_cache_factory: Callable[[int], Any],
    aux_cache_factory: Callable[[int], Any],
    use_cow_rebind: bool,
) -> dict[str, Any]:
    """Restore a swapped-out BAGEL session from Host Pinned DRAM back into GPU `VmmNaiveCache` / `NaiveCache`."""
    import torch

    host_u8 = bundle["host_u8"]
    branches = bundle["branches"]
    is_cuda_dst = str(device).startswith("cuda") and torch.cuda.is_available()
    non_block = bool(bundle.get("pinned", False)) and is_cuda_dst

    def _restore_branch(bname: str, factory: Callable[[int], Any]) -> dict[str, Any]:
        binfo = branches[bname]
        bmeta = binfo["meta"]
        pkv = factory(int(bmeta["num_layers"]))
        off = int(binfo["offset"])
        for lidx, lspec in enumerate(bmeta["layers"]):
            if lspec is None:
                continue
            dt = getattr(torch, lspec["dtype"])
            kb = int(lspec["k_bytes"])
            vb = int(lspec["v_bytes"])
            k_gpu = torch.empty(lspec["k_shape"], dtype=dt, device=device)
            k_gpu.view(torch.uint8).reshape(-1).copy_(host_u8[off : off + kb], non_blocking=non_block)
            off += kb
            v_gpu = torch.empty(lspec["v_shape"], dtype=dt, device=device)
            v_gpu.view(torch.uint8).reshape(-1).copy_(host_u8[off : off + vb], non_blocking=non_block)
            off += vb
            pkv.key_cache[lidx] = k_gpu
            pkv.value_cache[lidx] = v_gpu
        return {"kv_lens": list(bmeta["kv_lens"]), "ropes": list(bmeta["ropes"]), "past_key_values": pkv}

    main_ctx = _restore_branch("main_ctx", main_cache_factory)
    cfg_text_ctx = None
    if "cfg_text_ctx" in branches:
        cfg_text_ctx = _restore_branch("cfg_text_ctx", aux_cache_factory)
        if use_cow_rebind:
            rebind_cow_prefix(main_ctx["past_key_values"], cfg_text_ctx["past_key_values"])
    elif bundle.get("cfg_text_is_prefix_only") and int(bundle.get("cfg_text_prefix_len", 0)) > 0:
        cfg_text_ctx = _slice_prefix_ctx(
            main_ctx,
            int(bundle["cfg_text_prefix_len"]),
            aux_cache_factory,
            ropes=bundle.get("cfg_text_ropes"),
            kv_lens=bundle.get("cfg_text_kv_lens"),
            clone_slices=(not use_cow_rebind),
        )

    cfg_img_ctx = _restore_branch("cfg_img_ctx", aux_cache_factory) if "cfg_img_ctx" in branches else None
    if is_cuda_dst:
        torch.cuda.current_stream(device).synchronize()

    entry["main_ctx"] = main_ctx
    entry["cfg_text_ctx"] = cfg_text_ctx
    entry["cfg_img_ctx"] = cfg_img_ctx
    if bundle.get("pending_dit_turn") is not None:
        entry["pending_dit_turn"] = dict(bundle["pending_dit_turn"])
    return entry


class BagelCPTGAdapter(CPTGModelAdapter):
    """Production `BAGEL-7B-MoT` (`tight_mot`) adapter integrating `VmmNaiveCache`, `LayerExpertPager` & `BagelExecutionInterceptor`."""

    architecture = "tight_mot"

    def __init__(
        self,
        rank: int,
        policy: Any,
        model_dir: Path,
        bench_root: Path,
        numa_node: int = 0,
    ) -> None:
        import torch
        from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache

        self.rank = int(rank)
        self.policy = policy
        self.model_dir = Path(model_dir)
        self.bench_root = Path(bench_root)
        self.numa_node = int(numa_node)
        self.role = policy.role_of(self.rank)
        self.device = torch.device(f"cuda:{self.rank}")
        if getattr(policy, "gpu_memory_fraction", None) is not None:
            torch.cuda.set_per_process_memory_fraction(float(policy.gpu_memory_fraction), self.device)
        self._naive_cache_cls = NaiveCache

        self.pipeline = load_bagel_pipeline(self.model_dir, device_index=self.rank)
        self.num_expert_layers = int(self.pipeline.bagel.config.llm_config.num_hidden_layers)
        self.loopback = GpuInPlaceLoopback(self.pipeline)
        self.interceptor = BagelExecutionInterceptor.for_pipeline(self.pipeline)

        self.pager: LayerExpertPager | None = None
        self.naive_offloader: NaiveHostOffloader | None = None
        self.vmm: TriPoolVMM | None = None
        self.sched: UnifiedPreemptiveScheduler | None = None
        self.pacer: Any = None

        self._img_cache: dict[str, Any] = {}
        self._ctx_store: dict[str, dict[str, Any]] = {}
        self._active_ctx_pins: dict[str, int] = {}
        self._ctx_last_used_s: dict[str, float] = {}
        self._kv_host_swap_outs: int = 0
        self._kv_host_swap_ins: int = 0
        self._kv_host_pinned_bytes: int = 0
        self._vmm_initial_pool_mib: float = 0.0
        self._max_reserved_mib = 0.0
        self._max_pytorch_reserved_mib = 0.0
        self._calibrated_ar_step_ms: float | None = None
        self._calibrated_layer_ms_by_hw: dict[tuple[int, int], float] = {}

        gen_spec = ExpertPartitionSpec.for_bagel(
            self.pipeline,
            role="generation",
            super_handle_mib=policy.super_handle_mib,
            wrap_vae=policy.use_scoped_vae,
        )
        if policy.routing_mode == "static_nccl_fission" and self.role == "und":
            drop_partition_weights(gen_spec, device=self.device)
        elif policy.use_naive_offload:
            self.naive_offloader = NaiveHostOffloader(gen_spec, device=self.device)
            self.naive_offloader.offload_to_cpu()
        elif policy.use_expert_pager:
            self.pager = LayerExpertPager(gen_spec, numa_node=self.numa_node)
            self.interceptor.bind_pager(self.pager)

    @property
    def total_mirror_bytes(self) -> int:
        return self.pager.total_mirror_bytes if self.pager is not None else 0

    def attach_expert_mirror(self, shared_mirror: NumaMemfdMirror | None, populate: bool = True) -> None:
        if self.pager is None:
            return
        if shared_mirror is not None:
            self.pager.offload_to_shared_mirror(shared_mirror, populate=populate)
        else:
            self.pager.offload_to_host_mirror()

    def _new_main_ctx(self) -> dict[str, Any]:
        if self.vmm is not None:
            return {
                "kv_lens": [0],
                "ropes": [0],
                "past_key_values": VmmNaiveCache.create_for_bagel(
                    self._naive_cache_cls, self.num_expert_layers, self.vmm
                ),
            }
        return {"kv_lens": [0], "ropes": [0], "past_key_values": self._naive_cache_cls(self.num_expert_layers)}

    def _new_aux_ctx(self) -> dict[str, Any]:
        return {"kv_lens": [0], "ropes": [0], "past_key_values": self._naive_cache_cls(self.num_expert_layers)}

    def _vmm_physical_pool_mib(self) -> float:
        if self.vmm is None:
            return 0.0
        st = self.vmm.stats_dict()
        if "pool_total_bytes" in st:
            return float(st["pool_total_bytes"]) / MIB
        if "physical_pool_bytes" in st:
            return float(st["physical_pool_bytes"]) / MIB
        overflow_b = int(st.get("overflow_creates", 0)) * int(
            st.get("super_handle_bytes", int(self.policy.super_handle_mib) * MIB)
        )
        return self._vmm_initial_pool_mib + float(overflow_b) / MIB

    def _sync_ctx_vmm_activity(self, cid: str, *, active: bool) -> None:
        ent = self._ctx_store.get(cid)
        if ent is None:
            return
        m_ctx = ent.get("main_ctx")
        if m_ctx is None:
            return
        pkv = m_ctx.get("past_key_values")
        if pkv is None:
            return
        if active and callable(getattr(pkv, "mark_active", None)):
            pkv.mark_active()
        elif not active and callable(getattr(pkv, "mark_thinking", None)):
            pkv.mark_thinking(trigger_watermark_check=False)

    @contextmanager
    def _pinned_ctx(self, *ctx_ids: str):
        for cid in ctx_ids:
            self._active_ctx_pins[cid] = self._active_ctx_pins.get(cid, 0) + 1
            self._sync_ctx_vmm_activity(cid, active=True)
        try:
            yield
        finally:
            now_s = time.perf_counter()
            for cid in ctx_ids:
                rem = self._active_ctx_pins.get(cid, 1) - 1
                if rem <= 0:
                    self._active_ctx_pins.pop(cid, None)
                    self._sync_ctx_vmm_activity(cid, active=False)
                else:
                    self._active_ctx_pins[cid] = rem
                if cid in self._ctx_store:
                    self._ctx_last_used_s[cid] = now_s

    def available_kv_bytes(self, *, for_dit: bool = False) -> int:
        import torch

        if self.vmm is not None:
            vmm_avail = compute_vmm_available_kv_bytes(
                self.vmm,
                has_pageable_experts=(self.pager is not None),
                for_dit=for_dit,
            )
            vmm_st = self.vmm.stats_dict()
            vmm_includes_overflow = callable(getattr(self.vmm, "kv_free_bytes", None)) or (
                "kv_free_bytes" in vmm_st
            )
            if self.policy.allow_vmm_overflow and torch.cuda.is_available():
                free_drv, total_drv = torch.cuda.mem_get_info(self.device)
                if getattr(self.policy, "gpu_memory_fraction", None) is not None:
                    cap_b = int(total_drv * float(self.policy.gpu_memory_fraction))
                    cur_used_b = int(torch.cuda.memory_reserved(self.device)) + int(self._vmm_physical_pool_mib() * MIB)
                    extra_b = max(0, cap_b - cur_used_b - 512 * MIB)
                    if vmm_includes_overflow:
                        pool_free = int(vmm_st.get("pool_free_bytes", 0))
                        unpinned_exp = int(vmm_st.get("expert_unpinned_bytes", 0))
                        in_pool = pool_free + (0 if for_dit else unpinned_exp)
                        return min(vmm_avail, in_pool + extra_b)
                    return vmm_avail + extra_b
                if not vmm_includes_overflow:
                    return vmm_avail + max(0, int(free_drv) - 1024 * MIB)
            return vmm_avail

        if not torch.cuda.is_available():
            return 0
        free_drv, total_drv = torch.cuda.mem_get_info(self.device)
        if getattr(self.policy, "gpu_memory_fraction", None) is not None:
            quota_b = int(total_drv * float(self.policy.gpu_memory_fraction))
            alloc_b = int(torch.cuda.memory_allocated(self.device))
            return max(0, quota_b - alloc_b - 512 * MIB)
        unused_cached = max(
            0, int(torch.cuda.memory_reserved(self.device)) - int(torch.cuda.memory_allocated(self.device))
        )
        return max(0, int(free_drv) + unused_cached - 1024 * MIB)

    def _swap_out_ctx_to_host(self, cid: str, entry: dict[str, Any]) -> int:
        if entry.get("swapped_host_bundle") is not None or entry.get("main_ctx") is None:
            return 0
        pkv = entry["main_ctx"].get("past_key_values")
        if pkv is None or bool(getattr(pkv, "_is_swapped", False)) or bool(getattr(pkv, "is_swapped_out", False)):
            return 0
        # Prefer VA-preserving native `VmmNaiveCache.swap_out_async()` whenever no auxiliary non-CoW cache is attached
        if (
            self.vmm is not None
            and callable(getattr(pkv, "swap_out_async", None))
            and entry.get("cfg_img_ctx") is None
            and (entry.get("cfg_text_ctx") is None or self.policy.use_module3_cow_loopback)
        ):
            reclaimed = int(pkv.swap_out_async())
            if callable(getattr(self.vmm, "flush_kv_swaps", None)):
                self.vmm.flush_kv_swaps()
            return reclaimed

        bundle = pack_bagel_entry_to_pinned_host(entry)
        _close_ctx_vmm_if_any(entry.get("main_ctx"))
        _close_ctx_vmm_if_any(entry.get("cfg_text_ctx"))
        _close_ctx_vmm_if_any(entry.get("cfg_img_ctx"))
        entry["main_ctx"] = None
        entry["cfg_text_ctx"] = None
        entry["cfg_img_ctx"] = None
        entry["swapped_host_bundle"] = bundle
        nbytes = int(bundle["total_bytes"])
        self._kv_host_swap_outs += 1
        self._kv_host_pinned_bytes += nbytes
        return nbytes

    def evict_idle_kv_to_host(
        self,
        target_free_bytes: int,
        exclude_ctx_ids: Iterable[str] = (),
        *,
        for_dit: bool = False,
    ) -> int:
        """Evict LRU unpinned think-time sessions' VMM KV to Host Pinned DRAM until `available_kv_bytes >= target`."""
        excluded = set(exclude_ctx_ids) | {cid for cid, cnt in self._active_ctx_pins.items() if cnt > 0}
        candidates: list[tuple[float, str, dict[str, Any]]] = []
        for cid, ent in self._ctx_store.items():
            if cid in excluded or cid.startswith("__warm_"):
                continue
            if ent.get("swapped_host_bundle") is not None:
                continue
            m_ctx = ent.get("main_ctx")
            if m_ctx is None or m_ctx.get("past_key_values") is None:
                continue
            pkv = m_ctx["past_key_values"]
            if bool(getattr(pkv, "_is_swapped", False)) or bool(getattr(pkv, "is_swapped_out", False)):
                continue
            if _peek_cache_slot(pkv.key_cache, 0) is None:
                continue
            candidates.append((self._ctx_last_used_s.get(cid, 0.0), cid, ent))

        candidates.sort(key=lambda item: item[0])
        evicted_total = 0
        for _, cid, ent in candidates:
            if self.available_kv_bytes(for_dit=for_dit) >= int(target_free_bytes):
                break
            evicted_total += self._swap_out_ctx_to_host(cid, ent)
        if evicted_total > 0 and self.vmm is None:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return evicted_total

    def _ensure_ctx_resident(self, ctx_id: str, extra_needed_bytes: int = 0) -> dict[str, Any]:
        entry = self._ctx_store[ctx_id]
        bundle = entry.get("swapped_host_bundle")
        if bundle is not None:
            req_b = int(bundle["total_bytes"]) + int(extra_needed_bytes) + 256 * MIB
            if self.available_kv_bytes() < req_b:
                self.evict_idle_kv_to_host(req_b, exclude_ctx_ids=(ctx_id,))
            main_factory = (
                (lambda n: VmmNaiveCache.create_for_bagel(self._naive_cache_cls, n, self.vmm))
                if self.vmm is not None
                else (lambda n: self._naive_cache_cls(n))
            )
            try:
                unpack_bagel_entry_from_pinned_host(
                    bundle,
                    entry,
                    device=self.device,
                    main_cache_factory=main_factory,
                    aux_cache_factory=lambda n: self._naive_cache_cls(n),
                    use_cow_rebind=self.policy.use_module3_cow_loopback,
                )
            except BaseException as exc:
                if is_oom_exception(exc):
                    freed = self.evict_idle_kv_to_host(1 << 50, exclude_ctx_ids=(ctx_id,))
                    if freed > 0:
                        unpack_bagel_entry_from_pinned_host(
                            bundle,
                            entry,
                            device=self.device,
                            main_cache_factory=main_factory,
                            aux_cache_factory=lambda n: self._naive_cache_cls(n),
                            use_cow_rebind=self.policy.use_module3_cow_loopback,
                        )
                    else:
                        raise
                else:
                    raise
            entry["swapped_host_bundle"] = None
            self._kv_host_pinned_bytes = max(0, self._kv_host_pinned_bytes - int(bundle["total_bytes"]))
            self._kv_host_swap_ins += 1
            if self._active_ctx_pins.get(ctx_id, 0) > 0:
                self._sync_ctx_vmm_activity(ctx_id, active=True)
            return entry

        m_ctx = entry.get("main_ctx")
        pkv = m_ctx.get("past_key_values") if m_ctx is not None else None
        if pkv is not None and (bool(getattr(pkv, "_is_swapped", False)) or bool(getattr(pkv, "is_swapped_out", False))):
            req_b = bagel_entry_nbytes(entry) + int(extra_needed_bytes) + 256 * MIB
            if self.available_kv_bytes() < req_b:
                self.evict_idle_kv_to_host(req_b, exclude_ctx_ids=(ctx_id,))
            pkv.ensure_resident()
            if self.vmm is not None and callable(getattr(self.vmm, "flush_kv_swaps", None)):
                self.vmm.flush_kv_swaps()
            if self._active_ctx_pins.get(ctx_id, 0) > 0:
                self._sync_ctx_vmm_activity(ctx_id, active=True)
        elif extra_needed_bytes > 0:
            watermark_b = int(extra_needed_bytes) + 512 * MIB
            if self.available_kv_bytes() < watermark_b:
                self.evict_idle_kv_to_host(watermark_b, exclude_ctx_ids=(ctx_id,))
        return entry

    def bind_runtime_resources(self, pacer: Any | None = None) -> dict[str, Any]:
        self.pacer = pacer
        if self.policy.use_tripool_vmm:
            self._vmm_initial_pool_mib = float(self.policy.super_pool_mib + self.policy.fine_kv_pool_mib)
            self.vmm = TriPoolVMM(
                device=self.rank,
                super_pool_mib=self.policy.super_pool_mib,
                fine_kv_pool_mib=self.policy.fine_kv_pool_mib,
                super_handle_mib=self.policy.super_handle_mib,
                transient_va_gib=self.policy.transient_va_gib,
                kv_va_gib=self.policy.kv_va_gib,
                allow_overflow=self.policy.allow_vmm_overflow,
            )
            if self.pager is not None:
                init_alpha = float(self.policy.initial_alphas.get(self.rank, 1.0))
                self._initial_bound_alpha = init_alpha
                self.pager.bind_vmm(self.vmm, initial_alpha=init_alpha)

        if self.policy.use_module3_cow_loopback:
            self.interceptor.enable_shared_prefix_cfg(min_shared_tokens=32, allow_cascade_split=False)

        if self.policy.preemption_mode != "non_preemptive":
            self.sched = UnifiedPreemptiveScheduler(
                self.pipeline,
                mode=self.policy.preemption_mode,
                tpot_slo_ms=self.policy.ar_tpot_slo_ms,
                guard_ms=14.0,
                init_layer_ms=10.5,
                init_ar_step_ms=66.0,
                pager=self.pager,
            )

        self._update_hwm()
        return {
            "resident_alpha": self.resident_alpha(),
            "reserved_mib": round(self._max_pytorch_reserved_mib, 2),
            "combined_physical_mib": round(self._max_reserved_mib, 2),
            "vmm_physical_pool_mib": round(self._vmm_physical_pool_mib(), 2),
            "kv_free_bytes": self.available_kv_bytes(),
        }

    def warmup(self) -> None:
        import torch

        if self.role != "gen":
            w_ctx = self._new_aux_ctx()
            self._update_text_ctx(w_ctx, "Warmup prompt.")
            self._ctx_store["__warm_ar__"] = {
                "main_ctx": w_ctx,
                "cfg_text_ctx": None,
                "cfg_img_ctx": None,
                "cfg_text_prefix_len": 0,
                "last_img_u8": None,
                "last_img_pil": None,
            }
            warm_out = self.decode_ar("__warm_ar__", max_tokens=3)
            tpots = warm_out.get("tpots_ms") or []
            if tpots:
                self._calibrated_ar_step_ms = max(5.0, min(66.0, float(tpots[-1])))
                if self.sched is not None:
                    self.sched.ema_ar_step_ms = self._calibrated_ar_step_ms
                    self.sched.guard_ms = 6.0
                    self.sched.last_effective_chunk_layers = self.sched.compute_adaptive_chunk_layers()
            self.release_context("__warm_ar__")

        if self.role != "und" and self.naive_offloader is None:
            w_main = self._new_aux_ctx()
            w_img = self._new_aux_ctx()
            self._update_text_ctx(w_main, "Warmup.")
            self._update_text_ctx(w_img, "Warmup.")
            self._ctx_store["__warm_dit__"] = {
                "main_ctx": w_main,
                "cfg_text_ctx": None,
                "cfg_img_ctx": w_img,
                "cfg_text_prefix_len": 0,
                "last_img_u8": None,
                "last_img_pil": None,
            }
            _ = self.generate_dit(
                "__warm_dit__",
                {"ctx_id": "__warm_dit__", "height": 512, "width": 512, "dit_steps": 2, "seed": 1, "prompt": "Warmup."},
            )
            self.release_context("__warm_dit__")
            if self.pager is not None and self.vmm is not None:
                self.pager.set_residency_alpha(self.policy.initial_alphas.get(self.rank, 1.0))

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        pt_mib = float(torch.cuda.memory_reserved(self.device) / MIB)
        self._max_pytorch_reserved_mib = pt_mib
        self._max_reserved_mib = pt_mib + self._vmm_physical_pool_mib()

    def _update_hwm(self) -> None:
        import torch

        cur = float(torch.cuda.memory_reserved(self.device) / MIB)
        pk = float(torch.cuda.max_memory_reserved(self.device) / MIB)
        pt_peak = max(cur, pk)
        self._max_pytorch_reserved_mib = max(self._max_pytorch_reserved_mib, pt_peak)
        self._max_reserved_mib = max(self._max_reserved_mib, pt_peak + self._vmm_physical_pool_mib())

    def _load_pil(self, rel_path: str) -> Any:
        from PIL import Image

        im = self._img_cache.get(rel_path)
        if im is None:
            p = self.bench_root / rel_path if not os.path.isabs(rel_path) else Path(rel_path)
            with Image.open(p) as raw:
                im = raw.convert("RGB")
                im.load()
            if len(self._img_cache) < 512:
                self._img_cache[rel_path] = im
        return im

    def _update_text_ctx(self, ctx: dict[str, Any], prompt: str) -> None:
        import torch

        clean = prompt.removeprefix("<|im_start|>").removesuffix("<|im_end|>")
        inp, nl, nr = self.pipeline.bagel.prepare_prompts(
            curr_kvlens=ctx["kv_lens"],
            curr_rope=ctx["ropes"],
            prompts=[clean],
            tokenizer=self.pipeline.tokenizer,
            new_token_ids=self.pipeline.new_token_ids,
        )
        inp = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inp.items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ctx["past_key_values"] = self.pipeline.bagel.forward_cache_update_text(ctx["past_key_values"], **inp)
        ctx["kv_lens"], ctx["ropes"] = nl, nr

    def _update_image_ctx(
        self,
        ctx: dict[str, Any],
        img_or_u8: Any,
        update_vae: bool = True,
        update_vit: bool = True,
    ) -> None:
        import numpy as np
        import torch

        swapped = False
        if update_vae:
            if self.pager is not None and self.vmm is not None:
                self.pager.ensure_all_resident_sync()
            elif self.naive_offloader is not None:
                swapped = self.naive_offloader.swap_in_sync()
        try:
            if torch.is_tensor(img_or_u8):
                self.loopback.update_context_from_u8_gpu(ctx, img_or_u8, update_vae=update_vae, update_vit=update_vit)
                return

            pil_img = img_or_u8

            def vae_tf(v):
                arr = torch.from_numpy(np.array(v.convert("RGB"), copy=True)).float() / 127.5 - 1.0
                return arr.permute(2, 0, 1)

            def vit_tf(v):
                return self.pipeline.image_processor(images=v, return_tensors="pt").pixel_values[0]

            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if update_vae:
                    vae_seed = 10007 + (int(ctx["ropes"][0]) if ctx.get("ropes") else 0)
                    torch.manual_seed(vae_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(vae_seed)
                    vi, nl, nr = self.pipeline.bagel.prepare_vae_images(
                        curr_kvlens=ctx["kv_lens"],
                        curr_rope=ctx["ropes"],
                        images=[pil_img],
                        transforms=vae_tf,
                        new_token_ids=self.pipeline.new_token_ids,
                    )
                    vi = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in vi.items()}
                    ctx["past_key_values"] = self.pipeline.bagel.forward_cache_update_vae(
                        self.pipeline.vae, ctx["past_key_values"], **vi
                    )
                    ctx["kv_lens"], ctx["ropes"] = nl, nr
                if update_vit:
                    vti, nlv, nrv = self.pipeline.bagel.prepare_vit_images(
                        curr_kvlens=ctx["kv_lens"],
                        curr_rope=ctx["ropes"],
                        images=[pil_img],
                        transforms=vit_tf,
                        new_token_ids=self.pipeline.new_token_ids,
                    )
                    vti = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in vti.items()}
                    for k in ("packed_indexes", "packed_key_value_indexes", "key_values_lens"):
                        vti.pop(k, None)
                    ctx["past_key_values"] = self.pipeline.bagel.forward_cache_update_vit(ctx["past_key_values"], **vti)
                    ctx["kv_lens"], ctx["ropes"] = nlv, nrv
        finally:
            if swapped and self.naive_offloader is not None:
                self.naive_offloader.drop_from_gpu()

    def _apply_dit_turn_images_and_prompt(
        self,
        entry: dict[str, Any],
        cmd: dict[str, Any],
        *,
        can_run_vae: bool,
        prep_dit: bool,
    ) -> None:
        from PIL import Image

        main_ctx = entry["main_ctx"]
        entry.setdefault("pending_vae_paths", [])

        if cmd.get("use_prev_image"):
            if cmd.get("prev_image_path"):
                p_prev = str(cmd["prev_image_path"])
                self._update_image_ctx(main_ctx, self._load_pil(p_prev), update_vae=can_run_vae, update_vit=True)
                if prep_dit and not can_run_vae:
                    entry["pending_vae_paths"].append(p_prev)
            elif self.policy.use_module3_cow_loopback and entry.get("last_img_u8") is not None:
                self._update_image_ctx(main_ctx, entry["last_img_u8"], update_vae=can_run_vae, update_vit=True)
                if prep_dit and not can_run_vae:
                    entry["pending_vae_u8"] = True
            elif entry.get("last_img_pil") is not None:
                prev_pil = entry["last_img_pil"]
                if not self.policy.use_module3_cow_loopback:
                    buf = io.BytesIO()
                    prev_pil.save(buf, format="PNG")
                    buf.seek(0)
                    prev_pil = Image.open(buf).convert("RGB")
                    prev_pil.load()
                    entry["last_img_pil"] = prev_pil
                self._update_image_ctx(main_ctx, prev_pil, update_vae=can_run_vae, update_vit=True)
                if prep_dit and not can_run_vae:
                    entry["pending_vae_pil"] = True

        for rel_img in cmd.get("image_paths", []):
            self._update_image_ctx(main_ctx, self._load_pil(rel_img), update_vae=can_run_vae, update_vit=True)
            if prep_dit and not can_run_vae:
                entry["pending_vae_paths"].append(str(rel_img))

        prompt = str(cmd["prompt"])
        if prep_dit:
            _close_ctx_vmm_if_any(entry.get("cfg_text_ctx"))
            _close_ctx_vmm_if_any(entry.get("cfg_img_ctx"))
            k0_before = _peek_cache_slot(main_ctx["past_key_values"].key_cache, 0)
            has_hist = k0_before is not None
            if has_hist:
                prefix_len = int(k0_before.shape[0])
                if self.policy.use_module3_cow_loopback:
                    cfg_text_ctx = _slice_prefix_ctx(
                        main_ctx,
                        prefix_len,
                        self._naive_cache_cls,
                    )
                else:
                    cfg_text_ctx = deepcopy(main_ctx)
                entry["cfg_text_prefix_len"] = prefix_len
            else:
                cfg_text_ctx = None
                entry["cfg_text_prefix_len"] = 0

            self._update_text_ctx(main_ctx, prompt)
            if cfg_text_ctx is not None and self.policy.use_module3_cow_loopback:
                rebind_cow_prefix(main_ctx["past_key_values"], cfg_text_ctx["past_key_values"])

            k0_after = _peek_cache_slot(main_ctx["past_key_values"].key_cache, 0)
            if (not has_hist) and k0_after is not None:
                # Empty initial context + prompt -> `main_ctx` is bit-identical to `cfg_img_ctx`!
                cfg_img_ctx = _slice_prefix_ctx(
                    main_ctx,
                    int(k0_after.shape[0]),
                    self._naive_cache_cls,
                    ropes=main_ctx["ropes"],
                    kv_lens=main_ctx["kv_lens"],
                    clone_slices=(not self.policy.use_module3_cow_loopback),
                )
            else:
                cfg_img_ctx = self._new_aux_ctx()
                self._update_text_ctx(cfg_img_ctx, prompt)
            entry["cfg_text_ctx"] = cfg_text_ctx
            entry["cfg_img_ctx"] = cfg_img_ctx
        else:
            self._update_text_ctx(main_ctx, prompt)

    def _prepare_turn_inner(self, ctx_id: str, cmd: dict[str, Any]) -> int:
        has_images = bool(cmd.get("use_prev_image") or cmd.get("image_paths"))
        est_delta = (768 * MIB) if (has_images or cmd.get("prepare_for_dit")) else (128 * MIB)

        entry = self._ctx_store.get(ctx_id)
        if entry is None:
            if self.available_kv_bytes() < est_delta + 512 * MIB:
                self.evict_idle_kv_to_host(est_delta + 512 * MIB, exclude_ctx_ids=(ctx_id,))
            entry = {
                "main_ctx": self._new_main_ctx(),
                "cfg_text_ctx": None,
                "cfg_img_ctx": None,
                "cfg_text_prefix_len": 0,
                "pending_vae_paths": [],
                "pending_vae_u8": False,
                "pending_vae_pil": False,
                "pending_dit_turn": None,
                "swapped_host_bundle": None,
                "last_img_u8": None,
                "last_img_pil": None,
            }
            self._ctx_store[ctx_id] = entry
            if self._active_ctx_pins.get(ctx_id, 0) > 0:
                self._sync_ctx_vmm_activity(ctx_id, active=True)
        else:
            self._ensure_ctx_resident(ctx_id, extra_needed_bytes=est_delta)

        entry.setdefault("pending_vae_paths", [])
        prep_dit = bool(cmd.get("prepare_for_dit", False))
        is_ar_cmd = bool(cmd.get("max_ar_tokens")) or (cmd.get("op") in ("EXEC_AR", "EXEC_AR_PREEMPT"))
        is_fission_und = self.policy.routing_mode == "static_nccl_fission" and self.role == "und"
        pager_partial = self.pager is not None and self.vmm is not None and self.resident_alpha() < 0.999

        # When preparing a pure DiT turn (`PREP_DIT`) on a worker with partial expert residency (`e27bce5`),
        # defer the entire `[VAE, ViT, Prompt]` turn until `EXEC_DIT` so (a) zero premature H2D runs before
        # `DualPathMorphRouter` decides Path A vs Path B, AND (b) `EXEC_DIT` executes `[VAE, ViT, Prompt]` in
        # upstream BAGEL's exact canonical sequence order while preserving CoW pointer sharing for `FlashPrefix-CFG`!
        if prep_dit and (not is_ar_cmd) and has_images and pager_partial and (not is_fission_und):
            img_paths_list = [str(x) for x in cmd.get("image_paths", [])]
            entry["pending_dit_turn"] = {
                "use_prev_image": bool(cmd.get("use_prev_image")),
                "prev_image_path": cmd.get("prev_image_path"),
                "image_paths": img_paths_list,
                "prompt": str(cmd["prompt"]),
            }
            n_pending_imgs = (1 if cmd.get("use_prev_image") else 0) + len(img_paths_list)
            self._update_hwm()
            return bagel_entry_nbytes(entry) + max(1, n_pending_imgs) * (192 * MIB)

        can_run_vae_here = prep_dit and (not is_fission_und) and (not pager_partial)
        main_ctx = entry["main_ctx"]
        k0_init = _peek_cache_slot(main_ctx["past_key_values"].key_cache, 0)
        orig_seq_len = int(k0_init.shape[0]) if k0_init is not None else 0
        orig_kv_lens = list(main_ctx["kv_lens"])
        orig_ropes = list(main_ctx["ropes"])
        orig_pending_vae_paths = list(entry.get("pending_vae_paths", []))
        orig_pending_vae_u8 = bool(entry.get("pending_vae_u8", False))
        orig_pending_vae_pil = bool(entry.get("pending_vae_pil", False))
        orig_pending_dit_turn = entry.get("pending_dit_turn")
        try:
            self._apply_dit_turn_images_and_prompt(
                entry,
                cmd,
                can_run_vae=can_run_vae_here,
                prep_dit=prep_dit,
            )
        except BaseException:
            _close_ctx_vmm_if_any(entry.get("cfg_text_ctx"))
            _close_ctx_vmm_if_any(entry.get("cfg_img_ctx"))
            entry["cfg_text_ctx"] = None
            entry["cfg_img_ctx"] = None
            entry["cfg_text_prefix_len"] = 0
            _truncate_bagel_cache(main_ctx["past_key_values"], orig_seq_len)
            main_ctx["kv_lens"] = orig_kv_lens
            main_ctx["ropes"] = orig_ropes
            entry["pending_vae_paths"] = orig_pending_vae_paths
            entry["pending_vae_u8"] = orig_pending_vae_u8
            entry["pending_vae_pil"] = orig_pending_vae_pil
            entry["pending_dit_turn"] = orig_pending_dit_turn
            raise

        self._update_hwm()
        return bagel_entry_nbytes(entry)

    def prepare_turn(self, ctx_id: str, cmd: dict[str, Any]) -> int:
        had_entry = ctx_id in self._ctx_store
        with self._pinned_ctx(ctx_id):
            try:
                return self._prepare_turn_inner(ctx_id, cmd)
            except BaseException as exc:
                if not had_entry:
                    partial = self._ctx_store.pop(ctx_id, None)
                    if partial is not None:
                        _close_ctx_vmm_if_any(partial.get("main_ctx"))
                        _close_ctx_vmm_if_any(partial.get("cfg_text_ctx"))
                        _close_ctx_vmm_if_any(partial.get("cfg_img_ctx"))
                if is_oom_exception(exc):
                    freed = self.evict_idle_kv_to_host(1 << 50, exclude_ctx_ids=(ctx_id,))
                    if freed > 0:
                        return self._prepare_turn_inner(ctx_id, cmd)
                raise

    def decode_ar(self, ctx_id: str, max_tokens: int) -> dict[str, Any]:
        outs = self.decode_ar_batch([{"ctx_id": ctx_id, "max_ar_tokens": max_tokens, "_skip_prepare": True}])
        return outs[0]

    def decode_ar_batch(self, batch_cmds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """True `[B, D]` packed GEMM continuous batching across `B >= 1` concurrent BAGEL AR requests."""
        import torch

        if not batch_cmds:
            return []

        cids = [str(cmd["ctx_id"]) for cmd in batch_cmds]
        states: list[dict[str, Any]] = []
        end_id = self.pipeline.new_token_ids["eos_token_id"]

        with self._pinned_ctx(*cids):
            for cmd in batch_cmds:
                cid = str(cmd["ctx_id"])
                enq_s = float(cmd.get("enqueue_time_s", 0.0))
                t_prep0 = time.perf_counter()
                if not cmd.get("_skip_prepare", False):
                    self.prepare_turn(cid, cmd)
                else:
                    self._ensure_ctx_resident(cid, extra_needed_bytes=64 * MIB)
                entry = self._ctx_store[cid]
                ctx = entry["main_ctx"]
                start_inp = self.pipeline.bagel.prepare_start_tokens(
                    ctx["kv_lens"], ctx["ropes"], self.pipeline.new_token_ids
                )
                start_tok = start_inp["packed_start_tokens"].to(self.device)
                start_tok_id = int(start_inp["packed_start_tokens"].reshape(-1)[0].item())
                states.append(
                    {
                        "ctx_id": cid,
                        "entry": entry,
                        "ctx": ctx,
                        "curr_tokens": start_tok,
                        "curr_tok_id": start_tok_id,
                        "pos_ids": start_inp["packed_query_position_ids"].to(self.device),
                        "max_tokens": max(1, int(cmd.get("max_ar_tokens", 32))),
                        "gen_ids": [],
                        "stamps": [],
                        "t0": enq_s if enq_s > 0.0 else t_prep0,
                        "done": False,
                    }
                )

            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                while True:
                    active = [st for st in states if not st["done"]]
                    if not active:
                        break
                    for st in active:
                        st["gen_ids"].append(st["curr_tok_id"])

                    prev_ar_lens = [
                        int(k0.shape[0])
                        if (k0 := _peek_cache_slot(st["ctx"]["past_key_values"].key_cache, 0)) is not None
                        else 0
                        for st in active
                    ]
                    try:
                        next_tokens, updated_caches = self.interceptor.forward_ar_step_batch(
                            [st["curr_tokens"] for st in active],
                            [st["pos_ids"] for st in active],
                            [st["ctx"]["past_key_values"] for st in active],
                        )
                    except BaseException as exc:
                        for st, plen in zip(active, prev_ar_lens):
                            _truncate_bagel_cache(st["ctx"]["past_key_values"], plen)
                        if is_oom_exception(exc) and self.evict_idle_kv_to_host(1 << 50, exclude_ctx_ids=cids) > 0:
                            next_tokens, updated_caches = self.interceptor.forward_ar_step_batch(
                                [st["curr_tokens"] for st in active],
                                [st["pos_ids"] for st in active],
                                [st["ctx"]["past_key_values"] for st in active],
                            )
                        else:
                            for st in active:
                                st["gen_ids"].pop()
                            raise

                    # Single [B] int64 D2H readback synchronizes CUDA stream for the whole packed GEMM step
                    next_ids = [int(x) for x in next_tokens.detach().cpu().tolist()]
                    now_s = time.perf_counter()
                    for idx, st in enumerate(active):
                        nxt_id = next_ids[idx]
                        st["ctx"]["past_key_values"] = updated_caches[idx]
                        st["curr_tokens"] = next_tokens[idx : idx + 1]
                        st["curr_tok_id"] = nxt_id
                        st["pos_ids"] = st["pos_ids"] + 1
                        st["stamps"].append(now_s)
                        reached = len(st["gen_ids"]) >= st["max_tokens"]
                        hit_eos = end_id is not None and nxt_id == int(end_id)
                        if reached or hit_eos:
                            st["done"] = True

            self._update_hwm()
            alpha_now = self.resident_alpha()
            free_kv_now = self.available_kv_bytes()
            outputs: list[dict[str, Any]] = []
            for st in states:
                ctx = st["ctx"]
                entry = st["entry"]
                gen_ids = st["gen_ids"]
                stamps = st["stamps"]
                n_tok = len(gen_ids)
                ctx["kv_lens"] = [kl + n_tok for kl in ctx["kv_lens"]]
                ctx["ropes"] = [r + n_tok for r in ctx["ropes"]]
                if self.policy.use_module3_cow_loopback:
                    if entry.get("cfg_text_ctx") is not None:
                        rebind_cow_prefix(ctx["past_key_values"], entry["cfg_text_ctx"]["past_key_values"])
                    if entry.get("cfg_img_ctx") is not None and int(entry.get("cfg_text_prefix_len", 0)) == 0:
                        rebind_cow_prefix(ctx["past_key_values"], entry["cfg_img_ctx"]["past_key_values"])
                ttft_ms = max(0.0, (stamps[0] - st["t0"]) * 1000.0) if stamps else 0.0
                tpots_ms = [(stamps[i] - stamps[i - 1]) * 1000.0 for i in range(1, len(stamps))]
                decoded = self.pipeline.tokenizer.decode(gen_ids) if gen_ids else ""
                outputs.append(
                    {
                        "text": decoded.split("<|im_end|>")[0].split("<|im_start|>")[-1],
                        "tokens": n_tok,
                        "ttft_ms": round(ttft_ms, 2),
                        "tpots_ms": [round(x, 2) for x in tpots_ms],
                        "kv_bytes": bagel_entry_nbytes(entry),
                        "resident_alpha": alpha_now,
                        "kv_free_bytes": free_kv_now,
                        "batched_width": len(states),
                    }
                )
            return outputs

    def generate_dit(
        self,
        ctx_id: str,
        cmd: dict[str, Any],
        poll_preempt_cmds: Callable[[], list[dict[str, Any]]] | None = None,
        emit_preempt_ack: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        from PIL import Image
        import torch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        with self._pinned_ctx(ctx_id):
            entry = self._ensure_ctx_resident(ctx_id)
            has_pending_turn = entry.get("pending_dit_turn") is not None
            has_pending_vae = bool(
                entry.get("pending_vae_u8") or entry.get("pending_vae_pil") or entry.get("pending_vae_paths")
            )
            if self.pager is not None and self.vmm is not None:
                # `available_kv_bytes(for_dit=False)` measures `(pool_free + unpinned_experts + overflow)`;
                # ensure enough headroom to pin all 28 expert layers (`total_mirror_bytes`) + active KV!
                extra_dit_kv = (256 * MIB) if (has_pending_turn or has_pending_vae) else (64 * MIB)
                needed_for_dit = int(self.pager.total_mirror_bytes) + extra_dit_kv
                if self.available_kv_bytes(for_dit=False) < needed_for_dit:
                    self.evict_idle_kv_to_host(needed_for_dit, exclude_ctx_ids=(ctx_id,), for_dit=False)

            if has_pending_turn or has_pending_vae:
                early_pacer_cm = (
                    self.pacer.acquire_process_slot(self.rank)
                    if (
                        self.pacer is not None
                        and self.pager is not None
                        and self.vmm is not None
                        and self.vmm.expert_resident_count(self.pager.expert_id) < self.pager.num_layers
                    )
                    else nullcontext()
                )
                with early_pacer_cm:
                    pend_turn = entry.pop("pending_dit_turn", None)
                    if pend_turn is not None:
                        self._apply_dit_turn_images_and_prompt(
                            entry,
                            pend_turn,
                            can_run_vae=True,
                            prep_dit=True,
                        )
                    main_ctx = entry["main_ctx"]
                    cfg_text_ctx = entry.get("cfg_text_ctx")
                    if entry.pop("pending_vae_u8", False) and entry.get("last_img_u8") is not None:
                        self._update_image_ctx(main_ctx, entry["last_img_u8"], update_vae=True, update_vit=False)
                        if cfg_text_ctx is not None:
                            self._update_image_ctx(cfg_text_ctx, entry["last_img_u8"], update_vae=True, update_vit=False)

                    if entry.pop("pending_vae_pil", False) and entry.get("last_img_pil") is not None:
                        self._update_image_ctx(main_ctx, entry["last_img_pil"], update_vae=True, update_vit=False)
                        if cfg_text_ctx is not None:
                            self._update_image_ctx(cfg_text_ctx, entry["last_img_pil"], update_vae=True, update_vit=False)

                    for pend_img in entry.pop("pending_vae_paths", []) or []:
                        pil_v = self._load_pil(pend_img)
                        self._update_image_ctx(main_ctx, pil_v, update_vae=True, update_vit=False)
                        if cfg_text_ctx is not None:
                            self._update_image_ctx(cfg_text_ctx, pil_v, update_vae=True, update_vit=False)

            main_ctx = entry["main_ctx"]
            cfg_text_ctx = entry.get("cfg_text_ctx")
            cfg_img_ctx = entry.get("cfg_img_ctx")

            height = int(cmd.get("height", 512))
            width = int(cmd.get("width", 512))
            steps = int(cmd.get("dit_steps", 20))
            seed = int(cmd.get("seed", 52))
            prompt = str(cmd.get("prompt", ""))
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
            if cfg_text_ctx is not None and _peek_cache_slot(cfg_text_ctx["past_key_values"].key_cache, 0) is not None:
                sampling.cfg_text_past_key_values = cfg_text_ctx["past_key_values"]
                sampling.cfg_text_kv_metadata = {"ropes": cfg_text_ctx["ropes"], "image_shape": [height, width]}
            if cfg_img_ctx is not None and _peek_cache_slot(cfg_img_ctx["past_key_values"].key_cache, 0) is not None:
                sampling.cfg_img_past_key_values = cfg_img_ctx["past_key_values"]
                sampling.cfg_img_kv_metadata = {"ropes": cfg_img_ctx["ropes"], "image_shape": [height, width]}

            if self.naive_offloader is not None:
                self.naive_offloader.swap_in_sync()

            pacer_slot_cm = None
            if (
                self.pacer is not None
                and self.pager is not None
                and self.vmm is not None
                and self.vmm.expert_resident_count(self.pager.expert_id) < self.pager.num_layers
            ):
                pacer_slot_cm = self.pacer.acquire_process_slot(self.rank)

            inflight_ar_meta: dict[int, tuple[str, str, dict[str, Any]]] = {}

            def _poll_external_ar(now_s: float) -> list[ArDecodeTask]:
                if poll_preempt_cmds is None:
                    return []
                tasks: list[ArDecodeTask] = []
                for ar_msg in poll_preempt_cmds():
                    ar_cid = str(ar_msg["ctx_id"])
                    self._active_ctx_pins[ar_cid] = self._active_ctx_pins.get(ar_cid, 0) + 1
                    try:
                        self.prepare_turn(ar_cid, ar_msg)
                        ar_entry = self._ctx_store[ar_cid]
                        ar_ctx = ar_entry["main_ctx"]
                        st = self.pipeline.bagel.prepare_start_tokens(
                            ar_ctx["kv_lens"], ar_ctx["ropes"], self.pipeline.new_token_ids
                        )
                        start_tok_gpu = st["packed_start_tokens"].to(self.device)
                        start_tok_int = int(st["packed_start_tokens"].reshape(-1)[0].item())
                        task = ArDecodeTask(
                            req_id=str(ar_msg["cmd_id"]),
                            kv_cache=ar_ctx["past_key_values"],
                            curr_tokens=start_tok_gpu,
                            pos_ids=st["packed_query_position_ids"].to(self.device),
                            max_tokens=int(ar_msg.get("max_ar_tokens", 32)),
                            end_token_id=self.pipeline.new_token_ids["eos_token_id"],
                            tpot_slo_ms=float(ar_msg.get("tpot_slo_ms", self.policy.ar_tpot_slo_ms)),
                            arrival_time_s=float(ar_msg.get("enqueue_time_s", now_s)),
                            _curr_tok_int=start_tok_int,
                        )
                        task.last_step_time_s = task.arrival_time_s
                        inflight_ar_meta[id(task)] = (str(ar_msg["cmd_id"]), ar_cid, ar_entry)
                        tasks.append(task)
                    except BaseException:
                        rem = self._active_ctx_pins.get(ar_cid, 1) - 1
                        if rem <= 0:
                            self._active_ctx_pins.pop(ar_cid, None)
                            self._sync_ctx_vmm_activity(ar_cid, active=False)
                        else:
                            self._active_ctx_pins[ar_cid] = rem
                        raise
                return tasks

            def _unpin_inflight_ar(ar_cid: str) -> None:
                rem = self._active_ctx_pins.get(ar_cid, 1) - 1
                if rem <= 0:
                    self._active_ctx_pins.pop(ar_cid, None)
                    self._sync_ctx_vmm_activity(ar_cid, active=False)
                else:
                    self._active_ctx_pins[ar_cid] = rem
                if ar_cid in self._ctx_store:
                    self._ctx_last_used_s[ar_cid] = time.perf_counter()

            def _flush_completed_ar() -> None:
                if self.sched is None or emit_preempt_ack is None:
                    return
                while self.sched.completed_ar_tasks:
                    t = self.sched.completed_ar_tasks.pop(0)
                    meta_tuple = inflight_ar_meta.pop(id(t), None)
                    if meta_tuple is None:
                        continue
                    c_id, ar_ctx_id, ar_entry = meta_tuple
                    _unpin_inflight_ar(ar_ctx_id)
                    ar_ctx = ar_entry["main_ctx"]
                    ar_ctx["past_key_values"] = t.kv_cache
                    n_tok = len(t.generated_token_ids)
                    ar_ctx["kv_lens"] = [kl + n_tok for kl in ar_ctx["kv_lens"]]
                    ar_ctx["ropes"] = [r + n_tok for r in ar_ctx["ropes"]]
                    if self.policy.use_module3_cow_loopback:
                        if ar_entry.get("cfg_text_ctx") is not None:
                            rebind_cow_prefix(ar_ctx["past_key_values"], ar_entry["cfg_text_ctx"]["past_key_values"])
                        if (
                            ar_entry.get("cfg_img_ctx") is not None
                            and int(ar_entry.get("cfg_text_prefix_len", 0)) == 0
                        ):
                            rebind_cow_prefix(ar_ctx["past_key_values"], ar_entry["cfg_img_ctx"]["past_key_values"])
                    decoded = self.pipeline.tokenizer.decode(t.generated_token_ids) if t.generated_token_ids else ""
                    emit_preempt_ack(
                        {
                            "cmd_id": c_id,
                            "rank": self.rank,
                            "status": "ok",
                            "ctx_id": ar_ctx_id,
                            "text": decoded.split("<|im_end|>")[0].split("<|im_start|>")[-1],
                            "tokens": n_tok,
                            "ttft_ms": round(t.ttft_ms(), 2),
                            "tpots_ms": [round(x, 2) for x in t.inter_token_latencies_ms()],
                            "kv_bytes": bagel_entry_nbytes(ar_entry),
                            "resident_alpha": self.resident_alpha(),
                            "kv_free_bytes": self.available_kv_bytes(),
                            "preempted_mid_dit": True,
                        }
                    )

            hw_key = (height, width)
            if self.sched is not None:
                has_prior_hw = hw_key in self._calibrated_layer_ms_by_hw
                self.sched.reset_telemetry(preserve_layer_calibration=has_prior_hw)
                self.sched.ema_layer_ms = self._calibrated_layer_ms_by_hw.get(
                    hw_key, 37.0 if height >= 1024 else 10.5
                )
                if self._calibrated_ar_step_ms is not None:
                    self.sched.ema_ar_step_ms = self._calibrated_ar_step_ms
                self.sched.last_effective_chunk_layers = self.sched.compute_adaptive_chunk_layers()
                self.sched.external_arrival_poll = _poll_external_ar
                self.sched.on_task_complete = lambda _t: _flush_completed_ar()

            torch.cuda.current_stream(self.device).synchronize()
            t0 = time.perf_counter()
            latents = None
            img_shape = (height, width)
            try:
                with torch.inference_mode():
                    prep = self.pipeline._forward_single(
                        {"prompt": prompt, "modalities": ["image"]},
                        sampling,
                        prepare_only=True,
                    )
                    gp = prep["gen_params"]
                    img_shape = prep["image_shape"]

                    try:
                        with self.interceptor.generation_session(
                            pager=self.pager,
                            use_cfg_patch=self.policy.use_module3_cow_loopback,
                            scheduler=self.sched,
                            pacer_slot_cm=pacer_slot_cm,
                            prefetch_all=True,
                            reset_cfg_stats=True,
                        ):
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                latents, *_ = self.pipeline.bagel.generate_image(
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
                                    scheduler=self.pipeline.scheduler,
                                    scheduler_kwargs=self.pipeline.scheduler_kwargs,
                                )

                        if self.sched is not None and self.sched._layer_calibrated:
                            self._calibrated_layer_ms_by_hw[hw_key] = float(self.sched.ema_layer_ms)

                        # Drop dead per-turn CFG branch caches immediately after denoising completes so
                        # think-time residency and multi-turn return shipping hold only live `main_ctx`!
                        _close_ctx_vmm_if_any(entry.get("cfg_text_ctx"))
                        _close_ctx_vmm_if_any(entry.get("cfg_img_ctx"))
                        entry["cfg_text_ctx"] = None
                        entry["cfg_img_ctx"] = None
                        entry["cfg_text_prefix_len"] = 0

                        # Run FLUX VAE decode OUTSIDE `generation_session` so expert layers are unpinned:
                        # `TriPoolVMM` borrows tail expert super-handles for the 1024^2 VAE spike and leaves
                        # them lazily unmapped (`e27bce5`), avoiding any synchronous post-VAE H2D stall!
                        if self.policy.use_module3_cow_loopback:
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                img_u8 = decode_latent_to_u8_gpu(
                                    self.pipeline.bagel, self.pipeline.vae, latents[0], img_shape
                                )
                            entry["last_img_u8"] = img_u8
                            if want_pil or save_tmp_path:
                                pil_img = Image.fromarray(img_u8.permute(1, 2, 0).contiguous().cpu().numpy())
                                entry["last_img_pil"] = pil_img
                                if save_tmp_path:
                                    Path(save_tmp_path).parent.mkdir(parents=True, exist_ok=True)
                                    pil_img.save(save_tmp_path, format="PNG")
                            else:
                                entry["last_img_pil"] = None
                        else:
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                pil_img = self.pipeline._decode_image_from_latent(
                                    self.pipeline.bagel, self.pipeline.vae, latents[0], img_shape
                                )
                            entry["last_img_pil"] = pil_img
                            entry["last_img_u8"] = None
                            if save_tmp_path:
                                Path(save_tmp_path).parent.mkdir(parents=True, exist_ok=True)
                                pil_img.save(save_tmp_path, format="PNG")
                    finally:
                        if self.sched is not None and emit_preempt_ack is not None:
                            try:
                                _flush_completed_ar()
                                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                    self.sched.drain_remaining_ar_tasks()
                                _flush_completed_ar()
                            except Exception as ar_exc:
                                if is_oom_exception(ar_exc):
                                    torch.cuda.empty_cache()
                                    for c_id, ar_cid, _ in list(inflight_ar_meta.values()):
                                        _unpin_inflight_ar(ar_cid)
                                        emit_preempt_ack(
                                            {"cmd_id": c_id, "rank": self.rank, "status": "oom", "error": str(ar_exc)[:200]}
                                        )
                                else:
                                    for c_id, ar_cid, _ in list(inflight_ar_meta.values()):
                                        _unpin_inflight_ar(ar_cid)
                                        emit_preempt_ack(
                                            {
                                                "cmd_id": c_id,
                                                "rank": self.rank,
                                                "status": "error",
                                                "error": f"{type(ar_exc).__name__}: {ar_exc}",
                                            }
                                        )
                                inflight_ar_meta.clear()
                            finally:
                                for c_id, ar_cid, _ in list(inflight_ar_meta.values()):
                                    _unpin_inflight_ar(ar_cid)
                                    emit_preempt_ack(
                                        {"cmd_id": c_id, "rank": self.rank, "status": "oom", "error": "aborted mid-dit"}
                                    )
                                inflight_ar_meta.clear()
            finally:
                if self.sched is not None:
                    self.sched.external_arrival_poll = None
                    self.sched.on_task_complete = None
                if self.naive_offloader is not None:
                    self.naive_offloader.drop_from_gpu()

            torch.cuda.current_stream(self.device).synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._update_hwm()
            return {
                "status": "ok",
                "ctx_id": ctx_id,
                "elapsed_ms": round(elapsed_ms, 2),
                "kv_bytes": bagel_entry_nbytes(entry),
                "resident_alpha": self.resident_alpha(),
                "kv_free_bytes": self.available_kv_bytes(),
                "yield_count": self.sched.yield_count if self.sched is not None else 0,
            }

    def pack_state(self, ctx_id: str, include_cfg_dup: bool = False) -> PackedTransferState:
        with self._pinned_ctx(ctx_id):
            entry = self._ensure_ctx_resident(ctx_id)
            return pack_bagel_multi_context(entry, include_cfg_text_dup=include_cfg_dup, device=self.device)

    def unpack_state(self, new_ctx_id: str, meta: dict[str, Any], raw_payload: Any) -> int:
        self.release_context(new_ctx_id)
        with self._pinned_ctx(new_ctx_id):
            est_b = int(meta.get("logical_bytes", meta.get("total_bytes", 0)))
            if est_b > 0 and self.available_kv_bytes() < est_b + 512 * MIB:
                self.evict_idle_kv_to_host(est_b + 512 * MIB, exclude_ctx_ids=(new_ctx_id,))
            main_factory = (
                (lambda n: VmmNaiveCache.create_for_bagel(self._naive_cache_cls, n, self.vmm))
                if self.vmm is not None
                else (lambda n: self._naive_cache_cls(n))
            )
            try:
                entry = unpack_bagel_multi_context(
                    meta,
                    raw_payload,
                    main_cache_factory=main_factory,
                    aux_cache_factory=lambda n: self._naive_cache_cls(n),
                    use_cow_rebind=self.policy.use_module3_cow_loopback,
                )
            except BaseException as exc:
                if is_oom_exception(exc) and self.evict_idle_kv_to_host(1 << 50, exclude_ctx_ids=(new_ctx_id,)) > 0:
                    entry = unpack_bagel_multi_context(
                        meta,
                        raw_payload,
                        main_cache_factory=main_factory,
                        aux_cache_factory=lambda n: self._naive_cache_cls(n),
                        use_cow_rebind=self.policy.use_module3_cow_loopback,
                    )
                else:
                    raise
            entry["swapped_host_bundle"] = None
            self._ctx_store[new_ctx_id] = entry
            if self._active_ctx_pins.get(new_ctx_id, 0) > 0:
                self._sync_ctx_vmm_activity(new_ctx_id, active=True)
            self._update_hwm()
            return bagel_entry_nbytes(entry)

    def set_residency_alpha(self, alpha: float) -> float:
        if self.pager is not None and self.vmm is not None:
            self.pager.set_residency_alpha(float(alpha))
        return self.resident_alpha()

    def resident_alpha(self) -> float:
        if self.pager is not None and self.vmm is not None:
            return round(self.pager.resident_alpha(), 3)
        return round(float(self.policy.initial_alphas.get(self.rank, 1.0)), 3)

    def release_context(self, ctx_id: str) -> None:
        import torch

        self._active_ctx_pins.pop(ctx_id, None)
        self._ctx_last_used_s.pop(ctx_id, None)
        old = self._ctx_store.pop(ctx_id, None)
        if old is not None:
            swapped = old.get("swapped_host_bundle")
            if swapped is not None:
                self._kv_host_pinned_bytes = max(0, self._kv_host_pinned_bytes - int(swapped.get("total_bytes", 0)))
            _close_ctx_vmm_if_any(old.get("main_ctx"))
            _close_ctx_vmm_if_any(old.get("cfg_text_ctx"))
            _close_ctx_vmm_if_any(old.get("cfg_img_ctx"))
        if self.vmm is None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def memory_and_vmm_stats(self) -> dict[str, Any]:
        self._update_hwm()
        vmm_st = self.vmm.stats_dict() if self.vmm is not None else {}
        swapped_cnt = sum(
            1
            for e in self._ctx_store.values()
            if e.get("swapped_host_bundle") is not None
            or bool(getattr((e.get("main_ctx") or {}).get("past_key_values"), "_is_swapped", False))
        )
        return {
            "peak_reserved_mib": round(self._max_pytorch_reserved_mib, 2),
            "combined_physical_mib": round(self._max_reserved_mib, 2),
            "pytorch_reserved_mib": round(self._max_pytorch_reserved_mib, 2),
            "vmm_physical_pool_mib": round(self._vmm_physical_pool_mib(), 2),
            "vmm_stats": vmm_st,
            "active_contexts": len(self._ctx_store),
            "kv_free_bytes": self.available_kv_bytes(),
            "swapped_kv_contexts": swapped_cnt,
            "kv_host_swap_outs": self._kv_host_swap_outs + int(vmm_st.get("kv_swap_outs_total", 0)),
            "kv_host_swap_ins": self._kv_host_swap_ins + int(vmm_st.get("kv_swap_ins_total", 0)),
            "kv_host_pinned_bytes": self._kv_host_pinned_bytes + int(vmm_st.get("kv_swapped_bytes_current", 0)),
        }

    def close(self) -> None:
        import torch

        for cid in list(self._ctx_store.keys()):
            self.release_context(cid)
        self.interceptor.uninstall()
        if self.pager is not None:
            self.pager.close()
            self.pager = None
        if self.naive_offloader is not None:
            self.naive_offloader.close()
            self.naive_offloader = None
        if self.vmm is not None:
            self.vmm.shutdown()
            self.vmm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
