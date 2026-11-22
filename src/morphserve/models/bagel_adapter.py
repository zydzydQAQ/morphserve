"""Deep `BagelCPTGAdapter` (`tight_mot`) integrating `VmmNaiveCache`, `LayerExpertPager`, and `BagelExecutionInterceptor`."""

from __future__ import annotations

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
    cow_fork_context,
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

from .base import CPTGModelAdapter, PackedTransferState, is_oom_exception

MIB = 1024**2


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


def bagel_ctx_nbytes(ctx: dict[str, Any] | None) -> int:
    """Compute deduplicated logical GPU bytes occupied by one BAGEL KV context."""
    if ctx is None:
        return 0
    pkv = ctx.get("past_key_values")
    if pkv is None:
        return 0
    num_layers = int(pkv.num_layers)
    get_k = pkv.key_cache.get if isinstance(pkv.key_cache, dict) else (lambda i: pkv.key_cache[i])
    get_v = pkv.value_cache.get if isinstance(pkv.value_cache, dict) else (lambda i: pkv.value_cache[i])
    seen: dict[int, int] = {}
    for l in range(num_layers):
        for t in (get_k(l), get_v(l)):
            if t is not None and hasattr(t, "data_ptr"):
                ptr = int(t.data_ptr())
                seen[ptr] = max(seen.get(ptr, 0), _tensor_logical_bytes(t))
    return sum(seen.values())


def bagel_entry_nbytes(entry: dict[str, Any]) -> int:
    """Compute deduplicated logical GPU bytes across `main`, `cfg_text`, and `cfg_img` branches."""
    seen: dict[int, int] = {}
    for key in ("main_ctx", "cfg_text_ctx", "cfg_img_ctx"):
        ctx = entry.get(key)
        if ctx is None or ctx.get("past_key_values") is None:
            continue
        pkv = ctx["past_key_values"]
        num_layers = int(pkv.num_layers)
        get_k = pkv.key_cache.get if isinstance(pkv.key_cache, dict) else (lambda i: pkv.key_cache[i])
        get_v = pkv.value_cache.get if isinstance(pkv.value_cache, dict) else (lambda i: pkv.value_cache[i])
        for l in range(num_layers):
            for t in (get_k(l), get_v(l)):
                if t is not None and hasattr(t, "data_ptr"):
                    ptr = int(t.data_ptr())
                    seen[ptr] = max(seen.get(ptr, 0), _tensor_logical_bytes(t))
    return sum(seen.values())


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


def pack_bagel_multi_context(
    entry: dict[str, Any],
    include_cfg_text_dup: bool,
    device: Any,
) -> PackedTransferState:
    """Pack `main_ctx`, optional `cfg_text_ctx` (or zero-copy prefix slice metadata), and `cfg_img_ctx`."""
    import torch

    main_ctx = entry["main_ctx"]
    cfg_text_ctx = entry.get("cfg_text_ctx")
    cfg_img_ctx = entry.get("cfg_img_ctx")
    cfg_text_prefix_len = int(entry.get("cfg_text_prefix_len", 0))
    pending_vae_paths = list(entry.get("pending_vae_paths") or [])

    named_items: list[tuple[str, dict[str, Any]]] = [("main", main_ctx)]
    if include_cfg_text_dup and cfg_text_ctx is not None and cfg_text_ctx["past_key_values"].key_cache[0] is not None:
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

    flat = torch.cat(flat_chunks, dim=0) if flat_chunks else torch.empty((0,), dtype=torch.uint8, device=device)
    total_b = int(flat.numel())
    meta = {
        "total_bytes": total_b,
        "logical_bytes": total_b,
        "cfg_text_prefix_len": cfg_text_prefix_len,
        "include_cfg_text_dup": bool(include_cfg_text_dup),
        "pending_vae_paths": pending_vae_paths,
        "sections": sections,
    }
    return PackedTransferState(meta=meta, payload=flat, logical_bytes=total_b)


def unpack_bagel_multi_context(
    meta: dict[str, Any],
    flat_u8: Any,
    main_cache_factory: Callable[[int], Any],
    aux_cache_factory: Callable[[int], Any],
    use_cow_rebind: bool,
) -> dict[str, Any]:
    """Unpack multi-branch BAGEL session entry into `main_cache_factory` and rebind `cfg_text` CoW prefix."""
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
        cfg_text_ctx = _slice_prefix_ctx(main_ctx, int(meta["cfg_text_prefix_len"]), aux_cache_factory)

    cfg_img_ctx = None
    if "cfg_img" in sections:
        ci_sec = sections["cfg_img"]
        cfg_img_ctx = unpack_bagel_kv_context(
            ci_sec["meta"],
            flat_u8[ci_sec["offset"] : ci_sec["offset"] + ci_sec["nbytes"]],
            aux_cache_factory,
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


def _close_ctx_vmm_if_any(ctx: dict[str, Any] | None) -> None:
    if ctx is None:
        return
    pkv = ctx.get("past_key_values")
    if pkv is not None and hasattr(pkv, "close"):
        pkv.close()


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
        self._max_reserved_mib = 0.0

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
            return {"kv_lens": [0], "ropes": [0], "past_key_values": VmmNaiveCache(self.num_expert_layers, self.vmm)}
        return {"kv_lens": [0], "ropes": [0], "past_key_values": self._naive_cache_cls(self.num_expert_layers)}

    def _new_aux_ctx(self) -> dict[str, Any]:
        return {"kv_lens": [0], "ropes": [0], "past_key_values": self._naive_cache_cls(self.num_expert_layers)}

    def bind_runtime_resources(self, pacer: Any | None = None) -> dict[str, Any]:
        import torch

        self.pacer = pacer
        if self.policy.use_tripool_vmm:
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
                self.pager.bind_vmm(self.vmm, initial_alpha=self.policy.initial_alphas.get(self.rank, 1.0))

        if self.policy.use_module3_cow_loopback:
            self.interceptor.enable_shared_prefix_cfg(min_shared_tokens=32)

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
            "reserved_mib": round(float(torch.cuda.memory_reserved(self.device) / MIB), 2),
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
            _ = self.decode_ar("__warm_ar__", max_tokens=2)
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
        self._max_reserved_mib = float(torch.cuda.memory_reserved(self.device) / MIB)

    def _update_hwm(self) -> None:
        import torch

        res = float(torch.cuda.memory_reserved(self.device) / MIB)
        self._max_reserved_mib = max(self._max_reserved_mib, res)

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

    def prepare_turn(self, ctx_id: str, cmd: dict[str, Any]) -> int:
        from PIL import Image

        entry = self._ctx_store.get(ctx_id)
        if entry is None:
            entry = {
                "main_ctx": self._new_main_ctx(),
                "cfg_text_ctx": None,
                "cfg_img_ctx": None,
                "cfg_text_prefix_len": 0,
                "pending_vae_paths": [],
                "pending_vae_u8": False,
                "last_img_u8": None,
                "last_img_pil": None,
            }
            self._ctx_store[ctx_id] = entry

        entry.setdefault("pending_vae_paths", [])
        main_ctx = entry["main_ctx"]
        prep_dit = bool(cmd.get("prepare_for_dit", False))
        # Defer VAE encode until EXEC_DIT whenever this worker lacks full expert residency (`e27bce5`),
        # preventing premature synchronous H2D before `DualPathMorphRouter` evaluates Path A!
        can_run_vae_here = (
            prep_dit
            and not (self.policy.routing_mode == "static_nccl_fission" and self.role == "und")
            and not (self.pager is not None and self.vmm is not None and self.resident_alpha() < 0.999)
        )

        if cmd.get("use_prev_image"):
            if cmd.get("prev_image_path"):
                p_prev = str(cmd["prev_image_path"])
                self._update_image_ctx(main_ctx, self._load_pil(p_prev), update_vae=can_run_vae_here, update_vit=True)
                if prep_dit and not can_run_vae_here:
                    entry["pending_vae_paths"].append(p_prev)
            elif self.policy.use_module3_cow_loopback and entry.get("last_img_u8") is not None:
                self._update_image_ctx(main_ctx, entry["last_img_u8"], update_vae=can_run_vae_here, update_vit=True)
                if prep_dit and not can_run_vae_here:
                    entry["pending_vae_u8"] = True
            elif entry.get("last_img_pil") is not None:
                prev_pil = entry["last_img_pil"]
                if not self.policy.use_module3_cow_loopback:
                    buf = io.BytesIO()
                    prev_pil.save(buf, format="PNG")
                    buf.seek(0)
                    prev_pil = Image.open(buf).convert("RGB")
                    prev_pil.load()
                self._update_image_ctx(main_ctx, prev_pil, update_vae=can_run_vae_here, update_vit=True)

        for rel_img in cmd.get("image_paths", []):
            self._update_image_ctx(main_ctx, self._load_pil(rel_img), update_vae=can_run_vae_here, update_vit=True)
            if prep_dit and not can_run_vae_here:
                entry["pending_vae_paths"].append(str(rel_img))

        prompt = str(cmd["prompt"])
        if prep_dit:
            has_hist = main_ctx["past_key_values"].key_cache[0] is not None
            if has_hist:
                if self.policy.use_module3_cow_loopback:
                    cfg_text_ctx = _slice_prefix_ctx(
                        main_ctx,
                        int(main_ctx["past_key_values"].key_cache[0].shape[0]),
                        self._naive_cache_cls,
                    )
                else:
                    cfg_text_ctx = deepcopy(main_ctx)
                entry["cfg_text_prefix_len"] = int(main_ctx["past_key_values"].key_cache[0].shape[0])
            else:
                cfg_text_ctx = None
                entry["cfg_text_prefix_len"] = 0

            self._update_text_ctx(main_ctx, prompt)
            if cfg_text_ctx is not None and self.policy.use_module3_cow_loopback:
                rebind_cow_prefix(main_ctx["past_key_values"], cfg_text_ctx["past_key_values"])

            cfg_img_ctx = self._new_aux_ctx()
            self._update_text_ctx(cfg_img_ctx, prompt)
            entry["cfg_text_ctx"] = cfg_text_ctx
            entry["cfg_img_ctx"] = cfg_img_ctx
        else:
            self._update_text_ctx(main_ctx, prompt)

        self._update_hwm()
        return bagel_entry_nbytes(entry)

    def decode_ar(self, ctx_id: str, max_tokens: int) -> dict[str, Any]:
        outs = self.decode_ar_batch([{"ctx_id": ctx_id, "max_ar_tokens": max_tokens, "_skip_prepare": True}])
        return outs[0]

    def decode_ar_batch(self, batch_cmds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Token-round continuous interleaving across `B >= 1` concurrent BAGEL AR requests."""
        import torch

        if not batch_cmds:
            return []

        states: list[dict[str, Any]] = []
        end_id = self.pipeline.new_token_ids["eos_token_id"]

        for cmd in batch_cmds:
            cid = str(cmd["ctx_id"])
            enq_s = float(cmd.get("enqueue_time_s", 0.0))
            t_prep0 = time.perf_counter()
            if not cmd.get("_skip_prepare", False):
                self.prepare_turn(cid, cmd)
            entry = self._ctx_store[cid]
            ctx = entry["main_ctx"]
            start_inp = self.pipeline.bagel.prepare_start_tokens(
                ctx["kv_lens"], ctx["ropes"], self.pipeline.new_token_ids
            )
            states.append(
                {
                    "ctx_id": cid,
                    "entry": entry,
                    "ctx": ctx,
                    "curr_tokens": start_inp["packed_start_tokens"].to(self.device),
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
                    st["gen_ids"].append(int(st["curr_tokens"][0].item()))
                    ql = torch.ones_like(st["curr_tokens"])
                    out = self.pipeline.bagel.language_model(
                        packed_text_ids=st["curr_tokens"],
                        query_lens=ql,
                        packed_query_position_ids=st["pos_ids"],
                        past_key_values=st["ctx"]["past_key_values"],
                        update_past_key_values=True,
                        is_causal=True,
                        mode="und",
                    )
                    st["ctx"]["past_key_values"] = out.past_key_values
                    logits = self.pipeline.bagel.language_model.lm_head(out.packed_query_sequence)
                    next_tok = torch.argmax(logits, dim=-1)
                    st["curr_tokens"] = next_tok
                    st["pos_ids"] = st["pos_ids"] + 1
                    reached = len(st["gen_ids"]) >= st["max_tokens"]
                    hit_eos = end_id is not None and int(next_tok[0].item()) == int(end_id)
                    if reached or hit_eos:
                        st["done"] = True

                torch.cuda.synchronize(self.device)
                now_s = time.perf_counter()
                for st in active:
                    st["stamps"].append(now_s)

        self._update_hwm()
        alpha_now = self.resident_alpha()
        outputs: list[dict[str, Any]] = []
        for st in states:
            ctx = st["ctx"]
            entry = st["entry"]
            gen_ids = st["gen_ids"]
            stamps = st["stamps"]
            n_tok = len(gen_ids)
            ctx["kv_lens"] = [kl + n_tok for kl in ctx["kv_lens"]]
            ctx["ropes"] = [r + n_tok for r in ctx["ropes"]]
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

        entry = self._ctx_store[ctx_id]
        main_ctx = entry["main_ctx"]
        cfg_text_ctx = entry.get("cfg_text_ctx")
        cfg_img_ctx = entry.get("cfg_img_ctx")

        if entry.pop("pending_vae_u8", False) and entry.get("last_img_u8") is not None:
            self._update_image_ctx(main_ctx, entry["last_img_u8"], update_vae=True, update_vit=False)
            if cfg_text_ctx is not None:
                self._update_image_ctx(cfg_text_ctx, entry["last_img_u8"], update_vae=True, update_vit=False)

        for pend_img in entry.pop("pending_vae_paths", []) or []:
            pil_v = self._load_pil(pend_img)
            self._update_image_ctx(main_ctx, pil_v, update_vae=True, update_vit=False)
            if cfg_text_ctx is not None:
                self._update_image_ctx(cfg_text_ctx, pil_v, update_vae=True, update_vit=False)

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
        if cfg_text_ctx is not None and cfg_text_ctx["past_key_values"].key_cache[0] is not None:
            sampling.cfg_text_past_key_values = cfg_text_ctx["past_key_values"]
            sampling.cfg_text_kv_metadata = {"ropes": cfg_text_ctx["ropes"], "image_shape": [height, width]}
        if cfg_img_ctx is not None and cfg_img_ctx["past_key_values"].key_cache[0] is not None:
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
                self.prepare_turn(ar_cid, ar_msg)
                ar_entry = self._ctx_store[ar_cid]
                ar_ctx = ar_entry["main_ctx"]
                st = self.pipeline.bagel.prepare_start_tokens(
                    ar_ctx["kv_lens"], ar_ctx["ropes"], self.pipeline.new_token_ids
                )
                task = ArDecodeTask(
                    req_id=str(ar_msg["cmd_id"]),
                    kv_cache=ar_ctx["past_key_values"],
                    curr_tokens=st["packed_start_tokens"].to(self.device),
                    pos_ids=st["packed_query_position_ids"].to(self.device),
                    max_tokens=int(ar_msg.get("max_ar_tokens", 32)),
                    end_token_id=self.pipeline.new_token_ids["eos_token_id"],
                    tpot_slo_ms=float(ar_msg.get("tpot_slo_ms", self.policy.ar_tpot_slo_ms)),
                    arrival_time_s=float(ar_msg.get("enqueue_time_s", now_s)),
                )
                task.last_step_time_s = task.arrival_time_s
                inflight_ar_meta[id(task)] = (str(ar_msg["cmd_id"]), ar_cid, ar_entry)
                tasks.append(task)
            return tasks

        def _flush_completed_ar() -> None:
            if self.sched is None or emit_preempt_ack is None:
                return
            while self.sched.completed_ar_tasks:
                t = self.sched.completed_ar_tasks.pop(0)
                meta_tuple = inflight_ar_meta.pop(id(t), None)
                if meta_tuple is None:
                    continue
                c_id, ar_ctx_id, ar_entry = meta_tuple
                ar_ctx = ar_entry["main_ctx"]
                ar_ctx["past_key_values"] = t.kv_cache
                n_tok = len(t.generated_token_ids)
                ar_ctx["kv_lens"] = [kl + n_tok for kl in ar_ctx["kv_lens"]]
                ar_ctx["ropes"] = [r + n_tok for r in ar_ctx["ropes"]]
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
                        "preempted_mid_dit": True,
                    }
                )

        if self.sched is not None:
            self.sched.reset_telemetry()
            self.sched.ema_layer_ms = 37.0 if height >= 1024 else 10.5
            self.sched.external_arrival_poll = _poll_external_ar
            self.sched.on_task_complete = lambda _t: _flush_completed_ar()

        torch.cuda.synchronize(self.device)
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
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            pil_img = self.pipeline._decode_image_from_latent(
                                self.pipeline.bagel, self.pipeline.vae, latents[0], img_shape
                            )
                        entry["last_img_pil"] = pil_img
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
                                for c_id, _, _ in list(inflight_ar_meta.values()):
                                    emit_preempt_ack(
                                        {"cmd_id": c_id, "rank": self.rank, "status": "oom", "error": str(ar_exc)[:200]}
                                    )
                            else:
                                for c_id, _, _ in list(inflight_ar_meta.values()):
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
                            for c_id, _, _ in list(inflight_ar_meta.values()):
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

        torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._update_hwm()
        return {
            "status": "ok",
            "ctx_id": ctx_id,
            "elapsed_ms": round(elapsed_ms, 2),
            "kv_bytes": bagel_entry_nbytes(entry),
            "resident_alpha": self.resident_alpha(),
            "yield_count": self.sched.yield_count if self.sched is not None else 0,
        }

    def pack_state(self, ctx_id: str, include_cfg_dup: bool = False) -> PackedTransferState:
        entry = self._ctx_store[ctx_id]
        return pack_bagel_multi_context(entry, include_cfg_text_dup=include_cfg_dup, device=self.device)

    def unpack_state(self, new_ctx_id: str, meta: dict[str, Any], raw_payload: Any) -> int:
        old = self._ctx_store.pop(new_ctx_id, None)
        if old is not None:
            _close_ctx_vmm_if_any(old.get("main_ctx"))
        main_factory = (
            (lambda n: VmmNaiveCache(n, self.vmm))
            if self.vmm is not None
            else (lambda n: self._naive_cache_cls(n))
        )
        entry = unpack_bagel_multi_context(
            meta,
            raw_payload,
            main_cache_factory=main_factory,
            aux_cache_factory=lambda n: self._naive_cache_cls(n),
            use_cow_rebind=self.policy.use_module3_cow_loopback,
        )
        self._ctx_store[new_ctx_id] = entry
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

        old = self._ctx_store.pop(ctx_id, None)
        if old is not None:
            _close_ctx_vmm_if_any(old.get("main_ctx"))
            _close_ctx_vmm_if_any(old.get("cfg_text_ctx"))
            _close_ctx_vmm_if_any(old.get("cfg_img_ctx"))
        if self.vmm is None:
            torch.cuda.empty_cache()

    def memory_and_vmm_stats(self) -> dict[str, Any]:
        self._update_hwm()
        return {
            "peak_reserved_mib": round(self._max_reserved_mib, 2),
            "vmm_stats": self.vmm.stats_dict() if self.vmm is not None else {},
            "active_contexts": len(self._ctx_store),
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
