"""Deep `InternVLUCPTGAdapter` (`bridge_mllm_dit`) built atop `LayerExpertPager` + dual-regime HBM accounting."""

from __future__ import annotations

from contextlib import contextmanager
import gc
import inspect
import math
import os
from pathlib import Path
import sys
import time
import types
from typing import Any, Callable

from morphserve.vmm import (
    ExpertPartitionSpec,
    LayerExpertPager,
    NaiveHostOffloader,
    NumaMemfdMirror,
    TriPoolVMM,
    drop_partition_weights,
)

from .base import CPTGModelAdapter, PackedTransferState, compute_vmm_available_kv_bytes

MIB = 1024**2
GIB = 1024**3


def compute_internvlu_memory_plan(
    *,
    total_dev_mib: float,
    free_dev_mib: float,
    base_pytorch_alloc_mib: float,
    mirror_bytes: int,
    policy_super_pool_mib: int,
    policy_fine_kv_pool_mib: int,
    super_handle_mib: int = 64,
    allow_vmm_overflow: bool = True,
    gpu_memory_fraction: float | None = None,
    gpu_memory_quota_mib: float | None = None,
) -> dict[str, Any]:
    """Compute physical `TriPoolVMM` + PyTorch budget plan via C++17 `WorkerBudgetPlanner`."""
    from morphserve._core import call_json, encode_json, get_core_lib

    L = get_core_lib()
    req = {
        "total_dev_mib": float(total_dev_mib),
        "free_dev_mib": float(free_dev_mib),
        "base_pytorch_alloc_mib": float(base_pytorch_alloc_mib),
        "mirror_bytes": int(mirror_bytes),
        "policy_super_pool_mib": int(policy_super_pool_mib),
        "policy_fine_kv_pool_mib": int(policy_fine_kv_pool_mib),
        "super_handle_mib": int(super_handle_mib),
        "allow_vmm_overflow": bool(allow_vmm_overflow),
        "gpu_memory_fraction": None if gpu_memory_fraction is None else float(gpu_memory_fraction),
        "gpu_memory_quota_mib": None if gpu_memory_quota_mib is None else float(gpu_memory_quota_mib),
    }
    return call_json(L, L.ms_budget_internvlu_plan_json, encode_json(req))


def _install_runtime_compat_shims(vendor_dir: Path | None = None) -> None:
    """Install lightweight compatibility shims for offline container execution (`timm`, `transformers>=5`)."""
    if vendor_dir is not None and vendor_dir.is_dir():
        v_str = str(vendor_dir)
        if v_str not in sys.path:
            sys.path.insert(0, v_str)

    try:
        import timm.layers  # noqa: F401
    except ImportError:
        import torch.nn as nn

        class DropPath(nn.Module):
            def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
                super().__init__()
                self.drop_prob = float(drop_prob)
                self.scale_by_keep = bool(scale_by_keep)

            def forward(self, x):
                if self.drop_prob == 0.0 or not self.training:
                    return x
                keep = 1.0 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                rnd = x.new_empty(shape).bernoulli_(keep)
                if keep > 0.0 and self.scale_by_keep:
                    rnd.div_(keep)
                return x * rnd

        timm_mod = types.ModuleType("timm")
        layers_mod = types.ModuleType("timm.layers")
        layers_mod.DropPath = DropPath
        timm_mod.layers = layers_mod
        sys.modules.setdefault("timm", timm_mod)
        sys.modules.setdefault("timm.layers", layers_mod)

    import transformers.image_utils as iu
    from transformers.modeling_utils import PreTrainedModel
    import transformers.processing_utils as pu
    import transformers.utils as tu

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}
    if not hasattr(iu, "concatenate_list"):
        iu.concatenate_list = lambda items: [x for sub in items for x in (sub if isinstance(sub, list) else [sub])]
    if not hasattr(iu, "make_list_of_images"):
        iu.make_list_of_images = iu.make_flat_list_of_images
    if not hasattr(iu, "is_valid_image"):
        iu.is_valid_image = lambda img: img is not None
    if not hasattr(iu, "valid_images"):
        iu.valid_images = lambda imgs: True
    if not hasattr(iu, "validate_preprocess_arguments"):
        iu.validate_preprocess_arguments = lambda *a, **kw: None
    if not hasattr(tu, "is_flash_attn_greater_or_equal_2_10"):
        tu.is_flash_attn_greater_or_equal_2_10 = lambda: False

    try:
        import transformers.models.internvl.processing_internvl as pi

        if not hasattr(pi, "InternVLImagesKwargs"):
            pi.InternVLImagesKwargs = getattr(pu, "ImagesKwargs", dict)
        if not hasattr(pi, "InternVLProcessorKwargs"):
            pi.InternVLProcessorKwargs = getattr(pu, "ProcessingKwargs", dict)
    except ImportError:
        pi_mod = types.ModuleType("transformers.models.internvl.processing_internvl")
        pi_mod.InternVLImagesKwargs = getattr(pu, "ImagesKwargs", dict)
        pi_mod.InternVLProcessorKwargs = getattr(pu, "ProcessingKwargs", dict)
        sys.modules["transformers.models.internvl.processing_internvl"] = pi_mod

    try:
        import transformers.models.internvl.video_processing_internvl as vpi

        if not hasattr(vpi, "InternVLVideoProcessor"):
            vpi.InternVLVideoProcessor = type("InternVLVideoProcessor", (), {})
    except ImportError:
        vpi_mod = types.ModuleType("transformers.models.internvl.video_processing_internvl")
        vpi_mod.InternVLVideoProcessor = type("InternVLVideoProcessor", (), {})
        sys.modules["transformers.models.internvl.video_processing_internvl"] = vpi_mod


def _drop_module_params_to_empty_device(mod: Any, dev: Any) -> None:
    import torch

    for p in mod.parameters():
        p.data = torch.empty(0, dtype=p.dtype, device=dev)
    for b in mod.buffers():
        b.data = b.data.to(dev)


def load_internvlu_pipeline(
    model_dir: Path,
    vendor_dir: Path | None = None,
    device_index: int = 0,
    *,
    defer_decoder_gpu: bool = False,
    defer_vlm_gpu: bool = False,
    drop_vlm_on_load: bool = False,
    drop_vae_on_load: bool = False,
) -> Any:
    """Load `InternVLUPipeline` avoiding redundant GPU weight spikes when paging or fissioning (`c62ec05`)."""
    import torch

    if vendor_dir is None:
        repo_vendor = Path(__file__).resolve().parents[3] / "data/internvlu_pkg"
        if repo_vendor.is_dir():
            vendor_dir = repo_vendor

    _install_runtime_compat_shims(vendor_dir=vendor_dir)
    from diffusers.models import AutoencoderKLQwenImage
    from diffusers.schedulers import DPMSolverMultistepScheduler
    from internvlu import InternVLUPipeline
    from internvlu.diffusion import InternVLUGenerationDecoder
    from internvlu.processing_internvlu import InternVLUProcessor
    from internvlu.vlm import InternVLUChatModel
    from internvlu.vlm.modeling_intern_vit import InternVisionModel

    for cls_m in (InternVLUChatModel, InternVisionModel, InternVLUGenerationDecoder):
        if not getattr(cls_m, "_ms_post_init_patched", False):
            orig_init = cls_m.__init__

            def _make_patched_init(fn):
                def _patched_init(self, *a, **kw):
                    fn(self, *a, **kw)
                    if not hasattr(self, "all_tied_weights_keys") or "all_tied_weights_keys" not in self.__dict__:
                        self.all_tied_weights_keys = {}

                return _patched_init

            cls_m.__init__ = _make_patched_init(orig_init)
            cls_m._ms_post_init_patched = True

    if not getattr(InternVLUProcessor, "_ms_proc_patched", False):
        from internvlu.vlm.constants import IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN

        def _patched_proc_init(
            self,
            image_processor=None,
            image_gen_processor=None,
            tokenizer=None,
            video_processor=None,
            image_seq_length: int = 256,
            chat_template: str | None = None,
            blank_messages: dict | None = None,
            **kwargs,
        ) -> None:
            if blank_messages is None:
                blank_messages = {
                    "text_blank_message": "Generate an image based on reference images.",
                    "pure_blank_message": "Here is a random image <img_uncond>:",
                }
            self.image_seq_length = image_seq_length
            self.start_image_token = IMG_START_TOKEN
            self.end_image_token = IMG_END_TOKEN
            self.image_token = IMG_CONTEXT_TOKEN
            self.image_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            self.blank_messages = blank_messages
            self.image_processor = image_processor
            self.image_gen_processor = image_gen_processor
            self.tokenizer = tokenizer
            self.video_processor = video_processor
            self.chat_template = chat_template
            self.template_name = "qwen2_5-chat-v3"

        def _patched_merge_kwargs(self, processing_kwargs_cls=None, tokenizer_init_kwargs=None, **kwargs):
            if getattr(self, "tokenizer", None) is not None:
                self.tokenizer.padding_side = kwargs.get("padding_side", "left")
            text_kw = {
                k: v
                for k, v in kwargs.items()
                if k in ("padding", "truncation", "max_length", "return_tensors", "add_special_tokens")
            }
            img_kw: dict[str, Any] = {"crop_to_patches": False}
            if "return_tensors" in kwargs:
                img_kw["return_tensors"] = kwargs["return_tensors"]
            return {"text_kwargs": text_kw, "images_kwargs": img_kw, "videos_kwargs": {}}

        InternVLUProcessor.__init__ = _patched_proc_init
        InternVLUProcessor._merge_kwargs = _patched_merge_kwargs
        InternVLUProcessor._ms_proc_patched = True

    torch.cuda.set_device(device_index)
    dev = torch.device(f"cuda:{device_index}")

    def _load_bf16_pretrained(cls_obj: Any, sub_path: Path) -> Any:
        try:
            mod = cls_obj.from_pretrained(
                str(sub_path),
                dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
                local_files_only=True,
            )
        except TypeError:
            mod = cls_obj.from_pretrained(
                str(sub_path),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
                local_files_only=True,
            )
        # Explicitly cast floating-point parameters to BF16 (`transformers>=5` ignores `torch_dtype=`
        # on custom `PreTrainedModel.__init__` submodules when `low_cpu_mem_usage=False`), while
        # preserving FP32 rotary frequency buffers (`inv_freq` / `freqs`).
        for p in mod.parameters():
            if p.is_floating_point() and p.dtype != torch.bfloat16:
                p.data = p.data.to(dtype=torch.bfloat16)
        for bname, b in mod.named_buffers():
            if b.is_floating_point() and b.dtype != torch.bfloat16 and "freq" not in bname.lower():
                b.data = b.data.to(dtype=torch.bfloat16)
        return mod

    vlm = _load_bf16_pretrained(InternVLUChatModel, model_dir / "vlm")
    generation_decoder = _load_bf16_pretrained(InternVLUGenerationDecoder, model_dir / "generation_decoder")
    vae = _load_bf16_pretrained(AutoencoderKLQwenImage, model_dir / "vae")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        str(model_dir / "scheduler"),
        local_files_only=True,
    )

    class _MutableConfigDict(dict):
        def __getattr__(self, name: str) -> Any:
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name: str, value: Any) -> None:
            self[name] = value

    scheduler._internal_dict = _MutableConfigDict(dict(scheduler.config))

    processor = InternVLUProcessor.from_pretrained(
        str(model_dir / "processor"),
        local_files_only=True,
    )

    lm = getattr(vlm, "language_model", None)
    if lm is not None and hasattr(lm, "forward"):
        orig_lm_fwd = lm.forward
        sig_params = inspect.signature(orig_lm_fwd).parameters
        if "padding_type" not in sig_params:
            def _safe_lm_fwd(*a, **kw):
                kw.pop("padding_type", None)
                return orig_lm_fwd(*a, **kw)

            lm.forward = _safe_lm_fwd
        if getattr(lm, "generation_config", None) is not None and lm.generation_config.pad_token_id is None:
            lm.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    pipe = InternVLUPipeline(
        vlm=vlm,
        generation_decoder=generation_decoder,
        vae=vae,
        scheduler=scheduler,
        processor=processor,
    )

    if drop_vlm_on_load:
        _drop_module_params_to_empty_device(pipe.vlm, dev)
    elif defer_vlm_gpu:
        for b in pipe.vlm.buffers():
            b.data = b.data.to(dev)
    else:
        pipe.vlm.to(dev)

    if drop_vae_on_load:
        _drop_module_params_to_empty_device(pipe.vae, dev)
    else:
        pipe.vae.to(dev)

    if defer_decoder_gpu:
        # Keep 5 GiB decoder parameters on CPU for direct copy into NumaMemfdMirror / CPU offloader
        # without ever touching GPU `cudaMalloc` during model load!
        for b in pipe.generation_decoder.buffers():
            b.data = b.data.to(dev)
    else:
        pipe.generation_decoder.to(dev)

    # Materialize non-persistent plain tensor attributes (e.g. UnifiedMSRoPE.all_inv_freqs) (`c62ec05`)
    for root_mod in (pipe.vlm, pipe.generation_decoder, pipe.vae):
        for mod in root_mod.modules():
            if hasattr(mod, "all_inv_freqs") and hasattr(mod, "axes_dim") and hasattr(mod, "theta"):
                inv_f = getattr(mod, "all_inv_freqs", None)
                if inv_f is None or getattr(inv_f, "is_meta", False) or inv_f.device != dev:
                    inv_list = [
                        1.0
                        / torch.pow(
                            mod.theta,
                            torch.arange(0, dim, 2, device=dev, dtype=torch.float32).div(dim),
                        )
                        for dim in mod.axes_dim
                    ]
                    mod.all_inv_freqs = torch.cat(inv_list, dim=0).float()

    pipe.vlm.eval()
    pipe.generation_decoder.eval()
    pipe.vae.eval()
    gc.collect()
    torch.cuda.empty_cache()
    return pipe


_DTYPE_NAME_MAP = {
    "torch.bfloat16": "bfloat16",
    "torch.float16": "float16",
    "torch.float32": "float32",
    "torch.int64": "int64",
    "torch.int32": "int32",
    "torch.bool": "bool",
    "torch.uint8": "uint8",
}


def pack_internvlu_bridge_state(diffusion_inputs: dict[str, Any], device: Any) -> tuple[dict[str, Any], Any]:
    """Pack `InternVL-U` `diffusion_inputs` bridge tensors into a flat contiguous `uint8` CUDA tensor."""
    import torch

    chunks: list[Any] = []
    offset = 0

    def _rec(key: str, val: Any) -> Any:
        nonlocal offset
        if val is None:
            return {"kind": "none"}
        if torch.is_tensor(val):
            t_c = val.to(device).contiguous()
            u8 = t_c.view(torch.uint8).reshape(-1)
            nb = int(u8.numel())
            node = {
                "kind": "tensor",
                "dtype": _DTYPE_NAME_MAP[str(t_c.dtype)],
                "shape": list(t_c.shape),
                "offset": offset,
                "nbytes": nb,
            }
            if nb > 0:
                chunks.append(u8)
                offset += nb
            return node
        if isinstance(val, list):
            return {"kind": "list", "items": [_rec(f"{key}.{i}", x) for i, x in enumerate(val)]}
        return {"kind": "const", "value": val}

    tree = {k: _rec(k, v) for k, v in diffusion_inputs.items()}
    flat = torch.cat(chunks, dim=0) if chunks else torch.empty((0,), dtype=torch.uint8, device=device)
    meta = {"total_bytes": int(flat.numel()), "tree": tree}
    return meta, flat


def unpack_internvlu_bridge_state(meta: dict[str, Any], flat_u8: Any) -> dict[str, Any]:
    """Reconstruct `InternVL-U` `diffusion_inputs` dictionary from flat `uint8` CUDA buffer."""
    import torch

    dtype_lookup = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "int64": torch.int64,
        "int32": torch.int32,
        "bool": torch.bool,
        "uint8": torch.uint8,
    }

    def _dec(node: dict[str, Any]) -> Any:
        kind = node["kind"]
        if kind == "none":
            return None
        if kind == "tensor":
            dt = dtype_lookup[node["dtype"]]
            nb = int(node["nbytes"])
            off = int(node["offset"])
            if nb == 0:
                return torch.empty(node["shape"], dtype=dt, device=flat_u8.device)
            return flat_u8[off : off + nb].clone().view(dt).reshape(node["shape"])
        if kind == "list":
            return [_dec(x) for x in node["items"]]
        return node["value"]

    return {k: _dec(v) for k, v in meta["tree"].items()}


def _trim_internvlu_generated_tokens(seq: Any, tokenizer: Any) -> list[int]:
    """Strip right-padded PAD/EOS/`<|im_end|>`/`<img>` tokens so batch length never distorts `tpots_ms`."""
    raw_ids = [int(x) for x in (seq.detach().cpu().tolist() if hasattr(seq, "detach") else list(seq))]
    if not raw_ids:
        return []

    stop_ids: set[int] = set()
    pad_id: int | None = None
    if tokenizer is not None:
        raw_pad = getattr(tokenizer, "pad_token_id", None)
        if isinstance(raw_pad, int) and raw_pad >= 0:
            pad_id = int(raw_pad)
        raw_eos = getattr(tokenizer, "eos_token_id", None)
        if isinstance(raw_eos, int) and raw_eos >= 0:
            stop_ids.add(int(raw_eos))
        elif isinstance(raw_eos, (list, tuple, set)):
            for eid in raw_eos:
                if isinstance(eid, int) and eid >= 0:
                    stop_ids.add(int(eid))
        conv_fn = getattr(tokenizer, "convert_tokens_to_ids", None)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if callable(conv_fn):
            for tok_str in ("<|im_end|>", "<|endoftext|>", "<img>"):
                try:
                    tid = conv_fn(tok_str)
                except Exception:
                    tid = None
                if isinstance(tid, int) and tid >= 0 and tid != unk_id:
                    stop_ids.add(int(tid))

    # Truncate right after the first semantic stop token (`eos` / `<|im_end|>` / `<img>`) when `pad_id != stop_id`,
    # then strip trailing right-padding tokens.
    non_pad_stops = stop_ids - ({pad_id} if pad_id is not None else set())
    trimmed = raw_ids
    for idx, tid in enumerate(raw_ids):
        if tid in non_pad_stops:
            trimmed = raw_ids[: idx + 1]
            break
    if pad_id is not None:
        while len(trimmed) > 1 and trimmed[-1] == pad_id:
            trimmed.pop()
    return trimmed


class InternVLUCPTGAdapter(CPTGModelAdapter):
    """Production `InternVL-U` (`bridge_mllm_dit`) adapter running inside `MorphServeClusterEngine`."""

    architecture = "bridge_mllm_dit"

    def __init__(
        self,
        rank: int,
        policy: Any,
        model_dir: Path,
        bench_root: Path,
        numa_node: int = 0,
    ) -> None:
        import torch

        self.rank = int(rank)
        self.policy = policy
        self.model_dir = Path(model_dir)
        self.bench_root = Path(bench_root)
        self.numa_node = int(numa_node)
        self.role = policy.role_of(self.rank)
        self.device = torch.device(f"cuda:{self.rank}")

        is_static_fission = policy.routing_mode == "static_nccl_fission"
        raw_frac = getattr(policy, "gpu_memory_fraction", None)
        raw_quota = getattr(policy, "gpu_memory_quota_mib", None)
        copage_vlm = bool(
            (not is_static_fission)
            and policy.use_tripool_vmm
            and policy.use_expert_pager
            and ((raw_quota is not None and float(raw_quota) > 0.0) or (raw_frac is not None and float(raw_frac) < 0.999))
        )
        defer_decoder = bool(
            policy.use_expert_pager or policy.use_naive_offload or (is_static_fission and self.role == "und")
        )
        drop_vlm = bool(is_static_fission and self.role == "gen")
        drop_vae = bool(is_static_fission and self.role == "und")

        # Enforce startup cap immediately during `load_internvlu_pipeline` (even when `use_tripool_vmm=True`),
        # then tighten PyTorch's fraction in `bind_runtime_resources` before `TriPoolVMM` allocates `cuMemCreate`!
        if raw_frac is not None:
            torch.cuda.set_per_process_memory_fraction(float(raw_frac), self.device)

        vendor_p = Path(policy.vendor_dir) if getattr(policy, "vendor_dir", None) else None
        self.pipe = load_internvlu_pipeline(
            self.model_dir,
            vendor_dir=vendor_p,
            device_index=self.rank,
            defer_decoder_gpu=defer_decoder,
            defer_vlm_gpu=copage_vlm,
            drop_vlm_on_load=drop_vlm,
            drop_vae_on_load=drop_vae,
        )

        self.pager: LayerExpertPager | None = None
        self.vlm_pager: LayerExpertPager | None = None
        self.naive_offloader: NaiveHostOffloader | None = None
        self.vmm: TriPoolVMM | None = None
        self.pacer: Any = None
        self._ctx_store: dict[str, dict[str, Any]] = {}
        self._vmm_initial_pool_mib: float = 0.0
        self._memory_plan: dict[str, Any] = {}
        self._quota_constrained: bool = False
        self._max_reserved_mib = 0.0
        self._max_pytorch_reserved_mib = 0.0

        dec_spec = ExpertPartitionSpec.for_internvlu(
            self.pipe,
            super_handle_mib=policy.super_handle_mib,
            wrap_vae=policy.use_scoped_vae,
        )
        self.num_expert_layers = dec_spec.num_layers

        if is_static_fission:
            if self.role == "und":
                drop_partition_weights(dec_spec, device=self.device)
        elif policy.use_naive_offload:
            self.naive_offloader = NaiveHostOffloader(dec_spec, device=self.device)
            self.naive_offloader.offload_to_cpu()
        elif policy.use_expert_pager:
            self.pager = LayerExpertPager(dec_spec, numa_node=self.numa_node)
            if copage_vlm:
                vlm_spec = ExpertPartitionSpec.for_internvlu_vlm(
                    self.pipe,
                    super_handle_mib=policy.super_handle_mib,
                    install_module_hooks=False,
                )
                self.vlm_pager = LayerExpertPager(vlm_spec, numa_node=self.numa_node)

    @property
    def total_mirror_bytes(self) -> int:
        dec_b = self.pager.total_mirror_bytes if self.pager is not None else 0
        vlm_b = self.vlm_pager.total_mirror_bytes if self.vlm_pager is not None else 0
        return dec_b + vlm_b

    @property
    def max_stage_mirror_bytes(self) -> int:
        dec_b = self.pager.total_mirror_bytes if self.pager is not None else 0
        vlm_b = self.vlm_pager.total_mirror_bytes if self.vlm_pager is not None else 0
        return max(dec_b, vlm_b)

    def attach_expert_mirror(self, shared_mirror: NumaMemfdMirror | None, populate: bool = True) -> None:
        if self.pager is None:
            return
        if shared_mirror is not None:
            self.pager.offload_to_shared_mirror(shared_mirror, populate=populate, mirror_offset=0)
            if self.vlm_pager is not None:
                self.vlm_pager.offload_to_shared_mirror(
                    shared_mirror,
                    populate=populate,
                    mirror_offset=self.pager.total_mirror_bytes,
                )
        else:
            self.pager.offload_to_host_mirror()
            if self.vlm_pager is not None:
                self.vlm_pager.offload_to_host_mirror()

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

    def available_kv_bytes(self, *, for_dit: bool = False) -> int:
        import torch

        if self.vmm is not None:
            return compute_vmm_available_kv_bytes(
                self.vmm,
                has_pageable_experts=(self.pager is not None),
                for_dit=for_dit,
            )
        if not torch.cuda.is_available():
            return 0
        free_drv, total_drv = torch.cuda.mem_get_info(self.device)
        if getattr(self.policy, "gpu_memory_fraction", None) is not None:
            quota_b = int(total_drv * float(self.policy.gpu_memory_fraction))
            alloc_b = int(torch.cuda.memory_allocated(self.device))
            return max(0, quota_b - alloc_b - 256 * MIB)
        return max(0, int(free_drv) - 1024 * MIB)

    def bind_runtime_resources(self, pacer: Any | None = None) -> dict[str, Any]:
        import torch

        self.pacer = pacer
        gc.collect()
        torch.cuda.empty_cache()

        free_b, total_b = torch.cuda.mem_get_info(self.device)
        total_dev_mib = float(total_b / MIB)
        free_dev_mib = float(free_b / MIB)
        base_pt_mib = float(torch.cuda.memory_allocated(self.device) / MIB)

        if self.policy.use_tripool_vmm:
            plan = compute_internvlu_memory_plan(
                total_dev_mib=total_dev_mib,
                free_dev_mib=free_dev_mib,
                base_pytorch_alloc_mib=base_pt_mib,
                mirror_bytes=self.max_stage_mirror_bytes,
                policy_super_pool_mib=int(self.policy.super_pool_mib),
                policy_fine_kv_pool_mib=int(self.policy.fine_kv_pool_mib),
                super_handle_mib=int(self.policy.super_handle_mib),
                allow_vmm_overflow=bool(self.policy.allow_vmm_overflow),
                gpu_memory_fraction=getattr(self.policy, "gpu_memory_fraction", None),
                gpu_memory_quota_mib=getattr(self.policy, "gpu_memory_quota_mib", None),
            )
            self._memory_plan = plan
            self._quota_constrained = bool(plan["quota_constrained"])
            if plan["pytorch_memory_fraction"] is not None:
                torch.cuda.set_per_process_memory_fraction(float(plan["pytorch_memory_fraction"]), self.device)

            self._vmm_initial_pool_mib = float(plan["vmm_physical_pool_mib"])
            self.vmm = TriPoolVMM(
                device=self.rank,
                super_pool_mib=int(plan["super_pool_mib"]),
                fine_kv_pool_mib=int(plan["fine_kv_pool_mib"]),
                super_handle_mib=int(self.policy.super_handle_mib),
                transient_va_gib=int(self.policy.transient_va_gib),
                kv_va_gib=int(self.policy.kv_va_gib),
                allow_overflow=bool(plan["allow_overflow"]),
            )
            if self.pager is not None:
                init_alpha = 0.0 if self.vlm_pager is not None else float(self.policy.initial_alphas.get(self.rank, 1.0))
                if not bool(plan["allow_overflow"]):
                    max_fit_layers = int(plan["super_pool_mib"]) // max(1, self.pager.layer_stride_mib)
                    max_fit_alpha = min(1.0, float(max_fit_layers) / float(max(1, self.pager.num_layers)))
                    init_alpha = min(init_alpha, max_fit_alpha)
                self.pager.bind_vmm(self.vmm, initial_alpha=init_alpha)
            if self.vlm_pager is not None:
                vlm_fit_layers = int(plan["super_pool_mib"]) // max(1, self.vlm_pager.layer_stride_mib)
                vlm_init_alpha = min(1.0, float(vlm_fit_layers) / float(max(1, self.vlm_pager.num_layers)))
                self.vlm_pager.bind_vmm(self.vmm, initial_alpha=vlm_init_alpha)
        else:
            self._quota_constrained = getattr(self.policy, "gpu_memory_fraction", None) is not None

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        pt_mib = float(torch.cuda.memory_reserved(self.device) / MIB)
        self._max_pytorch_reserved_mib = pt_mib
        self._max_reserved_mib = pt_mib + self._vmm_physical_pool_mib()
        return {
            "resident_alpha": self.resident_alpha(),
            "reserved_mib": round(self._max_pytorch_reserved_mib, 2),
            "combined_physical_mib": round(self._max_reserved_mib, 2),
            "pytorch_reserved_mib": round(self._max_pytorch_reserved_mib, 2),
            "vmm_physical_pool_mib": round(self._vmm_physical_pool_mib(), 2),
            "kv_free_bytes": self.available_kv_bytes(),
            "memory_plan": dict(self._memory_plan),
        }

    def warmup(self) -> None:
        return None

    def _update_hwm(self) -> None:
        import torch

        cur = float(torch.cuda.memory_reserved(self.device) / MIB)
        pk = float(torch.cuda.max_memory_reserved(self.device) / MIB)
        pt_peak = max(cur, pk)
        self._max_pytorch_reserved_mib = max(self._max_pytorch_reserved_mib, pt_peak)
        self._max_reserved_mib = max(self._max_reserved_mib, pt_peak + self._vmm_physical_pool_mib())

    def _load_pil_if_any(self, cmd: dict[str, Any]) -> Any:
        from PIL import Image

        paths = cmd.get("image_paths") or []
        if not paths:
            return None
        rel = str(paths[0])
        p = self.bench_root / rel if not os.path.isabs(rel) else Path(rel)
        if not p.is_file():
            return None
        with Image.open(p) as raw:
            im = raw.convert("RGB")
            im.load()
            return im

    @contextmanager
    def _vlm_execution_scope(self):
        """Pin & materialize `vlm_pager` during VLM execution while leaving `decoder` unpinned for eviction."""
        if self.vlm_pager is None or self.vmm is None:
            yield
            return
        if self.pager is not None and self.pager.expert_id >= 0:
            self.vmm.set_expert_pinned(self.pager.expert_id, False)
        pacer_cm = None
        if (
            self.pacer is not None
            and self.vmm.expert_resident_count(self.vlm_pager.expert_id) < self.vlm_pager.num_layers
        ):
            pacer_cm = self.pacer.acquire_process_slot(self.rank)
        try:
            if pacer_cm is not None:
                pacer_cm.__enter__()
            with self.vlm_pager.generation_mode(prefetch_all=True):
                yield
        finally:
            if pacer_cm is not None:
                pacer_cm.__exit__(None, None, None)

    def prepare_turn(self, ctx_id: str, cmd: dict[str, Any]) -> int:
        import torch

        prompt = str(cmd["prompt"])
        prep_dit = bool(cmd.get("prepare_for_dit", False))
        prev_ent = self._ctx_store.get(ctx_id)
        pil_img = self._load_pil_if_any(cmd)
        if pil_img is None and prev_ent is not None and not (cmd.get("image_paths")):
            pil_img = prev_ent.get("pil_img")
        if not prep_dit:
            self._ctx_store[ctx_id] = {"prompt": prompt, "pil_img": pil_img, "bridge_bytes": 4096}
            return 4096

        if self._quota_constrained:
            torch.cuda.empty_cache()

        h = int(cmd.get("height", 512))
        w = int(cmd.get("width", 512))
        stride = self.pipe.processor.image_gen_processor.merge_size * self.pipe.processor.image_gen_processor.patch_size
        gh, gw = (h // stride) * stride, (w // stride) * stride
        with self._vlm_execution_scope(), torch.inference_mode():
            inp = self.pipe.processor(
                prompt=prompt,
                image=pil_img,
                generation_mode="image",
                padding=True,
                return_tensors="pt",
                height=gh,
                width=gw,
            )
            inp = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inp.items()}
            v_out = self.pipe.vlm.generate_hidden_states(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                pixel_values=inp.get("pixel_values"),
            )
            diff_inp = self.pipe._prepare_diffusion_inputs(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                pixel_values=inp.get("pixel_values"),
                pixel_values_gen=inp.get("pixel_values_gen"),
                image_grid_thw_gen=inp.get("image_grid_thw_gen"),
                generation_flags=inp["generation_flags"],
                vlm_hidden_states=v_out.hidden_states,
            )
            del inp, v_out
            meta, flat = pack_internvlu_bridge_state(diff_inp, device=self.device)
        torch.cuda.current_stream(self.device).synchronize()
        if self._quota_constrained:
            torch.cuda.empty_cache()
        b_bytes = int(meta["total_bytes"])
        self._ctx_store[ctx_id] = {
            "prompt": prompt,
            "pil_img": pil_img,
            "diff_inp": diff_inp,
            "gh": gh,
            "gw": gw,
            "packed_meta": meta,
            "packed_flat": flat,
            "bridge_bytes": b_bytes,
        }
        self._update_hwm()
        return b_bytes

    def decode_ar(self, ctx_id: str, max_tokens: int) -> dict[str, Any]:
        import torch

        entry = self._ctx_store[ctx_id]
        prompt = entry["prompt"]
        pil_img = entry.get("pil_img")
        if self._quota_constrained:
            torch.cuda.empty_cache()
        t0 = time.perf_counter()
        with self._vlm_execution_scope(), torch.inference_mode():
            out = self.pipe(
                prompt=prompt,
                image=pil_img,
                generation_mode="text",
                max_new_tokens=int(max_tokens),
                do_sample=False,
            )
        torch.cuda.current_stream(self.device).synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        tok_seq = _trim_internvlu_generated_tokens(out.generate_output[0], self.pipe.tokenizer)
        n_tok = max(1, int(len(tok_seq)))
        txt = self.pipe.tokenizer.decode(tok_seq, skip_special_tokens=True) if tok_seq else ""
        per_tok = dt_ms / float(n_tok)
        if self._quota_constrained:
            torch.cuda.empty_cache()
        self._update_hwm()
        return {
            "text": txt,
            "tokens": n_tok,
            "ttft_ms": round(per_tok, 2),
            "tpots_ms": [round(per_tok, 2) for _ in range(max(0, n_tok - 1))],
            "kv_bytes": int(entry.get("bridge_bytes", 4096)),
            "resident_alpha": self.resident_alpha(),
            "kv_free_bytes": self.available_kv_bytes(),
        }

    def decode_ar_batch(self, batch_cmds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Batched `[B, L]` VLM greedy decoding for concurrent text-only requests; CoT-aware fallback for multimodal."""
        import torch

        if not batch_cmds:
            return []

        if any(bool(c.get("prepare_for_dit")) for c in batch_cmds):
            # Visual-CoT (`cot_then_dit`): decode CoT text FIRST, then condition the diffusion bridge state on it!
            results: list[dict[str, Any]] = []
            for cmd in batch_cmds:
                cid = str(cmd["ctx_id"])
                enq_s = float(cmd.get("enqueue_time_s", 0.0))
                t_cmd0 = time.perf_counter()
                if not cmd.get("_skip_prepare", False):
                    self.prepare_turn(cid, dict(cmd, prepare_for_dit=False))
                ar_out = self.decode_ar(cid, max(1, int(cmd.get("max_ar_tokens", 32))))
                if bool(cmd.get("prepare_for_dit")):
                    base_prompt = str(cmd.get("prompt", self._ctx_store[cid].get("prompt", "")))
                    cot_prompt = f"{base_prompt}\n{ar_out['text']}".strip() if ar_out.get("text") else base_prompt
                    b_bytes = self.prepare_turn(cid, dict(cmd, prompt=cot_prompt, prepare_for_dit=True))
                    ar_out["kv_bytes"] = b_bytes
                    ar_out["kv_free_bytes"] = self.available_kv_bytes()
                    ar_out["resident_alpha"] = self.resident_alpha()
                if enq_s > 0.0:
                    ar_out["ttft_ms"] = round(float(ar_out["ttft_ms"]) + max(0.0, (t_cmd0 - enq_s) * 1000.0), 2)
                results.append(ar_out)
            return results

        if len(batch_cmds) <= 1 or any(bool(c.get("image_paths")) for c in batch_cmds):
            return super().decode_ar_batch(batch_cmds)

        t_start = time.perf_counter()
        prompts: list[str] = []
        max_toks: list[int] = []
        for cmd in batch_cmds:
            cid = str(cmd["ctx_id"])
            if not cmd.get("_skip_prepare", False):
                self.prepare_turn(cid, cmd)
            prompts.append(str(self._ctx_store[cid]["prompt"]))
            max_toks.append(max(1, int(cmd.get("max_ar_tokens", 32))))

        if self._quota_constrained:
            torch.cuda.empty_cache()

        t_gen0 = time.perf_counter()
        with self._vlm_execution_scope(), torch.inference_mode():
            out = self.pipe(
                prompt=prompts,
                image=None,
                generation_mode="text",
                max_new_tokens=max(max_toks),
                do_sample=False,
            )
        torch.cuda.current_stream(self.device).synchronize()
        dt_ms = (time.perf_counter() - t_gen0) * 1000.0
        if self._quota_constrained:
            torch.cuda.empty_cache()
        self._update_hwm()

        alpha_now = self.resident_alpha()
        free_kv_now = self.available_kv_bytes()
        trimmed_seqs = [
            _trim_internvlu_generated_tokens(out.generate_output[idx][: max_toks[idx]], self.pipe.tokenizer)
            for idx in range(len(batch_cmds))
        ]
        max_steps_executed = max((max(1, len(s)) for s in trimmed_seqs), default=1)
        per_step_ms = dt_ms / float(max_steps_executed)
        results = []
        for idx, cmd in enumerate(batch_cmds):
            enq_s = float(cmd.get("enqueue_time_s", 0.0))
            q_wait_ms = max(0.0, (t_start - enq_s) * 1000.0) if enq_s > 0.0 else 0.0
            seq = trimmed_seqs[idx]
            n_tok = max(1, int(len(seq)))
            txt = self.pipe.tokenizer.decode(seq, skip_special_tokens=True) if seq else ""
            results.append(
                {
                    "text": txt,
                    "tokens": n_tok,
                    "ttft_ms": round(q_wait_ms + per_step_ms, 2),
                    "tpots_ms": [round(per_step_ms, 2) for _ in range(max(0, n_tok - 1))],
                    "kv_bytes": 4096,
                    "resident_alpha": alpha_now,
                    "kv_free_bytes": free_kv_now,
                    "batched_width": len(batch_cmds),
                }
            )
        return results

    def pack_state(self, ctx_id: str, include_cfg_dup: bool = False) -> PackedTransferState:
        entry = self._ctx_store[ctx_id]
        if "packed_flat" in entry and entry["packed_flat"] is not None:
            meta = dict(entry["packed_meta"], gh=entry["gh"], gw=entry["gw"])
            flat = entry.pop("packed_flat")
        else:
            raw_meta, flat = pack_internvlu_bridge_state(entry["diff_inp"], device=self.device)
            meta = dict(raw_meta, gh=entry["gh"], gw=entry["gw"])
        total_b = int(meta["total_bytes"])
        meta["logical_bytes"] = total_b
        return PackedTransferState(meta=meta, payload=flat, logical_bytes=total_b)

    def unpack_state(self, new_ctx_id: str, meta: dict[str, Any], raw_payload: Any) -> int:
        diff_inp = unpack_internvlu_bridge_state(meta, raw_payload)
        total_b = int(meta["total_bytes"])
        self._ctx_store[new_ctx_id] = {
            "prompt": "",
            "diff_inp": diff_inp,
            "gh": int(meta["gh"]),
            "gw": int(meta["gw"]),
            "bridge_bytes": total_b,
        }
        self._update_hwm()
        return total_b

    def generate_dit(
        self,
        ctx_id: str,
        cmd: dict[str, Any],
        poll_preempt_cmds: Callable[[], list[dict[str, Any]]] | None = None,
        emit_preempt_ack: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        import numpy as np
        from PIL import Image
        import torch

        entry = self._ctx_store[ctx_id]
        stride = self.pipe.processor.image_gen_processor.merge_size * self.pipe.processor.image_gen_processor.patch_size
        req_h = int(cmd.get("height", entry.get("gh", 512)))
        req_w = int(cmd.get("width", entry.get("gw", 512)))
        req_gh, req_gw = (req_h // stride) * stride, (req_w // stride) * stride

        if "diff_inp" not in entry or (
            (req_gh, req_gw) != (int(entry["gh"]), int(entry["gw"])) and getattr(self.pipe, "vlm", None) is not None
        ):
            eff_prompt = str(entry.get("prompt") or cmd.get("prompt", ""))
            self.prepare_turn(
                ctx_id,
                dict(cmd, prompt=eff_prompt, height=req_gh, width=req_gw, prepare_for_dit=True),
            )
            entry = self._ctx_store[ctx_id]

        diff_inp = entry["diff_inp"]
        entry.pop("packed_flat", None)
        gh, gw = int(entry["gh"]), int(entry["gw"])
        steps = int(cmd.get("dit_steps", 16))
        seed = int(cmd.get("seed", 42))
        save_tmp_path = cmd.get("save_tmp_image_path")

        if self._quota_constrained:
            torch.cuda.empty_cache()

        t0 = time.perf_counter()
        if self.naive_offloader is not None:
            self.naive_offloader.swap_in_sync()

        pacer_cm = None
        if (
            self.pacer is not None
            and self.pager is not None
            and self.vmm is not None
            and self.vmm.expert_resident_count(self.pager.expert_id) < self.pager.num_layers
        ):
            pacer_cm = self.pacer.acquire_process_slot(self.rank)

        gen = torch.Generator(device=self.device).manual_seed(seed)
        try:
            if pacer_cm is not None:
                pacer_cm.__enter__()
            if self.vlm_pager is not None and self.vmm is not None and self.vlm_pager.expert_id >= 0:
                self.vmm.set_expert_pinned(self.vlm_pager.expert_id, False)
            with torch.inference_mode():
                if self.pager is not None:
                    with self.pager.generation_mode(prefetch_all=True):
                        # Bucket 0 holds `encoder_padding_token` and `decoder_projector`, which execute in
                        # `generation_decoder.prepare_forward_input` BEFORE `transformer_blocks[0]`'s pre-hook!
                        self.pager.wait_layer_if_needed(0, "prepare_forward_input")
                        latents_out = self.pipe.image_pipeline(
                            **diff_inp,
                            all_cfg_scale=4.5,
                            part_cfg_scale=2.0,
                            num_inference_steps=steps,
                            num_images_per_prompt=1,
                            use_resolution_binning=False,
                            height=gh,
                            width=gw,
                            generator=gen,
                            output_type="latent",
                        ).images
                    # Run QwenImage VAE decode OUTSIDE `generation_mode` so decoder blocks are unpinned
                    # under 12 GiB quota while staying 100% resident under ample 40/80 GB `super_pool`!
                    lat_bf16 = latents_out.to(torch.bfloat16)
                    del latents_out
                    if self._quota_constrained:
                        torch.cuda.empty_cache()
                    imgs_t = self.pipe.image_pipeline.latents_to_pixels(lat_bf16)
                    del lat_bf16
                else:
                    imgs_t = self.pipe.image_pipeline(
                        **diff_inp,
                        all_cfg_scale=4.5,
                        part_cfg_scale=2.0,
                        num_inference_steps=steps,
                        num_images_per_prompt=1,
                        use_resolution_binning=False,
                        height=gh,
                        width=gw,
                        generator=gen,
                        generator_device=None,
                    ).images if "generator_device" in inspect.signature(self.pipe.image_pipeline.__call__).parameters else self.pipe.image_pipeline(
                        **diff_inp,
                        all_cfg_scale=4.5,
                        part_cfg_scale=2.0,
                        num_inference_steps=steps,
                        num_images_per_prompt=1,
                        use_resolution_binning=False,
                        height=gh,
                        width=gw,
                        generator=gen,
                    ).images
        finally:
            if pacer_cm is not None:
                pacer_cm.__exit__(None, None, None)
            if self.naive_offloader is not None:
                self.naive_offloader.drop_from_gpu()

        torch.cuda.current_stream(self.device).synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if save_tmp_path:
            out_c = ((127.5 * imgs_t[0] + 128.0) / 255.0).clamp(0, 1)
            pil = Image.fromarray((out_c.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            Path(save_tmp_path).parent.mkdir(parents=True, exist_ok=True)
            pil.save(save_tmp_path, format="PNG")
        del imgs_t
        if self._quota_constrained:
            torch.cuda.empty_cache()

        self._update_hwm()
        return {
            "status": "ok",
            "ctx_id": ctx_id,
            "elapsed_ms": round(elapsed_ms, 2),
            "kv_bytes": int(entry.get("bridge_bytes", 0)),
            "resident_alpha": self.resident_alpha(),
            "kv_free_bytes": self.available_kv_bytes(),
            "yield_count": 0,
        }

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

        self._ctx_store.pop(ctx_id, None)
        if (self.vmm is None or self._quota_constrained) and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def memory_and_vmm_stats(self) -> dict[str, Any]:
        self._update_hwm()
        vmm_st = self.vmm.stats_dict() if self.vmm is not None else {}
        return {
            "peak_reserved_mib": round(self._max_pytorch_reserved_mib, 2),
            "combined_physical_mib": round(self._max_reserved_mib, 2),
            "pytorch_reserved_mib": round(self._max_pytorch_reserved_mib, 2),
            "vmm_physical_pool_mib": round(self._vmm_physical_pool_mib(), 2),
            "vmm_stats": vmm_st,
            "active_contexts": len(self._ctx_store),
            "kv_free_bytes": self.available_kv_bytes(),
            "swapped_kv_contexts": 0,
            "kv_host_swap_outs": int(vmm_st.get("kv_swap_outs_total", 0)),
            "kv_host_swap_ins": int(vmm_st.get("kv_swap_ins_total", 0)),
            "kv_host_pinned_bytes": int(vmm_st.get("kv_swapped_bytes_current", 0)),
            "memory_plan": dict(self._memory_plan),
        }

    def close(self) -> None:
        import torch

        self._ctx_store.clear()
        if self.vlm_pager is not None:
            self.vlm_pager.close()
            self.vlm_pager = None
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
