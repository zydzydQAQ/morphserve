"""Vendored `InternVL-U` runtime compatibility shims and checkpoint loader."""

from __future__ import annotations

import gc
import inspect
from pathlib import Path
import sys
import types
from typing import Any


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

            def forward(self, x: Any) -> Any:
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
    """Load `InternVLUPipeline` avoiding redundant GPU weight spikes when paging or fissioning."""
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

            def _make_patched_init(fn: Any):
                def _patched_init(self: Any, *a: Any, **kw: Any) -> None:
                    fn(self, *a, **kw)
                    if not hasattr(self, "all_tied_weights_keys") or "all_tied_weights_keys" not in self.__dict__:
                        self.all_tied_weights_keys = {}

                return _patched_init

            cls_m.__init__ = _make_patched_init(orig_init)
            cls_m._ms_post_init_patched = True

    if not getattr(InternVLUProcessor, "_ms_proc_patched", False):
        from internvlu.vlm.constants import IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN

        def _patched_proc_init(
            self: Any,
            image_processor: Any = None,
            image_gen_processor: Any = None,
            tokenizer: Any = None,
            video_processor: Any = None,
            image_seq_length: int = 256,
            chat_template: str | None = None,
            blank_messages: dict[str, Any] | None = None,
            **kwargs: Any,
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

        def _patched_merge_kwargs(
            self: Any,
            processing_kwargs_cls: Any = None,
            tokenizer_init_kwargs: Any = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
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
        # Cast floating-point weights to BF16 while preserving FP32 rotary frequency buffers.
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
            def _safe_lm_fwd(*a: Any, **kw: Any) -> Any:
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
        for b in pipe.generation_decoder.buffers():
            b.data = b.data.to(dev)
    else:
        pipe.generation_decoder.to(dev)

    # Materialize non-persistent plain tensor attributes (e.g. UnifiedMSRoPE.all_inv_freqs).
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


__all__ = ["load_internvlu_pipeline"]
