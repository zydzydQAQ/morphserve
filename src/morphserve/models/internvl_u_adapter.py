"""Deep `InternVLUCPTGAdapter` (`bridge_mllm_dit`) built atop `LayerExpertPager` + `c62ec05` shims."""

from __future__ import annotations

import gc
import inspect
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

from .base import CPTGModelAdapter, PackedTransferState

MIB = 1024**2


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


def load_internvlu_pipeline(
    model_dir: Path,
    vendor_dir: Path | None = None,
    device_index: int = 0,
) -> Any:
    """Load `InternVLUPipeline` with `low_cpu_mem_usage=False` & `UnifiedMSRoPE` meta-tensor guard (`c62ec05`)."""
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

    vlm = InternVLUChatModel.from_pretrained(
        str(model_dir / "vlm"),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        local_files_only=True,
    )
    generation_decoder = InternVLUGenerationDecoder.from_pretrained(
        str(model_dir / "generation_decoder"),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        local_files_only=True,
    )
    vae = AutoencoderKLQwenImage.from_pretrained(
        str(model_dir / "vae"),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        local_files_only=True,
    )
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
    pipe = pipe.to(dev)
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
        if getattr(policy, "gpu_memory_fraction", None) is not None:
            torch.cuda.set_per_process_memory_fraction(float(policy.gpu_memory_fraction), self.device)
        vendor_p = Path(policy.vendor_dir) if getattr(policy, "vendor_dir", None) else None

        self.pipe = load_internvlu_pipeline(self.model_dir, vendor_dir=vendor_p, device_index=self.rank)
        self.pager: LayerExpertPager | None = None
        self.naive_offloader: NaiveHostOffloader | None = None
        self.vmm: TriPoolVMM | None = None
        self.pacer: Any = None
        self._ctx_store: dict[str, dict[str, Any]] = {}
        self._max_reserved_mib = 0.0

        dec_spec = ExpertPartitionSpec.for_internvlu(
            self.pipe,
            super_handle_mib=policy.super_handle_mib,
            wrap_vae=policy.use_scoped_vae,
        )
        self.num_expert_layers = dec_spec.num_layers

        if policy.routing_mode == "static_nccl_fission":
            if self.role == "und":
                drop_partition_weights(dec_spec, device=self.device)
                for p in self.pipe.vae.parameters():
                    p.data = torch.empty(0, dtype=p.dtype, device=self.device)
                gc.collect()
                torch.cuda.empty_cache()
            elif self.role == "gen":
                for p in self.pipe.vlm.parameters():
                    p.data = torch.empty(0, dtype=p.dtype, device=self.device)
                gc.collect()
                torch.cuda.empty_cache()
        elif policy.use_naive_offload:
            self.naive_offloader = NaiveHostOffloader(dec_spec, device=self.device)
            self.naive_offloader.offload_to_cpu()
        elif policy.use_expert_pager:
            self.pager = LayerExpertPager(dec_spec, numa_node=self.numa_node)

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

    def bind_runtime_resources(self, pacer: Any | None = None) -> dict[str, Any]:
        import torch

        self.pacer = pacer
        if self.policy.use_tripool_vmm:
            pool_mib = int(self.policy.super_pool_mib)
            self.vmm = TriPoolVMM(
                device=self.rank,
                super_pool_mib=pool_mib,
                fine_kv_pool_mib=self.policy.fine_kv_pool_mib,
                super_handle_mib=self.policy.super_handle_mib,
                transient_va_gib=self.policy.transient_va_gib,
                kv_va_gib=self.policy.kv_va_gib,
                allow_overflow=self.policy.allow_vmm_overflow,
            )
            if self.pager is not None:
                self.pager.bind_vmm(self.vmm, initial_alpha=self.policy.initial_alphas.get(self.rank, 1.0))

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        self._max_reserved_mib = float(torch.cuda.memory_reserved(self.device) / MIB)
        return {
            "resident_alpha": self.resident_alpha(),
            "reserved_mib": round(self._max_reserved_mib, 2),
        }

    def warmup(self) -> None:
        return None

    def _update_hwm(self) -> None:
        import torch

        cur = float(torch.cuda.memory_reserved(self.device) / MIB)
        pk = float(torch.cuda.max_memory_reserved(self.device) / MIB)
        self._max_reserved_mib = max(self._max_reserved_mib, cur, pk)

    def _load_pil_if_any(self, cmd: dict[str, Any]) -> Any:
        from PIL import Image
        import os

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

    def prepare_turn(self, ctx_id: str, cmd: dict[str, Any]) -> int:
        import torch

        prompt = str(cmd["prompt"])
        prep_dit = bool(cmd.get("prepare_for_dit", False))
        pil_img = self._load_pil_if_any(cmd)
        if not prep_dit:
            self._ctx_store[ctx_id] = {"prompt": prompt, "pil_img": pil_img, "bridge_bytes": 4096}
            return 4096

        h = int(cmd.get("height", 512))
        w = int(cmd.get("width", 512))
        stride = self.pipe.processor.image_gen_processor.merge_size * self.pipe.processor.image_gen_processor.patch_size
        gh, gw = (h // stride) * stride, (w // stride) * stride
        with torch.inference_mode():
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
            meta, flat = pack_internvlu_bridge_state(diff_inp, device=self.device)
        torch.cuda.synchronize(self.device)
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
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.pipe(
                prompt=prompt,
                image=pil_img,
                generation_mode="text",
                max_new_tokens=int(max_tokens),
                do_sample=False,
            )
        torch.cuda.synchronize(self.device)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        tok_seq = out.generate_output[0]
        n_tok = max(1, int(len(tok_seq)))
        txt = self.pipe.tokenizer.decode(tok_seq, skip_special_tokens=True)
        per_tok = dt_ms / float(n_tok)
        self._update_hwm()
        return {
            "text": txt,
            "tokens": n_tok,
            "ttft_ms": round(per_tok, 2),
            "tpots_ms": [round(per_tok, 2) for _ in range(max(0, n_tok - 1))],
            "kv_bytes": int(entry.get("bridge_bytes", 4096)),
            "resident_alpha": self.resident_alpha(),
        }

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
        if "diff_inp" not in entry:
            self.prepare_turn(ctx_id, dict(cmd, prepare_for_dit=True))
            entry = self._ctx_store[ctx_id]

        diff_inp = entry["diff_inp"]
        entry.pop("packed_flat", None)
        gh, gw = int(entry["gh"]), int(entry["gw"])
        steps = int(cmd.get("dit_steps", 16))
        seed = int(cmd.get("seed", 42))
        save_tmp_path = cmd.get("save_tmp_image_path")

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
            with torch.inference_mode():
                if self.pager is not None:
                    with self.pager.generation_mode(prefetch_all=True):
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
                        ).images
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
                    ).images
        finally:
            if pacer_cm is not None:
                pacer_cm.__exit__(None, None, None)
            if self.naive_offloader is not None:
                self.naive_offloader.drop_from_gpu()

        torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if save_tmp_path:
            out_c = ((127.5 * imgs_t[0] + 128.0) / 255.0).clamp(0, 1)
            pil = Image.fromarray((out_c.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            Path(save_tmp_path).parent.mkdir(parents=True, exist_ok=True)
            pil.save(save_tmp_path, format="PNG")

        self._update_hwm()
        return {
            "status": "ok",
            "ctx_id": ctx_id,
            "elapsed_ms": round(elapsed_ms, 2),
            "kv_bytes": int(entry.get("bridge_bytes", 0)),
            "resident_alpha": self.resident_alpha(),
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

        self._ctx_store.clear()
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
