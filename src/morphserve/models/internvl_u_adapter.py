"""MorphServe CPTG + TriPool-VMM adapter for `InternVL-U` (`bridge_mllm_dit` architecture)."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from pathlib import Path
import sys
import types
from typing import Any

from morphserve.router import SharedNumaMemfdMirror
from morphserve.vmm import TriPoolVMM

MIB = 1024**2
ALIGN_BYTES = 256


def _align_up(x: int, align: int = ALIGN_BYTES) -> int:
    return ((x + align - 1) // align) * align


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
    import transformers.processing_utils as pu
    import transformers.utils as tu

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
    """Load `InternVLUPipeline` from verified local safetensors in `bfloat16` on `cuda:{device_index}`."""
    import inspect
    import torch

    _install_runtime_compat_shims(vendor_dir=vendor_dir)
    from diffusers.models import AutoencoderKLQwenImage
    from diffusers.schedulers import DPMSolverMultistepScheduler
    from internvlu import InternVLUPipeline
    from internvlu.diffusion import InternVLUGenerationDecoder
    from internvlu.processing_internvlu import InternVLUProcessor
    from internvlu.vlm import InternVLUChatModel

    torch.cuda.set_device(device_index)
    dev = torch.device(f"cuda:{device_index}")

    # Load each verified sub-directory directly to bypass diffusers>=0.40 custom module_path restrictions
    vlm = InternVLUChatModel.from_pretrained(
        str(model_dir / "vlm"),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    generation_decoder = InternVLUGenerationDecoder.from_pretrained(
        str(model_dir / "generation_decoder"),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    vae = AutoencoderKLQwenImage.from_pretrained(
        str(model_dir / "vae"),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        str(model_dir / "scheduler"),
        local_files_only=True,
    )
    processor = InternVLUProcessor.from_pretrained(
        str(model_dir / "processor"),
        local_files_only=True,
    )

    # Guard Qwen3ForCausalLM.forward against custom `padding_type` kwarg in transformers>=5
    lm = getattr(vlm, "language_model", None)
    if lm is not None and hasattr(lm, "forward"):
        orig_lm_fwd = lm.forward
        sig_params = inspect.signature(orig_lm_fwd).parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
        if "padding_type" not in sig_params and not has_var_kw:
            def _safe_lm_fwd(*a, **kw):
                kw.pop("padding_type", None)
                return orig_lm_fwd(*a, **kw)

            lm.forward = _safe_lm_fwd

    pipe = InternVLUPipeline(
        vlm=vlm,
        generation_decoder=generation_decoder,
        vae=vae,
        scheduler=scheduler,
        processor=processor,
    )
    pipe = pipe.to(dev)
    pipe.vlm.eval()
    pipe.generation_decoder.eval()
    pipe.vae.eval()
    return pipe


class InternVLUDecoderPager:
    """TriPool-VMM Drop-Clean 64 MiB super-handle pager for `InternVL-U` GenerationDecoder + Scoped VAE."""

    def __init__(
        self,
        pipeline: Any,
        numa_node: int = 0,
        super_handle_mib: int = 64,
        wrap_vae: bool = True,
    ) -> None:
        import torch

        self.pipeline = pipeline
        self.decoder = pipeline.generation_decoder
        self.vae = pipeline.vae
        self.numa_node = int(numa_node)
        self.super_handle_bytes = int(super_handle_mib) * MIB
        self.expert_id = 0
        self.wrap_vae = bool(wrap_vae)

        # Partition decoder parameters across 20 logical layer buckets matching transformer blocks
        dec_inner = getattr(self.decoder, "decoder", self.decoder)
        blocks = getattr(dec_inner, "transformer_blocks", None) or getattr(dec_inner, "blocks", None)
        self.num_layers = len(blocks) if blocks is not None and len(blocks) > 0 else 20

        self._layer_params: list[list[tuple[str, Any, tuple[int, ...], Any, int, int]]] = [
            [] for _ in range(self.num_layers)
        ]
        raw_params: list[tuple[int, str, Any]] = []
        for pname, p in self.decoder.named_parameters():
            lid = 0
            for marker in ("transformer_blocks.", "blocks.", "layers."):
                if marker in pname:
                    tail = pname.split(marker, 1)[1]
                    tok = tail.split(".", 1)[0]
                    if tok.isdigit():
                        lid = min(self.num_layers - 1, int(tok))
                    break
            raw_params.append((lid, pname, p))

        layer_offsets = [0 for _ in range(self.num_layers)]
        for lid, pname, p in raw_params:
            nbytes = int(p.numel() * p.element_size())
            off = _align_up(layer_offsets[lid], 512)
            self._layer_params[lid].append((pname, p, tuple(p.shape), p.dtype, off, nbytes))
            layer_offsets[lid] = off + nbytes

        self.layer_used_bytes = [max(x, 4096) for x in layer_offsets]
        max_used = max(self.layer_used_bytes)
        self.layer_stride_bytes = _align_up(max_used, self.super_handle_bytes)
        self.layer_stride_mib = self.layer_stride_bytes // MIB
        self.layer_mirror_offsets = [lid * self.layer_stride_bytes for lid in range(self.num_layers)]
        self.total_mirror_bytes = self.num_layers * self.layer_stride_bytes

        self._orig_vae_decode = None
        self._orig_vae_encode = None
        self._is_pinned = False
        self._va_views: list[Any] = []
        self.mirror: SharedNumaMemfdMirror | None = None
        self._owns_mirror = False
        self.vmm: TriPoolVMM | None = None
        self._device = next(self.decoder.parameters()).device

    def offload_to_shared_mirror(self, shared_mirror: SharedNumaMemfdMirror, populate: bool = True) -> None:
        import torch

        self.mirror = shared_mirror
        self._owns_mirror = False
        base_addr = shared_mirror.ptr
        for lid in range(self.num_layers):
            l_base = base_addr + self.layer_mirror_offsets[lid]
            for _, p, _, dtype, off, nbytes in self._layer_params[lid]:
                if populate and nbytes > 0:
                    cpu_t = p.detach().contiguous().cpu()
                    ctypes.memmove(l_base + off, cpu_t.data_ptr(), nbytes)
                p.data = torch.empty((0,), dtype=dtype, device=self._device)
        torch.cuda.empty_cache()

    def offload_to_host_mirror(self) -> SharedNumaMemfdMirror:
        m = SharedNumaMemfdMirror.create(
            nbytes=self.total_mirror_bytes,
            numa_node=self.numa_node,
            name="ms_internvlu_mirror",
        )
        self.offload_to_shared_mirror(m, populate=True)
        self._owns_mirror = True
        return m

    def bind_vmm(self, vmm: TriPoolVMM, initial_alpha: float = 1.0) -> None:
        assert self.mirror is not None
        self.vmm = vmm
        self.expert_id = vmm.register_expert(
            num_layers=self.num_layers,
            layer_stride_mib=self.layer_stride_mib,
            mirror=self.mirror,
        )
        self._va_views = []
        for lid in range(self.num_layers):
            vmm.set_expert_layer_used_bytes(self.expert_id, lid, self.layer_used_bytes[lid])
            u8_va = vmm.expert_layer_uint8_view(self.expert_id, lid)
            self._va_views.append(u8_va)
            for _, p, shape, dtype, off, nbytes in self._layer_params[lid]:
                p.data = u8_va[off : off + nbytes].view(dtype).view(shape)

        if self.wrap_vae:
            orig_dec = self.vae.decode
            orig_enc = self.vae.encode
            self._orig_vae_decode = orig_dec
            self._orig_vae_encode = orig_enc

            def _wrapped_decode(*a, **kw):
                was_pinned = self._is_pinned
                prev_res = vmm.expert_resident_count(self.expert_id)
                self._is_pinned = False
                vmm.set_expert_pinned(self.expert_id, False)
                try:
                    return vmm.run_in_transient_scope(orig_dec, *a, **kw)
                finally:
                    if vmm.expert_resident_count(self.expert_id) < prev_res:
                        vmm.load_expert_layers_sync(self.expert_id, 0, prev_res)
                    self._is_pinned = was_pinned
                    if was_pinned:
                        vmm.set_expert_pinned(self.expert_id, True)

            def _wrapped_encode(*a, **kw):
                was_pinned = self._is_pinned
                prev_res = vmm.expert_resident_count(self.expert_id)
                self._is_pinned = False
                vmm.set_expert_pinned(self.expert_id, False)
                try:
                    return vmm.run_in_transient_scope(orig_enc, *a, **kw)
                finally:
                    if vmm.expert_resident_count(self.expert_id) < prev_res:
                        vmm.load_expert_layers_sync(self.expert_id, 0, prev_res)
                    self._is_pinned = was_pinned
                    if was_pinned:
                        vmm.set_expert_pinned(self.expert_id, True)

            self.vae.decode = _wrapped_decode
            self.vae.encode = _wrapped_encode

        self.set_residency_alpha(initial_alpha)

    def set_residency_alpha(self, alpha: float) -> int:
        import torch

        assert self.vmm is not None
        target = max(0, min(self.num_layers, int(round(float(alpha) * self.num_layers))))
        s_ptr = int(torch.cuda.current_stream(self._device).cuda_stream)
        self._is_pinned = False
        self.vmm.set_expert_pinned(self.expert_id, False)
        if target < self.num_layers:
            self.vmm.evict_expert_range(self.expert_id, target, self.num_layers, s_ptr)
        if target > 0:
            self.vmm.load_expert_layers_sync(self.expert_id, 0, target)
        torch.cuda.synchronize(self._device)
        return self.vmm.expert_resident_count(self.expert_id)

    @contextmanager
    def generation_mode(self):
        import torch

        assert self.vmm is not None
        self._is_pinned = True
        self.vmm.set_expert_pinned(self.expert_id, True)
        if self.vmm.expert_resident_count(self.expert_id) < self.num_layers:
            self.vmm.load_expert_layers_sync(self.expert_id, 0, self.num_layers)
        torch.cuda.synchronize(self._device)
        try:
            yield
        finally:
            self._is_pinned = False
            self.vmm.set_expert_pinned(self.expert_id, False)

    def close(self) -> None:
        import torch

        if self._orig_vae_decode is not None:
            self.vae.decode = self._orig_vae_decode
            self._orig_vae_decode = None
        if self._orig_vae_encode is not None:
            self.vae.encode = self._orig_vae_encode
            self._orig_vae_encode = None
        for lid in range(self.num_layers):
            for _, p, _, dtype, _, _ in self._layer_params[lid]:
                p.data = torch.empty((0,), dtype=dtype, device=self._device)
        self._va_views.clear()
        self.vmm = None
        if self.mirror is not None and self._owns_mirror:
            self.mirror.close()
        self.mirror = None


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

    entries: list[dict[str, Any]] = []
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
    meta = {"total_bytes": int(flat.numel()), "tree": tree, "entries_count": len(entries)}
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
