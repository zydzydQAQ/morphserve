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

    if not hasattr(iu, "concatenate_list"):
        iu.concatenate_list = lambda items: [x for sub in items for x in (sub if isinstance(sub, list) else [sub])]
    if not hasattr(iu, "make_list_of_images"):
        iu.make_list_of_images = iu.make_flat_list_of_images


def load_internvlu_pipeline(
    model_dir: Path,
    vendor_dir: Path | None = None,
    device_index: int = 0,
) -> Any:
    """Load `InternVLUPipeline` from verified local safetensors in `bfloat16` on `cuda:{device_index}`."""
    import torch

    _install_runtime_compat_shims(vendor_dir=vendor_dir)
    from internvlu import InternVLUPipeline

    torch.cuda.set_device(device_index)
    dev = torch.device(f"cuda:{device_index}")
    pipe = InternVLUPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
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
            off = _align_up(layer_offsets[lid])
            self._layer_params[lid].append((pname, p, tuple(p.shape), p.dtype, off, nbytes))
            layer_offsets[lid] = off + nbytes

        self.layer_bytes = [_align_up(max(x, 4096), self.super_handle_bytes) for x in layer_offsets]
        self.layer_mirror_offsets: list[int] = []
        cur = 0
        for lb in self.layer_bytes:
            self.layer_mirror_offsets.append(cur)
            cur += lb
        self.total_mirror_bytes = cur

        # Optional VAE CPU mirror for Scoped Bulk-Unmap Transient Pool
        self._vae_entries: list[tuple[Any, Any, int, int]] = []
        vae_cur = 0
        if self.wrap_vae:
            for _, p in self.vae.named_parameters():
                nb = int(p.numel() * p.element_size())
                off = _align_up(vae_cur)
                self._vae_entries.append((p, p.detach().contiguous().cpu(), off, nb))
                vae_cur = off + nb
            for _, b in self.vae.named_buffers():
                nb = int(b.numel() * b.element_size())
                off = _align_up(vae_cur)
                self._vae_entries.append((b, b.detach().contiguous().cpu(), off, nb))
                vae_cur = off + nb
        self.vae_total_bytes = _align_up(max(vae_cur, 4096), 2 * MIB) if self.wrap_vae else 0

        self.mirror: SharedNumaMemfdMirror | None = None
        self._owns_mirror = False
        self.vmm: TriPoolVMM | None = None
        self._device = next(self.decoder.parameters()).device

    def offload_to_shared_mirror(self, shared_mirror: SharedNumaMemfdMirror, populate: bool = True) -> None:
        import torch

        self.mirror = shared_mirror
        self._owns_mirror = False
        base_addr = shared_mirror.host_ptr
        for lid in range(self.num_layers):
            l_base = base_addr + self.layer_mirror_offsets[lid]
            for _, p, _, dtype, off, nbytes in self._layer_params[lid]:
                if populate and nbytes > 0:
                    cpu_t = p.detach().contiguous().cpu()
                    ctypes.memmove(l_base + off, cpu_t.data_ptr(), nbytes)
                p.data = torch.empty((0,), dtype=dtype, device=self._device)
        if self.wrap_vae:
            for tensor_ref, _, _, _ in self._vae_entries:
                tensor_ref.data = torch.empty((0,), dtype=tensor_ref.dtype, device=self._device)
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
        self.mirror.pin_cuda(vmm)
        vmm.register_expert(self.expert_id, self.layer_bytes)
        self.set_residency_alpha(initial_alpha)

    def _rebind_layer_views(self, lid: int) -> None:
        import torch

        assert self.vmm is not None
        dev_ptr = self.vmm.expert_layer_ptr(self.expert_id, lid)
        total_b = self.layer_bytes[lid]
        raw_u8 = self.vmm.wrap_device_ptr_as_tensor(dev_ptr, (total_b,), torch.uint8)
        for _, p, shape, dtype, off, nbytes in self._layer_params[lid]:
            p.data = raw_u8[off : off + nbytes].view(dtype).reshape(shape)

    def set_residency_alpha(self, alpha: float) -> int:
        import torch

        assert self.vmm is not None and self.mirror is not None
        target = max(0, min(self.num_layers, int(round(float(alpha) * self.num_layers))))
        s_ptr = int(torch.cuda.current_stream(self._device).cuda_stream)
        for lid in range(self.num_layers):
            res = self.vmm.expert_is_resident(self.expert_id, lid)
            if lid < target and not res:
                h_ptr = self.mirror.host_ptr + self.layer_mirror_offsets[lid]
                self.vmm.expert_map_and_prefetch(self.expert_id, lid, h_ptr, self.layer_bytes[lid], stream_ptr=s_ptr)
                self._rebind_layer_views(lid)
            elif lid >= target and res:
                for _, p, _, dtype, _, _ in self._layer_params[lid]:
                    p.data = torch.empty((0,), dtype=dtype, device=self._device)
                self.vmm.expert_drop_clean(self.expert_id, lid, stream_ptr=s_ptr)
        torch.cuda.synchronize(self._device)
        return target

    @contextmanager
    def generation_mode(self):
        import torch

        assert self.vmm is not None and self.mirror is not None
        s_ptr = int(torch.cuda.current_stream(self._device).cuda_stream)
        for lid in range(self.num_layers):
            if not self.vmm.expert_is_resident(self.expert_id, lid):
                h_ptr = self.mirror.host_ptr + self.layer_mirror_offsets[lid]
                self.vmm.expert_map_and_prefetch(self.expert_id, lid, h_ptr, self.layer_bytes[lid], stream_ptr=s_ptr)
                self._rebind_layer_views(lid)
        vae_mounted = False
        if self.wrap_vae and self.vae_total_bytes > 0:
            v_ptr = self.vmm.transient_alloc(self.vae_total_bytes)
            vae_u8 = self.vmm.wrap_device_ptr_as_tensor(v_ptr, (self.vae_total_bytes,), torch.uint8)
            for tensor_ref, cpu_src, off, nb in self._vae_entries:
                dst_view = vae_u8[off : off + nb].view(tensor_ref.dtype).reshape(cpu_src.shape)
                dst_view.copy_(cpu_src, non_blocking=True)
                tensor_ref.data = dst_view
            vae_mounted = True
        torch.cuda.synchronize(self._device)
        try:
            yield
        finally:
            if vae_mounted:
                torch.cuda.synchronize(self._device)
                for tensor_ref, _, _, _ in self._vae_entries:
                    tensor_ref.data = torch.empty((0,), dtype=tensor_ref.dtype, device=self._device)
                self.vmm.transient_reset(stream_ptr=s_ptr)

    def close(self) -> None:
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
