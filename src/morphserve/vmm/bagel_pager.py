"""BAGEL-7B-MoT layer-wise TriPool-VMM pager: zero-copy Expert VA rebinding, Step-1 async streaming & VAE scoping."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import gc
import math
from typing import Any

from .tripool import NumaMemfdMirror, TriPoolVMM

ALIGN_BYTES = 512
MIB = 1024**2


@dataclass
class ParamSlice:
    name: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype_str: str


@dataclass
class LayerPackPlan:
    layer_idx: int
    used_bytes: int
    params: list[ParamSlice] = field(default_factory=list)


def align_up(n: int, align: int = ALIGN_BYTES) -> int:
    return (n + align - 1) & ~(align - 1)


def is_generation_param_name(name: str) -> bool:
    return (
        ("moe_gen" in name)
        or (".gen_exp." in name)
        or name.endswith(".gen_weight")
        or (name == "gen_weight")
    )


def build_layer_pack_plans(
    layers_module: Any,
    role: str = "generation",
    super_handle_mib: int = 64,
) -> tuple[list[LayerPackPlan], int]:
    """Inspect decoder layers and build 512B-aligned contiguous packing plans for the selected expert."""
    if role not in ("generation", "understanding"):
        raise ValueError(f"unsupported role: {role}")

    plans: list[LayerPackPlan] = []
    max_used = 0
    for layer_idx, layer in enumerate(layers_module):
        offset = 0
        entries: list[ParamSlice] = []
        for pname, param in layer.named_parameters():
            is_gen = is_generation_param_name(pname)
            if (role == "generation" and not is_gen) or (role == "understanding" and is_gen):
                continue
            nbytes = int(param.numel() * param.element_size())
            entries.append(
                ParamSlice(
                    name=pname,
                    offset=offset,
                    nbytes=nbytes,
                    shape=tuple(int(d) for d in param.shape),
                    dtype_str=str(param.dtype),
                )
            )
            offset = align_up(offset + nbytes, ALIGN_BYTES)
        max_used = max(max_used, offset)
        plans.append(LayerPackPlan(layer_idx=layer_idx, used_bytes=offset, params=entries))

    super_bytes = super_handle_mib * MIB
    stride_bytes = int(math.ceil(max_used / super_bytes)) * super_bytes
    return plans, stride_bytes


class BagelExpertPager:
    """Attaches TriPool-VMM to a live BAGEL-7B-MoT BagelPipeline with zero-copy Expert VA & scoped VAE MemPool.

    Two-phase initialization guarantees zero HBM overshoot on 40 GiB A100:
      1. `offload_to_host_mirror()`: packs expert weights into NUMA-bound memfd mirror and releases old GPU storage
         (drops GPU HBM from 28.15 GiB -> ~15.9 GiB BEFORE allocating TriPool-VMM's physical page pool!).
      2. `bind_vmm(vmm, initial_alpha=...)`: pins mirror, reserves Expert VA, rebinds `nn.Parameter.data` permanently
         to fixed Expert VA slices, loads initial `alpha` prefix, and installs layer/VAE hooks.
    """

    def __init__(
        self,
        pipeline: Any,
        role: str = "generation",
        numa_node: int = 0,
        super_handle_mib: int = 64,
        wrap_vae: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.vmm: TriPoolVMM | None = None
        self.role = role
        self.numa_node = numa_node
        self.super_handle_mib = super_handle_mib
        self.wrap_vae_enabled = wrap_vae

        self.layers = pipeline.bagel.language_model.model.layers
        self.num_layers = len(self.layers)
        self.plans, self.layer_stride_bytes = build_layer_pack_plans(
            self.layers, role=role, super_handle_mib=super_handle_mib
        )
        self.layer_stride_mib = self.layer_stride_bytes // MIB
        self.total_mirror_bytes = self.num_layers * self.layer_stride_bytes

        self.mirror: NumaMemfdMirror | None = None
        self.expert_id: int = -1
        self._va_views: list[Any] = []
        self._hook_handles: list[Any] = []
        self._wrapped_layer_methods: list[tuple[Any, str, Any]] = []
        self.layer_hook_calls: int = 0
        self.hook_methods_triggered: set[str] = set()
        self._active_guard = False
        self._orig_vae_decode = None
        self._orig_vae_encode = None
        self._packed_params_count = 0
        self._packed_data_bytes = 0

    def offload_to_host_mirror(self) -> dict[str, Any]:
        """Step 1: Copy all 28 expert layers into NUMA memfd mirror and release PyTorch GPU weight memory."""
        import torch

        self.mirror = NumaMemfdMirror(
            self.total_mirror_bytes,
            numa_node=self.numa_node,
            name=f"morphserve_bagel_{self.role}",
        )
        host_u8 = self.mirror.as_uint8_tensor()
        packed_params_count = 0
        packed_data_bytes = 0

        for plan in self.plans:
            layer = self.layers[plan.layer_idx]
            param_map = dict(layer.named_parameters())
            base_off = plan.layer_idx * self.layer_stride_bytes
            for ps in plan.params:
                p = param_map[ps.name]
                raw_cpu = p.detach().contiguous().cpu().view(torch.uint8).reshape(-1)
                host_u8[base_off + ps.offset : base_off + ps.offset + ps.nbytes].copy_(raw_cpu)
                # Drop GPU storage immediately so HBM shrinks before TriPoolVMM preallocates physical pages
                p.data = torch.empty(0, dtype=p.dtype, device=p.device)
                packed_params_count += 1
                packed_data_bytes += ps.nbytes

        self._packed_params_count = packed_params_count
        self._packed_data_bytes = packed_data_bytes
        del host_u8
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "role": self.role,
            "num_layers": self.num_layers,
            "layer_stride_mib": self.layer_stride_mib,
            "packed_params": packed_params_count,
            "packed_data_mib": round(packed_data_bytes / MIB, 2),
            "mirror_total_mib": round(self.total_mirror_bytes / MIB, 2),
            "packing_efficiency_pct": round(100.0 * packed_data_bytes / self.total_mirror_bytes, 2),
            "numa_bound": self.mirror.numa_bound,
        }

    def bind_vmm(self, vmm: TriPoolVMM, initial_alpha: float = 1.0) -> dict[str, Any]:
        """Step 2: Pin mirror, reserve Expert VA, permanently bind Parameter.data views, load initial alpha."""
        if self.mirror is None:
            raise RuntimeError("call offload_to_host_mirror() before bind_vmm()")
        if not 0.0 <= initial_alpha <= 1.0:
            raise ValueError("initial_alpha must be in [0.0, 1.0]")

        self.vmm = vmm
        self.expert_id = vmm.register_expert(
            num_layers=self.num_layers,
            layer_stride_mib=self.layer_stride_mib,
            mirror=self.mirror,
        )

        self._va_views = []
        for plan in self.plans:
            vmm.set_expert_layer_used_bytes(self.expert_id, plan.layer_idx, plan.used_bytes)
            layer = self.layers[plan.layer_idx]
            param_map = dict(layer.named_parameters())
            layer_u8_va = vmm.expert_layer_uint8_view(self.expert_id, plan.layer_idx)
            self._va_views.append(layer_u8_va)
            for ps in plan.params:
                p = param_map[ps.name]
                new_view = layer_u8_va[ps.offset : ps.offset + ps.nbytes].view(p.dtype).view(ps.shape)
                p.data = new_view

        initial_layers = int(round(initial_alpha * self.num_layers))
        if initial_layers > 0:
            vmm.load_expert_layers_sync(self.expert_id, 0, initial_layers)

        self._install_layer_hooks()
        if self.wrap_vae_enabled:
            self._install_vae_hooks()

        return {
            "expert_id": self.expert_id,
            "layer_stride_mib": self.layer_stride_mib,
            "initial_resident_layers": vmm.expert_resident_count(self.expert_id),
        }

    def _install_layer_hooks(self) -> None:
        import torch

        assert self.vmm is not None
        vmm = self.vmm

        def _wait_if_active(idx: int, method_tag: str) -> None:
            if not self._active_guard:
                return
            self.layer_hook_calls += 1
            self.hook_methods_triggered.add(method_tag)
            stream_ptr = torch.cuda.current_stream(vmm.device).cuda_stream
            vmm.wait_expert_layer(self.expert_id, idx, stream_ptr)

        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]

            def _make_pre_hook(idx: int):
                def _pre_hook(_module, _inputs):
                    _wait_if_active(idx, "forward")

                return _pre_hook

            h = layer.register_forward_pre_hook(_make_pre_hook(layer_idx))
            self._hook_handles.append(h)

            # BAGEL's Qwen2MoTModel calls layer.forward_inference(...) directly, bypassing nn.Module.__call__ hooks!
            for meth_name in ("forward_inference", "forward_train"):
                orig_fn = getattr(layer, meth_name, None)
                if callable(orig_fn):
                    def _make_method_wrapper(idx: int, tag: str, fn: Any):
                        def _wrapped(*args, **kwargs):
                            _wait_if_active(idx, tag)
                            return fn(*args, **kwargs)

                        return _wrapped

                    setattr(layer, meth_name, _make_method_wrapper(layer_idx, meth_name, orig_fn))
                    self._wrapped_layer_methods.append((layer, meth_name, orig_fn))

    def _install_vae_hooks(self) -> None:
        assert self.vmm is not None
        vmm = self.vmm
        vae = self.pipeline.vae
        self._orig_vae_decode = vae.decode
        self._orig_vae_encode = vae.encode

        def _wrapped_decode(*args, **kwargs):
            was_active = self._active_guard
            self._active_guard = False
            vmm.set_expert_pinned(self.expert_id, False)
            try:
                return vmm.run_in_transient_scope(self._orig_vae_decode, *args, **kwargs)
            finally:
                self._active_guard = was_active
                if was_active:
                    vmm.set_expert_pinned(self.expert_id, True)

        def _wrapped_encode(*args, **kwargs):
            was_active = self._active_guard
            self._active_guard = False
            vmm.set_expert_pinned(self.expert_id, False)
            try:
                return vmm.run_in_transient_scope(self._orig_vae_encode, *args, **kwargs)
            finally:
                self._active_guard = was_active
                if was_active:
                    vmm.set_expert_pinned(self.expert_id, True)

        vae.decode = _wrapped_decode
        vae.encode = _wrapped_encode

    def set_residency_alpha(self, alpha: float) -> int:
        import torch

        assert self.vmm is not None
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0.0, 1.0]")
        target = int(round(alpha * self.num_layers))
        stream_ptr = torch.cuda.current_stream(self.vmm.device).cuda_stream
        self.vmm.set_expert_pinned(self.expert_id, False)
        if target < self.num_layers:
            self.vmm.evict_expert_range(self.expert_id, target, self.num_layers, stream_ptr)
        if target > 0:
            self.vmm.load_expert_layers_sync(self.expert_id, 0, target)
        return self.vmm.expert_resident_count(self.expert_id)

    @contextmanager
    def generation_mode(self, prefetch_all: bool = True):
        assert self.vmm is not None
        self.vmm.set_expert_pinned(self.expert_id, True)
        self._active_guard = True
        if prefetch_all:
            self.vmm.prefetch_expert_async(self.expert_id, 0, self.num_layers)
        try:
            yield
        finally:
            self._active_guard = False
            self.vmm.set_expert_pinned(self.expert_id, False)

    def unbind_vmm_keep_mirror(self) -> None:
        """Detach hooks and VA views while keeping host mirror ready for rebinding to a new VMM instance."""
        import torch

        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        for layer, meth_name, orig_fn in self._wrapped_layer_methods:
            setattr(layer, meth_name, orig_fn)
        self._wrapped_layer_methods.clear()
        if self._orig_vae_decode is not None:
            self.pipeline.vae.decode = self._orig_vae_decode
            self._orig_vae_decode = None
        if self._orig_vae_encode is not None:
            self.pipeline.vae.encode = self._orig_vae_encode
            self._orig_vae_encode = None
        for plan in self.plans:
            layer = self.layers[plan.layer_idx]
            param_map = dict(layer.named_parameters())
            for ps in plan.params:
                p = param_map[ps.name]
                p.data = torch.empty(0, dtype=p.dtype, device=p.device)
        self._va_views.clear()
        self.expert_id = -1
        self.vmm = None

    def close(self) -> None:
        self.unbind_vmm_keep_mirror()
        if self.mirror is not None:
            self.mirror.close()
            self.mirror = None
