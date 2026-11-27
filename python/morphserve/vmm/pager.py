"""Unified TriPool-VMM LayerExpertPager supporting BAGEL-7B-MoT, InternVL-U, and CPU simulation."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import gc
import math
import threading
import time
from typing import Any, Callable, Iterator

from .tripool import NumaMemfdMirror

ALIGN_BYTES = 512
MIB = 1024**2
_VALID_VAE_ENCODE_RESTORE = ("none", "async", "sync")


@dataclass
class ParamSlice:
    name: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype_str: str
    param_ref: Any = field(default=None, repr=False)


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
    """Inspect decoder blocks and build 512B-aligned contiguous packing plans for the selected expert."""
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
                    param_ref=param,
                )
            )
            offset = align_up(offset + nbytes, ALIGN_BYTES)
        max_used = max(max_used, max(offset, 4096) if entries else 0)
        plans.append(LayerPackPlan(layer_idx=layer_idx, used_bytes=offset, params=entries))

    super_bytes = int(super_handle_mib) * MIB
    stride_bytes = max(super_bytes, int(math.ceil(max_used / super_bytes)) * super_bytes)
    return plans, stride_bytes


@dataclass
class ExpertPartitionSpec:
    """Declarative expert block & VAE partition specification driving `LayerExpertPager`."""

    name: str
    plans: list[LayerPackPlan]
    layer_stride_bytes: int
    hook_modules: list[Any]
    vae_module: Any | None = None
    extra_hook_methods: tuple[str, ...] = ()
    install_module_hooks: bool = True
    vae_encode_restore: str = "none"
    vae_output_cloner: Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        if self.vae_encode_restore not in _VALID_VAE_ENCODE_RESTORE:
            raise ValueError(
                f"vae_encode_restore must be one of {_VALID_VAE_ENCODE_RESTORE}, got {self.vae_encode_restore!r}"
            )

    @property
    def num_layers(self) -> int:
        return len(self.plans)

    @property
    def layer_stride_mib(self) -> int:
        return self.layer_stride_bytes // MIB

    @classmethod
    def for_bagel(
        cls,
        pipeline: Any,
        role: str = "generation",
        super_handle_mib: int = 64,
        wrap_vae: bool = True,
        install_module_hooks: bool = True,
        vae_encode_restore: str = "sync",
    ) -> "ExpertPartitionSpec":
        layers = list(pipeline.bagel.language_model.model.layers)
        plans, stride_bytes = build_layer_pack_plans(layers, role=role, super_handle_mib=super_handle_mib)
        return cls(
            name=f"bagel_{role}",
            plans=plans,
            layer_stride_bytes=stride_bytes,
            hook_modules=layers,
            vae_module=pipeline.vae if wrap_vae else None,
            extra_hook_methods=("forward_inference", "forward_train"),
            install_module_hooks=install_module_hooks,
            vae_encode_restore=vae_encode_restore,
        )

    @classmethod
    def for_internvlu(
        cls,
        pipeline: Any,
        super_handle_mib: int = 64,
        wrap_vae: bool = True,
        install_module_hooks: bool = True,
        vae_encode_restore: str = "none",
        vae_output_cloner: Callable[[Any], Any] | None = None,
    ) -> "ExpertPartitionSpec":
        decoder = pipeline.generation_decoder
        dec_inner = getattr(decoder, "decoder", decoder)
        blocks_seq = getattr(dec_inner, "transformer_blocks", None) or getattr(dec_inner, "blocks", None)
        hook_blocks = list(blocks_seq) if blocks_seq is not None and len(blocks_seq) > 0 else []
        num_layers = len(hook_blocks) if hook_blocks else 20

        buckets: list[list[tuple[str, Any]]] = [[] for _ in range(num_layers)]
        for pname, param in decoder.named_parameters():
            lid = 0
            for marker in ("transformer_blocks.", "blocks.", "layers."):
                if marker in pname:
                    tail = pname.split(marker, 1)[1]
                    tok = tail.split(".", 1)[0]
                    if tok.isdigit():
                        lid = min(num_layers - 1, int(tok))
                    break
            buckets[lid].append((pname, param))

        plans: list[LayerPackPlan] = []
        max_used = 4096
        for lid in range(num_layers):
            offset = 0
            entries: list[ParamSlice] = []
            for pname, param in buckets[lid]:
                nbytes = int(param.numel() * param.element_size())
                entries.append(
                    ParamSlice(
                        name=pname,
                        offset=offset,
                        nbytes=nbytes,
                        shape=tuple(int(d) for d in param.shape),
                        dtype_str=str(param.dtype),
                        param_ref=param,
                    )
                )
                offset = align_up(offset + nbytes, ALIGN_BYTES)
            used = max(offset, 4096)
            max_used = max(max_used, used)
            plans.append(LayerPackPlan(layer_idx=lid, used_bytes=used, params=entries))

        super_bytes = int(super_handle_mib) * MIB
        stride_bytes = max(super_bytes, int(math.ceil(max_used / super_bytes)) * super_bytes)
        return cls(
            name="internvlu_decoder",
            plans=plans,
            layer_stride_bytes=stride_bytes,
            hook_modules=hook_blocks,
            vae_module=pipeline.vae if wrap_vae else None,
            extra_hook_methods=(),
            install_module_hooks=install_module_hooks,
            vae_encode_restore=vae_encode_restore,
            vae_output_cloner=vae_output_cloner,
        )

    @classmethod
    def for_internvlu_vlm(
        cls,
        pipeline: Any,
        super_handle_mib: int = 16,
        install_module_hooks: bool = False,
    ) -> "ExpertPartitionSpec":
        """Build balanced FFD `LayerPackPlan` bins for `InternVL-U` VLM stage co-paging."""
        vlm = pipeline.vlm
        seen_ids: set[int] = set()
        unique_params: list[tuple[int, str, Any, int, int]] = []
        for idx, (pname, param) in enumerate(vlm.named_parameters()):
            if id(param) in seen_ids:
                continue
            seen_ids.add(id(param))
            nb = int(param.numel() * param.element_size())
            unique_params.append((idx, pname, param, nb, align_up(max(nb, 1), ALIGN_BYTES)))

        super_bytes = max(2 * MIB, int(super_handle_mib) * MIB)
        max_param_aligned = max((item[4] for item in unique_params), default=4096)
        target_min_stride = max(max_param_aligned, 256 * MIB)
        stride_bytes = max(super_bytes, int(math.ceil(target_min_stride / super_bytes)) * super_bytes)

        sorted_params = sorted(unique_params, key=lambda x: (-x[4], x[0]))
        bin_offsets: list[int] = []
        bin_entries: list[list[ParamSlice]] = []
        for _, pname, param, nb, aligned_nb in sorted_params:
            placed_bin = -1
            for b_idx, cur_off in enumerate(bin_offsets):
                if cur_off + aligned_nb <= stride_bytes:
                    placed_bin = b_idx
                    break
            if placed_bin < 0:
                placed_bin = len(bin_offsets)
                bin_offsets.append(0)
                bin_entries.append([])
            off = bin_offsets[placed_bin]
            bin_entries[placed_bin].append(
                ParamSlice(
                    name=pname,
                    offset=off,
                    nbytes=nb,
                    shape=tuple(int(d) for d in param.shape),
                    dtype_str=str(param.dtype),
                    param_ref=param,
                )
            )
            bin_offsets[placed_bin] = off + aligned_nb

        if not bin_offsets:
            plans = [LayerPackPlan(layer_idx=0, used_bytes=4096, params=[])]
        else:
            plans = [
                LayerPackPlan(layer_idx=b_idx, used_bytes=max(bin_offsets[b_idx], 4096), params=bin_entries[b_idx])
                for b_idx in range(len(bin_offsets))
            ]

        return cls(
            name="internvlu_vlm",
            plans=plans,
            layer_stride_bytes=stride_bytes,
            hook_modules=[],
            vae_module=None,
            extra_hook_methods=(),
            install_module_hooks=install_module_hooks,
            vae_encode_restore="none",
        )


def _current_cuda_stream_ptr(device: Any) -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.current_stream(device).cuda_stream)
    except Exception:
        pass
    return 0


def _sync_cuda_stream(device: Any) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.current_stream(device).synchronize()
    except Exception:
        pass


class LayerExpertPager:
    """Unified TriPool-VMM 64 MiB super-handle pager with async Step-1 H2D streaming & scoped VAE restoration."""

    def __init__(
        self,
        spec: ExpertPartitionSpec,
        numa_node: int = 0,
        should_wait_predicate: Callable[[], bool] | None = None,
        vae_output_cloner: Callable[[Any], Any] | None = None,
    ) -> None:
        self.spec = spec
        self.numa_node = int(numa_node)
        self.plans = spec.plans
        self.num_layers = spec.num_layers
        self.layer_stride_bytes = spec.layer_stride_bytes
        self.layer_stride_mib = spec.layer_stride_mib
        self.total_mirror_bytes = self.num_layers * self.layer_stride_bytes
        self._should_wait_predicate = should_wait_predicate
        self._vae_output_cloner = vae_output_cloner if vae_output_cloner is not None else spec.vae_output_cloner

        self.mirror: NumaMemfdMirror | None = None
        self._owns_mirror = False
        self._mirror_offset: int = 0
        self.vmm: Any = None
        self.expert_id: int = -1
        self._va_views: list[Any] = []
        self._hook_handles: list[Any] = []
        self._wrapped_layer_methods: list[tuple[Any, str, Any]] = []
        self._orig_vae_methods: list[tuple[Any, str, Any]] = []
        self._generation_active = False
        self._all_resident_synced = False
        self._has_pending_async_prefetch = False
        self._waited_layers: set[int] = set()
        self._active_pacer_release_cb: Callable[[bool], None] | None = None
        self.layer_hook_calls: int = 0
        self.hook_methods_triggered: set[str] = set()
        self._packed_params_count = 0
        self._packed_data_bytes = 0

    @property
    def is_generation_active(self) -> bool:
        return self._generation_active

    @property
    def all_resident_synced(self) -> bool:
        return self._all_resident_synced

    @property
    def owns_mirror(self) -> bool:
        return self._owns_mirror

    def attach_mirror(
        self,
        mirror: Any,
        *,
        owns_mirror: bool = False,
        mirror_offset: int = 0,
    ) -> None:
        """Bind a pre-populated host weight mirror cleanly without mutating private fields."""
        self.mirror = mirror
        self._owns_mirror = bool(owns_mirror)
        self._mirror_offset = max(0, int(mirror_offset))

    def release_active_pacer(self, *, sync_stream: bool = False) -> None:
        """Release any held `HalfNodeH2DPacer` slot for the active `generation_mode` scope."""
        cb = self._active_pacer_release_cb
        if cb is not None:
            cb(bool(sync_stream))

    def _populate_and_drop_gpu_weights(self, mirror: NumaMemfdMirror, populate: bool) -> dict[str, Any]:
        packed_params_count = 0
        packed_data_bytes = 0
        has_torch_tensors = False

        for plan in self.plans:
            base_off = self._mirror_offset + plan.layer_idx * self.layer_stride_bytes
            for ps in plan.params:
                p = ps.param_ref
                if p is not None and hasattr(p, "detach"):
                    has_torch_tensors = True
                    import torch

                    if populate and ps.nbytes > 0:
                        raw_cpu = p.detach().contiguous().cpu().view(torch.uint8).reshape(-1)
                        host_u8 = mirror.as_uint8_tensor()
                        host_u8[base_off + ps.offset : base_off + ps.offset + ps.nbytes].copy_(raw_cpu)
                        del host_u8
                    p.data = torch.empty(0, dtype=p.dtype, device=p.device)
                elif p is not None and hasattr(p, "raw_bytes"):
                    if populate and ps.nbytes > 0:
                        mirror.write_bytes(base_off + ps.offset, p.raw_bytes[: ps.nbytes])
                    p.bound_view = None
                packed_params_count += 1
                packed_data_bytes += ps.nbytes

        self._packed_params_count = packed_params_count
        self._packed_data_bytes = packed_data_bytes
        gc.collect()
        if has_torch_tensors:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            "name": self.spec.name,
            "num_layers": self.num_layers,
            "layer_stride_mib": self.layer_stride_mib,
            "packed_params": packed_params_count,
            "packed_data_mib": round(packed_data_bytes / MIB, 2),
            "mirror_total_mib": round(self.total_mirror_bytes / MIB, 2),
            "mirror_offset_mib": round(self._mirror_offset / MIB, 2),
            "packing_efficiency_pct": round(100.0 * packed_data_bytes / max(1, self.total_mirror_bytes), 2),
            "populated": populate,
            "numa_bound": bool(mirror.numa_bound),
        }

    def offload_to_host_mirror(self) -> dict[str, Any]:
        """Step 1 (Single-Process): Allocate NUMA memfd mirror, copy expert weights, and drop GPU tensors."""
        mirror = NumaMemfdMirror.create(
            nbytes=self.total_mirror_bytes,
            numa_node=self.numa_node,
            name=f"ms_{self.spec.name}",
        )
        self.attach_mirror(mirror, owns_mirror=True, mirror_offset=0)
        return self._populate_and_drop_gpu_weights(mirror, populate=True)

    def offload_to_shared_mirror(
        self,
        shared_mirror: NumaMemfdMirror,
        populate: bool = False,
        mirror_offset: int = 0,
    ) -> dict[str, Any]:
        """Step 1 (Multi-GPU Shared Socket): Attach to shared NUMA memfd mirror without duplicating host RAM."""
        off = max(0, int(mirror_offset))
        if off + self.total_mirror_bytes > int(shared_mirror.nbytes):
            raise ValueError(
                f"shared_mirror ({shared_mirror.nbytes} B) too small for {self.spec.name} at offset {off} "
                f"({self.total_mirror_bytes} B)"
            )
        self.attach_mirror(shared_mirror, owns_mirror=False, mirror_offset=off)
        return self._populate_and_drop_gpu_weights(shared_mirror, populate=populate)

    def bind_vmm(self, vmm: Any, initial_alpha: float = 1.0) -> dict[str, Any]:
        """Step 2: Pin mirror, reserve Expert VA, bind parameter views, load `initial_alpha`, and install hooks."""
        if self.mirror is None:
            raise RuntimeError("call offload_to_host_mirror(), offload_to_shared_mirror(), or attach_mirror() before bind_vmm()")
        if not 0.0 <= float(initial_alpha) <= 1.0:
            raise ValueError("initial_alpha must be in [0.0, 1.0]")

        self.vmm = vmm
        self.expert_id = vmm.register_expert(
            num_layers=self.num_layers,
            layer_stride_mib=self.layer_stride_mib,
            mirror=self.mirror,
            mirror_offset=self._mirror_offset,
            mirror_nbytes=self.total_mirror_bytes,
        )

        self._va_views = []
        is_sim = bool(getattr(vmm, "is_simulated", False))
        for plan in self.plans:
            vmm.set_expert_layer_used_bytes(self.expert_id, plan.layer_idx, max(1, plan.used_bytes))
            layer_u8_va = vmm.expert_layer_uint8_view(self.expert_id, plan.layer_idx)
            self._va_views.append(layer_u8_va)
            for ps in plan.params:
                p = ps.param_ref
                if p is None:
                    continue
                if not is_sim and hasattr(p, "data"):
                    p.data = layer_u8_va[ps.offset : ps.offset + ps.nbytes].view(p.dtype).view(ps.shape)
                elif hasattr(p, "bound_view"):
                    p.bound_view = (
                        layer_u8_va[ps.offset : ps.offset + ps.nbytes]
                        if len(layer_u8_va) >= ps.offset + ps.nbytes
                        else layer_u8_va
                    )

        initial_layers = int(round(float(initial_alpha) * self.num_layers))
        if initial_layers > 0:
            vmm.load_expert_layers_sync(self.expert_id, 0, initial_layers)
        self._has_pending_async_prefetch = False
        self._waited_layers.clear()
        self._all_resident_synced = bool(
            self.num_layers > 0 and vmm.expert_resident_count(self.expert_id) == self.num_layers
        )

        if self.spec.install_module_hooks and self.spec.hook_modules:
            self._install_layer_hooks()
        if self.spec.vae_module is not None:
            self._install_vae_hooks(self.spec.vae_module)

        return {
            "expert_id": self.expert_id,
            "layer_stride_mib": self.layer_stride_mib,
            "initial_resident_layers": vmm.expert_resident_count(self.expert_id),
        }

    def set_wait_gate(self, predicate: Callable[[], bool] | None) -> None:
        """Register an optional predicate gating `wait_layer_if_needed` (e.g. `mode == 'gen'`)."""
        self._should_wait_predicate = predicate

    def wait_layer_if_needed(self, layer_idx: int, method_tag: str = "forward") -> None:
        """Synchronize compute stream with background H2D prefetch for `layer_idx` when generation is active."""
        if self.vmm is None or not self._generation_active:
            return
        if self._should_wait_predicate is not None and not self._should_wait_predicate():
            return
        self.layer_hook_calls += 1
        self.hook_methods_triggered.add(method_tag)
        if self._all_resident_synced:
            return
        lid = int(layer_idx)
        s_ptr = _current_cuda_stream_ptr(self.vmm.device)
        self.vmm.wait_expert_layer(self.expert_id, lid, s_ptr)
        self._waited_layers.add(lid)
        if len(self._waited_layers) >= self.num_layers:
            self._all_resident_synced = True
            self._has_pending_async_prefetch = False

    def _install_layer_hooks(self) -> None:
        for layer_idx, mod in enumerate(self.spec.hook_modules):
            if hasattr(mod, "register_forward_pre_hook"):
                def _make_pre_hook(idx: int):
                    def _pre_hook(_module: Any, _inputs: Any) -> None:
                        self.wait_layer_if_needed(idx, "forward")

                    return _pre_hook

                self._hook_handles.append(mod.register_forward_pre_hook(_make_pre_hook(layer_idx)))

            for meth_name in self.spec.extra_hook_methods:
                orig_fn = getattr(mod, meth_name, None)
                if callable(orig_fn):
                    def _make_method_wrapper(idx: int, tag: str, fn: Any):
                        def _wrapped(*args: Any, **kwargs: Any) -> Any:
                            self.wait_layer_if_needed(idx, tag)
                            return fn(*args, **kwargs)

                        return _wrapped

                    setattr(mod, meth_name, _make_method_wrapper(layer_idx, meth_name, orig_fn))
                    self._wrapped_layer_methods.append((mod, meth_name, orig_fn))

    def _wrap_vae_callable(
        self,
        orig_fn: Callable[..., Any],
        *,
        is_decode: bool,
        output_cloner: Callable[[Any], Any] | None = None,
    ) -> Callable[..., Any]:
        eff_cloner = output_cloner if output_cloner is not None else self._vae_output_cloner

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            assert self.vmm is not None
            vmm = self.vmm
            was_gen_active = self._generation_active
            had_pending_prefetch = self._has_pending_async_prefetch
            prev_res = vmm.expert_resident_count(self.expert_id)
            self._generation_active = False
            self._all_resident_synced = False
            self._waited_layers.clear()
            vmm.set_expert_pinned(self.expert_id, False)
            completed_ok = False
            try:
                if eff_cloner is not None:
                    out = vmm.run_in_transient_scope(orig_fn, *args, clone_fn=eff_cloner, **kwargs)
                else:
                    out = vmm.run_in_transient_scope(orig_fn, *args, **kwargs)
                completed_ok = True
                return out
            finally:
                try:
                    if completed_ok and (not is_decode):
                        policy = self.spec.vae_encode_restore
                        cur_res = vmm.expert_resident_count(self.expert_id)
                        if was_gen_active:
                            target_res = self.num_layers if had_pending_prefetch else prev_res
                            if cur_res < target_res:
                                if policy == "sync" and not (self.spec.install_module_hooks and self.spec.hook_modules):
                                    self._ensure_expert_headroom(target_res - cur_res)
                                    vmm.load_expert_layers_sync(self.expert_id, 0, target_res)
                                    self._has_pending_async_prefetch = False
                                elif policy != "none" or (self.spec.install_module_hooks and self.spec.hook_modules):
                                    self._ensure_expert_headroom(target_res - cur_res)
                                    vmm.prefetch_expert_async(self.expert_id, 0, target_res)
                                    self._has_pending_async_prefetch = True
                                    if cur_res == 0:
                                        s_ptr = _current_cuda_stream_ptr(vmm.device)
                                        vmm.wait_expert_layer(self.expert_id, 0, s_ptr)
                        elif policy == "sync":
                            target_res = self.num_layers
                            if cur_res < target_res:
                                self._ensure_expert_headroom(target_res - cur_res)
                                vmm.load_expert_layers_sync(self.expert_id, 0, target_res)
                                self._has_pending_async_prefetch = False
                        elif policy == "async":
                            if cur_res < prev_res:
                                vmm.prefetch_expert_async(self.expert_id, 0, prev_res)
                                self._has_pending_async_prefetch = True
                        # `policy == "none"` outside `generation_mode` leaves borrowed slices lazily unmapped.
                finally:
                    self._generation_active = was_gen_active
                    if was_gen_active:
                        vmm.set_expert_pinned(self.expert_id, True)
                        self._all_resident_synced = bool(
                            (not self._has_pending_async_prefetch)
                            and vmm.expert_resident_count(self.expert_id) == self.num_layers
                        )

        return _wrapped

    def _install_vae_hooks(self, vae_mod: Any) -> None:
        for meth_name in ("decode", "encode"):
            orig_fn = getattr(vae_mod, meth_name, None)
            if callable(orig_fn):
                setattr(
                    vae_mod,
                    meth_name,
                    self._wrap_vae_callable(
                        orig_fn,
                        is_decode=(meth_name == "decode"),
                        output_cloner=self._vae_output_cloner,
                    ),
                )
                self._orig_vae_methods.append((vae_mod, meth_name, orig_fn))

    def _ensure_expert_headroom(
        self,
        missing_layers: int,
        *,
        exclude_session_ids: Sequence[int] = (),
    ) -> int:
        if self.vmm is None or missing_layers <= 0:
            return 0
        needed_bytes = int(missing_layers) * self.layer_stride_bytes
        s_ptr = _current_cuda_stream_ptr(self.vmm.device)
        return int(
            self.vmm.ensure_expert_headroom(
                self.expert_id,
                needed_bytes,
                exclude_session_ids=exclude_session_ids,
                stream=s_ptr,
            )
        )

    def resident_alpha(self) -> float:
        if self.vmm is None or self.num_layers <= 0:
            return 0.0
        return self.vmm.expert_resident_count(self.expert_id) / float(self.num_layers)

    def ensure_all_resident_sync(self, *, exclude_session_ids: Sequence[int] = ()) -> None:
        if self.vmm is not None:
            cur = self.vmm.expert_resident_count(self.expert_id)
            if cur < self.num_layers:
                self._ensure_expert_headroom(self.num_layers - cur, exclude_session_ids=exclude_session_ids)
                self.vmm.load_expert_layers_sync(self.expert_id, 0, self.num_layers)
            self._has_pending_async_prefetch = False
            self._waited_layers.clear()
            self._all_resident_synced = bool(self.vmm.expert_resident_count(self.expert_id) == self.num_layers)

    def set_residency_alpha(self, alpha: float) -> int:
        assert self.vmm is not None
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be in [0.0, 1.0]")
        target = max(0, min(self.num_layers, int(round(float(alpha) * self.num_layers))))
        s_ptr = _current_cuda_stream_ptr(self.vmm.device)
        self._generation_active = False
        self._all_resident_synced = False
        self._has_pending_async_prefetch = False
        self._waited_layers.clear()
        self.vmm.set_expert_pinned(self.expert_id, False)
        if target < self.num_layers:
            self.vmm.evict_expert_range(self.expert_id, target, self.num_layers, s_ptr)
        cur = self.vmm.expert_resident_count(self.expert_id)
        if target > cur:
            self._ensure_expert_headroom(target - cur)
        if target > 0:
            self.vmm.load_expert_layers_sync(self.expert_id, 0, target)
        res = int(self.vmm.expert_resident_count(self.expert_id))
        self._all_resident_synced = bool(res == self.num_layers)
        return res

    @contextmanager
    def generation_mode(
        self,
        prefetch_all: bool = True,
        pacer: Any = None,
        rank: int = 0,
        *,
        exclude_session_ids: Sequence[int] = (),
    ):
        """Pin expert weights, optionally acquire `HalfNodeH2DPacer` with async H2D release, and prefetch blocks."""
        assert self.vmm is not None
        vmm = self.vmm
        cur = vmm.expert_resident_count(self.expert_id)

        pacer_cm: Any = None
        if pacer is not None and prefetch_all and cur < self.num_layers:
            if hasattr(pacer, "acquire_process_slot"):
                pacer_cm = pacer.acquire_process_slot(int(rank))
            elif hasattr(pacer, "__enter__") and hasattr(pacer, "__exit__"):
                pacer_cm = pacer

        pacer_lock = threading.Lock() if pacer_cm is not None else None
        pacer_held = False
        h2d_release_thread: threading.Thread | None = None
        prev_release_cb = self._active_pacer_release_cb

        def _release_pacer_once(sync_stream: bool = False) -> None:
            nonlocal pacer_held
            if pacer_cm is None or pacer_lock is None:
                return
            with pacer_lock:
                if not pacer_held:
                    return
                if sync_stream:
                    _sync_cuda_stream(vmm.device)
                pacer_held = False
                pacer_cm.__exit__(None, None, None)

        if pacer_cm is not None:
            pacer_cm.__enter__()
            pacer_held = True
            self._active_pacer_release_cb = _release_pacer_once

        pinned_entered = False
        try:
            if prefetch_all and cur < self.num_layers:
                self._ensure_expert_headroom(self.num_layers - cur, exclude_session_ids=exclude_session_ids)
            vmm.set_expert_pinned(self.expert_id, True)
            pinned_entered = True
            self._generation_active = True
            self._waited_layers.clear()
            cur_pinned = vmm.expert_resident_count(self.expert_id)
            if cur_pinned == self.num_layers and not self._has_pending_async_prefetch:
                self._all_resident_synced = True
                _release_pacer_once(False)
            else:
                self._all_resident_synced = False
                if prefetch_all:
                    if (self.spec.install_module_hooks and self.spec.hook_modules) or self._should_wait_predicate is not None:
                        queued = vmm.prefetch_expert_async(self.expert_id, 0, self.num_layers)
                        if queued > 0:
                            self._has_pending_async_prefetch = True
                            if (
                                pacer_cm is not None
                                and callable(getattr(vmm, "sync_expert_h2d", None))
                                and not bool(getattr(vmm, "is_simulated", False))
                            ):
                                eid = int(self.expert_id)

                                def _wait_h2d_and_release() -> None:
                                    try:
                                        vmm.sync_expert_h2d(eid)
                                    except Exception:
                                        pass
                                    finally:
                                        _release_pacer_once(False)

                                h2d_release_thread = threading.Thread(
                                    target=_wait_h2d_and_release,
                                    name=f"ms-h2d-pacer-rel-{eid}",
                                    daemon=True,
                                )
                                h2d_release_thread.start()
                        else:
                            _release_pacer_once(False)
                    else:
                        self.ensure_all_resident_sync(exclude_session_ids=exclude_session_ids)
                        _release_pacer_once(False)
                else:
                    _release_pacer_once(False)
            yield self
        finally:
            self._active_pacer_release_cb = prev_release_cb
            if h2d_release_thread is not None:
                h2d_release_thread.join(timeout=10.0)
            _release_pacer_once(False)
            self._generation_active = False
            self._all_resident_synced = False
            self._waited_layers.clear()
            if pinned_entered:
                vmm.set_expert_pinned(self.expert_id, False)

    def unbind_vmm_keep_mirror(self) -> None:
        """Detach hooks and VA views while keeping host mirror ready for rebinding to a new VMM instance."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        for mod, meth_name, orig_fn in self._wrapped_layer_methods:
            setattr(mod, meth_name, orig_fn)
        self._wrapped_layer_methods.clear()
        for vae_mod, meth_name, orig_fn in self._orig_vae_methods:
            setattr(vae_mod, meth_name, orig_fn)
        self._orig_vae_methods.clear()

        for plan in self.plans:
            for ps in plan.params:
                p = ps.param_ref
                if p is not None and hasattr(p, "detach"):
                    import torch

                    p.data = torch.empty(0, dtype=p.dtype, device=p.device)
                elif p is not None and hasattr(p, "bound_view"):
                    p.bound_view = None
        self._va_views.clear()
        self._generation_active = False
        self.expert_id = -1
        self.vmm = None

    def close(self) -> None:
        self.unbind_vmm_keep_mirror()
        if self.mirror is not None and self._owns_mirror:
            self.mirror.close()
        self.mirror = None


def drop_partition_weights(spec: ExpertPartitionSpec, device: Any | None = None) -> int:
    """Drop GPU tensor storage for all parameters in `spec` (used by `static_nccl_fission` workers)."""
    import torch

    dropped_bytes = 0
    for plan in spec.plans:
        for ps in plan.params:
            p = ps.param_ref
            if p is not None and hasattr(p, "data"):
                target_dev = device if device is not None else p.device
                dropped_bytes += ps.nbytes
                p.data = torch.empty(0, dtype=p.dtype, device=target_dev)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dropped_bytes


class StaticFissionDropper:
    """Baseline `static_nccl_fission` secondary-expert dropper driven declaratively by `ExpertPartitionSpec`."""

    def __init__(self, spec: ExpertPartitionSpec, device: Any | None = None) -> None:
        self.spec = spec
        self.device = device
        self.dropped_bytes: int = 0

    def drop(self) -> int:
        self.dropped_bytes = drop_partition_weights(self.spec, device=self.device)
        return self.dropped_bytes

    def close(self) -> None:
        return None


class NaiveHostOffloader:
    """Baseline pinned-CPU async H2D/D2H weight offloader driven by `ExpertPartitionSpec`.

    Upgrades host staging buffers to pinned CPU memory (`pin_memory=True` when CUDA is available)
    with asynchronous `non_blocking=True` transfers and explicit stream synchronization, plus a
    reference-counted `hold_resident()` guard so multi-stage turns (`prepare_turn` -> `generate_dit`)
    do not offload and reload the full secondary expert twice within a single turn.
    """

    def __init__(self, spec: ExpertPartitionSpec, device: Any) -> None:
        self.spec = spec
        self.device = device
        self._cpu_shadows: list[tuple[Any, Any]] = []
        self._resident_on_gpu: bool = True
        self._hold_depth: int = 0
        self._pinned_host: bool = False
        self.swap_in_count: int = 0
        self.last_h2d_ms: float = 0.0
        self.total_h2d_ms: float = 0.0

    @property
    def is_resident(self) -> bool:
        return self._resident_on_gpu

    @property
    def is_pinned_host(self) -> bool:
        return self._pinned_host

    @property
    def hold_depth(self) -> int:
        return self._hold_depth

    def _is_cuda_device(self) -> bool:
        import torch

        if not torch.cuda.is_available():
            return False
        dev_type = getattr(self.device, "type", None)
        if dev_type is not None:
            return str(dev_type) == "cuda"
        return str(self.device).startswith("cuda")

    def offload_to_cpu(self) -> int:
        import torch

        is_cuda = self._is_cuda_device()
        moved_bytes = 0
        self._cpu_shadows.clear()
        pinned_all = is_cuda
        params_to_empty: list[Any] = []
        for plan in self.spec.plans:
            for ps in plan.params:
                p = ps.param_ref
                if p is not None and hasattr(p, "detach"):
                    src = p.detach().contiguous()
                    if is_cuda:
                        try:
                            cpu_t = torch.empty(src.shape, dtype=src.dtype, device="cpu", pin_memory=True)
                            cpu_t.copy_(src, non_blocking=True)
                        except RuntimeError:
                            pinned_all = False
                            cpu_t = src.cpu()
                    else:
                        pinned_all = False
                        cpu_t = src.cpu().clone()
                    self._cpu_shadows.append((p, cpu_t))
                    params_to_empty.append(p)
                    moved_bytes += ps.nbytes
        if is_cuda:
            torch.cuda.current_stream(self.device).synchronize()
        for p in params_to_empty:
            p.data = torch.empty(0, dtype=p.dtype, device=self.device)
        self._pinned_host = bool(pinned_all and self._cpu_shadows)
        self._resident_on_gpu = False
        self._hold_depth = 0
        gc.collect()
        if is_cuda:
            torch.cuda.empty_cache()
        return moved_bytes

    def swap_in_sync(self) -> bool:
        if self._resident_on_gpu or not self._cpu_shadows:
            return False
        import torch

        is_cuda = self._is_cuda_device()
        t0 = time.perf_counter()
        for p, cpu_t in self._cpu_shadows:
            if p.numel() == 0:
                if is_cuda:
                    p.data = cpu_t.to(self.device, non_blocking=True)
                else:
                    p.data = cpu_t.to(self.device)
        if is_cuda:
            torch.cuda.current_stream(self.device).synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.last_h2d_ms = round(dt_ms, 3)
        self.total_h2d_ms += dt_ms
        self.swap_in_count += 1
        self._resident_on_gpu = True
        return True

    def drop_from_gpu(self, *, force: bool = False) -> None:
        if (not force) and self._hold_depth > 0:
            return
        if not self._cpu_shadows or not self._resident_on_gpu:
            return
        import torch

        is_cuda = self._is_cuda_device()
        if is_cuda:
            torch.cuda.current_stream(self.device).synchronize()
        for p, _ in self._cpu_shadows:
            p.data = torch.empty(0, dtype=p.dtype, device=self.device)
        self._resident_on_gpu = False
        if is_cuda:
            torch.cuda.empty_cache()

    def acquire_resident(self, *, swap_in: bool = True) -> bool:
        """Increment residency hold refcount and swap expert weights onto GPU if not already resident."""
        self._hold_depth += 1
        if swap_in:
            try:
                return self.swap_in_sync()
            except BaseException:
                self._hold_depth = max(0, self._hold_depth - 1)
                raise
        return False

    def release_resident(self, *, drop_when_zero: bool = True) -> None:
        """Decrement residency hold refcount; drops GPU weights once refcount reaches 0."""
        if self._hold_depth > 0:
            self._hold_depth -= 1
        if self._hold_depth == 0 and drop_when_zero:
            self.drop_from_gpu()

    @contextmanager
    def hold_resident(
        self,
        *,
        swap_in: bool = True,
        drop_on_release: bool = True,
    ) -> Iterator["NaiveHostOffloader"]:
        """Keep expert weights resident across nested `prepare_turn` -> `generate_dit` phases."""
        self.acquire_resident(swap_in=swap_in)
        try:
            yield self
        finally:
            self.release_resident(drop_when_zero=drop_on_release)

    def close(self) -> None:
        self._hold_depth = 0
        self._resident_on_gpu = False
        self._pinned_host = False
        self._cpu_shadows.clear()
