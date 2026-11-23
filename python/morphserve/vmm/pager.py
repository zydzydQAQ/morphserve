"""Unified TriPool-VMM LayerExpertPager supporting BAGEL-7B-MoT, InternVL-U, and CPU simulation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import gc
import math
from typing import Any, Callable

from .tripool import NumaMemfdMirror

ALIGN_BYTES = 512
MIB = 1024**2


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
        )

    @classmethod
    def for_internvlu(
        cls,
        pipeline: Any,
        super_handle_mib: int = 64,
        wrap_vae: bool = True,
        install_module_hooks: bool = True,
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
        )

    @classmethod
    def for_internvlu_vlm(
        cls,
        pipeline: Any,
        super_handle_mib: int = 16,
        install_module_hooks: bool = False,
    ) -> "ExpertPartitionSpec":
        """Build balanced FFD `LayerPackPlan` bins for `InternVL-U` VLM stage co-paging (`Decision 2`)."""
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

        # First-Fit Decreasing (FFD) bin packing keeps `num_layers * stride_bytes` tightly bounded
        # (~4.6–4.9 GiB for 4.48 GiB VLM weights even with 594–742 MiB vocabulary embeddings).
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
        )


def _current_cuda_stream_ptr(device: Any) -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.current_stream(device).cuda_stream)
    except Exception:
        pass
    return 0


class LayerExpertPager:
    """Unified TriPool-VMM 64 MiB super-handle pager with async Step-1 H2D streaming & scoped VAE restoration."""

    def __init__(
        self,
        spec: ExpertPartitionSpec,
        numa_node: int = 0,
        should_wait_predicate: Callable[[], bool] | None = None,
    ) -> None:
        self.spec = spec
        self.numa_node = int(numa_node)
        self.plans = spec.plans
        self.num_layers = spec.num_layers
        self.layer_stride_bytes = spec.layer_stride_bytes
        self.layer_stride_mib = spec.layer_stride_mib
        self.total_mirror_bytes = self.num_layers * self.layer_stride_bytes
        self._should_wait_predicate = should_wait_predicate

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
        self.layer_hook_calls: int = 0
        self.hook_methods_triggered: set[str] = set()
        self._packed_params_count = 0
        self._packed_data_bytes = 0

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
        self.mirror = NumaMemfdMirror.create(
            nbytes=self.total_mirror_bytes,
            numa_node=self.numa_node,
            name=f"ms_{self.spec.name}",
        )
        self._owns_mirror = True
        self._mirror_offset = 0
        return self._populate_and_drop_gpu_weights(self.mirror, populate=True)

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
        self.mirror = shared_mirror
        self._owns_mirror = False
        self._mirror_offset = off
        return self._populate_and_drop_gpu_weights(shared_mirror, populate=populate)

    def bind_vmm(self, vmm: Any, initial_alpha: float = 1.0) -> dict[str, Any]:
        """Step 2: Pin mirror, reserve Expert VA, bind parameter views, load `initial_alpha`, and install hooks."""
        if self.mirror is None:
            raise RuntimeError("call offload_to_host_mirror() or offload_to_shared_mirror() before bind_vmm()")
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
        s_ptr = _current_cuda_stream_ptr(self.vmm.device)
        self.vmm.wait_expert_layer(self.expert_id, int(layer_idx), s_ptr)

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

    def _wrap_vae_callable(self, orig_fn: Callable[..., Any], *, is_decode: bool) -> Callable[..., Any]:
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            assert self.vmm is not None
            vmm = self.vmm
            was_gen_active = self._generation_active
            prev_res = vmm.expert_resident_count(self.expert_id)
            self._generation_active = False
            vmm.set_expert_pinned(self.expert_id, False)
            completed_ok = False
            try:
                out = vmm.run_in_transient_scope(orig_fn, *args, **kwargs)
                completed_ok = True
                return out
            finally:
                # Asymmetric lazy residency (`e27bce5`):
                # - After `vae.decode`, leave borrowed tail expert slices unmapped so `expert_alpha`
                #   naturally reflects demand-driven HBM pressure; the next DiT turn's
                #   `generation_mode(prefetch_all=True)` reloads them asynchronously during Step-1.
                # - After `vae.encode`: when inside `generation_mode`, queue async background prefetch;
                #   when outside `generation_mode` (e.g. `forward_cache_update_vae` right before
                #   `language_model(mode="gen")` runs without wait hooks), reload evicted layers
                #   synchronously now that `run_in_transient_scope` has released Arena-2 handles!
                try:
                    if completed_ok and (not is_decode) and vmm.expert_resident_count(self.expert_id) < prev_res:
                        if was_gen_active:
                            vmm.prefetch_expert_async(self.expert_id, 0, prev_res)
                        else:
                            missing = prev_res - vmm.expert_resident_count(self.expert_id)
                            self._ensure_expert_headroom(missing)
                            vmm.load_expert_layers_sync(self.expert_id, 0, prev_res)
                finally:
                    self._generation_active = was_gen_active
                    if was_gen_active:
                        vmm.set_expert_pinned(self.expert_id, True)

        return _wrapped

    def _install_vae_hooks(self, vae_mod: Any) -> None:
        for meth_name in ("decode", "encode"):
            orig_fn = getattr(vae_mod, meth_name, None)
            if callable(orig_fn):
                setattr(vae_mod, meth_name, self._wrap_vae_callable(orig_fn, is_decode=(meth_name == "decode")))
                self._orig_vae_methods.append((vae_mod, meth_name, orig_fn))

    def _ensure_expert_headroom(self, missing_layers: int, *, for_dit: bool = True) -> None:
        if self.vmm is None or missing_layers <= 0:
            return
        missing_bytes = int(missing_layers) * self.layer_stride_bytes
        s_ptr = _current_cuda_stream_ptr(self.vmm.device)
        opt_s = s_ptr if s_ptr != 0 else None
        if for_dit and hasattr(self.vmm, "maybe_evict_thinking_kv_on_high_watermark"):
            self.vmm.maybe_evict_thinking_kv_on_high_watermark(
                sync=True,
                for_dit=True,
                stream_ptr=opt_s,
            )
        st = self.vmm.stats_dict()
        pool_free = int(st.get("pool_free_bytes", 0))
        self_unpinned = int(self.vmm.expert_resident_count(self.expert_id)) * self.layer_stride_bytes
        other_unpinned = max(0, int(st.get("expert_unpinned_bytes", 0)) - self_unpinned)
        non_self_pool_free = pool_free + other_unpinned
        if non_self_pool_free < missing_bytes and hasattr(self.vmm, "reclaim_thinking_kv_bytes"):
            self.vmm.reclaim_thinking_kv_bytes(
                missing_bytes - non_self_pool_free,
                sync=True,
                stream_ptr=opt_s,
            )
        elif pool_free < missing_bytes and hasattr(self.vmm, "flush_kv_swaps"):
            self.vmm.flush_kv_swaps()

    def resident_alpha(self) -> float:
        if self.vmm is None or self.num_layers <= 0:
            return 0.0
        return self.vmm.expert_resident_count(self.expert_id) / float(self.num_layers)

    def ensure_all_resident_sync(self) -> None:
        if self.vmm is not None:
            cur = self.vmm.expert_resident_count(self.expert_id)
            if cur < self.num_layers:
                self._ensure_expert_headroom(self.num_layers - cur)
                self.vmm.load_expert_layers_sync(self.expert_id, 0, self.num_layers)

    def set_residency_alpha(self, alpha: float) -> int:
        assert self.vmm is not None
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be in [0.0, 1.0]")
        target = max(0, min(self.num_layers, int(round(float(alpha) * self.num_layers))))
        s_ptr = _current_cuda_stream_ptr(self.vmm.device)
        self._generation_active = False
        self.vmm.set_expert_pinned(self.expert_id, False)
        if target < self.num_layers:
            self.vmm.evict_expert_range(self.expert_id, target, self.num_layers, s_ptr)
        cur = self.vmm.expert_resident_count(self.expert_id)
        if target > cur:
            self._ensure_expert_headroom(target - cur, for_dit=(target == self.num_layers))
        if target > 0:
            self.vmm.load_expert_layers_sync(self.expert_id, 0, target)
        return int(self.vmm.expert_resident_count(self.expert_id))

    @contextmanager
    def generation_mode(self, prefetch_all: bool = True):
        assert self.vmm is not None
        if prefetch_all:
            cur = self.vmm.expert_resident_count(self.expert_id)
            if cur < self.num_layers:
                self._ensure_expert_headroom(self.num_layers - cur)
        self.vmm.set_expert_pinned(self.expert_id, True)
        self._generation_active = True
        if prefetch_all:
            if self.spec.install_module_hooks and self.spec.hook_modules or self._should_wait_predicate is not None:
                self.vmm.prefetch_expert_async(self.expert_id, 0, self.num_layers)
            else:
                self.ensure_all_resident_sync()
        try:
            yield self
        finally:
            self._generation_active = False
            self.vmm.set_expert_pinned(self.expert_id, False)

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


class NaiveHostOffloader:
    """Baseline synchronous `.to(device)` CPU weight offloader driven by `ExpertPartitionSpec`."""

    def __init__(self, spec: ExpertPartitionSpec, device: Any) -> None:
        self.spec = spec
        self.device = device
        self._cpu_shadows: list[tuple[Any, Any]] = []
        self._resident_on_gpu = True

    def offload_to_cpu(self) -> int:
        import torch

        moved_bytes = 0
        self._cpu_shadows.clear()
        for plan in self.spec.plans:
            for ps in plan.params:
                p = ps.param_ref
                if p is not None and hasattr(p, "detach"):
                    cpu_t = p.detach().contiguous().cpu()
                    self._cpu_shadows.append((p, cpu_t))
                    p.data = torch.empty(0, dtype=p.dtype, device=self.device)
                    moved_bytes += ps.nbytes
        self._resident_on_gpu = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return moved_bytes

    def swap_in_sync(self) -> bool:
        if self._resident_on_gpu or not self._cpu_shadows:
            return False
        import torch

        for p, cpu_t in self._cpu_shadows:
            if p.numel() == 0:
                p.data = cpu_t.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self._resident_on_gpu = True
        return True

    def drop_from_gpu(self) -> None:
        if not self._cpu_shadows:
            return
        import torch

        for p, _ in self._cpu_shadows:
            p.data = torch.empty(0, dtype=p.dtype, device=self.device)
        self._resident_on_gpu = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self) -> None:
        self._cpu_shadows.clear()
