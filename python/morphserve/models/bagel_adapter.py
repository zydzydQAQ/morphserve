"""Deep `BagelCPTGAdapter` (`tight_mot`) integrating `BagelSessionContext`, `VmmNaiveCache`, `LayerExpertPager`, and `BagelExecutionInterceptor`."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import gc
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

from morphserve.kernels import (
    cow_fork_context,
    decode_latent_to_u8_gpu,
    prepare_vae_images_gpu,
    prepare_vit_images_gpu,
    rebind_cow_prefix,
)
from morphserve.vmm import (
    ExpertPartitionSpec,
    NumaMemfdMirror,
    TriPoolVMM,
    VmmNaiveCache,
)
from morphserve.worker import (
    ArDecodeTask,
    BagelExecutionInterceptor,
    PreemptionSession,
    UnifiedPreemptiveScheduler,
)

from .base import (
    GpuCPTGAdapterMixin,
    PackedTransferState,
    compute_vmm_available_kv_bytes,
    is_oom_exception,
)

MIB = 1024**2
GIB = 1024**3


class _FallbackNaiveCache:
    """Minimal pure-Python `NaiveCache` fallback used only when offline tests inject a stub pipeline."""

    def __init__(self, num_layers: int) -> None:
        self.num_layers = int(num_layers)
        self.key_cache: dict[int, Any] = {l: None for l in range(self.num_layers)}
        self.value_cache: dict[int, Any] = {l: None for l in range(self.num_layers)}
        self.key_values_lens: list[int] | None = None


def apply_vllm_omni_mm_limit() -> None:
    from vllm_omni.model_executor.models.bagel.bagel import OmniBagelProcessingInfo

    max_images = int(os.environ.get("MORPHSERVE_MAX_IMAGE_ITEMS", "16"))
    if max_images < 1:
        raise ValueError("MORPHSERVE_MAX_IMAGE_ITEMS must be positive")

    def get_supported_mm_limits(self: Any) -> dict[str, int]:  # noqa: ARG001
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


def _cache_is_swapped(pkv: Any) -> bool:
    return bool(pkv is not None and getattr(pkv, "is_swapped", False))


def _cache_seq_len(pkv: Any, layer_idx: int = 0) -> int:
    """Inspect sequence length without triggering `VmmNaiveCache.ensure_resident()` or LRU touch."""
    if pkv is None:
        return 0
    if callable(getattr(pkv, "peek_seq_len", None)):
        return int(pkv.peek_seq_len(layer_idx))
    kc = pkv.key_cache
    k = dict.get(kc, layer_idx) if isinstance(kc, dict) else kc[layer_idx]
    return int(k.shape[0]) if k is not None else 0


def _peek_layer_kv(pkv: Any, layer_idx: int) -> tuple[Any | None, Any | None]:
    """Read `(k, v)` at `layer_idx` without triggering `VmmNaiveCache.ensure_resident()` or LRU touch."""
    if pkv is None:
        return None, None
    if callable(getattr(pkv, "peek_layer_kv", None)):
        return pkv.peek_layer_kv(layer_idx)
    kc, vc = pkv.key_cache, pkv.value_cache
    k = dict.get(kc, layer_idx) if isinstance(kc, dict) else kc[layer_idx]
    v = dict.get(vc, layer_idx) if isinstance(vc, dict) else vc[layer_idx]
    return k, v


def _truncate_bagel_cache(pkv: Any, target_seq_len: int) -> None:
    """Roll back a partially updated `NaiveCache` or `VmmNaiveCache` so all layers match `target_seq_len`."""
    if pkv is None:
        return
    if callable(getattr(pkv, "truncate", None)):
        pkv.truncate(target_seq_len)
        return
    target = max(0, int(target_seq_len))
    num_layers = int(pkv.num_layers)
    for lidx in range(num_layers):
        k, v = _peek_layer_kv(pkv, lidx)
        if target == 0:
            pkv.key_cache[lidx] = None
            pkv.value_cache[lidx] = None
        else:
            if k is not None and int(k.shape[0]) > target:
                pkv.key_cache[lidx] = k[:target]
            if v is not None and int(v.shape[0]) > target:
                pkv.value_cache[lidx] = v[:target]


def _close_ctx_vmm_if_any(ctx: dict[str, Any] | None) -> None:
    if ctx is None:
        return
    pkv = ctx.get("past_key_values")
    if pkv is not None and hasattr(pkv, "close"):
        pkv.close()


def bagel_ctx_nbytes(ctx: dict[str, Any] | None) -> int:
    """Compute deduplicated logical GPU bytes occupied by one BAGEL KV context without swapping in."""
    if ctx is None:
        return 0
    pkv = ctx.get("past_key_values")
    if pkv is None:
        return 0
    if callable(getattr(pkv, "logical_bytes", None)):
        return int(pkv.logical_bytes())
    if _cache_is_swapped(pkv):
        return int(getattr(pkv, "swapped_bytes", 0))
    num_layers = int(pkv.num_layers)
    seen: dict[int, int] = {}
    for lidx in range(num_layers):
        for t in _peek_layer_kv(pkv, lidx):
            if t is not None and hasattr(t, "data_ptr"):
                ptr = int(t.data_ptr())
                seen[ptr] = max(seen.get(ptr, 0), _tensor_logical_bytes(t))
    return sum(seen.values())


def pack_bagel_kv_context(
    ctx: dict[str, Any],
    device: Any = None,
    *,
    pin_memory: bool = False,
) -> tuple[dict[str, Any], Any]:
    """Single canonical BAGEL KV serializer for both wire shipping and pinned-host swap."""
    import torch

    pkv = ctx["past_key_values"]
    num_layers = int(pkv.num_layers)
    get_k = pkv.key_cache.get if isinstance(pkv.key_cache, dict) else (lambda i: pkv.key_cache[i])
    get_v = pkv.value_cache.get if isinstance(pkv.value_cache, dict) else (lambda i: pkv.value_cache[i])
    sample_k = next((get_k(l) for l in range(num_layers) if get_k(l) is not None), None)
    if device is None:
        device = sample_k.device if sample_k is not None else torch.device(f"cuda:{torch.cuda.current_device()}")

    to_cpu = str(device) == "cpu" or getattr(device, "type", None) == "cpu"
    layer_specs: list[list[Any] | None] = []
    tensors: list[Any] = []
    total_bytes = 0
    any_cuda = False
    cuda_dev: Any = None

    for lidx in range(num_layers):
        k = get_k(lidx)
        v = get_v(lidx)
        if k is None or v is None:
            layer_specs.append(None)
            continue
        layer_specs.append([list(k.shape), list(v.shape), str(k.dtype).replace("torch.", "")])
        tensors.append(k)
        tensors.append(v)
        total_bytes += _tensor_logical_bytes(k) + _tensor_logical_bytes(v)
        if torch.is_tensor(k) and k.is_cuda:
            any_cuda = True
            cuda_dev = k.device

    if to_cpu:
        use_pin = bool(pin_memory) and torch.cuda.is_available() and any_cuda
        flat = torch.empty((total_bytes,), dtype=torch.uint8, device="cpu", pin_memory=use_pin)
        cursor = 0
        for t in tensors:
            src_u8 = t.detach().contiguous().view(torch.uint8).reshape(-1)
            nb = int(src_u8.numel())
            if nb > 0:
                flat[cursor : cursor + nb].copy_(src_u8, non_blocking=use_pin)
                cursor += nb
        if any_cuda and cuda_dev is not None:
            torch.cuda.current_stream(cuda_dev).synchronize()
    else:
        chunks = [t.to(device).contiguous().view(torch.uint8).reshape(-1) for t in tensors]
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
    *,
    device: Any | None = None,
) -> dict[str, Any]:
    """Single canonical BAGEL KV deserializer for both wire shipping and pinned-host restore."""
    import torch

    num_layers = int(meta["num_layers"])
    pkv = cache_factory(num_layers)
    dst_device = device if device is not None else flat_u8.device
    need_copy_to_dst = str(dst_device) != str(flat_u8.device)
    is_cuda_dst = str(dst_device).startswith("cuda") and torch.cuda.is_available()
    non_block = bool(need_copy_to_dst and is_cuda_dst and flat_u8.is_pinned())

    off = 0
    for lidx, spec in enumerate(meta["layers"]):
        if spec is None:
            continue
        k_shape, v_shape, dt_name = spec
        dtype = getattr(torch, dt_name)
        elem_sz = int(torch.empty((), dtype=dtype).element_size())
        k_bytes = int(torch.Size(k_shape).numel() * elem_sz)
        v_bytes = int(torch.Size(v_shape).numel() * elem_sz)
        if need_copy_to_dst:
            k_dst = torch.empty(k_shape, dtype=dtype, device=dst_device)
            k_dst.view(torch.uint8).reshape(-1).copy_(flat_u8[off : off + k_bytes], non_blocking=non_block)
            off += k_bytes
            v_dst = torch.empty(v_shape, dtype=dtype, device=dst_device)
            v_dst.view(torch.uint8).reshape(-1).copy_(flat_u8[off : off + v_bytes], non_blocking=non_block)
            off += v_bytes
            pkv.key_cache[lidx] = k_dst
            pkv.value_cache[lidx] = v_dst
        else:
            pkv.key_cache[lidx] = flat_u8[off : off + k_bytes].view(dtype).reshape(k_shape).clone()
            off += k_bytes
            pkv.value_cache[lidx] = flat_u8[off : off + v_bytes].view(dtype).reshape(v_shape).clone()
            off += v_bytes

    if need_copy_to_dst and is_cuda_dst:
        torch.cuda.current_stream(dst_device).synchronize()
    return {"kv_lens": list(meta["kv_lens"]), "ropes": list(meta["ropes"]), "past_key_values": pkv}


@dataclass
class PendingDitDeferral:
    """Unified deferred DiT turn / VAE prefill state executed right before `EXEC_DIT`."""

    turn_cmd: dict[str, Any] | None = None
    vae_paths: list[str] = field(default_factory=list)
    vae_u8: bool = False
    vae_pil: bool = False

    @property
    def is_empty(self) -> bool:
        return (
            self.turn_cmd is None
            and not self.vae_paths
            and not self.vae_u8
            and not self.vae_pil
        )

    def copy(self) -> "PendingDitDeferral":
        return PendingDitDeferral(
            turn_cmd=dict(self.turn_cmd) if self.turn_cmd is not None else None,
            vae_paths=list(self.vae_paths),
            vae_u8=self.vae_u8,
            vae_pil=self.vae_pil,
        )

    def to_meta(self) -> dict[str, Any]:
        return {
            "pending_dit_turn": dict(self.turn_cmd) if self.turn_cmd is not None else None,
            "pending_vae_paths": list(self.vae_paths),
            "pending_vae_u8": self.vae_u8,
            "pending_vae_pil": self.vae_pil,
        }

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> "PendingDitDeferral | None":
        turn = dict(meta["pending_dit_turn"]) if meta.get("pending_dit_turn") is not None else None
        paths = [str(p) for p in (meta.get("pending_vae_paths") or [])]
        u8 = bool(meta.get("pending_vae_u8", False))
        pil = bool(meta.get("pending_vae_pil", False))
        obj = cls(turn_cmd=turn, vae_paths=paths, vae_u8=u8, vae_pil=pil)
        return None if obj.is_empty else obj


class BagelSessionContext:
    """Typed `SessionContext` for `BAGEL-7B-MoT` (`tight_mot`) with atomic OOM rollback, CoW lifecycle & unified swap."""

    def __init__(
        self,
        main_ctx: dict[str, Any] | None,
        *,
        cfg_text_ctx: dict[str, Any] | None = None,
        cfg_img_ctx: dict[str, Any] | None = None,
        cfg_text_prefix_len: int = 0,
        pending_deferral: PendingDitDeferral | None = None,
        pending_vae_only_paths: list[str] | None = None,
        last_img_u8: Any = None,
        last_img_pil: Any = None,
    ) -> None:
        self.main_ctx: dict[str, Any] | None = main_ctx
        self.cfg_text_ctx: dict[str, Any] | None = cfg_text_ctx
        self.cfg_img_ctx: dict[str, Any] | None = cfg_img_ctx
        self.cfg_text_prefix_len: int = int(cfg_text_prefix_len)
        self.pending_deferral: PendingDitDeferral | None = (
            None if (pending_deferral is None or pending_deferral.is_empty) else pending_deferral
        )
        self.pending_vae_only_paths: list[str] = [str(p) for p in (pending_vae_only_paths or [])]
        self.last_img_u8: Any = last_img_u8
        self.last_img_pil: Any = last_img_pil
        self._swapped_host_bundle: dict[str, Any] | None = None

    @property
    def main_pkv(self) -> Any | None:
        return self.main_ctx.get("past_key_values") if self.main_ctx is not None else None

    @property
    def main_session_id(self) -> int:
        pkv = self.main_pkv
        return int(getattr(pkv, "session_id", 0)) if pkv is not None else 0

    @property
    def is_swapped(self) -> bool:
        return self._swapped_host_bundle is not None or _cache_is_swapped(self.main_pkv)

    @property
    def has_pending_deferral(self) -> bool:
        return self.pending_deferral is not None and not self.pending_deferral.is_empty

    def ensure_deferral(self) -> PendingDitDeferral:
        if self.pending_deferral is None:
            self.pending_deferral = PendingDitDeferral()
        return self.pending_deferral

    def main_seq_len(self) -> int:
        return _cache_seq_len(self.main_pkv, 0)

    def has_live_main_kv(self) -> bool:
        if self.is_swapped or self.main_pkv is None:
            return False
        return self.main_seq_len() > 0

    def mark_active(self) -> None:
        pkv = self.main_pkv
        if pkv is not None and callable(getattr(pkv, "mark_active", None)):
            pkv.mark_active()

    def mark_thinking(self, *, trigger_watermark_check: bool = False) -> None:
        pkv = self.main_pkv
        if pkv is not None and callable(getattr(pkv, "mark_thinking", None)):
            pkv.mark_thinking(trigger_watermark_check=trigger_watermark_check)

    def drop_cfg_branches(self) -> None:
        _close_ctx_vmm_if_any(self.cfg_text_ctx)
        _close_ctx_vmm_if_any(self.cfg_img_ctx)
        self.cfg_text_ctx = None
        self.cfg_img_ctx = None
        self.cfg_text_prefix_len = 0

    def rollback_main_kv(self, target_seq_len: int) -> None:
        """Roll back `main_ctx` KV tensors cleanly to `target_seq_len`."""
        if self.main_ctx is not None:
            _truncate_bagel_cache(self.main_ctx.get("past_key_values"), target_seq_len)

    @contextmanager
    def atomic_kv_update(self):
        """Snapshot main KV lengths, CFG branches, and deferral state; roll back atomically on any exception."""
        orig_seq_len = self.main_seq_len()
        orig_kv_lens = list(self.main_ctx["kv_lens"]) if self.main_ctx is not None else [0]
        orig_ropes = list(self.main_ctx["ropes"]) if self.main_ctx is not None else [0]
        orig_deferral = self.pending_deferral.copy() if self.pending_deferral is not None else None
        orig_pending_vae_only = list(self.pending_vae_only_paths)
        try:
            yield self
        except BaseException:
            self.drop_cfg_branches()
            if self.main_ctx is not None:
                _truncate_bagel_cache(self.main_ctx.get("past_key_values"), orig_seq_len)
                self.main_ctx["kv_lens"] = orig_kv_lens
                self.main_ctx["ropes"] = orig_ropes
            self.pending_deferral = orig_deferral
            self.pending_vae_only_paths = orig_pending_vae_only
            raise

    def fork_cfg_text(self, aux_cache_factory: Callable[[int], Any], *, use_cow: bool) -> int:
        """Fork `cfg_text_ctx` from current `main_ctx` via `cow_fork_context`; returns prefix length."""
        assert self.main_ctx is not None
        prefix_len = self.main_seq_len()
        if prefix_len <= 0:
            self.cfg_text_ctx = None
            self.cfg_text_prefix_len = 0
            return 0
        self.cfg_text_ctx = cow_fork_context(
            self.main_ctx,
            prefix_len=prefix_len,
            cache_factory=aux_cache_factory,
            ropes=self.main_ctx["ropes"],
            kv_lens=self.main_ctx["kv_lens"],
            clone_slices=(not use_cow),
        )
        self.cfg_text_prefix_len = prefix_len
        return prefix_len

    def fork_cfg_img_from_main(self, aux_cache_factory: Callable[[int], Any], *, use_cow: bool) -> None:
        """Share or clone `cfg_img_ctx` directly from `main_ctx` when history was empty before prompt."""
        assert self.main_ctx is not None
        cur_len = self.main_seq_len()
        if cur_len <= 0:
            self.cfg_img_ctx = None
            return
        self.cfg_img_ctx = cow_fork_context(
            self.main_ctx,
            prefix_len=cur_len,
            cache_factory=aux_cache_factory,
            ropes=self.main_ctx["ropes"],
            kv_lens=self.main_ctx["kv_lens"],
            clone_slices=(not use_cow),
        )

    def rebind_cow_branches(self, *, use_cow: bool) -> None:
        """Restore O(1) CoW pointer sharing for `cfg_text_ctx` and zero-history `cfg_img_ctx`."""
        if not use_cow or self.main_pkv is None:
            return
        if self.cfg_text_ctx is not None and self.cfg_text_ctx.get("past_key_values") is not None:
            rebind_cow_prefix(self.main_pkv, self.cfg_text_ctx["past_key_values"])
        if (
            self.cfg_img_ctx is not None
            and self.cfg_img_ctx.get("past_key_values") is not None
            and self.cfg_text_prefix_len == 0
        ):
            rebind_cow_prefix(self.main_pkv, self.cfg_img_ctx["past_key_values"])

    def logical_bytes(self) -> int:
        """Compute deduplicated logical bytes across live branches (or swapped host bundle) without swapping in."""
        if self._swapped_host_bundle is not None:
            return int(self._swapped_host_bundle["total_bytes"])
        swapped_vmm_b = 0
        seen: dict[int, int] = {}
        for ctx in (self.main_ctx, self.cfg_text_ctx, self.cfg_img_ctx):
            if ctx is None or ctx.get("past_key_values") is None:
                continue
            pkv = ctx["past_key_values"]
            if _cache_is_swapped(pkv):
                swapped_vmm_b += int(getattr(pkv, "swapped_bytes", 0))
                continue
            num_layers = int(pkv.num_layers)
            for lidx in range(num_layers):
                for t in _peek_layer_kv(pkv, lidx):
                    if t is not None and hasattr(t, "data_ptr"):
                        ptr = int(t.data_ptr())
                        seen[ptr] = max(seen.get(ptr, 0), _tensor_logical_bytes(t))
        img_b = _tensor_logical_bytes(self.last_img_u8) if self.last_img_u8 is not None else 0
        return swapped_vmm_b + sum(seen.values()) + img_b

    def swap_out(
        self,
        *,
        vmm: Any = None,
        use_cow: bool = True,
        pin_memory: bool | None = None,
    ) -> tuple[int, bool]:
        """Swap session KV to Host Pinned DRAM via native VMM swap or the single KV serializer.

        Returns `(reclaimed_bytes, used_host_bundle)` where `used_host_bundle` is `True` only for
        the non-VMM/aux-branch pinned-host bundle path.
        """
        if self.is_swapped or self.main_ctx is None or self.main_pkv is None:
            return 0, False

        pkv = self.main_pkv
        if (
            vmm is not None
            and isinstance(pkv, VmmNaiveCache)
            and self.cfg_img_ctx is None
            and (self.cfg_text_ctx is None or use_cow)
        ):
            reclaimed = int(pkv.swap_out_async())
            if callable(getattr(vmm, "flush_kv_swaps", None)):
                vmm.flush_kv_swaps()
            return reclaimed, False

        import torch

        cfg_text_seq = _cache_seq_len(self.cfg_text_ctx["past_key_values"], 0) if self.cfg_text_ctx is not None else 0
        has_cfg_text = cfg_text_seq > 0
        cfg_text_is_prefix_only = bool(
            has_cfg_text and self.cfg_text_prefix_len > 0 and cfg_text_seq == self.cfg_text_prefix_len
        )

        use_pin = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        branches: dict[str, tuple[dict[str, Any], Any]] = {}
        total_bytes = 0

        m_meta, m_flat = pack_bagel_kv_context(self.main_ctx, device="cpu", pin_memory=use_pin)
        branches["main_ctx"] = (m_meta, m_flat)
        total_bytes += int(m_meta["total_bytes"])

        if has_cfg_text and not cfg_text_is_prefix_only and self.cfg_text_ctx is not None:
            ct_meta, ct_flat = pack_bagel_kv_context(self.cfg_text_ctx, device="cpu", pin_memory=use_pin)
            branches["cfg_text_ctx"] = (ct_meta, ct_flat)
            total_bytes += int(ct_meta["total_bytes"])

        if self.cfg_img_ctx is not None and _cache_seq_len(self.cfg_img_ctx.get("past_key_values"), 0) > 0:
            ci_meta, ci_flat = pack_bagel_kv_context(self.cfg_img_ctx, device="cpu", pin_memory=use_pin)
            branches["cfg_img_ctx"] = (ci_meta, ci_flat)
            total_bytes += int(ci_meta["total_bytes"])

        bundle = {
            "total_bytes": total_bytes,
            "branches": branches,
            "cfg_text_is_prefix_only": cfg_text_is_prefix_only,
            "cfg_text_prefix_len": self.cfg_text_prefix_len,
            "cfg_text_ropes": list(self.cfg_text_ctx["ropes"]) if (has_cfg_text and self.cfg_text_ctx is not None) else None,
            "cfg_text_kv_lens": list(self.cfg_text_ctx["kv_lens"]) if (has_cfg_text and self.cfg_text_ctx is not None) else None,
        }

        _close_ctx_vmm_if_any(self.main_ctx)
        self.drop_cfg_branches()
        self.main_ctx = None
        self._swapped_host_bundle = bundle
        return total_bytes, True

    def ensure_resident(
        self,
        *,
        device: Any,
        main_cache_factory: Callable[[int], Any],
        aux_cache_factory: Callable[[int], Any],
        use_cow_rebind: bool,
        vmm: Any = None,
    ) -> tuple[int, bool]:
        """Restore swapped session KV onto `device`. Returns `(restored_host_bundle_bytes, from_host_bundle)`."""
        bundle = self._swapped_host_bundle
        if bundle is not None:
            branches = bundle["branches"]
            m_meta, m_flat = branches["main_ctx"]
            main_ctx = unpack_bagel_kv_context(m_meta, m_flat, main_cache_factory, device=device)

            cfg_text_ctx = None
            prefix_len = int(bundle.get("cfg_text_prefix_len", 0))
            if "cfg_text_ctx" in branches:
                ct_meta, ct_flat = branches["cfg_text_ctx"]
                cfg_text_ctx = unpack_bagel_kv_context(ct_meta, ct_flat, aux_cache_factory, device=device)
                if use_cow_rebind:
                    rebind_cow_prefix(main_ctx["past_key_values"], cfg_text_ctx["past_key_values"])
            elif bundle.get("cfg_text_is_prefix_only") and prefix_len > 0:
                cfg_text_ctx = cow_fork_context(
                    main_ctx,
                    prefix_len=prefix_len,
                    cache_factory=aux_cache_factory,
                    ropes=bundle.get("cfg_text_ropes"),
                    kv_lens=bundle.get("cfg_text_kv_lens"),
                    clone_slices=(not use_cow_rebind),
                )

            cfg_img_ctx = None
            if "cfg_img_ctx" in branches:
                ci_meta, ci_flat = branches["cfg_img_ctx"]
                cfg_img_ctx = unpack_bagel_kv_context(ci_meta, ci_flat, aux_cache_factory, device=device)

            self.main_ctx = main_ctx
            self.cfg_text_ctx = cfg_text_ctx
            self.cfg_img_ctx = cfg_img_ctx
            self.cfg_text_prefix_len = prefix_len
            self._swapped_host_bundle = None
            return int(bundle["total_bytes"]), True

        pkv = self.main_pkv
        if pkv is not None and _cache_is_swapped(pkv):
            pkv.ensure_resident()
            if vmm is not None and callable(getattr(vmm, "flush_kv_swaps", None)):
                vmm.flush_kv_swaps()
        return 0, False

    def close(self) -> int:
        """Close all VMM regions and release any host swap buffer; returns released host-bundle bytes."""
        released_host_b = int(self._swapped_host_bundle["total_bytes"]) if self._swapped_host_bundle is not None else 0
        self._swapped_host_bundle = None
        _close_ctx_vmm_if_any(self.main_ctx)
        self.drop_cfg_branches()
        self.main_ctx = None
        self.pending_deferral = None
        self.pending_vae_only_paths.clear()
        self.last_img_u8 = None
        self.last_img_pil = None
        return released_host_b


def pack_bagel_multi_context(
    session: BagelSessionContext,
    include_cfg_text_dup: bool,
    device: Any,
) -> PackedTransferState:
    """Pack multi-turn `BagelSessionContext` (`main_ctx`, CFG branches/metadata, and loopback image state)."""
    import numpy as np
    import torch

    assert session.main_ctx is not None
    main_ctx = session.main_ctx
    cfg_text_ctx = session.cfg_text_ctx
    cfg_img_ctx = session.cfg_img_ctx
    cfg_text_prefix_len = session.cfg_text_prefix_len
    defer_meta = (
        session.pending_deferral.to_meta()
        if session.pending_deferral is not None
        else PendingDitDeferral().to_meta()
    )

    has_cfg_text = cfg_text_ctx is not None and _cache_seq_len(cfg_text_ctx.get("past_key_values"), 0) > 0
    cfg_text_ropes = list(cfg_text_ctx["ropes"]) if (has_cfg_text and cfg_text_ctx is not None) else None
    cfg_text_kv_lens = list(cfg_text_ctx["kv_lens"]) if (has_cfg_text and cfg_text_ctx is not None) else None

    named_items: list[tuple[str, dict[str, Any]]] = [("main", main_ctx)]
    if include_cfg_text_dup and has_cfg_text and cfg_text_ctx is not None:
        named_items.append(("cfg_text", cfg_text_ctx))
    if cfg_img_ctx is not None and _cache_seq_len(cfg_img_ctx.get("past_key_values"), 0) > 0:
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
    last_u8 = session.last_img_u8
    last_pil = session.last_img_pil
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
        "pending_vae_only_paths": list(session.pending_vae_only_paths),
        **defer_meta,
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
) -> BagelSessionContext:
    """Unpack wire metadata + buffer into a `BagelSessionContext` restoring CoW prefix & loopback image state."""
    sections = meta["sections"]
    m_sec = sections["main"]
    main_ctx = unpack_bagel_kv_context(
        m_sec["meta"],
        flat_u8[m_sec["offset"] : m_sec["offset"] + m_sec["nbytes"]],
        main_cache_factory,
    )
    cfg_text_ctx = None
    prefix_len = int(meta.get("cfg_text_prefix_len", 0))
    if "cfg_text" in sections:
        ct_sec = sections["cfg_text"]
        cfg_text_ctx = unpack_bagel_kv_context(
            ct_sec["meta"],
            flat_u8[ct_sec["offset"] : ct_sec["offset"] + ct_sec["nbytes"]],
            aux_cache_factory,
        )
        if use_cow_rebind:
            rebind_cow_prefix(main_ctx["past_key_values"], cfg_text_ctx["past_key_values"])
    elif prefix_len > 0:
        cfg_text_ctx = cow_fork_context(
            main_ctx,
            prefix_len=prefix_len,
            cache_factory=aux_cache_factory,
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

    return BagelSessionContext(
        main_ctx=main_ctx,
        cfg_text_ctx=cfg_text_ctx,
        cfg_img_ctx=cfg_img_ctx,
        cfg_text_prefix_len=prefix_len,
        pending_deferral=PendingDitDeferral.from_meta(meta),
        pending_vae_only_paths=[str(p) for p in (meta.get("pending_vae_only_paths") or [])],
        last_img_u8=last_img_u8,
        last_img_pil=last_img_pil,
    )


class BagelCPTGAdapter(GpuCPTGAdapterMixin):
    """Production `BAGEL-7B-MoT` (`tight_mot`) adapter integrating `BagelSessionContext`, `VmmNaiveCache`, `LayerExpertPager` & `BagelExecutionInterceptor`."""

    architecture = "tight_mot"

    def __init__(
        self,
        rank: int,
        policy: Any,
        model_dir: Path | str | None = None,
        bench_root: Path | str | None = None,
        numa_node: int = 0,
        *,
        pipeline: Any | None = None,
        naive_cache_cls: type | None = None,
        device: Any | None = None,
    ) -> None:
        self.rank = int(rank)
        self.policy = policy
        self.model_dir = Path(model_dir) if model_dir is not None else Path(".")
        self.bench_root = Path(bench_root) if bench_root is not None else Path(".")
        self.numa_node = int(numa_node)
        self.role = policy.role_of(self.rank)
        self._init_gpu_lifecycle_state()

        if naive_cache_cls is not None:
            self._naive_cache_cls = naive_cache_cls
        else:
            try:
                from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache

                self._naive_cache_cls = NaiveCache
            except ImportError:
                self._naive_cache_cls = getattr(pipeline, "naive_cache_cls", _FallbackNaiveCache)

        if device is not None:
            self.device = device
        elif pipeline is not None and getattr(pipeline, "device", None) is not None:
            self.device = pipeline.device
        else:
            import torch

            self.device = torch.device(f"cuda:{self.rank}")

        if pipeline is not None:
            self.pipeline = pipeline
        else:
            import torch

            if getattr(policy, "gpu_memory_fraction", None) is not None and torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(float(policy.gpu_memory_fraction), self.device)
            self.pipeline = load_bagel_pipeline(self.model_dir, device_index=self.rank)

        lm_cfg = getattr(getattr(self.pipeline.bagel, "config", None), "llm_config", None)
        if lm_cfg is not None and hasattr(lm_cfg, "num_hidden_layers"):
            self.num_expert_layers = int(lm_cfg.num_hidden_layers)
        else:
            self.num_expert_layers = len(self.pipeline.bagel.language_model.model.layers)

        self.interceptor = BagelExecutionInterceptor.for_pipeline(self.pipeline)
        self.sched: UnifiedPreemptiveScheduler | None = None

        self._img_cache: dict[str, Any] = {}
        self._ctx_store: dict[str, BagelSessionContext] = {}
        self._active_ctx_pins: dict[str, int] = {}
        self._ctx_last_used_s: dict[str, float] = {}
        self._kv_host_swap_outs: int = 0
        self._kv_host_swap_ins: int = 0
        self._kv_host_pinned_bytes: int = 0
        self._calibrated_ar_step_ms: float | None = None
        self._calibrated_layer_ms_by_hw: dict[tuple[int, int], float] = {}
        self._naive_offload_held_ctx_ids: set[str] = set()
        self._pending_h2d_weight_ms: dict[str, float] = {}
        self._pending_vae_encode_ms: dict[str, float] = {}

        if getattr(self.pipeline.bagel, "language_model", None) is not None:
            gen_spec = ExpertPartitionSpec.for_bagel(
                self.pipeline,
                role="generation",
                super_handle_mib=policy.super_handle_mib,
                wrap_vae=policy.use_scoped_vae,
                vae_encode_restore="sync",
            )
            pager = self.bind_expert_partition(gen_spec, drop_for_fission=(self.role == "und"))
            if pager is not None:
                self.interceptor.bind_pager(pager)

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

    def _main_cache_factory(self) -> Callable[[int], Any]:
        if self.vmm is not None:
            return lambda n: VmmNaiveCache.create_for_bagel(self._naive_cache_cls, n, self.vmm)
        return lambda n: self._naive_cache_cls(n)

    def _aux_cache_factory(self) -> Callable[[int], Any]:
        return lambda n: self._naive_cache_cls(n)

    def _new_main_ctx(self) -> dict[str, Any]:
        return {"kv_lens": [0], "ropes": [0], "past_key_values": self._main_cache_factory()(self.num_expert_layers)}

    def _new_aux_ctx(self) -> dict[str, Any]:
        return {"kv_lens": [0], "ropes": [0], "past_key_values": self._aux_cache_factory()(self.num_expert_layers)}

    def _sync_ctx_vmm_activity(self, cid: str, *, active: bool) -> None:
        session = self._ctx_store.get(cid)
        if session is None:
            return
        if active:
            session.mark_active()
        else:
            session.mark_thinking(trigger_watermark_check=False)

    def _pin_ctx_id(self, cid: str) -> None:
        self._active_ctx_pins[cid] = self._active_ctx_pins.get(cid, 0) + 1
        self._sync_ctx_vmm_activity(cid, active=True)

    def _unpin_ctx_id(self, cid: str) -> None:
        rem = self._active_ctx_pins.get(cid, 1) - 1
        if rem <= 0:
            self._active_ctx_pins.pop(cid, None)
            self._sync_ctx_vmm_activity(cid, active=False)
        else:
            self._active_ctx_pins[cid] = rem
        if cid in self._ctx_store:
            self._ctx_last_used_s[cid] = time.perf_counter()

    @contextmanager
    def _pinned_ctx(self, *ctx_ids: str):
        for cid in ctx_ids:
            self._pin_ctx_id(cid)
        try:
            yield
        finally:
            for cid in ctx_ids:
                self._unpin_ctx_id(cid)

    def available_kv_bytes(self, *, for_dit: bool = False) -> int:
        import torch

        is_cuda = torch.cuda.is_available() and getattr(self.device, "type", "cuda") == "cuda"
        if self.vmm is not None:
            vmm_avail = compute_vmm_available_kv_bytes(
                self.vmm,
                has_pageable_experts=(self.pager is not None),
                for_dit=for_dit,
            )
            vmm_st = self.vmm.stats_dict()
            vmm_includes_overflow = callable(getattr(self.vmm, "kv_free_bytes", None)) or ("kv_free_bytes" in vmm_st)
            if self.policy.allow_vmm_overflow and is_cuda:
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

        if not is_cuda:
            live_b = sum(s.logical_bytes() for s in self._ctx_store.values() if not s.is_swapped)
            return max(0, 16 * GIB - live_b)
        free_drv, total_drv = torch.cuda.mem_get_info(self.device)
        if getattr(self.policy, "gpu_memory_fraction", None) is not None:
            quota_b = int(total_drv * float(self.policy.gpu_memory_fraction))
            alloc_b = int(torch.cuda.memory_allocated(self.device))
            return max(0, quota_b - alloc_b - 512 * MIB)
        unused_cached = max(
            0, int(torch.cuda.memory_reserved(self.device)) - int(torch.cuda.memory_allocated(self.device))
        )
        return max(0, int(free_drv) + unused_cached - 1024 * MIB)

    def _auto_idle_kv_swap_enabled(self) -> bool:
        val = getattr(self.policy, "use_idle_kv_swap", None)
        if val is not None:
            return bool(val)
        return bool(self.pager is not None or getattr(self.policy, "kv_vmm_only", False))

    def _release_naive_hold(self, ctx_id: str | None = None, *, force: bool = False) -> None:
        if ctx_id is not None:
            self._naive_offload_held_ctx_ids.discard(ctx_id)
        if force:
            self._naive_offload_held_ctx_ids.clear()
        if self.naive_offloader is not None and (force or not self._naive_offload_held_ctx_ids):
            if bool(getattr(self.naive_offloader, "_resident_on_gpu", False)):
                self._update_hwm()
                self.naive_offloader.drop_from_gpu(force=force)

    def evict_idle_kv_to_host(
        self,
        target_free_bytes: int,
        exclude_ctx_ids: Iterable[str] = (),
        *,
        for_dit: bool = False,
    ) -> int:
        """Evict LRU unpinned think-time sessions' KV to Host Pinned DRAM until `available_kv_bytes >= target`."""
        excluded = set(exclude_ctx_ids) | {cid for cid, cnt in self._active_ctx_pins.items() if cnt > 0}
        candidates: list[tuple[float, str, BagelSessionContext]] = []
        for cid, session in self._ctx_store.items():
            if cid in excluded or cid.startswith("__warm_"):
                continue
            if not session.has_live_main_kv():
                continue
            candidates.append((self._ctx_last_used_s.get(cid, 0.0), cid, session))

        candidates.sort(key=lambda item: item[0])
        evicted_total = 0
        for _, _, session in candidates:
            if self.available_kv_bytes(for_dit=for_dit) >= int(target_free_bytes):
                break
            reclaimed, used_bundle = session.swap_out(
                vmm=self.vmm,
                use_cow=self.policy.use_module3_cow_loopback,
            )
            evicted_total += reclaimed
            if used_bundle:
                self._kv_host_swap_outs += 1
                self._kv_host_pinned_bytes += reclaimed
        if evicted_total > 0 and self.vmm is None:
            import torch

            if torch.cuda.is_available():
                self._update_hwm()
                torch.cuda.empty_cache()
        return evicted_total

    def _release_vmm_overflow_to_driver(self, target_bytes: int = 0) -> int:
        if self.vmm is None:
            return 0
        fn = getattr(self.vmm, "release_free_overflow_bytes", None)
        if not callable(fn):
            return 0
        return int(fn(int(max(0, target_bytes))))

    def _ensure_cuda_driver_headroom_for_dit(
        self,
        needed_driver_bytes: int,
        *,
        exclude_ctx_ids: Iterable[str] = (),
    ) -> int:
        """Ensure CUDA driver + PyTorch caching allocator has sufficient headroom before DiT/VAE spikes."""
        import torch

        if self.vmm is None or not (torch.cuda.is_available() and getattr(self.device, "type", "cuda") == "cuda"):
            return 0
        free_drv, _ = torch.cuda.mem_get_info(self.device)
        unused_pt = max(
            0,
            int(torch.cuda.memory_reserved(self.device)) - int(torch.cuda.memory_allocated(self.device)),
        )
        free_est = int(free_drv) + unused_pt
        trimmed = 0
        if free_est < int(needed_driver_bytes):
            deficit = int(needed_driver_bytes) - free_est
            trimmed += self._release_vmm_overflow_to_driver(deficit)
            free_drv, _ = torch.cuda.mem_get_info(self.device)
            unused_pt = max(
                0,
                int(torch.cuda.memory_reserved(self.device)) - int(torch.cuda.memory_allocated(self.device)),
            )
            free_est = int(free_drv) + unused_pt
            if free_est < int(needed_driver_bytes) and self._auto_idle_kv_swap_enabled():
                vmm_pool_b = int(self.vmm.stats_dict().get("pool_total_bytes", 0))
                init_pool_b = int(round(float(getattr(self, "_vmm_initial_pool_mib", 0.0)) * MIB))
                vmm_excess_b = max(0, vmm_pool_b - init_pool_b)
                if vmm_excess_b > 0:
                    rem_deficit = min(int(needed_driver_bytes) - free_est, vmm_excess_b)
                    target_avail = self.available_kv_bytes(for_dit=True) + rem_deficit
                    reclaimed = self.evict_idle_kv_to_host(
                        target_avail,
                        exclude_ctx_ids=exclude_ctx_ids,
                        for_dit=True,
                    )
                    if reclaimed > 0:
                        flush_fn = getattr(self.vmm, "flush_kv_swaps", None)
                        if callable(flush_fn):
                            flush_fn()
                        trimmed += self._release_vmm_overflow_to_driver(rem_deficit)
                        free_drv, _ = torch.cuda.mem_get_info(self.device)
                        unused_pt = max(
                            0,
                            int(torch.cuda.memory_reserved(self.device)) - int(torch.cuda.memory_allocated(self.device)),
                        )
        # Defragment PyTorch's caching allocator when raw driver free is tight so large contiguous
        # DiT/VAE/FlashAttn allocations (1.0-1.9 GiB) never fail with fragmented CUDACachingAllocator OOM.
        if int(free_drv) < min(int(needed_driver_bytes), 3072 * MIB) and unused_pt >= 512 * MIB:
            self._update_hwm()
            torch.cuda.empty_cache()
        return trimmed

    def reclaim_driver_headroom(
        self,
        needed_bytes: int,
        exclude_ctx_ids: Iterable[str] = (),
    ) -> int:
        """Reclaim CUDA driver + PyTorch caching allocator headroom for large NCCL staging buffers."""
        import torch

        needed = max(0, int(needed_bytes))
        if needed <= 0 or not (torch.cuda.is_available() and getattr(self.device, "type", "cuda") == "cuda"):
            return 0
        reclaimed = 0
        if self.vmm is not None:
            reclaimed += self._ensure_cuda_driver_headroom_for_dit(needed, exclude_ctx_ids=exclude_ctx_ids)
        elif self._auto_idle_kv_swap_enabled():
            free_drv, _ = torch.cuda.mem_get_info(self.device)
            unused_pt = max(
                0,
                int(torch.cuda.memory_reserved(self.device)) - int(torch.cuda.memory_allocated(self.device)),
            )
            if int(free_drv) + unused_pt < needed:
                deficit = needed - (int(free_drv) + unused_pt)
                target_avail = max(needed, self.available_kv_bytes(for_dit=False) + deficit)
                reclaimed += self.evict_idle_kv_to_host(
                    target_avail, exclude_ctx_ids=exclude_ctx_ids, for_dit=False
                )
        free_drv, _ = torch.cuda.mem_get_info(self.device)
        if int(free_drv) < needed:
            self._update_hwm()
            torch.cuda.empty_cache()
        return reclaimed

    def _ensure_ctx_resident(self, ctx_id: str, extra_needed_bytes: int = 0) -> BagelSessionContext:
        session = self._ctx_store[ctx_id]
        allow_auto_swap = self._auto_idle_kv_swap_enabled()
        if session.is_swapped:
            req_b = session.logical_bytes() + int(extra_needed_bytes) + 256 * MIB
            if allow_auto_swap and self.available_kv_bytes() < req_b:
                self.evict_idle_kv_to_host(req_b, exclude_ctx_ids=(ctx_id,))
            try:
                restored_b, from_bundle = session.ensure_resident(
                    device=self.device,
                    main_cache_factory=self._main_cache_factory(),
                    aux_cache_factory=self._aux_cache_factory(),
                    use_cow_rebind=self.policy.use_module3_cow_loopback,
                    vmm=self.vmm,
                )
            except BaseException as exc:
                if (
                    allow_auto_swap
                    and is_oom_exception(exc)
                    and self.evict_idle_kv_to_host(1 << 50, exclude_ctx_ids=(ctx_id,)) > 0
                ):
                    restored_b, from_bundle = session.ensure_resident(
                        device=self.device,
                        main_cache_factory=self._main_cache_factory(),
                        aux_cache_factory=self._aux_cache_factory(),
                        use_cow_rebind=self.policy.use_module3_cow_loopback,
                        vmm=self.vmm,
                    )
                else:
                    raise
            if from_bundle:
                self._kv_host_pinned_bytes = max(0, self._kv_host_pinned_bytes - restored_b)
                self._kv_host_swap_ins += 1
            if self._active_ctx_pins.get(ctx_id, 0) > 0:
                self._sync_ctx_vmm_activity(ctx_id, active=True)
            return session

        if allow_auto_swap and extra_needed_bytes > 0:
            watermark_b = int(extra_needed_bytes) + 512 * MIB
            if self.available_kv_bytes() < watermark_b:
                self.evict_idle_kv_to_host(watermark_b, exclude_ctx_ids=(ctx_id,))
        return session

    def bind_runtime_resources(self, pacer: Any | None = None) -> dict[str, Any]:
        import torch

        self.pacer = pacer
        is_cuda = torch.cuda.is_available() and getattr(self.device, "type", "cuda") == "cuda"
        if is_cuda:
            self._ensure_nvml_poller()
        if self.policy.use_tripool_vmm:
            gc.collect()
            if is_cuda:
                self._update_hwm()
                torch.cuda.empty_cache()
            init_pool_mib = int(self.policy.super_pool_mib + self.policy.fine_kv_pool_mib)
            self._vmm_initial_pool_mib = float(init_pool_mib)
            max_pool_b = 0
            if (
                (self.pager is not None or bool(getattr(self.policy, "kv_vmm_only", False)))
                and self.policy.allow_vmm_overflow
                and getattr(self.policy, "gpu_memory_fraction", None) is None
                and getattr(self.policy, "gpu_memory_quota_mib", None) is None
                and is_cuda
            ):
                free_drv, total_drv = torch.cuda.mem_get_info(self.device)
                if int(total_drv) > 0:
                    nccl_reserve_mib = (
                        (1024 + 64 * max(0, int(self.policy.num_gpus) - 8))
                        if (
                            int(self.policy.num_gpus) >= 2
                            and str(self.policy.routing_mode) in ("dual_path", "static_nccl_fission")
                        )
                        else 0
                    )
                    headroom_b = (nccl_reserve_mib + 5120) * MIB
                    max_pool_b = max(init_pool_mib * MIB, int(free_drv) - headroom_b)
            self.vmm = TriPoolVMM(
                device=self.rank,
                super_pool_mib=self.policy.super_pool_mib,
                fine_kv_pool_mib=self.policy.fine_kv_pool_mib,
                super_handle_mib=self.policy.super_handle_mib,
                transient_va_gib=self.policy.transient_va_gib,
                kv_va_gib=self.policy.kv_va_gib,
                allow_overflow=self.policy.allow_vmm_overflow,
                max_pool_bytes=max_pool_b,
            )
            if self.pager is not None:
                init_alpha = float(self.policy.initial_alphas.get(self.rank, 1.0))
                self._initial_bound_alpha = init_alpha
                self.pager.bind_vmm(self.vmm, initial_alpha=init_alpha)

        if self.policy.use_module3_cow_loopback:
            self.interceptor.enable_shared_prefix_cfg(
                min_shared_tokens=32,
                allow_cascade_split=bool(self.policy.allow_cascade_cfg_split),
            )

        if self.policy.preemption_mode != "non_preemptive":
            self.sched = UnifiedPreemptiveScheduler(
                self.pipeline,
                mode=self.policy.preemption_mode,
                tpot_slo_ms=self.policy.ar_tpot_slo_ms,
                guard_ms=26.0,
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
        if self.role != "gen":
            w_ctx = self._new_aux_ctx()
            self._update_text_ctx(w_ctx, "Warmup prompt.")
            self._ctx_store["__warm_ar__"] = BagelSessionContext(w_ctx)
            warm_out = self.decode_ar("__warm_ar__", max_tokens=3)
            tpots = warm_out.get("tpots_ms") or []
            if tpots:
                self._calibrated_ar_step_ms = max(5.0, min(66.0, float(tpots[-1])))
                if self.sched is not None:
                    self.sched.ema_ar_step_ms = self._calibrated_ar_step_ms
                    self.sched.guard_ms = 26.0
                    self.sched.last_effective_chunk_layers = self.sched.compute_adaptive_chunk_layers()
            self.release_context("__warm_ar__")

        if self.role != "und":
            w_main = self._new_aux_ctx()
            w_img = self._new_aux_ctx()
            self._update_text_ctx(w_main, "Warmup.")
            self._update_text_ctx(w_img, "Warmup.")
            self._ctx_store["__warm_dit__"] = BagelSessionContext(w_main, cfg_img_ctx=w_img)
            _ = self.generate_dit(
                "__warm_dit__",
                {"ctx_id": "__warm_dit__", "height": 512, "width": 512, "dit_steps": 2, "seed": 1, "prompt": "Warmup."},
            )
            self.release_context("__warm_dit__")
            if self.pager is not None and self.vmm is not None:
                self.pager.set_residency_alpha(self.policy.initial_alphas.get(self.rank, 1.0))

        self.reset_peak_memory_stats()

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
        ac_enabled = getattr(self.device, "type", "cuda") != "cpu"
        with torch.inference_mode(), torch.autocast(
            device_type=getattr(self.device, "type", "cuda"),
            enabled=ac_enabled,
            dtype=torch.bfloat16,
        ):
            ctx["past_key_values"] = self.pipeline.bagel.forward_cache_update_text(ctx["past_key_values"], **inp)
        ctx["kv_lens"], ctx["ropes"] = nl, nr

    def _update_image_ctx(
        self,
        ctx: dict[str, Any],
        img_or_u8: Any,
        update_vae: bool = True,
        update_vit: bool = True,
        *,
        keep_naive_resident: bool = False,
        pacer_already_held: bool = False,
        timing_out: dict[str, float] | None = None,
    ) -> None:
        """Unified PIL & GPU-uint8 image injection sharing VAE seed and running ViT prep outside autocast(bf16)."""
        import numpy as np
        import torch

        swapped = False
        failed = False
        if update_vae and self.naive_offloader is not None:
            t_sw0 = time.perf_counter()
            swapped = self.naive_offloader.swap_in_sync()
            if swapped and timing_out is not None:
                timing_out["h2d_weight_ms"] = timing_out.get("h2d_weight_ms", 0.0) + (
                    (time.perf_counter() - t_sw0) * 1000.0
                )
        try:
            is_u8 = torch.is_tensor(img_or_u8)
            dev_type = getattr(self.device, "type", "cuda")
            ac_enabled = dev_type != "cpu"
            with torch.inference_mode():
                if update_vae:
                    t_vae0 = time.perf_counter()
                    vae_seed = 10007 + (int(ctx["ropes"][0]) if ctx.get("ropes") else 0)
                    torch.manual_seed(vae_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(vae_seed)
                    if is_u8:
                        vi, nl, nr = prepare_vae_images_gpu(
                            self.pipeline.bagel,
                            [img_or_u8],
                            ctx["kv_lens"],
                            ctx["ropes"],
                            self.pipeline.new_token_ids,
                        )
                    else:
                        def _vae_tf(v: Any) -> Any:
                            arr = torch.from_numpy(np.array(v.convert("RGB"), copy=True)).float() / 127.5 - 1.0
                            return arr.permute(2, 0, 1)

                        vi, nl, nr = self.pipeline.bagel.prepare_vae_images(
                            curr_kvlens=ctx["kv_lens"],
                            curr_rope=ctx["ropes"],
                            images=[img_or_u8],
                            transforms=_vae_tf,
                            new_token_ids=self.pipeline.new_token_ids,
                        )
                        vi = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in vi.items()}
                    pkv_sid = int(getattr(ctx.get("past_key_values"), "session_id", 0) or 0)
                    excl_sids = (pkv_sid,) if pkv_sid > 0 else ()
                    # `forward_cache_update_vae` runs `language_model(..., mode="gen")` (reading `moe_gen`
                    # weights in Triton) right after `vae.encode`. Always enter `pager.generation_mode`
                    # so `moe_gen` is 100% resident and pinned even when `use_scoped_vae=False`.
                    gen_cm = (
                        self.pager.generation_mode(
                            prefetch_all=True,
                            pacer=(None if pacer_already_held else self.pacer),
                            rank=self.rank,
                            exclude_session_ids=excl_sids,
                        )
                        if (self.pager is not None and self.vmm is not None)
                        else nullcontext()
                    )
                    with gen_cm, torch.autocast(device_type=dev_type, enabled=ac_enabled, dtype=torch.bfloat16):
                        ctx["past_key_values"] = self.pipeline.bagel.forward_cache_update_vae(
                            self.pipeline.vae, ctx["past_key_values"], **vi
                        )
                    ctx["kv_lens"], ctx["ropes"] = nl, nr
                    if timing_out is not None:
                        timing_out["vae_ms"] = timing_out.get("vae_ms", 0.0) + (
                            (time.perf_counter() - t_vae0) * 1000.0
                        )

                if update_vit:
                    # Prepare ViT tensors strictly outside `autocast(bf16)` for both PIL and GPU-uint8 paths.
                    if is_u8:
                        vti, nlv, nrv = prepare_vit_images_gpu(
                            self.pipeline.bagel,
                            self.pipeline.image_processor,
                            [img_or_u8],
                            ctx["kv_lens"],
                            ctx["ropes"],
                            self.pipeline.new_token_ids,
                        )
                    else:
                        def _vit_tf(v: Any) -> Any:
                            return self.pipeline.image_processor(images=v, return_tensors="pt").pixel_values[0]

                        vti, nlv, nrv = self.pipeline.bagel.prepare_vit_images(
                            curr_kvlens=ctx["kv_lens"],
                            curr_rope=ctx["ropes"],
                            images=[img_or_u8],
                            transforms=_vit_tf,
                            new_token_ids=self.pipeline.new_token_ids,
                        )
                        vti = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in vti.items()}
                        for k in ("packed_indexes", "packed_key_value_indexes", "key_values_lens"):
                            vti.pop(k, None)
                    with torch.autocast(device_type=dev_type, enabled=ac_enabled, dtype=torch.bfloat16):
                        ctx["past_key_values"] = self.pipeline.bagel.forward_cache_update_vit(
                            ctx["past_key_values"], **vti
                        )
                    ctx["kv_lens"], ctx["ropes"] = nlv, nrv
        except BaseException:
            failed = True
            raise
        finally:
            if swapped and self.naive_offloader is not None and (failed or not keep_naive_resident):
                self._update_hwm()
                self.naive_offloader.drop_from_gpu()

    def _apply_dit_turn_images_and_prompt(
        self,
        session: BagelSessionContext,
        cmd: dict[str, Any],
        *,
        can_run_vae: bool,
        prep_dit: bool,
        pacer_already_held: bool = False,
        timing_out: dict[str, float] | None = None,
    ) -> None:
        assert session.main_ctx is not None
        main_ctx = session.main_ctx
        use_cow = bool(self.policy.use_module3_cow_loopback)
        keep_naive = bool(prep_dit and self.naive_offloader is not None)

        if cmd.get("use_prev_image"):
            if cmd.get("prev_image_path"):
                p_prev = str(cmd["prev_image_path"])
                self._update_image_ctx(
                    main_ctx,
                    self._load_pil(p_prev),
                    update_vae=can_run_vae,
                    update_vit=True,
                    keep_naive_resident=keep_naive,
                    pacer_already_held=pacer_already_held,
                    timing_out=timing_out,
                )
                if prep_dit and not can_run_vae:
                    session.ensure_deferral().vae_paths.append(p_prev)
            elif use_cow and session.last_img_u8 is not None:
                self._update_image_ctx(
                    main_ctx,
                    session.last_img_u8,
                    update_vae=can_run_vae,
                    update_vit=True,
                    keep_naive_resident=keep_naive,
                    pacer_already_held=pacer_already_held,
                    timing_out=timing_out,
                )
                if prep_dit and not can_run_vae:
                    session.ensure_deferral().vae_u8 = True
            elif session.last_img_pil is not None:
                # Keep in-memory `prev_pil` directly (no artificial PNG encode/decode penalty for non-CoW baselines).
                prev_pil = session.last_img_pil
                self._update_image_ctx(
                    main_ctx,
                    prev_pil,
                    update_vae=can_run_vae,
                    update_vit=True,
                    keep_naive_resident=keep_naive,
                    pacer_already_held=pacer_already_held,
                    timing_out=timing_out,
                )
                if prep_dit and not can_run_vae:
                    session.ensure_deferral().vae_pil = True
            elif session.last_img_u8 is not None:
                self._update_image_ctx(
                    main_ctx,
                    session.last_img_u8,
                    update_vae=can_run_vae,
                    update_vit=True,
                    keep_naive_resident=keep_naive,
                    pacer_already_held=pacer_already_held,
                    timing_out=timing_out,
                )
                if prep_dit and not can_run_vae:
                    session.ensure_deferral().vae_u8 = True

        explicit_img_paths = list(cmd.get("image_paths") or [])
        for rel_img in explicit_img_paths:
            self._update_image_ctx(
                main_ctx,
                self._load_pil(rel_img),
                update_vae=can_run_vae,
                update_vit=True,
                keep_naive_resident=keep_naive,
                pacer_already_held=pacer_already_held,
                timing_out=timing_out,
            )
            if prep_dit and not can_run_vae:
                session.ensure_deferral().vae_paths.append(str(rel_img))
            elif (not prep_dit) and (not can_run_vae):
                # Q1 `interleaved_co_creation` Turn 0 (`ar_only`) ran ViT-only; remember source path
                # so Turn 1 (`dit_edit`) injects VAE latents without duplicating ViT tokens.
                session.pending_vae_only_paths.append(str(rel_img))

        has_prior_gen_img = bool(
            cmd.get("prev_image_path")
            or session.last_img_u8 is not None
            or session.last_img_pil is not None
        )
        if prep_dit and (not explicit_img_paths) and (not has_prior_gen_img) and session.pending_vae_only_paths:
            pending_vae_paths = list(session.pending_vae_only_paths)
            session.pending_vae_only_paths.clear()
            for vae_rel in pending_vae_paths:
                if can_run_vae:
                    self._update_image_ctx(
                        main_ctx,
                        self._load_pil(vae_rel),
                        update_vae=True,
                        update_vit=False,
                        keep_naive_resident=keep_naive,
                        pacer_already_held=pacer_already_held,
                        timing_out=timing_out,
                    )
                else:
                    session.ensure_deferral().vae_paths.append(vae_rel)

        prompt = str(cmd["prompt"])
        if prep_dit:
            session.drop_cfg_branches()
            has_hist = session.fork_cfg_text(self._aux_cache_factory(), use_cow=use_cow) > 0
            self._update_text_ctx(main_ctx, prompt)
            session.rebind_cow_branches(use_cow=use_cow)

            if (not has_hist) and session.main_seq_len() > 0:
                session.fork_cfg_img_from_main(self._aux_cache_factory(), use_cow=use_cow)
            else:
                cfg_img_ctx = self._new_aux_ctx()
                self._update_text_ctx(cfg_img_ctx, prompt)
                session.cfg_img_ctx = cfg_img_ctx
        else:
            self._update_text_ctx(main_ctx, prompt)

    def _prepare_turn_inner(self, ctx_id: str, cmd: dict[str, Any]) -> int:
        existing_session = self._ctx_store.get(ctx_id)
        has_images = bool(
            cmd.get("use_prev_image")
            or cmd.get("image_paths")
            or (existing_session is not None and bool(existing_session.pending_vae_only_paths))
        )
        est_delta = (768 * MIB) if (has_images or cmd.get("prepare_for_dit")) else (128 * MIB)
        allow_auto_swap = self._auto_idle_kv_swap_enabled()

        session = existing_session
        if session is None:
            if allow_auto_swap and self.available_kv_bytes() < est_delta + 512 * MIB:
                self.evict_idle_kv_to_host(est_delta + 512 * MIB, exclude_ctx_ids=(ctx_id,))
            session = BagelSessionContext(self._new_main_ctx())
            self._ctx_store[ctx_id] = session
            if self._active_ctx_pins.get(ctx_id, 0) > 0:
                self._sync_ctx_vmm_activity(ctx_id, active=True)
        else:
            session = self._ensure_ctx_resident(ctx_id, extra_needed_bytes=est_delta)

        has_images = bool(cmd.get("use_prev_image") or cmd.get("image_paths") or session.pending_vae_only_paths)
        prep_dit = bool(cmd.get("prepare_for_dit", False))
        is_ar_cmd = bool(cmd.get("max_ar_tokens")) or (cmd.get("op") in ("EXEC_AR", "EXEC_AR_PREEMPT"))
        is_fission_und = self.policy.routing_mode in ("static_nccl_fission", "twostage_shm") and self.role == "und"
        pager_partial = self.pager is not None and self.vmm is not None and self.resident_alpha() < 0.999

        if prep_dit and (not is_ar_cmd) and has_images and pager_partial and (not is_fission_und):
            img_paths_list = [str(x) for x in cmd.get("image_paths", [])]
            session.ensure_deferral().turn_cmd = {
                "use_prev_image": bool(cmd.get("use_prev_image")),
                "prev_image_path": cmd.get("prev_image_path"),
                "image_paths": img_paths_list,
                "prompt": str(cmd["prompt"]),
            }
            n_pending_imgs = (
                (1 if cmd.get("use_prev_image") else 0)
                + len(img_paths_list)
                + len(session.pending_vae_only_paths)
            )
            self._update_hwm()
            return session.logical_bytes() + max(1, n_pending_imgs) * (192 * MIB)

        can_run_vae_here = prep_dit and (not is_fission_und) and (not pager_partial)
        timing_out: dict[str, float] = {"h2d_weight_ms": 0.0, "vae_ms": 0.0}
        try:
            with session.atomic_kv_update():
                self._apply_dit_turn_images_and_prompt(
                    session,
                    cmd,
                    can_run_vae=can_run_vae_here,
                    prep_dit=prep_dit,
                    timing_out=timing_out,
                )
        except BaseException:
            if prep_dit and self.naive_offloader is not None:
                self._release_naive_hold(ctx_id)
            raise

        if prep_dit:
            if timing_out["h2d_weight_ms"] > 0.0:
                self._pending_h2d_weight_ms[ctx_id] = (
                    self._pending_h2d_weight_ms.get(ctx_id, 0.0) + timing_out["h2d_weight_ms"]
                )
            if timing_out["vae_ms"] > 0.0:
                self._pending_vae_encode_ms[ctx_id] = (
                    self._pending_vae_encode_ms.get(ctx_id, 0.0) + timing_out["vae_ms"]
                )
            if self.naive_offloader is not None and bool(getattr(self.naive_offloader, "_resident_on_gpu", False)):
                self._naive_offload_held_ctx_ids.add(ctx_id)

        self._update_hwm()
        return session.logical_bytes()

    def prepare_turn(self, ctx_id: str, cmd: dict[str, Any]) -> int:
        had_entry = ctx_id in self._ctx_store
        with self._pinned_ctx(ctx_id):
            try:
                return self._prepare_turn_inner(ctx_id, cmd)
            except BaseException as exc:
                if not had_entry:
                    partial = self._ctx_store.pop(ctx_id, None)
                    if partial is not None:
                        partial.close()
                if self._auto_idle_kv_swap_enabled() and is_oom_exception(exc):
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
        dev_type = getattr(self.device, "type", "cuda")
        ac_enabled = dev_type != "cpu"

        with self._pinned_ctx(*cids):
            for cmd in batch_cmds:
                cid = str(cmd["ctx_id"])
                t_prep0 = time.perf_counter()
                enq_s = self.normalized_enqueue_time_s(cmd, default_s=t_prep0)
                if not cmd.get("_skip_prepare", False):
                    self.prepare_turn(cid, cmd)
                else:
                    self._ensure_ctx_resident(cid, extra_needed_bytes=64 * MIB)
                session = self._ctx_store[cid]
                assert session.main_ctx is not None
                ctx = session.main_ctx
                start_inp = self.pipeline.bagel.prepare_start_tokens(
                    ctx["kv_lens"], ctx["ropes"], self.pipeline.new_token_ids
                )
                start_tok = start_inp["packed_start_tokens"].to(self.device)
                start_tok_id = int(start_inp["packed_start_tokens"].reshape(-1)[0].item())
                states.append(
                    {
                        "ctx_id": cid,
                        "session": session,
                        "ctx": ctx,
                        "curr_tokens": start_tok,
                        "curr_tok_id": start_tok_id,
                        "pos_ids": start_inp["packed_query_position_ids"].to(self.device),
                        "max_tokens": max(1, int(cmd.get("max_ar_tokens", 32))),
                        "gen_ids": [],
                        "stamps": [],
                        "t0": enq_s,
                        "done": False,
                    }
                )

            for st in states:
                pkv_init = st["ctx"]["past_key_values"]
                if callable(getattr(pkv_init, "set_fast_ar_mode", None)):
                    pkv_init.set_fast_ar_mode(True, reserve_tokens=st["max_tokens"] + 4)

            try:
                with torch.inference_mode(), torch.autocast(
                    device_type=dev_type, enabled=ac_enabled, dtype=torch.bfloat16
                ):
                    while True:
                        active = [st for st in states if not st["done"]]
                        if not active:
                            break
                        prev_ar_lens = [st["session"].main_seq_len() for st in active]
                        try:
                            next_tokens, updated_caches = self.interceptor.forward_ar_step_batch(
                                [st["curr_tokens"] for st in active],
                                [st["pos_ids"] for st in active],
                                [st["ctx"]["past_key_values"] for st in active],
                            )
                        except BaseException as exc:
                            for st, plen in zip(active, prev_ar_lens):
                                st["session"].rollback_main_kv(plen)
                            if (
                                self._auto_idle_kv_swap_enabled()
                                and is_oom_exception(exc)
                                and self.evict_idle_kv_to_host(1 << 50, exclude_ctx_ids=cids) > 0
                            ):
                                for st in active:
                                    pkv_re = st["ctx"]["past_key_values"]
                                    if callable(getattr(pkv_re, "set_fast_ar_mode", None)):
                                        pkv_re.set_fast_ar_mode(True, reserve_tokens=st["max_tokens"] + 4)
                                try:
                                    next_tokens, updated_caches = self.interceptor.forward_ar_step_batch(
                                        [st["curr_tokens"] for st in active],
                                        [st["pos_ids"] for st in active],
                                        [st["ctx"]["past_key_values"] for st in active],
                                    )
                                except BaseException:
                                    for st, plen in zip(active, prev_ar_lens):
                                        st["session"].rollback_main_kv(plen)
                                    raise
                            else:
                                raise

                        next_ids = [int(x) for x in next_tokens.detach().cpu().tolist()]
                        now_s = time.perf_counter()
                        for idx, st in enumerate(active):
                            nxt_id = next_ids[idx]
                            st["gen_ids"].append(st["curr_tok_id"])
                            st["ctx"]["past_key_values"] = updated_caches[idx]
                            st["curr_tokens"] = next_tokens[idx : idx + 1]
                            st["curr_tok_id"] = nxt_id
                            st["pos_ids"] = st["pos_ids"] + 1
                            st["stamps"].append(now_s)
                            reached = len(st["gen_ids"]) >= st["max_tokens"]
                            hit_eos = end_id is not None and nxt_id == int(end_id)
                            if reached or hit_eos:
                                st["done"] = True
            finally:
                for st in states:
                    pkv_fin = st["ctx"].get("past_key_values")
                    if callable(getattr(pkv_fin, "set_fast_ar_mode", None)):
                        pkv_fin.set_fast_ar_mode(False)

            self._update_hwm()
            alpha_now = self.resident_alpha()
            free_kv_now = self.available_kv_bytes()
            outputs: list[dict[str, Any]] = []
            for st in states:
                ctx = st["ctx"]
                session: BagelSessionContext = st["session"]
                gen_ids = st["gen_ids"]
                stamps = st["stamps"]
                n_tok = len(gen_ids)
                ctx["kv_lens"] = [kl + n_tok for kl in ctx["kv_lens"]]
                ctx["ropes"] = [r + n_tok for r in ctx["ropes"]]
                session.rebind_cow_branches(use_cow=bool(self.policy.use_module3_cow_loopback))
                ttft_ms = max(0.0, (stamps[0] - st["t0"]) * 1000.0) if stamps else 0.0
                tpots_ms = [(stamps[i] - stamps[i - 1]) * 1000.0 for i in range(1, len(stamps))]
                decoded = self.pipeline.tokenizer.decode(gen_ids) if gen_ids else ""
                outputs.append(
                    {
                        "text": decoded.split("<|im_end|>")[0].split("<|im_start|>")[-1],
                        "tokens": n_tok,
                        "ttft_ms": round(ttft_ms, 2),
                        "tpots_ms": [round(x, 2) for x in tpots_ms],
                        "kv_bytes": session.logical_bytes(),
                        "resident_alpha": alpha_now,
                        "kv_free_bytes": free_kv_now,
                        "batched_width": len(states),
                    }
                )
            return outputs

    def _apply_pending_deferral(
        self,
        session: BagelSessionContext,
        *,
        timing_out: dict[str, float] | None = None,
    ) -> None:
        """Execute deferred `[VAE, ViT, Prompt]` or VAE-only updates atomically before `EXEC_DIT`."""
        deferral = session.pending_deferral
        if deferral is None or deferral.is_empty:
            return
        keep_naive = self.naive_offloader is not None
        early_pacer_acquired = bool(
            self.pacer is not None
            and self.pager is not None
            and self.vmm is not None
            and self.vmm.expert_resident_count(self.pager.expert_id) < self.pager.num_layers
        )
        early_pacer_cm = self.pacer.acquire_process_slot(self.rank) if early_pacer_acquired else nullcontext()
        with early_pacer_cm, session.atomic_kv_update():
            session.pending_deferral = None
            if deferral.turn_cmd is not None:
                self._apply_dit_turn_images_and_prompt(
                    session,
                    deferral.turn_cmd,
                    can_run_vae=True,
                    prep_dit=True,
                    pacer_already_held=early_pacer_acquired,
                    timing_out=timing_out,
                )
            assert session.main_ctx is not None
            main_ctx = session.main_ctx
            cfg_text_ctx = session.cfg_text_ctx
            if deferral.vae_u8 and session.last_img_u8 is not None:
                self._update_image_ctx(
                    main_ctx,
                    session.last_img_u8,
                    update_vae=True,
                    update_vit=False,
                    keep_naive_resident=keep_naive,
                    pacer_already_held=early_pacer_acquired,
                    timing_out=timing_out,
                )
                if cfg_text_ctx is not None:
                    self._update_image_ctx(
                        cfg_text_ctx,
                        session.last_img_u8,
                        update_vae=True,
                        update_vit=False,
                        keep_naive_resident=keep_naive,
                        pacer_already_held=early_pacer_acquired,
                        timing_out=timing_out,
                    )
            if deferral.vae_pil and session.last_img_pil is not None:
                self._update_image_ctx(
                    main_ctx,
                    session.last_img_pil,
                    update_vae=True,
                    update_vit=False,
                    keep_naive_resident=keep_naive,
                    pacer_already_held=early_pacer_acquired,
                    timing_out=timing_out,
                )
                if cfg_text_ctx is not None:
                    self._update_image_ctx(
                        cfg_text_ctx,
                        session.last_img_pil,
                        update_vae=True,
                        update_vit=False,
                        keep_naive_resident=keep_naive,
                        pacer_already_held=early_pacer_acquired,
                        timing_out=timing_out,
                    )
            for pend_img in deferral.vae_paths:
                pil_v = self._load_pil(pend_img)
                self._update_image_ctx(
                    main_ctx,
                    pil_v,
                    update_vae=True,
                    update_vit=False,
                    keep_naive_resident=keep_naive,
                    pacer_already_held=early_pacer_acquired,
                    timing_out=timing_out,
                )
                if cfg_text_ctx is not None:
                    self._update_image_ctx(
                        cfg_text_ctx,
                        pil_v,
                        update_vae=True,
                        update_vit=False,
                        keep_naive_resident=keep_naive,
                        pacer_already_held=early_pacer_acquired,
                        timing_out=timing_out,
                    )

    def generate_dit(
        self,
        ctx_id: str,
        cmd: dict[str, Any],
        poll_preempt_cmds: Callable[[], list[dict[str, Any]]] | None = None,
        emit_preempt_ack: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        from PIL import Image
        import torch

        try:
            from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        except ImportError:
            class OmniDiffusionSamplingParams:  # type: ignore[no-redef]
                def __init__(self, **kwargs: Any) -> None:
                    for k, v in kwargs.items():
                        setattr(self, k, v)

        dev_type = getattr(self.device, "type", "cuda")
        is_cuda = torch.cuda.is_available() and dev_type == "cuda"
        height = int(cmd.get("height", 512))
        width = int(cmd.get("width", 512))
        steps = int(cmd.get("dit_steps", 20))
        seed = int(cmd.get("seed", 52))
        prompt = str(cmd.get("prompt", ""))
        want_pil = bool(cmd.get("return_pil", False))
        save_tmp_path = cmd.get("save_tmp_image_path")
        use_idle_kv_swap = bool(getattr(self.policy, "use_idle_kv_swap", False))

        with self._pinned_ctx(ctx_id):
            h2d_weight_ms = float(self._pending_h2d_weight_ms.pop(ctx_id, 0.0))
            vae_ms = float(self._pending_vae_encode_ms.pop(ctx_id, 0.0))
            vmm_st0 = self.vmm.stats_dict() if (self.pager is not None and self.vmm is not None) else None

            session = self._ensure_ctx_resident(ctx_id)
            has_pending = session.has_pending_deferral
            if self.policy.use_scoped_vae:
                # With Scoped VAE active, FLUX VAE decode activations run inside TriPoolVMM Arena-2
                # rather than PyTorch driver memory; reserve 5.0/3.0 GiB so 1024px DiT + concurrent
                # preempted AR never fragment CUDACachingAllocator on 40 GiB A100.
                est_dit_vae_bytes = int(5120 * MIB) if (height >= 1024 or width >= 1024) else int(3072 * MIB)
            else:
                est_dit_vae_bytes = int(6144 * MIB) if (height >= 1024 or width >= 1024) else int(3584 * MIB)
            if self.pager is not None and self.vmm is not None:
                extra_dit_kv = (256 * MIB) if has_pending else (64 * MIB)
                # `for_dit=True` already reserves full DiT expert bytes in TriPoolVMM's accounting;
                # only reclaim idle KV when net DiT KV headroom is actually below `extra_dit_kv`.
                if self.available_kv_bytes(for_dit=True) < extra_dit_kv:
                    self.evict_idle_kv_to_host(extra_dit_kv, exclude_ctx_ids=(ctx_id,), for_dit=True)
                self._ensure_cuda_driver_headroom_for_dit(est_dit_vae_bytes, exclude_ctx_ids=(ctx_id,))
            elif use_idle_kv_swap:
                # Fair symmetric idle-KV protection for `monolithic_kv_swap` & `prism_tetriserve_coloc`:
                # proactively reclaim idle KV if free HBM is below estimated DiT+VAE spike (`4.5 GiB` / `7.5 GiB`).
                est_dit_vae_bytes_base = int(7.5 * GIB) if (height >= 1024 or width >= 1024) else int(4.5 * GIB)
                if is_cuda:
                    free_drv, _ = torch.cuda.mem_get_info(self.device)
                    unused_pt = max(
                        0,
                        int(torch.cuda.memory_reserved(self.device)) - int(torch.cuda.memory_allocated(self.device)),
                    )
                    free_hbm_est = int(free_drv) + unused_pt
                else:
                    free_hbm_est = self.available_kv_bytes(for_dit=False)
                if free_hbm_est < est_dit_vae_bytes_base:
                    deficit = est_dit_vae_bytes_base - free_hbm_est
                    target_avail = max(est_dit_vae_bytes_base, self.available_kv_bytes(for_dit=False) + deficit)
                    reclaimed_b = self.evict_idle_kv_to_host(target_avail, exclude_ctx_ids=(ctx_id,), for_dit=False)
                    if self.vmm is not None and reclaimed_b > 0:
                        flush_fn = getattr(self.vmm, "flush_kv_swaps", None)
                        if callable(flush_fn):
                            flush_fn()
                        self._release_vmm_overflow_to_driver(deficit)
                    if reclaimed_b > 0 and is_cuda:
                        self._update_hwm()
                        torch.cuda.empty_cache()

            if has_pending:
                defer_timing: dict[str, float] = {"h2d_weight_ms": 0.0, "vae_ms": 0.0}
                self._apply_pending_deferral(session, timing_out=defer_timing)
                h2d_weight_ms += defer_timing["h2d_weight_ms"]
                vae_ms += defer_timing["vae_ms"]

            assert session.main_ctx is not None
            main_ctx = session.main_ctx
            cfg_text_ctx = session.cfg_text_ctx
            cfg_img_ctx = session.cfg_img_ctx

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
            if cfg_text_ctx is not None and _cache_seq_len(cfg_text_ctx.get("past_key_values"), 0) > 0:
                sampling.cfg_text_past_key_values = cfg_text_ctx["past_key_values"]
                sampling.cfg_text_kv_metadata = {"ropes": cfg_text_ctx["ropes"], "image_shape": [height, width]}
            if cfg_img_ctx is not None and _cache_seq_len(cfg_img_ctx.get("past_key_values"), 0) > 0:
                sampling.cfg_img_past_key_values = cfg_img_ctx["past_key_values"]
                sampling.cfg_img_kv_metadata = {"ropes": cfg_img_ctx["ropes"], "image_shape": [height, width]}

            if self.naive_offloader is not None:
                t_sw0 = time.perf_counter()
                if self.naive_offloader.swap_in_sync():
                    h2d_weight_ms += (time.perf_counter() - t_sw0) * 1000.0

            def _admit_preempt_cmd(ar_msg: dict[str, Any], now_s: float) -> tuple[ArDecodeTask, tuple[str, str, BagelSessionContext]]:
                ar_cid = str(ar_msg["ctx_id"])
                self._pin_ctx_id(ar_cid)
                try:
                    self.prepare_turn(ar_cid, ar_msg)
                    ar_session = self._ctx_store[ar_cid]
                    assert ar_session.main_ctx is not None
                    ar_ctx = ar_session.main_ctx
                    st = self.pipeline.bagel.prepare_start_tokens(
                        ar_ctx["kv_lens"], ar_ctx["ropes"], self.pipeline.new_token_ids
                    )
                    start_tok_gpu = st["packed_start_tokens"].to(self.device)
                    start_tok_int = int(st["packed_start_tokens"].reshape(-1)[0].item())
                    max_ar_toks = max(1, int(ar_msg.get("max_ar_tokens", 32)))
                    pkv_ar = ar_ctx["past_key_values"]
                    if callable(getattr(pkv_ar, "set_fast_ar_mode", None)):
                        pkv_ar.set_fast_ar_mode(True, reserve_tokens=max_ar_toks + 4)
                    arr_s = self.normalized_enqueue_time_s(ar_msg, default_s=now_s)
                    task = ArDecodeTask(
                        req_id=str(ar_msg["cmd_id"]),
                        kv_cache=pkv_ar,
                        curr_tokens=start_tok_gpu,
                        pos_ids=st["packed_query_position_ids"].to(self.device),
                        max_tokens=max_ar_toks,
                        end_token_id=self.pipeline.new_token_ids["eos_token_id"],
                        tpot_slo_ms=float(ar_msg.get("tpot_slo_ms", self.policy.ar_tpot_slo_ms)),
                        arrival_time_s=arr_s,
                        _curr_tok_int=start_tok_int,
                    )
                    task.last_step_time_s = task.arrival_time_s
                    return task, (str(ar_msg["cmd_id"]), ar_cid, ar_session)
                except BaseException:
                    self._unpin_ctx_id(ar_cid)
                    raise

            def _complete_preempt_task(task: ArDecodeTask, handle: tuple[str, str, BagelSessionContext]) -> None:
                c_id, ar_ctx_id, ar_session = handle
                self._unpin_ctx_id(ar_ctx_id)
                assert ar_session.main_ctx is not None
                ar_ctx = ar_session.main_ctx
                ar_ctx["past_key_values"] = task.kv_cache
                n_tok = len(task.generated_token_ids)
                ar_ctx["kv_lens"] = [kl + n_tok for kl in ar_ctx["kv_lens"]]
                ar_ctx["ropes"] = [r + n_tok for r in ar_ctx["ropes"]]
                ar_session.rebind_cow_branches(use_cow=bool(self.policy.use_module3_cow_loopback))
                if emit_preempt_ack is not None:
                    decoded = self.pipeline.tokenizer.decode(task.generated_token_ids) if task.generated_token_ids else ""
                    emit_preempt_ack(
                        {
                            "cmd_id": c_id,
                            "rank": self.rank,
                            "status": "ok",
                            "ctx_id": ar_ctx_id,
                            "text": decoded.split("<|im_end|>")[0].split("<|im_start|>")[-1],
                            "tokens": n_tok,
                            "ttft_ms": round(task.ttft_ms(), 2),
                            "tpots_ms": [round(x, 2) for x in task.inter_token_latencies_ms()],
                            "kv_bytes": ar_session.logical_bytes(),
                            "resident_alpha": self.resident_alpha(),
                            "kv_free_bytes": self.available_kv_bytes(),
                            "preempted_mid_dit": True,
                        }
                    )

            def _abort_preempt_handle(handle: tuple[str, str, BagelSessionContext], exc: BaseException | None) -> None:
                c_id, ar_cid, _ = handle
                self._unpin_ctx_id(ar_cid)
                if emit_preempt_ack is None:
                    return
                if exc is not None and is_oom_exception(exc):
                    if is_cuda:
                        self._update_hwm()
                        torch.cuda.empty_cache()
                    emit_preempt_ack({"cmd_id": c_id, "rank": self.rank, "status": "oom", "error": str(exc)[:200]})
                elif exc is not None:
                    emit_preempt_ack(
                        {
                            "cmd_id": c_id,
                            "rank": self.rank,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                else:
                    emit_preempt_ack({"cmd_id": c_id, "rank": self.rank, "status": "oom", "error": "aborted mid-dit"})

            def _abort_deferred_cmd(d_msg: dict[str, Any]) -> None:
                if emit_preempt_ack is not None:
                    emit_preempt_ack(
                        {
                            "cmd_id": str(d_msg["cmd_id"]),
                            "rank": self.rank,
                            "status": "oom",
                            "error": "aborted deferred mid-dit",
                        }
                    )

            def _abort_preempt_cmd(d_msg: dict[str, Any], exc: BaseException | None) -> None:
                if emit_preempt_ack is None:
                    return
                if exc is not None and is_oom_exception(exc):
                    if is_cuda:
                        self._update_hwm()
                        torch.cuda.empty_cache()
                    emit_preempt_ack(
                        {
                            "cmd_id": str(d_msg["cmd_id"]),
                            "rank": self.rank,
                            "status": "oom",
                            "error": str(exc)[:200],
                        }
                    )
                elif exc is not None:
                    emit_preempt_ack(
                        {
                            "cmd_id": str(d_msg["cmd_id"]),
                            "rank": self.rank,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                else:
                    _abort_deferred_cmd(d_msg)

            def _is_heavy_preempt_cmd(ar_msg: dict[str, Any]) -> bool:
                if ar_msg.get("image_paths") or ar_msg.get("use_prev_image"):
                    return True
                ar_cid = str(ar_msg.get("ctx_id", ""))
                ar_sess = self._ctx_store.get(ar_cid)
                return bool(ar_sess is not None and ar_sess.is_swapped)

            def _reclaim_for_preempt(_tasks: list[ArDecodeTask]) -> int:
                # Reclaim ~2 GiB first so a single transient activation spike doesn't dump every warm session.
                target_b = self.available_kv_bytes(for_dit=False) + 2048 * MIB
                freed_b = self.evict_idle_kv_to_host(target_b, exclude_ctx_ids=(ctx_id,))
                if freed_b == 0:
                    freed_b = self.evict_idle_kv_to_host(1 << 50, exclude_ctx_ids=(ctx_id,))
                if self.vmm is not None:
                    flush_fn = getattr(self.vmm, "flush_kv_swaps", None)
                    if callable(flush_fn):
                        flush_fn()
                    freed_b += self._release_vmm_overflow_to_driver(2048 * MIB)
                return freed_b

            hw_key = (height, width)
            has_prior_hw = hw_key in self._calibrated_layer_ms_by_hw

            def _make_preempt_session() -> PreemptionSession:
                return PreemptionSession(
                    self.sched,
                    poll_cmds=poll_preempt_cmds,
                    admit_cmd=_admit_preempt_cmd if (poll_preempt_cmds is not None) else None,
                    on_complete=_complete_preempt_task if (emit_preempt_ack is not None) else None,
                    on_abort=_abort_preempt_handle,
                    on_abort_deferred=_abort_deferred_cmd,
                    on_abort_cmd=_abort_preempt_cmd,
                    on_oom_reclaim=_reclaim_for_preempt if self._auto_idle_kv_swap_enabled() else None,
                    is_heavy_cmd=_is_heavy_preempt_cmd,
                    drain_context_factory=lambda: torch.autocast(
                        device_type=dev_type, enabled=is_cuda, dtype=torch.bfloat16
                    ),
                    init_layer_ms=self._calibrated_layer_ms_by_hw.get(hw_key, 37.0 if height >= 1024 else 10.5),
                    preserve_layer_calibration=has_prior_hw,
                    init_ar_step_ms=self._calibrated_ar_step_ms,
                )

            if is_cuda:
                torch.cuda.current_stream(self.device).synchronize()
            t0 = time.perf_counter()
            latents = None
            img_shape = (height, width)
            excl_sids = (session.main_session_id,) if session.main_session_id > 0 else ()
            max_dit_oom_retries = 2 if (use_idle_kv_swap or (self.pager is not None and self.vmm is not None)) else 0
            total_yield_count = 0
            preempt_yield_ms = 0.0
            vae_decode_ev_pair: tuple[Any, Any] | None = None
            vae_decode_wall_ms = 0.0

            try:
                with torch.inference_mode():
                    # Phase 1: DiT denoising (with bounded OOM -> evict_idle_kv_to_host -> retry for idle-KV swap policies)
                    for dit_attempt in range(max_dit_oom_retries + 1):
                        preempt_session = _make_preempt_session()
                        try:
                            with preempt_session:
                                prep = self.pipeline._forward_single(
                                    {"prompt": prompt, "modalities": ["image"]},
                                    sampling,
                                    prepare_only=True,
                                )
                                gp = prep["gen_params"]
                                img_shape = prep["image_shape"]

                                if (
                                    (int(img_shape[0]) >= 1024 or int(img_shape[1]) >= 1024)
                                    and is_cuda
                                    and (
                                        int(torch.cuda.memory_reserved(self.device))
                                        - int(torch.cuda.memory_allocated(self.device))
                                    )
                                    >= 768 * MIB
                                ):
                                    self._update_hwm()
                                    torch.cuda.empty_cache()

                                with self.interceptor.generation_session(
                                    pager=self.pager,
                                    use_cfg_patch=self.policy.use_module3_cow_loopback,
                                    scheduler=self.sched,
                                    pacer=self.pacer,
                                    rank=self.rank,
                                    prefetch_all=True,
                                    reset_cfg_stats=True,
                                    exclude_session_ids=excl_sids,
                                ):
                                    with torch.autocast(device_type=dev_type, enabled=is_cuda, dtype=torch.bfloat16):
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

                                cal_ms = preempt_session.calibrated_layer_ms
                                if cal_ms is not None:
                                    self._calibrated_layer_ms_by_hw[hw_key] = cal_ms

                                # Drain any in-flight preempted AR tasks before FLUX `vae.decode` so VAE decode never stalls TPOT.
                                if emit_preempt_ack is not None:
                                    preempt_session.drain(stop_polling_after=True)
                                total_yield_count += preempt_session.yield_count
                                if self.sched is not None:
                                    preempt_yield_ms += float(self.sched.ar_compute_total_ms)
                            break
                        except BaseException as exc:
                            if dit_attempt < max_dit_oom_retries and is_oom_exception(exc):
                                self._update_hwm()
                                free_before = int(torch.cuda.mem_get_info(self.device)[0]) if is_cuda else 0
                                target_reclaim = (
                                    (self.available_kv_bytes(for_dit=False) + 2048 * MIB)
                                    if dit_attempt == 0
                                    else (1 << 50)
                                )
                                freed = self.evict_idle_kv_to_host(target_reclaim, exclude_ctx_ids=(ctx_id,), for_dit=False)
                                if freed == 0 and dit_attempt == 0:
                                    freed = self.evict_idle_kv_to_host(1 << 50, exclude_ctx_ids=(ctx_id,), for_dit=False)
                                vmm_trimmed = 0
                                if self.vmm is not None:
                                    flush_fn = getattr(self.vmm, "flush_kv_swaps", None)
                                    if callable(flush_fn):
                                        flush_fn()
                                    vmm_trimmed = self._release_vmm_overflow_to_driver(
                                        2048 * MIB if dit_attempt == 0 else 0
                                    )
                                if is_cuda:
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                free_after = int(torch.cuda.mem_get_info(self.device)[0]) if is_cuda else 0
                                if freed > 0 or vmm_trimmed > 0 or (free_after - free_before) >= 64 * MIB:
                                    continue
                            raise

                    # Drop per-turn CFG branch caches immediately after denoising completes.
                    session.drop_cfg_branches()
                    if self.vmm is not None and is_cuda and (int(img_shape[0]) >= 1024 or int(img_shape[1]) >= 1024):
                        if self.policy.use_scoped_vae:
                            free_drv, _ = torch.cuda.mem_get_info(self.device)
                            unused_pt = max(
                                0,
                                int(torch.cuda.memory_reserved(self.device))
                                - int(torch.cuda.memory_allocated(self.device)),
                            )
                            if int(free_drv) < 1536 * MIB and unused_pt >= 512 * MIB:
                                self._update_hwm()
                                torch.cuda.empty_cache()
                        else:
                            self._ensure_cuda_driver_headroom_for_dit(5120 * MIB, exclude_ctx_ids=(ctx_id,))

                    # Phase 2: FLUX VAE decode outside `generation_session` (bounded OOM retry reuses `latents`!).
                    for vae_attempt in range(max_dit_oom_retries + 1):
                        try:
                            ev0 = torch.cuda.Event(enable_timing=True) if is_cuda else None
                            ev1 = torch.cuda.Event(enable_timing=True) if is_cuda else None
                            t_vd0 = time.perf_counter()
                            if ev0 is not None:
                                ev0.record(torch.cuda.current_stream(self.device))
                            if self.policy.use_module3_cow_loopback:
                                with torch.autocast(device_type=dev_type, enabled=is_cuda, dtype=torch.bfloat16):
                                    img_u8 = decode_latent_to_u8_gpu(
                                        self.pipeline.bagel, self.pipeline.vae, latents[0], img_shape
                                    )
                                if ev1 is not None:
                                    ev1.record(torch.cuda.current_stream(self.device))
                                vae_decode_wall_ms = (time.perf_counter() - t_vd0) * 1000.0
                                vae_decode_ev_pair = (ev0, ev1) if (ev0 is not None and ev1 is not None) else None
                                session.last_img_u8 = img_u8
                                if want_pil or save_tmp_path:
                                    pil_img = Image.fromarray(img_u8.permute(1, 2, 0).contiguous().cpu().numpy())
                                    session.last_img_pil = pil_img
                                    if save_tmp_path:
                                        Path(save_tmp_path).parent.mkdir(parents=True, exist_ok=True)
                                        pil_img.save(save_tmp_path, format="PNG")
                                else:
                                    session.last_img_pil = None
                            else:
                                with torch.autocast(device_type=dev_type, enabled=is_cuda, dtype=torch.bfloat16):
                                    pil_img = self.pipeline._decode_image_from_latent(
                                        self.pipeline.bagel, self.pipeline.vae, latents[0], img_shape
                                    )
                                if ev1 is not None:
                                    ev1.record(torch.cuda.current_stream(self.device))
                                vae_decode_wall_ms = (time.perf_counter() - t_vd0) * 1000.0
                                vae_decode_ev_pair = (ev0, ev1) if (ev0 is not None and ev1 is not None) else None
                                if isinstance(pil_img, (list, tuple)) and pil_img:
                                    pil_img = pil_img[0]
                                session.last_img_pil = pil_img
                                session.last_img_u8 = None
                                if save_tmp_path:
                                    Path(save_tmp_path).parent.mkdir(parents=True, exist_ok=True)
                                    pil_img.save(save_tmp_path, format="PNG")
                            break
                        except BaseException as exc:
                            if vae_attempt < max_dit_oom_retries and is_oom_exception(exc):
                                self._update_hwm()
                                free_before = int(torch.cuda.mem_get_info(self.device)[0]) if is_cuda else 0
                                freed = self.evict_idle_kv_to_host(1 << 50, exclude_ctx_ids=(ctx_id,), for_dit=False)
                                vmm_trimmed = 0
                                if self.vmm is not None:
                                    flush_fn = getattr(self.vmm, "flush_kv_swaps", None)
                                    if callable(flush_fn):
                                        flush_fn()
                                    vmm_trimmed = self._release_vmm_overflow_to_driver()
                                if is_cuda:
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                free_after = int(torch.cuda.mem_get_info(self.device)[0]) if is_cuda else 0
                                if freed > 0 or vmm_trimmed > 0 or (free_after - free_before) >= 64 * MIB:
                                    continue
                            raise
            finally:
                if is_cuda:
                    try:
                        torch.cuda.current_stream(self.device).synchronize()
                    except Exception:
                        pass
                # Always record peak HBM BEFORE dropping offloaded expert weights / emptying cache!
                self._update_hwm()
                self._release_naive_hold(ctx_id)

            # After Scoped VAE `scope_end()` returns Arena-2 super handles to `free_super_handles_`,
            # stream back any evicted tail expert layers that fit directly in those free handles
            # (async on `h2d_stream_`, zero KV eviction, zero compute-stream stall) so `resident_alpha`
            # recovers during think-time/AR instead of causing H2D pacer stalls on the next DiT turn.
            if self.pager is not None and self.vmm is not None and self.policy.use_scoped_vae:
                restore_fn = getattr(self.pager, "restore_free_super_layers_async", None)
                if callable(restore_fn):
                    restore_fn()

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if vae_decode_ev_pair is not None:
                try:
                    vae_ms += float(vae_decode_ev_pair[0].elapsed_time(vae_decode_ev_pair[1]))
                except Exception:
                    vae_ms += vae_decode_wall_ms
            else:
                vae_ms += vae_decode_wall_ms

            if vmm_st0 is not None and self.vmm is not None:
                vmm_st1 = self.vmm.stats_dict()
                d_map_us = max(0.0, float(vmm_st1.get("expert_map_us", 0.0)) - float(vmm_st0.get("expert_map_us", 0.0)))
                d_sub_us = max(
                    0.0,
                    float(vmm_st1.get("expert_h2d_submit_us", 0.0)) - float(vmm_st0.get("expert_h2d_submit_us", 0.0)),
                )
                d_wait_us = max(
                    0.0,
                    float(vmm_st1.get("expert_wait_us", 0.0)) - float(vmm_st0.get("expert_wait_us", 0.0)),
                )
                h2d_weight_ms += (d_map_us + d_sub_us + d_wait_us) / 1000.0

            self._update_hwm()
            return {
                "status": "ok",
                "ctx_id": ctx_id,
                "elapsed_ms": round(elapsed_ms, 2),
                "h2d_weight_ms": round(h2d_weight_ms, 2),
                "vae_ms": round(vae_ms, 2),
                "preempt_yield_ms": round(preempt_yield_ms, 2),
                "kv_bytes": session.logical_bytes(),
                "resident_alpha": self.resident_alpha(),
                "kv_free_bytes": self.available_kv_bytes(),
                "yield_count": total_yield_count,
            }

    def pack_state(self, ctx_id: str, include_cfg_dup: bool = False) -> PackedTransferState:
        with self._pinned_ctx(ctx_id):
            session = self._ensure_ctx_resident(ctx_id)
            try:
                return pack_bagel_multi_context(session, include_cfg_text_dup=include_cfg_dup, device=self.device)
            except BaseException as exc:
                if is_oom_exception(exc):
                    est_need = max(1024 * MIB, int(session.logical_bytes()) + 512 * MIB)
                    self.reclaim_driver_headroom(est_need, exclude_ctx_ids=(ctx_id,))
                    return pack_bagel_multi_context(session, include_cfg_text_dup=include_cfg_dup, device=self.device)
                raise

    def unpack_state(self, new_ctx_id: str, meta: dict[str, Any], raw_payload: Any) -> int:
        self.release_context(new_ctx_id)
        allow_auto_swap = self._auto_idle_kv_swap_enabled()
        with self._pinned_ctx(new_ctx_id):
            est_b = int(meta.get("logical_bytes", meta.get("total_bytes", 0)))
            if allow_auto_swap and est_b > 0 and self.available_kv_bytes() < est_b + 512 * MIB:
                self.evict_idle_kv_to_host(est_b + 512 * MIB, exclude_ctx_ids=(new_ctx_id,))
            main_factory = self._main_cache_factory()
            aux_factory = self._aux_cache_factory()
            try:
                session = unpack_bagel_multi_context(
                    meta,
                    raw_payload,
                    main_cache_factory=main_factory,
                    aux_cache_factory=aux_factory,
                    use_cow_rebind=self.policy.use_module3_cow_loopback,
                )
            except BaseException as exc:
                if (
                    allow_auto_swap
                    and is_oom_exception(exc)
                    and self.evict_idle_kv_to_host(1 << 50, exclude_ctx_ids=(new_ctx_id,)) > 0
                ):
                    session = unpack_bagel_multi_context(
                        meta,
                        raw_payload,
                        main_cache_factory=main_factory,
                        aux_cache_factory=aux_factory,
                        use_cow_rebind=self.policy.use_module3_cow_loopback,
                    )
                else:
                    raise
            self._ctx_store[new_ctx_id] = session
            if self._active_ctx_pins.get(new_ctx_id, 0) > 0:
                self._sync_ctx_vmm_activity(new_ctx_id, active=True)
            self._update_hwm()
            return session.logical_bytes()

    def release_context(self, ctx_id: str) -> None:
        import torch

        self._active_ctx_pins.pop(ctx_id, None)
        self._ctx_last_used_s.pop(ctx_id, None)
        self._pending_h2d_weight_ms.pop(ctx_id, None)
        self._pending_vae_encode_ms.pop(ctx_id, None)
        if ctx_id in self._naive_offload_held_ctx_ids:
            self._release_naive_hold(ctx_id)
        old = self._ctx_store.pop(ctx_id, None)
        if old is not None:
            freed_host_b = old.close()
            if freed_host_b > 0:
                self._kv_host_pinned_bytes = max(0, self._kv_host_pinned_bytes - freed_host_b)
        if torch.cuda.is_available() and getattr(self.device, "type", "cuda") == "cuda":
            if self.vmm is None:
                self._update_hwm()
                torch.cuda.empty_cache()
            else:
                unused_pt = max(
                    0,
                    int(torch.cuda.memory_reserved(self.device)) - int(torch.cuda.memory_allocated(self.device)),
                )
                if unused_pt >= 1024 * MIB:
                    free_drv, _ = torch.cuda.mem_get_info(self.device)
                    if int(free_drv) < 4096 * MIB:
                        self._update_hwm()
                        torch.cuda.empty_cache()

    def reset_for_new_suite(self) -> float:
        self._pending_h2d_weight_ms.clear()
        self._pending_vae_encode_ms.clear()
        self._release_naive_hold(force=True)
        return super().reset_for_new_suite()

    def memory_and_vmm_stats(self) -> dict[str, Any]:
        stats = super().memory_and_vmm_stats()
        stats["swapped_kv_contexts"] = sum(1 for s in self._ctx_store.values() if s.is_swapped)
        stats["kv_host_swap_outs"] = int(stats["kv_host_swap_outs"]) + self._kv_host_swap_outs
        stats["kv_host_swap_ins"] = int(stats["kv_host_swap_ins"]) + self._kv_host_swap_ins
        stats["kv_host_pinned_bytes"] = int(stats["kv_host_pinned_bytes"]) + self._kv_host_pinned_bytes
        return stats

    def close(self) -> None:
        import torch

        self._stop_nvml_poller()
        self._pending_h2d_weight_ms.clear()
        self._pending_vae_encode_ms.clear()
        self._release_naive_hold(force=True)
        for cid in list(self._ctx_store.keys()):
            self.release_context(cid)
        self.interceptor.uninstall()
        if self.pager is not None:
            self.pager.close()
            self.pager = None
        if self.naive_offloader is not None:
            self.naive_offloader.close()
            self.naive_offloader = None
        if self.fission_dropper is not None:
            self.fission_dropper.close()
            self.fission_dropper = None
        if self.vmm is not None:
            self.vmm.shutdown()
            self.vmm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
