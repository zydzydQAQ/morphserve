"""Contiguous-VA `VmmNaiveCache` backed by `TriPoolVMM` Arena-1 `2 MiB` physical pages with in-place append & async Host Pinned swap."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
import ctypes
import time
from typing import Any
import weakref

MIB = 1024**2
DEFAULT_MAX_KV_TOKENS = 32_768

try:
    import torch as _TORCH
except ImportError:  # pragma: no cover
    _TORCH = None  # type: ignore[assignment]

try:
    import numpy as _NP
except ImportError:  # pragma: no cover
    _NP = None  # type: ignore[assignment]


def _stream_ptr_for_device(device: Any) -> int:
    if _TORCH is not None:
        try:
            if _TORCH.cuda.is_available() and str(device).startswith("cuda"):
                return int(_TORCH.cuda.current_stream(device).cuda_stream)
        except Exception:
            pass
    return 0


class _SimArrayKVTensor:
    """Zero-copy NumPy-backed tensor view over `SimulatedTriPoolBackend` fixed `mmap` VA (for PyTorch-free tests)."""

    def __init__(self, arr: Any, device: str = "cpu") -> None:
        self.array = arr
        self.shape = tuple(int(d) for d in arr.shape)
        self.dtype = arr.dtype
        self.device = device
        self._vmm_external_ro = False

    @property
    def ndim(self) -> int:
        return int(self.array.ndim)

    def data_ptr(self) -> int:
        return int(self.array.ctypes.data)

    def stride(self) -> tuple[int, ...]:
        isz = max(1, int(self.array.itemsize))
        return tuple(int(s // isz) for s in self.array.strides)

    def numel(self) -> int:
        return int(self.array.size)

    def element_size(self) -> int:
        return int(self.array.itemsize)

    def clone(self) -> "_SimArrayKVTensor":
        return _SimArrayKVTensor(self.array.copy(), device=self.device)

    def __getitem__(self, item: Any) -> "_SimArrayKVTensor":
        return _SimArrayKVTensor(self.array[item], device=self.device)

    def __array__(self, dtype: Any = None) -> Any:
        return self.array if dtype is None else self.array.astype(dtype)


class _GpuRawKVTensor:
    """Zero-copy GPU VA tensor descriptor supporting ctypes H2D/D2H verification when PyTorch is absent."""

    _cu_lib: Any = None
    _htod_fn: Any = None
    _dtoh_fn: Any = None

    @classmethod
    def _ensure_cuda_memcpy(cls) -> tuple[Any, Any]:
        if cls._htod_fn is None or cls._dtoh_fn is None:
            lib = ctypes.CDLL("libcuda.so.1")
            htod = getattr(lib, "cuMemcpyHtoD_v2", lib.cuMemcpyHtoD)
            htod.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
            htod.restype = ctypes.c_int
            dtoh = getattr(lib, "cuMemcpyDtoH_v2", lib.cuMemcpyDtoH)
            dtoh.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
            dtoh.restype = ctypes.c_int
            cls._cu_lib = lib
            cls._htod_fn = htod
            cls._dtoh_fn = dtoh
        return cls._htod_fn, cls._dtoh_fn

    def __init__(self, va_ptr: int, shape: tuple[int, ...], np_dtype: Any, device: str = "cuda:0") -> None:
        assert _NP is not None
        self._va_ptr = int(va_ptr)
        self.shape = tuple(int(d) for d in shape)
        self.dtype = _NP.dtype(np_dtype)
        self.device = device
        self._vmm_external_ro = False

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def data_ptr(self) -> int:
        return self._va_ptr

    def element_size(self) -> int:
        return int(self.dtype.itemsize)

    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def stride(self) -> tuple[int, ...]:
        if not self.shape:
            return ()
        st = [1] * len(self.shape)
        for i in range(len(self.shape) - 2, -1, -1):
            st[i] = st[i + 1] * self.shape[i + 1]
        return tuple(st)

    @property
    def row_bytes(self) -> int:
        if not self.shape or self.shape[0] == 0:
            return self.element_size()
        return (self.numel() // self.shape[0]) * self.element_size()

    def __getitem__(self, slc: slice) -> "_GpuRawKVTensor":
        if not isinstance(slc, slice):
            raise TypeError("_GpuRawKVTensor supports row-slice indexing")
        start, stop, step = slc.indices(self.shape[0])
        if step != 1:
            raise ValueError("_GpuRawKVTensor requires contiguous step=1 slicing")
        new_len = max(0, stop - start)
        off_b = start * self.row_bytes
        return _GpuRawKVTensor(self._va_ptr + off_b, (new_len, *self.shape[1:]), self.dtype, device=self.device)

    def copy_from_numpy(self, arr: Any) -> None:
        assert _NP is not None
        contig = _NP.ascontiguousarray(arr, dtype=self.dtype)
        nbytes = int(contig.nbytes)
        if nbytes > 0:
            htod, _ = self._ensure_cuda_memcpy()
            rc = int(htod(ctypes.c_uint64(self._va_ptr), ctypes.c_void_p(contig.ctypes.data), ctypes.c_size_t(nbytes)))
            if rc != 0:
                raise RuntimeError(f"cuMemcpyHtoD failed with CUresult={rc}")

    def to_numpy(self) -> Any:
        assert _NP is not None
        out = _NP.empty(self.shape, dtype=self.dtype)
        nbytes = int(out.nbytes)
        if nbytes > 0:
            _, dtoh = self._ensure_cuda_memcpy()
            rc = int(dtoh(ctypes.c_void_p(out.ctypes.data), ctypes.c_uint64(self._va_ptr), ctypes.c_size_t(nbytes)))
            if rc != 0:
                raise RuntimeError(f"cuMemcpyDtoH failed with CUresult={rc}")
        return out


class _VmmLayerTensorMap(dict):
    """Custom per-layer tensor dict intercepting `NaiveCache` reads & writes with auto swap-in."""

    def __init__(self, owner: "VmmNaiveCache", kind: str) -> None:
        super().__init__()
        self._owner = owner
        self._kind = kind  # "k" | "v"
        for lidx in range(owner.num_layers):
            super().__setitem__(lidx, None)

    def __getitem__(self, layer_idx: int) -> Any:
        self._owner._touch_access()
        self._owner._ensure_cow_parent_resident_if_any()
        if self._owner._is_swapped:
            self._owner.ensure_resident()
        return super().__getitem__(int(layer_idx))

    def get(self, layer_idx: int, default: Any = None) -> Any:
        self._owner._touch_access()
        self._owner._ensure_cow_parent_resident_if_any()
        if self._owner._is_swapped:
            self._owner.ensure_resident()
        return super().get(int(layer_idx), default)

    def values(self):  # type: ignore[override]
        self._owner._touch_access()
        self._owner._ensure_cow_parent_resident_if_any()
        if self._owner._is_swapped:
            self._owner.ensure_resident()
        return super().values()

    def items(self):  # type: ignore[override]
        self._owner._touch_access()
        self._owner._ensure_cow_parent_resident_if_any()
        if self._owner._is_swapped:
            self._owner.ensure_resident()
        return super().items()

    def copy(self) -> dict[int, Any]:
        self._owner._touch_access()
        self._owner._ensure_cow_parent_resident_if_any()
        if self._owner._is_swapped:
            self._owner.ensure_resident()
        return {k: dict.__getitem__(self, k) for k in self}

    def __setitem__(self, layer_idx: int, value: Any) -> None:
        lidx = int(layer_idx)
        self._owner._touch_access()
        if value is None:
            self._owner._release_slot(self._kind, lidx)
            super().__setitem__(lidx, None)
            return
        owner_ref = getattr(value, "_vmm_owner_ref", None)
        if self._owner.vmm is None:
            if owner_ref is not None and owner_ref() is not self._owner:
                self._owner._cow_parent_ref = owner_ref
            super().__setitem__(lidx, value)
            return
        if self._owner._is_swapped:
            self._owner.ensure_resident()
        bound = self._owner._store_layer_tensor(self._kind, lidx, value, super().get(lidx))
        super().__setitem__(lidx, bound)


def _normalize_step_tensor(t: Any) -> Any:
    """Accept `[H, D]` single-token or `[ΔL, H, D]` multi-token KV step tensors."""
    ndim = getattr(t, "ndim", None)
    if ndim is None and hasattr(t, "shape"):
        ndim = len(t.shape)
    if ndim == 2:
        if hasattr(t, "unsqueeze"):
            return t.unsqueeze(0)
        if _NP is not None and isinstance(t, _NP.ndarray):
            return _NP.expand_dims(t, axis=0)
    return t


def _concat_fallback(old_t: Any, new_t: Any) -> Any:
    if old_t is None:
        return new_t
    if _TORCH is not None and _TORCH.is_tensor(old_t):
        return _TORCH.cat([old_t, new_t], dim=0)
    if isinstance(old_t, _SimArrayKVTensor) and _NP is not None:
        rhs = new_t.array if isinstance(new_t, _SimArrayKVTensor) else _NP.asarray(new_t)
        return _SimArrayKVTensor(_NP.concatenate([old_t.array, rhs], axis=0), device=old_t.device)
    if _NP is not None and isinstance(old_t, _NP.ndarray):
        return _NP.concatenate([old_t, _NP.asarray(new_t)], axis=0)
    raise TypeError(f"unsupported fallback concat tensor type: {type(old_t)!r}")


class VmmNaiveCache:
    """Drop-in `vllm_omni` `NaiveCache` replacement growing K/V tensors in-place on `TriPoolVMM` Arena 1.

    Key properties:
      - Reserves contiguous virtual address space per `(layer, K/V)` (`0 B` physical RAM initially).
      - Maps `2 MiB` physical pages (`2,048` tokens/page for BAGEL `4x128 bf16`) strictly on demand,
        caching full-capacity typed tensor views so 2,047/2,048 token appends issue **zero** `ctypes`
        or `__cuda_array_interface__` calls and **zero** `torch.cat` allocations.
      - Supports async VA-preserving swap-out / swap-in of inactive thinking-period sessions to
        Host Pinned DRAM under high VMM watermark, maintaining base-pointer identity (`data_ptr()`)
        across page growth and across swap-out/swap-in cycles.
    """

    _bagel_subclasses: dict[type, type] = {}

    def __init__(
        self,
        num_layers: int,
        vmm: Any = None,
        max_seq_len: int = DEFAULT_MAX_KV_TOKENS,
        max_row_bytes: int = 1024,
    ) -> None:
        self._num_layers = int(num_layers)
        self.vmm = vmm
        self.max_seq_len = max(128, int(max_seq_len))
        self.default_max_va_bytes = max(2 * MIB, self.max_seq_len * int(max_row_bytes))
        self.key_values_lens: list[int] | None = None
        self._cow_shared_prefix_len: int = 0
        self._cow_origin_id: int | None = None

        self._region_ids: dict[tuple[str, int], int] = {}
        self._region_caps: dict[tuple[str, int], int] = {}
        self._mapped_bytes: dict[tuple[str, int], int] = {}
        self._row_bytes: dict[tuple[str, int], int] = {}
        self._lengths: dict[tuple[str, int], int] = {}
        self._capacity_views: dict[tuple[str, int], Any] = {}

        self._swapped_flag: bool = False
        self._c_evictable: bool = False
        self._is_thinking: bool = False
        self._is_pinned_active: bool = False
        self._last_access_mono: float = time.monotonic()
        self._closed = False
        self._self_ref: weakref.ReferenceType["VmmNaiveCache"] = weakref.ref(self)
        self._cow_parent_ref: weakref.ReferenceType["VmmNaiveCache"] | None = None

        self.key_cache = _VmmLayerTensorMap(self, "k")
        self.value_cache = _VmmLayerTensorMap(self, "v")

        if self.vmm is not None and hasattr(self.vmm, "register_kv_cache"):
            self.vmm.register_kv_cache(self)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _touch_access(self) -> None:
        self._last_access_mono = time.monotonic()

    def _tag_owner(self, view: Any) -> Any:
        try:
            view._vmm_owner_ref = self._self_ref
        except Exception:
            pass
        return view

    def _ensure_cow_parent_resident_if_any(self) -> None:
        pref = getattr(self, "_cow_parent_ref", None)
        if pref is not None:
            parent = pref()
            if parent is not None and parent is not self and not parent._closed and parent._is_swapped:
                parent.ensure_resident()

    def _resolve_stream_ptr(self, stream_ptr: int | None) -> int:
        if stream_ptr is not None:
            return int(stream_ptr)
        for mp in (self.key_cache, self.value_cache):
            for val in dict.values(mp):
                if val is not None:
                    dev = getattr(val, "device", "cpu")
                    if str(dev).startswith("cuda"):
                        return _stream_ptr_for_device(dev)
                    break
        if self.vmm is not None and hasattr(self.vmm, "_current_stream_ptr"):
            return int(self.vmm._current_stream_ptr())
        return 0

    @property
    def _is_swapped(self) -> bool:
        if self._swapped_flag:
            return True
        if not self._c_evictable or self._closed or self.vmm is None or not self._region_ids:
            return False
        if hasattr(self.vmm, "kv_region_is_swapped"):
            for rid in self._region_ids.values():
                if self.vmm.kv_region_is_swapped(rid):
                    self._swapped_flag = True
                    return True
        return False

    @_is_swapped.setter
    def _is_swapped(self, val: bool) -> None:
        self._swapped_flag = bool(val)

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def last_access_mono(self) -> float:
        return self._last_access_mono

    @property
    def idle_seconds(self) -> float:
        return max(0.0, time.monotonic() - self._last_access_mono)

    @property
    def is_thinking(self) -> bool:
        return self._is_thinking

    @property
    def is_pinned_active(self) -> bool:
        return self._is_pinned_active

    @property
    def is_swapped_out(self) -> bool:
        return self._is_swapped

    @property
    def swapped_bytes(self) -> int:
        if self._closed or self.vmm is None or not self._region_ids:
            return 0
        if hasattr(self.vmm, "kv_region_swapped_bytes"):
            return sum(int(self.vmm.kv_region_swapped_bytes(rid)) for rid in self._region_ids.values())
        return 0

    @classmethod
    def create_for_bagel(
        cls,
        naive_cache_cls: type,
        num_layers: int,
        vmm: Any = None,
        max_seq_len: int = DEFAULT_MAX_KV_TOKENS,
    ) -> "VmmNaiveCache":
        """Instantiate a `VmmNaiveCache` subclassing `vllm_omni`'s `NaiveCache` (`isinstance == True`)."""
        sub_cls = cls._bagel_subclasses.get(naive_cache_cls)
        if sub_cls is None:
            sub_cls = type("BagelVmmNaiveCache", (cls, naive_cache_cls), {})
            cls._bagel_subclasses[naive_cache_cls] = sub_cls
        return sub_cls(num_layers, vmm=vmm, max_seq_len=max_seq_len)

    def __deepcopy__(self, memo: dict[int, Any]) -> "VmmNaiveCache":
        """Deep-copy KV contents into a fresh Arena-1 `VmmNaiveCache` without pickling `self.vmm` (`CDLL`)."""
        was_pinned = self._is_pinned_active
        was_thinking = self._is_thinking
        self._is_pinned_active = True
        self._is_thinking = False
        dup = type(self)(self.num_layers, vmm=self.vmm, max_seq_len=self.max_seq_len)
        dup._is_pinned_active = True
        memo[id(self)] = dup
        try:
            self._ensure_cow_parent_resident_if_any()
            if self._is_swapped:
                self.ensure_resident()
            dup.key_values_lens = list(self.key_values_lens) if self.key_values_lens is not None else None
            dup._cow_shared_prefix_len = self._cow_shared_prefix_len
            dup._cow_origin_id = self._cow_origin_id
            for lidx in range(self.num_layers):
                k = dict.get(self.key_cache, lidx)
                v = dict.get(self.value_cache, lidx)
                if k is not None:
                    dup.key_cache[lidx] = k.clone() if (self.vmm is None and hasattr(k, "clone")) else k
                if v is not None:
                    dup.value_cache[lidx] = v.clone() if (self.vmm is None and hasattr(v, "clone")) else v
        except BaseException:
            dup.close()
            raise
        finally:
            self._is_pinned_active = was_pinned
            self._is_thinking = was_thinking
            dup._is_pinned_active = False
        return dup

    @property
    def seq_lens(self) -> int:
        if (
            self._lengths
            and ("k", 0) in self._lengths
        ):
            return int(self._lengths[("k", 0)])
        k0 = dict.get(self.key_cache, 0)
        return int(k0.shape[0]) if k0 is not None else 0

    @property
    def mapped_bytes(self) -> int:
        if self._closed or self.vmm is None or getattr(self.vmm, "_closed", False):
            return 0
        return sum(int(self.vmm.kv_region_mapped_bytes(rid)) for rid in self._region_ids.values())

    @property
    def allocated_regions(self) -> int:
        return len(self._region_ids)

    @staticmethod
    def merge(caches: list[Any]) -> Any:
        """Delegate static `merge` to `vllm_omni` `NaiveCache.merge` (or `BagelSharedPrefixPatch` hook)."""
        from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache

        return NaiveCache.merge(caches)

    @staticmethod
    def split_with_zeros(tensor: Any, lengths: list[int]) -> list[Any | None]:
        """Exact match to `vllm_omni` `NaiveCache.split_with_zeros`."""
        expected = sum(ln for ln in lengths if ln > 0)
        if int(tensor.shape[0]) != expected:
            raise ValueError(f"tensor dim 0 ({tensor.shape[0]}) != sum of nonzero lengths ({expected})")
        result: list[Any | None] = []
        offset = 0
        for ln in lengths:
            if ln > 0:
                result.append(tensor[offset : offset + ln])
                offset += ln
            else:
                result.append(None)
        return result

    @classmethod
    def from_object(cls, obj: Any) -> Any:
        if isinstance(obj, cls):
            return obj
        from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache

        return NaiveCache.from_object(obj)

    # ==================== Session Activity & Async Host Pinned DRAM Swap API ====================

    def mark_active(self, stream_ptr: int | None = None) -> None:
        """Pin this session's KV on GPU during an active AR/DiT turn and swap in if parked."""
        self._is_pinned_active = True
        self._is_thinking = False
        self._touch_access()
        self._ensure_cow_parent_resident_if_any()
        if self._is_swapped:
            self.ensure_resident(stream_ptr=stream_ptr)
        elif self._c_evictable and not self._closed and self.vmm is not None:
            if hasattr(self.vmm, "kv_region_set_evictable"):
                for rid in self._region_ids.values():
                    self.vmm.kv_region_set_evictable(rid, False, 0)
            self._c_evictable = False

    def mark_thinking(self, trigger_watermark_check: bool = True, stream_ptr: int | None = None) -> int:
        """Mark session as inactive (inter-turn user think time) and optionally run high-watermark eviction."""
        self._is_pinned_active = False
        self._is_thinking = True
        if self._closed or self.vmm is None:
            return 0
        eff_s = self._resolve_stream_ptr(stream_ptr)
        if self._region_ids and hasattr(self.vmm, "kv_region_set_evictable"):
            for key, rid in self._region_ids.items():
                if not self.vmm.kv_region_is_swapped(rid):
                    valid_b = int(self._lengths.get(key, 0)) * int(self._row_bytes.get(key, 0))
                    self.vmm.kv_region_set_evictable(rid, True, valid_b, stream_ptr=eff_s)
            self._c_evictable = True
        if trigger_watermark_check and hasattr(self.vmm, "maybe_evict_thinking_kv_on_high_watermark"):
            return int(self.vmm.maybe_evict_thinking_kv_on_high_watermark(sync=False, stream_ptr=stream_ptr))
        return 0

    mark_inactive = mark_thinking

    @contextmanager
    def active_turn_scope(self, mark_thinking_on_exit: bool = True, stream_ptr: int | None = None):
        """Context manager pinning KV during an active turn and marking thinking-period on exit."""
        self.mark_active(stream_ptr=stream_ptr)
        try:
            yield self
        finally:
            if mark_thinking_on_exit:
                self.mark_thinking(trigger_watermark_check=True, stream_ptr=stream_ptr)
            else:
                self._is_pinned_active = False

    def swap_out_async(self, stream_ptr: int | None = None) -> int:
        """Asynchronously DMA all mapped K/V regions of this cache to Host Pinned DRAM and free GPU pages."""
        if self._closed or self.vmm is None or getattr(self.vmm, "_closed", False) or not self._region_ids:
            return 0
        eff_stream = self._resolve_stream_ptr(stream_ptr)
        gran = int(self.vmm.stats_dict().get("min_granularity_bytes", 2 * MIB))
        reclaimed_bytes = 0
        any_swapped = False
        for key, rid in list(self._region_ids.items()):
            if self.vmm.kv_region_is_swapped(rid):
                any_swapped = True
                self._mapped_bytes[key] = 0
                continue
            seq_l = int(self._lengths.get(key, 0))
            row_b = int(self._row_bytes.get(key, 0))
            valid_b = seq_l * row_b
            pages = int(self.vmm.kv_region_swap_out_async(rid, valid_b, stream_ptr=eff_stream))
            if pages > 0 or self.vmm.kv_region_is_swapped(rid):
                any_swapped = True
                self._mapped_bytes[key] = 0
            reclaimed_bytes += pages * gran
        if any_swapped:
            self._c_evictable = False
            self._swapped_flag = True
        return reclaimed_bytes

    def swap_in_async(self, stream_ptr: int | None = None) -> int:
        """Asynchronously restore all swapped K/V regions back into their exact original GPU virtual addresses."""
        if self._closed or self.vmm is None or getattr(self.vmm, "_closed", False) or not self._is_swapped:
            return 0
        was_pinned = self._is_pinned_active
        self._is_pinned_active = True
        self._is_thinking = False
        try:
            eff_stream = self._resolve_stream_ptr(stream_ptr)
            if hasattr(self.vmm, "kv_region_set_evictable"):
                for rid in self._region_ids.values():
                    self.vmm.kv_region_set_evictable(rid, False, 0)
            self._c_evictable = False

            gran = int(self.vmm.stats_dict().get("min_granularity_bytes", 2 * MIB))
            needed_restore_b = 0
            for key, rid in self._region_ids.items():
                if self.vmm.kv_region_is_swapped(rid):
                    valid_b = int(self._lengths.get(key, 0)) * int(self._row_bytes.get(key, 0))
                    pages_est = max(1, (max(1, valid_b) + gran - 1) // gran)
                    needed_restore_b += pages_est * gran

            if needed_restore_b > 0 and hasattr(self.vmm, "maybe_evict_thinking_kv_on_high_watermark"):
                self.vmm.maybe_evict_thinking_kv_on_high_watermark(
                    sync=True,
                    exclude_caches=(self,),
                    extra_needed_bytes=needed_restore_b,
                    stream_ptr=eff_stream,
                )

            restored_bytes = 0
            restored_this_call: list[tuple[tuple[str, int], int]] = []
            try:
                for key, rid in list(self._region_ids.items()):
                    was_region_swapped = bool(self.vmm.kv_region_is_swapped(rid))
                    try:
                        pages = int(self.vmm.kv_region_swap_in_async(rid, stream_ptr=eff_stream))
                    except MemoryError:
                        if hasattr(self.vmm, "reclaim_thinking_kv_bytes"):
                            reclaimed = self.vmm.reclaim_thinking_kv_bytes(
                                max(gran, needed_restore_b),
                                sync=True,
                                exclude_caches=(self,),
                                stream_ptr=eff_stream,
                            )
                            if reclaimed > 0:
                                pages = int(self.vmm.kv_region_swap_in_async(rid, stream_ptr=eff_stream))
                            else:
                                raise
                        else:
                            raise
                    mapped_b = int(self.vmm.kv_region_mapped_bytes(rid))
                    self._mapped_bytes[key] = mapped_b
                    restored_bytes += pages * gran
                    if was_region_swapped or pages > 0:
                        restored_this_call.append((key, rid))
                        self._refresh_layer_view_from_capacity(key, rid, mapped_b)
            except BaseException:
                # Roll back partially swapped-in regions so a failed swap-in never leaves zombie GPU pages!
                for r_key, r_rid in restored_this_call:
                    valid_b = int(self._lengths.get(r_key, 0)) * int(self._row_bytes.get(r_key, 0))
                    try:
                        self.vmm.kv_region_swap_out_async(r_rid, valid_b, stream_ptr=eff_stream)
                    except Exception:
                        pass
                    self._mapped_bytes[r_key] = 0
                if restored_this_call:
                    self._swapped_flag = True
                raise

            self._swapped_flag = False
            self._touch_access()
            return restored_bytes
        finally:
            self._is_pinned_active = was_pinned

    def ensure_resident(self, stream_ptr: int | None = None) -> bool:
        """Ensure all K/V regions are mapped on GPU (no-op if already resident)."""
        if not self._is_swapped:
            return False
        self.swap_in_async(stream_ptr=stream_ptr)
        return True

    # ==================== In-Place Single-Step / Multi-Step / Batched Append API ====================

    def get_layer_kv(self, layer_idx: int) -> tuple[Any | None, Any | None, int]:
        """Return `(key_tensor, value_tensor, seq_len)` for `layer_idx`, auto-swapping in if needed."""
        lidx = int(layer_idx)
        self._touch_access()
        self._ensure_cow_parent_resident_if_any()
        if self._is_swapped:
            self.ensure_resident()
        k = dict.get(self.key_cache, lidx)
        v = dict.get(self.value_cache, lidx)
        ln = int(k.shape[0]) if k is not None else 0
        return k, v, ln

    def append_inplace(self, layer_idx: int, new_k: Any, new_v: Any) -> tuple[Any, Any]:
        """Append `[ΔL, H, D]` (or `[H, D]` single-step) K/V in-place without `torch.cat` reallocation."""
        lidx = int(layer_idx)
        self._touch_access()
        self._ensure_cow_parent_resident_if_any()
        if self._is_swapped:
            self.ensure_resident()

        nk = _normalize_step_tensor(new_k)
        nv = _normalize_step_tensor(new_v)
        delta_len = int(nk.shape[0])
        prev_k = dict.get(self.key_cache, lidx)
        prev_v = dict.get(self.value_cache, lidx)

        if delta_len == 0:
            return (nk if prev_k is None else prev_k), (nv if prev_v is None else prev_v)

        if self.vmm is None:
            fk = _concat_fallback(prev_k, nk)
            fv = _concat_fallback(prev_v, nv)
            dict.__setitem__(self.key_cache, lidx, fk)
            dict.__setitem__(self.value_cache, lidx, fv)
            return fk, fv

        fk = self._append_single_kind_inplace("k", lidx, nk, prev_k)
        fv = self._append_single_kind_inplace("v", lidx, nv, prev_v)
        dict.__setitem__(self.key_cache, lidx, fk)
        dict.__setitem__(self.value_cache, lidx, fv)
        return fk, fv

    append_kv_step = append_inplace

    @classmethod
    def append_layer_kv(cls, cache: Any, layer_idx: int, new_k: Any, new_v: Any) -> tuple[Any, Any]:
        """Polymorphic in-place KV append supporting both `VmmNaiveCache` and stock `NaiveCache`."""
        if hasattr(cache, "append_inplace"):
            return cache.append_inplace(layer_idx, new_k, new_v)
        lidx = int(layer_idx)
        nk = _normalize_step_tensor(new_k)
        nv = _normalize_step_tensor(new_v)
        old_k = cache.key_cache[lidx]
        old_v = cache.value_cache[lidx]
        fk = _concat_fallback(old_k, nk)
        fv = _concat_fallback(old_v, nv)
        cache.key_cache[lidx] = fk
        cache.value_cache[lidx] = fv
        return fk, fv

    @classmethod
    def append_batch_inplace(
        cls,
        caches: Sequence[Any],
        layer_idx: int,
        packed_k: Any,
        packed_v: Any,
        query_lens: Sequence[int] | None = None,
    ) -> tuple[list[Any], list[Any], list[int]]:
        """Append packed `[B, H, D]` or varlen `[sum(Q), H, D]` K/V across `B` caches in-place."""
        b_cnt = len(caches)
        full_ks: list[Any] = []
        full_vs: list[Any] = []
        kv_lens: list[int] = []
        if b_cnt == 0:
            return full_ks, full_vs, kv_lens

        lidx = int(layer_idx)
        if tuple(int(d) for d in packed_k.shape) != tuple(int(d) for d in packed_v.shape):
            raise ValueError(
                f"append_batch_inplace packed_k.shape {tuple(packed_k.shape)} != packed_v.shape {tuple(packed_v.shape)}"
            )
        if query_lens is None:
            # Uniform decode/chunk batch: packed_k is [B, H, D] or [B, Q, H, D]
            if int(packed_k.shape[0]) != b_cnt:
                raise ValueError(
                    f"append_batch_inplace batch dim ({packed_k.shape[0]}) != len(caches) ({b_cnt})"
                )
            is_4d = int(getattr(packed_k, "ndim", len(packed_k.shape))) == 4
            for b, cache in enumerate(caches):
                kb = packed_k[b] if is_4d else packed_k[b : b + 1]
                vb = packed_v[b] if is_4d else packed_v[b : b + 1]
                fk, fv = cls.append_layer_kv(cache, lidx, kb, vb)
                full_ks.append(fk)
                full_vs.append(fv)
                kv_lens.append(int(fk.shape[0]))
            return full_ks, full_vs, kv_lens

        if len(query_lens) != b_cnt:
            raise ValueError(f"append_batch_inplace len(query_lens) ({len(query_lens)}) != len(caches) ({b_cnt})")
        q_ints = [int(q) for q in query_lens]
        if any(q < 0 for q in q_ints):
            raise ValueError(f"append_batch_inplace negative query_lens not allowed: {q_ints}")
        expected_rows = sum(q_ints)
        if int(packed_k.shape[0]) != expected_rows:
            raise ValueError(
                f"append_batch_inplace packed_k.shape[0] ({packed_k.shape[0]}) != sum(query_lens) ({expected_rows})"
            )

        offset = 0
        for cache, ql in zip(caches, q_ints):
            if ql > 0:
                kb = packed_k[offset : offset + ql]
                vb = packed_v[offset : offset + ql]
                offset += ql
                fk, fv = cls.append_layer_kv(cache, lidx, kb, vb)
            else:
                fk = cache.key_cache[lidx]
                fv = cache.value_cache[lidx]
            full_ks.append(fk)
            full_vs.append(fv)
            kv_lens.append(int(fk.shape[0]) if fk is not None else 0)
        return full_ks, full_vs, kv_lens

    def truncate(self, new_seq_len: int) -> None:
        """Truncate active KV sequence length across all layers in O(1) without reallocating VA."""
        target = max(0, int(new_seq_len))
        self._ensure_cow_parent_resident_if_any()
        if self._is_swapped and target > 0:
            self.ensure_resident()
        for lidx in range(self.num_layers):
            if target == 0:
                self._release_slot("k", lidx)
                self._release_slot("v", lidx)
                dict.__setitem__(self.key_cache, lidx, None)
                dict.__setitem__(self.value_cache, lidx, None)
                continue
            for kind, mp in (("k", self.key_cache), ("v", self.value_cache)):
                cur = dict.get(mp, lidx)
                if cur is not None and int(cur.shape[0]) > target:
                    dict.__setitem__(mp, lidx, self._tag_owner(cur[:target]))
                    self._lengths[(kind, lidx)] = target

    # ==================== Internal VMM Region & Capacity View Management ====================

    def _ensure_region(self, kind: str, layer_idx: int, needed_bytes: int, row_bytes: int, stream_ptr: int) -> int:
        key = (kind, layer_idx)
        req_b = max(1, int(needed_bytes))
        cached_mapped = self._mapped_bytes.get(key, 0)
        rid = self._region_ids.get(key)
        cur_cap = self._region_caps.get(key, 0)
        if rid is not None and not self._is_swapped and req_b <= cur_cap and cached_mapped >= req_b:
            return rid

        delta_b = max(0, req_b - cached_mapped)
        if delta_b > 0 and hasattr(self.vmm, "maybe_evict_thinking_kv_on_high_watermark"):
            self.vmm.maybe_evict_thinking_kv_on_high_watermark(
                sync=False,
                exclude_caches=(self,),
                extra_needed_bytes=delta_b,
                stream_ptr=stream_ptr,
            )

        if rid is not None and req_b > cur_cap:
            # Transparently grow VA reservation if a marathon multi-turn session exceeds initial max_seq_len!
            target_map = self.key_cache if kind == "k" else self.value_cache
            old_view = dict.get(target_map, layer_idx)
            old_len = int(self._lengths.get(key, 0))
            new_cap = max(self.default_max_va_bytes, req_b * 2, cur_cap * 2)
            new_rid = int(self.vmm.kv_region_alloc(new_cap))
            try:
                try:
                    self.vmm.kv_region_ensure(new_rid, req_b, stream_ptr=stream_ptr)
                except MemoryError:
                    if hasattr(self.vmm, "reclaim_thinking_kv_bytes"):
                        reclaimed = self.vmm.reclaim_thinking_kv_bytes(
                            max(2 * MIB, req_b),
                            sync=True,
                            exclude_caches=(self,),
                            stream_ptr=stream_ptr,
                        )
                        if reclaimed > 0:
                            self.vmm.kv_region_ensure(new_rid, req_b, stream_ptr=stream_ptr)
                        else:
                            raise
                    else:
                        raise
                new_mapped = int(self.vmm.kv_region_mapped_bytes(new_rid))
                if old_view is not None and old_len > 0:
                    trailing = tuple(int(d) for d in old_view.shape[1:])
                    is_sim = bool(getattr(self.vmm, "is_simulated", False))
                    new_cap_view = self._build_capacity_view(
                        key, new_rid, new_mapped, max(1, int(row_bytes)), trailing, old_view
                    )
                    if new_cap_view is not None:
                        self._copy_rows_into(new_cap_view, 0, old_len, old_view[:old_len], is_sim=is_sim)
                        dict.__setitem__(target_map, layer_idx, self._tag_owner(new_cap_view[:old_len]))
                else:
                    self._capacity_views.pop(key, None)
            except BaseException:
                self.vmm.kv_region_free(new_rid, stream_ptr=stream_ptr)
                raise
            self.vmm.kv_region_free(rid, stream_ptr=stream_ptr)
            self._region_ids[key] = new_rid
            self._region_caps[key] = new_cap
            self._mapped_bytes[key] = new_mapped
            self._row_bytes[key] = max(1, int(row_bytes))
            return new_rid

        if rid is None:
            cap = max(self.default_max_va_bytes, self.max_seq_len * max(1, row_bytes), req_b)
            rid = int(self.vmm.kv_region_alloc(cap))
            self._region_ids[key] = rid
            self._region_caps[key] = cap

        try:
            self.vmm.kv_region_ensure(rid, req_b, stream_ptr=stream_ptr)
        except MemoryError:
            if hasattr(self.vmm, "reclaim_thinking_kv_bytes"):
                reclaimed = self.vmm.reclaim_thinking_kv_bytes(
                    max(2 * MIB, delta_b),
                    sync=True,
                    exclude_caches=(self,),
                    stream_ptr=stream_ptr,
                )
                if reclaimed > 0:
                    self.vmm.kv_region_ensure(rid, req_b, stream_ptr=stream_ptr)
                else:
                    raise
            else:
                raise

        self._mapped_bytes[key] = int(self.vmm.kv_region_mapped_bytes(rid))
        self._row_bytes[key] = max(1, int(row_bytes))
        return rid

    def _build_capacity_view(
        self,
        key: tuple[str, int],
        rid: int,
        mapped_b: int,
        row_bytes: int,
        trailing_shape: tuple[int, ...],
        sample_tensor: Any,
    ) -> Any:
        cap_tokens = max(1, mapped_b // max(1, row_bytes))
        usable_bytes = cap_tokens * row_bytes
        is_sim = bool(getattr(self.vmm, "is_simulated", False))

        if not is_sim:
            if _TORCH is not None and _TORCH.is_tensor(sample_tensor):
                raw_u8 = self.vmm.kv_region_uint8_view(rid, mapped_b)
                cap_view = raw_u8[:usable_bytes].view(sample_tensor.dtype).reshape((cap_tokens, *trailing_shape))
                self._capacity_views[key] = cap_view
                return cap_view
            if _NP is not None and isinstance(sample_tensor, (_NP.ndarray, _GpuRawKVTensor)):
                va_ptr = int(self.vmm.kv_region_va(rid))
                cap_view = _GpuRawKVTensor(
                    va_ptr,
                    (cap_tokens, *trailing_shape),
                    sample_tensor.dtype,
                    device=f"cuda:{getattr(self.vmm, 'device', 0)}",
                )
                self._capacity_views[key] = cap_view
                return cap_view
            return None

        raw_u8 = self.vmm.kv_region_uint8_view(rid, mapped_b)
        if _TORCH is not None and _TORCH.is_tensor(sample_tensor):
            cap_view = (
                _TORCH.frombuffer(raw_u8, dtype=_TORCH.uint8, count=usable_bytes)
                .view(sample_tensor.dtype)
                .reshape((cap_tokens, *trailing_shape))
            )
            self._capacity_views[key] = cap_view
            return cap_view
        if _NP is not None and isinstance(sample_tensor, (_NP.ndarray, _SimArrayKVTensor)):
            np_dt = sample_tensor.dtype
            elem_cnt = usable_bytes // int(_NP.dtype(np_dt).itemsize)
            arr = _NP.frombuffer(raw_u8, dtype=np_dt, count=elem_cnt).reshape((cap_tokens, *trailing_shape))
            cap_view = _SimArrayKVTensor(arr, device="cpu")
            self._capacity_views[key] = cap_view
            return cap_view
        return None

    def _get_or_grow_capacity_view(
        self,
        kind: str,
        layer_idx: int,
        needed_tokens: int,
        row_bytes: int,
        trailing_shape: tuple[int, ...],
        sample_tensor: Any,
        stream_ptr: int,
    ) -> Any:
        key = (kind, layer_idx)
        needed_bytes = max(1, needed_tokens * row_bytes)
        cap_view = self._capacity_views.get(key)
        if (
            cap_view is not None
            and not self._is_swapped
            and self._mapped_bytes.get(key, 0) >= needed_bytes
            and int(cap_view.shape[0]) >= needed_tokens
            and tuple(int(d) for d in cap_view.shape[1:]) == trailing_shape
            and cap_view.dtype == sample_tensor.dtype
        ):
            return cap_view

        rid = self._ensure_region(kind, layer_idx, needed_bytes, row_bytes, stream_ptr)
        mapped_b = self._mapped_bytes[key]
        return self._build_capacity_view(key, rid, mapped_b, row_bytes, trailing_shape, sample_tensor)

    def _refresh_layer_view_from_capacity(self, key: tuple[str, int], rid: int, mapped_b: int) -> None:
        kind, lidx = key
        target_map = self.key_cache if kind == "k" else self.value_cache
        cur = dict.get(target_map, lidx)
        seq_l = int(self._lengths.get(key, 0))
        row_b = int(self._row_bytes.get(key, 0))
        if cur is None or seq_l <= 0 or row_b <= 0 or mapped_b <= 0:
            return
        trailing = tuple(int(d) for d in cur.shape[1:])
        cap_view = self._build_capacity_view(key, rid, mapped_b, row_b, trailing, cur)
        if cap_view is not None:
            dict.__setitem__(target_map, lidx, self._tag_owner(cap_view[:seq_l]))

    @staticmethod
    def _copy_rows_into(cap_view: Any, start: int, end: int, src: Any, is_sim: bool) -> None:
        if _TORCH is not None and _TORCH.is_tensor(cap_view):
            rhs = src.detach().cpu() if is_sim else src
            cap_view[start:end].copy_(rhs)
            return
        if isinstance(cap_view, _GpuRawKVTensor):
            arr = src.to_numpy() if isinstance(src, _GpuRawKVTensor) else src
            cap_view[start:end].copy_from_numpy(arr)
            return
        if isinstance(cap_view, _SimArrayKVTensor):
            rhs_arr = src.array if isinstance(src, _SimArrayKVTensor) else src
            cap_view.array[start:end] = rhs_arr
            return
        raise TypeError(f"unsupported capacity view type: {type(cap_view)!r}")

    def _append_single_kind_inplace(self, kind: str, layer_idx: int, step_tensor: Any, prev_view: Any | None) -> Any:
        key = (kind, layer_idx)
        delta_len = int(step_tensor.shape[0])
        trailing = tuple(int(d) for d in step_tensor.shape[1:])
        elem_sz = int(step_tensor.element_size()) if hasattr(step_tensor, "element_size") else int(step_tensor.itemsize)
        numel = int(step_tensor.numel()) if hasattr(step_tensor, "numel") else int(step_tensor.size)
        row_bytes = max(1, (numel // max(1, delta_len)) * elem_sz)
        dev = getattr(step_tensor, "device", "cpu")
        stream_ptr = _stream_ptr_for_device(dev)
        is_sim = bool(getattr(self.vmm, "is_simulated", False))

        if prev_view is not None:
            owner_ref = getattr(prev_view, "_vmm_owner_ref", None)
            if owner_ref is not None:
                owner = owner_ref()
                if owner is not None and owner is not self and not owner._closed and owner._is_swapped:
                    owner.ensure_resident()

        old_len = int(prev_view.shape[0]) if prev_view is not None else 0
        new_len = old_len + delta_len

        cap_view = self._get_or_grow_capacity_view(
            kind, layer_idx, new_len, row_bytes, trailing, step_tensor, stream_ptr
        )
        if cap_view is None:
            # Bare duck-typed mock tensor fallback
            self._lengths[key] = new_len
            return step_tensor

        # Copy-on-Write check: if `prev_view` does not already alias `cap_view`'s base VA (e.g., external CoW fork),
        # materialize `[0 : old_len]` into our private VMM region before writing `[old_len : new_len]`!
        if old_len > 0 and prev_view is not None:
            same_base = (
                hasattr(prev_view, "data_ptr")
                and hasattr(cap_view, "data_ptr")
                and int(prev_view.data_ptr()) == int(cap_view.data_ptr())
            )
            if not same_base:
                self._copy_rows_into(cap_view, 0, old_len, prev_view, is_sim=is_sim)

        self._copy_rows_into(cap_view, old_len, new_len, step_tensor, is_sim=is_sim)
        self._lengths[key] = new_len
        out_view = self._tag_owner(cap_view[:new_len])
        if is_sim and _TORCH is not None and _TORCH.is_tensor(step_tensor) and step_tensor.device.type != "cpu":
            return out_view.to(step_tensor.device)
        return out_view

    def _store_layer_tensor(self, kind: str, layer_idx: int, tensor: Any, prev_view: Any | None) -> Any:
        key = (kind, layer_idx)
        if getattr(tensor, "_vmm_external_ro", False):
            owner_ref = getattr(tensor, "_vmm_owner_ref", None)
            if owner_ref is not None and owner_ref() is not self:
                self._cow_parent_ref = owner_ref
            self._lengths[key] = int(tensor.shape[0])
            return tensor

        seq_len = int(tensor.shape[0])
        if seq_len == 0:
            self._lengths[key] = 0
            return tensor

        owner_ref = getattr(tensor, "_vmm_owner_ref", None)
        if owner_ref is not None:
            owner = owner_ref()
            if owner is not None and owner is not self and not owner._closed and owner._is_swapped:
                owner.ensure_resident()

        trailing = tuple(int(d) for d in tensor.shape[1:])
        elem_sz = (
            int(tensor.element_size())
            if hasattr(tensor, "element_size")
            else (int(tensor.itemsize) if hasattr(tensor, "itemsize") else 2)
        )
        numel = (
            int(tensor.numel())
            if hasattr(tensor, "numel")
            else (int(tensor.size) if hasattr(tensor, "size") else (seq_len * 512))
        )
        total_bytes = numel * elem_sz
        row_bytes = max(1, total_bytes // max(1, seq_len))
        dev = getattr(tensor, "device", "cpu")
        stream_ptr = _stream_ptr_for_device(dev)
        is_sim = bool(getattr(self.vmm, "is_simulated", False))

        cap_view = self._get_or_grow_capacity_view(
            kind, layer_idx, seq_len, row_bytes, trailing, tensor, stream_ptr
        )
        if cap_view is None:
            self._lengths[key] = seq_len
            return tensor

        va_typed = self._tag_owner(cap_view[:seq_len])
        already_aliased = (
            hasattr(tensor, "data_ptr")
            and hasattr(va_typed, "data_ptr")
            and int(tensor.data_ptr()) == int(va_typed.data_ptr())
            and getattr(tensor, "stride", lambda: None)() == va_typed.stride()
        )
        if not already_aliased:
            self._copy_rows_into(cap_view, 0, seq_len, tensor, is_sim=is_sim)

        self._lengths[key] = seq_len
        if is_sim and _TORCH is not None and _TORCH.is_tensor(tensor) and tensor.device.type != "cpu":
            return va_typed.to(tensor.device)
        return va_typed

    def _release_slot(self, kind: str, layer_idx: int) -> None:
        key = (kind, layer_idx)
        target_map = self.key_cache if kind == "k" else self.value_cache
        cur = dict.get(target_map, layer_idx)
        eff_stream = _stream_ptr_for_device(getattr(cur, "device", "cpu")) if cur is not None else self._resolve_stream_ptr(None)
        dict.__setitem__(target_map, layer_idx, None)
        self._lengths.pop(key, None)
        self._mapped_bytes.pop(key, None)
        self._row_bytes.pop(key, None)
        self._capacity_views.pop(key, None)
        rid = self._region_ids.pop(key, None)
        self._region_caps.pop(key, None)
        if rid is not None and self.vmm is not None and not self._closed and not getattr(self.vmm, "_closed", False):
            self.vmm.kv_region_free(rid, stream_ptr=eff_stream)
        if not self._region_ids:
            self._swapped_flag = False
            self._c_evictable = False

    def close(self) -> None:
        if self._closed:
            return
        eff_stream = self._resolve_stream_ptr(None)
        self._closed = True
        self._is_pinned_active = False
        self._is_thinking = False
        self._c_evictable = False
        self._cow_parent_ref = None
        for lidx in range(self.num_layers):
            dict.__setitem__(self.key_cache, lidx, None)
            dict.__setitem__(self.value_cache, lidx, None)
        self._capacity_views.clear()
        if self.vmm is not None:
            if hasattr(self.vmm, "unregister_kv_cache"):
                try:
                    self.vmm.unregister_kv_cache(self)
                except Exception:
                    pass
            if not getattr(self.vmm, "_closed", False):
                for rid in list(self._region_ids.values()):
                    try:
                        self.vmm.kv_region_free(rid, stream_ptr=eff_stream)
                    except Exception:
                        pass
        self._region_ids.clear()
        self._region_caps.clear()
        self._mapped_bytes.clear()
        self._row_bytes.clear()
        self._lengths.clear()
        self._swapped_flag = False


__all__ = ["DEFAULT_MAX_KV_TOKENS", "VmmNaiveCache"]
