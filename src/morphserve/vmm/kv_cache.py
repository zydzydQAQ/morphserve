"""Contiguous-VA `VmmNaiveCache` backed by `TriPoolVMM` Arena-1 `2 MiB` physical pages."""

from __future__ import annotations

from typing import Any

MIB = 1024**2
DEFAULT_MAX_KV_TOKENS = 32_768


def _stream_ptr_for_device(device: Any) -> int:
    try:
        import torch

        if torch.cuda.is_available() and str(device).startswith("cuda"):
            return int(torch.cuda.current_stream(device).cuda_stream)
    except Exception:
        pass
    return 0


class _VmmLayerTensorMap(dict):
    """Custom per-layer tensor dict intercepting `NaiveCache` `torch.cat` assignments in-place."""

    def __init__(self, owner: "VmmNaiveCache", kind: str) -> None:
        super().__init__()
        self._owner = owner
        self._kind = kind  # "k" | "v"
        for lidx in range(owner.num_layers):
            super().__setitem__(lidx, None)

    def copy(self) -> dict[int, Any]:
        return {k: v for k, v in self.items()}

    def __setitem__(self, layer_idx: int, value: Any) -> None:
        lidx = int(layer_idx)
        if value is None:
            self._owner._release_slot(self._kind, lidx)
            super().__setitem__(lidx, None)
            return
        if self._owner.vmm is None:
            super().__setitem__(lidx, value)
            return
        bound = self._owner._store_layer_tensor(self._kind, lidx, value, super().get(lidx))
        super().__setitem__(lidx, bound)


class VmmNaiveCache:
    """Drop-in `vllm_omni` `NaiveCache` replacement growing K/V tensors in-place on `TriPoolVMM` Arena 1.

    Key properties:
      - Reserves contiguous virtual address space per `(layer, K/V)` (`0 B` physical RAM initially).
      - Maps `2 MiB` physical pages (`2,048` tokens/page for BAGEL `4x128 bf16`) strictly on demand,
        automatically carving `64 MiB -> 32 x 2 MiB` handles and evicting unpinned tail expert slices.
      - Intercepts `key_cache[l] = torch.cat([old, new], dim=0)` and copies ONLY the newly appended
        suffix `[old_len : new_len]` into fixed VA memory, freeing temporary `torch.cat` buffers
        immediately and preserving base-pointer identity across `cow_fork_context` branches.
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
        self._lengths: dict[tuple[str, int], int] = {}
        self._closed = False

        self.key_cache = _VmmLayerTensorMap(self, "k")
        self.value_cache = _VmmLayerTensorMap(self, "v")

    @property
    def num_layers(self) -> int:
        return self._num_layers

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
        dup = type(self)(self.num_layers, vmm=self.vmm, max_seq_len=self.max_seq_len)
        memo[id(self)] = dup
        dup.key_values_lens = list(self.key_values_lens) if self.key_values_lens is not None else None
        dup._cow_shared_prefix_len = self._cow_shared_prefix_len
        dup._cow_origin_id = self._cow_origin_id
        for lidx in range(self.num_layers):
            k = self.key_cache.get(lidx)
            v = self.value_cache.get(lidx)
            if k is not None:
                dup.key_cache[lidx] = k.clone() if (self.vmm is None and hasattr(k, "clone")) else k
            if v is not None:
                dup.value_cache[lidx] = v.clone() if (self.vmm is None and hasattr(v, "clone")) else v
        return dup

    @property
    def seq_lens(self) -> int:
        k0 = self.key_cache.get(0)
        return int(k0.shape[0]) if k0 is not None else 0

    @property
    def mapped_bytes(self) -> int:
        if self._closed or self.vmm is None:
            return 0
        return sum(self.vmm.kv_region_mapped_bytes(rid) for rid in self._region_ids.values())

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

    def _ensure_region(self, kind: str, layer_idx: int, needed_bytes: int, row_bytes: int, stream_ptr: int) -> int:
        key = (kind, layer_idx)
        rid = self._region_ids.get(key)
        if rid is None:
            cap = max(self.default_max_va_bytes, self.max_seq_len * max(1, row_bytes), needed_bytes)
            rid = self.vmm.kv_region_alloc(cap)
            self._region_ids[key] = rid
            self._region_caps[key] = cap
        self.vmm.kv_region_ensure(rid, max(1, needed_bytes), stream_ptr=stream_ptr)
        return rid

    def _store_layer_tensor(self, kind: str, layer_idx: int, tensor: Any, prev_view: Any | None) -> Any:
        # Preserve external read-only views (e.g., CoW forks pointing to another VmmNaiveCache's VA)
        if getattr(tensor, "_vmm_external_ro", False):
            return tensor

        seq_len = int(tensor.shape[0])
        if seq_len == 0:
            return tensor

        elem_sz = int(tensor.element_size()) if hasattr(tensor, "element_size") else 2
        numel = int(tensor.numel()) if hasattr(tensor, "numel") else (seq_len * 512)
        total_bytes = numel * elem_sz
        row_bytes = max(1, total_bytes // max(1, seq_len))
        dev = getattr(tensor, "device", "cpu")
        stream_ptr = _stream_ptr_for_device(dev)

        rid = self._ensure_region(kind, layer_idx, total_bytes, row_bytes, stream_ptr)
        mapped_b = self.vmm.kv_region_mapped_bytes(rid)
        raw_u8 = self.vmm.kv_region_uint8_view(rid, mapped_b)

        is_sim = bool(getattr(self.vmm, "is_simulated", False))
        key = (kind, layer_idx)
        old_len = self._lengths.get(key, 0)

        if not is_sim:
            # Production CUDA path: zero-copy Tensor view over fixed cuMemMap VA
            va_typed = raw_u8[:total_bytes].view(tensor.dtype).reshape(tensor.shape)
            same_va = (
                prev_view is not None
                and old_len > 0
                and seq_len >= old_len
                and hasattr(prev_view, "data_ptr")
                and int(prev_view.data_ptr()) == int(va_typed.data_ptr())
            )
            if same_va and hasattr(tensor, "data_ptr") and int(tensor.data_ptr()) == int(va_typed.data_ptr()):
                self._lengths[key] = seq_len
                return va_typed
            if same_va and seq_len > old_len:
                va_typed[old_len:seq_len].copy_(tensor[old_len:seq_len])
            elif not same_va or seq_len != old_len:
                va_typed.copy_(tensor)
            self._lengths[key] = seq_len
            return va_typed

        # SimulatedTriPoolBackend path (supports real CPU torch.Tensor or duck-typed mock tensors)
        try:
            import torch

            if torch.is_tensor(tensor):
                va_tensor = torch.frombuffer(raw_u8, dtype=torch.uint8, count=total_bytes).view(tensor.dtype).reshape(
                    tensor.shape
                )
                same_va = (
                    prev_view is not None
                    and torch.is_tensor(prev_view)
                    and old_len > 0
                    and seq_len >= old_len
                    and int(prev_view.data_ptr()) == int(va_tensor.data_ptr())
                )
                if same_va and int(tensor.data_ptr()) == int(va_tensor.data_ptr()):
                    self._lengths[key] = seq_len
                    return va_tensor
                if same_va and seq_len > old_len:
                    va_tensor[old_len:seq_len].copy_(tensor[old_len:seq_len].detach().cpu())
                else:
                    va_tensor.copy_(tensor.detach().cpu())
                self._lengths[key] = seq_len
                return va_tensor
        except ImportError:
            pass

        self._lengths[key] = seq_len
        return tensor

    def _release_slot(self, kind: str, layer_idx: int) -> None:
        key = (kind, layer_idx)
        self._lengths.pop(key, None)
        rid = self._region_ids.pop(key, None)
        self._region_caps.pop(key, None)
        if rid is not None and self.vmm is not None and not self._closed:
            self.vmm.kv_region_free(rid)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.vmm is not None:
            for rid in list(self._region_ids.values()):
                try:
                    self.vmm.kv_region_free(rid)
                except Exception:
                    pass
        self._region_ids.clear()
        self._region_caps.clear()
        self._lengths.clear()
        for lidx in range(self.num_layers):
            dict.__setitem__(self.key_cache, lidx, None)
            dict.__setitem__(self.value_cache, lidx, None)


__all__ = ["DEFAULT_MAX_KV_TOKENS", "VmmNaiveCache"]
