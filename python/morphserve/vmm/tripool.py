"""Python interface to C++17 `TriPoolVMM` (`CudaPageAdapter` + `HostMmapPageAdapter`) and `NumaMemfdMirror`."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
import ctypes
import gc
from pathlib import Path
import threading
from typing import Any, Callable
import weakref

from morphserve._core import (
    MsVmmConfigC,
    UnifiedVmmStatsC,
    get_core_lib,
    last_error,
    load_core_lib,
    resolve_core_so,
)

MIB = 1024**2
UnifiedVmmStats = UnifiedVmmStatsC

try:
    import torch as _TORCH
except ImportError:  # pragma: no cover - local CPU/WSL environments without PyTorch
    _TORCH = None  # type: ignore[assignment]


class _CudaVAView:
    """Zero-copy non-owning CUDA Array Interface v3 view over a raw cuMemMap VA pointer."""

    def __init__(self, ptr: int, nbytes: int, owner: Any) -> None:
        self._owner = owner
        self.ptr = int(ptr)
        self.nbytes = int(nbytes)
        self.__cuda_array_interface__ = {
            "shape": (int(nbytes),),
            "typestr": "|u1",
            "data": (int(ptr), False),
            "version": 3,
        }

    def __len__(self) -> int:
        return self.nbytes

    def __getitem__(self, item: slice) -> "_CudaVAView":
        if not isinstance(item, slice):
            raise TypeError("_CudaVAView only supports contiguous byte slicing")
        start, stop, step = item.indices(self.nbytes)
        if step != 1:
            raise ValueError("_CudaVAView requires step=1 slicing")
        length = max(0, stop - start)
        return _CudaVAView(self.ptr + start, length, self._owner)


def resolve_vmm_so(lib_dir: Path | str | None = None) -> Path:
    return resolve_core_so(lib_dir=lib_dir, prefer_vmm_alias=True, auto_build=(lib_dir is None))


def _clone_outputs(obj: Any) -> Any:
    if _TORCH is None:
        return obj
    if _TORCH.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, tuple):
        return tuple(_clone_outputs(x) for x in obj)
    if isinstance(obj, list):
        return [_clone_outputs(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _clone_outputs(v) for k, v in obj.items()}
    if hasattr(obj, "sample") and _TORCH.is_tensor(obj.sample):
        obj.sample = obj.sample.clone()
        return obj
    return obj


class NumaMemfdMirror:
    """Unified single-process & cross-process NUMA-bound, CUDA-pinned `memfd` host weight mirror (`ms_mirror_*`)."""

    def __init__(
        self,
        nbytes: int,
        numa_node: int = 0,
        name: str = "morphserve_expert",
        populate_bytes: bytes | None = None,
        *,
        _raw_handle: int | None = None,
        _is_owner: bool = True,
    ) -> None:
        self._lib = get_core_lib()
        self._handle: int = 0
        self._c_buf: Any = None
        self.mm: memoryview | None = None
        self.fd: int = -1
        self.inode: int = 0
        self.nbytes: int = int(nbytes)
        self.numa_node: int = int(numa_node)
        self.numa_bound: bool = False
        self.ptr: int = 0
        self.pinned: bool = False
        self.is_owner: bool = bool(_is_owner)

        if _raw_handle is not None:
            h = int(_raw_handle)
        else:
            if "\x00" in name:
                raise ValueError("embedded null character")
            if self.nbytes <= 0:
                raise ValueError("nbytes must be > 0")
            pop_len = 0
            pop_ptr = ctypes.c_void_p(0)
            pop_buf = None
            if populate_bytes is not None:
                raw_pop = bytes(populate_bytes)
                if len(raw_pop) > self.nbytes:
                    raise ValueError(f"populate_bytes ({len(raw_pop)}) exceeds mirror size ({self.nbytes})")
                pop_len = len(raw_pop)
                if pop_len > 0:
                    pop_buf = ctypes.create_string_buffer(raw_pop, pop_len)
                    pop_ptr = ctypes.cast(pop_buf, ctypes.c_void_p)
            h = int(
                self._lib.ms_mirror_create(
                    ctypes.c_size_t(self.nbytes),
                    ctypes.c_int(self.numa_node),
                    name.encode("utf-8"),
                    pop_ptr,
                    ctypes.c_size_t(pop_len),
                )
                or 0
            )
            del pop_buf
            if h == 0:
                err = last_error(self._lib)
                if any(k in err for k in ("memfd_create", "ftruncate", "mmap", "open(")):
                    raise OSError(err)
                raise ValueError(err)

        self._handle = h
        self.ptr = int(self._lib.ms_mirror_ptr(ctypes.c_void_p(h)))
        self.nbytes = int(self._lib.ms_mirror_nbytes(ctypes.c_void_p(h)))
        self.fd = int(self._lib.ms_mirror_fd(ctypes.c_void_p(h)))
        self.numa_node = int(self._lib.ms_mirror_numa_node(ctypes.c_void_p(h)))
        self.inode = int(self._lib.ms_mirror_inode(ctypes.c_void_p(h)))
        self.numa_bound = bool(self._lib.ms_mirror_numa_bound(ctypes.c_void_p(h)))
        self._c_buf = (ctypes.c_uint8 * self.nbytes).from_address(self.ptr)
        self.mm = memoryview(self._c_buf).cast("B")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @classmethod
    def create(
        cls,
        nbytes: int,
        numa_node: int = 0,
        name: str = "morphserve_shared_expert",
        populate_bytes: bytes | None = None,
    ) -> "NumaMemfdMirror":
        return cls(nbytes=nbytes, numa_node=numa_node, name=name, populate_bytes=populate_bytes)

    @classmethod
    def attach_proc_fd(
        cls,
        leader_pid: int,
        leader_fd: int,
        nbytes: int,
        numa_node: int = 0,
    ) -> "NumaMemfdMirror":
        L = get_core_lib()
        h = int(
            L.ms_mirror_attach_proc_fd(
                ctypes.c_int(int(leader_pid)),
                ctypes.c_int(int(leader_fd)),
                ctypes.c_size_t(int(nbytes)),
                ctypes.c_int(int(numa_node)),
            )
            or 0
        )
        if h == 0:
            raise OSError(last_error(L))
        return cls(nbytes=int(nbytes), numa_node=int(numa_node), _raw_handle=h, _is_owner=False)

    def as_uint8_tensor(self) -> Any:
        import torch

        if self._c_buf is None:
            raise RuntimeError("NumaMemfdMirror is closed")
        return torch.frombuffer(self._c_buf, dtype=torch.uint8)

    def write_bytes(self, offset: int, data: bytes | bytearray | memoryview) -> None:
        if not self._handle:
            raise RuntimeError("NumaMemfdMirror is closed")
        raw = bytes(data)
        if not raw:
            return
        buf = ctypes.create_string_buffer(raw, len(raw))
        rc = int(
            self._lib.ms_mirror_write(
                ctypes.c_void_p(self._handle),
                ctypes.c_size_t(int(offset)),
                ctypes.cast(buf, ctypes.c_void_p),
                ctypes.c_size_t(len(raw)),
            )
        )
        if rc != 0:
            raise ValueError(last_error(self._lib))

    def read_bytes(self, offset: int, nbytes: int) -> bytes:
        if not self._handle:
            raise RuntimeError("NumaMemfdMirror is closed")
        nb = max(0, int(nbytes))
        if nb == 0:
            return b""
        out = (ctypes.c_char * nb)()
        rc = int(
            self._lib.ms_mirror_read(
                ctypes.c_void_p(self._handle),
                ctypes.c_size_t(int(offset)),
                ctypes.cast(out, ctypes.c_void_p),
                ctypes.c_size_t(nb),
            )
        )
        if rc != 0:
            raise ValueError(last_error(self._lib))
        return out.raw

    def pin_cuda(self, vmm: Any = None) -> None:
        if not self._handle:
            raise RuntimeError("NumaMemfdMirror is closed")
        is_sim = bool(vmm is not None and getattr(vmm, "is_simulated", False))
        rc = int(self._lib.ms_mirror_pin_cuda_ex(ctypes.c_void_p(self._handle), 1 if is_sim else 0))
        if rc != 0:
            raise RuntimeError(f"ms_mirror_pin_cuda_ex failed: {last_error(self._lib)}")
        self.pinned = True
        if vmm is not None and hasattr(vmm, "_track_pinned_mirror"):
            vmm._track_pinned_mirror(self)

    def unpin_cuda(self) -> None:
        if self._handle:
            self._lib.ms_mirror_unpin_cuda(ctypes.c_void_p(self._handle))
        self.pinned = False

    def stream_h2d_chunks(self, dst_device_tensor: Any, chunk_bytes: int = 64 * MIB) -> float:
        import torch

        if not self.pinned:
            self.pin_cuda()
        dst_u8 = dst_device_tensor.view(torch.uint8).reshape(-1)
        total = min(self.nbytes, int(dst_u8.numel()))
        stream = torch.cuda.current_stream(dst_u8.device)
        torch.cuda.synchronize(dst_u8.device)
        gbps = float(
            self._lib.ms_mirror_stream_h2d_gbps(
                ctypes.c_void_p(self._handle),
                ctypes.c_uint64(int(dst_u8.data_ptr())),
                ctypes.c_size_t(total),
                ctypes.c_size_t(int(chunk_bytes)),
                ctypes.c_void_p(int(stream.cuda_stream)),
            )
        )
        if gbps < 0.0:
            raise RuntimeError(f"ms_mirror_stream_h2d_gbps failed: {last_error(self._lib)}")
        torch.cuda.synchronize(dst_u8.device)
        return gbps

    def close(self) -> None:
        h = getattr(self, "_handle", 0)
        if not h:
            return
        self.pinned = False
        if getattr(self, "mm", None) is not None:
            try:
                self.mm.release()
            except Exception:
                pass
            self.mm = None
        self._c_buf = None
        self._handle = 0
        self.fd = -1
        self.ptr = 0
        self._lib.ms_mirror_close(ctypes.c_void_p(h))
        self._lib.ms_mirror_destroy(ctypes.c_void_p(h))


class TriPoolVMM:
    """Instance-owned three-arena Virtual Memory Manager backed by C++17 `morphserve::vmm::TriPoolVMM`."""

    is_simulated = False

    def __init__(
        self,
        device: int = 0,
        super_pool_mib: int = 18432,
        fine_kv_pool_mib: int = 1024,
        super_handle_mib: int = 64,
        transient_va_gib: int = 16,
        kv_va_gib: int = 16,
        allow_overflow: bool = False,
        so_path: Path | str | None = None,
        *,
        simulated: bool = False,
        materialize_va_bytes: bool = False,
        kv_swap_high_watermark: float = 0.85,
        kv_swap_low_watermark: float = 0.65,
        kv_swap_min_idle_s: float = 1.0,
        max_pool_bytes: int | None = None,
        init_torch_cuda: bool = True,
    ) -> None:
        self._closed = True
        self._handle: int = 0
        self._has_torch_cuda = False
        self._pluggable: Any = None
        self._raw_transient_ptrs: list[tuple[int, int]] = []
        self._in_scope = False
        self._mu = threading.RLock()
        self._pinned_mirrors: dict[int, weakref.ReferenceType[NumaMemfdMirror]] = {}

        hi = float(kv_swap_high_watermark)
        lo = float(kv_swap_low_watermark)
        if not (0.0 < lo <= hi <= 1.0):
            raise ValueError(
                f"expected 0.0 < kv_swap_low_watermark ({lo}) <= kv_swap_high_watermark ({hi}) <= 1.0"
            )
        self.device = int(device)
        self.allow_overflow = bool(allow_overflow)
        self.super_handle_mib = int(super_handle_mib)
        self.is_simulated = bool(simulated or type(self).is_simulated)
        self.materialize_va_bytes = bool(materialize_va_bytes)
        self.kv_swap_high_watermark = hi
        self.kv_swap_low_watermark = lo
        self.kv_swap_min_idle_s = max(0.0, float(kv_swap_min_idle_s))
        self.max_pool_bytes = int(max_pool_bytes) if max_pool_bytes is not None else None

        self.lib, self.so_path = load_core_lib(so_path, prefer_vmm_alias=(not self.is_simulated))

        has_torch_cuda = bool(
            (not self.is_simulated) and init_torch_cuda and _TORCH is not None and _TORCH.cuda.is_available()
        )
        self._has_torch_cuda = has_torch_cuda
        if has_torch_cuda:
            assert _TORCH is not None
            _TORCH.cuda.set_device(self.device)
            _ = _TORCH.zeros(1, device=f"cuda:{self.device}")
            _TORCH.cuda.synchronize(self.device)

        cfg = MsVmmConfigC(
            device=self.device,
            super_pool_mib=max(0, int(super_pool_mib)),
            fine_kv_pool_mib=max(0, int(fine_kv_pool_mib)),
            super_handle_mib=max(2, int(super_handle_mib)),
            transient_va_gib=max(1, int(transient_va_gib)),
            kv_va_gib=max(1, int(kv_va_gib)),
            allow_overflow=1 if self.allow_overflow else 0,
            simulated=1 if self.is_simulated else 0,
            materialize_sim_va=1 if self.materialize_va_bytes else 0,
            kv_swap_high_watermark=self.kv_swap_high_watermark,
            kv_swap_low_watermark=self.kv_swap_low_watermark,
            kv_swap_min_idle_s=self.kv_swap_min_idle_s,
            max_pool_bytes=int(self.max_pool_bytes or 0),
        )
        h = int(self.lib.ms_vmm_create(ctypes.byref(cfg)) or 0)
        if h == 0:
            raise RuntimeError(f"ms_vmm_create failed: {last_error(self.lib)}")
        self._handle = h
        self._closed = False

        try:
            self._pluggable = (
                _TORCH.cuda.CUDAPluggableAllocator(str(self.so_path), "tripool_alloc", "tripool_free")
                if has_torch_cuda and _TORCH is not None
                else None
            )
        except BaseException:
            self.shutdown()
            raise

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            try:
                self._shutdown_impl(from_del=True)
            except Exception:
                pass

    def _torch_cuda_ready(self) -> bool:
        return bool((not self.is_simulated) and _TORCH is not None and (self._has_torch_cuda or _TORCH.cuda.is_initialized()))

    def _current_stream_ptr(self) -> int:
        if self._torch_cuda_ready():
            assert _TORCH is not None
            return int(_TORCH.cuda.current_stream(self.device).cuda_stream)
        return 0

    def stats(self) -> UnifiedVmmStats:
        st = UnifiedVmmStats()
        if self._handle and not self._closed:
            self.lib.ms_vmm_get_stats(ctypes.c_void_p(self._handle), ctypes.byref(st))
        return st

    def stats_dict(self) -> dict[str, Any]:
        d = self.stats().to_dict()
        d["physical_pool_bytes"] = int(d["pool_total_bytes"])
        return d

    def _extra_driver_overflow_bytes(self, include_overflow: bool) -> int:
        if (
            not self.is_simulated
            and include_overflow
            and self.allow_overflow
            and _TORCH is not None
            and _TORCH.cuda.is_available()
        ):
            free_drv, total_drv = _TORCH.cuda.mem_get_info(self.device)
            reserve = min(1536 * MIB, int(total_drv) // 16)
            return max(0, int(free_drv) - reserve)
        return 0

    def kv_free_bytes(self, for_dit: bool = False, include_overflow: bool = False) -> int:
        if self._closed or not self._handle:
            return 0
        extra = self._extra_driver_overflow_bytes(include_overflow)
        return int(
            self.lib.ms_vmm_kv_free_bytes(
                ctypes.c_void_p(self._handle),
                1 if for_dit else 0,
                ctypes.c_uint64(extra),
            )
        )

    def sync_kv_free_bytes(self, for_dit: bool = False, include_overflow: bool = False) -> int:
        if self._closed or not self._handle:
            return 0
        extra = self._extra_driver_overflow_bytes(include_overflow)
        return int(
            self.lib.ms_vmm_sync_kv_free_bytes(
                ctypes.c_void_p(self._handle),
                1 if for_dit else 0,
                ctypes.c_uint64(extra),
            )
        )

    # --- Session Lifecycle & Watermark Swap Governance (C++ `KvSession`) ---
    def kv_session_create(self) -> int:
        if self._closed or not self._handle:
            return 0
        return int(self.lib.ms_vmm_kv_session_create(ctypes.c_void_p(self._handle)))

    def kv_session_destroy(self, session_id: int) -> None:
        if not self._closed and self._handle and int(session_id) > 0:
            self.lib.ms_vmm_kv_session_destroy(ctypes.c_void_p(self._handle), ctypes.c_uint64(int(session_id)))

    def kv_session_touch(self, session_id: int) -> None:
        if not self._closed and self._handle and int(session_id) > 0:
            self.lib.ms_vmm_kv_session_touch(ctypes.c_void_p(self._handle), ctypes.c_uint64(int(session_id)))

    def kv_session_mark_active(self, session_id: int, stream_ptr: int = 0) -> None:
        if self._closed or not self._handle or int(session_id) <= 0:
            return
        rc = int(
            self.lib.ms_vmm_kv_session_mark_active(
                ctypes.c_void_p(self._handle),
                ctypes.c_uint64(int(session_id)),
                ctypes.c_void_p(int(stream_ptr)),
            )
        )
        if rc != 0:
            raise MemoryError(f"ms_vmm_kv_session_mark_active({session_id}) failed (rc={rc})")

    def kv_session_mark_thinking(
        self,
        session_id: int,
        region_ids: list[int],
        valid_bytes: list[int],
        *,
        trigger_watermark: bool = True,
        stream_ptr: int = 0,
    ) -> int:
        if self._closed or not self._handle:
            return 0
        n = len(region_ids)
        r_arr = (ctypes.c_int32 * n)(*(int(x) for x in region_ids)) if n > 0 else None
        v_arr = (ctypes.c_size_t * n)(*(max(0, int(x)) for x in valid_bytes)) if n > 0 else None
        return int(
            self.lib.ms_vmm_kv_session_mark_thinking(
                ctypes.c_void_p(self._handle),
                ctypes.c_uint64(int(session_id)),
                r_arr,
                v_arr,
                ctypes.c_size_t(n),
                1 if trigger_watermark else 0,
                ctypes.c_void_p(int(stream_ptr)),
            )
        )

    @staticmethod
    def _first_excluded_session_id(exclude_caches: Iterable[Any]) -> int:
        for c in exclude_caches:
            if c is not None:
                sid = int(getattr(c, "_session_id", 0) or 0)
                if sid > 0:
                    return sid
        return 0

    def reclaim_thinking_kv_bytes(
        self,
        target_reclaim_bytes: int,
        *,
        sync: bool = False,
        exclude_caches: Iterable[Any] = (),
        stream_ptr: int | None = None,
    ) -> int:
        if self._closed or not self._handle or int(target_reclaim_bytes) <= 0:
            return 0
        excl_sid = self._first_excluded_session_id(exclude_caches)
        s_ptr = int(stream_ptr) if stream_ptr is not None else self._current_stream_ptr()
        return int(
            self.lib.ms_vmm_reclaim_thinking_kv_bytes(
                ctypes.c_void_p(self._handle),
                ctypes.c_size_t(int(target_reclaim_bytes)),
                1 if sync else 0,
                ctypes.c_uint64(excl_sid),
                ctypes.c_void_p(s_ptr),
            )
        )

    def maybe_evict_thinking_kv_on_high_watermark(
        self,
        *,
        sync: bool = False,
        exclude_caches: Iterable[Any] = (),
        extra_needed_bytes: int = 0,
        for_dit: bool = False,
        stream_ptr: int | None = None,
    ) -> int:
        if self._closed or not self._handle:
            return 0
        excl_sid = self._first_excluded_session_id(exclude_caches)
        s_ptr = int(stream_ptr) if stream_ptr is not None else self._current_stream_ptr()
        return int(
            self.lib.ms_vmm_maybe_evict_watermark(
                ctypes.c_void_p(self._handle),
                1 if sync else 0,
                ctypes.c_uint64(excl_sid),
                ctypes.c_size_t(max(0, int(extra_needed_bytes))),
                1 if for_dit else 0,
                ctypes.c_void_p(s_ptr),
            )
        )

    # --- Arena 1: Persistent KV 2 MiB Paged Pool & Contiguous Regions ---
    def kv_va_base(self) -> int:
        return int(self.lib.ms_vmm_kv_va_base(ctypes.c_void_p(self._handle)))

    def kv_alloc_page(self) -> int:
        slot = int(self.lib.ms_vmm_kv_alloc_page(ctypes.c_void_p(self._handle)))
        if slot < 0:
            raise RuntimeError(f"ms_vmm_kv_alloc_page failed (rc={slot})")
        return slot

    def kv_free_page(self, slot_idx: int, stream_ptr: int = 0) -> None:
        if self._closed or not self._handle:
            return
        rc = int(
            self.lib.ms_vmm_kv_free_page(
                ctypes.c_void_p(self._handle), ctypes.c_int32(int(slot_idx)), ctypes.c_void_p(int(stream_ptr))
            )
        )
        if rc != 0:
            raise RuntimeError(f"ms_vmm_kv_free_page({slot_idx}) failed (rc={rc})")

    def kv_page_tensor(self, slot_idx: int) -> Any:
        gran = int(self.stats().min_granularity_bytes)
        ptr = self.kv_va_base() + int(slot_idx) * gran
        if self.is_simulated:
            return memoryview((ctypes.c_uint8 * gran).from_address(ptr)).cast("B")
        view = _CudaVAView(ptr, gran, self)
        if self._torch_cuda_ready():
            assert _TORCH is not None
            return _TORCH.as_tensor(view, device=f"cuda:{self.device}")
        return view

    def kv_region_alloc(self, max_va_bytes: int, *, session_id: int = 0) -> int:
        rid = int(
            self.lib.ms_vmm_kv_region_alloc(
                ctypes.c_void_p(self._handle),
                ctypes.c_size_t(int(max_va_bytes)),
                ctypes.c_uint64(max(0, int(session_id))),
            )
        )
        if rid < 0:
            raise RuntimeError(f"ms_vmm_kv_region_alloc({max_va_bytes}) failed (rc={rid})")
        return rid

    def kv_region_ensure(self, region_id: int, needed_bytes: int, stream_ptr: int = 0) -> int:
        req_b = max(1, int(needed_bytes))
        va = int(
            self.lib.ms_vmm_kv_region_ensure(
                ctypes.c_void_p(self._handle),
                ctypes.c_int32(int(region_id)),
                ctypes.c_size_t(req_b),
                ctypes.c_void_p(int(stream_ptr)),
            )
        )
        if va == 0:
            raise MemoryError(f"ms_vmm_kv_region_ensure(region={region_id}, bytes={needed_bytes}) out of VMM memory")
        return va

    def kv_region_va(self, region_id: int) -> int:
        return int(self.lib.ms_vmm_kv_region_va(ctypes.c_void_p(self._handle), ctypes.c_int32(int(region_id))))

    def kv_region_mapped_bytes(self, region_id: int) -> int:
        if self._closed or not self._handle:
            return 0
        return int(
            self.lib.ms_vmm_kv_region_mapped_bytes(ctypes.c_void_p(self._handle), ctypes.c_int32(int(region_id)))
        )

    def kv_region_uint8_view(self, region_id: int, nbytes: int | None = None) -> Any:
        if self.is_simulated and self.kv_region_is_swapped(region_id):
            raise RuntimeError(f"simulated kv_region {region_id} is currently swapped out to Host Pinned DRAM")
        ptr = self.kv_region_va(region_id)
        mapped = self.kv_region_mapped_bytes(region_id)
        target = mapped if nbytes is None else (min(mapped, int(nbytes)) if self.is_simulated else int(nbytes))
        if ptr == 0 or target <= 0:
            raise RuntimeError(f"kv_region {region_id} has no mapped VA")
        if self.is_simulated:
            return memoryview((ctypes.c_uint8 * target).from_address(ptr)).cast("B")
        view = _CudaVAView(ptr, target, self)
        if self._torch_cuda_ready():
            assert _TORCH is not None
            return _TORCH.as_tensor(view, device=f"cuda:{self.device}")
        return view

    def kv_region_set_evictable(
        self,
        region_id: int,
        evictable: bool,
        valid_bytes: int = 0,
        stream_ptr: int = 0,
    ) -> None:
        if self._closed or not self._handle:
            return
        self.lib.ms_vmm_kv_region_set_evictable(
            ctypes.c_void_p(self._handle),
            ctypes.c_int32(int(region_id)),
            1 if evictable else 0,
            ctypes.c_size_t(max(0, int(valid_bytes))),
            ctypes.c_void_p(int(stream_ptr)),
        )

    def kv_region_swap_out_async(self, region_id: int, valid_bytes: int, stream_ptr: int = 0) -> int:
        pages = int(
            self.lib.ms_vmm_kv_region_swap_out_async(
                ctypes.c_void_p(self._handle),
                ctypes.c_int32(int(region_id)),
                ctypes.c_size_t(max(0, int(valid_bytes))),
                ctypes.c_void_p(int(stream_ptr)),
            )
        )
        if pages < 0:
            raise RuntimeError(f"ms_vmm_kv_region_swap_out_async({region_id}) failed (rc={pages})")
        return pages

    kv_region_swap_out = kv_region_swap_out_async

    def kv_region_swap_in_async(self, region_id: int, stream_ptr: int = 0) -> int:
        pages = int(
            self.lib.ms_vmm_kv_region_swap_in_async(
                ctypes.c_void_p(self._handle),
                ctypes.c_int32(int(region_id)),
                ctypes.c_void_p(int(stream_ptr)),
            )
        )
        if pages < 0:
            raise MemoryError(f"ms_vmm_kv_region_swap_in_async({region_id}) failed (rc={pages})")
        return pages

    kv_region_swap_in = kv_region_swap_in_async

    def kv_region_is_swapped(self, region_id: int) -> bool:
        if self._closed or not self._handle:
            return False
        return bool(self.lib.ms_vmm_kv_region_is_swapped(ctypes.c_void_p(self._handle), ctypes.c_int32(int(region_id))))

    def kv_region_swapped_bytes(self, region_id: int) -> int:
        if self._closed or not self._handle:
            return 0
        return int(
            self.lib.ms_vmm_kv_region_swapped_bytes(ctypes.c_void_p(self._handle), ctypes.c_int32(int(region_id)))
        )

    def flush_kv_swaps(self) -> None:
        if self._closed or not self._handle:
            return
        rc = int(self.lib.ms_vmm_kv_flush_swaps(ctypes.c_void_p(self._handle)))
        if rc != 0:
            raise RuntimeError(f"ms_vmm_kv_flush_swaps failed (rc={rc})")

    def kv_region_free(self, region_id: int, stream_ptr: int = 0) -> None:
        if self._closed or not self._handle:
            return
        rc = int(
            self.lib.ms_vmm_kv_region_free(
                ctypes.c_void_p(self._handle), ctypes.c_int32(int(region_id)), ctypes.c_void_p(int(stream_ptr))
            )
        )
        if rc != 0:
            raise RuntimeError(f"ms_vmm_kv_region_free({region_id}) failed (rc={rc})")

    # --- Arena 2: Scoped Transient Scratchpad (VAE / DiT Spikes) ---
    def allocate_transient_mib(self, size_mib: int) -> int:
        if self.is_simulated and not self._in_scope:
            raise RuntimeError("allocate_transient_mib requires an active transient scope")
        nbytes = max(4096, int(size_mib) * MIB)
        s_ptr = self._current_stream_ptr()
        self.maybe_evict_thinking_kv_on_high_watermark(
            sync=self.is_simulated, extra_needed_bytes=nbytes, stream_ptr=s_ptr
        )
        ptr = int(
            self.lib.ms_vmm_transient_alloc(
                ctypes.c_void_p(self._handle), ctypes.c_ssize_t(nbytes), ctypes.c_void_p(s_ptr)
            )
            or 0
        )
        if ptr == 0:
            self.reclaim_thinking_kv_bytes(nbytes, sync=True, stream_ptr=s_ptr)
            ptr = int(
                self.lib.ms_vmm_transient_alloc(
                    ctypes.c_void_p(self._handle), ctypes.c_ssize_t(nbytes), ctypes.c_void_p(s_ptr)
                )
                or 0
            )
        if ptr == 0:
            if self.is_simulated:
                raise RuntimeError(f"SimulatedTriPoolBackend exhausted for {size_mib} MiB transient allocation")
            raise MemoryError(f"ms_vmm_transient_alloc({nbytes}) out of Arena-2 VMM memory")
        self._raw_transient_ptrs.append((ptr, nbytes))
        return ptr

    def _release_raw_transient_ptrs(self, stream_ptr: int) -> None:
        while self._raw_transient_ptrs:
            ptr, nbytes = self._raw_transient_ptrs.pop()
            self.lib.ms_vmm_transient_free(
                ctypes.c_void_p(self._handle),
                ctypes.c_void_p(ptr),
                ctypes.c_ssize_t(nbytes),
                ctypes.c_void_p(stream_ptr),
            )

    def run_in_transient_scope(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        s_ptr = self._current_stream_ptr()
        self.maybe_evict_thinking_kv_on_high_watermark(sync=False, stream_ptr=s_ptr)
        if self._pluggable is None or _TORCH is None:
            self.lib.ms_vmm_scope_begin(ctypes.c_void_p(self._handle))
            self._in_scope = True
            try:
                return fn(*args, **kwargs)
            finally:
                self._in_scope = False
                self._release_raw_transient_ptrs(s_ptr)
                self.lib.ms_vmm_scope_end(ctypes.c_void_p(self._handle), ctypes.c_void_p(s_ptr))

        pool = _TORCH.cuda.MemPool(self._pluggable.allocator())
        self.lib.ms_vmm_scope_begin(ctypes.c_void_p(self._handle))
        self._in_scope = True
        pooled_out = None
        try:
            with _TORCH.cuda.use_mem_pool(pool):
                pooled_out = fn(*args, **kwargs)
            try:
                safe_out = _clone_outputs(pooled_out)
            except BaseException:
                _TORCH.cuda.empty_cache()
                safe_out = _clone_outputs(pooled_out)
        except BaseException as exc:
            pooled_out = None
            try:
                import traceback

                if exc.__traceback__ is not None:
                    traceback.clear_frames(exc.__traceback__)
                gc.collect()
            except Exception:
                pass
            raise
        finally:
            self._in_scope = False
            pooled_out = None
            del pool
            end_s_ptr = self._current_stream_ptr()
            self._release_raw_transient_ptrs(end_s_ptr)
            self.lib.ms_vmm_scope_end(ctypes.c_void_p(self._handle), ctypes.c_void_p(end_s_ptr))
        return safe_out

    @contextmanager
    def transient_scope(self):
        s_ptr = self._current_stream_ptr()
        self.maybe_evict_thinking_kv_on_high_watermark(sync=False, stream_ptr=s_ptr)
        if self._pluggable is None or _TORCH is None:
            self.lib.ms_vmm_scope_begin(ctypes.c_void_p(self._handle))
            self._in_scope = True
            try:
                yield self
            finally:
                self._in_scope = False
                self._release_raw_transient_ptrs(s_ptr)
                self.lib.ms_vmm_scope_end(ctypes.c_void_p(self._handle), ctypes.c_void_p(s_ptr))
            return

        pool = _TORCH.cuda.MemPool(self._pluggable.allocator())
        self.lib.ms_vmm_scope_begin(ctypes.c_void_p(self._handle))
        self._in_scope = True
        try:
            with _TORCH.cuda.use_mem_pool(pool):
                yield pool
        finally:
            self._in_scope = False
            del pool
            end_s_ptr = self._current_stream_ptr()
            self._release_raw_transient_ptrs(end_s_ptr)
            self.lib.ms_vmm_scope_end(ctypes.c_void_p(self._handle), ctypes.c_void_p(end_s_ptr))

    # --- Arena 3: Opportunistic Expert Cache ---
    def register_expert(
        self,
        num_layers: int,
        layer_stride_mib: int,
        mirror: NumaMemfdMirror,
        *,
        mirror_offset: int = 0,
        mirror_nbytes: int | None = None,
    ) -> int:
        mirror.pin_cuda(self)
        off = max(0, int(mirror_offset))
        rem_bytes = max(0, int(mirror.nbytes) - off)
        eff_bytes = rem_bytes if mirror_nbytes is None else min(rem_bytes, max(0, int(mirror_nbytes)))
        base_ptr = (int(mirror.ptr) + off) if int(mirror.ptr) != 0 else 0
        eid = int(
            self.lib.ms_vmm_expert_register(
                ctypes.c_void_p(self._handle),
                int(num_layers),
                ctypes.c_size_t(int(layer_stride_mib)),
                ctypes.c_void_p(base_ptr),
                ctypes.c_size_t(eff_bytes),
            )
        )
        if eid < 0:
            raise RuntimeError(f"ms_vmm_expert_register failed (rc={eid})")
        return eid

    def expert_layer_va(self, expert_id: int, layer_idx: int) -> int:
        return int(self.lib.ms_vmm_expert_layer_va(ctypes.c_void_p(self._handle), int(expert_id), int(layer_idx)))

    def expert_layer_stride_bytes(self, expert_id: int) -> int:
        return int(self.lib.ms_vmm_expert_layer_stride_bytes(ctypes.c_void_p(self._handle), int(expert_id)))

    def expert_layer_uint8_view(self, expert_id: int, layer_idx: int) -> Any:
        if self.is_simulated:
            if not self.materialize_va_bytes:
                return memoryview(b"")
            ptr = self.expert_layer_va(expert_id, layer_idx)
            nbytes = self.expert_layer_stride_bytes(expert_id)
            return memoryview((ctypes.c_uint8 * nbytes).from_address(ptr)).cast("B")
        ptr = self.expert_layer_va(expert_id, layer_idx)
        nbytes = self.expert_layer_stride_bytes(expert_id)
        view = _CudaVAView(ptr, nbytes, self)
        if self._torch_cuda_ready():
            assert _TORCH is not None
            return _TORCH.as_tensor(view, device=f"cuda:{self.device}")
        return view

    def set_expert_pinned(self, expert_id: int, pinned: bool) -> None:
        rc = int(self.lib.ms_vmm_expert_set_pinned(ctypes.c_void_p(self._handle), int(expert_id), 1 if pinned else 0))
        if rc != 0:
            raise RuntimeError(f"ms_vmm_expert_set_pinned failed (rc={rc})")

    def set_expert_layer_used_bytes(self, expert_id: int, layer_idx: int, used_bytes: int) -> None:
        rc = int(
            self.lib.ms_vmm_expert_set_layer_used_bytes(
                ctypes.c_void_p(self._handle),
                int(expert_id),
                int(layer_idx),
                ctypes.c_size_t(int(used_bytes)),
            )
        )
        if rc != 0:
            raise RuntimeError(f"ms_vmm_expert_set_layer_used_bytes failed (rc={rc})")

    def load_expert_layers_sync(self, expert_id: int, start_layer: int, end_layer: int) -> None:
        rc = int(
            self.lib.ms_vmm_expert_load_layers_sync(
                ctypes.c_void_p(self._handle), int(expert_id), int(start_layer), int(end_layer)
            )
        )
        if rc != 0:
            raise RuntimeError(f"ms_vmm_expert_load_layers_sync failed (rc={rc})")

    def prefetch_expert_async(self, expert_id: int, start_layer: int, end_layer: int) -> int:
        queued = int(
            self.lib.ms_vmm_expert_prefetch_async(
                ctypes.c_void_p(self._handle), int(expert_id), int(start_layer), int(end_layer)
            )
        )
        if queued < 0:
            raise RuntimeError(f"ms_vmm_expert_prefetch_async failed (rc={queued})")
        return queued

    def wait_expert_layer(self, expert_id: int, layer_idx: int, stream_ptr: int = 0) -> None:
        rc = int(
            self.lib.ms_vmm_expert_wait_layer(
                ctypes.c_void_p(self._handle),
                int(expert_id),
                int(layer_idx),
                ctypes.c_void_p(int(stream_ptr)),
            )
        )
        if rc != 0:
            raise RuntimeError(f"ms_vmm_expert_wait_layer({expert_id}, {layer_idx}) failed (rc={rc})")

    def evict_expert_range(self, expert_id: int, start_layer: int, end_layer: int, stream_ptr: int = 0) -> int:
        n = int(
            self.lib.ms_vmm_expert_evict_range(
                ctypes.c_void_p(self._handle),
                int(expert_id),
                int(start_layer),
                int(end_layer),
                ctypes.c_void_p(int(stream_ptr)),
            )
        )
        if n < 0:
            raise RuntimeError(f"ms_vmm_expert_evict_range failed (rc={n})")
        return n

    def expert_resident_count(self, expert_id: int) -> int:
        return int(self.lib.ms_vmm_expert_resident_count(ctypes.c_void_p(self._handle), int(expert_id)))

    def expert_is_resident(self, expert_id: int, layer_idx: int) -> bool:
        return bool(self.lib.ms_vmm_expert_is_resident(ctypes.c_void_p(self._handle), int(expert_id), int(layer_idx)))

    def _track_pinned_mirror(self, mirror: NumaMemfdMirror) -> None:
        with self._mu:
            self._pinned_mirrors[id(mirror)] = weakref.ref(mirror)

    def _shutdown_impl(self, *, from_del: bool = False) -> None:
        if getattr(self, "_closed", True):
            return
        rc = 0
        with self._mu:
            if self._closed:
                return
            self._closed = True
            h = self._handle
            self._handle = 0
            self._pluggable = None
            mirrors = list(self._pinned_mirrors.values())
            self._pinned_mirrors.clear()

        for ref in mirrors:
            m_obj = ref()
            if m_obj is not None:
                try:
                    m_obj.unpin_cuda()
                except Exception:
                    pass
        if not from_del and gc is not None:
            gc.collect()

        if h:
            while self._raw_transient_ptrs:
                ptr, nbytes = self._raw_transient_ptrs.pop()
                self.lib.ms_vmm_transient_free(
                    ctypes.c_void_p(h), ctypes.c_void_p(ptr), ctypes.c_ssize_t(nbytes), ctypes.c_void_p(0)
                )
            rc = int(self.lib.ms_vmm_shutdown(ctypes.c_void_p(h)))
            self.lib.ms_vmm_destroy(ctypes.c_void_p(h))

        if rc != 0 and not from_del:
            raise RuntimeError(f"ms_vmm_shutdown failed (rc={rc})")

    def shutdown(self) -> None:
        self._shutdown_impl(from_del=False)


class SimulatedTriPoolBackend(TriPoolVMM):
    """Local-substitutable CPU VMM backend executing identical C++17 `TriPoolVMM` via `HostMmapPageAdapter`."""

    is_simulated = True

    def __init__(
        self,
        device: int = 0,
        super_pool_mib: int = 14336,
        fine_kv_pool_mib: int = 128,
        super_handle_mib: int = 64,
        allow_overflow: bool = False,
        materialize_va_bytes: bool = False,
        *,
        kv_swap_high_watermark: float = 0.85,
        kv_swap_low_watermark: float = 0.65,
        kv_swap_min_idle_s: float = 1.0,
        max_pool_bytes: int | None = None,
    ) -> None:
        pool_gib = max(16, (max(0, int(super_pool_mib)) + max(0, int(fine_kv_pool_mib)) + 2047) // 1024)
        super().__init__(
            device=device,
            super_pool_mib=super_pool_mib,
            fine_kv_pool_mib=fine_kv_pool_mib,
            super_handle_mib=super_handle_mib,
            transient_va_gib=pool_gib,
            kv_va_gib=16,
            allow_overflow=allow_overflow,
            simulated=True,
            materialize_va_bytes=materialize_va_bytes,
            kv_swap_high_watermark=kv_swap_high_watermark,
            kv_swap_low_watermark=kv_swap_low_watermark,
            kv_swap_min_idle_s=kv_swap_min_idle_s,
            max_pool_bytes=max_pool_bytes,
            init_torch_cuda=False,
        )
