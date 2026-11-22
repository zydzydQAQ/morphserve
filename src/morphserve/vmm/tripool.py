"""Python interface to libtripool_vmm.so (TriPool-VMM), local-substitutable VMM simulator, and unified NUMA memfd mirror."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import gc
import mmap
import os
from pathlib import Path
import time
from typing import Any, Callable

MIB = 1024**2
CU_MEMHOSTREGISTER_PORTABLE = 0x1


class UnifiedVmmStats(ctypes.Structure):
    _fields_ = [
        ("min_granularity_bytes", ctypes.c_uint64),
        ("super_handle_bytes", ctypes.c_uint64),
        ("super_pool_total_handles", ctypes.c_uint64),
        ("super_pool_free_handles", ctypes.c_uint64),
        ("fine_pool_total_pages", ctypes.c_uint64),
        ("fine_pool_free_pages", ctypes.c_uint64),
        ("kv_mapped_pages_current", ctypes.c_uint64),
        ("kv_mapped_pages_peak", ctypes.c_uint64),
        ("super_to_fine_carves", ctypes.c_uint64),
        ("transient_mapped_handles_current", ctypes.c_uint64),
        ("transient_mapped_handles_peak", ctypes.c_uint64),
        ("transient_live_allocs", ctypes.c_uint64),
        ("transient_alloc_calls", ctypes.c_uint64),
        ("transient_free_calls", ctypes.c_uint64),
        ("transient_scope_flushes", ctypes.c_uint64),
        ("expert_mapped_handles_current", ctypes.c_uint64),
        ("expert_resident_layers_total", ctypes.c_uint64),
        ("expert_prefetch_layers_completed", ctypes.c_uint64),
        ("expert_wait_calls", ctypes.c_uint64),
        ("expert_wait_stalls", ctypes.c_uint64),
        ("expert_manual_evictions", ctypes.c_uint64),
        ("expert_auto_evictions_for_transient", ctypes.c_uint64),
        ("expert_auto_evictions_for_kv", ctypes.c_uint64),
        ("expert_h2d_bytes_total", ctypes.c_uint64),
        ("overflow_creates", ctypes.c_uint64),
        ("transient_map_us", ctypes.c_double),
        ("transient_unmap_us", ctypes.c_double),
        ("expert_map_us", ctypes.c_double),
        ("expert_unmap_us", ctypes.c_double),
        ("expert_h2d_submit_us", ctypes.c_double),
    ]

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name, _ in self._fields_}


class _CudaVAView:
    """Zero-copy non-owning CUDA Array Interface v3 view over a raw cuMemMap VA pointer."""

    def __init__(self, ptr: int, nbytes: int, owner: Any) -> None:
        self._owner = owner
        self.__cuda_array_interface__ = {
            "shape": (int(nbytes),),
            "typestr": "|u1",
            "data": (int(ptr), False),
            "version": 3,
        }


def resolve_vmm_so(lib_dir: Path | str | None = None) -> Path:
    candidates = []
    if lib_dir is not None:
        candidates.append(Path(lib_dir) / "libtripool_vmm.so")
    env_dir = os.environ.get("MORPHSERVE_LIB_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / "libtripool_vmm.so")
    repo_guess = Path(__file__).resolve().parents[3] / "build/install/lib/libtripool_vmm.so"
    candidates.append(repo_guess)
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"libtripool_vmm.so not found in {[str(c) for c in candidates]}")


class _HostCudaPin:
    """Direct libcuda fallback for portable host pinning when TriPoolVMM is not yet instantiated."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.cuMemHostRegister = getattr(self.lib, "cuMemHostRegister_v2", self.lib.cuMemHostRegister)
        self.cuMemHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        self.cuMemHostRegister.restype = ctypes.c_int

        self.cuMemHostUnregister = self.lib.cuMemHostUnregister
        self.cuMemHostUnregister.argtypes = [ctypes.c_void_p]
        self.cuMemHostUnregister.restype = ctypes.c_int

    def register(self, ptr: int, nbytes: int) -> None:
        rc = int(self.cuMemHostRegister(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), CU_MEMHOSTREGISTER_PORTABLE))
        if rc != 0:
            raise RuntimeError(f"cuMemHostRegister(ptr=0x{ptr:x}, nbytes={nbytes}) failed with CUresult={rc}")

    def unregister(self, ptr: int) -> None:
        rc = int(self.cuMemHostUnregister(ctypes.c_void_p(ptr)))
        if rc != 0:
            raise RuntimeError(f"cuMemHostUnregister failed with CUresult={rc}")


def _bind_numa_memory(ptr: int, nbytes: int, numa_node: int) -> bool:
    try:
        libnuma = ctypes.CDLL("libnuma.so.1")
        libnuma.numa_available.restype = ctypes.c_int
        if int(libnuma.numa_available()) < 0:
            return False
        libnuma.numa_tonode_memory.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libnuma.numa_tonode_memory.restype = None
        libnuma.numa_tonode_memory(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), int(numa_node))
        return True
    except Exception:
        return False


class NumaMemfdMirror:
    """Unified single-process & cross-process NUMA-bound, CUDA-pinned `memfd` host weight mirror.

    Supports both standalone creation (`NumaMemfdMirror(...)` / `NumaMemfdMirror.create(...)`)
    and multi-process per-NUMA-socket sharing (`NumaMemfdMirror.attach_proc_fd(...)` via
    `/proc/<leader_pid>/fd/<fd>` + `MAP_SHARED`), bypassing `/dev/shm` 32 GiB caps.
    """

    def __init__(
        self,
        nbytes: int,
        numa_node: int = 0,
        name: str = "morphserve_expert",
        populate_bytes: bytes | None = None,
        *,
        _attached_fd: int | None = None,
        _attached_mm: mmap.mmap | None = None,
        _is_owner: bool = True,
    ) -> None:
        self.nbytes = int(nbytes)
        self.numa_node = int(numa_node)
        self.is_owner = bool(_is_owner)

        if _attached_fd is not None and _attached_mm is not None:
            self.fd = int(_attached_fd)
            self.mm = _attached_mm
            self.numa_bound = False
        else:
            memfd_fn = getattr(os, "memfd_create", None)
            if memfd_fn is None:
                raise RuntimeError("os.memfd_create is required")
            fd = int(memfd_fn(name, 0))
            try:
                os.ftruncate(fd, self.nbytes)
                mm = mmap.mmap(fd, self.nbytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
            except Exception:
                os.close(fd)
                raise
            self.fd = fd
            self.mm = mm
            # Touch 1 byte per 2 MiB stride so kernel faults backing pages cleanly before NUMA bind
            stride = 2 * MIB
            for off in range(0, self.nbytes, stride):
                self.mm[off] = 0
            if populate_bytes:
                mv = memoryview(self.mm)
                mv[: len(populate_bytes)] = populate_bytes

        self.inode = int(os.fstat(self.fd).st_ino)
        self._c_buf = (ctypes.c_uint8 * self.nbytes).from_buffer(self.mm)
        self.ptr = int(ctypes.addressof(self._c_buf))
        if _attached_fd is None:
            self.numa_bound = _bind_numa_memory(self.ptr, self.nbytes, self.numa_node)

        self._pinned_vmm_lib: Any = None
        self._pin_api: _HostCudaPin | None = None
        self.pinned = False

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
        proc_path = f"/proc/{int(leader_pid)}/fd/{int(leader_fd)}"
        fd = os.open(proc_path, os.O_RDWR)
        try:
            mm = mmap.mmap(fd, int(nbytes), mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        except Exception:
            os.close(fd)
            raise
        return cls(
            nbytes=int(nbytes),
            numa_node=numa_node,
            _attached_fd=fd,
            _attached_mm=mm,
            _is_owner=False,
        )

    def as_uint8_tensor(self) -> Any:
        import torch

        return torch.frombuffer(self._c_buf, dtype=torch.uint8)

    def write_bytes(self, offset: int, data: bytes | bytearray | memoryview) -> None:
        mv = memoryview(self.mm)
        mv[offset : offset + len(data)] = data

    def read_bytes(self, offset: int, nbytes: int) -> bytes:
        return bytes(self.mm[offset : offset + nbytes])

    def pin_cuda(self, vmm: Any = None) -> None:
        if self.pinned:
            return
        if vmm is not None and getattr(vmm, "is_simulated", False):
            self.pinned = True
            return
        if vmm is not None and getattr(vmm, "lib", None) is not None:
            rc = int(vmm.lib.tripool_host_register(ctypes.c_void_p(self.ptr), ctypes.c_size_t(self.nbytes)))
            if rc != 0:
                raise RuntimeError(f"tripool_host_register failed (rc={rc}) for {self.nbytes} bytes")
            self._pinned_vmm_lib = vmm.lib
            self.pinned = True
            return
        if self._pin_api is None:
            self._pin_api = _HostCudaPin()
        self._pin_api.register(self.ptr, self.nbytes)
        self.pinned = True

    def stream_h2d_chunks(self, dst_device_tensor: Any, chunk_bytes: int = 64 * MIB) -> float:
        """Copy shared pinned mirror -> GPU tensor via CUDA H2D and return achieved GB/s."""
        import torch

        if not self.pinned:
            self.pin_cuda()
        if self._pin_api is None:
            self._pin_api = _HostCudaPin()
        dst_u8 = dst_device_tensor.view(torch.uint8).reshape(-1)
        total = min(self.nbytes, int(dst_u8.numel()))
        lib = self._pin_api.lib
        h2d_fn = getattr(lib, "cuMemcpyHtoDAsync_v2", lib.cuMemcpyHtoDAsync)
        h2d_fn.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        h2d_fn.restype = ctypes.c_int

        stream = torch.cuda.current_stream(dst_u8.device)
        torch.cuda.synchronize(dst_u8.device)
        t0 = time.perf_counter()
        off = 0
        dst_base = int(dst_u8.data_ptr())
        while off < total:
            sz = min(chunk_bytes, total - off)
            rc = int(
                h2d_fn(
                    ctypes.c_uint64(dst_base + off),
                    ctypes.c_void_p(self.ptr + off),
                    ctypes.c_size_t(sz),
                    ctypes.c_void_p(int(stream.cuda_stream)),
                )
            )
            if rc != 0:
                raise RuntimeError(f"cuMemcpyHtoDAsync failed CUresult={rc}")
            off += sz
        torch.cuda.synchronize(dst_u8.device)
        dt_s = max(time.perf_counter() - t0, 1e-6)
        return (total / 1e9) / dt_s

    def close(self) -> None:
        if self.pinned:
            if self._pinned_vmm_lib is not None:
                try:
                    self._pinned_vmm_lib.tripool_host_unregister(ctypes.c_void_p(self.ptr))
                except Exception:
                    pass
                self._pinned_vmm_lib = None
            elif self._pin_api is not None:
                try:
                    self._pin_api.unregister(self.ptr)
                except Exception:
                    pass
            self.pinned = False
        if getattr(self, "_c_buf", None) is not None:
            del self._c_buf
            self._c_buf = None
        if getattr(self, "mm", None) is not None:
            self.mm.close()
            self.mm = None
        if getattr(self, "fd", -1) >= 0:
            os.close(self.fd)
            self.fd = -1


def _clone_outputs(obj: Any) -> Any:
    import torch

    if torch.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, tuple):
        return tuple(_clone_outputs(x) for x in obj)
    if isinstance(obj, list):
        return [_clone_outputs(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _clone_outputs(v) for k, v in obj.items()}
    if hasattr(obj, "sample") and torch.is_tensor(obj.sample):
        obj.sample = obj.sample.clone()
        return obj
    return obj


class TriPoolVMM:
    """Unified three-arena CUDA Virtual Memory Manager backed by libtripool_vmm.so."""

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
    ) -> None:
        import torch

        self.device = int(device)
        self.so_path = Path(so_path) if so_path is not None else resolve_vmm_so()
        self.lib = ctypes.CDLL(str(self.so_path))
        self._bind_symbols()

        torch.cuda.set_device(self.device)
        _ = torch.zeros(1, device=f"cuda:{self.device}")
        torch.cuda.synchronize(self.device)

        rc = self.lib.tripool_vmm_init(
            self.device,
            int(super_pool_mib),
            int(fine_kv_pool_mib),
            int(super_handle_mib),
            int(transient_va_gib),
            int(kv_va_gib),
            1 if allow_overflow else 0,
        )
        if rc != 0:
            raise RuntimeError(f"tripool_vmm_init failed with rc={rc}")

        self._pluggable = torch.cuda.CUDAPluggableAllocator(
            str(self.so_path), "tripool_alloc", "tripool_free"
        )
        self._closed = False

    def _bind_symbols(self) -> None:
        L = self.lib
        L.tripool_vmm_init.argtypes = [
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        L.tripool_vmm_init.restype = ctypes.c_int
        L.tripool_vmm_shutdown.argtypes = []
        L.tripool_vmm_shutdown.restype = ctypes.c_int
        L.tripool_vmm_get_stats.argtypes = [ctypes.POINTER(UnifiedVmmStats)]
        L.tripool_vmm_get_stats.restype = None

        L.tripool_scope_begin.argtypes = []
        L.tripool_scope_begin.restype = None
        L.tripool_scope_end.argtypes = [ctypes.c_void_p]
        L.tripool_scope_end.restype = ctypes.c_int

        L.tripool_kv_va_base.argtypes = []
        L.tripool_kv_va_base.restype = ctypes.c_uint64
        L.tripool_kv_alloc_page.argtypes = []
        L.tripool_kv_alloc_page.restype = ctypes.c_int32
        L.tripool_kv_free_page.argtypes = [ctypes.c_int32, ctypes.c_void_p]
        L.tripool_kv_free_page.restype = ctypes.c_int
        L.tripool_kv_region_alloc.argtypes = [ctypes.c_size_t]
        L.tripool_kv_region_alloc.restype = ctypes.c_int32
        L.tripool_kv_region_ensure.argtypes = [ctypes.c_int32, ctypes.c_size_t, ctypes.c_void_p]
        L.tripool_kv_region_ensure.restype = ctypes.c_uint64
        L.tripool_kv_region_va.argtypes = [ctypes.c_int32]
        L.tripool_kv_region_va.restype = ctypes.c_uint64
        L.tripool_kv_region_mapped_bytes.argtypes = [ctypes.c_int32]
        L.tripool_kv_region_mapped_bytes.restype = ctypes.c_size_t
        L.tripool_kv_region_free.argtypes = [ctypes.c_int32, ctypes.c_void_p]
        L.tripool_kv_region_free.restype = ctypes.c_int

        L.tripool_expert_register.argtypes = [
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        L.tripool_expert_register.restype = ctypes.c_int
        L.tripool_expert_va_base.argtypes = [ctypes.c_int]
        L.tripool_expert_va_base.restype = ctypes.c_uint64
        L.tripool_expert_layer_va.argtypes = [ctypes.c_int, ctypes.c_int]
        L.tripool_expert_layer_va.restype = ctypes.c_uint64
        L.tripool_expert_layer_stride_bytes.argtypes = [ctypes.c_int]
        L.tripool_expert_layer_stride_bytes.restype = ctypes.c_size_t

        L.tripool_expert_set_pinned.argtypes = [ctypes.c_int, ctypes.c_int]
        L.tripool_expert_set_pinned.restype = ctypes.c_int
        L.tripool_expert_set_layer_used_bytes.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_size_t]
        L.tripool_expert_set_layer_used_bytes.restype = ctypes.c_int
        L.tripool_expert_load_layers_sync.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        L.tripool_expert_load_layers_sync.restype = ctypes.c_int
        L.tripool_expert_prefetch_async.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        L.tripool_expert_prefetch_async.restype = ctypes.c_int
        L.tripool_expert_wait_layer.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        L.tripool_expert_wait_layer.restype = ctypes.c_int
        L.tripool_expert_evict_layer.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        L.tripool_expert_evict_layer.restype = ctypes.c_int
        L.tripool_expert_evict_range.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        L.tripool_expert_evict_range.restype = ctypes.c_int
        L.tripool_expert_is_resident.argtypes = [ctypes.c_int, ctypes.c_int]
        L.tripool_expert_is_resident.restype = ctypes.c_int
        L.tripool_expert_resident_count.argtypes = [ctypes.c_int]
        L.tripool_expert_resident_count.restype = ctypes.c_int

        L.tripool_host_register.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        L.tripool_host_register.restype = ctypes.c_int
        L.tripool_host_unregister.argtypes = [ctypes.c_void_p]
        L.tripool_host_unregister.restype = ctypes.c_int

    def stats(self) -> UnifiedVmmStats:
        st = UnifiedVmmStats()
        self.lib.tripool_vmm_get_stats(ctypes.byref(st))
        return st

    def stats_dict(self) -> dict[str, Any]:
        return self.stats().to_dict()

    # --- Arena 1: Persistent KV 2 MiB Paged Pool ---
    def kv_va_base(self) -> int:
        return int(self.lib.tripool_kv_va_base())

    def kv_alloc_page(self) -> int:
        slot = int(self.lib.tripool_kv_alloc_page())
        if slot < 0:
            raise RuntimeError(f"tripool_kv_alloc_page failed (rc={slot})")
        return slot

    def kv_free_page(self, slot_idx: int, stream_ptr: int = 0) -> None:
        rc = int(self.lib.tripool_kv_free_page(int(slot_idx), ctypes.c_void_p(stream_ptr)))
        if rc != 0:
            raise RuntimeError(f"tripool_kv_free_page({slot_idx}) failed (rc={rc})")

    def kv_page_tensor(self, slot_idx: int) -> Any:
        import torch

        gran = self.stats().min_granularity_bytes
        ptr = self.kv_va_base() + int(slot_idx) * gran
        return torch.as_tensor(_CudaVAView(ptr, gran, self), device=f"cuda:{self.device}")

    def kv_region_alloc(self, max_va_bytes: int) -> int:
        rid = int(self.lib.tripool_kv_region_alloc(ctypes.c_size_t(int(max_va_bytes))))
        if rid < 0:
            raise RuntimeError(f"tripool_kv_region_alloc({max_va_bytes}) failed (rc={rid})")
        return rid

    def kv_region_ensure(self, region_id: int, needed_bytes: int, stream_ptr: int = 0) -> int:
        va = int(
            self.lib.tripool_kv_region_ensure(
                ctypes.c_int32(int(region_id)),
                ctypes.c_size_t(max(1, int(needed_bytes))),
                ctypes.c_void_p(int(stream_ptr)),
            )
        )
        if va == 0:
            raise MemoryError(
                f"tripool_kv_region_ensure(region={region_id}, bytes={needed_bytes}) out of VMM memory"
            )
        return va

    def kv_region_va(self, region_id: int) -> int:
        return int(self.lib.tripool_kv_region_va(ctypes.c_int32(int(region_id))))

    def kv_region_mapped_bytes(self, region_id: int) -> int:
        return int(self.lib.tripool_kv_region_mapped_bytes(ctypes.c_int32(int(region_id))))

    def kv_region_uint8_view(self, region_id: int, nbytes: int | None = None) -> Any:
        import torch

        ptr = self.kv_region_va(region_id)
        mapped = self.kv_region_mapped_bytes(region_id) if nbytes is None else int(nbytes)
        if ptr == 0 or mapped <= 0:
            raise RuntimeError(f"kv_region {region_id} has no mapped VA")
        return torch.as_tensor(_CudaVAView(ptr, mapped, self), device=f"cuda:{self.device}")

    def kv_region_free(self, region_id: int, stream_ptr: int = 0) -> None:
        rc = int(self.lib.tripool_kv_region_free(ctypes.c_int32(int(region_id)), ctypes.c_void_p(int(stream_ptr))))
        if rc != 0:
            raise RuntimeError(f"tripool_kv_region_free({region_id}) failed (rc={rc})")

    # --- Arena 2: Scoped Transient Scratchpad (VAE / DiT Spikes) ---
    def run_in_transient_scope(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run `fn` inside a scoped TriPool MemPool, clone outputs to default allocator, bulk-unmap immediately."""
        import torch

        pool = torch.cuda.MemPool(self._pluggable.allocator())
        self.lib.tripool_scope_begin()
        pooled_out = None
        try:
            with torch.cuda.use_mem_pool(pool):
                pooled_out = fn(*args, **kwargs)
            safe_out = _clone_outputs(pooled_out)
        finally:
            pooled_out = None
            del pool
            stream_ptr = ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
            self.lib.tripool_scope_end(stream_ptr)
        return safe_out

    @contextmanager
    def transient_scope(self):
        """Low-level context manager; caller must not retain pooled tensors past exit."""
        import torch

        pool = torch.cuda.MemPool(self._pluggable.allocator())
        self.lib.tripool_scope_begin()
        try:
            with torch.cuda.use_mem_pool(pool):
                yield pool
        finally:
            del pool
            stream_ptr = ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
            self.lib.tripool_scope_end(stream_ptr)

    # --- Arena 3: Opportunistic Expert Cache ---
    def register_expert(self, num_layers: int, layer_stride_mib: int, mirror: NumaMemfdMirror) -> int:
        mirror.pin_cuda(self)
        eid = int(
            self.lib.tripool_expert_register(
                int(num_layers),
                int(layer_stride_mib),
                ctypes.c_void_p(mirror.ptr),
                ctypes.c_size_t(mirror.nbytes),
            )
        )
        if eid < 0:
            raise RuntimeError(f"tripool_expert_register failed (rc={eid})")
        return eid

    def expert_layer_va(self, expert_id: int, layer_idx: int) -> int:
        return int(self.lib.tripool_expert_layer_va(int(expert_id), int(layer_idx)))

    def expert_layer_stride_bytes(self, expert_id: int) -> int:
        return int(self.lib.tripool_expert_layer_stride_bytes(int(expert_id)))

    def expert_layer_uint8_view(self, expert_id: int, layer_idx: int) -> Any:
        import torch

        ptr = self.expert_layer_va(expert_id, layer_idx)
        nbytes = self.expert_layer_stride_bytes(expert_id)
        return torch.as_tensor(_CudaVAView(ptr, nbytes, self), device=f"cuda:{self.device}")

    def set_expert_pinned(self, expert_id: int, pinned: bool) -> None:
        rc = int(self.lib.tripool_expert_set_pinned(int(expert_id), 1 if pinned else 0))
        if rc != 0:
            raise RuntimeError(f"tripool_expert_set_pinned failed (rc={rc})")

    def set_expert_layer_used_bytes(self, expert_id: int, layer_idx: int, used_bytes: int) -> None:
        rc = int(
            self.lib.tripool_expert_set_layer_used_bytes(
                int(expert_id), int(layer_idx), ctypes.c_size_t(int(used_bytes))
            )
        )
        if rc != 0:
            raise RuntimeError(f"tripool_expert_set_layer_used_bytes failed (rc={rc})")

    def load_expert_layers_sync(self, expert_id: int, start_layer: int, end_layer: int) -> None:
        rc = int(self.lib.tripool_expert_load_layers_sync(int(expert_id), int(start_layer), int(end_layer)))
        if rc != 0:
            raise RuntimeError(f"tripool_expert_load_layers_sync failed (rc={rc})")

    def prefetch_expert_async(self, expert_id: int, start_layer: int, end_layer: int) -> int:
        queued = int(self.lib.tripool_expert_prefetch_async(int(expert_id), int(start_layer), int(end_layer)))
        if queued < 0:
            raise RuntimeError(f"tripool_expert_prefetch_async failed (rc={queued})")
        return queued

    def wait_expert_layer(self, expert_id: int, layer_idx: int, stream_ptr: int = 0) -> None:
        rc = int(self.lib.tripool_expert_wait_layer(int(expert_id), int(layer_idx), ctypes.c_void_p(stream_ptr)))
        if rc != 0:
            raise RuntimeError(f"tripool_expert_wait_layer({expert_id}, {layer_idx}) failed (rc={rc})")

    def evict_expert_range(self, expert_id: int, start_layer: int, end_layer: int, stream_ptr: int = 0) -> int:
        n = int(
            self.lib.tripool_expert_evict_range(
                int(expert_id), int(start_layer), int(end_layer), ctypes.c_void_p(stream_ptr)
            )
        )
        if n < 0:
            raise RuntimeError(f"tripool_expert_evict_range failed (rc={n})")
        return n

    def expert_resident_count(self, expert_id: int) -> int:
        return int(self.lib.tripool_expert_resident_count(int(expert_id)))

    def expert_is_resident(self, expert_id: int, layer_idx: int) -> bool:
        return bool(self.lib.tripool_expert_is_resident(int(expert_id), int(layer_idx)))

    def shutdown(self) -> None:
        if self._closed:
            return
        self._pluggable = None
        gc.collect()
        rc = int(self.lib.tripool_vmm_shutdown())
        self._closed = True
        if rc != 0:
            raise RuntimeError(f"tripool_vmm_shutdown failed (rc={rc})")


class SimulatedTriPoolBackend:
    """Local-substitutable CPU VMM backend enforcing super-handle conservation, pinning, and transient auto-eviction."""

    is_simulated = True

    def __init__(
        self,
        device: int = 0,
        super_pool_mib: int = 14336,
        fine_kv_pool_mib: int = 128,
        super_handle_mib: int = 64,
        allow_overflow: bool = False,
        materialize_va_bytes: bool = False,
    ) -> None:
        self.device = int(device)
        self.super_handle_mib = int(super_handle_mib)
        self.super_handle_bytes = self.super_handle_mib * MIB
        self.min_granularity_bytes = 2 * MIB
        self.allow_overflow = bool(allow_overflow)
        self.materialize_va_bytes = bool(materialize_va_bytes)

        self._super_total = max(1, int(super_pool_mib) // self.super_handle_mib)
        self._super_free = self._super_total
        self._fine_total = max(0, int(fine_kv_pool_mib) // 2)
        self._fine_free = self._fine_total
        self._pages_per_super = self.super_handle_mib // 2

        self._kv_allocated: set[int] = set()
        self._kv_regions: dict[int, dict[str, Any]] = {}
        self._next_region_id = 0
        self._next_kv_slot = 0
        self._kv_peak = 0
        self._carves = 0
        self._overflow_creates = 0

        self._in_scope = False
        self._transient_current = 0
        self._transient_peak = 0
        self._transient_flushes = 0

        self._experts: dict[int, dict[str, Any]] = {}
        self._closed = False

        self._prefetch_completed = 0
        self._wait_calls = 0
        self._wait_stalls = 0
        self._manual_evictions = 0
        self._auto_evictions_transient = 0
        self._auto_evictions_kv = 0
        self._h2d_bytes_total = 0

    def _evict_one_unpinned_tail_layer(self, for_kv: bool = False) -> bool:
        for exp in self._experts.values():
            if exp["pinned"]:
                continue
            for lid in range(exp["num_layers"] - 1, -1, -1):
                if exp["resident"][lid] or exp["pending_async"][lid]:
                    exp["resident"][lid] = False
                    exp["pending_async"][lid] = False
                    self._super_free += exp["handles_per_layer"]
                    if for_kv:
                        self._auto_evictions_kv += 1
                    else:
                        self._auto_evictions_transient += 1
                    return True
        return False

    def _acquire_super_handles(self, count: int, for_transient: bool = False, for_kv: bool = False) -> None:
        while self._super_free < count:
            if (for_transient or for_kv) and self._evict_one_unpinned_tail_layer(for_kv=for_kv):
                continue
            if self.allow_overflow:
                deficit = count - self._super_free
                self._super_total += deficit
                self._super_free += deficit
                self._overflow_creates += deficit
                break
            raise RuntimeError(
                f"SimulatedTriPoolBackend exhausted: need {count} handles, free={self._super_free}, overflow=False"
            )
        self._super_free -= count

    def stats_dict(self) -> dict[str, Any]:
        exp_handles = sum(
            sum(1 for r, p in zip(e["resident"], e["pending_async"]) if (r or p)) * e["handles_per_layer"]
            for e in self._experts.values()
        )
        exp_layers = sum(sum(1 for r in e["resident"] if r) for e in self._experts.values())
        reg_pages = sum(r["mapped_pages"] for r in self._kv_regions.values())
        kv_cur_pages = len(self._kv_allocated) + reg_pages
        return {
            "min_granularity_bytes": self.min_granularity_bytes,
            "super_handle_bytes": self.super_handle_bytes,
            "super_pool_total_handles": self._super_total,
            "super_pool_free_handles": self._super_free,
            "fine_pool_total_pages": self._fine_total,
            "fine_pool_free_pages": self._fine_free,
            "kv_mapped_pages_current": kv_cur_pages,
            "kv_mapped_pages_peak": self._kv_peak,
            "super_to_fine_carves": self._carves,
            "transient_mapped_handles_current": self._transient_current,
            "transient_mapped_handles_peak": self._transient_peak,
            "transient_scope_flushes": self._transient_flushes,
            "expert_mapped_handles_current": exp_handles,
            "expert_resident_layers_total": exp_layers,
            "expert_prefetch_layers_completed": self._prefetch_completed,
            "expert_wait_calls": self._wait_calls,
            "expert_wait_stalls": self._wait_stalls,
            "expert_manual_evictions": self._manual_evictions,
            "expert_auto_evictions_for_transient": self._auto_evictions_transient,
            "expert_auto_evictions_for_kv": self._auto_evictions_kv,
            "expert_h2d_bytes_total": self._h2d_bytes_total,
            "overflow_creates": self._overflow_creates,
        }

    def _pop_fine_page(self) -> None:
        if self._fine_free == 0:
            self._acquire_super_handles(1, for_kv=True)
            self._fine_total += self._pages_per_super
            self._fine_free += self._pages_per_super
            self._carves += 1
        self._fine_free -= 1

    def _cur_kv_pages_total(self) -> int:
        return len(self._kv_allocated) + sum(r["mapped_pages"] for r in self._kv_regions.values())

    def kv_alloc_page(self) -> int:
        self._pop_fine_page()
        slot = self._next_kv_slot
        self._next_kv_slot += 1
        self._kv_allocated.add(slot)
        self._kv_peak = max(self._kv_peak, self._cur_kv_pages_total())
        return slot

    def kv_free_page(self, slot_idx: int, stream_ptr: int = 0) -> None:
        if slot_idx in self._kv_allocated:
            self._kv_allocated.remove(slot_idx)
            self._fine_free += 1

    def kv_region_alloc(self, max_va_bytes: int) -> int:
        gran = self.min_granularity_bytes
        cap = max(gran, ((int(max_va_bytes) + gran - 1) // gran) * gran)
        rid = self._next_region_id
        self._next_region_id += 1
        self._kv_regions[rid] = {
            "cap_bytes": cap,
            "mapped_pages": 0,
            "va_base": 0x7000_0000_0000 + rid * (1 << 30),
            "buf": bytearray(0),
        }
        return rid

    def kv_region_ensure(self, region_id: int, needed_bytes: int, stream_ptr: int = 0) -> int:
        reg = self._kv_regions[int(region_id)]
        gran = self.min_granularity_bytes
        eff = max(1, int(needed_bytes))
        needed_pages = (eff + gran - 1) // gran
        if needed_pages * gran > reg["cap_bytes"]:
            raise MemoryError(f"simulated kv_region {region_id} needed {eff} > cap {reg['cap_bytes']}")
        old_pages = reg["mapped_pages"]
        if needed_pages > old_pages:
            delta = needed_pages - old_pages
            pulled = 0
            try:
                for _ in range(delta):
                    self._pop_fine_page()
                    pulled += 1
            except RuntimeError as exc:
                self._fine_free += pulled
                raise MemoryError(str(exc)) from exc
            reg["mapped_pages"] = needed_pages
            if self.materialize_va_bytes:
                reg["buf"].extend(b"\x00" * (delta * gran))
            self._kv_peak = max(self._kv_peak, self._cur_kv_pages_total())
        return int(reg["va_base"])

    def kv_region_va(self, region_id: int) -> int:
        return int(self._kv_regions[int(region_id)]["va_base"])

    def kv_region_mapped_bytes(self, region_id: int) -> int:
        reg = self._kv_regions[int(region_id)]
        return int(reg["mapped_pages"] * self.min_granularity_bytes)

    def kv_region_uint8_view(self, region_id: int, nbytes: int | None = None) -> memoryview:
        reg = self._kv_regions[int(region_id)]
        mapped = reg["mapped_pages"] * self.min_granularity_bytes
        target = mapped if nbytes is None else min(mapped, int(nbytes))
        if len(reg["buf"]) < mapped:
            reg["buf"].extend(b"\x00" * (mapped - len(reg["buf"])))
        return memoryview(reg["buf"])[:target]

    def kv_region_free(self, region_id: int, stream_ptr: int = 0) -> None:
        reg = self._kv_regions.pop(int(region_id), None)
        if reg is not None:
            self._fine_free += int(reg["mapped_pages"])
            reg["buf"] = bytearray(0)

    def allocate_transient_mib(self, size_mib: int) -> None:
        if not self._in_scope:
            raise RuntimeError("allocate_transient_mib requires an active transient scope")
        handles = max(1, (int(size_mib) + self.super_handle_mib - 1) // self.super_handle_mib)
        self._acquire_super_handles(handles, for_transient=True)
        self._transient_current += handles
        self._transient_peak = max(self._transient_peak, self._transient_current)

    def run_in_transient_scope(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._in_scope = True
        try:
            return fn(*args, **kwargs)
        finally:
            self._super_free += self._transient_current
            self._transient_current = 0
            self._transient_flushes += 1
            self._in_scope = False

    @contextmanager
    def transient_scope(self):
        self._in_scope = True
        try:
            yield self
        finally:
            self._super_free += self._transient_current
            self._transient_current = 0
            self._transient_flushes += 1
            self._in_scope = False

    def register_expert(self, num_layers: int, layer_stride_mib: int, mirror: NumaMemfdMirror) -> int:
        mirror.pin_cuda(self)
        eid = len(self._experts)
        stride_b = int(layer_stride_mib) * MIB
        handles_per = max(1, int(layer_stride_mib) // self.super_handle_mib)
        va_bufs = [bytearray(stride_b) if self.materialize_va_bytes else bytearray(0) for _ in range(num_layers)]
        self._experts[eid] = {
            "num_layers": int(num_layers),
            "stride_bytes": stride_b,
            "handles_per_layer": handles_per,
            "mirror": mirror,
            "pinned": False,
            "used_bytes": [stride_b for _ in range(num_layers)],
            "resident": [False for _ in range(num_layers)],
            "pending_async": [False for _ in range(num_layers)],
            "va_buffers": va_bufs,
        }
        return eid

    def expert_layer_stride_bytes(self, expert_id: int) -> int:
        return int(self._experts[expert_id]["stride_bytes"])

    def expert_layer_uint8_view(self, expert_id: int, layer_idx: int) -> memoryview:
        return memoryview(self._experts[expert_id]["va_buffers"][layer_idx])

    def set_expert_pinned(self, expert_id: int, pinned: bool) -> None:
        self._experts[expert_id]["pinned"] = bool(pinned)

    def set_expert_layer_used_bytes(self, expert_id: int, layer_idx: int, used_bytes: int) -> None:
        self._experts[expert_id]["used_bytes"][layer_idx] = int(used_bytes)

    def _copy_mirror_to_va(self, exp: dict[str, Any], lid: int) -> None:
        used = exp["used_bytes"][lid]
        self._h2d_bytes_total += used
        if self.materialize_va_bytes and used > 0:
            off = lid * exp["stride_bytes"]
            exp["va_buffers"][lid][:used] = exp["mirror"].mm[off : off + used]

    def load_expert_layers_sync(self, expert_id: int, start_layer: int, end_layer: int) -> None:
        exp = self._experts[expert_id]
        for lid in range(int(start_layer), int(end_layer)):
            if exp["resident"][lid]:
                continue
            if not exp["pending_async"][lid]:
                self._acquire_super_handles(exp["handles_per_layer"])
                self._copy_mirror_to_va(exp, lid)
            exp["pending_async"][lid] = False
            exp["resident"][lid] = True

    def prefetch_expert_async(self, expert_id: int, start_layer: int, end_layer: int) -> int:
        exp = self._experts[expert_id]
        queued = 0
        for lid in range(int(start_layer), int(end_layer)):
            if exp["resident"][lid] or exp["pending_async"][lid]:
                continue
            self._acquire_super_handles(exp["handles_per_layer"])
            self._copy_mirror_to_va(exp, lid)
            exp["pending_async"][lid] = True
            queued += 1
        return queued

    def wait_expert_layer(self, expert_id: int, layer_idx: int, stream_ptr: int = 0) -> None:
        exp = self._experts[expert_id]
        self._wait_calls += 1
        if exp["pending_async"][layer_idx]:
            exp["pending_async"][layer_idx] = False
            exp["resident"][layer_idx] = True
            self._prefetch_completed += 1
            self._wait_stalls += 1
            return
        if not exp["resident"][layer_idx]:
            self.load_expert_layers_sync(expert_id, layer_idx, layer_idx + 1)
            self._wait_stalls += 1

    def evict_expert_range(self, expert_id: int, start_layer: int, end_layer: int, stream_ptr: int = 0) -> int:
        exp = self._experts[expert_id]
        if exp["pinned"]:
            return 0
        evicted = 0
        for lid in range(int(start_layer), int(end_layer)):
            if exp["resident"][lid] or exp["pending_async"][lid]:
                exp["resident"][lid] = False
                exp["pending_async"][lid] = False
                self._super_free += exp["handles_per_layer"]
                self._manual_evictions += 1
                evicted += 1
        return evicted

    def expert_resident_count(self, expert_id: int) -> int:
        exp = self._experts[expert_id]
        return sum(1 for r in exp["resident"] if r)

    def expert_is_resident(self, expert_id: int, layer_idx: int) -> bool:
        return bool(self._experts[expert_id]["resident"][layer_idx])

    def shutdown(self) -> None:
        self._experts.clear()
        self._kv_allocated.clear()
        self._kv_regions.clear()
        self._closed = True
