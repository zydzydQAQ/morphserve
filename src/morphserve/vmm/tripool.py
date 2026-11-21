"""Python interface to libtripool_vmm.so (TriPool-VMM) and NUMA-pinned memfd weight mirrors."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import gc
import mmap
import os
from pathlib import Path
from typing import Any, Callable


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


class NumaMemfdMirror:
    """NUMA-bound, CUDA-pinned host weight mirror backed by memfd_create (bypasses /dev/shm cap)."""

    def __init__(self, nbytes: int, numa_node: int = 0, name: str = "morphserve_expert") -> None:
        self.nbytes = int(nbytes)
        self.numa_node = int(numa_node)
        self.fd = os.memfd_create(name, 0)
        os.ftruncate(self.fd, self.nbytes)
        self.mm = mmap.mmap(
            self.fd,
            self.nbytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        # Obtain stable base address via ctypes buffer view without copying
        self._c_buf = (ctypes.c_uint8 * self.nbytes).from_buffer(self.mm)
        self.ptr = ctypes.addressof(self._c_buf)
        self._pinned_lib: ctypes.CDLL | None = None
        self.numa_bound = False

        # Bind pages to requested NUMA node when libnuma is available
        try:
            libnuma = ctypes.CDLL("libnuma.so.1")
            libnuma.numa_available.restype = ctypes.c_int
            libnuma.numa_tonode_memory.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            libnuma.numa_tonode_memory.restype = None
            if libnuma.numa_available() >= 0:
                libnuma.numa_tonode_memory(
                    ctypes.c_void_p(self.ptr), ctypes.c_size_t(self.nbytes), self.numa_node
                )
                self.numa_bound = True
        except OSError:
            pass

    def as_uint8_tensor(self):
        import torch

        return torch.frombuffer(self.mm, dtype=torch.uint8)

    def pin_cuda(self, vmm: TriPoolVMM) -> None:
        if self._pinned_lib is not None:
            return
        rc = vmm.lib.tripool_host_register(ctypes.c_void_p(self.ptr), ctypes.c_size_t(self.nbytes))
        if rc != 0:
            raise RuntimeError(f"tripool_host_register failed (rc={rc}) for {self.nbytes} bytes")
        self._pinned_lib = vmm.lib

    def close(self) -> None:
        if self._pinned_lib is not None:
            try:
                self._pinned_lib.tripool_host_unregister(ctypes.c_void_p(self.ptr))
            except Exception:
                pass
            self._pinned_lib = None
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

        # Ensure PyTorch CUDA primary context is active before cuMemCreate
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

    def kv_page_tensor(self, slot_idx: int):
        import torch

        gran = self.stats().min_granularity_bytes
        ptr = self.kv_va_base() + int(slot_idx) * gran
        return torch.as_tensor(_CudaVAView(ptr, gran, self), device=f"cuda:{self.device}")

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
            # Clone compact return tensor(s) outside use_mem_pool -> allocated on default allocator
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

    def expert_layer_uint8_view(self, expert_id: int, layer_idx: int):
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
