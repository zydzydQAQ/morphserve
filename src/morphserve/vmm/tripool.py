"""Python interface to libtripool_vmm.so (TriPool-VMM), local-substitutable VMM simulator, and unified NUMA memfd mirror."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
import ctypes
import gc
import mmap
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable
import weakref

MIB = 1024**2
CU_MEMHOSTREGISTER_PORTABLE = 0x1

try:
    import torch as _TORCH
except ImportError:  # pragma: no cover - local CPU/WSL environments without PyTorch
    _TORCH = None  # type: ignore[assignment]


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
        ("fine_to_super_bundles", ctypes.c_uint64),
        ("transient_fine_bundles_current", ctypes.c_uint64),
        ("expert_fine_bundles_current", ctypes.c_uint64),
        ("pool_total_bytes", ctypes.c_uint64),
        ("pool_free_bytes", ctypes.c_uint64),
        ("expert_unpinned_bytes", ctypes.c_uint64),
        ("kv_free_bytes", ctypes.c_uint64),
        ("kv_dit_headroom_bytes", ctypes.c_uint64),
        ("kv_swapped_pages_current", ctypes.c_uint64),
        ("kv_swapped_bytes_current", ctypes.c_uint64),
        ("kv_swap_outs_total", ctypes.c_uint64),
        ("kv_swap_ins_total", ctypes.c_uint64),
        ("kv_swap_d2h_bytes_total", ctypes.c_uint64),
        ("kv_swap_h2d_bytes_total", ctypes.c_uint64),
        ("kv_host_pinned_pool_bytes", ctypes.c_uint64),
    ]

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name, _ in self._fields_}


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
        self.cuInit = self.lib.cuInit
        self.cuInit.argtypes = [ctypes.c_uint]
        self.cuInit.restype = ctypes.c_int

        self.cuDeviceGet = self.lib.cuDeviceGet
        self.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.cuDeviceGet.restype = ctypes.c_int

        self.cuDevicePrimaryCtxRetain = self.lib.cuDevicePrimaryCtxRetain
        self.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        self.cuDevicePrimaryCtxRetain.restype = ctypes.c_int

        self.cuDevicePrimaryCtxRelease = getattr(
            self.lib, "cuDevicePrimaryCtxRelease_v2", self.lib.cuDevicePrimaryCtxRelease
        )
        self.cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
        self.cuDevicePrimaryCtxRelease.restype = ctypes.c_int

        self.cuCtxGetCurrent = self.lib.cuCtxGetCurrent
        self.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.cuCtxGetCurrent.restype = ctypes.c_int

        self.cuCtxSetCurrent = self.lib.cuCtxSetCurrent
        self.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        self.cuCtxSetCurrent.restype = ctypes.c_int

        self.cuMemHostRegister = getattr(self.lib, "cuMemHostRegister_v2", self.lib.cuMemHostRegister)
        self.cuMemHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        self.cuMemHostRegister.restype = ctypes.c_int

        self.cuMemHostUnregister = self.lib.cuMemHostUnregister
        self.cuMemHostUnregister.argtypes = [ctypes.c_void_p]
        self.cuMemHostUnregister.restype = ctypes.c_int

        self._retained_ctx = ctypes.c_void_p()
        self._retained_dev: int | None = None

    def _ensure_ctx(self) -> None:
        rc = int(self.cuInit(0))
        if rc != 0:
            raise RuntimeError(f"cuInit(0) failed with CUresult={rc}")
        cur = ctypes.c_void_p()
        rc = int(self.cuCtxGetCurrent(ctypes.byref(cur)))
        if rc != 0:
            raise RuntimeError(f"cuCtxGetCurrent failed with CUresult={rc}")
        if not cur.value or self._retained_dev is not None:
            if self._retained_dev is None:
                dev = ctypes.c_int(0)
                rc = int(self.cuDeviceGet(ctypes.byref(dev), 0))
                if rc != 0:
                    raise RuntimeError(f"cuDeviceGet(0) failed with CUresult={rc}")
                ctx = ctypes.c_void_p()
                rc = int(self.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev.value))
                if rc != 0:
                    raise RuntimeError(f"cuDevicePrimaryCtxRetain failed with CUresult={rc}")
                self._retained_ctx = ctx
                self._retained_dev = int(dev.value)
            rc = int(self.cuCtxSetCurrent(self._retained_ctx))
            if rc != 0:
                raise RuntimeError(f"cuCtxSetCurrent failed with CUresult={rc}")

    def register(self, ptr: int, nbytes: int) -> None:
        self._ensure_ctx()
        rc = int(self.cuMemHostRegister(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), CU_MEMHOSTREGISTER_PORTABLE))
        if rc != 0:
            if self._retained_dev is not None:
                self.cuCtxSetCurrent(ctypes.c_void_p(0))
                self.cuDevicePrimaryCtxRelease(self._retained_dev)
                self._retained_dev = None
                self._retained_ctx = ctypes.c_void_p()
            raise RuntimeError(f"cuMemHostRegister(ptr=0x{ptr:x}, nbytes={nbytes}) failed with CUresult={rc}")

    def unregister(self, ptr: int) -> None:
        try:
            self._ensure_ctx()
            rc = int(self.cuMemHostUnregister(ctypes.c_void_p(ptr)))
            if rc != 0:
                raise RuntimeError(f"cuMemHostUnregister failed with CUresult={rc}")
        finally:
            if self._retained_dev is not None:
                self.cuCtxSetCurrent(ctypes.c_void_p(0))
                self.cuDevicePrimaryCtxRelease(self._retained_dev)
                self._retained_dev = None
                self._retained_ctx = ctypes.c_void_p()


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


def _create_memfd(name: str, flags: int = 0) -> int:
    """Create a genuine Linux kernel `memfd` via `os.memfd_create` or `libc.so.6` fallback (for cos7/glibc-2.17 conda builds)."""
    if "\x00" in name:
        raise ValueError("embedded null character")
    memfd_fn = getattr(os, "memfd_create", None)
    if memfd_fn is not None:
        return int(memfd_fn(name, int(flags)))
    name_bytes = name.encode("utf-8")
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    c_memfd_create = getattr(libc, "memfd_create", None)
    if c_memfd_create is not None:
        c_memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        c_memfd_create.restype = ctypes.c_int
        ctypes.set_errno(0)
        fd = int(c_memfd_create(name_bytes, ctypes.c_uint(int(flags))))
        if fd < 0:
            err = ctypes.get_errno()
            msg = f"memfd_create failed: {os.strerror(err)}" if err else "memfd_create failed"
            raise OSError(err, msg)
        return fd
    import platform

    arch = platform.machine().lower()
    sys_nr = 319 if arch in ("x86_64", "amd64") else (279 if arch in ("aarch64", "arm64") else None)
    c_syscall = getattr(libc, "syscall", None)
    if sys_nr is not None and c_syscall is not None:
        c_syscall.argtypes = [ctypes.c_long, ctypes.c_char_p, ctypes.c_uint]
        c_syscall.restype = ctypes.c_long
        ctypes.set_errno(0)
        fd = int(c_syscall(ctypes.c_long(sys_nr), name_bytes, ctypes.c_uint(int(flags))))
        if fd < 0:
            err = ctypes.get_errno()
            msg = f"memfd_create failed: {os.strerror(err)}" if err else "memfd_create failed"
            raise OSError(err, msg)
        return fd
    raise OSError("memfd_create failed: unavailable in os and libc.so.6")


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
        self.fd = -1
        self.mm: mmap.mmap | None = None
        self._c_buf: Any = None
        self._pinned_vmm_lib: Any = None
        self._pin_api: _HostCudaPin | None = None
        self.pinned = False
        self._cuda_pinned = False
        self.nbytes = int(nbytes)
        self.numa_node = int(numa_node)
        self.is_owner = bool(_is_owner)

        if _attached_fd is not None and _attached_mm is not None:
            self.fd = int(_attached_fd)
            self.mm = _attached_mm
            self.numa_bound = False
        else:
            fd = _create_memfd(name, 0)
            try:
                os.ftruncate(fd, self.nbytes)
                mm = mmap.mmap(fd, self.nbytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
            except Exception:
                os.close(fd)
                raise
            self.fd = fd
            self.mm = mm

        try:
            if _attached_fd is None:
                stride = 2 * MIB
                for off in range(0, self.nbytes, stride):
                    self.mm[off] = 0
                if populate_bytes:
                    mv = memoryview(self.mm)
                    try:
                        mv[: len(populate_bytes)] = populate_bytes
                    finally:
                        mv.release()

            self.inode = int(os.fstat(self.fd).st_ino)
            self._c_buf = (ctypes.c_uint8 * self.nbytes).from_buffer(self.mm)
            self.ptr = int(ctypes.addressof(self._c_buf))
            if _attached_fd is None:
                self.numa_bound = _bind_numa_memory(self.ptr, self.nbytes, self.numa_node)
        except BaseException:
            self.close()
            raise

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
        if self._cuda_pinned:
            if vmm is not None and hasattr(vmm, "_track_pinned_mirror"):
                vmm._track_pinned_mirror(self)
            return
        if vmm is not None and getattr(vmm, "is_simulated", False):
            self.pinned = True
            return
        if vmm is not None and getattr(vmm, "lib", None) is not None:
            rc = int(vmm.lib.tripool_host_register(ctypes.c_void_p(self.ptr), ctypes.c_size_t(self.nbytes)))
            if rc != 0:
                raise RuntimeError(f"tripool_host_register failed (rc={rc}) for {self.nbytes} bytes")
            self._pinned_vmm_lib = vmm.lib
            self._cuda_pinned = True
            self.pinned = True
            if hasattr(vmm, "_track_pinned_mirror"):
                vmm._track_pinned_mirror(self)
            return
        if self._pin_api is None:
            self._pin_api = _HostCudaPin()
        self._pin_api.register(self.ptr, self.nbytes)
        self._cuda_pinned = True
        self.pinned = True

    def unpin_cuda(self) -> None:
        if self._cuda_pinned:
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
            self._cuda_pinned = False
        self.pinned = False

    def stream_h2d_chunks(self, dst_device_tensor: Any, chunk_bytes: int = 64 * MIB) -> float:
        """Copy shared pinned mirror -> GPU tensor via CUDA H2D and return achieved GB/s."""
        import torch

        if not self._cuda_pinned:
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
        self.unpin_cuda()
        if getattr(self, "_c_buf", None) is not None:
            del self._c_buf
            self._c_buf = None
        mm = getattr(self, "mm", None)
        self.mm = None
        if mm is not None:
            try:
                mm.close()
            except BufferError:
                if gc is not None:
                    gc.collect()
                try:
                    mm.close()
                except BufferError:
                    # Caller stack still holds a buffer view (e.g. `as_uint8_tensor()`);
                    # `mmap.__del__` unmaps once view drops, while `os.close(fd)` releases FD immediately.
                    pass
        if getattr(self, "fd", -1) >= 0:
            os.close(self.fd)
            self.fd = -1


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


class _VmmKvSwapMixin:
    """Shared real-time `kv_free_bytes` ledger and high-watermark thinking-session KV swap orchestrator."""

    def _init_swap_governance(
        self,
        *,
        kv_swap_high_watermark: float = 0.85,
        kv_swap_low_watermark: float = 0.65,
        kv_swap_min_idle_s: float = 1.0,
        max_pool_bytes: int | None = None,
    ) -> None:
        hi = float(kv_swap_high_watermark)
        lo = float(kv_swap_low_watermark)
        if not (0.0 < lo <= hi <= 1.0):
            raise ValueError(
                f"expected 0.0 < kv_swap_low_watermark ({lo}) <= kv_swap_high_watermark ({hi}) <= 1.0"
            )
        self.kv_swap_high_watermark = hi
        self.kv_swap_low_watermark = lo
        self.kv_swap_min_idle_s = max(0.0, float(kv_swap_min_idle_s))
        self.max_pool_bytes = int(max_pool_bytes) if max_pool_bytes is not None else None
        self._swap_mu = threading.RLock()
        self._registered_kv_caches: dict[int, weakref.ReferenceType[Any]] = {}

    def register_kv_cache(self, cache: Any) -> None:
        with self._swap_mu:
            self._registered_kv_caches[id(cache)] = weakref.ref(cache)

    def unregister_kv_cache(self, cache: Any) -> None:
        with self._swap_mu:
            self._registered_kv_caches.pop(id(cache), None)

    def registered_kv_caches(self) -> list[Any]:
        live: list[Any] = []
        with self._swap_mu:
            dead_ids: list[int] = []
            for cid, ref in self._registered_kv_caches.items():
                obj = ref()
                if obj is None or getattr(obj, "_closed", False):
                    dead_ids.append(cid)
                else:
                    live.append(obj)
            for cid in dead_ids:
                self._registered_kv_caches.pop(cid, None)
        return live

    def _registered_expert_total_bytes(self) -> int:
        return 0

    def _active_kv_used_bytes(self, st: dict[str, Any]) -> int:
        gran = int(st.get("min_granularity_bytes", 2 * MIB))
        return int(st.get("kv_mapped_pages_current", 0)) * gran

    def _clamp_free_by_max_pool(self, raw_free: int, st: dict[str, Any], for_dit: bool) -> int:
        if self.max_pool_bytes is None:
            return max(0, int(raw_free))
        sup = int(st.get("super_handle_bytes", 64 * MIB))
        kv_used = self._active_kv_used_bytes(st)
        tr_used = int(st.get("transient_mapped_handles_current", 0)) * sup
        if for_dit:
            exp_full = self._registered_expert_total_bytes()
            soft_free = self.max_pool_bytes - kv_used - tr_used - exp_full
        else:
            exp_pinned = max(
                0, int(st.get("expert_mapped_handles_current", 0)) * sup - int(st.get("expert_unpinned_bytes", 0))
            )
            soft_free = self.max_pool_bytes - kv_used - tr_used - exp_pinned
        return max(0, min(int(raw_free), int(soft_free)))

    def sync_kv_free_bytes(self, for_dit: bool = False) -> int:
        """Flush any completed async KV swap-outs and return exact real-time `kv_free_bytes`."""
        self.flush_kv_swaps()  # type: ignore[attr-defined]
        return self.kv_free_bytes(for_dit=for_dit)  # type: ignore[attr-defined]

    def reclaim_thinking_kv_bytes(
        self,
        target_reclaim_bytes: int,
        *,
        sync: bool = False,
        exclude_caches: Iterable[Any] = (),
        stream_ptr: int | None = None,
    ) -> int:
        """Asynchronously swap out inactive/thinking-period `VmmNaiveCache` sessions to Host Pinned DRAM."""
        need = int(target_reclaim_bytes)
        if need <= 0:
            return 0
        exclude_ids = {id(c) for c in exclude_caches if c is not None}

        with self._swap_mu:
            candidates: list[Any] = []
            for cache in self.registered_kv_caches():
                if id(cache) in exclude_ids:
                    continue
                if getattr(cache, "is_pinned_active", False):
                    continue
                is_thinking = bool(getattr(cache, "is_thinking", False))
                idle_s = float(getattr(cache, "idle_seconds", 0.0))
                if not is_thinking and idle_s < self.kv_swap_min_idle_s:
                    continue
                if int(getattr(cache, "mapped_bytes", 0)) <= 0:
                    continue
                candidates.append(cache)

            candidates.sort(
                key=lambda c: (
                    0 if getattr(c, "is_thinking", False) else 1,
                    float(getattr(c, "last_access_mono", 0.0)),
                )
            )

            reclaimed = 0
            for victim in candidates:
                if reclaimed >= need:
                    break
                reclaimed += int(victim.swap_out_async(stream_ptr=stream_ptr))

        if sync and reclaimed > 0:
            self.flush_kv_swaps()  # type: ignore[attr-defined]
        return reclaimed

    def maybe_evict_thinking_kv_on_high_watermark(
        self,
        *,
        sync: bool = False,
        exclude_caches: Iterable[Any] = (),
        extra_needed_bytes: int = 0,
        for_dit: bool = False,
        stream_ptr: int | None = None,
    ) -> int:
        """Swap out thinking-period KV sessions when KV occupancy crosses `kv_swap_high_watermark` of KV budget."""
        st = self.stats_dict()  # type: ignore[attr-defined]
        pool_tot = int(st.get("pool_total_bytes", 0))
        eff_pool = min(pool_tot, self.max_pool_bytes) if self.max_pool_bytes is not None else pool_tot
        if eff_pool <= 0:
            return 0

        sup = int(st.get("super_handle_bytes", 64 * MIB))
        tr_used = int(st.get("transient_mapped_handles_current", 0)) * sup
        exp_pinned = max(
            0, int(st.get("expert_mapped_handles_current", 0)) * sup - int(st.get("expert_unpinned_bytes", 0))
        )
        fixed_non_kv = tr_used + (self._registered_expert_total_bytes() if for_dit else exp_pinned)
        kv_budget = max(0, eff_pool - fixed_non_kv)

        active_kv_used = self._active_kv_used_bytes(st)
        extra = max(0, int(extra_needed_bytes))
        projected_kv = active_kv_used + extra
        high_b = int(kv_budget * self.kv_swap_high_watermark)
        low_b = int(kv_budget * self.kv_swap_low_watermark)

        if projected_kv <= high_b and projected_kv <= kv_budget:
            return 0

        target_reclaim = max(0, projected_kv - low_b)
        return self.reclaim_thinking_kv_bytes(
            target_reclaim,
            sync=sync,
            exclude_caches=exclude_caches,
            stream_ptr=stream_ptr,
        )


class TriPoolVMM(_VmmKvSwapMixin):
    """Unified three-arena CUDA Virtual Memory Manager backed by libtripool_vmm.so."""

    is_simulated = False
    _so_lock = threading.RLock()
    _so_epochs: dict[tuple[int, int], list[int]] = {}
    _so_mirrors: dict[tuple[int, int], dict[int, weakref.ReferenceType[NumaMemfdMirror]]] = {}

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
        kv_swap_high_watermark: float = 0.85,
        kv_swap_low_watermark: float = 0.65,
        kv_swap_min_idle_s: float = 1.0,
        max_pool_bytes: int | None = None,
        init_torch_cuda: bool = True,
    ) -> None:
        self._closed = True
        self._so_key: tuple[int, int] = (-1, -1)
        self._vmm_epoch: int = -1
        self._has_torch_cuda = False
        self._pluggable = None
        self._raw_transient_ptrs: list[tuple[int, int]] = []
        self._registered_expert_bytes: int = 0
        self.device = int(device)
        self.allow_overflow = bool(allow_overflow)
        self.super_handle_mib = int(super_handle_mib)
        self._init_swap_governance(
            kv_swap_high_watermark=kv_swap_high_watermark,
            kv_swap_low_watermark=kv_swap_low_watermark,
            kv_swap_min_idle_s=kv_swap_min_idle_s,
            max_pool_bytes=max_pool_bytes,
        )

        self.so_path = Path(so_path) if so_path is not None else resolve_vmm_so()
        st_so = os.stat(self.so_path)
        self._so_key = (int(st_so.st_dev), int(st_so.st_ino))
        self.lib = ctypes.CDLL(str(self.so_path))
        self._bind_symbols()

        has_torch_cuda = bool(init_torch_cuda and _TORCH is not None and _TORCH.cuda.is_available())
        self._has_torch_cuda = has_torch_cuda
        if has_torch_cuda:
            assert _TORCH is not None
            _TORCH.cuda.set_device(self.device)
            _ = _TORCH.zeros(1, device=f"cuda:{self.device}")
            _TORCH.cuda.synchronize(self.device)

        cls = type(self)
        with cls._so_lock:
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
            epoch_state = cls._so_epochs.get(self._so_key)
            if epoch_state is None or epoch_state[1] == 0:
                next_epoch = 1 if epoch_state is None else (epoch_state[0] + 1)
                cls._so_epochs[self._so_key] = [next_epoch, 1]
                self._vmm_epoch = next_epoch
            else:
                epoch_state[1] += 1
                self._vmm_epoch = epoch_state[0]
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

        L.tripool_alloc.argtypes = [ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
        L.tripool_alloc.restype = ctypes.c_uint64
        L.tripool_free.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
        L.tripool_free.restype = None
        L.tripool_scope_begin.argtypes = []
        L.tripool_scope_begin.restype = None
        L.tripool_scope_end.argtypes = [ctypes.c_void_p]
        L.tripool_scope_end.restype = ctypes.c_int

        L.tripool_kv_va_base.argtypes = []
        L.tripool_kv_va_base.restype = ctypes.c_uint64
        L.tripool_kv_free_bytes.argtypes = [ctypes.c_int]
        L.tripool_kv_free_bytes.restype = ctypes.c_uint64
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

        L.tripool_kv_region_set_evictable.argtypes = [
            ctypes.c_int32,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        L.tripool_kv_region_set_evictable.restype = ctypes.c_int
        L.tripool_kv_region_swap_out_async.argtypes = [ctypes.c_int32, ctypes.c_size_t, ctypes.c_void_p]
        L.tripool_kv_region_swap_out_async.restype = ctypes.c_int
        L.tripool_kv_region_swap_in_async.argtypes = [ctypes.c_int32, ctypes.c_void_p]
        L.tripool_kv_region_swap_in_async.restype = ctypes.c_int
        L.tripool_kv_region_is_swapped.argtypes = [ctypes.c_int32]
        L.tripool_kv_region_is_swapped.restype = ctypes.c_int
        L.tripool_kv_region_swapped_bytes.argtypes = [ctypes.c_int32]
        L.tripool_kv_region_swapped_bytes.restype = ctypes.c_size_t
        L.tripool_kv_flush_swaps.argtypes = []
        L.tripool_kv_flush_swaps.restype = ctypes.c_int

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

    def _torch_cuda_ready(self) -> bool:
        return bool(_TORCH is not None and (self._has_torch_cuda or _TORCH.cuda.is_initialized()))

    def _current_stream_ptr(self) -> int:
        if self._torch_cuda_ready():
            assert _TORCH is not None
            return int(_TORCH.cuda.current_stream(self.device).cuda_stream)
        return 0

    def _registered_expert_total_bytes(self) -> int:
        return int(self._registered_expert_bytes)

    def _active_kv_used_bytes(self, st: dict[str, Any]) -> int:
        gran = int(st.get("min_granularity_bytes", 2 * MIB))
        raw_mapped = int(st.get("kv_mapped_pages_current", 0)) * gran
        raw_kv_free = int(self.lib.tripool_kv_free_bytes(0))
        pool_free = int(st.get("pool_free_bytes", 0))
        exp_unpinned = int(st.get("expert_unpinned_bytes", 0))
        inflight_swap_b = max(0, raw_kv_free - pool_free - exp_unpinned)
        return max(0, raw_mapped - inflight_swap_b)

    def stats(self) -> UnifiedVmmStats:
        st = UnifiedVmmStats()
        self.lib.tripool_vmm_get_stats(ctypes.byref(st))
        return st

    def stats_dict(self) -> dict[str, Any]:
        d = self.stats().to_dict()
        d["physical_pool_bytes"] = int(d["pool_total_bytes"])
        if self.max_pool_bytes is not None:
            d["kv_free_bytes"] = self._clamp_free_by_max_pool(int(d["kv_free_bytes"]), d, for_dit=False)
            d["kv_dit_headroom_bytes"] = self._clamp_free_by_max_pool(
                int(d["kv_dit_headroom_bytes"]), d, for_dit=True
            )
        return d

    def kv_free_bytes(self, for_dit: bool = False, include_overflow: bool = False) -> int:
        raw = int(self.lib.tripool_kv_free_bytes(1 if for_dit else 0))
        if include_overflow and self.allow_overflow and _TORCH is not None and _TORCH.cuda.is_available():
            free_drv, total_drv = _TORCH.cuda.mem_get_info(self.device)
            reserve = min(1536 * MIB, int(total_drv) // 16)
            raw += max(0, int(free_drv) - reserve)
        if self.max_pool_bytes is not None:
            return self._clamp_free_by_max_pool(raw, self.stats().to_dict(), for_dit=for_dit)
        return raw

    # --- Arena 1: Persistent KV 2 MiB Paged Pool & Contiguous Regions ---
    def kv_va_base(self) -> int:
        return int(self.lib.tripool_kv_va_base())

    def kv_alloc_page(self) -> int:
        slot = int(self.lib.tripool_kv_alloc_page())
        if slot < 0:
            self.reclaim_thinking_kv_bytes(2 * MIB, sync=True)
            slot = int(self.lib.tripool_kv_alloc_page())
        if slot < 0:
            raise RuntimeError(f"tripool_kv_alloc_page failed (rc={slot})")
        return slot

    def kv_free_page(self, slot_idx: int, stream_ptr: int = 0) -> None:
        if self._closed:
            return
        rc = int(self.lib.tripool_kv_free_page(int(slot_idx), ctypes.c_void_p(stream_ptr)))
        if rc != 0:
            raise RuntimeError(f"tripool_kv_free_page({slot_idx}) failed (rc={rc})")

    def kv_page_tensor(self, slot_idx: int) -> Any:
        gran = self.stats().min_granularity_bytes
        ptr = self.kv_va_base() + int(slot_idx) * gran
        view = _CudaVAView(ptr, gran, self)
        if self._torch_cuda_ready():
            assert _TORCH is not None
            return _TORCH.as_tensor(view, device=f"cuda:{self.device}")
        return view

    def kv_region_alloc(self, max_va_bytes: int) -> int:
        rid = int(self.lib.tripool_kv_region_alloc(ctypes.c_size_t(int(max_va_bytes))))
        if rid < 0:
            raise RuntimeError(f"tripool_kv_region_alloc({max_va_bytes}) failed (rc={rid})")
        return rid

    def kv_region_ensure(self, region_id: int, needed_bytes: int, stream_ptr: int = 0) -> int:
        req_b = max(1, int(needed_bytes))
        va = int(
            self.lib.tripool_kv_region_ensure(
                ctypes.c_int32(int(region_id)),
                ctypes.c_size_t(req_b),
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
        ptr = self.kv_region_va(region_id)
        mapped = self.kv_region_mapped_bytes(region_id) if nbytes is None else int(nbytes)
        if ptr == 0 or mapped <= 0:
            raise RuntimeError(f"kv_region {region_id} has no mapped VA")
        view = _CudaVAView(ptr, mapped, self)
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
        if self._closed:
            return
        self.lib.tripool_kv_region_set_evictable(
            ctypes.c_int32(int(region_id)),
            1 if evictable else 0,
            ctypes.c_size_t(max(0, int(valid_bytes))),
            ctypes.c_void_p(int(stream_ptr)),
        )

    def kv_region_swap_out_async(self, region_id: int, valid_bytes: int, stream_ptr: int = 0) -> int:
        pages = int(
            self.lib.tripool_kv_region_swap_out_async(
                ctypes.c_int32(int(region_id)),
                ctypes.c_size_t(max(0, int(valid_bytes))),
                ctypes.c_void_p(int(stream_ptr)),
            )
        )
        if pages < 0:
            raise RuntimeError(f"tripool_kv_region_swap_out_async({region_id}) failed (rc={pages})")
        return pages

    kv_region_swap_out = kv_region_swap_out_async

    def kv_region_swap_in_async(self, region_id: int, stream_ptr: int = 0) -> int:
        pages = int(
            self.lib.tripool_kv_region_swap_in_async(
                ctypes.c_int32(int(region_id)),
                ctypes.c_void_p(int(stream_ptr)),
            )
        )
        if pages < 0:
            raise MemoryError(f"tripool_kv_region_swap_in_async({region_id}) failed (rc={pages})")
        return pages

    kv_region_swap_in = kv_region_swap_in_async

    def kv_region_is_swapped(self, region_id: int) -> bool:
        if self._closed:
            return False
        return bool(self.lib.tripool_kv_region_is_swapped(ctypes.c_int32(int(region_id))))

    def kv_region_swapped_bytes(self, region_id: int) -> int:
        if self._closed:
            return 0
        return int(self.lib.tripool_kv_region_swapped_bytes(ctypes.c_int32(int(region_id))))

    def flush_kv_swaps(self) -> None:
        if self._closed:
            return
        rc = int(self.lib.tripool_kv_flush_swaps())
        if rc != 0:
            raise RuntimeError(f"tripool_kv_flush_swaps failed (rc={rc})")

    def kv_region_free(self, region_id: int, stream_ptr: int = 0) -> None:
        if self._closed:
            return
        rc = int(self.lib.tripool_kv_region_free(ctypes.c_int32(int(region_id)), ctypes.c_void_p(int(stream_ptr))))
        if rc != 0:
            raise RuntimeError(f"tripool_kv_region_free({region_id}) failed (rc={rc})")

    # --- Arena 2: Scoped Transient Scratchpad (VAE / DiT Spikes) ---
    def allocate_transient_mib(self, size_mib: int) -> int:
        """Direct Arena-2 allocation helper (also usable in pure-ctypes GPU tests without PyTorch)."""
        nbytes = max(4096, int(size_mib) * MIB)
        s_ptr = self._current_stream_ptr()
        self.maybe_evict_thinking_kv_on_high_watermark(sync=False, extra_needed_bytes=nbytes, stream_ptr=s_ptr)
        ptr = int(self.lib.tripool_alloc(ctypes.c_ssize_t(nbytes), self.device, ctypes.c_void_p(s_ptr)))
        if ptr == 0:
            self.reclaim_thinking_kv_bytes(nbytes, sync=True, stream_ptr=s_ptr)
            ptr = int(self.lib.tripool_alloc(ctypes.c_ssize_t(nbytes), self.device, ctypes.c_void_p(s_ptr)))
        if ptr == 0:
            raise MemoryError(f"tripool_alloc({nbytes}) out of Arena-2 VMM memory")
        self._raw_transient_ptrs.append((ptr, nbytes))
        return ptr

    def _release_raw_transient_ptrs(self, stream_ptr: int) -> None:
        while self._raw_transient_ptrs:
            ptr, nbytes = self._raw_transient_ptrs.pop()
            self.lib.tripool_free(ctypes.c_void_p(ptr), ctypes.c_ssize_t(nbytes), self.device, ctypes.c_void_p(stream_ptr))

    def run_in_transient_scope(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run `fn` inside a scoped TriPool MemPool, clone outputs to default allocator, bulk-unmap immediately."""
        s_ptr = self._current_stream_ptr()
        self.maybe_evict_thinking_kv_on_high_watermark(sync=False, stream_ptr=s_ptr)
        if self._pluggable is None or _TORCH is None:
            self.lib.tripool_scope_begin()
            try:
                return fn(*args, **kwargs)
            finally:
                self._release_raw_transient_ptrs(s_ptr)
                self.lib.tripool_scope_end(ctypes.c_void_p(s_ptr))

        pool = _TORCH.cuda.MemPool(self._pluggable.allocator())
        self.lib.tripool_scope_begin()
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
                import gc
                import traceback

                if exc.__traceback__ is not None:
                    traceback.clear_frames(exc.__traceback__)
                gc.collect()
            except Exception:
                pass
            raise
        finally:
            pooled_out = None
            del pool
            end_s_ptr = self._current_stream_ptr()
            self._release_raw_transient_ptrs(end_s_ptr)
            self.lib.tripool_scope_end(ctypes.c_void_p(end_s_ptr))
        return safe_out

    @contextmanager
    def transient_scope(self):
        """Low-level context manager; caller must not retain pooled tensors past exit."""
        s_ptr = self._current_stream_ptr()
        self.maybe_evict_thinking_kv_on_high_watermark(sync=False, stream_ptr=s_ptr)
        if self._pluggable is None or _TORCH is None:
            self.lib.tripool_scope_begin()
            try:
                yield self
            finally:
                self._release_raw_transient_ptrs(s_ptr)
                self.lib.tripool_scope_end(ctypes.c_void_p(s_ptr))
            return

        pool = _TORCH.cuda.MemPool(self._pluggable.allocator())
        self.lib.tripool_scope_begin()
        try:
            with _TORCH.cuda.use_mem_pool(pool):
                yield pool
        finally:
            del pool
            end_s_ptr = self._current_stream_ptr()
            self._release_raw_transient_ptrs(end_s_ptr)
            self.lib.tripool_scope_end(ctypes.c_void_p(end_s_ptr))

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
        self._registered_expert_bytes += int(num_layers) * self.expert_layer_stride_bytes(eid)
        return eid

    def expert_layer_va(self, expert_id: int, layer_idx: int) -> int:
        return int(self.lib.tripool_expert_layer_va(int(expert_id), int(layer_idx)))

    def expert_layer_stride_bytes(self, expert_id: int) -> int:
        return int(self.lib.tripool_expert_layer_stride_bytes(int(expert_id)))

    def expert_layer_uint8_view(self, expert_id: int, layer_idx: int) -> Any:
        ptr = self.expert_layer_va(expert_id, layer_idx)
        nbytes = self.expert_layer_stride_bytes(expert_id)
        view = _CudaVAView(ptr, nbytes, self)
        if self._torch_cuda_ready():
            assert _TORCH is not None
            return _TORCH.as_tensor(view, device=f"cuda:{self.device}")
        return view

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

    def _track_pinned_mirror(self, mirror: NumaMemfdMirror) -> None:
        cls = type(self)
        with cls._so_lock:
            bucket = cls._so_mirrors.setdefault(self._so_key, {})
            bucket[id(mirror)] = weakref.ref(mirror)

    def _shutdown_impl(self, *, from_del: bool = False) -> None:
        if getattr(self, "_closed", True):
            return
        rc = 0
        try:
            cls = type(self)
            with cls._so_lock:
                if self._closed:
                    return
                self._closed = True
                so_key = getattr(self, "_so_key", (-1, -1))
                epoch_state = cls._so_epochs.get(so_key)
                same_active_epoch = (
                    epoch_state is not None
                    and epoch_state[1] > 0
                    and epoch_state[0] == getattr(self, "_vmm_epoch", -1)
                )
                should_shutdown_cpp = False
                if same_active_epoch:
                    assert epoch_state is not None
                    self._release_raw_transient_ptrs(0)
                    if from_del and epoch_state[1] > 1:
                        epoch_state[1] -= 1
                    else:
                        epoch_state[0] += 1
                        epoch_state[1] = 0
                        should_shutdown_cpp = True
                else:
                    self._raw_transient_ptrs.clear()

                self._pluggable = None
                self._registered_expert_bytes = 0
                if hasattr(self, "_swap_mu"):
                    with self._swap_mu:
                        self._registered_kv_caches.clear()
                if should_shutdown_cpp:
                    mirror_bucket = cls._so_mirrors.pop(so_key, {})
                    for ref in mirror_bucket.values():
                        m_obj = ref()
                        if m_obj is not None:
                            try:
                                m_obj.unpin_cuda()
                            except Exception:
                                pass
                    if not from_del and gc is not None:
                        gc.collect()
                    rc = int(self.lib.tripool_vmm_shutdown())
        finally:
            self._closed = True
        if rc != 0 and not from_del:
            raise RuntimeError(f"tripool_vmm_shutdown failed (rc={rc})")

    def shutdown(self) -> None:
        self._shutdown_impl(from_del=False)


class SimulatedTriPoolBackend(_VmmKvSwapMixin):
    """Local-substitutable CPU VMM backend enforcing bidirectional handle mapping, real-time `kv_free_bytes`, and Host Pinned KV swap."""

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
        self.device = int(device)
        self.super_handle_mib = int(super_handle_mib)
        self.super_handle_bytes = self.super_handle_mib * MIB
        self.min_granularity_bytes = 2 * MIB
        self.allow_overflow = bool(allow_overflow)
        self.materialize_va_bytes = bool(materialize_va_bytes)
        self._init_swap_governance(
            kv_swap_high_watermark=kv_swap_high_watermark,
            kv_swap_low_watermark=kv_swap_low_watermark,
            kv_swap_min_idle_s=kv_swap_min_idle_s,
            max_pool_bytes=max_pool_bytes,
        )

        self._super_total = max(0, int(super_pool_mib) // self.super_handle_mib)
        self._super_free = self._super_total
        self._fine_total = max(0, int(fine_kv_pool_mib) // 2)
        self._fine_free = self._fine_total
        self._base_fine_pages = self._fine_total
        self._pages_per_super = max(1, self.super_handle_bytes // self.min_granularity_bytes)

        self._kv_allocated: set[int] = set()
        self._kv_regions: dict[int, dict[str, Any]] = {}
        self._next_region_id = 0
        self._next_kv_slot = 0
        self._kv_peak = 0
        self._carves = 0
        self._fine_to_super_bundles = 0
        self._overflow_creates = 0

        self._kv_swapped_pages = 0
        self._kv_swapped_bytes = 0
        self._kv_swap_outs_total = 0
        self._kv_swap_ins_total = 0
        self._kv_swap_d2h_bytes_total = 0
        self._kv_swap_h2d_bytes_total = 0
        self._kv_host_pinned_pool_bytes = 0
        self._evictable_clock = 0

        self._in_scope = False
        self._transient_current = 0
        self._transient_slot_bundles: list[bool] = []
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

    def _release_slots(self, slot_bundles: list[bool]) -> None:
        for is_bundle in slot_bundles:
            if is_bundle:
                self._fine_free += self._pages_per_super
            else:
                self._super_free += 1
        slot_bundles.clear()

    def _evict_one_unpinned_tail_layer(self, for_kv: bool = False, exclude_eid: int = -1) -> bool:
        for eid, exp in self._experts.items():
            if exp["pinned"] or eid == exclude_eid:
                continue
            for lid in range(exp["num_layers"] - 1, -1, -1):
                if exp["resident"][lid] or exp["pending_async"][lid]:
                    exp["resident"][lid] = False
                    exp["pending_async"][lid] = False
                    self._release_slots(exp["slot_bundles"][lid])
                    if for_kv:
                        self._auto_evictions_kv += 1
                    else:
                        self._auto_evictions_transient += 1
                    return True
        return False

    def _registered_expert_total_bytes(self) -> int:
        return sum(int(e["num_layers"]) * int(e["stride_bytes"]) for e in self._experts.values())

    def _total_evictable_thinking_pages(self, exclude_rid: int = -1) -> int:
        return sum(
            int(r["mapped_pages"])
            for rid, r in self._kv_regions.items()
            if rid != exclude_rid and not r["swapped"] and r.get("evictable", False) and r["mapped_pages"] > 0
        )

    def _evict_one_thinking_kv_region(self, exclude_rid: int = -1) -> bool:
        best_rid: int | None = None
        best_seq: int | None = None
        for rid, reg in self._kv_regions.items():
            if rid == exclude_rid or reg["swapped"] or not reg.get("evictable", False) or reg["mapped_pages"] <= 0:
                continue
            seq = int(reg.get("evictable_seq", 0))
            if best_seq is None or seq < best_seq:
                best_seq = seq
                best_rid = rid
        if best_rid is None:
            return False
        victim = self._kv_regions[best_rid]
        valid_b = int(victim.get("evictable_valid_bytes", 0))
        return self.kv_region_swap_out_async(best_rid, valid_b) > 0

    def _acquire_super_slots(
        self,
        count: int,
        *,
        for_transient: bool = False,
        for_kv: bool = False,
        exclude_eid: int = -1,
    ) -> list[bool]:
        acquired: list[bool] = []
        try:
            for _ in range(int(count)):
                while True:
                    if self._super_free > 0:
                        self._super_free -= 1
                        acquired.append(False)
                        break
                    # Reclaim carved surplus fine pages (> _base_fine_pages) before evicting resident experts!
                    if self._fine_free >= self._base_fine_pages + self._pages_per_super:
                        self._fine_free -= self._pages_per_super
                        self._fine_to_super_bundles += 1
                        acquired.append(True)
                        break
                    if (for_transient or for_kv or exclude_eid >= 0) and self._evict_one_unpinned_tail_layer(
                        for_kv=for_kv, exclude_eid=exclude_eid
                    ):
                        continue
                    # Once unpinned experts cannot supply more super-handles, bundle any remaining 32x2MiB pages!
                    if self._fine_free >= self._pages_per_super:
                        self._fine_free -= self._pages_per_super
                        self._fine_to_super_bundles += 1
                        acquired.append(True)
                        break
                    if (
                        self.allow_overflow
                        or (self._fine_free + self._total_evictable_thinking_pages() >= self._pages_per_super)
                    ) and self._evict_one_thinking_kv_region():
                        continue
                    if self.allow_overflow:
                        if self._fine_free > 0:
                            deficit = self._pages_per_super - self._fine_free
                            self._fine_total += deficit
                            self._fine_free += deficit
                            self._overflow_creates += deficit
                            self._fine_free -= self._pages_per_super
                            self._fine_to_super_bundles += 1
                            acquired.append(True)
                        else:
                            self._super_total += 1
                            self._overflow_creates += 1
                            acquired.append(False)
                        break
                    raise RuntimeError(
                        f"SimulatedTriPoolBackend exhausted: need {count} super-slots, "
                        f"super_free={self._super_free}, fine_free={self._fine_free}, overflow=False"
                    )
            return acquired
        except BaseException:
            self._release_slots(acquired)
            raise

    def _compute_capacities(self) -> tuple[int, int, int, int, int]:
        pool_total = self._super_total * self.super_handle_bytes + self._fine_total * self.min_granularity_bytes
        pool_free = self._super_free * self.super_handle_bytes + self._fine_free * self.min_granularity_bytes
        unpinned_exp = 0
        full_exp = 0
        for exp in self._experts.values():
            full_exp += exp["num_layers"] * exp["stride_bytes"]
            if not exp["pinned"]:
                active_layers = sum(1 for r, p in zip(exp["resident"], exp["pending_async"]) if (r or p))
                unpinned_exp += active_layers * exp["stride_bytes"]
        overflow_extra = (8 * 1024 * MIB) if self.allow_overflow else 0
        kv_free = pool_free + unpinned_exp + overflow_extra
        kv_used = self._cur_kv_pages_total() * self.min_granularity_bytes
        tr_used = self._transient_current * self.super_handle_bytes
        dit_headroom = max(0, (pool_total + overflow_extra) - kv_used - tr_used - full_exp)
        return pool_total, pool_free, unpinned_exp, kv_free, dit_headroom

    def kv_free_bytes(self, for_dit: bool = False) -> int:
        st = self.stats_dict()
        raw = int(st["kv_dit_headroom_bytes"] if for_dit else st["kv_free_bytes"])
        return self._clamp_free_by_max_pool(raw, st, for_dit=for_dit)

    def stats_dict(self) -> dict[str, Any]:
        exp_handles = sum(
            sum(1 for r, p in zip(e["resident"], e["pending_async"]) if (r or p)) * e["handles_per_layer"]
            for e in self._experts.values()
        )
        exp_layers = sum(sum(1 for r in e["resident"] if r) for e in self._experts.values())
        exp_fine_bundles = sum(
            sum(1 for b in bundles if b) for e in self._experts.values() for bundles in e["slot_bundles"]
        )
        tr_fine_bundles = sum(1 for b in self._transient_slot_bundles if b)
        kv_cur_pages = self._cur_kv_pages_total()
        pool_tot, pool_free, unpinned_exp, kv_free, dit_headroom = self._compute_capacities()
        d = {
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
            "fine_to_super_bundles": self._fine_to_super_bundles,
            "transient_fine_bundles_current": tr_fine_bundles,
            "expert_fine_bundles_current": exp_fine_bundles,
            "pool_total_bytes": pool_tot,
            "physical_pool_bytes": pool_tot,
            "pool_free_bytes": pool_free,
            "expert_unpinned_bytes": unpinned_exp,
            "kv_free_bytes": kv_free,
            "kv_dit_headroom_bytes": dit_headroom,
            "kv_swapped_pages_current": self._kv_swapped_pages,
            "kv_swapped_bytes_current": self._kv_swapped_bytes,
            "kv_swap_outs_total": self._kv_swap_outs_total,
            "kv_swap_ins_total": self._kv_swap_ins_total,
            "kv_swap_d2h_bytes_total": self._kv_swap_d2h_bytes_total,
            "kv_swap_h2d_bytes_total": self._kv_swap_h2d_bytes_total,
            "kv_host_pinned_pool_bytes": self._kv_host_pinned_pool_bytes,
        }
        if self.max_pool_bytes is not None:
            d["kv_free_bytes"] = self._clamp_free_by_max_pool(int(d["kv_free_bytes"]), d, for_dit=False)
            d["kv_dit_headroom_bytes"] = self._clamp_free_by_max_pool(
                int(d["kv_dit_headroom_bytes"]), d, for_dit=True
            )
        return d

    def _pop_fine_page(self, exclude_rid: int = -1) -> None:
        while self._fine_free == 0:
            if self._super_free > 0:
                self._super_free -= 1
                self._super_total -= 1
                self._fine_total += self._pages_per_super
                self._fine_free += self._pages_per_super
                self._carves += 1
                break
            if self._evict_one_unpinned_tail_layer(for_kv=True):
                continue
            if self._evict_one_thinking_kv_region(exclude_rid=exclude_rid):
                continue
            if self.allow_overflow:
                self._fine_total += 1
                self._fine_free += 1
                self._overflow_creates += 1
                break
            raise RuntimeError(
                f"SimulatedTriPoolBackend physical pool exhausted for 2 MiB KV page "
                f"(super_free={self._super_free}, fine_free={self._fine_free}, overflow=False)"
            )
        self._fine_free -= 1

    def _cur_kv_pages_total(self) -> int:
        return len(self._kv_allocated) + sum(r["mapped_pages"] for r in self._kv_regions.values())

    def kv_alloc_page(self) -> int:
        try:
            self._pop_fine_page()
        except RuntimeError:
            self.reclaim_thinking_kv_bytes(self.min_granularity_bytes, sync=True)
            self._pop_fine_page()
        slot = self._next_kv_slot
        self._next_kv_slot += 1
        self._kv_allocated.add(slot)
        self._kv_peak = max(self._kv_peak, self._cur_kv_pages_total())
        return slot

    def kv_free_page(self, slot_idx: int, stream_ptr: int = 0) -> None:
        if self._closed:
            return
        if slot_idx in self._kv_allocated:
            self._kv_allocated.remove(slot_idx)
            self._fine_free += 1

    def kv_region_alloc(self, max_va_bytes: int) -> int:
        gran = self.min_granularity_bytes
        cap = max(gran, ((int(max_va_bytes) + gran - 1) // gran) * gran)
        rid = self._next_region_id
        self._next_region_id += 1
        # Demand-zero anonymous mmap reserves stable CPU virtual address space (0 B RSS until touched!)
        mm = mmap.mmap(-1, cap, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        c_u8 = ctypes.c_uint8.from_buffer(mm)
        va_base = int(ctypes.addressof(c_u8))
        del c_u8
        self._kv_regions[rid] = {
            "cap_bytes": cap,
            "mapped_pages": 0,
            "va_base": va_base,
            "mm": mm,
            "swapped": False,
            "swapped_pages": 0,
            "swapped_valid_bytes": 0,
            "swap_buf": bytearray(0),
            "evictable": False,
            "evictable_valid_bytes": 0,
            "evictable_seq": 0,
        }
        return rid

    def kv_region_set_evictable(
        self,
        region_id: int,
        evictable: bool,
        valid_bytes: int = 0,
        stream_ptr: int = 0,
    ) -> None:
        if self._closed:
            return
        reg = self._kv_regions.get(int(region_id))
        if reg is None:
            return
        if evictable:
            self._evictable_clock += 1
            reg["evictable"] = True
            reg["evictable_valid_bytes"] = max(0, int(valid_bytes))
            reg["evictable_seq"] = self._evictable_clock
        else:
            reg["evictable"] = False
            reg["evictable_valid_bytes"] = 0

    def kv_region_ensure(self, region_id: int, needed_bytes: int, stream_ptr: int = 0) -> int:
        rid_i = int(region_id)
        reg = self._kv_regions[rid_i]
        reg["evictable"] = False
        gran = self.min_granularity_bytes
        eff = max(1, int(needed_bytes))
        needed_pages = (eff + gran - 1) // gran
        if needed_pages * gran > reg["cap_bytes"]:
            raise MemoryError(f"simulated kv_region {region_id} needed {eff} > cap {reg['cap_bytes']}")

        if reg["swapped"]:
            self.kv_region_swap_in_async(rid_i, stream_ptr=stream_ptr)

        old_pages = reg["mapped_pages"]
        if needed_pages > old_pages:
            delta = needed_pages - old_pages
            pulled = 0
            try:
                for _ in range(delta):
                    self._pop_fine_page(exclude_rid=rid_i)
                    pulled += 1
            except RuntimeError as exc:
                self._fine_free += pulled
                raise MemoryError(str(exc)) from exc
            reg["mapped_pages"] = needed_pages
            self._kv_peak = max(self._kv_peak, self._cur_kv_pages_total())
        return int(reg["va_base"])

    def kv_region_swap_out_async(self, region_id: int, valid_bytes: int, stream_ptr: int = 0) -> int:
        reg = self._kv_regions[int(region_id)]
        reg["evictable"] = False
        if reg["swapped"] or reg["mapped_pages"] == 0:
            return 0
        n_pages = int(reg["mapped_pages"])
        mapped_b = n_pages * self.min_granularity_bytes
        copy_b = min(max(0, int(valid_bytes)), mapped_b)
        mm: mmap.mmap = reg["mm"]

        if copy_b > 0:
            if len(reg["swap_buf"]) < copy_b:
                grow = copy_b - len(reg["swap_buf"])
                reg["swap_buf"].extend(b"\x00" * grow)
                self._kv_host_pinned_pool_bytes += grow
            mv_src = memoryview(mm)
            memoryview(reg["swap_buf"])[:copy_b] = mv_src[:copy_b]
            # Scrub unmapped simulated GPU VA pages so swap-in must genuinely restore data!
            mv_src[:copy_b] = b"\x00" * copy_b
            del mv_src

        self._fine_free += n_pages
        reg["mapped_pages"] = 0
        reg["swapped"] = True
        reg["swapped_pages"] = n_pages
        reg["swapped_valid_bytes"] = copy_b
        self._kv_swapped_pages += n_pages
        self._kv_swapped_bytes += copy_b
        self._kv_swap_outs_total += 1
        self._kv_swap_d2h_bytes_total += copy_b
        return n_pages

    kv_region_swap_out = kv_region_swap_out_async

    def kv_region_swap_in_async(self, region_id: int, stream_ptr: int = 0) -> int:
        rid_i = int(region_id)
        reg = self._kv_regions[rid_i]
        reg["evictable"] = False
        if not reg["swapped"]:
            return 0
        n_pages = int(reg["swapped_pages"])
        valid_b = int(reg["swapped_valid_bytes"])
        pulled = 0
        try:
            for _ in range(n_pages):
                self._pop_fine_page(exclude_rid=rid_i)
                pulled += 1
        except RuntimeError as exc:
            self._fine_free += pulled
            raise MemoryError(str(exc)) from exc

        if valid_b > 0:
            mm: mmap.mmap = reg["mm"]
            mv_dst = memoryview(mm)
            mv_dst[:valid_b] = memoryview(reg["swap_buf"])[:valid_b]
            del mv_dst

        reg["mapped_pages"] = n_pages
        reg["swapped"] = False
        self._kv_swapped_pages = max(0, self._kv_swapped_pages - n_pages)
        self._kv_swapped_bytes = max(0, self._kv_swapped_bytes - valid_b)
        self._kv_swap_ins_total += 1
        self._kv_swap_h2d_bytes_total += valid_b
        self._kv_peak = max(self._kv_peak, self._cur_kv_pages_total())
        return n_pages

    kv_region_swap_in = kv_region_swap_in_async

    def kv_region_is_swapped(self, region_id: int) -> bool:
        if self._closed:
            return False
        reg = self._kv_regions.get(int(region_id))
        return bool(reg["swapped"]) if reg is not None else False

    def kv_region_swapped_bytes(self, region_id: int) -> int:
        if self._closed:
            return 0
        reg = self._kv_regions.get(int(region_id))
        return int(reg["swapped_valid_bytes"]) if (reg is not None and reg["swapped"]) else 0

    def flush_kv_swaps(self) -> None:
        return None

    def kv_region_va(self, region_id: int) -> int:
        return int(self._kv_regions[int(region_id)]["va_base"])

    def kv_region_mapped_bytes(self, region_id: int) -> int:
        if self._closed:
            return 0
        reg = self._kv_regions.get(int(region_id))
        if reg is None or reg["swapped"]:
            return 0
        return int(reg["mapped_pages"] * self.min_granularity_bytes)

    def kv_region_uint8_view(self, region_id: int, nbytes: int | None = None) -> memoryview:
        reg = self._kv_regions[int(region_id)]
        if reg["swapped"]:
            raise RuntimeError(f"simulated kv_region {region_id} is currently swapped out to Host Pinned DRAM")
        mapped = reg["mapped_pages"] * self.min_granularity_bytes
        target = mapped if nbytes is None else min(mapped, int(nbytes))
        return memoryview(reg["mm"])[:target]

    def kv_region_free(self, region_id: int, stream_ptr: int = 0) -> None:
        if self._closed:
            return
        reg = self._kv_regions.pop(int(region_id), None)
        if reg is not None:
            if reg["swapped"]:
                self._kv_swapped_pages = max(0, self._kv_swapped_pages - int(reg["swapped_pages"]))
                self._kv_swapped_bytes = max(0, self._kv_swapped_bytes - int(reg["swapped_valid_bytes"]))
            else:
                self._fine_free += int(reg["mapped_pages"])
            buf_len = len(reg["swap_buf"])
            if buf_len > 0:
                self._kv_host_pinned_pool_bytes = max(0, self._kv_host_pinned_pool_bytes - buf_len)
            reg["swap_buf"] = bytearray(0)
            mm = reg.pop("mm", None)
            if mm is not None:
                try:
                    mm.close()
                except BufferError:
                    gc.collect()
                    try:
                        mm.close()
                    except BufferError:
                        # Caller stack still holds a slice view; mmap.__del__ unmaps once view drops.
                        pass

    def allocate_transient_mib(self, size_mib: int) -> None:
        if not self._in_scope:
            raise RuntimeError("allocate_transient_mib requires an active transient scope")
        handles = max(1, (int(size_mib) + self.super_handle_mib - 1) // self.super_handle_mib)
        req_bytes = handles * self.super_handle_bytes
        self.maybe_evict_thinking_kv_on_high_watermark(sync=True, extra_needed_bytes=req_bytes)
        try:
            slots = self._acquire_super_slots(handles, for_transient=True)
        except RuntimeError:
            self.reclaim_thinking_kv_bytes(req_bytes, sync=True)
            slots = self._acquire_super_slots(handles, for_transient=True)
        self._transient_slot_bundles.extend(slots)
        self._transient_current += handles
        self._transient_peak = max(self._transient_peak, self._transient_current)

    def run_in_transient_scope(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self.maybe_evict_thinking_kv_on_high_watermark(sync=False)
        self._in_scope = True
        try:
            return fn(*args, **kwargs)
        finally:
            self._release_slots(self._transient_slot_bundles)
            self._transient_current = 0
            self._transient_flushes += 1
            self._in_scope = False

    @contextmanager
    def transient_scope(self):
        self.maybe_evict_thinking_kv_on_high_watermark(sync=False)
        self._in_scope = True
        try:
            yield self
        finally:
            self._release_slots(self._transient_slot_bundles)
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
            "slot_bundles": [[] for _ in range(num_layers)],
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
                slots = self._acquire_super_slots(exp["handles_per_layer"], exclude_eid=int(expert_id))
                exp["slot_bundles"][lid] = slots
                self._copy_mirror_to_va(exp, lid)
            exp["pending_async"][lid] = False
            exp["resident"][lid] = True

    def prefetch_expert_async(self, expert_id: int, start_layer: int, end_layer: int) -> int:
        exp = self._experts[expert_id]
        queued = 0
        for lid in range(int(start_layer), int(end_layer)):
            if exp["resident"][lid] or exp["pending_async"][lid]:
                continue
            slots = self._acquire_super_slots(exp["handles_per_layer"], exclude_eid=int(expert_id))
            exp["slot_bundles"][lid] = slots
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
                self._release_slots(exp["slot_bundles"][lid])
                self._manual_evictions += 1
                evicted += 1
        return evicted

    def expert_resident_count(self, expert_id: int) -> int:
        exp = self._experts[expert_id]
        return sum(1 for r in exp["resident"] if r)

    def expert_is_resident(self, expert_id: int, layer_idx: int) -> bool:
        return bool(self._experts[expert_id]["resident"][layer_idx])

    def shutdown(self) -> None:
        for rid in list(self._kv_regions.keys()):
            self.kv_region_free(rid)
        with self._swap_mu:
            self._registered_kv_caches.clear()
        self._experts.clear()
        self._kv_allocated.clear()
        self._kv_regions.clear()
        self._closed = True
