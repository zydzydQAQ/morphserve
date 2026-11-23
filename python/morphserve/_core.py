"""Single authoritative `ctypes` binding layer to `libmorphserve_core.so` (`morphserve/c_api.h`)."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import subprocess
import threading
from typing import Any


class UnifiedVmmStatsC(ctypes.Structure):
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


class MsVmmConfigC(ctypes.Structure):
    _fields_ = [
        ("device", ctypes.c_int),
        ("super_pool_mib", ctypes.c_size_t),
        ("fine_kv_pool_mib", ctypes.c_size_t),
        ("super_handle_mib", ctypes.c_size_t),
        ("transient_va_gib", ctypes.c_size_t),
        ("kv_va_gib", ctypes.c_size_t),
        ("allow_overflow", ctypes.c_int),
        ("simulated", ctypes.c_int),
        ("materialize_sim_va", ctypes.c_int),
        ("kv_swap_high_watermark", ctypes.c_double),
        ("kv_swap_low_watermark", ctypes.c_double),
        ("kv_swap_min_idle_s", ctypes.c_double),
        ("max_pool_bytes", ctypes.c_uint64),
    ]


_LOAD_LOCK = threading.RLock()
_LIB_CACHE: dict[tuple[int, int], tuple[ctypes.CDLL, Path]] = {}


def _pkg_parent_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_core_so(
    so_path: Path | str | None = None,
    *,
    lib_dir: Path | str | None = None,
    prefer_vmm_alias: bool = False,
    auto_build: bool = True,
) -> Path:
    """Resolve `libmorphserve_core.so` (or `libtripool_vmm.so`) without mutating read-only staged builds."""
    names = ("libtripool_vmm.so", "libmorphserve_core.so") if prefer_vmm_alias else ("libmorphserve_core.so", "libtripool_vmm.so")

    if so_path is not None:
        p = Path(so_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"shared library not found at {p}")

    if lib_dir is not None:
        base = Path(lib_dir)
        for nm in names:
            cand = base / nm
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"neither libmorphserve_core.so nor libtripool_vmm.so found in {base}")

    candidates: list[Path] = []
    env_dir = os.environ.get("MORPHSERVE_LIB_DIR", "").strip()
    if env_dir:
        eb = Path(env_dir)
        for nm in names:
            candidates.append(eb / nm)

    root = _pkg_parent_root()
    for sub in ("lib", "build/install/lib", "build/cmake"):
        for nm in names:
            candidates.append(root / sub / nm)

    local_tmp = os.environ.get("MORPHSERVE_LOCAL_TMP", "").strip()
    if local_tmp:
        for nm in names:
            candidates.append(Path(local_tmp) / "native_install" / "lib" / nm)

    for cand in candidates:
        if cand.is_file():
            return cand

    if auto_build:
        with _LOAD_LOCK:
            for cand in candidates:
                if cand.is_file():
                    return cand
            build_sh = root / "deploy" / "kairos" / "scripts" / "build_stage.sh"
            hdr = root / "include" / "morphserve" / "c_api.h"
            if build_sh.is_file() and hdr.is_file():
                if os.access(root, os.W_OK):
                    b_tmp = root / "build" / "stage_cmake"
                    i_tmp = root / "build" / "install"
                else:
                    scratch = Path(local_tmp) if local_tmp else Path(f"/tmp/morphserve_native_{os.getuid()}")
                    b_tmp = scratch / "native_cmake"
                    i_tmp = scratch / "native_install"
                env = dict(os.environ, SRC_DIR=str(root), BUILD_TMP=str(b_tmp), INSTALL_TMP=str(i_tmp))
                proc = subprocess.run(
                    ["bash", str(build_sh)],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"auto-build of libmorphserve_core.so failed:\n{proc.stderr}")
                for nm in names:
                    built = i_tmp / "lib" / nm
                    if built.is_file():
                        return built

    raise FileNotFoundError(f"libmorphserve_core.so not found; checked {[str(c) for c in candidates]}")


def _bind_all_symbols(L: ctypes.CDLL) -> None:
    # Error & memory release
    L.ms_last_error.argtypes = []
    L.ms_last_error.restype = ctypes.c_char_p
    L.ms_free_cstr.argtypes = [ctypes.c_void_p]
    L.ms_free_cstr.restype = None
    L.ms_free_bytes.argtypes = [ctypes.c_void_p]
    L.ms_free_bytes.restype = None

    # Instance-owned TriPoolVMM (`ms_vmm_*`)
    L.ms_vmm_create.argtypes = [ctypes.POINTER(MsVmmConfigC)]
    L.ms_vmm_create.restype = ctypes.c_void_p
    L.ms_vmm_destroy.argtypes = [ctypes.c_void_p]
    L.ms_vmm_destroy.restype = None
    L.ms_vmm_shutdown.argtypes = [ctypes.c_void_p]
    L.ms_vmm_shutdown.restype = ctypes.c_int
    L.ms_vmm_is_simulated.argtypes = [ctypes.c_void_p]
    L.ms_vmm_is_simulated.restype = ctypes.c_int
    L.ms_vmm_device.argtypes = [ctypes.c_void_p]
    L.ms_vmm_device.restype = ctypes.c_int
    L.ms_vmm_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(UnifiedVmmStatsC)]
    L.ms_vmm_get_stats.restype = None
    L.ms_vmm_kv_free_bytes.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
    L.ms_vmm_kv_free_bytes.restype = ctypes.c_uint64
    L.ms_vmm_sync_kv_free_bytes.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
    L.ms_vmm_sync_kv_free_bytes.restype = ctypes.c_uint64

    L.ms_vmm_scope_begin.argtypes = [ctypes.c_void_p]
    L.ms_vmm_scope_begin.restype = None
    L.ms_vmm_scope_end.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    L.ms_vmm_scope_end.restype = ctypes.c_int
    L.ms_vmm_transient_alloc.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_void_p]
    L.ms_vmm_transient_alloc.restype = ctypes.c_uint64
    L.ms_vmm_transient_free.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_void_p]
    L.ms_vmm_transient_free.restype = None

    L.ms_vmm_kv_va_base.argtypes = [ctypes.c_void_p]
    L.ms_vmm_kv_va_base.restype = ctypes.c_uint64
    L.ms_vmm_kv_alloc_page.argtypes = [ctypes.c_void_p]
    L.ms_vmm_kv_alloc_page.restype = ctypes.c_int32
    L.ms_vmm_kv_free_page.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
    L.ms_vmm_kv_free_page.restype = ctypes.c_int

    L.ms_vmm_kv_session_create.argtypes = [ctypes.c_void_p]
    L.ms_vmm_kv_session_create.restype = ctypes.c_uint64
    L.ms_vmm_kv_session_destroy.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    L.ms_vmm_kv_session_destroy.restype = None
    L.ms_vmm_kv_session_touch.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    L.ms_vmm_kv_session_touch.restype = None
    L.ms_vmm_kv_session_mark_active.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
    L.ms_vmm_kv_session_mark_active.restype = ctypes.c_int
    L.ms_vmm_kv_session_mark_thinking.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    L.ms_vmm_kv_session_mark_thinking.restype = ctypes.c_size_t
    L.ms_vmm_reclaim_thinking_kv_bytes.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
    ]
    L.ms_vmm_reclaim_thinking_kv_bytes.restype = ctypes.c_size_t
    L.ms_vmm_maybe_evict_watermark.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    L.ms_vmm_maybe_evict_watermark.restype = ctypes.c_size_t

    L.ms_vmm_kv_region_alloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64]
    L.ms_vmm_kv_region_alloc.restype = ctypes.c_int32
    L.ms_vmm_kv_region_ensure.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_size_t, ctypes.c_void_p]
    L.ms_vmm_kv_region_ensure.restype = ctypes.c_uint64
    L.ms_vmm_kv_region_va.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    L.ms_vmm_kv_region_va.restype = ctypes.c_uint64
    L.ms_vmm_kv_region_mapped_bytes.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    L.ms_vmm_kv_region_mapped_bytes.restype = ctypes.c_size_t
    L.ms_vmm_kv_region_free.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
    L.ms_vmm_kv_region_free.restype = ctypes.c_int
    L.ms_vmm_kv_region_set_evictable.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    L.ms_vmm_kv_region_set_evictable.restype = ctypes.c_int
    L.ms_vmm_kv_region_swap_out_async.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_size_t, ctypes.c_void_p]
    L.ms_vmm_kv_region_swap_out_async.restype = ctypes.c_int
    L.ms_vmm_kv_region_swap_in_async.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
    L.ms_vmm_kv_region_swap_in_async.restype = ctypes.c_int
    L.ms_vmm_kv_region_is_swapped.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    L.ms_vmm_kv_region_is_swapped.restype = ctypes.c_int
    L.ms_vmm_kv_region_swapped_bytes.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    L.ms_vmm_kv_region_swapped_bytes.restype = ctypes.c_size_t
    L.ms_vmm_kv_flush_swaps.argtypes = [ctypes.c_void_p]
    L.ms_vmm_kv_flush_swaps.restype = ctypes.c_int

    L.ms_vmm_expert_register.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    L.ms_vmm_expert_register.restype = ctypes.c_int
    L.ms_vmm_expert_va_base.argtypes = [ctypes.c_void_p, ctypes.c_int]
    L.ms_vmm_expert_va_base.restype = ctypes.c_uint64
    L.ms_vmm_expert_layer_va.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    L.ms_vmm_expert_layer_va.restype = ctypes.c_uint64
    L.ms_vmm_expert_layer_stride_bytes.argtypes = [ctypes.c_void_p, ctypes.c_int]
    L.ms_vmm_expert_layer_stride_bytes.restype = ctypes.c_size_t
    L.ms_vmm_expert_set_pinned.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    L.ms_vmm_expert_set_pinned.restype = ctypes.c_int
    L.ms_vmm_expert_set_layer_used_bytes.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_size_t]
    L.ms_vmm_expert_set_layer_used_bytes.restype = ctypes.c_int
    L.ms_vmm_expert_load_layers_sync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    L.ms_vmm_expert_load_layers_sync.restype = ctypes.c_int
    L.ms_vmm_expert_prefetch_async.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    L.ms_vmm_expert_prefetch_async.restype = ctypes.c_int
    L.ms_vmm_expert_wait_layer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    L.ms_vmm_expert_wait_layer.restype = ctypes.c_int
    L.ms_vmm_expert_evict_layer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    L.ms_vmm_expert_evict_layer.restype = ctypes.c_int
    L.ms_vmm_expert_evict_range.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    L.ms_vmm_expert_evict_range.restype = ctypes.c_int
    L.ms_vmm_expert_is_resident.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    L.ms_vmm_expert_is_resident.restype = ctypes.c_int
    L.ms_vmm_expert_resident_count.argtypes = [ctypes.c_void_p, ctypes.c_int]
    L.ms_vmm_expert_resident_count.restype = ctypes.c_int

    L.ms_cuda_memcpy_htod.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    L.ms_cuda_memcpy_htod.restype = ctypes.c_int
    L.ms_cuda_memcpy_dtoh.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    L.ms_cuda_memcpy_dtoh.restype = ctypes.c_int

    # NumaMemfdMirror (`ms_mirror_*`)
    L.ms_mirror_create.argtypes = [ctypes.c_size_t, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
    L.ms_mirror_create.restype = ctypes.c_void_p
    L.ms_mirror_attach_proc_fd.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_int]
    L.ms_mirror_attach_proc_fd.restype = ctypes.c_void_p
    L.ms_mirror_destroy.argtypes = [ctypes.c_void_p]
    L.ms_mirror_destroy.restype = None
    L.ms_mirror_close.argtypes = [ctypes.c_void_p]
    L.ms_mirror_close.restype = None
    L.ms_mirror_ptr.argtypes = [ctypes.c_void_p]
    L.ms_mirror_ptr.restype = ctypes.c_uint64
    L.ms_mirror_nbytes.argtypes = [ctypes.c_void_p]
    L.ms_mirror_nbytes.restype = ctypes.c_size_t
    L.ms_mirror_fd.argtypes = [ctypes.c_void_p]
    L.ms_mirror_fd.restype = ctypes.c_int
    L.ms_mirror_numa_node.argtypes = [ctypes.c_void_p]
    L.ms_mirror_numa_node.restype = ctypes.c_int
    L.ms_mirror_inode.argtypes = [ctypes.c_void_p]
    L.ms_mirror_inode.restype = ctypes.c_uint64
    L.ms_mirror_numa_bound.argtypes = [ctypes.c_void_p]
    L.ms_mirror_numa_bound.restype = ctypes.c_int
    L.ms_mirror_is_pinned.argtypes = [ctypes.c_void_p]
    L.ms_mirror_is_pinned.restype = ctypes.c_int
    L.ms_mirror_pin_cuda.argtypes = [ctypes.c_void_p]
    L.ms_mirror_pin_cuda.restype = ctypes.c_int
    L.ms_mirror_pin_cuda_ex.argtypes = [ctypes.c_void_p, ctypes.c_int]
    L.ms_mirror_pin_cuda_ex.restype = ctypes.c_int
    L.ms_mirror_unpin_cuda.argtypes = [ctypes.c_void_p]
    L.ms_mirror_unpin_cuda.restype = None
    L.ms_mirror_stream_h2d_gbps.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    L.ms_mirror_stream_h2d_gbps.restype = ctypes.c_double
    L.ms_mirror_write.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    L.ms_mirror_write.restype = ctypes.c_int
    L.ms_mirror_read.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    L.ms_mirror_read.restype = ctypes.c_int

    # HalfNodeH2DPacer (`ms_pacer_*`)
    L.ms_pacer_create.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_char_p,
    ]
    L.ms_pacer_create.restype = ctypes.c_void_p
    L.ms_pacer_destroy.argtypes = [ctypes.c_void_p]
    L.ms_pacer_destroy.restype = None
    L.ms_pacer_max_concurrent.argtypes = [ctypes.c_void_p]
    L.ms_pacer_max_concurrent.restype = ctypes.c_int
    L.ms_pacer_group_of.argtypes = [ctypes.c_void_p, ctypes.c_int]
    L.ms_pacer_group_of.restype = ctypes.c_int
    L.ms_pacer_predicted_wait_ms.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
    L.ms_pacer_predicted_wait_ms.restype = ctypes.c_double
    L.ms_pacer_reserve_slot.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_double]
    L.ms_pacer_reserve_slot.restype = ctypes.c_double
    L.ms_pacer_reset_slots.argtypes = [ctypes.c_void_p]
    L.ms_pacer_reset_slots.restype = None
    L.ms_pacer_acquire_thread.argtypes = [ctypes.c_void_p, ctypes.c_int]
    L.ms_pacer_acquire_thread.restype = None
    L.ms_pacer_release_thread.argtypes = [ctypes.c_void_p, ctypes.c_int]
    L.ms_pacer_release_thread.restype = None
    L.ms_pacer_acquire_process_flock.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    L.ms_pacer_acquire_process_flock.restype = ctypes.c_int
    L.ms_pacer_release_process_flock.argtypes = [ctypes.c_int]
    L.ms_pacer_release_process_flock.restype = None

    # PeerByteTransport (`ms_nccl_*`, `ms_tcp_*`, `ms_mem_bus_*`, `ms_transport_*`)
    L.ms_find_free_tcp_port.argtypes = []
    L.ms_find_free_tcp_port.restype = ctypes.c_int
    L.ms_nccl_generate_uid.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8)]
    L.ms_nccl_generate_uid.restype = ctypes.c_int
    L.ms_nccl_create.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_char_p,
    ]
    L.ms_nccl_create.restype = ctypes.c_void_p
    L.ms_nccl_transfer_stream.argtypes = [ctypes.c_void_p]
    L.ms_nccl_transfer_stream.restype = ctypes.c_uint64

    L.ms_tcp_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
    L.ms_tcp_create.restype = ctypes.c_void_p
    L.ms_tcp_bound_port.argtypes = [ctypes.c_void_p]
    L.ms_tcp_bound_port.restype = ctypes.c_int
    L.ms_tcp_set_peer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    L.ms_tcp_set_peer.restype = ctypes.c_int

    L.ms_mem_bus_create.argtypes = [ctypes.c_int]
    L.ms_mem_bus_create.restype = ctypes.c_void_p
    L.ms_mem_bus_destroy.argtypes = [ctypes.c_void_p]
    L.ms_mem_bus_destroy.restype = None
    L.ms_mem_bus_endpoint.argtypes = [ctypes.c_void_p, ctypes.c_int]
    L.ms_mem_bus_endpoint.restype = ctypes.c_void_p
    L.ms_mem_bus_put_shm.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
    L.ms_mem_bus_put_shm.restype = ctypes.c_int
    L.ms_mem_bus_pop_shm_size.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    L.ms_mem_bus_pop_shm_size.restype = ctypes.c_size_t

    L.ms_transport_send.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    L.ms_transport_send.restype = ctypes.c_double
    L.ms_transport_recv.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_double),
    ]
    L.ms_transport_recv.restype = ctypes.c_int
    L.ms_transport_close.argtypes = [ctypes.c_void_p]
    L.ms_transport_close.restype = None
    L.ms_transport_destroy.argtypes = [ctypes.c_void_p]
    L.ms_transport_destroy.restype = None

    # PreemptEdfController (`ms_edf_*`)
    L.ms_edf_create.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
    ]
    L.ms_edf_create.restype = ctypes.c_void_p
    L.ms_edf_destroy.argtypes = [ctypes.c_void_p]
    L.ms_edf_destroy.restype = None
    L.ms_edf_set_state.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
    ]
    L.ms_edf_set_state.restype = None
    L.ms_edf_get_emas.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    L.ms_edf_get_emas.restype = None
    L.ms_edf_adaptive_chunk_layers.argtypes = [ctypes.c_void_p, ctypes.c_double]
    L.ms_edf_adaptive_chunk_layers.restype = ctypes.c_int
    L.ms_edf_record_layer.argtypes = [ctypes.c_void_p, ctypes.c_double]
    L.ms_edf_record_layer.restype = ctypes.c_double
    L.ms_edf_record_ar_batch.argtypes = [ctypes.c_void_p, ctypes.c_double]
    L.ms_edf_record_ar_batch.restype = ctypes.c_double
    L.ms_edf_should_yield.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    L.ms_edf_should_yield.restype = ctypes.c_int
    L.ms_edf_select_microbatch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    L.ms_edf_select_microbatch.restype = ctypes.c_int

    # JSON-returning control/router/budget/sim C APIs (`restype = c_void_p` so `ms_free_cstr` frees heap string!)
    L.ms_policy_from_preset_json.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
    L.ms_policy_from_preset_json.restype = ctypes.c_void_p
    L.ms_topology_a100_default_json.argtypes = []
    L.ms_topology_a100_default_json.restype = ctypes.c_void_p
    L.ms_topology_multinode_json.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_double,
    ]
    L.ms_topology_multinode_json.restype = ctypes.c_void_p
    L.ms_topology_from_job0c_json.argtypes = [ctypes.c_char_p]
    L.ms_topology_from_job0c_json.restype = ctypes.c_void_p
    L.ms_physics_from_phase01_json.argtypes = [ctypes.c_char_p]
    L.ms_physics_from_phase01_json.restype = ctypes.c_void_p

    L.ms_budget_internvlu_plan_json.argtypes = [ctypes.c_char_p]
    L.ms_budget_internvlu_plan_json.restype = ctypes.c_void_p
    L.ms_budget_internvlu_quota_overrides_json.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_double,
    ]
    L.ms_budget_internvlu_quota_overrides_json.restype = ctypes.c_void_p
    L.ms_budget_verify_internvlu_quota_json.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double]
    L.ms_budget_verify_internvlu_quota_json.restype = ctypes.c_void_p

    L.ms_router_create.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p]
    L.ms_router_create.restype = ctypes.c_void_p
    L.ms_router_destroy.argtypes = [ctypes.c_void_p]
    L.ms_router_destroy.restype = None
    L.ms_router_init_workers_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    L.ms_router_init_workers_json.restype = ctypes.c_void_p
    L.ms_router_get_workers_json.argtypes = [ctypes.c_void_p]
    L.ms_router_get_workers_json.restype = ctypes.c_void_p
    L.ms_router_set_worker_fields.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int,
    ]
    L.ms_router_set_worker_fields.restype = None
    L.ms_router_update_telemetry.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int64,
    ]
    L.ms_router_update_telemetry.restype = None
    L.ms_router_select_initial_gpu.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    L.ms_router_select_initial_gpu.restype = ctypes.c_int
    L.ms_router_select_static_gen_gpu.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    L.ms_router_select_static_gen_gpu.restype = ctypes.c_int
    L.ms_router_select_oom_retry_gpu.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    L.ms_router_select_oom_retry_gpu.restype = ctypes.c_int
    L.ms_router_decide_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    L.ms_router_decide_json.restype = ctypes.c_void_p
    L.ms_router_plan_and_reserve_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    L.ms_router_plan_and_reserve_json.restype = ctypes.c_void_p
    L.ms_router_route_batch_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    L.ms_router_route_batch_json.restype = ctypes.c_void_p
    L.ms_router_reserve_kv_bytes.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int64]
    L.ms_router_reserve_kv_bytes.restype = None
    L.ms_router_release_kv_reservation.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int64,
    ]
    L.ms_router_release_kv_reservation.restype = None
    L.ms_router_release_dit_reservation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
    L.ms_router_release_dit_reservation.restype = None
    L.ms_router_crossover_kv_mib.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
    ]
    L.ms_router_crossover_kv_mib.restype = ctypes.c_double

    L.ms_ctrl_deterministic_think_s.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_double, ctypes.c_double]
    L.ms_ctrl_deterministic_think_s.restype = ctypes.c_double
    L.ms_ctrl_aggregate_report_json.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    L.ms_ctrl_aggregate_report_json.restype = ctypes.c_void_p

    L.ms_ledger_create.argtypes = [ctypes.c_char_p]
    L.ms_ledger_create.restype = ctypes.c_void_p
    L.ms_ledger_destroy.argtypes = [ctypes.c_void_p]
    L.ms_ledger_destroy.restype = None
    L.ms_ledger_upsert_kv.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
    L.ms_ledger_upsert_kv.restype = ctypes.c_int
    L.ms_ledger_ensure_resident.argtypes = [ctypes.c_void_p, ctypes.c_int]
    L.ms_ledger_ensure_resident.restype = ctypes.c_double
    L.ms_ledger_evict_idle_kv.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.c_int]
    L.ms_ledger_evict_idle_kv.restype = ctypes.c_double
    L.ms_ledger_exec_dit_vae.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    L.ms_ledger_exec_dit_vae.restype = ctypes.c_int
    L.ms_ledger_release_session.argtypes = [ctypes.c_void_p, ctypes.c_int]
    L.ms_ledger_release_session.restype = None
    L.ms_ledger_set_peak_used_mib.argtypes = [ctypes.c_void_p, ctypes.c_double]
    L.ms_ledger_set_peak_used_mib.restype = None
    L.ms_ledger_stats_json.argtypes = [ctypes.c_void_p]
    L.ms_ledger_stats_json.restype = ctypes.c_void_p

    L.ms_sim_point_json.argtypes = [ctypes.c_char_p]
    L.ms_sim_point_json.restype = ctypes.c_void_p
    L.ms_sim_evaluate_real_json.argtypes = [ctypes.c_char_p]
    L.ms_sim_evaluate_real_json.restype = ctypes.c_void_p
    L.ms_sim_h100_study_json.argtypes = [ctypes.c_char_p]
    L.ms_sim_h100_study_json.restype = ctypes.c_void_p
    L.ms_sim_sweep_json.argtypes = [ctypes.c_char_p]
    L.ms_sim_sweep_json.restype = ctypes.c_void_p

    # Legacy `tripool_*` symbols (`CUDAPluggableAllocator` & direct C-ABI callers)
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
    L.tripool_vmm_get_stats.argtypes = [ctypes.POINTER(UnifiedVmmStatsC)]
    L.tripool_vmm_get_stats.restype = None
    L.tripool_host_register.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    L.tripool_host_register.restype = ctypes.c_int
    L.tripool_host_unregister.argtypes = [ctypes.c_void_p]
    L.tripool_host_unregister.restype = ctypes.c_int
    L.tripool_scope_begin.argtypes = []
    L.tripool_scope_begin.restype = None
    L.tripool_scope_end.argtypes = [ctypes.c_void_p]
    L.tripool_scope_end.restype = ctypes.c_int
    L.tripool_alloc.argtypes = [ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
    L.tripool_alloc.restype = ctypes.c_uint64
    L.tripool_free.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
    L.tripool_free.restype = None


def load_core_lib(
    so_path: Path | str | None = None,
    *,
    prefer_vmm_alias: bool = False,
) -> tuple[ctypes.CDLL, Path]:
    resolved = resolve_core_so(so_path, prefer_vmm_alias=prefer_vmm_alias, auto_build=(so_path is None))
    st = os.stat(resolved)
    key = (int(st.st_dev), int(st.st_ino))
    with _LOAD_LOCK:
        cached = _LIB_CACHE.get(key)
        if cached is not None:
            return cached[0], resolved
        lib = ctypes.CDLL(str(resolved))
        _bind_all_symbols(lib)
        _LIB_CACHE[key] = (lib, resolved)
        return lib, resolved


def get_core_lib() -> ctypes.CDLL:
    return load_core_lib()[0]


def last_error(lib: ctypes.CDLL | None = None) -> str:
    L = lib if lib is not None else get_core_lib()
    raw = L.ms_last_error()
    return raw.decode("utf-8", errors="replace") if raw else "unknown C++ error"


def encode_json(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def call_json(lib: ctypes.CDLL, fn: Any, *args: Any) -> Any:
    ptr = fn(*args)
    if not ptr:
        err = last_error(lib)
        if any(tok in err for tok in ("unknown", "unsupported", "must be", "requires", "invalid")):
            raise ValueError(err)
        raise RuntimeError(err)
    try:
        raw_str = ctypes.string_at(ptr).decode("utf-8")
    finally:
        lib.ms_free_cstr(ctypes.c_void_p(ptr))
    return json.loads(raw_str)


def cuda_memcpy_htod(device: int, dst_va: int, src_ptr: int, nbytes: int) -> None:
    if nbytes <= 0:
        return
    L = get_core_lib()
    rc = int(L.ms_cuda_memcpy_htod(int(device), ctypes.c_uint64(int(dst_va)), ctypes.c_void_p(int(src_ptr)), ctypes.c_size_t(int(nbytes))))
    if rc != 0:
        raise RuntimeError(f"ms_cuda_memcpy_htod failed: {last_error(L)}")


def cuda_memcpy_dtoh(device: int, dst_ptr: int, src_va: int, nbytes: int) -> None:
    if nbytes <= 0:
        return
    L = get_core_lib()
    rc = int(L.ms_cuda_memcpy_dtoh(int(device), ctypes.c_void_p(int(dst_ptr)), ctypes.c_uint64(int(src_va)), ctypes.c_size_t(int(nbytes))))
    if rc != 0:
        raise RuntimeError(f"ms_cuda_memcpy_dtoh failed: {last_error(L)}")
