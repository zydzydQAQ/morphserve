"""Deep `CPTGModelAdapter` interface, shared GPU adapter lifecycle mixin, and `SyntheticCPTGAdapter`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass, field
import gc
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable

from morphserve.vmm import (
    ExpertPartitionSpec,
    LayerExpertPager,
    LayerPackPlan,
    NaiveHostOffloader,
    ParamSlice,
    SimulatedTriPoolBackend,
)
from morphserve.vmm.pager import StaticFissionDropper

try:
    from morphserve.vmm.tripool import VmmOutOfMemoryError
except ImportError:  # pragma: no cover - seam fallback prior to Wave 2 tripool update

    class VmmOutOfMemoryError(MemoryError):  # type: ignore[no-redef]
        """Typed VMM physical pool exhaustion exception."""


MIB = 1024**2
GIB = 1024**3


class SyntheticOutOfMemoryError(MemoryError):
    """Deterministic CPU test exception treated identically to CUDA/VMM OOM."""


def is_oom_exception(exc: BaseException) -> bool:
    """Classify VMM/Python MemoryError, PyTorch CUDA OutOfMemoryError, and allocator OOM messages."""
    if isinstance(exc, (MemoryError, VmmOutOfMemoryError)) or type(exc).__name__ == "OutOfMemoryError":
        return True
    low = str(exc).lower()
    return (
        "out of memory" in low
        or "out of vmm memory" in low
        or "cudaerrormemoryallocation" in low
        or "cublas_status_alloc_failed" in low
        or "cudnn_status_alloc_failed" in low
        or ("torch_call_dispatcher" in low and "aten::new_empty" in low)
    )


def compute_vmm_available_kv_bytes(
    vmm: Any | None,
    *,
    has_pageable_experts: bool = False,
    for_dit: bool = False,
) -> int:
    """Compute real-time free + drop-clean reclaimable KV capacity in bytes for a `TriPoolVMM` backend."""
    if vmm is None:
        return 0
    kv_free_fn = getattr(vmm, "kv_free_bytes", None)
    if callable(kv_free_fn):
        try:
            return max(0, int(kv_free_fn(for_dit=bool(for_dit))))
        except TypeError:
            if not for_dit:
                return max(0, int(kv_free_fn()))
    if not for_dit:
        direct_fn = getattr(vmm, "available_kv_bytes", None)
        if callable(direct_fn):
            return max(0, int(direct_fn()))
    st = vmm.stats_dict()
    if for_dit and "kv_dit_headroom_bytes" in st:
        return max(0, int(st["kv_dit_headroom_bytes"]))
    if not for_dit and "kv_free_bytes" in st:
        return max(0, int(st["kv_free_bytes"]))
    min_gran = int(st.get("min_granularity_bytes", 2 * MIB))
    super_bytes = int(st.get("super_handle_bytes", 64 * MIB))
    fine_free = int(st.get("fine_pool_free_pages", 0)) * min_gran
    super_free = int(st.get("super_pool_free_handles", 0)) * super_bytes
    reclaimable_exp = int(st.get("expert_mapped_handles_current", 0)) * super_bytes if has_pageable_experts else 0
    if for_dit:
        return max(0, super_free + reclaimable_exp)
    return max(0, fine_free + super_free + reclaimable_exp)


@dataclass(frozen=True)
class PackedTransferState:
    """Serialized cross-paradigm state (`NaiveCache` KV or conditioning bridge) ready for wire shipping."""

    meta: dict[str, Any]
    payload: Any  # CUDA uint8 Tensor in GPU mode, bytes in CPU synthetic mode
    logical_bytes: int


class CPTGModelAdapter(ABC):
    """Deep model runtime interface hiding architecture-specific loading, KV/bridge packing, AR, and DiT."""

    architecture: str = "tight_mot"
    num_expert_layers: int = 28

    @staticmethod
    def normalized_enqueue_time_s(cmd: dict[str, Any], default_s: float = 0.0) -> float:
        """Return clock-skew-safe enqueue timestamp preferring worker-local `_worker_recv_s`."""
        recv_s = float(cmd.get("_worker_recv_s", 0.0))
        if recv_s > 0.0:
            return recv_s
        enq_s = float(cmd.get("enqueue_time_s", 0.0))
        return enq_s if enq_s > 0.0 else float(default_s)

    @property
    @abstractmethod
    def total_mirror_bytes(self) -> int:
        """Size in bytes required for the NUMA-socket host expert weight mirror (0 if unused)."""

    @abstractmethod
    def attach_expert_mirror(self, shared_mirror: Any | None, populate: bool = True) -> None:
        """Populate or attach the host `NumaMemfdMirror` and release duplicate GPU expert weights."""

    @abstractmethod
    def bind_runtime_resources(self, pacer: Any | None = None) -> dict[str, Any]:
        """Initialize `TriPoolVMM`, `LayerExpertPager`, execution interceptor, and scheduler; return init telemetry."""

    @abstractmethod
    def warmup(self) -> None:
        """Warm up non-dropped execution paths and reset peak memory watermarks."""

    @abstractmethod
    def prepare_turn(self, ctx_id: str, cmd: dict[str, Any]) -> int:
        """Prepare/append turn inputs into `ctx_id` and return current logical CPTG state bytes."""

    @abstractmethod
    def decode_ar(self, ctx_id: str, max_tokens: int) -> dict[str, Any]:
        """Run greedy AR decoding on `ctx_id` and return `{text, tokens, ttft_ms, tpots_ms, kv_bytes}`."""

    def decode_ar_batch(self, batch_cmds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute 1..B concurrent AR requests (subclasses override with `[B, D]` continuous batching)."""
        results: list[dict[str, Any]] = []
        for cmd in batch_cmds:
            cid = str(cmd["ctx_id"])
            enq_s = self.normalized_enqueue_time_s(cmd)
            t_start = time.perf_counter()
            self.prepare_turn(cid, cmd)
            out = self.decode_ar(cid, int(cmd.get("max_ar_tokens", 32)))
            q_wait_ms = max(0.0, (t_start - enq_s) * 1000.0) if enq_s > 0.0 else 0.0
            out = dict(out)
            out["ttft_ms"] = round(q_wait_ms + float(out.get("ttft_ms", 0.0)), 2)
            out["resident_alpha"] = self.resident_alpha()
            out["kv_free_bytes"] = self.reported_kv_free_bytes()
            results.append(out)
        return results

    @abstractmethod
    def generate_dit(
        self,
        ctx_id: str,
        cmd: dict[str, Any],
        poll_preempt_cmds: Callable[[], list[dict[str, Any]]] | None = None,
        emit_preempt_ack: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run DiT denoising + VAE decode, servicing urgent preemptive AR commands if configured."""

    @abstractmethod
    def pack_state(self, ctx_id: str, include_cfg_dup: bool = False) -> PackedTransferState:
        """Pack `ctx_id` into a contiguous byte buffer + metadata dict for Path-A shipping."""

    @abstractmethod
    def unpack_state(self, new_ctx_id: str, meta: dict[str, Any], raw_payload: Any) -> int:
        """Reconstruct `new_ctx_id` from wire metadata + byte buffer and return logical state bytes."""

    @abstractmethod
    def set_residency_alpha(self, alpha: float) -> float:
        """Update `LayerExpertPager` residency to `alpha` and return resulting `resident_alpha` in `[0, 1]`."""

    @abstractmethod
    def resident_alpha(self) -> float:
        """Return current `expert_alpha` residency fraction in `[0, 1]`."""

    def reset_peak_memory_stats(self) -> None:
        """Reset per-suite CUDA allocator & NVML/driver physical HBM high-water marks before `run_suite`."""
        return None

    def reset_for_new_suite(self) -> float:
        """Reset worker residency to its initial bound alpha and clear peak memory stats for a new suite."""
        pol = getattr(self, "policy", None)
        default_alpha = (
            float(pol.initial_alphas.get(getattr(self, "rank", 0), 1.0))
            if hasattr(pol, "initial_alphas")
            else 1.0
        )
        target = float(getattr(self, "_initial_bound_alpha", default_alpha))
        alpha_now = self.set_residency_alpha(target)
        self.reset_peak_memory_stats()
        return alpha_now

    @abstractmethod
    def available_kv_bytes(self, *, for_dit: bool = False) -> int:
        """Return current free + drop-clean reclaimable GPU KV capacity in bytes on this worker."""

    def reported_kv_free_bytes(self) -> int:
        """Return scheduler/router-facing GPU memory headroom in bytes.

        Defaults to `available_kv_bytes()`; tight-MoT VMM adapters override this to report true
        DiT-usable HBM headroom without inflating free capacity by unpinned Arena-0 expert handles.
        """
        return self.available_kv_bytes()

    def evict_idle_kv_to_host(
        self,
        target_free_bytes: int,
        exclude_ctx_ids: Iterable[str] = (),
        *,
        for_dit: bool = False,
    ) -> int:
        """Swap unpinned idle think-time sessions' KV to Host Pinned DRAM until `available_kv_bytes >= target`."""
        return 0

    def reclaim_driver_headroom(
        self,
        needed_bytes: int,
        exclude_ctx_ids: Iterable[str] = (),
    ) -> int:
        """Reclaim CUDA driver / PyTorch allocator headroom (e.g., before large NCCL staging buffers)."""
        return 0

    @abstractmethod
    def release_context(self, ctx_id: str) -> None:
        """Release all worker state and VMM KV regions associated with `ctx_id`."""

    @abstractmethod
    def memory_and_vmm_stats(self) -> dict[str, Any]:
        """Return `{peak_reserved_mib, vmm_stats, active_contexts, kv_free_bytes, ...}`."""

    @abstractmethod
    def close(self) -> None:
        """Release VMM, pager, hooks, and contexts cleanly."""


class _NvmlMemoryStruct(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _NvmlUtilizationStruct(ctypes.Structure):
    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint),
    ]


class _GpuNvmlPoller:
    """Lightweight background NVML poller tracking true peak HBM (`MiB`) and SM utilization (`%`)."""

    def __init__(self, device: Any, rank: int = 0, interval_s: float = 0.075) -> None:
        self.device = device
        self.rank = int(rank)
        self.interval_s = max(0.025, float(interval_s))
        self._lock = threading.Lock()
        self._stop_ev = threading.Event()
        self._thread: threading.Thread | None = None

        self._backend: str = "none"
        self._pynvml: Any = None
        self._nvml_lib: Any = None
        self._nvml_handle: Any = None

        self._peak_used_mib: float = 0.0
        self._sm_util_sum: float = 0.0
        self._sm_util_samples: int = 0
        self._mem_util_sum: float = 0.0
        self._mem_util_samples: int = 0
        self._hbm_util_sum: float = 0.0
        self._hbm_util_samples: int = 0

        self._init_backend()

    def _resolve_device_selector(self) -> tuple[str, int | str]:
        dev_idx = getattr(self.device, "index", None)
        logical_idx = int(dev_idx) if dev_idx is not None else int(self.rank)
        raw_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if raw_vis:
            entries = [tok.strip() for tok in raw_vis.split(",") if tok.strip()]
            if 0 <= logical_idx < len(entries):
                entry = entries[logical_idx]
                if entry.startswith("GPU-") or entry.startswith("MIG-"):
                    return ("uuid", entry)
                try:
                    return ("index", int(entry))
                except ValueError:
                    pass
        return ("index", max(0, logical_idx))

    def _init_backend(self) -> None:
        kind, selector = self._resolve_device_selector()

        # 1. Preferred: `pynvml` (`nvidia-ml-py`)
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            if kind == "uuid":
                handle = pynvml.nvmlDeviceGetHandleByUUID(str(selector).encode("utf-8"))
            else:
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(selector))
            self._pynvml = pynvml
            self._nvml_handle = handle
            self._backend = "pynvml"
            return
        except Exception:
            pass

        # 2. Zero-dependency Linux driver library via `ctypes`: `libnvidia-ml.so.1`
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            init_fn = getattr(lib, "nvmlInit_v2", getattr(lib, "nvmlInit", None))
            if init_fn is not None and int(init_fn()) == 0:
                handle = ctypes.c_void_p()
                rc = -1
                if kind == "uuid":
                    by_uuid = getattr(lib, "nvmlDeviceGetHandleByUUID", None)
                    if by_uuid is not None:
                        by_uuid.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
                        by_uuid.restype = ctypes.c_int
                        rc = int(by_uuid(str(selector).encode("utf-8"), ctypes.byref(handle)))
                else:
                    by_idx = getattr(
                        lib, "nvmlDeviceGetHandleByIndex_v2", getattr(lib, "nvmlDeviceGetHandleByIndex", None)
                    )
                    if by_idx is not None:
                        by_idx.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
                        by_idx.restype = ctypes.c_int
                        rc = int(by_idx(ctypes.c_uint(int(selector)), ctypes.byref(handle)))
                if rc == 0 and handle.value:
                    mem_fn = lib.nvmlDeviceGetMemoryInfo
                    mem_fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlMemoryStruct)]
                    mem_fn.restype = ctypes.c_int
                    util_fn = getattr(lib, "nvmlDeviceGetUtilizationRates", None)
                    if util_fn is not None:
                        util_fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlUtilizationStruct)]
                        util_fn.restype = ctypes.c_int
                    self._nvml_lib = lib
                    self._nvml_handle = handle
                    self._backend = "ctypes_nvml"
                    return
        except Exception:
            pass

        # 3. Fallback: CUDA driver `mem_get_info` (memory only; avoids CUDA context contention if NVML exists)
        self._backend = "cuda_mem_info"

    def _query_raw(self) -> tuple[float, float | None, float | None, float | None]:
        if self._backend == "pynvml" and self._pynvml is not None and self._nvml_handle is not None:
            used_mib = 0.0
            hbm_pct: float | None = None
            try:
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                used_mib = float(mem.used) / MIB
                if float(mem.total) > 0.0:
                    hbm_pct = 100.0 * float(mem.used) / float(mem.total)
            except Exception:
                used_mib = 0.0
            sm_pct: float | None = None
            mem_pct: float | None = None
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                sm_pct = float(util.gpu)
                mem_pct = float(util.memory)
            except Exception:
                sm_pct = None
                mem_pct = None
            return used_mib, sm_pct, mem_pct, hbm_pct

        if self._backend == "ctypes_nvml" and self._nvml_lib is not None and self._nvml_handle is not None:
            used_mib = 0.0
            hbm_pct = None
            sm_pct = None
            mem_pct = None
            try:
                mem_s = _NvmlMemoryStruct()
                if int(self._nvml_lib.nvmlDeviceGetMemoryInfo(self._nvml_handle, ctypes.byref(mem_s))) == 0:
                    used_mib = float(mem_s.used) / MIB
                    if float(mem_s.total) > 0.0:
                        hbm_pct = 100.0 * float(mem_s.used) / float(mem_s.total)
            except Exception:
                used_mib = 0.0
            try:
                util_fn = getattr(self._nvml_lib, "nvmlDeviceGetUtilizationRates", None)
                if util_fn is not None:
                    util_s = _NvmlUtilizationStruct()
                    if int(util_fn(self._nvml_handle, ctypes.byref(util_s))) == 0:
                        sm_pct = float(util_s.gpu)
                        mem_pct = float(util_s.memory)
            except Exception:
                sm_pct = None
                mem_pct = None
            return used_mib, sm_pct, mem_pct, hbm_pct

        try:
            import torch

            if torch.cuda.is_available() and getattr(self.device, "type", "cuda") == "cuda":
                free_b, total_b = torch.cuda.mem_get_info(self.device)
                if int(total_b) > 0:
                    used_b = float(int(total_b) - int(free_b))
                    return used_b / MIB, None, None, 100.0 * used_b / float(int(total_b))
        except Exception:
            pass
        return 0.0, None, None, None

    def sample_once(self) -> float:
        used_mib, sm_pct, mem_pct, hbm_pct = self._query_raw()
        with self._lock:
            if used_mib > self._peak_used_mib:
                self._peak_used_mib = used_mib
            if sm_pct is not None:
                self._sm_util_sum += sm_pct
                self._sm_util_samples += 1
            if mem_pct is not None:
                self._mem_util_sum += mem_pct
                self._mem_util_samples += 1
            if hbm_pct is not None:
                self._hbm_util_sum += hbm_pct
                self._hbm_util_samples += 1
        return used_mib

    def reset(self) -> float:
        used_mib, _, _, _ = self._query_raw()
        with self._lock:
            self._peak_used_mib = used_mib
            self._sm_util_sum = 0.0
            self._sm_util_samples = 0
            self._mem_util_sum = 0.0
            self._mem_util_samples = 0
            self._hbm_util_sum = 0.0
            self._hbm_util_samples = 0
        return used_mib

    def snapshot(self) -> tuple[float, float, int]:
        with self._lock:
            mean_sm = (self._sm_util_sum / float(self._sm_util_samples)) if self._sm_util_samples > 0 else 0.0
            return float(self._peak_used_mib), float(mean_sm), int(self._sm_util_samples)

    def snapshot_extended(self) -> dict[str, float | int]:
        with self._lock:
            mean_sm = (self._sm_util_sum / float(self._sm_util_samples)) if self._sm_util_samples > 0 else 0.0
            mean_mem = (self._mem_util_sum / float(self._mem_util_samples)) if self._mem_util_samples > 0 else 0.0
            mean_hbm = (self._hbm_util_sum / float(self._hbm_util_samples)) if self._hbm_util_samples > 0 else 0.0
            return {
                "peak_used_mib": float(self._peak_used_mib),
                "sm_util_mean_pct": float(mean_sm),
                "sm_util_samples": int(self._sm_util_samples),
                "mem_util_mean_pct": float(mean_mem),
                "mem_util_samples": int(self._mem_util_samples),
                "hbm_util_mean_pct": float(mean_hbm),
                "hbm_util_samples": int(self._hbm_util_samples),
            }

    def _run_loop(self) -> None:
        while not self._stop_ev.wait(self.interval_s):
            self.sample_once()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_ev.clear()
        self.sample_once()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"ms-nvml-poller-r{self.rank}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_ev.set()
        thr = self._thread
        self._thread = None
        if thr is not None and thr.is_alive():
            thr.join(timeout=1.0)


class GpuCPTGAdapterMixin(CPTGModelAdapter):
    """Shared GPU lifecycle, HWM/VMM accounting, and 3-way expert partition binding for production adapters."""

    rank: int
    policy: Any
    device: Any
    numa_node: int
    pager: LayerExpertPager | None
    naive_offloader: NaiveHostOffloader | None
    fission_dropper: StaticFissionDropper | None
    vmm: Any
    pacer: Any
    _ctx_store: dict[str, Any]
    _vmm_initial_pool_mib: float
    _max_reserved_mib: float
    _max_driver_used_mib: float
    _max_pytorch_reserved_mib: float
    _nvml_poller: _GpuNvmlPoller | None
    _nvml_sm_util_mean_pct: float
    _nvml_sm_util_samples: int
    _nvml_mem_util_mean_pct: float
    _nvml_mem_util_samples: int
    _effective_hbm_util_mean_pct: float
    _effective_hbm_util_samples: int

    def _init_gpu_lifecycle_state(self) -> None:
        self.pager = None
        self.naive_offloader = None
        self.fission_dropper = None
        self.vmm = None
        self.pacer = None
        self._ctx_store = {}
        self._vmm_initial_pool_mib = 0.0
        self._max_reserved_mib = 0.0
        self._max_driver_used_mib = 0.0
        self._max_pytorch_reserved_mib = 0.0
        self._nvml_poller = None
        self._nvml_sm_util_mean_pct = 0.0
        self._nvml_sm_util_samples = 0
        self._nvml_mem_util_mean_pct = 0.0
        self._nvml_mem_util_samples = 0
        self._effective_hbm_util_mean_pct = 0.0
        self._effective_hbm_util_samples = 0

    def _ensure_nvml_poller(self) -> _GpuNvmlPoller | None:
        import torch

        if not torch.cuda.is_available() or getattr(self.device, "type", "cuda") != "cuda":
            return None
        if self._nvml_poller is None:
            self._nvml_poller = _GpuNvmlPoller(self.device, rank=self.rank, interval_s=0.075)
        self._nvml_poller.start()
        return self._nvml_poller

    def _sync_poller_snapshot(self, poller: _GpuNvmlPoller) -> float:
        snap = poller.snapshot_extended()
        pk = float(snap["peak_used_mib"])
        if int(snap["sm_util_samples"]) > 0:
            self._nvml_sm_util_mean_pct = float(snap["sm_util_mean_pct"])
            self._nvml_sm_util_samples = int(snap["sm_util_samples"])
        if int(snap["mem_util_samples"]) > 0:
            self._nvml_mem_util_mean_pct = float(snap["mem_util_mean_pct"])
            self._nvml_mem_util_samples = int(snap["mem_util_samples"])
        if int(snap["hbm_util_samples"]) > 0:
            self._effective_hbm_util_mean_pct = float(snap["hbm_util_mean_pct"])
            self._effective_hbm_util_samples = int(snap["hbm_util_samples"])
        return pk

    def _stop_nvml_poller(self) -> None:
        poller = getattr(self, "_nvml_poller", None)
        if poller is None:
            return
        try:
            pk = self._sync_poller_snapshot(poller)
            self._max_driver_used_mib = max(self._max_driver_used_mib, pk)
            self._max_reserved_mib = max(self._max_reserved_mib, pk)
        finally:
            poller.stop()
            self._nvml_poller = None

    def bind_expert_partition(
        self,
        spec: ExpertPartitionSpec,
        *,
        drop_for_fission: bool = False,
        vae_output_cloner: Callable[[Any], Any] | None = None,
    ) -> LayerExpertPager | None:
        """Bind a secondary expert via 3-way policy dispatch (`StaticFissionDropper` / `NaiveHostOffloader` / `LayerExpertPager`)."""
        if self.policy.routing_mode in ("static_nccl_fission", "twostage_shm"):
            if drop_for_fission:
                self.fission_dropper = StaticFissionDropper(spec, device=self.device)
                self._update_hwm()
                self.fission_dropper.drop()
            return None
        if self.policy.use_naive_offload:
            self.naive_offloader = NaiveHostOffloader(spec, device=self.device)
            self._update_hwm()
            self.naive_offloader.offload_to_cpu()
            return None
        if self.policy.use_expert_pager:
            self.pager = LayerExpertPager(
                spec,
                numa_node=self.numa_node,
                vae_output_cloner=vae_output_cloner,
            )
            return self.pager
        return None

    def _vmm_physical_pool_mib(self) -> float:
        if self.vmm is None:
            return 0.0
        st = self.vmm.stats_dict()
        if "pool_total_bytes" in st:
            return float(st["pool_total_bytes"]) / MIB
        if "physical_pool_bytes" in st:
            return float(st["physical_pool_bytes"]) / MIB
        overflow_b = int(st.get("overflow_creates", 0)) * int(
            st.get("super_handle_bytes", int(self.policy.super_handle_mib) * MIB)
        )
        return self._vmm_initial_pool_mib + float(overflow_b) / MIB

    def reset_peak_memory_stats(self) -> None:
        import torch

        gc.collect()
        if not torch.cuda.is_available() or getattr(self.device, "type", "cuda") != "cuda":
            self._max_pytorch_reserved_mib = 0.0
            self._max_driver_used_mib = self._vmm_physical_pool_mib()
            self._max_reserved_mib = self._max_driver_used_mib
            self._nvml_sm_util_mean_pct = 0.0
            self._nvml_sm_util_samples = 0
            self._nvml_mem_util_mean_pct = 0.0
            self._nvml_mem_util_samples = 0
            self._effective_hbm_util_mean_pct = 0.0
            self._effective_hbm_util_samples = 0
            return
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        poller = self._ensure_nvml_poller()
        poller_base = poller.reset() if poller is not None else 0.0
        pt_mib = float(torch.cuda.memory_reserved(self.device) / MIB)
        free_drv, total_drv = torch.cuda.mem_get_info(self.device)
        drv_used_mib = (
            float((int(total_drv) - int(free_drv)) / MIB)
            if int(total_drv) > 0
            else (pt_mib + self._vmm_physical_pool_mib())
        )
        base_phys = max(drv_used_mib, poller_base)
        self._max_pytorch_reserved_mib = pt_mib
        self._max_driver_used_mib = base_phys
        self._max_reserved_mib = base_phys
        self._nvml_sm_util_mean_pct = 0.0
        self._nvml_sm_util_samples = 0
        self._nvml_mem_util_mean_pct = 0.0
        self._nvml_mem_util_samples = 0
        self._effective_hbm_util_mean_pct = 0.0
        self._effective_hbm_util_samples = 0

    def _update_hwm(self) -> None:
        import torch

        if not torch.cuda.is_available() or getattr(self.device, "type", "cuda") != "cuda":
            return
        cur = float(torch.cuda.memory_reserved(self.device) / MIB)
        pk = float(torch.cuda.max_memory_reserved(self.device) / MIB)
        pt_peak = max(cur, pk)
        self._max_pytorch_reserved_mib = max(self._max_pytorch_reserved_mib, pt_peak)
        poller_peak = 0.0
        poller = getattr(self, "_nvml_poller", None)
        if poller is not None:
            sample_used = poller.sample_once()
            p_pk = self._sync_poller_snapshot(poller)
            poller_peak = max(p_pk, sample_used)
        # Sample instantaneous driver/NVML physical HBM (`cuMemGetInfo_v2` + background NVML poller)
        # so PyTorch peak at t1 and dynamic VMM pool size at t2 are never double-counted across timestamps,
        # while transient allocator spikes (`pt_peak`) are never lost.
        free_drv, total_drv = torch.cuda.mem_get_info(self.device)
        vmm_pool_mib = self._vmm_physical_pool_mib()
        if int(total_drv) > 0:
            drv_used_mib = float((int(total_drv) - int(free_drv)) / MIB)
            tot_mib = float(int(total_drv) / MIB)
            nvml_now = min(tot_mib, max(drv_used_mib, poller_peak))
            inst_comb_mib = min(tot_mib, cur + vmm_pool_mib)
            phys_now = max(nvml_now, inst_comb_mib, min(tot_mib, pt_peak))
            if self.vmm is None or not bool(getattr(self.policy, "allow_vmm_overflow", True)):
                phys_now = max(phys_now, min(tot_mib, pt_peak + vmm_pool_mib))
            self._max_driver_used_mib = max(self._max_driver_used_mib, nvml_now)
            self._max_reserved_mib = max(self._max_reserved_mib, phys_now)
        else:
            self._max_reserved_mib = max(self._max_reserved_mib, pt_peak + vmm_pool_mib)

    def set_residency_alpha(self, alpha: float) -> float:
        if self.pager is not None and self.vmm is not None:
            self.pager.set_residency_alpha(float(alpha))
        return self.resident_alpha()

    def resident_alpha(self) -> float:
        if self.pager is not None and self.vmm is not None:
            return round(self.pager.resident_alpha(), 3)
        return round(float(self.policy.initial_alphas.get(self.rank, 1.0)), 3)

    def memory_and_vmm_stats(self) -> dict[str, Any]:
        self._update_hwm()
        vmm_st = self.vmm.stats_dict() if self.vmm is not None else {}
        nvml_peak = self._max_driver_used_mib if self._max_driver_used_mib > 0.0 else self._max_reserved_mib
        return {
            "peak_reserved_mib": round(self._max_pytorch_reserved_mib, 2),
            "combined_physical_mib": round(self._max_reserved_mib, 2),
            "nvml_peak_used_mib": round(nvml_peak, 2),
            "peak_nvml_used_mib": round(nvml_peak, 2),
            "nvml_sm_util_mean_pct": round(self._nvml_sm_util_mean_pct, 2),
            "nvml_sm_util_samples": int(self._nvml_sm_util_samples),
            "nvml_mem_util_mean_pct": round(self._nvml_mem_util_mean_pct, 2),
            "nvml_mem_util_samples": int(self._nvml_mem_util_samples),
            "effective_hbm_util_mean_pct": round(self._effective_hbm_util_mean_pct, 2),
            "effective_hbm_util_samples": int(self._effective_hbm_util_samples),
            "pytorch_reserved_mib": round(self._max_pytorch_reserved_mib, 2),
            "vmm_physical_pool_mib": round(self._vmm_physical_pool_mib(), 2),
            "vmm_stats": vmm_st,
            "active_contexts": len(self._ctx_store),
            "kv_free_bytes": self.reported_kv_free_bytes(),
            "swapped_kv_contexts": 0,
            "kv_host_swap_outs": int(vmm_st.get("kv_swap_outs_total", 0)),
            "kv_host_swap_ins": int(vmm_st.get("kv_swap_ins_total", 0)),
            "kv_host_pinned_bytes": int(vmm_st.get("kv_swapped_bytes_current", 0)),
        }


class _FallbackHostMirror:
    """In-memory host mirror fallback for synthetic tests on Python builds lacking `os.memfd_create`."""

    def __init__(self, nbytes: int, numa_node: int = 0) -> None:
        self.nbytes = int(nbytes)
        self.numa_node = int(numa_node)
        self.numa_bound = False
        self.inode = 0
        self.mm = bytearray(self.nbytes)
        self._c_buf = (ctypes.c_uint8 * self.nbytes).from_buffer(self.mm)
        self.ptr = int(ctypes.addressof(self._c_buf))
        self.pinned = False

    def write_bytes(self, offset: int, data: bytes | bytearray | memoryview) -> None:
        self.mm[offset : offset + len(data)] = data

    def read_bytes(self, offset: int, nbytes: int) -> bytes:
        return bytes(self.mm[offset : offset + nbytes])

    def pin_cuda(self, vmm: Any = None) -> None:
        self.pinned = True

    def close(self) -> None:
        self.pinned = False
        self._c_buf = None
        self.mm = bytearray(0)


class _SyntheticVae:
    def __init__(self, vmm_getter: Callable[[], SimulatedTriPoolBackend | None]) -> None:
        self._vmm_getter = vmm_getter

    def decode(self, spike_mib: int) -> None:
        vmm = self._vmm_getter()
        if vmm is not None:
            vmm.allocate_transient_mib(spike_mib)

    def encode(self, spike_mib: int) -> None:
        vmm = self._vmm_getter()
        if vmm is not None:
            vmm.allocate_transient_mib(spike_mib)


@dataclass
class _SyntheticSessionEntry:
    turns_prepared: int = 0
    logical_bytes: int = 0
    prompt_history: list[str] = field(default_factory=list)
    last_image_tag: str | None = None
    kv_region_id: int | None = None
    swapped_to_host: bool = False
    swapped_bytes: int = 0
    last_used_s: float = field(default_factory=time.perf_counter)


class SyntheticCPTGAdapter(CPTGModelAdapter):
    """Pure-Python deterministic `CPTGModelAdapter` exercising real `SimulatedTriPoolBackend` + `LayerExpertPager`."""

    def __init__(
        self,
        rank: int,
        policy: Any,
        *,
        architecture: str | None = None,
        dit_step_sleep_s: float = 0.0,
        oom_prompts: tuple[str, ...] = (),
        oom_ranks: tuple[int, ...] = (),
        oom_on_mid_dit_preempt: bool = False,
        max_kv_region_mib: int = 64,
    ) -> None:
        self.rank = int(rank)
        self.policy = policy
        self.architecture = architecture or policy.cptg_architecture
        self.num_expert_layers = 20 if self.architecture == "bridge_mllm_dit" else 28
        self.dit_step_sleep_s = float(dit_step_sleep_s)
        self.oom_prompts = tuple(oom_prompts)
        self.oom_ranks = tuple(int(r) for r in oom_ranks)
        self.oom_on_mid_dit_preempt = bool(oom_on_mid_dit_preempt)
        self._max_kv_region_bytes = max(2 * MIB, int(max_kv_region_mib) * MIB)

        self._vmm: SimulatedTriPoolBackend | None = None
        self._vae = _SyntheticVae(lambda: self._vmm)
        self._pager: LayerExpertPager | None = None
        self._alpha = float(policy.initial_alphas.get(self.rank, 1.0))
        self._contexts: dict[str, _SyntheticSessionEntry] = {}
        self._active_pins: dict[str, int] = {}
        self._kv_host_swap_outs: int = 0
        self._kv_host_swap_ins: int = 0
        self._kv_host_pinned_bytes: int = 0
        self._peak_mib = 15_500.0 + 12_200.0 * self._alpha
        self._rank_oom_triggered = False

        if policy.use_expert_pager:
            stride_mib = 256 if self.architecture == "bridge_mllm_dit" else 448
            plans = [
                LayerPackPlan(
                    layer_idx=lid,
                    used_bytes=stride_mib * MIB,
                    params=[ParamSlice(name="w", offset=0, nbytes=4096, shape=(2048,), dtype_str="bf16")],
                )
                for lid in range(self.num_expert_layers)
            ]
            spec = ExpertPartitionSpec(
                name=f"syn_{self.architecture}",
                plans=plans,
                layer_stride_bytes=stride_mib * MIB,
                hook_modules=[],
                vae_module=self._vae if policy.use_scoped_vae else None,
                install_module_hooks=False,
                vae_encode_restore="none",
            )
            self._pager = LayerExpertPager(spec, numa_node=0)

    @property
    def total_mirror_bytes(self) -> int:
        return 4 * MIB if self.policy.use_expert_pager else 0

    def attach_expert_mirror(self, shared_mirror: Any | None, populate: bool = True) -> None:
        if self._pager is None:
            return
        from morphserve.vmm import NumaMemfdMirror

        if shared_mirror is not None:
            self._pager.attach_mirror(shared_mirror, owns_mirror=False)
        elif hasattr(os, "memfd_create"):
            mirror = NumaMemfdMirror.create(nbytes=4 * MIB, numa_node=0, name=f"syn_m_{self.rank}")
            self._pager.attach_mirror(mirror, owns_mirror=True)
        else:
            mirror = _FallbackHostMirror(nbytes=4 * MIB, numa_node=0)
            self._pager.attach_mirror(mirror, owns_mirror=True)

    def bind_runtime_resources(self, pacer: Any | None = None) -> dict[str, Any]:
        if self.policy.use_tripool_vmm:
            self._vmm = SimulatedTriPoolBackend(
                device=self.rank,
                super_pool_mib=self.policy.super_pool_mib,
                fine_kv_pool_mib=self.policy.fine_kv_pool_mib,
                super_handle_mib=self.policy.super_handle_mib,
                allow_overflow=self.policy.allow_vmm_overflow,
            )
            if self._pager is not None:
                for plan in self._pager.plans:
                    plan.used_bytes = 4096
                self._pager.bind_vmm(self._vmm, initial_alpha=self._alpha)
        return {
            "resident_alpha": self.resident_alpha(),
            "reserved_mib": round(self._peak_mib, 2),
            "kv_free_bytes": self.available_kv_bytes(),
        }

    def warmup(self) -> None:
        return None

    @contextmanager
    def _pinned_ctx(self, *ctx_ids: str):
        for cid in ctx_ids:
            self._active_pins[cid] = self._active_pins.get(cid, 0) + 1
        try:
            yield
        finally:
            now_s = time.perf_counter()
            for cid in ctx_ids:
                rem = self._active_pins.get(cid, 1) - 1
                if rem <= 0:
                    self._active_pins.pop(cid, None)
                else:
                    self._active_pins[cid] = rem
                ent = self._contexts.get(cid)
                if ent is not None:
                    ent.last_used_s = now_s

    def available_kv_bytes(
        self,
        *,
        for_dit: bool = False,
        include_reclaimable_experts: bool = True,
    ) -> int:
        if self._vmm is not None:
            return compute_vmm_available_kv_bytes(
                self._vmm,
                has_pageable_experts=(self._pager is not None and include_reclaimable_experts),
                for_dit=for_dit,
            )
        budget = 22 * GIB if self.policy.role_of(self.rank) == "und" else 12 * GIB
        used = sum(e.logical_bytes for e in self._contexts.values() if not e.swapped_to_host)
        return max(0, budget - used)

    def evict_idle_kv_to_host(
        self,
        target_free_bytes: int,
        exclude_ctx_ids: Iterable[str] = (),
        *,
        for_dit: bool = False,
        include_reclaimable_experts: bool = True,
    ) -> int:
        """Swap LRU unpinned idle sessions' KV regions to simulated Host Pinned DRAM."""
        if self._vmm is None or self.architecture == "bridge_mllm_dit":
            return 0
        excluded = set(exclude_ctx_ids) | {cid for cid, cnt in self._active_pins.items() if cnt > 0}
        candidates = [
            (ent.last_used_s, cid, ent)
            for cid, ent in self._contexts.items()
            if cid not in excluded and not ent.swapped_to_host and ent.kv_region_id is not None
        ]
        candidates.sort(key=lambda item: item[0])
        swap_out_fn = getattr(self._vmm, "kv_region_swap_out", None)
        evicted_bytes = 0
        for _, _, ent in candidates:
            cur_avail = self.available_kv_bytes(
                for_dit=for_dit,
                include_reclaimable_experts=include_reclaimable_experts,
            )
            if cur_avail >= int(target_free_bytes):
                break
            rid = ent.kv_region_id
            assert rid is not None
            mapped = self._vmm.kv_region_mapped_bytes(rid)
            valid_b = max(mapped, min(self._max_kv_region_bytes, max(2 * MIB, ent.logical_bytes)))
            if callable(swap_out_fn):
                swap_out_fn(rid, valid_b)
            else:
                self._vmm.kv_region_free(rid)
                ent.kv_region_id = None
            ent.swapped_to_host = True
            ent.swapped_bytes = valid_b
            self._kv_host_swap_outs += 1
            self._kv_host_pinned_bytes += ent.swapped_bytes
            evicted_bytes += ent.swapped_bytes
        return evicted_bytes

    def _ensure_kv_region(self, ctx_id: str, entry: _SyntheticSessionEntry) -> None:
        """Back synthetic session KV with real `kv_region_*` contiguous-VA pages, swapping in/out on watermark."""
        if self._vmm is None or self.architecture == "bridge_mllm_dit":
            return
        needed = max(2 * MIB, min(self._max_kv_region_bytes, entry.logical_bytes))
        was_swapped = entry.swapped_to_host
        cur_mapped = (
            self._vmm.kv_region_mapped_bytes(entry.kv_region_id)
            if (entry.kv_region_id is not None and not was_swapped)
            else 0
        )
        delta = max(0, needed - cur_mapped)
        if delta > 0:
            if self.available_kv_bytes() < delta + 4 * MIB:
                self.evict_idle_kv_to_host(delta + 8 * MIB, exclude_ctx_ids=(ctx_id,))
            elif (
                not self.policy.allow_vmm_overflow
                and self._pager is not None
                and "kv_dit_headroom_bytes" not in self._vmm.stats_dict()
                and self.available_kv_bytes(include_reclaimable_experts=False) < delta
            ):
                self.evict_idle_kv_to_host(
                    delta,
                    exclude_ctx_ids=(ctx_id,),
                    include_reclaimable_experts=False,
                )

        if entry.kv_region_id is None:
            entry.kv_region_id = self._vmm.kv_region_alloc(self._max_kv_region_bytes)
        elif was_swapped:
            swap_in_fn = getattr(self._vmm, "kv_region_swap_in", None)
            if callable(swap_in_fn):
                try:
                    swap_in_fn(entry.kv_region_id)
                except (MemoryError, RuntimeError):
                    freed = self.evict_idle_kv_to_host(1 << 50, exclude_ctx_ids=(ctx_id,))
                    if freed <= 0:
                        raise
                    swap_in_fn(entry.kv_region_id)

        try:
            self._vmm.kv_region_ensure(entry.kv_region_id, needed)
        except (MemoryError, RuntimeError):
            freed = self.evict_idle_kv_to_host(1 << 50, exclude_ctx_ids=(ctx_id,))
            if freed <= 0:
                raise
            self._vmm.kv_region_ensure(entry.kv_region_id, needed)

        if was_swapped:
            self._kv_host_pinned_bytes = max(0, self._kv_host_pinned_bytes - entry.swapped_bytes)
            entry.swapped_to_host = False
            entry.swapped_bytes = 0
            self._kv_host_swap_ins += 1

    def _maybe_raise_synthetic_oom(self, prompt: str) -> None:
        if self.rank in self.oom_ranks and not self._rank_oom_triggered:
            self._rank_oom_triggered = True
            raise SyntheticOutOfMemoryError(f"synthetic rank-{self.rank} transient OOM")
        if any(tag in prompt for tag in self.oom_prompts):
            raise SyntheticOutOfMemoryError(f"synthetic OOM on prompt={prompt!r}")

    def prepare_turn(self, ctx_id: str, cmd: dict[str, Any]) -> int:
        prompt = str(cmd.get("prompt", ""))
        self._maybe_raise_synthetic_oom(prompt)

        with self._pinned_ctx(ctx_id):
            entry = self._contexts.setdefault(ctx_id, _SyntheticSessionEntry())
            entry.turns_prepared += 1
            entry.prompt_history.append(prompt)
            prep_dit = bool(cmd.get("prepare_for_dit", False))

            if self.architecture == "bridge_mllm_dit":
                entry.logical_bytes = 24 * MIB if prep_dit else 2 * MIB
            else:
                if entry.turns_prepared == 1 and not cmd.get("image_paths"):
                    delta = 4 * MIB
                else:
                    base_img_kv = 620 * MIB
                    delta = base_img_kv if self.policy.use_module3_cow_loopback else int(base_img_kv * 1.95)
                entry.logical_bytes += delta

            self._ensure_kv_region(ctx_id, entry)
            self._peak_mib = max(self._peak_mib, 15_500.0 + (entry.logical_bytes / MIB))
            return entry.logical_bytes

    def decode_ar(self, ctx_id: str, max_tokens: int) -> dict[str, Any]:
        with self._pinned_ctx(ctx_id):
            entry = self._contexts[ctx_id]
            n_tok = max(1, min(int(max_tokens), 8))
            entry.logical_bytes += n_tok * 4096
            self._ensure_kv_region(ctx_id, entry)
            prompt_tail = entry.prompt_history[-1] if entry.prompt_history else ""
            return {
                "text": f"syn_ar({prompt_tail[:24]}):{n_tok}",
                "tokens": n_tok,
                "ttft_ms": 12.0,
                "tpots_ms": [18.0 for _ in range(max(0, n_tok - 1))],
                "kv_bytes": entry.logical_bytes,
                "resident_alpha": self.resident_alpha(),
                "kv_free_bytes": self.available_kv_bytes(),
            }

    def decode_ar_batch(self, batch_cmds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Token-interleaved continuous batching across `B >= 1` concurrent synthetic AR requests."""
        if not batch_cmds:
            return []
        t_batch0 = time.perf_counter()
        cids = [str(cmd["ctx_id"]) for cmd in batch_cmds]
        results: list[dict[str, Any]] = []
        with self._pinned_ctx(*cids):
            for idx, cmd in enumerate(batch_cmds):
                cid = str(cmd["ctx_id"])
                enq_s = self.normalized_enqueue_time_s(cmd)
                if not cmd.get("_skip_prepare", False):
                    self.prepare_turn(cid, cmd)
                out = self.decode_ar(cid, int(cmd.get("max_ar_tokens", 8)))
                q_wait_ms = max(0.0, (t_batch0 - enq_s) * 1000.0) if enq_s > 0.0 else 0.0
                out["ttft_ms"] = round(q_wait_ms + float(out["ttft_ms"]) + idx * 1.5, 2)
                out["batched_width"] = len(batch_cmds)
                out["kv_free_bytes"] = self.available_kv_bytes()
                results.append(out)
        return results

    def generate_dit(
        self,
        ctx_id: str,
        cmd: dict[str, Any],
        poll_preempt_cmds: Callable[[], list[dict[str, Any]]] | None = None,
        emit_preempt_ack: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        prompt = str(cmd.get("prompt", ""))
        self._maybe_raise_synthetic_oom(prompt)

        with self._pinned_ctx(ctx_id):
            entry = self._contexts[ctx_id]
            self._ensure_kv_region(ctx_id, entry)
            if self._pager is not None and self._vmm is not None:
                needed_exp_bytes = self._pager.num_layers * self._pager.layer_stride_bytes
                if self.available_kv_bytes(for_dit=False) < needed_exp_bytes:
                    self.evict_idle_kv_to_host(needed_exp_bytes, exclude_ctx_ids=(ctx_id,), for_dit=False)

            steps = max(1, min(int(cmd.get("dit_steps", 4)), 6))
            height = int(cmd.get("height", 512))
            yield_count = 0
            t0 = time.perf_counter()

            def _run_denoising() -> None:
                nonlocal yield_count
                for _ in range(steps):
                    if self.dit_step_sleep_s > 0.0:
                        time.sleep(self.dit_step_sleep_s)
                    if (
                        poll_preempt_cmds is not None
                        and emit_preempt_ack is not None
                        and self.policy.preemption_mode != "non_preemptive"
                    ):
                        urgent = poll_preempt_cmds()
                        if urgent:
                            for ar_msg in urgent:
                                if self.oom_on_mid_dit_preempt or any(
                                    tag in str(ar_msg.get("prompt", "")) for tag in self.oom_prompts
                                ):
                                    raise SyntheticOutOfMemoryError("synthetic mid-DiT AR preemption OOM")
                            ar_outs = self.decode_ar_batch(urgent)
                            yield_count += 1
                            for ar_msg, ar_out in zip(urgent, ar_outs):
                                emit_preempt_ack(
                                    {
                                        "cmd_id": ar_msg["cmd_id"],
                                        "rank": self.rank,
                                        "status": "ok",
                                        "ctx_id": str(ar_msg["ctx_id"]),
                                        "preempted_mid_dit": True,
                                        **ar_out,
                                    }
                                )

            if self._pager is not None and self._vmm is not None:
                with self._pager.generation_mode(prefetch_all=True):
                    _run_denoising()
                spike_mib = (
                    (1_800 if height >= 1024 else 512)
                    if self.architecture == "bridge_mllm_dit"
                    else (5_600 if height >= 1024 else 1_600)
                )
                if self.policy.use_scoped_vae:
                    self._vae.decode(spike_mib)
            else:
                _run_denoising()

            save_tmp = cmd.get("save_tmp_image_path")
            img_tag = f"img_{ctx_id}_{cmd.get('seed', 52)}"
            entry.last_image_tag = img_tag
            if save_tmp:
                p = Path(str(save_tmp))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(img_tag.encode("utf-8"))

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._peak_mib = max(self._peak_mib, 24_000.0 if height >= 1024 else 19_000.0)
            return {
                "status": "ok",
                "ctx_id": ctx_id,
                "elapsed_ms": round(elapsed_ms, 2),
                "h2d_weight_ms": 0.0,
                "vae_ms": 0.0,
                "preempt_yield_ms": 0.0,
                "kv_bytes": entry.logical_bytes,
                "resident_alpha": self.resident_alpha(),
                "kv_free_bytes": self.available_kv_bytes(),
                "yield_count": yield_count,
            }

    def pack_state(self, ctx_id: str, include_cfg_dup: bool = False) -> PackedTransferState:
        with self._pinned_ctx(ctx_id):
            entry = self._contexts[ctx_id]
            self._ensure_kv_region(ctx_id, entry)
            mult = 2 if (include_cfg_dup and self.architecture == "tight_mot" and entry.turns_prepared > 1) else 1
            wire_logical_b = int(entry.logical_bytes * mult)
            wire_obj = {
                "turns_prepared": entry.turns_prepared,
                "base_logical_bytes": int(entry.logical_bytes),
                "logical_bytes": wire_logical_b,
                "prompt_history": list(entry.prompt_history),
                "last_image_tag": entry.last_image_tag,
            }
            raw = json.dumps(wire_obj, separators=(",", ":")).encode("utf-8")
            meta = {
                "total_bytes": len(raw),
                "logical_bytes": wire_logical_b,
                "base_logical_bytes": int(entry.logical_bytes),
                "architecture": self.architecture,
            }
            return PackedTransferState(meta=meta, payload=raw, logical_bytes=wire_logical_b)

    def unpack_state(self, new_ctx_id: str, meta: dict[str, Any], raw_payload: Any) -> int:
        raw_bytes = bytes(raw_payload)
        wire_obj = json.loads(raw_bytes.decode("utf-8")) if raw_bytes else {}
        wire_logical_b = int(meta.get("logical_bytes", wire_obj.get("logical_bytes", 0)))
        base_logical_b = int(
            meta.get("base_logical_bytes", wire_obj.get("base_logical_bytes", wire_logical_b))
        )
        self.release_context(new_ctx_id)
        with self._pinned_ctx(new_ctx_id):
            new_entry = _SyntheticSessionEntry(
                turns_prepared=int(wire_obj.get("turns_prepared", 1)),
                logical_bytes=base_logical_b,
                prompt_history=list(wire_obj.get("prompt_history", [])),
                last_image_tag=wire_obj.get("last_image_tag"),
            )
            self._contexts[new_ctx_id] = new_entry
            self._ensure_kv_region(new_ctx_id, new_entry)
        return base_logical_b

    def set_residency_alpha(self, alpha: float) -> float:
        if self._pager is not None and self._vmm is not None:
            self._pager.set_residency_alpha(float(alpha))
            return self.resident_alpha()
        if self.policy.use_expert_pager:
            q = round(float(alpha) * self.num_expert_layers) / float(self.num_expert_layers)
            self._alpha = max(0.0, min(1.0, q))
        return round(self._alpha, 3)

    def resident_alpha(self) -> float:
        if self._pager is not None and self._vmm is not None:
            return round(self._pager.resident_alpha(), 3)
        return round(self._alpha, 3)

    def release_context(self, ctx_id: str) -> None:
        entry = self._contexts.pop(ctx_id, None)
        self._active_pins.pop(ctx_id, None)
        if entry is None:
            return
        if entry.swapped_to_host:
            self._kv_host_pinned_bytes = max(0, self._kv_host_pinned_bytes - entry.swapped_bytes)
        if entry.kv_region_id is not None and self._vmm is not None:
            self._vmm.kv_region_free(entry.kv_region_id)

    def memory_and_vmm_stats(self) -> dict[str, Any]:
        vmm_st = self._vmm.stats_dict() if self._vmm is not None else {}
        vmm_pool_mib = round(float(vmm_st.get("pool_total_bytes", 0)) / MIB, 2) if vmm_st else 0.0
        swapped_cnt = sum(1 for e in self._contexts.values() if e.swapped_to_host)
        return {
            "peak_reserved_mib": round(self._peak_mib, 2),
            "combined_physical_mib": round(self._peak_mib + vmm_pool_mib, 2),
            "nvml_peak_used_mib": round(self._peak_mib + vmm_pool_mib, 2),
            "peak_nvml_used_mib": round(self._peak_mib + vmm_pool_mib, 2),
            "nvml_sm_util_mean_pct": 0.0,
            "nvml_sm_util_samples": 0,
            "vmm_physical_pool_mib": vmm_pool_mib,
            "vmm_stats": vmm_st,
            "active_contexts": len(self._contexts),
            "kv_free_bytes": self.available_kv_bytes(),
            "swapped_kv_contexts": swapped_cnt,
            "kv_host_swap_outs": self._kv_host_swap_outs,
            "kv_host_swap_ins": self._kv_host_swap_ins,
            "kv_host_pinned_bytes": self._kv_host_pinned_bytes,
        }

    def close(self) -> None:
        for cid in list(self._contexts.keys()):
            self.release_context(cid)
        if self._pager is not None:
            self._pager.close()
            self._pager = None
        if self._vmm is not None:
            self._vmm.shutdown()
            self._vmm = None
