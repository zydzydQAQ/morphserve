"""Deep `CPTGModelAdapter` interface and local-substitutable `SyntheticCPTGAdapter`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
import ctypes
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

from morphserve.vmm import (
    ExpertPartitionSpec,
    LayerExpertPager,
    LayerPackPlan,
    ParamSlice,
    SimulatedTriPoolBackend,
)

MIB = 1024**2
GIB = 1024**3


class SyntheticOutOfMemoryError(MemoryError):
    """Deterministic CPU test exception treated identically to CUDA/FA3 OOM."""


def is_oom_exception(exc: BaseException) -> bool:
    """Classify PyTorch CUDA OOM, VMM MemoryError, cuBLAS/cuDNN workspace OOM, and FA3 C++ dispatcher OOM."""
    if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
        return True
    msg = str(exc)
    low = msg.lower()
    return (
        "out of memory" in low
        or "out of vmm memory" in low
        or "memory allocation failed" in low
        or "cudaerrormemoryallocation" in low
        or "cublas_status_alloc_failed" in low
        or "cudnn_status_alloc_failed" in low
        or "ms_vmm_expert_load_layers_sync failed" in low
        or "exhausted: need" in low
        or "pool exhausted" in low
        or 'torch_call_dispatcher( "aten::new_empty"' in msg
        or 'torch_call_dispatcher( "aten::empty' in msg
        or 'torch_call_dispatcher( "aten::resize_' in msg
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
            enq_s = float(cmd.get("enqueue_time_s", 0.0))
            t_start = time.perf_counter()
            self.prepare_turn(cid, cmd)
            out = self.decode_ar(cid, int(cmd.get("max_ar_tokens", 32)))
            q_wait_ms = max(0.0, (t_start - enq_s) * 1000.0) if enq_s > 0.0 else 0.0
            out = dict(out)
            out["ttft_ms"] = round(q_wait_ms + float(out.get("ttft_ms", 0.0)), 2)
            out["resident_alpha"] = self.resident_alpha()
            out["kv_free_bytes"] = self.available_kv_bytes()
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

    def reset_for_new_suite(self) -> float:
        """Reset worker residency to its initial bound alpha before starting an independent benchmark suite."""
        pol = getattr(self, "policy", None)
        default_alpha = (
            float(pol.initial_alphas.get(getattr(self, "rank", 0), 1.0))
            if hasattr(pol, "initial_alphas")
            else 1.0
        )
        target = float(getattr(self, "_initial_bound_alpha", default_alpha))
        return self.set_residency_alpha(target)

    def available_kv_bytes(self, *, for_dit: bool = False) -> int:
        """Return current free + drop-clean reclaimable GPU KV capacity in bytes on this worker."""
        return 0

    def evict_idle_kv_to_host(
        self,
        target_free_bytes: int,
        exclude_ctx_ids: Iterable[str] = (),
        *,
        for_dit: bool = False,
    ) -> int:
        """Swap unpinned idle think-time sessions' KV to Host Pinned DRAM until `available_kv_bytes >= target`."""
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
            self._pager.mirror = shared_mirror
            self._pager._owns_mirror = False
        elif hasattr(os, "memfd_create"):
            self._pager.mirror = NumaMemfdMirror.create(nbytes=4 * MIB, numa_node=0, name=f"syn_m_{self.rank}")
            self._pager._owns_mirror = True
        else:
            self._pager.mirror = _FallbackHostMirror(nbytes=4 * MIB, numa_node=0)
            self._pager._owns_mirror = True

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
                # Reclaim fine pages from idle sessions before irreversible 1-way carving in non-coalescing sim
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
                enq_s = float(cmd.get("enqueue_time_s", 0.0))
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
                # Run VAE decode OUTSIDE generation_mode so borrowed tail handles stay lazily unmapped!
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
