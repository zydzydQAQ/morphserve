"""Public-seam TDD tests for `TriPoolVMM`, `VmmNaiveCache`, `LayerExpertPager`, and `NumaMemfdMirror`."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import pytest

from morphserve.kernels import cow_fork_context
from morphserve.vmm import (
    ExpertPartitionSpec,
    LayerExpertPager,
    LayerPackPlan,
    NumaMemfdMirror,
    ParamSlice,
    TriPoolVMM,
    VmmNaiveCache,
    VmmOutOfMemoryError,
)

MIB = 1024**2


@dataclass
class _FakeOpaqueOutput:
    """Non-dict, non-tensor output verifying `TriPoolVMM` delegates custom cloning strictly to `clone_fn`."""

    sample: bytearray
    tag: str


class _SpyPacer:
    """Deterministic `HalfNodeH2DPacer` double recording process-slot enter/exit transitions."""

    def __init__(self) -> None:
        self.entered_ranks: list[int] = []
        self.exited_ranks: list[int] = []
        self.active_count: int = 0

    @contextmanager
    def acquire_process_slot(self, rank: int):
        self.entered_ranks.append(int(rank))
        self.active_count += 1
        try:
            yield self
        finally:
            self.active_count -= 1
            self.exited_ranks.append(int(rank))


class _FakeScopedVae:
    """Synthetic VAE exercising `TriPoolVMM` Arena-2 transient spikes during encode/decode."""

    def __init__(self, vmm: TriPoolVMM, spike_mib: int) -> None:
        self._vmm = vmm
        self._spike_mib = int(spike_mib)
        self.encode_calls = 0
        self.decode_calls = 0

    def encode(self, payload: Any) -> Any:
        self.encode_calls += 1
        self._vmm.allocate_transient_mib(self._spike_mib)
        return payload

    def decode(self, payload: Any) -> Any:
        self.decode_calls += 1
        self._vmm.allocate_transient_mib(self._spike_mib)
        return payload


def _make_sim_vmm(
    *,
    super_pool_mib: int = 16,
    fine_kv_pool_mib: int = 8,
    super_handle_mib: int = 8,
    allow_overflow: bool = False,
    materialize_va_bytes: bool = True,
    kv_swap_high_watermark: float = 0.85,
    kv_swap_low_watermark: float = 0.65,
    kv_swap_min_idle_s: float = 0.0,
) -> TriPoolVMM:
    return TriPoolVMM(
        device=0,
        super_pool_mib=super_pool_mib,
        fine_kv_pool_mib=fine_kv_pool_mib,
        super_handle_mib=super_handle_mib,
        transient_va_gib=1,
        kv_va_gib=1,
        allow_overflow=allow_overflow,
        simulated=True,
        materialize_va_bytes=materialize_va_bytes,
        kv_swap_high_watermark=kv_swap_high_watermark,
        kv_swap_low_watermark=kv_swap_low_watermark,
        kv_swap_min_idle_s=kv_swap_min_idle_s,
        init_torch_cuda=False,
    )


def _fill_cache_4mib(cache: VmmNaiveCache, tag: int) -> tuple[np.ndarray, np.ndarray]:
    """Append 1 layer of K and V (2 MiB each = 4 MiB total) into `cache` using public `append_kv`."""
    # 32 tokens x [128, 128] float32 = 32 * 65536 = 2 MiB per K or V
    k = np.full((32, 128, 128), float(tag), dtype=np.float32)
    v = np.full((32, 128, 128), float(tag) + 0.5, dtype=np.float32)
    cache.append_kv(0, k, v)
    return k, v


def test_kv_inplace_va_stability_and_cow_isolation() -> None:
    vmm = _make_sim_vmm(super_pool_mib=16, fine_kv_pool_mib=16, super_handle_mib=8)
    cache = VmmNaiveCache(num_layers=2, vmm=vmm, max_seq_len=4096)
    fork_cache: VmmNaiveCache | None = None
    try:
        # BAGEL-like row size: 4 x 128 float16 = 1024 B/token -> 2048 tokens per 2 MiB page
        rng = np.random.default_rng(2026)
        k_part1 = rng.standard_normal((1500, 4, 128)).astype(np.float16)
        v_part1 = rng.standard_normal((1500, 4, 128)).astype(np.float16)
        k_part2 = rng.standard_normal((1000, 4, 128)).astype(np.float16)
        v_part2 = rng.standard_normal((1000, 4, 128)).astype(np.float16)

        for lidx in range(2):
            cache.append_kv(lidx, k_part1, v_part1)
        k0_before, _, len_before = cache.get_layer_kv(0)
        ptr_before = int(k0_before.data_ptr())
        assert len_before == 1500
        assert cache.peek_seq_len(0) == 1500
        assert cache.logical_bytes() == 2 * 2 * 1500 * 1024

        # Grow across the 2 MiB physical page boundary (1500 -> 2500 tokens): VA base must stay identical
        for lidx in range(2):
            cache.append_kv(lidx, k_part2, v_part2)
        k0_after, v0_after, len_after = cache.get_layer_kv(0)
        assert len_after == 2500
        assert int(k0_after.data_ptr()) == ptr_before
        assert np.array_equal(np.asarray(k0_after[:1500]), k_part1)
        assert np.array_equal(np.asarray(k0_after[1500:]), k_part2)
        assert np.array_equal(np.asarray(v0_after[1500:]), v_part2)

        # Read-only append (`update_past_key_values=False`) leaves cached length unchanged
        k_probe = np.zeros((4, 4, 128), dtype=np.float16)
        v_probe = np.zeros((4, 4, 128), dtype=np.float16)
        k_ro, _ = cache.append_kv(0, k_probe, v_probe, update_past_key_values=False)
        assert k_ro.shape[0] == 2504
        assert cache.peek_seq_len(0) == 2500

        # CoW fork aliases parent storage initially, then isolates on write without mutating parent
        forked_ctx = cow_fork_context(
            {"past_key_values": cache, "kv_lens": [2500], "ropes": [1]},
            prefix_len=1500,
            cache_factory=lambda n: VmmNaiveCache(n, vmm=vmm, max_seq_len=4096),
        )
        fork_cache = forked_ctx["past_key_values"]
        assert fork_cache is not None
        k_fork_before, _, _ = fork_cache.get_layer_kv(0)
        assert int(k_fork_before.data_ptr()) == ptr_before

        k_delta = np.ones((8, 4, 128), dtype=np.float16) * np.float16(7.0)
        v_delta = np.ones((8, 4, 128), dtype=np.float16) * np.float16(-3.0)
        fork_cache.append_kv(0, k_delta, v_delta)

        k_fork_after, _, fork_len = fork_cache.get_layer_kv(0)
        assert fork_len == 1508
        assert int(k_fork_after.data_ptr()) != ptr_before
        assert np.array_equal(np.asarray(k_fork_after[:1500]), k_part1)
        assert np.array_equal(np.asarray(k_fork_after[1500:]), k_delta)
        # Parent remains completely isolated
        k_parent_check, _, parent_len = cache.get_layer_kv(0)
        assert parent_len == 2500
        assert np.array_equal(np.asarray(k_parent_check[:1500]), k_part1)
        assert np.array_equal(np.asarray(k_parent_check[1500:]), k_part2)
    finally:
        if fork_cache is not None:
            fork_cache.close()
        cache.close()
        vmm.shutdown()


def test_nested_transient_scope_and_custom_clone_fn() -> None:
    vmm = _make_sim_vmm(super_pool_mib=24, fine_kv_pool_mib=0, super_handle_mib=8)
    try:
        with pytest.raises(RuntimeError):
            vmm.allocate_transient_mib(8)

        assert vmm.scope_depth == 0
        with vmm.transient_scope():
            assert vmm.scope_depth == 1
            p_outer = vmm.allocate_transient_mib(8)
            assert p_outer != 0
            assert vmm.stats().transient_mapped_handles_current == 1

            with vmm.transient_scope():
                assert vmm.scope_depth == 2
                p_inner = vmm.allocate_transient_mib(8)
                assert p_inner != 0 and p_inner != p_outer
                assert vmm.stats().transient_mapped_handles_current == 2

            # Exiting inner scope must keep outer scope mapped (no premature bulk unmap)
            assert vmm.scope_depth == 1
            assert vmm.stats().transient_mapped_handles_current >= 1
            assert vmm.stats().transient_scope_flushes == 0

        # Exiting outermost scope flushes and reclaims all Arena-2 super handles
        assert vmm.scope_depth == 0
        assert vmm.stats().transient_mapped_handles_current == 0
        assert vmm.stats().transient_scope_flushes == 1

        # Custom `clone_fn` clones opaque outputs without `tripool.py` knowing `diffusers`
        source_buf = bytearray(b"latent-payload-0123456789")
        cloner_calls: list[str] = []

        def custom_cloner(obj: _FakeOpaqueOutput) -> _FakeOpaqueOutput:
            cloner_calls.append(obj.tag)
            return _FakeOpaqueOutput(sample=bytearray(obj.sample), tag=obj.tag)

        def produce_inside_scope() -> _FakeOpaqueOutput:
            vmm.allocate_transient_mib(8)
            return _FakeOpaqueOutput(sample=source_buf, tag="vae_out")

        cloned = vmm.run_in_transient_scope(produce_inside_scope, clone_fn=custom_cloner)
        assert cloner_calls == ["vae_out"]
        assert isinstance(cloned, _FakeOpaqueOutput)
        assert bytes(cloned.sample) == bytes(source_buf)
        source_buf[:] = b"X" * len(source_buf)
        assert bytes(cloned.sample) == b"latent-payload-0123456789"
        assert vmm.stats().transient_mapped_handles_current == 0

        # Without `clone_fn`, generic `_clone_outputs` does not inspect custom object attributes
        opaque = _FakeOpaqueOutput(sample=bytearray(b"untouched"), tag="plain")
        returned = vmm.run_in_transient_scope(lambda: opaque)
        assert returned is opaque
    finally:
        vmm.shutdown()


def test_active_turn_scope_unpinning_and_multi_session_exclusion() -> None:
    # 16 MiB fine pool -> holds 3 caches x 4 MiB = 12 MiB (75% occupancy)
    vmm = _make_sim_vmm(
        super_pool_mib=0,
        fine_kv_pool_mib=16,
        super_handle_mib=8,
        kv_swap_high_watermark=0.80,
        kv_swap_low_watermark=0.50,
        kv_swap_min_idle_s=0.0,
    )
    c1 = VmmNaiveCache(num_layers=1, vmm=vmm, max_seq_len=128, max_row_bytes=65536)
    c2 = VmmNaiveCache(num_layers=1, vmm=vmm, max_seq_len=128, max_row_bytes=65536)
    c3 = VmmNaiveCache(num_layers=1, vmm=vmm, max_seq_len=128, max_row_bytes=65536)
    try:
        k1, _ = _fill_cache_4mib(c1, tag=11)
        k2, _ = _fill_cache_4mib(c2, tag=22)
        k3, _ = _fill_cache_4mib(c3, tag=33)

        # `active_turn_scope(mark_thinking_on_exit=False)` must pin during turn and unpin in C++ on exit
        c1.mark_thinking(trigger_watermark_check=False)
        with c1.active_turn_scope(mark_thinking_on_exit=False):
            assert c1.is_pinned_active is True
            # Even with no exclude list, C++ pinned session cannot be evicted
            c2.mark_thinking(trigger_watermark_check=False)
            c3.mark_thinking(trigger_watermark_check=False)
            reclaimed = vmm.reclaim_thinking_kv_bytes(
                4 * MIB, sync=True, exclude_caches=(c2, c3)
            )
            assert reclaimed == 0
            assert not c1.is_swapped

        # After exiting with `mark_thinking_on_exit=False`, `c1` is unpinned in both Python and C++
        assert c1.is_pinned_active is False
        assert c1.is_thinking is False
        c1.mark_thinking(trigger_watermark_check=False)

        # Multi-session exclusion via `exclude_caches=(c1, c2)` must spare BOTH c1 and c2 and evict ONLY c3
        reclaimed_c3 = vmm.reclaim_thinking_kv_bytes(
            4 * MIB, sync=True, exclude_caches=(c1, c2)
        )
        assert reclaimed_c3 >= 4 * MIB
        assert not c1.is_swapped
        assert not c2.is_swapped
        assert c3.is_swapped

        # Restore c3 and verify bit-exact round-trip through Host Pinned DRAM
        assert c3.ensure_resident() is True
        k3_restored, _, _ = c3.get_layer_kv(0)
        assert np.array_equal(np.asarray(k3_restored), k3)
        c3.mark_thinking(trigger_watermark_check=False)

        # Multi-session exclusion via `exclude_session_ids` on watermark eviction must spare BOTH c1 and c3
        wm_evicted = vmm.maybe_evict_thinking_kv_on_high_watermark(
            sync=True,
            exclude_session_ids=(c1.session_id, c3.session_id),
            extra_needed_bytes=4 * MIB,
        )
        assert wm_evicted >= 4 * MIB
        assert not c1.is_swapped
        assert c2.is_swapped
        assert not c3.is_swapped

        # Restore c2 and confirm exact contents across all caches
        assert c2.ensure_resident() is True
        assert np.array_equal(np.asarray(c1.get_layer_kv(0)[0]), k1)
        assert np.array_equal(np.asarray(c2.get_layer_kv(0)[0]), k2)
    finally:
        c1.close()
        c2.close()
        c3.close()
        vmm.shutdown()


@pytest.mark.parametrize("vae_encode_restore,expected_alpha", [("sync", 1.0), ("none", 0.0)])
def test_expert_headroom_generation_mode_pacer_and_vae_restore(
    vae_encode_restore: str, expected_alpha: float
) -> None:
    # 16 MiB total pool = 2 x 8 MiB super handles (2 expert layers x 8 MiB stride)
    vmm = _make_sim_vmm(
        super_pool_mib=16,
        fine_kv_pool_mib=0,
        super_handle_mib=8,
        allow_overflow=False,
        materialize_va_bytes=True,
        kv_swap_min_idle_s=0.0,
    )
    vae = _FakeScopedVae(vmm, spike_mib=16)
    plans = [
        LayerPackPlan(
            layer_idx=lid,
            used_bytes=4096,
            params=[ParamSlice(name=f"w{lid}", offset=0, nbytes=4096, shape=(1024,), dtype_str="float32")],
        )
        for lid in range(2)
    ]
    spec = ExpertPartitionSpec(
        name=f"test_exp_{vae_encode_restore}",
        plans=plans,
        layer_stride_bytes=8 * MIB,
        hook_modules=[],
        vae_module=vae,
        install_module_hooks=False,
        vae_encode_restore=vae_encode_restore,
    )
    pager = LayerExpertPager(spec, numa_node=0)
    mirror = NumaMemfdMirror.create(nbytes=16 * MIB, numa_node=0, name=f"exp_{vae_encode_restore}")
    think_cache = VmmNaiveCache(num_layers=2, vmm=vmm, max_seq_len=128, max_row_bytes=65536)
    pacer = _SpyPacer()
    try:
        pager.attach_mirror(mirror, owns_mirror=True)
        assert pager.owns_mirror is True
        pager.bind_vmm(vmm, initial_alpha=0.5)
        assert pager.resident_alpha() == pytest.approx(0.5)

        # Occupy the remaining 8 MiB with a thinking KV cache (2 layers x 4 MiB)
        for lidx in range(2):
            k = np.full((32, 128, 128), float(lidx + 1), dtype=np.float32)
            v = np.full((32, 128, 128), float(lidx + 2), dtype=np.float32)
            think_cache.append_kv(lidx, k, v)
        think_cache.mark_thinking(trigger_watermark_check=False)

        # `generation_mode` atomically calls `ensure_expert_headroom`, acquires pacer, and streams missing layers
        with pager.generation_mode(prefetch_all=True, pacer=pacer, rank=3):
            assert pager.is_generation_active is True
            assert pager.resident_alpha() == pytest.approx(1.0)
            assert pager.all_resident_synced is True
            assert think_cache.is_swapped is True
        assert pager.is_generation_active is False
        assert pacer.entered_ranks == [3]
        assert pacer.exited_ranks == [3]
        assert pacer.active_count == 0

        # When already 100% resident, `generation_mode` must NOT acquire the H2D pacer
        with pager.generation_mode(prefetch_all=True, pacer=pacer, rank=3):
            assert pager.resident_alpha() == pytest.approx(1.0)
        assert pacer.entered_ranks == [3]

        # Declarative `vae_encode_restore`: encode triggers 16 MiB transient spike evicting unpinned expert layers
        assert vae.encode("marker") == "marker"
        assert vae.encode_calls == 1
        assert pager.resident_alpha() == pytest.approx(expected_alpha)
        assert vmm.stats().transient_mapped_handles_current == 0
    finally:
        think_cache.close()
        pager.close()
        vmm.shutdown()


def test_invalid_vae_encode_restore_rejected() -> None:
    with pytest.raises(ValueError):
        ExpertPartitionSpec(
            name="bad_spec",
            plans=[],
            layer_stride_bytes=8 * MIB,
            hook_modules=[],
            vae_encode_restore="invalid_mode",
        )


def test_vmm_out_of_memory_error_uniform_on_exhaustion() -> None:
    vmm = _make_sim_vmm(super_pool_mib=8, fine_kv_pool_mib=4, super_handle_mib=8, allow_overflow=False)
    cache = VmmNaiveCache(num_layers=3, vmm=vmm, max_seq_len=128, max_row_bytes=65536)
    try:
        with cache.active_turn_scope(mark_thinking_on_exit=False):
            for lidx in range(3):
                k = np.ones((32, 128, 128), dtype=np.float32)
                v = np.ones((32, 128, 128), dtype=np.float32)
                cache.append_kv(lidx, k, v)  # 3 x 4 MiB = 12 MiB -> pool completely full

            with pytest.raises(VmmOutOfMemoryError):
                vmm.kv_alloc_page()

            with vmm.transient_scope():
                with pytest.raises(VmmOutOfMemoryError):
                    vmm.allocate_transient_mib(8)
    finally:
        cache.close()
        vmm.shutdown()

    # Verify explicit `max_pool_bytes` cap is accepted cleanly (used by `kv_vmm_only` / `prism_tetriserve_coloc`)
    vmm_cap = TriPoolVMM(
        device=0,
        super_pool_mib=8,
        fine_kv_pool_mib=4,
        super_handle_mib=8,
        transient_va_gib=1,
        kv_va_gib=1,
        allow_overflow=False,
        simulated=True,
        max_pool_bytes=12 * MIB,
        init_torch_cuda=False,
    )
    try:
        assert vmm_cap.max_pool_bytes == 12 * MIB
        assert vmm_cap.stats().total_physical_pool_bytes >= 12 * MIB
    finally:
        vmm_cap.shutdown()


def test_numa_memfd_mirror_cross_fd_shared_mapping() -> None:
    payload = bytes((i * 73 + 19) & 0xFF for i in range(4096))
    owner = NumaMemfdMirror.create(
        nbytes=64 * 1024,
        numa_node=0,
        name="ms_test_shared_mirror",
        populate_bytes=payload,
    )
    attached: NumaMemfdMirror | None = None
    try:
        assert owner.is_owner is True
        assert owner.fd >= 0
        assert owner.inode != 0
        assert owner.read_bytes(0, len(payload)) == payload

        attached = NumaMemfdMirror.attach_proc_fd(os.getpid(), owner.fd, owner.nbytes, numa_node=0)
        assert attached.is_owner is False
        assert attached.inode == owner.inode
        assert attached.read_bytes(0, len(payload)) == payload

        patch = b"morphserve-shared-memfd-coherency-check"
        attached.write_bytes(8192, patch)
        assert owner.read_bytes(8192, len(patch)) == patch
    finally:
        if attached is not None:
            attached.close()
        owner.close()
