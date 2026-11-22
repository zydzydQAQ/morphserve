"""Offline & CUDA-driver unit tests for Module 1 (TriPool-VMM C++17 build, bidirectional mapping, async KV swap & VmmNaiveCache)."""

from __future__ import annotations

from copy import deepcopy
import ctypes
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from morphserve.kernels import cow_fork_context, detect_lecp_prefix_len  # noqa: E402
from morphserve.vmm import (  # noqa: E402
    ALIGN_BYTES,
    ExpertPartitionSpec,
    LayerExpertPager,
    LayerPackPlan,
    NumaMemfdMirror,
    ParamSlice,
    SimulatedTriPoolBackend,
    TriPoolVMM,
    UnifiedVmmStats,
    VmmNaiveCache,
    build_layer_pack_plans,
)

MIB = 1024**2


def _scratch_dir() -> str:
    d_root = Path("/mnt/d/Transfer/MORPHSERVE")
    if Path("/mnt/d/Transfer").is_dir():
        d_root.mkdir(parents=True, exist_ok=True)
        return str(d_root)
    fallback = REPO_ROOT / "build" / "pytest_tmp"
    fallback.mkdir(parents=True, exist_ok=True)
    return str(fallback)


def _probe_live_vmm_gpu(min_free_mib: int = 384) -> bool:
    """Return True if a local CUDA Driver VMM-capable GPU with >= `min_free_mib` free HBM is available."""
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        if int(lib.cuInit(0)) != 0:
            return False
        cnt = ctypes.c_int(0)
        if int(lib.cuDeviceGetCount(ctypes.byref(cnt))) != 0 or cnt.value < 1:
            return False
        dev = ctypes.c_int(0)
        if int(lib.cuDeviceGet(ctypes.byref(dev), 0)) != 0:
            return False
        vmm_sup = ctypes.c_int(0)
        if int(lib.cuDeviceGetAttribute(ctypes.byref(vmm_sup), 102, dev.value)) != 0 or vmm_sup.value == 0:
            return False
        ctx = ctypes.c_void_p()
        if int(lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev.value)) != 0:
            return False
        try:
            lib.cuCtxSetCurrent(ctx)
            free_b = ctypes.c_size_t(0)
            tot_b = ctypes.c_size_t(0)
            mem_info = getattr(lib, "cuMemGetInfo_v2", lib.cuMemGetInfo)
            if int(mem_info(ctypes.byref(free_b), ctypes.byref(tot_b))) != 0:
                return False
            return (free_b.value // MIB) >= min_free_mib
        finally:
            lib.cuDevicePrimaryCtxRelease(dev.value)
    except Exception:
        return False


class _MockParam:
    def __init__(self, shape: tuple[int, ...], elem_bytes: int = 2, dtype: str = "torch.bfloat16"):
        self.shape = shape
        self._elem = elem_bytes
        self.dtype = dtype

    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def element_size(self) -> int:
        return self._elem


class _MockBagelLayer:
    def __init__(self) -> None:
        self._params = [
            ("self_attn.qkv_proj.weight", _MockParam((4608, 3584))),
            ("self_attn.qkv_proj.bias", _MockParam((4608,))),
            ("self_attn.qkv_proj.gen_exp.weight", _MockParam((4608, 3584))),
            ("self_attn.qkv_proj.gen_exp.bias", _MockParam((4608,))),
            ("self_attn.o_proj.weight", _MockParam((3584, 3584))),
            ("self_attn.o_proj.gen_exp.weight", _MockParam((3584, 3584))),
            ("self_attn.q_norm.weight", _MockParam((128,))),
            ("self_attn.q_norm.gen_weight", _MockParam((128,))),
            ("self_attn.k_norm.weight", _MockParam((128,))),
            ("self_attn.k_norm.gen_weight", _MockParam((128,))),
            ("input_layernorm.weight", _MockParam((3584,))),
            ("input_layernorm.gen_weight", _MockParam((3584,))),
            ("mlp.gate_up_proj.weight", _MockParam((37888, 3584))),
            ("mlp.down_proj.weight", _MockParam((3584, 18944))),
            ("mlp_moe_gen.gate_up_proj.weight", _MockParam((37888, 3584))),
            ("mlp_moe_gen.down_proj.weight", _MockParam((3584, 18944))),
            ("post_attention_layernorm.weight", _MockParam((3584,))),
            ("post_attention_layernorm.gen_weight", _MockParam((3584,))),
        ]

    def named_parameters(self):
        return iter(self._params)


class _DummyVae:
    def __init__(self, vmm_getter) -> None:
        self._vmm_getter = vmm_getter

    def decode(self, spike_mib: int) -> str:
        self._vmm_getter().allocate_transient_mib(spike_mib)
        return "decoded_ok"

    def encode(self, spike_mib: int) -> str:
        self._vmm_getter().allocate_transient_mib(spike_mib)
        return "encoded_ok"


class _StockNaiveCache:
    """Minimal pure-Python replica of `vllm_omni` `NaiveCache` for polymorphic compatibility checks."""

    def __init__(self, n: int):
        self.key_cache = {k: None for k in range(n)}
        self.value_cache = {k: None for k in range(n)}
        self.key_values_lens = None

    @property
    def num_layers(self) -> int:
        return len(self.key_cache)

    @classmethod
    def from_object(cls, obj):
        if isinstance(obj, cls):
            return obj
        out = cls(len(obj.key_cache))
        for i, (k, v) in enumerate(zip(obj.key_cache, obj.value_cache, strict=True)):
            out.key_cache[i] = k
            out.value_cache[i] = v
        return out


class TestModule1VmmOffline(unittest.TestCase):
    def test_libtripool_vmm_compiles_zero_warnings_and_exports_kv_region_abi(self) -> None:
        if not shutil.which("g++"):
            self.skipTest("g++ not present")
        with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmpdir:
            tmp = Path(tmpdir)
            env = {
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "SRC_DIR": str(REPO_ROOT),
                "BUILD_TMP": str(tmp / "build"),
                "INSTALL_TMP": str(tmp / "install"),
            }
            proc = subprocess.run(
                ["bash", str(REPO_ROOT / "csrc/vmm_tripool/build.sh")],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
            self.assertEqual(proc.stderr.strip(), "", msg=f"unexpected compiler warnings:\n{proc.stderr}")
            so = tmp / "install/lib/libtripool_vmm.so"
            self.assertTrue(so.is_file())
            self.assertTrue((tmp / "install/tests/test_phase2_m21_m22_offline.py").is_file())
            self.assertTrue((tmp / "install/deploy/kairos/scripts/16_osdi27_stage1_1gpu_gate.sh").is_file())
            self.assertTrue((tmp / "install/GLOSSARY.md").is_file())
            self.assertTrue((tmp / "install/results/phase0-job0c/topo-8gpu-report.json").is_file())
            self.assertEqual(list((tmp / "install/results").rglob("*.log")), [])
            from morphserve.sim import MeasuredPhysicsProfile

            staged_ph = MeasuredPhysicsProfile.from_phase01_results(tmp / "install/results")
            self.assertEqual(len(staged_ph.provenance_files), 7)
            lib = ctypes.CDLL(str(so))
            for sym in (
                "tripool_vmm_init",
                "tripool_vmm_shutdown",
                "tripool_vmm_get_stats",
                "tripool_alloc",
                "tripool_free",
                "tripool_scope_begin",
                "tripool_scope_end",
                "tripool_kv_free_bytes",
                "tripool_kv_alloc_page",
                "tripool_kv_free_page",
                "tripool_kv_region_alloc",
                "tripool_kv_region_ensure",
                "tripool_kv_region_va",
                "tripool_kv_region_mapped_bytes",
                "tripool_kv_region_free",
                "tripool_kv_region_set_evictable",
                "tripool_kv_region_swap_out_async",
                "tripool_kv_region_swap_in_async",
                "tripool_kv_region_is_swapped",
                "tripool_kv_region_swapped_bytes",
                "tripool_kv_flush_swaps",
                "tripool_expert_register",
                "tripool_expert_set_layer_used_bytes",
                "tripool_expert_prefetch_async",
                "tripool_expert_wait_layer",
                "tripool_expert_evict_range",
            ):
                self.assertTrue(hasattr(lib, sym), msg=f"missing symbol {sym}")

    def test_bagel_28layer_pack_plan_alignment(self) -> None:
        layers = [_MockBagelLayer() for _ in range(28)]
        plans, stride_bytes = build_layer_pack_plans(layers, role="generation", super_handle_mib=64)
        self.assertEqual(len(plans), 28)
        self.assertEqual(stride_bytes, 448 * MIB)
        for plan in plans:
            self.assertEqual(len(plan.params), 9)
            for ps in plan.params:
                self.assertEqual(ps.offset % ALIGN_BYTES, 0)
            self.assertLessEqual(plan.used_bytes, stride_bytes)
            self.assertGreater(plan.used_bytes / stride_bytes, 0.99)

    def test_arena1_kv_region_carving_and_tail_expert_eviction(self) -> None:
        # Super pool = 4 x 64 MiB (occupied by 4 expert layers), Fine KV pool = 64 MiB (32 x 2 MiB)
        vmm = SimulatedTriPoolBackend(
            device=0,
            super_pool_mib=256,
            fine_kv_pool_mib=64,
            super_handle_mib=64,
            allow_overflow=False,
        )
        mirror = NumaMemfdMirror.create(nbytes=4 * MIB, numa_node=0, name="kv_carve_test")
        plans = [
            LayerPackPlan(
                layer_idx=i,
                used_bytes=4096,
                params=[ParamSlice(name="w", offset=0, nbytes=4096, shape=(2048,), dtype_str="bf16")],
            )
            for i in range(4)
        ]
        pager = LayerExpertPager(
            ExpertPartitionSpec(
                name="syn_carve",
                plans=plans,
                layer_stride_bytes=64 * MIB,
                hook_modules=[],
                install_module_hooks=False,
            ),
            numa_node=0,
        )
        try:
            pager.mirror = mirror
            pager._owns_mirror = False
            pager.bind_vmm(vmm, initial_alpha=1.0)
            self.assertEqual(vmm.expert_resident_count(pager.expert_id), 4)

            # Reserve 128 MiB contiguous VA region and map 96 MiB (> 64 MiB initial fine pool):
            # MUST evict 1 unpinned tail expert super-handle and carve 64 MiB -> 32 x 2 MiB pages!
            rid = vmm.kv_region_alloc(128 * MIB)
            vmm.kv_region_ensure(rid, 96 * MIB)
            self.assertEqual(vmm.kv_region_mapped_bytes(rid), 96 * MIB)
            st = vmm.stats_dict()
            self.assertGreaterEqual(st["super_to_fine_carves"], 1)
            self.assertGreaterEqual(st["expert_auto_evictions_for_kv"], 1)
            self.assertEqual(vmm.expert_resident_count(pager.expert_id), 3)

            vmm.kv_region_free(rid)
            self.assertEqual(vmm.stats_dict()["kv_mapped_pages_current"], 0)
        finally:
            pager.close()
            mirror.close()
            vmm.shutdown()

    def test_asymmetric_lazy_vae_decode_reclamation_avoids_sync_reload(self) -> None:
        vmm = SimulatedTriPoolBackend(
            device=0,
            super_pool_mib=256,  # 4 x 64 MiB handles
            fine_kv_pool_mib=64,
            super_handle_mib=64,
            allow_overflow=False,
        )
        vae = _DummyVae(lambda: vmm)
        mirror = NumaMemfdMirror.create(nbytes=4 * MIB, numa_node=0, name="lazy_vae_test")
        plans = [
            LayerPackPlan(
                layer_idx=i,
                used_bytes=4096,
                params=[ParamSlice(name="w", offset=0, nbytes=4096, shape=(2048,), dtype_str="bf16")],
            )
            for i in range(4)
        ]
        pager = LayerExpertPager(
            ExpertPartitionSpec(
                name="syn_vae",
                plans=plans,
                layer_stride_bytes=64 * MIB,
                hook_modules=[],
                vae_module=vae,
                install_module_hooks=False,
            ),
            numa_node=0,
        )
        try:
            pager.mirror = mirror
            pager._owns_mirror = False
            pager.bind_vmm(vmm, initial_alpha=1.0)
            self.assertAlmostEqual(pager.resident_alpha(), 1.0)

            self.assertEqual(vae.decode(128), "decoded_ok")
            st = vmm.stats_dict()
            self.assertEqual(st["transient_mapped_handles_current"], 0)
            self.assertGreaterEqual(st["expert_auto_evictions_for_transient"], 2)
            self.assertAlmostEqual(pager.resident_alpha(), 0.5)

            with pager.generation_mode(prefetch_all=True):
                self.assertAlmostEqual(pager.resident_alpha(), 1.0)
        finally:
            pager.close()
            mirror.close()
            vmm.shutdown()

    def test_vmm_naive_cache_cow_fork_and_deepcopy(self) -> None:
        vmm = SimulatedTriPoolBackend(device=0, super_pool_mib=128, fine_kv_pool_mib=64)
        c0 = VmmNaiveCache(num_layers=2, vmm=vmm, max_seq_len=128)
        try:
            class _FakeKV:
                shape = (16, 4, 128)
                dtype = "bf16"
                device = "cpu"
                def numel(self) -> int:
                    return 16 * 4 * 128
                def element_size(self) -> int:
                    return 2

            for l in range(2):
                c0.key_cache[l] = _FakeKV()
                c0.value_cache[l] = _FakeKV()
            regions_before = c0.allocated_regions
            fork_ctx = cow_fork_context({"kv_lens": [16], "ropes": [16], "past_key_values": c0})
            self.assertEqual(fork_ctx["past_key_values"].allocated_regions, 0)
            self.assertEqual(c0.allocated_regions, regions_before)
            c_dup = deepcopy(c0)
            self.assertEqual(c_dup.allocated_regions, regions_before)
            c_dup.close()

            cb = VmmNaiveCache.create_for_bagel(_StockNaiveCache, 2, vmm=vmm, max_seq_len=128)
            cb.key_cache[0] = _FakeKV()
            cb.value_cache[0] = _FakeKV()
            self.assertIsInstance(cb, _StockNaiveCache)
            self.assertIs(_StockNaiveCache.from_object(cb), cb)
            self.assertEqual(_StockNaiveCache.from_object(cb).key_cache[0].shape, (16, 4, 128))
            cb.close()
        finally:
            c0.close()
            vmm.shutdown()

    def test_memfd_mirror_read_write(self) -> None:
        mirror = NumaMemfdMirror(4 * MIB, numa_node=0, name="test_mirror")
        try:
            self.assertGreater(mirror.ptr, 0)
            ctypes.memset(ctypes.c_void_p(mirror.ptr), 0xAB, 4096)
            self.assertEqual(mirror.mm[0], 0xAB)
            self.assertEqual(mirror.mm[4095], 0xAB)
        finally:
            mirror.close()

        # Verify `libc.so.6` `memfd_create` fallback (used by cos7/glibc-2.17 conda Python builds)
        # preserves genuine kernel `/proc/<pid>/fd/<fd>` `MAP_SHARED` sharing and error handling.
        with mock.patch.object(os, "memfd_create", None, create=True):
            m_owner = NumaMemfdMirror.create(
                nbytes=4 * MIB, numa_node=0, name="libc_fallback_mfd", populate_bytes=b"\x5A" * 128
            )
            m_attached: NumaMemfdMirror | None = None
            try:
                self.assertIn("memfd:", os.readlink(f"/proc/{os.getpid()}/fd/{m_owner.fd}"))
                m_attached = NumaMemfdMirror.attach_proc_fd(os.getpid(), m_owner.fd, nbytes=4 * MIB, numa_node=0)
                self.assertEqual(m_attached.inode, m_owner.inode)
                self.assertEqual(m_attached.read_bytes(0, 128), b"\x5A" * 128)
                m_owner.write_bytes(256, b"morphserve_mfd_ok")
                self.assertEqual(m_attached.read_bytes(256, 17), b"morphserve_mfd_ok")
            finally:
                if m_attached is not None:
                    m_attached.close()
                m_owner.close()

            with self.assertRaises(ValueError):
                NumaMemfdMirror.create(nbytes=4096, name="bad\x00name")
            with self.assertRaises(OSError):
                NumaMemfdMirror.create(nbytes=4096, name="x" * 300)

            # Also exercise the secondary `libc.syscall(SYS_memfd_create, ...)` fallback when
            # `libc.so.6` lacks a `memfd_create` wrapper symbol (pre-2.27 glibc sysroots).
            real_libc = ctypes.CDLL("libc.so.6", use_errno=True)

            class _SyscallOnlyLibc:
                memfd_create = None
                syscall = real_libc.syscall

            with mock.patch("morphserve.vmm.tripool.ctypes.CDLL", return_value=_SyscallOnlyLibc()):
                m_sys = NumaMemfdMirror.create(
                    nbytes=64 * 1024, numa_node=0, name="syscall_fallback_mfd", populate_bytes=b"\x7E" * 64
                )
                try:
                    self.assertIn("memfd:", os.readlink(f"/proc/{os.getpid()}/fd/{m_sys.fd}"))
                    self.assertEqual(m_sys.read_bytes(0, 64), b"\x7E" * 64)
                finally:
                    m_sys.close()
                with self.assertRaises(OSError):
                    NumaMemfdMirror.create(nbytes=4096, name="y" * 300)

        # Oversized `populate_bytes` must fail cleanly without leaking any open memfd FD.
        fds_before = set(os.listdir(f"/proc/{os.getpid()}/fd"))
        with self.assertRaises(ValueError):
            NumaMemfdMirror.create(nbytes=4096, name="oversized_pop", populate_bytes=b"\x01" * 8192)
        gc.collect()
        self.assertEqual(set(os.listdir(f"/proc/{os.getpid()}/fd")), fds_before)

        # Exercise real PyTorch tensors (`bfloat16` KV + `as_uint8_tensor`) when PyTorch is installed.
        try:
            import torch as _torch
        except ImportError:
            _torch = None  # type: ignore[assignment]
        if _torch is not None:
            m_pt = NumaMemfdMirror.create(nbytes=64 * 1024, numa_node=0, name="pt_mirror_test")
            vmm_pt = SimulatedTriPoolBackend(device=0, super_pool_mib=64, fine_kv_pool_mib=16, allow_overflow=False)
            c_pt = VmmNaiveCache(num_layers=1, vmm=vmm_pt, max_seq_len=256, max_row_bytes=256)
            try:
                t_u8 = m_pt.as_uint8_tensor()
                t_u8[:8].copy_(_torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=_torch.uint8))
                self.assertEqual(m_pt.read_bytes(0, 8), bytes([1, 2, 3, 4, 5, 6, 7, 8]))

                k_bf16 = _torch.arange(64 * 4 * 32, dtype=_torch.float32).reshape(64, 4, 32).to(_torch.bfloat16)
                v_bf16 = (-k_bf16).contiguous()
                fk_pt, fv_pt = c_pt.append_inplace(0, k_bf16[:48], v_bf16[:48])
                ptr0 = int(fk_pt.data_ptr())
                fk_pt, fv_pt = c_pt.append_inplace(0, k_bf16[48:], v_bf16[48:])
                self.assertEqual(int(fk_pt.data_ptr()), ptr0)
                c_pt.swap_out_async()
                self.assertTrue(c_pt.is_swapped_out)
                c_pt.mark_active()
                self.assertTrue(_torch.equal(c_pt.key_cache[0], k_bf16))
                self.assertTrue(_torch.equal(c_pt.value_cache[0], v_bf16))
            finally:
                c_pt.close()
                vmm_pt.shutdown()
                m_pt.close()

    def test_module1_script_syntax_and_stats_struct(self) -> None:
        script = REPO_ROOT / "deploy/kairos/scripts/04_phase1_module1_validate.sh"
        proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        st = UnifiedVmmStats()
        d = st.to_dict()
        self.assertIn("expert_auto_evictions_for_transient", d)
        self.assertIn("expert_auto_evictions_for_kv", d)
        self.assertIn("fine_to_super_bundles", d)
        self.assertIn("kv_free_bytes", d)
        self.assertIn("kv_dit_headroom_bytes", d)
        self.assertIn("kv_swap_outs_total", d)
        self.assertIn("kv_swap_ins_total", d)

    def test_bidirectional_pool_recovery_eliminates_one_way_decay(self) -> None:
        """C2.a: Carving 100% of 64 MiB super-handles into 2 MiB pages must NOT prevent Arena 2/3 recovery."""
        vmm = SimulatedTriPoolBackend(
            device=0,
            super_pool_mib=256,  # 4 x 64 MiB handles
            fine_kv_pool_mib=64,  # 32 x 2 MiB pages -> total = 320 MiB
            super_handle_mib=64,
            allow_overflow=False,
        )
        vae = _DummyVae(lambda: vmm)
        mirror = NumaMemfdMirror.create(nbytes=4 * MIB, numa_node=0, name="bidir_pool_test")
        plans = [
            LayerPackPlan(
                layer_idx=i,
                used_bytes=4096,
                params=[ParamSlice(name="w", offset=0, nbytes=4096, shape=(2048,), dtype_str="bf16")],
            )
            for i in range(4)
        ]
        pager = LayerExpertPager(
            ExpertPartitionSpec(
                name="syn_bidir",
                plans=plans,
                layer_stride_bytes=64 * MIB,
                hook_modules=[],
                vae_module=vae,
                install_module_hooks=False,
            ),
            numa_node=0,
        )
        try:
            pager.mirror = mirror
            pager._owns_mirror = False
            pager.bind_vmm(vmm, initial_alpha=1.0)
            self.assertEqual(vmm.stats_dict()["pool_total_bytes"], 320 * MIB)

            # Allocate 320 MiB of KV -> evicts ALL 4 expert layers and carves ALL 4 super-handles -> 160 fine pages!
            rid = vmm.kv_region_alloc(320 * MIB)
            vmm.kv_region_ensure(rid, 320 * MIB)
            st_carved = vmm.stats_dict()
            self.assertEqual(st_carved["super_pool_total_handles"], 0)
            self.assertEqual(st_carved["fine_pool_total_pages"], 160)
            self.assertEqual(st_carved["pool_total_bytes"], 320 * MIB)
            self.assertEqual(st_carved["super_to_fine_carves"], 4)
            self.assertEqual(pager.resident_alpha(), 0.0)

            # Release KV -> 0 super-handles exist, 160 fine pages are free
            vmm.kv_region_free(rid)
            self.assertEqual(vmm.stats_dict()["super_pool_free_handles"], 0)
            self.assertEqual(vmm.stats_dict()["fine_pool_free_pages"], 160)

            # Arena 3 Expert reload MUST assemble 4 x (32 x 2 MiB) fine bundles with ZERO overflow!
            with pager.generation_mode(prefetch_all=True):
                self.assertAlmostEqual(pager.resident_alpha(), 1.0)
                st_exp = vmm.stats_dict()
                self.assertEqual(st_exp["expert_fine_bundles_current"], 4)
                self.assertGreaterEqual(st_exp["fine_to_super_bundles"], 4)
                self.assertEqual(st_exp["overflow_creates"], 0)

            # Arena 2 VAE 192 MiB transient spike evicts 3 fine-bundled expert layers (preserving 64 MiB KV base pool)
            self.assertEqual(vae.decode(192), "decoded_ok")
            st_post_vae = vmm.stats_dict()
            self.assertEqual(st_post_vae["transient_mapped_handles_current"], 0)
            self.assertEqual(st_post_vae["overflow_creates"], 0)
            self.assertEqual(st_post_vae["pool_total_bytes"], 320 * MIB)
            self.assertAlmostEqual(pager.resident_alpha(), 0.25)

            # Arena 2 VAE 320 MiB spike absorbs 100% of the 320 MiB pool (5 x 32x2MiB bundles) with zero overflow!
            self.assertEqual(vae.decode(320), "decoded_ok")
            self.assertEqual(vmm.stats_dict()["overflow_creates"], 0)
            self.assertEqual(vmm.stats_dict()["transient_mapped_handles_current"], 0)
            self.assertAlmostEqual(pager.resident_alpha(), 0.0)
        finally:
            pager.close()
            mirror.close()
            vmm.shutdown()

    def test_realtime_kv_free_bytes_and_dit_headroom_ledger(self) -> None:
        """C2.b: `kv_free_bytes(for_dit=False/True)` reflects real-time KV, expert pinning, and swap state."""
        vmm = SimulatedTriPoolBackend(
            device=0,
            super_pool_mib=256,  # 4 x 64 MiB
            fine_kv_pool_mib=64,  # 32 x 2 MiB -> 320 MiB total
            super_handle_mib=64,
            allow_overflow=False,
        )
        mirror = NumaMemfdMirror.create(nbytes=4 * MIB, numa_node=0, name="kv_ledger_test")
        plans = [
            LayerPackPlan(layer_idx=i, used_bytes=4096, params=[])
            for i in range(3)  # 3 x 64 = 192 MiB expert
        ]
        pager = LayerExpertPager(
            ExpertPartitionSpec(
                name="syn_ledger",
                plans=plans,
                layer_stride_bytes=64 * MIB,
                hook_modules=[],
                install_module_hooks=False,
            ),
            numa_node=0,
        )
        try:
            pager.mirror = mirror
            pager._owns_mirror = False
            pager.bind_vmm(vmm, initial_alpha=1.0)

            # Unpinned expert (192 MiB) is drop-clean reclaimable for AR KV, so kv_free=320 MiB, dit_headroom=128 MiB
            self.assertEqual(vmm.kv_free_bytes(for_dit=False), 320 * MIB)
            self.assertEqual(vmm.kv_free_bytes(for_dit=True), 128 * MIB)

            # Pinning expert during active DiT drops kv_free_bytes to 128 MiB immediately!
            with pager.generation_mode(prefetch_all=False):
                self.assertEqual(vmm.kv_free_bytes(for_dit=False), 128 * MIB)
                self.assertEqual(vmm.kv_free_bytes(for_dit=True), 128 * MIB)

            # Allocate 80 MiB KV -> kv_free drops to 240 MiB, dit_headroom drops to 48 MiB
            rid = vmm.kv_region_alloc(128 * MIB)
            vmm.kv_region_ensure(rid, 80 * MIB)
            self.assertEqual(vmm.kv_free_bytes(for_dit=False), 240 * MIB)
            self.assertEqual(vmm.kv_free_bytes(for_dit=True), 48 * MIB)

            # Swapping out the 80 MiB KV region restores kv_free=320 MiB and dit_headroom=128 MiB!
            vmm.kv_region_swap_out_async(rid, 80 * MIB)
            self.assertEqual(vmm.sync_kv_free_bytes(for_dit=False), 320 * MIB)
            self.assertEqual(vmm.sync_kv_free_bytes(for_dit=True), 128 * MIB)
            vmm.kv_region_free(rid)
        finally:
            pager.close()
            mirror.close()
            vmm.shutdown()

    def test_high_watermark_thinking_kv_async_swap_prevents_multiturn_oom(self) -> None:
        """C2.c: Thinking-period sessions asynchronously swap KV to Host Pinned DRAM and restore bit-exact at same VA."""
        vmm = SimulatedTriPoolBackend(
            device=0,
            super_pool_mib=256,   # 4 x 64 MiB
            fine_kv_pool_mib=64,  # 32 x 2 MiB -> 320 MiB strict pool
            super_handle_mib=64,
            allow_overflow=False,
            kv_swap_high_watermark=0.80,
            kv_swap_low_watermark=0.50,
        )
        mirror = NumaMemfdMirror.create(nbytes=4 * MIB, numa_node=0, name="swap_oom_test")
        plans = [LayerPackPlan(layer_idx=i, used_bytes=4096, params=[]) for i in range(4)]  # 256 MiB expert
        pager = LayerExpertPager(
            ExpertPartitionSpec(
                name="syn_swap",
                plans=plans,
                layer_stride_bytes=64 * MIB,
                hook_modules=[],
                install_module_hooks=False,
            ),
            numa_node=0,
        )
        sessions: list[VmmNaiveCache] = []
        rng = np.random.default_rng(2026)
        expected_arrays: list[tuple[np.ndarray, np.ndarray, int]] = []

        try:
            pager.mirror = mirror
            pager._owns_mirror = False
            pager.bind_vmm(vmm, initial_alpha=1.0)

            # Each session has 2 layers x (K,V) = 4 regions, each 24 MiB -> 96 MiB/session.
            # 4 sessions x 96 MiB = 384 MiB KV (> 320 MiB total pool!), plus 256 MiB Expert!
            rows = 3072  # 3072 * (16 * 256 * 2B) = 3072 * 8192 B = 24 MiB per tensor
            for s_idx in range(4):
                sc = VmmNaiveCache(num_layers=2, vmm=vmm, max_seq_len=4096, max_row_bytes=8192)
                sc.mark_active()
                k0_ref = rng.integers(-1000, 1000, size=(rows, 16, 256), dtype=np.int16)
                v0_ref = rng.integers(-1000, 1000, size=(rows, 16, 256), dtype=np.int16)
                k1_ref = rng.integers(-1000, 1000, size=(rows, 16, 256), dtype=np.int16)
                v1_ref = rng.integers(-1000, 1000, size=(rows, 16, 256), dtype=np.int16)
                fk0, _ = sc.append_inplace(0, k0_ref, v0_ref)
                sc.append_inplace(1, k1_ref, v1_ref)
                expected_arrays.append((k0_ref, v0_ref, int(fk0.data_ptr())))
                sessions.append(sc)
                if s_idx < 3:
                    # Sessions 0..2 enter user think time; Session 3 stays pinned active!
                    sc.mark_thinking()

            # Session 3 is active so it MUST remain resident on GPU, while older thinking sessions swapped out!
            self.assertFalse(sessions[3].is_swapped_out)
            self.assertTrue(sessions[0].is_swapped_out)
            self.assertGreater(sessions[0].swapped_bytes, 0)
            self.assertEqual(sessions[0].mapped_bytes, 0)

            # deepcopy of a swapped-out VmmNaiveCache must transparently restore and copy valid data
            dup0 = deepcopy(sessions[0])
            self.assertTrue(np.array_equal(np.asarray(dup0.key_cache[0]), expected_arrays[0][0]))
            dup0.close()
            sessions[0].mark_thinking()

            # Now Session 3 also enters think time, and DiT generation demands ALL 4 expert layers (256 MiB)!
            sessions[3].mark_thinking(trigger_watermark_check=False)
            with pager.generation_mode(prefetch_all=True):
                self.assertAlmostEqual(pager.resident_alpha(), 1.0)
                st_dit = vmm.stats_dict()
                self.assertEqual(st_dit["overflow_creates"], 0)
                self.assertGreater(st_dit["kv_swap_outs_total"], 0)
                self.assertGreater(st_dit["fine_to_super_bundles"], 0)

            # Wake all 4 sessions and verify fixed-VA pointer identity + 100% bit-exact KV restoration!
            for s_idx, sc in enumerate(sessions):
                k_ref, v_ref, orig_ptr = expected_arrays[s_idx]
                sc.mark_active()
                self.assertFalse(sc.is_swapped_out)
                k_restored = sc.key_cache[0]
                v_restored = sc.value_cache[0]
                self.assertEqual(int(k_restored.data_ptr()), orig_ptr)
                self.assertTrue(np.array_equal(np.asarray(k_restored), k_ref))
                self.assertTrue(np.array_equal(np.asarray(v_restored), v_ref))

                # Verify LECP CoW prefix pointer identity works immediately on restored cache
                fork_c = cow_fork_context({"kv_lens": [rows], "ropes": [rows], "past_key_values": sc})[
                    "past_key_values"
                ]
                self.assertEqual(detect_lecp_prefix_len(sc, fork_c), rows)
                sc.mark_thinking()
        finally:
            for sc in sessions:
                sc.close()
            pager.close()
            mirror.close()
            vmm.shutdown()

    def test_vmm_naive_cache_inplace_append_single_multi_batch_and_edge_cases(self) -> None:
        """C1: `append_inplace`, `append_layer_kv`, `append_batch_inplace`, CoW isolation & `truncate`."""
        vmm = SimulatedTriPoolBackend(device=0, super_pool_mib=128, fine_kv_pool_mib=32, allow_overflow=False)
        rng = np.random.default_rng(99)
        c_main = VmmNaiveCache(num_layers=2, vmm=vmm, max_seq_len=4096, max_row_bytes=1024)
        c_peer = VmmNaiveCache(num_layers=2, vmm=vmm, max_seq_len=4096, max_row_bytes=1024)
        c_stock = _StockNaiveCache(2)
        try:
            # 1. Prefill 2,044 tokens (< 2,048 tokens = 2 MiB boundary), then 12 single-step appends crossing 2 MiB!
            prefill_k = rng.integers(-500, 500, size=(2044, 4, 128), dtype=np.int16)
            prefill_v = rng.integers(-500, 500, size=(2044, 4, 128), dtype=np.int16)
            fk, fv = c_main.append_inplace(0, prefill_k, prefill_v)
            base_ptr_k = int(fk.data_ptr())
            self.assertEqual(c_main.mapped_bytes, 4 * MIB)  # 2 MiB K + 2 MiB V on layer 0

            step_ks = [prefill_k]
            step_vs = [prefill_v]
            for i in range(12):
                # Mix 2D [H, D] and 3D [1, H, D] single-step shapes!
                sk = rng.integers(-500, 500, size=(4, 128) if i % 2 == 0 else (1, 4, 128), dtype=np.int16)
                sv = rng.integers(-500, 500, size=(4, 128) if i % 2 == 0 else (1, 4, 128), dtype=np.int16)
                fk, fv = c_main.append_kv_step(0, sk, sv)
                step_ks.append(sk.reshape(1, 4, 128))
                step_vs.append(sv.reshape(1, 4, 128))

            self.assertEqual(fk.shape, (2056, 4, 128))
            self.assertEqual(int(fk.data_ptr()), base_ptr_k)  # Fixed VA across 2 MiB boundary!
            self.assertEqual(c_main.mapped_bytes, 8 * MIB)    # Grew to 4 MiB K + 4 MiB V seamlessly
            self.assertTrue(np.array_equal(np.asarray(fk), np.concatenate(step_ks, axis=0)))
            self.assertTrue(np.array_equal(np.asarray(fv), np.concatenate(step_vs, axis=0)))

            # Empty ΔL=0 append must be a safe no-op
            empty_step = np.zeros((0, 4, 128), dtype=np.int16)
            fk_noop, _ = c_main.append_inplace(0, empty_step, empty_step)
            self.assertEqual(fk_noop.shape, (2056, 4, 128))

            # 2. CoW child branch append MUST NOT clobber parent c_main!
            c_main.append_inplace(1, prefill_k[:32], prefill_v[:32])
            c_main.truncate(32)
            parent_snap = np.asarray(c_main.key_cache[0]).copy()
            fork_cache = cow_fork_context({"kv_lens": [32], "ropes": [32], "past_key_values": c_main})[
                "past_key_values"
            ]
            branch_tok = np.full((2, 4, 128), 777, dtype=np.int16)
            bfk, _ = fork_cache.append_inplace(0, branch_tok, branch_tok)
            self.assertEqual(bfk.shape, (34, 4, 128))
            # Parent c_main still has length 32 and exact uncorrupted values!
            self.assertEqual(c_main.key_cache[0].shape, (32, 4, 128))
            self.assertTrue(np.array_equal(np.asarray(c_main.key_cache[0]), parent_snap))

            # Also test VMM-backed CoW child write isolation
            vmm_child = VmmNaiveCache(num_layers=2, vmm=vmm, max_seq_len=128, max_row_bytes=1024)
            ro_slice = c_main.key_cache[0][:16]
            ro_slice._vmm_external_ro = True
            ro_v = c_main.value_cache[0][:16]
            ro_v._vmm_external_ro = True
            vmm_child.key_cache[0] = ro_slice
            vmm_child.value_cache[0] = ro_v
            self.assertEqual(vmm_child.allocated_regions, 0)
            vmm_child.append_inplace(0, branch_tok, branch_tok)
            self.assertEqual(vmm_child.allocated_regions, 2)
            self.assertEqual(vmm_child.key_cache[0].shape, (18, 4, 128))
            self.assertTrue(np.array_equal(np.asarray(c_main.key_cache[0]), parent_snap))
            vmm_child.close()

            # 3. Batched `append_batch_inplace` across heterogeneous [c_main, c_peer, c_stock]
            c_peer.append_inplace(0, prefill_k[:10], prefill_v[:10])
            c_stock.key_cache[0] = prefill_k[:5].copy()
            c_stock.value_cache[0] = prefill_v[:5].copy()

            # Uniform 1-token decode batch [B=3, H=4, D=128]
            b_k = rng.integers(-100, 100, size=(3, 4, 128), dtype=np.int16)
            b_v = rng.integers(-100, 100, size=(3, 4, 128), dtype=np.int16)
            ks_out, vs_out, lens_out = VmmNaiveCache.append_batch_inplace(
                [c_main, c_peer, c_stock], layer_idx=0, packed_k=b_k, packed_v=b_v
            )
            self.assertEqual(lens_out, [33, 11, 6])
            self.assertTrue(np.array_equal(np.asarray(ks_out[0][-1]), b_k[0]))
            self.assertTrue(np.array_equal(np.asarray(ks_out[1][-1]), b_k[1]))
            self.assertTrue(np.array_equal(np.asarray(ks_out[2][-1]), b_k[2]))

            # Varlen continuous batching [q_lens = (4, 0, 2)]
            var_k = rng.integers(-100, 100, size=(6, 4, 128), dtype=np.int16)
            var_v = rng.integers(-100, 100, size=(6, 4, 128), dtype=np.int16)
            _, _, lens_var = VmmNaiveCache.append_batch_inplace(
                [c_main, c_peer, c_stock], layer_idx=0, packed_k=var_k, packed_v=var_v, query_lens=[4, 0, 2]
            )
            self.assertEqual(lens_var, [37, 11, 8])

            c_main.truncate(0)
            self.assertEqual(c_main.seq_lens, 0)
            self.assertEqual(c_main.allocated_regions, 0)
        finally:
            c_main.close()
            c_peer.close()
            vmm.shutdown()

    def test_live_cuda_driver_vmm_bidirectional_and_async_host_pinned_swap(self) -> None:
        """Live NVIDIA `libcuda.so.1` VMM verification of heterogeneous 64MiB+32x2MiB map/unmap & async KV swap."""
        if not shutil.which("g++"):
            self.skipTest("g++ not present")
        if not _probe_live_vmm_gpu(min_free_mib=384):
            self.skipTest("no local VMM-capable NVIDIA GPU with >= 384 MiB free HBM")

        with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmpdir:
            tmp = Path(tmpdir)
            env = {
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "SRC_DIR": str(REPO_ROOT),
                "BUILD_TMP": str(tmp / "build"),
                "INSTALL_TMP": str(tmp / "install"),
            }
            proc = subprocess.run(
                ["bash", str(REPO_ROOT / "csrc/vmm_tripool/build.sh")],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            so_path = tmp / "install/lib/libtripool_vmm.so"

            # 2 x 64 MiB super + 32 MiB fine = 160 MiB strict pool (allow_overflow=False)
            mirror: NumaMemfdMirror | None = None
            pager: LayerExpertPager | None = None
            c_think: VmmNaiveCache | None = None
            vmm = TriPoolVMM(
                device=0,
                super_pool_mib=128,
                fine_kv_pool_mib=32,
                super_handle_mib=64,
                transient_va_gib=1,
                kv_va_gib=1,
                allow_overflow=False,
                so_path=so_path,
                init_torch_cuda=False,
            )
            try:
                mirror = NumaMemfdMirror.create(
                    nbytes=128 * MIB,
                    numa_node=0,
                    name="live_cuda_vmm_test",
                    populate_bytes=b"\x3D" * 8192,
                )
                plans = [LayerPackPlan(layer_idx=i, used_bytes=8192, params=[]) for i in range(2)]  # 2 x 64 MiB
                pager = LayerExpertPager(
                    ExpertPartitionSpec(
                        name="live_cuda_exp",
                        plans=plans,
                        layer_stride_bytes=64 * MIB,
                        hook_modules=[],
                        install_module_hooks=False,
                    ),
                    numa_node=0,
                )
                c_think = VmmNaiveCache(num_layers=1, vmm=vmm, max_seq_len=8192, max_row_bytes=8192)
                pager.mirror = mirror
                pager._owns_mirror = False
                pager.bind_vmm(vmm, initial_alpha=1.0)
                self.assertEqual(vmm.kv_free_bytes(for_dit=False), 160 * MIB)
                self.assertEqual(vmm.kv_free_bytes(for_dit=True), 32 * MIB)

                # Append 4,608 rows * 8 KiB = 36 MiB K + 36 MiB V = 72 MiB KV (> 32 MiB fine pool!):
                # Forces 1 expert eviction + 1 super->fine carve on real NVIDIA VMM!
                rng = np.random.default_rng(7)
                k_src = rng.integers(-10000, 10000, size=(4608, 32, 128), dtype=np.int16)
                v_src = rng.integers(-10000, 10000, size=(4608, 32, 128), dtype=np.int16)
                c_think.mark_active()
                fk_gpu, _ = c_think.append_inplace(0, k_src[:250], v_src[:250])
                va_ptr_0 = int(fk_gpu.data_ptr())
                fk_gpu, fv_gpu = c_think.append_inplace(0, k_src[250:], v_src[250:])
                self.assertEqual(int(fk_gpu.data_ptr()), va_ptr_0)
                c_think.mark_thinking(trigger_watermark_check=False)

                # Entering DiT generation_mode requires 128 MiB expert (only 1 super-handle remains!):
                # MUST async swap out c_think to Host Pinned DRAM AND map mixed 1x64MiB + 32x2MiB fine bundle!
                with pager.generation_mode(prefetch_all=True):
                    self.assertAlmostEqual(pager.resident_alpha(), 1.0)
                    st_live = vmm.stats_dict()
                    self.assertEqual(st_live["overflow_creates"], 0)
                    self.assertGreater(st_live["kv_swap_outs_total"], 0)
                    self.assertGreater(st_live["fine_to_super_bundles"], 0)

                # Also verify Arena-2 Transient heterogeneous 120 MiB spike + bulk cuMemUnmap
                with vmm.transient_scope():
                    t_ptr = vmm.allocate_transient_mib(120)
                    self.assertGreater(t_ptr, 0)
                self.assertEqual(vmm.stats_dict()["transient_mapped_handles_current"], 0)

                # Wake c_think and verify bit-exact D2H readback at identical CUDA virtual address!
                self.assertTrue(c_think.is_swapped_out)
                c_think.mark_active()
                self.assertEqual(int(c_think.key_cache[0].data_ptr()), va_ptr_0)
                self.assertTrue(np.array_equal(c_think.key_cache[0].to_numpy(), k_src))
                self.assertTrue(np.array_equal(c_think.value_cache[0].to_numpy(), v_src))

                # Test C++-level on-demand thinking KV swap during raw `tripool_alloc` (CUDAPluggableAllocator path):
                # c_think holds 72 MiB (< 85% of 160 MiB pool), then raw C++ `tripool_alloc(120 MiB)` demands 2x64MiB!
                c_think.mark_thinking(trigger_watermark_check=False)
                vmm.lib.tripool_scope_begin()
                try:
                    raw_tptr = int(vmm.lib.tripool_alloc(ctypes.c_ssize_t(120 * MIB), 0, ctypes.c_void_p(0)))
                    self.assertGreater(raw_tptr, 0)
                    vmm.lib.tripool_free(ctypes.c_void_p(raw_tptr), ctypes.c_ssize_t(120 * MIB), 0, ctypes.c_void_p(0))
                finally:
                    vmm.lib.tripool_scope_end(ctypes.c_void_p(0))

                # Without calling `mark_active()`, `_is_swapped` property detects C++ demand swap & auto-restores!
                self.assertTrue(c_think.is_swapped_out)
                self.assertTrue(np.array_equal(c_think.key_cache[0].to_numpy(), k_src))
                self.assertEqual(int(c_think.key_cache[0].data_ptr()), va_ptr_0)

                # Test M4 mid-DiT AR preemption: all expert layers pinned (128 MiB), c_think parked in thinking state
                c_think.mark_thinking(trigger_watermark_check=False)
                with pager.generation_mode(prefetch_all=True):
                    c_urgent_ar = VmmNaiveCache(num_layers=1, vmm=vmm, max_seq_len=1024, max_row_bytes=8192)
                    try:
                        c_urgent_ar.mark_active()
                        ar_k = rng.integers(-500, 500, size=(512, 32, 128), dtype=np.int16)
                        ar_v = rng.integers(-500, 500, size=(512, 32, 128), dtype=np.int16)
                        c_urgent_ar.append_inplace(0, ar_k, ar_v)
                        self.assertTrue(np.array_equal(c_urgent_ar.key_cache[0].to_numpy(), ar_k))
                        self.assertEqual(vmm.stats_dict()["overflow_creates"], 0)
                    finally:
                        c_urgent_ar.close()

                # Multithreaded `kv_region_alloc` concurrent with async swap-out/swap-in (`std::deque` stability)
                extra_rids: list[int] = []
                errs: list[Exception] = []

                def _alloc_worker() -> None:
                    try:
                        for _ in range(48):
                            extra_rids.append(vmm.kv_region_alloc(2 * MIB))
                    except Exception as exc:  # pragma: no cover
                        errs.append(exc)

                th = threading.Thread(target=_alloc_worker)
                th.start()
                for _ in range(4):
                    c_think.swap_out_async()
                    c_think.mark_active()
                th.join(timeout=10.0)
                self.assertEqual(errs, [])
                for erid in extra_rids:
                    vmm.kv_region_free(erid)
                self.assertTrue(np.array_equal(c_think.key_cache[0].to_numpy(), k_src))
            finally:
                if c_think is not None:
                    c_think.close()
                if pager is not None:
                    pager.close()
                if mirror is not None:
                    mirror.close()
                vmm.shutdown()

            # Verify:
            # 1) `TriPoolVMM` re-assignment (`vmm_re = TriPoolVMM(...); vmm_re = TriPoolVMM(...)`) does NOT
            #    let the old wrapper's `__del__` tear down the live C++ singleton underneath the new wrapper.
            # 2) `unbind_vmm_keep_mirror() -> vmm1.shutdown() -> vmm2 = TriPoolVMM(...) -> bind_vmm(vmm2)`
            #    cleanly unpins/re-pins `NumaMemfdMirror` (including `bound_view` `_CudaVAView` slicing)
            #    with zero `CUDA_ERROR_CONTEXT_IS_DESTROYED` errors.
            # 3) `TriPoolVMM.__del__` GC cleanup when the last wrapper drops.
            # 4) Standalone `NumaMemfdMirror.pin_cuda()` before `TriPoolVMM` instantiation AND
            #    `~UnifiedTriPoolVMM()` static `atexit` teardown with in-flight streamer jobs.
            class _RawViewParam:
                def __init__(self) -> None:
                    self.raw_bytes = b"\x2A" * 4096
                    self.bound_view = None

            m_rebind = NumaMemfdMirror.create(
                nbytes=64 * MIB, numa_node=0, name="live_rebind_mfd", populate_bytes=b"\x2A" * 4096
            )
            raw_param = _RawViewParam()
            pager_rebind = LayerExpertPager(
                ExpertPartitionSpec(
                    name="live_rebind_exp",
                    plans=[
                        LayerPackPlan(
                            layer_idx=0,
                            used_bytes=4096,
                            params=[ParamSlice("w", offset=512, nbytes=2048, shape=(1024,), dtype_str="bf16", param_ref=raw_param)],
                        )
                    ],
                    layer_stride_bytes=64 * MIB,
                    hook_modules=[],
                    install_module_hooks=False,
                ),
                numa_node=0,
            )
            try:
                pager_rebind.mirror = m_rebind
                pager_rebind._owns_mirror = False
                vmm_re = TriPoolVMM(
                    device=0,
                    super_pool_mib=64,
                    fine_kv_pool_mib=4,
                    super_handle_mib=64,
                    transient_va_gib=1,
                    kv_va_gib=1,
                    allow_overflow=False,
                    so_path=so_path,
                    init_torch_cuda=False,
                )
                # Re-assign without intermediate `shutdown()`: old wrapper's `__del__` must NOT kill singleton!
                vmm_re = TriPoolVMM(
                    device=0,
                    super_pool_mib=64,
                    fine_kv_pool_mib=4,
                    super_handle_mib=64,
                    transient_va_gib=1,
                    kv_va_gib=1,
                    allow_overflow=False,
                    so_path=so_path,
                    init_torch_cuda=False,
                )
                gc.collect()
                rid_re = vmm_re.kv_region_alloc(2 * MIB)
                vmm_re.kv_region_free(rid_re)
                pager_rebind.bind_vmm(vmm_re, initial_alpha=1.0)
                self.assertIsNotNone(raw_param.bound_view)
                self.assertEqual(len(raw_param.bound_view), 2048)
                pager_rebind.unbind_vmm_keep_mirror()
                vmm_re.shutdown()
                self.assertFalse(m_rebind.pinned)

                vmm_gc = TriPoolVMM(
                    device=0,
                    super_pool_mib=64,
                    fine_kv_pool_mib=4,
                    super_handle_mib=64,
                    transient_va_gib=1,
                    kv_va_gib=1,
                    allow_overflow=False,
                    so_path=so_path,
                    init_torch_cuda=False,
                )
                pager_rebind.bind_vmm(vmm_gc, initial_alpha=1.0)
                self.assertTrue(m_rebind.pinned)
                pager_rebind.unbind_vmm_keep_mirror()
                del vmm_gc
                gc.collect()
                self.assertFalse(m_rebind.pinned)
            finally:
                pager_rebind.close()
                m_rebind.close()

            atexit_probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import ctypes, sys; "
                        f"sys.path.insert(0, {str(SRC_DIR)!r}); "
                        "from morphserve.vmm import NumaMemfdMirror; "
                        "m0 = NumaMemfdMirror.create(4096, name='standalone_pin'); "
                        "m0.pin_cuda(); m0.close(); "
                        f"lib = ctypes.CDLL({str(so_path)!r}); "
                        "lib.tripool_vmm_init.argtypes = ["
                        "ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, "
                        "ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int]; "
                        "lib.tripool_vmm_init.restype = ctypes.c_int; "
                        "lib.tripool_host_register.argtypes = [ctypes.c_void_p, ctypes.c_size_t]; "
                        "lib.tripool_host_register.restype = ctypes.c_int; "
                        "lib.tripool_expert_register.argtypes = ["
                        "ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]; "
                        "lib.tripool_expert_register.restype = ctypes.c_int; "
                        "lib.tripool_expert_prefetch_async.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]; "
                        "lib.tripool_expert_prefetch_async.restype = ctypes.c_int; "
                        "lib.tripool_kv_region_alloc.argtypes = [ctypes.c_size_t]; "
                        "lib.tripool_kv_region_alloc.restype = ctypes.c_int32; "
                        "lib.tripool_kv_region_ensure.argtypes = [ctypes.c_int32, ctypes.c_size_t, ctypes.c_void_p]; "
                        "lib.tripool_kv_region_ensure.restype = ctypes.c_uint64; "
                        "lib.tripool_kv_region_swap_out_async.argtypes = ["
                        "ctypes.c_int32, ctypes.c_size_t, ctypes.c_void_p]; "
                        "lib.tripool_kv_region_swap_out_async.restype = ctypes.c_int; "
                        "rc = lib.tripool_vmm_init(0, 64, 4, 64, 1, 1, 0); "
                        "assert rc == 0, rc; "
                        "m1 = NumaMemfdMirror.create(64 * 1024 * 1024, name='atexit_exp'); "
                        "assert lib.tripool_host_register(ctypes.c_void_p(m1.ptr), ctypes.c_size_t(m1.nbytes)) == 0; "
                        "eid = lib.tripool_expert_register(1, 64, ctypes.c_void_p(m1.ptr), ctypes.c_size_t(m1.nbytes)); "
                        "assert eid == 0, eid; "
                        "rid = lib.tripool_kv_region_alloc(2 * 1024 * 1024); "
                        "assert lib.tripool_kv_region_ensure(rid, 2 * 1024 * 1024, None) != 0; "
                        "lib.tripool_expert_prefetch_async(eid, 0, 1); "
                        "lib.tripool_kv_region_swap_out_async(rid, 65536, None)"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(atexit_probe.returncode, 0, msg=atexit_probe.stderr)
            self.assertNotIn("terminate called", atexit_probe.stderr)
            self.assertNotIn("host_unregister error", atexit_probe.stderr)

    def test_setitem_overwrite_va_growth_cow_parent_swap_and_gc_cleanup(self) -> None:
        """Regression suite for `__setitem__` same/longer-len overwrite, VA growth, CoW swap & `__del__` GC."""
        vmm = SimulatedTriPoolBackend(device=0, super_pool_mib=128, fine_kv_pool_mib=32, allow_overflow=False)
        rng = np.random.default_rng(1234)
        c = VmmNaiveCache(num_layers=1, vmm=vmm, max_seq_len=128, max_row_bytes=256)
        try:
            # 1. `__setitem__` with same length (L=32 -> 32) and longer unrelated tensor (32 -> 48) MUST copy!
            t_a = rng.integers(-200, 200, size=(32, 4, 32), dtype=np.int16)
            t_b_same_len = rng.integers(500, 900, size=(32, 4, 32), dtype=np.int16)
            t_c_longer = rng.integers(-900, -500, size=(48, 4, 32), dtype=np.int16)

            c.key_cache[0] = t_a
            self.assertTrue(np.array_equal(np.asarray(c.key_cache[0]), t_a))
            c.key_cache[0] = t_b_same_len
            self.assertTrue(np.array_equal(np.asarray(c.key_cache[0]), t_b_same_len))
            c.key_cache[0] = t_c_longer
            self.assertTrue(np.array_equal(np.asarray(c.key_cache[0]), t_c_longer))

            # Assigning 0-length tensor updates `seq_lens` to 0 cleanly
            c.key_cache[0] = np.zeros((0, 4, 32), dtype=np.int16)
            self.assertEqual(c.seq_lens, 0)

            # 2. Exceeding initial `max_seq_len=128` VA reservation (2 MiB min cap -> > 8,192 rows = > 2 MiB)
            #    transparently grows region VA and preserves prefix data!
            big_prefill = rng.integers(-100, 100, size=(8192, 4, 32), dtype=np.int16)  # exact 2 MiB
            big_extra = rng.integers(200, 300, size=(512, 4, 32), dtype=np.int16)     # crosses 2 MiB cap!
            c.key_cache[0] = big_prefill
            c.value_cache[0] = big_prefill
            fk, _ = c.append_inplace(0, big_extra, big_extra)
            self.assertEqual(fk.shape, (8704, 4, 32))
            self.assertTrue(np.array_equal(np.asarray(fk[:8192]), big_prefill))
            self.assertTrue(np.array_equal(np.asarray(fk[8192:]), big_extra))

            # 3. CoW child forked via `cow_fork_context` auto-restores swapped-out parent on read/append,
            #    and CoW rebind onto a cache that ALREADY owned a region preserves external prefix!
            c.truncate(64)
            snap_64 = np.asarray(c.key_cache[0]).copy()
            fork_child = cow_fork_context({"kv_lens": [64], "ropes": [64], "past_key_values": c})["past_key_values"]
            c.swap_out_async()
            self.assertTrue(c.is_swapped_out)
            # Accessing `fork_child.key_cache[0]` while `c` is swapped out auto-swaps `c` back in!
            self.assertTrue(np.array_equal(np.asarray(fork_child.key_cache[0]), snap_64))
            self.assertFalse(c.is_swapped_out)

            reused_child = VmmNaiveCache(num_layers=1, vmm=vmm, max_seq_len=128, max_row_bytes=256)
            reused_child.append_inplace(0, t_a[:8], t_a[:8])  # allocates private region first!
            ext_ro = c.key_cache[0][:20]
            ext_ro._vmm_external_ro = True
            ext_v = c.value_cache[0][:20]
            ext_v._vmm_external_ro = True
            reused_child.key_cache[0] = ext_ro
            reused_child.value_cache[0] = ext_v
            step_2 = np.full((2, 4, 32), 42, dtype=np.int16)
            rc_k, _ = reused_child.append_inplace(0, step_2, step_2)
            self.assertEqual(rc_k.shape, (22, 4, 32))
            self.assertTrue(np.array_equal(np.asarray(rc_k[:20]), snap_64[:20]))
            self.assertTrue(np.array_equal(np.asarray(rc_k[20:]), step_2))
            reused_child.close()

            # 4. `append_batch_inplace` shape validation rejects mismatched `query_lens` before mutating
            with self.assertRaises(ValueError):
                VmmNaiveCache.append_batch_inplace(
                    [c], layer_idx=0, packed_k=step_2, packed_v=step_2, query_lens=[5]
                )
            self.assertEqual(c.seq_lens, 64)

            # 5. Unclosed `VmmNaiveCache` releases Arena-1 physical pages automatically upon GC (`__del__`)
            pages_before = vmm.stats_dict()["kv_mapped_pages_current"]
            ephemeral = VmmNaiveCache(num_layers=1, vmm=vmm, max_seq_len=128, max_row_bytes=256)
            ephemeral.append_inplace(0, t_a, t_a)
            self.assertGreater(vmm.stats_dict()["kv_mapped_pages_current"], pages_before)
            del ephemeral
            gc.collect()
            self.assertEqual(vmm.stats_dict()["kv_mapped_pages_current"], pages_before)
        finally:
            c.close()
            vmm.shutdown()

    def test_for_dit_watermark_no_false_thrashing_and_demand_eviction(self) -> None:
        """Regression test: 86%-expert pool must NOT thrash small thinking KV, while demand spikes DO evict."""
        # Mirrors BAGEL tight pool ratio: 14,336 MiB super + 256 MiB fine = 14,592 MiB; 28 x 448 = 12,544 MiB (85.96%)
        vmm = SimulatedTriPoolBackend(
            device=0,
            super_pool_mib=14_336,
            fine_kv_pool_mib=256,
            super_handle_mib=64,
            allow_overflow=False,
            kv_swap_high_watermark=0.85,
            kv_swap_low_watermark=0.65,
        )
        vae = _DummyVae(lambda: vmm)
        mirror = NumaMemfdMirror.create(nbytes=4 * MIB, numa_node=0, name="no_thrash_test")
        plans = [LayerPackPlan(layer_idx=i, used_bytes=4096, params=[]) for i in range(28)]
        pager = LayerExpertPager(
            ExpertPartitionSpec(
                name="syn_bagel_ratio",
                plans=plans,
                layer_stride_bytes=448 * MIB,
                hook_modules=[],
                vae_module=vae,
                install_module_hooks=False,
            ),
            numa_node=0,
        )
        c_small = VmmNaiveCache(num_layers=1, vmm=vmm, max_seq_len=1024, max_row_bytes=8192)
        try:
            pager.mirror = mirror
            pager._owns_mirror = False
            pager.bind_vmm(vmm, initial_alpha=1.0)
            self.assertEqual(vmm.stats_dict()["physical_pool_bytes"], 14_592 * MIB)

            # Small 16 MiB thinking session (<< 85% of 2,048 MiB DiT KV budget = 1,740 MiB):
            arr = np.ones((1024, 32, 128), dtype=np.int16)  # 8 MiB K + 8 MiB V = 16 MiB
            c_small.mark_active()
            c_small.append_inplace(0, arr, arr)
            c_small.mark_thinking()

            # Entering DiT `generation_mode` MUST keep `c_small` hot in HBM (`0` false swap-outs)!
            with pager.generation_mode(prefetch_all=True):
                self.assertFalse(c_small.is_swapped_out)
                self.assertEqual(vmm.stats_dict()["kv_swap_outs_total"], 0)

            # Now simulate direct Arena-2 transient spike (`_acquire_super_slots` without Python pre-hook)
            # demanding all 14,592 MiB = 228 x 64 MiB slots: on-demand thinking-KV eviction swaps `c_small` cleanly!
            pager.set_residency_alpha(0.0)
            slots = vmm._acquire_super_slots(228, for_transient=True)
            vmm._release_slots(slots)
            self.assertTrue(c_small.is_swapped_out)
            self.assertGreater(vmm.stats_dict()["kv_host_pinned_pool_bytes"], 0)
            self.assertTrue(np.array_equal(np.asarray(c_small.key_cache[0]), arr))
            c_small.close()
            self.assertEqual(vmm.stats_dict()["kv_host_pinned_pool_bytes"], 0)
        finally:
            c_small.close()
            pager.close()
            mirror.close()
            vmm.shutdown()


if __name__ == "__main__":
    unittest.main()
