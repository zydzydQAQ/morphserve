"""Offline unit tests for Module 1 (TriPool-VMM C++17 build, Arena 1 KV regions, and LayerExpertPager)."""

from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from morphserve.vmm import (  # noqa: E402
    ALIGN_BYTES,
    ExpertPartitionSpec,
    LayerExpertPager,
    LayerPackPlan,
    NumaMemfdMirror,
    ParamSlice,
    SimulatedTriPoolBackend,
    UnifiedVmmStats,
    VmmNaiveCache,
    build_layer_pack_plans,
)

MIB = 1024**2


def _scratch_dir() -> str | None:
    d_root = Path("/mnt/d/Transfer/MORPHSERVE")
    if Path("/mnt/d/Transfer").is_dir():
        d_root.mkdir(parents=True, exist_ok=True)
        return str(d_root)
    return None


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
            lib = ctypes.CDLL(str(so))
            for sym in (
                "tripool_vmm_init",
                "tripool_vmm_shutdown",
                "tripool_vmm_get_stats",
                "tripool_alloc",
                "tripool_free",
                "tripool_scope_begin",
                "tripool_scope_end",
                "tripool_kv_alloc_page",
                "tripool_kv_free_page",
                "tripool_kv_region_alloc",
                "tripool_kv_region_ensure",
                "tripool_kv_region_va",
                "tripool_kv_region_mapped_bytes",
                "tripool_kv_region_free",
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

            # 128 MiB transient VAE decode spike borrows 2 tail expert handles:
            # transient handles are reclaimed immediately, while borrowed tail slices stay lazily unmapped!
            self.assertEqual(vae.decode(128), "decoded_ok")
            st = vmm.stats_dict()
            self.assertEqual(st["transient_mapped_handles_current"], 0)
            self.assertGreaterEqual(st["expert_auto_evictions_for_transient"], 2)
            self.assertAlmostEqual(pager.resident_alpha(), 0.5)

            # Next DiT generation_mode asynchronously streams the 2 missing tail layers back to alpha=1.0
            with pager.generation_mode(prefetch_all=True):
                self.assertAlmostEqual(pager.resident_alpha(), 1.0)
        finally:
            pager.close()
            mirror.close()
            vmm.shutdown()

    def test_vmm_naive_cache_cow_fork_and_deepcopy(self) -> None:
        from copy import deepcopy
        from morphserve.kernels import cow_fork_context

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
            # cow_fork_context creates a zero-copy view without allocating new VMM regions
            fork_ctx = cow_fork_context({"kv_lens": [16], "ropes": [16], "past_key_values": c0})
            self.assertEqual(fork_ctx["past_key_values"].allocated_regions, 0)
            self.assertEqual(c0.allocated_regions, regions_before)
            # deepcopy allocates independent VMM Arena-1 regions without pickling vmm CDLL
            c_dup = deepcopy(c0)
            self.assertEqual(c_dup.allocated_regions, regions_before)
            c_dup.close()
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

    def test_module1_script_syntax_and_stats_struct(self) -> None:
        script = REPO_ROOT / "deploy/kairos/scripts/04_phase1_module1_validate.sh"
        proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        st = UnifiedVmmStats()
        self.assertIn("expert_auto_evictions_for_transient", st.to_dict())
        self.assertIn("expert_auto_evictions_for_kv", st.to_dict())


if __name__ == "__main__":
    unittest.main()
