"""Offline unit tests for Module 1 (TriPool-VMM C++17 build, memfd mirror, and BAGEL layer pack plans)."""

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

from morphserve.vmm.bagel_pager import ALIGN_BYTES, build_layer_pack_plans  # noqa: E402
from morphserve.vmm.tripool import NumaMemfdMirror, UnifiedVmmStats  # noqa: E402


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
        # Exact vllm_omni Qwen2MoTDecoderLayer parameter names & dimensions (444.52 MiB per expert!)
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


class TestModule1VmmOffline(unittest.TestCase):
    def test_libtripool_vmm_compiles_zero_warnings(self) -> None:
        if not shutil.which("g++"):
            self.skipTest("g++ not present")
        with tempfile.TemporaryDirectory() as tmpdir:
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
            # Verify exported C symbols load cleanly via ctypes
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
        # 444.5 MiB aligns cleanly into 7 x 64 MiB = 448 MiB (>99% efficiency!)
        self.assertEqual(stride_bytes, 448 * 1024 * 1024)
        for plan in plans:
            self.assertEqual(len(plan.params), 9)
            for ps in plan.params:
                self.assertEqual(ps.offset % ALIGN_BYTES, 0)
            self.assertLessEqual(plan.used_bytes, stride_bytes)
            self.assertGreater(plan.used_bytes / stride_bytes, 0.99)

    def test_memfd_mirror_read_write(self) -> None:
        mirror = NumaMemfdMirror(4 * 1024 * 1024, numa_node=0, name="test_mirror")
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


if __name__ == "__main__":
    unittest.main()
