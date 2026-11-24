"""Offline unit tests for Module 3 (LECP CoW Cache, Cascade-CFG gate, GPU bicubic math, and syntax)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "python"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from morphserve.kernels import (  # noqa: E402
    SharedPrefixCache,
    cow_fork_context,
    detect_lecp_prefix_len,
    rebind_cow_prefix,
    should_use_cascade_cfg_split,
    should_use_flash_prefix_split,
)
import morphserve.kernels.cfg_prefix_attn as cfg_prefix_attn  # noqa: E402
from morphserve.kernels.gpu_loopback import (  # noqa: E402
    _extract_target_hw,
    pil_bicubic_resize_gpu,
    pil_bicubic_resize_np,
    tv_bicubic_u8_resize_gpu,
)


class _MockTensor:
    """Minimal CPU tensor duck-type for offline LECP tests when torch isn't installed on WSL."""

    def __init__(self, rows: list[int], base_id: int | None = None, offset: int = 0) -> None:
        self._rows = rows
        self._base_id = id(rows) if base_id is None else base_id
        self._offset = offset
        self.shape = (len(rows), 4, 128)
        self.dtype = "bf16"
        self.device = "mock:0"

    def data_ptr(self) -> int:
        return self._base_id * 100000 + self._offset

    def stride(self) -> tuple[int, int, int]:
        return (512, 128, 1)

    def __getitem__(self, slc):
        assert isinstance(slc, slice)
        start = slc.start or 0
        stop = len(self._rows) if slc.stop is None else slc.stop
        return _MockTensor(self._rows[start:stop], base_id=self._base_id, offset=self._offset + start)


class _MockCache:
    def __init__(self, num_layers: int) -> None:
        self.key_cache = {l: None for l in range(num_layers)}
        self.value_cache = {l: None for l in range(num_layers)}
        self.key_values_lens = None

    @property
    def num_layers(self) -> int:
        return len(self.key_cache)


class TestModule3Offline(unittest.TestCase):
    def test_cow_fork_and_shared_prefix_cache_zero_copy(self) -> None:
        SharedPrefixCache.clear_memo()
        base = _MockCache(4)
        shared_k = _MockTensor(list(range(2048)))
        shared_v = _MockTensor(list(range(2048)))
        for l in range(4):
            base.key_cache[l] = shared_k
            base.value_cache[l] = shared_v
        ctx_main = {"kv_lens": [2048], "ropes": [2048], "past_key_values": base}

        # Zero-copy fork BEFORE appending turn prompt
        ctx_cfg_text = cow_fork_context(ctx_main)
        self.assertEqual(ctx_cfg_text["past_key_values"].key_cache[0].data_ptr(), shared_k.data_ptr())

        # Verify SharedPrefixCache detects exact O(1) prefix sharing when storage aliases
        ctx_cfg_img = _MockCache(4)
        img_k = _MockTensor(list(range(20)))
        img_v = _MockTensor(list(range(20)))
        for l in range(4):
            ctx_cfg_img.key_cache[l] = img_k
            ctx_cfg_img.value_cache[l] = img_v

        sp = SharedPrefixCache.from_branch_caches(
            [ctx_main["past_key_values"], ctx_cfg_text["past_key_values"], ctx_cfg_img],
            min_shared_tokens=32,
        )
        self.assertEqual(sp.shared_prefix_len, 2048)
        self.assertEqual(sp.shared_branch_ids, [0, 1])
        self.assertEqual(sp.unshared_branch_ids, [2])
        self.assertEqual(sp.unique_kv_tokens, 2048 + 20)
        self.assertEqual(sp.naive_kv_tokens, 2048 + 2048 + 20)
        self.assertEqual(sp.padded_naive_kv_tokens, 2048 * 3)
        self.assertGreater(sp.kv_reduction_ratio, 1.98)

    def test_lecp_rebind_and_empty_boundary_cases(self) -> None:
        SharedPrefixCache.clear_memo()
        empty_a = _MockCache(2)
        empty_b = _MockCache(2)
        self.assertEqual(detect_lecp_prefix_len(empty_a, empty_b), 0)
        self.assertEqual(rebind_cow_prefix(empty_a, empty_b), 0)

        # Simulate parent appending new suffix into a fresh contiguous buffer `[0..2099]`
        # while `cfg_text` still points to the old `[0..2047]` buffer; `rebind_cow_prefix` restores zero-copy aliasing.
        parent = _MockCache(2)
        child = _MockCache(2)
        parent_k = _MockTensor(list(range(2100)), base_id=11)
        parent_v = _MockTensor(list(range(2100)), base_id=12)
        old_child_k = _MockTensor(list(range(2048)), base_id=99)
        old_child_v = _MockTensor(list(range(2048)), base_id=98)
        for l in range(2):
            parent.key_cache[l] = parent_k
            parent.value_cache[l] = parent_v
            child.key_cache[l] = old_child_k
            child.value_cache[l] = old_child_v

        rebound = rebind_cow_prefix(parent, child, prefix_len=2048)
        self.assertEqual(rebound, 2048)
        self.assertEqual(child.key_cache[0].data_ptr(), parent.key_cache[0].data_ptr())
        self.assertEqual(detect_lecp_prefix_len(parent, child), 2048)
        # Mismatched target length is safely rejected
        self.assertEqual(rebind_cow_prefix(parent, child, prefix_len=1024), 0)

    def test_extract_target_hw_supports_dict_and_sizedict_object(self) -> None:
        class _SizeDict:
            height = 980
            width = 980

        self.assertEqual(_extract_target_hw({"height": 980, "width": 980}, 512, 512), (980, 980))
        self.assertEqual(_extract_target_hw(_SizeDict(), 512, 512), (980, 980))
        self.assertEqual(_extract_target_hw({"shortest_edge": 384}, 512, 512), (384, 384))
        self.assertEqual(_extract_target_hw(None, 512, 768), (512, 768))

    def test_pil_bicubic_resize_np_matches_pillow_catmull_rom_offline(self) -> None:
        import numpy as np
        from PIL import Image

        rng = np.random.default_rng(2026)
        for src_hw, dst_hw in (
            ((64, 64), (122, 122)),
            ((122, 122), (37, 53)),
            ((512, 512), (980, 980)),
            ((500, 500), (496, 496)),
            ((48, 48), (48, 48)),
            ((2, 2), (7, 5)),
        ):
            arr = rng.integers(0, 256, size=(src_hw[0], src_hw[1], 3), dtype=np.uint8)
            pil_out = np.asarray(Image.fromarray(arr).resize((dst_hw[1], dst_hw[0]), Image.BICUBIC), dtype=np.int16)
            chw_in = arr.transpose(2, 0, 1).astype(np.float32)
            np_out = pil_bicubic_resize_np(chw_in, dst_hw[0], dst_hw[1]).astype(np.int16).transpose(1, 2, 0)
            max_diff = int(np.abs(np_out - pil_out).max())
            self.assertEqual(max_diff, 0, msg=f"src={src_hw} dst={dst_hw} max_diff={max_diff}")

    def test_pil_bicubic_resize_matches_pillow_catmull_rom(self) -> None:
        try:
            import numpy as np
            from PIL import Image
            import torch
        except ImportError:
            self.skipTest("torch/PIL not available in offline WSL environment")

        rng = np.random.default_rng(2026)
        for src_hw, dst_hw in (((64, 64), (122, 122)), ((512, 512), (980, 980)), ((500, 500), (496, 496))):
            arr = rng.integers(0, 256, size=(src_hw[0], src_hw[1], 3), dtype=np.uint8)
            pil_out = np.asarray(Image.fromarray(arr).resize((dst_hw[1], dst_hw[0]), Image.BICUBIC), dtype=np.int16)
            gpu_in = torch.from_numpy(arr.transpose(2, 0, 1).copy()).float()
            gpu_out = pil_bicubic_resize_gpu(gpu_in, dst_hw[0], dst_hw[1]).cpu().numpy().astype(np.int16).transpose(1, 2, 0)
            max_diff = int(np.abs(gpu_out - pil_out).max())
            self.assertEqual(max_diff, 0, msg=f"src={src_hw} dst={dst_hw} max_diff={max_diff}")

    def test_tv_bicubic_u8_resize_and_vae_norm_bit_exact_to_cpu_torchvision(self) -> None:
        try:
            import numpy as np
            import torch
            from torchvision.transforms import InterpolationMode
            from torchvision.transforms.v2 import functional as F_v2
        except ImportError:
            self.skipTest("torch/torchvision not available in offline WSL environment")

        # 1. Exact 0-ULP VAE normalization across all 256 uint8 values
        u = torch.arange(256, dtype=torch.uint8)
        cpu_vae = u.float() / 127.5 - 1.0
        gpu_vae = (u.to(torch.float64) / 127.5).to(torch.float32) - 1.0
        self.assertTrue(torch.equal(cpu_vae, gpu_vae))

        # 2. Exact 0-pixel-diff `UpSampleKernel.cpp` `int16_t` bicubic antialias resize across upsample & downsample
        rng = np.random.default_rng(2027)
        for src_hw in ((512, 512), (1024, 1024), (512, 768), (2048, 2048)):
            arr = rng.integers(0, 256, size=(3, src_hw[0], src_hw[1]), dtype=np.uint8)
            t_in = torch.from_numpy(arr)
            ref = F_v2.resize(t_in, [980, 980], interpolation=InterpolationMode.BICUBIC, antialias=True)
            cand = tv_bicubic_u8_resize_gpu(t_in, 980, 980)
            self.assertTrue(torch.equal(ref, cand), msg=f"src={src_hw} nz={int((ref != cand).sum().item())}")

    def test_flash_prefix_split_arithmetic_intensity_gate(self) -> None:
        self.assertTrue(should_use_flash_prefix_split(1026, 2048, 2))
        self.assertTrue(should_use_flash_prefix_split(1026, 16384, 2))
        self.assertFalse(should_use_flash_prefix_split(4098, 8192, 2))
        self.assertTrue(should_use_flash_prefix_split(4098, 16392, 2))
        self.assertFalse(should_use_flash_prefix_split(1026, 128, 2))

    def test_converged_cfg_prefix_attn_dead_single_pass_kernels_removed(self) -> None:
        self.assertFalse(hasattr(cfg_prefix_attn, "_shared_prefix_cfg2_attn_fwd"))
        self.assertFalse(hasattr(cfg_prefix_attn, "_shared_prefix_cfg3_attn_fwd"))
        src_text = Path(cfg_prefix_attn.__file__).read_text(encoding="utf-8")
        self.assertNotIn("_shared_prefix_cfg2_attn_fwd", src_text)
        self.assertNotIn("_shared_prefix_cfg3_attn_fwd", src_text)
        self.assertIn("def _merge_two_lse_fwd(", src_text)
        self.assertIn("def _run_flash_prefix_cfg(", src_text)

        # Exact boundary conditions on `should_use_cascade_cfg_split`
        self.assertFalse(should_use_cascade_cfg_split(0, 4096, 2))
        self.assertFalse(should_use_cascade_cfg_split(-4, 4096, 2))
        self.assertFalse(should_use_cascade_cfg_split(1026, 0, 2))
        self.assertFalse(should_use_cascade_cfg_split(1026, 4096, 1))
        self.assertTrue(should_use_cascade_cfg_split(1026, 1539, 2))
        self.assertFalse(should_use_cascade_cfg_split(1026, 1538, 2))
        self.assertFalse(should_use_cascade_cfg_split(4098, 16391, 2))
        self.assertTrue(should_use_cascade_cfg_split(4098, 16392, 2))

        # Verify `BagelSharedPrefixPatch.uninstall()` flushes workspace caches cleanly
        class _DummyAttn:
            def _forward_gen(self, *a, **kw):
                return a, kw

        class _DummyLayer:
            def __init__(self) -> None:
                self.self_attn = _DummyAttn()

        class _DummyPipe:
            class bagel:
                class language_model:
                    class model:
                        layers = [_DummyLayer()]

        cfg_prefix_attn._DIRECT_WS_CACHE[("mock:0", "bf16")] = {"dummy": 1}
        cfg_prefix_attn._CU_SEQLENS_CACHE[("mock:0", (10,))] = [0, 10]
        patch = cfg_prefix_attn.BagelSharedPrefixPatch(_DummyPipe())
        patch._installed = True
        patch._orig_forward_gens = [(_DummyPipe.bagel.language_model.model.layers[0].self_attn, lambda *a, **kw: None)]
        patch.uninstall()
        self.assertEqual(len(cfg_prefix_attn._DIRECT_WS_CACHE), 0)
        self.assertEqual(len(cfg_prefix_attn._CU_SEQLENS_CACHE), 0)

    def test_validate_module3_chat_strip_and_whole_word_rubric(self) -> None:
        from morphserve.bench.exactness import score_mmvet_answer
        from morphserve.bench.validate_module3_cfg_loopback import (
            _extract_cot_rubric_clause,
            _strip_chat_markers,
        )

        raw = "<|im_start|><think>Observing the vivid blue fire hydrant, centered in foreground.</think><|im_end|>"
        self.assertEqual(
            _strip_chat_markers(raw),
            "<think>Observing the vivid blue fire hydrant, centered in foreground.</think>",
        )
        clause = _extract_cot_rubric_clause(raw)
        self.assertEqual(clause, "Observing<AND>the<AND>vivid<AND>blue")
        self.assertEqual(score_mmvet_answer(raw, clause), 1.0)

    def test_module3_script_syntax_and_imports(self) -> None:
        script = REPO_ROOT / "deploy/kairos/scripts/05_phase1_module3_validate.sh"
        proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        for rel in (
            "python/morphserve/kernels/__init__.py",
            "python/morphserve/kernels/lecp_cache.py",
            "python/morphserve/kernels/cfg_prefix_attn.py",
            "python/morphserve/kernels/gpu_loopback.py",
            "python/morphserve/bench/exactness.py",
            "python/morphserve/bench/validate_module3_cfg_loopback.py",
        ):
            p = REPO_ROOT / rel
            res = subprocess.run([sys.executable, "-m", "py_compile", str(p)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, msg=f"py_compile failed on {rel}: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
