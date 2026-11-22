"""Offline unit tests for Module 3 (LECP CoW Cache, SharedPrefixCache, and shell syntax)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from morphserve.kernels import SharedPrefixCache, cow_fork_context, should_use_flash_prefix_split  # noqa: E402
from morphserve.kernels.gpu_loopback import _extract_target_hw, pil_bicubic_resize_gpu  # noqa: E402


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

    def test_extract_target_hw_supports_dict_and_sizedict_object(self) -> None:
        class _SizeDict:
            height = 980
            width = 980

        self.assertEqual(_extract_target_hw({"height": 980, "width": 980}, 512, 512), (980, 980))
        self.assertEqual(_extract_target_hw(_SizeDict(), 512, 512), (980, 980))

    def test_pil_bicubic_resize_matches_pillow_catmull_rom(self) -> None:
        try:
            import numpy as np
            from PIL import Image
            import torch
        except ImportError:
            self.skipTest("torch/PIL not available in offline WSL environment")

        rng = np.random.default_rng(2026)
        arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        pil_out = np.asarray(Image.fromarray(arr).resize((122, 122), Image.BICUBIC), dtype=np.int16)
        gpu_in = torch.from_numpy(arr.transpose(2, 0, 1).copy()).float()
        gpu_out = pil_bicubic_resize_gpu(gpu_in, 122, 122).cpu().numpy().astype(np.int16).transpose(1, 2, 0)
        max_diff = int(np.abs(gpu_out - pil_out).max())
        self.assertLessEqual(max_diff, 1)

    def test_flash_prefix_split_arithmetic_intensity_gate(self) -> None:
        self.assertTrue(should_use_flash_prefix_split(1026, 2048, 2))
        self.assertTrue(should_use_flash_prefix_split(1026, 16384, 2))
        self.assertFalse(should_use_flash_prefix_split(4098, 8192, 2))
        self.assertTrue(should_use_flash_prefix_split(4098, 16392, 2))
        self.assertFalse(should_use_flash_prefix_split(1026, 128, 2))

    def test_module3_script_syntax_and_imports(self) -> None:
        script = REPO_ROOT / "deploy/kairos/scripts/05_phase1_module3_validate.sh"
        proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        for rel in (
            "src/morphserve/kernels/__init__.py",
            "src/morphserve/kernels/lecp_cache.py",
            "src/morphserve/kernels/cfg_prefix_attn.py",
            "src/morphserve/kernels/gpu_loopback.py",
            "src/morphserve/bench/validate_module3_cfg_loopback.py",
        ):
            p = REPO_ROOT / rel
            res = subprocess.run([sys.executable, "-m", "py_compile", str(p)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, msg=f"py_compile failed on {rel}: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
