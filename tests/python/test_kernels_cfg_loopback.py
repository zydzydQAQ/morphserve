"""Public-seam TDD tests for `LECP CoW`, `FlashPrefix-CFG`, and `GpuInPlaceLoopback`."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from morphserve.kernels import (
    GpuInPlaceLoopback,
    SharedPrefixCache,
    cow_fork_context,
    decode_latent_to_u8_gpu,
    detect_lecp_prefix_len,
    pil_bicubic_resize_gpu,
    pil_bicubic_resize_np,
    prepare_vae_images_gpu,
    rebind_cow_prefix,
    shared_prefix_cfg_attention,
    shared_prefix_cfg_attention_ref,
    should_use_cascade_cfg_split,
    should_use_flash_prefix_split,
)
from morphserve.vmm import VmmNaiveCache


@pytest.mark.parametrize(
    "q_len,l_shared,num_shared_branches,expected",
    [
        # Invalid / degenerate inputs
        (0, 1024, 2, False),
        (128, 512, 1, False),
        (100, 255, 2, False),
        # Q <= 2048 regime: L_shared >= max(256, int(1.5 * Q))
        (100, 256, 2, True),
        (1024, 1535, 2, False),
        (1024, 1536, 2, True),
        (2048, 3071, 2, False),
        (2048, 3072, 2, True),
        # Q > 2048 regime: L_shared >= 4 * Q
        (2049, 8195, 2, False),
        (2049, 8196, 2, True),
        (4096, 16383, 3, False),
        (4096, 16384, 3, True),
    ],
)
def test_flash_prefix_split_threshold_boundaries(
    q_len: int,
    l_shared: int,
    num_shared_branches: int,
    expected: bool,
) -> None:
    assert should_use_flash_prefix_split is should_use_cascade_cfg_split
    assert should_use_flash_prefix_split(q_len, l_shared, num_shared_branches) is expected


@pytest.mark.parametrize("target_hw", [(384, 384), (256, 384), (512, 512)])
def test_pil_bicubic_resize_np_bit_exact_to_pillow(target_hw: tuple[int, int]) -> None:
    PIL_Image = pytest.importorskip("PIL.Image")

    rng = np.random.default_rng(2026)
    img_hwc = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    target_h, target_w = target_hw

    pil_ref = np.asarray(
        PIL_Image.fromarray(img_hwc, mode="RGB").resize((target_w, target_h), PIL_Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )
    img_chw_f32 = np.transpose(img_hwc, (2, 0, 1)).astype(np.float32)
    got_chw = pil_bicubic_resize_np(img_chw_f32, target_h, target_w)
    got_hwc = np.transpose(got_chw, (1, 2, 0)).astype(np.uint8)

    assert got_hwc.shape == pil_ref.shape
    assert int(np.max(np.abs(got_hwc.astype(np.int32) - pil_ref.astype(np.int32)))) == 0


def test_cow_fork_rebind_and_shared_prefix_memo_clearing() -> None:
    torch = pytest.importorskip("torch")

    torch.manual_seed(2026)
    num_layers = 2
    main_cache = VmmNaiveCache(num_layers=num_layers, vmm=None)
    for lidx in range(num_layers):
        main_cache.key_cache[lidx] = torch.randn(64, 4, 32, dtype=torch.float32)
        main_cache.value_cache[lidx] = torch.randn(64, 4, 32, dtype=torch.float32)

    ctx = {"past_key_values": main_cache, "kv_lens": [64], "ropes": [1]}
    forked = cow_fork_context(ctx, prefix_len=48, cache_factory=lambda n: VmmNaiveCache(n, vmm=None))
    cfg_text = forked["past_key_values"]
    assert forked["kv_lens"] == [48]
    assert cfg_text.key_cache[0].data_ptr() == main_cache.key_cache[0].data_ptr()
    assert detect_lecp_prefix_len(main_cache, cfg_text) == 48

    # Simulate main cache reallocation on token append (`torch.cat`), breaking pointer identity
    for lidx in range(num_layers):
        extra_k = torch.randn(16, 4, 32, dtype=torch.float32)
        extra_v = torch.randn(16, 4, 32, dtype=torch.float32)
        main_cache.key_cache[lidx] = torch.cat([main_cache.key_cache[lidx], extra_k], dim=0)
        main_cache.value_cache[lidx] = torch.cat([main_cache.value_cache[lidx], extra_v], dim=0)
    assert cfg_text.key_cache[0].data_ptr() != main_cache.key_cache[0].data_ptr()

    # `rebind_cow_prefix` restores O(1) pointer aliasing to `main_cache[:48]`
    rebound = rebind_cow_prefix(main_cache, cfg_text, prefix_len=48)
    assert rebound == 48
    for lidx in range(num_layers):
        assert cfg_text.key_cache[lidx].data_ptr() == main_cache.key_cache[lidx].data_ptr()
        assert cfg_text.value_cache[lidx].data_ptr() == main_cache.value_cache[lidx].data_ptr()
        assert torch.equal(cfg_text.key_cache[lidx], main_cache.key_cache[lidx][:48])

    # Unshared third branch (`cfg_img`) with distinct content
    cfg_img = VmmNaiveCache(num_layers=num_layers, vmm=None)
    for lidx in range(num_layers):
        cfg_img.key_cache[lidx] = torch.randn(20, 4, 32, dtype=torch.float32)
        cfg_img.value_cache[lidx] = torch.randn(20, 4, 32, dtype=torch.float32)

    SharedPrefixCache.clear_memo()
    sp = SharedPrefixCache.from_branch_caches([main_cache, cfg_text, cfg_img], min_shared_tokens=32)
    assert sp.shared_prefix_len == 48
    assert sp.shared_branch_ids == [0, 1]
    assert sp.unshared_branch_ids == [2]
    assert sp.unique_kv_tokens == 48 + 32 + 0 + 20
    assert sp.naive_kv_tokens == 80 + 48 + 20
    assert sp.kv_reduction_ratio == pytest.approx(148.0 / 100.0)

    # Mutate `cfg_text` storage in-place after cloning it away from `main_cache` and verify `clear_memo()` recomputes
    for lidx in range(num_layers):
        cfg_text.key_cache[lidx] = cfg_text.key_cache[lidx].clone()
        cfg_text.value_cache[lidx] = cfg_text.value_cache[lidx].clone()
    SharedPrefixCache.clear_memo()
    sp_after_clone = SharedPrefixCache.from_branch_caches([main_cache, cfg_text, cfg_img], min_shared_tokens=32)
    assert sp_after_clone.shared_prefix_len == 48

    # Diverge token 10 in-place -> after `clear_memo()`, LECP must drop to 10 (< 32 -> unshared)
    cfg_text.key_cache[0][10].add_(100.0)
    SharedPrefixCache.clear_memo()
    sp_diverged = SharedPrefixCache.from_branch_caches([main_cache, cfg_text, cfg_img], min_shared_tokens=32)
    assert sp_diverged.shared_prefix_len == 0
    assert sp_diverged.shared_branch_ids == []


def test_shared_prefix_cfg_attention_parity_and_explicit_workspace_cache() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(2026)

    q_len, h_q, h_kv, head_dim = 24, 8, 4, 64
    sm_scale = 1.0 / math.sqrt(head_dim)
    k_prefix = torch.randn(48, h_kv, head_dim, device=device, dtype=dtype)
    v_prefix = torch.randn(48, h_kv, head_dim, device=device, dtype=dtype)
    q_list = [torch.randn(q_len, h_q, head_dim, device=device, dtype=dtype) for _ in range(3)]
    k_loc_list = [torch.randn(16, h_kv, head_dim, device=device, dtype=dtype) for _ in range(3)]
    v_loc_list = [torch.randn(16, h_kv, head_dim, device=device, dtype=dtype) for _ in range(3)]
    k_suf_list = [
        torch.randn(12, h_kv, head_dim, device=device, dtype=dtype),
        None,
        torch.randn(10, h_kv, head_dim, device=device, dtype=dtype),
    ]
    v_suf_list = [
        torch.randn(12, h_kv, head_dim, device=device, dtype=dtype),
        None,
        torch.randn(10, h_kv, head_dim, device=device, dtype=dtype),
    ]
    shared_branch_ids = [0, 1]

    def sdpa_noncausal(q_4d: Any, k_4d: Any, v_4d: Any, meta: Any) -> Any:
        # `q_4d`: [B, Q, H_q, D], `k_4d`/`v_4d`: [B, K, H_kv, D]
        attn_mask = None
        if meta is not None and getattr(meta, "attn_mask", None) is not None:
            attn_mask = meta.attn_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q_4d.permute(0, 2, 1, 3),
            k_4d.permute(0, 2, 1, 3),
            v_4d.permute(0, 2, 1, 3),
            attn_mask=attn_mask,
            is_causal=False,
            scale=sm_scale,
            enable_gqa=(h_q != h_kv),
        )
        return out.permute(0, 2, 1, 3).contiguous()

    ref_outs = shared_prefix_cfg_attention_ref(
        q_list=q_list,
        k_loc_list=k_loc_list,
        v_loc_list=v_loc_list,
        k_prefix=k_prefix,
        v_prefix=v_prefix,
        shared_branch_ids=shared_branch_ids,
        k_suffix_list=k_suf_list,
        v_suffix_list=v_suf_list,
        sm_scale=sm_scale,
        fp32_accum=(dtype == torch.float32),
    )

    ws_cache: dict[tuple[str, str], dict[str, Any]] = {}
    meta_cache: dict[tuple[str, tuple[int, ...]], Any] = {}
    cu_cache: dict[tuple[str, tuple[int, ...]], Any] = {}

    got_1 = shared_prefix_cfg_attention(
        q_list=q_list,
        k_loc_list=k_loc_list,
        v_loc_list=v_loc_list,
        k_prefix=k_prefix,
        v_prefix=v_prefix,
        shared_branch_ids=shared_branch_ids,
        k_suffix_list=k_suf_list,
        v_suffix_list=v_suf_list,
        sm_scale=sm_scale,
        attn_noncausal=sdpa_noncausal,
        backend="lossless_varlen",
        direct_ws_cache=ws_cache,
        cu_seqlens_cache=cu_cache,
        attn_meta_cache=meta_cache,
    )
    atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    for b in range(3):
        assert torch.allclose(got_1[b].float(), ref_outs[b].float(), atol=atol, rtol=1e-4)

    if device.type == "cuda":
        assert len(ws_cache) == 1
        entry_1 = next(iter(ws_cache.values()))
        k_ptr_1 = int(entry_1["k"].data_ptr())
        q_ptr_1 = int(entry_1["q"].data_ptr())

        got_2 = shared_prefix_cfg_attention(
            q_list=q_list,
            k_loc_list=k_loc_list,
            v_loc_list=v_loc_list,
            k_prefix=k_prefix,
            v_prefix=v_prefix,
            shared_branch_ids=shared_branch_ids,
            k_suffix_list=k_suf_list,
            v_suffix_list=v_suf_list,
            sm_scale=sm_scale,
            attn_noncausal=sdpa_noncausal,
            backend="lossless_varlen",
            direct_ws_cache=ws_cache,
            cu_seqlens_cache=cu_cache,
            attn_meta_cache=meta_cache,
        )
        entry_2 = next(iter(ws_cache.values()))
        assert int(entry_2["k"].data_ptr()) == k_ptr_1
        assert int(entry_2["q"].data_ptr()) == q_ptr_1
        for b in range(3):
            assert torch.allclose(got_2[b].float(), ref_outs[b].float(), atol=atol, rtol=1e-4)


def test_gpu_loopback_bicubic_vae_norm_and_context_update_parity() -> None:
    torch = pytest.importorskip("torch")
    PIL_Image = pytest.importorskip("PIL.Image")

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    rng = np.random.default_rng(2026)
    img_hwc = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)

    # 1. Bit-exact Pillow `Image.BICUBIC` resize parity in PyTorch
    pil_ref = np.asarray(
        PIL_Image.fromarray(img_hwc, mode="RGB").resize((384, 384), PIL_Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )
    img_chw_u8 = torch.from_numpy(np.transpose(img_hwc, (2, 0, 1)).copy()).to(device=device)
    resized_gpu = pil_bicubic_resize_gpu(img_chw_u8.to(torch.float32), 384, 384)
    got_hwc = resized_gpu.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    assert int(np.max(np.abs(got_hwc.astype(np.int32) - pil_ref.astype(np.int32)))) == 0

    # 2. Bit-exact 0-ULP VAE normalization across all 256 uint8 pixel values
    ramp = torch.arange(256, dtype=torch.uint8, device=device).repeat(3, 256, 1)  # [3, 256, 256]

    class _MinimalBagel:
        latent_downsample = 16
        latent_patch_size = 2
        latent_channel = 4
        max_latent_size = 64
        vit_patch_size = 14
        vit_max_num_patch_per_side = 70

        @staticmethod
        def get_flattened_position_ids(h: int, w: int, stride: int, max_num_patches_per_side: int) -> Any:
            return torch.arange((h // stride) * (w // stride), dtype=torch.long)

        @staticmethod
        def forward_cache_update_vae(_vae: Any, pkv: Any, **kwargs: Any) -> Any:
            pkv["vae_tokens"] = int(kwargs["packed_vae_token_indexes"].numel())
            return pkv

        @staticmethod
        def forward_cache_update_vit(pkv: Any, **kwargs: Any) -> Any:
            pkv["vit_tokens"] = int(kwargs["packed_vit_token_indexes"].numel())
            return pkv

    class _MinimalVae:
        def __init__(self) -> None:
            self._p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))
            self.last_lat_shape: tuple[int, ...] | None = None

        def parameters(self):
            yield self._p

        def decode(self, lat: Any) -> Any:
            self.last_lat_shape = tuple(int(d) for d in lat.shape)
            # Include out-of-range values (-1.5, +1.5) and exact boundary values (-1.0, 0.0, +1.0)
            out = torch.zeros((1, 3, 32, 32), device=lat.device, dtype=lat.dtype)
            out[0, :, 0, 0] = -1.5
            out[0, :, 0, 1] = -1.0
            out[0, :, 0, 2] = 0.0
            out[0, :, 0, 3] = 1.0
            out[0, :, 0, 4] = 1.5
            return out

    bagel = _MinimalBagel()
    new_token_ids = {"start_of_image": 100, "end_of_image": 101}
    vae_inp, new_lens, new_rope = prepare_vae_images_gpu(bagel, [ramp], [10], [2], new_token_ids)
    # Compare against independent CPU NumPy/float32 reference (`arr.astype(float32) / 127.5 - 1.0`)
    cpu_ref = torch.from_numpy(ramp.cpu().numpy().astype(np.float32) / np.float32(127.5) - np.float32(1.0)).to(device)
    assert torch.equal(vae_inp["padded_images"][0], cpu_ref)
    assert new_lens == [10 + (256 // 16) * (256 // 16) + 2]
    assert new_rope == [3]

    # 3. `decode_latent_to_u8_gpu` patch un-packing shape & clamping parity
    vae = _MinimalVae()
    dummy_latent = torch.zeros((4, 16), dtype=torch.float32, device=device)  # (32/16)*(32/16)=4, 2*2*4=16
    u8_out = decode_latent_to_u8_gpu(bagel, vae, dummy_latent, (32, 32))
    assert vae.last_lat_shape == (1, 4, 4, 4)
    assert tuple(u8_out.shape) == (3, 32, 32)
    assert u8_out.dtype == torch.uint8
    assert int(u8_out[0, 0, 0].item()) == 0
    assert int(u8_out[0, 0, 1].item()) == 0
    assert int(u8_out[0, 0, 2].item()) == 127
    assert int(u8_out[0, 0, 3].item()) == 255
    assert int(u8_out[0, 0, 4].item()) == 255

    # 4. `GpuInPlaceLoopback` end-to-end context update (VAE + ViT)
    img_proc = SimpleNamespace(
        do_resize=True,
        size={"height": 224, "width": 224},
        resample=3,
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )
    pipeline = SimpleNamespace(
        bagel=bagel,
        vae=vae,
        image_processor=img_proc,
        new_token_ids=new_token_ids,
        device=device,
        od_config=SimpleNamespace(dtype=torch.float32),
    )
    loopback = GpuInPlaceLoopback(pipeline)
    ctx: dict[str, Any] = {"past_key_values": {}, "kv_lens": [10], "ropes": [1]}
    updated = loopback.update_context_from_u8_gpu(ctx, ramp, update_vae=True, update_vit=True)
    expected_vae_toks = (256 // 16) * (256 // 16)
    expected_vit_toks = (224 // 14) * (224 // 14)
    assert updated["past_key_values"]["vae_tokens"] == expected_vae_toks
    assert updated["past_key_values"]["vit_tokens"] == expected_vit_toks
    assert updated["kv_lens"] == [10 + (expected_vae_toks + 2) + (expected_vit_toks + 2)]
    assert updated["ropes"] == [3]
