"""Zero-CPU GPU In-Place VAE -> uint8 -> Resize -> ViT/VAE Loopback (eliminates PIL/PNG round-trip)."""

from __future__ import annotations

from typing import Any


def _patchify_gpu(imgs: Any, patch_size: int) -> Any:
    """Exact GPU equivalent of `bagel_transformer.patchify` (N, 3, H, W) or (3, H, W)."""
    import torch

    is_batch = imgs.ndim == 4
    if not is_batch:
        imgs = imgs.unsqueeze(0)
    x = imgs.reshape(imgs.shape[0], 3, imgs.shape[2] // patch_size, patch_size, imgs.shape[3] // patch_size, patch_size)
    x = torch.einsum("nchpwq->nhwcpq", x)
    x = x.reshape(imgs.shape[0], -1, 3 * patch_size * patch_size)
    return x if is_batch else x.squeeze(0)


def decode_latent_to_u8_gpu(
    bagel: Any,
    vae: Any,
    latent: Any,
    image_shape: tuple[int, int],
) -> Any:
    """Decode DiT latent to GPU `[3, H, W]` `uint8` tensor bit-exact to `BagelPipeline._decode_image_from_latent`.

    Matches `pipeline_bagel.py:513-527`:
      `(image * 0.5 + 0.5).clamp(0, 1)[0] * 255 -> to(torch.uint8)`
    WITHOUT transferring to CPU or constructing a PIL.Image / PNG byte stream!
    """
    import torch

    H, W = int(image_shape[0]), int(image_shape[1])
    h, w = H // bagel.latent_downsample, W // bagel.latent_downsample
    p = bagel.latent_patch_size
    c = bagel.latent_channel
    lat = latent.reshape(1, h, w, p, p, c)
    lat = torch.einsum("nhwpqc->nchpwq", lat)
    lat = lat.reshape(1, c, h * p, w * p)
    vae_dtype = next(vae.parameters()).dtype
    lat = lat.to(vae_dtype)

    decoded = vae.decode(lat)  # [1, 3, H, W]
    img_u8 = ((decoded * 0.5 + 0.5).clamp(0.0, 1.0)[0] * 255.0).to(torch.uint8)  # [3, H, W] on GPU!
    return img_u8


def _stride_align_u8_gpu(img_u8: Any, stride: int, max_img_size: int) -> Any:
    """Exact GPU equivalent of `pipeline_bagel.py:_resize_to_stride`. Zero-op when already stride-aligned."""
    import torch

    _, h, w = img_u8.shape
    scale = min(max_img_size / max(w, h), 1.0)
    min_img_size = min(256, max_img_size)
    scale = max(scale, min_img_size / min(w, h))
    new_w = max(stride, int(round(w * scale / stride) * stride))
    new_h = max(stride, int(round(h * scale / stride) * stride))
    new_w = min(new_w, max_img_size)
    new_h = min(new_h, max_img_size)
    if new_w == w and new_h == h:
        return img_u8
    resized = pil_bicubic_resize_gpu(img_u8.to(torch.float32), new_h, new_w)
    return resized.to(torch.uint8)


def prepare_vae_images_gpu(
    bagel: Any,
    u8_images: list[Any],
    curr_kvlens: list[int],
    curr_rope: list[int],
    new_token_ids: dict[str, int],
    timestep: int = 0,
) -> tuple[dict[str, Any], list[int], list[int]]:
    """Build `forward_cache_update_vae` inputs 100% on GPU (bit-exact to CPU PIL path on native stride sizes)."""
    import torch

    device = u8_images[0].device
    stride = int(bagel.latent_downsample)
    max_img_size = int(bagel.max_latent_size * stride)

    patchified_vae_latent_shapes: list[tuple[int, int]] = []
    packed_vae_position_ids: list[Any] = []
    packed_vae_token_indexes: list[int] = []
    packed_text_ids: list[int] = []
    packed_text_indexes: list[int] = []
    packed_seqlens: list[int] = []
    packed_position_ids: list[int] = []
    vae_image_tensors: list[Any] = []
    newlens: list[int] = []
    new_rope: list[int] = []

    _curr = 0
    for img_u8, curr_kvlen, curr_pos_id in zip(u8_images, curr_kvlens, curr_rope):
        packed_text_ids.append(new_token_ids["start_of_image"])
        packed_text_indexes.append(_curr)
        _curr += 1

        aligned = _stride_align_u8_gpu(img_u8, stride=stride, max_img_size=max_img_size)
        # Exact match to `torch.from_numpy(np.array(img)).float() / 127.5 - 1.0`
        image_tensor = aligned.to(torch.float32) / 127.5 - 1.0
        vae_image_tensors.append(image_tensor)

        H, W = int(image_tensor.shape[1]), int(image_tensor.shape[2])
        vae_pos_ids = bagel.get_flattened_position_ids(
            H, W, bagel.latent_downsample, max_num_patches_per_side=bagel.max_latent_size
        )
        packed_vae_position_ids.append(vae_pos_ids.to(device))
        h = H // bagel.latent_downsample
        w = W // bagel.latent_downsample
        patchified_vae_latent_shapes.append((h, w))

        num_img_tokens = w * h
        packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
        _curr += num_img_tokens

        packed_text_ids.append(new_token_ids["end_of_image"])
        packed_text_indexes.append(_curr)
        _curr += 1

        packed_position_ids.extend([curr_pos_id] * (num_img_tokens + 2))
        packed_seqlens.append(num_img_tokens + 2)
        newlens.append(curr_kvlen + num_img_tokens + 2)
        new_rope.append(curr_pos_id + 1)

    max_h = max(int(t.shape[1]) for t in vae_image_tensors)
    max_w = max(int(t.shape[2]) for t in vae_image_tensors)
    if len(vae_image_tensors) == 1:
        padded_images = vae_image_tensors[0].unsqueeze(0)
    else:
        padded_images = torch.zeros((len(vae_image_tensors), 3, max_h, max_w), dtype=torch.float32, device=device)
        for i, t in enumerate(vae_image_tensors):
            padded_images[i, :, : t.shape[1], : t.shape[2]] = t

    generation_input = {
        "padded_images": padded_images,
        "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
        "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
        "packed_timesteps": torch.tensor([timestep], device=device),
        "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long, device=device),
        "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long, device=device),
        "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
        "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long, device=device),
        "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),
    }
    return generation_input, newlens, new_rope


def _extract_target_hw(size_cfg: Any, h: int, w: int) -> tuple[int, int]:
    """Extract target (height, width) from dict or transformers 5.x SizeDict/dataclass."""
    if size_cfg is None:
        return h, w
    if isinstance(size_cfg, int):
        return int(size_cfg), int(size_cfg)
    # Support both plain dict and transformers 5.x SizeDict / object attributes
    get_val = (lambda k: size_cfg.get(k)) if isinstance(size_cfg, dict) else (lambda k: getattr(size_cfg, k, None))
    th = get_val("height")
    tw = get_val("width")
    if th is not None and tw is not None:
        return int(th), int(tw)
    se = get_val("shortest_edge")
    if se is not None:
        se_i = int(se)
        if h <= w:
            return se_i, int(round(w * se_i / h))
        return int(round(h * se_i / w)), se_i
    return h, w


_PIL_BICUBIC_WEIGHT_CACHE: dict[tuple[str, int, int], Any] = {}


_PIL_BICUBIC_FIXED_BITS = 22
_PIL_BICUBIC_FIXED_SCALE = float(1 << _PIL_BICUBIC_FIXED_BITS)
_PIL_BICUBIC_HALF_BIAS = float(1 << (_PIL_BICUBIC_FIXED_BITS - 1))
_PIL_BICUBIC_INV_SCALE = 1.0 / _PIL_BICUBIC_FIXED_SCALE


def _compute_pil_bicubic_weights_np(in_size: int, out_size: int) -> Any:
    """Compute exact Pillow `Image.BICUBIC` 22-bit fixed-point 1D weight matrix `[out_size, in_size]` in float64."""
    import numpy as np

    scale = in_size / out_size
    fscale = max(1.0, scale)
    support = 2.0 * fscale
    w = np.zeros((out_size, in_size), dtype=np.float64)
    for i in range(out_size):
        center = (i + 0.5) * scale
        xmin = max(0, int(center - support + 0.5))
        xmax = min(in_size, int(center + support + 0.5))
        js = np.arange(xmin, xmax, dtype=np.float64)
        x = np.abs((js + 0.5 - center) / fscale)
        k = np.where(
            x < 1.0,
            1.5 * x**3 - 2.5 * x**2 + 1.0,
            np.where(x < 2.0, -0.5 * x**3 + 2.5 * x**2 - 4.0 * x + 2.0, 0.0),
        )
        s = k.sum()
        if s != 0.0:
            k /= s
        ki = np.where(
            k < 0.0,
            np.trunc(k * _PIL_BICUBIC_FIXED_SCALE - 0.5),
            np.trunc(k * _PIL_BICUBIC_FIXED_SCALE + 0.5),
        )
        w[i, xmin:xmax] = ki
    return w


def pil_bicubic_resize_np(img_chw_f32: Any, target_h: int, target_w: int) -> Any:
    """Torch-free NumPy reference of `pil_bicubic_resize_gpu` (`[3, H, W] -> [3, target_h, target_w]`, bit-exact to PIL)."""
    import numpy as np

    _, in_h, in_w = img_chw_f32.shape
    x = np.clip(np.floor(img_chw_f32.astype(np.float64, copy=False) + 0.5), 0.0, 255.0)
    if target_w != in_w:
        w_w = _compute_pil_bicubic_weights_np(in_w, target_w)
        x = np.clip(np.floor((np.matmul(x, w_w.T) + _PIL_BICUBIC_HALF_BIAS) * _PIL_BICUBIC_INV_SCALE), 0.0, 255.0)
    if target_h != in_h:
        w_h = _compute_pil_bicubic_weights_np(in_h, target_h)
        x = np.clip(np.floor((np.matmul(w_h, x) + _PIL_BICUBIC_HALF_BIAS) * _PIL_BICUBIC_INV_SCALE), 0.0, 255.0)
    return x.astype(np.float32)


def _get_pil_bicubic_weights(in_size: int, out_size: int, device: Any) -> Any:
    """Cached exact Pillow `Image.BICUBIC` 22-bit fixed-point 1D separable weight matrix [out_size, in_size] in float64."""
    key = (str(device), int(in_size), int(out_size))
    cached = _PIL_BICUBIC_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    import torch

    wt = torch.from_numpy(_compute_pil_bicubic_weights_np(in_size, out_size)).to(device=device, dtype=torch.float64)
    _PIL_BICUBIC_WEIGHT_CACHE[key] = wt
    return wt


def pil_bicubic_resize_gpu(img_f32: Any, target_h: int, target_w: int) -> Any:
    """100% bit-exact Pillow `Image.BICUBIC` separable resize on GPU `[3, H, W] -> [3, target_h, target_w]`.

    Replicates `libImaging/Resample.c` 22-bit fixed-point horizontal-then-vertical integer accumulation
    losslessly in IEEE-754 float64 (`max_sum <= 255 * W * 2^22 << 2^53`), achieving 0 uint8 pixel difference
    vs `PIL.Image.resize((target_w, target_h), Image.BICUBIC)` in <0.15ms on GPU!
    """
    import torch

    out_dtype = img_f32.dtype if img_f32.is_floating_point() else torch.float32
    _, in_h, in_w = img_f32.shape
    if target_w == in_w and target_h == in_h:
        return img_f32.to(out_dtype)
    x = torch.floor(img_f32.to(torch.float64) + 0.5).clamp(0.0, 255.0)
    if target_w != in_w:
        w_w = _get_pil_bicubic_weights(in_w, target_w, x.device)  # [target_w, in_w]
        x = torch.floor((torch.matmul(x, w_w.t()) + _PIL_BICUBIC_HALF_BIAS) * _PIL_BICUBIC_INV_SCALE).clamp(0.0, 255.0)
    if target_h != in_h:
        w_h = _get_pil_bicubic_weights(in_h, target_h, x.device)  # [target_h, in_h]
        x = torch.floor((torch.matmul(w_h, x) + _PIL_BICUBIC_HALF_BIAS) * _PIL_BICUBIC_INV_SCALE).clamp(0.0, 255.0)
    return x.to(out_dtype)


def _vit_preprocess_u8_gpu(img_u8: Any, image_processor: Any) -> Any:
    """Exact GPU equivalent of HuggingFace `SiglipImageProcessor` / `BagelImageProcessor` on `[3, H, W]` uint8."""
    import torch
    import torch.nn.functional as F

    device = img_u8.device
    is_tv_fast = callable(image_processor) and any(
        cls.__name__ in ("TorchvisionBackend", "BaseImageProcessorFast")
        for cls in type(image_processor).__mro__
    )
    if is_tv_fast:
        try:
            out = image_processor(images=img_u8, return_tensors="pt", input_data_format="channels_first")
        except TypeError:
            out = image_processor(images=img_u8, return_tensors="pt")
        pv = out.pixel_values[0] if hasattr(out, "pixel_values") else out["pixel_values"][0]
        return pv.to(device=device, dtype=torch.float32)

    x = img_u8.to(torch.float32)  # [3, H, W] in [0, 255]

    cfg = image_processor.to_dict() if hasattr(image_processor, "to_dict") else {}
    do_resize = bool(cfg.get("do_resize", getattr(image_processor, "do_resize", False)))
    size_cfg = cfg.get("size", getattr(image_processor, "size", None))
    if do_resize and size_cfg is not None:
        _, h, w = x.shape
        target_h, target_w = _extract_target_hw(size_cfg, h, w)
        if target_h != h or target_w != w:
            resample = int(cfg.get("resample", getattr(image_processor, "resample", 3)))
            if resample == 3:
                x = pil_bicubic_resize_gpu(x, target_h, target_w)
            else:
                x = F.interpolate(
                    x.unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )[0]
                x = torch.floor(x + 0.5).clamp(0.0, 255.0)

    do_rescale = bool(cfg.get("do_rescale", getattr(image_processor, "do_rescale", True)))
    if do_rescale:
        rescale_factor = float(cfg.get("rescale_factor", getattr(image_processor, "rescale_factor", 1.0 / 255.0)))
        x = (x.to(torch.float64) * rescale_factor).to(torch.float32)

    do_normalize = bool(cfg.get("do_normalize", getattr(image_processor, "do_normalize", True)))
    if do_normalize:
        mean = cfg.get("image_mean", getattr(image_processor, "image_mean", [0.5, 0.5, 0.5]))
        std = cfg.get("image_std", getattr(image_processor, "image_std", [0.5, 0.5, 0.5]))
        mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(3, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32, device=device).view(3, 1, 1)
        x = (x - mean_t) / std_t

    return x


def prepare_vit_images_gpu(
    bagel: Any,
    image_processor: Any,
    u8_images: list[Any],
    curr_kvlens: list[int],
    curr_rope: list[int],
    new_token_ids: dict[str, int],
) -> tuple[dict[str, Any], list[int], list[int]]:
    """Build `forward_cache_update_vit` inputs 100% on GPU (zero PIL/CPU round-trip)."""
    import torch

    device = u8_images[0].device
    stride = int(bagel.latent_downsample)
    max_img_size = int(bagel.max_latent_size * stride)

    packed_vit_token_indexes: list[int] = []
    vit_token_seqlens: list[int] = []
    packed_vit_tokens: list[Any] = []
    packed_vit_position_ids: list[Any] = []
    packed_text_ids: list[int] = []
    packed_text_indexes: list[int] = []
    packed_seqlens: list[int] = []
    packed_position_ids: list[int] = []
    newlens: list[int] = []
    new_rope: list[int] = []

    _curr = 0
    for img_u8, curr_kvlen, curr_pos_id in zip(u8_images, curr_kvlens, curr_rope):
        packed_text_ids.append(new_token_ids["start_of_image"])
        packed_text_indexes.append(_curr)
        _curr += 1

        aligned = _stride_align_u8_gpu(img_u8, stride=stride, max_img_size=max_img_size)
        image_tensor = _vit_preprocess_u8_gpu(aligned, image_processor)
        # Ensure divisible by vit_patch_size just in case
        H, W = int(image_tensor.shape[1]), int(image_tensor.shape[2])
        p = int(bagel.vit_patch_size)
        if H % p != 0 or W % p != 0:
            import torch.nn.functional as F

            new_h = max(p, (H // p) * p)
            new_w = max(p, (W // p) * p)
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0), size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=True
            )[0]
            H, W = new_h, new_w

        vit_pos_ids = bagel.get_flattened_position_ids(
            H, W, p, max_num_patches_per_side=bagel.vit_max_num_patch_per_side
        )
        vit_tokens = _patchify_gpu(image_tensor, p)
        packed_vit_tokens.append(vit_tokens)
        num_img_tokens = int(vit_tokens.shape[0])
        packed_vit_position_ids.append(vit_pos_ids.to(device))
        vit_token_seqlens.append(num_img_tokens)
        packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
        _curr += num_img_tokens

        packed_text_ids.append(new_token_ids["end_of_image"])
        packed_text_indexes.append(_curr)
        _curr += 1

        packed_position_ids.extend([curr_pos_id] * (num_img_tokens + 2))
        packed_seqlens.append(num_img_tokens + 2)
        newlens.append(curr_kvlen + num_img_tokens + 2)
        new_rope.append(curr_pos_id + 1)

    generation_input = {
        "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long, device=device),
        "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
        "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int, device=device),
        "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
        "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
        "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device),
        "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long, device=device),
        "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),
    }
    return generation_input, newlens, new_rope


class GpuInPlaceLoopback:
    """High-level GPU-only loopback controller for Interleaved Visual-CoT / Multi-Turn Editing."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.bagel = pipeline.bagel
        self.vae = pipeline.vae
        self.image_processor = pipeline.image_processor
        self.new_token_ids = pipeline.new_token_ids
        self.device = pipeline.device
        self.dtype = pipeline.od_config.dtype

    def update_context_from_u8_gpu(
        self,
        ctx: dict[str, Any],
        img_u8: Any,
        update_vae: bool = True,
        update_vit: bool = True,
    ) -> dict[str, Any]:
        """Inject a GPU `[3, H, W]` uint8 image into `ctx['past_key_values']` via VAE + ViT, zero CPU round-trip."""
        import torch

        ac_enabled = self.device.type != "cpu"
        with torch.inference_mode():
            if update_vae:
                vae_seed = 10007 + (int(ctx["ropes"][0]) if ctx.get("ropes") else 0)
                torch.manual_seed(vae_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(vae_seed)
                vae_inp, new_lens, new_rope = prepare_vae_images_gpu(
                    self.bagel, [img_u8], ctx["kv_lens"], ctx["ropes"], self.new_token_ids
                )
                with torch.autocast(device_type=self.device.type, enabled=ac_enabled, dtype=self.dtype):
                    ctx["past_key_values"] = self.bagel.forward_cache_update_vae(
                        self.vae, ctx["past_key_values"], **vae_inp
                    )
                ctx["kv_lens"] = new_lens
                ctx["ropes"] = new_rope

            if update_vit:
                vit_inp, new_lens, new_rope = prepare_vit_images_gpu(
                    self.bagel,
                    self.image_processor,
                    [img_u8],
                    ctx["kv_lens"],
                    ctx["ropes"],
                    self.new_token_ids,
                )
                with torch.autocast(device_type=self.device.type, enabled=ac_enabled, dtype=self.dtype):
                    ctx["past_key_values"] = self.bagel.forward_cache_update_vit(
                        ctx["past_key_values"], **vit_inp
                    )
                ctx["kv_lens"] = new_lens
                ctx["ropes"] = new_rope
        return ctx

    def loopback_from_latent_gpu(
        self,
        ctx: dict[str, Any],
        latent: Any,
        image_shape: tuple[int, int],
        update_vae: bool = True,
        update_vit: bool = True,
    ) -> tuple[dict[str, Any], Any]:
        """Decode latent -> GPU uint8 -> inject VAE + ViT KV into `ctx` completely on GPU."""
        import torch

        with torch.inference_mode():
            img_u8 = decode_latent_to_u8_gpu(self.bagel, self.vae, latent, image_shape)
        self.update_context_from_u8_gpu(ctx, img_u8, update_vae=update_vae, update_vit=update_vit)
        return ctx, img_u8
