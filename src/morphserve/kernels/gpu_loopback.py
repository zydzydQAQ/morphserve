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
    """GPU equivalent of `pipeline_bagel.py:_resize_to_stride`. Zero-op when already stride-aligned."""
    import torch
    import torch.nn.functional as F

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
    resized = F.interpolate(
        img_u8.unsqueeze(0).to(torch.float32),
        size=(new_h, new_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[0]
    return resized.round().clamp(0.0, 255.0).to(torch.uint8)


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


def _vit_preprocess_u8_gpu(img_u8: Any, image_processor: Any) -> Any:
    """Exact GPU equivalent of HuggingFace `SiglipImageProcessor` / `BagelImageProcessor` on `[3, H, W]` uint8."""
    import torch
    import torch.nn.functional as F

    device = img_u8.device
    x = img_u8.to(torch.float32)  # [3, H, W] in [0, 255]

    do_resize = bool(getattr(image_processor, "do_resize", False))
    size_cfg = getattr(image_processor, "size", None)
    if do_resize and size_cfg:
        _, h, w = x.shape
        target_h, target_w = h, w
        if isinstance(size_cfg, dict):
            if "height" in size_cfg and "width" in size_cfg:
                target_h, target_w = int(size_cfg["height"]), int(size_cfg["width"])
            elif "shortest_edge" in size_cfg:
                se = int(size_cfg["shortest_edge"])
                if h <= w:
                    target_h, target_w = se, int(round(w * se / h))
                else:
                    target_w, target_h = se, int(round(h * se / w))
        elif isinstance(size_cfg, int):
            target_h = target_w = int(size_cfg)
        if target_h != h or target_w != w:
            resample = int(getattr(image_processor, "resample", 3))
            # PIL resample: 2=BILINEAR, 3=BICUBIC
            mode = "bilinear" if resample == 2 else "bicubic"
            x = F.interpolate(
                x.unsqueeze(0),
                size=(target_h, target_w),
                mode=mode,
                align_corners=False,
                antialias=True,
            )[0]

    # Ensure dimensions are multiples of vit_patch_size if processor has patch alignment
    do_rescale = bool(getattr(image_processor, "do_rescale", True))
    if do_rescale:
        rescale_factor = float(getattr(image_processor, "rescale_factor", 1.0 / 255.0))
        x = x * rescale_factor

    do_normalize = bool(getattr(image_processor, "do_normalize", True))
    if do_normalize:
        mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
        std = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])
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

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.dtype,
        ):
            if update_vae:
                vae_inp, new_lens, new_rope = prepare_vae_images_gpu(
                    self.bagel, [img_u8], ctx["kv_lens"], ctx["ropes"], self.new_token_ids
                )
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
