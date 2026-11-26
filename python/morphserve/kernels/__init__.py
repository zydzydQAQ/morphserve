"""MorphServe Module 3: Pure Cross-Paradigm State De-Redundancy kernels (`LECP CoW`, `GpuInPlaceLoopback`, `FlashPrefix-CFG`)."""

from .cfg_prefix_attn import (
    get_fa2_backend_status,
    shared_prefix_cfg_attention,
    shared_prefix_cfg_attention_ref,
    should_use_cascade_cfg_split,
    should_use_flash_prefix_split,
)
from .gpu_loopback import (
    GpuInPlaceLoopback,
    decode_latent_to_u8_gpu,
    pil_bicubic_resize_gpu,
    pil_bicubic_resize_np,
    prepare_vae_images_gpu,
    prepare_vit_images_gpu,
    tv_bicubic_u8_resize_gpu,
    update_context_from_u8_gpu,
)
from .lecp_cache import (
    CoWBranchSpec,
    SharedPrefixCache,
    cow_fork_context,
    detect_lecp_prefix_len,
    rebind_cow_prefix,
)

__all__ = [
    "CoWBranchSpec",
    "GpuInPlaceLoopback",
    "SharedPrefixCache",
    "cow_fork_context",
    "decode_latent_to_u8_gpu",
    "detect_lecp_prefix_len",
    "get_fa2_backend_status",
    "pil_bicubic_resize_gpu",
    "pil_bicubic_resize_np",
    "prepare_vae_images_gpu",
    "prepare_vit_images_gpu",
    "rebind_cow_prefix",
    "shared_prefix_cfg_attention",
    "shared_prefix_cfg_attention_ref",
    "should_use_cascade_cfg_split",
    "should_use_flash_prefix_split",
    "tv_bicubic_u8_resize_gpu",
    "update_context_from_u8_gpu",
]
