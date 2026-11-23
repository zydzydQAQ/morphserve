"""MorphServe Module 3: Cross-Paradigm State De-Redundancy (LECP CoW, GPU Loopback & Cascade-CFG Attention)."""

from .cfg_prefix_attn import (
    BagelSharedPrefixPatch,
    get_fa2_backend_status,
    install_shared_prefix_bagel_hooks,
    shared_prefix_cfg_attention,
    shared_prefix_cfg_attention_ref,
    should_use_cascade_cfg_split,
    should_use_flash_prefix_split,
)
from .gpu_loopback import (
    GpuInPlaceLoopback,
    decode_latent_to_u8_gpu,
    prepare_vae_images_gpu,
    prepare_vit_images_gpu,
)
from .lecp_cache import (
    CoWBranchSpec,
    SharedPrefixCache,
    cow_fork_context,
    detect_lecp_prefix_len,
    rebind_cow_prefix,
)

__all__ = [
    "BagelSharedPrefixPatch",
    "CoWBranchSpec",
    "GpuInPlaceLoopback",
    "SharedPrefixCache",
    "cow_fork_context",
    "decode_latent_to_u8_gpu",
    "detect_lecp_prefix_len",
    "get_fa2_backend_status",
    "install_shared_prefix_bagel_hooks",
    "prepare_vae_images_gpu",
    "prepare_vit_images_gpu",
    "rebind_cow_prefix",
    "shared_prefix_cfg_attention",
    "shared_prefix_cfg_attention_ref",
    "should_use_cascade_cfg_split",
    "should_use_flash_prefix_split",
]
