"""MorphServe unified CPTGModelAdapter implementations (`BAGEL-7B-MoT`, `InternVL-U`, `Synthetic`)."""

from .bagel_adapter import (
    BagelCPTGAdapter,
    apply_vllm_omni_mm_limit,
    bagel_ctx_nbytes,
    bagel_entry_nbytes,
    load_bagel_pipeline,
    pack_bagel_entry_to_pinned_host,
    pack_bagel_kv_context,
    pack_bagel_multi_context,
    unpack_bagel_entry_from_pinned_host,
    unpack_bagel_kv_context,
    unpack_bagel_multi_context,
)
from .base import (
    CPTGModelAdapter,
    PackedTransferState,
    SyntheticCPTGAdapter,
    SyntheticOutOfMemoryError,
    compute_vmm_available_kv_bytes,
)
from .internvl_u_adapter import (
    InternVLUCPTGAdapter,
    compute_internvlu_memory_plan,
    load_internvlu_pipeline,
    pack_internvlu_bridge_state,
    unpack_internvlu_bridge_state,
)

__all__ = [
    "BagelCPTGAdapter",
    "CPTGModelAdapter",
    "InternVLUCPTGAdapter",
    "PackedTransferState",
    "SyntheticCPTGAdapter",
    "SyntheticOutOfMemoryError",
    "apply_vllm_omni_mm_limit",
    "bagel_ctx_nbytes",
    "bagel_entry_nbytes",
    "compute_internvlu_memory_plan",
    "compute_vmm_available_kv_bytes",
    "load_bagel_pipeline",
    "load_internvlu_pipeline",
    "pack_bagel_entry_to_pinned_host",
    "pack_bagel_kv_context",
    "pack_bagel_multi_context",
    "pack_internvlu_bridge_state",
    "unpack_bagel_entry_from_pinned_host",
    "unpack_bagel_kv_context",
    "unpack_bagel_multi_context",
    "unpack_internvlu_bridge_state",
]
