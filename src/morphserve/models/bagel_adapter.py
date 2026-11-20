"""BAGEL-7B-MoT adapter helpers for vLLM-Omni multi-turn multimodal sessions."""

from __future__ import annotations

import json
import os


def apply_vllm_omni_mm_limit() -> None:
    from vllm_omni.model_executor.models.bagel.bagel import OmniBagelProcessingInfo

    max_images = int(os.environ.get("MORPHSERVE_MAX_IMAGE_ITEMS", "16"))
    if max_images < 1:
        raise ValueError("MORPHSERVE_MAX_IMAGE_ITEMS must be positive")

    def get_supported_mm_limits(self):  # noqa: ARG001
        return {"image": max_images, "img2img": 1}

    OmniBagelProcessingInfo.get_supported_mm_limits = get_supported_mm_limits
    print(
        json.dumps({"morphserve_bagel_mm_adapter": True, "supported_mm_limits": {"image": max_images, "img2img": 1}}),
        flush=True,
    )
