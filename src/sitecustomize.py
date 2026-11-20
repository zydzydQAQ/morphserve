"""Optional process-level hook for vLLM-Omni multi-image BAGEL limit override."""

from __future__ import annotations

import os

if os.environ.get("MORPHSERVE_ENABLE_BAGEL_MM_ADAPTER") == "1":
    from morphserve.models.bagel_adapter import apply_vllm_omni_mm_limit

    apply_vllm_omni_mm_limit()
