"""Zero-dependency runtime telemetry utilities shared by engine, router, and simulator."""

from __future__ import annotations

import math
from typing import Sequence

MIB = 1024**2


def percentile(vals: Sequence[float], p: float) -> float:
    """Deterministic linear-interpolated percentile (zero external dependencies)."""
    if not vals:
        return 0.0
    s = sorted(float(x) for x in vals)
    idx = (len(s) - 1) * (float(p) / 100.0)
    lo = int(math.floor(idx))
    hi = min(len(s) - 1, int(math.ceil(idx)))
    if lo == hi:
        return round(s[lo], 2)
    frac = idx - lo
    return round(s[lo] * (1.0 - frac) + s[hi] * frac, 2)


def read_shmem_kib() -> int:
    """Read host `/proc/meminfo` `Shmem:` footprint in KiB."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("Shmem:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def gpu_mem_info_mib(device: int = 0) -> dict[str, float]:
    """Snapshot PyTorch allocator and CUDA driver HBM usage in MiB on `device`."""
    import torch

    free_b, total_b = torch.cuda.mem_get_info(device)
    return {
        "torch_alloc_mib": round(torch.cuda.memory_allocated(device) / MIB, 2),
        "torch_reserved_mib": round(torch.cuda.memory_reserved(device) / MIB, 2),
        "driver_used_mib": round((total_b - free_b) / MIB, 2),
        "driver_free_mib": round(free_b / MIB, 2),
    }


__all__ = ["MIB", "gpu_mem_info_mib", "percentile", "read_shmem_kib"]
