"""MorphServe Module 2: Three-tier topology, `OperatorLatencyProfile`, and C++-backed `DualPathMorphRouter`."""

from typing import Any

from .cost_router import (
    CPTGRequestSpec,
    DualPathMorphRouter,
    HalfNodeH2DPacer,
    RouteDecision,
    WorkerState,
)
from .topology import (
    ClusterTopologyProfile,
    OperatorLatencyProfile,
    PcieHalfGroup,
    default_operator_profile,
    default_topology_for,
    probe_live_topology,
)


def __getattr__(name: str) -> Any:
    if name == "DEFAULT_OPERATOR_PROFILE":
        return default_operator_profile()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CPTGRequestSpec",
    "ClusterTopologyProfile",
    "DEFAULT_OPERATOR_PROFILE",
    "DualPathMorphRouter",
    "HalfNodeH2DPacer",
    "OperatorLatencyProfile",
    "PcieHalfGroup",
    "RouteDecision",
    "WorkerState",
    "default_operator_profile",
    "default_topology_for",
    "probe_live_topology",
]
