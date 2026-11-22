"""Pure session, turn, and arrival-schedule contracts consumed by `MorphServeClusterEngine`."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class InterleavedTurn:
    """Single conversational turn inside an Interleaved-Bench session."""

    turn_idx: int
    modality_plan: str  # "ar_only" | "cot_then_dit" | "dit_edit" | "dit_t2i"
    prompt: str
    image_paths: list[str] = field(default_factory=list)
    height: int = 512
    width: int = 512
    dit_steps: int = 20
    max_ar_tokens: int = 48
    expected_answer: str = ""
    geneval_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterleavedSession:
    """One multi-turn or single-turn multimodal request session."""

    session_id: str
    quadrant: str  # "visual_cot" | "multiturn_edit" | "multi_image_vqa" | "highres_t2i"
    source_dataset: str
    arrival_offset_s: float = 0.0
    turns: list[InterleavedTurn] = field(default_factory=list)


@dataclass
class WorkloadSuite:
    """Serializable multimodal session suite consumed by `MorphServeClusterEngine`."""

    name: str
    trace_source: str
    base_qps: float
    sessions: list[InterleavedSession]
    schema: str = "morphserve.interleaved_bench.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "name": self.name,
            "trace_source": self.trace_source,
            "base_qps": self.base_qps,
            "num_sessions": len(self.sessions),
            "sessions": [asdict(s) for s in self.sessions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkloadSuite":
        sessions: list[InterleavedSession] = []
        for row in data.get("sessions", []):
            turns = [InterleavedTurn(**t) for t in row.get("turns", [])]
            sessions.append(
                InterleavedSession(
                    session_id=row["session_id"],
                    quadrant=row["quadrant"],
                    source_dataset=row["source_dataset"],
                    arrival_offset_s=float(row.get("arrival_offset_s", 0.0)),
                    turns=turns,
                )
            )
        return cls(
            name=data["name"],
            trace_source=data.get("trace_source", "burstgpt+gentd26"),
            base_qps=float(data.get("base_qps", 1.0)),
            sessions=sessions,
            schema=data.get("schema", "morphserve.interleaved_bench.v1"),
        )

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "WorkloadSuite":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def scale_arrivals_to_qps(norm_deltas: list[float], target_qps: float, count: int) -> list[float]:
    """Convert normalized inter-arrival deltas (`mean=1.0`) into absolute arrival offsets in seconds."""
    if count <= 0:
        return []
    qps = max(float(target_qps), 1e-6)
    offsets = [0.0]
    for i in range(1, count):
        d = norm_deltas[(i - 1) % len(norm_deltas)]
        offsets.append(round(offsets[-1] + d / qps, 4))
    return offsets


def assign_dual_trace_arrivals(
    sessions: list[InterleavedSession],
    burstgpt_deltas: list[float],
    gentd26_deltas: list[float],
    target_qps: float,
) -> list[InterleavedSession]:
    """Assign BurstGPT offsets to AR/CoT sessions and GenTD26 offsets to DiT sessions."""
    ar_indices = [i for i, s in enumerate(sessions) if s.quadrant in ("multi_image_vqa", "visual_cot")]
    dit_indices = [i for i, s in enumerate(sessions) if s.quadrant not in ("multi_image_vqa", "visual_cot")]
    total = max(1, len(sessions))
    ar_qps = max(1e-4, target_qps * (len(ar_indices) / float(total)))
    dit_qps = max(1e-4, target_qps * (len(dit_indices) / float(total)))

    ar_offs = scale_arrivals_to_qps(burstgpt_deltas, ar_qps, len(ar_indices))
    dit_offs = scale_arrivals_to_qps(gentd26_deltas, dit_qps, len(dit_indices))

    for pos, idx in enumerate(ar_indices):
        sessions[idx].arrival_offset_s = ar_offs[pos]
    for pos, idx in enumerate(dit_indices):
        sessions[idx].arrival_offset_s = dit_offs[pos]
    sessions.sort(key=lambda s: (s.arrival_offset_s, s.session_id))
    return sessions


__all__ = [
    "InterleavedSession",
    "InterleavedTurn",
    "WorkloadSuite",
    "assign_dual_trace_arrivals",
    "scale_arrivals_to_qps",
]
