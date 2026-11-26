"""Pure session, turn, apportionment, and arrival-schedule contracts consumed by `MorphServeClusterEngine`."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import math
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
    use_prev_image: bool = False


@dataclass
class InterleavedSession:
    """One multi-turn or single-turn multimodal request session."""

    session_id: str
    quadrant: str  # "visual_cot" | "visual_cot_highres" | "interleaved_co_creation" | "multiturn_edit" | "multi_image_vqa" | "highres_t2i"
    source_dataset: str
    arrival_offset_s: float = 0.0
    turns: list[InterleavedTurn] = field(default_factory=list)
    phase_label: str = ""


@dataclass
class WorkloadSuite:
    """Serializable multimodal session suite consumed by `MorphServeClusterEngine`."""

    name: str
    trace_source: str
    base_qps: float
    sessions: list[InterleavedSession]
    schema: str = "morphserve.interleaved_bench.v2"
    warmup_s: float = 60.0
    steady_s: float = 180.0
    arrival_duration_s: float = 240.0
    phase_schedule: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "name": self.name,
            "trace_source": self.trace_source,
            "base_qps": self.base_qps,
            "num_sessions": len(self.sessions),
            "warmup_s": float(self.warmup_s),
            "steady_s": float(self.steady_s),
            "arrival_duration_s": float(self.arrival_duration_s),
            "phase_schedule": [dict(p) for p in self.phase_schedule],
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
                    phase_label=str(row.get("phase_label", "")),
                )
            )
        return cls(
            name=data["name"],
            trace_source=data.get("trace_source", "burstgpt+gentd26"),
            base_qps=float(data.get("base_qps", 1.0)),
            sessions=sessions,
            schema=data.get("schema", "morphserve.interleaved_bench.v2"),
            warmup_s=float(data.get("warmup_s", 60.0)),
            steady_s=float(data.get("steady_s", 180.0)),
            arrival_duration_s=float(data.get("arrival_duration_s", 240.0)),
            phase_schedule=[dict(p) for p in (data.get("phase_schedule") or [])],
        )

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "WorkloadSuite":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def hamilton_apportion(
    total: int,
    weights: Mapping[str, float],
    *,
    caps: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Apportion `total` integer slots across `weights` using the largest-remainder (Hamilton) method."""
    keys = list(weights.keys())
    if int(total) < 0:
        raise ValueError(f"hamilton_apportion requires total >= 0, got {total}")
    if not keys:
        return {}
    n_total = int(total)
    pos_weights = {k: max(0.0, float(weights[k])) for k in keys}
    w_sum = sum(pos_weights.values())
    if n_total == 0 or w_sum <= 0.0:
        return {k: 0 for k in keys}

    exact = {k: n_total * (pos_weights[k] / w_sum) for k in keys}
    counts: dict[str, int] = {}
    for k in keys:
        c = int(math.floor(exact[k]))
        if caps is not None and k in caps:
            c = min(max(0, int(caps[k])), c)
        counts[k] = c

    rem = n_total - sum(counts.values())
    if rem > 0:
        order = sorted(range(len(keys)), key=lambda idx: (-(exact[keys[idx]] - math.floor(exact[keys[idx]])), idx))
        if caps is None:
            for idx in order[:rem]:
                counts[keys[idx]] += 1
        else:
            cursor = 0
            while rem > 0:
                progressed = False
                for pos in range(len(order)):
                    k = keys[order[(cursor + pos) % len(order)]]
                    cap = max(0, int(caps[k])) if k in caps else n_total
                    if counts[k] < cap:
                        counts[k] += 1
                        rem -= 1
                        cursor = (cursor + pos + 1) % len(order)
                        progressed = True
                        break
                if not progressed:
                    break
    return counts


def scale_arrivals_to_qps(norm_deltas: list[float], target_qps: float, count: int) -> list[float]:
    """Convert normalized inter-arrival deltas (`mean=1.0`) into absolute arrival offsets in seconds."""
    if count <= 0:
        return []
    qps = max(float(target_qps), 1e-6)
    deltas = norm_deltas if norm_deltas else [1.0]
    offsets = [0.0]
    for i in range(1, count):
        d = float(deltas[(i - 1) % len(deltas)])
        offsets.append(round(offsets[-1] + d / qps, 4))
    return offsets


def _is_ar_initiating_session(session: InterleavedSession) -> bool:
    """Return True for AR-initiating sessions (BurstGPT trace), False for DiT-initiating sessions (GenTD26)."""
    if session.quadrant in (
        "multi_image_vqa",
        "visual_cot",
        "visual_cot_highres",
        "interleaved_co_creation",
    ):
        return True
    if session.quadrant in ("multiturn_edit", "highres_t2i"):
        return False
    if session.turns and session.turns[0].modality_plan in ("ar_only", "cot_then_dit"):
        return True
    if session.turns and session.turns[0].modality_plan in ("dit_edit", "dit_t2i"):
        return False
    return False


def assign_dual_trace_arrivals(
    sessions: list[InterleavedSession],
    burstgpt_deltas: list[float],
    gentd26_deltas: list[float],
    target_qps: float,
) -> list[InterleavedSession]:
    """Assign BurstGPT offsets to AR-initiating sessions and GenTD26 offsets to DiT-initiating sessions."""
    ar_indices = [i for i, s in enumerate(sessions) if _is_ar_initiating_session(s)]
    dit_indices = [i for i, s in enumerate(sessions) if not _is_ar_initiating_session(s)]
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


def build_time_window_suite(
    base_suite: WorkloadSuite,
    target_qps: float,
    burstgpt_deltas: list[float] | None,
    gentd26_deltas: list[float] | None,
    *,
    arrival_duration_s: float = 240.0,
    min_sessions: int = 40,
    max_sessions: int | None = None,
    warmup_s: float = 60.0,
    steady_s: float = 180.0,
) -> WorkloadSuite:
    """Build a fixed-duration time-window suite preserving quadrant proportions and assigning arrivals."""
    eff_qps = max(float(target_qps), 1e-4)
    eff_dur = max(0.0, float(arrival_duration_s))
    target_n = max(int(min_sessions), int(round(eff_qps * eff_dur)))
    if max_sessions is not None:
        target_n = min(target_n, max(1, int(max_sessions)))

    if not base_suite.sessions or target_n <= 0:
        return WorkloadSuite(
            name=base_suite.name,
            trace_source=base_suite.trace_source,
            base_qps=eff_qps,
            sessions=[],
            schema="morphserve.interleaved_bench.v2",
            warmup_s=float(warmup_s),
            steady_s=float(steady_s),
            arrival_duration_s=float(arrival_duration_s),
            phase_schedule=[dict(p) for p in base_suite.phase_schedule],
        )

    by_quad: dict[str, list[InterleavedSession]] = {}
    for s in base_suite.sessions:
        by_quad.setdefault(s.quadrant, []).append(s)

    quads = list(by_quad.keys())
    counts = hamilton_apportion(target_n, {q: float(len(by_quad[q])) for q in quads})

    # Smooth proportional interleaving across quadrants keeps quadrant mix stationary across time windows.
    active_quads = [q for q in quads if counts[q] > 0]
    emitted = {q: 0 for q in active_quads}
    used_ids: set[str] = set()
    expanded: list[InterleavedSession] = []

    for _ in range(target_n):
        best_q = min(
            (q for q in active_quads if emitted[q] < counts[q]),
            key=lambda q: ((emitted[q] + 0.5) / float(counts[q]), active_quads.index(q)),
        )
        pos = emitted[best_q]
        emitted[best_q] = pos + 1
        pool = by_quad[best_q]
        tmpl = pool[pos % len(pool)]
        rep = pos // len(pool)

        cand_id = tmpl.session_id if (rep == 0 and tmpl.session_id not in used_ids) else f"{tmpl.session_id}__w{rep}"
        if cand_id in used_ids:
            suffix = 1
            while f"{cand_id}_{suffix}" in used_ids:
                suffix += 1
            cand_id = f"{cand_id}_{suffix}"
        used_ids.add(cand_id)

        cloned = deepcopy(tmpl)
        cloned.session_id = cand_id
        cloned.arrival_offset_s = 0.0
        expanded.append(cloned)

    if burstgpt_deltas and gentd26_deltas:
        assign_dual_trace_arrivals(expanded, burstgpt_deltas, gentd26_deltas, target_qps=eff_qps)
    else:
        for idx_s, s_item in enumerate(expanded):
            s_item.arrival_offset_s = round(idx_s / eff_qps, 4)

    return WorkloadSuite(
        name=base_suite.name,
        trace_source=base_suite.trace_source,
        base_qps=eff_qps,
        sessions=expanded,
        schema="morphserve.interleaved_bench.v2",
        warmup_s=float(warmup_s),
        steady_s=float(steady_s),
        arrival_duration_s=float(arrival_duration_s),
        phase_schedule=[dict(p) for p in base_suite.phase_schedule],
    )


__all__ = [
    "InterleavedSession",
    "InterleavedTurn",
    "WorkloadSuite",
    "assign_dual_trace_arrivals",
    "build_time_window_suite",
    "hamilton_apportion",
    "scale_arrivals_to_qps",
]
