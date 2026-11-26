"""Phase-2 Milestone M2.3: 8xA100 BAGEL-7B-MoT Interleaved-Bench v2 Time-Windowed Evaluation, Phase-Shift, Think-Time Sweep & Two-Tier Accuracy."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Sequence

from morphserve.bench.exactness import (
    ExactnessEvaluator,
    build_m3_active_accuracy_suite,
    image_pair_metrics,
)
from morphserve.bench.interleaved_bench import (
    InterleavedSession,
    InterleavedTurn,
    WorkloadSuite,
    build_online_phase_shift_suite,
    load_burstgpt_interarrivals,
    load_gentd26_interarrivals,
)
from morphserve.engine import EnginePolicyConfig, MorphServeClusterEngine, VALID_POLICIES
from morphserve.engine.workload import build_time_window_suite
from morphserve.telemetry import percentile as _pct

DEFAULT_MAIN_SESSIONS = 96
DEFAULT_ABLATION_SESSIONS = 80
DEFAULT_SKEW_SESSIONS = 80

DEFAULT_ARRIVAL_DURATION_S = 240.0
DEFAULT_WARMUP_S = 60.0
DEFAULT_STEADY_S = 180.0
DEFAULT_MIN_SESSIONS = 40

DEFAULT_SUITE_A_QPS = (0.15, 0.30, 0.50, 0.80, 1.10)
DEFAULT_LOWQPS_POINTS = (0.05, 0.10, 0.15, 0.25, 0.40, 0.80)
DEFAULT_THINK_TIME_POLICIES = (
    "morphserve",
    "monolithic_dp",
    "prism_tetriserve_coloc",
    "static_nccl_fission",
)
DEFAULT_THINK_TIMES_S = (0.0, 5.0, 20.0, 60.0)
DEFAULT_PHASE_SHIFT_POLICIES = (
    "morphserve",
    "monolithic_dp",
    "prism_tetriserve_coloc",
    "static_nccl_fission",
    "static_2und_6gen",
    "static_6und_2gen",
)


def _resolve_sources_dir(bench_root: Path) -> Path:
    for cand in (bench_root / "sources", bench_root.parent / "bench_sources"):
        if cand.is_dir():
            return cand
    return bench_root / "sources"


def _load_main_mix_suite(bench_root: Path) -> WorkloadSuite:
    for rel in ("suites/main_mix_v2_80.json", "suites/main_mix_80.json"):
        cand = bench_root / rel
        if cand.is_file():
            return WorkloadSuite.load_json(cand)
    raise FileNotFoundError(f"Missing main_mix_v2_80.json / main_mix_80.json under {bench_root / 'suites'}")


def _hamilton_quadrant_counts(
    by_quad: dict[str, list[InterleavedSession]],
    n_tot: int,
    target_sessions: int,
    *,
    cap_to_available: bool,
) -> dict[str, int]:
    quotas: list[tuple[str, int, float]] = []
    used = 0
    for q, items in sorted(by_quad.items()):
        exact = target_sessions * (len(items) / float(max(1, n_tot)))
        cnt = int(math.floor(exact))
        if cap_to_available:
            cnt = min(len(items), cnt)
        quotas.append((q, cnt, exact - math.floor(exact)))
        used += cnt

    quotas.sort(key=lambda x: (-x[2], x[0]))
    counts = {q: c for q, c, _ in quotas}
    rem = target_sessions - used
    idx = 0
    while rem > 0 and quotas:
        q = quotas[idx % len(quotas)][0]
        if (not cap_to_available) or counts[q] < len(by_quad[q]):
            counts[q] += 1
            rem -= 1
        elif all(counts[k] >= len(by_quad[k]) for k in counts):
            break
        idx += 1
    return counts


def stratify_suite(suite: WorkloadSuite, max_sessions: int) -> WorkloadSuite:
    """Select a deterministic, quadrant-balanced slice of `suite` preserving arrival ordering."""
    if max_sessions <= 0 or len(suite.sessions) <= max_sessions:
        return suite
    by_quad: dict[str, list[InterleavedSession]] = defaultdict(list)
    for s in suite.sessions:
        by_quad[s.quadrant].append(s)

    counts = _hamilton_quadrant_counts(by_quad, len(suite.sessions), max_sessions, cap_to_available=True)
    picked: list[InterleavedSession] = []
    for q, items in sorted(by_quad.items()):
        picked.extend(items[: counts[q]])
    picked.sort(key=lambda s: (s.arrival_offset_s, s.session_id))
    return WorkloadSuite(
        name=f"{suite.name}_n{len(picked)}",
        trace_source=suite.trace_source,
        base_qps=suite.base_qps,
        sessions=picked,
        schema=suite.schema,
    )


def _local_tmp_base() -> Path:
    raw = os.environ.get("MORPHSERVE_LOCAL_TMP", "").strip()
    return Path(raw) if raw else Path("/tmp")


def scale_suite_sessions(suite: WorkloadSuite, target_sessions: int) -> WorkloadSuite:
    """Align `suite` to `target_sessions`, downsampling or cyclically expanding by quadrant."""
    target = int(target_sessions)
    n_base = len(suite.sessions)
    if target <= 0 or n_base == 0 or target == n_base:
        return suite
    if target < n_base:
        return stratify_suite(suite, max_sessions=target)

    ordered_base = sorted(suite.sessions, key=lambda s: (float(s.arrival_offset_s), s.session_id))
    t_min = float(ordered_base[0].arrival_offset_s)
    t_max = float(ordered_base[-1].arrival_offset_s)
    span_s = max(0.0, t_max - t_min)
    mean_step_s = max(0.5, span_s / float(max(1, n_base - 1))) if n_base > 1 else 1.0
    cycle_span_s = span_s + mean_step_s

    full_cycles = target // n_base
    rem = target % n_base
    expanded: list[InterleavedSession] = []

    for rep in range(full_cycles):
        base_shift_s = rep * cycle_span_s
        for src in ordered_base:
            sid = src.session_id if rep == 0 else f"{src.session_id}_r{rep}"
            cloned_turns = [InterleavedTurn(**asdict(t)) for t in src.turns]
            expanded.append(
                InterleavedSession(
                    session_id=sid,
                    quadrant=src.quadrant,
                    source_dataset=src.source_dataset,
                    arrival_offset_s=round(float(src.arrival_offset_s) + base_shift_s, 4),
                    turns=cloned_turns,
                    phase_label=src.phase_label,
                )
            )

    if rem > 0:
        tail_slice = stratify_suite(
            WorkloadSuite(
                name=suite.name,
                trace_source=suite.trace_source,
                base_qps=suite.base_qps,
                sessions=ordered_base,
                schema=suite.schema,
            ),
            max_sessions=rem,
        ).sessions
        tail_start_s = (full_cycles * cycle_span_s) + t_min
        for idx_t, src in enumerate(tail_slice):
            sid = f"{src.session_id}_r{full_cycles}"
            cloned_turns = [InterleavedTurn(**asdict(t)) for t in src.turns]
            expanded.append(
                InterleavedSession(
                    session_id=sid,
                    quadrant=src.quadrant,
                    source_dataset=src.source_dataset,
                    arrival_offset_s=round(tail_start_s + idx_t * mean_step_s, 4),
                    turns=cloned_turns,
                    phase_label=src.phase_label,
                )
            )

    expanded.sort(key=lambda s: (s.arrival_offset_s, s.session_id))
    return WorkloadSuite(
        name=f"{suite.name}_n{len(expanded)}",
        trace_source=suite.trace_source,
        base_qps=suite.base_qps,
        sessions=expanded,
        schema=suite.schema,
    )


def _v2_time_window_policy_overrides(
    *,
    warmup_s: float = DEFAULT_WARMUP_S,
    steady_s: float = DEFAULT_STEADY_S,
    think_range: tuple[float, float] = (2.0, 25.0),
    think_dist: str = "lognormal",
    think_median_s: float = 6.0,
    think_sigma: float = 0.75,
) -> dict[str, Any]:
    return {
        "inter_turn_think_s_range": think_range,
        "think_time_dist": think_dist,
        "think_time_median_s": float(think_median_s),
        "think_time_sigma": float(think_sigma),
        "ar_tpot_slo_mode": "turn_p95",
        "steady_window_mode": "time",
        "warmup_window_s": float(warmup_s),
        "steady_window_s": float(steady_s),
    }


def _build_policy_config(
    policy_name: str,
    num_gpus: int = 8,
    **overrides: Any,
) -> EnginePolicyConfig:
    """Resolve standard presets plus asymmetric static split baselines (`static_2und_6gen`, `static_6und_2gen`)."""
    n = int(num_gpus)
    if policy_name in ("static_2und_6gen", "static_6und_2gen"):
        base_static = EnginePolicyConfig.from_preset("static_nccl_fission", num_gpus=n, **overrides)
        if n >= 4 and n % 4 == 0:
            if n % 8 == 0:
                und_1_3 = tuple(g for g in range(n) if (g % 8) in (0, 4))
                und_3_1 = tuple(g for g in range(n) if (g % 8) in (0, 1, 2, 4, 5, 6))
            else:
                q = n // 4
                und_1_3 = tuple(range(q))
                und_3_1 = tuple(range(3 * q))
            und_ranks = und_1_3 if policy_name == "static_2und_6gen" else und_3_1
        elif n >= 2:
            und_ranks = (0,) if policy_name == "static_2und_6gen" else tuple(range(n - 1))
        else:
            und_ranks = (0,)
        gen_ranks = tuple(g for g in range(n) if g not in und_ranks) or (0,)
        return replace(
            base_static,
            policy_name=policy_name,
            static_und_ranks=und_ranks,
            static_gen_ranks=gen_ranks,
            initial_alphas={g: (0.0 if g in und_ranks else 1.0) for g in range(n)},
        )
    return EnginePolicyConfig.from_preset(policy_name, num_gpus=n, **overrides)


def build_ablation_configs(num_gpus: int = 8, **policy_overrides: Any) -> list[tuple[str, EnginePolicyConfig]]:
    """Construct strict single-variable Leave-One-Out M1-M4 ablations relative to `full_morphserve`."""
    base_ms = EnginePolicyConfig.from_preset("morphserve", num_gpus=num_gpus, **policy_overrides)
    no_m4 = replace(base_ms, policy_name="no_m4_layer_chunk", preemption_mode="non_preemptive")
    no_m3 = replace(base_ms, policy_name="no_m3_cow_loopback", use_module3_cow_loopback=False)
    no_m2 = replace(
        base_ms,
        policy_name="no_m2_dual_path",
        routing_mode="sticky_least_loaded",
        use_h2d_pacer=False,
        use_shared_numa_mirror=False,
    )
    no_m1 = replace(
        base_ms,
        policy_name="no_m1_tripool_vmm",
        use_tripool_vmm=False,
        use_expert_pager=False,
        use_scoped_vae=False,
        use_shared_numa_mirror=False,
        use_h2d_pacer=False,
    )
    return [
        ("full_morphserve", base_ms),
        ("no_m4_layer_chunk", no_m4),
        ("no_m3_cow_loopback", no_m3),
        ("no_m2_dual_path", no_m2),
        ("no_m1_tripool_vmm", no_m1),
    ]


def _interpolate_slo_threshold_qps(
    points: list[tuple[float, float]],
    threshold_pct: float,
) -> float:
    """Compute highest sustainable QPS where piecewise-linear `SLO Attainment (%) vs. QPS` >= `threshold_pct`."""
    if not points:
        return 0.0
    pts = sorted(points, key=lambda x: float(x[0]))
    best_q = 0.0
    for idx, (q_cur, att_cur) in enumerate(pts):
        if att_cur >= threshold_pct:
            best_q = max(best_q, q_cur)
            if idx + 1 < len(pts):
                q_nxt, att_nxt = pts[idx + 1]
                if att_nxt < threshold_pct and att_cur > att_nxt and q_nxt > q_cur:
                    frac = (att_cur - threshold_pct) / (att_cur - att_nxt)
                    best_q = max(best_q, q_cur + frac * (q_nxt - q_cur))
    return round(best_q, 4)


def _interpolate_zero_oom_slo_qps(
    points: list[tuple[float, float, int]],
    threshold_pct: float,
) -> float:
    """Highest sustainable QPS with strictly zero steady-state OOMs (`steady_oom_sessions == 0`).

    Interpolates only between adjacent zero-OOM points; never interpolates into a point that suffered OOM.
    """
    if not points:
        return 0.0
    pts = sorted(points, key=lambda x: float(x[0]))
    best_q = 0.0
    for idx, (q_cur, att_cur, oom_cur) in enumerate(pts):
        if oom_cur == 0 and att_cur >= threshold_pct:
            best_q = max(best_q, q_cur)
            if idx + 1 < len(pts):
                q_nxt, att_nxt, oom_nxt = pts[idx + 1]
                if oom_nxt == 0 and att_nxt < threshold_pct and att_cur > att_nxt and q_nxt > q_cur:
                    frac = (att_cur - threshold_pct) / (att_cur - att_nxt)
                    best_q = max(best_q, q_cur + frac * (q_nxt - q_cur))
    return round(best_q, 4)


def _compute_sustainable_qps_summary(qps_sweep_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize steady-state `SLO Attainment (%) vs. QPS`, zero-OOM sustainable QPS & Active JCT curves."""
    by_pol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qps_sweep_results:
        by_pol[str(row["policy_name"])].append(row)
    summary: dict[str, dict[str, Any]] = {}
    for pol, rows in sorted(by_pol.items()):
        rows_sorted = sorted(rows, key=lambda r: float(r["offered_qps"]))
        steady_pts = [
            (float(r["offered_qps"]), float(r.get("steady_state_attainment_pct", r["joint_slo_attainment_pct"])))
            for r in rows_sorted
        ]
        steady_turn_pts = [
            (
                float(r["offered_qps"]),
                float(
                    r.get(
                        "steady_turn_slo_attainment_pct",
                        r.get("steady_state_attainment_pct", r["joint_slo_attainment_pct"]),
                    )
                ),
            )
            for r in rows_sorted
        ]
        steady_zero_oom_pts = [
            (
                float(r["offered_qps"]),
                float(r.get("steady_state_attainment_pct", r["joint_slo_attainment_pct"])),
                int(r.get("steady_oom_sessions", r.get("oom_sessions", 0))),
            )
            for r in rows_sorted
        ]
        steady_turn_zero_oom_pts = [
            (
                float(r["offered_qps"]),
                float(
                    r.get(
                        "steady_turn_slo_attainment_pct",
                        r.get("steady_state_attainment_pct", r["joint_slo_attainment_pct"]),
                    )
                ),
                int(r.get("steady_oom_sessions", r.get("oom_sessions", 0))),
            )
            for r in rows_sorted
        ]
        full_pts = [
            (float(r["offered_qps"]), float(r["joint_slo_attainment_pct"]))
            for r in rows_sorted
        ]
        grid_80 = max((q for q, a in steady_pts if a >= 80.0), default=0.0)
        grid_90 = max((q for q, a in steady_pts if a >= 90.0), default=0.0)
        turn_grid_80 = max((q for q, a in steady_turn_pts if a >= 80.0), default=0.0)
        turn_grid_90 = max((q for q, a in steady_turn_pts if a >= 90.0), default=0.0)
        curve = []
        for r in rows_sorted:
            tot_s = max(1, int(r.get("total_sessions", 1)))
            ok_s = int(r.get("completed_sessions", tot_s - int(r.get("oom_sessions", 0)) - int(r.get("error_sessions", 0))))
            surv_pct = float(r.get("survival_rate_pct", round(100.0 * ok_s / tot_s, 1)))
            steady_surv_pct = float(r.get("steady_state_survival_pct", surv_pct))
            steady_oom = int(r.get("steady_oom_sessions", r.get("oom_sessions", 0)))
            steady_att = float(r.get("steady_state_attainment_pct", r["joint_slo_attainment_pct"]))
            steady_turn_att = float(r.get("steady_turn_slo_attainment_pct", steady_att))
            curve.append(
                {
                    "offered_qps": float(r["offered_qps"]),
                    "steady_turn_slo_attainment_pct": steady_turn_att,
                    "steady_turn_slo_goodput_rps": float(
                        r.get("steady_turn_slo_goodput_rps", r.get("steady_state_goodput_rps", 0.0))
                    ),
                    "steady_turn_survival_pct": float(r.get("steady_turn_survival_pct", steady_surv_pct)),
                    "steady_planned_turns": int(r.get("steady_planned_turns", 0)),
                    "steady_state_attainment_pct": steady_att,
                    "joint_slo_attainment_pct": float(r["joint_slo_attainment_pct"]),
                    "steady_state_survival_pct": steady_surv_pct,
                    "survival_rate_pct": surv_pct,
                    "steady_oom_sessions": steady_oom,
                    "oom_sessions": int(r.get("oom_sessions", 0)),
                    "steady_state_sessions": int(r.get("steady_state_sessions", tot_s)),
                    "total_sessions": tot_s,
                    "effective_slo_throughput_rps": float(
                        r.get("effective_slo_throughput_rps", r.get("steady_state_goodput_rps", 0.0))
                    ),
                    "steady_state_goodput_rps": float(r.get("steady_state_goodput_rps", 0.0)),
                    "ar_ttft_p95_ms": float(r.get("ar_ttft_p95_ms", 0.0)),
                    "ar_tpot_p95_ms": float(r.get("ar_tpot_p95_ms", 0.0)),
                    "survivor_dit_turn_p95_ms": float(r.get("dit_turn_p95_ms", 0.0)),
                    "all_session_dit_p95_ms": r.get("all_session_dit_p95_ms"),
                    "active_jct_p95_ms": r.get("active_jct_p95_ms"),
                    "all_session_active_jct_p95_ms": r.get("all_session_active_jct_p95_ms"),
                    "all_session_e2e_p95_ms": r.get("all_session_e2e_p95_ms"),
                    "kv_host_swap_outs": int(r.get("kv_host_swap_outs", 0)),
                    "peak_nvml_used_mib": float(r.get("peak_nvml_used_mib", r.get("peak_gpu_reserved_mib", 0.0))),
                }
            )
        max_q_80 = _interpolate_slo_threshold_qps(steady_pts, 80.0)
        max_q_90 = _interpolate_slo_threshold_qps(steady_pts, 90.0)
        turn_max_q_80 = _interpolate_slo_threshold_qps(steady_turn_pts, 80.0)
        turn_max_q_90 = _interpolate_slo_threshold_qps(steady_turn_pts, 90.0)
        summary[pol] = {
            "primary_metric": "steady_turn_slo_attainment_pct_vs_offered_qps",
            "secondary_session_metric": "steady_state_slo_attainment_pct_vs_offered_qps",
            "turn_max_sustainable_qps_80pct": turn_max_q_80,
            "turn_max_sustainable_qps_90pct": turn_max_q_90,
            "turn_max_sustainable_qps_zero_oom_80pct": _interpolate_zero_oom_slo_qps(steady_turn_zero_oom_pts, 80.0),
            "turn_max_sustainable_qps_zero_oom_90pct": _interpolate_zero_oom_slo_qps(steady_turn_zero_oom_pts, 90.0),
            "turn_max_grid_qps_80pct": round(turn_grid_80, 3),
            "turn_max_grid_qps_90pct": round(turn_grid_90, 3),
            "max_sustainable_qps_80pct": max_q_80,
            "max_sustainable_qps_90pct": max_q_90,
            "max_sustainable_qps_zero_oom_80pct": _interpolate_zero_oom_slo_qps(steady_zero_oom_pts, 80.0),
            "max_sustainable_qps_zero_oom_90pct": _interpolate_zero_oom_slo_qps(steady_zero_oom_pts, 90.0),
            "steady_max_sustainable_qps_80pct": max_q_80,
            "steady_max_sustainable_qps_90pct": max_q_90,
            "full_window_max_sustainable_qps_80pct": _interpolate_slo_threshold_qps(full_pts, 80.0),
            "full_window_max_sustainable_qps_90pct": _interpolate_slo_threshold_qps(full_pts, 90.0),
            "max_grid_qps_80pct": round(grid_80, 3),
            "max_grid_qps_90pct": round(grid_90, 3),
            "min_steady_survival_rate_pct": min((c["steady_state_survival_pct"] for c in curve), default=100.0),
            "min_survival_rate_pct": min((c["survival_rate_pct"] for c in curve), default=100.0),
            "total_steady_oom_sessions": sum(c["steady_oom_sessions"] for c in curve),
            "total_oom_sessions": sum(c["oom_sessions"] for c in curve),
            "max_nvml_peak_used_mib": max((c["peak_nvml_used_mib"] for c in curve), default=0.0),
            "slo_attainment_curve": curve,
        }
    return summary


def _extract_compact_report_dict(
    rep: Any,
    *,
    init_info: dict[str, Any] | None = None,
    include_cdf_outcomes: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    d = rep.to_dict()
    err_samples = [str(s.get("error", ""))[:240] for s in rep.sessions if s.get("status") == "error"][:5]
    if err_samples:
        d["error_samples"] = err_samples
    if include_cdf_outcomes:
        d["cdf_outcomes"] = [
            {
                "status": s.get("status", "error"),
                "quadrant": s.get("quadrant", ""),
                "phase_label": s.get("phase_label", ""),
                "e2e_ms": s.get("e2e_ms", 0.0),
                "active_jct_ms": s.get("active_jct_ms", s.get("e2e_ms", 0.0)),
                "dit_turns_ms": s.get("dit_turns_ms", []),
                "ar_ttfts_ms": s.get("ar_ttfts_ms", []),
            }
            for s in rep.sessions
        ]
    d.pop("sessions", None)
    if init_info is not None:
        d["host_shmem_delta_gib"] = init_info.get("host_shmem_delta_gib", rep.host_shmem_delta_gib)
    return d, err_samples


def _compute_phase_breakdown(
    suite_sessions: Sequence[InterleavedSession],
    outcomes: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate per-phase metrics across the continuous 360s online phase-shift run."""
    phase_by_sid = {s.session_id: (s.phase_label or "unlabeled") for s in suite_sessions}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        pl = str(row.get("phase_label") or phase_by_sid.get(str(row.get("session_id", "")), "unlabeled"))
        grouped[pl].append(row)

    phase_order: list[str] = []
    for s in suite_sessions:
        pl = s.phase_label or "unlabeled"
        if pl not in phase_order:
            phase_order.append(pl)
    for pl in sorted(grouped.keys()):
        if pl not in phase_order:
            phase_order.append(pl)

    breakdown: dict[str, dict[str, Any]] = {}
    for pl in phase_order:
        rows = grouped.get(pl, [])
        n_tot = len(rows)
        ok = [r for r in rows if r.get("status") == "ok"]
        oom = [r for r in rows if r.get("status") == "oom"]
        err = [r for r in rows if r.get("status") == "error"]
        joint_ok = [r for r in ok if r.get("joint_slo_met")]
        ar_ok = [r for r in ok if r.get("ar_slo_met")]
        dit_ok = [r for r in ok if r.get("dit_slo_met")]
        e2es = [float(r.get("e2e_ms", 0.0)) for r in ok]
        act_jcts = [float(r.get("active_jct_ms", r.get("e2e_ms", 0.0))) for r in ok]
        ttfts = [float(v) for r in ok for v in r.get("ar_ttfts_ms", [])]
        tpots = [float(v) for r in ok for v in r.get("ar_tpots_ms", [])]
        dits = [float(v) for r in ok for v in r.get("dit_turns_ms", [])]
        breakdown[pl] = {
            "total_sessions": n_tot,
            "completed_sessions": len(ok),
            "oom_sessions": len(oom),
            "error_sessions": len(err),
            "survival_rate_pct": round(100.0 * len(ok) / max(1, n_tot), 1),
            "joint_slo_attainment_pct": round(100.0 * len(joint_ok) / max(1, n_tot), 1),
            "ar_slo_attainment_pct": round(100.0 * len(ar_ok) / max(1, n_tot), 1),
            "dit_slo_attainment_pct": round(100.0 * len(dit_ok) / max(1, n_tot), 1),
            "active_jct_p50_ms": _pct(act_jcts, 50),
            "active_jct_p95_ms": _pct(act_jcts, 95),
            "e2e_p95_ms": _pct(e2es, 95),
            "ar_ttft_p95_ms": _pct(ttfts, 95),
            "ar_tpot_p95_ms": _pct(tpots, 95),
            "dit_turn_p95_ms": _pct(dits, 95),
            "path_a_ships": sum(int(r.get("path_a_ships", 0)) for r in rows),
            "path_b_morphs": sum(int(r.get("path_b_morphs", 0)) for r in rows),
            "total_yields": sum(int(r.get("total_yields", 0)) for r in rows),
        }
    return breakdown


def _load_or_build_phase_shift_suite(
    bench_root: Path,
    burst_deltas: list[float],
    gentd_deltas: list[float],
    *,
    target_qps: float = 0.80,
    phase_duration_s: float = 120.0,
) -> WorkloadSuite:
    ps_path = bench_root / "suites/phase_shift_360s.json"
    if ps_path.is_file():
        loaded = WorkloadSuite.load_json(ps_path)
        if abs(float(loaded.base_qps) - float(target_qps)) < 1e-6 and abs(
            float(loaded.arrival_duration_s) - (3.0 * float(phase_duration_s))
        ) < 1.0:
            return loaded

    raw_main = _load_main_mix_suite(bench_root)
    pools: dict[str, list[InterleavedSession]] = defaultdict(list)
    for s in raw_main.sessions:
        pools[s.quadrant].append(s)
    return build_online_phase_shift_suite(
        dict(pools),
        burst_deltas,
        gentd_deltas,
        phase_duration_s=phase_duration_s,
        target_qps=target_qps,
        seed=2026,
    )


def _effective_steady_window_s(suite: WorkloadSuite) -> float:
    """Return `suite.steady_s`, expanding to `max_arrival - warmup_s` when `min_sessions` extends a low-QPS trace."""
    max_arr = max((float(s.arrival_offset_s) for s in suite.sessions), default=0.0)
    span_after_warmup = max(0.0, max_arr - float(suite.warmup_s))
    return round(max(float(suite.steady_s), span_after_warmup), 3)


def run_suite_a_qps_and_ablations(
    model_dir: Path,
    bench_root: Path,
    output_dir: Path,
    num_gpus: int = 8,
    max_sessions: int | None = None,
    min_sessions: int = DEFAULT_MIN_SESSIONS,
    arrival_duration_s: float = DEFAULT_ARRIVAL_DURATION_S,
    warmup_s: float = DEFAULT_WARMUP_S,
    steady_s: float = DEFAULT_STEADY_S,
    qps_points: tuple[float, ...] = DEFAULT_SUITE_A_QPS,
    ablation_qps: float = 0.80,
    policies: tuple[str, ...] = VALID_POLICIES,
) -> dict[str, Any]:
    """Suite A (1x8 GPUs): Interleaved-Bench v2 240s time-windowed QPS sweep + LOO M1-M4 ablations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_main = _load_main_mix_suite(bench_root)

    sources_dir = _resolve_sources_dir(bench_root)
    burst_deltas = load_burstgpt_interarrivals(sources_dir / "burstgpt_1_sample.csv")
    gentd_deltas = load_gentd26_interarrivals(sources_dir / "gentd26_lora_request_trace.csv")

    suites_by_qps: dict[float, WorkloadSuite] = {}
    for qps in qps_points:
        suites_by_qps[float(qps)] = build_time_window_suite(
            raw_main,
            float(qps),
            burst_deltas,
            gentd_deltas,
            arrival_duration_s=arrival_duration_s,
            min_sessions=min_sessions,
            max_sessions=max_sessions,
            warmup_s=warmup_s,
            steady_s=steady_s,
        )
    ablation_suite = suites_by_qps.get(float(ablation_qps)) or build_time_window_suite(
        raw_main,
        float(ablation_qps),
        burst_deltas,
        gentd_deltas,
        arrival_duration_s=arrival_duration_s,
        min_sessions=min_sessions,
        max_sessions=max_sessions,
        warmup_s=warmup_s,
        steady_s=steady_s,
    )

    v2_ov = _v2_time_window_policy_overrides(warmup_s=warmup_s, steady_s=steady_s)
    tmp_base = _local_tmp_base()
    qps_sweep_results: list[dict[str, Any]] = []

    for preset in policies:
        policy = _build_policy_config(preset, num_gpus=num_gpus, **v2_ov)
        engine = MorphServeClusterEngine(
            policy=policy,
            model_dir=model_dir,
            bench_root=bench_root,
            tmp_root=tmp_base / f"ms_m23a_{preset}",
        )
        try:
            init_info = engine.start()
            for qps in qps_points:
                active_suite = suites_by_qps[float(qps)]
                engine.policy = replace(
                    engine.policy,
                    warmup_window_s=float(active_suite.warmup_s),
                    steady_window_s=_effective_steady_window_s(active_suite),
                )
                rep = engine.run_suite(active_suite, target_qps=float(qps))
                d, err_samples = _extract_compact_report_dict(rep, init_info=init_info, include_cdf_outcomes=True)
                qps_sweep_results.append(d)
                print(
                    f"[M2.3-QPS-v2] policy={preset} n={len(active_suite.sessions)} qps={qps} "
                    f"turn_slo={getattr(rep, 'steady_turn_slo_attainment_pct', rep.steady_state_attainment_pct)}% "
                    f"turn_gp={getattr(rep, 'steady_turn_slo_goodput_rps', rep.steady_state_goodput_rps):.3f} "
                    f"sess_slo={rep.steady_state_attainment_pct}% steady_surv={rep.steady_state_survival_pct}% "
                    f"active_jct_p95={getattr(rep, 'active_jct_p95_ms', 0.0)}ms "
                    f"ar_tpot_p95={rep.ar_tpot_p95_ms}ms dit_p95={rep.dit_turn_p95_ms}ms "
                    f"swaps={getattr(rep, 'kv_host_swap_outs', 0)} "
                    f"steady_oom={getattr(rep, 'steady_oom_sessions', rep.oom_sessions)} "
                    f"oom={rep.oom_sessions} err={rep.error_sessions}"
                    + (f" sample_err={err_samples[0]!r}" if err_samples else ""),
                    flush=True,
                )
        finally:
            engine.shutdown()
            gc.collect()

    ablation_results: list[dict[str, Any]] = []
    ab_ov = _v2_time_window_policy_overrides(
        warmup_s=float(ablation_suite.warmup_s),
        steady_s=_effective_steady_window_s(ablation_suite),
    )
    for ab_name, ab_cfg in build_ablation_configs(num_gpus=num_gpus, **ab_ov):
        engine = MorphServeClusterEngine(
            policy=ab_cfg,
            model_dir=model_dir,
            bench_root=bench_root,
            tmp_root=tmp_base / f"ms_m23ab_{ab_name}",
        )
        try:
            engine.start()
            rep = engine.run_suite(ablation_suite, target_qps=float(ablation_qps))
            d, err_samples = _extract_compact_report_dict(rep, include_cdf_outcomes=False)
            d["ablation_name"] = ab_name
            ablation_results.append(d)
            print(
                f"[M2.3-ABLATION-v2] {ab_name} n={len(ablation_suite.sessions)} qps={ablation_qps} "
                f"eff_tput={getattr(rep, 'effective_slo_throughput_rps', rep.steady_state_goodput_rps):.3f} "
                f"steady_slo={rep.steady_state_attainment_pct}% steady_surv={rep.steady_state_survival_pct}% "
                f"active_jct_p95={getattr(rep, 'active_jct_p95_ms', 0.0)}ms "
                f"ar_ttft_p99={rep.ar_ttft_p99_ms}ms peak_nvml={rep.peak_nvml_used_mib}MiB "
                f"yields={rep.total_yields} swaps={getattr(rep, 'kv_host_swap_outs', 0)} "
                f"steady_oom={getattr(rep, 'steady_oom_sessions', rep.oom_sessions)} err={rep.error_sessions}"
                + (f" sample_err={err_samples[0]!r}" if err_samples else ""),
                flush=True,
            )
        finally:
            engine.shutdown()
            gc.collect()

    all_off_ref = next(
        (
            r
            for r in qps_sweep_results
            if r.get("policy_name") == "monolithic_dp" and abs(float(r.get("offered_qps", 0.0)) - ablation_qps) < 1e-6
        ),
        None,
    )

    report = {
        "schema": "morphserve.osdi27.m23_suite_a.v3",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_gpus": num_gpus,
        "steady_window_mode": "time",
        "arrival_duration_s": float(arrival_duration_s),
        "warmup_s": float(warmup_s),
        "steady_s": float(steady_s),
        "min_sessions": int(min_sessions),
        "max_sessions_cap": max_sessions,
        "sessions_by_qps": {str(q): len(suites_by_qps[float(q)].sessions) for q in qps_points},
        "ablation_sessions_per_point": len(ablation_suite.sessions),
        "ablation_design": "leave_one_out_vs_full_morphserve",
        "think_time_config": {
            "dist": v2_ov["think_time_dist"],
            "range_s": list(v2_ov["inter_turn_think_s_range"]),
            "median_s": v2_ov["think_time_median_s"],
            "sigma": v2_ov["think_time_sigma"],
        },
        "qps_points": [float(q) for q in qps_points],
        "ablation_qps": float(ablation_qps),
        "sustainable_qps_summary": _compute_sustainable_qps_summary(qps_sweep_results),
        "qps_sweep": qps_sweep_results,
        "ablations": ablation_results,
        "ablation_all_off_ref": all_off_ref,
        "status": "ok" if all(r["error_sessions"] == 0 for r in qps_sweep_results + ablation_results) else "check_failed",
    }
    out_path = output_dir / "m23-suite-a-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return report


def run_suite_a_lowqps_and_cdf(
    model_dir: Path,
    bench_root: Path,
    output_dir: Path,
    num_gpus: int = 8,
    qps_points: tuple[float, ...] = DEFAULT_LOWQPS_POINTS,
    arrival_duration_s: float = DEFAULT_ARRIVAL_DURATION_S,
    warmup_s: float = DEFAULT_WARMUP_S,
    steady_s: float = DEFAULT_STEADY_S,
    min_sessions: int = DEFAULT_MIN_SESSIONS,
    max_sessions: int | None = None,
    policies: tuple[str, ...] = VALID_POLICIES,
    merge_suite_a_report: Path | None = None,
) -> dict[str, Any]:
    """Suite A Low-to-Knee QPS Sweep (`0.05..0.80` RPS) via time-windowed `build_time_window_suite`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_main = _load_main_mix_suite(bench_root)

    sources_dir = _resolve_sources_dir(bench_root)
    burst_deltas = load_burstgpt_interarrivals(sources_dir / "burstgpt_1_sample.csv")
    gentd_deltas = load_gentd26_interarrivals(sources_dir / "gentd26_lora_request_trace.csv")

    suites_by_qps: dict[float, WorkloadSuite] = {}
    for qps in qps_points:
        eff_min = int(min_sessions) if float(qps) <= 0.15 else min(int(min_sessions), 40)
        suites_by_qps[float(qps)] = build_time_window_suite(
            raw_main,
            float(qps),
            burst_deltas,
            gentd_deltas,
            arrival_duration_s=arrival_duration_s,
            min_sessions=eff_min,
            max_sessions=max_sessions,
            warmup_s=warmup_s,
            steady_s=steady_s,
        )

    v2_ov = _v2_time_window_policy_overrides(warmup_s=warmup_s, steady_s=steady_s)
    tmp_base = _local_tmp_base()
    qps_sweep_results: list[dict[str, Any]] = []
    for preset in policies:
        policy = _build_policy_config(preset, num_gpus=num_gpus, **v2_ov)
        engine = MorphServeClusterEngine(
            policy=policy,
            model_dir=model_dir,
            bench_root=bench_root,
            tmp_root=tmp_base / f"ms_m23low_{preset}",
        )
        try:
            init_info = engine.start()
            for qps in qps_points:
                active_suite = suites_by_qps[float(qps)]
                engine.policy = replace(
                    engine.policy,
                    warmup_window_s=float(active_suite.warmup_s),
                    steady_window_s=_effective_steady_window_s(active_suite),
                )
                rep = engine.run_suite(active_suite, target_qps=float(qps))
                d, err_samples = _extract_compact_report_dict(rep, init_info=init_info, include_cdf_outcomes=True)
                qps_sweep_results.append(d)
                print(
                    f"[M2.3-LOWQPS-v2] policy={preset} n={len(active_suite.sessions)} qps={qps} "
                    f"steady_slo={rep.steady_state_attainment_pct}% steady_surv={rep.steady_state_survival_pct}% "
                    f"nvml_peak={rep.peak_nvml_used_mib}MiB "
                    f"active_jct_p95={getattr(rep, 'active_jct_p95_ms', 0.0)}ms "
                    f"all_active_jct_p95={getattr(rep, 'all_session_active_jct_p95_ms', None)} "
                    f"all_dit_p95={rep.all_session_dit_p95_ms} "
                    f"steady_oom={getattr(rep, 'steady_oom_sessions', rep.oom_sessions)} "
                    f"oom={rep.oom_sessions} err={rep.error_sessions}"
                    + (f" sample_err={err_samples[0]!r}" if err_samples else ""),
                    flush=True,
                )
        finally:
            engine.shutdown()
            gc.collect()

    combined_sweep = list(qps_sweep_results)
    if merge_suite_a_report is not None and merge_suite_a_report.is_file():
        prev = json.loads(merge_suite_a_report.read_text(encoding="utf-8"))
        seen_keys = {(str(r["policy_name"]), round(float(r["offered_qps"]), 3)) for r in combined_sweep}
        for old_r in prev.get("qps_sweep", []):
            key = (str(old_r["policy_name"]), round(float(old_r["offered_qps"]), 3))
            if key not in seen_keys:
                combined_sweep.append(old_r)

    report = {
        "schema": "morphserve.osdi27.m23_lowqps_slo_cdf.v2",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary_metric": "steady_state_slo_attainment_pct_vs_offered_qps",
        "num_gpus": num_gpus,
        "steady_window_mode": "time",
        "arrival_duration_s": float(arrival_duration_s),
        "warmup_s": float(warmup_s),
        "steady_s": float(steady_s),
        "min_sessions": int(min_sessions),
        "sessions_by_qps": {str(q): len(suites_by_qps[float(q)].sessions) for q in qps_points},
        "qps_points": [float(q) for q in qps_points],
        "sustainable_qps_summary": _compute_sustainable_qps_summary(combined_sweep),
        "qps_sweep": qps_sweep_results,
        "status": "ok" if all(r["error_sessions"] == 0 for r in qps_sweep_results) else "check_failed",
    }
    out_path = output_dir / "m23-lowqps-slo-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return report


def run_think_time_sweep(
    model_dir: Path,
    bench_root: Path,
    output_dir: Path,
    num_gpus: int = 8,
    offered_qps: float = 0.60,
    think_times_s: tuple[float, ...] = DEFAULT_THINK_TIMES_S,
    arrival_duration_s: float = DEFAULT_ARRIVAL_DURATION_S,
    warmup_s: float = DEFAULT_WARMUP_S,
    steady_s: float = DEFAULT_STEADY_S,
    min_sessions: int = DEFAULT_MIN_SESSIONS,
    max_sessions: int | None = None,
    policies: tuple[str, ...] = DEFAULT_THINK_TIME_POLICIES,
) -> dict[str, Any]:
    """Fixed knee-QPS sweep of constant inter-turn think times `(0, 5, 20, 60)s` verifying KV residency & swap scaling."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_main = _load_main_mix_suite(bench_root)

    sources_dir = _resolve_sources_dir(bench_root)
    burst_deltas = load_burstgpt_interarrivals(sources_dir / "burstgpt_1_sample.csv")
    gentd_deltas = load_gentd26_interarrivals(sources_dir / "gentd26_lora_request_trace.csv")

    window_suite = build_time_window_suite(
        raw_main,
        float(offered_qps),
        burst_deltas,
        gentd_deltas,
        arrival_duration_s=arrival_duration_s,
        min_sessions=min_sessions,
        max_sessions=max_sessions,
        warmup_s=warmup_s,
        steady_s=steady_s,
    )

    tmp_base = _local_tmp_base()
    sweep_results: list[dict[str, Any]] = []
    summary_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for preset in policies:
        init_ov = _v2_time_window_policy_overrides(
            warmup_s=float(window_suite.warmup_s),
            steady_s=_effective_steady_window_s(window_suite),
            think_range=(0.0, 0.0),
            think_dist="constant",
            think_median_s=0.0,
            think_sigma=0.0,
        )
        policy = _build_policy_config(preset, num_gpus=num_gpus, **init_ov)
        engine = MorphServeClusterEngine(
            policy=policy,
            model_dir=model_dir,
            bench_root=bench_root,
            tmp_root=tmp_base / f"ms_m23think_{preset}",
        )
        try:
            init_info = engine.start()
            for t_think in think_times_s:
                t_val = max(0.0, float(t_think))
                engine.policy = replace(
                    engine.policy,
                    inter_turn_think_s_range=(t_val, t_val),
                    think_time_dist="constant",
                    think_time_median_s=t_val,
                    think_time_sigma=0.0,
                    steady_window_mode="time",
                    warmup_window_s=float(window_suite.warmup_s),
                    steady_window_s=_effective_steady_window_s(window_suite),
                )
                rep = engine.run_suite(window_suite, target_qps=float(offered_qps))
                d, err_samples = _extract_compact_report_dict(rep, init_info=init_info, include_cdf_outcomes=False)
                d["think_time_s"] = t_val
                sweep_results.append(d)

                row_summary = {
                    "think_time_s": t_val,
                    "kv_host_swap_outs": int(getattr(rep, "kv_host_swap_outs", 0)),
                    "steady_oom_sessions": int(getattr(rep, "steady_oom_sessions", rep.oom_sessions)),
                    "oom_sessions": int(rep.oom_sessions),
                    "steady_state_survival_pct": float(rep.steady_state_survival_pct),
                    "steady_state_attainment_pct": float(rep.steady_state_attainment_pct),
                    "active_jct_p95_ms": float(getattr(rep, "active_jct_p95_ms", 0.0)),
                    "all_session_active_jct_p95_ms": getattr(rep, "all_session_active_jct_p95_ms", None),
                    "per_turn_stats": getattr(rep, "per_turn_stats", {}),
                    "peak_nvml_used_mib": float(rep.peak_nvml_used_mib),
                }
                summary_by_policy[preset].append(row_summary)
                print(
                    f"[M2.3-THINK] policy={preset} qps={offered_qps} think_s={t_val:.1f} n={len(window_suite.sessions)} "
                    f"steady_slo={rep.steady_state_attainment_pct}% steady_surv={rep.steady_state_survival_pct}% "
                    f"active_jct_p95={getattr(rep, 'active_jct_p95_ms', 0.0)}ms "
                    f"swaps={getattr(rep, 'kv_host_swap_outs', 0)} nvml_peak={rep.peak_nvml_used_mib}MiB "
                    f"steady_oom={getattr(rep, 'steady_oom_sessions', rep.oom_sessions)} err={rep.error_sessions}"
                    + (f" sample_err={err_samples[0]!r}" if err_samples else ""),
                    flush=True,
                )
        finally:
            engine.shutdown()
            gc.collect()

    report = {
        "schema": "morphserve.osdi27.m23_think_time_sweep.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_gpus": num_gpus,
        "offered_qps": float(offered_qps),
        "arrival_duration_s": float(window_suite.arrival_duration_s),
        "warmup_s": float(window_suite.warmup_s),
        "steady_s": float(window_suite.steady_s),
        "num_sessions": len(window_suite.sessions),
        "think_times_s": [float(t) for t in think_times_s],
        "policies": list(policies),
        "summary_by_policy": dict(summary_by_policy),
        "think_time_sweep": sweep_results,
        "status": "ok" if all(r["error_sessions"] == 0 for r in sweep_results) else "check_failed",
    }
    out_path = output_dir / "m23-think-time-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return report


def run_suite_b_skews_and_parity(
    model_dir: Path,
    bench_root: Path,
    output_dir: Path,
    num_gpus: int = 8,
    phase_shift_qps: float = 0.80,
    phase_shift_policies: tuple[str, ...] = DEFAULT_PHASE_SHIFT_POLICIES,
    include_static_skews: bool = False,
    skew_sessions: int = DEFAULT_SKEW_SESSIONS,
    skew_qps: float = 0.80,
    policies: tuple[str, ...] = VALID_POLICIES,
    allow_eval_download: bool = False,
    arrival_duration_s: float = DEFAULT_ARRIVAL_DURATION_S,
    warmup_s: float = DEFAULT_WARMUP_S,
    steady_s: float = DEFAULT_STEADY_S,
    min_sessions: int = DEFAULT_MIN_SESSIONS,
) -> dict[str, Any]:
    """Suite B (1x8 GPUs): Part 1 Continuous 360s Online Phase-Shift + Part 2 Two-Tier Accuracy & Lossless Suite."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sources_dir = _resolve_sources_dir(bench_root)
    burst_deltas = load_burstgpt_interarrivals(sources_dir / "burstgpt_1_sample.csv")
    gentd_deltas = load_gentd26_interarrivals(sources_dir / "gentd26_lora_request_trace.csv")

    tmp_base = _local_tmp_base()

    # ---- Part 1: Continuous 360s Online Modality Phase-Shift (3 x 120s without restarting workers) ----
    phase_suite = _load_or_build_phase_shift_suite(
        bench_root,
        burst_deltas,
        gentd_deltas,
        target_qps=phase_shift_qps,
        phase_duration_s=120.0,
    )
    phase_ov = _v2_time_window_policy_overrides(
        warmup_s=float(phase_suite.warmup_s),
        steady_s=float(phase_suite.steady_s),
    )

    phase_shift_results: list[dict[str, Any]] = []
    for preset in phase_shift_policies:
        policy = _build_policy_config(preset, num_gpus=num_gpus, **phase_ov)
        engine = MorphServeClusterEngine(
            policy=policy,
            model_dir=model_dir,
            bench_root=bench_root,
            tmp_root=tmp_base / f"ms_m23b_ps_{preset}",
        )
        try:
            init_info = engine.start()
            # Pass `target_qps=None` and no deltas so `phase_suite`'s exact 3-phase arrival schedule is preserved!
            rep = engine.run_suite(phase_suite)
            per_phase = _compute_phase_breakdown(phase_suite.sessions, rep.sessions)
            d, err_samples = _extract_compact_report_dict(rep, init_info=init_info, include_cdf_outcomes=False)
            d["per_phase"] = per_phase
            d["time_series_10s"] = list(getattr(rep, "time_series_10s", []))
            phase_shift_results.append(d)
            print(
                f"[M2.3-PHASE-SHIFT] policy={preset} n={len(phase_suite.sessions)} qps={phase_suite.base_qps} "
                f"eff_tput={getattr(rep, 'effective_slo_throughput_rps', rep.steady_state_goodput_rps):.3f} "
                f"steady_slo={rep.steady_state_attainment_pct}% steady_surv={rep.steady_state_survival_pct}% "
                f"active_jct_p95={getattr(rep, 'active_jct_p95_ms', 0.0)}ms "
                f"ship={rep.path_a_ships} morph={rep.path_b_morphs} "
                f"steady_oom={getattr(rep, 'steady_oom_sessions', rep.oom_sessions)} err={rep.error_sessions}"
                + (f" sample_err={err_samples[0]!r}" if err_samples else ""),
                flush=True,
            )
        finally:
            engine.shutdown()
            gc.collect()

    # Optional static skew sweep when `--include-static-skews` is explicitly requested
    skew_results: list[dict[str, Any]] = []
    if include_static_skews:
        skew_files = (
            ("ar80_dit20", "suites/skew_ar80_dit20.json"),
            ("ar50_dit50", "suites/skew_ar50_dit50.json"),
            ("ar20_dit80", "suites/skew_ar20_dit80.json"),
        )
        skew_suites = [
            (
                tag,
                build_time_window_suite(
                    WorkloadSuite.load_json(bench_root / rel),
                    float(skew_qps),
                    burst_deltas,
                    gentd_deltas,
                    arrival_duration_s=arrival_duration_s,
                    min_sessions=min_sessions,
                    max_sessions=skew_sessions if skew_sessions > 0 else None,
                    warmup_s=warmup_s,
                    steady_s=steady_s,
                ),
            )
            for tag, rel in skew_files
        ]
        sk_ov = _v2_time_window_policy_overrides(warmup_s=warmup_s, steady_s=steady_s)
        for preset in policies:
            policy = _build_policy_config(preset, num_gpus=num_gpus, **sk_ov)
            engine = MorphServeClusterEngine(
                policy=policy,
                model_dir=model_dir,
                bench_root=bench_root,
                tmp_root=tmp_base / f"ms_m23b_sk_{preset}",
            )
            try:
                engine.start()
                for skew_tag, sk_suite in skew_suites:
                    engine.policy = replace(
                        engine.policy,
                        warmup_window_s=float(sk_suite.warmup_s),
                        steady_window_s=float(sk_suite.steady_s),
                    )
                    rep = engine.run_suite(sk_suite, target_qps=float(skew_qps))
                    d, err_samples = _extract_compact_report_dict(rep, include_cdf_outcomes=False)
                    d["skew_tag"] = skew_tag
                    skew_results.append(d)
                    print(
                        f"[M2.3-SKEW] policy={preset} skew={skew_tag} n={len(sk_suite.sessions)} "
                        f"steady_goodput={rep.steady_state_goodput_rps:.3f} "
                        f"steady_slo={rep.steady_state_attainment_pct}% ship={rep.path_a_ships} morph={rep.path_b_morphs} "
                        f"oom={rep.oom_sessions} err={rep.error_sessions}"
                        + (f" sample_err={err_samples[0]!r}" if err_samples else ""),
                        flush=True,
                    )
            finally:
                engine.shutdown()
                gc.collect()

    # ---- Part 2: Two-Tier Accuracy & Lossless Suite (144 sessions across all 4 v2 quadrants) ----
    #   1. `monolithic_dp` (reference)
    #   2. `strict_lossless`: `morphserve` with `allow_cascade_cfg_split=False` (hard gate: 100% AR exact + SSIM == 1.0)
    #   3. `cascade_cfg_fast`: `morphserve` with `allow_cascade_cfg_split=True` (quality-latency Pareto, non-gating on SHA256)
    acc_suite = build_m3_active_accuracy_suite(bench_root)
    tmp_acc_root = tmp_base / "ms_m23_parity_outputs"
    shutil.rmtree(tmp_acc_root, ignore_errors=True)
    parity_runs: dict[str, dict[str, dict[str, Any]]] = {}
    tier_latencies: dict[str, dict[str, Any]] = {}

    tier_specs = (
        ("monolithic_dp", "monolithic_dp", {"allow_cascade_cfg_split": False}),
        ("strict_lossless", "morphserve", {"allow_cascade_cfg_split": False}),
        ("cascade_cfg_fast", "morphserve", {"allow_cascade_cfg_split": True}),
    )

    try:
        for tier_label, preset_name, extra_ov in tier_specs:
            pol = EnginePolicyConfig.from_preset(
                preset_name,
                num_gpus=num_gpus,
                inter_turn_think_s_range=(0.0, 0.0),
                think_time_dist="constant",
                think_time_median_s=0.0,
                think_time_sigma=0.0,
                steady_window_mode="session_trim",
                steady_state_trim_frac=0.0,
                **extra_ov,
            )
            eng = MorphServeClusterEngine(
                policy=pol,
                model_dir=model_dir,
                bench_root=bench_root,
                tmp_root=tmp_base / f"ms_m23_acc_{tier_label}",
            )
            try:
                eng.start()
                rep = eng.run_suite(
                    acc_suite,
                    target_qps=0.20,
                    save_outputs_dir=tmp_acc_root / tier_label,
                )
                parity_runs[tier_label] = {s["session_id"]: s for s in rep.sessions}
                tier_latencies[tier_label] = {
                    "policy_preset": preset_name,
                    "allow_cascade_cfg_split": bool(extra_ov["allow_cascade_cfg_split"]),
                    "wall_seconds": round(float(rep.wall_seconds), 2),
                    "completed_sessions": int(rep.completed_sessions),
                    "oom_sessions": int(rep.oom_sessions),
                    "error_sessions": int(rep.error_sessions),
                    "ar_ttft_p50_ms": float(rep.ar_ttft_p50_ms),
                    "ar_tpot_p50_ms": float(rep.ar_tpot_p50_ms),
                    "ar_tpot_p95_ms": float(rep.ar_tpot_p95_ms),
                    "dit_turn_p50_ms": float(rep.dit_turn_p50_ms),
                    "dit_turn_p95_ms": float(rep.dit_turn_p95_ms),
                    "active_jct_p50_ms": float(getattr(rep, "active_jct_p50_ms", rep.session_e2e_p50_ms)),
                    "active_jct_p95_ms": float(getattr(rep, "active_jct_p95_ms", rep.session_e2e_p95_ms)),
                }
                acc_err = [str(s.get("error", ""))[:240] for s in rep.sessions if s.get("status") == "error"][:5]
                print(
                    f"[M2.3-ACCURACY] tier={tier_label} preset={preset_name} "
                    f"cascade_cfg={extra_ov['allow_cascade_cfg_split']} n={len(acc_suite.sessions)} "
                    f"ok={rep.completed_sessions} dit_p50={rep.dit_turn_p50_ms}ms dit_p95={rep.dit_turn_p95_ms}ms "
                    f"oom={rep.oom_sessions} err={rep.error_sessions}"
                    + (f" sample_err={acc_err[0]!r}" if acc_err else ""),
                    flush=True,
                )
            finally:
                eng.shutdown()
                gc.collect()

        evaluator = ExactnessEvaluator(
            bench_root=bench_root,
            allow_hf_download=allow_eval_download,
        )

        def _summarize_tier(cand_key: str, *, strict_ssim_1: bool) -> dict[str, Any]:
            ev = evaluator.evaluate_parity_and_quality(
                acc_suite.sessions,
                list(parity_runs["monolithic_dp"].values()),
                list(parity_runs[cand_key].values()),
                bench_root=bench_root,
                strict_ssim_1=strict_ssim_1,
            )
            geneval_metrics: list[dict[str, Any]] = []
            for s in acc_suite.sessions:
                if s.source_dataset != "geneval":
                    continue
                sid = s.session_id
                ref_s = parity_runs["monolithic_dp"].get(sid, {})
                ms_s = parity_runs[cand_key].get(sid, {})
                ref_imgs = ref_s.get("image_paths_out") or []
                ms_imgs = ms_s.get("image_paths_out") or []
                if ref_imgs and ms_imgs and Path(ref_imgs[0]).is_file() and Path(ms_imgs[0]).is_file():
                    m = image_pair_metrics(Path(ref_imgs[0]), Path(ms_imgs[0]))
                    m["session_id"] = sid
                    geneval_metrics.append(m)

            ref_errs = [
                f"{sid}: {str(s.get('error', ''))[:200]}"
                for sid, s in parity_runs["monolithic_dp"].items()
                if s.get("status") == "error"
            ]
            cand_errs = [
                f"{sid}: {str(s.get('error', ''))[:200]}"
                for sid, s in parity_runs[cand_key].items()
                if s.get("status") == "error"
            ]
            return {
                **ev,
                "tier_name": cand_key,
                "gating": bool(strict_ssim_1),
                "accuracy_suite_name": acc_suite.name,
                "accuracy_total_sessions": len(acc_suite.sessions),
                "ref_error_sessions": len(ref_errs),
                "morphserve_error_sessions": len(cand_errs),
                "ref_error_samples": ref_errs[:5],
                "morphserve_error_samples": cand_errs[:5],
                "interleaved_co_creation_sessions": sum(
                    1 for s in acc_suite.sessions if s.quadrant == "interleaved_co_creation"
                ),
                "multiturn_4turn_edit_sessions": sum(
                    1 for s in acc_suite.sessions if s.quadrant == "multiturn_edit" and len(s.turns) == 4
                ),
                "multiturn_mmvet_sessions": sum(1 for s in acc_suite.sessions if s.quadrant == "multi_image_vqa"),
                "visual_cot_highres_sessions": sum(
                    1 for s in acc_suite.sessions if s.quadrant in ("visual_cot_highres", "visual_cot")
                ),
                "geneval_sessions": len(geneval_metrics),
                "geneval_sha256_exact_count": sum(1 for m in geneval_metrics if m["sha256_equal"]),
                "geneval_pairs": geneval_metrics,
                "latency": tier_latencies[cand_key],
            }

        strict_summary = _summarize_tier("strict_lossless", strict_ssim_1=True)
        fast_summary = _summarize_tier("cascade_cfg_fast", strict_ssim_1=False)
        ref_dit_p50 = max(1e-6, float(tier_latencies["monolithic_dp"]["dit_turn_p50_ms"]))
        strict_dit_p50 = max(1e-6, float(tier_latencies["strict_lossless"]["dit_turn_p50_ms"]))
        fast_dit_p50 = max(1e-6, float(tier_latencies["cascade_cfg_fast"]["dit_turn_p50_ms"]))
        pareto_comparison = {
            "strict_lossless_dit_p50_ms": tier_latencies["strict_lossless"]["dit_turn_p50_ms"],
            "cascade_cfg_fast_dit_p50_ms": tier_latencies["cascade_cfg_fast"]["dit_turn_p50_ms"],
            "monolithic_dp_dit_p50_ms": tier_latencies["monolithic_dp"]["dit_turn_p50_ms"],
            "cascade_cfg_speedup_vs_strict_x": round(strict_dit_p50 / fast_dit_p50, 3),
            "cascade_cfg_speedup_vs_monolithic_x": round(ref_dit_p50 / fast_dit_p50, 3),
            "strict_lossless_pass": bool(strict_summary["lossless_pass"]),
            "cascade_cfg_fast_mean_ssim": fast_summary["image_mean_ssim"],
            "cascade_cfg_fast_chained_mean_ssim": fast_summary["chained_multiturn_mean_ssim"],
            "cascade_cfg_fast_post_dit_ar_exact_pct": fast_summary["post_dit_ar_exact_match_pct"],
            "cascade_cfg_fast_geneval_delta_pct": fast_summary["geneval_delta_pct"],
            "cascade_cfg_fast_clip_t_delta_pct": fast_summary["clip_t_delta_pct"],
        }
    finally:
        shutil.rmtree(tmp_acc_root, ignore_errors=True)

    all_phase_ok = all(r["error_sessions"] == 0 for r in phase_shift_results)
    all_skew_ok = all(r["error_sessions"] == 0 for r in skew_results)
    strict_gate_ok = bool(
        strict_summary["lossless_pass"]
        and strict_summary["ref_error_sessions"] == 0
        and strict_summary["morphserve_error_sessions"] == 0
    )

    report = {
        "schema": "morphserve.osdi27.m23_suite_b.v3",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_gpus": num_gpus,
        "phase_shift_360s": {
            "suite_name": phase_suite.name,
            "offered_qps": float(phase_suite.base_qps),
            "arrival_duration_s": float(phase_suite.arrival_duration_s),
            "warmup_s": float(phase_suite.warmup_s),
            "steady_s": float(phase_suite.steady_s),
            "num_sessions": len(phase_suite.sessions),
            "phase_schedule": list(phase_suite.phase_schedule),
            "policies": list(phase_shift_policies),
            "results": phase_shift_results,
        },
        "include_static_skews": bool(include_static_skews),
        "skew_qps": float(skew_qps),
        "skew_sessions_per_point": int(skew_sessions),
        "skew_sweep": skew_results,
        "lossless_parity": strict_summary,
        "two_tier_accuracy": {
            "suite_name": acc_suite.name,
            "total_sessions": len(acc_suite.sessions),
            "reference_latency": tier_latencies["monolithic_dp"],
            "strict_lossless": strict_summary,
            "cascade_cfg_fast": fast_summary,
            "pareto_comparison": pareto_comparison,
        },
        "status": "ok" if (all_phase_ok and all_skew_ok and strict_gate_ok) else "check_failed",
    }
    out_path = output_dir / "m23-suite-b-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return report


def _parse_float_csv(raw: str | None) -> tuple[float, ...] | None:
    if not raw:
        return None
    vals = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    return vals if vals else None


def _parse_str_csv(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    vals = tuple(x.strip() for x in raw.split(",") if x.strip())
    return vals if vals else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MorphServe Phase-2 M2.3 BAGEL-7B-MoT Interleaved-Bench v2 Runner")
    parser.add_argument(
        "--mode",
        choices=("suite_a", "suite_a_lowqps", "think_time_sweep", "suite_b"),
        required=True,
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--arrival-duration-s", type=float, default=DEFAULT_ARRIVAL_DURATION_S)
    parser.add_argument("--warmup-s", type=float, default=DEFAULT_WARMUP_S)
    parser.add_argument("--steady-s", type=float, default=DEFAULT_STEADY_S)
    parser.add_argument("--min-sessions", type=int, default=DEFAULT_MIN_SESSIONS)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--low-sessions", type=int, default=None, help="Deprecated legacy option (kept for CLI compat)")
    parser.add_argument("--knee-sessions", type=int, default=None, help="Deprecated legacy option (kept for CLI compat)")
    parser.add_argument("--qps-points", type=str, default=None)
    parser.add_argument("--ablation-qps", type=float, default=0.80)
    parser.add_argument("--policies", type=str, default=None)
    parser.add_argument("--think-qps", type=float, default=None)
    parser.add_argument("--think-times", type=str, default="0.0,5.0,20.0,60.0")
    parser.add_argument("--phase-shift-qps", type=float, default=None)
    parser.add_argument("--include-static-skews", action="store_true")
    parser.add_argument("--skew-qps", type=float, default=0.80)
    parser.add_argument("--skew-sessions", type=int, default=None)
    parser.add_argument("--merge-suite-a-report", type=Path, default=None)
    parser.add_argument("--allow-eval-download", action="store_true")
    args = parser.parse_args(argv)

    parsed_qps = _parse_float_csv(args.qps_points)
    parsed_policies = _parse_str_csv(args.policies)

    if args.mode == "suite_a":
        rep = run_suite_a_qps_and_ablations(
            model_dir=args.model_dir,
            bench_root=args.bench_root,
            output_dir=args.output_dir,
            num_gpus=args.num_gpus,
            max_sessions=args.max_sessions,
            min_sessions=args.min_sessions,
            arrival_duration_s=args.arrival_duration_s,
            warmup_s=args.warmup_s,
            steady_s=args.steady_s,
            qps_points=parsed_qps or DEFAULT_SUITE_A_QPS,
            ablation_qps=args.ablation_qps,
            policies=parsed_policies or VALID_POLICIES,
        )
    elif args.mode == "suite_a_lowqps":
        rep = run_suite_a_lowqps_and_cdf(
            model_dir=args.model_dir,
            bench_root=args.bench_root,
            output_dir=args.output_dir,
            num_gpus=args.num_gpus,
            qps_points=parsed_qps or DEFAULT_LOWQPS_POINTS,
            arrival_duration_s=args.arrival_duration_s,
            warmup_s=args.warmup_s,
            steady_s=args.steady_s,
            min_sessions=args.min_sessions,
            max_sessions=args.max_sessions,
            policies=parsed_policies or VALID_POLICIES,
            merge_suite_a_report=args.merge_suite_a_report,
        )
    elif args.mode == "think_time_sweep":
        think_vals = _parse_float_csv(args.think_times) or DEFAULT_THINK_TIMES_S
        eff_think_qps = float(args.think_qps) if args.think_qps is not None else (float(parsed_qps[0]) if parsed_qps else 0.60)
        rep = run_think_time_sweep(
            model_dir=args.model_dir,
            bench_root=args.bench_root,
            output_dir=args.output_dir,
            num_gpus=args.num_gpus,
            offered_qps=eff_think_qps,
            think_times_s=think_vals,
            arrival_duration_s=args.arrival_duration_s,
            warmup_s=args.warmup_s,
            steady_s=args.steady_s,
            min_sessions=args.min_sessions,
            max_sessions=args.max_sessions,
            policies=parsed_policies or DEFAULT_THINK_TIME_POLICIES,
        )
    else:
        eff_skew = int(args.skew_sessions) if args.skew_sessions is not None else (
            int(args.max_sessions) if args.max_sessions is not None else DEFAULT_SKEW_SESSIONS
        )
        eff_ps_qps = (
            float(args.phase_shift_qps)
            if args.phase_shift_qps is not None
            else (float(parsed_qps[0]) if parsed_qps else 0.80)
        )
        rep = run_suite_b_skews_and_parity(
            model_dir=args.model_dir,
            bench_root=args.bench_root,
            output_dir=args.output_dir,
            num_gpus=args.num_gpus,
            phase_shift_qps=eff_ps_qps,
            phase_shift_policies=parsed_policies or DEFAULT_PHASE_SHIFT_POLICIES,
            include_static_skews=args.include_static_skews,
            skew_sessions=eff_skew,
            skew_qps=args.skew_qps,
            policies=parsed_policies or VALID_POLICIES,
            allow_eval_download=args.allow_eval_download,
            arrival_duration_s=args.arrival_duration_s,
            warmup_s=args.warmup_s,
            steady_s=args.steady_s,
            min_sessions=args.min_sessions,
        )
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
