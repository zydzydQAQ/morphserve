"""Shared benchmark harness, `BenchmarkRunner` lifecycle coordinator, and canonical telemetry aggregation."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import gc
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence

from morphserve.engine.workload import InterleavedSession, WorkloadSuite
from morphserve.telemetry import percentile as _pct

if TYPE_CHECKING:
    from morphserve.engine.config import EnginePolicyConfig, RunWindowOptions


def resolve_sources_dir(bench_root: str | Path) -> Path:
    """Resolve the Interleaved-Bench raw sources directory beneath or beside `bench_root`."""
    root = Path(bench_root)
    for cand in (root / "sources", root.parent / "bench_sources"):
        if cand.is_dir():
            return cand
    return root / "sources"


def local_tmp_base() -> Path:
    """Return container-local scratch directory (`MORPHSERVE_LOCAL_TMP` or `/tmp`)."""
    raw = os.environ.get("MORPHSERVE_LOCAL_TMP", "").strip()
    return Path(raw) if raw else Path("/tmp")


def resolve_default_main_suite_path(bench_root: str | Path) -> Path:
    """Resolve the default Interleaved-Bench main mix JSON (preferring v2 over v1)."""
    root = Path(bench_root)
    for rel in ("suites/main_mix_v2_80.json", "suites/main_mix_80.json"):
        cand = root / rel
        if cand.is_file():
            return cand
    return root / "suites/main_mix_v2_80.json"


def load_main_mix_suite(path: str | Path) -> WorkloadSuite:
    """Load the Interleaved-Bench main mix from an explicit JSON path or a `bench_root` directory."""
    target = Path(path)
    if target.is_dir():
        resolved = resolve_default_main_suite_path(target)
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Missing main_mix_v2_80.json / main_mix_80.json under {target / 'suites'}"
            )
        return WorkloadSuite.load_json(resolved)
    if not target.is_file():
        raise FileNotFoundError(f"Missing WorkloadSuite JSON: {target}")
    return WorkloadSuite.load_json(target)


def effective_steady_window_s(suite: WorkloadSuite) -> float:
    """Return steady-state window duration strictly excluding leading `warmup_s` and trailing `cooldown_s`."""
    max_arr = max((float(s.arrival_offset_s) for s in suite.sessions), default=0.0)
    warmup_s = max(0.0, float(suite.warmup_s))
    cooldown_s = max(0.0, float(getattr(suite, "cooldown_s", 30.0)))
    return round(max(1.0, max_arr - warmup_s - cooldown_s), 3)


def parse_csv_floats(raw: str | None) -> list[float]:
    """Parse a comma-separated list of floats (returns `[]` when empty or `None`)."""
    if not raw:
        return []
    return [float(tok.strip()) for tok in str(raw).split(",") if tok.strip()]


def parse_csv_strings(raw: str | None) -> list[str]:
    """Parse a comma-separated list of non-empty strings (returns `[]` when empty or `None`)."""
    if not raw:
        return []
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def compact_report_dict(
    report: Any,
    *,
    max_error_samples: int = 8,
    include_cdf_outcomes: bool = False,
    include_failed_sessions: bool = False,
    init_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strip bulk per-session records from a `ClusterRunReport` (or dict) while preserving error & CDF samples."""
    d = dict(report.to_dict()) if hasattr(report, "to_dict") else dict(report)
    raw_sessions = d.pop("sessions", None)
    if raw_sessions is None and hasattr(report, "sessions"):
        raw_sessions = list(report.sessions)
    sessions_list = list(raw_sessions or ())

    limit = max(0, int(max_error_samples))
    err_samples = [str(s.get("error", ""))[:240] for s in sessions_list if s.get("status") == "error"][:limit]
    if err_samples:
        d["error_samples"] = err_samples

    if include_failed_sessions:
        failed = [
            {
                "session_id": str(s.get("session_id", "")),
                "quadrant": str(s.get("quadrant", "")),
                "status": str(s.get("status", "")),
                "error": str(s.get("error", ""))[:400],
            }
            for s in sessions_list
            if s.get("status") != "ok"
        ][:limit]
        if failed:
            d["failed_sessions"] = failed

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
            for s in sessions_list
        ]

    if init_info is not None and "host_shmem_delta_gib" in init_info:
        d["host_shmem_delta_gib"] = init_info["host_shmem_delta_gib"]
    return d


def compute_phase_breakdown(
    sessions: Sequence[dict[str, Any]],
    suite_sessions: Sequence[InterleavedSession] | None = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate per-phase metrics across a continuous online modality phase-shift run."""
    phase_by_sid: dict[str, str] = {}
    phase_order: list[str] = []
    if suite_sessions is not None:
        for s in suite_sessions:
            pl = s.phase_label or "unlabeled"
            phase_by_sid[s.session_id] = pl
            if pl not in phase_order:
                phase_order.append(pl)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sessions:
        pl = str(row.get("phase_label") or phase_by_sid.get(str(row.get("session_id", "")), "unlabeled"))
        grouped[pl].append(row)
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


def compute_quadrant_breakdown(sessions: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate per-quadrant completion, OOM, SLO attainment, and tail latency statistics."""
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sessions:
        by_q[str(row.get("quadrant", "unknown"))].append(row)

    out: dict[str, dict[str, Any]] = {}
    for q, items in sorted(by_q.items()):
        ok = [x for x in items if x.get("status") == "ok"]
        joint = [x for x in ok if x.get("joint_slo_met")]
        e2es = [float(x.get("e2e_ms", 0.0)) for x in ok]
        act_jcts = [float(x.get("active_jct_ms", x.get("e2e_ms", 0.0))) for x in ok]
        tpots = [float(v) for x in ok for v in x.get("ar_tpots_ms", [])]
        dits = [float(v) for x in ok for v in x.get("dit_turns_ms", [])]
        out[q] = {
            "count": len(items),
            "completed_sessions": len(ok),
            "joint_slo_pct": round(100.0 * len(joint) / max(1, len(items)), 1),
            "oom_count": sum(1 for x in items if x.get("status") == "oom"),
            "error_count": sum(1 for x in items if x.get("status") == "error"),
            "e2e_p50_ms": _pct(e2es, 50),
            "e2e_p95_ms": _pct(e2es, 95),
            "active_jct_p95_ms": _pct(act_jcts, 95),
            "ar_tpot_p95_ms": _pct(tpots, 95),
            "dit_turn_p95_ms": _pct(dits, 95),
        }
    return out


def interpolate_slo_threshold_qps(
    points: Sequence[tuple[float, ...]],
    *,
    threshold_pct: float = 90.0,
    require_zero_oom: bool = False,
) -> float:
    """Compute highest sustainable QPS where piecewise-linear SLO attainment >= `threshold_pct`.

    Accepts `(qps, attainment_pct)` pairs or `(qps, attainment_pct, oom_sessions)` triples.
    When `require_zero_oom=True`, only zero-OOM points qualify and interpolation never enters an OOM point.
    """
    if not points:
        return 0.0

    normalized: list[tuple[float, float, int]] = []
    for pt in points:
        q = float(pt[0])
        att = float(pt[1])
        oom = int(pt[2]) if len(pt) >= 3 else 0
        normalized.append((q, att, oom))
    normalized.sort(key=lambda x: x[0])

    best_q = 0.0
    for idx, (q_cur, att_cur, oom_cur) in enumerate(normalized):
        if require_zero_oom and oom_cur != 0:
            continue
        if att_cur >= threshold_pct:
            best_q = max(best_q, q_cur)
            if idx + 1 < len(normalized):
                q_nxt, att_nxt, oom_nxt = normalized[idx + 1]
                nxt_ok = (not require_zero_oom) or (oom_nxt == 0)
                if nxt_ok and att_nxt < threshold_pct and att_cur > att_nxt and q_nxt > q_cur:
                    frac = (att_cur - threshold_pct) / (att_cur - att_nxt)
                    best_q = max(best_q, q_cur + frac * (q_nxt - q_cur))
    return round(best_q, 4)


def compute_sustainable_qps_summary(sweep_results: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize canonical steady-state turn/session SLO attainment, zero-OOM sustainable QPS & Active-JCT curves."""
    by_pol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sweep_results:
        by_pol[str(row["policy_name"])].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for pol, rows in sorted(by_pol.items()):
        rows_sorted = sorted(rows, key=lambda r: float(r["offered_qps"]))

        steady_pts = [(float(r["offered_qps"]), float(r["joint_slo_attainment_pct"])) for r in rows_sorted]
        steady_turn_pts = [
            (float(r["offered_qps"]), float(r["steady_turn_slo_attainment_pct"])) for r in rows_sorted
        ]
        steady_zero_oom_pts = [
            (float(r["offered_qps"]), float(r["joint_slo_attainment_pct"]), int(r["steady_oom_sessions"]))
            for r in rows_sorted
        ]
        steady_turn_zero_oom_pts = [
            (
                float(r["offered_qps"]),
                float(r["steady_turn_slo_attainment_pct"]),
                int(r["steady_oom_sessions"]),
            )
            for r in rows_sorted
        ]
        full_pts = [
            (
                float(r["offered_qps"]),
                float(r.get("full_trace_joint_slo_attainment_pct", r["joint_slo_attainment_pct"])),
            )
            for r in rows_sorted
        ]

        grid_80 = max((q for q, a in steady_pts if a >= 80.0), default=0.0)
        grid_90 = max((q for q, a in steady_pts if a >= 90.0), default=0.0)
        turn_grid_80 = max((q for q, a in steady_turn_pts if a >= 80.0), default=0.0)
        turn_grid_90 = max((q for q, a in steady_turn_pts if a >= 90.0), default=0.0)

        curve: list[dict[str, Any]] = []
        for r in rows_sorted:
            curve.append(
                {
                    "offered_qps": float(r["offered_qps"]),
                    "steady_turn_slo_attainment_pct": float(r["steady_turn_slo_attainment_pct"]),
                    "steady_turn_slo_goodput_rps": float(r["steady_turn_slo_goodput_rps"]),
                    "steady_turn_survival_pct": float(r.get("steady_turn_survival_pct", r["steady_state_survival_pct"])),
                    "steady_planned_turns": int(r.get("steady_planned_turns", 0)),
                    "joint_slo_attainment_pct": float(r["joint_slo_attainment_pct"]),
                    "joint_slo_goodput_rps": float(r["joint_slo_goodput_rps"]),
                    "full_trace_joint_slo_attainment_pct": float(
                        r.get("full_trace_joint_slo_attainment_pct", r["joint_slo_attainment_pct"])
                    ),
                    "full_trace_goodput_rps": float(r.get("full_trace_goodput_rps", r["joint_slo_goodput_rps"])),
                    "steady_state_survival_pct": float(r["steady_state_survival_pct"]),
                    "survival_rate_pct": float(r["survival_rate_pct"]),
                    "steady_oom_sessions": int(r["steady_oom_sessions"]),
                    "oom_sessions": int(r["oom_sessions"]),
                    "steady_state_sessions": int(r.get("steady_state_sessions", r["total_sessions"])),
                    "total_sessions": int(r["total_sessions"]),
                    "ar_ttft_p95_ms": float(r.get("ar_ttft_p95_ms", 0.0)),
                    "ar_tpot_p95_ms": float(r.get("ar_tpot_p95_ms", 0.0)),
                    "survivor_dit_turn_p95_ms": float(r.get("dit_turn_p95_ms", 0.0)),
                    "all_session_dit_p95_ms": r.get("all_session_dit_p95_ms"),
                    "active_jct_p95_ms": r.get("active_jct_p95_ms"),
                    "all_session_active_jct_p95_ms": r.get("all_session_active_jct_p95_ms"),
                    "all_session_e2e_p95_ms": r.get("all_session_e2e_p95_ms"),
                    "kv_host_swap_outs": int(r.get("kv_host_swap_outs", 0)),
                    "kv_host_swap_ins": int(r.get("kv_host_swap_ins", 0)),
                    "vmm_physical_pool_mib": float(r.get("vmm_physical_pool_mib", 0.0)),
                    "peak_nvml_used_mib": float(r.get("peak_nvml_used_mib", 0.0)),
                }
            )

        max_q_80 = interpolate_slo_threshold_qps(steady_pts, threshold_pct=80.0)
        max_q_90 = interpolate_slo_threshold_qps(steady_pts, threshold_pct=90.0)
        turn_max_q_80 = interpolate_slo_threshold_qps(steady_turn_pts, threshold_pct=80.0)
        turn_max_q_90 = interpolate_slo_threshold_qps(steady_turn_pts, threshold_pct=90.0)

        summary[pol] = {
            "primary_metric": "steady_turn_slo_attainment_pct_vs_offered_qps",
            "secondary_session_metric": "joint_slo_attainment_pct_vs_offered_qps",
            "turn_max_sustainable_qps_80pct": turn_max_q_80,
            "turn_max_sustainable_qps_90pct": turn_max_q_90,
            "turn_max_sustainable_qps_zero_oom_80pct": interpolate_slo_threshold_qps(
                steady_turn_zero_oom_pts, threshold_pct=80.0, require_zero_oom=True
            ),
            "turn_max_sustainable_qps_zero_oom_90pct": interpolate_slo_threshold_qps(
                steady_turn_zero_oom_pts, threshold_pct=90.0, require_zero_oom=True
            ),
            "turn_max_grid_qps_80pct": round(turn_grid_80, 3),
            "turn_max_grid_qps_90pct": round(turn_grid_90, 3),
            "max_sustainable_qps_80pct": max_q_80,
            "max_sustainable_qps_90pct": max_q_90,
            "max_sustainable_qps_zero_oom_80pct": interpolate_slo_threshold_qps(
                steady_zero_oom_pts, threshold_pct=80.0, require_zero_oom=True
            ),
            "max_sustainable_qps_zero_oom_90pct": interpolate_slo_threshold_qps(
                steady_zero_oom_pts, threshold_pct=90.0, require_zero_oom=True
            ),
            "full_window_max_sustainable_qps_80pct": interpolate_slo_threshold_qps(full_pts, threshold_pct=80.0),
            "full_window_max_sustainable_qps_90pct": interpolate_slo_threshold_qps(full_pts, threshold_pct=90.0),
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


@dataclass
class SuiteRunSpec:
    """One suite execution step inside a `BenchmarkRunner` policy sweep."""

    suite: WorkloadSuite
    target_qps: float | None = None
    run_options: RunWindowOptions | None = None
    burstgpt_deltas: list[float] | None = None
    gentd26_deltas: list[float] | None = None
    save_outputs_dir: Path | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)


class ActiveBenchmarkSession:
    """Live started `MorphServeClusterEngine` handle scoped inside `BenchmarkRunner.session(...)`."""

    def __init__(self, engine: Any, init_info: dict[str, Any], startup_seconds: float) -> None:
        self.engine = engine
        self.init_info = init_info
        self.startup_seconds = float(startup_seconds)

    def run_suite(
        self,
        suite: WorkloadSuite,
        *,
        target_qps: float | None = None,
        burstgpt_deltas: list[float] | None = None,
        gentd26_deltas: list[float] | None = None,
        save_outputs_dir: Path | None = None,
        run_options: RunWindowOptions | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if target_qps is not None:
            kwargs["target_qps"] = float(target_qps)
        if burstgpt_deltas is not None:
            kwargs["burstgpt_deltas"] = burstgpt_deltas
        if gentd26_deltas is not None:
            kwargs["gentd26_deltas"] = gentd26_deltas
        if save_outputs_dir is not None:
            kwargs["save_outputs_dir"] = save_outputs_dir
        if run_options is not None:
            kwargs["run_options"] = run_options
        return self.engine.run_suite(suite, **kwargs)

    def run_compact(
        self,
        suite: WorkloadSuite,
        *,
        target_qps: float | None = None,
        burstgpt_deltas: list[float] | None = None,
        gentd26_deltas: list[float] | None = None,
        save_outputs_dir: Path | None = None,
        run_options: RunWindowOptions | None = None,
        max_error_samples: int = 8,
        include_cdf_outcomes: bool = False,
        include_failed_sessions: bool = False,
        attach_init_shmem: bool = True,
    ) -> tuple[Any, dict[str, Any]]:
        rep = self.run_suite(
            suite,
            target_qps=target_qps,
            burstgpt_deltas=burstgpt_deltas,
            gentd26_deltas=gentd26_deltas,
            save_outputs_dir=save_outputs_dir,
            run_options=run_options,
        )
        compact = compact_report_dict(
            rep,
            max_error_samples=max_error_samples,
            include_cdf_outcomes=include_cdf_outcomes,
            include_failed_sessions=include_failed_sessions,
            init_info=self.init_info if attach_init_shmem else None,
        )
        return rep, compact


class BenchmarkRunner:
    """Lifecycle coordinator owning `MorphServeClusterEngine` startup, suite sweeps, compaction, and teardown."""

    def __init__(
        self,
        *,
        model_dir: Path | str = Path("."),
        bench_root: Path | str = Path("."),
        tmp_root_base: Path | str | None = None,
        engine_factory: Callable[..., Any] | None = None,
        fleet_factory: Callable[[EnginePolicyConfig], Any] | None = None,
        topology: Any | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.bench_root = Path(bench_root)
        self.tmp_root_base = Path(tmp_root_base) if tmp_root_base is not None else local_tmp_base()
        self.engine_factory = engine_factory
        self.fleet_factory = fleet_factory
        self.topology = topology

    def _create_engine(self, policy: EnginePolicyConfig, *, tmp_tag: str) -> Any:
        tmp_root = self.tmp_root_base / tmp_tag
        fleet = self.fleet_factory(policy) if self.fleet_factory is not None else None
        if self.engine_factory is not None:
            return self.engine_factory(
                policy=policy,
                model_dir=self.model_dir,
                bench_root=self.bench_root,
                tmp_root=tmp_root,
                fleet=fleet,
                topology=self.topology,
            )
        from morphserve.engine import MorphServeClusterEngine

        return MorphServeClusterEngine(
            policy=policy,
            model_dir=self.model_dir,
            bench_root=self.bench_root,
            tmp_root=tmp_root,
            fleet=fleet,
            topology=self.topology,
        )

    @contextmanager
    def session(
        self,
        policy: EnginePolicyConfig,
        *,
        tmp_tag: str,
    ) -> Iterator[ActiveBenchmarkSession]:
        """Start one `MorphServeClusterEngine`, yield an `ActiveBenchmarkSession`, and guarantee teardown + GC."""
        engine = self._create_engine(policy, tmp_tag=tmp_tag)
        t0 = time.perf_counter()
        try:
            init_info = engine.start() or {}
            startup_s = round(time.perf_counter() - t0, 2)
            yield ActiveBenchmarkSession(engine=engine, init_info=dict(init_info), startup_seconds=startup_s)
        finally:
            engine.shutdown()
            gc.collect()

    def run_policy_sweep(
        self,
        policy: EnginePolicyConfig,
        specs: Sequence[SuiteRunSpec],
        *,
        tmp_tag: str,
        max_error_samples: int = 8,
        include_cdf_outcomes: bool = False,
        include_failed_sessions: bool = False,
        attach_init_shmem: bool = True,
        on_result: Callable[[SuiteRunSpec, Any, dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Run one or more `SuiteRunSpec` items against a single warm engine instance and return compact dicts."""
        out: list[dict[str, Any]] = []
        with self.session(policy, tmp_tag=tmp_tag) as sess:
            for spec in specs:
                rep, compact = sess.run_compact(
                    spec.suite,
                    target_qps=spec.target_qps,
                    burstgpt_deltas=spec.burstgpt_deltas,
                    gentd26_deltas=spec.gentd26_deltas,
                    save_outputs_dir=spec.save_outputs_dir,
                    run_options=spec.run_options,
                    max_error_samples=max_error_samples,
                    include_cdf_outcomes=include_cdf_outcomes,
                    include_failed_sessions=include_failed_sessions,
                    attach_init_shmem=attach_init_shmem,
                )
                if spec.extra_fields:
                    compact.update(spec.extra_fields)
                if on_result is not None:
                    on_result(spec, rep, compact)
                out.append(compact)
        return out


__all__ = [
    "ActiveBenchmarkSession",
    "BenchmarkRunner",
    "SuiteRunSpec",
    "compact_report_dict",
    "compute_phase_breakdown",
    "compute_quadrant_breakdown",
    "compute_sustainable_qps_summary",
    "effective_steady_window_s",
    "interpolate_slo_threshold_qps",
    "load_main_mix_suite",
    "local_tmp_base",
    "parse_csv_floats",
    "parse_csv_strings",
    "resolve_default_main_suite_path",
    "resolve_sources_dir",
]
