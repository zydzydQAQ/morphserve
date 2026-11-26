"""Phase-2 Milestone M2.3: 8xA100 BAGEL-7B-MoT Interleaved-Bench v2 Time-Windowed Evaluation, Phase-Shift, Think-Time Sweep & Two-Tier Accuracy."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
from typing import Any, Callable

from morphserve.bench.exactness import ExactnessEvaluator
from morphserve.bench.harness import (
    BenchmarkRunner,
    SuiteRunSpec,
    compute_phase_breakdown,
    compute_sustainable_qps_summary,
    effective_steady_window_s,
    load_main_mix_suite,
    local_tmp_base,
    parse_csv_floats,
    parse_csv_strings,
    resolve_sources_dir,
)
from morphserve.bench.interleaved_bench import (
    InterleavedSession,
    InterleavedTurn,
    WorkloadSuite,
    build_m3_active_accuracy_suite,
    build_online_phase_shift_suite,
    load_burstgpt_interarrivals,
    load_gentd26_interarrivals,
)
from morphserve.engine.config import (
    EnginePolicyConfig,
    RunWindowOptions,
    VALID_POLICIES,
    build_ablation_configs,
)
from morphserve.engine.workload import build_time_window_suite, hamilton_apportion

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


def stratify_suite(suite: WorkloadSuite, max_sessions: int) -> WorkloadSuite:
    """Select a deterministic, quadrant-balanced slice of `suite` preserving arrival ordering."""
    if max_sessions <= 0 or len(suite.sessions) <= max_sessions:
        return suite
    by_quad: dict[str, list[InterleavedSession]] = defaultdict(list)
    for s in suite.sessions:
        by_quad[s.quadrant].append(s)

    weights = {q: float(len(items)) for q, items in sorted(by_quad.items())}
    counts = hamilton_apportion(int(max_sessions), weights)
    picked: list[InterleavedSession] = []
    for q, items in sorted(by_quad.items()):
        picked.extend(items[: counts.get(q, 0)])
    picked.sort(key=lambda s: (s.arrival_offset_s, s.session_id))
    return WorkloadSuite(
        name=f"{suite.name}_n{len(picked)}",
        trace_source=suite.trace_source,
        base_qps=suite.base_qps,
        sessions=picked,
        schema=suite.schema,
    )


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

    raw_main = load_main_mix_suite(bench_root)
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
    *,
    engine_factory: Callable[..., Any] | None = None,
    fleet_factory: Callable[[EnginePolicyConfig], Any] | None = None,
) -> dict[str, Any]:
    """Suite A (1x8 GPUs): Interleaved-Bench v2 240s time-windowed QPS sweep + LOO M1-M4 ablations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_main = load_main_mix_suite(bench_root)

    sources_dir = resolve_sources_dir(bench_root)
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
    runner = BenchmarkRunner(
        model_dir=model_dir,
        bench_root=bench_root,
        tmp_root_base=local_tmp_base(),
        engine_factory=engine_factory,
        fleet_factory=fleet_factory,
    )
    qps_sweep_results: list[dict[str, Any]] = []

    for preset in policies:
        policy = EnginePolicyConfig.from_preset(preset, num_gpus=num_gpus, **v2_ov)
        specs = [
            SuiteRunSpec(
                suite=suites_by_qps[float(qps)],
                target_qps=float(qps),
                run_options=RunWindowOptions(
                    steady_window_mode="time",
                    warmup_window_s=float(suites_by_qps[float(qps)].warmup_s),
                    steady_window_s=effective_steady_window_s(suites_by_qps[float(qps)]),
                ),
            )
            for qps in qps_points
        ]

        def _log_qps(spec: SuiteRunSpec, rep: Any, compact: dict[str, Any], _preset: str = preset) -> None:
            errs = compact.get("error_samples") or ()
            print(
                f"[M2.3-QPS-v2] policy={_preset} n={len(spec.suite.sessions)} qps={spec.target_qps} "
                f"turn_slo={rep.steady_turn_slo_attainment_pct}% "
                f"turn_gp={rep.steady_turn_slo_goodput_rps:.3f} "
                f"sess_slo={rep.joint_slo_attainment_pct}% steady_surv={rep.steady_state_survival_pct}% "
                f"active_jct_p95={rep.active_jct_p95_ms}ms "
                f"ar_tpot_p95={rep.ar_tpot_p95_ms}ms dit_p95={rep.dit_turn_p95_ms}ms "
                f"swaps={rep.kv_host_swap_outs} "
                f"steady_oom={rep.steady_oom_sessions} "
                f"oom={rep.oom_sessions} err={rep.error_sessions}"
                + (f" sample_err={errs[0]!r}" if errs else ""),
                flush=True,
            )

        qps_sweep_results.extend(
            runner.run_policy_sweep(
                policy,
                specs,
                tmp_tag=f"ms_m23a_{preset}",
                max_error_samples=5,
                include_cdf_outcomes=True,
                on_result=_log_qps,
            )
        )

    ablation_results: list[dict[str, Any]] = []
    ab_ov = _v2_time_window_policy_overrides(
        warmup_s=float(ablation_suite.warmup_s),
        steady_s=effective_steady_window_s(ablation_suite),
    )
    ab_run_opts = RunWindowOptions(
        steady_window_mode="time",
        warmup_window_s=float(ablation_suite.warmup_s),
        steady_window_s=effective_steady_window_s(ablation_suite),
    )
    for ab_name, ab_cfg in build_ablation_configs(num_gpus=num_gpus, **ab_ov):
        def _log_ab(spec: SuiteRunSpec, rep: Any, compact: dict[str, Any], _ab_name: str = ab_name) -> None:
            errs = compact.get("error_samples") or ()
            print(
                f"[M2.3-ABLATION-v2] {_ab_name} n={len(spec.suite.sessions)} qps={ablation_qps} "
                f"eff_tput={rep.joint_slo_goodput_rps:.3f} "
                f"steady_slo={rep.joint_slo_attainment_pct}% steady_surv={rep.steady_state_survival_pct}% "
                f"active_jct_p95={rep.active_jct_p95_ms}ms "
                f"ar_ttft_p99={rep.ar_ttft_p99_ms}ms peak_nvml={rep.peak_nvml_used_mib}MiB "
                f"yields={rep.total_yields} swaps={rep.kv_host_swap_outs} "
                f"steady_oom={rep.steady_oom_sessions} err={rep.error_sessions}"
                + (f" sample_err={errs[0]!r}" if errs else ""),
                flush=True,
            )

        ablation_results.extend(
            runner.run_policy_sweep(
                ab_cfg,
                [
                    SuiteRunSpec(
                        suite=ablation_suite,
                        target_qps=float(ablation_qps),
                        run_options=ab_run_opts,
                        extra_fields={"ablation_name": ab_name},
                    )
                ],
                tmp_tag=f"ms_m23ab_{ab_name}",
                max_error_samples=5,
                include_cdf_outcomes=False,
                attach_init_shmem=False,
                on_result=_log_ab,
            )
        )

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
        "sustainable_qps_summary": compute_sustainable_qps_summary(qps_sweep_results),
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
    *,
    engine_factory: Callable[..., Any] | None = None,
    fleet_factory: Callable[[EnginePolicyConfig], Any] | None = None,
) -> dict[str, Any]:
    """Suite A Low-to-Knee QPS Sweep (`0.05..0.80` RPS) via time-windowed `build_time_window_suite`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_main = load_main_mix_suite(bench_root)

    sources_dir = resolve_sources_dir(bench_root)
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
    runner = BenchmarkRunner(
        model_dir=model_dir,
        bench_root=bench_root,
        tmp_root_base=local_tmp_base(),
        engine_factory=engine_factory,
        fleet_factory=fleet_factory,
    )
    qps_sweep_results: list[dict[str, Any]] = []
    for preset in policies:
        policy = EnginePolicyConfig.from_preset(preset, num_gpus=num_gpus, **v2_ov)
        specs = [
            SuiteRunSpec(
                suite=suites_by_qps[float(qps)],
                target_qps=float(qps),
                run_options=RunWindowOptions(
                    steady_window_mode="time",
                    warmup_window_s=float(suites_by_qps[float(qps)].warmup_s),
                    steady_window_s=effective_steady_window_s(suites_by_qps[float(qps)]),
                ),
            )
            for qps in qps_points
        ]

        def _log_low(spec: SuiteRunSpec, rep: Any, compact: dict[str, Any], _preset: str = preset) -> None:
            errs = compact.get("error_samples") or ()
            print(
                f"[M2.3-LOWQPS-v2] policy={_preset} n={len(spec.suite.sessions)} qps={spec.target_qps} "
                f"steady_slo={rep.joint_slo_attainment_pct}% steady_surv={rep.steady_state_survival_pct}% "
                f"nvml_peak={rep.peak_nvml_used_mib}MiB "
                f"active_jct_p95={rep.active_jct_p95_ms}ms "
                f"all_active_jct_p95={rep.all_session_active_jct_p95_ms} "
                f"all_dit_p95={rep.all_session_dit_p95_ms} "
                f"steady_oom={rep.steady_oom_sessions} "
                f"oom={rep.oom_sessions} err={rep.error_sessions}"
                + (f" sample_err={errs[0]!r}" if errs else ""),
                flush=True,
            )

        qps_sweep_results.extend(
            runner.run_policy_sweep(
                policy,
                specs,
                tmp_tag=f"ms_m23low_{preset}",
                max_error_samples=5,
                include_cdf_outcomes=True,
                on_result=_log_low,
            )
        )

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
        "primary_metric": "joint_slo_attainment_pct_vs_offered_qps",
        "num_gpus": num_gpus,
        "steady_window_mode": "time",
        "arrival_duration_s": float(arrival_duration_s),
        "warmup_s": float(warmup_s),
        "steady_s": float(steady_s),
        "min_sessions": int(min_sessions),
        "sessions_by_qps": {str(q): len(suites_by_qps[float(q)].sessions) for q in qps_points},
        "qps_points": [float(q) for q in qps_points],
        "sustainable_qps_summary": compute_sustainable_qps_summary(combined_sweep),
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
    *,
    engine_factory: Callable[..., Any] | None = None,
    fleet_factory: Callable[[EnginePolicyConfig], Any] | None = None,
) -> dict[str, Any]:
    """Fixed knee-QPS sweep of constant inter-turn think times `(0, 5, 20, 60)s` verifying KV residency & swap scaling."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_main = load_main_mix_suite(bench_root)

    sources_dir = resolve_sources_dir(bench_root)
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
    eff_steady_s = effective_steady_window_s(window_suite)

    runner = BenchmarkRunner(
        model_dir=model_dir,
        bench_root=bench_root,
        tmp_root_base=local_tmp_base(),
        engine_factory=engine_factory,
        fleet_factory=fleet_factory,
    )
    sweep_results: list[dict[str, Any]] = []
    summary_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for preset in policies:
        init_ov = _v2_time_window_policy_overrides(
            warmup_s=float(window_suite.warmup_s),
            steady_s=eff_steady_s,
            think_range=(0.0, 0.0),
            think_dist="constant",
            think_median_s=0.0,
            think_sigma=0.0,
        )
        policy = EnginePolicyConfig.from_preset(preset, num_gpus=num_gpus, **init_ov)
        specs = [
            SuiteRunSpec(
                suite=window_suite,
                target_qps=float(offered_qps),
                run_options=RunWindowOptions(
                    inter_turn_think_s_range=(max(0.0, float(t_think)), max(0.0, float(t_think))),
                    think_time_dist="constant",
                    think_time_median_s=max(0.0, float(t_think)),
                    think_time_sigma=0.0,
                    steady_window_mode="time",
                    warmup_window_s=float(window_suite.warmup_s),
                    steady_window_s=eff_steady_s,
                ),
                extra_fields={"think_time_s": max(0.0, float(t_think))},
            )
            for t_think in think_times_s
        ]

        def _on_think(spec: SuiteRunSpec, rep: Any, compact: dict[str, Any], _preset: str = preset) -> None:
            t_val = float(spec.extra_fields["think_time_s"])
            errs = compact.get("error_samples") or ()
            summary_by_policy[_preset].append(
                {
                    "think_time_s": t_val,
                    "kv_host_swap_outs": int(rep.kv_host_swap_outs),
                    "kv_host_swap_ins": int(rep.kv_host_swap_ins),
                    "steady_oom_sessions": int(rep.steady_oom_sessions),
                    "oom_sessions": int(rep.oom_sessions),
                    "steady_state_survival_pct": float(rep.steady_state_survival_pct),
                    "joint_slo_attainment_pct": float(rep.joint_slo_attainment_pct),
                    "active_jct_p95_ms": float(rep.active_jct_p95_ms),
                    "all_session_active_jct_p95_ms": rep.all_session_active_jct_p95_ms,
                    "per_turn_stats": rep.per_turn_stats,
                    "peak_nvml_used_mib": float(rep.peak_nvml_used_mib),
                }
            )
            print(
                f"[M2.3-THINK] policy={_preset} qps={offered_qps} think_s={t_val:.1f} n={len(window_suite.sessions)} "
                f"steady_slo={rep.joint_slo_attainment_pct}% steady_surv={rep.steady_state_survival_pct}% "
                f"active_jct_p95={rep.active_jct_p95_ms}ms "
                f"swaps={rep.kv_host_swap_outs} nvml_peak={rep.peak_nvml_used_mib}MiB "
                f"steady_oom={rep.steady_oom_sessions} err={rep.error_sessions}"
                + (f" sample_err={errs[0]!r}" if errs else ""),
                flush=True,
            )

        sweep_results.extend(
            runner.run_policy_sweep(
                policy,
                specs,
                tmp_tag=f"ms_m23think_{preset}",
                max_error_samples=5,
                include_cdf_outcomes=False,
                on_result=_on_think,
            )
        )

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
    *,
    engine_factory: Callable[..., Any] | None = None,
    fleet_factory: Callable[[EnginePolicyConfig], Any] | None = None,
) -> dict[str, Any]:
    """Suite B (1x8 GPUs): Part 1 Continuous 360s Online Phase-Shift + Part 2 Two-Tier Accuracy & Lossless Suite."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sources_dir = resolve_sources_dir(bench_root)
    burst_deltas = load_burstgpt_interarrivals(sources_dir / "burstgpt_1_sample.csv")
    gentd_deltas = load_gentd26_interarrivals(sources_dir / "gentd26_lora_request_trace.csv")

    tmp_base = local_tmp_base()
    runner = BenchmarkRunner(
        model_dir=model_dir,
        bench_root=bench_root,
        tmp_root_base=tmp_base,
        engine_factory=engine_factory,
        fleet_factory=fleet_factory,
    )

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
        policy = EnginePolicyConfig.from_preset(preset, num_gpus=num_gpus, **phase_ov)

        def _on_phase(spec: SuiteRunSpec, rep: Any, compact: dict[str, Any], _preset: str = preset) -> None:
            compact["per_phase"] = compute_phase_breakdown(rep.sessions, suite_sessions=spec.suite.sessions)
            compact["time_series_10s"] = list(rep.time_series_10s)
            errs = compact.get("error_samples") or ()
            print(
                f"[M2.3-PHASE-SHIFT] policy={_preset} n={len(spec.suite.sessions)} qps={spec.suite.base_qps} "
                f"eff_tput={rep.joint_slo_goodput_rps:.3f} "
                f"steady_slo={rep.joint_slo_attainment_pct}% steady_surv={rep.steady_state_survival_pct}% "
                f"active_jct_p95={rep.active_jct_p95_ms}ms "
                f"ship={rep.path_a_ships} morph={rep.path_b_morphs} "
                f"steady_oom={rep.steady_oom_sessions} err={rep.error_sessions}"
                + (f" sample_err={errs[0]!r}" if errs else ""),
                flush=True,
            )

        # Preserve `phase_suite`'s exact 3-phase arrival schedule (`target_qps=None`).
        phase_shift_results.extend(
            runner.run_policy_sweep(
                policy,
                [SuiteRunSpec(suite=phase_suite, target_qps=None)],
                tmp_tag=f"ms_m23b_ps_{preset}",
                max_error_samples=5,
                include_cdf_outcomes=False,
                on_result=_on_phase,
            )
        )

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
            policy = EnginePolicyConfig.from_preset(preset, num_gpus=num_gpus, **sk_ov)
            sk_specs = [
                SuiteRunSpec(
                    suite=sk_suite,
                    target_qps=float(skew_qps),
                    run_options=RunWindowOptions(
                        steady_window_mode="time",
                        warmup_window_s=float(sk_suite.warmup_s),
                        steady_window_s=float(sk_suite.steady_s),
                    ),
                    extra_fields={"skew_tag": skew_tag},
                )
                for skew_tag, sk_suite in skew_suites
            ]

            def _on_skew(spec: SuiteRunSpec, rep: Any, compact: dict[str, Any], _preset: str = preset) -> None:
                errs = compact.get("error_samples") or ()
                print(
                    f"[M2.3-SKEW] policy={_preset} skew={spec.extra_fields['skew_tag']} n={len(spec.suite.sessions)} "
                    f"steady_goodput={rep.joint_slo_goodput_rps:.3f} "
                    f"steady_slo={rep.joint_slo_attainment_pct}% ship={rep.path_a_ships} morph={rep.path_b_morphs} "
                    f"oom={rep.oom_sessions} err={rep.error_sessions}"
                    + (f" sample_err={errs[0]!r}" if errs else ""),
                    flush=True,
                )

            skew_results.extend(
                runner.run_policy_sweep(
                    policy,
                    sk_specs,
                    tmp_tag=f"ms_m23b_sk_{preset}",
                    max_error_samples=5,
                    include_cdf_outcomes=False,
                    attach_init_shmem=False,
                    on_result=_on_skew,
                )
            )

    # ---- Part 2: Two-Tier Accuracy & Lossless Suite (144 sessions across all 4 v2 quadrants) ----
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
    acc_run_opts = RunWindowOptions(
        inter_turn_think_s_range=(0.0, 0.0),
        think_time_dist="constant",
        think_time_median_s=0.0,
        think_time_sigma=0.0,
        steady_window_mode="session_trim",
        steady_state_trim_frac=0.0,
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
            with runner.session(pol, tmp_tag=f"ms_m23_acc_{tier_label}") as sess:
                rep = sess.run_suite(
                    acc_suite,
                    target_qps=0.20,
                    save_outputs_dir=tmp_acc_root / tier_label,
                    run_options=acc_run_opts,
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
                    "active_jct_p50_ms": float(rep.active_jct_p50_ms),
                    "active_jct_p95_ms": float(rep.active_jct_p95_ms),
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the Phase-2 M2.3 BAGEL-7B-MoT Interleaved-Bench v2 runner."""
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    parsed_qps = tuple(parse_csv_floats(args.qps_points))
    parsed_policies = tuple(parse_csv_strings(args.policies))

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
        think_vals = tuple(parse_csv_floats(args.think_times)) or DEFAULT_THINK_TIMES_S
        eff_think_qps = (
            float(args.think_qps) if args.think_qps is not None else (float(parsed_qps[0]) if parsed_qps else 0.60)
        )
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
        eff_skew = (
            int(args.skew_sessions)
            if args.skew_sessions is not None
            else (int(args.max_sessions) if args.max_sessions is not None else DEFAULT_SKEW_SESSIONS)
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
