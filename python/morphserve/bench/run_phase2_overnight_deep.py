"""Phase-2 Wave-3 Overnight Deepening: Steady-State (`n=80-96`) Curve & Static Split-Ratio (2:6/4:4/6:2) Sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Callable

from morphserve.bench.harness import (
    BenchmarkRunner,
    SuiteRunSpec,
    compute_quadrant_breakdown,
    load_main_mix_suite,
    local_tmp_base,
    resolve_sources_dir,
)
from morphserve.bench.interleaved_bench import (
    WorkloadSuite,
    load_burstgpt_interarrivals,
    load_gentd26_interarrivals,
)
from morphserve.bench.run_phase2_bagel_main import (
    DEFAULT_MAIN_SESSIONS,
    DEFAULT_SKEW_SESSIONS,
    scale_suite_sessions,
)
from morphserve.engine.config import EnginePolicyConfig, build_static_split_configs


def run_overnight_deep(
    model_dir: Path,
    bench_root: Path,
    output_dir: Path,
    num_gpus: int = 8,
    main_sessions: int = DEFAULT_MAIN_SESSIONS,
    skew_sessions: int = DEFAULT_SKEW_SESSIONS,
    *,
    engine_factory: Callable[..., Any] | None = None,
    fleet_factory: Callable[[EnginePolicyConfig], Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources_dir = resolve_sources_dir(bench_root)
    burst_deltas = load_burstgpt_interarrivals(sources_dir / "burstgpt_1_sample.csv")
    gentd_deltas = load_gentd26_interarrivals(sources_dir / "gentd26_lora_request_trace.csv")

    runner = BenchmarkRunner(
        model_dir=model_dir,
        bench_root=bench_root,
        tmp_root_base=local_tmp_base(),
        engine_factory=engine_factory,
        fleet_factory=fleet_factory,
    )

    # Part 1: Steady-state `n=96` main_mix comparison (`morphserve` vs top static/colocation baselines)
    raw_main = load_main_mix_suite(bench_root)
    full_suite = scale_suite_sessions(raw_main, target_sessions=main_sessions)
    full_results: list[dict[str, Any]] = []
    for pol_name in ("morphserve", "prism_tetriserve_coloc", "static_nccl_fission", "monolithic_dp"):
        pol = EnginePolicyConfig.from_preset(pol_name, num_gpus=num_gpus)
        specs = [
            SuiteRunSpec(
                suite=full_suite,
                target_qps=qps,
                burstgpt_deltas=burst_deltas,
                gentd26_deltas=gentd_deltas,
            )
            for qps in (0.6, 1.1, 1.4)
        ]

        def _on_main(spec: SuiteRunSpec, rep: Any, compact: dict[str, Any], _pol_name: str = pol_name) -> None:
            compact["per_quadrant"] = compute_quadrant_breakdown(rep.sessions)
            errs = compact.get("error_samples") or ()
            print(
                f"[DEEP-MAIN] policy={_pol_name} n={len(spec.suite.sessions)} qps={spec.target_qps} "
                f"steady_goodput={rep.joint_slo_goodput_rps:.3f} "
                f"steady_slo={rep.joint_slo_attainment_pct}% oom={rep.oom_sessions} err={rep.error_sessions}"
                + (f" sample_err={errs[0]!r}" if errs else ""),
                flush=True,
            )

        full_results.extend(
            runner.run_policy_sweep(
                pol,
                specs,
                tmp_tag=f"ms_deep_{pol_name}",
                max_error_samples=5,
                attach_init_shmem=False,
                on_result=_on_main,
            )
        )

    # Part 2: Steady-state `n=80` static split-ratio sensitivity (2:6 vs 4:4 vs 6:2 vs MorphServe) across 3 skews
    skew_suites = [
        (tag, scale_suite_sessions(WorkloadSuite.load_json(bench_root / rel), target_sessions=skew_sessions))
        for tag, rel in (
            ("ar80_dit20", "suites/skew_ar80_dit20.json"),
            ("ar50_dit50", "suites/skew_ar50_dit50.json"),
            ("ar20_dit80", "suites/skew_ar20_dit80.json"),
        )
    ]
    ratio_results: list[dict[str, Any]] = []
    for cfg_label, cfg in build_static_split_configs(num_gpus=num_gpus):
        ratio_specs = [
            SuiteRunSpec(
                suite=sk_suite,
                target_qps=1.0,
                burstgpt_deltas=burst_deltas,
                gentd26_deltas=gentd_deltas,
                extra_fields={"split_config": cfg_label, "skew_tag": sk_tag},
            )
            for sk_tag, sk_suite in skew_suites
        ]

        def _on_ratio(spec: SuiteRunSpec, rep: Any, compact: dict[str, Any], _cfg_label: str = cfg_label) -> None:
            errs = compact.get("error_samples") or ()
            print(
                f"[DEEP-RATIO] cfg={_cfg_label} skew={spec.extra_fields['skew_tag']} n={len(spec.suite.sessions)} "
                f"steady_goodput={rep.joint_slo_goodput_rps:.3f} "
                f"steady_slo={rep.joint_slo_attainment_pct}% oom={rep.oom_sessions} err={rep.error_sessions}"
                + (f" sample_err={errs[0]!r}" if errs else ""),
                flush=True,
            )

        ratio_results.extend(
            runner.run_policy_sweep(
                cfg,
                ratio_specs,
                tmp_tag=f"ms_ratio_{cfg_label}",
                max_error_samples=5,
                attach_init_shmem=False,
                on_result=_on_ratio,
            )
        )

    report = {
        "schema": "morphserve.osdi27.wave3_deep.v2",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "main_sessions_per_point": len(full_suite.sessions),
        "skew_sessions_per_point": skew_sessions,
        "full_80session_runs": full_results,
        "static_ratio_sensitivity": ratio_results,
        "status": "ok" if all(r["error_sessions"] == 0 for r in full_results + ratio_results) else "check_failed",
    }
    out_path = output_dir / "wave3-deep-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MorphServe Wave-3 Overnight Deep Statistical Suite")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--main-sessions", type=int, default=DEFAULT_MAIN_SESSIONS)
    parser.add_argument("--skew-sessions", type=int, default=DEFAULT_SKEW_SESSIONS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rep = run_overnight_deep(
        args.model_dir,
        args.bench_root,
        args.output_dir,
        num_gpus=args.num_gpus,
        main_sessions=args.main_sessions,
        skew_sessions=args.skew_sessions,
    )
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
