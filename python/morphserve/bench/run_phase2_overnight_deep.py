"""Phase-2 Wave-3 Overnight Deepening: Steady-State (`n=80-96`) Curve & Static Split-Ratio (2:6/4:4/6:2) Sensitivity."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import time
from typing import Any

from morphserve.bench.interleaved_bench import (
    WorkloadSuite,
    load_burstgpt_interarrivals,
    load_gentd26_interarrivals,
)
from morphserve.bench.run_phase2_bagel_main import (
    DEFAULT_MAIN_SESSIONS,
    DEFAULT_SKEW_SESSIONS,
    _local_tmp_base,
    _resolve_sources_dir,
    scale_suite_sessions,
)
from morphserve.engine import EnginePolicyConfig, MorphServeClusterEngine
from morphserve.telemetry import percentile as _pct


def _quadrant_breakdown(sessions_raw: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_q: dict[str, list[dict[str, Any]]] = {}
    for s in sessions_raw:
        by_q.setdefault(s["quadrant"], []).append(s)
    out: dict[str, dict[str, Any]] = {}
    for q, items in sorted(by_q.items()):
        ok = [x for x in items if x.get("status") == "ok"]
        joint = [x for x in ok if x.get("joint_slo_met")]
        e2es = [float(x["e2e_ms"]) for x in ok]
        tpots = [float(v) for x in ok for v in x.get("ar_tpots_ms", [])]
        dits = [float(v) for x in ok for v in x.get("dit_turns_ms", [])]
        out[q] = {
            "count": len(items),
            "joint_slo_pct": round(100.0 * len(joint) / max(1, len(items)), 1),
            "oom_count": sum(1 for x in items if x.get("status") == "oom"),
            "e2e_p50_ms": _pct(e2es, 50),
            "e2e_p95_ms": _pct(e2es, 95),
            "ar_tpot_p95_ms": _pct(tpots, 95),
            "dit_turn_p95_ms": _pct(dits, 95),
        }
    return out


def build_static_split_configs(num_gpus: int = 8) -> list[tuple[str, EnginePolicyConfig]]:
    """Compare MorphServe elastic alpha against 1:3 (2:6), 2:2 (4:4), and 3:1 (6:2) static NCCL P2P splits."""
    n = int(num_gpus)
    if n < 4 or n % 4 != 0:
        raise ValueError(f"build_static_split_configs requires num_gpus >= 4 and divisible by 4, got {n}")
    base_static = EnginePolicyConfig.from_preset("static_nccl_fission", num_gpus=n)
    if n % 8 == 0:
        und_1_3 = tuple(g for g in range(n) if (g % 8) in (0, 4))
        und_3_1 = tuple(g for g in range(n) if (g % 8) in (0, 1, 2, 4, 5, 6))
    else:
        q = n // 4
        und_1_3 = tuple(range(q))
        und_3_1 = tuple(range(3 * q))
    gen_1_3 = tuple(g for g in range(n) if g not in und_1_3)
    gen_3_1 = tuple(g for g in range(n) if g not in und_3_1)

    splits = [
        ("morphserve_elastic", EnginePolicyConfig.from_preset("morphserve", num_gpus=n)),
        (
            "static_2und_6gen",
            replace(
                base_static,
                policy_name="static_2und_6gen",
                static_und_ranks=und_1_3,
                static_gen_ranks=gen_1_3,
                initial_alphas={g: (0.0 if g in und_1_3 else 1.0) for g in range(n)},
            ),
        ),
        (
            "static_4und_4gen",
            replace(base_static, policy_name="static_4und_4gen"),
        ),
        (
            "static_6und_2gen",
            replace(
                base_static,
                policy_name="static_6und_2gen",
                static_und_ranks=und_3_1,
                static_gen_ranks=gen_3_1,
                initial_alphas={g: (0.0 if g in und_3_1 else 1.0) for g in range(n)},
            ),
        ),
    ]
    return splits


def run_overnight_deep(
    model_dir: Path,
    bench_root: Path,
    output_dir: Path,
    num_gpus: int = 8,
    main_sessions: int = DEFAULT_MAIN_SESSIONS,
    skew_sessions: int = DEFAULT_SKEW_SESSIONS,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources_dir = _resolve_sources_dir(bench_root)
    burst_deltas = load_burstgpt_interarrivals(sources_dir / "burstgpt_1_sample.csv")
    gentd_deltas = load_gentd26_interarrivals(sources_dir / "gentd26_lora_request_trace.csv")

    tmp_base = _local_tmp_base()
    # Part 1: Steady-state `n=96` main_mix comparison (`morphserve` vs top static/colocation baselines)
    raw_main = WorkloadSuite.load_json(bench_root / "suites/main_mix_80.json")
    full_suite = scale_suite_sessions(raw_main, target_sessions=main_sessions)
    full_results: list[dict[str, Any]] = []
    for pol_name in ("morphserve", "prism_tetriserve_coloc", "static_nccl_fission", "monolithic_dp"):
        pol = EnginePolicyConfig.from_preset(pol_name, num_gpus=num_gpus)
        eng = MorphServeClusterEngine(
            policy=pol,
            model_dir=model_dir,
            bench_root=bench_root,
            tmp_root=tmp_base / f"ms_deep_{pol_name}",
        )
        try:
            eng.start()
            for qps in (0.6, 1.1, 1.4):
                rep = eng.run_suite(
                    full_suite,
                    target_qps=qps,
                    burstgpt_deltas=burst_deltas,
                    gentd26_deltas=gentd_deltas,
                )
                d = rep.to_dict()
                err_samples = [str(s.get("error", ""))[:240] for s in rep.sessions if s.get("status") == "error"][:5]
                if err_samples:
                    d["error_samples"] = err_samples
                d["per_quadrant"] = _quadrant_breakdown(rep.sessions)
                d.pop("sessions", None)
                full_results.append(d)
                print(
                    f"[DEEP-MAIN] policy={pol_name} n={len(full_suite.sessions)} qps={qps} "
                    f"steady_goodput={rep.steady_state_goodput_rps:.3f} "
                    f"steady_slo={rep.steady_state_attainment_pct}% oom={rep.oom_sessions} err={rep.error_sessions}"
                    + (f" sample_err={err_samples[0]!r}" if err_samples else ""),
                    flush=True,
                )
        finally:
            eng.shutdown()
            gc.collect()

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
        eng = MorphServeClusterEngine(
            policy=cfg,
            model_dir=model_dir,
            bench_root=bench_root,
            tmp_root=tmp_base / f"ms_ratio_{cfg_label}",
        )
        try:
            eng.start()
            for sk_tag, sk_suite in skew_suites:
                rep = eng.run_suite(
                    sk_suite,
                    target_qps=1.0,
                    burstgpt_deltas=burst_deltas,
                    gentd26_deltas=gentd_deltas,
                )
                d = rep.to_dict()
                err_samples = [str(s.get("error", ""))[:240] for s in rep.sessions if s.get("status") == "error"][:5]
                if err_samples:
                    d["error_samples"] = err_samples
                d.pop("sessions", None)
                d["split_config"] = cfg_label
                d["skew_tag"] = sk_tag
                ratio_results.append(d)
                print(
                    f"[DEEP-RATIO] cfg={cfg_label} skew={sk_tag} n={len(sk_suite.sessions)} "
                    f"steady_goodput={rep.steady_state_goodput_rps:.3f} "
                    f"steady_slo={rep.steady_state_attainment_pct}% oom={rep.oom_sessions} err={rep.error_sessions}"
                    + (f" sample_err={err_samples[0]!r}" if err_samples else ""),
                    flush=True,
                )
        finally:
            eng.shutdown()
            gc.collect()

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MorphServe Wave-3 Overnight Deep Statistical Suite")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--main-sessions", type=int, default=DEFAULT_MAIN_SESSIONS)
    parser.add_argument("--skew-sessions", type=int, default=DEFAULT_SKEW_SESSIONS)
    args = parser.parse_args(argv)
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
