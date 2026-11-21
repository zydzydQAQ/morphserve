"""Phase-2 Wave-3 Overnight Deepening: Full 80-Session Curve & Static Split-Ratio (2:6/4:4/6:2) Sensitivity."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import time
from typing import Any

from morphserve.bench.interleaved_bench import (
    WorkloadSuite,
    load_burstgpt_interarrivals,
    load_gentd26_interarrivals,
)
from morphserve.bench.run_phase2_bagel_main import stratify_suite
from morphserve.engine import EnginePolicyConfig, MorphServeClusterEngine


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = (len(s) - 1) * (p / 100.0)
    lo = int(math.floor(idx))
    hi = min(len(s) - 1, int(math.ceil(idx)))
    return round(float(s[lo] + (s[hi] - s[lo]) * (idx - lo)), 2)


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
    """Compare MorphServe elastic alpha against 2:6, 4:4, and 6:2 static NCCL P2P splits."""
    assert num_gpus == 8
    base_static = EnginePolicyConfig.from_preset("static_nccl_fission", num_gpus=8)
    splits = [
        ("morphserve_elastic", EnginePolicyConfig.from_preset("morphserve", num_gpus=8)),
        (
            "static_2und_6gen",
            replace(
                base_static,
                policy_name="static_2und_6gen",
                static_und_ranks=(0, 4),
                static_gen_ranks=(1, 2, 3, 5, 6, 7),
                initial_alphas={g: (0.0 if g in (0, 4) else 1.0) for g in range(8)},
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
                static_und_ranks=(0, 1, 2, 4, 5, 6),
                static_gen_ranks=(3, 7),
                initial_alphas={g: (1.0 if g in (3, 7) else 0.0) for g in range(8)},
            ),
        ),
    ]
    return splits


def run_overnight_deep(
    model_dir: Path,
    bench_root: Path,
    output_dir: Path,
    num_gpus: int = 8,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources_dir = bench_root / "sources"
    burst_deltas = load_burstgpt_interarrivals(sources_dir / "burstgpt_1_sample.csv")
    gentd_deltas = load_gentd26_interarrivals(sources_dir / "gentd26_lora_request_trace.csv")

    # Part 1: Full un-subsampled 80-session `main_mix_80.json` comparison (`morphserve` vs top 2 static/coloc baselines)
    full_suite = WorkloadSuite.load_json(bench_root / "suites/main_mix_80.json")
    full80_results: list[dict[str, Any]] = []
    for pol_name in ("morphserve", "prism_tetriserve_coloc", "static_nccl_fission", "monolithic_dp"):
        pol = EnginePolicyConfig.from_preset(pol_name, num_gpus=num_gpus)
        eng = MorphServeClusterEngine(
            policy=pol,
            model_dir=model_dir,
            bench_root=bench_root,
            tmp_root=Path(f"/tmp/ms_deep80_{pol_name}"),
        )
        try:
            eng.start()
            for qps in (0.6, 1.1):
                rep = eng.run_suite(
                    full_suite,
                    target_qps=qps,
                    burstgpt_deltas=burst_deltas,
                    gentd_deltas=gentd_deltas,
                )
                d = rep.to_dict()
                d["per_quadrant"] = _quadrant_breakdown(rep.sessions)
                d.pop("sessions", None)
                full80_results.append(d)
                print(
                    f"[DEEP-80] policy={pol_name} qps={qps} goodput={rep.joint_slo_goodput_rps:.3f} "
                    f"joint_slo={rep.joint_slo_attainment_pct}% oom={rep.oom_sessions}",
                    flush=True,
                )
        finally:
            eng.shutdown()
            gc.collect()

    # Part 2: Static split-ratio sensitivity (2:6 vs 4:4 vs 6:2 vs MorphServe) across 3 modality skews
    skew_suites = [
        (tag, stratify_suite(WorkloadSuite.load_json(bench_root / rel), max_sessions=24))
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
            tmp_root=Path(f"/tmp/ms_ratio_{cfg_label}"),
        )
        try:
            eng.start()
            for sk_tag, sk_suite in skew_suites:
                rep = eng.run_suite(
                    sk_suite,
                    target_qps=1.0,
                    burstgpt_deltas=burst_deltas,
                    gentd_deltas=gentd_deltas,
                )
                d = rep.to_dict()
                d.pop("sessions", None)
                d["split_config"] = cfg_label
                d["skew_tag"] = sk_tag
                ratio_results.append(d)
                print(
                    f"[DEEP-RATIO] cfg={cfg_label} skew={sk_tag} goodput={rep.joint_slo_goodput_rps:.3f} "
                    f"joint_slo={rep.joint_slo_attainment_pct}%",
                    flush=True,
                )
        finally:
            eng.shutdown()
            gc.collect()

    report = {
        "schema": "morphserve.phase2.wave3_deep.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "full_80session_runs": full80_results,
        "static_ratio_sensitivity": ratio_results,
        "status": "ok" if all(r["error_sessions"] == 0 for r in full80_results + ratio_results) else "check_failed",
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
    args = parser.parse_args(argv)
    rep = run_overnight_deep(args.model_dir, args.bench_root, args.output_dir, num_gpus=args.num_gpus)
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
