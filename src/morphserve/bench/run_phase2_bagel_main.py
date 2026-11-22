"""Phase-2 Milestone M2.3: 8xA100 BAGEL-7B-MoT 5-Baseline Evaluation, Skew, Ablations & Lossless Parity."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

from morphserve.bench.exactness import ExactnessEvaluator, image_pair_metrics
from morphserve.bench.interleaved_bench import (
    InterleavedSession,
    WorkloadSuite,
    load_burstgpt_interarrivals,
    load_gentd26_interarrivals,
)
from morphserve.engine import EnginePolicyConfig, MorphServeClusterEngine, VALID_POLICIES


def stratify_suite(suite: WorkloadSuite, max_sessions: int) -> WorkloadSuite:
    """Select a deterministic, quadrant-balanced slice of `suite` preserving arrival ordering."""
    if max_sessions <= 0 or len(suite.sessions) <= max_sessions:
        return suite
    by_quad: dict[str, list[InterleavedSession]] = defaultdict(list)
    for s in suite.sessions:
        by_quad[s.quadrant].append(s)

    n_tot = len(suite.sessions)
    picked: list[InterleavedSession] = []
    quotas: list[tuple[str, int, float]] = []
    used = 0
    for q, items in sorted(by_quad.items()):
        exact = max_sessions * (len(items) / float(n_tot))
        cnt = min(len(items), int(math.floor(exact)))
        quotas.append((q, cnt, exact - cnt))
        used += cnt

    quotas.sort(key=lambda x: (-x[2], x[0]))
    counts = {q: c for q, c, _ in quotas}
    rem = max_sessions - used
    for q, _, _ in quotas:
        if rem <= 0:
            break
        if counts[q] < len(by_quad[q]):
            counts[q] += 1
            rem -= 1

    for q, items in sorted(by_quad.items()):
        picked.extend(items[: counts[q]])
    picked.sort(key=lambda s: (s.arrival_offset_s, s.session_id))
    return WorkloadSuite(
        name=f"{suite.name}_n{len(picked)}",
        trace_source=suite.trace_source,
        base_qps=suite.base_qps,
        sessions=picked,
    )


def build_ablation_configs(num_gpus: int = 8) -> list[tuple[str, EnginePolicyConfig]]:
    """Construct the cumulative 5-step M1-M4 ablation ladder on BAGEL-7B-MoT."""
    base_ms = EnginePolicyConfig.from_preset("morphserve", num_gpus=num_gpus)
    no_m4 = replace(base_ms, policy_name="no_m4_layer_chunk", preemption_mode="non_preemptive")
    no_m3 = replace(no_m4, policy_name="no_m3_cow_loopback", use_module3_cow_loopback=False)
    no_m2 = replace(
        no_m3,
        policy_name="no_m2_dual_path",
        routing_mode="sticky_least_loaded",
        use_h2d_pacer=False,
        use_shared_numa_mirror=False,
    )
    base_mono = EnginePolicyConfig.from_preset("monolithic_dp", num_gpus=num_gpus)
    return [
        ("full_morphserve", base_ms),
        ("no_m4_layer_chunk", no_m4),
        ("no_m3_cow_loopback", no_m3),
        ("no_m2_dual_path", no_m2),
        ("no_m1_tripool_vmm", replace(base_mono, policy_name="no_m1_tripool_vmm")),
    ]


def run_suite_a_qps_and_ablations(
    model_dir: Path,
    bench_root: Path,
    output_dir: Path,
    num_gpus: int = 8,
    max_sessions: int = 28,
    qps_points: tuple[float, ...] = (0.4, 0.8, 1.3),
    ablation_qps: float = 1.1,
    policies: tuple[str, ...] = VALID_POLICIES,
) -> dict[str, Any]:
    """Suite A (1x8 GPUs): Main 4-quadrant QPS sweep across 6 policies + M1-M4 ablation ladder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_main = WorkloadSuite.load_json(bench_root / "suites/main_mix_80.json")
    suite = stratify_suite(raw_main, max_sessions=max_sessions)

    sources_dir = bench_root / "sources"
    burst_deltas = load_burstgpt_interarrivals(sources_dir / "burstgpt_1_sample.csv")
    gentd_deltas = load_gentd26_interarrivals(sources_dir / "gentd26_lora_request_trace.csv")

    qps_sweep_results: list[dict[str, Any]] = []
    for preset in policies:
        policy = EnginePolicyConfig.from_preset(preset, num_gpus=num_gpus)
        engine = MorphServeClusterEngine(
            policy=policy,
            model_dir=model_dir,
            bench_root=bench_root,
            tmp_root=Path(f"/tmp/ms_m23a_{preset}"),
        )
        try:
            init_info = engine.start()
            for qps in qps_points:
                rep = engine.run_suite(
                    suite,
                    target_qps=qps,
                    burstgpt_deltas=burst_deltas,
                    gentd26_deltas=gentd_deltas,
                )
                d = rep.to_dict()
                d.pop("sessions", None)
                d["host_shmem_delta_gib"] = init_info.get("host_shmem_delta_gib", rep.host_shmem_delta_gib)
                qps_sweep_results.append(d)
                print(
                    f"[M2.3-QPS] policy={preset} qps={qps} goodput={rep.joint_slo_goodput_rps:.3f} "
                    f"joint_slo={rep.joint_slo_attainment_pct}% ar_tpot_p95={rep.ar_tpot_p95_ms}ms "
                    f"dit_p95={rep.dit_turn_p95_ms}ms oom={rep.oom_sessions}",
                    flush=True,
                )
        finally:
            engine.shutdown()
            gc.collect()

    ablation_results: list[dict[str, Any]] = []
    for ab_name, ab_cfg in build_ablation_configs(num_gpus=num_gpus):
        engine = MorphServeClusterEngine(
            policy=ab_cfg,
            model_dir=model_dir,
            bench_root=bench_root,
            tmp_root=Path(f"/tmp/ms_m23ab_{ab_name}"),
        )
        try:
            engine.start()
            rep = engine.run_suite(
                suite,
                target_qps=ablation_qps,
                burstgpt_deltas=burst_deltas,
                gentd26_deltas=gentd_deltas,
            )
            d = rep.to_dict()
            d.pop("sessions", None)
            d["ablation_name"] = ab_name
            ablation_results.append(d)
            print(
                f"[M2.3-ABLATION] {ab_name} qps={ablation_qps} goodput={rep.joint_slo_goodput_rps:.3f} "
                f"joint_slo={rep.joint_slo_attainment_pct}% ar_tpot_p95={rep.ar_tpot_p95_ms}ms "
                f"peak_mib={rep.peak_gpu_reserved_mib} oom={rep.oom_sessions}",
                flush=True,
            )
        finally:
            engine.shutdown()
            gc.collect()

    report = {
        "schema": "morphserve.phase2.m23_suite_a.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_gpus": num_gpus,
        "sessions_per_point": len(suite.sessions),
        "qps_points": list(qps_points),
        "ablation_qps": ablation_qps,
        "qps_sweep": qps_sweep_results,
        "ablations": ablation_results,
        "status": "ok" if all(r["error_sessions"] == 0 for r in qps_sweep_results + ablation_results) else "check_failed",
    }
    out_path = output_dir / "m23-suite-a-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return report


def run_suite_b_skews_and_parity(
    model_dir: Path,
    bench_root: Path,
    output_dir: Path,
    num_gpus: int = 8,
    skew_sessions: int = 24,
    skew_qps: float = 1.0,
    policies: tuple[str, ...] = VALID_POLICIES,
) -> dict[str, Any]:
    """Suite B (1x8 GPUs): Modality-skew sweep (80:20 / 50:50 / 20:80) + GenEval/MMVet lossless parity."""
    output_dir.mkdir(parents=True, exist_ok=True)
    skew_files = (
        ("ar80_dit20", "suites/skew_ar80_dit20.json"),
        ("ar50_dit50", "suites/skew_ar50_dit50.json"),
        ("ar20_dit80", "suites/skew_ar20_dit80.json"),
    )
    skew_suites = [
        (tag, stratify_suite(WorkloadSuite.load_json(bench_root / rel), max_sessions=skew_sessions))
        for tag, rel in skew_files
    ]

    sources_dir = bench_root / "sources"
    burst_deltas = load_burstgpt_interarrivals(sources_dir / "burstgpt_1_sample.csv")
    gentd_deltas = load_gentd26_interarrivals(sources_dir / "gentd26_lora_request_trace.csv")

    skew_results: list[dict[str, Any]] = []
    for preset in policies:
        policy = EnginePolicyConfig.from_preset(preset, num_gpus=num_gpus)
        engine = MorphServeClusterEngine(
            policy=policy,
            model_dir=model_dir,
            bench_root=bench_root,
            tmp_root=Path(f"/tmp/ms_m23b_{preset}"),
        )
        try:
            engine.start()
            for skew_tag, sk_suite in skew_suites:
                rep = engine.run_suite(
                    sk_suite,
                    target_qps=skew_qps,
                    burstgpt_deltas=burst_deltas,
                    gentd26_deltas=gentd_deltas,
                )
                d = rep.to_dict()
                d.pop("sessions", None)
                d["skew_tag"] = skew_tag
                skew_results.append(d)
                print(
                    f"[M2.3-SKEW] policy={preset} skew={skew_tag} goodput={rep.joint_slo_goodput_rps:.3f} "
                    f"joint_slo={rep.joint_slo_attainment_pct}% ship={rep.path_a_ships} morph={rep.path_b_morphs}",
                    flush=True,
                )
        finally:
            engine.shutdown()
            gc.collect()

    acc_suite = WorkloadSuite.load_json(bench_root / "suites/accuracy_geneval_mmvet.json")
    tmp_acc_root = Path("/tmp/ms_m23_parity_outputs")
    shutil.rmtree(tmp_acc_root, ignore_errors=True)
    parity_runs: dict[str, dict[str, dict[str, Any]]] = {}

    try:
        for preset in ("monolithic_dp", "morphserve"):
            pol = EnginePolicyConfig.from_preset(preset, num_gpus=num_gpus)
            eng = MorphServeClusterEngine(
                policy=pol,
                model_dir=model_dir,
                bench_root=bench_root,
                tmp_root=Path(f"/tmp/ms_m23_acc_{preset}"),
            )
            try:
                eng.start()
                rep = eng.run_suite(
                    acc_suite,
                    target_qps=0.4,
                    save_outputs_dir=tmp_acc_root / preset,
                )
                parity_runs[preset] = {s["session_id"]: s for s in rep.sessions}
            finally:
                eng.shutdown()
                gc.collect()

        evaluator = ExactnessEvaluator()
        parity_eval = evaluator.evaluate_parity_and_quality(
            acc_suite.sessions,
            list(parity_runs["monolithic_dp"].values()),
            list(parity_runs["morphserve"].values()),
        )
        geneval_metrics: list[dict[str, Any]] = []
        for s in acc_suite.sessions:
            sid = s.session_id
            ref_s = parity_runs["monolithic_dp"][sid]
            ms_s = parity_runs["morphserve"][sid]
            ref_imgs = ref_s.get("image_paths_out") or []
            ms_imgs = ms_s.get("image_paths_out") or []
            if ref_imgs and ms_imgs and Path(ref_imgs[0]).is_file() and Path(ms_imgs[0]).is_file():
                m = image_pair_metrics(Path(ref_imgs[0]), Path(ms_imgs[0]))
                m["session_id"] = sid
                geneval_metrics.append(m)

        parity_summary = {
            **parity_eval,
            "geneval_sessions": len(geneval_metrics),
            "geneval_sha256_exact_count": sum(1 for m in geneval_metrics if m["sha256_equal"]),
            "geneval_pairs": geneval_metrics,
        }
    finally:
        shutil.rmtree(tmp_acc_root, ignore_errors=True)

    report = {
        "schema": "morphserve.phase2.m23_suite_b.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_gpus": num_gpus,
        "skew_qps": skew_qps,
        "skew_sessions_per_point": skew_sessions,
        "skew_sweep": skew_results,
        "lossless_parity": parity_summary,
        "status": (
            "ok"
            if all(r["error_sessions"] == 0 for r in skew_results) and parity_summary["lossless_pass"]
            else "check_failed"
        ),
    }
    out_path = output_dir / "m23-suite-b-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MorphServe Phase-2 M2.3 BAGEL-7B-MoT Main Evaluation")
    parser.add_argument("--mode", choices=("suite_a", "suite_b"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--max-sessions", type=int, default=28)
    args = parser.parse_args(argv)

    if args.mode == "suite_a":
        rep = run_suite_a_qps_and_ablations(
            model_dir=args.model_dir,
            bench_root=args.bench_root,
            output_dir=args.output_dir,
            num_gpus=args.num_gpus,
            max_sessions=args.max_sessions,
        )
    else:
        rep = run_suite_b_skews_and_parity(
            model_dir=args.model_dir,
            bench_root=args.bench_root,
            output_dir=args.output_dir,
            num_gpus=args.num_gpus,
            skew_sessions=min(24, args.max_sessions),
        )
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
