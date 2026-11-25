"""Phase-2 Milestone M2.3: 8xA100 BAGEL-7B-MoT Steady-State (`n=80-96`) Evaluation, Skew, Ablations & M3 Parity."""

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
from typing import Any

from morphserve.bench.exactness import (
    ExactnessEvaluator,
    build_m3_active_accuracy_suite,
    image_pair_metrics,
)
from morphserve.bench.interleaved_bench import (
    InterleavedSession,
    InterleavedTurn,
    WorkloadSuite,
    load_burstgpt_interarrivals,
    load_gentd26_interarrivals,
)
from morphserve.engine import EnginePolicyConfig, MorphServeClusterEngine, VALID_POLICIES

DEFAULT_MAIN_SESSIONS = 96
DEFAULT_ABLATION_SESSIONS = 80
DEFAULT_SKEW_SESSIONS = 80


def _resolve_sources_dir(bench_root: Path) -> Path:
    for cand in (bench_root / "sources", bench_root.parent / "bench_sources"):
        if cand.is_dir():
            return cand
    return bench_root / "sources"


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
        elif all(counts[ k ] >= len(by_quad[k]) for k in counts):
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
    )


def _local_tmp_base() -> Path:
    raw = os.environ.get("MORPHSERVE_LOCAL_TMP", "").strip()
    return Path(raw) if raw else Path("/tmp")


def scale_suite_sessions(suite: WorkloadSuite, target_sessions: int) -> WorkloadSuite:
    """Align `suite` to steady-state `target_sessions` (`n=80-96`), downsampling or cyclically expanding by quadrant."""
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
                )
            )

    if rem > 0:
        tail_slice = stratify_suite(
            WorkloadSuite(
                name=suite.name,
                trace_source=suite.trace_source,
                base_qps=suite.base_qps,
                sessions=ordered_base,
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
                )
            )

    expanded.sort(key=lambda s: (s.arrival_offset_s, s.session_id))
    return WorkloadSuite(
        name=f"{suite.name}_n{len(expanded)}",
        trace_source=suite.trace_source,
        base_qps=suite.base_qps,
        sessions=expanded,
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


def _compute_sustainable_qps_summary(qps_sweep_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize primary OSDI/SOSP `SLO Attainment (%) vs. QPS` curves and `@80% / @90% SLO` sustainable capacity."""
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
        full_pts = [
            (float(r["offered_qps"]), float(r["joint_slo_attainment_pct"]))
            for r in rows_sorted
        ]
        grid_80 = max((q for q, a in full_pts if a >= 80.0), default=0.0)
        grid_90 = max((q for q, a in full_pts if a >= 90.0), default=0.0)
        curve = []
        for r in rows_sorted:
            tot_s = max(1, int(r.get("total_sessions", 1)))
            ok_s = int(r.get("completed_sessions", tot_s - int(r.get("oom_sessions", 0)) - int(r.get("error_sessions", 0))))
            surv_pct = float(r.get("survival_rate_pct", round(100.0 * ok_s / tot_s, 1)))
            curve.append(
                {
                    "offered_qps": float(r["offered_qps"]),
                    "joint_slo_attainment_pct": float(r["joint_slo_attainment_pct"]),
                    "steady_state_attainment_pct": float(r.get("steady_state_attainment_pct", r["joint_slo_attainment_pct"])),
                    "survival_rate_pct": surv_pct,
                    "oom_sessions": int(r.get("oom_sessions", 0)),
                    "total_sessions": tot_s,
                    "ar_ttft_p95_ms": float(r.get("ar_ttft_p95_ms", 0.0)),
                    "ar_tpot_p95_ms": float(r.get("ar_tpot_p95_ms", 0.0)),
                    "survivor_dit_turn_p95_ms": float(r.get("dit_turn_p95_ms", 0.0)),
                    "all_session_dit_p95_ms": r.get("all_session_dit_p95_ms"),
                    "all_session_e2e_p95_ms": r.get("all_session_e2e_p95_ms"),
                    "peak_nvml_used_mib": float(r.get("peak_nvml_used_mib", r.get("peak_gpu_reserved_mib", 0.0))),
                }
            )
        summary[pol] = {
            "primary_metric": "slo_attainment_pct_vs_offered_qps",
            "max_sustainable_qps_80pct": _interpolate_slo_threshold_qps(full_pts, 80.0),
            "max_sustainable_qps_90pct": _interpolate_slo_threshold_qps(full_pts, 90.0),
            "steady_max_sustainable_qps_80pct": _interpolate_slo_threshold_qps(steady_pts, 80.0),
            "steady_max_sustainable_qps_90pct": _interpolate_slo_threshold_qps(steady_pts, 90.0),
            "max_grid_qps_80pct": round(grid_80, 3),
            "max_grid_qps_90pct": round(grid_90, 3),
            "min_survival_rate_pct": min((c["survival_rate_pct"] for c in curve), default=100.0),
            "total_oom_sessions": sum(c["oom_sessions"] for c in curve),
            "max_nvml_peak_used_mib": max((c["peak_nvml_used_mib"] for c in curve), default=0.0),
            "slo_attainment_curve": curve,
        }
    return summary


def run_suite_a_qps_and_ablations(
    model_dir: Path,
    bench_root: Path,
    output_dir: Path,
    num_gpus: int = 8,
    max_sessions: int = DEFAULT_MAIN_SESSIONS,
    ablation_sessions: int = DEFAULT_ABLATION_SESSIONS,
    qps_points: tuple[float, ...] = (0.4, 0.8, 1.1, 1.4),
    ablation_qps: float = 1.1,
    policies: tuple[str, ...] = VALID_POLICIES,
) -> dict[str, Any]:
    """Suite A (1x8 GPUs): Steady-state `n=96` 4-quadrant QPS sweep + `n=80` M1-M4 ablation ladder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_main = WorkloadSuite.load_json(bench_root / "suites/main_mix_80.json")
    suite = scale_suite_sessions(raw_main, target_sessions=max_sessions)
    eff_ab_sessions = min(max_sessions, ablation_sessions) if max_sessions < ablation_sessions else ablation_sessions
    ablation_suite = scale_suite_sessions(raw_main, target_sessions=eff_ab_sessions)

    sources_dir = _resolve_sources_dir(bench_root)
    burst_deltas = load_burstgpt_interarrivals(sources_dir / "burstgpt_1_sample.csv")
    gentd_deltas = load_gentd26_interarrivals(sources_dir / "gentd26_lora_request_trace.csv")

    tmp_base = _local_tmp_base()
    qps_sweep_results: list[dict[str, Any]] = []
    for preset in policies:
        policy = EnginePolicyConfig.from_preset(preset, num_gpus=num_gpus)
        engine = MorphServeClusterEngine(
            policy=policy,
            model_dir=model_dir,
            bench_root=bench_root,
            tmp_root=tmp_base / f"ms_m23a_{preset}",
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
                err_samples = [str(s.get("error", ""))[:240] for s in rep.sessions if s.get("status") == "error"][:5]
                if err_samples:
                    d["error_samples"] = err_samples
                d["cdf_outcomes"] = [
                    {
                        "status": s.get("status", "error"),
                        "quadrant": s.get("quadrant", ""),
                        "e2e_ms": s.get("e2e_ms", 0.0),
                        "dit_turns_ms": s.get("dit_turns_ms", []),
                        "ar_ttfts_ms": s.get("ar_ttfts_ms", []),
                    }
                    for s in rep.sessions
                ]
                d.pop("sessions", None)
                d["host_shmem_delta_gib"] = init_info.get("host_shmem_delta_gib", rep.host_shmem_delta_gib)
                qps_sweep_results.append(d)
                print(
                    f"[M2.3-QPS] policy={preset} n={len(suite.sessions)} qps={qps} "
                    f"steady_goodput={rep.steady_state_goodput_rps:.3f} "
                    f"steady_slo={rep.steady_state_attainment_pct}% ar_tpot_p95={rep.ar_tpot_p95_ms}ms "
                    f"dit_p95={rep.dit_turn_p95_ms}ms oom={rep.oom_sessions} err={rep.error_sessions}"
                    + (f" sample_err={err_samples[0]!r}" if err_samples else ""),
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
            tmp_root=tmp_base / f"ms_m23ab_{ab_name}",
        )
        try:
            engine.start()
            rep = engine.run_suite(
                ablation_suite,
                target_qps=ablation_qps,
                burstgpt_deltas=burst_deltas,
                gentd26_deltas=gentd_deltas,
            )
            d = rep.to_dict()
            err_samples = [str(s.get("error", ""))[:240] for s in rep.sessions if s.get("status") == "error"][:5]
            if err_samples:
                d["error_samples"] = err_samples
            d.pop("sessions", None)
            d["ablation_name"] = ab_name
            ablation_results.append(d)
            print(
                f"[M2.3-ABLATION] {ab_name} n={len(ablation_suite.sessions)} qps={ablation_qps} "
                f"steady_goodput={rep.steady_state_goodput_rps:.3f} "
                f"steady_slo={rep.steady_state_attainment_pct}% ar_ttft_p99={rep.ar_ttft_p99_ms}ms "
                f"peak_mib={rep.peak_gpu_reserved_mib} yields={rep.total_yields} oom={rep.oom_sessions} err={rep.error_sessions}"
                + (f" sample_err={err_samples[0]!r}" if err_samples else ""),
                flush=True,
            )
        finally:
            engine.shutdown()
            gc.collect()

    report = {
        "schema": "morphserve.osdi27.m23_suite_a.v2",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_gpus": num_gpus,
        "sessions_per_point": len(suite.sessions),
        "ablation_sessions_per_point": len(ablation_suite.sessions),
        "qps_points": list(qps_points),
        "ablation_qps": ablation_qps,
        "sustainable_qps_summary": _compute_sustainable_qps_summary(qps_sweep_results),
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
    skew_sessions: int = DEFAULT_SKEW_SESSIONS,
    skew_qps: float = 1.0,
    policies: tuple[str, ...] = VALID_POLICIES,
    allow_eval_download: bool = False,
) -> dict[str, Any]:
    """Suite B (1x8 GPUs): Steady-state `n=80` Modality-Skew Sweep + Multi-Turn M3-Active Downstream Accuracy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    skew_files = (
        ("ar80_dit20", "suites/skew_ar80_dit20.json"),
        ("ar50_dit50", "suites/skew_ar50_dit50.json"),
        ("ar20_dit80", "suites/skew_ar20_dit80.json"),
    )
    skew_suites = [
        (tag, scale_suite_sessions(WorkloadSuite.load_json(bench_root / rel), target_sessions=skew_sessions))
        for tag, rel in skew_files
    ]

    sources_dir = _resolve_sources_dir(bench_root)
    burst_deltas = load_burstgpt_interarrivals(sources_dir / "burstgpt_1_sample.csv")
    gentd_deltas = load_gentd26_interarrivals(sources_dir / "gentd26_lora_request_trace.csv")

    tmp_base = _local_tmp_base()
    skew_results: list[dict[str, Any]] = []
    for preset in policies:
        policy = EnginePolicyConfig.from_preset(preset, num_gpus=num_gpus)
        engine = MorphServeClusterEngine(
            policy=policy,
            model_dir=model_dir,
            bench_root=bench_root,
            tmp_root=tmp_base / f"ms_m23b_{preset}",
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
                err_samples = [str(s.get("error", ""))[:240] for s in rep.sessions if s.get("status") == "error"][:5]
                if err_samples:
                    d["error_samples"] = err_samples
                d.pop("sessions", None)
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

    acc_suite = build_m3_active_accuracy_suite(bench_root)
    tmp_acc_root = tmp_base / "ms_m23_parity_outputs"
    shutil.rmtree(tmp_acc_root, ignore_errors=True)
    parity_runs: dict[str, dict[str, dict[str, Any]]] = {}

    try:
        for preset in ("monolithic_dp", "morphserve"):
            pol = EnginePolicyConfig.from_preset(
                preset,
                num_gpus=num_gpus,
                inter_turn_think_s_range=(0.0, 0.0),
            )
            eng = MorphServeClusterEngine(
                policy=pol,
                model_dir=model_dir,
                bench_root=bench_root,
                tmp_root=tmp_base / f"ms_m23_acc_{preset}",
            )
            try:
                eng.start()
                rep = eng.run_suite(
                    acc_suite,
                    target_qps=0.20,
                    save_outputs_dir=tmp_acc_root / preset,
                )
                parity_runs[preset] = {s["session_id"]: s for s in rep.sessions}
                acc_err = [str(s.get("error", ""))[:240] for s in rep.sessions if s.get("status") == "error"][:5]
                print(
                    f"[M2.3-PARITY] policy={preset} n={len(acc_suite.sessions)} "
                    f"ok={rep.completed_sessions} oom={rep.oom_sessions} err={rep.error_sessions}"
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
        parity_eval = evaluator.evaluate_parity_and_quality(
            acc_suite.sessions,
            list(parity_runs["monolithic_dp"].values()),
            list(parity_runs["morphserve"].values()),
            bench_root=bench_root,
        )
        geneval_metrics: list[dict[str, Any]] = []
        for s in acc_suite.sessions:
            if s.source_dataset != "geneval":
                continue
            sid = s.session_id
            ref_s = parity_runs["monolithic_dp"].get(sid, {})
            ms_s = parity_runs["morphserve"].get(sid, {})
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
        ms_errs = [
            f"{sid}: {str(s.get('error', ''))[:200]}"
            for sid, s in parity_runs["morphserve"].items()
            if s.get("status") == "error"
        ]
        parity_summary = {
            **parity_eval,
            "accuracy_suite_name": acc_suite.name,
            "accuracy_total_sessions": len(acc_suite.sessions),
            "ref_error_sessions": len(ref_errs),
            "morphserve_error_sessions": len(ms_errs),
            "ref_error_samples": ref_errs[:5],
            "morphserve_error_samples": ms_errs[:5],
            "visual_cot_geneval_sessions": sum(1 for s in acc_suite.sessions if s.quadrant == "visual_cot"),
            "multiturn_4turn_edit_sessions": sum(
                1 for s in acc_suite.sessions if s.quadrant == "multiturn_edit" and len(s.turns) == 4
            ),
            "multiturn_mmvet_sessions": sum(1 for s in acc_suite.sessions if s.quadrant == "multi_image_vqa"),
            "geneval_sessions": len(geneval_metrics),
            "geneval_sha256_exact_count": sum(1 for m in geneval_metrics if m["sha256_equal"]),
            "geneval_pairs": geneval_metrics,
        }
    finally:
        shutil.rmtree(tmp_acc_root, ignore_errors=True)

    report = {
        "schema": "morphserve.osdi27.m23_suite_b.v2",
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


def run_suite_a_lowqps_and_cdf(
    model_dir: Path,
    bench_root: Path,
    output_dir: Path,
    num_gpus: int = 8,
    qps_points: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.40, 1.10),
    knee_sessions: int = 48,
    low_sessions: int = 28,
    policies: tuple[str, ...] = VALID_POLICIES,
    merge_suite_a_report: Path | None = None,
) -> dict[str, Any]:
    """Suite A Low-to-Knee QPS Sweep (`0.05..0.25` + `0.40, 1.10` RPS): `SLO Attainment vs. QPS`, `@80%/90% SLO` QPS & CDFs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_main = WorkloadSuite.load_json(bench_root / "suites/main_mix_80.json")
    suite_low = scale_suite_sessions(raw_main, target_sessions=low_sessions)
    suite_knee = scale_suite_sessions(raw_main, target_sessions=knee_sessions)

    sources_dir = _resolve_sources_dir(bench_root)
    burst_deltas = load_burstgpt_interarrivals(sources_dir / "burstgpt_1_sample.csv")
    gentd_deltas = load_gentd26_interarrivals(sources_dir / "gentd26_lora_request_trace.csv")

    tmp_base = _local_tmp_base()
    qps_sweep_results: list[dict[str, Any]] = []
    for preset in policies:
        policy = EnginePolicyConfig.from_preset(preset, num_gpus=num_gpus)
        engine = MorphServeClusterEngine(
            policy=policy,
            model_dir=model_dir,
            bench_root=bench_root,
            tmp_root=tmp_base / f"ms_m23low_{preset}",
        )
        try:
            init_info = engine.start()
            for qps in qps_points:
                active_suite = suite_low if float(qps) <= 0.10 else suite_knee
                rep = engine.run_suite(
                    active_suite,
                    target_qps=qps,
                    burstgpt_deltas=burst_deltas,
                    gentd26_deltas=gentd_deltas,
                )
                d = rep.to_dict()
                err_samples = [str(s.get("error", ""))[:240] for s in rep.sessions if s.get("status") == "error"][:5]
                if err_samples:
                    d["error_samples"] = err_samples
                d["cdf_outcomes"] = [
                    {
                        "status": s.get("status", "error"),
                        "quadrant": s.get("quadrant", ""),
                        "e2e_ms": s.get("e2e_ms", 0.0),
                        "dit_turns_ms": s.get("dit_turns_ms", []),
                        "ar_ttfts_ms": s.get("ar_ttfts_ms", []),
                    }
                    for s in rep.sessions
                ]
                d.pop("sessions", None)
                d["host_shmem_delta_gib"] = init_info.get("host_shmem_delta_gib", rep.host_shmem_delta_gib)
                qps_sweep_results.append(d)
                print(
                    f"[M2.3-LOWQPS] policy={preset} n={len(active_suite.sessions)} qps={qps} "
                    f"slo={rep.joint_slo_attainment_pct}% steady_slo={rep.steady_state_attainment_pct}% "
                    f"survival={rep.survival_rate_pct}% nvml_peak={rep.peak_nvml_used_mib}MiB "
                    f"ar_tpot_p95={rep.ar_tpot_p95_ms}ms dit_p95={rep.dit_turn_p95_ms}ms "
                    f"all_dit_p95={rep.all_session_dit_p95_ms} oom={rep.oom_sessions} err={rep.error_sessions}"
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
        "schema": "morphserve.osdi27.m23_lowqps_slo_cdf.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary_metric": "slo_attainment_pct_vs_offered_qps",
        "num_gpus": num_gpus,
        "low_sessions_per_point": len(suite_low.sessions),
        "knee_sessions_per_point": len(suite_knee.sessions),
        "qps_points": list(qps_points),
        "sustainable_qps_summary": _compute_sustainable_qps_summary(combined_sweep),
        "qps_sweep": qps_sweep_results,
        "status": "ok" if all(r["error_sessions"] == 0 for r in qps_sweep_results) else "check_failed",
    }
    out_path = output_dir / "m23-lowqps-slo-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MorphServe Phase-2 M2.3 BAGEL-7B-MoT Main Evaluation")
    parser.add_argument("--mode", choices=("suite_a", "suite_a_lowqps", "suite_b"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--max-sessions", type=int, default=DEFAULT_MAIN_SESSIONS)
    parser.add_argument("--low-sessions", type=int, default=28)
    parser.add_argument("--knee-sessions", type=int, default=48)
    parser.add_argument("--qps-points", type=str, default=None)
    parser.add_argument("--merge-suite-a-report", type=Path, default=None)
    parser.add_argument("--skew-sessions", type=int, default=None)
    parser.add_argument("--allow-eval-download", action="store_true")
    args = parser.parse_args(argv)

    parsed_qps = (
        tuple(float(x.strip()) for x in args.qps_points.split(",") if x.strip())
        if args.qps_points
        else None
    )

    if args.mode == "suite_a":
        rep = run_suite_a_qps_and_ablations(
            model_dir=args.model_dir,
            bench_root=args.bench_root,
            output_dir=args.output_dir,
            num_gpus=args.num_gpus,
            max_sessions=args.max_sessions,
            qps_points=parsed_qps or (0.4, 0.8, 1.1, 1.4),
        )
    elif args.mode == "suite_a_lowqps":
        rep = run_suite_a_lowqps_and_cdf(
            model_dir=args.model_dir,
            bench_root=args.bench_root,
            output_dir=args.output_dir,
            num_gpus=args.num_gpus,
            qps_points=parsed_qps or (0.05, 0.10, 0.15, 0.20, 0.25, 0.40, 1.10),
            knee_sessions=min(args.max_sessions, args.knee_sessions),
            low_sessions=min(args.max_sessions, args.low_sessions),
            merge_suite_a_report=args.merge_suite_a_report,
        )
    else:
        eff_skew = (
            int(args.skew_sessions)
            if args.skew_sessions is not None
            else (DEFAULT_SKEW_SESSIONS if args.max_sessions >= DEFAULT_SKEW_SESSIONS else args.max_sessions)
        )
        rep = run_suite_b_skews_and_parity(
            model_dir=args.model_dir,
            bench_root=args.bench_root,
            output_dir=args.output_dir,
            num_gpus=args.num_gpus,
            skew_sessions=eff_skew,
            allow_eval_download=args.allow_eval_download,
        )
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
