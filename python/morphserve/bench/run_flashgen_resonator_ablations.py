"""Controlled simulator ablations for Resonator-inspired 3D Performance Atlas,
FlashGen-Sched-inspired Memory+Slack GPU queue reordering, and FlashGen-Cache-inspired
Think-Time KV Swap vs. History Recomputation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from morphserve._core import call_json, encode_json, get_core_lib
from morphserve.sim.cluster_simulator import (
    HardwareInterconnectSpec,
    MeasuredPhysicsProfile,
    WorkloadScaleSpec,
    simulate_cluster_point,
)

MIB = 1024**2


def run_atlas_prediction_accuracy_table() -> dict[str, Any]:
    """Compare exposed H2D stall predictions against live Stage-1 A100 ground truth (`tab:eval_microbench`)."""
    L = get_core_lib()
    # Ground-truth live A100 Stage-1 measurements from `06-evaluation.tex` Table `tab:eval_microbench`.
    benchmarks = [
        {"label": "1024^2 steady (alpha=0.00)", "res": 1024, "alpha": 0.00, "post_evict": False, "actual_ms": 173.97},
        {"label": "1024^2 steady (alpha=0.50)", "res": 1024, "alpha": 0.50, "post_evict": False, "actual_ms": 7.53},
        {"label": "1024^2 post-evict (alpha=0.75)", "res": 1024, "alpha": 0.75, "post_evict": True, "actual_ms": 267.69},
        {"label": "512^2 steady (alpha=0.75)", "res": 512, "alpha": 0.75, "post_evict": False, "actual_ms": 41.02},
        {"label": "512^2 steady (alpha=0.50)", "res": 512, "alpha": 0.50, "post_evict": False, "actual_ms": 312.26},
        {"label": "512^2 steady (alpha=0.00)", "res": 512, "alpha": 0.00, "post_evict": False, "actual_ms": 998.17},
    ]
    rows: list[dict[str, Any]] = []
    mae_by_mode: dict[str, list[float]] = {"legacy_analytical": [], "pipeline_analytical": [], "atlas_3d": []}
    for b in benchmarks:
        row: dict[str, Any] = dict(b)
        for mode in ("legacy_analytical", "pipeline_analytical", "atlas_3d"):
            est = call_json(
                L,
                L.ms_atlas_estimate_json,
                encode_json(
                    {
                        "cost_model_mode": mode,
                        "model_family": "bagel",
                        "resolution": b["res"],
                        "alpha_g": b["alpha"],
                        "is_post_evict": b["post_evict"],
                    }
                ),
            )
            pred = round(float(est["exposed_stall_ms"]), 2)
            err = round(abs(pred - b["actual_ms"]), 2)
            row[f"{mode}_ms"] = pred
            row[f"{mode}_abs_err_ms"] = err
            mae_by_mode[mode].append(err)
        rows.append(row)

    summary = {m: round(sum(v) / max(1, len(v)), 2) for m, v in mae_by_mode.items()}
    return {"points": rows, "mae_ms": summary}


def _summarize(tag: str, rep: dict[str, Any]) -> dict[str, Any]:
    return {
        "tag": tag,
        "cost_model_mode": rep.get("cost_model_mode"),
        "gpu_queue_mode": rep.get("gpu_queue_mode"),
        "kv_eviction_mode": rep.get("kv_eviction_mode"),
        "oom_sessions": int(rep.get("oom_sessions", 0)),
        "steady_turn_slo_attainment_pct": float(rep.get("steady_turn_slo_attainment_pct", 0.0)),
        "steady_turn_goodput_tps": float(rep.get("steady_turn_goodput_tps", 0.0)),
        "joint_slo_attainment_pct": float(rep.get("joint_slo_attainment_pct", 0.0)),
        "ar_ttft_mean_ms": float(rep.get("ar_ttft_mean_ms", 0.0)),
        "ar_ttft_p95_ms": float(rep.get("ar_ttft_p95_ms", 0.0)),
        "ar_tpot_mean_ms": float(rep.get("ar_tpot_mean_ms", 0.0)),
        "ar_tpot_p95_ms": float(rep.get("ar_tpot_p95_ms", 0.0)),
        "dit_turn_mean_ms": float(rep.get("dit_turn_mean_ms", 0.0)),
        "dit_turn_p95_ms": float(rep.get("dit_turn_p95_ms", 0.0)),
        "dit_turn_p99_ms": float(rep.get("dit_turn_p99_ms", 0.0)),
        "active_jct_mean_ms": float(rep.get("active_jct_mean_ms", 0.0)),
        "active_jct_p95_ms": float(rep.get("active_jct_p95_ms", 0.0)),
        "effective_hbm_util_mean_pct": rep.get("effective_hbm_util_mean_pct"),
        "path_a_ships": int(rep.get("path_a_ships", 0)),
        "path_b_morphs": int(rep.get("path_b_morphs", 0)),
        "queue_reorders_total": int(rep.get("queue_reorders_total", 0)),
        "queue_memory_bypasses_total": int(rep.get("queue_memory_bypasses_total", 0)),
        "queue_starvation_promotions_total": int(rep.get("queue_starvation_promotions_total", 0)),
        "kv_host_swap_outs": int(rep.get("kv_host_swap_outs", 0)),
        "kv_host_swap_ins": int(rep.get("kv_host_swap_ins", 0)),
        "kv_swap_hit_rate_pct": float(rep.get("kv_swap_hit_rate_pct", 0.0)),
        "kv_swap_in_mean_ms": float(rep.get("kv_swap_in_mean_ms", 0.0)),
        "kv_swap_in_p95_ms": float(rep.get("kv_swap_in_p95_ms", 0.0)),
        "kv_recompute_events": int(rep.get("kv_recompute_events", 0)),
        "kv_recompute_total_ms": float(rep.get("kv_recompute_total_ms", 0.0)),
        "kv_saved_recompute_ms": float(rep.get("kv_saved_recompute_ms", 0.0)),
    }


def run_all_ablations(bench_root: str = "") -> dict[str, Any]:
    ph = MeasuredPhysicsProfile()
    hw_8a100 = HardwareInterconnectSpec.presets(ph)[0]
    ws_std = WorkloadScaleSpec.bagel_7b_standard()

    # 1. Stage-1 Stall Prediction Accuracy (Resonator Performance Atlas vs. Analytical Formula)
    atlas_acc = run_atlas_prediction_accuracy_table()

    # 2. Ablation A: Cost Model Routing (legacy_analytical vs pipeline_analytical vs atlas_3d)
    # 8xA100 mixed workload with cold/partial-alpha heterogeneous GPUs & heavy multi-turn 1024^2 edits.
    ablation_a: list[dict[str, Any]] = []
    init_het_alphas = {str(g): (0.0 if (g % 2 == 0) else 0.5) for g in range(8)}
    for mode in ("legacy_analytical", "pipeline_analytical", "atlas_3d"):
        rep = simulate_cluster_point(
            policy_name="morphserve",
            num_gpus=8,
            hw=hw_8a100,
            offered_qps=1.0,
            num_sessions=160,
            seed=2026,
            physics=ph,
            workload_scale=ws_std,
            skew_tag="main_mix",
            bench_root=bench_root,
            policy_overrides={
                "cost_model_mode": mode,
                "gpu_queue_mode": "fcfs",
                "kv_eviction_mode": "swap",
                "initial_alphas": init_het_alphas,
                "steady_window_mode": "session_trim",
                "steady_state_trim_frac": 0.10,
            },
        )
        ablation_a.append(_summarize(f"cost_{mode}", rep))

    # 3. Ablation B: Per-GPU Queue Reordering (fcfs vs memory_slack) at high contention (1.2 QPS, 8xA100)
    ablation_b: list[dict[str, Any]] = []
    for q_mode in ("fcfs", "memory_slack"):
        rep = simulate_cluster_point(
            policy_name="morphserve",
            num_gpus=8,
            hw=hw_8a100,
            offered_qps=1.2,
            num_sessions=192,
            seed=2026,
            physics=ph,
            workload_scale=ws_std,
            skew_tag="main_mix",
            bench_root=bench_root,
            policy_overrides={
                "cost_model_mode": "atlas_3d",
                "gpu_queue_mode": q_mode,
                "kv_eviction_mode": "swap",
                "steady_window_mode": "session_trim",
                "steady_state_trim_frac": 0.10,
            },
        )
        ablation_b.append(_summarize(f"queue_{q_mode}", rep))

    # 4. Ablation C: Think-Time KV Eviction (swap vs recompute) under tight-memory multi-turn pressure
    # 8x36 GiB constrained HBM with extended multi-turn editing & CoT sessions where think-time KV eviction triggers.
    hw_tight8 = HardwareInterconnectSpec(
        name="A100_36GB_Pressure_8GPU",
        gpu_hbm_gib=36.0,
        compute_speedup_x=1.0,
        solo_h2d_gbps=hw_8a100.solo_h2d_gbps,
        agg_h2d_gbps=hw_8a100.agg_h2d_gbps,
        intra_half_p2p_gbps=hw_8a100.intra_half_p2p_gbps,
        inter_socket_p2p_gbps=hw_8a100.inter_socket_p2p_gbps,
        inter_node_rdma_gbps=hw_8a100.inter_node_rdma_gbps,
        gpus_per_node=8,
    )
    ws_long_mt = WorkloadScaleSpec(
        name="bagel_7b_multiturn_pressure",
        weight_scale=1.0,
        kv_per_turn_scale=1.5,
        vae_spike_scale=1.0,
        compute_scale=1.0,
        extra_edit_turns=2,
    )
    ablation_c: list[dict[str, Any]] = []
    for kv_mode in ("swap", "recompute"):
        rep = simulate_cluster_point(
            policy_name="morphserve",
            num_gpus=8,
            hw=hw_tight8,
            offered_qps=1.0,
            num_sessions=160,
            seed=2026,
            physics=ph,
            workload_scale=ws_long_mt,
            skew_tag="main_mix",
            bench_root=bench_root,
            policy_overrides={
                "cost_model_mode": "atlas_3d",
                "gpu_queue_mode": "memory_slack",
                "kv_eviction_mode": kv_mode,
                "steady_window_mode": "session_trim",
                "steady_state_trim_frac": 0.10,
            },
        )
        ablation_c.append(_summarize(f"kv_{kv_mode}", rep))

    return {
        "source": "calibrated_cpp_cluster_simulator",
        "atlas_stall_prediction_accuracy": atlas_acc,
        "ablation_a_cost_model_8gpu": ablation_a,
        "ablation_b_queue_reordering_8gpu": ablation_b,
        "ablation_c_kv_swap_vs_recompute_8gpu": ablation_c,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    res = run_all_ablations(bench_root=args.bench_root)
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    print("=== 1. Stage-1 A100 H2D Stall Prediction MAE (ms) ===")
    print(json.dumps(res["atlas_stall_prediction_accuracy"]["mae_ms"], indent=2))
    print("\n=== 2. Ablation A: Cost Model (8xA100, 1.0 QPS, heterogeneous alpha) ===")
    for r in res["ablation_a_cost_model_8gpu"]:
        print(
            f"{r['tag']:26s} | OOM={r['oom_sessions']} | TurnSLO={r['steady_turn_slo_attainment_pct']:5.1f}% "
            f"| DiT mean/p95={r['dit_turn_mean_ms']:7.1f}/{r['dit_turn_p95_ms']:7.1f} ms "
            f"| PathA={r['path_a_ships']:4d} PathB={r['path_b_morphs']:4d} "
            f"| EffHBM={r['effective_hbm_util_mean_pct']}%"
        )
    print("\n=== 3. Ablation B: Per-GPU Queue Reordering (8xA100, 1.2 QPS) ===")
    for r in res["ablation_b_queue_reordering_8gpu"]:
        print(
            f"{r['tag']:26s} | OOM={r['oom_sessions']} | TurnSLO={r['steady_turn_slo_attainment_pct']:5.1f}% "
            f"| DiT mean/p95/p99={r['dit_turn_mean_ms']:7.1f}/{r['dit_turn_p95_ms']:7.1f}/{r['dit_turn_p99_ms']:7.1f} ms "
            f"| Reord={r['queue_reorders_total']:4d} MemByp={r['queue_memory_bypasses_total']:3d} "
            f"Starv={r['queue_starvation_promotions_total']:3d} | EffHBM={r['effective_hbm_util_mean_pct']}%"
        )
    print("\n=== 4. Ablation C: Think-Time KV Swap vs. Recompute (8x36GB pressure, 1.0 QPS) ===")
    for r in res["ablation_c_kv_swap_vs_recompute_8gpu"]:
        print(
            f"{r['tag']:26s} | OOM={r['oom_sessions']} | TurnSLO={r['steady_turn_slo_attainment_pct']:5.1f}% "
            f"| TTFT mean/p95={r['ar_ttft_mean_ms']:6.1f}/{r['ar_ttft_p95_ms']:6.1f} ms "
            f"| DiT mean/p95={r['dit_turn_mean_ms']:7.1f}/{r['dit_turn_p95_ms']:7.1f} ms "
            f"| SwapIn={r['kv_host_swap_ins']:3d} (p95={r['kv_swap_in_p95_ms']:5.1f}ms, hit={r['kv_swap_hit_rate_pct']:5.1f}%) "
            f"| RecompEv={r['kv_recompute_events']:3d} RecompMs={r['kv_recompute_total_ms']:8.1f} SavedMs={r['kv_saved_recompute_ms']:8.1f}"
        )


if __name__ == "__main__":
    main()
