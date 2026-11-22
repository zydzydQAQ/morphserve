"""Phase-2 Milestone M2.4: Unified `MorphServeClusterEngine` Dual-Regime `InternVL-U` (`bridge_mllm_dit`) Suite."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Any

from morphserve.bench.exactness import ExactnessEvaluator
from morphserve.bench.interleaved_bench import InterleavedSession, InterleavedTurn, WorkloadSuite
from morphserve.engine import EnginePolicyConfig, MorphServeClusterEngine


def build_internvlu_suite(num_sessions: int = 48, offered_qps: float = 1.6) -> WorkloadSuite:
    """Construct bursty mixed VLM + 512/1024 MMDiT suite exercising `bridge_mllm_dit` routing."""
    prompts_img = (
        "A cinematic minimalist Japanese tea room overlooking a snowy bamboo garden at dusk.",
        "Macro photograph of a mechanical watch movement with sapphire gears on dark slate.",
        "Watercolor architectural sketch of a coastal lighthouse under dramatic cumulus clouds.",
        "Studio product shot of a ceramic pour-over coffee dripper beside roasted beans.",
    )
    prompts_vqa = (
        "Summarize in two sentences why compact bridge conditioning lowers cross-GPU state transfer cost.",
        "List three differences between autoregressive causal decoding and bidirectional diffusion steps.",
    )
    sessions: list[InterleavedSession] = []
    t_off = 0.0
    dt_base = 1.0 / max(0.1, float(offered_qps))
    for i in range(int(num_sessions)):
        if i > 0:
            # Deterministic bursty arrivals (clusters of 4 tight requests followed by pause)
            t_off += dt_base * (0.25 if (i % 4 != 0) else 2.95)
        if i % 3 == 0:
            sessions.append(
                InterleavedSession(
                    session_id=f"ivlu_vqa_{i:03d}",
                    quadrant="multi_image_vqa",
                    source_dataset="internvlu_burst",
                    arrival_offset_s=round(t_off, 3),
                    turns=[
                        InterleavedTurn(
                            turn_idx=0,
                            modality_plan="ar_only",
                            prompt=prompts_vqa[i % len(prompts_vqa)],
                            max_ar_tokens=32,
                        )
                    ],
                )
            )
        else:
            res = 1024 if (i % 6 == 1) else 512
            sessions.append(
                InterleavedSession(
                    session_id=f"ivlu_gen_{i:03d}",
                    quadrant="highres_t2i" if res == 1024 else "visual_cot",
                    source_dataset="internvlu_burst",
                    arrival_offset_s=round(t_off, 3),
                    turns=[
                        InterleavedTurn(
                            turn_idx=0,
                            modality_plan="dit_t2i",
                            prompt=prompts_img[i % len(prompts_img)],
                            height=res,
                            width=res,
                            dit_steps=16,
                        )
                    ],
                )
            )
    return WorkloadSuite(
        name=f"internvlu_dual_regime_n{num_sessions}",
        trace_source="burst_clustered",
        base_qps=offered_qps,
        sessions=sessions,
    )


def run_internvlu_generality_validation(
    model_dir: Path,
    vendor_dir: Path,
    output_dir: Path,
    num_gpus: int = 8,
    num_sessions: int = 48,
    offered_qps: float = 1.6,
    constrained_gib: float = 12.0,
    total_gpu_gib: float = 40.0,
    smoke_1gpu: bool = False,
) -> dict[str, Any]:
    """Drive `InternVL-U` (`bridge_mllm_dit`) through `MorphServeClusterEngine` across Regime A (40GB) & Regime B (12GB)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if smoke_1gpu:
        num_gpus = 1
        num_sessions = min(int(num_sessions), 4)
    suite = build_internvlu_suite(num_sessions=num_sessions, offered_qps=offered_qps)
    parity_slice = WorkloadSuite(
        name="internvlu_parity_4",
        trace_source="deterministic",
        base_qps=0.5,
        sessions=suite.sessions[:4],
    )

    regime_a_policies = (
        ("morphserve", "monolithic_dp")
        if smoke_1gpu
        else ("morphserve", "monolithic_dp", "static_nccl_fission", "naive_host_offload")
    )
    regime_a_results: list[dict[str, Any]] = []
    parity_outcomes: dict[str, list[dict[str, Any]]] = {}

    for pol_name in regime_a_policies:
        cfg = EnginePolicyConfig.from_preset(
            pol_name,
            num_gpus=num_gpus,
            model_family="internvl_u",
            vendor_dir=str(vendor_dir),
            inter_turn_think_s_range=(0.0, 0.0) if smoke_1gpu else (0.5, 1.5),
        )
        eng = MorphServeClusterEngine(
            policy=cfg,
            model_dir=model_dir,
            bench_root=output_dir,
            tmp_root=output_dir / f"_tmp_ivlu_a_{pol_name}",
        )
        try:
            eng.start()
            if smoke_1gpu:
                p_dir = output_dir / f"parity_{pol_name}"
                p_rep = eng.run_suite(parity_slice, target_qps=0.5, save_outputs_dir=p_dir)
                d = p_rep.to_dict()
                d.pop("sessions", None)
                d["regime"] = "smoke_1gpu_parity"
                regime_a_results.append(d)
                parity_outcomes[pol_name] = p_rep.sessions
            else:
                rep = eng.run_suite(suite, target_qps=offered_qps)
                d = rep.to_dict()
                d.pop("sessions", None)
                d["regime"] = "regime_a_40gb_burst"
                regime_a_results.append(d)

                if pol_name in ("morphserve", "monolithic_dp"):
                    p_dir = output_dir / f"parity_{pol_name}"
                    p_rep = eng.run_suite(parity_slice, target_qps=0.5, save_outputs_dir=p_dir)
                    parity_outcomes[pol_name] = p_rep.sessions
        finally:
            eng.shutdown()
            gc.collect()

    # Regime B: Memory-Constrained Quota (`12 GiB` per GPU via `gpu_memory_fraction`)
    mem_frac = round(min(1.0, max(0.15, float(constrained_gib) / float(total_gpu_gib))), 4)
    regime_b_results: list[dict[str, Any]] = []
    for pol_name in (() if smoke_1gpu else ("morphserve", "monolithic_dp", "naive_host_offload")):
        overrides: dict[str, Any] = {
            "model_family": "internvl_u",
            "vendor_dir": str(vendor_dir),
            "gpu_memory_fraction": mem_frac,
            "inter_turn_think_s_range": (0.5, 1.5),
        }
        if pol_name == "morphserve":
            overrides["super_pool_mib"] = 4864
            overrides["allow_vmm_overflow"] = False
            overrides["initial_alphas"] = {g: (1.0 if g % 2 == 1 else 0.5) for g in range(num_gpus)}
        cfg = EnginePolicyConfig.from_preset(pol_name, num_gpus=num_gpus, **overrides)
        eng = MorphServeClusterEngine(
            policy=cfg,
            model_dir=model_dir,
            bench_root=output_dir,
            tmp_root=output_dir / f"_tmp_ivlu_b_{pol_name}",
        )
        try:
            eng.start()
            rep = eng.run_suite(suite, target_qps=round(offered_qps * 0.75, 2))
            d = rep.to_dict()
            d.pop("sessions", None)
            d["regime"] = f"regime_b_{int(constrained_gib)}gib_capped"
            regime_b_results.append(d)
        finally:
            eng.shutdown()
            gc.collect()

    evaluator = ExactnessEvaluator()
    parity = evaluator.evaluate_parity_and_quality(
        parity_slice.sessions,
        parity_outcomes.get("monolithic_dp", []),
        parity_outcomes.get("morphserve", []),
    )

    report = {
        "schema": "morphserve.osdi27.m24_internvlu.v2",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_gpus": num_gpus,
        "regime_a_40gb_burst": regime_a_results,
        "regime_b_memory_capped": regime_b_results,
        "lossless_parity": parity,
        "status": "ok" if parity["lossless_pass"] else "check_failed",
    }
    out_path = output_dir / "m24-internvlu-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MorphServe Unified InternVL-U Dual-Regime Benchmark")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--num-sessions", type=int, default=48)
    parser.add_argument("--qps", type=float, default=1.6)
    parser.add_argument("--smoke-1gpu", action="store_true")
    args = parser.parse_args(argv)
    rep = run_internvlu_generality_validation(
        model_dir=args.model_dir,
        vendor_dir=args.vendor_dir,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        num_sessions=args.num_sessions,
        offered_qps=args.qps,
        smoke_1gpu=args.smoke_1gpu,
    )
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
