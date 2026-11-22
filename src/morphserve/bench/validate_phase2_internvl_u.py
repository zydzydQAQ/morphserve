"""Phase-2 Milestone M2.4: Unified `MorphServeClusterEngine` Dual-Regime `InternVL-U` (`bridge_mllm_dit`) Suite."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
from pathlib import Path
import time
from typing import Any

from morphserve.bench.exactness import ExactnessEvaluator
from morphserve.bench.interleaved_bench import InterleavedSession, InterleavedTurn, WorkloadSuite
from morphserve.engine import EnginePolicyConfig, MorphServeClusterEngine


def build_internvlu_suite(
    num_sessions: int = 48,
    offered_qps: float = 1.6,
    *,
    highres_size: int = 1024,
) -> WorkloadSuite:
    """Construct bursty mixed VLM + MMDiT suite exercising `bridge_mllm_dit` routing."""
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
            res = int(highres_size) if (i % 6 == 1) else 512
            sessions.append(
                InterleavedSession(
                    session_id=f"ivlu_gen_{i:03d}",
                    quadrant="highres_t2i" if res >= 1024 else "visual_cot",
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
        name=f"internvlu_dual_regime_n{num_sessions}_h{highres_size}",
        trace_source="burst_clustered",
        base_qps=offered_qps,
        sessions=sessions,
    )


def _detect_gpu_total_gib(default_gib: float = 40.0) -> float:
    """Query actual device total HBM GiB when CUDA is present (handles ECC 39.5GiB A100 & 80GiB H100 cleanly)."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            tot_b = int(getattr(props, "total_memory", 0))
            if tot_b > 0:
                return round(float(tot_b) / float(1024**3), 4)
    except Exception:
        pass
    return float(default_gib)


def _local_tmp_base() -> Path:
    raw = os.environ.get("MORPHSERVE_LOCAL_TMP", "").strip()
    return Path(raw) if raw else Path("/tmp")


def compute_internvlu_memory_quota_overrides(
    policy_name: str,
    num_gpus: int,
    vendor_dir: Path | str,
    *,
    constrained_gib: float = 12.0,
    total_gpu_gib: float | None = 40.0,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Compute honest per-GPU `12 GiB` quota overrides accounting for BOTH PyTorch and `TriPool-VMM` pools.

    For `morphserve`:
      - `load_internvlu_pipeline(..., defer_decoder_gpu=True)` keeps the `~4.5 GiB` `generation_decoder` on CPU
        during load (`vlm` + `vae` base = `~3,950–5,474 MiB`), then `attach_expert_mirror` populates the shared
        NUMA `memfd` mirror directly.
      - `TriPool-VMM` (`cuMemCreate`, outside PyTorch's caching allocator) is strictly bounded to
        `super_pool_mib=4800` (`300 x 16 MiB` handles covering all 20 decoder blocks and scoped `1024^2`
        `latents_to_pixels` VAE spikes) + `fine_kv_pool_mib=32` = `4,832 MiB` (`allow_vmm_overflow=False`).
      - `InternVLUCPTGAdapter.compute_internvlu_memory_plan` caps PyTorch's allocator to the remaining budget
        (`<= 7,456 MiB`) so `pytorch_peak_reserved_mib + vmm_physical_pool_mib <= 12,288 MiB` (`12.0 GiB`).
    """
    eff_total_gib = _detect_gpu_total_gib(40.0) if total_gpu_gib is None else float(total_gpu_gib)
    quota_mib = round(float(constrained_gib) * 1024.0, 2)
    total_mib = max(quota_mib, eff_total_gib * 1024.0)
    total_quota_frac = round(min(1.0, max(0.05, quota_mib / total_mib)), 6)

    overrides: dict[str, Any] = {
        "model_family": "internvl_u",
        "vendor_dir": str(vendor_dir),
        "inter_turn_think_s_range": (0.5, 1.5),
        "gpu_memory_fraction": total_quota_frac,
    }

    if policy_name == "morphserve":
        super_handle_mib = 16
        super_pool_mib = 4800
        fine_kv_pool_mib = 32
        vmm_pool_mib = float(super_pool_mib + fine_kv_pool_mib)
        runtime_pytorch_budget_mib = max(0.0, quota_mib - vmm_pool_mib)
        overrides.update(
            {
                "super_handle_mib": super_handle_mib,
                "super_pool_mib": super_pool_mib,
                "fine_kv_pool_mib": fine_kv_pool_mib,
                "allow_vmm_overflow": False,
                "initial_alphas": {g: 1.0 for g in range(int(num_gpus))},
            }
        )
        budget_plan = {
            "quota_limit_mib": quota_mib,
            "vmm_physical_pool_mib": vmm_pool_mib,
            "runtime_pytorch_budget_mib": round(runtime_pytorch_budget_mib, 2),
            "runtime_combined_budget_mib": round(runtime_pytorch_budget_mib + vmm_pool_mib, 2),
            "total_quota_fraction": total_quota_frac,
        }
        return overrides, budget_plan

    budget_plan = {
        "quota_limit_mib": quota_mib,
        "vmm_physical_pool_mib": 0.0,
        "runtime_pytorch_budget_mib": quota_mib,
        "runtime_combined_budget_mib": quota_mib,
        "total_quota_fraction": total_quota_frac,
    }
    return overrides, budget_plan


def verify_internvlu_memory_quota(
    run_result: dict[str, Any],
    policy: EnginePolicyConfig,
    *,
    constrained_gib: float = 12.0,
) -> dict[str, Any]:
    """Audit that observed `PyTorch peak_reserved_mib + TriPool-VMM physical_pool_mib <= quota_limit_mib`."""
    quota_limit_mib = round(float(constrained_gib) * 1024.0, 2)
    if "vmm_physical_pool_mib" in run_result:
        vmm_physical_pool_mib = round(float(run_result["vmm_physical_pool_mib"]), 2)
    elif policy.use_tripool_vmm:
        base_vmm_mib = float(policy.super_pool_mib + policy.fine_kv_pool_mib)
        overflow_mib = float(int(run_result.get("vmm_overflow_creates", 0)) * policy.super_handle_mib)
        vmm_physical_pool_mib = round(base_vmm_mib + overflow_mib, 2)
    else:
        vmm_physical_pool_mib = 0.0

    if "pytorch_peak_reserved_mib" in run_result or "pytorch_reserved_mib" in run_result:
        pt_raw = run_result.get("pytorch_peak_reserved_mib", run_result.get("pytorch_reserved_mib", 0.0))
        pytorch_reserved_mib = round(float(pt_raw), 2)
        combined_physical_mib = round(pytorch_reserved_mib + vmm_physical_pool_mib, 2)
    elif policy.use_tripool_vmm:
        # `InternVLUCPTGAdapter` reports `peak_reserved_mib = pytorch_peak + vmm_physical_pool_mib`
        combined_physical_mib = round(float(run_result.get("peak_gpu_reserved_mib", 0.0)), 2)
        pytorch_reserved_mib = round(max(0.0, combined_physical_mib - vmm_physical_pool_mib), 2)
    else:
        pytorch_reserved_mib = round(float(run_result.get("peak_gpu_reserved_mib", 0.0)), 2)
        combined_physical_mib = pytorch_reserved_mib

    overflow_ok = (not policy.use_tripool_vmm) or int(run_result.get("vmm_overflow_creates", 0)) == 0
    return {
        "quota_limit_mib": quota_limit_mib,
        "pytorch_peak_reserved_mib": pytorch_reserved_mib,
        "vmm_physical_pool_mib": vmm_physical_pool_mib,
        "combined_physical_mib": combined_physical_mib,
        "within_quota": bool(combined_physical_mib <= quota_limit_mib and overflow_ok),
    }


def run_internvlu_generality_validation(
    model_dir: Path,
    vendor_dir: Path,
    output_dir: Path,
    num_gpus: int = 8,
    num_sessions: int = 48,
    offered_qps: float = 1.6,
    constrained_gib: float = 12.0,
    total_gpu_gib: float | None = None,
    smoke_1gpu: bool = False,
) -> dict[str, Any]:
    """Drive `InternVL-U` (`bridge_mllm_dit`) across Regime A (40GB burst) & Regime B (strict `<= 12 GiB` quota)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_base = _local_tmp_base()
    eff_total_gib = _detect_gpu_total_gib(40.0) if total_gpu_gib is None else float(total_gpu_gib)
    if smoke_1gpu:
        num_gpus = 1
        num_sessions = min(int(num_sessions), 4)

    suite_a = build_internvlu_suite(num_sessions=num_sessions, offered_qps=offered_qps, highres_size=1024)
    parity_slice = WorkloadSuite(
        name="internvlu_parity_4",
        trace_source="deterministic",
        base_qps=0.5,
        sessions=suite_a.sessions[:4],
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
            tmp_root=tmp_base / f"ms_ivlu_a_{pol_name}",
        )
        try:
            eng.start()
            if smoke_1gpu:
                p_dir = tmp_base / f"ivlu_parity_{pol_name}"
                p_rep = eng.run_suite(parity_slice, target_qps=0.5, save_outputs_dir=p_dir)
                d = p_rep.to_dict()
                d.pop("sessions", None)
                d["regime"] = "smoke_1gpu_parity"
                regime_a_results.append(d)
                parity_outcomes[pol_name] = p_rep.sessions
            else:
                rep = eng.run_suite(suite_a, target_qps=offered_qps)
                d = rep.to_dict()
                d.pop("sessions", None)
                d["regime"] = "regime_a_40gb_burst"
                regime_a_results.append(d)

                if pol_name in ("morphserve", "monolithic_dp"):
                    p_dir = tmp_base / f"ivlu_parity_{pol_name}"
                    p_rep = eng.run_suite(parity_slice, target_qps=0.5, save_outputs_dir=p_dir)
                    parity_outcomes[pol_name] = p_rep.sessions
        finally:
            eng.shutdown()
            gc.collect()

    # Regime B: Strict `12 GiB` Per-GPU Quota (`PyTorch reserved + VMM physical pool <= 12 GiB`)
    # Uses `highres_size=1024` (`D4.2`) so `monolithic_dp` (`8.3 GiB` weights + unscoped `1024^2` VAE) hits the
    # physical 12 GiB wall while `morphserve` (`TriPool-VMM` unpinned decoder + scoped VAE) survives cleanly.
    regime_b_qps = 0.5 if smoke_1gpu else round(offered_qps * 0.75, 2)
    b_sessions = 2 if smoke_1gpu else num_sessions
    suite_b = build_internvlu_suite(num_sessions=b_sessions, offered_qps=regime_b_qps, highres_size=1024)
    regime_b_policies = (
        ("morphserve", "monolithic_dp")
        if smoke_1gpu
        else ("morphserve", "monolithic_dp", "naive_host_offload")
    )
    regime_b_results: list[dict[str, Any]] = []
    morphserve_quota_ok = False

    for pol_name in regime_b_policies:
        overrides, budget_plan = compute_internvlu_memory_quota_overrides(
            pol_name,
            num_gpus=num_gpus,
            vendor_dir=vendor_dir,
            constrained_gib=constrained_gib,
            total_gpu_gib=eff_total_gib,
        )
        if smoke_1gpu:
            overrides["inter_turn_think_s_range"] = (0.0, 0.0)
        cfg = EnginePolicyConfig.from_preset(pol_name, num_gpus=num_gpus, **overrides)
        eng = MorphServeClusterEngine(
            policy=cfg,
            model_dir=model_dir,
            bench_root=output_dir,
            tmp_root=tmp_base / f"ms_ivlu_b_{pol_name}",
        )
        try:
            eng.start()
            rep = eng.run_suite(suite_b, target_qps=regime_b_qps)
            w_stats = asyncio.run(
                asyncio.gather(*(eng._rpc(g, {"op": "GET_STATS"}) for g in range(num_gpus)))
            )
            d = rep.to_dict()
            d.pop("sessions", None)
            if w_stats and any("pytorch_reserved_mib" in st for st in w_stats):
                d["pytorch_peak_reserved_mib"] = round(
                    max(float(st.get("pytorch_reserved_mib", 0.0)) for st in w_stats), 2
                )
                d["vmm_physical_pool_mib"] = round(
                    max(float(st.get("vmm_physical_pool_mib", 0.0)) for st in w_stats), 2
                )
            d["regime"] = f"regime_b_{int(constrained_gib)}gib_capped"
            d["quota_budget_plan"] = budget_plan
            audit = verify_internvlu_memory_quota(d, cfg, constrained_gib=constrained_gib)
            d["memory_quota_audit"] = audit
            if pol_name == "morphserve":
                morphserve_quota_ok = bool(audit["within_quota"] and int(d.get("oom_sessions", 0)) == 0)
            regime_b_results.append(d)
        finally:
            eng.shutdown()
            gc.collect()

    import shutil

    evaluator = ExactnessEvaluator()
    parity = evaluator.evaluate_parity_and_quality(
        parity_slice.sessions,
        parity_outcomes.get("monolithic_dp", []),
        parity_outcomes.get("morphserve", []),
    )
    for pol_clean in ("morphserve", "monolithic_dp"):
        shutil.rmtree(tmp_base / f"ivlu_parity_{pol_clean}", ignore_errors=True)

    report = {
        "schema": "morphserve.osdi27.m24_internvlu.v3",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_gpus": num_gpus,
        "constrained_quota_gib": constrained_gib,
        "morphserve_12gib_quota_verified": morphserve_quota_ok,
        "regime_a_40gb_burst": regime_a_results,
        "regime_b_memory_capped": regime_b_results,
        "lossless_parity": parity,
        "status": "ok" if (parity["lossless_pass"] and morphserve_quota_ok) else "check_failed",
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
    parser.add_argument("--constrained-gib", type=float, default=12.0)
    parser.add_argument("--total-gpu-gib", type=float, default=None)
    parser.add_argument("--smoke-1gpu", action="store_true")
    args = parser.parse_args(argv)
    rep = run_internvlu_generality_validation(
        model_dir=args.model_dir,
        vendor_dir=args.vendor_dir,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        num_sessions=args.num_sessions,
        offered_qps=args.qps,
        constrained_gib=args.constrained_gib,
        total_gpu_gib=args.total_gpu_gib,
        smoke_1gpu=args.smoke_1gpu,
    )
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
