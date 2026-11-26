"""Phase-2 Milestone M2.4: Unified `MorphServeClusterEngine` Dual-Regime Multi-Turn `InternVL-U` (`bridge_mllm_dit`)."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from morphserve.bench.exactness import ExactnessEvaluator
from morphserve.bench.interleaved_bench import (
    InterleavedSession,
    InterleavedTurn,
    POST_DIT_INSPECT_PROMPT,
    REFINEMENT_TEMPLATES,
    WorkloadSuite,
)
from morphserve.engine import EnginePolicyConfig, MorphServeClusterEngine


def _load_optional_bench_prompts(bench_root: Path | None) -> tuple[list[dict[str, str]], list[str]]:
    """Load real MM-Vet VQA items and GenEval generation prompts when `bench_root` is available."""
    mmvet_items: list[dict[str, str]] = []
    geneval_prompts: list[str] = []
    if bench_root is None or not Path(bench_root).is_dir():
        return mmvet_items, geneval_prompts

    root = Path(bench_root)
    mmvet_json = root / "mmvet_questions.json"
    if mmvet_json.is_file():
        try:
            for item in json.loads(mmvet_json.read_text(encoding="utf-8")):
                img_rel = str(item.get("image_path", ""))
                if img_rel and (root / img_rel).is_file():
                    mmvet_items.append(
                        {
                            "image_path": img_rel,
                            "question": str(item.get("question", "")).strip(),
                            "answer": str(item.get("answer", "")).strip(),
                        }
                    )
        except Exception:
            mmvet_items.clear()

    acc_json = root / "suites/accuracy_geneval_mmvet.json"
    if acc_json.is_file():
        try:
            acc = WorkloadSuite.load_json(acc_json)
            for s in acc.sessions:
                if s.source_dataset == "geneval" and s.turns:
                    p = str(s.turns[0].prompt).strip()
                    if p:
                        geneval_prompts.append(p)
        except Exception:
            geneval_prompts.clear()

    if not geneval_prompts:
        for cand in (
            root / "sources/geneval_evaluation_metadata.jsonl",
            root.parent / "bench_sources/geneval_evaluation_metadata.jsonl",
        ):
            if cand.is_file():
                try:
                    for line in cand.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        p = str(obj.get("prompt", "")).strip()
                        if p:
                            geneval_prompts.append(p)
                    if geneval_prompts:
                        break
                except Exception:
                    geneval_prompts.clear()

    return mmvet_items, geneval_prompts


def build_internvlu_suite(
    num_sessions: int = 48,
    offered_qps: float = 1.6,
    *,
    highres_size: int = 1024,
    bench_root: Path | None = None,
) -> WorkloadSuite:
    """Construct genuine 2-4 turn multi-turn interleaved `InternVL-U` (`bridge_mllm_dit`) suite.

    Mix across every 10 sessions:
      - 40%: 3-turn `ar_only -> dit_t2i -> ar_only(use_prev_image=True)` (`interleaved_co_creation`)
      - 30%: 4-turn `dit_t2i -> ar_only(use_prev_image=True) -> dit_edit(use_prev_image=True) -> ar_only(use_prev_image=True)` (`multiturn_edit`)
      - 30%: 2-turn `ar_only -> ar_only` (`multi_image_vqa`)
    """
    fallback_img_prompts = (
        "A cinematic minimalist Japanese tea room overlooking a snowy bamboo garden at dusk.",
        "Macro photograph of a mechanical watch movement with sapphire gears on dark slate.",
        "Watercolor architectural sketch of a coastal lighthouse under dramatic cumulus clouds.",
        "Studio product shot of a ceramic pour-over coffee dripper beside roasted beans.",
        "A serene alpine cabin beside an emerald glacial lake under soft morning light.",
        "A vibrant glass conservatory filled with botanical ferns and warm sunlit mist.",
    )
    fallback_vqa_pairs = (
        (
            "Summarize in two sentences why compact bridge conditioning lowers cross-GPU state transfer cost.",
            "Now compare how multi-turn visual feedback changes GPU memory residency compared to single-turn generation.",
        ),
        (
            "List three differences between autoregressive causal decoding and bidirectional diffusion steps.",
            "Explain in two concise sentences how layer-chunked preemption protects interactive AR token latency.",
        ),
    )

    mmvet_items, geneval_prompts = _load_optional_bench_prompts(bench_root)
    gen_pool = geneval_prompts if geneval_prompts else list(fallback_img_prompts)
    has_real_mmvet = len(mmvet_items) >= 2
    src_tag = "interleaved_bench_v2" if (has_real_mmvet or geneval_prompts) else "internvlu_multiturn_fallback"

    sessions: list[InterleavedSession] = []
    t_off = 0.0
    dt_base = 1.0 / max(0.1, float(offered_qps))

    for i in range(int(num_sessions)):
        if i > 0:
            # Deterministic bursty arrivals (clusters of 4 tight requests followed by pause)
            t_off += dt_base * (0.25 if (i % 4 != 0) else 2.95)

        slot = i % 10
        res = int(highres_size) if (i % 4 == 1) else 512
        gen_prompt = gen_pool[i % len(gen_pool)]
        refine_prompt = REFINEMENT_TEMPLATES[i % len(REFINEMENT_TEMPLATES)]

        if slot in (0, 2, 5, 8):
            # 40%: 3-turn `ar_only -> dit_t2i -> ar_only(use_prev_image=True)`
            sessions.append(
                InterleavedSession(
                    session_id=f"ivlu_cocreate_{i:03d}",
                    quadrant="interleaved_co_creation",
                    source_dataset="geneval" if geneval_prompts else src_tag,
                    arrival_offset_s=round(t_off, 3),
                    turns=[
                        InterleavedTurn(
                            turn_idx=0,
                            modality_plan="ar_only",
                            prompt=(
                                "Plan the key visual subjects, framing, and lighting in two concise sentences "
                                f"before generating: {gen_prompt}"
                            ),
                            image_paths=[],
                            max_ar_tokens=48,
                            use_prev_image=False,
                        ),
                        InterleavedTurn(
                            turn_idx=1,
                            modality_plan="dit_t2i",
                            prompt=gen_prompt,
                            image_paths=[],
                            height=res,
                            width=res,
                            dit_steps=16,
                            use_prev_image=False,
                        ),
                        InterleavedTurn(
                            turn_idx=2,
                            modality_plan="ar_only",
                            prompt=POST_DIT_INSPECT_PROMPT,
                            image_paths=[],
                            max_ar_tokens=48,
                            use_prev_image=True,
                        ),
                    ],
                )
            )
        elif slot in (1, 4, 7):
            # 30%: 4-turn `dit_t2i -> ar_only(use_prev_image=True) -> dit_edit(use_prev_image=True) -> ar_only(use_prev_image=True)`
            sessions.append(
                InterleavedSession(
                    session_id=f"ivlu_edit4_{i:03d}",
                    quadrant="multiturn_edit",
                    source_dataset="geneval" if geneval_prompts else src_tag,
                    arrival_offset_s=round(t_off, 3),
                    turns=[
                        InterleavedTurn(
                            turn_idx=0,
                            modality_plan="dit_t2i",
                            prompt=gen_prompt,
                            image_paths=[],
                            height=res,
                            width=res,
                            dit_steps=16,
                            use_prev_image=False,
                        ),
                        InterleavedTurn(
                            turn_idx=1,
                            modality_plan="ar_only",
                            prompt=POST_DIT_INSPECT_PROMPT,
                            image_paths=[],
                            max_ar_tokens=48,
                            use_prev_image=True,
                        ),
                        InterleavedTurn(
                            turn_idx=2,
                            modality_plan="dit_edit",
                            prompt=refine_prompt,
                            image_paths=[],
                            height=res,
                            width=res,
                            dit_steps=16,
                            use_prev_image=True,
                        ),
                        InterleavedTurn(
                            turn_idx=3,
                            modality_plan="ar_only",
                            prompt=(
                                "Verify whether the requested visual refinement was faithfully applied "
                                "and describe the updated scene concisely."
                            ),
                            image_paths=[],
                            max_ar_tokens=48,
                            use_prev_image=True,
                        ),
                    ],
                )
            )
        else:
            # 30%: 2-turn `ar_only -> ar_only`
            if has_real_mmvet:
                a = mmvet_items[(i * 2) % len(mmvet_items)]
                b = mmvet_items[(i * 2 + 1) % len(mmvet_items)]
                vqa_turns = [
                    InterleavedTurn(
                        turn_idx=0,
                        modality_plan="ar_only",
                        prompt=a["question"],
                        image_paths=[a["image_path"]],
                        max_ar_tokens=64,
                        expected_answer=a["answer"],
                        use_prev_image=False,
                    ),
                    InterleavedTurn(
                        turn_idx=1,
                        modality_plan="ar_only",
                        prompt=f"Now inspect the second image as well: {b['question']}",
                        image_paths=[b["image_path"]],
                        max_ar_tokens=64,
                        expected_answer=b["answer"],
                        use_prev_image=False,
                    ),
                ]
                vqa_ds = "mmvet"
            else:
                q0, q1 = fallback_vqa_pairs[i % len(fallback_vqa_pairs)]
                vqa_turns = [
                    InterleavedTurn(
                        turn_idx=0,
                        modality_plan="ar_only",
                        prompt=q0,
                        image_paths=[],
                        max_ar_tokens=48,
                        use_prev_image=False,
                    ),
                    InterleavedTurn(
                        turn_idx=1,
                        modality_plan="ar_only",
                        prompt=q1,
                        image_paths=[],
                        max_ar_tokens=48,
                        use_prev_image=False,
                    ),
                ]
                vqa_ds = src_tag
            sessions.append(
                InterleavedSession(
                    session_id=f"ivlu_vqa2_{i:03d}",
                    quadrant="multi_image_vqa",
                    source_dataset=vqa_ds,
                    arrival_offset_s=round(t_off, 3),
                    turns=vqa_turns,
                )
            )

    span_s = round(max((float(s.arrival_offset_s) for s in sessions), default=0.0), 3)
    return WorkloadSuite(
        name=f"internvlu_multiturn_n{num_sessions}_h{highres_size}",
        trace_source="burst_clustered_multiturn_v2",
        base_qps=offered_qps,
        sessions=sessions,
        schema="morphserve.interleaved_bench.v2",
        warmup_s=round(min(20.0, span_s * 0.2), 3),
        steady_s=round(max(0.0, span_s - min(20.0, span_s * 0.2)), 3),
        arrival_duration_s=span_s,
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
    """Compute honest per-GPU `12 GiB` quota overrides via C++17 `WorkerBudgetPlanner`."""
    from morphserve._core import call_json, get_core_lib

    eff_total_gib = _detect_gpu_total_gib(40.0) if total_gpu_gib is None else float(total_gpu_gib)
    L = get_core_lib()
    out = call_json(
        L,
        L.ms_budget_internvlu_quota_overrides_json,
        str(policy_name).encode("utf-8"),
        int(num_gpus),
        str(vendor_dir).encode("utf-8"),
        float(constrained_gib),
        float(eff_total_gib),
    )
    ov = dict(out["overrides"])
    tr = ov.get("inter_turn_think_s_range")
    if isinstance(tr, list) and len(tr) == 2:
        ov["inter_turn_think_s_range"] = (float(tr[0]), float(tr[1]))
    if "initial_alphas" in ov and isinstance(ov["initial_alphas"], dict):
        ov["initial_alphas"] = {int(k): float(v) for k, v in ov["initial_alphas"].items()}
    return ov, dict(out.get("budget") or out["budget_plan"])


def verify_internvlu_memory_quota(
    run_result: dict[str, Any],
    policy: EnginePolicyConfig,
    *,
    constrained_gib: float = 12.0,
) -> dict[str, Any]:
    """Audit observed `PyTorch peak_reserved_mib + TriPool-VMM physical_pool_mib <= quota_limit_mib` via C++."""
    from morphserve._core import call_json, encode_json, get_core_lib
    from morphserve.engine.multinode_fleet import serialize_policy_config

    L = get_core_lib()
    return call_json(
        L,
        L.ms_budget_verify_internvlu_quota_json,
        encode_json(run_result),
        encode_json(serialize_policy_config(policy)),
        float(constrained_gib),
    )


def _think_time_overrides(smoke_1gpu: bool) -> dict[str, Any]:
    if smoke_1gpu:
        return {
            "inter_turn_think_s_range": (0.0, 0.0),
            "think_time_dist": "constant",
            "think_time_median_s": 0.0,
            "think_time_sigma": 0.0,
        }
    return {
        "inter_turn_think_s_range": (1.0, 8.0),
        "think_time_dist": "lognormal",
        "think_time_median_s": 3.0,
        "think_time_sigma": 0.75,
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
    bench_root: Path | None = None,
) -> dict[str, Any]:
    """Drive `InternVL-U` (`bridge_mllm_dit`) across Regime A (40GB burst) & Regime B (strict `<= 12 GiB` quota)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_base = _local_tmp_base()
    eff_total_gib = _detect_gpu_total_gib(40.0) if total_gpu_gib is None else float(total_gpu_gib)
    eff_bench_root = Path(bench_root) if (bench_root is not None and Path(bench_root).is_dir()) else output_dir

    if smoke_1gpu:
        num_gpus = 1
        num_sessions = min(int(num_sessions), 4)

    suite_a = build_internvlu_suite(
        num_sessions=num_sessions,
        offered_qps=offered_qps,
        highres_size=1024,
        bench_root=bench_root,
    )
    # First 4 sessions span all 3 multi-turn templates (3-turn co-creation, 4-turn edit+inspect, 2-turn VQA)
    parity_slice = WorkloadSuite(
        name="internvlu_multiturn_parity_4",
        trace_source="deterministic",
        base_qps=0.5,
        sessions=suite_a.sessions[:4],
        schema="morphserve.interleaved_bench.v2",
        warmup_s=0.0,
        steady_s=0.0,
        arrival_duration_s=0.0,
    )

    think_ov = _think_time_overrides(smoke_1gpu)
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
            steady_window_mode="session_trim" if smoke_1gpu else "time",
            warmup_window_s=suite_a.warmup_s,
            steady_window_s=suite_a.steady_s,
            **think_ov,
        )
        eng = MorphServeClusterEngine(
            policy=cfg,
            model_dir=model_dir,
            bench_root=eff_bench_root,
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
                    saved_policy = eng.policy
                    eng.policy = replace(
                        saved_policy,
                        inter_turn_think_s_range=(0.0, 0.0),
                        think_time_dist="constant",
                        think_time_median_s=0.0,
                        think_time_sigma=0.0,
                        steady_window_mode="session_trim",
                        steady_state_trim_frac=0.0,
                    )
                    try:
                        p_rep = eng.run_suite(parity_slice, target_qps=0.5, save_outputs_dir=p_dir)
                    finally:
                        eng.policy = saved_policy
                    parity_outcomes[pol_name] = p_rep.sessions
        finally:
            eng.shutdown()
            gc.collect()

    # Regime B: Strict `12 GiB` Per-GPU Quota (`PyTorch reserved + VMM physical pool <= 12 GiB`)
    regime_b_qps = 0.5 if smoke_1gpu else round(offered_qps * 0.75, 2)
    b_sessions = 2 if smoke_1gpu else num_sessions
    suite_b = build_internvlu_suite(
        num_sessions=b_sessions,
        offered_qps=regime_b_qps,
        highres_size=1024,
        bench_root=bench_root,
    )
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
        overrides.update(think_ov)
        overrides["steady_window_mode"] = "session_trim" if smoke_1gpu else "time"
        overrides["warmup_window_s"] = suite_b.warmup_s
        overrides["steady_window_s"] = suite_b.steady_s

        cfg = EnginePolicyConfig.from_preset(pol_name, num_gpus=num_gpus, **overrides)
        eng = MorphServeClusterEngine(
            policy=cfg,
            model_dir=model_dir,
            bench_root=eff_bench_root,
            tmp_root=tmp_base / f"ms_ivlu_b_{pol_name}",
        )
        try:
            eng.start()
            rep = eng.run_suite(suite_b, target_qps=regime_b_qps)

            async def _fetch_w_stats() -> list[dict[str, Any]]:
                return list(await asyncio.gather(*(eng._rpc(g, {"op": "GET_STATS"}) for g in range(num_gpus))))

            w_stats = asyncio.run(_fetch_w_stats())
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

    evaluator = ExactnessEvaluator(bench_root=eff_bench_root)
    parity = evaluator.evaluate_parity_and_quality(
        parity_slice.sessions,
        parity_outcomes.get("monolithic_dp", []),
        parity_outcomes.get("morphserve", []),
        bench_root=eff_bench_root,
    )
    parity["parity_sessions_spec"] = [
        {
            "session_id": s.session_id,
            "quadrant": s.quadrant,
            "num_turns": len(s.turns),
            "turn_plans": [t.modality_plan for t in s.turns],
        }
        for s in parity_slice.sessions
    ]
    for pol_clean in ("morphserve", "monolithic_dp"):
        shutil.rmtree(tmp_base / f"ivlu_parity_{pol_clean}", ignore_errors=True)

    report = {
        "schema": "morphserve.osdi27.m24_internvlu.v4",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_gpus": num_gpus,
        "constrained_quota_gib": constrained_gib,
        "think_time_config": think_ov,
        "bench_root_used": str(bench_root) if bench_root is not None else None,
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
    parser = argparse.ArgumentParser(description="MorphServe Unified InternVL-U Dual-Regime Multi-Turn Benchmark")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, default=None)
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
        bench_root=args.bench_root,
    )
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
