"""Seam tests for `ClusterSimulator`, `Interleaved-Bench`, `ExactnessEvaluator`, `BenchmarkRunner`, and bench CLIs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import socket
import struct
import threading
from typing import Any

import numpy as np
from PIL import Image
import pytest

from morphserve.bench.exactness import (
    ClipEditingScorer,
    ExactnessEvaluator,
    OwlV2GenEvalScorer,
    compute_ssim_rgb,
    image_pair_metrics,
    pil_sha256,
    score_mmvet_answer,
)
from morphserve.bench.harness import (
    BenchmarkRunner,
    SuiteRunSpec,
    compact_report_dict,
    compute_phase_breakdown,
    compute_quadrant_breakdown,
    compute_sustainable_qps_summary,
    effective_steady_window_s,
    interpolate_slo_threshold_qps,
    load_main_mix_suite,
    local_tmp_base,
    parse_csv_floats,
    parse_csv_strings,
    resolve_default_main_suite_path,
    resolve_sources_dir,
)
from morphserve.bench.interleaved_bench import (
    InterleavedSession,
    InterleavedTurn,
    WorkloadSuite,
    build_m3_active_accuracy_suite,
    build_online_phase_shift_suite,
    compose_weighted_suite,
    load_burstgpt_interarrivals,
    load_gentd26_interarrivals,
)
from morphserve.bench import (
    run_phase2_bagel_main,
    run_phase2_overnight_deep,
    validate_phase2_engine_smoke,
    validate_phase2_internvl_u,
    validate_phase2_multinode_2x8,
)
from morphserve.bench.run_phase2_bagel_main import scale_suite_sessions, stratify_suite
from morphserve.bench.validate_phase2_internvl_u import (
    build_internvlu_suite,
    compute_internvlu_memory_quota_overrides,
    verify_internvlu_memory_quota,
)
from morphserve.bench.validate_phase2_multinode_2x8 import (
    _recv_exact,
    _recv_json_on_master,
    evaluate_cross_node_routing_check,
)
from morphserve.engine import InProcessWorkerFleet
from morphserve.engine.config import (
    EnginePolicyConfig,
    RunWindowOptions,
    build_ablation_configs,
    build_static_split_configs,
)
from morphserve.engine.workload import assign_dual_trace_arrivals, build_time_window_suite, hamilton_apportion
from morphserve.runtime.peer_transport import find_free_tcp_port
from morphserve.sim.cluster_simulator import (
    HardwareInterconnectSpec,
    MeasuredPhysicsProfile,
    SimGpuMemoryLedger,
    WorkloadScaleSpec,
    evaluate_against_real_runs,
    simulate_cluster_point,
)


def _make_structured_rgb(height: int = 64, width: int = 64, seed: int = 17) -> np.ndarray:
    """Create a deterministic, non-trivial RGB uint8 image with spatial gradients and texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(20, 210, height), np.linspace(30, 200, width), indexing="ij")
    r = np.clip(yy + rng.integers(-12, 13, size=(height, width)), 0, 255)
    g = np.clip(xx + rng.integers(-12, 13, size=(height, width)), 0, 255)
    b = np.clip(0.5 * (yy + xx) + rng.integers(-12, 13, size=(height, width)), 0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


# ==================== 1. ClusterSimulator Public Seam ====================


def test_simulator_custom_sessions_public_seam_and_canonical_report_fields() -> None:
    physics = MeasuredPhysicsProfile()
    hw_a100 = HardwareInterconnectSpec.presets(physics)[0]
    custom_sessions = [
        (0.0, "interleaved_co_creation", (("ar_only", 512), ("dit_t2i", 512), ("ar_only", 512), ("dit_edit", 512))),
        (350.0, "multiturn_edit", (("dit_edit", 512), ("dit_edit", 512), ("dit_edit", 512))),
        (700.0, "multi_image_vqa", (("ar_only", 512), ("ar_only", 512))),
        (1050.0, "visual_cot_highres", (("cot_then_dit", 1024),)),
        (1400.0, "multiturn_edit", (("dit_edit", 512), ("dit_edit", 512))),
        (1750.0, "multi_image_vqa", (("ar_only", 512), ("ar_only", 512))),
    ]

    rep = simulate_cluster_point(
        "morphserve",
        num_gpus=8,
        hw=hw_a100,
        offered_qps=0.5,
        num_sessions=len(custom_sessions),
        seed=2026,
        physics=physics,
        workload_scale=WorkloadScaleSpec.bagel_7b_standard(),
        custom_sessions=custom_sessions,
    )
    assert rep["policy_name"] == "morphserve"
    assert rep["total_sessions"] == len(custom_sessions)
    assert rep["oom_sessions"] == 0
    assert rep["joint_slo_goodput_rps"] == rep["steady_joint_slo_goodput_rps"]
    assert rep["joint_slo_attainment_pct"] == rep["steady_joint_slo_attainment_pct"]
    assert "full_trace_goodput_rps" in rep
    assert "full_trace_joint_slo_attainment_pct" in rep
    assert "vmm_physical_pool_mib" in rep
    assert "kv_host_swap_ins" in rep


def test_sim_gpu_memory_ledger_physical_conservation_and_swap() -> None:
    physics = MeasuredPhysicsProfile()
    pol_ms = EnginePolicyConfig.from_preset("morphserve", num_gpus=8)
    ledger = SimGpuMemoryLedger(
        gpu_id=0,
        total_hbm_mib=physics.driver_total_hbm_40g_mib,
        policy=pol_ms,
        role="hybrid",
        resident_expert_layers=14,
        physics=physics,
    )
    assert ledger.free_kv_bytes() > 0
    for sid in range(1, 6):
        assert ledger.upsert_session_kv(sid, 1200.0) is True
    dit_ok, _ = ledger.execute_dit_and_vae(height=1024, active_sid=5)
    assert dit_ok is True
    assert ledger.peak_used_mib <= physics.driver_total_hbm_40g_mib + 1e-3
    # Bringing back an evicted session must account for swap-ins cleanly when swapped.
    ledger.ensure_session_resident(1)
    assert ledger.kv_host_swap_outs >= ledger.kv_host_swap_ins
    for sid in range(1, 6):
        ledger.release_session(sid)
    assert ledger.active_kv_mib == 0.0

    # `monolithic_kv_swap` must allow idle-KV eviction while `monolithic_dp` does not.
    pol_mono = EnginePolicyConfig.from_preset("monolithic_dp", num_gpus=8)
    pol_mono_swap = EnginePolicyConfig.from_preset("monolithic_kv_swap", num_gpus=8)
    with SimGpuMemoryLedger(
        gpu_id=0,
        total_hbm_mib=physics.driver_total_hbm_40g_mib,
        policy=pol_mono,
        role="gen",
        resident_expert_layers=28,
        physics=physics,
    ) as led_mono, SimGpuMemoryLedger(
        gpu_id=0,
        total_hbm_mib=physics.driver_total_hbm_40g_mib,
        policy=pol_mono_swap,
        role="gen",
        resident_expert_layers=28,
        physics=physics,
    ) as led_swap:
        assert led_mono.upsert_session_kv(1, 1200.0) is True
        assert led_mono.upsert_session_kv(2, 1200.0) is True
        assert led_mono.evict_idle_kv(1000.0, exclude_sid=2) == 0.0
        assert led_swap.upsert_session_kv(1, 1200.0) is True
        assert led_swap.upsert_session_kv(2, 1200.0) is True
        assert led_swap.evict_idle_kv(1000.0, exclude_sid=2) >= 1200.0
        assert led_swap.kv_host_swap_outs >= 1

    # Strict calibration gate: 0 real runs MUST fail both 8% OOS and 5% gates.
    empty_val = evaluate_against_real_runs([], physics=physics)
    assert empty_val.points_checked == 0
    assert empty_val.within_8pct_oos_gate is False
    assert empty_val.within_5pct_gate is False

    # Simulator supports `monolithic_kv_swap` end-to-end.
    hw_a100 = HardwareInterconnectSpec.presets(physics)[0]
    swap_sim = simulate_cluster_point(
        "monolithic_kv_swap",
        num_gpus=8,
        hw=hw_a100,
        offered_qps=0.4,
        num_sessions=8,
        seed=2026,
        physics=physics,
        workload_scale=WorkloadScaleSpec.bagel_7b_standard(),
        custom_sessions=[
            (0.0, "interleaved_co_creation", (("ar_only", 512), ("dit_t2i", 512))),
            (400.0, "multiturn_edit", (("dit_edit", 512), ("dit_edit", 512))),
        ],
    )
    assert swap_sim["policy_name"] == "monolithic_kv_swap"
    assert swap_sim["total_sessions"] == 2


def test_cross_node_routing_check_suppresses_large_tight_mot_ship() -> None:
    chk = evaluate_cross_node_routing_check(
        world_size=16,
        gpus_per_node=8,
        intra_bw=10.5,
        socket_bw=9.5,
        inter_bw=1.1,
    )
    assert chk["routing_pass"] is True
    assert chk["bridge_24mib_decision"]["path"] == "state_ship"
    assert chk["tight_mot_1536mib_decision"]["path"] == "in_place_morph"
    assert chk["suppressed_cross_node_1gib_kv_ship"] is True


# ==================== 2. Workload, Apportionment, Suite Builders & Policies ====================


def test_hamilton_apportion_and_suite_stratification_scaling() -> None:
    alloc = hamilton_apportion(10, {"a": 0.34, "b": 0.33, "c": 0.33})
    assert sum(alloc.values()) == 10
    assert alloc["a"] == 4 and alloc["b"] == 3 and alloc["c"] == 3

    quads = (
        ("interleaved_co_creation",) * 12
        + ("multiturn_edit",) * 12
        + ("multi_image_vqa",) * 10
        + ("visual_cot_highres",) * 6
    )
    sessions = [
        InterleavedSession(
            session_id=f"s_{i:02d}",
            quadrant=q,
            source_dataset="unit",
            arrival_offset_s=float(i) * 0.5,
            turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="hello")],
        )
        for i, q in enumerate(quads)
    ]
    suite = WorkloadSuite(name="full40", trace_source="burstgpt+gentd26", base_qps=1.0, sessions=sessions)

    sliced = stratify_suite(suite, max_sessions=20)
    assert len(sliced.sessions) == 20
    by_q: dict[str, int] = {}
    for s in sliced.sessions:
        by_q[s.quadrant] = by_q.get(s.quadrant, 0) + 1
    assert by_q == {
        "interleaved_co_creation": 6,
        "multiturn_edit": 6,
        "multi_image_vqa": 5,
        "visual_cot_highres": 3,
    }

    expanded = scale_suite_sessions(suite, target_sessions=96)
    assert len(expanded.sessions) == 96
    offsets = [s.arrival_offset_s for s in expanded.sessions]
    assert offsets == sorted(offsets)


def test_trace_parsers_phase_shift_time_window_and_accuracy_suite(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    burst_csv = sources / "burstgpt_1_sample.csv"
    burst_csv.write_text("Timestamp,Model\n0.0,a\n0.5,a\n2.0,a\n2.5,a\n", encoding="utf-8")
    gentd_csv = sources / "gentd26_lora_request_trace.csv"
    gentd_csv.write_text(
        "gmt_create,predict_status\n"
        "2026-01-01 00:00:00,SUCCEED\n"
        "2026-01-01 00:00:01,SUCCEED\n"
        "2026-01-01 00:00:04,SUCCEED\n"
        "2026-01-01 00:00:05,FAILED\n"
        "2026-01-01 00:00:06,SUCCEED\n",
        encoding="utf-8",
    )
    b_deltas = load_burstgpt_interarrivals(burst_csv)
    g_deltas = load_gentd26_interarrivals(gentd_csv)
    assert len(b_deltas) == 3 and abs(sum(b_deltas) / len(b_deltas) - 1.0) < 1e-6
    assert len(g_deltas) == 3 and abs(sum(g_deltas) / len(g_deltas) - 1.0) < 1e-6

    # Build minimal synthetic pools and verify compose, time-window, and online phase-shift suites.
    pools: dict[str, list[InterleavedSession]] = {}
    for quad, plan in (
        ("interleaved_co_creation", "ar_only"),
        ("multiturn_edit", "dit_edit"),
        ("multi_image_vqa", "ar_only"),
        ("visual_cot_highres", "cot_then_dit"),
    ):
        pools[quad] = [
            InterleavedSession(
                session_id=f"{quad}_{i}",
                quadrant=quad,
                source_dataset="unit",
                turns=[InterleavedTurn(turn_idx=0, modality_plan=plan, prompt="test prompt")],
            )
            for i in range(6)
        ]

    weighted = compose_weighted_suite(
        "mini_main",
        pools,
        {
            "interleaved_co_creation": 6,
            "multiturn_edit": 5,
            "multi_image_vqa": 5,
            "visual_cot_highres": 4,
        },
        b_deltas,
        g_deltas,
        target_qps=0.5,
    )
    suite_file = tmp_path / "suites" / "main_mix_v2_80.json"
    weighted.save_json(suite_file)
    reloaded = load_main_mix_suite(tmp_path)
    assert len(reloaded.sessions) == 20
    assert resolve_default_main_suite_path(tmp_path) == suite_file

    tw = build_time_window_suite(
        reloaded,
        target_qps=0.5,
        burstgpt_deltas=b_deltas,
        gentd26_deltas=g_deltas,
        arrival_duration_s=60.0,
        min_sessions=20,
        warmup_s=15.0,
        steady_s=45.0,
        cooldown_s=0.0,
    )
    assert len(tw.sessions) >= 20
    assert effective_steady_window_s(tw) >= 45.0

    # Verify cooldown exclusion and <2% actual-vs-target QPS normalization
    tw_cd = build_time_window_suite(
        reloaded,
        target_qps=0.5,
        burstgpt_deltas=b_deltas,
        gentd26_deltas=g_deltas,
        arrival_duration_s=240.0,
        min_sessions=40,
        warmup_s=60.0,
        steady_s=150.0,
        cooldown_s=30.0,
    )
    assert abs(effective_steady_window_s(tw_cd) - 150.0) < 1e-3
    span_cd = max(s.arrival_offset_s for s in tw_cd.sessions)
    actual_qps = len(tw_cd.sessions) / max(1e-6, span_cd)
    assert abs(actual_qps - 0.5) / 0.5 < 0.02

    # Verify `dit_t2i` under `visual_cot_highres` routes to GenTD26 (distinct from BurstGPT AR arrivals)
    mixed_routing = [
        InterleavedSession(
            session_id=f"ar_{i}",
            quadrant="multi_image_vqa",
            source_dataset="unit",
            turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="q")],
        )
        for i in range(4)
    ] + [
        InterleavedSession(
            session_id=f"t2i_{i}",
            quadrant="visual_cot_highres",
            source_dataset="unit",
            turns=[InterleavedTurn(turn_idx=0, modality_plan="dit_t2i", prompt="img")],
        )
        for i in range(4)
    ]
    assign_dual_trace_arrivals(
        mixed_routing,
        burstgpt_deltas=[0.05, 0.05, 2.90],
        gentd26_deltas=[1.0, 1.0, 1.0],
        target_qps=0.5,
        target_span_s=16.0,
    )
    t2i_offs = sorted(s.arrival_offset_s for s in mixed_routing if s.session_id.startswith("t2i_"))
    ar_offs = sorted(s.arrival_offset_s for s in mixed_routing if s.session_id.startswith("ar_"))
    # Uniform GenTD26 deltas yield evenly spaced offsets; skewed BurstGPT deltas cluster early.
    assert abs(t2i_offs[1] - (16.0 / 3.0)) < 0.05
    assert ar_offs[1] < 1.0

    # Verify balanced Phase-Shift 360s: Phase 2 (`ar50_dit50`) AR arrival span >= 80.0s (no 6.3s collapse!)
    ps = build_online_phase_shift_suite(
        pools,
        b_deltas,
        g_deltas,
        phase_duration_s=120.0,
        target_qps=0.8,
        seed=2026,
    )
    assert [p["phase_label"] for p in ps.phase_schedule] == ["ar80_dit20", "ar20_dit80", "ar50_dit50"]
    assert all(s.phase_label in {"ar80_dit20", "ar20_dit80", "ar50_dit50"} for s in ps.sessions)
    p2_ar_offs = [
        float(s.arrival_offset_s)
        for s in ps.sessions
        if s.phase_label == "ar50_dit50"
        and s.turns
        and s.turns[0].modality_plan in ("ar_only", "cot_then_dit")
    ]
    assert len(p2_ar_offs) >= 2
    assert (max(p2_ar_offs) - min(p2_ar_offs)) >= 80.0

    # Verify `build_m3_active_accuracy_suite` (moved to `interleaved_bench.py`) on self-contained mini fixtures.
    for rel_img in ("images/mse-bench/m0.png", "images/magicbrush/mb0.png", "images/mmvet/v0.png"):
        p_img = tmp_path / rel_img
        p_img.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(_make_structured_rgb(32, 32, seed=3)).save(p_img)

    (sources / "mse_editing_sessions.json").write_text(
        json.dumps(
            {
                "templates": [
                    {
                        "source_image": {"path": "m0.png"},
                        "turns": [{"prompt": "edit one"}, {"prompt": "edit two"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "magicbrush_sessions.json").write_text(
        json.dumps(
            [
                {
                    "img_id": "mb0",
                    "source_image": "images/magicbrush/mb0.png",
                    "turns": [{"turn_index": 1, "instruction": "add red hat"}],
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "mmvet_questions.json").write_text(
        json.dumps(
            [
                {"image_path": "images/mmvet/v0.png", "question": "What color is the sky?", "answer": "blue"},
                {"image_path": "images/mmvet/v0.png", "question": "Count objects.", "answer": "two<OR>2"},
            ]
        ),
        encoding="utf-8",
    )
    (sources / "geneval_evaluation_metadata.jsonl").write_text(
        "\n".join(
            json.dumps({"tag": "colors", "prompt": f"a blue cup {k}", "include": [{"class": "cup", "count": 1}]})
            for k in range(6)
        )
        + "\n",
        encoding="utf-8",
    )

    acc = build_m3_active_accuracy_suite(
        tmp_path,
        num_co_creation=3,
        num_edit_4turn=2,
        num_multiturn_mmvet=2,
        num_cot_geneval=2,
    )
    assert acc.schema == "morphserve.interleaved_bench.v2"
    quads_acc = {s.quadrant for s in acc.sessions}
    assert quads_acc == {"interleaved_co_creation", "multiturn_edit", "multi_image_vqa", "visual_cot_highres"}
    q1 = [s for s in acc.sessions if s.quadrant == "interleaved_co_creation"]
    assert len(q1) == 3 and all(len(s.turns) == 4 and s.turns[2].use_prev_image and s.turns[3].use_prev_image for s in q1)
    q4 = [s for s in acc.sessions if s.quadrant == "visual_cot_highres"]
    assert len(q4) == 2 and all(len(s.turns) == 2 and bool(s.turns[1].geneval_meta) for s in q4)


def test_policy_presets_ablations_static_splits_and_internvlu_budget(tmp_path: Path) -> None:
    ablations = build_ablation_configs(num_gpus=8)
    assert [name for name, _ in ablations] == [
        "full_morphserve",
        "no_m1_tripool_vmm",
        "no_m1_scoped_vae",
        "no_m2_dual_path",
        "no_m2_pacer_numa",
        "no_m3_cow_loopback",
        "no_m4_layer_chunk",
        "no_queue_sched",
    ]
    splits = build_static_split_configs(num_gpus=8)
    assert [name for name, _ in splits] == [
        "morphserve_elastic",
        "static_2und_6gen",
        "static_4und_4gen",
        "static_6und_2gen",
    ]
    p26 = EnginePolicyConfig.from_preset("static_2und_6gen", num_gpus=8)
    assert len(p26.static_und_ranks) == 2 and len(p26.static_gen_ranks) == 6

    ov, budget = compute_internvlu_memory_quota_overrides(
        "morphserve", num_gpus=8, vendor_dir=tmp_path, constrained_gib=12.0, total_gpu_gib=40.0
    )
    assert ov["allow_vmm_overflow"] is False
    assert isinstance(ov["inter_turn_think_s_range"], tuple)
    assert budget["runtime_combined_budget_mib"] <= budget["quota_limit_mib"] + 1e-3

    cfg = EnginePolicyConfig.from_preset("morphserve", num_gpus=8, **ov)
    ok_audit = verify_internvlu_memory_quota(
        {
            "peak_pytorch_reserved_mib": 4800.0,
            "vmm_physical_pool_mib": 6688.0,
            "peak_gpu_reserved_mib": 11488.0,
            "vmm_overflow_creates": 0,
        },
        cfg,
        constrained_gib=12.0,
    )
    assert ok_audit["within_quota"] is True
    bad_audit = verify_internvlu_memory_quota(
        {
            "peak_pytorch_reserved_mib": 7000.0,
            "vmm_physical_pool_mib": 6688.0,
            "peak_gpu_reserved_mib": 13688.0,
            "vmm_overflow_creates": 0,
        },
        cfg,
        constrained_gib=12.0,
    )
    assert bad_audit["within_quota"] is False

    ivlu_suite = build_internvlu_suite(num_sessions=10, offered_qps=1.6, bench_root=None)
    assert len(ivlu_suite.sessions) == 10
    q_counts: dict[str, int] = {}
    for s in ivlu_suite.sessions:
        q_counts[s.quadrant] = q_counts.get(s.quadrant, 0) + 1
    assert q_counts == {"interleaved_co_creation": 4, "multiturn_edit": 3, "multi_image_vqa": 3}


# ==================== 3. ExactnessEvaluator Tier-1 & Tier-2 ====================


def test_exactness_tier1_sha256_and_tier2_ssim_psnr_mae_against_reference(tmp_path: Path) -> None:
    base = _make_structured_rgb(64, 64, seed=42)
    shifted = np.clip(base.astype(np.int16) + 10, 0, 255).astype(np.uint8)
    # Construct a pair with exact uniform +10 diff across every channel (stay inside [20..220] so zero clipping).
    unclipped_base = np.clip(base, 20, 220).astype(np.uint8)
    unclipped_plus10 = (unclipped_base.astype(np.int16) + 10).astype(np.uint8)
    one_lsb = unclipped_base.copy()
    one_lsb[0, 0, 0] = np.uint8(int(one_lsb[0, 0, 0]) + 1)
    inverted = (255 - unclipped_base).astype(np.uint8)

    img_base = Image.fromarray(unclipped_base)
    img_plus10 = Image.fromarray(unclipped_plus10)
    img_lsb = Image.fromarray(one_lsb)
    img_inv = Image.fromarray(inverted)

    ev = ExactnessEvaluator()
    t1_same = ev.verify_tier1_bit_exact(img_base, Image.fromarray(unclipped_base.copy()))
    assert t1_same["pass"] is True
    t1_lsb = ev.verify_tier1_bit_exact(img_base, img_lsb)
    assert t1_lsb["pass"] is False

    # Independent reference PSNR = 20 * log10(255 / 10) = 28.1308 dB, MAE = 10.0, max_abs_diff = 10.
    qual_plus10 = compute_ssim_rgb(img_base, img_plus10)
    expected_psnr = round(20.0 * math.log10(255.0 / 10.0), 2)
    assert qual_plus10["mae_u8"] == 10.0
    assert qual_plus10["max_abs_diff_u8"] == 10
    assert abs(qual_plus10["psnr_db"] - expected_psnr) <= 0.02
    assert 0.90 < qual_plus10["ssim"] < 0.999

    qual_same = compute_ssim_rgb(img_base, Image.fromarray(unclipped_base.copy()))
    assert qual_same["ssim"] == 1.0 and qual_same["psnr_db"] == 99.99 and qual_same["mae_u8"] == 0.0

    qual_inv = compute_ssim_rgb(img_base, img_inv)
    assert qual_inv["ssim"] < 0.2

    # Save PNGs and verify `image_pair_metrics` & `evaluate_parity_and_quality` including `geneval_pairs`.
    p_ref = tmp_path / "ref.png"
    p_same = tmp_path / "same.png"
    p_shift = tmp_path / "shift.png"
    img_base.save(p_ref)
    img_base.save(p_same)
    Image.fromarray(shifted).save(p_shift)

    m_same = image_pair_metrics(p_ref, p_same)
    assert m_same["sha256_equal"] is True and m_same["ssim"] == 1.0
    assert pil_sha256(img_base) != pil_sha256(img_lsb)

    # Verify MMVet rubric (including negative numbers, fractions, and multi-word content-token recall).
    assert score_mmvet_answer("The sky is clearly blue and bright", "blue<AND>bright") == 1.0
    assert score_mmvet_answer("only blue", "blue<AND>bright") == 0.5
    assert score_mmvet_answer("two apples", "2<OR>two") == 1.0
    assert score_mmvet_answer("The solutions are -1 and -5.", "-1<AND>-5") == 1.0
    assert score_mmvet_answer("The simplified ratio is 5/4.", "5/4") == 1.0
    assert score_mmvet_answer("The image shows a red double-decker bus on a street.", "a red double decker bus") == 1.0

    owl = OwlV2GenEvalScorer(
        detector_fn=lambda _im, qs: [
            {"query": q, "score": 0.92, "box": [5.0, 5.0, 40.0, 40.0]}
            for q in qs
            if "cup" in q
        ]
    )
    clip = ClipEditingScorer(
        image_embed_fn=lambda _im: [3.0, 4.0],
        text_embed_fn=lambda _t: [3.0, 4.0],
    )
    evaluator = ExactnessEvaluator(owlv2_scorer=owl, clip_scorer=clip)

    sessions = [
        InterleavedSession(
            session_id="gv_0",
            quadrant="interleaved_co_creation",
            source_dataset="geneval",
            turns=[
                InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="plan"),
                InterleavedTurn(
                    turn_idx=1,
                    modality_plan="dit_t2i",
                    prompt="a photo of a cup",
                    geneval_meta={"tag": "single_object", "include": [{"class": "cup", "count": 1}]},
                ),
                InterleavedTurn(turn_idx=2, modality_plan="ar_only", prompt="inspect", use_prev_image=True),
                InterleavedTurn(turn_idx=3, modality_plan="dit_edit", prompt="refine", use_prev_image=True),
            ],
        )
    ]
    ref_out = [
        {
            "session_id": "gv_0",
            "status": "ok",
            "ar_texts": ["plan ok", "inspect ok"],
            "image_paths_out": [str(p_ref), str(p_ref)],
        }
    ]
    cand_strict_ok = [
        {
            "session_id": "gv_0",
            "status": "ok",
            "ar_texts": ["plan ok", "inspect ok"],
            "image_paths_out": [str(p_same), str(p_same)],
        }
    ]
    rep_strict = evaluator.evaluate_parity_and_quality(sessions, ref_out, cand_strict_ok, strict_ssim_1=True)
    assert rep_strict["lossless_pass"] is True
    assert rep_strict["post_dit_ar_exact_match_pct"] == 100.0
    assert rep_strict["geneval_sessions"] == 1
    assert rep_strict["geneval_sha256_exact_count"] == 1
    assert len(rep_strict["geneval_pairs"]) == 1 and rep_strict["geneval_pairs"][0]["session_id"] == "gv_0"
    assert rep_strict["geneval_ref_score_pct"] == 100.0
    assert rep_strict["clip_t_delta_pct"] == 0.0

    # Divergent post-DiT AR text or candidate error must fail `lossless_pass`.
    cand_bad_ar = [
        {
            "session_id": "gv_0",
            "status": "ok",
            "ar_texts": ["plan ok", "inspect DRIFT"],
            "image_paths_out": [str(p_same), str(p_same)],
        }
    ]
    rep_bad_ar = evaluator.evaluate_parity_and_quality(sessions, ref_out, cand_bad_ar, strict_ssim_1=True)
    assert rep_bad_ar["post_dit_ar_exact_match_pct"] == 0.0
    assert rep_bad_ar["lossless_pass"] is False

    cand_err = [{"session_id": "gv_0", "status": "error", "error": "boom"}]
    rep_err = evaluator.evaluate_parity_and_quality(sessions, ref_out, cand_err, strict_ssim_1=True)
    assert rep_err["cand_missing_or_failed_sessions"] == 1
    assert rep_err["lossless_pass"] is False


# ==================== 4. BenchmarkRunner, Interpolation & Breakdowns ====================


@dataclass
class _FakeReport:
    policy_name: str = "morphserve"
    offered_qps: float = 0.5
    status: str = "ok"
    total_sessions: int = 2
    completed_sessions: int = 2
    oom_sessions: int = 0
    error_sessions: int = 0
    joint_slo_goodput_rps: float = 0.5
    joint_slo_attainment_pct: float = 100.0
    host_shmem_delta_gib: float = 1.2
    sessions: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"session_id": "s0", "quadrant": "multiturn_edit", "status": "ok", "e2e_ms": 100.0},
            {"session_id": "s1", "quadrant": "multi_image_vqa", "status": "error", "error": "synthetic failure"},
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "offered_qps": self.offered_qps,
            "status": self.status,
            "total_sessions": self.total_sessions,
            "completed_sessions": self.completed_sessions,
            "oom_sessions": self.oom_sessions,
            "error_sessions": self.error_sessions,
            "joint_slo_goodput_rps": self.joint_slo_goodput_rps,
            "joint_slo_attainment_pct": self.joint_slo_attainment_pct,
            "host_shmem_delta_gib": self.host_shmem_delta_gib,
            "sessions": list(self.sessions),
        }


def test_benchmark_runner_lifecycle_run_options_and_exception_teardown(tmp_path: Path) -> None:
    events: list[str] = []
    seen_run_opts: list[RunWindowOptions | None] = []

    class _FakeEngine:
        def __init__(self, *, policy: Any, fail_on_run: bool = False, **_kw: Any) -> None:
            self.policy = policy
            self._fail = fail_on_run

        def start(self) -> dict[str, Any]:
            events.append("start")
            return {"host_shmem_delta_gib": 2.5}

        def run_suite(self, suite: WorkloadSuite, **kwargs: Any) -> _FakeReport:
            events.append(f"run:{suite.name}")
            seen_run_opts.append(kwargs.get("run_options"))
            if self._fail:
                raise RuntimeError("injected run_suite error")
            return _FakeReport(policy_name=self.policy.policy_name, offered_qps=float(kwargs.get("target_qps", 0.5)))

        def shutdown(self) -> None:
            events.append("shutdown")

    pol = EnginePolicyConfig.from_preset("morphserve", num_gpus=8)
    orig_warmup = pol.warmup_window_s
    suite = WorkloadSuite(name="tiny", trace_source="unit", base_qps=0.5, sessions=[])
    opts = RunWindowOptions(steady_window_mode="time", warmup_window_s=12.0, steady_window_s=48.0)

    runner = BenchmarkRunner(
        model_dir=tmp_path,
        bench_root=tmp_path,
        tmp_root_base=tmp_path,
        engine_factory=lambda **kw: _FakeEngine(**kw),
    )
    rows = runner.run_policy_sweep(
        pol,
        [SuiteRunSpec(suite=suite, target_qps=0.5, run_options=opts, extra_fields={"tag": "x"})],
        tmp_tag="fake_ok",
        include_cdf_outcomes=True,
        include_failed_sessions=True,
    )
    assert events == ["start", "run:tiny", "shutdown"]
    assert seen_run_opts == [opts]
    # Policy object must never be mutated in place.
    assert pol.warmup_window_s == orig_warmup
    assert len(rows) == 1
    assert "sessions" not in rows[0]
    assert rows[0]["error_samples"] == ["synthetic failure"]
    assert len(rows[0]["failed_sessions"]) == 1
    assert len(rows[0]["cdf_outcomes"]) == 2
    assert rows[0]["host_shmem_delta_gib"] == 2.5
    assert rows[0]["tag"] == "x"

    # Guaranteed `shutdown()` even when `run_suite` raises!
    events.clear()
    failing_runner = BenchmarkRunner(
        model_dir=tmp_path,
        bench_root=tmp_path,
        tmp_root_base=tmp_path,
        engine_factory=lambda **kw: _FakeEngine(fail_on_run=True, **kw),
    )
    with pytest.raises(RuntimeError, match="injected run_suite error"):
        failing_runner.run_policy_sweep(pol, [SuiteRunSpec(suite=suite, target_qps=0.5)], tmp_tag="fake_fail")
    assert events == ["start", "run:tiny", "shutdown"]


def test_interpolate_slo_threshold_qps_zero_oom_and_summary() -> None:
    clean_pts = [(0.20, 95.0, 0), (0.40, 85.0, 0), (0.60, 75.0, 0)]
    assert abs(interpolate_slo_threshold_qps(clean_pts, threshold_pct=90.0, require_zero_oom=False) - 0.30) < 1e-4
    assert abs(interpolate_slo_threshold_qps(clean_pts, threshold_pct=90.0, require_zero_oom=True) - 0.30) < 1e-4
    assert abs(interpolate_slo_threshold_qps(clean_pts, threshold_pct=80.0, require_zero_oom=True) - 0.50) < 1e-4

    # OOM at the drop point: `require_zero_oom=False` interpolates to 0.30, while `require_zero_oom=True` MUST stop at 0.20!
    oom_drop_pts = [(0.20, 95.0, 0), (0.40, 85.0, 2)]
    assert abs(interpolate_slo_threshold_qps(oom_drop_pts, threshold_pct=90.0, require_zero_oom=False) - 0.30) < 1e-4
    assert abs(interpolate_slo_threshold_qps(oom_drop_pts, threshold_pct=90.0, require_zero_oom=True) - 0.20) < 1e-4

    # OOM on a high-attainment point itself: `require_zero_oom=True` ignores the OOM point!
    oom_high_pts = [(0.20, 95.0, 0), (0.40, 92.0, 1)]
    assert abs(interpolate_slo_threshold_qps(oom_high_pts, threshold_pct=90.0, require_zero_oom=False) - 0.40) < 1e-4
    assert abs(interpolate_slo_threshold_qps(oom_high_pts, threshold_pct=90.0, require_zero_oom=True) - 0.20) < 1e-4

    # Canonical `compute_sustainable_qps_summary`
    sweep_rows = [
        {
            "policy_name": "morphserve",
            "offered_qps": 0.20,
            "steady_turn_slo_attainment_pct": 95.0,
            "steady_turn_slo_goodput_rps": 0.19,
            "joint_slo_attainment_pct": 85.0,
            "joint_slo_goodput_rps": 0.17,
            "full_trace_joint_slo_attainment_pct": 88.0,
            "full_trace_goodput_rps": 0.176,
            "steady_state_survival_pct": 100.0,
            "survival_rate_pct": 100.0,
            "steady_oom_sessions": 0,
            "oom_sessions": 0,
            "total_sessions": 48,
            "peak_nvml_used_mib": 34000.0,
            "vmm_physical_pool_mib": 14592.0,
            "kv_host_swap_outs": 2,
            "kv_host_swap_ins": 2,
        },
        {
            "policy_name": "morphserve",
            "offered_qps": 0.40,
            "steady_turn_slo_attainment_pct": 85.0,
            "steady_turn_slo_goodput_rps": 0.34,
            "joint_slo_attainment_pct": 65.0,
            "joint_slo_goodput_rps": 0.26,
            "full_trace_joint_slo_attainment_pct": 70.0,
            "full_trace_goodput_rps": 0.28,
            "steady_state_survival_pct": 100.0,
            "survival_rate_pct": 100.0,
            "steady_oom_sessions": 0,
            "oom_sessions": 0,
            "total_sessions": 48,
            "peak_nvml_used_mib": 36000.0,
            "vmm_physical_pool_mib": 14592.0,
            "kv_host_swap_outs": 4,
            "kv_host_swap_ins": 4,
        },
    ]
    summary = compute_sustainable_qps_summary(sweep_rows)["morphserve"]
    assert abs(summary["turn_max_sustainable_qps_90pct"] - 0.30) < 1e-4
    assert abs(summary["turn_max_sustainable_qps_zero_oom_90pct"] - 0.30) < 1e-4
    assert abs(summary["max_sustainable_qps_80pct"] - 0.25) < 1e-4
    assert summary["slo_attainment_curve"][0]["kv_host_swap_ins"] == 2
    assert summary["slo_attainment_curve"][0]["vmm_physical_pool_mib"] == 14592.0


def test_phase_quadrant_breakdowns_and_harness_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = [
        {
            "session_id": "s0",
            "phase_label": "ar80_dit20",
            "quadrant": "multi_image_vqa",
            "status": "ok",
            "joint_slo_met": True,
            "ar_slo_met": True,
            "dit_slo_met": True,
            "e2e_ms": 120.0,
            "active_jct_ms": 100.0,
            "ar_ttfts_ms": [20.0],
            "ar_tpots_ms": [12.0],
            "dit_turns_ms": [],
            "path_a_ships": 1,
            "path_b_morphs": 0,
            "total_yields": 2,
        },
        {
            "session_id": "s1",
            "phase_label": "ar20_dit80",
            "quadrant": "multiturn_edit",
            "status": "oom",
            "joint_slo_met": False,
            "ar_slo_met": False,
            "dit_slo_met": False,
            "e2e_ms": 0.0,
            "active_jct_ms": 0.0,
            "path_a_ships": 0,
            "path_b_morphs": 1,
            "total_yields": 0,
        },
    ]
    pb = compute_phase_breakdown(outcomes)
    assert pb["ar80_dit20"]["joint_slo_attainment_pct"] == 100.0
    assert pb["ar80_dit20"]["path_a_ships"] == 1
    assert pb["ar20_dit80"]["oom_sessions"] == 1 and pb["ar20_dit80"]["survival_rate_pct"] == 0.0

    qb = compute_quadrant_breakdown(outcomes)
    assert qb["multi_image_vqa"]["joint_slo_pct"] == 100.0
    assert qb["multiturn_edit"]["oom_count"] == 1

    assert parse_csv_floats("0.1, 0.5, 1.1") == [0.1, 0.5, 1.1]
    assert parse_csv_floats(None) == []
    assert parse_csv_strings("morphserve, monolithic_dp ") == ["morphserve", "monolithic_dp"]

    bench = tmp_path / "bench"
    bench.mkdir()
    alt_src = tmp_path / "bench_sources"
    alt_src.mkdir()
    assert resolve_sources_dir(bench) == alt_src
    direct_src = bench / "sources"
    direct_src.mkdir()
    assert resolve_sources_dir(bench) == direct_src

    monkeypatch.setenv("MORPHSERVE_LOCAL_TMP", str(tmp_path / "ltmp"))
    assert local_tmp_base() == tmp_path / "ltmp"

    comp = compact_report_dict(_FakeReport(), max_error_samples=1)
    assert "sessions" not in comp and comp["error_samples"] == ["synthetic failure"]


# ==================== 5. Smoke Validation Seam (Zero Monkey-Patching) ====================


def test_run_smoke_validation_with_fleet_factory_seam(tmp_path: Path) -> None:
    bench_root = tmp_path / "bench"
    out_dir = tmp_path / "out"
    smoke_suite = WorkloadSuite(
        name="smoke_8gpu_12",
        trace_source="unit",
        base_qps=1.0,
        sessions=[
            InterleavedSession(
                session_id="smk_ar_0",
                quadrant="multi_image_vqa",
                source_dataset="unit",
                arrival_offset_s=0.0,
                turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="describe scene", max_ar_tokens=4)],
            ),
            InterleavedSession(
                session_id="smk_dit_1",
                quadrant="visual_cot_highres",
                source_dataset="unit",
                arrival_offset_s=0.05,
                turns=[InterleavedTurn(turn_idx=0, modality_plan="dit_t2i", prompt="mountain lake", dit_steps=2)],
            ),
        ],
    )
    smoke_suite.save_json(bench_root / "suites" / "smoke_8gpu_12.json")

    rep = validate_phase2_engine_smoke.run_smoke_validation(
        model_dir=tmp_path,
        bench_root=bench_root,
        output_dir=out_dir,
        qps=2.0,
        num_gpus=1,
        policies=("morphserve", "monolithic_dp"),
        fleet_factory=lambda pol: InProcessWorkerFleet(pol),
    )
    assert rep["status"] == "ok"
    assert len(rep["results"]) == 2
    assert (out_dir / "m22-smoke-report.json").is_file()


# ==================== 6. Multinode TCP Partial-Chunk & EOF Handling ====================


def test_recv_exact_and_recv_json_on_master_partial_chunks_and_eof() -> None:
    # 1. Partial-chunk assembly over socketpair
    a, b = socket.socketpair()
    try:
        payload = b"morphserve_exact_read_test_payload"

        def _drip_sender() -> None:
            for idx in range(0, len(payload), 3):
                a.sendall(payload[idx : idx + 3])
            a.shutdown(socket.SHUT_WR)

        th = threading.Thread(target=_drip_sender, daemon=True)
        th.start()
        assert _recv_exact(b, len(payload)) == payload
        th.join(timeout=5.0)
    finally:
        a.close()
        b.close()

    # 2. Premature EOF raises ConnectionError cleanly
    c, d = socket.socketpair()
    try:
        c.sendall(b"short")
        c.close()
        with pytest.raises(ConnectionError, match="unexpected EOF"):
            _recv_exact(d, 16)
    finally:
        d.close()

    # 3. `_recv_json_on_master` with 1-byte header trickle and chunked JSON body
    port = find_free_tcp_port()
    expected_obj = {"node_rank": 1, "ranks": [{"global_rank": 8, "vmm_ok": True}]}
    raw = json.dumps(expected_obj).encode("utf-8")
    frame = struct.pack("!I", len(raw)) + raw
    received: dict[str, Any] = {}

    def _server() -> None:
        received["data"] = _recv_json_on_master(port)

    srv_th = threading.Thread(target=_server, daemon=True)
    srv_th.start()
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0) as cli:
                # Drip header 1 byte at a time, then body 5 bytes at a time
                for i in range(4):
                    cli.sendall(frame[i : i + 1])
                for i in range(4, len(frame), 5):
                    cli.sendall(frame[i : i + 5])
                break
        except OSError:
            threading.Event().wait(0.02)
    srv_th.join(timeout=5.0)
    assert received["data"] == expected_obj

    # 4. `_recv_json_on_master` fails cleanly on truncated peer close
    port2 = find_free_tcp_port()
    err_box: list[BaseException] = []

    def _server_eof() -> None:
        try:
            _recv_json_on_master(port2)
        except BaseException as exc:
            err_box.append(exc)

    srv_th2 = threading.Thread(target=_server_eof, daemon=True)
    srv_th2.start()
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port2), timeout=1.0) as cli:
                cli.sendall(struct.pack("!I", 64) + b'{"trunc":')
                break
        except OSError:
            threading.Event().wait(0.02)
    srv_th2.join(timeout=5.0)
    assert len(err_box) == 1 and isinstance(err_box[0], ConnectionError)


# ==================== 7. CLI Argument Parser Contracts ====================


def test_phase2_cli_arg_parsers_and_deprecated_flag_removal() -> None:
    p_main = run_phase2_bagel_main.build_arg_parser()
    parsed = p_main.parse_args(
        [
            "--mode",
            "suite_a_lowqps",
            "--model-dir",
            "/m",
            "--bench-root",
            "/b",
            "--output-dir",
            "/o",
            "--min-sessions",
            "32",
            "--max-sessions",
            "64",
            "--qps-points",
            "0.1,0.4",
            "--policies",
            "morphserve,monolithic_dp",
        ]
    )
    assert parsed.mode == "suite_a_lowqps"
    assert parsed.min_sessions == 32
    assert parsed.max_sessions == 64

    # Deprecated flags `--low-sessions` and `--knee-sessions` must be cleanly removed.
    for removed_flag in ("--low-sessions", "--knee-sessions"):
        with pytest.raises(SystemExit):
            p_main.parse_args(
                [
                    "--mode",
                    "suite_a_lowqps",
                    "--model-dir",
                    "/m",
                    "--bench-root",
                    "/b",
                    "--output-dir",
                    "/o",
                    removed_flag,
                    "24",
                ]
            )

    p_deep = run_phase2_overnight_deep.build_arg_parser()
    a_deep = p_deep.parse_args(["--model-dir", "/m", "--bench-root", "/b", "--output-dir", "/o"])
    assert a_deep.main_sessions == 96 and a_deep.skew_sessions == 80

    p_ivlu = validate_phase2_internvl_u.build_arg_parser()
    a_ivlu = p_ivlu.parse_args(
        ["--model-dir", "/m", "--vendor-dir", "/v", "--output-dir", "/o", "--constrained-gib", "12.0", "--smoke-1gpu"]
    )
    assert a_ivlu.smoke_1gpu is True and a_ivlu.constrained_gib == 12.0

    p_mn = validate_phase2_multinode_2x8.build_arg_parser()
    a_mn = p_mn.parse_args(["--output-dir", "/o", "--mode", "both", "--qps", "1.6", "--synthetic-e2e"])
    assert a_mn.mode == "both" and a_mn.synthetic_e2e is True

    p_smk = validate_phase2_engine_smoke.build_arg_parser()
    a_smk = p_smk.parse_args(["--model-dir", "/m", "--bench-root", "/b", "--output-dir", "/o", "--num-gpus", "8"])
    assert a_smk.num_gpus == 8 and a_smk.qps == 0.8
