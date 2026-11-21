"""Offline unit tests for Milestones M2.3 (stratifier + ablations) and M2.5 (64-GPU simulator)."""

from __future__ import annotations

from pathlib import Path

from morphserve.bench.interleaved_bench import InterleavedSession, InterleavedTurn, WorkloadSuite
from morphserve.bench.run_phase2_bagel_main import build_ablation_configs, stratify_suite
from morphserve.sim import run_scale_and_interconnect_sweep


def test_stratify_suite_preserves_quadrant_balance() -> None:
    sessions = []
    quads = ("visual_cot",) * 12 + ("multiturn_edit",) * 12 + ("multi_image_vqa",) * 10 + ("highres_t2i",) * 6
    for i, q in enumerate(quads):
        sessions.append(
            InterleavedSession(
                session_id=f"s_{i:02d}",
                quadrant=q,
                source_dataset="unit",
                arrival_offset_s=float(i) * 0.5,
                turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="hello")],
            )
        )
    suite = WorkloadSuite(name="full40", trace_source="BurstGPT+GenTD26", base_qps=1.0, sessions=sessions)
    sliced = stratify_suite(suite, max_sessions=20)
    assert len(sliced.sessions) == 20
    by_q = {}
    for s in sliced.sessions:
        by_q[s.quadrant] = by_q.get(s.quadrant, 0) + 1
    assert by_q == {"visual_cot": 6, "multiturn_edit": 6, "multi_image_vqa": 5, "highres_t2i": 3}


def test_build_ablation_configs_ladder() -> None:
    ladder = build_ablation_configs(num_gpus=8)
    names = [n for n, _ in ladder]
    assert names == [
        "full_morphserve",
        "no_m4_layer_chunk",
        "no_m3_cow_loopback",
        "no_m2_dual_path",
        "no_m1_tripool_vmm",
    ]
    cfgs = dict(ladder)
    assert cfgs["no_m4_layer_chunk"].preemption_mode == "non_preemptive"
    assert not cfgs["no_m3_cow_loopback"].use_module3_cow_loopback
    assert cfgs["no_m2_dual_path"].routing_mode == "sticky_least_loaded"
    assert not cfgs["no_m1_tripool_vmm"].use_tripool_vmm


def test_cluster_simulator_64gpu_sweep(tmp_path: Path) -> None:
    rep = run_scale_and_interconnect_sweep(output_dir=tmp_path)
    assert rep["status"] == "ok"
    assert rep["calibration"]["within_5pct_gate"] is True
    # 3 interconnects x 3 scales (8,16,64) x 6 policies = 54 entries
    assert len(rep["scaling_matrix"]) == 54
    nvlink_64 = [
        r
        for r in rep["scaling_matrix"]
        if r["interconnect"] == "H100_SXM5_NVLink900G_RDMA400G" and r["num_gpus"] == 64
    ]
    by_pol = {r["policy_name"]: r for r in nvlink_64}
    assert by_pol["morphserve"]["joint_slo_goodput_rps"] > by_pol["monolithic_dp"]["joint_slo_goodput_rps"]
    assert by_pol["morphserve"]["crossover_kv_mib"] > 1000.0
