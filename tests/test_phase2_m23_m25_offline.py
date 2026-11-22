"""Offline unit tests for M2.3 (stratifier, ablations, ExactnessEvaluator) and M2.5 (physics 64-GPU simulator)."""

from __future__ import annotations

from pathlib import Path
import tempfile

from morphserve.bench.exactness import ExactnessEvaluator, score_mmvet_answer
from morphserve.bench.interleaved_bench import InterleavedSession, InterleavedTurn, WorkloadSuite
from morphserve.bench.run_phase2_bagel_main import build_ablation_configs, stratify_suite
from morphserve.bench.run_phase2_overnight_deep import build_static_split_configs
from morphserve.bench.validate_phase2_internvl_u import run_internvlu_generality_validation  # noqa: F401
from morphserve.bench.validate_phase2_multinode_2x8 import evaluate_cross_node_routing_check
from morphserve.engine import EnginePolicyConfig
from morphserve.sim import run_scale_and_interconnect_sweep
from morphserve.sim.cluster_simulator import SimGpuMemoryLedger


def _scratch_dir() -> str | None:
    d_root = Path("/mnt/d/Transfer/MORPHSERVE")
    if Path("/mnt/d/Transfer").is_dir():
        d_root.mkdir(parents=True, exist_ok=True)
        return str(d_root)
    return None


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


def test_exactness_evaluator_mmvet_and_parity() -> None:
    assert score_mmvet_answer("The painting is by Vincent van Gogh in 1889.", "van gogh<AND>1889") == 1.0
    assert score_mmvet_answer("It shows four apples.", "4<OR>four") == 1.0
    assert score_mmvet_answer("Unrelated answer", "van gogh<AND>1889") == 0.0

    sessions = [
        InterleavedSession(
            session_id="vqa_0",
            quadrant="multi_image_vqa",
            source_dataset="mmvet",
            turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="who?", expected_answer="van gogh")],
        )
    ]
    ref_out = [{"session_id": "vqa_0", "status": "ok", "ar_texts": ["Vincent van Gogh"], "image_paths_out": []}]
    cand_out = [{"session_id": "vqa_0", "status": "ok", "ar_texts": ["Vincent van Gogh"], "image_paths_out": []}]
    res = ExactnessEvaluator().evaluate_parity_and_quality(sessions, ref_out, cand_out)
    assert res["lossless_pass"] is True
    assert res["ar_greedy_exact_match_pct"] == 100.0
    assert res["mmvet_delta_pct"] == 0.0


def test_sim_gpu_memory_ledger_physical_conservation() -> None:
    ms_pol = EnginePolicyConfig.from_preset("morphserve", num_gpus=8)
    mono_pol = EnginePolicyConfig.from_preset("monolithic_dp", num_gpus=8)
    ms_ledger = SimGpuMemoryLedger(gpu_id=0, total_hbm_mib=40_960.0, policy=ms_pol, role="hybrid")
    mono_ledger = SimGpuMemoryLedger(gpu_id=0, total_hbm_mib=40_960.0, policy=mono_pol, role="hybrid")

    # Under heavy multi-session KV + 1024px VAE spike (~5.8 GiB):
    # MorphServe auto-evicts clean tail expert handles (`expert_evictions > 0`) with zero OOM,
    # whereas unpaged monolithic_dp exceeds 40 GiB HBM cleanly from byte conservation!
    for sid in range(4):
        assert ms_ledger.upsert_session_kv(sid, 1800.0) is True
        mono_ledger.upsert_session_kv(sid, 1800.0)
    assert ms_ledger.execute_dit_and_vae(1024) is True
    assert ms_ledger.expert_evictions > 0
    assert mono_ledger.execute_dit_and_vae(1024) is False


def test_cluster_simulator_64gpu_sweep() -> None:
    with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmp:
        rep = run_scale_and_interconnect_sweep(output_dir=Path(tmp))
    assert rep["status"] == "ok"
    assert rep["calibration"]["zero_parameter_fitting"] is True
    assert rep["calibration"]["within_8pct_oos_gate"] is True
    assert len(rep["scaling_matrix"]) == 54
    nvlink_64 = [
        r
        for r in rep["scaling_matrix"]
        if r["interconnect"] == "H100_SXM5_NVLink900G_RDMA400G" and r["num_gpus"] == 64
    ]
    by_pol = {r["policy_name"]: r for r in nvlink_64}
    assert by_pol["morphserve"]["joint_slo_goodput_rps"] > by_pol["monolithic_dp"]["joint_slo_goodput_rps"]
    assert by_pol["morphserve"]["crossover_kv_mib"] > 1000.0


def test_build_static_split_configs() -> None:
    splits = dict(build_static_split_configs(num_gpus=8))
    assert set(splits.keys()) == {"morphserve_elastic", "static_2und_6gen", "static_4und_4gen", "static_6und_2gen"}
    assert len(splits["static_2und_6gen"].static_und_ranks) == 2
    assert len(splits["static_6und_2gen"].static_und_ranks) == 6


def test_multinode_16gpu_cross_node_routing_check() -> None:
    for inter_bw in (1.1, 11.5):
        chk = evaluate_cross_node_routing_check(
            world_size=16,
            gpus_per_node=8,
            intra_bw=24.0,
            socket_bw=18.0,
            inter_bw=inter_bw,
        )
        assert chk["routing_pass"] is True
        assert chk["bridge_24mib_decision"]["dst_gpu"] == 8
        assert chk["suppressed_cross_node_1gib_kv_ship"] is True
