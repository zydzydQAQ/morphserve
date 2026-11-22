"""Offline unit & integration tests for Phase-2 M2.1 (Interleaved-Bench) and M2.2 (MorphServeClusterEngine)."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from morphserve.bench.interleaved_bench import (  # noqa: E402
    InterleavedSession,
    InterleavedTurn,
    WorkloadSuite,
    assign_dual_trace_arrivals,
    compose_weighted_suite,
    load_burstgpt_interarrivals,
    load_gentd26_interarrivals,
    scale_arrivals_to_qps,
)
from morphserve.engine import (  # noqa: E402
    EnginePolicyConfig,
    InProcessWorkerFleet,
    MorphServeClusterEngine,
    MultinodeCoordinatorHub,
    TcpMultinodeWorkerFleet,
    VALID_POLICIES,
    run_multinode_worker_daemon,
)
from morphserve.models import SyntheticCPTGAdapter  # noqa: E402
from morphserve.router import ClusterTopologyProfile, find_free_tcp_port  # noqa: E402


def _scratch_dir() -> str | None:
    d_root = Path("/mnt/d/Transfer/MORPHSERVE")
    if Path("/mnt/d/Transfer").is_dir():
        d_root.mkdir(parents=True, exist_ok=True)
        return str(d_root)
    return None


def _build_mini_interleaved_suite() -> WorkloadSuite:
    return WorkloadSuite(
        name="inproc_mini_4",
        trace_source="unit_test",
        base_qps=4.0,
        sessions=[
            InterleavedSession(
                session_id="s_vcot",
                quadrant="visual_cot",
                source_dataset="unit",
                arrival_offset_s=0.00,
                turns=[
                    InterleavedTurn(turn_idx=0, modality_plan="cot_then_dit", prompt="sketch layout", max_ar_tokens=6, dit_steps=3),
                    InterleavedTurn(turn_idx=1, modality_plan="ar_only", prompt="verify objects", max_ar_tokens=6),
                ],
            ),
            InterleavedSession(
                session_id="s_edit",
                quadrant="multiturn_edit",
                source_dataset="unit",
                arrival_offset_s=0.01,
                turns=[
                    InterleavedTurn(turn_idx=0, modality_plan="dit_t2i", prompt="cabin by lake", dit_steps=3),
                    InterleavedTurn(turn_idx=1, modality_plan="dit_edit", prompt="add autumn leaves", dit_steps=3),
                    InterleavedTurn(turn_idx=2, modality_plan="dit_edit", prompt="make it twilight", dit_steps=3),
                    InterleavedTurn(turn_idx=3, modality_plan="dit_edit", prompt="add snow peaks", dit_steps=3),
                ],
            ),
            InterleavedSession(
                session_id="s_vqa",
                quadrant="multi_image_vqa",
                source_dataset="unit",
                arrival_offset_s=0.02,
                turns=[
                    InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="compare scenes", max_ar_tokens=6),
                ],
            ),
            InterleavedSession(
                session_id="s_hires",
                quadrant="highres_t2i",
                source_dataset="unit",
                arrival_offset_s=0.03,
                turns=[
                    InterleavedTurn(
                        turn_idx=0, modality_plan="dit_t2i", prompt="mountain vista", height=1024, width=1024, dit_steps=3
                    ),
                ],
            ),
        ],
    )


class TestPhase2M21M22Offline(unittest.TestCase):
    def test_trace_parsers_and_qps_scaling(self) -> None:
        with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmp:
            td = Path(tmp)
            burst_csv = td / "burst.csv"
            burst_csv.write_text(
                "Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n"
                "5,ChatGPT,100,20,120,Conversation log\n"
                "15,ChatGPT,100,20,120,Conversation log\n"
                "35,GPT-4,100,20,120,Conversation log\n",
                encoding="utf-8",
            )
            gentd_csv = td / "gentd.csv"
            gentd_csv.write_text(
                "gmt_create,predict_type,predict_status,exec_time_seconds,groupId\n"
                "2024-11-15 16:57:50,TXT_2_IMG,SUCCEED,32.0,G0\n"
                "2024-11-15 16:58:00,TXT_2_IMG,FAILED,1.0,G0\n"
                "2024-11-15 16:58:10,IMG_2_IMG,SUCCEED,40.0,G0\n"
                "2024-11-15 16:58:50,TXT_2_IMG,SUCCEED,20.0,G0\n",
                encoding="utf-8",
            )
            bd = load_burstgpt_interarrivals(burst_csv)
            gd = load_gentd26_interarrivals(gentd_csv)
            self.assertAlmostEqual(sum(bd) / len(bd), 1.0, places=5)
            self.assertAlmostEqual(sum(gd) / len(gd), 1.0, places=5)

            offs = scale_arrivals_to_qps(bd, target_qps=2.0, count=5)
            self.assertEqual(len(offs), 5)
            self.assertEqual(offs[0], 0.0)
            self.assertTrue(all(offs[i] >= offs[i - 1] for i in range(1, len(offs))))

    def test_weighted_suite_composition_and_roundtrip(self) -> None:
        pools = {
            q: [
                InterleavedSession(
                    session_id=f"{q}_0",
                    quadrant=q,
                    source_dataset="test",
                    turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only" if q == "multi_image_vqa" else "dit_t2i", prompt="p")],
                )
            ]
            for q in ("visual_cot", "multiturn_edit", "multi_image_vqa", "highres_t2i")
        }
        suite = compose_weighted_suite(
            "mix_test",
            pools,
            {"visual_cot": 6, "multiturn_edit": 6, "multi_image_vqa": 5, "highres_t2i": 3},
            [0.5, 1.5],
            [0.8, 1.2],
            target_qps=1.0,
        )
        self.assertEqual(len(suite.sessions), 20)
        raw = json.dumps(suite.to_dict())
        loaded = WorkloadSuite.from_dict(json.loads(raw))
        self.assertEqual(len(loaded.sessions), 20)
        ts = [s.arrival_offset_s for s in loaded.sessions]
        self.assertEqual(ts, sorted(ts))

        assign_dual_trace_arrivals(loaded.sessions, [1.0], [1.0], target_qps=2.0)
        self.assertEqual(loaded.sessions[0].arrival_offset_s, 0.0)

    def test_engine_policy_presets_matrix(self) -> None:
        self.assertEqual(len(VALID_POLICIES), 6)
        ms = EnginePolicyConfig.from_preset("morphserve", num_gpus=8)
        self.assertTrue(ms.use_tripool_vmm and ms.use_expert_pager and ms.use_scoped_vae)
        self.assertTrue(ms.use_shared_numa_mirror and ms.use_h2d_pacer and ms.use_module3_cow_loopback)
        self.assertEqual(ms.preemption_mode, "layer_chunk")
        self.assertEqual(ms.routing_mode, "dual_path")
        self.assertEqual(ms.initial_alphas, {g: 1.0 for g in range(8)})

        prism = EnginePolicyConfig.from_preset("prism_tetriserve_coloc", num_gpus=8)
        self.assertTrue(prism.use_tripool_vmm and prism.kv_vmm_only)
        self.assertFalse(prism.use_expert_pager)
        self.assertEqual(prism.preemption_mode, "step_preemptive")
        self.assertLessEqual(prism.super_pool_mib + prism.fine_kv_pool_mib, 4096)

        fission = EnginePolicyConfig.from_preset("static_nccl_fission", num_gpus=8)
        self.assertEqual(fission.static_und_ranks, (0, 1, 4, 5))
        self.assertEqual(fission.static_gen_ranks, (2, 3, 6, 7))

        shm = EnginePolicyConfig.from_preset("twostage_shm", num_gpus=8)
        self.assertEqual(shm.static_und_ranks, (0, 2, 4, 6))
        self.assertEqual(shm.static_gen_ranks, (1, 3, 5, 7))

    def test_inprocess_cluster_engine_all_6_policies_and_internvlu(self) -> None:
        suite = _build_mini_interleaved_suite()
        with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmp:
            td = Path(tmp)
            for preset in VALID_POLICIES:
                base_pol = EnginePolicyConfig.from_preset(preset, num_gpus=4)
                pol = replace(base_pol, inter_turn_think_s_range=(0.0, 0.0))
                fleet = InProcessWorkerFleet(pol)
                engine = MorphServeClusterEngine(
                    policy=pol,
                    model_dir=td,
                    bench_root=td,
                    tmp_root=td / preset,
                    fleet=fleet,
                )
                try:
                    engine.start()
                    rep = engine.run_suite(suite)
                    vmm_pool_mib = max(
                        (float(rt.adapter.memory_and_vmm_stats().get("vmm_physical_pool_mib", 0.0)) for rt in fleet._runtimes),
                        default=0.0,
                    )
                finally:
                    engine.shutdown()
                self.assertEqual(rep.status, "ok", msg=f"preset={preset} failed: {rep.to_dict()}")
                self.assertEqual(rep.completed_sessions, 4)
                self.assertEqual(rep.error_sessions, 0)
                self.assertEqual(rep.oom_sessions, 0)
                self.assertIn("2.0x", rep.slo_curve_by_k)
                self.assertGreaterEqual(rep.peak_gpu_reserved_mib, rep.peak_pytorch_reserved_mib)
                if pol.use_tripool_vmm:
                    self.assertGreater(vmm_pool_mib, 0.0)
                    self.assertAlmostEqual(
                        rep.peak_gpu_reserved_mib,
                        round(rep.peak_pytorch_reserved_mib + vmm_pool_mib, 2),
                        places=1,
                    )
                else:
                    self.assertEqual(rep.peak_gpu_reserved_mib, rep.peak_pytorch_reserved_mib)
                if preset == "morphserve":
                    self.assertGreater(rep.path_a_ships + rep.path_b_morphs, 0)

            # Also verify decoupled Bridge MLLM+DiT (`model_family="internvl_u"`) over unified MorphServeClusterEngine
            ivl_pol = replace(
                EnginePolicyConfig.from_preset("morphserve", num_gpus=4, model_family="internvl_u"),
                inter_turn_think_s_range=(0.0, 0.0),
            )
            ivl_fleet = InProcessWorkerFleet(ivl_pol)
            ivl_engine = MorphServeClusterEngine(
                policy=ivl_pol,
                model_dir=td,
                bench_root=td,
                tmp_root=td / "internvl_u",
                fleet=ivl_fleet,
            )
            try:
                ivl_engine.start()
                ivl_rep = ivl_engine.run_suite(suite)
                ivl_vmm_mib = max(
                    (float(rt.adapter.memory_and_vmm_stats().get("vmm_physical_pool_mib", 0.0)) for rt in ivl_fleet._runtimes),
                    default=0.0,
                )
            finally:
                ivl_engine.shutdown()
            self.assertEqual(ivl_rep.status, "ok")
            self.assertEqual(ivl_rep.completed_sessions, 4)
            self.assertGreater(ivl_vmm_mib, 0.0)
            self.assertGreaterEqual(ivl_rep.peak_gpu_reserved_mib, ivl_rep.peak_pytorch_reserved_mib)
            self.assertAlmostEqual(
                ivl_rep.peak_gpu_reserved_mib,
                round(ivl_rep.peak_pytorch_reserved_mib + ivl_vmm_mib, 2),
                places=1,
            )

    def test_inprocess_mid_dit_preemption_and_bounded_oom_retry(self) -> None:
        with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmp:
            td = Path(tmp)
            # 1. Single-worker mid-DiT preemption (`dit_step_sleep_s=0.04` ensures AR arrives mid-DiT)
            pol_1g = replace(
                EnginePolicyConfig.from_preset("morphserve", num_gpus=1),
                inter_turn_think_s_range=(0.0, 0.0),
            )
            preempt_suite = WorkloadSuite(
                name="preempt_check",
                trace_source="unit",
                base_qps=10.0,
                sessions=[
                    InterleavedSession(
                        session_id="dit_holder",
                        quadrant="highres_t2i",
                        source_dataset="unit",
                        arrival_offset_s=0.0,
                        turns=[InterleavedTurn(turn_idx=0, modality_plan="dit_t2i", prompt="slow dit", dit_steps=4)],
                    ),
                    InterleavedSession(
                        session_id="ar_urgent",
                        quadrant="multi_image_vqa",
                        source_dataset="unit",
                        arrival_offset_s=0.02,
                        turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="urgent ar", max_ar_tokens=4)],
                    ),
                ],
            )
            fleet_1g = InProcessWorkerFleet(
                pol_1g,
                adapter_factory=lambda r, p: SyntheticCPTGAdapter(r, p, dit_step_sleep_s=0.04),
            )
            eng_1g = MorphServeClusterEngine(pol_1g, td, td, tmp_root=td / "preempt", fleet=fleet_1g)
            try:
                eng_1g.start()
                rep_p = eng_1g.run_suite(preempt_suite)
            finally:
                eng_1g.shutdown()
            self.assertEqual(rep_p.completed_sessions, 2)
            self.assertGreaterEqual(rep_p.preempted_ar_count, 1)

            # 2. Transient rank-0 OOM recovered cleanly on peer rank via bounded 1-retry (`max_oom_retries=1`)
            pol_2g = replace(
                EnginePolicyConfig.from_preset("morphserve", num_gpus=2),
                inter_turn_think_s_range=(0.0, 0.0),
                max_oom_retries=1,
            )
            oom_suite = WorkloadSuite(
                name="oom_retry_check",
                trace_source="unit",
                base_qps=10.0,
                sessions=[
                    InterleavedSession(
                        session_id="s_retry",
                        quadrant="multi_image_vqa",
                        source_dataset="unit",
                        arrival_offset_s=0.0,
                        turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="recover me", max_ar_tokens=4)],
                    )
                ],
            )
            fleet_2g = InProcessWorkerFleet(
                pol_2g,
                adapter_factory=lambda r, p: SyntheticCPTGAdapter(r, p, oom_ranks=(0,)),
            )
            eng_2g = MorphServeClusterEngine(pol_2g, td, td, tmp_root=td / "oom_retry", fleet=fleet_2g)
            try:
                eng_2g.start()
                rep_o = eng_2g.run_suite(oom_suite)
            finally:
                eng_2g.shutdown()
            self.assertEqual(rep_o.completed_sessions, 1)
            self.assertEqual(rep_o.oom_sessions, 0)
            self.assertGreaterEqual(rep_o.oom_retry_count, 1)

            # 3. 2-GPU `1:1` disaggregated policies (`static_und_ranks == (0,)`): OOM retry never falls back to `gen`-only rank 1!
            for disagg_preset in ("static_nccl_fission", "twostage_shm"):
                pol_dis = replace(
                    EnginePolicyConfig.from_preset(disagg_preset, num_gpus=2),
                    inter_turn_think_s_range=(0.0, 0.0),
                    max_oom_retries=1,
                )
                self.assertEqual(pol_dis.static_und_ranks, (0,))
                self.assertEqual(pol_dis.static_gen_ranks, (1,))
                prep_ranks_seen: list[int] = []

                class _TrackPrepAdapter(SyntheticCPTGAdapter):
                    def prepare_turn(self, ctx_id: str, cmd: dict[str, object]) -> int:
                        prep_ranks_seen.append(self.rank)
                        return super().prepare_turn(ctx_id, cmd)

                fleet_dis = InProcessWorkerFleet(
                    pol_dis,
                    adapter_factory=lambda r, p: _TrackPrepAdapter(r, p, oom_ranks=(0,)),
                )
                eng_dis = MorphServeClusterEngine(
                    pol_dis, td, td, tmp_root=td / f"oom_dis_{disagg_preset}", fleet=fleet_dis
                )
                try:
                    eng_dis.start()
                    self.assertEqual(eng_dis._pick_oom_retry_gpu(0, "ar_only"), 0)
                    self.assertEqual(eng_dis._pick_oom_retry_gpu(1, "dit_t2i"), 0)
                    rep_dis = eng_dis.run_suite(oom_suite)
                finally:
                    eng_dis.shutdown()
                self.assertEqual(rep_dis.completed_sessions, 1)
                self.assertEqual(rep_dis.oom_sessions, 0)
                self.assertGreaterEqual(rep_dis.oom_retry_count, 1)
                self.assertEqual(prep_ranks_seen, [0, 0])

                # 4. Persistent mid-turn OOM on 2-GPU `1:1` split (`max_oom_retries=1`): retries on rank 0,
                # terminates cleanly with `oom_sessions == 1` & `oom_retry_count == 1`, and leaks ZERO contexts!
                perm_ranks_seen: list[int] = []
                adapters_by_rank: dict[int, SyntheticCPTGAdapter] = {}

                class _PersistentDecodeOomAdapter(SyntheticCPTGAdapter):
                    def prepare_turn(self, ctx_id: str, cmd: dict[str, object]) -> int:
                        perm_ranks_seen.append(self.rank)
                        return super().prepare_turn(ctx_id, cmd)

                    def decode_ar(self, ctx_id: str, max_tokens: int) -> dict[str, object]:
                        raise MemoryError(f"persistent decode OOM after context alloc on rank {self.rank}")

                def _make_perm_adapter(r: int, p: EnginePolicyConfig) -> _PersistentDecodeOomAdapter:
                    inst = _PersistentDecodeOomAdapter(r, p)
                    adapters_by_rank[r] = inst
                    return inst

                fleet_perm = InProcessWorkerFleet(pol_dis, adapter_factory=_make_perm_adapter)
                eng_perm = MorphServeClusterEngine(
                    pol_dis, td, td, tmp_root=td / f"oom_perm_{disagg_preset}", fleet=fleet_perm
                )
                try:
                    eng_perm.start()
                    rep_perm = eng_perm.run_suite(oom_suite)
                    leaked_ctxs = {r: dict(ad._contexts) for r, ad in adapters_by_rank.items()}
                finally:
                    eng_perm.shutdown()
                self.assertEqual(rep_perm.completed_sessions, 0)
                self.assertEqual(rep_perm.oom_sessions, 1)
                self.assertEqual(rep_perm.oom_retry_count, 1)
                self.assertEqual(perm_ranks_seen, [0, 0])
                self.assertEqual(leaked_ctxs, {0: {}, 1: {}})

    def test_internvlu_40gb_default_vs_12gb_capped_pool_presets(self) -> None:
        # Decision 5/6: 40GB unconstrained InternVL-U avoids cramped 4,864 MiB false evictions
        ivlu_40g = EnginePolicyConfig.from_preset("morphserve", num_gpus=8, model_family="internvl_u")
        self.assertEqual(ivlu_40g.super_pool_mib, 12288)
        self.assertTrue(ivlu_40g.allow_vmm_overflow)

        # 12GiB capped Regime B (`gpu_memory_fraction <= 0.5`) tightens to 4,864 MiB with strict VMM cap
        ivlu_12g = EnginePolicyConfig.from_preset(
            "morphserve",
            num_gpus=8,
            model_family="internvl_u",
            gpu_memory_fraction=0.30,
        )
        self.assertEqual(ivlu_12g.super_pool_mib, 4864)
        self.assertFalse(ivlu_12g.allow_vmm_overflow)

        suite = _build_mini_interleaved_suite()
        with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmp:
            td = Path(tmp)
            pol = replace(
                EnginePolicyConfig.from_preset("morphserve", num_gpus=4, model_family="internvl_u"),
                inter_turn_think_s_range=(0.0, 0.0),
            )
            eng = MorphServeClusterEngine(pol, td, td, tmp_root=td / "ivlu_40g", fleet=InProcessWorkerFleet(pol))
            try:
                eng.start()
                rep = eng.run_suite(suite)
            finally:
                eng.shutdown()
            self.assertEqual(rep.status, "ok")
            self.assertEqual(rep.vmm_overflow_creates, 0)
            self.assertEqual(rep.vmm_expert_evictions_for_transient, 0)

    def test_ship_nccl_decouples_src_und_lock_from_slow_dst_dit_and_no_deadlock(self) -> None:
        import os
        import threading
        import time

        with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmp:
            td = Path(tmp)
            # 2-GPU `static_nccl_fission`: rank 0 = `und`, rank 1 = `gen` (with slow 0.32s DiT).
            # Session `dit_1` occupies `gen` rank 1 (`0.32s`); session `dit_2` packs on `und` rank 0 & ships to rank 1;
            # sessions `ar_fast_1` and `ar_fast_2` arrive on `und` rank 0 while rank 1 is mid-DiT and MUST NOT block!
            pol = replace(
                EnginePolicyConfig.from_preset("static_nccl_fission", num_gpus=2),
                inter_turn_think_s_range=(0.0, 0.0),
            )
            suite = WorkloadSuite(
                name="lock_decouple_regression",
                trace_source="unit",
                base_qps=50.0,
                sessions=[
                    InterleavedSession(
                        session_id="dit_1",
                        quadrant="highres_t2i",
                        source_dataset="unit",
                        arrival_offset_s=0.00,
                        turns=[InterleavedTurn(turn_idx=0, modality_plan="dit_t2i", prompt="occupy gen", dit_steps=4)],
                    ),
                    InterleavedSession(
                        session_id="dit_2",
                        quadrant="highres_t2i",
                        source_dataset="unit",
                        arrival_offset_s=0.015,
                        turns=[InterleavedTurn(turn_idx=0, modality_plan="dit_t2i", prompt="queued ship", dit_steps=4)],
                    ),
                    InterleavedSession(
                        session_id="ar_fast_1",
                        quadrant="multi_image_vqa",
                        source_dataset="unit",
                        arrival_offset_s=0.035,
                        turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="do not block 1", max_ar_tokens=4)],
                    ),
                    InterleavedSession(
                        session_id="ar_fast_2",
                        quadrant="multi_image_vqa",
                        source_dataset="unit",
                        arrival_offset_s=0.045,
                        turns=[InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="do not block 2", max_ar_tokens=4)],
                    ),
                ],
            )
            fleet = InProcessWorkerFleet(
                pol,
                adapter_factory=lambda r, p: SyntheticCPTGAdapter(r, p, dit_step_sleep_s=0.08 if r == 1 else 0.0),
            )
            eng = MorphServeClusterEngine(pol, td, td, tmp_root=td / "decouple", fleet=fleet)
            try:
                eng.start()
                rep = eng.run_suite(suite)
            finally:
                eng.shutdown()
            self.assertEqual(rep.completed_sessions, 4)
            by_id = {s["session_id"]: s for s in rep.sessions}
            # Both AR sessions on `und` rank 0 finish in << 150ms despite `gen` rank 1 running >320ms DiTs!
            self.assertLess(by_id["ar_fast_1"]["e2e_ms"], 150.0)
            self.assertLess(by_id["ar_fast_2"]["e2e_ms"], 150.0)
            self.assertGreater(by_id["dit_1"]["e2e_ms"], 280.0)

            # Bidirectional concurrent Path-A shipping (`0 -> 1` and `1 -> 0` simultaneously) is 100% deadlock-free!
            bi_pol = replace(
                EnginePolicyConfig.from_preset("morphserve", num_gpus=2, model_family="internvl_u"),
                inter_turn_think_s_range=(0.0, 0.0),
                initial_alphas={0: 0.5, 1: 0.5},
            )
            bi_suite = WorkloadSuite(
                name="bidirectional_nccl",
                trace_source="unit",
                base_qps=100.0,
                sessions=[
                    InterleavedSession(
                        session_id=f"bi_{i}",
                        quadrant="visual_cot",
                        source_dataset="unit",
                        arrival_offset_s=0.0,
                        turns=[InterleavedTurn(turn_idx=0, modality_plan="cot_then_dit", prompt=f"bi {i}", max_ar_tokens=4, dit_steps=2)],
                    )
                    for i in range(8)
                ],
            )
            bi_eng = MorphServeClusterEngine(bi_pol, td, td, tmp_root=td / "bi", fleet=InProcessWorkerFleet(bi_pol))
            t_bi = time.perf_counter()
            try:
                bi_eng.start()
                # Perturb worker alphas symmetrically to trigger concurrent cross-ships in both directions
                bi_eng.worker_states[0].expert_alpha = 0.25
                bi_eng.worker_states[1].expert_alpha = 1.0
                bi_rep = bi_eng.run_suite(bi_suite)
            finally:
                bi_eng.shutdown()
            self.assertEqual(bi_rep.completed_sessions, 8)
            self.assertLess(time.perf_counter() - t_bi, 5.0)

            # Edge case: `ALLOC_RECV_BUF` OOM on `dst_gpu` drops `src_gpu`'s packed buffer cleanly without hanging `SEND_NCCL`!
            oom_ship_pol = replace(
                EnginePolicyConfig.from_preset("static_nccl_fission", num_gpus=2),
                inter_turn_think_s_range=(0.0, 0.0),
                max_oom_retries=0,
            )
            oom_ship_fleet = InProcessWorkerFleet(oom_ship_pol)
            oom_ship_eng = MorphServeClusterEngine(
                oom_ship_pol, td, td, tmp_root=td / "oom_ship", fleet=oom_ship_fleet
            )
            try:
                oom_ship_eng.start()

                def _fail_alloc(_nb: int):
                    raise MemoryError("simulated out of memory in alloc_recv_u8")

                oom_ship_fleet._runtimes[1].peer_mesh.alloc_recv_u8 = _fail_alloc
                oom_ship_rep = oom_ship_eng.run_suite(
                    WorkloadSuite(
                        name="alloc_recv_oom_edge",
                        trace_source="unit",
                        base_qps=10.0,
                        sessions=[
                            InterleavedSession(
                                session_id="s_alloc_oom",
                                quadrant="highres_t2i",
                                source_dataset="unit",
                                arrival_offset_s=0.0,
                                turns=[InterleavedTurn(turn_idx=0, modality_plan="dit_t2i", prompt="oom recv", dit_steps=2)],
                            )
                        ],
                    )
                )
                self.assertEqual(oom_ship_rep.oom_sessions, 1)
                self.assertEqual(oom_ship_rep.error_sessions, 0)
                # Verify `DROP_PACKED_CTX` emptied `src_gpu`'s `_packed_store` completely!
                self.assertEqual(len(oom_ship_fleet._runtimes[0]._packed_store), 0)
            finally:
                oom_ship_eng.shutdown()

            # Edge case 2: Optimistic mid-DiT `ALLOC_RECV_BUF(probe_only=True)` transient failure + post-DiT retry
            # succeeds with `oom_sessions == 0` AND `worker_oom_count == 0`, and `PACK_CTX(release_after_pack=True)`
            # synchronizes `ready_ev` BEFORE `release_context` unmaps VMM VA pages!
            event_log: list[str] = []

            class _FakeCudaPayload(bytes):
                is_cuda = True
                device = "cpu"

            class _SyncCheckAdapter(SyntheticCPTGAdapter):
                def pack_state(self, ctx_id: str, include_cfg_dup: bool = False):
                    st = super().pack_state(ctx_id, include_cfg_dup=include_cfg_dup)
                    from morphserve.models.base import PackedTransferState

                    return PackedTransferState(
                        meta=st.meta,
                        payload=_FakeCudaPayload(st.payload),
                        logical_bytes=st.logical_bytes,
                    )

                def release_context(self, ctx_id: str) -> None:
                    event_log.append(f"release:{ctx_id}")
                    super().release_context(ctx_id)

            import morphserve.engine.worker_runtime as wr_mod

            orig_rec_ev = wr_mod._record_cuda_event_if_tensor

            class _FakeEvent:
                def synchronize(self) -> None:
                    event_log.append("sync_ready_ev")

            wr_mod._record_cuda_event_if_tensor = (
                lambda p: _FakeEvent() if getattr(p, "is_cuda", False) else None
            )
            try:
                probe_pol = replace(
                    EnginePolicyConfig.from_preset("static_nccl_fission", num_gpus=2),
                    inter_turn_think_s_range=(0.0, 0.0),
                )
                probe_fleet = InProcessWorkerFleet(
                    probe_pol,
                    adapter_factory=lambda r, p: _SyncCheckAdapter(r, p),
                )
                probe_eng = MorphServeClusterEngine(
                    probe_pol, td, td, tmp_root=td / "probe_retry", fleet=probe_fleet
                )
                try:
                    probe_eng.start()
                    orig_alloc = probe_fleet._runtimes[1].peer_mesh.alloc_recv_u8
                    alloc_calls = {"n": 0}

                    def _flaky_first_alloc(nb: int):
                        alloc_calls["n"] += 1
                        if alloc_calls["n"] == 1:
                            raise MemoryError("transient mid-DiT VAE peak")
                        return orig_alloc(nb)

                    probe_fleet._runtimes[1].peer_mesh.alloc_recv_u8 = _flaky_first_alloc
                    probe_rep = probe_eng.run_suite(
                        WorkloadSuite(
                            name="probe_retry_ok",
                            trace_source="unit",
                            base_qps=10.0,
                            sessions=[
                                InterleavedSession(
                                    session_id="s_probe_ok",
                                    quadrant="highres_t2i",
                                    source_dataset="unit",
                                    arrival_offset_s=0.0,
                                    turns=[InterleavedTurn(turn_idx=0, modality_plan="dit_t2i", prompt="ok", dit_steps=2)],
                                )
                            ],
                        )
                    )
                    self.assertEqual(probe_rep.completed_sessions, 1)
                    self.assertEqual(probe_rep.oom_sessions, 0)
                    self.assertEqual(probe_rep.worker_oom_count, 0)
                    sync_idx = event_log.index("sync_ready_ev")
                    rel_idx = event_log.index("release:ctx_s_probe_ok")
                    self.assertLess(sync_idx, rel_idx)
                finally:
                    probe_eng.shutdown()
            finally:
                wr_mod._record_cuda_event_if_tensor = orig_rec_ev
            _ = (os, threading)

    def test_multiturn_kv_continuity_and_static_fission_vs_shm_fairness(self) -> None:
        with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmp:
            td = Path(tmp)
            suite = _build_mini_interleaved_suite()

            # Track deferred `pending_vae` encoding on `gen` ranks, post-DiT `cfg_text` cleanup, and
            # `prev_image_path=None` in-memory loopback preservation (`GpuInPlaceLoopback` when `save_outputs_dir` set!)
            class _RealisticTightMotAdapter(SyntheticCPTGAdapter):
                def __init__(self, rank: int, policy: EnginePolicyConfig) -> None:
                    super().__init__(rank, policy)
                    self.role = policy.role_of(rank)
                    self.tokens_by_ctx: dict[str, list[str]] = {}
                    self.pending_vae_by_ctx: dict[str, list[str]] = {}
                    self.live_cfg_dup_by_ctx: dict[str, bool] = {}
                    self.saw_prev_image_paths: list[object] = []

                def prepare_turn(self, ctx_id: str, cmd: dict[str, object]) -> int:
                    lb = super().prepare_turn(ctx_id, cmd)
                    toks = self.tokens_by_ctx.setdefault(ctx_id, [])
                    pend = self.pending_vae_by_ctx.setdefault(ctx_id, [])
                    can_vae = bool(cmd.get("prepare_for_dit")) and not (
                        self.policy.routing_mode == "static_nccl_fission" and self.role == "und"
                    )
                    if cmd.get("use_prev_image"):
                        self.saw_prev_image_paths.append(cmd.get("prev_image_path"))
                        ent = self._contexts[ctx_id]
                        assert ent.last_image_tag is not None or cmd.get("prev_image_path") is not None
                        img_id = str(ent.last_image_tag or cmd.get("prev_image_path"))
                        if can_vae:
                            toks.append(f"vae:{img_id}")
                        else:
                            pend.append(img_id)
                        toks.append(f"vit:{img_id}")
                    toks.append(f"txt:{cmd.get('prompt')}")
                    if cmd.get("prepare_for_dit"):
                        self.live_cfg_dup_by_ctx[ctx_id] = True
                    return lb

                def generate_dit(self, ctx_id: str, cmd: dict[str, object], **kw):
                    toks = self.tokens_by_ctx.setdefault(ctx_id, [])
                    for img_id in self.pending_vae_by_ctx.pop(ctx_id, []):
                        toks.append(f"vae:{img_id}")
                    out = super().generate_dit(ctx_id, cmd, **kw)
                    self.live_cfg_dup_by_ctx[ctx_id] = False
                    return out

                def pack_state(self, ctx_id: str, include_cfg_dup: bool = False):
                    st = super().pack_state(ctx_id, include_cfg_dup=include_cfg_dup)
                    meta = dict(
                        st.meta,
                        toks=list(self.tokens_by_ctx.get(ctx_id, [])),
                        pend=list(self.pending_vae_by_ctx.get(ctx_id, [])),
                    )
                    from morphserve.models.base import PackedTransferState

                    return PackedTransferState(meta=meta, payload=st.payload, logical_bytes=st.logical_bytes)

                def unpack_state(self, new_ctx_id: str, meta: dict[str, object], raw_payload: object) -> int:
                    b = super().unpack_state(new_ctx_id, meta, raw_payload)
                    self.tokens_by_ctx[new_ctx_id] = list(meta.get("toks", []))
                    self.pending_vae_by_ctx[new_ctx_id] = list(meta.get("pend", []))
                    self.live_cfg_dup_by_ctx[new_ctx_id] = False
                    return b

                def release_context(self, ctx_id: str) -> None:
                    self.tokens_by_ctx.pop(ctx_id, None)
                    self.pending_vae_by_ctx.pop(ctx_id, None)
                    self.live_cfg_dup_by_ctx.pop(ctx_id, None)
                    super().release_context(ctx_id)

            fis_pol = replace(
                EnginePolicyConfig.from_preset("static_nccl_fission", num_gpus=4),
                inter_turn_think_s_range=(0.0, 0.0),
            )
            shm_pol = replace(
                EnginePolicyConfig.from_preset("twostage_shm", num_gpus=4),
                inter_turn_think_s_range=(0.0, 0.0),
            )

            fis_adapters: dict[int, _RealisticTightMotAdapter] = {}

            def _make_fis(r: int, p: EnginePolicyConfig) -> _RealisticTightMotAdapter:
                ad = _RealisticTightMotAdapter(r, p)
                fis_adapters[r] = ad
                return ad

            fis_eng = MorphServeClusterEngine(
                fis_pol, td, td, tmp_root=td / "fis", fleet=InProcessWorkerFleet(fis_pol, adapter_factory=_make_fis)
            )
            final_edit_toks: list[str] = []
            orig_gen_dit = _RealisticTightMotAdapter.generate_dit

            def _spy_gen_dit(ad_self: _RealisticTightMotAdapter, cid: str, cmd: dict[str, object], **kw):
                res = orig_gen_dit(ad_self, cid, cmd, **kw)
                if "s_edit" in cid and cmd.get("prompt") == "add snow peaks":
                    final_edit_toks.extend(ad_self.tokens_by_ctx.get(cid, []))
                return res

            _RealisticTightMotAdapter.generate_dit = _spy_gen_dit  # type: ignore[method-assign]
            try:
                fis_eng.start()
                init_kv_free = fis_eng.worker_states[0].kv_free_bytes
                fis_rep = fis_eng.run_suite(suite)
                # After all sessions complete and release contexts, `kv_free_bytes` returns to full capacity!
                self.assertEqual(fis_eng.worker_states[0].kv_free_bytes, init_kv_free)
                for ad in fis_adapters.values():
                    self.assertEqual(len(ad.tokens_by_ctx), 0)
            finally:
                _RealisticTightMotAdapter.generate_dit = orig_gen_dit  # type: ignore[method-assign]
                fis_eng.shutdown()

            # All 3 historical edit images have BOTH `vit:` and `vae:` KV tokens preserved in exact order (no O(T^2) dup!)
            vae_toks = [t for t in final_edit_toks if t.startswith("vae:")]
            vit_toks = [t for t in final_edit_toks if t.startswith("vit:")]
            self.assertEqual(len(vae_toks), 3)
            self.assertEqual(len(vit_toks), 3)
            self.assertEqual(len(set(vae_toks)), 3)

            shm_eng = MorphServeClusterEngine(shm_pol, td, td, tmp_root=td / "shm", fleet=InProcessWorkerFleet(shm_pol))
            try:
                shm_eng.start()
                shm_rep = shm_eng.run_suite(suite)
            finally:
                shm_eng.shutdown()

            fis_edit = next(s for s in fis_rep.sessions if s["session_id"] == "s_edit")
            shm_edit = next(s for s in shm_rep.sessions if s["session_id"] == "s_edit")
            self.assertEqual(len(fis_edit["ship_bytes_list"]), 4)
            self.assertEqual(len(shm_edit["ship_bytes_list"]), 4)
            # Multi-turn KV history grows strictly across all 4 edit turns (never wiped between turns!)
            for idx in range(1, 4):
                self.assertGreater(fis_edit["ship_bytes_list"][idx], fis_edit["ship_bytes_list"][idx - 1])
                self.assertGreater(shm_edit["ship_bytes_list"][idx], shm_edit["ship_bytes_list"][idx - 1])
            # Only `twostage_shm` incurs stock 2x redundant CFG serialization; `static_nccl_fission` ships 1x fairly!
            self.assertGreater(shm_edit["ship_bytes_list"][3], int(1.8 * fis_edit["ship_bytes_list"][3]))

            # Verify `save_outputs_dir` (Accuracy Suite `D6`) does NOT bypass in-memory `GpuInPlaceLoopback` (`prev_image_path=None`)!
            ms_pol = replace(
                EnginePolicyConfig.from_preset("morphserve", num_gpus=2),
                inter_turn_think_s_range=(0.0, 0.0),
            )
            ms_ads: list[_RealisticTightMotAdapter] = []

            def _make_ms(r: int, p: EnginePolicyConfig) -> _RealisticTightMotAdapter:
                ad = _RealisticTightMotAdapter(r, p)
                ms_ads.append(ad)
                return ad

            ms_eng = MorphServeClusterEngine(
                ms_pol, td, td, tmp_root=td / "ms_loopback", fleet=InProcessWorkerFleet(ms_pol, adapter_factory=_make_ms)
            )
            try:
                ms_eng.start()
                ms_rep = ms_eng.run_suite(suite, save_outputs_dir=td / "saved_eval_imgs")
                self.assertEqual(ms_rep.completed_sessions, 4)
                seen_paths = [p for ad in ms_ads for p in ad.saw_prev_image_paths]
                self.assertEqual(len(seen_paths), 3)
                self.assertTrue(all(p is None for p in seen_paths))
            finally:
                ms_eng.shutdown()

    def test_local_tcp_multinode_2x2_distributed_engine_and_cli_e2e(self) -> None:
        import os
        import threading

        with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmp:
            td = Path(tmp)
            # 1. Live in-process 2-node (`2x2 = 4` ranks) TCP hub + remote Node-1 daemon across 3 policies
            port = find_free_tcp_port()
            daemon_res: dict[str, object] = {}

            def _node1_worker() -> None:
                daemon_res["out"] = run_multinode_worker_daemon(
                    master_addr="127.0.0.1",
                    port=port,
                    node_rank=1,
                    num_nodes=2,
                    gpus_per_node=2,
                    advertise_host="127.0.0.1",
                )

            t_daemon = threading.Thread(target=_node1_worker, daemon=True)
            t_daemon.start()

            topo4 = ClusterTopologyProfile.from_multinode_measurements(
                world_size=4,
                gpus_per_node=2,
                intra_bw=10.6,
                socket_bw=9.55,
                inter_bw=1.15,
            )
            hub = MultinodeCoordinatorHub(
                num_nodes=2,
                gpus_per_node=2,
                bind_host="127.0.0.1",
                port=port,
                advertise_host="127.0.0.1",
                backend="inproc_synthetic",
            )
            suite = _build_mini_interleaved_suite()
            try:
                hub.wait_for_remote_nodes()
                for pol_name in ("morphserve", "static_nccl_fission", "monolithic_dp"):
                    pol = replace(
                        EnginePolicyConfig.from_preset(pol_name, num_gpus=4),
                        inter_turn_think_s_range=(0.0, 0.0),
                    )
                    fleet = TcpMultinodeWorkerFleet(pol, td, td, hub=hub)
                    eng = MorphServeClusterEngine(
                        pol,
                        td,
                        td,
                        tmp_root=td / f"mn_{pol_name}",
                        fleet=fleet,
                        topology=topo4,
                    )
                    try:
                        eng.start()
                        rep = eng.run_suite(suite)
                    finally:
                        eng.shutdown()
                    self.assertEqual(rep.status, "ok")
                    self.assertEqual(rep.completed_sessions, 4)
                    self.assertEqual(rep.error_sessions, 0)

                # Also verify `max_cross_node_ship_bytes` tracks ONLY inter-node ships (not intra-node >500 MiB ships!)
                mix_pol = replace(
                    EnginePolicyConfig.from_preset("morphserve", num_gpus=4),
                    inter_turn_think_s_range=(0.0, 0.0),
                    initial_alphas={0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0},
                )
                topo_fast_inter_short = ClusterTopologyProfile.from_multinode_measurements(
                    world_size=4,
                    gpus_per_node=2,
                    intra_bw=10.6,
                    socket_bw=9.55,
                    inter_bw=5.0,
                )
                mix_fleet = TcpMultinodeWorkerFleet(mix_pol, td, td, hub=hub)
                mix_eng = MorphServeClusterEngine(
                    mix_pol,
                    td,
                    td,
                    tmp_root=td / "mn_mix_check",
                    fleet=mix_fleet,
                    topology=topo_fast_inter_short,
                )
                try:
                    mix_eng.start()
                    # Force GPU 0 (alpha=0, Node 0) -> Turn 0 short 4 MiB cross-node ship to GPU 2 (alpha=1, Node 1),
                    # followed by Turn 1 large 624 MiB intra-node ship GPU 2 (alpha perturbed to 0.25) -> GPU 3 (Node 1)!
                    mix_eng.worker_states[0].expert_alpha = 0.0
                    mix_eng.worker_states[1].expert_alpha = 0.0
                    mix_eng.worker_states[2].expert_alpha = 0.5
                    mix_eng.worker_states[3].expert_alpha = 1.0
                    mix_rep = mix_eng.run_suite(
                        WorkloadSuite(
                            name="cross_vs_intra_ship_bytes",
                            trace_source="unit",
                            base_qps=5.0,
                            sessions=[suite.sessions[1]],
                        )
                    )
                    self.assertEqual(mix_rep.completed_sessions, 1)
                    self.assertLessEqual(mix_rep.max_cross_node_ship_bytes, 128 * (1024**2))
                finally:
                    mix_eng.shutdown()
            finally:
                hub.close()
                t_daemon.join(timeout=5.0)
            self.assertFalse(t_daemon.is_alive())
            self.assertEqual(daemon_res.get("out"), {"status": "ok", "node_rank": 1, "fleets_served": 4})

            # 2. Multi-process `validate_phase2_multinode_2x8.py --mode e2e --synthetic-e2e` across RANK=0 & RANK=1
            base_port = find_free_tcp_port()
            while base_port > 60000:
                base_port = find_free_tcp_port()
            out_dir = td / "cli_2node_e2e"
            cmd = [
                sys.executable,
                "-m",
                "morphserve.bench.validate_phase2_multinode_2x8",
                "--mode",
                "e2e",
                "--synthetic-e2e",
                "--gpus-per-node",
                "2",
                "--num-sessions",
                "8",
                "--qps",
                "20.0",
                "--output-dir",
                str(out_dir),
            ]
            base_env = dict(os.environ, PYTHONPATH=str(SRC_DIR), WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(base_port))
            p0 = subprocess.Popen(cmd, env=dict(base_env, RANK="0"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            p1 = subprocess.Popen(cmd, env=dict(base_env, RANK="1"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out0, err0 = p0.communicate(timeout=45.0)
            out1, err1 = p1.communicate(timeout=45.0)
            self.assertEqual(p0.returncode, 0, msg=f"node0 failed: {out0}\n{err0}")
            self.assertEqual(p1.returncode, 0, msg=f"node1 failed: {out1}\n{err1}")
            rep_json = json.loads((out_dir / "m25-multinode-2x8-e2e-report.json").read_text(encoding="utf-8"))
            self.assertEqual(rep_json["status"], "ok")
            self.assertEqual(rep_json["world_size"], 4)
            self.assertTrue(rep_json["suppressed_cross_node_multigib_kv"])
            self.assertEqual(len(rep_json["results"]), 3)
            for row in rep_json["results"]:
                self.assertEqual(row["completed_sessions"], 8)
                self.assertEqual(row["error_sessions"], 0)

    def test_validate_phase2_engine_smoke_1gpu_and_bench_sources_resolution(self) -> None:
        from unittest import mock
        import morphserve.bench.validate_phase2_engine_smoke as smoke_mod

        with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmp:
            td = Path(tmp)
            bench_parent = td / "data_root"
            bench_root = bench_parent / "interleaved-bench-v1"
            (bench_root / "suites").mkdir(parents=True, exist_ok=True)
            sibling_sources = bench_parent / "bench_sources"
            sibling_sources.mkdir(parents=True, exist_ok=True)
            (sibling_sources / "burstgpt_1_sample.csv").write_text(
                "Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n"
                "5,ChatGPT,100,20,120,Conversation log\n"
                "15,ChatGPT,100,20,120,Conversation log\n"
                "25,ChatGPT,100,20,120,Conversation log\n",
                encoding="utf-8",
            )
            (sibling_sources / "gentd26_lora_request_trace.csv").write_text(
                "gmt_create,predict_type,predict_status,exec_time_seconds,groupId\n"
                "2024-11-15 16:57:50,TXT_2_IMG,SUCCEED,32.0,G0\n"
                "2024-11-15 16:58:10,IMG_2_IMG,SUCCEED,40.0,G0\n"
                "2024-11-15 16:58:50,TXT_2_IMG,SUCCEED,20.0,G0\n",
                encoding="utf-8",
            )
            # Sibling `bench_sources` resolved when `bench_root / "sources"` does not exist
            self.assertEqual(smoke_mod._resolve_sources_dir(bench_root), sibling_sources)

            mini_suite = _build_mini_interleaved_suite()
            mini_suite.save_json(bench_root / "suites/smoke_8gpu_12.json")

            orig_engine_cls = smoke_mod.MorphServeClusterEngine

            def _inproc_engine_factory(policy: EnginePolicyConfig, model_dir: Path, bench_root: Path, tmp_root: Path):
                fast_pol = replace(policy, inter_turn_think_s_range=(0.0, 0.0))
                return orig_engine_cls(
                    policy=fast_pol,
                    model_dir=model_dir,
                    bench_root=bench_root,
                    tmp_root=tmp_root,
                    fleet=InProcessWorkerFleet(fast_pol),
                )

            with mock.patch.object(smoke_mod, "MorphServeClusterEngine", side_effect=_inproc_engine_factory):
                # Pass a single-pass generator for `policies` to verify no iterator-exhaustion bug on `num_gpus=1`
                rep_1g = smoke_mod.run_smoke_validation(
                    model_dir=td,
                    bench_root=bench_root,
                    output_dir=td / "smoke_out_1g",
                    qps=10.0,
                    num_gpus=1,
                    policies=(p for p in VALID_POLICIES),  # type: ignore[arg-type]
                )
                with self.assertRaises(ValueError):
                    smoke_mod.run_smoke_validation(
                        model_dir=td,
                        bench_root=bench_root,
                        output_dir=td / "smoke_out_0g",
                        qps=10.0,
                        num_gpus=0,
                    )
            self.assertEqual(rep_1g["status"], "ok", msg=f"1-GPU smoke failed: {rep_1g}")
            ran_policies_1g = [r["policy_name"] for r in rep_1g["results"]]
            self.assertEqual(
                ran_policies_1g,
                ["morphserve", "monolithic_dp", "prism_tetriserve_coloc", "naive_host_offload"],
            )
            ms_1g = next(r for r in rep_1g["results"] if r["policy_name"] == "morphserve")
            self.assertEqual(ms_1g["path_a_ships"] + ms_1g["path_b_morphs"], 0)
            self.assertGreaterEqual(ms_1g["peak_gpu_reserved_mib"], ms_1g["peak_pytorch_reserved_mib"])

            # Direct `sources` directory inside `bench_root` takes precedence when present
            direct_sources = bench_root / "sources"
            direct_sources.mkdir(parents=True, exist_ok=True)
            self.assertEqual(smoke_mod._resolve_sources_dir(bench_root), direct_sources)

    def test_stage1_1gpu_gate_and_kairos_scripts_behavioral_contracts(self) -> None:
        import os
        import stat

        kairos_common_src = REPO_ROOT.parent / "kairos-toolkit/runtime/kairos_common.sh"
        if not kairos_common_src.is_file():
            self.skipTest("kairos-toolkit/runtime/kairos_common.sh not present alongside repo")

        with tempfile.TemporaryDirectory(dir=_scratch_dir()) as tmp:
            td = Path(tmp)
            shared = td / "shared"
            scripts_rt = shared / "data/MORPHSERVE/scripts"
            models_dir = shared / "data/MORPHSERVE/models"
            bench_dir = shared / "data/MORPHSERVE/benchmarks/interleaved-bench-v1"
            vendor_dir = shared / "data/MORPHSERVE/vendor/internvlu_pkg/internvlu"
            eval_dir = shared / "data/MORPHSERVE/eval_models"
            build_sha = "deadbee"
            build_dir = shared / f"code/MORPHSERVE/builds/{build_sha}"
            src_stage = shared / f"code/MORPHSERVE/builds/.src-{build_sha}"

            for p in (
                scripts_rt,
                models_dir / "BAGEL-7B-MoT",
                models_dir / "InternVL-U",
                bench_dir,
                vendor_dir,
                eval_dir / "owlv2-base-patch16-ensemble",
                build_dir / "data",
                build_dir / "build",
                src_stage / "tests",
                src_stage / "deploy",
            ):
                p.mkdir(parents=True, exist_ok=True)

            (models_dir / "BAGEL-7B-MoT/MODEL_MANIFEST.json").write_text("{}\n", encoding="utf-8")
            (models_dir / "InternVL-U/MODEL_MANIFEST.json").write_text("{}\n", encoding="utf-8")
            (bench_dir / "BENCH_MANIFEST.json").write_text("{}\n", encoding="utf-8")
            # Pre-stage OWLv2, leave CLIP unstaged so Step (b)'s `snapshot_download` path is exercised!
            (eval_dir / "owlv2-base-patch16-ensemble/config.json").write_text("{}\n", encoding="utf-8")
            (eval_dir / "owlv2-base-patch16-ensemble/model.safetensors").write_bytes(b"weights")

            (build_dir / "BUILD_INFO.json").write_text(
                json.dumps({"commit": build_sha * 5, "short_commit": build_sha}) + "\n",
                encoding="utf-8",
            )
            import shutil

            shutil.copytree(
                SRC_DIR,
                build_dir / "src",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
            shutil.copytree(
                REPO_ROOT / "results",
                build_dir / "results",
                ignore=shutil.ignore_patterns("*.log", "*.nsys-rep", "*.sqlite", "vllm_omni_ref", "raw"),
            )
            (src_stage / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
            (src_stage / "tests/test_stage1_ws_contract.py").write_text(
                "import os\n"
                "from pathlib import Path\n"
                "import unittest\n"
                "import morphserve\n"
                "from morphserve.sim import MeasuredPhysicsProfile\n"
                "class TestStage1WsContract(unittest.TestCase):\n"
                "    def test_ws_local_scratch_and_bench_link(self) -> None:\n"
                "        root = Path(__file__).resolve().parents[1]\n"
                "        scratch = root / 'build' / 'pytest_tmp'\n"
                "        scratch.mkdir(parents=True, exist_ok=True)\n"
                "        (scratch / 'probe.txt').write_text('ok', encoding='utf-8')\n"
                "        self.assertTrue((root / 'data/interleaved-bench-v1/BENCH_MANIFEST.json').is_file())\n"
                "        ms_file = Path(morphserve.__file__).resolve()\n"
                "        self.assertTrue(ms_file.is_file())\n"
                "        self.assertTrue(ms_file.is_relative_to(Path(os.environ['MORPHSERVE_BIN_DIR']).resolve().parent))\n"
                "        self.assertEqual(len(MeasuredPhysicsProfile.from_phase01_results().provenance_files), 7)\n",
                encoding="utf-8",
            )
            # Lock `build_dir` read-only (`0555`) to simulate NFS `root_squash` on immutable Kairos builds!
            for ro_p in (build_dir / "data", build_dir / "build", build_dir):
                os.chmod(ro_p, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

            try:
                # Populate runtime helper scripts under mocked `/shared/data/MORPHSERVE/scripts`
                (scripts_rt / "kairos_common.sh").write_text(
                    kairos_common_src.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                ms_common_txt = (REPO_ROOT / "deploy/kairos/scripts/morphserve_common.sh").read_text(encoding="utf-8")
                (scripts_rt / "morphserve_common.sh").write_text(
                    ms_common_txt.replace("/shared/data/MORPHSERVE/scripts", str(scripts_rt)),
                    encoding="utf-8",
                )
                (scripts_rt / "kairos_runtime_audit.py").write_text(
                    "import argparse, json, signal, time\n"
                    "from pathlib import Path\n"
                    "ap = argparse.ArgumentParser()\n"
                    "ap.add_argument('--output', type=Path)\n"
                    "ap.add_argument('--model-dir', action='append', default=[])\n"
                    "ap.add_argument('--memory-summary', type=Path)\n"
                    "args = ap.parse_args()\n"
                    "if args.memory_summary:\n"
                    "    stop = False\n"
                    "    signal.signal(signal.SIGTERM, lambda *_: globals().update(stop=True))\n"
                    "    args.memory_summary.parent.mkdir(parents=True, exist_ok=True)\n"
                    "    args.memory_summary.write_text(json.dumps({'status': 'sampling'}) + '\\n')\n"
                    "    while not stop:\n"
                    "        time.sleep(0.05)\n"
                    "    args.memory_summary.write_text(json.dumps({'status': 'completed'}) + '\\n')\n"
                    "elif args.output:\n"
                    "    args.output.parent.mkdir(parents=True, exist_ok=True)\n"
                    "    args.output.write_text(json.dumps({'status': 'ok'}) + '\\n')\n",
                    encoding="utf-8",
                )

                # Overlay Python site-packages in `${CODE}/venv/lib/python3.12/site-packages` with stubbed
                # `huggingface_hub` + GPU validator shims while keeping REAL `MorphServeClusterEngine` via InProcessWorkerFleet!
                overlay_sp = shared / "code/MORPHSERVE/venv/lib/python3.12/site-packages"
                (overlay_sp / "huggingface_hub").mkdir(parents=True, exist_ok=True)
                (overlay_sp / "huggingface_hub/__init__.py").write_text(
                    "import json, os\n"
                    "from pathlib import Path\n"
                    "def snapshot_download(*, repo_id, local_dir, endpoint=None, ignore_patterns=None):\n"
                    "    assert os.environ.get('HF_HUB_OFFLINE') is None, 'HF_HUB_OFFLINE was not cleared!'\n"
                    "    assert endpoint == 'https://hf-mirror.com'\n"
                    "    assert '*.msgpack' in (ignore_patterns or []) and '*.h5' in (ignore_patterns or [])\n"
                    "    d = Path(local_dir)\n"
                    "    d.mkdir(parents=True, exist_ok=True)\n"
                    "    (d / 'config.json').write_text(json.dumps({'repo': repo_id}) + '\\n')\n"
                    "    (d / 'preprocessor_config.json').write_text('{}\\n')\n"
                    "    (d / 'tokenizer_config.json').write_text('{}\\n')\n"
                    "    (d / 'model.safetensors').write_bytes(b'stub_weights')\n"
                    "    return str(d)\n",
                    encoding="utf-8",
                )
                (overlay_sp / "sitecustomize.py").write_text(
                    "from pathlib import Path\n"
                    "import morphserve.engine.cluster_engine as _ce\n"
                    "from morphserve.engine.worker_runtime import InProcessWorkerFleet\n"
                    "from morphserve.models.base import SyntheticCPTGAdapter\n"
                    "_orig_init = _ce.MorphServeClusterEngine.__init__\n"
                    "def _patched_init(self, policy, model_dir=Path('.'), bench_root=Path('.'), tmp_root=Path('/tmp/ms'), *, fleet=None, topology=None):\n"
                    "    if fleet is None:\n"
                    "        fleet = InProcessWorkerFleet(policy, adapter_factory=lambda r, p: SyntheticCPTGAdapter(r, p, dit_step_sleep_s=0.12))\n"
                    "    _orig_init(self, policy, model_dir=model_dir, bench_root=bench_root, tmp_root=tmp_root, fleet=fleet, topology=topology)\n"
                    "_ce.MorphServeClusterEngine.__init__ = _patched_init\n",
                    encoding="utf-8",
                )

                # Mock `df` and GPU-only `-m morphserve.bench.validate_*` entrypoints in `fake_bin` while delegating
                # Step (b), Step (c) `pytest`, Step (g.2) `MorphServeClusterEngine` smoke, and summary consolidation
                # directly to `sys.executable`!
                fake_bin = td / "fake_bin"
                fake_bin.mkdir(parents=True, exist_ok=True)
                df_script = fake_bin / "df"
                df_script.write_text(
                    "#!/usr/bin/env bash\n"
                    "if [[ \"$*\" == *\"-BG\"* ]]; then\n"
                    "  printf 'Filesystem 1G-blocks Used Available Use%% Mounted on\\n192.168.1.1:/data 1000G 400G 600G 42%% %s\\n' \"${@: -1}\"\n"
                    "else\n"
                    "  printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n192.168.1.1:/data 1048576000 419430400 629145600 42%% %s\\n' \"${@: -1}\"\n"
                    "fi\n",
                    encoding="utf-8",
                )
                os.chmod(df_script, 0o755)
                py_shim = fake_bin / "python3"
                py_shim.write_text(
                    f"#!{sys.executable}\n"
                    "import argparse, json, os, sys\n"
                    "from pathlib import Path\n"
                    "argv = sys.argv[1:]\n"
                    "if len(argv) >= 2 and argv[0] == '-c' and 'import pytest' in argv[1] and os.environ.get('TEST_FORCE_NO_PYTEST') == '1':\n"
                    "    sys.exit(1)\n"
                    "if len(argv) >= 2 and argv[0] == '-m':\n"
                    "    mod, rest = argv[1], argv[2:]\n"
                    "    if mod == 'morphserve.bench.validate_module1_bagel':\n"
                    "        ap = argparse.ArgumentParser(); ap.add_argument('--model-dir'); ap.add_argument('--output-dir', type=Path); ap.add_argument('--steps', type=int)\n"
                    "        a = ap.parse_args(rest); a.output_dir.mkdir(parents=True, exist_ok=True)\n"
                    "        (a.output_dir / 'module1-report.json').write_text(json.dumps({'status': 'ok'}) + '\\n'); sys.exit(0)\n"
                    "    if mod == 'morphserve.bench.validate_module3_cfg_loopback':\n"
                    "        ap = argparse.ArgumentParser(); ap.add_argument('--model-dir'); ap.add_argument('--output-dir', type=Path); ap.add_argument('--steps', type=int); ap.add_argument('--allow-eval-download', action='store_true')\n"
                    "        a = ap.parse_args(rest); assert a.allow_eval_download; a.output_dir.mkdir(parents=True, exist_ok=True)\n"
                    "        owl_be = os.environ.get('TEST_FORCE_OWL_BACKEND', 'transformers_owlv2')\n"
                    "        rep = {'status': 'ok', 'test4_visual_cot_and_downstream': {'owlv2_backend': owl_be, 'clip_backend': 'transformers_clip'}, 'hf_offline': os.environ.get('HF_HUB_OFFLINE')}\n"
                    "        (a.output_dir / 'module3-report.json').write_text(json.dumps(rep) + '\\n'); sys.exit(0)\n"
                    "    if mod == 'morphserve.bench.validate_module4_preemption':\n"
                    "        ap = argparse.ArgumentParser(); ap.add_argument('--model-dir'); ap.add_argument('--output-dir', type=Path); ap.add_argument('--steps', type=int)\n"
                    "        a = ap.parse_args(rest); a.output_dir.mkdir(parents=True, exist_ok=True)\n"
                    "        (a.output_dir / 'module4-report.json').write_text(json.dumps({'status': 'ok'}) + '\\n'); sys.exit(0)\n"
                    "    if mod == 'morphserve.bench.validate_phase2_internvl_u':\n"
                    "        import time\n"
                    "        ap = argparse.ArgumentParser(); ap.add_argument('--model-dir'); ap.add_argument('--vendor-dir'); ap.add_argument('--output-dir', type=Path); ap.add_argument('--num-gpus', type=int); ap.add_argument('--smoke-1gpu', action='store_true')\n"
                    "        a = ap.parse_args(rest); assert (a.num_gpus == 1 and a.smoke_1gpu) or (a.num_gpus == 8 and not a.smoke_1gpu); a.output_dir.mkdir(parents=True, exist_ok=True)\n"
                    "        if a.num_gpus == 8:\n"
                    "            t_dl = time.monotonic() + 5.0\n"
                    "            while not (a.output_dir / 'memory-summary.json').is_file() and time.monotonic() < t_dl:\n"
                    "                time.sleep(0.05)\n"
                    "        (a.output_dir / 'm24-internvlu-report.json').write_text(json.dumps({'status': 'ok', 'morphserve_12gib_quota_verified': True, 'num_gpus': a.num_gpus}) + '\\n'); sys.exit(0)\n"
                    "    if mod == 'morphserve.bench.run_phase2_overnight_deep':\n"
                    "        import time\n"
                    "        out_idx = rest.index('--output-dir') + 1\n"
                    "        od = Path(rest[out_idx]); od.mkdir(parents=True, exist_ok=True)\n"
                    "        t_dl = time.monotonic() + 5.0\n"
                    "        while not (od / 'memory-summary.json').is_file() and time.monotonic() < t_dl:\n"
                    "            time.sleep(0.05)\n"
                    "        (od / 'wave3-cli.json').write_text(json.dumps({'argv': rest}) + '\\n'); sys.exit(0)\n"
                    "    if mod == 'morphserve.bench.run_phase2_bagel_main':\n"
                    "        out_idx = rest.index('--output-dir') + 1\n"
                    "        od = Path(rest[out_idx]); od.mkdir(parents=True, exist_ok=True)\n"
                    "        (od / 'cli-captured.json').write_text(json.dumps({'argv': rest, 'eval_dir': os.environ.get('MORPHSERVE_EVAL_MODELS_DIR'), 'hf_offline': os.environ.get('HF_HUB_OFFLINE')}) + '\\n')\n"
                    "        sys.exit(0)\n"
                    "    if mod == 'morphserve.bench.validate_phase2_multinode_2x8':\n"
                    "        import time\n"
                    "        out_idx = rest.index('--output-dir') + 1\n"
                    "        od = Path(rest[out_idx]); od.mkdir(parents=True, exist_ok=True)\n"
                    "        rk = os.environ.get('RANK', '0')\n"
                    "        if rk == '0':\n"
                    "            t_dl = time.monotonic() + 5.0\n"
                    "            while not (od / 'memory-summary-node0.json').is_file() and time.monotonic() < t_dl:\n"
                    "                time.sleep(0.05)\n"
                    "        (od / f'm25-cli-rank{rk}.json').write_text(json.dumps({'argv': rest, 'rank': rk}) + '\\n')\n"
                    "        print(f'hello_from_rank_{rk}', flush=True)\n"
                    "        sys.exit(0)\n"
                    "if len(argv) >= 1 and argv[0].endswith('ms_stage1_1g_smoke_driver.py'):\n"
                    "    import runpy\n"
                    "    import morphserve.engine.cluster_engine as _ce\n"
                    "    from morphserve.engine.worker_runtime import InProcessWorkerFleet\n"
                    "    from morphserve.models.base import SyntheticCPTGAdapter\n"
                    "    _orig_init = _ce.MorphServeClusterEngine.__init__\n"
                    "    def _patched_init(self, policy, model_dir=Path('.'), bench_root=Path('.'), tmp_root=Path('/tmp/ms'), *, fleet=None, topology=None):\n"
                    "        if fleet is None:\n"
                    "            fleet = InProcessWorkerFleet(policy, adapter_factory=lambda r, p: SyntheticCPTGAdapter(r, p, dit_step_sleep_s=0.22))\n"
                    "        _orig_init(self, policy, model_dir=model_dir, bench_root=bench_root, tmp_root=tmp_root, fleet=fleet, topology=topology)\n"
                    "    _ce.MorphServeClusterEngine.__init__ = _patched_init\n"
                    "    sys.argv = argv\n"
                    "    runpy.run_path(argv[0], run_name='__main__')\n"
                    "    sys.exit(0)\n"
                    f"os.execv({sys.executable!r}, [{sys.executable!r}] + argv)\n",
                    encoding="utf-8",
                )
                os.chmod(py_shim, 0o755)

                stage1_src = (REPO_ROOT / "deploy/kairos/scripts/16_osdi27_stage1_1gpu_gate.sh").read_text(encoding="utf-8")
                stage1_test = td / "16_osdi27_stage1_1gpu_gate.sh"
                stage1_test.write_text(
                    stage1_src.replace("/shared/data/MORPHSERVE/scripts", str(scripts_rt)),
                    encoding="utf-8",
                )
                os.chmod(stage1_test, 0o755)

                uid = str(os.getpid())
                ok_run_id = f"stage1-unit-ok-{uid}"
                env = dict(
                    os.environ,
                    PATH=f"{fake_bin}:{os.environ.get('PATH', '')}",
                    KAIROS_SHARED_ROOT=str(shared),
                    KAIROS_PROJECT="MORPHSERVE",
                    KAIROS_RUN_ID=ok_run_id,
                    KAIROS_BUILD=build_sha,
                    KAIROS_DISABLE_TEE="1",
                )
                proc = subprocess.run(["bash", str(stage1_test)], env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(proc.returncode, 0, msg=f"stage1 gate failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

                run_dir = shared / f"output/MORPHSERVE/runs/{ok_run_id}"
                summary = json.loads((run_dir / "stage1-gate-summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["status"], "ok")
                self.assertEqual(
                    summary["eval_models_staged"],
                    {"owlv2-base-patch16-ensemble": True, "clip-vit-large-patch14": True},
                )
                self.assertEqual(summary["gates"]["module3_cfg_loopback"]["owlv2_backend"], "transformers_owlv2")
                self.assertEqual(summary["gates"]["module3_cfg_loopback"]["clip_backend"], "transformers_clip")
                bagel_gate = summary["gates"]["bagel_1gpu_cluster_smoke"]
                self.assertEqual(bagel_gate["status"], "ok")
                self.assertEqual(bagel_gate["completed_sessions"], 3)
                self.assertEqual(bagel_gate["oom_sessions"], 0)
                self.assertGreaterEqual(bagel_gate["preempted_ar_count"], 2)
                self.assertGreater(bagel_gate["total_yields"], 0)
                self.assertGreater(bagel_gate["peak_gpu_reserved_mib"], bagel_gate["peak_pytorch_reserved_mib"])
                # Verify read-only `build_dir/build` was untouched (zero writes to `/shared` build!)
                self.assertEqual(list((build_dir / "build").iterdir()), [])

                # Negative check: silent `unavailable_offline` evaluator backend in Module 3 MUST fail Stage-1 gate!
                # Also set `TEST_FORCE_NO_PYTEST=1` here to exercise the `unittest discover` fallback + summary parser!
                bad_run_id = f"stage1-unit-bad-eval-{uid}"
                env_bad = dict(
                    env,
                    KAIROS_RUN_ID=bad_run_id,
                    TEST_FORCE_OWL_BACKEND="unavailable_offline",
                    TEST_FORCE_NO_PYTEST="1",
                )
                proc_bad = subprocess.run(["bash", str(stage1_test)], env=env_bad, capture_output=True, text=True, timeout=60)
                self.assertEqual(proc_bad.returncode, 2)
                bad_run_dir = shared / f"output/MORPHSERVE/runs/{bad_run_id}"
                bad_sum = json.loads((bad_run_dir / "stage1-gate-summary.json").read_text(encoding="utf-8"))
                self.assertEqual(bad_sum["status"], "check_failed")
                self.assertEqual(bad_sum["gates"]["pytest_a100"]["status"], "ok")
                self.assertTrue(any(ln.startswith("OK") for ln in bad_sum["gates"]["pytest_a100"]["tail"]))
                self.assertEqual(bad_sum["gates"]["module3_cfg_loopback"]["status"], "check_failed")

                # Exercise Stage-1 consolidation script directly against adversarial `pytest-a100.txt` outputs!
                consolidation_py = stage1_src.split(
                    '# Consolidate Stage-1 gate results into ${RUN_DIR}/stage1-gate-summary.json\npython3 - "${RUN_DIR}" "${EVAL_MODELS_DIR}" <<\'PY\'\n',
                    1,
                )[1].split("\nPY\n", 1)[0]
                adv_dir = td / "adv_consolidate"
                for sub_rel, fname, payload in (
                    ("module1", "module1-report.json", {"status": "ok"}),
                    (
                        "module3",
                        "module3-report.json",
                        {
                            "status": "ok",
                            "test4_visual_cot_and_downstream": {
                                "owlv2_backend": "transformers_owlv2",
                                "clip_backend": "transformers_clip",
                            },
                        },
                    ),
                    ("module4", "module4-report.json", {"status": "ok"}),
                    ("internvlu_1gpu", "m24-internvlu-report.json", {"status": "ok", "morphserve_12gib_quota_verified": True}),
                    (
                        "bagel_1gpu",
                        "bagel-1gpu-smoke.json",
                        {
                            "status": "ok",
                            "completed_sessions": 3,
                            "oom_sessions": 0,
                            "preempted_ar_count": 2,
                            "total_yields": 4,
                        },
                    ),
                ):
                    (adv_dir / sub_rel).mkdir(parents=True, exist_ok=True)
                    (adv_dir / sub_rel / fname).write_text(json.dumps(payload) + "\n", encoding="utf-8")

                for case_name, txt_body, expect_ok in (
                    ("ansi_color_pytest", "\x1b[32m=== \x1b[1m67 passed\x1b[0m\x1b[32m, 5 skipped in 12.3s ===\x1b[0m\n", True),
                    (
                        "pytest_trailing_cpp_dtor",
                        "=== 67 passed, 5 skipped in 12.34s ===\n[W CudaIPCTypes.cpp:16] Producer process terminated\n",
                        True,
                    ),
                    (
                        "unittest_skipped_trailing_dtor",
                        "----------------------------------------------------------------------\n"
                        "Ran 72 tests in 118.0s\n\nOK (skipped=5)\n[W CudaIPCTypes.cpp:16] Producer process terminated\n",
                        True,
                    ),
                    ("pytest_zero_passed", "=== 0 passed in 0.01s ===\n", False),
                    ("pytest_partial_fail", "=== 1 failed, 71 passed in 10.0s ===\n", False),
                    ("pytest_error_banner", "=== 71 passed, 1 error in 10.0s ===\n", False),
                    ("unittest_zero_tests_py311", "Ran 0 tests in 0.000s\n\nOK\n", False),
                    ("unittest_zero_tests_py312", "Ran 0 tests in 0.000s\n\nNO TESTS RAN\n", False),
                    (
                        "unittest_docstring_1_passed_then_fail",
                        "test_x (t.T.test_x)\nVerify 1 passed session ... FAIL\n"
                        "Ran 1 test in 0.01s\n\nFAILED (failures=1)\n",
                        False,
                    ),
                    (
                        "unittest_subproc_ok_outer_fail",
                        "STDOUT:\nRan 1 test in 0.01s\n\nOK\n----------------------------------------------------------------------\n"
                        "Ran 72 tests in 117.0s\n\nFAILED (failures=1, skipped=2)\n",
                        False,
                    ),
                ):
                    (adv_dir / "pytest-a100.txt").write_text(txt_body, encoding="utf-8")
                    p_adv = subprocess.run(
                        [sys.executable, "-c", consolidation_py, str(adv_dir), str(eval_dir)],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    self.assertEqual(p_adv.returncode == 0, expect_ok, msg=f"case={case_name} rc={p_adv.returncode} err={p_adv.stderr}")
                    adv_sum = json.loads((adv_dir / "stage1-gate-summary.json").read_text(encoding="utf-8"))
                    self.assertEqual(
                        adv_sum["gates"]["pytest_a100"]["status"],
                        "ok" if expect_ok else "check_failed",
                        msg=f"case={case_name} summary={adv_sum}",
                    )

                # Verify `05`, `10`, `11`, `12`, `13`, and `14` script runtime flags, sampler cleanup, & multi-node logs!
                for s_name in (
                    "05_phase1_module3_validate.sh",
                    "10_phase2_m23_suite_a.sh",
                    "11_phase2_m23_suite_b.sh",
                    "12_phase2_m24_internvl_u_8gpu.sh",
                    "13_phase2_m25_multinode_2x8.sh",
                    "14_phase2_wave3_deep_8gpu.sh",
                ):
                    raw_sh = (REPO_ROOT / "deploy/kairos/scripts" / s_name).read_text(encoding="utf-8")
                    dst_sh = td / s_name
                    dst_sh.write_text(raw_sh.replace("/shared/data/MORPHSERVE/scripts", str(scripts_rt)), encoding="utf-8")
                    os.chmod(dst_sh, 0o755)

                # Suite A (`10`): default omits `--max-sessions`; setting `MORPHSERVE_MAX_SESSIONS=64` forwards it
                m23a_id = f"m23a-default-{uid}"
                p10 = subprocess.run(
                    ["bash", str(td / "10_phase2_m23_suite_a.sh")],
                    env=dict(env, KAIROS_RUN_ID=m23a_id),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(p10.returncode, 0, msg=p10.stderr)
                c10 = json.loads((shared / f"output/MORPHSERVE/runs/{m23a_id}/cli-captured.json").read_text(encoding="utf-8"))
                self.assertNotIn("--max-sessions", c10["argv"])

                # Suite B (`11`): when `clip` has `config.json` ONLY (missing weights!), `HF_HUB_OFFLINE` is cleared
                (eval_dir / "clip-vit-large-patch14/model.safetensors").unlink()
                m23b_id = f"m23b-partial-eval-{uid}"
                p11 = subprocess.run(
                    ["bash", str(td / "11_phase2_m23_suite_b.sh")],
                    env=dict(env, KAIROS_RUN_ID=m23b_id),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(p11.returncode, 0, msg=p11.stderr)
                c11 = json.loads(
                    (shared / f"output/MORPHSERVE/runs/{m23b_id}/cli-captured.json").read_text(encoding="utf-8")
                )
                self.assertNotIn("--max-sessions", c11["argv"])
                self.assertIn("--allow-eval-download", c11["argv"])
                self.assertEqual(c11["eval_dir"], str(eval_dir))
                self.assertIsNone(c11["hf_offline"])

                # Multi-node 2x8 (`13`): RANK=0 and RANK=1 write disjoint logs & RANK=0 records `memory-summary-node0.json`
                m25_id = f"m25-2x8-logs-{uid}"
                env_m25_base = {k: v for k, v in env.items() if k != "KAIROS_DISABLE_TEE"}
                env_m25_base["KAIROS_RUN_ID"] = m25_id
                p13_r0 = subprocess.run(
                    ["bash", str(td / "13_phase2_m25_multinode_2x8.sh")],
                    env=dict(env_m25_base, RANK="0"),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                p13_r1 = subprocess.run(
                    ["bash", str(td / "13_phase2_m25_multinode_2x8.sh")],
                    env=dict(env_m25_base, RANK="1"),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(p13_r0.returncode, 0, msg=p13_r0.stderr)
                self.assertEqual(p13_r1.returncode, 0, msg=p13_r1.stderr)
                m25_run = shared / f"output/MORPHSERVE/runs/{m25_id}"
                mem_sum = json.loads((m25_run / "memory-summary-node0.json").read_text(encoding="utf-8"))
                self.assertEqual(mem_sum["status"], "completed")
                log_r0 = (m25_run / "13_phase2_m25_multinode_2x8.log").read_text(encoding="utf-8")
                log_r1 = (m25_run / "13_phase2_m25_multinode_2x8-node1.log").read_text(encoding="utf-8")
                self.assertIn("hello_from_rank_0", log_r0)
                self.assertNotIn("hello_from_rank_1", log_r0)
                self.assertIn("hello_from_rank_1", log_r1)
                c13 = json.loads((m25_run / "m25-cli-rank0.json").read_text(encoding="utf-8"))
                self.assertEqual(
                    c13["argv"][:8],
                    [
                        "--mode",
                        "both",
                        "--model-dir",
                        str(models_dir / "BAGEL-7B-MoT"),
                        "--bench-root",
                        str(bench_dir),
                        "--num-sessions",
                        "160",
                    ],
                )

                # InternVL-U (`12`) and Wave-3 Deep (`14`): verify clean sampler SIGTERM/wait and local tmp cleanup
                for s_id, s_file in ((f"m24-8g-{uid}", "12_phase2_m24_internvl_u_8gpu.sh"), (f"w3-8g-{uid}", "14_phase2_wave3_deep_8gpu.sh")):
                    p_sh = subprocess.run(
                        ["bash", str(td / s_file)],
                        env=dict(env, KAIROS_RUN_ID=s_id),
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(p_sh.returncode, 0, msg=f"{s_file} failed:\n{p_sh.stdout}\n{p_sh.stderr}")
                    s_run = shared / f"output/MORPHSERVE/runs/{s_id}"
                    s_mem = json.loads((s_run / "memory-summary.json").read_text(encoding="utf-8"))
                    self.assertEqual(s_mem["status"], "completed")
                    self.assertFalse(Path(f"/tmp/morphserve-{s_id}").exists())
            finally:
                for ro_p in (build_dir, build_dir / "data", build_dir / "build"):
                    if ro_p.exists():
                        os.chmod(ro_p, 0o755)

    def test_phase2_scripts_syntax_and_py_compile(self) -> None:
        import os

        scripts_dir = REPO_ROOT / "deploy/kairos/scripts"
        all_scripts = sorted(scripts_dir.glob("*.sh"))
        self.assertGreaterEqual(len(all_scripts), 18)
        stage1_gate = scripts_dir / "16_osdi27_stage1_1gpu_gate.sh"
        self.assertTrue(stage1_gate.is_file(), "missing 16_osdi27_stage1_1gpu_gate.sh")
        self.assertTrue(os.access(stage1_gate, os.X_OK), "16_osdi27_stage1_1gpu_gate.sh must be executable")
        for script in all_scripts:
            proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=f"bash -n failed on {script.name}: {proc.stderr}")

        for py_rel in (
            "src/morphserve/telemetry.py",
            "src/morphserve/tools/stage_bench_sources.py",
            "src/morphserve/bench/exactness.py",
            "src/morphserve/bench/interleaved_bench.py",
            "src/morphserve/bench/validate_phase2_engine_smoke.py",
            "src/morphserve/engine/__init__.py",
            "src/morphserve/engine/config.py",
            "src/morphserve/engine/worker_runtime.py",
            "src/morphserve/engine/worker_proc.py",
            "src/morphserve/engine/multinode_fleet.py",
            "src/morphserve/engine/cluster_engine.py",
            "src/morphserve/bench/validate_phase2_multinode_2x8.py",
        ):
            p = REPO_ROOT / py_rel
            res = subprocess.run([sys.executable, "-m", "py_compile", str(p)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, msg=f"py_compile failed on {py_rel}: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
