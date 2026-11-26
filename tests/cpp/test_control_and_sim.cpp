#include "morphserve/c_api.h"
#include "morphserve/control/budget_planner.hpp"
#include "morphserve/control/morph_controller.hpp"
#include "morphserve/control/policy.hpp"
#include "morphserve/control/preempt_edf.hpp"
#include "morphserve/control/router.hpp"
#include "morphserve/control/simulator.hpp"
#include "morphserve/control/topology.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#define MS_CHECK(cond)                                                                         \
    do {                                                                                       \
        if (!(cond)) {                                                                         \
            std::cerr << "CHECK FAILED at " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::_Exit(1);                                                                     \
        }                                                                                      \
    } while (0)

namespace {

void test_router_edf_and_budget_planner() {
    using namespace morphserve::control;

    const auto topo = ClusterTopologyProfile::a100_pcie_8gpu_default();
    MS_CHECK(topo.same_half(0, 1));
    MS_CHECK(!topo.same_half(0, 4));

    DualPathMorphRouter router(topo);
    router.init_workers(
        4,
        [](int g) { return (g == 1 || g == 3) ? "gen" : "hybrid"; },
        {{0, 0.5}, {1, 1.0}, {2, 0.5}, {3, 1.0}},
        {{0, 16 * kGiB}, {1, 600 * kMiB}, {2, 16 * kGiB}, {3, 16 * kGiB}});

    CPTGRequestSpec req1;
    req1.req_id = "r1";
    req1.src_gpu = 0;
    req1.architecture = "tight_mot";
    req1.kv_bytes = 400 * kMiB;
    req1.dit_step1_ms = 280.0;
    req1.dit_total_steps = 20;

    const auto res1 = router.plan_and_reserve_transition("dual_path", {}, {}, req1, 50.0);
    MS_CHECK(res1.first.path == "state_ship" && res1.first.dst_gpu == 1 && res1.second == "nccl_dual");
    MS_CHECK(router.get_worker_state(1).kv_free_bytes == 200 * kMiB);

    // Stale telemetry before unpack must preserve 400 MiB inflight reservation
    router.update_worker_telemetry(1, 1.0, 600 * kMiB, 5);
    MS_CHECK(router.get_worker_state(1).kv_free_bytes == 200 * kMiB);
    router.update_worker_telemetry(1, 0.25, 16 * kGiB, 4);  // stale seq 4 < 5 ignored!
    MS_CHECK(router.get_worker_state(1).kv_free_bytes == 200 * kMiB);

    // PreemptEdfController adaptive chunking & EDF slack coalescing (window = 2*20 + 8.5 + 4 = 52.5 ms)
    PreemptEdfController edf(28, 60.0, 4.0, 8.5, 20.0, 3);
    MS_CHECK(edf.compute_adaptive_chunk_layers(60.0) == 4);
    MS_CHECK(edf.compute_adaptive_chunk_layers(100.0) == 8);
    const double dls[4] = {100.032, 100.015, 100.50, 100.025};
    std::vector<size_t> due;
    std::vector<size_t> def;
    edf.select_edf_microbatch(100.00, dls, 4, /*force_all_urgent=*/false, due, def);
    MS_CHECK(due.size() == 3);
    MS_CHECK(due[0] == 1 && due[1] == 3 && due[2] == 0);
    MS_CHECK(def.size() == 1 && def[0] == 2);

    // Anti-phase lock prevention at 100 ms TPOT SLO (50 ms AR step): 110 ms slack peer MUST coalesce into B=2!
    PreemptEdfController edf_bagel(28, 100.0, 14.0, 10.5, 50.0, 4);
    const double dls_antiphase[3] = {100.040, 100.110, 100.500};
    edf_bagel.select_edf_microbatch(100.00, dls_antiphase, 3, /*force_all_urgent=*/false, due, def);
    MS_CHECK(due.size() == 2 && due[0] == 0 && due[1] == 1);
    MS_CHECK(def.size() == 1 && def[0] == 2);

    // Architecture-aware initial placement:
    // - `bridge_mllm_dit` (InternVL-U) prefers LOW decoder alpha (VLM-resident GPU) for `dit_t2i` & `cot_then_dit`
    // - `tight_mot` (BAGEL) prefers HIGH gen alpha
    // - `ar_only` preemption prefers non-DiT busy GPUs over GPUs actively executing DiT
    DualPathMorphRouter router_arch(topo);
    router_arch.init_workers(
        2,
        [](int) { return "hybrid"; },
        {{0, 0.0}, {1, 1.0}},
        {{0, 16 * kGiB}, {1, 16 * kGiB}});
    MS_CHECK(
        router_arch.select_initial_gpu(
            2, "dual_path", "layer_chunk", {}, "dit_t2i", nullptr, {{0, false}, {1, false}}, {}, {}, 2, "bridge_mllm_dit") ==
        0);
    MS_CHECK(
        router_arch.select_initial_gpu(
            2, "dual_path", "layer_chunk", {}, "dit_t2i", nullptr, {{0, false}, {1, false}}, {}, {}, 2, "tight_mot") ==
        1);
    MS_CHECK(
        router_arch.select_initial_gpu(
            2, "dual_path", "layer_chunk", {}, "cot_then_dit", nullptr, {{0, false}, {1, false}}, {}, {}, 2, "bridge_mllm_dit") ==
        0);
    MS_CHECK(
        router_arch.select_initial_gpu(
            2,
            "dual_path",
            "layer_chunk",
            {},
            "ar_only",
            nullptr,
            {{0, true}, {1, true}},
            {{0, 1}, {1, 0}},
            {{0, 0}, {1, 0}},
            2,
            "tight_mot") == 1);

    // WorkerBudgetPlanner 12 GiB Regime B conservation (VLM + Decoder co-paged in 6688 MiB VMM pool)
    const auto q_pair = WorkerBudgetPlanner::compute_internvlu_quota_overrides(
        "morphserve", 8, "/shared/data/MORPHSERVE/models/InternVL-U", 12.0, 40.0);
    MS_CHECK(q_pair.second.at("runtime_combined_budget_mib").get<double>() == 12288.0);
    const auto pol = PolicyConfig::from_preset("morphserve", 8, q_pair.first);
    const auto plan = WorkerBudgetPlanner::compute_internvlu_memory_plan(
        40960.0,
        35000.0,
        260.0,
        4864LL * kMiB,
        pol.super_pool_mib,
        pol.fine_kv_pool_mib,
        pol.super_handle_mib,
        pol.allow_vmm_overflow,
        pol.gpu_memory_fraction);
    MS_CHECK(plan.at("quota_constrained").get<bool>());
    MS_CHECK(plan.at("vmm_physical_pool_mib").get<int>() == 6688);
    MS_CHECK(plan.at("pytorch_cap_mib").get<double>() + plan.at("vmm_physical_pool_mib").get<double>() <= 12288.0);
}

void test_physics_ledger_and_simulator() {
    using namespace morphserve::control;

    const auto ph = ClusterPhysicsProfile::from_phase01_results("results");
    MS_CHECK(ph.provenance_files.size() == 7);

    const auto ms_pol = PolicyConfig::from_preset("morphserve", 8);
    const auto mono_pol = PolicyConfig::from_preset("monolithic_dp", 8);
    SimGpuMemoryLedger ms_ledger(0, 40960.0, ms_pol, "hybrid");
    SimGpuMemoryLedger mono_ledger(0, 40960.0, mono_pol, "hybrid");

    for (int sid = 0; sid < 4; ++sid) {
        MS_CHECK(ms_ledger.upsert_session_kv(sid, 1800.0));
        mono_ledger.upsert_session_kv(sid, 1800.0);
    }
    MS_CHECK(ms_ledger.execute_dit_and_vae(1024, 0).first);
    MS_CHECK(ms_ledger.expert_evictions() > 0);
    MS_CHECK(!mono_ledger.execute_dit_and_vae(1024, 0).first);

    HardwareInterconnectSpec hw_xnode{
        "unit_2node_2gpu", 40.0, 1.0, 10.0, 10.0, 10.0, 10.0, 5.0, 1};
    // Explicit 4-turn `multiturn_edit` session verifies 4 forward ships + 3 return ships = 7 cross-node ships
    // independently of `main_mix_80.json` session-0 ordering.
    SimSessionSpec edit_4turn;
    edit_4turn.arrival_ms = 0.0;
    edit_4turn.quadrant = "multiturn_edit";
    edit_4turn.session_id = "unit_edit_4turn";
    edit_4turn.turns = {
        SimTurnSpec{"dit_edit", 512, 20, 48, 1},
        SimTurnSpec{"dit_edit", 512, 20, 48, 0},
        SimTurnSpec{"dit_edit", 512, 20, 48, 0},
        SimTurnSpec{"dit_edit", 512, 20, 48, 0},
    };
    const std::vector<SimSessionSpec> custom_edit{edit_4turn};
    const auto r_nccl = ClusterSimulator::simulate_cluster_point(
        "static_nccl_fission",
        2,
        hw_xnode,
        0.5,
        1,
        2026,
        ph,
        WorkloadScaleSpec::bagel_7b_standard(),
        "",
        "data/interleaved-bench-v1",
        "data/bench_sources",
        &custom_edit);
    MS_CHECK(r_nccl.at("oom_sessions").get<int>() == 0);
    MS_CHECK(r_nccl.at("path_a_ships").get<int>() == 4);
    MS_CHECK(r_nccl.at("cross_node_ships").get<int>() == 7);
}

void test_v2_policy_think_slo_and_steady_window() {
    using namespace morphserve::control;

    // 1. PolicyConfig::from_preset v2 defaults across all presets and synthetic preset
    for (const char* pname : kValidPolicies) {
        const auto p = PolicyConfig::from_preset(pname, 8);
        MS_CHECK(p.allow_cascade_cfg_split == false);
        MS_CHECK(std::abs(p.inter_turn_think_s_range.first - 2.0) < 1e-9);
        MS_CHECK(std::abs(p.inter_turn_think_s_range.second - 25.0) < 1e-9);
        MS_CHECK(p.think_time_dist == "lognormal");
        MS_CHECK(std::abs(p.think_time_median_s - 6.0) < 1e-9);
        MS_CHECK(std::abs(p.think_time_sigma - 0.75) < 1e-9);
        MS_CHECK(p.ar_tpot_slo_mode == "turn_p95");
        MS_CHECK(p.steady_window_mode == "time");
        MS_CHECK(std::abs(p.warmup_window_s - 60.0) < 1e-9);
        MS_CHECK(std::abs(p.steady_window_s - 180.0) < 1e-9);
    }

    const auto syn = PolicyConfig::from_preset("morphserve", 4, {{"model_family", "synthetic"}});
    MS_CHECK(syn.allow_cascade_cfg_split == false);
    MS_CHECK(std::abs(syn.inter_turn_think_s_range.first - 0.002) < 1e-9);
    MS_CHECK(std::abs(syn.inter_turn_think_s_range.second - 0.005) < 1e-9);
    MS_CHECK(syn.think_time_dist == "uniform");
    MS_CHECK(std::abs(syn.think_time_median_s - 0.003) < 1e-9);

    // JSON round-trip preserves all v2 fields
    const auto custom_pol = PolicyConfig::from_preset(
        "morphserve",
        8,
        {
            {"allow_cascade_cfg_split", true},
            {"think_time_dist", "constant"},
            {"think_time_median_s", 15.0},
            {"think_time_sigma", 0.5},
            {"ar_tpot_slo_mode", "token_max"},
            {"steady_window_mode", "session_trim"},
            {"warmup_window_s", 30.0},
            {"steady_window_s", 120.0},
        });
    const auto roundtrip = PolicyConfig::from_json(custom_pol.to_json());
    MS_CHECK(roundtrip.allow_cascade_cfg_split == true);
    MS_CHECK(roundtrip.think_time_dist == "constant");
    MS_CHECK(std::abs(roundtrip.think_time_median_s - 15.0) < 1e-9);
    MS_CHECK(std::abs(roundtrip.think_time_sigma - 0.5) < 1e-9);
    MS_CHECK(roundtrip.ar_tpot_slo_mode == "token_max");
    MS_CHECK(roundtrip.steady_window_mode == "session_trim");
    MS_CHECK(std::abs(roundtrip.warmup_window_s - 30.0) < 1e-9);
    MS_CHECK(std::abs(roundtrip.steady_window_s - 120.0) < 1e-9);

    // 2. deterministic_think_dist_s: lognormal, uniform, constant, and (0.0, 0.0) -> 0.0
    double min_ln = 1e9;
    double max_ln = -1e9;
    std::set<int> distinct_ln_tenths;
    for (int i = 0; i < 40; ++i) {
        const std::string sid = "sess_v2_" + std::to_string(i);
        const double v1 = MorphControlEngine::deterministic_think_dist_s(sid, 1, 2.0, 25.0, "lognormal", 6.0, 0.75);
        const double v2 = MorphControlEngine::deterministic_think_dist_s(sid, 1, 2.0, 25.0, "lognormal", 6.0, 0.75);
        const double vc = ms_ctrl_deterministic_think_dist_s(sid.c_str(), 1, 2.0, 25.0, "lognormal", 6.0, 0.75);
        MS_CHECK(v1 == v2);
        MS_CHECK(std::abs(v1 - vc) < 1e-12);
        MS_CHECK(v1 >= 2.0 && v1 <= 25.0);
        min_ln = std::min(min_ln, v1);
        max_ln = std::max(max_ln, v1);
        distinct_ln_tenths.insert(static_cast<int>(std::round(v1 * 10.0)));
    }
    MS_CHECK((max_ln - min_ln) > 2.0);
    MS_CHECK(distinct_ln_tenths.size() >= 10);

    const double u1 = MorphControlEngine::deterministic_think_dist_s("sess_u", 2, 2.0, 25.0, "uniform", 6.0, 0.75);
    const double u2 = MorphControlEngine::deterministic_think_s("sess_u", 2, 2.0, 25.0);
    MS_CHECK(std::abs(u1 - u2) < 1e-12);
    MS_CHECK(u1 >= 2.0 && u1 <= 25.0);

    const double c1 = MorphControlEngine::deterministic_think_dist_s("sess_c", 1, 2.0, 25.0, "constant", 6.0, 0.75);
    const double c_clamp = MorphControlEngine::deterministic_think_dist_s("sess_c", 1, 2.0, 25.0, "constant", 50.0, 0.75);
    MS_CHECK(std::abs(c1 - 6.0) < 1e-12);
    MS_CHECK(std::abs(c_clamp - 25.0) < 1e-12);

    MS_CHECK(MorphControlEngine::deterministic_think_dist_s("sess_z", 1, 0.0, 0.0, "lognormal", 6.0, 0.75) == 0.0);
    MS_CHECK(MorphControlEngine::deterministic_think_dist_s("sess_z", 1, 0.0, 0.0, "uniform", 6.0, 0.75) == 0.0);
    MS_CHECK(MorphControlEngine::deterministic_think_dist_s("sess_z", 1, 0.0, 0.0, "constant", 6.0, 0.75) == 0.0);

    // 3. evaluate_session_slos: 1 outlier token out of 20 passes under `turn_p95` but fails under `token_max`
    std::vector<double> tpots_1_outlier(19, 50.0);
    tpots_1_outlier.push_back(250.0);  // single outlier > 125.64 ms; interpolated P95 = 60.0 ms <= 125.64 ms
    const std::vector<double> ttfts_ok = {300.0};
    const std::vector<double> dits_ok = {4000.0};
    const std::vector<int> heights_ok = {512};
    const std::vector<int> steps_ok = {20};

    const auto pol_p95 = PolicyConfig::from_preset("morphserve", 8, {{"ar_tpot_slo_mode", "turn_p95"}});
    const auto pol_max = PolicyConfig::from_preset("morphserve", 8, {{"ar_tpot_slo_mode", "token_max"}});
    const auto pol_mean = PolicyConfig::from_preset("morphserve", 8, {{"ar_tpot_slo_mode", "turn_mean"}});

    const auto slo_p95 = MorphControlEngine::evaluate_session_slos(
        pol_p95, ttfts_ok, tpots_1_outlier, dits_ok, heights_ok, steps_ok, {tpots_1_outlier});
    MS_CHECK(slo_p95.at("ar_slo_met").get<bool>() == true);
    MS_CHECK(slo_p95.at("ar_slo_p95_tpot_met").get<bool>() == true);
    MS_CHECK(slo_p95.at("ar_slo_mean_tpot_met").get<bool>() == true);
    MS_CHECK(slo_p95.at("ar_slo_strict_max_tpot_met").get<bool>() == false);
    MS_CHECK(slo_p95.at("joint_slo_met").get<bool>() == true);

    const auto slo_max = MorphControlEngine::evaluate_session_slos(
        pol_max, ttfts_ok, tpots_1_outlier, dits_ok, heights_ok, steps_ok, {tpots_1_outlier});
    MS_CHECK(slo_max.at("ar_slo_met").get<bool>() == false);
    MS_CHECK(slo_max.at("joint_slo_met").get<bool>() == false);

    // 2 outliers out of 20 -> P95 = 250.0 ms > 125.64 ms (fails `turn_p95`), mean = 70.0 ms (passes `turn_mean`)
    std::vector<double> tpots_2_outliers(18, 50.0);
    tpots_2_outliers.push_back(250.0);
    tpots_2_outliers.push_back(250.0);
    const auto slo_p95_fail = MorphControlEngine::evaluate_session_slos(
        pol_p95, ttfts_ok, tpots_2_outliers, dits_ok, heights_ok, steps_ok, {tpots_2_outliers});
    const auto slo_mean_pass = MorphControlEngine::evaluate_session_slos(
        pol_mean, ttfts_ok, tpots_2_outliers, dits_ok, heights_ok, steps_ok, {tpots_2_outliers});
    MS_CHECK(slo_p95_fail.at("ar_slo_met").get<bool>() == false);
    MS_CHECK(slo_mean_pass.at("ar_slo_met").get<bool>() == true);

    // 4. aggregate_cluster_run_report wall-clock steady-state window [60s, 240s] filtering + v2 telemetry
    const auto pol_win = PolicyConfig::from_preset(
        "morphserve",
        8,
        {
            {"steady_window_mode", "time"},
            {"warmup_window_s", 60.0},
            {"steady_window_s", 180.0},
        });

    auto make_outcome = [](const std::string& sid,
                           double arr_s,
                           const std::string& status,
                           bool joint_ok,
                           double active_ms,
                           double think_ms,
                           const std::string& phase_label) -> nlohmann::json {
        nlohmann::json j = {
            {"session_id", sid},
            {"quadrant", "interleaved_co_creation"},
            {"phase_label", phase_label},
            {"arrival_offset_s", arr_s},
            {"status", status},
            {"e2e_ms", active_ms + think_ms},
            {"active_jct_ms", active_ms},
            {"think_time_ms", think_ms},
            {"ar_ttfts_ms", status == "ok" ? nlohmann::json::array({220.0}) : nlohmann::json::array()},
            {"ar_tpots_ms", status == "ok" ? nlohmann::json::array({48.0, 52.0, 50.0}) : nlohmann::json::array()},
            {"dit_turns_ms", status == "ok" ? nlohmann::json::array({4100.0}) : nlohmann::json::array()},
            {"dit_turn_heights", status == "ok" ? nlohmann::json::array({512}) : nlohmann::json::array()},
            {"dit_turn_steps", status == "ok" ? nlohmann::json::array({20}) : nlohmann::json::array()},
            {"ar_slo_met", joint_ok},
            {"dit_slo_met", joint_ok},
            {"joint_slo_met", joint_ok},
            {"resident_alpha_sample", 0.75},
            {"path_a_ships", 1},
            {"path_b_morphs", 0},
            {"total_yields", 2},
            {"turn_records",
             nlohmann::json::array({
                 {
                     {"turn_idx", 0},
                     {"status", status},
                     {"ar_ttft_ms", 220.0},
                     {"ar_tpots_ms", nlohmann::json::array({48.0, 52.0})},
                     {"kv_mib", 128.0},
                 },
                 {
                     {"turn_idx", 1},
                     {"status", status},
                     {"dit_ms", 4100.0},
                     {"kv_mib", 384.0},
                 },
             })},
        };
        return j;
    };

    nlohmann::json outcomes = nlohmann::json::array({
        // Warmup [0, 60): OOM & slow sessions must be excluded from steady-state metrics!
        make_outcome("w_0", 10.0, "oom", false, 99999.0, 0.0, "ar80_dit20"),
        make_outcome("w_1", 35.0, "ok", false, 80000.0, 10000.0, "ar80_dit20"),
        // Steady [60, 240]: 4 clean ok sessions
        make_outcome("s_0", 70.0, "ok", true, 5000.0, 6000.0, "ar80_dit20"),
        make_outcome("s_1", 125.0, "ok", true, 6000.0, 8000.0, "ar20_dit80"),
        make_outcome("s_2", 185.0, "ok", true, 7000.0, 10000.0, "ar20_dit80"),
        make_outcome("s_3", 230.0, "ok", true, 8000.0, 12000.0, "ar50_dit50"),
        // Drain (> 240): OOM & failed sessions must be excluded from steady-state metrics!
        make_outcome("d_0", 255.0, "oom", false, 99999.0, 0.0, "ar50_dit50"),
        make_outcome("d_1", 275.0, "ok", false, 90000.0, 15000.0, "ar50_dit50"),
    });
    nlohmann::json worker_stats = nlohmann::json::array({
        {
            {"nvml_peak_used_mib", 31200.0},
            {"pytorch_reserved_mib", 19200.0},
            {"oom_count", 2},
            {"vmm_stats", {{"kv_swap_outs_total", 7}, {"overflow_creates", 0}}},
        },
        {
            {"nvml_peak_used_mib", 30800.0},
            {"pytorch_reserved_mib", 18900.0},
            {"oom_count", 0},
            {"vmm_stats", {{"kv_swap_outs_total", 5}, {"overflow_creates", 0}}},
        },
    });

    const auto rep = MorphControlEngine::aggregate_cluster_run_report(
        pol_win, "unit_v2_steady_window", 0.5, 300.0, 0.1, outcomes, worker_stats);

    MS_CHECK(rep.at("steady_state_window_mode").get<std::string>() == "time");
    MS_CHECK(std::abs(rep.at("steady_window_start_s").get<double>() - 60.0) < 1e-6);
    MS_CHECK(std::abs(rep.at("steady_window_end_s").get<double>() - 240.0) < 1e-6);
    MS_CHECK(rep.at("total_sessions").get<int>() == 8);
    MS_CHECK(rep.at("oom_sessions").get<int>() == 2);
    MS_CHECK(rep.at("steady_state_sessions").get<int>() == 4);
    MS_CHECK(rep.at("steady_oom_sessions").get<int>() == 0);
    MS_CHECK(std::abs(rep.at("steady_state_attainment_pct").get<double>() - 100.0) < 1e-6);
    MS_CHECK(std::abs(rep.at("steady_ar_slo_attainment_pct").get<double>() - 100.0) < 1e-6);
    MS_CHECK(std::abs(rep.at("steady_ar_slo_mean_pct").get<double>() - 100.0) < 1e-6);
    MS_CHECK(std::abs(rep.at("steady_ar_slo_strict_max_pct").get<double>() - 100.0) < 1e-6);
    MS_CHECK(std::abs(rep.at("steady_dit_slo_attainment_pct").get<double>() - 100.0) < 1e-6);
    MS_CHECK(std::abs(rep.at("effective_slo_throughput_rps").get<double>() - 0.5) < 1e-6);
    MS_CHECK(std::abs(rep.at("steady_state_goodput_rps").get<double>() - 0.5) < 1e-6);
    // Active JCT in steady window is [5000, 6000, 7000, 8000] ms (warmup/drain 80s/90s excluded!)
    MS_CHECK(rep.at("active_jct_p50_ms").get<double>() >= 6000.0 && rep.at("active_jct_p50_ms").get<double>() <= 7000.0);
    MS_CHECK(rep.at("active_jct_p95_ms").get<double>() <= 8000.0);
    MS_CHECK(rep.at("active_jct_p95_ms").get<double>() < rep.at("session_e2e_p95_ms").get<double>());
    MS_CHECK(!rep.at("all_session_active_jct_p95_ms").is_null());
    MS_CHECK(rep.at("active_jct_cdf_points").is_array() && !rep.at("active_jct_cdf_points").empty());
    MS_CHECK(rep.at("kv_host_swap_outs").get<int>() == 12);
    MS_CHECK(rep.at("per_turn_stats").contains("turn_0") && rep.at("per_turn_stats").contains("turn_1"));
    MS_CHECK(rep.at("per_turn_stats").at("turn_0").at("total").get<int>() == 4);
    MS_CHECK(rep.at("time_series_10s").is_array() && rep.at("time_series_10s").size() >= 6);
}

void test_d1_d5_multiturn_slo_contract_and_cot_decoupling() {
    using namespace morphserve::control;
    // D4: Default preset decouples k_AR=2.0 (1500ms / 125.64ms) and k_DiT=3.0 (18360ms / 65160ms BAGEL; 9750ms / 21900ms InternVL-U)
    const PolicyConfig pol_bagel = PolicyConfig::from_preset("morphserve", 8);
    MS_CHECK(std::abs(pol_bagel.slo_scale_k - 2.0) < 1e-9);
    MS_CHECK(std::abs(pol_bagel.dit_slo_scale_k - 3.0) < 1e-9);
    MS_CHECK(std::abs(pol_bagel.ar_ttft_slo_ms - 1500.0) < 1e-6);
    MS_CHECK(std::abs(pol_bagel.ar_tpot_slo_ms - 125.64) < 1e-6);
    MS_CHECK(std::abs(pol_bagel.dit_512_slo_ms - 18360.0) < 1e-6);
    MS_CHECK(std::abs(pol_bagel.dit_1024_slo_ms - 65160.0) < 1e-6);
    MS_CHECK(std::abs(pol_bagel.dit_turn_slo_for(512, 20) - 18360.0) < 1e-6);
    MS_CHECK(std::abs(pol_bagel.dit_turn_slo_for(1024, 20) - 65160.0) < 1e-6);

    const PolicyConfig pol_ivl = PolicyConfig::from_preset("morphserve", 8, {{"model_family", "internvl_u"}});
    MS_CHECK(std::abs(pol_ivl.dit_slo_scale_k - 3.0) < 1e-9);
    MS_CHECK(std::abs(pol_ivl.dit_512_slo_ms - 9750.0) < 1e-6);
    MS_CHECK(std::abs(pol_ivl.dit_1024_slo_ms - 21900.0) < 1e-6);

    // D1 + D2 + D4 + D5: Two 4-turn steady sessions (`planned_turns=4` each => 8 planned turns):
    // - Session A (`ok`): T0 ar_only ok, T1 cot_then_dit ok at 3.0x (dit=15000ms > 12240ms@2.0x, <= 18360ms@3.0x),
    //   T2 ar_only ok, T3 dit_edit late at 3.0x (dit=20000ms > 18360ms@3.0x, <= 24480ms@4.0x).
    //   => Session A has 3/4 turns met at 3.0x, 2/4 at 2.0x, 4/4 at 4.0x; session_ok = false at 3.0x, true at 4.0x!
    // - Session B (`oom` at T1): T0 ar_only met SLO, T1 dit_edit OOM, T2 & T3 unexecuted!
    //   => D2 planned-turns denominator: 1/4 turns met (T0=1, T1..T3=miss), session_ok = false, steady_oom = 1!
    // Combined over 8 planned turns at k_DiT=3.0: 4/8 = 50.0% turn SLO, 0/2 = 0.0% session SLO, 50.0% session survival, 5/8 = 62.5% turn survival!
    nlohmann::json out_a = {
        {"session_id", "s_a"},
        {"quadrant", "interleaved_co_creation"},
        {"arrival_offset_s", 80.0},
        {"status", "ok"},
        {"e2e_ms", 42000.0},
        {"active_jct_ms", 36000.0},
        {"think_time_ms", 6000.0},
        {"planned_turns", 4},
        {"planned_ar_turns", 3},
        {"planned_dit_turns", 2},
        {"ar_ttfts_ms", nlohmann::json::array({300.0, 400.0, 350.0})},
        {"ar_tpots_ms", nlohmann::json::array({55.0, 60.0, 58.0})},
        {"ar_turn_tpots_ms", nlohmann::json::array({{55.0}, {60.0}, {58.0}})},
        {"dit_turns_ms", nlohmann::json::array({15000.0, 20000.0})},
        {"dit_turn_heights", nlohmann::json::array({512, 512})},
        {"dit_turn_steps", nlohmann::json::array({20, 20})},
        {"ar_slo_met", true},
        {"dit_slo_met", false},
        {"joint_slo_met", false},
        {"turn_records",
         nlohmann::json::array({
             {{"turn_idx", 0}, {"modality_plan", "ar_only"}, {"status", "ok"}, {"ar_ttft_ms", 300.0}, {"ar_tpot_p95_ms", 55.0}, {"height", 0}, {"dit_steps", 0}},
             {{"turn_idx", 1}, {"modality_plan", "cot_then_dit"}, {"status", "ok"}, {"ar_ttft_ms", 400.0}, {"ar_tpot_p95_ms", 60.0}, {"dit_ms", 15000.0}, {"height", 512}, {"dit_steps", 20}},
             {{"turn_idx", 2}, {"modality_plan", "ar_only"}, {"status", "ok"}, {"ar_ttft_ms", 350.0}, {"ar_tpot_p95_ms", 58.0}, {"height", 0}, {"dit_steps", 0}},
             {{"turn_idx", 3}, {"modality_plan", "dit_edit"}, {"status", "ok"}, {"dit_ms", 20000.0}, {"height", 512}, {"dit_steps", 20}},
         })},
    };
    nlohmann::json out_b = {
        {"session_id", "s_b"},
        {"quadrant", "interleaved_co_creation"},
        {"arrival_offset_s", 120.0},
        {"status", "oom"},
        {"e2e_ms", 5000.0},
        {"active_jct_ms", 5000.0},
        {"think_time_ms", 0.0},
        {"planned_turns", 4},
        {"planned_ar_turns", 2},
        {"planned_dit_turns", 2},
        {"ar_ttfts_ms", nlohmann::json::array({250.0})},
        {"ar_tpots_ms", nlohmann::json::array({50.0})},
        {"dit_turns_ms", nlohmann::json::array()},
        {"ar_slo_met", false},
        {"dit_slo_met", false},
        {"joint_slo_met", false},
        {"turn_records",
         nlohmann::json::array({
             {{"turn_idx", 0}, {"modality_plan", "ar_only"}, {"status", "ok"}, {"ar_ttft_ms", 250.0}, {"ar_tpot_p95_ms", 50.0}, {"height", 0}, {"dit_steps", 0}},
             {{"turn_idx", 1}, {"modality_plan", "dit_edit"}, {"status", "oom"}, {"height", 512}, {"dit_steps", 20}},
         })},
    };
    const auto rep = MorphControlEngine::aggregate_cluster_run_report(
        pol_bagel, "unit_d1_d5_multiturn", 0.5, 200.0, 0.1, nlohmann::json::array({out_a, out_b}), nlohmann::json::array());

    MS_CHECK(rep.at("steady_planned_turns").get<int>() == 8);
    MS_CHECK(rep.at("steady_completed_turns").get<int>() == 5);
    MS_CHECK(std::abs(rep.at("steady_turn_survival_pct").get<double>() - 62.5) < 1e-6);
    MS_CHECK(std::abs(rep.at("steady_turn_slo_attainment_pct").get<double>() - 50.0) < 1e-6);
    MS_CHECK(std::abs(rep.at("steady_turn_ar_slo_attainment_pct").get<double>() - 80.0) < 1e-6);   // 4/5 AR turns
    MS_CHECK(std::abs(rep.at("steady_turn_dit_slo_attainment_pct").get<double>() - 25.0) < 1e-6);  // 1/4 DiT turns
    MS_CHECK(std::abs(rep.at("steady_state_attainment_pct").get<double>() - 0.0) < 1e-6);
    MS_CHECK(std::abs(rep.at("steady_state_survival_pct").get<double>() - 50.0) < 1e-6);
    MS_CHECK(rep.at("slo_config").at("k_ar").get<double>() == 2.0);
    MS_CHECK(rep.at("slo_config").at("k_dit").get<double>() == 3.0);
    MS_CHECK(rep.at("slo_config").at("cot_dit_decoupled").get<bool>() == true);
    MS_CHECK(rep.at("slo_config").at("primary_granularity").get<std::string>() == "turn_level");
    const auto& by_dk = rep.at("slo_curve_by_dit_k");
    MS_CHECK(std::abs(by_dk.at("2.0x").at("turn_attainment_pct").get<double>() - 37.5) < 1e-6);  // 3/8 (A:T0,T2 + B:T0)
    MS_CHECK(std::abs(by_dk.at("3.0x").at("turn_attainment_pct").get<double>() - 50.0) < 1e-6);  // 4/8 (+ A:T1)
    MS_CHECK(std::abs(by_dk.at("4.0x").at("turn_attainment_pct").get<double>() - 62.5) < 1e-6);  // 5/8 (+ A:T3)
    MS_CHECK(std::abs(by_dk.at("4.0x").at("session_attainment_pct").get<double>() - 50.0) < 1e-6);  // A passes at 4.0x

    // D3 in C++17 Simulator: `cot_then_dit` solo 512px turn must measure DiT latency strictly AFTER AR CoT finishes (~5-6s, NOT ~8-9s!)
    const ClusterPhysicsProfile ph{};
    const auto hw0 = HardwareInterconnectSpec::presets(&ph)[0];
    std::vector<SimSessionSpec> cot_solo = {
        {0.0, "visual_cot_highres", {{"cot_then_dit", 512, 20, 48, 0, false}}, "solo_cot_0", "main"},
    };
    const auto sim_rep = ClusterSimulator::simulate_cluster_point(
        "morphserve", 8, hw0, 0.1, 1, 2026, ph, WorkloadScaleSpec::bagel_7b_standard(), "", "", "", &cot_solo);
    MS_CHECK(sim_rep.at("dit_turn_p50_ms").get<double>() < 6500.0);
    MS_CHECK(std::abs(sim_rep.at("steady_turn_slo_attainment_pct").get<double>() - 100.0) < 1e-6);
    MS_CHECK(sim_rep.contains("slo_curve_by_dit_k") && sim_rep.at("slo_curve_by_dit_k").contains("3.0x"));
}

}  // namespace

int main() {
    test_router_edf_and_budget_planner();
    test_physics_ledger_and_simulator();
    test_v2_policy_think_slo_and_steady_window();
    test_d1_d5_multiturn_slo_contract_and_cot_decoupling();
    std::cout << "[PASS] test_control_and_sim\n";
    return 0;
}
