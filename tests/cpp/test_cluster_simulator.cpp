#include "morphserve/control/morph_controller.hpp"
#include "morphserve/control/policy.hpp"
#include "morphserve/control/preempt_edf.hpp"
#include "morphserve/control/sim_memory_ledger.hpp"
#include "morphserve/control/sim_trace_loader.hpp"
#include "morphserve/control/simulator.hpp"
#include "morphserve/control/topology.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, const char* file, int line) {
    if (!cond) {
        ++g_failures;
        std::cerr << "[FAIL] " << file << ":" << line << ": " << expr << "\n";
    }
}

#define MS_CHECK(cond) check_impl(static_cast<bool>(cond), #cond, __FILE__, __LINE__)

void test_sim_memory_ledger() {
    using namespace morphserve::control;

    const ClusterPhysicsProfile ph{};
    const PolicyConfig ms_pol = PolicyConfig::from_preset("morphserve", 8);
    const PolicyConfig mono_pol = PolicyConfig::from_preset("monolithic_dp", 8);

    SimGpuMemoryLedger ms_ledger(0, 40960.0, ms_pol, "hybrid", 28, 28, 448, ph);
    SimGpuMemoryLedger mono_ledger(0, 40960.0, mono_pol, "hybrid", 28, 28, 448, ph);

    for (int sid = 0; sid < 4; ++sid) {
        MS_CHECK(ms_ledger.upsert_session_kv(sid, 1800.0));
        mono_ledger.upsert_session_kv(sid, 1800.0);
    }
    // Under heavy KV + 1024px DiT+VAE, TriPool-VMM evicts expert layers and succeeds; monolithic OOMs.
    MS_CHECK(ms_ledger.execute_dit_and_vae(1024, 0).first);
    MS_CHECK(ms_ledger.expert_evictions() > 0);
    MS_CHECK(!mono_ledger.execute_dit_and_vae(1024, 0).first);

    // Idle KV host swap-out and swap-in accounting.
    const double freed = ms_ledger.evict_idle_kv_to_host(2000.0, /*exclude_sid=*/0);
    MS_CHECK(freed >= 1800.0);
    MS_CHECK(ms_ledger.kv_host_swap_outs() >= 1);
    const double swap_in_mib = ms_ledger.ensure_session_resident(1);
    MS_CHECK(swap_in_mib >= 0.0);
    if (swap_in_mib > 0.0) {
        MS_CHECK(ms_ledger.kv_host_swap_ins() >= 1);
    }

    const auto stats = ms_ledger.stats_json();
    MS_CHECK(stats.at("gpu_id").get<int>() == 0);
    MS_CHECK(stats.at("peak_used_mib").get<double>() > 0.0);
    MS_CHECK(stats.at("expert_evictions").get<int>() == ms_ledger.expert_evictions());
    MS_CHECK(stats.at("kv_host_swap_outs").get<int>() == ms_ledger.kv_host_swap_outs());

    for (int sid = 0; sid < 4; ++sid) {
        ms_ledger.release_session(sid);
    }
    ms_ledger.reclaim_empty_slabs_to_super();
    MS_CHECK(ms_ledger.active_session_count() == 0);
    MS_CHECK(ms_ledger.active_kv_mib() == 0.0);
}

void test_sim_trace_loader_helpers() {
    using namespace morphserve::control;

    std::vector<RawSuiteSession> raw;
    for (int i = 0; i < 12; ++i) {
        RawSuiteSession rs;
        rs.session_id = "raw_" + std::to_string(i);
        rs.quadrant = (i % 2 == 0) ? "visual_cot" : "multiturn_edit";
        rs.arrival_offset_s = static_cast<double>(i) * 2.5;
        rs.turns = {SimTurnSpec{"ar_only", 512, 20, 48, 0, false}};
        raw.push_back(std::move(rs));
    }

    const auto strat = stratify_raw_sessions(raw, 6);
    MS_CHECK(strat.size() == 6);
    int n_cot = 0;
    int n_edit = 0;
    for (size_t i = 0; i < strat.size(); ++i) {
        if (strat[i].quadrant == "visual_cot") ++n_cot;
        if (strat[i].quadrant == "multiturn_edit") ++n_edit;
        if (i > 0) {
            MS_CHECK(strat[i].arrival_offset_s >= strat[i - 1].arrival_offset_s);
        }
    }
    MS_CHECK(n_cot == 3 && n_edit == 3);

    const std::vector<RawSuiteSession> base4(raw.begin(), raw.begin() + 4);
    const auto scaled = scale_raw_sessions(base4, 10);
    MS_CHECK(scaled.size() == 10);
    for (size_t i = 1; i < scaled.size(); ++i) {
        MS_CHECK(scaled[i].arrival_offset_s >= scaled[i - 1].arrival_offset_s);
    }

    // Built-in canonical stream requires zero on-disk fixture files.
    const auto built = build_sim_session_stream(12, 0.8, 8, 2026, 0, "", "", "");
    MS_CHECK(built.size() == 12);
    const auto interleaved = build_sim_session_stream(6, 0.5, 8, 2026, 0, "interleaved_co_creation", "", "");
    MS_CHECK(interleaved.size() == 6);
    MS_CHECK(interleaved[0].quadrant == "interleaved_co_creation");
    MS_CHECK(interleaved[0].turns.size() == 4);
}

void test_static_fission_multiturn_and_asymmetric_splits() {
    using namespace morphserve::control;

    const ClusterPhysicsProfile ph{};
    const HardwareInterconnectSpec hw_xnode{
        "unit_2node_2gpu", 40.0, 1.0, 10.0, 10.0, 10.0, 10.0, 5.0, 1};

    // 4-turn `multiturn_edit` on 2-node `static_nccl_fission`: 4 forward + 3 return = 7 cross-node ships.
    SimSessionSpec edit_4turn;
    edit_4turn.arrival_ms = 0.0;
    edit_4turn.quadrant = "multiturn_edit";
    edit_4turn.session_id = "unit_edit_4turn";
    edit_4turn.turns = {
        SimTurnSpec{"dit_edit", 512, 20, 48, 1, false},
        SimTurnSpec{"dit_edit", 512, 20, 48, 0, false},
        SimTurnSpec{"dit_edit", 512, 20, 48, 0, false},
        SimTurnSpec{"dit_edit", 512, 20, 48, 0, false},
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
        "",
        "",
        &custom_edit);
    MS_CHECK(r_nccl.at("oom_sessions").get<int>() == 0);
    MS_CHECK(r_nccl.at("path_a_ships").get<int>() == 4);
    MS_CHECK(r_nccl.at("cross_node_ships").get<int>() == 7);
    MS_CHECK(r_nccl.at("per_turn_stats").contains("turn_0") && r_nccl.at("per_turn_stats").contains("turn_3"));
    MS_CHECK(r_nccl.at("per_turn_stats").at("turn_3").at("ok").get<int>() == 1);

    // Asymmetric static split presets (`static_2und_6gen`, `static_6und_2gen`) directly in simulator.
    const HardwareInterconnectSpec hw8 = HardwareInterconnectSpec::presets(&ph)[0];
    const std::vector<SimSessionSpec> mixed = {
        {0.0, "multi_image_vqa", {{"ar_only", 512, 20, 48, 1, false}}, "mix_0", "main"},
        {1000.0, "highres_t2i", {{"dit_t2i", 512, 20, 48, 0, false}}, "mix_1", "main"},
        {2000.0, "visual_cot", {{"cot_then_dit", 512, 20, 48, 0, false}}, "mix_2", "main"},
        {3000.0, "multiturn_edit", {{"dit_edit", 512, 20, 48, 1, false}, {"dit_edit", 512, 20, 48, 0, false}}, "mix_3", "main"},
    };
    for (const std::string& preset : {"static_2und_6gen", "static_6und_2gen"}) {
        const auto rep = ClusterSimulator::simulate_cluster_point(
            preset, 8, hw8, 0.4, 4, 2026, ph, WorkloadScaleSpec::bagel_7b_standard(), "", "", "", &mixed);
        MS_CHECK(rep.at("status").get<std::string>() == "ok");
        MS_CHECK(rep.at("policy_name").get<std::string>() == preset);
        MS_CHECK(rep.at("oom_sessions").get<int>() == 0);
        MS_CHECK(rep.at("path_a_ships").get<int>() > 0);
    }
}

void test_cot_decoupling_and_edf_controller_integration() {
    using namespace morphserve::control;

    const ClusterPhysicsProfile ph{};
    const HardwareInterconnectSpec hw0 = HardwareInterconnectSpec::presets(&ph)[0];
    const std::vector<SimSessionSpec> cot_solo = {
        {0.0, "visual_cot_highres", {{"cot_then_dit", 512, 20, 48, 0, false}}, "solo_cot_0", "main"},
    };

    // D3 timing decoupling + PreemptEdfController (guard_ms = 14.0, chunk_layers_512 = 5 on A100 1x).
    const auto rep = ClusterSimulator::simulate_cluster_point(
        "morphserve", 8, hw0, 0.1, 1, 2026, ph, WorkloadScaleSpec::bagel_7b_standard(), "", "", "", &cot_solo);
    MS_CHECK(rep.at("dit_turn_p50_ms").get<double>() < 6500.0);
    MS_CHECK(std::abs(rep.at("steady_turn_slo_attainment_pct").get<double>() - 100.0) < 1e-6);
    MS_CHECK(std::abs(rep.at("edf_guard_ms").get<double>() - 14.0) < 1e-6);

    const PreemptEdfController ref_edf = MorphControlEngine::make_edf_controller(
        ph.num_expert_layers,
        125.64,
        ph.dit_512_layer_p50_ms,
        ph.ar_unloaded_tpot_p50_ms + ph.vmm_ar_kv_step_overhead_ms,
        1.0);
    MS_CHECK(ref_edf.compute_adaptive_chunk_layers() == 5);
    MS_CHECK(rep.at("edf_chunk_layers_512").get<int>() == ref_edf.compute_adaptive_chunk_layers());

    // Loose AR TPOT SLO (`250.0 ms`) MUST yield `edf_chunk_layers_512 == 17 > 8` (no legacy cap of 8).
    const auto rep_loose = ClusterSimulator::simulate_cluster_point(
        "morphserve",
        8,
        hw0,
        0.1,
        1,
        2026,
        ph,
        WorkloadScaleSpec::bagel_7b_standard(),
        "",
        "",
        "",
        &cot_solo,
        {{"ar_tpot_slo_ms", 250.0}});
    MS_CHECK(rep_loose.at("edf_chunk_layers_512").get<int>() == 17);
}

void test_canonical_schema_parity_and_internvlu_steps() {
    using namespace morphserve::control;

    const ClusterPhysicsProfile ph{};
    const HardwareInterconnectSpec hw0 = HardwareInterconnectSpec::presets(&ph)[0];
    const std::vector<SimSessionSpec> timed_sessions = {
        {10000.0, "multi_image_vqa", {{"ar_only", 512, 20, 32, 0, false}}, "w_0", "warmup"},
        {80000.0, "visual_cot", {{"cot_then_dit", 512, 20, 32, 0, false}}, "s_0", "steady"},
        {120000.0, "highres_t2i", {{"dit_t2i", 1024, 20, 32, 0, false}}, "s_1", "steady"},
        {160000.0, "multi_image_vqa", {{"ar_only", 512, 20, 32, 0, false}}, "s_2", "steady"},
        {260000.0, "multi_image_vqa", {{"ar_only", 512, 20, 32, 0, false}}, "d_0", "drain"},
    };
    const nlohmann::json win_ov = {
        {"steady_window_mode", "time"},
        {"warmup_window_s", 60.0},
        {"steady_window_s", 180.0},
    };
    const auto rep_bagel = ClusterSimulator::simulate_cluster_point(
        "morphserve", 8, hw0, 0.4, 5, 2026, ph, WorkloadScaleSpec::bagel_7b_standard(), "", "", "", &timed_sessions, win_ov);

    // Canonical unified report keys shared verbatim with `MorphControlEngine::aggregate_cluster_run_report`.
    MS_CHECK(rep_bagel.at("steady_state_sessions").get<int>() == 3);
    MS_CHECK(
        std::abs(rep_bagel.at("joint_slo_goodput_rps").get<double>() -
                 rep_bagel.at("steady_joint_slo_goodput_rps").get<double>()) < 1e-9);
    MS_CHECK(
        std::abs(rep_bagel.at("joint_slo_attainment_pct").get<double>() -
                 rep_bagel.at("steady_joint_slo_attainment_pct").get<double>()) < 1e-9);
    MS_CHECK(rep_bagel.contains("full_trace_goodput_rps"));
    MS_CHECK(rep_bagel.contains("full_trace_joint_slo_attainment_pct"));
    MS_CHECK(std::abs(rep_bagel.at("vmm_physical_pool_mib").get<double>() - 14592.0) < 1e-6);
    MS_CHECK(std::abs(rep_bagel.at("slo_config").at("dit_1024_slo_ms").get<double>() - 65160.0) < 1e-6);
    MS_CHECK(std::abs(rep_bagel.at("slo_curve_by_k").at("3.0x").at("dit_1024_slo_ms").get<double>() - 65160.0) < 1e-6);
    MS_CHECK(std::abs(rep_bagel.at("slo_curve_by_dit_k").at("3.0x").at("dit_1024_slo_ms").get<double>() - 65160.0) < 1e-6);

    // InternVL-U (`model_family="internvl_u"`) uses 16 steps for 1024px (`dit_1024_slo_ms == 21900.0` at 3.0x).
    nlohmann::json ivl_ov = win_ov;
    ivl_ov["model_family"] = "internvl_u";
    const auto rep_ivl = ClusterSimulator::simulate_cluster_point(
        "morphserve", 8, hw0, 0.4, 5, 2026, ph, WorkloadScaleSpec::bagel_7b_standard(), "", "", "", &timed_sessions, ivl_ov);
    MS_CHECK(std::abs(rep_ivl.at("slo_config").at("dit_512_slo_ms").get<double>() - 9750.0) < 1e-6);
    MS_CHECK(std::abs(rep_ivl.at("slo_config").at("dit_1024_slo_ms").get<double>() - 21900.0) < 1e-6);
    MS_CHECK(std::abs(rep_ivl.at("slo_curve_by_k").at("3.0x").at("dit_1024_slo_ms").get<double>() - 21900.0) < 1e-6);
    MS_CHECK(std::abs(rep_ivl.at("slo_curve_by_dit_k").at("3.0x").at("dit_1024_slo_ms").get<double>() - 21900.0) < 1e-6);
}

void test_oom_retry_and_hbm_conservation_under_pressure() {
    using namespace morphserve::control;

    const ClusterPhysicsProfile ph{};
    // Tight 36 GiB HBM on 2 GPUs with 6 concurrent 1024px multi-turn edit sessions:
    // Monolithic (28.8 GiB static weights + 6.5 GiB unscoped 1024px VAE + 1.5 GiB DiT tail + KV > 36 GiB) triggers OOM;
    // MorphServe (TriPool-VMM + expert paging + scoped VAE) survives cleanly!
    const HardwareInterconnectSpec hw_tight{
        "unit_tight_36g", 36.0, 1.0, 8.755, 10.348, 10.581, 9.55, 11.5, 2};
    std::vector<SimSessionSpec> heavy;
    for (int i = 0; i < 6; ++i) {
        heavy.push_back(SimSessionSpec{
            static_cast<double>(i) * 200.0,
            "multiturn_edit",
            {
                SimTurnSpec{"dit_edit", 1024, 20, 48, 1, false},
                SimTurnSpec{"dit_edit", 1024, 20, 48, 0, false},
            },
            "heavy_" + std::to_string(i),
            "main",
        });
    }
    const nlohmann::json ov = {{"steady_window_mode", "session_trim"}, {"steady_state_trim_frac", 0.0}};
    const auto rep_mono = ClusterSimulator::simulate_cluster_point(
        "monolithic_dp", 2, hw_tight, 1.0, 6, 2026, ph, WorkloadScaleSpec::bagel_7b_standard(), "", "", "", &heavy, ov);
    const auto rep_ms = ClusterSimulator::simulate_cluster_point(
        "morphserve", 2, hw_tight, 1.0, 6, 2026, ph, WorkloadScaleSpec::bagel_7b_standard(), "", "", "", &heavy, ov);

    MS_CHECK(rep_mono.at("oom_sessions").get<int>() > 0);
    MS_CHECK(rep_mono.at("steady_turn_survival_pct").get<double>() < 100.0);
    MS_CHECK(rep_ms.at("oom_sessions").get<int>() == 0);
    MS_CHECK(std::abs(rep_ms.at("steady_turn_survival_pct").get<double>() - 100.0) < 1e-6);
    MS_CHECK(rep_ms.at("vmm_expert_evictions").get<int>() > 0);
}

void test_calibration_evaluate_against_real_runs_in_memory() {
    using namespace morphserve::control;

    const ClusterPhysicsProfile ph{};
    const HardwareInterconnectSpec hw0 = HardwareInterconnectSpec::presets(&ph)[0];

    // 1. Canonical self-consistent report.
    const auto sim_canon = ClusterSimulator::simulate_cluster_point(
        "morphserve", 8, hw0, 0.4, 16, 2026, ph, WorkloadScaleSpec::bagel_7b_standard());

    // 2. Legacy pre-refactor report schema (uniform k_dit=2.0, `joint_slo_goodput_rps` == full-trace).
    const auto sim_legacy_ref = ClusterSimulator::simulate_cluster_point(
        "morphserve",
        8,
        hw0,
        0.4,
        16,
        2026,
        ph,
        WorkloadScaleSpec::bagel_7b_standard(),
        "",
        "",
        "",
        nullptr,
        {{"dit_slo_scale_k", 2.0}});
    const nlohmann::json legacy_real = {
        {"policy_name", "morphserve"},
        {"num_gpus", 8},
        {"offered_qps", 0.4},
        {"total_sessions", 16},
        {"joint_slo_goodput_rps", sim_legacy_ref.at("full_trace_goodput_rps")},
        {"ar_tpot_p95_ms", sim_legacy_ref.at("ar_tpot_p95_ms")},
        {"dit_turn_p95_ms", sim_legacy_ref.at("dit_turn_p95_ms")},
    };

    const auto cal = ClusterSimulator::evaluate_against_real_runs(
        nlohmann::json::array({sim_canon, legacy_real}), ph, "", "");
    MS_CHECK(cal.at("points_checked").get<int>() == 2);
    MS_CHECK(cal.at("goodput_mape_pct").get<double>() < 1e-6);
    MS_CHECK(cal.at("within_5pct_gate").get<bool>());
    MS_CHECK(cal.at("within_8pct_oos_gate").get<bool>());
}

}  // namespace

int main() {
    test_sim_memory_ledger();
    test_sim_trace_loader_helpers();
    test_static_fission_multiturn_and_asymmetric_splits();
    test_cot_decoupling_and_edf_controller_integration();
    test_canonical_schema_parity_and_internvlu_steps();
    test_oom_retry_and_hbm_conservation_under_pressure();
    test_calibration_evaluate_against_real_runs_in_memory();

    if (g_failures > 0) {
        std::cerr << "[FAILED] test_cluster_simulator (" << g_failures << " checks failed)\n";
        return 1;
    }
    std::cout << "[PASS] test_cluster_simulator\n";
    return 0;
}
