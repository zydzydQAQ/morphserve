#include "morphserve/control/simulator.hpp"

#include "morphserve/control/hash_util.hpp"
#include "morphserve/control/policy.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace morphserve::control {

nlohmann::json ClusterSimulator::evaluate_against_real_runs(
    const nlohmann::json& real_runs_array,
    const ClusterPhysicsProfile& ph,
    const std::string& bench_root,
    const std::string& sources_dir) {
    if (!real_runs_array.is_array() || real_runs_array.empty()) {
        return {
            {"reference_source", "phase01_measured_json_profile"},
            {"zero_parameter_fitting", true},
            {"points_checked", 0},
            {"goodput_mape_pct", 0.0},
            {"ar_tpot_p95_mape_pct", 0.0},
            {"dit_p95_mape_pct", 0.0},
            {"max_point_error_pct", 0.0},
            {"within_8pct_oos_gate", false},
            {"within_5pct_gate", false},
            {"per_point", nlohmann::json::array()},
        };
    }

    const HardwareInterconnectSpec hw0 = HardwareInterconnectSpec::presets(&ph)[0];
    nlohmann::json per_pt = nlohmann::json::array();
    std::vector<double> gp_errs;
    std::vector<double> tp_errs;
    std::vector<double> dt_errs;
    double max_err = 0.0;

    for (const auto& ref : real_runs_array) {
        const std::string pol = ref.value("policy_name", "");
        if (!is_valid_policy_name(pol)) continue;
        const int n_gpus = ref.value("num_gpus", 8);
        const double qps = ref.value("offered_qps", 0.8);
        const int n_s = ref.value("total_sessions", 80);
        const std::string sk_tag =
            (ref.contains("skew_tag") && ref["skew_tag"].is_string()) ? ref["skew_tag"].get<std::string>() : "";
        HardwareInterconnectSpec eff_hw = hw0;
        if (n_gpus > 8) {
            eff_hw = HardwareInterconnectSpec{
                "Kairos_2x8_Measured_Fabric",
                40.0,
                1.0,
                ph.pcie_solo_h2d_gbps,
                ph.pcie_half0_agg_h2d_gbps,
                ph.kairos_2x8_intra_half_gbps,
                ph.kairos_2x8_cross_socket_gbps,
                ph.kairos_2x8_inter_node_gbps,
                8};
        }
        nlohmann::json ref_ov = nlohmann::json::object();
        for (const char* opt_k : {"slo_scale_k",
                                  "dit_slo_scale_k",
                                  "inter_turn_think_s_range",
                                  "think_time_dist",
                                  "think_time_median_s",
                                  "think_time_sigma",
                                  "ar_tpot_slo_mode",
                                  "steady_window_mode",
                                  "warmup_window_s",
                                  "steady_window_s",
                                  "cooldown_s",
                                  "steady_state_trim_frac"}) {
            if (ref.contains(opt_k) && !ref[opt_k].is_null()) {
                ref_ov[opt_k] = ref[opt_k];
            }
        }
        if (ref.contains("slo_config") && ref["slo_config"].is_object()) {
            const auto& sc = ref["slo_config"];
            if (sc.contains("k_ar") && sc["k_ar"].is_number() && !ref_ov.contains("slo_scale_k")) {
                ref_ov["slo_scale_k"] = sc["k_ar"];
            }
            if (sc.contains("k_dit") && sc["k_dit"].is_number() && !ref_ov.contains("dit_slo_scale_k")) {
                ref_ov["dit_slo_scale_k"] = sc["k_dit"];
            }
        } else if (!ref_ov.contains("dit_slo_scale_k") && !ref.contains("steady_turn_slo_goodput_rps")) {
            // Legacy v1 live reports evaluated DiT against uniform `k=2.0`.
            ref_ov["dit_slo_scale_k"] = 2.0;
        }

        const auto sim = simulate_cluster_point(
            pol,
            n_gpus,
            eff_hw,
            qps,
            n_s,
            2026,
            ph,
            WorkloadScaleSpec::bagel_7b_standard(),
            sk_tag,
            bench_root,
            sources_dir,
            nullptr,
            ref_ov);

        double real_gp = 0.0;
        double sim_gp = 0.0;
        const bool is_canonical_ref =
            ref.contains("steady_joint_slo_goodput_rps") || ref.contains("full_trace_goodput_rps");
        if (ref.contains("steady_turn_slo_goodput_rps") && ref["steady_turn_slo_goodput_rps"].is_number()) {
            real_gp = ref["steady_turn_slo_goodput_rps"].get<double>();
            sim_gp = sim.at("steady_turn_slo_goodput_rps").get<double>();
        } else if (is_canonical_ref && ref.contains("joint_slo_goodput_rps") &&
                   ref["joint_slo_goodput_rps"].is_number()) {
            real_gp = ref["joint_slo_goodput_rps"].get<double>();
            sim_gp = sim.at("joint_slo_goodput_rps").get<double>();
        } else if (ref.contains("effective_slo_throughput_rps") && ref["effective_slo_throughput_rps"].is_number() &&
                   ref.value("steady_state_window_mode", "") == "time") {
            real_gp = ref["effective_slo_throughput_rps"].get<double>();
            sim_gp = sim.at("joint_slo_goodput_rps").get<double>();
        } else if (!is_canonical_ref && ref.contains("joint_slo_goodput_rps") &&
                   ref["joint_slo_goodput_rps"].is_number()) {
            // Historical pre-canonical live reports stored full-trace goodput under `joint_slo_goodput_rps`.
            real_gp = ref["joint_slo_goodput_rps"].get<double>();
            sim_gp = sim.at("full_trace_goodput_rps").get<double>();
        } else {
            real_gp = ref.value("steady_state_goodput_rps", 0.0);
            sim_gp = sim.at("joint_slo_goodput_rps").get<double>();
        }

        double gp_e = 0.0;
        std::string gp_status = "ok";
        if (real_gp <= 1e-9 && sim_gp <= 1e-9) {
            gp_e = 0.0;
            gp_status = "both_zero_oom";
        } else if (real_gp <= 1e-9 || sim_gp <= 1e-9) {
            gp_e = 100.0;
            gp_status = "oom_mismatch";
        } else {
            gp_e = std::abs(sim_gp - real_gp) / real_gp * 100.0;
        }
        gp_errs.push_back(gp_e);
        max_err = std::max(max_err, gp_e);

        // Transparent zero-real OOM handling: NEVER overwrite real <= 0 with sim!
        const bool has_real_tp = ref.contains("ar_tpot_p95_ms") && ref["ar_tpot_p95_ms"].is_number();
        const double raw_real_tp = has_real_tp ? ref["ar_tpot_p95_ms"].get<double>() : 0.0;
        const double sim_tp = sim.at("ar_tpot_p95_ms").get<double>();
        nlohmann::json tp_err_j = nullptr;
        std::string tp_status = "ok";
        if (!has_real_tp) {
            tp_status = "missing_real";
        } else if (raw_real_tp <= 1e-9 && sim_tp <= 1e-9) {
            tp_status = "both_zero_oom";
        } else if (raw_real_tp <= 1e-9 || sim_tp <= 1e-9) {
            const double tp_e = 100.0;
            tp_errs.push_back(tp_e);
            max_err = std::max(max_err, tp_e);
            tp_err_j = round_2(tp_e);
            tp_status = "oom_mismatch";
        } else {
            const double tp_e = std::abs(sim_tp - raw_real_tp) / raw_real_tp * 100.0;
            tp_errs.push_back(tp_e);
            max_err = std::max(max_err, tp_e);
            tp_err_j = round_2(tp_e);
        }

        const bool has_real_dt = ref.contains("dit_turn_p95_ms") && ref["dit_turn_p95_ms"].is_number();
        const double raw_real_dt = has_real_dt ? ref["dit_turn_p95_ms"].get<double>() : 0.0;
        const double sim_dt = sim.at("dit_turn_p95_ms").get<double>();
        nlohmann::json dt_err_j = nullptr;
        std::string dt_status = "ok";
        if (!has_real_dt) {
            dt_status = "missing_real";
        } else if (raw_real_dt <= 1e-9 && sim_dt <= 1e-9) {
            dt_status = "both_zero_oom";
        } else if (raw_real_dt <= 1e-9 || sim_dt <= 1e-9) {
            const double dt_e = 100.0;
            dt_errs.push_back(dt_e);
            max_err = std::max(max_err, dt_e);
            dt_err_j = round_2(dt_e);
            dt_status = "oom_mismatch";
        } else {
            const double dt_e = std::abs(sim_dt - raw_real_dt) / raw_real_dt * 100.0;
            dt_errs.push_back(dt_e);
            max_err = std::max(max_err, dt_e);
            dt_err_j = round_2(dt_e);
        }

        per_pt.push_back({
            {"policy_name", pol},
            {"num_gpus", n_gpus},
            {"offered_qps", qps},
            {"real_goodput_rps", round_4(real_gp)},
            {"sim_goodput_rps", round_4(sim_gp)},
            {"goodput_err_pct", round_2(gp_e)},
            {"goodput_status", gp_status},
            {"real_ar_tpot_p95_ms", has_real_tp ? nlohmann::json(round_2(raw_real_tp)) : nlohmann::json(nullptr)},
            {"sim_ar_tpot_p95_ms", round_2(sim_tp)},
            {"ar_tpot_p95_err_pct", tp_err_j},
            {"ar_tpot_p95_status", tp_status},
            {"real_dit_p95_ms", has_real_dt ? nlohmann::json(round_2(raw_real_dt)) : nlohmann::json(nullptr)},
            {"sim_dit_p95_ms", round_2(sim_dt)},
            {"dit_p95_err_pct", dt_err_j},
            {"dit_p95_status", dt_status},
        });
    }

    const bool has_points = !per_pt.empty();
    const double mgp = round_2(mean_raw(gp_errs));
    const double mtp = round_2(mean_raw(tp_errs));
    const double mdt = round_2(mean_raw(dt_errs));
    const bool within_8 = has_points && (mgp < 8.0) && (tp_errs.empty() || mtp < 8.0) && (dt_errs.empty() || mdt < 8.0);
    const bool within_5 = has_points && (mgp < 5.0) && (tp_errs.empty() || mtp < 5.0) && (dt_errs.empty() || mdt < 5.0);

    return {
        {"reference_source", "live_cluster_out_of_sample"},
        {"zero_parameter_fitting", true},
        {"points_checked", per_pt.size()},
        {"goodput_mape_pct", mgp},
        {"ar_tpot_p95_mape_pct", mtp},
        {"dit_p95_mape_pct", mdt},
        {"max_point_error_pct", round_2(max_err)},
        {"within_8pct_oos_gate", within_8},
        {"within_5pct_gate", within_5},
        {"per_point", per_pt},
    };
}

nlohmann::json ClusterSimulator::run_h100_80gb_generality_study(
    const ClusterPhysicsProfile& ph,
    const std::string& bench_root,
    const std::string& sources_dir) {
    const HardwareInterconnectSpec h100 = HardwareInterconnectSpec::presets(&ph)[2];
    nlohmann::json regime1 = nlohmann::json::array();
    for (double qps : {0.9, 1.4, 1.9, 2.5}) {
        for (const char* pol : kValidPolicies) {
            regime1.push_back(simulate_cluster_point(
                pol,
                8,
                h100,
                qps,
                96,
                2026,
                ph,
                WorkloadScaleSpec::bagel_7b_standard(),
                "",
                bench_root,
                sources_dir));
        }
    }

    nlohmann::json regime2_56g = nlohmann::json::array();
    const auto ws_56g = WorkloadScaleSpec::frontier_mot_56gib();
    for (const auto& [ng, qps, ns] : std::vector<std::tuple<int, double, int>>{{8, 1.35, 96}, {64, 10.8, 192}}) {
        for (const char* pol : kValidPolicies) {
            regime2_56g.push_back(
                simulate_cluster_point(pol, ng, h100, qps, ns, 2026, ph, ws_56g, "", bench_root, sources_dir));
        }
    }

    nlohmann::json regime2_long_kv = nlohmann::json::array();
    const auto ws_lkv = WorkloadScaleSpec::bagel_7b_long_kv_high_concurrency();
    for (const auto& [ng, qps, ns] : std::vector<std::tuple<int, double, int>>{{8, 2.4, 96}, {64, 19.2, 192}}) {
        for (const char* pol : kValidPolicies) {
            regime2_long_kv.push_back(
                simulate_cluster_point(pol, ng, h100, qps, ns, 2026, ph, ws_lkv, "", bench_root, sources_dir));
        }
    }

    return {
        {"hardware", h100.to_json()},
        {"regime1_7b_qps_latency_curve", regime1},
        {"regime2_scaled_56gib_weights", regime2_56g},
        {"regime2_long_kv_high_concurrency", regime2_long_kv},
    };
}

nlohmann::json ClusterSimulator::run_scale_and_interconnect_sweep(
    const std::string& output_dir,
    const std::string& real_report_path,
    const std::string& results_dir,
    const std::string& bench_root,
    const std::string& sources_dir) {
    if (!output_dir.empty()) {
        std::filesystem::create_directories(output_dir);
    }
    const ClusterPhysicsProfile ph = ClusterPhysicsProfile::from_phase01_results(results_dir);

    nlohmann::json real_runs = nlohmann::json::array();
    if (!real_report_path.empty()) {
        std::error_code ec;
        if (std::filesystem::is_regular_file(real_report_path, ec)) {
            std::ifstream in(real_report_path, std::ios::binary);
            if (in) {
                const auto raw = nlohmann::json::parse(in);
                if (raw.contains("qps_sweep") && raw["qps_sweep"].is_array()) {
                    for (const auto& x : raw["qps_sweep"]) real_runs.push_back(x);
                }
                if (raw.contains("skew_sweep") && raw["skew_sweep"].is_array()) {
                    for (const auto& x : raw["skew_sweep"]) real_runs.push_back(x);
                }
                if (raw.contains("results") && raw["results"].is_array() && !raw.contains("qps_sweep")) {
                    for (const auto& x : raw["results"]) real_runs.push_back(x);
                }
            }
        }
    }

    const auto val = evaluate_against_real_runs(real_runs, ph, bench_root, sources_dir);
    const auto presets = HardwareInterconnectSpec::presets(&ph);
    nlohmann::json presets_j = nlohmann::json::array();
    for (const auto& h : presets) presets_j.push_back(h.to_json());

    nlohmann::json matrix = nlohmann::json::array();
    for (const auto& hw : presets) {
        for (int num_gpus : {8, 16, 64}) {
            const double scaled_qps = round_2(1.15 * (static_cast<double>(num_gpus) / 8.0) * hw.compute_speedup_x);
            const int n_sess = (num_gpus <= 16) ? 96 : 240;
            for (const char* pol : kValidPolicies) {
                matrix.push_back(simulate_cluster_point(
                    pol,
                    num_gpus,
                    hw,
                    scaled_qps,
                    n_sess,
                    2026,
                    ph,
                    WorkloadScaleSpec::bagel_7b_standard(),
                    "",
                    bench_root,
                    sources_dir));
            }
        }
    }

    const auto h100_study = run_h100_80gb_generality_study(ph, bench_root, sources_dir);

    const auto a100_pcie = presets.front();
    nlohmann::json slo_vs_qps_8gpu = nlohmann::json::object();
    for (const char* pol : kValidPolicies) {
        nlohmann::json pts = nlohmann::json::array();
        std::vector<std::pair<double, double>> full_curve;
        std::vector<std::pair<double, double>> steady_curve;
        for (double q : {0.05, 0.10, 0.15, 0.20, 0.25, 0.40, 0.80, 1.10, 1.40}) {
            const int ns = (q <= 0.10) ? 28 : ((q <= 0.25) ? 48 : 96);
            auto sr = simulate_cluster_point(
                pol, 8, a100_pcie, q, ns, 2026, ph, WorkloadScaleSpec::bagel_7b_standard(), "", bench_root, sources_dir);
            full_curve.emplace_back(q, sr.at("full_trace_joint_slo_attainment_pct").get<double>());
            steady_curve.emplace_back(q, sr.at("joint_slo_attainment_pct").get<double>());
            pts.push_back(std::move(sr));
        }
        auto interp_qps = [](const std::vector<std::pair<double, double>>& cv, double thr) -> double {
            double best = 0.0;
            for (size_t i = 0; i < cv.size(); ++i) {
                const auto [q_cur, a_cur] = cv[i];
                if (a_cur >= thr) {
                    best = std::max(best, q_cur);
                    if (i + 1 < cv.size() && cv[i + 1].second < thr && a_cur > cv[i + 1].second) {
                        const double frac = (a_cur - thr) / (a_cur - cv[i + 1].second);
                        best = std::max(best, q_cur + frac * (cv[i + 1].first - q_cur));
                    }
                }
            }
            return round_4(best);
        };
        slo_vs_qps_8gpu[pol] = {
            {"max_sustainable_qps_80pct", interp_qps(full_curve, 80.0)},
            {"max_sustainable_qps_90pct", interp_qps(full_curve, 90.0)},
            {"steady_max_sustainable_qps_80pct", interp_qps(steady_curve, 80.0)},
            {"steady_max_sustainable_qps_90pct", interp_qps(steady_curve, 90.0)},
            {"points", std::move(pts)},
        };
    }

    std::time_t now_t = std::time(nullptr);
    std::tm gmt = {};
    gmtime_r(&now_t, &gmt);
    char ts_buf[32];
    std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%dT%H:%M:%SZ", &gmt);

    const bool gate_ok = val.value("within_8pct_oos_gate", false);
    nlohmann::json report = {
        {"schema", "morphserve.osdi27.simulator.v3"},
        {"timestamp_utc", std::string(ts_buf)},
        {"zero_manual_bias_constants", true},
        {"physics_profile", ph.to_json()},
        {"calibration", val},
        {"interconnect_presets", presets_j},
        {"slo_attainment_vs_qps_8gpu", slo_vs_qps_8gpu},
        {"scaling_matrix", matrix},
        {"h100_80gb_generality", h100_study},
        {"status", gate_ok ? "ok" : "check_failed"},
    };

    if (!output_dir.empty()) {
        std::filesystem::path out_p = std::filesystem::path(output_dir) / "m25-simulator-64gpu-report.json";
        std::ofstream out(out_p, std::ios::binary | std::ios::trunc);
        if (out) {
            out << report.dump(2) << "\n";
        }
    }
    return report;
}

}  // namespace morphserve::control
