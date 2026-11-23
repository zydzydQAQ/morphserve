#include "morphserve/control/morph_controller.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace morphserve::control {

namespace {

inline double round_1(double x) noexcept { return std::round(x * 10.0) / 10.0; }
inline double round_2(double x) noexcept { return std::round(x * 100.0) / 100.0; }
inline double round_3(double x) noexcept { return std::round(x * 1000.0) / 1000.0; }
inline double round_4(double x) noexcept { return std::round(x * 10000.0) / 10000.0; }

bool evaluate_outcome_at_k(const PolicyConfig& policy, const nlohmann::json& out, double k) {
    if (out.value("status", "") != "ok") {
        return false;
    }
    const std::string fam = (policy.model_family == "internvl_u") ? "internvl_u" : "bagel";
    const auto slo = default_operator_profile().slo_thresholds_ms(k, fam);
    const double ttft_cap = slo.at("ar_ttft_slo_ms");
    const double tpot_cap = slo.at("ar_tpot_slo_ms");
    if (out.contains("ar_ttfts_ms") && out["ar_ttfts_ms"].is_array()) {
        for (const auto& v : out["ar_ttfts_ms"]) {
            if (v.get<double>() > ttft_cap) return false;
        }
    }
    if (out.contains("ar_tpots_ms") && out["ar_tpots_ms"].is_array()) {
        for (const auto& v : out["ar_tpots_ms"]) {
            if (v.get<double>() > tpot_cap) return false;
        }
    }
    if (out.contains("dit_turns_ms") && out["dit_turns_ms"].is_array()) {
        const auto& lats = out["dit_turns_ms"];
        const auto& hs = out.value("dit_turn_heights", nlohmann::json::array());
        const auto& sts = out.value("dit_turn_steps", nlohmann::json::array());
        for (size_t i = 0; i < lats.size(); ++i) {
            const int h = (i < hs.size()) ? hs[i].get<int>() : 512;
            const int st = (i < sts.size()) ? sts[i].get<int>() : 20;
            if (lats[i].get<double>() > policy.dit_turn_slo_for(h, st, k)) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

nlohmann::json TurnTransitionPlan::to_json() const {
    return {
        {"decision", decision.to_json()},
        {"ship_mode", ship_mode},
        {"src_gpu", src_gpu},
        {"dst_gpu", dst_gpu},
        {"est_dit_ms", est_dit_ms},
        {"reserved_kv_bytes", reserved_kv_bytes},
        {"ephemeral_gen_ctx", ephemeral_gen_ctx},
        {"needs_return_ship", needs_return_ship},
        {"release_src_on_fwd", release_src_on_fwd},
        {"include_cfg_dup_on_wire", include_cfg_dup_on_wire},
        {"dit_ctx_id", dit_ctx_id},
    };
}

MorphControlEngine::MorphControlEngine(
    PolicyConfig policy,
    ClusterTopologyProfile topology,
    std::shared_ptr<DualPathMorphRouter> router)
    : policy_(std::move(policy)),
      topology_(std::move(topology)),
      router_(router ? std::move(router) : std::make_shared<DualPathMorphRouter>(topology_)) {}

double MorphControlEngine::deterministic_think_s(
    const std::string& session_id,
    int turn_idx,
    double lo_s,
    double hi_s) {
    if (hi_s <= lo_s) {
        return std::max(0.0, lo_s);
    }
    const uint32_t h = md5_prefix_u32(session_id + ":" + std::to_string(turn_idx));
    const double frac = static_cast<double>(h % 10000U) / 10000.0;
    return lo_s + frac * (hi_s - lo_s);
}

double MorphControlEngine::deterministic_think_ms_sim(
    int sid,
    int turn_idx,
    double lo_s,
    double hi_s) {
    const double lo_ms = lo_s * 1000.0;
    const double hi_ms = hi_s * 1000.0;
    if (hi_ms <= lo_ms) {
        return std::max(0.0, lo_ms);
    }
    const uint32_t h = md5_prefix_u32("sim_" + std::to_string(sid) + "_" + std::to_string(turn_idx));
    const double frac = static_cast<double>(h % 10000U) / 10000.0;
    return lo_ms + frac * (hi_ms - lo_ms);
}

TurnTransitionPlan MorphControlEngine::plan_turn_transition(
    const std::string& session_id,
    const std::string& base_ctx_id,
    int turn_idx,
    int attempt,
    int src_gpu,
    int64_t state_bytes,
    const std::string& modality_plan,
    int height,
    int dit_steps,
    bool has_more_turns,
    bool has_more_edits,
    double now_ms,
    const std::map<int, bool>& gpu_busy) {
    const std::string arch = policy_.cptg_architecture();
    const bool is_tight_mot = (arch == "tight_mot");
    const std::string fam = (arch == "bridge_mllm_dit") ? "internvl_u" : "bagel";
    const auto& op_prof = default_operator_profile();
    const int gamma = (modality_plan == "dit_edit" && has_more_edits) ? 2 : 1;

    CPTGRequestSpec req;
    req.req_id = session_id + "_t" + std::to_string(turn_idx) + "_a" + std::to_string(attempt);
    req.src_gpu = src_gpu;
    req.architecture = arch;
    req.gen_expert_bytes = op_prof.expert_bytes(fam);
    req.dit_step1_ms = op_prof.step_ms(fam, height);
    req.dit_total_steps = dit_steps;
    req.resolution = height;
    req.interleaved_roundtrip_gamma = gamma;
    if (arch == "bridge_mllm_dit") {
        const int64_t eff = std::max<int64_t>(state_bytes, 64LL * 1024LL);
        req.kv_bytes = eff;
        req.bridge_bytes = eff;
    } else {
        req.kv_bytes = std::max<int64_t>(state_bytes, 1LL * kMiB);
    }

    const std::string est_fam = (policy_.model_family == "internvl_u") ? "internvl_u" : "bagel";
    const double est_dit_ms = op_prof.unloaded_dit_turn_ms(est_fam, height, dit_steps);
    auto [dec, ship_mode] = router_->plan_and_reserve_transition(
        policy_.routing_mode,
        policy_.static_und_ranks,
        policy_.static_gen_ranks,
        req,
        est_dit_ms,
        nullptr,
        gpu_busy,
        policy_.use_h2d_pacer,
        now_ms);

    const int dst_gpu = dec.dst_gpu;
    const bool ephemeral_gen_ctx = (ship_mode == "nccl_static" || ship_mode == "shm_static");
    const bool needs_return_ship = ephemeral_gen_ctx && has_more_turns && is_tight_mot;
    bool release_src = false;
    if (ship_mode == "nccl_dual") {
        release_src = true;
    } else if (ephemeral_gen_ctx) {
        release_src = is_tight_mot ? true : (!has_more_turns);
    }

    TurnTransitionPlan out;
    out.decision = std::move(dec);
    out.ship_mode = ship_mode;
    out.src_gpu = src_gpu;
    out.dst_gpu = dst_gpu;
    out.est_dit_ms = est_dit_ms;
    out.reserved_kv_bytes = (ship_mode != "none" && dst_gpu != src_gpu) ? req.effective_state_bytes() : 0;
    out.ephemeral_gen_ctx = ephemeral_gen_ctx;
    out.needs_return_ship = needs_return_ship;
    out.release_src_on_fwd = release_src;
    out.include_cfg_dup_on_wire = (ship_mode == "shm_static");
    out.dit_ctx_id = ephemeral_gen_ctx ? (base_ctx_id + "_gen_t" + std::to_string(turn_idx)) : base_ctx_id;
    return out;
}

nlohmann::json MorphControlEngine::evaluate_session_slos(
    const PolicyConfig& policy,
    const std::vector<double>& ar_ttfts_ms,
    const std::vector<double>& ar_tpots_ms,
    const std::vector<double>& dit_turns_ms,
    const std::vector<int>& dit_heights,
    const std::vector<int>& dit_steps) {
    bool ar_ok = true;
    for (double x : ar_ttfts_ms) {
        if (x > policy.ar_ttft_slo_ms) {
            ar_ok = false;
            break;
        }
    }
    if (ar_ok) {
        for (double x : ar_tpots_ms) {
            if (x > policy.ar_tpot_slo_ms) {
                ar_ok = false;
                break;
            }
        }
    }
    bool dit_ok = true;
    for (size_t i = 0; i < dit_turns_ms.size(); ++i) {
        const int h = (i < dit_heights.size()) ? dit_heights[i] : 512;
        const int st = (i < dit_steps.size()) ? dit_steps[i] : 20;
        if (dit_turns_ms[i] > policy.dit_turn_slo_for(h, st)) {
            dit_ok = false;
            break;
        }
    }
    return {
        {"ar_slo_met", ar_ok},
        {"dit_slo_met", dit_ok},
        {"joint_slo_met", ar_ok && dit_ok},
    };
}

nlohmann::json MorphControlEngine::aggregate_cluster_run_report(
    const PolicyConfig& policy,
    const std::string& suite_name,
    double offered_qps,
    double wall_seconds,
    double host_shmem_delta_gib,
    const nlohmann::json& outcomes_array,
    const nlohmann::json& worker_stats_array) {
    double peak_res_mib = 0.0;
    double peak_pytorch_res_mib = 0.0;
    int worker_ooms = 0;
    int overflow_creates = 0;
    int carves = 0;
    int evict_kv = 0;
    int evict_tr = 0;

    if (worker_stats_array.is_array()) {
        for (const auto& st : worker_stats_array) {
            const double comb =
                (st.contains("combined_physical_mib") && st["combined_physical_mib"].is_number())
                    ? st["combined_physical_mib"].get<double>()
                    : ((st.contains("peak_reserved_mib") && st["peak_reserved_mib"].is_number())
                           ? st["peak_reserved_mib"].get<double>()
                           : 0.0);
            const double pt =
                (st.contains("pytorch_reserved_mib") && st["pytorch_reserved_mib"].is_number())
                    ? st["pytorch_reserved_mib"].get<double>()
                    : ((st.contains("peak_reserved_mib") && st["peak_reserved_mib"].is_number())
                           ? st["peak_reserved_mib"].get<double>()
                           : 0.0);
            peak_res_mib = std::max(peak_res_mib, comb);
            peak_pytorch_res_mib = std::max(peak_pytorch_res_mib, pt);
            worker_ooms += st.value("oom_count", 0);
            if (st.contains("vmm_stats") && st["vmm_stats"].is_object()) {
                const auto& vs = st["vmm_stats"];
                overflow_creates += vs.value("overflow_creates", 0);
                carves += vs.value("super_to_fine_carves", 0);
                evict_kv += vs.value("expert_auto_evictions_for_kv", 0);
                evict_tr += vs.value("expert_auto_evictions_for_transient", 0);
            }
        }
    }

    const size_t n_outcomes = outcomes_array.is_array() ? outcomes_array.size() : 0;
    const size_t n_tot = std::max<size_t>(1, n_outcomes);
    const size_t trim =
        (n_outcomes >= 20) ? static_cast<size_t>(static_cast<double>(n_outcomes) * policy.steady_state_trim_frac) : 0;
    const size_t core_begin = (trim > 0 && trim * 2 < n_outcomes) ? trim : 0;
    const size_t core_end = (trim > 0 && trim * 2 < n_outcomes) ? (n_outcomes - trim) : n_outcomes;
    const size_t n_core = std::max<size_t>(1, core_end - core_begin);

    size_t core_joint_cnt = 0;
    size_t ok_cnt = 0;
    size_t oom_cnt = 0;
    size_t err_cnt = 0;
    size_t joint_cnt = 0;
    size_t ar_cnt = 0;
    size_t dit_cnt = 0;
    int path_a_ships = 0;
    int path_b_morphs = 0;
    int preempted_ar = 0;
    int total_yields = 0;
    int oom_retries = 0;
    int cross_node_ships = 0;
    int64_t max_xnode_bytes = 0;

    std::vector<double> all_ttfts;
    std::vector<double> all_tpots;
    std::vector<double> all_dits;

    for (size_t i = 0; i < n_outcomes; ++i) {
        const auto& o = outcomes_array[i];
        const std::string status = o.value("status", "error");
        const bool is_ok = (status == "ok");
        const bool joint_ok = is_ok && o.value("joint_slo_met", false);
        if (is_ok) {
            ++ok_cnt;
            if (joint_ok) ++joint_cnt;
            if (o.value("ar_slo_met", false)) ++ar_cnt;
            if (o.value("dit_slo_met", false)) ++dit_cnt;
        } else if (status == "oom") {
            ++oom_cnt;
        } else {
            ++err_cnt;
        }
        if (i >= core_begin && i < core_end && joint_ok) {
            ++core_joint_cnt;
        }
        if (o.contains("ar_ttfts_ms") && o["ar_ttfts_ms"].is_array()) {
            for (const auto& v : o["ar_ttfts_ms"]) all_ttfts.push_back(v.get<double>());
        }
        if (o.contains("ar_tpots_ms") && o["ar_tpots_ms"].is_array()) {
            for (const auto& v : o["ar_tpots_ms"]) all_tpots.push_back(v.get<double>());
        }
        if (o.contains("dit_turns_ms") && o["dit_turns_ms"].is_array()) {
            for (const auto& v : o["dit_turns_ms"]) all_dits.push_back(v.get<double>());
        }
        path_a_ships += o.value("path_a_ships", 0);
        path_b_morphs += o.value("path_b_morphs", 0);
        preempted_ar += o.value("preempted_ar_turns", 0);
        total_yields += o.value("total_yields", 0);
        oom_retries += o.value("oom_retries", 0);
        cross_node_ships += o.value("cross_node_ships", 0);
        if (o.contains("cross_node_ship_bytes_list") && o["cross_node_ship_bytes_list"].is_array()) {
            for (const auto& b : o["cross_node_ship_bytes_list"]) {
                max_xnode_bytes = std::max(max_xnode_bytes, b.get<int64_t>());
            }
        }
    }

    nlohmann::json slo_curve = nlohmann::json::object();
    for (const auto& [k_label, k_val] : std::vector<std::pair<std::string, double>>{
             {"1.5x", 1.5}, {"2.0x", 2.0}, {"2.5x", 2.5}, {"3.0x", 3.0}}) {
        size_t k_cnt = 0;
        for (size_t i = core_begin; i < core_end; ++i) {
            if (evaluate_outcome_at_k(policy, outcomes_array[i], k_val)) {
                ++k_cnt;
            }
        }
        const double frac = static_cast<double>(k_cnt) / static_cast<double>(n_core);
        slo_curve[k_label] = {
            {"attainment_pct", round_1(100.0 * frac)},
            {"goodput_rps", round_4(offered_qps * frac)},
        };
    }

    const double wall_s = std::max(1e-6, wall_seconds);
    return {
        {"policy_name", policy.policy_name},
        {"suite_name", suite_name},
        {"num_gpus", policy.num_gpus},
        {"offered_qps", round_3(offered_qps)},
        {"wall_seconds", round_2(wall_s)},
        {"total_sessions", n_outcomes},
        {"completed_sessions", ok_cnt},
        {"oom_sessions", oom_cnt},
        {"error_sessions", err_cnt},
        {"joint_slo_goodput_rps", round_4(offered_qps * (static_cast<double>(joint_cnt) / static_cast<double>(n_tot)))},
        {"raw_wall_goodput_rps", round_4(static_cast<double>(joint_cnt) / wall_s)},
        {"steady_state_goodput_rps",
         round_4(offered_qps * (static_cast<double>(core_joint_cnt) / static_cast<double>(n_core)))},
        {"steady_state_attainment_pct", round_1(100.0 * static_cast<double>(core_joint_cnt) / static_cast<double>(n_core))},
        {"joint_slo_attainment_pct", round_1(100.0 * static_cast<double>(joint_cnt) / static_cast<double>(n_tot))},
        {"ar_slo_attainment_pct", round_1(100.0 * static_cast<double>(ar_cnt) / static_cast<double>(n_tot))},
        {"dit_slo_attainment_pct", round_1(100.0 * static_cast<double>(dit_cnt) / static_cast<double>(n_tot))},
        {"ar_ttft_p50_ms", percentile_rounded(all_ttfts, 50.0)},
        {"ar_ttft_p95_ms", percentile_rounded(all_ttfts, 95.0)},
        {"ar_ttft_p99_ms", percentile_rounded(all_ttfts, 99.0)},
        {"ar_tpot_p50_ms", percentile_rounded(all_tpots, 50.0)},
        {"ar_tpot_p95_ms", percentile_rounded(all_tpots, 95.0)},
        {"ar_tpot_p99_ms", percentile_rounded(all_tpots, 99.0)},
        {"dit_turn_p50_ms", percentile_rounded(all_dits, 50.0)},
        {"dit_turn_p95_ms", percentile_rounded(all_dits, 95.0)},
        {"dit_turn_p99_ms", percentile_rounded(all_dits, 99.0)},
        {"peak_gpu_reserved_mib", round_2(peak_res_mib)},
        {"peak_pytorch_reserved_mib", round_2(peak_pytorch_res_mib)},
        {"host_shmem_delta_gib", host_shmem_delta_gib},
        {"vmm_overflow_creates", overflow_creates},
        {"vmm_super_to_fine_carves", carves},
        {"vmm_expert_evictions_for_kv", evict_kv},
        {"vmm_expert_evictions_for_transient", evict_tr},
        {"path_a_ships", path_a_ships},
        {"path_b_morphs", path_b_morphs},
        {"preempted_ar_count", preempted_ar},
        {"total_yields", total_yields},
        {"worker_oom_count", worker_ooms},
        {"oom_retry_count", oom_retries},
        {"cross_node_ships", cross_node_ships},
        {"max_cross_node_ship_bytes", max_xnode_bytes},
        {"slo_curve_by_k", slo_curve},
        {"status", (err_cnt == 0) ? "ok" : "check_failed"},
        {"sessions", outcomes_array},
    };
}

}  // namespace morphserve::control
