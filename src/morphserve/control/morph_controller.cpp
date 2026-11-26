#include "morphserve/control/morph_controller.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

namespace morphserve::control {

namespace {

inline double round_1(double x) noexcept { return std::round(x * 10.0) / 10.0; }
inline double round_2(double x) noexcept { return std::round(x * 100.0) / 100.0; }
inline double round_3(double x) noexcept { return std::round(x * 1000.0) / 1000.0; }
inline double round_4(double x) noexcept { return std::round(x * 10000.0) / 10000.0; }

double percentile_raw(const std::vector<double>& vals, double p) {
    if (vals.empty()) {
        return 0.0;
    }
    std::vector<double> s = vals;
    std::sort(s.begin(), s.end());
    const double idx = static_cast<double>(s.size() - 1) * (p / 100.0);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = std::min(s.size() - 1, static_cast<size_t>(std::ceil(idx)));
    if (lo == hi) {
        return s[lo];
    }
    const double frac = idx - static_cast<double>(lo);
    return s[lo] * (1.0 - frac) + s[hi] * frac;
}

double mean_raw(const std::vector<double>& vals) {
    if (vals.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (double v : vals) {
        sum += v;
    }
    return sum / static_cast<double>(vals.size());
}

struct ArTpotFlags {
    bool p95_ok = true;
    bool mean_ok = true;
    bool strict_max_ok = true;
};

ArTpotFlags eval_ar_slos(
    const std::vector<double>& ar_ttfts_ms,
    const std::vector<double>& ar_tpots_ms,
    const std::vector<std::vector<double>>& ar_turn_tpots_ms,
    double ttft_cap,
    double tpot_cap) {
    for (double x : ar_ttfts_ms) {
        if (x > ttft_cap) {
            return ArTpotFlags{false, false, false};
        }
    }
    ArTpotFlags flags{true, true, true};
    bool has_turn_buckets = false;
    for (const auto& tv : ar_turn_tpots_ms) {
        if (!tv.empty()) {
            has_turn_buckets = true;
            break;
        }
    }
    if (has_turn_buckets) {
        for (const auto& tv : ar_turn_tpots_ms) {
            if (tv.empty()) continue;
            if (percentile_raw(tv, 95.0) > tpot_cap) {
                flags.p95_ok = false;
            }
            if (mean_raw(tv) > tpot_cap) {
                flags.mean_ok = false;
            }
            for (double x : tv) {
                if (x > tpot_cap) {
                    flags.strict_max_ok = false;
                    break;
                }
            }
        }
    } else if (!ar_tpots_ms.empty()) {
        if (percentile_raw(ar_tpots_ms, 95.0) > tpot_cap) {
            flags.p95_ok = false;
        }
        if (mean_raw(ar_tpots_ms) > tpot_cap) {
            flags.mean_ok = false;
        }
        for (double x : ar_tpots_ms) {
            if (x > tpot_cap) {
                flags.strict_max_ok = false;
                break;
            }
        }
    }
    return flags;
}

bool select_ar_slo_by_mode(const ArTpotFlags& f, const std::string& mode) noexcept {
    if (mode == "turn_mean") {
        return f.mean_ok;
    }
    if (mode == "token_max") {
        return f.strict_max_ok;
    }
    return f.p95_ok;
}

std::vector<double> parse_double_array(const nlohmann::json& j, const char* key) {
    std::vector<double> out;
    if (j.contains(key) && j[key].is_array()) {
        out.reserve(j[key].size());
        for (const auto& v : j[key]) {
            if (v.is_number()) {
                out.push_back(v.get<double>());
            }
        }
    }
    return out;
}

std::vector<std::vector<double>> parse_nested_double_array(const nlohmann::json& j, const char* key) {
    std::vector<std::vector<double>> out;
    if (j.contains(key) && j[key].is_array()) {
        out.reserve(j[key].size());
        for (const auto& row : j[key]) {
            if (!row.is_array()) continue;
            std::vector<double> tv;
            tv.reserve(row.size());
            for (const auto& v : row) {
                if (v.is_number()) {
                    tv.push_back(v.get<double>());
                }
            }
            out.push_back(std::move(tv));
        }
    }
    return out;
}

bool evaluate_outcome_at_k(const PolicyConfig& policy, const nlohmann::json& out, double k) {
    if (out.value("status", "") != "ok") {
        return false;
    }
    const std::string fam = (policy.model_family == "internvl_u") ? "internvl_u" : "bagel";
    const auto slo = default_operator_profile().slo_thresholds_ms(k, fam);
    const double ttft_cap = slo.at("ar_ttft_slo_ms");
    const double tpot_cap = slo.at("ar_tpot_slo_ms");
    const auto ttfts = parse_double_array(out, "ar_ttfts_ms");
    const auto tpots = parse_double_array(out, "ar_tpots_ms");
    const auto turn_tpots = parse_nested_double_array(out, "ar_turn_tpots_ms");
    const ArTpotFlags ar_flags = eval_ar_slos(ttfts, tpots, turn_tpots, ttft_cap, tpot_cap);
    if (!select_ar_slo_by_mode(ar_flags, policy.ar_tpot_slo_mode)) {
        return false;
    }
    if (out.contains("dit_turns_ms") && out["dit_turns_ms"].is_array()) {
        const auto& lats = out["dit_turns_ms"];
        const auto& hs = out.value("dit_turn_heights", nlohmann::json::array());
        const auto& sts = out.value("dit_turn_steps", nlohmann::json::array());
        for (size_t i = 0; i < lats.size(); ++i) {
            if (!lats[i].is_number()) continue;
            const int h = (i < hs.size() && hs[i].is_number()) ? hs[i].get<int>() : 512;
            const int st = (i < sts.size() && sts[i].is_number()) ? sts[i].get<int>() : 20;
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

double MorphControlEngine::deterministic_think_dist_s(
    const std::string& session_id,
    int turn_idx,
    double lo_s,
    double hi_s,
    const std::string& dist,
    double median_s,
    double sigma) {
    if (hi_s <= lo_s) {
        return std::max(0.0, lo_s);
    }
    const double eff_lo = std::max(0.0, lo_s);
    const double eff_hi = std::max(eff_lo, hi_s);
    if (dist == "constant") {
        return std::clamp(std::max(0.0, median_s), eff_lo, eff_hi);
    }
    if (dist == "uniform") {
        return deterministic_think_s(session_id, turn_idx, lo_s, hi_s);
    }
    constexpr double kPi = 3.14159265358979323846;
    const std::string key = session_id + ":" + std::to_string(turn_idx);
    const uint32_t h1 = md5_prefix_u32(key + ":a");
    const uint32_t h2 = md5_prefix_u32(key + ":b");
    const double u1 = std::clamp(static_cast<double>(h1) / 4294967296.0, 1e-6, 1.0 - 1e-6);
    const double u2 = std::clamp(static_cast<double>(h2) / 4294967296.0, 1e-6, 1.0 - 1e-6);
    const double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
    const double val = std::max(1e-6, median_s) * std::exp(std::max(0.0, sigma) * z);
    return std::clamp(val, eff_lo, eff_hi);
}

double MorphControlEngine::deterministic_think_dist_ms_sim(
    int sid,
    int turn_idx,
    double lo_s,
    double hi_s,
    const std::string& dist,
    double median_s,
    double sigma) {
    if (hi_s <= lo_s) {
        return std::max(0.0, lo_s) * 1000.0;
    }
    const double eff_lo = std::max(0.0, lo_s);
    const double eff_hi = std::max(eff_lo, hi_s);
    if (dist == "constant") {
        return std::clamp(std::max(0.0, median_s), eff_lo, eff_hi) * 1000.0;
    }
    if (dist == "uniform") {
        return deterministic_think_ms_sim(sid, turn_idx, lo_s, hi_s);
    }
    constexpr double kPi = 3.14159265358979323846;
    const std::string key = "sim_" + std::to_string(sid) + "_" + std::to_string(turn_idx);
    const uint32_t h1 = md5_prefix_u32(key + ":a");
    const uint32_t h2 = md5_prefix_u32(key + ":b");
    const double u1 = std::clamp(static_cast<double>(h1) / 4294967296.0, 1e-6, 1.0 - 1e-6);
    const double u2 = std::clamp(static_cast<double>(h2) / 4294967296.0, 1e-6, 1.0 - 1e-6);
    const double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
    const double val = std::max(1e-6, median_s) * std::exp(std::max(0.0, sigma) * z);
    return std::clamp(val, eff_lo, eff_hi) * 1000.0;
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
    const std::vector<int>& dit_steps,
    const std::vector<std::vector<double>>& ar_turn_tpots_ms) {
    const ArTpotFlags ar_flags = eval_ar_slos(
        ar_ttfts_ms, ar_tpots_ms, ar_turn_tpots_ms, policy.ar_ttft_slo_ms, policy.ar_tpot_slo_ms);
    const bool ar_ok = select_ar_slo_by_mode(ar_flags, policy.ar_tpot_slo_mode);

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
        {"ar_slo_p95_tpot_met", ar_flags.p95_ok},
        {"ar_slo_mean_tpot_met", ar_flags.mean_ok},
        {"ar_slo_strict_max_tpot_met", ar_flags.strict_max_ok},
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
    double peak_nvml_mib = 0.0;
    double peak_pytorch_res_mib = 0.0;
    int worker_ooms = 0;
    int kv_host_swap_outs = 0;
    int overflow_creates = 0;
    int carves = 0;
    int evict_kv = 0;
    int evict_tr = 0;

    if (worker_stats_array.is_array()) {
        for (const auto& st : worker_stats_array) {
            const double nvml_pk =
                (st.contains("nvml_peak_used_mib") && st["nvml_peak_used_mib"].is_number())
                    ? st["nvml_peak_used_mib"].get<double>()
                    : 0.0;
            const double comb =
                (nvml_pk > 0.0)
                    ? nvml_pk
                    : ((st.contains("combined_physical_mib") && st["combined_physical_mib"].is_number())
                           ? st["combined_physical_mib"].get<double>()
                           : ((st.contains("peak_reserved_mib") && st["peak_reserved_mib"].is_number())
                                  ? st["peak_reserved_mib"].get<double>()
                                  : 0.0));
            const double pt =
                (st.contains("pytorch_reserved_mib") && st["pytorch_reserved_mib"].is_number())
                    ? st["pytorch_reserved_mib"].get<double>()
                    : ((st.contains("peak_reserved_mib") && st["peak_reserved_mib"].is_number())
                           ? st["peak_reserved_mib"].get<double>()
                           : 0.0);
            peak_res_mib = std::max(peak_res_mib, comb);
            peak_nvml_mib = std::max(peak_nvml_mib, (nvml_pk > 0.0) ? nvml_pk : comb);
            peak_pytorch_res_mib = std::max(peak_pytorch_res_mib, pt);
            worker_ooms += st.value("oom_count", 0);
            if (st.contains("kv_host_swap_outs") && st["kv_host_swap_outs"].is_number()) {
                kv_host_swap_outs += st["kv_host_swap_outs"].get<int>();
            } else if (st.contains("vmm_stats") && st["vmm_stats"].is_object()) {
                kv_host_swap_outs += st["vmm_stats"].value("kv_swap_outs_total", 0);
            }
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

    std::vector<double> arrival_offsets(n_outcomes, 0.0);
    double max_arrival = 0.0;
    for (size_t i = 0; i < n_outcomes; ++i) {
        const auto& o = outcomes_array[i];
        const double arr = (o.contains("arrival_offset_s") && o["arrival_offset_s"].is_number())
                               ? o["arrival_offset_s"].get<double>()
                               : 0.0;
        arrival_offsets[i] = arr;
        if (arr > max_arrival) {
            max_arrival = arr;
        }
    }

    std::vector<size_t> core_indices;
    std::string eff_window_mode = "session_trim";
    double win_start_s = 0.0;
    double win_end_s = 0.0;

    if (n_outcomes > 0 && policy.steady_window_mode == "time" &&
        max_arrival >= policy.warmup_window_s + 30.0) {
        const double ws_s = policy.warmup_window_s;
        const double we_s = policy.warmup_window_s + policy.steady_window_s;
        for (size_t i = 0; i < n_outcomes; ++i) {
            if (arrival_offsets[i] >= ws_s - 1e-9 && arrival_offsets[i] <= we_s + 1e-9) {
                core_indices.push_back(i);
            }
        }
        if (!core_indices.empty()) {
            eff_window_mode = "time";
            win_start_s = ws_s;
            win_end_s = we_s;
        }
    }
    if (core_indices.empty() && n_outcomes > 0) {
        eff_window_mode = "session_trim";
        const size_t trim =
            (n_outcomes >= 20)
                ? static_cast<size_t>(static_cast<double>(n_outcomes) * policy.steady_state_trim_frac)
                : 0;
        const size_t core_begin = (trim > 0 && trim * 2 < n_outcomes) ? trim : 0;
        const size_t core_end = (trim > 0 && trim * 2 < n_outcomes) ? (n_outcomes - trim) : n_outcomes;
        core_indices.reserve(core_end - core_begin);
        for (size_t i = core_begin; i < core_end; ++i) {
            core_indices.push_back(i);
        }
        if (!core_indices.empty()) {
            win_start_s = arrival_offsets[core_indices.front()];
            win_end_s = arrival_offsets[core_indices.back()];
        }
    }

    std::vector<uint8_t> in_core(n_outcomes, 0);
    for (size_t idx : core_indices) {
        in_core[idx] = 1;
    }
    const size_t steady_sessions_cnt = core_indices.size();
    const size_t n_core = std::max<size_t>(1, steady_sessions_cnt);

    // Full-trace accumulators
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

    // Steady-state core accumulators
    size_t core_ok_cnt = 0;
    size_t core_oom_cnt = 0;
    size_t core_err_cnt = 0;
    size_t core_joint_cnt = 0;
    size_t core_ar_cnt = 0;
    size_t core_ar_mean_cnt = 0;
    size_t core_ar_strict_max_cnt = 0;
    size_t core_dit_cnt = 0;

    std::vector<double> core_ok_e2es;
    std::vector<double> core_ok_active_jcts;
    std::vector<double> core_ok_ttfts;
    std::vector<double> core_ok_tpots;
    std::vector<double> core_ok_dits;

    struct TurnAcc {
        int total = 0;
        int ok = 0;
        int oom = 0;
        std::vector<double> ttfts;
        std::vector<double> tpots;
        std::vector<double> dits;
        std::vector<double> kv_mibs;
    };
    std::map<int, TurnAcc> turn_accs;

    struct BucketAcc {
        int sessions = 0;
        int ok = 0;
        int oom = 0;
        int slo_met = 0;
        double alpha_sum = 0.0;
        int alpha_cnt = 0;
        int path_a_ships = 0;
        int path_b_morphs = 0;
        int yields = 0;
        std::map<std::string, int> phase_counts;
        std::vector<double> ttfts;
        std::vector<double> dits;
    };
    std::map<int, BucketAcc> ts_buckets;

    for (size_t i = 0; i < n_outcomes; ++i) {
        const auto& o = outcomes_array[i];
        const std::string status = o.value("status", "error");
        const bool is_ok = (status == "ok");
        const bool is_oom = (status == "oom");
        const bool core = (in_core[i] != 0);

        const auto o_ttfts = parse_double_array(o, "ar_ttfts_ms");
        const auto o_tpots = parse_double_array(o, "ar_tpots_ms");
        const auto o_turn_tpots = parse_nested_double_array(o, "ar_turn_tpots_ms");
        const auto o_dits = parse_double_array(o, "dit_turns_ms");

        bool ar_primary_ok = false;
        bool ar_mean_ok = false;
        bool ar_strict_max_ok = false;
        bool dit_ok = false;
        bool joint_ok = false;

        if (is_ok) {
            const ArTpotFlags ar_flags = eval_ar_slos(
                o_ttfts, o_tpots, o_turn_tpots, policy.ar_ttft_slo_ms, policy.ar_tpot_slo_ms);
            ar_primary_ok = (o.contains("ar_slo_met") && o["ar_slo_met"].is_boolean())
                                ? o["ar_slo_met"].get<bool>()
                                : select_ar_slo_by_mode(ar_flags, policy.ar_tpot_slo_mode);
            ar_mean_ok = (o.contains("ar_slo_mean_tpot_met") && o["ar_slo_mean_tpot_met"].is_boolean())
                             ? o["ar_slo_mean_tpot_met"].get<bool>()
                             : ar_flags.mean_ok;
            ar_strict_max_ok =
                (o.contains("ar_slo_strict_max_tpot_met") && o["ar_slo_strict_max_tpot_met"].is_boolean())
                    ? o["ar_slo_strict_max_tpot_met"].get<bool>()
                    : ar_flags.strict_max_ok;
            dit_ok = o.value("dit_slo_met", false);
            joint_ok = (o.contains("joint_slo_met") && o["joint_slo_met"].is_boolean())
                           ? o["joint_slo_met"].get<bool>()
                           : (ar_primary_ok && dit_ok);

            ++ok_cnt;
            if (joint_ok) ++joint_cnt;
            if (ar_primary_ok) ++ar_cnt;
            if (dit_ok) ++dit_cnt;

            if (core) {
                ++core_ok_cnt;
                if (joint_ok) ++core_joint_cnt;
                if (ar_primary_ok) ++core_ar_cnt;
                if (ar_mean_ok) ++core_ar_mean_cnt;
                if (ar_strict_max_ok) ++core_ar_strict_max_cnt;
                if (dit_ok) ++core_dit_cnt;

                if (o.contains("e2e_ms") && o["e2e_ms"].is_number()) {
                    const double e2e = o["e2e_ms"].get<double>();
                    core_ok_e2es.push_back(e2e);
                    if (o.contains("active_jct_ms") && o["active_jct_ms"].is_number()) {
                        core_ok_active_jcts.push_back(std::max(0.0, o["active_jct_ms"].get<double>()));
                    } else {
                        const double think_ms =
                            (o.contains("think_time_ms") && o["think_time_ms"].is_number())
                                ? o["think_time_ms"].get<double>()
                                : 0.0;
                        core_ok_active_jcts.push_back(std::max(0.0, e2e - think_ms));
                    }
                } else if (o.contains("active_jct_ms") && o["active_jct_ms"].is_number()) {
                    core_ok_active_jcts.push_back(std::max(0.0, o["active_jct_ms"].get<double>()));
                }
                for (double v : o_ttfts) core_ok_ttfts.push_back(v);
                for (double v : o_tpots) core_ok_tpots.push_back(v);
                for (double v : o_dits) core_ok_dits.push_back(v);
            }
        } else if (is_oom) {
            ++oom_cnt;
            if (core) ++core_oom_cnt;
        } else {
            ++err_cnt;
            if (core) ++core_err_cnt;
        }

        const int o_ships = o.value("path_a_ships", 0);
        const int o_morphs = o.value("path_b_morphs", 0);
        const int o_yields = o.value("total_yields", 0);
        path_a_ships += o_ships;
        path_b_morphs += o_morphs;
        preempted_ar += o.value("preempted_ar_turns", 0);
        total_yields += o_yields;
        oom_retries += o.value("oom_retries", 0);
        cross_node_ships += o.value("cross_node_ships", 0);
        if (o.contains("cross_node_ship_bytes_list") && o["cross_node_ship_bytes_list"].is_array()) {
            for (const auto& b : o["cross_node_ship_bytes_list"]) {
                if (b.is_number()) {
                    max_xnode_bytes = std::max(max_xnode_bytes, b.get<int64_t>());
                }
            }
        }

        // Per-turn stats strictly over steady-state core outcomes
        if (core && o.contains("turn_records") && o["turn_records"].is_array()) {
            const auto& trecs = o["turn_records"];
            for (size_t pos = 0; pos < trecs.size(); ++pos) {
                const auto& tr = trecs[pos];
                if (!tr.is_object()) continue;
                const int tidx = tr.value("turn_idx", static_cast<int>(pos));
                auto& ta = turn_accs[tidx];
                ++ta.total;
                const std::string t_st = tr.value("status", "ok");
                if (t_st == "ok") {
                    ++ta.ok;
                    if (tr.contains("ar_ttft_ms") && tr["ar_ttft_ms"].is_number()) {
                        ta.ttfts.push_back(tr["ar_ttft_ms"].get<double>());
                    } else if (tr.contains("ttft_ms") && tr["ttft_ms"].is_number()) {
                        ta.ttfts.push_back(tr["ttft_ms"].get<double>());
                    }
                    if (tr.contains("ar_tpots_ms") && tr["ar_tpots_ms"].is_array()) {
                        for (const auto& v : tr["ar_tpots_ms"]) {
                            if (v.is_number()) ta.tpots.push_back(v.get<double>());
                        }
                    } else if (tr.contains("ar_tpot_p95_ms") && tr["ar_tpot_p95_ms"].is_number()) {
                        ta.tpots.push_back(tr["ar_tpot_p95_ms"].get<double>());
                    } else if (tr.contains("ar_tpot_ms") && tr["ar_tpot_ms"].is_number()) {
                        ta.tpots.push_back(tr["ar_tpot_ms"].get<double>());
                    }
                    if (tr.contains("dit_ms") && tr["dit_ms"].is_number()) {
                        ta.dits.push_back(tr["dit_ms"].get<double>());
                    } else if (tr.contains("dit_turn_ms") && tr["dit_turn_ms"].is_number()) {
                        ta.dits.push_back(tr["dit_turn_ms"].get<double>());
                    } else if (tr.contains("dit_lat_ms") && tr["dit_lat_ms"].is_number()) {
                        ta.dits.push_back(tr["dit_lat_ms"].get<double>());
                    }
                    if (tr.contains("kv_mib") && tr["kv_mib"].is_number()) {
                        ta.kv_mibs.push_back(tr["kv_mib"].get<double>());
                    } else if (tr.contains("kv_bytes") && tr["kv_bytes"].is_number()) {
                        ta.kv_mibs.push_back(tr["kv_bytes"].get<double>() / static_cast<double>(kMiB));
                    }
                } else if (t_st == "oom") {
                    ++ta.oom;
                }
            }
        }

        // 10-second time-series bucket over ALL outcomes
        const int bidx = static_cast<int>(std::floor(std::max(0.0, arrival_offsets[i]) / 10.0));
        auto& ba = ts_buckets[bidx];
        ++ba.sessions;
        if (is_ok) ++ba.ok;
        if (is_oom) ++ba.oom;
        if (joint_ok) ++ba.slo_met;
        if (o.contains("resident_alpha_sample") && o["resident_alpha_sample"].is_number()) {
            ba.alpha_sum += o["resident_alpha_sample"].get<double>();
            ++ba.alpha_cnt;
        }
        ba.path_a_ships += o_ships;
        ba.path_b_morphs += o_morphs;
        ba.yields += o_yields;
        if (o.contains("phase_label") && o["phase_label"].is_string()) {
            const auto& pl = o["phase_label"].get_ref<const std::string&>();
            if (!pl.empty()) {
                ++ba.phase_counts[pl];
            }
        }
        if (is_ok) {
            for (double v : o_ttfts) ba.ttfts.push_back(v);
            for (double v : o_dits) ba.dits.push_back(v);
        }
    }

    // Improper empirical CDF F_all(t) over steady-state core requests/turns (OOM/failed mass placed at +inf):
    // if target quantile `q_pct > survival_pct`, F_all^{-1}(q) = +inf (`nullptr` in JSON).
    auto failure_aware_quantile = [](const std::vector<double>& survivors, size_t total_units, double q_pct) -> nlohmann::json {
        if (total_units == 0 || survivors.empty()) {
            return nullptr;
        }
        const double surv_frac = static_cast<double>(survivors.size()) / static_cast<double>(total_units);
        const double q_frac = q_pct / 100.0;
        if (q_frac > surv_frac + 1e-9) {
            return nullptr;
        }
        const double eff_survivor_pct = std::min(100.0, (q_frac / std::max(1e-9, surv_frac)) * 100.0);
        return percentile_rounded(survivors, eff_survivor_pct);
    };

    auto build_improper_cdf_points = [&](const std::vector<double>& survivors, size_t total_units) -> nlohmann::json {
        nlohmann::json pts = nlohmann::json::array();
        for (double q_pct : {5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0, 90.0, 95.0, 99.0}) {
            pts.push_back({
                {"p", q_pct},
                {"ms", failure_aware_quantile(survivors, total_units, q_pct)},
            });
        }
        return pts;
    };

    const size_t core_failed_sess = core_oom_cnt + core_err_cnt;
    const size_t core_total_dit_units = core_ok_dits.size() + core_failed_sess;
    const double survival_rate_pct =
        (n_outcomes == 0) ? 100.0 : round_1(100.0 * static_cast<double>(ok_cnt) / static_cast<double>(n_tot));
    const double steady_survival_pct =
        (steady_sessions_cnt == 0)
            ? 100.0
            : round_1(100.0 * static_cast<double>(core_ok_cnt) / static_cast<double>(n_core));
    const double dit_survival_pct =
        (core_total_dit_units == 0)
            ? 100.0
            : round_1(100.0 * static_cast<double>(core_ok_dits.size()) / static_cast<double>(core_total_dit_units));

    nlohmann::json slo_curve = nlohmann::json::object();
    for (const auto& [k_label, k_val] : std::vector<std::pair<std::string, double>>{
             {"1.5x", 1.5}, {"2.0x", 2.0}, {"2.5x", 2.5}, {"3.0x", 3.0}}) {
        size_t k_cnt = 0;
        for (size_t idx : core_indices) {
            if (evaluate_outcome_at_k(policy, outcomes_array[idx], k_val)) {
                ++k_cnt;
            }
        }
        const double frac = static_cast<double>(k_cnt) / static_cast<double>(n_core);
        const double eff_rps = round_4(offered_qps * frac);
        slo_curve[k_label] = {
            {"attainment_pct", round_1(100.0 * frac)},
            {"effective_slo_throughput_rps", eff_rps},
            {"goodput_rps", eff_rps},
        };
    }

    nlohmann::json per_turn_stats = nlohmann::json::object();
    for (const auto& [tidx, acc] : turn_accs) {
        const double surv =
            (acc.total == 0) ? 100.0 : round_1(100.0 * static_cast<double>(acc.ok) / static_cast<double>(acc.total));
        per_turn_stats["turn_" + std::to_string(tidx)] = {
            {"total", acc.total},
            {"ok", acc.ok},
            {"oom", acc.oom},
            {"survival_pct", surv},
            {"ar_ttft_p95_ms", percentile_rounded(acc.ttfts, 95.0)},
            {"ar_tpot_p95_ms", percentile_rounded(acc.tpots, 95.0)},
            {"dit_p95_ms", percentile_rounded(acc.dits, 95.0)},
            {"kv_mib_mean", round_2(mean_raw(acc.kv_mibs))},
            {"kv_mib_p95", percentile_rounded(acc.kv_mibs, 95.0)},
        };
    }

    nlohmann::json time_series_10s = nlohmann::json::array();
    for (const auto& [bidx, ba] : ts_buckets) {
        std::string maj_phase;
        int best_cnt = 0;
        for (const auto& [pl, cnt] : ba.phase_counts) {
            if (cnt > best_cnt) {
                best_cnt = cnt;
                maj_phase = pl;
            }
        }
        const double b_att =
            (ba.sessions == 0) ? 0.0 : round_1(100.0 * static_cast<double>(ba.slo_met) / static_cast<double>(ba.sessions));
        const nlohmann::json mean_alpha_j =
            (ba.alpha_cnt > 0)
                ? nlohmann::json(round_3(ba.alpha_sum / static_cast<double>(ba.alpha_cnt)))
                : nlohmann::json(nullptr);
        time_series_10s.push_back({
            {"t_start_s", static_cast<double>(bidx) * 10.0},
            {"t_end_s", static_cast<double>(bidx + 1) * 10.0},
            {"phase_label", maj_phase},
            {"sessions", ba.sessions},
            {"ok", ba.ok},
            {"oom", ba.oom},
            {"slo_met", ba.slo_met},
            {"slo_attainment_pct", b_att},
            {"mean_alpha", mean_alpha_j},
            {"path_a_ships", ba.path_a_ships},
            {"path_b_morphs", ba.path_b_morphs},
            {"yields", ba.yields},
            {"ar_ttft_p95_ms", percentile_rounded(ba.ttfts, 95.0)},
            {"dit_p95_ms", percentile_rounded(ba.dits, 95.0)},
        });
    }

    const double wall_s = std::max(1e-6, wall_seconds);
    const double eff_slo_throughput_rps =
        round_4(offered_qps * (static_cast<double>(core_joint_cnt) / static_cast<double>(n_core)));

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
        {"survival_rate_pct", survival_rate_pct},
        {"steady_state_window_mode", eff_window_mode},
        {"steady_window_start_s", round_2(win_start_s)},
        {"steady_window_end_s", round_2(win_end_s)},
        {"steady_state_sessions", steady_sessions_cnt},
        {"steady_state_survival_pct", steady_survival_pct},
        {"steady_oom_sessions", core_oom_cnt},
        {"dit_turn_survival_pct", dit_survival_pct},
        {"joint_slo_goodput_rps", round_4(offered_qps * (static_cast<double>(joint_cnt) / static_cast<double>(n_tot)))},
        {"raw_wall_goodput_rps", round_4(static_cast<double>(joint_cnt) / wall_s)},
        {"effective_slo_throughput_rps", eff_slo_throughput_rps},
        {"steady_state_goodput_rps", eff_slo_throughput_rps},
        {"steady_state_attainment_pct",
         round_1(100.0 * static_cast<double>(core_joint_cnt) / static_cast<double>(n_core))},
        {"steady_ar_slo_attainment_pct",
         round_1(100.0 * static_cast<double>(core_ar_cnt) / static_cast<double>(n_core))},
        {"steady_ar_slo_mean_pct",
         round_1(100.0 * static_cast<double>(core_ar_mean_cnt) / static_cast<double>(n_core))},
        {"steady_ar_slo_strict_max_pct",
         round_1(100.0 * static_cast<double>(core_ar_strict_max_cnt) / static_cast<double>(n_core))},
        {"steady_dit_slo_attainment_pct",
         round_1(100.0 * static_cast<double>(core_dit_cnt) / static_cast<double>(n_core))},
        {"joint_slo_attainment_pct", round_1(100.0 * static_cast<double>(joint_cnt) / static_cast<double>(n_tot))},
        {"ar_slo_attainment_pct", round_1(100.0 * static_cast<double>(ar_cnt) / static_cast<double>(n_tot))},
        {"dit_slo_attainment_pct", round_1(100.0 * static_cast<double>(dit_cnt) / static_cast<double>(n_tot))},
        {"ar_ttft_p50_ms", percentile_rounded(core_ok_ttfts, 50.0)},
        {"ar_ttft_p95_ms", percentile_rounded(core_ok_ttfts, 95.0)},
        {"ar_ttft_p99_ms", percentile_rounded(core_ok_ttfts, 99.0)},
        {"ar_tpot_p50_ms", percentile_rounded(core_ok_tpots, 50.0)},
        {"ar_tpot_p95_ms", percentile_rounded(core_ok_tpots, 95.0)},
        {"ar_tpot_p99_ms", percentile_rounded(core_ok_tpots, 99.0)},
        {"dit_turn_p50_ms", percentile_rounded(core_ok_dits, 50.0)},
        {"dit_turn_p95_ms", percentile_rounded(core_ok_dits, 95.0)},
        {"dit_turn_p99_ms", percentile_rounded(core_ok_dits, 99.0)},
        {"session_e2e_p50_ms", percentile_rounded(core_ok_e2es, 50.0)},
        {"session_e2e_p95_ms", percentile_rounded(core_ok_e2es, 95.0)},
        {"session_e2e_p99_ms", percentile_rounded(core_ok_e2es, 99.0)},
        {"active_jct_p50_ms", percentile_rounded(core_ok_active_jcts, 50.0)},
        {"active_jct_p95_ms", percentile_rounded(core_ok_active_jcts, 95.0)},
        {"active_jct_p99_ms", percentile_rounded(core_ok_active_jcts, 99.0)},
        {"all_session_active_jct_p50_ms", failure_aware_quantile(core_ok_active_jcts, steady_sessions_cnt, 50.0)},
        {"all_session_active_jct_p90_ms", failure_aware_quantile(core_ok_active_jcts, steady_sessions_cnt, 90.0)},
        {"all_session_active_jct_p95_ms", failure_aware_quantile(core_ok_active_jcts, steady_sessions_cnt, 95.0)},
        {"all_session_e2e_p50_ms", failure_aware_quantile(core_ok_e2es, steady_sessions_cnt, 50.0)},
        {"all_session_e2e_p90_ms", failure_aware_quantile(core_ok_e2es, steady_sessions_cnt, 90.0)},
        {"all_session_e2e_p95_ms", failure_aware_quantile(core_ok_e2es, steady_sessions_cnt, 95.0)},
        {"all_session_dit_p50_ms", failure_aware_quantile(core_ok_dits, core_total_dit_units, 50.0)},
        {"all_session_dit_p90_ms", failure_aware_quantile(core_ok_dits, core_total_dit_units, 90.0)},
        {"all_session_dit_p95_ms", failure_aware_quantile(core_ok_dits, core_total_dit_units, 95.0)},
        {"active_jct_cdf_points", build_improper_cdf_points(core_ok_active_jcts, steady_sessions_cnt)},
        {"e2e_cdf_points", build_improper_cdf_points(core_ok_e2es, steady_sessions_cnt)},
        {"dit_cdf_points", build_improper_cdf_points(core_ok_dits, core_total_dit_units)},
        {"peak_gpu_reserved_mib", round_2(peak_res_mib)},
        {"peak_nvml_used_mib", round_2(peak_nvml_mib)},
        {"peak_pytorch_reserved_mib", round_2(peak_pytorch_res_mib)},
        {"host_shmem_delta_gib", host_shmem_delta_gib},
        {"kv_host_swap_outs", kv_host_swap_outs},
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
        {"per_turn_stats", per_turn_stats},
        {"time_series_10s", time_series_10s},
        {"status", (err_cnt == 0) ? "ok" : "check_failed"},
        {"sessions", outcomes_array},
    };
}

}  // namespace morphserve::control
