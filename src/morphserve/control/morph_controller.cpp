#include "morphserve/control/morph_controller.hpp"

#include "morphserve/control/hash_util.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <map>
#include <vector>

namespace morphserve::control {

namespace {

struct ArTpotFlags {
    bool p95_ok = true;
    bool mean_ok = true;
    bool strict_max_ok = true;
};

ArTpotFlags eval_ar_slos(
    const std::vector<double>& ar_ttfts_ms,
    const std::vector<double>& ar_tpots_ms,
    const std::vector<std::vector<double>>& ar_turn_tpots_ms,
    double default_ttft_cap,
    double tpot_cap,
    const std::vector<double>& per_turn_ttft_caps = {}) {
    for (size_t i = 0; i < ar_ttfts_ms.size(); ++i) {
        const double cap = (i < per_turn_ttft_caps.size() && per_turn_ttft_caps[i] > 0.0)
                               ? per_turn_ttft_caps[i]
                               : default_ttft_cap;
        if (ar_ttfts_ms[i] > cap) {
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
    if (mode == "token_max" || mode == "strict_max") {
        return f.strict_max_ok;
    }
    return f.p95_ok;
}

std::vector<double> parse_double_array(const nlohmann::json& j, const char* key) {
    std::vector<double> out;
    if (j.is_object() && j.contains(key) && j[key].is_array()) {
        out.reserve(j[key].size());
        for (const auto& v : j[key]) {
            if (v.is_number()) {
                out.push_back(v.get<double>());
            }
        }
    }
    return out;
}

std::vector<int> parse_int_array(const nlohmann::json& j, const char* key) {
    std::vector<int> out;
    if (j.is_object() && j.contains(key) && j[key].is_array()) {
        out.reserve(j[key].size());
        for (const auto& v : j[key]) {
            if (v.is_number()) {
                out.push_back(v.get<int>());
            }
        }
    }
    return out;
}

std::vector<std::vector<double>> parse_nested_double_array(const nlohmann::json& j, const char* key) {
    std::vector<std::vector<double>> out;
    if (j.is_object() && j.contains(key) && j[key].is_array()) {
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

std::optional<double> first_number_opt(const nlohmann::json& j, std::initializer_list<const char*> keys) {
    if (!j.is_object()) return std::nullopt;
    for (const char* k : keys) {
        if (j.contains(k) && j[k].is_number()) {
            return j[k].get<double>();
        }
    }
    return std::nullopt;
}

double sample_think_dist_s(
    const std::string& hash_key,
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
        const uint32_t h = md5_prefix_u32(hash_key);
        const double frac = static_cast<double>(h % 10000U) / 10000.0;
        return lo_s + frac * (hi_s - lo_s);
    }
    constexpr double kPi = 3.14159265358979323846;
    const uint32_t h1 = md5_prefix_u32(hash_key + ":a");
    const uint32_t h2 = md5_prefix_u32(hash_key + ":b");
    const double u1 = std::clamp(static_cast<double>(h1) / 4294967296.0, 1e-6, 1.0 - 1e-6);
    const double u2 = std::clamp(static_cast<double>(h2) / 4294967296.0, 1e-6, 1.0 - 1e-6);
    const double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
    const double val = std::max(1e-6, median_s) * std::exp(std::max(0.0, sigma) * z);
    return std::clamp(val, eff_lo, eff_hi);
}

std::vector<int> parse_bool_or_int_array(const nlohmann::json& j, const char* key) {
    std::vector<int> out;
    if (j.is_object() && j.contains(key) && j[key].is_array()) {
        out.reserve(j[key].size());
        for (const auto& v : j[key]) {
            if (v.is_boolean()) {
                out.push_back(v.get<bool>() ? 1 : 0);
            } else if (v.is_number()) {
                out.push_back(v.get<int>());
            }
        }
    }
    return out;
}

int default_dit_steps_for(const PolicyConfig& policy, int height) noexcept {
    (void)height;
    return (policy.model_family == "internvl_u") ? 16 : 20;
}

double resolve_turn_ar_ttft_cap_ms(
    const PolicyConfig& policy,
    const TurnRecordInput& tr,
    const std::string& session_quadrant,
    int prev_dit_height,
    std::optional<double> k_ar_opt,
    double slo_time_scale) {
    bool has_img = false;
    int n_img = 0;
    const bool has_explicit =
        tr.has_image.has_value() || tr.num_images.has_value() || tr.use_prev_image;
    if (has_explicit) {
        if (tr.has_image.has_value()) {
            has_img = *tr.has_image;
        }
        if (tr.num_images.has_value()) {
            n_img = std::max(0, *tr.num_images);
            if (n_img > 0) has_img = true;
        }
        if (tr.use_prev_image) {
            has_img = true;
            n_img = std::max(1, n_img);
        }
        if (has_img && n_img <= 0) {
            n_img = 1;
        }
    } else if (session_quadrant == "multi_image_vqa") {
        has_img = true;
        n_img = 2;
    } else if (prev_dit_height > 0) {
        has_img = true;
        n_img = 1;
    } else if (tr.modality_plan == "ar_only" && tr.kv_mib.has_value() && *tr.kv_mib >= 100.0) {
        has_img = true;
        n_img = 1;
    } else {
        has_img = false;
        n_img = 0;
    }
    const int res = (tr.image_res > 0)
                        ? tr.image_res
                        : ((tr.height > 0) ? tr.height : ((prev_dit_height > 0) ? prev_dit_height : 512));
    return policy.ar_ttft_slo_for(has_img, n_img, res, k_ar_opt) * slo_time_scale;
}

std::vector<double> build_session_ar_ttft_caps(
    const PolicyConfig& policy,
    const SessionOutcomeInput& out,
    std::optional<double> k_ar_opt,
    double slo_time_scale) {
    std::vector<double> caps;
    if (!out.ar_turn_has_image.empty() || !out.ar_turn_num_images.empty()) {
        const size_t n = std::max({out.ar_ttfts_ms.size(), out.ar_turn_has_image.size(), out.ar_turn_num_images.size()});
        caps.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            int n_img = (i < out.ar_turn_num_images.size()) ? std::max(0, out.ar_turn_num_images[i]) : 0;
            bool has_img = (i < out.ar_turn_has_image.size()) ? (out.ar_turn_has_image[i] != 0) : (n_img > 0);
            if (has_img && n_img <= 0) n_img = 1;
            if (n_img > 0) has_img = true;
            const int res = (i < out.ar_turn_image_res.size() && out.ar_turn_image_res[i] > 0)
                                ? out.ar_turn_image_res[i]
                                : 512;
            caps.push_back(policy.ar_ttft_slo_for(has_img, n_img, res, k_ar_opt) * slo_time_scale);
        }
        return caps;
    }
    if (!out.turn_records.empty()) {
        int prev_dit_height = 0;
        for (const auto& tr : out.turn_records) {
            const bool has_ar =
                (tr.modality_plan == "ar_only" || tr.modality_plan == "cot_then_dit" || tr.ar_ttft_ms.has_value());
            if (has_ar) {
                caps.push_back(resolve_turn_ar_ttft_cap_ms(
                    policy, tr, out.quadrant, prev_dit_height, k_ar_opt, slo_time_scale));
            }
            const bool has_dit =
                (tr.modality_plan == "dit_t2i" || tr.modality_plan == "dit_edit" ||
                 tr.modality_plan == "cot_then_dit" || tr.dit_ms.has_value());
            if (has_dit) {
                prev_dit_height = (tr.height > 0) ? tr.height : 512;
            }
        }
        return caps;
    }
    if (!out.quadrant.empty() && !out.ar_ttfts_ms.empty()) {
        caps.reserve(out.ar_ttfts_ms.size());
        const bool is_vqa = (out.quadrant == "multi_image_vqa");
        const bool is_interleaved = (out.quadrant == "interleaved_co_creation");
        for (size_t i = 0; i < out.ar_ttfts_ms.size(); ++i) {
            const bool has_img = is_vqa || (is_interleaved && i > 0);
            const int n_img = is_vqa ? 2 : (has_img ? 1 : 0);
            const int res = (!out.dit_turn_heights.empty() && out.dit_turn_heights[0] > 0)
                                ? out.dit_turn_heights[0]
                                : 512;
            caps.push_back(policy.ar_ttft_slo_for(has_img, n_img, res, k_ar_opt) * slo_time_scale);
        }
    }
    return caps;
}

bool evaluate_dit_turns_ok(
    const PolicyConfig& policy,
    const std::vector<double>& dit_turns_ms,
    const std::vector<int>& dit_heights,
    const std::vector<int>& dit_steps,
    std::optional<double> k_dit_opt,
    double slo_time_scale) {
    for (size_t i = 0; i < dit_turns_ms.size(); ++i) {
        const int h = (i < dit_heights.size() && dit_heights[i] > 0) ? dit_heights[i] : 512;
        const int st = (i < dit_steps.size() && dit_steps[i] > 0) ? dit_steps[i] : default_dit_steps_for(policy, h);
        const double cap = policy.dit_turn_slo_for(h, st, k_dit_opt) * slo_time_scale;
        if (dit_turns_ms[i] > cap) {
            return false;
        }
    }
    return true;
}

bool evaluate_outcome_at_k(
    const PolicyConfig& policy,
    const SessionOutcomeInput& out,
    double k_ar,
    double k_dit,
    double slo_time_scale) {
    if (out.status != "ok") {
        return false;
    }
    const std::string fam = (policy.model_family == "internvl_u") ? "internvl_u" : "bagel";
    const auto slo = default_operator_profile().slo_thresholds_ms(k_ar, fam);
    const double ttft_cap = slo.at("ar_ttft_slo_ms") * slo_time_scale;
    const double tpot_cap = slo.at("ar_tpot_slo_ms") * slo_time_scale;
    const std::vector<double> per_turn_caps = build_session_ar_ttft_caps(policy, out, k_ar, slo_time_scale);
    const ArTpotFlags ar_flags = eval_ar_slos(
        out.ar_ttfts_ms, out.ar_tpots_ms, out.ar_turn_tpots_ms, ttft_cap, tpot_cap, per_turn_caps);
    if (!select_ar_slo_by_mode(ar_flags, policy.ar_tpot_slo_mode)) {
        return false;
    }
    return evaluate_dit_turns_ok(policy, out.dit_turns_ms, out.dit_turn_heights, out.dit_turn_steps, k_dit, slo_time_scale);
}

struct TurnSloEval {
    bool is_ok = false;
    bool has_ar = false;
    bool has_dit = false;
    bool ar_ok = true;
    bool dit_ok = true;
    bool joint_ok = false;
};

TurnSloEval evaluate_turn_record_slo(
    const PolicyConfig& policy,
    const TurnRecordInput& tr,
    double ar_ttft_cap,
    double ar_tpot_cap,
    std::optional<double> k_dit_opt,
    double slo_time_scale) {
    TurnSloEval ev;
    ev.has_ar = (tr.modality_plan == "ar_only" || tr.modality_plan == "cot_then_dit" || tr.ar_ttft_ms.has_value());
    ev.has_dit = (tr.modality_plan == "dit_t2i" || tr.modality_plan == "dit_edit" ||
                  tr.modality_plan == "cot_then_dit" || tr.dit_ms.has_value());
    if (tr.status != "ok") {
        ev.is_ok = false;
        ev.ar_ok = false;
        ev.dit_ok = false;
        ev.joint_ok = false;
        return ev;
    }
    ev.is_ok = true;
    if (ev.has_ar) {
        const double ttft = tr.ar_ttft_ms.value_or(0.0);
        bool tpot_ok = true;
        if (!tr.ar_tpots_ms.empty()) {
            if (policy.ar_tpot_slo_mode == "turn_mean") {
                tpot_ok = mean_raw(tr.ar_tpots_ms) <= ar_tpot_cap;
            } else if (policy.ar_tpot_slo_mode == "token_max" || policy.ar_tpot_slo_mode == "strict_max") {
                tpot_ok = *std::max_element(tr.ar_tpots_ms.begin(), tr.ar_tpots_ms.end()) <= ar_tpot_cap;
            } else {
                tpot_ok = percentile_raw(tr.ar_tpots_ms, 95.0) <= ar_tpot_cap;
            }
        } else if (policy.ar_tpot_slo_mode == "turn_mean" && tr.ar_tpot_mean_ms.has_value()) {
            tpot_ok = *tr.ar_tpot_mean_ms <= ar_tpot_cap;
        } else if ((policy.ar_tpot_slo_mode == "token_max" || policy.ar_tpot_slo_mode == "strict_max") &&
                   tr.ar_tpot_strict_max_ms.has_value()) {
            tpot_ok = *tr.ar_tpot_strict_max_ms <= ar_tpot_cap;
        } else if (tr.ar_tpot_p95_ms.has_value()) {
            tpot_ok = *tr.ar_tpot_p95_ms <= ar_tpot_cap;
        }
        ev.ar_ok = (ttft <= ar_ttft_cap) && tpot_ok;
    }
    if (ev.has_dit) {
        const int h = (tr.height > 0) ? tr.height : 512;
        const int st = (tr.dit_steps > 0) ? tr.dit_steps : default_dit_steps_for(policy, h);
        const double cap = policy.dit_turn_slo_for(h, st, k_dit_opt) * slo_time_scale;
        ev.dit_ok = (tr.dit_ms.value_or(0.0) <= cap);
    }
    ev.joint_ok = (!ev.has_ar || ev.ar_ok) && (!ev.has_dit || ev.dit_ok);
    return ev;
}

struct SessionTurnCounts {
    size_t planned_turns = 0;
    size_t completed_turns = 0;
    size_t joint_ok_turns = 0;
    size_t planned_ar_turns = 0;
    size_t ar_ok_turns = 0;
    size_t planned_dit_turns = 0;
    size_t dit_ok_turns = 0;
};

SessionTurnCounts evaluate_session_turn_counts(
    const PolicyConfig& policy,
    const SessionOutcomeInput& out,
    std::optional<double> k_ar_opt,
    double ar_tpot_cap,
    std::optional<double> k_dit_opt,
    double slo_time_scale,
    bool fallback_session_joint_ok,
    bool fallback_session_ar_ok,
    bool fallback_session_dit_ok) {
    SessionTurnCounts c;
    const size_t n_rec = out.turn_records.size();
    const size_t raw_planned = (out.planned_turns > 0) ? static_cast<size_t>(out.planned_turns) : 0;
    if (n_rec == 0) {
        c.planned_turns = std::max<size_t>(1, raw_planned);
        const bool is_ok = (out.status == "ok");
        c.completed_turns = is_ok ? c.planned_turns : 0;
        c.joint_ok_turns = (is_ok && fallback_session_joint_ok) ? c.planned_turns : 0;
        c.planned_ar_turns = (out.planned_ar_turns >= 0) ? static_cast<size_t>(out.planned_ar_turns) : out.ar_ttfts_ms.size();
        c.planned_dit_turns = (out.planned_dit_turns >= 0) ? static_cast<size_t>(out.planned_dit_turns) : out.dit_turns_ms.size();
        c.ar_ok_turns = (is_ok && fallback_session_ar_ok) ? c.planned_ar_turns : 0;
        c.dit_ok_turns = (is_ok && fallback_session_dit_ok) ? c.planned_dit_turns : 0;
        return c;
    }

    size_t seen_ar = 0;
    size_t seen_dit = 0;
    int prev_dit_height = 0;
    for (const auto& tr : out.turn_records) {
        const double turn_ttft_cap = resolve_turn_ar_ttft_cap_ms(
            policy, tr, out.quadrant, prev_dit_height, k_ar_opt, slo_time_scale);
        const TurnSloEval ev = evaluate_turn_record_slo(policy, tr, turn_ttft_cap, ar_tpot_cap, k_dit_opt, slo_time_scale);
        if (ev.is_ok) ++c.completed_turns;
        if (ev.joint_ok) ++c.joint_ok_turns;
        if (ev.has_ar) {
            ++seen_ar;
            if (ev.ar_ok) ++c.ar_ok_turns;
        }
        if (ev.has_dit) {
            ++seen_dit;
            if (ev.dit_ok) ++c.dit_ok_turns;
            prev_dit_height = (tr.height > 0) ? tr.height : 512;
        }
    }
    // D2: Planned turns form the denominator; OOM turns and unexecuted follow-up turns count as misses.
    c.planned_turns = std::max(n_rec, raw_planned);
    const size_t raw_ar = (out.planned_ar_turns >= 0) ? static_cast<size_t>(out.planned_ar_turns) : seen_ar;
    const size_t raw_dit = (out.planned_dit_turns >= 0) ? static_cast<size_t>(out.planned_dit_turns) : seen_dit;
    c.planned_ar_turns = std::max(seen_ar, raw_ar);
    c.planned_dit_turns = std::max(seen_dit, raw_dit);
    return c;
}

struct PreparedTransition {
    CPTGRequestSpec req;
    double est_dit_ms = 0.0;
    int gamma = 1;
    bool is_tight_mot = true;
};

PreparedTransition prepare_transition_request(
    const PolicyConfig& policy,
    const OperatorLatencyProfile& op_prof,
    const std::string& session_id,
    int turn_idx,
    int attempt,
    int src_gpu,
    int64_t state_bytes,
    const std::string& modality_plan,
    int height,
    int dit_steps,
    bool has_more_edits,
    std::optional<int64_t> gen_expert_bytes_override,
    std::optional<double> dit_step1_ms_override,
    std::optional<double> est_dit_ms_override) {
    PreparedTransition prep;
    const std::string arch = policy.cptg_architecture();
    prep.is_tight_mot = (arch == "tight_mot");
    const std::string fam = (arch == "bridge_mllm_dit") ? "internvl_u" : "bagel";
    prep.gamma = (modality_plan == "dit_edit" && has_more_edits) ? 2 : 1;

    prep.req.req_id = session_id + "_t" + std::to_string(turn_idx) + "_a" + std::to_string(attempt);
    prep.req.src_gpu = src_gpu;
    prep.req.architecture = arch;
    prep.req.model_family = fam;
    prep.req.gen_expert_bytes = gen_expert_bytes_override.value_or(op_prof.expert_bytes(fam));
    prep.req.dit_step1_ms = dit_step1_ms_override.value_or(op_prof.step_ms(fam, height));
    prep.req.dit_total_steps = dit_steps;
    prep.req.resolution = height;
    prep.req.interleaved_roundtrip_gamma = prep.gamma;
    if (arch == "bridge_mllm_dit") {
        const int64_t eff = std::max<int64_t>(state_bytes, 64LL * 1024LL);
        prep.req.kv_bytes = eff;
        prep.req.bridge_bytes = eff;
    } else {
        prep.req.kv_bytes = std::max<int64_t>(state_bytes, 1LL * kMiB);
    }

    const std::string est_fam = (policy.model_family == "internvl_u") ? "internvl_u" : "bagel";
    prep.est_dit_ms = est_dit_ms_override.value_or(op_prof.unloaded_dit_turn_ms(est_fam, height, dit_steps));
    return prep;
}

TurnTransitionPlan finalize_transition_plan(
    RouteDecision dec,
    const std::string& ship_mode,
    const PreparedTransition& prep,
    const std::string& base_ctx_id,
    int turn_idx,
    int src_gpu,
    bool has_more_turns) {
    const int dst_gpu = dec.dst_gpu;
    const bool ephemeral_gen_ctx = (ship_mode == "nccl_static" || ship_mode == "shm_static");
    const bool needs_return_ship = ephemeral_gen_ctx && has_more_turns && prep.is_tight_mot;
    bool release_src = false;
    if (ship_mode == "nccl_dual") {
        release_src = true;
    } else if (ephemeral_gen_ctx) {
        release_src = prep.is_tight_mot ? true : (!has_more_turns);
    }

    TurnTransitionPlan out;
    out.decision = std::move(dec);
    out.ship_mode = ship_mode;
    out.src_gpu = src_gpu;
    out.dst_gpu = dst_gpu;
    out.est_dit_ms = prep.est_dit_ms;
    out.reserved_kv_bytes = (ship_mode != "none" && dst_gpu != src_gpu) ? prep.req.effective_state_bytes() : 0;
    out.interleaved_roundtrip_gamma = prep.gamma;
    out.ephemeral_gen_ctx = ephemeral_gen_ctx;
    out.needs_return_ship = needs_return_ship;
    out.release_src_on_fwd = release_src;
    out.include_cfg_dup_on_wire = false;
    out.dit_ctx_id = ephemeral_gen_ctx ? (base_ctx_id + "_gen_t" + std::to_string(turn_idx)) : base_ctx_id;
    return out;
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
        {"interleaved_roundtrip_gamma", interleaved_roundtrip_gamma},
        {"ephemeral_gen_ctx", ephemeral_gen_ctx},
        {"needs_return_ship", needs_return_ship},
        {"release_src_on_fwd", release_src_on_fwd},
        {"include_cfg_dup_on_wire", include_cfg_dup_on_wire},
        {"dit_ctx_id", dit_ctx_id},
    };
}

TurnRecordInput TurnRecordInput::from_json(const nlohmann::json& j, int default_turn_idx) {
    TurnRecordInput tr;
    tr.turn_idx = default_turn_idx;
    if (!j.is_object()) {
        return tr;
    }
    tr.turn_idx = j.value("turn_idx", default_turn_idx);
    tr.modality_plan = j.value("modality_plan", "");
    tr.status = j.value("status", "ok");
    tr.ar_ttft_ms = first_number_opt(j, {"ar_ttft_ms", "ttft_ms"});
    tr.ar_tpots_ms = parse_double_array(j, "ar_tpots_ms");
    tr.ar_tpot_p95_ms = first_number_opt(j, {"ar_tpot_p95_ms", "ar_tpot_ms"});
    tr.ar_tpot_mean_ms = first_number_opt(j, {"ar_tpot_mean_ms"});
    tr.ar_tpot_strict_max_ms = first_number_opt(j, {"ar_tpot_strict_max_ms"});
    tr.dit_ms = first_number_opt(j, {"dit_ms", "dit_turn_ms", "dit_lat_ms"});
    tr.height = (j.contains("height") && j["height"].is_number()) ? j["height"].get<int>() : 0;
    tr.dit_steps = (j.contains("dit_steps") && j["dit_steps"].is_number()) ? j["dit_steps"].get<int>() : 0;
    if (j.contains("kv_mib") && j["kv_mib"].is_number()) {
        tr.kv_mib = j["kv_mib"].get<double>();
    } else if (j.contains("kv_bytes") && j["kv_bytes"].is_number()) {
        tr.kv_mib = j["kv_bytes"].get<double>() / static_cast<double>(kMiB);
    }
    if (j.contains("has_image") && j["has_image"].is_boolean()) {
        tr.has_image = j["has_image"].get<bool>();
    }
    if (j.contains("num_images") && j["num_images"].is_number()) {
        tr.num_images = j["num_images"].get<int>();
    } else if (j.contains("image_paths") && j["image_paths"].is_array()) {
        tr.num_images = static_cast<int>(j["image_paths"].size());
    }
    if (j.contains("image_res") && j["image_res"].is_number()) {
        tr.image_res = j["image_res"].get<int>();
    }
    if (j.contains("use_prev_image") && j["use_prev_image"].is_boolean()) {
        tr.use_prev_image = j["use_prev_image"].get<bool>();
    }
    tr.queue_ms = first_number_opt(j, {"queue_ms", "queue_wait_ms"});
    tr.h2d_weight_ms = first_number_opt(j, {"h2d_weight_ms", "h2d_ms", "morph_h2d_ms"});
    tr.kv_transfer_ms = first_number_opt(j, {"kv_transfer_ms", "kv_ship_ms", "state_ship_ms"});
    tr.vae_ms = first_number_opt(j, {"vae_ms", "vae_decode_ms"});
    tr.preempt_yield_ms = first_number_opt(j, {"preempt_yield_ms", "yield_ms"});
    return tr;
}

SessionOutcomeInput SessionOutcomeInput::from_json(const nlohmann::json& j) {
    SessionOutcomeInput o;
    if (!j.is_object()) {
        return o;
    }
    o.status = j.value("status", "error");
    if (j.contains("quadrant") && j["quadrant"].is_string()) {
        o.quadrant = j["quadrant"].get<std::string>();
    }
    if (j.contains("phase_label") && j["phase_label"].is_string()) {
        o.phase_label = j["phase_label"].get<std::string>();
    }
    if (j.contains("arrival_offset_s") && j["arrival_offset_s"].is_number()) {
        o.arrival_offset_s = j["arrival_offset_s"].get<double>();
    }
    if (j.contains("e2e_ms") && j["e2e_ms"].is_number()) {
        o.e2e_ms = j["e2e_ms"].get<double>();
    }
    if (j.contains("active_jct_ms") && j["active_jct_ms"].is_number()) {
        o.active_jct_ms = std::max(0.0, j["active_jct_ms"].get<double>());
    }
    if (j.contains("think_time_ms") && j["think_time_ms"].is_number()) {
        o.think_time_ms = j["think_time_ms"].get<double>();
    }
    if (j.contains("planned_turns") && j["planned_turns"].is_number()) {
        o.planned_turns = j["planned_turns"].get<int>();
    }
    if (j.contains("planned_ar_turns") && j["planned_ar_turns"].is_number()) {
        o.planned_ar_turns = j["planned_ar_turns"].get<int>();
    }
    if (j.contains("planned_dit_turns") && j["planned_dit_turns"].is_number()) {
        o.planned_dit_turns = j["planned_dit_turns"].get<int>();
    }
    o.ar_ttfts_ms = parse_double_array(j, "ar_ttfts_ms");
    o.ar_tpots_ms = parse_double_array(j, "ar_tpots_ms");
    o.ar_turn_tpots_ms = parse_nested_double_array(j, "ar_turn_tpots_ms");
    o.ar_turn_has_image = parse_bool_or_int_array(j, "ar_turn_has_image");
    o.ar_turn_num_images = parse_int_array(j, "ar_turn_num_images");
    o.ar_turn_image_res = parse_int_array(j, "ar_turn_image_res");
    o.dit_turns_ms = parse_double_array(j, "dit_turns_ms");
    o.dit_turn_heights = parse_int_array(j, "dit_turn_heights");
    o.dit_turn_steps = parse_int_array(j, "dit_turn_steps");

    if (j.contains("ar_slo_met") && j["ar_slo_met"].is_boolean()) {
        o.ar_slo_met = j["ar_slo_met"].get<bool>();
    }
    if (j.contains("ar_slo_mean_tpot_met") && j["ar_slo_mean_tpot_met"].is_boolean()) {
        o.ar_slo_mean_tpot_met = j["ar_slo_mean_tpot_met"].get<bool>();
    }
    if (j.contains("ar_slo_strict_max_tpot_met") && j["ar_slo_strict_max_tpot_met"].is_boolean()) {
        o.ar_slo_strict_max_tpot_met = j["ar_slo_strict_max_tpot_met"].get<bool>();
    }
    if (j.contains("dit_slo_met") && j["dit_slo_met"].is_boolean()) {
        o.dit_slo_met = j["dit_slo_met"].get<bool>();
    }
    if (j.contains("joint_slo_met") && j["joint_slo_met"].is_boolean()) {
        o.joint_slo_met = j["joint_slo_met"].get<bool>();
    }
    if (j.contains("resident_alpha_sample") && j["resident_alpha_sample"].is_number()) {
        o.resident_alpha_sample = j["resident_alpha_sample"].get<double>();
    }
    o.queue_ms = first_number_opt(j, {"queue_ms", "queue_wait_ms"});
    o.h2d_weight_ms = first_number_opt(j, {"h2d_weight_ms", "h2d_ms", "morph_h2d_ms"});
    o.kv_transfer_ms = first_number_opt(j, {"kv_transfer_ms", "kv_ship_ms", "state_ship_ms"});
    o.vae_ms = first_number_opt(j, {"vae_ms", "vae_decode_ms"});
    o.preempt_yield_ms = first_number_opt(j, {"preempt_yield_ms", "yield_ms"});
    o.path_a_ships = j.value("path_a_ships", 0);
    o.path_b_morphs = j.value("path_b_morphs", 0);
    o.preempted_ar_turns = j.value("preempted_ar_turns", 0);
    o.total_yields = j.value("total_yields", 0);
    o.oom_retries = j.value("oom_retries", 0);
    o.cross_node_ships = j.value("cross_node_ships", 0);
    if (j.contains("cross_node_ship_bytes_list") && j["cross_node_ship_bytes_list"].is_array()) {
        for (const auto& b : j["cross_node_ship_bytes_list"]) {
            if (b.is_number()) {
                o.max_cross_node_ship_bytes = std::max(o.max_cross_node_ship_bytes, b.get<int64_t>());
            }
        }
    }
    if (j.contains("turn_records") && j["turn_records"].is_array()) {
        o.turn_records.reserve(j["turn_records"].size());
        for (size_t pos = 0; pos < j["turn_records"].size(); ++pos) {
            if (j["turn_records"][pos].is_object()) {
                o.turn_records.push_back(TurnRecordInput::from_json(j["turn_records"][pos], static_cast<int>(pos)));
            }
        }
    }
    return o;
}

WorkerStatsInput WorkerStatsInput::from_json(const nlohmann::json& st) {
    WorkerStatsInput w;
    if (!st.is_object()) {
        return w;
    }
    const double nvml_pk = first_number_opt(st, {"nvml_peak_used_mib"}).value_or(0.0);
    const double comb = (nvml_pk > 0.0)
                            ? nvml_pk
                            : first_number_opt(st, {"combined_physical_mib", "peak_reserved_mib"}).value_or(0.0);
    const double pt = first_number_opt(st, {"pytorch_reserved_mib", "peak_reserved_mib"}).value_or(0.0);
    w.nvml_peak_used_mib = (nvml_pk > 0.0) ? nvml_pk : comb;
    w.combined_physical_mib = comb;
    w.pytorch_reserved_mib = pt;
    w.nvml_sm_util_mean_pct = first_number_opt(st, {"nvml_sm_util_mean_pct", "sm_util_mean_pct"});
    w.nvml_mem_util_mean_pct = first_number_opt(st, {"nvml_mem_util_mean_pct", "mem_util_mean_pct"});
    w.effective_hbm_util_mean_pct = first_number_opt(st, {"effective_hbm_util_mean_pct", "hbm_util_mean_pct"});

    if (st.contains("vmm_physical_pool_mib") && st["vmm_physical_pool_mib"].is_number()) {
        w.vmm_physical_pool_mib = st["vmm_physical_pool_mib"].get<double>();
    } else if (st.contains("vmm_stats") && st["vmm_stats"].is_object() &&
               st["vmm_stats"].contains("pool_total_bytes") && st["vmm_stats"]["pool_total_bytes"].is_number()) {
        w.vmm_physical_pool_mib = st["vmm_stats"]["pool_total_bytes"].get<double>() / static_cast<double>(kMiB);
    }

    w.oom_count = st.value("oom_count", 0);
    if (st.contains("kv_host_swap_outs") && st["kv_host_swap_outs"].is_number()) {
        w.kv_host_swap_outs = st["kv_host_swap_outs"].get<int>();
    } else if (st.contains("vmm_stats") && st["vmm_stats"].is_object()) {
        w.kv_host_swap_outs = st["vmm_stats"].value("kv_swap_outs_total", 0);
    }
    if (st.contains("kv_host_swap_ins") && st["kv_host_swap_ins"].is_number()) {
        w.kv_host_swap_ins = st["kv_host_swap_ins"].get<int>();
    } else if (st.contains("vmm_stats") && st["vmm_stats"].is_object()) {
        w.kv_host_swap_ins = st["vmm_stats"].value("kv_swap_ins_total", 0);
    }
    w.kv_recompute_evictions = st.value("kv_recompute_evictions", 0);
    w.kv_recompute_restores = st.value("kv_recompute_restores", 0);
    if (st.contains("vmm_stats") && st["vmm_stats"].is_object()) {
        const auto& vs = st["vmm_stats"];
        w.overflow_creates = vs.value("overflow_creates", 0);
        w.super_to_fine_carves = vs.value("super_to_fine_carves", 0);
        w.expert_auto_evictions_for_kv = vs.value("expert_auto_evictions_for_kv", 0);
        w.expert_auto_evictions_for_transient = vs.value("expert_auto_evictions_for_transient", 0);
    }
    return w;
}

MorphControlEngine::MorphControlEngine(
    PolicyConfig policy,
    ClusterTopologyProfile topology,
    std::shared_ptr<DualPathMorphRouter> router)
    : policy_(std::move(policy)),
      topology_(std::move(topology)),
      router_(router ? std::move(router)
                     : std::make_shared<DualPathMorphRouter>(
                           topology_,
                           nullptr,
                           default_operator_profile(),
                           std::nullopt,
                           12.0,
                           0.90,
                           true,
                           true,
                           128LL * kMiB,
                           policy_.cost_model_mode)) {}

PreemptEdfController MorphControlEngine::make_edf_controller(
    int num_layers,
    double tpot_slo_ms,
    double init_layer_ms,
    double init_ar_step_ms,
    double time_scale,
    int max_microbatch_cap) noexcept {
    constexpr double kDefaultGuardMs = 24.0;
    const double eff_scale = (time_scale > 0.0) ? time_scale : 1.0;
    return PreemptEdfController(
        num_layers,
        tpot_slo_ms,
        kDefaultGuardMs * eff_scale,
        init_layer_ms,
        init_ar_step_ms,
        max_microbatch_cap);
}

double MorphControlEngine::deterministic_think_s(
    const std::string& session_id,
    int turn_idx,
    double lo_s,
    double hi_s) {
    return sample_think_dist_s(
        session_id + ":" + std::to_string(turn_idx), lo_s, hi_s, "uniform", 0.0, 0.0);
}

double MorphControlEngine::deterministic_think_ms_sim(
    int sid,
    int turn_idx,
    double lo_s,
    double hi_s) {
    return sample_think_dist_s(
               "sim_" + std::to_string(sid) + "_" + std::to_string(turn_idx), lo_s, hi_s, "uniform", 0.0, 0.0) *
           1000.0;
}

double MorphControlEngine::deterministic_think_dist_s(
    const std::string& session_id,
    int turn_idx,
    double lo_s,
    double hi_s,
    const std::string& dist,
    double median_s,
    double sigma) {
    return sample_think_dist_s(
        session_id + ":" + std::to_string(turn_idx), lo_s, hi_s, dist, median_s, sigma);
}

double MorphControlEngine::deterministic_think_dist_ms_sim(
    int sid,
    int turn_idx,
    double lo_s,
    double hi_s,
    const std::string& dist,
    double median_s,
    double sigma) {
    return sample_think_dist_s(
               "sim_" + std::to_string(sid) + "_" + std::to_string(turn_idx), lo_s, hi_s, dist, median_s, sigma) *
           1000.0;
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
    const PreparedTransition prep = prepare_transition_request(
        policy_,
        router_->operator_profile(),
        session_id,
        turn_idx,
        attempt,
        src_gpu,
        state_bytes,
        modality_plan,
        height,
        dit_steps,
        has_more_edits,
        std::nullopt,
        std::nullopt,
        std::nullopt);

    auto [dec, ship_mode] = router_->plan_and_reserve_transition(
        policy_.routing_mode,
        policy_.static_und_ranks,
        policy_.static_gen_ranks,
        prep.req,
        prep.est_dit_ms,
        nullptr,
        gpu_busy,
        policy_.use_h2d_pacer,
        now_ms);

    return finalize_transition_plan(std::move(dec), ship_mode, prep, base_ctx_id, turn_idx, src_gpu, has_more_turns);
}

TurnTransitionPlan MorphControlEngine::plan_turn_transition_readonly(
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
    const std::map<int, WorkerState>* ext_workers,
    const std::map<int, bool>& gpu_busy,
    std::optional<int64_t> gen_expert_bytes_override,
    std::optional<double> dit_step1_ms_override,
    std::optional<double> est_dit_ms_override) const {
    const PreparedTransition prep = prepare_transition_request(
        policy_,
        router_->operator_profile(),
        session_id,
        turn_idx,
        attempt,
        src_gpu,
        state_bytes,
        modality_plan,
        height,
        dit_steps,
        has_more_edits,
        gen_expert_bytes_override,
        dit_step1_ms_override,
        est_dit_ms_override);

    auto [dec, ship_mode] = router_->plan_transition_readonly(
        policy_.routing_mode,
        policy_.static_und_ranks,
        policy_.static_gen_ranks,
        prep.req,
        prep.est_dit_ms,
        ext_workers,
        gpu_busy,
        now_ms);

    return finalize_transition_plan(std::move(dec), ship_mode, prep, base_ctx_id, turn_idx, src_gpu, has_more_turns);
}

nlohmann::json MorphControlEngine::evaluate_session_slos(
    const PolicyConfig& policy,
    const std::vector<double>& ar_ttfts_ms,
    const std::vector<double>& ar_tpots_ms,
    const std::vector<double>& dit_turns_ms,
    const std::vector<int>& dit_heights,
    const std::vector<int>& dit_steps,
    const std::vector<std::vector<double>>& ar_turn_tpots_ms,
    double slo_time_scale,
    const std::vector<int>& ar_turn_has_image,
    const std::vector<int>& ar_turn_num_images,
    const std::vector<int>& ar_turn_image_res) {
    const double scale = (slo_time_scale > 0.0) ? slo_time_scale : 1.0;
    std::vector<double> per_turn_ttft_caps;
    if (!ar_turn_has_image.empty() || !ar_turn_num_images.empty()) {
        const size_t n = std::max({ar_ttfts_ms.size(), ar_turn_has_image.size(), ar_turn_num_images.size()});
        per_turn_ttft_caps.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            int n_img = (i < ar_turn_num_images.size()) ? std::max(0, ar_turn_num_images[i]) : 0;
            bool has_img = (i < ar_turn_has_image.size()) ? (ar_turn_has_image[i] != 0) : (n_img > 0);
            if (has_img && n_img <= 0) n_img = 1;
            if (n_img > 0) has_img = true;
            const int res = (i < ar_turn_image_res.size() && ar_turn_image_res[i] > 0) ? ar_turn_image_res[i] : 512;
            per_turn_ttft_caps.push_back(policy.ar_ttft_slo_for(has_img, n_img, res, std::nullopt) * scale);
        }
    }
    const ArTpotFlags ar_flags = eval_ar_slos(
        ar_ttfts_ms,
        ar_tpots_ms,
        ar_turn_tpots_ms,
        policy.ar_ttft_slo_ms * scale,
        policy.ar_tpot_slo_ms * scale,
        per_turn_ttft_caps);
    const bool ar_ok = select_ar_slo_by_mode(ar_flags, policy.ar_tpot_slo_mode);
    const bool dit_ok = evaluate_dit_turns_ok(policy, dit_turns_ms, dit_heights, dit_steps, std::nullopt, scale);
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
    const nlohmann::json& worker_stats_array,
    double slo_time_scale,
    const nlohmann::json& sim_extra_fields,
    bool include_sessions) {
    const double scale = (slo_time_scale > 0.0) ? slo_time_scale : 1.0;
    const double eff_ar_ttft_slo_ms = policy.ar_ttft_slo_ms * scale;
    const double eff_ar_tpot_slo_ms = policy.ar_tpot_slo_ms * scale;

    double peak_res_mib = 0.0;
    double peak_nvml_mib = 0.0;
    double peak_pytorch_res_mib = 0.0;
    double max_vmm_pool_mib = 0.0;
    int worker_ooms = 0;
    int kv_host_swap_outs = 0;
    int kv_host_swap_ins = 0;
    int kv_recompute_evictions = 0;
    int kv_recompute_restores = 0;
    int overflow_creates = 0;
    int carves = 0;
    int evict_kv = 0;
    int evict_tr = 0;
    std::vector<double> sm_util_samples;
    std::vector<double> mem_util_samples;
    std::vector<double> eff_hbm_util_samples;
    std::vector<std::pair<double, double>> realtime_alpha_samples;

    auto collect_alpha_timeline = [&](const nlohmann::json& arr) {
        if (!arr.is_array()) return;
        for (const auto& sample : arr) {
            if (!sample.is_object()) continue;
            const auto t_opt = first_number_opt(sample, {"t_s", "time_s", "offset_s"});
            const auto a_opt = first_number_opt(sample, {"mean_alpha", "alpha", "resident_alpha"});
            if (t_opt.has_value() && a_opt.has_value()) {
                realtime_alpha_samples.emplace_back(*t_opt, *a_opt);
            }
        }
    };

    auto ingest_worker_entry = [&](const nlohmann::json& raw_st) {
        if (!raw_st.is_object()) return;
        const WorkerStatsInput ws = WorkerStatsInput::from_json(raw_st);
        peak_res_mib = std::max(peak_res_mib, ws.combined_physical_mib);
        peak_nvml_mib = std::max(peak_nvml_mib, ws.nvml_peak_used_mib);
        peak_pytorch_res_mib = std::max(peak_pytorch_res_mib, ws.pytorch_reserved_mib);
        max_vmm_pool_mib = std::max(max_vmm_pool_mib, ws.vmm_physical_pool_mib);
        if (ws.nvml_sm_util_mean_pct.has_value()) {
            sm_util_samples.push_back(*ws.nvml_sm_util_mean_pct);
        }
        if (ws.nvml_mem_util_mean_pct.has_value()) {
            mem_util_samples.push_back(*ws.nvml_mem_util_mean_pct);
        }
        if (ws.effective_hbm_util_mean_pct.has_value()) {
            eff_hbm_util_samples.push_back(*ws.effective_hbm_util_mean_pct);
        }
        if (raw_st.contains("alpha_timeline")) {
            collect_alpha_timeline(raw_st["alpha_timeline"]);
        }
        worker_ooms += ws.oom_count;
        kv_host_swap_outs += ws.kv_host_swap_outs;
        kv_host_swap_ins += ws.kv_host_swap_ins;
        kv_recompute_evictions += ws.kv_recompute_evictions;
        kv_recompute_restores += ws.kv_recompute_restores;
        overflow_creates += ws.overflow_creates;
        carves += ws.super_to_fine_carves;
        evict_kv += ws.expert_auto_evictions_for_kv;
        evict_tr += ws.expert_auto_evictions_for_transient;
    };

    if (worker_stats_array.is_array()) {
        for (const auto& raw_st : worker_stats_array) {
            ingest_worker_entry(raw_st);
        }
    } else if (worker_stats_array.is_object()) {
        if (worker_stats_array.contains("alpha_timeline")) {
            collect_alpha_timeline(worker_stats_array["alpha_timeline"]);
        }
        if (auto sm = first_number_opt(worker_stats_array, {"nvml_sm_util_mean_pct", "sm_util_mean_pct"}); sm.has_value()) {
            sm_util_samples.push_back(*sm);
        }
        if (auto mu = first_number_opt(worker_stats_array, {"nvml_mem_util_mean_pct", "mem_util_mean_pct"}); mu.has_value()) {
            mem_util_samples.push_back(*mu);
        }
        if (auto hu = first_number_opt(worker_stats_array, {"effective_hbm_util_mean_pct", "hbm_util_mean_pct"}); hu.has_value()) {
            eff_hbm_util_samples.push_back(*hu);
        }
        if (worker_stats_array.contains("workers") && worker_stats_array["workers"].is_array()) {
            for (const auto& raw_st : worker_stats_array["workers"]) {
                ingest_worker_entry(raw_st);
            }
        } else {
            ingest_worker_entry(worker_stats_array);
        }
    }
    if (sim_extra_fields.is_object()) {
        if (sim_extra_fields.contains("alpha_timeline")) {
            collect_alpha_timeline(sim_extra_fields["alpha_timeline"]);
        }
        if (auto sm = first_number_opt(sim_extra_fields, {"nvml_sm_util_mean_pct", "sm_util_mean_pct"}); sm.has_value()) {
            sm_util_samples.push_back(*sm);
        }
        if (auto mu = first_number_opt(sim_extra_fields, {"nvml_mem_util_mean_pct", "mem_util_mean_pct"}); mu.has_value()) {
            mem_util_samples.push_back(*mu);
        }
        if (auto hu = first_number_opt(sim_extra_fields, {"effective_hbm_util_mean_pct", "hbm_util_mean_pct"}); hu.has_value()) {
            eff_hbm_util_samples.push_back(*hu);
        }
    }

    const size_t n_outcomes = outcomes_array.is_array() ? outcomes_array.size() : 0;
    const size_t n_tot = std::max<size_t>(1, n_outcomes);

    std::vector<SessionOutcomeInput> parsed_outcomes;
    parsed_outcomes.reserve(n_outcomes);
    double min_arrival = 0.0;
    double max_arrival = 0.0;
    for (size_t i = 0; i < n_outcomes; ++i) {
        SessionOutcomeInput o = SessionOutcomeInput::from_json(outcomes_array[i]);
        if (i == 0 || o.arrival_offset_s < min_arrival) {
            min_arrival = o.arrival_offset_s;
        }
        if (i == 0 || o.arrival_offset_s > max_arrival) {
            max_arrival = o.arrival_offset_s;
        }
        parsed_outcomes.push_back(std::move(o));
    }

    std::vector<size_t> core_indices;
    std::string eff_window_mode = "session_trim";
    double win_start_s = 0.0;
    double win_end_s = 0.0;

    const double ws_s = std::max(0.0, policy.warmup_window_s);
    const double cd_s = std::max(0.0, policy.cooldown_s);
    if (n_outcomes > 0 && policy.steady_window_mode == "time" &&
        max_arrival >= ws_s + cd_s + 10.0) {
        const double max_end_s = max_arrival - cd_s;
        const double we_s = (policy.steady_window_s > 0.0)
                                ? std::min(ws_s + policy.steady_window_s, max_end_s)
                                : max_end_s;
        if (we_s > ws_s + 1e-6) {
            for (size_t i = 0; i < n_outcomes; ++i) {
                const double arr = parsed_outcomes[i].arrival_offset_s;
                if (arr >= ws_s - 1e-9 && arr <= we_s + 1e-9) {
                    core_indices.push_back(i);
                }
            }
            if (!core_indices.empty()) {
                eff_window_mode = "time";
                win_start_s = ws_s;
                win_end_s = we_s;
            }
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
            win_start_s = parsed_outcomes[core_indices.front()].arrival_offset_s;
            win_end_s = parsed_outcomes[core_indices.back()].arrival_offset_s;
            for (size_t idx : core_indices) {
                win_start_s = std::min(win_start_s, parsed_outcomes[idx].arrival_offset_s);
                win_end_s = std::max(win_end_s, parsed_outcomes[idx].arrival_offset_s);
            }
        }
    }

    std::vector<uint8_t> in_core(n_outcomes, 0);
    for (size_t idx : core_indices) {
        in_core[idx] = 1;
    }
    const size_t steady_sessions_cnt = core_indices.size();
    const size_t n_core = std::max<size_t>(1, steady_sessions_cnt);

    // Effective arrival durations (NEVER multiply by nominal offered_qps!).
    const double wall_s = std::max(1e-6, wall_seconds);
    double eff_steady_s = wall_s;
    if (eff_window_mode == "time" && (win_end_s - win_start_s) > 1e-6) {
        eff_steady_s = win_end_s - win_start_s;
    } else if (steady_sessions_cnt >= 2 && (win_end_s - win_start_s) > 1e-9) {
        eff_steady_s = (win_end_s - win_start_s) *
                       static_cast<double>(steady_sessions_cnt) /
                       static_cast<double>(steady_sessions_cnt - 1);
    }
    eff_steady_s = std::max(1e-6, eff_steady_s);

    double eff_full_s = wall_s;
    if (n_outcomes >= 2 && (max_arrival - min_arrival) > 1e-9) {
        eff_full_s = (max_arrival - min_arrival) *
                     static_cast<double>(n_outcomes) /
                     static_cast<double>(n_outcomes - 1);
    }
    eff_full_s = std::max(1e-6, eff_full_s);

    const double actual_arrival_qps =
        (n_outcomes == 0) ? 0.0 : (static_cast<double>(n_outcomes) / eff_full_s);
    const double actual_steady_arrival_qps =
        (steady_sessions_cnt == 0) ? 0.0 : (static_cast<double>(steady_sessions_cnt) / eff_steady_s);

    // Full-trace accumulators
    size_t ok_cnt = 0;
    size_t oom_cnt = 0;
    size_t err_cnt = 0;
    size_t joint_cnt = 0;
    size_t total_planned_turns = 0;
    size_t total_completed_turns = 0;
    size_t total_turn_joint_ok = 0;
    int path_a_ships = 0;
    int path_b_morphs = 0;
    int preempted_ar = 0;
    int total_yields = 0;
    int oom_retries = 0;
    int cross_node_ships = 0;
    int64_t max_xnode_bytes = 0;

    // Steady-state core accumulators (D5: turn-level and session-level share the exact same core cohort)
    size_t core_ok_cnt = 0;
    size_t core_oom_cnt = 0;
    size_t core_err_cnt = 0;
    size_t core_joint_cnt = 0;
    size_t core_sessions_with_ar = 0;
    size_t core_sessions_with_dit = 0;
    size_t core_ar_cnt = 0;
    size_t core_ar_mean_cnt = 0;
    size_t core_ar_strict_max_cnt = 0;
    size_t core_dit_cnt = 0;
    size_t core_planned_turns = 0;
    size_t core_completed_turns = 0;
    size_t core_turn_joint_ok = 0;
    size_t core_planned_ar_turns = 0;
    size_t core_turn_ar_ok = 0;
    size_t core_planned_dit_turns = 0;
    size_t core_turn_dit_ok = 0;

    std::vector<double> core_ok_e2es;
    std::vector<double> core_ok_active_jcts;
    std::vector<double> core_ok_ttfts;
    std::vector<double> core_ok_tpots;
    std::vector<double> core_ok_dits;

    // Optional per-component latency breakdown across steady-state turns/sessions
    std::vector<double> bd_queue_ms;
    std::vector<double> bd_h2d_weight_ms;
    std::vector<double> bd_kv_transfer_ms;
    std::vector<double> bd_vae_ms;
    std::vector<double> bd_preempt_yield_ms;

    struct TurnAcc {
        int total = 0;
        int ok = 0;
        int oom = 0;
        int unreached = 0;
        int slo_met = 0;
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
    const bool has_realtime_alpha = !realtime_alpha_samples.empty();
    if (has_realtime_alpha) {
        for (const auto& [t_s, a_val] : realtime_alpha_samples) {
            const int bidx = static_cast<int>(std::floor(std::max(0.0, t_s) / 10.0));
            auto& ba = ts_buckets[bidx];
            ba.alpha_sum += a_val;
            ++ba.alpha_cnt;
        }
    }

    for (size_t i = 0; i < n_outcomes; ++i) {
        const auto& o = parsed_outcomes[i];
        const bool is_ok = (o.status == "ok");
        const bool is_oom = (o.status == "oom");
        const bool core = (in_core[i] != 0);

        bool ar_primary_ok = false;
        bool ar_mean_ok = false;
        bool ar_strict_max_ok = false;
        bool dit_ok = false;
        bool joint_ok = false;

        if (is_ok) {
            const std::vector<double> per_turn_caps =
                build_session_ar_ttft_caps(policy, o, std::nullopt, scale);
            const ArTpotFlags ar_flags = eval_ar_slos(
                o.ar_ttfts_ms, o.ar_tpots_ms, o.ar_turn_tpots_ms, eff_ar_ttft_slo_ms, eff_ar_tpot_slo_ms, per_turn_caps);
            ar_primary_ok = o.ar_slo_met.value_or(select_ar_slo_by_mode(ar_flags, policy.ar_tpot_slo_mode));
            ar_mean_ok = o.ar_slo_mean_tpot_met.value_or(ar_flags.mean_ok);
            ar_strict_max_ok = o.ar_slo_strict_max_tpot_met.value_or(ar_flags.strict_max_ok);
            dit_ok = o.dit_slo_met.value_or(
                evaluate_dit_turns_ok(policy, o.dit_turns_ms, o.dit_turn_heights, o.dit_turn_steps, std::nullopt, scale));
            joint_ok = o.joint_slo_met.value_or(ar_primary_ok && dit_ok);

            ++ok_cnt;
            if (joint_ok) ++joint_cnt;

            if (core) {
                ++core_ok_cnt;
                if (joint_ok) ++core_joint_cnt;

                if (o.e2e_ms.has_value()) {
                    core_ok_e2es.push_back(*o.e2e_ms);
                    if (o.active_jct_ms.has_value()) {
                        core_ok_active_jcts.push_back(*o.active_jct_ms);
                    } else {
                        core_ok_active_jcts.push_back(std::max(0.0, *o.e2e_ms - o.think_time_ms));
                    }
                } else if (o.active_jct_ms.has_value()) {
                    core_ok_active_jcts.push_back(*o.active_jct_ms);
                }
                for (double v : o.ar_ttfts_ms) core_ok_ttfts.push_back(v);
                for (double v : o.ar_tpots_ms) core_ok_tpots.push_back(v);
                if (o.turn_records.empty()) {
                    for (double v : o.dit_turns_ms) core_ok_dits.push_back(v);
                }
            }
        } else if (is_oom) {
            ++oom_cnt;
            if (core) ++core_oom_cnt;
        } else {
            ++err_cnt;
            if (core) ++core_err_cnt;
        }

        const SessionTurnCounts tc = evaluate_session_turn_counts(
            policy,
            o,
            std::nullopt,
            eff_ar_tpot_slo_ms,
            std::nullopt,
            scale,
            joint_ok,
            ar_primary_ok,
            dit_ok);
        total_planned_turns += tc.planned_turns;
        total_completed_turns += tc.completed_turns;
        total_turn_joint_ok += tc.joint_ok_turns;
        if (core) {
            core_planned_turns += tc.planned_turns;
            core_completed_turns += tc.completed_turns;
            core_turn_joint_ok += tc.joint_ok_turns;
            core_planned_ar_turns += tc.planned_ar_turns;
            core_turn_ar_ok += tc.ar_ok_turns;
            core_planned_dit_turns += tc.planned_dit_turns;
            core_turn_dit_ok += tc.dit_ok_turns;

            const bool sess_has_ar =
                (tc.planned_ar_turns > 0) || !o.ar_ttfts_ms.empty() || !o.ar_tpots_ms.empty();
            const bool sess_has_dit =
                (tc.planned_dit_turns > 0) || !o.dit_turns_ms.empty();
            if (sess_has_ar) {
                ++core_sessions_with_ar;
                if (is_ok && ar_primary_ok) ++core_ar_cnt;
                if (is_ok && ar_mean_ok) ++core_ar_mean_cnt;
                if (is_ok && ar_strict_max_ok) ++core_ar_strict_max_cnt;
            }
            if (sess_has_dit) {
                ++core_sessions_with_dit;
                if (is_ok && dit_ok) ++core_dit_cnt;
            }
        }

        path_a_ships += o.path_a_ships;
        path_b_morphs += o.path_b_morphs;
        preempted_ar += o.preempted_ar_turns;
        total_yields += o.total_yields;
        oom_retries += o.oom_retries;
        cross_node_ships += o.cross_node_ships;
        max_xnode_bytes = std::max(max_xnode_bytes, o.max_cross_node_ship_bytes);

        // Unbiased per-turn stats strictly over steady-state core outcomes:
        // iterate planned turns across ALL steady-state sessions (including unreached turns
        // after an earlier OOM/error so turn_1/2/3 denominators have zero survivor bias).
        if (core) {
            std::map<int, const TurnRecordInput*> rec_by_idx;
            int max_rec_idx = -1;
            bool any_turn_breakdown = false;
            for (const auto& tr : o.turn_records) {
                rec_by_idx[tr.turn_idx] = &tr;
                max_rec_idx = std::max(max_rec_idx, tr.turn_idx);
                if (tr.status == "ok") {
                    if (tr.dit_ms.has_value()) {
                        core_ok_dits.push_back(*tr.dit_ms);
                    }
                    if (tr.queue_ms.has_value() || tr.h2d_weight_ms.has_value() || tr.kv_transfer_ms.has_value() ||
                        tr.vae_ms.has_value() || tr.preempt_yield_ms.has_value()) {
                        any_turn_breakdown = true;
                        if (tr.queue_ms.has_value()) bd_queue_ms.push_back(std::max(0.0, *tr.queue_ms));
                        if (tr.h2d_weight_ms.has_value()) bd_h2d_weight_ms.push_back(std::max(0.0, *tr.h2d_weight_ms));
                        if (tr.kv_transfer_ms.has_value()) bd_kv_transfer_ms.push_back(std::max(0.0, *tr.kv_transfer_ms));
                        if (tr.vae_ms.has_value()) bd_vae_ms.push_back(std::max(0.0, *tr.vae_ms));
                        if (tr.preempt_yield_ms.has_value()) bd_preempt_yield_ms.push_back(std::max(0.0, *tr.preempt_yield_ms));
                    }
                }
            }
            if (!any_turn_breakdown && is_ok) {
                if (o.queue_ms.has_value()) bd_queue_ms.push_back(std::max(0.0, *o.queue_ms));
                if (o.h2d_weight_ms.has_value()) bd_h2d_weight_ms.push_back(std::max(0.0, *o.h2d_weight_ms));
                if (o.kv_transfer_ms.has_value()) bd_kv_transfer_ms.push_back(std::max(0.0, *o.kv_transfer_ms));
                if (o.vae_ms.has_value()) bd_vae_ms.push_back(std::max(0.0, *o.vae_ms));
                if (o.preempt_yield_ms.has_value()) bd_preempt_yield_ms.push_back(std::max(0.0, *o.preempt_yield_ms));
            }

            if (!o.turn_records.empty() || (!is_ok && o.planned_turns > 0)) {
                const int n_planned_idx = std::max({o.planned_turns, static_cast<int>(o.turn_records.size()), max_rec_idx + 1});
                int prev_dit_height = 0;
                for (int tidx = 0; tidx < n_planned_idx; ++tidx) {
                    auto& ta = turn_accs[tidx];
                    ++ta.total;
                    auto rit = rec_by_idx.find(tidx);
                    if (rit == rec_by_idx.end()) {
                        ++ta.unreached;
                        continue;
                    }
                    const auto& tr = *(rit->second);
                    const double turn_ttft_cap = resolve_turn_ar_ttft_cap_ms(
                        policy, tr, o.quadrant, prev_dit_height, std::nullopt, scale);
                    if (tr.status == "ok") {
                        ++ta.ok;
                        const TurnSloEval tev = evaluate_turn_record_slo(
                            policy, tr, turn_ttft_cap, eff_ar_tpot_slo_ms, std::nullopt, scale);
                        if (tev.joint_ok) {
                            ++ta.slo_met;
                        }
                        if (tr.ar_ttft_ms.has_value()) {
                            ta.ttfts.push_back(*tr.ar_ttft_ms);
                        }
                        if (!tr.ar_tpots_ms.empty()) {
                            for (double v : tr.ar_tpots_ms) ta.tpots.push_back(v);
                        } else if (tr.ar_tpot_p95_ms.has_value()) {
                            ta.tpots.push_back(*tr.ar_tpot_p95_ms);
                        }
                        if (tr.dit_ms.has_value()) {
                            ta.dits.push_back(*tr.dit_ms);
                        }
                        if (tr.kv_mib.has_value()) {
                            ta.kv_mibs.push_back(*tr.kv_mib);
                        }
                        if (tev.has_dit) {
                            prev_dit_height = (tr.height > 0) ? tr.height : 512;
                        }
                    } else if (tr.status == "oom") {
                        ++ta.oom;
                    }
                }
            }
        }

        // 10-second time-series bucket over all outcomes
        const int bidx = static_cast<int>(std::floor(std::max(0.0, o.arrival_offset_s) / 10.0));
        auto& ba = ts_buckets[bidx];
        ++ba.sessions;
        if (is_ok) ++ba.ok;
        if (is_oom) ++ba.oom;
        if (joint_ok) ++ba.slo_met;
        if (!has_realtime_alpha && o.resident_alpha_sample.has_value()) {
            ba.alpha_sum += *o.resident_alpha_sample;
            ++ba.alpha_cnt;
        }
        ba.path_a_ships += o.path_a_ships;
        ba.path_b_morphs += o.path_b_morphs;
        ba.yields += o.total_yields;
        if (!o.phase_label.empty()) {
            ++ba.phase_counts[o.phase_label];
        }
        if (is_ok) {
            for (double v : o.ar_ttfts_ms) ba.ttfts.push_back(v);
            for (double v : o.dit_turns_ms) ba.dits.push_back(v);
        }
    }

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

    // Improper DiT CDF uses planned steady-state DiT turns as denominator (not ok_dits + failed_sessions).
    const size_t core_total_dit_units = std::max(core_planned_dit_turns, core_ok_dits.size());
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

    const size_t n_core_turns = std::max<size_t>(1, core_planned_turns);
    const size_t n_tot_turns = std::max<size_t>(1, total_planned_turns);

    const std::string eff_fam = (policy.model_family == "internvl_u") ? "internvl_u" : "bagel";
    const int eff_steps_512 = (eff_fam == "internvl_u") ? 16 : 20;
    const int eff_steps_1024 = (eff_fam == "internvl_u") ? 16 : 20;

    nlohmann::json slo_curve = nlohmann::json::object();
    for (const auto& [k_label, k_val] : std::vector<std::pair<std::string, double>>{
             {"1.5x", 1.5}, {"2.0x", 2.0}, {"2.5x", 2.5}, {"3.0x", 3.0}}) {
        const auto slo_k = default_operator_profile().slo_thresholds_ms(k_val, eff_fam);
        const double tpot_k = slo_k.at("ar_tpot_slo_ms") * scale;
        size_t k_cnt = 0;
        size_t k_turn_ok = 0;
        size_t k_turn_tot = 0;
        for (size_t idx : core_indices) {
            const bool s_ok = evaluate_outcome_at_k(policy, parsed_outcomes[idx], k_val, k_val, scale);
            if (s_ok) {
                ++k_cnt;
            }
            const auto tck = evaluate_session_turn_counts(
                policy, parsed_outcomes[idx], k_val, tpot_k, k_val, scale, s_ok, s_ok, s_ok);
            k_turn_ok += tck.joint_ok_turns;
            k_turn_tot += tck.planned_turns;
        }
        const double frac = static_cast<double>(k_cnt) / static_cast<double>(n_core);
        const double eff_rps = round_4(static_cast<double>(k_cnt) / eff_steady_s);
        const double t_frac = static_cast<double>(k_turn_ok) / static_cast<double>(std::max<size_t>(1, k_turn_tot));
        const double t_rps = round_4(actual_steady_arrival_qps * t_frac);
        const double t_tps = round_4(static_cast<double>(k_turn_ok) / eff_steady_s);
        slo_curve[k_label] = {
            {"attainment_pct", round_1(100.0 * frac)},
            {"effective_slo_throughput_rps", eff_rps},
            {"goodput_rps", eff_rps},
            {"turn_attainment_pct", round_1(100.0 * t_frac)},
            {"turn_goodput_rps", t_rps},
            {"turn_goodput_tps", t_tps},
            {"dit_512_slo_ms", round_2(policy.dit_turn_slo_for(512, eff_steps_512, k_val) * scale)},
            {"dit_1024_slo_ms", round_2(policy.dit_turn_slo_for(1024, eff_steps_1024, k_val) * scale)},
        };
    }

    // D4: Decoupled DiT sensitivity table (`k_AR` fixed at policy AR thresholds, `k_DiT in {2.0x, 3.0x, 4.0x}`).
    nlohmann::json slo_curve_by_dit_k = nlohmann::json::object();
    for (const auto& [k_label, k_dit] : std::vector<std::pair<std::string, double>>{
             {"2.0x", 2.0}, {"3.0x", 3.0}, {"4.0x", 4.0}}) {
        size_t s_ok_cnt = 0;
        size_t t_ok_cnt = 0;
        size_t t_tot_cnt = 0;
        for (size_t idx : core_indices) {
            const bool s_ok = evaluate_outcome_at_k(policy, parsed_outcomes[idx], policy.slo_scale_k, k_dit, scale);
            if (s_ok) {
                ++s_ok_cnt;
            }
            const auto tck = evaluate_session_turn_counts(
                policy, parsed_outcomes[idx], std::nullopt, eff_ar_tpot_slo_ms, k_dit, scale, s_ok, s_ok, s_ok);
            t_ok_cnt += tck.joint_ok_turns;
            t_tot_cnt += tck.planned_turns;
        }
        const double s_frac = static_cast<double>(s_ok_cnt) / static_cast<double>(n_core);
        const double t_frac = static_cast<double>(t_ok_cnt) / static_cast<double>(std::max<size_t>(1, t_tot_cnt));
        slo_curve_by_dit_k[k_label] = {
            {"turn_attainment_pct", round_1(100.0 * t_frac)},
            {"turn_goodput_rps", round_4(actual_steady_arrival_qps * t_frac)},
            {"turn_goodput_tps", round_4(static_cast<double>(t_ok_cnt) / eff_steady_s)},
            {"session_attainment_pct", round_1(100.0 * s_frac)},
            {"session_goodput_rps", round_4(static_cast<double>(s_ok_cnt) / eff_steady_s)},
            {"ar_ttft_slo_ms", round_2(eff_ar_ttft_slo_ms)},
            {"ar_tpot_slo_ms", round_2(eff_ar_tpot_slo_ms)},
            {"dit_512_slo_ms", round_2(policy.dit_turn_slo_for(512, eff_steps_512, k_dit) * scale)},
            {"dit_1024_slo_ms", round_2(policy.dit_turn_slo_for(1024, eff_steps_1024, k_dit) * scale)},
        };
    }

    nlohmann::json per_turn_stats = nlohmann::json::object();
    for (const auto& [tidx, acc] : turn_accs) {
        const double surv =
            (acc.total == 0) ? 100.0 : round_1(100.0 * static_cast<double>(acc.ok) / static_cast<double>(acc.total));
        const double t_slo_pct =
            (acc.total == 0) ? 0.0 : round_1(100.0 * static_cast<double>(acc.slo_met) / static_cast<double>(acc.total));
        per_turn_stats["turn_" + std::to_string(tidx)] = {
            {"total", acc.total},
            {"ok", acc.ok},
            {"oom", acc.oom},
            {"unreached", acc.unreached},
            {"slo_met", acc.slo_met},
            {"slo_attainment_pct", t_slo_pct},
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

    const double steady_joint_frac = static_cast<double>(core_joint_cnt) / static_cast<double>(n_core);
    const double full_joint_frac = static_cast<double>(joint_cnt) / static_cast<double>(n_tot);
    const double steady_joint_goodput_rps = round_4(static_cast<double>(core_joint_cnt) / eff_steady_s);
    const double steady_joint_att_pct = round_1(100.0 * steady_joint_frac);
    const double full_trace_goodput_rps = round_4(static_cast<double>(joint_cnt) / eff_full_s);
    const double full_trace_joint_att_pct = round_1(100.0 * full_joint_frac);
    const double steady_ar_att_pct =
        (core_sessions_with_ar == 0)
            ? 100.0
            : round_1(100.0 * static_cast<double>(core_ar_cnt) / static_cast<double>(core_sessions_with_ar));
    const double steady_ar_mean_pct =
        (core_sessions_with_ar == 0)
            ? 100.0
            : round_1(100.0 * static_cast<double>(core_ar_mean_cnt) / static_cast<double>(core_sessions_with_ar));
    const double steady_ar_strict_max_pct =
        (core_sessions_with_ar == 0)
            ? 100.0
            : round_1(100.0 * static_cast<double>(core_ar_strict_max_cnt) / static_cast<double>(core_sessions_with_ar));
    const double steady_dit_att_pct =
        (core_sessions_with_dit == 0)
            ? 100.0
            : round_1(100.0 * static_cast<double>(core_dit_cnt) / static_cast<double>(core_sessions_with_dit));

    const double steady_turn_att_frac = static_cast<double>(core_turn_joint_ok) / static_cast<double>(n_core_turns);
    const double steady_turn_goodput_rps = round_4(actual_steady_arrival_qps * steady_turn_att_frac);
    const double steady_turn_goodput_tps = round_4(static_cast<double>(core_turn_joint_ok) / eff_steady_s);
    const double full_turn_att_frac = static_cast<double>(total_turn_joint_ok) / static_cast<double>(n_tot_turns);
    const double steady_turn_surv_pct =
        (core_planned_turns == 0)
            ? 100.0
            : round_1(100.0 * static_cast<double>(core_completed_turns) / static_cast<double>(n_core_turns));
    const double steady_turn_ar_pct =
        (core_planned_ar_turns == 0)
            ? 100.0
            : round_1(100.0 * static_cast<double>(core_turn_ar_ok) / static_cast<double>(core_planned_ar_turns));
    const double steady_turn_dit_pct =
        (core_planned_dit_turns == 0)
            ? 100.0
            : round_1(100.0 * static_cast<double>(core_turn_dit_ok) / static_cast<double>(core_planned_dit_turns));

    auto summarize_breakdown = [](const std::vector<double>& vals) -> nlohmann::json {
        return {
            {"mean", round_2(mean_raw(vals))},
            {"p95", percentile_rounded(vals, 95.0)},
            {"count", vals.size()},
        };
    };
    const nlohmann::json latency_breakdown_ms = {
        {"queue_ms", summarize_breakdown(bd_queue_ms)},
        {"h2d_weight_ms", summarize_breakdown(bd_h2d_weight_ms)},
        {"kv_transfer_ms", summarize_breakdown(bd_kv_transfer_ms)},
        {"vae_ms", summarize_breakdown(bd_vae_ms)},
        {"preempt_yield_ms", summarize_breakdown(bd_preempt_yield_ms)},
    };
    const nlohmann::json nvml_sm_util_j =
        sm_util_samples.empty() ? nlohmann::json(nullptr) : nlohmann::json(round_2(mean_raw(sm_util_samples)));
    const nlohmann::json nvml_mem_util_j =
        mem_util_samples.empty() ? nlohmann::json(nullptr) : nlohmann::json(round_2(mean_raw(mem_util_samples)));
    const nlohmann::json eff_hbm_util_j =
        eff_hbm_util_samples.empty() ? nlohmann::json(nullptr) : nlohmann::json(round_2(mean_raw(eff_hbm_util_samples)));

    const int kv_total_restores = kv_host_swap_ins + kv_recompute_restores;
    const double kv_swap_hit_rate_pct =
        (kv_total_restores > 0)
            ? round_1(100.0 * static_cast<double>(kv_host_swap_ins) / static_cast<double>(kv_total_restores))
            : (policy.kv_eviction_mode == "recompute" ? 0.0 : 100.0);

    const nlohmann::json slo_config = {
        {"k_ar", policy.slo_scale_k},
        {"k_dit", policy.dit_slo_scale_k},
        {"ar_ttft_slo_ms", round_2(eff_ar_ttft_slo_ms)},
        {"ar_ttft_slo_text_ms", round_2(policy.ar_ttft_slo_for(false, 0, 512) * scale)},
        {"ar_ttft_slo_img512_ms", round_2(policy.ar_ttft_slo_for(true, 1, 512) * scale)},
        {"ar_ttft_slo_img1024_ms", round_2(policy.ar_ttft_slo_for(true, 1, 1024) * scale)},
        {"ar_ttft_modality_normalized", true},
        {"ar_tpot_slo_ms", round_2(eff_ar_tpot_slo_ms)},
        {"ar_tpot_slo_mode", policy.ar_tpot_slo_mode},
        {"dit_512_slo_ms", round_2(policy.dit_turn_slo_for(512, eff_steps_512) * scale)},
        {"dit_1024_slo_ms", round_2(policy.dit_turn_slo_for(1024, eff_steps_1024) * scale)},
        {"cot_dit_decoupled", true},
        {"primary_granularity", "turn_level"},
        {"turn_oom_accounting", "planned_turns_denominator"},
        {"cost_model_mode", policy.cost_model_mode},
        {"gpu_queue_mode", policy.gpu_queue_mode},
        {"kv_eviction_mode", policy.kv_eviction_mode},
    };

    nlohmann::json report = {
        {"policy_name", policy.policy_name},
        {"suite_name", suite_name},
        {"num_gpus", policy.num_gpus},
        {"offered_qps", round_3(offered_qps)},
        {"actual_arrival_qps", round_4(actual_arrival_qps)},
        {"actual_steady_arrival_qps", round_4(actual_steady_arrival_qps)},
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
        {"steady_sessions_with_ar", core_sessions_with_ar},
        {"steady_sessions_with_dit", core_sessions_with_dit},
        {"steady_state_survival_pct", steady_survival_pct},
        {"steady_oom_sessions", core_oom_cnt},
        {"total_planned_turns", total_planned_turns},
        {"completed_turns", total_completed_turns},
        {"turn_slo_attainment_pct", round_1(100.0 * full_turn_att_frac)},
        {"turn_slo_goodput_rps", round_4(actual_arrival_qps * full_turn_att_frac)},
        {"steady_planned_turns", core_planned_turns},
        {"steady_completed_turns", core_completed_turns},
        {"steady_turn_survival_pct", steady_turn_surv_pct},
        {"steady_turn_slo_attainment_pct", round_1(100.0 * steady_turn_att_frac)},
        {"steady_turn_ar_slo_attainment_pct", steady_turn_ar_pct},
        {"steady_turn_dit_slo_attainment_pct", steady_turn_dit_pct},
        {"steady_turn_slo_goodput_rps", steady_turn_goodput_rps},
        {"steady_turn_goodput_tps", steady_turn_goodput_tps},
        {"dit_turn_survival_pct", dit_survival_pct},
        // Unified canonical steady-state Joint SLO semantics across live coordinator and simulator:
        {"joint_slo_goodput_rps", steady_joint_goodput_rps},
        {"steady_joint_slo_goodput_rps", steady_joint_goodput_rps},
        {"effective_slo_throughput_rps", steady_joint_goodput_rps},
        {"steady_state_goodput_rps", steady_joint_goodput_rps},
        {"full_trace_goodput_rps", full_trace_goodput_rps},
        {"raw_wall_goodput_rps", round_4(static_cast<double>(joint_cnt) / wall_s)},
        {"joint_slo_attainment_pct", steady_joint_att_pct},
        {"steady_joint_slo_attainment_pct", steady_joint_att_pct},
        {"steady_state_attainment_pct", steady_joint_att_pct},
        {"full_trace_joint_slo_attainment_pct", full_trace_joint_att_pct},
        {"full_trace_attainment_pct", full_trace_joint_att_pct},
        {"steady_ar_slo_attainment_pct", steady_ar_att_pct},
        {"steady_ar_slo_mean_pct", steady_ar_mean_pct},
        {"steady_ar_slo_strict_max_pct", steady_ar_strict_max_pct},
        {"steady_dit_slo_attainment_pct", steady_dit_att_pct},
        {"ar_slo_attainment_pct", steady_ar_att_pct},
        {"dit_slo_attainment_pct", steady_dit_att_pct},
        {"ar_ttft_mean_ms", round_2(mean_raw(core_ok_ttfts))},
        {"ar_ttft_p50_ms", percentile_rounded(core_ok_ttfts, 50.0)},
        {"ar_ttft_p95_ms", percentile_rounded(core_ok_ttfts, 95.0)},
        {"ar_ttft_p99_ms", percentile_rounded(core_ok_ttfts, 99.0)},
        {"ar_tpot_mean_ms", round_2(mean_raw(core_ok_tpots))},
        {"ar_tpot_p50_ms", percentile_rounded(core_ok_tpots, 50.0)},
        {"ar_tpot_p95_ms", percentile_rounded(core_ok_tpots, 95.0)},
        {"ar_tpot_p99_ms", percentile_rounded(core_ok_tpots, 99.0)},
        {"dit_turn_mean_ms", round_2(mean_raw(core_ok_dits))},
        {"dit_turn_p50_ms", percentile_rounded(core_ok_dits, 50.0)},
        {"dit_turn_p95_ms", percentile_rounded(core_ok_dits, 95.0)},
        {"dit_turn_p99_ms", percentile_rounded(core_ok_dits, 99.0)},
        {"session_e2e_mean_ms", round_2(mean_raw(core_ok_e2es))},
        {"session_e2e_p50_ms", percentile_rounded(core_ok_e2es, 50.0)},
        {"session_e2e_p95_ms", percentile_rounded(core_ok_e2es, 95.0)},
        {"session_e2e_p99_ms", percentile_rounded(core_ok_e2es, 99.0)},
        {"active_jct_mean_ms", round_2(mean_raw(core_ok_active_jcts))},
        {"active_jct_p50_ms", percentile_rounded(core_ok_active_jcts, 50.0)},
        {"active_jct_p95_ms", percentile_rounded(core_ok_active_jcts, 95.0)},
        {"active_jct_p99_ms", percentile_rounded(core_ok_active_jcts, 99.0)},
        {"all_session_active_jct_p50_ms", failure_aware_quantile(core_ok_active_jcts, steady_sessions_cnt, 50.0)},
        {"all_session_active_jct_p90_ms", failure_aware_quantile(core_ok_active_jcts, steady_sessions_cnt, 90.0)},
        {"all_session_active_jct_p95_ms", failure_aware_quantile(core_ok_active_jcts, steady_sessions_cnt, 95.0)},
        {"all_session_active_jct_p99_ms", failure_aware_quantile(core_ok_active_jcts, steady_sessions_cnt, 99.0)},
        {"all_session_e2e_p50_ms", failure_aware_quantile(core_ok_e2es, steady_sessions_cnt, 50.0)},
        {"all_session_e2e_p90_ms", failure_aware_quantile(core_ok_e2es, steady_sessions_cnt, 90.0)},
        {"all_session_e2e_p95_ms", failure_aware_quantile(core_ok_e2es, steady_sessions_cnt, 95.0)},
        {"all_session_e2e_p99_ms", failure_aware_quantile(core_ok_e2es, steady_sessions_cnt, 99.0)},
        {"all_session_dit_p50_ms", failure_aware_quantile(core_ok_dits, core_total_dit_units, 50.0)},
        {"all_session_dit_p90_ms", failure_aware_quantile(core_ok_dits, core_total_dit_units, 90.0)},
        {"all_session_dit_p95_ms", failure_aware_quantile(core_ok_dits, core_total_dit_units, 95.0)},
        {"all_session_dit_p99_ms", failure_aware_quantile(core_ok_dits, core_total_dit_units, 99.0)},
        {"active_jct_cdf_points", build_improper_cdf_points(core_ok_active_jcts, steady_sessions_cnt)},
        {"e2e_cdf_points", build_improper_cdf_points(core_ok_e2es, steady_sessions_cnt)},
        {"dit_cdf_points", build_improper_cdf_points(core_ok_dits, core_total_dit_units)},
        {"latency_breakdown_ms", latency_breakdown_ms},
        {"nvml_sm_util_mean_pct", nvml_sm_util_j},
        {"nvml_mem_util_mean_pct", nvml_mem_util_j},
        {"effective_hbm_util_mean_pct", eff_hbm_util_j},
        {"cost_model_mode", policy.cost_model_mode},
        {"gpu_queue_mode", policy.gpu_queue_mode},
        {"kv_eviction_mode", policy.kv_eviction_mode},
        {"queue_reorders_total", 0},
        {"queue_memory_bypasses_total", 0},
        {"queue_starvation_promotions_total", 0},
        {"peak_gpu_reserved_mib", round_2(peak_res_mib)},
        {"peak_nvml_used_mib", round_2(peak_nvml_mib)},
        {"peak_pytorch_reserved_mib", round_2(peak_pytorch_res_mib)},
        {"vmm_physical_pool_mib", round_2(max_vmm_pool_mib)},
        {"host_shmem_delta_gib", host_shmem_delta_gib},
        {"kv_host_swap_outs", kv_host_swap_outs},
        {"kv_host_swap_ins", kv_host_swap_ins},
        {"kv_recompute_events", kv_recompute_restores},
        {"kv_recompute_evictions", kv_recompute_evictions},
        {"kv_swap_hit_rate_pct", kv_swap_hit_rate_pct},
        {"kv_swap_in_mean_ms", 0.0},
        {"kv_swap_in_p95_ms", 0.0},
        {"kv_recompute_total_ms", 0.0},
        {"kv_saved_recompute_ms", 0.0},
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
        {"slo_config", slo_config},
        {"slo_curve_by_k", slo_curve},
        {"slo_curve_by_dit_k", slo_curve_by_dit_k},
        {"per_turn_stats", per_turn_stats},
        {"time_series_10s", time_series_10s},
        {"status", (err_cnt == 0) ? "ok" : "check_failed"},
    };

    if (include_sessions) {
        report["sessions"] = outcomes_array;
    }
    if (sim_extra_fields.is_object()) {
        for (auto it = sim_extra_fields.begin(); it != sim_extra_fields.end(); ++it) {
            if (it.key() == "alpha_timeline" || it.key() == "nvml_sm_util_mean_pct" ||
                it.key() == "nvml_mem_util_mean_pct" || it.key() == "effective_hbm_util_mean_pct") {
                continue;
            }
            report[it.key()] = it.value();
        }
    }
    return report;
}

}  // namespace morphserve::control
