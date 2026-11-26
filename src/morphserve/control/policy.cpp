#include "morphserve/control/policy.hpp"

#include <algorithm>
#include <stdexcept>

namespace morphserve::control {

namespace {

void configure_static_fission_ranks(
    PolicyConfig& cfg,
    const std::string& policy_name,
    int num_gpus) {
    if (num_gpus < 2) {
        throw std::invalid_argument(policy_name + " requires >= 2 GPUs");
    }
    cfg.static_und_ranks.clear();
    cfg.static_gen_ranks.clear();

    if (policy_name == "static_2und_6gen") {
        if (num_gpus >= 8 && (num_gpus % 8) == 0) {
            for (int g = 0; g < num_gpus; ++g) {
                const int r = g % 8;
                if (r == 0 || r == 4) cfg.static_und_ranks.push_back(g);
                else cfg.static_gen_ranks.push_back(g);
            }
        } else if (num_gpus >= 4 && (num_gpus % 4) == 0) {
            const int q = num_gpus / 4;
            for (int g = 0; g < q; ++g) cfg.static_und_ranks.push_back(g);
            for (int g = q; g < num_gpus; ++g) cfg.static_gen_ranks.push_back(g);
        } else {
            cfg.static_und_ranks.push_back(0);
            for (int g = 1; g < num_gpus; ++g) cfg.static_gen_ranks.push_back(g);
        }
    } else if (policy_name == "static_6und_2gen") {
        if (num_gpus >= 8 && (num_gpus % 8) == 0) {
            for (int g = 0; g < num_gpus; ++g) {
                const int r = g % 8;
                if (r == 0 || r == 1 || r == 2 || r == 4 || r == 5 || r == 6) {
                    cfg.static_und_ranks.push_back(g);
                } else {
                    cfg.static_gen_ranks.push_back(g);
                }
            }
        } else if (num_gpus >= 4 && (num_gpus % 4) == 0) {
            const int q = num_gpus / 4;
            for (int g = 0; g < 3 * q; ++g) cfg.static_und_ranks.push_back(g);
            for (int g = 3 * q; g < num_gpus; ++g) cfg.static_gen_ranks.push_back(g);
        } else {
            for (int g = 0; g < num_gpus - 1; ++g) cfg.static_und_ranks.push_back(g);
            cfg.static_gen_ranks.push_back(num_gpus - 1);
        }
    } else {
        // `static_nccl_fission` and `static_4und_4gen` (balanced 50/50 split)
        if (num_gpus >= 8 && (num_gpus % 8) == 0) {
            for (int g = 0; g < num_gpus; ++g) {
                const int r = g % 8;
                if (r == 0 || r == 1 || r == 4 || r == 5) cfg.static_und_ranks.push_back(g);
                else cfg.static_gen_ranks.push_back(g);
            }
        } else {
            const int half = num_gpus / 2;
            for (int g = 0; g < half; ++g) cfg.static_und_ranks.push_back(g);
            for (int g = half; g < num_gpus; ++g) cfg.static_gen_ranks.push_back(g);
        }
    }

    cfg.routing_mode = "static_nccl_fission";
    cfg.use_tripool_vmm = false;
    cfg.use_expert_pager = false;
    cfg.use_scoped_vae = false;
    cfg.use_shared_numa_mirror = false;
    cfg.use_h2d_pacer = false;
    cfg.use_module3_cow_loopback = false;
    cfg.use_naive_offload = false;
    cfg.kv_vmm_only = false;
    cfg.preemption_mode = "non_preemptive";
    for (int g = 0; g < num_gpus; ++g) {
        const bool is_und = std::find(cfg.static_und_ranks.begin(), cfg.static_und_ranks.end(), g) !=
                            cfg.static_und_ranks.end();
        cfg.initial_alphas[g] = is_und ? 0.0 : 1.0;
    }
}

void apply_policy_overrides(PolicyConfig& cfg, const nlohmann::json& j) {
    if (!j.is_object()) return;
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string& k = it.key();
        const auto& v = it.value();
        if (k == "cptg_arch_override") {
            if (v.is_null()) cfg.cptg_arch_override.reset();
            else cfg.cptg_arch_override = v.get<std::string>();
            continue;
        }
        if (k == "vendor_dir") {
            if (v.is_null()) cfg.vendor_dir.reset();
            else cfg.vendor_dir = v.get<std::string>();
            continue;
        }
        if (k == "dit_turn_slo_override_ms") {
            if (v.is_null()) cfg.dit_turn_slo_override_ms.reset();
            else cfg.dit_turn_slo_override_ms = v.get<double>();
            continue;
        }
        if (k == "gpu_memory_fraction") {
            if (v.is_null()) cfg.gpu_memory_fraction.reset();
            else cfg.gpu_memory_fraction = v.get<double>();
            continue;
        }
        if (k == "gpu_memory_quota_mib") {
            if (v.is_null()) cfg.gpu_memory_quota_mib.reset();
            else cfg.gpu_memory_quota_mib = v.get<double>();
            continue;
        }

        if (k == "policy_name") {
            if (!v.is_null()) cfg.policy_name = v.get<std::string>();
        } else if (k == "num_gpus") {
            if (!v.is_null()) cfg.num_gpus = v.get<int>();
        } else if (k == "model_family") {
            if (!v.is_null()) cfg.model_family = v.get<std::string>();
        } else if (k == "routing_mode") {
            if (!v.is_null()) cfg.routing_mode = v.get<std::string>();
        } else if (k == "use_tripool_vmm") {
            if (!v.is_null()) cfg.use_tripool_vmm = v.get<bool>();
        } else if (k == "use_expert_pager") {
            if (!v.is_null()) cfg.use_expert_pager = v.get<bool>();
        } else if (k == "use_scoped_vae") {
            if (!v.is_null()) cfg.use_scoped_vae = v.get<bool>();
        } else if (k == "use_shared_numa_mirror") {
            if (!v.is_null()) cfg.use_shared_numa_mirror = v.get<bool>();
        } else if (k == "use_h2d_pacer") {
            if (!v.is_null()) cfg.use_h2d_pacer = v.get<bool>();
        } else if (k == "use_module3_cow_loopback") {
            if (!v.is_null()) cfg.use_module3_cow_loopback = v.get<bool>();
        } else if (k == "allow_cascade_cfg_split") {
            if (!v.is_null()) cfg.allow_cascade_cfg_split = v.get<bool>();
        } else if (k == "use_naive_offload") {
            if (!v.is_null()) cfg.use_naive_offload = v.get<bool>();
        } else if (k == "kv_vmm_only") {
            if (!v.is_null()) cfg.kv_vmm_only = v.get<bool>();
        } else if (k == "preemption_mode") {
            if (!v.is_null()) cfg.preemption_mode = v.get<std::string>();
        } else if (k == "static_und_ranks") {
            if (!v.is_null()) cfg.static_und_ranks = v.get<std::vector<int>>();
        } else if (k == "static_gen_ranks") {
            if (!v.is_null()) cfg.static_gen_ranks = v.get<std::vector<int>>();
        } else if (k == "initial_alphas") {
            if (v.is_object()) {
                cfg.initial_alphas.clear();
                for (auto ait = v.begin(); ait != v.end(); ++ait) {
                    cfg.initial_alphas[std::stoi(ait.key())] = ait.value().get<double>();
                }
            }
        } else if (k == "slo_scale_k") {
            if (!v.is_null()) cfg.slo_scale_k = v.get<double>();
        } else if (k == "dit_slo_scale_k") {
            if (!v.is_null()) cfg.dit_slo_scale_k = v.get<double>();
        } else if (k == "ar_tpot_slo_ms") {
            if (!v.is_null()) cfg.ar_tpot_slo_ms = v.get<double>();
        } else if (k == "ar_ttft_slo_ms") {
            if (!v.is_null()) cfg.ar_ttft_slo_ms = v.get<double>();
        } else if (k == "dit_512_slo_ms") {
            if (!v.is_null()) cfg.dit_512_slo_ms = v.get<double>();
        } else if (k == "dit_1024_slo_ms") {
            if (!v.is_null()) cfg.dit_1024_slo_ms = v.get<double>();
        } else if (k == "max_oom_retries") {
            if (!v.is_null()) cfg.max_oom_retries = v.get<int>();
        } else if (k == "inter_turn_think_s_range") {
            if (v.is_array() && v.size() == 2) {
                cfg.inter_turn_think_s_range = {v[0].get<double>(), v[1].get<double>()};
            }
        } else if (k == "think_time_dist") {
            if (!v.is_null()) cfg.think_time_dist = v.get<std::string>();
        } else if (k == "think_time_median_s") {
            if (!v.is_null()) cfg.think_time_median_s = v.get<double>();
        } else if (k == "think_time_sigma") {
            if (!v.is_null()) cfg.think_time_sigma = v.get<double>();
        } else if (k == "ar_tpot_slo_mode") {
            if (!v.is_null()) cfg.ar_tpot_slo_mode = v.get<std::string>();
        } else if (k == "steady_window_mode") {
            if (!v.is_null()) cfg.steady_window_mode = v.get<std::string>();
        } else if (k == "warmup_window_s") {
            if (!v.is_null()) cfg.warmup_window_s = v.get<double>();
        } else if (k == "steady_window_s") {
            if (!v.is_null()) cfg.steady_window_s = v.get<double>();
        } else if (k == "steady_state_trim_frac") {
            if (!v.is_null()) cfg.steady_state_trim_frac = v.get<double>();
        } else if (k == "super_pool_mib") {
            if (!v.is_null()) cfg.super_pool_mib = v.get<int>();
        } else if (k == "fine_kv_pool_mib") {
            if (!v.is_null()) cfg.fine_kv_pool_mib = v.get<int>();
        } else if (k == "super_handle_mib") {
            if (!v.is_null()) cfg.super_handle_mib = v.get<int>();
        } else if (k == "transient_va_gib") {
            if (!v.is_null()) cfg.transient_va_gib = v.get<int>();
        } else if (k == "kv_va_gib") {
            if (!v.is_null()) cfg.kv_va_gib = v.get<int>();
        } else if (k == "allow_vmm_overflow") {
            if (!v.is_null()) cfg.allow_vmm_overflow = v.get<bool>();
        } else {
            throw std::invalid_argument("unknown EnginePolicyConfig field: " + k);
        }
    }
}

}  // namespace

bool is_valid_policy_name(const std::string& policy_name) noexcept {
    for (const char* name : kValidPolicies) {
        if (policy_name == name) {
            return true;
        }
    }
    for (const char* name : kStaticSplitPolicies) {
        if (policy_name == name) {
            return true;
        }
    }
    return false;
}

std::string PolicyConfig::cptg_architecture() const {
    if (cptg_arch_override.has_value() && !cptg_arch_override->empty()) {
        return *cptg_arch_override;
    }
    return (model_family == "internvl_u") ? "bridge_mllm_dit" : "tight_mot";
}

std::string PolicyConfig::role_of(int rank) const {
    if (!static_und_ranks.empty() &&
        std::find(static_und_ranks.begin(), static_und_ranks.end(), rank) != static_und_ranks.end()) {
        return "und";
    }
    if (!static_gen_ranks.empty() &&
        std::find(static_gen_ranks.begin(), static_gen_ranks.end(), rank) != static_gen_ranks.end()) {
        return "gen";
    }
    auto it = initial_alphas.find(rank);
    const double alpha = (it != initial_alphas.end()) ? it->second : 1.0;
    if (alpha >= 0.99) {
        return use_expert_pager ? "hybrid" : "gen";
    }
    if (alpha <= 0.01) {
        return "und";
    }
    return "hybrid";
}

double PolicyConfig::dit_turn_slo_for(
    int height,
    int steps,
    std::optional<double> k_opt) const {
    if (dit_turn_slo_override_ms.has_value() && !k_opt.has_value()) {
        return *dit_turn_slo_override_ms;
    }
    const double eff_k = k_opt.has_value() ? *k_opt : dit_slo_scale_k;
    const std::string fam = (model_family == "internvl_u") ? "internvl_u" : "bagel";
    return default_operator_profile().dit_turn_slo_ms(height, eff_k, fam, steps);
}

nlohmann::json PolicyConfig::to_json() const {
    nlohmann::json alphas = nlohmann::json::object();
    for (const auto& [k, v] : initial_alphas) {
        alphas[std::to_string(k)] = v;
    }
    return {
        {"policy_name", policy_name},
        {"num_gpus", num_gpus},
        {"model_family", model_family},
        {"cptg_arch_override", cptg_arch_override.has_value() ? nlohmann::json(*cptg_arch_override) : nlohmann::json(nullptr)},
        {"vendor_dir", vendor_dir.has_value() ? nlohmann::json(*vendor_dir) : nlohmann::json(nullptr)},
        {"routing_mode", routing_mode},
        {"use_tripool_vmm", use_tripool_vmm},
        {"use_expert_pager", use_expert_pager},
        {"use_scoped_vae", use_scoped_vae},
        {"use_shared_numa_mirror", use_shared_numa_mirror},
        {"use_h2d_pacer", use_h2d_pacer},
        {"use_module3_cow_loopback", use_module3_cow_loopback},
        {"allow_cascade_cfg_split", allow_cascade_cfg_split},
        {"use_naive_offload", use_naive_offload},
        {"kv_vmm_only", kv_vmm_only},
        {"preemption_mode", preemption_mode},
        {"static_und_ranks", static_und_ranks},
        {"static_gen_ranks", static_gen_ranks},
        {"initial_alphas", alphas},
        {"slo_scale_k", slo_scale_k},
        {"dit_slo_scale_k", dit_slo_scale_k},
        {"ar_tpot_slo_ms", ar_tpot_slo_ms},
        {"ar_ttft_slo_ms", ar_ttft_slo_ms},
        {"dit_512_slo_ms", dit_512_slo_ms},
        {"dit_1024_slo_ms", dit_1024_slo_ms},
        {"dit_turn_slo_override_ms", dit_turn_slo_override_ms.has_value() ? nlohmann::json(*dit_turn_slo_override_ms) : nlohmann::json(nullptr)},
        {"max_oom_retries", max_oom_retries},
        {"inter_turn_think_s_range", nlohmann::json::array({inter_turn_think_s_range.first, inter_turn_think_s_range.second})},
        {"think_time_dist", think_time_dist},
        {"think_time_median_s", think_time_median_s},
        {"think_time_sigma", think_time_sigma},
        {"ar_tpot_slo_mode", ar_tpot_slo_mode},
        {"steady_window_mode", steady_window_mode},
        {"warmup_window_s", warmup_window_s},
        {"steady_window_s", steady_window_s},
        {"steady_state_trim_frac", steady_state_trim_frac},
        {"gpu_memory_fraction", gpu_memory_fraction.has_value() ? nlohmann::json(*gpu_memory_fraction) : nlohmann::json(nullptr)},
        {"gpu_memory_quota_mib", gpu_memory_quota_mib.has_value() ? nlohmann::json(*gpu_memory_quota_mib) : nlohmann::json(nullptr)},
        {"super_pool_mib", super_pool_mib},
        {"fine_kv_pool_mib", fine_kv_pool_mib},
        {"super_handle_mib", super_handle_mib},
        {"transient_va_gib", transient_va_gib},
        {"kv_va_gib", kv_va_gib},
        {"allow_vmm_overflow", allow_vmm_overflow},
    };
}

PolicyConfig PolicyConfig::from_json(const nlohmann::json& j) {
    PolicyConfig cfg;
    apply_policy_overrides(cfg, j);
    return cfg;
}

PolicyConfig PolicyConfig::from_preset(
    const std::string& policy_name,
    int num_gpus,
    const nlohmann::json& overrides) {
    if (!is_valid_policy_name(policy_name)) {
        throw std::invalid_argument("unsupported policy_name=" + policy_name);
    }
    if (num_gpus < 1) {
        throw std::invalid_argument("num_gpus must be >= 1");
    }
    const std::string fam =
        (overrides.is_object() && overrides.contains("model_family") && overrides["model_family"].is_string())
            ? overrides["model_family"].get<std::string>()
            : "bagel";
    const bool has_k_override =
        overrides.is_object() && overrides.contains("slo_scale_k") && overrides["slo_scale_k"].is_number();
    const double k_val = has_k_override ? overrides["slo_scale_k"].get<double>() : 2.0;
    const double k_dit_val =
        (overrides.is_object() && overrides.contains("dit_slo_scale_k") && overrides["dit_slo_scale_k"].is_number())
            ? overrides["dit_slo_scale_k"].get<double>()
            : (has_k_override ? k_val : 3.0);
    const std::string eff_fam = (fam == "internvl_u") ? "internvl_u" : "bagel";
    const auto ar_slo_map = default_operator_profile().slo_thresholds_ms(k_val, eff_fam);
    const auto dit_slo_map = default_operator_profile().slo_thresholds_ms(k_dit_val, eff_fam);
    const std::pair<double, double> think_default =
        (fam == "synthetic") ? std::make_pair(0.002, 0.005) : std::make_pair(2.0, 25.0);

    PolicyConfig cfg;
    cfg.policy_name = policy_name;
    cfg.num_gpus = num_gpus;
    cfg.model_family = fam;
    cfg.allow_cascade_cfg_split = false;
    cfg.slo_scale_k = k_val;
    cfg.dit_slo_scale_k = k_dit_val;
    cfg.ar_tpot_slo_ms = ar_slo_map.at("ar_tpot_slo_ms");
    cfg.ar_ttft_slo_ms = ar_slo_map.at("ar_ttft_slo_ms");
    cfg.dit_512_slo_ms = dit_slo_map.at("dit_512_slo_ms");
    cfg.dit_1024_slo_ms = dit_slo_map.at("dit_1024_slo_ms");
    cfg.inter_turn_think_s_range = think_default;
    cfg.think_time_dist = (fam == "synthetic") ? "uniform" : "lognormal";
    cfg.think_time_median_s = (fam == "synthetic") ? 0.003 : 6.0;
    cfg.think_time_sigma = 0.75;
    cfg.ar_tpot_slo_mode = "turn_p95";
    cfg.steady_window_mode = "time";
    cfg.warmup_window_s = 60.0;
    cfg.steady_window_s = 180.0;

    if (policy_name == "morphserve") {
        for (int g = 0; g < num_gpus; ++g) {
            cfg.initial_alphas[g] = 1.0;
        }
        const bool has_mem_frac =
            overrides.is_object() && overrides.contains("gpu_memory_fraction") &&
            overrides["gpu_memory_fraction"].is_number();
        const bool has_quota_mib =
            overrides.is_object() && overrides.contains("gpu_memory_quota_mib") &&
            overrides["gpu_memory_quota_mib"].is_number();
        const bool is_ivlu_capped = (fam == "internvl_u") &&
            ((has_mem_frac && overrides["gpu_memory_fraction"].get<double>() <= 0.5) ||
             (has_quota_mib && overrides["gpu_memory_quota_mib"].get<double>() > 0.0 &&
              overrides["gpu_memory_quota_mib"].get<double>() <= 20480.0));
        const int ivlu_super_mib = is_ivlu_capped ? 4864 : 12288;

        cfg.routing_mode = (num_gpus >= 2) ? "dual_path" : "sticky_least_loaded";
        cfg.use_tripool_vmm = true;
        cfg.use_expert_pager = true;
        cfg.use_scoped_vae = true;
        cfg.use_shared_numa_mirror = (num_gpus >= 2);
        cfg.use_h2d_pacer = (num_gpus >= 2);
        cfg.use_module3_cow_loopback = (fam != "internvl_u");
        cfg.use_naive_offload = false;
        cfg.kv_vmm_only = false;
        cfg.preemption_mode = (fam == "internvl_u") ? "non_preemptive" : "layer_chunk";
        cfg.super_pool_mib = (fam == "internvl_u") ? ivlu_super_mib : 14336;
        cfg.fine_kv_pool_mib = (fam == "internvl_u") ? 512 : 256;
        cfg.transient_va_gib = 16;
        cfg.kv_va_gib = (fam == "internvl_u") ? 8 : 16;
        cfg.allow_vmm_overflow = !is_ivlu_capped;
    } else if (policy_name == "monolithic_dp") {
        cfg.routing_mode = "sticky_least_loaded";
        cfg.use_tripool_vmm = false;
        cfg.use_expert_pager = false;
        cfg.use_scoped_vae = false;
        cfg.use_shared_numa_mirror = false;
        cfg.use_h2d_pacer = false;
        cfg.use_module3_cow_loopback = false;
        cfg.use_naive_offload = false;
        cfg.kv_vmm_only = false;
        cfg.preemption_mode = "non_preemptive";
        for (int g = 0; g < num_gpus; ++g) {
            cfg.initial_alphas[g] = 1.0;
        }
    } else if (policy_name == "twostage_shm") {
        if (num_gpus < 2 || (num_gpus % 2) != 0) {
            throw std::invalid_argument("twostage_shm requires an even number of GPUs >= 2");
        }
        for (int g = 0; g < num_gpus; g += 2) cfg.static_und_ranks.push_back(g);
        for (int g = 1; g < num_gpus; g += 2) cfg.static_gen_ranks.push_back(g);
        cfg.routing_mode = "twostage_shm";
        cfg.use_tripool_vmm = false;
        cfg.use_expert_pager = false;
        cfg.use_scoped_vae = false;
        cfg.use_shared_numa_mirror = false;
        cfg.use_h2d_pacer = false;
        cfg.use_module3_cow_loopback = false;
        cfg.use_naive_offload = false;
        cfg.kv_vmm_only = false;
        cfg.preemption_mode = "non_preemptive";
        for (int g = 0; g < num_gpus; ++g) {
            cfg.initial_alphas[g] = ((g % 2) == 0) ? 0.0 : 1.0;
        }
    } else if (policy_name == "static_nccl_fission" ||
               policy_name == "static_2und_6gen" ||
               policy_name == "static_4und_4gen" ||
               policy_name == "static_6und_2gen") {
        configure_static_fission_ranks(cfg, policy_name, num_gpus);
    } else if (policy_name == "prism_tetriserve_coloc") {
        cfg.routing_mode = "sticky_least_loaded";
        cfg.use_tripool_vmm = true;
        cfg.use_expert_pager = false;
        cfg.use_scoped_vae = false;
        cfg.use_shared_numa_mirror = false;
        cfg.use_h2d_pacer = false;
        cfg.use_module3_cow_loopback = false;
        cfg.use_naive_offload = false;
        cfg.kv_vmm_only = true;
        cfg.preemption_mode = (fam == "internvl_u") ? "non_preemptive" : "step_preemptive";
        for (int g = 0; g < num_gpus; ++g) {
            cfg.initial_alphas[g] = 1.0;
        }
        cfg.super_pool_mib = 128;
        cfg.fine_kv_pool_mib = 2048;
        cfg.kv_va_gib = 16;
        cfg.allow_vmm_overflow = true;
    } else {
        cfg.routing_mode = "sticky_least_loaded";
        cfg.use_tripool_vmm = false;
        cfg.use_expert_pager = false;
        cfg.use_scoped_vae = false;
        cfg.use_shared_numa_mirror = false;
        cfg.use_h2d_pacer = false;
        cfg.use_module3_cow_loopback = false;
        cfg.use_naive_offload = true;
        cfg.kv_vmm_only = false;
        cfg.preemption_mode = "non_preemptive";
        for (int g = 0; g < num_gpus; ++g) {
            cfg.initial_alphas[g] = 0.0;
        }
    }

    apply_policy_overrides(cfg, overrides);
    return cfg;
}

std::vector<std::pair<std::string, PolicyConfig>> build_ablation_configs(
    int num_gpus,
    const nlohmann::json& overrides) {
    // Strict single-variable Leave-One-Out ablations relative to `full_morphserve` (M1+M2+M3+M4).
    PolicyConfig full = PolicyConfig::from_preset("morphserve", num_gpus, overrides);

    PolicyConfig no_m4 = full;
    no_m4.policy_name = "no_m4_layer_chunk";
    no_m4.preemption_mode = "non_preemptive";

    PolicyConfig no_m3 = full;
    no_m3.policy_name = "no_m3_cow_loopback";
    no_m3.use_module3_cow_loopback = false;

    PolicyConfig no_m2 = full;
    no_m2.policy_name = "no_m2_dual_path";
    no_m2.routing_mode = "sticky_least_loaded";
    no_m2.use_shared_numa_mirror = false;
    no_m2.use_h2d_pacer = false;

    PolicyConfig no_m1 = full;
    no_m1.policy_name = "no_m1_tripool_vmm";
    no_m1.use_tripool_vmm = false;
    no_m1.use_expert_pager = false;
    no_m1.use_scoped_vae = false;
    no_m1.use_shared_numa_mirror = false;
    no_m1.use_h2d_pacer = false;

    return {
        {"full_morphserve", full},
        {"no_m4_layer_chunk", no_m4},
        {"no_m3_cow_loopback", no_m3},
        {"no_m2_dual_path", no_m2},
        {"no_m1_tripool_vmm", no_m1},
    };
}

std::vector<std::pair<std::string, PolicyConfig>> build_static_split_configs(
    int num_gpus,
    const nlohmann::json& overrides) {
    if (num_gpus < 4 || (num_gpus % 4) != 0) {
        throw std::invalid_argument(
            "build_static_split_configs requires num_gpus >= 4 and divisible by 4, got " +
            std::to_string(num_gpus));
    }
    return {
        {"morphserve_elastic", PolicyConfig::from_preset("morphserve", num_gpus, overrides)},
        {"static_2und_6gen", PolicyConfig::from_preset("static_2und_6gen", num_gpus, overrides)},
        {"static_4und_4gen", PolicyConfig::from_preset("static_4und_4gen", num_gpus, overrides)},
        {"static_6und_2gen", PolicyConfig::from_preset("static_6und_2gen", num_gpus, overrides)},
    };
}

}  // namespace morphserve::control
