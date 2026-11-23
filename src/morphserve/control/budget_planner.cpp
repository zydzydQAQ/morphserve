#include "morphserve/control/budget_planner.hpp"

#include <algorithm>
#include <cmath>

namespace morphserve::control {

namespace {

inline double round_2(double x) noexcept {
    return std::round(x * 100.0) / 100.0;
}

inline double round_6(double x) noexcept {
    return std::round(x * 1000000.0) / 1000000.0;
}

}  // namespace

nlohmann::json WorkerBudgetPlanner::compute_internvlu_memory_plan(
    double total_dev_mib,
    double free_dev_mib,
    double base_pytorch_alloc_mib,
    int64_t mirror_bytes,
    int policy_super_pool_mib,
    int policy_fine_kv_pool_mib,
    int super_handle_mib,
    bool allow_vmm_overflow,
    std::optional<double> gpu_memory_fraction,
    std::optional<double> gpu_memory_quota_mib) {
    const int handle_mib = std::max(2, super_handle_mib);
    const int mirror_mib = static_cast<int>(
        std::ceil(static_cast<double>(std::max<int64_t>(0, mirror_bytes)) / static_cast<double>(kMiB)));
    const int aligned_mirror_mib =
        (mirror_mib > 0) ? (((mirror_mib + handle_mib - 1) / handle_mib) * handle_mib) : 0;

    std::optional<double> raw_quota_mib;
    if (gpu_memory_quota_mib.has_value() && *gpu_memory_quota_mib > 0.0) {
        raw_quota_mib = *gpu_memory_quota_mib;
    } else if (gpu_memory_fraction.has_value() && *gpu_memory_fraction < 0.999) {
        const double frac_mib = total_dev_mib * (*gpu_memory_fraction);
        if (!allow_vmm_overflow && aligned_mirror_mib > 0 && policy_super_pool_mib >= aligned_mirror_mib &&
            frac_mib < static_cast<double>(aligned_mirror_mib) + base_pytorch_alloc_mib + 1024.0) {
            int super_mib = std::max(aligned_mirror_mib, (policy_super_pool_mib / handle_mib) * handle_mib);
            const int fine_mib = std::max(0, (policy_fine_kv_pool_mib / 2) * 2);
            int vmm_mib = super_mib + fine_mib;
            double inferred_quota_mib = static_cast<double>(vmm_mib) + frac_mib;
            if (inferred_quota_mib >= 11500.0 && inferred_quota_mib <= 14000.0) {
                inferred_quota_mib = 12288.0;
            }
            const double usable_quota_mib =
                std::max(static_cast<double>(handle_mib * 2), inferred_quota_mib - 128.0);
            if (static_cast<double>(vmm_mib) + base_pytorch_alloc_mib + 256.0 > usable_quota_mib) {
                super_mib = std::max(
                    handle_mib,
                    static_cast<int>(std::floor(
                        (usable_quota_mib - base_pytorch_alloc_mib - 256.0 - static_cast<double>(fine_mib)) /
                        static_cast<double>(handle_mib))) *
                        handle_mib);
                vmm_mib = super_mib + fine_mib;
            }
            double pytorch_cap_mib = std::min(
                frac_mib, std::max(base_pytorch_alloc_mib, usable_quota_mib - static_cast<double>(vmm_mib)));
            if (pytorch_cap_mib + static_cast<double>(vmm_mib) > inferred_quota_mib) {
                super_mib = std::max(
                    handle_mib,
                    static_cast<int>(std::floor(
                        (inferred_quota_mib - pytorch_cap_mib - static_cast<double>(fine_mib)) /
                        static_cast<double>(handle_mib))) *
                        handle_mib);
                vmm_mib = super_mib + fine_mib;
            }
            const double pt_frac = std::min(
                *gpu_memory_fraction, std::max(0.05, pytorch_cap_mib / std::max(1.0, total_dev_mib)));
            return {
                {"quota_constrained", true},
                {"quota_mib", round_2(inferred_quota_mib)},
                {"usable_quota_mib", round_2(usable_quota_mib)},
                {"super_pool_mib", super_mib},
                {"fine_kv_pool_mib", fine_mib},
                {"vmm_physical_pool_mib", vmm_mib},
                {"pytorch_cap_mib", round_2(pytorch_cap_mib)},
                {"pytorch_memory_fraction", round_6(pt_frac)},
                {"allow_overflow", false},
            };
        }
        double q = frac_mib;
        if (q >= 11500.0 && q <= 12500.0) {
            q = std::min(q, 12288.0);
        }
        raw_quota_mib = q;
    }

    if (raw_quota_mib.has_value()) {
        const double usable_quota_mib = std::max(static_cast<double>(handle_mib * 2), *raw_quota_mib - 128.0);
        const int fine_mib = std::min(std::max(0, (policy_fine_kv_pool_mib / 2) * 2), 64);
        const int min_super_mib = std::max(handle_mib, aligned_mirror_mib);
        const double min_pt_act_mib = 1536.0;

        const int hard_max_super = std::max(
            handle_mib,
            static_cast<int>(std::floor(
                (usable_quota_mib - (base_pytorch_alloc_mib + min_pt_act_mib) - static_cast<double>(fine_mib)) /
                static_cast<double>(handle_mib))) *
                handle_mib);
        const int desired_super = std::max(min_super_mib, policy_super_pool_mib);
        int super_mib = 0;
        if (static_cast<double>(min_super_mib + fine_mib) + base_pytorch_alloc_mib + 1024.0 <= usable_quota_mib) {
            super_mib = std::max(min_super_mib, std::min(desired_super, hard_max_super));
        } else {
            super_mib = std::min(desired_super, hard_max_super);
        }
        super_mib = std::max(handle_mib, (super_mib / handle_mib) * handle_mib);

        double pytorch_cap_mib = std::max(
            base_pytorch_alloc_mib + 512.0, usable_quota_mib - static_cast<double>(super_mib + fine_mib));
        if (pytorch_cap_mib + static_cast<double>(super_mib + fine_mib) > usable_quota_mib) {
            super_mib = std::max(
                handle_mib,
                static_cast<int>(std::floor(
                    (usable_quota_mib - pytorch_cap_mib - static_cast<double>(fine_mib)) /
                    static_cast<double>(handle_mib))) *
                    handle_mib);
            pytorch_cap_mib = std::max(
                base_pytorch_alloc_mib, usable_quota_mib - static_cast<double>(super_mib + fine_mib));
        }
        const double max_frac = gpu_memory_fraction.value_or(1.0);
        const double pt_frac =
            std::min(max_frac, std::max(0.05, pytorch_cap_mib / std::max(1.0, total_dev_mib)));
        return {
            {"quota_constrained", true},
            {"quota_mib", round_2(*raw_quota_mib)},
            {"usable_quota_mib", round_2(usable_quota_mib)},
            {"super_pool_mib", super_mib},
            {"fine_kv_pool_mib", fine_mib},
            {"vmm_physical_pool_mib", super_mib + fine_mib},
            {"pytorch_cap_mib", round_2(pytorch_cap_mib)},
            {"pytorch_memory_fraction", round_6(pt_frac)},
            {"allow_overflow", false},
        };
    }

    const int fine_mib = std::max(0, (policy_fine_kv_pool_mib / 2) * 2);
    const int desired_super = std::max(
        policy_super_pool_mib,
        (aligned_mirror_mib > 0) ? (aligned_mirror_mib + 5632) : policy_super_pool_mib);
    int super_mib = desired_super;
    if (free_dev_mib > 0.0) {
        const int safe_ceiling = std::max(
            std::max(handle_mib, aligned_mirror_mib),
            static_cast<int>(free_dev_mib - 8192.0 - static_cast<double>(fine_mib)));
        super_mib = std::min(desired_super, safe_ceiling);
    }
    super_mib = std::max(handle_mib, ((super_mib + handle_mib - 1) / handle_mib) * handle_mib);
    return {
        {"quota_constrained", false},
        {"quota_mib", nullptr},
        {"usable_quota_mib", nullptr},
        {"super_pool_mib", super_mib},
        {"fine_kv_pool_mib", fine_mib},
        {"vmm_physical_pool_mib", super_mib + fine_mib},
        {"pytorch_cap_mib", nullptr},
        {"pytorch_memory_fraction", nullptr},
        {"allow_overflow", allow_vmm_overflow},
    };
}

std::pair<nlohmann::json, nlohmann::json> WorkerBudgetPlanner::compute_internvlu_quota_overrides(
    const std::string& policy_name,
    int num_gpus,
    const std::string& vendor_dir,
    double constrained_gib,
    double total_gpu_gib) {
    const double quota_mib = round_2(constrained_gib * 1024.0);
    const double total_mib = std::max(quota_mib, total_gpu_gib * 1024.0);
    const double total_quota_frac = round_6(std::min(1.0, std::max(0.05, quota_mib / total_mib)));

    nlohmann::json overrides = {
        {"model_family", "internvl_u"},
        {"vendor_dir", vendor_dir},
        {"inter_turn_think_s_range", nlohmann::json::array({0.5, 1.5})},
        {"gpu_memory_fraction", total_quota_frac},
    };

    if (policy_name == "morphserve") {
        const int super_handle_mib = 16;
        const int fine_kv_pool_mib = 32;
        const int super_pool_mib = (quota_mib >= 11500.0 && quota_mib <= 13500.0)
                                       ? 6656
                                       : std::max(
                                             4352,
                                             (static_cast<int>(std::floor(quota_mib * 0.54)) / super_handle_mib) *
                                                 super_handle_mib);
        const double vmm_pool_mib = static_cast<double>(super_pool_mib + fine_kv_pool_mib);
        const double runtime_pt_mib = std::max(0.0, quota_mib - vmm_pool_mib);
        nlohmann::json alphas = nlohmann::json::object();
        for (int g = 0; g < num_gpus; ++g) {
            alphas[std::to_string(g)] = 1.0;
        }
        overrides["super_handle_mib"] = super_handle_mib;
        overrides["super_pool_mib"] = super_pool_mib;
        overrides["fine_kv_pool_mib"] = fine_kv_pool_mib;
        overrides["allow_vmm_overflow"] = false;
        overrides["initial_alphas"] = alphas;
        nlohmann::json budget = {
            {"quota_limit_mib", quota_mib},
            {"vmm_physical_pool_mib", vmm_pool_mib},
            {"runtime_pytorch_budget_mib", round_2(runtime_pt_mib)},
            {"runtime_combined_budget_mib", round_2(runtime_pt_mib + vmm_pool_mib)},
            {"total_quota_fraction", total_quota_frac},
        };
        return {overrides, budget};
    }

    nlohmann::json budget = {
        {"quota_limit_mib", quota_mib},
        {"vmm_physical_pool_mib", 0.0},
        {"runtime_pytorch_budget_mib", quota_mib},
        {"runtime_combined_budget_mib", quota_mib},
        {"total_quota_fraction", total_quota_frac},
    };
    return {overrides, budget};
}

nlohmann::json WorkerBudgetPlanner::verify_internvlu_memory_quota(
    const nlohmann::json& run_result,
    const PolicyConfig& policy,
    double constrained_gib) {
    const double quota_limit_mib = round_2(constrained_gib * 1024.0);
    double vmm_physical_pool_mib = 0.0;
    if (run_result.contains("vmm_physical_pool_mib") && run_result["vmm_physical_pool_mib"].is_number()) {
        vmm_physical_pool_mib = round_2(run_result["vmm_physical_pool_mib"].get<double>());
    } else if (policy.use_tripool_vmm) {
        const double base_vmm = static_cast<double>(policy.super_pool_mib + policy.fine_kv_pool_mib);
        const double ovf = static_cast<double>(run_result.value("vmm_overflow_creates", 0) * policy.super_handle_mib);
        vmm_physical_pool_mib = round_2(base_vmm + ovf);
    }

    double pytorch_reserved_mib = 0.0;
    double combined_physical_mib = 0.0;
    if ((run_result.contains("pytorch_peak_reserved_mib") && run_result["pytorch_peak_reserved_mib"].is_number()) ||
        (run_result.contains("pytorch_reserved_mib") && run_result["pytorch_reserved_mib"].is_number())) {
        const double pt_raw = run_result.contains("pytorch_peak_reserved_mib")
                                  ? run_result["pytorch_peak_reserved_mib"].get<double>()
                                  : run_result["pytorch_reserved_mib"].get<double>();
        pytorch_reserved_mib = round_2(pt_raw);
        combined_physical_mib = round_2(pytorch_reserved_mib + vmm_physical_pool_mib);
    } else if (policy.use_tripool_vmm) {
        combined_physical_mib = round_2(run_result.value("peak_gpu_reserved_mib", 0.0));
        pytorch_reserved_mib = round_2(std::max(0.0, combined_physical_mib - vmm_physical_pool_mib));
    } else {
        pytorch_reserved_mib = round_2(run_result.value("peak_gpu_reserved_mib", 0.0));
        combined_physical_mib = pytorch_reserved_mib;
    }

    const bool overflow_ok = (!policy.use_tripool_vmm) || (run_result.value("vmm_overflow_creates", 0) == 0);
    return {
        {"quota_limit_mib", quota_limit_mib},
        {"pytorch_peak_reserved_mib", pytorch_reserved_mib},
        {"vmm_physical_pool_mib", vmm_physical_pool_mib},
        {"combined_physical_mib", combined_physical_mib},
        {"within_quota", (combined_physical_mib <= quota_limit_mib) && overflow_ok},
    };
}

}  // namespace morphserve::control
