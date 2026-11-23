#pragma once

#include "morphserve/control/policy.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace morphserve::control {

// Unified physical HBM & TriPool-VMM budget planner for Regime A (40/80GB) & Regime B (12GiB).
class WorkerBudgetPlanner {
public:
    static nlohmann::json compute_internvlu_memory_plan(
        double total_dev_mib,
        double free_dev_mib,
        double base_pytorch_alloc_mib,
        int64_t mirror_bytes,
        int policy_super_pool_mib,
        int policy_fine_kv_pool_mib,
        int super_handle_mib = 64,
        bool allow_vmm_overflow = true,
        std::optional<double> gpu_memory_fraction = std::nullopt,
        std::optional<double> gpu_memory_quota_mib = std::nullopt);

    static std::pair<nlohmann::json, nlohmann::json> compute_internvlu_quota_overrides(
        const std::string& policy_name,
        int num_gpus,
        const std::string& vendor_dir,
        double constrained_gib = 12.0,
        double total_gpu_gib = 40.0);

    static nlohmann::json verify_internvlu_memory_quota(
        const nlohmann::json& run_result,
        const PolicyConfig& policy,
        double constrained_gib = 12.0);
};

}  // namespace morphserve::control
