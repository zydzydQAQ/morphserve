#pragma once

#include "morphserve/control/topology.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace morphserve::control {

inline constexpr std::array<const char*, 7> kValidPolicies = {
    "morphserve",
    "monolithic_dp",
    "monolithic_kv_swap",
    "twostage_shm",
    "static_nccl_fission",
    "prism_tetriserve_coloc",
    "naive_host_offload",
};

inline constexpr std::array<const char*, 3> kStaticSplitPolicies = {
    "static_2und_6gen",
    "static_4und_4gen",
    "static_6und_2gen",
};

bool is_valid_policy_name(const std::string& policy_name) noexcept;

struct PolicyConfig {
    std::string policy_name = "morphserve";
    int num_gpus = 8;
    std::string model_family = "bagel";  // "bagel" | "internvl_u" | "synthetic"
    std::optional<std::string> cptg_arch_override;
    std::optional<std::string> vendor_dir;
    std::string routing_mode = "dual_path";
    bool use_tripool_vmm = true;
    bool use_expert_pager = true;
    bool use_scoped_vae = true;
    bool use_shared_numa_mirror = true;
    bool use_h2d_pacer = true;
    bool use_module3_cow_loopback = true;
    bool allow_cascade_cfg_split = false;
    bool use_naive_offload = false;
    bool kv_vmm_only = false;
    bool use_idle_kv_swap = false;
    std::string preemption_mode = "layer_chunk";
    std::string cost_model_mode = "atlas_3d";     // "atlas_3d" | "pipeline_analytical" | "legacy_analytical"
    std::string gpu_queue_mode = "memory_slack";  // "memory_slack" | "fcfs"
    double gpu_queue_max_aging_ms = 45000.0;
    int gpu_queue_max_bypasses = 5;
    std::string kv_eviction_mode = "swap";        // "swap" | "recompute"
    std::vector<int> static_und_ranks;
    std::vector<int> static_gen_ranks;
    std::map<int, double> initial_alphas;

    double slo_scale_k = 2.0;
    double dit_slo_scale_k = 3.0;
    double ar_tpot_slo_ms = 101.40;
    double ar_ttft_slo_ms = 1500.0;
    double dit_512_slo_ms = 14916.30;
    double dit_1024_slo_ms = 57219.60;
    std::optional<double> dit_turn_slo_override_ms;
    int max_oom_retries = 1;
    std::pair<double, double> inter_turn_think_s_range = {2.0, 25.0};
    std::string think_time_dist = "lognormal";
    double think_time_median_s = 6.0;
    double think_time_sigma = 0.75;
    std::string ar_tpot_slo_mode = "turn_p95";
    std::string steady_window_mode = "time";
    double warmup_window_s = 60.0;
    double steady_window_s = 150.0;
    double cooldown_s = 30.0;
    double steady_state_trim_frac = 0.15;
    std::optional<double> gpu_memory_fraction;
    std::optional<double> gpu_memory_quota_mib;

    int super_pool_mib = 14336;
    int fine_kv_pool_mib = 256;
    int super_handle_mib = 64;
    int transient_va_gib = 16;
    int kv_va_gib = 16;
    bool allow_vmm_overflow = true;

    std::string cptg_architecture() const;
    std::string role_of(int rank) const;
    double dit_turn_slo_for(int height, int steps = 20, std::optional<double> k_opt = std::nullopt) const;
    double ar_ttft_slo_for(
        bool has_image,
        int num_images = 0,
        int image_res = 512,
        std::optional<double> k_opt = std::nullopt) const;

    nlohmann::json to_json() const;
    static PolicyConfig from_json(const nlohmann::json& j);
    static PolicyConfig from_preset(
        const std::string& policy_name,
        int num_gpus = 8,
        const nlohmann::json& overrides = nlohmann::json::object());
};

// Static split-ratio sensitivity (`2:6`, `4:4`, `6:2`) & Leave-One-Out M1-M4 ablation suite.
std::vector<std::pair<std::string, PolicyConfig>> build_ablation_configs(
    int num_gpus = 8,
    const nlohmann::json& overrides = nlohmann::json::object());
std::vector<std::pair<std::string, PolicyConfig>> build_static_split_configs(
    int num_gpus = 8,
    const nlohmann::json& overrides = nlohmann::json::object());

}  // namespace morphserve::control
