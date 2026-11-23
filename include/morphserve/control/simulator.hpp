#pragma once

#include "morphserve/control/policy.hpp"
#include "morphserve/control/topology.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace morphserve::control {

struct SimTurnSpec {
    std::string modality_plan = "ar_only";
    int height = 512;
};

struct SimSessionSpec {
    double arrival_ms = 0.0;
    std::string quadrant = "visual_cot";
    std::vector<SimTurnSpec> turns;
};

// Byte-accurate per-GPU physical HBM ledger from Phase-0/1 driver measurements.
class SimGpuMemoryLedger {
public:
    SimGpuMemoryLedger(
        int gpu_id,
        double total_hbm_mib,
        PolicyConfig policy,
        std::string role,
        int resident_expert_layers = 28,
        int num_expert_layers = 28,
        int layer_stride_mib = 448,
        ClusterPhysicsProfile physics = ClusterPhysicsProfile{},
        double weight_scale = 1.0,
        double vae_spike_scale = 1.0);

    int gpu_id() const noexcept { return gpu_id_; }
    double total_hbm_mib() const noexcept { return total_hbm_mib_; }
    const PolicyConfig& policy() const noexcept { return policy_; }
    const std::string& role() const noexcept { return role_; }
    int resident_expert_layers() const noexcept { return resident_expert_layers_; }
    int num_expert_layers() const noexcept { return num_expert_layers_; }
    int layer_stride_mib() const noexcept { return layer_stride_mib_; }

    int super_to_fine_carves() const noexcept { return super_to_fine_carves_; }
    int expert_evictions() const noexcept { return expert_evictions_; }
    int kv_host_swap_outs() const noexcept { return kv_host_swap_outs_; }
    int kv_host_swap_ins() const noexcept { return kv_host_swap_ins_; }
    double peak_used_mib() const noexcept { return peak_used_mib_; }
    void set_peak_used_mib(double v) noexcept { peak_used_mib_ = v; }

    double offloaded_base_mib() const noexcept;
    double monolithic_weights_mib() const noexcept;
    double pageable_expert_mib() const noexcept;
    double base_weights_mib() const noexcept;
    double expert_alpha() const noexcept;
    double active_kv_mib() const noexcept;
    int64_t free_kv_bytes() const noexcept;

    double evict_idle_kv_to_host(double needed_free_mib, std::optional<int> exclude_sid = std::nullopt);
    double ensure_session_resident(int sid);
    bool upsert_session_kv(int sid, double logical_kv_mib);
    std::pair<bool, double> execute_dit_and_vae(int height, std::optional<int> active_sid = std::nullopt);
    void release_session(int sid);

    nlohmann::json stats_json() const;

private:
    int gpu_id_ = 0;
    double total_hbm_mib_ = 40960.0;
    PolicyConfig policy_;
    std::string role_ = "hybrid";
    int resident_expert_layers_ = 28;
    int num_expert_layers_ = 28;
    int layer_stride_mib_ = 448;
    ClusterPhysicsProfile physics_;
    double weight_scale_ = 1.0;
    double vae_spike_scale_ = 1.0;

    std::map<int, double> session_kv_mib_;
    std::map<int, double> swapped_session_kv_mib_;
    int super_to_fine_carves_ = 0;
    int expert_evictions_ = 0;
    int kv_host_swap_outs_ = 0;
    int kv_host_swap_ins_ = 0;
    double peak_used_mib_ = 0.0;
};

// Native C++17 zero-bias discrete-event cluster simulator (`8 -> 16 -> 64` GPUs & H100-80GB generality).
class ClusterSimulator {
public:
    static std::vector<SimSessionSpec> build_session_stream(
        int num_sessions,
        double offered_qps,
        int num_gpus,
        uint64_t seed = 2026,
        int extra_edit_turns = 0,
        const std::string& skew_tag = "",
        const std::string& bench_root = "",
        const std::string& sources_dir = "");

    static nlohmann::json simulate_cluster_point(
        const std::string& policy_name,
        int num_gpus,
        const HardwareInterconnectSpec& hw,
        double offered_qps,
        int num_sessions = 96,
        uint64_t seed = 2026,
        const ClusterPhysicsProfile& physics = ClusterPhysicsProfile{},
        const WorkloadScaleSpec& workload_scale = WorkloadScaleSpec::bagel_7b_standard(),
        const std::string& skew_tag = "",
        const std::string& bench_root = "",
        const std::string& sources_dir = "",
        const std::vector<SimSessionSpec>* custom_sessions = nullptr);

    static nlohmann::json evaluate_against_real_runs(
        const nlohmann::json& real_runs_array,
        const ClusterPhysicsProfile& physics,
        const std::string& bench_root = "",
        const std::string& sources_dir = "");

    static nlohmann::json run_h100_80gb_generality_study(
        const ClusterPhysicsProfile& physics,
        const std::string& bench_root = "",
        const std::string& sources_dir = "");

    static nlohmann::json run_scale_and_interconnect_sweep(
        const std::string& output_dir,
        const std::string& real_report_path = "",
        const std::string& results_dir = "",
        const std::string& bench_root = "",
        const std::string& sources_dir = "");
};

}  // namespace morphserve::control
