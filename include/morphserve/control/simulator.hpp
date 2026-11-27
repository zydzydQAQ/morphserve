#pragma once

#include "morphserve/control/policy.hpp"
#include "morphserve/control/sim_memory_ledger.hpp"
#include "morphserve/control/sim_trace_loader.hpp"
#include "morphserve/control/topology.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace morphserve::control {

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
        const std::string& sources_dir = "",
        double arrival_duration_s = 0.0);

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
        const std::vector<SimSessionSpec>* custom_sessions = nullptr,
        const nlohmann::json& policy_overrides = nlohmann::json::object());

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
