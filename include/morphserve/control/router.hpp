#pragma once

#include "morphserve/control/performance_atlas.hpp"
#include "morphserve/control/topology.hpp"
#include "morphserve/runtime/h2d_pacer.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace morphserve::control {

struct CPTGRequestSpec {
    std::string req_id;
    int src_gpu = 0;
    std::string architecture = "tight_mot";  // "tight_mot" | "bridge_mllm_dit"
    int64_t kv_bytes = 0;
    int64_t bridge_bytes = 0;
    int64_t gen_expert_bytes = 12446LL * kMiB;
    double dit_step1_ms = 287.0;
    int dit_total_steps = 20;
    int resolution = 512;
    int interleaved_roundtrip_gamma = 1;
    std::string model_family;  // optional ("bagel" | "internvl_u"); inferred from architecture if empty

    int64_t effective_state_bytes() const;
    nlohmann::json to_json() const;
    static CPTGRequestSpec from_json(const nlohmann::json& j);
};

struct WorkerState {
    int gpu_id = 0;
    std::string role = "hybrid";  // "und" | "gen" | "hybrid"
    double expert_alpha = 1.0;
    double queue_wait_ms = 0.0;
    int active_h2d_transfers = 0;
    int64_t kv_free_bytes = 12LL * kGiB;
    int p2p_concurrent_links = 1;
    bool post_evict = false;   // true right after VMM expert eviction (remap contention regime)
    int colocated_ar_b = 0;    // active colocated AR decode tasks (3rd Performance Atlas axis)

    nlohmann::json to_json() const;
    static WorkerState from_json(const nlohmann::json& j);
};

struct RouteDecision {
    std::string req_id;
    std::string path;  // "state_ship" | "in_place_morph"
    int src_gpu = 0;
    int dst_gpu = 0;
    double t_ship_ms = -1.0;
    double t_morph_ms = 0.0;
    double chosen_cost_ms = 0.0;
    double selected_bw_gbps = 0.0;
    double effective_state_mib = 0.0;
    std::string reason;
    std::string hop_tier = "in_place";

    nlohmann::json to_json() const;
};

// Stateful three-tier PCIe/Socket/RDMA Dual-Path Morphing Router and reservation ledger.
// Live coordinators should use the router's internal authoritative worker ledger (`ext_workers = nullptr`);
// `ext_workers` is intended for read-only simulation snapshots and offline batch routing.
class DualPathMorphRouter {
public:
    explicit DualPathMorphRouter(
        ClusterTopologyProfile topology,
        std::shared_ptr<runtime::HalfNodeH2DPacer> pacer = nullptr,
        OperatorLatencyProfile operator_profile = default_operator_profile(),
        std::optional<double> state_handoff_ms = std::nullopt,
        double map_overhead_ms_full = 12.0,
        double overlap_efficiency = 0.90,
        bool allow_cross_half_state_ship = true,
        bool allow_inter_node_state_ship = true,
        int64_t max_inter_node_state_bytes = 128LL * kMiB,
        std::string cost_model_mode = "atlas_3d",
        PerformanceAtlas performance_atlas = PerformanceAtlas::a100_default());

    const ClusterTopologyProfile& topology() const noexcept { return topology_; }
    std::shared_ptr<runtime::HalfNodeH2DPacer> pacer() const noexcept { return pacer_; }
    const OperatorLatencyProfile& operator_profile() const noexcept { return operator_profile_; }
    const std::string& cost_model_mode() const noexcept { return cost_model_mode_; }
    void set_cost_model_mode(std::string mode);
    const PerformanceAtlas& performance_atlas() const noexcept { return atlas_; }
    void set_performance_atlas(PerformanceAtlas atlas);

    std::map<int, WorkerState> init_workers(
        int num_gpus,
        const std::function<std::string(int)>& role_of,
        const std::map<int, double>& initial_alphas,
        const std::map<int, int64_t>& kv_free_bytes);

    std::map<int, WorkerState> worker_states_snapshot() const;
    WorkerState get_worker_state(int gpu_id) const;
    void set_worker_state_fields(
        int gpu_id,
        std::optional<double> expert_alpha,
        std::optional<double> queue_wait_ms,
        std::optional<int> active_h2d_transfers,
        std::optional<int64_t> kv_free_bytes,
        std::optional<int> p2p_concurrent_links);

    // Monotonic `telemetry_seq` and `inflight_kv_reserved_bytes_` reconciliation are enforced
    // consistently whether updating the internal ledger or an external worker map.
    void update_worker_telemetry(
        int gpu_id,
        std::optional<double> expert_alpha,
        std::optional<int64_t> kv_free_bytes,
        std::optional<int64_t> telemetry_seq = std::nullopt,
        std::map<int, WorkerState>* ext_workers = nullptr);

    void reserve_ar_queue(int gpu_id, double est_ar_ms);
    void release_ar_queue(int gpu_id, double est_ar_ms);

    int select_initial_gpu(
        int num_gpus,
        const std::string& routing_mode,
        const std::string& preemption_mode,
        const std::vector<int>& static_und_ranks,
        const std::string& modality_plan,
        const std::map<int, WorkerState>* ext_workers = nullptr,
        const std::map<int, bool>& gpu_busy = {},
        const std::map<int, int>& active_dit_ranks = {},
        const std::map<int, int>& active_preemptions = {},
        int max_preemptions_per_gpu = 6,
        const std::string& architecture = "tight_mot");

    int select_static_gen_gpu(
        const std::string& routing_mode,
        const std::vector<int>& static_und_ranks,
        const std::vector<int>& static_gen_ranks,
        int src_und_gpu,
        const std::map<int, WorkerState>* ext_workers = nullptr,
        const std::map<int, bool>& gpu_busy = {},
        int64_t required_kv_bytes = 64LL * kMiB) const;

    int select_oom_retry_gpu(
        int failed_gpu,
        int num_gpus,
        const std::string& routing_mode,
        const std::vector<int>& static_und_ranks,
        const std::map<int, WorkerState>* ext_workers = nullptr,
        const std::map<int, bool>& gpu_busy = {}) const;

    // Pure read-only transition planner with zero pacer side effects and zero worker mutation.
    std::pair<RouteDecision, std::string> plan_transition_readonly(
        const std::string& routing_mode,
        const std::vector<int>& static_und_ranks,
        const std::vector<int>& static_gen_ranks,
        const CPTGRequestSpec& req,
        double est_dit_ms,
        const std::map<int, WorkerState>* ext_workers = nullptr,
        const std::map<int, bool>& gpu_busy = {},
        double now_ms = 0.0) const;

    std::pair<RouteDecision, std::string> plan_and_reserve_transition(
        const std::string& routing_mode,
        const std::vector<int>& static_und_ranks,
        const std::vector<int>& static_gen_ranks,
        const CPTGRequestSpec& req,
        double est_dit_ms,
        std::map<int, WorkerState>* ext_workers = nullptr,
        const std::map<int, bool>& gpu_busy = {},
        bool use_pacer = true,
        double now_ms = 0.0);

    void reserve_kv_bytes(
        int gpu_id,
        int64_t reserved_kv_bytes,
        std::map<int, WorkerState>* ext_workers = nullptr);

    void release_kv_reservation(
        int gpu_id,
        int64_t reserved_kv_bytes,
        std::optional<int64_t> latest_kv_free_bytes = std::nullopt,
        std::optional<int64_t> telemetry_seq = std::nullopt,
        std::map<int, WorkerState>* ext_workers = nullptr);

    void release_dit_reservation(
        int gpu_id,
        double est_dit_ms,
        std::map<int, WorkerState>* ext_workers = nullptr);

    double analytical_crossover_kv_mib(
        double alpha = 0.50,
        int64_t gen_expert_bytes = 12446LL * kMiB,
        double dit_step1_ms = 287.0,
        int src_gpu = 0,
        int dst_gpu = 1) const;

    std::pair<double, double> evaluate_path_b_morph_ms(
        const CPTGRequestSpec& req,
        const WorkerState& src_worker,
        double now_ms = 0.0) const;

    std::pair<double, double> evaluate_path_a_ship_ms(
        const CPTGRequestSpec& req,
        const WorkerState& src_worker,
        const WorkerState& dst_worker,
        double now_ms = 0.0) const;

    RouteDecision decide(
        const CPTGRequestSpec& req,
        const std::map<int, WorkerState>* ext_workers = nullptr,
        double now_ms = 0.0) const;

    std::vector<RouteDecision> route_batch(
        const std::vector<CPTGRequestSpec>& requests,
        const std::map<int, WorkerState>& workers,
        double now_ms = 0.0);

private:
    ClusterTopologyProfile topology_;
    std::shared_ptr<runtime::HalfNodeH2DPacer> pacer_;
    OperatorLatencyProfile operator_profile_;
    std::optional<double> state_handoff_ms_;
    double map_overhead_ms_full_ = 12.0;
    double overlap_efficiency_ = 0.90;
    bool allow_cross_half_state_ship_ = true;
    bool allow_inter_node_state_ship_ = true;
    int64_t max_inter_node_state_bytes_ = 128LL * kMiB;
    std::string cost_model_mode_ = "atlas_3d";
    PerformanceAtlas atlas_ = PerformanceAtlas::a100_default();

    mutable std::mutex mu_;
    std::map<int, WorkerState> worker_states_;
    std::map<int, int64_t> raw_kv_free_bytes_;
    std::map<int, int64_t> inflight_kv_reserved_bytes_;
    std::map<int, int64_t> last_telemetry_seq_;
    int rr_cursor_ = 0;
};

}  // namespace morphserve::control
