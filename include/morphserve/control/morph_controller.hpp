#pragma once

#include "morphserve/control/policy.hpp"
#include "morphserve/control/preempt_edf.hpp"
#include "morphserve/control/router.hpp"
#include "morphserve/control/topology.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace morphserve::control {

struct TurnTransitionPlan {
    RouteDecision decision;
    std::string ship_mode;  // "none" | "nccl_dual" | "nccl_static" | "shm_static"
    int src_gpu = 0;
    int dst_gpu = 0;
    double est_dit_ms = 0.0;
    int64_t reserved_kv_bytes = 0;
    int interleaved_roundtrip_gamma = 1;
    bool ephemeral_gen_ctx = false;
    bool needs_return_ship = false;
    bool release_src_on_fwd = false;
    bool include_cfg_dup_on_wire = false;
    std::string dit_ctx_id;

    nlohmann::json to_json() const;
};

// Typed per-turn input parsed once from canonical live or simulator outcome records.
struct TurnRecordInput {
    int turn_idx = 0;
    std::string modality_plan;
    std::string status = "ok";
    std::optional<double> ar_ttft_ms;
    std::vector<double> ar_tpots_ms;
    std::optional<double> ar_tpot_p95_ms;
    std::optional<double> ar_tpot_mean_ms;
    std::optional<double> ar_tpot_strict_max_ms;
    std::optional<double> dit_ms;
    int height = 0;
    int dit_steps = 0;
    std::optional<double> kv_mib;
    std::optional<bool> has_image;
    std::optional<int> num_images;
    int image_res = 0;
    bool use_prev_image = false;
    std::optional<double> queue_ms;
    std::optional<double> h2d_weight_ms;
    std::optional<double> kv_transfer_ms;
    std::optional<double> vae_ms;
    std::optional<double> preempt_yield_ms;

    static TurnRecordInput from_json(const nlohmann::json& j, int default_turn_idx);
};

// Typed per-session input parsed once from canonical live or simulator outcomes.
struct SessionOutcomeInput {
    std::string status = "error";
    std::string quadrant;
    std::string phase_label;
    double arrival_offset_s = 0.0;
    std::optional<double> e2e_ms;
    std::optional<double> active_jct_ms;
    double think_time_ms = 0.0;
    int planned_turns = 0;
    int planned_ar_turns = -1;
    int planned_dit_turns = -1;
    std::vector<double> ar_ttfts_ms;
    std::vector<double> ar_tpots_ms;
    std::vector<std::vector<double>> ar_turn_tpots_ms;
    std::vector<int> ar_turn_has_image;
    std::vector<int> ar_turn_num_images;
    std::vector<int> ar_turn_image_res;
    std::vector<double> dit_turns_ms;
    std::vector<int> dit_turn_heights;
    std::vector<int> dit_turn_steps;
    std::optional<bool> ar_slo_met;
    std::optional<bool> ar_slo_mean_tpot_met;
    std::optional<bool> ar_slo_strict_max_tpot_met;
    std::optional<bool> dit_slo_met;
    std::optional<bool> joint_slo_met;
    std::optional<double> resident_alpha_sample;
    std::optional<double> queue_ms;
    std::optional<double> h2d_weight_ms;
    std::optional<double> kv_transfer_ms;
    std::optional<double> vae_ms;
    std::optional<double> preempt_yield_ms;
    int path_a_ships = 0;
    int path_b_morphs = 0;
    int preempted_ar_turns = 0;
    int total_yields = 0;
    int oom_retries = 0;
    int cross_node_ships = 0;
    int64_t max_cross_node_ship_bytes = 0;
    std::vector<TurnRecordInput> turn_records;

    static SessionOutcomeInput from_json(const nlohmann::json& j);
};

// Typed per-worker telemetry input parsed once from live `GET_STATS` or simulator ledgers.
struct WorkerStatsInput {
    double nvml_peak_used_mib = 0.0;
    double combined_physical_mib = 0.0;
    double pytorch_reserved_mib = 0.0;
    double vmm_physical_pool_mib = 0.0;
    std::optional<double> nvml_sm_util_mean_pct;
    std::optional<double> nvml_mem_util_mean_pct;
    std::optional<double> effective_hbm_util_mean_pct;
    int oom_count = 0;
    int kv_host_swap_outs = 0;
    int kv_host_swap_ins = 0;
    int kv_recompute_evictions = 0;
    int kv_recompute_restores = 0;
    int overflow_creates = 0;
    int super_to_fine_carves = 0;
    int expert_auto_evictions_for_kv = 0;
    int expert_auto_evictions_for_transient = 0;

    static WorkerStatsInput from_json(const nlohmann::json& j);
};

// Unified CPTG session control & canonical Joint-SLO aggregation engine shared by live coordinator & simulator.
class MorphControlEngine {
public:
    MorphControlEngine(
        PolicyConfig policy,
        ClusterTopologyProfile topology,
        std::shared_ptr<DualPathMorphRouter> router = nullptr);

    const PolicyConfig& policy() const noexcept { return policy_; }
    const ClusterTopologyProfile& topology() const noexcept { return topology_; }
    std::shared_ptr<DualPathMorphRouter> router() const noexcept { return router_; }

    // Canonical `PreemptEdfController` constructor shared across live workers and `ClusterSimulator`
    // (aligned with live `bagel_adapter.py`: `guard_ms = 24.0 * time_scale`).
    static PreemptEdfController make_edf_controller(
        int num_layers,
        double tpot_slo_ms,
        double init_layer_ms,
        double init_ar_step_ms,
        double time_scale = 1.0,
        int max_microbatch_cap = 8) noexcept;

    static double deterministic_think_s(const std::string& session_id, int turn_idx, double lo_s, double hi_s);
    static double deterministic_think_ms_sim(int sid, int turn_idx, double lo_s, double hi_s);
    static double deterministic_think_dist_s(
        const std::string& session_id,
        int turn_idx,
        double lo_s,
        double hi_s,
        const std::string& dist = "lognormal",
        double median_s = 6.0,
        double sigma = 0.75);
    static double deterministic_think_dist_ms_sim(
        int sid,
        int turn_idx,
        double lo_s,
        double hi_s,
        const std::string& dist = "lognormal",
        double median_s = 6.0,
        double sigma = 0.75);

    // Plans and atomically reserves router/pacer transition state for live coordinator execution.
    TurnTransitionPlan plan_turn_transition(
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
        const std::map<int, bool>& gpu_busy = {},
        bool src_lock_held = false);

    // Read-only transition plan with zero router or pacer mutation (used by `ClusterSimulator` prior to dispatch).
    TurnTransitionPlan plan_turn_transition_readonly(
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
        const std::map<int, WorkerState>* ext_workers = nullptr,
        const std::map<int, bool>& gpu_busy = {},
        std::optional<int64_t> gen_expert_bytes_override = std::nullopt,
        std::optional<double> dit_step1_ms_override = std::nullopt,
        std::optional<double> est_dit_ms_override = std::nullopt) const;

    static nlohmann::json evaluate_session_slos(
        const PolicyConfig& policy,
        const std::vector<double>& ar_ttfts_ms,
        const std::vector<double>& ar_tpots_ms,
        const std::vector<double>& dit_turns_ms,
        const std::vector<int>& dit_heights,
        const std::vector<int>& dit_steps,
        const std::vector<std::vector<double>>& ar_turn_tpots_ms = {},
        double slo_time_scale = 1.0,
        const std::vector<int>& ar_turn_has_image = {},
        const std::vector<int>& ar_turn_num_images = {},
        const std::vector<int>& ar_turn_image_res = {});

    // Single canonical report aggregator shared by live `MorphServeClusterEngine` and `ClusterSimulator`.
    static nlohmann::json aggregate_cluster_run_report(
        const PolicyConfig& policy,
        const std::string& suite_name,
        double offered_qps,
        double wall_seconds,
        double host_shmem_delta_gib,
        const nlohmann::json& outcomes_array,
        const nlohmann::json& worker_stats_array,
        double slo_time_scale = 1.0,
        const nlohmann::json& sim_extra_fields = nlohmann::json::object(),
        bool include_sessions = true);

private:
    PolicyConfig policy_;
    ClusterTopologyProfile topology_;
    std::shared_ptr<DualPathMorphRouter> router_;
};

}  // namespace morphserve::control
