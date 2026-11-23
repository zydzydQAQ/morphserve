#pragma once

#include "morphserve/control/policy.hpp"
#include "morphserve/control/router.hpp"
#include "morphserve/control/topology.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>

namespace morphserve::control {

struct TurnTransitionPlan {
    RouteDecision decision;
    std::string ship_mode;  // "none" | "nccl_dual" | "nccl_static" | "shm_static"
    int src_gpu = 0;
    int dst_gpu = 0;
    double est_dit_ms = 0.0;
    int64_t reserved_kv_bytes = 0;
    bool ephemeral_gen_ctx = false;
    bool needs_return_ship = false;
    bool release_src_on_fwd = false;
    bool include_cfg_dup_on_wire = false;
    std::string dit_ctx_id;

    nlohmann::json to_json() const;
};

// Unified CPTG session state-machine & Joint-SLO aggregation engine shared by live coordinator & simulator.
class MorphControlEngine {
public:
    MorphControlEngine(
        PolicyConfig policy,
        ClusterTopologyProfile topology,
        std::shared_ptr<DualPathMorphRouter> router = nullptr);

    const PolicyConfig& policy() const noexcept { return policy_; }
    const ClusterTopologyProfile& topology() const noexcept { return topology_; }
    std::shared_ptr<DualPathMorphRouter> router() const noexcept { return router_; }

    static double deterministic_think_s(const std::string& session_id, int turn_idx, double lo_s, double hi_s);
    static double deterministic_think_ms_sim(int sid, int turn_idx, double lo_s, double hi_s);

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
        const std::map<int, bool>& gpu_busy = {});

    static nlohmann::json evaluate_session_slos(
        const PolicyConfig& policy,
        const std::vector<double>& ar_ttfts_ms,
        const std::vector<double>& ar_tpots_ms,
        const std::vector<double>& dit_turns_ms,
        const std::vector<int>& dit_heights,
        const std::vector<int>& dit_steps);

    static nlohmann::json aggregate_cluster_run_report(
        const PolicyConfig& policy,
        const std::string& suite_name,
        double offered_qps,
        double wall_seconds,
        double host_shmem_delta_gib,
        const nlohmann::json& outcomes_array,
        const nlohmann::json& worker_stats_array);

private:
    PolicyConfig policy_;
    ClusterTopologyProfile topology_;
    std::shared_ptr<DualPathMorphRouter> router_;
};

}  // namespace morphserve::control
