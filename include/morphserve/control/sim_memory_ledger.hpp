#pragma once

#include "morphserve/control/policy.hpp"
#include "morphserve/control/topology.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>

namespace morphserve::control {

struct SessionRestoreResult {
    bool ok = true;
    double swapped_in_mib = 0.0;
    double recompute_kv_mib = 0.0;
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
    int kv_recompute_evictions() const noexcept { return kv_recompute_evictions_; }
    int kv_recompute_restores() const noexcept { return kv_recompute_restores_; }
    bool is_post_evict() const noexcept { return post_evict_; }
    double peak_used_mib() const noexcept { return peak_used_mib_; }
    double effective_hbm_util_mean_pct() const noexcept;
    double last_fail_dit_frac() const noexcept { return last_fail_dit_frac_; }
    void set_peak_used_mib(double v) noexcept { peak_used_mib_ = v; }

    double offloaded_base_mib() const noexcept;
    double monolithic_weights_mib() const noexcept;
    double pageable_expert_mib() const noexcept;
    double base_weights_mib() const noexcept;
    double expert_alpha() const noexcept;
    double active_kv_mib() const noexcept;
    size_t active_session_count() const noexcept { return session_kv_mib_.size(); }
    int64_t free_kv_bytes() const noexcept;

    double evict_idle_kv_to_host(double needed_free_mib, std::optional<int> exclude_sid = std::nullopt);
    void reclaim_empty_slabs_to_super() noexcept;
    SessionRestoreResult restore_session_resident(int sid);
    double ensure_session_resident(int sid);
    bool upsert_session_kv(int sid, double logical_kv_mib);
    std::pair<bool, double> execute_dit_and_vae(
        int height,
        std::optional<int> active_sid = std::nullopt,
        std::optional<double> post_denoise_sid_kv_mib = std::nullopt,
        double unpack_staging_mib = 0.0,
        bool run_pre_denoise_vae = false);
    bool check_transient_spike(
        double spike_mib,
        std::optional<int> active_sid = std::nullopt,
        bool contiguous_alloc = false,
        double retain_unscoped_frag_mib = 0.0);
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
    double cur_super_pool_mib_ = 0.0;
    double cur_fine_pool_mib_ = 0.0;
    double retained_unscoped_vae_mib_ = 0.0;

    std::map<int, double> session_kv_mib_;
    std::map<int, double> swapped_session_kv_mib_;
    std::map<int, double> dropped_recompute_kv_mib_;
    std::map<int, uint64_t> session_last_touch_;
    uint64_t touch_clock_ = 0;
    int super_to_fine_carves_ = 0;
    int expert_evictions_ = 0;
    int kv_host_swap_outs_ = 0;
    int kv_host_swap_ins_ = 0;
    int kv_recompute_evictions_ = 0;
    int kv_recompute_restores_ = 0;
    bool post_evict_ = false;
    double peak_used_mib_ = 0.0;
    double hbm_used_samples_sum_ = 0.0;
    int64_t hbm_used_samples_cnt_ = 0;
    double last_fail_dit_frac_ = 0.35;

    void touch_session(int sid) noexcept { session_last_touch_[sid] = ++touch_clock_; }
    void record_hbm_sample(double used_mib) noexcept;
};

}  // namespace morphserve::control
