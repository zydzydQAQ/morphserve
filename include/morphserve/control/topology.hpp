#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace morphserve::control {

constexpr int64_t kMiB = 1024LL * 1024LL;
constexpr int64_t kGiB = 1024LL * 1024LL * 1024LL;

// Pure C++17 cryptographic/deterministic helpers (zero external OpenSSL dependency).
std::string sha256_hex_bytes(const uint8_t* data, size_t len);
std::string sha256_hex_file(const std::string& path);
uint32_t md5_prefix_u32(const std::string& text);

// Linear-interpolated percentile rounded to 2 decimal places (`morphserve/telemetry.py`).
double percentile_rounded(const std::vector<double>& vals, double p);

// Arithmetic-intensity gate for two-pass Cascade-CFG + fused LSE merge (`cfg_prefix_attn.py`).
bool should_use_cascade_cfg_split(int q_len, int l_shared, int num_shared_branches) noexcept;

struct OperatorLatencyProfile {
    double ar_prefill_ms = 64.8;
    double ar_tpot_ms = 62.82;
    double prep_overhead_ms = 240.0;
    double bagel_dit_512_step_ms = 287.0;
    double bagel_dit_1024_step_ms = 1040.0;
    double bagel_vae_512_ms = 140.0;
    double bagel_vae_1024_ms = 680.0;
    int64_t bagel_expert_bytes = 12446LL * kMiB;
    double internvlu_dit_512_step_ms = 145.0;
    double internvlu_dit_1024_step_ms = 420.0;
    double internvlu_vae_512_ms = 110.0;
    double internvlu_vae_1024_ms = 340.0;
    int64_t internvlu_expert_bytes = 4350LL * kMiB;

    double step_ms(const std::string& model_family, int height) const noexcept;
    double vae_ms(const std::string& model_family, int height) const noexcept;
    int64_t expert_bytes(const std::string& model_family) const noexcept;
    double unloaded_dit_turn_ms(const std::string& model_family, int height, int steps = 20) const noexcept;
    double dit_turn_slo_ms(
        int height,
        double k = 2.0,
        const std::string& model_family = "bagel",
        int steps = 20) const noexcept;
    std::map<std::string, double> slo_thresholds_ms(
        double k = 2.0,
        const std::string& model_family = "bagel") const;
};

const OperatorLatencyProfile& default_operator_profile() noexcept;

struct PcieHalfGroup {
    int group_id = 0;
    int numa_node = 0;
    std::vector<int> gpu_ids;
    double aggregate_h2d_gbps = 10.35;
    double solo_h2d_gbps = 8.75;
};

struct ClusterTopologyProfile {
    int num_gpus = 8;
    std::vector<std::string> gpu_names;
    std::vector<PcieHalfGroup> half_groups;
    std::vector<std::vector<double>> p2p_bw_gbps;
    double intra_half_p2p_cap_gbps = 19.67;
    double cross_half_p2p_cap_gbps = 9.98;
    double inter_node_p2p_cap_gbps = 11.5;
    int gpus_per_node = 8;
    double intra_node_handoff_ms = 3.0;
    double inter_node_handoff_ms = 8.0;

    int node_of(int gpu_id) const noexcept;
    bool same_node(int gpu_a, int gpu_b) const noexcept;
    const PcieHalfGroup& half_group_of(int gpu_id) const;
    int numa_node_of(int gpu_id) const;
    bool same_half(int gpu_a, int gpu_b) const;
    std::string fabric_tier(int src_gpu, int dst_gpu) const;
    double handoff_ms(int src_gpu, int dst_gpu) const noexcept;
    double solo_h2d_gbps(int gpu_id) const;
    double effective_h2d_gbps(int gpu_id, int active_h2d_same_half = 1) const;
    double effective_p2p_gbps(
        int src_gpu,
        int dst_gpu,
        int concurrent_intra_half = 1,
        int concurrent_cross_half = 1,
        int concurrent_inter_node = 1) const;
    std::vector<int> peer_candidates(int src_gpu, bool prefer_same_half = true) const;
    std::unordered_map<int, int> gpu_to_group_map() const;

    nlohmann::json to_json() const;
    static ClusterTopologyProfile from_json(const nlohmann::json& j);
    static ClusterTopologyProfile a100_pcie_8gpu_default();
    static ClusterTopologyProfile from_multinode_measurements(
        int world_size,
        int gpus_per_node,
        double intra_bw,
        double socket_bw,
        double inter_bw,
        double solo_h2d_bw = 8.75,
        double agg_h2d_bw = 10.35,
        const std::string& gpu_name = "NVIDIA A100-PCIE-40GB",
        double intra_handoff_ms = 3.0,
        double inter_handoff_ms = 8.0);
    static ClusterTopologyProfile from_job0c_report(const std::string& path);
};

// Empirical Phase-0/1 hardware & operator physics profile (`MeasuredPhysicsProfile`).
struct ClusterPhysicsProfile {
    double vmm_64m_acquire_4gib_ms = 5.437;
    double vmm_solo_h2d_stream_gbps = 12.334;
    double pcie_solo_h2d_gbps = 8.755;
    double pcie_half0_agg_h2d_gbps = 10.348;
    double pcie_half1_agg_h2d_gbps = 10.405;
    double pcie_intra_half_p2p_gbps = 10.581;
    double pcie_cross_half_p2p_gbps = 9.55;

    double vae_512_ms = 52.60;
    double vae_1024_ms = 212.32;
    double scoped_vae_512_spike_mib = 960.0;
    double unscoped_vae_512_reserved_mib = 2022.78;
    double scoped_vae_1024_spike_mib = 3648.0;
    double unscoped_vae_1024_reserved_mib = 6508.78;
    double twostage_shm_base_ms = 5.20;
    double twostage_shm_ms_per_mib = 4.2080;

    double monolithic_loaded_driver_mib = 28776.81;
    double offloaded_base_driver_mib = 16366.81;
    int num_expert_layers = 28;
    int layer_stride_mib = 448;
    double packed_expert_data_mib = 12446.64;
    double dit_512_20step_base_ms = 4958.59;
    double dit_512_transient_mib = 1088.0;
    double dit_1024_step_ms = 966.335;
    double vmm_full_expert_map_ms = 11.677;

    double cpu_pil_prep_only_ms = 69.80;
    double gpu_inplace_prep_only_ms = 2.452;
    double stock_loopback_turn_ms = 887.943;
    double morphserve_loopback_turn_ms = 793.547;
    double non_cow_kv_mult = 1.9979;
    double turn_edit_512_kv_delta_mib = 325.28;
    double kv_mib_per_token = 0.05469;
    double stock_dit_512_ms_per_kv_tok = 0.107796;
    double morphserve_dit_512_ms_per_kv_tok = 0.097462;

    double ar_prefill_ms = 64.80;
    double ar_unloaded_tpot_p50_ms = 62.82;
    double dit_512_cfg_step_mean_ms = 286.55;
    double dit_512_layer_p50_ms = 10.52;
    int m4_adaptive_chunk_layers = 5;
    double m4_colocated_ar_tpot_p50_ms = 110.98;
    double step_preempt_ar_tpot_p50_ms = 346.13;

    double paced_h2d_per_gpu_gbps = 12.34;
    double unpaced_4gpu_half_h2d_per_gpu_gbps = 3.10;
    double nccl_intra_half_kv_gbps = 10.362;

    double kairos_2x8_intra_half_gbps = 8.37;
    double kairos_2x8_cross_socket_gbps = 8.59;
    double kairos_2x8_inter_node_gbps = 1.11;

    std::map<std::string, std::string> provenance_files;

    nlohmann::json to_json() const;
    static ClusterPhysicsProfile from_json(const nlohmann::json& j);
    static ClusterPhysicsProfile from_phase01_results(const std::string& results_dir);
};

struct WorkloadScaleSpec {
    std::string name = "bagel_7b_standard";
    double weight_scale = 1.0;
    double compute_scale = 1.0;
    double kv_per_turn_scale = 1.0;
    double vae_spike_scale = 1.0;
    int extra_edit_turns = 0;
    std::optional<std::pair<double, double>> think_s_range_override;
    int super_pool_mib_80g = 28672;
    int fine_kv_pool_mib_80g = 512;

    nlohmann::json to_json() const;
    static WorkloadScaleSpec from_json(const nlohmann::json& j);
    static WorkloadScaleSpec bagel_7b_standard();
    static WorkloadScaleSpec frontier_mot_56gib();
    static WorkloadScaleSpec bagel_7b_long_kv_high_concurrency();
};

struct HardwareInterconnectSpec {
    std::string name = "A100_PCIe_Gen4_RDMA100G";
    double gpu_hbm_gib = 40.0;
    double compute_speedup_x = 1.0;
    double solo_h2d_gbps = 8.755;
    double agg_h2d_gbps = 10.348;
    double intra_half_p2p_gbps = 10.581;
    double inter_socket_p2p_gbps = 9.55;
    double inter_node_rdma_gbps = 11.5;
    int gpus_per_node = 8;

    ClusterTopologyProfile to_topology(int num_gpus) const;
    nlohmann::json to_json() const;
    static HardwareInterconnectSpec from_json(const nlohmann::json& j);
    static std::vector<HardwareInterconnectSpec> presets(
        const ClusterPhysicsProfile* physics = nullptr);
};

}  // namespace morphserve::control
