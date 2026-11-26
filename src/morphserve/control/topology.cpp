#include "morphserve/control/topology.hpp"

#include "morphserve/control/hash_util.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace morphserve::control {

namespace {

nlohmann::json read_json_file(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open JSON file: " + p.string());
    }
    return nlohmann::json::parse(in);
}

std::optional<std::filesystem::path> first_existing_file(
    const std::filesystem::path& root,
    const std::vector<std::string>& candidates) {
    for (const auto& rel : candidates) {
        std::filesystem::path cand = root / rel;
        std::error_code ec;
        if (std::filesystem::is_regular_file(cand, ec)) {
            return cand;
        }
    }
    return std::nullopt;
}

}  // namespace

double OperatorLatencyProfile::step_ms(const std::string& model_family, int height) const noexcept {
    const bool is_1024 = height >= 1024;
    if (model_family == "internvl_u") {
        return is_1024 ? internvlu_dit_1024_step_ms : internvlu_dit_512_step_ms;
    }
    return is_1024 ? bagel_dit_1024_step_ms : bagel_dit_512_step_ms;
}

double OperatorLatencyProfile::vae_ms(const std::string& model_family, int height) const noexcept {
    const bool is_1024 = height >= 1024;
    if (model_family == "internvl_u") {
        return is_1024 ? internvlu_vae_1024_ms : internvlu_vae_512_ms;
    }
    return is_1024 ? bagel_vae_1024_ms : bagel_vae_512_ms;
}

int64_t OperatorLatencyProfile::expert_bytes(const std::string& model_family) const noexcept {
    return (model_family == "internvl_u") ? internvlu_expert_bytes : bagel_expert_bytes;
}

double OperatorLatencyProfile::unloaded_dit_turn_ms(
    const std::string& model_family,
    int height,
    int steps) const noexcept {
    const int eff_steps = std::max(1, steps);
    return static_cast<double>(eff_steps) * step_ms(model_family, height) +
           vae_ms(model_family, height) + prep_overhead_ms;
}

double OperatorLatencyProfile::dit_turn_slo_ms(
    int height,
    double k,
    const std::string& model_family,
    int steps) const noexcept {
    return round_n(k * unloaded_dit_turn_ms(model_family, height, steps), 2);
}

std::map<std::string, double> OperatorLatencyProfile::slo_thresholds_ms(
    double k,
    const std::string& model_family) const {
    const int steps_1024 = (model_family == "internvl_u") ? 16 : 20;
    return {
        {"slo_scale_k", k},
        {"ar_tpot_slo_ms", round_n(k * ar_tpot_ms, 2)},
        {"ar_ttft_slo_ms", round_n(750.0 * k, 2)},
        {"dit_512_slo_ms", dit_turn_slo_ms(512, k, model_family, 20)},
        {"dit_1024_slo_ms", dit_turn_slo_ms(1024, k, model_family, steps_1024)},
    };
}

nlohmann::json OperatorLatencyProfile::to_json() const {
    return {
        {"ar_prefill_ms", ar_prefill_ms},
        {"ar_tpot_ms", ar_tpot_ms},
        {"prep_overhead_ms", prep_overhead_ms},
        {"bagel_dit_512_step_ms", bagel_dit_512_step_ms},
        {"bagel_dit_1024_step_ms", bagel_dit_1024_step_ms},
        {"bagel_vae_512_ms", bagel_vae_512_ms},
        {"bagel_vae_1024_ms", bagel_vae_1024_ms},
        {"bagel_expert_bytes", bagel_expert_bytes},
        {"internvlu_dit_512_step_ms", internvlu_dit_512_step_ms},
        {"internvlu_dit_1024_step_ms", internvlu_dit_1024_step_ms},
        {"internvlu_vae_512_ms", internvlu_vae_512_ms},
        {"internvlu_vae_1024_ms", internvlu_vae_1024_ms},
        {"internvlu_expert_bytes", internvlu_expert_bytes},
    };
}

OperatorLatencyProfile OperatorLatencyProfile::from_json(const nlohmann::json& j) {
    OperatorLatencyProfile p;
    if (!j.is_object()) {
        return p;
    }
    p.ar_prefill_ms = j.value("ar_prefill_ms", p.ar_prefill_ms);
    p.ar_tpot_ms = j.value("ar_tpot_ms", p.ar_tpot_ms);
    p.prep_overhead_ms = j.value("prep_overhead_ms", p.prep_overhead_ms);
    p.bagel_dit_512_step_ms = j.value("bagel_dit_512_step_ms", p.bagel_dit_512_step_ms);
    p.bagel_dit_1024_step_ms = j.value("bagel_dit_1024_step_ms", p.bagel_dit_1024_step_ms);
    p.bagel_vae_512_ms = j.value("bagel_vae_512_ms", p.bagel_vae_512_ms);
    p.bagel_vae_1024_ms = j.value("bagel_vae_1024_ms", p.bagel_vae_1024_ms);
    p.bagel_expert_bytes = j.value("bagel_expert_bytes", p.bagel_expert_bytes);
    p.internvlu_dit_512_step_ms = j.value("internvlu_dit_512_step_ms", p.internvlu_dit_512_step_ms);
    p.internvlu_dit_1024_step_ms = j.value("internvlu_dit_1024_step_ms", p.internvlu_dit_1024_step_ms);
    p.internvlu_vae_512_ms = j.value("internvlu_vae_512_ms", p.internvlu_vae_512_ms);
    p.internvlu_vae_1024_ms = j.value("internvlu_vae_1024_ms", p.internvlu_vae_1024_ms);
    p.internvlu_expert_bytes = j.value("internvlu_expert_bytes", p.internvlu_expert_bytes);
    return p;
}

const OperatorLatencyProfile& default_operator_profile() noexcept {
    static const OperatorLatencyProfile kDefault{};
    return kDefault;
}

int ClusterTopologyProfile::node_of(int gpu_id) const noexcept {
    return gpu_id / std::max(1, gpus_per_node);
}

bool ClusterTopologyProfile::same_node(int gpu_a, int gpu_b) const noexcept {
    return node_of(gpu_a) == node_of(gpu_b);
}

const PcieHalfGroup& ClusterTopologyProfile::half_group_of(int gpu_id) const {
    for (const auto& grp : half_groups) {
        if (std::find(grp.gpu_ids.begin(), grp.gpu_ids.end(), gpu_id) != grp.gpu_ids.end()) {
            return grp;
        }
    }
    throw std::out_of_range("gpu_id=" + std::to_string(gpu_id) + " not mapped to any PcieHalfGroup");
}

int ClusterTopologyProfile::numa_node_of(int gpu_id) const {
    return half_group_of(gpu_id).numa_node;
}

bool ClusterTopologyProfile::same_half(int gpu_a, int gpu_b) const {
    return same_node(gpu_a, gpu_b) && (half_group_of(gpu_a).group_id == half_group_of(gpu_b).group_id);
}

std::string ClusterTopologyProfile::fabric_tier(int src_gpu, int dst_gpu) const {
    if (src_gpu == dst_gpu) {
        return "self";
    }
    if (!same_node(src_gpu, dst_gpu)) {
        return "inter_node";
    }
    if (same_half(src_gpu, dst_gpu)) {
        return "intra_half";
    }
    return "inter_socket";
}

double ClusterTopologyProfile::handoff_ms(int src_gpu, int dst_gpu) const noexcept {
    if (src_gpu == dst_gpu) {
        return 0.0;
    }
    return same_node(src_gpu, dst_gpu) ? intra_node_handoff_ms : inter_node_handoff_ms;
}

double ClusterTopologyProfile::solo_h2d_gbps(int gpu_id) const {
    return half_group_of(gpu_id).solo_h2d_gbps;
}

double ClusterTopologyProfile::effective_h2d_gbps(int gpu_id, int active_h2d_same_half) const {
    const auto& grp = half_group_of(gpu_id);
    const int k = std::max(1, active_h2d_same_half);
    return std::min(grp.solo_h2d_gbps, grp.aggregate_h2d_gbps / static_cast<double>(k));
}

double ClusterTopologyProfile::effective_p2p_gbps(
    int src_gpu,
    int dst_gpu,
    int concurrent_intra_half,
    int concurrent_cross_half,
    int concurrent_inter_node) const {
    const std::string tier = fabric_tier(src_gpu, dst_gpu);
    if (tier == "self") {
        return 900.0;
    }
    const double raw = p2p_bw_gbps.at(static_cast<size_t>(src_gpu)).at(static_cast<size_t>(dst_gpu));
    double cap = 0.0;
    if (tier == "intra_half") {
        cap = intra_half_p2p_cap_gbps / static_cast<double>(std::max(1, concurrent_intra_half));
    } else if (tier == "inter_socket") {
        cap = cross_half_p2p_cap_gbps / static_cast<double>(std::max(1, concurrent_cross_half));
    } else {
        cap = inter_node_p2p_cap_gbps / static_cast<double>(std::max(1, concurrent_inter_node));
    }
    return std::max(0.5, std::min(raw, cap));
}

std::vector<int> ClusterTopologyProfile::peer_candidates(int src_gpu, bool prefer_same_half) const {
    std::vector<int> peers;
    peers.reserve(static_cast<size_t>(std::max(0, num_gpus - 1)));
    for (int g = 0; g < num_gpus; ++g) {
        if (g != src_gpu) {
            peers.push_back(g);
        }
    }
    if (prefer_same_half) {
        auto tier_rank = [&](int g) -> int {
            const std::string t = fabric_tier(src_gpu, g);
            if (t == "intra_half") return 0;
            if (t == "inter_socket") return 1;
            if (t == "inter_node") return 2;
            return 3;
        };
        std::sort(peers.begin(), peers.end(), [&](int a, int b) {
            const int ta = tier_rank(a);
            const int tb = tier_rank(b);
            if (ta != tb) return ta < tb;
            const double bwa = p2p_bw_gbps.at(static_cast<size_t>(src_gpu)).at(static_cast<size_t>(a));
            const double bwb = p2p_bw_gbps.at(static_cast<size_t>(src_gpu)).at(static_cast<size_t>(b));
            if (bwa != bwb) return bwa > bwb;
            return a < b;
        });
    }
    return peers;
}

std::unordered_map<int, int> ClusterTopologyProfile::gpu_to_group_map() const {
    std::unordered_map<int, int> out;
    for (const auto& grp : half_groups) {
        for (int gid : grp.gpu_ids) {
            out[gid] = grp.group_id;
        }
    }
    return out;
}

nlohmann::json ClusterTopologyProfile::to_json() const {
    nlohmann::json grps = nlohmann::json::array();
    for (const auto& g : half_groups) {
        grps.push_back({
            {"group_id", g.group_id},
            {"numa_node", g.numa_node},
            {"gpu_ids", g.gpu_ids},
            {"aggregate_h2d_gbps", g.aggregate_h2d_gbps},
            {"solo_h2d_gbps", g.solo_h2d_gbps},
        });
    }
    return {
        {"num_gpus", num_gpus},
        {"gpus_per_node", gpus_per_node},
        {"gpu_names", gpu_names},
        {"half_groups", grps},
        {"intra_half_p2p_cap_gbps", intra_half_p2p_cap_gbps},
        {"cross_half_p2p_cap_gbps", cross_half_p2p_cap_gbps},
        {"inter_node_p2p_cap_gbps", inter_node_p2p_cap_gbps},
        {"intra_node_handoff_ms", intra_node_handoff_ms},
        {"inter_node_handoff_ms", inter_node_handoff_ms},
        {"p2p_bw_gbps", p2p_bw_gbps},
    };
}

ClusterTopologyProfile ClusterTopologyProfile::from_json(const nlohmann::json& j) {
    ClusterTopologyProfile out;
    out.num_gpus = j.value("num_gpus", 8);
    out.gpus_per_node = j.value("gpus_per_node", 8);
    if (j.contains("gpu_names") && j["gpu_names"].is_array()) {
        out.gpu_names = j["gpu_names"].get<std::vector<std::string>>();
    } else {
        out.gpu_names.assign(static_cast<size_t>(out.num_gpus), "NVIDIA A100-PCIE-40GB");
    }
    if (j.contains("half_groups") && j["half_groups"].is_array()) {
        for (const auto& gj : j["half_groups"]) {
            PcieHalfGroup grp;
            grp.group_id = gj.value("group_id", 0);
            grp.numa_node = gj.value("numa_node", 0);
            grp.gpu_ids = gj.value("gpu_ids", std::vector<int>{});
            grp.aggregate_h2d_gbps = gj.value("aggregate_h2d_gbps", 10.35);
            grp.solo_h2d_gbps = gj.value("solo_h2d_gbps", 8.75);
            out.half_groups.push_back(std::move(grp));
        }
    }
    out.intra_half_p2p_cap_gbps = j.value("intra_half_p2p_cap_gbps", 19.67);
    out.cross_half_p2p_cap_gbps = j.value("cross_half_p2p_cap_gbps", 9.98);
    out.inter_node_p2p_cap_gbps = j.value("inter_node_p2p_cap_gbps", 11.5);
    out.intra_node_handoff_ms = j.value("intra_node_handoff_ms", 3.0);
    out.inter_node_handoff_ms = j.value("inter_node_handoff_ms", 8.0);
    if (j.contains("p2p_bw_gbps") && j["p2p_bw_gbps"].is_array()) {
        out.p2p_bw_gbps = j["p2p_bw_gbps"].get<std::vector<std::vector<double>>>();
    }
    return out;
}

ClusterTopologyProfile ClusterTopologyProfile::a100_pcie_8gpu_default() {
    ClusterTopologyProfile prof;
    prof.num_gpus = 8;
    prof.gpu_names.assign(8, "NVIDIA A100-PCIE-40GB");
    prof.half_groups = {
        PcieHalfGroup{0, 0, {0, 1, 2, 3}, 10.35, 8.75},
        PcieHalfGroup{1, 1, {4, 5, 6, 7}, 10.40, 8.75},
    };
    prof.p2p_bw_gbps.assign(8, std::vector<double>(8, 0.0));
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            if (i == j) continue;
            const bool same = ((i < 4) && (j < 4)) || ((i >= 4) && (j >= 4));
            prof.p2p_bw_gbps[static_cast<size_t>(i)][static_cast<size_t>(j)] = same ? 10.6 : 9.55;
        }
    }
    prof.intra_half_p2p_cap_gbps = 19.67;
    prof.cross_half_p2p_cap_gbps = 9.98;
    prof.inter_node_p2p_cap_gbps = 11.5;
    prof.gpus_per_node = 8;
    prof.intra_node_handoff_ms = 3.0;
    prof.inter_node_handoff_ms = 8.0;
    return prof;
}

ClusterTopologyProfile ClusterTopologyProfile::from_multinode_measurements(
    int world_size,
    int gpus_per_node,
    double intra_bw,
    double socket_bw,
    double inter_bw,
    double solo_h2d_bw,
    double agg_h2d_bw,
    const std::string& gpu_name,
    double intra_handoff_ms,
    double inter_handoff_ms) {
    const int gpn = std::max(1, gpus_per_node);
    const int half_sz = std::max(1, std::min(4, (gpn >= 2) ? (gpn / 2) : 1));
    const int n_groups = std::max(1, (world_size + half_sz - 1) / half_sz);
    std::vector<PcieHalfGroup> groups;
    groups.reserve(static_cast<size_t>(n_groups));
    for (int gi = 0; gi < n_groups; ++gi) {
        std::vector<int> ids;
        for (int g = gi * half_sz; g < std::min(world_size, (gi + 1) * half_sz); ++g) {
            ids.push_back(g);
        }
        groups.push_back(PcieHalfGroup{gi, gi % 2, std::move(ids), agg_h2d_bw, solo_h2d_bw});
    }
    std::vector<std::vector<double>> bw_mat(
        static_cast<size_t>(world_size), std::vector<double>(static_cast<size_t>(world_size), 0.0));
    for (int s = 0; s < world_size; ++s) {
        for (int d = 0; d < world_size; ++d) {
            if (s == d) {
                bw_mat[static_cast<size_t>(s)][static_cast<size_t>(d)] = 900.0;
            } else if ((s / gpn) != (d / gpn)) {
                bw_mat[static_cast<size_t>(s)][static_cast<size_t>(d)] = inter_bw;
            } else if ((s / half_sz) == (d / half_sz)) {
                bw_mat[static_cast<size_t>(s)][static_cast<size_t>(d)] = intra_bw;
            } else {
                bw_mat[static_cast<size_t>(s)][static_cast<size_t>(d)] = socket_bw;
            }
        }
    }
    ClusterTopologyProfile out;
    out.num_gpus = world_size;
    out.gpu_names.assign(static_cast<size_t>(world_size), gpu_name);
    out.half_groups = std::move(groups);
    out.p2p_bw_gbps = std::move(bw_mat);
    out.intra_half_p2p_cap_gbps = std::max(0.5, intra_bw);
    out.cross_half_p2p_cap_gbps = std::max(0.5, socket_bw);
    out.inter_node_p2p_cap_gbps = std::max(0.5, inter_bw);
    out.gpus_per_node = gpn;
    out.intra_node_handoff_ms = intra_handoff_ms;
    out.inter_node_handoff_ms = inter_handoff_ms;
    return out;
}

ClusterTopologyProfile ClusterTopologyProfile::from_job0c_report(const std::string& path) {
    const auto data = read_json_file(path);
    auto prof = a100_pcie_8gpu_default();
    const nlohmann::json* mat = nullptr;
    if (data.contains("p2p_matrix_gbps") && data["p2p_matrix_gbps"].is_array()) {
        mat = &data["p2p_matrix_gbps"];
    } else if (data.contains("p2p_bw_gbps") && data["p2p_bw_gbps"].is_array()) {
        mat = &data["p2p_bw_gbps"];
    } else if (data.contains("p2p") && data["p2p"].is_object() &&
               data["p2p"].contains("bandwidth_gbps") && data["p2p"]["bandwidth_gbps"].is_array()) {
        mat = &data["p2p"]["bandwidth_gbps"];
    }
    if (mat != nullptr && static_cast<int>(mat->size()) == prof.num_gpus) {
        for (int i = 0; i < prof.num_gpus; ++i) {
            const auto& row = (*mat)[static_cast<size_t>(i)];
            if (row.is_array() && static_cast<int>(row.size()) == prof.num_gpus) {
                for (int j = 0; j < prof.num_gpus; ++j) {
                    prof.p2p_bw_gbps[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                        row[static_cast<size_t>(j)].is_number() ? row[static_cast<size_t>(j)].get<double>() : 0.0;
                }
            }
        }
    }
    nlohmann::json h2d = nlohmann::json::object();
    if (data.contains("h2d_contention") && data["h2d_contention"].is_object()) {
        h2d = data["h2d_contention"];
    } else if (data.contains("numa_h2d") && data["numa_h2d"].is_object()) {
        h2d = data["numa_h2d"];
    }
    if (!h2d.empty()) {
        double h0 = 10.35;
        double h1 = 10.40;
        double solo = 8.75;
        if (h2d.contains("half0_aggregate_gbps") && h2d["half0_aggregate_gbps"].is_number()) {
            h0 = h2d["half0_aggregate_gbps"].get<double>();
        } else if (h2d.contains("half0_4gpu_agg_gbps") && h2d["half0_4gpu_agg_gbps"].is_number()) {
            h0 = h2d["half0_4gpu_agg_gbps"].get<double>();
        }
        if (h2d.contains("half1_aggregate_gbps") && h2d["half1_aggregate_gbps"].is_number()) {
            h1 = h2d["half1_aggregate_gbps"].get<double>();
        } else if (h2d.contains("half1_4gpu_agg_gbps") && h2d["half1_4gpu_agg_gbps"].is_number()) {
            h1 = h2d["half1_4gpu_agg_gbps"].get<double>();
        }
        if (h2d.contains("solo_gpu0_gbps") && h2d["solo_gpu0_gbps"].is_number()) {
            solo = h2d["solo_gpu0_gbps"].get<double>();
        } else if (h2d.contains("memfd_numa0_gbps") && h2d["memfd_numa0_gbps"].is_number()) {
            solo = h2d["memfd_numa0_gbps"].get<double>();
        }
        prof.half_groups = {
            PcieHalfGroup{0, 0, {0, 1, 2, 3}, h0 > 0.0 ? h0 : 10.35, solo > 0.0 ? solo : 8.75},
            PcieHalfGroup{1, 1, {4, 5, 6, 7}, h1 > 0.0 ? h1 : 10.40, solo > 0.0 ? solo : 8.75},
        };
    }
    return prof;
}

nlohmann::json ClusterPhysicsProfile::to_json() const {
    return {
        {"vmm_64m_acquire_4gib_ms", vmm_64m_acquire_4gib_ms},
        {"vmm_solo_h2d_stream_gbps", vmm_solo_h2d_stream_gbps},
        {"pcie_solo_h2d_gbps", pcie_solo_h2d_gbps},
        {"pcie_half0_agg_h2d_gbps", pcie_half0_agg_h2d_gbps},
        {"pcie_half1_agg_h2d_gbps", pcie_half1_agg_h2d_gbps},
        {"pcie_intra_half_p2p_gbps", pcie_intra_half_p2p_gbps},
        {"pcie_cross_half_p2p_gbps", pcie_cross_half_p2p_gbps},
        {"vae_512_ms", vae_512_ms},
        {"vae_1024_ms", vae_1024_ms},
        {"scoped_vae_512_spike_mib", scoped_vae_512_spike_mib},
        {"unscoped_vae_512_reserved_mib", unscoped_vae_512_reserved_mib},
        {"scoped_vae_1024_spike_mib", scoped_vae_1024_spike_mib},
        {"unscoped_vae_1024_reserved_mib", unscoped_vae_1024_reserved_mib},
        {"twostage_shm_base_ms", twostage_shm_base_ms},
        {"twostage_shm_ms_per_mib", twostage_shm_ms_per_mib},
        {"monolithic_loaded_driver_mib", monolithic_loaded_driver_mib},
        {"offloaded_base_driver_mib", offloaded_base_driver_mib},
        {"driver_total_hbm_40g_mib", driver_total_hbm_40g_mib},
        {"num_expert_layers", num_expert_layers},
        {"layer_stride_mib", layer_stride_mib},
        {"packed_expert_data_mib", packed_expert_data_mib},
        {"dit_512_20step_base_ms", dit_512_20step_base_ms},
        {"dit_512_transient_mib", dit_512_transient_mib},
        {"dit_1024_step_ms", dit_1024_step_ms},
        {"vmm_full_expert_map_ms", vmm_full_expert_map_ms},
        {"cpu_pil_prep_only_ms", cpu_pil_prep_only_ms},
        {"gpu_inplace_prep_only_ms", gpu_inplace_prep_only_ms},
        {"stock_loopback_turn_ms", stock_loopback_turn_ms},
        {"morphserve_loopback_turn_ms", morphserve_loopback_turn_ms},
        {"non_cow_kv_mult", non_cow_kv_mult},
        {"turn_edit_512_kv_delta_mib", turn_edit_512_kv_delta_mib},
        {"kv_mib_per_token", kv_mib_per_token},
        {"stock_dit_512_ms_per_kv_tok", stock_dit_512_ms_per_kv_tok},
        {"morphserve_dit_512_ms_per_kv_tok", morphserve_dit_512_ms_per_kv_tok},
        {"ar_prefill_ms", ar_prefill_ms},
        {"ar_unloaded_tpot_p50_ms", ar_unloaded_tpot_p50_ms},
        {"ar_unloaded_tpot_p99_ms", ar_unloaded_tpot_p99_ms},
        {"ar_batched2_step_p50_ms", ar_batched2_step_p50_ms},
        {"vmm_ar_kv_step_overhead_ms", vmm_ar_kv_step_overhead_ms},
        {"dit_512_cfg_step_mean_ms", dit_512_cfg_step_mean_ms},
        {"dit_512_layer_p50_ms", dit_512_layer_p50_ms},
        {"m4_adaptive_chunk_layers", m4_adaptive_chunk_layers},
        {"m4_colocated_ar_tpot_p50_ms", m4_colocated_ar_tpot_p50_ms},
        {"step_preempt_ar_tpot_p50_ms", step_preempt_ar_tpot_p50_ms},
        {"paced_h2d_per_gpu_gbps", paced_h2d_per_gpu_gbps},
        {"unpaced_4gpu_half_h2d_per_gpu_gbps", unpaced_4gpu_half_h2d_per_gpu_gbps},
        {"nccl_intra_half_kv_gbps", nccl_intra_half_kv_gbps},
        {"kairos_2x8_intra_half_gbps", kairos_2x8_intra_half_gbps},
        {"kairos_2x8_cross_socket_gbps", kairos_2x8_cross_socket_gbps},
        {"kairos_2x8_inter_node_gbps", kairos_2x8_inter_node_gbps},
        {"provenance_files", provenance_files},
    };
}

ClusterPhysicsProfile ClusterPhysicsProfile::from_json(const nlohmann::json& j) {
    ClusterPhysicsProfile p;
    p.vmm_64m_acquire_4gib_ms = j.value("vmm_64m_acquire_4gib_ms", p.vmm_64m_acquire_4gib_ms);
    p.vmm_solo_h2d_stream_gbps = j.value("vmm_solo_h2d_stream_gbps", p.vmm_solo_h2d_stream_gbps);
    p.pcie_solo_h2d_gbps = j.value("pcie_solo_h2d_gbps", p.pcie_solo_h2d_gbps);
    p.pcie_half0_agg_h2d_gbps = j.value("pcie_half0_agg_h2d_gbps", p.pcie_half0_agg_h2d_gbps);
    p.pcie_half1_agg_h2d_gbps = j.value("pcie_half1_agg_h2d_gbps", p.pcie_half1_agg_h2d_gbps);
    p.pcie_intra_half_p2p_gbps = j.value("pcie_intra_half_p2p_gbps", p.pcie_intra_half_p2p_gbps);
    p.pcie_cross_half_p2p_gbps = j.value("pcie_cross_half_p2p_gbps", p.pcie_cross_half_p2p_gbps);
    p.vae_512_ms = j.value("vae_512_ms", p.vae_512_ms);
    p.vae_1024_ms = j.value("vae_1024_ms", p.vae_1024_ms);
    p.scoped_vae_512_spike_mib = j.value("scoped_vae_512_spike_mib", p.scoped_vae_512_spike_mib);
    p.unscoped_vae_512_reserved_mib = j.value("unscoped_vae_512_reserved_mib", p.unscoped_vae_512_reserved_mib);
    p.scoped_vae_1024_spike_mib = j.value("scoped_vae_1024_spike_mib", p.scoped_vae_1024_spike_mib);
    p.unscoped_vae_1024_reserved_mib = j.value("unscoped_vae_1024_reserved_mib", p.unscoped_vae_1024_reserved_mib);
    p.twostage_shm_base_ms = j.value("twostage_shm_base_ms", p.twostage_shm_base_ms);
    p.twostage_shm_ms_per_mib = j.value("twostage_shm_ms_per_mib", p.twostage_shm_ms_per_mib);
    p.monolithic_loaded_driver_mib = j.value("monolithic_loaded_driver_mib", p.monolithic_loaded_driver_mib);
    p.offloaded_base_driver_mib = j.value("offloaded_base_driver_mib", p.offloaded_base_driver_mib);
    p.driver_total_hbm_40g_mib = j.value("driver_total_hbm_40g_mib", p.driver_total_hbm_40g_mib);
    p.num_expert_layers = j.value("num_expert_layers", p.num_expert_layers);
    p.layer_stride_mib = j.value("layer_stride_mib", p.layer_stride_mib);
    p.packed_expert_data_mib = j.value("packed_expert_data_mib", p.packed_expert_data_mib);
    p.dit_512_20step_base_ms = j.value("dit_512_20step_base_ms", p.dit_512_20step_base_ms);
    p.dit_512_transient_mib = j.value("dit_512_transient_mib", p.dit_512_transient_mib);
    p.dit_1024_step_ms = j.value("dit_1024_step_ms", p.dit_1024_step_ms);
    p.vmm_full_expert_map_ms = j.value("vmm_full_expert_map_ms", p.vmm_full_expert_map_ms);
    p.cpu_pil_prep_only_ms = j.value("cpu_pil_prep_only_ms", p.cpu_pil_prep_only_ms);
    p.gpu_inplace_prep_only_ms = j.value("gpu_inplace_prep_only_ms", p.gpu_inplace_prep_only_ms);
    p.stock_loopback_turn_ms = j.value("stock_loopback_turn_ms", p.stock_loopback_turn_ms);
    p.morphserve_loopback_turn_ms = j.value("morphserve_loopback_turn_ms", p.morphserve_loopback_turn_ms);
    p.non_cow_kv_mult = j.value("non_cow_kv_mult", p.non_cow_kv_mult);
    p.turn_edit_512_kv_delta_mib = j.value("turn_edit_512_kv_delta_mib", p.turn_edit_512_kv_delta_mib);
    p.kv_mib_per_token = j.value("kv_mib_per_token", p.kv_mib_per_token);
    p.stock_dit_512_ms_per_kv_tok = j.value("stock_dit_512_ms_per_kv_tok", p.stock_dit_512_ms_per_kv_tok);
    p.morphserve_dit_512_ms_per_kv_tok = j.value("morphserve_dit_512_ms_per_kv_tok", p.morphserve_dit_512_ms_per_kv_tok);
    p.ar_prefill_ms = j.value("ar_prefill_ms", p.ar_prefill_ms);
    p.ar_unloaded_tpot_p50_ms = j.value("ar_unloaded_tpot_p50_ms", p.ar_unloaded_tpot_p50_ms);
    p.ar_unloaded_tpot_p99_ms = j.value("ar_unloaded_tpot_p99_ms", p.ar_unloaded_tpot_p99_ms);
    p.ar_batched2_step_p50_ms = j.value("ar_batched2_step_p50_ms", p.ar_batched2_step_p50_ms);
    p.vmm_ar_kv_step_overhead_ms = j.value("vmm_ar_kv_step_overhead_ms", p.vmm_ar_kv_step_overhead_ms);
    p.dit_512_cfg_step_mean_ms = j.value("dit_512_cfg_step_mean_ms", p.dit_512_cfg_step_mean_ms);
    p.dit_512_layer_p50_ms = j.value("dit_512_layer_p50_ms", p.dit_512_layer_p50_ms);
    p.m4_adaptive_chunk_layers = j.value("m4_adaptive_chunk_layers", p.m4_adaptive_chunk_layers);
    p.m4_colocated_ar_tpot_p50_ms = j.value("m4_colocated_ar_tpot_p50_ms", p.m4_colocated_ar_tpot_p50_ms);
    p.step_preempt_ar_tpot_p50_ms = j.value("step_preempt_ar_tpot_p50_ms", p.step_preempt_ar_tpot_p50_ms);
    p.paced_h2d_per_gpu_gbps = j.value("paced_h2d_per_gpu_gbps", p.paced_h2d_per_gpu_gbps);
    p.unpaced_4gpu_half_h2d_per_gpu_gbps = j.value(
        "unpaced_4gpu_half_h2d_per_gpu_gbps", p.unpaced_4gpu_half_h2d_per_gpu_gbps);
    p.nccl_intra_half_kv_gbps = j.value("nccl_intra_half_kv_gbps", p.nccl_intra_half_kv_gbps);
    p.kairos_2x8_intra_half_gbps = j.value("kairos_2x8_intra_half_gbps", p.kairos_2x8_intra_half_gbps);
    p.kairos_2x8_cross_socket_gbps = j.value("kairos_2x8_cross_socket_gbps", p.kairos_2x8_cross_socket_gbps);
    p.kairos_2x8_inter_node_gbps = j.value("kairos_2x8_inter_node_gbps", p.kairos_2x8_inter_node_gbps);
    if (j.contains("provenance_files") && j["provenance_files"].is_object()) {
        p.provenance_files = j["provenance_files"].get<std::map<std::string, std::string>>();
    }
    return p;
}

ClusterPhysicsProfile ClusterPhysicsProfile::from_phase01_results(const std::string& results_dir) {
    ClusterPhysicsProfile p;
    std::filesystem::path root(results_dir);
    std::error_code ec;
    if (results_dir.empty() || !std::filesystem::is_directory(root, ec)) {
        return p;
    }

    auto load = [&](const std::string& key, const std::vector<std::string>& cands) -> std::optional<nlohmann::json> {
        auto found = first_existing_file(root, cands);
        if (!found) {
            return std::nullopt;
        }
        std::string rel_str;
        std::error_code rel_ec;
        auto rel_p = std::filesystem::relative(*found, root, rel_ec);
        rel_str = rel_ec ? found->generic_string() : rel_p.generic_string();
        const std::string sha = sha256_hex_file(found->string()).substr(0, 16);
        p.provenance_files[key] = rel_str + "@sha256:" + sha;
        return read_json_file(*found);
    };

    if (auto j0a = load(
            "job0a_vmm",
            {"hardware-profile/vmm-microbench/vmm-bench.json",
             "phase0-job0a-v2/vmm-bench.json",
             "phase0-job0a-r2/vmm-bench.json"})) {
        if (j0a->contains("h2d_stream_gbps") && (*j0a)["h2d_stream_gbps"].is_object()) {
            p.vmm_solo_h2d_stream_gbps = (*j0a)["h2d_stream_gbps"].value("mean", p.vmm_solo_h2d_stream_gbps);
        }
        if (j0a->contains("granularity_sweep") && (*j0a)["granularity_sweep"].is_array()) {
            for (const auto& row : (*j0a)["granularity_sweep"]) {
                if (row.value("handle_mib", 0) == 64 && row.contains("vae_4gib_acquire_ms")) {
                    p.vmm_64m_acquire_4gib_ms = row["vae_4gib_acquire_ms"].value("mean", p.vmm_64m_acquire_4gib_ms);
                    break;
                }
            }
        }
    }

    if (auto j0c = load(
            "job0c_topo",
            {"hardware-profile/topo-8gpu/topo-8gpu-report.json", "phase0-job0c/topo-8gpu-report.json"})) {
        if (j0c->contains("numa_h2d") && (*j0c)["numa_h2d"].is_object()) {
            const auto& nh = (*j0c)["numa_h2d"];
            if (nh.contains("single_gpu_h2d_gbps") && nh["single_gpu_h2d_gbps"].is_object()) {
                double sum = 0.0;
                int cnt = 0;
                for (const char* k : {"gpu_0", "gpu_1", "gpu_2", "gpu_3"}) {
                    if (nh["single_gpu_h2d_gbps"].contains(k) && nh["single_gpu_h2d_gbps"][k].is_object()) {
                        sum += nh["single_gpu_h2d_gbps"][k].value("numa_0", 8.75);
                        ++cnt;
                    }
                }
                if (cnt > 0) {
                    p.pcie_solo_h2d_gbps = round_n(sum / static_cast<double>(cnt), 3);
                }
            }
            if (nh.contains("concurrent_h2d") && nh["concurrent_h2d"].is_object()) {
                const auto& ch = nh["concurrent_h2d"];
                if (ch.contains("socket0_4gpu_remote_numa1")) {
                    p.pcie_half0_agg_h2d_gbps =
                        ch["socket0_4gpu_remote_numa1"].value("aggregate_gbps", p.pcie_half0_agg_h2d_gbps);
                }
                if (ch.contains("socket1_4gpu_local_numa1")) {
                    p.pcie_half1_agg_h2d_gbps =
                        ch["socket1_4gpu_local_numa1"].value("aggregate_gbps", p.pcie_half1_agg_h2d_gbps);
                }
            }
        }
        if (j0c->contains("nccl_p2p") && (*j0c)["nccl_p2p"].contains("pairs") &&
            (*j0c)["nccl_p2p"]["pairs"].contains("0->2")) {
            p.pcie_intra_half_p2p_gbps =
                (*j0c)["nccl_p2p"]["pairs"]["0->2"].value("gbps_p50", p.pcie_intra_half_p2p_gbps);
        }
    }

    if (auto j0d = load(
            "job0d_motivation",
            {"hardware-profile/motivation-baseline/motivation-report.json",
             "phase0-job0d-r2/motivation-report.json",
             "phase0-job0d/motivation-report.json"})) {
        if (j0d->contains("single_gpu") && (*j0d)["single_gpu"].contains("vae_batch_and_res_stress")) {
            const auto& vae = (*j0d)["single_gpu"]["vae_batch_and_res_stress"];
            if (vae.contains("512_b1")) {
                const auto& v512 = vae["512_b1"];
                p.vae_512_ms = v512.value("elapsed_ms", p.vae_512_ms);
                const double sp512 = v512.value("spike_delta_mib", 896.25);
                p.scoped_vae_512_spike_mib = std::ceil(sp512 / 64.0) * 64.0;
                p.unscoped_vae_512_reserved_mib =
                    round_n(v512.value("peak_reserved_mib", 33114.0) - v512.value("before_alloc_mib", 31091.22), 2);
            }
            if (vae.contains("1024_b1")) {
                const auto& v1024 = vae["1024_b1"];
                p.vae_1024_ms = v1024.value("elapsed_ms", p.vae_1024_ms);
                const double sp1024 = v1024.value("spike_delta_mib", 3585.0);
                p.scoped_vae_1024_spike_mib = std::ceil(sp1024 / 64.0) * 64.0;
                p.unscoped_vae_1024_reserved_mib =
                    round_n(v1024.value("peak_reserved_mib", 37600.0) - v1024.value("before_alloc_mib", 31091.22), 2);
            }
        }
        if (j0d->contains("two_stage") && (*j0d)["two_stage"].is_object()) {
            const auto& ts = (*j0d)["two_stage"];
            if (ts.contains("turns") && ts["turns"].is_array() && !ts["turns"].empty() &&
                ts.contains("aggregate_kv_transfer") && ts["aggregate_kv_transfer"].is_object()) {
                const auto& turns_ts = ts["turns"];
                const auto& agg_ts = ts["aggregate_kv_transfer"];
                const auto t0_kv = turns_ts[0].value("kv_transfer", nlohmann::json::object());
                const double t0_mib = t0_kv.value("total_transfer_mib", 0.99);
                const double base_ms = (t0_kv.value("total_transfer_seconds", 0.001) +
                                        t0_kv.value("total_serialize_seconds", 0.0042)) *
                                       1000.0;
                const double tot_mib = agg_ts.value("total_transfer_mib", 1964.32);
                const double tot_ms = (agg_ts.value("total_transfer_seconds", 3.403) +
                                       agg_ts.value("total_serialize_seconds", 4.8797)) *
                                      1000.0;
                const double slope =
                    (tot_ms - static_cast<double>(turns_ts.size()) * base_ms) / std::max(1.0, tot_mib - t0_mib);
                p.twostage_shm_base_ms = round_n(base_ms, 3);
                p.twostage_shm_ms_per_mib = round_n(std::max(0.1, slope), 4);
            }
        }
    }

    if (auto m1 = load(
            "module1_vmm",
            {"osdi27-v2/stage1-gate/module1/module1-report.json",
             "osdi27-v1/stage1-gate/module1/module1-report.json",
             "refactor-20261122/module1-report.json",
             "phase1-job1a-r3/module1-report.json"})) {
        if (m1->contains("mem_monolithic_loaded")) {
            const auto& ml = (*m1)["mem_monolithic_loaded"];
            p.monolithic_loaded_driver_mib = ml.value("driver_used_mib", p.monolithic_loaded_driver_mib);
            if (ml.contains("driver_used_mib") && ml.contains("driver_free_mib")) {
                p.driver_total_hbm_40g_mib =
                    round_n(ml.value("driver_used_mib", 28776.81) + ml.value("driver_free_mib", 11665.56), 2);
            }
        }
        if (m1->contains("mem_after_offload")) {
            p.offloaded_base_driver_mib =
                (*m1)["mem_after_offload"].value("driver_used_mib", p.offloaded_base_driver_mib);
        }
        if (m1->contains("offload_info") && (*m1)["offload_info"].is_object()) {
            const auto& oi = (*m1)["offload_info"];
            p.num_expert_layers = oi.value("num_layers", p.num_expert_layers);
            p.layer_stride_mib = oi.value("layer_stride_mib", p.layer_stride_mib);
            p.packed_expert_data_mib = oi.value("packed_data_mib", p.packed_expert_data_mib);
        }
        if (m1->contains("test1_alpha_sweep") && (*m1)["test1_alpha_sweep"].is_array()) {
            for (const auto& row : (*m1)["test1_alpha_sweep"]) {
                const double a = row.value("alpha", -1.0);
                if (std::abs(a - 1.0) < 1e-6) {
                    p.dit_512_20step_base_ms = row.value("gen_total_ms", p.dit_512_20step_base_ms);
                    p.dit_512_transient_mib = row.value("transient_peak_mib", p.dit_512_transient_mib);
                } else if (std::abs(a - 0.0) < 1e-6) {
                    p.vmm_full_expert_map_ms = row.value("expert_map_delta_ms", p.vmm_full_expert_map_ms);
                }
            }
        }
        if (m1->contains("test1b_1024_overlap_sweep") && (*m1)["test1b_1024_overlap_sweep"].is_array()) {
            for (const auto& row : (*m1)["test1b_1024_overlap_sweep"]) {
                if (std::abs(row.value("alpha", -1.0) - 1.0) < 1e-6) {
                    p.dit_1024_step_ms = round_n(row.value("gen_total_ms", p.dit_1024_step_ms * 10.0) / 10.0, 3);
                    break;
                }
            }
        }
    }

    if (auto m3 = load(
            "module3_cow_loopback",
            {"osdi27-v2/stage1-gate/module3/module3-report.json",
             "osdi27-v1/stage1-gate/module3/module3-report.json",
             "morphserve-p2-m21gate-20261121-r4/module3-report.json",
             "phase1-job1b-r7/module3-report.json"})) {
        if (m3->contains("test2_gpu_inplace_loopback") && (*m3)["test2_gpu_inplace_loopback"].is_object()) {
            const auto& t2 = (*m3)["test2_gpu_inplace_loopback"];
            p.cpu_pil_prep_only_ms = t2.value("cpu_png_pil_prep_only_ms", p.cpu_pil_prep_only_ms);
            p.gpu_inplace_prep_only_ms = t2.value("gpu_inplace_prep_only_ms", p.gpu_inplace_prep_only_ms);
        }
        if (m3->contains("test3_4turn_interleaved_e2e") &&
            (*m3)["test3_4turn_interleaved_e2e"].contains("turns") &&
            (*m3)["test3_4turn_interleaved_e2e"]["turns"].is_array() &&
            (*m3)["test3_4turn_interleaved_e2e"]["turns"].size() >= 4) {
            const auto& t3_turns = (*m3)["test3_4turn_interleaved_e2e"]["turns"];
            const auto& t0 = t3_turns[0];
            const auto& t1 = t3_turns[1];
            const auto& t3 = t3_turns[3];
            double sum_s = 0.0;
            double sum_m = 0.0;
            const size_t n_tail = t3_turns.size() - 1;
            for (size_t idx = 1; idx < t3_turns.size(); ++idx) {
                sum_s += t3_turns[idx].value("stock_loopback_ms", 0.0);
                sum_m += t3_turns[idx].value("morphserve_loopback_ms", 0.0);
            }
            p.stock_loopback_turn_ms = round_n(sum_s / static_cast<double>(n_tail), 3);
            p.morphserve_loopback_turn_ms = round_n(sum_m / static_cast<double>(n_tail), 3);
            const double ms_kv3 = std::max(1e-3, t3.value("morphserve_kv_hbm_mib", 1.0));
            const double st_kv3 = t3.value("stock_kv_hbm_mib", 2.0);
            p.non_cow_kv_mult = round_n(st_kv3 / ms_kv3, 4);
            p.turn_edit_512_kv_delta_mib =
                round_n(t1.value("morphserve_kv_hbm_mib", 0.0) - t0.value("morphserve_kv_hbm_mib", 0.0), 2);
            const int dtok = std::max(1, t3.value("main_seq_len", 1) - t0.value("main_seq_len", 0));
            p.kv_mib_per_token = round_n(
                (t3.value("morphserve_kv_hbm_mib", 0.0) - t0.value("morphserve_kv_hbm_mib", 0.0)) /
                    static_cast<double>(dtok),
                5);
            const double base_gen = t0.value("stock_gen_ms", 0.0);
            p.stock_dit_512_ms_per_kv_tok =
                round_n((t3.value("stock_gen_ms", 0.0) - base_gen) / static_cast<double>(dtok), 6);
            p.morphserve_dit_512_ms_per_kv_tok = round_n(
                std::max(0.0, (t3.value("morphserve_gen_ms", 0.0) - base_gen) / static_cast<double>(dtok)), 6);
        }
    }

    if (auto m4 = load(
            "module4_scheduler",
            {"osdi27-v2/stage1-gate/module4/module4-report.json",
             "osdi27-v1/stage1-gate/module4/module4-report.json",
             "phase1-job1c-r3/module4-report.json"})) {
        if (m4->contains("solo_baselines") && (*m4)["solo_baselines"].is_object()) {
            const auto& sb = (*m4)["solo_baselines"];
            p.ar_unloaded_tpot_p50_ms = sb.value("unloaded_ar_tpot_p50_ms", p.ar_unloaded_tpot_p50_ms);
            p.ar_unloaded_tpot_p99_ms = sb.value("unloaded_ar_tpot_p99_ms", p.ar_unloaded_tpot_p99_ms);
            if (sb.contains("vmm_ar_kv_step_overhead_ms") && sb["vmm_ar_kv_step_overhead_ms"].is_number()) {
                p.vmm_ar_kv_step_overhead_ms = std::max(0.0, sb["vmm_ar_kv_step_overhead_ms"].get<double>());
            }
            p.dit_512_cfg_step_mean_ms = sb.value("dit_cfg_step_mean_ms", p.dit_512_cfg_step_mean_ms);
            p.dit_512_layer_p50_ms = sb.value("dit_layer_p50_ms", p.dit_512_layer_p50_ms);
        }
        if (m4->contains("test0_batched_ar_scaling") && (*m4)["test0_batched_ar_scaling"].is_array()) {
            for (const auto& row : (*m4)["test0_batched_ar_scaling"]) {
                if (row.value("batch_size", 0) == 2) {
                    p.ar_batched2_step_p50_ms = row.value("step_p50_ms", p.ar_batched2_step_p50_ms);
                    break;
                }
            }
        }
        if (m4->contains("test1_policy_comparison") && (*m4)["test1_policy_comparison"].is_array()) {
            for (const auto& row : (*m4)["test1_policy_comparison"]) {
                const std::string pname = row.value("policy", "");
                if (pname == "morphserve_layer_chunk_20x") {
                    p.m4_adaptive_chunk_layers = row.value("adaptive_chunk_layers", p.m4_adaptive_chunk_layers);
                    p.m4_colocated_ar_tpot_p50_ms = row.value("ar_tpot_p50_ms", p.m4_colocated_ar_tpot_p50_ms);
                } else if (pname == "step_preemptive") {
                    p.step_preempt_ar_tpot_p50_ms = row.value("ar_tpot_p50_ms", p.step_preempt_ar_tpot_p50_ms);
                }
            }
        }
    }

    if (auto m2 = load(
            "module2_router",
            {"hardware-profile/module2-pcie-numa/module2-report.json", "phase1-job1d-r4/module2-report.json"})) {
        if (m2->contains("test1_shared_numa_memfd_and_h2d_pacer") &&
            (*m2)["test1_shared_numa_memfd_and_h2d_pacer"].is_object()) {
            const auto& t1 = (*m2)["test1_shared_numa_memfd_and_h2d_pacer"];
            p.paced_h2d_per_gpu_gbps = t1.value("paced_dual_half_per_gpu_gbps", p.paced_h2d_per_gpu_gbps);
            p.unpaced_4gpu_half_h2d_per_gpu_gbps =
                t1.value("unpaced_4gpu_half0_per_gpu_gbps", p.unpaced_4gpu_half_h2d_per_gpu_gbps);
        }
        if (m2->contains("test2_bagel_dualpath_crossover") &&
            (*m2)["test2_bagel_dualpath_crossover"].contains("t3_long_interleaved_kv")) {
            const auto& t2 = (*m2)["test2_bagel_dualpath_crossover"]["t3_long_interleaved_kv"];
            if (t2.contains("path_a_bw_gbps")) {
                p.nccl_intra_half_kv_gbps = t2.value("path_a_bw_gbps", p.nccl_intra_half_kv_gbps);
            }
        }
    }

    if (auto m2x8_p = first_existing_file(
            root,
            {"hardware-profile/multinode-2x8-net/m25-multinode-2x8-report.json",
             "osdi27-v1/r3-multinode-2x8/m25-multinode-2x8-report.json",
             "morphserve-p2-m25node2x8-20261121-r2/m25-multinode-2x8-report.json"})) {
        const auto m2x8 = read_json_file(*m2x8_p);
        if (m2x8.contains("calibrated_gbps") && m2x8["calibrated_gbps"].is_object()) {
            const auto& cal = m2x8["calibrated_gbps"];
            p.kairos_2x8_intra_half_gbps = cal.value("intra_half_pcie", p.kairos_2x8_intra_half_gbps);
            p.kairos_2x8_cross_socket_gbps = cal.value("intra_node_cross_socket", p.kairos_2x8_cross_socket_gbps);
            p.kairos_2x8_inter_node_gbps = cal.value("inter_node_fabric", p.kairos_2x8_inter_node_gbps);
        }
    }

    return p;
}

nlohmann::json WorkloadScaleSpec::to_json() const {
    nlohmann::json think_j = nullptr;
    if (think_s_range_override.has_value()) {
        think_j = nlohmann::json::array({think_s_range_override->first, think_s_range_override->second});
    }
    return {
        {"name", name},
        {"weight_scale", weight_scale},
        {"compute_scale", compute_scale},
        {"kv_per_turn_scale", kv_per_turn_scale},
        {"vae_spike_scale", vae_spike_scale},
        {"extra_edit_turns", extra_edit_turns},
        {"think_s_range_override", think_j},
        {"super_pool_mib_80g", super_pool_mib_80g},
        {"fine_kv_pool_mib_80g", fine_kv_pool_mib_80g},
    };
}

WorkloadScaleSpec WorkloadScaleSpec::from_json(const nlohmann::json& j) {
    WorkloadScaleSpec ws;
    ws.name = j.value("name", ws.name);
    ws.weight_scale = j.value("weight_scale", ws.weight_scale);
    ws.compute_scale = j.value("compute_scale", ws.compute_scale);
    ws.kv_per_turn_scale = j.value("kv_per_turn_scale", ws.kv_per_turn_scale);
    ws.vae_spike_scale = j.value("vae_spike_scale", ws.vae_spike_scale);
    ws.extra_edit_turns = j.value("extra_edit_turns", ws.extra_edit_turns);
    if (j.contains("think_s_range_override") && j["think_s_range_override"].is_array() &&
        j["think_s_range_override"].size() == 2) {
        ws.think_s_range_override = std::make_pair(
            j["think_s_range_override"][0].get<double>(),
            j["think_s_range_override"][1].get<double>());
    }
    ws.super_pool_mib_80g = j.value("super_pool_mib_80g", ws.super_pool_mib_80g);
    ws.fine_kv_pool_mib_80g = j.value("fine_kv_pool_mib_80g", ws.fine_kv_pool_mib_80g);
    return ws;
}

WorkloadScaleSpec WorkloadScaleSpec::bagel_7b_standard() {
    return WorkloadScaleSpec{};
}

WorkloadScaleSpec WorkloadScaleSpec::frontier_mot_56gib() {
    WorkloadScaleSpec ws;
    ws.name = "frontier_mot_56gib";
    ws.weight_scale = 2.0;
    ws.compute_scale = 1.85;
    ws.kv_per_turn_scale = 2.0;
    ws.vae_spike_scale = 1.5;
    ws.extra_edit_turns = 1;
    ws.think_s_range_override = std::make_pair(6.0, 14.0);
    ws.super_pool_mib_80g = 28672;
    ws.fine_kv_pool_mib_80g = 512;
    return ws;
}

WorkloadScaleSpec WorkloadScaleSpec::bagel_7b_long_kv_high_concurrency() {
    WorkloadScaleSpec ws;
    ws.name = "bagel_7b_long_kv_high_concurrency";
    ws.weight_scale = 1.0;
    ws.compute_scale = 1.0;
    ws.kv_per_turn_scale = 2.8;
    ws.vae_spike_scale = 1.0;
    ws.extra_edit_turns = 6;
    ws.think_s_range_override = std::make_pair(20.0, 38.0);
    ws.super_pool_mib_80g = 28672;
    ws.fine_kv_pool_mib_80g = 1024;
    return ws;
}

ClusterTopologyProfile HardwareInterconnectSpec::to_topology(int num_gpus) const {
    return ClusterTopologyProfile::from_multinode_measurements(
        num_gpus,
        gpus_per_node,
        intra_half_p2p_gbps,
        inter_socket_p2p_gbps,
        inter_node_rdma_gbps,
        solo_h2d_gbps,
        agg_h2d_gbps,
        name,
        (intra_half_p2p_gbps > 50.0) ? 0.35 : 3.0,
        (inter_node_rdma_gbps > 20.0) ? 1.8 : 8.0);
}

nlohmann::json HardwareInterconnectSpec::to_json() const {
    return {
        {"name", name},
        {"gpu_hbm_gib", gpu_hbm_gib},
        {"compute_speedup_x", compute_speedup_x},
        {"solo_h2d_gbps", solo_h2d_gbps},
        {"agg_h2d_gbps", agg_h2d_gbps},
        {"intra_half_p2p_gbps", intra_half_p2p_gbps},
        {"inter_socket_p2p_gbps", inter_socket_p2p_gbps},
        {"inter_node_rdma_gbps", inter_node_rdma_gbps},
        {"gpus_per_node", gpus_per_node},
    };
}

HardwareInterconnectSpec HardwareInterconnectSpec::from_json(const nlohmann::json& j) {
    HardwareInterconnectSpec hw;
    hw.name = j.value("name", hw.name);
    hw.gpu_hbm_gib = j.value("gpu_hbm_gib", hw.gpu_hbm_gib);
    hw.compute_speedup_x = j.value("compute_speedup_x", hw.compute_speedup_x);
    hw.solo_h2d_gbps = j.value("solo_h2d_gbps", hw.solo_h2d_gbps);
    hw.agg_h2d_gbps = j.value("agg_h2d_gbps", hw.agg_h2d_gbps);
    hw.intra_half_p2p_gbps = j.value("intra_half_p2p_gbps", hw.intra_half_p2p_gbps);
    hw.inter_socket_p2p_gbps = j.value("inter_socket_p2p_gbps", hw.inter_socket_p2p_gbps);
    hw.inter_node_rdma_gbps = j.value("inter_node_rdma_gbps", hw.inter_node_rdma_gbps);
    hw.gpus_per_node = j.value("gpus_per_node", hw.gpus_per_node);
    return hw;
}

std::vector<HardwareInterconnectSpec> HardwareInterconnectSpec::presets(
    const ClusterPhysicsProfile* physics) {
    const ClusterPhysicsProfile default_ph{};
    const auto& ph = physics ? *physics : default_ph;
    return {
        HardwareInterconnectSpec{
            "A100_PCIe_Gen4_RDMA100G",
            40.0,
            1.0,
            ph.pcie_solo_h2d_gbps,
            ph.pcie_half0_agg_h2d_gbps,
            ph.pcie_intra_half_p2p_gbps,
            ph.pcie_cross_half_p2p_gbps,
            11.5,
            8},
        HardwareInterconnectSpec{
            "A100_SXM4_NVLink600G_RDMA200G",
            40.0,
            1.08,
            22.9,
            24.5,
            265.0,
            265.0,
            23.0,
            8},
        HardwareInterconnectSpec{
            "H100_SXM5_NVLink900G_RDMA400G",
            80.0,
            2.25,
            52.0,
            56.0,
            410.0,
            410.0,
            46.0,
            8},
    };
}

}  // namespace morphserve::control
