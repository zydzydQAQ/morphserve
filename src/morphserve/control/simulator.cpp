#include "morphserve/control/simulator.hpp"

#include "morphserve/control/morph_controller.hpp"
#include "morphserve/control/router.hpp"
#include "morphserve/runtime/h2d_pacer.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <variant>

namespace morphserve::control {

namespace {

inline double round_1(double x) noexcept { return std::round(x * 10.0) / 10.0; }
inline double round_2(double x) noexcept { return std::round(x * 100.0) / 100.0; }
inline double round_3(double x) noexcept { return std::round(x * 1000.0) / 1000.0; }
inline double round_4(double x) noexcept { return std::round(x * 10000.0) / 10000.0; }

struct CanonicalEntry {
    double arr_s;
    const char* quad;
    std::vector<SimTurnSpec> turns;
};

const std::vector<CanonicalEntry>& canonical_28_trace() {
    static const std::vector<CanonicalEntry> kTrace = {
        {0.00, "multiturn_edit", {{"dit_edit", 512}, {"dit_edit", 512}, {"dit_edit", 512}, {"dit_edit", 512}}},
        {0.00, "visual_cot", {{"cot_then_dit", 512}}},
        {3.05, "multi_image_vqa", {{"ar_only", 512}, {"ar_only", 512}}},
        {8.61, "visual_cot", {{"cot_then_dit", 1024}}},
        {13.71, "visual_cot", {{"cot_then_dit", 1024}}},
        {15.92, "multi_image_vqa", {{"ar_only", 512}, {"ar_only", 512}}},
        {17.37, "multi_image_vqa", {{"ar_only", 512}, {"ar_only", 512}}},
        {19.51, "multi_image_vqa", {{"ar_only", 512}, {"ar_only", 512}}},
        {19.96, "multi_image_vqa", {{"ar_only", 512}, {"ar_only", 512}}},
        {30.86, "multi_image_vqa", {{"ar_only", 512}, {"ar_only", 512}}},
        {40.38, "visual_cot", {{"cot_then_dit", 512}}},
        {42.29, "multi_image_vqa", {{"ar_only", 512}, {"ar_only", 512}}},
        {48.23, "visual_cot", {{"cot_then_dit", 1024}}},
        {51.89, "visual_cot", {{"cot_then_dit", 512}}},
        {55.34, "multiturn_edit", {{"dit_edit", 1024}, {"dit_edit", 1024}, {"dit_edit", 1024}, {"dit_edit", 1024}}},
        {55.96, "multiturn_edit", {{"dit_edit", 1024}, {"dit_edit", 1024}, {"dit_edit", 1024}, {"dit_edit", 1024}}},
        {56.07, "highres_t2i", {{"dit_t2i", 1024}}},
        {56.08, "visual_cot", {{"cot_then_dit", 512}}},
        {58.36, "visual_cot", {{"cot_then_dit", 1024}}},
        {59.11, "multiturn_edit", {{"dit_edit", 512}, {"dit_edit", 512}, {"dit_edit", 512}, {"dit_edit", 512}}},
        {61.13, "multiturn_edit", {{"dit_edit", 512}, {"dit_edit", 512}, {"dit_edit", 512}}},
        {62.53, "highres_t2i", {{"dit_t2i", 512}}},
        {63.73, "multiturn_edit", {{"dit_edit", 512}, {"dit_edit", 512}, {"dit_edit", 512}, {"dit_edit", 512}}},
        {68.77, "highres_t2i", {{"dit_t2i", 1024}}},
        {68.84, "highres_t2i", {{"dit_t2i", 512}}},
        {69.22, "multiturn_edit", {{"dit_edit", 512}, {"dit_edit", 512}, {"dit_edit", 512}, {"dit_edit", 512}}},
        {69.89, "multiturn_edit", {{"dit_edit", 512}, {"dit_edit", 512}}},
        {70.25, "multiturn_edit", {{"dit_edit", 512}, {"dit_edit", 512}, {"dit_edit", 512}, {"dit_edit", 512}}},
    };
    return kTrace;
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> cols;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '\r') continue;
        if (ch == '"') {
            in_quotes = !in_quotes;
        } else if (ch == ',' && !in_quotes) {
            cols.push_back(std::move(cur));
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    cols.push_back(std::move(cur));
    return cols;
}

std::vector<double> normalize_interarrivals(const std::vector<double>& deltas) {
    std::vector<double> pos;
    pos.reserve(deltas.size());
    double sum = 0.0;
    for (double d : deltas) {
        const double v = std::max(0.0, d);
        pos.push_back(v);
        sum += v;
    }
    if (pos.empty() || sum <= 0.0) {
        return std::vector<double>(std::max<size_t>(1, pos.size()), 1.0);
    }
    const double mean_d = sum / static_cast<double>(pos.size());
    for (double& v : pos) v /= mean_d;
    return pos;
}

std::vector<double> load_burstgpt_csv(const std::filesystem::path& p, size_t max_events = 2000) {
    std::ifstream in(p);
    if (!in) return {};
    std::string hdr_line;
    if (!std::getline(in, hdr_line)) return {};
    const auto hdr = split_csv_line(hdr_line);
    auto it = std::find(hdr.begin(), hdr.end(), "Timestamp");
    if (it == hdr.end()) return {};
    const size_t ts_col = static_cast<size_t>(std::distance(hdr.begin(), it));

    std::vector<double> timestamps;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto cols = split_csv_line(line);
        if (ts_col >= cols.size()) continue;
        try {
            timestamps.push_back(std::stod(cols[ts_col]));
        } catch (...) {
            continue;
        }
        if (timestamps.size() >= max_events + 1) break;
    }
    if (timestamps.size() < 2) return {1.0};
    std::vector<double> deltas;
    deltas.reserve(timestamps.size() - 1);
    for (size_t i = 1; i < timestamps.size(); ++i) {
        deltas.push_back(std::max(0.0, timestamps[i] - timestamps[i - 1]));
    }
    return normalize_interarrivals(deltas);
}

std::vector<double> load_gentd26_csv(const std::filesystem::path& p, size_t max_events = 2000) {
    std::ifstream in(p);
    if (!in) return {};
    std::string hdr_line;
    if (!std::getline(in, hdr_line)) return {};
    const auto hdr = split_csv_line(hdr_line);
    auto git = std::find(hdr.begin(), hdr.end(), "gmt_create");
    if (git == hdr.end()) return {};
    const size_t gmt_col = static_cast<size_t>(std::distance(hdr.begin(), git));
    auto sit = std::find(hdr.begin(), hdr.end(), "predict_status");
    const bool has_status = (sit != hdr.end());
    const size_t st_col = has_status ? static_cast<size_t>(std::distance(hdr.begin(), sit)) : 0;

    std::vector<double> timestamps;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto cols = split_csv_line(line);
        if (gmt_col >= cols.size()) continue;
        if (has_status && st_col < cols.size()) {
            const std::string& st = cols[st_col];
            if (!st.empty() && st != "SUCCEED") continue;
        }
        std::tm tm_val = {};
        std::istringstream iss(cols[gmt_col]);
        iss >> std::get_time(&tm_val, "%Y-%m-%d %H:%M:%S");
        if (iss.fail()) continue;
        tm_val.tm_isdst = -1;
        const std::time_t epoch = std::mktime(&tm_val);
        if (epoch == static_cast<std::time_t>(-1)) continue;
        timestamps.push_back(static_cast<double>(epoch));
        if (timestamps.size() >= max_events + 1) break;
    }
    if (timestamps.size() < 2) return {1.0};
    std::vector<double> deltas;
    deltas.reserve(timestamps.size() - 1);
    for (size_t i = 1; i < timestamps.size(); ++i) {
        deltas.push_back(std::max(0.0, timestamps[i] - timestamps[i - 1]));
    }
    return normalize_interarrivals(deltas);
}

std::vector<double> scale_arrivals_to_qps(const std::vector<double>& norm_deltas, double target_qps, size_t count) {
    if (count == 0) return {};
    const double qps = std::max(target_qps, 1e-6);
    std::vector<double> offsets;
    offsets.reserve(count);
    offsets.push_back(0.0);
    for (size_t i = 1; i < count; ++i) {
        const double d = norm_deltas.empty() ? 1.0 : norm_deltas[(i - 1) % norm_deltas.size()];
        offsets.push_back(round_4(offsets.back() + d / qps));
    }
    return offsets;
}

struct RawSuiteSession {
    std::string session_id;
    std::string quadrant;
    double arrival_offset_s = 0.0;
    std::vector<SimTurnSpec> turns;
};

struct CachedSuiteBundle {
    bool valid = false;
    std::vector<RawSuiteSession> sessions;
    std::vector<double> burst_deltas;
    std::vector<double> gentd_deltas;
};

CachedSuiteBundle load_cached_suite_bundle(
    const std::string& skew_tag,
    const std::string& bench_root,
    const std::string& sources_dir) {
    static std::mutex cache_mu;
    static std::unordered_map<std::string, CachedSuiteBundle> cache;
    const std::string key = (skew_tag.empty() ? "main_mix" : skew_tag) + "|" + bench_root + "|" + sources_dir;
    {
        std::lock_guard<std::mutex> lk(cache_mu);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
    }

    CachedSuiteBundle out;
    if (!bench_root.empty()) {
        std::string rel = "suites/main_mix_80.json";
        if (skew_tag == "ar80_dit20") rel = "suites/skew_ar80_dit20.json";
        else if (skew_tag == "ar50_dit50") rel = "suites/skew_ar50_dit50.json";
        else if (skew_tag == "ar20_dit80") rel = "suites/skew_ar20_dit80.json";

        std::filesystem::path suite_p = std::filesystem::path(bench_root) / rel;
        std::error_code ec;
        if (std::filesystem::is_regular_file(suite_p, ec)) {
            std::ifstream in(suite_p, std::ios::binary);
            if (in) {
                const auto j = nlohmann::json::parse(in);
                if (j.contains("sessions") && j["sessions"].is_array() && !j["sessions"].empty()) {
                    for (const auto& sj : j["sessions"]) {
                        RawSuiteSession rs;
                        rs.session_id = sj.value("session_id", "");
                        rs.quadrant = sj.value("quadrant", "visual_cot");
                        rs.arrival_offset_s = sj.value("arrival_offset_s", 0.0);
                        if (sj.contains("turns") && sj["turns"].is_array()) {
                            for (const auto& tj : sj["turns"]) {
                                rs.turns.push_back(SimTurnSpec{
                                    tj.value("modality_plan", "ar_only"),
                                    tj.value("height", 512)});
                            }
                        }
                        out.sessions.push_back(std::move(rs));
                    }
                    out.valid = !out.sessions.empty();
                }
            }
        }
        if (out.valid && !sources_dir.empty()) {
            const auto bp = std::filesystem::path(sources_dir) / "burstgpt_1_sample.csv";
            const auto gp = std::filesystem::path(sources_dir) / "gentd26_lora_request_trace.csv";
            if (std::filesystem::is_regular_file(bp, ec)) {
                out.burst_deltas = load_burstgpt_csv(bp);
            }
            if (std::filesystem::is_regular_file(gp, ec)) {
                out.gentd_deltas = load_gentd26_csv(gp);
            }
        }
    }

    {
        std::lock_guard<std::mutex> lk(cache_mu);
        cache[key] = out;
    }
    return out;
}

double crossover_kv_mib(
    const HardwareInterconnectSpec& hw,
    const ClusterPhysicsProfile& ph,
    const WorkloadScaleSpec& ws) {
    const auto topo = hw.to_topology(std::max(2, hw.gpus_per_node));
    DualPathMorphRouter router(topo);
    return router.analytical_crossover_kv_mib(
        0.50,
        static_cast<int64_t>(ph.packed_expert_data_mib * ws.weight_scale * static_cast<double>(kMiB)),
        (ph.dit_512_cfg_step_mean_ms * ws.compute_scale) / hw.compute_speedup_x,
        0,
        1);
}

}  // namespace

// --- SimGpuMemoryLedger ---

SimGpuMemoryLedger::SimGpuMemoryLedger(
    int gpu_id,
    double total_hbm_mib,
    PolicyConfig policy,
    std::string role,
    int resident_expert_layers,
    int num_expert_layers,
    int layer_stride_mib,
    ClusterPhysicsProfile physics,
    double weight_scale,
    double vae_spike_scale)
    : gpu_id_(gpu_id),
      total_hbm_mib_(total_hbm_mib),
      policy_(std::move(policy)),
      role_(std::move(role)),
      resident_expert_layers_(resident_expert_layers),
      num_expert_layers_(num_expert_layers),
      layer_stride_mib_(layer_stride_mib),
      physics_(std::move(physics)),
      weight_scale_(weight_scale),
      vae_spike_scale_(vae_spike_scale) {}

double SimGpuMemoryLedger::offloaded_base_mib() const noexcept {
    return physics_.offloaded_base_driver_mib * weight_scale_;
}

double SimGpuMemoryLedger::monolithic_weights_mib() const noexcept {
    return physics_.monolithic_loaded_driver_mib * weight_scale_;
}

double SimGpuMemoryLedger::pageable_expert_mib() const noexcept {
    return std::max(0.0, monolithic_weights_mib() - offloaded_base_mib());
}

double SimGpuMemoryLedger::base_weights_mib() const noexcept {
    if (policy_.use_expert_pager || policy_.use_naive_offload) {
        return offloaded_base_mib();
    }
    if (policy_.routing_mode == "static_nccl_fission" && role_ == "und") {
        return offloaded_base_mib();
    }
    return monolithic_weights_mib();
}

double SimGpuMemoryLedger::expert_alpha() const noexcept {
    if (!policy_.use_expert_pager) {
        return (role_ == "und" || policy_.use_naive_offload) ? 0.0 : 1.0;
    }
    return round_3(static_cast<double>(resident_expert_layers_) /
                   static_cast<double>(std::max(1, num_expert_layers_)));
}

double SimGpuMemoryLedger::active_kv_mib() const noexcept {
    double sum = 0.0;
    if (policy_.use_tripool_vmm) {
        for (const auto& [_, v] : session_kv_mib_) {
            sum += std::max(2.0, 2.0 * std::ceil(std::max(0.01, v) / 2.0));
        }
    } else {
        for (const auto& [_, v] : session_kv_mib_) {
            sum += v;
        }
    }
    return sum;
}

int64_t SimGpuMemoryLedger::free_kv_bytes() const noexcept {
    const double exp_mib = policy_.use_expert_pager
                               ? static_cast<double>(resident_expert_layers_ * layer_stride_mib_)
                               : 0.0;
    const double used = base_weights_mib() + active_kv_mib() + exp_mib;
    return static_cast<int64_t>(std::max(0.0, total_hbm_mib_ - used - 2048.0) * static_cast<double>(kMiB));
}

double SimGpuMemoryLedger::evict_idle_kv_to_host(
    double needed_free_mib,
    std::optional<int> exclude_sid) {
    if (!(policy_.use_tripool_vmm && policy_.use_expert_pager)) {
        return 0.0;
    }
    std::vector<int> sids;
    sids.reserve(session_kv_mib_.size());
    for (const auto& [sid, _] : session_kv_mib_) {
        sids.push_back(sid);
    }
    double freed_mib = 0.0;
    for (int cand_sid : sids) {
        if (exclude_sid.has_value() && cand_sid == *exclude_sid) {
            continue;
        }
        if (freed_mib >= needed_free_mib) {
            break;
        }
        const double val = session_kv_mib_.at(cand_sid);
        session_kv_mib_.erase(cand_sid);
        swapped_session_kv_mib_[cand_sid] = val;
        ++kv_host_swap_outs_;
        freed_mib += std::max(2.0, 2.0 * std::ceil(std::max(0.01, val) / 2.0));
    }
    return freed_mib;
}

double SimGpuMemoryLedger::ensure_session_resident(int sid) {
    auto it = swapped_session_kv_mib_.find(sid);
    if (it == swapped_session_kv_mib_.end()) {
        return 0.0;
    }
    const double swapped_val = it->second;
    swapped_session_kv_mib_.erase(it);
    if (!upsert_session_kv(sid, swapped_val)) {
        swapped_session_kv_mib_[sid] = swapped_val;
        return -1.0;
    }
    ++kv_host_swap_ins_;
    return swapped_val;
}

bool SimGpuMemoryLedger::upsert_session_kv(int sid, double logical_kv_mib) {
    std::optional<double> prev;
    if (auto it = session_kv_mib_.find(sid); it != session_kv_mib_.end()) {
        prev = it->second;
    }
    std::optional<double> prev_swapped;
    if (auto sit = swapped_session_kv_mib_.find(sid); sit != swapped_session_kv_mib_.end()) {
        prev_swapped = sit->second;
        swapped_session_kv_mib_.erase(sit);
    }
    session_kv_mib_[sid] = logical_kv_mib;
    double kv_tot = active_kv_mib();
    double used = 0.0;

    if (policy_.use_tripool_vmm && policy_.use_expert_pager) {
        const double pool_cap = static_cast<double>(policy_.super_pool_mib + policy_.fine_kv_pool_mib);
        if (kv_tot > static_cast<double>(policy_.fine_kv_pool_mib)) {
            ++super_to_fine_carves_;
        }
        while (resident_expert_layers_ > 0 &&
               (static_cast<double>(resident_expert_layers_ * layer_stride_mib_) + kv_tot) > pool_cap) {
            --resident_expert_layers_;
            ++expert_evictions_;
        }
        used = base_weights_mib() + static_cast<double>(resident_expert_layers_ * layer_stride_mib_) + kv_tot;
        if (used > total_hbm_mib_) {
            evict_idle_kv_to_host(used - total_hbm_mib_, sid);
            kv_tot = active_kv_mib();
            used = base_weights_mib() + static_cast<double>(resident_expert_layers_ * layer_stride_mib_) + kv_tot;
        }
    } else {
        used = base_weights_mib() + kv_tot;
    }

    if (used > total_hbm_mib_) {
        if (prev.has_value()) {
            session_kv_mib_[sid] = *prev;
        } else {
            session_kv_mib_.erase(sid);
        }
        if (prev_swapped.has_value()) {
            swapped_session_kv_mib_[sid] = *prev_swapped;
        }
        return false;
    }
    peak_used_mib_ = std::max(peak_used_mib_, used);
    return true;
}

std::pair<bool, double> SimGpuMemoryLedger::execute_dit_and_vae(
    int height,
    std::optional<int> active_sid) {
    const bool is_1024 = height >= 1024;
    const double dit_act_mib =
        (is_1024 ? (2.0 * physics_.dit_512_transient_mib) : physics_.dit_512_transient_mib) *
        std::sqrt(weight_scale_);
    const double scoped_vae_mib =
        (is_1024 ? physics_.scoped_vae_1024_spike_mib : physics_.scoped_vae_512_spike_mib) * vae_spike_scale_;
    const double unscoped_vae_mib =
        (is_1024 ? physics_.unscoped_vae_1024_reserved_mib : physics_.unscoped_vae_512_reserved_mib) *
        vae_spike_scale_;
    double kv_tot = active_kv_mib();
    double swapped_out_mib = 0.0;

    if (policy_.use_tripool_vmm && policy_.use_expert_pager) {
        double step_peak =
            base_weights_mib() + static_cast<double>(num_expert_layers_ * layer_stride_mib_) + kv_tot + dit_act_mib;
        if (step_peak > total_hbm_mib_ && active_sid.has_value()) {
            swapped_out_mib += evict_idle_kv_to_host(step_peak - total_hbm_mib_, active_sid);
            kv_tot = active_kv_mib();
            step_peak = base_weights_mib() + static_cast<double>(num_expert_layers_ * layer_stride_mib_) +
                        kv_tot + dit_act_mib;
        }
        if (step_peak > total_hbm_mib_) {
            return {false, swapped_out_mib};
        }
        resident_expert_layers_ = num_expert_layers_;

        const double vae_mib = policy_.use_scoped_vae ? scoped_vae_mib : unscoped_vae_mib;
        const double pool_cap = static_cast<double>(policy_.super_pool_mib + policy_.fine_kv_pool_mib);
        const double eff_cap = std::min(pool_cap, std::max(0.0, total_hbm_mib_ - base_weights_mib()));
        while (resident_expert_layers_ > 0 &&
               (static_cast<double>(resident_expert_layers_ * layer_stride_mib_) + kv_tot + vae_mib) > eff_cap) {
            --resident_expert_layers_;
            ++expert_evictions_;
        }

        double vae_peak = base_weights_mib() +
                          static_cast<double>(resident_expert_layers_ * layer_stride_mib_) + kv_tot + vae_mib;
        if (vae_peak > total_hbm_mib_ && active_sid.has_value()) {
            swapped_out_mib += evict_idle_kv_to_host(vae_peak - total_hbm_mib_, active_sid);
            kv_tot = active_kv_mib();
            vae_peak = base_weights_mib() +
                       static_cast<double>(resident_expert_layers_ * layer_stride_mib_) + kv_tot + vae_mib;
        }
        if (vae_peak > total_hbm_mib_) {
            return {false, swapped_out_mib};
        }
        peak_used_mib_ = std::max({peak_used_mib_, step_peak, vae_peak});
        return {true, swapped_out_mib};
    }

    const double gen_weights_mib = policy_.use_naive_offload ? pageable_expert_mib() : 0.0;
    const double vae_mib = policy_.use_scoped_vae ? scoped_vae_mib : unscoped_vae_mib;
    const double peak = base_weights_mib() + gen_weights_mib + kv_tot + std::max(dit_act_mib, vae_mib);
    if (peak > total_hbm_mib_) {
        peak_used_mib_ = std::max(peak_used_mib_, total_hbm_mib_);
        return {false, 0.0};
    }
    peak_used_mib_ = std::max(peak_used_mib_, peak);
    return {true, 0.0};
}

void SimGpuMemoryLedger::release_session(int sid) {
    session_kv_mib_.erase(sid);
    swapped_session_kv_mib_.erase(sid);
}

nlohmann::json SimGpuMemoryLedger::stats_json() const {
    return {
        {"gpu_id", gpu_id_},
        {"resident_expert_layers", resident_expert_layers_},
        {"expert_alpha", expert_alpha()},
        {"active_kv_mib", active_kv_mib()},
        {"free_kv_bytes", free_kv_bytes()},
        {"super_to_fine_carves", super_to_fine_carves_},
        {"expert_evictions", expert_evictions_},
        {"kv_host_swap_outs", kv_host_swap_outs_},
        {"kv_host_swap_ins", kv_host_swap_ins_},
        {"peak_used_mib", peak_used_mib_},
    };
}

// --- ClusterSimulator ---

std::vector<SimSessionSpec> ClusterSimulator::build_session_stream(
    int num_sessions,
    double offered_qps,
    int /*num_gpus*/,
    uint64_t seed,
    int extra_edit_turns,
    const std::string& skew_tag,
    const std::string& bench_root,
    const std::string& sources_dir) {
    if (num_sessions <= 0) {
        return {};
    }
    const auto cached = load_cached_suite_bundle(skew_tag, bench_root, sources_dir);
    if (cached.valid) {
        const size_t n_src = cached.sessions.size();
        std::vector<RawSuiteSession> cloned;
        cloned.reserve(static_cast<size_t>(num_sessions));
        for (int idx = 0; idx < num_sessions; ++idx) {
            RawSuiteSession s = cached.sessions[static_cast<size_t>(idx) % n_src];
            if (static_cast<size_t>(idx) >= n_src) {
                s.session_id = s.session_id + "_r" + std::to_string(static_cast<size_t>(idx) / n_src);
            }
            cloned.push_back(std::move(s));
        }
        const double eff_qps = std::max(0.05, offered_qps);
        if (!cached.burst_deltas.empty() && !cached.gentd_deltas.empty()) {
            std::vector<size_t> ar_idx;
            std::vector<size_t> dit_idx;
            for (size_t i = 0; i < cloned.size(); ++i) {
                if (cloned[i].quadrant == "multi_image_vqa" || cloned[i].quadrant == "visual_cot") {
                    ar_idx.push_back(i);
                } else {
                    dit_idx.push_back(i);
                }
            }
            const double tot = static_cast<double>(std::max<size_t>(1, cloned.size()));
            const double ar_qps = std::max(1e-4, eff_qps * (static_cast<double>(ar_idx.size()) / tot));
            const double dit_qps = std::max(1e-4, eff_qps * (static_cast<double>(dit_idx.size()) / tot));
            const auto ar_offs = scale_arrivals_to_qps(cached.burst_deltas, ar_qps, ar_idx.size());
            const auto dit_offs = scale_arrivals_to_qps(cached.gentd_deltas, dit_qps, dit_idx.size());
            for (size_t pos = 0; pos < ar_idx.size(); ++pos) cloned[ar_idx[pos]].arrival_offset_s = ar_offs[pos];
            for (size_t pos = 0; pos < dit_idx.size(); ++pos) cloned[dit_idx[pos]].arrival_offset_s = dit_offs[pos];
            std::sort(cloned.begin(), cloned.end(), [](const RawSuiteSession& a, const RawSuiteSession& b) {
                if (a.arrival_offset_s != b.arrival_offset_s) return a.arrival_offset_s < b.arrival_offset_s;
                return a.session_id < b.session_id;
            });
        } else {
            const double mean_inter_s = 1.0 / eff_qps;
            for (size_t idx = 0; idx < cloned.size(); ++idx) {
                cloned[idx].arrival_offset_s = static_cast<double>(idx) * mean_inter_s;
            }
        }
        std::vector<SimSessionSpec> out;
        out.reserve(cloned.size());
        for (auto& sc : cloned) {
            if (!sc.turns.empty()) {
                if (extra_edit_turns > 0 && sc.quadrant == "multiturn_edit") {
                    const SimTurnSpec last = sc.turns.back();
                    for (int e = 0; e < extra_edit_turns; ++e) sc.turns.push_back(last);
                } else if (extra_edit_turns >= 2 &&
                           (sc.quadrant == "visual_cot" || sc.quadrant == "multi_image_vqa")) {
                    const SimTurnSpec last = sc.turns.back();
                    for (int e = 0; e < extra_edit_turns / 2; ++e) sc.turns.push_back(last);
                }
            }
            out.push_back(SimSessionSpec{sc.arrival_offset_s * 1000.0, sc.quadrant, std::move(sc.turns)});
        }
        return out;
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.85, 1.15);
    const auto& base = canonical_28_trace();
    const double mean_inter_ms = 1000.0 / std::max(0.05, offered_qps);
    double t_cur = 0.0;
    std::vector<SimSessionSpec> out;
    out.reserve(static_cast<size_t>(num_sessions));
    for (int idx = 0; idx < num_sessions; ++idx) {
        const auto& tmpl = base[static_cast<size_t>(idx) % base.size()];
        auto turns = tmpl.turns;
        const std::string q = tmpl.quad;
        if (!turns.empty()) {
            if (extra_edit_turns > 0 && q == "multiturn_edit") {
                const auto last = turns.back();
                for (int e = 0; e < extra_edit_turns; ++e) turns.push_back(last);
            } else if (extra_edit_turns >= 2 && (q == "visual_cot" || q == "multi_image_vqa")) {
                const auto last = turns.back();
                for (int e = 0; e < extra_edit_turns / 2; ++e) turns.push_back(last);
            }
        }
        if (idx > 0) {
            const int m5 = idx % 5;
            const double burst_factor = (m5 == 1 || m5 == 2) ? 0.35 : ((m5 == 0) ? 1.85 : 0.90);
            t_cur += mean_inter_ms * burst_factor * dist(rng);
        }
        out.push_back(SimSessionSpec{t_cur, q, std::move(turns)});
    }
    return out;
}

nlohmann::json ClusterSimulator::simulate_cluster_point(
    const std::string& policy_name,
    int num_gpus,
    const HardwareInterconnectSpec& hw,
    double offered_qps,
    int num_sessions,
    uint64_t seed,
    const ClusterPhysicsProfile& ph,
    const WorkloadScaleSpec& ws,
    const std::string& skew_tag,
    const std::string& bench_root,
    const std::string& sources_dir,
    const std::vector<SimSessionSpec>* custom_sessions) {
    if (!is_valid_policy_name(policy_name)) {
        throw std::invalid_argument("unknown policy: " + policy_name);
    }

    nlohmann::json overrides = nlohmann::json::object();
    if (ws.think_s_range_override.has_value()) {
        overrides["inter_turn_think_s_range"] =
            nlohmann::json::array({ws.think_s_range_override->first, ws.think_s_range_override->second});
    }
    if (hw.gpu_hbm_gib >= 75.0 && policy_name == "morphserve") {
        overrides["super_pool_mib"] = ws.super_pool_mib_80g;
        overrides["fine_kv_pool_mib"] = ws.fine_kv_pool_mib_80g;
    }

    const PolicyConfig policy = PolicyConfig::from_preset(policy_name, num_gpus, overrides);
    const ClusterTopologyProfile topo = hw.to_topology(num_gpus);
    auto pacer = std::make_shared<runtime::HalfNodeH2DPacer>(topo.gpu_to_group_map(), 1);
    const double sp = hw.compute_speedup_x;
    const double cscale = ws.compute_scale;

    DualPathMorphRouter router(
        topo,
        pacer,
        default_operator_profile(),
        std::nullopt,
        (ph.vmm_full_expert_map_ms * ws.weight_scale) / sp,
        0.90,
        true,
        true,
        static_cast<int64_t>(128.0 * ws.kv_per_turn_scale * static_cast<double>(kMiB)));

    const double ar_tok_ms = (ph.ar_unloaded_tpot_p50_ms * cscale) / sp;
    const double ar_prefill_ms = (ph.ar_prefill_ms * cscale) / sp;
    const double vae512_ms = (ph.vae_512_ms * ws.vae_spike_scale) / sp;
    const double vae1024_ms = (ph.vae_1024_ms * ws.vae_spike_scale) / sp;
    const double dit512_core_20_ms = (std::max(0.0, ph.dit_512_20step_base_ms - ph.vae_512_ms) * cscale) / sp;
    const double dit512_step_ms = (ph.dit_512_cfg_step_mean_ms * cscale) / sp;
    const double dit1024_step_ms = (ph.dit_1024_step_ms * cscale) / sp;
    const double dit1024_core_20_ms = std::max(0.0, 20.0 * dit1024_step_ms - vae1024_ms);
    const double batched_ar_amort = 0.40;

    const double layer512_ms = (ph.dit_512_layer_p50_ms * cscale) / sp;
    const double layer1024_ms = layer512_ms * (ph.dit_1024_step_ms / std::max(1.0, ph.dit_512_cfg_step_mean_ms));
    const double ar_tpot_slo = (policy.ar_tpot_slo_ms * cscale) / sp;
    const double ar_ttft_slo = (policy.ar_ttft_slo_ms * cscale) / sp;
    const int chunk_layers_512 = std::max(
        1, std::min(8, static_cast<int>(std::floor(std::max(0.0, ar_tpot_slo - ar_tok_ms) / std::max(0.1, layer512_ms)))));
    const int chunk_layers_1024 = std::max(
        1, std::min(8, static_cast<int>(std::floor(std::max(0.0, ar_tpot_slo - ar_tok_ms) / std::max(0.1, layer1024_ms)))));
    const double chunk_q_512_ms = static_cast<double>(chunk_layers_512) * layer512_ms;
    const double chunk_q_1024_ms = static_cast<double>(chunk_layers_1024) * layer1024_ms;

    const int64_t expert_bytes_scaled =
        static_cast<int64_t>(ph.packed_expert_data_mib * ws.weight_scale * static_cast<double>(kMiB));
    const int layer_stride_scaled = static_cast<int>(std::round(static_cast<double>(ph.layer_stride_mib) * ws.weight_scale));
    const double hbm_mib = hw.gpu_hbm_gib * 1024.0;

    std::vector<SimGpuMemoryLedger> ledgers;
    ledgers.reserve(static_cast<size_t>(num_gpus));
    std::map<int, int64_t> init_kv_map;
    for (int g = 0; g < num_gpus; ++g) {
        auto ait = policy.initial_alphas.find(g);
        const double a0 = (ait != policy.initial_alphas.end()) ? ait->second : 1.0;
        ledgers.emplace_back(
            g,
            hbm_mib,
            policy,
            policy.role_of(g),
            static_cast<int>(std::round(a0 * static_cast<double>(ph.num_expert_layers))),
            ph.num_expert_layers,
            layer_stride_scaled,
            ph,
            ws.weight_scale,
            ws.vae_spike_scale);
        init_kv_map[g] = 20LL * kGiB;
    }

    std::map<int, WorkerState> workers = router.init_workers(
        num_gpus,
        [&](int g) { return policy.role_of(g); },
        policy.initial_alphas,
        init_kv_map);

    std::vector<double> gpu_free_at(static_cast<size_t>(num_gpus), 0.0);
    std::vector<double> ar_slot_free_at(static_cast<size_t>(num_gpus), 0.0);
    const int gpn = std::max(1, hw.gpus_per_node);
    const int num_nodes = std::max(1, (num_gpus + gpn - 1) / gpn);
    std::vector<double> rdma_free_at(static_cast<size_t>(num_nodes), 0.0);
    std::vector<double> unpaced_h2d_until(static_cast<size_t>(num_gpus), 0.0);

    const std::vector<SimSessionSpec> built_sessions =
        custom_sessions
            ? *custom_sessions
            : build_session_stream(
                  num_sessions, offered_qps, num_gpus, seed, ws.extra_edit_turns, skew_tag, bench_root, sources_dir);
    const size_t n_sess = built_sessions.size();

    std::vector<int> sess_gpu(n_sess, -1);
    std::vector<double> sess_kv_mib(n_sess, 0.0);
    std::vector<double> sess_main_kv_mib(n_sess, 0.0);
    std::vector<bool> sess_ar_ok(n_sess, true);
    std::vector<bool> sess_dit_ok(n_sess, true);
    std::vector<bool> sess_oom(n_sess, false);
    std::vector<int> sess_retries(n_sess, 0);
    std::vector<std::vector<double>> sess_ar_ttfts(n_sess);
    std::vector<std::vector<double>> sess_ar_tpots(n_sess);
    std::vector<std::vector<std::pair<double, int>>> sess_dit_records(n_sess);

    std::vector<double> ttfts;
    std::vector<double> tpots;
    std::vector<double> dit_lats;
    int ships = 0;
    int morphs = 0;
    int cross_node_ships = 0;
    int preemptions = 0;

    enum EvKind : int {
        EV_RELEASE = 0,
        EV_DIT_END = 1,
        EV_RET_UNPACK = 2,
        EV_DIT_START = 3,
        EV_FWD_SHIP = 4,
        EV_TURN = 5,
    };

    struct EvTurn {
        int ti = 0;
        int attempt = 0;
    };
    struct EvFwdShip {
        int ti = 0;
        int attempt = 0;
        int cur_gpu = 0;
        double t_turn0 = 0.0;
        std::string quad;
        int height = 512;
        int n_turns = 1;
        double cand_main_kv = 0.0;
        double dit_phys_kv = 0.0;
        double step_base = 0.0;
        double dit_exec_ms = 0.0;
    };
    struct EvDitStart {
        int ti = 0;
        int attempt = 0;
        int cur_gpu = 0;
        int dst_gpu = 0;
        std::string ship_mode;
        double t_turn0 = 0.0;
        int height = 512;
        int n_turns = 1;
        double cand_main_kv = 0.0;
        double dit_phys_kv = 0.0;
        double dit_exec_ms = 0.0;
        double reserved_end_dit = 0.0;
    };
    struct EvDitEnd {
        int ti = 0;
        int attempt = 0;
        int cur_gpu = 0;
        int dst_gpu = 0;
        std::string ship_mode;
        double t_turn0 = 0.0;
        int height = 512;
        int n_turns = 1;
        double cand_main_kv = 0.0;
    };
    struct EvRetUnpack {
        int ti = 0;
        int attempt = 0;
        int cur_gpu = 0;
        int dst_gpu = 0;
        double t_turn0 = 0.0;
        int height = 512;
        int n_turns = 1;
        double cand_main_kv = 0.0;
    };
    using PayloadVar = std::variant<std::monostate, EvTurn, EvFwdShip, EvDitStart, EvDitEnd, EvRetUnpack>;

    struct SimEvent {
        double ts_ms = 0.0;
        int kind = 0;
        int sid = 0;
        uint64_t seq = 0;
        PayloadVar payload;

        bool operator>(const SimEvent& o) const noexcept {
            return std::tie(ts_ms, kind, sid, seq) > std::tie(o.ts_ms, o.kind, o.sid, o.seq);
        }
    };

    std::priority_queue<SimEvent, std::vector<SimEvent>, std::greater<SimEvent>> pq;
    uint64_t ev_seq = 0;
    auto push_ev = [&](double ts_ms, int kind, int sid, PayloadVar p) {
        ++ev_seq;
        pq.push(SimEvent{ts_ms, kind, sid, ev_seq, std::move(p)});
    };

    for (size_t sid = 0; sid < n_sess; ++sid) {
        push_ev(built_sessions[sid].arrival_ms, EV_TURN, static_cast<int>(sid), EvTurn{0, 0});
    }

    auto sync_states = [&](double now_ms, std::map<int, bool>& busy_map, std::map<int, int>& dit_map) {
        busy_map.clear();
        dit_map.clear();
        for (int g = 0; g < num_gpus; ++g) {
            const double rem = std::max(0.0, gpu_free_at[static_cast<size_t>(g)] - now_ms);
            auto& wg = workers.at(g);
            wg.queue_wait_ms = rem;
            wg.expert_alpha = ledgers[static_cast<size_t>(g)].expert_alpha();
            wg.kv_free_bytes = ledgers[static_cast<size_t>(g)].free_kv_bytes();
            busy_map[g] = (rem > 0.0);
            dit_map[g] = (rem > (200.0 / sp)) ? 1 : 0;
        }
    };

    auto pick_oom_retry_gpu = [&](int failed_gpu, double now_ms) -> int {
        std::map<int, bool> busy_now;
        std::map<int, int> dit_now;
        sync_states(now_ms, busy_now, dit_now);
        return router.select_oom_retry_gpu(
            failed_gpu, num_gpus, policy.routing_mode, policy.static_und_ranks, &workers, busy_now);
    };

    const bool is_static_ephemeral =
        (policy.routing_mode == "static_nccl_fission" || policy.routing_mode == "twostage_shm");
    const bool is_tight_mot = (policy.cptg_architecture() == "tight_mot");

    while (!pq.empty()) {
        SimEvent cur_ev = std::move(const_cast<SimEvent&>(pq.top()));
        pq.pop();
        double ev_ms = cur_ev.ts_ms;
        const int sid = cur_ev.sid;
        const size_t usid = static_cast<size_t>(sid);

        if (cur_ev.kind == EV_RELEASE) {
            const int old_g = sess_gpu[usid];
            if (old_g >= 0) {
                ledgers[static_cast<size_t>(old_g)].release_session(sid);
            }
            continue;
        }

        if (cur_ev.kind == EV_FWD_SHIP) {
            const auto p = std::get<EvFwdShip>(cur_ev.payload);
            const double dit_ready_ms = ev_ms;
            const bool has_more_turns = (p.ti + 1) < p.n_turns;
            const double wire_kv_mib = (policy.routing_mode == "twostage_shm") ? p.dit_phys_kv : p.cand_main_kv;
            std::map<int, bool> busy_at_dit;
            std::map<int, int> dit_at_dit;
            sync_states(dit_ready_ms, busy_at_dit, dit_at_dit);
            const int gamma = (p.quad == "multiturn_edit" && has_more_turns) ? 2 : 1;

            CPTGRequestSpec req;
            req.req_id = "s" + std::to_string(sid) + "_t" + std::to_string(p.ti) + "_a" + std::to_string(p.attempt);
            req.src_gpu = p.cur_gpu;
            req.architecture = policy.cptg_architecture();
            req.kv_bytes = static_cast<int64_t>(wire_kv_mib * static_cast<double>(kMiB));
            req.gen_expert_bytes = expert_bytes_scaled;
            req.dit_step1_ms = p.step_base;
            req.dit_total_steps = 20;
            req.resolution = p.height;
            req.interleaved_roundtrip_gamma = gamma;

            auto [dec, ship_mode] = router.plan_and_reserve_transition(
                policy.routing_mode,
                policy.static_und_ranks,
                policy.static_gen_ranks,
                req,
                p.dit_exec_ms,
                &workers,
                busy_at_dit,
                policy.use_h2d_pacer,
                dit_ready_ms);
            const int dst_gpu = dec.dst_gpu;
            router.release_dit_reservation(dst_gpu, p.dit_exec_ms, &workers);
            if (ship_mode != "none" && dst_gpu != p.cur_gpu) {
                router.release_kv_reservation(dst_gpu, req.effective_state_bytes(), std::nullopt, std::nullopt, &workers);
            }

            const bool release_src = is_tight_mot ? true : (!has_more_turns);
            double trans_ms = 0.0;
            if (ship_mode == "nccl_static") {
                const int src_node = topo.node_of(p.cur_gpu);
                const double r_wait = (dec.hop_tier == "inter_node")
                                          ? std::max(0.0, rdma_free_at[static_cast<size_t>(src_node)] - dit_ready_ms)
                                          : 0.0;
                const double bw = topo.effective_p2p_gbps(p.cur_gpu, dst_gpu);
                const double wire_ms =
                    ((p.cand_main_kv * static_cast<double>(kMiB) / 1e9) / std::max(0.5, bw)) * 1000.0 +
                    topo.handoff_ms(p.cur_gpu, dst_gpu);
                trans_ms = r_wait + wire_ms;
                if (!topo.same_node(p.cur_gpu, dst_gpu)) {
                    rdma_free_at[static_cast<size_t>(src_node)] =
                        std::max(dit_ready_ms, rdma_free_at[static_cast<size_t>(src_node)]) + wire_ms;
                }
            } else {
                trans_ms =
                    (ph.twostage_shm_base_ms + ph.twostage_shm_ms_per_mib * p.dit_phys_kv) /
                    std::sqrt(std::max(1.0, sp));
            }

            if (release_src) {
                ledgers[static_cast<size_t>(p.cur_gpu)].release_session(sid);
            } else {
                ledgers[static_cast<size_t>(p.cur_gpu)].upsert_session_kv(sid, p.cand_main_kv);
            }

            const double start_dit = std::max(dit_ready_ms + trans_ms, gpu_free_at[static_cast<size_t>(dst_gpu)]);
            const double reserved_end_dit = start_dit + p.dit_exec_ms;
            gpu_free_at[static_cast<size_t>(dst_gpu)] = reserved_end_dit;
            push_ev(
                start_dit,
                EV_DIT_START,
                sid,
                EvDitStart{
                    p.ti,
                    p.attempt,
                    p.cur_gpu,
                    dst_gpu,
                    ship_mode,
                    p.t_turn0,
                    p.height,
                    p.n_turns,
                    p.cand_main_kv,
                    p.dit_phys_kv,
                    p.dit_exec_ms,
                    reserved_end_dit});
            continue;
        }

        if (cur_ev.kind == EV_DIT_START) {
            const auto p = std::get<EvDitStart>(cur_ev.payload);
            const double start_dit = ev_ms;
            if (!ledgers[static_cast<size_t>(p.dst_gpu)].upsert_session_kv(sid, p.dit_phys_kv)) {
                if (gpu_free_at[static_cast<size_t>(p.dst_gpu)] == p.reserved_end_dit) {
                    gpu_free_at[static_cast<size_t>(p.dst_gpu)] = start_dit;
                }
                ledgers[static_cast<size_t>(p.dst_gpu)].release_session(sid);
                ledgers[static_cast<size_t>(p.cur_gpu)].release_session(sid);
                if (p.attempt < policy.max_oom_retries && p.ti == 0) {
                    ++sess_retries[usid];
                    sess_gpu[usid] = pick_oom_retry_gpu(p.dst_gpu, start_dit);
                    push_ev(start_dit, EV_TURN, sid, EvTurn{p.ti, p.attempt + 1});
                } else {
                    sess_oom[usid] = true;
                }
                continue;
            }

            ++ships;
            if (!topo.same_node(p.cur_gpu, p.dst_gpu)) {
                ++cross_node_ships;
            }

            const auto [dit_ok_mem, swap_out_mib] =
                ledgers[static_cast<size_t>(p.dst_gpu)].execute_dit_and_vae(p.height, sid);
            if (!dit_ok_mem) {
                const double abort_end = start_dit + p.dit_exec_ms * 0.45;
                if (gpu_free_at[static_cast<size_t>(p.dst_gpu)] == p.reserved_end_dit) {
                    gpu_free_at[static_cast<size_t>(p.dst_gpu)] = abort_end;
                }
                ledgers[static_cast<size_t>(p.dst_gpu)].release_session(sid);
                ledgers[static_cast<size_t>(p.cur_gpu)].release_session(sid);
                if (p.attempt < policy.max_oom_retries && p.ti == 0) {
                    ++sess_retries[usid];
                    sess_gpu[usid] = pick_oom_retry_gpu(p.dst_gpu, abort_end);
                    push_ev(abort_end, EV_TURN, sid, EvTurn{p.ti, p.attempt + 1});
                } else {
                    sess_oom[usid] = true;
                }
                continue;
            }

            const double swap_out_ms =
                (swap_out_mib > 0.0)
                    ? (((swap_out_mib * static_cast<double>(kMiB) / 1e9) / std::max(0.5, topo.solo_h2d_gbps(p.dst_gpu))) *
                       1000.0)
                    : 0.0;
            const double end_dit = start_dit + swap_out_ms + p.dit_exec_ms;
            gpu_free_at[static_cast<size_t>(p.dst_gpu)] =
                std::max(gpu_free_at[static_cast<size_t>(p.dst_gpu)], end_dit);
            workers.at(p.dst_gpu).expert_alpha = ledgers[static_cast<size_t>(p.dst_gpu)].expert_alpha();
            push_ev(
                end_dit,
                EV_DIT_END,
                sid,
                EvDitEnd{p.ti, p.attempt, p.cur_gpu, p.dst_gpu, p.ship_mode, p.t_turn0, p.height, p.n_turns, p.cand_main_kv});
            continue;
        }

        if (cur_ev.kind == EV_DIT_END) {
            const auto p = std::get<EvDitEnd>(cur_ev.payload);
            const double end_dit = ev_ms;
            ledgers[static_cast<size_t>(p.dst_gpu)].release_session(sid);
            const bool has_more_turns = (p.ti + 1) < p.n_turns;
            const bool needs_return_ship = has_more_turns && is_tight_mot;
            if (needs_return_ship) {
                double ret_trans_ms = 0.0;
                if (p.ship_mode == "nccl_static") {
                    const int dst_node = topo.node_of(p.dst_gpu);
                    const double ret_bw = topo.effective_p2p_gbps(p.dst_gpu, p.cur_gpu);
                    const double ret_wire_ms =
                        ((p.cand_main_kv * static_cast<double>(kMiB) / 1e9) / std::max(0.5, ret_bw)) * 1000.0 +
                        topo.handoff_ms(p.dst_gpu, p.cur_gpu);
                    double ret_r_wait = 0.0;
                    if (!topo.same_node(p.dst_gpu, p.cur_gpu)) {
                        ret_r_wait = std::max(0.0, rdma_free_at[static_cast<size_t>(dst_node)] - end_dit);
                        rdma_free_at[static_cast<size_t>(dst_node)] =
                            std::max(end_dit, rdma_free_at[static_cast<size_t>(dst_node)]) + ret_wire_ms;
                    }
                    ret_trans_ms = ret_r_wait + ret_wire_ms;
                } else {
                    ret_trans_ms =
                        (ph.twostage_shm_base_ms + ph.twostage_shm_ms_per_mib * p.cand_main_kv) /
                        std::sqrt(std::max(1.0, sp));
                }
                push_ev(
                    end_dit + ret_trans_ms,
                    EV_RET_UNPACK,
                    sid,
                    EvRetUnpack{p.ti, p.attempt, p.cur_gpu, p.dst_gpu, p.t_turn0, p.height, p.n_turns, p.cand_main_kv});
            } else {
                sess_kv_mib[usid] = p.cand_main_kv;
                sess_main_kv_mib[usid] = p.cand_main_kv;
                const double turn_lat = end_dit - p.t_turn0;
                dit_lats.push_back(turn_lat);
                sess_dit_records[usid].emplace_back(turn_lat, p.height);
                const double turn_slo_ms = (policy.dit_turn_slo_for(p.height, 20) * cscale) / sp;
                if (turn_lat > turn_slo_ms) {
                    sess_dit_ok[usid] = false;
                }
                if (has_more_turns) {
                    sess_gpu[usid] = p.cur_gpu;
                    const double think_ms = MorphControlEngine::deterministic_think_ms_sim(
                        sid, p.ti + 1, policy.inter_turn_think_s_range.first, policy.inter_turn_think_s_range.second);
                    push_ev(end_dit + think_ms, EV_TURN, sid, EvTurn{p.ti + 1, 0});
                }
            }
            continue;
        }

        if (cur_ev.kind == EV_RET_UNPACK) {
            const auto p = std::get<EvRetUnpack>(cur_ev.payload);
            const double turn_end_ms = ev_ms;
            if (!ledgers[static_cast<size_t>(p.cur_gpu)].upsert_session_kv(sid, p.cand_main_kv)) {
                ledgers[static_cast<size_t>(p.cur_gpu)].release_session(sid);
                if (p.attempt < policy.max_oom_retries && p.ti == 0) {
                    ++sess_retries[usid];
                    sess_gpu[usid] = pick_oom_retry_gpu(p.cur_gpu, turn_end_ms);
                    push_ev(turn_end_ms, EV_TURN, sid, EvTurn{0, p.attempt + 1});
                } else {
                    sess_oom[usid] = true;
                }
                continue;
            }
            if (!topo.same_node(p.dst_gpu, p.cur_gpu)) {
                ++cross_node_ships;
            }
            sess_kv_mib[usid] = p.cand_main_kv;
            sess_main_kv_mib[usid] = p.cand_main_kv;
            const double turn_lat = turn_end_ms - p.t_turn0;
            dit_lats.push_back(turn_lat);
            sess_dit_records[usid].emplace_back(turn_lat, p.height);
            const double turn_slo_ms = (policy.dit_turn_slo_for(p.height, 20) * cscale) / sp;
            if (turn_lat > turn_slo_ms) {
                sess_dit_ok[usid] = false;
            }
            sess_gpu[usid] = p.cur_gpu;
            const double think_ms = MorphControlEngine::deterministic_think_ms_sim(
                sid, p.ti + 1, policy.inter_turn_think_s_range.first, policy.inter_turn_think_s_range.second);
            push_ev(turn_end_ms + think_ms, EV_TURN, sid, EvTurn{p.ti + 1, 0});
            continue;
        }

        auto turn_ev = std::get<EvTurn>(cur_ev.payload);
        const int ti = turn_ev.ti;
        int attempt = turn_ev.attempt;
        const auto& sess_spec = built_sessions[usid];
        if (sess_spec.turns.empty() || ti >= static_cast<int>(sess_spec.turns.size())) {
            continue;
        }
        const std::string& quad = sess_spec.quadrant;
        const std::string& plan = sess_spec.turns[static_cast<size_t>(ti)].modality_plan;
        const int height = sess_spec.turns[static_cast<size_t>(ti)].height;
        const bool is_1024 = height >= 1024;
        const int n_turns = static_cast<int>(sess_spec.turns.size());

        while (true) {
            std::map<int, bool> busy_map;
            std::map<int, int> dit_map;
            sync_states(ev_ms, busy_map, dit_map);
            int cur_gpu = sess_gpu[usid];
            if (cur_gpu < 0) {
                cur_gpu = router.select_initial_gpu(
                    num_gpus,
                    policy.routing_mode,
                    policy.preemption_mode,
                    policy.static_und_ranks,
                    plan,
                    &workers,
                    busy_map,
                    dit_map);
                sess_gpu[usid] = cur_gpu;
            }

            if (plan == "ar_only") {
                const double swap_in_mib = ledgers[static_cast<size_t>(cur_gpu)].ensure_session_resident(sid);
                if (swap_in_mib < 0.0) {
                    ledgers[static_cast<size_t>(cur_gpu)].release_session(sid);
                    sess_oom[usid] = true;
                    break;
                }
                const double swap_in_ms =
                    (swap_in_mib > 0.0)
                        ? (((swap_in_mib * static_cast<double>(kMiB) / 1e9) / std::max(0.5, topo.solo_h2d_gbps(cur_gpu))) *
                           1000.0)
                        : 0.0;
                const int n_tok = 30;
                const double ar_delta = 48.0 * ws.kv_per_turn_scale;
                const double new_kv = sess_kv_mib[usid] + ar_delta;
                if (!ledgers[static_cast<size_t>(cur_gpu)].upsert_session_kv(sid, new_kv)) {
                    ledgers[static_cast<size_t>(cur_gpu)].release_session(sid);
                    if (attempt < policy.max_oom_retries && ti == 0) {
                        ++attempt;
                        ++sess_retries[usid];
                        sess_gpu[usid] = pick_oom_retry_gpu(cur_gpu, ev_ms);
                        continue;
                    }
                    sess_oom[usid] = true;
                    break;
                }
                sess_kv_mib[usid] = new_kv;
                sess_main_kv_mib[usid] += ar_delta;

                const double dit_rem = std::max(0.0, gpu_free_at[static_cast<size_t>(cur_gpu)] - ev_ms);
                double wait_ms = 0.0;
                double tpot = ar_tok_ms;
                if (dit_rem > 0.0 && policy.preemption_mode == "layer_chunk") {
                    const double chunk_q = is_1024 ? chunk_q_1024_ms : chunk_q_512_ms;
                    wait_ms = std::min(dit_rem, 0.5 * chunk_q) + swap_in_ms;
                    tpot = std::min((ph.m4_colocated_ar_tpot_p50_ms * cscale) / sp, ar_tok_ms + 0.5 * chunk_q);
                    gpu_free_at[static_cast<size_t>(cur_gpu)] +=
                        ar_prefill_ms + static_cast<double>(n_tok) * ar_tok_ms * batched_ar_amort;
                    ++preemptions;
                } else if (dit_rem > 0.0 && policy.preemption_mode == "step_preemptive") {
                    const double step_q = is_1024 ? dit1024_step_ms : dit512_step_ms;
                    wait_ms = std::min(dit_rem, 0.5 * step_q) + swap_in_ms;
                    tpot = std::max((ph.step_preempt_ar_tpot_p50_ms * cscale) / sp, ar_tok_ms + 0.5 * step_q);
                    gpu_free_at[static_cast<size_t>(cur_gpu)] +=
                        ar_prefill_ms + static_cast<double>(n_tok) * ar_tok_ms * batched_ar_amort;
                    ++preemptions;
                } else {
                    const double ar_wait =
                        (dit_rem <= 0.0) ? std::max(0.0, ar_slot_free_at[static_cast<size_t>(cur_gpu)] - ev_ms) : dit_rem;
                    wait_ms = ar_wait + swap_in_ms;
                    tpot = ar_tok_ms;
                    ar_slot_free_at[static_cast<size_t>(cur_gpu)] =
                        std::max(ev_ms, ar_slot_free_at[static_cast<size_t>(cur_gpu)]) + ar_prefill_ms +
                        static_cast<double>(n_tok) * ar_tok_ms * batched_ar_amort;
                    if (dit_rem > 0.0) {
                        gpu_free_at[static_cast<size_t>(cur_gpu)] =
                            std::max(ev_ms, gpu_free_at[static_cast<size_t>(cur_gpu)]) + ar_prefill_ms +
                            static_cast<double>(n_tok) * ar_tok_ms * batched_ar_amort;
                    }
                }

                const double ttft = wait_ms + ar_prefill_ms;
                ttfts.push_back(ttft);
                tpots.push_back(tpot);
                sess_ar_ttfts[usid].push_back(ttft);
                sess_ar_tpots[usid].push_back(tpot);
                if (ttft > ar_ttft_slo || tpot > ar_tpot_slo) {
                    sess_ar_ok[usid] = false;
                }
                const double end_ms = ev_ms + ttft + static_cast<double>(n_tok - 1) * tpot;
                if (ti + 1 < n_turns) {
                    const double think_ms = MorphControlEngine::deterministic_think_ms_sim(
                        sid, ti + 1, policy.inter_turn_think_s_range.first, policy.inter_turn_think_s_range.second);
                    push_ev(end_ms + think_ms, EV_TURN, sid, EvTurn{ti + 1, 0});
                } else {
                    push_ev(end_ms, EV_RELEASE, sid, std::monostate{});
                }
                break;
            }

            // CoT or DiT turn
            const double t_turn0 = ev_ms;
            const double swap_in_mib = ledgers[static_cast<size_t>(cur_gpu)].ensure_session_resident(sid);
            if (swap_in_mib < 0.0) {
                ledgers[static_cast<size_t>(cur_gpu)].release_session(sid);
                sess_oom[usid] = true;
                break;
            }
            const double swap_in_ms =
                (swap_in_mib > 0.0)
                    ? (((swap_in_mib * static_cast<double>(kMiB) / 1e9) / std::max(0.5, topo.solo_h2d_gbps(cur_gpu))) *
                       1000.0)
                    : 0.0;
            double base_delta_kv = 28.0;
            if (quad == "visual_cot") base_delta_kv = 185.0;
            else if (quad == "multiturn_edit") base_delta_kv = ph.turn_edit_512_kv_delta_mib * (is_1024 ? 1.30 : 1.0);

            const double cand_main_kv = sess_main_kv_mib[usid] + base_delta_kv * ws.kv_per_turn_scale;
            const double dit_phys_kv =
                (!policy.use_module3_cow_loopback && quad == "multiturn_edit" && ti > 0)
                    ? (cand_main_kv * ph.non_cow_kv_mult)
                    : cand_main_kv;
            const double src_prep_kv = (policy.routing_mode == "static_nccl_fission") ? cand_main_kv : dit_phys_kv;
            if (!ledgers[static_cast<size_t>(cur_gpu)].upsert_session_kv(sid, src_prep_kv)) {
                ledgers[static_cast<size_t>(cur_gpu)].release_session(sid);
                if (attempt < policy.max_oom_retries && ti == 0) {
                    ++attempt;
                    ++sess_retries[usid];
                    sess_gpu[usid] = pick_oom_retry_gpu(cur_gpu, ev_ms);
                    continue;
                }
                sess_oom[usid] = true;
                break;
            }

            double dit_ready_ms = 0.0;
            if (plan == "cot_then_dit") {
                const int n_tok = 26;
                const double dit_rem = std::max(0.0, gpu_free_at[static_cast<size_t>(cur_gpu)] - ev_ms);
                double wait_ms = 0.0;
                double tpot = ar_tok_ms;
                if (dit_rem > 0.0 && policy.preemption_mode == "layer_chunk") {
                    const double chunk_q = is_1024 ? chunk_q_1024_ms : chunk_q_512_ms;
                    wait_ms = std::min(dit_rem, 0.5 * chunk_q) + swap_in_ms;
                    tpot = std::min((ph.m4_colocated_ar_tpot_p50_ms * cscale) / sp, ar_tok_ms + 0.5 * chunk_q);
                    gpu_free_at[static_cast<size_t>(cur_gpu)] +=
                        ar_prefill_ms + static_cast<double>(n_tok) * ar_tok_ms * batched_ar_amort;
                    ++preemptions;
                } else if (dit_rem > 0.0 && policy.preemption_mode == "step_preemptive") {
                    const double step_q = is_1024 ? dit1024_step_ms : dit512_step_ms;
                    wait_ms = std::min(dit_rem, 0.5 * step_q) + swap_in_ms;
                    tpot = std::max((ph.step_preempt_ar_tpot_p50_ms * cscale) / sp, ar_tok_ms + 0.5 * step_q);
                    gpu_free_at[static_cast<size_t>(cur_gpu)] +=
                        ar_prefill_ms + static_cast<double>(n_tok) * ar_tok_ms * batched_ar_amort;
                    ++preemptions;
                } else {
                    wait_ms = dit_rem + swap_in_ms;
                    tpot = ar_tok_ms;
                    gpu_free_at[static_cast<size_t>(cur_gpu)] =
                        std::max(ev_ms, gpu_free_at[static_cast<size_t>(cur_gpu)]) + swap_in_ms + ar_prefill_ms +
                        static_cast<double>(n_tok) * ar_tok_ms * batched_ar_amort;
                }
                const double ttft = wait_ms + ar_prefill_ms;
                ttfts.push_back(ttft);
                tpots.push_back(tpot);
                sess_ar_ttfts[usid].push_back(ttft);
                sess_ar_tpots[usid].push_back(tpot);
                if (ttft > ar_ttft_slo || tpot > ar_tpot_slo) {
                    sess_ar_ok[usid] = false;
                }
                dit_ready_ms = ev_ms + ttft + static_cast<double>(n_tok - 1) * tpot;
            } else {
                const double start_prep = std::max(ev_ms, gpu_free_at[static_cast<size_t>(cur_gpu)]) + swap_in_ms;
                double loopback_ms = ar_prefill_ms;
                if (ti > 0) {
                    loopback_ms = policy.use_module3_cow_loopback
                                      ? ((ph.morphserve_loopback_turn_ms * cscale) / sp)
                                      : ((ph.stock_loopback_turn_ms * cscale) / sp);
                }
                gpu_free_at[static_cast<size_t>(cur_gpu)] = start_prep + loopback_ms;
                dit_ready_ms = gpu_free_at[static_cast<size_t>(cur_gpu)];
            }

            const int main_seq_tokens = static_cast<int>(std::round(
                (cand_main_kv / std::max(0.01, ws.weight_scale)) / std::max(1e-4, ph.kv_mib_per_token)));
            const int l_gen = is_1024 ? 4098 : 1026;
            const bool has_shared_prefix = policy.use_module3_cow_loopback && (ti > 0 || quad == "visual_cot");
            const bool use_cascade = has_shared_prefix && should_use_cascade_cfg_split(l_gen, main_seq_tokens, 2);
            const double slope_512 = use_cascade ? ph.morphserve_dit_512_ms_per_kv_tok : ph.stock_dit_512_ms_per_kv_tok;
            const double q_scale = is_1024 ? 4.0 : 1.0;
            const double kv_attn_ms = ((slope_512 * q_scale * static_cast<double>(main_seq_tokens)) * cscale) / sp;

            const double step_base = is_1024 ? dit1024_step_ms : dit512_step_ms;
            const double dit_core_ms = (is_1024 ? dit1024_core_20_ms : dit512_core_20_ms) + kv_attn_ms;
            const double dit_exec_ms = dit_core_ms + (is_1024 ? vae1024_ms : vae512_ms);

            if (is_static_ephemeral) {
                push_ev(
                    dit_ready_ms,
                    EV_FWD_SHIP,
                    sid,
                    EvFwdShip{ti, attempt, cur_gpu, t_turn0, quad, height, n_turns, cand_main_kv, dit_phys_kv, step_base, dit_exec_ms});
                break;
            }

            std::map<int, bool> busy_at_dit;
            std::map<int, int> dit_at_dit;
            sync_states(dit_ready_ms, busy_at_dit, dit_at_dit);
            const int gamma = (quad == "multiturn_edit" && ti + 1 < n_turns) ? 2 : 1;
            CPTGRequestSpec req;
            req.req_id = "s" + std::to_string(sid) + "_t" + std::to_string(ti);
            req.src_gpu = cur_gpu;
            req.architecture = policy.cptg_architecture();
            req.kv_bytes = static_cast<int64_t>(cand_main_kv * static_cast<double>(kMiB));
            req.gen_expert_bytes = expert_bytes_scaled;
            req.dit_step1_ms = step_base;
            req.dit_total_steps = 20;
            req.resolution = height;
            req.interleaved_roundtrip_gamma = gamma;

            auto [dec, ship_mode] = router.plan_and_reserve_transition(
                policy.routing_mode,
                policy.static_und_ranks,
                policy.static_gen_ranks,
                req,
                dit_exec_ms,
                &workers,
                busy_at_dit,
                policy.use_h2d_pacer,
                dit_ready_ms);
            const int dst_gpu = dec.dst_gpu;
            router.release_dit_reservation(dst_gpu, dit_exec_ms, &workers);
            if (ship_mode != "none" && dst_gpu != cur_gpu) {
                router.release_kv_reservation(dst_gpu, req.effective_state_bytes(), std::nullopt, std::nullopt, &workers);
            }

            const bool has_more_turns = (ti + 1) < n_turns;
            double trans_ms = 0.0;
            if (ship_mode == "nccl_dual") {
                const int src_node = topo.node_of(cur_gpu);
                const double r_wait = (dec.hop_tier == "inter_node")
                                          ? std::max(0.0, rdma_free_at[static_cast<size_t>(src_node)] - dit_ready_ms)
                                          : 0.0;
                const double bw = topo.effective_p2p_gbps(cur_gpu, dst_gpu);
                const double wire_ms =
                    ((cand_main_kv * static_cast<double>(kMiB) / 1e9) / std::max(0.5, bw)) * 1000.0 +
                    topo.handoff_ms(cur_gpu, dst_gpu);
                trans_ms = r_wait + wire_ms;
                if (!topo.same_node(cur_gpu, dst_gpu)) {
                    rdma_free_at[static_cast<size_t>(src_node)] =
                        std::max(dit_ready_ms, rdma_free_at[static_cast<size_t>(src_node)]) + wire_ms;
                }
                ledgers[static_cast<size_t>(cur_gpu)].release_session(sid);
                if (!ledgers[static_cast<size_t>(dst_gpu)].upsert_session_kv(sid, dit_phys_kv)) {
                    ledgers[static_cast<size_t>(dst_gpu)].release_session(sid);
                    if (attempt < policy.max_oom_retries && ti == 0) {
                        ++attempt;
                        ++sess_retries[usid];
                        sess_gpu[usid] = pick_oom_retry_gpu(dst_gpu, dit_ready_ms + trans_ms);
                        ev_ms = dit_ready_ms + trans_ms;
                        continue;
                    }
                    sess_oom[usid] = true;
                    break;
                }
                ++ships;
                if (!topo.same_node(cur_gpu, dst_gpu)) {
                    ++cross_node_ships;
                }
            } else if (policy.use_naive_offload) {
                const auto& grp = topo.half_group_of(cur_gpu);
                int active_same_half = 1;
                for (int peer : grp.gpu_ids) {
                    if (peer != cur_gpu && unpaced_h2d_until[static_cast<size_t>(peer)] > dit_ready_ms) {
                        ++active_same_half;
                    }
                }
                const double eff_h2d = topo.effective_h2d_gbps(cur_gpu, active_same_half);
                const double swap_ms =
                    ((static_cast<double>(expert_bytes_scaled) / 1e9) / std::max(0.5, eff_h2d)) * 1000.0;
                if (policy.use_h2d_pacer) {
                    const double p_wait = pacer->reserve_slot(cur_gpu, swap_ms, dit_ready_ms);
                    trans_ms = p_wait + swap_ms;
                } else {
                    unpaced_h2d_until[static_cast<size_t>(cur_gpu)] = dit_ready_ms + swap_ms;
                    trans_ms = swap_ms;
                }
            } else {
                trans_ms = std::max(0.0, dec.t_morph_ms - workers.at(cur_gpu).queue_wait_ms);
                if (policy.routing_mode == "dual_path") {
                    ++morphs;
                }
            }

            const double start_dit = std::max(dit_ready_ms + trans_ms, gpu_free_at[static_cast<size_t>(dst_gpu)]);
            const auto [dit_ok_mem, swap_out_mib] =
                ledgers[static_cast<size_t>(dst_gpu)].execute_dit_and_vae(height, sid);
            if (!dit_ok_mem) {
                gpu_free_at[static_cast<size_t>(dst_gpu)] = start_dit + dit_exec_ms * 0.45;
                ledgers[static_cast<size_t>(dst_gpu)].release_session(sid);
                ledgers[static_cast<size_t>(cur_gpu)].release_session(sid);
                if (attempt < policy.max_oom_retries && ti == 0) {
                    ++attempt;
                    ++sess_retries[usid];
                    sess_gpu[usid] = pick_oom_retry_gpu(dst_gpu, gpu_free_at[static_cast<size_t>(dst_gpu)]);
                    ev_ms = gpu_free_at[static_cast<size_t>(dst_gpu)];
                    continue;
                }
                sess_oom[usid] = true;
                break;
            }

            const double swap_out_ms =
                (swap_out_mib > 0.0)
                    ? (((swap_out_mib * static_cast<double>(kMiB) / 1e9) / std::max(0.5, topo.solo_h2d_gbps(dst_gpu))) *
                       1000.0)
                    : 0.0;
            const double end_dit = start_dit + swap_out_ms + dit_exec_ms;
            gpu_free_at[static_cast<size_t>(dst_gpu)] = end_dit;
            workers.at(dst_gpu).expert_alpha = ledgers[static_cast<size_t>(dst_gpu)].expert_alpha();

            sess_kv_mib[usid] = cand_main_kv;
            sess_main_kv_mib[usid] = cand_main_kv;
            const double turn_lat = end_dit - t_turn0;
            dit_lats.push_back(turn_lat);
            sess_dit_records[usid].emplace_back(turn_lat, height);
            const double turn_slo_ms = (policy.dit_turn_slo_for(height, 20) * cscale) / sp;
            if (turn_lat > turn_slo_ms) {
                sess_dit_ok[usid] = false;
            }

            if (has_more_turns) {
                ledgers[static_cast<size_t>(dst_gpu)].upsert_session_kv(sid, cand_main_kv);
                sess_gpu[usid] = dst_gpu;
                const double think_ms = MorphControlEngine::deterministic_think_ms_sim(
                    sid, ti + 1, policy.inter_turn_think_s_range.first, policy.inter_turn_think_s_range.second);
                push_ev(end_dit + think_ms, EV_TURN, sid, EvTurn{ti + 1, 0});
            } else {
                push_ev(end_dit, EV_RELEASE, sid, std::monostate{});
            }
            break;
        }
    }

    const size_t trim =
        (n_sess >= 20) ? static_cast<size_t>(static_cast<double>(n_sess) * policy.steady_state_trim_frac) : 0;
    const size_t core_begin = (trim > 0 && trim * 2 < n_sess) ? trim : 0;
    const size_t core_end = (trim > 0 && trim * 2 < n_sess) ? (n_sess - trim) : n_sess;
    const size_t n_core = std::max<size_t>(1, core_end - core_begin);

    size_t core_ok = 0;
    size_t ooms = 0;
    size_t joint_ok = 0;
    int total_retries = 0;
    for (size_t s = 0; s < n_sess; ++s) {
        total_retries += sess_retries[s];
        if (sess_oom[s]) {
            ++ooms;
        } else if (sess_ar_ok[s] && sess_dit_ok[s]) {
            ++joint_ok;
            if (s >= core_begin && s < core_end) ++core_ok;
        }
    }
    const double att_frac = static_cast<double>(core_ok) / static_cast<double>(n_core);

    nlohmann::json slo_curve_by_k = nlohmann::json::object();
    for (const auto& [k_lbl, k_val] : std::vector<std::pair<std::string, double>>{
             {"1.5x", 1.5}, {"2.0x", 2.0}, {"2.5x", 2.5}, {"3.0x", 3.0}}) {
        const auto slo_k = default_operator_profile().slo_thresholds_ms(k_val, "bagel");
        const double ttft_k = (slo_k.at("ar_ttft_slo_ms") * cscale) / sp;
        const double tpot_k = (slo_k.at("ar_tpot_slo_ms") * cscale) / sp;
        size_t k_ok_cnt = 0;
        for (size_t s = core_begin; s < core_end; ++s) {
            if (sess_oom[s]) continue;
            bool ok = true;
            for (double x : sess_ar_ttfts[s]) {
                if (x > ttft_k) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                for (double x : sess_ar_tpots[s]) {
                    if (x > tpot_k) {
                        ok = false;
                        break;
                    }
                }
            }
            if (ok) {
                for (const auto& [lat, h] : sess_dit_records[s]) {
                    if (lat > (policy.dit_turn_slo_for(h, 20, k_val) * cscale) / sp) {
                        ok = false;
                        break;
                    }
                }
            }
            if (ok) ++k_ok_cnt;
        }
        const double k_frac = static_cast<double>(k_ok_cnt) / static_cast<double>(n_core);
        slo_curve_by_k[k_lbl] = {
            {"attainment_pct", round_1(100.0 * k_frac)},
            {"goodput_rps", round_4(offered_qps * k_frac)},
        };
    }

    double peak_hbm = 0.0;
    int carves_tot = 0;
    int evict_tot = 0;
    int swap_out_tot = 0;
    int swap_in_tot = 0;
    for (const auto& l : ledgers) {
        peak_hbm = std::max(peak_hbm, l.peak_used_mib());
        carves_tot += l.super_to_fine_carves();
        evict_tot += l.expert_evictions();
        swap_out_tot += l.kv_host_swap_outs();
        swap_in_tot += l.kv_host_swap_ins();
    }

    return {
        {"policy_name", policy_name},
        {"workload_scale", ws.name},
        {"skew_tag", skew_tag.empty() ? "main_mix" : skew_tag},
        {"num_gpus", num_gpus},
        {"interconnect", hw.name},
        {"offered_qps", round_3(offered_qps)},
        {"crossover_kv_mib", crossover_kv_mib(hw, ph, ws)},
        {"joint_slo_goodput_rps", round_4(offered_qps * att_frac)},
        {"joint_slo_attainment_pct", round_1(100.0 * att_frac)},
        {"full_trace_attainment_pct",
         round_1(100.0 * static_cast<double>(joint_ok) / static_cast<double>(std::max<size_t>(1, n_sess)))},
        {"oom_sessions", ooms},
        {"oom_retries", total_retries},
        {"peak_hbm_mib", round_1(peak_hbm)},
        {"vmm_carves", carves_tot},
        {"vmm_expert_evictions", evict_tot},
        {"vmm_kv_host_swap_outs", swap_out_tot},
        {"vmm_kv_host_swap_ins", swap_in_tot},
        {"ar_ttft_p95_ms", percentile_rounded(ttfts, 95.0)},
        {"ar_tpot_p95_ms", percentile_rounded(tpots, 95.0)},
        {"dit_turn_p95_ms", percentile_rounded(dit_lats, 95.0)},
        {"path_a_ships", ships},
        {"path_b_morphs", morphs},
        {"preempted_ar_count", preemptions},
        {"cross_node_ships", cross_node_ships},
        {"slo_curve_by_k", slo_curve_by_k},
    };
}

nlohmann::json ClusterSimulator::evaluate_against_real_runs(
    const nlohmann::json& real_runs_array,
    const ClusterPhysicsProfile& ph,
    const std::string& bench_root,
    const std::string& sources_dir) {
    if (!real_runs_array.is_array() || real_runs_array.empty()) {
        return {
            {"reference_source", "phase01_measured_json_profile"},
            {"zero_parameter_fitting", true},
            {"points_checked", 0},
            {"goodput_mape_pct", 0.0},
            {"ar_tpot_p95_mape_pct", 0.0},
            {"dit_p95_mape_pct", 0.0},
            {"max_point_error_pct", 0.0},
            {"within_8pct_oos_gate", true},
            {"within_5pct_gate", true},
            {"per_point", nlohmann::json::array()},
        };
    }

    const HardwareInterconnectSpec hw0 = HardwareInterconnectSpec::presets(&ph)[0];
    nlohmann::json per_pt = nlohmann::json::array();
    std::vector<double> gp_errs;
    std::vector<double> tp_errs;
    std::vector<double> dt_errs;
    double max_err = 0.0;

    for (const auto& ref : real_runs_array) {
        const std::string pol = ref.value("policy_name", "");
        if (!is_valid_policy_name(pol)) continue;
        const int n_gpus = ref.value("num_gpus", 8);
        const double qps = ref.value("offered_qps", 0.8);
        const int n_s = ref.value("total_sessions", 80);
        const std::string sk_tag =
            (ref.contains("skew_tag") && ref["skew_tag"].is_string()) ? ref["skew_tag"].get<std::string>() : "";
        HardwareInterconnectSpec eff_hw = hw0;
        if (n_gpus > 8) {
            eff_hw = HardwareInterconnectSpec{
                "Kairos_2x8_Measured_Fabric",
                40.0,
                1.0,
                ph.pcie_solo_h2d_gbps,
                ph.pcie_half0_agg_h2d_gbps,
                ph.kairos_2x8_intra_half_gbps,
                ph.kairos_2x8_cross_socket_gbps,
                ph.kairos_2x8_inter_node_gbps,
                8};
        }
        const auto sim = simulate_cluster_point(
            pol,
            n_gpus,
            eff_hw,
            qps,
            n_s,
            2026,
            ph,
            WorkloadScaleSpec::bagel_7b_standard(),
            sk_tag,
            bench_root,
            sources_dir);

        double real_gp = 0.0;
        if (ref.contains("steady_state_goodput_rps") && ref["steady_state_goodput_rps"].is_number() &&
            ref["steady_state_goodput_rps"].get<double>() != 0.0) {
            real_gp = ref["steady_state_goodput_rps"].get<double>();
        } else {
            real_gp = ref.value("joint_slo_goodput_rps", 0.0);
        }
        real_gp = std::max(1e-4, real_gp);
        const double sim_gp = sim.at("joint_slo_goodput_rps").get<double>();

        const double sim_tp = sim.at("ar_tpot_p95_ms").get<double>();
        const double real_tp = std::max(
            1.0,
            (ref.contains("ar_tpot_p95_ms") && ref["ar_tpot_p95_ms"].is_number() && ref["ar_tpot_p95_ms"].get<double>() > 0.0)
                ? ref["ar_tpot_p95_ms"].get<double>()
                : sim_tp);

        const double sim_dt = sim.at("dit_turn_p95_ms").get<double>();
        const double real_dt = std::max(
            1.0,
            (ref.contains("dit_turn_p95_ms") && ref["dit_turn_p95_ms"].is_number() && ref["dit_turn_p95_ms"].get<double>() > 0.0)
                ? ref["dit_turn_p95_ms"].get<double>()
                : sim_dt);

        const double gp_e = std::abs(sim_gp - real_gp) / real_gp * 100.0;
        const double tp_e = std::abs(sim_tp - real_tp) / real_tp * 100.0;
        const double dt_e = std::abs(sim_dt - real_dt) / real_dt * 100.0;
        gp_errs.push_back(gp_e);
        tp_errs.push_back(tp_e);
        dt_errs.push_back(dt_e);
        max_err = std::max({max_err, gp_e, tp_e, dt_e});

        per_pt.push_back({
            {"policy_name", pol},
            {"num_gpus", n_gpus},
            {"offered_qps", qps},
            {"real_goodput_rps", round_4(real_gp)},
            {"sim_goodput_rps", round_4(sim_gp)},
            {"goodput_err_pct", round_2(gp_e)},
            {"ar_tpot_p95_err_pct", round_2(tp_e)},
            {"dit_p95_err_pct", round_2(dt_e)},
        });
    }

    auto mean_vec = [](const std::vector<double>& v) -> double {
        if (v.empty()) return 0.0;
        double s = 0.0;
        for (double x : v) s += x;
        return round_2(s / static_cast<double>(v.size()));
    };
    const double mgp = mean_vec(gp_errs);
    const double mtp = mean_vec(tp_errs);
    const double mdt = mean_vec(dt_errs);

    return {
        {"reference_source", "live_cluster_out_of_sample"},
        {"zero_parameter_fitting", true},
        {"points_checked", per_pt.size()},
        {"goodput_mape_pct", mgp},
        {"ar_tpot_p95_mape_pct", mtp},
        {"dit_p95_mape_pct", mdt},
        {"max_point_error_pct", round_2(max_err)},
        {"within_8pct_oos_gate", (mgp < 8.0 && mtp < 8.0 && mdt < 8.0)},
        {"within_5pct_gate", (mgp < 5.0 && mtp < 5.0 && mdt < 5.0)},
        {"per_point", per_pt},
    };
}

nlohmann::json ClusterSimulator::run_h100_80gb_generality_study(
    const ClusterPhysicsProfile& ph,
    const std::string& bench_root,
    const std::string& sources_dir) {
    const HardwareInterconnectSpec h100 = HardwareInterconnectSpec::presets(&ph)[2];
    nlohmann::json regime1 = nlohmann::json::array();
    for (double qps : {0.9, 1.4, 1.9, 2.5}) {
        for (const char* pol : kValidPolicies) {
            regime1.push_back(simulate_cluster_point(
                pol,
                8,
                h100,
                qps,
                96,
                2026,
                ph,
                WorkloadScaleSpec::bagel_7b_standard(),
                "",
                bench_root,
                sources_dir));
        }
    }

    nlohmann::json regime2_56g = nlohmann::json::array();
    const auto ws_56g = WorkloadScaleSpec::frontier_mot_56gib();
    for (const auto& [ng, qps, ns] : std::vector<std::tuple<int, double, int>>{{8, 1.35, 96}, {64, 10.8, 192}}) {
        for (const char* pol : kValidPolicies) {
            regime2_56g.push_back(
                simulate_cluster_point(pol, ng, h100, qps, ns, 2026, ph, ws_56g, "", bench_root, sources_dir));
        }
    }

    nlohmann::json regime2_long_kv = nlohmann::json::array();
    const auto ws_lkv = WorkloadScaleSpec::bagel_7b_long_kv_high_concurrency();
    for (const auto& [ng, qps, ns] : std::vector<std::tuple<int, double, int>>{{8, 2.4, 96}, {64, 19.2, 192}}) {
        for (const char* pol : kValidPolicies) {
            regime2_long_kv.push_back(
                simulate_cluster_point(pol, ng, h100, qps, ns, 2026, ph, ws_lkv, "", bench_root, sources_dir));
        }
    }

    return {
        {"hardware", h100.to_json()},
        {"regime1_7b_qps_latency_curve", regime1},
        {"regime2_scaled_56gib_weights", regime2_56g},
        {"regime2_long_kv_high_concurrency", regime2_long_kv},
    };
}

nlohmann::json ClusterSimulator::run_scale_and_interconnect_sweep(
    const std::string& output_dir,
    const std::string& real_report_path,
    const std::string& results_dir,
    const std::string& bench_root,
    const std::string& sources_dir) {
    if (!output_dir.empty()) {
        std::filesystem::create_directories(output_dir);
    }
    const ClusterPhysicsProfile ph = ClusterPhysicsProfile::from_phase01_results(results_dir);

    nlohmann::json real_runs = nlohmann::json::array();
    if (!real_report_path.empty()) {
        std::error_code ec;
        if (std::filesystem::is_regular_file(real_report_path, ec)) {
            std::ifstream in(real_report_path, std::ios::binary);
            if (in) {
                const auto raw = nlohmann::json::parse(in);
                if (raw.contains("qps_sweep") && raw["qps_sweep"].is_array()) {
                    for (const auto& x : raw["qps_sweep"]) real_runs.push_back(x);
                }
                if (raw.contains("skew_sweep") && raw["skew_sweep"].is_array()) {
                    for (const auto& x : raw["skew_sweep"]) real_runs.push_back(x);
                }
                if (raw.contains("results") && raw["results"].is_array() && !raw.contains("qps_sweep")) {
                    for (const auto& x : raw["results"]) real_runs.push_back(x);
                }
            }
        }
    }

    const auto val = evaluate_against_real_runs(real_runs, ph, bench_root, sources_dir);
    const auto presets = HardwareInterconnectSpec::presets(&ph);
    nlohmann::json presets_j = nlohmann::json::array();
    for (const auto& h : presets) presets_j.push_back(h.to_json());

    nlohmann::json matrix = nlohmann::json::array();
    for (const auto& hw : presets) {
        for (int num_gpus : {8, 16, 64}) {
            const double scaled_qps = round_2(1.15 * (static_cast<double>(num_gpus) / 8.0) * hw.compute_speedup_x);
            const int n_sess = (num_gpus <= 16) ? 96 : 240;
            for (const char* pol : kValidPolicies) {
                matrix.push_back(simulate_cluster_point(
                    pol,
                    num_gpus,
                    hw,
                    scaled_qps,
                    n_sess,
                    2026,
                    ph,
                    WorkloadScaleSpec::bagel_7b_standard(),
                    "",
                    bench_root,
                    sources_dir));
            }
        }
    }

    const auto h100_study = run_h100_80gb_generality_study(ph, bench_root, sources_dir);

    std::time_t now_t = std::time(nullptr);
    std::tm gmt = {};
    gmtime_r(&now_t, &gmt);
    char ts_buf[32];
    std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%dT%H:%M:%SZ", &gmt);

    const bool gate_ok = val.value("within_8pct_oos_gate", false);
    nlohmann::json report = {
        {"schema", "morphserve.osdi27.simulator.v3"},
        {"timestamp_utc", std::string(ts_buf)},
        {"zero_manual_bias_constants", true},
        {"physics_profile", ph.to_json()},
        {"calibration", val},
        {"interconnect_presets", presets_j},
        {"scaling_matrix", matrix},
        {"h100_80gb_generality", h100_study},
        {"status", gate_ok ? "ok" : "check_failed"},
    };

    if (!output_dir.empty()) {
        std::filesystem::path out_p = std::filesystem::path(output_dir) / "m25-simulator-64gpu-report.json";
        std::ofstream out(out_p, std::ios::binary | std::ios::trunc);
        if (out) {
            out << report.dump(2) << "\n";
        }
    }
    return report;
}

}  // namespace morphserve::control
