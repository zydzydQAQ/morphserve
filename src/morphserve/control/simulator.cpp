#include "morphserve/control/simulator.hpp"

#include "morphserve/control/morph_controller.hpp"
#include "morphserve/control/router.hpp"
#include "morphserve/runtime/h2d_pacer.hpp"

#include <algorithm>
#include <array>
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

double percentile_raw_sim(const std::vector<double>& vals, double p) {
    if (vals.empty()) {
        return 0.0;
    }
    std::vector<double> s = vals;
    std::sort(s.begin(), s.end());
    const double idx = static_cast<double>(s.size() - 1) * (p / 100.0);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = std::min(s.size() - 1, static_cast<size_t>(std::ceil(idx)));
    if (lo == hi) {
        return s[lo];
    }
    const double frac = idx - static_cast<double>(lo);
    return s[lo] * (1.0 - frac) + s[hi] * frac;
}

double mean_raw_sim(const std::vector<double>& vals) {
    if (vals.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (double v : vals) {
        sum += v;
    }
    return sum / static_cast<double>(vals.size());
}

struct SimArFlags {
    bool p95_ok = true;
    bool mean_ok = true;
    bool strict_max_ok = true;
};

SimArFlags eval_sim_ar_slos(
    const std::vector<double>& ttfts,
    const std::vector<double>& flat_tpots,
    const std::vector<std::vector<double>>& turn_tpots,
    double ttft_cap,
    double tpot_cap) {
    for (double x : ttfts) {
        if (x > ttft_cap) {
            return SimArFlags{false, false, false};
        }
    }
    SimArFlags flags{true, true, true};
    bool has_turns = false;
    for (const auto& tv : turn_tpots) {
        if (!tv.empty()) {
            has_turns = true;
            break;
        }
    }
    if (has_turns) {
        for (const auto& tv : turn_tpots) {
            if (tv.empty()) continue;
            if (percentile_raw_sim(tv, 95.0) > tpot_cap) flags.p95_ok = false;
            if (mean_raw_sim(tv) > tpot_cap) flags.mean_ok = false;
            for (double x : tv) {
                if (x > tpot_cap) {
                    flags.strict_max_ok = false;
                    break;
                }
            }
        }
    } else if (!flat_tpots.empty()) {
        if (percentile_raw_sim(flat_tpots, 95.0) > tpot_cap) flags.p95_ok = false;
        if (mean_raw_sim(flat_tpots) > tpot_cap) flags.mean_ok = false;
        for (double x : flat_tpots) {
            if (x > tpot_cap) {
                flags.strict_max_ok = false;
                break;
            }
        }
    }
    return flags;
}

bool select_sim_ar_slo(const SimArFlags& f, const std::string& mode) noexcept {
    if (mode == "turn_mean") return f.mean_ok;
    if (mode == "token_max") return f.strict_max_ok;
    return f.p95_ok;
}

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
    std::string phase_label;
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
        std::vector<std::string> cand_rels;
        if (skew_tag == "ar80_dit20") {
            cand_rels = {"suites/skew_ar80_dit20.json"};
        } else if (skew_tag == "ar50_dit50") {
            cand_rels = {"suites/skew_ar50_dit50.json"};
        } else if (skew_tag == "ar20_dit80") {
            cand_rels = {"suites/skew_ar20_dit80.json"};
        } else if (skew_tag.empty() || skew_tag == "main_mix") {
            cand_rels = {"suites/main_mix_80.json", "suites/main_mix_v2.json", "suites/main_mix.json"};
        } else {
            cand_rels = {"suites/" + skew_tag + ".json", "suites/skew_" + skew_tag + ".json"};
        }

        std::filesystem::path suite_p;
        std::error_code ec;
        for (const auto& rel : cand_rels) {
            const auto cand = std::filesystem::path(bench_root) / rel;
            if (std::filesystem::is_regular_file(cand, ec)) {
                suite_p = cand;
                break;
            }
        }
        if (!suite_p.empty() && std::filesystem::is_regular_file(suite_p, ec)) {
            std::ifstream in(suite_p, std::ios::binary);
            if (in) {
                const auto j = nlohmann::json::parse(in);
                if (j.contains("sessions") && j["sessions"].is_array() && !j["sessions"].empty()) {
                    for (const auto& sj : j["sessions"]) {
                        RawSuiteSession rs;
                        rs.session_id = sj.value("session_id", "");
                        rs.quadrant = sj.value("quadrant", "visual_cot");
                        rs.phase_label = sj.value("phase_label", "");
                        rs.arrival_offset_s = sj.value("arrival_offset_s", 0.0);
                        if (sj.contains("turns") && sj["turns"].is_array()) {
                            for (const auto& tj : sj["turns"]) {
                                int n_img = 0;
                                if (tj.contains("image_paths") && tj["image_paths"].is_array()) {
                                    n_img = static_cast<int>(tj["image_paths"].size());
                                }
                                rs.turns.push_back(SimTurnSpec{
                                    tj.value("modality_plan", "ar_only"),
                                    tj.value("height", 512),
                                    tj.value("dit_steps", 20),
                                    tj.value("max_ar_tokens", 48),
                                    n_img,
                                    tj.value("use_prev_image", false)});
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
      vae_spike_scale_(vae_spike_scale),
      cur_super_pool_mib_(static_cast<double>(policy_.super_pool_mib)),
      cur_fine_pool_mib_(static_cast<double>(policy_.fine_kv_pool_mib)) {}

double SimGpuMemoryLedger::offloaded_base_mib() const noexcept {
    const bool is_real_40g_pcie = (total_hbm_mib_ < 40900.0) && (total_hbm_mib_ > 39500.0);
    const double nccl_comm_mib =
        (is_real_40g_pcie && policy_.num_gpus >= 2 &&
         (policy_.routing_mode == "dual_path" || policy_.routing_mode == "static_nccl_fission"))
            ? 720.0
            : 0.0;
    return physics_.offloaded_base_driver_mib * weight_scale_ + nccl_comm_mib;
}

double SimGpuMemoryLedger::monolithic_weights_mib() const noexcept {
    const bool is_real_40g_pcie = (total_hbm_mib_ < 40900.0) && (total_hbm_mib_ > 39500.0);
    const double nccl_comm_mib =
        (is_real_40g_pcie && policy_.num_gpus >= 2 &&
         (policy_.routing_mode == "dual_path" || policy_.routing_mode == "static_nccl_fission"))
            ? 720.0
            : 0.0;
    return physics_.monolithic_loaded_driver_mib * weight_scale_ + nccl_comm_mib;
}

double SimGpuMemoryLedger::pageable_expert_mib() const noexcept {
    return std::max(0.0, (physics_.monolithic_loaded_driver_mib - physics_.offloaded_base_driver_mib) * weight_scale_);
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
        // Each `VmmNaiveCache` allocates 28 layer slabs at `2 MiB` physical `cuMemCreate` granularity;
        // without Module-3 CoW, edit/CoT turns instantiate separate `main`/`cfg` caches (`38 MiB` slab rounding).
        const double slab_step_mib = policy_.use_module3_cow_loopback ? 16.0 : 56.0;
        const double non_cow_slab_rem_mib = policy_.use_module3_cow_loopback ? 0.0 : 38.0;
        for (const auto& [_, v] : session_kv_mib_) {
            sum += std::max(slab_step_mib, 2.0 * std::ceil(std::max(0.01, v) / 2.0) + non_cow_slab_rem_mib);
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
    reclaim_empty_slabs_to_super();
    return freed_mib;
}

void SimGpuMemoryLedger::reclaim_empty_slabs_to_super() noexcept {
    if (!(policy_.use_tripool_vmm && policy_.use_expert_pager)) {
        return;
    }
    const double base_fine = static_cast<double>(policy_.fine_kv_pool_mib);
    const double super_gran = std::max(16.0, static_cast<double>(policy_.super_handle_mib));
    const double kv_tot = active_kv_mib();
    while (cur_fine_pool_mib_ > base_fine && (cur_fine_pool_mib_ - kv_tot) >= super_gran) {
        cur_fine_pool_mib_ -= super_gran;
        cur_super_pool_mib_ += super_gran;
    }
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
        const double super_gran_mib = std::max(16.0, static_cast<double>(policy_.super_handle_mib));
        const double full_exp_mib = static_cast<double>(num_expert_layers_ * layer_stride_mib_);
        // Reserve physical headroom for `1024x1024` DiT + Scoped-VAE spikes (`4*dit_512 + scoped_vae_1024 + 2 GiB`);
        // when `allow_vmm_overflow` still has free driver HBM above that (e.g. 80 GiB H100), allocate 2 MiB overflow
        // handles before carving/evicting resident experts.
        const double transient_headroom_mib =
            (4.0 * physics_.dit_512_transient_mib * std::sqrt(weight_scale_)) +
            (physics_.scoped_vae_1024_spike_mib * vae_spike_scale_) + 2048.0;
        while (kv_tot > cur_fine_pool_mib_ && cur_super_pool_mib_ >= super_gran_mib) {
            const bool would_evict_experts = (cur_super_pool_mib_ - super_gran_mib) < full_exp_mib;
            const double projected_phys_mib =
                base_weights_mib() + std::max(cur_super_pool_mib_, full_exp_mib) + kv_tot + transient_headroom_mib;
            if (would_evict_experts && policy_.allow_vmm_overflow && (projected_phys_mib <= total_hbm_mib_)) {
                break;
            }
            cur_super_pool_mib_ -= super_gran_mib;
            cur_fine_pool_mib_ += super_gran_mib;
            ++super_to_fine_carves_;
        }
        const double pool_cap = cur_super_pool_mib_ + cur_fine_pool_mib_;
        while (resident_expert_layers_ > 0 &&
               (static_cast<double>(resident_expert_layers_ * layer_stride_mib_) + kv_tot) > pool_cap) {
            const double projected_phys_mib =
                base_weights_mib() + static_cast<double>(resident_expert_layers_ * layer_stride_mib_) +
                kv_tot + transient_headroom_mib;
            if (policy_.allow_vmm_overflow && (projected_phys_mib <= total_hbm_mib_)) {
                break;
            }
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
        // `prism_tetriserve_coloc` (`kv_vmm_only`) pre-reserves `fine_kv_pool_mib + super_pool_mib` physical VMM handles.
        const double phys_kv =
            policy_.kv_vmm_only
                ? std::max(kv_tot, static_cast<double>(policy_.fine_kv_pool_mib + policy_.super_pool_mib))
                : kv_tot;
        used = base_weights_mib() + phys_kv;
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

bool SimGpuMemoryLedger::check_transient_spike(
    double spike_mib,
    std::optional<int> active_sid,
    bool contiguous_alloc,
    double retain_unscoped_frag_mib) {
    if (spike_mib <= 0.0) return true;
    double kv_tot = active_kv_mib();
    if (policy_.use_tripool_vmm && policy_.use_expert_pager) {
        double used = base_weights_mib() + static_cast<double>(resident_expert_layers_ * layer_stride_mib_) +
                      kv_tot + spike_mib;
        if (used > total_hbm_mib_ && active_sid.has_value()) {
            evict_idle_kv_to_host(used - total_hbm_mib_, active_sid);
            kv_tot = active_kv_mib();
            used = base_weights_mib() + static_cast<double>(resident_expert_layers_ * layer_stride_mib_) +
                   kv_tot + spike_mib;
        }
        if (used > total_hbm_mib_) return false;
        peak_used_mib_ = std::max(peak_used_mib_, used);
        return true;
    }
    const double gen_w = policy_.use_naive_offload ? pageable_expert_mib() : 0.0;
    const double held_frag = (policy_.use_scoped_vae || policy_.use_naive_offload) ? 0.0 : retained_unscoped_vae_mib_;
    const double eff_spike = (policy_.use_naive_offload && !contiguous_alloc && retain_unscoped_frag_mib > 0.0)
                                 ? std::max(0.0, spike_mib - retain_unscoped_frag_mib)
                                 : spike_mib;
    const double phys_kv =
        policy_.kv_vmm_only
            ? std::max(kv_tot, static_cast<double>(policy_.fine_kv_pool_mib + policy_.super_pool_mib))
            : kv_tot;
    const double eff_transient = contiguous_alloc ? (eff_spike + held_frag) : std::max(eff_spike, held_frag);
    const double peak = base_weights_mib() + gen_w + phys_kv + eff_transient;
    if (peak > total_hbm_mib_) {
        retained_unscoped_vae_mib_ = 0.0;
        peak_used_mib_ = std::max(peak_used_mib_, total_hbm_mib_);
        return false;
    }
    if (!policy_.use_scoped_vae && !policy_.use_naive_offload && retain_unscoped_frag_mib > 0.0) {
        retained_unscoped_vae_mib_ = std::max(retained_unscoped_vae_mib_, retain_unscoped_frag_mib);
    }
    peak_used_mib_ = std::max(peak_used_mib_, peak);
    return true;
}

std::pair<bool, double> SimGpuMemoryLedger::execute_dit_and_vae(
    int height,
    std::optional<int> active_sid,
    std::optional<double> post_denoise_sid_kv_mib,
    double unpack_staging_mib,
    bool run_pre_denoise_vae) {
    const bool is_1024 = height >= 1024;
    // Spatial sequence length scales as (1024/512)^2 = 4.0x (`64x64=4096` vs `32x32=1024` latent tokens).
    const double dit_act_mib =
        (is_1024 ? (4.0 * physics_.dit_512_transient_mib) : physics_.dit_512_transient_mib) *
        std::sqrt(weight_scale_);
    const double scoped_vae_mib =
        (is_1024 ? physics_.scoped_vae_1024_spike_mib : physics_.scoped_vae_512_spike_mib) * vae_spike_scale_;
    const double unscoped_vae_mib =
        (is_1024 ? physics_.unscoped_vae_1024_reserved_mib : physics_.unscoped_vae_512_reserved_mib) *
        vae_spike_scale_;
    const double vae_frag_slack_mib = std::max(0.0, unscoped_vae_mib - scoped_vae_mib);
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
            last_fail_dit_frac_ = 0.12;
            return {false, swapped_out_mib};
        }
        resident_expert_layers_ = num_expert_layers_;

        if (active_sid.has_value() && post_denoise_sid_kv_mib.has_value()) {
            session_kv_mib_[*active_sid] = *post_denoise_sid_kv_mib;
            kv_tot = active_kv_mib();
        }

        const double vae_mib = policy_.use_scoped_vae ? scoped_vae_mib : unscoped_vae_mib;
        const double kv_in_fine = std::min(kv_tot, cur_fine_pool_mib_);
        const double kv_in_super = std::max(0.0, kv_tot - cur_fine_pool_mib_);
        const double super_cap = std::min(
            cur_super_pool_mib_,
            std::max(0.0, total_hbm_mib_ - base_weights_mib() - kv_in_fine));
        while (resident_expert_layers_ > 0 &&
               (static_cast<double>(resident_expert_layers_ * layer_stride_mib_) + kv_in_super + vae_mib) > super_cap) {
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
            last_fail_dit_frac_ = 1.00;
            return {false, swapped_out_mib};
        }
        peak_used_mib_ = std::max({peak_used_mib_, step_peak, vae_peak});
        return {true, swapped_out_mib};
    }

    const double gen_weights_mib = policy_.use_naive_offload ? pageable_expert_mib() : 0.0;
    // `NaiveHostOffloader` calls `torch.cuda.empty_cache()` after each swap (`pager.py:686`), so VAE allocates from clean pool!
    const double vae_mib = (policy_.use_scoped_vae || policy_.use_naive_offload) ? scoped_vae_mib : unscoped_vae_mib;
    double held_frag_mib = (policy_.use_scoped_vae || policy_.use_naive_offload) ? 0.0 : retained_unscoped_vae_mib_;
    auto eff_kv_mib = [&](double logical_sum) -> double {
        return policy_.kv_vmm_only
                   ? std::max(logical_sum, static_cast<double>(policy_.fine_kv_pool_mib + policy_.super_pool_mib))
                   : logical_sum;
    };

    // 1. `RECV_SHM` unpack staging (`torch.frombuffer(raw).to(device)` + `unpack_state` coexist before `del flat`).
    if (unpack_staging_mib > 0.0 && active_sid.has_value()) {
        const double cur_sid_kv = session_kv_mib_.count(*active_sid) ? session_kv_mib_.at(*active_sid) : 0.0;
        const double others_kv = std::max(0.0, kv_tot - cur_sid_kv);
        const double unpack_peak = base_weights_mib() + gen_weights_mib + others_kv + 2.0 * unpack_staging_mib + held_frag_mib;
        if (unpack_peak > total_hbm_mib_) {
            retained_unscoped_vae_mib_ = 0.0;
            last_fail_dit_frac_ = 0.02;
            peak_used_mib_ = std::max(peak_used_mib_, total_hbm_mib_);
            return {false, 0.0};
        }
        peak_used_mib_ = std::max(peak_used_mib_, unpack_peak);
    }

    // 1b. Deferred `pending_vae` (`static_nccl_fission` `gen` rank `bagel_adapter.py:1495`) runs `vae.encode` before denoising!
    if (run_pre_denoise_vae) {
        const double enc_peak = base_weights_mib() + gen_weights_mib + eff_kv_mib(kv_tot) + std::max(vae_mib, held_frag_mib);
        if (enc_peak > total_hbm_mib_) {
            retained_unscoped_vae_mib_ = 0.0;
            last_fail_dit_frac_ = 0.04;
            peak_used_mib_ = std::max(peak_used_mib_, total_hbm_mib_);
            return {false, 0.0};
        }
        if (!policy_.use_scoped_vae && !policy_.use_naive_offload) {
            retained_unscoped_vae_mib_ = std::max(retained_unscoped_vae_mib_, vae_frag_slack_mib);
            held_frag_mib = retained_unscoped_vae_mib_;
        }
        peak_used_mib_ = std::max(peak_used_mib_, enc_peak);
    }

    // 2. Denoising (`generate_image`): `dit_phys_kv` (`main_ctx + cfg_text_ctx`) + `dit_act_mib` + non-reusable Conv2d slack
    const double step_peak = base_weights_mib() + gen_weights_mib + eff_kv_mib(kv_tot) + dit_act_mib + held_frag_mib;
    if (step_peak > total_hbm_mib_) {
        retained_unscoped_vae_mib_ = 0.0;
        last_fail_dit_frac_ = 0.12;
        peak_used_mib_ = std::max(peak_used_mib_, total_hbm_mib_);
        return {false, 0.0};
    }

    // 3. `entry["cfg_text_ctx"] = None` before `vae.decode` (`bagel_adapter.py:1722`)
    if (active_sid.has_value() && post_denoise_sid_kv_mib.has_value()) {
        session_kv_mib_[*active_sid] = *post_denoise_sid_kv_mib;
        kv_tot = active_kv_mib();
    }
    const double vae_peak = base_weights_mib() + gen_weights_mib + eff_kv_mib(kv_tot) + vae_mib + 0.35 * dit_act_mib;
    if (vae_peak > total_hbm_mib_) {
        retained_unscoped_vae_mib_ = 0.0;
        last_fail_dit_frac_ = 1.00;
        peak_used_mib_ = std::max(peak_used_mib_, total_hbm_mib_);
        return {false, 0.0};
    }
    // `NaiveHostOffloader.drop_from_gpu()` calls `torch.cuda.empty_cache()` after each DiT turn (`pager.py:686`).
    if (!policy_.use_scoped_vae && !policy_.use_naive_offload) {
        retained_unscoped_vae_mib_ = std::max(retained_unscoped_vae_mib_, vae_frag_slack_mib);
    }
    peak_used_mib_ = std::max({peak_used_mib_, step_peak, vae_peak});
    return {true, 0.0};
}

void SimGpuMemoryLedger::release_session(int sid) {
    session_kv_mib_.erase(sid);
    swapped_session_kv_mib_.erase(sid);
    // `BagelCPTGAdapter.release_context` calls `torch.cuda.empty_cache()` when `self.vmm is None` (`bagel_adapter.py:1873`).
    if (!policy_.use_tripool_vmm && session_kv_mib_.empty()) {
        retained_unscoped_vae_mib_ = 0.0;
    }
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

namespace {

std::vector<RawSuiteSession> stratify_raw_sessions(
    const std::vector<RawSuiteSession>& sessions,
    int max_sessions) {
    if (max_sessions <= 0 || static_cast<int>(sessions.size()) <= max_sessions) {
        return sessions;
    }
    std::map<std::string, std::vector<RawSuiteSession>> by_quad;
    for (const auto& s : sessions) {
        by_quad[s.quadrant].push_back(s);
    }
    struct QuadQuota {
        std::string q;
        int cnt;
        double rem_frac;
    };
    std::vector<QuadQuota> quotas;
    int used = 0;
    const double n_tot = static_cast<double>(std::max<size_t>(1, sessions.size()));
    for (const auto& [q, items] : by_quad) {
        const double exact = static_cast<double>(max_sessions) * (static_cast<double>(items.size()) / n_tot);
        const int fl = static_cast<int>(std::floor(exact));
        const int cnt = std::min(static_cast<int>(items.size()), fl);
        quotas.push_back(QuadQuota{q, cnt, exact - std::floor(exact)});
        used += cnt;
    }
    std::sort(quotas.begin(), quotas.end(), [](const QuadQuota& a, const QuadQuota& b) {
        if (std::abs(a.rem_frac - b.rem_frac) > 1e-12) return a.rem_frac > b.rem_frac;
        return a.q < b.q;
    });
    std::map<std::string, int> counts;
    for (const auto& qq : quotas) counts[qq.q] = qq.cnt;
    int rem = max_sessions - used;
    size_t idx = 0;
    while (rem > 0 && !quotas.empty()) {
        const std::string& q = quotas[idx % quotas.size()].q;
        if (counts[q] < static_cast<int>(by_quad[q].size())) {
            counts[q] += 1;
            --rem;
        } else {
            bool all_full = true;
            for (const auto& [kq, c] : counts) {
                if (c < static_cast<int>(by_quad[kq].size())) {
                    all_full = false;
                    break;
                }
            }
            if (all_full) break;
        }
        ++idx;
    }
    std::vector<RawSuiteSession> picked;
    picked.reserve(static_cast<size_t>(max_sessions));
    for (const auto& [q, items] : by_quad) {
        const int take = std::min(counts[q], static_cast<int>(items.size()));
        for (int i = 0; i < take; ++i) picked.push_back(items[static_cast<size_t>(i)]);
    }
    std::sort(picked.begin(), picked.end(), [](const RawSuiteSession& a, const RawSuiteSession& b) {
        if (a.arrival_offset_s != b.arrival_offset_s) return a.arrival_offset_s < b.arrival_offset_s;
        return a.session_id < b.session_id;
    });
    return picked;
}

std::vector<RawSuiteSession> scale_raw_sessions(
    const std::vector<RawSuiteSession>& base_sessions,
    int target_sessions) {
    const int n_base = static_cast<int>(base_sessions.size());
    if (target_sessions <= 0 || n_base == 0 || target_sessions == n_base) {
        return base_sessions;
    }
    if (target_sessions < n_base) {
        if (target_sessions <= 20) {
            return std::vector<RawSuiteSession>(
                base_sessions.begin(), base_sessions.begin() + target_sessions);
        }
        return stratify_raw_sessions(base_sessions, target_sessions);
    }
    std::vector<RawSuiteSession> ordered = base_sessions;
    std::sort(ordered.begin(), ordered.end(), [](const RawSuiteSession& a, const RawSuiteSession& b) {
        if (a.arrival_offset_s != b.arrival_offset_s) return a.arrival_offset_s < b.arrival_offset_s;
        return a.session_id < b.session_id;
    });
    const double t_min = ordered.front().arrival_offset_s;
    const double t_max = ordered.back().arrival_offset_s;
    const double span_s = std::max(0.0, t_max - t_min);
    const double mean_step_s = (n_base > 1) ? std::max(0.5, span_s / static_cast<double>(n_base - 1)) : 1.0;
    const double cycle_span_s = span_s + mean_step_s;
    const int full_cycles = target_sessions / n_base;
    const int rem = target_sessions % n_base;

    std::vector<RawSuiteSession> expanded;
    expanded.reserve(static_cast<size_t>(target_sessions));
    for (int rep = 0; rep < full_cycles; ++rep) {
        const double base_shift_s = static_cast<double>(rep) * cycle_span_s;
        for (const auto& src : ordered) {
            RawSuiteSession cp = src;
            if (rep > 0) cp.session_id = src.session_id + "_r" + std::to_string(rep);
            cp.arrival_offset_s = round_4(src.arrival_offset_s + base_shift_s);
            expanded.push_back(std::move(cp));
        }
    }
    if (rem > 0) {
        const auto tail_slice = stratify_raw_sessions(ordered, rem);
        const double tail_start_s = static_cast<double>(full_cycles) * cycle_span_s + t_min;
        for (size_t idx_t = 0; idx_t < tail_slice.size(); ++idx_t) {
            RawSuiteSession cp = tail_slice[idx_t];
            cp.session_id = cp.session_id + "_r" + std::to_string(full_cycles);
            cp.arrival_offset_s = round_4(tail_start_s + static_cast<double>(idx_t) * mean_step_s);
            expanded.push_back(std::move(cp));
        }
    }
    std::sort(expanded.begin(), expanded.end(), [](const RawSuiteSession& a, const RawSuiteSession& b) {
        if (a.arrival_offset_s != b.arrival_offset_s) return a.arrival_offset_s < b.arrival_offset_s;
        return a.session_id < b.session_id;
    });
    return expanded;
}

}  // namespace

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
        std::vector<RawSuiteSession> cloned = scale_raw_sessions(cached.sessions, num_sessions);
        const double eff_qps = std::max(0.05, offered_qps);
        if (!cached.burst_deltas.empty() && !cached.gentd_deltas.empty()) {
            std::vector<size_t> ar_idx;
            std::vector<size_t> dit_idx;
            for (size_t i = 0; i < cloned.size(); ++i) {
                const auto& q = cloned[i].quadrant;
                const bool is_ar_init =
                    (q == "multi_image_vqa" || q == "visual_cot" || q == "visual_cot_highres" ||
                     q == "interleaved_co_creation") ||
                    (q != "multiturn_edit" && q != "highres_t2i" && !cloned[i].turns.empty() &&
                     (cloned[i].turns[0].modality_plan == "ar_only" ||
                      cloned[i].turns[0].modality_plan == "cot_then_dit"));
                if (is_ar_init) {
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
                } else if (extra_edit_turns > 0 && sc.quadrant == "interleaved_co_creation" &&
                           sc.turns.size() >= 2) {
                    const SimTurnSpec prev_a = sc.turns[sc.turns.size() - 2];
                    const SimTurnSpec prev_b = sc.turns[sc.turns.size() - 1];
                    for (int e = 0; e < extra_edit_turns; ++e) {
                        sc.turns.push_back((e % 2 == 0) ? prev_a : prev_b);
                    }
                } else if (extra_edit_turns >= 2 &&
                           (sc.quadrant == "visual_cot" || sc.quadrant == "visual_cot_highres" ||
                            sc.quadrant == "multi_image_vqa")) {
                    const SimTurnSpec last = sc.turns.back();
                    for (int e = 0; e < extra_edit_turns / 2; ++e) sc.turns.push_back(last);
                }
            }
            out.push_back(SimSessionSpec{
                sc.arrival_offset_s * 1000.0,
                sc.quadrant,
                std::move(sc.turns),
                std::move(sc.session_id),
                std::move(sc.phase_label)});
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
        std::vector<SimTurnSpec> turns;
        std::string q;
        if (skew_tag == "interleaved_co_creation") {
            const int h = (idx % 3 == 0) ? 1024 : 512;
            q = "interleaved_co_creation";
            turns = {
                SimTurnSpec{"ar_only", h, 20, 48, 1},
                SimTurnSpec{"dit_edit", h, 20, 48, 0},
                SimTurnSpec{"ar_only", h, 20, 48, 0},
                SimTurnSpec{"dit_edit", h, 20, 48, 0},
            };
        } else {
            const auto& tmpl = base[static_cast<size_t>(idx) % base.size()];
            turns = tmpl.turns;
            q = tmpl.quad;
        }
        if (!turns.empty()) {
            if (extra_edit_turns > 0 && q == "multiturn_edit") {
                const auto last = turns.back();
                for (int e = 0; e < extra_edit_turns; ++e) turns.push_back(last);
            } else if (extra_edit_turns > 0 && q == "interleaved_co_creation" && turns.size() >= 2) {
                const auto prev_a = turns[turns.size() - 2];
                const auto prev_b = turns[turns.size() - 1];
                for (int e = 0; e < extra_edit_turns; ++e) {
                    turns.push_back((e % 2 == 0) ? prev_a : prev_b);
                }
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
        out.push_back(SimSessionSpec{t_cur, q, std::move(turns), "", ""});
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
    const std::vector<SimSessionSpec>* custom_sessions,
    const nlohmann::json& policy_overrides) {
    if (!is_valid_policy_name(policy_name)) {
        throw std::invalid_argument("unknown policy: " + policy_name);
    }

    nlohmann::json overrides = nlohmann::json::object();
    if (ws.think_s_range_override.has_value()) {
        const double lo_ov = ws.think_s_range_override->first;
        const double hi_ov = ws.think_s_range_override->second;
        overrides["inter_turn_think_s_range"] = nlohmann::json::array({lo_ov, hi_ov});
        if (lo_ov > 0.0 && hi_ov > lo_ov &&
            !(policy_overrides.is_object() && policy_overrides.contains("think_time_median_s"))) {
            overrides["think_time_median_s"] = std::sqrt(lo_ov * hi_ov);
        }
    }
    if (hw.gpu_hbm_gib >= 75.0 && policy_name == "morphserve") {
        overrides["super_pool_mib"] = ws.super_pool_mib_80g;
        overrides["fine_kv_pool_mib"] = ws.fine_kv_pool_mib_80g;
    }
    if (policy_overrides.is_object()) {
        for (auto it = policy_overrides.begin(); it != policy_overrides.end(); ++it) {
            overrides[it.key()] = it.value();
        }
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

    const double vmm_ar_ovh = policy.use_tripool_vmm ? ph.vmm_ar_kv_step_overhead_ms : 0.0;
    const double ar_tok_p50_ms = ((ph.ar_unloaded_tpot_p50_ms + vmm_ar_ovh) * cscale) / sp;
    const double ar_tok_p99_ms = ((ph.ar_unloaded_tpot_p99_ms + vmm_ar_ovh) * cscale) / sp;
    const double ar_batch2_step_ms = ((ph.ar_batched2_step_p50_ms + vmm_ar_ovh) * cscale) / sp;
    const double ar_prefill_ms = (ph.ar_prefill_ms * cscale) / sp;
    const double vae512_ms = (ph.vae_512_ms * ws.vae_spike_scale) / sp;
    const double vae1024_ms = (ph.vae_1024_ms * ws.vae_spike_scale) / sp;
    const double vit_only_img_ms =
        (std::max(100.0, ph.stock_loopback_turn_ms - ph.vae_512_ms - ph.ar_prefill_ms) * cscale) / sp;
    const double vae_encode_512_ms = (ph.vae_512_ms * cscale) / sp;
    const double vae_encode_1024_ms = (ph.vae_1024_ms * cscale) / sp;
    const double stock_img_loop_512_ms = (ph.stock_loopback_turn_ms * cscale) / sp;
    const double ms_img_loop_512_ms = (ph.morphserve_loopback_turn_ms * cscale) / sp;
    const double dit512_base_core_20_ms = (std::max(0.0, ph.dit_512_20step_base_ms - ph.vae_512_ms) * cscale) / sp;
    const double dit512_step_ms = (ph.dit_512_cfg_step_mean_ms * cscale) / sp;
    const double dit1024_step_ms = (ph.dit_1024_step_ms * cscale) / sp;
    const double dit1024_base_core_20_ms = std::max(0.0, 20.0 * dit1024_step_ms - vae1024_ms);

    // `module3-report.json` (`test3_4turn_interleaved_e2e`): Turn-0 unloaded `morphserve_gen_ms - stock_gen_ms = 56.13 ms`
    // (`0.050 ms/layer` across 1,120 layer calls; CUDA launches overlap host hooks asynchronously).
    const double host_hook_scale = (num_gpus >= 2) ? (1.0 / sp) : 0.0;
    const double pager_hook_per_layer_ms = policy.use_expert_pager ? (0.032 * host_hook_scale) : 0.0;
    const double sched_poll_per_layer_ms = (policy.preemption_mode == "layer_chunk") ? (0.018 * host_hook_scale) : 0.0;

    // `module3-report.json`: `stock_dit_512_ms_per_kv_tok = 0.10734` and `morphserve_dit_512_ms_per_kv_tok = 0.095679`
    // directly measure pure 20-step `gen_ms` slope per KV token starting from `main_seq_len = 18` (`loopback_ms` excluded).
    const double pure_stock_kv_slope = ph.stock_dit_512_ms_per_kv_tok;
    const double pure_ms_kv_slope = ph.morphserve_dit_512_ms_per_kv_tok;

    const double layer512_ms = (ph.dit_512_layer_p50_ms * cscale) / sp;
    const double layer1024_ms = layer512_ms * (ph.dit_1024_step_ms / std::max(1.0, ph.dit_512_cfg_step_mean_ms));
    const double ar_tpot_slo = (policy.ar_tpot_slo_ms * cscale) / sp;
    const double ar_ttft_slo = (policy.ar_ttft_slo_ms * cscale) / sp;
    const double guard_ms = (24.0 * cscale) / sp;
    const int chunk_layers_512 = std::max(
        1,
        std::min(8, static_cast<int>(std::floor(std::max(0.0, ar_tpot_slo - ar_tok_p50_ms - guard_ms) / std::max(0.1, layer512_ms)))));
    const int chunk_layers_1024 = std::max(
        1,
        std::min(8, static_cast<int>(std::floor(std::max(0.0, ar_tpot_slo - ar_tok_p50_ms - guard_ms) / std::max(0.1, layer1024_ms)))));
    const double chunk_q_512_ms = static_cast<double>(chunk_layers_512) * layer512_ms;
    const double chunk_q_1024_ms = static_cast<double>(chunk_layers_1024) * layer1024_ms;

    // 1024x1024 image produces 64*64=4096 VAE tokens + 70*70=4900 ViT tokens vs 1024+4900=5924 at 512x512.
    constexpr double kEdit1024KvRatio = (4096.0 + 4900.0) / (1024.0 + 4900.0);
    // 224-tensor pageable `.to(device)` + `torch.cuda.empty_cache()` effective throughput ratio vs pinned DMA.
    constexpr double kPageableOffloadBwRatio = 0.66;

    const int64_t expert_bytes_scaled =
        static_cast<int64_t>(ph.packed_expert_data_mib * ws.weight_scale * static_cast<double>(kMiB));
    const int layer_stride_scaled = static_cast<int>(std::round(static_cast<double>(ph.layer_stride_mib) * ws.weight_scale));
    const bool is_real_40g_preset =
        (std::abs(hw.gpu_hbm_gib - 40.0) < 1e-3) &&
        (hw.name.rfind("A100_", 0) == 0 || hw.name.rfind("Kairos_", 0) == 0);
    const double hbm_mib = is_real_40g_preset ? ph.driver_total_hbm_40g_mib : (hw.gpu_hbm_gib * 1024.0);

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
        init_kv_map[g] = ledgers.back().free_kv_bytes();
    }

    std::map<int, WorkerState> workers = router.init_workers(
        num_gpus,
        [&](int g) { return policy.role_of(g); },
        policy.initial_alphas,
        init_kv_map);

    std::vector<double> gpu_free_at(static_cast<size_t>(num_gpus), 0.0);
    std::vector<double> ar_slot_free_at(static_cast<size_t>(num_gpus), 0.0);
    std::vector<double> dit_denoise_start(static_cast<size_t>(num_gpus), -1.0);
    std::vector<double> dit_denoise_cfg_start(static_cast<size_t>(num_gpus), -1.0);
    std::vector<double> dit_denoise_end(static_cast<size_t>(num_gpus), -1.0);
    std::vector<double> dit_full_end(static_cast<size_t>(num_gpus), -1.0);
    std::vector<double> dit_active_uncond_step_ms(static_cast<size_t>(num_gpus), dit512_step_ms);
    std::vector<double> dit_active_cfg_step_ms(static_cast<size_t>(num_gpus), dit512_step_ms);
    std::vector<int> dit_active_sid(static_cast<size_t>(num_gpus), -1);
    std::vector<bool> dit_active_is1024(static_cast<size_t>(num_gpus), false);
    std::vector<std::array<double, 2>> preempt_slot_end(
        static_cast<size_t>(num_gpus), std::array<double, 2>{0.0, 0.0});
    std::vector<double> queued_work_ms(static_cast<size_t>(num_gpus), 0.0);
    std::vector<int64_t> inflight_ship_kv_bytes(static_cast<size_t>(num_gpus), 0);

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

    struct SimDitRec {
        double lat_ms = 0.0;
        int height = 512;
        int steps = 20;
    };
    struct SimTurnRec {
        int turn_idx = 0;
        std::string modality_plan;
        std::string status = "ok";
        std::optional<double> ar_ttft_ms;
        std::vector<double> ar_tpots_ms;
        std::optional<double> dit_ms;
        int height = 0;
        int dit_steps = 0;
        double kv_mib = 0.0;
    };
    struct PendingCotAr {
        bool valid = false;
        int turn_idx = -1;
        double ttft_ms = 0.0;
        std::vector<double> tpots_ms;
    };

    std::vector<int> sess_gpu(n_sess, -1);
    std::vector<double> sess_kv_mib(n_sess, 0.0);
    std::vector<double> sess_main_kv_mib(n_sess, 0.0);
    std::vector<double> sess_dit_actual_end(n_sess, 0.0);
    std::vector<double> sess_think_ms(n_sess, 0.0);
    std::vector<double> sess_end_ms(n_sess, 0.0);
    std::vector<bool> sess_ar_ok(n_sess, true);
    std::vector<bool> sess_dit_ok(n_sess, true);
    std::vector<bool> sess_oom(n_sess, false);
    std::vector<int> sess_retries(n_sess, 0);
    std::vector<int> sess_ships(n_sess, 0);
    std::vector<int> sess_morphs(n_sess, 0);
    std::vector<int> sess_xnode_ships(n_sess, 0);
    std::vector<int> sess_preemptions(n_sess, 0);
    std::vector<std::vector<double>> sess_ar_ttfts(n_sess);
    std::vector<std::vector<double>> sess_ar_tpots(n_sess);
    std::vector<std::vector<std::vector<double>>> sess_ar_turn_tpots(n_sess);
    std::vector<std::vector<SimDitRec>> sess_dit_records(n_sess);
    std::vector<std::vector<SimTurnRec>> sess_turn_records(n_sess);
    std::vector<PendingCotAr> sess_pending_cot_ar(n_sess);

    auto record_oom = [&](size_t usid_idx, int turn_idx, double end_t_ms) {
        sess_oom[usid_idx] = true;
        sess_end_ms[usid_idx] = std::max(sess_end_ms[usid_idx], end_t_ms);
        SimTurnRec rec;
        rec.turn_idx = turn_idx;
        if (turn_idx >= 0 && static_cast<size_t>(turn_idx) < built_sessions[usid_idx].turns.size()) {
            const auto& ts = built_sessions[usid_idx].turns[static_cast<size_t>(turn_idx)];
            rec.modality_plan = ts.modality_plan;
            rec.height = ts.height;
            rec.dit_steps = ts.dit_steps;
        }
        rec.status = "oom";
        rec.kv_mib = sess_main_kv_mib[usid_idx];
        if (sess_pending_cot_ar[usid_idx].valid && sess_pending_cot_ar[usid_idx].turn_idx == turn_idx) {
            rec.ar_ttft_ms = sess_pending_cot_ar[usid_idx].ttft_ms;
            rec.ar_tpots_ms = std::move(sess_pending_cot_ar[usid_idx].tpots_ms);
            sess_pending_cot_ar[usid_idx].valid = false;
        }
        sess_turn_records[usid_idx].push_back(std::move(rec));
    };

    enum EvKind : int {
        EV_RELEASE = 0,
        EV_DIT_END = 1,
        EV_RET_SHM_SEND = 2,
        EV_RET_UNPACK = 3,
        EV_DIT_START = 4,
        EV_SHM_RECV = 5,
        EV_SHM_SEND = 6,
        EV_DIT_READY = 7,
        EV_PREP_START = 8,
        EV_TURN = 9,
    };

    struct EvTurn {
        int ti = 0;
        int attempt = 0;
        double ar_enq_ms = -1.0;
        bool waiting_ar_lock = false;
    };
    struct EvPrepStart {
        int ti = 0;
        int attempt = 0;
        int cur_gpu = 0;
        double t_turn0 = 0.0;
        double est_prep_q = 0.0;
    };
    struct EvDitReady {
        int ti = 0;
        int attempt = 0;
        int cur_gpu = 0;
        double t_turn0 = 0.0;
        std::string quad;
        std::string plan;
        int height = 512;
        int dit_steps = 20;
        int n_turns = 1;
        double cand_main_kv = 0.0;
        double dit_phys_kv = 0.0;
        double step_base = 0.0;
        double dit_core_ms = 0.0;
        double vae_dec_ms = 0.0;
        double deferred_dit_prep_ms = 0.0;
        bool deferred_full_turn = false;
    };
    struct EvShmSend {
        EvDitReady r;
        int dst_gpu = 0;
        double send_ms = 0.0;
        double recv_ms = 0.0;
        double dst_est_q = 0.0;
        int64_t dst_inflight_b = 0;
        bool release_src = true;
    };
    struct EvDitStart {
        int ti = 0;
        int attempt = 0;
        int cur_gpu = 0;
        int dst_gpu = 0;
        std::string ship_mode;
        double t_turn0 = 0.0;
        int height = 512;
        int dit_steps = 20;
        int n_turns = 1;
        double cand_main_kv = 0.0;
        double dit_phys_kv = 0.0;
        double dit_core_ms = 0.0;
        double vae_dec_ms = 0.0;
        double pre_denoise_gpu_ms = 0.0;
        double reserved_est_ms = 0.0;
        int64_t inflight_kv_b = 0;
        double unpack_staging_mib = 0.0;
        bool run_pre_denoise_vae = false;
    };
    struct EvDitEnd {
        int ti = 0;
        int attempt = 0;
        int cur_gpu = 0;
        int dst_gpu = 0;
        std::string ship_mode;
        double t_turn0 = 0.0;
        int height = 512;
        int dit_steps = 20;
        int n_turns = 1;
        double cand_main_kv = 0.0;
        double reserved_est_ms = 0.0;
    };
    struct EvRetShmSend {
        EvDitEnd e;
        double send_ms = 0.0;
        double recv_ms = 0.0;
    };
    struct EvRetUnpack {
        int ti = 0;
        int attempt = 0;
        int cur_gpu = 0;
        int dst_gpu = 0;
        double t_turn0 = 0.0;
        int height = 512;
        int dit_steps = 20;
        int n_turns = 1;
        double cand_main_kv = 0.0;
        double recv_gpu_ms = 0.0;
    };
    using PayloadVar = std::variant<
        std::monostate,
        EvTurn,
        EvPrepStart,
        EvDitReady,
        EvShmSend,
        EvDitStart,
        EvDitEnd,
        EvRetShmSend,
        EvRetUnpack>;

    struct SimEvent {
        double ts_ms = 0.0;
        double fifo_enq_ms = 0.0;
        uint64_t fifo_seq = 0;
        int kind = 0;
        int sid = 0;
        PayloadVar payload;

        int lock_prio_group() const noexcept {
            if (kind <= EV_DIT_END) return 0;
            if (kind == EV_DIT_READY) return 1;
            if (kind == EV_TURN) {
                if (const auto* tv = std::get_if<EvTurn>(&payload)) {
                    if (tv->waiting_ar_lock) return 2;
                }
                return 3;
            }
            return 2;
        }

        bool operator>(const SimEvent& o) const noexcept {
            // Non-lock completion events (`EV_RELEASE`, `EV_DIT_END`, `EV_DIT_READY`) fire before lock acquisitions at `ts_ms`;
            // competing lock acquisitions (`EV_RET_*`, `EV_DIT_START`, `EV_SHM_SEND`, `EV_PREP_START`, and lock-waiting `EvTurn`)
            // wake in strict FIFO enqueue order (`fifo_enq_ms, fifo_seq`), matching Python `asyncio.Lock` (`cluster_engine.py:374`)!
            const int prio_grp = lock_prio_group();
            const int o_grp = o.lock_prio_group();
            return std::tie(ts_ms, prio_grp, fifo_enq_ms, fifo_seq, sid) >
                   std::tie(o.ts_ms, o_grp, o.fifo_enq_ms, o.fifo_seq, o.sid);
        }
    };

    std::priority_queue<SimEvent, std::vector<SimEvent>, std::greater<SimEvent>> pq;
    uint64_t ev_seq = 0;
    auto push_ev = [&](double ts_ms, int kind, int sid, PayloadVar p, double enq_ms = -1.0, uint64_t enq_seq = 0) {
        if (enq_seq == 0) {
            ++ev_seq;
            enq_seq = ev_seq;
        }
        const double eff_enq = (enq_ms >= 0.0) ? enq_ms : ts_ms;
        pq.push(SimEvent{ts_ms, eff_enq, enq_seq, kind, sid, std::move(p)});
    };

    for (size_t sid = 0; sid < n_sess; ++sid) {
        push_ev(built_sessions[sid].arrival_ms, EV_TURN, static_cast<int>(sid), EvTurn{0, 0});
    }

    auto compute_think_ms = [&](size_t usid_idx, int next_ti) -> double {
        const auto& s_id = built_sessions[usid_idx].session_id;
        if (!s_id.empty()) {
            return MorphControlEngine::deterministic_think_dist_s(
                       s_id,
                       next_ti,
                       policy.inter_turn_think_s_range.first,
                       policy.inter_turn_think_s_range.second,
                       policy.think_time_dist,
                       policy.think_time_median_s,
                       policy.think_time_sigma) *
                   1000.0;
        }
        return MorphControlEngine::deterministic_think_dist_ms_sim(
            static_cast<int>(usid_idx),
            next_ti,
            policy.inter_turn_think_s_range.first,
            policy.inter_turn_think_s_range.second,
            policy.think_time_dist,
            policy.think_time_median_s,
            policy.think_time_sigma);
    };

    auto compute_naive_swap_ms = [&](int gpu_id, double at_ms) -> double {
        const auto& grp = topo.half_group_of(gpu_id);
        int active_same_half = 1;
        for (int peer : grp.gpu_ids) {
            if (peer != gpu_id && unpaced_h2d_until[static_cast<size_t>(peer)] > at_ms) {
                ++active_same_half;
            }
        }
        const double eff_h2d = std::max(0.5, topo.effective_h2d_gbps(gpu_id, active_same_half) * kPageableOffloadBwRatio);
        const double swap_ms = ((static_cast<double>(expert_bytes_scaled) / 1e9) / eff_h2d) * 1000.0;
        if (policy.use_h2d_pacer) {
            const double p_wait = pacer->reserve_slot(gpu_id, swap_ms, at_ms);
            return p_wait + swap_ms;
        }
        unpaced_h2d_until[static_cast<size_t>(gpu_id)] = std::max(unpaced_h2d_until[static_cast<size_t>(gpu_id)], at_ms) + swap_ms;
        return swap_ms;
    };

    auto sync_states = [&](double now_ms,
                           std::map<int, bool>& busy_map,
                           std::map<int, int>& dit_map,
                           std::map<int, int>& pre_map) {
        busy_map.clear();
        dit_map.clear();
        pre_map.clear();
        for (int g = 0; g < num_gpus; ++g) {
            const size_t ug = static_cast<size_t>(g);
            const double rem_excl = std::max(0.0, gpu_free_at[ug] - now_ms);
            const double rem_ar = std::max(0.0, ar_slot_free_at[ug] - now_ms);
            auto& wg = workers.at(g);
            // `MorphServeClusterEngine` holds step-function reservations (`queued_work_ms`) across active + queued DiT/PREP!
            wg.queue_wait_ms = queued_work_ms[ug] + 0.5 * rem_ar;
            wg.expert_alpha = ledgers[ug].expert_alpha();
            wg.kv_free_bytes = std::max<int64_t>(0, ledgers[ug].free_kv_bytes() - inflight_ship_kv_bytes[ug]);
            busy_map[g] = (rem_excl > 0.0 || rem_ar > 0.0 || queued_work_ms[ug] > 0.0);
            dit_map[g] = (now_ms >= dit_denoise_start[ug] && now_ms < dit_full_end[ug]) ? 1 : 0;
            int p_cnt = 0;
            for (double pe : preempt_slot_end[ug]) {
                if (pe > now_ms) ++p_cnt;
            }
            pre_map[g] = p_cnt;
        }
    };

    auto pick_oom_retry_gpu = [&](int failed_gpu, double now_ms) -> int {
        std::map<int, bool> busy_now;
        std::map<int, int> dit_now;
        std::map<int, int> pre_now;
        sync_states(now_ms, busy_now, dit_now, pre_now);
        return router.select_oom_retry_gpu(
            failed_gpu, num_gpus, policy.routing_mode, policy.static_und_ranks, &workers, busy_now);
    };

    auto compute_dit_core_ms = [&](int gpu_hint,
                                   bool is_1024,
                                   bool has_3branch_cfg,
                                   bool has_shared_prefix_cfg,
                                   int main_seq_tokens,
                                   int dit_steps) -> std::pair<double, double> {
        const double steps_ratio = static_cast<double>(std::max(1, dit_steps)) / 20.0;
        const double slope_512 = has_shared_prefix_cfg ? pure_ms_kv_slope : pure_stock_kv_slope;
        // `module3-report.json`: `q_len = 4098` at `1024x1024` vs `1026` at `512x512` (`4098 / 1026 = 3.994x` Q*K FLOPs).
        const double q_scale = is_1024 ? (4098.0 / 1026.0) : 1.0;
        const double kv_tokens = std::max(0.0, static_cast<double>(main_seq_tokens) - 18.0);
        const double kv_attn_ms = ((slope_512 * q_scale * kv_tokens * steps_ratio) * cscale) / sp;
        const double base_20_ms = is_1024 ? dit1024_base_core_20_ms : dit512_base_core_20_ms;
        const int n_branches = has_3branch_cfg ? 3 : 2;
        const double layer_calls = static_cast<double>(ph.num_expert_layers * n_branches * std::max(1, dit_steps));
        const auto& lg = ledgers[static_cast<size_t>(std::clamp(gpu_hint, 0, num_gpus - 1))];
        const bool pager_paging_active =
            policy.use_expert_pager &&
            (lg.resident_expert_layers() < lg.num_expert_layers() || lg.expert_evictions() > 0 ||
             static_cast<double>(policy.super_pool_mib) <
                 (static_cast<double>(ph.num_expert_layers * layer_stride_scaled) +
                  ph.scoped_vae_1024_spike_mib * ws.vae_spike_scale));
        const double eff_per_layer_ovh =
            (pager_paging_active ? pager_hook_per_layer_ms : 0.0) + sched_poll_per_layer_ms;
        const double hook_ovh_ms = layer_calls * eff_per_layer_ovh;
        const double core_ms = (base_20_ms * steps_ratio) + kv_attn_ms + hook_ovh_ms;
        const double eff_step_ms = core_ms / static_cast<double>(std::max(1, dit_steps));
        return {core_ms, eff_step_ms};
    };

    auto excl_gpu_free_at = [&](size_t ug) -> double {
        return std::max(gpu_free_at[ug], ar_slot_free_at[ug]);
    };

    auto run_ar_phase_sim = [&](size_t usid_idx,
                                int cur_gpu,
                                double ev_ms,
                                double swap_in_ms,
                                double lock_wait_ms,
                                const SimTurnSpec& turn,
                                bool prepare_for_dit,
                                bool loopback_prev_dit_img = false,
                                double* out_ttft_ms = nullptr,
                                std::vector<double>* out_turn_tpots = nullptr) -> double {
        const size_t ug = static_cast<size_t>(cur_gpu);
        const int max_t = std::max(2, turn.max_ar_tokens);
        // `bagel-1gpu-smoke.json` & MM-Vet VQA (`ar_only`, `max_ar_tokens=64`) average ~21 tokens; `cot_then_dit` (`max_ar_tokens=48`) ~36 tokens.
        const int n_tok = prepare_for_dit ? std::max(12, static_cast<int>(std::round(0.75 * max_t))) : std::min(max_t, 21);
        const bool is_fission_und = (policy.routing_mode == "static_nccl_fission" && policy.role_of(cur_gpu) == "und");

        double prefill_ms = ar_prefill_ms;
        if (turn.num_images > 0) {
            if (prepare_for_dit && !is_fission_und) {
                const double base_img_ms = policy.use_module3_cow_loopback ? ms_img_loop_512_ms : stock_img_loop_512_ms;
                prefill_ms = static_cast<double>(turn.num_images) * base_img_ms;
            } else {
                prefill_ms = static_cast<double>(turn.num_images) * vit_only_img_ms + ar_prefill_ms;
            }
        } else if (!prepare_for_dit && loopback_prev_dit_img) {
            const double loop_prep_saving =
                policy.use_module3_cow_loopback
                    ? std::max(0.0, ((ph.cpu_pil_prep_only_ms - ph.gpu_inplace_prep_only_ms) * cscale) / sp)
                    : 0.0;
            const double eff_vit_ms = std::max(50.0, vit_only_img_ms - loop_prep_saving);
            prefill_ms = eff_vit_ms + ar_prefill_ms;
        }

        int free_pre_slot = -1;
        for (int sl = 0; sl < 2; ++sl) {
            if (preempt_slot_end[ug][static_cast<size_t>(sl)] <= ev_ms) {
                free_pre_slot = sl;
                break;
            }
        }
        // If `EXEC_AR_PREEMPT` arrives during short `pre_denoise_gpu_ms` right before `dit_denoise_start[ug]`, `ar_preempt_q`
        // holds it until `dit_denoise_start[ug]` and immediately preempts Step 0 (`bagel_adapter.py:1660`).
        const bool dit_session_live =
            (dit_full_end[ug] > ev_ms) && (ev_ms < dit_denoise_end[ug]) &&
            (ev_ms >= dit_denoise_start[ug] - (900.0 * cscale / sp));
        const bool can_preempt_mid_dit =
            (policy.preemption_mode != "non_preemptive") &&
            dit_session_live &&
            (free_pre_slot >= 0);

        auto step_quantum_at = [&](double t_ms) -> double {
            if (policy.preemption_mode == "layer_chunk") {
                return dit_active_is1024[ug] ? chunk_q_1024_ms : chunk_q_512_ms;
            }
            if (!dit_active_is1024[ug]) {
                return dit_active_cfg_step_ms[ug];
            }
            return (t_ms < dit_denoise_cfg_start[ug]) ? dit_active_uncond_step_ms[ug] : dit_active_cfg_step_ms[ug];
        };

        auto stretch_active_dit = [&](double delta_ms, bool stretch_denoise, double at_ms) {
            if (delta_ms <= 0.0) return;
            if (stretch_denoise && dit_denoise_cfg_start[ug] > at_ms) {
                dit_denoise_cfg_start[ug] += delta_ms;
            }
            if (stretch_denoise && dit_denoise_end[ug] > ev_ms) {
                dit_denoise_end[ug] += delta_ms;
            }
            if (dit_full_end[ug] > ev_ms) {
                dit_full_end[ug] += delta_ms;
            }
            gpu_free_at[ug] += delta_ms;
            const int dsid = dit_active_sid[ug];
            if (dsid >= 0 && static_cast<size_t>(dsid) < n_sess) {
                sess_dit_actual_end[static_cast<size_t>(dsid)] =
                    std::max(sess_dit_actual_end[static_cast<size_t>(dsid)], dit_full_end[ug]);
            }
        };

        if (can_preempt_mid_dit) {
            const double pre_denoise_rem = std::max(0.0, dit_denoise_start[ug] - ev_ms);
            const double other_pre_rem =
                std::max(0.0, preempt_slot_end[ug][static_cast<size_t>(1 - free_pre_slot)] - (ev_ms + pre_denoise_rem));
            // `_poll_external_ar` waits if another preempted `ArDecodeTask` is already mid-decode on `cur_gpu` (`5d81343`)!
            const double ready_in_dit = ev_ms + pre_denoise_rem + other_pre_rem;
            const double init_q_ms = step_quantum_at(ready_in_dit);
            const double phase_wait =
                pre_denoise_rem + other_pre_rem +
                std::min(std::max(0.0, dit_denoise_end[ug] - ready_in_dit), 0.5 * init_q_ms);
            const double wait_ms = phase_wait + swap_in_ms;

            const double start_prefill_ms = ev_ms + wait_ms;
            const bool still_mid_denoise_at_start = (start_prefill_ms < dit_denoise_end[ug]);
            stretch_active_dit(prefill_ms + ar_tok_p50_ms, /*stretch_denoise=*/still_mid_denoise_at_start, start_prefill_ms);
            const double ttft = lock_wait_ms + wait_ms + prefill_ms + ar_tok_p50_ms;
            sess_ar_ttfts[usid_idx].push_back(ttft);

            std::vector<double> turn_tpots;
            turn_tpots.reserve(static_cast<size_t>(std::max(0, n_tok - 1)));
            double t_cur = start_prefill_ms + prefill_ms + ar_tok_p50_ms;
            for (int tok_i = 1; tok_i < n_tok; ++tok_i) {
                const double frac = static_cast<double>(tok_i) / static_cast<double>(std::max(1, n_tok - 1));
                const double step_c = ar_tok_p50_ms + (ar_tok_p99_ms - ar_tok_p50_ms) * frac;
                const bool co_preempted = (t_cur < preempt_slot_end[ug][static_cast<size_t>(1 - free_pre_slot)]);
                const double eff_step_c = co_preempted ? ar_batch2_step_ms : step_c;
                const double dit_stretch_c = co_preempted ? (0.5 * ar_batch2_step_ms) : step_c;
                double tok_tpot = eff_step_c;
                if (t_cur < dit_denoise_end[ug]) {
                    const double q_now_ms = step_quantum_at(t_cur);
                    const double rem_denoise_ms = std::max(0.0, dit_denoise_end[ug] - t_cur);
                    tok_tpot = std::min(q_now_ms, rem_denoise_ms) + eff_step_c;
                    stretch_active_dit(dit_stretch_c, /*stretch_denoise=*/true, t_cur + std::min(q_now_ms, rem_denoise_ms));
                } else if (t_cur < dit_full_end[ug]) {
                    // `drain_remaining_ar_tasks()` runs unpreempted right before `vae.decode` (`5d81343`)!
                    stretch_active_dit(dit_stretch_c, /*stretch_denoise=*/false, t_cur);
                }
                turn_tpots.push_back(tok_tpot);
                sess_ar_tpots[usid_idx].push_back(tok_tpot);
                t_cur += tok_tpot;
            }
            sess_ar_turn_tpots[usid_idx].push_back(turn_tpots);
            if (out_ttft_ms != nullptr) *out_ttft_ms = ttft;
            if (out_turn_tpots != nullptr) *out_turn_tpots = std::move(turn_tpots);
            preempt_slot_end[ug][static_cast<size_t>(free_pre_slot)] = t_cur;
            if (still_mid_denoise_at_start) {
                ++sess_preemptions[usid_idx];
            }
            return t_cur;
        }

        // For `preemption_mode != "non_preemptive"` arriving during `PREP_DIT` / `vae.decode`, `ar_preempt_q` drains
        // immediately via `_drain_late_preempt_cmds()` (`worker_runtime.py:646`) at `gpu_free_at[ug]` before `main_cmd_q`!
        const double crit_rem_ms = std::max(0.0, gpu_free_at[ug] - ev_ms);
        const double excl_wait = lock_wait_ms + crit_rem_ms;
        const double start_ar = ev_ms + crit_rem_ms;
        const bool overlaps_ar = (ar_slot_free_at[ug] > start_ar + 1.0);
        const double prefill_wait = overlaps_ar ? std::min(ar_prefill_ms, ar_slot_free_at[ug] - start_ar) : 0.0;
        const double ttft = excl_wait + prefill_wait + swap_in_ms + prefill_ms + ar_tok_p50_ms;
        sess_ar_ttfts[usid_idx].push_back(ttft);

        std::vector<double> turn_tpots;
        turn_tpots.reserve(static_cast<size_t>(std::max(0, n_tok - 1)));
        double t_cur = start_ar + prefill_wait + swap_in_ms + prefill_ms + ar_tok_p50_ms;
        double gpu_added_ms = swap_in_ms + prefill_ms + ar_tok_p50_ms;
        for (int tok_i = 1; tok_i < n_tok; ++tok_i) {
            const double frac = static_cast<double>(tok_i) / static_cast<double>(std::max(1, n_tok - 1));
            const double solo_step = ar_tok_p50_ms + (ar_tok_p99_ms - ar_tok_p50_ms) * frac;
            const bool batched_now = (t_cur < ar_slot_free_at[ug]);
            const double tok_tpot = batched_now ? ar_batch2_step_ms : solo_step;
            gpu_added_ms += batched_now ? (0.5 * ar_batch2_step_ms) : solo_step;
            turn_tpots.push_back(tok_tpot);
            sess_ar_tpots[usid_idx].push_back(tok_tpot);
            t_cur += tok_tpot;
        }
        sess_ar_turn_tpots[usid_idx].push_back(turn_tpots);
        if (out_ttft_ms != nullptr) *out_ttft_ms = ttft;
        if (out_turn_tpots != nullptr) *out_turn_tpots = std::move(turn_tpots);
        ar_slot_free_at[ug] = std::max(start_ar, ar_slot_free_at[ug]) + gpu_added_ms;
        if (crit_rem_ms > 0.0 && policy.preemption_mode != "non_preemptive") {
            gpu_free_at[ug] = std::max(gpu_free_at[ug], ar_slot_free_at[ug]);
        }
        return t_cur;
    };

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

        if (cur_ev.kind == EV_PREP_START) {
            auto p = std::get<EvPrepStart>(cur_ev.payload);
            const size_t ug = static_cast<size_t>(p.cur_gpu);
            if (excl_gpu_free_at(ug) > ev_ms + 1e-6) {
                push_ev(excl_gpu_free_at(ug), EV_PREP_START, sid, p, cur_ev.fifo_enq_ms, cur_ev.fifo_seq);
                continue;
            }
            queued_work_ms[ug] = std::max(0.0, queued_work_ms[ug] - p.est_prep_q);
            p.est_prep_q = 0.0;

            const auto& sess_spec = built_sessions[usid];
            const auto& turn = sess_spec.turns[static_cast<size_t>(p.ti)];
            const std::string& quad = sess_spec.quadrant;
            const std::string& plan = turn.modality_plan;
            const int height = turn.height;
            const bool is_1024 = height >= 1024;
            const int n_turns = static_cast<int>(sess_spec.turns.size());

            const double swap_in_mib = ledgers[ug].ensure_session_resident(sid);
            if (swap_in_mib < 0.0) {
                ledgers[ug].release_session(sid);
                record_oom(usid, p.ti, ev_ms);
                continue;
            }
            const double swap_in_ms =
                (swap_in_mib > 0.0)
                    ? (((swap_in_mib * static_cast<double>(kMiB) / 1e9) / std::max(0.5, topo.solo_h2d_gbps(p.cur_gpu))) *
                       1000.0)
                    : 0.0;

            const bool has_img_input = (turn.num_images > 0) || (plan == "dit_edit" && p.ti > 0);
            const bool prev_was_ar_after_dit =
                (p.ti >= 2 &&
                 sess_spec.turns[static_cast<size_t>(p.ti - 1)].modality_plan == "ar_only" &&
                 sess_spec.turns[static_cast<size_t>(p.ti - 2)].modality_plan != "ar_only");
            double base_delta_kv = 28.0;
            if (quad == "multiturn_edit" || has_img_input) {
                const double full_edit_delta = ph.turn_edit_512_kv_delta_mib * (is_1024 ? kEdit1024KvRatio : 1.0);
                if (quad == "interleaved_co_creation" && prev_was_ar_after_dit && turn.num_images == 0) {
                    // Turn `p.ti - 1` (`ar_only`) already appended ViT loopback KV (~268 MiB); add remaining VAE+prompt delta.
                    base_delta_kv = std::max(28.0, full_edit_delta - 268.0);
                } else {
                    base_delta_kv = full_edit_delta;
                }
            }
            const double cand_main_kv = sess_main_kv_mib[usid] + base_delta_kv * ws.kv_per_turn_scale;
            const bool has_hist_for_cfg = has_img_input || (p.ti > 0);
            const double dit_phys_kv =
                (!policy.use_module3_cow_loopback && has_hist_for_cfg)
                    ? (cand_main_kv * ph.non_cow_kv_mult)
                    : cand_main_kv;

            const bool is_fission_und = (policy.routing_mode == "static_nccl_fission" && policy.role_of(p.cur_gpu) == "und");
            const bool pager_partial = policy.use_expert_pager && (ledgers[ug].expert_alpha() < 0.999);
            const bool defer_full_turn = has_img_input && pager_partial && !is_fission_und;
            const double src_prep_kv =
                defer_full_turn ? std::max(4.0, sess_main_kv_mib[usid]) : (is_fission_und ? cand_main_kv : dit_phys_kv);

            bool prep_mem_ok = ledgers[ug].upsert_session_kv(sid, src_prep_kv);
            if (prep_mem_ok && has_img_input && !defer_full_turn && !is_fission_und) {
                const double sc_vae = (is_1024 ? ph.scoped_vae_1024_spike_mib : ph.scoped_vae_512_spike_mib) * ws.vae_spike_scale;
                const double un_vae = (is_1024 ? ph.unscoped_vae_1024_reserved_mib : ph.unscoped_vae_512_reserved_mib) * ws.vae_spike_scale;
                const double vae_enc_spike = policy.use_scoped_vae ? sc_vae : un_vae;
                prep_mem_ok = ledgers[ug].check_transient_spike(
                    vae_enc_spike, sid, /*contiguous_alloc=*/false, std::max(0.0, un_vae - sc_vae));
            }
            if (!prep_mem_ok) {
                ledgers[ug].release_session(sid);
                if (p.attempt < policy.max_oom_retries && p.ti == 0) {
                    ++sess_retries[usid];
                    sess_gpu[usid] = pick_oom_retry_gpu(p.cur_gpu, ev_ms);
                    push_ev(ev_ms, EV_TURN, sid, EvTurn{0, p.attempt + 1});
                } else {
                    record_oom(usid, p.ti, ev_ms);
                }
                continue;
            }

            const double img_vae_ms = is_1024 ? vae_encode_1024_ms : vae_encode_512_ms;
            const double full_loop_ms =
                (policy.use_module3_cow_loopback ? ms_img_loop_512_ms : stock_img_loop_512_ms) +
                (is_1024 ? (vae_encode_1024_ms - vae_encode_512_ms) : 0.0);

            double prep_gpu_ms = swap_in_ms + ar_prefill_ms;
            double deferred_dit_prep_ms = 0.0;
            if (has_img_input) {
                if (defer_full_turn) {
                    prep_gpu_ms = swap_in_ms + (5.0 / sp);
                    deferred_dit_prep_ms = full_loop_ms;
                } else if (is_fission_und) {
                    prep_gpu_ms = swap_in_ms + vit_only_img_ms;
                    deferred_dit_prep_ms = img_vae_ms;
                } else {
                    const double naive_vae_swap =
                        policy.use_naive_offload ? (1.08 * compute_naive_swap_ms(p.cur_gpu, ev_ms + swap_in_ms)) : 0.0;
                    prep_gpu_ms = swap_in_ms + naive_vae_swap + full_loop_ms;
                }
            }

            const int main_seq_tokens = static_cast<int>(std::round(
                (cand_main_kv / std::max(0.01, ws.weight_scale)) / std::max(1e-4, ph.kv_mib_per_token)));
            const bool has_shared_prefix = policy.use_module3_cow_loopback && has_hist_for_cfg;
            const auto [dit_core_ms, eff_step_ms] =
                compute_dit_core_ms(p.cur_gpu, is_1024, has_img_input, has_shared_prefix, main_seq_tokens, turn.dit_steps);
            const double step_base = is_1024 ? dit1024_step_ms : dit512_step_ms;
            const double vae_dec_ms = is_1024 ? vae1024_ms : vae512_ms;
            (void)eff_step_ms;

            const double dit_ready_ms = ev_ms + prep_gpu_ms;
            gpu_free_at[ug] = std::max(gpu_free_at[ug], dit_ready_ms);
            push_ev(
                dit_ready_ms,
                EV_DIT_READY,
                sid,
                EvDitReady{
                    p.ti,
                    p.attempt,
                    p.cur_gpu,
                    p.t_turn0,
                    quad,
                    plan,
                    height,
                    turn.dit_steps,
                    n_turns,
                    cand_main_kv,
                    dit_phys_kv,
                    step_base,
                    dit_core_ms,
                    vae_dec_ms,
                    deferred_dit_prep_ms,
                    defer_full_turn});
            continue;
        }

        if (cur_ev.kind == EV_DIT_READY) {
            const auto p = std::get<EvDitReady>(cur_ev.payload);
            const double dit_ready_ms = ev_ms;
            const bool has_more_turns = (p.ti + 1) < p.n_turns;
            const auto& sess_spec = built_sessions[usid];
            bool has_more_edits = false;
            for (int nxt = p.ti + 1; nxt < p.n_turns; ++nxt) {
                if (sess_spec.turns[static_cast<size_t>(nxt)].modality_plan == "dit_edit") {
                    has_more_edits = true;
                    break;
                }
            }
            const int gamma = (p.plan == "dit_edit" && has_more_edits) ? 2 : 1;

            std::map<int, bool> busy_at_dit;
            std::map<int, int> dit_at_dit;
            std::map<int, int> pre_at_dit;
            sync_states(dit_ready_ms, busy_at_dit, dit_at_dit, pre_at_dit);

            const double wire_kv_mib = (policy.routing_mode == "twostage_shm") ? p.dit_phys_kv : p.cand_main_kv;
            CPTGRequestSpec req;
            req.req_id = "s" + std::to_string(sid) + "_t" + std::to_string(p.ti) + "_a" + std::to_string(p.attempt);
            req.src_gpu = p.cur_gpu;
            req.architecture = policy.cptg_architecture();
            req.kv_bytes = std::max<int64_t>(1LL * kMiB, static_cast<int64_t>(wire_kv_mib * static_cast<double>(kMiB)));
            req.gen_expert_bytes = expert_bytes_scaled;
            req.dit_step1_ms = p.step_base;
            req.dit_total_steps = p.dit_steps;
            req.resolution = p.height;
            req.interleaved_roundtrip_gamma = gamma;

            // Match `MorphServeClusterEngine` (`cluster_engine.py:834`): reserve unloaded `est_dit_ms` across full DiT execution!
            const std::string fam = (policy.model_family == "internvl_u") ? "internvl_u" : "bagel";
            const double est_dit_res_ms =
                (default_operator_profile().unloaded_dit_turn_ms(fam, p.height, p.dit_steps) * cscale) / sp;
            auto [dec, ship_mode] = router.plan_and_reserve_transition(
                policy.routing_mode,
                policy.static_und_ranks,
                policy.static_gen_ranks,
                req,
                est_dit_res_ms,
                &workers,
                busy_at_dit,
                policy.use_h2d_pacer,
                dit_ready_ms);
            const int dst_gpu = dec.dst_gpu;
            router.release_dit_reservation(dst_gpu, est_dit_res_ms, &workers);
            if (ship_mode != "none" && dst_gpu != p.cur_gpu) {
                router.release_kv_reservation(dst_gpu, req.effective_state_bytes(), std::nullopt, std::nullopt, &workers);
            }

            if (ship_mode == "shm_static") {
                const double shm_tot_ms =
                    (ph.twostage_shm_base_ms + ph.twostage_shm_ms_per_mib * p.dit_phys_kv) /
                    std::sqrt(std::max(1.0, sp));
                // `SEND_SHM` performs 2 host memory passes (`D2H .cpu()` + `/dev/shm` `memcpy`), while `RECV_SHM`
                // performs 1 pass (`torch.frombuffer` zero-copy view + `H2D .to(device)`): 0.64 vs 0.36 split.
                const double send_ms = 0.71 * shm_tot_ms;
                const double recv_ms = 0.29 * shm_tot_ms;
                const bool release_src = is_tight_mot ? true : (!has_more_turns);
                const int64_t inflight_b = req.effective_state_bytes();
                // Reserve `dst_gpu` queue & KV immediately (`cluster_engine.py:836`) so concurrent SHM sends balance across gen GPUs!
                queued_work_ms[static_cast<size_t>(dst_gpu)] += est_dit_res_ms;
                inflight_ship_kv_bytes[static_cast<size_t>(dst_gpu)] += inflight_b;
                push_ev(
                    std::max(dit_ready_ms, gpu_free_at[static_cast<size_t>(p.cur_gpu)]),
                    EV_SHM_SEND,
                    sid,
                    EvShmSend{p, dst_gpu, send_ms, recv_ms, est_dit_res_ms, inflight_b, release_src},
                    dit_ready_ms);
                continue;
            }

            double wire_ready_ms = dit_ready_ms;
            double pre_denoise_gpu_ms = p.deferred_dit_prep_ms;
            int64_t inflight_b = 0;
            double unpack_staging_mib = 0.0;
            const bool run_pre_vae = (ship_mode == "nccl_static" && p.deferred_dit_prep_ms > 0.0);

            if (ship_mode == "nccl_dual" || ship_mode == "nccl_static") {
                const size_t usrc = static_cast<size_t>(p.cur_gpu);
                const double state_on_wire_mib = p.deferred_full_turn ? std::max(4.0, sess_main_kv_mib[usid]) : p.cand_main_kv;
                const double pack_ms = (0.05 * state_on_wire_mib) / sp;
                const double src_pack_done = dit_ready_ms + pack_ms;
                const bool release_src =
                    (ship_mode == "nccl_dual") ? true : (is_tight_mot ? true : (!has_more_turns));
                if (release_src) {
                    ledgers[usrc].release_session(sid);
                } else {
                    ledgers[usrc].upsert_session_kv(sid, p.cand_main_kv);
                }
                const int src_node = topo.node_of(p.cur_gpu);
                const double r_wait = (dec.hop_tier == "inter_node")
                                          ? std::max(0.0, rdma_free_at[static_cast<size_t>(src_node)] - src_pack_done)
                                          : 0.0;
                const double bw = topo.effective_p2p_gbps(p.cur_gpu, dst_gpu);
                const double wire_ms =
                    ((state_on_wire_mib * static_cast<double>(kMiB) / 1e9) / std::max(0.5, bw)) * 1000.0 +
                    topo.handoff_ms(p.cur_gpu, dst_gpu);
                if (!topo.same_node(p.cur_gpu, dst_gpu)) {
                    rdma_free_at[static_cast<size_t>(src_node)] =
                        std::max(src_pack_done, rdma_free_at[static_cast<size_t>(src_node)]) + wire_ms;
                }
                wire_ready_ms = src_pack_done + r_wait + wire_ms;
                inflight_b = req.effective_state_bytes();
                inflight_ship_kv_bytes[static_cast<size_t>(dst_gpu)] += inflight_b;
                unpack_staging_mib = 0.0;
            } else if (!policy.use_naive_offload) {
                const double morph_ovh = std::max(0.0, dec.t_morph_ms - workers.at(p.cur_gpu).queue_wait_ms);
                pre_denoise_gpu_ms += morph_ovh;
                if (policy.routing_mode == "dual_path") {
                    ++sess_morphs[usid];
                }
            }

            queued_work_ms[static_cast<size_t>(dst_gpu)] += est_dit_res_ms;
            push_ev(
                std::max(wire_ready_ms, excl_gpu_free_at(static_cast<size_t>(dst_gpu))),
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
                    p.dit_steps,
                    p.n_turns,
                    p.cand_main_kv,
                    p.dit_phys_kv,
                    p.dit_core_ms,
                    p.vae_dec_ms,
                    pre_denoise_gpu_ms,
                    est_dit_res_ms,
                    inflight_b,
                    unpack_staging_mib,
                    run_pre_vae},
                wire_ready_ms);
            continue;
        }

        if (cur_ev.kind == EV_SHM_SEND) {
            const auto s = std::get<EvShmSend>(cur_ev.payload);
            const size_t usrc = static_cast<size_t>(s.r.cur_gpu);
            if (excl_gpu_free_at(usrc) > ev_ms + 1e-6) {
                push_ev(excl_gpu_free_at(usrc), EV_SHM_SEND, sid, s, cur_ev.fifo_enq_ms, cur_ev.fifo_seq);
                continue;
            }
            // `pack_bagel_multi_context(..., include_cfg_dup=True)` (`bagel_adapter.py:318-354`) packs `main + cfg_text + cfg_img`
            // (`~1.25 * dit_phys_kv`) into `flat_chunks` AND `torch.cat(flat_chunks)` (`2.4 * dit_phys_kv` contiguous staging)!
            if (!ledgers[usrc].check_transient_spike(2.4 * s.r.dit_phys_kv, sid, /*contiguous_alloc=*/true)) {
                queued_work_ms[static_cast<size_t>(s.dst_gpu)] =
                    std::max(0.0, queued_work_ms[static_cast<size_t>(s.dst_gpu)] - s.dst_est_q);
                inflight_ship_kv_bytes[static_cast<size_t>(s.dst_gpu)] =
                    std::max<int64_t>(0, inflight_ship_kv_bytes[static_cast<size_t>(s.dst_gpu)] - s.dst_inflight_b);
                ledgers[usrc].release_session(sid);
                if (s.r.attempt < policy.max_oom_retries && s.r.ti == 0) {
                    ++sess_retries[usid];
                    sess_gpu[usid] = pick_oom_retry_gpu(s.r.cur_gpu, ev_ms);
                    push_ev(ev_ms, EV_TURN, sid, EvTurn{0, s.r.attempt + 1});
                } else {
                    record_oom(usid, s.r.ti, ev_ms);
                }
                continue;
            }
            const double end_send = ev_ms + s.send_ms;
            gpu_free_at[usrc] = end_send;
            if (s.release_src) {
                ledgers[usrc].release_session(sid);
            } else {
                ledgers[usrc].upsert_session_kv(sid, s.r.cand_main_kv);
            }
            push_ev(
                std::max(end_send, excl_gpu_free_at(static_cast<size_t>(s.dst_gpu))),
                EV_DIT_START,
                sid,
                EvDitStart{
                    s.r.ti,
                    s.r.attempt,
                    s.r.cur_gpu,
                    s.dst_gpu,
                    "shm_static",
                    s.r.t_turn0,
                    s.r.height,
                    s.r.dit_steps,
                    s.r.n_turns,
                    s.r.cand_main_kv,
                    s.r.dit_phys_kv,
                    s.r.dit_core_ms,
                    s.r.vae_dec_ms,
                    s.recv_ms + s.r.deferred_dit_prep_ms,
                    s.dst_est_q,
                    s.dst_inflight_b,
                    s.r.dit_phys_kv,
                    false},
                end_send);
            continue;
        }

        if (cur_ev.kind == EV_DIT_START) {
            auto p = std::get<EvDitStart>(cur_ev.payload);
            const size_t udst = static_cast<size_t>(p.dst_gpu);
            if (excl_gpu_free_at(udst) > ev_ms + 1e-6) {
                push_ev(excl_gpu_free_at(udst), EV_DIT_START, sid, p, cur_ev.fifo_enq_ms, cur_ev.fifo_seq);
                continue;
            }
            inflight_ship_kv_bytes[udst] = std::max<int64_t>(0, inflight_ship_kv_bytes[udst] - p.inflight_kv_b);
            p.inflight_kv_b = 0;

            const double start_dit = ev_ms;
            if (!ledgers[udst].upsert_session_kv(sid, p.dit_phys_kv)) {
                queued_work_ms[udst] = std::max(0.0, queued_work_ms[udst] - p.reserved_est_ms);
                ledgers[udst].release_session(sid);
                ledgers[static_cast<size_t>(p.cur_gpu)].release_session(sid);
                if (p.attempt < policy.max_oom_retries && p.ti == 0) {
                    ++sess_retries[usid];
                    sess_gpu[usid] = pick_oom_retry_gpu(p.dst_gpu, start_dit);
                    push_ev(start_dit, EV_TURN, sid, EvTurn{p.ti, p.attempt + 1});
                } else {
                    record_oom(usid, p.ti, start_dit);
                }
                continue;
            }

            if (p.ship_mode == "nccl_dual" || p.ship_mode == "nccl_static" || p.ship_mode == "shm_static") {
                ++sess_ships[usid];
                if (!topo.same_node(p.cur_gpu, p.dst_gpu)) {
                    ++sess_xnode_ships[usid];
                }
                if (p.ship_mode == "nccl_dual") {
                    sess_gpu[usid] = p.dst_gpu;
                }
            }

            double naive_dit_swap_ms = 0.0;
            if (policy.use_naive_offload) {
                // `NaiveHostOffloader`: synchronous `.to(device)` H2D load before DiT + `drop_from_gpu()` after DiT.
                naive_dit_swap_ms = 1.08 * compute_naive_swap_ms(p.dst_gpu, start_dit + p.pre_denoise_gpu_ms);
            }

            const auto [dit_ok_mem, swap_out_mib] = ledgers[udst].execute_dit_and_vae(
                p.height, sid, p.cand_main_kv, p.unpack_staging_mib, p.run_pre_denoise_vae);
            if (!dit_ok_mem) {
                queued_work_ms[udst] = std::max(0.0, queued_work_ms[udst] - p.reserved_est_ms);
                const double fail_frac = ledgers[udst].last_fail_dit_frac();
                const double vae_wasted_ms = (fail_frac >= 0.99) ? (0.5 * p.vae_dec_ms + (350.0 / sp)) : 0.0;
                const double abort_end =
                    start_dit + p.pre_denoise_gpu_ms + naive_dit_swap_ms + p.dit_core_ms * fail_frac + vae_wasted_ms;
                gpu_free_at[udst] = std::max(gpu_free_at[udst], abort_end);
                ledgers[udst].release_session(sid);
                ledgers[static_cast<size_t>(p.cur_gpu)].release_session(sid);
                if (p.attempt < policy.max_oom_retries && p.ti == 0) {
                    ++sess_retries[usid];
                    sess_gpu[usid] = pick_oom_retry_gpu(p.dst_gpu, abort_end);
                    push_ev(abort_end, EV_TURN, sid, EvTurn{p.ti, p.attempt + 1});
                } else {
                    record_oom(usid, p.ti, abort_end);
                }
                continue;
            }

            const double swap_out_ms =
                (swap_out_mib > 0.0)
                    ? (((swap_out_mib * static_cast<double>(kMiB) / 1e9) / std::max(0.5, topo.solo_h2d_gbps(p.dst_gpu))) *
                       1000.0)
                    : 0.0;
            // Unscoped PyTorch `CUDACachingAllocator` (`!use_tripool_vmm && !use_scoped_vae && !use_naive_offload`) on
            // persistent-context workers (`static_gen_ranks.empty()`, unlike disaggregated `gen` workers that call `RELEASE_CTX`
            // -> `empty_cache()`) triggers synchronous `release_cached_blocks()` once Little's-Law concurrency saturates the
            // cluster (`cluster_active_sessions >= 3 * num_gpus + 3` and `>= 3` sessions on `udst` above 84% HBM).
            size_t cluster_active_sessions = 0;
            for (const auto& lg_ref : ledgers) {
                cluster_active_sessions += lg_ref.active_session_count();
            }
            const double un_vae_est =
                ((p.height >= 1024) ? ph.unscoped_vae_1024_reserved_mib : ph.unscoped_vae_512_reserved_mib) *
                ws.vae_spike_scale;
            const double pressure_over_84_mib =
                (ledgers[udst].base_weights_mib() + ledgers[udst].active_kv_mib() + un_vae_est) - (0.84 * hbm_mib);
            const double alloc_compact_ms =
                (!policy.use_tripool_vmm && !policy.use_scoped_vae && !policy.use_naive_offload &&
                 policy.static_gen_ranks.empty() &&
                 cluster_active_sessions >= static_cast<size_t>(3 * num_gpus + 3) &&
                 ledgers[udst].active_session_count() >= 3 &&
                 ledgers[udst].active_kv_mib() > 1650.0 && pressure_over_84_mib > 0.0)
                    ? ((std::min(9600.0, 2.50 * pressure_over_84_mib) * cscale) / sp)
                    : 0.0;
            const double denoise_start = start_dit + p.pre_denoise_gpu_ms + naive_dit_swap_ms + swap_out_ms;
            const double denoise_end = denoise_start + p.dit_core_ms;
            const double end_dit = denoise_end + p.vae_dec_ms + alloc_compact_ms;

            // BAGEL `cfg_interval = [0.4, 1.0]` (`bagel_adapter.py:1526`): at `1024x1024`, first 40% of steps run 1 unconditional
            // branch and remaining 60% run CFG branches; at `512x512`, `dit_512_cfg_step_mean_ms` (`module4-report.json`) is uniform.
            const int tot_steps = std::max(1, p.dit_steps);
            const double avg_step_ms = p.dit_core_ms / static_cast<double>(tot_steps);
            const int uncond_steps = static_cast<int>(std::floor(0.4 * static_cast<double>(tot_steps)));
            const int cfg_steps = std::max(0, tot_steps - uncond_steps);
            const double cfg_branches = (p.cand_main_kv > 220.0 * ws.kv_per_turn_scale) ? 3.0 : 2.0;
            const double tot_branch_passes =
                static_cast<double>(uncond_steps) * 1.0 + static_cast<double>(cfg_steps) * cfg_branches;
            const double one_branch_ms = p.dit_core_ms / std::max(1.0, tot_branch_passes);

            dit_denoise_start[udst] = denoise_start;
            dit_denoise_cfg_start[udst] = denoise_start + static_cast<double>(uncond_steps) * one_branch_ms;
            dit_denoise_end[udst] = denoise_end;
            dit_full_end[udst] = end_dit;
            dit_active_uncond_step_ms[udst] = (p.height >= 1024) ? one_branch_ms : avg_step_ms;
            dit_active_cfg_step_ms[udst] = (p.height >= 1024) ? (cfg_branches * one_branch_ms) : avg_step_ms;
            dit_active_sid[udst] = sid;
            dit_active_is1024[udst] = (p.height >= 1024);
            sess_dit_actual_end[usid] = end_dit;
            gpu_free_at[udst] = std::max(gpu_free_at[udst], end_dit);
            workers.at(p.dst_gpu).expert_alpha = ledgers[udst].expert_alpha();

            push_ev(
                end_dit,
                EV_DIT_END,
                sid,
                EvDitEnd{
                    p.ti,
                    p.attempt,
                    p.cur_gpu,
                    p.dst_gpu,
                    p.ship_mode,
                    p.t_turn0,
                    p.height,
                    p.dit_steps,
                    p.n_turns,
                    p.cand_main_kv,
                    p.reserved_est_ms});
            continue;
        }

        if (cur_ev.kind == EV_DIT_END) {
            auto p = std::get<EvDitEnd>(cur_ev.payload);
            if (sess_dit_actual_end[usid] > ev_ms + 1e-6) {
                push_ev(sess_dit_actual_end[usid], EV_DIT_END, sid, p);
                continue;
            }
            const double end_dit = ev_ms;
            const size_t udst = static_cast<size_t>(p.dst_gpu);
            queued_work_ms[udst] = std::max(0.0, queued_work_ms[udst] - p.reserved_est_ms);
            p.reserved_est_ms = 0.0;
            if (dit_active_sid[udst] == sid) {
                dit_active_sid[udst] = -1;
            }
            const bool has_more_turns = (p.ti + 1) < p.n_turns;
            const bool ephemeral_gen =
                (p.ship_mode == "nccl_static" || p.ship_mode == "shm_static" || p.ship_mode == "shm_static_recvd");
            const bool needs_return_ship = ephemeral_gen && has_more_turns && is_tight_mot;

            if (ephemeral_gen && !needs_return_ship) {
                ledgers[udst].release_session(sid);
            }

            if (needs_return_ship) {
                if (p.ship_mode == "nccl_static") {
                    ledgers[udst].release_session(sid);
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
                    const double wire_done = end_dit + ret_r_wait + ret_wire_ms;
                    push_ev(
                        std::max(wire_done, gpu_free_at[static_cast<size_t>(p.cur_gpu)]),
                        EV_RET_UNPACK,
                        sid,
                        EvRetUnpack{
                            p.ti,
                            p.attempt,
                            p.cur_gpu,
                            p.dst_gpu,
                            p.t_turn0,
                            p.height,
                            p.dit_steps,
                            p.n_turns,
                            p.cand_main_kv,
                            0.0},
                        wire_done);
                } else {
                    const double ret_shm_ms =
                        (ph.twostage_shm_base_ms + ph.twostage_shm_ms_per_mib * p.cand_main_kv) /
                        std::sqrt(std::max(1.0, sp));
                    push_ev(
                        std::max(end_dit, gpu_free_at[udst]),
                        EV_RET_SHM_SEND,
                        sid,
                        EvRetShmSend{p, 0.58 * ret_shm_ms, 0.42 * ret_shm_ms},
                        end_dit);
                }
            } else {
                sess_kv_mib[usid] = p.cand_main_kv;
                sess_main_kv_mib[usid] = p.cand_main_kv;
                const double turn_lat = end_dit - p.t_turn0;
                sess_dit_records[usid].push_back(SimDitRec{turn_lat, p.height, p.dit_steps});
                SimTurnRec rec;
                rec.turn_idx = p.ti;
                rec.modality_plan = p.plan;
                rec.status = "ok";
                rec.dit_ms = turn_lat;
                rec.height = p.height;
                rec.dit_steps = p.dit_steps;
                rec.kv_mib = p.cand_main_kv;
                if (sess_pending_cot_ar[usid].valid && sess_pending_cot_ar[usid].turn_idx == p.ti) {
                    rec.ar_ttft_ms = sess_pending_cot_ar[usid].ttft_ms;
                    rec.ar_tpots_ms = std::move(sess_pending_cot_ar[usid].tpots_ms);
                    sess_pending_cot_ar[usid].valid = false;
                }
                sess_turn_records[usid].push_back(std::move(rec));
                const double turn_slo_ms = (policy.dit_turn_slo_for(p.height, p.dit_steps) * cscale) / sp;
                if (turn_lat > turn_slo_ms) {
                    sess_dit_ok[usid] = false;
                }
                if (has_more_turns) {
                    sess_gpu[usid] = ephemeral_gen ? p.cur_gpu : p.dst_gpu;
                    const double think_ms = compute_think_ms(usid, p.ti + 1);
                    sess_think_ms[usid] += think_ms;
                    push_ev(end_dit + think_ms, EV_TURN, sid, EvTurn{p.ti + 1, 0});
                } else {
                    sess_end_ms[usid] = end_dit;
                    if (!ephemeral_gen) {
                        push_ev(end_dit, EV_RELEASE, sid, std::monostate{});
                    }
                }
            }
            continue;
        }

        if (cur_ev.kind == EV_RET_SHM_SEND) {
            const auto rs = std::get<EvRetShmSend>(cur_ev.payload);
            const size_t udst = static_cast<size_t>(rs.e.dst_gpu);
            if (gpu_free_at[udst] > ev_ms + 1e-6) {
                push_ev(gpu_free_at[udst], EV_RET_SHM_SEND, sid, rs, cur_ev.fifo_enq_ms, cur_ev.fifo_seq);
                continue;
            }
            if (!ledgers[udst].check_transient_spike(2.0 * rs.e.cand_main_kv, sid, /*contiguous_alloc=*/true)) {
                ledgers[udst].release_session(sid);
                ledgers[static_cast<size_t>(rs.e.cur_gpu)].release_session(sid);
                if (rs.e.attempt < policy.max_oom_retries && rs.e.ti == 0) {
                    ++sess_retries[usid];
                    sess_gpu[usid] = pick_oom_retry_gpu(rs.e.dst_gpu, ev_ms);
                    push_ev(ev_ms, EV_TURN, sid, EvTurn{0, rs.e.attempt + 1});
                } else {
                    record_oom(usid, rs.e.ti, ev_ms);
                }
                continue;
            }
            const double end_send = ev_ms + rs.send_ms;
            gpu_free_at[udst] = end_send;
            ledgers[udst].release_session(sid);
            push_ev(
                std::max(end_send, gpu_free_at[static_cast<size_t>(rs.e.cur_gpu)]),
                EV_RET_UNPACK,
                sid,
                EvRetUnpack{
                    rs.e.ti,
                    rs.e.attempt,
                    rs.e.cur_gpu,
                    rs.e.dst_gpu,
                    rs.e.t_turn0,
                    rs.e.height,
                    rs.e.dit_steps,
                    rs.e.n_turns,
                    rs.e.cand_main_kv,
                    rs.recv_ms},
                end_send);
            continue;
        }

        if (cur_ev.kind == EV_RET_UNPACK) {
            const auto p = std::get<EvRetUnpack>(cur_ev.payload);
            const size_t ucur = static_cast<size_t>(p.cur_gpu);
            if (p.recv_gpu_ms > 0.0 && gpu_free_at[ucur] > ev_ms + 1e-6) {
                push_ev(gpu_free_at[ucur], EV_RET_UNPACK, sid, p, cur_ev.fifo_enq_ms, cur_ev.fifo_seq);
                continue;
            }
            const double turn_end_ms = ev_ms + p.recv_gpu_ms;
            if (p.recv_gpu_ms > 0.0) {
                gpu_free_at[ucur] = std::max(gpu_free_at[ucur], turn_end_ms);
                // `RECV_SHM` on `und` rank stages `flat.to(device)` + `unpack_state` (`2.0 * cand_main_kv`) before `del flat`.
                if (!ledgers[ucur].check_transient_spike(2.0 * p.cand_main_kv, sid, /*contiguous_alloc=*/true)) {
                    ledgers[ucur].release_session(sid);
                    if (p.attempt < policy.max_oom_retries && p.ti == 0) {
                        ++sess_retries[usid];
                        sess_gpu[usid] = pick_oom_retry_gpu(p.cur_gpu, turn_end_ms);
                        push_ev(turn_end_ms, EV_TURN, sid, EvTurn{0, p.attempt + 1});
                    } else {
                        record_oom(usid, p.ti, turn_end_ms);
                    }
                    continue;
                }
            }
            if (!ledgers[ucur].upsert_session_kv(sid, p.cand_main_kv)) {
                ledgers[ucur].release_session(sid);
                if (p.attempt < policy.max_oom_retries && p.ti == 0) {
                    ++sess_retries[usid];
                    sess_gpu[usid] = pick_oom_retry_gpu(p.cur_gpu, turn_end_ms);
                    push_ev(turn_end_ms, EV_TURN, sid, EvTurn{0, p.attempt + 1});
                } else {
                    record_oom(usid, p.ti, turn_end_ms);
                }
                continue;
            }
            if (!topo.same_node(p.dst_gpu, p.cur_gpu)) {
                ++sess_xnode_ships[usid];
            }
            sess_kv_mib[usid] = p.cand_main_kv;
            sess_main_kv_mib[usid] = p.cand_main_kv;
            const double turn_lat = turn_end_ms - p.t_turn0;
            sess_dit_records[usid].push_back(SimDitRec{turn_lat, p.height, p.dit_steps});
            SimTurnRec rec;
            rec.turn_idx = p.ti;
            if (p.ti >= 0 && static_cast<size_t>(p.ti) < built_sessions[usid].turns.size()) {
                rec.modality_plan = built_sessions[usid].turns[static_cast<size_t>(p.ti)].modality_plan;
            }
            rec.status = "ok";
            rec.dit_ms = turn_lat;
            rec.height = p.height;
            rec.dit_steps = p.dit_steps;
            rec.kv_mib = p.cand_main_kv;
            if (sess_pending_cot_ar[usid].valid && sess_pending_cot_ar[usid].turn_idx == p.ti) {
                rec.ar_ttft_ms = sess_pending_cot_ar[usid].ttft_ms;
                rec.ar_tpots_ms = std::move(sess_pending_cot_ar[usid].tpots_ms);
                sess_pending_cot_ar[usid].valid = false;
            }
            sess_turn_records[usid].push_back(std::move(rec));
            const double turn_slo_ms = (policy.dit_turn_slo_for(p.height, p.dit_steps) * cscale) / sp;
            if (turn_lat > turn_slo_ms) {
                sess_dit_ok[usid] = false;
            }
            sess_gpu[usid] = p.cur_gpu;
            const double think_ms = compute_think_ms(usid, p.ti + 1);
            sess_think_ms[usid] += think_ms;
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
        const auto& turn = sess_spec.turns[static_cast<size_t>(ti)];
        const std::string& plan = turn.modality_plan;
        const int height = turn.height;
        const bool is_1024 = height >= 1024;
        const int n_turns = static_cast<int>(sess_spec.turns.size());

        while (true) {
            std::map<int, bool> busy_map;
            std::map<int, int> dit_map;
            std::map<int, int> pre_map;
            sync_states(ev_ms, busy_map, dit_map, pre_map);
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
                    dit_map,
                    pre_map);
                sess_gpu[usid] = cur_gpu;
            }

            if (plan == "ar_only") {
                const size_t ucur = static_cast<size_t>(cur_gpu);
                const double ar_enq = (turn_ev.ar_enq_ms >= 0.0) ? turn_ev.ar_enq_ms : ev_ms;
                if (policy.preemption_mode == "non_preemptive" && gpu_free_at[ucur] > ev_ms + 1e-6) {
                    push_ev(
                        gpu_free_at[ucur],
                        EV_TURN,
                        sid,
                        EvTurn{ti, attempt, ar_enq, true},
                        ar_enq,
                        turn_ev.waiting_ar_lock ? cur_ev.fifo_seq : 0);
                    break;
                }
                const double swap_in_mib = ledgers[ucur].ensure_session_resident(sid);
                if (swap_in_mib < 0.0) {
                    ledgers[ucur].release_session(sid);
                    record_oom(usid, ti, ev_ms);
                    break;
                }
                const double swap_in_ms =
                    (swap_in_mib > 0.0)
                        ? (((swap_in_mib * static_cast<double>(kMiB) / 1e9) / std::max(0.5, topo.solo_h2d_gbps(cur_gpu))) *
                           1000.0)
                        : 0.0;
                const bool prev_was_dit =
                    (ti > 0 && sess_spec.turns[static_cast<size_t>(ti - 1)].modality_plan != "ar_only");
                const bool loopback_prev_dit_img =
                    (turn.num_images == 0) && (turn.use_prev_image || (prev_was_dit && quad == "interleaved_co_creation"));
                const int eff_ar_images = loopback_prev_dit_img ? 1 : turn.num_images;
                const double ar_img_kv = static_cast<double>(eff_ar_images) * 268.0;
                const double ar_delta = (ar_img_kv + 3.5) * ws.kv_per_turn_scale;
                const double new_kv = sess_kv_mib[usid] + ar_delta;
                if (!ledgers[ucur].upsert_session_kv(sid, new_kv)) {
                    ledgers[ucur].release_session(sid);
                    if (attempt < policy.max_oom_retries && ti == 0) {
                        ++attempt;
                        ++sess_retries[usid];
                        sess_gpu[usid] = pick_oom_retry_gpu(cur_gpu, ev_ms);
                        turn_ev.ar_enq_ms = ar_enq;
                        turn_ev.waiting_ar_lock = false;
                        continue;
                    }
                    record_oom(usid, ti, ev_ms);
                    break;
                }
                sess_kv_mib[usid] = new_kv;
                sess_main_kv_mib[usid] += ar_delta;

                const double lock_w = std::max(0.0, ev_ms - ar_enq);
                double ar_ttft_val = 0.0;
                std::vector<double> ar_turn_tp;
                const double end_ms = run_ar_phase_sim(
                    usid,
                    cur_gpu,
                    ev_ms,
                    swap_in_ms,
                    lock_w,
                    turn,
                    /*prepare_for_dit=*/false,
                    loopback_prev_dit_img,
                    &ar_ttft_val,
                    &ar_turn_tp);
                sess_turn_records[usid].push_back(SimTurnRec{
                    ti, "ar_only", "ok", ar_ttft_val, std::move(ar_turn_tp), std::nullopt, 0, 0, sess_main_kv_mib[usid]});
                if (ti + 1 < n_turns) {
                    const double think_ms = compute_think_ms(usid, ti + 1);
                    sess_think_ms[usid] += think_ms;
                    push_ev(end_ms + think_ms, EV_TURN, sid, EvTurn{ti + 1, 0});
                } else {
                    sess_end_ms[usid] = end_ms;
                    push_ev(end_ms, EV_RELEASE, sid, std::monostate{});
                }
                break;
            }

            if (plan == "cot_then_dit") {
                const size_t ucur = static_cast<size_t>(cur_gpu);
                const double ar_enq = (turn_ev.ar_enq_ms >= 0.0) ? turn_ev.ar_enq_ms : ev_ms;
                if (policy.preemption_mode == "non_preemptive" && gpu_free_at[ucur] > ev_ms + 1e-6) {
                    push_ev(
                        gpu_free_at[ucur],
                        EV_TURN,
                        sid,
                        EvTurn{ti, attempt, ar_enq, true},
                        ar_enq,
                        turn_ev.waiting_ar_lock ? cur_ev.fifo_seq : 0);
                    break;
                }
                const double t_turn0 = ar_enq;
                const double swap_in_mib = ledgers[ucur].ensure_session_resident(sid);
                if (swap_in_mib < 0.0) {
                    ledgers[ucur].release_session(sid);
                    record_oom(usid, ti, ev_ms);
                    break;
                }
                double swap_in_ms =
                    (swap_in_mib > 0.0)
                        ? (((swap_in_mib * static_cast<double>(kMiB) / 1e9) / std::max(0.5, topo.solo_h2d_gbps(cur_gpu))) *
                           1000.0)
                        : 0.0;
                const double base_delta_kv =
                    (turn.num_images > 0)
                        ? (ph.turn_edit_512_kv_delta_mib * (is_1024 ? kEdit1024KvRatio : 1.0) + 4.0)
                        : (sess_spec.session_id.empty() ? 185.0 : 8.0);
                const double cand_main_kv = sess_main_kv_mib[usid] + base_delta_kv * ws.kv_per_turn_scale;
                const bool has_hist_for_cfg = (turn.num_images > 0) || (ti > 0);
                const double dit_phys_kv =
                    (!policy.use_module3_cow_loopback && has_hist_for_cfg)
                        ? (cand_main_kv * ph.non_cow_kv_mult)
                        : cand_main_kv;
                const bool is_fission_und =
                    (policy.routing_mode == "static_nccl_fission" && policy.role_of(cur_gpu) == "und");
                const bool pager_partial = policy.use_expert_pager && (ledgers[ucur].expert_alpha() < 0.999);
                const bool can_vae_here = (turn.num_images > 0) && (!is_fission_und) && (!pager_partial);
                const double src_prep_kv = is_fission_und ? cand_main_kv : dit_phys_kv;
                bool cot_mem_ok = ledgers[ucur].upsert_session_kv(sid, src_prep_kv);
                if (cot_mem_ok && can_vae_here) {
                    const double sc_vae = (is_1024 ? ph.scoped_vae_1024_spike_mib : ph.scoped_vae_512_spike_mib) * ws.vae_spike_scale;
                    const double un_vae = (is_1024 ? ph.unscoped_vae_1024_reserved_mib : ph.unscoped_vae_512_reserved_mib) * ws.vae_spike_scale;
                    const double vae_enc_spike = policy.use_scoped_vae ? sc_vae : un_vae;
                    cot_mem_ok = ledgers[ucur].check_transient_spike(
                        vae_enc_spike, sid, /*contiguous_alloc=*/false, std::max(0.0, un_vae - sc_vae));
                }
                if (!cot_mem_ok) {
                    ledgers[ucur].release_session(sid);
                    if (attempt < policy.max_oom_retries && ti == 0) {
                        ++attempt;
                        ++sess_retries[usid];
                        sess_gpu[usid] = pick_oom_retry_gpu(cur_gpu, ev_ms);
                        turn_ev.ar_enq_ms = ar_enq;
                        turn_ev.waiting_ar_lock = false;
                        continue;
                    }
                    record_oom(usid, ti, ev_ms);
                    break;
                }
                if (policy.use_naive_offload && can_vae_here) {
                    swap_in_ms += 1.08 * compute_naive_swap_ms(cur_gpu, ev_ms + swap_in_ms);
                }

                const double lock_w = std::max(0.0, ev_ms - ar_enq);
                double cot_ttft_val = 0.0;
                std::vector<double> cot_turn_tp;
                const double dit_ready_ms = run_ar_phase_sim(
                    usid,
                    cur_gpu,
                    ev_ms,
                    swap_in_ms,
                    lock_w,
                    turn,
                    /*prepare_for_dit=*/true,
                    /*loopback_prev_dit_img=*/false,
                    &cot_ttft_val,
                    &cot_turn_tp);
                sess_pending_cot_ar[usid] = PendingCotAr{true, ti, cot_ttft_val, std::move(cot_turn_tp)};
                const double deferred_vae_ms =
                    (turn.num_images > 0 && !can_vae_here) ? (is_1024 ? vae_encode_1024_ms : vae_encode_512_ms) : 0.0;

                const int main_seq_tokens = static_cast<int>(std::round(
                    (cand_main_kv / std::max(0.01, ws.weight_scale)) / std::max(1e-4, ph.kv_mib_per_token)));
                const bool has_shared_prefix =
                    policy.use_module3_cow_loopback &&
                    (ti > 0 || quad == "visual_cot" || quad == "visual_cot_highres");
                const auto [dit_core_ms, eff_step_ms] =
                    compute_dit_core_ms(cur_gpu, is_1024, (turn.num_images > 0), has_shared_prefix, main_seq_tokens, turn.dit_steps);
                const double step_base = is_1024 ? dit1024_step_ms : dit512_step_ms;
                const double vae_dec_ms = is_1024 ? vae1024_ms : vae512_ms;
                (void)eff_step_ms;

                // D3: Decouple AR CoT latency from DiT stage latency (`dit_ready_ms` is the exact instant
                // AR CoT finishes and DiT routing/transfer/morph/queue/denoising/VAE begins).
                (void)t_turn0;
                push_ev(
                    dit_ready_ms,
                    EV_DIT_READY,
                    sid,
                    EvDitReady{
                        ti,
                        attempt,
                        cur_gpu,
                        dit_ready_ms,
                        quad,
                        plan,
                        height,
                        turn.dit_steps,
                        n_turns,
                        cand_main_kv,
                        dit_phys_kv,
                        step_base,
                        dit_core_ms,
                        vae_dec_ms,
                        deferred_vae_ms,
                        /*deferred_full_turn=*/false});
                break;
            }

            // `dit_edit` or `dit_t2i`: acquire exclusive GPU lock on `cur_gpu` causally (`EV_PREP_START`)
            const double est_prep_q = (240.0 * cscale) / sp;
            queued_work_ms[static_cast<size_t>(cur_gpu)] += est_prep_q;
            push_ev(
                std::max(ev_ms, excl_gpu_free_at(static_cast<size_t>(cur_gpu))),
                EV_PREP_START,
                sid,
                EvPrepStart{ti, attempt, cur_gpu, ev_ms, est_prep_q},
                ev_ms);
            break;
        }
    }

    double max_arrival_s = 0.0;
    for (size_t s = 0; s < n_sess; ++s) {
        const double arr_s = built_sessions[s].arrival_ms / 1000.0;
        if (arr_s > max_arrival_s) max_arrival_s = arr_s;
    }

    std::vector<size_t> core_indices;
    std::string eff_window_mode = "session_trim";
    double win_start_s = 0.0;
    double win_end_s = 0.0;
    if (n_sess > 0 && policy.steady_window_mode == "time" &&
        max_arrival_s >= policy.warmup_window_s + 30.0) {
        const double ws_s = policy.warmup_window_s;
        const double we_s = policy.warmup_window_s + policy.steady_window_s;
        for (size_t s = 0; s < n_sess; ++s) {
            const double arr_s = built_sessions[s].arrival_ms / 1000.0;
            if (arr_s >= ws_s - 1e-9 && arr_s <= we_s + 1e-9) {
                core_indices.push_back(s);
            }
        }
        if (!core_indices.empty()) {
            eff_window_mode = "time";
            win_start_s = ws_s;
            win_end_s = we_s;
        }
    }
    if (core_indices.empty() && n_sess > 0) {
        eff_window_mode = "session_trim";
        const size_t trim =
            (n_sess >= 20) ? static_cast<size_t>(static_cast<double>(n_sess) * policy.steady_state_trim_frac) : 0;
        const size_t core_begin = (trim > 0 && trim * 2 < n_sess) ? trim : 0;
        const size_t core_end = (trim > 0 && trim * 2 < n_sess) ? (n_sess - trim) : n_sess;
        core_indices.reserve(core_end - core_begin);
        for (size_t s = core_begin; s < core_end; ++s) {
            core_indices.push_back(s);
        }
        if (!core_indices.empty()) {
            win_start_s = built_sessions[core_indices.front()].arrival_ms / 1000.0;
            win_end_s = built_sessions[core_indices.back()].arrival_ms / 1000.0;
        }
    }

    std::vector<uint8_t> in_core(n_sess, 0);
    for (size_t idx : core_indices) in_core[idx] = 1;
    const size_t steady_sessions_cnt = core_indices.size();
    const size_t n_core = std::max<size_t>(1, steady_sessions_cnt);

    std::vector<double> core_ttfts;
    std::vector<double> core_tpots;
    std::vector<double> core_dits;
    std::vector<double> core_e2es;
    std::vector<double> core_active_jcts;

    struct SimTurnAcc {
        int total = 0;
        int ok = 0;
        int oom = 0;
        int slo_met = 0;
        std::vector<double> ttfts;
        std::vector<double> tpots;
        std::vector<double> dits;
        std::vector<double> kv_mibs;
    };
    std::map<int, SimTurnAcc> turn_accs;

    auto eval_sim_turn_slo = [&](const SimTurnRec& tr, double ttft_cap, double tpot_cap, std::optional<double> k_dit_opt) -> std::tuple<bool, bool, bool, bool, bool> {
        const bool has_ar = (tr.modality_plan == "ar_only" || tr.modality_plan == "cot_then_dit" || tr.ar_ttft_ms.has_value());
        const bool has_dit = (tr.modality_plan == "dit_t2i" || tr.modality_plan == "dit_edit" || tr.modality_plan == "cot_then_dit" || tr.dit_ms.has_value());
        if (tr.status != "ok") {
            return {has_ar, has_dit, false, false, false};
        }
        bool ar_ok = true;
        if (has_ar) {
            const double ttft_v = tr.ar_ttft_ms.value_or(0.0);
            bool tp_ok = true;
            if (!tr.ar_tpots_ms.empty()) {
                if (policy.ar_tpot_slo_mode == "turn_mean") {
                    tp_ok = mean_raw_sim(tr.ar_tpots_ms) <= tpot_cap;
                } else if (policy.ar_tpot_slo_mode == "token_max" || policy.ar_tpot_slo_mode == "strict_max") {
                    tp_ok = *std::max_element(tr.ar_tpots_ms.begin(), tr.ar_tpots_ms.end()) <= tpot_cap;
                } else {
                    tp_ok = percentile_raw_sim(tr.ar_tpots_ms, 95.0) <= tpot_cap;
                }
            }
            ar_ok = (ttft_v <= ttft_cap) && tp_ok;
        }
        bool dit_ok = true;
        if (has_dit) {
            const int h = (tr.height > 0) ? tr.height : 512;
            const int st = (tr.dit_steps > 0) ? tr.dit_steps : 20;
            const double cap = (policy.dit_turn_slo_for(h, st, k_dit_opt) * cscale) / sp;
            dit_ok = (tr.dit_ms.value_or(0.0) <= cap);
        }
        const bool j_ok = (!has_ar || ar_ok) && (!has_dit || dit_ok);
        return {has_ar, has_dit, ar_ok, dit_ok, j_ok};
    };

    int ships = 0;
    int morphs = 0;
    int cross_node_ships = 0;
    int preemptions = 0;
    size_t core_ok = 0;
    size_t core_surv = 0;
    size_t core_ooms = 0;
    size_t core_ar_ok = 0;
    size_t core_ar_mean_ok = 0;
    size_t core_ar_strict_max_ok = 0;
    size_t core_dit_ok = 0;
    size_t total_planned_turns = 0;
    size_t total_completed_turns = 0;
    size_t total_turn_joint_ok = 0;
    size_t core_planned_turns = 0;
    size_t core_completed_turns = 0;
    size_t core_turn_joint_ok = 0;
    size_t core_planned_ar_turns = 0;
    size_t core_turn_ar_ok = 0;
    size_t core_planned_dit_turns = 0;
    size_t core_turn_dit_ok = 0;
    size_t ooms = 0;
    size_t joint_ok = 0;
    int total_retries = 0;

    for (size_t s = 0; s < n_sess; ++s) {
        total_retries += sess_retries[s];
        const bool core = (in_core[s] != 0);
        const auto& s_spec = built_sessions[s];
        const size_t s_planned = std::max<size_t>(s_spec.turns.size(), sess_turn_records[s].size());
        size_t s_planned_ar = 0;
        size_t s_planned_dit = 0;
        for (const auto& tspec : s_spec.turns) {
            if (tspec.modality_plan == "ar_only" || tspec.modality_plan == "cot_then_dit") ++s_planned_ar;
            if (tspec.modality_plan != "ar_only") ++s_planned_dit;
        }
        total_planned_turns += s_planned;
        if (core) {
            core_planned_turns += s_planned;
            core_planned_ar_turns += s_planned_ar;
            core_planned_dit_turns += s_planned_dit;
        }
        for (const auto& tr : sess_turn_records[s]) {
            const auto [h_ar, h_dit, t_ar_ok, t_dit_ok, t_j_ok] =
                eval_sim_turn_slo(tr, ar_ttft_slo, ar_tpot_slo, std::nullopt);
            if (tr.status == "ok") ++total_completed_turns;
            if (t_j_ok) ++total_turn_joint_ok;
            if (core) {
                if (tr.status == "ok") ++core_completed_turns;
                if (t_j_ok) ++core_turn_joint_ok;
                if (h_ar && t_ar_ok) ++core_turn_ar_ok;
                if (h_dit && t_dit_ok) ++core_turn_dit_ok;
            }
        }
        if (sess_oom[s]) {
            ++ooms;
            if (core) ++core_ooms;
        } else {
            const SimArFlags ar_flags = eval_sim_ar_slos(
                sess_ar_ttfts[s], sess_ar_tpots[s], sess_ar_turn_tpots[s], ar_ttft_slo, ar_tpot_slo);
            const bool ar_primary = select_sim_ar_slo(ar_flags, policy.ar_tpot_slo_mode);
            sess_ar_ok[s] = ar_primary;
            const bool j_ok = ar_primary && sess_dit_ok[s];
            if (j_ok) ++joint_ok;
            ships += sess_ships[s];
            morphs += sess_morphs[s];
            cross_node_ships += sess_xnode_ships[s];
            preemptions += sess_preemptions[s];

            if (core) {
                ++core_surv;
                if (j_ok) ++core_ok;
                if (ar_primary) ++core_ar_ok;
                if (ar_flags.mean_ok) ++core_ar_mean_ok;
                if (ar_flags.strict_max_ok) ++core_ar_strict_max_ok;
                if (sess_dit_ok[s]) ++core_dit_ok;

                const double e2e_ms = std::max(0.0, sess_end_ms[s] - built_sessions[s].arrival_ms);
                const double active_jct_ms = std::max(0.0, e2e_ms - sess_think_ms[s]);
                core_e2es.push_back(e2e_ms);
                core_active_jcts.push_back(active_jct_ms);
                for (double x : sess_ar_ttfts[s]) core_ttfts.push_back(x);
                for (double x : sess_ar_tpots[s]) core_tpots.push_back(x);
                for (const auto& rec : sess_dit_records[s]) core_dits.push_back(rec.lat_ms);
            }
        }
        if (core) {
            for (const auto& tr : sess_turn_records[s]) {
                auto& ta = turn_accs[tr.turn_idx];
                ++ta.total;
                if (tr.status == "ok") {
                    ++ta.ok;
                    const auto [h_ar2, h_dit2, t_ar2, t_dit2, t_j2] =
                        eval_sim_turn_slo(tr, ar_ttft_slo, ar_tpot_slo, std::nullopt);
                    (void)h_ar2;
                    (void)h_dit2;
                    (void)t_ar2;
                    (void)t_dit2;
                    if (t_j2) ++ta.slo_met;
                    if (tr.ar_ttft_ms.has_value()) ta.ttfts.push_back(*tr.ar_ttft_ms);
                    for (double tp : tr.ar_tpots_ms) ta.tpots.push_back(tp);
                    if (tr.dit_ms.has_value()) ta.dits.push_back(*tr.dit_ms);
                    ta.kv_mibs.push_back(tr.kv_mib);
                } else {
                    ++ta.oom;
                }
            }
        }
    }
    const double att_frac = static_cast<double>(core_ok) / static_cast<double>(n_core);
    const double eff_slo_throughput_rps = round_4(offered_qps * att_frac);
    const size_t n_core_turns = std::max<size_t>(1, core_planned_turns);
    const size_t n_tot_turns = std::max<size_t>(1, total_planned_turns);
    const double mean_turns_per_core =
        (steady_sessions_cnt == 0)
            ? 1.0
            : (static_cast<double>(core_planned_turns) / static_cast<double>(steady_sessions_cnt));
    const double steady_turn_att_frac = static_cast<double>(core_turn_joint_ok) / static_cast<double>(n_core_turns);
    const double steady_turn_goodput_rps = round_4(offered_qps * steady_turn_att_frac);
    const double steady_turn_goodput_tps = round_4(offered_qps * mean_turns_per_core * steady_turn_att_frac);
    const double full_turn_att_frac = static_cast<double>(total_turn_joint_ok) / static_cast<double>(n_tot_turns);
    const double steady_turn_surv_pct =
        (core_planned_turns == 0)
            ? 100.0
            : round_1(100.0 * static_cast<double>(core_completed_turns) / static_cast<double>(n_core_turns));
    const double steady_turn_ar_pct =
        (core_planned_ar_turns == 0)
            ? 100.0
            : round_1(100.0 * static_cast<double>(core_turn_ar_ok) / static_cast<double>(core_planned_ar_turns));
    const double steady_turn_dit_pct =
        (core_planned_dit_turns == 0)
            ? 100.0
            : round_1(100.0 * static_cast<double>(core_turn_dit_ok) / static_cast<double>(core_planned_dit_turns));

    nlohmann::json slo_curve_by_k = nlohmann::json::object();
    for (const auto& [k_lbl, k_val] : std::vector<std::pair<std::string, double>>{
             {"1.5x", 1.5}, {"2.0x", 2.0}, {"2.5x", 2.5}, {"3.0x", 3.0}}) {
        const auto slo_k = default_operator_profile().slo_thresholds_ms(k_val, "bagel");
        const double ttft_k = (slo_k.at("ar_ttft_slo_ms") * cscale) / sp;
        const double tpot_k = (slo_k.at("ar_tpot_slo_ms") * cscale) / sp;
        size_t k_ok_cnt = 0;
        size_t k_t_ok = 0;
        for (size_t s : core_indices) {
            for (const auto& tr : sess_turn_records[s]) {
                const auto [ha, hd, aok, dok, tj] = eval_sim_turn_slo(tr, ttft_k, tpot_k, k_val);
                (void)ha; (void)hd; (void)aok; (void)dok;
                if (tj) ++k_t_ok;
            }
            if (sess_oom[s]) continue;
            const SimArFlags k_ar_flags = eval_sim_ar_slos(
                sess_ar_ttfts[s], sess_ar_tpots[s], sess_ar_turn_tpots[s], ttft_k, tpot_k);
            bool ok = select_sim_ar_slo(k_ar_flags, policy.ar_tpot_slo_mode);
            if (ok) {
                for (const auto& rec : sess_dit_records[s]) {
                    if (rec.lat_ms > (policy.dit_turn_slo_for(rec.height, rec.steps, k_val) * cscale) / sp) {
                        ok = false;
                        break;
                    }
                }
            }
            if (ok) ++k_ok_cnt;
        }
        const double k_frac = static_cast<double>(k_ok_cnt) / static_cast<double>(n_core);
        const double k_rps = round_4(offered_qps * k_frac);
        const double kt_frac = static_cast<double>(k_t_ok) / static_cast<double>(n_core_turns);
        slo_curve_by_k[k_lbl] = {
            {"attainment_pct", round_1(100.0 * k_frac)},
            {"effective_slo_throughput_rps", k_rps},
            {"goodput_rps", k_rps},
            {"turn_attainment_pct", round_1(100.0 * kt_frac)},
            {"turn_goodput_rps", round_4(offered_qps * kt_frac)},
        };
    }

    nlohmann::json slo_curve_by_dit_k = nlohmann::json::object();
    for (const auto& [k_lbl, k_dit] : std::vector<std::pair<std::string, double>>{
             {"2.0x", 2.0}, {"3.0x", 3.0}, {"4.0x", 4.0}}) {
        size_t s_ok_cnt = 0;
        size_t t_ok_cnt = 0;
        for (size_t s : core_indices) {
            for (const auto& tr : sess_turn_records[s]) {
                const auto [ha, hd, aok, dok, tj] = eval_sim_turn_slo(tr, ar_ttft_slo, ar_tpot_slo, k_dit);
                (void)ha; (void)hd; (void)aok; (void)dok;
                if (tj) ++t_ok_cnt;
            }
            if (sess_oom[s]) continue;
            const SimArFlags k_ar_flags = eval_sim_ar_slos(
                sess_ar_ttfts[s], sess_ar_tpots[s], sess_ar_turn_tpots[s], ar_ttft_slo, ar_tpot_slo);
            bool ok = select_sim_ar_slo(k_ar_flags, policy.ar_tpot_slo_mode);
            if (ok) {
                for (const auto& rec : sess_dit_records[s]) {
                    if (rec.lat_ms > (policy.dit_turn_slo_for(rec.height, rec.steps, k_dit) * cscale) / sp) {
                        ok = false;
                        break;
                    }
                }
            }
            if (ok) ++s_ok_cnt;
        }
        const double s_frac = static_cast<double>(s_ok_cnt) / static_cast<double>(n_core);
        const double t_frac = static_cast<double>(t_ok_cnt) / static_cast<double>(n_core_turns);
        slo_curve_by_dit_k[k_lbl] = {
            {"turn_attainment_pct", round_1(100.0 * t_frac)},
            {"turn_goodput_rps", round_4(offered_qps * t_frac)},
            {"session_attainment_pct", round_1(100.0 * s_frac)},
            {"session_goodput_rps", round_4(offered_qps * s_frac)},
            {"ar_ttft_slo_ms", round_2(ar_ttft_slo)},
            {"ar_tpot_slo_ms", round_2(ar_tpot_slo)},
            {"dit_512_slo_ms", round_2((policy.dit_turn_slo_for(512, 20, k_dit) * cscale) / sp)},
            {"dit_1024_slo_ms", round_2((policy.dit_turn_slo_for(1024, 20, k_dit) * cscale) / sp)},
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

    const double full_att_frac =
        static_cast<double>(joint_ok) / static_cast<double>(std::max<size_t>(1, n_sess));
    const double survival_pct =
        (n_sess == 0) ? 100.0 : round_1(100.0 * static_cast<double>(n_sess - ooms) / static_cast<double>(n_sess));
    const double steady_survival_pct =
        (steady_sessions_cnt == 0)
            ? 100.0
            : round_1(100.0 * static_cast<double>(core_surv) / static_cast<double>(n_core));

    const size_t total_dit_units = core_dits.size() + core_ooms;
    const double dit_survival_pct =
        (total_dit_units == 0)
            ? 100.0
            : round_1(100.0 * static_cast<double>(core_dits.size()) / static_cast<double>(total_dit_units));
    auto sim_failure_aware_quantile = [](const std::vector<double>& surv, size_t tot, double q_pct) -> nlohmann::json {
        if (tot == 0 || surv.empty()) return nullptr;
        const double s_frac = static_cast<double>(surv.size()) / static_cast<double>(tot);
        const double q_frac = q_pct / 100.0;
        if (q_frac > s_frac + 1e-9) return nullptr;
        return percentile_rounded(surv, std::min(100.0, (q_frac / std::max(1e-9, s_frac)) * 100.0));
    };
    auto build_sim_cdf = [&](const std::vector<double>& surv, size_t tot) -> nlohmann::json {
        nlohmann::json pts = nlohmann::json::array();
        for (double q_pct : {5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0, 90.0, 95.0, 99.0}) {
            pts.push_back({
                {"p", q_pct},
                {"ms", sim_failure_aware_quantile(surv, tot, q_pct)},
            });
        }
        return pts;
    };

    nlohmann::json per_turn_stats = nlohmann::json::object();
    for (const auto& [tidx, ta] : turn_accs) {
        const double t_surv =
            (ta.total == 0) ? 100.0 : round_1(100.0 * static_cast<double>(ta.ok) / static_cast<double>(ta.total));
        const double t_slo_pct =
            (ta.total == 0) ? 0.0 : round_1(100.0 * static_cast<double>(ta.slo_met) / static_cast<double>(ta.total));
        per_turn_stats["turn_" + std::to_string(tidx)] = {
            {"total", ta.total},
            {"ok", ta.ok},
            {"oom", ta.oom},
            {"slo_met", ta.slo_met},
            {"slo_attainment_pct", t_slo_pct},
            {"survival_pct", t_surv},
            {"ar_ttft_p95_ms", percentile_rounded(ta.ttfts, 95.0)},
            {"ar_tpot_p95_ms", percentile_rounded(ta.tpots, 95.0)},
            {"dit_p95_ms", percentile_rounded(ta.dits, 95.0)},
            {"kv_mib_mean", round_2(mean_raw_sim(ta.kv_mibs))},
            {"kv_mib_p95", percentile_rounded(ta.kv_mibs, 95.0)},
        };
    }

    const nlohmann::json slo_config = {
        {"k_ar", policy.slo_scale_k},
        {"k_dit", policy.dit_slo_scale_k},
        {"ar_ttft_slo_ms", round_2(ar_ttft_slo)},
        {"ar_tpot_slo_ms", round_2(ar_tpot_slo)},
        {"ar_tpot_slo_mode", policy.ar_tpot_slo_mode},
        {"dit_512_slo_ms", round_2((policy.dit_turn_slo_for(512, 20) * cscale) / sp)},
        {"dit_1024_slo_ms", round_2((policy.dit_turn_slo_for(1024, 20) * cscale) / sp)},
        {"cot_dit_decoupled", true},
        {"primary_granularity", "turn_level"},
        {"turn_oom_accounting", "planned_turns_denominator"},
    };

    return {
        {"policy_name", policy_name},
        {"workload_scale", ws.name},
        {"skew_tag", skew_tag.empty() ? "main_mix" : skew_tag},
        {"num_gpus", num_gpus},
        {"interconnect", hw.name},
        {"offered_qps", round_3(offered_qps)},
        {"crossover_kv_mib", crossover_kv_mib(hw, ph, ws)},
        {"steady_state_window_mode", eff_window_mode},
        {"steady_window_start_s", round_2(win_start_s)},
        {"steady_window_end_s", round_2(win_end_s)},
        {"steady_state_sessions", steady_sessions_cnt},
        {"steady_state_survival_pct", steady_survival_pct},
        {"steady_oom_sessions", core_ooms},
        {"total_planned_turns", total_planned_turns},
        {"completed_turns", total_completed_turns},
        {"turn_slo_attainment_pct", round_1(100.0 * full_turn_att_frac)},
        {"turn_slo_goodput_rps", round_4(offered_qps * full_turn_att_frac)},
        {"steady_planned_turns", core_planned_turns},
        {"steady_completed_turns", core_completed_turns},
        {"steady_turn_survival_pct", steady_turn_surv_pct},
        {"steady_turn_slo_attainment_pct", round_1(100.0 * steady_turn_att_frac)},
        {"steady_turn_ar_slo_attainment_pct", steady_turn_ar_pct},
        {"steady_turn_dit_slo_attainment_pct", steady_turn_dit_pct},
        {"steady_turn_slo_goodput_rps", steady_turn_goodput_rps},
        {"steady_turn_goodput_tps", steady_turn_goodput_tps},
        {"effective_slo_throughput_rps", eff_slo_throughput_rps},
        {"steady_state_goodput_rps", eff_slo_throughput_rps},
        {"joint_slo_goodput_rps", eff_slo_throughput_rps},
        {"full_trace_goodput_rps", round_4(offered_qps * full_att_frac)},
        {"steady_state_attainment_pct", round_1(100.0 * att_frac)},
        {"joint_slo_attainment_pct", round_1(100.0 * att_frac)},
        {"steady_ar_slo_attainment_pct", round_1(100.0 * static_cast<double>(core_ar_ok) / static_cast<double>(n_core))},
        {"steady_ar_slo_mean_pct", round_1(100.0 * static_cast<double>(core_ar_mean_ok) / static_cast<double>(n_core))},
        {"steady_ar_slo_strict_max_pct",
         round_1(100.0 * static_cast<double>(core_ar_strict_max_ok) / static_cast<double>(n_core))},
        {"steady_dit_slo_attainment_pct",
         round_1(100.0 * static_cast<double>(core_dit_ok) / static_cast<double>(n_core))},
        {"full_trace_attainment_pct", round_1(100.0 * full_att_frac)},
        {"survival_rate_pct", survival_pct},
        {"dit_turn_survival_pct", dit_survival_pct},
        {"oom_sessions", ooms},
        {"oom_retries", total_retries},
        {"peak_hbm_mib", round_1(peak_hbm)},
        {"vmm_carves", carves_tot},
        {"vmm_expert_evictions", evict_tot},
        {"vmm_kv_host_swap_outs", swap_out_tot},
        {"vmm_kv_host_swap_ins", swap_in_tot},
        {"ar_ttft_p50_ms", percentile_rounded(core_ttfts, 50.0)},
        {"ar_ttft_p95_ms", percentile_rounded(core_ttfts, 95.0)},
        {"ar_ttft_p99_ms", percentile_rounded(core_ttfts, 99.0)},
        {"ar_tpot_p50_ms", percentile_rounded(core_tpots, 50.0)},
        {"ar_tpot_p95_ms", percentile_rounded(core_tpots, 95.0)},
        {"ar_tpot_p99_ms", percentile_rounded(core_tpots, 99.0)},
        {"dit_turn_p50_ms", percentile_rounded(core_dits, 50.0)},
        {"dit_turn_p95_ms", percentile_rounded(core_dits, 95.0)},
        {"dit_turn_p99_ms", percentile_rounded(core_dits, 99.0)},
        {"session_e2e_p50_ms", percentile_rounded(core_e2es, 50.0)},
        {"session_e2e_p95_ms", percentile_rounded(core_e2es, 95.0)},
        {"session_e2e_p99_ms", percentile_rounded(core_e2es, 99.0)},
        {"active_jct_p50_ms", percentile_rounded(core_active_jcts, 50.0)},
        {"active_jct_p95_ms", percentile_rounded(core_active_jcts, 95.0)},
        {"active_jct_p99_ms", percentile_rounded(core_active_jcts, 99.0)},
        {"all_session_active_jct_p50_ms", sim_failure_aware_quantile(core_active_jcts, steady_sessions_cnt, 50.0)},
        {"all_session_active_jct_p90_ms", sim_failure_aware_quantile(core_active_jcts, steady_sessions_cnt, 90.0)},
        {"all_session_active_jct_p95_ms", sim_failure_aware_quantile(core_active_jcts, steady_sessions_cnt, 95.0)},
        {"all_session_e2e_p50_ms", sim_failure_aware_quantile(core_e2es, steady_sessions_cnt, 50.0)},
        {"all_session_e2e_p90_ms", sim_failure_aware_quantile(core_e2es, steady_sessions_cnt, 90.0)},
        {"all_session_e2e_p95_ms", sim_failure_aware_quantile(core_e2es, steady_sessions_cnt, 95.0)},
        {"all_session_dit_p50_ms", sim_failure_aware_quantile(core_dits, total_dit_units, 50.0)},
        {"all_session_dit_p90_ms", sim_failure_aware_quantile(core_dits, total_dit_units, 90.0)},
        {"all_session_dit_p95_ms", sim_failure_aware_quantile(core_dits, total_dit_units, 95.0)},
        {"active_jct_cdf_points", build_sim_cdf(core_active_jcts, steady_sessions_cnt)},
        {"e2e_cdf_points", build_sim_cdf(core_e2es, steady_sessions_cnt)},
        {"dit_cdf_points", build_sim_cdf(core_dits, total_dit_units)},
        {"path_a_ships", ships},
        {"path_b_morphs", morphs},
        {"preempted_ar_count", preemptions},
        {"cross_node_ships", cross_node_ships},
        {"slo_config", slo_config},
        {"slo_curve_by_k", slo_curve_by_k},
        {"slo_curve_by_dit_k", slo_curve_by_dit_k},
        {"per_turn_stats", per_turn_stats},
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
        nlohmann::json ref_ov = nlohmann::json::object();
        for (const char* opt_k : {"slo_scale_k",
                                  "dit_slo_scale_k",
                                  "inter_turn_think_s_range",
                                  "think_time_dist",
                                  "think_time_median_s",
                                  "think_time_sigma",
                                  "ar_tpot_slo_mode",
                                  "steady_window_mode",
                                  "warmup_window_s",
                                  "steady_window_s",
                                  "steady_state_trim_frac"}) {
            if (ref.contains(opt_k) && !ref[opt_k].is_null()) {
                ref_ov[opt_k] = ref[opt_k];
            }
        }
        if (ref.contains("slo_config") && ref["slo_config"].is_object()) {
            const auto& sc = ref["slo_config"];
            if (sc.contains("k_ar") && sc["k_ar"].is_number() && !ref_ov.contains("slo_scale_k")) {
                ref_ov["slo_scale_k"] = sc["k_ar"];
            }
            if (sc.contains("k_dit") && sc["k_dit"].is_number() && !ref_ov.contains("dit_slo_scale_k")) {
                ref_ov["dit_slo_scale_k"] = sc["k_dit"];
            }
        } else if (!ref_ov.contains("dit_slo_scale_k") && !ref.contains("steady_turn_slo_goodput_rps")) {
            // Legacy v1 live reports evaluated DiT against uniform `k=2.0`.
            ref_ov["dit_slo_scale_k"] = 2.0;
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
            sources_dir,
            nullptr,
            ref_ov);

        double real_gp = 0.0;
        double sim_gp = 0.0;
        if (ref.contains("steady_turn_slo_goodput_rps") && ref["steady_turn_slo_goodput_rps"].is_number() &&
            ref["steady_turn_slo_goodput_rps"].get<double>() > 0.0) {
            real_gp = ref["steady_turn_slo_goodput_rps"].get<double>();
            sim_gp = sim.at("steady_turn_slo_goodput_rps").get<double>();
        } else if (ref.contains("effective_slo_throughput_rps") && ref["effective_slo_throughput_rps"].is_number() &&
                   ref["effective_slo_throughput_rps"].get<double>() > 0.0 &&
                   ref.value("steady_state_window_mode", "") == "time") {
            real_gp = ref["effective_slo_throughput_rps"].get<double>();
            sim_gp = sim.at("effective_slo_throughput_rps").get<double>();
        } else if (ref.contains("joint_slo_goodput_rps") && ref["joint_slo_goodput_rps"].is_number() &&
                   ref["joint_slo_goodput_rps"].get<double>() > 0.0) {
            real_gp = ref["joint_slo_goodput_rps"].get<double>();
            sim_gp = sim.at("full_trace_goodput_rps").get<double>();
        } else {
            real_gp = ref.value("steady_state_goodput_rps", 0.0);
            sim_gp = sim.at("joint_slo_goodput_rps").get<double>();
        }
        real_gp = std::max(1e-4, real_gp);

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

    // Full 9-point low-to-saturation `SLO Attainment (%) vs. QPS` curve (`0.05 -> 1.40` RPS) + `@80%/@90%` sustainable QPS.
    const auto a100_pcie = presets.front();
    nlohmann::json slo_vs_qps_8gpu = nlohmann::json::object();
    for (const char* pol : kValidPolicies) {
        nlohmann::json pts = nlohmann::json::array();
        std::vector<std::pair<double, double>> full_curve;
        std::vector<std::pair<double, double>> steady_curve;
        for (double q : {0.05, 0.10, 0.15, 0.20, 0.25, 0.40, 0.80, 1.10, 1.40}) {
            const int ns = (q <= 0.10) ? 28 : ((q <= 0.25) ? 48 : 96);
            auto sr = simulate_cluster_point(
                pol, 8, a100_pcie, q, ns, 2026, ph, WorkloadScaleSpec::bagel_7b_standard(), "", bench_root, sources_dir);
            full_curve.emplace_back(q, sr.at("full_trace_attainment_pct").get<double>());
            steady_curve.emplace_back(q, sr.at("joint_slo_attainment_pct").get<double>());
            pts.push_back(std::move(sr));
        }
        auto interp_qps = [](const std::vector<std::pair<double, double>>& cv, double thr) -> double {
            double best = 0.0;
            for (size_t i = 0; i < cv.size(); ++i) {
                const auto [q_cur, a_cur] = cv[i];
                if (a_cur >= thr) {
                    best = std::max(best, q_cur);
                    if (i + 1 < cv.size() && cv[i + 1].second < thr && a_cur > cv[i + 1].second) {
                        const double frac = (a_cur - thr) / (a_cur - cv[i + 1].second);
                        best = std::max(best, q_cur + frac * (cv[i + 1].first - q_cur));
                    }
                }
            }
            return round_4(best);
        };
        slo_vs_qps_8gpu[pol] = {
            {"max_sustainable_qps_80pct", interp_qps(full_curve, 80.0)},
            {"max_sustainable_qps_90pct", interp_qps(full_curve, 90.0)},
            {"steady_max_sustainable_qps_80pct", interp_qps(steady_curve, 80.0)},
            {"steady_max_sustainable_qps_90pct", interp_qps(steady_curve, 90.0)},
            {"points", std::move(pts)},
        };
    }

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
        {"slo_attainment_vs_qps_8gpu", slo_vs_qps_8gpu},
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
