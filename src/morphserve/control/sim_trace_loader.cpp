#include "morphserve/control/sim_trace_loader.hpp"

#include "morphserve/control/hash_util.hpp"
#include "morphserve/control/simulator.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <unordered_map>

namespace morphserve::control {

namespace {

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

void append_extra_turns(std::vector<SimTurnSpec>& turns, const std::string& quadrant, int extra_edit_turns) {
    if (turns.empty() || extra_edit_turns <= 0) {
        return;
    }
    if (quadrant == "multiturn_edit") {
        const SimTurnSpec last = turns.back();
        for (int e = 0; e < extra_edit_turns; ++e) turns.push_back(last);
    } else if (quadrant == "interleaved_co_creation" && turns.size() >= 2) {
        const SimTurnSpec prev_a = turns[turns.size() - 2];
        const SimTurnSpec prev_b = turns[turns.size() - 1];
        for (int e = 0; e < extra_edit_turns; ++e) {
            turns.push_back((e % 2 == 0) ? prev_a : prev_b);
        }
    } else if (extra_edit_turns >= 2 &&
               (quadrant == "visual_cot" || quadrant == "visual_cot_highres" || quadrant == "multi_image_vqa")) {
        const SimTurnSpec last = turns.back();
        for (int e = 0; e < extra_edit_turns / 2; ++e) turns.push_back(last);
    }
}

}  // namespace

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

std::vector<SimSessionSpec> build_sim_session_stream(
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
            append_extra_turns(sc.turns, sc.quadrant, extra_edit_turns);
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
        append_extra_turns(turns, q, extra_edit_turns);
        if (idx > 0) {
            const int m5 = idx % 5;
            const double burst_factor = (m5 == 1 || m5 == 2) ? 0.35 : ((m5 == 0) ? 1.85 : 0.90);
            t_cur += mean_inter_ms * burst_factor * dist(rng);
        }
        out.push_back(SimSessionSpec{t_cur, q, std::move(turns), "", ""});
    }
    return out;
}

std::vector<SimSessionSpec> ClusterSimulator::build_session_stream(
    int num_sessions,
    double offered_qps,
    int num_gpus,
    uint64_t seed,
    int extra_edit_turns,
    const std::string& skew_tag,
    const std::string& bench_root,
    const std::string& sources_dir) {
    return build_sim_session_stream(
        num_sessions, offered_qps, num_gpus, seed, extra_edit_turns, skew_tag, bench_root, sources_dir);
}

}  // namespace morphserve::control
