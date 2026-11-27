#include "morphserve/control/performance_atlas.hpp"

#include <algorithm>
#include <cmath>
#include <set>

namespace morphserve::control {

nlohmann::json AtlasMorphEntry::to_json() const {
    return {
        {"model_family", model_family},
        {"resolution", resolution},
        {"alpha_g", alpha_g},
        {"is_post_evict", is_post_evict},
        {"raw_h2d_ms", raw_h2d_ms},
        {"exposed_stall_ms", exposed_stall_ms},
        {"map_ms", map_ms},
    };
}

AtlasMorphEntry AtlasMorphEntry::from_json(const nlohmann::json& j) {
    AtlasMorphEntry e;
    e.model_family = j.value("model_family", "bagel");
    e.resolution = j.value("resolution", 512);
    e.alpha_g = j.value("alpha_g", 0.0);
    e.is_post_evict = j.value("is_post_evict", false);
    e.raw_h2d_ms = j.value("raw_h2d_ms", 0.0);
    e.exposed_stall_ms = j.value("exposed_stall_ms", 0.0);
    e.map_ms = j.value("map_ms", 0.0);
    return e;
}

nlohmann::json AtlasColocStretchEntry::to_json() const {
    return {
        {"model_family", model_family},
        {"resolution", resolution},
        {"colocated_ar_b", colocated_ar_b},
        {"step_stretch_factor", step_stretch_factor},
    };
}

AtlasColocStretchEntry AtlasColocStretchEntry::from_json(const nlohmann::json& j) {
    AtlasColocStretchEntry e;
    e.model_family = j.value("model_family", "bagel");
    e.resolution = j.value("resolution", 512);
    e.colocated_ar_b = j.value("colocated_ar_b", 0);
    e.step_stretch_factor = j.value("step_stretch_factor", 1.0);
    return e;
}

PerformanceAtlas PerformanceAtlas::a100_default() {
    PerformanceAtlas atlas;
    atlas.hardware_tag_ = "A100_40GB_PCIe_Stage1";
    atlas.ref_solo_h2d_gbps_ = 12.34;
    atlas.ref_bagel_expert_bytes_ = 12446LL * kMiB;
    atlas.ref_num_layers_ = 28;

    constexpr double kRawFull = 1058.50;
    constexpr double kMapFull = 11.71;

    // Live Stage-1 A100 measurements (`tab:eval_microbench`): BAGEL 1024^2 steady.
    // alpha=0.00 -> 173.97 ms (83.6% hidden); alpha=0.50 -> 7.53 ms (98.6% hidden).
    atlas.add_morph_entry({"bagel", 1024, 0.00, false, kRawFull, 173.97, kMapFull});
    atlas.add_morph_entry({"bagel", 1024, 0.50, false, 0.50 * kRawFull, 7.53, 0.50 * kMapFull});
    atlas.add_morph_entry({"bagel", 1024, 0.75, false, 0.25 * kRawFull, 0.00, 0.25 * kMapFull});
    atlas.add_morph_entry({"bagel", 1024, 1.00, false, 0.00, 0.00, 0.00});

    // Live Stage-1 A100 measurements (`tab:eval_microbench` footnote): BAGEL 1024^2 post-eviction.
    // alpha=0.75 incurs 267.69 ms wall delta (0% hidden due to VMM remap/eviction contention).
    atlas.add_morph_entry({"bagel", 1024, 0.00, true, kRawFull, kRawFull, kMapFull});
    atlas.add_morph_entry({"bagel", 1024, 0.50, true, 0.50 * kRawFull, 0.50 * kRawFull, 0.50 * kMapFull});
    atlas.add_morph_entry({"bagel", 1024, 0.75, true, 0.25 * kRawFull, 267.69, 0.25 * kMapFull});
    atlas.add_morph_entry({"bagel", 1024, 1.00, true, 0.00, 0.00, 0.00});

    // Live Stage-1 A100 measurements (`tab:eval_microbench` footnote): BAGEL 512^2 steady.
    // Hidden fractions: 84.5% (alpha=0.75), 41.0% (alpha=0.50), 5.7% (alpha=0.00).
    atlas.add_morph_entry({"bagel", 512, 0.00, false, kRawFull, kRawFull * (1.0 - 0.057), kMapFull});
    atlas.add_morph_entry({"bagel", 512, 0.50, false, 0.50 * kRawFull, 0.50 * kRawFull * (1.0 - 0.410), 0.50 * kMapFull});
    atlas.add_morph_entry({"bagel", 512, 0.75, false, 0.25 * kRawFull, 0.25 * kRawFull * (1.0 - 0.845), 0.25 * kMapFull});
    atlas.add_morph_entry({"bagel", 512, 1.00, false, 0.00, 0.00, 0.00});

    // Post-eviction 512^2 conservative zero-overlap curve.
    atlas.add_morph_entry({"bagel", 512, 0.00, true, kRawFull, kRawFull, kMapFull});
    atlas.add_morph_entry({"bagel", 512, 0.50, true, 0.50 * kRawFull, 0.50 * kRawFull, 0.50 * kMapFull});
    atlas.add_morph_entry({"bagel", 512, 0.75, true, 0.25 * kRawFull, 0.25 * kRawFull, 0.25 * kMapFull});
    atlas.add_morph_entry({"bagel", 512, 1.00, true, 0.00, 0.00, 0.00});

    // 3rd axis: Colocated AR microbatch preemption stretch grounded in live A100 M4 benchmarks
    // (packed [B, d] AR decode: B=1 -> 50.64 ms, B=8 -> 75.40 ms per yield under layer-chunk EDF).
    // Sublinear growth in B reflects single weight-read batch amortization across colocated AR tasks.
    atlas.add_coloc_entry({"bagel", 512, 0, 1.00});
    atlas.add_coloc_entry({"bagel", 512, 1, 1.48});
    atlas.add_coloc_entry({"bagel", 512, 2, 1.54});
    atlas.add_coloc_entry({"bagel", 512, 4, 1.68});
    atlas.add_coloc_entry({"bagel", 512, 8, 1.92});

    atlas.add_coloc_entry({"bagel", 1024, 0, 1.00});
    atlas.add_coloc_entry({"bagel", 1024, 1, 1.24});
    atlas.add_coloc_entry({"bagel", 1024, 2, 1.27});
    atlas.add_coloc_entry({"bagel", 1024, 4, 1.33});
    atlas.add_coloc_entry({"bagel", 1024, 8, 1.44});

    return atlas;
}

void PerformanceAtlas::add_morph_entry(AtlasMorphEntry entry) {
    morph_entries_.push_back(std::move(entry));
}

void PerformanceAtlas::add_coloc_entry(AtlasColocStretchEntry entry) {
    coloc_entries_.push_back(std::move(entry));
}

bool PerformanceAtlas::interpolate_alpha_at_res(
    const std::string& model_family,
    int resolution,
    bool is_post_evict,
    double alpha_g,
    MorphStallEstimate* out) const {
    std::vector<const AtlasMorphEntry*> pts;
    for (const auto& e : morph_entries_) {
        if (e.model_family == model_family && e.resolution == resolution && e.is_post_evict == is_post_evict) {
            pts.push_back(&e);
        }
    }
    if (pts.empty()) {
        return false;
    }
    std::sort(pts.begin(), pts.end(), [](const AtlasMorphEntry* a, const AtlasMorphEntry* b) {
        return a->alpha_g < b->alpha_g;
    });
    const double a = std::clamp(alpha_g, 0.0, 1.0);
    const AtlasMorphEntry* lo = pts.front();
    const AtlasMorphEntry* hi = pts.back();
    if (a <= lo->alpha_g) {
        hi = lo;
    } else if (a >= hi->alpha_g) {
        lo = hi;
    } else {
        for (size_t i = 0; i + 1 < pts.size(); ++i) {
            if (a >= pts[i]->alpha_g && a <= pts[i + 1]->alpha_g) {
                lo = pts[i];
                hi = pts[i + 1];
                break;
            }
        }
    }
    const double span = hi->alpha_g - lo->alpha_g;
    const double t = (span > 1e-9) ? std::clamp((a - lo->alpha_g) / span, 0.0, 1.0) : 0.0;
    const double raw_ms = lo->raw_h2d_ms * (1.0 - t) + hi->raw_h2d_ms * t;
    const double stall_ms = lo->exposed_stall_ms * (1.0 - t) + hi->exposed_stall_ms * t;
    const double map_ms = lo->map_ms * (1.0 - t) + hi->map_ms * t;
    out->raw_h2d_ms = std::max(0.0, raw_ms);
    out->exposed_stall_ms = std::max(0.0, stall_ms);
    out->map_ms = std::max(0.0, map_ms);
    out->hidden_frac = (raw_ms > 1e-9) ? std::clamp(1.0 - stall_ms / raw_ms, 0.0, 1.0) : 1.0;
    out->from_atlas = true;
    return true;
}

MorphStallEstimate PerformanceAtlas::estimate_legacy_analytical(
    double alpha_g,
    int64_t gen_expert_bytes,
    double bw_h2d_gbps,
    double dit_step1_ms,
    double map_overhead_ms_full,
    double overlap_efficiency) {
    const double alpha = std::clamp(alpha_g, 0.0, 1.0);
    const double missing_bytes = (1.0 - alpha) * static_cast<double>(std::max<int64_t>(0, gen_expert_bytes));
    if (missing_bytes <= 0.0) {
        return MorphStallEstimate{0.0, 0.0, 0.0, 1.0, false};
    }
    const double eff_bw = std::max(0.1, bw_h2d_gbps);
    const double raw_h2d_ms = (missing_bytes / 1e9) / eff_bw * 1000.0;
    const double map_ms = std::max(0.0, map_overhead_ms_full) * (1.0 - alpha);
    const double resident_compute_ms = std::max(0.0, overlap_efficiency) * alpha * std::max(0.0, dit_step1_ms);
    const double exposed_stall_ms = std::max(0.0, raw_h2d_ms - resident_compute_ms);
    const double hidden_frac = (raw_h2d_ms > 1e-9) ? std::clamp(1.0 - exposed_stall_ms / raw_h2d_ms, 0.0, 1.0) : 1.0;
    return MorphStallEstimate{raw_h2d_ms, exposed_stall_ms, map_ms, hidden_frac, false};
}

MorphStallEstimate PerformanceAtlas::estimate_pipeline_analytical(
    double alpha_g,
    bool is_post_evict,
    int64_t gen_expert_bytes,
    double bw_h2d_gbps,
    double dit_step1_ms,
    double map_overhead_ms_full,
    double overlap_efficiency,
    int num_layers) {
    const double alpha = std::clamp(alpha_g, 0.0, 1.0);
    const double missing_bytes = (1.0 - alpha) * static_cast<double>(std::max<int64_t>(0, gen_expert_bytes));
    if (missing_bytes <= 0.0) {
        return MorphStallEstimate{0.0, 0.0, 0.0, 1.0, false};
    }
    const double eff_bw = std::max(0.1, bw_h2d_gbps);
    const double raw_h2d_ms = (missing_bytes / 1e9) / eff_bw * 1000.0;
    const double map_ms = std::max(0.0, map_overhead_ms_full) * (1.0 - alpha);
    if (is_post_evict) {
        // Post-eviction VMM remap/TLB serialization exposes full H2D latency.
        return MorphStallEstimate{raw_h2d_ms, raw_h2d_ms, map_ms, 0.0, false};
    }
    // Exact first-principles L-layer pipelined DMA+compute induction:
    // Prefix p = alpha*L resident layers give initial compute lead p*c_eff; each streamed layer j
    // overlaps layer j+1's H2D while computing, yielding exposed stall:
    //   max(0, h - p*c_eff, m*h - (L - 1)*c_eff).
    const double L = static_cast<double>(std::max(1, num_layers));
    const double p = alpha * L;
    const double m = (1.0 - alpha) * L;
    const double h = raw_h2d_ms / std::max(1e-6, m);
    const double c_eff = std::max(0.0, overlap_efficiency) * (std::max(0.0, dit_step1_ms) / L);
    const double stall_head = h - p * c_eff;
    const double stall_tail = m * h - std::max(0.0, L - 1.0) * c_eff;
    const double exposed_stall_ms = std::max({0.0, stall_head, stall_tail});
    const double hidden_frac = (raw_h2d_ms > 1e-9) ? std::clamp(1.0 - exposed_stall_ms / raw_h2d_ms, 0.0, 1.0) : 1.0;
    return MorphStallEstimate{raw_h2d_ms, exposed_stall_ms, map_ms, hidden_frac, false};
}

MorphStallEstimate PerformanceAtlas::estimate_morph_stall(
    const std::string& model_family,
    int resolution,
    double alpha_g,
    bool is_post_evict,
    int64_t gen_expert_bytes,
    double bw_h2d_gbps,
    double dit_step1_ms,
    double map_overhead_ms_full,
    double overlap_efficiency,
    int num_layers) const {
    const double alpha = std::clamp(alpha_g, 0.0, 1.0);
    if (alpha >= 1.0 - 1e-9 || gen_expert_bytes <= 0) {
        return MorphStallEstimate{0.0, 0.0, 0.0, 1.0, true};
    }

    // Check whether the query matches the calibrated hardware/model envelope.
    // Unprofiled models, custom expert byte sizes (e.g., synthetic/unit-test models), or
    // substantially different interconnects (e.g., H100 PCIe Gen5) fall back cleanly to the
    // corrected L-layer pipelined analytical model.
    const bool bytes_match =
        (ref_bagel_expert_bytes_ > 0) &&
        (std::abs(static_cast<double>(gen_expert_bytes - ref_bagel_expert_bytes_)) /
             static_cast<double>(ref_bagel_expert_bytes_) <=
         0.05);
    const bool bw_match =
        (ref_solo_h2d_gbps_ > 0.0) &&
        (std::abs(bw_h2d_gbps - ref_solo_h2d_gbps_) / ref_solo_h2d_gbps_ <= 0.30);

    if (bytes_match && bw_match && !morph_entries_.empty()) {
        std::set<int> res_set;
        for (const auto& e : morph_entries_) {
            if (e.model_family == model_family && e.is_post_evict == is_post_evict) {
                res_set.insert(e.resolution);
            }
        }
        if (!res_set.empty()) {
            std::vector<int> res_list(res_set.begin(), res_set.end());
            int lo_res = res_list.front();
            int hi_res = res_list.back();
            if (resolution <= lo_res) {
                hi_res = lo_res;
            } else if (resolution >= hi_res) {
                lo_res = hi_res;
            } else {
                for (size_t i = 0; i + 1 < res_list.size(); ++i) {
                    if (resolution >= res_list[i] && resolution <= res_list[i + 1]) {
                        lo_res = res_list[i];
                        hi_res = res_list[i + 1];
                        break;
                    }
                }
            }
            MorphStallEstimate est_lo{};
            MorphStallEstimate est_hi{};
            if (interpolate_alpha_at_res(model_family, lo_res, is_post_evict, alpha, &est_lo) &&
                interpolate_alpha_at_res(model_family, hi_res, is_post_evict, alpha, &est_hi)) {
                const double r_span = static_cast<double>(hi_res - lo_res);
                const double rt = (r_span > 1e-9) ? std::clamp(static_cast<double>(resolution - lo_res) / r_span, 0.0, 1.0) : 0.0;
                MorphStallEstimate est;
                est.raw_h2d_ms = est_lo.raw_h2d_ms * (1.0 - rt) + est_hi.raw_h2d_ms * rt;
                est.exposed_stall_ms = est_lo.exposed_stall_ms * (1.0 - rt) + est_hi.exposed_stall_ms * rt;
                est.map_ms = est_lo.map_ms * (1.0 - rt) + est_hi.map_ms * rt;
                // Adjust proportionally for minor PCIe H2D bandwidth drift within envelope.
                const double bw_scale = ref_solo_h2d_gbps_ / std::max(0.1, bw_h2d_gbps);
                est.raw_h2d_ms *= bw_scale;
                est.exposed_stall_ms *= bw_scale;
                est.hidden_frac =
                    (est.raw_h2d_ms > 1e-9) ? std::clamp(1.0 - est.exposed_stall_ms / est.raw_h2d_ms, 0.0, 1.0) : 1.0;
                est.from_atlas = true;
                return est;
            }
        }
    }

    return estimate_pipeline_analytical(
        alpha,
        is_post_evict,
        gen_expert_bytes,
        bw_h2d_gbps,
        dit_step1_ms,
        map_overhead_ms_full,
        overlap_efficiency,
        num_layers);
}

double PerformanceAtlas::dit_step_stretch_factor(
    const std::string& model_family,
    int resolution,
    int colocated_ar_b) const {
    if (colocated_ar_b <= 0) {
        return 1.0;
    }
    std::set<int> res_set;
    for (const auto& e : coloc_entries_) {
        if (e.model_family == model_family) {
            res_set.insert(e.resolution);
        }
    }
    if (res_set.empty()) {
        // Analytical sublinear preemption stretch fallback for uncalibrated models.
        const double b = static_cast<double>(colocated_ar_b);
        const double base_pen = (resolution >= 1024) ? 0.22 : 0.44;
        return 1.0 + base_pen * (1.0 + 0.14 * std::log2(b));
    }

    auto interp_b = [&](int res_anchor) -> double {
        std::vector<const AtlasColocStretchEntry*> pts;
        for (const auto& e : coloc_entries_) {
            if (e.model_family == model_family && e.resolution == res_anchor) {
                pts.push_back(&e);
            }
        }
        if (pts.empty()) return 1.0;
        std::sort(pts.begin(), pts.end(), [](const AtlasColocStretchEntry* a, const AtlasColocStretchEntry* b) {
            return a->colocated_ar_b < b->colocated_ar_b;
        });
        if (colocated_ar_b <= pts.front()->colocated_ar_b) return pts.front()->step_stretch_factor;
        if (colocated_ar_b >= pts.back()->colocated_ar_b) return pts.back()->step_stretch_factor;
        for (size_t i = 0; i + 1 < pts.size(); ++i) {
            if (colocated_ar_b >= pts[i]->colocated_ar_b && colocated_ar_b <= pts[i + 1]->colocated_ar_b) {
                const double span = static_cast<double>(pts[i + 1]->colocated_ar_b - pts[i]->colocated_ar_b);
                const double t = (span > 1e-9)
                                     ? static_cast<double>(colocated_ar_b - pts[i]->colocated_ar_b) / span
                                     : 0.0;
                return pts[i]->step_stretch_factor * (1.0 - t) + pts[i + 1]->step_stretch_factor * t;
            }
        }
        return pts.back()->step_stretch_factor;
    };

    std::vector<int> res_list(res_set.begin(), res_set.end());
    int lo_res = res_list.front();
    int hi_res = res_list.back();
    if (resolution <= lo_res) {
        return interp_b(lo_res);
    }
    if (resolution >= hi_res) {
        return interp_b(hi_res);
    }
    for (size_t i = 0; i + 1 < res_list.size(); ++i) {
        if (resolution >= res_list[i] && resolution <= res_list[i + 1]) {
            lo_res = res_list[i];
            hi_res = res_list[i + 1];
            break;
        }
    }
    const double f_lo = interp_b(lo_res);
    const double f_hi = interp_b(hi_res);
    const double span = static_cast<double>(hi_res - lo_res);
    const double rt = (span > 1e-9) ? static_cast<double>(resolution - lo_res) / span : 0.0;
    return f_lo * (1.0 - rt) + f_hi * rt;
}

nlohmann::json PerformanceAtlas::to_json() const {
    nlohmann::json morph_j = nlohmann::json::array();
    for (const auto& e : morph_entries_) {
        morph_j.push_back(e.to_json());
    }
    nlohmann::json coloc_j = nlohmann::json::array();
    for (const auto& e : coloc_entries_) {
        coloc_j.push_back(e.to_json());
    }
    return {
        {"hardware_tag", hardware_tag_},
        {"ref_solo_h2d_gbps", ref_solo_h2d_gbps_},
        {"ref_bagel_expert_bytes", ref_bagel_expert_bytes_},
        {"ref_num_layers", ref_num_layers_},
        {"morph_entries", std::move(morph_j)},
        {"coloc_entries", std::move(coloc_j)},
    };
}

PerformanceAtlas PerformanceAtlas::from_json(const nlohmann::json& j) {
    PerformanceAtlas atlas;
    atlas.hardware_tag_ = j.value("hardware_tag", "custom");
    atlas.ref_solo_h2d_gbps_ = j.value("ref_solo_h2d_gbps", 12.34);
    atlas.ref_bagel_expert_bytes_ = j.value("ref_bagel_expert_bytes", 12446LL * kMiB);
    atlas.ref_num_layers_ = j.value("ref_num_layers", 28);
    if (j.contains("morph_entries") && j["morph_entries"].is_array()) {
        for (const auto& item : j["morph_entries"]) {
            atlas.add_morph_entry(AtlasMorphEntry::from_json(item));
        }
    }
    if (j.contains("coloc_entries") && j["coloc_entries"].is_array()) {
        for (const auto& item : j["coloc_entries"]) {
            atlas.add_coloc_entry(AtlasColocStretchEntry::from_json(item));
        }
    }
    return atlas;
}

}  // namespace morphserve::control
