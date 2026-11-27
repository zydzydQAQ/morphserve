#pragma once

#include "morphserve/control/topology.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace morphserve::control {

// Result of evaluating Path-B in-place morph stall & map overhead.
struct MorphStallEstimate {
    double raw_h2d_ms = 0.0;
    double exposed_stall_ms = 0.0;
    double map_ms = 0.0;
    double hidden_frac = 0.0;
    bool from_atlas = false;
};

// Offline-profiled hardware entry for (model_family, resolution, alpha_g, is_post_evict).
struct AtlasMorphEntry {
    std::string model_family = "bagel";
    int resolution = 512;
    double alpha_g = 0.0;
    bool is_post_evict = false;
    double raw_h2d_ms = 0.0;
    double exposed_stall_ms = 0.0;
    double map_ms = 0.0;

    nlohmann::json to_json() const;
    static AtlasMorphEntry from_json(const nlohmann::json& j);
};

// Offline-profiled 3rd-axis entry for colocated AR preemption DiT step stretch.
struct AtlasColocStretchEntry {
    std::string model_family = "bagel";
    int resolution = 512;
    int colocated_ar_b = 0;
    double step_stretch_factor = 1.0;

    nlohmann::json to_json() const;
    static AtlasColocStretchEntry from_json(const nlohmann::json& j);
};

// 3D Hardware Performance Atlas (Resonator ISCA'26-inspired) with O(1) piecewise-linear
// interpolation over `(model_family, resolution, alpha_g, is_post_evict, colocated_ar_B)`
// and automatic fallback to the corrected L-layer pipelined analytical model on unprofiled hardware.
class PerformanceAtlas {
public:
    PerformanceAtlas() = default;

    // Grounded in live Stage-1 A100-40GB PCIe microbenchmarks (Table `tab:eval_microbench`).
    static PerformanceAtlas a100_default();

    const std::string& hardware_tag() const noexcept { return hardware_tag_; }
    double ref_solo_h2d_gbps() const noexcept { return ref_solo_h2d_gbps_; }
    int64_t ref_bagel_expert_bytes() const noexcept { return ref_bagel_expert_bytes_; }
    int ref_num_layers() const noexcept { return ref_num_layers_; }

    void add_morph_entry(AtlasMorphEntry entry);
    void add_coloc_entry(AtlasColocStretchEntry entry);

    // Query calibrated 3D Atlas; falls back to corrected per-layer pipeline analytical model
    // when hardware/expert/resolution lies outside the calibrated envelope.
    MorphStallEstimate estimate_morph_stall(
        const std::string& model_family,
        int resolution,
        double alpha_g,
        bool is_post_evict,
        int64_t gen_expert_bytes,
        double bw_h2d_gbps,
        double dit_step1_ms,
        double map_overhead_ms_full,
        double overlap_efficiency = 0.90,
        int num_layers = 28) const;

    // Corrected L-layer pipelined DMA+compute analytical model (used directly in `pipeline_analytical`
    // mode and as automatic fallback for uncalibrated hardware such as H100/synthetic sweeps).
    static MorphStallEstimate estimate_pipeline_analytical(
        double alpha_g,
        bool is_post_evict,
        int64_t gen_expert_bytes,
        double bw_h2d_gbps,
        double dit_step1_ms,
        double map_overhead_ms_full,
        double overlap_efficiency = 0.90,
        int num_layers = 28);

    // Original linear analytical formula preserved verbatim for controlled ablations (`legacy_analytical`).
    static MorphStallEstimate estimate_legacy_analytical(
        double alpha_g,
        int64_t gen_expert_bytes,
        double bw_h2d_gbps,
        double dit_step1_ms,
        double map_overhead_ms_full,
        double overlap_efficiency = 0.90);

    // 3rd axis: effective DiT step time stretch factor under `colocated_ar_b` concurrent AR decode tasks.
    double dit_step_stretch_factor(
        const std::string& model_family,
        int resolution,
        int colocated_ar_b) const;

    nlohmann::json to_json() const;
    static PerformanceAtlas from_json(const nlohmann::json& j);

private:
    std::string hardware_tag_ = "A100_40GB_PCIe_Stage1";
    double ref_solo_h2d_gbps_ = 12.34;
    int64_t ref_bagel_expert_bytes_ = 12446LL * kMiB;
    int ref_num_layers_ = 28;
    std::vector<AtlasMorphEntry> morph_entries_;
    std::vector<AtlasColocStretchEntry> coloc_entries_;

    bool interpolate_alpha_at_res(
        const std::string& model_family,
        int resolution,
        bool is_post_evict,
        double alpha_g,
        MorphStallEstimate* out) const;
};

}  // namespace morphserve::control
