#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace morphserve::control {

struct SimTurnSpec {
    std::string modality_plan = "ar_only";
    int height = 512;
    int dit_steps = 20;
    int max_ar_tokens = 48;
    int num_images = 0;
    bool use_prev_image = false;
};

struct SimSessionSpec {
    double arrival_ms = 0.0;
    std::string quadrant = "visual_cot";
    std::vector<SimTurnSpec> turns;
    std::string session_id;
    std::string phase_label;
};

struct RawSuiteSession {
    std::string session_id;
    std::string quadrant;
    std::string phase_label;
    double arrival_offset_s = 0.0;
    std::vector<SimTurnSpec> turns;
};

std::vector<RawSuiteSession> stratify_raw_sessions(
    const std::vector<RawSuiteSession>& sessions,
    int max_sessions);

std::vector<RawSuiteSession> scale_raw_sessions(
    const std::vector<RawSuiteSession>& base_sessions,
    int target_sessions);

std::vector<SimSessionSpec> build_sim_session_stream(
    int num_sessions,
    double offered_qps,
    int num_gpus,
    uint64_t seed = 2026,
    int extra_edit_turns = 0,
    const std::string& skew_tag = "",
    const std::string& bench_root = "",
    const std::string& sources_dir = "");

}  // namespace morphserve::control
