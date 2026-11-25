#include "morphserve/control/preempt_edf.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace morphserve::control {

PreemptEdfController::PreemptEdfController(
    int num_layers,
    double tpot_slo_ms,
    double guard_ms,
    double init_layer_ms,
    double init_ar_step_ms,
    int max_microbatch_cap) noexcept
    : num_layers_(std::max(1, num_layers)),
      tpot_slo_ms_(tpot_slo_ms),
      guard_ms_(guard_ms),
      ema_layer_ms_(init_layer_ms),
      ema_ar_step_ms_(init_ar_step_ms),
      max_microbatch_cap_(std::max(1, max_microbatch_cap)) {}

void PreemptEdfController::set_max_microbatch_cap(int v) noexcept {
    max_microbatch_cap_ = std::max(1, v);
}

int PreemptEdfController::compute_adaptive_chunk_layers(double override_tpot_slo_ms) const noexcept {
    const double slo = (override_tpot_slo_ms > 0.0) ? override_tpot_slo_ms : tpot_slo_ms_;
    const double budget_ms = std::max(0.0, slo - ema_ar_step_ms_ - guard_ms_);
    if (ema_layer_ms_ <= 0.0) {
        return 1;
    }
    const int k = static_cast<int>(std::floor(budget_ms / ema_layer_ms_));
    return std::max(1, std::min(num_layers_, k));
}

double PreemptEdfController::record_layer_sample(double layer_dt_ms) noexcept {
    ema_layer_ms_ = 0.7 * ema_layer_ms_ + 0.3 * layer_dt_ms;
    return ema_layer_ms_;
}

double PreemptEdfController::record_ar_batch_sample(double batch_dt_ms) noexcept {
    ema_ar_step_ms_ = 0.5 * ema_ar_step_ms_ + 0.5 * batch_dt_ms;
    return ema_ar_step_ms_;
}

bool PreemptEdfController::should_yield_now(
    double now_s,
    double earliest_deadline_s,
    double next_unit_ms) const noexcept {
    const double predicted_finish_ms =
        (now_s - earliest_deadline_s) * 1000.0 + next_unit_ms + ema_ar_step_ms_ + guard_ms_;
    return predicted_finish_ms >= 0.0;
}

void PreemptEdfController::select_edf_microbatch(
    double now_s,
    const double* deadlines_s,
    size_t num_tasks,
    bool force_all_urgent,
    std::vector<size_t>& out_due_indices,
    std::vector<size_t>& out_deferred_indices) const {
    out_due_indices.clear();
    out_deferred_indices.clear();
    if (num_tasks == 0 || deadlines_s == nullptr) {
        return;
    }
    std::vector<size_t> ordered(num_tasks);
    std::iota(ordered.begin(), ordered.end(), 0);
    std::stable_sort(ordered.begin(), ordered.end(), [&](size_t a, size_t b) {
        return deadlines_s[a] < deadlines_s[b];
    });

    const size_t max_serve = static_cast<size_t>(std::max(1, max_microbatch_cap_));
    // Account for the current yield's own `ema_ar_step_ms_` execution time: if a peer task's
    // remaining slack after this yield (`slack_ms - ema_ar_step_ms_`) would fall within one
    // decode-plus-chunk window (`ema_ar_step_ms_ + ema_layer_ms_ + guard_ms_`), coalesce it into
    // the same `[B, D]` packed microbatch now instead of anti-phase serial ping-pong!
    const double coalesce_window_ms = 2.0 * ema_ar_step_ms_ + ema_layer_ms_ + guard_ms_;
    for (size_t rank_idx = 0; rank_idx < ordered.size(); ++rank_idx) {
        const size_t orig_idx = ordered[rank_idx];
        const double slack_ms = (deadlines_s[orig_idx] - now_s) * 1000.0;
        if (force_all_urgent || rank_idx == 0 ||
            (out_due_indices.size() < max_serve && slack_ms <= coalesce_window_ms)) {
            out_due_indices.push_back(orig_idx);
        } else {
            out_deferred_indices.push_back(orig_idx);
        }
    }
}

}  // namespace morphserve::control
