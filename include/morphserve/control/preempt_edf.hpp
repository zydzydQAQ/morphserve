#pragma once

#include <cstddef>
#include <vector>

namespace morphserve::control {

// GIL-free EDF & adaptive layer-chunk preemption controller for `UnifiedPreemptiveScheduler`.
class PreemptEdfController {
public:
    PreemptEdfController(
        int num_layers = 28,
        double tpot_slo_ms = 125.6,
        double guard_ms = 14.0,
        double init_layer_ms = 10.5,
        double init_ar_step_ms = 66.0,
        int max_microbatch_cap = 8) noexcept;

    int num_layers() const noexcept { return num_layers_; }
    double tpot_slo_ms() const noexcept { return tpot_slo_ms_; }
    double guard_ms() const noexcept { return guard_ms_; }
    double ema_layer_ms() const noexcept { return ema_layer_ms_; }
    double ema_ar_step_ms() const noexcept { return ema_ar_step_ms_; }
    int max_microbatch_cap() const noexcept { return max_microbatch_cap_; }

    void set_tpot_slo_ms(double v) noexcept { tpot_slo_ms_ = v; }
    void set_guard_ms(double v) noexcept { guard_ms_ = v; }
    void set_ema_layer_ms(double v) noexcept { ema_layer_ms_ = v; }
    void set_ema_ar_step_ms(double v) noexcept { ema_ar_step_ms_ = v; }
    void set_max_microbatch_cap(int v) noexcept;

    int compute_adaptive_chunk_layers(double override_tpot_slo_ms = -1.0) const noexcept;
    double record_layer_sample(double layer_dt_ms) noexcept;
    double record_ar_batch_sample(double batch_dt_ms) noexcept;
    bool should_yield_now(double now_s, double earliest_deadline_s, double next_unit_ms) const noexcept;

    // Given task `deadlines_s` (arbitrary order), partitions indices into `{due_edf_order, deferred_order}`.
    void select_edf_microbatch(
        double now_s,
        const double* deadlines_s,
        size_t num_tasks,
        bool force_all_urgent,
        std::vector<size_t>& out_due_indices,
        std::vector<size_t>& out_deferred_indices) const;

private:
    int num_layers_ = 28;
    double tpot_slo_ms_ = 125.6;
    double guard_ms_ = 14.0;
    double ema_layer_ms_ = 10.5;
    double ema_ar_step_ms_ = 66.0;
    int max_microbatch_cap_ = 8;
};

}  // namespace morphserve::control
