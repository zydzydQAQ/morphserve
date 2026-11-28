#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace morphserve::control {

// One request waiting for exclusive GPU execution (PREP / PACK / EXEC_DIT / SHM_SEND).
struct GpuQueueWaiter {
    uint64_t waiter_id = 0;
    double enqueue_ms = 0.0;
    int64_t est_mem_bytes = 0;
    double est_exec_ms = 0.0;
    double slo_deadline_ms = 0.0;  // absolute deadline in ms (<= 0 means unprioritized/infinite)
    int op_priority = 1;           // 0 = control/prep/pack, 1 = DiT/heavy compute
    int bypass_count = 0;
    bool session_slo_intact = true;  // true iff all prior turns in this session met SLO
    int remaining_turns = 1;         // remaining turns in session including this turn (>= 1)

    nlohmann::json to_json() const;
    static GpuQueueWaiter from_json(const nlohmann::json& j);
};

struct GpuQueuePolicy {
    std::string mode = "memory_slack";  // "memory_slack" | "fcfs"
    double max_wait_aging_ms = 45000.0; // hard anti-starvation age threshold
    int max_bypasses = 5;               // max times a waiter may be skipped before promotion

    nlohmann::json to_json() const;
    static GpuQueuePolicy from_json(const nlohmann::json& j);
};

struct GpuQueueSelection {
    uint64_t waiter_id = 0;
    size_t chosen_index = 0;
    bool reordered = false;
    bool memory_bypass = false;
    bool starvation_promoted = false;
};

struct GpuQueueStats {
    int64_t dispatches_total = 0;
    int64_t reorders_total = 0;
    int64_t memory_bypasses_total = 0;
    int64_t starvation_promotions_total = 0;

    nlohmann::json to_json() const;
};

// Shared C++17 per-GPU wait-queue selector (FlashGen-Sched ASPLOS'25-inspired).
// Used identically by live `cluster_engine.py` (`MemorySlackGpuLock`) and `ClusterSimulator`
// to eliminate DiT/memory head-of-line blocking under strict 0-OOM & bounded-starvation guards.
class GpuWaitQueueScheduler {
public:
    explicit GpuWaitQueueScheduler(GpuQueuePolicy policy = GpuQueuePolicy{});

    const GpuQueuePolicy& policy() const noexcept { return policy_; }
    void set_policy(GpuQueuePolicy policy) noexcept { policy_ = std::move(policy); }
    const GpuQueueStats& stats() const noexcept { return stats_; }
    void reset_stats() noexcept { stats_ = GpuQueueStats{}; }

    // Selects and erases the next waiter from `waiters`, incrementing `bypass_count` on skipped older waiters.
    std::optional<GpuQueueSelection> select_and_pop(
        std::vector<GpuQueueWaiter>& waiters,
        double now_ms,
        int64_t live_free_mem_bytes);

    // Pure selection without mutating `waiters` or counters (useful for inspection/FFI).
    std::optional<GpuQueueSelection> peek_next(
        const std::vector<GpuQueueWaiter>& waiters,
        double now_ms,
        int64_t live_free_mem_bytes) const;

private:
    GpuQueuePolicy policy_;
    GpuQueueStats stats_;
};

}  // namespace morphserve::control
