#include "morphserve/control/gpu_queue_scheduler.hpp"

#include <algorithm>
#include <limits>
#include <tuple>

namespace morphserve::control {

namespace {

double effective_slack_ms(const GpuQueueWaiter& w, double now_ms) noexcept {
    if (w.slo_deadline_ms <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return w.slo_deadline_ms - now_ms - std::max(0.0, w.est_exec_ms);
}

bool is_older_waiter(const GpuQueueWaiter& a, const GpuQueueWaiter& b) noexcept {
    return std::tie(a.enqueue_ms, a.waiter_id) < std::tie(b.enqueue_ms, b.waiter_id);
}

size_t find_fifo_head_index(const std::vector<GpuQueueWaiter>& waiters) noexcept {
    size_t best = 0;
    for (size_t i = 1; i < waiters.size(); ++i) {
        if (is_older_waiter(waiters[i], waiters[best])) {
            best = i;
        }
    }
    return best;
}

}  // namespace

nlohmann::json GpuQueueWaiter::to_json() const {
    return {
        {"waiter_id", waiter_id},
        {"enqueue_ms", enqueue_ms},
        {"est_mem_bytes", est_mem_bytes},
        {"est_exec_ms", est_exec_ms},
        {"slo_deadline_ms", slo_deadline_ms},
        {"op_priority", op_priority},
        {"bypass_count", bypass_count},
        {"session_slo_intact", session_slo_intact},
        {"remaining_turns", remaining_turns},
    };
}

GpuQueueWaiter GpuQueueWaiter::from_json(const nlohmann::json& j) {
    GpuQueueWaiter w;
    w.waiter_id = j.value("waiter_id", static_cast<uint64_t>(0));
    w.enqueue_ms = j.value("enqueue_ms", 0.0);
    w.est_mem_bytes = j.value("est_mem_bytes", static_cast<int64_t>(0));
    w.est_exec_ms = j.value("est_exec_ms", 0.0);
    w.slo_deadline_ms = j.value("slo_deadline_ms", 0.0);
    w.op_priority = j.value("op_priority", 1);
    w.bypass_count = j.value("bypass_count", 0);
    w.session_slo_intact = j.value("session_slo_intact", true);
    w.remaining_turns = std::max(1, j.value("remaining_turns", 1));
    return w;
}

nlohmann::json GpuQueuePolicy::to_json() const {
    return {
        {"mode", mode},
        {"max_wait_aging_ms", max_wait_aging_ms},
        {"max_bypasses", max_bypasses},
    };
}

GpuQueuePolicy GpuQueuePolicy::from_json(const nlohmann::json& j) {
    GpuQueuePolicy p;
    p.mode = j.value("mode", "memory_slack");
    p.max_wait_aging_ms = j.value("max_wait_aging_ms", 45000.0);
    p.max_bypasses = j.value("max_bypasses", 5);
    return p;
}

nlohmann::json GpuQueueStats::to_json() const {
    return {
        {"dispatches_total", dispatches_total},
        {"reorders_total", reorders_total},
        {"memory_bypasses_total", memory_bypasses_total},
        {"starvation_promotions_total", starvation_promotions_total},
    };
}

GpuWaitQueueScheduler::GpuWaitQueueScheduler(GpuQueuePolicy policy) : policy_(std::move(policy)) {}

std::optional<GpuQueueSelection> GpuWaitQueueScheduler::peek_next(
    const std::vector<GpuQueueWaiter>& waiters,
    double now_ms,
    int64_t live_free_mem_bytes) const {
    if (waiters.empty()) {
        return std::nullopt;
    }

    const size_t fifo_idx = find_fifo_head_index(waiters);
    if (waiters.size() == 1 || policy_.mode == "fcfs") {
        return GpuQueueSelection{
            waiters[fifo_idx].waiter_id,
            fifo_idx,
            /*reordered=*/false,
            /*memory_bypass=*/false,
            /*starvation_promoted=*/false,
        };
    }

    auto fits_mem = [&](const GpuQueueWaiter& w) noexcept {
        return w.est_mem_bytes <= 0 || w.est_mem_bytes <= live_free_mem_bytes;
    };
    auto is_starving = [&](const GpuQueueWaiter& w) noexcept {
        const double waited = std::max(0.0, now_ms - w.enqueue_ms);
        return (policy_.max_wait_aging_ms > 0.0 && waited >= policy_.max_wait_aging_ms) ||
               (policy_.max_bypasses > 0 && w.bypass_count >= policy_.max_bypasses);
    };

    // 1. Bounded anti-starvation tier: promote waiters that hit max age or max bypass cap.
    std::vector<size_t> urgent;
    for (size_t i = 0; i < waiters.size(); ++i) {
        if (is_starving(waiters[i])) {
            urgent.push_back(i);
        }
    }
    if (!urgent.empty()) {
        size_t best = urgent.front();
        for (size_t idx : urgent) {
            const auto& a = waiters[idx];
            const auto& b = waiters[best];
            const int a_fit = fits_mem(a) ? 0 : 1;
            const int b_fit = fits_mem(b) ? 0 : 1;
            if (std::tie(a.op_priority, a_fit, a.enqueue_ms, a.waiter_id) <
                std::tie(b.op_priority, b_fit, b.enqueue_ms, b.waiter_id)) {
                best = idx;
            }
        }
        const bool reordered = (best != fifo_idx);
        const bool mem_bypass = reordered && !fits_mem(waiters[fifo_idx]) && fits_mem(waiters[best]);
        return GpuQueueSelection{
            waiters[best].waiter_id,
            best,
            reordered,
            mem_bypass,
            /*starvation_promoted=*/true,
        };
    }

    // 2. Memory-fit tier (0-OOM guard): skip waiters whose memory footprint exceeds current HBM headroom.
    std::vector<size_t> fitting;
    fitting.reserve(waiters.size());
    for (size_t i = 0; i < waiters.size(); ++i) {
        if (fits_mem(waiters[i])) {
            fitting.push_back(i);
        }
    }

    size_t chosen = fifo_idx;
    if (!fitting.empty()) {
        // Session-SLO-aware Moore-Hodgson Session-SRPT + EDF-Feasible-Pair Interchange (Autellix/Parrot):
        // Partition fitting waiters into 3 buckets:
        //   0: SLO-feasible (`slack >= 0` or unprioritized) AND `session_slo_intact` (preserves BOTH Turn & Joint SLO)
        //      - Primary winner `best_srpt`: minimizes remaining session work `rem_turns * max(1, est_exec_ms)`
        //        so overload banks maximum completed multi-turn sessions per GPU-second (preventing EDF
        //        processor-sharing collapse across half-finished sessions and protecting short 512px turns
        //        from heavy 1024px turns).
        //      - EDF-Feasible-Pair Interchange: a tighter-slack Bucket-0 waiter `u` may run before `best_srpt`
        //        IFF `best_srpt` remains comfortably feasible after `u` executes (`slack(best_srpt) >= safety * u.exec`),
        //        preserving Liu-Layland EDF optimality under non-overloaded load so BOTH finish within SLO!
        //   1: SLO-feasible (`slack >= 0`) with prior turn SLO missed (`!session_slo_intact`; tightest slack EDF)
        //   2: Already-expired (`slack < 0`), ordered by SRPT (`est_exec_ms` ascending).
        auto waiter_bucket = [&](const GpuQueueWaiter& w) noexcept {
            const double slack = effective_slack_ms(w, now_ms);
            const bool feasible = (w.slo_deadline_ms <= 0.0) || (slack >= 0.0);
            return feasible ? (w.session_slo_intact ? 0 : 1) : 2;
        };

        int best_prio = std::numeric_limits<int>::max();
        int best_bucket = 3;
        for (size_t idx : fitting) {
            const auto& w = waiters[idx];
            const int b = waiter_bucket(w);
            if (std::tie(w.op_priority, b) < std::tie(best_prio, best_bucket)) {
                best_prio = w.op_priority;
                best_bucket = b;
            }
        }

        std::vector<size_t> top_group;
        top_group.reserve(fitting.size());
        for (size_t idx : fitting) {
            const auto& w = waiters[idx];
            if (w.op_priority == best_prio && waiter_bucket(w) == best_bucket) {
                top_group.push_back(idx);
            }
        }

        if (best_bucket == 0) {
            auto srpt_key = [&](const GpuQueueWaiter& w) noexcept {
                const int rem = std::max(1, w.remaining_turns);
                const double exec_ms = std::max(1.0, w.est_exec_ms);
                const double rem_work = static_cast<double>(rem) * exec_ms;
                const double slack = effective_slack_ms(w, now_ms);
                return std::make_tuple(rem_work, rem, slack, exec_ms, w.enqueue_ms, w.waiter_id);
            };
            size_t best_srpt = top_group.front();
            for (size_t idx : top_group) {
                if (srpt_key(waiters[idx]) < srpt_key(waiters[best_srpt])) {
                    best_srpt = idx;
                }
            }
            chosen = best_srpt;
            const double best_slack = effective_slack_ms(waiters[best_srpt], now_ms);
            const double safety = (top_group.size() <= 2) ? 1.0 : 1.5;
            size_t best_edf = best_srpt;
            double best_edf_slack = best_slack;
            for (size_t idx : top_group) {
                if (idx == best_srpt) continue;
                const auto& u = waiters[idx];
                const double u_slack = effective_slack_ms(u, now_ms);
                const double u_exec = std::max(0.0, u.est_exec_ms);
                if (u_slack < best_edf_slack && best_slack >= safety * u_exec) {
                    best_edf = idx;
                    best_edf_slack = u_slack;
                }
            }
            chosen = best_edf;
        } else {
            auto fallback_key = [&](const GpuQueueWaiter& w) noexcept {
                const double slack = effective_slack_ms(w, now_ms);
                const bool feasible = (best_bucket == 1);
                const double primary = feasible ? slack : std::max(0.0, w.est_exec_ms);
                const double secondary = feasible ? std::max(0.0, w.est_exec_ms) : -slack;
                return std::make_tuple(primary, secondary, w.enqueue_ms, w.waiter_id);
            };
            chosen = top_group.front();
            for (size_t idx : top_group) {
                if (fallback_key(waiters[idx]) < fallback_key(waiters[chosen])) {
                    chosen = idx;
                }
            }
        }
    } else {
        // Deadlock-free fallback when every waiter exceeds free_mem_bytes: pick smallest footprint
        // (highest chance to succeed after VMM drop-clean / KV swap), tie-breaking by FIFO order.
        chosen = 0;
        for (size_t i = 1; i < waiters.size(); ++i) {
            const auto& a = waiters[i];
            const auto& b = waiters[chosen];
            if (std::tie(a.est_mem_bytes, a.enqueue_ms, a.waiter_id) <
                std::tie(b.est_mem_bytes, b.enqueue_ms, b.waiter_id)) {
                chosen = i;
            }
        }
    }

    const bool reordered = (chosen != fifo_idx);
    const bool mem_bypass = reordered && !fits_mem(waiters[fifo_idx]) && fits_mem(waiters[chosen]);
    return GpuQueueSelection{
        waiters[chosen].waiter_id,
        chosen,
        reordered,
        mem_bypass,
        /*starvation_promoted=*/false,
    };
}

std::optional<GpuQueueSelection> GpuWaitQueueScheduler::select_and_pop(
    std::vector<GpuQueueWaiter>& waiters,
    double now_ms,
    int64_t live_free_mem_bytes) {
    auto sel = peek_next(waiters, now_ms, live_free_mem_bytes);
    if (!sel.has_value()) {
        return std::nullopt;
    }

    const GpuQueueWaiter chosen = waiters[sel->chosen_index];
    waiters.erase(waiters.begin() + static_cast<std::ptrdiff_t>(sel->chosen_index));

    // Track bypasses strictly on older waiters that were jumped over.
    if (sel->reordered) {
        for (auto& w : waiters) {
            if (is_older_waiter(w, chosen)) {
                ++w.bypass_count;
            }
        }
    }

    ++stats_.dispatches_total;
    if (sel->reordered) {
        ++stats_.reorders_total;
    }
    if (sel->memory_bypass) {
        ++stats_.memory_bypasses_total;
    }
    if (sel->starvation_promoted) {
        ++stats_.starvation_promotions_total;
    }
    return sel;
}

}  // namespace morphserve::control
