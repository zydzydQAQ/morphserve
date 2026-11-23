#pragma once

#include <condition_variable>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace morphserve::runtime {

// Serializes heavy expert H2D DMA per PCIe root half-node (analytical ledger + thread/process locks).
class HalfNodeH2DPacer {
public:
    HalfNodeH2DPacer(
        std::unordered_map<int, int> gpu_to_group,
        int max_concurrent_per_half = 1,
        std::string lock_dir = "/tmp/morphserve_h2d_locks");

    int max_concurrent_per_half() const noexcept { return max_concurrent_; }
    int group_of(int gpu_id) const;

    double predicted_wait_ms(int gpu_id, double now_ms = 0.0) const;
    double reserve_slot(int gpu_id, double duration_ms, double now_ms = 0.0);
    void reset_analytical_slots();

    void acquire_thread_slot(int gpu_id);
    void release_thread_slot(int gpu_id);

    // Acquires an exclusive cross-process POSIX `flock` for `gpu_id`'s half-node; returns fd >= 0.
    int acquire_process_flock(int gpu_id, int* out_group_id = nullptr);
    static void release_process_flock(int fd) noexcept;

private:
    std::unordered_map<int, int> gpu_to_group_;
    int max_concurrent_ = 1;
    std::string lock_dir_;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::unordered_map<int, std::vector<double>> busy_until_ms_;
    std::unordered_map<int, int> active_threads_;
};

}  // namespace morphserve::runtime
