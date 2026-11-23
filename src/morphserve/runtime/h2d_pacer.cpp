#include "morphserve/runtime/h2d_pacer.hpp"

#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <filesystem>
#include <stdexcept>
#include <sys/file.h>
#include <system_error>
#include <unistd.h>

namespace morphserve::runtime {

HalfNodeH2DPacer::HalfNodeH2DPacer(
    std::unordered_map<int, int> gpu_to_group,
    int max_concurrent_per_half,
    std::string lock_dir)
    : gpu_to_group_(std::move(gpu_to_group)),
      max_concurrent_(std::max(1, max_concurrent_per_half)),
      lock_dir_(std::move(lock_dir)) {
    for (const auto& kv : gpu_to_group_) {
        busy_until_ms_[kv.second].assign(static_cast<size_t>(max_concurrent_), 0.0);
        active_threads_[kv.second] = 0;
    }
}

int HalfNodeH2DPacer::group_of(int gpu_id) const {
    auto it = gpu_to_group_.find(gpu_id);
    if (it == gpu_to_group_.end()) {
        throw std::out_of_range("gpu_id=" + std::to_string(gpu_id) + " not mapped in HalfNodeH2DPacer");
    }
    return it->second;
}

double HalfNodeH2DPacer::predicted_wait_ms(int gpu_id, double now_ms) const {
    const int gid = group_of(gpu_id);
    std::lock_guard<std::mutex> lock(mu_);
    const auto& slots = busy_until_ms_.at(gid);
    const double earliest = *std::min_element(slots.begin(), slots.end());
    return std::max(0.0, earliest - now_ms);
}

double HalfNodeH2DPacer::reserve_slot(int gpu_id, double duration_ms, double now_ms) {
    const int gid = group_of(gpu_id);
    std::lock_guard<std::mutex> lock(mu_);
    auto& slots = busy_until_ms_.at(gid);
    auto min_it = std::min_element(slots.begin(), slots.end());
    const double start_ms = std::max(now_ms, *min_it);
    *min_it = start_ms + std::max(0.0, duration_ms);
    return start_ms - now_ms;
}

void HalfNodeH2DPacer::reset_analytical_slots() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& kv : busy_until_ms_) {
        std::fill(kv.second.begin(), kv.second.end(), 0.0);
    }
}

void HalfNodeH2DPacer::acquire_thread_slot(int gpu_id) {
    const int gid = group_of(gpu_id);
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&] { return active_threads_[gid] < max_concurrent_; });
    ++active_threads_[gid];
}

void HalfNodeH2DPacer::release_thread_slot(int gpu_id) {
    const int gid = group_of(gpu_id);
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (active_threads_[gid] > 0) --active_threads_[gid];
    }
    cv_.notify_all();
}

int HalfNodeH2DPacer::acquire_process_flock(int gpu_id, int* out_group_id) {
    const int gid = group_of(gpu_id);
    if (out_group_id) *out_group_id = gid;
    std::error_code ec;
    std::filesystem::create_directories(lock_dir_, ec);
    const std::string path = lock_dir_ + "/pcie_half_" + std::to_string(gid) + ".lock";
    const int fd = ::open(path.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        throw std::system_error(errno, std::generic_category(), "open(" + path + ") failed");
    }
    if (::flock(fd, LOCK_EX) != 0) {
        const int err = errno;
        ::close(fd);
        throw std::system_error(err, std::generic_category(), "flock(" + path + ") failed");
    }
    return fd;
}

void HalfNodeH2DPacer::release_process_flock(int fd) noexcept {
    if (fd >= 0) {
        ::flock(fd, LOCK_UN);
        ::close(fd);
    }
}

}  // namespace morphserve::runtime
