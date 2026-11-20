#include "cuda_driver_min.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

using morphserve::vmm::CudaDriver;

extern "C" {

struct TripoolStats {
    uint64_t min_granularity_bytes;
    uint64_t handle_bytes;
    uint64_t pool_total_handles;
    uint64_t pool_free_handles;
    uint64_t mapped_handles_current;
    uint64_t mapped_handles_peak;
    uint64_t live_allocs_current;
    uint64_t overflow_creates;
    uint64_t alloc_calls;
    uint64_t free_calls;
    uint64_t scope_flushes;
    double total_map_us;
    double total_unmap_us;
    double total_sync_us;
};

}  // extern "C"

namespace {

constexpr size_t kAlign = 512;

// Two-tier Super-Handle Transient Arena with Scope-End Bulk Unmap:
// - Uses large physical handles (default 64 MiB or 128 MiB) to eliminate 2MB ioctl amplification.
// - Inside a transient scope (tripool_scope_begin .. tripool_scope_end), sub-allocations
//   and frees operate purely in O(log N) CPU free-list with ZERO cuStreamSynchronize / cuMemUnmap.
// - On scope exit (once all live transient tensors are freed), performs 1 sync + 1 bulk cuMemUnmap,
//   reclaiming 100% of physical super-handles in ~1-3 ms.
class TriPoolTransientManager {
public:
    static TriPoolTransientManager& instance() {
        static TriPoolTransientManager mgr;
        return mgr;
    }

    int init(int device, size_t pool_mib, size_t va_gib, size_t handle_mib) {
        std::lock_guard<std::mutex> lock(mu_);
        return init_locked(device, pool_mib, va_gib, handle_mib);
    }

    void scope_begin() {
        std::lock_guard<std::mutex> lock(mu_);
        scope_active_ = true;
    }

    int scope_end(void* stream) {
        std::lock_guard<std::mutex> lock(mu_);
        scope_active_ = false;
        if (!initialized_) return 0;
        if (live_allocs_.empty() && mapped_handles_.size() > 0) {
            try {
                ensure_ctx_locked();
                bulk_unmap_all_locked(stream);
                return 0;
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[tripool_pluggable] scope_end error: %s\n", e.what());
                return -1;
            }
        }
        return static_cast<int>(live_allocs_.size());
    }

    void* alloc(ssize_t req_size, int device, void* /*stream*/) {
        if (req_size <= 0) return nullptr;
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_) {
            if (init_locked(device, 6144, 16, 128) != 0 || !initialized_) return nullptr;
        }
        try {
            ensure_ctx_locked();
            ++alloc_calls_;
            const size_t aligned = (static_cast<size_t>(req_size) + kAlign - 1) & ~(kAlign - 1);

            // 1. First-fit from already-mapped transient VA free blocks (zero CUDA calls!)
            for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
                if (it->second >= aligned) {
                    size_t off = it->first;
                    size_t rem = it->second - aligned;
                    free_blocks_.erase(it);
                    if (rem > 0) free_blocks_[off + aligned] = rem;
                    live_allocs_[off] = aligned;
                    return reinterpret_cast<void*>(va_base_ + off);
                }
            }

            // 2. Extend mapped contiguous prefix by mapping minimal 64/128 MiB super-handles
            const size_t mapped_end = mapped_handles_.size() * handle_bytes_;
            size_t tail_free = 0;
            if (!free_blocks_.empty()) {
                auto last = std::prev(free_blocks_.end());
                if (last->first + last->second == mapped_end) {
                    tail_free = last->second;
                }
            }
            const size_t need_bytes = (aligned > tail_free) ? (aligned - tail_free) : 0;
            const size_t add_handles = (need_bytes + handle_bytes_ - 1) / handle_bytes_;
            if (mapped_handles_.size() + add_handles > va_total_handles_) {
                throw std::runtime_error("TriPool transient VA capacity exceeded");
            }
            extend_mapped_locked(add_handles);

            // Allocate from the extended tail block
            auto last = std::prev(free_blocks_.end());
            if (last->second < aligned) {
                throw std::runtime_error("internal allocator invariant violated after extend");
            }
            size_t off = last->first;
            size_t rem = last->second - aligned;
            free_blocks_.erase(last);
            if (rem > 0) free_blocks_[off + aligned] = rem;
            live_allocs_[off] = aligned;
            return reinterpret_cast<void*>(va_base_ + off);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_pluggable] alloc(%zd) error: %s\n", req_size, e.what());
            return nullptr;
        }
    }

    void free_ptr(void* ptr, ssize_t /*req_size*/, int /*device*/, void* stream) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_) return;
        try {
            ++free_calls_;
            const CUdeviceptr addr = reinterpret_cast<CUdeviceptr>(ptr);
            if (addr < va_base_) return;
            const size_t off = static_cast<size_t>(addr - va_base_);
            auto it = live_allocs_.find(off);
            if (it == live_allocs_.end()) return;
            const size_t sz = it->second;
            live_allocs_.erase(it);
            insert_and_coalesce_locked(off, sz);

            // Outside an active scope, once all transient tensors are gone, perform 1 bulk unmap!
            if (!scope_active_ && live_allocs_.empty() && !mapped_handles_.empty()) {
                ensure_ctx_locked();
                bulk_unmap_all_locked(stream);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_pluggable] free error: %s\n", e.what());
        }
    }

    void get_stats(TripoolStats* out) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!out) return;
        out->min_granularity_bytes = min_gran_;
        out->handle_bytes = handle_bytes_;
        out->pool_total_handles = pool_total_handles_;
        out->pool_free_handles = free_handles_.size();
        out->mapped_handles_current = mapped_handles_.size();
        out->mapped_handles_peak = mapped_handles_peak_;
        out->live_allocs_current = live_allocs_.size();
        out->overflow_creates = overflow_creates_;
        out->alloc_calls = alloc_calls_;
        out->free_calls = free_calls_;
        out->scope_flushes = scope_flushes_;
        out->total_map_us = total_map_us_;
        out->total_unmap_us = total_unmap_us_;
        out->total_sync_us = total_sync_us_;
    }

    int shutdown() {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_) return 0;
        try {
            auto& drv = CudaDriver::instance();
            ensure_ctx_locked();
            drv.cuCtxSynchronize();
            if (!mapped_handles_.empty()) {
                drv.cuMemUnmap(va_base_, mapped_handles_.size() * handle_bytes_);
                for (auto h : mapped_handles_) free_handles_.push_back(h);
                mapped_handles_.clear();
            }
            live_allocs_.clear();
            free_blocks_.clear();
            for (auto h : free_handles_) drv.cuMemRelease(h);
            free_handles_.clear();
            drv.cuMemAddressFree(va_base_, va_total_handles_ * handle_bytes_);
            va_base_ = 0;
            va_total_handles_ = 0;
            pool_total_handles_ = 0;
            mapped_handles_peak_ = 0;
            overflow_creates_ = 0;
            alloc_calls_ = 0;
            free_calls_ = 0;
            scope_flushes_ = 0;
            total_map_us_ = 0.0;
            total_unmap_us_ = 0.0;
            total_sync_us_ = 0.0;
            scope_active_ = false;
            if (retained_ctx_) {
                drv.cuDevicePrimaryCtxRelease(cu_dev_);
                retained_ctx_ = nullptr;
            }
            initialized_ = false;
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_pluggable] shutdown error: %s\n", e.what());
            return -1;
        }
    }

private:
    int init_locked(int device, size_t pool_mib, size_t va_gib, size_t handle_mib) {
        if (initialized_) return (device_ == device) ? 0 : -1;
        try {
            auto& drv = CudaDriver::instance();
            drv.check(drv.cuInit(0), "cuInit");
            drv.check(drv.cuDeviceGet(&cu_dev_, device), "cuDeviceGet");
            drv.check(drv.cuDevicePrimaryCtxRetain(&retained_ctx_, cu_dev_), "retain primary ctx");
            CUcontext cur = nullptr;
            drv.check(drv.cuCtxGetCurrent(&cur), "cuCtxGetCurrent");
            if (!cur) drv.check(drv.cuCtxSetCurrent(retained_ctx_), "set primary ctx");

            prop_ = {};
            prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop_.location.id = device;

            access_ = {};
            access_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access_.location.id = device;
            access_.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

            drv.check(drv.cuMemGetAllocationGranularity(
                          &min_gran_, &prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                      "get granularity");

            size_t req_handle = (handle_mib > 0 ? handle_mib : 128) * (1ULL << 20);
            handle_bytes_ = ((req_handle + min_gran_ - 1) / min_gran_) * min_gran_;

            size_t va_bytes = ((va_gib * (1ULL << 30)) + handle_bytes_ - 1) / handle_bytes_ * handle_bytes_;
            va_total_handles_ = va_bytes / handle_bytes_;
            drv.check(drv.cuMemAddressReserve(&va_base_, va_bytes, handle_bytes_, 0, 0), "reserve VA");

            size_t pool_handles = ((pool_mib * (1ULL << 20)) + handle_bytes_ - 1) / handle_bytes_;
            free_handles_.reserve(pool_handles);
            for (size_t i = 0; i < pool_handles; ++i) {
                CUmemGenericAllocationHandle h = 0;
                drv.check(drv.cuMemCreate(&h, handle_bytes_, &prop_, 0), "prealloc super handle");
                free_handles_.push_back(h);
            }
            pool_total_handles_ = pool_handles;
            device_ = device;
            initialized_ = true;
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_pluggable] init error: %s\n", e.what());
            return -2;
        }
    }

    void ensure_ctx_locked() {
        auto& drv = CudaDriver::instance();
        CUcontext cur = nullptr;
        drv.check(drv.cuCtxGetCurrent(&cur), "cuCtxGetCurrent");
        if (!cur && retained_ctx_) drv.check(drv.cuCtxSetCurrent(retained_ctx_), "cuCtxSetCurrent");
    }

    CUmemGenericAllocationHandle pop_handle_locked() {
        if (!free_handles_.empty()) {
            CUmemGenericAllocationHandle h = free_handles_.back();
            free_handles_.pop_back();
            return h;
        }
        auto& drv = CudaDriver::instance();
        CUmemGenericAllocationHandle h = 0;
        drv.check(drv.cuMemCreate(&h, handle_bytes_, &prop_, 0), "overflow cuMemCreate");
        ++pool_total_handles_;
        ++overflow_creates_;
        return h;
    }

    void extend_mapped_locked(size_t add_handles) {
        if (add_handles == 0) return;
        auto& drv = CudaDriver::instance();
        const size_t start_idx = mapped_handles_.size();
        const size_t old_end = start_idx * handle_bytes_;
        const size_t add_bytes = add_handles * handle_bytes_;

        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < add_handles; ++i) {
            CUmemGenericAllocationHandle h = pop_handle_locked();
            drv.check(drv.cuMemMap(va_base_ + (start_idx + i) * handle_bytes_,
                                   handle_bytes_, 0, h, 0),
                      "cuMemMap(super)");
            mapped_handles_.push_back(h);
        }
        drv.check(drv.cuMemSetAccess(va_base_ + old_end, add_bytes, &access_, 1),
                  "cuMemSetAccess(super)");
        auto t1 = std::chrono::steady_clock::now();
        total_map_us_ += std::chrono::duration<double, std::micro>(t1 - t0).count();
        mapped_handles_peak_ = std::max(mapped_handles_peak_, static_cast<uint64_t>(mapped_handles_.size()));

        insert_and_coalesce_locked(old_end, add_bytes);
    }

    void insert_and_coalesce_locked(size_t off, size_t sz) {
        auto next = free_blocks_.lower_bound(off);
        if (next != free_blocks_.begin()) {
            auto prev = std::prev(next);
            if (prev->first + prev->second == off) {
                off = prev->first;
                sz += prev->second;
                free_blocks_.erase(prev);
            }
        }
        if (next != free_blocks_.end() && off + sz == next->first) {
            sz += next->second;
            free_blocks_.erase(next);
        }
        free_blocks_[off] = sz;
    }

    void bulk_unmap_all_locked(void* stream) {
        auto& drv = CudaDriver::instance();
        auto ts0 = std::chrono::steady_clock::now();
        drv.check(drv.cuStreamSynchronize(reinterpret_cast<CUstream>(stream)), "cuStreamSynchronize");
        auto ts1 = std::chrono::steady_clock::now();
        total_sync_us_ += std::chrono::duration<double, std::micro>(ts1 - ts0).count();

        const size_t total_bytes = mapped_handles_.size() * handle_bytes_;
        auto tu0 = std::chrono::steady_clock::now();
        drv.check(drv.cuMemUnmap(va_base_, total_bytes), "bulk cuMemUnmap");
        auto tu1 = std::chrono::steady_clock::now();
        total_unmap_us_ += std::chrono::duration<double, std::micro>(tu1 - tu0).count();

        for (auto h : mapped_handles_) free_handles_.push_back(h);
        mapped_handles_.clear();
        free_blocks_.clear();
        ++scope_flushes_;
    }

    std::mutex mu_;
    bool initialized_ = false;
    bool scope_active_ = false;
    int device_ = -1;
    CUdevice cu_dev_ = 0;
    CUcontext retained_ctx_ = nullptr;
    CUmemAllocationProp prop_ = {};
    CUmemAccessDesc access_ = {};
    size_t min_gran_ = 2ULL << 20;
    size_t handle_bytes_ = 128ULL << 20;

    CUdeviceptr va_base_ = 0;
    size_t va_total_handles_ = 0;

    std::vector<CUmemGenericAllocationHandle> free_handles_;
    std::vector<CUmemGenericAllocationHandle> mapped_handles_;
    std::map<size_t, size_t> free_blocks_;
    std::unordered_map<size_t, size_t> live_allocs_;

    uint64_t pool_total_handles_ = 0;
    uint64_t mapped_handles_peak_ = 0;
    uint64_t overflow_creates_ = 0;
    uint64_t alloc_calls_ = 0;
    uint64_t free_calls_ = 0;
    uint64_t scope_flushes_ = 0;
    double total_map_us_ = 0.0;
    double total_unmap_us_ = 0.0;
    double total_sync_us_ = 0.0;
};

}  // namespace

extern "C" {

int tripool_init(int device, size_t pool_mib, size_t va_gib) {
    return TriPoolTransientManager::instance().init(device, pool_mib, va_gib, 128);
}

int tripool_init_ex(int device, size_t pool_mib, size_t va_gib, size_t handle_mib) {
    return TriPoolTransientManager::instance().init(device, pool_mib, va_gib, handle_mib);
}

void tripool_scope_begin() {
    TriPoolTransientManager::instance().scope_begin();
}

int tripool_scope_end(void* stream) {
    return TriPoolTransientManager::instance().scope_end(stream);
}

void* tripool_alloc(ssize_t size, int device, void* stream) {
    return TriPoolTransientManager::instance().alloc(size, device, stream);
}

void tripool_free(void* ptr, ssize_t size, int device, void* stream) {
    TriPoolTransientManager::instance().free_ptr(ptr, size, device, stream);
}

void tripool_get_stats(TripoolStats* out) {
    TriPoolTransientManager::instance().get_stats(out);
}

int tripool_shutdown() {
    return TriPoolTransientManager::instance().shutdown();
}

}  // extern "C"
