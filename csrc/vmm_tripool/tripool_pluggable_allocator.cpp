#include "cuda_driver_min.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>

using morphserve::vmm::CudaDriver;

extern "C" {

struct TripoolStats {
    uint64_t granularity_bytes;
    uint64_t pool_total_pages;
    uint64_t pool_free_pages;
    uint64_t mapped_pages_current;
    uint64_t mapped_pages_peak;
    uint64_t small_slab_pages_current;
    uint64_t overflow_creates;
    uint64_t alloc_calls;
    uint64_t free_calls;
    double total_map_us;
    double total_unmap_us;
};

}  // extern "C"

namespace {

constexpr size_t kAlign = 512;
constexpr size_t kSmallThreshold = 1ULL << 20;  // <= 1 MiB goes to 2MB slabs

struct LargeAlloc {
    size_t slot_start = 0;
    size_t num_pages = 0;
    std::vector<CUmemGenericAllocationHandle> handles;
};

struct SmallSlab {
    size_t slot = 0;
    CUdeviceptr base = 0;
    CUmemGenericAllocationHandle handle = 0;
    size_t offset = 0;
    size_t active_count = 0;
};

class TriPoolTransientManager {
public:
    static TriPoolTransientManager& instance() {
        static TriPoolTransientManager mgr;
        return mgr;
    }

    int init(int device, size_t pool_mib, size_t va_gib) {
        std::lock_guard<std::mutex> lock(mu_);
        return init_locked(device, pool_mib, va_gib);
    }

    int init_locked(int device, size_t pool_mib, size_t va_gib) {
        if (initialized_) {
            return (device_ == device) ? 0 : -1;
        }
        try {
            auto& drv = CudaDriver::instance();
            drv.check(drv.cuInit(0), "cuInit");
            drv.check(drv.cuDeviceGet(&cu_dev_, device), "cuDeviceGet");

            drv.check(drv.cuDevicePrimaryCtxRetain(&retained_ctx_, cu_dev_), "retain primary ctx");
            CUcontext cur = nullptr;
            drv.check(drv.cuCtxGetCurrent(&cur), "cuCtxGetCurrent");
            if (!cur) {
                drv.check(drv.cuCtxSetCurrent(retained_ctx_), "set primary ctx");
            }

            prop_ = {};
            prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop_.location.id = device;

            access_ = {};
            access_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access_.location.id = device;
            access_.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

            drv.check(drv.cuMemGetAllocationGranularity(
                          &gran_, &prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                      "get granularity");

            size_t va_bytes = ((va_gib * (1ULL << 30)) + gran_ - 1) / gran_ * gran_;
            total_slots_ = va_bytes / gran_;
            drv.check(drv.cuMemAddressReserve(&va_base_, va_bytes, gran_, 0, 0), "reserve VA");
            slot_used_.assign(total_slots_, 0);

            size_t pool_pages = ((pool_mib * (1ULL << 20)) + gran_ - 1) / gran_;
            free_handles_.reserve(pool_pages);
            for (size_t i = 0; i < pool_pages; ++i) {
                CUmemGenericAllocationHandle h = 0;
                drv.check(drv.cuMemCreate(&h, gran_, &prop_, 0), "prealloc cuMemCreate");
                free_handles_.push_back(h);
            }
            pool_total_pages_ = pool_pages;
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
        if (!cur && retained_ctx_) {
            drv.check(drv.cuCtxSetCurrent(retained_ctx_), "cuCtxSetCurrent");
        }
    }

    void* alloc(ssize_t req_size, int device, void* /*stream*/) {
        if (req_size <= 0) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_) {
            if (init_locked(device, 4096, 16) != 0 || !initialized_) return nullptr;
        }
        try {
            ensure_ctx_locked();
            const size_t size = static_cast<size_t>(req_size);
            ++alloc_calls_;

            if (size <= kSmallThreshold && size <= gran_) {
                return alloc_small_locked(size);
            }
            return alloc_large_locked(size);
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
            ensure_ctx_locked();
            ++free_calls_;
            const CUdeviceptr addr = reinterpret_cast<CUdeviceptr>(ptr);

            auto lit = large_allocs_.find(addr);
            if (lit != large_allocs_.end()) {
                sync_stream_locked(stream);
                unmap_pages_locked(lit->second.slot_start, lit->second.num_pages, lit->second.handles);
                large_allocs_.erase(lit);
                return;
            }

            auto sit = small_ptr_to_slab_.find(addr);
            if (sit != small_ptr_to_slab_.end()) {
                size_t slab_slot = sit->second;
                small_ptr_to_slab_.erase(sit);
                auto slab_it = slabs_.find(slab_slot);
                if (slab_it != slabs_.end()) {
                    SmallSlab& slab = slab_it->second;
                    if (slab.active_count > 0) --slab.active_count;
                    if (slab.active_count == 0) {
                        sync_stream_locked(stream);
                        std::vector<CUmemGenericAllocationHandle> one{slab.handle};
                        unmap_pages_locked(slab.slot, 1, one);
                        --small_slab_pages_current_;
                        slabs_.erase(slab_it);
                    }
                }
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_pluggable] free error: %s\n", e.what());
        }
    }

    void get_stats(TripoolStats* out) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!out) return;
        out->granularity_bytes = gran_;
        out->pool_total_pages = pool_total_pages_;
        out->pool_free_pages = free_handles_.size();
        out->mapped_pages_current = mapped_pages_current_;
        out->mapped_pages_peak = mapped_pages_peak_;
        out->small_slab_pages_current = small_slab_pages_current_;
        out->overflow_creates = overflow_creates_;
        out->alloc_calls = alloc_calls_;
        out->free_calls = free_calls_;
        out->total_map_us = total_map_us_;
        out->total_unmap_us = total_unmap_us_;
    }

    int shutdown() {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_) return 0;
        try {
            auto& drv = CudaDriver::instance();
            drv.cuCtxSynchronize();
            for (auto& kv : large_allocs_) {
                unmap_pages_locked(kv.second.slot_start, kv.second.num_pages, kv.second.handles);
            }
            large_allocs_.clear();
            for (auto& kv : slabs_) {
                std::vector<CUmemGenericAllocationHandle> one{kv.second.handle};
                unmap_pages_locked(kv.second.slot, 1, one);
            }
            slabs_.clear();
            small_ptr_to_slab_.clear();
            small_slab_pages_current_ = 0;

            for (auto h : free_handles_) {
                drv.cuMemRelease(h);
            }
            free_handles_.clear();
            drv.cuMemAddressFree(va_base_, total_slots_ * gran_);
            va_base_ = 0;
            total_slots_ = 0;
            slot_used_.clear();
            pool_total_pages_ = 0;
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

    size_t granularity() const { return gran_; }

private:
    size_t find_free_slots_locked(size_t count) {
        size_t run = 0;
        size_t start = 0;
        for (size_t i = 0; i < total_slots_; ++i) {
            if (!slot_used_[i]) {
                if (run == 0) start = i;
                ++run;
                if (run == count) {
                    for (size_t j = start; j < start + count; ++j) slot_used_[j] = 1;
                    return start;
                }
            } else {
                run = 0;
            }
        }
        throw std::runtime_error("TriPool VA range exhausted");
    }

    CUmemGenericAllocationHandle pop_handle_locked() {
        if (!free_handles_.empty()) {
            CUmemGenericAllocationHandle h = free_handles_.back();
            free_handles_.pop_back();
            return h;
        }
        auto& drv = CudaDriver::instance();
        CUmemGenericAllocationHandle h = 0;
        drv.check(drv.cuMemCreate(&h, gran_, &prop_, 0), "overflow cuMemCreate");
        ++pool_total_pages_;
        ++overflow_creates_;
        return h;
    }

    void map_pages_locked(size_t slot_start, const std::vector<CUmemGenericAllocationHandle>& handles) {
        auto& drv = CudaDriver::instance();
        auto t0 = std::chrono::steady_clock::now();
        CUdeviceptr base = va_base_ + slot_start * gran_;
        for (size_t i = 0; i < handles.size(); ++i) {
            drv.check(drv.cuMemMap(base + i * gran_, gran_, 0, handles[i], 0), "cuMemMap");
        }
        drv.check(drv.cuMemSetAccess(base, handles.size() * gran_, &access_, 1), "cuMemSetAccess");
        auto t1 = std::chrono::steady_clock::now();
        total_map_us_ += std::chrono::duration<double, std::micro>(t1 - t0).count();
        mapped_pages_current_ += handles.size();
        mapped_pages_peak_ = std::max(mapped_pages_peak_, mapped_pages_current_);
    }

    void unmap_pages_locked(size_t slot_start, size_t num_pages,
                            const std::vector<CUmemGenericAllocationHandle>& handles) {
        auto& drv = CudaDriver::instance();
        auto t0 = std::chrono::steady_clock::now();
        CUdeviceptr base = va_base_ + slot_start * gran_;
        drv.check(drv.cuMemUnmap(base, num_pages * gran_), "cuMemUnmap");
        auto t1 = std::chrono::steady_clock::now();
        total_unmap_us_ += std::chrono::duration<double, std::micro>(t1 - t0).count();

        for (size_t j = slot_start; j < slot_start + num_pages; ++j) slot_used_[j] = 0;
        for (auto h : handles) free_handles_.push_back(h);
        if (mapped_pages_current_ >= num_pages) mapped_pages_current_ -= num_pages;
        else mapped_pages_current_ = 0;
    }

    void sync_stream_locked(void* stream) {
        auto& drv = CudaDriver::instance();
        if (stream) {
            drv.check(drv.cuStreamSynchronize(reinterpret_cast<CUstream>(stream)), "cuStreamSynchronize");
        } else {
            drv.check(drv.cuStreamSynchronize(nullptr), "cuStreamSynchronize(default)");
        }
    }

    void* alloc_large_locked(size_t size) {
        size_t num_pages = (size + gran_ - 1) / gran_;
        size_t slot_start = find_free_slots_locked(num_pages);
        std::vector<CUmemGenericAllocationHandle> handles;
        handles.reserve(num_pages);
        for (size_t i = 0; i < num_pages; ++i) {
            handles.push_back(pop_handle_locked());
        }
        map_pages_locked(slot_start, handles);
        CUdeviceptr addr = va_base_ + slot_start * gran_;
        large_allocs_[addr] = LargeAlloc{slot_start, num_pages, std::move(handles)};
        return reinterpret_cast<void*>(addr);
    }

    void* alloc_small_locked(size_t size) {
        size_t aligned = (size + kAlign - 1) & ~(kAlign - 1);
        for (auto& kv : slabs_) {
            SmallSlab& slab = kv.second;
            if (slab.offset + aligned <= gran_) {
                CUdeviceptr addr = slab.base + slab.offset;
                slab.offset += aligned;
                ++slab.active_count;
                small_ptr_to_slab_[addr] = slab.slot;
                return reinterpret_cast<void*>(addr);
            }
        }
        size_t slot = find_free_slots_locked(1);
        CUmemGenericAllocationHandle h = pop_handle_locked();
        std::vector<CUmemGenericAllocationHandle> one{h};
        map_pages_locked(slot, one);
        CUdeviceptr base = va_base_ + slot * gran_;
        SmallSlab slab{slot, base, h, aligned, 1};
        slabs_[slot] = slab;
        ++small_slab_pages_current_;
        small_ptr_to_slab_[base] = slot;
        return reinterpret_cast<void*>(base);
    }

    std::mutex mu_;
    bool initialized_ = false;
    int device_ = -1;
    CUdevice cu_dev_ = 0;
    CUcontext retained_ctx_ = nullptr;
    CUmemAllocationProp prop_ = {};
    CUmemAccessDesc access_ = {};
    size_t gran_ = 2ULL << 20;

    CUdeviceptr va_base_ = 0;
    size_t total_slots_ = 0;
    std::vector<uint8_t> slot_used_;

    std::vector<CUmemGenericAllocationHandle> free_handles_;
    std::unordered_map<CUdeviceptr, LargeAlloc> large_allocs_;
    std::unordered_map<size_t, SmallSlab> slabs_;
    std::unordered_map<CUdeviceptr, size_t> small_ptr_to_slab_;

    uint64_t pool_total_pages_ = 0;
    uint64_t mapped_pages_current_ = 0;
    uint64_t mapped_pages_peak_ = 0;
    uint64_t small_slab_pages_current_ = 0;
    uint64_t overflow_creates_ = 0;
    uint64_t alloc_calls_ = 0;
    uint64_t free_calls_ = 0;
    double total_map_us_ = 0.0;
    double total_unmap_us_ = 0.0;
};

}  // namespace

extern "C" {

int tripool_init(int device, size_t pool_mib, size_t va_gib) {
    return TriPoolTransientManager::instance().init(device, pool_mib, va_gib);
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

size_t tripool_granularity() {
    return TriPoolTransientManager::instance().granularity();
}

}  // extern "C"
