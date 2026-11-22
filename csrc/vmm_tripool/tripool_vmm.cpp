#include "tripool_vmm.hpp"
#include "cuda_driver_min.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

using morphserve::vmm::CudaDriver;

namespace {

constexpr size_t kAlign = 512;

enum class LayerState : int {
    EVICTED = 0,
    QUEUED = 1,
    STREAMING = 2,
    RESIDENT = 3,
};

struct ExpertLayer {
    LayerState state = LayerState::EVICTED;
    bool needs_stream_wait = false;
    size_t used_bytes = 0;  // 0 means copy full layer_stride_bytes
    std::vector<CUmemGenericAllocationHandle> handles;
    CUevent ready_event = nullptr;
};

struct ExpertBank {
    int id = -1;
    int num_layers = 0;
    size_t layer_stride_bytes = 0;
    size_t handles_per_layer = 0;
    CUdeviceptr va_base = 0;
    size_t va_bytes = 0;
    uint8_t* host_pinned_base = nullptr;
    size_t host_bytes = 0;
    bool pinned = false;
    std::vector<ExpertLayer> layers;
};

struct PrefetchJob {
    int expert_id = -1;
    int layer_idx = -1;
};

struct KvRegion {
    CUdeviceptr va_base = 0;
    size_t cap_bytes = 0;
    std::vector<CUmemGenericAllocationHandle> handles;
    bool active = false;
};

class UnifiedTriPoolVMM {
public:
    static UnifiedTriPoolVMM& instance() {
        static UnifiedTriPoolVMM mgr;
        return mgr;
    }

    int init(
        int device,
        size_t super_pool_mib,
        size_t fine_kv_pool_mib,
        size_t super_handle_mib,
        size_t transient_va_gib,
        size_t kv_va_gib,
        int allow_overflow) {
        std::lock_guard<std::mutex> lock(mu_);
        if (initialized_) {
            return (device_ == device) ? 0 : -1;
        }
        try {
            auto& drv = CudaDriver::instance();
            drv.check(drv.cuInit(0), "cuInit");
            drv.check(drv.cuDeviceGet(&cu_dev_, device), "cuDeviceGet");
            drv.check(drv.cuDevicePrimaryCtxRetain(&primary_ctx_, cu_dev_), "cuDevicePrimaryCtxRetain");

            CUcontext cur = nullptr;
            drv.check(drv.cuCtxGetCurrent(&cur), "cuCtxGetCurrent");
            if (!cur) {
                drv.check(drv.cuCtxSetCurrent(primary_ctx_), "cuCtxSetCurrent");
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
                          &min_gran_, &prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                      "get min granularity");

            size_t req_super = (super_handle_mib > 0 ? super_handle_mib : 64) * (1ULL << 20);
            super_bytes_ = ((req_super + min_gran_ - 1) / min_gran_) * min_gran_;
            allow_overflow_ = (allow_overflow != 0);

            // Create high-priority non-blocking H2D stream for expert layer streaming
            int least_prio = 0, greatest_prio = 0;
            if (drv.cuCtxGetStreamPriorityRange(&least_prio, &greatest_prio) == CUDA_SUCCESS) {
                drv.check(drv.cuStreamCreateWithPriority(&h2d_stream_, CU_STREAM_NON_BLOCKING, greatest_prio),
                          "cuStreamCreateWithPriority");
            } else {
                drv.check(drv.cuStreamCreate(&h2d_stream_, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
            }

            // Arena 2: Reserve Transient Scratchpad VA
            size_t t_va_bytes = ((transient_va_gib * (1ULL << 30)) + super_bytes_ - 1) / super_bytes_ * super_bytes_;
            transient_va_handles_ = t_va_bytes / super_bytes_;
            drv.check(drv.cuMemAddressReserve(&transient_va_, t_va_bytes, super_bytes_, 0, 0),
                      "reserve transient VA");

            // Arena 1: Reserve Persistent KV VA (2 MiB aligned)
            size_t kv_va_bytes = ((kv_va_gib * (1ULL << 30)) + min_gran_ - 1) / min_gran_ * min_gran_;
            kv_va_slots_ = kv_va_bytes / min_gran_;
            drv.check(drv.cuMemAddressReserve(&kv_va_, kv_va_bytes, min_gran_, 0, 0),
                      "reserve KV VA");
            kv_slot_handles_.assign(kv_va_slots_, 0);
            kv_free_slots_.reserve(kv_va_slots_);
            for (size_t i = kv_va_slots_; i > 0; --i) {
                kv_free_slots_.push_back(static_cast<int32_t>(i - 1));
            }

            // Pre-allocate Super Physical Pool
            size_t super_n = ((super_pool_mib * (1ULL << 20)) + super_bytes_ - 1) / super_bytes_;
            free_super_handles_.reserve(super_n);
            for (size_t i = 0; i < super_n; ++i) {
                CUmemGenericAllocationHandle h = 0;
                drv.check(drv.cuMemCreate(&h, super_bytes_, &prop_, 0), "prealloc super handle");
                free_super_handles_.push_back(h);
            }
            super_pool_total_ = super_n;

            // Pre-allocate Fine (2 MiB) KV Physical Pool
            size_t fine_n = (fine_kv_pool_mib * (1ULL << 20)) / min_gran_;
            free_fine_handles_.reserve(fine_n);
            for (size_t i = 0; i < fine_n; ++i) {
                CUmemGenericAllocationHandle h = 0;
                drv.check(drv.cuMemCreate(&h, min_gran_, &prop_, 0), "prealloc fine handle");
                free_fine_handles_.push_back(h);
            }
            fine_pool_total_ = fine_n;

            device_ = device;
            stop_streamer_ = false;
            initialized_ = true;
            streamer_thread_ = std::thread(&UnifiedTriPoolVMM::streamer_loop, this);
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] init error: %s\n", e.what());
            return -2;
        }
    }

    int shutdown() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!initialized_) return 0;
            stop_streamer_ = true;
            prefetch_queue_.clear();
            cv_req_.notify_all();
        }
        if (streamer_thread_.joinable()) {
            streamer_thread_.join();
        }

        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_) return 0;
        try {
            auto& drv = CudaDriver::instance();
            ensure_ctx_locked();
            drv.cuStreamSynchronize(h2d_stream_);
            drv.cuCtxSynchronize();

            // Tear down Arena 3 (Experts)
            for (auto& bank : experts_) {
                for (int l = 0; l < bank.num_layers; ++l) {
                    auto& lyr = bank.layers[l];
                    if (!lyr.handles.empty()) {
                        drv.cuMemUnmap(bank.va_base + static_cast<size_t>(l) * bank.layer_stride_bytes,
                                       bank.layer_stride_bytes);
                        for (auto h : lyr.handles) free_super_handles_.push_back(h);
                        lyr.handles.clear();
                    }
                    if (lyr.ready_event) {
                        drv.cuEventDestroy(lyr.ready_event);
                        lyr.ready_event = nullptr;
                    }
                    lyr.state = LayerState::EVICTED;
                }
                if (bank.va_base) {
                    drv.cuMemAddressFree(bank.va_base, bank.va_bytes);
                    bank.va_base = 0;
                }
            }
            experts_.clear();
            expert_mapped_handles_ = 0;
            expert_resident_layers_ = 0;

            // Tear down Arena 2 (Transient)
            if (!transient_mapped_handles_.empty()) {
                drv.cuMemUnmap(transient_va_, transient_mapped_handles_.size() * super_bytes_);
                for (auto h : transient_mapped_handles_) free_super_handles_.push_back(h);
                transient_mapped_handles_.clear();
            }
            transient_live_allocs_.clear();
            transient_free_blocks_.clear();
            if (transient_va_) {
                drv.cuMemAddressFree(transient_va_, transient_va_handles_ * super_bytes_);
                transient_va_ = 0;
            }

            // Tear down Arena 1 contiguous KV regions
            for (auto& reg : kv_regions_) {
                if (reg.active) {
                    if (!reg.handles.empty()) {
                        drv.cuMemUnmap(reg.va_base, reg.handles.size() * min_gran_);
                        for (auto h : reg.handles) free_fine_handles_.push_back(h);
                        reg.handles.clear();
                    }
                    if (reg.va_base) {
                        drv.cuMemAddressFree(reg.va_base, reg.cap_bytes);
                        reg.va_base = 0;
                    }
                    reg.active = false;
                }
            }
            kv_regions_.clear();
            kv_free_regions_.clear();

            // Tear down Arena 1 (KV)
            for (size_t s = 0; s < kv_va_slots_; ++s) {
                if (kv_slot_handles_[s]) {
                    drv.cuMemUnmap(kv_va_ + s * min_gran_, min_gran_);
                    free_fine_handles_.push_back(kv_slot_handles_[s]);
                    kv_slot_handles_[s] = 0;
                }
            }
            kv_slot_handles_.clear();
            kv_free_slots_.clear();
            if (kv_va_) {
                drv.cuMemAddressFree(kv_va_, kv_va_slots_ * min_gran_);
                kv_va_ = 0;
            }

            // Release all physical handles
            for (auto h : free_super_handles_) drv.cuMemRelease(h);
            free_super_handles_.clear();
            for (auto h : free_fine_handles_) drv.cuMemRelease(h);
            free_fine_handles_.clear();

            if (h2d_stream_) {
                drv.cuStreamDestroy(h2d_stream_);
                h2d_stream_ = nullptr;
            }
            if (primary_ctx_) {
                drv.cuDevicePrimaryCtxRelease(cu_dev_);
                primary_ctx_ = nullptr;
            }

            reset_counters_locked();
            initialized_ = false;
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] shutdown error: %s\n", e.what());
            return -1;
        }
    }

    // ==================== Arena 2: Transient Scratchpad (CUDAPluggableAllocator) ====================

    void scope_begin() {
        std::lock_guard<std::mutex> lock(mu_);
        scope_active_ = true;
    }

    int scope_end(void* stream) {
        std::lock_guard<std::mutex> lock(mu_);
        scope_active_ = false;
        if (!initialized_) return 0;
        if (transient_live_allocs_.empty() && !transient_mapped_handles_.empty()) {
            try {
                ensure_ctx_locked();
                bulk_unmap_transient_locked(stream);
                return 0;
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[tripool_vmm] scope_end error: %s\n", e.what());
                return -1;
            }
        }
        return static_cast<int>(transient_live_allocs_.size());
    }

    void* transient_alloc(ssize_t req_size, int device, void* stream) {
        if (req_size <= 0) return nullptr;
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_) {
            std::fprintf(stderr, "[tripool_vmm] transient_alloc called before tripool_vmm_init (dev=%d)\n", device);
            return nullptr;
        }
        try {
            ensure_ctx_locked();
            ++transient_alloc_calls_;
            const size_t aligned = (static_cast<size_t>(req_size) + kAlign - 1) & ~(kAlign - 1);

            for (auto it = transient_free_blocks_.begin(); it != transient_free_blocks_.end(); ++it) {
                if (it->second >= aligned) {
                    size_t off = it->first;
                    size_t rem = it->second - aligned;
                    transient_free_blocks_.erase(it);
                    if (rem > 0) transient_free_blocks_[off + aligned] = rem;
                    transient_live_allocs_[off] = aligned;
                    return reinterpret_cast<void*>(transient_va_ + off);
                }
            }

            const size_t mapped_end = transient_mapped_handles_.size() * super_bytes_;
            size_t tail_free = 0;
            if (!transient_free_blocks_.empty()) {
                auto last = std::prev(transient_free_blocks_.end());
                if (last->first + last->second == mapped_end) {
                    tail_free = last->second;
                }
            }
            const size_t need_bytes = (aligned > tail_free) ? (aligned - tail_free) : 0;
            const size_t add_handles = (need_bytes + super_bytes_ - 1) / super_bytes_;
            if (transient_mapped_handles_.size() + add_handles > transient_va_handles_) {
                throw std::runtime_error("Transient VA exhausted");
            }
            extend_transient_mapped_locked(add_handles, stream);

            auto last = std::prev(transient_free_blocks_.end());
            if (last->second < aligned) {
                throw std::runtime_error("Transient allocator invariant broken");
            }
            size_t off = last->first;
            size_t rem = last->second - aligned;
            transient_free_blocks_.erase(last);
            if (rem > 0) transient_free_blocks_[off + aligned] = rem;
            transient_live_allocs_[off] = aligned;
            return reinterpret_cast<void*>(transient_va_ + off);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] transient_alloc(%zd) error: %s\n", req_size, e.what());
            return nullptr;
        }
    }

    void transient_free(void* ptr, ssize_t /*req_size*/, int /*device*/, void* stream) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_) return;
        try {
            ++transient_free_calls_;
            const CUdeviceptr addr = reinterpret_cast<CUdeviceptr>(ptr);
            if (addr < transient_va_) return;
            const size_t off = static_cast<size_t>(addr - transient_va_);
            auto it = transient_live_allocs_.find(off);
            if (it == transient_live_allocs_.end()) return;
            const size_t sz = it->second;
            transient_live_allocs_.erase(it);
            insert_coalesce_transient_locked(off, sz);

            if (!scope_active_ && transient_live_allocs_.empty() && !transient_mapped_handles_.empty()) {
                ensure_ctx_locked();
                bulk_unmap_transient_locked(stream);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] transient_free error: %s\n", e.what());
        }
    }

    // ==================== Arena 1: Persistent KV (2 MiB Pages & Contiguous Regions) ====================

    CUmemGenericAllocationHandle pop_fine_handle_locked(void* stream) {
        if (free_fine_handles_.empty()) {
            CUmemGenericAllocationHandle super_h = pop_super_handle_locked(stream, /*for_kv=*/true);
            auto& drv = CudaDriver::instance();
            drv.check(drv.cuMemRelease(super_h), "carve cuMemRelease(super)");
            --super_pool_total_;
            const size_t sub_pages = super_bytes_ / min_gran_;
            for (size_t i = 0; i < sub_pages; ++i) {
                CUmemGenericAllocationHandle fh = 0;
                drv.check(drv.cuMemCreate(&fh, min_gran_, &prop_, 0), "carve cuMemCreate(fine)");
                free_fine_handles_.push_back(fh);
                ++fine_pool_total_;
            }
            ++super_to_fine_carves_;
        }
        CUmemGenericAllocationHandle fh = free_fine_handles_.back();
        free_fine_handles_.pop_back();
        return fh;
    }

    uint64_t kv_va_base() {
        std::lock_guard<std::mutex> lock(mu_);
        return static_cast<uint64_t>(kv_va_);
    }

    int32_t kv_region_alloc(size_t max_va_bytes) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_) return -1;
        try {
            ensure_ctx_locked();
            const size_t req = (max_va_bytes > min_gran_) ? max_va_bytes : min_gran_;
            const size_t cap = ((req + min_gran_ - 1) / min_gran_) * min_gran_;
            CUdeviceptr va = 0;
            auto& drv = CudaDriver::instance();
            drv.check(drv.cuMemAddressReserve(&va, cap, min_gran_, 0, 0), "kv_region cuMemAddressReserve");

            int32_t rid = -1;
            if (!kv_free_regions_.empty()) {
                rid = kv_free_regions_.back();
                kv_free_regions_.pop_back();
                kv_regions_[rid] = KvRegion{va, cap, {}, true};
            } else {
                rid = static_cast<int32_t>(kv_regions_.size());
                kv_regions_.push_back(KvRegion{va, cap, {}, true});
            }
            return rid;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] kv_region_alloc error: %s\n", e.what());
            return -2;
        }
    }

    uint64_t kv_region_ensure(int32_t region_id, size_t needed_bytes, void* stream) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_) return 0;
        if (region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return 0;
        auto& reg = kv_regions_[region_id];
        if (!reg.active) return 0;

        const size_t eff_bytes = (needed_bytes > 0) ? needed_bytes : 1;
        const size_t needed_pages = (eff_bytes + min_gran_ - 1) / min_gran_;
        if (needed_pages * min_gran_ > reg.cap_bytes) {
            std::fprintf(stderr, "[tripool_vmm] kv_region_ensure(%d) exceeds cap_bytes=%zu\n",
                         region_id, reg.cap_bytes);
            return 0;
        }
        const size_t old_pages = reg.handles.size();
        if (needed_pages <= old_pages) {
            return static_cast<uint64_t>(reg.va_base);
        }

        std::vector<CUmemGenericAllocationHandle> staged;
        size_t mapped_now = 0;
        auto& drv = CudaDriver::instance();
        try {
            ensure_ctx_locked();
            const size_t delta = needed_pages - old_pages;
            staged.reserve(delta);
            for (size_t i = 0; i < delta; ++i) {
                staged.push_back(pop_fine_handle_locked(stream));
            }
            for (size_t i = 0; i < delta; ++i) {
                CUdeviceptr addr = reg.va_base + (old_pages + i) * min_gran_;
                drv.check(drv.cuMemMap(addr, min_gran_, 0, staged[i], 0), "kv_region cuMemMap");
                ++mapped_now;
            }
            drv.check(
                drv.cuMemSetAccess(reg.va_base + old_pages * min_gran_, delta * min_gran_, &access_, 1),
                "kv_region cuMemSetAccess");
            for (auto h : staged) reg.handles.push_back(h);
            kv_mapped_pages_ += delta;
            kv_mapped_pages_peak_ = std::max(kv_mapped_pages_peak_, kv_mapped_pages_);
            return static_cast<uint64_t>(reg.va_base);
        } catch (const std::exception& e) {
            if (mapped_now > 0) {
                drv.cuMemUnmap(reg.va_base + old_pages * min_gran_, mapped_now * min_gran_);
            }
            for (auto h : staged) free_fine_handles_.push_back(h);
            std::fprintf(stderr, "[tripool_vmm] kv_region_ensure error: %s\n", e.what());
            return 0;
        }
    }

    uint64_t kv_region_va(int32_t region_id) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_ || region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return 0;
        const auto& reg = kv_regions_[region_id];
        return reg.active ? static_cast<uint64_t>(reg.va_base) : 0;
    }

    size_t kv_region_mapped_bytes(int32_t region_id) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_ || region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return 0;
        const auto& reg = kv_regions_[region_id];
        return reg.active ? (reg.handles.size() * min_gran_) : 0;
    }

    int kv_region_free(int32_t region_id, void* stream) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_) return -1;
        if (region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return -2;
        auto& reg = kv_regions_[region_id];
        if (!reg.active) return 0;
        try {
            ensure_ctx_locked();
            auto& drv = CudaDriver::instance();
            if (stream && !reg.handles.empty()) {
                drv.check(drv.cuStreamSynchronize(reinterpret_cast<CUstream>(stream)), "kv_region sync");
            }
            if (!reg.handles.empty()) {
                const size_t n = reg.handles.size();
                drv.check(drv.cuMemUnmap(reg.va_base, n * min_gran_), "kv_region cuMemUnmap");
                for (auto h : reg.handles) free_fine_handles_.push_back(h);
                reg.handles.clear();
                kv_mapped_pages_ = (kv_mapped_pages_ >= n) ? (kv_mapped_pages_ - n) : 0;
            }
            if (reg.va_base) {
                drv.check(drv.cuMemAddressFree(reg.va_base, reg.cap_bytes), "kv_region cuMemAddressFree");
                reg.va_base = 0;
            }
            reg.cap_bytes = 0;
            reg.active = false;
            kv_free_regions_.push_back(region_id);
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] kv_region_free error: %s\n", e.what());
            return -3;
        }
    }

    int32_t kv_alloc_page() {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_) return -1;
        try {
            ensure_ctx_locked();
            if (kv_free_slots_.empty()) return -2;

            CUmemGenericAllocationHandle fh = pop_fine_handle_locked(nullptr);
            int32_t slot = kv_free_slots_.back();
            kv_free_slots_.pop_back();

            auto& drv = CudaDriver::instance();
            CUdeviceptr addr = kv_va_ + static_cast<size_t>(slot) * min_gran_;
            drv.check(drv.cuMemMap(addr, min_gran_, 0, fh, 0), "kv cuMemMap");
            drv.check(drv.cuMemSetAccess(addr, min_gran_, &access_, 1), "kv cuMemSetAccess");
            kv_slot_handles_[slot] = fh;
            ++kv_mapped_pages_;
            kv_mapped_pages_peak_ = std::max(kv_mapped_pages_peak_, kv_mapped_pages_);
            return slot;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] kv_alloc_page error: %s\n", e.what());
            return -3;
        }
    }

    int kv_free_page(int32_t slot_idx, void* stream) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_) return -1;
        if (slot_idx < 0 || static_cast<size_t>(slot_idx) >= kv_va_slots_) return -2;
        if (!kv_slot_handles_[slot_idx]) return 0;
        try {
            ensure_ctx_locked();
            auto& drv = CudaDriver::instance();
            if (stream) {
                drv.check(drv.cuStreamSynchronize(reinterpret_cast<CUstream>(stream)), "kv sync");
            }
            CUdeviceptr addr = kv_va_ + static_cast<size_t>(slot_idx) * min_gran_;
            drv.check(drv.cuMemUnmap(addr, min_gran_), "kv cuMemUnmap");
            free_fine_handles_.push_back(kv_slot_handles_[slot_idx]);
            kv_slot_handles_[slot_idx] = 0;
            kv_free_slots_.push_back(slot_idx);
            if (kv_mapped_pages_ > 0) --kv_mapped_pages_;
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] kv_free_page error: %s\n", e.what());
            return -3;
        }
    }

    // ==================== Arena 3: Opportunistic Expert Cache & Async Streamer ====================

    int expert_register(int num_layers, size_t layer_stride_mib, void* host_pinned_base, size_t host_bytes) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_ || num_layers <= 0 || !host_pinned_base) return -1;
        try {
            ensure_ctx_locked();
            auto& drv = CudaDriver::instance();

            size_t req_stride = layer_stride_mib * (1ULL << 20);
            size_t aligned_stride = ((req_stride + super_bytes_ - 1) / super_bytes_) * super_bytes_;
            size_t handles_per_layer = aligned_stride / super_bytes_;
            size_t total_va = static_cast<size_t>(num_layers) * aligned_stride;
            if (host_bytes < total_va) {
                throw std::runtime_error("host_bytes < num_layers * aligned_stride");
            }

            ExpertBank bank;
            bank.id = static_cast<int>(experts_.size());
            bank.num_layers = num_layers;
            bank.layer_stride_bytes = aligned_stride;
            bank.handles_per_layer = handles_per_layer;
            bank.va_bytes = total_va;
            bank.host_pinned_base = reinterpret_cast<uint8_t*>(host_pinned_base);
            bank.host_bytes = host_bytes;
            bank.pinned = false;
            drv.check(drv.cuMemAddressReserve(&bank.va_base, total_va, super_bytes_, 0, 0),
                      "reserve expert VA");
            bank.layers.resize(num_layers);
            for (int l = 0; l < num_layers; ++l) {
                drv.check(drv.cuEventCreate(&bank.layers[l].ready_event, CU_EVENT_DISABLE_TIMING),
                          "create layer ready_event");
            }
            experts_.push_back(std::move(bank));
            return static_cast<int>(experts_.size() - 1);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] expert_register error: %s\n", e.what());
            return -2;
        }
    }

    uint64_t expert_va_base(int expert_id) {
        std::lock_guard<std::mutex> lock(mu_);
        if (expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return 0;
        return static_cast<uint64_t>(experts_[expert_id].va_base);
    }

    uint64_t expert_layer_va(int expert_id, int layer_idx) {
        std::lock_guard<std::mutex> lock(mu_);
        if (expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return 0;
        const auto& b = experts_[expert_id];
        if (layer_idx < 0 || layer_idx >= b.num_layers) return 0;
        return static_cast<uint64_t>(b.va_base + static_cast<size_t>(layer_idx) * b.layer_stride_bytes);
    }

    size_t expert_layer_stride_bytes(int expert_id) {
        std::lock_guard<std::mutex> lock(mu_);
        if (expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return 0;
        return experts_[expert_id].layer_stride_bytes;
    }

    int expert_set_pinned(int expert_id, int pinned) {
        std::lock_guard<std::mutex> lock(mu_);
        if (expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return -1;
        experts_[expert_id].pinned = (pinned != 0);
        return 0;
    }

    int expert_set_layer_used_bytes(int expert_id, int layer_idx, size_t used_bytes) {
        std::lock_guard<std::mutex> lock(mu_);
        if (expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return -1;
        auto& bank = experts_[expert_id];
        if (layer_idx < 0 || layer_idx >= bank.num_layers) return -2;
        if (used_bytes > bank.layer_stride_bytes) return -3;
        bank.layers[layer_idx].used_bytes = used_bytes;
        return 0;
    }

    int expert_load_layers_sync(int expert_id, int start_layer, int end_layer_exclusive) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_ || expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return -1;
        auto& bank = experts_[expert_id];
        start_layer = std::max(0, start_layer);
        end_layer_exclusive = std::min(bank.num_layers, end_layer_exclusive);
        try {
            ensure_ctx_locked();
            auto& drv = CudaDriver::instance();
            int l = start_layer;
            while (l < end_layer_exclusive) {
                while (l < end_layer_exclusive && bank.layers[l].state != LayerState::EVICTED) ++l;
                int run_start = l;
                while (l < end_layer_exclusive && bank.layers[l].state == LayerState::EVICTED) ++l;
                if (l > run_start) {
                    map_and_submit_contiguous_locked(bank, run_start, l);
                }
            }
            drv.check(drv.cuStreamSynchronize(h2d_stream_), "load_layers_sync wait h2d");
            for (int i = start_layer; i < end_layer_exclusive; ++i) {
                bank.layers[i].needs_stream_wait = false;
            }
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] expert_load_layers_sync error: %s\n", e.what());
            return -2;
        }
    }

    int expert_prefetch_async(int expert_id, int start_layer, int end_layer_exclusive) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_ || expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return -1;
        auto& bank = experts_[expert_id];
        start_layer = std::max(0, start_layer);
        end_layer_exclusive = std::min(bank.num_layers, end_layer_exclusive);
        int queued = 0;
        for (int l = start_layer; l < end_layer_exclusive; ++l) {
            if (bank.layers[l].state == LayerState::EVICTED) {
                bank.layers[l].state = LayerState::QUEUED;
                prefetch_queue_.push_back(PrefetchJob{expert_id, l});
                ++queued;
            }
        }
        if (queued > 0) {
            cv_req_.notify_all();
        }
        return queued;
    }

    int expert_wait_layer(int expert_id, int layer_idx, void* compute_stream) {
        std::unique_lock<std::mutex> lock(mu_);
        if (!initialized_ || expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return -1;
        auto& bank = experts_[expert_id];
        if (layer_idx < 0 || layer_idx >= bank.num_layers) return -2;
        ++expert_wait_calls_;

        auto& lyr = bank.layers[layer_idx];
        last_compute_stream_ = compute_stream;
        // Fast path: already resident and already synchronized with compute stream!
        if (lyr.state == LayerState::RESIDENT && !lyr.needs_stream_wait) {
            return 0;
        }
        if (lyr.state == LayerState::EVICTED) {
            lyr.state = LayerState::QUEUED;
            prefetch_queue_.push_front(PrefetchJob{expert_id, layer_idx});
            cv_req_.notify_all();
        }
        if (lyr.state != LayerState::RESIDENT) {
            ++expert_wait_stalls_;
            cv_ready_.wait(lock, [&] {
                return lyr.state == LayerState::RESIDENT || stop_streamer_ || streamer_error_;
            });
            if (lyr.state != LayerState::RESIDENT) return -3;
        }
        if (lyr.needs_stream_wait) {
            try {
                ensure_ctx_locked();
                auto& drv = CudaDriver::instance();
                drv.check(drv.cuStreamWaitEvent(
                              reinterpret_cast<CUstream>(compute_stream), lyr.ready_event, 0),
                          "cuStreamWaitEvent(expert)");
                lyr.needs_stream_wait = false;
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[tripool_vmm] expert_wait_layer error: %s\n", e.what());
                return -4;
            }
        }
        return 0;
    }

    int expert_evict_layer(int expert_id, int layer_idx, void* stream) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_ || expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return -1;
        auto& bank = experts_[expert_id];
        if (layer_idx < 0 || layer_idx >= bank.num_layers) return -2;
        try {
            ensure_ctx_locked();
            if (evict_layer_locked(bank, layer_idx, stream, /*is_auto=*/false, /*for_kv=*/false)) {
                return 1;
            }
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] expert_evict_layer error: %s\n", e.what());
            return -3;
        }
    }

    int expert_evict_range(int expert_id, int start_layer, int end_layer_exclusive, void* stream) {
        std::unique_lock<std::mutex> lock(mu_);
        if (!initialized_ || expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return -1;
        auto& bank = experts_[expert_id];
        start_layer = std::max(0, start_layer);
        end_layer_exclusive = std::min(bank.num_layers, end_layer_exclusive);
        // Cancel any pending prefetch jobs in this range
        prefetch_queue_.erase(
            std::remove_if(prefetch_queue_.begin(), prefetch_queue_.end(), [&](const PrefetchJob& j) {
                if (j.expert_id == expert_id && j.layer_idx >= start_layer && j.layer_idx < end_layer_exclusive) {
                    if (bank.layers[j.layer_idx].state == LayerState::QUEUED) {
                        bank.layers[j.layer_idx].state = LayerState::EVICTED;
                    }
                    return true;
                }
                return false;
            }),
            prefetch_queue_.end());
        // Wait if streamer thread is currently mid-submit on any layer in this range
        cv_ready_.wait(lock, [&] {
            for (int l = start_layer; l < end_layer_exclusive; ++l) {
                if (bank.layers[l].state == LayerState::STREAMING) return false;
            }
            return true;
        });

        try {
            ensure_ctx_locked();
            auto& drv = CudaDriver::instance();
            drv.check(drv.cuStreamSynchronize(h2d_stream_), "evict_range sync h2d");
            drv.check(drv.cuStreamSynchronize(reinterpret_cast<CUstream>(stream)), "evict_range sync compute");
            int evicted = 0;
            for (int l = end_layer_exclusive - 1; l >= start_layer; --l) {
                if (evict_layer_locked(bank, l, /*stream=*/nullptr, /*is_auto=*/false, /*for_kv=*/false,
                                       /*already_synced=*/true)) {
                    ++evicted;
                }
            }
            return evicted;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] expert_evict_range error: %s\n", e.what());
            return -2;
        }
    }

    int expert_is_resident(int expert_id, int layer_idx) {
        std::lock_guard<std::mutex> lock(mu_);
        if (expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return 0;
        const auto& b = experts_[expert_id];
        if (layer_idx < 0 || layer_idx >= b.num_layers) return 0;
        return b.layers[layer_idx].state == LayerState::RESIDENT ? 1 : 0;
    }

    int expert_resident_count(int expert_id) {
        std::lock_guard<std::mutex> lock(mu_);
        if (expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return 0;
        int cnt = 0;
        for (const auto& lyr : experts_[expert_id].layers) {
            if (lyr.state == LayerState::RESIDENT) ++cnt;
        }
        return cnt;
    }

    void get_stats(UnifiedVmmStats* out) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!out) return;
        std::memset(out, 0, sizeof(*out));
        out->min_granularity_bytes = min_gran_;
        out->super_handle_bytes = super_bytes_;
        out->super_pool_total_handles = super_pool_total_;
        out->super_pool_free_handles = free_super_handles_.size();
        out->fine_pool_total_pages = fine_pool_total_;
        out->fine_pool_free_pages = free_fine_handles_.size();
        out->kv_mapped_pages_current = kv_mapped_pages_;
        out->kv_mapped_pages_peak = kv_mapped_pages_peak_;
        out->super_to_fine_carves = super_to_fine_carves_;
        out->transient_mapped_handles_current = transient_mapped_handles_.size();
        out->transient_mapped_handles_peak = transient_mapped_handles_peak_;
        out->transient_live_allocs = transient_live_allocs_.size();
        out->transient_alloc_calls = transient_alloc_calls_;
        out->transient_free_calls = transient_free_calls_;
        out->transient_scope_flushes = transient_scope_flushes_;
        out->expert_mapped_handles_current = expert_mapped_handles_;
        out->expert_resident_layers_total = expert_resident_layers_;
        out->expert_prefetch_layers_completed = expert_prefetch_layers_completed_;
        out->expert_wait_calls = expert_wait_calls_;
        out->expert_wait_stalls = expert_wait_stalls_;
        out->expert_manual_evictions = expert_manual_evictions_;
        out->expert_auto_evictions_for_transient = expert_auto_evictions_for_transient_;
        out->expert_auto_evictions_for_kv = expert_auto_evictions_for_kv_;
        out->expert_h2d_bytes_total = expert_h2d_bytes_total_;
        out->overflow_creates = overflow_creates_;
        out->transient_map_us = transient_map_us_;
        out->transient_unmap_us = transient_unmap_us_;
        out->expert_map_us = expert_map_us_;
        out->expert_unmap_us = expert_unmap_us_;
        out->expert_h2d_submit_us = expert_h2d_submit_us_;
    }

private:
    void ensure_ctx_locked() {
        auto& drv = CudaDriver::instance();
        CUcontext cur = nullptr;
        drv.check(drv.cuCtxGetCurrent(&cur), "cuCtxGetCurrent");
        if (!cur && primary_ctx_) {
            drv.check(drv.cuCtxSetCurrent(primary_ctx_), "cuCtxSetCurrent");
        }
    }

    void reset_counters_locked() {
        super_pool_total_ = 0;
        fine_pool_total_ = 0;
        kv_mapped_pages_ = 0;
        kv_mapped_pages_peak_ = 0;
        super_to_fine_carves_ = 0;
        transient_mapped_handles_peak_ = 0;
        transient_alloc_calls_ = 0;
        transient_free_calls_ = 0;
        transient_scope_flushes_ = 0;
        expert_mapped_handles_ = 0;
        expert_resident_layers_ = 0;
        expert_prefetch_layers_completed_ = 0;
        expert_wait_calls_ = 0;
        expert_wait_stalls_ = 0;
        expert_manual_evictions_ = 0;
        expert_auto_evictions_for_transient_ = 0;
        expert_auto_evictions_for_kv_ = 0;
        expert_h2d_bytes_total_ = 0;
        overflow_creates_ = 0;
        transient_map_us_ = 0.0;
        transient_unmap_us_ = 0.0;
        expert_map_us_ = 0.0;
        expert_unmap_us_ = 0.0;
        expert_h2d_submit_us_ = 0.0;
        scope_active_ = false;
        streamer_error_ = false;
    }

    bool evict_layer_locked(
        ExpertBank& bank,
        int layer_idx,
        void* stream,
        bool is_auto,
        bool for_kv,
        bool already_synced = false) {
        auto& lyr = bank.layers[layer_idx];
        if (lyr.state != LayerState::RESIDENT || lyr.handles.empty()) return false;
        auto& drv = CudaDriver::instance();
        if (!already_synced) {
            drv.check(drv.cuStreamSynchronize(h2d_stream_), "evict sync h2d");
            drv.check(drv.cuStreamSynchronize(reinterpret_cast<CUstream>(stream)), "evict sync compute");
            if (last_compute_stream_ && last_compute_stream_ != stream) {
                drv.check(drv.cuStreamSynchronize(reinterpret_cast<CUstream>(last_compute_stream_)),
                          "evict sync last_compute");
            }
        }
        CUdeviceptr base = bank.va_base + static_cast<size_t>(layer_idx) * bank.layer_stride_bytes;
        auto t0 = std::chrono::steady_clock::now();
        drv.check(drv.cuMemUnmap(base, bank.layer_stride_bytes), "expert cuMemUnmap");
        auto t1 = std::chrono::steady_clock::now();
        expert_unmap_us_ += std::chrono::duration<double, std::micro>(t1 - t0).count();

        const size_t n = lyr.handles.size();
        for (auto h : lyr.handles) free_super_handles_.push_back(h);
        lyr.handles.clear();
        lyr.state = LayerState::EVICTED;
        lyr.needs_stream_wait = false;
        if (expert_mapped_handles_ >= n) expert_mapped_handles_ -= n;
        if (expert_resident_layers_ > 0) --expert_resident_layers_;
        if (is_auto) {
            if (for_kv) ++expert_auto_evictions_for_kv_;
            else ++expert_auto_evictions_for_transient_;
        } else {
            ++expert_manual_evictions_;
        }
        return true;
    }

    bool evict_one_tail_expert_layer_locked(void* stream, bool for_kv, int exclude_expert_id = -1) {
        // Evict highest-index resident layer from any unpinned expert (drop-clean, zero writeback!)
        for (auto& bank : experts_) {
            if (bank.pinned || bank.id == exclude_expert_id) continue;
            for (int l = bank.num_layers - 1; l >= 0; --l) {
                if (bank.layers[l].state == LayerState::RESIDENT) {
                    if (evict_layer_locked(bank, l, stream, /*is_auto=*/true, for_kv)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    CUmemGenericAllocationHandle pop_super_handle_locked(
        void* stream, bool for_kv, int exclude_expert_id = -1) {
        while (free_super_handles_.empty()) {
            if (!evict_one_tail_expert_layer_locked(stream, for_kv, exclude_expert_id)) {
                break;
            }
        }
        if (!free_super_handles_.empty()) {
            CUmemGenericAllocationHandle h = free_super_handles_.back();
            free_super_handles_.pop_back();
            return h;
        }
        if (!allow_overflow_) {
            throw std::runtime_error("TriPool super-handle pool exhausted (allow_overflow=0)");
        }
        auto& drv = CudaDriver::instance();
        CUmemGenericAllocationHandle h = 0;
        drv.check(drv.cuMemCreate(&h, super_bytes_, &prop_, 0), "overflow cuMemCreate(super)");
        ++super_pool_total_;
        ++overflow_creates_;
        return h;
    }

    // Batch-map a contiguous range [start_layer, end_layer) with ONE single cuMemSetAccess
    // (avoids per-layer GPU MMU TLB shootdown while DMA is active!), then submit per-layer
    // H2D DMA + ready_event so compute_stream can pipeline layer-by-layer with zero mid-DMA stall.
    void map_and_submit_contiguous_locked(ExpertBank& bank, int start_layer, int end_layer) {
        if (start_layer >= end_layer) return;
        auto& drv = CudaDriver::instance();
        const int count = end_layer - start_layer;
        const size_t total_handles = static_cast<size_t>(count) * bank.handles_per_layer;

        std::vector<CUmemGenericAllocationHandle> all_handles;
        all_handles.reserve(total_handles);
        size_t mapped_count = 0;
        const CUdeviceptr batch_va =
            bank.va_base + static_cast<size_t>(start_layer) * bank.layer_stride_bytes;
        const size_t batch_bytes = static_cast<size_t>(count) * bank.layer_stride_bytes;

        try {
            for (size_t i = 0; i < total_handles; ++i) {
                all_handles.push_back(pop_super_handle_locked(nullptr, /*for_kv=*/false, bank.id));
            }

            auto tm0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < total_handles; ++i) {
                drv.check(drv.cuMemMap(batch_va + i * super_bytes_, super_bytes_, 0, all_handles[i], 0),
                          "expert batch cuMemMap");
                ++mapped_count;
            }
            drv.check(drv.cuMemSetAccess(batch_va, batch_bytes, &access_, 1),
                      "expert batch cuMemSetAccess");
            auto tm1 = std::chrono::steady_clock::now();
            expert_map_us_ += std::chrono::duration<double, std::micro>(tm1 - tm0).count();

            auto th0 = std::chrono::steady_clock::now();
            for (int idx = 0; idx < count; ++idx) {
                const int l = start_layer + idx;
                auto& lyr = bank.layers[l];
                const size_t byte_off = static_cast<size_t>(l) * bank.layer_stride_bytes;
                const CUdeviceptr dst_va = bank.va_base + byte_off;
                const uint8_t* src_host = bank.host_pinned_base + byte_off;
                const size_t copy_bytes =
                    (lyr.used_bytes > 0 && lyr.used_bytes <= bank.layer_stride_bytes)
                        ? lyr.used_bytes
                        : bank.layer_stride_bytes;

                drv.check(drv.cuMemcpyHtoDAsync(dst_va, src_host, copy_bytes, h2d_stream_),
                          "expert cuMemcpyHtoDAsync");
                drv.check(drv.cuEventRecord(lyr.ready_event, h2d_stream_), "expert cuEventRecord");

                lyr.handles.assign(
                    all_handles.begin() + static_cast<std::ptrdiff_t>(idx * bank.handles_per_layer),
                    all_handles.begin() + static_cast<std::ptrdiff_t>((idx + 1) * bank.handles_per_layer));
                lyr.state = LayerState::RESIDENT;
                lyr.needs_stream_wait = true;
                expert_mapped_handles_ += bank.handles_per_layer;
                ++expert_resident_layers_;
                ++expert_prefetch_layers_completed_;
                expert_h2d_bytes_total_ += copy_bytes;
            }
            auto th1 = std::chrono::steady_clock::now();
            expert_h2d_submit_us_ += std::chrono::duration<double, std::micro>(th1 - th0).count();
        } catch (...) {
            if (mapped_count > 0) {
                drv.cuMemUnmap(batch_va, mapped_count * super_bytes_);
            }
            for (auto h : all_handles) {
                free_super_handles_.push_back(h);
            }
            throw;
        }
    }

    void streamer_loop() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            try {
                CudaDriver::instance().check(
                    CudaDriver::instance().cuCtxSetCurrent(primary_ctx_), "streamer cuCtxSetCurrent");
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[tripool_vmm] streamer ctx error: %s\n", e.what());
                streamer_error_ = true;
                cv_ready_.notify_all();
                return;
            }
        }

        while (true) {
            std::unique_lock<std::mutex> lock(mu_);
            cv_req_.wait(lock, [&] { return stop_streamer_ || !prefetch_queue_.empty(); });
            if (stop_streamer_) return;

            PrefetchJob first = prefetch_queue_.front();
            prefetch_queue_.pop_front();
            if (first.expert_id < 0 || static_cast<size_t>(first.expert_id) >= experts_.size()) continue;
            auto& bank = experts_[first.expert_id];
            if (first.layer_idx < 0 || first.layer_idx >= bank.num_layers) continue;
            if (bank.layers[first.layer_idx].state != LayerState::QUEUED) continue;

            // Coalesce contiguous queued layers of the same expert into one TLB-friendly batch map!
            int start_l = first.layer_idx;
            int end_l = start_l + 1;
            while (!prefetch_queue_.empty()) {
                const auto& nxt = prefetch_queue_.front();
                if (nxt.expert_id == first.expert_id &&
                    nxt.layer_idx == end_l &&
                    end_l < bank.num_layers &&
                    bank.layers[end_l].state == LayerState::QUEUED) {
                    prefetch_queue_.pop_front();
                    ++end_l;
                } else {
                    break;
                }
            }
            for (int l = start_l; l < end_l; ++l) {
                bank.layers[l].state = LayerState::STREAMING;
            }
            try {
                ensure_ctx_locked();
                map_and_submit_contiguous_locked(bank, start_l, end_l);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[tripool_vmm] streamer layers [%d,%d) error: %s\n",
                             start_l, end_l, e.what());
                for (int l = start_l; l < end_l; ++l) {
                    if (bank.layers[l].state != LayerState::RESIDENT) {
                        bank.layers[l].state = LayerState::EVICTED;
                    }
                }
                streamer_error_ = true;
            }
            lock.unlock();
            cv_ready_.notify_all();
        }
    }

    void extend_transient_mapped_locked(size_t add_handles, void* stream) {
        if (add_handles == 0) return;
        auto& drv = CudaDriver::instance();
        const size_t start_idx = transient_mapped_handles_.size();
        const size_t old_end = start_idx * super_bytes_;
        const size_t add_bytes = add_handles * super_bytes_;

        // First acquire all required super-handles (auto-evicting clean tail expert layers if needed!)
        std::vector<CUmemGenericAllocationHandle> acquired;
        acquired.reserve(add_handles);
        for (size_t i = 0; i < add_handles; ++i) {
            acquired.push_back(pop_super_handle_locked(stream, /*for_kv=*/false));
        }

        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < add_handles; ++i) {
            drv.check(drv.cuMemMap(transient_va_ + (start_idx + i) * super_bytes_,
                                   super_bytes_, 0, acquired[i], 0),
                      "transient cuMemMap");
            transient_mapped_handles_.push_back(acquired[i]);
        }
        drv.check(drv.cuMemSetAccess(transient_va_ + old_end, add_bytes, &access_, 1),
                  "transient cuMemSetAccess");
        auto t1 = std::chrono::steady_clock::now();
        transient_map_us_ += std::chrono::duration<double, std::micro>(t1 - t0).count();
        transient_mapped_handles_peak_ = std::max(
            transient_mapped_handles_peak_, static_cast<uint64_t>(transient_mapped_handles_.size()));

        insert_coalesce_transient_locked(old_end, add_bytes);
    }

    void insert_coalesce_transient_locked(size_t off, size_t sz) {
        auto next = transient_free_blocks_.lower_bound(off);
        if (next != transient_free_blocks_.begin()) {
            auto prev = std::prev(next);
            if (prev->first + prev->second == off) {
                off = prev->first;
                sz += prev->second;
                transient_free_blocks_.erase(prev);
            }
        }
        if (next != transient_free_blocks_.end() && off + sz == next->first) {
            sz += next->second;
            transient_free_blocks_.erase(next);
        }
        transient_free_blocks_[off] = sz;
    }

    void bulk_unmap_transient_locked(void* stream) {
        auto& drv = CudaDriver::instance();
        drv.check(drv.cuStreamSynchronize(reinterpret_cast<CUstream>(stream)), "transient bulk sync");
        const size_t total_bytes = transient_mapped_handles_.size() * super_bytes_;
        auto t0 = std::chrono::steady_clock::now();
        drv.check(drv.cuMemUnmap(transient_va_, total_bytes), "transient bulk cuMemUnmap");
        auto t1 = std::chrono::steady_clock::now();
        transient_unmap_us_ += std::chrono::duration<double, std::micro>(t1 - t0).count();

        for (auto h : transient_mapped_handles_) free_super_handles_.push_back(h);
        transient_mapped_handles_.clear();
        transient_free_blocks_.clear();
        ++transient_scope_flushes_;
    }

    std::mutex mu_;
    std::condition_variable cv_req_;
    std::condition_variable cv_ready_;
    std::thread streamer_thread_;
    bool stop_streamer_ = false;
    bool streamer_error_ = false;

    bool initialized_ = false;
    bool scope_active_ = false;
    bool allow_overflow_ = false;
    int device_ = -1;
    CUdevice cu_dev_ = 0;
    CUcontext primary_ctx_ = nullptr;
    CUstream h2d_stream_ = nullptr;
    void* last_compute_stream_ = nullptr;
    CUmemAllocationProp prop_ = {};
    CUmemAccessDesc access_ = {};
    size_t min_gran_ = 2ULL << 20;
    size_t super_bytes_ = 64ULL << 20;

    // Physical page pools
    std::vector<CUmemGenericAllocationHandle> free_super_handles_;
    std::vector<CUmemGenericAllocationHandle> free_fine_handles_;
    uint64_t super_pool_total_ = 0;
    uint64_t fine_pool_total_ = 0;

    // Arena 1: Persistent KV
    CUdeviceptr kv_va_ = 0;
    size_t kv_va_slots_ = 0;
    std::vector<CUmemGenericAllocationHandle> kv_slot_handles_;
    std::vector<int32_t> kv_free_slots_;
    std::vector<KvRegion> kv_regions_;
    std::vector<int32_t> kv_free_regions_;
    uint64_t kv_mapped_pages_ = 0;
    uint64_t kv_mapped_pages_peak_ = 0;
    uint64_t super_to_fine_carves_ = 0;

    // Arena 2: Transient Scratchpad
    CUdeviceptr transient_va_ = 0;
    size_t transient_va_handles_ = 0;
    std::vector<CUmemGenericAllocationHandle> transient_mapped_handles_;
    std::map<size_t, size_t> transient_free_blocks_;
    std::unordered_map<size_t, size_t> transient_live_allocs_;
    uint64_t transient_mapped_handles_peak_ = 0;
    uint64_t transient_alloc_calls_ = 0;
    uint64_t transient_free_calls_ = 0;
    uint64_t transient_scope_flushes_ = 0;
    double transient_map_us_ = 0.0;
    double transient_unmap_us_ = 0.0;

    // Arena 3: Opportunistic Expert Cache
    std::vector<ExpertBank> experts_;
    std::deque<PrefetchJob> prefetch_queue_;
    uint64_t expert_mapped_handles_ = 0;
    uint64_t expert_resident_layers_ = 0;
    uint64_t expert_prefetch_layers_completed_ = 0;
    uint64_t expert_wait_calls_ = 0;
    uint64_t expert_wait_stalls_ = 0;
    uint64_t expert_manual_evictions_ = 0;
    uint64_t expert_auto_evictions_for_transient_ = 0;
    uint64_t expert_auto_evictions_for_kv_ = 0;
    uint64_t expert_h2d_bytes_total_ = 0;
    double expert_map_us_ = 0.0;
    double expert_unmap_us_ = 0.0;
    double expert_h2d_submit_us_ = 0.0;

    uint64_t overflow_creates_ = 0;
};

}  // namespace

extern "C" {

int tripool_vmm_init(
    int device,
    size_t super_pool_mib,
    size_t fine_kv_pool_mib,
    size_t super_handle_mib,
    size_t transient_va_gib,
    size_t kv_va_gib,
    int allow_overflow) {
    return UnifiedTriPoolVMM::instance().init(
        device, super_pool_mib, fine_kv_pool_mib, super_handle_mib,
        transient_va_gib, kv_va_gib, allow_overflow);
}

int tripool_vmm_shutdown() {
    return UnifiedTriPoolVMM::instance().shutdown();
}

void tripool_vmm_get_stats(UnifiedVmmStats* out) {
    UnifiedTriPoolVMM::instance().get_stats(out);
}

void* tripool_alloc(ssize_t size, int device, void* stream) {
    return UnifiedTriPoolVMM::instance().transient_alloc(size, device, stream);
}

void tripool_free(void* ptr, ssize_t size, int device, void* stream) {
    UnifiedTriPoolVMM::instance().transient_free(ptr, size, device, stream);
}

void tripool_scope_begin() {
    UnifiedTriPoolVMM::instance().scope_begin();
}

int tripool_scope_end(void* stream) {
    return UnifiedTriPoolVMM::instance().scope_end(stream);
}

uint64_t tripool_kv_va_base() {
    return UnifiedTriPoolVMM::instance().kv_va_base();
}

int32_t tripool_kv_alloc_page() {
    return UnifiedTriPoolVMM::instance().kv_alloc_page();
}

int tripool_kv_free_page(int32_t slot_idx, void* stream) {
    return UnifiedTriPoolVMM::instance().kv_free_page(slot_idx, stream);
}

int32_t tripool_kv_region_alloc(size_t max_va_bytes) {
    return UnifiedTriPoolVMM::instance().kv_region_alloc(max_va_bytes);
}

uint64_t tripool_kv_region_ensure(int32_t region_id, size_t needed_bytes, void* stream) {
    return UnifiedTriPoolVMM::instance().kv_region_ensure(region_id, needed_bytes, stream);
}

uint64_t tripool_kv_region_va(int32_t region_id) {
    return UnifiedTriPoolVMM::instance().kv_region_va(region_id);
}

size_t tripool_kv_region_mapped_bytes(int32_t region_id) {
    return UnifiedTriPoolVMM::instance().kv_region_mapped_bytes(region_id);
}

int tripool_kv_region_free(int32_t region_id, void* stream) {
    return UnifiedTriPoolVMM::instance().kv_region_free(region_id, stream);
}

int tripool_expert_register(
    int num_layers,
    size_t layer_stride_mib,
    void* host_pinned_base,
    size_t host_bytes) {
    return UnifiedTriPoolVMM::instance().expert_register(
        num_layers, layer_stride_mib, host_pinned_base, host_bytes);
}

uint64_t tripool_expert_va_base(int expert_id) {
    return UnifiedTriPoolVMM::instance().expert_va_base(expert_id);
}

uint64_t tripool_expert_layer_va(int expert_id, int layer_idx) {
    return UnifiedTriPoolVMM::instance().expert_layer_va(expert_id, layer_idx);
}

size_t tripool_expert_layer_stride_bytes(int expert_id) {
    return UnifiedTriPoolVMM::instance().expert_layer_stride_bytes(expert_id);
}

int tripool_expert_set_pinned(int expert_id, int pinned) {
    return UnifiedTriPoolVMM::instance().expert_set_pinned(expert_id, pinned);
}

int tripool_expert_set_layer_used_bytes(int expert_id, int layer_idx, size_t used_bytes) {
    return UnifiedTriPoolVMM::instance().expert_set_layer_used_bytes(expert_id, layer_idx, used_bytes);
}

int tripool_expert_load_layers_sync(int expert_id, int start_layer, int end_layer_exclusive) {
    return UnifiedTriPoolVMM::instance().expert_load_layers_sync(
        expert_id, start_layer, end_layer_exclusive);
}

int tripool_expert_prefetch_async(int expert_id, int start_layer, int end_layer_exclusive) {
    return UnifiedTriPoolVMM::instance().expert_prefetch_async(
        expert_id, start_layer, end_layer_exclusive);
}

int tripool_expert_wait_layer(int expert_id, int layer_idx, void* compute_stream) {
    return UnifiedTriPoolVMM::instance().expert_wait_layer(expert_id, layer_idx, compute_stream);
}

int tripool_expert_evict_layer(int expert_id, int layer_idx, void* stream) {
    return UnifiedTriPoolVMM::instance().expert_evict_layer(expert_id, layer_idx, stream);
}

int tripool_expert_evict_range(int expert_id, int start_layer, int end_layer_exclusive, void* stream) {
    return UnifiedTriPoolVMM::instance().expert_evict_range(
        expert_id, start_layer, end_layer_exclusive, stream);
}

int tripool_expert_is_resident(int expert_id, int layer_idx) {
    return UnifiedTriPoolVMM::instance().expert_is_resident(expert_id, layer_idx);
}

int tripool_expert_resident_count(int expert_id) {
    return UnifiedTriPoolVMM::instance().expert_resident_count(expert_id);
}

int tripool_host_register(void* ptr, size_t nbytes) {
    try {
        auto& drv = CudaDriver::instance();
        drv.check(drv.cuInit(0), "cuInit");
        drv.check(drv.cuMemHostRegister(ptr, nbytes, CU_MEMHOSTREGISTER_PORTABLE), "cuMemHostRegister");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tripool_vmm] host_register error: %s\n", e.what());
        return -1;
    }
}

int tripool_host_unregister(void* ptr) {
    try {
        auto& drv = CudaDriver::instance();
        drv.check(drv.cuMemHostUnregister(ptr), "cuMemHostUnregister");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tripool_vmm] host_unregister error: %s\n", e.what());
        return -1;
    }
}

}  // extern "C"
