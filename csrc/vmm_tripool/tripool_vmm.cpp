#include "tripool_vmm.hpp"
#include "cuda_driver_min.hpp"

#include <algorithm>
#include <atomic>
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
constexpr size_t kMaxIdleHostPinnedBytes = 2ULL << 30;  // 2 GiB cap on cached idle cuMemHostAlloc blocks

enum class LayerState : int {
    EVICTED = 0,
    QUEUED = 1,
    STREAMING = 2,
    RESIDENT = 3,
};

enum class KvRegionState : int {
    RESIDENT = 0,
    SWAP_OUT_QUEUED = 1,
    SWAP_OUT_FINISHING = 2,
    SWAPPED = 3,
};

// Bidirectional super-slot backing: either 1 x super_bytes_ (64 MiB) handle
// OR sub_pages_ x min_gran_ (e.g. 32 x 2 MiB) fine handles mapped contiguously.
struct SuperSlot {
    CUmemGenericAllocationHandle super_h = 0;
    std::vector<CUmemGenericAllocationHandle> fine_hs;

    bool is_fine_bundle() const { return !fine_hs.empty(); }
};

struct ExpertLayer {
    LayerState state = LayerState::EVICTED;
    bool needs_stream_wait = false;
    size_t used_bytes = 0;  // 0 means copy full layer_stride_bytes
    std::vector<SuperSlot> slots;
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

struct SwapReapJob {
    int32_t region_id = -1;
    uint64_t generation = 0;
};

struct KvRegion {
    CUdeviceptr va_base = 0;
    size_t cap_bytes = 0;
    std::vector<CUmemGenericAllocationHandle> handles;
    void* swap_host_pinned = nullptr;
    size_t swap_cap_bytes = 0;
    size_t swapped_valid_bytes = 0;
    size_t swapped_pages = 0;
    CUevent swap_event = nullptr;
    uint64_t generation = 0;
    KvRegionState state = KvRegionState::RESIDENT;
    bool evictable = false;
    size_t evictable_valid_bytes = 0;
    uint64_t evictable_seq = 0;
    uint64_t evictable_stream_u64 = 0;
    bool active = false;
};

struct CapacitySnapshot {
    uint64_t pool_total_bytes = 0;
    uint64_t pool_free_bytes = 0;
    uint64_t expert_unpinned_bytes = 0;
    uint64_t kv_free_bytes = 0;
    uint64_t kv_dit_headroom_bytes = 0;
};

class UnifiedTriPoolVMM {
public:
    static UnifiedTriPoolVMM& instance() {
        static UnifiedTriPoolVMM mgr;
        return mgr;
    }

    UnifiedTriPoolVMM() = default;
    UnifiedTriPoolVMM(const UnifiedTriPoolVMM&) = delete;
    UnifiedTriPoolVMM& operator=(const UnifiedTriPoolVMM&) = delete;

    ~UnifiedTriPoolVMM() noexcept {
        // Static singleton teardown guard: stop and join `streamer_thread_` without invoking
        // `CudaDriver` APIs (since `libcuda.so` TLS may already be torn down at `atexit`).
        // Normal runtime CUDA resource release happens in `shutdown()`.
        try {
            {
                std::lock_guard<std::mutex> lock(mu_);
                atexit_teardown_.store(true, std::memory_order_release);
                stop_streamer_ = true;
                prefetch_queue_.clear();
                swap_reap_queue_.clear();
                cv_req_.notify_all();
                cv_ready_.notify_all();
            }
            if (streamer_thread_.joinable()) {
                streamer_thread_.join();
            }
        } catch (...) {
        }
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
            if (cur != primary_ctx_) {
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
            sub_pages_ = std::max<size_t>(1, super_bytes_ / min_gran_);
            allow_overflow_ = (allow_overflow != 0);

            // High-priority non-blocking H2D stream for expert layer streaming
            int least_prio = 0, greatest_prio = 0;
            if (drv.cuCtxGetStreamPriorityRange(&least_prio, &greatest_prio) == CUDA_SUCCESS) {
                drv.check(drv.cuStreamCreateWithPriority(&h2d_stream_, CU_STREAM_NON_BLOCKING, greatest_prio),
                          "cuStreamCreateWithPriority");
            } else {
                drv.check(drv.cuStreamCreate(&h2d_stream_, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
            }

            // Dedicated non-blocking stream & bridge event for async KV swap-out / swap-in
            drv.check(drv.cuStreamCreate(&swap_stream_, CU_STREAM_NON_BLOCKING), "cuStreamCreate(swap)");
            drv.check(drv.cuEventCreate(&compute_bridge_event_, CU_EVENT_DISABLE_TIMING),
                      "cuEventCreate(bridge)");

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
            size_t super_n =
                (super_pool_mib > 0) ? (((super_pool_mib * (1ULL << 20)) + super_bytes_ - 1) / super_bytes_) : 0;
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
            base_fine_pages_ = fine_n;

            device_ = device;
            atexit_teardown_.store(false, std::memory_order_relaxed);
            stop_streamer_ = false;
            streamer_error_ = false;
            streamer_thread_ = std::thread(&UnifiedTriPoolVMM::streamer_loop, this);
            initialized_ = true;
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] init error: %s\n", e.what());
            cleanup_failed_init_locked();
            return -2;
        }
    }

    int shutdown() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!initialized_) return 0;
            stop_streamer_ = true;
            prefetch_queue_.clear();
            swap_reap_queue_.clear();
            cv_req_.notify_all();
            cv_ready_.notify_all();
        }
        if (streamer_thread_.joinable()) {
            streamer_thread_.join();
        }

        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_) return 0;
        try {
            auto& drv = CudaDriver::instance();
            ensure_ctx_locked();
            if (h2d_stream_) drv.cuStreamSynchronize(h2d_stream_);
            if (swap_stream_) drv.cuStreamSynchronize(swap_stream_);
            drv.cuCtxSynchronize();

            // Tear down Arena 3 (Experts)
            for (auto& bank : experts_) {
                for (int l = 0; l < bank.num_layers; ++l) {
                    auto& lyr = bank.layers[l];
                    if (!lyr.slots.empty()) {
                        drv.cuMemUnmap(bank.va_base + static_cast<size_t>(l) * bank.layer_stride_bytes,
                                       bank.layer_stride_bytes);
                        for (auto& s : lyr.slots) release_super_slot_to_pool_locked(s);
                        lyr.slots.clear();
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
            expert_fine_bundles_current_ = 0;
            expert_resident_layers_ = 0;

            // Tear down Arena 2 (Transient)
            if (!transient_mapped_slots_.empty()) {
                drv.cuMemUnmap(transient_va_, transient_mapped_slots_.size() * super_bytes_);
                for (auto& s : transient_mapped_slots_) release_super_slot_to_pool_locked(s);
                transient_mapped_slots_.clear();
            }
            transient_fine_bundles_current_ = 0;
            transient_live_allocs_.clear();
            transient_free_blocks_.clear();
            if (transient_va_) {
                drv.cuMemAddressFree(transient_va_, transient_va_handles_ * super_bytes_);
                transient_va_ = 0;
            }

            // Tear down Arena 1 contiguous KV regions & pinned swap buffers
            for (auto& reg : kv_regions_) {
                if (reg.active) {
                    if (!reg.handles.empty()) {
                        drv.cuMemUnmap(reg.va_base, reg.handles.size() * min_gran_);
                        for (auto h : reg.handles) free_fine_handles_.push_back(h);
                        reg.handles.clear();
                    }
                    if (reg.swap_host_pinned) {
                        drv.cuMemFreeHost(reg.swap_host_pinned);
                        reg.swap_host_pinned = nullptr;
                        reg.swap_cap_bytes = 0;
                    }
                    if (reg.swap_event) {
                        drv.cuEventDestroy(reg.swap_event);
                        reg.swap_event = nullptr;
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

            for (auto& kv : free_host_blocks_) {
                for (void* p : kv.second) {
                    if (p) drv.cuMemFreeHost(p);
                }
            }
            free_host_blocks_.clear();

            // Tear down Arena 1 (KV paged slots)
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

            if (compute_bridge_event_) {
                drv.cuEventDestroy(compute_bridge_event_);
                compute_bridge_event_ = nullptr;
            }
            if (swap_stream_) {
                drv.cuStreamDestroy(swap_stream_);
                swap_stream_ = nullptr;
            }
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
        if (transient_live_allocs_.empty() && !transient_mapped_slots_.empty()) {
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
        std::unique_lock<std::mutex> lock(mu_);
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

            const size_t mapped_end = transient_mapped_slots_.size() * super_bytes_;
            size_t tail_free = 0;
            if (!transient_free_blocks_.empty()) {
                auto last = std::prev(transient_free_blocks_.end());
                if (last->first + last->second == mapped_end) {
                    tail_free = last->second;
                }
            }
            const size_t need_bytes = (aligned > tail_free) ? (aligned - tail_free) : 0;
            const size_t add_handles = (need_bytes + super_bytes_ - 1) / super_bytes_;
            if (transient_mapped_slots_.size() + add_handles > transient_va_handles_) {
                throw std::runtime_error("Transient VA exhausted");
            }
            extend_transient_mapped_locked(add_handles, stream, &lock);

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

            if (!scope_active_ && transient_live_allocs_.empty() && !transient_mapped_slots_.empty()) {
                ensure_ctx_locked();
                bulk_unmap_transient_locked(stream);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] transient_free error: %s\n", e.what());
        }
    }

    // ==================== Arena 1: Persistent KV (2 MiB Pages, Contiguous Regions & Async Host Swap) ====================

    CUmemGenericAllocationHandle pop_fine_handle_locked(
        void* stream,
        std::unique_lock<std::mutex>* opt_lock = nullptr,
        int32_t exclude_region_id = -1) {
        auto& drv = CudaDriver::instance();
        while (free_fine_handles_.empty()) {
            if (!free_super_handles_.empty()) {
                CUmemGenericAllocationHandle super_h = free_super_handles_.back();
                free_super_handles_.pop_back();
                drv.check(drv.cuMemRelease(super_h), "carve cuMemRelease(super)");
                --super_pool_total_;
                for (size_t i = 0; i < sub_pages_; ++i) {
                    CUmemGenericAllocationHandle fh = 0;
                    drv.check(drv.cuMemCreate(&fh, min_gran_, &prop_, 0), "carve cuMemCreate(fine)");
                    free_fine_handles_.push_back(fh);
                    ++fine_pool_total_;
                }
                ++super_to_fine_carves_;
                break;
            }
            if (drain_one_queued_swap_out_locked()) {
                continue;
            }
            if (opt_lock && has_finishing_swap_locked()) {
                cv_ready_.wait(*opt_lock, [&] { return !has_finishing_swap_locked() || stop_streamer_; });
                if (stop_streamer_) {
                    throw std::runtime_error("TriPool shutting down during KV page allocation");
                }
                continue;
            }
            if (evict_one_tail_expert_layer_locked(stream, /*for_kv=*/true)) {
                // Evicting a layer may have returned fine handles directly OR super handles!
                continue;
            }
            if (evict_one_thinking_kv_region_locked(stream, exclude_region_id)) {
                continue;
            }
            if (!allow_overflow_) {
                throw std::runtime_error("TriPool physical pool exhausted for KV page (allow_overflow=0)");
            }
            CUmemGenericAllocationHandle fh = 0;
            drv.check(drv.cuMemCreate(&fh, min_gran_, &prop_, 0), "overflow cuMemCreate(fine)");
            free_fine_handles_.push_back(fh);
            ++fine_pool_total_;
            ++overflow_creates_;
            break;
        }
        CUmemGenericAllocationHandle fh = free_fine_handles_.back();
        free_fine_handles_.pop_back();
        return fh;
    }

    uint64_t kv_va_base() {
        std::lock_guard<std::mutex> lock(mu_);
        return static_cast<uint64_t>(kv_va_);
    }

    uint64_t kv_free_bytes(int for_dit) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_) return 0;
        const CapacitySnapshot snap = compute_kv_capacities_locked();
        return for_dit ? snap.kv_dit_headroom_bytes : snap.kv_free_bytes;
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

            KvRegion fresh;
            fresh.va_base = va;
            fresh.cap_bytes = cap;
            fresh.state = KvRegionState::RESIDENT;
            fresh.active = true;

            int32_t rid = -1;
            if (!kv_free_regions_.empty()) {
                rid = kv_free_regions_.back();
                kv_free_regions_.pop_back();
                fresh.generation = kv_regions_[rid].generation + 1;
                kv_regions_[rid] = std::move(fresh);
            } else {
                rid = static_cast<int32_t>(kv_regions_.size());
                kv_regions_.push_back(std::move(fresh));
            }
            return rid;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] kv_region_alloc error: %s\n", e.what());
            return -2;
        }
    }

    int kv_region_set_evictable(int32_t region_id, int evictable, size_t valid_bytes, void* compute_stream) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_) return -1;
        if (region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return -2;
        auto& reg = kv_regions_[region_id];
        if (!reg.active) return -2;
        if (evictable != 0) {
            reg.evictable = true;
            reg.evictable_valid_bytes = valid_bytes;
            reg.evictable_seq = ++evictable_clock_;
            reg.evictable_stream_u64 = reinterpret_cast<uint64_t>(compute_stream);
        } else {
            reg.evictable = false;
            reg.evictable_valid_bytes = 0;
        }
        return 0;
    }

    uint64_t kv_region_ensure(int32_t region_id, size_t needed_bytes, void* stream) {
        std::unique_lock<std::mutex> lock(mu_);
        if (!initialized_) return 0;
        if (region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return 0;
        auto& reg = kv_regions_[region_id];
        if (!reg.active) return 0;
        reg.evictable = false;

        const size_t eff_bytes = (needed_bytes > 0) ? needed_bytes : 1;
        const size_t needed_pages = (eff_bytes + min_gran_ - 1) / min_gran_;
        if (needed_pages * min_gran_ > reg.cap_bytes) {
            std::fprintf(stderr, "[tripool_vmm] kv_region_ensure(%d) exceeds cap_bytes=%zu\n",
                         region_id, reg.cap_bytes);
            return 0;
        }

        auto& drv = CudaDriver::instance();
        try {
            ensure_ctx_locked();
            // Transparently cancel or swap-in if region was parked in Host Pinned DRAM!
            if (reg.state == KvRegionState::SWAP_OUT_FINISHING) {
                cv_ready_.wait(lock, [&] {
                    return reg.state != KvRegionState::SWAP_OUT_FINISHING || stop_streamer_;
                });
                if (stop_streamer_ || !reg.active) return 0;
            }
            if (reg.state == KvRegionState::SWAP_OUT_QUEUED) {
                cancel_queued_swap_out_locked(reg, stream);
            } else if (reg.state == KvRegionState::SWAPPED) {
                swap_in_region_locked(region_id, reg, stream, &lock);
                if (!reg.active) return 0;
            }

            const size_t old_pages = reg.handles.size();
            if (needed_pages <= old_pages) {
                return static_cast<uint64_t>(reg.va_base);
            }

            std::vector<CUmemGenericAllocationHandle> staged;
            size_t mapped_now = 0;
            try {
                const size_t delta = needed_pages - old_pages;
                staged.reserve(delta);
                for (size_t i = 0; i < delta; ++i) {
                    staged.push_back(pop_fine_handle_locked(stream, &lock, region_id));
                }
                if (!reg.active) {
                    for (auto h : staged) free_fine_handles_.push_back(h);
                    return 0;
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
            } catch (...) {
                if (mapped_now > 0) {
                    drv.cuMemUnmap(reg.va_base + old_pages * min_gran_, mapped_now * min_gran_);
                }
                for (auto h : staged) free_fine_handles_.push_back(h);
                throw;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] kv_region_ensure error: %s\n", e.what());
            return 0;
        }
    }

    int kv_region_swap_out_async(int32_t region_id, size_t valid_bytes, void* compute_stream) {
        std::unique_lock<std::mutex> lock(mu_);
        if (!initialized_) return -1;
        if (region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return -2;
        auto& reg = kv_regions_[region_id];
        if (!reg.active) return -2;
        reg.evictable = false;
        if (reg.state != KvRegionState::RESIDENT || reg.handles.empty()) {
            return 0;
        }
        try {
            ensure_ctx_locked();
            auto& drv = CudaDriver::instance();
            const size_t n_pages = reg.handles.size();
            const size_t mapped_bytes = n_pages * min_gran_;
            const size_t copy_bytes = std::min(valid_bytes, mapped_bytes);

            if (copy_bytes == 0) {
                drv.check(drv.cuStreamSynchronize(reinterpret_cast<CUstream>(compute_stream)),
                          "kv swap_out empty sync");
                if (reg.swap_event) {
                    drv.check(drv.cuEventSynchronize(reg.swap_event), "kv swap_out empty event sync");
                }
                drv.check(drv.cuMemUnmap(reg.va_base, mapped_bytes), "kv swap_out empty unmap");
                for (auto fh : reg.handles) free_fine_handles_.push_back(fh);
                reg.handles.clear();
                kv_mapped_pages_ = (kv_mapped_pages_ >= n_pages) ? (kv_mapped_pages_ - n_pages) : 0;
                reg.swapped_valid_bytes = 0;
                reg.swapped_pages = n_pages;
                reg.state = KvRegionState::SWAPPED;
                ++reg.generation;
                kv_swapped_pages_ += n_pages;
                ++kv_swap_outs_total_;
                return static_cast<int>(n_pages);
            }

            if (!reg.swap_event) {
                drv.check(drv.cuEventCreate(&reg.swap_event, CU_EVENT_DISABLE_TIMING), "kv create swap_event");
            }
            if (reg.swap_host_pinned && reg.swap_cap_bytes < copy_bytes) {
                drv.check(drv.cuEventSynchronize(reg.swap_event), "kv resize swap_host sync");
                release_host_pinned_locked(reg.swap_host_pinned, reg.swap_cap_bytes);
                reg.swap_host_pinned = nullptr;
                reg.swap_cap_bytes = 0;
            }
            if (!reg.swap_host_pinned) {
                reg.swap_host_pinned = acquire_host_pinned_locked(copy_bytes, reg.swap_cap_bytes);
            }

            // Always bridge `compute_stream` (including nullptr == legacy default stream 0!) onto
            // `swap_stream_` (`CU_STREAM_NON_BLOCKING`) so DtoH never races in-flight KV writes/reads.
            drv.check(
                drv.cuEventRecord(compute_bridge_event_, reinterpret_cast<CUstream>(compute_stream)),
                "kv swap bridge record");
            drv.check(drv.cuStreamWaitEvent(swap_stream_, compute_bridge_event_, 0),
                      "kv swap bridge wait");
            drv.check(drv.cuMemcpyDtoHAsync(reg.swap_host_pinned, reg.va_base, copy_bytes, swap_stream_),
                      "kv swap cuMemcpyDtoHAsync");
            drv.check(drv.cuEventRecord(reg.swap_event, swap_stream_), "kv swap event record");

            reg.swapped_valid_bytes = copy_bytes;
            reg.swapped_pages = n_pages;
            reg.state = KvRegionState::SWAP_OUT_QUEUED;
            ++reg.generation;
            swap_reap_queue_.push_back(SwapReapJob{region_id, reg.generation});
            cv_req_.notify_all();
            return static_cast<int>(n_pages);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] kv_region_swap_out_async error: %s\n", e.what());
            return -3;
        }
    }

    int kv_region_swap_in_async(int32_t region_id, void* target_stream) {
        std::unique_lock<std::mutex> lock(mu_);
        if (!initialized_) return -1;
        if (region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return -2;
        auto& reg = kv_regions_[region_id];
        if (!reg.active) return -2;
        reg.evictable = false;
        try {
            ensure_ctx_locked();
            if (reg.state == KvRegionState::SWAP_OUT_FINISHING) {
                cv_ready_.wait(lock, [&] {
                    return reg.state != KvRegionState::SWAP_OUT_FINISHING || stop_streamer_;
                });
                if (stop_streamer_ || !reg.active) return -2;
            }
            if (reg.state == KvRegionState::RESIDENT) {
                return 0;
            }
            if (reg.state == KvRegionState::SWAP_OUT_QUEUED) {
                cancel_queued_swap_out_locked(reg, target_stream);
                return static_cast<int>(reg.handles.size());
            }
            const size_t pages = reg.swapped_pages;
            swap_in_region_locked(region_id, reg, target_stream, &lock);
            return static_cast<int>(pages);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] kv_region_swap_in_async error: %s\n", e.what());
            return -3;
        }
    }

    int kv_region_is_swapped(int32_t region_id) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_ || region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return 0;
        const auto& reg = kv_regions_[region_id];
        if (!reg.active) return 0;
        return (reg.state != KvRegionState::RESIDENT) ? 1 : 0;
    }

    size_t kv_region_swapped_bytes(int32_t region_id) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_ || region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return 0;
        const auto& reg = kv_regions_[region_id];
        if (!reg.active || reg.state == KvRegionState::RESIDENT) return 0;
        return reg.swapped_valid_bytes;
    }

    int kv_flush_swaps() {
        std::unique_lock<std::mutex> lock(mu_);
        if (!initialized_) return -1;
        try {
            ensure_ctx_locked();
            while (drain_one_queued_swap_out_locked()) {
            }
            if (has_finishing_swap_locked()) {
                cv_ready_.wait(lock, [&] { return !has_finishing_swap_locked() || stop_streamer_; });
            }
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] kv_flush_swaps error: %s\n", e.what());
            return -2;
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
        if (!reg.active || reg.state != KvRegionState::RESIDENT) return 0;
        return reg.handles.size() * min_gran_;
    }

    int kv_region_free(int32_t region_id, void* stream) {
        std::unique_lock<std::mutex> lock(mu_);
        if (!initialized_) return -1;
        if (region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return -2;
        auto& reg = kv_regions_[region_id];
        if (!reg.active) return 0;
        reg.evictable = false;
        try {
            ensure_ctx_locked();
            auto& drv = CudaDriver::instance();
            if (reg.state == KvRegionState::SWAP_OUT_FINISHING) {
                cv_ready_.wait(lock, [&] {
                    return reg.state != KvRegionState::SWAP_OUT_FINISHING || stop_streamer_;
                });
                if (!reg.active) return 0;
            }
            // Synchronize any in-flight `swap_stream_` DMA (DtoH or HtoD) BEFORE unmapping VA or freeing host buf!
            if (reg.swap_event) {
                drv.check(drv.cuEventSynchronize(reg.swap_event), "kv_region_free sync swap_event");
            }
            if (reg.state == KvRegionState::SWAPPED) {
                if (kv_swapped_pages_ >= reg.swapped_pages) kv_swapped_pages_ -= reg.swapped_pages;
                else kv_swapped_pages_ = 0;
                if (kv_swapped_bytes_current_ >= reg.swapped_valid_bytes) {
                    kv_swapped_bytes_current_ -= reg.swapped_valid_bytes;
                } else {
                    kv_swapped_bytes_current_ = 0;
                }
            }
            if (!reg.handles.empty()) {
                drv.check(drv.cuStreamSynchronize(reinterpret_cast<CUstream>(stream)), "kv_region sync");
                const size_t n = reg.handles.size();
                drv.check(drv.cuMemUnmap(reg.va_base, n * min_gran_), "kv_region cuMemUnmap");
                for (auto h : reg.handles) free_fine_handles_.push_back(h);
                reg.handles.clear();
                kv_mapped_pages_ = (kv_mapped_pages_ >= n) ? (kv_mapped_pages_ - n) : 0;
            }
            if (reg.swap_host_pinned) {
                release_host_pinned_locked(reg.swap_host_pinned, reg.swap_cap_bytes);
                reg.swap_host_pinned = nullptr;
                reg.swap_cap_bytes = 0;
            }
            if (reg.swap_event) {
                drv.cuEventDestroy(reg.swap_event);
                reg.swap_event = nullptr;
            }
            if (reg.va_base) {
                drv.check(drv.cuMemAddressFree(reg.va_base, reg.cap_bytes), "kv_region cuMemAddressFree");
                reg.va_base = 0;
            }
            reg.cap_bytes = 0;
            reg.swapped_valid_bytes = 0;
            reg.swapped_pages = 0;
            reg.evictable = false;
            reg.evictable_valid_bytes = 0;
            reg.state = KvRegionState::RESIDENT;
            ++reg.generation;
            reg.active = false;
            kv_free_regions_.push_back(region_id);
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] kv_region_free error: %s\n", e.what());
            return -3;
        }
    }

    int32_t kv_alloc_page() {
        std::unique_lock<std::mutex> lock(mu_);
        if (!initialized_) return -1;
        try {
            ensure_ctx_locked();
            if (kv_free_slots_.empty()) return -2;

            CUmemGenericAllocationHandle fh = pop_fine_handle_locked(nullptr, &lock);
            int32_t slot = kv_free_slots_.back();
            kv_free_slots_.pop_back();

            auto& drv = CudaDriver::instance();
            CUdeviceptr addr = kv_va_ + static_cast<size_t>(slot) * min_gran_;
            try {
                drv.check(drv.cuMemMap(addr, min_gran_, 0, fh, 0), "kv cuMemMap");
                drv.check(drv.cuMemSetAccess(addr, min_gran_, &access_, 1), "kv cuMemSetAccess");
            } catch (...) {
                drv.cuMemUnmap(addr, min_gran_);
                free_fine_handles_.push_back(fh);
                kv_free_slots_.push_back(slot);
                throw;
            }
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
            drv.check(drv.cuStreamSynchronize(reinterpret_cast<CUstream>(stream)), "kv sync");
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
        std::unique_lock<std::mutex> lock(mu_);
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
                    map_and_submit_contiguous_locked(bank, run_start, l, &lock);
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
        cv_ready_.wait(lock, [&] {
            if (stop_streamer_) return true;
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
        out->transient_mapped_handles_current = transient_mapped_slots_.size();
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

        out->fine_to_super_bundles = fine_to_super_bundles_;
        out->transient_fine_bundles_current = transient_fine_bundles_current_;
        out->expert_fine_bundles_current = expert_fine_bundles_current_;

        const CapacitySnapshot snap = compute_kv_capacities_locked();
        out->pool_total_bytes = snap.pool_total_bytes;
        out->pool_free_bytes = snap.pool_free_bytes;
        out->expert_unpinned_bytes = snap.expert_unpinned_bytes;
        out->kv_free_bytes = snap.kv_free_bytes;
        out->kv_dit_headroom_bytes = snap.kv_dit_headroom_bytes;

        out->kv_swapped_pages_current = kv_swapped_pages_;
        out->kv_swapped_bytes_current = kv_swapped_bytes_current_;
        out->kv_swap_outs_total = kv_swap_outs_total_;
        out->kv_swap_ins_total = kv_swap_ins_total_;
        out->kv_swap_d2h_bytes_total = kv_swap_d2h_bytes_total_;
        out->kv_swap_h2d_bytes_total = kv_swap_h2d_bytes_total_;
        out->kv_host_pinned_pool_bytes = kv_host_pinned_pool_bytes_;
    }

    int host_register(void* ptr, size_t nbytes) {
        if (!ptr || nbytes == 0) return -1;
        std::lock_guard<std::mutex> lock(mu_);
        try {
            auto& drv = CudaDriver::instance();
            drv.check(drv.cuInit(0), "cuInit");
            bool retained_temp = false;
            if (initialized_ && primary_ctx_) {
                ensure_ctx_locked();
            } else {
                if (!standalone_host_ctx_) {
                    CUdevice dev0 = 0;
                    drv.check(drv.cuDeviceGet(&dev0, 0), "cuDeviceGet(0)");
                    drv.check(drv.cuDevicePrimaryCtxRetain(&standalone_host_ctx_, dev0),
                              "cuDevicePrimaryCtxRetain(host)");
                    standalone_host_dev_ = dev0;
                    retained_temp = true;
                }
                drv.check(drv.cuCtxSetCurrent(standalone_host_ctx_), "cuCtxSetCurrent(host)");
            }
            try {
                drv.check(drv.cuMemHostRegister(ptr, nbytes, CU_MEMHOSTREGISTER_PORTABLE), "cuMemHostRegister");
            } catch (...) {
                if (retained_temp && standalone_host_pins_ == 0 && standalone_host_ctx_) {
                    drv.cuCtxSetCurrent(nullptr);
                    drv.cuDevicePrimaryCtxRelease(standalone_host_dev_);
                    standalone_host_ctx_ = nullptr;
                }
                throw;
            }
            if (standalone_host_ctx_ && !initialized_) {
                ++standalone_host_pins_;
            }
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] host_register error: %s\n", e.what());
            return -1;
        }
    }

    int host_unregister(void* ptr) {
        if (!ptr) return -1;
        std::lock_guard<std::mutex> lock(mu_);
        try {
            auto& drv = CudaDriver::instance();
            if (initialized_ && primary_ctx_) {
                ensure_ctx_locked();
            } else if (standalone_host_ctx_) {
                drv.check(drv.cuCtxSetCurrent(standalone_host_ctx_), "cuCtxSetCurrent(host)");
            }
            drv.check(drv.cuMemHostUnregister(ptr), "cuMemHostUnregister");
            if (standalone_host_pins_ > 0) {
                --standalone_host_pins_;
                if (standalone_host_pins_ == 0 && standalone_host_ctx_) {
                    if (!initialized_) {
                        drv.cuCtxSetCurrent(nullptr);
                    }
                    drv.cuDevicePrimaryCtxRelease(standalone_host_dev_);
                    standalone_host_ctx_ = nullptr;
                }
            }
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] host_unregister error: %s\n", e.what());
            return -1;
        }
    }

private:
    void ensure_ctx_locked() {
        auto& drv = CudaDriver::instance();
        CUcontext cur = nullptr;
        drv.check(drv.cuCtxGetCurrent(&cur), "cuCtxGetCurrent");
        if (cur != primary_ctx_ && primary_ctx_) {
            drv.check(drv.cuCtxSetCurrent(primary_ctx_), "cuCtxSetCurrent");
        }
    }

    void cleanup_failed_init_locked() noexcept {
        try {
            auto& drv = CudaDriver::instance();
            for (auto h : free_super_handles_) {
                if (h) drv.cuMemRelease(h);
            }
            free_super_handles_.clear();
            for (auto h : free_fine_handles_) {
                if (h) drv.cuMemRelease(h);
            }
            free_fine_handles_.clear();
            kv_slot_handles_.clear();
            kv_free_slots_.clear();
            if (kv_va_) {
                drv.cuMemAddressFree(kv_va_, kv_va_slots_ * min_gran_);
                kv_va_ = 0;
            }
            kv_va_slots_ = 0;
            if (transient_va_) {
                drv.cuMemAddressFree(transient_va_, transient_va_handles_ * super_bytes_);
                transient_va_ = 0;
            }
            transient_va_handles_ = 0;
            if (compute_bridge_event_) {
                drv.cuEventDestroy(compute_bridge_event_);
                compute_bridge_event_ = nullptr;
            }
            if (swap_stream_) {
                drv.cuStreamDestroy(swap_stream_);
                swap_stream_ = nullptr;
            }
            if (h2d_stream_) {
                drv.cuStreamDestroy(h2d_stream_);
                h2d_stream_ = nullptr;
            }
            if (primary_ctx_) {
                drv.cuDevicePrimaryCtxRelease(cu_dev_);
                primary_ctx_ = nullptr;
            }
        } catch (...) {
        }
        reset_counters_locked();
        device_ = -1;
        stop_streamer_ = false;
        initialized_ = false;
    }

    void reset_counters_locked() {
        super_pool_total_ = 0;
        fine_pool_total_ = 0;
        base_fine_pages_ = 0;
        kv_mapped_pages_ = 0;
        kv_mapped_pages_peak_ = 0;
        super_to_fine_carves_ = 0;
        fine_to_super_bundles_ = 0;
        transient_fine_bundles_current_ = 0;
        expert_fine_bundles_current_ = 0;
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
        kv_swapped_pages_ = 0;
        kv_swapped_bytes_current_ = 0;
        kv_swap_outs_total_ = 0;
        kv_swap_ins_total_ = 0;
        kv_swap_d2h_bytes_total_ = 0;
        kv_swap_h2d_bytes_total_ = 0;
        kv_host_pinned_pool_bytes_ = 0;
        free_host_pinned_bytes_ = 0;
        evictable_clock_ = 0;
        last_compute_stream_ = nullptr;
        scope_active_ = false;
        streamer_error_ = false;
    }

    CapacitySnapshot compute_kv_capacities_locked() {
        CapacitySnapshot s;
        if (!initialized_) return s;
        s.pool_total_bytes = super_pool_total_ * super_bytes_ + fine_pool_total_ * min_gran_;
        s.pool_free_bytes = free_super_handles_.size() * super_bytes_ + free_fine_handles_.size() * min_gran_;

        uint64_t inflight_swap_bytes = 0;
        for (const auto& reg : kv_regions_) {
            if (reg.active &&
                (reg.state == KvRegionState::SWAP_OUT_QUEUED || reg.state == KvRegionState::SWAP_OUT_FINISHING)) {
                inflight_swap_bytes += reg.handles.size() * min_gran_;
            }
        }

        uint64_t registered_expert_full_bytes = 0;
        for (const auto& bank : experts_) {
            registered_expert_full_bytes += bank.va_bytes;
            if (!bank.pinned) {
                for (const auto& lyr : bank.layers) {
                    if (lyr.state == LayerState::RESIDENT && !lyr.slots.empty()) {
                        s.expert_unpinned_bytes += bank.layer_stride_bytes;
                    }
                }
            }
        }

        s.kv_free_bytes = s.pool_free_bytes + inflight_swap_bytes + s.expert_unpinned_bytes;

        const uint64_t raw_kv_mapped = kv_mapped_pages_ * min_gran_;
        const uint64_t active_kv_bytes =
            (raw_kv_mapped > inflight_swap_bytes) ? (raw_kv_mapped - inflight_swap_bytes) : 0;
        const uint64_t transient_bytes = transient_mapped_slots_.size() * super_bytes_;
        const uint64_t committed_for_dit = active_kv_bytes + transient_bytes + registered_expert_full_bytes;
        s.kv_dit_headroom_bytes =
            (s.pool_total_bytes > committed_for_dit) ? (s.pool_total_bytes - committed_for_dit) : 0;
        return s;
    }

    void trim_idle_host_pinned_locked(size_t max_idle_bytes) {
        auto& drv = CudaDriver::instance();
        while (free_host_pinned_bytes_ > max_idle_bytes && !free_host_blocks_.empty()) {
            auto it = std::prev(free_host_blocks_.end());
            const size_t blk_cap = it->first;
            if (it->second.empty()) {
                free_host_blocks_.erase(it);
                continue;
            }
            void* p = it->second.back();
            it->second.pop_back();
            if (it->second.empty()) {
                free_host_blocks_.erase(it);
            }
            if (p) {
                drv.cuMemFreeHost(p);
            }
            free_host_pinned_bytes_ = (free_host_pinned_bytes_ >= blk_cap) ? (free_host_pinned_bytes_ - blk_cap) : 0;
            kv_host_pinned_pool_bytes_ =
                (kv_host_pinned_pool_bytes_ >= blk_cap) ? (kv_host_pinned_pool_bytes_ - blk_cap) : 0;
        }
    }

    void* acquire_host_pinned_locked(size_t min_bytes, size_t& out_cap) {
        const size_t cap = ((std::max<size_t>(min_bytes, 1) + min_gran_ - 1) / min_gran_) * min_gran_;
        auto it = free_host_blocks_.lower_bound(cap);
        if (it != free_host_blocks_.end() && !it->second.empty() && it->first <= cap * 2) {
            void* p = it->second.back();
            it->second.pop_back();
            out_cap = it->first;
            free_host_pinned_bytes_ =
                (free_host_pinned_bytes_ >= out_cap) ? (free_host_pinned_bytes_ - out_cap) : 0;
            if (it->second.empty()) free_host_blocks_.erase(it);
            return p;
        }
        if (free_host_pinned_bytes_ + cap > kMaxIdleHostPinnedBytes) {
            const size_t budget = (kMaxIdleHostPinnedBytes > cap) ? (kMaxIdleHostPinnedBytes - cap) : 0;
            trim_idle_host_pinned_locked(budget);
        }
        void* p = nullptr;
        CudaDriver::instance().check(
            CudaDriver::instance().cuMemHostAlloc(&p, cap, CU_MEMHOSTALLOC_PORTABLE),
            "kv swap cuMemHostAlloc");
        kv_host_pinned_pool_bytes_ += cap;
        out_cap = cap;
        return p;
    }

    void release_host_pinned_locked(void* p, size_t cap) {
        if (p && cap > 0) {
            free_host_blocks_[cap].push_back(p);
            free_host_pinned_bytes_ += cap;
            trim_idle_host_pinned_locked(kMaxIdleHostPinnedBytes);
        }
    }

    size_t total_evictable_thinking_pages_locked(int32_t exclude_region_id = -1) const {
        size_t total = 0;
        for (size_t idx = 0; idx < kv_regions_.size(); ++idx) {
            if (static_cast<int32_t>(idx) == exclude_region_id) continue;
            const auto& reg = kv_regions_[idx];
            if (reg.active && reg.evictable && reg.state == KvRegionState::RESIDENT) {
                total += reg.handles.size();
            }
        }
        return total;
    }

    bool evict_one_thinking_kv_region_locked(void* caller_stream, int32_t exclude_region_id = -1) {
        int32_t victim_id = -1;
        uint64_t best_seq = UINT64_MAX;
        for (size_t idx = 0; idx < kv_regions_.size(); ++idx) {
            if (static_cast<int32_t>(idx) == exclude_region_id) continue;
            const auto& reg = kv_regions_[idx];
            if (reg.active && reg.evictable && reg.state == KvRegionState::RESIDENT && !reg.handles.empty()) {
                if (reg.evictable_seq < best_seq) {
                    best_seq = reg.evictable_seq;
                    victim_id = static_cast<int32_t>(idx);
                }
            }
        }
        if (victim_id < 0) return false;

        auto& drv = CudaDriver::instance();
        auto& reg = kv_regions_[victim_id];
        const size_t n_pages = reg.handles.size();
        const size_t mapped_bytes = n_pages * min_gran_;
        const size_t copy_bytes = std::min(reg.evictable_valid_bytes, mapped_bytes);
        CUstream reg_stream = reinterpret_cast<CUstream>(reg.evictable_stream_u64);
        CUstream cur_stream = reinterpret_cast<CUstream>(caller_stream);

        if (copy_bytes > 0) {
            if (!reg.swap_event) {
                drv.check(drv.cuEventCreate(&reg.swap_event, CU_EVENT_DISABLE_TIMING),
                          "kv demand_swap create event");
            }
            if (reg.swap_host_pinned && reg.swap_cap_bytes < copy_bytes) {
                drv.check(drv.cuEventSynchronize(reg.swap_event), "kv demand_swap resize sync");
                release_host_pinned_locked(reg.swap_host_pinned, reg.swap_cap_bytes);
                reg.swap_host_pinned = nullptr;
                reg.swap_cap_bytes = 0;
            }
            if (!reg.swap_host_pinned) {
                reg.swap_host_pinned = acquire_host_pinned_locked(copy_bytes, reg.swap_cap_bytes);
            }

            drv.check(drv.cuEventRecord(compute_bridge_event_, reg_stream), "kv demand_swap bridge reg");
            drv.check(drv.cuStreamWaitEvent(swap_stream_, compute_bridge_event_, 0),
                      "kv demand_swap wait reg");
            if (cur_stream != reg_stream) {
                drv.check(drv.cuEventRecord(compute_bridge_event_, cur_stream), "kv demand_swap bridge cur");
                drv.check(drv.cuStreamWaitEvent(swap_stream_, compute_bridge_event_, 0),
                          "kv demand_swap wait cur");
            }
            drv.check(drv.cuMemcpyDtoHAsync(reg.swap_host_pinned, reg.va_base, copy_bytes, swap_stream_),
                      "kv demand_swap cuMemcpyDtoHAsync");
            drv.check(drv.cuEventRecord(reg.swap_event, swap_stream_), "kv demand_swap record");
            drv.check(drv.cuEventSynchronize(reg.swap_event), "kv demand_swap sync");
        } else {
            drv.check(drv.cuStreamSynchronize(reg_stream), "kv demand_swap empty reg sync");
            if (cur_stream != reg_stream) {
                drv.check(drv.cuStreamSynchronize(cur_stream), "kv demand_swap empty cur sync");
            }
            if (reg.swap_event) {
                drv.check(drv.cuEventSynchronize(reg.swap_event), "kv demand_swap empty event sync");
            }
        }

        drv.check(drv.cuMemUnmap(reg.va_base, mapped_bytes), "kv demand_swap unmap");
        for (auto fh : reg.handles) free_fine_handles_.push_back(fh);
        reg.handles.clear();
        kv_mapped_pages_ = (kv_mapped_pages_ >= n_pages) ? (kv_mapped_pages_ - n_pages) : 0;
        reg.swapped_valid_bytes = copy_bytes;
        reg.swapped_pages = n_pages;
        reg.evictable = false;
        reg.state = KvRegionState::SWAPPED;
        ++reg.generation;
        kv_swapped_pages_ += n_pages;
        kv_swapped_bytes_current_ += copy_bytes;
        ++kv_swap_outs_total_;
        kv_swap_d2h_bytes_total_ += copy_bytes;
        cv_ready_.notify_all();
        return true;
    }

    bool has_finishing_swap_locked() const {
        for (const auto& reg : kv_regions_) {
            if (reg.active && reg.state == KvRegionState::SWAP_OUT_FINISHING) return true;
        }
        return false;
    }

    bool drain_one_queued_swap_out_locked() {
        auto& drv = CudaDriver::instance();
        for (auto& reg : kv_regions_) {
            if (!reg.active || reg.state != KvRegionState::SWAP_OUT_QUEUED) continue;
            if (reg.swap_event) {
                drv.check(drv.cuEventSynchronize(reg.swap_event), "drain swap_out sync");
            }
            const size_t n = reg.handles.size();
            if (n > 0) {
                drv.check(drv.cuMemUnmap(reg.va_base, n * min_gran_), "drain swap_out unmap");
                for (auto fh : reg.handles) free_fine_handles_.push_back(fh);
                reg.handles.clear();
                kv_mapped_pages_ = (kv_mapped_pages_ >= n) ? (kv_mapped_pages_ - n) : 0;
            }
            reg.state = KvRegionState::SWAPPED;
            ++reg.generation;
            kv_swapped_pages_ += reg.swapped_pages;
            kv_swapped_bytes_current_ += reg.swapped_valid_bytes;
            ++kv_swap_outs_total_;
            kv_swap_d2h_bytes_total_ += reg.swapped_valid_bytes;
            cv_ready_.notify_all();
            return true;
        }
        return false;
    }

    void cancel_queued_swap_out_locked(KvRegion& reg, void* target_stream) {
        auto& drv = CudaDriver::instance();
        if (reg.swap_event) {
            if (target_stream) {
                drv.check(
                    drv.cuStreamWaitEvent(reinterpret_cast<CUstream>(target_stream), reg.swap_event, 0),
                    "cancel swap_out stream wait");
            } else {
                drv.check(drv.cuEventSynchronize(reg.swap_event), "cancel swap_out sync");
            }
        }
        reg.state = KvRegionState::RESIDENT;
        ++reg.generation;
    }

    void swap_in_region_locked(
        int32_t region_id,
        KvRegion& reg,
        void* target_stream,
        std::unique_lock<std::mutex>* opt_lock) {
        auto& drv = CudaDriver::instance();
        const size_t pages = reg.swapped_pages;
        const size_t valid_bytes = reg.swapped_valid_bytes;
        if (pages == 0) {
            reg.state = KvRegionState::RESIDENT;
            ++reg.generation;
            return;
        }

        std::vector<CUmemGenericAllocationHandle> staged;
        staged.reserve(pages);
        size_t mapped_now = 0;
        try {
            for (size_t i = 0; i < pages; ++i) {
                staged.push_back(pop_fine_handle_locked(target_stream, opt_lock, region_id));
            }
            if (!reg.active) {
                for (auto fh : staged) free_fine_handles_.push_back(fh);
                return;
            }
            for (size_t i = 0; i < pages; ++i) {
                drv.check(drv.cuMemMap(reg.va_base + i * min_gran_, min_gran_, 0, staged[i], 0),
                          "kv swap_in cuMemMap");
                ++mapped_now;
            }
            drv.check(drv.cuMemSetAccess(reg.va_base, pages * min_gran_, &access_, 1),
                      "kv swap_in cuMemSetAccess");

            if (valid_bytes > 0 && reg.swap_host_pinned) {
                if (!reg.swap_event) {
                    drv.check(drv.cuEventCreate(&reg.swap_event, CU_EVENT_DISABLE_TIMING),
                              "kv swap_in create event");
                }
                drv.check(drv.cuMemcpyHtoDAsync(reg.va_base, reg.swap_host_pinned, valid_bytes, swap_stream_),
                          "kv swap_in cuMemcpyHtoDAsync");
                drv.check(drv.cuEventRecord(reg.swap_event, swap_stream_), "kv swap_in record");
                if (target_stream) {
                    drv.check(
                        drv.cuStreamWaitEvent(reinterpret_cast<CUstream>(target_stream), reg.swap_event, 0),
                        "kv swap_in stream wait");
                } else {
                    drv.check(drv.cuStreamSynchronize(swap_stream_), "kv swap_in sync");
                }
            }

            reg.handles = std::move(staged);
            reg.state = KvRegionState::RESIDENT;
            ++reg.generation;
            kv_mapped_pages_ += pages;
            kv_mapped_pages_peak_ = std::max(kv_mapped_pages_peak_, kv_mapped_pages_);
            kv_swapped_pages_ = (kv_swapped_pages_ >= pages) ? (kv_swapped_pages_ - pages) : 0;
            kv_swapped_bytes_current_ =
                (kv_swapped_bytes_current_ >= valid_bytes) ? (kv_swapped_bytes_current_ - valid_bytes) : 0;
            ++kv_swap_ins_total_;
            kv_swap_h2d_bytes_total_ += valid_bytes;
        } catch (...) {
            if (mapped_now > 0) {
                drv.cuMemUnmap(reg.va_base, mapped_now * min_gran_);
            }
            for (auto fh : staged) free_fine_handles_.push_back(fh);
            throw;
        }
    }

    void release_super_slot_to_pool_locked(SuperSlot& slot) {
        if (slot.is_fine_bundle()) {
            for (auto fh : slot.fine_hs) {
                free_fine_handles_.push_back(fh);
            }
            slot.fine_hs.clear();
        } else if (slot.super_h) {
            free_super_handles_.push_back(slot.super_h);
            slot.super_h = 0;
        }
    }

    void map_super_slot_locked(CUdeviceptr slot_va, const SuperSlot& slot) {
        auto& drv = CudaDriver::instance();
        if (!slot.is_fine_bundle()) {
            drv.check(drv.cuMemMap(slot_va, super_bytes_, 0, slot.super_h, 0), "cuMemMap(super)");
        } else {
            size_t mapped_sub = 0;
            try {
                for (size_t j = 0; j < slot.fine_hs.size(); ++j) {
                    drv.check(drv.cuMemMap(slot_va + j * min_gran_, min_gran_, 0, slot.fine_hs[j], 0),
                              "cuMemMap(fine_bundle)");
                    ++mapped_sub;
                }
            } catch (...) {
                if (mapped_sub > 0) {
                    drv.cuMemUnmap(slot_va, mapped_sub * min_gran_);
                }
                throw;
            }
        }
    }

    bool evict_layer_locked(
        ExpertBank& bank,
        int layer_idx,
        void* stream,
        bool is_auto,
        bool for_kv,
        bool already_synced = false) {
        auto& lyr = bank.layers[layer_idx];
        if (lyr.state != LayerState::RESIDENT || lyr.slots.empty()) return false;
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

        const size_t n = lyr.slots.size();
        size_t bundles_freed = 0;
        for (auto& s : lyr.slots) {
            if (s.is_fine_bundle()) ++bundles_freed;
            release_super_slot_to_pool_locked(s);
        }
        lyr.slots.clear();
        lyr.state = LayerState::EVICTED;
        lyr.needs_stream_wait = false;
        if (expert_mapped_handles_ >= n) expert_mapped_handles_ -= n;
        if (expert_fine_bundles_current_ >= bundles_freed) expert_fine_bundles_current_ -= bundles_freed;
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

    SuperSlot pop_fine_bundle_locked() {
        SuperSlot s;
        s.fine_hs.reserve(sub_pages_);
        for (size_t j = 0; j < sub_pages_; ++j) {
            s.fine_hs.push_back(free_fine_handles_.back());
            free_fine_handles_.pop_back();
        }
        ++fine_to_super_bundles_;
        return s;
    }

    SuperSlot acquire_super_slot_locked(
        void* stream,
        int exclude_expert_id = -1,
        std::unique_lock<std::mutex>* opt_lock = nullptr) {
        while (true) {
            if (!free_super_handles_.empty()) {
                CUmemGenericAllocationHandle h = free_super_handles_.back();
                free_super_handles_.pop_back();
                return SuperSlot{h, {}};
            }
            // Reclaim carved surplus fine pages (> base_fine_pages_) before evicting resident experts!
            if (free_fine_handles_.size() >= base_fine_pages_ + sub_pages_) {
                return pop_fine_bundle_locked();
            }
            if (drain_one_queued_swap_out_locked()) {
                continue;
            }
            if (opt_lock && has_finishing_swap_locked()) {
                cv_ready_.wait(*opt_lock, [&] { return !has_finishing_swap_locked() || stop_streamer_; });
                if (stop_streamer_) {
                    throw std::runtime_error("TriPool shutting down during super-slot acquisition");
                }
                continue;
            }
            if (evict_one_tail_expert_layer_locked(stream, /*for_kv=*/false, exclude_expert_id)) {
                continue;
            }
            // Once unpinned experts cannot supply more super-handles, bundle any remaining 32x2MiB pages!
            if (free_fine_handles_.size() >= sub_pages_) {
                return pop_fine_bundle_locked();
            }
            // Reclaim LRU thinking-period KV regions on demand (e.g. during 1024^2 VAE spikes via tripool_alloc)!
            if (allow_overflow_ ||
                (free_fine_handles_.size() + total_evictable_thinking_pages_locked() >= sub_pages_)) {
                if (evict_one_thinking_kv_region_locked(stream, /*exclude_region_id=*/-1)) {
                    continue;
                }
            }
            break;
        }

        if (!allow_overflow_) {
            throw std::runtime_error("TriPool super-handle pool exhausted (allow_overflow=0)");
        }
        auto& drv = CudaDriver::instance();
        if (!free_fine_handles_.empty()) {
            // Top up partial free fine pages to sub_pages_ instead of wasting a full 64 MiB allocation!
            const size_t deficit = sub_pages_ - free_fine_handles_.size();
            for (size_t j = 0; j < deficit; ++j) {
                CUmemGenericAllocationHandle fh = 0;
                drv.check(drv.cuMemCreate(&fh, min_gran_, &prop_, 0), "overflow cuMemCreate(fine_topup)");
                free_fine_handles_.push_back(fh);
                ++fine_pool_total_;
                ++overflow_creates_;
            }
            SuperSlot s;
            s.fine_hs.reserve(sub_pages_);
            for (size_t j = 0; j < sub_pages_; ++j) {
                s.fine_hs.push_back(free_fine_handles_.back());
                free_fine_handles_.pop_back();
            }
            ++fine_to_super_bundles_;
            return s;
        }
        CUmemGenericAllocationHandle h = 0;
        drv.check(drv.cuMemCreate(&h, super_bytes_, &prop_, 0), "overflow cuMemCreate(super)");
        ++super_pool_total_;
        ++overflow_creates_;
        return SuperSlot{h, {}};
    }

    // Batch-map a contiguous range [start_layer, end_layer) (supporting mixed 64 MiB super-handles
    // and 32 x 2 MiB fine bundles) with ONE single cuMemSetAccess, then submit per-layer H2D DMA.
    void map_and_submit_contiguous_locked(
        ExpertBank& bank,
        int start_layer,
        int end_layer,
        std::unique_lock<std::mutex>* opt_lock = nullptr) {
        if (start_layer >= end_layer) return;
        auto& drv = CudaDriver::instance();
        const int count = end_layer - start_layer;
        const size_t total_handles = static_cast<size_t>(count) * bank.handles_per_layer;

        std::vector<SuperSlot> all_slots;
        all_slots.reserve(total_handles);
        size_t mapped_slots = 0;
        const CUdeviceptr batch_va =
            bank.va_base + static_cast<size_t>(start_layer) * bank.layer_stride_bytes;
        const size_t batch_bytes = static_cast<size_t>(count) * bank.layer_stride_bytes;

        try {
            for (size_t i = 0; i < total_handles; ++i) {
                all_slots.push_back(acquire_super_slot_locked(nullptr, bank.id, opt_lock));
            }

            auto tm0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < total_handles; ++i) {
                map_super_slot_locked(batch_va + i * super_bytes_, all_slots[i]);
                ++mapped_slots;
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

                lyr.slots.clear();
                lyr.slots.reserve(bank.handles_per_layer);
                size_t layer_bundles = 0;
                for (size_t k = 0; k < bank.handles_per_layer; ++k) {
                    auto& sl = all_slots[static_cast<size_t>(idx) * bank.handles_per_layer + k];
                    if (sl.is_fine_bundle()) ++layer_bundles;
                    lyr.slots.push_back(std::move(sl));
                }
                lyr.state = LayerState::RESIDENT;
                lyr.needs_stream_wait = true;
                expert_mapped_handles_ += bank.handles_per_layer;
                expert_fine_bundles_current_ += layer_bundles;
                ++expert_resident_layers_;
                ++expert_prefetch_layers_completed_;
                expert_h2d_bytes_total_ += copy_bytes;
            }
            auto th1 = std::chrono::steady_clock::now();
            expert_h2d_submit_us_ += std::chrono::duration<double, std::micro>(th1 - th0).count();
        } catch (...) {
            if (mapped_slots > 0) {
                drv.cuMemUnmap(batch_va, mapped_slots * super_bytes_);
            }
            for (auto& s : all_slots) {
                release_super_slot_to_pool_locked(s);
            }
            throw;
        }
    }

    void streamer_loop() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stop_streamer_) {
                cv_ready_.notify_all();
                return;
            }
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
            cv_req_.wait(lock, [&] {
                return stop_streamer_ || !prefetch_queue_.empty() || !swap_reap_queue_.empty();
            });
            if (stop_streamer_) return;

            // Priority 1: Expert Step-1 H2D Prefetch
            if (!prefetch_queue_.empty()) {
                PrefetchJob first = prefetch_queue_.front();
                prefetch_queue_.pop_front();
                if (first.expert_id < 0 || static_cast<size_t>(first.expert_id) >= experts_.size()) continue;
                auto& bank = experts_[first.expert_id];
                if (first.layer_idx < 0 || first.layer_idx >= bank.num_layers) continue;
                if (bank.layers[first.layer_idx].state != LayerState::QUEUED) continue;

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
                    map_and_submit_contiguous_locked(bank, start_l, end_l, &lock);
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
                continue;
            }

            // Priority 2: Async KV DtoH Swap-Out Completion & Physical Page Unmap
            if (!swap_reap_queue_.empty()) {
                SwapReapJob job = swap_reap_queue_.front();
                swap_reap_queue_.pop_front();
                if (job.region_id < 0 || static_cast<size_t>(job.region_id) >= kv_regions_.size()) continue;
                auto& reg = kv_regions_[job.region_id];
                if (!reg.active || reg.generation != job.generation || reg.state != KvRegionState::SWAP_OUT_QUEUED) {
                    continue;
                }
                CUevent ev = reg.swap_event;
                reg.state = KvRegionState::SWAP_OUT_FINISHING;
                const uint64_t fin_gen = reg.generation;
                lock.unlock();

                CUresult sync_rc = (ev && !atexit_teardown_.load(std::memory_order_acquire))
                                       ? CudaDriver::instance().cuEventSynchronize(ev)
                                       : CUDA_SUCCESS;

                lock.lock();
                if (stop_streamer_) {
                    lock.unlock();
                    cv_ready_.notify_all();
                    return;
                }
                auto& reg_after = kv_regions_[job.region_id];
                if (reg_after.active && reg_after.generation == fin_gen &&
                    reg_after.state == KvRegionState::SWAP_OUT_FINISHING) {
                    if (sync_rc == CUDA_SUCCESS) {
                        try {
                            ensure_ctx_locked();
                            const size_t n = reg_after.handles.size();
                            if (n > 0) {
                                CudaDriver::instance().check(
                                    CudaDriver::instance().cuMemUnmap(reg_after.va_base, n * min_gran_),
                                    "async swap_out unmap");
                                for (auto fh : reg_after.handles) free_fine_handles_.push_back(fh);
                                reg_after.handles.clear();
                                kv_mapped_pages_ = (kv_mapped_pages_ >= n) ? (kv_mapped_pages_ - n) : 0;
                            }
                            reg_after.state = KvRegionState::SWAPPED;
                            ++reg_after.generation;
                            kv_swapped_pages_ += reg_after.swapped_pages;
                            kv_swapped_bytes_current_ += reg_after.swapped_valid_bytes;
                            ++kv_swap_outs_total_;
                            kv_swap_d2h_bytes_total_ += reg_after.swapped_valid_bytes;
                        } catch (const std::exception& e) {
                            std::fprintf(stderr, "[tripool_vmm] async swap_out unmap error: %s\n", e.what());
                            reg_after.state = KvRegionState::RESIDENT;
                            ++reg_after.generation;
                        }
                    } else {
                        reg_after.state = KvRegionState::RESIDENT;
                        ++reg_after.generation;
                    }
                }
                lock.unlock();
                cv_ready_.notify_all();
            }
        }
    }

    void extend_transient_mapped_locked(
        size_t add_handles,
        void* stream,
        std::unique_lock<std::mutex>* opt_lock = nullptr) {
        if (add_handles == 0) return;
        auto& drv = CudaDriver::instance();
        const size_t start_idx = transient_mapped_slots_.size();
        const size_t old_end = start_idx * super_bytes_;
        const size_t add_bytes = add_handles * super_bytes_;

        std::vector<SuperSlot> acquired;
        acquired.reserve(add_handles);
        size_t mapped_now = 0;
        try {
            for (size_t i = 0; i < add_handles; ++i) {
                acquired.push_back(acquire_super_slot_locked(stream, /*exclude_expert_id=*/-1, opt_lock));
            }

            auto t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < add_handles; ++i) {
                map_super_slot_locked(transient_va_ + (start_idx + i) * super_bytes_, acquired[i]);
                ++mapped_now;
            }
            drv.check(drv.cuMemSetAccess(transient_va_ + old_end, add_bytes, &access_, 1),
                      "transient cuMemSetAccess");
            auto t1 = std::chrono::steady_clock::now();
            transient_map_us_ += std::chrono::duration<double, std::micro>(t1 - t0).count();

            for (auto& s : acquired) {
                if (s.is_fine_bundle()) ++transient_fine_bundles_current_;
                transient_mapped_slots_.push_back(std::move(s));
            }
            transient_mapped_handles_peak_ = std::max(
                transient_mapped_handles_peak_, static_cast<uint64_t>(transient_mapped_slots_.size()));

            insert_coalesce_transient_locked(old_end, add_bytes);
        } catch (...) {
            if (mapped_now > 0) {
                drv.cuMemUnmap(transient_va_ + old_end, mapped_now * super_bytes_);
            }
            for (auto& s : acquired) {
                release_super_slot_to_pool_locked(s);
            }
            throw;
        }
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
        const size_t total_bytes = transient_mapped_slots_.size() * super_bytes_;
        auto t0 = std::chrono::steady_clock::now();
        drv.check(drv.cuMemUnmap(transient_va_, total_bytes), "transient bulk cuMemUnmap");
        auto t1 = std::chrono::steady_clock::now();
        transient_unmap_us_ += std::chrono::duration<double, std::micro>(t1 - t0).count();

        for (auto& s : transient_mapped_slots_) {
            release_super_slot_to_pool_locked(s);
        }
        transient_mapped_slots_.clear();
        transient_fine_bundles_current_ = 0;
        transient_free_blocks_.clear();
        ++transient_scope_flushes_;
    }

    std::mutex mu_;
    std::condition_variable cv_req_;
    std::condition_variable cv_ready_;
    std::thread streamer_thread_;
    std::atomic<bool> atexit_teardown_{false};
    bool stop_streamer_ = false;
    bool streamer_error_ = false;

    bool initialized_ = false;
    bool scope_active_ = false;
    bool allow_overflow_ = false;
    int device_ = -1;
    CUdevice cu_dev_ = 0;
    CUcontext primary_ctx_ = nullptr;
    CUdevice standalone_host_dev_ = 0;
    CUcontext standalone_host_ctx_ = nullptr;
    size_t standalone_host_pins_ = 0;
    CUstream h2d_stream_ = nullptr;
    CUstream swap_stream_ = nullptr;
    CUevent compute_bridge_event_ = nullptr;
    void* last_compute_stream_ = nullptr;
    CUmemAllocationProp prop_ = {};
    CUmemAccessDesc access_ = {};
    size_t min_gran_ = 2ULL << 20;
    size_t super_bytes_ = 64ULL << 20;
    size_t sub_pages_ = 32;

    // Physical page pools
    std::vector<CUmemGenericAllocationHandle> free_super_handles_;
    std::vector<CUmemGenericAllocationHandle> free_fine_handles_;
    uint64_t super_pool_total_ = 0;
    uint64_t fine_pool_total_ = 0;
    size_t base_fine_pages_ = 0;
    uint64_t fine_to_super_bundles_ = 0;
    uint64_t transient_fine_bundles_current_ = 0;
    uint64_t expert_fine_bundles_current_ = 0;

    // Arena 1: Persistent KV & Host Pinned DRAM Swap Pool
    CUdeviceptr kv_va_ = 0;
    size_t kv_va_slots_ = 0;
    std::vector<CUmemGenericAllocationHandle> kv_slot_handles_;
    std::vector<int32_t> kv_free_slots_;
    std::deque<KvRegion> kv_regions_;
    std::vector<int32_t> kv_free_regions_;
    std::deque<SwapReapJob> swap_reap_queue_;
    std::map<size_t, std::vector<void*>> free_host_blocks_;
    uint64_t free_host_pinned_bytes_ = 0;
    uint64_t evictable_clock_ = 0;
    uint64_t kv_mapped_pages_ = 0;
    uint64_t kv_mapped_pages_peak_ = 0;
    uint64_t super_to_fine_carves_ = 0;
    uint64_t kv_swapped_pages_ = 0;
    uint64_t kv_swapped_bytes_current_ = 0;
    uint64_t kv_swap_outs_total_ = 0;
    uint64_t kv_swap_ins_total_ = 0;
    uint64_t kv_swap_d2h_bytes_total_ = 0;
    uint64_t kv_swap_h2d_bytes_total_ = 0;
    uint64_t kv_host_pinned_pool_bytes_ = 0;

    // Arena 2: Transient Scratchpad
    CUdeviceptr transient_va_ = 0;
    size_t transient_va_handles_ = 0;
    std::vector<SuperSlot> transient_mapped_slots_;
    std::map<size_t, size_t> transient_free_blocks_;
    std::unordered_map<size_t, size_t> transient_live_allocs_;
    uint64_t transient_mapped_handles_peak_ = 0;
    uint64_t transient_alloc_calls_ = 0;
    uint64_t transient_free_calls_ = 0;
    uint64_t transient_scope_flushes_ = 0;
    double transient_map_us_ = 0.0;
    double transient_unmap_us_ = 0.0;

    // Arena 3: Opportunistic Expert Cache
    std::deque<ExpertBank> experts_;
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

uint64_t tripool_kv_free_bytes(int for_dit) {
    return UnifiedTriPoolVMM::instance().kv_free_bytes(for_dit);
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

int tripool_kv_region_set_evictable(
    int32_t region_id, int evictable, size_t valid_bytes, void* compute_stream) {
    return UnifiedTriPoolVMM::instance().kv_region_set_evictable(
        region_id, evictable, valid_bytes, compute_stream);
}

int tripool_kv_region_swap_out_async(int32_t region_id, size_t valid_bytes, void* compute_stream) {
    return UnifiedTriPoolVMM::instance().kv_region_swap_out_async(region_id, valid_bytes, compute_stream);
}

int tripool_kv_region_swap_in_async(int32_t region_id, void* target_stream) {
    return UnifiedTriPoolVMM::instance().kv_region_swap_in_async(region_id, target_stream);
}

int tripool_kv_region_is_swapped(int32_t region_id) {
    return UnifiedTriPoolVMM::instance().kv_region_is_swapped(region_id);
}

size_t tripool_kv_region_swapped_bytes(int32_t region_id) {
    return UnifiedTriPoolVMM::instance().kv_region_swapped_bytes(region_id);
}

int tripool_kv_flush_swaps() {
    return UnifiedTriPoolVMM::instance().kv_flush_swaps();
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
    return UnifiedTriPoolVMM::instance().host_register(ptr, nbytes);
}

int tripool_host_unregister(void* ptr) {
    return UnifiedTriPoolVMM::instance().host_unregister(ptr);
}

}  // extern "C"
