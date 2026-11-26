#pragma once

#include "morphserve/vmm/page_backend.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace morphserve::vmm {

// Typed out-of-memory error raised whenever physical pool or driver OOM prevents KV growth,
// transient scratchpad allocation, expert layer load/wait, or KV swap-in.
class VmmOutOfMemoryError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct UnifiedVmmStatsPod {
    uint64_t min_granularity_bytes;
    uint64_t super_handle_bytes;

    uint64_t super_pool_total_handles;
    uint64_t super_pool_free_handles;
    uint64_t fine_pool_total_pages;
    uint64_t fine_pool_free_pages;

    uint64_t kv_mapped_pages_current;
    uint64_t kv_mapped_pages_peak;
    uint64_t super_to_fine_carves;

    uint64_t transient_mapped_handles_current;
    uint64_t transient_mapped_handles_peak;
    uint64_t transient_live_allocs;
    uint64_t transient_alloc_calls;
    uint64_t transient_free_calls;
    uint64_t transient_scope_flushes;

    uint64_t expert_mapped_handles_current;
    uint64_t expert_resident_layers_total;
    uint64_t expert_prefetch_layers_completed;
    uint64_t expert_wait_calls;
    uint64_t expert_wait_stalls;
    uint64_t expert_manual_evictions;
    uint64_t expert_auto_evictions_for_transient;
    uint64_t expert_auto_evictions_for_kv;
    uint64_t expert_h2d_bytes_total;

    uint64_t overflow_creates;
    double transient_map_us;
    double transient_unmap_us;
    double expert_map_us;
    double expert_unmap_us;
    double expert_h2d_submit_us;

    uint64_t fine_to_super_bundles;
    uint64_t transient_fine_bundles_current;
    uint64_t expert_fine_bundles_current;
    uint64_t pool_total_bytes;
    uint64_t pool_free_bytes;
    uint64_t expert_unpinned_bytes;
    uint64_t kv_free_bytes;
    uint64_t kv_dit_headroom_bytes;

    uint64_t kv_swapped_pages_current;
    uint64_t kv_swapped_bytes_current;
    uint64_t kv_swap_outs_total;
    uint64_t kv_swap_ins_total;
    uint64_t kv_swap_d2h_bytes_total;
    uint64_t kv_swap_h2d_bytes_total;
    uint64_t kv_host_pinned_pool_bytes;
};

struct TriPoolVMMConfig {
    int device = 0;
    size_t super_pool_mib = 14336;
    size_t fine_kv_pool_mib = 256;
    size_t super_handle_mib = 64;
    size_t transient_va_gib = 16;
    size_t kv_va_gib = 16;
    bool allow_overflow = false;
    bool simulated = false;
    bool materialize_sim_va = false;
    double kv_swap_high_watermark = 0.85;
    double kv_swap_low_watermark = 0.65;
    double kv_swap_min_idle_s = 1.0;
    uint64_t max_pool_bytes = 0;  // 0 = unbounded by soft pool cap
};

// Instance-owned three-arena Virtual Memory Manager over `PageBackend` (`CudaPageAdapter` / `HostMmapPageAdapter`).
// Owns bidirectional 64MiB <-> 32x2MiB handle conversion, session-level thinking KV watermark LRU swap,
// and single authoritative `kv_free_bytes` / `kv_dit_headroom_bytes` accounting.
class TriPoolVMM : public std::enable_shared_from_this<TriPoolVMM> {
public:
    static std::shared_ptr<TriPoolVMM> create(const TriPoolVMMConfig& cfg);

    TriPoolVMM(const TriPoolVMM&) = delete;
    TriPoolVMM& operator=(const TriPoolVMM&) = delete;
    ~TriPoolVMM() noexcept;

    bool is_simulated() const noexcept { return backend_ ? backend_->is_simulated() : true; }
    int device() const noexcept { return cfg_.device; }
    const TriPoolVMMConfig& config() const noexcept { return cfg_; }

    int shutdown();
    void notify_atexit_teardown() noexcept;
    void get_stats(UnifiedVmmStatsPod* out);
    uint64_t kv_free_bytes(bool for_dit, uint64_t extra_driver_free_bytes = 0);
    uint64_t sync_kv_free_bytes(bool for_dit, uint64_t extra_driver_free_bytes = 0);

    // Arena 2: Scoped Transient Scratchpad (`CUDAPluggableAllocator` compatible)
    void scope_begin();
    int scope_end(void* stream);
    void* transient_alloc(ssize_t req_size, void* stream);
    void transient_free(void* ptr, ssize_t req_size, void* stream);

    // Arena 1: 2 MiB Paged Pool, Contiguous-VA Growable Regions & Session Watermark LRU Swap
    uint64_t kv_va_base();
    int32_t kv_alloc_page();
    int kv_free_page(int32_t slot_idx, void* stream);

    uint64_t kv_session_create();
    void kv_session_destroy(uint64_t session_id);
    void kv_session_touch(uint64_t session_id);
    void kv_session_set_pinned(uint64_t session_id, bool pinned);
    int kv_session_mark_active(uint64_t session_id, void* stream);
    size_t kv_session_mark_thinking(
        uint64_t session_id,
        const int32_t* region_ids,
        const size_t* valid_bytes,
        size_t num_regions,
        bool trigger_watermark,
        void* stream);

    size_t reclaim_thinking_kv_bytes(
        size_t target_reclaim_bytes,
        bool sync,
        uint64_t exclude_session_id,
        void* stream);

    size_t reclaim_thinking_kv_bytes_multi(
        size_t target_reclaim_bytes,
        bool sync,
        const uint64_t* exclude_session_ids,
        size_t num_excludes,
        void* stream);

    size_t maybe_evict_thinking_kv_on_high_watermark(
        bool sync,
        uint64_t exclude_session_id,
        size_t extra_needed_bytes,
        bool for_dit,
        void* stream);

    size_t maybe_evict_thinking_kv_on_high_watermark_multi(
        bool sync,
        const uint64_t* exclude_session_ids,
        size_t num_excludes,
        size_t extra_needed_bytes,
        bool for_dit,
        void* stream);

    int32_t kv_region_alloc(size_t max_va_bytes, uint64_t session_id = 0);
    uint64_t kv_region_ensure(int32_t region_id, size_t needed_bytes, void* stream);
    uint64_t kv_region_va(int32_t region_id);
    size_t kv_region_mapped_bytes(int32_t region_id);
    int kv_region_free(int32_t region_id, void* stream);

    int kv_region_set_evictable(int32_t region_id, int evictable, size_t valid_bytes, void* compute_stream);
    int kv_region_swap_out_async(int32_t region_id, size_t valid_bytes, void* compute_stream);
    int kv_region_swap_in_async(int32_t region_id, void* target_stream);
    int kv_region_is_swapped(int32_t region_id);
    size_t kv_region_swapped_bytes(int32_t region_id);
    int kv_flush_swaps();

    // Arena 3: Opportunistic Secondary Expert Cache & Background Streamer
    int expert_register(int num_layers, size_t layer_stride_mib, void* host_pinned_base, size_t host_bytes);
    uint64_t expert_va_base(int expert_id);
    uint64_t expert_layer_va(int expert_id, int layer_idx);
    size_t expert_layer_stride_bytes(int expert_id);
    int expert_set_pinned(int expert_id, int pinned);
    int expert_set_layer_used_bytes(int expert_id, int layer_idx, size_t used_bytes);
    int expert_load_layers_sync(int expert_id, int start_layer, int end_layer_exclusive);
    int expert_prefetch_async(int expert_id, int start_layer, int end_layer_exclusive);
    int expert_wait_layer(int expert_id, int layer_idx, void* compute_stream);
    int expert_sync_h2d(int expert_id);
    size_t coalesce_fine_to_super(size_t max_super = SIZE_MAX);
    int expert_evict_layer(int expert_id, int layer_idx, void* stream);
    int expert_evict_range(int expert_id, int start_layer, int end_layer_exclusive, void* stream);
    int expert_is_resident(int expert_id, int layer_idx);
    int expert_resident_count(int expert_id);
    size_t ensure_expert_headroom(
        int expert_id,
        size_t needed_bytes,
        const uint64_t* exclude_session_ids,
        size_t num_excludes,
        void* stream);

    // Active CUDA device registry for `CUDAPluggableAllocator` (`tripool_alloc` / `tripool_free`)
    static void register_active_device_instance(int device, const std::shared_ptr<TriPoolVMM>& inst);
    static void unregister_active_device_instance(int device, const TriPoolVMM* inst);
    static std::shared_ptr<TriPoolVMM> active_device_instance(int device);

private:
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

    struct SuperSlot {
        uint64_t super_h = 0;
        std::vector<uint64_t> fine_hs;
        bool is_fine_bundle() const noexcept { return !fine_hs.empty(); }
    };

    struct ExpertLayer {
        LayerState state = LayerState::EVICTED;
        bool needs_stream_wait = false;
        bool failed = false;
        bool failed_oom = false;
        std::string failed_msg;
        size_t used_bytes = 0;
        std::vector<SuperSlot> slots;
        void* ready_event = nullptr;
    };

    struct ExpertBank {
        int id = -1;
        int num_layers = 0;
        size_t layer_stride_bytes = 0;
        size_t handles_per_layer = 0;
        uint64_t va_base = 0;
        size_t va_bytes = 0;
        uint8_t* host_pinned_base = nullptr;
        size_t host_bytes = 0;
        bool pinned = false;
        bool last_async_error = false;
        bool last_async_error_oom = false;
        std::string last_async_error_msg;
        void* last_h2d_event = nullptr;
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
        uint64_t va_base = 0;
        size_t cap_bytes = 0;
        uint64_t session_id = 0;
        std::vector<uint64_t> handles;
        void* swap_host_pinned = nullptr;
        size_t swap_cap_bytes = 0;
        size_t swapped_valid_bytes = 0;
        size_t swapped_pages = 0;
        void* swap_event = nullptr;
        uint64_t generation = 0;
        KvRegionState state = KvRegionState::RESIDENT;
        bool evictable = false;
        size_t evictable_valid_bytes = 0;
        uint64_t evictable_seq = 0;
        uint64_t evictable_stream_u64 = 0;
        bool active = false;
    };

    struct KvSession {
        uint64_t session_id = 0;
        bool pinned_active = false;
        bool thinking = false;
        uint64_t last_access_ns = 0;
        std::unordered_set<int32_t> region_ids;
    };

    struct KvBudgetSnapshot {
        uint64_t pool_total_bytes = 0;
        uint64_t pool_free_bytes = 0;
        uint64_t inflight_swap_bytes = 0;
        uint64_t active_kv_bytes = 0;
        uint64_t transient_bytes = 0;
        uint64_t expert_mapped_bytes = 0;
        uint64_t expert_unpinned_bytes = 0;
        uint64_t expert_pinned_bytes = 0;
        uint64_t dit_expert_full_bytes = 0;
        uint64_t fixed_non_kv_bytes = 0;
        uint64_t effective_pool_bytes = 0;
        uint64_t kv_budget_bytes = 0;
        uint64_t high_watermark_bytes = 0;
        uint64_t low_watermark_bytes = 0;
        uint64_t kv_free_bytes = 0;
        uint64_t kv_dit_headroom_bytes = 0;
    };

    explicit TriPoolVMM(const TriPoolVMMConfig& cfg);
    void init_pools();
    void cleanup_failed_init_locked() noexcept;
    void reset_counters_locked();

    KvBudgetSnapshot compute_kv_budget_locked(bool for_dit) const;
    uint64_t clamp_by_max_pool_locked(uint64_t raw_free, const KvBudgetSnapshot& snap) const;
    size_t compute_watermark_reclaim_target_locked(bool for_dit, size_t extra_needed_bytes) const;

    uint64_t pop_fine_handle_locked(
        void* stream,
        std::unique_lock<std::mutex>* opt_lock = nullptr,
        int32_t exclude_region_id = -1);
    SuperSlot pop_fine_bundle_locked();
    SuperSlot acquire_super_slot_locked(
        void* stream,
        bool for_kv,
        int exclude_expert_id = -1,
        std::unique_lock<std::mutex>* opt_lock = nullptr);
    void release_super_slot_to_pool_locked(SuperSlot& slot);
    size_t try_coalesce_fine_to_super_locked(size_t max_super = SIZE_MAX);
    void map_super_slot_locked(uint64_t slot_va, const SuperSlot& slot, bool host_access);

    void trim_idle_host_pinned_locked(size_t max_idle_bytes);
    void* acquire_host_pinned_locked(size_t min_bytes, size_t& out_cap);
    void release_host_pinned_locked(void* p, size_t cap);

    size_t total_evictable_thinking_pages_locked(int32_t exclude_region_id = -1) const;
    bool evict_one_thinking_kv_region_locked(void* caller_stream, int32_t exclude_region_id = -1);
    bool has_finishing_swap_locked() const;
    bool drain_one_queued_swap_out_locked();
    void cancel_queued_swap_out_locked(KvRegion& reg, void* target_stream);
    int submit_region_swap_out_locked(int32_t region_id, KvRegion& reg, size_t valid_bytes, void* compute_stream);
    void swap_in_region_locked(
        int32_t region_id,
        KvRegion& reg,
        void* target_stream,
        std::unique_lock<std::mutex>* opt_lock);

    size_t reclaim_thinking_kv_locked(
        size_t target_reclaim_bytes,
        bool sync,
        const uint64_t* exclude_session_ids,
        size_t num_excludes,
        void* stream,
        std::unique_lock<std::mutex>& lock);

    bool evict_layer_locked(
        ExpertBank& bank,
        int layer_idx,
        void* stream,
        bool is_auto,
        bool for_kv,
        bool already_synced = false);
    bool evict_one_tail_expert_layer_locked(void* stream, bool for_kv, int exclude_expert_id = -1);
    void map_and_submit_contiguous_locked(
        ExpertBank& bank,
        int start_layer,
        int end_layer,
        std::unique_lock<std::mutex>* opt_lock = nullptr);
    void extend_transient_mapped_locked(
        size_t add_handles,
        void* stream,
        std::unique_lock<std::mutex>* opt_lock = nullptr);
    void insert_coalesce_transient_locked(size_t off, size_t sz);
    void bulk_unmap_transient_locked(void* stream);
    void streamer_loop();

    TriPoolVMMConfig cfg_;
    std::unique_ptr<PageBackend> backend_;

    std::mutex mu_;
    std::condition_variable cv_req_;
    std::condition_variable cv_ready_;
    std::thread streamer_thread_;
    std::atomic<bool> atexit_teardown_{false};
    bool stop_streamer_ = false;
    bool streamer_error_ = false;
    size_t active_h2d_syncs_ = 0;

    bool initialized_ = false;
    int scope_depth_ = 0;
    void* h2d_stream_ = nullptr;
    void* swap_stream_ = nullptr;
    void* compute_bridge_event_ = nullptr;
    void* last_compute_stream_ = nullptr;
    size_t min_gran_ = 2ULL << 20;
    size_t super_bytes_ = 64ULL << 20;
    size_t sub_pages_ = 32;

    std::vector<uint64_t> free_super_handles_;
    std::vector<uint64_t> free_fine_handles_;
    uint64_t super_pool_total_ = 0;
    uint64_t fine_pool_total_ = 0;
    size_t base_fine_pages_ = 0;
    uint64_t fine_to_super_bundles_ = 0;
    uint64_t transient_fine_bundles_current_ = 0;
    uint64_t expert_fine_bundles_current_ = 0;

    uint64_t kv_va_ = 0;
    size_t kv_va_slots_ = 0;
    std::vector<uint64_t> kv_slot_handles_;
    std::vector<int32_t> kv_free_slots_;
    std::deque<KvRegion> kv_regions_;
    std::vector<int32_t> kv_free_regions_;
    std::deque<SwapReapJob> swap_reap_queue_;
    std::unordered_map<uint64_t, KvSession> kv_sessions_;
    uint64_t next_session_id_ = 1;
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

    uint64_t transient_va_ = 0;
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

}  // namespace morphserve::vmm
