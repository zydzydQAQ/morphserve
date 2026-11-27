#include "morphserve/vmm/tripool_vmm.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace morphserve::vmm {

namespace {

constexpr size_t kAlign = 512;
constexpr size_t kMaxIdleHostPinnedBytes = 2ULL << 30;  // 2 GiB cap on cached idle swap blocks

uint64_t mono_now_ns() noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

struct GlobalVmmRegistry {
    std::mutex mu;
    std::unordered_map<int, std::weak_ptr<TriPoolVMM>> by_device;
};

GlobalVmmRegistry& registry() {
    static GlobalVmmRegistry reg;
    return reg;
}

}  // namespace

std::shared_ptr<TriPoolVMM> TriPoolVMM::create(const TriPoolVMMConfig& cfg) {
    if (!(0.0 < cfg.kv_swap_low_watermark &&
          cfg.kv_swap_low_watermark <= cfg.kv_swap_high_watermark &&
          cfg.kv_swap_high_watermark <= 1.0)) {
        throw std::invalid_argument("expected 0.0 < kv_swap_low_watermark <= kv_swap_high_watermark <= 1.0");
    }
    std::shared_ptr<TriPoolVMM> inst(new TriPoolVMM(cfg));
    inst->init_pools();
    if (!cfg.simulated) {
        register_active_device_instance(cfg.device, inst);
    }
    return inst;
}

TriPoolVMM::TriPoolVMM(const TriPoolVMMConfig& cfg) : cfg_(cfg) {}

TriPoolVMM::~TriPoolVMM() noexcept {
    unregister_active_device_instance(cfg_.device, this);
    if (atexit_teardown_.load(std::memory_order_acquire)) {
        try {
            {
                std::lock_guard<std::mutex> lock(mu_);
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
        return;
    }
    try {
        shutdown();
    } catch (...) {
    }
}

void TriPoolVMM::notify_atexit_teardown() noexcept {
    atexit_teardown_.store(true, std::memory_order_release);
}

void TriPoolVMM::init_pools() {
    std::lock_guard<std::mutex> lock(mu_);
    try {
        if (cfg_.simulated) {
            backend_ = std::make_unique<HostMmapPageAdapter>(cfg_.device, 2ULL << 20, cfg_.materialize_sim_va);
        } else {
            backend_ = std::make_unique<CudaPageAdapter>(cfg_.device);
        }
        min_gran_ = backend_->min_granularity_bytes();
        const size_t req_super = (cfg_.super_handle_mib > 0 ? cfg_.super_handle_mib : 64) * (1ULL << 20);
        super_bytes_ = ((req_super + min_gran_ - 1) / min_gran_) * min_gran_;
        sub_pages_ = std::max<size_t>(1, super_bytes_ / min_gran_);

        h2d_stream_ = backend_->create_stream(/*high_priority=*/true);
        swap_stream_ = backend_->create_stream(/*high_priority=*/false);
        compute_bridge_event_ = backend_->create_event();

        const size_t t_va_gib = std::max<size_t>(1, cfg_.transient_va_gib);
        const size_t t_va_bytes = ((t_va_gib * (1ULL << 30)) + super_bytes_ - 1) / super_bytes_ * super_bytes_;
        transient_va_handles_ = t_va_bytes / super_bytes_;
        transient_va_ = backend_->reserve_va(t_va_bytes, super_bytes_);

        const size_t kv_va_gib = std::max<size_t>(1, cfg_.kv_va_gib);
        const size_t kv_va_bytes = ((kv_va_gib * (1ULL << 30)) + min_gran_ - 1) / min_gran_ * min_gran_;
        kv_va_slots_ = kv_va_bytes / min_gran_;
        kv_va_ = backend_->reserve_va(kv_va_bytes, min_gran_);
        kv_slot_handles_.assign(kv_va_slots_, 0);
        kv_free_slots_.reserve(kv_va_slots_);
        for (size_t i = kv_va_slots_; i > 0; --i) {
            kv_free_slots_.push_back(static_cast<int32_t>(i - 1));
        }

        const size_t super_n =
            (cfg_.super_pool_mib > 0) ? (((cfg_.super_pool_mib * (1ULL << 20)) + super_bytes_ - 1) / super_bytes_) : 0;
        free_super_handles_.reserve(super_n);
        for (size_t i = 0; i < super_n; ++i) {
            free_super_handles_.push_back(backend_->create_phys_handle(super_bytes_));
        }
        super_pool_total_ = super_n;
        init_super_handles_ = super_n;

        const size_t fine_n = (cfg_.fine_kv_pool_mib * (1ULL << 20)) / min_gran_;
        free_fine_handles_.reserve(fine_n);
        for (size_t i = 0; i < fine_n; ++i) {
            free_fine_handles_.push_back(backend_->create_phys_handle(min_gran_));
        }
        fine_pool_total_ = fine_n;
        base_fine_pages_ = fine_n;

        atexit_teardown_.store(false, std::memory_order_relaxed);
        stop_streamer_ = false;
        streamer_error_ = false;
        streamer_thread_ = std::thread(&TriPoolVMM::streamer_loop, this);
        initialized_ = true;
    } catch (...) {
        cleanup_failed_init_locked();
        throw;
    }
}

int TriPoolVMM::shutdown() {
    unregister_active_device_instance(cfg_.device, this);
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

    std::unique_lock<std::mutex> lock(mu_);
    cv_ready_.wait(lock, [&] { return active_h2d_syncs_ == 0; });
    if (!initialized_) return 0;
    try {
        backend_->ensure_context();
        if (h2d_stream_) backend_->sync_stream(h2d_stream_);
        if (swap_stream_) backend_->sync_stream(swap_stream_);
        backend_->sync_context();

        for (auto& bank : experts_) {
            for (int l = 0; l < bank.num_layers; ++l) {
                auto& lyr = bank.layers[static_cast<size_t>(l)];
                if (!lyr.slots.empty()) {
                    backend_->unmap_range(
                        bank.va_base + static_cast<size_t>(l) * bank.layer_stride_bytes,
                        bank.layer_stride_bytes,
                        /*was_host_accessible=*/true);
                    for (auto& s : lyr.slots) release_super_slot_to_pool_locked(s);
                    lyr.slots.clear();
                }
                if (lyr.ready_event) {
                    backend_->destroy_event(lyr.ready_event);
                    lyr.ready_event = nullptr;
                }
                lyr.state = LayerState::EVICTED;
            }
            if (bank.va_base) {
                backend_->free_va(bank.va_base, bank.va_bytes);
                bank.va_base = 0;
            }
        }
        experts_.clear();
        expert_use_clock_ = 0;
    expert_mapped_handles_ = 0;
        expert_fine_bundles_current_ = 0;
        expert_resident_layers_ = 0;

        if (!transient_mapped_slots_.empty()) {
            backend_->unmap_range(
                transient_va_,
                transient_mapped_slots_.size() * super_bytes_,
                /*was_host_accessible=*/true);
            for (auto& s : transient_mapped_slots_) release_super_slot_to_pool_locked(s);
            transient_mapped_slots_.clear();
        }
        transient_fine_bundles_current_ = 0;
        transient_live_allocs_.clear();
        transient_free_blocks_.clear();
        if (transient_va_) {
            backend_->free_va(transient_va_, transient_va_handles_ * super_bytes_);
            transient_va_ = 0;
        }

        for (auto& reg : kv_regions_) {
            if (reg.active) {
                if (!reg.handles.empty()) {
                    backend_->unmap_range(reg.va_base, reg.handles.size() * min_gran_, /*was_host_accessible=*/true);
                    for (auto h : reg.handles) free_fine_handles_.push_back(h);
                    reg.handles.clear();
                }
                if (reg.swap_host_pinned) {
                    backend_->free_host_pinned(reg.swap_host_pinned);
                    reg.swap_host_pinned = nullptr;
                    reg.swap_cap_bytes = 0;
                }
                if (reg.swap_event) {
                    backend_->destroy_event(reg.swap_event);
                    reg.swap_event = nullptr;
                }
                if (reg.va_base) {
                    backend_->free_va(reg.va_base, reg.cap_bytes);
                    reg.va_base = 0;
                }
                reg.active = false;
            }
        }
        kv_regions_.clear();
        kv_free_regions_.clear();
        kv_sessions_.clear();

        for (auto& kv : free_host_blocks_) {
            for (void* p : kv.second) {
                if (p) backend_->free_host_pinned(p);
            }
        }
        free_host_blocks_.clear();

        for (size_t s = 0; s < kv_va_slots_; ++s) {
            if (kv_slot_handles_[s]) {
                backend_->unmap_range(kv_va_ + s * min_gran_, min_gran_, /*was_host_accessible=*/true);
                free_fine_handles_.push_back(kv_slot_handles_[s]);
                kv_slot_handles_[s] = 0;
            }
        }
        kv_slot_handles_.clear();
        kv_free_slots_.clear();
        if (kv_va_) {
            backend_->free_va(kv_va_, kv_va_slots_ * min_gran_);
            kv_va_ = 0;
        }

        for (auto h : free_super_handles_) backend_->release_phys_handle(h, super_bytes_);
        free_super_handles_.clear();
        for (auto h : free_fine_handles_) backend_->release_phys_handle(h, min_gran_);
        free_fine_handles_.clear();

        if (compute_bridge_event_) {
            backend_->destroy_event(compute_bridge_event_);
            compute_bridge_event_ = nullptr;
        }
        if (swap_stream_) {
            backend_->destroy_stream(swap_stream_);
            swap_stream_ = nullptr;
        }
        if (h2d_stream_) {
            backend_->destroy_stream(h2d_stream_);
            h2d_stream_ = nullptr;
        }
        backend_.reset();
        reset_counters_locked();
        initialized_ = false;
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tripool_vmm] shutdown error: %s\n", e.what());
        return -1;
    }
}

void TriPoolVMM::cleanup_failed_init_locked() noexcept {
    try {
        if (backend_) {
            for (auto h : free_super_handles_) {
                if (h) backend_->release_phys_handle(h, super_bytes_);
            }
            free_super_handles_.clear();
            for (auto h : free_fine_handles_) {
                if (h) backend_->release_phys_handle(h, min_gran_);
            }
            free_fine_handles_.clear();
            kv_slot_handles_.clear();
            kv_free_slots_.clear();
            if (kv_va_) {
                backend_->free_va(kv_va_, kv_va_slots_ * min_gran_);
                kv_va_ = 0;
            }
            if (transient_va_) {
                backend_->free_va(transient_va_, transient_va_handles_ * super_bytes_);
                transient_va_ = 0;
            }
            if (compute_bridge_event_) {
                backend_->destroy_event(compute_bridge_event_);
                compute_bridge_event_ = nullptr;
            }
            if (swap_stream_) {
                backend_->destroy_stream(swap_stream_);
                swap_stream_ = nullptr;
            }
            if (h2d_stream_) {
                backend_->destroy_stream(h2d_stream_);
                h2d_stream_ = nullptr;
            }
            backend_.reset();
        }
    } catch (...) {
    }
    reset_counters_locked();
    initialized_ = false;
}

void TriPoolVMM::reset_counters_locked() {
    super_pool_total_ = 0;
    fine_pool_total_ = 0;
    init_super_handles_ = 0;
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
    expert_use_clock_ = 0;
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
    scope_depth_ = 0;
    streamer_error_ = false;
}

TriPoolVMM::KvBudgetSnapshot TriPoolVMM::compute_kv_budget_locked(bool for_dit) const {
    KvBudgetSnapshot b;
    if (!initialized_) return b;
    b.pool_total_bytes = super_pool_total_ * super_bytes_ + fine_pool_total_ * min_gran_;
    b.pool_free_bytes = free_super_handles_.size() * super_bytes_ + free_fine_handles_.size() * min_gran_;

    for (const auto& reg : kv_regions_) {
        if (reg.active &&
            (reg.state == KvRegionState::SWAP_OUT_QUEUED || reg.state == KvRegionState::SWAP_OUT_FINISHING)) {
            b.inflight_swap_bytes += reg.handles.size() * min_gran_;
        }
    }

    const uint64_t raw_kv_mapped = kv_mapped_pages_ * min_gran_;
    b.active_kv_bytes = (raw_kv_mapped > b.inflight_swap_bytes) ? (raw_kv_mapped - b.inflight_swap_bytes) : 0;
    b.transient_bytes = transient_mapped_slots_.size() * super_bytes_;
    b.expert_mapped_bytes = expert_mapped_handles_ * super_bytes_;

    uint64_t pinned_expert_va_bytes = 0;
    for (const auto& bank : experts_) {
        if (bank.pinned) {
            pinned_expert_va_bytes += bank.va_bytes;
        } else {
            for (const auto& lyr : bank.layers) {
                if (lyr.state == LayerState::RESIDENT && !lyr.slots.empty()) {
                    b.expert_unpinned_bytes += bank.layer_stride_bytes;
                }
            }
        }
    }
    b.expert_pinned_bytes =
        (b.expert_mapped_bytes > b.expert_unpinned_bytes)
            ? (b.expert_mapped_bytes - b.expert_unpinned_bytes)
            : 0;
    b.dit_expert_full_bytes = std::max<uint64_t>(
        experts_.empty() ? uint64_t{0} : static_cast<uint64_t>(experts_[0].va_bytes),
        pinned_expert_va_bytes);

    b.fixed_non_kv_bytes = b.transient_bytes + (for_dit ? b.dit_expert_full_bytes : b.expert_pinned_bytes);
    b.effective_pool_bytes =
        (cfg_.max_pool_bytes > 0)
            ? (cfg_.allow_overflow ? cfg_.max_pool_bytes : std::min(b.pool_total_bytes, cfg_.max_pool_bytes))
            : b.pool_total_bytes;

    b.kv_budget_bytes =
        (b.effective_pool_bytes > b.fixed_non_kv_bytes) ? (b.effective_pool_bytes - b.fixed_non_kv_bytes) : 0;
    b.high_watermark_bytes =
        static_cast<uint64_t>(static_cast<double>(b.kv_budget_bytes) * cfg_.kv_swap_high_watermark);
    b.low_watermark_bytes =
        static_cast<uint64_t>(static_cast<double>(b.kv_budget_bytes) * cfg_.kv_swap_low_watermark);

    b.kv_free_bytes = b.pool_free_bytes + b.inflight_swap_bytes + b.expert_unpinned_bytes;
    const uint64_t committed_for_dit = b.active_kv_bytes + b.transient_bytes + b.dit_expert_full_bytes;
    b.kv_dit_headroom_bytes = (b.pool_total_bytes > committed_for_dit) ? (b.pool_total_bytes - committed_for_dit) : 0;
    return b;
}

uint64_t TriPoolVMM::clamp_by_max_pool_locked(
    uint64_t raw_free,
    const KvBudgetSnapshot& b) const {
    if (cfg_.max_pool_bytes == 0) {
        return raw_free;
    }
    const uint64_t committed = b.active_kv_bytes + b.fixed_non_kv_bytes;
    const uint64_t soft_free = (cfg_.max_pool_bytes > committed) ? (cfg_.max_pool_bytes - committed) : 0;
    return std::min(raw_free, soft_free);
}

size_t TriPoolVMM::compute_watermark_reclaim_target_locked(bool for_dit, size_t extra_needed_bytes) const {
    const KvBudgetSnapshot b = compute_kv_budget_locked(for_dit);
    if (b.effective_pool_bytes == 0) return 0;
    const uint64_t projected_kv = b.active_kv_bytes + static_cast<uint64_t>(extra_needed_bytes);
    if (projected_kv <= b.high_watermark_bytes && projected_kv <= b.kv_budget_bytes) {
        return 0;
    }
    if (projected_kv > b.low_watermark_bytes) {
        return static_cast<size_t>(projected_kv - b.low_watermark_bytes);
    }
    return (extra_needed_bytes > 0) ? extra_needed_bytes : static_cast<size_t>(projected_kv);
}

uint64_t TriPoolVMM::kv_free_bytes(bool for_dit, uint64_t extra_driver_free_bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!initialized_) return 0;
    const KvBudgetSnapshot b = compute_kv_budget_locked(for_dit);
    uint64_t extra = extra_driver_free_bytes;
    if (extra == 0 && cfg_.allow_overflow) {
        extra = backend_->driver_free_overflow_bytes();
    }
    const uint64_t raw = (for_dit ? b.kv_dit_headroom_bytes : b.kv_free_bytes) + extra;
    return clamp_by_max_pool_locked(raw, b);
}

uint64_t TriPoolVMM::sync_kv_free_bytes(bool for_dit, uint64_t extra_driver_free_bytes) {
    kv_flush_swaps();
    return kv_free_bytes(for_dit, extra_driver_free_bytes);
}

// ==================== Arena 2: Transient Scratchpad ====================

void TriPoolVMM::scope_begin() {
    std::lock_guard<std::mutex> lock(mu_);
    ++scope_depth_;
}

int TriPoolVMM::scope_end(void* stream) {
    std::lock_guard<std::mutex> lock(mu_);
    if (scope_depth_ <= 0) {
        scope_depth_ = 0;
        return initialized_ ? static_cast<int>(transient_live_allocs_.size()) : 0;
    }
    --scope_depth_;
    if (!initialized_) return 0;
    if (scope_depth_ == 0 && transient_live_allocs_.empty() && !transient_mapped_slots_.empty()) {
        try {
            backend_->ensure_context();
            bulk_unmap_transient_locked(stream);
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] scope_end error: %s\n", e.what());
            return -1;
        }
    }
    return static_cast<int>(transient_live_allocs_.size());
}

void* TriPoolVMM::transient_alloc(ssize_t req_size, void* stream) {
    if (req_size <= 0) return nullptr;
    std::unique_lock<std::mutex> lock(mu_);
    if (!initialized_) return nullptr;
    try {
        backend_->ensure_context();
        ++transient_alloc_calls_;
        const size_t aligned = (static_cast<size_t>(req_size) + kAlign - 1) & ~(kAlign - 1);

        for (auto it = transient_free_blocks_.begin(); it != transient_free_blocks_.end(); ++it) {
            if (it->second >= aligned) {
                const size_t off = it->first;
                const size_t rem = it->second - aligned;
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
            throw VmmOutOfMemoryError("Transient VA exhausted");
        }
        extend_transient_mapped_locked(add_handles, stream, &lock);

        auto last = std::prev(transient_free_blocks_.end());
        if (last->second < aligned) {
            throw std::runtime_error("Transient allocator invariant broken");
        }
        const size_t off = last->first;
        const size_t rem = last->second - aligned;
        transient_free_blocks_.erase(last);
        if (rem > 0) transient_free_blocks_[off + aligned] = rem;
        transient_live_allocs_[off] = aligned;
        return reinterpret_cast<void*>(transient_va_ + off);
    } catch (const VmmOutOfMemoryError&) {
        throw;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tripool_vmm] transient_alloc(%zd) error: %s\n", req_size, e.what());
        return nullptr;
    }
}

void TriPoolVMM::transient_free(void* ptr, ssize_t /*req_size*/, void* stream) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mu_);
    if (!initialized_) return;
    try {
        ++transient_free_calls_;
        const uint64_t addr = reinterpret_cast<uint64_t>(ptr);
        if (addr < transient_va_) return;
        const size_t off = static_cast<size_t>(addr - transient_va_);
        auto it = transient_live_allocs_.find(off);
        if (it == transient_live_allocs_.end()) return;
        const size_t sz = it->second;
        transient_live_allocs_.erase(it);
        insert_coalesce_transient_locked(off, sz);

        if (scope_depth_ == 0 && transient_live_allocs_.empty() && !transient_mapped_slots_.empty()) {
            backend_->ensure_context();
            bulk_unmap_transient_locked(stream);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tripool_vmm] transient_free error: %s\n", e.what());
    }
}

// ==================== Arena 1: KV Pages, Regions & Session Watermark Governance ====================

uint64_t TriPoolVMM::pop_fine_handle_locked(
    void* stream,
    std::unique_lock<std::mutex>* opt_lock,
    int32_t exclude_region_id) {
    while (free_fine_handles_.empty()) {
        if (!free_super_handles_.empty()) {
            const uint64_t super_h = free_super_handles_.back();
            free_super_handles_.pop_back();
            backend_->release_phys_handle(super_h, super_bytes_);
            --super_pool_total_;
            for (size_t i = 0; i < sub_pages_; ++i) {
                free_fine_handles_.push_back(backend_->create_phys_handle(min_gran_));
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
            continue;
        }
        if (evict_one_thinking_kv_region_locked(stream, exclude_region_id)) {
            continue;
        }
        const uint64_t cur_pool_b = super_pool_total_ * super_bytes_ + fine_pool_total_ * min_gran_;
        const bool over_cap = (cfg_.max_pool_bytes > 0) && ((cur_pool_b + min_gran_) > cfg_.max_pool_bytes);
        if (!cfg_.allow_overflow || over_cap) {
            throw VmmOutOfMemoryError(
                over_cap ? "TriPool physical pool reached max_pool_bytes cap for KV page"
                         : "TriPool physical pool exhausted for KV page (allow_overflow=0)");
        }
        free_fine_handles_.push_back(backend_->create_phys_handle(min_gran_));
        ++fine_pool_total_;
        ++overflow_creates_;
        break;
    }
    const uint64_t fh = free_fine_handles_.back();
    free_fine_handles_.pop_back();
    return fh;
}

uint64_t TriPoolVMM::kv_va_base() {
    std::lock_guard<std::mutex> lock(mu_);
    return kv_va_;
}

int32_t TriPoolVMM::kv_alloc_page() {
    std::unique_lock<std::mutex> lock(mu_);
    if (!initialized_) return -1;
    try {
        backend_->ensure_context();
        if (kv_free_slots_.empty()) return -2;

        uint64_t fh = 0;
        try {
            fh = pop_fine_handle_locked(nullptr, &lock);
        } catch (const VmmOutOfMemoryError&) {
            reclaim_thinking_kv_locked(min_gran_, /*sync=*/true, nullptr, 0, nullptr, lock);
            fh = pop_fine_handle_locked(nullptr, &lock);
        }
        const int32_t slot = kv_free_slots_.back();
        kv_free_slots_.pop_back();

        const uint64_t addr = kv_va_ + static_cast<size_t>(slot) * min_gran_;
        try {
            backend_->map_range(addr, min_gran_, fh, /*need_host_access=*/true);
            backend_->set_access_rw(addr, min_gran_, /*need_host_access=*/true);
        } catch (...) {
            backend_->unmap_range(addr, min_gran_, /*was_host_accessible=*/true);
            free_fine_handles_.push_back(fh);
            kv_free_slots_.push_back(slot);
            throw;
        }
        kv_slot_handles_[static_cast<size_t>(slot)] = fh;
        ++kv_mapped_pages_;
        kv_mapped_pages_peak_ = std::max(kv_mapped_pages_peak_, kv_mapped_pages_);
        return slot;
    } catch (const VmmOutOfMemoryError&) {
        throw;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tripool_vmm] kv_alloc_page error: %s\n", e.what());
        return -3;
    }
}

int TriPoolVMM::kv_free_page(int32_t slot_idx, void* stream) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!initialized_) return -1;
    if (slot_idx < 0 || static_cast<size_t>(slot_idx) >= kv_va_slots_) return -2;
    const size_t s = static_cast<size_t>(slot_idx);
    if (!kv_slot_handles_[s]) return 0;
    try {
        backend_->ensure_context();
        backend_->sync_stream(stream);
        const uint64_t addr = kv_va_ + s * min_gran_;
        backend_->unmap_range(addr, min_gran_, /*was_host_accessible=*/true);
        free_fine_handles_.push_back(kv_slot_handles_[s]);
        kv_slot_handles_[s] = 0;
        kv_free_slots_.push_back(slot_idx);
        if (kv_mapped_pages_ > 0) --kv_mapped_pages_;
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tripool_vmm] kv_free_page error: %s\n", e.what());
        return -3;
    }
}

uint64_t TriPoolVMM::kv_session_create() {
    std::lock_guard<std::mutex> lock(mu_);
    const uint64_t sid = next_session_id_++;
    KvSession sess;
    sess.session_id = sid;
    sess.last_access_ns = mono_now_ns();
    kv_sessions_.emplace(sid, std::move(sess));
    return sid;
}

void TriPoolVMM::kv_session_destroy(uint64_t session_id) {
    if (session_id == 0) return;
    std::lock_guard<std::mutex> lock(mu_);
    auto it = kv_sessions_.find(session_id);
    if (it != kv_sessions_.end()) {
        for (int32_t rid : it->second.region_ids) {
            if (rid >= 0 && static_cast<size_t>(rid) < kv_regions_.size()) {
                kv_regions_[static_cast<size_t>(rid)].session_id = 0;
            }
        }
        kv_sessions_.erase(it);
    }
}

void TriPoolVMM::kv_session_touch(uint64_t session_id) {
    if (session_id == 0) return;
    std::lock_guard<std::mutex> lock(mu_);
    auto it = kv_sessions_.find(session_id);
    if (it != kv_sessions_.end()) {
        it->second.last_access_ns = mono_now_ns();
    }
}

void TriPoolVMM::kv_session_set_pinned(uint64_t session_id, bool pinned) {
    if (session_id == 0) return;
    std::lock_guard<std::mutex> lock(mu_);
    auto it = kv_sessions_.find(session_id);
    if (it == kv_sessions_.end()) return;
    it->second.pinned_active = pinned;
    if (pinned) {
        it->second.thinking = false;
        it->second.last_access_ns = mono_now_ns();
        for (int32_t rid : it->second.region_ids) {
            if (rid >= 0 && static_cast<size_t>(rid) < kv_regions_.size()) {
                auto& reg = kv_regions_[static_cast<size_t>(rid)];
                if (reg.active) {
                    reg.evictable = false;
                }
            }
        }
    }
}

int TriPoolVMM::kv_session_mark_active(uint64_t session_id, void* stream) {
    if (session_id == 0) return 0;
    std::unique_lock<std::mutex> lock(mu_);
    if (!initialized_) return -1;
    auto sit = kv_sessions_.find(session_id);
    if (sit == kv_sessions_.end()) return -2;

    sit->second.pinned_active = true;
    sit->second.thinking = false;
    sit->second.last_access_ns = mono_now_ns();

    std::vector<int32_t> swapped_rids;
    size_t needed_restore_bytes = 0;
    for (int32_t rid : sit->second.region_ids) {
        if (rid < 0 || static_cast<size_t>(rid) >= kv_regions_.size()) continue;
        auto& reg = kv_regions_[static_cast<size_t>(rid)];
        if (!reg.active) continue;
        reg.evictable = false;
        if (reg.state != KvRegionState::RESIDENT) {
            swapped_rids.push_back(rid);
            needed_restore_bytes += std::max<size_t>(1, reg.swapped_pages) * min_gran_;
        }
    }

    if (swapped_rids.empty()) {
        return 0;
    }

    try {
        backend_->ensure_context();
        const size_t wm_target = compute_watermark_reclaim_target_locked(/*for_dit=*/false, needed_restore_bytes);
        if (wm_target > 0) {
            reclaim_thinking_kv_locked(wm_target, /*sync=*/true, &session_id, 1, stream, lock);
        }

        std::sort(swapped_rids.begin(), swapped_rids.end());
        std::vector<std::pair<int32_t, size_t>> restored;
        try {
            for (int32_t rid : swapped_rids) {
                auto& reg = kv_regions_[static_cast<size_t>(rid)];
                if (!reg.active) continue;
                const size_t prev_valid = reg.swapped_valid_bytes;
                if (reg.state == KvRegionState::SWAP_OUT_FINISHING) {
                    cv_ready_.wait(lock, [&] {
                        return reg.state != KvRegionState::SWAP_OUT_FINISHING || stop_streamer_;
                    });
                }
                if (reg.state == KvRegionState::SWAP_OUT_QUEUED) {
                    cancel_queued_swap_out_locked(reg, stream);
                } else if (reg.state == KvRegionState::SWAPPED) {
                    try {
                        swap_in_region_locked(rid, reg, stream, &lock);
                    } catch (const VmmOutOfMemoryError&) {
                        const size_t reclaimed = reclaim_thinking_kv_locked(
                            std::max(min_gran_, needed_restore_bytes), /*sync=*/true, &session_id, 1, stream, lock);
                        if (reclaimed > 0) {
                            swap_in_region_locked(rid, reg, stream, &lock);
                        } else {
                            throw;
                        }
                    }
                    restored.emplace_back(rid, prev_valid);
                }
            }
        } catch (...) {
            for (const auto& rv : restored) {
                auto& r_reg = kv_regions_[static_cast<size_t>(rv.first)];
                if (r_reg.active && r_reg.state == KvRegionState::RESIDENT) {
                    submit_region_swap_out_locked(rv.first, r_reg, rv.second, stream);
                }
            }
            throw;
        }
        return 0;
    } catch (const VmmOutOfMemoryError&) {
        throw;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tripool_vmm] kv_session_mark_active error: %s\n", e.what());
        return -3;
    }
}

size_t TriPoolVMM::kv_session_mark_thinking(
    uint64_t session_id,
    const int32_t* region_ids,
    const size_t* valid_bytes,
    size_t num_regions,
    bool trigger_watermark,
    void* stream) {
    std::unique_lock<std::mutex> lock(mu_);
    if (!initialized_) return 0;
    if (session_id != 0) {
        auto sit = kv_sessions_.find(session_id);
        if (sit != kv_sessions_.end()) {
            sit->second.pinned_active = false;
            sit->second.thinking = true;
        }
    }
    for (size_t i = 0; i < num_regions; ++i) {
        const int32_t rid = region_ids[i];
        if (rid < 0 || static_cast<size_t>(rid) >= kv_regions_.size()) continue;
        auto& reg = kv_regions_[static_cast<size_t>(rid)];
        if (!reg.active || reg.state != KvRegionState::RESIDENT) continue;
        reg.evictable = true;
        reg.evictable_valid_bytes = valid_bytes ? valid_bytes[i] : (reg.handles.size() * min_gran_);
        reg.evictable_seq = ++evictable_clock_;
        reg.evictable_stream_u64 = reinterpret_cast<uint64_t>(stream);
    }
    if (!trigger_watermark) {
        return 0;
    }
    lock.unlock();
    return maybe_evict_thinking_kv_on_high_watermark(
        /*sync=*/false, /*exclude_session_id=*/0, /*extra_needed_bytes=*/0, /*for_dit=*/false, stream);
}

size_t TriPoolVMM::reclaim_thinking_kv_locked(
    size_t target_reclaim_bytes,
    bool sync,
    const uint64_t* exclude_session_ids,
    size_t num_excludes,
    void* stream,
    std::unique_lock<std::mutex>& lock) {
    if (!initialized_ || target_reclaim_bytes == 0) return 0;
    backend_->ensure_context();

    std::unordered_set<uint64_t> excluded;
    if (exclude_session_ids && num_excludes > 0) {
        excluded.reserve(num_excludes);
        for (size_t i = 0; i < num_excludes; ++i) {
            if (exclude_session_ids[i] != 0) {
                excluded.insert(exclude_session_ids[i]);
            }
        }
    }
    const auto is_excluded = [&](uint64_t sid) -> bool {
        return sid != 0 && !excluded.empty() && excluded.find(sid) != excluded.end();
    };

    const uint64_t now_ns = mono_now_ns();
    const uint64_t min_idle_ns = static_cast<uint64_t>(std::max(0.0, cfg_.kv_swap_min_idle_s) * 1e9);

    struct Candidate {
        int priority = 1;
        uint64_t last_access_ns = 0;
        uint64_t session_id = 0;
    };
    std::vector<Candidate> cands;
    cands.reserve(kv_sessions_.size());
    for (const auto& kv : kv_sessions_) {
        const KvSession& s = kv.second;
        if (is_excluded(s.session_id) || s.pinned_active) continue;
        const uint64_t idle_ns = (now_ns > s.last_access_ns) ? (now_ns - s.last_access_ns) : 0;
        if (!s.thinking && idle_ns < min_idle_ns) continue;

        bool has_resident = false;
        for (int32_t rid : s.region_ids) {
            if (rid >= 0 && static_cast<size_t>(rid) < kv_regions_.size()) {
                const auto& r = kv_regions_[static_cast<size_t>(rid)];
                if (r.active && r.state == KvRegionState::RESIDENT && !r.handles.empty()) {
                    has_resident = true;
                    break;
                }
            }
        }
        if (has_resident) {
            cands.push_back(Candidate{s.thinking ? 0 : 1, s.last_access_ns, s.session_id});
        }
    }

    std::sort(cands.begin(), cands.end(), [](const Candidate& a, const Candidate& b) {
        if (a.priority != b.priority) return a.priority < b.priority;
        if (a.last_access_ns != b.last_access_ns) return a.last_access_ns < b.last_access_ns;
        return a.session_id < b.session_id;
    });

    size_t reclaimed_bytes = 0;
    for (const auto& cand : cands) {
        if (reclaimed_bytes >= target_reclaim_bytes) break;
        auto sit = kv_sessions_.find(cand.session_id);
        if (sit == kv_sessions_.end()) continue;
        std::vector<int32_t> rids(sit->second.region_ids.begin(), sit->second.region_ids.end());
        std::sort(rids.begin(), rids.end());
        for (int32_t rid : rids) {
            if (rid < 0 || static_cast<size_t>(rid) >= kv_regions_.size()) continue;
            auto& reg = kv_regions_[static_cast<size_t>(rid)];
            if (!reg.active || reg.state != KvRegionState::RESIDENT || reg.handles.empty()) continue;
            const size_t valid_b =
                reg.evictable ? reg.evictable_valid_bytes : (reg.handles.size() * min_gran_);
            void* eff_s = stream ? stream : reinterpret_cast<void*>(reg.evictable_stream_u64);
            const int pages = submit_region_swap_out_locked(rid, reg, valid_b, eff_s);
            if (pages > 0) {
                reclaimed_bytes += static_cast<size_t>(pages) * min_gran_;
            }
        }
    }

    // Fallback: also reclaim standalone evictable regions not pinned or excluded
    while (reclaimed_bytes < target_reclaim_bytes) {
        int32_t best_rid = -1;
        uint64_t best_seq = UINT64_MAX;
        for (size_t idx = 0; idx < kv_regions_.size(); ++idx) {
            const auto& reg = kv_regions_[idx];
            if (!reg.active || !reg.evictable || reg.state != KvRegionState::RESIDENT || reg.handles.empty()) {
                continue;
            }
            if (is_excluded(reg.session_id)) continue;
            if (reg.session_id != 0) {
                auto sit = kv_sessions_.find(reg.session_id);
                if (sit != kv_sessions_.end() && sit->second.pinned_active) continue;
            }
            if (reg.evictable_seq < best_seq) {
                best_seq = reg.evictable_seq;
                best_rid = static_cast<int32_t>(idx);
            }
        }
        if (best_rid < 0) break;
        auto& reg = kv_regions_[static_cast<size_t>(best_rid)];
        const int pages = submit_region_swap_out_locked(best_rid, reg, reg.evictable_valid_bytes, stream);
        if (pages <= 0) break;
        reclaimed_bytes += static_cast<size_t>(pages) * min_gran_;
    }

    if (sync && reclaimed_bytes > 0) {
        while (drain_one_queued_swap_out_locked()) {
        }
        if (has_finishing_swap_locked()) {
            cv_ready_.wait(lock, [&] { return !has_finishing_swap_locked() || stop_streamer_; });
        }
    }
    return reclaimed_bytes;
}

size_t TriPoolVMM::reclaim_thinking_kv_bytes_multi(
    size_t target_reclaim_bytes,
    bool sync,
    const uint64_t* exclude_session_ids,
    size_t num_excludes,
    void* stream) {
    std::unique_lock<std::mutex> lock(mu_);
    return reclaim_thinking_kv_locked(target_reclaim_bytes, sync, exclude_session_ids, num_excludes, stream, lock);
}

size_t TriPoolVMM::reclaim_thinking_kv_bytes(
    size_t target_reclaim_bytes,
    bool sync,
    uint64_t exclude_session_id,
    void* stream) {
    return reclaim_thinking_kv_bytes_multi(
        target_reclaim_bytes,
        sync,
        exclude_session_id ? &exclude_session_id : nullptr,
        exclude_session_id ? 1 : 0,
        stream);
}

size_t TriPoolVMM::maybe_evict_thinking_kv_on_high_watermark_multi(
    bool sync,
    const uint64_t* exclude_session_ids,
    size_t num_excludes,
    size_t extra_needed_bytes,
    bool for_dit,
    void* stream) {
    std::unique_lock<std::mutex> lock(mu_);
    if (!initialized_) return 0;
    const size_t target_reclaim = compute_watermark_reclaim_target_locked(for_dit, extra_needed_bytes);
    if (target_reclaim == 0) return 0;
    return reclaim_thinking_kv_locked(target_reclaim, sync, exclude_session_ids, num_excludes, stream, lock);
}

size_t TriPoolVMM::maybe_evict_thinking_kv_on_high_watermark(
    bool sync,
    uint64_t exclude_session_id,
    size_t extra_needed_bytes,
    bool for_dit,
    void* stream) {
    return maybe_evict_thinking_kv_on_high_watermark_multi(
        sync,
        exclude_session_id ? &exclude_session_id : nullptr,
        exclude_session_id ? 1 : 0,
        extra_needed_bytes,
        for_dit,
        stream);
}

int32_t TriPoolVMM::kv_region_alloc(size_t max_va_bytes, uint64_t session_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!initialized_) return -1;
    try {
        backend_->ensure_context();
        const size_t req = (max_va_bytes > min_gran_) ? max_va_bytes : min_gran_;
        const size_t cap = ((req + min_gran_ - 1) / min_gran_) * min_gran_;
        const uint64_t va = backend_->reserve_va(cap, min_gran_);

        KvRegion fresh;
        fresh.va_base = va;
        fresh.cap_bytes = cap;
        fresh.session_id = session_id;
        fresh.state = KvRegionState::RESIDENT;
        fresh.active = true;

        int32_t rid = -1;
        if (!kv_free_regions_.empty()) {
            rid = kv_free_regions_.back();
            kv_free_regions_.pop_back();
            fresh.generation = kv_regions_[static_cast<size_t>(rid)].generation + 1;
            kv_regions_[static_cast<size_t>(rid)] = std::move(fresh);
        } else {
            rid = static_cast<int32_t>(kv_regions_.size());
            kv_regions_.push_back(std::move(fresh));
        }

        if (session_id != 0) {
            auto sit = kv_sessions_.find(session_id);
            if (sit != kv_sessions_.end()) {
                sit->second.region_ids.insert(rid);
            }
        }
        return rid;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tripool_vmm] kv_region_alloc error: %s\n", e.what());
        return -2;
    }
}

int TriPoolVMM::kv_region_set_evictable(
    int32_t region_id,
    int evictable,
    size_t valid_bytes,
    void* compute_stream) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!initialized_) return -1;
    if (region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return -2;
    auto& reg = kv_regions_[static_cast<size_t>(region_id)];
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

uint64_t TriPoolVMM::kv_region_ensure(int32_t region_id, size_t needed_bytes, void* stream) {
    std::unique_lock<std::mutex> lock(mu_);
    if (!initialized_) return 0;
    if (region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return 0;
    auto& reg = kv_regions_[static_cast<size_t>(region_id)];
    if (!reg.active) return 0;
    reg.evictable = false;

    const size_t eff_bytes = (needed_bytes > 0) ? needed_bytes : 1;
    const size_t needed_pages = (eff_bytes + min_gran_ - 1) / min_gran_;
    if (needed_pages * min_gran_ > reg.cap_bytes) {
        return 0;
    }

    try {
        backend_->ensure_context();
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
            return reg.va_base;
        }

        const size_t delta = needed_pages - old_pages;
        const size_t delta_bytes = delta * min_gran_;

        // Proactive watermark reclamation before mapping delta pages
        const size_t wm_target = compute_watermark_reclaim_target_locked(/*for_dit=*/false, delta_bytes);
        if (wm_target > 0) {
            const uint64_t ex_sid = reg.session_id;
            reclaim_thinking_kv_locked(
                wm_target, /*sync=*/false, ex_sid ? &ex_sid : nullptr, ex_sid ? 1 : 0, stream, lock);
        }

        std::vector<uint64_t> staged;
        size_t mapped_now = 0;
        try {
            staged.reserve(delta);
            for (size_t i = 0; i < delta; ++i) {
                try {
                    staged.push_back(pop_fine_handle_locked(stream, &lock, region_id));
                } catch (const VmmOutOfMemoryError&) {
                    const uint64_t ex_sid = reg.session_id;
                    const size_t rec = reclaim_thinking_kv_locked(
                        std::max(min_gran_, delta_bytes),
                        /*sync=*/true,
                        ex_sid ? &ex_sid : nullptr,
                        ex_sid ? 1 : 0,
                        stream,
                        lock);
                    if (rec > 0) {
                        staged.push_back(pop_fine_handle_locked(stream, &lock, region_id));
                    } else {
                        throw;
                    }
                }
            }
            if (!reg.active) {
                for (auto h : staged) free_fine_handles_.push_back(h);
                return 0;
            }
            for (size_t i = 0; i < delta; ++i) {
                const uint64_t addr = reg.va_base + (old_pages + i) * min_gran_;
                backend_->map_range(addr, min_gran_, staged[i], /*need_host_access=*/true);
                ++mapped_now;
            }
            backend_->set_access_rw(
                reg.va_base + old_pages * min_gran_, delta * min_gran_, /*need_host_access=*/true);
            for (auto h : staged) reg.handles.push_back(h);
            kv_mapped_pages_ += delta;
            kv_mapped_pages_peak_ = std::max(kv_mapped_pages_peak_, kv_mapped_pages_);
            return reg.va_base;
        } catch (...) {
            if (mapped_now > 0) {
                backend_->unmap_range(
                    reg.va_base + old_pages * min_gran_, mapped_now * min_gran_, /*was_host_accessible=*/true);
            }
            for (auto h : staged) free_fine_handles_.push_back(h);
            throw;
        }
    } catch (const VmmOutOfMemoryError&) {
        throw;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tripool_vmm] kv_region_ensure error: %s\n", e.what());
        return 0;
    }
}

int TriPoolVMM::submit_region_swap_out_locked(
    int32_t region_id,
    KvRegion& reg,
    size_t valid_bytes,
    void* compute_stream) {
    reg.evictable = false;
    if (reg.state != KvRegionState::RESIDENT || reg.handles.empty()) {
        return 0;
    }
    const size_t n_pages = reg.handles.size();
    const size_t mapped_bytes = n_pages * min_gran_;
    const size_t copy_bytes = std::min(valid_bytes, mapped_bytes);

    if (copy_bytes == 0) {
        backend_->sync_stream(compute_stream);
        if (reg.swap_event) {
            backend_->sync_event(reg.swap_event);
        }
        backend_->unmap_range(reg.va_base, mapped_bytes, /*was_host_accessible=*/true);
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
        reg.swap_event = backend_->create_event();
    }
    if (reg.swap_host_pinned && reg.swap_cap_bytes < copy_bytes) {
        backend_->sync_event(reg.swap_event);
        release_host_pinned_locked(reg.swap_host_pinned, reg.swap_cap_bytes);
        reg.swap_host_pinned = nullptr;
        reg.swap_cap_bytes = 0;
    }
    if (!reg.swap_host_pinned) {
        reg.swap_host_pinned = acquire_host_pinned_locked(copy_bytes, reg.swap_cap_bytes);
    }

    backend_->record_event(compute_bridge_event_, compute_stream);
    backend_->stream_wait_event(swap_stream_, compute_bridge_event_);
    backend_->memcpy_d2h_async(reg.swap_host_pinned, reg.va_base, copy_bytes, swap_stream_);
    backend_->record_event(reg.swap_event, swap_stream_);

    reg.swapped_valid_bytes = copy_bytes;
    reg.swapped_pages = n_pages;
    reg.state = KvRegionState::SWAP_OUT_QUEUED;
    ++reg.generation;
    swap_reap_queue_.push_back(SwapReapJob{region_id, reg.generation});
    cv_req_.notify_all();
    return static_cast<int>(n_pages);
}

int TriPoolVMM::kv_region_swap_out_async(int32_t region_id, size_t valid_bytes, void* compute_stream) {
    std::unique_lock<std::mutex> lock(mu_);
    if (!initialized_) return -1;
    if (region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return -2;
    auto& reg = kv_regions_[static_cast<size_t>(region_id)];
    if (!reg.active) return -2;
    try {
        backend_->ensure_context();
        return submit_region_swap_out_locked(region_id, reg, valid_bytes, compute_stream);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tripool_vmm] kv_region_swap_out_async error: %s\n", e.what());
        return -3;
    }
}

int TriPoolVMM::kv_region_swap_in_async(int32_t region_id, void* target_stream) {
    std::unique_lock<std::mutex> lock(mu_);
    if (!initialized_) return -1;
    if (region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return -2;
    auto& reg = kv_regions_[static_cast<size_t>(region_id)];
    if (!reg.active) return -2;
    reg.evictable = false;
    try {
        backend_->ensure_context();
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
    } catch (const VmmOutOfMemoryError&) {
        throw;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tripool_vmm] kv_region_swap_in_async error: %s\n", e.what());
        return -3;
    }
}

int TriPoolVMM::kv_region_is_swapped(int32_t region_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!initialized_ || region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return 0;
    const auto& reg = kv_regions_[static_cast<size_t>(region_id)];
    if (!reg.active) return 0;
    return (reg.state != KvRegionState::RESIDENT) ? 1 : 0;
}

size_t TriPoolVMM::kv_region_swapped_bytes(int32_t region_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!initialized_ || region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return 0;
    const auto& reg = kv_regions_[static_cast<size_t>(region_id)];
    if (!reg.active || reg.state == KvRegionState::RESIDENT) return 0;
    return reg.swapped_valid_bytes;
}

int TriPoolVMM::kv_flush_swaps() {
    std::unique_lock<std::mutex> lock(mu_);
    if (!initialized_) return -1;
    try {
        backend_->ensure_context();
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

uint64_t TriPoolVMM::kv_region_va(int32_t region_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!initialized_ || region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return 0;
    const auto& reg = kv_regions_[static_cast<size_t>(region_id)];
    return reg.active ? reg.va_base : 0;
}

size_t TriPoolVMM::kv_region_mapped_bytes(int32_t region_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!initialized_ || region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return 0;
    const auto& reg = kv_regions_[static_cast<size_t>(region_id)];
    if (!reg.active || reg.state != KvRegionState::RESIDENT) return 0;
    return reg.handles.size() * min_gran_;
}

int TriPoolVMM::kv_region_free(int32_t region_id, void* stream) {
    std::unique_lock<std::mutex> lock(mu_);
    if (!initialized_) return -1;
    if (region_id < 0 || static_cast<size_t>(region_id) >= kv_regions_.size()) return -2;
    auto& reg = kv_regions_[static_cast<size_t>(region_id)];
    if (!reg.active) return 0;
    reg.evictable = false;
    try {
        backend_->ensure_context();
        if (reg.state == KvRegionState::SWAP_OUT_FINISHING) {
            cv_ready_.wait(lock, [&] {
                return reg.state != KvRegionState::SWAP_OUT_FINISHING || stop_streamer_;
            });
            if (!reg.active) return 0;
        }
        if (reg.swap_event) {
            backend_->sync_event(reg.swap_event);
        }
        if (reg.state == KvRegionState::SWAPPED) {
            kv_swapped_pages_ = (kv_swapped_pages_ >= reg.swapped_pages) ? (kv_swapped_pages_ - reg.swapped_pages) : 0;
            kv_swapped_bytes_current_ =
                (kv_swapped_bytes_current_ >= reg.swapped_valid_bytes)
                    ? (kv_swapped_bytes_current_ - reg.swapped_valid_bytes)
                    : 0;
        }
        if (!reg.handles.empty()) {
            backend_->sync_stream(stream);
            const size_t n = reg.handles.size();
            backend_->unmap_range(reg.va_base, n * min_gran_, /*was_host_accessible=*/true);
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
            backend_->destroy_event(reg.swap_event);
            reg.swap_event = nullptr;
        }
        if (reg.va_base) {
            backend_->free_va(reg.va_base, reg.cap_bytes);
            reg.va_base = 0;
        }
        if (reg.session_id != 0) {
            auto sit = kv_sessions_.find(reg.session_id);
            if (sit != kv_sessions_.end()) {
                sit->second.region_ids.erase(region_id);
            }
            reg.session_id = 0;
        }
        reg.cap_bytes = 0;
        reg.swapped_valid_bytes = 0;
        reg.swapped_pages = 0;
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

// ==================== Arena 3: Opportunistic Expert Cache ====================

int TriPoolVMM::expert_register(
    int num_layers,
    size_t layer_stride_mib,
    void* host_pinned_base,
    size_t host_bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!initialized_ || num_layers <= 0 || layer_stride_mib == 0) return -1;
    try {
        backend_->ensure_context();
        const size_t req_stride = layer_stride_mib * (1ULL << 20);
        const size_t aligned_stride = ((req_stride + super_bytes_ - 1) / super_bytes_) * super_bytes_;
        const size_t handles_per_layer = aligned_stride / super_bytes_;
        const size_t total_va = static_cast<size_t>(num_layers) * aligned_stride;

        ExpertBank bank;
        bank.id = static_cast<int>(experts_.size());
        bank.num_layers = num_layers;
        bank.layer_stride_bytes = aligned_stride;
        bank.handles_per_layer = handles_per_layer;
        bank.va_bytes = total_va;
        bank.host_pinned_base = static_cast<uint8_t*>(host_pinned_base);
        bank.host_bytes = host_bytes;
        bank.pinned = false;
        bank.last_used_seq = ++expert_use_clock_;
        bank.va_base = backend_->reserve_va(total_va, super_bytes_);
        bank.layers.resize(static_cast<size_t>(num_layers));
        for (int l = 0; l < num_layers; ++l) {
            bank.layers[static_cast<size_t>(l)].ready_event = backend_->create_event();
        }
        experts_.push_back(std::move(bank));
        return static_cast<int>(experts_.size() - 1);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tripool_vmm] expert_register error: %s\n", e.what());
        return -2;
    }
}

uint64_t TriPoolVMM::expert_va_base(int expert_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return 0;
    return experts_[static_cast<size_t>(expert_id)].va_base;
}

uint64_t TriPoolVMM::expert_layer_va(int expert_id, int layer_idx) {
    std::lock_guard<std::mutex> lock(mu_);
    if (expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return 0;
    const auto& b = experts_[static_cast<size_t>(expert_id)];
    if (layer_idx < 0 || layer_idx >= b.num_layers) return 0;
    return b.va_base + static_cast<size_t>(layer_idx) * b.layer_stride_bytes;
}

size_t TriPoolVMM::expert_layer_stride_bytes(int expert_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return 0;
    return experts_[static_cast<size_t>(expert_id)].layer_stride_bytes;
}

int TriPoolVMM::expert_set_pinned(int expert_id, int pinned) {
    std::lock_guard<std::mutex> lock(mu_);
    if (expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return -1;
    auto& bank = experts_[static_cast<size_t>(expert_id)];
    bank.pinned = (pinned != 0);
    if (bank.pinned) {
        bank.last_used_seq = ++expert_use_clock_;
    }
    return 0;
}

int TriPoolVMM::expert_set_layer_used_bytes(int expert_id, int layer_idx, size_t used_bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    if (expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return -1;
    auto& bank = experts_[static_cast<size_t>(expert_id)];
    if (layer_idx < 0 || layer_idx >= bank.num_layers) return -2;
    if (used_bytes > bank.layer_stride_bytes) return -3;
    bank.layers[static_cast<size_t>(layer_idx)].used_bytes = used_bytes;
    return 0;
}

int TriPoolVMM::expert_load_layers_sync(int expert_id, int start_layer, int end_layer_exclusive) {
    std::unique_lock<std::mutex> lock(mu_);
    if (!initialized_ || expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return -1;
    auto& bank = experts_[static_cast<size_t>(expert_id)];
    bank.last_used_seq = ++expert_use_clock_;
    start_layer = std::max(0, start_layer);
    end_layer_exclusive = std::min(bank.num_layers, end_layer_exclusive);
    try {
        backend_->ensure_context();
        bank.last_async_error = false;
        bank.last_async_error_oom = false;
        bank.last_async_error_msg.clear();

        cv_ready_.wait(lock, [&] {
            if (stop_streamer_ || streamer_error_) return true;
            for (int i = start_layer; i < end_layer_exclusive; ++i) {
                const auto st = bank.layers[static_cast<size_t>(i)].state;
                if (st == LayerState::QUEUED || st == LayerState::STREAMING) return false;
            }
            return true;
        });
        if (stop_streamer_ || streamer_error_) return -2;

        for (int i = start_layer; i < end_layer_exclusive; ++i) {
            auto& lyr = bank.layers[static_cast<size_t>(i)];
            lyr.failed = false;
            lyr.failed_oom = false;
            lyr.failed_msg.clear();
        }

        int l = start_layer;
        while (l < end_layer_exclusive) {
            while (l < end_layer_exclusive &&
                   bank.layers[static_cast<size_t>(l)].state != LayerState::EVICTED) {
                ++l;
            }
            const int run_start = l;
            while (l < end_layer_exclusive &&
                   bank.layers[static_cast<size_t>(l)].state == LayerState::EVICTED) {
                ++l;
            }
            if (l > run_start) {
                map_and_submit_contiguous_locked(bank, run_start, l, &lock);
            }
        }
        backend_->sync_stream(h2d_stream_);
        for (int i = start_layer; i < end_layer_exclusive; ++i) {
            bank.layers[static_cast<size_t>(i)].needs_stream_wait = false;
        }
        return 0;
    } catch (const VmmOutOfMemoryError&) {
        throw;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tripool_vmm] expert_load_layers_sync error: %s\n", e.what());
        return -2;
    }
}

int TriPoolVMM::expert_prefetch_async(int expert_id, int start_layer, int end_layer_exclusive) {
    std::unique_lock<std::mutex> lock(mu_);
    if (!initialized_ || expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return -1;
    auto& bank = experts_[static_cast<size_t>(expert_id)];
    bank.last_used_seq = ++expert_use_clock_;
    start_layer = std::max(0, start_layer);
    end_layer_exclusive = std::min(bank.num_layers, end_layer_exclusive);

    bank.last_async_error = false;
    bank.last_async_error_oom = false;
    bank.last_async_error_msg.clear();

    int queued = 0;
    for (int l = start_layer; l < end_layer_exclusive; ++l) {
        auto& lyr = bank.layers[static_cast<size_t>(l)];
        if (lyr.state == LayerState::EVICTED) {
            lyr.failed = false;
            lyr.failed_oom = false;
            lyr.failed_msg.clear();
            lyr.state = LayerState::QUEUED;
            prefetch_queue_.push_back(PrefetchJob{expert_id, l});
            ++queued;
        }
    }
    if (queued > 0) {
        cv_req_.notify_all();
    }
    return queued;
}

int TriPoolVMM::expert_wait_layer(int expert_id, int layer_idx, void* compute_stream) {
    std::unique_lock<std::mutex> lock(mu_);
    if (!initialized_ || expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return -1;
    auto& bank = experts_[static_cast<size_t>(expert_id)];
    if (layer_idx < 0 || layer_idx >= bank.num_layers) return -2;
    bank.last_used_seq = ++expert_use_clock_;
    ++expert_wait_calls_;

    auto& lyr = bank.layers[static_cast<size_t>(layer_idx)];
    last_compute_stream_ = compute_stream;

    if (lyr.state == LayerState::RESIDENT && !lyr.needs_stream_wait) {
        return 0;
    }
    if (lyr.state == LayerState::EVICTED) {
        lyr.failed = false;
        lyr.failed_oom = false;
        lyr.failed_msg.clear();
        lyr.state = LayerState::QUEUED;
        prefetch_queue_.push_front(PrefetchJob{expert_id, layer_idx});
        cv_req_.notify_all();
    }
    if (lyr.state != LayerState::RESIDENT) {
        ++expert_wait_stalls_;
        cv_ready_.wait(lock, [&] {
            return lyr.state == LayerState::RESIDENT ||
                   lyr.state == LayerState::EVICTED ||
                   lyr.failed || stop_streamer_ || streamer_error_;
        });
        if (lyr.failed) {
            const bool is_oom = lyr.failed_oom;
            const std::string msg = lyr.failed_msg;
            lyr.failed = false;
            lyr.failed_oom = false;
            lyr.failed_msg.clear();
            bool any_other_failed = false;
            for (const auto& other : bank.layers) {
                if (other.failed) {
                    any_other_failed = true;
                    break;
                }
            }
            if (!any_other_failed) {
                bank.last_async_error = false;
                bank.last_async_error_oom = false;
                bank.last_async_error_msg.clear();
            }
            if (is_oom) {
                throw VmmOutOfMemoryError(msg);
            }
            return -3;
        }
        if (lyr.state != LayerState::RESIDENT) return -3;
    }
    if (lyr.needs_stream_wait) {
        try {
            backend_->ensure_context();
            backend_->stream_wait_event(compute_stream, lyr.ready_event);
            lyr.needs_stream_wait = false;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[tripool_vmm] expert_wait_layer error: %s\n", e.what());
            return -4;
        }
    }
    return 0;
}

int TriPoolVMM::expert_sync_h2d(int expert_id) {
    void* ev = nullptr;
    {
        std::unique_lock<std::mutex> lock(mu_);
        if (!initialized_ || expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return -1;
        auto& bank = experts_[static_cast<size_t>(expert_id)];
        cv_ready_.wait(lock, [&] {
            if (stop_streamer_ || streamer_error_) return true;
            for (const auto& lyr : bank.layers) {
                if (lyr.state == LayerState::QUEUED || lyr.state == LayerState::STREAMING) return false;
            }
            return true;
        });
        if (stop_streamer_ || streamer_error_) return -2;
        if (bank.last_async_error) {
            const bool is_oom = bank.last_async_error_oom;
            const std::string msg = bank.last_async_error_msg;
            bank.last_async_error = false;
            bank.last_async_error_oom = false;
            bank.last_async_error_msg.clear();
            for (auto& lyr : bank.layers) {
                lyr.failed = false;
                lyr.failed_oom = false;
                lyr.failed_msg.clear();
            }
            if (is_oom) {
                throw VmmOutOfMemoryError(msg);
            }
            return -2;
        }
        ev = bank.last_h2d_event;
        if (!ev) return 0;
        ++active_h2d_syncs_;
    }
    int rc = 0;
    try {
        backend_->ensure_context();
        if (!backend_->sync_event_noexcept(ev)) rc = -3;
    } catch (...) {
        rc = -3;
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (active_h2d_syncs_ > 0) --active_h2d_syncs_;
        cv_ready_.notify_all();
    }
    return rc;
}

size_t TriPoolVMM::coalesce_fine_to_super(size_t max_super) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!initialized_) return 0;
    try {
        backend_->ensure_context();
        return try_coalesce_fine_to_super_locked(max_super == 0 ? SIZE_MAX : max_super);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tripool_vmm] coalesce_fine_to_super error: %s\n", e.what());
        return 0;
    }
}

size_t TriPoolVMM::release_free_overflow_to_driver(size_t target_bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!initialized_ || !cfg_.allow_overflow || target_bytes == 0) return 0;
    const uint64_t cur_pool_b = super_pool_total_ * super_bytes_ + fine_pool_total_ * min_gran_;
    const uint64_t init_pool_b =
        static_cast<uint64_t>(init_super_handles_) * super_bytes_ +
        static_cast<uint64_t>(base_fine_pages_) * min_gran_;
    if (cur_pool_b <= init_pool_b) return 0;

    size_t excess_b = static_cast<size_t>(cur_pool_b - init_pool_b);
    size_t released = 0;
    try {
        backend_->ensure_context();
        while (released < target_bytes && excess_b >= super_bytes_ && !free_super_handles_.empty()) {
            const uint64_t h = free_super_handles_.back();
            free_super_handles_.pop_back();
            backend_->release_phys_handle(h, super_bytes_);
            if (super_pool_total_ > 0) --super_pool_total_;
            released += super_bytes_;
            excess_b -= super_bytes_;
        }
        while (released < target_bytes && excess_b >= min_gran_ && free_fine_handles_.size() > base_fine_pages_) {
            const uint64_t h = free_fine_handles_.back();
            free_fine_handles_.pop_back();
            backend_->release_phys_handle(h, min_gran_);
            if (fine_pool_total_ > 0) --fine_pool_total_;
            released += min_gran_;
            excess_b -= min_gran_;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tripool_vmm] release_free_overflow_to_driver error: %s\n", e.what());
    }
    return released;
}

int TriPoolVMM::expert_evict_layer(int expert_id, int layer_idx, void* stream) {
    std::unique_lock<std::mutex> lock(mu_);
    if (!initialized_ || expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return -1;
    auto& bank = experts_[static_cast<size_t>(expert_id)];
    if (layer_idx < 0 || layer_idx >= bank.num_layers) return -2;

    auto& lyr = bank.layers[static_cast<size_t>(layer_idx)];
    if (lyr.state == LayerState::QUEUED) {
        prefetch_queue_.erase(
            std::remove_if(prefetch_queue_.begin(), prefetch_queue_.end(), [&](const PrefetchJob& j) {
                return j.expert_id == expert_id && j.layer_idx == layer_idx;
            }),
            prefetch_queue_.end());
        lyr.state = LayerState::EVICTED;
        cv_ready_.notify_all();
        return 0;
    }
    if (lyr.state == LayerState::STREAMING) {
        cv_ready_.wait(lock, [&] { return lyr.state != LayerState::STREAMING || stop_streamer_; });
    }

    try {
        backend_->ensure_context();
        if (evict_layer_locked(bank, layer_idx, stream, /*is_auto=*/false, /*for_kv=*/false)) {
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tripool_vmm] expert_evict_layer error: %s\n", e.what());
        return -3;
    }
}

int TriPoolVMM::expert_evict_range(int expert_id, int start_layer, int end_layer_exclusive, void* stream) {
    std::unique_lock<std::mutex> lock(mu_);
    if (!initialized_ || expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return -1;
    auto& bank = experts_[static_cast<size_t>(expert_id)];
    start_layer = std::max(0, start_layer);
    end_layer_exclusive = std::min(bank.num_layers, end_layer_exclusive);

    bool cancelled_any = false;
    prefetch_queue_.erase(
        std::remove_if(prefetch_queue_.begin(), prefetch_queue_.end(), [&](const PrefetchJob& j) {
            if (j.expert_id == expert_id && j.layer_idx >= start_layer && j.layer_idx < end_layer_exclusive) {
                if (bank.layers[static_cast<size_t>(j.layer_idx)].state == LayerState::QUEUED) {
                    bank.layers[static_cast<size_t>(j.layer_idx)].state = LayerState::EVICTED;
                    cancelled_any = true;
                }
                return true;
            }
            return false;
        }),
        prefetch_queue_.end());
    if (cancelled_any) {
        cv_ready_.notify_all();
    }
    cv_ready_.wait(lock, [&] {
        if (stop_streamer_) return true;
        for (int l = start_layer; l < end_layer_exclusive; ++l) {
            if (bank.layers[static_cast<size_t>(l)].state == LayerState::STREAMING) return false;
        }
        return true;
    });

    try {
        backend_->ensure_context();
        backend_->sync_stream(h2d_stream_);
        backend_->sync_stream(stream);
        int evicted = 0;
        for (int l = end_layer_exclusive - 1; l >= start_layer; --l) {
            if (evict_layer_locked(bank, l, nullptr, /*is_auto=*/false, /*for_kv=*/false, /*already_synced=*/true)) {
                ++evicted;
            }
        }
        return evicted;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tripool_vmm] expert_evict_range error: %s\n", e.what());
        return -2;
    }
}

int TriPoolVMM::expert_is_resident(int expert_id, int layer_idx) {
    std::lock_guard<std::mutex> lock(mu_);
    if (expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return 0;
    const auto& b = experts_[static_cast<size_t>(expert_id)];
    if (layer_idx < 0 || layer_idx >= b.num_layers) return 0;
    return b.layers[static_cast<size_t>(layer_idx)].state == LayerState::RESIDENT ? 1 : 0;
}

int TriPoolVMM::expert_resident_count(int expert_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) return 0;
    int cnt = 0;
    for (const auto& lyr : experts_[static_cast<size_t>(expert_id)].layers) {
        if (lyr.state == LayerState::RESIDENT) ++cnt;
    }
    return cnt;
}

size_t TriPoolVMM::ensure_expert_headroom(
    int expert_id,
    size_t needed_bytes,
    const uint64_t* exclude_session_ids,
    size_t num_excludes,
    void* stream) {
    std::unique_lock<std::mutex> lock(mu_);
    if (!initialized_ || needed_bytes == 0) return 0;
    if (expert_id >= 0 && static_cast<size_t>(expert_id) >= experts_.size()) return 0;

    auto compute_avail = [&](uint64_t& out_avail_now, uint64_t& out_inflight) {
        const KvBudgetSnapshot snap = compute_kv_budget_locked(/*for_dit=*/true);
        uint64_t other_unpinned_exp = 0;
        for (const auto& bank : experts_) {
            if (bank.pinned || bank.id == expert_id) continue;
            for (const auto& lyr : bank.layers) {
                if (lyr.state == LayerState::RESIDENT && !lyr.slots.empty()) {
                    other_unpinned_exp += bank.layer_stride_bytes;
                }
            }
        }
        uint64_t growth_headroom = 0;
        if (cfg_.allow_overflow) {
            const uint64_t drv_free = backend_->driver_free_overflow_bytes();
            if (cfg_.max_pool_bytes > 0) {
                const uint64_t cap_rem =
                    (cfg_.max_pool_bytes > snap.pool_total_bytes)
                        ? (cfg_.max_pool_bytes - snap.pool_total_bytes)
                        : 0;
                growth_headroom = (drv_free > 0) ? std::min(cap_rem, drv_free) : cap_rem;
            } else {
                growth_headroom = drv_free;
            }
        }
        out_avail_now = snap.pool_free_bytes + other_unpinned_exp + growth_headroom;
        out_inflight = snap.inflight_swap_bytes;
    };

    size_t total_reclaimed = 0;
    const size_t wm_target = compute_watermark_reclaim_target_locked(/*for_dit=*/true, /*extra_needed_bytes=*/0);
    if (wm_target > 0) {
        total_reclaimed += reclaim_thinking_kv_locked(
            wm_target, /*sync=*/false, exclude_session_ids, num_excludes, stream, lock);
    }

    uint64_t avail_now = 0;
    uint64_t inflight_swap = 0;
    compute_avail(avail_now, inflight_swap);
    if (avail_now + inflight_swap < needed_bytes) {
        const size_t deficit = static_cast<size_t>(needed_bytes - (avail_now + inflight_swap));
        total_reclaimed += reclaim_thinking_kv_locked(
            deficit, /*sync=*/true, exclude_session_ids, num_excludes, stream, lock);
        compute_avail(avail_now, inflight_swap);
    }

    if (avail_now < needed_bytes && (inflight_swap > 0 || has_finishing_swap_locked())) {
        backend_->ensure_context();
        while (drain_one_queued_swap_out_locked()) {
        }
        if (has_finishing_swap_locked()) {
            cv_ready_.wait(lock, [&] { return !has_finishing_swap_locked() || stop_streamer_; });
        }
    }
    return total_reclaimed;
}

void TriPoolVMM::get_stats(UnifiedVmmStatsPod* out) {
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

    const KvBudgetSnapshot snap_ar = compute_kv_budget_locked(/*for_dit=*/false);
    const KvBudgetSnapshot snap_dit = compute_kv_budget_locked(/*for_dit=*/true);
    const uint64_t extra = (initialized_ && cfg_.allow_overflow) ? backend_->driver_free_overflow_bytes() : 0ULL;
    out->pool_total_bytes = snap_ar.pool_total_bytes;
    out->pool_free_bytes = snap_ar.pool_free_bytes;
    out->expert_unpinned_bytes = snap_ar.expert_unpinned_bytes;
    out->kv_free_bytes = clamp_by_max_pool_locked(snap_ar.kv_free_bytes + extra, snap_ar);
    out->kv_dit_headroom_bytes = clamp_by_max_pool_locked(snap_dit.kv_dit_headroom_bytes + extra, snap_dit);

    out->kv_swapped_pages_current = kv_swapped_pages_;
    out->kv_swapped_bytes_current = kv_swapped_bytes_current_;
    out->kv_swap_outs_total = kv_swap_outs_total_;
    out->kv_swap_ins_total = kv_swap_ins_total_;
    out->kv_swap_d2h_bytes_total = kv_swap_d2h_bytes_total_;
    out->kv_swap_h2d_bytes_total = kv_swap_h2d_bytes_total_;
    out->kv_host_pinned_pool_bytes = kv_host_pinned_pool_bytes_;
}

// ==================== Internal Helpers ====================

void TriPoolVMM::trim_idle_host_pinned_locked(size_t max_idle_bytes) {
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
        if (p) backend_->free_host_pinned(p);
        free_host_pinned_bytes_ = (free_host_pinned_bytes_ >= blk_cap) ? (free_host_pinned_bytes_ - blk_cap) : 0;
        kv_host_pinned_pool_bytes_ =
            (kv_host_pinned_pool_bytes_ >= blk_cap) ? (kv_host_pinned_pool_bytes_ - blk_cap) : 0;
    }
}

void* TriPoolVMM::acquire_host_pinned_locked(size_t min_bytes, size_t& out_cap) {
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
    void* p = backend_->alloc_host_pinned(cap);
    kv_host_pinned_pool_bytes_ += cap;
    out_cap = cap;
    return p;
}

void TriPoolVMM::release_host_pinned_locked(void* p, size_t cap) {
    if (!p || cap == 0) return;
    free_host_blocks_[cap].push_back(p);
    free_host_pinned_bytes_ += cap;
    trim_idle_host_pinned_locked(kMaxIdleHostPinnedBytes);
}

size_t TriPoolVMM::total_evictable_thinking_pages_locked(int32_t exclude_region_id) const {
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

bool TriPoolVMM::evict_one_thinking_kv_region_locked(void* caller_stream, int32_t exclude_region_id) {
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

    auto& reg = kv_regions_[static_cast<size_t>(victim_id)];
    const size_t n_pages = reg.handles.size();
    const size_t mapped_bytes = n_pages * min_gran_;
    const size_t copy_bytes = std::min(reg.evictable_valid_bytes, mapped_bytes);
    void* reg_stream = reinterpret_cast<void*>(reg.evictable_stream_u64);

    if (copy_bytes > 0) {
        if (!reg.swap_event) {
            reg.swap_event = backend_->create_event();
        }
        if (reg.swap_host_pinned && reg.swap_cap_bytes < copy_bytes) {
            backend_->sync_event(reg.swap_event);
            release_host_pinned_locked(reg.swap_host_pinned, reg.swap_cap_bytes);
            reg.swap_host_pinned = nullptr;
            reg.swap_cap_bytes = 0;
        }
        if (!reg.swap_host_pinned) {
            reg.swap_host_pinned = acquire_host_pinned_locked(copy_bytes, reg.swap_cap_bytes);
        }
        backend_->record_event(compute_bridge_event_, reg_stream);
        backend_->stream_wait_event(swap_stream_, compute_bridge_event_);
        if (caller_stream != reg_stream) {
            backend_->record_event(compute_bridge_event_, caller_stream);
            backend_->stream_wait_event(swap_stream_, compute_bridge_event_);
        }
        backend_->memcpy_d2h_async(reg.swap_host_pinned, reg.va_base, copy_bytes, swap_stream_);
        backend_->record_event(reg.swap_event, swap_stream_);
        backend_->sync_event(reg.swap_event);
    } else {
        backend_->sync_stream(reg_stream);
        if (caller_stream != reg_stream) {
            backend_->sync_stream(caller_stream);
        }
        if (reg.swap_event) {
            backend_->sync_event(reg.swap_event);
        }
    }

    backend_->unmap_range(reg.va_base, mapped_bytes, /*was_host_accessible=*/true);
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

bool TriPoolVMM::has_finishing_swap_locked() const {
    for (const auto& reg : kv_regions_) {
        if (reg.active && reg.state == KvRegionState::SWAP_OUT_FINISHING) return true;
    }
    return false;
}

bool TriPoolVMM::drain_one_queued_swap_out_locked() {
    for (auto& reg : kv_regions_) {
        if (!reg.active || reg.state != KvRegionState::SWAP_OUT_QUEUED) continue;
        if (reg.swap_event) {
            backend_->sync_event(reg.swap_event);
        }
        const size_t n = reg.handles.size();
        if (n > 0) {
            backend_->unmap_range(reg.va_base, n * min_gran_, /*was_host_accessible=*/true);
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

void TriPoolVMM::cancel_queued_swap_out_locked(KvRegion& reg, void* target_stream) {
    if (reg.swap_event) {
        if (target_stream) {
            backend_->stream_wait_event(target_stream, reg.swap_event);
        } else {
            backend_->sync_event(reg.swap_event);
        }
    }
    reg.state = KvRegionState::RESIDENT;
    ++reg.generation;
}

void TriPoolVMM::swap_in_region_locked(
    int32_t region_id,
    KvRegion& reg,
    void* target_stream,
    std::unique_lock<std::mutex>* opt_lock) {
    const size_t pages = reg.swapped_pages;
    const size_t valid_bytes = reg.swapped_valid_bytes;
    if (pages == 0) {
        reg.state = KvRegionState::RESIDENT;
        ++reg.generation;
        return;
    }

    std::vector<uint64_t> staged;
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
            backend_->map_range(reg.va_base + i * min_gran_, min_gran_, staged[i], /*need_host_access=*/true);
            ++mapped_now;
        }
        backend_->set_access_rw(reg.va_base, pages * min_gran_, /*need_host_access=*/true);

        if (valid_bytes > 0 && reg.swap_host_pinned) {
            if (!reg.swap_event) {
                reg.swap_event = backend_->create_event();
            }
            backend_->memcpy_h2d_async(reg.va_base, reg.swap_host_pinned, valid_bytes, swap_stream_);
            backend_->record_event(reg.swap_event, swap_stream_);
            if (target_stream) {
                backend_->stream_wait_event(target_stream, reg.swap_event);
            } else {
                backend_->sync_stream(swap_stream_);
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
            backend_->unmap_range(reg.va_base, mapped_now * min_gran_, /*was_host_accessible=*/true);
        }
        for (auto fh : staged) free_fine_handles_.push_back(fh);
        throw;
    }
}

size_t TriPoolVMM::try_coalesce_fine_to_super_locked(size_t max_super) {
    const size_t limit = (max_super == 0) ? SIZE_MAX : max_super;
    size_t coalesced = 0;
    while (coalesced < limit &&
           free_fine_handles_.size() >= sub_pages_ &&
           fine_pool_total_ >= base_fine_pages_ + sub_pages_) {
        std::vector<uint64_t> batch;
        batch.reserve(sub_pages_);
        for (size_t j = 0; j < sub_pages_; ++j) {
            batch.push_back(free_fine_handles_.back());
            free_fine_handles_.pop_back();
        }
        // Strictly release all fine handles BEFORE creating 1 super handle so peak `pool_total_bytes` never grows.
        for (uint64_t fh : batch) {
            backend_->release_phys_handle(fh, min_gran_);
        }
        fine_pool_total_ -= sub_pages_;
        try {
            const uint64_t sh = backend_->create_phys_handle(super_bytes_);
            ++super_pool_total_;
            free_super_handles_.push_back(sh);
            ++coalesced;
        } catch (...) {
            for (size_t j = 0; j < sub_pages_; ++j) {
                try {
                    free_fine_handles_.push_back(backend_->create_phys_handle(min_gran_));
                    ++fine_pool_total_;
                } catch (...) {
                    break;
                }
            }
            break;
        }
    }
    return coalesced;
}

void TriPoolVMM::release_super_slot_to_pool_locked(SuperSlot& slot) {
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

void TriPoolVMM::map_super_slot_locked(uint64_t slot_va, const SuperSlot& slot, bool host_access) {
    if (!slot.is_fine_bundle()) {
        backend_->map_range(slot_va, super_bytes_, slot.super_h, host_access);
    } else {
        size_t mapped_sub = 0;
        try {
            for (size_t j = 0; j < slot.fine_hs.size(); ++j) {
                backend_->map_range(slot_va + j * min_gran_, min_gran_, slot.fine_hs[j], host_access);
                ++mapped_sub;
            }
        } catch (...) {
            if (mapped_sub > 0) {
                backend_->unmap_range(slot_va, mapped_sub * min_gran_, host_access);
            }
            throw;
        }
    }
}

bool TriPoolVMM::evict_layer_locked(
    ExpertBank& bank,
    int layer_idx,
    void* stream,
    bool is_auto,
    bool for_kv,
    bool already_synced) {
    auto& lyr = bank.layers[static_cast<size_t>(layer_idx)];
    if (lyr.state != LayerState::RESIDENT || lyr.slots.empty()) return false;

    if (!already_synced) {
        backend_->sync_stream(h2d_stream_);
        backend_->sync_stream(stream);
        if (last_compute_stream_ && last_compute_stream_ != stream) {
            backend_->sync_stream(last_compute_stream_);
        }
    }

    const uint64_t base = bank.va_base + static_cast<size_t>(layer_idx) * bank.layer_stride_bytes;
    const auto t0 = std::chrono::steady_clock::now();
    backend_->unmap_range(base, bank.layer_stride_bytes, /*was_host_accessible=*/true);
    const auto t1 = std::chrono::steady_clock::now();
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

bool TriPoolVMM::evict_one_tail_expert_layer_locked(void* stream, bool for_kv, int exclude_expert_id) {
    // LRU multi-bank eviction: when multiple expert banks share super_pool (e.g. VLM + Decoder in
    // constrained bridge_mllm_dit), always evict the least-recently-used unpinned bank first so an
    // active bank's tail layers are not discarded while a stale inactive bank occupies physical handles.
    ExpertBank* victim_bank = nullptr;
    int victim_layer = -1;
    for (auto& bank : experts_) {
        if (bank.pinned || bank.id == exclude_expert_id) continue;
        int tail_res = -1;
        for (int l = bank.num_layers - 1; l >= 0; --l) {
            if (bank.layers[static_cast<size_t>(l)].state == LayerState::RESIDENT) {
                tail_res = l;
                break;
            }
        }
        if (tail_res < 0) continue;
        if (victim_bank == nullptr ||
            bank.last_used_seq < victim_bank->last_used_seq ||
            (bank.last_used_seq == victim_bank->last_used_seq && bank.id < victim_bank->id)) {
            victim_bank = &bank;
            victim_layer = tail_res;
        }
    }
    if (victim_bank != nullptr && victim_layer >= 0) {
        return evict_layer_locked(*victim_bank, victim_layer, stream, /*is_auto=*/true, for_kv);
    }
    return false;
}

TriPoolVMM::SuperSlot TriPoolVMM::pop_fine_bundle_locked() {
    SuperSlot s;
    s.fine_hs.reserve(sub_pages_);
    for (size_t j = 0; j < sub_pages_; ++j) {
        s.fine_hs.push_back(free_fine_handles_.back());
        free_fine_handles_.pop_back();
    }
    ++fine_to_super_bundles_;
    return s;
}

TriPoolVMM::SuperSlot TriPoolVMM::acquire_super_slot_locked(
    void* stream,
    bool for_kv,
    int exclude_expert_id,
    std::unique_lock<std::mutex>* opt_lock) {
    while (true) {
        if (!free_super_handles_.empty()) {
            const uint64_t h = free_super_handles_.back();
            free_super_handles_.pop_back();
            return SuperSlot{h, {}};
        }
        // Reuse excess carved fine pages as zero-allocation fine-to-super bundles by default;
        // lazily coalesce back to a 64 MiB super handle only when overflow is enabled and free
        // fine pages exceed a 2*sub_pages_ hysteresis margin above base_fine_pages_ (preventing
        // carve<->coalesce thrashing and preserving strict-cap fine-to-super bundling).
        if (free_fine_handles_.size() >= base_fine_pages_ + sub_pages_) {
            if (cfg_.allow_overflow &&
                free_fine_handles_.size() >= base_fine_pages_ + 2 * sub_pages_ &&
                try_coalesce_fine_to_super_locked(1) > 0 &&
                !free_super_handles_.empty()) {
                const uint64_t h = free_super_handles_.back();
                free_super_handles_.pop_back();
                return SuperSlot{h, {}};
            }
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
        if (cfg_.allow_overflow && cfg_.max_pool_bytes > 0) {
            const uint64_t cur_pool_b = super_pool_total_ * super_bytes_ + fine_pool_total_ * min_gran_;
            if (cur_pool_b + super_bytes_ <= cfg_.max_pool_bytes) {
                const uint64_t h = backend_->create_phys_handle(super_bytes_);
                ++super_pool_total_;
                ++overflow_creates_;
                return SuperSlot{h, {}};
            }
        }
        if (evict_one_tail_expert_layer_locked(stream, for_kv, exclude_expert_id)) {
            continue;
        }
        if (free_fine_handles_.size() >= sub_pages_) {
            return pop_fine_bundle_locked();
        }
        if (cfg_.allow_overflow ||
            (free_fine_handles_.size() + total_evictable_thinking_pages_locked() >= sub_pages_)) {
            if (evict_one_thinking_kv_region_locked(stream, /*exclude_region_id=*/-1)) {
                continue;
            }
        }
        break;
    }

    if (!cfg_.allow_overflow) {
        throw VmmOutOfMemoryError("TriPool super-handle pool exhausted (allow_overflow=0)");
    }
    const uint64_t cur_pool_b = super_pool_total_ * super_bytes_ + fine_pool_total_ * min_gran_;
    if (!free_fine_handles_.empty()) {
        const size_t deficit = sub_pages_ - free_fine_handles_.size();
        const uint64_t add_b = static_cast<uint64_t>(deficit) * min_gran_;
        if (cfg_.max_pool_bytes > 0 && (cur_pool_b + add_b > cfg_.max_pool_bytes)) {
            throw VmmOutOfMemoryError("TriPool super-handle pool reached max_pool_bytes cap");
        }
        for (size_t j = 0; j < deficit; ++j) {
            free_fine_handles_.push_back(backend_->create_phys_handle(min_gran_));
            ++fine_pool_total_;
            ++overflow_creates_;
        }
        return pop_fine_bundle_locked();
    }
    if (cfg_.max_pool_bytes > 0 && (cur_pool_b + super_bytes_ > cfg_.max_pool_bytes)) {
        throw VmmOutOfMemoryError("TriPool super-handle pool reached max_pool_bytes cap");
    }
    const uint64_t h = backend_->create_phys_handle(super_bytes_);
    ++super_pool_total_;
    ++overflow_creates_;
    return SuperSlot{h, {}};
}

void TriPoolVMM::map_and_submit_contiguous_locked(
    ExpertBank& bank,
    int start_layer,
    int end_layer,
    std::unique_lock<std::mutex>* opt_lock) {
    if (start_layer >= end_layer) return;
    const int count = end_layer - start_layer;
    const size_t total_handles = static_cast<size_t>(count) * bank.handles_per_layer;

    std::vector<SuperSlot> all_slots;
    all_slots.reserve(total_handles);
    size_t mapped_slots = 0;
    const uint64_t batch_va = bank.va_base + static_cast<size_t>(start_layer) * bank.layer_stride_bytes;
    const size_t batch_bytes = static_cast<size_t>(count) * bank.layer_stride_bytes;

    try {
        for (size_t i = 0; i < total_handles; ++i) {
            all_slots.push_back(acquire_super_slot_locked(nullptr, /*for_kv=*/false, bank.id, opt_lock));
        }

        const auto tm0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < total_handles; ++i) {
            map_super_slot_locked(batch_va + i * super_bytes_, all_slots[i], /*host_access=*/true);
            ++mapped_slots;
        }
        backend_->set_access_rw(batch_va, batch_bytes, /*need_host_access=*/true);
        const auto tm1 = std::chrono::steady_clock::now();
        expert_map_us_ += std::chrono::duration<double, std::micro>(tm1 - tm0).count();

        const auto th0 = std::chrono::steady_clock::now();
        for (int idx = 0; idx < count; ++idx) {
            const int l = start_layer + idx;
            auto& lyr = bank.layers[static_cast<size_t>(l)];
            const size_t byte_off = static_cast<size_t>(l) * bank.layer_stride_bytes;
            const uint64_t dst_va = bank.va_base + byte_off;
            const size_t copy_bytes =
                (lyr.used_bytes > 0 && lyr.used_bytes <= bank.layer_stride_bytes)
                    ? lyr.used_bytes
                    : bank.layer_stride_bytes;

            if (bank.host_pinned_base && byte_off < bank.host_bytes) {
                const size_t avail = std::min(copy_bytes, bank.host_bytes - byte_off);
                if (avail > 0) {
                    backend_->memcpy_h2d_async(dst_va, bank.host_pinned_base + byte_off, avail, h2d_stream_);
                }
            }
            backend_->record_event(lyr.ready_event, h2d_stream_);
            bank.last_h2d_event = lyr.ready_event;

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
            lyr.failed = false;
            lyr.failed_oom = false;
            lyr.failed_msg.clear();
            expert_mapped_handles_ += bank.handles_per_layer;
            expert_fine_bundles_current_ += layer_bundles;
            ++expert_resident_layers_;
            ++expert_prefetch_layers_completed_;
            expert_h2d_bytes_total_ += copy_bytes;
        }
        const auto th1 = std::chrono::steady_clock::now();
        expert_h2d_submit_us_ += std::chrono::duration<double, std::micro>(th1 - th0).count();
    } catch (...) {
        if (mapped_slots > 0) {
            backend_->unmap_range(batch_va, mapped_slots * super_bytes_, /*was_host_accessible=*/true);
        }
        for (auto& s : all_slots) {
            release_super_slot_to_pool_locked(s);
        }
        throw;
    }
}

void TriPoolVMM::extend_transient_mapped_locked(
    size_t add_handles,
    void* stream,
    std::unique_lock<std::mutex>* opt_lock) {
    if (add_handles == 0) return;
    const size_t start_idx = transient_mapped_slots_.size();
    const size_t old_end = start_idx * super_bytes_;
    const size_t add_bytes = add_handles * super_bytes_;

    std::vector<SuperSlot> acquired;
    acquired.reserve(add_handles);
    size_t mapped_now = 0;
    try {
        for (size_t i = 0; i < add_handles; ++i) {
            acquired.push_back(acquire_super_slot_locked(stream, /*for_kv=*/false, /*exclude_expert_id=*/-1, opt_lock));
        }

        const auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < add_handles; ++i) {
            map_super_slot_locked(transient_va_ + (start_idx + i) * super_bytes_, acquired[i], /*host_access=*/true);
            ++mapped_now;
        }
        backend_->set_access_rw(transient_va_ + old_end, add_bytes, /*need_host_access=*/true);
        const auto t1 = std::chrono::steady_clock::now();
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
            backend_->unmap_range(transient_va_ + old_end, mapped_now * super_bytes_, /*was_host_accessible=*/true);
        }
        for (auto& s : acquired) {
            release_super_slot_to_pool_locked(s);
        }
        throw;
    }
}

void TriPoolVMM::insert_coalesce_transient_locked(size_t off, size_t sz) {
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

void TriPoolVMM::bulk_unmap_transient_locked(void* stream) {
    backend_->sync_stream(stream);
    const size_t total_bytes = transient_mapped_slots_.size() * super_bytes_;
    const auto t0 = std::chrono::steady_clock::now();
    backend_->unmap_range(transient_va_, total_bytes, /*was_host_accessible=*/true);
    const auto t1 = std::chrono::steady_clock::now();
    transient_unmap_us_ += std::chrono::duration<double, std::micro>(t1 - t0).count();

    for (auto& s : transient_mapped_slots_) {
        release_super_slot_to_pool_locked(s);
    }
    transient_mapped_slots_.clear();
    transient_fine_bundles_current_ = 0;
    transient_free_blocks_.clear();
    ++transient_scope_flushes_;
}

void TriPoolVMM::streamer_loop() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (stop_streamer_) {
            cv_ready_.notify_all();
            return;
        }
        try {
            backend_->ensure_context();
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

        if (!prefetch_queue_.empty()) {
            PrefetchJob first = prefetch_queue_.front();
            prefetch_queue_.pop_front();
            if (first.expert_id < 0 || static_cast<size_t>(first.expert_id) >= experts_.size()) continue;
            auto& bank = experts_[static_cast<size_t>(first.expert_id)];
            if (first.layer_idx < 0 || first.layer_idx >= bank.num_layers) continue;
            if (bank.layers[static_cast<size_t>(first.layer_idx)].state != LayerState::QUEUED) continue;

            const int start_l = first.layer_idx;
            int end_l = start_l + 1;
            while (!prefetch_queue_.empty()) {
                const auto& nxt = prefetch_queue_.front();
                if (nxt.expert_id == first.expert_id &&
                    nxt.layer_idx == end_l &&
                    end_l < bank.num_layers &&
                    bank.layers[static_cast<size_t>(end_l)].state == LayerState::QUEUED) {
                    prefetch_queue_.pop_front();
                    ++end_l;
                } else {
                    break;
                }
            }
            for (int l = start_l; l < end_l; ++l) {
                bank.layers[static_cast<size_t>(l)].state = LayerState::STREAMING;
            }
            try {
                backend_->ensure_context();
                map_and_submit_contiguous_locked(bank, start_l, end_l, &lock);
            } catch (const VmmOutOfMemoryError& e) {
                std::fprintf(stderr, "[tripool_vmm] streamer layers [%d,%d) OOM: %s\n", start_l, end_l, e.what());
                for (int l = start_l; l < end_l; ++l) {
                    auto& lyr = bank.layers[static_cast<size_t>(l)];
                    if (lyr.state != LayerState::RESIDENT) {
                        lyr.state = LayerState::EVICTED;
                        lyr.failed = true;
                        lyr.failed_oom = true;
                        lyr.failed_msg = e.what();
                    }
                }
                bank.last_async_error = true;
                bank.last_async_error_oom = true;
                bank.last_async_error_msg = e.what();
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[tripool_vmm] streamer layers [%d,%d) error: %s\n", start_l, end_l, e.what());
                for (int l = start_l; l < end_l; ++l) {
                    auto& lyr = bank.layers[static_cast<size_t>(l)];
                    if (lyr.state != LayerState::RESIDENT) {
                        lyr.state = LayerState::EVICTED;
                        lyr.failed = true;
                        lyr.failed_oom = false;
                        lyr.failed_msg = e.what();
                    }
                }
                bank.last_async_error = true;
                bank.last_async_error_oom = false;
                bank.last_async_error_msg = e.what();
                streamer_error_ = true;
            }
            lock.unlock();
            cv_ready_.notify_all();
            continue;
        }

        if (!swap_reap_queue_.empty()) {
            const SwapReapJob job = swap_reap_queue_.front();
            swap_reap_queue_.pop_front();
            if (job.region_id < 0 || static_cast<size_t>(job.region_id) >= kv_regions_.size()) continue;
            auto& reg = kv_regions_[static_cast<size_t>(job.region_id)];
            if (!reg.active || reg.generation != job.generation || reg.state != KvRegionState::SWAP_OUT_QUEUED) {
                continue;
            }
            void* ev = reg.swap_event;
            reg.state = KvRegionState::SWAP_OUT_FINISHING;
            const uint64_t fin_gen = reg.generation;
            lock.unlock();

            const bool sync_ok = (!ev || atexit_teardown_.load(std::memory_order_acquire))
                                     ? true
                                     : backend_->sync_event_noexcept(ev);

            lock.lock();
            if (stop_streamer_) {
                lock.unlock();
                cv_ready_.notify_all();
                return;
            }
            auto& reg_after = kv_regions_[static_cast<size_t>(job.region_id)];
            if (reg_after.active && reg_after.generation == fin_gen &&
                reg_after.state == KvRegionState::SWAP_OUT_FINISHING) {
                if (sync_ok) {
                    try {
                        backend_->ensure_context();
                        const size_t n = reg_after.handles.size();
                        if (n > 0) {
                            backend_->unmap_range(
                                reg_after.va_base, n * min_gran_, /*was_host_accessible=*/true);
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

// ==================== Active Device Registry ====================

void TriPoolVMM::register_active_device_instance(int device, const std::shared_ptr<TriPoolVMM>& inst) {
    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mu);
    reg.by_device[device] = inst;
}

void TriPoolVMM::unregister_active_device_instance(int device, const TriPoolVMM* inst) {
    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mu);
    auto it = reg.by_device.find(device);
    if (it != reg.by_device.end()) {
        auto sp = it->second.lock();
        if (!sp || sp.get() == inst) {
            reg.by_device.erase(it);
        }
    }
}

std::shared_ptr<TriPoolVMM> TriPoolVMM::active_device_instance(int device) {
    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mu);
    auto it = reg.by_device.find(device);
    if (it != reg.by_device.end()) {
        if (auto sp = it->second.lock()) return sp;
    }
    return nullptr;
}

}  // namespace morphserve::vmm
