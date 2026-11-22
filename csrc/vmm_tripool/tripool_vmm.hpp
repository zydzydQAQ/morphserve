#pragma once

#include <cstddef>
#include <cstdint>
#include <sys/types.h>

extern "C" {

struct UnifiedVmmStats {
    uint64_t min_granularity_bytes;
    uint64_t super_handle_bytes;

    // Physical page pool inventory
    uint64_t super_pool_total_handles;
    uint64_t super_pool_free_handles;
    uint64_t fine_pool_total_pages;
    uint64_t fine_pool_free_pages;

    // Arena 1: Persistent KV (2 MiB pages)
    uint64_t kv_mapped_pages_current;
    uint64_t kv_mapped_pages_peak;
    uint64_t super_to_fine_carves;

    // Arena 2: Transient Scratchpad (64 MiB super-handles)
    uint64_t transient_mapped_handles_current;
    uint64_t transient_mapped_handles_peak;
    uint64_t transient_live_allocs;
    uint64_t transient_alloc_calls;
    uint64_t transient_free_calls;
    uint64_t transient_scope_flushes;

    // Arena 3: Opportunistic Expert Cache (64 MiB super-handles)
    uint64_t expert_mapped_handles_current;
    uint64_t expert_resident_layers_total;
    uint64_t expert_prefetch_layers_completed;
    uint64_t expert_wait_calls;
    uint64_t expert_wait_stalls;
    uint64_t expert_manual_evictions;
    uint64_t expert_auto_evictions_for_transient;
    uint64_t expert_auto_evictions_for_kv;
    uint64_t expert_h2d_bytes_total;

    // Overflow & timings
    uint64_t overflow_creates;
    double transient_map_us;
    double transient_unmap_us;
    double expert_map_us;
    double expert_unmap_us;
    double expert_h2d_submit_us;
};

// Initialize unified TriPool-VMM on `device`.
// - super_pool_mib: pre-allocated physical pool in super_handle_mib units (shared by Transient + Expert)
// - fine_kv_pool_mib: pre-allocated 2 MiB physical pages for Persistent KV
// - super_handle_mib: super-handle granularity (recommended: 64 MiB, aligns 7x64=448 MiB to BAGEL's 444.52 MiB/layer)
// - transient_va_gib: reserved VA for Transient Scratchpad (e.g. 16 GiB)
// - kv_va_gib: reserved VA for Persistent KV (e.g. 16 GiB)
// - allow_overflow: if 0, strictly cap physical memory to the pre-allocated pool (proves zero-OOM pressure eviction!)
int tripool_vmm_init(
    int device,
    size_t super_pool_mib,
    size_t fine_kv_pool_mib,
    size_t super_handle_mib,
    size_t transient_va_gib,
    size_t kv_va_gib,
    int allow_overflow);

int tripool_vmm_shutdown();
void tripool_vmm_get_stats(UnifiedVmmStats* out);

// PyTorch CUDAPluggableAllocator ABI (Arena 2: Transient Scratchpad)
void* tripool_alloc(ssize_t size, int device, void* stream);
void tripool_free(void* ptr, ssize_t size, int device, void* stream);
void tripool_scope_begin();
int tripool_scope_end(void* stream);

// Arena 1: Persistent KV 2 MiB Paged Pool & Contiguous-VA Growable Regions
uint64_t tripool_kv_va_base();
int32_t tripool_kv_alloc_page();
int tripool_kv_free_page(int32_t slot_idx, void* stream);

int32_t tripool_kv_region_alloc(size_t max_va_bytes);
uint64_t tripool_kv_region_ensure(int32_t region_id, size_t needed_bytes, void* stream);
uint64_t tripool_kv_region_va(int32_t region_id);
size_t tripool_kv_region_mapped_bytes(int32_t region_id);
int tripool_kv_region_free(int32_t region_id, void* stream);

// Arena 3: Opportunistic Expert Cache & Background Async H2D Streamer
// Registers an expert with `num_layers`, each occupying `layer_stride_mib` (rounded to super_handle_mib).
// `host_pinned_base` points to a contiguous pinned buffer of size `num_layers * aligned_stride`.
// Returns expert_id >= 0 on success, < 0 on error.
int tripool_expert_register(
    int num_layers,
    size_t layer_stride_mib,
    void* host_pinned_base,
    size_t host_bytes);

uint64_t tripool_expert_va_base(int expert_id);
uint64_t tripool_expert_layer_va(int expert_id, int layer_idx);
size_t tripool_expert_layer_stride_bytes(int expert_id);

int tripool_expert_set_pinned(int expert_id, int pinned);
int tripool_expert_set_layer_used_bytes(int expert_id, int layer_idx, size_t used_bytes);
int tripool_expert_load_layers_sync(int expert_id, int start_layer, int end_layer_exclusive);
int tripool_expert_prefetch_async(int expert_id, int start_layer, int end_layer_exclusive);
int tripool_expert_wait_layer(int expert_id, int layer_idx, void* compute_stream);
int tripool_expert_evict_layer(int expert_id, int layer_idx, void* stream);
int tripool_expert_evict_range(int expert_id, int start_layer, int end_layer_exclusive, void* stream);
int tripool_expert_is_resident(int expert_id, int layer_idx);
int tripool_expert_resident_count(int expert_id);

// Helper to pin/unpin host memory (e.g., memfd mmap) via CUDA Driver API
int tripool_host_register(void* ptr, size_t nbytes);
int tripool_host_unregister(void* ptr);

}  // extern "C"
