#pragma once

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

// Must match `morphserve::vmm::UnifiedVmmStatsPod` layout byte-for-byte.
typedef struct UnifiedVmmStatsC {
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
} UnifiedVmmStatsC;

typedef struct MsVmmConfigC {
    int device;
    size_t super_pool_mib;
    size_t fine_kv_pool_mib;
    size_t super_handle_mib;
    size_t transient_va_gib;
    size_t kv_va_gib;
    int allow_overflow;
    int simulated;
    int materialize_sim_va;
    double kv_swap_high_watermark;
    double kv_swap_low_watermark;
    double kv_swap_min_idle_s;
    uint64_t max_pool_bytes;
} MsVmmConfigC;

typedef struct ms_vmm_handle ms_vmm_t;
typedef struct ms_mirror_handle ms_mirror_t;
typedef struct ms_pacer_handle ms_pacer_t;
typedef struct ms_transport_handle ms_transport_t;
typedef struct ms_mem_bus_handle ms_mem_bus_t;
typedef struct ms_edf_handle ms_edf_t;
typedef struct ms_router_handle ms_router_t;
typedef struct ms_ctrl_handle ms_ctrl_t;
typedef struct ms_ledger_handle ms_ledger_t;

// Error handling & heap string release
const char* ms_last_error(void);
void ms_free_cstr(char* p);

// --- Instance-owned TriPoolVMM C API ---
ms_vmm_t* ms_vmm_create(const MsVmmConfigC* cfg);
void ms_vmm_destroy(ms_vmm_t* h);
int ms_vmm_shutdown(ms_vmm_t* h);
int ms_vmm_is_simulated(const ms_vmm_t* h);
int ms_vmm_device(const ms_vmm_t* h);
void ms_vmm_get_stats(ms_vmm_t* h, UnifiedVmmStatsC* out);
uint64_t ms_vmm_kv_free_bytes(ms_vmm_t* h, int for_dit, uint64_t extra_driver_free_bytes);
uint64_t ms_vmm_sync_kv_free_bytes(ms_vmm_t* h, int for_dit, uint64_t extra_driver_free_bytes);

void ms_vmm_scope_begin(ms_vmm_t* h);
int ms_vmm_scope_end(ms_vmm_t* h, void* stream);
void* ms_vmm_transient_alloc(ms_vmm_t* h, ssize_t size, void* stream);
void ms_vmm_transient_free(ms_vmm_t* h, void* ptr, ssize_t size, void* stream);

uint64_t ms_vmm_kv_va_base(ms_vmm_t* h);
int32_t ms_vmm_kv_alloc_page(ms_vmm_t* h);
int ms_vmm_kv_free_page(ms_vmm_t* h, int32_t slot_idx, void* stream);

uint64_t ms_vmm_kv_session_create(ms_vmm_t* h);
void ms_vmm_kv_session_destroy(ms_vmm_t* h, uint64_t session_id);
void ms_vmm_kv_session_touch(ms_vmm_t* h, uint64_t session_id);
int ms_vmm_kv_session_mark_active(ms_vmm_t* h, uint64_t session_id, void* stream);
size_t ms_vmm_kv_session_mark_thinking(
    ms_vmm_t* h,
    uint64_t session_id,
    const int32_t* region_ids,
    const size_t* valid_bytes,
    size_t num_regions,
    int trigger_watermark,
    void* stream);
size_t ms_vmm_reclaim_thinking_kv_bytes(
    ms_vmm_t* h,
    size_t target_reclaim_bytes,
    int sync,
    uint64_t exclude_session_id,
    void* stream);
size_t ms_vmm_maybe_evict_watermark(
    ms_vmm_t* h,
    int sync,
    uint64_t exclude_session_id,
    size_t extra_needed_bytes,
    int for_dit,
    void* stream);

int32_t ms_vmm_kv_region_alloc(ms_vmm_t* h, size_t max_va_bytes, uint64_t session_id);
uint64_t ms_vmm_kv_region_ensure(ms_vmm_t* h, int32_t region_id, size_t needed_bytes, void* stream);
uint64_t ms_vmm_kv_region_va(ms_vmm_t* h, int32_t region_id);
size_t ms_vmm_kv_region_mapped_bytes(ms_vmm_t* h, int32_t region_id);
int ms_vmm_kv_region_free(ms_vmm_t* h, int32_t region_id, void* stream);
int ms_vmm_kv_region_set_evictable(
    ms_vmm_t* h,
    int32_t region_id,
    int evictable,
    size_t valid_bytes,
    void* compute_stream);
int ms_vmm_kv_region_swap_out_async(ms_vmm_t* h, int32_t region_id, size_t valid_bytes, void* compute_stream);
int ms_vmm_kv_region_swap_in_async(ms_vmm_t* h, int32_t region_id, void* target_stream);
int ms_vmm_kv_region_is_swapped(ms_vmm_t* h, int32_t region_id);
size_t ms_vmm_kv_region_swapped_bytes(ms_vmm_t* h, int32_t region_id);
int ms_vmm_kv_flush_swaps(ms_vmm_t* h);

int ms_vmm_expert_register(
    ms_vmm_t* h,
    int num_layers,
    size_t layer_stride_mib,
    void* host_pinned_base,
    size_t host_bytes);
uint64_t ms_vmm_expert_va_base(ms_vmm_t* h, int expert_id);
uint64_t ms_vmm_expert_layer_va(ms_vmm_t* h, int expert_id, int layer_idx);
size_t ms_vmm_expert_layer_stride_bytes(ms_vmm_t* h, int expert_id);
int ms_vmm_expert_set_pinned(ms_vmm_t* h, int expert_id, int pinned);
int ms_vmm_expert_set_layer_used_bytes(ms_vmm_t* h, int expert_id, int layer_idx, size_t used_bytes);
int ms_vmm_expert_load_layers_sync(ms_vmm_t* h, int expert_id, int start_layer, int end_layer_exclusive);
int ms_vmm_expert_prefetch_async(ms_vmm_t* h, int expert_id, int start_layer, int end_layer_exclusive);
int ms_vmm_expert_wait_layer(ms_vmm_t* h, int expert_id, int layer_idx, void* compute_stream);
int ms_vmm_expert_sync_h2d(ms_vmm_t* h, int expert_id);
size_t ms_vmm_coalesce_fine_to_super(ms_vmm_t* h, size_t max_super);
int ms_vmm_expert_evict_layer(ms_vmm_t* h, int expert_id, int layer_idx, void* stream);
int ms_vmm_expert_evict_range(ms_vmm_t* h, int expert_id, int start_layer, int end_layer_exclusive, void* stream);
int ms_vmm_expert_is_resident(ms_vmm_t* h, int expert_id, int layer_idx);
int ms_vmm_expert_resident_count(ms_vmm_t* h, int expert_id);

// Direct CUDA driver HtoD / DtoH copy helpers for non-PyTorch raw VMM VA verification
int ms_cuda_memcpy_htod(int device, uint64_t dst_va, const void* src_host, size_t nbytes);
int ms_cuda_memcpy_dtoh(int device, void* dst_host, uint64_t src_va, size_t nbytes);

// --- NumaMemfdMirror C API ---
ms_mirror_t* ms_mirror_create(
    size_t nbytes,
    int numa_node,
    const char* name,
    const void* populate_data,
    size_t populate_len);
ms_mirror_t* ms_mirror_attach_proc_fd(int owner_pid, int owner_fd, size_t nbytes, int numa_node);
void ms_mirror_destroy(ms_mirror_t* m);
void ms_mirror_close(ms_mirror_t* m);
uint64_t ms_mirror_ptr(const ms_mirror_t* m);
size_t ms_mirror_nbytes(const ms_mirror_t* m);
int ms_mirror_fd(const ms_mirror_t* m);
int ms_mirror_numa_node(const ms_mirror_t* m);
uint64_t ms_mirror_inode(const ms_mirror_t* m);
int ms_mirror_numa_bound(const ms_mirror_t* m);
int ms_mirror_is_pinned(const ms_mirror_t* m);
int ms_mirror_pin_cuda(ms_mirror_t* m);
int ms_mirror_pin_cuda_ex(ms_mirror_t* m, int simulated);
void ms_mirror_unpin_cuda(ms_mirror_t* m);
double ms_mirror_stream_h2d_gbps(
    ms_mirror_t* m,
    uint64_t dst_dev_ptr,
    size_t dst_nbytes,
    size_t chunk_bytes,
    void* cuda_stream);
int ms_mirror_write(ms_mirror_t* m, size_t offset, const void* src, size_t len);
int ms_mirror_read(const ms_mirror_t* m, size_t offset, void* dst, size_t len);

// --- HalfNodeH2DPacer C API ---
ms_pacer_t* ms_pacer_create(
    const int* gpu_ids,
    const int* group_ids,
    size_t count,
    int max_concurrent_per_half,
    const char* lock_dir);
void ms_pacer_destroy(ms_pacer_t* p);
int ms_pacer_max_concurrent(const ms_pacer_t* p);
int ms_pacer_group_of(const ms_pacer_t* p, int gpu_id);
double ms_pacer_predicted_wait_ms(const ms_pacer_t* p, int gpu_id, double now_ms);
double ms_pacer_reserve_slot(ms_pacer_t* p, int gpu_id, double duration_ms, double now_ms);
void ms_pacer_reset_slots(ms_pacer_t* p);
void ms_pacer_acquire_thread(ms_pacer_t* p, int gpu_id);
void ms_pacer_release_thread(ms_pacer_t* p, int gpu_id);
int ms_pacer_acquire_process_flock(ms_pacer_t* p, int gpu_id, int* out_group_id);
void ms_pacer_release_process_flock(int fd);

// --- PeerByteTransport (`NcclPeerAdapter`, `TcpPeerAdapter`, `InMemoryPeerAdapter`) C API ---
int ms_find_free_tcp_port(void);
int ms_nccl_generate_uid(const char* hint_lib_path, uint8_t out_uid_128[128]);
ms_transport_t* ms_nccl_create(
    int rank,
    int world_size,
    const uint8_t uid_128[128],
    int device_index,
    const char* hint_lib_path);
uint64_t ms_nccl_transfer_stream(const ms_transport_t* t);

ms_transport_t* ms_tcp_create(int rank, int world_size, const char* bind_host);
int ms_tcp_bound_port(const ms_transport_t* t);
int ms_tcp_set_peer(ms_transport_t* t, int peer_rank, const char* host, int port);

ms_mem_bus_t* ms_mem_bus_create(int world_size);
void ms_mem_bus_destroy(ms_mem_bus_t* b);
ms_transport_t* ms_mem_bus_endpoint(ms_mem_bus_t* b, int rank);
int ms_mem_bus_put_shm(ms_mem_bus_t* b, const char* name, const void* data, size_t nbytes);
size_t ms_mem_bus_pop_shm_size(ms_mem_bus_t* b, const char* name, uint8_t** out_buf);
void ms_free_bytes(uint8_t* p);

double ms_transport_send(
    ms_transport_t* t,
    const void* src_ptr,
    size_t nbytes,
    int dst_rank,
    void* cuda_stream,
    int sync_stream);
int ms_transport_recv(
    ms_transport_t* t,
    void* dst_ptr,
    size_t nbytes,
    int src_rank,
    void* cuda_stream,
    int sync_stream,
    size_t* out_actual_bytes,
    double* out_elapsed_ms);
void ms_transport_close(ms_transport_t* t);
void ms_transport_destroy(ms_transport_t* t);

// --- PreemptEdfController C API ---
ms_edf_t* ms_edf_create(
    int num_layers,
    double tpot_slo_ms,
    double guard_ms,
    double init_layer_ms,
    double init_ar_step_ms,
    int max_microbatch_cap);
void ms_edf_destroy(ms_edf_t* e);
void ms_edf_set_state(
    ms_edf_t* e,
    double tpot_slo_ms,
    double guard_ms,
    double ema_layer_ms,
    double ema_ar_step_ms,
    int max_microbatch_cap);
void ms_edf_get_emas(const ms_edf_t* e, double* out_ema_layer_ms, double* out_ema_ar_step_ms);
int ms_edf_adaptive_chunk_layers(const ms_edf_t* e, double override_tpot_slo_ms);
double ms_edf_record_layer(ms_edf_t* e, double layer_dt_ms);
double ms_edf_record_ar_batch(ms_edf_t* e, double batch_dt_ms);
int ms_edf_should_yield(const ms_edf_t* e, double now_s, double earliest_deadline_s, double next_unit_ms);
int ms_edf_select_microbatch(
    const ms_edf_t* e,
    double now_s,
    const double* deadlines_s,
    size_t num_tasks,
    int force_all_urgent,
    size_t* out_due_indices,
    size_t* out_due_count,
    size_t* out_deferred_indices,
    size_t* out_deferred_count);

// --- Control Plane, Router, BudgetPlanner, Ledger & Simulator JSON/POD C API ---
char* ms_policy_from_preset_json(const char* policy_name, int num_gpus, const char* overrides_json);
char* ms_topology_a100_default_json(void);
char* ms_topology_multinode_json(
    int world_size,
    int gpus_per_node,
    double intra_bw,
    double socket_bw,
    double inter_bw,
    double solo_h2d_bw,
    double agg_h2d_bw,
    const char* gpu_name,
    double intra_handoff_ms,
    double inter_handoff_ms);
char* ms_topology_from_job0c_json(const char* path);
char* ms_physics_from_phase01_json(const char* results_dir);

char* ms_budget_internvlu_plan_json(const char* req_json);
char* ms_budget_internvlu_quota_overrides_json(
    const char* policy_name,
    int num_gpus,
    const char* vendor_dir,
    double constrained_gib,
    double total_gpu_gib);
char* ms_budget_verify_internvlu_quota_json(
    const char* run_result_json,
    const char* policy_json,
    double constrained_gib);

ms_router_t* ms_router_create(const char* topology_json, ms_pacer_t* opt_pacer, const char* opts_json);
void ms_router_destroy(ms_router_t* r);
char* ms_router_init_workers_json(ms_router_t* r, const char* init_json);
char* ms_router_get_workers_json(const ms_router_t* r);
void ms_router_set_worker_fields(
    ms_router_t* r,
    int gpu_id,
    int has_alpha,
    double alpha,
    int has_qwait,
    double qwait_ms,
    int has_h2d,
    int active_h2d,
    int has_kv,
    int64_t kv_free_bytes,
    int has_p2p,
    int p2p_links);
void ms_router_update_telemetry(
    ms_router_t* r,
    int gpu_id,
    int has_alpha,
    double alpha,
    int has_kv,
    int64_t kv_free_bytes,
    int has_seq,
    int64_t telemetry_seq);
int ms_router_select_initial_gpu(ms_router_t* r, const char* req_json);
int ms_router_select_static_gen_gpu(const ms_router_t* r, const char* req_json);
int ms_router_select_oom_retry_gpu(const ms_router_t* r, const char* req_json);
char* ms_router_decide_json(const ms_router_t* r, const char* req_json);
char* ms_router_plan_and_reserve_json(ms_router_t* r, const char* req_json);
char* ms_router_route_batch_json(ms_router_t* r, const char* req_json);
void ms_router_reserve_kv_bytes(ms_router_t* r, int gpu_id, int64_t reserved_kv_bytes);
void ms_router_release_kv_reservation(
    ms_router_t* r,
    int gpu_id,
    int64_t reserved_kv_bytes,
    int has_latest_kv,
    int64_t latest_kv_free_bytes,
    int has_seq,
    int64_t telemetry_seq);
void ms_router_release_dit_reservation(ms_router_t* r, int gpu_id, double est_dit_ms);
double ms_router_crossover_kv_mib(
    const ms_router_t* r,
    double alpha,
    int64_t gen_expert_bytes,
    double dit_step1_ms,
    int src_gpu,
    int dst_gpu);

double ms_ctrl_deterministic_think_s(const char* session_id, int turn_idx, double lo_s, double hi_s);
char* ms_ctrl_aggregate_report_json(
    const char* policy_json,
    const char* suite_name,
    double offered_qps,
    double wall_seconds,
    double host_shmem_delta_gib,
    const char* outcomes_json,
    const char* worker_stats_json);

ms_ledger_t* ms_ledger_create(const char* init_json);
void ms_ledger_destroy(ms_ledger_t* l);
int ms_ledger_upsert_kv(ms_ledger_t* l, int sid, double logical_kv_mib);
double ms_ledger_ensure_resident(ms_ledger_t* l, int sid);
double ms_ledger_evict_idle_kv(ms_ledger_t* l, double needed_free_mib, int has_excl, int exclude_sid);
int ms_ledger_exec_dit_vae(
    ms_ledger_t* l,
    int height,
    int has_active_sid,
    int active_sid,
    double* out_swapped_mib);
void ms_ledger_release_session(ms_ledger_t* l, int sid);
void ms_ledger_set_peak_used_mib(ms_ledger_t* l, double v);
char* ms_ledger_stats_json(const ms_ledger_t* l);

char* ms_sim_point_json(const char* req_json);
char* ms_sim_evaluate_real_json(const char* req_json);
char* ms_sim_h100_study_json(const char* req_json);
char* ms_sim_sweep_json(const char* req_json);

// --- Legacy `tripool_*` C ABI (`CUDAPluggableAllocator` & Phase-0/1 binary compatibility) ---
typedef struct TripoolPhase0StatsC {
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
} TripoolPhase0StatsC;

int tripool_init_ex(int device, size_t pool_mib, size_t va_gib, size_t handle_mib);
void tripool_get_stats(TripoolPhase0StatsC* out);
int tripool_shutdown(void);

int tripool_vmm_init(
    int device,
    size_t super_pool_mib,
    size_t fine_kv_pool_mib,
    size_t super_handle_mib,
    size_t transient_va_gib,
    size_t kv_va_gib,
    int allow_overflow);
int tripool_vmm_shutdown(void);
void tripool_vmm_get_stats(UnifiedVmmStatsC* out);

int tripool_host_register(void* ptr, size_t bytes);
int tripool_host_unregister(void* ptr);
double tripool_host_h2d_bench_gbps(const void* host_ptr, int device, size_t total_bytes, int iters);

void tripool_scope_begin(void);
int tripool_scope_end(void* stream);
void* tripool_alloc(ssize_t size, int device, void* stream);
void tripool_free(void* ptr, ssize_t size, int device, void* stream);

uint64_t tripool_kv_va_base(void);
uint64_t tripool_kv_free_bytes(int for_dit);
int32_t tripool_kv_alloc_page(void);
int tripool_kv_free_page(int32_t slot_idx, void* stream);

int32_t tripool_kv_region_alloc(size_t max_va_bytes);
uint64_t tripool_kv_region_ensure(int32_t region_id, size_t needed_bytes, void* stream);
uint64_t tripool_kv_region_va(int32_t region_id);
size_t tripool_kv_region_mapped_bytes(int32_t region_id);
int tripool_kv_region_free(int32_t region_id, void* stream);

int tripool_kv_region_set_evictable(int32_t region_id, int evictable, size_t valid_bytes, void* compute_stream);
int tripool_kv_region_swap_out_async(int32_t region_id, size_t valid_bytes, void* compute_stream);
int tripool_kv_region_swap_in_async(int32_t region_id, void* target_stream);
int tripool_kv_region_is_swapped(int32_t region_id);
size_t tripool_kv_region_swapped_bytes(int32_t region_id);
int tripool_kv_flush_swaps(void);

int tripool_expert_register(int num_layers, size_t layer_stride_mib, void* host_pinned_base, size_t host_bytes);
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

#ifdef __cplusplus
}  // extern "C"
#endif
