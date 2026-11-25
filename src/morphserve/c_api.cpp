#include "morphserve/c_api.h"

#include "morphserve/control/budget_planner.hpp"
#include "morphserve/control/morph_controller.hpp"
#include "morphserve/control/policy.hpp"
#include "morphserve/control/preempt_edf.hpp"
#include "morphserve/control/router.hpp"
#include "morphserve/control/simulator.hpp"
#include "morphserve/control/topology.hpp"
#include "morphserve/runtime/cuda_driver.hpp"
#include "morphserve/runtime/h2d_pacer.hpp"
#include "morphserve/runtime/numa_mirror.hpp"
#include "morphserve/runtime/peer_transport.hpp"
#include "morphserve/vmm/tripool_vmm.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct ms_vmm_handle {
    std::shared_ptr<morphserve::vmm::TriPoolVMM> ptr;
};

struct ms_mirror_handle {
    std::unique_ptr<morphserve::runtime::NumaMemfdMirror> ptr;
};

struct ms_pacer_handle {
    std::shared_ptr<morphserve::runtime::HalfNodeH2DPacer> ptr;
};

struct ms_transport_handle {
    std::unique_ptr<morphserve::runtime::PeerByteTransport> ptr;
};

struct ms_mem_bus_handle {
    std::shared_ptr<morphserve::runtime::InMemoryTransportBus> ptr;
};

struct ms_edf_handle {
    morphserve::control::PreemptEdfController ctrl;
};

struct ms_router_handle {
    std::shared_ptr<morphserve::control::DualPathMorphRouter> ptr;
};

struct ms_ledger_handle {
    std::unique_ptr<morphserve::control::SimGpuMemoryLedger> ptr;
};

static_assert(
    sizeof(UnifiedVmmStatsC) == sizeof(morphserve::vmm::UnifiedVmmStatsPod),
    "UnifiedVmmStatsC and UnifiedVmmStatsPod must have identical byte size");

namespace {

thread_local std::string g_tls_error;

void set_err(const std::exception& e) noexcept {
    try {
        g_tls_error = e.what();
    } catch (...) {
    }
}

void set_err_msg(const char* msg) noexcept {
    try {
        g_tls_error = msg ? msg : "unknown error";
    } catch (...) {
    }
}

char* dup_json_str(const nlohmann::json& j) {
    const std::string s = j.dump();
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (!out) {
        throw std::bad_alloc();
    }
    std::memcpy(out, s.data(), s.size() + 1);
    return out;
}

std::map<int, morphserve::control::WorkerState> parse_ext_workers(const nlohmann::json& j) {
    std::map<int, morphserve::control::WorkerState> out;
    if (j.is_object()) {
        for (auto it = j.begin(); it != j.end(); ++it) {
            const int gid = std::stoi(it.key());
            auto ws = morphserve::control::WorkerState::from_json(it.value());
            ws.gpu_id = gid;
            out[gid] = std::move(ws);
        }
    }
    return out;
}

std::map<int, bool> parse_bool_map(const nlohmann::json& j, const char* key) {
    std::map<int, bool> out;
    if (j.contains(key) && j[key].is_object()) {
        for (auto it = j[key].begin(); it != j[key].end(); ++it) {
            out[std::stoi(it.key())] = it.value().get<bool>();
        }
    }
    return out;
}

std::map<int, int> parse_int_map(const nlohmann::json& j, const char* key) {
    std::map<int, int> out;
    if (j.contains(key) && j[key].is_object()) {
        for (auto it = j[key].begin(); it != j[key].end(); ++it) {
            out[std::stoi(it.key())] = it.value().get<int>();
        }
    }
    return out;
}

std::shared_ptr<morphserve::vmm::TriPoolVMM> resolve_vmm_for_device(int device) {
    auto inst = morphserve::vmm::TriPoolVMM::active_device_instance(device);
    if (inst) return inst;
    return morphserve::vmm::TriPoolVMM::legacy_singleton();
}

}  // namespace

extern "C" {

const char* ms_last_error(void) {
    return g_tls_error.c_str();
}

void ms_free_cstr(char* p) {
    std::free(p);
}

void ms_free_bytes(uint8_t* p) {
    std::free(p);
}

// --- TriPoolVMM ---

ms_vmm_t* ms_vmm_create(const MsVmmConfigC* cfg) {
    if (!cfg) {
        set_err_msg("ms_vmm_create: null cfg");
        return nullptr;
    }
    try {
        morphserve::vmm::TriPoolVMMConfig c;
        c.device = cfg->device;
        c.super_pool_mib = cfg->super_pool_mib;
        c.fine_kv_pool_mib = cfg->fine_kv_pool_mib;
        c.super_handle_mib = cfg->super_handle_mib;
        c.transient_va_gib = cfg->transient_va_gib;
        c.kv_va_gib = cfg->kv_va_gib;
        c.allow_overflow = (cfg->allow_overflow != 0);
        c.simulated = (cfg->simulated != 0);
        c.materialize_sim_va = (cfg->materialize_sim_va != 0);
        c.kv_swap_high_watermark = (cfg->kv_swap_high_watermark > 0.0) ? cfg->kv_swap_high_watermark : 0.85;
        c.kv_swap_low_watermark = (cfg->kv_swap_low_watermark > 0.0) ? cfg->kv_swap_low_watermark : 0.65;
        c.kv_swap_min_idle_s = cfg->kv_swap_min_idle_s;
        c.max_pool_bytes = cfg->max_pool_bytes;
        auto inst = morphserve::vmm::TriPoolVMM::create(c);
        return new ms_vmm_handle{std::move(inst)};
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

void ms_vmm_destroy(ms_vmm_t* h) {
    delete h;
}

int ms_vmm_shutdown(ms_vmm_t* h) {
    if (!h || !h->ptr) return 0;
    try {
        return h->ptr->shutdown();
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

int ms_vmm_is_simulated(const ms_vmm_t* h) {
    return (h && h->ptr && h->ptr->is_simulated()) ? 1 : 0;
}

int ms_vmm_device(const ms_vmm_t* h) {
    return (h && h->ptr) ? h->ptr->device() : 0;
}

void ms_vmm_get_stats(ms_vmm_t* h, UnifiedVmmStatsC* out) {
    if (!out) return;
    if (!h || !h->ptr) {
        std::memset(out, 0, sizeof(*out));
        return;
    }
    h->ptr->get_stats(reinterpret_cast<morphserve::vmm::UnifiedVmmStatsPod*>(out));
}

uint64_t ms_vmm_kv_free_bytes(ms_vmm_t* h, int for_dit, uint64_t extra_driver_free_bytes) {
    return (h && h->ptr) ? h->ptr->kv_free_bytes(for_dit != 0, extra_driver_free_bytes) : 0;
}

uint64_t ms_vmm_sync_kv_free_bytes(ms_vmm_t* h, int for_dit, uint64_t extra_driver_free_bytes) {
    return (h && h->ptr) ? h->ptr->sync_kv_free_bytes(for_dit != 0, extra_driver_free_bytes) : 0;
}

void ms_vmm_scope_begin(ms_vmm_t* h) {
    if (h && h->ptr) h->ptr->scope_begin();
}

int ms_vmm_scope_end(ms_vmm_t* h, void* stream) {
    if (!h || !h->ptr) return -1;
    try {
        return h->ptr->scope_end(stream);
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

void* ms_vmm_transient_alloc(ms_vmm_t* h, ssize_t size, void* stream) {
    if (!h || !h->ptr) return nullptr;
    try {
        return h->ptr->transient_alloc(size, stream);
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

void ms_vmm_transient_free(ms_vmm_t* h, void* ptr, ssize_t size, void* stream) {
    if (h && h->ptr) h->ptr->transient_free(ptr, size, stream);
}

uint64_t ms_vmm_kv_va_base(ms_vmm_t* h) {
    return (h && h->ptr) ? h->ptr->kv_va_base() : 0;
}

int32_t ms_vmm_kv_alloc_page(ms_vmm_t* h) {
    if (!h || !h->ptr) return -1;
    try {
        return h->ptr->kv_alloc_page();
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

int ms_vmm_kv_free_page(ms_vmm_t* h, int32_t slot_idx, void* stream) {
    if (!h || !h->ptr) return -1;
    try {
        return h->ptr->kv_free_page(slot_idx, stream);
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

uint64_t ms_vmm_kv_session_create(ms_vmm_t* h) {
    return (h && h->ptr) ? h->ptr->kv_session_create() : 0;
}

void ms_vmm_kv_session_destroy(ms_vmm_t* h, uint64_t session_id) {
    if (h && h->ptr) h->ptr->kv_session_destroy(session_id);
}

void ms_vmm_kv_session_touch(ms_vmm_t* h, uint64_t session_id) {
    if (h && h->ptr) h->ptr->kv_session_touch(session_id);
}

int ms_vmm_kv_session_mark_active(ms_vmm_t* h, uint64_t session_id, void* stream) {
    if (!h || !h->ptr) return -1;
    try {
        return h->ptr->kv_session_mark_active(session_id, stream);
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

size_t ms_vmm_kv_session_mark_thinking(
    ms_vmm_t* h,
    uint64_t session_id,
    const int32_t* region_ids,
    const size_t* valid_bytes,
    size_t num_regions,
    int trigger_watermark,
    void* stream) {
    if (!h || !h->ptr) return 0;
    try {
        return h->ptr->kv_session_mark_thinking(
            session_id, region_ids, valid_bytes, num_regions, trigger_watermark != 0, stream);
    } catch (const std::exception& e) {
        set_err(e);
        return 0;
    }
}

size_t ms_vmm_reclaim_thinking_kv_bytes(
    ms_vmm_t* h,
    size_t target_reclaim_bytes,
    int sync,
    uint64_t exclude_session_id,
    void* stream) {
    if (!h || !h->ptr) return 0;
    try {
        return h->ptr->reclaim_thinking_kv_bytes(target_reclaim_bytes, sync != 0, exclude_session_id, stream);
    } catch (const std::exception& e) {
        set_err(e);
        return 0;
    }
}

size_t ms_vmm_maybe_evict_watermark(
    ms_vmm_t* h,
    int sync,
    uint64_t exclude_session_id,
    size_t extra_needed_bytes,
    int for_dit,
    void* stream) {
    if (!h || !h->ptr) return 0;
    try {
        return h->ptr->maybe_evict_thinking_kv_on_high_watermark(
            sync != 0, exclude_session_id, extra_needed_bytes, for_dit != 0, stream);
    } catch (const std::exception& e) {
        set_err(e);
        return 0;
    }
}

int32_t ms_vmm_kv_region_alloc(ms_vmm_t* h, size_t max_va_bytes, uint64_t session_id) {
    if (!h || !h->ptr) return -1;
    try {
        return h->ptr->kv_region_alloc(max_va_bytes, session_id);
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

uint64_t ms_vmm_kv_region_ensure(ms_vmm_t* h, int32_t region_id, size_t needed_bytes, void* stream) {
    if (!h || !h->ptr) return 0;
    try {
        return h->ptr->kv_region_ensure(region_id, needed_bytes, stream);
    } catch (const std::exception& e) {
        set_err(e);
        return 0;
    }
}

uint64_t ms_vmm_kv_region_va(ms_vmm_t* h, int32_t region_id) {
    return (h && h->ptr) ? h->ptr->kv_region_va(region_id) : 0;
}

size_t ms_vmm_kv_region_mapped_bytes(ms_vmm_t* h, int32_t region_id) {
    return (h && h->ptr) ? h->ptr->kv_region_mapped_bytes(region_id) : 0;
}

int ms_vmm_kv_region_free(ms_vmm_t* h, int32_t region_id, void* stream) {
    if (!h || !h->ptr) return -1;
    try {
        return h->ptr->kv_region_free(region_id, stream);
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

int ms_vmm_kv_region_set_evictable(
    ms_vmm_t* h,
    int32_t region_id,
    int evictable,
    size_t valid_bytes,
    void* compute_stream) {
    return (h && h->ptr) ? h->ptr->kv_region_set_evictable(region_id, evictable, valid_bytes, compute_stream) : -1;
}

int ms_vmm_kv_region_swap_out_async(
    ms_vmm_t* h,
    int32_t region_id,
    size_t valid_bytes,
    void* compute_stream) {
    if (!h || !h->ptr) return -1;
    try {
        return h->ptr->kv_region_swap_out_async(region_id, valid_bytes, compute_stream);
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

int ms_vmm_kv_region_swap_in_async(ms_vmm_t* h, int32_t region_id, void* target_stream) {
    if (!h || !h->ptr) return -1;
    try {
        return h->ptr->kv_region_swap_in_async(region_id, target_stream);
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

int ms_vmm_kv_region_is_swapped(ms_vmm_t* h, int32_t region_id) {
    return (h && h->ptr) ? h->ptr->kv_region_is_swapped(region_id) : 0;
}

size_t ms_vmm_kv_region_swapped_bytes(ms_vmm_t* h, int32_t region_id) {
    return (h && h->ptr) ? h->ptr->kv_region_swapped_bytes(region_id) : 0;
}

int ms_vmm_kv_flush_swaps(ms_vmm_t* h) {
    if (!h || !h->ptr) return -1;
    try {
        return h->ptr->kv_flush_swaps();
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

int ms_vmm_expert_register(
    ms_vmm_t* h,
    int num_layers,
    size_t layer_stride_mib,
    void* host_pinned_base,
    size_t host_bytes) {
    if (!h || !h->ptr) return -1;
    try {
        return h->ptr->expert_register(num_layers, layer_stride_mib, host_pinned_base, host_bytes);
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

uint64_t ms_vmm_expert_va_base(ms_vmm_t* h, int expert_id) {
    return (h && h->ptr) ? h->ptr->expert_va_base(expert_id) : 0;
}

uint64_t ms_vmm_expert_layer_va(ms_vmm_t* h, int expert_id, int layer_idx) {
    return (h && h->ptr) ? h->ptr->expert_layer_va(expert_id, layer_idx) : 0;
}

size_t ms_vmm_expert_layer_stride_bytes(ms_vmm_t* h, int expert_id) {
    return (h && h->ptr) ? h->ptr->expert_layer_stride_bytes(expert_id) : 0;
}

int ms_vmm_expert_set_pinned(ms_vmm_t* h, int expert_id, int pinned) {
    return (h && h->ptr) ? h->ptr->expert_set_pinned(expert_id, pinned) : -1;
}

int ms_vmm_expert_set_layer_used_bytes(ms_vmm_t* h, int expert_id, int layer_idx, size_t used_bytes) {
    return (h && h->ptr) ? h->ptr->expert_set_layer_used_bytes(expert_id, layer_idx, used_bytes) : -1;
}

int ms_vmm_expert_load_layers_sync(ms_vmm_t* h, int expert_id, int start_layer, int end_layer_exclusive) {
    if (!h || !h->ptr) return -1;
    try {
        return h->ptr->expert_load_layers_sync(expert_id, start_layer, end_layer_exclusive);
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

int ms_vmm_expert_prefetch_async(ms_vmm_t* h, int expert_id, int start_layer, int end_layer_exclusive) {
    return (h && h->ptr) ? h->ptr->expert_prefetch_async(expert_id, start_layer, end_layer_exclusive) : -1;
}

int ms_vmm_expert_wait_layer(ms_vmm_t* h, int expert_id, int layer_idx, void* compute_stream) {
    if (!h || !h->ptr) return -1;
    try {
        return h->ptr->expert_wait_layer(expert_id, layer_idx, compute_stream);
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

int ms_vmm_expert_sync_h2d(ms_vmm_t* h, int expert_id) {
    if (!h || !h->ptr) return -1;
    try {
        return h->ptr->expert_sync_h2d(expert_id);
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

size_t ms_vmm_coalesce_fine_to_super(ms_vmm_t* h, size_t max_super) {
    if (!h || !h->ptr) return 0;
    try {
        return h->ptr->coalesce_fine_to_super(max_super == 0 ? SIZE_MAX : max_super);
    } catch (const std::exception& e) {
        set_err(e);
        return 0;
    }
}

int ms_vmm_expert_evict_layer(ms_vmm_t* h, int expert_id, int layer_idx, void* stream) {
    if (!h || !h->ptr) return -1;
    try {
        return h->ptr->expert_evict_layer(expert_id, layer_idx, stream);
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

int ms_vmm_expert_evict_range(ms_vmm_t* h, int expert_id, int start_layer, int end_layer_exclusive, void* stream) {
    if (!h || !h->ptr) return -1;
    try {
        return h->ptr->expert_evict_range(expert_id, start_layer, end_layer_exclusive, stream);
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

int ms_vmm_expert_is_resident(ms_vmm_t* h, int expert_id, int layer_idx) {
    return (h && h->ptr) ? h->ptr->expert_is_resident(expert_id, layer_idx) : 0;
}

int ms_vmm_expert_resident_count(ms_vmm_t* h, int expert_id) {
    return (h && h->ptr) ? h->ptr->expert_resident_count(expert_id) : 0;
}

int ms_cuda_memcpy_htod(int device, uint64_t dst_va, const void* src_host, size_t nbytes) {
    if (nbytes == 0) return 0;
    try {
        auto& drv = morphserve::runtime::CudaDriver::instance();
        if (drv.cuInit(0) != CUDA_SUCCESS) return -1;
        CUdevice dev = 0;
        CUcontext ctx = nullptr;
        if (drv.cuDeviceGet(&dev, device) != CUDA_SUCCESS) return -1;
        if (drv.cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) return -1;
        drv.cuCtxSetCurrent(ctx);
        const CUresult rc = drv.cuMemcpyHtoD(static_cast<CUdeviceptr>(dst_va), src_host, nbytes);
        drv.cuDevicePrimaryCtxRelease(dev);
        return (rc == CUDA_SUCCESS) ? 0 : -1;
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

int ms_cuda_memcpy_dtoh(int device, void* dst_host, uint64_t src_va, size_t nbytes) {
    if (nbytes == 0) return 0;
    try {
        auto& drv = morphserve::runtime::CudaDriver::instance();
        if (drv.cuInit(0) != CUDA_SUCCESS) return -1;
        CUdevice dev = 0;
        CUcontext ctx = nullptr;
        if (drv.cuDeviceGet(&dev, device) != CUDA_SUCCESS) return -1;
        if (drv.cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) return -1;
        drv.cuCtxSetCurrent(ctx);
        const CUresult rc = drv.cuMemcpyDtoH(dst_host, static_cast<CUdeviceptr>(src_va), nbytes);
        drv.cuDevicePrimaryCtxRelease(dev);
        return (rc == CUDA_SUCCESS) ? 0 : -1;
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

// --- NumaMemfdMirror ---

ms_mirror_t* ms_mirror_create(
    size_t nbytes,
    int numa_node,
    const char* name,
    const void* populate_data,
    size_t populate_len) {
    try {
        auto m = morphserve::runtime::NumaMemfdMirror::create(
            nbytes, numa_node, name ? name : "morphserve_expert", populate_data, populate_len);
        return new ms_mirror_handle{std::move(m)};
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

ms_mirror_t* ms_mirror_attach_proc_fd(int owner_pid, int owner_fd, size_t nbytes, int numa_node) {
    try {
        auto m = morphserve::runtime::NumaMemfdMirror::attach_proc_fd(owner_pid, owner_fd, nbytes, numa_node);
        return new ms_mirror_handle{std::move(m)};
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

void ms_mirror_destroy(ms_mirror_t* m) { delete m; }
void ms_mirror_close(ms_mirror_t* m) { if (m && m->ptr) m->ptr->close(); }
uint64_t ms_mirror_ptr(const ms_mirror_t* m) { return (m && m->ptr) ? m->ptr->ptr_u64() : 0; }
size_t ms_mirror_nbytes(const ms_mirror_t* m) { return (m && m->ptr) ? m->ptr->nbytes() : 0; }
int ms_mirror_fd(const ms_mirror_t* m) { return (m && m->ptr) ? m->ptr->fd() : -1; }
int ms_mirror_numa_node(const ms_mirror_t* m) { return (m && m->ptr) ? m->ptr->numa_node() : -1; }
uint64_t ms_mirror_inode(const ms_mirror_t* m) { return (m && m->ptr) ? m->ptr->inode() : 0; }
int ms_mirror_numa_bound(const ms_mirror_t* m) { return (m && m->ptr && m->ptr->numa_bound()) ? 1 : 0; }
int ms_mirror_is_pinned(const ms_mirror_t* m) { return (m && m->ptr && m->ptr->pinned()) ? 1 : 0; }

int ms_mirror_pin_cuda(ms_mirror_t* m) {
    return ms_mirror_pin_cuda_ex(m, 0);
}

int ms_mirror_pin_cuda_ex(ms_mirror_t* m, int simulated) {
    if (!m || !m->ptr) return -1;
    try {
        m->ptr->pin_cuda(simulated != 0);
        return 0;
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

void ms_mirror_unpin_cuda(ms_mirror_t* m) {
    if (m && m->ptr) m->ptr->unpin_cuda();
}

double ms_mirror_stream_h2d_gbps(
    ms_mirror_t* m,
    uint64_t dst_dev_ptr,
    size_t dst_nbytes,
    size_t chunk_bytes,
    void* cuda_stream) {
    if (!m || !m->ptr) return -1.0;
    try {
        return m->ptr->stream_h2d_chunks_gbps(dst_dev_ptr, dst_nbytes, chunk_bytes, cuda_stream);
    } catch (const std::exception& e) {
        set_err(e);
        return -1.0;
    }
}

int ms_mirror_write(ms_mirror_t* m, size_t offset, const void* src, size_t len) {
    if (!m || !m->ptr) return -1;
    try {
        m->ptr->write_bytes(offset, src, len);
        return 0;
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

int ms_mirror_read(const ms_mirror_t* m, size_t offset, void* dst, size_t len) {
    if (!m || !m->ptr) return -1;
    try {
        m->ptr->read_bytes(offset, dst, len);
        return 0;
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

// --- HalfNodeH2DPacer ---

ms_pacer_t* ms_pacer_create(
    const int* gpu_ids,
    const int* group_ids,
    size_t count,
    int max_concurrent_per_half,
    const char* lock_dir) {
    try {
        std::unordered_map<int, int> gmap;
        for (size_t i = 0; i < count; ++i) {
            gmap[gpu_ids[i]] = group_ids[i];
        }
        auto p = std::make_shared<morphserve::runtime::HalfNodeH2DPacer>(
            std::move(gmap),
            max_concurrent_per_half,
            lock_dir ? lock_dir : "/tmp/morphserve_h2d_locks");
        return new ms_pacer_handle{std::move(p)};
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

void ms_pacer_destroy(ms_pacer_t* p) { delete p; }
int ms_pacer_max_concurrent(const ms_pacer_t* p) {
    return (p && p->ptr) ? p->ptr->max_concurrent_per_half() : 1;
}

int ms_pacer_group_of(const ms_pacer_t* p, int gpu_id) {
    if (!p || !p->ptr) return -1;
    try {
        return p->ptr->group_of(gpu_id);
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

double ms_pacer_predicted_wait_ms(const ms_pacer_t* p, int gpu_id, double now_ms) {
    if (!p || !p->ptr) return 0.0;
    try {
        return p->ptr->predicted_wait_ms(gpu_id, now_ms);
    } catch (const std::exception& e) {
        set_err(e);
        return -1.0;
    }
}

double ms_pacer_reserve_slot(ms_pacer_t* p, int gpu_id, double duration_ms, double now_ms) {
    if (!p || !p->ptr) return 0.0;
    try {
        return p->ptr->reserve_slot(gpu_id, duration_ms, now_ms);
    } catch (const std::exception& e) {
        set_err(e);
        return -1.0;
    }
}

void ms_pacer_reset_slots(ms_pacer_t* p) {
    if (p && p->ptr) p->ptr->reset_analytical_slots();
}

void ms_pacer_acquire_thread(ms_pacer_t* p, int gpu_id) {
    if (p && p->ptr) p->ptr->acquire_thread_slot(gpu_id);
}

void ms_pacer_release_thread(ms_pacer_t* p, int gpu_id) {
    if (p && p->ptr) p->ptr->release_thread_slot(gpu_id);
}

int ms_pacer_acquire_process_flock(ms_pacer_t* p, int gpu_id, int* out_group_id) {
    if (!p || !p->ptr) return -1;
    try {
        return p->ptr->acquire_process_flock(gpu_id, out_group_id);
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

void ms_pacer_release_process_flock(int fd) {
    morphserve::runtime::HalfNodeH2DPacer::release_process_flock(fd);
}

// --- PeerByteTransport ---

int ms_find_free_tcp_port(void) {
    try {
        return morphserve::runtime::find_free_tcp_port();
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

int ms_nccl_generate_uid(const char* hint_lib_path, uint8_t out_uid_128[128]) {
    try {
        const auto v = morphserve::runtime::NcclPeerAdapter::generate_unique_id(hint_lib_path);
        std::memcpy(out_uid_128, v.data(), 128);
        return 0;
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

ms_transport_t* ms_nccl_create(
    int rank,
    int world_size,
    const uint8_t uid_128[128],
    int device_index,
    const char* hint_lib_path) {
    try {
        auto tp = std::make_unique<morphserve::runtime::NcclPeerAdapter>(
            rank, world_size, uid_128, device_index, hint_lib_path);
        return new ms_transport_handle{std::move(tp)};
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

uint64_t ms_nccl_transfer_stream(const ms_transport_t* t) {
    if (!t || !t->ptr) return 0;
    const auto* nccl = dynamic_cast<const morphserve::runtime::NcclPeerAdapter*>(t->ptr.get());
    return nccl ? nccl->transfer_stream_u64() : 0;
}

ms_transport_t* ms_tcp_create(int rank, int world_size, const char* bind_host) {
    try {
        auto tp = std::make_unique<morphserve::runtime::TcpPeerAdapter>(
            rank, world_size, bind_host ? bind_host : "127.0.0.1");
        return new ms_transport_handle{std::move(tp)};
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

int ms_tcp_bound_port(const ms_transport_t* t) {
    if (!t || !t->ptr) return -1;
    const auto* tcp = dynamic_cast<const morphserve::runtime::TcpPeerAdapter*>(t->ptr.get());
    return tcp ? tcp->bound_port() : -1;
}

int ms_tcp_set_peer(ms_transport_t* t, int peer_rank, const char* host, int port) {
    if (!t || !t->ptr || !host) return -1;
    auto* tcp = dynamic_cast<morphserve::runtime::TcpPeerAdapter*>(t->ptr.get());
    if (!tcp) return -1;
    tcp->set_peer_endpoint(peer_rank, host, port);
    return 0;
}

ms_mem_bus_t* ms_mem_bus_create(int world_size) {
    try {
        auto bus = std::make_shared<morphserve::runtime::InMemoryTransportBus>(world_size);
        return new ms_mem_bus_handle{std::move(bus)};
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

void ms_mem_bus_destroy(ms_mem_bus_t* b) { delete b; }

ms_transport_t* ms_mem_bus_endpoint(ms_mem_bus_t* b, int rank) {
    if (!b || !b->ptr) return nullptr;
    try {
        return new ms_transport_handle{b->ptr->create_endpoint(rank)};
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

int ms_mem_bus_put_shm(ms_mem_bus_t* b, const char* name, const void* data, size_t nbytes) {
    if (!b || !b->ptr || !name) return -1;
    try {
        b->ptr->put_shm(name, data, nbytes);
        return 0;
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

size_t ms_mem_bus_pop_shm_size(ms_mem_bus_t* b, const char* name, uint8_t** out_buf) {
    if (out_buf) *out_buf = nullptr;
    if (!b || !b->ptr || !name) return 0;
    try {
        auto vec = b->ptr->pop_shm(name);
        if (vec.empty()) return 0;
        if (out_buf) {
            uint8_t* mem = static_cast<uint8_t*>(std::malloc(vec.size()));
            if (!mem) return 0;
            std::memcpy(mem, vec.data(), vec.size());
            *out_buf = mem;
        }
        return vec.size();
    } catch (const std::exception& e) {
        set_err(e);
        return 0;
    }
}

double ms_transport_send(
    ms_transport_t* t,
    const void* src_ptr,
    size_t nbytes,
    int dst_rank,
    void* cuda_stream,
    int sync_stream) {
    if (!t || !t->ptr) return -1.0;
    try {
        return t->ptr->send_bytes(src_ptr, nbytes, dst_rank, cuda_stream, sync_stream != 0);
    } catch (const std::exception& e) {
        set_err(e);
        return -1.0;
    }
}

int ms_transport_recv(
    ms_transport_t* t,
    void* dst_ptr,
    size_t nbytes,
    int src_rank,
    void* cuda_stream,
    int sync_stream,
    size_t* out_actual_bytes,
    double* out_elapsed_ms) {
    if (!t || !t->ptr) return -1;
    try {
        const auto [actual, ms] = t->ptr->recv_bytes(dst_ptr, nbytes, src_rank, cuda_stream, sync_stream != 0);
        if (out_actual_bytes) *out_actual_bytes = actual;
        if (out_elapsed_ms) *out_elapsed_ms = ms;
        return 0;
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

void ms_transport_close(ms_transport_t* t) { if (t && t->ptr) t->ptr->close(); }
void ms_transport_destroy(ms_transport_t* t) { delete t; }

// --- PreemptEdfController ---

ms_edf_t* ms_edf_create(
    int num_layers,
    double tpot_slo_ms,
    double guard_ms,
    double init_layer_ms,
    double init_ar_step_ms,
    int max_microbatch_cap) {
    try {
        return new ms_edf_handle{morphserve::control::PreemptEdfController(
            num_layers, tpot_slo_ms, guard_ms, init_layer_ms, init_ar_step_ms, max_microbatch_cap)};
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

void ms_edf_destroy(ms_edf_t* e) { delete e; }

void ms_edf_set_state(
    ms_edf_t* e,
    double tpot_slo_ms,
    double guard_ms,
    double ema_layer_ms,
    double ema_ar_step_ms,
    int max_microbatch_cap) {
    if (!e) return;
    e->ctrl.set_tpot_slo_ms(tpot_slo_ms);
    e->ctrl.set_guard_ms(guard_ms);
    e->ctrl.set_ema_layer_ms(ema_layer_ms);
    e->ctrl.set_ema_ar_step_ms(ema_ar_step_ms);
    e->ctrl.set_max_microbatch_cap(max_microbatch_cap);
}

void ms_edf_get_emas(const ms_edf_t* e, double* out_ema_layer_ms, double* out_ema_ar_step_ms) {
    if (!e) return;
    if (out_ema_layer_ms) *out_ema_layer_ms = e->ctrl.ema_layer_ms();
    if (out_ema_ar_step_ms) *out_ema_ar_step_ms = e->ctrl.ema_ar_step_ms();
}

int ms_edf_adaptive_chunk_layers(const ms_edf_t* e, double override_tpot_slo_ms) {
    return e ? e->ctrl.compute_adaptive_chunk_layers(override_tpot_slo_ms) : 1;
}

double ms_edf_record_layer(ms_edf_t* e, double layer_dt_ms) {
    return e ? e->ctrl.record_layer_sample(layer_dt_ms) : 0.0;
}

double ms_edf_record_ar_batch(ms_edf_t* e, double batch_dt_ms) {
    return e ? e->ctrl.record_ar_batch_sample(batch_dt_ms) : 0.0;
}

int ms_edf_should_yield(const ms_edf_t* e, double now_s, double earliest_deadline_s, double next_unit_ms) {
    return (e && e->ctrl.should_yield_now(now_s, earliest_deadline_s, next_unit_ms)) ? 1 : 0;
}

int ms_edf_select_microbatch(
    const ms_edf_t* e,
    double now_s,
    const double* deadlines_s,
    size_t num_tasks,
    int force_all_urgent,
    size_t* out_due_indices,
    size_t* out_due_count,
    size_t* out_deferred_indices,
    size_t* out_deferred_count) {
    if (!e) return -1;
    try {
        std::vector<size_t> due;
        std::vector<size_t> def;
        e->ctrl.select_edf_microbatch(now_s, deadlines_s, num_tasks, force_all_urgent != 0, due, def);
        if (out_due_count) *out_due_count = due.size();
        if (out_deferred_count) *out_deferred_count = def.size();
        if (out_due_indices && !due.empty()) {
            std::memcpy(out_due_indices, due.data(), due.size() * sizeof(size_t));
        }
        if (out_deferred_indices && !def.empty()) {
            std::memcpy(out_deferred_indices, def.data(), def.size() * sizeof(size_t));
        }
        return 0;
    } catch (const std::exception& e_err) {
        set_err(e_err);
        return -1;
    }
}

// --- Control Plane, Router, BudgetPlanner, Ledger & Simulator ---

char* ms_policy_from_preset_json(const char* policy_name, int num_gpus, const char* overrides_json) {
    try {
        nlohmann::json ov = (overrides_json && overrides_json[0])
                                ? nlohmann::json::parse(overrides_json)
                                : nlohmann::json::object();
        const auto cfg = morphserve::control::PolicyConfig::from_preset(
            policy_name ? policy_name : "", num_gpus, ov);
        return dup_json_str(cfg.to_json());
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

char* ms_topology_a100_default_json(void) {
    try {
        return dup_json_str(morphserve::control::ClusterTopologyProfile::a100_pcie_8gpu_default().to_json());
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

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
    double inter_handoff_ms) {
    try {
        const auto t = morphserve::control::ClusterTopologyProfile::from_multinode_measurements(
            world_size,
            gpus_per_node,
            intra_bw,
            socket_bw,
            inter_bw,
            solo_h2d_bw,
            agg_h2d_bw,
            gpu_name ? gpu_name : "NVIDIA A100-PCIE-40GB",
            intra_handoff_ms,
            inter_handoff_ms);
        return dup_json_str(t.to_json());
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

char* ms_topology_from_job0c_json(const char* path) {
    try {
        return dup_json_str(morphserve::control::ClusterTopologyProfile::from_job0c_report(path ? path : "").to_json());
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

char* ms_physics_from_phase01_json(const char* results_dir) {
    try {
        return dup_json_str(
            morphserve::control::ClusterPhysicsProfile::from_phase01_results(results_dir ? results_dir : "").to_json());
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

char* ms_budget_internvlu_plan_json(const char* req_json) {
    try {
        const auto j = nlohmann::json::parse(req_json ? req_json : "{}");
        std::optional<double> frac;
        if (j.contains("gpu_memory_fraction") && j["gpu_memory_fraction"].is_number()) {
            frac = j["gpu_memory_fraction"].get<double>();
        }
        std::optional<double> quota;
        if (j.contains("gpu_memory_quota_mib") && j["gpu_memory_quota_mib"].is_number()) {
            quota = j["gpu_memory_quota_mib"].get<double>();
        }
        const auto plan = morphserve::control::WorkerBudgetPlanner::compute_internvlu_memory_plan(
            j.value("total_dev_mib", 40960.0),
            j.value("free_dev_mib", 0.0),
            j.value("base_pytorch_alloc_mib", 4200.0),
            j.value("mirror_bytes", int64_t{0}),
            j.value("policy_super_pool_mib", 4864),
            j.value("policy_fine_kv_pool_mib", 512),
            j.value("super_handle_mib", 64),
            j.value("allow_vmm_overflow", true),
            frac,
            quota);
        return dup_json_str(plan);
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

char* ms_budget_internvlu_quota_overrides_json(
    const char* policy_name,
    int num_gpus,
    const char* vendor_dir,
    double constrained_gib,
    double total_gpu_gib) {
    try {
        const auto [ov, budget] = morphserve::control::WorkerBudgetPlanner::compute_internvlu_quota_overrides(
            policy_name ? policy_name : "morphserve",
            num_gpus,
            vendor_dir ? vendor_dir : "",
            constrained_gib,
            total_gpu_gib);
        return dup_json_str({{"overrides", ov}, {"budget", budget}});
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

char* ms_budget_verify_internvlu_quota_json(
    const char* run_result_json,
    const char* policy_json,
    double constrained_gib) {
    try {
        const auto rr = nlohmann::json::parse(run_result_json ? run_result_json : "{}");
        const auto pol = morphserve::control::PolicyConfig::from_json(
            nlohmann::json::parse(policy_json ? policy_json : "{}"));
        return dup_json_str(
            morphserve::control::WorkerBudgetPlanner::verify_internvlu_memory_quota(rr, pol, constrained_gib));
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

ms_router_t* ms_router_create(const char* topology_json, ms_pacer_t* opt_pacer, const char* opts_json) {
    try {
        const auto topo = morphserve::control::ClusterTopologyProfile::from_json(
            nlohmann::json::parse(topology_json ? topology_json : "{}"));
        const auto opts = (opts_json && opts_json[0])
                              ? nlohmann::json::parse(opts_json)
                              : nlohmann::json::object();
        std::optional<double> handoff;
        if (opts.contains("state_handoff_ms") && opts["state_handoff_ms"].is_number()) {
            handoff = opts["state_handoff_ms"].get<double>();
        }
        auto router = std::make_shared<morphserve::control::DualPathMorphRouter>(
            topo,
            opt_pacer ? opt_pacer->ptr : nullptr,
            morphserve::control::default_operator_profile(),
            handoff,
            opts.value("map_overhead_ms_full", 12.0),
            opts.value("overlap_efficiency", 0.90),
            opts.value("allow_cross_half_state_ship", true),
            opts.value("allow_inter_node_state_ship", true),
            opts.value("max_inter_node_state_bytes", int64_t{128LL * morphserve::control::kMiB}));
        return new ms_router_handle{std::move(router)};
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

void ms_router_destroy(ms_router_t* r) { delete r; }

char* ms_router_init_workers_json(ms_router_t* r, const char* init_json) {
    if (!r || !r->ptr) return nullptr;
    try {
        const auto j = nlohmann::json::parse(init_json ? init_json : "{}");
        const int n = j.value("num_gpus", 8);
        std::map<int, std::string> roles;
        if (j.contains("roles") && j["roles"].is_object()) {
            for (auto it = j["roles"].begin(); it != j["roles"].end(); ++it) {
                roles[std::stoi(it.key())] = it.value().get<std::string>();
            }
        }
        std::map<int, double> alphas;
        if (j.contains("initial_alphas") && j["initial_alphas"].is_object()) {
            for (auto it = j["initial_alphas"].begin(); it != j["initial_alphas"].end(); ++it) {
                alphas[std::stoi(it.key())] = it.value().get<double>();
            }
        }
        std::map<int, int64_t> kv_frees;
        if (j.contains("kv_free_bytes") && j["kv_free_bytes"].is_object()) {
            for (auto it = j["kv_free_bytes"].begin(); it != j["kv_free_bytes"].end(); ++it) {
                kv_frees[std::stoi(it.key())] = it.value().get<int64_t>();
            }
        }
        const auto ws_map = r->ptr->init_workers(
            n,
            [&](int g) { return roles.count(g) ? roles.at(g) : std::string("hybrid"); },
            alphas,
            kv_frees);
        nlohmann::json out = nlohmann::json::object();
        for (const auto& [gid, ws] : ws_map) {
            out[std::to_string(gid)] = ws.to_json();
        }
        return dup_json_str(out);
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

char* ms_router_get_workers_json(const ms_router_t* r) {
    if (!r || !r->ptr) return nullptr;
    try {
        const auto ws_map = r->ptr->worker_states_snapshot();
        nlohmann::json out = nlohmann::json::object();
        for (const auto& [gid, ws] : ws_map) {
            out[std::to_string(gid)] = ws.to_json();
        }
        return dup_json_str(out);
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

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
    int p2p_links) {
    if (!r || !r->ptr) return;
    r->ptr->set_worker_state_fields(
        gpu_id,
        has_alpha ? std::optional<double>(alpha) : std::nullopt,
        has_qwait ? std::optional<double>(qwait_ms) : std::nullopt,
        has_h2d ? std::optional<int>(active_h2d) : std::nullopt,
        has_kv ? std::optional<int64_t>(kv_free_bytes) : std::nullopt,
        has_p2p ? std::optional<int>(p2p_links) : std::nullopt);
}

void ms_router_update_telemetry(
    ms_router_t* r,
    int gpu_id,
    int has_alpha,
    double alpha,
    int has_kv,
    int64_t kv_free_bytes,
    int has_seq,
    int64_t telemetry_seq) {
    if (!r || !r->ptr) return;
    r->ptr->update_worker_telemetry(
        gpu_id,
        has_alpha ? std::optional<double>(alpha) : std::nullopt,
        has_kv ? std::optional<int64_t>(kv_free_bytes) : std::nullopt,
        has_seq ? std::optional<int64_t>(telemetry_seq) : std::nullopt,
        nullptr);
}

int ms_router_select_initial_gpu(ms_router_t* r, const char* req_json) {
    if (!r || !r->ptr) return -1;
    try {
        const auto j = nlohmann::json::parse(req_json ? req_json : "{}");
        std::map<int, morphserve::control::WorkerState> ext;
        const bool has_ext = j.contains("workers") && j["workers"].is_object();
        if (has_ext) ext = parse_ext_workers(j["workers"]);
        return r->ptr->select_initial_gpu(
            j.value("num_gpus", 8),
            j.value("routing_mode", "dual_path"),
            j.value("preemption_mode", "layer_chunk"),
            j.value("static_und_ranks", std::vector<int>{}),
            j.value("modality_plan", "ar_only"),
            has_ext ? &ext : nullptr,
            parse_bool_map(j, "gpu_busy"),
            parse_int_map(j, "active_dit_ranks"),
            parse_int_map(j, "active_preemptions"),
            j.value("max_preemptions_per_gpu", 2),
            j.value("architecture", std::string("tight_mot")));
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

int ms_router_select_static_gen_gpu(const ms_router_t* r, const char* req_json) {
    if (!r || !r->ptr) return -1;
    try {
        const auto j = nlohmann::json::parse(req_json ? req_json : "{}");
        std::map<int, morphserve::control::WorkerState> ext;
        const bool has_ext = j.contains("workers") && j["workers"].is_object();
        if (has_ext) ext = parse_ext_workers(j["workers"]);
        return r->ptr->select_static_gen_gpu(
            j.value("routing_mode", "static_nccl_fission"),
            j.value("static_und_ranks", std::vector<int>{}),
            j.value("static_gen_ranks", std::vector<int>{}),
            j.value("src_und_gpu", 0),
            has_ext ? &ext : nullptr,
            parse_bool_map(j, "gpu_busy"),
            j.value("required_kv_bytes", int64_t{64LL * morphserve::control::kMiB}));
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

int ms_router_select_oom_retry_gpu(const ms_router_t* r, const char* req_json) {
    if (!r || !r->ptr) return -1;
    try {
        const auto j = nlohmann::json::parse(req_json ? req_json : "{}");
        std::map<int, morphserve::control::WorkerState> ext;
        const bool has_ext = j.contains("workers") && j["workers"].is_object();
        if (has_ext) ext = parse_ext_workers(j["workers"]);
        return r->ptr->select_oom_retry_gpu(
            j.value("failed_gpu", 0),
            j.value("num_gpus", 8),
            j.value("routing_mode", "dual_path"),
            j.value("static_und_ranks", std::vector<int>{}),
            has_ext ? &ext : nullptr,
            parse_bool_map(j, "gpu_busy"));
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

char* ms_router_decide_json(const ms_router_t* r, const char* req_json) {
    if (!r || !r->ptr) return nullptr;
    try {
        const auto j = nlohmann::json::parse(req_json ? req_json : "{}");
        const auto req = morphserve::control::CPTGRequestSpec::from_json(j.at("req"));
        std::map<int, morphserve::control::WorkerState> ext;
        const bool has_ext = j.contains("workers") && j["workers"].is_object();
        if (has_ext) ext = parse_ext_workers(j["workers"]);
        const auto dec = r->ptr->decide(req, has_ext ? &ext : nullptr, j.value("now_ms", 0.0));
        return dup_json_str(dec.to_json());
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

char* ms_router_plan_and_reserve_json(ms_router_t* r, const char* req_json) {
    if (!r || !r->ptr) return nullptr;
    try {
        const auto j = nlohmann::json::parse(req_json ? req_json : "{}");
        const auto req = morphserve::control::CPTGRequestSpec::from_json(j.at("req"));
        std::map<int, morphserve::control::WorkerState> ext;
        const bool has_ext = j.contains("workers") && j["workers"].is_object();
        if (has_ext) ext = parse_ext_workers(j["workers"]);
        const auto [dec, ship_mode] = r->ptr->plan_and_reserve_transition(
            j.value("routing_mode", "dual_path"),
            j.value("static_und_ranks", std::vector<int>{}),
            j.value("static_gen_ranks", std::vector<int>{}),
            req,
            j.value("est_dit_ms", 0.0),
            has_ext ? &ext : nullptr,
            parse_bool_map(j, "gpu_busy"),
            j.value("use_pacer", true),
            j.value("now_ms", 0.0));
        nlohmann::json out = {{"decision", dec.to_json()}, {"ship_mode", ship_mode}};
        if (has_ext) {
            nlohmann::json w_out = nlohmann::json::object();
            for (const auto& [gid, ws] : ext) w_out[std::to_string(gid)] = ws.to_json();
            out["workers"] = w_out;
        }
        return dup_json_str(out);
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

char* ms_router_route_batch_json(ms_router_t* r, const char* req_json) {
    if (!r || !r->ptr) return nullptr;
    try {
        const auto j = nlohmann::json::parse(req_json ? req_json : "{}");
        std::vector<morphserve::control::CPTGRequestSpec> reqs;
        for (const auto& rj : j.at("requests")) {
            reqs.push_back(morphserve::control::CPTGRequestSpec::from_json(rj));
        }
        const auto workers = parse_ext_workers(j.at("workers"));
        const auto decs = r->ptr->route_batch(reqs, workers, j.value("now_ms", 0.0));
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& d : decs) arr.push_back(d.to_json());
        return dup_json_str(arr);
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

void ms_router_reserve_kv_bytes(ms_router_t* r, int gpu_id, int64_t reserved_kv_bytes) {
    if (r && r->ptr) r->ptr->reserve_kv_bytes(gpu_id, reserved_kv_bytes, nullptr);
}

void ms_router_release_kv_reservation(
    ms_router_t* r,
    int gpu_id,
    int64_t reserved_kv_bytes,
    int has_latest_kv,
    int64_t latest_kv_free_bytes,
    int has_seq,
    int64_t telemetry_seq) {
    if (!r || !r->ptr) return;
    r->ptr->release_kv_reservation(
        gpu_id,
        reserved_kv_bytes,
        has_latest_kv ? std::optional<int64_t>(latest_kv_free_bytes) : std::nullopt,
        has_seq ? std::optional<int64_t>(telemetry_seq) : std::nullopt,
        nullptr);
}

void ms_router_release_dit_reservation(ms_router_t* r, int gpu_id, double est_dit_ms) {
    if (r && r->ptr) r->ptr->release_dit_reservation(gpu_id, est_dit_ms, nullptr);
}

double ms_router_crossover_kv_mib(
    const ms_router_t* r,
    double alpha,
    int64_t gen_expert_bytes,
    double dit_step1_ms,
    int src_gpu,
    int dst_gpu) {
    if (!r || !r->ptr) return 0.0;
    try {
        return r->ptr->analytical_crossover_kv_mib(alpha, gen_expert_bytes, dit_step1_ms, src_gpu, dst_gpu);
    } catch (const std::exception& e) {
        set_err(e);
        return -1.0;
    }
}

double ms_ctrl_deterministic_think_s(const char* session_id, int turn_idx, double lo_s, double hi_s) {
    return morphserve::control::MorphControlEngine::deterministic_think_s(
        session_id ? session_id : "", turn_idx, lo_s, hi_s);
}

char* ms_ctrl_aggregate_report_json(
    const char* policy_json,
    const char* suite_name,
    double offered_qps,
    double wall_seconds,
    double host_shmem_delta_gib,
    const char* outcomes_json,
    const char* worker_stats_json) {
    try {
        const auto pol = morphserve::control::PolicyConfig::from_json(
            nlohmann::json::parse(policy_json ? policy_json : "{}"));
        const auto outs = nlohmann::json::parse(outcomes_json ? outcomes_json : "[]");
        const auto wstats = nlohmann::json::parse(worker_stats_json ? worker_stats_json : "[]");
        return dup_json_str(morphserve::control::MorphControlEngine::aggregate_cluster_run_report(
            pol, suite_name ? suite_name : "", offered_qps, wall_seconds, host_shmem_delta_gib, outs, wstats));
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

// --- SimGpuMemoryLedger ---

ms_ledger_t* ms_ledger_create(const char* init_json) {
    try {
        const auto j = nlohmann::json::parse(init_json ? init_json : "{}");
        const auto pol = morphserve::control::PolicyConfig::from_json(j.at("policy"));
        const auto ph = j.contains("physics") && j["physics"].is_object()
                            ? morphserve::control::ClusterPhysicsProfile::from_json(j["physics"])
                            : morphserve::control::ClusterPhysicsProfile{};
        auto l = std::make_unique<morphserve::control::SimGpuMemoryLedger>(
            j.value("gpu_id", 0),
            j.value("total_hbm_mib", 40960.0),
            pol,
            j.value("role", std::string("hybrid")),
            j.value("resident_expert_layers", 28),
            j.value("num_expert_layers", 28),
            j.value("layer_stride_mib", 448),
            ph,
            j.value("weight_scale", 1.0),
            j.value("vae_spike_scale", 1.0));
        return new ms_ledger_handle{std::move(l)};
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

void ms_ledger_destroy(ms_ledger_t* l) { delete l; }

int ms_ledger_upsert_kv(ms_ledger_t* l, int sid, double logical_kv_mib) {
    return (l && l->ptr && l->ptr->upsert_session_kv(sid, logical_kv_mib)) ? 1 : 0;
}

double ms_ledger_ensure_resident(ms_ledger_t* l, int sid) {
    return (l && l->ptr) ? l->ptr->ensure_session_resident(sid) : -1.0;
}

double ms_ledger_evict_idle_kv(ms_ledger_t* l, double needed_free_mib, int has_excl, int exclude_sid) {
    return (l && l->ptr)
               ? l->ptr->evict_idle_kv_to_host(
                     needed_free_mib, has_excl ? std::optional<int>(exclude_sid) : std::nullopt)
               : 0.0;
}

int ms_ledger_exec_dit_vae(
    ms_ledger_t* l,
    int height,
    int has_active_sid,
    int active_sid,
    double* out_swapped_mib) {
    if (!l || !l->ptr) return 0;
    const auto [ok, sw] = l->ptr->execute_dit_and_vae(
        height, has_active_sid ? std::optional<int>(active_sid) : std::nullopt);
    if (out_swapped_mib) *out_swapped_mib = sw;
    return ok ? 1 : 0;
}

void ms_ledger_release_session(ms_ledger_t* l, int sid) {
    if (l && l->ptr) l->ptr->release_session(sid);
}

void ms_ledger_set_peak_used_mib(ms_ledger_t* l, double v) {
    if (l && l->ptr) l->ptr->set_peak_used_mib(v);
}

char* ms_ledger_stats_json(const ms_ledger_t* l) {
    if (!l || !l->ptr) return nullptr;
    try {
        return dup_json_str(l->ptr->stats_json());
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

// --- ClusterSimulator ---

char* ms_sim_point_json(const char* req_json) {
    try {
        const auto j = nlohmann::json::parse(req_json ? req_json : "{}");
        const auto hw = morphserve::control::HardwareInterconnectSpec::from_json(j.at("hw"));
        const auto ph = j.contains("physics") && j["physics"].is_object()
                            ? morphserve::control::ClusterPhysicsProfile::from_json(j["physics"])
                            : morphserve::control::ClusterPhysicsProfile{};
        const auto ws = j.contains("workload_scale") && j["workload_scale"].is_object()
                            ? morphserve::control::WorkloadScaleSpec::from_json(j["workload_scale"])
                            : morphserve::control::WorkloadScaleSpec::bagel_7b_standard();
        std::vector<morphserve::control::SimSessionSpec> custom;
        const bool has_custom = j.contains("custom_sessions") && j["custom_sessions"].is_array();
        if (has_custom) {
            for (const auto& sj : j["custom_sessions"]) {
                morphserve::control::SimSessionSpec s;
                s.arrival_ms = sj.at(0).get<double>();
                s.quadrant = sj.at(1).get<std::string>();
                for (const auto& tj : sj.at(2)) {
                    s.turns.push_back({tj.at(0).get<std::string>(), tj.at(1).get<int>()});
                }
                custom.push_back(std::move(s));
            }
        }
        const auto res = morphserve::control::ClusterSimulator::simulate_cluster_point(
            j.at("policy_name").get<std::string>(),
            j.value("num_gpus", 8),
            hw,
            j.value("offered_qps", 1.0),
            j.value("num_sessions", 96),
            j.value("seed", uint64_t{2026}),
            ph,
            ws,
            j.value("skew_tag", std::string("")),
            j.value("bench_root", std::string("")),
            j.value("sources_dir", std::string("")),
            has_custom ? &custom : nullptr);
        return dup_json_str(res);
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

char* ms_sim_evaluate_real_json(const char* req_json) {
    try {
        const auto j = nlohmann::json::parse(req_json ? req_json : "{}");
        const auto ph = j.contains("physics") && j["physics"].is_object()
                            ? morphserve::control::ClusterPhysicsProfile::from_json(j["physics"])
                            : morphserve::control::ClusterPhysicsProfile{};
        const auto res = morphserve::control::ClusterSimulator::evaluate_against_real_runs(
            j.value("real_runs", nlohmann::json::array()),
            ph,
            j.value("bench_root", std::string("")),
            j.value("sources_dir", std::string("")));
        return dup_json_str(res);
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

char* ms_sim_h100_study_json(const char* req_json) {
    try {
        const auto j = nlohmann::json::parse(req_json ? req_json : "{}");
        const auto ph = j.contains("physics") && j["physics"].is_object()
                            ? morphserve::control::ClusterPhysicsProfile::from_json(j["physics"])
                            : morphserve::control::ClusterPhysicsProfile{};
        const auto res = morphserve::control::ClusterSimulator::run_h100_80gb_generality_study(
            ph,
            j.value("bench_root", std::string("")),
            j.value("sources_dir", std::string("")));
        return dup_json_str(res);
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

char* ms_sim_sweep_json(const char* req_json) {
    try {
        const auto j = nlohmann::json::parse(req_json ? req_json : "{}");
        const auto res = morphserve::control::ClusterSimulator::run_scale_and_interconnect_sweep(
            j.value("output_dir", std::string("")),
            j.value("real_report_path", std::string("")),
            j.value("results_dir", std::string("")),
            j.value("bench_root", std::string("")),
            j.value("sources_dir", std::string("")));
        return dup_json_str(res);
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

// --- Legacy `tripool_*` C ABI ---

int tripool_vmm_init(
    int device,
    size_t super_pool_mib,
    size_t fine_kv_pool_mib,
    size_t super_handle_mib,
    size_t transient_va_gib,
    size_t kv_va_gib,
    int allow_overflow) {
    morphserve::vmm::TriPoolVMMConfig cfg;
    cfg.device = device;
    cfg.super_pool_mib = super_pool_mib;
    cfg.fine_kv_pool_mib = fine_kv_pool_mib;
    cfg.super_handle_mib = super_handle_mib;
    cfg.transient_va_gib = transient_va_gib;
    cfg.kv_va_gib = kv_va_gib;
    cfg.allow_overflow = (allow_overflow != 0);
    cfg.simulated = false;
    try {
        return morphserve::vmm::TriPoolVMM::init_legacy_singleton(cfg);
    } catch (const std::exception& e) {
        set_err(e);
        return -1;
    }
}

int tripool_vmm_shutdown(void) {
    return morphserve::vmm::TriPoolVMM::shutdown_legacy_singleton();
}

void tripool_vmm_get_stats(UnifiedVmmStatsC* out) {
    if (!out) return;
    auto inst = morphserve::vmm::TriPoolVMM::legacy_singleton();
    if (!inst) inst = morphserve::vmm::TriPoolVMM::active_device_instance(0);
    if (!inst) {
        std::memset(out, 0, sizeof(*out));
        return;
    }
    inst->get_stats(reinterpret_cast<morphserve::vmm::UnifiedVmmStatsPod*>(out));
}

int tripool_host_register(void* ptr, size_t bytes) {
    return morphserve::runtime::HostCudaPinner::instance().register_host(ptr, bytes);
}

int tripool_host_unregister(void* ptr) {
    return morphserve::runtime::HostCudaPinner::instance().unregister_host(ptr);
}

double tripool_host_h2d_bench_gbps(const void* host_ptr, int device, size_t total_bytes, int iters) {
    if (!host_ptr || total_bytes == 0 || iters <= 0) return -1.0;
    try {
        auto& drv = morphserve::runtime::CudaDriver::instance();
        if (drv.cuInit(0) != CUDA_SUCCESS) return -1.0;
        CUdevice dev = 0;
        CUcontext ctx = nullptr;
        if (drv.cuDeviceGet(&dev, device) != CUDA_SUCCESS) return -1.0;
        if (drv.cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) return -1.0;
        drv.cuCtxSetCurrent(ctx);
        size_t gran = 2ULL << 20;
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        drv.cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        const size_t alloc_b = ((total_bytes + gran - 1) / gran) * gran;
        CUdeviceptr va = 0;
        CUmemGenericAllocationHandle h = 0;
        double gbps = -1.0;
        if (drv.cuMemAddressReserve(&va, alloc_b, gran, 0, 0) == CUDA_SUCCESS) {
            if (drv.cuMemCreate(&h, alloc_b, &prop, 0) == CUDA_SUCCESS) {
                if (drv.cuMemMap(va, alloc_b, 0, h, 0) == CUDA_SUCCESS) {
                    CUmemAccessDesc acc{};
                    acc.location = prop.location;
                    acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    drv.cuMemSetAccess(va, alloc_b, &acc, 1);
                    double sum_gbps = 0.0;
                    for (int i = 0; i < iters; ++i) {
                        sum_gbps += morphserve::runtime::HostCudaPinner::instance().stream_h2d_chunks_gbps(
                            static_cast<uint64_t>(va), host_ptr, total_bytes, 64ULL << 20, nullptr);
                    }
                    gbps = sum_gbps / static_cast<double>(iters);
                    drv.cuMemUnmap(va, alloc_b);
                }
                drv.cuMemRelease(h);
            }
            drv.cuMemAddressFree(va, alloc_b);
        }
        drv.cuDevicePrimaryCtxRelease(dev);
        return gbps;
    } catch (...) {
        return -1.0;
    }
}

void tripool_scope_begin(void) {
    auto leg = morphserve::vmm::TriPoolVMM::legacy_singleton();
    if (leg) leg->scope_begin();
    for (int d = 0; d < 16; ++d) {
        auto act = morphserve::vmm::TriPoolVMM::active_device_instance(d);
        if (act && act != leg) act->scope_begin();
    }
}

int tripool_scope_end(void* stream) {
    int rc = 0;
    auto leg = morphserve::vmm::TriPoolVMM::legacy_singleton();
    if (leg) rc = leg->scope_end(stream);
    for (int d = 0; d < 16; ++d) {
        auto act = morphserve::vmm::TriPoolVMM::active_device_instance(d);
        if (act && act != leg) {
            const int r2 = act->scope_end(stream);
            if (rc == 0) rc = r2;
        }
    }
    return rc;
}

void* tripool_alloc(ssize_t size, int device, void* stream) {
    auto inst = resolve_vmm_for_device(device);
    if (!inst) return nullptr;
    try {
        return inst->transient_alloc(size, stream);
    } catch (const std::exception& e) {
        set_err(e);
        return nullptr;
    }
}

void tripool_free(void* ptr, ssize_t size, int device, void* stream) {
    auto inst = resolve_vmm_for_device(device);
    if (inst) inst->transient_free(ptr, size, stream);
}

uint64_t tripool_kv_va_base(void) {
    auto inst = resolve_vmm_for_device(0);
    return inst ? inst->kv_va_base() : 0;
}

uint64_t tripool_kv_free_bytes(int for_dit) {
    auto inst = resolve_vmm_for_device(0);
    return inst ? inst->kv_free_bytes(for_dit != 0, 0) : 0;
}

int32_t tripool_kv_alloc_page(void) {
    auto inst = resolve_vmm_for_device(0);
    if (!inst) return -1;
    try {
        return inst->kv_alloc_page();
    } catch (...) {
        return -1;
    }
}

int tripool_kv_free_page(int32_t slot_idx, void* stream) {
    auto inst = resolve_vmm_for_device(0);
    if (!inst) return -1;
    try {
        return inst->kv_free_page(slot_idx, stream);
    } catch (...) {
        return -1;
    }
}

int32_t tripool_kv_region_alloc(size_t max_va_bytes) {
    auto inst = resolve_vmm_for_device(0);
    if (!inst) return -1;
    try {
        return inst->kv_region_alloc(max_va_bytes, 0);
    } catch (...) {
        return -1;
    }
}

uint64_t tripool_kv_region_ensure(int32_t region_id, size_t needed_bytes, void* stream) {
    auto inst = resolve_vmm_for_device(0);
    if (!inst) return 0;
    try {
        return inst->kv_region_ensure(region_id, needed_bytes, stream);
    } catch (...) {
        return 0;
    }
}

uint64_t tripool_kv_region_va(int32_t region_id) {
    auto inst = resolve_vmm_for_device(0);
    return inst ? inst->kv_region_va(region_id) : 0;
}

size_t tripool_kv_region_mapped_bytes(int32_t region_id) {
    auto inst = resolve_vmm_for_device(0);
    return inst ? inst->kv_region_mapped_bytes(region_id) : 0;
}

int tripool_kv_region_free(int32_t region_id, void* stream) {
    auto inst = resolve_vmm_for_device(0);
    if (!inst) return -1;
    try {
        return inst->kv_region_free(region_id, stream);
    } catch (...) {
        return -1;
    }
}

int tripool_kv_region_set_evictable(int32_t region_id, int evictable, size_t valid_bytes, void* compute_stream) {
    auto inst = resolve_vmm_for_device(0);
    return inst ? inst->kv_region_set_evictable(region_id, evictable, valid_bytes, compute_stream) : -1;
}

int tripool_kv_region_swap_out_async(int32_t region_id, size_t valid_bytes, void* compute_stream) {
    auto inst = resolve_vmm_for_device(0);
    if (!inst) return -1;
    try {
        return inst->kv_region_swap_out_async(region_id, valid_bytes, compute_stream);
    } catch (...) {
        return -1;
    }
}

int tripool_kv_region_swap_in_async(int32_t region_id, void* target_stream) {
    auto inst = resolve_vmm_for_device(0);
    if (!inst) return -1;
    try {
        return inst->kv_region_swap_in_async(region_id, target_stream);
    } catch (...) {
        return -1;
    }
}

int tripool_kv_region_is_swapped(int32_t region_id) {
    auto inst = resolve_vmm_for_device(0);
    return inst ? inst->kv_region_is_swapped(region_id) : 0;
}

size_t tripool_kv_region_swapped_bytes(int32_t region_id) {
    auto inst = resolve_vmm_for_device(0);
    return inst ? inst->kv_region_swapped_bytes(region_id) : 0;
}

int tripool_kv_flush_swaps(void) {
    auto inst = resolve_vmm_for_device(0);
    if (!inst) return -1;
    try {
        return inst->kv_flush_swaps();
    } catch (...) {
        return -1;
    }
}

int tripool_expert_register(int num_layers, size_t layer_stride_mib, void* host_pinned_base, size_t host_bytes) {
    auto inst = resolve_vmm_for_device(0);
    if (!inst) return -1;
    try {
        return inst->expert_register(num_layers, layer_stride_mib, host_pinned_base, host_bytes);
    } catch (...) {
        return -1;
    }
}

uint64_t tripool_expert_va_base(int expert_id) {
    auto inst = resolve_vmm_for_device(0);
    return inst ? inst->expert_va_base(expert_id) : 0;
}

uint64_t tripool_expert_layer_va(int expert_id, int layer_idx) {
    auto inst = resolve_vmm_for_device(0);
    return inst ? inst->expert_layer_va(expert_id, layer_idx) : 0;
}

size_t tripool_expert_layer_stride_bytes(int expert_id) {
    auto inst = resolve_vmm_for_device(0);
    return inst ? inst->expert_layer_stride_bytes(expert_id) : 0;
}

int tripool_expert_set_pinned(int expert_id, int pinned) {
    auto inst = resolve_vmm_for_device(0);
    return inst ? inst->expert_set_pinned(expert_id, pinned) : -1;
}

int tripool_expert_set_layer_used_bytes(int expert_id, int layer_idx, size_t used_bytes) {
    auto inst = resolve_vmm_for_device(0);
    return inst ? inst->expert_set_layer_used_bytes(expert_id, layer_idx, used_bytes) : -1;
}

int tripool_expert_load_layers_sync(int expert_id, int start_layer, int end_layer_exclusive) {
    auto inst = resolve_vmm_for_device(0);
    if (!inst) return -1;
    try {
        return inst->expert_load_layers_sync(expert_id, start_layer, end_layer_exclusive);
    } catch (...) {
        return -1;
    }
}

int tripool_expert_prefetch_async(int expert_id, int start_layer, int end_layer_exclusive) {
    auto inst = resolve_vmm_for_device(0);
    return inst ? inst->expert_prefetch_async(expert_id, start_layer, end_layer_exclusive) : -1;
}

int tripool_expert_wait_layer(int expert_id, int layer_idx, void* compute_stream) {
    auto inst = resolve_vmm_for_device(0);
    if (!inst) return -1;
    try {
        return inst->expert_wait_layer(expert_id, layer_idx, compute_stream);
    } catch (...) {
        return -1;
    }
}

int tripool_expert_evict_layer(int expert_id, int layer_idx, void* stream) {
    auto inst = resolve_vmm_for_device(0);
    if (!inst) return -1;
    try {
        return inst->expert_evict_layer(expert_id, layer_idx, stream);
    } catch (...) {
        return -1;
    }
}

int tripool_expert_evict_range(int expert_id, int start_layer, int end_layer_exclusive, void* stream) {
    auto inst = resolve_vmm_for_device(0);
    if (!inst) return -1;
    try {
        return inst->expert_evict_range(expert_id, start_layer, end_layer_exclusive, stream);
    } catch (...) {
        return -1;
    }
}

int tripool_expert_is_resident(int expert_id, int layer_idx) {
    auto inst = resolve_vmm_for_device(0);
    return inst ? inst->expert_is_resident(expert_id, layer_idx) : 0;
}

int tripool_expert_resident_count(int expert_id) {
    auto inst = resolve_vmm_for_device(0);
    return inst ? inst->expert_resident_count(expert_id) : 0;
}

int tripool_init_ex(int device, size_t pool_mib, size_t va_gib, size_t handle_mib) {
    return tripool_vmm_init(device, pool_mib, 64, handle_mib, va_gib, 2, 0);
}

void tripool_get_stats(TripoolPhase0StatsC* out) {
    if (!out) return;
    std::memset(out, 0, sizeof(*out));
    UnifiedVmmStatsC st{};
    tripool_vmm_get_stats(&st);
    out->min_granularity_bytes = st.min_granularity_bytes;
    out->handle_bytes = st.super_handle_bytes;
    out->pool_total_handles = st.super_pool_total_handles;
    out->pool_free_handles = st.super_pool_free_handles;
    out->mapped_handles_current = st.transient_mapped_handles_current;
    out->mapped_handles_peak = st.transient_mapped_handles_peak;
    out->live_allocs_current = st.transient_live_allocs;
    out->overflow_creates = st.overflow_creates;
    out->alloc_calls = st.transient_alloc_calls;
    out->free_calls = st.transient_free_calls;
    out->scope_flushes = st.transient_scope_flushes;
    out->total_map_us = st.transient_map_us;
    out->total_unmap_us = st.transient_unmap_us;
    out->total_sync_us = 0.0;
}

int tripool_shutdown(void) {
    return tripool_vmm_shutdown();
}

}  // extern "C"
