#include "morphserve/vmm/page_backend.hpp"
#include "morphserve/runtime/cuda_driver.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <system_error>
#include <unistd.h>

namespace morphserve::vmm {

using morphserve::runtime::CudaDriver;

namespace {

CUmemAllocationProp make_device_prop(int device) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    return prop;
}

CUmemAccessDesc make_device_rw_access(int device) {
    CUmemAccessDesc acc = {};
    acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    acc.location.id = device;
    acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return acc;
}

}  // namespace

// ==================== CudaPageAdapter ====================

CudaPageAdapter::CudaPageAdapter(int device) : device_(device) {
    auto& drv = CudaDriver::instance();
    drv.check(drv.cuInit(0), "cuInit");
    CUdevice dev = 0;
    drv.check(drv.cuDeviceGet(&dev, device_), "cuDeviceGet");
    cu_dev_ = dev;
    CUcontext ctx = nullptr;
    drv.check(drv.cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
    primary_ctx_ = ctx;

    CUcontext cur = nullptr;
    drv.check(drv.cuCtxGetCurrent(&cur), "cuCtxGetCurrent");
    if (cur != ctx) {
        drv.check(drv.cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
    }

    const CUmemAllocationProp prop = make_device_prop(device_);
    drv.check(
        drv.cuMemGetAllocationGranularity(&min_gran_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "cuMemGetAllocationGranularity");
}

CudaPageAdapter::~CudaPageAdapter() noexcept {
    if (primary_ctx_) {
        try {
            CudaDriver::instance().cuDevicePrimaryCtxRelease(static_cast<CUdevice>(cu_dev_));
        } catch (...) {
        }
        primary_ctx_ = nullptr;
    }
}

void CudaPageAdapter::ensure_context() {
    auto& drv = CudaDriver::instance();
    CUcontext cur = nullptr;
    drv.check(drv.cuCtxGetCurrent(&cur), "cuCtxGetCurrent");
    if (cur != reinterpret_cast<CUcontext>(primary_ctx_) && primary_ctx_) {
        drv.check(drv.cuCtxSetCurrent(reinterpret_cast<CUcontext>(primary_ctx_)), "cuCtxSetCurrent");
    }
}

void CudaPageAdapter::sync_context() {
    ensure_context();
    CudaDriver::instance().check(CudaDriver::instance().cuCtxSynchronize(), "cuCtxSynchronize");
}

uint64_t CudaPageAdapter::reserve_va(size_t bytes, size_t alignment) {
    CUdeviceptr va = 0;
    CudaDriver::instance().check(
        CudaDriver::instance().cuMemAddressReserve(&va, bytes, alignment, 0, 0),
        "cuMemAddressReserve");
    return static_cast<uint64_t>(va);
}

void CudaPageAdapter::free_va(uint64_t va, size_t bytes) noexcept {
    if (va && bytes > 0) {
        try {
            CudaDriver::instance().cuMemAddressFree(static_cast<CUdeviceptr>(va), bytes);
        } catch (...) {
        }
    }
}

uint64_t CudaPageAdapter::create_phys_handle(size_t bytes) {
    const CUmemAllocationProp prop = make_device_prop(device_);
    CUmemGenericAllocationHandle h = 0;
    CudaDriver::instance().check(
        CudaDriver::instance().cuMemCreate(&h, bytes, &prop, 0),
        "cuMemCreate");
    return static_cast<uint64_t>(h);
}

void CudaPageAdapter::release_phys_handle(uint64_t handle, size_t /*bytes*/) noexcept {
    if (handle) {
        try {
            CudaDriver::instance().cuMemRelease(static_cast<CUmemGenericAllocationHandle>(handle));
        } catch (...) {
        }
    }
}

void CudaPageAdapter::map_range(uint64_t va, size_t bytes, uint64_t handle, bool /*need_host_access*/) {
    CudaDriver::instance().check(
        CudaDriver::instance().cuMemMap(
            static_cast<CUdeviceptr>(va),
            bytes,
            0,
            static_cast<CUmemGenericAllocationHandle>(handle),
            0),
        "cuMemMap");
}

void CudaPageAdapter::unmap_range(uint64_t va, size_t bytes, bool /*was_host_accessible*/) {
    CudaDriver::instance().check(
        CudaDriver::instance().cuMemUnmap(static_cast<CUdeviceptr>(va), bytes),
        "cuMemUnmap");
}

void CudaPageAdapter::set_access_rw(uint64_t va, size_t bytes, bool /*need_host_access*/) {
    const CUmemAccessDesc acc = make_device_rw_access(device_);
    CudaDriver::instance().check(
        CudaDriver::instance().cuMemSetAccess(static_cast<CUdeviceptr>(va), bytes, &acc, 1),
        "cuMemSetAccess");
}

void* CudaPageAdapter::create_stream(bool high_priority) {
    auto& drv = CudaDriver::instance();
    CUstream st = nullptr;
    if (high_priority) {
        int least = 0, greatest = 0;
        if (drv.cuCtxGetStreamPriorityRange(&least, &greatest) == CUDA_SUCCESS) {
            drv.check(
                drv.cuStreamCreateWithPriority(&st, CU_STREAM_NON_BLOCKING, greatest),
                "cuStreamCreateWithPriority");
            return st;
        }
    }
    drv.check(drv.cuStreamCreate(&st, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
    return st;
}

void CudaPageAdapter::destroy_stream(void* stream) noexcept {
    if (stream) {
        try {
            CudaDriver::instance().cuStreamDestroy(reinterpret_cast<CUstream>(stream));
        } catch (...) {
        }
    }
}

void CudaPageAdapter::sync_stream(void* stream) {
    CudaDriver::instance().check(
        CudaDriver::instance().cuStreamSynchronize(reinterpret_cast<CUstream>(stream)),
        "cuStreamSynchronize");
}

void* CudaPageAdapter::create_event() {
    CUevent ev = nullptr;
    CudaDriver::instance().check(
        CudaDriver::instance().cuEventCreate(&ev, CU_EVENT_DISABLE_TIMING),
        "cuEventCreate");
    return ev;
}

void CudaPageAdapter::destroy_event(void* ev) noexcept {
    if (ev) {
        try {
            CudaDriver::instance().cuEventDestroy(reinterpret_cast<CUevent>(ev));
        } catch (...) {
        }
    }
}

void CudaPageAdapter::record_event(void* ev, void* stream) {
    CudaDriver::instance().check(
        CudaDriver::instance().cuEventRecord(reinterpret_cast<CUevent>(ev), reinterpret_cast<CUstream>(stream)),
        "cuEventRecord");
}

void CudaPageAdapter::stream_wait_event(void* stream, void* ev) {
    CudaDriver::instance().check(
        CudaDriver::instance().cuStreamWaitEvent(
            reinterpret_cast<CUstream>(stream), reinterpret_cast<CUevent>(ev), 0),
        "cuStreamWaitEvent");
}

void CudaPageAdapter::sync_event(void* ev) {
    if (ev) {
        CudaDriver::instance().check(
            CudaDriver::instance().cuEventSynchronize(reinterpret_cast<CUevent>(ev)),
            "cuEventSynchronize");
    }
}

bool CudaPageAdapter::sync_event_noexcept(void* ev) noexcept {
    if (!ev) return true;
    try {
        return CudaDriver::instance().cuEventSynchronize(reinterpret_cast<CUevent>(ev)) == CUDA_SUCCESS;
    } catch (...) {
        return false;
    }
}

void* CudaPageAdapter::alloc_host_pinned(size_t bytes) {
    void* ptr = nullptr;
    CudaDriver::instance().check(
        CudaDriver::instance().cuMemHostAlloc(&ptr, bytes, CU_MEMHOSTALLOC_PORTABLE),
        "cuMemHostAlloc");
    return ptr;
}

void CudaPageAdapter::free_host_pinned(void* ptr) noexcept {
    if (ptr) {
        try {
            CudaDriver::instance().cuMemFreeHost(ptr);
        } catch (...) {
        }
    }
}

void CudaPageAdapter::memcpy_h2d_async(uint64_t dst_va, const void* src_host, size_t bytes, void* stream) {
    if (bytes == 0) return;
    CudaDriver::instance().check(
        CudaDriver::instance().cuMemcpyHtoDAsync(
            static_cast<CUdeviceptr>(dst_va), src_host, bytes, reinterpret_cast<CUstream>(stream)),
        "cuMemcpyHtoDAsync");
}

void CudaPageAdapter::memcpy_d2h_async(void* dst_host, uint64_t src_va, size_t bytes, void* stream) {
    if (bytes == 0) return;
    CudaDriver::instance().check(
        CudaDriver::instance().cuMemcpyDtoHAsync(
            dst_host, static_cast<CUdeviceptr>(src_va), bytes, reinterpret_cast<CUstream>(stream)),
        "cuMemcpyDtoHAsync");
}

// ==================== HostMmapPageAdapter ====================

HostMmapPageAdapter::HostMmapPageAdapter(int device, size_t min_gran_bytes)
    : device_(device), min_gran_(std::max<size_t>(4096, min_gran_bytes)) {}

HostMmapPageAdapter::~HostMmapPageAdapter() noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& kv : reserved_vas_) {
        ::munmap(reinterpret_cast<void*>(kv.first), kv.second);
    }
    reserved_vas_.clear();
}

uint64_t HostMmapPageAdapter::reserve_va(size_t bytes, size_t alignment) {
    const size_t page_sz = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    const size_t eff_align = std::max({page_sz, size_t(4096), alignment});
    const size_t eff_bytes = ((std::max<size_t>(1, bytes) + page_sz - 1) / page_sz) * page_sz;
    const size_t over_bytes = eff_bytes + eff_align;

    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    void* raw = ::mmap(nullptr, over_bytes, PROT_NONE, flags, -1, 0);
    if (raw == MAP_FAILED) {
        throw std::system_error(errno, std::generic_category(), "HostMmapPageAdapter mmap(PROT_NONE) failed");
    }

    const uintptr_t raw_u = reinterpret_cast<uintptr_t>(raw);
    const uintptr_t aligned_u = (raw_u + eff_align - 1) & ~(eff_align - 1);
    const size_t lead = aligned_u - raw_u;
    const size_t trail = over_bytes - lead - eff_bytes;
    if (lead > 0) {
        ::munmap(reinterpret_cast<void*>(raw_u), lead);
    }
    if (trail > 0) {
        ::munmap(reinterpret_cast<void*>(aligned_u + eff_bytes), trail);
    }

    const uint64_t va = static_cast<uint64_t>(aligned_u);
    std::lock_guard<std::mutex> lock(mu_);
    reserved_vas_[va] = eff_bytes;
    return va;
}

void HostMmapPageAdapter::free_va(uint64_t va, size_t bytes) noexcept {
    if (!va) return;
    size_t unmap_len = bytes;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = reserved_vas_.find(va);
        if (it != reserved_vas_.end()) {
            unmap_len = it->second;
            reserved_vas_.erase(it);
        }
    }
    if (unmap_len > 0) {
        ::munmap(reinterpret_cast<void*>(va), unmap_len);
    }
}

uint64_t HostMmapPageAdapter::create_phys_handle(size_t /*bytes*/) {
    std::lock_guard<std::mutex> lock(mu_);
    return next_id_++;
}

void HostMmapPageAdapter::release_phys_handle(uint64_t /*handle*/, size_t /*bytes*/) noexcept {}

void HostMmapPageAdapter::map_range(uint64_t va, size_t bytes, uint64_t /*handle*/, bool need_host_access) {
    if (!need_host_access || bytes == 0) return;
    if (::mprotect(reinterpret_cast<void*>(va), bytes, PROT_READ | PROT_WRITE) != 0) {
        throw std::system_error(errno, std::generic_category(), "HostMmapPageAdapter mprotect(RW) failed");
    }
}

void HostMmapPageAdapter::unmap_range(uint64_t va, size_t bytes, bool was_host_accessible) {
    if (!was_host_accessible || bytes == 0) return;
    ::madvise(reinterpret_cast<void*>(va), bytes, MADV_DONTNEED);
    ::mprotect(reinterpret_cast<void*>(va), bytes, PROT_NONE);
}

void HostMmapPageAdapter::set_access_rw(uint64_t /*va*/, size_t /*bytes*/, bool /*need_host_access*/) {}

void* HostMmapPageAdapter::create_stream(bool /*high_priority*/) {
    std::lock_guard<std::mutex> lock(mu_);
    return reinterpret_cast<void*>(static_cast<uintptr_t>(next_id_++));
}

void HostMmapPageAdapter::destroy_stream(void* /*stream*/) noexcept {}
void HostMmapPageAdapter::sync_stream(void* /*stream*/) {}

void* HostMmapPageAdapter::create_event() {
    std::lock_guard<std::mutex> lock(mu_);
    return reinterpret_cast<void*>(static_cast<uintptr_t>(next_id_++));
}

void HostMmapPageAdapter::destroy_event(void* /*ev*/) noexcept {}
void HostMmapPageAdapter::record_event(void* /*ev*/, void* /*stream*/) {}
void HostMmapPageAdapter::stream_wait_event(void* /*stream*/, void* /*ev*/) {}
void HostMmapPageAdapter::sync_event(void* /*ev*/) {}
bool HostMmapPageAdapter::sync_event_noexcept(void* /*ev*/) noexcept { return true; }

void* HostMmapPageAdapter::alloc_host_pinned(size_t bytes) {
    void* ptr = nullptr;
    const size_t sz = std::max<size_t>(64, bytes);
    if (::posix_memalign(&ptr, 4096, sz) != 0 || !ptr) {
        throw std::bad_alloc();
    }
    std::memset(ptr, 0, sz);
    return ptr;
}

void HostMmapPageAdapter::free_host_pinned(void* ptr) noexcept {
    std::free(ptr);
}

void HostMmapPageAdapter::memcpy_h2d_async(uint64_t dst_va, const void* src_host, size_t bytes, void* /*stream*/) {
    if (bytes > 0 && dst_va && src_host) {
        std::memcpy(reinterpret_cast<void*>(dst_va), src_host, bytes);
    }
}

void HostMmapPageAdapter::memcpy_d2h_async(void* dst_host, uint64_t src_va, size_t bytes, void* /*stream*/) {
    if (bytes > 0 && dst_host && src_va) {
        std::memcpy(dst_host, reinterpret_cast<const void*>(src_va), bytes);
    }
}

}  // namespace morphserve::vmm
