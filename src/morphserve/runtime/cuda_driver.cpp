#include "morphserve/runtime/cuda_driver.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <dlfcn.h>
#include <stdexcept>

namespace morphserve::runtime {

namespace {

template <typename Fn>
void load_sym(void* lib, Fn& out, const char* primary, const char* fallback = nullptr) {
    void* sym = dlsym(lib, primary);
    if (!sym && fallback) {
        sym = dlsym(lib, fallback);
    }
    if (!sym) {
        throw std::runtime_error(std::string("missing CUDA driver symbol: ") + primary);
    }
    out = reinterpret_cast<Fn>(sym);
}

}  // namespace

CudaDriver& CudaDriver::instance() {
    static CudaDriver drv;
    return drv;
}

bool CudaDriver::is_available() noexcept {
    try {
        auto& drv = instance();
        if (drv.cuInit(0) != CUDA_SUCCESS) return false;
        int cnt = 0;
        if (drv.cuDeviceGetCount(&cnt) != CUDA_SUCCESS) return false;
        return cnt > 0;
    } catch (...) {
        return false;
    }
}

CudaDriver::CudaDriver() {
    lib_ = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!lib_) {
        lib_ = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
    }
    if (!lib_) {
        const char* err = dlerror();
        throw std::runtime_error(std::string("failed to dlopen libcuda.so.1: ") + (err ? err : "unknown"));
    }
    load_sym(lib_, cuInit, "cuInit");
    load_sym(lib_, cuDriverGetVersion, "cuDriverGetVersion");
    load_sym(lib_, cuGetErrorName, "cuGetErrorName");
    load_sym(lib_, cuGetErrorString, "cuGetErrorString");
    load_sym(lib_, cuDeviceGetCount, "cuDeviceGetCount");
    load_sym(lib_, cuDeviceGet, "cuDeviceGet");
    load_sym(lib_, cuDeviceGetName, "cuDeviceGetName");
    load_sym(lib_, cuDeviceGetAttribute, "cuDeviceGetAttribute");
    load_sym(lib_, cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
    load_sym(lib_, cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease");
    load_sym(lib_, cuCtxSetCurrent, "cuCtxSetCurrent");
    load_sym(lib_, cuCtxGetCurrent, "cuCtxGetCurrent");
    load_sym(lib_, cuCtxSynchronize, "cuCtxSynchronize");
    load_sym(lib_, cuMemGetInfo, "cuMemGetInfo_v2", "cuMemGetInfo");

    load_sym(lib_, cuMemGetAllocationGranularity, "cuMemGetAllocationGranularity");
    load_sym(lib_, cuMemCreate, "cuMemCreate");
    load_sym(lib_, cuMemRelease, "cuMemRelease");
    load_sym(lib_, cuMemAddressReserve, "cuMemAddressReserve");
    load_sym(lib_, cuMemAddressFree, "cuMemAddressFree");
    load_sym(lib_, cuMemMap, "cuMemMap");
    load_sym(lib_, cuMemUnmap, "cuMemUnmap");
    load_sym(lib_, cuMemSetAccess, "cuMemSetAccess");

    load_sym(lib_, cuMemHostAlloc, "cuMemHostAlloc");
    load_sym(lib_, cuMemFreeHost, "cuMemFreeHost");
    load_sym(lib_, cuMemHostRegister, "cuMemHostRegister_v2", "cuMemHostRegister");
    load_sym(lib_, cuMemHostUnregister, "cuMemHostUnregister");
    load_sym(lib_, cuMemcpyHtoD, "cuMemcpyHtoD_v2", "cuMemcpyHtoD");
    load_sym(lib_, cuMemcpyDtoH, "cuMemcpyDtoH_v2", "cuMemcpyDtoH");
    load_sym(lib_, cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync");
    load_sym(lib_, cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync");
    load_sym(lib_, cuMemsetD8Async, "cuMemsetD8Async");

    load_sym(lib_, cuCtxGetStreamPriorityRange, "cuCtxGetStreamPriorityRange");
    load_sym(lib_, cuStreamCreate, "cuStreamCreate");
    load_sym(lib_, cuStreamCreateWithPriority, "cuStreamCreateWithPriority");
    load_sym(lib_, cuStreamDestroy, "cuStreamDestroy_v2", "cuStreamDestroy");
    load_sym(lib_, cuStreamSynchronize, "cuStreamSynchronize");
    load_sym(lib_, cuStreamWaitEvent, "cuStreamWaitEvent");
    load_sym(lib_, cuEventCreate, "cuEventCreate");
    load_sym(lib_, cuEventDestroy, "cuEventDestroy_v2", "cuEventDestroy");
    load_sym(lib_, cuEventRecord, "cuEventRecord");
    load_sym(lib_, cuEventSynchronize, "cuEventSynchronize");
    load_sym(lib_, cuEventElapsedTime, "cuEventElapsedTime");
}

std::string CudaDriver::err_str(CUresult res) const {
    const char* name = nullptr;
    const char* desc = nullptr;
    if (cuGetErrorName) cuGetErrorName(res, &name);
    if (cuGetErrorString) cuGetErrorString(res, &desc);
    return std::string(name ? name : ("CUresult(" + std::to_string(res) + ")")) +
           ": " + (desc ? desc : "unknown error");
}

void CudaDriver::check(CUresult res, const char* what) const {
    if (res != CUDA_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed: " + err_str(res));
    }
}

HostCudaPinner& HostCudaPinner::instance() {
    static HostCudaPinner pinner;
    return pinner;
}

void HostCudaPinner::ensure_ctx_locked(CudaDriver& drv) {
    drv.check(drv.cuInit(0), "cuInit(0)");
    CUcontext cur = nullptr;
    drv.check(drv.cuCtxGetCurrent(&cur), "cuCtxGetCurrent");
    if (!cur) {
        if (!retained_ctx_) {
            CUdevice dev0 = 0;
            drv.check(drv.cuDeviceGet(&dev0, 0), "cuDeviceGet(0)");
            drv.check(drv.cuDevicePrimaryCtxRetain(&retained_ctx_, dev0), "cuDevicePrimaryCtxRetain");
            retained_dev_ = dev0;
        }
        drv.check(drv.cuCtxSetCurrent(retained_ctx_), "cuCtxSetCurrent");
    }
}

void HostCudaPinner::release_ctx_if_idle_locked(CudaDriver& drv) noexcept {
    if (active_pins_ == 0 && retained_ctx_) {
        CUcontext cur = nullptr;
        if (drv.cuCtxGetCurrent(&cur) == CUDA_SUCCESS && cur == retained_ctx_) {
            drv.cuCtxSetCurrent(nullptr);
        }
        drv.cuDevicePrimaryCtxRelease(retained_dev_);
        retained_ctx_ = nullptr;
    }
}

int HostCudaPinner::register_host(void* ptr, size_t nbytes) {
    if (!ptr || nbytes == 0) return -1;
    std::lock_guard<std::mutex> lock(mu_);
    try {
        auto& drv = CudaDriver::instance();
        ensure_ctx_locked(drv);
        try {
            drv.check(drv.cuMemHostRegister(ptr, nbytes, CU_MEMHOSTREGISTER_PORTABLE), "cuMemHostRegister");
        } catch (...) {
            release_ctx_if_idle_locked(drv);
            throw;
        }
        if (retained_ctx_) {
            ++active_pins_;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[morphserve] register_host error: %s\n", e.what());
        return -1;
    }
}

int HostCudaPinner::unregister_host(void* ptr) noexcept {
    if (!ptr) return -1;
    std::lock_guard<std::mutex> lock(mu_);
    try {
        auto& drv = CudaDriver::instance();
        CUcontext cur = nullptr;
        if (drv.cuCtxGetCurrent(&cur) == CUDA_SUCCESS && !cur && retained_ctx_) {
            drv.cuCtxSetCurrent(retained_ctx_);
        }
        CUresult rc = drv.cuMemHostUnregister(ptr);
        if (active_pins_ > 0) {
            --active_pins_;
            release_ctx_if_idle_locked(drv);
        }
        return (rc == CUDA_SUCCESS) ? 0 : -1;
    } catch (...) {
        return -1;
    }
}

double HostCudaPinner::stream_h2d_chunks_gbps(
    uint64_t dst_dev_ptr,
    const void* src_host,
    size_t nbytes,
    size_t chunk_bytes,
    void* cuda_stream) {
    if (nbytes == 0) return 0.0;
    auto& drv = CudaDriver::instance();
    CUstream st = reinterpret_cast<CUstream>(cuda_stream);
    drv.check(drv.cuStreamSynchronize(st), "pre-stream_h2d sync");
    const size_t eff_chunk = std::max<size_t>(4096, chunk_bytes);
    auto t0 = std::chrono::steady_clock::now();
    size_t off = 0;
    const auto* src_u8 = static_cast<const uint8_t*>(src_host);
    while (off < nbytes) {
        const size_t sz = std::min(eff_chunk, nbytes - off);
        drv.check(
            drv.cuMemcpyHtoDAsync(static_cast<CUdeviceptr>(dst_dev_ptr + off), src_u8 + off, sz, st),
            "stream_h2d cuMemcpyHtoDAsync");
        off += sz;
    }
    drv.check(drv.cuStreamSynchronize(st), "post-stream_h2d sync");
    const double dt_s = std::max(
        1e-6, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return (static_cast<double>(nbytes) / 1e9) / dt_s;
}

}  // namespace morphserve::runtime
