#pragma once

#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <stdexcept>
#include <string>

#if __has_include(<cuda.h>)
#include <cuda.h>
#define MORPHSERVE_HAS_CUDA_H 1
#else
#define MORPHSERVE_HAS_CUDA_H 0

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUstream_st* CUstream;
typedef struct CUevent_st* CUevent;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;

constexpr CUresult CUDA_SUCCESS = 0;

typedef enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,
} CUdevice_attribute;

typedef enum CUmemAllocationType_enum {
    CU_MEM_ALLOCATION_TYPE_INVALID = 0x0,
    CU_MEM_ALLOCATION_TYPE_PINNED = 0x1,
    CU_MEM_ALLOCATION_TYPE_MAX = 0x7FFFFFFF,
} CUmemAllocationType;

typedef enum CUmemAllocationHandleType_enum {
    CU_MEM_HANDLE_TYPE_NONE = 0x0,
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1,
    CU_MEM_HANDLE_TYPE_WIN32 = 0x2,
    CU_MEM_HANDLE_TYPE_WIN32_KMT = 0x4,
    CU_MEM_HANDLE_TYPE_MAX = 0x7FFFFFFF,
} CUmemAllocationHandleType;

typedef enum CUmemLocationType_enum {
    CU_MEM_LOCATION_TYPE_INVALID = 0x0,
    CU_MEM_LOCATION_TYPE_DEVICE = 0x1,
    CU_MEM_LOCATION_TYPE_HOST = 0x2,
    CU_MEM_LOCATION_TYPE_HOST_NUMA = 0x3,
    CU_MEM_LOCATION_TYPE_MAX = 0x7FFFFFFF,
} CUmemLocationType;

typedef struct CUmemLocation_st {
    CUmemLocationType type;
    int id;
} CUmemLocation;

typedef struct CUmemAllocationProp_st {
    CUmemAllocationType type;
    CUmemAllocationHandleType requestedHandleTypes;
    CUmemLocation location;
    void* win32HandleMetaData;
    struct {
        unsigned char compressionType;
        unsigned char gpuDirectRDMACapable;
        unsigned short usage;
        unsigned char reserved[4];
    } allocFlags;
} CUmemAllocationProp;

typedef enum CUmemAccess_flags_enum {
    CU_MEM_ACCESS_FLAGS_PROT_NONE = 0x0,
    CU_MEM_ACCESS_FLAGS_PROT_READ = 0x1,
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3,
    CU_MEM_ACCESS_FLAGS_PROT_MAX = 0x7FFFFFFF,
} CUmemAccess_flags;

typedef struct CUmemAccessDesc_st {
    CUmemLocation location;
    CUmemAccess_flags flags;
} CUmemAccessDesc;

typedef enum CUmemAllocationGranularity_flags_enum {
    CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0x0,
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1,
} CUmemAllocationGranularity_flags;

constexpr unsigned int CU_STREAM_NON_BLOCKING = 0x1;
constexpr unsigned int CU_EVENT_DEFAULT = 0x0;
constexpr unsigned int CU_MEMHOSTALLOC_PORTABLE = 0x1;

#endif  // __has_include(<cuda.h>)

namespace morphserve::vmm {

struct CudaDriver {
    void* lib = nullptr;

    CUresult (*cuInit)(unsigned int) = nullptr;
    CUresult (*cuDriverGetVersion)(int*) = nullptr;
    CUresult (*cuGetErrorName)(CUresult, const char**) = nullptr;
    CUresult (*cuGetErrorString)(CUresult, const char**) = nullptr;
    CUresult (*cuDeviceGetCount)(int*) = nullptr;
    CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
    CUresult (*cuDeviceGetName)(char*, int, CUdevice) = nullptr;
    CUresult (*cuDeviceGetAttribute)(int*, CUdevice_attribute, CUdevice) = nullptr;
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
    CUresult (*cuDevicePrimaryCtxRelease)(CUdevice) = nullptr;
    CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
    CUresult (*cuCtxGetCurrent)(CUcontext*) = nullptr;
    CUresult (*cuCtxSynchronize)() = nullptr;
    CUresult (*cuMemGetInfo)(size_t*, size_t*) = nullptr;

    CUresult (*cuMemGetAllocationGranularity)(
        size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) = nullptr;
    CUresult (*cuMemCreate)(
        CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long) = nullptr;
    CUresult (*cuMemRelease)(CUmemGenericAllocationHandle) = nullptr;
    CUresult (*cuMemAddressReserve)(
        CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) = nullptr;
    CUresult (*cuMemAddressFree)(CUdeviceptr, size_t) = nullptr;
    CUresult (*cuMemMap)(
        CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) = nullptr;
    CUresult (*cuMemUnmap)(CUdeviceptr, size_t) = nullptr;
    CUresult (*cuMemSetAccess)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) = nullptr;

    CUresult (*cuMemHostAlloc)(void**, size_t, unsigned int) = nullptr;
    CUresult (*cuMemFreeHost)(void*) = nullptr;
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;
    CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream) = nullptr;
    CUresult (*cuMemcpyDtoHAsync)(void*, CUdeviceptr, size_t, CUstream) = nullptr;
    CUresult (*cuMemsetD8Async)(CUdeviceptr, unsigned char, size_t, CUstream) = nullptr;

    CUresult (*cuStreamCreate)(CUstream*, unsigned int) = nullptr;
    CUresult (*cuStreamDestroy)(CUstream) = nullptr;
    CUresult (*cuStreamSynchronize)(CUstream) = nullptr;
    CUresult (*cuEventCreate)(CUevent*, unsigned int) = nullptr;
    CUresult (*cuEventDestroy)(CUevent) = nullptr;
    CUresult (*cuEventRecord)(CUevent, CUstream) = nullptr;
    CUresult (*cuEventSynchronize)(CUevent) = nullptr;
    CUresult (*cuEventElapsedTime)(float*, CUevent, CUevent) = nullptr;

    static CudaDriver& instance() {
        static CudaDriver drv;
        return drv;
    }

    std::string err_str(CUresult res) const {
        const char* name = nullptr;
        const char* desc = nullptr;
        if (cuGetErrorName) cuGetErrorName(res, &name);
        if (cuGetErrorString) cuGetErrorString(res, &desc);
        return std::string(name ? name : "CUresult(" + std::to_string(res) + ")") +
               ": " + (desc ? desc : "unknown error");
    }

    void check(CUresult res, const char* what) const {
        if (res != CUDA_SUCCESS) {
            throw std::runtime_error(std::string(what) + " failed: " + err_str(res));
        }
    }

private:
    template <typename Fn>
    void load(Fn& out, const char* primary, const char* fallback = nullptr) {
        void* sym = dlsym(lib, primary);
        if (!sym && fallback) {
            sym = dlsym(lib, fallback);
        }
        if (!sym) {
            throw std::runtime_error(std::string("missing CUDA driver symbol: ") + primary);
        }
        out = reinterpret_cast<Fn>(sym);
    }

    CudaDriver() {
        lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!lib) {
            lib = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
        }
        if (!lib) {
            throw std::runtime_error(std::string("failed to dlopen libcuda.so.1: ") +
                                     (dlerror() ? dlerror() : "unknown"));
        }
        load(cuInit, "cuInit");
        load(cuDriverGetVersion, "cuDriverGetVersion");
        load(cuGetErrorName, "cuGetErrorName");
        load(cuGetErrorString, "cuGetErrorString");
        load(cuDeviceGetCount, "cuDeviceGetCount");
        load(cuDeviceGet, "cuDeviceGet");
        load(cuDeviceGetName, "cuDeviceGetName");
        load(cuDeviceGetAttribute, "cuDeviceGetAttribute");
        load(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
        load(cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease");
        load(cuCtxSetCurrent, "cuCtxSetCurrent");
        load(cuCtxGetCurrent, "cuCtxGetCurrent");
        load(cuCtxSynchronize, "cuCtxSynchronize");
        load(cuMemGetInfo, "cuMemGetInfo_v2", "cuMemGetInfo");

        load(cuMemGetAllocationGranularity, "cuMemGetAllocationGranularity");
        load(cuMemCreate, "cuMemCreate");
        load(cuMemRelease, "cuMemRelease");
        load(cuMemAddressReserve, "cuMemAddressReserve");
        load(cuMemAddressFree, "cuMemAddressFree");
        load(cuMemMap, "cuMemMap");
        load(cuMemUnmap, "cuMemUnmap");
        load(cuMemSetAccess, "cuMemSetAccess");

        load(cuMemHostAlloc, "cuMemHostAlloc");
        load(cuMemFreeHost, "cuMemFreeHost");
        load(cuMemcpyHtoD, "cuMemcpyHtoD_v2", "cuMemcpyHtoD");
        load(cuMemcpyDtoH, "cuMemcpyDtoH_v2", "cuMemcpyDtoH");
        load(cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync");
        load(cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync");
        load(cuMemsetD8Async, "cuMemsetD8Async");

        load(cuStreamCreate, "cuStreamCreate");
        load(cuStreamDestroy, "cuStreamDestroy_v2", "cuStreamDestroy");
        load(cuStreamSynchronize, "cuStreamSynchronize");
        load(cuEventCreate, "cuEventCreate");
        load(cuEventDestroy, "cuEventDestroy_v2", "cuEventDestroy");
        load(cuEventRecord, "cuEventRecord");
        load(cuEventSynchronize, "cuEventSynchronize");
        load(cuEventElapsedTime, "cuEventElapsedTime");
    }
};

}  // namespace morphserve::vmm
