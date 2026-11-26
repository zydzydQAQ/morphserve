#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

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
constexpr CUresult CUDA_ERROR_OUT_OF_MEMORY = 2;

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
constexpr unsigned int CU_EVENT_DISABLE_TIMING = 0x2;
constexpr unsigned int CU_MEMHOSTALLOC_PORTABLE = 0x1;
constexpr unsigned int CU_MEMHOSTREGISTER_PORTABLE = 0x1;

#endif  // __has_include(<cuda.h>)

namespace morphserve::runtime {

// Runtime dynamic loader (`dlopen("libcuda.so.1")`) with zero build-time `nvcc` or `cuda.h` dependency.
class CudaDriver {
public:
    static CudaDriver& instance();
    static bool is_available() noexcept;

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
    CUresult (*cuCtxGetDevice)(CUdevice*) = nullptr;
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
    CUresult (*cuMemHostRegister)(void*, size_t, unsigned int) = nullptr;
    CUresult (*cuMemHostUnregister)(void*) = nullptr;
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;
    CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream) = nullptr;
    CUresult (*cuMemcpyDtoHAsync)(void*, CUdeviceptr, size_t, CUstream) = nullptr;
    CUresult (*cuMemsetD8Async)(CUdeviceptr, unsigned char, size_t, CUstream) = nullptr;

    CUresult (*cuCtxGetStreamPriorityRange)(int*, int*) = nullptr;
    CUresult (*cuStreamCreate)(CUstream*, unsigned int) = nullptr;
    CUresult (*cuStreamCreateWithPriority)(CUstream*, unsigned int, int) = nullptr;
    CUresult (*cuStreamDestroy)(CUstream) = nullptr;
    CUresult (*cuStreamSynchronize)(CUstream) = nullptr;
    CUresult (*cuStreamWaitEvent)(CUstream, CUevent, unsigned int) = nullptr;
    CUresult (*cuEventCreate)(CUevent*, unsigned int) = nullptr;
    CUresult (*cuEventDestroy)(CUevent) = nullptr;
    CUresult (*cuEventRecord)(CUevent, CUstream) = nullptr;
    CUresult (*cuEventSynchronize)(CUevent) = nullptr;
    CUresult (*cuEventElapsedTime)(float*, CUevent, CUevent) = nullptr;

    std::string err_str(CUresult res) const;
    void check(CUresult res, const char* what) const;

private:
    CudaDriver();
    void* lib_ = nullptr;
};

// Safe process-wide CUDA portable host pinning (`cuMemHostRegister`) even when no `TriPoolVMM` is active.
class HostCudaPinner {
public:
    static HostCudaPinner& instance();

    int register_host(void* ptr, size_t nbytes, int device = -1);
    int unregister_host(void* ptr) noexcept;
    double stream_h2d_chunks_gbps(
        uint64_t dst_dev_ptr,
        const void* src_host,
        size_t nbytes,
        size_t chunk_bytes,
        void* cuda_stream);

private:
    HostCudaPinner() = default;
    std::mutex mu_;
    CUdevice retained_dev_ = 0;
    CUcontext retained_ctx_ = nullptr;
    size_t active_pins_ = 0;
    std::unordered_map<void*, bool> pin_used_retained_ctx_;

    bool ensure_ctx_locked(CudaDriver& drv, int device);
    void release_ctx_if_idle_locked(CudaDriver& drv) noexcept;
};

}  // namespace morphserve::runtime
