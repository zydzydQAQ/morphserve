#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace morphserve::vmm {

// Abstract physical/virtual page-mapping seam beneath `TriPoolVMM`.
// Two concrete adapters: `CudaPageAdapter` (`libcuda.so.1` cuMemMap) and `HostMmapPageAdapter` (POSIX mmap).
class PageBackend {
public:
    virtual ~PageBackend() = default;

    virtual bool is_simulated() const noexcept = 0;
    virtual int device() const noexcept = 0;
    virtual size_t min_granularity_bytes() const noexcept = 0;

    virtual void ensure_context() = 0;
    virtual void sync_context() = 0;

    virtual uint64_t reserve_va(size_t bytes, size_t alignment) = 0;
    virtual void free_va(uint64_t va, size_t bytes) noexcept = 0;

    virtual uint64_t create_phys_handle(size_t bytes) = 0;
    virtual void release_phys_handle(uint64_t handle, size_t bytes) noexcept = 0;

    virtual void map_range(uint64_t va, size_t bytes, uint64_t handle, bool need_host_access = true) = 0;
    virtual void unmap_range(uint64_t va, size_t bytes, bool was_host_accessible = true) = 0;
    virtual void set_access_rw(uint64_t va, size_t bytes, bool need_host_access = true) = 0;

    virtual void* create_stream(bool high_priority) = 0;
    virtual void destroy_stream(void* stream) noexcept = 0;
    virtual void sync_stream(void* stream) = 0;

    virtual void* create_event() = 0;
    virtual void destroy_event(void* ev) noexcept = 0;
    virtual void record_event(void* ev, void* stream) = 0;
    virtual void stream_wait_event(void* stream, void* ev) = 0;
    virtual void sync_event(void* ev) = 0;
    virtual bool sync_event_noexcept(void* ev) noexcept = 0;

    virtual void* alloc_host_pinned(size_t bytes) = 0;
    virtual void free_host_pinned(void* ptr) noexcept = 0;

    virtual void memcpy_h2d_async(uint64_t dst_va, const void* src_host, size_t bytes, void* stream) = 0;
    virtual void memcpy_d2h_async(void* dst_host, uint64_t src_va, size_t bytes, void* stream) = 0;
};

class CudaPageAdapter final : public PageBackend {
public:
    explicit CudaPageAdapter(int device);
    ~CudaPageAdapter() noexcept override;

    bool is_simulated() const noexcept override { return false; }
    int device() const noexcept override { return device_; }
    size_t min_granularity_bytes() const noexcept override { return min_gran_; }

    void ensure_context() override;
    void sync_context() override;

    uint64_t reserve_va(size_t bytes, size_t alignment) override;
    void free_va(uint64_t va, size_t bytes) noexcept override;

    uint64_t create_phys_handle(size_t bytes) override;
    void release_phys_handle(uint64_t handle, size_t bytes) noexcept override;

    void map_range(uint64_t va, size_t bytes, uint64_t handle, bool need_host_access = true) override;
    void unmap_range(uint64_t va, size_t bytes, bool was_host_accessible = true) override;
    void set_access_rw(uint64_t va, size_t bytes, bool need_host_access = true) override;

    void* create_stream(bool high_priority) override;
    void destroy_stream(void* stream) noexcept override;
    void sync_stream(void* stream) override;

    void* create_event() override;
    void destroy_event(void* ev) noexcept override;
    void record_event(void* ev, void* stream) override;
    void stream_wait_event(void* stream, void* ev) override;
    void sync_event(void* ev) override;
    bool sync_event_noexcept(void* ev) noexcept override;

    void* alloc_host_pinned(size_t bytes) override;
    void free_host_pinned(void* ptr) noexcept override;

    void memcpy_h2d_async(uint64_t dst_va, const void* src_host, size_t bytes, void* stream) override;
    void memcpy_d2h_async(void* dst_host, uint64_t src_va, size_t bytes, void* stream) override;

private:
    int device_ = 0;
    int cu_dev_ = 0;
    void* primary_ctx_ = nullptr;
    size_t min_gran_ = 2ULL << 20;
};

class HostMmapPageAdapter final : public PageBackend {
public:
    explicit HostMmapPageAdapter(int device = 0, size_t min_gran_bytes = (2ULL << 20));
    ~HostMmapPageAdapter() noexcept override;

    bool is_simulated() const noexcept override { return true; }
    int device() const noexcept override { return device_; }
    size_t min_granularity_bytes() const noexcept override { return min_gran_; }

    void ensure_context() override {}
    void sync_context() override {}

    uint64_t reserve_va(size_t bytes, size_t alignment) override;
    void free_va(uint64_t va, size_t bytes) noexcept override;

    uint64_t create_phys_handle(size_t bytes) override;
    void release_phys_handle(uint64_t handle, size_t bytes) noexcept override;

    void map_range(uint64_t va, size_t bytes, uint64_t handle, bool need_host_access = true) override;
    void unmap_range(uint64_t va, size_t bytes, bool was_host_accessible = true) override;
    void set_access_rw(uint64_t va, size_t bytes, bool need_host_access = true) override;

    void* create_stream(bool high_priority) override;
    void destroy_stream(void* stream) noexcept override;
    void sync_stream(void* stream) override;

    void* create_event() override;
    void destroy_event(void* ev) noexcept override;
    void record_event(void* ev, void* stream) override;
    void stream_wait_event(void* stream, void* ev) override;
    void sync_event(void* ev) override;
    bool sync_event_noexcept(void* ev) noexcept override;

    void* alloc_host_pinned(size_t bytes) override;
    void free_host_pinned(void* ptr) noexcept override;

    void memcpy_h2d_async(uint64_t dst_va, const void* src_host, size_t bytes, void* stream) override;
    void memcpy_d2h_async(void* dst_host, uint64_t src_va, size_t bytes, void* stream) override;

private:
    int device_ = 0;
    size_t min_gran_ = 2ULL << 20;
    std::mutex mu_;
    uint64_t next_id_ = 1;
    std::unordered_map<uint64_t, size_t> reserved_vas_;
};

}  // namespace morphserve::vmm
