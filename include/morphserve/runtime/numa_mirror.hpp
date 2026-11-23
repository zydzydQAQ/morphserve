#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace morphserve::runtime {

// RAII single-process & cross-process NUMA-bound, optionally CUDA-pinned Linux `memfd` weight mirror.
class NumaMemfdMirror {
public:
    static std::unique_ptr<NumaMemfdMirror> create(
        size_t nbytes,
        int numa_node = 0,
        const std::string& name = "morphserve_expert",
        const void* populate_data = nullptr,
        size_t populate_len = 0);

    static std::unique_ptr<NumaMemfdMirror> attach_proc_fd(
        int leader_pid,
        int leader_fd,
        size_t nbytes,
        int numa_node = 0);

    NumaMemfdMirror(const NumaMemfdMirror&) = delete;
    NumaMemfdMirror& operator=(const NumaMemfdMirror&) = delete;
    ~NumaMemfdMirror() noexcept;

    int fd() const noexcept { return fd_; }
    uint64_t inode() const noexcept { return inode_; }
    size_t nbytes() const noexcept { return nbytes_; }
    int numa_node() const noexcept { return numa_node_; }
    bool is_owner() const noexcept { return is_owner_; }
    bool numa_bound() const noexcept { return numa_bound_; }
    bool pinned() const noexcept { return pinned_; }
    bool cuda_pinned() const noexcept { return cuda_pinned_; }

    uint8_t* data() noexcept { return static_cast<uint8_t*>(addr_); }
    const uint8_t* data() const noexcept { return static_cast<const uint8_t*>(addr_); }
    uint64_t ptr_u64() const noexcept { return reinterpret_cast<uint64_t>(addr_); }

    void write_bytes(size_t offset, const void* src, size_t len);
    void read_bytes(size_t offset, void* dst, size_t len) const;

    // Pins host memory via `cuMemHostRegister` when `simulated == false`; marks logical pin when `simulated == true`.
    void pin_cuda(bool simulated = false);
    void unpin_cuda() noexcept;
    double stream_h2d_chunks_gbps(uint64_t dst_dev_ptr, size_t dst_nbytes, size_t chunk_bytes, void* cuda_stream);
    void close() noexcept;

private:
    NumaMemfdMirror() = default;

    int fd_ = -1;
    void* addr_ = nullptr;
    size_t nbytes_ = 0;
    uint64_t inode_ = 0;
    int numa_node_ = 0;
    bool is_owner_ = false;
    bool numa_bound_ = false;
    bool pinned_ = false;
    bool cuda_pinned_ = false;
};

}  // namespace morphserve::runtime
