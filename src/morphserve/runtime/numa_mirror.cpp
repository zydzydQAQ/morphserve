#include "morphserve/runtime/numa_mirror.hpp"
#include "morphserve/runtime/cuda_driver.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <system_error>
#include <unistd.h>

namespace morphserve::runtime {

namespace {

constexpr size_t kPrefaultStride = 2ULL << 20;  // 2 MiB

int create_linux_memfd(const std::string& name, size_t raw_len_with_check) {
    if (name.size() != raw_len_with_check || name.find('\0') != std::string::npos) {
        throw std::invalid_argument("embedded null character in memfd name");
    }
    using MemfdFn = int (*)(const char*, unsigned int);
    static MemfdFn libc_memfd = reinterpret_cast<MemfdFn>(dlsym(RTLD_DEFAULT, "memfd_create"));
    int fd = -1;
    errno = 0;
    if (libc_memfd) {
        fd = libc_memfd(name.c_str(), 0U);
    } else {
#ifdef SYS_memfd_create
        fd = static_cast<int>(::syscall(SYS_memfd_create, name.c_str(), 0U));
#elif defined(__x86_64__)
        fd = static_cast<int>(::syscall(319L, name.c_str(), 0U));
#elif defined(__aarch64__)
        fd = static_cast<int>(::syscall(279L, name.c_str(), 0U));
#else
        errno = ENOSYS;
#endif
    }
    if (fd < 0) {
        const int err = errno ? errno : EINVAL;
        throw std::system_error(err, std::generic_category(), "memfd_create failed");
    }
    return fd;
}

struct LibNumaApi {
    using AvailFn = int (*)();
    using ToNodeFn = void (*)(void*, size_t, int);

    void* handle = nullptr;
    AvailFn numa_available = nullptr;
    ToNodeFn numa_tonode_memory = nullptr;
    bool available = false;

    LibNumaApi() noexcept {
        handle = dlopen("libnuma.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!handle) return;
        numa_available = reinterpret_cast<AvailFn>(dlsym(handle, "numa_available"));
        numa_tonode_memory = reinterpret_cast<ToNodeFn>(dlsym(handle, "numa_tonode_memory"));
        available = (numa_available != nullptr && numa_tonode_memory != nullptr && numa_available() >= 0);
    }
};

bool try_bind_numa_node(void* ptr, size_t nbytes, int numa_node) noexcept {
    if (!ptr || nbytes == 0 || numa_node < 0) return false;
    static const LibNumaApi api;
    if (!api.available) return false;
    api.numa_tonode_memory(ptr, nbytes, numa_node);
    return true;
}

}  // namespace

std::unique_ptr<NumaMemfdMirror> NumaMemfdMirror::create(
    size_t nbytes,
    int numa_node,
    const std::string& name,
    const void* populate_data,
    size_t populate_len) {
    if (nbytes == 0) {
        throw std::invalid_argument("NumaMemfdMirror nbytes must be > 0");
    }
    if (populate_len > nbytes) {
        throw std::invalid_argument("populate_len exceeds mirror nbytes");
    }

    int fd = create_linux_memfd(name, name.size());
    if (::ftruncate(fd, static_cast<off_t>(nbytes)) != 0) {
        const int err = errno;
        ::close(fd);
        throw std::system_error(err, std::generic_category(), "ftruncate(memfd) failed");
    }

    void* addr = ::mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        const int err = errno;
        ::close(fd);
        throw std::system_error(err, std::generic_category(), "mmap(memfd) failed");
    }

    struct stat st = {};
    if (::fstat(fd, &st) != 0) {
        const int err = errno;
        ::munmap(addr, nbytes);
        ::close(fd);
        throw std::system_error(err, std::generic_category(), "fstat(memfd) failed");
    }

    std::unique_ptr<NumaMemfdMirror> out(new NumaMemfdMirror());
    out->fd_ = fd;
    out->addr_ = addr;
    out->nbytes_ = nbytes;
    out->inode_ = static_cast<uint64_t>(st.st_ino);
    out->numa_node_ = numa_node;
    out->is_owner_ = true;

    // Bind NUMA policy BEFORE first-touch prefaulting/population so Linux mbind(MPOL_BIND, flags=0)
    // governs physical page placement on `numa_node`.
    out->numa_bound_ = try_bind_numa_node(addr, nbytes, numa_node);

    auto* bytes = static_cast<volatile uint8_t*>(addr);
    for (size_t off = 0; off < nbytes; off += kPrefaultStride) {
        bytes[off] = 0;
    }
    if (populate_data && populate_len > 0) {
        std::memcpy(addr, populate_data, populate_len);
    }
    return out;
}

std::unique_ptr<NumaMemfdMirror> NumaMemfdMirror::attach_proc_fd(
    int leader_pid,
    int leader_fd,
    size_t nbytes,
    int numa_node) {
    if (nbytes == 0) {
        throw std::invalid_argument("NumaMemfdMirror attach nbytes must be > 0");
    }
    const std::string proc_path =
        "/proc/" + std::to_string(leader_pid) + "/fd/" + std::to_string(leader_fd);
    const int fd = ::open(proc_path.c_str(), O_RDWR);
    if (fd < 0) {
        throw std::system_error(errno, std::generic_category(), "open(" + proc_path + ") failed");
    }

    void* addr = ::mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        const int err = errno;
        ::close(fd);
        throw std::system_error(err, std::generic_category(), "mmap(" + proc_path + ") failed");
    }

    struct stat st = {};
    if (::fstat(fd, &st) != 0) {
        const int err = errno;
        ::munmap(addr, nbytes);
        ::close(fd);
        throw std::system_error(err, std::generic_category(), "fstat(" + proc_path + ") failed");
    }

    std::unique_ptr<NumaMemfdMirror> out(new NumaMemfdMirror());
    out->fd_ = fd;
    out->addr_ = addr;
    out->nbytes_ = nbytes;
    out->inode_ = static_cast<uint64_t>(st.st_ino);
    out->numa_node_ = numa_node;
    out->is_owner_ = false;
    out->numa_bound_ = false;
    return out;
}

NumaMemfdMirror::~NumaMemfdMirror() noexcept {
    close();
}

void NumaMemfdMirror::write_bytes(size_t offset, const void* src, size_t len) {
    if (!addr_) throw std::runtime_error("NumaMemfdMirror is closed");
    if (offset > nbytes_ || len > nbytes_ - offset) {
        throw std::out_of_range("NumaMemfdMirror::write_bytes out of bounds");
    }
    if (len > 0 && src) {
        std::memcpy(static_cast<uint8_t*>(addr_) + offset, src, len);
    }
}

void NumaMemfdMirror::read_bytes(size_t offset, void* dst, size_t len) const {
    if (!addr_) throw std::runtime_error("NumaMemfdMirror is closed");
    if (offset > nbytes_ || len > nbytes_ - offset) {
        throw std::out_of_range("NumaMemfdMirror::read_bytes out of bounds");
    }
    if (len > 0 && dst) {
        std::memcpy(dst, static_cast<const uint8_t*>(addr_) + offset, len);
    }
}

void NumaMemfdMirror::pin_cuda(bool simulated) {
    if (!addr_) throw std::runtime_error("cannot pin closed NumaMemfdMirror");
    if (cuda_pinned_) return;
    if (simulated) {
        pinned_ = true;
        return;
    }
    if (HostCudaPinner::instance().register_host(addr_, nbytes_) != 0) {
        throw std::runtime_error("cuMemHostRegister failed for NumaMemfdMirror");
    }
    cuda_pinned_ = true;
    pinned_ = true;
}

void NumaMemfdMirror::unpin_cuda() noexcept {
    if (cuda_pinned_ && addr_) {
        HostCudaPinner::instance().unregister_host(addr_);
        cuda_pinned_ = false;
    }
    pinned_ = false;
}

double NumaMemfdMirror::stream_h2d_chunks_gbps(
    uint64_t dst_dev_ptr,
    size_t dst_nbytes,
    size_t chunk_bytes,
    void* cuda_stream) {
    if (!addr_) throw std::runtime_error("NumaMemfdMirror is closed");
    if (!cuda_pinned_) {
        pin_cuda(false);
    }
    const size_t total = std::min(nbytes_, dst_nbytes);
    return HostCudaPinner::instance().stream_h2d_chunks_gbps(
        dst_dev_ptr, addr_, total, chunk_bytes, cuda_stream);
}

void NumaMemfdMirror::close() noexcept {
    unpin_cuda();
    if (addr_) {
        ::munmap(addr_, nbytes_);
        addr_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

}  // namespace morphserve::runtime
