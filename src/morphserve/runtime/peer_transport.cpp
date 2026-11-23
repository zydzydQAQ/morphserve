#include "morphserve/runtime/peer_transport.hpp"
#include "morphserve/runtime/cuda_driver.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdexcept>
#include <sys/socket.h>
#include <system_error>
#include <unistd.h>

namespace morphserve::runtime {

namespace {

struct RawNcclUniqueId {
    char internal[128];
};

void* open_libnccl(const char* hint_path) {
    if (hint_path && hint_path[0] != '\0') {
        if (void* h = dlopen(hint_path, RTLD_LAZY | RTLD_LOCAL)) return h;
    }
    for (const char* cand : {"libnccl.so.2", "libnccl.so"}) {
        if (void* h = dlopen(cand, RTLD_LAZY | RTLD_LOCAL)) return h;
    }
    const char* err = dlerror();
    throw std::runtime_error(
        std::string("Could not locate libnccl.so.2 for NcclPeerAdapter: ") + (err ? err : "not found"));
}

bool send_all_exact(int fd, const void* buf, size_t len) {
    const auto* p = static_cast<const uint8_t*>(buf);
    size_t sent = 0;
    while (sent < len) {
        const ssize_t n = ::send(fd, p + sent, len - sent, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

bool recv_all_exact(int fd, void* buf, size_t len) {
    auto* p = static_cast<uint8_t*>(buf);
    size_t got = 0;
    while (got < len) {
        const ssize_t n = ::recv(fd, p + got, len - got, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        got += static_cast<size_t>(n);
    }
    return true;
}

uint64_t host_to_be64(uint64_t x) {
    uint64_t out = 0;
    auto* b = reinterpret_cast<uint8_t*>(&out);
    for (int i = 7; i >= 0; --i) {
        b[i] = static_cast<uint8_t>(x & 0xFFU);
        x >>= 8U;
    }
    return out;
}

uint64_t be64_to_host(uint64_t net) {
    const auto* b = reinterpret_cast<const uint8_t*>(&net);
    uint64_t out = 0;
    for (int i = 0; i < 8; ++i) {
        out = (out << 8U) | static_cast<uint64_t>(b[i]);
    }
    return out;
}

}  // namespace

int find_free_tcp_port() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::system_error(errno, std::generic_category(), "socket() failed");
    }
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        const int err = errno;
        ::close(fd);
        throw std::system_error(err, std::generic_category(), "bind(0) failed");
    }
    socklen_t slen = sizeof(addr);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &slen) != 0) {
        const int err = errno;
        ::close(fd);
        throw std::system_error(err, std::generic_category(), "getsockname() failed");
    }
    const int port = ntohs(addr.sin_port);
    ::close(fd);
    return port;
}

// ==================== NcclPeerAdapter ====================

std::vector<uint8_t> NcclPeerAdapter::generate_unique_id(const char* hint_lib_path) {
    void* lib = open_libnccl(hint_lib_path);
    using GetUidFn = int (*)(RawNcclUniqueId*);
    auto fn = reinterpret_cast<GetUidFn>(dlsym(lib, "ncclGetUniqueId"));
    if (!fn) {
        throw std::runtime_error("missing symbol ncclGetUniqueId in libnccl");
    }
    RawNcclUniqueId uid = {};
    const int rc = fn(&uid);
    if (rc != 0) {
        throw std::runtime_error("ncclGetUniqueId failed with ncclResult_t=" + std::to_string(rc));
    }
    return std::vector<uint8_t>(
        reinterpret_cast<const uint8_t*>(uid.internal),
        reinterpret_cast<const uint8_t*>(uid.internal) + kUniqueIdBytes);
}

NcclPeerAdapter::NcclPeerAdapter(
    int rank,
    int world_size,
    const uint8_t* uid_128,
    int device_index,
    const char* hint_lib_path)
    : rank_(rank), world_size_(world_size), device_index_(device_index) {
    if (!uid_128) {
        throw std::invalid_argument("NcclPeerAdapter requires 128-byte unique ID");
    }
    auto& drv = CudaDriver::instance();
    drv.check(drv.cuInit(0), "NcclPeerAdapter cuInit");
    CUdevice dev = 0;
    drv.check(drv.cuDeviceGet(&dev, device_index_), "NcclPeerAdapter cuDeviceGet");
    CUcontext ctx = nullptr;
    drv.check(drv.cuDevicePrimaryCtxRetain(&ctx, dev), "NcclPeerAdapter PrimaryCtxRetain");
    primary_ctx_ = ctx;
    drv.check(drv.cuCtxSetCurrent(ctx), "NcclPeerAdapter cuCtxSetCurrent");

    CUstream st = nullptr;
    drv.check(drv.cuStreamCreate(&st, CU_STREAM_NON_BLOCKING), "NcclPeerAdapter cuStreamCreate");
    transfer_stream_ = st;

    nccl_lib_ = open_libnccl(hint_lib_path);
    using InitRankFn = int (*)(void**, int, RawNcclUniqueId, int);
    auto init_fn = reinterpret_cast<InitRankFn>(dlsym(nccl_lib_, "ncclCommInitRank"));
    nccl_comm_destroy_ = reinterpret_cast<int (*)(void*)>(dlsym(nccl_lib_, "ncclCommDestroy"));
    nccl_send_ = reinterpret_cast<int (*)(const void*, size_t, int, int, void*, void*)>(
        dlsym(nccl_lib_, "ncclSend"));
    nccl_recv_ = reinterpret_cast<int (*)(void*, size_t, int, int, void*, void*)>(
        dlsym(nccl_lib_, "ncclRecv"));

    if (!init_fn || !nccl_comm_destroy_ || !nccl_send_ || !nccl_recv_) {
        close();
        throw std::runtime_error("missing required NCCL symbols in libnccl.so.2");
    }

    RawNcclUniqueId raw_uid = {};
    std::memcpy(raw_uid.internal, uid_128, kUniqueIdBytes);
    const int rc = init_fn(&comm_, world_size_, raw_uid, rank_);
    if (rc != 0) {
        close();
        throw std::runtime_error("ncclCommInitRank failed with ncclResult_t=" + std::to_string(rc));
    }
}

NcclPeerAdapter::~NcclPeerAdapter() noexcept {
    close();
}

double NcclPeerAdapter::send_bytes(
    const void* src_ptr,
    size_t nbytes,
    int dst_rank,
    void* cuda_stream,
    bool sync_stream) {
    auto t0 = std::chrono::steady_clock::now();
    if (nbytes > 0) {
        std::lock_guard<std::mutex> lock(comm_mu_);
        auto& drv = CudaDriver::instance();
        drv.check(drv.cuCtxSetCurrent(reinterpret_cast<CUcontext>(primary_ctx_)), "ncclSend ctx");
        void* eff_stream = cuda_stream ? cuda_stream : transfer_stream_;
        const int rc = nccl_send_(src_ptr, nbytes, /*ncclInt8=*/0, dst_rank, comm_, eff_stream);
        if (rc != 0) {
            throw std::runtime_error("ncclSend failed with ncclResult_t=" + std::to_string(rc));
        }
        if (sync_stream) {
            drv.check(drv.cuStreamSynchronize(reinterpret_cast<CUstream>(eff_stream)), "ncclSend sync");
        }
    }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::pair<size_t, double> NcclPeerAdapter::recv_bytes(
    void* dst_ptr,
    size_t nbytes,
    int src_rank,
    void* cuda_stream,
    bool sync_stream) {
    auto t0 = std::chrono::steady_clock::now();
    if (nbytes > 0) {
        std::lock_guard<std::mutex> lock(comm_mu_);
        auto& drv = CudaDriver::instance();
        drv.check(drv.cuCtxSetCurrent(reinterpret_cast<CUcontext>(primary_ctx_)), "ncclRecv ctx");
        void* eff_stream = cuda_stream ? cuda_stream : transfer_stream_;
        const int rc = nccl_recv_(dst_ptr, nbytes, /*ncclInt8=*/0, src_rank, comm_, eff_stream);
        if (rc != 0) {
            throw std::runtime_error("ncclRecv failed with ncclResult_t=" + std::to_string(rc));
        }
        if (sync_stream) {
            drv.check(drv.cuStreamSynchronize(reinterpret_cast<CUstream>(eff_stream)), "ncclRecv sync");
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return {nbytes, ms};
}

void NcclPeerAdapter::close() noexcept {
    std::lock_guard<std::mutex> lock(comm_mu_);
    try {
        if (comm_ && nccl_comm_destroy_) {
            nccl_comm_destroy_(comm_);
            comm_ = nullptr;
        }
        if (transfer_stream_) {
            CudaDriver::instance().cuStreamDestroy(reinterpret_cast<CUstream>(transfer_stream_));
            transfer_stream_ = nullptr;
        }
        if (primary_ctx_) {
            CUdevice dev = 0;
            if (CudaDriver::instance().cuDeviceGet(&dev, device_index_) == CUDA_SUCCESS) {
                CudaDriver::instance().cuDevicePrimaryCtxRelease(dev);
            }
            primary_ctx_ = nullptr;
        }
    } catch (...) {
    }
}

// ==================== TcpPeerAdapter ====================

TcpPeerAdapter::TcpPeerAdapter(int rank, int world_size, std::string bind_host)
    : rank_(rank), world_size_(world_size), bind_host_(std::move(bind_host)) {
    for (int r = 0; r < world_size_; ++r) {
        if (r != rank_) {
            out_locks_[r] = std::make_unique<std::mutex>();
            inboxes_[r] = std::make_unique<InboxQueue>();
        }
    }
    srv_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (srv_fd_ < 0) {
        throw std::system_error(errno, std::generic_category(), "TcpPeerAdapter socket() failed");
    }
    int one = 1;
    ::setsockopt(srv_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    if (::inet_pton(AF_INET, bind_host_.c_str(), &addr.sin_addr) != 1) {
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    }
    addr.sin_port = 0;
    if (::bind(srv_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        const int err = errno;
        ::close(srv_fd_);
        srv_fd_ = -1;
        throw std::system_error(err, std::generic_category(), "TcpPeerAdapter bind() failed");
    }
    if (::listen(srv_fd_, std::max(16, world_size_ * 2)) != 0) {
        const int err = errno;
        ::close(srv_fd_);
        srv_fd_ = -1;
        throw std::system_error(err, std::generic_category(), "TcpPeerAdapter listen() failed");
    }
    socklen_t slen = sizeof(addr);
    ::getsockname(srv_fd_, reinterpret_cast<sockaddr*>(&addr), &slen);
    bound_port_ = ntohs(addr.sin_port);
    accept_thread_ = std::thread(&TcpPeerAdapter::accept_loop, this);
}

TcpPeerAdapter::~TcpPeerAdapter() noexcept {
    close();
}

void TcpPeerAdapter::set_peer_endpoint(int peer_rank, const std::string& host, int port) {
    std::lock_guard<std::mutex> lock(dir_mu_);
    directory_[peer_rank] = {host, port};
}

void TcpPeerAdapter::accept_loop() {
    while (!closed_.load(std::memory_order_acquire)) {
        sockaddr_in cli = {};
        socklen_t clen = sizeof(cli);
        const int cfd = ::accept(srv_fd_, reinterpret_cast<sockaddr*>(&cli), &clen);
        if (cfd < 0) {
            if (errno == EINTR && !closed_.load(std::memory_order_acquire)) continue;
            break;
        }
        int one = 1;
        ::setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        std::lock_guard<std::mutex> lock(in_mu_);
        if (closed_.load(std::memory_order_acquire)) {
            ::close(cfd);
            break;
        }
        in_socks_.push_back(cfd);
        reader_threads_.emplace_back(&TcpPeerAdapter::peer_reader_loop, this, cfd);
    }
}

void TcpPeerAdapter::peer_reader_loop(int conn_fd) {
    uint32_t net_src = 0;
    if (!recv_all_exact(conn_fd, &net_src, sizeof(net_src))) return;
    const int src_rank = static_cast<int>(ntohl(net_src));
    auto it = inboxes_.find(src_rank);
    if (it == inboxes_.end()) return;
    InboxQueue& inbox = *it->second;

    while (!closed_.load(std::memory_order_acquire)) {
        uint64_t net_len = 0;
        if (!recv_all_exact(conn_fd, &net_len, sizeof(net_len))) break;
        const size_t nbytes = static_cast<size_t>(be64_to_host(net_len));
        std::vector<uint8_t> buf(nbytes);
        if (nbytes > 0 && !recv_all_exact(conn_fd, buf.data(), nbytes)) break;
        {
            std::lock_guard<std::mutex> lock(inbox.mu);
            inbox.msgs.push_back(std::move(buf));
        }
        inbox.cv.notify_one();
    }
}

double TcpPeerAdapter::send_bytes(
    const void* src_ptr,
    size_t nbytes,
    int dst_rank,
    void* /*cuda_stream*/,
    bool /*sync_stream*/) {
    auto lock_it = out_locks_.find(dst_rank);
    if (lock_it == out_locks_.end()) {
        throw std::invalid_argument("invalid dst_rank=" + std::to_string(dst_rank));
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(*lock_it->second);
    int sock = -1;
    auto sit = out_socks_.find(dst_rank);
    if (sit != out_socks_.end()) {
        sock = sit->second;
    } else {
        std::pair<std::string, int> target;
        {
            std::lock_guard<std::mutex> dlock(dir_mu_);
            auto dit = directory_.find(dst_rank);
            if (dit == directory_.end()) {
                throw std::runtime_error("TcpPeerAdapter directory missing rank " + std::to_string(dst_rank));
            }
            target = dit->second;
        }
        sock = ::socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            throw std::system_error(errno, std::generic_category(), "TcpPeerAdapter connect socket() failed");
        }
        int one = 1;
        ::setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        if (::inet_pton(AF_INET, target.first.c_str(), &addr.sin_addr) != 1) {
            ::close(sock);
            throw std::invalid_argument("invalid IPv4 host: " + target.first);
        }
        addr.sin_port = htons(static_cast<uint16_t>(target.second));
        if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            const int err = errno;
            ::close(sock);
            throw std::system_error(err, std::generic_category(), "TcpPeerAdapter connect() failed");
        }
        const uint32_t net_rank = htonl(static_cast<uint32_t>(rank_));
        if (!send_all_exact(sock, &net_rank, sizeof(net_rank))) {
            ::close(sock);
            throw std::runtime_error("TcpPeerAdapter handshake send failed");
        }
        out_socks_[dst_rank] = sock;
    }

    const uint64_t net_len = host_to_be64(static_cast<uint64_t>(nbytes));
    if (!send_all_exact(sock, &net_len, sizeof(net_len)) ||
        (nbytes > 0 && !send_all_exact(sock, src_ptr, nbytes))) {
        throw std::runtime_error("TcpPeerAdapter frame send failed");
    }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::pair<size_t, double> TcpPeerAdapter::recv_bytes(
    void* dst_ptr,
    size_t nbytes,
    int src_rank,
    void* /*cuda_stream*/,
    bool /*sync_stream*/) {
    auto it = inboxes_.find(src_rank);
    if (it == inboxes_.end()) {
        throw std::invalid_argument("invalid src_rank=" + std::to_string(src_rank));
    }
    const auto t0 = std::chrono::steady_clock::now();
    InboxQueue& inbox = *it->second;
    std::vector<uint8_t> msg;
    {
        std::unique_lock<std::mutex> lock(inbox.mu);
        if (!inbox.cv.wait_for(lock, std::chrono::seconds(60), [&] {
                return !inbox.msgs.empty() || closed_.load(std::memory_order_acquire);
            })) {
            throw std::runtime_error("TcpPeerAdapter recv_bytes timed out");
        }
        if (inbox.msgs.empty()) {
            throw std::runtime_error("TcpPeerAdapter closed while waiting in recv_bytes");
        }
        msg = std::move(inbox.msgs.front());
        inbox.msgs.pop_front();
    }
    const size_t copy_len = std::min(nbytes, msg.size());
    if (copy_len > 0 && dst_ptr) {
        std::memcpy(dst_ptr, msg.data(), copy_len);
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return {copy_len, ms};
}

void TcpPeerAdapter::close() noexcept {
    if (closed_.exchange(true, std::memory_order_acq_rel)) return;
    if (srv_fd_ >= 0) {
        ::shutdown(srv_fd_, SHUT_RDWR);
        ::close(srv_fd_);
        srv_fd_ = -1;
    }
    for (auto& kv : out_locks_) {
        std::lock_guard<std::mutex> lock(*kv.second);
        auto sit = out_socks_.find(kv.first);
        if (sit != out_socks_.end() && sit->second >= 0) {
            ::shutdown(sit->second, SHUT_RDWR);
            ::close(sit->second);
        }
    }
    out_socks_.clear();
    {
        std::lock_guard<std::mutex> lock(in_mu_);
        for (int fd : in_socks_) {
            if (fd >= 0) {
                ::shutdown(fd, SHUT_RDWR);
                ::close(fd);
            }
        }
        in_socks_.clear();
    }
    for (auto& kv : inboxes_) {
        kv.second->cv.notify_all();
    }
    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }
    for (auto& t : reader_threads_) {
        if (t.joinable()) t.join();
    }
    reader_threads_.clear();
}

// ==================== InMemoryTransportBus & InMemoryPeerAdapter ====================

InMemoryTransportBus::InMemoryTransportBus(int world_size)
    : world_size_(std::max(1, world_size)),
      pipes_(static_cast<size_t>(world_size_ * world_size_)) {
    for (auto& p : pipes_) {
        p = std::make_unique<PipeQueue>();
    }
}

std::unique_ptr<PeerByteTransport> InMemoryTransportBus::create_endpoint(int rank) {
    return std::make_unique<InMemoryPeerAdapter>(shared_from_this(), rank);
}

void InMemoryTransportBus::push_message(int src_rank, int dst_rank, const void* data, size_t nbytes) {
    if (src_rank < 0 || src_rank >= world_size_ || dst_rank < 0 || dst_rank >= world_size_ || src_rank == dst_rank) {
        throw std::invalid_argument("invalid InMemoryTransportBus rank pair");
    }
    PipeQueue& pq = *pipes_[static_cast<size_t>(src_rank * world_size_ + dst_rank)];
    std::vector<uint8_t> buf(nbytes);
    if (nbytes > 0 && data) {
        std::memcpy(buf.data(), data, nbytes);
    }
    {
        std::lock_guard<std::mutex> lock(pq.mu);
        pq.q.push_back(std::move(buf));
    }
    pq.cv.notify_one();
}

std::vector<uint8_t> InMemoryTransportBus::pop_message(int src_rank, int dst_rank, double timeout_s) {
    if (src_rank < 0 || src_rank >= world_size_ || dst_rank < 0 || dst_rank >= world_size_ || src_rank == dst_rank) {
        throw std::invalid_argument("invalid InMemoryTransportBus rank pair");
    }
    PipeQueue& pq = *pipes_[static_cast<size_t>(src_rank * world_size_ + dst_rank)];
    std::unique_lock<std::mutex> lock(pq.mu);
    if (!pq.cv.wait_for(lock, std::chrono::duration<double>(timeout_s), [&] { return !pq.q.empty(); })) {
        throw std::runtime_error("InMemoryTransportBus pop_message timed out");
    }
    std::vector<uint8_t> out = std::move(pq.q.front());
    pq.q.pop_front();
    return out;
}

void InMemoryTransportBus::put_shm(const std::string& name, const void* data, size_t nbytes) {
    std::vector<uint8_t> buf(nbytes);
    if (nbytes > 0 && data) {
        std::memcpy(buf.data(), data, nbytes);
    }
    std::lock_guard<std::mutex> lock(shm_mu_);
    shm_blobs_[name] = std::move(buf);
}

std::vector<uint8_t> InMemoryTransportBus::pop_shm(const std::string& name) {
    std::lock_guard<std::mutex> lock(shm_mu_);
    auto it = shm_blobs_.find(name);
    if (it == shm_blobs_.end()) return {};
    std::vector<uint8_t> out = std::move(it->second);
    shm_blobs_.erase(it);
    return out;
}

InMemoryPeerAdapter::InMemoryPeerAdapter(std::shared_ptr<InMemoryTransportBus> bus, int rank)
    : bus_(std::move(bus)), rank_(rank) {}

double InMemoryPeerAdapter::send_bytes(
    const void* src_ptr,
    size_t nbytes,
    int dst_rank,
    void* /*cuda_stream*/,
    bool /*sync_stream*/) {
    bus_->push_message(rank_, dst_rank, src_ptr, nbytes);
    return 0.15;
}

std::pair<size_t, double> InMemoryPeerAdapter::recv_bytes(
    void* dst_ptr,
    size_t nbytes,
    int src_rank,
    void* /*cuda_stream*/,
    bool /*sync_stream*/) {
    std::vector<uint8_t> raw = bus_->pop_message(src_rank, rank_);
    const size_t copy_len = std::min(nbytes, raw.size());
    if (copy_len > 0 && dst_ptr) {
        std::memcpy(dst_ptr, raw.data(), copy_len);
    }
    return {copy_len, 0.15};
}

}  // namespace morphserve::runtime
