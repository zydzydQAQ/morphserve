#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace morphserve::runtime {

int find_free_tcp_port();

// Unified point-to-point byte transport seam (`send_bytes` / `recv_bytes`).
// Three adapters: `NcclPeerAdapter` (GPU P2P), `TcpPeerAdapter` (multi-node/CPU TCP), `InMemoryPeerAdapter` (in-proc).
class PeerByteTransport {
public:
    virtual ~PeerByteTransport() = default;
    virtual int rank() const noexcept = 0;
    virtual int world_size() const noexcept = 0;

    // Sends `nbytes` from `src_ptr` to `dst_rank`; returns elapsed milliseconds.
    virtual double send_bytes(
        const void* src_ptr,
        size_t nbytes,
        int dst_rank,
        void* cuda_stream = nullptr,
        bool sync_stream = true) = 0;

    // Receives up to `nbytes` from `src_rank` into `dst_ptr`; returns `{actual_bytes, elapsed_ms}`.
    virtual std::pair<size_t, double> recv_bytes(
        void* dst_ptr,
        size_t nbytes,
        int src_rank,
        void* cuda_stream = nullptr,
        bool sync_stream = true) = 0;

    virtual void close() noexcept = 0;
};

// Adapter 1: Standalone N-rank `libnccl.so.2` P2P byte communicator on a dedicated CUDA stream.
class NcclPeerAdapter final : public PeerByteTransport {
public:
    static constexpr size_t kUniqueIdBytes = 128;
    static std::vector<uint8_t> generate_unique_id(const char* hint_lib_path = nullptr);

    NcclPeerAdapter(
        int rank,
        int world_size,
        const uint8_t* uid_128,
        int device_index,
        const char* hint_lib_path = nullptr);
    ~NcclPeerAdapter() noexcept override;

    int rank() const noexcept override { return rank_; }
    int world_size() const noexcept override { return world_size_; }
    uint64_t transfer_stream_u64() const noexcept { return reinterpret_cast<uint64_t>(transfer_stream_); }

    double send_bytes(
        const void* src_ptr,
        size_t nbytes,
        int dst_rank,
        void* cuda_stream = nullptr,
        bool sync_stream = true) override;

    std::pair<size_t, double> recv_bytes(
        void* dst_ptr,
        size_t nbytes,
        int src_rank,
        void* cuda_stream = nullptr,
        bool sync_stream = true) override;

    void close() noexcept override;

private:
    int rank_ = 0;
    int world_size_ = 1;
    int device_index_ = 0;
    void* nccl_lib_ = nullptr;
    void* comm_ = nullptr;
    void* primary_ctx_ = nullptr;
    void* transfer_stream_ = nullptr;
    std::mutex comm_mu_;

    int (*nccl_comm_destroy_)(void*) = nullptr;
    int (*nccl_send_)(const void*, size_t, int, int, void*, void*) = nullptr;
    int (*nccl_recv_)(void*, size_t, int, int, void*, void*) = nullptr;
};

// Adapter 2: Full-duplex POSIX `TCP_NODELAY` framed P2P socket transport.
class TcpPeerAdapter final : public PeerByteTransport {
public:
    TcpPeerAdapter(int rank, int world_size, std::string bind_host = "127.0.0.1");
    ~TcpPeerAdapter() noexcept override;

    int rank() const noexcept override { return rank_; }
    int world_size() const noexcept override { return world_size_; }
    int bound_port() const noexcept { return bound_port_; }

    void set_peer_endpoint(int peer_rank, const std::string& host, int port);

    double send_bytes(
        const void* src_ptr,
        size_t nbytes,
        int dst_rank,
        void* cuda_stream = nullptr,
        bool sync_stream = true) override;

    std::pair<size_t, double> recv_bytes(
        void* dst_ptr,
        size_t nbytes,
        int src_rank,
        void* cuda_stream = nullptr,
        bool sync_stream = true) override;

    void close() noexcept override;

private:
    struct InboxQueue {
        std::mutex mu;
        std::condition_variable cv;
        std::deque<std::vector<uint8_t>> msgs;
    };

    void accept_loop();
    void peer_reader_loop(int conn_fd);

    int rank_ = 0;
    int world_size_ = 1;
    std::string bind_host_;
    int srv_fd_ = -1;
    int bound_port_ = 0;
    std::atomic<bool> closed_{false};

    std::mutex dir_mu_;
    std::unordered_map<int, std::pair<std::string, int>> directory_;
    std::unordered_map<int, std::unique_ptr<std::mutex>> out_locks_;
    std::unordered_map<int, int> out_socks_;

    std::unordered_map<int, std::unique_ptr<InboxQueue>> inboxes_;
    std::mutex in_mu_;
    std::vector<int> in_socks_;
    std::thread accept_thread_;
    std::vector<std::thread> reader_threads_;
};

// Shared in-process pairwise FIFO channel + named SHM blob store for `InMemoryPeerAdapter`.
class InMemoryTransportBus : public std::enable_shared_from_this<InMemoryTransportBus> {
public:
    explicit InMemoryTransportBus(int world_size);

    int world_size() const noexcept { return world_size_; }
    std::unique_ptr<PeerByteTransport> create_endpoint(int rank);

    void push_message(int src_rank, int dst_rank, const void* data, size_t nbytes);
    std::vector<uint8_t> pop_message(int src_rank, int dst_rank, double timeout_s = 60.0);

    void put_shm(const std::string& name, const void* data, size_t nbytes);
    std::vector<uint8_t> pop_shm(const std::string& name);

private:
    struct PipeQueue {
        std::mutex mu;
        std::condition_variable cv;
        std::deque<std::vector<uint8_t>> q;
    };

    int world_size_ = 1;
    std::vector<std::unique_ptr<PipeQueue>> pipes_;
    std::mutex shm_mu_;
    std::unordered_map<std::string, std::vector<uint8_t>> shm_blobs_;
};

class InMemoryPeerAdapter final : public PeerByteTransport {
public:
    InMemoryPeerAdapter(std::shared_ptr<InMemoryTransportBus> bus, int rank);
    ~InMemoryPeerAdapter() noexcept override = default;

    int rank() const noexcept override { return rank_; }
    int world_size() const noexcept override { return bus_->world_size(); }

    double send_bytes(
        const void* src_ptr,
        size_t nbytes,
        int dst_rank,
        void* cuda_stream = nullptr,
        bool sync_stream = true) override;

    std::pair<size_t, double> recv_bytes(
        void* dst_ptr,
        size_t nbytes,
        int src_rank,
        void* cuda_stream = nullptr,
        bool sync_stream = true) override;

    void close() noexcept override {}

private:
    std::shared_ptr<InMemoryTransportBus> bus_;
    int rank_ = 0;
};

}  // namespace morphserve::runtime
