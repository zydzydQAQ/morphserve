#include "morphserve/runtime/h2d_pacer.hpp"
#include "morphserve/runtime/numa_mirror.hpp"
#include "morphserve/runtime/peer_transport.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

int g_failures = 0;

#define MS_EXPECT(cond)                                                                         \
    do {                                                                                        \
        if (!(cond)) {                                                                          \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << ": " #cond "\n";           \
            ++g_failures;                                                                       \
        }                                                                                       \
    } while (0)

#define MS_EXPECT_THROW(expr, ExType)                                                           \
    do {                                                                                        \
        bool caught_expected = false;                                                           \
        try {                                                                                   \
            (void)(expr);                                                                       \
        } catch (const ExType&) {                                                               \
            caught_expected = true;                                                             \
        } catch (const std::exception& ex) {                                                    \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << ": " #expr                 \
                      << " threw unexpected std::exception: " << ex.what() << "\n";             \
            ++g_failures;                                                                       \
        }                                                                                       \
        if (!caught_expected) {                                                                 \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << ": expected " #ExType      \
                      << " from " #expr "\n";                                                   \
            ++g_failures;                                                                       \
        }                                                                                       \
    } while (0)

bool approx_eq(double a, double b, double eps = 1e-6) noexcept {
    return std::fabs(a - b) <= eps;
}

using morphserve::runtime::HalfNodeH2DPacer;
using morphserve::runtime::InMemoryTransportBus;
using morphserve::runtime::NumaMemfdMirror;
using morphserve::runtime::TcpPeerAdapter;
using morphserve::runtime::find_free_tcp_port;

void test_numa_memfd_mirror() {
    constexpr size_t kSize = 256 * 1024;
    std::vector<uint8_t> init_payload(4096);
    for (size_t i = 0; i < init_payload.size(); ++i) {
        init_payload[i] = static_cast<uint8_t>((i * 29U + 7U) & 0xFFU);
    }

    auto m0 = NumaMemfdMirror::create(
        kSize, /*numa_node=*/0, "rt_test_mirror_0", init_payload.data(), init_payload.size());
    MS_EXPECT(m0 != nullptr);
    MS_EXPECT(m0->fd() >= 0);
    MS_EXPECT(m0->inode() != 0);
    MS_EXPECT(m0->nbytes() == kSize);
    MS_EXPECT(m0->numa_node() == 0);
    MS_EXPECT(m0->is_owner());
    MS_EXPECT(m0->ptr_u64() != 0);
    MS_EXPECT(!m0->pinned());
    MS_EXPECT(!m0->cuda_pinned());

    // Populated prefix matches bit-for-bit; prefaulted tail is zero-initialized
    std::vector<uint8_t> verify_head(init_payload.size(), 0);
    m0->read_bytes(0, verify_head.data(), verify_head.size());
    MS_EXPECT(verify_head == init_payload);
    uint8_t tail_sample[64] = {};
    std::memset(tail_sample, 0xFF, sizeof(tail_sample));
    m0->read_bytes(kSize - sizeof(tail_sample), tail_sample, sizeof(tail_sample));
    for (uint8_t b : tail_sample) {
        MS_EXPECT(b == 0);
    }

    // Cross-descriptor /proc/<pid>/fd/<fd> MAP_SHARED attachment shares inode and live mutations
    auto m1 = NumaMemfdMirror::attach_proc_fd(getpid(), m0->fd(), kSize, /*numa_node=*/0);
    MS_EXPECT(m1 != nullptr);
    MS_EXPECT(m1->inode() == m0->inode());
    MS_EXPECT(!m1->is_owner());

    std::vector<uint8_t> m1_head(init_payload.size(), 0);
    m1->read_bytes(0, m1_head.data(), m1_head.size());
    MS_EXPECT(m1_head == init_payload);

    const std::string tag = "BIDIRECTIONAL_MEMFD_VISIBILITY";
    m1->write_bytes(8192, tag.data(), tag.size());
    std::string readback(tag.size(), '\0');
    m0->read_bytes(8192, &readback[0], readback.size());
    MS_EXPECT(readback == tag);

    // Simulated logical pinning and repeated creation (cached libnuma handle)
    m0->pin_cuda(/*simulated=*/true);
    MS_EXPECT(m0->pinned());
    MS_EXPECT(!m0->cuda_pinned());
    m0->unpin_cuda();
    MS_EXPECT(!m0->pinned());

    auto m2 = NumaMemfdMirror::create(64 * 1024, /*numa_node=*/0, "rt_test_mirror_2");
    MS_EXPECT(m2->fd() >= 0);
    MS_EXPECT(m2->inode() != m0->inode());
    MS_EXPECT(m2->numa_bound() == m0->numa_bound());
    m2->close();

    // Bounds & argument validation
    MS_EXPECT_THROW(m0->write_bytes(kSize, tag.data(), 1), std::out_of_range);
    MS_EXPECT_THROW(m0->read_bytes(kSize - 4, &readback[0], 8), std::out_of_range);
    MS_EXPECT_THROW(NumaMemfdMirror::create(0, 0, "zero"), std::invalid_argument);
    MS_EXPECT_THROW(NumaMemfdMirror::create(16, 0, "overflow", tag.data(), 32), std::invalid_argument);
    MS_EXPECT_THROW(NumaMemfdMirror::create(64, 0, std::string("bad\0name", 8)), std::invalid_argument);
    MS_EXPECT_THROW(NumaMemfdMirror::attach_proc_fd(getpid(), m0->fd(), 0, 0), std::invalid_argument);

    m1->close();
    m0->close();
    MS_EXPECT_THROW(m0->read_bytes(0, &readback[0], 1), std::runtime_error);
    MS_EXPECT_THROW(m0->write_bytes(0, tag.data(), 1), std::runtime_error);
}

void test_half_node_h2d_pacer() {
    const std::string lock_dir = "/tmp/morphserve_rt_pacer_test_" + std::to_string(getpid());
    std::error_code ec;
    std::filesystem::remove_all(lock_dir, ec);

    // 1. Single-slot analytical timeline + reset_analytical_slots + out-of-range check
    HalfNodeH2DPacer pacer(
        {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 1}, {5, 1}},
        /*max_concurrent_per_half=*/1,
        lock_dir);
    MS_EXPECT(pacer.max_concurrent_per_half() == 1);
    MS_EXPECT(pacer.group_of(0) == 0);
    MS_EXPECT(pacer.group_of(4) == 1);
    MS_EXPECT_THROW(pacer.group_of(99), std::out_of_range);

    MS_EXPECT(approx_eq(pacer.predicted_wait_ms(0, 0.0), 0.0));
    MS_EXPECT(approx_eq(pacer.reserve_slot(0, 200.0, 10.0), 0.0));     // half 0 busy [10, 210]
    MS_EXPECT(approx_eq(pacer.predicted_wait_ms(1, 50.0), 160.0));     // GPU 1 shares half 0
    MS_EXPECT(approx_eq(pacer.predicted_wait_ms(4, 50.0), 0.0));       // GPU 4 on half 1
    MS_EXPECT(approx_eq(pacer.reserve_slot(1, 100.0, 50.0), 160.0));   // half 0 busy [210, 310]
    MS_EXPECT(approx_eq(pacer.reserve_slot(4, 150.0, 50.0), 0.0));     // half 1 busy [50, 200]
    MS_EXPECT(approx_eq(pacer.predicted_wait_ms(2, 350.0), 0.0));      // past busy_until

    pacer.reset_analytical_slots();
    MS_EXPECT(approx_eq(pacer.predicted_wait_ms(0, 0.0), 0.0));
    MS_EXPECT(approx_eq(pacer.predicted_wait_ms(4, 0.0), 0.0));

    // 2. Dual-concurrency analytical slots
    HalfNodeH2DPacer pacer2({{0, 0}, {1, 0}, {2, 0}}, /*max_concurrent_per_half=*/2, lock_dir);
    MS_EXPECT(approx_eq(pacer2.reserve_slot(0, 100.0, 0.0), 0.0));
    MS_EXPECT(approx_eq(pacer2.reserve_slot(1, 250.0, 0.0), 0.0));
    MS_EXPECT(approx_eq(pacer2.predicted_wait_ms(2, 0.0), 100.0));
    MS_EXPECT(approx_eq(pacer2.reserve_slot(2, 50.0, 0.0), 100.0));

    // 3. Thread slot serialization (max_concurrent = 1)
    std::atomic<int> in_flight{0};
    std::atomic<int> max_in_flight{0};
    std::vector<std::thread> workers;
    workers.reserve(4);
    for (int gpu = 0; gpu < 4; ++gpu) {
        workers.emplace_back([&, gpu]() {
            pacer.acquire_thread_slot(gpu);
            const int cur = in_flight.fetch_add(1, std::memory_order_acq_rel) + 1;
            int prev = max_in_flight.load(std::memory_order_acquire);
            while (cur > prev && !max_in_flight.compare_exchange_weak(prev, cur)) {
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            in_flight.fetch_sub(1, std::memory_order_acq_rel);
            pacer.release_thread_slot(gpu);
        });
    }
    for (auto& t : workers) t.join();
    MS_EXPECT(in_flight.load() == 0);
    MS_EXPECT(max_in_flight.load() == 1);

    // 4. POSIX flock across distinct PCIe half-nodes
    int gid0 = -1;
    int gid1 = -1;
    const int fd0 = pacer.acquire_process_flock(0, &gid0);
    const int fd1 = pacer.acquire_process_flock(4, &gid1);
    MS_EXPECT(fd0 >= 0 && gid0 == 0);
    MS_EXPECT(fd1 >= 0 && gid1 == 1);
    HalfNodeH2DPacer::release_process_flock(fd1);
    HalfNodeH2DPacer::release_process_flock(fd0);

    std::filesystem::remove_all(lock_dir, ec);
}

void test_in_memory_transport_bus_and_adapter() {
    auto bus = std::make_shared<InMemoryTransportBus>(3);
    MS_EXPECT(bus->world_size() == 3);
    auto e0 = bus->create_endpoint(0);
    auto e1 = bus->create_endpoint(1);
    auto e2 = bus->create_endpoint(2);
    MS_EXPECT(e0->rank() == 0 && e0->world_size() == 3);
    MS_EXPECT(e1->rank() == 1 && e2->rank() == 2);

    // Multi-message FIFO ordering including zero-length and 256 KiB binary payload
    std::vector<uint8_t> big(256 * 1024);
    for (size_t i = 0; i < big.size(); ++i) {
        big[i] = static_cast<uint8_t>((i * 53U + 19U) & 0xFFU);
    }
    const std::string small_msg = "INMEM_FIFO_FRAME_2";

    MS_EXPECT(e0->send_bytes(nullptr, 0, 1) >= 0.0);
    MS_EXPECT(e0->send_bytes(big.data(), big.size(), 1) >= 0.0);
    MS_EXPECT(e0->send_bytes(small_msg.data(), small_msg.size(), 1) >= 0.0);
    MS_EXPECT(e2->send_bytes(small_msg.data(), small_msg.size(), 0) >= 0.0);

    uint8_t dummy = 0;
    const auto r_zero = e1->recv_bytes(&dummy, 1, 0);
    MS_EXPECT(r_zero.first == 0);

    std::vector<uint8_t> recv_big(big.size(), 0);
    const auto r_big = e1->recv_bytes(recv_big.data(), recv_big.size(), 0);
    MS_EXPECT(r_big.first == big.size());
    MS_EXPECT(recv_big == big);

    std::string recv_small(small_msg.size(), '\0');
    const auto r_small = e1->recv_bytes(&recv_small[0], recv_small.size(), 0);
    MS_EXPECT(r_small.first == small_msg.size());
    MS_EXPECT(recv_small == small_msg);

    std::string recv_20(small_msg.size(), '\0');
    MS_EXPECT(e0->recv_bytes(&recv_20[0], recv_20.size(), 2).first == small_msg.size());
    MS_EXPECT(recv_20 == small_msg);

    // Named SHM blob store roundtrip
    bus->put_shm("blob_a", big.data(), 1024);
    const auto popped = bus->pop_shm("blob_a");
    MS_EXPECT(popped.size() == 1024);
    MS_EXPECT(std::memcmp(popped.data(), big.data(), 1024) == 0);
    MS_EXPECT(bus->pop_shm("blob_a").empty());

    // Error paths
    MS_EXPECT_THROW(bus->create_endpoint(-1), std::invalid_argument);
    MS_EXPECT_THROW(bus->create_endpoint(3), std::invalid_argument);
    MS_EXPECT_THROW(e0->send_bytes(small_msg.data(), small_msg.size(), 0), std::invalid_argument);
    MS_EXPECT_THROW(bus->pop_message(0, 1, /*timeout_s=*/0.02), std::runtime_error);
}

void test_tcp_peer_adapter_full_duplex_and_errors() {
    const int probe_port = find_free_tcp_port();
    MS_EXPECT(probe_port > 0 && probe_port <= 65535);

    TcpPeerAdapter t0(0, 3, "127.0.0.1");
    TcpPeerAdapter t1(1, 3, "127.0.0.1");
    TcpPeerAdapter t2(2, 3, "127.0.0.1");

    // Error before peer endpoint is registered
    const std::string unreg = "UNREGISTERED";
    MS_EXPECT_THROW(t0.send_bytes(unreg.data(), unreg.size(), 1), std::runtime_error);
    MS_EXPECT_THROW(t0.send_bytes(unreg.data(), unreg.size(), 0), std::invalid_argument);
    MS_EXPECT_THROW(t0.send_bytes(unreg.data(), unreg.size(), 99), std::invalid_argument);
    char scratch[16] = {};
    MS_EXPECT_THROW(t0.recv_bytes(scratch, sizeof(scratch), 0), std::invalid_argument);
    MS_EXPECT_THROW(t0.recv_bytes(scratch, sizeof(scratch), -1), std::invalid_argument);

    // Wire full 3-rank mesh
    t0.set_peer_endpoint(1, "127.0.0.1", t1.bound_port());
    t0.set_peer_endpoint(2, "127.0.0.1", t2.bound_port());
    t1.set_peer_endpoint(0, "127.0.0.1", t0.bound_port());
    t1.set_peer_endpoint(2, "127.0.0.1", t2.bound_port());
    t2.set_peer_endpoint(0, "127.0.0.1", t0.bound_port());
    t2.set_peer_endpoint(1, "127.0.0.1", t1.bound_port());

    std::vector<uint8_t> blob_01(512 * 1024);
    std::vector<uint8_t> blob_12(256 * 1024);
    std::vector<uint8_t> blob_20(128 * 1024);
    for (size_t i = 0; i < blob_01.size(); ++i) blob_01[i] = static_cast<uint8_t>((i * 17U + 3U) & 0xFFU);
    for (size_t i = 0; i < blob_12.size(); ++i) blob_12[i] = static_cast<uint8_t>((i * 31U + 11U) & 0xFFU);
    for (size_t i = 0; i < blob_20.size(); ++i) blob_20[i] = static_cast<uint8_t>((i * 43U + 23U) & 0xFFU);

    // Concurrent full-duplex ring + zero-byte frame
    std::thread s0([&]() {
        t0.send_bytes(nullptr, 0, 1);
        t0.send_bytes(blob_01.data(), blob_01.size(), 1);
    });
    std::thread s1([&]() { t1.send_bytes(blob_12.data(), blob_12.size(), 2); });
    std::thread s2([&]() { t2.send_bytes(blob_20.data(), blob_20.size(), 0); });

    uint8_t zbuf = 0;
    MS_EXPECT(t1.recv_bytes(&zbuf, 1, 0).first == 0);
    std::vector<uint8_t> out_01(blob_01.size(), 0);
    std::vector<uint8_t> out_12(blob_12.size(), 0);
    std::vector<uint8_t> out_20(blob_20.size(), 0);

    const auto r01 = t1.recv_bytes(out_01.data(), out_01.size(), 0);
    const auto r12 = t2.recv_bytes(out_12.data(), out_12.size(), 1);
    const auto r20 = t0.recv_bytes(out_20.data(), out_20.size(), 2);

    s0.join();
    s1.join();
    s2.join();

    MS_EXPECT(r01.first == blob_01.size() && r01.second >= 0.0);
    MS_EXPECT(r12.first == blob_12.size() && r12.second >= 0.0);
    MS_EXPECT(r20.first == blob_20.size() && r20.second >= 0.0);
    MS_EXPECT(out_01 == blob_01);
    MS_EXPECT(out_12 == blob_12);
    MS_EXPECT(out_20 == blob_20);

    // Closing wakes a blocked recv_bytes cleanly with std::runtime_error
    std::thread closer([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        t0.close();
    });
    MS_EXPECT_THROW(t0.recv_bytes(scratch, sizeof(scratch), 1), std::runtime_error);
    closer.join();

    t1.close();
    t2.close();
}

}  // namespace

int main() {
    test_numa_memfd_mirror();
    test_half_node_h2d_pacer();
    test_in_memory_transport_bus_and_adapter();
    test_tcp_peer_adapter_full_duplex_and_errors();

    if (g_failures > 0) {
        std::cerr << "[FAIL] test_runtime_transport (" << g_failures << " check(s) failed)\n";
        return 1;
    }
    std::cout << "[PASS] test_runtime_transport\n";
    return 0;
}
