#include "morphserve/c_api.h"
#include "morphserve/runtime/h2d_pacer.hpp"
#include "morphserve/runtime/numa_mirror.hpp"
#include "morphserve/runtime/peer_transport.hpp"
#include "morphserve/vmm/tripool_vmm.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#define MS_CHECK(cond)                                                                         \
    do {                                                                                       \
        if (!(cond)) {                                                                         \
            std::cerr << "CHECK FAILED at " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::_Exit(1);                                                                     \
        }                                                                                      \
    } while (0)

namespace {

constexpr size_t kMiB = 1ULL << 20;

void test_tripool_vmm_bidirectional_and_session_swap() {
    morphserve::vmm::TriPoolVMMConfig cfg;
    cfg.device = 0;
    cfg.super_pool_mib = 256;   // 4 x 64 MiB
    cfg.fine_kv_pool_mib = 64;  // 32 x 2 MiB -> 320 MiB total
    cfg.super_handle_mib = 64;
    cfg.transient_va_gib = 1;
    cfg.kv_va_gib = 1;
    cfg.allow_overflow = false;
    cfg.simulated = true;
    cfg.materialize_sim_va = true;
    cfg.kv_swap_high_watermark = 0.80;
    cfg.kv_swap_low_watermark = 0.50;

    auto vmm = morphserve::vmm::TriPoolVMM::create(cfg);
    MS_CHECK(vmm->is_simulated());

    auto mirror = morphserve::runtime::NumaMemfdMirror::create(4 * kMiB, 0, "cpp_test_mirror");
    std::vector<uint8_t> pat(4096, 0x5A);
    mirror->write_bytes(0, pat.data(), pat.size());

    const int eid = vmm->expert_register(4, 64, mirror->data(), mirror->nbytes());
    MS_CHECK(eid == 0);
    for (int i = 0; i < 4; ++i) {
        MS_CHECK(vmm->expert_set_layer_used_bytes(eid, i, 4096) == 0);
    }
    MS_CHECK(vmm->expert_load_layers_sync(eid, 0, 4) == 0);
    MS_CHECK(vmm->expert_resident_count(eid) == 4);
    MS_CHECK(vmm->kv_free_bytes(/*for_dit=*/false) == 320 * kMiB);
    MS_CHECK(vmm->kv_free_bytes(/*for_dit=*/true) == 64 * kMiB);

    // Carve all 4 super-handles into 2 MiB fine pages via a 320 MiB KV region
    const uint64_t sid0 = vmm->kv_session_create();
    const int32_t rid0 = vmm->kv_region_alloc(320 * kMiB, sid0);
    MS_CHECK(rid0 >= 0);
    const uint64_t va0 = vmm->kv_region_ensure(rid0, 320 * kMiB, nullptr);
    MS_CHECK(va0 != 0);
    std::memset(reinterpret_cast<void*>(va0), 0x3C, 8192);
    MS_CHECK(vmm->expert_resident_count(eid) == 0);

    // Park sid0 into thinking state and reload all 4 expert layers (forces swap-out + fine->super bundle!)
    const size_t valid_b = 320 * kMiB;
    vmm->kv_session_mark_thinking(sid0, &rid0, &valid_b, 1, /*trigger_watermark=*/false, nullptr);
    MS_CHECK(vmm->expert_set_pinned(eid, 1) == 0);
    MS_CHECK(vmm->expert_load_layers_sync(eid, 0, 4) == 0);
    MS_CHECK(vmm->expert_set_pinned(eid, 0) == 0);
    MS_CHECK(vmm->kv_region_is_swapped(rid0) == 1);

    UnifiedVmmStatsC st{};
    vmm->get_stats(reinterpret_cast<morphserve::vmm::UnifiedVmmStatsPod*>(&st));
    MS_CHECK(st.overflow_creates == 0);
    MS_CHECK(st.kv_swap_outs_total >= 1);
    MS_CHECK(st.fine_to_super_bundles >= 4);

    // Evict experts (4 layers) and restore sid0 at exact same VA with bit-exact contents
    MS_CHECK(vmm->expert_evict_range(eid, 0, 4, nullptr) == 4);
    MS_CHECK(vmm->kv_session_mark_active(sid0, nullptr) == 0);
    MS_CHECK(vmm->kv_region_va(rid0) == va0);
    const auto* p_u8 = reinterpret_cast<const uint8_t*>(va0);
    for (size_t i = 0; i < 8192; ++i) {
        MS_CHECK(p_u8[i] == 0x3C);
    }

    MS_CHECK(vmm->kv_region_free(rid0, nullptr) == 0);
    vmm->kv_session_destroy(sid0);
    MS_CHECK(vmm->shutdown() == 0);
}

void test_tripool_vmm_max_pool_bytes_hard_cap() {
    morphserve::vmm::TriPoolVMMConfig cfg;
    cfg.device = 0;
    cfg.super_pool_mib = 128;   // 2 x 64 MiB
    cfg.fine_kv_pool_mib = 64;  // 32 x 2 MiB -> 192 MiB initial
    cfg.super_handle_mib = 64;
    cfg.transient_va_gib = 1;
    cfg.kv_va_gib = 1;
    cfg.allow_overflow = true;
    cfg.max_pool_bytes = 256 * kMiB;  // Cap overflow at +64 MiB max
    cfg.simulated = true;
    cfg.materialize_sim_va = true;

    auto vmm = morphserve::vmm::TriPoolVMM::create(cfg);
    MS_CHECK(vmm->kv_free_bytes(/*for_dit=*/false) == 256 * kMiB);

    const uint64_t sid = vmm->kv_session_create();
    const int32_t rid = vmm->kv_region_alloc(384 * kMiB, sid);
    MS_CHECK(rid >= 0);

    // Growing to 256 MiB succeeds (+64 MiB overflow, hitting max_pool_bytes exactly)
    MS_CHECK(vmm->kv_region_ensure(rid, 256 * kMiB, nullptr) != 0);
    UnifiedVmmStatsC st{};
    vmm->get_stats(reinterpret_cast<morphserve::vmm::UnifiedVmmStatsPod*>(&st));
    MS_CHECK(st.pool_total_bytes == 256 * kMiB);
    MS_CHECK(vmm->kv_free_bytes(/*for_dit=*/false) == 0);

    // Growing past 256 MiB must fail cleanly (return VA == 0) instead of exceeding max_pool_bytes!
    MS_CHECK(vmm->kv_region_ensure(rid, 258 * kMiB, nullptr) == 0);

    // Super-handle transient allocation must also respect max_pool_bytes (return VA == 0)!
    MS_CHECK(vmm->transient_alloc(64 * kMiB, nullptr) == 0);

    vmm->get_stats(reinterpret_cast<morphserve::vmm::UnifiedVmmStatsPod*>(&st));
    MS_CHECK(st.pool_total_bytes == 256 * kMiB);

    MS_CHECK(vmm->kv_region_free(rid, nullptr) == 0);
    vmm->kv_session_destroy(sid);
    MS_CHECK(vmm->shutdown() == 0);
}

void test_mirror_pacer_and_transports() {
    const char* msg = "SHARED_MEMFD_CPP_OK";
    auto m0 = morphserve::runtime::NumaMemfdMirror::create(64 * 1024, 0, "cpp_mfd_share", msg, std::strlen(msg));
    auto m1 = morphserve::runtime::NumaMemfdMirror::attach_proc_fd(getpid(), m0->fd(), 64 * 1024, 0);
    MS_CHECK(m0->inode() == m1->inode());
    char buf[32] = {};
    m1->read_bytes(0, buf, std::strlen(msg));
    MS_CHECK(std::string(buf) == msg);
    m1->close();
    m0->close();

    // HalfNodeH2DPacer analytical + flock
    morphserve::runtime::HalfNodeH2DPacer pacer({{0, 0}, {1, 0}, {4, 1}}, 1, "/tmp/ms_cpp_test_locks");
    MS_CHECK(pacer.reserve_slot(0, 250.0, 0.0) == 0.0);
    MS_CHECK(pacer.reserve_slot(1, 250.0, 0.0) == 250.0);
    MS_CHECK(pacer.reserve_slot(4, 250.0, 0.0) == 0.0);
    int gid = -1;
    const int lfd = pacer.acquire_process_flock(0, &gid);
    MS_CHECK(lfd >= 0 && gid == 0);
    morphserve::runtime::HalfNodeH2DPacer::release_process_flock(lfd);

    // InMemoryTransportBus + InMemoryPeerAdapter
    auto bus = std::make_shared<morphserve::runtime::InMemoryTransportBus>(2);
    auto e0 = bus->create_endpoint(0);
    auto e1 = bus->create_endpoint(1);
    const std::string wire = "PING_INMEM_TRANSPORT";
    e0->send_bytes(wire.data(), wire.size(), 1);
    std::vector<char> rbuf(wire.size(), 0);
    const auto recv_res = e1->recv_bytes(rbuf.data(), rbuf.size(), 0);
    MS_CHECK(recv_res.first == wire.size());
    MS_CHECK(recv_res.second >= 0.0);
    MS_CHECK(std::string(rbuf.data(), rbuf.size()) == wire);

    // TcpPeerAdapter full-duplex bidirectional exchange
    morphserve::runtime::TcpPeerAdapter t0(0, 2, "127.0.0.1");
    morphserve::runtime::TcpPeerAdapter t1(1, 2, "127.0.0.1");
    t0.set_peer_endpoint(1, "127.0.0.1", t1.bound_port());
    t1.set_peer_endpoint(0, "127.0.0.1", t0.bound_port());

    const std::string payload_01 = "TCP_0_TO_1_PAYLOAD";
    const std::string payload_10 = "TCP_1_TO_0_PAYLOAD";
    std::thread sender([&]() {
        t0.send_bytes(payload_01.data(), payload_01.size(), 1);
        t1.send_bytes(payload_10.data(), payload_10.size(), 0);
    });
    std::vector<char> out_01(payload_01.size(), 0);
    std::vector<char> out_10(payload_10.size(), 0);
    MS_CHECK(t1.recv_bytes(out_01.data(), out_01.size(), 0).first == payload_01.size());
    MS_CHECK(t0.recv_bytes(out_10.data(), out_10.size(), 1).first == payload_10.size());
    sender.join();
    MS_CHECK(std::string(out_01.data(), out_01.size()) == payload_01);
    MS_CHECK(std::string(out_10.data(), out_10.size()) == payload_10);
    t0.close();
    t1.close();
}

}  // namespace

int main() {
    test_tripool_vmm_bidirectional_and_session_swap();
    test_tripool_vmm_max_pool_bytes_hard_cap();
    test_mirror_pacer_and_transports();
    std::cout << "[PASS] test_vmm_and_runtime\n";
    return 0;
}
