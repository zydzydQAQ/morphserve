#include "morphserve/runtime/numa_mirror.hpp"
#include "morphserve/vmm/tripool_vmm.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

constexpr size_t kMiB = 1ULL << 20;

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

uint8_t pattern_byte(size_t tag, size_t idx) noexcept {
    return static_cast<uint8_t>(((idx * 131U) ^ (tag * 37U) ^ (idx >> 8U)) & 0xFFU);
}

void fill_pattern(void* dst, size_t len, size_t tag) {
    auto* p = static_cast<uint8_t*>(dst);
    for (size_t i = 0; i < len; ++i) {
        p[i] = pattern_byte(tag, i);
    }
}

bool verify_pattern(const void* src, size_t len, size_t tag) {
    const auto* p = static_cast<const uint8_t*>(src);
    for (size_t i = 0; i < len; ++i) {
        if (p[i] != pattern_byte(tag, i)) {
            return false;
        }
    }
    return true;
}

using morphserve::runtime::NumaMemfdMirror;
using morphserve::vmm::TriPoolVMM;
using morphserve::vmm::TriPoolVMMConfig;
using morphserve::vmm::UnifiedVmmStatsPod;
using morphserve::vmm::VmmOutOfMemoryError;

void test_pool_conservation_carving_bundling_and_coalesce() {
    TriPoolVMMConfig cfg;
    cfg.device = 0;
    cfg.super_pool_mib = 192;   // 3 x 64 MiB super handles
    cfg.fine_kv_pool_mib = 64;  // 32 x 2 MiB fine pages -> 256 MiB total
    cfg.super_handle_mib = 64;
    cfg.transient_va_gib = 1;
    cfg.kv_va_gib = 1;
    cfg.allow_overflow = false;
    cfg.simulated = true;
    cfg.materialize_sim_va = false;

    auto vmm = TriPoolVMM::create(cfg);
    MS_EXPECT(vmm->is_simulated());

    const int eid = vmm->expert_register(3, 64, nullptr, 0);
    MS_EXPECT(eid == 0);
    MS_EXPECT(vmm->expert_load_layers_sync(eid, 0, 3) == 0);
    MS_EXPECT(vmm->expert_resident_count(eid) == 3);
    MS_EXPECT(vmm->kv_free_bytes(/*for_dit=*/false) == 256 * kMiB);
    MS_EXPECT(vmm->kv_free_bytes(/*for_dit=*/true) == 64 * kMiB);

    // Grow KV across full 256 MiB: evicts all 3 unpinned expert layers and carves 3 x 64 MiB -> 96 x 2 MiB
    const uint64_t sid = vmm->kv_session_create();
    const int32_t rid = vmm->kv_region_alloc(256 * kMiB, sid);
    MS_EXPECT(rid >= 0);
    MS_EXPECT(vmm->kv_region_ensure(rid, 256 * kMiB, nullptr) != 0);
    MS_EXPECT(vmm->expert_resident_count(eid) == 0);

    UnifiedVmmStatsPod st{};
    vmm->get_stats(&st);
    MS_EXPECT(st.pool_total_bytes == 256 * kMiB);
    MS_EXPECT(st.super_to_fine_carves == 3);
    MS_EXPECT(st.super_pool_total_handles == 0);
    MS_EXPECT(st.fine_pool_total_pages == 128);
    MS_EXPECT(st.kv_mapped_pages_current == 128);
    MS_EXPECT(st.overflow_creates == 0);

    // Free KV -> fine pages remain warm (no eager coalescing on free)
    MS_EXPECT(vmm->kv_region_free(rid, nullptr) == 0);
    vmm->get_stats(&st);
    MS_EXPECT(st.fine_pool_free_pages == 128);
    MS_EXPECT(st.super_pool_total_handles == 0);
    MS_EXPECT(st.pool_total_bytes == 256 * kMiB);

    // Bounded coalesce_fine_to_super(1) converts exactly 32 fine pages into 1 super handle
    MS_EXPECT(vmm->coalesce_fine_to_super(1) == 1);
    vmm->get_stats(&st);
    MS_EXPECT(st.super_pool_total_handles == 1);
    MS_EXPECT(st.fine_pool_total_pages == 96);
    MS_EXPECT(st.pool_total_bytes == 256 * kMiB);

    // Unlimited coalesce_fine_to_super(0) converts remaining 2 excess super handles and preserves base_fine_pages_
    MS_EXPECT(vmm->coalesce_fine_to_super(0) == 2);
    vmm->get_stats(&st);
    MS_EXPECT(st.super_pool_total_handles == 3);
    MS_EXPECT(st.fine_pool_total_pages == 32);
    MS_EXPECT(st.fine_pool_free_pages == 32);
    MS_EXPECT(st.pool_total_bytes == 256 * kMiB);
    MS_EXPECT(vmm->coalesce_fine_to_super(0) == 0);

    // Pin all 3 expert layers on super handles; 64 MiB transient scratchpad bundles the 32 base fine pages
    MS_EXPECT(vmm->expert_load_layers_sync(eid, 0, 3) == 0);
    MS_EXPECT(vmm->expert_set_pinned(eid, 1) == 0);
    vmm->scope_begin();
    void* tr = vmm->transient_alloc(64 * kMiB, nullptr);
    MS_EXPECT(tr != nullptr);
    vmm->get_stats(&st);
    MS_EXPECT(st.fine_to_super_bundles == 1);
    MS_EXPECT(st.transient_fine_bundles_current == 1);
    MS_EXPECT(st.fine_pool_free_pages == 0);
    MS_EXPECT(st.pool_total_bytes == 256 * kMiB);

    vmm->transient_free(tr, 64 * kMiB, nullptr);
    MS_EXPECT(vmm->scope_end(nullptr) == 0);
    vmm->get_stats(&st);
    MS_EXPECT(st.transient_fine_bundles_current == 0);
    MS_EXPECT(st.fine_pool_free_pages == 32);
    MS_EXPECT(st.pool_total_bytes == 256 * kMiB);

    MS_EXPECT(vmm->expert_set_pinned(eid, 0) == 0);
    vmm->kv_session_destroy(sid);
    MS_EXPECT(vmm->shutdown() == 0);
}

void test_strict_cap_bundling_carve_before_overflow_and_no_phantom_overflow() {
    // 1. Strict-cap (allow_overflow=false) preserves fine-to-super bundling without eager coalescing
    //    (Module 1 Test 5 contract: fine_to_super_bundles > 0, overflow_creates == 0, explicit coalesce still works).
    {
        TriPoolVMMConfig cfg;
        cfg.device = 0;
        cfg.super_pool_mib = 256;   // 4 x 64 MiB
        cfg.fine_kv_pool_mib = 64;  // 32 x 2 MiB (base_fine_pages_ = 32)
        cfg.super_handle_mib = 64;
        cfg.transient_va_gib = 1;
        cfg.kv_va_gib = 1;
        cfg.allow_overflow = false;
        cfg.max_pool_bytes = 320 * kMiB;
        cfg.simulated = true;
        cfg.materialize_sim_va = false;

        auto vmm = TriPoolVMM::create(cfg);
        const uint64_t sid_a = vmm->kv_session_create();
        const uint64_t sid_b = vmm->kv_session_create();
        const int32_t rid_a = vmm->kv_region_alloc(64 * kMiB, sid_a);
        const int32_t rid_b = vmm->kv_region_alloc(192 * kMiB, sid_b);
        MS_EXPECT(vmm->kv_region_ensure(rid_a, 64 * kMiB, nullptr) != 0);
        MS_EXPECT(vmm->kv_region_ensure(rid_b, 192 * kMiB, nullptr) != 0);

        // Session B carved 3 super handles -> super_pool_total=1, fine_pool_total=128
        UnifiedVmmStatsPod st{};
        vmm->get_stats(&st);
        MS_EXPECT(st.super_to_fine_carves == 3);
        MS_EXPECT(st.super_pool_total_handles == 1);
        MS_EXPECT(st.fine_pool_total_pages == 128);
        MS_EXPECT(st.pool_total_bytes == 320 * kMiB);

        // Free Session B -> 96 free fine pages (>= base_fine_pages_ + sub_pages_ = 64)
        MS_EXPECT(vmm->kv_region_free(rid_b, nullptr) == 0);

        // 192 MiB transient allocation takes 1 free super handle + 2 fine-to-super bundles (NO eager coalesce / thrashing)
        vmm->scope_begin();
        void* tr = vmm->transient_alloc(192 * kMiB, nullptr);
        MS_EXPECT(tr != nullptr);
        vmm->get_stats(&st);
        MS_EXPECT(st.fine_to_super_bundles == 2);
        MS_EXPECT(st.transient_fine_bundles_current == 2);
        MS_EXPECT(st.overflow_creates == 0);
        MS_EXPECT(st.pool_total_bytes == 320 * kMiB);

        vmm->transient_free(tr, 192 * kMiB, nullptr);
        MS_EXPECT(vmm->scope_end(nullptr) == 0);

        // Explicit coalesce_fine_to_super still works cleanly under strict cap
        MS_EXPECT(vmm->coalesce_fine_to_super(1) == 1);
        vmm->get_stats(&st);
        MS_EXPECT(st.super_pool_total_handles == 2);
        MS_EXPECT(st.fine_pool_total_pages == 96);
        MS_EXPECT(st.pool_total_bytes == 320 * kMiB);
        MS_EXPECT(st.overflow_creates == 0);

        MS_EXPECT(vmm->kv_region_free(rid_a, nullptr) == 0);
        vmm->kv_session_destroy(sid_a);
        vmm->kv_session_destroy(sid_b);
        MS_EXPECT(vmm->shutdown() == 0);
    }

    // 2. pop_fine_handle_locked MUST carve free_super_handles_ BEFORE allocating overflow handles
    //    when allow_overflow=true && max_pool_bytes > 0.
    {
        TriPoolVMMConfig cfg;
        cfg.device = 0;
        cfg.super_pool_mib = 128;   // 2 x 64 MiB
        cfg.fine_kv_pool_mib = 4;   // 2 x 2 MiB
        cfg.super_handle_mib = 64;
        cfg.transient_va_gib = 1;
        cfg.kv_va_gib = 1;
        cfg.allow_overflow = true;
        cfg.max_pool_bytes = 256 * kMiB;
        cfg.simulated = true;
        cfg.materialize_sim_va = false;

        auto vmm = TriPoolVMM::create(cfg);
        const uint64_t sid = vmm->kv_session_create();
        const int32_t rid = vmm->kv_region_alloc(140 * kMiB, sid);
        MS_EXPECT(vmm->kv_region_ensure(rid, 6 * kMiB, nullptr) != 0);

        UnifiedVmmStatsPod st{};
        vmm->get_stats(&st);
        MS_EXPECT(st.super_to_fine_carves == 1);
        MS_EXPECT(st.overflow_creates == 0);
        MS_EXPECT(st.pool_total_bytes == 132 * kMiB);

        // Only after all free super handles are carved does fallback overflow trigger
        MS_EXPECT(vmm->kv_region_ensure(rid, 134 * kMiB, nullptr) != 0);
        vmm->get_stats(&st);
        MS_EXPECT(st.super_to_fine_carves == 2);
        MS_EXPECT(st.overflow_creates == 1);
        MS_EXPECT(st.pool_total_bytes == 134 * kMiB);

        MS_EXPECT(vmm->kv_region_free(rid, nullptr) == 0);
        vmm->kv_session_destroy(sid);
        MS_EXPECT(vmm->shutdown() == 0);
    }

    // 3. Verify HostMmapPageAdapter has no phantom 8 GiB overflow when allow_overflow=true
    {
        TriPoolVMMConfig cfg;
        cfg.device = 0;
        cfg.super_pool_mib = 16;
        cfg.fine_kv_pool_mib = 8;
        cfg.super_handle_mib = 8;
        cfg.transient_va_gib = 1;
        cfg.kv_va_gib = 1;
        cfg.allow_overflow = true;
        cfg.simulated = true;
        cfg.materialize_sim_va = false;

        auto vmm = TriPoolVMM::create(cfg);
        MS_EXPECT(vmm->kv_free_bytes(/*for_dit=*/false) == 24 * kMiB);
        MS_EXPECT(vmm->kv_free_bytes(/*for_dit=*/false, 16 * kMiB) == 40 * kMiB);
        MS_EXPECT(vmm->shutdown() == 0);
    }
}

void test_async_streamer_expert_h2d_bit_exact() {
    // 1. materialize_sim_va=true: real async streamer_thread_ bit-exact H2D payload verification
    {
        TriPoolVMMConfig cfg;
        cfg.device = 0;
        cfg.super_pool_mib = 16;   // 4 x 4 MiB super handles
        cfg.fine_kv_pool_mib = 4;  // 2 x 2 MiB fine pages
        cfg.super_handle_mib = 4;
        cfg.transient_va_gib = 1;
        cfg.kv_va_gib = 1;
        cfg.allow_overflow = false;
        cfg.simulated = true;
        cfg.materialize_sim_va = true;

        auto vmm = TriPoolVMM::create(cfg);
        constexpr int kLayers = 4;
        constexpr size_t kStrideBytes = 4 * kMiB;
        auto mirror = NumaMemfdMirror::create(kLayers * kStrideBytes, 0, "tripool_async_expert");
        for (int l = 0; l < kLayers; ++l) {
            fill_pattern(
                mirror->data() + static_cast<size_t>(l) * kStrideBytes,
                kStrideBytes,
                100U + static_cast<size_t>(l));
        }

        const int eid = vmm->expert_register(kLayers, 4, mirror->data(), mirror->nbytes());
        MS_EXPECT(eid == 0);
        MS_EXPECT(vmm->expert_layer_stride_bytes(eid) == kStrideBytes);
        for (int l = 0; l < kLayers; ++l) {
            MS_EXPECT(vmm->expert_set_layer_used_bytes(eid, l, kStrideBytes) == 0);
        }

        const int queued = vmm->expert_prefetch_async(eid, 0, kLayers);
        MS_EXPECT(queued == kLayers);
        for (int l = 0; l < kLayers; ++l) {
            MS_EXPECT(vmm->expert_wait_layer(eid, l, nullptr) == 0);
            MS_EXPECT(vmm->expert_is_resident(eid, l) == 1);
            const uint64_t lva = vmm->expert_layer_va(eid, l);
            MS_EXPECT(lva != 0);
            MS_EXPECT(verify_pattern(
                reinterpret_cast<const void*>(lva), kStrideBytes, 100U + static_cast<size_t>(l)));
        }
        MS_EXPECT(vmm->expert_sync_h2d(eid) == 0);
        MS_EXPECT(vmm->expert_resident_count(eid) == kLayers);

        UnifiedVmmStatsPod st{};
        vmm->get_stats(&st);
        MS_EXPECT(st.expert_prefetch_layers_completed == static_cast<uint64_t>(kLayers));
        MS_EXPECT(st.expert_h2d_bytes_total == static_cast<uint64_t>(kLayers) * kStrideBytes);

        // Evict range and re-prefetch single layer via wait_layer demand-load
        MS_EXPECT(vmm->expert_evict_range(eid, 2, 4, nullptr) == 2);
        MS_EXPECT(vmm->expert_resident_count(eid) == 2);
        MS_EXPECT(vmm->expert_wait_layer(eid, 3, nullptr) == 0);
        MS_EXPECT(vmm->expert_is_resident(eid, 3) == 1);
        MS_EXPECT(verify_pattern(
            reinterpret_cast<const void*>(vmm->expert_layer_va(eid, 3)), kStrideBytes, 103U));

        MS_EXPECT(vmm->shutdown() == 0);
    }

    // 2. materialize_sim_va=false: async streamer_thread_ must safely skip non-materialized VAs without segfault
    {
        TriPoolVMMConfig cfg;
        cfg.device = 0;
        cfg.super_pool_mib = 128;
        cfg.fine_kv_pool_mib = 16;
        cfg.super_handle_mib = 64;
        cfg.transient_va_gib = 1;
        cfg.kv_va_gib = 1;
        cfg.allow_overflow = false;
        cfg.simulated = true;
        cfg.materialize_sim_va = false;

        auto vmm = TriPoolVMM::create(cfg);
        std::vector<uint8_t> small_host(4096, 0xAB);
        const int eid = vmm->expert_register(2, 64, small_host.data(), small_host.size());
        MS_EXPECT(eid == 0);
        MS_EXPECT(vmm->expert_prefetch_async(eid, 0, 2) == 2);
        MS_EXPECT(vmm->expert_wait_layer(eid, 0, nullptr) == 0);
        MS_EXPECT(vmm->expert_wait_layer(eid, 1, nullptr) == 0);
        MS_EXPECT(vmm->expert_sync_h2d(eid) == 0);
        MS_EXPECT(vmm->expert_resident_count(eid) == 2);
        MS_EXPECT(vmm->shutdown() == 0);
    }
}

void test_async_kv_swap_roundtrip_and_cancel_race() {
    // 1. Bit-exact async swap-out -> kv_flush_swaps -> swap-in roundtrip + cancel_queued_swap_out race
    {
        TriPoolVMMConfig cfg;
        cfg.device = 0;
        cfg.super_pool_mib = 32;
        cfg.fine_kv_pool_mib = 32;  // 16 x 2 MiB fine pages
        cfg.super_handle_mib = 8;
        cfg.transient_va_gib = 1;
        cfg.kv_va_gib = 1;
        cfg.allow_overflow = false;
        cfg.simulated = true;
        cfg.materialize_sim_va = true;

        auto vmm = TriPoolVMM::create(cfg);
        const uint64_t sid = vmm->kv_session_create();
        const int32_t rid = vmm->kv_region_alloc(8 * kMiB, sid);
        MS_EXPECT(rid >= 0);
        const uint64_t va = vmm->kv_region_ensure(rid, 4 * kMiB, nullptr);
        MS_EXPECT(va != 0);
        fill_pattern(reinterpret_cast<void*>(va), 4 * kMiB, 77U);

        MS_EXPECT(vmm->kv_region_swap_out_async(rid, 4 * kMiB, nullptr) == 2);
        MS_EXPECT(vmm->kv_flush_swaps() == 0);
        MS_EXPECT(vmm->kv_region_is_swapped(rid) == 1);
        MS_EXPECT(vmm->kv_region_mapped_bytes(rid) == 0);
        MS_EXPECT(vmm->kv_region_swapped_bytes(rid) == 4 * kMiB);

        UnifiedVmmStatsPod st{};
        vmm->get_stats(&st);
        MS_EXPECT(st.kv_swap_outs_total == 1);
        MS_EXPECT(st.kv_swapped_pages_current == 2);
        MS_EXPECT(st.kv_swap_d2h_bytes_total == 4 * kMiB);
        MS_EXPECT(st.kv_host_pinned_pool_bytes >= 4 * kMiB);

        MS_EXPECT(vmm->kv_region_swap_in_async(rid, nullptr) == 2);
        MS_EXPECT(vmm->kv_region_is_swapped(rid) == 0);
        MS_EXPECT(vmm->kv_region_va(rid) == va);
        MS_EXPECT(vmm->kv_region_mapped_bytes(rid) == 4 * kMiB);
        MS_EXPECT(verify_pattern(reinterpret_cast<const void*>(va), 4 * kMiB, 77U));

        vmm->get_stats(&st);
        MS_EXPECT(st.kv_swap_ins_total == 1);
        MS_EXPECT(st.kv_swapped_pages_current == 0);
        MS_EXPECT(st.kv_swap_h2d_bytes_total == 4 * kMiB);
        MS_EXPECT(vmm->kv_region_free(rid, nullptr) == 0);

        // Race kv_region_swap_in_async against queued swap-outs across 16 regions to exercise
        // cancel_queued_swap_out_locked and SWAP_OUT_FINISHING synchronization concurrently with streamer_thread_.
        constexpr size_t kNumRegions = 16;
        std::vector<int32_t> rids(kNumRegions, -1);
        std::vector<uint64_t> vas(kNumRegions, 0);
        for (size_t i = 0; i < kNumRegions; ++i) {
            rids[i] = vmm->kv_region_alloc(2 * kMiB, sid);
            MS_EXPECT(rids[i] >= 0);
            vas[i] = vmm->kv_region_ensure(rids[i], 2 * kMiB, nullptr);
            MS_EXPECT(vas[i] != 0);
            fill_pattern(reinterpret_cast<void*>(vas[i]), 2 * kMiB, 200U + i);
        }

        UnifiedVmmStatsPod before{};
        vmm->get_stats(&before);
        for (size_t i = 0; i < kNumRegions; ++i) {
            MS_EXPECT(vmm->kv_region_swap_out_async(rids[i], 2 * kMiB, nullptr) == 1);
        }
        // Immediately swap back in reverse order so tail regions cancel while still SWAP_OUT_QUEUED
        for (size_t i = kNumRegions; i > 0; --i) {
            MS_EXPECT(vmm->kv_region_swap_in_async(rids[i - 1], nullptr) >= 1);
        }
        MS_EXPECT(vmm->kv_flush_swaps() == 0);

        UnifiedVmmStatsPod after{};
        vmm->get_stats(&after);
        MS_EXPECT(after.kv_swapped_pages_current == 0);
        MS_EXPECT(after.kv_mapped_pages_current == kNumRegions);
        MS_EXPECT(
            (after.kv_swap_outs_total - before.kv_swap_outs_total) ==
            (after.kv_swap_ins_total - before.kv_swap_ins_total));

        for (size_t i = 0; i < kNumRegions; ++i) {
            MS_EXPECT(vmm->kv_region_is_swapped(rids[i]) == 0);
            MS_EXPECT(vmm->kv_region_va(rids[i]) == vas[i]);
            MS_EXPECT(verify_pattern(reinterpret_cast<const void*>(vas[i]), 2 * kMiB, 200U + i));
            MS_EXPECT(vmm->kv_region_free(rids[i], nullptr) == 0);
        }

        vmm->kv_session_destroy(sid);
        MS_EXPECT(vmm->shutdown() == 0);
    }

    // 2. Non-materialized swap-out -> flush -> swap-in completes safely with zero segfaults
    {
        TriPoolVMMConfig cfg;
        cfg.device = 0;
        cfg.super_pool_mib = 16;
        cfg.fine_kv_pool_mib = 8;
        cfg.super_handle_mib = 8;
        cfg.transient_va_gib = 1;
        cfg.kv_va_gib = 1;
        cfg.allow_overflow = false;
        cfg.simulated = true;
        cfg.materialize_sim_va = false;

        auto vmm = TriPoolVMM::create(cfg);
        const uint64_t sid = vmm->kv_session_create();
        const int32_t rid = vmm->kv_region_alloc(8 * kMiB, sid);
        MS_EXPECT(vmm->kv_region_ensure(rid, 4 * kMiB, nullptr) != 0);
        MS_EXPECT(vmm->kv_region_swap_out_async(rid, 4 * kMiB, nullptr) == 2);
        MS_EXPECT(vmm->kv_flush_swaps() == 0);
        MS_EXPECT(vmm->kv_region_is_swapped(rid) == 1);
        MS_EXPECT(vmm->kv_region_swap_in_async(rid, nullptr) == 2);
        MS_EXPECT(vmm->kv_region_is_swapped(rid) == 0);
        MS_EXPECT(vmm->kv_region_free(rid, nullptr) == 0);
        vmm->kv_session_destroy(sid);
        MS_EXPECT(vmm->shutdown() == 0);
    }
}

void test_session_pinned_multi_exclude_and_ensure_expert_headroom() {
    TriPoolVMMConfig cfg;
    cfg.device = 0;
    cfg.super_pool_mib = 16;   // 2 x 8 MiB
    cfg.fine_kv_pool_mib = 8;  // 4 x 2 MiB -> 24 MiB total
    cfg.super_handle_mib = 8;
    cfg.transient_va_gib = 1;
    cfg.kv_va_gib = 1;
    cfg.allow_overflow = false;
    cfg.simulated = true;
    cfg.materialize_sim_va = true;
    cfg.kv_swap_high_watermark = 0.85;
    cfg.kv_swap_low_watermark = 0.50;
    cfg.kv_swap_min_idle_s = 0.0;

    auto vmm = TriPoolVMM::create(cfg);

    // 1. kv_session_set_pinned(sid, false) makes a session evictable WITHOUT kv_session_mark_thinking
    const uint64_t sid1 = vmm->kv_session_create();
    const int32_t rid1 = vmm->kv_region_alloc(4 * kMiB, sid1);
    const uint64_t va1 = vmm->kv_region_ensure(rid1, 4 * kMiB, nullptr);
    MS_EXPECT(va1 != 0);
    fill_pattern(reinterpret_cast<void*>(va1), 4 * kMiB, 11U);
    MS_EXPECT(vmm->kv_session_mark_active(sid1, nullptr) == 0);

    MS_EXPECT(vmm->reclaim_thinking_kv_bytes(4 * kMiB, /*sync=*/true, /*exclude=*/0, nullptr) == 0);
    MS_EXPECT(vmm->kv_region_is_swapped(rid1) == 0);

    vmm->kv_session_set_pinned(sid1, false);
    MS_EXPECT(vmm->reclaim_thinking_kv_bytes(4 * kMiB, /*sync=*/true, /*exclude=*/0, nullptr) == 4 * kMiB);
    MS_EXPECT(vmm->kv_region_is_swapped(rid1) == 1);

    MS_EXPECT(vmm->kv_session_mark_active(sid1, nullptr) == 0);
    MS_EXPECT(vmm->kv_region_is_swapped(rid1) == 0);
    MS_EXPECT(verify_pattern(reinterpret_cast<const void*>(va1), 4 * kMiB, 11U));

    // 2. Multi-session exclusion in reclaim_thinking_kv_bytes_multi
    const uint64_t sid2 = vmm->kv_session_create();
    const uint64_t sid3 = vmm->kv_session_create();
    vmm->kv_session_set_pinned(sid2, true);
    vmm->kv_session_set_pinned(sid3, true);
    const int32_t rid2 = vmm->kv_region_alloc(8 * kMiB, sid2);
    const int32_t rid3 = vmm->kv_region_alloc(12 * kMiB, sid3);
    MS_EXPECT(vmm->kv_region_ensure(rid2, 8 * kMiB, nullptr) != 0);
    MS_EXPECT(vmm->kv_region_ensure(rid3, 12 * kMiB, nullptr) != 0);

    vmm->kv_session_set_pinned(sid1, false);
    vmm->kv_session_set_pinned(sid2, false);
    vmm->kv_session_set_pinned(sid3, false);

    const uint64_t excl_pair[2] = {sid1, sid2};
    const size_t reclaimed =
        vmm->reclaim_thinking_kv_bytes_multi(24 * kMiB, /*sync=*/true, excl_pair, 2, nullptr);
    MS_EXPECT(reclaimed == 12 * kMiB);
    MS_EXPECT(vmm->kv_region_is_swapped(rid1) == 0);
    MS_EXPECT(vmm->kv_region_is_swapped(rid2) == 0);
    MS_EXPECT(vmm->kv_region_is_swapped(rid3) == 1);

    // 3. Atomic ensure_expert_headroom with multi-session exclusion
    vmm->kv_session_set_pinned(sid1, true);
    vmm->kv_session_set_pinned(sid2, true);
    MS_EXPECT(vmm->kv_session_mark_active(sid3, nullptr) == 0);
    const size_t valid2 = 8 * kMiB;
    const size_t valid3 = 12 * kMiB;
    vmm->kv_session_mark_thinking(sid2, &rid2, &valid2, 1, /*trigger_watermark=*/false, nullptr);
    vmm->kv_session_mark_thinking(sid3, &rid3, &valid3, 1, /*trigger_watermark=*/false, nullptr);
    vmm->kv_session_set_pinned(sid1, false);  // Unpinned, protected ONLY via exclude_session_ids

    const int eid = vmm->expert_register(2, 8, nullptr, 0);
    MS_EXPECT(eid == 0);

    const size_t reclaimed_for_exp = vmm->ensure_expert_headroom(eid, 16 * kMiB, &sid1, 1, nullptr);
    MS_EXPECT(reclaimed_for_exp >= 16 * kMiB);
    MS_EXPECT(vmm->kv_region_is_swapped(rid1) == 0);
    MS_EXPECT(vmm->expert_load_layers_sync(eid, 0, 2) == 0);
    MS_EXPECT(vmm->expert_resident_count(eid) == 2);
    MS_EXPECT(vmm->kv_region_is_swapped(rid1) == 0);

    MS_EXPECT(vmm->kv_region_free(rid1, nullptr) == 0);
    MS_EXPECT(vmm->kv_region_free(rid2, nullptr) == 0);
    MS_EXPECT(vmm->kv_region_free(rid3, nullptr) == 0);
    vmm->kv_session_destroy(sid1);
    vmm->kv_session_destroy(sid2);
    vmm->kv_session_destroy(sid3);
    MS_EXPECT(vmm->shutdown() == 0);
}

void test_nested_transient_scopes() {
    TriPoolVMMConfig cfg;
    cfg.device = 0;
    cfg.super_pool_mib = 16;  // 2 x 8 MiB
    cfg.fine_kv_pool_mib = 4;
    cfg.super_handle_mib = 8;
    cfg.transient_va_gib = 1;
    cfg.kv_va_gib = 1;
    cfg.allow_overflow = false;
    cfg.simulated = true;
    cfg.materialize_sim_va = true;

    auto vmm = TriPoolVMM::create(cfg);
    UnifiedVmmStatsPod st{};

    // Underflow guard when scope_depth_ == 0
    MS_EXPECT(vmm->scope_end(nullptr) == 0);
    vmm->get_stats(&st);
    MS_EXPECT(st.transient_scope_flushes == 0);

    // Outer scope -> inner scope: inner scope_end must NOT flush even if live_allocs == 0
    vmm->scope_begin();
    void* p1 = vmm->transient_alloc(4 * kMiB, nullptr);
    MS_EXPECT(p1 != nullptr);
    vmm->scope_begin();
    void* p2 = vmm->transient_alloc(2 * kMiB, nullptr);
    MS_EXPECT(p2 != nullptr);

    vmm->transient_free(p2, 2 * kMiB, nullptr);
    MS_EXPECT(vmm->scope_end(nullptr) == 1);  // p1 still live, depth 2 -> 1
    vmm->get_stats(&st);
    MS_EXPECT(st.transient_mapped_handles_current == 1);
    MS_EXPECT(st.transient_scope_flushes == 0);

    // Second nested scope where all allocs are freed before inner scope_end: still depth 2 -> 1 => NO flush!
    vmm->scope_begin();
    vmm->transient_free(p1, 4 * kMiB, nullptr);
    MS_EXPECT(vmm->scope_end(nullptr) == 0);  // depth 2 -> 1
    vmm->get_stats(&st);
    MS_EXPECT(st.transient_live_allocs == 0);
    MS_EXPECT(st.transient_mapped_handles_current == 1);
    MS_EXPECT(st.transient_scope_flushes == 0);

    // Outer scope_end (depth 1 -> 0): flushes 100% of transient mappings
    MS_EXPECT(vmm->scope_end(nullptr) == 0);
    vmm->get_stats(&st);
    MS_EXPECT(st.transient_mapped_handles_current == 0);
    MS_EXPECT(st.transient_scope_flushes == 1);
    MS_EXPECT(st.super_pool_free_handles == 2);

    // Extra unmatched scope_end stays guarded at 0
    MS_EXPECT(vmm->scope_end(nullptr) == 0);
    vmm->get_stats(&st);
    MS_EXPECT(st.transient_scope_flushes == 1);

    MS_EXPECT(vmm->shutdown() == 0);
}

void test_oom_exceptions_strict_and_capped() {
    // 1. allow_overflow=false exhaustion throws VmmOutOfMemoryError across KV, transient, expert load/wait
    {
        TriPoolVMMConfig cfg;
        cfg.device = 0;
        cfg.super_pool_mib = 8;    // 1 x 8 MiB
        cfg.fine_kv_pool_mib = 4;  // 2 x 2 MiB -> 12 MiB total
        cfg.super_handle_mib = 8;
        cfg.transient_va_gib = 1;
        cfg.kv_va_gib = 1;
        cfg.allow_overflow = false;
        cfg.simulated = true;
        cfg.materialize_sim_va = true;

        auto vmm = TriPoolVMM::create(cfg);
        const uint64_t sid = vmm->kv_session_create();
        const int32_t rid = vmm->kv_region_alloc(16 * kMiB, sid);
        MS_EXPECT(vmm->kv_region_ensure(rid, 12 * kMiB, nullptr) != 0);
        MS_EXPECT(vmm->kv_session_mark_active(sid, nullptr) == 0);

        MS_EXPECT_THROW(vmm->kv_region_ensure(rid, 14 * kMiB, nullptr), VmmOutOfMemoryError);
        MS_EXPECT_THROW(vmm->kv_alloc_page(), VmmOutOfMemoryError);
        MS_EXPECT_THROW(vmm->transient_alloc(8 * kMiB, nullptr), VmmOutOfMemoryError);

        const int eid = vmm->expert_register(1, 8, nullptr, 0);
        MS_EXPECT(eid == 0);
        MS_EXPECT_THROW(vmm->expert_load_layers_sync(eid, 0, 1), VmmOutOfMemoryError);
        MS_EXPECT_THROW(vmm->expert_wait_layer(eid, 0, nullptr), VmmOutOfMemoryError);

        // Verify VMM and streamer_thread_ recover cleanly once memory is freed
        MS_EXPECT(vmm->kv_region_free(rid, nullptr) == 0);
        MS_EXPECT(vmm->expert_prefetch_async(eid, 0, 1) == 1);
        MS_EXPECT(vmm->expert_wait_layer(eid, 0, nullptr) == 0);
        MS_EXPECT(vmm->expert_sync_h2d(eid) == 0);
        MS_EXPECT(vmm->expert_resident_count(eid) == 1);

        vmm->kv_session_destroy(sid);
        MS_EXPECT(vmm->shutdown() == 0);
    }

    // 2. allow_overflow=true + max_pool_bytes hard cap throws VmmOutOfMemoryError on KV, transient, and swap-in
    {
        TriPoolVMMConfig cfg;
        cfg.device = 0;
        cfg.super_pool_mib = 8;
        cfg.fine_kv_pool_mib = 4;  // 12 MiB initial
        cfg.super_handle_mib = 8;
        cfg.transient_va_gib = 1;
        cfg.kv_va_gib = 1;
        cfg.allow_overflow = true;
        cfg.max_pool_bytes = 20 * kMiB;  // Cap at +8 MiB growth
        cfg.simulated = true;
        cfg.materialize_sim_va = true;

        auto vmm = TriPoolVMM::create(cfg);
        const uint64_t sid1 = vmm->kv_session_create();
        const int32_t rid1 = vmm->kv_region_alloc(24 * kMiB, sid1);
        MS_EXPECT(vmm->kv_region_ensure(rid1, 20 * kMiB, nullptr) != 0);
        MS_EXPECT(vmm->kv_session_mark_active(sid1, nullptr) == 0);

        UnifiedVmmStatsPod st{};
        vmm->get_stats(&st);
        MS_EXPECT(st.pool_total_bytes == 20 * kMiB);

        MS_EXPECT_THROW(vmm->kv_region_ensure(rid1, 22 * kMiB, nullptr), VmmOutOfMemoryError);
        MS_EXPECT_THROW(vmm->transient_alloc(8 * kMiB, nullptr), VmmOutOfMemoryError);

        // Swap out rid1, fill entire 20 MiB cap with pinned sid2, then verify swap-in & mark_active throw OOM
        MS_EXPECT(vmm->kv_region_swap_out_async(rid1, 2 * kMiB, nullptr) == 10);
        MS_EXPECT(vmm->kv_flush_swaps() == 0);

        const uint64_t sid2 = vmm->kv_session_create();
        const int32_t rid2 = vmm->kv_region_alloc(20 * kMiB, sid2);
        MS_EXPECT(vmm->kv_region_ensure(rid2, 20 * kMiB, nullptr) != 0);
        MS_EXPECT(vmm->kv_session_mark_active(sid2, nullptr) == 0);

        MS_EXPECT_THROW(vmm->kv_region_swap_in_async(rid1, nullptr), VmmOutOfMemoryError);
        MS_EXPECT_THROW(vmm->kv_session_mark_active(sid1, nullptr), VmmOutOfMemoryError);

        vmm->get_stats(&st);
        MS_EXPECT(st.pool_total_bytes == 20 * kMiB);

        MS_EXPECT(vmm->kv_region_free(rid1, nullptr) == 0);
        MS_EXPECT(vmm->kv_region_free(rid2, nullptr) == 0);
        vmm->kv_session_destroy(sid1);
        vmm->kv_session_destroy(sid2);
        MS_EXPECT(vmm->shutdown() == 0);
    }
}

void test_lru_multibank_eviction_and_release_free_overflow() {
    // 1. Multi-bank LRU eviction: evict the least-recently-used unpinned expert bank first.
    {
        TriPoolVMMConfig cfg;
        cfg.device = 0;
        cfg.super_pool_mib = 32;  // 4 x 8 MiB
        cfg.fine_kv_pool_mib = 4;
        cfg.super_handle_mib = 8;
        cfg.transient_va_gib = 1;
        cfg.kv_va_gib = 1;
        cfg.allow_overflow = false;
        cfg.simulated = true;
        cfg.materialize_sim_va = true;

        auto vmm = TriPoolVMM::create(cfg);
        const int bank0_dec = vmm->expert_register(2, 8, nullptr, 0);
        const int bank1_vlm = vmm->expert_register(2, 8, nullptr, 0);
        MS_EXPECT(bank0_dec == 0 && bank1_vlm == 1);

        // Load VLM (older) first, then Decoder (newer); unpin both.
        MS_EXPECT(vmm->expert_load_layers_sync(bank1_vlm, 0, 2) == 0);
        MS_EXPECT(vmm->expert_load_layers_sync(bank0_dec, 0, 2) == 0);
        MS_EXPECT(vmm->expert_resident_count(bank0_dec) == 2);
        MS_EXPECT(vmm->expert_resident_count(bank1_vlm) == 2);

        // Transient scratchpad requires 1 super handle -> must evict older VLM (bank 1), NOT Decoder (bank 0).
        vmm->scope_begin();
        void* p1 = vmm->transient_alloc(8 * kMiB, nullptr);
        MS_EXPECT(p1 != nullptr);
        MS_EXPECT(vmm->expert_resident_count(bank0_dec) == 2);
        MS_EXPECT(vmm->expert_resident_count(bank1_vlm) == 1);

        // Touch VLM layer 0 so VLM is now MORE recently used than Decoder; next transient evicts Decoder.
        MS_EXPECT(vmm->expert_wait_layer(bank1_vlm, 0, nullptr) == 0);
        void* p2 = vmm->transient_alloc(8 * kMiB, nullptr);
        MS_EXPECT(p2 != nullptr);
        MS_EXPECT(vmm->expert_resident_count(bank0_dec) == 1);
        MS_EXPECT(vmm->expert_resident_count(bank1_vlm) == 1);

        vmm->transient_free(p2, 8 * kMiB, nullptr);
        vmm->transient_free(p1, 8 * kMiB, nullptr);
        MS_EXPECT(vmm->scope_end(nullptr) == 0);
        MS_EXPECT(vmm->shutdown() == 0);
    }

    // 2. release_free_overflow_to_driver shrinks idle overflow handles back to initial pool floor.
    {
        TriPoolVMMConfig cfg;
        cfg.device = 0;
        cfg.super_pool_mib = 8;   // 1 x 8 MiB
        cfg.fine_kv_pool_mib = 4; // 2 x 2 MiB -> 12 MiB initial pool floor
        cfg.super_handle_mib = 8;
        cfg.transient_va_gib = 1;
        cfg.kv_va_gib = 1;
        cfg.allow_overflow = true;
        cfg.max_pool_bytes = 32 * kMiB;
        cfg.simulated = true;
        cfg.materialize_sim_va = true;

        auto vmm = TriPoolVMM::create(cfg);
        const uint64_t sid = vmm->kv_session_create();
        const int32_t rid = vmm->kv_region_alloc(24 * kMiB, sid);
        MS_EXPECT(vmm->kv_region_ensure(rid, 20 * kMiB, nullptr) != 0);

        UnifiedVmmStatsPod st_peak{};
        vmm->get_stats(&st_peak);
        MS_EXPECT(st_peak.pool_total_bytes >= 20 * kMiB);
        MS_EXPECT(st_peak.overflow_creates > 0);

        // Free KV pages -> overflow handles sit idle in free pool until released.
        MS_EXPECT(vmm->kv_region_free(rid, nullptr) == 0);
        const size_t trimmed = vmm->release_free_overflow_to_driver(SIZE_MAX);
        MS_EXPECT(trimmed >= 8 * kMiB);

        UnifiedVmmStatsPod st_after{};
        vmm->get_stats(&st_after);
        MS_EXPECT(st_after.pool_total_bytes == 12 * kMiB);
        MS_EXPECT(vmm->release_free_overflow_to_driver(SIZE_MAX) == 0);

        vmm->kv_session_destroy(sid);
        MS_EXPECT(vmm->shutdown() == 0);
    }

    // 3. pop_fine_handle_locked grows within max_pool_bytes BEFORE evicting unpinned experts or
    //    thinking KV, and release_free_overflow_to_driver trims fine pages first without shrinking
    //    super_pool_total_ below init_super_handles_.
    {
        TriPoolVMMConfig cfg;
        cfg.device = 0;
        cfg.super_pool_mib = 16;   // 2 x 8 MiB super handles (init_super_handles_ = 2)
        cfg.fine_kv_pool_mib = 4;  // 2 x 2 MiB fine pages (base_fine_pages_ = 2)
        cfg.super_handle_mib = 8;
        cfg.transient_va_gib = 1;
        cfg.kv_va_gib = 1;
        cfg.allow_overflow = true;
        cfg.max_pool_bytes = 28 * kMiB;  // +8 MiB headroom above 20 MiB initial pool
        cfg.simulated = true;
        cfg.materialize_sim_va = true;

        auto vmm = TriPoolVMM::create(cfg);
        const int eid = vmm->expert_register(2, 8, nullptr, 0);
        MS_EXPECT(vmm->expert_load_layers_sync(eid, 0, 2) == 0);  // Consumes both super handles (free_super=0)
        MS_EXPECT(vmm->expert_resident_count(eid) == 2);

        const uint64_t sid_think = vmm->kv_session_create();
        const int32_t rid_think = vmm->kv_region_alloc(4 * kMiB, sid_think);
        MS_EXPECT(vmm->kv_region_ensure(rid_think, 4 * kMiB, nullptr) != 0);  // Consumes base fine pages
        MS_EXPECT(vmm->kv_session_mark_thinking(sid_think, nullptr) == 0);

        // Active KV session needs 6 MiB (3 fine pages) -> must grow within max_pool_bytes (20->26 MiB)
        // WITHOUT evicting the unpinned expert layers or swapping the thinking KV region!
        const uint64_t sid_act = vmm->kv_session_create();
        const int32_t rid_act = vmm->kv_region_alloc(12 * kMiB, sid_act);
        MS_EXPECT(vmm->kv_region_ensure(rid_act, 6 * kMiB, nullptr) != 0);
        MS_EXPECT(vmm->expert_resident_count(eid) == 2);
        UnifiedVmmStatsPod st_grow{};
        vmm->get_stats(&st_grow);
        MS_EXPECT(st_grow.overflow_creates == 3);
        MS_EXPECT(st_grow.expert_auto_evictions_for_kv == 0);
        MS_EXPECT(st_grow.kv_host_swap_outs == 0);
        MS_EXPECT(st_grow.pool_total_bytes == 26 * kMiB);

        // Exceeding max_pool_bytes (need +4 MiB -> 30 > 28 MiB): grows 2 MiB to cap, then evicts expert.
        MS_EXPECT(vmm->kv_region_ensure(rid_act, 10 * kMiB, nullptr) != 0);
        MS_EXPECT(vmm->expert_resident_count(eid) == 1);
        UnifiedVmmStatsPod st_cap{};
        vmm->get_stats(&st_cap);
        MS_EXPECT(st_cap.pool_total_bytes == 28 * kMiB);
        MS_EXPECT(st_cap.expert_auto_evictions_for_kv == 1);

        // Free active KV and evict remaining expert layer -> 1 free super handle + excess fine pages.
        // release_free_overflow_to_driver MUST release excess fine pages first and keep super_pool_total_ intact.
        MS_EXPECT(vmm->kv_region_free(rid_act, nullptr) == 0);
        MS_EXPECT(vmm->expert_evict_range(eid, 0, 2, nullptr) >= 1);
        UnifiedVmmStatsPod st_pre_trim{};
        vmm->get_stats(&st_pre_trim);
        const uint64_t super_before_trim = st_pre_trim.super_pool_total_handles;
        MS_EXPECT(super_before_trim <= 2);

        const size_t trimmed = vmm->release_free_overflow_to_driver(8 * kMiB);
        MS_EXPECT(trimmed == 8 * kMiB);
        UnifiedVmmStatsPod st_post_trim{};
        vmm->get_stats(&st_post_trim);
        MS_EXPECT(st_post_trim.super_pool_total_handles == super_before_trim);
        MS_EXPECT(st_post_trim.pool_total_bytes == 20 * kMiB);

        MS_EXPECT(vmm->kv_region_free(rid_think, nullptr) == 0);
        vmm->kv_session_destroy(sid_act);
        vmm->kv_session_destroy(sid_think);
        MS_EXPECT(vmm->shutdown() == 0);
    }
}

}  // namespace

int main() {
    test_pool_conservation_carving_bundling_and_coalesce();
    test_strict_cap_bundling_carve_before_overflow_and_no_phantom_overflow();
    test_async_streamer_expert_h2d_bit_exact();
    test_async_kv_swap_roundtrip_and_cancel_race();
    test_session_pinned_multi_exclude_and_ensure_expert_headroom();
    test_nested_transient_scopes();
    test_oom_exceptions_strict_and_capped();
    test_lru_multibank_eviction_and_release_free_overflow();

    if (g_failures > 0) {
        std::cerr << "[FAIL] test_tripool_vmm (" << g_failures << " check(s) failed)\n";
        return 1;
    }
    std::cout << "[PASS] test_tripool_vmm\n";
    return 0;
}
