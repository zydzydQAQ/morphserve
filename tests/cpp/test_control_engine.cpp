#include "morphserve/control/budget_planner.hpp"
#include "morphserve/control/hash_util.hpp"
#include "morphserve/control/morph_controller.hpp"
#include "morphserve/control/policy.hpp"
#include "morphserve/control/preempt_edf.hpp"
#include "morphserve/control/router.hpp"
#include "morphserve/control/topology.hpp"
#include "morphserve/runtime/h2d_pacer.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, const char* file, int line) {
    if (!cond) {
        ++g_failures;
        std::cerr << "[FAIL] " << file << ":" << line << ": " << expr << "\n";
    }
}

#define MS_CHECK(cond) check_impl(static_cast<bool>(cond), #cond, __FILE__, __LINE__)

void test_hash_util_and_operator_profile() {
    using namespace morphserve::control;

    // Known NIST SHA-256 vectors.
    MS_CHECK(
        sha256_hex_bytes(nullptr, 0) ==
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    const uint8_t abc[3] = {'a', 'b', 'c'};
    MS_CHECK(
        sha256_hex_bytes(abc, 3) ==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    const uint32_t h1 = md5_prefix_u32("morphserve_session_0");
    const uint32_t h2 = md5_prefix_u32("morphserve_session_0");
    const uint32_t h3 = md5_prefix_u32("morphserve_session_1");
    MS_CHECK(h1 == h2);
    MS_CHECK(h1 != h3);

    const std::vector<double> vals = {10.0, 20.0, 30.0, 40.0};
    MS_CHECK(std::abs(percentile_raw(vals, 50.0) - 25.0) < 1e-9);
    MS_CHECK(std::abs(mean_raw(vals) - 25.0) < 1e-9);
    MS_CHECK(std::abs(percentile_rounded(vals, 95.0) - 38.5) < 1e-9);
    MS_CHECK(percentile_raw({}, 95.0) == 0.0);
    MS_CHECK(mean_raw({}) == 0.0);

    // FlashPrefix-CFG arithmetic-intensity gate.
    MS_CHECK(should_use_cascade_cfg_split(1024, 2048, 2));
    MS_CHECK(!should_use_cascade_cfg_split(1024, 1024, 2));
    MS_CHECK(!should_use_cascade_cfg_split(1024, 2048, 1));
    MS_CHECK(!should_use_cascade_cfg_split(4096, 8192, 2));
    MS_CHECK(should_use_cascade_cfg_split(4096, 16384, 2));

    // OperatorLatencyProfile JSON round-trip and BAGEL vs InternVL-U SLO thresholds.
    OperatorLatencyProfile op;
    op.ar_prefill_ms = 71.5;
    op.ar_tpot_ms = 58.25;
    op.prep_overhead_ms = 210.0;
    op.bagel_dit_512_step_ms = 260.0;
    op.bagel_dit_1024_step_ms = 980.0;
    op.bagel_expert_bytes = 11000LL * kMiB;
    op.internvlu_dit_1024_step_ms = 390.0;
    const OperatorLatencyProfile op_rt = OperatorLatencyProfile::from_json(op.to_json());
    MS_CHECK(std::abs(op_rt.ar_prefill_ms - 71.5) < 1e-9);
    MS_CHECK(std::abs(op_rt.ar_tpot_ms - 58.25) < 1e-9);
    MS_CHECK(std::abs(op_rt.prep_overhead_ms - 210.0) < 1e-9);
    MS_CHECK(std::abs(op_rt.bagel_dit_512_step_ms - 260.0) < 1e-9);
    MS_CHECK(std::abs(op_rt.bagel_dit_1024_step_ms - 980.0) < 1e-9);
    MS_CHECK(op_rt.bagel_expert_bytes == 11000LL * kMiB);
    MS_CHECK(std::abs(op_rt.internvlu_dit_1024_step_ms - 390.0) < 1e-9);

    const auto bagel_slo = default_operator_profile().slo_thresholds_ms(3.0, "bagel");
    const auto ivl_slo = default_operator_profile().slo_thresholds_ms(3.0, "internvl_u");
    MS_CHECK(std::abs(bagel_slo.at("dit_512_slo_ms") - 18360.0) < 1e-6);
    MS_CHECK(std::abs(bagel_slo.at("dit_1024_slo_ms") - 65160.0) < 1e-6);
    MS_CHECK(std::abs(ivl_slo.at("dit_512_slo_ms") - 9750.0) < 1e-6);
    MS_CHECK(std::abs(ivl_slo.at("dit_1024_slo_ms") - 21900.0) < 1e-6);
}

void test_policy_presets_and_overrides() {
    using namespace morphserve::control;

    for (const char* p : kValidPolicies) {
        MS_CHECK(is_valid_policy_name(p));
    }
    for (const char* p : kStaticSplitPolicies) {
        MS_CHECK(is_valid_policy_name(p));
    }
    MS_CHECK(!is_valid_policy_name("unknown_policy_xyz"));

    // Asymmetric static-fission presets directly through PolicyConfig::from_preset.
    const PolicyConfig p26 = PolicyConfig::from_preset("static_2und_6gen", 8);
    MS_CHECK(p26.policy_name == "static_2und_6gen");
    MS_CHECK(p26.routing_mode == "static_nccl_fission");
    MS_CHECK(p26.static_und_ranks == (std::vector<int>{0, 4}));
    MS_CHECK(p26.static_gen_ranks == (std::vector<int>{1, 2, 3, 5, 6, 7}));
    MS_CHECK(p26.role_of(0) == "und" && p26.role_of(4) == "und");
    MS_CHECK(p26.role_of(1) == "gen" && p26.role_of(7) == "gen");
    MS_CHECK(std::abs(p26.initial_alphas.at(0) - 0.0) < 1e-9);
    MS_CHECK(std::abs(p26.initial_alphas.at(1) - 1.0) < 1e-9);

    const PolicyConfig p62 = PolicyConfig::from_preset("static_6und_2gen", 8);
    MS_CHECK(p62.policy_name == "static_6und_2gen");
    MS_CHECK(p62.routing_mode == "static_nccl_fission");
    MS_CHECK(p62.static_und_ranks == (std::vector<int>{0, 1, 2, 4, 5, 6}));
    MS_CHECK(p62.static_gen_ranks == (std::vector<int>{3, 7}));
    MS_CHECK(p62.role_of(2) == "und" && p62.role_of(3) == "gen");

    const PolicyConfig p44 = PolicyConfig::from_preset("static_4und_4gen", 8);
    MS_CHECK(p44.static_und_ranks == (std::vector<int>{0, 1, 4, 5}));
    MS_CHECK(p44.static_gen_ranks == (std::vector<int>{2, 3, 6, 7}));

    // Full override coverage + JSON round-trip.
    const PolicyConfig custom = PolicyConfig::from_preset(
        "morphserve",
        8,
        {
            {"dit_slo_scale_k", 3.5},
            {"allow_cascade_cfg_split", true},
            {"think_time_dist", "constant"},
            {"think_time_median_s", 11.5},
            {"think_time_sigma", 0.4},
            {"ar_tpot_slo_mode", "strict_max"},
            {"steady_window_mode", "session_trim"},
            {"warmup_window_s", 45.0},
            {"steady_window_s", 150.0},
            {"steady_state_trim_frac", 0.2},
            {"gpu_memory_quota_mib", 12288.0},
        });
    const PolicyConfig rt = PolicyConfig::from_json(custom.to_json());
    MS_CHECK(std::abs(rt.dit_slo_scale_k - 3.5) < 1e-9);
    MS_CHECK(rt.allow_cascade_cfg_split == true);
    MS_CHECK(rt.think_time_dist == "constant");
    MS_CHECK(std::abs(rt.think_time_median_s - 11.5) < 1e-9);
    MS_CHECK(std::abs(rt.think_time_sigma - 0.4) < 1e-9);
    MS_CHECK(rt.ar_tpot_slo_mode == "strict_max");
    MS_CHECK(rt.steady_window_mode == "session_trim");
    MS_CHECK(std::abs(rt.warmup_window_s - 45.0) < 1e-9);
    MS_CHECK(std::abs(rt.steady_window_s - 150.0) < 1e-9);
    MS_CHECK(std::abs(rt.steady_state_trim_frac - 0.2) < 1e-9);
    MS_CHECK(rt.gpu_memory_quota_mib.has_value() && std::abs(*rt.gpu_memory_quota_mib - 12288.0) < 1e-9);

    // Leave-One-Out M1-M4 ablation suite & static split suite propagate overrides.
    const auto abl = build_ablation_configs(8, {{"dit_slo_scale_k", 3.5}});
    MS_CHECK(abl.size() == 5);
    MS_CHECK(abl[0].first == "full_morphserve" && abl[0].second.use_tripool_vmm && abl[0].second.preemption_mode == "layer_chunk");
    MS_CHECK(abl[1].first == "no_m4_layer_chunk" && abl[1].second.preemption_mode == "non_preemptive");
    MS_CHECK(abl[2].first == "no_m3_cow_loopback" && !abl[2].second.use_module3_cow_loopback);
    MS_CHECK(abl[3].first == "no_m2_dual_path" && abl[3].second.routing_mode == "sticky_least_loaded");
    MS_CHECK(abl[4].first == "no_m1_tripool_vmm" && !abl[4].second.use_tripool_vmm);
    for (const auto& [_, cfg] : abl) {
        MS_CHECK(std::abs(cfg.dit_slo_scale_k - 3.5) < 1e-9);
    }

    const auto splits = build_static_split_configs(8, {{"dit_slo_scale_k", 2.5}});
    MS_CHECK(splits.size() == 4);
    MS_CHECK(splits[0].first == "morphserve_elastic" && splits[0].second.routing_mode == "dual_path");
    MS_CHECK(splits[1].first == "static_2und_6gen" && splits[1].second.static_und_ranks.size() == 2);
    MS_CHECK(splits[2].first == "static_4und_4gen" && splits[2].second.static_und_ranks.size() == 4);
    MS_CHECK(splits[3].first == "static_6und_2gen" && splits[3].second.static_und_ranks.size() == 6);
    for (const auto& [_, cfg] : splits) {
        MS_CHECK(std::abs(cfg.dit_slo_scale_k - 2.5) < 1e-9);
    }
}

void test_router_seams() {
    using namespace morphserve::control;

    const auto topo = ClusterTopologyProfile::a100_pcie_8gpu_default();
    MS_CHECK(topo.same_half(0, 1));
    MS_CHECK(!topo.same_half(0, 4));

    // Custom OperatorLatencyProfile is honored by DualPathMorphRouter & crossover math.
    OperatorLatencyProfile custom_op;
    custom_op.bagel_expert_bytes = 6000LL * kMiB;
    custom_op.bagel_dit_512_step_ms = 200.0;
    auto pacer = std::make_shared<morphserve::runtime::HalfNodeH2DPacer>(topo.gpu_to_group_map(), 1);
    DualPathMorphRouter router(topo, pacer, custom_op);
    MS_CHECK(router.operator_profile().bagel_expert_bytes == 6000LL * kMiB);
    const double xover_custom = router.analytical_crossover_kv_mib(0.50, 0, 0.0, 0, 1);
    const double xover_default = DualPathMorphRouter(topo).analytical_crossover_kv_mib(0.50, 0, 0.0, 0, 1);
    MS_CHECK(xover_custom > 0.0 && xover_custom < xover_default);

    // `plan_transition_readonly` MUST cause ZERO pacer mutation and ZERO worker mutation.
    router.init_workers(
        2,
        [](int) { return "hybrid"; },
        {{0, 0.0}, {1, 1.0}},
        {{0, 16 * kGiB}, {1, 16 * kMiB}});  // GPU 1 lacks free KV -> forces `in_place_morph` on GPU 0.

    CPTGRequestSpec req_morph;
    req_morph.req_id = "ro_morph";
    req_morph.src_gpu = 0;
    req_morph.architecture = "tight_mot";
    req_morph.kv_bytes = 400 * kMiB;
    req_morph.gen_expert_bytes = 6000LL * kMiB;
    req_morph.dit_step1_ms = 200.0;
    req_morph.dit_total_steps = 20;

    MS_CHECK(pacer->predicted_wait_ms(0, 0.0) == 0.0);
    const auto ro_morph = router.plan_transition_readonly("dual_path", {}, {}, req_morph, 5000.0, nullptr, {}, 0.0);
    MS_CHECK(ro_morph.first.path == "in_place_morph" && ro_morph.first.dst_gpu == 0 && ro_morph.second == "none");
    MS_CHECK(pacer->predicted_wait_ms(0, 0.0) == 0.0);
    MS_CHECK(router.get_worker_state(0).queue_wait_ms == 0.0);
    MS_CHECK(router.get_worker_state(0).kv_free_bytes == 16 * kGiB);

    // Calling mutating `plan_and_reserve_transition` reserves both pacer DMA slot and queue wait.
    const auto mut_morph = router.plan_and_reserve_transition("dual_path", {}, {}, req_morph, 5000.0, nullptr, {}, true, 0.0);
    MS_CHECK(mut_morph.first.path == "in_place_morph");
    MS_CHECK(pacer->predicted_wait_ms(0, 0.0) > 0.0);
    MS_CHECK(std::abs(router.get_worker_state(0).queue_wait_ms - 5000.0) < 1e-6);
    router.release_dit_reservation(0, 5000.0);
    MS_CHECK(std::abs(router.get_worker_state(0).queue_wait_ms - 0.0) < 1e-6);

    // Explicit AR queue reservation seam.
    router.reserve_ar_queue(0, 145.5);
    MS_CHECK(std::abs(router.get_worker_state(0).queue_wait_ms - 145.5) < 1e-6);
    router.release_ar_queue(0, 45.5);
    MS_CHECK(std::abs(router.get_worker_state(0).queue_wait_ms - 100.0) < 1e-6);
    router.release_ar_queue(0, 500.0);
    MS_CHECK(std::abs(router.get_worker_state(0).queue_wait_ms - 0.0) < 1e-6);

    // Read-only `state_ship` check & `select_static_gen_gpu` & `route_batch`.
    DualPathMorphRouter router_ship(topo);
    router_ship.init_workers(
        4,
        [](int g) { return (g == 1 || g == 3) ? "gen" : "hybrid"; },
        {{0, 0.5}, {1, 1.0}, {2, 0.5}, {3, 1.0}},
        {{0, 16 * kGiB}, {1, 600 * kMiB}, {2, 16 * kGiB}, {3, 16 * kGiB}});

    CPTGRequestSpec req_ship;
    req_ship.req_id = "r_ship";
    req_ship.src_gpu = 0;
    req_ship.architecture = "tight_mot";
    req_ship.kv_bytes = 400 * kMiB;
    req_ship.dit_step1_ms = 280.0;
    req_ship.dit_total_steps = 20;

    const auto ro_ship = router_ship.plan_transition_readonly("dual_path", {}, {}, req_ship, 50.0);
    MS_CHECK(ro_ship.first.path == "state_ship" && ro_ship.first.dst_gpu == 1 && ro_ship.second == "nccl_dual");
    MS_CHECK(router_ship.get_worker_state(1).kv_free_bytes == 600 * kMiB);
    MS_CHECK(router_ship.get_worker_state(1).queue_wait_ms == 0.0);

    MS_CHECK(
        router_ship.select_static_gen_gpu("static_nccl_fission", {0, 2}, {1, 3}, 0, nullptr, {}, 256 * kMiB) == 1);

    const auto batch_decs = router_ship.route_batch({req_ship, req_ship}, router_ship.worker_states_snapshot(), 0.0);
    MS_CHECK(batch_decs.size() == 2);
    MS_CHECK(batch_decs[0].path == "state_ship" && batch_decs[0].dst_gpu == 1);
    MS_CHECK(batch_decs[1].path == "state_ship" && batch_decs[1].dst_gpu == 3);

    // Telemetry sequence monotonicity & inflight KV reconciliation on BOTH internal ledger and `ext_workers`.
    const auto mut_ship = router_ship.plan_and_reserve_transition("dual_path", {}, {}, req_ship, 50.0);
    MS_CHECK(mut_ship.first.dst_gpu == 1);
    MS_CHECK(router_ship.get_worker_state(1).kv_free_bytes == 200 * kMiB);
    router_ship.update_worker_telemetry(1, 1.0, 600 * kMiB, 5);
    MS_CHECK(router_ship.get_worker_state(1).kv_free_bytes == 200 * kMiB);
    router_ship.update_worker_telemetry(1, 0.25, 16 * kGiB, 4);  // Stale seq 4 < 5 ignored.
    MS_CHECK(router_ship.get_worker_state(1).kv_free_bytes == 200 * kMiB);
    MS_CHECK(std::abs(router_ship.get_worker_state(1).expert_alpha - 1.0) < 1e-9);

    DualPathMorphRouter router_ext(topo);
    std::map<int, WorkerState> ext_map = router_ext.init_workers(
        2,
        [](int g) { return (g == 1) ? "gen" : "hybrid"; },
        {{0, 0.5}, {1, 1.0}},
        {{0, 16 * kGiB}, {1, 600 * kMiB}});
    router_ext.reserve_kv_bytes(1, 400 * kMiB, &ext_map);
    MS_CHECK(ext_map.at(1).kv_free_bytes == 200 * kMiB);
    router_ext.update_worker_telemetry(1, 1.0, 600 * kMiB, 5, &ext_map);
    MS_CHECK(ext_map.at(1).kv_free_bytes == 200 * kMiB);
    router_ext.update_worker_telemetry(1, 0.25, 16 * kGiB, 4, &ext_map);  // Stale seq ignored on ext_map too.
    MS_CHECK(ext_map.at(1).kv_free_bytes == 200 * kMiB);
    MS_CHECK(std::abs(ext_map.at(1).expert_alpha - 1.0) < 1e-9);
    router_ext.release_kv_reservation(1, 400 * kMiB, 250 * kMiB, 6, &ext_map);
    MS_CHECK(ext_map.at(1).kv_free_bytes == 250 * kMiB);
    router_ext.release_kv_reservation(1, 0, 900 * kMiB, 5, &ext_map);  // Stale seq ignored on release too.
    MS_CHECK(ext_map.at(1).kv_free_bytes == 250 * kMiB);

    // Inter-node state-ship size cap & cross-node penalty.
    const auto topo_2node = ClusterTopologyProfile::from_multinode_measurements(
        2, 1, 10.0, 9.0, 2.0, 8.75, 10.35, "A100", 3.0, 12.0);
    MS_CHECK(topo_2node.fabric_tier(0, 1) == "inter_node");
    DualPathMorphRouter router_xnode(
        topo_2node, nullptr, default_operator_profile(), std::nullopt, 12.0, 0.90, true, true, 128LL * kMiB);
    router_xnode.init_workers(
        2,
        [](int g) { return (g == 1) ? "gen" : "hybrid"; },
        {{0, 0.0}, {1, 1.0}},
        {{0, 16 * kGiB}, {1, 16 * kGiB}});

    CPTGRequestSpec small_xnode = req_ship;
    small_xnode.kv_bytes = 64 * kMiB;  // <= 128 MiB cap -> allowed across inter_node fabric.
    const auto dec_small = router_xnode.decide(small_xnode);
    MS_CHECK(dec_small.path == "state_ship" && dec_small.dst_gpu == 1 && dec_small.hop_tier == "inter_node");
    MS_CHECK(dec_small.t_ship_ms > 12.0);

    CPTGRequestSpec large_xnode = req_ship;
    large_xnode.kv_bytes = 256 * kMiB;  // > 128 MiB cap -> blocked, must morph in place.
    const auto ship_large = router_xnode.evaluate_path_a_ship_ms(
        large_xnode, router_xnode.get_worker_state(0), router_xnode.get_worker_state(1), 0.0);
    MS_CHECK(std::isinf(ship_large.first));
    const auto dec_large = router_xnode.decide(large_xnode);
    MS_CHECK(dec_large.path == "in_place_morph" && dec_large.dst_gpu == 0);

    // Architecture-aware initial placement (`bridge_mllm_dit` vs `tight_mot`, `ar_only` non-DiT preference).
    DualPathMorphRouter router_arch(topo);
    router_arch.init_workers(2, [](int) { return "hybrid"; }, {{0, 0.0}, {1, 1.0}}, {{0, 16 * kGiB}, {1, 16 * kGiB}});
    MS_CHECK(
        router_arch.select_initial_gpu(
            2, "dual_path", "layer_chunk", {}, "dit_t2i", nullptr, {{0, false}, {1, false}}, {}, {}, 2, "bridge_mllm_dit") ==
        0);
    MS_CHECK(
        router_arch.select_initial_gpu(
            2, "dual_path", "layer_chunk", {}, "dit_t2i", nullptr, {{0, false}, {1, false}}, {}, {}, 2, "tight_mot") ==
        1);
    MS_CHECK(
        router_arch.select_initial_gpu(
            2, "dual_path", "layer_chunk", {}, "ar_only", nullptr, {{0, true}, {1, true}}, {{0, 1}, {1, 0}}, {{0, 0}, {1, 0}}, 2, "tight_mot") ==
        1);
}

void test_preempt_edf_controller() {
    using namespace morphserve::control;

    PreemptEdfController edf(28, 60.0, 4.0, 8.5, 20.0, 3);
    MS_CHECK(edf.compute_adaptive_chunk_layers(60.0) == 4);
    MS_CHECK(edf.compute_adaptive_chunk_layers(100.0) == 8);

    // Yield decision near vs far from deadline.
    MS_CHECK(edf.should_yield_now(10.00, 10.03, 10.0));   // -30 + 10 + 20 + 4 = +4 >= 0
    MS_CHECK(!edf.should_yield_now(10.00, 10.20, 10.0));  // -200 + 10 + 20 + 4 < 0

    // Microbatch selection with `force_all_urgent = false` vs `true`.
    const double dls[4] = {100.032, 100.015, 100.500, 100.025};
    std::vector<size_t> due;
    std::vector<size_t> deferred;
    edf.select_edf_microbatch(100.00, dls, 4, /*force_all_urgent=*/false, due, deferred);
    MS_CHECK(due == (std::vector<size_t>{1, 3, 0}));
    MS_CHECK(deferred == (std::vector<size_t>{2}));

    edf.select_edf_microbatch(100.00, dls, 4, /*force_all_urgent=*/true, due, deferred);
    MS_CHECK(due == (std::vector<size_t>{1, 3, 0, 2}));
    MS_CHECK(deferred.empty());

    // Anti-phase lock prevention at 100 ms TPOT SLO (50 ms AR step): 110 ms slack peer coalesces into B=2.
    PreemptEdfController edf_bagel(28, 100.0, 14.0, 10.5, 50.0, 4);
    const double dls_anti[3] = {100.040, 100.110, 100.500};
    edf_bagel.select_edf_microbatch(100.00, dls_anti, 3, /*force_all_urgent=*/false, due, deferred);
    MS_CHECK(due == (std::vector<size_t>{0, 1}));
    MS_CHECK(deferred == (std::vector<size_t>{2}));

    // EMA updates.
    const double new_layer = edf_bagel.record_layer_sample(20.5);
    const double new_ar = edf_bagel.record_ar_batch_sample(70.0);
    MS_CHECK(std::abs(new_layer - 13.5) < 1e-9);
    MS_CHECK(std::abs(new_ar - 60.0) < 1e-9);
    MS_CHECK(std::abs(edf_bagel.ema_layer_ms() - 13.5) < 1e-9);
    MS_CHECK(std::abs(edf_bagel.ema_ar_step_ms() - 60.0) < 1e-9);
}

void test_worker_budget_planner() {
    using namespace morphserve::control;

    // Unconstrained 40 GiB Regime A plan.
    const auto unconstrained = WorkerBudgetPlanner::compute_internvlu_memory_plan(
        40960.0, 34000.0, 260.0, 4864LL * kMiB, 12288, 512, 64, true, std::nullopt, std::nullopt);
    MS_CHECK(!unconstrained.at("quota_constrained").get<bool>());
    MS_CHECK(unconstrained.at("allow_overflow").get<bool>());
    MS_CHECK(unconstrained.at("vmm_physical_pool_mib").get<int>() > 6688);

    // 12 GiB Regime B quota plan & verification using canonical `peak_pytorch_reserved_mib` + `vmm_physical_pool_mib`.
    const auto q_pair = WorkerBudgetPlanner::compute_internvlu_quota_overrides(
        "morphserve", 8, "/shared/data/MORPHSERVE/models/InternVL-U", 12.0, 40.0);
    MS_CHECK(std::abs(q_pair.second.at("runtime_combined_budget_mib").get<double>() - 12288.0) < 1e-6);

    const PolicyConfig pol_12g = PolicyConfig::from_preset("morphserve", 8, q_pair.first);
    const auto plan_12g = WorkerBudgetPlanner::compute_internvlu_memory_plan(
        40960.0,
        35000.0,
        260.0,
        4864LL * kMiB,
        pol_12g.super_pool_mib,
        pol_12g.fine_kv_pool_mib,
        pol_12g.super_handle_mib,
        pol_12g.allow_vmm_overflow,
        pol_12g.gpu_memory_fraction);
    MS_CHECK(plan_12g.at("quota_constrained").get<bool>());
    MS_CHECK(plan_12g.at("vmm_physical_pool_mib").get<int>() == 6688);
    MS_CHECK(plan_12g.at("pytorch_cap_mib").get<double>() + plan_12g.at("vmm_physical_pool_mib").get<double>() <= 12288.0);

    const nlohmann::json ok_rep = {
        {"peak_pytorch_reserved_mib", 5100.0},
        {"vmm_physical_pool_mib", 6688.0},
        {"vmm_overflow_creates", 0},
    };
    const auto v_ok = WorkerBudgetPlanner::verify_internvlu_memory_quota(ok_rep, pol_12g, 12.0);
    MS_CHECK(v_ok.at("within_quota").get<bool>());
    MS_CHECK(std::abs(v_ok.at("vmm_physical_pool_mib").get<double>() - 6688.0) < 1e-6);
    MS_CHECK(std::abs(v_ok.at("combined_physical_mib").get<double>() - 11788.0) < 1e-6);

    const nlohmann::json over_rep = {
        {"peak_pytorch_reserved_mib", 6000.0},
        {"vmm_physical_pool_mib", 6688.0},
        {"vmm_overflow_creates", 0},
    };
    const auto v_over = WorkerBudgetPlanner::verify_internvlu_memory_quota(over_rep, pol_12g, 12.0);
    MS_CHECK(!v_over.at("within_quota").get<bool>());
    MS_CHECK(std::abs(v_over.at("combined_physical_mib").get<double>() - 12688.0) < 1e-6);

    const nlohmann::json ovf_rep = {
        {"peak_pytorch_reserved_mib", 5100.0},
        {"vmm_physical_pool_mib", 6688.0},
        {"vmm_overflow_creates", 1},
    };
    MS_CHECK(!WorkerBudgetPlanner::verify_internvlu_memory_quota(ovf_rep, pol_12g, 12.0).at("within_quota").get<bool>());
}

void test_morph_control_engine_transitions_slos_and_aggregation() {
    using namespace morphserve::control;

    const auto topo = ClusterTopologyProfile::a100_pcie_8gpu_default();

    // 1. `plan_turn_transition_readonly` vs `plan_turn_transition` on `tight_mot` & `bridge_mllm_dit`.
    const PolicyConfig pol_fission = PolicyConfig::from_preset("static_nccl_fission", 8);
    auto r_fission = std::make_shared<DualPathMorphRouter>(topo);
    r_fission->init_workers(
        8,
        [&](int g) { return pol_fission.role_of(g); },
        pol_fission.initial_alphas,
        {{0, 16 * kGiB}, {1, 16 * kGiB}, {2, 16 * kGiB}, {3, 16 * kGiB},
         {4, 16 * kGiB}, {5, 16 * kGiB}, {6, 16 * kGiB}, {7, 16 * kGiB}});
    MorphControlEngine eng_fission(pol_fission, topo, r_fission);

    const auto ro_plan = eng_fission.plan_turn_transition_readonly(
        "sess_f0", "ctx_f0", 0, 0, 0, 320 * kMiB, "dit_edit", 512, 20, true, true, 0.0);
    MS_CHECK(ro_plan.ship_mode == "nccl_static");
    MS_CHECK(ro_plan.interleaved_roundtrip_gamma == 2);
    MS_CHECK(ro_plan.needs_return_ship == true);
    MS_CHECK(ro_plan.ephemeral_gen_ctx == true);
    MS_CHECK(ro_plan.release_src_on_fwd == true);
    MS_CHECK(ro_plan.include_cfg_dup_on_wire == false);
    MS_CHECK(ro_plan.dit_ctx_id == "ctx_f0_gen_t0");
    // Read-only plan must leave destination worker untouched.
    MS_CHECK(r_fission->get_worker_state(ro_plan.dst_gpu).queue_wait_ms == 0.0);
    MS_CHECK(r_fission->get_worker_state(ro_plan.dst_gpu).kv_free_bytes == 16 * kGiB);

    const auto mut_plan = eng_fission.plan_turn_transition(
        "sess_f0", "ctx_f0", 0, 0, 0, 320 * kMiB, "dit_edit", 512, 20, true, true, 0.0);
    MS_CHECK(mut_plan.dst_gpu == ro_plan.dst_gpu);
    MS_CHECK(r_fission->get_worker_state(mut_plan.dst_gpu).queue_wait_ms > 0.0);
    MS_CHECK(r_fission->get_worker_state(mut_plan.dst_gpu).kv_free_bytes < 16 * kGiB);

    // `twostage_shm` ships over SHM with CFG duplication on wire (`include_cfg_dup_on_wire == true`).
    const PolicyConfig pol_shm = PolicyConfig::from_preset("twostage_shm", 8);
    auto r_shm = std::make_shared<DualPathMorphRouter>(topo);
    r_shm->init_workers(
        8,
        [&](int g) { return pol_shm.role_of(g); },
        pol_shm.initial_alphas,
        {{0, 16 * kGiB}, {1, 16 * kGiB}, {2, 16 * kGiB}, {3, 16 * kGiB},
         {4, 16 * kGiB}, {5, 16 * kGiB}, {6, 16 * kGiB}, {7, 16 * kGiB}});
    MorphControlEngine eng_shm(pol_shm, topo, r_shm);
    const auto shm_plan = eng_shm.plan_turn_transition_readonly(
        "sess_shm", "ctx_shm", 1, 0, 0, 320 * kMiB, "dit_edit", 512, 20, true, true, 0.0);
    MS_CHECK(shm_plan.ship_mode == "shm_static");
    MS_CHECK(shm_plan.include_cfg_dup_on_wire == true);
    MS_CHECK(shm_plan.ephemeral_gen_ctx == true);
    MS_CHECK(shm_plan.dit_ctx_id == "ctx_shm_gen_t1");

    // `bridge_mllm_dit` (InternVL-U):
    // - under `morphserve` (`nccl_dual`): ships compact bridge, no return ship, preserves base ctx id.
    // - under `static_nccl_fission` (`nccl_static`): ephemeral gen ctx, no return ship, keeps source VLM ctx when `has_more_turns=true`.
    const PolicyConfig pol_ivl = PolicyConfig::from_preset("morphserve", 8, {{"model_family", "internvl_u"}});
    auto r_ivl = std::make_shared<DualPathMorphRouter>(topo);
    r_ivl->init_workers(
        8,
        [](int) { return "hybrid"; },
        {{0, 0.0}, {1, 1.0}, {2, 1.0}, {3, 1.0}, {4, 1.0}, {5, 1.0}, {6, 1.0}, {7, 1.0}},
        {{0, 16 * kGiB}, {1, 16 * kGiB}, {2, 16 * kGiB}, {3, 16 * kGiB},
         {4, 16 * kGiB}, {5, 16 * kGiB}, {6, 16 * kGiB}, {7, 16 * kGiB}});
    MorphControlEngine eng_ivl(pol_ivl, topo, r_ivl);
    const auto bridge_plan = eng_ivl.plan_turn_transition_readonly(
        "sess_ivl", "ctx_ivl", 0, 0, 0, 16 * kMiB, "dit_t2i", 1024, 16, true, true, 0.0);
    MS_CHECK(bridge_plan.decision.path == "state_ship");
    MS_CHECK(bridge_plan.ship_mode == "nccl_dual");
    MS_CHECK(bridge_plan.ephemeral_gen_ctx == false);
    MS_CHECK(bridge_plan.needs_return_ship == false);
    MS_CHECK(bridge_plan.release_src_on_fwd == true);
    MS_CHECK(bridge_plan.include_cfg_dup_on_wire == false);
    MS_CHECK(bridge_plan.dit_ctx_id == "ctx_ivl");

    const PolicyConfig pol_ivl_fis = PolicyConfig::from_preset("static_nccl_fission", 8, {{"model_family", "internvl_u"}});
    auto r_ivl_fis = std::make_shared<DualPathMorphRouter>(topo);
    r_ivl_fis->init_workers(
        8,
        [&](int g) { return pol_ivl_fis.role_of(g); },
        pol_ivl_fis.initial_alphas,
        {{0, 16 * kGiB}, {1, 16 * kGiB}, {2, 16 * kGiB}, {3, 16 * kGiB},
         {4, 16 * kGiB}, {5, 16 * kGiB}, {6, 16 * kGiB}, {7, 16 * kGiB}});
    MorphControlEngine eng_ivl_fis(pol_ivl_fis, topo, r_ivl_fis);
    const auto bridge_fis = eng_ivl_fis.plan_turn_transition_readonly(
        "sess_ivl_f", "ctx_ivl_f", 2, 0, 0, 16 * kMiB, "dit_t2i", 1024, 16, true, true, 0.0);
    MS_CHECK(bridge_fis.ship_mode == "nccl_static");
    MS_CHECK(bridge_fis.ephemeral_gen_ctx == true);
    MS_CHECK(bridge_fis.needs_return_ship == false);
    MS_CHECK(bridge_fis.release_src_on_fwd == false);
    MS_CHECK(bridge_fis.dit_ctx_id == "ctx_ivl_f_gen_t2");

    // 2. `deterministic_think_dist_s` & `deterministic_think_dist_ms_sim` shared helper checks.
    std::set<int> distinct_tenths;
    for (int i = 0; i < 32; ++i) {
        const std::string sid = "sess_think_" + std::to_string(i);
        const double v1 = MorphControlEngine::deterministic_think_dist_s(sid, 1, 2.0, 25.0, "lognormal", 6.0, 0.75);
        const double v2 = MorphControlEngine::deterministic_think_dist_s(sid, 1, 2.0, 25.0, "lognormal", 6.0, 0.75);
        MS_CHECK(v1 == v2);
        MS_CHECK(v1 >= 2.0 && v1 <= 25.0);
        distinct_tenths.insert(static_cast<int>(std::round(v1 * 10.0)));
    }
    MS_CHECK(distinct_tenths.size() >= 10);
    MS_CHECK(std::abs(MorphControlEngine::deterministic_think_dist_s("s", 1, 2.0, 25.0, "constant", 6.0, 0.75) - 6.0) < 1e-12);
    MS_CHECK(MorphControlEngine::deterministic_think_dist_s("s", 1, 0.0, 0.0, "lognormal", 6.0, 0.75) == 0.0);
    const double sim_ms = MorphControlEngine::deterministic_think_dist_ms_sim(7, 2, 2.0, 25.0, "lognormal", 6.0, 0.75);
    MS_CHECK(sim_ms >= 2000.0 && sim_ms <= 25000.0);
    MS_CHECK(std::abs(MorphControlEngine::deterministic_think_dist_ms_sim(7, 2, 2.0, 25.0, "constant", 6.0, 0.75) - 6000.0) < 1e-9);

    // 3. `evaluate_session_slos`: `"strict_max"` MUST behave identically to `"token_max"`.
    std::vector<double> tpots_1_outlier(19, 50.0);
    tpots_1_outlier.push_back(250.0);
    const std::vector<double> ttfts_ok = {300.0};
    const std::vector<double> dits_ok = {4000.0};
    const std::vector<int> heights_ok = {512};
    const std::vector<int> steps_ok = {20};

    const PolicyConfig pol_p95 = PolicyConfig::from_preset("morphserve", 8, {{"ar_tpot_slo_mode", "turn_p95"}});
    const PolicyConfig pol_tmax = PolicyConfig::from_preset("morphserve", 8, {{"ar_tpot_slo_mode", "token_max"}});
    const PolicyConfig pol_smax = PolicyConfig::from_preset("morphserve", 8, {{"ar_tpot_slo_mode", "strict_max"}});

    const auto r_p95 = MorphControlEngine::evaluate_session_slos(
        pol_p95, ttfts_ok, tpots_1_outlier, dits_ok, heights_ok, steps_ok, {tpots_1_outlier});
    const auto r_tmax = MorphControlEngine::evaluate_session_slos(
        pol_tmax, ttfts_ok, tpots_1_outlier, dits_ok, heights_ok, steps_ok, {tpots_1_outlier});
    const auto r_smax = MorphControlEngine::evaluate_session_slos(
        pol_smax, ttfts_ok, tpots_1_outlier, dits_ok, heights_ok, steps_ok, {tpots_1_outlier});
    MS_CHECK(r_p95.at("ar_slo_met").get<bool>() == true);
    MS_CHECK(r_tmax.at("ar_slo_met").get<bool>() == false);
    MS_CHECK(r_smax.at("ar_slo_met").get<bool>() == false);
    MS_CHECK(r_smax == r_tmax);

    // `slo_time_scale` tightens AR TPOT cap cleanly (80 ms passes at 1.0x=125.64ms, fails at 0.5x=62.82ms).
    const auto r_1x = MorphControlEngine::evaluate_session_slos(
        pol_p95, {300.0}, {80.0, 80.0}, {4000.0}, {512}, {20}, {{80.0, 80.0}}, 1.0);
    const auto r_half = MorphControlEngine::evaluate_session_slos(
        pol_p95, {300.0}, {80.0, 80.0}, {4000.0}, {512}, {20}, {{80.0, 80.0}}, 0.5);
    MS_CHECK(r_1x.at("ar_slo_met").get<bool>() == true);
    MS_CHECK(r_half.at("ar_slo_met").get<bool>() == false);

    // 4. Canonical `aggregate_cluster_run_report`: steady vs full-trace semantics, `vmm_physical_pool_mib`,
    //    InternVL-U (16 steps) vs BAGEL (20 steps) parity, and D1-D5 multi-turn SLO contract.
    const PolicyConfig pol_win = PolicyConfig::from_preset(
        "morphserve",
        8,
        {
            {"steady_window_mode", "time"},
            {"warmup_window_s", 60.0},
            {"steady_window_s", 180.0},
        });
    auto mk_out = [](const std::string& sid, double arr_s, const std::string& st, bool ok_slo) -> nlohmann::json {
        return {
            {"session_id", sid},
            {"quadrant", "interleaved_co_creation"},
            {"arrival_offset_s", arr_s},
            {"status", st},
            {"e2e_ms", 12000.0},
            {"active_jct_ms", 6000.0},
            {"think_time_ms", 6000.0},
            {"planned_turns", 2},
            {"planned_ar_turns", 1},
            {"planned_dit_turns", 1},
            {"ar_ttfts_ms", st == "ok" ? nlohmann::json::array({220.0}) : nlohmann::json::array()},
            {"ar_tpots_ms", st == "ok" ? nlohmann::json::array({50.0, 52.0}) : nlohmann::json::array()},
            {"dit_turns_ms", st == "ok" ? nlohmann::json::array({4200.0}) : nlohmann::json::array()},
            {"dit_turn_heights", st == "ok" ? nlohmann::json::array({512}) : nlohmann::json::array()},
            {"dit_turn_steps", st == "ok" ? nlohmann::json::array({20}) : nlohmann::json::array()},
            {"ar_slo_met", ok_slo},
            {"dit_slo_met", ok_slo},
            {"joint_slo_met", ok_slo},
            {"turn_records",
             nlohmann::json::array({
                 {{"turn_idx", 0}, {"modality_plan", "ar_only"}, {"status", st}, {"ar_ttft_ms", 220.0}, {"ar_tpots_ms", nlohmann::json::array({50.0, 52.0})}},
                 {{"turn_idx", 1}, {"modality_plan", "dit_edit"}, {"status", st}, {"dit_ms", 4200.0}, {"height", 512}, {"dit_steps", 20}},
             })},
        };
    };
    const nlohmann::json outcomes_win = nlohmann::json::array({
        mk_out("w0", 15.0, "oom", false),
        mk_out("s0", 80.0, "ok", true),
        mk_out("s1", 150.0, "ok", true),
        mk_out("d0", 260.0, "oom", false),
    });
    const nlohmann::json wstats = nlohmann::json::array({
        {
            {"nvml_peak_used_mib", 11800.0},
            {"pytorch_reserved_mib", 5100.0},
            {"vmm_stats", {{"pool_total_bytes", 6688LL * kMiB}, {"kv_swap_outs_total", 4}, {"kv_swap_ins_total", 3}}},
        },
        {
            {"nvml_peak_used_mib", 11500.0},
            {"pytorch_reserved_mib", 4900.0},
            {"vmm_physical_pool_mib", 6688.0},
            {"kv_host_swap_outs", 2},
            {"kv_host_swap_ins", 1},
        },
    });
    const auto rep = MorphControlEngine::aggregate_cluster_run_report(
        pol_win, "unit_canonical_window", 0.5, 300.0, 0.0, outcomes_win, wstats);

    MS_CHECK(rep.at("steady_state_sessions").get<int>() == 2);
    MS_CHECK(std::abs(rep.at("joint_slo_goodput_rps").get<double>() - 0.5) < 1e-6);
    MS_CHECK(std::abs(rep.at("steady_joint_slo_goodput_rps").get<double>() - 0.5) < 1e-6);
    MS_CHECK(std::abs(rep.at("joint_slo_attainment_pct").get<double>() - 100.0) < 1e-6);
    MS_CHECK(std::abs(rep.at("steady_joint_slo_attainment_pct").get<double>() - 100.0) < 1e-6);
    MS_CHECK(std::abs(rep.at("full_trace_goodput_rps").get<double>() - 0.25) < 1e-6);
    MS_CHECK(std::abs(rep.at("full_trace_joint_slo_attainment_pct").get<double>() - 50.0) < 1e-6);
    MS_CHECK(std::abs(rep.at("vmm_physical_pool_mib").get<double>() - 6688.0) < 1e-6);
    MS_CHECK(rep.at("kv_host_swap_outs").get<int>() == 6);
    MS_CHECK(rep.at("kv_host_swap_ins").get<int>() == 4);
    MS_CHECK(std::abs(rep.at("slo_config").at("dit_1024_slo_ms").get<double>() - 65160.0) < 1e-6);
    MS_CHECK(std::abs(rep.at("slo_curve_by_dit_k").at("3.0x").at("dit_1024_slo_ms").get<double>() - 65160.0) < 1e-6);

    // InternVL-U 16-step 1024px parity in `slo_config`, `slo_curve_by_k`, and `slo_curve_by_dit_k`.
    const PolicyConfig pol_ivl_win = PolicyConfig::from_preset(
        "morphserve",
        8,
        {{"model_family", "internvl_u"}, {"steady_window_mode", "time"}, {"warmup_window_s", 60.0}, {"steady_window_s", 180.0}});
    const auto rep_ivl = MorphControlEngine::aggregate_cluster_run_report(
        pol_ivl_win, "unit_ivl_steps", 0.5, 300.0, 0.0, outcomes_win, wstats);
    MS_CHECK(std::abs(rep_ivl.at("slo_config").at("dit_512_slo_ms").get<double>() - 9750.0) < 1e-6);
    MS_CHECK(std::abs(rep_ivl.at("slo_config").at("dit_1024_slo_ms").get<double>() - 21900.0) < 1e-6);
    MS_CHECK(std::abs(rep_ivl.at("slo_curve_by_k").at("3.0x").at("dit_1024_slo_ms").get<double>() - 21900.0) < 1e-6);
    MS_CHECK(std::abs(rep_ivl.at("slo_curve_by_dit_k").at("3.0x").at("dit_1024_slo_ms").get<double>() - 21900.0) < 1e-6);

    // D1-D5 planned-turns denominator with mid-session OOM.
    const nlohmann::json out_a = {
        {"session_id", "s_a"},
        {"arrival_offset_s", 80.0},
        {"status", "ok"},
        {"e2e_ms", 42000.0},
        {"active_jct_ms", 36000.0},
        {"think_time_ms", 6000.0},
        {"planned_turns", 4},
        {"planned_ar_turns", 3},
        {"planned_dit_turns", 2},
        {"ar_ttfts_ms", nlohmann::json::array({300.0, 400.0, 350.0})},
        {"ar_tpots_ms", nlohmann::json::array({55.0, 60.0, 58.0})},
        {"ar_turn_tpots_ms", nlohmann::json::array({{55.0}, {60.0}, {58.0}})},
        {"dit_turns_ms", nlohmann::json::array({15000.0, 20000.0})},
        {"dit_turn_heights", nlohmann::json::array({512, 512})},
        {"dit_turn_steps", nlohmann::json::array({20, 20})},
        {"ar_slo_met", true},
        {"dit_slo_met", false},
        {"joint_slo_met", false},
        {"turn_records",
         nlohmann::json::array({
             {{"turn_idx", 0}, {"modality_plan", "ar_only"}, {"status", "ok"}, {"ar_ttft_ms", 300.0}, {"ar_tpot_p95_ms", 55.0}},
             {{"turn_idx", 1}, {"modality_plan", "cot_then_dit"}, {"status", "ok"}, {"ar_ttft_ms", 400.0}, {"ar_tpot_p95_ms", 60.0}, {"dit_ms", 15000.0}, {"height", 512}, {"dit_steps", 20}},
             {{"turn_idx", 2}, {"modality_plan", "ar_only"}, {"status", "ok"}, {"ar_ttft_ms", 350.0}, {"ar_tpot_p95_ms", 58.0}},
             {{"turn_idx", 3}, {"modality_plan", "dit_edit"}, {"status", "ok"}, {"dit_ms", 20000.0}, {"height", 512}, {"dit_steps", 20}},
         })},
    };
    const nlohmann::json out_b = {
        {"session_id", "s_b"},
        {"arrival_offset_s", 120.0},
        {"status", "oom"},
        {"e2e_ms", 5000.0},
        {"active_jct_ms", 5000.0},
        {"planned_turns", 4},
        {"planned_ar_turns", 2},
        {"planned_dit_turns", 2},
        {"ar_ttfts_ms", nlohmann::json::array({250.0})},
        {"ar_tpots_ms", nlohmann::json::array({50.0})},
        {"dit_turns_ms", nlohmann::json::array()},
        {"ar_slo_met", false},
        {"dit_slo_met", false},
        {"joint_slo_met", false},
        {"turn_records",
         nlohmann::json::array({
             {{"turn_idx", 0}, {"modality_plan", "ar_only"}, {"status", "ok"}, {"ar_ttft_ms", 250.0}, {"ar_tpot_p95_ms", 50.0}},
             {{"turn_idx", 1}, {"modality_plan", "dit_edit"}, {"status", "oom"}, {"height", 512}, {"dit_steps", 20}},
         })},
    };
    const auto rep_d = MorphControlEngine::aggregate_cluster_run_report(
        pol_win, "unit_d1_d5", 0.5, 200.0, 0.0, nlohmann::json::array({out_a, out_b}), nlohmann::json::array());
    MS_CHECK(rep_d.at("steady_planned_turns").get<int>() == 8);
    MS_CHECK(rep_d.at("steady_completed_turns").get<int>() == 5);
    MS_CHECK(std::abs(rep_d.at("steady_turn_survival_pct").get<double>() - 62.5) < 1e-6);
    MS_CHECK(std::abs(rep_d.at("steady_turn_slo_attainment_pct").get<double>() - 50.0) < 1e-6);
    MS_CHECK(std::abs(rep_d.at("slo_curve_by_dit_k").at("2.0x").at("turn_attainment_pct").get<double>() - 37.5) < 1e-6);
    MS_CHECK(std::abs(rep_d.at("slo_curve_by_dit_k").at("4.0x").at("session_attainment_pct").get<double>() - 50.0) < 1e-6);
}

}  // namespace

int main() {
    test_hash_util_and_operator_profile();
    test_policy_presets_and_overrides();
    test_router_seams();
    test_preempt_edf_controller();
    test_worker_budget_planner();
    test_morph_control_engine_transitions_slos_and_aggregation();

    if (g_failures > 0) {
        std::cerr << "[FAILED] test_control_engine (" << g_failures << " checks failed)\n";
        return 1;
    }
    std::cout << "[PASS] test_control_engine\n";
    return 0;
}
