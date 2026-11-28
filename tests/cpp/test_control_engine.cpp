#include "morphserve/control/budget_planner.hpp"
#include "morphserve/control/gpu_queue_scheduler.hpp"
#include "morphserve/control/hash_util.hpp"
#include "morphserve/control/morph_controller.hpp"
#include "morphserve/control/performance_atlas.hpp"
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
    MS_CHECK(std::abs(bagel_slo.at("ar_tpot_slo_ms") - 152.10) < 1e-6);
    MS_CHECK(std::abs(bagel_slo.at("dit_512_slo_ms") - 14916.30) < 1e-6);
    MS_CHECK(std::abs(bagel_slo.at("dit_1024_slo_ms") - 57219.60) < 1e-6);
    MS_CHECK(std::abs(default_operator_profile().unloaded_dit_turn_ms("bagel", 1024, 10) - 9675.16) < 1e-6);
    MS_CHECK(std::abs(ivl_slo.at("ar_tpot_slo_ms") - 132.0) < 1e-6);
    MS_CHECK(std::abs(ivl_slo.at("dit_512_slo_ms") - 7484.40) < 1e-6);
    MS_CHECK(std::abs(ivl_slo.at("dit_1024_slo_ms") - 21374.40) < 1e-6);

    // Modality-normalized AR TTFT thresholds (text-only vs 512^2 / 1024^2 image inputs).
    MS_CHECK(std::abs(default_operator_profile().ar_ttft_slo_for(false, 0, 512, 2.0, "bagel") - 250.0) < 1e-6);
    MS_CHECK(std::abs(default_operator_profile().ar_ttft_slo_for(true, 1, 512, 2.0, "bagel") - 2800.0) < 1e-6);
    MS_CHECK(default_operator_profile().ar_ttft_slo_for(true, 2, 512, 2.0, "bagel") >
             default_operator_profile().ar_ttft_slo_for(true, 1, 512, 2.0, "bagel"));
    MS_CHECK(default_operator_profile().ar_ttft_slo_for(true, 1, 1024, 2.0, "bagel") >
             default_operator_profile().ar_ttft_slo_for(true, 1, 512, 2.0, "bagel"));
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

    // `use_idle_kv_swap` policy contract & `monolithic_kv_swap` preset parity.
    const PolicyConfig p_ms = PolicyConfig::from_preset("morphserve", 8);
    const PolicyConfig p_mono = PolicyConfig::from_preset("monolithic_dp", 8);
    const PolicyConfig p_mono_swap = PolicyConfig::from_preset("monolithic_kv_swap", 8);
    const PolicyConfig p_prism = PolicyConfig::from_preset("prism_tetriserve_coloc", 8);
    const PolicyConfig p_shm = PolicyConfig::from_preset("twostage_shm", 8);
    const PolicyConfig p_naive = PolicyConfig::from_preset("naive_host_offload", 8);
    MS_CHECK(p_ms.use_idle_kv_swap == true);
    MS_CHECK(p_ms.cost_model_mode == "pipeline_analytical");
    MS_CHECK(std::abs(p_ms.gpu_queue_max_aging_ms - 45000.0) < 1e-9);
    MS_CHECK(p_ms.gpu_queue_max_bypasses == 5);
    MS_CHECK(p_mono.use_idle_kv_swap == false);
    MS_CHECK(p_mono_swap.use_idle_kv_swap == true);
    MS_CHECK(p_mono_swap.routing_mode == p_mono.routing_mode && !p_mono_swap.use_tripool_vmm && !p_mono_swap.use_scoped_vae);
    MS_CHECK(p_prism.use_idle_kv_swap == true);
    MS_CHECK(p_prism.gpu_queue_mode == "memory_slack");
    MS_CHECK(std::abs(p_prism.gpu_queue_max_aging_ms - 45000.0) < 1e-9);
    MS_CHECK(p_prism.gpu_queue_max_bypasses == 5);
    MS_CHECK(p_prism.super_pool_mib == 0 && p_prism.fine_kv_pool_mib == 64);
    MS_CHECK(p_shm.use_idle_kv_swap == false);
    MS_CHECK(p_naive.use_idle_kv_swap == false);
    MS_CHECK(p44.use_idle_kv_swap == false);
    MS_CHECK(std::abs(p_ms.warmup_window_s - 60.0) < 1e-9);
    MS_CHECK(std::abs(p_ms.steady_window_s - 150.0) < 1e-9);
    MS_CHECK(std::abs(p_ms.cooldown_s - 30.0) < 1e-9);

    // InternVL-U Regime A enables step-level preemption (VLM + Decoder both resident, tight 2.5 GiB VMM super pool);
    // quota-capped Regime B stays non-preemptive because VLM & Decoder are co-paged.
    const PolicyConfig p_ivlu_a = PolicyConfig::from_preset("morphserve", 8, {{"model_family", "internvl_u"}});
    const PolicyConfig p_ivlu_b_q = PolicyConfig::from_preset(
        "morphserve", 8, {{"model_family", "internvl_u"}, {"gpu_memory_quota_mib", 12288.0}});
    const PolicyConfig p_ivlu_b_f = PolicyConfig::from_preset(
        "morphserve", 8, {{"model_family", "internvl_u"}, {"gpu_memory_fraction", 0.30}});
    MS_CHECK(p_ivlu_a.preemption_mode == "step_preemptive");
    MS_CHECK(p_ivlu_a.allow_vmm_overflow == true);
    MS_CHECK(p_ivlu_a.super_pool_mib == 2560 && p_ivlu_a.fine_kv_pool_mib == 128);
    MS_CHECK(p_ivlu_b_q.preemption_mode == "non_preemptive");
    MS_CHECK(p_ivlu_b_q.allow_vmm_overflow == false);
    MS_CHECK(p_ivlu_b_f.preemption_mode == "non_preemptive");
    MS_CHECK(p_ivlu_b_f.allow_vmm_overflow == false);

    // Full override coverage + JSON round-trip.
    const PolicyConfig custom = PolicyConfig::from_preset(
        "morphserve",
        8,
        {
            {"dit_slo_scale_k", 3.5},
            {"allow_cascade_cfg_split", true},
            {"use_idle_kv_swap", false},
            {"think_time_dist", "constant"},
            {"think_time_median_s", 11.5},
            {"think_time_sigma", 0.4},
            {"ar_tpot_slo_mode", "strict_max"},
            {"steady_window_mode", "session_trim"},
            {"warmup_window_s", 45.0},
            {"steady_window_s", 150.0},
            {"cooldown_s", 20.0},
            {"steady_state_trim_frac", 0.2},
            {"gpu_memory_quota_mib", 12288.0},
        });
    const PolicyConfig rt = PolicyConfig::from_json(custom.to_json());
    MS_CHECK(std::abs(rt.dit_slo_scale_k - 3.5) < 1e-9);
    MS_CHECK(rt.allow_cascade_cfg_split == true);
    MS_CHECK(rt.use_idle_kv_swap == false);
    MS_CHECK(rt.think_time_dist == "constant");
    MS_CHECK(std::abs(rt.think_time_median_s - 11.5) < 1e-9);
    MS_CHECK(std::abs(rt.think_time_sigma - 0.4) < 1e-9);
    MS_CHECK(rt.ar_tpot_slo_mode == "strict_max");
    MS_CHECK(rt.steady_window_mode == "session_trim");
    MS_CHECK(std::abs(rt.warmup_window_s - 45.0) < 1e-9);
    MS_CHECK(std::abs(rt.steady_window_s - 150.0) < 1e-9);
    MS_CHECK(std::abs(rt.cooldown_s - 20.0) < 1e-9);
    MS_CHECK(std::abs(rt.steady_state_trim_frac - 0.2) < 1e-9);
    MS_CHECK(rt.gpu_memory_quota_mib.has_value() && std::abs(*rt.gpu_memory_quota_mib - 12288.0) < 1e-9);

    // Decoupled 7-mechanism Leave-One-Out M1-M4 + Queue ablation suite & static split suite propagate overrides.
    const auto abl = build_ablation_configs(8, {{"dit_slo_scale_k", 3.5}});
    MS_CHECK(abl.size() == 8);
    MS_CHECK(abl[0].first == "full_morphserve" && abl[0].second.use_tripool_vmm && abl[0].second.preemption_mode == "layer_chunk");
    MS_CHECK(abl[1].first == "no_m1_tripool_vmm" && !abl[1].second.use_tripool_vmm && !abl[1].second.use_expert_pager &&
             abl[1].second.use_scoped_vae && abl[1].second.use_h2d_pacer && abl[1].second.use_shared_numa_mirror);
    MS_CHECK(abl[2].first == "no_m1_scoped_vae" && !abl[2].second.use_scoped_vae && abl[2].second.use_tripool_vmm &&
             abl[2].second.use_expert_pager);
    MS_CHECK(abl[3].first == "no_m2_dual_path" && abl[3].second.routing_mode == "sticky_least_loaded" &&
             abl[3].second.use_h2d_pacer && abl[3].second.use_shared_numa_mirror);
    MS_CHECK(abl[4].first == "no_m2_pacer_numa" && !abl[4].second.use_h2d_pacer && !abl[4].second.use_shared_numa_mirror &&
             abl[4].second.routing_mode == "dual_path");
    MS_CHECK(abl[5].first == "no_m3_cow_loopback" && !abl[5].second.use_module3_cow_loopback && abl[5].second.use_tripool_vmm);
    MS_CHECK(abl[6].first == "no_m4_layer_chunk" && abl[6].second.preemption_mode == "non_preemptive" &&
             abl[6].second.use_tripool_vmm);
    MS_CHECK(abl[7].first == "no_queue_sched" && abl[7].second.gpu_queue_mode == "fcfs" &&
             abl[7].second.use_tripool_vmm && abl[7].second.preemption_mode == "layer_chunk");
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

    // `twostage_shm` load-balances across ALL static_gen_ranks instead of hardcoded 2k -> 2k+1 pairing.
    router_ship.set_worker_state_fields(1, std::nullopt, 5000.0, std::nullopt, std::nullopt, std::nullopt);
    MS_CHECK(
        router_ship.select_static_gen_gpu("twostage_shm", {0, 2}, {1, 3}, 0, nullptr, {}, 256 * kMiB) == 3);
    router_ship.set_worker_state_fields(1, std::nullopt, 0.0, std::nullopt, std::nullopt, std::nullopt);

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
            2, "dual_path", "layer_chunk", {}, "ar_only", nullptr, {{0, true}, {1, true}}, {{0, 1}, {1, 0}}, {{0, 0}, {1, 0}}, 6, "tight_mot") ==
        1);

    // Low-load tight_mot ar_only, dit_t2i, and cot_then_dit MUST spread uniformly across idle GPUs
    // despite minor post-VAE alpha dips (0.75 vs 1.0) so new sessions never pile onto 1-2 GPUs.
    DualPathMorphRouter router_spread(topo);
    router_spread.init_workers(
        4,
        [](int) { return "hybrid"; },
        {{0, 0.75}, {1, 1.0}, {2, 1.0}, {3, 1.0}},
        {{0, 16 * kGiB}, {1, 16 * kGiB}, {2, 16 * kGiB}, {3, 16 * kGiB}});
    std::map<int, bool> all_idle = {{0, false}, {1, false}, {2, false}, {3, false}};
    std::set<int> picked_idle;
    for (int i = 0; i < 4; ++i) {
        picked_idle.insert(
            router_spread.select_initial_gpu(4, "dual_path", "layer_chunk", {}, "ar_only", nullptr, all_idle, {}, {}, 6, "tight_mot"));
    }
    MS_CHECK(picked_idle.size() == 4);
    std::set<int> picked_dit;
    for (int i = 0; i < 4; ++i) {
        picked_dit.insert(
            router_spread.select_initial_gpu(4, "dual_path", "layer_chunk", {}, "dit_t2i", nullptr, all_idle, {}, {}, 6, "tight_mot"));
    }
    MS_CHECK(picked_dit.size() == 4);
    std::set<int> picked_cot;
    for (int i = 0; i < 4; ++i) {
        picked_cot.insert(
            router_spread.select_initial_gpu(4, "dual_path", "layer_chunk", {}, "cot_then_dit", nullptr, all_idle, {}, {}, 6, "tight_mot"));
    }
    MS_CHECK(picked_cot.size() == 4);

    // Genuinely KV-heavy GPU (alpha <= 0.50) IS still preferred for ar_only, and bridge_mllm_dit prefers VLM-hot (<= 0.01).
    DualPathMorphRouter router_ivlu(topo);
    router_ivlu.init_workers(
        4,
        [](int) { return "hybrid"; },
        {{0, 0.0}, {1, 0.25}, {2, 0.0}, {3, 1.0}},
        {{0, 16 * kGiB}, {1, 16 * kGiB}, {2, 16 * kGiB}, {3, 16 * kGiB}});
    const int iv_a = router_ivlu.select_initial_gpu(4, "dual_path", "non_preemptive", {}, "ar_only", nullptr, all_idle, {}, {}, 4, "bridge_mllm_dit");
    const int iv_b = router_ivlu.select_initial_gpu(4, "dual_path", "non_preemptive", {}, "ar_only", nullptr, all_idle, {}, {}, 4, "bridge_mllm_dit");
    // Post-VAE Decoder-hot rank 1 (alpha=0.25) MUST NOT be misclassified as VLM-hot (tier 0).
    MS_CHECK((iv_a == 0 && iv_b == 2) || (iv_a == 2 && iv_b == 0));

    // For bridge_mllm_dit dit_t2i, VLM-hot ranks (0, 2) are preferred over Decoder-hot ranks (1, 3)
    // while VLM queue wait <= 1200 ms, and elastically spill over when VLM ranks backlog > 1200 ms.
    router_ivlu.set_worker_state_fields(0, 0.0, 500.0, 0, 16 * kGiB, 1);
    router_ivlu.set_worker_state_fields(2, 0.0, 800.0, 0, 16 * kGiB, 1);
    router_ivlu.set_worker_state_fields(1, 0.25, 0.0, 0, 16 * kGiB, 1);
    router_ivlu.set_worker_state_fields(3, 1.0, 0.0, 0, 16 * kGiB, 1);
    const int iv_dit_warm = router_ivlu.select_initial_gpu(
        4, "dual_path", "non_preemptive", {}, "dit_t2i", nullptr, all_idle, {}, {}, 4, "bridge_mllm_dit");
    MS_CHECK(iv_dit_warm == 0);
    router_ivlu.set_worker_state_fields(0, 0.0, 2000.0, 0, 16 * kGiB, 1);
    router_ivlu.set_worker_state_fields(2, 0.0, 2000.0, 0, 16 * kGiB, 1);
    const int iv_dit_spill = router_ivlu.select_initial_gpu(
        4, "dual_path", "non_preemptive", {}, "dit_t2i", nullptr, all_idle, {}, {}, 4, "bridge_mllm_dit");
    MS_CHECK(iv_dit_spill == 1 || iv_dit_spill == 3);

    // reserve_ar_queue must NOT falsely inflate colocated_ar_b; set_colocated_ar_count controls it explicitly.
    router_ivlu.reserve_ar_queue(0, 100.0);
    MS_CHECK(router_ivlu.get_worker_state(0).colocated_ar_b == 0);
    router_ivlu.release_ar_queue(0, 100.0);
    router_ivlu.set_colocated_ar_count(0, 3);
    MS_CHECK(router_ivlu.get_worker_state(0).colocated_ar_b == 3);
    router_ivlu.set_colocated_ar_count(0, 0);
    MS_CHECK(router_ivlu.get_worker_state(0).colocated_ar_b == 0);

    // Warm src_hold_credit_ms prevents busy-to-busy double-queuing hops while still offloading to idle peers.
    DualPathMorphRouter router_hold(topo);
    router_hold.init_workers(
        2,
        [](int) { return "hybrid"; },
        {{0, 1.0}, {1, 1.0}},
        {{0, 16 * kGiB}, {1, 16 * kGiB}});
    router_hold.set_worker_state_fields(0, 1.0, 10000.0, 0, 16 * kGiB, 1);
    router_hold.set_worker_state_fields(1, 1.0, 8000.0, 0, 16 * kGiB, 1);
    CPTGRequestSpec req_hold;
    req_hold.req_id = "hold_busy";
    req_hold.src_gpu = 0;
    req_hold.architecture = "bridge_mllm_dit";
    req_hold.bridge_bytes = 4 * kMiB;
    req_hold.src_hold_credit_ms = 5000.0;
    const RouteDecision dec_busy = router_hold.decide(req_hold);
    MS_CHECK(dec_busy.path == "in_place_morph" && dec_busy.dst_gpu == 0);
    // When peer 1 is idle, still offload to peer 1 despite holding src lock!
    router_hold.set_worker_state_fields(1, 1.0, 0.0, 0, 16 * kGiB, 1);
    const RouteDecision dec_idle = router_hold.decide(req_hold);
    MS_CHECK(dec_idle.path == "state_ship" && dec_idle.dst_gpu == 1);
    // src_hold_credit_ms MUST also apply when src has partial post-VAE alpha < 1.0 so holding
    // src's lock never causes spurious Path-A hops onto busy peers.
    router_hold.set_worker_state_fields(0, 0.75, 10000.0, 0, 16 * kGiB, 1);
    router_hold.set_worker_state_fields(1, 1.0, 8000.0, 0, 16 * kGiB, 1);
    CPTGRequestSpec req_hold_partial = req_hold;
    req_hold_partial.architecture = "tight_mot";
    req_hold_partial.kv_bytes = 512 * kMiB;
    req_hold_partial.src_hold_credit_ms = 5000.0;
    const RouteDecision dec_busy_partial = router_hold.decide(req_hold_partial);
    MS_CHECK(dec_busy_partial.path == "in_place_morph" && dec_busy_partial.dst_gpu == 0);

    // Co-paged bridge_mllm_dit Regime B: VLM-hot src (alpha=0.0, post_evict=false) MUST NOT
    // appear cheaper to morph than a post-VAE Decoder-hot peer (alpha=0.40, post_evict=true).
    DualPathMorphRouter router_regb(topo);
    router_regb.init_workers(
        2,
        [](int) { return "hybrid"; },
        {{0, 0.0}, {1, 1.0}},
        {{0, 16 * kGiB}, {1, 16 * kGiB}});
    router_regb.update_worker_telemetry(1, 0.40, 16 * kGiB, 1);  // Triggers post_evict=true on Decoder-hot rank 1
    CPTGRequestSpec req_regb;
    req_regb.req_id = "ivlu_regb_ship_to_warm_dec";
    req_regb.src_gpu = 0;
    req_regb.architecture = "bridge_mllm_dit";
    req_regb.model_family = "internvl_u";
    req_regb.bridge_bytes = 4 * kMiB;
    req_regb.gen_expert_bytes = 5600LL * kMiB;
    req_regb.dit_step1_ms = 350.0;
    req_regb.dit_total_steps = 16;
    const RouteDecision dec_regb = router_regb.decide(req_regb);
    MS_CHECK(dec_regb.path == "state_ship" && dec_regb.dst_gpu == 1);
    MS_CHECK(dec_regb.t_ship_ms < dec_regb.t_morph_ms);
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

    // Unconstrained 40 GiB Regime A plan (explicit large policy_super_pool_mib vs tight default 2560 MiB without +5632 MiB bloat).
    const auto unconstrained = WorkerBudgetPlanner::compute_internvlu_memory_plan(
        40960.0, 34000.0, 260.0, 4864LL * kMiB, 12288, 512, 64, true, std::nullopt, std::nullopt);
    MS_CHECK(!unconstrained.at("quota_constrained").get<bool>());
    MS_CHECK(unconstrained.at("allow_overflow").get<bool>());
    MS_CHECK(unconstrained.at("vmm_physical_pool_mib").get<int>() > 6688);

    const auto regime_a_tight = WorkerBudgetPlanner::compute_internvlu_memory_plan(
        40960.0, 26000.0, 12000.0, 2496LL * kMiB, 2560, 128, 64, true, std::nullopt, std::nullopt);
    MS_CHECK(!regime_a_tight.at("quota_constrained").get<bool>());
    MS_CHECK(regime_a_tight.at("vmm_physical_pool_mib").get<int>() <= 2816);

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

    // `twostage_shm` ships over SHM WITHOUT redundant CFG duplication on wire (`include_cfg_dup_on_wire == false`).
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
    MS_CHECK(shm_plan.include_cfg_dup_on_wire == false);
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

    // `slo_time_scale` tightens AR TPOT cap cleanly (80 ms passes at 1.0x=101.40ms, fails at 0.5x=50.70ms).
    const auto r_1x = MorphControlEngine::evaluate_session_slos(
        pol_p95, {200.0}, {80.0, 80.0}, {4000.0}, {512}, {20}, {{80.0, 80.0}}, 1.0);
    const auto r_half = MorphControlEngine::evaluate_session_slos(
        pol_p95, {200.0}, {80.0, 80.0}, {4000.0}, {512}, {20}, {{80.0, 80.0}}, 0.5);
    MS_CHECK(r_1x.at("ar_slo_met").get<bool>() == true);
    MS_CHECK(r_half.at("ar_slo_met").get<bool>() == false);

    // Modality-normalized AR TTFT in `evaluate_session_slos`: 300ms fails text-only (250ms), 1800ms passes 512^2 image (2800ms).
    const auto r_text_fail = MorphControlEngine::evaluate_session_slos(
        pol_p95, {300.0}, {60.0}, {}, {}, {}, {{60.0}}, 1.0, {0}, {0}, {512});
    const auto r_img_ok = MorphControlEngine::evaluate_session_slos(
        pol_p95, {1800.0}, {60.0}, {}, {}, {}, {{60.0}}, 1.0, {1}, {1}, {512});
    MS_CHECK(r_text_fail.at("ar_slo_met").get<bool>() == false);
    MS_CHECK(r_img_ok.at("ar_slo_met").get<bool>() == true);

    // 4. Canonical `aggregate_cluster_run_report`: cooldown window, actual-rate goodput, `vmm_physical_pool_mib`,
    //    real-time `alpha_timeline`, `latency_breakdown_ms`, `nvml_sm_util_mean_pct`, InternVL-U 16-step parity,
    //    and unbiased `per_turn_stats` + improper DiT CDF.
    const PolicyConfig pol_win = PolicyConfig::from_preset(
        "morphserve",
        8,
        {
            {"steady_window_mode", "time"},
            {"warmup_window_s", 60.0},
            {"steady_window_s", 180.0},
            {"cooldown_s", 30.0},
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
            {"resident_alpha_sample", 0.99},
            {"turn_records",
             nlohmann::json::array({
                 {{"turn_idx", 0}, {"modality_plan", "ar_only"}, {"status", st}, {"ar_ttft_ms", 220.0}, {"ar_tpots_ms", nlohmann::json::array({50.0, 52.0})}, {"queue_ms", 12.0}, {"preempt_yield_ms", 5.0}},
                 {{"turn_idx", 1}, {"modality_plan", "dit_edit"}, {"status", st}, {"dit_ms", 4200.0}, {"height", 512}, {"dit_steps", 20}, {"h2d_weight_ms", 40.0}, {"kv_transfer_ms", 18.0}, {"vae_ms", 52.6}},
             })},
        };
    };
    const nlohmann::json outcomes_win = nlohmann::json::array({
        mk_out("w0", 15.0, "oom", false),
        mk_out("s0", 80.0, "ok", true),
        mk_out("s1", 150.0, "ok", true),
        mk_out("d0", 260.0, "oom", false),
    });
    const nlohmann::json wstats = {
        {"nvml_sm_util_mean_pct", 78.5},
        {"alpha_timeline",
         nlohmann::json::array({
             {{"t_s", 12.0}, {"mean_alpha", 0.25}},
             {{"t_s", 85.0}, {"mean_alpha", 0.75}},
         })},
        {"workers",
         nlohmann::json::array({
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
         })},
    };
    const auto rep = MorphControlEngine::aggregate_cluster_run_report(
        pol_win, "unit_canonical_window", 0.5, 300.0, 0.0, outcomes_win, wstats);

    // Steady window: [60.0, min(240.0, 260.0 - 30.0)] == [60.0, 230.0] (170.0 s), actual-rate goodput = 2 / 170.0.
    MS_CHECK(rep.at("steady_state_sessions").get<int>() == 2);
    MS_CHECK(std::abs(rep.at("steady_window_end_s").get<double>() - 230.0) < 1e-6);
    const double expected_steady_rps = round_4(2.0 / 170.0);
    MS_CHECK(std::abs(rep.at("actual_steady_arrival_qps").get<double>() - expected_steady_rps) < 1e-6);
    MS_CHECK(std::abs(rep.at("joint_slo_goodput_rps").get<double>() - expected_steady_rps) < 1e-6);
    MS_CHECK(std::abs(rep.at("steady_joint_slo_goodput_rps").get<double>() - expected_steady_rps) < 1e-6);
    MS_CHECK(std::abs(rep.at("steady_turn_goodput_tps").get<double>() - round_4(4.0 / 170.0)) < 1e-6);
    MS_CHECK(std::abs(rep.at("joint_slo_attainment_pct").get<double>() - 100.0) < 1e-6);
    MS_CHECK(std::abs(rep.at("steady_joint_slo_attainment_pct").get<double>() - 100.0) < 1e-6);
    MS_CHECK(rep.at("actual_arrival_qps").get<double>() > 0.0);
    MS_CHECK(std::abs(rep.at("full_trace_joint_slo_attainment_pct").get<double>() - 50.0) < 1e-6);
    MS_CHECK(std::abs(rep.at("vmm_physical_pool_mib").get<double>() - 6688.0) < 1e-6);
    MS_CHECK(rep.at("kv_host_swap_outs").get<int>() == 6);
    MS_CHECK(rep.at("kv_host_swap_ins").get<int>() == 4);
    MS_CHECK(std::abs(rep.at("nvml_sm_util_mean_pct").get<double>() - 78.5) < 1e-6);
    MS_CHECK(std::abs(rep.at("latency_breakdown_ms").at("vae_ms").at("mean").get<double>() - 52.6) < 1e-6);
    MS_CHECK(std::abs(rep.at("latency_breakdown_ms").at("queue_ms").at("mean").get<double>() - 12.0) < 1e-6);
    // Verify real-time `alpha_timeline` took precedence over session `resident_alpha_sample = 0.99`.
    bool checked_alpha_bucket = false;
    for (const auto& b : rep.at("time_series_10s")) {
        if (std::abs(b.at("t_start_s").get<double>() - 80.0) < 1e-6) {
            MS_CHECK(std::abs(b.at("mean_alpha").get<double>() - 0.75) < 1e-6);
            checked_alpha_bucket = true;
        }
    }
    MS_CHECK(checked_alpha_bucket);
    MS_CHECK(std::abs(rep.at("slo_config").at("dit_1024_slo_ms").get<double>() - 57219.60) < 1e-6);
    MS_CHECK(std::abs(rep.at("slo_curve_by_dit_k").at("3.0x").at("dit_1024_slo_ms").get<double>() - 57219.60) < 1e-6);

    // InternVL-U 16-step 512px & 1024px parity in `slo_config`, `slo_curve_by_k`, and `slo_curve_by_dit_k`.
    const PolicyConfig pol_ivl_win = PolicyConfig::from_preset(
        "morphserve",
        8,
        {{"model_family", "internvl_u"}, {"steady_window_mode", "time"}, {"warmup_window_s", 60.0}, {"steady_window_s", 180.0}});
    const auto rep_ivl = MorphControlEngine::aggregate_cluster_run_report(
        pol_ivl_win, "unit_ivl_steps", 0.5, 300.0, 0.0, outcomes_win, wstats);
    MS_CHECK(std::abs(rep_ivl.at("slo_config").at("dit_512_slo_ms").get<double>() - 7484.40) < 1e-6);
    MS_CHECK(std::abs(rep_ivl.at("slo_config").at("dit_1024_slo_ms").get<double>() - 21374.40) < 1e-6);
    MS_CHECK(std::abs(rep_ivl.at("slo_curve_by_k").at("3.0x").at("dit_1024_slo_ms").get<double>() - 21374.40) < 1e-6);
    MS_CHECK(std::abs(rep_ivl.at("slo_curve_by_dit_k").at("3.0x").at("dit_1024_slo_ms").get<double>() - 21374.40) < 1e-6);

    // D1-D5 planned-turns denominator + unbiased `per_turn_stats` + improper DiT CDF with mid-session OOM.
    const PolicyConfig pol_trim = PolicyConfig::from_preset(
        "morphserve",
        8,
        {{"steady_window_mode", "session_trim"}, {"steady_state_trim_frac", 0.0}});
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
        {"ar_ttfts_ms", nlohmann::json::array({180.0, 200.0, 350.0})},
        {"ar_tpots_ms", nlohmann::json::array({55.0, 60.0, 58.0})},
        {"ar_turn_tpots_ms", nlohmann::json::array({{55.0}, {60.0}, {58.0}})},
        {"dit_turns_ms", nlohmann::json::array({12000.0, 17500.0})},
        {"dit_turn_heights", nlohmann::json::array({512, 512})},
        {"dit_turn_steps", nlohmann::json::array({20, 20})},
        {"ar_slo_met", true},
        {"dit_slo_met", false},
        {"joint_slo_met", false},
        {"turn_records",
         nlohmann::json::array({
             {{"turn_idx", 0}, {"modality_plan", "ar_only"}, {"status", "ok"}, {"ar_ttft_ms", 180.0}, {"ar_tpot_p95_ms", 55.0}, {"has_image", false}},
             {{"turn_idx", 1}, {"modality_plan", "cot_then_dit"}, {"status", "ok"}, {"ar_ttft_ms", 200.0}, {"ar_tpot_p95_ms", 60.0}, {"has_image", false}, {"dit_ms", 12000.0}, {"height", 512}, {"dit_steps", 20}},
             {{"turn_idx", 2}, {"modality_plan", "ar_only"}, {"status", "ok"}, {"ar_ttft_ms", 350.0}, {"ar_tpot_p95_ms", 58.0}, {"has_image", true}, {"num_images", 1}, {"image_res", 512}},
             {{"turn_idx", 3}, {"modality_plan", "dit_edit"}, {"status", "ok"}, {"dit_ms", 17500.0}, {"height", 512}, {"dit_steps", 20}},
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
        {"ar_ttfts_ms", nlohmann::json::array({190.0})},
        {"ar_tpots_ms", nlohmann::json::array({50.0})},
        {"dit_turns_ms", nlohmann::json::array()},
        {"ar_slo_met", false},
        {"dit_slo_met", false},
        {"joint_slo_met", false},
        {"turn_records",
         nlohmann::json::array({
             {{"turn_idx", 0}, {"modality_plan", "ar_only"}, {"status", "ok"}, {"ar_ttft_ms", 190.0}, {"ar_tpot_p95_ms", 50.0}, {"has_image", false}},
             {{"turn_idx", 1}, {"modality_plan", "dit_edit"}, {"status", "oom"}, {"height", 512}, {"dit_steps", 20}},
         })},
    };
    const auto rep_d = MorphControlEngine::aggregate_cluster_run_report(
        pol_trim, "unit_d1_d5", 0.5, 200.0, 0.0, nlohmann::json::array({out_a, out_b}), nlohmann::json::array());
    MS_CHECK(rep_d.at("steady_planned_turns").get<int>() == 8);
    MS_CHECK(rep_d.at("steady_completed_turns").get<int>() == 5);
    MS_CHECK(std::abs(rep_d.at("steady_turn_survival_pct").get<double>() - 62.5) < 1e-6);
    MS_CHECK(std::abs(rep_d.at("steady_turn_slo_attainment_pct").get<double>() - 50.0) < 1e-6);
    MS_CHECK(std::abs(rep_d.at("slo_curve_by_dit_k").at("2.0x").at("turn_attainment_pct").get<double>() - 37.5) < 1e-6);
    MS_CHECK(std::abs(rep_d.at("slo_curve_by_dit_k").at("4.0x").at("session_attainment_pct").get<double>() - 50.0) < 1e-6);
    // Unbiased per_turn_stats: turn_2 & turn_3 include dead session s_b (total == 2, unreached == 1, survival == 50%).
    MS_CHECK(rep_d.at("per_turn_stats").at("turn_2").at("total").get<int>() == 2);
    MS_CHECK(rep_d.at("per_turn_stats").at("turn_2").at("unreached").get<int>() == 1);
    MS_CHECK(std::abs(rep_d.at("per_turn_stats").at("turn_2").at("survival_pct").get<double>() - 50.0) < 1e-6);
    MS_CHECK(rep_d.at("per_turn_stats").at("turn_3").at("total").get<int>() == 2);
    MS_CHECK(rep_d.at("per_turn_stats").at("turn_3").at("unreached").get<int>() == 1);
    // Improper DiT CDF uses planned DiT turns (4) as denominator: 2 ok / 4 planned == 50.0%, p90 is null.
    MS_CHECK(std::abs(rep_d.at("dit_turn_survival_pct").get<double>() - 50.0) < 1e-6);
    MS_CHECK(std::abs(rep_d.at("all_session_dit_p50_ms").get<double>() - 17500.0) < 1e-6);
    MS_CHECK(rep_d.at("all_session_dit_p90_ms").is_null());

    // Session-level AR/DiT SLO denominators exclude sessions without that modality.
    const nlohmann::json pure_ar_fail = {
        {"session_id", "pure_ar"},
        {"arrival_offset_s", 1.0},
        {"status", "ok"},
        {"planned_turns", 1},
        {"planned_ar_turns", 1},
        {"planned_dit_turns", 0},
        {"ar_ttfts_ms", nlohmann::json::array({999.0})},
        {"ar_tpots_ms", nlohmann::json::array({50.0})},
        {"ar_slo_met", false},
        {"dit_slo_met", true},
        {"joint_slo_met", false},
    };
    const nlohmann::json pure_dit_ok = {
        {"session_id", "pure_dit"},
        {"arrival_offset_s", 2.0},
        {"status", "ok"},
        {"planned_turns", 1},
        {"planned_ar_turns", 0},
        {"planned_dit_turns", 1},
        {"dit_turns_ms", nlohmann::json::array({4500.0})},
        {"dit_turn_heights", nlohmann::json::array({512})},
        {"dit_turn_steps", nlohmann::json::array({20})},
        {"ar_slo_met", true},
        {"dit_slo_met", true},
        {"joint_slo_met", true},
    };
    const auto rep_mod = MorphControlEngine::aggregate_cluster_run_report(
        pol_trim, "unit_mod_denom", 1.0, 10.0, 0.0, nlohmann::json::array({pure_ar_fail, pure_dit_ok}), nlohmann::json::array());
    MS_CHECK(rep_mod.at("steady_sessions_with_ar").get<int>() == 1);
    MS_CHECK(rep_mod.at("steady_sessions_with_dit").get<int>() == 1);
    MS_CHECK(std::abs(rep_mod.at("steady_ar_slo_attainment_pct").get<double>() - 0.0) < 1e-6);
    MS_CHECK(std::abs(rep_mod.at("steady_dit_slo_attainment_pct").get<double>() - 100.0) < 1e-6);
    // New Mean latency + P99 improper CDF + policy-mode report fields.
    MS_CHECK(rep_mod.contains("ar_ttft_mean_ms") && std::abs(rep_mod.at("ar_ttft_mean_ms").get<double>() - 999.0) < 1e-6);
    MS_CHECK(rep_mod.contains("dit_turn_mean_ms") && std::abs(rep_mod.at("dit_turn_mean_ms").get<double>() - 4500.0) < 1e-6);
    MS_CHECK(rep_mod.contains("all_session_e2e_p99_ms"));
    MS_CHECK(rep_mod.contains("all_session_dit_p99_ms"));
    MS_CHECK(rep_mod.contains("all_session_active_jct_p99_ms"));
    MS_CHECK(rep_mod.contains("nvml_mem_util_mean_pct"));
    MS_CHECK(rep_mod.contains("effective_hbm_util_mean_pct"));
    MS_CHECK(rep_mod.contains("kv_swap_hit_rate_pct"));
}

void test_performance_atlas_3d_and_fallbacks() {
    using namespace morphserve::control;

    const PerformanceAtlas atlas = PerformanceAtlas::a100_default();
    const int64_t bagel_exp = 12446LL * kMiB;

    // 1. Live A100 Stage-1 calibrated anchors: 1024^2 alpha=0.0 -> 173.97 ms (vs legacy ~1058 ms).
    const auto a1024_0 = atlas.estimate_morph_stall("bagel", 1024, 0.0, false, bagel_exp, 12.34, 939.804, 11.71);
    const auto leg1024_0 = PerformanceAtlas::estimate_legacy_analytical(0.0, bagel_exp, 12.34, 939.804, 11.71, 0.90);
    MS_CHECK(a1024_0.from_atlas);
    MS_CHECK(std::abs(a1024_0.exposed_stall_ms - 173.97) < 0.05);
    MS_CHECK(leg1024_0.exposed_stall_ms > 1000.0);
    MS_CHECK(leg1024_0.exposed_stall_ms > 5.0 * a1024_0.exposed_stall_ms);

    // 2. Live A100 1024^2 alpha=0.5 steady (7.53 ms) and post-eviction alpha=0.75 (267.69 ms vs legacy 0 ms).
    const auto a1024_50 = atlas.estimate_morph_stall("bagel", 1024, 0.50, false, bagel_exp, 12.34, 939.804, 11.71);
    MS_CHECK(a1024_50.from_atlas && std::abs(a1024_50.exposed_stall_ms - 7.53) < 0.05);
    const auto a1024_pe = atlas.estimate_morph_stall("bagel", 1024, 0.75, true, bagel_exp, 12.34, 939.804, 11.71);
    const auto leg1024_75 = PerformanceAtlas::estimate_legacy_analytical(0.75, bagel_exp, 12.34, 939.804, 11.71, 0.90);
    MS_CHECK(a1024_pe.from_atlas && std::abs(a1024_pe.exposed_stall_ms - 267.69) < 0.05);
    MS_CHECK(leg1024_75.exposed_stall_ms < 1e-6);

    // 3. Smooth piecewise-linear interpolation at intermediate alpha=0.25 and resolution=768.
    const auto a1024_25 = atlas.estimate_morph_stall("bagel", 1024, 0.25, false, bagel_exp, 12.34, 939.804, 11.71);
    MS_CHECK(a1024_25.from_atlas);
    MS_CHECK(a1024_25.exposed_stall_ms > a1024_50.exposed_stall_ms &&
             a1024_25.exposed_stall_ms < a1024_0.exposed_stall_ms);
    const auto a768_50 = atlas.estimate_morph_stall("bagel", 768, 0.50, false, bagel_exp, 12.34, 600.0, 11.71);
    const auto a512_50 = atlas.estimate_morph_stall("bagel", 512, 0.50, false, bagel_exp, 12.34, 242.735, 11.71);
    MS_CHECK(a768_50.from_atlas);
    MS_CHECK(a768_50.exposed_stall_ms > a1024_50.exposed_stall_ms &&
             a768_50.exposed_stall_ms < a512_50.exposed_stall_ms);

    // 4. Uncalibrated hardware (e.g. H100 Gen5 32 GB/s or custom 6 GiB expert) falls back to pipeline analytical.
    const auto h100_fb = atlas.estimate_morph_stall("bagel", 1024, 0.0, false, bagel_exp, 32.0, 400.0, 6.0);
    MS_CHECK(!h100_fb.from_atlas);
    MS_CHECK(h100_fb.hidden_frac > 0.70);
    const auto custom_fb = atlas.estimate_morph_stall("bagel", 512, 0.50, false, 6000LL * kMiB, 12.34, 200.0, 12.0);
    MS_CHECK(!custom_fb.from_atlas);

    // 5. 3rd axis (`colocated_ar_b`) sublinear stretch & router Path-A/B awareness.
    const double s0 = atlas.dit_step_stretch_factor("bagel", 512, 0);
    const double s1 = atlas.dit_step_stretch_factor("bagel", 512, 1);
    const double s8 = atlas.dit_step_stretch_factor("bagel", 512, 8);
    MS_CHECK(std::abs(s0 - 1.0) < 1e-9);
    MS_CHECK(s1 > 1.0 && s8 > s1 && s8 < 2.0 * s1);

    const auto topo = ClusterTopologyProfile::a100_pcie_8gpu_default();
    DualPathMorphRouter r_atlas(topo, nullptr, default_operator_profile(), std::nullopt, 11.71, 0.90, true, true, 128LL * kMiB, "atlas_3d");
    DualPathMorphRouter r_legacy(topo, nullptr, default_operator_profile(), std::nullopt, 11.71, 0.90, true, true, 128LL * kMiB, "legacy_analytical");

    // 1024^2 alpha=0.0 with 1.5 GiB KV: legacy overestimates Path B (~1070 ms) and ships via Path A;
    // atlas_3d knows in-place morph is only ~185 ms and stays in-place!
    std::map<int, WorkerState> ws_map;
    WorkerState w0;
    w0.gpu_id = 0;
    w0.expert_alpha = 0.0;
    w0.kv_free_bytes = 16LL * kGiB;
    WorkerState w1;
    w1.gpu_id = 1;
    w1.expert_alpha = 1.0;
    w1.kv_free_bytes = 16LL * kGiB;
    ws_map[0] = w0;
    ws_map[1] = w1;

    CPTGRequestSpec req1024;
    req1024.req_id = "atlas_vs_legacy_1024";
    req1024.src_gpu = 0;
    req1024.architecture = "tight_mot";
    req1024.model_family = "bagel";
    req1024.kv_bytes = 1600LL * kMiB;
    req1024.gen_expert_bytes = bagel_exp;
    req1024.dit_step1_ms = 939.804;
    req1024.resolution = 1024;
    req1024.interleaved_roundtrip_gamma = 2;

    const RouteDecision d_atlas = r_atlas.decide(req1024, &ws_map, 0.0);
    const RouteDecision d_legacy = r_legacy.decide(req1024, &ws_map, 0.0);
    MS_CHECK(d_atlas.path == "in_place_morph" && d_atlas.t_morph_ms < 300.0);
    MS_CHECK(d_legacy.path == "state_ship" && d_legacy.t_morph_ms > 5.0 * d_atlas.t_morph_ms);

    // Bounded single-turn colocated AR stretch: a 60s queue with 1 active AR turn stretches ONLY
    // the currently executing turn (~5.6s * 0.48 = ~2.7s), NOT the entire 60s queue (+28.8s)!
    WorkerState w_long;
    w_long.gpu_id = 0;
    w_long.expert_alpha = 1.0;
    w_long.queue_wait_ms = 60000.0;
    w_long.colocated_ar_b = 1;
    CPTGRequestSpec req512;
    req512.req_id = "bounded_coloc_stretch";
    req512.src_gpu = 0;
    req512.model_family = "bagel";
    req512.resolution = 512;
    req512.dit_step1_ms = 280.0;
    req512.dit_total_steps = 20;
    const double q_stretched_ms = r_atlas.evaluate_path_b_morph_ms(req512, w_long, 0.0).first;
    MS_CHECK(q_stretched_ms > 62000.0 && q_stretched_ms < 64000.0);

    // JSON round-trip preserves exact entries.
    const PerformanceAtlas rt = PerformanceAtlas::from_json(atlas.to_json());
    const auto rt_pt = rt.estimate_morph_stall("bagel", 1024, 0.0, false, bagel_exp, 12.34, 939.804, 11.71);
    MS_CHECK(rt_pt.from_atlas && std::abs(rt_pt.exposed_stall_ms - 173.97) < 0.05);
}

void test_gpu_wait_queue_scheduler() {
    using namespace morphserve::control;

    // 1. FCFS mode strictly follows enqueue order.
    GpuWaitQueueScheduler fcfs_sched(GpuQueuePolicy{"fcfs", 6000.0, 3});
    std::vector<GpuQueueWaiter> q_fcfs = {
        {1, 100.0, 8LL * kGiB, 9000.0, 50000.0, 1, 0},
        {2, 110.0, 1LL * kGiB, 2000.0, 5000.0, 1, 0},
    };
    const auto s_fcfs = fcfs_sched.select_and_pop(q_fcfs, 120.0, 4LL * kGiB);
    MS_CHECK(s_fcfs.has_value() && s_fcfs->waiter_id == 1 && !s_fcfs->reordered);

    // 2. Memory+Slack mode skips memory-exceeding FIFO head to run fitting request (0-OOM bypass).
    GpuWaitQueueScheduler ms_sched(GpuQueuePolicy{"memory_slack", 6000.0, 2});
    std::vector<GpuQueueWaiter> q_ms = {
        {10, 100.0, 8LL * kGiB, 9000.0, 60000.0, 1, 0},  // does NOT fit in 4 GiB
        {11, 110.0, 2LL * kGiB, 4000.0, 20000.0, 1, 0},  // fits, looser slack
        {12, 120.0, 1LL * kGiB, 3000.0, 6000.0, 1, 0},   // fits, tightest slack
    };
    const auto sel1 = ms_sched.select_and_pop(q_ms, 200.0, 4LL * kGiB);
    MS_CHECK(sel1.has_value() && sel1->waiter_id == 12 && sel1->reordered && sel1->memory_bypass);
    MS_CHECK(ms_sched.stats().memory_bypasses_total == 1);
    // Older waiters 10 & 11 had bypass_count incremented.
    MS_CHECK(q_ms[0].waiter_id == 10 && q_ms[0].bypass_count == 1);
    MS_CHECK(q_ms[1].waiter_id == 11 && q_ms[1].bypass_count == 1);

    // Next dispatch: waiter 11 fits while 10 doesn't -> bypasses 10 a second time (bypass_count == 2 == max_bypasses).
    const auto sel2 = ms_sched.select_and_pop(q_ms, 300.0, 4LL * kGiB);
    MS_CHECK(sel2.has_value() && sel2->waiter_id == 11 && sel2->memory_bypass);
    MS_CHECK(q_ms.size() == 1 && q_ms[0].waiter_id == 10 && q_ms[0].bypass_count == 2);

    // Anti-starvation: add a new small waiter 13; waiter 10 hit max_bypasses=2 -> MUST be promoted!
    q_ms.push_back({13, 310.0, 1LL * kGiB, 1000.0, 2000.0, 1, 0});
    const auto sel3 = ms_sched.select_and_pop(q_ms, 320.0, 4LL * kGiB);
    MS_CHECK(sel3.has_value() && sel3->waiter_id == 10 && sel3->starvation_promoted);
    MS_CHECK(ms_sched.stats().starvation_promotions_total == 1);

    // Deadlock-free fallback when NO waiter fits live_free_mem_bytes: picks smallest footprint.
    std::vector<GpuQueueWaiter> q_tight = {
        {20, 100.0, 10LL * kGiB, 5000.0, 20000.0, 1, 0},
        {21, 110.0, 5LL * kGiB, 5000.0, 20000.0, 1, 0},
    };
    const auto sel_fb = ms_sched.select_and_pop(q_tight, 120.0, 1LL * kGiB);
    MS_CHECK(sel_fb.has_value() && sel_fb->waiter_id == 21);

    // Domino-free Feasible-Slack EDF + SRPT fallback: already-expired heavy 1024^2 waiter MUST NOT
    // preempt an SLO-feasible 512^2 waiter, and among expired waiters shorter work runs first (SRPT).
    GpuWaitQueueScheduler dom_sched(GpuQueuePolicy{"memory_slack", 45000.0, 5});
    std::vector<GpuQueueWaiter> q_dom = {
        {30, 100.0, 2LL * kGiB, 24000.0, 10000.0, 1, 0},  // already expired (slack = -19000), heavy
        {31, 110.0, 1LL * kGiB, 5000.0, 12000.0, 1, 0},   // feasible (slack = +2000)
        {32, 120.0, 1LL * kGiB, 4000.0, 6000.0, 1, 0},    // already expired (slack = -3000), short
    };
    const auto d1 = dom_sched.select_and_pop(q_dom, 5000.0, 8LL * kGiB);
    MS_CHECK(d1.has_value() && d1->waiter_id == 31);
    const auto d2 = dom_sched.select_and_pop(q_dom, 5000.0, 8LL * kGiB);
    MS_CHECK(d2.has_value() && d2->waiter_id == 32);
    const auto d3 = dom_sched.select_and_pop(q_dom, 5000.0, 8LL * kGiB);
    MS_CHECK(d3.has_value() && d3->waiter_id == 30);

    // Session-SLO-aware Moore-Hodgson Session-SRPT + EDF-Feasible-Pair Interchange:
    // 1. Under overload (running tighter-slack multi-turn job would break best_srpt's deadline):
    //    best_srpt (minimum remaining session work `rem_turns * est_exec_ms`) runs FIRST!
    // 2. Under non-overload (best_srpt stays comfortably feasible after tighter-slack job runs):
    //    EDF interchange runs tighter-slack job first so BOTH finish on time!
    // 3. Bucket 0 (intact) > Bucket 1 (prior turn missed, EDF) > Bucket 2 (expired, SRPT).
    GpuWaitQueueScheduler sess_sched(GpuQueuePolicy{"memory_slack", 120000.0, 16});
    std::vector<GpuQueueWaiter> q_overload = {
        {40, 100.0, 1LL * kGiB, 5000.0, 11000.0, 1, 0, false, 1}, // Bucket 1: feasible slack=+1000, prior missed
        {41, 110.0, 1LL * kGiB, 5000.0, 11000.0, 1, 0, true, 3},  // Bucket 0: slack=+1000, work=15000 (would kill 44!)
        {44, 115.0, 1LL * kGiB, 5000.0, 12000.0, 1, 0, true, 1},  // Bucket 0: slack=+2000, work=5000 (best_srpt!)
        {42, 120.0, 1LL * kGiB, 7100.0, 24000.0, 1, 0, true, 1},  // Bucket 0: heavy 1024px work=7100, slack=+11900
        {43, 130.0, 1LL * kGiB, 3000.0, 6000.0, 1, 0, true, 1},   // Bucket 2: expired slack=-2000
    };
    const auto s0 = sess_sched.select_and_pop(q_overload, 5000.0, 8LL * kGiB);
    MS_CHECK(s0.has_value() && s0->waiter_id == 44);  // 44.slack(2000) < 41.exec(5000) -> 44 banks Joint SLO!
    const auto s1 = sess_sched.select_and_pop(q_overload, 5000.0, 8LL * kGiB);
    MS_CHECK(s1.has_value() && s1->waiter_id == 41);  // Now best_srpt=42 (slack=11900 >= 1.0*5000) -> 41 EDF-interchanges cleanly!
    const auto s2 = sess_sched.select_and_pop(q_overload, 5000.0, 8LL * kGiB);
    MS_CHECK(s2.has_value() && s2->waiter_id == 42);
    const auto s3 = sess_sched.select_and_pop(q_overload, 5000.0, 8LL * kGiB);
    MS_CHECK(s3.has_value() && s3->waiter_id == 40);
    const auto s4 = sess_sched.select_and_pop(q_overload, 5000.0, 8LL * kGiB);
    MS_CHECK(s4.has_value() && s4->waiter_id == 43);

    // 3-waiter overload anti-chasing: two tight-slack 3-turn jobs (51, 52) cannot both meet their
    // deadlines -> EDF is jointly infeasible -> pure Session-SRPT banks 1-turn job 53 immediately
    // even though 53 alone could tolerate one interchange!
    std::vector<GpuQueueWaiter> q_chase = {
        {51, 100.0, 1LL * kGiB, 5000.0, 11000.0, 1, 0, true, 3},   // EDF: 5000+5000=10000 <= 11000
        {52, 110.0, 1LL * kGiB, 5000.0, 12000.0, 1, 0, true, 3},   // EDF: 10000+5000=15000 > 12000 (OVERLOAD!)
        {53, 120.0, 1LL * kGiB, 5000.0, 22000.0, 1, 0, true, 1},   // best_srpt (rem_work=5000)
    };
    const auto sc0 = sess_sched.select_and_pop(q_chase, 5000.0, 8LL * kGiB);
    MS_CHECK(sc0.has_value() && sc0->waiter_id == 53);

    // Under overload, minimum `remaining_turns` runs BEFORE `rem_turns * current_exec_ms`:
    // a 1-turn 1024px session (18s) must not be starved behind a 4-turn session whose current
    // turn happens to be a cheap 512px turn (4s * 4 = 16s < 18s)!
    std::vector<GpuQueueWaiter> q_rem_first = {
        {61, 100.0, 1LL * kGiB, 4000.0, 10000.0, 1, 0, true, 4},   // 4 turns left, short current turn
        {62, 110.0, 1LL * kGiB, 4000.0, 10500.0, 1, 0, true, 4},   // EDF: 5000+4000+4000=13000 > 10500 (OVERLOAD!)
        {63, 120.0, 1LL * kGiB, 18000.0, 40000.0, 1, 0, true, 1},  // 1 turn left, heavy 1024px -> MUST win!
    };
    const auto sr0 = sess_sched.select_and_pop(q_rem_first, 5000.0, 8LL * kGiB);
    MS_CHECK(sr0.has_value() && sr0->waiter_id == 63);

    // JSON roundtrip & backward-compatible defaults for session_slo_intact / remaining_turns.
    const GpuQueueWaiter w_rt = GpuQueueWaiter::from_json(
        GpuQueueWaiter{99, 10.0, 2LL * kGiB, 100.0, 500.0, 1, 0, false, 3}.to_json());
    MS_CHECK(w_rt.session_slo_intact == false && w_rt.remaining_turns == 3);
    const GpuQueueWaiter w_legacy = GpuQueueWaiter::from_json(nlohmann::json{{"waiter_id", 7}});
    MS_CHECK(w_legacy.session_slo_intact == true && w_legacy.remaining_turns == 1);
}

}  // namespace

int main() {
    test_hash_util_and_operator_profile();
    test_policy_presets_and_overrides();
    test_router_seams();
    test_preempt_edf_controller();
    test_worker_budget_planner();
    test_morph_control_engine_transitions_slos_and_aggregation();
    test_performance_atlas_3d_and_fallbacks();
    test_gpu_wait_queue_scheduler();

    if (g_failures > 0) {
        std::cerr << "[FAILED] test_control_engine (" << g_failures << " checks failed)\n";
        return 1;
    }
    std::cout << "[PASS] test_control_engine\n";
    return 0;
}
