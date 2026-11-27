#include "morphserve/c_api.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr size_t kMiB = 1ULL << 20;

int g_failures = 0;

void check_impl(bool cond, const char* expr, const char* file, int line) {
    if (!cond) {
        ++g_failures;
        std::cerr << "[FAIL] " << file << ":" << line << ": " << expr << "\n";
    }
}

#define MS_CHECK(cond) check_impl(static_cast<bool>(cond), #cond, __FILE__, __LINE__)

nlohmann::json take_json(char* raw) {
    MS_CHECK(raw != nullptr);
    if (!raw) {
        return nlohmann::json{};
    }
    const std::string s(raw);
    ms_free_cstr(raw);
    return nlohmann::json::parse(s);
}

MsVmmConfigC make_sim_vmm_cfg(size_t super_mib, size_t fine_mib, size_t super_handle_mib = 64) {
    MsVmmConfigC cfg{};
    cfg.device = 0;
    cfg.super_pool_mib = super_mib;
    cfg.fine_kv_pool_mib = fine_mib;
    cfg.super_handle_mib = super_handle_mib;
    cfg.transient_va_gib = 1;
    cfg.kv_va_gib = 1;
    cfg.allow_overflow = 0;
    cfg.simulated = 1;
    cfg.materialize_sim_va = 0;
    cfg.kv_swap_high_watermark = 0.85;
    cfg.kv_swap_low_watermark = 0.65;
    cfg.kv_swap_min_idle_s = 0.0;
    cfg.max_pool_bytes = 0;
    return cfg;
}

void test_vmm_null_guards_and_tripool_pluggable_safety() {
    MS_CHECK(ms_vmm_create(nullptr) == nullptr);
    MS_CHECK(ms_last_error_code() == MS_ERR_INVALID_ARG);
    MS_CHECK(std::strlen(ms_last_error()) > 0);

    MS_CHECK(ms_vmm_kv_alloc_page(nullptr) < 0);
    MS_CHECK(ms_last_error_code() == MS_ERR_INVALID_ARG);

    MS_CHECK(ms_vmm_kv_region_ensure(nullptr, 0, 2 * kMiB, nullptr) == 0);
    MS_CHECK(ms_last_error_code() == MS_ERR_INVALID_ARG);

    MS_CHECK(ms_vmm_transient_alloc(nullptr, 4096, nullptr) == nullptr);
    MS_CHECK(ms_last_error_code() == MS_ERR_INVALID_ARG);

    MS_CHECK(ms_vmm_coalesce_fine_to_super(nullptr, 0) == 0);
    MS_CHECK(ms_last_error_code() == MS_ERR_INVALID_ARG);

    MS_CHECK(ms_vmm_ensure_expert_headroom(nullptr, 0, 4096, nullptr, 0, nullptr) == 0);
    MS_CHECK(ms_last_error_code() == MS_ERR_INVALID_ARG);

    ms_vmm_kv_session_set_pinned(nullptr, 1, 1);

    // PyTorch `CUDAPluggableAllocator` C ABI entrypoints fail cleanly without unwinding when unbound.
    MS_CHECK(tripool_alloc(4096, 7, nullptr) == nullptr);
    MS_CHECK(ms_last_error_code() == MS_ERR_INVALID_ARG);
    tripool_free(nullptr, 0, 7, nullptr);
}

void test_vmm_coalesce_oom_pinning_multi_exclude_and_headroom() {
    // 3 x 64 MiB super + 32 x 2 MiB fine = 256 MiB total physical pool
    MsVmmConfigC cfg = make_sim_vmm_cfg(192, 64, 64);
    ms_vmm_t* h = ms_vmm_create(&cfg);
    MS_CHECK(h != nullptr);
    if (!h) return;
    MS_CHECK(ms_vmm_is_simulated(h) == 1);
    MS_CHECK(ms_vmm_device(h) == 0);

    // 1. Bounded vs unbounded `ms_vmm_coalesce_fine_to_super`
    const uint64_t sid_carve = ms_vmm_kv_session_create(h);
    const int32_t rid_carve = ms_vmm_kv_region_alloc(h, 256 * kMiB, sid_carve);
    MS_CHECK(rid_carve >= 0);
    MS_CHECK(ms_vmm_kv_region_ensure(h, rid_carve, 256 * kMiB, nullptr) != 0);

    UnifiedVmmStatsC st{};
    ms_vmm_get_stats(h, &st);
    MS_CHECK(st.super_to_fine_carves == 3);
    MS_CHECK(st.super_pool_total_handles == 0);
    MS_CHECK(st.fine_pool_total_pages == 128);

    // Pool exhaustion while pinned -> structured `MS_ERR_OOM`
    ms_vmm_kv_session_set_pinned(h, sid_carve, 1);
    MS_CHECK(ms_vmm_kv_alloc_page(h) < 0);
    MS_CHECK(ms_last_error_code() == MS_ERR_OOM);

    ms_vmm_scope_begin(h);
    MS_CHECK(ms_vmm_transient_alloc(h, static_cast<ssize_t>(64 * kMiB), nullptr) == nullptr);
    MS_CHECK(ms_last_error_code() == MS_ERR_OOM);
    MS_CHECK(ms_vmm_scope_end(h, nullptr) == 0);

    const int eid = ms_vmm_expert_register(h, 2, 64, nullptr, 0);
    MS_CHECK(eid == 0);
    MS_CHECK(ms_vmm_expert_load_layers_sync(h, eid, 0, 1) < 0);
    MS_CHECK(ms_last_error_code() == MS_ERR_OOM);

    // Free carved KV and verify bounded (`max_super=1`) then unbounded (`max_super=0`) coalescing
    MS_CHECK(ms_vmm_kv_region_free(h, rid_carve, nullptr) == 0);
    ms_vmm_kv_session_destroy(h, sid_carve);

    MS_CHECK(ms_vmm_coalesce_fine_to_super(h, 1) == 1);
    ms_vmm_get_stats(h, &st);
    MS_CHECK(st.super_pool_total_handles == 1);
    MS_CHECK(st.fine_pool_total_pages == 96);

    MS_CHECK(ms_vmm_coalesce_fine_to_super(h, 0) == 2);
    ms_vmm_get_stats(h, &st);
    MS_CHECK(st.super_pool_total_handles == 3);
    MS_CHECK(st.fine_pool_total_pages == 32);
    MS_CHECK(ms_vmm_coalesce_fine_to_super(h, 0) == 0);

    // 2. Multi-session exclusion (`reclaim_thinking_kv_bytes_multi` & `maybe_evict_watermark_multi`)
    const uint64_t s1 = ms_vmm_kv_session_create(h);
    const uint64_t s2 = ms_vmm_kv_session_create(h);
    const uint64_t s3 = ms_vmm_kv_session_create(h);
    const int32_t r1 = ms_vmm_kv_region_alloc(h, 64 * kMiB, s1);
    const int32_t r2 = ms_vmm_kv_region_alloc(h, 64 * kMiB, s2);
    const int32_t r3 = ms_vmm_kv_region_alloc(h, 64 * kMiB, s3);
    MS_CHECK(ms_vmm_kv_region_ensure(h, r1, 64 * kMiB, nullptr) != 0);
    MS_CHECK(ms_vmm_kv_region_ensure(h, r2, 64 * kMiB, nullptr) != 0);
    MS_CHECK(ms_vmm_kv_region_ensure(h, r3, 64 * kMiB, nullptr) != 0);

    const size_t valid_64m = 64 * kMiB;
    ms_vmm_kv_session_mark_thinking(h, s1, &r1, &valid_64m, 1, 0, nullptr);
    ms_vmm_kv_session_mark_thinking(h, s2, &r2, &valid_64m, 1, 0, nullptr);
    ms_vmm_kv_session_mark_thinking(h, s3, &r3, &valid_64m, 1, 0, nullptr);

    // Excluding BOTH s1 and s2 must evict ONLY s3
    const uint64_t excl_12[2] = {s1, s2};
    const size_t reclaimed = ms_vmm_reclaim_thinking_kv_bytes_multi(h, 64 * kMiB, 1, excl_12, 2, nullptr);
    MS_CHECK(reclaimed >= 64 * kMiB);
    MS_CHECK(ms_vmm_kv_region_is_swapped(h, r1) == 0);
    MS_CHECK(ms_vmm_kv_region_is_swapped(h, r2) == 0);
    MS_CHECK(ms_vmm_kv_region_is_swapped(h, r3) == 1);

    // Restore s3, re-mark thinking, and test `maybe_evict_watermark_multi` excluding s1 and s3
    MS_CHECK(ms_vmm_kv_session_mark_active(h, s3, nullptr) == 0);
    MS_CHECK(ms_vmm_kv_region_is_swapped(h, r3) == 0);
    ms_vmm_kv_session_mark_thinking(h, s3, &r3, &valid_64m, 1, 0, nullptr);

    const uint64_t excl_13[2] = {s1, s3};
    const size_t wm_evicted = ms_vmm_maybe_evict_watermark_multi(h, 1, excl_13, 2, 64 * kMiB, 0, nullptr);
    MS_CHECK(wm_evicted >= 64 * kMiB);
    MS_CHECK(ms_vmm_kv_region_is_swapped(h, r1) == 0);
    MS_CHECK(ms_vmm_kv_region_is_swapped(h, r2) == 1);
    MS_CHECK(ms_vmm_kv_region_is_swapped(h, r3) == 0);

    // 3. `kv_session_set_pinned` + `ensure_expert_headroom`
    // Pin s1, keep s3 unpinned, exclude s3 -> s1 is protected from reclaim
    ms_vmm_kv_session_set_pinned(h, s1, 1);
    const uint64_t excl_3[1] = {s3};
    MS_CHECK(ms_vmm_reclaim_thinking_kv_bytes_multi(h, 64 * kMiB, 1, excl_3, 1, nullptr) == 0);
    MS_CHECK(ms_vmm_kv_region_is_swapped(h, r1) == 0);

    // Unpin s1 -> `ensure_expert_headroom` for 2 x 64 MiB (128 MiB) reclaims s1 while excluding s3
    ms_vmm_kv_session_set_pinned(h, s1, 0);
    const size_t exp_reclaimed = ms_vmm_ensure_expert_headroom(h, eid, 128 * kMiB, excl_3, 1, nullptr);
    MS_CHECK(exp_reclaimed >= 64 * kMiB || ms_vmm_kv_free_bytes(h, 1, 0) >= 128 * kMiB);
    MS_CHECK(ms_vmm_kv_region_is_swapped(h, r3) == 0);
    MS_CHECK(ms_vmm_expert_load_layers_sync(h, eid, 0, 2) == 0);
    MS_CHECK(ms_vmm_expert_resident_count(h, eid) == 2);

    MS_CHECK(ms_vmm_kv_region_free(h, r1, nullptr) == 0);
    MS_CHECK(ms_vmm_kv_region_free(h, r2, nullptr) == 0);
    MS_CHECK(ms_vmm_kv_region_free(h, r3, nullptr) == 0);
    ms_vmm_kv_session_destroy(h, s1);
    ms_vmm_kv_session_destroy(h, s2);
    ms_vmm_kv_session_destroy(h, s3);
    MS_CHECK(ms_vmm_shutdown(h) == 0);
    ms_vmm_destroy(h);
}

void test_policy_and_sim_override_forwarding() {
    // 1. `static_2und_6gen` preset + invalid preset error code
    const auto split26 = take_json(ms_policy_from_preset_json("static_2und_6gen", 8, nullptr));
    MS_CHECK(split26.at("policy_name").get<std::string>() == "static_2und_6gen");
    MS_CHECK(split26.at("static_und_ranks").get<std::vector<int>>() == (std::vector<int>{0, 4}));
    MS_CHECK(split26.at("static_gen_ranks").get<std::vector<int>>() == (std::vector<int>{1, 2, 3, 5, 6, 7}));

    MS_CHECK(ms_policy_from_preset_json("nonexistent_policy_xyz", 8, nullptr) == nullptr);
    MS_CHECK(ms_last_error_code() == MS_ERR_INVALID_ARG);

    // 1b. `monolithic_kv_swap` preset exposes `use_idle_kv_swap=true` & `cooldown_s=30.0`
    const auto mono_swap = take_json(ms_policy_from_preset_json("monolithic_kv_swap", 8, nullptr));
    MS_CHECK(mono_swap.at("policy_name").get<std::string>() == "monolithic_kv_swap");
    MS_CHECK(mono_swap.at("use_idle_kv_swap").get<bool>() == true);
    MS_CHECK(std::fabs(mono_swap.at("cooldown_s").get<double>() - 30.0) < 1e-6);

    // 2. `ms_policy_build_ablation_configs_json` (full + 6 decoupled single-mechanism LOO ablations)
    const auto ablations = take_json(ms_policy_build_ablation_configs_json(8, "{\"slo_scale_k\":2.5}"));
    MS_CHECK(ablations.is_array() && ablations.size() == 7);
    if (ablations.size() == 7) {
        MS_CHECK(ablations[0].at("label").get<std::string>() == "full_morphserve");
        MS_CHECK(ablations[1].at("label").get<std::string>() == "no_m1_tripool_vmm");
        MS_CHECK(ablations[2].at("label").get<std::string>() == "no_m1_scoped_vae");
        MS_CHECK(ablations[3].at("label").get<std::string>() == "no_m2_dual_path");
        MS_CHECK(ablations[4].at("label").get<std::string>() == "no_m2_pacer_numa");
        MS_CHECK(ablations[5].at("label").get<std::string>() == "no_m3_cow_loopback");
        MS_CHECK(ablations[6].at("label").get<std::string>() == "no_m4_layer_chunk");
        MS_CHECK(std::fabs(ablations[0].at("config").at("slo_scale_k").get<double>() - 2.5) < 1e-6);
        MS_CHECK(ablations[1].at("config").at("use_tripool_vmm").get<bool>() == false);
        MS_CHECK(ablations[1].at("config").at("use_scoped_vae").get<bool>() == true);
        MS_CHECK(ablations[2].at("config").at("use_scoped_vae").get<bool>() == false);
        MS_CHECK(ablations[3].at("config").at("routing_mode").get<std::string>() == "sticky_least_loaded");
        MS_CHECK(ablations[3].at("config").at("use_h2d_pacer").get<bool>() == true);
        MS_CHECK(ablations[4].at("config").at("use_h2d_pacer").get<bool>() == false);
        MS_CHECK(ablations[4].at("config").at("use_shared_numa_mirror").get<bool>() == false);
        MS_CHECK(ablations[5].at("config").at("use_module3_cow_loopback").get<bool>() == false);
        MS_CHECK(ablations[6].at("config").at("preemption_mode").get<std::string>() == "non_preemptive");
    }

    // 3. `ms_policy_build_static_split_configs_json`
    const auto splits = take_json(ms_policy_build_static_split_configs_json(8, nullptr));
    MS_CHECK(splits.is_array() && splits.size() == 4);
    if (splits.size() == 4) {
        MS_CHECK(splits[0].at("label").get<std::string>() == "morphserve_elastic");
        MS_CHECK(splits[1].at("label").get<std::string>() == "static_2und_6gen");
        MS_CHECK(splits[2].at("label").get<std::string>() == "static_4und_4gen");
        MS_CHECK(splits[3].at("label").get<std::string>() == "static_6und_2gen");
    }
    MS_CHECK(ms_policy_build_static_split_configs_json(3, nullptr) == nullptr);
    MS_CHECK(ms_last_error_code() == MS_ERR_INVALID_ARG);

    // 4. `ms_sim_point_json` MUST NOT drop `dit_slo_scale_k` (inside `policy_overrides` or top-level)
    const nlohmann::json hw = {
        {"name", "unit_a100_8gpu"},
        {"gpu_hbm_gib", 40.0},
        {"compute_speedup_x", 1.0},
        {"solo_h2d_gbps", 8.755},
        {"agg_h2d_gbps", 10.348},
        {"intra_half_p2p_gbps", 10.581},
        {"inter_socket_p2p_gbps", 9.55},
        {"inter_node_rdma_gbps", 11.5},
        {"gpus_per_node", 8},
    };
    const nlohmann::json sessions = nlohmann::json::array({
        {{"arrival_ms", 0.0},
         {"quadrant", "visual_cot"},
         {"session_id", "s0"},
         {"phase_label", "steady"},
         {"turns", nlohmann::json::array({{{"modality_plan", "dit_t2i"}, {"height", 512}, {"dit_steps", 20}}})}},
        {{"arrival_ms", 500.0},
         {"quadrant", "visual_cot"},
         {"session_id", "s1"},
         {"phase_label", "steady"},
         {"turns", nlohmann::json::array({{{"modality_plan", "dit_t2i"}, {"height", 512}, {"dit_steps", 20}}})}},
    });
    const nlohmann::json req_tight = {
        {"policy_name", "morphserve"},
        {"num_gpus", 4},
        {"hw", hw},
        {"offered_qps", 0.5},
        {"num_sessions", 2},
        {"seed", 42},
        {"custom_sessions", sessions},
        {"policy_overrides",
         {{"dit_slo_scale_k", 0.1},
          {"steady_window_mode", "trim"},
          {"steady_state_trim_frac", 0.0}}},
    };
    const nlohmann::json req_loose = {
        {"policy_name", "morphserve"},
        {"num_gpus", 4},
        {"hw", hw},
        {"offered_qps", 0.5},
        {"num_sessions", 2},
        {"seed", 42},
        {"custom_sessions", sessions},
        {"dit_slo_scale_k", 20.0},
        {"steady_window_mode", "trim"},
        {"steady_state_trim_frac", 0.0},
    };
    const auto rep_tight = take_json(ms_sim_point_json(req_tight.dump().c_str()));
    const auto rep_loose = take_json(ms_sim_point_json(req_loose.dump().c_str()));
    MS_CHECK(std::fabs(rep_tight.at("slo_config").at("k_dit").get<double>() - 0.1) < 1e-6);
    MS_CHECK(std::fabs(rep_loose.at("slo_config").at("k_dit").get<double>() - 20.0) < 1e-6);
    MS_CHECK(rep_tight.at("slo_config").at("dit_512_slo_ms").get<double>() <
             rep_loose.at("slo_config").at("dit_512_slo_ms").get<double>());
    MS_CHECK(rep_tight.at("dit_slo_attainment_pct").get<double>() < 1.0);
    MS_CHECK(rep_loose.at("dit_slo_attainment_pct").get<double>() > 99.0);
}

void test_router_and_control_engine_c_api() {
    // 1. Default operator profile JSON
    const auto op_def = take_json(ms_topology_default_operator_profile_json());
    MS_CHECK(op_def.at("bagel_dit_512_step_ms").get<double>() > 0.0);
    MS_CHECK(op_def.at("internvlu_expert_bytes").get<int64_t>() > 0);

    const auto topo_j = take_json(ms_topology_a100_default_json());
    const std::string topo_s = topo_j.dump();

    // Custom operator profile forwarded through `ms_router_create`
    nlohmann::json custom_op = op_def;
    custom_op["bagel_dit_512_step_ms"] = 500.0;
    custom_op["bagel_vae_512_ms"] = 300.0;
    custom_op["prep_overhead_ms"] = 100.0;
    const nlohmann::json router_opts = {{"operator_profile", custom_op}};

    const int gpus[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const int groups[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    ms_pacer_t* pacer = ms_pacer_create(gpus, groups, 8, 1, "/tmp/morphserve_test_c_api_pacer");
    MS_CHECK(pacer != nullptr);

    ms_router_t* router = ms_router_create(topo_s.c_str(), pacer, router_opts.dump().c_str());
    MS_CHECK(router != nullptr);
    if (!router) {
        ms_pacer_destroy(pacer);
        return;
    }

    const nlohmann::json init_req = {
        {"num_gpus", 8},
        {"initial_alphas", {{"0", 0.25}, {"1", 1.0}, {"2", 1.0}, {"3", 1.0}, {"4", 1.0}, {"5", 1.0}, {"6", 1.0}, {"7", 1.0}}},
        {"kv_free_bytes", {{"0", 16LL * 1024 * kMiB}, {"1", 16LL * 1024 * kMiB}}},
    };
    const auto init_ws = take_json(ms_router_init_workers_json(router, init_req.dump().c_str()));
    MS_CHECK(init_ws.size() == 8);

    // 2. `ms_router_reserve_ar_queue` / `ms_router_release_ar_queue`
    const auto before_ar = take_json(ms_router_get_workers_json(router));
    const double q0_before = before_ar.at("0").at("queue_wait_ms").get<double>();
    ms_router_reserve_ar_queue(router, 0, 275.0);
    const auto mid_ar = take_json(ms_router_get_workers_json(router));
    MS_CHECK(std::fabs(mid_ar.at("0").at("queue_wait_ms").get<double>() - (q0_before + 275.0)) < 1e-6);
    ms_router_release_ar_queue(router, 0, 275.0);
    const auto after_ar = take_json(ms_router_get_workers_json(router));
    MS_CHECK(std::fabs(after_ar.at("0").at("queue_wait_ms").get<double>() - q0_before) < 1e-6);

    // 3. `ms_router_plan_transition_readonly_json` has ZERO worker or pacer side effects
    const nlohmann::json trans_req = {
        {"routing_mode", "dual_path"},
        {"est_dit_ms", 4200.0},
        {"now_ms", 100.0},
        {"req",
         {{"req_id", "r_ro"},
          {"src_gpu", 0},
          {"architecture", "tight_mot"},
          {"kv_bytes", 128LL * kMiB},
          {"gen_expert_bytes", 12446LL * kMiB},
          {"dit_step1_ms", 500.0},
          {"dit_total_steps", 20},
          {"resolution", 512}}},
    };
    const double pacer_wait_before = ms_pacer_predicted_wait_ms(pacer, 0, 100.0);
    const auto ro_plan = take_json(ms_router_plan_transition_readonly_json(router, trans_req.dump().c_str()));
    MS_CHECK(ro_plan.contains("decision") && ro_plan.contains("ship_mode"));
    const auto after_ro_ws = take_json(ms_router_get_workers_json(router));
    MS_CHECK(after_ro_ws == after_ar);
    MS_CHECK(std::fabs(ms_pacer_predicted_wait_ms(pacer, 0, 100.0) - pacer_wait_before) < 1e-9);

    // 4. `ms_ctrl_create`, `ms_ctrl_plan_turn_transition_readonly_json`, `ms_ctrl_plan_turn_transition_json`
    const auto pol_j = take_json(ms_policy_from_preset_json("morphserve", 8, nullptr));
    const std::string pol_s = pol_j.dump();
    ms_ctrl_t* ctrl = ms_ctrl_create(pol_s.c_str(), topo_s.c_str(), router);
    MS_CHECK(ctrl != nullptr);
    if (!ctrl) {
        ms_router_destroy(router);
        ms_pacer_destroy(pacer);
        return;
    }

    const nlohmann::json turn_req = {
        {"session_id", "sess_42"},
        {"base_ctx_id", "ctx_42"},
        {"turn_idx", 1},
        {"attempt", 0},
        {"src_gpu", 0},
        {"state_bytes", 128LL * kMiB},
        {"modality_plan", "dit_edit"},
        {"height", 512},
        {"dit_steps", 10},
        {"has_more_turns", true},
        {"has_more_edits", true},
        {"now_ms", 200.0},
    };
    const auto ctrl_ro = take_json(ms_ctrl_plan_turn_transition_readonly_json(ctrl, turn_req.dump().c_str()));
    // Custom operator profile: 10 * 500 + 300 + 100 = 5400 ms (proves `ms_router_create` parsed `operator_profile`!)
    MS_CHECK(std::fabs(ctrl_ro.at("est_dit_ms").get<double>() - 5400.0) < 1e-6);
    MS_CHECK(ctrl_ro.at("interleaved_roundtrip_gamma").get<int>() == 2);
    const auto ws_after_ctrl_ro = take_json(ms_router_get_workers_json(router));
    MS_CHECK(ws_after_ctrl_ro == after_ar);

    const auto ctrl_res = take_json(ms_ctrl_plan_turn_transition_json(ctrl, turn_req.dump().c_str()));
    MS_CHECK(std::fabs(ctrl_res.at("est_dit_ms").get<double>() - 5400.0) < 1e-6);
    const int dst_gpu = ctrl_res.at("dst_gpu").get<int>();
    const auto ws_after_ctrl_res = take_json(ms_router_get_workers_json(router));
    MS_CHECK(ws_after_ctrl_res.at(std::to_string(dst_gpu)).at("queue_wait_ms").get<double>() >= 5400.0 - 1e-6);

    // Null-handle guard on `ms_ctrl_*`
    MS_CHECK(ms_ctrl_plan_turn_transition_json(nullptr, "{}") == nullptr);
    MS_CHECK(ms_last_error_code() == MS_ERR_INVALID_ARG);
    MS_CHECK(ms_ctrl_plan_turn_transition_readonly_json(nullptr, "{}") == nullptr);
    MS_CHECK(ms_last_error_code() == MS_ERR_INVALID_ARG);

    // 5. `ms_ctrl_evaluate_session_slos_json`
    const nlohmann::json slo_ok_req = {
        {"ar_ttfts_ms", nlohmann::json::array({200.0})},
        {"ar_tpots_ms", nlohmann::json::array({55.0, 60.0, 58.0})},
        {"dit_turns_ms", nlohmann::json::array({6000.0})},
        {"dit_heights", nlohmann::json::array({512})},
        {"dit_steps", nlohmann::json::array({20})},
        {"slo_time_scale", 1.0},
    };
    const auto slo_ok = take_json(ms_ctrl_evaluate_session_slos_json(pol_s.c_str(), slo_ok_req.dump().c_str()));
    MS_CHECK(slo_ok.at("ar_slo_met").get<bool>() == true);
    MS_CHECK(slo_ok.at("dit_slo_met").get<bool>() == true);
    MS_CHECK(slo_ok.at("joint_slo_met").get<bool>() == true);

    const nlohmann::json slo_fail_req = {
        {"ar_ttfts_ms", nlohmann::json::array({200.0})},
        {"ar_tpots_ms", nlohmann::json::array({55.0, 60.0})},
        {"dit_turns_ms", nlohmann::json::array({999999.0})},
        {"dit_heights", nlohmann::json::array({512})},
        {"dit_steps", nlohmann::json::array({20})},
        {"slo_time_scale", 1.0},
    };
    const auto slo_fail = take_json(ms_ctrl_evaluate_session_slos_json(pol_s.c_str(), slo_fail_req.dump().c_str()));
    MS_CHECK(slo_fail.at("ar_slo_met").get<bool>() == true);
    MS_CHECK(slo_fail.at("dit_slo_met").get<bool>() == false);
    MS_CHECK(slo_fail.at("joint_slo_met").get<bool>() == false);

    // Modality-normalized AR TTFT through C API (`ar_turn_has_image` / `ar_turn_num_images` / `ar_turn_image_res`)
    const nlohmann::json slo_text_ttft_fail = {
        {"ar_ttfts_ms", nlohmann::json::array({320.0})},
        {"ar_tpots_ms", nlohmann::json::array({55.0})},
        {"ar_turn_has_image", nlohmann::json::array({false})},
        {"ar_turn_num_images", nlohmann::json::array({0})},
        {"ar_turn_image_res", nlohmann::json::array({512})},
    };
    const auto slo_mod_fail =
        take_json(ms_ctrl_evaluate_session_slos_json(pol_s.c_str(), slo_text_ttft_fail.dump().c_str()));
    MS_CHECK(slo_mod_fail.at("ar_slo_met").get<bool>() == false);

    // 6. `ms_ctrl_aggregate_report_json`
    const nlohmann::json outcomes = nlohmann::json::array({
        {{"status", "ok"},
         {"phase_label", "steady"},
         {"arrival_offset_s", 1.0},
         {"e2e_ms", 6500.0},
         {"active_jct_ms", 6500.0},
         {"planned_turns", 1},
         {"ar_ttfts_ms", nlohmann::json::array({150.0})},
         {"ar_tpots_ms", nlohmann::json::array({50.0, 52.0})},
         {"dit_turns_ms", nlohmann::json::array({6000.0})},
         {"dit_turn_heights", nlohmann::json::array({512})},
         {"dit_turn_steps", nlohmann::json::array({20})}},
        {{"status", "oom"},
         {"phase_label", "steady"},
         {"arrival_offset_s", 2.0},
         {"planned_turns", 1}},
    });
    const nlohmann::json wstats = nlohmann::json::array({
        {{"combined_physical_mib", 24000.0}, {"oom_count", 0}},
        {{"combined_physical_mib", 39000.0}, {"oom_count", 1}},
    });
    const auto agg = take_json(ms_ctrl_aggregate_report_json(
        pol_s.c_str(),
        "unit_suite",
        1.0,
        10.0,
        0.5,
        outcomes.dump().c_str(),
        wstats.dump().c_str()));
    MS_CHECK(agg.at("total_sessions").get<int>() == 2);
    MS_CHECK(agg.at("completed_sessions").get<int>() == 1);
    MS_CHECK(agg.at("oom_sessions").get<int>() == 1);
    MS_CHECK(agg.at("worker_oom_count").get<int>() == 1);
    MS_CHECK(std::fabs(agg.at("joint_slo_attainment_pct").get<double>() - 50.0) < 1e-6);

    ms_ctrl_destroy(ctrl);
    ms_router_destroy(router);
    ms_pacer_destroy(pacer);
}

void test_c_api_gpu_queue_and_atlas() {
    // Null guards
    MS_CHECK(ms_gpu_queue_select_json(nullptr, "{}") == nullptr);
    MS_CHECK(ms_last_error_code() == MS_ERR_INVALID_ARG);
    MS_CHECK(ms_gpu_queue_stats_json(nullptr) == nullptr);
    MS_CHECK(ms_last_error_code() == MS_ERR_INVALID_ARG);

    ms_gpu_queue_t* q = ms_gpu_queue_create(R"({"mode":"memory_slack","max_wait_aging_ms":5000.0,"max_bypasses":2})");
    MS_CHECK(q != nullptr);
    if (!q) return;

    const std::string req = R"({
        "now_ms": 200.0,
        "live_free_mem_bytes": 4294967296,
        "waiters": [
            {"waiter_id": 1, "enqueue_ms": 100.0, "est_mem_bytes": 8589934592, "est_exec_ms": 8000.0, "slo_deadline_ms": 50000.0, "op_priority": 1, "bypass_count": 0},
            {"waiter_id": 2, "enqueue_ms": 120.0, "est_mem_bytes": 1073741824, "est_exec_ms": 2000.0, "slo_deadline_ms": 6000.0, "op_priority": 1, "bypass_count": 0}
        ]
    })";
    const auto resp = take_json(ms_gpu_queue_select_json(q, req.c_str()));
    MS_CHECK(resp.at("selected").at("waiter_id").get<uint64_t>() == 2);
    MS_CHECK(resp.at("selected").at("memory_bypass").get<bool>() == true);
    MS_CHECK(resp.at("remaining_waiters").size() == 1);
    MS_CHECK(resp.at("remaining_waiters")[0].at("bypass_count").get<int>() == 1);
    const auto st = take_json(ms_gpu_queue_stats_json(q));
    MS_CHECK(st.at("memory_bypasses_total").get<int64_t>() == 1);
    ms_gpu_queue_destroy(q);

    // 3D Performance Atlas C ABI estimate across all 3 modes.
    const auto est_atlas = take_json(ms_atlas_estimate_json(
        R"({"cost_model_mode":"atlas_3d","model_family":"bagel","resolution":1024,"alpha_g":0.0,"is_post_evict":false,"colocated_ar_b":2})"));
    const auto est_legacy = take_json(ms_atlas_estimate_json(
        R"({"cost_model_mode":"legacy_analytical","model_family":"bagel","resolution":1024,"alpha_g":0.0,"is_post_evict":false,"colocated_ar_b":2})"));
    MS_CHECK(est_atlas.at("from_atlas").get<bool>() == true);
    MS_CHECK(std::fabs(est_atlas.at("exposed_stall_ms").get<double>() - 173.97) < 0.1);
    MS_CHECK(est_atlas.at("dit_step_stretch_factor").get<double>() > 1.2);
    MS_CHECK(est_legacy.at("exposed_stall_ms").get<double>() > 1000.0);
}

}  // namespace

int main() {
    test_vmm_null_guards_and_tripool_pluggable_safety();
    test_vmm_coalesce_oom_pinning_multi_exclude_and_headroom();
    test_policy_and_sim_override_forwarding();
    test_router_and_control_engine_c_api();
    test_c_api_gpu_queue_and_atlas();

    if (g_failures != 0) {
        std::cerr << "[FAIL] test_c_api (" << g_failures << " failures)\n";
        return 1;
    }
    std::cout << "[PASS] test_c_api\n";
    return 0;
}
