#include "morphserve/control/simulator.hpp"

#include "morphserve/control/gpu_queue_scheduler.hpp"
#include "morphserve/control/hash_util.hpp"
#include "morphserve/control/morph_controller.hpp"
#include "morphserve/control/performance_atlas.hpp"
#include "morphserve/control/preempt_edf.hpp"
#include "morphserve/control/router.hpp"
#include "morphserve/runtime/h2d_pacer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace morphserve::control {

namespace {

double crossover_kv_mib(
    const HardwareInterconnectSpec& hw,
    const ClusterPhysicsProfile& ph,
    const WorkloadScaleSpec& ws) {
    const auto topo = hw.to_topology(std::max(2, hw.gpus_per_node));
    DualPathMorphRouter router(topo);
    return router.analytical_crossover_kv_mib(
        0.50,
        static_cast<int64_t>(ph.packed_expert_data_mib * ws.weight_scale * static_cast<double>(kMiB)),
        (ph.dit_512_cfg_step_mean_ms * ws.compute_scale) / hw.compute_speedup_x,
        0,
        1);
}

enum EvKind : int {
    EV_RELEASE = 0,
    EV_DIT_END = 1,
    EV_RET_SHM_SEND = 2,
    EV_RET_UNPACK = 3,
    EV_DIT_START = 4,
    EV_SHM_SEND = 5,
    EV_DIT_READY = 6,
    EV_PREP_START = 7,
    EV_TURN = 8,
    EV_GPU_DISPATCH = 9,
};

struct EvTurn {
    int ti = 0;
    int attempt = 0;
    double ar_enq_ms = -1.0;
    bool waiting_ar_lock = false;
};

struct EvPrepStart {
    int ti = 0;
    int attempt = 0;
    int cur_gpu = 0;
    double t_turn0 = 0.0;
    double est_prep_q = 0.0;
};

struct EvDitReady {
    int ti = 0;
    int attempt = 0;
    int cur_gpu = 0;
    double t_turn0 = 0.0;
    std::string quad;
    std::string plan;
    int height = 512;
    int dit_steps = 20;
    int n_turns = 1;
    double cand_main_kv = 0.0;
    double dit_phys_kv = 0.0;
    double step_base = 0.0;
    double dit_core_ms = 0.0;
    double vae_dec_ms = 0.0;
    double deferred_dit_prep_ms = 0.0;
    bool deferred_full_turn = false;
};

struct EvShmSend {
    EvDitReady r;
    int dst_gpu = 0;
    double send_ms = 0.0;
    double recv_ms = 0.0;
    double dst_est_q = 0.0;
    int64_t dst_inflight_b = 0;
    bool release_src = true;
    bool ephemeral_gen_ctx = true;
    bool needs_return_ship = false;
};

struct EvDitStart {
    int ti = 0;
    int attempt = 0;
    int cur_gpu = 0;
    int dst_gpu = 0;
    std::string ship_mode;
    double t_turn0 = 0.0;
    int height = 512;
    int dit_steps = 20;
    int n_turns = 1;
    double cand_main_kv = 0.0;
    double dit_phys_kv = 0.0;
    double dit_core_ms = 0.0;
    double vae_dec_ms = 0.0;
    double pre_denoise_gpu_ms = 0.0;
    double reserved_est_ms = 0.0;
    int64_t inflight_kv_b = 0;
    double unpack_staging_mib = 0.0;
    bool run_pre_denoise_vae = false;
    bool ephemeral_gen_ctx = false;
    bool needs_return_ship = false;
};

struct EvDitEnd {
    int ti = 0;
    int attempt = 0;
    int cur_gpu = 0;
    int dst_gpu = 0;
    std::string ship_mode;
    double t_turn0 = 0.0;
    int height = 512;
    int dit_steps = 20;
    int n_turns = 1;
    double cand_main_kv = 0.0;
    double reserved_est_ms = 0.0;
    bool ephemeral_gen_ctx = false;
    bool needs_return_ship = false;
};

struct EvRetShmSend {
    EvDitEnd e;
    double send_ms = 0.0;
    double recv_ms = 0.0;
};

struct EvRetUnpack {
    int ti = 0;
    int attempt = 0;
    int cur_gpu = 0;
    int dst_gpu = 0;
    double t_turn0 = 0.0;
    int height = 512;
    int dit_steps = 20;
    int n_turns = 1;
    double cand_main_kv = 0.0;
    double recv_gpu_ms = 0.0;
};

struct EvGpuDispatch {
    int gpu_id = 0;
};

struct PendingGpuWork {
    uint64_t waiter_id = 0;
    int sid = 0;
    bool is_prep = false;
    EvPrepStart prep;
    EvDitStart dit;
};

using PayloadVar = std::variant<
    std::monostate,
    EvTurn,
    EvPrepStart,
    EvDitReady,
    EvShmSend,
    EvDitStart,
    EvDitEnd,
    EvRetShmSend,
    EvRetUnpack,
    EvGpuDispatch>;

struct SimEvent {
    double ts_ms = 0.0;
    double fifo_enq_ms = 0.0;
    uint64_t fifo_seq = 0;
    int kind = 0;
    int sid = 0;
    PayloadVar payload;

    int lock_prio_group() const noexcept {
        if (kind <= EV_DIT_END) return 0;
        if (kind == EV_DIT_READY || kind == EV_DIT_START || kind == EV_PREP_START) return 1;
        if (kind == EV_GPU_DISPATCH) return 2;
        if (kind == EV_TURN) {
            if (const auto* tv = std::get_if<EvTurn>(&payload)) {
                if (tv->waiting_ar_lock) return 2;
            }
            return 3;
        }
        return 2;
    }

    bool operator>(const SimEvent& o) const noexcept {
        // Non-lock completion events fire before lock acquisitions at `ts_ms`;
        // competing lock acquisitions wake in strict FIFO enqueue order (`fifo_enq_ms, fifo_seq`).
        const int prio_grp = lock_prio_group();
        const int o_grp = o.lock_prio_group();
        return std::tie(ts_ms, prio_grp, fifo_enq_ms, fifo_seq, sid) >
               std::tie(o.ts_ms, o_grp, o.fifo_enq_ms, o.fifo_seq, o.sid);
    }
};

struct SimDitRec {
    double lat_ms = 0.0;
    int height = 512;
    int steps = 20;
};

struct SimTurnRec {
    int turn_idx = 0;
    std::string modality_plan;
    std::string status = "ok";
    std::optional<double> ar_ttft_ms;
    std::vector<double> ar_tpots_ms;
    std::optional<double> dit_ms;
    int height = 0;
    int dit_steps = 0;
    double kv_mib = 0.0;
    bool has_image = false;
    int num_images = 0;
    int image_res = 512;
    bool use_prev_image = false;
};

struct PendingCotAr {
    bool valid = false;
    int turn_idx = -1;
    double ttft_ms = 0.0;
    std::vector<double> tpots_ms;
    bool has_image = false;
    int num_images = 0;
    int image_res = 512;
    bool use_prev_image = false;
};

}  // namespace

nlohmann::json ClusterSimulator::simulate_cluster_point(
    const std::string& policy_name,
    int num_gpus,
    const HardwareInterconnectSpec& hw,
    double offered_qps,
    int num_sessions,
    uint64_t seed,
    const ClusterPhysicsProfile& ph,
    const WorkloadScaleSpec& ws,
    const std::string& skew_tag,
    const std::string& bench_root,
    const std::string& sources_dir,
    const std::vector<SimSessionSpec>* custom_sessions,
    const nlohmann::json& policy_overrides) {
    if (!is_valid_policy_name(policy_name)) {
        throw std::invalid_argument("unknown policy: " + policy_name);
    }

    nlohmann::json overrides = nlohmann::json::object();
    if (ws.think_s_range_override.has_value()) {
        const double lo_ov = ws.think_s_range_override->first;
        const double hi_ov = ws.think_s_range_override->second;
        overrides["inter_turn_think_s_range"] = nlohmann::json::array({lo_ov, hi_ov});
        if (lo_ov > 0.0 && hi_ov > lo_ov &&
            !(policy_overrides.is_object() && policy_overrides.contains("think_time_median_s"))) {
            overrides["think_time_median_s"] = std::sqrt(lo_ov * hi_ov);
        }
    }
    if (hw.gpu_hbm_gib >= 75.0 && policy_name == "morphserve") {
        overrides["super_pool_mib"] = ws.super_pool_mib_80g;
        overrides["fine_kv_pool_mib"] = ws.fine_kv_pool_mib_80g;
    }
    if (policy_overrides.is_object()) {
        for (auto it = policy_overrides.begin(); it != policy_overrides.end(); ++it) {
            overrides[it.key()] = it.value();
        }
    }

    const PolicyConfig policy = PolicyConfig::from_preset(policy_name, num_gpus, overrides);
    const ClusterTopologyProfile topo = hw.to_topology(num_gpus);
    auto pacer = std::make_shared<runtime::HalfNodeH2DPacer>(topo.gpu_to_group_map(), 1);
    const double sp = hw.compute_speedup_x;
    const double cscale = ws.compute_scale;
    const double slo_time_scale = cscale / sp;

    const PerformanceAtlas gt_atlas = PerformanceAtlas::a100_default();
    auto router = std::make_shared<DualPathMorphRouter>(
        topo,
        pacer,
        default_operator_profile(),
        std::nullopt,
        (ph.vmm_full_expert_map_ms * ws.weight_scale) / sp,
        0.90,
        true,
        true,
        static_cast<int64_t>(128.0 * ws.kv_per_turn_scale * static_cast<double>(kMiB)),
        policy.cost_model_mode,
        gt_atlas);
    MorphControlEngine ctrl(policy, topo, router);

    const double vmm_ar_ovh = policy.use_tripool_vmm ? ph.vmm_ar_kv_step_overhead_ms : 0.0;
    const double ar_tok_p50_ms = ((ph.ar_unloaded_tpot_p50_ms + vmm_ar_ovh) * cscale) / sp;
    const double ar_tok_p99_ms = ((ph.ar_unloaded_tpot_p99_ms + vmm_ar_ovh) * cscale) / sp;
    const double ar_batch2_step_ms = ((ph.ar_batched2_step_p50_ms + vmm_ar_ovh) * cscale) / sp;
    const double ar_prefill_ms = (ph.ar_prefill_ms * cscale) / sp;
    const double vae512_ms = (ph.vae_512_ms * ws.vae_spike_scale) / sp;
    const double vae1024_ms = (ph.vae_1024_ms * ws.vae_spike_scale) / sp;
    const double vit_only_img_ms =
        (std::max(100.0, ph.stock_loopback_turn_ms - ph.vae_512_ms - ph.ar_prefill_ms) * cscale) / sp;
    const double vae_encode_512_ms = (ph.vae_512_ms * cscale) / sp;
    const double vae_encode_1024_ms = (ph.vae_1024_ms * cscale) / sp;
    const double stock_img_loop_512_ms = (ph.stock_loopback_turn_ms * cscale) / sp;
    const double ms_img_loop_512_ms = (ph.morphserve_loopback_turn_ms * cscale) / sp;
    const double dit512_base_core_20_ms = (std::max(0.0, ph.dit_512_20step_base_ms - ph.vae_512_ms) * cscale) / sp;
    const double dit512_step_ms = (ph.dit_512_cfg_step_mean_ms * cscale) / sp;
    const double dit1024_step_ms = (ph.dit_1024_step_ms * cscale) / sp;
    const double dit1024_base_core_20_ms = std::max(0.0, 20.0 * dit1024_step_ms - vae1024_ms);

    const double host_hook_scale = (num_gpus >= 2) ? (1.0 / sp) : 0.0;
    const double pager_hook_per_layer_ms = policy.use_expert_pager ? (0.032 * host_hook_scale) : 0.0;
    const double sched_poll_per_layer_ms = (policy.preemption_mode == "layer_chunk") ? (0.018 * host_hook_scale) : 0.0;

    const double pure_stock_kv_slope = ph.stock_dit_512_ms_per_kv_tok;
    const double pure_ms_kv_slope = ph.morphserve_dit_512_ms_per_kv_tok;

    const double layer512_ms = (ph.dit_512_layer_p50_ms * cscale) / sp;
    const double layer1024_ms = layer512_ms * (ph.dit_1024_step_ms / std::max(1.0, ph.dit_512_cfg_step_mean_ms));
    const double ar_tpot_slo = policy.ar_tpot_slo_ms * slo_time_scale;

    // Shared `PreemptEdfController` (`guard_ms = 24.0 * slo_time_scale`, aligned with live `bagel_adapter.py`).
    const PreemptEdfController edf_512 = MorphControlEngine::make_edf_controller(
        ph.num_expert_layers, ar_tpot_slo, layer512_ms, ar_tok_p50_ms, slo_time_scale);
    const PreemptEdfController edf_1024 = MorphControlEngine::make_edf_controller(
        ph.num_expert_layers, ar_tpot_slo, layer1024_ms, ar_tok_p50_ms, slo_time_scale);
    const int chunk_layers_512 = edf_512.compute_adaptive_chunk_layers();
    const int chunk_layers_1024 = edf_1024.compute_adaptive_chunk_layers();
    const double chunk_q_512_ms = static_cast<double>(chunk_layers_512) * layer512_ms;
    const double chunk_q_1024_ms = static_cast<double>(chunk_layers_1024) * layer1024_ms;

    constexpr double kEdit1024KvRatio = (4096.0 + 4900.0) / (1024.0 + 4900.0);
    constexpr double kPageableOffloadBwRatio = 0.66;

    const int64_t expert_bytes_scaled =
        static_cast<int64_t>(ph.packed_expert_data_mib * ws.weight_scale * static_cast<double>(kMiB));
    const int layer_stride_scaled = static_cast<int>(std::round(static_cast<double>(ph.layer_stride_mib) * ws.weight_scale));
    const bool is_real_40g_preset =
        (std::abs(hw.gpu_hbm_gib - 40.0) < 1e-3) &&
        (hw.name.rfind("A100_", 0) == 0 || hw.name.rfind("Kairos_", 0) == 0);
    const double hbm_mib = is_real_40g_preset ? ph.driver_total_hbm_40g_mib : (hw.gpu_hbm_gib * 1024.0);

    std::vector<SimGpuMemoryLedger> ledgers;
    ledgers.reserve(static_cast<size_t>(num_gpus));
    std::map<int, int64_t> init_kv_map;
    for (int g = 0; g < num_gpus; ++g) {
        auto ait = policy.initial_alphas.find(g);
        const double a0 = (ait != policy.initial_alphas.end()) ? ait->second : 1.0;
        ledgers.emplace_back(
            g,
            hbm_mib,
            policy,
            policy.role_of(g),
            static_cast<int>(std::round(a0 * static_cast<double>(ph.num_expert_layers))),
            ph.num_expert_layers,
            layer_stride_scaled,
            ph,
            ws.weight_scale,
            ws.vae_spike_scale);
        init_kv_map[g] = ledgers.back().free_kv_bytes();
    }

    std::map<int, WorkerState> workers = router->init_workers(
        num_gpus,
        [&](int g) { return policy.role_of(g); },
        policy.initial_alphas,
        init_kv_map);

    std::vector<double> gpu_free_at(static_cast<size_t>(num_gpus), 0.0);
    std::vector<double> ar_slot_free_at(static_cast<size_t>(num_gpus), 0.0);
    std::vector<double> dit_denoise_start(static_cast<size_t>(num_gpus), -1.0);
    std::vector<double> dit_denoise_cfg_start(static_cast<size_t>(num_gpus), -1.0);
    std::vector<double> dit_denoise_end(static_cast<size_t>(num_gpus), -1.0);
    std::vector<double> dit_full_end(static_cast<size_t>(num_gpus), -1.0);
    std::vector<double> dit_active_uncond_step_ms(static_cast<size_t>(num_gpus), dit512_step_ms);
    std::vector<double> dit_active_cfg_step_ms(static_cast<size_t>(num_gpus), dit512_step_ms);
    std::vector<int> dit_active_sid(static_cast<size_t>(num_gpus), -1);
    std::vector<bool> dit_active_is1024(static_cast<size_t>(num_gpus), false);
    std::vector<std::array<double, 2>> preempt_slot_end(
        static_cast<size_t>(num_gpus), std::array<double, 2>{0.0, 0.0});
    std::vector<double> queued_work_ms(static_cast<size_t>(num_gpus), 0.0);
    std::vector<int64_t> inflight_ship_kv_bytes(static_cast<size_t>(num_gpus), 0);
    std::vector<int> gpu_oom_counts(static_cast<size_t>(num_gpus), 0);

    // Per-GPU Memory+Slack non-FCFS wait-queue schedulers (shared C++17 logic with live coordinator).
    const GpuQueuePolicy gpu_q_policy{
        policy.gpu_queue_mode, policy.gpu_queue_max_aging_ms, policy.gpu_queue_max_bypasses};
    std::vector<GpuWaitQueueScheduler> gpu_schedulers;
    gpu_schedulers.reserve(static_cast<size_t>(num_gpus));
    for (int g = 0; g < num_gpus; ++g) {
        gpu_schedulers.emplace_back(gpu_q_policy);
    }
    std::vector<std::vector<GpuQueueWaiter>> gpu_wait_specs(static_cast<size_t>(num_gpus));
    std::vector<std::vector<PendingGpuWork>> gpu_wait_payloads(static_cast<size_t>(num_gpus));
    std::vector<double> gpu_dispatch_armed_at(static_cast<size_t>(num_gpus), -1.0);
    uint64_t next_waiter_id = 0;

    // Think-Time KV Swap vs. Recomputation telemetry accumulators.
    std::vector<double> sim_swap_in_latencies_ms;
    int sim_kv_recompute_events = 0;
    double sim_kv_recompute_total_ms = 0.0;
    double sim_kv_saved_recompute_ms = 0.0;

    const int gpn = std::max(1, hw.gpus_per_node);
    const int num_nodes = std::max(1, (num_gpus + gpn - 1) / gpn);
    std::vector<double> rdma_free_at(static_cast<size_t>(num_nodes), 0.0);
    std::vector<double> unpaced_h2d_until(static_cast<size_t>(num_gpus), 0.0);

    const std::vector<SimSessionSpec> built_sessions =
        custom_sessions
            ? *custom_sessions
            : build_session_stream(
                  num_sessions, offered_qps, num_gpus, seed, ws.extra_edit_turns, skew_tag, bench_root, sources_dir);
    const size_t n_sess = built_sessions.size();

    std::vector<int> sess_gpu(n_sess, -1);
    std::vector<double> sess_kv_mib(n_sess, 0.0);
    std::vector<double> sess_main_kv_mib(n_sess, 0.0);
    std::vector<double> sess_dit_actual_end(n_sess, 0.0);
    std::vector<double> sess_think_ms(n_sess, 0.0);
    std::vector<double> sess_end_ms(n_sess, 0.0);
    std::vector<bool> sess_oom(n_sess, false);
    std::vector<int> sess_retries(n_sess, 0);
    std::vector<int> sess_ships(n_sess, 0);
    std::vector<int> sess_morphs(n_sess, 0);
    std::vector<int> sess_xnode_ships(n_sess, 0);
    std::vector<int> sess_preemptions(n_sess, 0);
    std::vector<std::vector<double>> sess_ar_ttfts(n_sess);
    std::vector<std::vector<double>> sess_ar_tpots(n_sess);
    std::vector<std::vector<std::vector<double>>> sess_ar_turn_tpots(n_sess);
    std::vector<std::vector<int>> sess_ar_has_image(n_sess);
    std::vector<std::vector<int>> sess_ar_num_images(n_sess);
    std::vector<std::vector<int>> sess_ar_image_res(n_sess);
    std::vector<std::vector<SimDitRec>> sess_dit_records(n_sess);
    std::vector<std::vector<SimTurnRec>> sess_turn_records(n_sess);
    std::vector<PendingCotAr> sess_pending_cot_ar(n_sess);

    std::priority_queue<SimEvent, std::vector<SimEvent>, std::greater<SimEvent>> pq;
    uint64_t ev_seq = 0;
    auto push_ev = [&](double ts_ms, int kind, int sid, PayloadVar p, double enq_ms = -1.0, uint64_t enq_seq = 0) {
        if (enq_seq == 0) {
            ++ev_seq;
            enq_seq = ev_seq;
        }
        const double eff_enq = (enq_ms >= 0.0) ? enq_ms : ts_ms;
        pq.push(SimEvent{ts_ms, eff_enq, enq_seq, kind, sid, std::move(p)});
    };

    for (size_t sid = 0; sid < n_sess; ++sid) {
        push_ev(built_sessions[sid].arrival_ms, EV_TURN, static_cast<int>(sid), EvTurn{0, 0});
    }

    auto compute_think_ms = [&](size_t usid_idx, int next_ti) -> double {
        const auto& s_id = built_sessions[usid_idx].session_id;
        if (!s_id.empty()) {
            return MorphControlEngine::deterministic_think_dist_s(
                       s_id,
                       next_ti,
                       policy.inter_turn_think_s_range.first,
                       policy.inter_turn_think_s_range.second,
                       policy.think_time_dist,
                       policy.think_time_median_s,
                       policy.think_time_sigma) *
                   1000.0;
        }
        return MorphControlEngine::deterministic_think_dist_ms_sim(
            static_cast<int>(usid_idx),
            next_ti,
            policy.inter_turn_think_s_range.first,
            policy.inter_turn_think_s_range.second,
            policy.think_time_dist,
            policy.think_time_median_s,
            policy.think_time_sigma);
    };

    auto record_oom = [&](size_t usid_idx, int turn_idx, double end_t_ms) {
        sess_oom[usid_idx] = true;
        sess_end_ms[usid_idx] = std::max(sess_end_ms[usid_idx], end_t_ms);
        SimTurnRec rec;
        rec.turn_idx = turn_idx;
        if (turn_idx >= 0 && static_cast<size_t>(turn_idx) < built_sessions[usid_idx].turns.size()) {
            const auto& ts = built_sessions[usid_idx].turns[static_cast<size_t>(turn_idx)];
            rec.modality_plan = ts.modality_plan;
            rec.height = ts.height;
            rec.dit_steps = ts.dit_steps;
        }
        rec.status = "oom";
        rec.kv_mib = sess_main_kv_mib[usid_idx];
        if (sess_pending_cot_ar[usid_idx].valid && sess_pending_cot_ar[usid_idx].turn_idx == turn_idx) {
            rec.ar_ttft_ms = sess_pending_cot_ar[usid_idx].ttft_ms;
            rec.ar_tpots_ms = std::move(sess_pending_cot_ar[usid_idx].tpots_ms);
            rec.has_image = sess_pending_cot_ar[usid_idx].has_image;
            rec.num_images = sess_pending_cot_ar[usid_idx].num_images;
            rec.image_res = sess_pending_cot_ar[usid_idx].image_res;
            rec.use_prev_image = sess_pending_cot_ar[usid_idx].use_prev_image;
            sess_pending_cot_ar[usid_idx].valid = false;
        }
        sess_turn_records[usid_idx].push_back(std::move(rec));
    };

    auto sync_states = [&](double now_ms,
                           std::map<int, bool>& busy_map,
                           std::map<int, int>& dit_map,
                           std::map<int, int>& pre_map) {
        busy_map.clear();
        dit_map.clear();
        pre_map.clear();
        for (int g = 0; g < num_gpus; ++g) {
            const size_t ug = static_cast<size_t>(g);
            const double rem_excl = std::max(0.0, gpu_free_at[ug] - now_ms);
            const double rem_ar = std::max(0.0, ar_slot_free_at[ug] - now_ms);
            auto& wg = workers.at(g);
            wg.queue_wait_ms = queued_work_ms[ug] + 0.5 * rem_ar;
            wg.expert_alpha = ledgers[ug].expert_alpha();
            wg.post_evict = ledgers[ug].is_post_evict();
            wg.kv_free_bytes = std::max<int64_t>(0, ledgers[ug].free_kv_bytes() - inflight_ship_kv_bytes[ug]);
            busy_map[g] = (rem_excl > 0.0 || rem_ar > 0.0 || queued_work_ms[ug] > 0.0);
            dit_map[g] = (now_ms >= dit_denoise_start[ug] && now_ms < dit_full_end[ug]) ? 1 : 0;
            int p_cnt = 0;
            for (double pe : preempt_slot_end[ug]) {
                if (pe > now_ms) ++p_cnt;
            }
            pre_map[g] = p_cnt;
            wg.colocated_ar_b = p_cnt;
        }
    };

    auto pick_oom_retry_gpu = [&](int failed_gpu, double now_ms) -> int {
        std::map<int, bool> busy_now;
        std::map<int, int> dit_now;
        std::map<int, int> pre_now;
        sync_states(now_ms, busy_now, dit_now, pre_now);
        return router->select_oom_retry_gpu(
            failed_gpu, num_gpus, policy.routing_mode, policy.static_und_ranks, &workers, busy_now);
    };

    // Shared Turn-0 OOM retry / terminal OOM handler replacing duplicated OOM-retry blocks.
    auto handle_oom_or_retry = [&](size_t usid_idx,
                                   int turn_idx,
                                   int attempt_idx,
                                   int failed_gpu,
                                   int secondary_release_gpu,
                                   double oom_time_ms,
                                   bool push_retry_event) -> bool {
        const int sid_i = static_cast<int>(usid_idx);
        if (failed_gpu >= 0 && failed_gpu < num_gpus) {
            ledgers[static_cast<size_t>(failed_gpu)].release_session(sid_i);
            ++gpu_oom_counts[static_cast<size_t>(failed_gpu)];
        }
        if (secondary_release_gpu >= 0 && secondary_release_gpu < num_gpus && secondary_release_gpu != failed_gpu) {
            ledgers[static_cast<size_t>(secondary_release_gpu)].release_session(sid_i);
        }
        if (attempt_idx < policy.max_oom_retries && turn_idx == 0) {
            ++sess_retries[usid_idx];
            sess_gpu[usid_idx] = pick_oom_retry_gpu(failed_gpu, oom_time_ms);
            if (push_retry_event) {
                push_ev(oom_time_ms, EV_TURN, sid_i, EvTurn{0, attempt_idx + 1});
            }
            return true;
        }
        record_oom(usid_idx, turn_idx, oom_time_ms);
        return false;
    };

    auto complete_dit_turn = [&](size_t usid_idx,
                                 int turn_idx,
                                 int n_turns,
                                 int next_gpu,
                                 int height,
                                 int dit_steps,
                                 double cand_main_kv,
                                 double t_turn0,
                                 double turn_end_ms,
                                 bool ephemeral_gen_ctx) {
        const int sid_i = static_cast<int>(usid_idx);
        sess_kv_mib[usid_idx] = cand_main_kv;
        sess_main_kv_mib[usid_idx] = cand_main_kv;
        const double turn_lat = turn_end_ms - t_turn0;
        sess_dit_records[usid_idx].push_back(SimDitRec{turn_lat, height, dit_steps});

        SimTurnRec rec;
        rec.turn_idx = turn_idx;
        if (turn_idx >= 0 && static_cast<size_t>(turn_idx) < built_sessions[usid_idx].turns.size()) {
            rec.modality_plan = built_sessions[usid_idx].turns[static_cast<size_t>(turn_idx)].modality_plan;
        }
        rec.status = "ok";
        rec.dit_ms = turn_lat;
        rec.height = height;
        rec.dit_steps = dit_steps;
        rec.kv_mib = cand_main_kv;
        if (sess_pending_cot_ar[usid_idx].valid && sess_pending_cot_ar[usid_idx].turn_idx == turn_idx) {
            rec.ar_ttft_ms = sess_pending_cot_ar[usid_idx].ttft_ms;
            rec.ar_tpots_ms = std::move(sess_pending_cot_ar[usid_idx].tpots_ms);
            rec.has_image = sess_pending_cot_ar[usid_idx].has_image;
            rec.num_images = sess_pending_cot_ar[usid_idx].num_images;
            rec.image_res = sess_pending_cot_ar[usid_idx].image_res;
            rec.use_prev_image = sess_pending_cot_ar[usid_idx].use_prev_image;
            sess_pending_cot_ar[usid_idx].valid = false;
        }
        sess_turn_records[usid_idx].push_back(std::move(rec));

        if ((turn_idx + 1) < n_turns) {
            sess_gpu[usid_idx] = next_gpu;
            const double think_ms = compute_think_ms(usid_idx, turn_idx + 1);
            sess_think_ms[usid_idx] += think_ms;
            push_ev(turn_end_ms + think_ms, EV_TURN, sid_i, EvTurn{turn_idx + 1, 0});
        } else {
            sess_end_ms[usid_idx] = turn_end_ms;
            if (!ephemeral_gen_ctx) {
                push_ev(turn_end_ms, EV_RELEASE, sid_i, std::monostate{});
            }
        }
    };

    auto compute_naive_swap_ms = [&](int gpu_id, double at_ms) -> double {
        const auto& grp = topo.half_group_of(gpu_id);
        int active_same_half = 1;
        for (int peer : grp.gpu_ids) {
            if (peer != gpu_id && unpaced_h2d_until[static_cast<size_t>(peer)] > at_ms) {
                ++active_same_half;
            }
        }
        const double eff_h2d = std::max(0.5, topo.effective_h2d_gbps(gpu_id, active_same_half) * kPageableOffloadBwRatio);
        const double swap_ms = ((static_cast<double>(expert_bytes_scaled) / 1e9) / eff_h2d) * 1000.0;
        if (policy.use_h2d_pacer) {
            const double p_wait = pacer->reserve_slot(gpu_id, swap_ms, at_ms);
            return p_wait + swap_ms;
        }
        unpaced_h2d_until[static_cast<size_t>(gpu_id)] =
            std::max(unpaced_h2d_until[static_cast<size_t>(gpu_id)], at_ms) + swap_ms;
        return swap_ms;
    };

    auto compute_dit_core_ms = [&](int gpu_hint,
                                   bool is_1024,
                                   bool has_3branch_cfg,
                                   bool has_shared_prefix_cfg,
                                   int main_seq_tokens,
                                   int dit_steps) -> double {
        const double steps_ratio = static_cast<double>(std::max(1, dit_steps)) / 20.0;
        const double slope_512 =
            (has_shared_prefix_cfg && policy.allow_cascade_cfg_split) ? pure_ms_kv_slope : pure_stock_kv_slope;
        const double q_scale = is_1024 ? (4098.0 / 1026.0) : 1.0;
        const double kv_tokens = std::max(0.0, static_cast<double>(main_seq_tokens) - 18.0);
        const double kv_attn_ms = ((slope_512 * q_scale * kv_tokens * steps_ratio) * cscale) / sp;
        const double base_20_ms = is_1024 ? dit1024_base_core_20_ms : dit512_base_core_20_ms;
        const int n_branches = has_3branch_cfg ? 3 : 2;
        const double layer_calls = static_cast<double>(ph.num_expert_layers * n_branches * std::max(1, dit_steps));
        const auto& lg = ledgers[static_cast<size_t>(std::clamp(gpu_hint, 0, num_gpus - 1))];
        const bool pager_paging_active =
            policy.use_expert_pager &&
            (lg.resident_expert_layers() < lg.num_expert_layers() || lg.expert_evictions() > 0 ||
             static_cast<double>(policy.super_pool_mib) <
                 (static_cast<double>(ph.num_expert_layers * layer_stride_scaled) +
                  ph.scoped_vae_1024_spike_mib * ws.vae_spike_scale));
        const double eff_per_layer_ovh =
            (pager_paging_active ? pager_hook_per_layer_ms : 0.0) + sched_poll_per_layer_ms;
        const double hook_ovh_ms = layer_calls * eff_per_layer_ovh;
        return (base_20_ms * steps_ratio) + kv_attn_ms + hook_ovh_ms;
    };

    auto excl_gpu_free_at = [&](size_t ug) -> double {
        return std::max(gpu_free_at[ug], ar_slot_free_at[ug]);
    };

    auto arm_gpu_dispatch = [&](int gpu_id, double wake_ms) {
        const size_t ug = static_cast<size_t>(gpu_id);
        if (gpu_wait_specs[ug].empty()) {
            return;
        }
        const double eff_wake = std::max(wake_ms, excl_gpu_free_at(ug));
        if (gpu_dispatch_armed_at[ug] >= 0.0 && gpu_dispatch_armed_at[ug] <= eff_wake + 1e-6) {
            return;
        }
        gpu_dispatch_armed_at[ug] = eff_wake;
        push_ev(eff_wake, EV_GPU_DISPATCH, -1, EvGpuDispatch{gpu_id}, eff_wake);
    };

    auto enqueue_gpu_work = [&](int gpu_id, double ready_ms, GpuQueueWaiter waiter, PendingGpuWork work) {
        const size_t ug = static_cast<size_t>(gpu_id);
        const uint64_t wid = ++next_waiter_id;
        waiter.waiter_id = wid;
        waiter.enqueue_ms = ready_ms;
        work.waiter_id = wid;
        gpu_wait_specs[ug].push_back(waiter);
        gpu_wait_payloads[ug].push_back(std::move(work));
        arm_gpu_dispatch(gpu_id, ready_ms);
    };

    auto estimate_history_recompute_ms = [&](double kv_mib, bool is_1024) -> double {
        if (kv_mib <= 0.0) return 0.0;
        const double unit_img_kv = std::max(16.0, 268.0 * ws.kv_per_turn_scale);
        const double n_hist_imgs = std::floor(kv_mib / unit_img_kv);
        const double text_kv_mib = std::max(0.0, kv_mib - n_hist_imgs * unit_img_kv);
        const double img_recomp_ms =
            n_hist_imgs * (vit_only_img_ms + (is_1024 ? vae_encode_1024_ms : vae_encode_512_ms));
        const double text_recomp_ms = ar_prefill_ms * std::max(1.0, text_kv_mib / std::max(1.0, 8.0 * ws.kv_per_turn_scale));
        return img_recomp_ms + text_recomp_ms;
    };

    auto restore_kv_resident = [&](int sid_i, int gpu_id, bool is_1024) -> std::pair<bool, double> {
        const size_t ug = static_cast<size_t>(gpu_id);
        const SessionRestoreResult r = ledgers[ug].restore_session_resident(sid_i);
        if (!r.ok) {
            return {false, 0.0};
        }
        if (r.swapped_in_mib > 0.0) {
            const double swap_ms =
                ((r.swapped_in_mib * static_cast<double>(kMiB) / 1e9) / std::max(0.5, topo.solo_h2d_gbps(gpu_id))) *
                1000.0;
            sim_swap_in_latencies_ms.push_back(swap_ms);
            sim_kv_saved_recompute_ms += estimate_history_recompute_ms(r.swapped_in_mib, is_1024);
            return {true, swap_ms};
        }
        if (r.recompute_kv_mib > 0.0) {
            const double recomp_ms = estimate_history_recompute_ms(r.recompute_kv_mib, is_1024);
            ++sim_kv_recompute_events;
            sim_kv_recompute_total_ms += recomp_ms;
            return {true, recomp_ms};
        }
        return {true, 0.0};
    };

    auto run_ar_phase_sim = [&](size_t usid_idx,
                                int cur_gpu,
                                double ev_ms,
                                double swap_in_ms,
                                double lock_wait_ms,
                                const SimTurnSpec& turn,
                                bool prepare_for_dit,
                                bool loopback_prev_dit_img = false,
                                double* out_ttft_ms = nullptr,
                                std::vector<double>* out_turn_tpots = nullptr) -> double {
        const size_t ug = static_cast<size_t>(cur_gpu);
        const int max_t = std::max(2, turn.max_ar_tokens);
        const int n_tok = prepare_for_dit ? std::max(12, static_cast<int>(std::round(0.75 * max_t))) : max_t;
        const bool is_fission_und =
            ((policy.routing_mode == "static_nccl_fission" || policy.routing_mode == "twostage_shm") &&
             policy.role_of(cur_gpu) == "und");

        double prefill_ms = ar_prefill_ms;
        if (turn.num_images > 0) {
            if (prepare_for_dit && !is_fission_und) {
                const double base_img_ms = policy.use_module3_cow_loopback ? ms_img_loop_512_ms : stock_img_loop_512_ms;
                prefill_ms = static_cast<double>(turn.num_images) * base_img_ms;
            } else {
                prefill_ms = static_cast<double>(turn.num_images) * vit_only_img_ms + ar_prefill_ms;
            }
        } else if (!prepare_for_dit && loopback_prev_dit_img) {
            const double loop_prep_saving =
                policy.use_module3_cow_loopback
                    ? std::max(0.0, ((ph.cpu_pil_prep_only_ms - ph.gpu_inplace_prep_only_ms) * cscale) / sp)
                    : 0.0;
            const double eff_vit_ms = std::max(50.0, vit_only_img_ms - loop_prep_saving);
            prefill_ms = eff_vit_ms + ar_prefill_ms;
        }

        int free_pre_slot = -1;
        for (int sl = 0; sl < 2; ++sl) {
            if (preempt_slot_end[ug][static_cast<size_t>(sl)] <= ev_ms) {
                free_pre_slot = sl;
                break;
            }
        }
        const bool dit_session_live =
            (dit_full_end[ug] > ev_ms) && (ev_ms < dit_denoise_end[ug]) &&
            (ev_ms >= dit_denoise_start[ug] - (900.0 * cscale / sp));
        const bool can_preempt_mid_dit =
            (policy.preemption_mode != "non_preemptive") &&
            dit_session_live &&
            (free_pre_slot >= 0);

        auto step_quantum_at = [&](double t_ms) -> double {
            if (policy.preemption_mode == "layer_chunk") {
                return dit_active_is1024[ug] ? chunk_q_1024_ms : chunk_q_512_ms;
            }
            if (!dit_active_is1024[ug]) {
                return dit_active_cfg_step_ms[ug];
            }
            return (t_ms < dit_denoise_cfg_start[ug]) ? dit_active_uncond_step_ms[ug] : dit_active_cfg_step_ms[ug];
        };

        auto stretch_active_dit = [&](double delta_ms, bool stretch_denoise, double at_ms) {
            if (delta_ms <= 0.0) return;
            if (stretch_denoise && dit_denoise_cfg_start[ug] > at_ms) {
                dit_denoise_cfg_start[ug] += delta_ms;
            }
            if (stretch_denoise && dit_denoise_end[ug] > ev_ms) {
                dit_denoise_end[ug] += delta_ms;
            }
            if (dit_full_end[ug] > ev_ms) {
                dit_full_end[ug] += delta_ms;
            }
            gpu_free_at[ug] += delta_ms;
            const int dsid = dit_active_sid[ug];
            if (dsid >= 0 && static_cast<size_t>(dsid) < n_sess) {
                sess_dit_actual_end[static_cast<size_t>(dsid)] =
                    std::max(sess_dit_actual_end[static_cast<size_t>(dsid)], dit_full_end[ug]);
            }
        };

        if (can_preempt_mid_dit) {
            const double pre_denoise_rem = std::max(0.0, dit_denoise_start[ug] - ev_ms);
            const double other_pre_rem =
                std::max(0.0, preempt_slot_end[ug][static_cast<size_t>(1 - free_pre_slot)] - (ev_ms + pre_denoise_rem));
            const double ready_in_dit = ev_ms + pre_denoise_rem + other_pre_rem;
            const double init_q_ms = step_quantum_at(ready_in_dit);
            const double phase_wait =
                pre_denoise_rem + other_pre_rem +
                std::min(std::max(0.0, dit_denoise_end[ug] - ready_in_dit), 0.5 * init_q_ms);
            const double wait_ms = phase_wait + swap_in_ms;

            const double start_prefill_ms = ev_ms + wait_ms;
            const bool still_mid_denoise_at_start = (start_prefill_ms < dit_denoise_end[ug]);
            stretch_active_dit(prefill_ms + ar_tok_p50_ms, still_mid_denoise_at_start, start_prefill_ms);
            const double ttft = lock_wait_ms + wait_ms + prefill_ms + ar_tok_p50_ms;
            sess_ar_ttfts[usid_idx].push_back(ttft);

            std::vector<double> turn_tpots;
            turn_tpots.reserve(static_cast<size_t>(std::max(0, n_tok - 1)));
            double t_cur = start_prefill_ms + prefill_ms + ar_tok_p50_ms;
            for (int tok_i = 1; tok_i < n_tok; ++tok_i) {
                const double frac = static_cast<double>(tok_i) / static_cast<double>(std::max(1, n_tok - 1));
                const double step_c = ar_tok_p50_ms + (ar_tok_p99_ms - ar_tok_p50_ms) * frac;
                const bool co_preempted = (t_cur < preempt_slot_end[ug][static_cast<size_t>(1 - free_pre_slot)]);
                const double eff_step_c = co_preempted ? ar_batch2_step_ms : step_c;
                const double dit_stretch_c = co_preempted ? (0.5 * ar_batch2_step_ms) : step_c;
                double tok_tpot = eff_step_c;
                if (t_cur < dit_denoise_end[ug]) {
                    const double q_now_ms = step_quantum_at(t_cur);
                    const double rem_denoise_ms = std::max(0.0, dit_denoise_end[ug] - t_cur);
                    tok_tpot = std::min(q_now_ms, rem_denoise_ms) + eff_step_c;
                    stretch_active_dit(dit_stretch_c, true, t_cur + std::min(q_now_ms, rem_denoise_ms));
                } else if (t_cur < dit_full_end[ug]) {
                    stretch_active_dit(dit_stretch_c, false, t_cur);
                }
                turn_tpots.push_back(tok_tpot);
                sess_ar_tpots[usid_idx].push_back(tok_tpot);
                t_cur += tok_tpot;
            }
            sess_ar_turn_tpots[usid_idx].push_back(turn_tpots);
            if (out_ttft_ms != nullptr) *out_ttft_ms = ttft;
            if (out_turn_tpots != nullptr) *out_turn_tpots = std::move(turn_tpots);
            preempt_slot_end[ug][static_cast<size_t>(free_pre_slot)] = t_cur;
            if (still_mid_denoise_at_start) {
                ++sess_preemptions[usid_idx];
            }
            return t_cur;
        }

        const double crit_rem_ms = std::max(0.0, gpu_free_at[ug] - ev_ms);
        const double excl_wait = lock_wait_ms + crit_rem_ms;
        const double start_ar = ev_ms + crit_rem_ms;
        const bool overlaps_ar = (ar_slot_free_at[ug] > start_ar + 1.0);
        const double prefill_wait = overlaps_ar ? std::min(ar_prefill_ms, ar_slot_free_at[ug] - start_ar) : 0.0;
        const double ttft = excl_wait + prefill_wait + swap_in_ms + prefill_ms + ar_tok_p50_ms;
        sess_ar_ttfts[usid_idx].push_back(ttft);

        std::vector<double> turn_tpots;
        turn_tpots.reserve(static_cast<size_t>(std::max(0, n_tok - 1)));
        double t_cur = start_ar + prefill_wait + swap_in_ms + prefill_ms + ar_tok_p50_ms;
        double gpu_added_ms = swap_in_ms + prefill_ms + ar_tok_p50_ms;
        for (int tok_i = 1; tok_i < n_tok; ++tok_i) {
            const double frac = static_cast<double>(tok_i) / static_cast<double>(std::max(1, n_tok - 1));
            const double solo_step = ar_tok_p50_ms + (ar_tok_p99_ms - ar_tok_p50_ms) * frac;
            const bool batched_now = (t_cur < ar_slot_free_at[ug]);
            const double tok_tpot = batched_now ? ar_batch2_step_ms : solo_step;
            gpu_added_ms += batched_now ? (0.5 * ar_batch2_step_ms) : solo_step;
            turn_tpots.push_back(tok_tpot);
            sess_ar_tpots[usid_idx].push_back(tok_tpot);
            t_cur += tok_tpot;
        }
        sess_ar_turn_tpots[usid_idx].push_back(turn_tpots);
        if (out_ttft_ms != nullptr) *out_ttft_ms = ttft;
        if (out_turn_tpots != nullptr) *out_turn_tpots = std::move(turn_tpots);
        ar_slot_free_at[ug] = std::max(start_ar, ar_slot_free_at[ug]) + gpu_added_ms;
        if (crit_rem_ms > 0.0 && policy.preemption_mode != "non_preemptive") {
            gpu_free_at[ug] = std::max(gpu_free_at[ug], ar_slot_free_at[ug]);
        }
        return t_cur;
    };

    auto execute_prep_start = [&](int sid_i, EvPrepStart p, double ev_ms) {
        const size_t usid = static_cast<size_t>(sid_i);
        const size_t ug = static_cast<size_t>(p.cur_gpu);
        queued_work_ms[ug] = std::max(0.0, queued_work_ms[ug] - p.est_prep_q);
        p.est_prep_q = 0.0;

        const auto& sess_spec = built_sessions[usid];
        const auto& turn = sess_spec.turns[static_cast<size_t>(p.ti)];
        const std::string& quad = sess_spec.quadrant;
        const std::string& plan = turn.modality_plan;
        const int height = turn.height;
        const bool is_1024 = height >= 1024;
        const int n_turns = static_cast<int>(sess_spec.turns.size());

        const auto [restore_ok, swap_in_ms] = restore_kv_resident(sid_i, p.cur_gpu, is_1024);
        if (!restore_ok) {
            ledgers[ug].release_session(sid_i);
            ++gpu_oom_counts[ug];
            record_oom(usid, p.ti, ev_ms);
            return;
        }

        const bool has_img_input = (turn.num_images > 0) || (plan == "dit_edit" && p.ti > 0);
        const bool prev_was_ar_after_dit =
            (p.ti >= 2 &&
             sess_spec.turns[static_cast<size_t>(p.ti - 1)].modality_plan == "ar_only" &&
             sess_spec.turns[static_cast<size_t>(p.ti - 2)].modality_plan != "ar_only");
        double base_delta_kv = 28.0;
        if (quad == "multiturn_edit" || has_img_input) {
            const double full_edit_delta = ph.turn_edit_512_kv_delta_mib * (is_1024 ? kEdit1024KvRatio : 1.0);
            if (quad == "interleaved_co_creation" && prev_was_ar_after_dit && turn.num_images == 0) {
                base_delta_kv = std::max(28.0, full_edit_delta - 268.0);
            } else {
                base_delta_kv = full_edit_delta;
            }
        }
        const double cand_main_kv = sess_main_kv_mib[usid] + base_delta_kv * ws.kv_per_turn_scale;
        const bool has_hist_for_cfg = has_img_input || (p.ti > 0);
        const double dit_phys_kv =
            (!policy.use_module3_cow_loopback && has_hist_for_cfg)
                ? (cand_main_kv * ph.non_cow_kv_mult)
                : cand_main_kv;

        const bool is_fission_und =
            ((policy.routing_mode == "static_nccl_fission" || policy.routing_mode == "twostage_shm") &&
             policy.role_of(p.cur_gpu) == "und");
        const bool pager_partial = policy.use_expert_pager && (ledgers[ug].expert_alpha() < 0.999);
        const bool defer_full_turn = has_img_input && pager_partial && !is_fission_und;
        const double src_prep_kv =
            defer_full_turn ? std::max(4.0, sess_main_kv_mib[usid]) : (is_fission_und ? cand_main_kv : dit_phys_kv);

        bool prep_mem_ok = ledgers[ug].upsert_session_kv(sid_i, src_prep_kv);
        if (prep_mem_ok && has_img_input && !defer_full_turn && !is_fission_und) {
            const double sc_vae = (is_1024 ? ph.scoped_vae_1024_spike_mib : ph.scoped_vae_512_spike_mib) * ws.vae_spike_scale;
            const double un_vae = (is_1024 ? ph.unscoped_vae_1024_reserved_mib : ph.unscoped_vae_512_reserved_mib) * ws.vae_spike_scale;
            const double vae_enc_spike = policy.use_scoped_vae ? sc_vae : un_vae;
            prep_mem_ok = ledgers[ug].check_transient_spike(
                vae_enc_spike, sid_i, /*contiguous_alloc=*/false, std::max(0.0, un_vae - sc_vae));
        }
        if (!prep_mem_ok) {
            handle_oom_or_retry(usid, p.ti, p.attempt, p.cur_gpu, -1, ev_ms, /*push_retry_event=*/true);
            return;
        }

        const double img_vae_ms = is_1024 ? vae_encode_1024_ms : vae_encode_512_ms;
        const double full_loop_ms =
            (policy.use_module3_cow_loopback ? ms_img_loop_512_ms : stock_img_loop_512_ms) +
            (is_1024 ? (vae_encode_1024_ms - vae_encode_512_ms) : 0.0);

        double prep_gpu_ms = swap_in_ms + ar_prefill_ms;
        double deferred_dit_prep_ms = 0.0;
        if (has_img_input) {
            if (defer_full_turn) {
                prep_gpu_ms = swap_in_ms + (5.0 / sp);
                deferred_dit_prep_ms = full_loop_ms;
            } else if (is_fission_und) {
                prep_gpu_ms = swap_in_ms + vit_only_img_ms;
                deferred_dit_prep_ms = img_vae_ms;
            } else {
                const double naive_vae_swap =
                    policy.use_naive_offload ? (1.08 * compute_naive_swap_ms(p.cur_gpu, ev_ms + swap_in_ms)) : 0.0;
                prep_gpu_ms = swap_in_ms + naive_vae_swap + full_loop_ms;
            }
        }

        const int main_seq_tokens = static_cast<int>(std::round(
            (cand_main_kv / std::max(0.01, ws.weight_scale)) / std::max(1e-4, ph.kv_mib_per_token)));
        const bool has_shared_prefix = policy.use_module3_cow_loopback && has_hist_for_cfg;
        const double dit_core_ms =
            compute_dit_core_ms(p.cur_gpu, is_1024, has_img_input, has_shared_prefix, main_seq_tokens, turn.dit_steps);
        const double step_base = is_1024 ? dit1024_step_ms : dit512_step_ms;
        const double vae_dec_ms = is_1024 ? vae1024_ms : vae512_ms;

        const double dit_ready_ms = ev_ms + prep_gpu_ms;
        gpu_free_at[ug] = std::max(gpu_free_at[ug], dit_ready_ms);
        push_ev(
            dit_ready_ms,
            EV_DIT_READY,
            sid_i,
            EvDitReady{
                p.ti,
                p.attempt,
                p.cur_gpu,
                p.t_turn0,
                quad,
                plan,
                height,
                turn.dit_steps,
                n_turns,
                cand_main_kv,
                dit_phys_kv,
                step_base,
                dit_core_ms,
                vae_dec_ms,
                deferred_dit_prep_ms,
                defer_full_turn});
    };

    auto execute_dit_start = [&](int sid_i, EvDitStart p, double ev_ms) {
        const size_t usid = static_cast<size_t>(sid_i);
        const size_t udst = static_cast<size_t>(p.dst_gpu);
        inflight_ship_kv_bytes[udst] = std::max<int64_t>(0, inflight_ship_kv_bytes[udst] - p.inflight_kv_b);
        p.inflight_kv_b = 0;

        const double start_dit = ev_ms;
        if (!ledgers[udst].upsert_session_kv(sid_i, p.dit_phys_kv)) {
            queued_work_ms[udst] = std::max(0.0, queued_work_ms[udst] - p.reserved_est_ms);
            handle_oom_or_retry(usid, p.ti, p.attempt, p.dst_gpu, p.cur_gpu, start_dit, /*push_retry_event=*/true);
            return;
        }

        if (p.ship_mode == "nccl_dual" || p.ship_mode == "nccl_static" || p.ship_mode == "shm_static") {
            ++sess_ships[usid];
            if (!topo.same_node(p.cur_gpu, p.dst_gpu)) {
                ++sess_xnode_ships[usid];
            }
            if (p.ship_mode == "nccl_dual") {
                sess_gpu[usid] = p.dst_gpu;
            }
        }

        double naive_dit_swap_ms = 0.0;
        if (policy.use_naive_offload) {
            naive_dit_swap_ms = 1.08 * compute_naive_swap_ms(p.dst_gpu, start_dit + p.pre_denoise_gpu_ms);
        }

        const auto [dit_ok_mem, swap_out_mib] = ledgers[udst].execute_dit_and_vae(
            p.height, sid_i, p.cand_main_kv, p.unpack_staging_mib, p.run_pre_denoise_vae);
        if (!dit_ok_mem) {
            queued_work_ms[udst] = std::max(0.0, queued_work_ms[udst] - p.reserved_est_ms);
            const double fail_frac = ledgers[udst].last_fail_dit_frac();
            const double vae_wasted_ms = (fail_frac >= 0.99) ? (0.5 * p.vae_dec_ms + (350.0 / sp)) : 0.0;
            const double abort_end =
                start_dit + p.pre_denoise_gpu_ms + naive_dit_swap_ms + p.dit_core_ms * fail_frac + vae_wasted_ms;
            gpu_free_at[udst] = std::max(gpu_free_at[udst], abort_end);
            handle_oom_or_retry(usid, p.ti, p.attempt, p.dst_gpu, p.cur_gpu, abort_end, /*push_retry_event=*/true);
            return;
        }

        const double swap_out_ms =
            (swap_out_mib > 0.0 && policy.kv_eviction_mode != "recompute")
                ? (((swap_out_mib * static_cast<double>(kMiB) / 1e9) / std::max(0.5, topo.solo_h2d_gbps(p.dst_gpu))) *
                   1000.0)
                : 0.0;
        const double denoise_start = start_dit + p.pre_denoise_gpu_ms + naive_dit_swap_ms + swap_out_ms;
        const double denoise_end = denoise_start + p.dit_core_ms;
        const double end_dit = denoise_end + p.vae_dec_ms;

        const int tot_steps = std::max(1, p.dit_steps);
        const double avg_step_ms = p.dit_core_ms / static_cast<double>(tot_steps);
        const int uncond_steps = static_cast<int>(std::floor(0.4 * static_cast<double>(tot_steps)));
        const int cfg_steps = std::max(0, tot_steps - uncond_steps);
        const double cfg_branches = (p.cand_main_kv > 220.0 * ws.kv_per_turn_scale) ? 3.0 : 2.0;
        const double tot_branch_passes =
            static_cast<double>(uncond_steps) * 1.0 + static_cast<double>(cfg_steps) * cfg_branches;
        const double one_branch_ms = p.dit_core_ms / std::max(1.0, tot_branch_passes);

        dit_denoise_start[udst] = denoise_start;
        dit_denoise_cfg_start[udst] = denoise_start + static_cast<double>(uncond_steps) * one_branch_ms;
        dit_denoise_end[udst] = denoise_end;
        dit_full_end[udst] = end_dit;
        dit_active_uncond_step_ms[udst] = (p.height >= 1024) ? one_branch_ms : avg_step_ms;
        dit_active_cfg_step_ms[udst] = (p.height >= 1024) ? (cfg_branches * one_branch_ms) : avg_step_ms;
        dit_active_sid[udst] = sid_i;
        dit_active_is1024[udst] = (p.height >= 1024);
        sess_dit_actual_end[usid] = end_dit;
        gpu_free_at[udst] = std::max(gpu_free_at[udst], end_dit);
        workers.at(p.dst_gpu).expert_alpha = ledgers[udst].expert_alpha();
        workers.at(p.dst_gpu).post_evict = ledgers[udst].is_post_evict();

        push_ev(
            end_dit,
            EV_DIT_END,
            sid_i,
            EvDitEnd{
                p.ti,
                p.attempt,
                p.cur_gpu,
                p.dst_gpu,
                p.ship_mode,
                p.t_turn0,
                p.height,
                p.dit_steps,
                p.n_turns,
                p.cand_main_kv,
                p.reserved_est_ms,
                p.ephemeral_gen_ctx,
                p.needs_return_ship});
    };

    while (!pq.empty()) {
        SimEvent cur_ev = std::move(const_cast<SimEvent&>(pq.top()));
        pq.pop();
        const double ev_ms = cur_ev.ts_ms;
        const int sid = cur_ev.sid;
        const size_t usid = static_cast<size_t>(std::max(0, sid));

        if (cur_ev.kind == EV_GPU_DISPATCH) {
            const auto gd = std::get<EvGpuDispatch>(cur_ev.payload);
            const size_t ug = static_cast<size_t>(gd.gpu_id);
            gpu_dispatch_armed_at[ug] = -1.0;
            if (gpu_wait_specs[ug].empty()) {
                continue;
            }
            if (excl_gpu_free_at(ug) > ev_ms + 1e-6) {
                arm_gpu_dispatch(gd.gpu_id, excl_gpu_free_at(ug));
                continue;
            }
            const auto sel = gpu_schedulers[ug].select_and_pop(
                gpu_wait_specs[ug], ev_ms, ledgers[ug].free_kv_bytes());
            if (!sel.has_value()) {
                continue;
            }
            PendingGpuWork chosen_work;
            bool found = false;
            auto& payloads = gpu_wait_payloads[ug];
            for (size_t i = 0; i < payloads.size(); ++i) {
                if (payloads[i].waiter_id == sel->waiter_id) {
                    chosen_work = std::move(payloads[i]);
                    payloads.erase(payloads.begin() + static_cast<std::ptrdiff_t>(i));
                    found = true;
                    break;
                }
            }
            if (found) {
                if (chosen_work.is_prep) {
                    execute_prep_start(chosen_work.sid, chosen_work.prep, ev_ms);
                } else {
                    execute_dit_start(chosen_work.sid, chosen_work.dit, ev_ms);
                }
            }
            if (!gpu_wait_specs[ug].empty()) {
                arm_gpu_dispatch(gd.gpu_id, std::max(ev_ms, excl_gpu_free_at(ug)));
            }
            continue;
        }

        if (cur_ev.kind == EV_RELEASE) {
            const int old_g = sess_gpu[usid];
            if (old_g >= 0) {
                ledgers[static_cast<size_t>(old_g)].release_session(sid);
            }
            continue;
        }

        if (cur_ev.kind == EV_PREP_START) {
            const auto p = std::get<EvPrepStart>(cur_ev.payload);
            const auto& turn = built_sessions[usid].turns[static_cast<size_t>(p.ti)];
            const bool is_1024 = turn.height >= 1024;
            const double est_kv_mib = sess_main_kv_mib[usid] + (is_1024 ? 512.0 : 256.0);
            GpuQueueWaiter w;
            w.est_mem_bytes = static_cast<int64_t>(est_kv_mib * static_cast<double>(kMiB));
            w.est_exec_ms = p.est_prep_q;
            w.slo_deadline_ms = p.t_turn0 + policy.dit_turn_slo_for(turn.height, turn.dit_steps) * slo_time_scale;
            w.op_priority = 1;
            PendingGpuWork work;
            work.sid = sid;
            work.is_prep = true;
            work.prep = p;
            enqueue_gpu_work(p.cur_gpu, ev_ms, w, std::move(work));
            continue;
        }

        if (cur_ev.kind == EV_DIT_READY) {
            const auto p = std::get<EvDitReady>(cur_ev.payload);
            const double dit_ready_ms = ev_ms;
            const bool has_more_turns = (p.ti + 1) < p.n_turns;
            const auto& sess_spec = built_sessions[usid];
            bool has_more_edits = false;
            for (int nxt = p.ti + 1; nxt < p.n_turns; ++nxt) {
                if (sess_spec.turns[static_cast<size_t>(nxt)].modality_plan == "dit_edit") {
                    has_more_edits = true;
                    break;
                }
            }

            std::map<int, bool> busy_at_dit;
            std::map<int, int> dit_at_dit;
            std::map<int, int> pre_at_dit;
            sync_states(dit_ready_ms, busy_at_dit, dit_at_dit, pre_at_dit);

            const double wire_kv_mib = p.cand_main_kv;
            const int64_t wire_kv_bytes =
                std::max<int64_t>(1LL * kMiB, static_cast<int64_t>(wire_kv_mib * static_cast<double>(kMiB)));
            const std::string fam = (policy.model_family == "internvl_u") ? "internvl_u" : "bagel";
            const double est_dit_res_ms =
                (default_operator_profile().unloaded_dit_turn_ms(fam, p.height, p.dit_steps) * cscale) / sp;

            // Single source of truth for cross-paradigm transition planning (read-only query; zero router/pacer mutation).
            const TurnTransitionPlan tplan = ctrl.plan_turn_transition_readonly(
                "s" + std::to_string(sid),
                "ctx_s" + std::to_string(sid),
                p.ti,
                p.attempt,
                p.cur_gpu,
                wire_kv_bytes,
                p.plan,
                p.height,
                p.dit_steps,
                has_more_turns,
                has_more_edits,
                dit_ready_ms,
                &workers,
                busy_at_dit,
                expert_bytes_scaled,
                p.step_base,
                est_dit_res_ms);
            const RouteDecision& dec = tplan.decision;
            const std::string& ship_mode = tplan.ship_mode;
            const int dst_gpu = tplan.dst_gpu;

            if (ship_mode == "shm_static") {
                const double shm_wire_mib = tplan.include_cfg_dup_on_wire ? p.dit_phys_kv : p.cand_main_kv;
                const double shm_tot_ms =
                    (ph.twostage_shm_base_ms + ph.twostage_shm_ms_per_mib * shm_wire_mib) /
                    std::sqrt(std::max(1.0, sp));
                const double send_ms = 0.71 * shm_tot_ms;
                const double recv_ms = 0.29 * shm_tot_ms;
                const int64_t inflight_b = tplan.reserved_kv_bytes;
                queued_work_ms[static_cast<size_t>(dst_gpu)] += tplan.est_dit_ms;
                inflight_ship_kv_bytes[static_cast<size_t>(dst_gpu)] += inflight_b;
                push_ev(
                    std::max(dit_ready_ms, gpu_free_at[static_cast<size_t>(p.cur_gpu)]),
                    EV_SHM_SEND,
                    sid,
                    EvShmSend{
                        p,
                        dst_gpu,
                        send_ms,
                        recv_ms,
                        tplan.est_dit_ms,
                        inflight_b,
                        tplan.release_src_on_fwd,
                        tplan.ephemeral_gen_ctx,
                        tplan.needs_return_ship},
                    dit_ready_ms);
                continue;
            }

            double wire_ready_ms = dit_ready_ms;
            double pre_denoise_gpu_ms = p.deferred_dit_prep_ms;
            int64_t inflight_b = 0;
            const double unpack_staging_mib = 0.0;
            const bool run_pre_vae = (ship_mode == "nccl_static" && p.deferred_dit_prep_ms > 0.0);

            if (ship_mode == "nccl_dual" || ship_mode == "nccl_static") {
                const size_t usrc = static_cast<size_t>(p.cur_gpu);
                const double state_on_wire_mib =
                    p.deferred_full_turn ? std::max(4.0, sess_main_kv_mib[usid]) : p.cand_main_kv;
                const double pack_ms = (0.05 * state_on_wire_mib) / sp;
                const double src_pack_done = dit_ready_ms + pack_ms;
                if (tplan.release_src_on_fwd) {
                    ledgers[usrc].release_session(sid);
                } else {
                    ledgers[usrc].upsert_session_kv(sid, p.cand_main_kv);
                }
                const int src_node = topo.node_of(p.cur_gpu);
                const double r_wait = (dec.hop_tier == "inter_node")
                                          ? std::max(0.0, rdma_free_at[static_cast<size_t>(src_node)] - src_pack_done)
                                          : 0.0;
                const double bw = topo.effective_p2p_gbps(p.cur_gpu, dst_gpu);
                const double wire_ms =
                    ((state_on_wire_mib * static_cast<double>(kMiB) / 1e9) / std::max(0.5, bw)) * 1000.0 +
                    topo.handoff_ms(p.cur_gpu, dst_gpu);
                if (!topo.same_node(p.cur_gpu, dst_gpu)) {
                    rdma_free_at[static_cast<size_t>(src_node)] =
                        std::max(src_pack_done, rdma_free_at[static_cast<size_t>(src_node)]) + wire_ms;
                }
                wire_ready_ms = src_pack_done + r_wait + wire_ms;
                inflight_b = tplan.reserved_kv_bytes;
                inflight_ship_kv_bytes[static_cast<size_t>(dst_gpu)] += inflight_b;
            } else {
                // Commit H2D pacer reservation strictly at in-place morph dispatch and evaluate
                // ground-truth hardware stall from the calibrated `gt_atlas` physics oracle so
                // miscalibrated routing cost models pay real hardware stall rather than their own estimate.
                double pacer_wait_ms = 0.0;
                const auto& dst_w = workers.at(dst_gpu);
                const double alpha = dst_w.expert_alpha;
                const double missing = (1.0 - alpha) * static_cast<double>(expert_bytes_scaled);
                const double bw_gt = std::max(1.0, dec.selected_bw_gbps);
                if (policy.routing_mode == "dual_path" && dec.path == "in_place_morph" &&
                    policy.use_h2d_pacer && pacer && missing > 0.0) {
                    const double dma_ms = (missing / 1e9) / bw_gt * 1000.0;
                    pacer_wait_ms = pacer->reserve_slot(dst_gpu, dma_ms, dit_ready_ms + dst_w.queue_wait_ms);
                }
                if (!policy.use_naive_offload) {
                    const MorphStallEstimate gt_est = gt_atlas.estimate_morph_stall(
                        fam,
                        p.height,
                        alpha,
                        dst_w.post_evict,
                        expert_bytes_scaled,
                        bw_gt,
                        p.step_base,
                        (ph.vmm_full_expert_map_ms * ws.weight_scale) / sp,
                        0.90,
                        ph.num_expert_layers);
                    const double morph_ovh = std::max(0.0, pacer_wait_ms + gt_est.map_ms + gt_est.exposed_stall_ms);
                    pre_denoise_gpu_ms += morph_ovh;
                    if (policy.routing_mode == "dual_path") {
                        ++sess_morphs[usid];
                    }
                }
            }

            queued_work_ms[static_cast<size_t>(dst_gpu)] += tplan.est_dit_ms;
            push_ev(
                wire_ready_ms,
                EV_DIT_START,
                sid,
                EvDitStart{
                    p.ti,
                    p.attempt,
                    p.cur_gpu,
                    dst_gpu,
                    ship_mode,
                    p.t_turn0,
                    p.height,
                    p.dit_steps,
                    p.n_turns,
                    p.cand_main_kv,
                    p.dit_phys_kv,
                    p.dit_core_ms,
                    p.vae_dec_ms,
                    pre_denoise_gpu_ms,
                    tplan.est_dit_ms,
                    inflight_b,
                    unpack_staging_mib,
                    run_pre_vae,
                    tplan.ephemeral_gen_ctx,
                    tplan.needs_return_ship},
                wire_ready_ms);
            continue;
        }

        if (cur_ev.kind == EV_SHM_SEND) {
            const auto s = std::get<EvShmSend>(cur_ev.payload);
            const size_t usrc = static_cast<size_t>(s.r.cur_gpu);
            if (excl_gpu_free_at(usrc) > ev_ms + 1e-6) {
                push_ev(excl_gpu_free_at(usrc), EV_SHM_SEND, sid, s, cur_ev.fifo_enq_ms, cur_ev.fifo_seq);
                continue;
            }
            if (!ledgers[usrc].check_transient_spike(2.4 * s.r.cand_main_kv, sid, /*contiguous_alloc=*/true)) {
                queued_work_ms[static_cast<size_t>(s.dst_gpu)] =
                    std::max(0.0, queued_work_ms[static_cast<size_t>(s.dst_gpu)] - s.dst_est_q);
                inflight_ship_kv_bytes[static_cast<size_t>(s.dst_gpu)] =
                    std::max<int64_t>(0, inflight_ship_kv_bytes[static_cast<size_t>(s.dst_gpu)] - s.dst_inflight_b);
                handle_oom_or_retry(usid, s.r.ti, s.r.attempt, s.r.cur_gpu, -1, ev_ms, /*push_retry_event=*/true);
                continue;
            }
            const double end_send = ev_ms + s.send_ms;
            gpu_free_at[usrc] = end_send;
            if (s.release_src) {
                ledgers[usrc].release_session(sid);
            } else {
                ledgers[usrc].upsert_session_kv(sid, s.r.cand_main_kv);
            }
            push_ev(
                end_send,
                EV_DIT_START,
                sid,
                EvDitStart{
                    s.r.ti,
                    s.r.attempt,
                    s.r.cur_gpu,
                    s.dst_gpu,
                    "shm_static",
                    s.r.t_turn0,
                    s.r.height,
                    s.r.dit_steps,
                    s.r.n_turns,
                    s.r.cand_main_kv,
                    s.r.dit_phys_kv,
                    s.r.dit_core_ms,
                    s.r.vae_dec_ms,
                    s.recv_ms + s.r.deferred_dit_prep_ms,
                    s.dst_est_q,
                    s.dst_inflight_b,
                    s.r.cand_main_kv,
                    s.r.deferred_dit_prep_ms > 0.0,
                    s.ephemeral_gen_ctx,
                    s.needs_return_ship},
                end_send);
            continue;
        }

        if (cur_ev.kind == EV_DIT_START) {
            const auto p = std::get<EvDitStart>(cur_ev.payload);
            const bool is_1024 = p.height >= 1024;
            const double dit_act_mib =
                (is_1024 ? (4.0 * ph.dit_512_transient_mib) : ph.dit_512_transient_mib) * std::sqrt(ws.weight_scale);
            const double vae_mib =
                (policy.use_scoped_vae ? (is_1024 ? ph.scoped_vae_1024_spike_mib : ph.scoped_vae_512_spike_mib)
                                       : (is_1024 ? ph.unscoped_vae_1024_reserved_mib : ph.unscoped_vae_512_reserved_mib)) *
                ws.vae_spike_scale;
            const double est_mem_mib = p.dit_phys_kv + std::max(dit_act_mib, vae_mib);
            GpuQueueWaiter w;
            w.est_mem_bytes = static_cast<int64_t>(est_mem_mib * static_cast<double>(kMiB));
            w.est_exec_ms = p.reserved_est_ms;
            w.slo_deadline_ms = p.t_turn0 + policy.dit_turn_slo_for(p.height, p.dit_steps) * slo_time_scale;
            w.op_priority = 1;
            PendingGpuWork work;
            work.sid = sid;
            work.is_prep = false;
            work.dit = p;
            enqueue_gpu_work(p.dst_gpu, ev_ms, w, std::move(work));
            continue;
        }

        if (cur_ev.kind == EV_DIT_END) {
            auto p = std::get<EvDitEnd>(cur_ev.payload);
            if (sess_dit_actual_end[usid] > ev_ms + 1e-6) {
                push_ev(sess_dit_actual_end[usid], EV_DIT_END, sid, p);
                continue;
            }
            const double end_dit = ev_ms;
            const size_t udst = static_cast<size_t>(p.dst_gpu);
            queued_work_ms[udst] = std::max(0.0, queued_work_ms[udst] - p.reserved_est_ms);
            p.reserved_est_ms = 0.0;
            if (dit_active_sid[udst] == sid) {
                dit_active_sid[udst] = -1;
            }

            if (p.ephemeral_gen_ctx && !p.needs_return_ship) {
                ledgers[udst].release_session(sid);
            }

            if (p.needs_return_ship) {
                if (p.ship_mode == "nccl_static") {
                    ledgers[udst].release_session(sid);
                    const int dst_node = topo.node_of(p.dst_gpu);
                    const double ret_bw = topo.effective_p2p_gbps(p.dst_gpu, p.cur_gpu);
                    const double ret_wire_ms =
                        ((p.cand_main_kv * static_cast<double>(kMiB) / 1e9) / std::max(0.5, ret_bw)) * 1000.0 +
                        topo.handoff_ms(p.dst_gpu, p.cur_gpu);
                    double ret_r_wait = 0.0;
                    if (!topo.same_node(p.dst_gpu, p.cur_gpu)) {
                        ret_r_wait = std::max(0.0, rdma_free_at[static_cast<size_t>(dst_node)] - end_dit);
                        rdma_free_at[static_cast<size_t>(dst_node)] =
                            std::max(end_dit, rdma_free_at[static_cast<size_t>(dst_node)]) + ret_wire_ms;
                    }
                    const double wire_done = end_dit + ret_r_wait + ret_wire_ms;
                    push_ev(
                        std::max(wire_done, gpu_free_at[static_cast<size_t>(p.cur_gpu)]),
                        EV_RET_UNPACK,
                        sid,
                        EvRetUnpack{
                            p.ti,
                            p.attempt,
                            p.cur_gpu,
                            p.dst_gpu,
                            p.t_turn0,
                            p.height,
                            p.dit_steps,
                            p.n_turns,
                            p.cand_main_kv,
                            0.0},
                        wire_done);
                } else {
                    const double ret_shm_ms =
                        (ph.twostage_shm_base_ms + ph.twostage_shm_ms_per_mib * p.cand_main_kv) /
                        std::sqrt(std::max(1.0, sp));
                    push_ev(
                        std::max(end_dit, gpu_free_at[udst]),
                        EV_RET_SHM_SEND,
                        sid,
                        EvRetShmSend{p, 0.58 * ret_shm_ms, 0.42 * ret_shm_ms},
                        end_dit);
                }
            } else {
                const int next_gpu = p.ephemeral_gen_ctx ? p.cur_gpu : p.dst_gpu;
                complete_dit_turn(
                    usid,
                    p.ti,
                    p.n_turns,
                    next_gpu,
                    p.height,
                    p.dit_steps,
                    p.cand_main_kv,
                    p.t_turn0,
                    end_dit,
                    p.ephemeral_gen_ctx);
            }
            continue;
        }

        if (cur_ev.kind == EV_RET_SHM_SEND) {
            const auto rs = std::get<EvRetShmSend>(cur_ev.payload);
            const size_t udst = static_cast<size_t>(rs.e.dst_gpu);
            if (gpu_free_at[udst] > ev_ms + 1e-6) {
                push_ev(gpu_free_at[udst], EV_RET_SHM_SEND, sid, rs, cur_ev.fifo_enq_ms, cur_ev.fifo_seq);
                continue;
            }
            if (!ledgers[udst].check_transient_spike(2.0 * rs.e.cand_main_kv, sid, /*contiguous_alloc=*/true)) {
                handle_oom_or_retry(usid, rs.e.ti, rs.e.attempt, rs.e.dst_gpu, rs.e.cur_gpu, ev_ms, /*push_retry_event=*/true);
                continue;
            }
            const double end_send = ev_ms + rs.send_ms;
            gpu_free_at[udst] = end_send;
            ledgers[udst].release_session(sid);
            push_ev(
                std::max(end_send, gpu_free_at[static_cast<size_t>(rs.e.cur_gpu)]),
                EV_RET_UNPACK,
                sid,
                EvRetUnpack{
                    rs.e.ti,
                    rs.e.attempt,
                    rs.e.cur_gpu,
                    rs.e.dst_gpu,
                    rs.e.t_turn0,
                    rs.e.height,
                    rs.e.dit_steps,
                    rs.e.n_turns,
                    rs.e.cand_main_kv,
                    rs.recv_ms},
                end_send);
            continue;
        }

        if (cur_ev.kind == EV_RET_UNPACK) {
            const auto p = std::get<EvRetUnpack>(cur_ev.payload);
            const size_t ucur = static_cast<size_t>(p.cur_gpu);
            if (p.recv_gpu_ms > 0.0 && gpu_free_at[ucur] > ev_ms + 1e-6) {
                push_ev(gpu_free_at[ucur], EV_RET_UNPACK, sid, p, cur_ev.fifo_enq_ms, cur_ev.fifo_seq);
                continue;
            }
            const double turn_end_ms = ev_ms + p.recv_gpu_ms;
            if (p.recv_gpu_ms > 0.0) {
                gpu_free_at[ucur] = std::max(gpu_free_at[ucur], turn_end_ms);
                if (!ledgers[ucur].check_transient_spike(2.0 * p.cand_main_kv, sid, /*contiguous_alloc=*/true)) {
                    handle_oom_or_retry(usid, p.ti, p.attempt, p.cur_gpu, -1, turn_end_ms, /*push_retry_event=*/true);
                    continue;
                }
            }
            if (!ledgers[ucur].upsert_session_kv(sid, p.cand_main_kv)) {
                handle_oom_or_retry(usid, p.ti, p.attempt, p.cur_gpu, -1, turn_end_ms, /*push_retry_event=*/true);
                continue;
            }
            if (!topo.same_node(p.dst_gpu, p.cur_gpu)) {
                ++sess_xnode_ships[usid];
            }
            complete_dit_turn(
                usid,
                p.ti,
                p.n_turns,
                p.cur_gpu,
                p.height,
                p.dit_steps,
                p.cand_main_kv,
                p.t_turn0,
                turn_end_ms,
                /*ephemeral_gen_ctx=*/true);
            continue;
        }

        auto turn_ev = std::get<EvTurn>(cur_ev.payload);
        const int ti = turn_ev.ti;
        int attempt = turn_ev.attempt;
        const auto& sess_spec = built_sessions[usid];
        if (sess_spec.turns.empty() || ti >= static_cast<int>(sess_spec.turns.size())) {
            continue;
        }
        const std::string& quad = sess_spec.quadrant;
        const auto& turn = sess_spec.turns[static_cast<size_t>(ti)];
        const std::string& plan = turn.modality_plan;
        const int height = turn.height;
        const bool is_1024 = height >= 1024;
        const int n_turns = static_cast<int>(sess_spec.turns.size());

        while (true) {
            std::map<int, bool> busy_map;
            std::map<int, int> dit_map;
            std::map<int, int> pre_map;
            sync_states(ev_ms, busy_map, dit_map, pre_map);
            int cur_gpu = sess_gpu[usid];
            if (cur_gpu < 0) {
                cur_gpu = router->select_initial_gpu(
                    num_gpus,
                    policy.routing_mode,
                    policy.preemption_mode,
                    policy.static_und_ranks,
                    plan,
                    &workers,
                    busy_map,
                    dit_map,
                    pre_map,
                    2,
                    policy.cptg_architecture());
                sess_gpu[usid] = cur_gpu;
            }

            if (plan == "ar_only") {
                const size_t ucur = static_cast<size_t>(cur_gpu);
                const double ar_enq = (turn_ev.ar_enq_ms >= 0.0) ? turn_ev.ar_enq_ms : ev_ms;
                if (policy.preemption_mode == "non_preemptive" && gpu_free_at[ucur] > ev_ms + 1e-6) {
                    push_ev(
                        gpu_free_at[ucur],
                        EV_TURN,
                        sid,
                        EvTurn{ti, attempt, ar_enq, true},
                        ar_enq,
                        turn_ev.waiting_ar_lock ? cur_ev.fifo_seq : 0);
                    break;
                }
                const auto [ar_restore_ok, swap_in_ms] = restore_kv_resident(sid, cur_gpu, is_1024);
                if (!ar_restore_ok) {
                    ledgers[ucur].release_session(sid);
                    ++gpu_oom_counts[ucur];
                    record_oom(usid, ti, ev_ms);
                    break;
                }
                const bool prev_was_dit =
                    (ti > 0 && sess_spec.turns[static_cast<size_t>(ti - 1)].modality_plan != "ar_only");
                const bool loopback_prev_dit_img =
                    (turn.num_images == 0) && (turn.use_prev_image || (prev_was_dit && quad == "interleaved_co_creation"));
                const int eff_ar_images = loopback_prev_dit_img ? 1 : turn.num_images;
                const double ar_img_kv = static_cast<double>(eff_ar_images) * 268.0;
                const double ar_delta = (ar_img_kv + 3.5) * ws.kv_per_turn_scale;
                const double new_kv = sess_kv_mib[usid] + ar_delta;
                if (!ledgers[ucur].upsert_session_kv(sid, new_kv)) {
                    if (handle_oom_or_retry(usid, ti, attempt, cur_gpu, -1, ev_ms, /*push_retry_event=*/false)) {
                        ++attempt;
                        turn_ev.ar_enq_ms = ar_enq;
                        turn_ev.waiting_ar_lock = false;
                        continue;
                    }
                    break;
                }
                sess_kv_mib[usid] = new_kv;
                sess_main_kv_mib[usid] += ar_delta;

                const double lock_w = std::max(0.0, ev_ms - ar_enq);
                double ar_ttft_val = 0.0;
                std::vector<double> ar_turn_tp;
                const double end_ms = run_ar_phase_sim(
                    usid,
                    cur_gpu,
                    ev_ms,
                    swap_in_ms,
                    lock_w,
                    turn,
                    /*prepare_for_dit=*/false,
                    loopback_prev_dit_img,
                    &ar_ttft_val,
                    &ar_turn_tp);
                const bool ar_has_img = (eff_ar_images > 0);
                const int ar_img_res = (height > 0) ? height : 512;
                sess_ar_has_image[usid].push_back(ar_has_img ? 1 : 0);
                sess_ar_num_images[usid].push_back(eff_ar_images);
                sess_ar_image_res[usid].push_back(ar_img_res);
                sess_turn_records[usid].push_back(SimTurnRec{
                    ti,
                    "ar_only",
                    "ok",
                    ar_ttft_val,
                    std::move(ar_turn_tp),
                    std::nullopt,
                    0,
                    0,
                    sess_main_kv_mib[usid],
                    ar_has_img,
                    eff_ar_images,
                    ar_img_res,
                    loopback_prev_dit_img});
                if (ti + 1 < n_turns) {
                    const double think_ms = compute_think_ms(usid, ti + 1);
                    sess_think_ms[usid] += think_ms;
                    push_ev(end_ms + think_ms, EV_TURN, sid, EvTurn{ti + 1, 0});
                } else {
                    sess_end_ms[usid] = end_ms;
                    push_ev(end_ms, EV_RELEASE, sid, std::monostate{});
                }
                break;
            }

            if (plan == "cot_then_dit") {
                const size_t ucur = static_cast<size_t>(cur_gpu);
                const double ar_enq = (turn_ev.ar_enq_ms >= 0.0) ? turn_ev.ar_enq_ms : ev_ms;
                if (policy.preemption_mode == "non_preemptive" && gpu_free_at[ucur] > ev_ms + 1e-6) {
                    push_ev(
                        gpu_free_at[ucur],
                        EV_TURN,
                        sid,
                        EvTurn{ti, attempt, ar_enq, true},
                        ar_enq,
                        turn_ev.waiting_ar_lock ? cur_ev.fifo_seq : 0);
                    break;
                }
                const auto [cot_restore_ok, cot_restore_ms] = restore_kv_resident(sid, cur_gpu, is_1024);
                if (!cot_restore_ok) {
                    ledgers[ucur].release_session(sid);
                    ++gpu_oom_counts[ucur];
                    record_oom(usid, ti, ev_ms);
                    break;
                }
                double swap_in_ms = cot_restore_ms;
                const double base_delta_kv =
                    (turn.num_images > 0)
                        ? (ph.turn_edit_512_kv_delta_mib * (is_1024 ? kEdit1024KvRatio : 1.0) + 4.0)
                        : (sess_spec.session_id.empty() ? 185.0 : 8.0);
                const double cand_main_kv = sess_main_kv_mib[usid] + base_delta_kv * ws.kv_per_turn_scale;
                const bool has_hist_for_cfg = (turn.num_images > 0) || (ti > 0);
                const double dit_phys_kv =
                    (!policy.use_module3_cow_loopback && has_hist_for_cfg)
                        ? (cand_main_kv * ph.non_cow_kv_mult)
                        : cand_main_kv;
                const bool is_fission_und =
                    ((policy.routing_mode == "static_nccl_fission" || policy.routing_mode == "twostage_shm") &&
                     policy.role_of(cur_gpu) == "und");
                const bool pager_partial = policy.use_expert_pager && (ledgers[ucur].expert_alpha() < 0.999);
                const bool can_vae_here = (turn.num_images > 0) && (!is_fission_und) && (!pager_partial);
                const double src_prep_kv = is_fission_und ? cand_main_kv : dit_phys_kv;
                bool cot_mem_ok = ledgers[ucur].upsert_session_kv(sid, src_prep_kv);
                if (cot_mem_ok && can_vae_here) {
                    const double sc_vae = (is_1024 ? ph.scoped_vae_1024_spike_mib : ph.scoped_vae_512_spike_mib) * ws.vae_spike_scale;
                    const double un_vae = (is_1024 ? ph.unscoped_vae_1024_reserved_mib : ph.unscoped_vae_512_reserved_mib) * ws.vae_spike_scale;
                    const double vae_enc_spike = policy.use_scoped_vae ? sc_vae : un_vae;
                    cot_mem_ok = ledgers[ucur].check_transient_spike(
                        vae_enc_spike, sid, /*contiguous_alloc=*/false, std::max(0.0, un_vae - sc_vae));
                }
                if (!cot_mem_ok) {
                    if (handle_oom_or_retry(usid, ti, attempt, cur_gpu, -1, ev_ms, /*push_retry_event=*/false)) {
                        ++attempt;
                        turn_ev.ar_enq_ms = ar_enq;
                        turn_ev.waiting_ar_lock = false;
                        continue;
                    }
                    break;
                }
                if (policy.use_naive_offload && can_vae_here) {
                    swap_in_ms += 1.08 * compute_naive_swap_ms(cur_gpu, ev_ms + swap_in_ms);
                }

                const double lock_w = std::max(0.0, ev_ms - ar_enq);
                double cot_ttft_val = 0.0;
                std::vector<double> cot_turn_tp;
                const double dit_ready_ms = run_ar_phase_sim(
                    usid,
                    cur_gpu,
                    ev_ms,
                    swap_in_ms,
                    lock_w,
                    turn,
                    /*prepare_for_dit=*/true,
                    /*loopback_prev_dit_img=*/false,
                    &cot_ttft_val,
                    &cot_turn_tp);
                const bool cot_has_img = (turn.num_images > 0) || turn.use_prev_image;
                const int cot_n_img = (turn.num_images > 0) ? turn.num_images : (turn.use_prev_image ? 1 : 0);
                const int cot_img_res = (height > 0) ? height : 512;
                sess_ar_has_image[usid].push_back(cot_has_img ? 1 : 0);
                sess_ar_num_images[usid].push_back(cot_n_img);
                sess_ar_image_res[usid].push_back(cot_img_res);
                sess_pending_cot_ar[usid] = PendingCotAr{
                    true, ti, cot_ttft_val, std::move(cot_turn_tp), cot_has_img, cot_n_img, cot_img_res, turn.use_prev_image};
                const double deferred_vae_ms =
                    (turn.num_images > 0 && !can_vae_here) ? (is_1024 ? vae_encode_1024_ms : vae_encode_512_ms) : 0.0;

                const int main_seq_tokens = static_cast<int>(std::round(
                    (cand_main_kv / std::max(0.01, ws.weight_scale)) / std::max(1e-4, ph.kv_mib_per_token)));
                const bool has_shared_prefix =
                    policy.use_module3_cow_loopback &&
                    (ti > 0 || quad == "visual_cot" || quad == "visual_cot_highres");
                const double dit_core_ms =
                    compute_dit_core_ms(cur_gpu, is_1024, (turn.num_images > 0), has_shared_prefix, main_seq_tokens, turn.dit_steps);
                const double step_base = is_1024 ? dit1024_step_ms : dit512_step_ms;
                const double vae_dec_ms = is_1024 ? vae1024_ms : vae512_ms;

                // D3: Measure DiT stage latency strictly after AR CoT completes (`dit_ready_ms`).
                push_ev(
                    dit_ready_ms,
                    EV_DIT_READY,
                    sid,
                    EvDitReady{
                        ti,
                        attempt,
                        cur_gpu,
                        dit_ready_ms,
                        quad,
                        plan,
                        height,
                        turn.dit_steps,
                        n_turns,
                        cand_main_kv,
                        dit_phys_kv,
                        step_base,
                        dit_core_ms,
                        vae_dec_ms,
                        deferred_vae_ms,
                        /*deferred_full_turn=*/false});
                break;
            }

            // `dit_edit` or `dit_t2i`: enqueue onto per-GPU Memory+Slack wait queue causally (`EV_PREP_START`).
            const double est_prep_q = (240.0 * cscale) / sp;
            queued_work_ms[static_cast<size_t>(cur_gpu)] += est_prep_q;
            push_ev(
                ev_ms,
                EV_PREP_START,
                sid,
                EvPrepStart{ti, attempt, cur_gpu, ev_ms, est_prep_q},
                ev_ms);
            break;
        }
    }

    // Build canonical outcome and worker-stats arrays and delegate aggregation to `MorphControlEngine`.
    nlohmann::json outcomes_array = nlohmann::json::array();
    double max_end_ms = 0.0;
    int total_retries = 0;

    for (size_t s = 0; s < n_sess; ++s) {
        const auto& s_spec = built_sessions[s];
        total_retries += sess_retries[s];
        max_end_ms = std::max(max_end_ms, std::max(s_spec.arrival_ms, sess_end_ms[s]));

        size_t s_planned_ar = 0;
        size_t s_planned_dit = 0;
        for (const auto& tspec : s_spec.turns) {
            if (tspec.modality_plan == "ar_only" || tspec.modality_plan == "cot_then_dit") ++s_planned_ar;
            if (tspec.modality_plan != "ar_only") ++s_planned_dit;
        }

        std::vector<double> dit_lats;
        std::vector<int> dit_hs;
        std::vector<int> dit_sts;
        dit_lats.reserve(sess_dit_records[s].size());
        dit_hs.reserve(sess_dit_records[s].size());
        dit_sts.reserve(sess_dit_records[s].size());
        for (const auto& r : sess_dit_records[s]) {
            dit_lats.push_back(r.lat_ms);
            dit_hs.push_back(r.height);
            dit_sts.push_back(r.steps);
        }

        nlohmann::json trecs = nlohmann::json::array();
        for (const auto& tr : sess_turn_records[s]) {
            nlohmann::json tj = {
                {"turn_idx", tr.turn_idx},
                {"modality_plan", tr.modality_plan},
                {"status", tr.status},
                {"height", tr.height},
                {"dit_steps", tr.dit_steps},
                {"kv_mib", tr.kv_mib},
                {"has_image", tr.has_image},
                {"num_images", tr.num_images},
                {"image_res", tr.image_res},
                {"use_prev_image", tr.use_prev_image},
            };
            if (tr.ar_ttft_ms.has_value()) tj["ar_ttft_ms"] = *tr.ar_ttft_ms;
            if (!tr.ar_tpots_ms.empty()) tj["ar_tpots_ms"] = tr.ar_tpots_ms;
            if (tr.dit_ms.has_value()) tj["dit_ms"] = *tr.dit_ms;
            trecs.push_back(std::move(tj));
        }

        const bool is_oom = sess_oom[s];
        const double e2e_ms = std::max(0.0, sess_end_ms[s] - s_spec.arrival_ms);
        const double active_jct_ms = std::max(0.0, e2e_ms - sess_think_ms[s]);
        const nlohmann::json slo_j = is_oom
            ? nlohmann::json{
                  {"ar_slo_met", false},
                  {"ar_slo_p95_tpot_met", false},
                  {"ar_slo_mean_tpot_met", false},
                  {"ar_slo_strict_max_tpot_met", false},
                  {"dit_slo_met", false},
                  {"joint_slo_met", false},
              }
            : MorphControlEngine::evaluate_session_slos(
                  policy,
                  sess_ar_ttfts[s],
                  sess_ar_tpots[s],
                  dit_lats,
                  dit_hs,
                  dit_sts,
                  sess_ar_turn_tpots[s],
                  slo_time_scale,
                  sess_ar_has_image[s],
                  sess_ar_num_images[s],
                  sess_ar_image_res[s]);

        outcomes_array.push_back({
            {"session_id", s_spec.session_id.empty() ? ("sim_" + std::to_string(s)) : s_spec.session_id},
            {"quadrant", s_spec.quadrant},
            {"phase_label", s_spec.phase_label},
            {"arrival_offset_s", s_spec.arrival_ms / 1000.0},
            {"status", is_oom ? "oom" : "ok"},
            {"e2e_ms", e2e_ms},
            {"active_jct_ms", active_jct_ms},
            {"think_time_ms", sess_think_ms[s]},
            {"planned_turns", std::max<size_t>(s_spec.turns.size(), sess_turn_records[s].size())},
            {"planned_ar_turns", s_planned_ar},
            {"planned_dit_turns", s_planned_dit},
            {"ar_ttfts_ms", sess_ar_ttfts[s]},
            {"ar_tpots_ms", sess_ar_tpots[s]},
            {"ar_turn_tpots_ms", sess_ar_turn_tpots[s]},
            {"ar_turn_has_image", sess_ar_has_image[s]},
            {"ar_turn_num_images", sess_ar_num_images[s]},
            {"ar_turn_image_res", sess_ar_image_res[s]},
            {"dit_turns_ms", dit_lats},
            {"dit_turn_heights", dit_hs},
            {"dit_turn_steps", dit_sts},
            {"ar_slo_met", slo_j.at("ar_slo_met")},
            {"ar_slo_p95_tpot_met", slo_j.at("ar_slo_p95_tpot_met")},
            {"ar_slo_mean_tpot_met", slo_j.at("ar_slo_mean_tpot_met")},
            {"ar_slo_strict_max_tpot_met", slo_j.at("ar_slo_strict_max_tpot_met")},
            {"dit_slo_met", slo_j.at("dit_slo_met")},
            {"joint_slo_met", slo_j.at("joint_slo_met")},
            {"path_a_ships", sess_ships[s]},
            {"path_b_morphs", sess_morphs[s]},
            {"cross_node_ships", sess_xnode_ships[s]},
            {"preempted_ar_turns", sess_preemptions[s]},
            {"total_yields", 0},
            {"oom_retries", sess_retries[s]},
            {"turn_records", std::move(trecs)},
        });
    }

    double peak_hbm = 0.0;
    int carves_tot = 0;
    int evict_tot = 0;
    int swap_out_tot = 0;
    int swap_in_tot = 0;
    int64_t q_reorders_tot = 0;
    int64_t q_mem_bypasses_tot = 0;
    int64_t q_starv_promos_tot = 0;
    for (const auto& qs : gpu_schedulers) {
        q_reorders_tot += qs.stats().reorders_total;
        q_mem_bypasses_tot += qs.stats().memory_bypasses_total;
        q_starv_promos_tot += qs.stats().starvation_promotions_total;
    }
    nlohmann::json worker_stats_array = nlohmann::json::array();
    const double vmm_pool_mib =
        policy.use_tripool_vmm ? static_cast<double>(policy.super_pool_mib + policy.fine_kv_pool_mib) : 0.0;
    for (size_t g = 0; g < ledgers.size(); ++g) {
        const auto& l = ledgers[g];
        peak_hbm = std::max(peak_hbm, l.peak_used_mib());
        carves_tot += l.super_to_fine_carves();
        evict_tot += l.expert_evictions();
        swap_out_tot += l.kv_host_swap_outs();
        swap_in_tot += l.kv_host_swap_ins();
        worker_stats_array.push_back({
            {"gpu_id", l.gpu_id()},
            {"nvml_peak_used_mib", l.peak_used_mib()},
            {"combined_physical_mib", l.peak_used_mib()},
            {"pytorch_reserved_mib", std::max(0.0, l.peak_used_mib() - vmm_pool_mib)},
            {"vmm_physical_pool_mib", vmm_pool_mib},
            {"effective_hbm_util_mean_pct", round_2(l.effective_hbm_util_mean_pct())},
            {"oom_count", gpu_oom_counts[g]},
            {"kv_host_swap_outs", l.kv_host_swap_outs()},
            {"kv_host_swap_ins", l.kv_host_swap_ins()},
            {"kv_recompute_evictions", l.kv_recompute_evictions()},
            {"kv_recompute_restores", l.kv_recompute_restores()},
            {"vmm_stats",
             {
                 {"overflow_creates", 0},
                 {"super_to_fine_carves", l.super_to_fine_carves()},
                 {"expert_auto_evictions_for_kv", l.expert_evictions()},
                 {"expert_auto_evictions_for_transient", 0},
                 {"kv_swap_outs_total", l.kv_host_swap_outs()},
                 {"kv_swap_ins_total", l.kv_host_swap_ins()},
             }},
        });
    }

    const std::string eff_skew = skew_tag.empty() ? "main_mix" : skew_tag;
    const nlohmann::json sim_extra = {
        {"workload_scale", ws.name},
        {"skew_tag", eff_skew},
        {"interconnect", hw.name},
        {"crossover_kv_mib", crossover_kv_mib(hw, ph, ws)},
        {"edf_guard_ms", round_2(edf_512.guard_ms())},
        {"edf_chunk_layers_512", chunk_layers_512},
        {"edf_chunk_layers_1024", chunk_layers_1024},
        {"oom_retries", total_retries},
        {"peak_hbm_mib", round_1(peak_hbm)},
        {"vmm_carves", carves_tot},
        {"vmm_expert_evictions", evict_tot},
        {"vmm_kv_host_swap_outs", swap_out_tot},
        {"vmm_kv_host_swap_ins", swap_in_tot},
        {"queue_reorders_total", q_reorders_tot},
        {"queue_memory_bypasses_total", q_mem_bypasses_tot},
        {"queue_starvation_promotions_total", q_starv_promos_tot},
        {"kv_swap_in_mean_ms", round_2(mean_raw(sim_swap_in_latencies_ms))},
        {"kv_swap_in_p95_ms", percentile_rounded(sim_swap_in_latencies_ms, 95.0)},
        {"kv_recompute_events", sim_kv_recompute_events},
        {"kv_recompute_total_ms", round_2(sim_kv_recompute_total_ms)},
        {"kv_saved_recompute_ms", round_2(sim_kv_saved_recompute_ms)},
    };

    const double wall_seconds = std::max(1e-3, max_end_ms / 1000.0);
    return MorphControlEngine::aggregate_cluster_run_report(
        policy,
        eff_skew,
        offered_qps,
        wall_seconds,
        0.0,
        outcomes_array,
        worker_stats_array,
        slo_time_scale,
        sim_extra,
        /*include_sessions=*/false);
}

}  // namespace morphserve::control
