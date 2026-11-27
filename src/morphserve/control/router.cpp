#include "morphserve/control/router.hpp"

#include "morphserve/control/hash_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace morphserve::control {

namespace {

inline int pos_mod(int x, int m) noexcept {
    if (m <= 1) return 0;
    const int r = x % m;
    return (r < 0) ? (r + m) : r;
}

int select_static_gen_gpu_unlocked(
    const ClusterTopologyProfile& topo,
    const std::string& routing_mode,
    const std::vector<int>& static_und_ranks,
    const std::vector<int>& static_gen_ranks,
    int src_und_gpu,
    const std::map<int, WorkerState>& w_map,
    const std::map<int, bool>& gpu_busy,
    int64_t required_kv_bytes) {
    (void)static_und_ranks;
    if (static_gen_ranks.empty()) {
        return src_und_gpu;
    }

    const int64_t need_b = std::max<int64_t>(64LL * kMiB, required_kv_bytes);
    const double state_b = static_cast<double>(std::max<int64_t>(1LL * kMiB, required_kv_bytes));
    auto tier_rank = [&](int g) -> int {
        const std::string t = topo.fabric_tier(src_und_gpu, g);
        if (t == "intra_half") return 0;
        if (t == "inter_socket") return 1;
        if (t == "inter_node") return 2;
        return 3;
    };
    auto is_busy = [&](int g) -> bool {
        auto bit = gpu_busy.find(g);
        return (bit != gpu_busy.end()) ? bit->second : false;
    };

    std::vector<int> cands = static_gen_ranks;
    if (routing_mode == "twostage_shm") {
        // Fair baseline: load-balance across ALL static_gen_ranks (preferring same-node SHM,
        // sufficient free KV, lowest queue wait, and idle ranks) instead of hardcoded 2k->2k+1 pairing.
        std::sort(cands.begin(), cands.end(), [&](int a, int b) {
            auto key_of = [&](int g) {
                const auto& wg = w_map.at(g);
                return std::make_tuple(
                    topo.same_node(src_und_gpu, g) ? 0 : 1,
                    wg.kv_free_bytes >= need_b ? 0 : 1,
                    round_3(wg.queue_wait_ms),
                    is_busy(g),
                    tier_rank(g),
                    g);
            };
            return key_of(a) < key_of(b);
        });
        return cands.front();
    }

    std::sort(cands.begin(), cands.end(), [&](int a, int b) {
        auto key_of = [&](int g) {
            const double bw = std::max(0.5, topo.effective_p2p_gbps(src_und_gpu, g));
            const double t_ship = (state_b / 1e9) / bw * 1000.0 + topo.handoff_ms(src_und_gpu, g);
            const auto& wg = w_map.at(g);
            const double tot_wait = round_3(t_ship + wg.queue_wait_ms);
            return std::make_tuple(
                wg.kv_free_bytes >= need_b ? 0 : 1,
                tot_wait,
                tier_rank(g),
                is_busy(g),
                g);
        };
        return key_of(a) < key_of(b);
    });
    return cands.front();
}

RouteDecision decide_unlocked(
    const DualPathMorphRouter& router,
    const ClusterTopologyProfile& topo,
    const CPTGRequestSpec& req,
    const std::map<int, WorkerState>& w_map,
    double now_ms) {
    const auto& src_w = w_map.at(req.src_gpu);
    const auto [t_morph_ms, h2d_bw] = router.evaluate_path_b_morph_ms(req, src_w, now_ms);

    int best_dst = req.src_gpu;
    double best_ship_ms = std::numeric_limits<double>::infinity();
    double best_p2p_bw = 0.0;

    for (int cand_id : topo.peer_candidates(req.src_gpu, true)) {
        auto it = w_map.find(cand_id);
        if (it == w_map.end()) continue;
        const auto [t_ship, bw_p2p] = router.evaluate_path_a_ship_ms(req, src_w, it->second, now_ms);
        if (t_ship < best_ship_ms) {
            best_ship_ms = t_ship;
            best_dst = cand_id;
            best_p2p_bw = bw_p2p;
        }
    }

    const double state_mib = round_2(static_cast<double>(req.effective_state_bytes()) / static_cast<double>(kMiB));
    char reason_buf[256];
    if (best_ship_ms < t_morph_ms) {
        const std::string tier = topo.fabric_tier(req.src_gpu, best_dst);
        const std::string hop_label = (tier == "intra_half") ? "same_half" : tier;
        std::snprintf(
            reason_buf,
            sizeof(reason_buf),
            "%s: %s state_ship(%.2fMiB) %.1fms < morph %.1fms",
            req.architecture.c_str(),
            hop_label.c_str(),
            state_mib,
            best_ship_ms,
            t_morph_ms);
        return RouteDecision{
            req.req_id,
            "state_ship",
            req.src_gpu,
            best_dst,
            best_ship_ms,
            t_morph_ms,
            best_ship_ms,
            best_p2p_bw,
            state_mib,
            reason_buf,
            tier};
    }

    std::snprintf(
        reason_buf,
        sizeof(reason_buf),
        "%s: in_place_morph(alpha=%.2f) %.1fms <= ship %.1fms",
        req.architecture.c_str(),
        src_w.expert_alpha,
        t_morph_ms,
        best_ship_ms);
    return RouteDecision{
        req.req_id,
        "in_place_morph",
        req.src_gpu,
        req.src_gpu,
        (best_ship_ms < 1e12) ? best_ship_ms : -1.0,
        t_morph_ms,
        t_morph_ms,
        h2d_bw,
        state_mib,
        reason_buf,
        "in_place"};
}

std::pair<RouteDecision, std::string> plan_transition_unlocked(
    const DualPathMorphRouter& router,
    const ClusterTopologyProfile& topo,
    const std::string& routing_mode,
    const std::vector<int>& static_und_ranks,
    const std::vector<int>& static_gen_ranks,
    const CPTGRequestSpec& req,
    const std::map<int, WorkerState>& w_map,
    const std::map<int, bool>& gpu_busy,
    double now_ms) {
    if (routing_mode == "dual_path") {
        RouteDecision dec = decide_unlocked(router, topo, req, w_map, now_ms);
        const std::string ship_mode =
            (dec.path == "state_ship" && dec.dst_gpu != req.src_gpu) ? "nccl_dual" : "none";
        return {std::move(dec), ship_mode};
    }

    if (routing_mode == "static_nccl_fission" || routing_mode == "twostage_shm") {
        const int dst_gpu = select_static_gen_gpu_unlocked(
            topo,
            routing_mode,
            static_und_ranks,
            static_gen_ranks,
            req.src_gpu,
            w_map,
            gpu_busy,
            req.effective_state_bytes());
        const std::string ship_mode = (routing_mode == "static_nccl_fission") ? "nccl_static" : "shm_static";
        const double bw = topo.effective_p2p_gbps(req.src_gpu, dst_gpu);
        const double t_ship =
            (static_cast<double>(req.effective_state_bytes()) / 1e9) / std::max(0.5, bw) * 1000.0 +
            topo.handoff_ms(req.src_gpu, dst_gpu);
        RouteDecision dec{
            req.req_id,
            "state_ship",
            req.src_gpu,
            dst_gpu,
            round_3(t_ship),
            std::numeric_limits<double>::infinity(),
            round_3(t_ship),
            round_3(bw),
            round_2(static_cast<double>(req.effective_state_bytes()) / static_cast<double>(kMiB)),
            "static " + routing_mode + " " + std::to_string(req.src_gpu) + "->" + std::to_string(dst_gpu),
            topo.fabric_tier(req.src_gpu, dst_gpu)};
        return {std::move(dec), ship_mode};
    }

    const int dst_gpu = req.src_gpu;
    const double q_ms = round_3(w_map.at(dst_gpu).queue_wait_ms);
    RouteDecision dec{
        req.req_id,
        "in_place_morph",
        req.src_gpu,
        dst_gpu,
        -1.0,
        q_ms,
        q_ms,
        topo.solo_h2d_gbps(dst_gpu),
        round_2(static_cast<double>(req.effective_state_bytes()) / static_cast<double>(kMiB)),
        "sticky " + routing_mode + " on gpu " + std::to_string(dst_gpu),
        "in_place"};
    return {std::move(dec), "none"};
}

}  // namespace

int64_t CPTGRequestSpec::effective_state_bytes() const {
    if (architecture == "bridge_mllm_dit") {
        return (bridge_bytes > 0) ? bridge_bytes : kv_bytes;
    }
    if (architecture == "tight_mot") {
        return kv_bytes;
    }
    throw std::invalid_argument("unsupported CPTG architecture: " + architecture);
}

nlohmann::json CPTGRequestSpec::to_json() const {
    return {
        {"req_id", req_id},
        {"src_gpu", src_gpu},
        {"architecture", architecture},
        {"kv_bytes", kv_bytes},
        {"bridge_bytes", bridge_bytes},
        {"gen_expert_bytes", gen_expert_bytes},
        {"dit_step1_ms", dit_step1_ms},
        {"dit_total_steps", dit_total_steps},
        {"resolution", resolution},
        {"interleaved_roundtrip_gamma", interleaved_roundtrip_gamma},
        {"model_family", model_family},
    };
}

CPTGRequestSpec CPTGRequestSpec::from_json(const nlohmann::json& j) {
    CPTGRequestSpec r;
    r.req_id = j.value("req_id", "");
    r.src_gpu = j.value("src_gpu", 0);
    r.architecture = j.value("architecture", "tight_mot");
    r.kv_bytes = j.value("kv_bytes", int64_t{0});
    r.bridge_bytes = j.value("bridge_bytes", int64_t{0});
    r.gen_expert_bytes = j.value("gen_expert_bytes", default_operator_profile().bagel_expert_bytes);
    r.dit_step1_ms = j.value("dit_step1_ms", default_operator_profile().bagel_dit_512_step_ms);
    r.dit_total_steps = j.value("dit_total_steps", 20);
    r.resolution = j.value("resolution", 512);
    r.interleaved_roundtrip_gamma = j.value("interleaved_roundtrip_gamma", 1);
    r.model_family = j.value("model_family", "");
    return r;
}

nlohmann::json WorkerState::to_json() const {
    return {
        {"gpu_id", gpu_id},
        {"role", role},
        {"expert_alpha", expert_alpha},
        {"queue_wait_ms", queue_wait_ms},
        {"active_h2d_transfers", active_h2d_transfers},
        {"kv_free_bytes", kv_free_bytes},
        {"p2p_concurrent_links", p2p_concurrent_links},
        {"post_evict", post_evict},
        {"colocated_ar_b", colocated_ar_b},
    };
}

WorkerState WorkerState::from_json(const nlohmann::json& j) {
    WorkerState w;
    w.gpu_id = j.value("gpu_id", 0);
    w.role = j.value("role", "hybrid");
    w.expert_alpha = j.value("expert_alpha", 1.0);
    w.queue_wait_ms = j.value("queue_wait_ms", 0.0);
    w.active_h2d_transfers = j.value("active_h2d_transfers", 0);
    w.kv_free_bytes = j.value("kv_free_bytes", 12LL * kGiB);
    w.p2p_concurrent_links = j.value("p2p_concurrent_links", 1);
    w.post_evict = j.value("post_evict", false);
    w.colocated_ar_b = j.value("colocated_ar_b", 0);
    return w;
}

nlohmann::json RouteDecision::to_json() const {
    return {
        {"req_id", req_id},
        {"path", path},
        {"src_gpu", src_gpu},
        {"dst_gpu", dst_gpu},
        {"t_ship_ms", std::isinf(t_ship_ms) ? -1.0 : t_ship_ms},
        {"t_morph_ms", std::isinf(t_morph_ms) ? -1.0 : t_morph_ms},
        {"chosen_cost_ms", chosen_cost_ms},
        {"selected_bw_gbps", selected_bw_gbps},
        {"effective_state_mib", effective_state_mib},
        {"reason", reason},
        {"hop_tier", hop_tier},
    };
}

DualPathMorphRouter::DualPathMorphRouter(
    ClusterTopologyProfile topology,
    std::shared_ptr<runtime::HalfNodeH2DPacer> pacer,
    OperatorLatencyProfile operator_profile,
    std::optional<double> state_handoff_ms,
    double map_overhead_ms_full,
    double overlap_efficiency,
    bool allow_cross_half_state_ship,
    bool allow_inter_node_state_ship,
    int64_t max_inter_node_state_bytes,
    std::string cost_model_mode,
    PerformanceAtlas performance_atlas)
    : topology_(std::move(topology)),
      pacer_(pacer ? std::move(pacer)
                   : std::make_shared<runtime::HalfNodeH2DPacer>(topology_.gpu_to_group_map(), 1)),
      operator_profile_(operator_profile),
      state_handoff_ms_(state_handoff_ms),
      map_overhead_ms_full_(map_overhead_ms_full),
      overlap_efficiency_(overlap_efficiency),
      allow_cross_half_state_ship_(allow_cross_half_state_ship),
      allow_inter_node_state_ship_(allow_inter_node_state_ship),
      max_inter_node_state_bytes_(max_inter_node_state_bytes),
      cost_model_mode_(std::move(cost_model_mode)),
      atlas_(std::move(performance_atlas)) {}

void DualPathMorphRouter::set_cost_model_mode(std::string mode) {
    std::lock_guard<std::mutex> lock(mu_);
    cost_model_mode_ = std::move(mode);
}

void DualPathMorphRouter::set_performance_atlas(PerformanceAtlas atlas) {
    std::lock_guard<std::mutex> lock(mu_);
    atlas_ = std::move(atlas);
}

std::map<int, WorkerState> DualPathMorphRouter::init_workers(
    int num_gpus,
    const std::function<std::string(int)>& role_of,
    const std::map<int, double>& initial_alphas,
    const std::map<int, int64_t>& kv_free_bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    raw_kv_free_bytes_.clear();
    inflight_kv_reserved_bytes_.clear();
    last_telemetry_seq_.clear();
    worker_states_.clear();
    for (int g = 0; g < num_gpus; ++g) {
        auto kit = kv_free_bytes.find(g);
        const int64_t raw_b = (kit != kv_free_bytes.end()) ? std::max<int64_t>(0, kit->second) : (16LL * kGiB);
        auto ait = initial_alphas.find(g);
        const double alpha = (ait != initial_alphas.end()) ? ait->second : 1.0;
        raw_kv_free_bytes_[g] = raw_b;
        inflight_kv_reserved_bytes_[g] = 0;
        last_telemetry_seq_[g] = -1;
        WorkerState ws;
        ws.gpu_id = g;
        ws.role = role_of ? role_of(g) : "hybrid";
        ws.expert_alpha = alpha;
        ws.queue_wait_ms = 0.0;
        ws.kv_free_bytes = raw_b;
        worker_states_[g] = std::move(ws);
    }
    rr_cursor_ = 0;
    return worker_states_;
}

std::map<int, WorkerState> DualPathMorphRouter::worker_states_snapshot() const {
    std::lock_guard<std::mutex> lock(mu_);
    return worker_states_;
}

WorkerState DualPathMorphRouter::get_worker_state(int gpu_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = worker_states_.find(gpu_id);
    if (it == worker_states_.end()) {
        throw std::out_of_range("worker state not found for gpu_id=" + std::to_string(gpu_id));
    }
    return it->second;
}

void DualPathMorphRouter::set_worker_state_fields(
    int gpu_id,
    std::optional<double> expert_alpha,
    std::optional<double> queue_wait_ms,
    std::optional<int> active_h2d_transfers,
    std::optional<int64_t> kv_free_bytes,
    std::optional<int> p2p_concurrent_links) {
    std::lock_guard<std::mutex> lock(mu_);
    auto& w = worker_states_[gpu_id];
    w.gpu_id = gpu_id;
    if (expert_alpha.has_value()) w.expert_alpha = *expert_alpha;
    if (queue_wait_ms.has_value()) w.queue_wait_ms = std::max(0.0, *queue_wait_ms);
    if (active_h2d_transfers.has_value()) w.active_h2d_transfers = std::max(0, *active_h2d_transfers);
    if (kv_free_bytes.has_value()) {
        const int64_t v = std::max<int64_t>(0, *kv_free_bytes);
        w.kv_free_bytes = v;
        raw_kv_free_bytes_[gpu_id] = v + inflight_kv_reserved_bytes_[gpu_id];
    }
    if (p2p_concurrent_links.has_value()) w.p2p_concurrent_links = std::max(1, *p2p_concurrent_links);
}

void DualPathMorphRouter::update_worker_telemetry(
    int gpu_id,
    std::optional<double> expert_alpha,
    std::optional<int64_t> kv_free_bytes,
    std::optional<int64_t> telemetry_seq,
    std::map<int, WorkerState>* ext_workers) {
    std::lock_guard<std::mutex> lock(mu_);
    auto& w_map = ext_workers ? *ext_workers : worker_states_;
    auto it = w_map.find(gpu_id);
    if (it == w_map.end()) {
        return;
    }
    if (telemetry_seq.has_value()) {
        const int64_t prev_seq = last_telemetry_seq_.count(gpu_id) ? last_telemetry_seq_[gpu_id] : -1;
        if (*telemetry_seq < prev_seq) {
            return;
        }
        last_telemetry_seq_[gpu_id] = *telemetry_seq;
    }
    if (expert_alpha.has_value()) {
        const double prev_a = it->second.expert_alpha;
        const double new_a = *expert_alpha;
        if (new_a < prev_a - 1e-4) {
            it->second.post_evict = true;
        } else if (new_a >= 0.999) {
            it->second.post_evict = false;
        }
        it->second.expert_alpha = new_a;
    }
    if (kv_free_bytes.has_value()) {
        const int64_t raw_b = std::max<int64_t>(0, *kv_free_bytes);
        raw_kv_free_bytes_[gpu_id] = raw_b;
        const int64_t inf = inflight_kv_reserved_bytes_.count(gpu_id) ? inflight_kv_reserved_bytes_[gpu_id] : 0;
        it->second.kv_free_bytes = std::max<int64_t>(0, raw_b - inf);
    }
}

void DualPathMorphRouter::reserve_ar_queue(int gpu_id, double est_ar_ms) {
    const double delta = std::max(0.0, est_ar_ms);
    if (delta <= 0.0) {
        return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    auto it = worker_states_.find(gpu_id);
    if (it != worker_states_.end()) {
        it->second.queue_wait_ms += delta;
        ++it->second.colocated_ar_b;
    }
}

void DualPathMorphRouter::release_ar_queue(int gpu_id, double est_ar_ms) {
    const double delta = std::max(0.0, est_ar_ms);
    if (delta <= 0.0) {
        return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    auto it = worker_states_.find(gpu_id);
    if (it != worker_states_.end()) {
        it->second.queue_wait_ms = std::max(0.0, it->second.queue_wait_ms - delta);
        it->second.colocated_ar_b = std::max(0, it->second.colocated_ar_b - 1);
    }
}

int DualPathMorphRouter::select_initial_gpu(
    int num_gpus,
    const std::string& routing_mode,
    const std::string& preemption_mode,
    const std::vector<int>& static_und_ranks,
    const std::string& modality_plan,
    const std::map<int, WorkerState>* ext_workers,
    const std::map<int, bool>& gpu_busy,
    const std::map<int, int>& active_dit_ranks,
    const std::map<int, int>& active_preemptions,
    int max_preemptions_per_gpu,
    const std::string& architecture) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto& w_map = ext_workers ? *ext_workers : worker_states_;
    const int n = std::max(1, num_gpus);
    const int64_t min_kv_b = 128LL * kMiB;
    const int cur = rr_cursor_;

    auto is_busy = [&](int g) -> bool {
        auto it = gpu_busy.find(g);
        return (it != gpu_busy.end()) ? it->second : false;
    };
    auto dit_cnt = [&](int g) -> int {
        auto it = active_dit_ranks.find(g);
        return (it != active_dit_ranks.end()) ? it->second : 0;
    };
    auto pre_cnt = [&](int g) -> int {
        auto it = active_preemptions.find(g);
        return (it != active_preemptions.end()) ? it->second : 0;
    };

    if ((routing_mode == "twostage_shm" || routing_mode == "static_nccl_fission") && !static_und_ranks.empty()) {
        std::vector<int> cands = static_und_ranks;
        std::sort(cands.begin(), cands.end(), [&](int a, int b) {
            const auto& wa = w_map.at(a);
            const auto& wb = w_map.at(b);
            auto ka = std::make_tuple(wa.kv_free_bytes >= min_kv_b ? 0 : 1, wa.queue_wait_ms, is_busy(a), pos_mod(a - cur, n));
            auto kb = std::make_tuple(wb.kv_free_bytes >= min_kv_b ? 0 : 1, wb.queue_wait_ms, is_busy(b), pos_mod(b - cur, n));
            return ka < kb;
        });
        const int chosen = cands.front();
        rr_cursor_ = (chosen + 1) % n;
        return chosen;
    }

    if (modality_plan == "ar_only") {
        std::vector<int> idle;
        for (int g = 0; g < n; ++g) {
            const auto& wg = w_map.at(g);
            if (!is_busy(g) && wg.queue_wait_ms <= 1.0 && wg.kv_free_bytes >= min_kv_b) {
                idle.push_back(g);
            }
        }
        if (!idle.empty()) {
            std::sort(idle.begin(), idle.end(), [&](int a, int b) {
                return std::make_tuple(w_map.at(a).expert_alpha, pos_mod(a - cur, n)) <
                       std::make_tuple(w_map.at(b).expert_alpha, pos_mod(b - cur, n));
            });
            const int chosen = idle.front();
            rr_cursor_ = (chosen + 1) % n;
            return chosen;
        }
        if (preemption_mode != "non_preemptive") {
            std::vector<int> elig;
            for (int g = 0; g < n; ++g) {
                if (pre_cnt(g) < max_preemptions_per_gpu && w_map.at(g).kv_free_bytes >= min_kv_b) {
                    elig.push_back(g);
                }
            }
            if (!elig.empty()) {
                std::sort(elig.begin(), elig.end(), [&](int a, int b) {
                    return std::make_tuple(pre_cnt(a), dit_cnt(a) > 0 ? 1 : 0, w_map.at(a).expert_alpha, pos_mod(a - cur, n)) <
                           std::make_tuple(pre_cnt(b), dit_cnt(b) > 0 ? 1 : 0, w_map.at(b).expert_alpha, pos_mod(b - cur, n));
                });
                const int chosen = elig.front();
                rr_cursor_ = (chosen + 1) % n;
                return chosen;
            }
        }
    }

    if (modality_plan == "cot_then_dit") {
        std::vector<int> idle;
        for (int g = 0; g < n; ++g) {
            const auto& wg = w_map.at(g);
            if (!is_busy(g) && wg.queue_wait_ms <= 1.0 && wg.kv_free_bytes >= min_kv_b) {
                idle.push_back(g);
            }
        }
        if (!idle.empty()) {
            const bool prefer_high_alpha = (routing_mode == "dual_path" && architecture != "bridge_mllm_dit");
            std::sort(idle.begin(), idle.end(), [&](int a, int b) {
                const double aa = prefer_high_alpha ? -w_map.at(a).expert_alpha : w_map.at(a).expert_alpha;
                const double ab = prefer_high_alpha ? -w_map.at(b).expert_alpha : w_map.at(b).expert_alpha;
                return std::make_tuple(aa, pos_mod(a - cur, n)) <
                       std::make_tuple(ab, pos_mod(b - cur, n));
            });
            const int chosen = idle.front();
            rr_cursor_ = (chosen + 1) % n;
            return chosen;
        }
        if (preemption_mode != "non_preemptive") {
            std::vector<int> elig;
            for (int g = 0; g < n; ++g) {
                if (pre_cnt(g) < max_preemptions_per_gpu && w_map.at(g).kv_free_bytes >= min_kv_b) {
                    elig.push_back(g);
                }
            }
            if (!elig.empty()) {
                std::sort(elig.begin(), elig.end(), [&](int a, int b) {
                    return std::make_tuple(pre_cnt(a), w_map.at(a).queue_wait_ms, pos_mod(a - cur, n)) <
                           std::make_tuple(pre_cnt(b), w_map.at(b).queue_wait_ms, pos_mod(b - cur, n));
                });
                const int chosen = elig.front();
                rr_cursor_ = (chosen + 1) % n;
                return chosen;
            }
        }
    }

    std::vector<int> cands(static_cast<size_t>(n));
    for (int g = 0; g < n; ++g) cands[static_cast<size_t>(g)] = g;
    std::sort(cands.begin(), cands.end(), [&](int a, int b) {
        const auto& wa = w_map.at(a);
        const auto& wb = w_map.at(b);
        double aa = 0.0;
        double ab = 0.0;
        if (routing_mode == "dual_path") {
            // For `bridge_mllm_dit` (InternVL-U), `PREP_DIT` runs VLM first and ships a compact
            // bridge via Path A to hot Decoder GPUs: prefer low decoder alpha (VLM resident).
            aa = (architecture == "bridge_mllm_dit") ? wa.expert_alpha : -wa.expert_alpha;
            ab = (architecture == "bridge_mllm_dit") ? wb.expert_alpha : -wb.expert_alpha;
        }
        return std::make_tuple(wa.kv_free_bytes >= min_kv_b ? 0 : 1, wa.queue_wait_ms, is_busy(a), aa, pos_mod(a - cur, n)) <
               std::make_tuple(wb.kv_free_bytes >= min_kv_b ? 0 : 1, wb.queue_wait_ms, is_busy(b), ab, pos_mod(b - cur, n));
    });
    const int chosen = cands.front();
    rr_cursor_ = (chosen + 1) % n;
    return chosen;
}

int DualPathMorphRouter::select_static_gen_gpu(
    const std::string& routing_mode,
    const std::vector<int>& static_und_ranks,
    const std::vector<int>& static_gen_ranks,
    int src_und_gpu,
    const std::map<int, WorkerState>* ext_workers,
    const std::map<int, bool>& gpu_busy,
    int64_t required_kv_bytes) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto& w_map = ext_workers ? *ext_workers : worker_states_;
    return select_static_gen_gpu_unlocked(
        topology_, routing_mode, static_und_ranks, static_gen_ranks, src_und_gpu, w_map, gpu_busy, required_kv_bytes);
}

int DualPathMorphRouter::select_oom_retry_gpu(
    int failed_gpu,
    int num_gpus,
    const std::string& routing_mode,
    const std::vector<int>& static_und_ranks,
    const std::map<int, WorkerState>* ext_workers,
    const std::map<int, bool>& gpu_busy) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto& w_map = ext_workers ? *ext_workers : worker_states_;
    std::vector<int> cands;
    if ((routing_mode == "twostage_shm" || routing_mode == "static_nccl_fission") && !static_und_ranks.empty()) {
        for (int g : static_und_ranks) {
            if (g != failed_gpu) cands.push_back(g);
        }
        if (cands.empty()) {
            return static_und_ranks.front();
        }
    } else {
        for (int g = 0; g < num_gpus; ++g) {
            if (g != failed_gpu) cands.push_back(g);
        }
        if (cands.empty()) {
            return failed_gpu;
        }
    }
    auto is_busy = [&](int g) -> bool {
        auto bit = gpu_busy.find(g);
        return (bit != gpu_busy.end()) ? bit->second : false;
    };
    std::sort(cands.begin(), cands.end(), [&](int a, int b) {
        const auto& wa = w_map.at(a);
        const auto& wb = w_map.at(b);
        return std::make_tuple(wa.kv_free_bytes >= 128LL * kMiB ? 0 : 1, wa.queue_wait_ms, is_busy(a), a) <
               std::make_tuple(wb.kv_free_bytes >= 128LL * kMiB ? 0 : 1, wb.queue_wait_ms, is_busy(b), b);
    });
    return cands.front();
}

std::pair<double, double> DualPathMorphRouter::evaluate_path_b_morph_ms(
    const CPTGRequestSpec& req,
    const WorkerState& src_worker,
    double now_ms) const {
    const std::string eff_fam = !req.model_family.empty()
                                    ? req.model_family
                                    : (req.architecture == "bridge_mllm_dit" ? "internvl_u" : "bagel");
    const double coloc_stretch =
        (cost_model_mode_ == "atlas_3d" && src_worker.colocated_ar_b > 0)
            ? atlas_.dit_step_stretch_factor(eff_fam, req.resolution, src_worker.colocated_ar_b)
            : 1.0;
    const double eff_q_wait_ms = src_worker.queue_wait_ms * coloc_stretch;

    const double alpha = std::max(0.0, std::min(1.0, src_worker.expert_alpha));
    const double missing_bytes = (1.0 - alpha) * static_cast<double>(req.gen_expert_bytes);
    if (missing_bytes <= 0.0) {
        return {round_3(eff_q_wait_ms), topology_.solo_h2d_gbps(src_worker.gpu_id)};
    }
    const double h2d_wait_ms = pacer_ ? pacer_->predicted_wait_ms(src_worker.gpu_id, now_ms) : 0.0;
    const int active = (!pacer_ || pacer_->max_concurrent_per_half() == 1)
                           ? 1
                           : std::max(1, src_worker.active_h2d_transfers + 1);
    const double bw_h2d_gbps = topology_.effective_h2d_gbps(src_worker.gpu_id, active);
    const int num_layers = (eff_fam == "internvl_u") ? 20 : atlas_.ref_num_layers();

    MorphStallEstimate stall_est{};
    if (cost_model_mode_ == "legacy_analytical") {
        stall_est = PerformanceAtlas::estimate_legacy_analytical(
            alpha, req.gen_expert_bytes, bw_h2d_gbps, req.dit_step1_ms, map_overhead_ms_full_, overlap_efficiency_);
    } else if (cost_model_mode_ == "pipeline_analytical") {
        stall_est = PerformanceAtlas::estimate_pipeline_analytical(
            alpha,
            src_worker.post_evict,
            req.gen_expert_bytes,
            bw_h2d_gbps,
            req.dit_step1_ms,
            map_overhead_ms_full_,
            overlap_efficiency_,
            num_layers);
    } else {
        // Default `atlas_3d`: calibrated hardware lookup + O(1) interpolation with pipeline fallback.
        stall_est = atlas_.estimate_morph_stall(
            eff_fam,
            req.resolution,
            alpha,
            src_worker.post_evict,
            req.gen_expert_bytes,
            bw_h2d_gbps,
            req.dit_step1_ms,
            map_overhead_ms_full_,
            overlap_efficiency_,
            num_layers);
    }
    const double t_morph = eff_q_wait_ms + h2d_wait_ms + stall_est.map_ms + stall_est.exposed_stall_ms;
    return {round_3(t_morph), round_3(bw_h2d_gbps)};
}

std::pair<double, double> DualPathMorphRouter::evaluate_path_a_ship_ms(
    const CPTGRequestSpec& req,
    const WorkerState& src_worker,
    const WorkerState& dst_worker,
    double now_ms) const {
    const double state_bytes = static_cast<double>(req.effective_state_bytes());
    const double inf = std::numeric_limits<double>::infinity();
    if (dst_worker.gpu_id == src_worker.gpu_id) {
        return {inf, 0.0};
    }
    if (static_cast<double>(dst_worker.kv_free_bytes) < state_bytes) {
        return {inf, 0.0};
    }
    const std::string tier = topology_.fabric_tier(src_worker.gpu_id, dst_worker.gpu_id);
    if (tier == "inter_socket" && !allow_cross_half_state_ship_) {
        return {inf, 0.0};
    }
    if (tier == "inter_node") {
        if (!allow_inter_node_state_ship_) {
            return {inf, 0.0};
        }
        if (state_bytes > static_cast<double>(max_inter_node_state_bytes_)) {
            return {inf, 0.0};
        }
    }
    const int concurrent = std::max(1, std::min(src_worker.p2p_concurrent_links, dst_worker.p2p_concurrent_links));
    const double bw_p2p = topology_.effective_p2p_gbps(
        src_worker.gpu_id,
        dst_worker.gpu_id,
        (tier == "intra_half") ? concurrent : 1,
        (tier == "inter_socket") ? concurrent : 1,
        (tier == "inter_node") ? concurrent : 1);
    const int gamma = std::max(1, req.interleaved_roundtrip_gamma);
    const double handoff = state_handoff_ms_.has_value()
                               ? *state_handoff_ms_
                               : topology_.handoff_ms(src_worker.gpu_id, dst_worker.gpu_id);
    const double ship_xfer_ms = (static_cast<double>(gamma) * state_bytes / 1e9) / bw_p2p * 1000.0 + handoff;

    const std::string eff_fam = !req.model_family.empty()
                                    ? req.model_family
                                    : (req.architecture == "bridge_mllm_dit" ? "internvl_u" : "bagel");
    const double dst_stretch =
        (cost_model_mode_ == "atlas_3d" && dst_worker.colocated_ar_b > 0)
            ? atlas_.dit_step_stretch_factor(eff_fam, req.resolution, dst_worker.colocated_ar_b)
            : 1.0;

    double dst_morph_ms = 0.0;
    if (dst_worker.expert_alpha < 1.0) {
        WorkerState probe = dst_worker;
        probe.queue_wait_ms = 0.0;
        dst_morph_ms = evaluate_path_b_morph_ms(req, probe, now_ms).first;
    }
    const double t_ship = ship_xfer_ms + (dst_worker.queue_wait_ms * dst_stretch) + dst_morph_ms;
    return {round_3(t_ship), round_3(bw_p2p)};
}

RouteDecision DualPathMorphRouter::decide(
    const CPTGRequestSpec& req,
    const std::map<int, WorkerState>* ext_workers,
    double now_ms) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto& w_map = ext_workers ? *ext_workers : worker_states_;
    return decide_unlocked(*this, topology_, req, w_map, now_ms);
}

std::pair<RouteDecision, std::string> DualPathMorphRouter::plan_transition_readonly(
    const std::string& routing_mode,
    const std::vector<int>& static_und_ranks,
    const std::vector<int>& static_gen_ranks,
    const CPTGRequestSpec& req,
    double est_dit_ms,
    const std::map<int, WorkerState>* ext_workers,
    const std::map<int, bool>& gpu_busy,
    double now_ms) const {
    (void)est_dit_ms;
    std::lock_guard<std::mutex> lock(mu_);
    const auto& w_map = ext_workers ? *ext_workers : worker_states_;
    return plan_transition_unlocked(
        *this, topology_, routing_mode, static_und_ranks, static_gen_ranks, req, w_map, gpu_busy, now_ms);
}

std::pair<RouteDecision, std::string> DualPathMorphRouter::plan_and_reserve_transition(
    const std::string& routing_mode,
    const std::vector<int>& static_und_ranks,
    const std::vector<int>& static_gen_ranks,
    const CPTGRequestSpec& req,
    double est_dit_ms,
    std::map<int, WorkerState>* ext_workers,
    const std::map<int, bool>& gpu_busy,
    bool use_pacer,
    double now_ms) {
    std::lock_guard<std::mutex> lock(mu_);
    auto& w_map = ext_workers ? *ext_workers : worker_states_;

    auto [dec, ship_mode] = plan_transition_unlocked(
        *this, topology_, routing_mode, static_und_ranks, static_gen_ranks, req, w_map, gpu_busy, now_ms);
    const int dst_gpu = dec.dst_gpu;

    if (routing_mode == "dual_path" && dec.path == "in_place_morph" && use_pacer && pacer_) {
        const double alpha = w_map.at(dst_gpu).expert_alpha;
        const double missing = (1.0 - alpha) * static_cast<double>(req.gen_expert_bytes);
        if (missing > 0.0) {
            const double bw = std::max(1.0, dec.selected_bw_gbps);
            const double dma_ms = (missing / 1e9) / bw * 1000.0;
            pacer_->reserve_slot(dst_gpu, dma_ms, now_ms + w_map.at(dst_gpu).queue_wait_ms);
        }
    }

    if (ship_mode != "none" && dst_gpu != req.src_gpu) {
        const int64_t res_kv = std::max<int64_t>(0, req.effective_state_bytes());
        if (!raw_kv_free_bytes_.count(dst_gpu)) {
            raw_kv_free_bytes_[dst_gpu] = std::max<int64_t>(0, w_map.at(dst_gpu).kv_free_bytes);
        }
        inflight_kv_reserved_bytes_[dst_gpu] += res_kv;
        w_map.at(dst_gpu).kv_free_bytes = std::max<int64_t>(0, w_map.at(dst_gpu).kv_free_bytes - res_kv);
    }
    w_map.at(dst_gpu).queue_wait_ms += std::max(0.0, est_dit_ms);
    return {dec, ship_mode};
}

void DualPathMorphRouter::reserve_kv_bytes(
    int gpu_id,
    int64_t reserved_kv_bytes,
    std::map<int, WorkerState>* ext_workers) {
    const int64_t res_kv = std::max<int64_t>(0, reserved_kv_bytes);
    if (res_kv <= 0) return;
    std::lock_guard<std::mutex> lock(mu_);
    auto& w_map = ext_workers ? *ext_workers : worker_states_;
    auto it = w_map.find(gpu_id);
    if (it == w_map.end()) return;
    if (!raw_kv_free_bytes_.count(gpu_id)) {
        raw_kv_free_bytes_[gpu_id] = std::max<int64_t>(0, it->second.kv_free_bytes);
    }
    inflight_kv_reserved_bytes_[gpu_id] += res_kv;
    it->second.kv_free_bytes = std::max<int64_t>(0, it->second.kv_free_bytes - res_kv);
}

void DualPathMorphRouter::release_kv_reservation(
    int gpu_id,
    int64_t reserved_kv_bytes,
    std::optional<int64_t> latest_kv_free_bytes,
    std::optional<int64_t> telemetry_seq,
    std::map<int, WorkerState>* ext_workers) {
    const int64_t rel = std::max<int64_t>(0, reserved_kv_bytes);
    std::lock_guard<std::mutex> lock(mu_);
    auto& w_map = ext_workers ? *ext_workers : worker_states_;
    auto it = w_map.find(gpu_id);
    if (it == w_map.end()) return;

    const int64_t cur_inf = std::max<int64_t>(0, inflight_kv_reserved_bytes_[gpu_id] - rel);
    inflight_kv_reserved_bytes_[gpu_id] = cur_inf;
    bool accept_latest = latest_kv_free_bytes.has_value();
    if (accept_latest && telemetry_seq.has_value()) {
        const int64_t prev_seq = last_telemetry_seq_.count(gpu_id) ? last_telemetry_seq_[gpu_id] : -1;
        if (*telemetry_seq < prev_seq) {
            accept_latest = false;
        } else {
            last_telemetry_seq_[gpu_id] = *telemetry_seq;
        }
    }
    if (accept_latest && latest_kv_free_bytes.has_value()) {
        raw_kv_free_bytes_[gpu_id] = std::max<int64_t>(0, *latest_kv_free_bytes);
    }
    const int64_t raw_b = raw_kv_free_bytes_.count(gpu_id)
                              ? raw_kv_free_bytes_[gpu_id]
                              : (it->second.kv_free_bytes + rel);
    it->second.kv_free_bytes = std::max<int64_t>(0, raw_b - cur_inf);
}

void DualPathMorphRouter::release_dit_reservation(
    int gpu_id,
    double est_dit_ms,
    std::map<int, WorkerState>* ext_workers) {
    std::lock_guard<std::mutex> lock(mu_);
    auto& w_map = ext_workers ? *ext_workers : worker_states_;
    auto it = w_map.find(gpu_id);
    if (it != w_map.end()) {
        it->second.queue_wait_ms = std::max(0.0, it->second.queue_wait_ms - std::max(0.0, est_dit_ms));
    }
}

double DualPathMorphRouter::analytical_crossover_kv_mib(
    double alpha,
    int64_t gen_expert_bytes,
    double dit_step1_ms,
    int src_gpu,
    int dst_gpu) const {
    WorkerState w_src;
    w_src.gpu_id = src_gpu;
    w_src.role = "hybrid";
    w_src.expert_alpha = alpha;
    w_src.queue_wait_ms = 0.0;

    CPTGRequestSpec req_probe;
    req_probe.req_id = "crossover_probe";
    req_probe.src_gpu = src_gpu;
    req_probe.architecture = "tight_mot";
    req_probe.kv_bytes = 0;
    req_probe.gen_expert_bytes = (gen_expert_bytes > 0) ? gen_expert_bytes : operator_profile_.bagel_expert_bytes;
    req_probe.dit_step1_ms = (dit_step1_ms > 0.0) ? dit_step1_ms : operator_profile_.bagel_dit_512_step_ms;

    const double t_morph_ms = evaluate_path_b_morph_ms(req_probe, w_src, 0.0).first;
    const double handoff = state_handoff_ms_.has_value()
                               ? *state_handoff_ms_
                               : topology_.handoff_ms(src_gpu, dst_gpu);
    const double net_budget_ms = std::max(0.0, t_morph_ms - handoff);
    const double bw_p2p = topology_.effective_p2p_gbps(src_gpu, dst_gpu);
    const double kv_bytes = (net_budget_ms / 1000.0) * bw_p2p * 1e9;
    return round_1(kv_bytes / static_cast<double>(kMiB));
}

std::vector<RouteDecision> DualPathMorphRouter::route_batch(
    const std::vector<CPTGRequestSpec>& requests,
    const std::map<int, WorkerState>& workers,
    double now_ms) {
    std::map<int, WorkerState> state_copy = workers;
    std::vector<RouteDecision> decisions;
    decisions.reserve(requests.size());
    for (const auto& req : requests) {
        RouteDecision dec = decide_unlocked(*this, topology_, req, state_copy, now_ms);
        decisions.push_back(dec);
        auto& dst_w = state_copy.at(dec.dst_gpu);
        if (dec.path == "in_place_morph") {
            const double alpha = dst_w.expert_alpha;
            const double missing = (1.0 - alpha) * static_cast<double>(req.gen_expert_bytes);
            if (missing > 0.0 && pacer_) {
                const double bw = std::max(1.0, dec.selected_bw_gbps);
                const double dma_ms = (missing / 1e9) / bw * 1000.0;
                pacer_->reserve_slot(dec.dst_gpu, dma_ms, now_ms + dst_w.queue_wait_ms);
            }
            dst_w.expert_alpha = 1.0;
        } else {
            dst_w.kv_free_bytes = std::max<int64_t>(0, dst_w.kv_free_bytes - req.effective_state_bytes());
        }
        const double dit_dur_ms = req.dit_step1_ms * static_cast<double>(std::max(1, req.dit_total_steps)) * 0.85;
        dst_w.queue_wait_ms += dit_dur_ms;
    }
    return decisions;
}

}  // namespace morphserve::control
