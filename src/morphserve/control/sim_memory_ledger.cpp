#include "morphserve/control/sim_memory_ledger.hpp"

#include "morphserve/control/hash_util.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace morphserve::control {

SimGpuMemoryLedger::SimGpuMemoryLedger(
    int gpu_id,
    double total_hbm_mib,
    PolicyConfig policy,
    std::string role,
    int resident_expert_layers,
    int num_expert_layers,
    int layer_stride_mib,
    ClusterPhysicsProfile physics,
    double weight_scale,
    double vae_spike_scale)
    : gpu_id_(gpu_id),
      total_hbm_mib_(total_hbm_mib),
      policy_(std::move(policy)),
      role_(std::move(role)),
      resident_expert_layers_(resident_expert_layers),
      num_expert_layers_(num_expert_layers),
      layer_stride_mib_(layer_stride_mib),
      physics_(std::move(physics)),
      weight_scale_(weight_scale),
      vae_spike_scale_(vae_spike_scale),
      cur_super_pool_mib_(static_cast<double>(policy_.super_pool_mib)),
      cur_fine_pool_mib_(static_cast<double>(policy_.fine_kv_pool_mib)) {}

double SimGpuMemoryLedger::offloaded_base_mib() const noexcept {
    const bool is_real_40g_pcie = (total_hbm_mib_ < 40900.0) && (total_hbm_mib_ > 39500.0);
    const double nccl_comm_mib =
        (is_real_40g_pcie && policy_.num_gpus >= 2 &&
         (policy_.routing_mode == "dual_path" || policy_.routing_mode == "static_nccl_fission"))
            ? 720.0
            : 0.0;
    return physics_.offloaded_base_driver_mib * weight_scale_ + nccl_comm_mib;
}

double SimGpuMemoryLedger::monolithic_weights_mib() const noexcept {
    const bool is_real_40g_pcie = (total_hbm_mib_ < 40900.0) && (total_hbm_mib_ > 39500.0);
    const double nccl_comm_mib =
        (is_real_40g_pcie && policy_.num_gpus >= 2 &&
         (policy_.routing_mode == "dual_path" || policy_.routing_mode == "static_nccl_fission"))
            ? 720.0
            : 0.0;
    return physics_.monolithic_loaded_driver_mib * weight_scale_ + nccl_comm_mib;
}

double SimGpuMemoryLedger::pageable_expert_mib() const noexcept {
    return std::max(0.0, (physics_.monolithic_loaded_driver_mib - physics_.offloaded_base_driver_mib) * weight_scale_);
}

double SimGpuMemoryLedger::base_weights_mib() const noexcept {
    if (policy_.use_expert_pager || policy_.use_naive_offload) {
        return offloaded_base_mib();
    }
    if ((policy_.routing_mode == "static_nccl_fission" || policy_.routing_mode == "twostage_shm") && role_ == "und") {
        return offloaded_base_mib();
    }
    return monolithic_weights_mib();
}

double SimGpuMemoryLedger::expert_alpha() const noexcept {
    if (!policy_.use_expert_pager) {
        return (role_ == "und" || policy_.use_naive_offload) ? 0.0 : 1.0;
    }
    return round_3(static_cast<double>(resident_expert_layers_) /
                   static_cast<double>(std::max(1, num_expert_layers_)));
}

double SimGpuMemoryLedger::active_kv_mib() const noexcept {
    double sum = 0.0;
    if (policy_.use_tripool_vmm) {
        // Each `VmmNaiveCache` allocates 28 layer slabs at `2 MiB` physical `cuMemCreate` granularity;
        // without LECP CoW, edit/CoT turns instantiate separate `main`/`cfg` caches (`38 MiB` slab rounding).
        const double slab_step_mib = policy_.use_module3_cow_loopback ? 16.0 : 56.0;
        const double non_cow_slab_rem_mib = policy_.use_module3_cow_loopback ? 0.0 : 38.0;
        for (const auto& [_, v] : session_kv_mib_) {
            sum += std::max(slab_step_mib, 2.0 * std::ceil(std::max(0.01, v) / 2.0) + non_cow_slab_rem_mib);
        }
    } else {
        for (const auto& [_, v] : session_kv_mib_) {
            sum += v;
        }
    }
    return sum;
}

int64_t SimGpuMemoryLedger::free_kv_bytes() const noexcept {
    const double exp_mib = policy_.use_expert_pager
                               ? static_cast<double>(resident_expert_layers_ * layer_stride_mib_)
                               : 0.0;
    const double used = base_weights_mib() + active_kv_mib() + exp_mib;
    return static_cast<int64_t>(std::max(0.0, total_hbm_mib_ - used - 2048.0) * static_cast<double>(kMiB));
}

void SimGpuMemoryLedger::record_hbm_sample(double used_mib) noexcept {
    if (total_hbm_mib_ <= 0.0) {
        return;
    }
    const double clamped = std::clamp(used_mib, 0.0, total_hbm_mib_);
    hbm_used_samples_sum_ += clamped;
    ++hbm_used_samples_cnt_;
}

double SimGpuMemoryLedger::effective_hbm_util_mean_pct() const noexcept {
    if (total_hbm_mib_ <= 0.0) {
        return 0.0;
    }
    if (hbm_used_samples_cnt_ <= 0) {
        return std::clamp(100.0 * peak_used_mib_ / total_hbm_mib_, 0.0, 100.0);
    }
    const double mean_used = hbm_used_samples_sum_ / static_cast<double>(hbm_used_samples_cnt_);
    return std::clamp(100.0 * mean_used / total_hbm_mib_, 0.0, 100.0);
}

double SimGpuMemoryLedger::evict_idle_kv_to_host(
    double needed_free_mib,
    std::optional<int> exclude_sid) {
    if (!policy_.use_idle_kv_swap) {
        return 0.0;
    }
    // LRU eviction order aligned with live `TriPoolVMM` (oldest `session_last_touch_` first).
    std::vector<int> sids;
    sids.reserve(session_kv_mib_.size());
    for (const auto& [sid, _] : session_kv_mib_) {
        sids.push_back(sid);
    }
    std::sort(sids.begin(), sids.end(), [&](int a, int b) {
        const uint64_t ta = session_last_touch_.count(a) ? session_last_touch_.at(a) : 0;
        const uint64_t tb = session_last_touch_.count(b) ? session_last_touch_.at(b) : 0;
        if (ta != tb) return ta < tb;
        return a < b;
    });
    const bool drop_for_recompute = (policy_.kv_eviction_mode == "recompute");
    double freed_mib = 0.0;
    for (int cand_sid : sids) {
        if (exclude_sid.has_value() && cand_sid == *exclude_sid) {
            continue;
        }
        if (freed_mib >= needed_free_mib) {
            break;
        }
        const double kv_before = active_kv_mib();
        const double val = session_kv_mib_.at(cand_sid);
        session_kv_mib_.erase(cand_sid);
        if (drop_for_recompute) {
            dropped_recompute_kv_mib_[cand_sid] = val;
            ++kv_recompute_evictions_;
        } else {
            swapped_session_kv_mib_[cand_sid] = val;
            ++kv_host_swap_outs_;
        }
        const double kv_after = active_kv_mib();
        freed_mib += std::max(0.0, kv_before - kv_after);
    }
    reclaim_empty_slabs_to_super();
    return freed_mib;
}

void SimGpuMemoryLedger::reclaim_empty_slabs_to_super() noexcept {
    if (!(policy_.use_tripool_vmm && policy_.use_expert_pager)) {
        return;
    }
    const double base_fine = static_cast<double>(policy_.fine_kv_pool_mib);
    const double super_gran = std::max(16.0, static_cast<double>(policy_.super_handle_mib));
    const double kv_tot = active_kv_mib();
    while (cur_fine_pool_mib_ > base_fine && (cur_fine_pool_mib_ - kv_tot) >= super_gran) {
        cur_fine_pool_mib_ -= super_gran;
        cur_super_pool_mib_ += super_gran;
    }
}

SessionRestoreResult SimGpuMemoryLedger::restore_session_resident(int sid) {
    if (auto dit = dropped_recompute_kv_mib_.find(sid); dit != dropped_recompute_kv_mib_.end()) {
        const double recomp_val = dit->second;
        dropped_recompute_kv_mib_.erase(dit);
        if (!upsert_session_kv(sid, recomp_val)) {
            dropped_recompute_kv_mib_[sid] = recomp_val;
            return SessionRestoreResult{/*ok=*/false, 0.0, 0.0};
        }
        ++kv_recompute_restores_;
        touch_session(sid);
        return SessionRestoreResult{/*ok=*/true, 0.0, recomp_val};
    }
    auto it = swapped_session_kv_mib_.find(sid);
    if (it == swapped_session_kv_mib_.end()) {
        if (session_kv_mib_.count(sid)) {
            touch_session(sid);
        }
        return SessionRestoreResult{/*ok=*/true, 0.0, 0.0};
    }
    const double swapped_val = it->second;
    swapped_session_kv_mib_.erase(it);
    if (!upsert_session_kv(sid, swapped_val)) {
        swapped_session_kv_mib_[sid] = swapped_val;
        return SessionRestoreResult{/*ok=*/false, 0.0, 0.0};
    }
    ++kv_host_swap_ins_;
    touch_session(sid);
    return SessionRestoreResult{/*ok=*/true, swapped_val, 0.0};
}

double SimGpuMemoryLedger::ensure_session_resident(int sid) {
    const SessionRestoreResult res = restore_session_resident(sid);
    if (!res.ok) {
        return -1.0;
    }
    return res.swapped_in_mib;
}

bool SimGpuMemoryLedger::upsert_session_kv(int sid, double logical_kv_mib) {
    std::optional<double> prev;
    if (auto it = session_kv_mib_.find(sid); it != session_kv_mib_.end()) {
        prev = it->second;
    }
    std::optional<double> prev_swapped;
    if (auto sit = swapped_session_kv_mib_.find(sid); sit != swapped_session_kv_mib_.end()) {
        prev_swapped = sit->second;
        swapped_session_kv_mib_.erase(sit);
    }
    session_kv_mib_[sid] = logical_kv_mib;
    double kv_tot = active_kv_mib();
    double used = 0.0;

    if (policy_.use_tripool_vmm && policy_.use_expert_pager) {
        const double super_gran_mib = std::max(16.0, static_cast<double>(policy_.super_handle_mib));
        const double full_exp_mib = static_cast<double>(num_expert_layers_ * layer_stride_mib_);
        // Reserve physical headroom for `1024x1024` DiT + Scoped-VAE spikes (`4*dit_512 + scoped_vae_1024 + 2 GiB`);
        // when `allow_vmm_overflow` still has free driver HBM above that (e.g. 80 GiB H100), allocate 2 MiB overflow
        // handles before carving/evicting resident experts.
        const double transient_headroom_mib =
            (4.0 * physics_.dit_512_transient_mib * std::sqrt(weight_scale_)) +
            (physics_.scoped_vae_1024_spike_mib * vae_spike_scale_) + 2048.0;
        while (kv_tot > cur_fine_pool_mib_ && cur_super_pool_mib_ >= super_gran_mib) {
            const bool would_evict_experts = (cur_super_pool_mib_ - super_gran_mib) < full_exp_mib;
            const double projected_phys_mib =
                base_weights_mib() + std::max(cur_super_pool_mib_, full_exp_mib) + kv_tot + transient_headroom_mib;
            if (would_evict_experts && policy_.allow_vmm_overflow && (projected_phys_mib <= total_hbm_mib_)) {
                break;
            }
            cur_super_pool_mib_ -= super_gran_mib;
            cur_fine_pool_mib_ += super_gran_mib;
            ++super_to_fine_carves_;
        }
        const double pool_cap = cur_super_pool_mib_ + cur_fine_pool_mib_;
        while (resident_expert_layers_ > 0 &&
               (static_cast<double>(resident_expert_layers_ * layer_stride_mib_) + kv_tot) > pool_cap) {
            const double projected_phys_mib =
                base_weights_mib() + static_cast<double>(resident_expert_layers_ * layer_stride_mib_) +
                kv_tot + transient_headroom_mib;
            if (policy_.allow_vmm_overflow && (projected_phys_mib <= total_hbm_mib_)) {
                break;
            }
            --resident_expert_layers_;
            ++expert_evictions_;
            post_evict_ = true;
        }
        used = base_weights_mib() + static_cast<double>(resident_expert_layers_ * layer_stride_mib_) + kv_tot;
        if (used > total_hbm_mib_) {
            evict_idle_kv_to_host(used - total_hbm_mib_, sid);
            kv_tot = active_kv_mib();
            used = base_weights_mib() + static_cast<double>(resident_expert_layers_ * layer_stride_mib_) + kv_tot;
        }
    } else {
        // `prism_tetriserve_coloc` (`kv_vmm_only`) pre-reserves `fine_kv_pool_mib + super_pool_mib` physical VMM handles.
        auto compute_used = [&]() {
            const double kt = active_kv_mib();
            const double phys_kv =
                policy_.kv_vmm_only
                    ? std::max(kt, static_cast<double>(policy_.fine_kv_pool_mib + policy_.super_pool_mib))
                    : kt;
            return base_weights_mib() + phys_kv;
        };
        used = compute_used();
        if (used > total_hbm_mib_ && policy_.use_idle_kv_swap) {
            evict_idle_kv_to_host(used - total_hbm_mib_, sid);
            used = compute_used();
        }
    }

    if (used > total_hbm_mib_) {
        if (prev.has_value()) {
            session_kv_mib_[sid] = *prev;
        } else {
            session_kv_mib_.erase(sid);
        }
        if (prev_swapped.has_value()) {
            swapped_session_kv_mib_[sid] = *prev_swapped;
        }
        return false;
    }
    touch_session(sid);
    peak_used_mib_ = std::max(peak_used_mib_, used);
    record_hbm_sample(used);
    return true;
}

bool SimGpuMemoryLedger::check_transient_spike(
    double spike_mib,
    std::optional<int> active_sid,
    bool contiguous_alloc,
    double retain_unscoped_frag_mib) {
    if (spike_mib <= 0.0) return true;
    double kv_tot = active_kv_mib();
    if (policy_.use_tripool_vmm && policy_.use_expert_pager) {
        double used = base_weights_mib() + static_cast<double>(resident_expert_layers_ * layer_stride_mib_) +
                      kv_tot + spike_mib;
        if (used > total_hbm_mib_ && active_sid.has_value()) {
            evict_idle_kv_to_host(used - total_hbm_mib_, active_sid);
            kv_tot = active_kv_mib();
            used = base_weights_mib() + static_cast<double>(resident_expert_layers_ * layer_stride_mib_) +
                   kv_tot + spike_mib;
        }
        if (used > total_hbm_mib_) return false;
        peak_used_mib_ = std::max(peak_used_mib_, used);
        return true;
    }
    const double gen_w = policy_.use_naive_offload ? pageable_expert_mib() : 0.0;
    const double held_frag = (policy_.use_scoped_vae || policy_.use_naive_offload) ? 0.0 : retained_unscoped_vae_mib_;
    const double eff_spike = (policy_.use_naive_offload && !contiguous_alloc && retain_unscoped_frag_mib > 0.0)
                                 ? std::max(0.0, spike_mib - retain_unscoped_frag_mib)
                                 : spike_mib;
    const double eff_transient = contiguous_alloc ? (eff_spike + held_frag) : std::max(eff_spike, held_frag);
    auto compute_peak = [&]() {
        const double kt = active_kv_mib();
        const double phys_kv =
            policy_.kv_vmm_only
                ? std::max(kt, static_cast<double>(policy_.fine_kv_pool_mib + policy_.super_pool_mib))
                : kt;
        return base_weights_mib() + gen_w + phys_kv + eff_transient;
    };
    double peak = compute_peak();
    if (peak > total_hbm_mib_ && policy_.use_idle_kv_swap && active_sid.has_value()) {
        evict_idle_kv_to_host(peak - total_hbm_mib_, active_sid);
        peak = compute_peak();
    }
    if (peak > total_hbm_mib_) {
        retained_unscoped_vae_mib_ = 0.0;
        peak_used_mib_ = std::max(peak_used_mib_, total_hbm_mib_);
        return false;
    }
    if (!policy_.use_scoped_vae && !policy_.use_naive_offload && retain_unscoped_frag_mib > 0.0) {
        retained_unscoped_vae_mib_ = std::max(retained_unscoped_vae_mib_, retain_unscoped_frag_mib);
    }
    peak_used_mib_ = std::max(peak_used_mib_, peak);
    return true;
}

std::pair<bool, double> SimGpuMemoryLedger::execute_dit_and_vae(
    int height,
    std::optional<int> active_sid,
    std::optional<double> post_denoise_sid_kv_mib,
    double unpack_staging_mib,
    bool run_pre_denoise_vae) {
    const bool is_1024 = height >= 1024;
    // Spatial sequence length scales as (1024/512)^2 = 4.0x (`64x64=4096` vs `32x32=1024` latent tokens).
    const double dit_act_mib =
        (is_1024 ? (4.0 * physics_.dit_512_transient_mib) : physics_.dit_512_transient_mib) *
        std::sqrt(weight_scale_);
    const double scoped_vae_mib =
        (is_1024 ? physics_.scoped_vae_1024_spike_mib : physics_.scoped_vae_512_spike_mib) * vae_spike_scale_;
    const double unscoped_vae_mib =
        (is_1024 ? physics_.unscoped_vae_1024_reserved_mib : physics_.unscoped_vae_512_reserved_mib) *
        vae_spike_scale_;
    const double vae_frag_slack_mib = std::max(0.0, unscoped_vae_mib - scoped_vae_mib);
    double kv_tot = active_kv_mib();
    double swapped_out_mib = 0.0;

    if (policy_.use_tripool_vmm && policy_.use_expert_pager) {
        double step_peak =
            base_weights_mib() + static_cast<double>(num_expert_layers_ * layer_stride_mib_) + kv_tot + dit_act_mib;
        if (step_peak > total_hbm_mib_ && active_sid.has_value()) {
            swapped_out_mib += evict_idle_kv_to_host(step_peak - total_hbm_mib_, active_sid);
            kv_tot = active_kv_mib();
            step_peak = base_weights_mib() + static_cast<double>(num_expert_layers_ * layer_stride_mib_) +
                        kv_tot + dit_act_mib;
        }
        if (step_peak > total_hbm_mib_) {
            last_fail_dit_frac_ = 0.12;
            return {false, swapped_out_mib};
        }
        resident_expert_layers_ = num_expert_layers_;
        post_evict_ = false;

        if (active_sid.has_value()) {
            touch_session(*active_sid);
            if (post_denoise_sid_kv_mib.has_value()) {
                session_kv_mib_[*active_sid] = *post_denoise_sid_kv_mib;
                kv_tot = active_kv_mib();
            }
        }

        const double vae_mib = policy_.use_scoped_vae ? scoped_vae_mib : unscoped_vae_mib;
        const double kv_in_fine = std::min(kv_tot, cur_fine_pool_mib_);
        const double kv_in_super = std::max(0.0, kv_tot - cur_fine_pool_mib_);
        const double super_cap = std::min(
            cur_super_pool_mib_,
            std::max(0.0, total_hbm_mib_ - base_weights_mib() - kv_in_fine));
        while (resident_expert_layers_ > 0 &&
               (static_cast<double>(resident_expert_layers_ * layer_stride_mib_) + kv_in_super + vae_mib) > super_cap) {
            --resident_expert_layers_;
            ++expert_evictions_;
            post_evict_ = true;
        }

        double vae_peak = base_weights_mib() +
                          static_cast<double>(resident_expert_layers_ * layer_stride_mib_) + kv_tot + vae_mib;
        if (vae_peak > total_hbm_mib_ && active_sid.has_value()) {
            swapped_out_mib += evict_idle_kv_to_host(vae_peak - total_hbm_mib_, active_sid);
            kv_tot = active_kv_mib();
            vae_peak = base_weights_mib() +
                       static_cast<double>(resident_expert_layers_ * layer_stride_mib_) + kv_tot + vae_mib;
        }
        if (vae_peak > total_hbm_mib_) {
            last_fail_dit_frac_ = 1.00;
            return {false, swapped_out_mib};
        }
        peak_used_mib_ = std::max({peak_used_mib_, step_peak, vae_peak});
        record_hbm_sample(step_peak);
        record_hbm_sample(vae_peak);
        return {true, swapped_out_mib};
    }

    const double gen_weights_mib = policy_.use_naive_offload ? pageable_expert_mib() : 0.0;
    const double vae_mib = (policy_.use_scoped_vae || policy_.use_naive_offload) ? scoped_vae_mib : unscoped_vae_mib;
    double held_frag_mib = (policy_.use_scoped_vae || policy_.use_naive_offload) ? 0.0 : retained_unscoped_vae_mib_;
    auto eff_kv_mib = [&](double logical_sum) -> double {
        return policy_.kv_vmm_only
                   ? std::max(logical_sum, static_cast<double>(policy_.fine_kv_pool_mib + policy_.super_pool_mib))
                   : logical_sum;
    };

    // 1. `RECV_SHM` unpack staging (`torch.frombuffer(raw).to(device)` + `unpack_state` coexist before `del flat`).
    if (unpack_staging_mib > 0.0 && active_sid.has_value()) {
        const double cur_sid_kv = session_kv_mib_.count(*active_sid) ? session_kv_mib_.at(*active_sid) : 0.0;
        const double others_kv = std::max(0.0, kv_tot - cur_sid_kv);
        const double unpack_peak = base_weights_mib() + gen_weights_mib + others_kv + 2.0 * unpack_staging_mib + held_frag_mib;
        if (unpack_peak > total_hbm_mib_) {
            retained_unscoped_vae_mib_ = 0.0;
            last_fail_dit_frac_ = 0.02;
            peak_used_mib_ = std::max(peak_used_mib_, total_hbm_mib_);
            return {false, 0.0};
        }
        peak_used_mib_ = std::max(peak_used_mib_, unpack_peak);
    }

    // 1b. Deferred `pending_vae` (`static_nccl_fission` / `twostage_shm` `gen` rank) runs `vae.encode` before denoising.
    if (run_pre_denoise_vae) {
        double enc_peak = base_weights_mib() + gen_weights_mib + eff_kv_mib(kv_tot) + std::max(vae_mib, held_frag_mib);
        if (enc_peak > total_hbm_mib_ && policy_.use_idle_kv_swap && active_sid.has_value()) {
            swapped_out_mib += evict_idle_kv_to_host(enc_peak - total_hbm_mib_, active_sid);
            kv_tot = active_kv_mib();
            enc_peak = base_weights_mib() + gen_weights_mib + eff_kv_mib(kv_tot) + std::max(vae_mib, held_frag_mib);
        }
        if (enc_peak > total_hbm_mib_) {
            retained_unscoped_vae_mib_ = 0.0;
            last_fail_dit_frac_ = 0.04;
            peak_used_mib_ = std::max(peak_used_mib_, total_hbm_mib_);
            return {false, swapped_out_mib};
        }
        if (!policy_.use_scoped_vae && !policy_.use_naive_offload) {
            retained_unscoped_vae_mib_ = std::max(retained_unscoped_vae_mib_, vae_frag_slack_mib);
            held_frag_mib = retained_unscoped_vae_mib_;
        }
        peak_used_mib_ = std::max(peak_used_mib_, enc_peak);
    }

    // 2. Denoising (`generate_image`): `dit_phys_kv` (`main_ctx + cfg_text_ctx`) + `dit_act_mib` + non-reusable Conv2d slack.
    double step_peak = base_weights_mib() + gen_weights_mib + eff_kv_mib(kv_tot) + dit_act_mib + held_frag_mib;
    if (step_peak > total_hbm_mib_ && policy_.use_idle_kv_swap && active_sid.has_value()) {
        swapped_out_mib += evict_idle_kv_to_host(step_peak - total_hbm_mib_, active_sid);
        kv_tot = active_kv_mib();
        step_peak = base_weights_mib() + gen_weights_mib + eff_kv_mib(kv_tot) + dit_act_mib + held_frag_mib;
    }
    if (step_peak > total_hbm_mib_) {
        retained_unscoped_vae_mib_ = 0.0;
        last_fail_dit_frac_ = 0.12;
        peak_used_mib_ = std::max(peak_used_mib_, total_hbm_mib_);
        return {false, swapped_out_mib};
    }

    // 3. Release `cfg_text_ctx` before `vae.decode` (symmetric VAE accounting with zero asymmetric DiT penalty).
    if (active_sid.has_value() && post_denoise_sid_kv_mib.has_value()) {
        session_kv_mib_[*active_sid] = *post_denoise_sid_kv_mib;
        kv_tot = active_kv_mib();
    }
    double vae_peak = base_weights_mib() + gen_weights_mib + eff_kv_mib(kv_tot) + vae_mib;
    if (vae_peak > total_hbm_mib_ && policy_.use_idle_kv_swap && active_sid.has_value()) {
        swapped_out_mib += evict_idle_kv_to_host(vae_peak - total_hbm_mib_, active_sid);
        kv_tot = active_kv_mib();
        vae_peak = base_weights_mib() + gen_weights_mib + eff_kv_mib(kv_tot) + vae_mib;
    }
    if (vae_peak > total_hbm_mib_) {
        retained_unscoped_vae_mib_ = 0.0;
        last_fail_dit_frac_ = 1.00;
        peak_used_mib_ = std::max(peak_used_mib_, total_hbm_mib_);
        return {false, swapped_out_mib};
    }
    if (!policy_.use_scoped_vae && !policy_.use_naive_offload) {
        retained_unscoped_vae_mib_ = std::max(retained_unscoped_vae_mib_, vae_frag_slack_mib);
    }
    peak_used_mib_ = std::max({peak_used_mib_, step_peak, vae_peak});
    record_hbm_sample(step_peak);
    record_hbm_sample(vae_peak);
    return {true, swapped_out_mib};
}

void SimGpuMemoryLedger::release_session(int sid) {
    session_kv_mib_.erase(sid);
    swapped_session_kv_mib_.erase(sid);
    dropped_recompute_kv_mib_.erase(sid);
    session_last_touch_.erase(sid);
    if (!policy_.use_tripool_vmm && session_kv_mib_.empty()) {
        retained_unscoped_vae_mib_ = 0.0;
    }
}

nlohmann::json SimGpuMemoryLedger::stats_json() const {
    return {
        {"gpu_id", gpu_id_},
        {"resident_expert_layers", resident_expert_layers_},
        {"expert_alpha", expert_alpha()},
        {"post_evict", post_evict_},
        {"active_kv_mib", active_kv_mib()},
        {"free_kv_bytes", free_kv_bytes()},
        {"super_to_fine_carves", super_to_fine_carves_},
        {"expert_evictions", expert_evictions_},
        {"kv_host_swap_outs", kv_host_swap_outs_},
        {"kv_host_swap_ins", kv_host_swap_ins_},
        {"kv_recompute_evictions", kv_recompute_evictions_},
        {"kv_recompute_restores", kv_recompute_restores_},
        {"peak_used_mib", peak_used_mib_},
        {"effective_hbm_util_mean_pct", round_2(effective_hbm_util_mean_pct())},
    };
}

}  // namespace morphserve::control
