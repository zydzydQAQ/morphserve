#include "morphserve/control/budget_planner.hpp"
#include "morphserve/control/morph_controller.hpp"
#include "morphserve/control/policy.hpp"
#include "morphserve/control/preempt_edf.hpp"
#include "morphserve/control/router.hpp"
#include "morphserve/control/simulator.hpp"
#include "morphserve/control/topology.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>

#define MS_CHECK(cond)                                                                         \
    do {                                                                                       \
        if (!(cond)) {                                                                         \
            std::cerr << "CHECK FAILED at " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::_Exit(1);                                                                     \
        }                                                                                      \
    } while (0)

namespace {

void test_router_edf_and_budget_planner() {
    using namespace morphserve::control;

    const auto topo = ClusterTopologyProfile::a100_pcie_8gpu_default();
    MS_CHECK(topo.same_half(0, 1));
    MS_CHECK(!topo.same_half(0, 4));

    DualPathMorphRouter router(topo);
    router.init_workers(
        4,
        [](int g) { return (g == 1 || g == 3) ? "gen" : "hybrid"; },
        {{0, 0.5}, {1, 1.0}, {2, 0.5}, {3, 1.0}},
        {{0, 16 * kGiB}, {1, 600 * kMiB}, {2, 16 * kGiB}, {3, 16 * kGiB}});

    CPTGRequestSpec req1;
    req1.req_id = "r1";
    req1.src_gpu = 0;
    req1.architecture = "tight_mot";
    req1.kv_bytes = 400 * kMiB;
    req1.dit_step1_ms = 280.0;
    req1.dit_total_steps = 20;

    const auto res1 = router.plan_and_reserve_transition("dual_path", {}, {}, req1, 50.0);
    MS_CHECK(res1.first.path == "state_ship" && res1.first.dst_gpu == 1 && res1.second == "nccl_dual");
    MS_CHECK(router.get_worker_state(1).kv_free_bytes == 200 * kMiB);

    // Stale telemetry before unpack must preserve 400 MiB inflight reservation
    router.update_worker_telemetry(1, 1.0, 600 * kMiB, 5);
    MS_CHECK(router.get_worker_state(1).kv_free_bytes == 200 * kMiB);
    router.update_worker_telemetry(1, 0.25, 16 * kGiB, 4);  // stale seq 4 < 5 ignored!
    MS_CHECK(router.get_worker_state(1).kv_free_bytes == 200 * kMiB);

    // PreemptEdfController adaptive chunking & EDF slack coalescing (window = 20 + 8.5 + 4 = 32.5 ms)
    PreemptEdfController edf(28, 60.0, 4.0, 8.5, 20.0, 3);
    MS_CHECK(edf.compute_adaptive_chunk_layers(60.0) == 4);
    MS_CHECK(edf.compute_adaptive_chunk_layers(100.0) == 8);
    const double dls[4] = {100.032, 100.015, 100.50, 100.025};
    std::vector<size_t> due;
    std::vector<size_t> def;
    edf.select_edf_microbatch(100.00, dls, 4, /*force_all_urgent=*/false, due, def);
    MS_CHECK(due.size() == 3);
    MS_CHECK(due[0] == 1 && due[1] == 3 && due[2] == 0);
    MS_CHECK(def.size() == 1 && def[0] == 2);

    // WorkerBudgetPlanner 12 GiB Regime B conservation (VLM + Decoder co-paged in 6688 MiB VMM pool)
    const auto q_pair = WorkerBudgetPlanner::compute_internvlu_quota_overrides(
        "morphserve", 8, "/shared/data/MORPHSERVE/models/InternVL-U", 12.0, 40.0);
    MS_CHECK(q_pair.second.at("runtime_combined_budget_mib").get<double>() == 12288.0);
    const auto pol = PolicyConfig::from_preset("morphserve", 8, q_pair.first);
    const auto plan = WorkerBudgetPlanner::compute_internvlu_memory_plan(
        40960.0,
        35000.0,
        260.0,
        4864LL * kMiB,
        pol.super_pool_mib,
        pol.fine_kv_pool_mib,
        pol.super_handle_mib,
        pol.allow_vmm_overflow,
        pol.gpu_memory_fraction);
    MS_CHECK(plan.at("quota_constrained").get<bool>());
    MS_CHECK(plan.at("vmm_physical_pool_mib").get<int>() == 6688);
    MS_CHECK(plan.at("pytorch_cap_mib").get<double>() + plan.at("vmm_physical_pool_mib").get<double>() <= 12288.0);
}

void test_physics_ledger_and_simulator() {
    using namespace morphserve::control;

    const auto ph = ClusterPhysicsProfile::from_phase01_results("results");
    MS_CHECK(ph.provenance_files.size() == 7);

    const auto ms_pol = PolicyConfig::from_preset("morphserve", 8);
    const auto mono_pol = PolicyConfig::from_preset("monolithic_dp", 8);
    SimGpuMemoryLedger ms_ledger(0, 40960.0, ms_pol, "hybrid");
    SimGpuMemoryLedger mono_ledger(0, 40960.0, mono_pol, "hybrid");

    for (int sid = 0; sid < 4; ++sid) {
        MS_CHECK(ms_ledger.upsert_session_kv(sid, 1800.0));
        mono_ledger.upsert_session_kv(sid, 1800.0);
    }
    MS_CHECK(ms_ledger.execute_dit_and_vae(1024, 0).first);
    MS_CHECK(ms_ledger.expert_evictions() > 0);
    MS_CHECK(!mono_ledger.execute_dit_and_vae(1024, 0).first);

    HardwareInterconnectSpec hw_xnode{
        "unit_2node_2gpu", 40.0, 1.0, 10.0, 10.0, 10.0, 10.0, 5.0, 1};
    const auto r_nccl = ClusterSimulator::simulate_cluster_point(
        "static_nccl_fission",
        2,
        hw_xnode,
        0.5,
        1,
        2026,
        ph,
        WorkloadScaleSpec::bagel_7b_standard(),
        "",
        "data/interleaved-bench-v1",
        "data/bench_sources");
    MS_CHECK(r_nccl.at("oom_sessions").get<int>() == 0);
    MS_CHECK(r_nccl.at("path_a_ships").get<int>() == 4);
    MS_CHECK(r_nccl.at("cross_node_ships").get<int>() == 7);
}

}  // namespace

int main() {
    test_router_edf_and_budget_planner();
    test_physics_ledger_and_simulator();
    std::cout << "[PASS] test_control_and_sim\n";
    return 0;
}
