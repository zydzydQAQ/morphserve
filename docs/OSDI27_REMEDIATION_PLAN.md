# MorphServe OSDI '27 Audit & Remediation Plan

> **Status**: Locked via full 8-branch `/grill-me` design review on `2026-11-22` (awaiting user execution sign-off before code changes or Kairos job submissions).
> **Cluster Guardrails**: Zero container image builds (`192.168.1.114:30500/jjjkairos/mllmserve:vllm-omni0.30.0-r2` + `deploy_build.sh --mode stage`); `/data` currently at `752G / 74%` (strictly enforce `< 80%` deploy gate and `810G / 82%` hard stop); `user-jjjkairos` quota = `16× NVIDIA A100-PCIE-40GB`.

---

## 1. Ground-Truth Audit of Existing Experiments

| Stage / Suite | Run Directory & Commit | Status | Reusable Role | Technical Verdict |
| :--- | :--- | :---: | :--- | :--- |
| **Phase-0 Hardware & Motivation** | `results/phase0-job0a-v2` (`df6a266`), `phase0-job0c` (`df6a266`), `phase0-job0d-r2` (`3d0f60c`) | **VALID** | **Paper §2 Motivation & §4 Design** | Verified `2 MiB` vs `64/128 MiB` VMM `ioctl` amplification (`193.9 ms -> 2.8–5.4 ms`), dual PCIe half-node `~10.35 GB/s` root bottleneck, `BAGEL-7B-MoT` memory/VAE spike anatomy (`39.65/40 GiB`), and stock `vLLM-Omni` 2-stage `2.61–2.87 s/turn` KV transfer stall. *(Earlier `job0a-r2`, `job0d` superseded.)* |
| **Module 1 (`TriPool-VMM`)** | `results/phase1-job1a-r3` (`d5b854a`) | **VALID** | **Paper §6.5 Microbenchmarks** | `99.22%` `64 MiB` layer packing, `17.4×` batched `cuMemSetAccess` (`202.5 -> 11.65 ms`), `89.7%–100%` Step-1 H2D compute overlap, strict `14 GiB` zero-OOM `1024² b=2` VAE absorption (`1.33×` faster than PyTorch allocator), `100%` SHA256 bit-exact. *(Earlier `job1a`, `job1a-r2` superseded.)* |
| **Module 2 (`DualPathMorphRouter`)** | `results/phase1-job1d-r4` (`16c21dd`) | **VALID** | **Paper §6.5 Microbenchmarks** | Shared dual-socket NUMA `memfd` mirror (`98.0 -> 24.5 GiB` Host Shmem, `4.0×` reduction), `HalfNodeH2DPacer` (`3.10 -> 12.34 GB/s`, `3.98×`), and live `BAGEL` Path-A (`0.324 ms` short KV) vs Path-B (`214.9 < 329.1 ms` Turn-3 KV) crossover. *(`job1d-r1..r3` superseded.)* |
| **Module 3 (`FlashPrefix-CFG` + CoW + Loopback)** | `results/phase1-job1b-r7` (`bb92fec`) + `morphserve-p2-m21gate-20261121-r4` (`b1aaa84`) | **VALID** | **Paper §6.5 Microbenchmarks** | `-49.9%` multi-turn KV HBM via LECP CoW, `28.5×–29.4×` GPU in-place loopback prep (`100%` greedy AR match), `1.11×–1.29×` `512²` CFG attention speedup and `1.02×` gated `1024²` parity. *(`job1b-r1..r6` superseded.)* |
| **Module 4 (`Layer-Chunk Preemption`)** | `results/phase1-job1c-r3` (`bb92fec`) | **VALID** | **Paper §6.5 Microbenchmarks** | Exploits `2.17%` DiT step CV; cuts mid-DiT AR `TTFT Max` from `4,619 ms` (FCFS) / `350.7 ms` (step-preemptive, `15.8%` `2.0×` SLO) to `88.2–116.6 ms` (`94.7%–100%` SLO, `< 1%` overhead, bit-exact). *(`job1c-r1..r2` superseded.)* |
| **Multi-Node Fabric Probe (`2×8`)** | `results/morphserve-p2-m25node2x8-20261121-r2` (`b16a50a`) | **VALID (Micro)** | **Paper §6.4 Fabric Calibration** | Live 16-GPU bandwidth calibration (`8.37 GB/s` intra-half, `8.59 GB/s` cross-socket, `1.11 GB/s` inter-node) + analytical cross-node 1.5 GiB KV suppression. Needs live `2×8` E2E serving run alongside it. |
| **Pre-`076e8c1` Phase-2 Cluster Runs** | `m22smoke-r4`, `m23a-20261121-r2`, `m23b-20261121-r2/r4`, `w3deep-20261121-r2` | **EXPIRED** | **Do Not Cite** | Executed prior to goodput formula fix, post-VAE lazy residency, HBM KV parking, and out-of-order ACK fixes (`e27bce5`..`076e8c1`). |
| **Latest Phase-2 Cluster Runs (`0805fc1` / `076e8c1`)** | `m23a-20261122-r3`, `m23b-20261122-r6`, `m24ivlu-20261122-r6`, remote `w3deep-20261122-r3` | **DIAGNOSTIC ONLY** | **Internal Root-Cause Baseline** | Proved `morphserve_elastic` beats all static fission ratios (`2:6 / 4:4 / 6:2` across `80:20 / 50:50 / 20:80` skews: `0.875 / 0.708 / 0.292`), **but exposed fatal E2E inversions against `monolithic_dp` (`0.330` vs `0.468` goodput on `n=80` at `1.1` QPS; `1.415` vs `2.233` rps on `InternVL-U`), `0` M4 yields on `n=28/24`, and flat ablations**. Must be superseded after remediation. |
| **64-GPU Discrete-Event Simulator** | `results/phase2-m25-sim` | **INVALID** | **Must Replace** | In-sample grid search on single `0.8` QPS point (`11.12%` error on `morphserve`), hardcoded `sid % 2` OOM triggers, and non-decaying `gpu_alpha = 1.0` distorting NVLink Path-A/B behavior. |

---

## 2. OSDI '27 Reviewer Blocking Weaknesses -> Remediation Mapping

| Reviewer Issue | Root Cause Identified in Code & Logs | Locked Design Fix (`D1–D8`) |
| :--- | :--- | :--- |
| **W1: E2E Goodput Inversion vs. `monolithic_dp` & Residual OOM** | (1) `dit_turn_slo_ms = 15,000 ms` is physically impossible for `1024²` (`~20.8 s` solo), rewarding fast-OOM crashes over completion; (2) per-GPU `asyncio.Lock` (`batch=1`) + `0.25 s` inter-turn sleep + cold `alpha=0.50` start on even GPUs wasted `~16 GiB` HBM while paying H2D stalls. | **D1 + D3 + D4**: Warm `alpha=1.0` start on all GPUs, `3–8 s` multi-turn KV co-residency, multi-request AR continuous batching, `k × T_unloaded` per-class SLOs, and 1 bounded OOM retry. |
| **W2: Artificial Handicapping of `prism_tetriserve_coloc`** | `_alloc_kv_vmm_if_needed` in `worker_proc.py` allocated unused `2 MiB` VMM shadow pages on top of PyTorch `NaiveCache` + `2.5 GiB` static reserve, doubling KV HBM pressure. | **D2**: Delete shadow-page code completely; both `morphserve` and `prism_tetriserve_coloc` share the exact same real `2 MiB` contiguous-VA KV-VMM allocator. |
| **W3: `TriPool-VMM` Arena 1 (KV) Unconnected in E2E** | Phase-2 workers left `NaiveCache` on PyTorch's default allocator (`torch.cat` copies + synchronous `Tensor.to("cpu")` parking) instead of `TriPool-VMM`'s `2 MiB` KV arena. | **D2**: Add `vAttention`-style contiguous-VA `2 MiB` region growth (`tripool_kv_region_*`) in `csrc/vmm_tripool` backing `NaiveCache` K/V tensors in-place, triggering real `super_to_fine_carves` and `expert_auto_evictions_for_kv`. |
| **W4: Flat Ablations & Dormant M4 (`0` Preemptions)** | `n=28/24` subsampling ran `< 2.5 min` dominated by cold ramp-up/drain with only 7 VQA sessions across 8 GPUs, leaving idle GPUs available almost always. | **D5**: Full `n = 80–120` session runs, steady-state middle-window telemetry (trimming first/last `15%`), dense BurstGPT AR arrivals under DiT saturation, and multi-metric ablation table. |
| **W5: Single-Turn Accuracy Suite & Unscored Task Quality** | `accuracy_geneval_mmvet.json` used Turn-0 (`L_shared = 0`) only — never invoking `FlashPrefix-CFG` or `GpuInPlaceLoopback` — and checked pixel/token parity without task evaluators. | **D6**: Evaluate Visual-CoT GenEval (`L_shared > 0`), 4-turn MagicBrush/MSE editing (CFG + GPU loopback), and multi-turn MMVet using in-image `transformers` evaluators (`OWLv2`/`Grounding-DINO`, `CLIP-T`/`DINO`, `MMVet` scorer) + `SSIM/SHA256`. |
| **W6: Overfitted & Hardcoded 64-GPU Simulator** | Single-point grid search (`405` combos), `sid % 2` OOM branches, and monotonic `alpha -> 1.0` preventing NVLink Path-A shift. | **D7**: Run live `2×8` (`16-GPU`) Kairos E2E benchmark + rewrite simulator around Phase-0/1 operator profiles and a per-GPU byte-level physical HBM ledger with zero-tuning out-of-sample validation across 8-GPU and 16-GPU runs. |

---

## 3. Locked Technical Specifications (`D1–D8`)

### D1. Stateful Multi-Session Residency, Demand-Driven `alpha`, and Batched AR Worker
1. **Warm `alpha = 1.0` Startup**: All GPU workers initialize with `initial_alpha = 1.0` (`28/28` BAGEL generation blocks or `20/20` InternVL-U decoder blocks resident), eliminating artificial cold-start H2D stalls at low QPS or AR-heavy skew.
2. **Realistic Multi-Turn Session Residency (`3–8 s` Think Time)**: Multi-turn editing and multi-turn VQA sessions hold their live KV context in GPU memory across realistic user think-time intervals (`3.0–8.0 s` deterministic distribution), creating genuine multi-session KV co-residency (`6–18 GiB` concurrent KV per GPU) under moderate-to-high QPS.
3. **Demand-Driven `alpha` Differentiation**: Tail expert layers are evicted (`alpha: 1.0 -> 0.75 -> 0.50 -> 0.0`) **strictly on demand** when Arena-1 (`2 MiB` KV growth) or Arena-2 (`64 MiB` scoped FLUX VAE transient spike) requests physical handles beyond free pool capacity. Once workers organically diverge in `expert_alpha` and queue backlog, `DualPathMorphRouter` evaluates `Path A (state_ship)` vs. `Path B (in_place_morph)` on real physical state.
4. **Multi-Request Batched AR Decoding**:
   - Replace the exclusive single-AR per-GPU serialization with batched causal decoding (`packed_text_ids` / `query_lens` varlen batching across `B >= 1` active AR/CoT tasks in `BagelPipeline.bagel.language_model(..., mode="und")` + batched `UnifiedPreemptiveScheduler` step execution during M4 layer-chunk yields).
   - Remove per-token unnecessary host-blocking overhead where batching consolidates token steps, recording exact per-request TTFT and TPOT timestamps.

### D2. Real 3-Arena `TriPool-VMM` (`2 MiB` Contiguous-VA KV) & 100% Fair `prism_tetriserve_coloc`
1. **C++ `csrc/vmm_tripool` Contiguous-VA Region Extension (~60 LOC)**:
   - Add `tripool_kv_region_alloc(size_t max_va_bytes) -> int32_t region_id`, `tripool_kv_region_ensure(int32_t region_id, size_t needed_bytes, void* stream) -> uint64_t va_ptr`, `tripool_kv_region_shrink_or_free(int32_t region_id, void* stream)`.
   - Each `NaiveCache` layer's `K` and `V` tensor binds to a virtual slice (`max_seq_len` VA, `0 B` physical HBM initially) and maps `2 MiB` physical pages (`2,048` tokens/page for BAGEL `4×128 bf16`) in-place as `seq_len` grows — **zero `torch.cat` tensor reallocations, zero PyTorch allocator fragmentation**.
   - When `free_fine_handles_` is empty, `tripool_vmm` carves `64 MiB -> 32 × 2 MiB` (`super_to_fine_carves`) and automatically drop-clean evicts unpinned tail expert layers (`expert_auto_evictions_for_kv`); freed fine pages coalesce back or are reused across sessions.
2. **100% Apple-to-Apple `prism_tetriserve_coloc` Baseline**:
   - Delete `_alloc_kv_vmm_if_needed` shadow-page allocation completely.
   - `prism_tetriserve_coloc` uses the **exact same `tripool_kv_region_*` `2 MiB` contiguous-VA KV allocator** as `morphserve` (sizing its `2 MiB` KV physical pool to the full remaining HBM after static `28.15 GiB` model weights, ~`10.5 GiB`, with `allow_overflow=True` up to hardware HBM limit), plus honest `step_preemptive` AR scheduling.
   - Its only differences from `morphserve` are genuine architectural differences: (a) static un-pageable Generation Expert weights (`12.25 GiB` cannot be drop-clean borrowed by KV or VAE), (b) un-scoped VAE on default allocator, (c) no LECP CoW / dual-path routing, and (d) step-boundary (`~287 ms`) rather than layer-chunk (`~30–85 ms`) preemption.

### D3. Per-Class Normalized `k×` SLOs & Production OOM Semantics
1. **Unloaded Solo Latency Anchors (`T_unloaded`, measured on `1×A100-PCIE-40GB`)**:
   - Anchor every modality/resolution class to `SLO(k) = k × T_unloaded(class)`, sweeping `k ∈ {1.5, 2.0, 2.5, 3.0}` with **`k = 2.0×` as the primary headline SLO**:
     - `AR TPOT`: `k × T_unloaded_tpot` (calibrated from clean solo measurement on the unified worker);
     - `AR TTFT`: `k × T_unloaded_prefill_vqa` (with `1.5 s` interactive cap at `2.0×`);
     - `512² DiT Turn (20 steps)`: `T_unloaded ≈ 6.2 s` -> `2.0× SLO ≈ 12.5–14.0 s`;
     - `1024² DiT Turn (20 steps)`: `T_unloaded ≈ 21.0 s` -> `2.0× SLO ≈ 42.0 s` (eliminating the impossible `15 s` cliff).
2. **Bounded 1-Retry OOM Recovery (No Free Load-Shedding)**:
   - If a worker hits CUDA/VMM physical OOM during a turn, it flushes caches (`gc.collect()` + `torch.cuda.empty_cache()`), increments `worker_oom_count`, and the coordinator **re-queues the failed turn once (`max_oom_retries = 1`)** onto the least-loaded eligible worker while keeping the user's arrival clock running (`e2e_ms` includes wasted failed execution + retry queueing). If the retry also OOMs, the session fails (`status = "oom"`).
3. **Dual Headline Metrics**:
   - **Effective Joint SLO Goodput (`rps`)** across QPS and `k ∈ {1.5, 2.0, 2.5, 3.0}`;
   - **Maximum Sustainable QPS** achieving `>= 80%` and `>= 90%` steady-state Joint SLO Attainment with `0%` unrecovered session drops.

### D4. `InternVL-U` (`bridge_mllm_dit`) Unified Engine & Dual-Regime Evaluation
1. Drive `InternVL-U` (`8.16 GiB`) through the unified async coordinator/worker architecture (eliminating the sequential `while got_sr < 2` stall and 4-GPU VLM bottleneck in `validate_phase2_internvl_u.py`).
2. Evaluate two explicit regimes on `8×A100`:
   - **Regime A — Unconstrained `40 GB` HBM (Dynamic Burst Load-Balancing)**: All 8 GPUs start warm (`alpha = 1.0`) and serve VLM; under bursty/skewed arrivals, `DualPathMorphRouter` ships `~455 KiB` bridge states in `< 1 ms` via `DirectNcclMesh` to idle peers, matching `monolithic_dp` at low load and outperforming sticky `monolithic_dp` and `static_nccl_fission` (`2:6 / 4:4 / 6:2`) on tail latency & goodput under bursts.
   - **Regime B — Memory-Constrained `12 GiB` Per-GPU Quota (`cuda.set_per_process_memory_fraction` + bounded `TriPool-VMM`)**: Simulates commodity/sliced multi-tenant GPU budgets where `monolithic_dp` (`~9.8–13 GiB` under concurrent multi-image VLM + `1024²` MMDiT) OOMs, demonstrating zero-OOM `TriPool-VMM + Path A` superiority over `naive_host_offload`.

### D5. Steady-State Workload Scale (`n = 80–120`) & Multi-Metric Ablation Ladder
1. **Main & Skew Suites**: Run full `n = 80–120` sessions per QPS point (no `n=28` truncation). Compute steady-state SLO attainment and P50/P95/P99 latencies over the **steady-state core window** (excluding the first `15%` ramp-up and final `15%` drain sessions, while also archiving full-trace totals in JSON for complete transparency).
2. **Active Preemption Regime**: Pair BurstGPT interactive AR arrivals (`multi_image_vqa` + `visual_cot`) with sustained 8-GPU DiT occupancy so M4 layer-chunk preemption is genuinely exercised (`preempted_ar_count >> 0`, `total_yields >> 0`).
3. **5-Step Ablation Ladder (`Full -> -M4 -> -M3 -> -M2 -> -M1`)**: Run at the high-contention knee QPS (`n >= 80`), reporting Joint Goodput alongside each module's direct physical telemetry (`AR TTFT/TPOT P99`, `Peak KV HBM & Loopback ms`, `Transition Wait & Host Shmem GiB`, `OOM Retries & Max Sustainable QPS`).

### D6. Multi-Turn M3-Active Downstream Accuracy & Exactness Suite
1. **Path Coverage (`L_shared > 0` + Loopback Active)**:
   - `48` **Visual-CoT GenEval** sessions (`cot_then_dit`, `<think>` shared causal prefix triggers `FlashPrefix-CFG`);
   - `45` **4-Turn MagicBrush / MSE-Bench Editing** sessions (simultaneously exercising `LECP CoW`, `FlashPrefix-CFG`, and `GpuInPlaceLoopback` across 4 chained turns);
   - `48` **Multi-Turn MMVet + Post-Edit VQA** sessions.
2. **Real Downstream Task Scorers (Zero Image Build)**:
   - Stage lightweight evaluator weights (`< 1.5 GiB` total, verified via `hf-mirror.com` into `/tmp` or `/shared/data/MORPHSERVE/eval_models` after `/data` gate check) using pre-installed `transformers 5.14.1`:
     - **GenEval Object/Attribute Accuracy** via open-vocabulary detector (`google/owlv2-base-patch16-ensemble`);
     - **Multi-Turn Editing Alignment & Fidelity** via `CLIP` (`openai/clip-vit-large-patch14`: prompt-image `CLIP-T` & target-image `CLIP-I`) + `SSIM / PSNR / MAE_u8`;
     - **MMVet VQA Score** (official normalized keyword/equivalence rubric + greedy token exact-match rate).
   - Compare **Stock `vLLM-Omni` / `monolithic_dp`** vs. **Full `MorphServe`**, proving `|Delta Score| < 0.2%` (`0.0%` degradation) alongside `100%` SHA256 bit-exactness when M3's `FlashPrefix-CFG` split is toggled to lossless fallback.

### D7. Live `2×8` (`16-GPU`) Multi-Node E2E Benchmark & Byte-Accurate Out-of-Sample Simulator
1. **Live `2×8 = 16-GPU` Multi-Node Kairos Run**:
   - Launch a `2`-node × `8`-GPU (`16` GPUs total) `MorphServeClusterEngine` distributed job comparing `morphserve`, `static_nccl_fission`, and `monolithic_dp` on `Interleaved-Bench` (`n = 160` sessions), verifying real inter-node coordinator/NCCL transport, zero cross-node multi-GiB KV shipping, and near-linear `8 -> 16` GPU goodput scaling.
2. **Physics-Driven `64-GPU` Simulator (`src/morphserve/sim/cluster_simulator.py`)**:
   - **Zero Grid Search & Zero Hardcoded OOMs**: Parameterize directly from Phase-0/1 measured operator latencies (`ar_batch_ms(B)`, `dit_step_ms(res, seq_len)`, `vae_ms(res)`, `h2d_gbps`, `p2p_gbps`, `rdma_gbps`) and a **per-GPU byte-accurate physical HBM ledger** tracking base weights + mapped `64 MiB` expert super-handles + active `2 MiB` KV pages + scoped/un-scoped VAE transient spikes + 1-retry OOM recovery.
   - **Out-of-Sample Validation Gate**: Require `< 8%` MAPE across **all** live `8-GPU` QPS points, modality-skew points, and the live `16-GPU` (`2×8`) multi-node run **without per-policy or per-QPS tuning**.
   - **Honest Interconnect Physics (`PCIe Gen4` vs `A100-NVLink` vs `H100-NVLink + RDMA`)**: Accurately reflect intra-node NVLink shifting short/medium transitions toward `Path A` while inter-node RDMA retains `Path B`, and elasticity + 3-arena VMM + M4 preemption sustain MorphServe's advantage over static fission and static colocation.

### D8. Codebase Convergence & 3-Stage Gated Execution Plan

1. **Step 0 — Safe Git Convergence (Zero Data Loss)**:
   - Snapshot all uncommitted refactoring in `/mnt/e/000_code/zyd-2026/morphserve` onto backup branch `wip/modular-refactor-20261122`.
   - Sync remote `morphserve-p2-w3deep-20261122-r3/wave3-deep-report.json` locally for historical archival.
   - Reset `main` onto verified `phase2-fa3-oom-guard` (`fd69779` / `076e8c1`), selectively cherry-picking only `BagelExecutionInterceptor` and `ExactnessEvaluator` where they reduce complexity, keeping `pytest` 100% green at every commit.

2. **Stage 1 — Implementation & `1×1` A100 Verification Gate (`~15–20 min` GPU time)**:
   - Implement `tripool_kv_region_*` (`2 MiB` contiguous-VA `NaiveCache`), fair `prism_tetriserve_coloc`, batched AR worker + M4 batched yields, `alpha=1.0` warm start + `3–8 s` multi-turn residency, `k×` normalized SLOs + 1-retry OOM, and unified `InternVL-U` driver.
   - **Hard Gate (`Job R1`, `1×1` A100)**: Verify `0` memory leaks, `super_to_fine_carves > 0` & `expert_auto_evictions_for_kv > 0` under multi-session `NaiveCache` growth, batched AR speedup & `100%` greedy parity, `prism_tetriserve_coloc` zero shadow pages, and solo unloaded `T_unloaded` calibration table. **Stop and inspect before launching 8-GPU jobs.**

3. **Stage 2 — `1×8` A100 Core Evaluation Campaign (`2` concurrent `1×8` jobs = `16/16` GPUs)**:
   - **`Job R2a` (`1×8` A100)**: `BAGEL-7B-MoT` steady-state QPS sweep (`n=96`, 4 QPS points × 6 policies) + `k ∈ {1.5, 2.0, 2.5, 3.0}` SLO curves.
   - **`Job R2b` (`1×8` A100)**: Modality-skew sweep (`80:20`, `50:50`, `20:80` × 6 policies + `2:6 / 4:4 / 6:2` static splits, `n=80`) + 5-step multi-metric M1–M4 ablation ladder (`n=80`).
   - **`Job R2c` (`1×8` A100)**: `InternVL-U` dual-regime (`40 GB` burst load-balancing + `12 GiB` memory-capped survival) + M3-active downstream accuracy suite (`OWLv2` GenEval + `CLIP/DINO` 4-turn MagicBrush/MSE + `MMVet`).

4. **Stage 3 — `2×8` (`16-GPU`) Multi-Node Run, Out-of-Sample Simulator & Paper Synthesis**:
   - **`Job R3` (`2×8 = 16` A100)**: Live 2-node 16-GPU end-to-end serving comparison (`morphserve` vs `static_nccl_fission` vs `monolithic_dp`, `n=160`).
   - Run zero-tuning out-of-sample `cluster_simulator.py` validation (`8 + 16` real points -> `64-GPU` PCIe/NVLink/H100 sweep), archive all artifacts under `results/osdi27-v1/`, and publish `docs/OSDI27_EVALUATION_REPORT.md`.
