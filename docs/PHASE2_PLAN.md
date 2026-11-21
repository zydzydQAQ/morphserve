# MorphServe Phase-2 Architecture, Evaluation & Execution Plan

> Locked on 2026-11-21 after completing Phase-0 + Phase-1 (Modules 1–4 verified on `8× NVIDIA A100-PCIE-40GB`) and full design-tree alignment (`/grill-me`).

---

## 1. Current Verified Baseline (Phase-0 & Phase-1 Complete)

All four core modules of MorphServe have passed live hardware verification on `jjjkairos` (`8× NVIDIA A100-PCIE-40GB`, CUDA 13.0, Driver `570.133.20`, PyTorch `2.13.0+cu130`, `vLLM-Omni 0.30.0`, NCCL `2.29.7`) under the zero-image-build policy:

| Phase / Module | Verified Run & Commit | Key Ground-Truth Findings & Verified Gates | Report |
| --- | --- | --- | --- |
| **Phase-0: Hardware & Motivation** | `job0a-v2`, `job0c`, `job0d-r2` (`33f779a`) | • `BAGEL-7B-MoT` (`27.52 GiB`): `14.185 GiB` Und + `12.208 GiB` Gen (uniform `444.52 MiB/layer × 28`), `0.814 GiB` ViT, `0.313 GiB` FLUX VAE.<br>• Monolithic 1-GPU hits `39.65 / 40 GiB` on `1024² b=2` VAE (`+7.00 GiB` spike); `cfg_text` duplicates `99.7%–99.9%` of `main` KV; CPU PNG loopback costs `68.7–140.7 ms/turn`; `286.6 ms/step` DiT blocks AR (`62.82 ms` unloaded TPOT).<br>• Stock 2-GPU `vLLM-Omni` consumes `71.29 GiB` HBM and stalls `2.61–2.87 s/turn` (`654 MiB/turn` KV serialize+transfer via `SharedMemoryConnector`).<br>• 8-GPU PCIe forms two half-nodes sharing `~10.35 GB/s` H2D root each; cross-half 4-pair P2P drops `19.67 → 9.98 GB/s`; `/dev/shm` capped at `32.0 GiB`; `2 MiB` VMM handles incur `193.94 ms` `ioctl` overhead at `4 GiB` vs `2.84–5.44 ms` with `64/128 MiB` Super-Handles. | [PHASE0_REPORT.md](file:///mnt/e/000_code/zyd-2026/morphserve/docs/PHASE0_REPORT.md) |
| **Module 1: `TriPool-VMM`** | `job1a-r3` (`d5b854a`, `status: ok`) | • Dual-tier `2 MiB` KV + `64 MiB` Super-Handle (`7×64=448 MiB/layer`, `99.22%` packing efficiency), batched `cuMemSetAccess` (`202.51 → 11.65 ms`, `17.4×`).<br>• Unlocks `23.51 GiB` free HBM; Step-1 H2D compute masking reaches `89.7%` (`512²`, `alpha=0.75`, `27.17 ms` stall) and `100.0%` (`1024²`, `alpha=0.50`, `~0 ms` stall).<br>• In a strict `14 GiB` (`allow_overflow=False`) pool, absorbs a `11.94 GiB` `1024² b=2` FLUX VAE spike via `6.87 ms` drop-clean eviction of 24 tail layers (`1.33×` faster than stock PyTorch allocator: `379.21` vs `505.10 ms`, `0` OOM, 100% bit-exact SHA256, `100%` post-VAE reclamation). | [MODULE1_REPORT.md](file:///mnt/e/000_code/zyd-2026/morphserve/docs/MODULE1_REPORT.md) |
| **Module 2: Dual-Path Morphing Router** | `job1d-r4` (`16c21dd`, `status: ok`) | • Dual-socket `SharedNumaMemfdMirror` (`98.0 → 24.5 GiB` Host RAM, `4.0×` savings, bypassing `32 GiB` `/dev/shm` cap) + `HalfNodeH2DPacer` (`3.10 → 12.34 GB/s` per-GPU H2D, `3.98×` speedup).<br>• Crossover verified on `BAGEL-7B-MoT`: Turn-0 short KV (`0.98 MiB`, `alpha=0.50`) selects `state_ship` (`0.324 ms` vs `258.79 ms` morph); Turn-3 long interleaved (`1,302.77 MiB`, `gamma=2`, `seq=23,822`, `alpha=0.75`) selects `in_place_morph` (`214.92 ms` vs `329.08 ms` NCCL wire ship). Both paths 100% bit-exact SHA256.<br>• Mixed 8-request CPTG burst routes `InternVL-U` bridge & Turn-0 `BAGEL` via Path A and long/`1024px` `BAGEL` via Path B. | [MODULE2_REPORT.md](file:///mnt/e/000_code/zyd-2026/morphserve/docs/MODULE2_REPORT.md) |
| **Module 3: `FlashPrefix-CFG` + CoW + GPU Loopback** | `job1b-r7` (`bb92fec`, `status: ok`) | • LECP CoW cuts multi-turn KV HBM by `49.9%` (`1,953.11 → 977.59 MiB` at Turn 3; fork `4.18 → 0.18 ms`).<br>• GPU In-Place Loopback speeds prep `74.99 → 2.55 ms` (`29.4×`, `u8` bit-exact, `vit_cos = 0.9999982`, AR greedy text 100% exact match).<br>• `FlashPrefix-CFG` (`vllm_varlen` + Triton LSE merge) speeds `512px` attention `1.10×–1.28×` (`1.80×–2.68×` KV HBM read reduction; Turn-3 same-context `SSIM = 0.9964`, `PSNR = 44.27 dB`), while `1024px` (`Q=4,098, L_shared=8,192`) measured `0.95×` — motivating an arithmetic-intensity gate in Phase-2. | [MODULE3_REPORT.md](file:///mnt/e/000_code/zyd-2026/morphserve/docs/MODULE3_REPORT.md) |
| **Module 4: Layer-Chunk Preemption & M1+M3+M4 Stack** | `job1c-r3` (`bb92fec`, `status: ok`) | • Exploits low DiT step variance (`CV = 2.17%`) for analytical EDF layer-chunk yielding (`k = 2/5/8`).<br>• Mid-DiT AR `TTFT Max` drops `4,619.23 ms` (FCFS) / `350.65 ms` (Step-Preemptive, `15.8%` `2.0×` SLO attainment) to `88.18 ms` (`52.4×` vs FCFS, `100.0%` `1.5×` SLO attainment, `< 1%` switching overhead, 100% bit-exact AR tokens & DiT SHA256).<br>• Simultaneous M1+M3+M4 Turn-1 composition (`steps=12`, `alpha=0.50`, `SLO=125.6 ms`) passes all checks (`m4_ar_tpot_p99 = 117.03 ms`, `SSIM = 0.9936`, `overflow_creates = 0`). | [MODULE4_REPORT.md](file:///mnt/e/000_code/zyd-2026/morphserve/docs/MODULE4_REPORT.md) |

---

## 2. Target Venue & 2024–2026 Literature Positioning

- **Target Venue**: **OSDI / SOSP / ASPLOS** — *100% Lossless, Paradigm-Aware Serving for Native Unified Multimodal (Interleaved AR + DiT) Models*.
- **Primary Architectural Narrative**:
  1. `TriPool-VMM`: Dual-tier physical page pool unifying persistent KV (`2 MiB`), scoped transient VAE/DiT spikes (bulk-unmap `MemPool`), and drop-clean opportunistic secondary expert layers (`64 MiB` super-handles), solving driver `ioctl` amplification and HBM fragmentation without approximation.
  2. Topology-Aware Dual-Path Morphing: Shared dual-socket NUMA `memfd` mirrors + PCIe half-node H2D pacing + cost-aware switching between NCCL P2P state shipping and Step-1 in-place expert streaming.
  3. Deterministic Layer-Chunk Preemptive AR+DiT Scheduling: Sub-step (`~20–85 ms`) lossless preemption protecting AR TPOT/TTFT SLOs while executing DiT denoising on the same elastic GPU.
  4. Module 3 (`LECP CoW` + `FlashPrefix-CFG` + GPU In-Place Loopback): Positioned as the **-50% KV HBM capacity multiplier and zero-CPU-stall enabler** unlocking higher multi-turn concurrency under strict exactness.

### Verified 2024–2026 Related Work Comparison Matrix

| Research Axis | Published / Concurrent Works (2024–2026) | Gap Addressed by MorphServe |
| --- | --- | --- |
| **Multimodal / Any-to-Any Serving** | **Resonator** (*ISCA 2026*), **TriInfer** (*MLSys 2026*), **TiDAR** (*MLSys 2025/26*), **Eevee** (*EuroSys 2026*), **ModServe** (*ACM SoCC 2025*), **Cornserve** (*CAIS 2026 Demo / arXiv*), **M\*** (*arXiv 2026.06*, Walk Graph supporting BAGEL) | Resonator & TriInfer focus on `ViT→AR` understanding MLLMs (no 12+ GiB generation expert, no multi-step DiT blocking, no FLUX VAE spikes). Concurrent **M\*** operates at dataflow-graph placement granularity and does not provide unified `cuMemMap` physical memory sharing, PCIe dual-path morphing, or layer-chunk preemption. |
| **Diffusion / DiT Serving** | **DiFlow** (*SOSP 2026*), **TetriServe** (*ASPLOS 2026*), **ScaleFusion** & **DiffServe** (*MLSys 2025*), **EXION** (*HPCA 2025*), **MixFusion** (*PPoPP 2026*), **FlashPS** (*EuroSys 2026*) | Target diffusion-only or T2I/editing workflows. TetriServe/FlashPS schedule at denoising **step granularity** (~287 ms in BAGEL), causing severe AR TPOT SLO violations under colocation (`15.8%` attainment at `2.0×`). DiffServe/EXION/FlashPS also rely on lossy quality/cache approximations. |
| **GPU VMM (`cuMemMap`) & Weight Caching** | **Prism / `kvcached`** (*OSDI 2026*), **Janus / `xTensor`** (*SOSP 2026*), **vAttention** (*ASPLOS 2025*), **GMLake** (*ASPLOS 2024*), **BlitzScale** (*OSDI 2025*), **ServerlessLLM** (*OSDI 2024*), **ELORA** (*HPCA 2026*) | Prism and vAttention virtualize **KV Cache only**. MorphServe is the first to unify **persistent KV + scoped bulk-unmap transient scratchpads + drop-clean secondary expert weights** inside one dual-tier physical pool (`2 MiB` + `64 MiB`), eliminating `2 MiB` `ioctl` amplification under multi-GiB VAE spikes. |
| **Disaggregation vs. Colocation & Preemption** | **EcoServe** (*OSDI 2026*), **NanoFlow** (*OSDI 2025*), **POD-Attention** (*ASPLOS 2025*), **DistServe** & **Sarathi-Serve** (*OSDI 2024*), **Splitwise** (*ISCA 2024*), **WindServe** (*ISCA 2025*) | EcoServe validates that blind disaggregation collapses on commodity PCIe interconnects. MorphServe models the dynamic `KV size × γ` vs. `(1 - α)` expert morphing crossover across PCIe half-nodes and achieves `< 1%` overhead cross-paradigm layer-chunk preemption. |
| **Clean Separation from `DualFlow` (`mllmserve/`)** | Internal sibling project (`DualFlow`) | `DualFlow` targets multi-turn image-editing session routing, async host KV tiering, lazy commit, and **lossy** historical VAE/ViT compaction (`Plan B`). `MorphServe` targets **100% lossless** native unified interleaved AR+DiT serving via GPU VMM tri-arena sharing, dual-path morphing, CFG prefix CoW, and sub-step AR/DiT preemption. |

---

## 3. Locked Phase-2 Design Decisions

### D1. Five-Baseline Fair Comparison Matrix (Unified Operator Stack)
To prevent strawman comparisons (e.g., comparing only against slow `SharedMemoryConnector`), all cluster evaluations will compare MorphServe against five carefully implemented baselines sharing the same underlying `vllm-omni` BAGEL operators:
1. **`Stock vLLM-Omni Monolithic` (`8×1-GPU` DP)**: Native PyTorch caching allocator, monolithic model loading (`28.15 GiB`), `NaiveCache` deepcopy for CFG, CPU PNG/PIL loopback, non-preemptive execution.
2. **`Stock vLLM-Omni Two-Stage` (`4×2-GPU` Disaggregated)**: Official `SharedMemoryConnector` cross-stage KV serialization & transfer (`1×Und + 1×Gen` per pair).
3. **`Cornserve / M*-Style Static Fission + Direct NCCL P2P`**: Strong disaggregated baseline using MorphServe's fast `DirectNcclP2PChannel` (no serialization strawman), tuned to the best static `Und:Gen` GPU ratio per workload (e.g., `2:6` / `4:4`).
4. **`Prism / TetriServe-Style Colocation`**: Strong colocated baseline combining KV-only VMM paging + DiT step-boundary preemption, **without** unified Drop-Clean Expert / Bulk-Unmap Transient arenas, dual-path routing, or layer-chunk preemption.
5. **`Naive Host Offloading`**: Synchronous per-layer CPU↔GPU weight offloading without `TriPool-VMM` super-handles, shared NUMA `memfd` mirrors, or `HalfNodeH2DPacer`.

### D2. Cross-Architecture Model Generality (CPTG Dual-Model Real Execution)
- **Primary Model (`tight_mot`)**: `ByteDance-Seed/BAGEL-7B-MoT` (`27.52 GiB` already staged on `/shared/data/MORPHSERVE/models/BAGEL-7B-MoT`), exercised across text-only AR, `bagel_think` Visual-CoT (`<think>` -> DiT), multi-turn editing, and `512² / 1024²` T2I.
- **Secondary Model (`bridge_mllm_dit`)**: `InternVL-U` (`~10 GiB`, pending `/data` storage gate verification), integrated via a lightweight Transformers/Diffusers CPTG adapter + `TriPool-VMM` pager and evaluated on `8×A100` to prove the router's architectural generality: compact bridge tensors (`24–32 MiB`) prefer NCCL `state_ship`, whereas multi-GiB `tight_mot` KV contexts prefer `in_place_morph`.

### D3. `Interleaved-Bench` (`< 1 GiB`), Production Traces & Primary Metrics
- **4-Quadrant Workload Mix (`< 1 GiB` total on `/data`)**:
  - **30% Interleaved Visual-CoT** (`bagel_think`: AR `<think>` reasoning followed by image synthesis/editing);
  - **30% Multi-Turn Conversational Editing** (`mse-bench` + curated `MagicBrush` subset across `512²` and `1024²`);
  - **25% Interactive Multi-Image VQA** (`MMVet` / `ShareGPT4V` subset enforcing strict AR TTFT & TPOT SLOs);
  - **15% High-Resolution Text-to-Image** (`GenEval` prompts at `512²` and `1024²` stress-testing FLUX VAE transient spikes).
- **Production Arrival Traces**: Dual-trace replay combining **BurstGPT** (AR / Visual-CoT arrivals) and **GenTD26** (diffusion arrivals), sweeping offered QPS and dynamic modality-skew ratios (`80% AR : 20% Gen` ↔ `50:50` ↔ `20% AR : 80% Gen`).
- **Evaluation Metrics**:
  - **Primary**: **Joint SLO Goodput** (requests/turns per minute simultaneously meeting AR `TTFT`/`TPOT` and DiT Turn SLOs across `1.5× / 2.0× / 2.5×` curves);
  - **Latency**: P50/P95/P99 AR `TTFT`, AR `TPOT`, and DiT/Turn E2E latency;
  - **Memory & Efficiency**: Peak GPU HBM, Host `Shmem` footprint, Zero-OOM rate under `1024²` concurrent bursts, KV cache HBM savings;
  - **Exactness / Accuracy**: Zero-degradation validation on **GenEval** and **MMVet**, plus bit-exact `SHA256`, `SSIM`, and 100% greedy AR token match.

### D4. `MorphServeClusterEngine` & Module 3 High-Res Arithmetic-Intensity Gate
- **Cluster Engine (`src/morphserve/engine/`)**:
  - **1 central async coordinator + 8 persistent GPU worker processes** sharing 2 socket-bound `SharedNumaMemfdMirror` regions, connected via full-mesh `DirectNcclP2PChannel` and governed by `HalfNodeH2DPacer`.
  - Soft-biased elastic `expert_alpha` adaptation: workers start with soft role bias (e.g., mixed Und/Hybrid/Gen) and dynamically adapt `alpha` via drop-clean eviction under KV/VAE pressure or idle background H2D prefetch under modality skew, while clean feature flags toggle between MorphServe and all 5 baselines inside the same runtime.
- **Module 3 High-Resolution Gate**:
  - Keep **LECP CoW** (`-49.9%` KV HBM) and **GPU In-Place Loopback** (`29.4×`) always enabled across all resolutions.
  - Gate `FlashPrefix-CFG` dual-pass LSE merge on **`L_shared >= 2 * Q_len`** (verified in `job1b-r7`: all `512²` turns with `Q=1,026, L_shared >= 2,048` achieve `1.10×–1.28×` speedup; `1024²` with `Q=4,098, L_shared=8,192` hits `0.95×` and cleanly falls back to single-pass FA2 with zero regression until `L_shared >= 8,196`).

### D5. Scale-Out & Interconnect Sensitivity (`8 -> 16 -> 64` GPUs)
1. **Primary Real Testbed**: `1×8 NVIDIA A100-PCIE-40GB` running all main QPS, modality-skew, ablation, and accuracy suites.
2. **Cross-Node Spot-Check (`2×8 = 16 GPUs`)**: One targeted multi-node Kairos verification job (when cluster quota/idle nodes allow) measuring intra-node PCIe vs. inter-node fabric routing and proving automatic suppression of cross-node multi-GiB KV shipping.
3. **Calibrated `64-GPU` Discrete-Event Simulator (`src/morphserve/sim/`)**: Calibrated against `8×A100` measurements to `< 5%` prediction error, sweeping:
   - Commodity `PCIe Gen4/Gen5` vs. `A100-NVLink (600 GB/s)` vs. `H100-NVLink (900 GB/s)` coupled with cross-node `RDMA (100–400 Gbps)` across 8 nodes;
   - Demonstrating that even with intra-node NVLink, static fission still loses to MorphServe due to dual-weight HBM waste, modality-skew pool stranding, inter-node RDMA KV bottlenecks, and `1024²` VAE transient OOMs.

---

## 4. Phase-2 Execution Milestones & Verification Gates

| Milestone | Scope & Deliverables | Hard Exit Criteria |
| --- | --- | --- |
| **M2.1: M3 Gate & `Interleaved-Bench` Builder** | Add `L_shared >= 2 * Q_len` dispatch gate in [cfg_prefix_attn.py](file:///mnt/e/000_code/zyd-2026/morphserve/src/morphserve/kernels/cfg_prefix_attn.py); implement `< 1 GiB` `Interleaved-Bench` dataset & dual-trace generator (`mse-bench` + `GenTD26` + `MagicBrush` + `MMVet` + `GenEval` + `BurstGPT`) + offline unit tests. | `pytest` passes; `1024px_turn2_8k` regression (`0.95×`) eliminated (`>= 1.00×`); staged dataset `< 1 GiB` verified before `/data` transfer. |
| **M2.2: `MorphServeClusterEngine` (8-GPU Runtime)** | Build coordinator + 8-worker runtime integrating M1–M4 + elastic `alpha` controller + toggles for all 5 baselines; validate on `1×2` and `1×8` smoke tests. | All 5 baseline modes + MorphServe pass short trace smoke on `8×A100` with zero OOM, `0` VMM leaks, and exact parity checks. |
| **M2.3: `BAGEL-7B-MoT` Main `8×A100` Evaluation** | Execute QPS sweeps, modality-skew shifts (`80:20 ↔ 20:80`), component ablations (`M1`, `M2`, `M3`, `M4`), `1024²` stress tests, and `GenEval`/`MMVet` accuracy verification. | Complete reproducible result JSONs + logs archived under `results/phase2-*`; `< 5%` variance across runs; zero accuracy drop on GenEval/MMVet. |
| **M2.4: `InternVL-U` Cross-Architecture Validation** | Check `/data` gate (`< 790G / 80%`); download & SHA256-verify `InternVL-U` (`~10 GiB`); implement CPTG adapter + `TriPool-VMM` pager; run `8×A100` `tight_mot` vs. `bridge_mllm_dit` comparison. | Verified dual-architecture routing behavior (`bridge` ships via NCCL P2P, `tight_mot` adapts via dual-path crossover) on real `8×A100`. |
| **M2.5: `2×8` Cross-Node Check & `64-GPU` Simulator** | Run `2×8` (`16-GPU`) cross-node validation (subject to idle cluster GPUs); calibrate and run `64-GPU` PCIe/NVLink/RDMA simulator; generate publication-ready tables/figures. | `64-GPU` simulator error `< 5%` vs `8×A100` real runs; full sensitivity curves ready for paper integration. |
