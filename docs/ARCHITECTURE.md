# MorphServe Architecture & Confirmed Design Decisions

> Locked on 2026-11-20 through full design-tree review against `jjjkairos` cluster constraints and `kairos-toolkit`.

## 1. Confirmed Design Decisions (1–10)

| # | Design Branch | Locked Decision |
| :---: | :--- | :--- |
| 1 | **Project Isolation & Repo** | New independent Git repo `morphserve/` (remote `<PROJECT>=MORPHSERVE`). Keep `mllmserve/` (`DualFlow`) completely untouched. Reuse registered Kairos image `192.168.1.114:30500/jjjkairos/mllmserve:vllm-omni0.30.0-r2` and borrow BAGEL adapters / `GenTD26` + `mse-bench` traces as needed. |
| 2 | **Toolchain & Zero-Build Policy** | Run 1-GPU Phase-0 capability probe first. Compile `TriPool-VMM` via C++17 + `dlopen("libcuda.so.1")` (no hard `nvcc` dependency) using `deploy_build.sh --mode stage` lazy in-container `/tmp` builds. Implement `cfg_attn` in **Triton** first (optional FlashInfer/CUDA if present). Put lightweight missing Python deps in `/shared/code/MORPHSERVE/venv`; only pause to request a prebuilt `.tar` image if a hard dependency is missing. |
| 3 | **Model Acquisition & `/data` Budget** | Staged in-cluster download + per-file SHA256 verification. Phase-0 downloads `ByteDance-Seed/BAGEL-7B-MoT` (~30 GiB) first; `InternVL-U/InternVL-U` (~10 GiB) later when implementing CPTG generality. Override `HOME`, `HF_HOME`, `TRITON_CACHE_DIR`, and `TORCHINDUCTOR_CACHE_DIR` to container `/tmp` in `morphserve_common.sh`. |
| 4 | **Unified Execution Engine** | Build a single-process, fully controllable `MorphServe Unified Worker` directly reusing `vllm-omni`'s BAGEL model layers/operators for exact equivalence. Preserve unmodified stock `vllm-omni` (Single-Stage & Two-Stage) as external baselines, and toggle MorphServe's 4 modules via flags for clean ablations. |
| 5 | **`TriPool-VMM` PyTorch Integration** | Three disjoint CUDA VA ranges backed by one 2MB physical-page pool (`CUmemGenericAllocationHandle`): (1) 2MB paged Persistent KV tensors, (2) 2MB layer-aligned Opportunistic Secondary Expert Cache with drop-clean eviction & async H2D streaming, (3) scoped `torch.cuda.MemPool` + `CUDAPluggableAllocator` wrapping only VAE/DiT transient execution scopes so NCCL & PyTorch default allocators are undisturbed. |
| 6 | **Multi-GPU Transport & NUMA Pool** | Primary real-cluster testbed: single-node `8×A100-PCIE-40GB` (+ 64-GPU calibrated simulator). Path A (State-Shipping) uses NCCL P2P on a dedicated transfer stream (retaining multi-node extensibility). Host Expert Weight Pool uses **1 shared read-only Pinned DRAM mirror per NUMA socket** (`shm_open` + NUMA `mbind` + `cudaHostRegister`, ~60 GiB total on dual-socket instead of ~240 GiB). Auto-probe P2P/H2D bandwidth matrix at startup. |
| 7 | **Exactness Red Line (Tiered)** | Zero approximation (no token pruning, no step dropping, no approximate CFG caching). **Bit-Exact SHA256** required for `TriPool-VMM`, Dual-Path Morphing, and Preemptive Scheduling. **Zero-Approximation Mathematical Equivalence** required for custom Triton Shared-Prefix CFG Attention & GPU In-Place Loopback (within `bf16` associative reduction tolerance matching FA2 vs. SDPA, exact `uint8` + PIL-equivalent resize, ~100% AR greedy token match, image `SSIM >= 0.999`, zero GenEval/MMVet degradation). |
| 8 | **Preemption Quantum, SLO & Baselines** | Calibrate preemption granularity from Phase-0 A100 measurements: because 20-step 3-way CFG BAGEL-7B-MoT takes ~8s (~200–400ms/step on A100), upgrade Module 4 to **CFG-branch or layer-chunk lossless preemption (~30–80ms quantum)** whenever a full DiT step exceeds the AR TPOT budget. Anchor SLOs at `1.5×/2.0×` unloaded measured latency and plot multi-SLO curves. Run stock `vllm-omni` externally and faithfully reproduce `Cornserve-Style Static Fission` and `Naive Host Offloading` on the same runtime. |
| 9 | **`Interleaved-Bench` Datasets** | Staged lightweight construction (< 2 GiB total on `/data`). Phase-0 reuses local `mse-bench` + `GenTD26` to build the minimal 4-class interleaved suite (30% Visual-CoT, 30% multi-turn edit, 20% T2I, 20% multi-image VQA). Main evaluation adds curated MagicBrush/SeedEdit subsets, BurstGPT trace, and standard GenEval / MMVet accuracy suites. |
| 10 | **Phase-0 Rollout Gate** | Execute Milestones 1 + 2 first: local repo scaffold + C++ `cuMemMap` / `CUDAPluggableAllocator` prototype + Job 0a (`1×1 GPU` toolchain/MemPool/VMM live probe on `jjjkairos`). Review Job 0a measurements before launching Job 0b (30 GiB BAGEL download), Job 0c (`8×A100` PCIe/NUMA bandwidth matrix), and Job 0d (`vllm-omni` Motivation Profiling). |

## 2. Phase-0 Verification Gates (Job 0a)

Job 0a (`1×1 GPU`) must establish hard ground truth on `192.168.1.114:30500/jjjkairos/mllmserve:vllm-omni0.30.0-r2` before heavier investments:
1. **Build & Runtime Toolchain**: `g++`, `cmake`, `nvcc`, `cuda.h`, `libcuda.so.1`, `triton`, `flashinfer`, `nsys`, `numactl`, `/dev/shm`, host RAM & NUMA layout.
2. **PyTorch VMM Hooks**: `torch.cuda.CUDAPluggableAllocator`, `torch.cuda.MemPool`, `torch.cuda.use_mem_pool`.
3. **`vllm-omni` Introspection**: Exact BAGEL model class definitions, parameter partition (Understanding vs. Generation vs. ViT vs. VAE), and `InternVL-U` support status.
4. **Live A100 `TriPool-VMM` Microbenchmark**:
   - Granularity check (`2 MiB` expected on A100-PCIE-40GB).
   - Single 2MB page `cuMemUnmap -> cuMemMap -> cuMemSetAccess` latency (P50 / P95 / P99 µs).
   - Single-layer (512 MiB) & full VAE-spike (4 GiB) drop-clean + transient map + unmap latency.
   - Async H2D streaming bandwidth from pinned host DRAM into freshly mapped 2MB VA pages.
   - End-to-end PyTorch `MemPool` + `libtripool_pluggable.so` live GPU Conv2d/VAE-like spike test verifying **bit-exact output** and **100% physical page reclamation** upon scope exit.
