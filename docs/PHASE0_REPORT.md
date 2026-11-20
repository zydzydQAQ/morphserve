# MorphServe Phase-0 Ground-Truth Profiling Report (`8×A100-PCIE-40GB`)

All measurements below were captured directly on `jjjkairos` (`NVIDIA A100-PCIE-40GB`, CUDA 13.0, PyTorch `2.13.0+cu130`, `vLLM-Omni 0.30.0`, `Triton 3.7.1`, `FlashInfer 0.6.18.post1`) under zero-image-build discipline.

## 1. Real Hardware Topology & Fabric (`results/phase0-job0c/topo-8gpu-report.json`)

- **Host**: Dual-socket 80 vCPU, `645 GiB` RAM (NUMA 0: `322 GB`, NUMA 1: `322 GB`), `/dev/shm` capped at **`32.0 GiB`**.
- **Shared Weight Pool Design Validation**: `memfd_create + numa_tonode_memory + cuMemHostRegister` verified (`memfd_numa_ok = true`, byte-exact), cleanly bypassing the `32 GiB` `/dev/shm` cap.
- **Measured PCIe Topology**:
  - GPUs `{0,1,2,3}` and `{4,5,6,7}` form two PCIe root/switch groups, each sharing an aggregate **`~10.3–10.4 GB/s`** uplink (`all_8gpu` aggregate H2D = **`20.12 GB/s`**; 4-GPU concurrent H2D on one group = **`2.58–2.60 GB/s/GPU`**).
  - Single-pair P2P: **`9.11–10.00 GB/s`** (NCCL P2P: **`10.47–11.27 GB/s`**).
  - **Concurrent P2P Collapse**:
    - 4 simultaneous intra-half pairs (`0→1, 2→3, 4→5, 6→7`): aggregate **`19.67 GB/s`** (`4.92 GB/s/link`).
    - 4 simultaneous cross-half pairs (`0→4, 1→5, 2→6, 3→7`): aggregate collapses **2×** to **`9.98 GB/s`** (**`2.50 GB/s/link`**).
  - **Router Implication**: Dual-Path Cost Router must enforce (1) intra-half affinity for State-Shipping and (2) per-half H2D streaming concurrency pacing so concurrent Step-1 weight streamers do not saturate the `10.3 GB/s` half-node root.

## 2. `TriPool-VMM` A100 Microbenchmarks (`results/phase0-job0a-v2/vmm-bench.json`)

- **Hardware VMM Granularity**: Exact **`2 MiB`** (`2,097,152 B`), **`0 B` residual HBM delta** (zero leak).
- **Single 2 MiB KV Page Remap** (`cuMemUnmap -> cuMemMap -> cuMemSetAccess`): **P50 = `84.0 µs`** (P99 `108.7 µs`).
- **Key OSDI Insight — Driver ioctl Amplification & Two-Tier Super-Handle Granularity**:

| Physical Handle Size | 512 MiB Layer Remap (P50) | 4 GiB VAE Spike Acquire (P50) | 4 GiB Bulk Unmap Only (P50) | Speedup vs. 2 MiB |
| :---: | :---: | :---: | :---: | :---: |
| **`2 MiB`** (2,048 handles) | `23.11 ms` | `193.94 ms` | `75.42 ms` | `1.0×` |
| **`32 MiB`** (128 handles) | `1.30 ms` | `10.57 ms` | `3.95 ms` | `18.3×` |
| **`64 MiB`** (64 handles) | `0.67 ms` | `5.44 ms` | `2.03 ms` | `35.6×` |
| **`128 MiB`** (32 handles) | **`0.36 ms`** | **`2.84 ms`** | **`1.06 ms`** | **`68.3×`** |
| **`256 MiB`** (16 handles) | `2.12 ms` (min `0.20`) | **`1.57 ms`** | **`0.57 ms`** | **`123.6×`** |

- **128 MiB Super-Handle + Async Pinned H2D (512 MiB layer)**: Remap = **`0.393 ms`** (< 0.9% overhead), H2D = **`12.33 GB/s`** (`43.94 ms` total, `h2d_bit_exact = true`).
- **PyTorch `CUDAPluggableAllocator` + `MemPool` (3.5 GiB simulated VAE Conv2d spike)**:
  - **100% Bit-Exact SHA256** (`max_abs_diff = 0.0`), `overflow_creates = 0`.
  - Naive per-tensor sync/unmap at 2 MiB cost `636 ms`; **128 MiB Super-Handle + Scope-End Bulk Unmap** reduced total 3.5 GiB map to **`2.13 ms`** + bulk unmap **`1.08 ms`**, steady-state **`12.33 ms` vs `12.04 ms` native PyTorch (`+0.29 ms`)** with 100% reclamation upon scoped `MemPool` teardown.

## 3. `BAGEL-7B-MoT` Motivation Profiling (`results/phase0-job0d-r2/motivation-report.json`)

### 3.1 Exact Weight & Layer Anatomy (`27.52 GiB` total, `28.15 GiB` GPU allocated)
- **Understanding Expert (AR LLM)**: **`14.185 GiB`** (`7.62B` params; `attn 1.53 GiB + mlp 10.62 GiB + io/embed 2.03 GiB`) — **uniform `444.52 MiB / layer` × 28 layers**.
- **Generation Expert (DiT MoT)**: **`12.208 GiB`** (`6.55B` params; `attn 1.53 GiB + mlp 10.65 GiB + io 28 MiB`) — **uniform `444.52 MiB / layer` × 28 layers**.
- **SigLIP ViT + Connector**: **`0.814 GiB`** (`437.1M` params).
- **FLUX VAE (`ae.safetensors`, FP32)**: **`0.313 GiB`** (`84.1M` params).
- **`TriPool-VMM` Budget Unlock**: Keeping only active Understanding Expert + ViT + VAE resident uses **`15.31 GiB`**, freeing **`~23.7 GiB`** for the unified 2MB/128MB physical page pool (vs. only `~9–11 GiB` left under monolithic `28.15 GiB` loading).

### 3.2 Observation 1 & 2 — Static Colocation Memory Wall vs. Static Disaggregation Communication Wall
- **Monolithic 1-GPU (`40 GiB` HBM)**:
  - Single 4-turn `512²` session grows KV `4.9 → 326.1 → 651.2 → 976.4 MiB`, pushing reserved HBM to **`35.20 GiB`**.
  - **FLUX VAE Decode Transient Spike**: `512² b=1`: `+0.55–0.90 GiB` (`44–53 ms`); `512² b=2`: **`+1.75 GiB`** (`94 ms`); `1024² b=1`: **`+3.50 GiB`** (`212 ms`, reserved `37.60 GiB`); `1024² b=2`: **`+7.00 GiB`** (`488 ms`, reserved **`39.65 GiB / 40 GiB`**, only `120–266 MiB` left before hard OOM!).
  - **3-Way CFG Prefix Redundancy**: `cfg_text` duplicates **99.7%–99.9%** of `main`'s history tokens (`5,946/5,963` → `17,836/17,854`).
  - **CPU PNG Loopback Overhead**: **`68.7–140.7 ms/turn`** (`PIL.save` + `PIL.open`).
- **Stock `vLLM-Omni` Two-Stage Disaggregated (`2×A100-40GB`)**:
  - Consumes **`71.29 GiB` total HBM** across 2 GPUs (`GPU 0: 40.18 GiB`, `GPU 1: 31.11 GiB`), yet **still** serializes and ships **`main` (`343 MiB`) + duplicate `cfg_text` (`340 MiB`) = `654 MiB/turn`** (`1,964 MiB` across 4 turns even with 1-image context) through `SharedMemoryConnector`.
  - **Measured Cross-Stage Stall**: **`1.53–1.69 s` serialize + `1.09–1.17 s` transfer = `2.61–2.87 s/turn`** (**`8.28 s`** total across 4 turns, **~25%** of single-image turn latency and scaling linearly past compute time as multi-turn visual history grows!).

## 4. Calibrated Parameters for Phase-1 Implementation

1. **`TriPool-VMM`**: Two-tier handles (`2 MiB` for Persistent KV blocks, `128 MiB` Super-Handles for Opportunistic Expert Layers & scoped Transient VAE/DiT `MemPool` with Scope-End Bulk Unmap). Each BAGEL expert layer (`444.52 MiB`) fits cleanly in **4 × 128 MiB** (`512 MiB`) or **7 × 64 MiB** (`448 MiB`, `< 0.8%` internal padding!! — **64 MiB Super-Handles align almost perfectly to `444.52 MiB/layer` at `0.67 ms/layer` remap!!**).
2. **Host Shared Weight Pool**: `memfd_create` + `numa_tonode_memory` + `cuMemHostRegister`, 2 read-only mirrors (~`24.4 GiB` per expert × 2 or `55 GiB` total out of `645 GiB` host RAM).
3. **Dual-Path Cost Router**: Calibrated with measured `10.3 GB/s` half-node root cap, `9.1–11.3 GB/s` single-pair P2P, `2.50 GB/s` cross-half concurrent P2P, `12.33 GB/s` single-GPU H2D, and per-half H2D concurrency pacing.
4. **Module 3 (Triton Shared-Prefix CFG Attention + GPU Loopback)**: Eliminates both the `99.7–99.9%` duplicate HBM reads between `main` and `cfg_text` AND the `69–141 ms/turn` CPU PNG loopback.
5. **Module 4 (Unified AR+DiT Worker & Preemption)**: `512²` 20-step BAGEL DiT takes `~5.6–6.9s` (`~280–345 ms/step`, ~`95–115 ms` per CFG branch), confirming our Decision 8 upgrade to **CFG-branch / layer-chunk lossless preemption (~35–60 ms quantum)** to protect AR TPOT SLO.
