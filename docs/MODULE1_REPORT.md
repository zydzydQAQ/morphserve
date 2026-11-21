# MorphServe Module 1 (`TriPool-VMM`) Verification Report

- **Date**: 2026-11-21
- **Hardware**: `NVIDIA A100-PCIE-40GB` (`jjjkairos` cluster, CUDA 13.0, Driver `570.133.20`, PyTorch `2.13.0+cu130`, `vllm-omni 0.30.0`)
- **Model**: `ByteDance-Seed/BAGEL-7B-MoT` (`27.52 GiB` verified checkpoint, 28 `Qwen2MoTDecoderLayer` layers)
- **Verified Build / Run**: commit `d5b854a`, Kairos run `morphserve-p1-m1vmm-20261121-r3` ([module1-report.json](file:///mnt/e/000_code/zyd-2026/morphserve/results/phase1-job1a-r3/module1-report.json))
- **Verdict**: **ALL ACCEPTANCE GATES PASSED (`status: "ok"`) — 100% BIT-EXACT SHA256**

---

## 1. Implementation Summary

| Component | Source | Core Mechanism |
| --- | --- | --- |
| Unified C++17 VMM Library | [tripool_vmm.hpp](file:///mnt/e/000_code/zyd-2026/morphserve/csrc/vmm_tripool/tripool_vmm.hpp), [tripool_vmm.cpp](file:///mnt/e/000_code/zyd-2026/morphserve/csrc/vmm_tripool/tripool_vmm.cpp) | Two-tier physical pool (`2 MiB` KV + `64 MiB` Super-Handles), 3 disjoint CUDA VA arenas (Persistent KV / Transient Scratchpad / Opportunistic Expert Cache), `CUDAPluggableAllocator` ABI (`tripool_alloc`/`tripool_free`) with Scope-End Bulk Unmap, automatic drop-clean tail expert eviction under memory pressure, elastic `64 MiB -> 32×2 MiB` KV carving, dedicated background H2D streamer thread with batched `cuMemSetAccess` TLB update and GPU-side `cuStreamWaitEvent` sync. |
| Python VMM & NUMA Mirror | [tripool.py](file:///mnt/e/000_code/zyd-2026/morphserve/src/morphserve/vmm/tripool.py) | `TriPoolVMM` ctypes binding, zero-copy CUDA Array Interface v3 `_CudaVAView`, NUMA-bound `memfd_create` + `cuMemHostRegister` pinned host mirror (`NumaMemfdMirror`, bypassing `/dev/shm` `32 GiB` cap), scoped `torch.cuda.MemPool` wrapper (`run_in_transient_scope`). |
| BAGEL-7B-MoT Pager | [bagel_pager.py](file:///mnt/e/000_code/zyd-2026/morphserve/src/morphserve/vmm/bagel_pager.py) | Two-phase zero-overshoot attach (`offload_to_host_mirror()` then `bind_vmm()`), packing all 9 generation params/layer (`qkv_proj.gen_exp.*`, `o_proj.gen_exp.*`, `*.gen_weight`, `mlp_moe_gen.*`), zero-copy permanent `Parameter.data` VA rebinding, per-layer `wait_expert_layer` hooks, and automatic VAE transient scoping. |
| Live A100 Validator | [validate_module1_bagel.py](file:///mnt/e/000_code/zyd-2026/morphserve/src/morphserve/bench/validate_module1_bagel.py) | End-to-end exactness, AR isolation, `alpha` sweep (`512²` & `1024²`), KV carving, and tight `14 GiB` zero-OOM `1024² b=2` FLUX VAE pressure proof. |

---

## 2. Live A100-40GB Measured Results (`r3`, `d5b854a`)

### 2.1 Memory Packing & AR Isolation

- **Generation Expert Packing**: `252` parameters (`9/layer × 28 layers`) = `12,446.64 MiB` packed into `28 × 448 MiB` (`7 × 64 MiB` super-handles/layer, `12,544 MiB` total) → **`99.22%` packing efficiency**, `numa_bound = true`.
- **HBM Unlocked**: Monolithic `BAGEL-7B-MoT` `torch_alloc = 28,149.21 MiB` (`driver_free = 11,665.56 MiB`) → after offloading Generation Expert to NUMA `memfd`: `torch_alloc = 15,665.52 MiB` (**`24,075.56 MiB` = `23.51 GiB` free HBM unlocked** for `TriPool-VMM`!).
- **AR Isolation (`alpha = 0.0`)**: With all 28 generation layers evicted, AR text prefill executes cleanly with **`0` generation expert faults**.

### 2.2 Batched `cuMemSetAccess` Optimization (`r1` → `r3`)

Coalescing contiguous missing layers into a single batched `cuMemSetAccess` before per-layer `cuMemcpyHtoDAsync` eliminates mid-DMA GPU MMU TLB shootdowns:

| Missing Layers | `r1` Per-Layer Map (`ms`) | `r3` Batched Map (`ms`) | Speedup |
| ---: | ---: | ---: | ---: |
| 7 layers (`alpha=0.75`) | `45.55` | **`3.11`** | **`14.6×`** |
| 14 layers (`alpha=0.50`) | `97.38` | **`6.03`** | **`16.1×`** |
| 21 layers (`alpha=0.25`) | `149.54` | **`9.00`** | **`16.6×`** |
| 28 layers (`alpha=0.00`) | `202.51` | **`11.65`** | **`17.4×`** |

### 2.3 Step-1 Compute-H2D Overlap & 100% Bit-Exactness

Baseline unmodified `BAGEL-7B-MoT` (`512×512`, 20 steps): **`4,909.80 ms`**, SHA256 `cf80eb13...d0134`.

#### `512×512` (20 steps, Turn-0 short context)

| `alpha` (Resident) | Streamed H2D (`MiB`) | Map (`ms`) | Un-overlapped H2D (`ms`) | Total Gen (`ms`) | Morph Stall vs `alpha=1` (`ms`) | Overlap Masked (`ms` / `%`) | Bit-Exact (`max_px_diff`) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `1.00` (28/28) | `0.00` | `0.00` | `0.00` | `4,931.16` | `0.00` | `—` | **Yes (`0`)** |
| `0.75` (21/28) | `3,111.66` | `3.11` | `264.62` | `4,958.33` | **`27.17`** | **`237.45` (`89.7%`)** | **Yes (`0`)** |
| `0.50` (14/28) | `6,223.33` | `6.03` | `529.25` | `5,232.43` | **`301.27`** | **`227.98` (`43.1%`)** | **Yes (`0`)** |
| `0.25` (7/28) | `9,334.99` | `9.00` | `793.87` | `5,498.05` | **`566.89`** | **`226.98` (`28.6%`)** | **Yes (`0`)** |
| `0.00` (0/28) | `12,446.66` | `11.65` | `1,058.50` | `5,818.55` | **`887.39`** | **`171.11` (`16.2%`)** | **Yes (`0`)** |

#### `1024×1024` (10 steps, high-res DiT)

| `alpha` (Resident) | Streamed H2D (`MiB`) | Map (`ms`) | Un-overlapped H2D (`ms`) | Total Gen (`ms`) | Morph Stall vs `alpha=1` (`ms`) | Overlap Masked (`ms` / `%`) | Bit-Exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `1.00` (28/28) | `0.00` | `0.00` | `0.00` | `9,711.63` | `0.00` | `—` | **Yes** |
| `0.50` (14/28) | `6,223.33` | `6.00` | `529.25` | `9,680.25` | **`~0` (`-31.38`)** | **`529.25` (`100.0%`)** | **Yes** |
| `0.00` (0/28) | `12,446.66` | `11.45` | `1,058.50` | `9,788.17` | **`76.54`** | **`981.96` (`92.8%`)** | **Yes** |

*Comparison vs. Stock Two-Stage `vLLM-Omni` (Job 0d-r2)*: Stock 2-GPU separation burns `71.29 GiB` total HBM and stalls `2,610–2,870 ms/turn` transferring KV; `TriPool-VMM` runs on **1 GPU (`<36 GiB`)** with **`0–301 ms` morph stall at `alpha = 0.50`** (**`8.7×–∞` faster transition**).

### 2.4 Persistent KV `2 MiB` Arena + Elastic Super-to-Fine Carving

- Allocated `540 × 2 MiB` KV pages (`1,080 MiB > 1,024 MiB` initial fine pool) → automatically triggered **`super_to_fine_carves = 1`** (`64 MiB -> 32 × 2 MiB`), GPU write/readback (`0x4B`) verified, 100% pages reclaimed, `overflow_creates = 0`.

### 2.5 Tight `14 GiB` Pool + `1024² b=2` FLUX VAE Spike (Zero-OOM Proof)

| Metric | Stock PyTorch Allocator | `TriPool-VMM` (`14 GiB` strict, `allow_overflow=False`) |
| --- | --- | --- |
| Pre-VAE Resident Gen Expert | `28 / 28` (`12.25 GiB`) | `28 / 28` (`12.25 GiB`, only `1.75 GiB` free!) |
| Peak Transient VAE Demand | Hit internal OOM & emergency cache flush | **`12,224 MiB` (`11.94 GiB`)** absorbed with **`overflow_creates = 0`** |
| Auto Drop-Clean Eviction | N/A | **`24` tail layers (`28 -> 4`)** unmapped in **`6.87 ms`** total |
| `1024² b=2` VAE Decode Latency | `505.10 ms` | **`379.21 ms` (`1.33× FASTER`, `-24.9%`)** |
| Output SHA256 Exactness | `fd681ffa...40c668` | **`fd681ffa...40c668` (100% Bit-Exact)** |
| Post-VAE Transient Reclamation | Cached / fragmented | **`100%` reclaimed (`0` mapped handles)** |
| Immediate Post-Eviction DiT | N/A | Streams back 24 dropped layers → **100% Bit-Exact (`cf80eb13...`)** |
