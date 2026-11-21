# MorphServe Module 2 (8-GPU Topology-Aware Dual-Path Morphing Router) Verification Report

- **Date**: 2026-11-21
- **Hardware**: `8× NVIDIA A100-PCIE-40GB` (`jjjkairos` cluster, CUDA 13.0, Driver `570.133.20`, PyTorch `2.13.0+cu130`, `vllm-omni 0.30.0`, NCCL `2.29.7`)
- **Model**: `ByteDance-Seed/BAGEL-7B-MoT` (`27.52 GiB`, 28 `Qwen2MoTDecoderLayer` layers, plus CPTG bridge-model validation for `InternVL-U`)
- **Verified Build / Run**: commit `16c21dd`, Kairos `1×8` GPU run `morphserve-p1-m2rout-20261121-r4` ([module2-report.json](file:///mnt/e/000_code/zyd-2026/morphserve/results/phase1-job1d-r4/module2-report.json))
- **Verdict**: **ALL ACCEPTANCE GATES PASSED (`status: "ok"`) — 100% BIT-EXACT SHA256**

---

## 1. Implementation Summary

| Component | Source | Core Mechanism |
| --- | --- | --- |
| Live PCIe/NUMA Topology Profile | [topology.py](file:///mnt/e/000_code/zyd-2026/morphserve/src/morphserve/router/topology.py) | `PcieHalfGroup`, `ClusterTopologyProfile`, `probe_live_topology()` — captures real `8×A100` P2P bandwidth matrix (`9.08–9.99 GB/s`) and dual PCIe half-node H2D root groups (`{0,1,2,3}` & `{4,5,6,7}`) with container-safe sysfs fallback. |
| Cost-Aware CPTG Router & H2D Pacer | [cost_router.py](file:///mnt/e/000_code/zyd-2026/morphserve/src/morphserve/router/cost_router.py) | `CPTGRequestSpec` (uniformly handles `tight_mot` BAGEL KV state and `bridge_mllm_dit` InternVL-U compact bridge tensors), `WorkerState`, `HalfNodeH2DPacer` (`fcntl.flock` cross-process + scheduler reservation), and `DualPathMorphRouter` weighing Path A `T_ship` vs Path B `T_morph`. |
| Shared NUMA `memfd` & Direct NCCL P2P | [nccl_state_shipper.py](file:///mnt/e/000_code/zyd-2026/morphserve/src/morphserve/router/nccl_state_shipper.py) | `SharedNumaMemfdMirror` (`memfd_create` + `numa_tonode_memory` + `/proc/<pid>/fd/<fd>` cross-process `mmap(MAP_SHARED)` + `cuMemHostRegister`), `DirectNcclP2PChannel` (`libnccl.so.2` direct P2P communicator avoiding `vllm_omni` `WORLD` group collision), `NcclStateShipper` (contiguous `NaiveCache` packing). |
| Live 8-GPU Validator | [validate_module2_dualpath.py](file:///mnt/e/000_code/zyd-2026/morphserve/src/morphserve/bench/validate_module2_dualpath.py) | Full `8×A100` shared host mirror & H2D pacing benchmark (`Test 1`), real `BAGEL-7B-MoT` Rank 0 ↔ Rank 1 Path A vs `TriPool-VMM` Path B crossover (`Test 2`), and 8-request mixed CPTG burst routing (`Test 3`). |

---

## 2. Live `8×A100` Measured Results (`r4`, `16c21dd`)

### 2.1 Shared NUMA `memfd` Pool & `HalfNodeH2DPacer` (`Test 1`, 8 GPUs)

- **Shared Host Mirror Memory**: 8 GPU workers attach via `/proc/<leader_pid>/fd/<fd>` to **2 shared NUMA `memfd` mirrors (`12.25 GiB` each)** (`all_inodes_matched = true`, `all_sentinels_matched = true`).
  - Measured `/proc/meminfo` `Shmem` delta: **`24.5 GiB`** (vs `98.0 GiB` with per-worker mirrors) → **`4.0×` Host DRAM savings**.
- **PCIe Root H2D Pacing (`4,480 MiB` DMA per GPU)**:

| Mode | Active GPUs | Per-GPU H2D (`GB/s`) | Per-GPU DMA (`ms`) | 4-Req Wall (`ms`) | Per-GPU Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Unpaced Same-Half (`{0,1,2,3}` simultaneous) | `4` on Half-0 | `3.10` (`12.40 agg`) | `1,517.50` | `1,517.53` | `1.00×` |
| **`HalfNodeH2DPacer` Dual-Half (`{0,1}` + `{4,5}`)** | `1` / half × 2 halves | **`12.34`** | **`381.37`** | **`764.28`** | **`3.98×`** |

### 2.2 Real `BAGEL-7B-MoT` Dual-Path Crossover & 100% Bit-Exactness (`Test 2`, Rank 0 ↔ Rank 1)

| Workload Regime | Context Size | Router Decision | Path A `NcclStateShipper` (`ms` / `GB/s`) | Path B `TriPool-VMM` Overhead (`ms`) | Path A Bit-Exact | Path B Bit-Exact |
| --- | ---: | :---: | ---: | ---: | :---: | :---: |
| **Turn-0 Short KV (`alpha_src=0.50`, `gamma=1`)** | `0.98 MiB` | **`state_ship` (`GPU 1`)** | **`0.324 ms`** (`3.184 GB/s`) | `258.79 ms` | **Yes** | **Yes** |
| **Turn-3 Long Interleaved (`alpha_src=0.75`, `gamma=2`, `seq=23,822`)** | `1,302.77 MiB` | **`in_place_morph` (`GPU 0`)** | `329.08 ms` (`10.362 GB/s`) | **`214.92 ms`** | **Yes** | **Yes** |

### 2.3 Mixed 8-Request CPTG Burst Routing (`Test 3`, 8×A100 Live Profile)

- **`InternVL-U` Bridge Requests (`24 / 32 MiB`)**: `bridge_all_shipped = true` (`req1 → GPU 3` in `8.53 ms`, `req4 → GPU 5` in `10.37 ms`).
- **Turn-0 Short `BAGEL` Requests (`4 / 6 MiB`)**: `short_all_shipped = true` (`req0 → GPU 1` same-half `3.46 ms`, `req6 → GPU 7` cross-half spill `3.69 ms`).
- **Long Interleaved & `1024px` `BAGEL` Requests (`1,200–2,100 MiB`)**: `long_and_1024_all_in_place = true` (`req2 → GPU 2`, `req3 → GPU 4`, `req5 → GPU 6`, `req7 → GPU 0`, utilizing all 8 GPUs cleanly across both PCIe halves).
