# MorphServe

**MorphServe** is a 100% exact (zero-accuracy-loss), paradigm-aware serving system for native unified multimodal (Interleaved AR + DiT) models such as `BAGEL-7B-MoT` and `InternVL-U` on commodity multi-GPU PCIe servers (`8×A100 40GB PCIe`).

## Core Contributions

1. **`TriPool-VMM` (`csrc/vmm_tripool/`)**: 2MB physical-page (`cuMemMap`) virtual memory pool unifying three disjoint VA arenas — Persistent Multimodal KV, scoped `torch.cuda.MemPool` + `CUDAPluggableAllocator` Transient Scratchpad (VAE/DiT spikes), and Opportunistic Layer-Wise Secondary Expert Cache (drop-clean eviction + async H2D streaming).
2. **Topology-Aware Dual-Path Morphing (`src/morphserve/router/`)**: Startup auto-probed PCIe/NUMA bandwidth matrix choosing between same-switch NCCL P2P State-Shipping and In-Place KV Residency + Step-1 Layer-Wise Expert Weight Streaming backed by per-NUMA-socket shared read-only Pinned DRAM weight mirrors (`shm_open` + `mbind` + `cudaHostRegister`).
3. **100% Exact Shared-Prefix CFG Attention & GPU In-Place Loopback (`csrc/kernels/cfg_attn/`)**: Longest Exact Causal Prefix (LECP) CoW block table + Triton grouped-query shared-prefix CFG attention kernel + GPU in-place VAE→`uint8`→resize→ViT loopback (zero approximation, mathematically exact).
4. **Deterministic Adaptive Sub-Step / Step-Preemptive AR+DiT Scheduler (`src/morphserve/worker/`)**: Unified single-process elastic worker executing both AR continuous batching and DiT denoising over `vllm-omni`'s BAGEL model layers, preempting DiT at calibrated step / CFG-branch / layer-chunk boundaries to protect AR TPOT SLO.

## Repository Layout

```text
morphserve/
├── csrc/
│   ├── vmm_tripool/           # C++17 CUDA Driver API (cuMemMap) VMM & CUDAPluggableAllocator
│   └── kernels/cfg_attn/      # Triton shared-prefix CFG attention & GPU loopback kernels
├── src/morphserve/
│   ├── probe.py               # Phase-0 toolchain, topology, vllm-omni & VMM live audit
│   ├── router/                # Dual-path topology-aware cost router
│   ├── worker/                # Unified AR+DiT elastic worker & step-preemptive scheduler
│   ├── models/                # CPTG adapters (BAGEL-7B-MoT, InternVL-U)
│   └── bench/                 # Interleaved-Bench driver & 64-GPU calibrated simulator
├── deploy/kairos/
│   ├── scripts/               # Cluster job launchers (built atop kairos-toolkit/runtime)
│   └── jobs/                  # Kairos JSON job specifications (validated by kairos_queue.py)
├── configs/                   # Model, topology & workload configs
├── docs/ARCHITECTURE.md       # Confirmed architectural decisions & Phase-0 plan
└── tests/                     # Offline unit tests
```

## Kairos Cluster Workflow (`jjjkairos`)

All remote operations strictly follow workspace `AGENTS.md` and `../kairos-toolkit/README.md`:
- **No image builds** (local WSL or cluster). Reuse `192.168.1.114:30500/jjjkairos/mllmserve:vllm-omni0.30.0-r2`.
- **Write-once staged C++/Python publish**:
  ```bash
  bash ../kairos-toolkit/tools/deploy_build.sh \
    --project MORPHSERVE \
    --repo /mnt/e/000_code/zyd-2026/morphserve \
    --mode stage \
    --runtime-image 192.168.1.114:30500/jjjkairos/mllmserve:vllm-omni0.30.0-r2 \
    --build-cmd 'bash "$SRC_DIR/csrc/vmm_tripool/build.sh"'
  ```
- **Dry-run & submit jobs** via `../kairos-toolkit/tools/kairos_queue.py`.
- **Shared storage discipline**: Ephemeral caches (`HF_HOME`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`) are forced to container-local `/tmp`; `/data` stop line is strictly enforced (`<= 790G / < 80%` deploy gate, `<= 805G / < 81%` runtime gate, `810G / 82%` hard stop).
