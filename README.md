# MorphServe

**MorphServe** is a 100% exact (zero-accuracy-loss), paradigm-aware serving system for native unified multimodal (Interleaved AR + DiT) models such as `BAGEL-7B-MoT` and `InternVL-U` on commodity multi-GPU PCIe servers (`8×A100 40GB PCIe`).

## Two-Plane Architecture

1. **C++17 Control & Runtime Engine (`include/morphserve/` + `src/morphserve/` + `cmd/`)**:
   - **`vmm/` (`TriPoolVMM` + `PageBackend`)**: Instance-owned dual-tier (`64 MiB` super-handles ↔ `2 MiB` pages) virtual memory pool unifying Persistent Multimodal KV, Scoped `MemPool` + `CUDAPluggableAllocator` Transient Scratchpads, and Opportunistic Secondary Expert slices across `CudaPageAdapter` (`dlopen("libcuda.so.1")` `cuMemMap`) and `HostMmapPageAdapter` (`mmap` CPU tests).
   - **`runtime/` (`CudaDriver`, `NumaMemfdMirror`, `HalfNodeH2DPacer`, `PeerByteTransport`)**: RAII Linux `memfd_create` + `/proc/<pid>/fd/<fd>` NUMA-bound shared weight mirrors, cross-process `flock` + analytical PCIe half-node H2D pacing, and unified P2P byte transport (`NcclPeerAdapter`, `TcpPeerAdapter`, `InMemoryPeerAdapter`).
   - **`control/` (`ClusterTopologyProfile`, `PolicyConfig`, `WorkerBudgetPlanner`, `PreemptEdfController`, `DualPathMorphRouter`, `MorphControlEngine`, `ClusterSimulator`)**: Single authoritative C++17 control plane, EDF slack-coalesced preemption math, dual-regime memory planner, and native `8 -> 16 -> 64` GPU discrete-event simulator (`morphserve-sim`), exposed via a pure C ABI (`include/morphserve/c_api.h` → `libmorphserve_core.so`).
2. **Python Data & Model Engine (`python/morphserve/`)**:
   - **`models/` (`CPTGModelAdapter`)**: `BagelCPTGAdapter` (`tight_mot`), `InternVLUCPTGAdapter` (`bridge_mllm_dit`), and `SyntheticCPTGAdapter`.
   - **`kernels/`**: Longest Exact Causal Prefix (`LECP`) CoW KV sharing, Triton `FlashPrefix-CFG` two-pass attention + online-softmax LSE merge, and zero-CPU `GpuInPlaceLoopback` Catmull-Rom bicubic resize.
   - **`vmm/`, `worker/`, `router/`, `engine/`, `sim/`, `bench/`**: Thin Python tensor/lifecycle bindings atop `morphserve._core` (`libmorphserve_core.so`) plus `Interleaved-Bench` and `ExactnessEvaluator`.

## Repository Layout

```text
morphserve/
├── include/morphserve/        # Public C++17 module interfaces (.hpp) & pure C ABI (c_api.h)
│   ├── control/               # Topology, policy, budget_planner, preempt_edf, router, controller, simulator
│   ├── runtime/               # cuda_driver, numa_mirror, h2d_pacer, peer_transport
│   └── vmm/                   # page_backend, tripool_vmm
├── src/morphserve/            # 1:1 paired C++17 module implementations (.cpp) -> libmorphserve_core.so
├── cmd/                       # Standalone C++17 CLIs (morphserve_sim.cpp, tripool_vmm_probe.cpp)
├── python/morphserve/         # Python upper-level data & model engine (torch / Triton / vllm-omni)
├── tests/
│   ├── cpp/                   # Native C++17 CTest suites (test_vmm_and_runtime, test_control_and_sim)
│   └── python/                # Offline & CUDA-driver pytest/unittest suites
├── deploy/kairos/             # Cluster build_stage.sh, job launchers & JSON specs
├── configs/                   # Model, topology & workload configs
├── GLOSSARY.md                # Canonical domain & architectural vocabulary
└── CMakeLists.txt             # Zero-nvcc / zero-pybind11 C++17 build definition
```

## Local Build & Verification

```bash
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake -j"$(nproc)"
ctest --test-dir build/cmake --output-on-failure
./deploy/kairos/scripts/build_stage.sh
PYTHONPATH=python pytest -q tests/python/
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
    --build-cmd 'bash "$SRC_DIR/deploy/kairos/scripts/build_stage.sh"'
  ```
- **Dry-run & submit jobs** via `../kairos-toolkit/tools/kairos_queue.py`.
- **Shared storage discipline**: Ephemeral caches (`HF_HOME`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`) are forced to container-local `/tmp`; `/data` stop line is strictly enforced (`<= 790G / < 80%` deploy gate, `<= 805G / < 81%` runtime gate, `810G / 82%` hard stop).
