# MorphServe Curated Experimental Results (`results/`)

This directory stores only **reusable hardware/network characterization profiles** and **valid OSDI '27 v2 (Interleaved-Bench v2) experimental artifacts** calibrated against the current MorphServe runtime. All expired v1 runs, pre-refactor logs (`*.log`), and redundant reference copies (`vllm_omni_ref/`) have been removed.

## Directory Layout

### 1. `hardware-profile/` — Reusable Platform & Microbenchmark Profiles

Hardware-level measurements on the Kairos A100-PCIE-40GB cluster that remain valid across codebase revisions and serve as calibration inputs to `ClusterPhysicsProfile::from_phase01_results()` (`src/morphserve/control/topology.cpp`):

| Directory | Provenance Key / Consumer | Build Commit | Source Kairos Run | Primary Report | Summary |
| --- | --- | --- | --- | --- | --- |
| `hardware-profile/vmm-microbench/` | `job0a_vmm` | `df6a266` | `morphserve-p0-vmmv2-20261120-r1` | `vmm-bench.json` | CUDA VMM granularity sweep (`64 MiB` super-handles), solo H2D bandwidth, zero residual memory leak |
| `hardware-profile/topo-8gpu/` | `job0c_topo` (`ClusterTopologyProfile`) | `df6a266` | `morphserve-p0-topo8g-20261120-r1` | `topo-8gpu-report.json` | 8×A100 PCIe/NUMA topology, dual-half H2D contention (`socket0`/`socket1`), NCCL P2P pair bandwidth matrix |
| `hardware-profile/motivation-baseline/` | `job0d_motivation` | `3d0f60c` | `morphserve-p0-motiv-20261120-r2` | `motivation-report.json` | Monolithic single-GPU VAE/KV OOM stress (`512` / `1024`) and two-stage SHM KV transfer baseline |
| `hardware-profile/module2-pcie-numa/` | `module2_router` | `16c21dd` | `morphserve-p1-m2rout-20261121-r4` | `module2-report.json` | Shared NUMA `memfd` mirror, half-node H2D pacer (`paced` vs `unpaced`), BAGEL dual-path crossover |
| `hardware-profile/multinode-2x8-net/` | `m2x8` (optional fabric probe) | `403e7aa` | `morphserve-osdi27-r3-node2x8-20261124` | `m25-multinode-2x8-report.json` | 2×8-GPU inter-node NCCL/TCP fabric probe (`intra_half_pcie`, `intra_node_cross_socket`, `inter_node_fabric`); expired v1 E2E file excluded |

### 2. `osdi27-v2/` — Valid Interleaved-Bench v2 Evaluation Results

Current-protocol evaluation artifacts under the OSDI '27 v2 specification (turn-level SLOs D1–D5, log-normal inter-turn think time, time-based warmup/steady-state windows, and `cot_then_dit` timing decoupling):

| Directory | Provenance Keys | Build Commit | Source Kairos Run | Primary Reports | Summary |
| --- | --- | --- | --- | --- | --- |
| `osdi27-v2/stage1-gate/` | `module1_vmm`, `module3_cow_loopback`, `module4_scheduler` | `8b16ef2` | `morphserve-osdi27-v2-r1-8b16ef2-gate1g` | `stage1-gate-summary.json`, `module1/module1-report.json`, `module3/module3-report.json`, `module4/module4-report.json`, `internvlu_1gpu/m24-internvlu-report.json`, `bagel_1gpu/bagel-1gpu-smoke.json`, `ctest-a100.txt`, `pytest-a100.txt` | Passed Stage-1 1×A100 gate (`status: "ok"`): CTest `2/2`, pytest `81 passed, 1 skipped`, M1/M3/M4 bit-exact/SLO verified, InternVL-U 12 GiB quota & BAGEL 1-GPU cluster smoke verified |

Future multi-GPU v2 campaign results (`r2a-suite-a`, `r2b-phaseshift-acc`, `r2c-internvlu`, `r2e-lowqps-8gpu`, `r2f-thinktime-8gpu`, `r3-multinode-2x8`, `sim`) will be synced into `results/osdi27-v2/<suite>/` upon completion.

## Provenance Contract (`ClusterPhysicsProfile`)

`ClusterPhysicsProfile::from_phase01_results("results")` loads **7 tracked provenance files** (`provenance_files.size() == 7`, verified by `test_control_and_sim` and `test_module1_vmm_offline.py`):

1. `job0a_vmm` → `hardware-profile/vmm-microbench/vmm-bench.json`
2. `job0c_topo` → `hardware-profile/topo-8gpu/topo-8gpu-report.json`
3. `job0d_motivation` → `hardware-profile/motivation-baseline/motivation-report.json`
4. `module1_vmm` → `osdi27-v2/stage1-gate/module1/module1-report.json`
5. `module3_cow_loopback` → `osdi27-v2/stage1-gate/module3/module3-report.json`
6. `module4_scheduler` → `osdi27-v2/stage1-gate/module4/module4-report.json`
7. `module2_router` → `hardware-profile/module2-pcie-numa/module2-report.json`

Plus 1 supplementary multi-node fabric calibration file (loaded outside `provenance_files`):

- `hardware-profile/multinode-2x8-net/m25-multinode-2x8-report.json`

Legacy paths (`phase0-job0*`, `phase1-job1*`, `osdi27-v1/*`) are retained strictly as fallbacks in `topology.cpp` for unit tests constructing synthetic temp directories.

## Hygiene Policy

- Do **not** commit `*.log`, `*.nsys-rep`, `*.sqlite`, raw checkpoint dumps, or duplicated `vllm_omni_ref/` trees into `results/` (`test_module1_vmm_offline.py` enforces zero `*.log` files under staged `results/`).
- Every directory under `results/` must preserve its `BUILD_INFO.json`, `runtime-facts*.json`, and `memory-summary*.json` provenance metadata.
