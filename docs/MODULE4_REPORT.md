# MorphServe Module 4 (Layer-Chunk Preemptive AR+DiT Scheduler & Full M1+M3+M4 Stack) Verification Report

- **Date**: 2026-11-21
- **Hardware**: `NVIDIA A100-PCIE-40GB` (`jjjkairos` cluster, CUDA 13.0, Driver `570.133.20`, PyTorch `2.13.0+cu130`, `vllm-omni 0.30.0`)
- **Model**: `ByteDance-Seed/BAGEL-7B-MoT` (`512×512`, `steps=20`, 28 `Qwen2MoTDecoderLayer` layers)
- **Verified Build / Run**: commit `bb92fec`, Kairos run `morphserve-p1-m4pre-20261121-r3` ([module4-report.json](file:///mnt/e/000_code/zyd-2026/morphserve/results/phase1-job1c-r3/module4-report.json))
- **Verdict**: **ALL ACCEPTANCE GATES PASSED (`status: "ok"`) — 100% BIT-EXACT SHA256 & AR TOKENS**

---

## 1. Implementation Summary

| Component | Source | Core Mechanism |
| --- | --- | --- |
| Deterministic Step & Layer Preemption Engine | [preemptive_scheduler.py](file:///mnt/e/000_code/zyd-2026/morphserve/src/morphserve/worker/preemptive_scheduler.py) | `ArDecodeTask`, `UnifiedPreemptiveScheduler` — exploits low-CV DiT compute horizon via async launch-horizon tracking inside `Qwen2MoTModel.forward`, computing analytical chunk budget `k = floor((SLO - t_AR - guard) / t_layer)` and yielding at exact layer boundaries in EDF order while pausing Module-1 gen-expert guards cleanly. Also implements `service_one_step_round()` for honest step-level preemption baselines. |
| Single-GPU Preemption & Stack Validator | [validate_module4_preemption.py](file:///mnt/e/000_code/zyd-2026/morphserve/src/morphserve/bench/validate_module4_preemption.py) | Compares `non_preemptive_fcfs`, `step_preemptive`, and `morphserve_layer_chunk` across `1.5× / 2.0× / 2.5×` unloaded AR TPOT SLOs on mixed DiT + mid-flight AR requests (`ar1@400ms`, `ar2@2500ms`), plus full simultaneous **Module 1 (`TriPool-VMM alpha=0.5`) + Module 3 (`FlashPrefix-CFG` + Loopback) + Module 4 (`layer_chunk`)** Turn-1 editing validation. |

---

## 2. Live A100-40GB Measured Results (`r3`, `bb92fec`)

### 2.1 Solo Unloaded Baselines & DiT Determinism

- **Unloaded AR Decode TPOT**: `P50 = 62.82 ms`, `P99 = 66.50 ms` → Anchored SLOs: **`1.5× = 94.2 ms`**, **`2.0× = 125.6 ms`**, **`2.5× = 157.1 ms`**.
- **Solo `512×512` DiT (20 steps)**: `4,962.73 ms`, CFG step mean `286.55 ms`, **step CV = `2.17%`**, per-layer P50 = `10.52 ms`, SHA256 `cf80eb13...d0134`.

### 2.2 Policy Comparison Under Concurrent DiT (`t=0`) + 2 Mid-Flight AR Requests (`ar1@400ms`, `ar2@2500ms`)

| Policy | SLO Target | Chunk `k` | Yields | `AR TTFT Max (ms)` | `TPOT P50 (ms)` | `TPOT P99 (ms)` | `SLO Attainment` | Switching Overhead | Bit-Exact (DiT + AR) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `non_preemptive_fcfs` | `125.6 ms` (`2.0×`) | `—` | `0` | `4,619.23` | `125.33` | `127.30` | `73.7%` (blocked 4.6s) | `-0.12%` | **Yes** |
| `step_preemptive` | `125.6 ms` (`2.0×`) | `step` | `18` | `350.65` | `346.13` | `417.75` | `15.8%` | `0.16%` | **Yes** |
| **`morphserve_layer_chunk_15x`** | **`94.2 ms` (`1.5×`)** | **`2`** | `40` | **`88.18`** (**`52.4×`** vs FCFS) | **`81.98`** | **`85.26`** | **`100.0%`** | **`-0.75%`** | **Yes** |
| **`morphserve_layer_chunk_20x`** | **`125.6 ms` (`2.0×`)** | **`5`** | `39` | **`116.56`** | **`110.98`** | `153.89` | **`94.7%`** | **`-2.08%`** | **Yes** |
| **`morphserve_layer_chunk_25x`** | **`157.1 ms` (`2.5×`)** | **`8`** | `39` | **`148.78`** | **`142.46`** | `159.23` | **`94.7%`** | **`-1.35%`** | **Yes** |

### 2.3 Full Single-GPU Stack Composition (Module 1 + Module 3 + Module 4 Simultaneously)

Running Turn-1 multi-turn editing (`5,946`-token shared prefix, `steps=12`) with active `TriPool-VMM` (`alpha=0.50` streaming remaining `14` layers + scoped FLUX VAE), `FlashPrefix-CFG` (`280` calls), and `layer_chunk` (`SLO = 125.6 ms`, concurrent `16`-token AR arriving at `t=500ms`):

- **Total Mixed Wall Time**: `4,529.04 ms`
- **Module 1 (`TriPool-VMM`)**: `alpha_start = 0.50` → `28/28` resident after Step 1, **`transient_mapped_handles_current = 0`**, **`overflow_creates = 0`**
- **Module 3 (`FlashPrefix-CFG`)**: `shared_prefix_len = 5,946`, `280` shared attention calls
- **Module 4 (`Layer-Chunk`)**: `yield_count = 16`, **`m4_ar_ttft_ms = 122.75 ms`**, **`m4_ar_tpot_p99_ms = 117.03 ms`**, **`m4_ar_tokens_bit_exact = true`**
- **Image Parity**: **`SSIM = 0.9936`**, **`PSNR = 40.89 dB`**, `MAE = 0.9447 / 255`.
