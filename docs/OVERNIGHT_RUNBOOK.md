# MorphServe Phase-2 Overnight Campaign Runbook (`2026-11-21` -> `2026-11-22`)

## 1. Objective & Scope
Execute, verify, and archive the complete Phase-2 OSDI/SOSP/ASPLOS evaluation campaign for `MorphServe` on `jjjkairos` (`user-jjjkairos`, 16-GPU A100-PCIE-40GB quota) unattended overnight, plus deep statistical hardening (`n=80` full suite & static fission split-ratio sensitivity), calibrated `8 -> 16 -> 64` GPU simulation, local artifact synchronization, and `docs/PHASE2_REPORT.md` publication.

## 2. Hard Safety Guardrails (Enforced on Every Check)
- **Zero Container Image Builds**: Strictly reuse `192.168.1.114:30500/jjjkairos/mllmserve:vllm-omni0.30.0-r2` + `kairos-toolkit/tools/deploy_build.sh --mode stage`.
- **Shared Storage Stop Lines (`/data`)**:
  - Pre-download gate for `InternVL-U` (`8.16 GiB`): require `/data < 790G` and `< 80%` (`current: 742G / 73%`).
  - Global alert line: `> 80%` -> halt new submissions and alert user.
  - Hard emergency stop (`AGENTS.md`): `> 810G` or `>= 82%` -> immediately stop all MorphServe submissions/transfers and report.
  - All ephemeral `HF_HOME`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, parquet unpacks, and temporary images stay strictly in container-local `/tmp` and are cleaned on `EXIT`.
- **Bounded Failure Recovery**:
  - Never delete or stop other jobs; preserve failed run directories (`/shared/output/MORPHSERVE/runs/<job>`).
  - Diagnose logs first, change at most one variable, publish a new immutable Git SHA, and resubmit under a new `-rN` job name (maximum **2 retries** per job stage to prevent runaway retry loops).

## 3. Three-Wave Overnight Execution Pipeline

### Wave 1 — Stage-1 Gate & 8-GPU Runtime Smoke (`Active Now`, `--max-active-jobs 1`)
1. **`Job 2a` (`morphserve-p2-m21gate-20261121-r4`, `1×1` A100, build `b1aaa84`)**:
   - Verify Module-3 `FlashPrefix-CFG` + direct-workspace `1024px` (`speedup >= 1.00x` across all 5 cases, `-49.9%` KV HBM, bit-exact GPU loopback).
   - Build `< 1 GiB` `Interleaved-Bench` (`main_mix_80`, `skew_ar80_dit20`, `skew_ar50_dit50`, `skew_ar20_dit80`, `smoke_8gpu_12`, `accuracy_geneval_mmvet`) + `BENCH_MANIFEST.json`.
2. **`Job 2b` (`morphserve-p2-m22smoke-20261121-r1`, `1×8` A100, build `b1aaa84`)**:
   - Validate all 6 policy presets (`morphserve`, `monolithic_dp`, `twostage_shm`, `static_nccl_fission`, `prism_tetriserve_coloc`, `naive_host_offload`) on `smoke_8gpu_12.json` (`0` OOM, `0` VMM overflow, `Host Shmem <= 32 GiB`, both Path A & Path B exercised).

### Wave 2 — Core Phase-2 Milestones `M2.3 + M2.4 + M2.5` (`--max-active-jobs 2`, 16/16 GPUs Concurrent)
Launched automatically as soon as Wave 1 passes:
1. **`Job 2c` (`morphserve-p2-m23a-20261121-r1`, `1×8` A100)**:
   - `BAGEL-7B-MoT` 4-quadrant QPS sweep (`0.4, 0.8, 1.3` rps × 6 policies) + 5-step M1–M4 component ablation ladder (`Full`, `-M4`, `-M3`, `-M2`, `-M1`).
2. **`Job 2d` (`morphserve-p2-m23b-20261121-r1`, `1×8` A100, runs concurrently with `Job 2c`)**:
   - Modality-skew sweep (`80:20`, `50:50`, `20:80` × 6 policies) + lossless output parity on `GenEval` (`SSIM`, `PSNR`, `SHA256`) and `MMVet` (`100%` greedy AR token match).
3. **`Job 2e` (`morphserve-p2-m24ivlu-20261121-r1`, `1×8` A100, backfills as soon as `Job 2d` frees 8 GPUs)**:
   - Mirror fetch + SHA256 verify `InternVL-U/InternVL-U` (`8.16 GiB`) + live `8×A100` `bridge_mllm_dit` generality (`InternVLUDecoderPager` + `DirectNcclMesh` compact bridge shipping).
4. **`Job 2f` (`morphserve-p2-m25node2x8-20261121-r1`, `2×8 = 16` A100)**:
   - Live 2-node 16-GPU fabric benchmark (intra-half PCIe vs cross-socket vs cross-node fabric) + automatic suppression verification of cross-node 1 GiB KV shipping.

### Wave 3 — Overnight Deep Statistical Hardening & Final Synthesis (`Post-Wave-2`, Optional Deepening + Mandatory Report)
1. **`Job 2g` (Full-80-Session Deep Sweep + Static Split Ratio Sensitivity `2:6 / 4:4 / 6:2`)**:
   - Executed if `/data < 78%` and cluster slots remain healthy after Wave 2: runs full un-subsampled `main_mix_80.json` (`n=80`) + sweeps static Und:Gen ratios (`2:6`, `4:4`, `6:2`) against `morphserve` elastic `alpha` to eliminate static-partition tuning bias.
2. **Local Sync, Calibrated 64-GPU Simulator & Final Archival**:
   - `rsync -av` all remote run directories (`JSON` & `.log` reports only) into `/mnt/e/000_code/zyd-2026/morphserve/results/`.
   - Execute `python3 -m morphserve.sim.cluster_simulator --output-dir results/phase2-m25-sim --real-report results/<m23a>/m23-suite-a-report.json` (`< 5%` MAPE gate across `8 -> 16 -> 64` GPUs and PCIe Gen4 / A100-NVLink / H100-NVLink + RDMA).
   - Run `pytest -q tests`, compile `/mnt/e/000_code/zyd-2026/morphserve/docs/PHASE2_REPORT.md`, commit cleanly, and cancel the active overnight cron supervisor.
