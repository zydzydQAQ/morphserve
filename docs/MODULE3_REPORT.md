# MorphServe Module 3 (`FlashPrefix-CFG` + LECP CoW KV + Zero-CPU Loopback) Verification Report

- **Date**: 2026-11-21
- **Hardware**: `NVIDIA A100-PCIE-40GB` (`jjjkairos` cluster, CUDA 13.0, Driver `570.133.20`, PyTorch `2.13.0+cu130`, `vllm-omni 0.30.0`)
- **Model**: `ByteDance-Seed/BAGEL-7B-MoT` (28 layers, `H_q = 28`, `H_kv = 4`, `D = 128`, 7:1 GQA)
- **Verified Build / Run**: commit `bb92fec`, Kairos run `morphserve-p1-m3cfg-20261121-r7` ([module3-report.json](file:///mnt/e/000_code/zyd-2026/morphserve/results/phase1-job1b-r7/module3-report.json))
- **Verdict**: **ALL ACCEPTANCE GATES PASSED (`status: "ok"`)**

---

## 1. Implementation Summary

| Component | Source | Core Mechanism |
| --- | --- | --- |
| Zero-Copy LECP Prefix & CoW Block Sharing | [lecp_cache.py](file:///mnt/e/000_code/zyd-2026/morphserve/src/morphserve/kernels/lecp_cache.py) | `detect_lecp_prefix_len`, `cow_fork_context`, `rebind_cow_prefix`, `SharedPrefixCache` — detects exact token- and pointer-level causal prefix across multi-branch CFG (`main`, `cfg_text`, `cfg_img`) without duplicating prefix KV pages in HBM. |
| `FlashPrefix-CFG` & LSE Merge Kernel | [cfg_prefix_attn.py](file:///mnt/e/000_code/zyd-2026/morphserve/src/morphserve/kernels/cfg_prefix_attn.py) | Native `vllm.vllm_flash_attn.flash_attn_varlen_func` (`vllm_varlen`) on shared prefix (single HBM prefix read across `main` + `cfg_text`) + compact unpadded tail varlen pass + cached `cu_seqlens` GPU tensors + fused Triton online-softmax LSE merge (`_merge_two_lse_fwd`, `BLOCK_R=64`). |
| Zero-CPU GPU In-Place Loopback | [gpu_loopback.py](file:///mnt/e/000_code/zyd-2026/morphserve/src/morphserve/kernels/gpu_loopback.py) | `decode_latent_to_u8_gpu`, `pil_bicubic_resize_gpu` (Pillow-exact Catmull-Rom `a=-0.5` separable matmul + horizontal uint8 clamp), `prepare_vae_images_gpu`, `prepare_vit_images_gpu`, `GpuInPlaceLoopback`. |
| Live Multi-Turn Validator | [validate_module3_cfg_loopback.py](file:///mnt/e/000_code/zyd-2026/morphserve/src/morphserve/bench/validate_module3_cfg_loopback.py) | 3-branch CFG microbenchmark vs real `BAGEL` `attn_noncausal`, CPU PNG/PIL vs GPU loopback parity, 4-turn interleaved `512×512` editing E2E, and same-context Turn-3 isolation. |

---

## 2. Live A100-40GB Measured Results (`r7`, `bb92fec`)

### 2.1 `FlashPrefix-CFG` (`vllm_varlen` + Triton LSE Merge) vs Stock `BAGEL` `attn_noncausal`

Active self-verified backend: **`vllm_varlen` (`cuda:0_hq28_hkv4_d128`)**.

| Workload Config | `Q_len` | `L_shared` | Stock (`ms`) | MorphServe (`ms`) | Speedup | KV HBM Read Reduction | Max Err vs FP32 (`Stock / MS`) | Min Cosine Sim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| `512px_turn1_2k` | `1,026` | `2,048` | `1.744` | **`1.359`** | **`1.28×`** | `1.80×` | `0.00049 / 0.00049` | `0.9999946` |
| `512px_turn2_4k` | `1,026` | `4,096` | `2.351` | **`1.916`** | **`1.23×`** | `2.14×` | `0.00049 / 0.00049` | `0.9999945` |
| `512px_turn4_8k` | `1,026` | `8,192` | `2.469` | **`2.083`** | **`1.19×`** | `2.45×` | `0.00024 / 0.00024` | `0.9999948` |
| `512px_turn8_16k` | `1,026` | `16,384` | `3.940` | **`3.597`** | **`1.10×`** | `2.68×` | `0.00024 / 0.00024` | `0.9999946` |
| `1024px_turn2_8k` | `4,098` | `8,192` | `9.810` | `10.300` | `0.95×` | `1.80×` | `0.00024 / 0.00024` | `0.9999945` |

### 2.2 Zero-CPU GPU In-Place Loopback vs Stock CPU PNG/PIL Pipeline

- **Image Processor Matched**: `SiglipImageProcessor` (`980×980`, `mean=[0.5,0.5,0.5]`, `std=[0.5,0.5,0.5]`).
- **Loopback Prep Latency (VAE `uint8` → Resize/Normalize)**: `74.99 ms` (CPU PIL/PNG) → **`2.55 ms`** (GPU In-Place) — **`29.4×` speedup** (end-to-end decode+prep `186.81 ms → 43.27 ms`).
- **Exactness Red Line**:
  - `u8_decode_max_abs_diff = 0` (**100% bit-exact**)
  - `vae_input_max_abs_diff = 1.19e-07` (**FP32 exact**)
  - `vit_tokens_cosine_sim = 0.9999982`
  - Greedy AR verification text: **`ar_text_exact_match = true`** (token-for-token identical output).

### 2.3 End-to-End 4-Turn Interleaved Generation (`512×512`, `steps=20`)

| Turn | Seq Len | Shared Prefix | Stock KV (`MiB`) | MorphServe KV (`MiB`) | KV HBM Saved | Fork (`Stock / MS ms`) | Gen (`Stock / MS ms`) | SSIM | PSNR (`dB`) |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | ---: | ---: |
| `0` | `18` | `0` | `1.97` | `1.97` | `0.0%` | `0.17 / 0.10` | `4,960.32 / 4,883.90` | `1.0000` | `99.99` |
| `1` | `5,965` | `5,946` | `652.42` | **`327.25`** | **`49.8%`** | `3.95 / 0.17` | `5,585.19 / 5,471.92` | `0.9925` | `40.01` |
| `2` | `11,910` | `11,893` | `1,302.66` | **`652.26`** | **`49.9%`** | `3.87 / 0.17` | `6,207.94 / 6,058.59` | `0.9764` | `34.12` |
| `3` | `17,857` | `17,838` | `1,953.11` | **`977.59`** | **`49.9%`** | `4.18 / 0.18` | `6,852.91 / 6,645.66` | `0.9691` | `33.37` |
| **Total** | — | — | — | — | **`~49.9%`** | — | **`23,606.36 → 23,060.07`** | — | — |

- **Same-Context Turn-3 Isolation (`17,857` tokens)**: Stock `6,852.91 ms` → MorphServe **`6,630.66 ms` (`1.034×`)**, **`SSIM = 0.996424`**, **`PSNR = 44.27 dB`**, `MAE = 0.7328 / 255`.
- **Total Loopback Overhead (Turns 0–3)**: `2,678.14 ms → 2,382.51 ms`.
