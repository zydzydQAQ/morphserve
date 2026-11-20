# Module 3: Exact Shared-Prefix CFG Attention & GPU In-Place Loopback

Triton-first kernels (with optional FlashInfer/CUDA fast paths if supported by the verified runtime image):
- `lecp_block_table.py`: Longest Exact Causal Prefix (LECP) CoW block table across 2–3 CFG branches.
- `cfg_shared_prefix_triton.py`: Single-HBM-read grouped-query shared-prefix + branch-suffix attention kernel.
- `gpu_loopback_triton.py`: Zero-CPU VAE→`clamp+round+uint8`→PIL-equivalent resize→ViT loopback kernel.
