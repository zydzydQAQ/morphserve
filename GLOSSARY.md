# MorphServe Domain Glossary

Canonical domain and architectural vocabulary for `morphserve`. All modules, interfaces, tests, and design documents use these exact terms.

## Paradigm & Model Concepts

- **CPTG (Cross-Paradigm Transition Graph)**: Unified execution graph representation across autoregressive understanding/reasoning (`AR` / `VLM`) and diffusion generation (`DiT` / `MMDiT`) turns within one session.
  - **`tight_mot` (`BAGEL-7B-MoT`)**: Mixture-of-Transformers architecture where understanding and generation share token sequence state; cross-paradigm state is a multi-turn `NaiveCache` KV context (`main`, `cfg_text`, `cfg_img`).
  - **`bridge_mllm_dit` (`InternVL-U`)**: Bridge-coupled MLLM + MMDiT architecture where cross-paradigm state is a compact conditioning tensor bundle (`diffusion_inputs`, ~24–32 MiB) rather than full VLM KV history.
- **`CPTGModelAdapter`**: Deep model runtime module in `src/morphserve/models/` hiding model loading, runtime compatibility shims, turn preparation, `AR` token decoding, `DiT` denoising, `CPTGRequestSpec` construction, and flat `uint8` state pack/unpack behind one uniform interface (`BagelCPTGAdapter`, `InternVLUCPTGAdapter`, `SyntheticCPTGAdapter`).
- **`SessionContext`**: Opaque per-session worker state managed by a `CPTGModelAdapter` (`NaiveCache` + CoW prefix + GPU loopback tensor for `tight_mot`; conditioning bridge tensors + target grid dimensions for `bridge_mllm_dit`).

## Memory & Paging (`TriPool-VMM`)

- **`TriPoolVMM`**: Dual-tier CUDA Virtual Memory Manager (`csrc/vmm_tripool/` + `src/morphserve/vmm/tripool.py`) sharing one physical pool across three virtual address arenas: `2 MiB` Contiguous-VA Persistent KV regions (`tripool_kv_region_*`), `64 MiB` super-handle Scoped Transient Scratchpads (`MemPool` + `CUDAPluggableAllocator` for FLUX/Qwen VAE spikes), and `64 MiB` Opportunistic Secondary Expert slices with drop-clean eviction and background H2D streaming. Paired with `SimulatedTriPoolBackend` in CPU tests.
- **`VmmNaiveCache`**: Drop-in `NaiveCache` replacement (`src/morphserve/vmm/`) binding per-layer K/V tensors to growable contiguous `TriPoolVMM` Arena-1 VA regions (`2,048` tokens per `2 MiB` page for `BAGEL`), appending new tokens in-place without `torch.cat` reallocations and preserving base-pointer identity across `LECP CoW` forks. Shared identically by `morphserve` and `prism_tetriserve_coloc`.
- **`NumaMemfdMirror`**: Process-shareable, NUMA-bound (`mbind` / `numa_tonode_memory`), CUDA-pinned host weight mirror backed by `memfd_create` and `/proc/<leader_pid>/fd/<fd>` `MAP_SHARED` attachment (1 shared mirror per NUMA socket, bypassing `/dev/shm` caps).
- **`ExpertPartitionSpec`**: Declarative specification of which decoder blocks, parameter names, forward entrypoints, and optional `VAE` module belong to a pageable secondary expert (`BAGEL` 28-block generation expert; `InternVL-U` 20-block `GenerationDecoder`).
- **`LayerExpertPager`**: Unified `TriPool-VMM` expert pager (`src/morphserve/vmm/pager.py`) binding `Parameter.data` permanently to fixed Expert VA views, streaming missing blocks asynchronously during Step-1 compute (`expert_alpha`), and applying asymmetric transient `VAE` scoping (leaving borrowed tail expert slices lazily unmapped after `vae.decode` so `expert_alpha` adapts strictly on demand without synchronous H2D stalls).
- **`NaiveHostOffloader` & `StaticFissionDropper`**: Baseline secondary-weight lifecycle helpers (`src/morphserve/vmm/pager.py`) driven by the same `ExpertPartitionSpec` as `LayerExpertPager`, eliminating duplicated parameter-walking loops across `BagelCPTGAdapter` and `InternVLUCPTGAdapter`.
- **`expert_alpha`**: Fraction (`[0.0, 1.0]`) of secondary generation expert blocks currently mapped in GPU HBM on a worker.

## Execution Interception, Kernels & Preemption

- **`BagelExecutionInterceptor`**: Single permanent, `mode`-aware execution coordinator installed once per `BagelPipeline`, owning generation-only (`mode == "gen"`) `LayerExpertPager` wait gating, Step-0 `HalfNodeH2DPacer` slot release, multi-request varlen batched `AR` + `UnifiedPreemptiveScheduler` yields with immediate `on_task_complete` ACKs, and `BagelSharedPrefixPatch` activation.
- **`LECP CoW` (`SharedPrefixCache`)**: Longest Exact Causal Prefix Copy-on-Write KV sharing across `CFG` branches (`cow_fork_context`, `rebind_cow_prefix`), eliminating ~50% multi-turn KV HBM duplication with O(1) storage-pointer identity detection.
- **`FlashPrefix-CFG`**: Two-pass `FA2` shared-prefix attention + fused Triton online-softmax `LSE` merge, gated by `should_use_flash_prefix_split` (`L_shared >= max(256, 1.5 * Q_len)` for `Q_len <= 2048`; `L_shared >= 4 * Q_len` for `Q_len > 2048`), falling back losslessly to single-copy direct-workspace non-causal attention below threshold.
- **`GpuInPlaceLoopback`**: Zero-CPU `VAE -> uint8 -> Catmull-Rom PIL-exact bicubic resize -> ViT/VAE` loopback on GPU, eliminating PNG/PIL round-trips across interleaved turns.
- **`UnifiedPreemptiveScheduler`**: Deterministic sub-step (`layer_chunk`) and `step_preemptive` EDF scheduler yielding `DiT` denoising (`CV < 3%`) to protect interactive `AR` `TTFT` and `TPOT` SLOs with zero recomputation.

## Topology, Routing, Cluster Engine & Exactness

- **`ClusterTopologyProfile` & `OperatorLatencyProfile`**: Three-tier interconnect bandwidth/contention model (`intra_half` PCIe switch, `inter_socket` cross-NUMA, `inter_node` RDMA) paired with a single calibrated operator-latency and unloaded-SLO anchor profile (`T_unloaded`, `k×` normalized class SLOs) shared across `MorphServeClusterEngine`, `DualPathMorphRouter`, and `cluster_simulator.py`.
- **`DualPathMorphRouter`**: Stateful cluster router and placement authority (`src/morphserve/router/`) owning live/simulated `WorkerState` ledgers, initial GPU admission, static-fission rank pairing, `HalfNodeH2DPacer` slot reservations, and per-transition analytical routing between **Path A (`state_ship`)** over `DirectNcclMesh` and **Path B (`in_place_morph`)** via `LayerExpertPager` Step-1 H2D streaming.
- **`WorkerRuntime` & `WorkerFleet`**: `torch`-free per-worker command/session/preemption state machine (`WorkerRuntime`) exposed to `MorphServeClusterEngine` through two `WorkerFleet` adapters (`MpSpawnGpuWorkerFleet` over `mp.spawn` + `DirectNcclMesh`/POSIX SHM; `InProcessWorkerFleet` over `InMemoryByteBus` matching `DirectNcclMesh`'s `send_u8`/`recv_u8` interface in CPU integration tests).
- **`DirectNcclMesh`**: Standalone N-rank `libnccl.so.2` P2P byte transport on a dedicated CUDA transfer stream (`send_u8` / `recv_u8`), bypassing `torch.distributed` global group collisions.
- **`ExactnessEvaluator` (`Exactness Red Line`)**: Canonical verification module (`src/morphserve/bench/exactness.py`) enforcing **Bit-Exact `SHA256`** for `TriPool-VMM`, Dual-Path Morphing, and Preemptive Scheduling, and **Zero-Approximation Mathematical Equivalence** (`100%` greedy `AR` token match, `8x8` windowed `SSIM`, `PSNR`, `MAE_u8`) for `FlashPrefix-CFG` and `GpuInPlaceLoopback`.
