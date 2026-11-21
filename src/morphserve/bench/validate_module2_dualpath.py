"""Phase-1 Module 2 live 8xA100 validation: Shared NUMA memfd, Half-Node H2D Pacer, and Dual-Path Router."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from morphserve.bench.validate_module1_bagel import load_bagel_pipeline
from morphserve.router import (
    CPTGRequestSpec,
    ClusterTopologyProfile,
    DirectNcclP2PChannel,
    DualPathMorphRouter,
    HalfNodeH2DPacer,
    NcclStateShipper,
    SharedNumaMemfdMirror,
    WorkerState,
    find_free_tcp_port,
    probe_live_topology,
)
from morphserve.vmm import BagelExpertPager, TriPoolVMM

MIB = 1024**2
GIB = 1024**3


def _pil_sha256(pil_img: Any) -> str:
    import numpy as np

    arr = np.ascontiguousarray(np.asarray(pil_img.convert("RGB"), dtype=np.uint8))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _read_shmem_kib() -> int:
    with open("/proc/meminfo", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("Shmem:"):
                return int(line.split()[1])
    return 0


def _test1_worker(rank: int, world_size: int, master_port: int, payload_bytes: int, out_file: str) -> None:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    mirror: SharedNumaMemfdMirror | None = None
    try:
        topo = ClusterTopologyProfile.a100_pcie_8gpu_default()
        numa_node = topo.numa_node_of(rank)
        is_leader = rank in (0, 4)
        sentinel = f"NUMA{numa_node}_EXPERT_MIRROR_VERIFY_V1".encode("utf-8")

        dist.barrier()
        shmem_before_kib = _read_shmem_kib() if rank == 0 else 0
        dist.barrier()

        my_meta: dict[str, Any] | None = None
        if is_leader:
            mirror = SharedNumaMemfdMirror.create(
                nbytes=payload_bytes,
                numa_node=numa_node,
                name=f"morphserve_m2_numa{numa_node}",
                populate_bytes=sentinel,
            )
            my_meta = {"rank": rank, "pid": os.getpid(), "fd": mirror.fd, "inode": mirror.inode, "numa": numa_node}

        all_leaders = [None] * world_size
        dist.all_gather_object(all_leaders, my_meta)
        leader_map = {m["numa"]: m for m in all_leaders if m is not None}

        if not is_leader:
            lm = leader_map[numa_node]
            mirror = SharedNumaMemfdMirror.attach_proc_fd(
                leader_pid=lm["pid"],
                leader_fd=lm["fd"],
                nbytes=payload_bytes,
                numa_node=numa_node,
            )

        assert mirror is not None
        inode_match = mirror.inode == leader_map[numa_node]["inode"]
        sentinel_match = bytes(mirror.mm[: len(sentinel)]) == sentinel
        mirror.pin_cuda()

        dist.barrier()
        shmem_after_kib = _read_shmem_kib() if rank == 0 else 0
        dist.barrier()

        # Allocate GPU destination buffer for H2D streaming tests (4,480 MiB = 10 BAGEL layers)
        dma_bytes = min(payload_bytes, 4480 * MIB)
        dst_buf = torch.empty((dma_bytes,), dtype=torch.uint8, device=device)

        # Warmup CUDA H2D stream cleanly on every rank
        _ = mirror.stream_h2d_chunks(dst_buf[: 128 * MIB])

        # Benchmark 1: Unpaced 4-GPU same-half H2D contention on Half-0 ({0, 1, 2, 3})
        dist.barrier()
        if rank < 4:
            t_u0 = time.perf_counter()
            unpaced_bw = mirror.stream_h2d_chunks(dst_buf)
            t_unpaced_ms = (time.perf_counter() - t_u0) * 1000.0
        else:
            unpaced_bw = 0.0
            t_unpaced_ms = 0.0
        dist.barrier()

        # Benchmark 2: HalfNodeH2DPacer across BOTH half-nodes ({0,1} on Half-0 and {4,5} on Half-1)
        pacer = HalfNodeH2DPacer(topo, max_concurrent_per_half=1)
        dist.barrier()
        paced_bw = 0.0
        t_paced_ms = 0.0
        if rank in (0, 1, 4, 5):
            # Force sequential slots inside each half-node while running Half-0 & Half-1 in parallel
            t_wait0 = time.perf_counter()
            with pacer.acquire_process_slot(rank):
                t_dma0 = time.perf_counter()
                paced_bw = mirror.stream_h2d_chunks(dst_buf)
                t_paced_ms = (time.perf_counter() - t_dma0) * 1000.0
            total_slot_ms = (time.perf_counter() - t_wait0) * 1000.0
        else:
            total_slot_ms = 0.0
        dist.barrier()

        row = {
            "rank": rank,
            "numa_node": numa_node,
            "inode_match": inode_match,
            "sentinel_match": sentinel_match,
            "unpaced_half0_bw_gbps": round(unpaced_bw, 2),
            "unpaced_half0_ms": round(t_unpaced_ms, 2),
            "paced_bw_gbps": round(paced_bw, 2),
            "paced_dma_ms": round(t_paced_ms, 2),
            "paced_wait_plus_dma_ms": round(total_slot_ms, 2),
        }
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(row, gathered, dst=0)

        if rank == 0:
            half0_bws = [r["unpaced_half0_bw_gbps"] for r in gathered if r["rank"] < 4]
            half0_ms = [r["unpaced_half0_ms"] for r in gathered if r["rank"] < 4]
            paced_bws = [r["paced_bw_gbps"] for r in gathered if r["rank"] in (0, 1, 4, 5)]
            paced_wall_ms = max(r["paced_wait_plus_dma_ms"] for r in gathered if r["rank"] in (0, 1, 4, 5))
            shmem_delta_gib = max(0.0, (shmem_after_kib - shmem_before_kib) / (1024**2))
            mirror_gib = payload_bytes / GIB
            naive_8gpu_gib = round(8.0 * mirror_gib, 2)
            shared_2socket_gib = round(2.0 * mirror_gib, 2)
            mean_unpaced = sum(half0_bws) / len(half0_bws)
            mean_paced = sum(paced_bws) / len(paced_bws)
            summary = {
                "world_size": world_size,
                "payload_per_mirror_gib": round(mirror_gib, 2),
                "naive_8gpu_host_ram_gib": naive_8gpu_gib,
                "shared_2socket_host_ram_gib": shared_2socket_gib,
                "observed_shmem_delta_gib": round(shmem_delta_gib, 2),
                "host_ram_savings_x": round(naive_8gpu_gib / shared_2socket_gib, 2),
                "all_inodes_matched": all(r["inode_match"] for r in gathered),
                "all_sentinels_matched": all(r["sentinel_match"] for r in gathered),
                "unpaced_4gpu_half0_agg_gbps": round(sum(half0_bws), 2),
                "unpaced_4gpu_half0_per_gpu_gbps": round(mean_unpaced, 2),
                "unpaced_4gpu_half0_wall_ms": round(max(half0_ms), 2),
                "paced_dual_half_per_gpu_gbps": round(mean_paced, 2),
                "paced_dual_half_4req_wall_ms": round(paced_wall_ms, 2),
                "h2d_per_gpu_speedup_x": round(mean_paced / max(mean_unpaced, 1e-6), 2),
                "ranks": gathered,
            }
            Path(out_file).write_text(json.dumps(summary, indent=2), encoding="utf-8")

        del dst_buf
        dist.barrier()
    finally:
        if mirror is not None:
            mirror.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _prepare_text_ctx(pipeline, prompt: str) -> dict[str, Any]:
    import torch
    from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache

    num_layers = pipeline.bagel.config.llm_config.num_hidden_layers
    ctx = {"kv_lens": [0], "ropes": [0], "past_key_values": NaiveCache(num_layers)}
    inp, nl, nr = pipeline.bagel.prepare_prompts(
        curr_kvlens=ctx["kv_lens"],
        curr_rope=ctx["ropes"],
        prompts=[prompt],
        tokenizer=pipeline.tokenizer,
        new_token_ids=pipeline.new_token_ids,
    )
    inp = {k: (v.to(pipeline.device) if torch.is_tensor(v) else v) for k, v in inp.items()}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ctx["past_key_values"] = pipeline.bagel.forward_cache_update_text(ctx["past_key_values"], **inp)
    ctx["kv_lens"], ctx["ropes"] = nl, nr
    return ctx


def _append_image_to_ctx(pipeline, ctx: dict[str, Any], pil_img: Any, prompt: str) -> None:
    import numpy as np
    import torch

    def vae_tf(img):
        arr = torch.from_numpy(np.array(img, copy=True)).float() / 127.5 - 1.0
        return arr.permute(2, 0, 1)

    def vit_tf(img):
        return pipeline.image_processor(images=img, return_tensors="pt").pixel_values[0]

    vi, nl, nr = pipeline.bagel.prepare_vae_images(
        curr_kvlens=ctx["kv_lens"], curr_rope=ctx["ropes"], images=[pil_img], transforms=vae_tf, new_token_ids=pipeline.new_token_ids
    )
    vi = {k: (v.to(pipeline.device) if torch.is_tensor(v) else v) for k, v in vi.items()}
    vti, nlv, nrv = pipeline.bagel.prepare_vit_images(
        curr_kvlens=nl, curr_rope=nr, images=[pil_img], transforms=vit_tf, new_token_ids=pipeline.new_token_ids
    )
    vti = {k: (v.to(pipeline.device) if torch.is_tensor(v) else v) for k, v in vti.items()}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ctx["past_key_values"] = pipeline.bagel.forward_cache_update_vae(pipeline.vae, ctx["past_key_values"], **vi)
        ctx["kv_lens"], ctx["ropes"] = nl, nr
        ctx["past_key_values"] = pipeline.bagel.forward_cache_update_vit(ctx["past_key_values"], **vti)
        ctx["kv_lens"], ctx["ropes"] = nlv, nrv

    inp, nlt, nrt = pipeline.bagel.prepare_prompts(
        curr_kvlens=ctx["kv_lens"], curr_rope=ctx["ropes"], prompts=[prompt], tokenizer=pipeline.tokenizer, new_token_ids=pipeline.new_token_ids
    )
    inp = {k: (v.to(pipeline.device) if torch.is_tensor(v) else v) for k, v in inp.items()}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ctx["past_key_values"] = pipeline.bagel.forward_cache_update_text(ctx["past_key_values"], **inp)
    ctx["kv_lens"], ctx["ropes"] = nlt, nrt


def _run_bagel_dit(pipeline, main_ctx, cfg_text_ctx, cfg_img_ctx, prompt: str, steps: int = 20, seed: int = 52, hw: int = 512):
    import torch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sampling = OmniDiffusionSamplingParams(
        height=hw,
        width=hw,
        num_inference_steps=steps,
        seed=seed,
        extra_args={
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.5,
            "cfg_interval": [0.4, 1.0],
            "cfg_renorm_type": "global",
            "cfg_renorm_min": 0.0,
            "timestep_shift": 3.0,
        },
    )
    sampling.past_key_values = main_ctx["past_key_values"]
    sampling.kv_metadata = {"ropes": main_ctx["ropes"], "image_shape": [hw, hw]}
    if cfg_text_ctx is not None:
        sampling.cfg_text_past_key_values = cfg_text_ctx["past_key_values"]
        sampling.cfg_text_kv_metadata = {"ropes": cfg_text_ctx["ropes"], "image_shape": [hw, hw]}
    if cfg_img_ctx is not None:
        sampling.cfg_img_past_key_values = cfg_img_ctx["past_key_values"]
        sampling.cfg_img_kv_metadata = {"ropes": cfg_img_ctx["ropes"], "image_shape": [hw, hw]}
    torch.cuda.synchronize(pipeline.device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = pipeline._forward_single({"prompt": prompt, "modalities": ["image"]}, sampling)
    torch.cuda.synchronize(pipeline.device)
    return out.output["payload"]["image"], (time.perf_counter() - t0) * 1000.0


def _ctx_nbytes(ctx: dict[str, Any]) -> int:
    pkv = ctx["past_key_values"]
    total = 0
    for k, v in zip(pkv.key_cache, pkv.value_cache):
        if k is not None:
            total += int(k.numel() * k.element_size())
        if v is not None:
            total += int(v.numel() * v.element_size())
    return total


def _test2_dualpath_worker(rank: int, world_size: int, master_port: int, model_dir_str: str, steps: int, out_file: str) -> None:
    import torch
    import torch.distributed as dist
    from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    pipeline = None
    channel: DirectNcclP2PChannel | None = None
    pager = None
    vmm = None
    try:
        model_dir = Path(model_dir_str)
        pipeline = load_bagel_pipeline(model_dir, device_index=rank)
        channel = DirectNcclP2PChannel(rank=rank, world_size=world_size, master_port=master_port)
        shipper = NcclStateShipper(channel=channel)

        prompt0 = "A serene wooden cabin beside an alpine lake at golden hour, realistic photography."
        edit_p = "Add a warm lantern near the cabin door and gentle ripples on the lake."

        # Build contexts + reference outputs BEFORE attaching TriPool-VMM so baselines match stock BAGEL cleanly
        ctx_t0_main = _prepare_text_ctx(pipeline, prompt0)
        ctx_t0_img = _prepare_text_ctx(pipeline, prompt0)

        # Warmup DiT once on both ranks
        _ = _run_bagel_dit(pipeline, ctx_t0_main, None, ctx_t0_img, prompt0, steps=2, seed=52, hw=512)
        channel.barrier()

        if rank == 1:
            ref_t0_img, stock_t0_gen_ms = _run_bagel_dit(pipeline, ctx_t0_main, None, ctx_t0_img, prompt0, steps=steps, seed=52, hw=512)
            ref_t0_sha = _pil_sha256(ref_t0_img)

        # Build genuine 3-turn (~18k tokens, ~1.92 GiB) BAGEL context on Rank 0 & Rank 1 reference
        ctx_long_main = deepcopy(ctx_t0_main)
        if rank == 0:
            t0_ref_r0, _ = _run_bagel_dit(pipeline, ctx_t0_main, None, ctx_t0_img, prompt0, steps=steps, seed=52, hw=512)
            img_for_hist = t0_ref_r0
        else:
            img_for_hist = ref_t0_img
        for k_turn in range(3):
            _append_image_to_ctx(pipeline, ctx_long_main, img_for_hist, f"{edit_p} (turn {k_turn})")
        ctx_long_text = deepcopy(ctx_long_main)
        ctx_long_img = _prepare_text_ctx(pipeline, f"{edit_p} (turn 3)")
        _append_image_to_ctx(pipeline, ctx_long_main, img_for_hist, f"{edit_p} (turn 3)")
        # Measure steady 512px Turn-3 solo reference SHA & time on Rank 1
        if rank == 1:
            ref_t3_img, stock_t3_gen_ms = _run_bagel_dit(
                pipeline, ctx_long_main, ctx_long_text, ctx_long_img, f"{edit_p} (turn 3)", steps=steps, seed=61, hw=512
            )
            ref_t3_sha = _pil_sha256(ref_t3_img)
            del ref_t0_img, ref_t3_img

        channel.barrier()
        # Rank 0 ships NCCL warmup tensor to Rank 1 so NCCL P2P buffers are pre-warmed
        if rank == 0:
            _ = shipper.send_bagel_context(ctx_t0_main, dst_rank=1)
        else:
            _w_ctx, _ = shipper.recv_bagel_context(src_rank=0, naive_cache_cls=NaiveCache, device=device)
            del _w_ctx

        # ---- Regime 1: Turn-0 Short KV (~2 MiB) Path A (NcclStateShipper) vs Path B (TriPool-VMM alpha=0.75) ----
        channel.barrier()
        if rank == 0:
            t_a0 = time.perf_counter()
            m_t0 = shipper.send_bagel_context(ctx_t0_main, dst_rank=1)
            channel.barrier()  # Wait for Rank 1 finish timestamp barrier
            path_a_t0_total_ms = (time.perf_counter() - t_a0) * 1000.0
        else:
            recv_t0_main, m_t0 = shipper.recv_bagel_context(src_rank=0, naive_cache_cls=NaiveCache, device=device)
            img_a_t0, gen_a_t0_ms = _run_bagel_dit(pipeline, recv_t0_main, None, ctx_t0_img, prompt0, steps=steps, seed=52, hw=512)
            sha_a_t0 = _pil_sha256(img_a_t0)
            del recv_t0_main, img_a_t0
            channel.barrier()

        # ---- Regime 2: Long 4-turn KV (~1.92 GiB) Path A (NcclStateShipper gamma=2 interleaved) ----
        channel.barrier()
        if rank == 0:
            m_long_fwd = shipper.send_bagel_context(ctx_long_main, dst_rank=1)
            # Also ship back updated KV state after DiT roundtrip (gamma=2 interleaved!)
            recv_back, m_long_back = shipper.recv_bagel_context(src_rank=1, naive_cache_cls=NaiveCache, device=device)
            del recv_back
            channel.barrier()
        else:
            recv_long, m_long_fwd = shipper.recv_bagel_context(src_rank=0, naive_cache_cls=NaiveCache, device=device)
            img_a_t3, gen_a_t3_ms = _run_bagel_dit(
                pipeline, recv_long, ctx_long_text, ctx_long_img, f"{edit_p} (turn 3)", steps=steps, seed=61, hw=512
            )
            sha_a_t3 = _pil_sha256(img_a_t3)
            m_long_back = shipper.send_bagel_context(recv_long, dst_rank=0)
            del recv_long, img_a_t3
            channel.barrier()

        # ---- Path B Execution on Rank 0 using TriPool-VMM (`alpha = 0.75` and `1024px alpha = 0.50`) ----
        r0_results: dict[str, Any] = {}
        if rank == 0:
            # Attach TriPool-VMM on Rank 0
            pager = BagelExpertPager(pipeline, role="generation", numa_node=0, super_handle_mib=64, wrap_vae=True)
            pager.offload_to_host_mirror()
            vmm = TriPoolVMM(
                device=0,
                super_pool_mib=16384,
                fine_kv_pool_mib=512,
                super_handle_mib=64,
                transient_va_gib=16,
                kv_va_gib=8,
                allow_overflow=False,
            )
            pager.bind_vmm(vmm, initial_alpha=0.75)

            # Turn-0 Path B (alpha=0.75 -> ~3.11 GiB H2D)
            pager.set_residency_alpha(0.75)
            torch.cuda.synchronize(device)
            with pager.generation_mode(prefetch_all=True):
                img_b_t0, gen_b_t0_ms = _run_bagel_dit(pipeline, ctx_t0_main, None, ctx_t0_img, prompt0, steps=steps, seed=52, hw=512)
            sha_b_t0 = _pil_sha256(img_b_t0)
            del img_b_t0

            # Long Turn-3 Path B (alpha=0.75)
            pager.set_residency_alpha(0.75)
            torch.cuda.synchronize(device)
            with pager.generation_mode(prefetch_all=True):
                img_b_t3, gen_b_t3_ms = _run_bagel_dit(
                    pipeline, ctx_long_main, ctx_long_text, ctx_long_img, f"{edit_p} (turn 3)", steps=steps, seed=61, hw=512
                )
            sha_b_t3 = _pil_sha256(img_b_t3)
            del img_b_t3

            # Measure pure alpha=1.0 vs alpha=0.75 morph stall on Turn-3 + Turn-0
            pager.set_residency_alpha(1.0)
            with pager.generation_mode(prefetch_all=True):
                _, gen_local_t0_ms = _run_bagel_dit(pipeline, ctx_t0_main, None, ctx_t0_img, prompt0, steps=steps, seed=52, hw=512)
                _, gen_local_t3_ms = _run_bagel_dit(
                    pipeline, ctx_long_main, ctx_long_text, ctx_long_img, f"{edit_p} (turn 3)", steps=steps, seed=61, hw=512
                )

            r0_results = {
                "t0_ctx_mib": round(_ctx_nbytes(ctx_t0_main) / MIB, 2),
                "long_ctx_mib": round(_ctx_nbytes(ctx_long_main) / MIB, 2),
                "long_seq_len": int(ctx_long_main["past_key_values"].seq_lens),
                "path_b_t0_gen_ms": round(gen_b_t0_ms, 2),
                "path_b_t0_morph_overhead_ms": round(max(0.0, gen_b_t0_ms - gen_local_t0_ms), 2),
                "path_b_t0_sha256": sha_b_t0,
                "path_b_t3_gen_ms": round(gen_b_t3_ms, 2),
                "path_b_t3_morph_overhead_ms": round(max(0.0, gen_b_t3_ms - gen_local_t3_ms), 2),
                "path_b_t3_sha256": sha_b_t3,
                "m_t0_ship_ms": m_t0.transfer_ms,
                "m_t0_ship_gbps": m_t0.bandwidth_gbps,
                "m_long_ship_fwd_ms": m_long_fwd.transfer_ms,
                "m_long_ship_fwd_gbps": m_long_fwd.bandwidth_gbps,
                "m_long_ship_back_ms": m_long_back.transfer_ms,
                "m_long_ship_gamma2_ms": round(m_long_fwd.transfer_ms + m_long_back.transfer_ms, 2),
            }

        channel.barrier()
        r1_info = (
            {
                "ref_t0_sha": ref_t0_sha,
                "ref_t3_sha": ref_t3_sha,
                "sha_a_t0": sha_a_t0,
                "sha_a_t3": sha_a_t3,
                "gen_a_t0_ms": round(gen_a_t0_ms, 2),
                "gen_a_t3_ms": round(gen_a_t3_ms, 2),
                "stock_t0_gen_ms": round(stock_t0_gen_ms, 2),
                "stock_t3_gen_ms": round(stock_t3_gen_ms, 2),
            }
            if rank == 1
            else None
        )
        gathered = channel.exchange_json(r1_info if rank == 1 else r0_results)

        if rank == 0:
            r0 = gathered[0]
            r1 = gathered[1]
            # Evaluate router decisions on live ClusterTopologyProfile
            topo = ClusterTopologyProfile.a100_pcie_8gpu_default()
            router = DualPathMorphRouter(topo)
            workers = {
                0: WorkerState(gpu_id=0, role="hybrid", expert_alpha=0.75, kv_free_bytes=16 * GIB),
                1: WorkerState(gpu_id=1, role="gen", expert_alpha=1.00, kv_free_bytes=16 * GIB),
            }
            dec_short = router.decide(
                CPTGRequestSpec(
                    req_id="turn0-short",
                    src_gpu=0,
                    architecture="tight_mot",
                    kv_bytes=int(r0["t0_ctx_mib"] * MIB),
                    dit_step1_ms=340.0,
                    interleaved_roundtrip_gamma=1,
                ),
                workers,
            )
            dec_long = router.decide(
                CPTGRequestSpec(
                    req_id="turn3-long",
                    src_gpu=0,
                    architecture="tight_mot",
                    kv_bytes=int(r0["long_ctx_mib"] * MIB),
                    dit_step1_ms=340.0,
                    interleaved_roundtrip_gamma=2,
                ),
                workers,
            )
            payload = {
                "t0_short_kv": {
                    "kv_mib": r0["t0_ctx_mib"],
                    "router_decision": dec_short.path,
                    "router_dst_gpu": dec_short.dst_gpu,
                    "path_a_ship_ms": r0["m_t0_ship_ms"],
                    "path_a_bw_gbps": r0["m_t0_ship_gbps"],
                    "path_b_morph_overhead_ms": r0["path_b_t0_morph_overhead_ms"],
                    "path_a_bit_exact": r1["sha_a_t0"] == r1["ref_t0_sha"],
                    "path_b_bit_exact": r0["path_b_t0_sha256"] == r1["ref_t0_sha"],
                },
                "t3_long_interleaved_kv": {
                    "seq_len": r0["long_seq_len"],
                    "kv_mib": r0["long_ctx_mib"],
                    "router_decision": dec_long.path,
                    "router_dst_gpu": dec_long.dst_gpu,
                    "path_a_gamma2_ship_ms": r0["m_long_ship_gamma2_ms"],
                    "path_a_bw_gbps": r0["m_long_ship_fwd_gbps"],
                    "path_b_morph_overhead_ms": r0["path_b_t3_morph_overhead_ms"],
                    "path_a_bit_exact": r1["sha_a_t3"] == r1["ref_t3_sha"],
                    "path_b_bit_exact": r0["path_b_t3_sha256"] == r1["ref_t3_sha"],
                },
            }
            Path(out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")

        channel.barrier()
    finally:
        if channel is not None:
            channel.close()
        if pager is not None:
            pager.close()
        if vmm is not None:
            vmm.shutdown()
        if dist.is_initialized():
            dist.destroy_process_group()


def _test3_mixed_cptg_burst(topo: ClusterTopologyProfile) -> dict[str, Any]:
    router = DualPathMorphRouter(topo, pacer=HalfNodeH2DPacer(topo, max_concurrent_per_half=1))
    workers = {
        0: WorkerState(gpu_id=0, role="und", expert_alpha=0.50, kv_free_bytes=16 * GIB),
        1: WorkerState(gpu_id=1, role="gen", expert_alpha=1.00, kv_free_bytes=16 * GIB),
        2: WorkerState(gpu_id=2, role="hybrid", expert_alpha=0.75, kv_free_bytes=16 * GIB),
        3: WorkerState(gpu_id=3, role="gen", expert_alpha=1.00, kv_free_bytes=16 * GIB),
        4: WorkerState(gpu_id=4, role="und", expert_alpha=0.50, kv_free_bytes=16 * GIB),
        5: WorkerState(gpu_id=5, role="gen", expert_alpha=1.00, kv_free_bytes=16 * GIB),
        6: WorkerState(gpu_id=6, role="hybrid", expert_alpha=0.75, kv_free_bytes=16 * GIB),
        7: WorkerState(gpu_id=7, role="gen", expert_alpha=1.00, kv_free_bytes=16 * GIB),
    }
    burst = [
        CPTGRequestSpec(req_id="req0_bagel_t0_short", src_gpu=0, architecture="tight_mot", kv_bytes=4 * MIB, dit_step1_ms=287.0),
        CPTGRequestSpec(req_id="req1_internvl_u_bridge", src_gpu=0, architecture="bridge_mllm_dit", bridge_bytes=24 * MIB, kv_bytes=2 * GIB, dit_step1_ms=310.0, interleaved_roundtrip_gamma=2),
        CPTGRequestSpec(req_id="req2_bagel_t4_long", src_gpu=2, architecture="tight_mot", kv_bytes=1950 * MIB, dit_step1_ms=340.0, interleaved_roundtrip_gamma=2),
        CPTGRequestSpec(req_id="req3_bagel_1024px", src_gpu=4, architecture="tight_mot", kv_bytes=1200 * MIB, dit_step1_ms=1650.0, resolution=1024, interleaved_roundtrip_gamma=2),
        CPTGRequestSpec(req_id="req4_internvl_u_h1", src_gpu=4, architecture="bridge_mllm_dit", bridge_bytes=32 * MIB, kv_bytes=3 * GIB, dit_step1_ms=310.0, interleaved_roundtrip_gamma=2),
        CPTGRequestSpec(req_id="req5_bagel_t3_long_h1", src_gpu=6, architecture="tight_mot", kv_bytes=2100 * MIB, dit_step1_ms=340.0, interleaved_roundtrip_gamma=2),
        CPTGRequestSpec(req_id="req6_bagel_t0_spill", src_gpu=0, architecture="tight_mot", kv_bytes=6 * MIB, dit_step1_ms=287.0),
        CPTGRequestSpec(req_id="req7_bagel_1024_h0", src_gpu=0, architecture="tight_mot", kv_bytes=1500 * MIB, dit_step1_ms=1650.0, resolution=1024, interleaved_roundtrip_gamma=2),
    ]
    decs = router.route_batch(burst, workers)
    return {
        "decisions": [
            {
                "req_id": d.req_id,
                "path": d.path,
                "src_gpu": d.src_gpu,
                "dst_gpu": d.dst_gpu,
                "t_ship_ms": d.t_ship_ms,
                "t_morph_ms": d.t_morph_ms,
                "reason": d.reason,
            }
            for d in decs
        ],
        "bridge_all_shipped": all(d.path == "state_ship" for d in decs if "internvl" in d.req_id),
        "short_all_shipped": all(d.path == "state_ship" for d in decs if "_t0_" in d.req_id),
        "long_and_1024_all_in_place": all(d.path == "in_place_morph" for d in decs if ("_long" in d.req_id or "_1024" in d.req_id)),
    }


def run_validation(model_dir: Path, output_dir: Path, steps: int = 20) -> dict[str, Any]:
    import torch
    import torch.multiprocessing as mp

    output_dir.mkdir(parents=True, exist_ok=True)
    # Count visible GPUs without initializing a CUDA context in the parent process
    nvidia_gpus_dir = Path("/proc/driver/nvidia/gpus")
    if nvidia_gpus_dir.exists():
        gpu_count = len(list(nvidia_gpus_dir.iterdir()))
    else:
        gpu_count = int(torch.cuda.device_count())
    if gpu_count < 8:
        raise RuntimeError(f"Module 2 8-GPU validation requires 8 CUDA devices, saw {gpu_count}")

    port1 = find_free_tcp_port()
    test1_json = output_dir / "_tmp_test1_8gpu.json"
    # Allocate full 12,544 MiB (12.25 GiB) BAGEL Generation Expert mirror per NUMA socket (2x total vs 8x = 4.0x savings!)
    payload_bytes = 12_544 * MIB
    mp.spawn(_test1_worker, args=(8, port1, payload_bytes, str(test1_json)), nprocs=8, join=True)
    test1_res = json.loads(test1_json.read_text(encoding="utf-8"))
    test1_json.unlink(missing_ok=True)

    port2 = find_free_tcp_port()
    test2_json = output_dir / "_tmp_test2_dualpath.json"
    mp.spawn(_test2_dualpath_worker, args=(2, port2, str(model_dir), steps, str(test2_json)), nprocs=2, join=True)
    test2_res = json.loads(test2_json.read_text(encoding="utf-8"))
    test2_json.unlink(missing_ok=True)

    # Probe live 8-GPU topology AFTER child workers exit so parent CUDA context stays idle during Tests 1 & 2
    topo = probe_live_topology(quick_p2p_mib=128)
    test3_res = _test3_mixed_cptg_burst(topo)

    all_pass = (
        test1_res["all_inodes_matched"]
        and test1_res["all_sentinels_matched"]
        and test1_res["host_ram_savings_x"] == 4.0
        and test1_res["h2d_per_gpu_speedup_x"] >= 2.5
        and test2_res["t0_short_kv"]["path_a_bit_exact"]
        and test2_res["t0_short_kv"]["path_b_bit_exact"]
        and test2_res["t0_short_kv"]["router_decision"] == "state_ship"
        and test2_res["t0_short_kv"]["path_a_ship_ms"] < test2_res["t0_short_kv"]["path_b_morph_overhead_ms"]
        and test2_res["t3_long_interleaved_kv"]["path_a_bit_exact"]
        and test2_res["t3_long_interleaved_kv"]["path_b_bit_exact"]
        and test2_res["t3_long_interleaved_kv"]["router_decision"] == "in_place_morph"
        and test2_res["t3_long_interleaved_kv"]["path_b_morph_overhead_ms"]
        < test2_res["t3_long_interleaved_kv"]["path_a_gamma2_ship_ms"]
        and test3_res["bridge_all_shipped"]
        and test3_res["short_all_shipped"]
        and test3_res["long_and_1024_all_in_place"]
    )

    report = {
        "schema": "morphserve.phase1.module2.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_dir": str(model_dir),
        "steps": steps,
        "live_topology": topo.to_dict(),
        "test1_shared_numa_memfd_and_h2d_pacer": test1_res,
        "test2_bagel_dualpath_crossover": test2_res,
        "test3_mixed_cptg_8req_burst": test3_res,
        "status": "ok" if all_pass else "check_failed",
    }

    out_path = output_dir / "module2-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    print(
        "MODULE2_SUMMARY="
        + json.dumps(
            {
                "status": report["status"],
                "host_ram_savings_x": test1_res["host_ram_savings_x"],
                "unpaced_h2d_gbps": test1_res["unpaced_4gpu_half0_per_gpu_gbps"],
                "paced_h2d_gbps": test1_res["paced_dual_half_per_gpu_gbps"],
                "h2d_speedup_x": test1_res["h2d_per_gpu_speedup_x"],
                "t0_ship_ms": test2_res["t0_short_kv"]["path_a_ship_ms"],
                "t0_morph_ms": test2_res["t0_short_kv"]["path_b_morph_overhead_ms"],
                "t3_ship_gamma2_ms": test2_res["t3_long_interleaved_kv"]["path_a_gamma2_ship_ms"],
                "t3_morph_ms": test2_res["t3_long_interleaved_kv"]["path_b_morph_overhead_ms"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MorphServe Module 2 (8-GPU Dual-Path Router) validation")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args(argv)
    rep = run_validation(args.model_dir, args.output_dir, steps=args.steps)
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
