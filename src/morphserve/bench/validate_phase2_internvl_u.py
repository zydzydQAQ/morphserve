"""Phase-2 Milestone M2.4: Live 8xA100 `InternVL-U` (`bridge_mllm_dit`) Cross-Architecture Generality."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import queue
import time
import traceback
from typing import Any

from morphserve.models import (
    InternVLUDecoderPager,
    load_internvlu_pipeline,
    pack_internvlu_bridge_state,
    unpack_internvlu_bridge_state,
)
from morphserve.router import (
    CPTGRequestSpec,
    ClusterTopologyProfile,
    DirectNcclMesh,
    DualPathMorphRouter,
    SharedNumaMemfdMirror,
    WorkerState,
)
from morphserve.vmm import TriPoolVMM

MIB = 1024**2
GIB = 1024**3


def _ssim_pair(img_a: Any, img_b: Any) -> float:
    import numpy as np

    a = np.asarray(img_a.convert("RGB"), dtype=np.float64)
    b = np.asarray(img_b.convert("RGB"), dtype=np.float64)
    h, w, c = a.shape
    bh, bw = (h // 8) * 8, (w // 8) * 8
    ab = a[:bh, :bw].reshape(bh // 8, 8, bw // 8, 8, c).transpose(0, 2, 4, 1, 3).reshape(-1, 64)
    bb = b[:bh, :bw].reshape(bh // 8, 8, bw // 8, 8, c).transpose(0, 2, 4, 1, 3).reshape(-1, 64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = ab.mean(1), bb.mean(1)
    va, vb = ab.var(1), bb.var(1)
    cov = ((ab - ma[:, None]) * (bb - mb[:, None])).mean(1)
    return round(float((((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma**2 + mb**2 + c1) * (va + vb + c2))).mean()), 6)


def _internvlu_worker_proc(
    rank: int,
    num_gpus: int,
    policy_name: str,
    model_dir_str: str,
    vendor_dir_str: str,
    cmd_q: Any,
    ack_q: Any,
) -> None:
    import numpy as np
    from PIL import Image
    import torch

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    topo = ClusterTopologyProfile.a100_pcie_8gpu_default()
    numa = topo.numa_node_of(rank)

    pipe = None
    pager: InternVLUDecoderPager | None = None
    vmm: TriPoolVMM | None = None
    mirror: SharedNumaMemfdMirror | None = None
    mesh: DirectNcclMesh | None = None
    cpu_naive: dict[str, Any] = {}
    bridge_store: dict[str, tuple[dict[str, Any], int, int]] = {}
    packed_store: dict[str, Any] = {}

    try:
        pipe = load_internvlu_pipeline(Path(model_dir_str), vendor_dir=Path(vendor_dir_str), device_index=rank)

        if policy_name == "static_nccl_fission":
            is_und = rank in (0, 1, 4, 5)
            if is_und:
                for p in pipe.generation_decoder.parameters():
                    p.data = torch.empty(0, dtype=p.dtype, device=dev)
                for p in pipe.vae.parameters():
                    p.data = torch.empty(0, dtype=p.dtype, device=dev)
            else:
                for p in pipe.vlm.parameters():
                    p.data = torch.empty(0, dtype=p.dtype, device=dev)
            gc.collect()
            torch.cuda.empty_cache()

        elif policy_name == "naive_host_offload":
            for n, p in pipe.generation_decoder.named_parameters():
                cpu_naive[n] = p.detach().contiguous().cpu()
                p.data = torch.empty(0, dtype=p.dtype, device=dev)
            gc.collect()
            torch.cuda.empty_cache()

        elif policy_name == "morphserve":
            pager = InternVLUDecoderPager(pipe, numa_node=numa, super_handle_mib=64, wrap_vae=True)
            grp = topo.half_group_of(rank)
            is_leader = rank == grp.gpu_ids[0]
            if is_leader:
                mirror = SharedNumaMemfdMirror.create(
                    nbytes=pager.total_mirror_bytes,
                    numa_node=numa,
                    name=f"ms_ivlu_numa{numa}_{os.getpid()}",
                )
                pager.offload_to_shared_mirror(mirror, populate=True)
                ack_q.put(
                    {
                        "op": "LEADER",
                        "rank": rank,
                        "numa": numa,
                        "pid": os.getpid(),
                        "fd": mirror.fd,
                        "nbytes": mirror.nbytes,
                    }
                )
            else:
                ack_q.put({"op": "FOLLOWER", "rank": rank, "numa": numa})
            att = cmd_q.get()
            if not is_leader:
                inf = att["leaders"][str(numa)]
                mirror = SharedNumaMemfdMirror.attach_proc_fd(
                    int(inf["pid"]), int(inf["fd"]), int(inf["nbytes"]), numa_node=numa
                )
                pager.offload_to_shared_mirror(mirror, populate=False)
            pool_mib = max(4608, int(pager.num_layers * pager.layer_stride_mib + 256))
            vmm = TriPoolVMM(
                device=rank,
                super_pool_mib=pool_mib,
                fine_kv_pool_mib=512,
                super_handle_mib=64,
                transient_va_gib=4,
                kv_va_gib=4,
            )
            init_alpha = 1.0 if (rank % 2 == 1) else 0.0
            pager.bind_vmm(vmm, initial_alpha=init_alpha)

        ack_q.put({"op": "READY_NCCL", "rank": rank})
        nccl_msg = cmd_q.get()
        if nccl_msg.get("uid_bytes") is not None and num_gpus >= 2:
            mesh = DirectNcclMesh(rank=rank, world_size=num_gpus, uid_bytes=nccl_msg["uid_bytes"], device_index=rank)

        torch.cuda.empty_cache()
        res_mib = float(torch.cuda.memory_reserved(dev) / MIB)
        ack_q.put({"op": "READY", "rank": rank, "reserved_mib": round(res_mib, 2)})

        while True:
            cmd = cmd_q.get()
            op = cmd.get("op")
            cid = cmd.get("cmd_id")
            if op == "SHUTDOWN":
                ack_q.put({"cmd_id": cid, "rank": rank, "status": "ok"})
                break
            try:
                if op == "VLM_TEXT":
                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        out = pipe(prompt=cmd["prompt"], generation_mode="text", max_new_tokens=int(cmd.get("max_tokens", 32)), do_sample=False)
                    torch.cuda.synchronize(dev)
                    txt = pipe.tokenizer.decode(out.generate_output[0], skip_special_tokens=True)
                    ack_q.put({"cmd_id": cid, "rank": rank, "status": "ok", "ms": round((time.perf_counter() - t0) * 1000.0, 2), "text": txt})

                elif op == "VLM_BRIDGE":
                    t0 = time.perf_counter()
                    h, w = int(cmd.get("height", 512)), int(cmd.get("width", 512))
                    stride = pipe.processor.image_gen_processor.merge_size * pipe.processor.image_gen_processor.patch_size
                    gh, gw = (h // stride) * stride, (w // stride) * stride
                    with torch.inference_mode():
                        inp = pipe.processor(
                            prompt=cmd["prompt"],
                            image=None,
                            generation_mode="image",
                            padding=True,
                            return_tensors="pt",
                            height=gh,
                            width=gw,
                        )
                        inp = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inp.items()}
                        v_out = pipe.vlm.generate_hidden_states(
                            input_ids=inp["input_ids"],
                            attention_mask=inp["attention_mask"],
                            pixel_values=inp.get("pixel_values"),
                        )
                        diff_inp = pipe._prepare_diffusion_inputs(
                            input_ids=inp["input_ids"],
                            attention_mask=inp["attention_mask"],
                            pixel_values=inp.get("pixel_values"),
                            pixel_values_gen=inp.get("pixel_values_gen"),
                            image_grid_thw_gen=inp.get("image_grid_thw_gen"),
                            generation_flags=inp["generation_flags"],
                            vlm_hidden_states=v_out.hidden_states,
                        )
                        meta, flat = pack_internvlu_bridge_state(diff_inp, device=dev)
                    torch.cuda.synchronize(dev)
                    bid = cmd["bridge_id"]
                    bridge_store[bid] = (diff_inp, gh, gw)
                    packed_store[bid] = (meta, flat)
                    ack_q.put(
                        {
                            "cmd_id": cid,
                            "rank": rank,
                            "status": "ok",
                            "ms": round((time.perf_counter() - t0) * 1000.0, 2),
                            "bridge_bytes": int(meta["total_bytes"]),
                            "meta": meta,
                            "gh": gh,
                            "gw": gw,
                        }
                    )

                elif op == "SEND_BRIDGE":
                    assert mesh is not None
                    bid = cmd["bridge_id"]
                    _, flat = packed_store.pop(bid)
                    bridge_store.pop(bid, None)
                    ms = mesh.send_u8(flat, int(cmd["dst_rank"]))
                    del flat
                    ack_q.put({"cmd_id": cid, "rank": rank, "status": "ok", "ship_ms": round(ms, 3)})

                elif op == "RECV_BRIDGE":
                    assert mesh is not None
                    bid = cmd["bridge_id"]
                    meta = cmd["meta"]
                    flat, ms = mesh.recv_u8(int(meta["total_bytes"]), int(cmd["src_rank"]))
                    diff_inp = unpack_internvlu_bridge_state(meta, flat)
                    del flat
                    bridge_store[bid] = (diff_inp, int(cmd["gh"]), int(cmd["gw"]))
                    ack_q.put({"cmd_id": cid, "rank": rank, "status": "ok", "ship_ms": round(ms, 3)})

                elif op == "EXEC_DECODER":
                    bid = cmd["bridge_id"]
                    diff_inp, gh, gw = bridge_store.pop(bid)
                    packed_store.pop(bid, None)
                    t0 = time.perf_counter()
                    if policy_name == "naive_host_offload":
                        for n, p in pipe.generation_decoder.named_parameters():
                            p.data = cpu_naive[n].to(dev)
                        torch.cuda.synchronize(dev)

                    gen = torch.Generator(device=dev).manual_seed(int(cmd.get("seed", 42)))
                    steps = int(cmd.get("steps", 16))
                    with torch.inference_mode():
                        if pager is not None:
                            with pager.generation_mode():
                                imgs_t = pipe.image_pipeline(
                                    **diff_inp,
                                    all_cfg_scale=4.5,
                                    part_cfg_scale=2.0,
                                    num_inference_steps=steps,
                                    num_images_per_prompt=1,
                                    use_resolution_binning=False,
                                    height=gh,
                                    width=gw,
                                    generator=gen,
                                ).images
                        else:
                            imgs_t = pipe.image_pipeline(
                                **diff_inp,
                                all_cfg_scale=4.5,
                                part_cfg_scale=2.0,
                                num_inference_steps=steps,
                                num_images_per_prompt=1,
                                use_resolution_binning=False,
                                height=gh,
                                width=gw,
                                generator=gen,
                            ).images

                    if policy_name == "naive_host_offload":
                        for _, p in pipe.generation_decoder.named_parameters():
                            p.data = torch.empty(0, dtype=p.dtype, device=dev)
                        torch.cuda.empty_cache()

                    torch.cuda.synchronize(dev)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    save_p = cmd.get("save_path")
                    if save_p:
                        out_c = ((127.5 * imgs_t[0] + 128.0) / 255.0).clamp(0, 1)
                        pil = Image.fromarray((out_c.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                        pil.save(save_p, format="PNG")
                    ack_q.put(
                        {
                            "cmd_id": cid,
                            "rank": rank,
                            "status": "ok",
                            "ms": round(dt_ms, 2),
                            "peak_reserved_mib": round(float(torch.cuda.max_memory_reserved(dev) / MIB), 2),
                        }
                    )
            except Exception as exc:
                ack_q.put(
                    {
                        "cmd_id": cid,
                        "rank": rank,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-1000:],
                    }
                )
    except Exception as exc:
        ack_q.put({"op": "FATAL", "rank": rank, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-1200:]})
    finally:
        if mesh is not None:
            mesh.close()
        if pager is not None:
            pager.close()
        if vmm is not None:
            vmm.shutdown()
        if mirror is not None:
            mirror.close()


def run_internvlu_generality_validation(
    model_dir: Path,
    vendor_dir: Path,
    output_dir: Path,
    num_gpus: int = 8,
    smoke_1gpu: bool = False,
) -> dict[str, Any]:
    """Validate `InternVL-U` (`bridge_mllm_dit`) lossless VMM paging + 8-GPU serving comparison."""
    from PIL import Image
    import torch.multiprocessing as mp

    output_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")

    prompts = [
        ("vqa", "Explain in two sentences why liquid water is denser than ice at atmospheric pressure."),
        ("t2i", "A crystalline glass prism refracting sunlight onto dark slate, macro photography."),
        ("t2i", "An autumn mountain valley with mist over pine forests at sunrise, realistic landscape."),
        ("vqa", "Summarize the key difference between covalent and metallic bonding concisely."),
        ("t2i", "A minimalist ceramic teacup beside an open notebook on a warm oak desk."),
        ("t2i", "Snowflakes settling on evergreen needles under warm streetlamp bokeh."),
        ("t2i", "A tranquil Japanese rock garden with raked gravel and moss stones."),
        ("t2i", "Golden hour light illuminating terraced rice paddies after summer rain."),
        ("vqa", "List three thermodynamic state variables in classical equilibrium statistical mechanics."),
        ("t2i", "A brass astronomical compass on aged parchment charts, studio lighting."),
        ("t2i", "Bioluminescent waves breaking along a quiet pebble cove under stars."),
        ("t2i", "An architectural wooden pavilion beside a still lotus pond."),
        ("vqa", "Why does phase coexistence produce a plateau in temperature during melting?"),
        ("t2i", "Freshly fallen red maple leaves floating on a clear forest stream."),
        ("t2i", "A cozy reading alcove overlooking snowy alpine peaks at dusk."),
        ("t2i", "Sunlight filtering through bamboo stalks onto a stone footpath."),
    ]
    if smoke_1gpu or num_gpus == 1:
        num_gpus = 1
        prompts = prompts[:4]
        policies = ("morphserve", "monolithic_dp")
    else:
        policies = ("morphserve", "monolithic_dp", "static_nccl_fission", "naive_host_offload")
    policy_reports: list[dict[str, Any]] = []
    parity_imgs: dict[str, Path] = {}

    for pol in policies:
        cmd_qs = [ctx.Queue() for _ in range(num_gpus)]
        ack_q = ctx.Queue()
        procs = [
            ctx.Process(
                target=_internvlu_worker_proc,
                args=(r, num_gpus, pol, str(model_dir), str(vendor_dir), cmd_qs[r], ack_q),
            )
            for r in range(num_gpus)
        ]
        for p in procs:
            p.start()

        try:
            if pol == "morphserve":
                init_m = [ack_q.get(timeout=240.0) for _ in range(num_gpus)]
                for m in init_m:
                    if m.get("op") == "FATAL":
                        raise RuntimeError(f"rank {m['rank']} fatal: {m['error']}\n{m['traceback']}")
                leaders = {
                    str(m["numa"]): {"pid": m["pid"], "fd": m["fd"], "nbytes": m["nbytes"]}
                    for m in init_m
                    if m.get("op") == "LEADER"
                }
                for r in range(num_gpus):
                    cmd_qs[r].put({"leaders": leaders})

            nccl_r = [ack_q.get(timeout=240.0) for _ in range(num_gpus)]
            for m in nccl_r:
                if m.get("op") == "FATAL":
                    raise RuntimeError(f"rank {m['rank']} fatal: {m['error']}\n{m['traceback']}")
            uid = DirectNcclMesh.generate_unique_id_bytes() if num_gpus >= 2 else None
            for r in range(num_gpus):
                cmd_qs[r].put({"uid_bytes": uid})

            ready_m = [ack_q.get(timeout=240.0) for _ in range(num_gpus)]
            for m in ready_m:
                if m.get("op") == "FATAL":
                    raise RuntimeError(f"rank {m['rank']} fatal: {m['error']}\n{m['traceback']}")
            ready_by_rank = {int(m["rank"]): m for m in ready_m}

            topo = ClusterTopologyProfile.a100_pcie_8gpu_default()
            router = DualPathMorphRouter(topo)
            gpu_free_ms = [0.0 for _ in range(num_gpus)]

            ack_map: dict[str, dict[str, Any]] = {}
            seq = 0

            def _rpc_sync(rk: int, payload: dict[str, Any]) -> dict[str, Any]:
                nonlocal seq
                seq += 1
                cid = f"c_{seq}"
                payload = dict(payload, cmd_id=cid)
                cmd_qs[rk].put(payload)
                while cid not in ack_map:
                    msg = ack_q.get(timeout=300.0)
                    ack_map[str(msg.get("cmd_id"))] = msg
                res = ack_map.pop(cid)
                if res.get("status") == "error":
                    raise RuntimeError(f"rank {rk} error: {res.get('error')}\n{res.get('traceback', '')}")
                return res

            # Deterministic lossless parity image on first T2I prompt (`morphserve` vs `monolithic_dp`)
            if pol in ("morphserve", "monolithic_dp"):
                p_img = Path(f"/tmp/ivlu_parity_{pol}.png")
                rk_p = (1 if num_gpus >= 2 else 0) if pol == "morphserve" else 0
                b_res = _rpc_sync(0, {"op": "VLM_BRIDGE", "bridge_id": "par_0", "prompt": prompts[1][1], "height": 512, "width": 512})
                if rk_p != 0:
                    cmd_qs[0].put({"op": "SEND_BRIDGE", "cmd_id": "ps", "bridge_id": "par_0", "dst_rank": rk_p})
                    cmd_qs[rk_p].put({"op": "RECV_BRIDGE", "cmd_id": "pr", "bridge_id": "par_0", "src_rank": 0, "meta": b_res["meta"], "gh": b_res["gh"], "gw": b_res["gw"]})
                    _ = ack_q.get(timeout=60.0), ack_q.get(timeout=60.0)
                _ = _rpc_sync(rk_p, {"op": "EXEC_DECODER", "bridge_id": "par_0", "steps": 16, "seed": 77, "save_path": str(p_img)})
                parity_imgs[pol] = p_img

            # Dispatch 16-request burst across the 8-GPU cluster using async pipeline scheduling
            t0 = time.perf_counter()
            inflight: list[tuple[str, int, str, float]] = []
            ships = 0
            morphs = 0
            bridge_bytes_samples: list[int] = []
            ship_ms_samples: list[float] = []
            peak_mib = max(float(m["reserved_mib"]) for m in ready_by_rank.values())

            und_cands = [0, 1, 4, 5] if pol == "static_nccl_fission" else list(range(num_gpus))
            gen_cands = [2, 3, 6, 7] if pol == "static_nccl_fission" else list(range(num_gpus))

            # Stage 1: Launch VLM text / bridge extraction across cluster concurrently
            pending_vlm: dict[str, tuple[int, str, int]] = {}
            for idx, (kind, prm) in enumerate(prompts):
                src_g = und_cands[idx % len(und_cands)]
                if pol == "morphserve":
                    # Prefer hybrid even ranks for VLM prefill
                    even = [g for g in range(0, num_gpus, 2)]
                    src_g = even[idx % len(even)]
                seq += 1
                cid = f"vlm_{seq}"
                if kind == "vqa":
                    cmd_qs[src_g].put({"op": "VLM_TEXT", "cmd_id": cid, "prompt": prm, "max_tokens": 28})
                else:
                    cmd_qs[src_g].put({"op": "VLM_BRIDGE", "cmd_id": cid, "bridge_id": f"b_{idx}", "prompt": prm, "height": 512, "width": 512})
                pending_vlm[cid] = (idx, kind, src_g)

            vlm_done: list[tuple[int, int, dict[str, Any]]] = []
            ar_lats: list[float] = []
            while pending_vlm:
                msg = ack_q.get(timeout=300.0)
                cid = str(msg.get("cmd_id"))
                if cid not in pending_vlm:
                    continue
                if msg.get("status") == "error":
                    raise RuntimeError(msg.get("error"))
                idx, kind, src_g = pending_vlm.pop(cid)
                if kind == "vqa":
                    ar_lats.append(float(msg["ms"]))
                else:
                    vlm_done.append((idx, src_g, msg))
                    bridge_bytes_samples.append(int(msg["bridge_bytes"]))

            # Stage 2: Route compact conditioning bridge (`state_ship` vs `in_place_morph`) & run concurrent DiT decoders!
            dec_pending: dict[str, float] = {}
            deferred_acks: list[dict[str, Any]] = []
            dit_assign_load = {g: 0.0 for g in range(num_gpus)}
            rank_alphas = {g: (1.0 if (g % 2 == 1) else 0.0) for g in range(num_gpus)}
            for idx, src_g, b_msg in vlm_done:
                bid = f"b_{idx}"
                b_bytes = int(b_msg["bridge_bytes"])
                dst_g = src_g
                if pol == "morphserve" and num_gpus >= 2:
                    states = {
                        g: WorkerState(
                            gpu_id=g,
                            role="gen" if (g % 2 == 1) else "hybrid",
                            expert_alpha=rank_alphas[g],
                            queue_wait_ms=dit_assign_load[g],
                            kv_free_bytes=24 * GIB,
                        )
                        for g in range(num_gpus)
                    }
                    dec = router.decide(
                        CPTGRequestSpec(
                            req_id=bid,
                            src_gpu=src_g,
                            architecture="bridge_mllm_dit",
                            bridge_bytes=b_bytes,
                            kv_bytes=b_bytes,
                            dit_step1_ms=150.0,
                            dit_total_steps=16,
                            resolution=512,
                        ),
                        states,
                    )
                    if dec.path == "state_ship" and dec.dst_gpu != src_g:
                        dst_g = dec.dst_gpu
                        seq += 1
                        s_cid, r_cid = f"sb_{seq}", f"rb_{seq}"
                        cmd_qs[src_g].put({"op": "SEND_BRIDGE", "cmd_id": s_cid, "bridge_id": bid, "dst_rank": dst_g})
                        cmd_qs[dst_g].put({"op": "RECV_BRIDGE", "cmd_id": r_cid, "bridge_id": bid, "src_rank": src_g, "meta": b_msg["meta"], "gh": b_msg["gh"], "gw": b_msg["gw"]})
                        got_sr = 0
                        while got_sr < 2:
                            m2 = ack_q.get(timeout=120.0)
                            if m2.get("cmd_id") in (s_cid, r_cid):
                                if m2.get("status") == "error":
                                    raise RuntimeError(m2.get("error"))
                                got_sr += 1
                                if "ship_ms" in m2:
                                    ship_ms_samples.append(float(m2["ship_ms"]))
                            else:
                                deferred_acks.append(m2)
                        ships += 1
                    else:
                        rank_alphas[dst_g] = 1.0
                        morphs += 1
                elif pol == "morphserve":
                    rank_alphas[dst_g] = 1.0
                    morphs += 1
                elif pol == "static_nccl_fission":
                    dst_g = min(gen_cands, key=lambda g: dit_assign_load[g])
                    seq += 1
                    s_cid, r_cid = f"sb_{seq}", f"rb_{seq}"
                    cmd_qs[src_g].put({"op": "SEND_BRIDGE", "cmd_id": s_cid, "bridge_id": bid, "dst_rank": dst_g})
                    cmd_qs[dst_g].put({"op": "RECV_BRIDGE", "cmd_id": r_cid, "bridge_id": bid, "src_rank": src_g, "meta": b_msg["meta"], "gh": b_msg["gh"], "gw": b_msg["gw"]})
                    got_sr = 0
                    while got_sr < 2:
                        m2 = ack_q.get(timeout=120.0)
                        if m2.get("cmd_id") in (s_cid, r_cid):
                            if m2.get("status") == "error":
                                raise RuntimeError(m2.get("error"))
                            got_sr += 1
                        else:
                            deferred_acks.append(m2)
                    ships += 1

                dit_assign_load[dst_g] += 2200.0
                seq += 1
                dcid = f"dec_{seq}"
                cmd_qs[dst_g].put({"op": "EXEC_DECODER", "cmd_id": dcid, "bridge_id": bid, "steps": 16, "seed": 100 + idx})
                dec_pending[dcid] = float(b_msg["ms"])

            dit_lats: list[float] = []
            while dec_pending:
                m3 = deferred_acks.pop(0) if deferred_acks else ack_q.get(timeout=300.0)
                dcid = str(m3.get("cmd_id"))
                if dcid not in dec_pending:
                    continue
                if m3.get("status") == "error":
                    raise RuntimeError(m3.get("error"))
                vlm_ms = dec_pending.pop(dcid)
                dit_lats.append(vlm_ms + float(m3["ms"]))
                peak_mib = max(peak_mib, float(m3.get("peak_reserved_mib", 0.0)))

            wall_s = max(1e-3, time.perf_counter() - t0)
            policy_reports.append(
                {
                    "policy_name": pol,
                    "wall_seconds": round(wall_s, 2),
                    "throughput_rps": round(len(prompts) / wall_s, 3),
                    "vqa_mean_ms": round(sum(ar_lats) / max(1, len(ar_lats)), 2),
                    "t2i_p95_ms": round(sorted(dit_lats)[min(len(dit_lats) - 1, int(0.95 * len(dit_lats)))], 2),
                    "mean_bridge_kib": round(sum(bridge_bytes_samples) / max(1, len(bridge_bytes_samples)) / 1024.0, 1),
                    "mean_nccl_ship_ms": round(sum(ship_ms_samples) / max(1, len(ship_ms_samples)), 3) if ship_ms_samples else 0.0,
                    "path_a_ships": ships,
                    "path_b_morphs": morphs,
                    "hybrid_idle_reserved_mib": ready_by_rank[0]["reserved_mib"],
                    "peak_reserved_mib": round(peak_mib, 2),
                }
            )
            print(f"[M2.4-InternVLU] policy={pol} rps={len(prompts)/wall_s:.3f} ship={ships} morph={morphs}", flush=True)
        finally:
            for r in range(num_gpus):
                try:
                    cmd_qs[r].put({"op": "SHUTDOWN", "cmd_id": f"sh_{r}"})
                except Exception:
                    pass
            for p in procs:
                p.join(timeout=15.0)
                if p.is_alive():
                    p.terminate()

    parity_ssim = 0.0
    if "morphserve" in parity_imgs and "monolithic_dp" in parity_imgs:
        with Image.open(parity_imgs["morphserve"]) as ia, Image.open(parity_imgs["monolithic_dp"]) as ib:
            parity_ssim = _ssim_pair(ia, ib)
        for p_f in parity_imgs.values():
            p_f.unlink(missing_ok=True)

    by_pol = {r["policy_name"]: r for r in policy_reports}
    ms_r = by_pol["morphserve"]
    if smoke_1gpu or num_gpus == 1:
        ok = (
            parity_ssim >= 0.995
            and ms_r["hybrid_idle_reserved_mib"] < by_pol["monolithic_dp"]["hybrid_idle_reserved_mib"] - 2500.0
        )
    else:
        ok = (
            parity_ssim >= 0.995
            and ms_r["path_a_ships"] > 0
            and ms_r["throughput_rps"] >= by_pol["naive_host_offload"]["throughput_rps"]
            and ms_r["hybrid_idle_reserved_mib"] < by_pol["monolithic_dp"]["hybrid_idle_reserved_mib"] - 2500.0
        )
    report = {
        "schema": "morphserve.phase2.m24_internvlu.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "architecture": "bridge_mllm_dit",
        "model_repo": "InternVL-U/InternVL-U",
        "num_gpus": num_gpus,
        "smoke_1gpu": bool(smoke_1gpu or num_gpus == 1),
        "vmm_lossless_parity_ssim": parity_ssim,
        "hybrid_hbm_saved_mib": round(by_pol["monolithic_dp"]["hybrid_idle_reserved_mib"] - ms_r["hybrid_idle_reserved_mib"], 1),
        "results": policy_reports,
        "status": "ok" if ok else "check_failed",
    }
    out_path = output_dir / "m24-internvlu-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MorphServe Phase-2 M2.4 InternVL-U 8xA100 Generality Validator")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--smoke-1gpu", action="store_true")
    args = parser.parse_args(argv)
    rep = run_internvlu_generality_validation(
        args.model_dir,
        args.vendor_dir,
        args.output_dir,
        num_gpus=args.num_gpus,
        smoke_1gpu=args.smoke_1gpu,
    )
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
