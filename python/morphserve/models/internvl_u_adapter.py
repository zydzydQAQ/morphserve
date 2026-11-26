"""Deep `InternVLUCPTGAdapter` (`bridge_mllm_dit`) built atop `LayerExpertPager` + dual-regime HBM accounting."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import gc
import inspect
import os
from pathlib import Path
import time
from typing import Any, Callable

from morphserve.vmm import (
    ExpertPartitionSpec,
    LayerExpertPager,
    NumaMemfdMirror,
    TriPoolVMM,
)

from .base import GpuCPTGAdapterMixin, PackedTransferState, compute_vmm_available_kv_bytes
from .internvl_u_loader import load_internvlu_pipeline

MIB = 1024**2
GIB = 1024**3


def compute_internvlu_memory_plan(
    *,
    total_dev_mib: float,
    free_dev_mib: float,
    base_pytorch_alloc_mib: float,
    mirror_bytes: int,
    policy_super_pool_mib: int,
    policy_fine_kv_pool_mib: int,
    super_handle_mib: int = 64,
    allow_vmm_overflow: bool = True,
    gpu_memory_fraction: float | None = None,
    gpu_memory_quota_mib: float | None = None,
) -> dict[str, Any]:
    """Compute physical `TriPoolVMM` + PyTorch budget plan via C++17 `WorkerBudgetPlanner`."""
    from morphserve._core import call_json, encode_json, get_core_lib

    L = get_core_lib()
    req = {
        "total_dev_mib": float(total_dev_mib),
        "free_dev_mib": float(free_dev_mib),
        "base_pytorch_alloc_mib": float(base_pytorch_alloc_mib),
        "mirror_bytes": int(mirror_bytes),
        "policy_super_pool_mib": int(policy_super_pool_mib),
        "policy_fine_kv_pool_mib": int(policy_fine_kv_pool_mib),
        "super_handle_mib": int(super_handle_mib),
        "allow_vmm_overflow": bool(allow_vmm_overflow),
        "gpu_memory_fraction": None if gpu_memory_fraction is None else float(gpu_memory_fraction),
        "gpu_memory_quota_mib": None if gpu_memory_quota_mib is None else float(gpu_memory_quota_mib),
    }
    return call_json(L, L.ms_budget_internvlu_plan_json, encode_json(req))


def _clone_diffusers_output(obj: Any) -> Any:
    """Clone tensors inside `diffusers` `BaseOutput` / `DiagonalGaussianDistribution` objects out of Arena-2."""
    import torch

    if torch.is_tensor(obj):
        return obj.clone()
    # Inspect `diffusers.utils.BaseOutput` subclasses (`AutoencoderKLOutput`, `DecoderOutput`, ...)
    # before plain `dict` because `BaseOutput` inherits from `OrderedDict`.
    if hasattr(obj, "latent_dist"):
        ld = obj.latent_dist
        for attr in ("parameters", "mean", "logvar", "std", "var"):
            val = getattr(ld, attr, None)
            if torch.is_tensor(val):
                setattr(ld, attr, val.clone())
        if isinstance(obj, dict) and "latent_dist" in obj:
            obj["latent_dist"] = ld
        return obj
    if hasattr(obj, "sample") and torch.is_tensor(obj.sample):
        cloned = obj.sample.clone()
        obj.sample = cloned
        if isinstance(obj, dict) and "sample" in obj:
            obj["sample"] = cloned
        return obj
    if hasattr(obj, "latents") and torch.is_tensor(obj.latents):
        cloned = obj.latents.clone()
        obj.latents = cloned
        if isinstance(obj, dict) and "latents" in obj:
            obj["latents"] = cloned
        return obj
    if any(torch.is_tensor(getattr(obj, attr, None)) for attr in ("parameters", "mean", "logvar", "std", "var")):
        for attr in ("parameters", "mean", "logvar", "std", "var"):
            val = getattr(obj, attr, None)
            if torch.is_tensor(val):
                setattr(obj, attr, val.clone())
        return obj
    if isinstance(obj, tuple):
        return tuple(_clone_diffusers_output(x) for x in obj)
    if isinstance(obj, list):
        return [_clone_diffusers_output(x) for x in obj]
    if isinstance(obj, dict):
        if type(obj) is dict:
            return {k: _clone_diffusers_output(v) for k, v in obj.items()}
        for k, v in list(obj.items()):
            cloned_v = _clone_diffusers_output(v)
            obj[k] = cloned_v
            if hasattr(obj, k):
                try:
                    setattr(obj, k, cloned_v)
                except Exception:
                    pass
        return obj
    return obj


_DTYPE_NAME_MAP = {
    "torch.bfloat16": "bfloat16",
    "torch.float16": "float16",
    "torch.float32": "float32",
    "torch.int64": "int64",
    "torch.int32": "int32",
    "torch.bool": "bool",
    "torch.uint8": "uint8",
}


def pack_internvlu_bridge_state(diffusion_inputs: dict[str, Any], device: Any) -> tuple[dict[str, Any], Any]:
    """Pack `InternVL-U` `diffusion_inputs` bridge tensors into a flat contiguous `uint8` CUDA tensor."""
    import torch

    chunks: list[Any] = []
    offset = 0

    def _rec(key: str, val: Any) -> Any:
        nonlocal offset
        if val is None:
            return {"kind": "none"}
        if torch.is_tensor(val):
            t_c = val.to(device).contiguous()
            u8 = t_c.view(torch.uint8).reshape(-1)
            nb = int(u8.numel())
            node = {
                "kind": "tensor",
                "dtype": _DTYPE_NAME_MAP[str(t_c.dtype)],
                "shape": list(t_c.shape),
                "offset": offset,
                "nbytes": nb,
            }
            if nb > 0:
                chunks.append(u8)
                offset += nb
            return node
        if isinstance(val, list):
            return {"kind": "list", "items": [_rec(f"{key}.{i}", x) for i, x in enumerate(val)]}
        return {"kind": "const", "value": val}

    tree = {k: _rec(k, v) for k, v in diffusion_inputs.items()}
    flat = torch.cat(chunks, dim=0) if chunks else torch.empty((0,), dtype=torch.uint8, device=device)
    meta = {"total_bytes": int(flat.numel()), "tree": tree}
    return meta, flat


def unpack_internvlu_bridge_state(meta: dict[str, Any], flat_u8: Any) -> dict[str, Any]:
    """Reconstruct `InternVL-U` `diffusion_inputs` dictionary from flat `uint8` CUDA buffer."""
    import torch

    dtype_lookup = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "int64": torch.int64,
        "int32": torch.int32,
        "bool": torch.bool,
        "uint8": torch.uint8,
    }

    def _dec(node: dict[str, Any]) -> Any:
        kind = node["kind"]
        if kind == "none":
            return None
        if kind == "tensor":
            dt = dtype_lookup[node["dtype"]]
            nb = int(node["nbytes"])
            off = int(node["offset"])
            if nb == 0:
                return torch.empty(node["shape"], dtype=dt, device=flat_u8.device)
            return flat_u8[off : off + nb].clone().view(dt).reshape(node["shape"])
        if kind == "list":
            return [_dec(x) for x in node["items"]]
        return node["value"]

    return {k: _dec(v) for k, v in meta["tree"].items()}


def _trim_internvlu_generated_tokens(seq: Any, tokenizer: Any) -> list[int]:
    """Strip right-padded PAD/EOS/`<|im_end|>`/`<img>` tokens so batch length never distorts `tpots_ms`."""
    raw_ids = [int(x) for x in (seq.detach().cpu().tolist() if hasattr(seq, "detach") else list(seq))]
    if not raw_ids:
        return []

    stop_ids: set[int] = set()
    pad_id: int | None = None
    if tokenizer is not None:
        raw_pad = getattr(tokenizer, "pad_token_id", None)
        if isinstance(raw_pad, int) and raw_pad >= 0:
            pad_id = int(raw_pad)
        raw_eos = getattr(tokenizer, "eos_token_id", None)
        if isinstance(raw_eos, int) and raw_eos >= 0:
            stop_ids.add(int(raw_eos))
        elif isinstance(raw_eos, (list, tuple, set)):
            for eid in raw_eos:
                if isinstance(eid, int) and eid >= 0:
                    stop_ids.add(int(eid))
        conv_fn = getattr(tokenizer, "convert_tokens_to_ids", None)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if callable(conv_fn):
            for tok_str in ("<|im_end|>", "<|endoftext|>", "<img>"):
                try:
                    tid = conv_fn(tok_str)
                except Exception:
                    tid = None
                if isinstance(tid, int) and tid >= 0 and tid != unk_id:
                    stop_ids.add(int(tid))

    non_pad_stops = stop_ids - ({pad_id} if pad_id is not None else set())
    trimmed = raw_ids
    for idx, tid in enumerate(raw_ids):
        if tid in non_pad_stops:
            trimmed = raw_ids[: idx + 1]
            break
    if pad_id is not None:
        while len(trimmed) > 1 and trimmed[-1] == pad_id:
            trimmed.pop()
    return trimmed


class InternVLUCPTGAdapter(GpuCPTGAdapterMixin):
    """Production `InternVL-U` (`bridge_mllm_dit`) adapter running inside `MorphServeClusterEngine`."""

    architecture = "bridge_mllm_dit"

    def __init__(
        self,
        rank: int,
        policy: Any,
        model_dir: Path | str | None = None,
        bench_root: Path | str | None = None,
        numa_node: int = 0,
        *,
        pipeline: Any | None = None,
        device: Any | None = None,
    ) -> None:
        self.rank = int(rank)
        self.policy = policy
        self.model_dir = Path(model_dir) if model_dir is not None else Path(".")
        self.bench_root = Path(bench_root) if bench_root is not None else Path(".")
        self.numa_node = int(numa_node)
        self.role = policy.role_of(self.rank)
        self._init_gpu_lifecycle_state()
        self.vlm_pager: LayerExpertPager | None = None
        self._memory_plan: dict[str, Any] = {}
        self._quota_constrained: bool = False

        if device is not None:
            self.device = device
        elif pipeline is not None and getattr(pipeline, "device", None) is not None:
            self.device = pipeline.device
        else:
            import torch

            self.device = torch.device(f"cuda:{self.rank}")

        is_static_fission = policy.routing_mode == "static_nccl_fission"
        raw_frac = getattr(policy, "gpu_memory_fraction", None)
        raw_quota = getattr(policy, "gpu_memory_quota_mib", None)
        copage_vlm = bool(
            (not is_static_fission)
            and policy.use_tripool_vmm
            and policy.use_expert_pager
            and ((raw_quota is not None and float(raw_quota) > 0.0) or (raw_frac is not None and float(raw_frac) < 0.999))
        )

        if pipeline is not None:
            self.pipe = pipeline
        else:
            import torch

            defer_decoder = bool(
                policy.use_expert_pager or policy.use_naive_offload or (is_static_fission and self.role == "und")
            )
            drop_vlm = bool(is_static_fission and self.role == "gen")
            drop_vae = bool(is_static_fission and self.role == "und")

            if raw_frac is not None and torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(float(raw_frac), self.device)

            vendor_p = Path(policy.vendor_dir) if getattr(policy, "vendor_dir", None) else None
            self.pipe = load_internvlu_pipeline(
                self.model_dir,
                vendor_dir=vendor_p,
                device_index=self.rank,
                defer_decoder_gpu=defer_decoder,
                defer_vlm_gpu=copage_vlm,
                drop_vlm_on_load=drop_vlm,
                drop_vae_on_load=drop_vae,
            )

        if getattr(self.pipe, "generation_decoder", None) is not None:
            dec_spec = ExpertPartitionSpec.for_internvlu(
                self.pipe,
                super_handle_mib=policy.super_handle_mib,
                wrap_vae=policy.use_scoped_vae,
                vae_encode_restore="none",
                vae_output_cloner=_clone_diffusers_output,
            )
            self.num_expert_layers = dec_spec.num_layers
            self.bind_expert_partition(
                dec_spec,
                drop_for_fission=(self.role == "und"),
                vae_output_cloner=_clone_diffusers_output,
            )
            if copage_vlm and self.pager is not None and getattr(self.pipe, "vlm", None) is not None:
                vlm_spec = ExpertPartitionSpec.for_internvlu_vlm(
                    self.pipe,
                    super_handle_mib=policy.super_handle_mib,
                    install_module_hooks=False,
                )
                self.vlm_pager = LayerExpertPager(vlm_spec, numa_node=self.numa_node)
        else:
            self.num_expert_layers = 20

    @property
    def total_mirror_bytes(self) -> int:
        dec_b = self.pager.total_mirror_bytes if self.pager is not None else 0
        vlm_b = self.vlm_pager.total_mirror_bytes if self.vlm_pager is not None else 0
        return dec_b + vlm_b

    @property
    def max_stage_mirror_bytes(self) -> int:
        dec_b = self.pager.total_mirror_bytes if self.pager is not None else 0
        vlm_b = self.vlm_pager.total_mirror_bytes if self.vlm_pager is not None else 0
        return max(dec_b, vlm_b)

    def attach_expert_mirror(self, shared_mirror: NumaMemfdMirror | None, populate: bool = True) -> None:
        if self.pager is None:
            return
        if shared_mirror is not None:
            self.pager.offload_to_shared_mirror(shared_mirror, populate=populate, mirror_offset=0)
            if self.vlm_pager is not None:
                self.vlm_pager.offload_to_shared_mirror(
                    shared_mirror,
                    populate=populate,
                    mirror_offset=self.pager.total_mirror_bytes,
                )
        else:
            self.pager.offload_to_host_mirror()
            if self.vlm_pager is not None:
                self.vlm_pager.offload_to_host_mirror()

    def available_kv_bytes(self, *, for_dit: bool = False) -> int:
        import torch

        if self.vmm is not None:
            return compute_vmm_available_kv_bytes(
                self.vmm,
                has_pageable_experts=(self.pager is not None),
                for_dit=for_dit,
            )
        if not torch.cuda.is_available() or getattr(self.device, "type", "cuda") != "cuda":
            return 0
        free_drv, total_drv = torch.cuda.mem_get_info(self.device)
        if getattr(self.policy, "gpu_memory_fraction", None) is not None:
            quota_b = int(total_drv * float(self.policy.gpu_memory_fraction))
            alloc_b = int(torch.cuda.memory_allocated(self.device))
            return max(0, quota_b - alloc_b - 256 * MIB)
        return max(0, int(free_drv) - 1024 * MIB)

    def bind_runtime_resources(self, pacer: Any | None = None) -> dict[str, Any]:
        import torch

        self.pacer = pacer
        gc.collect()
        is_cuda = torch.cuda.is_available() and getattr(self.device, "type", "cuda") == "cuda"
        if is_cuda:
            torch.cuda.empty_cache()
            free_b, total_b = torch.cuda.mem_get_info(self.device)
            total_dev_mib = float(total_b / MIB)
            free_dev_mib = float(free_b / MIB)
            base_pt_mib = float(torch.cuda.memory_allocated(self.device) / MIB)
        else:
            total_dev_mib = 40_960.0
            free_dev_mib = 32_768.0
            base_pt_mib = 0.0

        if self.policy.use_tripool_vmm:
            plan = compute_internvlu_memory_plan(
                total_dev_mib=total_dev_mib,
                free_dev_mib=free_dev_mib,
                base_pytorch_alloc_mib=base_pt_mib,
                mirror_bytes=self.max_stage_mirror_bytes,
                policy_super_pool_mib=int(self.policy.super_pool_mib),
                policy_fine_kv_pool_mib=int(self.policy.fine_kv_pool_mib),
                super_handle_mib=int(self.policy.super_handle_mib),
                allow_vmm_overflow=bool(self.policy.allow_vmm_overflow),
                gpu_memory_fraction=getattr(self.policy, "gpu_memory_fraction", None),
                gpu_memory_quota_mib=getattr(self.policy, "gpu_memory_quota_mib", None),
            )
            self._memory_plan = plan
            self._quota_constrained = bool(plan["quota_constrained"])
            if is_cuda and plan["pytorch_memory_fraction"] is not None:
                torch.cuda.set_per_process_memory_fraction(float(plan["pytorch_memory_fraction"]), self.device)

            self._vmm_initial_pool_mib = float(plan["vmm_physical_pool_mib"])
            self.vmm = TriPoolVMM(
                device=self.rank,
                super_pool_mib=int(plan["super_pool_mib"]),
                fine_kv_pool_mib=int(plan["fine_kv_pool_mib"]),
                super_handle_mib=int(self.policy.super_handle_mib),
                transient_va_gib=int(self.policy.transient_va_gib),
                kv_va_gib=int(self.policy.kv_va_gib),
                allow_overflow=bool(plan["allow_overflow"]),
            )
            if self.pager is not None:
                init_alpha = 0.0 if self.vlm_pager is not None else float(self.policy.initial_alphas.get(self.rank, 1.0))
                if not bool(plan["allow_overflow"]):
                    max_fit_layers = int(plan["super_pool_mib"]) // max(1, self.pager.layer_stride_mib)
                    max_fit_alpha = min(1.0, float(max_fit_layers) / float(max(1, self.pager.num_layers)))
                    init_alpha = min(init_alpha, max_fit_alpha)
                self._initial_bound_alpha = float(init_alpha)
                self.pager.bind_vmm(self.vmm, initial_alpha=init_alpha)
            if self.vlm_pager is not None:
                vlm_fit_layers = int(plan["super_pool_mib"]) // max(1, self.vlm_pager.layer_stride_mib)
                vlm_init_alpha = min(1.0, float(vlm_fit_layers) / float(max(1, self.vlm_pager.num_layers)))
                self._initial_vlm_alpha = float(vlm_init_alpha)
                self.vlm_pager.bind_vmm(self.vmm, initial_alpha=vlm_init_alpha)
        else:
            self._quota_constrained = getattr(self.policy, "gpu_memory_fraction", None) is not None

        if is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            pt_mib = float(torch.cuda.memory_reserved(self.device) / MIB)
            self._max_pytorch_reserved_mib = pt_mib
            self._max_reserved_mib = pt_mib + self._vmm_physical_pool_mib()
        self._update_hwm()
        return {
            "resident_alpha": self.resident_alpha(),
            "reserved_mib": round(self._max_pytorch_reserved_mib, 2),
            "combined_physical_mib": round(self._max_reserved_mib, 2),
            "pytorch_reserved_mib": round(self._max_pytorch_reserved_mib, 2),
            "vmm_physical_pool_mib": round(self._vmm_physical_pool_mib(), 2),
            "kv_free_bytes": self.available_kv_bytes(),
            "memory_plan": dict(self._memory_plan),
        }

    def warmup(self) -> None:
        self.reset_peak_memory_stats()

    def _load_pil_path(self, rel_or_abs: str) -> Any:
        from PIL import Image

        p = self.bench_root / rel_or_abs if not os.path.isabs(rel_or_abs) else Path(rel_or_abs)
        if not p.is_file():
            return None
        with Image.open(p) as raw:
            im = raw.convert("RGB")
            im.load()
            return im

    def _load_pil_if_any(self, cmd: dict[str, Any]) -> Any:
        paths = cmd.get("image_paths") or []
        if not paths:
            return None
        return self._load_pil_path(str(paths[0]))

    @contextmanager
    def _vlm_execution_scope(self):
        """Pin & materialize `vlm_pager` during VLM execution, holding H2D pacer only during H2D weight loads."""
        if self.vlm_pager is None or self.vmm is None:
            yield
            return
        if self.pager is not None and self.pager.expert_id >= 0:
            self.vmm.set_expert_pinned(self.pager.expert_id, False)
        if self.vmm.expert_resident_count(self.vlm_pager.expert_id) < self.vlm_pager.num_layers:
            pacer_cm = self.pacer.acquire_process_slot(self.rank) if self.pacer is not None else nullcontext()
            with pacer_cm:
                self.vmm.set_expert_pinned(self.vlm_pager.expert_id, True)
                try:
                    self.vlm_pager.ensure_all_resident_sync()
                except BaseException:
                    self.vmm.set_expert_pinned(self.vlm_pager.expert_id, False)
                    raise
        with self.vlm_pager.generation_mode(prefetch_all=True):
            yield

    def prepare_turn(self, ctx_id: str, cmd: dict[str, Any]) -> int:
        import torch

        prompt = str(cmd["prompt"])
        prep_dit = bool(cmd.get("prepare_for_dit", False))
        prev_ent = self._ctx_store.get(ctx_id)
        use_prev = cmd.get("use_prev_image")
        pil_img = None
        if use_prev:
            prev_path = cmd.get("prev_image_path")
            if prev_path:
                pil_img = self._load_pil_path(str(prev_path))
            if pil_img is None and prev_ent is not None:
                pil_img = prev_ent.get("pil_img")
        if pil_img is None:
            pil_img = self._load_pil_if_any(cmd)
        if pil_img is None and use_prev is None and prev_ent is not None and not (cmd.get("image_paths")):
            pil_img = prev_ent.get("pil_img")
        if not prep_dit:
            self._ctx_store[ctx_id] = {"prompt": prompt, "pil_img": pil_img, "bridge_bytes": 4096}
            return 4096

        is_cuda = torch.cuda.is_available() and getattr(self.device, "type", "cuda") == "cuda"
        if self._quota_constrained and is_cuda:
            torch.cuda.empty_cache()

        h = int(cmd.get("height", 512))
        w = int(cmd.get("width", 512))
        stride = self.pipe.processor.image_gen_processor.merge_size * self.pipe.processor.image_gen_processor.patch_size
        gh, gw = (h // stride) * stride, (w // stride) * stride
        with self._vlm_execution_scope(), torch.inference_mode():
            inp = self.pipe.processor(
                prompt=prompt,
                image=pil_img,
                generation_mode="image",
                padding=True,
                return_tensors="pt",
                height=gh,
                width=gw,
            )
            inp = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inp.items()}
            v_out = self.pipe.vlm.generate_hidden_states(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                pixel_values=inp.get("pixel_values"),
            )
            diff_inp = self.pipe._prepare_diffusion_inputs(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                pixel_values=inp.get("pixel_values"),
                pixel_values_gen=inp.get("pixel_values_gen"),
                image_grid_thw_gen=inp.get("image_grid_thw_gen"),
                generation_flags=inp["generation_flags"],
                vlm_hidden_states=v_out.hidden_states,
            )
            del inp, v_out
            meta, flat = pack_internvlu_bridge_state(diff_inp, device=self.device)
        if is_cuda:
            torch.cuda.current_stream(self.device).synchronize()
            if self._quota_constrained:
                torch.cuda.empty_cache()
        b_bytes = int(meta["total_bytes"])
        self._ctx_store[ctx_id] = {
            "prompt": prompt,
            "pil_img": pil_img,
            "diff_inp": diff_inp,
            "gh": gh,
            "gw": gw,
            "packed_meta": meta,
            "packed_flat": flat,
            "bridge_bytes": b_bytes,
        }
        self._update_hwm()
        return b_bytes

    def decode_ar(self, ctx_id: str, max_tokens: int) -> dict[str, Any]:
        import torch

        entry = self._ctx_store[ctx_id]
        prompt = entry["prompt"]
        pil_img = entry.get("pil_img")
        is_cuda = torch.cuda.is_available() and getattr(self.device, "type", "cuda") == "cuda"
        if self._quota_constrained and is_cuda:
            torch.cuda.empty_cache()
        t0 = time.perf_counter()
        with self._vlm_execution_scope(), torch.inference_mode():
            out = self.pipe(
                prompt=prompt,
                image=pil_img,
                generation_mode="text",
                max_new_tokens=int(max_tokens),
                do_sample=False,
            )
        if is_cuda:
            torch.cuda.current_stream(self.device).synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        tok_seq = _trim_internvlu_generated_tokens(out.generate_output[0], self.pipe.tokenizer)
        n_tok = max(1, int(len(tok_seq)))
        txt = self.pipe.tokenizer.decode(tok_seq, skip_special_tokens=True) if tok_seq else ""
        per_tok = dt_ms / float(n_tok)
        if self._quota_constrained and is_cuda:
            torch.cuda.empty_cache()
        self._update_hwm()
        return {
            "text": txt,
            "tokens": n_tok,
            "ttft_ms": round(per_tok, 2),
            "tpots_ms": [round(per_tok, 2) for _ in range(max(0, n_tok - 1))],
            "kv_bytes": int(entry.get("bridge_bytes", 4096)),
            "resident_alpha": self.resident_alpha(),
            "kv_free_bytes": self.available_kv_bytes(),
        }

    def decode_ar_batch(self, batch_cmds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Batched `[B, L]` VLM greedy decoding for concurrent text-only requests; CoT-aware fallback for multimodal."""
        import torch

        if not batch_cmds:
            return []

        if any(bool(c.get("prepare_for_dit")) for c in batch_cmds):
            results: list[dict[str, Any]] = []
            for cmd in batch_cmds:
                cid = str(cmd["ctx_id"])
                enq_s = self.normalized_enqueue_time_s(cmd)
                t_cmd0 = time.perf_counter()
                if not cmd.get("_skip_prepare", False):
                    self.prepare_turn(cid, dict(cmd, prepare_for_dit=False))
                ar_out = self.decode_ar(cid, max(1, int(cmd.get("max_ar_tokens", 32))))
                if bool(cmd.get("prepare_for_dit")):
                    base_prompt = str(cmd.get("prompt", self._ctx_store[cid].get("prompt", "")))
                    cot_prompt = f"{base_prompt}\n{ar_out['text']}".strip() if ar_out.get("text") else base_prompt
                    b_bytes = self.prepare_turn(cid, dict(cmd, prompt=cot_prompt, prepare_for_dit=True))
                    ar_out["kv_bytes"] = b_bytes
                    ar_out["kv_free_bytes"] = self.available_kv_bytes()
                    ar_out["resident_alpha"] = self.resident_alpha()
                if enq_s > 0.0:
                    ar_out["ttft_ms"] = round(float(ar_out["ttft_ms"]) + max(0.0, (t_cmd0 - enq_s) * 1000.0), 2)
                results.append(ar_out)
            return results

        if len(batch_cmds) <= 1 or any(
            bool(c.get("image_paths"))
            or bool(c.get("use_prev_image"))
            or (self._ctx_store.get(str(c["ctx_id"]), {}).get("pil_img") is not None)
            for c in batch_cmds
        ):
            return super().decode_ar_batch(batch_cmds)

        t_start = time.perf_counter()
        prompts: list[str] = []
        max_toks: list[int] = []
        for cmd in batch_cmds:
            cid = str(cmd["ctx_id"])
            if not cmd.get("_skip_prepare", False):
                self.prepare_turn(cid, cmd)
            prompts.append(str(self._ctx_store[cid]["prompt"]))
            max_toks.append(max(1, int(cmd.get("max_ar_tokens", 32))))

        is_cuda = torch.cuda.is_available() and getattr(self.device, "type", "cuda") == "cuda"
        if self._quota_constrained and is_cuda:
            torch.cuda.empty_cache()

        t_gen0 = time.perf_counter()
        with self._vlm_execution_scope(), torch.inference_mode():
            out = self.pipe(
                prompt=prompts,
                image=None,
                generation_mode="text",
                max_new_tokens=max(max_toks),
                do_sample=False,
            )
        if is_cuda:
            torch.cuda.current_stream(self.device).synchronize()
        dt_ms = (time.perf_counter() - t_gen0) * 1000.0
        if self._quota_constrained and is_cuda:
            torch.cuda.empty_cache()
        self._update_hwm()

        alpha_now = self.resident_alpha()
        free_kv_now = self.available_kv_bytes()
        trimmed_seqs = [
            _trim_internvlu_generated_tokens(out.generate_output[idx][: max_toks[idx]], self.pipe.tokenizer)
            for idx in range(len(batch_cmds))
        ]
        max_steps_executed = max((max(1, len(s)) for s in trimmed_seqs), default=1)
        per_step_ms = dt_ms / float(max_steps_executed)
        results = []
        for idx, cmd in enumerate(batch_cmds):
            enq_s = self.normalized_enqueue_time_s(cmd)
            q_wait_ms = max(0.0, (t_start - enq_s) * 1000.0) if enq_s > 0.0 else 0.0
            seq = trimmed_seqs[idx]
            n_tok = max(1, int(len(seq)))
            txt = self.pipe.tokenizer.decode(seq, skip_special_tokens=True) if seq else ""
            results.append(
                {
                    "text": txt,
                    "tokens": n_tok,
                    "ttft_ms": round(q_wait_ms + per_step_ms, 2),
                    "tpots_ms": [round(per_step_ms, 2) for _ in range(max(0, n_tok - 1))],
                    "kv_bytes": 4096,
                    "resident_alpha": alpha_now,
                    "kv_free_bytes": free_kv_now,
                    "batched_width": len(batch_cmds),
                }
            )
        return results

    def pack_state(self, ctx_id: str, include_cfg_dup: bool = False) -> PackedTransferState:
        entry = self._ctx_store[ctx_id]
        if "packed_flat" in entry and entry["packed_flat"] is not None:
            meta = dict(entry["packed_meta"], gh=entry["gh"], gw=entry["gw"])
            flat = entry.pop("packed_flat")
        else:
            raw_meta, flat = pack_internvlu_bridge_state(entry["diff_inp"], device=self.device)
            meta = dict(raw_meta, gh=entry["gh"], gw=entry["gw"])
        total_b = int(meta["total_bytes"])
        meta["logical_bytes"] = total_b
        return PackedTransferState(meta=meta, payload=flat, logical_bytes=total_b)

    def unpack_state(self, new_ctx_id: str, meta: dict[str, Any], raw_payload: Any) -> int:
        diff_inp = unpack_internvlu_bridge_state(meta, raw_payload)
        total_b = int(meta["total_bytes"])
        self._ctx_store[new_ctx_id] = {
            "prompt": "",
            "diff_inp": diff_inp,
            "gh": int(meta["gh"]),
            "gw": int(meta["gw"]),
            "bridge_bytes": total_b,
        }
        self._update_hwm()
        return total_b

    def generate_dit(
        self,
        ctx_id: str,
        cmd: dict[str, Any],
        poll_preempt_cmds: Callable[[], list[dict[str, Any]]] | None = None,
        emit_preempt_ack: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        import numpy as np
        from PIL import Image
        import torch

        entry = self._ctx_store[ctx_id]
        stride = self.pipe.processor.image_gen_processor.merge_size * self.pipe.processor.image_gen_processor.patch_size
        req_h = int(cmd.get("height", entry.get("gh", 512)))
        req_w = int(cmd.get("width", entry.get("gw", 512)))
        req_gh, req_gw = (req_h // stride) * stride, (req_w // stride) * stride

        if "diff_inp" not in entry or (
            (req_gh, req_gw) != (int(entry["gh"]), int(entry["gw"])) and getattr(self.pipe, "vlm", None) is not None
        ):
            eff_prompt = str(entry.get("prompt") or cmd.get("prompt", ""))
            self.prepare_turn(
                ctx_id,
                dict(cmd, prompt=eff_prompt, height=req_gh, width=req_gw, prepare_for_dit=True),
            )
            entry = self._ctx_store[ctx_id]

        diff_inp = entry["diff_inp"]
        entry.pop("packed_flat", None)
        gh, gw = int(entry["gh"]), int(entry["gw"])
        steps = int(cmd.get("dit_steps", 16))
        seed = int(cmd.get("seed", 42))
        save_tmp_path = cmd.get("save_tmp_image_path")

        is_cuda = torch.cuda.is_available() and getattr(self.device, "type", "cuda") == "cuda"
        if self._quota_constrained and is_cuda:
            torch.cuda.empty_cache()

        t0 = time.perf_counter()
        if self.naive_offloader is not None:
            self.naive_offloader.swap_in_sync()

        gen = torch.Generator(device=self.device).manual_seed(seed)
        try:
            if self.vlm_pager is not None and self.vmm is not None and self.vlm_pager.expert_id >= 0:
                self.vmm.set_expert_pinned(self.vlm_pager.expert_id, False)
            with torch.inference_mode():
                if self.pager is not None and self.vmm is not None:
                    with self.pager.generation_mode(prefetch_all=True, pacer=self.pacer, rank=self.rank):
                        # Bucket 0 holds `encoder_padding_token` and `decoder_projector`, which execute in
                        # `generation_decoder.prepare_forward_input` before `transformer_blocks[0]`'s pre-hook.
                        self.pager.wait_layer_if_needed(0, "prepare_forward_input")
                        latents_out = self.pipe.image_pipeline(
                            **diff_inp,
                            all_cfg_scale=4.5,
                            part_cfg_scale=2.0,
                            num_inference_steps=steps,
                            num_images_per_prompt=1,
                            use_resolution_binning=False,
                            height=gh,
                            width=gw,
                            generator=gen,
                            output_type="latent",
                        ).images
                    # Run QwenImage VAE decode outside `generation_mode` so decoder blocks remain unpinned
                    # under strict 12 GiB quotas while staying 100% resident under ample 40/80 GB `super_pool`.
                    lat_bf16 = latents_out.to(torch.bfloat16)
                    del latents_out
                    if self._quota_constrained and is_cuda:
                        torch.cuda.empty_cache()
                    imgs_t = self.pipe.image_pipeline.latents_to_pixels(lat_bf16)
                    del lat_bf16
                else:
                    call_kwargs: dict[str, Any] = {
                        **diff_inp,
                        "all_cfg_scale": 4.5,
                        "part_cfg_scale": 2.0,
                        "num_inference_steps": steps,
                        "num_images_per_prompt": 1,
                        "use_resolution_binning": False,
                        "height": gh,
                        "width": gw,
                        "generator": gen,
                    }
                    if "generator_device" in inspect.signature(self.pipe.image_pipeline.__call__).parameters:
                        call_kwargs["generator_device"] = None
                    imgs_t = self.pipe.image_pipeline(**call_kwargs).images
        finally:
            if self.naive_offloader is not None:
                self.naive_offloader.drop_from_gpu()

        if is_cuda:
            torch.cuda.current_stream(self.device).synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        release_after_dit = bool(cmd.get("release_after_dit", False))
        if (not release_after_dit) or save_tmp_path:
            img0 = imgs_t[0]
            if isinstance(img0, Image.Image):
                pil_img = img0.convert("RGB")
            else:
                out_c = ((127.5 * img0 + 128.0) / 255.0).clamp(0, 1)
                pil_img = Image.fromarray((out_c.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            if not release_after_dit:
                entry["pil_img"] = pil_img
            if save_tmp_path:
                Path(save_tmp_path).parent.mkdir(parents=True, exist_ok=True)
                pil_img.save(save_tmp_path, format="PNG")
        del imgs_t
        if self._quota_constrained and is_cuda:
            torch.cuda.empty_cache()

        self._update_hwm()
        return {
            "status": "ok",
            "ctx_id": ctx_id,
            "elapsed_ms": round(elapsed_ms, 2),
            "kv_bytes": int(entry.get("bridge_bytes", 0)),
            "resident_alpha": self.resident_alpha(),
            "kv_free_bytes": self.available_kv_bytes(),
            "yield_count": 0,
        }

    def reset_for_new_suite(self) -> float:
        if self.vmm is not None and self.vlm_pager is not None:
            if self.pager is not None:
                self.pager.set_residency_alpha(0.0)
            self.vlm_pager.set_residency_alpha(float(getattr(self, "_initial_vlm_alpha", 1.0)))
            self.reset_peak_memory_stats()
            return self.resident_alpha()
        return super().reset_for_new_suite()

    def release_context(self, ctx_id: str) -> None:
        import torch

        self._ctx_store.pop(ctx_id, None)
        if (self.vmm is None or self._quota_constrained) and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def memory_and_vmm_stats(self) -> dict[str, Any]:
        stats = super().memory_and_vmm_stats()
        stats["memory_plan"] = dict(self._memory_plan)
        return stats

    def close(self) -> None:
        import torch

        self._ctx_store.clear()
        if self.vlm_pager is not None:
            self.vlm_pager.close()
            self.vlm_pager = None
        if self.pager is not None:
            self.pager.close()
            self.pager = None
        if self.naive_offloader is not None:
            self.naive_offloader.close()
            self.naive_offloader = None
        if self.fission_dropper is not None:
            self.fission_dropper.close()
            self.fission_dropper = None
        if self.vmm is not None:
            self.vmm.shutdown()
            self.vmm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
