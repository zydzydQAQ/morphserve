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

try:
    from transformers.generation.streamers import BaseStreamer as _HFBaseStreamer  # type: ignore
except Exception:
    class _HFBaseStreamer:  # type: ignore[no-redef]
        def put(self, value: Any) -> None:
            return None

        def end(self) -> None:
            return None


class _TokenTimingStreamer(_HFBaseStreamer):
    """Lightweight HuggingFace streamer recording wall-clock `perf_counter()` per generated token step."""

    def __init__(self) -> None:
        self.step_timestamps_s: list[float] = []
        self._prompt_skipped: bool = False

    def put(self, value: Any) -> None:
        ndim = int(getattr(value, "ndim", 1))
        shape = getattr(value, "shape", None)
        if not self._prompt_skipped:
            self._prompt_skipped = True
            # `GenerationMixin.generate()` emits the prompt `input_ids` as a 2D `[B, L_prompt]`
            # tensor (with `L_prompt == 0` when using `inputs_embeds`) before decode step 0.
            if ndim >= 2:
                return
        if ndim >= 2 and shape is not None and len(shape) >= 2 and int(shape[-1]) == 0:
            return
        self.step_timestamps_s.append(time.perf_counter())

    def end(self) -> None:
        return None


def _format_prompt_with_history(
    prompt: str,
    history: list[tuple[str, str]],
    *,
    max_turns: int = 6,
) -> str:
    """Format rolling multi-turn conversation history cleanly into the current VLM turn prompt."""
    if not history:
        return prompt
    lines: list[str] = []
    for old_q, old_a in history[-max(1, int(max_turns)) :]:
        q_s = str(old_q).strip()
        a_s = str(old_a).strip()
        if q_s:
            lines.append(f"User: {q_s}")
        if a_s:
            lines.append(f"Assistant: {a_s}")
    if not lines:
        return prompt
    lines.append(f"User: {str(prompt).strip()}")
    return "\n".join(lines)


def _estimate_vlm_context_bytes(
    history: list[tuple[str, str]],
    prompt: str,
    num_images: int = 0,
) -> int:
    """Estimate accumulated multi-turn VLM context KV footprint (`>= 4096` B) for InternVL-U."""
    text_chars = len(str(prompt)) + sum(len(str(q)) + len(str(a)) for q, a in history)
    est_tokens = max(1, text_chars // 4) + max(0, int(num_images)) * 256
    # InternVL-U Qwen2.5 VLM: 36 layers * 2 (K+V) * 2 KV heads * 128 head_dim * 2 bytes (bf16) = 36,864 B/tok.
    return max(4096, int(est_tokens) * 36_864)


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


def _bridge_tensor_nbytes(val: Any) -> int:
    """Compute exact byte length produced by `pack_internvlu_bridge_state` without allocating/concatenating."""
    import torch

    if val is None:
        return 0
    if torch.is_tensor(val):
        return int(val.numel()) * int(val.element_size())
    if isinstance(val, list):
        return sum(_bridge_tensor_nbytes(x) for x in val)
    if isinstance(val, dict):
        return sum(_bridge_tensor_nbytes(v) for v in val.values())
    return 0


def pack_internvlu_bridge_state(diffusion_inputs: dict[str, Any], device: Any) -> tuple[dict[str, Any], Any]:
    """Pack `InternVL-U` `diffusion_inputs` bridge tensors into a 16B-aligned contiguous `uint8` CUDA tensor.

    Small integer/bool shape-metadata tensors (`<= 64` elements, e.g. `image_grid_thw_gen`,
    `image_fhw_cond`, `image_grid_thw_gen_cond`) are serialized inline in JSON `meta` so
    control-plane grid dimensions are immune to byte-offset/wire corruption.
    """
    import torch

    chunks: list[Any] = []
    offset = 0
    _INLINE_DTYPES = (torch.int64, torch.int32, torch.bool)

    def _rec(key: str, val: Any) -> Any:
        nonlocal offset
        if val is None:
            return {"kind": "none"}
        if torch.is_tensor(val):
            dt_name = _DTYPE_NAME_MAP[str(val.dtype)]
            if val.dtype in _INLINE_DTYPES and int(val.numel()) <= 64:
                return {
                    "kind": "inline_tensor",
                    "dtype": dt_name,
                    "shape": list(val.shape),
                    "values": val.detach().cpu().tolist(),
                }
            pad = (-offset) & 15
            if pad > 0:
                chunks.append(torch.zeros((pad,), dtype=torch.uint8, device=device))
                offset += pad
            t_c = val.to(device).contiguous()
            u8 = t_c.view(torch.uint8).reshape(-1)
            nb = int(u8.numel())
            node = {
                "kind": "tensor",
                "dtype": dt_name,
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
    """Reconstruct `InternVL-U` `diffusion_inputs` dictionary from flat `uint8` CUDA buffer + inline metadata."""
    import torch

    expected_total = int(meta.get("total_bytes", 0))
    actual_total = int(flat_u8.numel()) if hasattr(flat_u8, "numel") else len(flat_u8)
    if actual_total < expected_total:
        raise ValueError(f"InternVL-U bridge payload truncated: got {actual_total} < expected {expected_total} bytes")

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
        if kind == "inline_tensor":
            dt = dtype_lookup[node["dtype"]]
            return torch.tensor(node["values"], dtype=dt, device=flat_u8.device).reshape(node["shape"])
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

        is_static_fission = policy.routing_mode in ("static_nccl_fission", "twostage_shm")
        raw_frac = getattr(policy, "gpu_memory_fraction", None)
        raw_quota = getattr(policy, "gpu_memory_quota_mib", None)
        is_quota_constrained = bool(
            (raw_quota is not None and float(raw_quota) > 0.0)
            or (raw_frac is not None and float(raw_frac) < 0.999)
        )
        self._quota_constrained = is_quota_constrained
        copage_vlm = bool(
            (not is_static_fission)
            and policy.use_tripool_vmm
            and policy.use_expert_pager
            and is_quota_constrained
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
            # Regime A (40 GiB unconstrained) zero-overhead fast-path: skip per-layer forward hooks
            # and scoped VAE transient arena wrappers when VLM + Decoder fit comfortably at alpha=1.0.
            dec_spec = ExpertPartitionSpec.for_internvlu(
                self.pipe,
                super_handle_mib=policy.super_handle_mib,
                wrap_vae=bool(policy.use_scoped_vae and is_quota_constrained),
                install_module_hooks=bool(is_quota_constrained),
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
            self._ensure_nvml_poller()
            self._update_hwm()
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
                use_elastic_split = bool(
                    self._quota_constrained
                    and self.vlm_pager is not None
                    and int(self.policy.num_gpus) >= 2
                    and str(self.policy.routing_mode) == "dual_path"
                )
                if not self._quota_constrained:
                    # Regime A (40 GiB unconstrained): keep Decoder 100% resident (`alpha=1.0`) alongside VLM.
                    init_alpha = 1.0
                elif use_elastic_split:
                    # Regime B (12 GiB constrained, >=2 GPUs): initialize even ranks VLM-hot (`alpha=0.0`)
                    # and odd ranks Decoder-hot (`alpha=1.0`) so dual-path ships ~2-8 MiB bridges across
                    # warm ranks while retaining full on-demand elastic morphing when demand shifts.
                    init_alpha = 1.0 if (self.rank % 2 == 1) else 0.0
                    if not bool(plan["allow_overflow"]):
                        max_fit_layers = int(plan["super_pool_mib"]) // max(1, self.pager.layer_stride_mib)
                        max_fit_alpha = min(1.0, float(max_fit_layers) / float(max(1, self.pager.num_layers)))
                        init_alpha = min(init_alpha, max_fit_alpha)
                else:
                    init_alpha = (
                        0.0 if self.vlm_pager is not None else float(self.policy.initial_alphas.get(self.rank, 1.0))
                    )
                    if not bool(plan["allow_overflow"]):
                        max_fit_layers = int(plan["super_pool_mib"]) // max(1, self.pager.layer_stride_mib)
                        max_fit_alpha = min(1.0, float(max_fit_layers) / float(max(1, self.pager.num_layers)))
                        init_alpha = min(init_alpha, max_fit_alpha)
                self._initial_bound_alpha = float(init_alpha)
                self.pager.bind_vmm(self.vmm, initial_alpha=init_alpha)
            if self.vlm_pager is not None:
                vlm_fit_layers = int(plan["super_pool_mib"]) // max(1, self.vlm_pager.layer_stride_mib)
                vlm_fit_alpha = min(1.0, float(vlm_fit_layers) / float(max(1, self.vlm_pager.num_layers)))
                vlm_init_alpha = 0.0 if float(getattr(self, "_initial_bound_alpha", 0.0)) > 0.0 else vlm_fit_alpha
                self._initial_vlm_alpha = float(vlm_init_alpha)
                self.vlm_pager.bind_vmm(self.vmm, initial_alpha=vlm_init_alpha)
        else:
            self._quota_constrained = bool(
                (getattr(self.policy, "gpu_memory_quota_mib", None) or 0.0) > 0.0
                or (
                    getattr(self.policy, "gpu_memory_fraction", None) is not None
                    and float(getattr(self.policy, "gpu_memory_fraction")) < 0.999
                )
            )

        if is_cuda:
            self._update_hwm()
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

    def _maybe_empty_cache(self, *, min_unused_mib: int = 256) -> None:
        import torch

        if not (self._quota_constrained and torch.cuda.is_available() and getattr(self.device, "type", "cuda") == "cuda"):
            return
        unused_b = max(
            0,
            int(torch.cuda.memory_reserved(self.device)) - int(torch.cuda.memory_allocated(self.device)),
        )
        if unused_b >= int(min_unused_mib) * MIB:
            self._update_hwm()
            torch.cuda.empty_cache()

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

    def _run_vlm_text_generate(
        self,
        prompt_arg: str | list[str],
        image_arg: Any,
        max_new_tokens: int,
    ) -> tuple[Any, float, _TokenTimingStreamer]:
        """Run VLM greedy text generation with `_TokenTimingStreamer` for real TTFT & per-token TPOT."""
        import torch

        streamer = _TokenTimingStreamer()
        t0 = time.perf_counter()
        pipe_gen_fn = getattr(self.pipe, "_generate", None)
        vlm_gen_fn = getattr(getattr(self.pipe, "vlm", None), "generate", None)
        proc_fn = getattr(self.pipe, "processor", None)
        if callable(pipe_gen_fn) and callable(vlm_gen_fn) and callable(proc_fn):
            # Bypass `InternVLUPipeline._split_kwargs`, which drops `streamer` into `unused_kwargs`,
            # while preserving the exact `processor(...) -> _generate(...) -> vlm.generate(...)` path.
            inputs = proc_fn(
                prompt=prompt_arg,
                image=image_arg,
                generation_mode="text",
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
            out = pipe_gen_fn(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                streamer=streamer,
            )
            return out, t0, streamer

        call_kwargs: dict[str, Any] = {
            "prompt": prompt_arg,
            "image": image_arg,
            "generation_mode": "text",
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
        }
        try:
            sig = inspect.signature(self.pipe.__call__).parameters
            if "streamer" in sig or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values()):
                call_kwargs["streamer"] = streamer
        except Exception:
            pass
        out = self.pipe(**call_kwargs)
        return out, t0, streamer

    def prepare_turn(self, ctx_id: str, cmd: dict[str, Any]) -> int:
        import torch

        prompt = str(cmd["prompt"])
        prep_dit = bool(cmd.get("prepare_for_dit", False))
        prev_ent = self._ctx_store.get(ctx_id)
        history: list[tuple[str, str]] = list(prev_ent.get("history") or []) if prev_ent is not None else []
        images: list[Any] = list(prev_ent.get("images") or []) if prev_ent is not None else []

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

        if pil_img is not None and (not images or images[-1] is not pil_img):
            images.append(pil_img)
            if len(images) > 4:
                images = images[-4:]

        if not prep_dit:
            est_b = _estimate_vlm_context_bytes(history, prompt, num_images=len(images))
            self._ctx_store[ctx_id] = {
                "prompt": prompt,
                "pil_img": pil_img,
                "images": images,
                "history": history,
                "bridge_bytes": est_b,
            }
            return est_b

        is_cuda = torch.cuda.is_available() and getattr(self.device, "type", "cuda") == "cuda"
        self._maybe_empty_cache(min_unused_mib=256)

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
        if is_cuda:
            torch.cuda.current_stream(self.device).synchronize()
        self._update_hwm()
        self._maybe_empty_cache(min_unused_mib=256)
        b_bytes = _bridge_tensor_nbytes(diff_inp)
        self._ctx_store[ctx_id] = {
            "prompt": prompt,
            "pil_img": pil_img,
            "images": images,
            "history": history,
            "diff_inp": diff_inp,
            "gh": gh,
            "gw": gw,
            "bridge_bytes": b_bytes,
        }
        return b_bytes

    def decode_ar(self, ctx_id: str, max_tokens: int) -> dict[str, Any]:
        import torch

        entry = self._ctx_store[ctx_id]
        prompt = str(entry["prompt"])
        history: list[tuple[str, str]] = list(entry.get("history") or [])
        eff_prompt = _format_prompt_with_history(prompt, history)
        pil_img = entry.get("pil_img")
        is_cuda = torch.cuda.is_available() and getattr(self.device, "type", "cuda") == "cuda"
        self._maybe_empty_cache(min_unused_mib=256)
        with self._vlm_execution_scope(), torch.inference_mode():
            out, t0, streamer = self._run_vlm_text_generate(eff_prompt, pil_img, int(max_tokens))
        if is_cuda:
            torch.cuda.current_stream(self.device).synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._update_hwm()
        self._maybe_empty_cache(min_unused_mib=256)

        tok_seq = _trim_internvlu_generated_tokens(out.generate_output[0], self.pipe.tokenizer)
        n_tok = max(1, int(len(tok_seq)))
        txt = self.pipe.tokenizer.decode(tok_seq, skip_special_tokens=True) if tok_seq else ""

        stamps = streamer.step_timestamps_s
        if stamps:
            used_stamps = stamps[:n_tok]
            ttft_ms = max(0.0, (used_stamps[0] - t0) * 1000.0)
            tpots_ms = [
                round(max(0.0, (used_stamps[i] - used_stamps[i - 1]) * 1000.0), 2)
                for i in range(1, len(used_stamps))
            ]
        else:
            per_tok = dt_ms / float(n_tok)
            ttft_ms = per_tok
            tpots_ms = [round(per_tok, 2) for _ in range(max(0, n_tok - 1))]

        history.append((prompt, txt))
        entry["history"] = history
        kv_b = max(
            int(entry.get("bridge_bytes", 4096)),
            _estimate_vlm_context_bytes(history, prompt, num_images=len(entry.get("images") or [])),
        )
        entry["bridge_bytes"] = kv_b
        return {
            "text": txt,
            "tokens": n_tok,
            "ttft_ms": round(ttft_ms, 2),
            "tpots_ms": tpots_ms,
            "kv_bytes": kv_b,
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
        raw_prompts: list[str] = []
        prompts: list[str] = []
        max_toks: list[int] = []
        for cmd in batch_cmds:
            cid = str(cmd["ctx_id"])
            if not cmd.get("_skip_prepare", False):
                self.prepare_turn(cid, cmd)
            ent = self._ctx_store[cid]
            raw_p = str(ent["prompt"])
            raw_prompts.append(raw_p)
            prompts.append(_format_prompt_with_history(raw_p, list(ent.get("history") or [])))
            max_toks.append(max(1, int(cmd.get("max_ar_tokens", 32))))

        is_cuda = torch.cuda.is_available() and getattr(self.device, "type", "cuda") == "cuda"
        self._maybe_empty_cache(min_unused_mib=256)

        with self._vlm_execution_scope(), torch.inference_mode():
            out, t_gen0, streamer = self._run_vlm_text_generate(prompts, None, max(max_toks))
        if is_cuda:
            torch.cuda.current_stream(self.device).synchronize()
        dt_ms = (time.perf_counter() - t_gen0) * 1000.0
        self._update_hwm()
        self._maybe_empty_cache(min_unused_mib=256)

        alpha_now = self.resident_alpha()
        free_kv_now = self.available_kv_bytes()
        trimmed_seqs = [
            _trim_internvlu_generated_tokens(out.generate_output[idx][: max_toks[idx]], self.pipe.tokenizer)
            for idx in range(len(batch_cmds))
        ]
        max_steps_executed = max((max(1, len(s)) for s in trimmed_seqs), default=1)
        per_step_ms = dt_ms / float(max_steps_executed)
        stamps = streamer.step_timestamps_s
        results = []
        for idx, cmd in enumerate(batch_cmds):
            cid = str(cmd["ctx_id"])
            ent = self._ctx_store[cid]
            enq_s = self.normalized_enqueue_time_s(cmd)
            q_wait_ms = max(0.0, (t_start - enq_s) * 1000.0) if enq_s > 0.0 else 0.0
            seq = trimmed_seqs[idx]
            n_tok = max(1, int(len(seq)))
            txt = self.pipe.tokenizer.decode(seq, skip_special_tokens=True) if seq else ""
            if stamps:
                used_stamps = stamps[:n_tok]
                req_ttft_ms = q_wait_ms + max(0.0, (used_stamps[0] - t_start) * 1000.0)
                req_tpots_ms = [
                    round(max(0.0, (used_stamps[k] - used_stamps[k - 1]) * 1000.0), 2)
                    for k in range(1, len(used_stamps))
                ]
            else:
                req_ttft_ms = q_wait_ms + per_step_ms
                req_tpots_ms = [round(per_step_ms, 2) for _ in range(max(0, n_tok - 1))]

            hist = list(ent.get("history") or [])
            hist.append((raw_prompts[idx], txt))
            ent["history"] = hist
            kv_b = max(
                int(ent.get("bridge_bytes", 4096)),
                _estimate_vlm_context_bytes(hist, raw_prompts[idx], num_images=len(ent.get("images") or [])),
            )
            ent["bridge_bytes"] = kv_b
            results.append(
                {
                    "text": txt,
                    "tokens": n_tok,
                    "ttft_ms": round(req_ttft_ms, 2),
                    "tpots_ms": req_tpots_ms,
                    "kv_bytes": kv_b,
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
        meta["prompt"] = str(entry.get("prompt", ""))
        meta["history"] = [[str(q), str(a)] for q, a in (entry.get("history") or [])]
        return PackedTransferState(meta=meta, payload=flat, logical_bytes=total_b)

    def unpack_state(self, new_ctx_id: str, meta: dict[str, Any], raw_payload: Any) -> int:
        diff_inp = unpack_internvlu_bridge_state(meta, raw_payload)
        total_b = int(meta["total_bytes"])
        restored_hist = [
            (str(pair[0]), str(pair[1]))
            for pair in (meta.get("history") or [])
            if isinstance(pair, (list, tuple)) and len(pair) >= 2
        ]
        self._ctx_store[new_ctx_id] = {
            "prompt": str(meta.get("prompt", "")),
            "history": restored_hist,
            "images": [],
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
        self._maybe_empty_cache(min_unused_mib=256)

        vmm_st0 = self.vmm.stats_dict() if (self.pager is not None and self.vmm is not None) else None
        h2d_weight_ms = 0.0
        vae_ms = 0.0
        t0 = time.perf_counter()
        if self.naive_offloader is not None:
            t_sw0 = time.perf_counter()
            if self.naive_offloader.swap_in_sync():
                h2d_weight_ms += (time.perf_counter() - t_sw0) * 1000.0

        # Regime A (40 GiB unconstrained) zero-overhead fast-path: when Decoder is already 100% resident
        # (`alpha=1.0`), bypass `generation_mode`, per-layer waits, and latent/VAE split cleanly.
        use_regime_a_fast_path = (
            (not self._quota_constrained)
            and self.pager is not None
            and self.vmm is not None
            and self.vmm.expert_resident_count(self.pager.expert_id) == self.pager.num_layers
        )

        gen = torch.Generator(device=self.device).manual_seed(seed)
        try:
            if self.vlm_pager is not None and self.vmm is not None and self.vlm_pager.expert_id >= 0:
                self.vmm.set_expert_pinned(self.vlm_pager.expert_id, False)
            with torch.inference_mode():
                if self.pager is not None and self.vmm is not None and not use_regime_a_fast_path:
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
                    self._maybe_empty_cache(min_unused_mib=128)
                    t_vae0 = time.perf_counter()
                    imgs_t = self.pipe.image_pipeline.latents_to_pixels(lat_bf16)
                    if is_cuda:
                        torch.cuda.current_stream(self.device).synchronize()
                    vae_ms = (time.perf_counter() - t_vae0) * 1000.0
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
            if is_cuda:
                try:
                    torch.cuda.current_stream(self.device).synchronize()
                except Exception:
                    pass
            # Always sample peak HBM BEFORE dropping naive-offloaded decoder weights!
            self._update_hwm()
            if self.naive_offloader is not None:
                self.naive_offloader.drop_from_gpu()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if vmm_st0 is not None and self.vmm is not None:
            vmm_st1 = self.vmm.stats_dict()
            d_map_us = max(0.0, float(vmm_st1.get("expert_map_us", 0.0)) - float(vmm_st0.get("expert_map_us", 0.0)))
            d_sub_us = max(
                0.0,
                float(vmm_st1.get("expert_h2d_submit_us", 0.0)) - float(vmm_st0.get("expert_h2d_submit_us", 0.0)),
            )
            d_wait_us = max(
                0.0,
                float(vmm_st1.get("expert_wait_us", 0.0)) - float(vmm_st0.get("expert_wait_us", 0.0)),
            )
            h2d_weight_ms += (d_map_us + d_sub_us + d_wait_us) / 1000.0

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
                imgs_hist = list(entry.get("images") or [])
                imgs_hist.append(pil_img)
                entry["images"] = imgs_hist[-4:]
            if save_tmp_path:
                Path(save_tmp_path).parent.mkdir(parents=True, exist_ok=True)
                pil_img.save(save_tmp_path, format="PNG")
        del imgs_t
        self._update_hwm()
        self._maybe_empty_cache(min_unused_mib=256)

        return {
            "status": "ok",
            "ctx_id": ctx_id,
            "elapsed_ms": round(elapsed_ms, 2),
            "h2d_weight_ms": round(h2d_weight_ms, 2),
            "vae_ms": round(vae_ms, 2),
            "preempt_yield_ms": 0.0,
            "kv_bytes": int(entry.get("bridge_bytes", 0)),
            "resident_alpha": self.resident_alpha(),
            "kv_free_bytes": self.available_kv_bytes(),
            "yield_count": 0,
        }

    def reset_for_new_suite(self) -> float:
        if self.vmm is not None and self.vlm_pager is not None:
            init_dec = float(getattr(self, "_initial_bound_alpha", 0.0))
            init_vlm = float(getattr(self, "_initial_vlm_alpha", 1.0))
            if self.pager is not None and init_dec <= 0.0:
                self.pager.set_residency_alpha(0.0)
            if init_vlm <= 0.0:
                self.vlm_pager.set_residency_alpha(0.0)
            if self.pager is not None and init_dec > 0.0:
                self.pager.set_residency_alpha(init_dec)
            if init_vlm > 0.0:
                self.vlm_pager.set_residency_alpha(init_vlm)
            self.reset_peak_memory_stats()
            return self.resident_alpha()
        return super().reset_for_new_suite()

    def release_context(self, ctx_id: str) -> None:
        import torch

        self._ctx_store.pop(ctx_id, None)
        if self.vmm is None and torch.cuda.is_available():
            self._update_hwm()
            torch.cuda.empty_cache()
        elif self._quota_constrained:
            self._maybe_empty_cache(min_unused_mib=256)

    def memory_and_vmm_stats(self) -> dict[str, Any]:
        stats = super().memory_and_vmm_stats()
        stats["memory_plan"] = dict(self._memory_plan)
        return stats

    def close(self) -> None:
        import torch

        self._stop_nvml_poller()
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
