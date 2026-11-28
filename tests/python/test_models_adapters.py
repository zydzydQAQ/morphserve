"""Seam tests for `CPTGModelAdapter`, `BagelCPTGAdapter`, `InternVLUCPTGAdapter`, `NaiveHostOffloader`, and `StaticFissionDropper`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image
import pytest

from morphserve.engine.config import EnginePolicyConfig
from morphserve.models import (
    BagelCPTGAdapter,
    InternVLUCPTGAdapter,
    SyntheticCPTGAdapter,
    SyntheticOutOfMemoryError,
    is_oom_exception,
)
from morphserve.vmm import (
    ExpertPartitionSpec,
    NaiveHostOffloader,
    StaticFissionDropper,
    VmmOutOfMemoryError,
)


# ==================== 1. Pure-Python OOM Classification & SyntheticCPTGAdapter ====================


def test_is_oom_exception_recognizes_typed_errors() -> None:
    assert is_oom_exception(VmmOutOfMemoryError("pool exhausted")) is True
    assert is_oom_exception(SyntheticOutOfMemoryError("injected")) is True
    assert is_oom_exception(MemoryError("out of memory")) is True
    assert (
        is_oom_exception(
            RuntimeError(
                'torch_call_dispatcher( "aten::new_empty", "", stack.data(), TORCH_ABI_VERSION) API call failed'
            )
        )
        is True
    )
    assert is_oom_exception(ValueError("bad argument")) is False
    assert is_oom_exception(RuntimeError("unrelated transport failure")) is False


def test_synthetic_cptg_adapter_public_contract(tmp_path: Path) -> None:
    pol = EnginePolicyConfig.from_preset("morphserve", num_gpus=2, model_family="synthetic")
    adapter = SyntheticCPTGAdapter(rank=0, policy=pol, max_kv_region_mib=64)
    try:
        assert adapter.available_kv_bytes() > 0
        kv_b = adapter.prepare_turn("s0", {"prompt": "draw mountain", "prepare_for_dit": True, "height": 512, "width": 512})
        assert kv_b > 0

        ar_batch = adapter.decode_ar_batch(
            [
                {"ctx_id": "s_ar0", "prompt": "alpha", "max_ar_tokens": 3},
                {"ctx_id": "s_ar1", "prompt": "beta", "max_ar_tokens": 3},
            ]
        )
        assert len(ar_batch) == 2
        assert all(r["tokens"] >= 1 for r in ar_batch)

        out_png = tmp_path / "synth_out.png"
        dit_res = adapter.generate_dit(
            "s0",
            {"height": 512, "width": 512, "dit_steps": 2, "save_tmp_image_path": str(out_png)},
        )
        assert dit_res["status"] == "ok"
        assert out_png.is_file()

        packed = adapter.pack_state("s0")
        assert packed.logical_bytes > 0
        unpacked_b = adapter.unpack_state("s0_copy", packed.meta, packed.payload)
        assert unpacked_b == packed.logical_bytes

        adapter.reset_peak_memory_stats()
        stats = adapter.memory_and_vmm_stats()
        assert "peak_reserved_mib" in stats
        adapter.release_context("s0")
        adapter.release_context("s0_copy")
    finally:
        adapter.close()


# ==================== 2. Offline CPU Stub Builders for BAGEL & InternVL-U ====================


def _build_bagel_stub(torch: Any, *, num_layers: int = 2, hidden: int = 8, num_kv_heads: int = 2, head_dim: int = 4):
    """Minimal CPU BAGEL pipeline stub implementing the exact public contract expected by `BagelCPTGAdapter`."""

    class _StubNaiveCache:
        def __init__(self, n_layers: int) -> None:
            self.num_layers = int(n_layers)
            self.key_cache: dict[int, Any] = {i: None for i in range(self.num_layers)}
            self.value_cache: dict[int, Any] = {i: None for i in range(self.num_layers)}

    class _StubAttn(torch.nn.Module):
        def __init__(self, layer_idx: int) -> None:
            super().__init__()
            self.layer_idx = layer_idx
            self.num_heads = num_kv_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.q_size = num_kv_heads * head_dim
            self.kv_size = num_kv_heads * head_dim
            self.hidden_size = hidden
            self.moe_gen_qkv = torch.nn.Parameter(torch.ones((hidden, hidden), dtype=torch.float32))
            self.und_qkv = torch.nn.Parameter(torch.ones((hidden, hidden), dtype=torch.float32))

        def _forward_und(self, packed_query_sequence, packed_query_position_embeddings, past_key_values, is_causal, update_past_key_values=True):
            return packed_query_sequence, past_key_values

        def _forward_gen(self, packed_query_sequence, query_lens, packed_query_position_embeddings, past_key_values, packed_vae_token_indexes, packed_text_indexes, update_past_key_values=False):
            return packed_query_sequence, past_key_values

    class _StubLayer(torch.nn.Module):
        def __init__(self, layer_idx: int) -> None:
            super().__init__()
            self.self_attn = _StubAttn(layer_idx)
            self.mlp_moe_gen = torch.nn.Linear(hidden, hidden, bias=False)
            self.mlp_und = torch.nn.Linear(hidden, hidden, bias=False)

    class _StubVae(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((4, 4), dtype=torch.float32))

        def encode(self, x: Any) -> Any:
            return x

        def decode(self, lat: Any) -> Any:
            return lat

    class _StubLMModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([_StubLayer(i) for i in range(num_layers)])
            self.fail_ar_oom = False

        def forward(self, **kwargs: Any) -> Any:
            pkv = kwargs["past_key_values"]
            tok = kwargs["packed_text_ids"]
            for lidx in range(num_layers):
                k_new = torch.full((1, num_kv_heads, head_dim), float(lidx + 1), dtype=torch.float32)
                v_new = torch.full((1, num_kv_heads, head_dim), float(lidx + 10), dtype=torch.float32)
                prev_k = pkv.key_cache[lidx]
                prev_v = pkv.value_cache[lidx]
                pkv.key_cache[lidx] = k_new if prev_k is None else torch.cat([prev_k, k_new], dim=0)
                pkv.value_cache[lidx] = v_new if prev_v is None else torch.cat([prev_v, v_new], dim=0)
                if self.fail_ar_oom and lidx == 0:
                    raise VmmOutOfMemoryError("injected AR step OOM after layer 0 append")
            seq = torch.zeros((int(tok.shape[0]), hidden), dtype=torch.float32)
            return SimpleNamespace(packed_query_sequence=seq, past_key_values=pkv)

    class _StubLanguageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _StubLMModel()
            self.lm_head = torch.nn.Linear(hidden, 32, bias=False)
            with torch.no_grad():
                self.lm_head.weight.zero_()
                self.lm_head.weight[5, :] = 1.0

        def forward(self, **kwargs: Any) -> Any:
            return self.model(**kwargs)

    class _StubBagel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = _StubLanguageModel()
            self.fail_text_oom_on_call: int | None = None
            self.text_calls: int = 0
            self.vit_autocast_flags: list[bool] = []
            self.vae_calls: int = 0

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return None

        def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
            n_tok = max(2, len(str(prompts[0]).split()))
            return (
                {"packed_text_ids": torch.arange(n_tok, dtype=torch.long)},
                [int(curr_kvlens[0]) + n_tok],
                [int(curr_rope[0]) + n_tok],
            )

        def forward_cache_update_text(self, pkv, **inp):
            self.text_calls += 1
            n_tok = int(inp["packed_text_ids"].shape[0])
            for lidx in range(pkv.num_layers):
                k_new = torch.full((n_tok, num_kv_heads, head_dim), float(self.text_calls + lidx), dtype=torch.float32)
                v_new = torch.full((n_tok, num_kv_heads, head_dim), float(self.text_calls * 2 + lidx), dtype=torch.float32)
                prev_k = pkv.key_cache[lidx]
                prev_v = pkv.value_cache[lidx]
                pkv.key_cache[lidx] = k_new if prev_k is None else torch.cat([prev_k, k_new], dim=0)
                pkv.value_cache[lidx] = v_new if prev_v is None else torch.cat([prev_v, v_new], dim=0)
                if self.fail_text_oom_on_call is not None and self.text_calls == self.fail_text_oom_on_call and lidx == 0:
                    raise VmmOutOfMemoryError("injected prepare_turn OOM after layer 0 mutation")
            return pkv

        def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
            t_img = transforms(images[0])
            return {"padded_latent": t_img.unsqueeze(0)}, [int(curr_kvlens[0]) + 4], [int(curr_rope[0]) + 4]

        def forward_cache_update_vae(self, vae, pkv, **vi):
            self.vae_calls += 1
            for lidx in range(pkv.num_layers):
                k_new = torch.ones((4, num_kv_heads, head_dim), dtype=torch.float32)
                v_new = torch.ones((4, num_kv_heads, head_dim), dtype=torch.float32)
                prev_k = pkv.key_cache[lidx]
                prev_v = pkv.value_cache[lidx]
                pkv.key_cache[lidx] = k_new if prev_k is None else torch.cat([prev_k, k_new], dim=0)
                pkv.value_cache[lidx] = v_new if prev_v is None else torch.cat([prev_v, v_new], dim=0)
            return pkv

        def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
            self.vit_autocast_flags.append(bool(torch.is_autocast_enabled()))
            _ = transforms(images[0])
            return {"packed_vit_tokens": torch.ones((3, hidden), dtype=torch.float32)}, [int(curr_kvlens[0]) + 3], [int(curr_rope[0]) + 3]

        def forward_cache_update_vit(self, pkv, **vti):
            for lidx in range(pkv.num_layers):
                k_new = torch.full((3, num_kv_heads, head_dim), 2.0, dtype=torch.float32)
                v_new = torch.full((3, num_kv_heads, head_dim), 3.0, dtype=torch.float32)
                prev_k = pkv.key_cache[lidx]
                prev_v = pkv.value_cache[lidx]
                pkv.key_cache[lidx] = k_new if prev_k is None else torch.cat([prev_k, k_new], dim=0)
                pkv.value_cache[lidx] = v_new if prev_v is None else torch.cat([prev_v, v_new], dim=0)
            return pkv

        def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
            return {
                "packed_start_tokens": torch.tensor([1], dtype=torch.long),
                "packed_query_position_ids": torch.tensor([int(curr_rope[0])], dtype=torch.long),
            }

        def generate_image(self, **kwargs: Any) -> tuple[Any]:
            return (torch.zeros((1, 3, 8, 8), dtype=torch.float32),)

    class _StubTokenizer:
        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            return " ".join(f"tok{i}" for i in ids)

    class _StubImageProcessor:
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.5, 0.5, 0.5]

        def __call__(self, images: Any, return_tensors: str = "pt") -> Any:
            return SimpleNamespace(pixel_values=torch.zeros((1, 3, 16, 16), dtype=torch.float32))

    bagel = _StubBagel()
    vae = _StubVae()

    def _forward_single(req: Any, sampling: Any, prepare_only: bool = True) -> dict[str, Any]:
        return {
            "gen_params": SimpleNamespace(
                num_timesteps=int(sampling.num_inference_steps),
                timestep_shift=3.0,
                cfg_text_scale=4.0,
                cfg_img_scale=1.5,
                cfg_interval=[0.4, 1.0],
                cfg_renorm_min=0.0,
                cfg_renorm_type="global",
            ),
            "image_shape": [int(sampling.height), int(sampling.width)],
            "gen_context": {"past_key_values": getattr(sampling, "past_key_values", None)},
            "cfg_text_context": {"past_key_values": getattr(sampling, "cfg_text_past_key_values", None)},
            "cfg_img_context": {"past_key_values": getattr(sampling, "cfg_img_past_key_values", None)},
            "generation_input": {"packed_init_noises": torch.zeros((4, hidden), dtype=torch.float32)},
            "cfg_text_packed_position_ids": None,
            "cfg_img_packed_position_ids": None,
        }

    def _decode_image_from_latent(bagel_mod: Any, vae_mod: Any, lat: Any, image_shape: list[int]) -> list[Any]:
        return [Image.new("RGB", (16, 16), color=(24, 96, 180))]

    pipeline = SimpleNamespace(
        device=torch.device("cpu"),
        bagel=bagel,
        vae=vae,
        tokenizer=_StubTokenizer(),
        image_processor=_StubImageProcessor(),
        new_token_ids={"eos_token_id": 99, "bos_token_id": 1, "start_of_image": 2, "end_of_image": 3},
        _forward_single=_forward_single,
        _decode_image_from_latent=_decode_image_from_latent,
        scheduler=None,
        scheduler_kwargs={},
    )
    return pipeline, _StubNaiveCache


def _build_internvlu_stub(torch: Any):
    """Minimal CPU InternVL-U pipeline stub implementing the exact public contract expected by `InternVLUCPTGAdapter`."""

    class _StubBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones((4, 4), dtype=torch.float32))

    class _StubDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList([_StubBlock(), _StubBlock()])

    class _StubVlm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones((4, 4), dtype=torch.float32))

        def generate_hidden_states(self, input_ids, attention_mask, pixel_values=None):
            b, s = input_ids.shape
            return SimpleNamespace(hidden_states=torch.ones((b, s, 8), dtype=torch.float32))

    class _StubVae(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones((4, 4), dtype=torch.float32))

        def encode(self, x: Any) -> Any:
            return x

        def decode(self, x: Any) -> Any:
            return x

    class _StubImagePipeline:
        def __init__(self) -> None:
            self.pixels_to_latents_calls: int = 0
            self.call_kwargs_history: list[dict[str, Any]] = []

        def pixels_to_latents(self, x: Any) -> Any:
            self.pixels_to_latents_calls += 1
            return x

        def latents_to_pixels(self, lat: Any) -> Any:
            return torch.full((1, 3, 16, 16), 0.25, dtype=torch.float32)

        def __call__(self, **kwargs: Any) -> Any:
            self.call_kwargs_history.append(dict(kwargs))
            return SimpleNamespace(images=torch.full((1, 3, 16, 16), 0.25, dtype=torch.float32))

    class _StubProcessor:
        def __init__(self) -> None:
            self.image_gen_processor = SimpleNamespace(merge_size=2, patch_size=8)

        def __call__(self, prompt, image=None, generation_mode="image", padding=True, return_tensors="pt", height=512, width=512):
            out = {
                "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
                "generation_flags": torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
            }
            if image is not None:
                out["pixel_values"] = torch.ones((1, 3, 16, 16), dtype=torch.float32)
                out["pixel_values_gen"] = torch.ones((1, 3, 16, 16), dtype=torch.float32)
            return out

    class _StubTokenizer:
        pad_token_id = 0
        eos_token_id = 99
        unk_token_id = 98

        def convert_tokens_to_ids(self, tok: str) -> int:
            return 98

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            return " ".join(f"w{i}" for i in ids)

    class _StubPipeline:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.generation_decoder = _StubDecoder()
            self.vlm = _StubVlm()
            self.vae = _StubVae()
            self.image_pipeline = _StubImagePipeline()
            self.processor = _StubProcessor()
            self.tokenizer = _StubTokenizer()
            self.prepare_diff_calls: int = 0

        def _prepare_diffusion_inputs(self, **kwargs: Any) -> dict[str, Any]:
            self.prepare_diff_calls += 1
            return {
                "prompt_embeds": kwargs["vlm_hidden_states"].clone(),
                "cond_flag": int(kwargs.get("pixel_values_gen") is not None),
            }

        def __call__(self, prompt, image=None, generation_mode="text", max_new_tokens=8, do_sample=False):
            prompts = prompt if isinstance(prompt, list) else [prompt]
            # Emit genuine tokens followed by EOS (99) and right-padding (0) to verify trimming.
            rows = []
            for idx, _ in enumerate(prompts):
                row = [10 + idx, 20 + idx, 99, 0, 0][: max(3, int(max_new_tokens))]
                rows.append(row)
            return SimpleNamespace(generate_output=torch.tensor(rows, dtype=torch.long))

    return _StubPipeline()


# ==================== 3. BagelCPTGAdapter Public Seams ====================


def test_bagel_adapter_cascade_cfg_runtime_wiring_and_atomic_oom_rollback(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    # 1. Verify `allow_cascade_cfg_split` is wired at runtime onto `adapter.interceptor` (no source grep!).
    for flag in (False, True):
        pipe, cache_cls = _build_bagel_stub(torch)
        pol = EnginePolicyConfig.from_preset(
            "morphserve",
            num_gpus=1,
            use_tripool_vmm=False,
            use_expert_pager=False,
            use_module3_cow_loopback=True,
            allow_cascade_cfg_split=flag,
            preemption_mode="non_preemptive",
        )
        adapter = BagelCPTGAdapter(
            rank=0,
            policy=pol,
            bench_root=tmp_path,
            pipeline=pipe,
            naive_cache_cls=cache_cls,
            device=torch.device("cpu"),
        )
        try:
            adapter.bind_runtime_resources()
            assert adapter.interceptor.cfg_enabled is True
            assert adapter.interceptor.cfg_allow_cascade_split is flag
        finally:
            adapter.close()

    # 2. Verify atomic KV rollback on OOM in `prepare_turn` AND `decode_ar_batch`.
    pipe, cache_cls = _build_bagel_stub(torch)
    pol = EnginePolicyConfig.from_preset(
        "morphserve",
        num_gpus=1,
        use_tripool_vmm=False,
        use_expert_pager=False,
        use_module3_cow_loopback=False,
        preemption_mode="non_preemptive",
    )
    adapter = BagelCPTGAdapter(
        rank=0,
        policy=pol,
        bench_root=tmp_path,
        pipeline=pipe,
        naive_cache_cls=cache_cls,
        device=torch.device("cpu"),
    )
    try:
        adapter.bind_runtime_resources()
        b0 = adapter.prepare_turn("s1", {"prompt": "first clean turn", "prepare_for_dit": False})
        packed_before = adapter.pack_state("s1")
        assert b0 == packed_before.logical_bytes and b0 > 0

        # Inject mid-layer OOM on the next `prepare_turn` call.
        pipe.bagel.fail_text_oom_on_call = pipe.bagel.text_calls + 1
        with pytest.raises(VmmOutOfMemoryError):
            adapter.prepare_turn("s1", {"prompt": "second failing turn", "prepare_for_dit": False})

        # KV state must be rolled back atomically to the exact pre-turn length & bytes across all layers!
        packed_after_prep = adapter.pack_state("s1")
        assert packed_after_prep.logical_bytes == packed_before.logical_bytes
        assert torch.equal(packed_after_prep.payload, packed_before.payload)

        # Inject mid-layer OOM during `decode_ar_batch`.
        pipe.bagel.fail_text_oom_on_call = None
        pipe.bagel.language_model.model.fail_ar_oom = True
        with pytest.raises(VmmOutOfMemoryError):
            adapter.decode_ar_batch([{"ctx_id": "s1", "max_ar_tokens": 2, "_skip_prepare": True}])

        packed_after_ar = adapter.pack_state("s1")
        assert packed_after_ar.logical_bytes == packed_before.logical_bytes
        assert torch.equal(packed_after_ar.payload, packed_before.payload)
    finally:
        adapter.close()


def test_bagel_adapter_non_vmm_host_swap_and_unified_image_update(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    img_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color=(40, 120, 200)).save(img_path)

    pipe, cache_cls = _build_bagel_stub(torch)
    pol = EnginePolicyConfig.from_preset(
        "monolithic_dp",
        num_gpus=1,
        use_tripool_vmm=False,
        use_expert_pager=False,
        use_module3_cow_loopback=False,
        preemption_mode="non_preemptive",
    )
    adapter = BagelCPTGAdapter(
        rank=0,
        policy=pol,
        bench_root=tmp_path,
        pipeline=pipe,
        naive_cache_cls=cache_cls,
        device=torch.device("cpu"),
    )
    try:
        adapter.bind_runtime_resources()
        adapter.prepare_turn("s_idle", {"prompt": "alpha idle turn", "prepare_for_dit": False})
        adapter.prepare_turn("s_keep", {"prompt": "beta active turn", "prepare_for_dit": False})
        before_swap = adapter.pack_state("s_idle")

        # Evict idle context to host pinned bundle via canonical serializer.
        reclaimed = adapter.evict_idle_kv_to_host(target_free_bytes=1 << 40, exclude_ctx_ids=("s_keep",))
        assert reclaimed > 0
        st_out = adapter.memory_and_vmm_stats()
        assert st_out["kv_host_swap_outs"] == 1
        assert st_out["kv_host_swap_ins"] == 0

        # Accessing `pack_state("s_idle")` restores from host bundle bit-for-bit and increments swap-ins!
        after_restore = adapter.pack_state("s_idle")
        assert torch.equal(after_restore.payload, before_swap.payload)
        st_in = adapter.memory_and_vmm_stats()
        assert st_in["kv_host_swap_ins"] == 1

        # Unified PIL image update must run ViT preparation strictly outside `autocast(bf16)`.
        adapter.prepare_turn(
            "s_keep",
            {"prompt": "edit image", "image_paths": [str(img_path)], "prepare_for_dit": True},
        )
        assert pipe.bagel.vit_autocast_flags == [False]
        assert pipe.bagel.vae_calls >= 1

        # Generate DiT image and verify output PNG + follow-up `use_prev_image=True`.
        out_png = tmp_path / "bagel_dit.png"
        dit_out = adapter.generate_dit(
            "s_keep",
            {"height": 512, "width": 512, "dit_steps": 2, "prompt": "edit image", "save_tmp_image_path": str(out_png)},
        )
        assert dit_out["status"] == "ok"
        assert out_png.is_file()

        # Non-CoW `use_prev_image=True` reuses in-memory `last_img_pil` directly (no PNG disk reload)
        sess_keep = adapter._ctx_store["s_keep"]
        assert sess_keep.last_img_pil is not None
        load_calls: list[str] = []
        orig_load_pil = adapter._load_pil

        def _spy_load_pil(rel_path: str) -> Any:
            load_calls.append(rel_path)
            return orig_load_pil(rel_path)

        adapter._load_pil = _spy_load_pil  # type: ignore[method-assign]
        adapter.prepare_turn("s_keep", {"prompt": "inspect prev", "use_prev_image": True, "prepare_for_dit": False})
        assert load_calls == []
    finally:
        adapter.close()


def test_bagel_adapter_q1_pending_vae_idle_kv_swap_retry_and_hwm_order(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    img_path = tmp_path / "q1_src.png"
    Image.new("RGB", (32, 32), color=(110, 70, 190)).save(img_path)

    # 1. Q1 Turn 0 (`ar_only` + image) records `pending_vae_only_paths` -> Turn 1 (`dit_edit`) injects VAE once without duplicating ViT
    pipe, cache_cls = _build_bagel_stub(torch)
    pol_swap = EnginePolicyConfig.from_preset(
        "monolithic_kv_swap",
        num_gpus=1,
        use_tripool_vmm=False,
        use_expert_pager=False,
        use_module3_cow_loopback=False,
        preemption_mode="non_preemptive",
    )
    adapter = BagelCPTGAdapter(
        rank=0,
        policy=pol_swap,
        bench_root=tmp_path,
        pipeline=pipe,
        naive_cache_cls=cache_cls,
        device=torch.device("cpu"),
    )
    try:
        adapter.bind_runtime_resources()
        adapter.prepare_turn(
            "q1_sess",
            {"prompt": "describe source image", "image_paths": [str(img_path)], "prepare_for_dit": False},
        )
        assert len(pipe.bagel.vit_autocast_flags) == 1
        assert pipe.bagel.vae_calls == 0
        assert adapter._ctx_store["q1_sess"].pending_vae_only_paths == [str(img_path)]

        # Turn 1: `dit_edit` with no explicit `image_paths` consumes `pending_vae_only_paths` for VAE-only update!
        adapter.prepare_turn(
            "q1_sess",
            {"prompt": "turn into oil painting", "image_paths": [], "use_prev_image": False, "prepare_for_dit": True},
        )
        assert pipe.bagel.vae_calls == 1
        assert len(pipe.bagel.vit_autocast_flags) == 1
        assert adapter._ctx_store["q1_sess"].pending_vae_only_paths == []

        # 2. `monolithic_kv_swap` (`use_idle_kv_swap=True`) recovers from transient OOM by evicting idle KV and retrying!
        adapter.prepare_turn("idle_victim", {"prompt": "old idle context", "prepare_for_dit": False})
        pipe.bagel.fail_text_oom_on_call = pipe.bagel.text_calls + 1
        kv_after_retry = adapter.prepare_turn("fresh_req", {"prompt": "retry after swap", "prepare_for_dit": False})
        assert kv_after_retry > 0
        assert adapter.memory_and_vmm_stats()["kv_host_swap_outs"] >= 1
    finally:
        adapter.close()

    # 3. Verify `_update_hwm()` executes BEFORE `naive_offloader.drop_from_gpu()` in `generate_dit` teardown.
    pipe2, cache_cls2 = _build_bagel_stub(torch)
    pol_naive = EnginePolicyConfig.from_preset(
        "naive_host_offload",
        num_gpus=1,
        use_tripool_vmm=False,
        use_expert_pager=False,
        use_naive_offload=True,
        use_module3_cow_loopback=False,
        preemption_mode="non_preemptive",
    )
    adapter2 = BagelCPTGAdapter(
        rank=0,
        policy=pol_naive,
        bench_root=tmp_path,
        pipeline=pipe2,
        naive_cache_cls=cache_cls2,
        device=torch.device("cpu"),
    )
    try:
        adapter2.bind_runtime_resources()
        assert adapter2.naive_offloader is not None
        adapter2.prepare_turn("n0", {"prompt": "draw bridge", "prepare_for_dit": True})
        order: list[str] = []
        orig_hwm = adapter2._update_hwm
        orig_drop = adapter2.naive_offloader.drop_from_gpu

        def _rec_hwm() -> None:
            order.append("hwm")
            orig_hwm()

        def _rec_drop(*, force: bool = False) -> None:
            order.append("drop")
            orig_drop(force=force)

        adapter2._update_hwm = _rec_hwm  # type: ignore[method-assign]
        adapter2.naive_offloader.drop_from_gpu = _rec_drop  # type: ignore[method-assign]
        out = adapter2.generate_dit("n0", {"height": 512, "width": 512, "dit_steps": 2, "prompt": "draw bridge"})
        assert out["status"] == "ok"
        assert "drop" in order
        assert order.index("hwm") < order.index("drop")
    finally:
        adapter2.close()


# ==================== 4. InternVLUCPTGAdapter Public Seams ====================


def test_internvl_u_adapter_edit_preencode_no_monkeypatch_and_hwm(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    img_path = tmp_path / "edit_src.png"
    Image.new("RGB", (32, 32), color=(90, 150, 60)).save(img_path)

    pipe = _build_internvlu_stub(torch)
    orig_pixels_to_latents = pipe.image_pipeline.pixels_to_latents
    pol = EnginePolicyConfig.from_preset(
        "morphserve",
        num_gpus=1,
        model_family="internvl_u",
        use_tripool_vmm=False,
        use_expert_pager=True,
        preemption_mode="non_preemptive",
    )
    adapter = InternVLUCPTGAdapter(
        rank=0,
        policy=pol,
        bench_root=tmp_path,
        pipeline=pipe,
        device=torch.device("cpu"),
    )
    try:
        assert adapter.pager is not None
        assert adapter.pager.spec.vae_encode_restore == "none"
        assert adapter.available_kv_bytes() >= 0

        # Pre-encode edit turn into bridge tensors outside generation_mode (lazy pack: no eager flat concat).
        b_bytes = adapter.prepare_turn(
            "iv0",
            {
                "prompt": "make sky dramatic",
                "image_paths": [str(img_path)],
                "height": 512,
                "width": 512,
                "prepare_for_dit": True,
            },
        )
        assert b_bytes > 0
        assert "packed_flat" not in adapter._ctx_store["iv0"]
        assert pipe.prepare_diff_calls == 1

        # Pack & unpack bridge state roundtrip
        packed = adapter.pack_state("iv0")
        assert packed.logical_bytes == b_bytes
        unpacked_b = adapter.unpack_state("iv0_rx", packed.meta, packed.payload)
        assert unpacked_b == b_bytes

        # Run generate_dit and confirm `pixels_to_latents` was NEVER monkey-patched!
        out_png = tmp_path / "ivlu_out.png"
        res = adapter.generate_dit(
            "iv0_rx",
            {"height": 512, "width": 512, "dit_steps": 2, "save_tmp_image_path": str(out_png)},
        )
        assert res["status"] == "ok"
        assert out_png.is_file()
        assert pipe.image_pipeline.pixels_to_latents == orig_pixels_to_latents

        # Batched AR decoding strips right-padding and EOS cleanly.
        ar_outs = adapter.decode_ar_batch(
            [
                {"ctx_id": "ar0", "prompt": "question 0", "max_ar_tokens": 5},
                {"ctx_id": "ar1", "prompt": "question 1", "max_ar_tokens": 5},
            ]
        )
        assert len(ar_outs) == 2
        assert all(o["tokens"] == 3 for o in ar_outs)  # [tok0, tok1, EOS], padding stripped

        adapter.reset_peak_memory_stats()
        stats = adapter.memory_and_vmm_stats()
        assert "peak_reserved_mib" in stats
        assert "vmm_physical_pool_mib" in stats
    finally:
        adapter.close()


def test_internvl_u_adapter_token_timing_streamer_and_rolling_history(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    import time

    pipe = _build_internvlu_stub(torch)
    seen_prompts: list[str] = []

    def _stub_generate(input_ids, attention_mask, max_new_tokens=8, do_sample=False, streamer=None, **_kw):
        if streamer is not None:
            # Prompt 2D tensor must be skipped; each subsequent 1D token step records a real timestamp.
            streamer.put(torch.zeros((1, 4), dtype=torch.long))
            for tok_id in (11, 22, 99):
                time.sleep(0.002)
                streamer.put(torch.tensor([tok_id], dtype=torch.long))
            streamer.end()
        return SimpleNamespace(generate_output=torch.tensor([[11, 22, 99, 0]], dtype=torch.long))

    orig_proc = pipe.processor.__call__

    def _spy_proc(prompt, image=None, generation_mode="image", padding=True, return_tensors="pt", height=512, width=512):
        if generation_mode == "text":
            seen_prompts.append(str(prompt))
        return orig_proc(
            prompt=prompt,
            image=image,
            generation_mode=generation_mode,
            padding=padding,
            return_tensors=return_tensors,
            height=height,
            width=width,
        )

    pipe.processor = _spy_proc
    pipe.vlm.generate = lambda **kw: None
    pipe._generate = _stub_generate

    pol = EnginePolicyConfig.from_preset(
        "morphserve",
        num_gpus=1,
        model_family="internvl_u",
        use_tripool_vmm=False,
        use_expert_pager=False,
        preemption_mode="non_preemptive",
    )
    adapter = InternVLUCPTGAdapter(
        rank=0,
        policy=pol,
        bench_root=tmp_path,
        pipeline=pipe,
        device=torch.device("cpu"),
    )
    try:
        kv0 = adapter.prepare_turn("hist0", {"prompt": "What is in the room?", "prepare_for_dit": False})
        out0 = adapter.decode_ar("hist0", max_tokens=4)
        assert out0["tokens"] == 3
        assert out0["ttft_ms"] > 0.0
        assert len(out0["tpots_ms"]) == 2
        assert all(t > 0.0 for t in out0["tpots_ms"])

        # Turn 2 must prepend rolling multi-turn User/Assistant history into the VLM prompt and grow context bytes.
        kv1 = adapter.prepare_turn("hist0", {"prompt": "What color is the chair?", "prepare_for_dit": False})
        assert kv1 > kv0
        _ = adapter.decode_ar("hist0", max_tokens=4)
        assert len(seen_prompts) == 2
        assert "User: What is in the room?" in seen_prompts[1]
        assert "Assistant:" in seen_prompts[1]
        assert "User: What color is the chair?" in seen_prompts[1]

        # Rolling history must survive `pack_state` / `unpack_state` across GPU handoff!
        adapter._ctx_store["hist0"]["diff_inp"] = {"prompt_embeds": torch.ones((1, 2, 4), dtype=torch.float32)}
        adapter._ctx_store["hist0"]["gh"] = 512
        adapter._ctx_store["hist0"]["gw"] = 512
        packed = adapter.pack_state("hist0")
        assert len(packed.meta["history"]) == 2
        adapter.unpack_state("hist0_rx", packed.meta, packed.payload)
        assert len(adapter._ctx_store["hist0_rx"]["history"]) == 2
    finally:
        adapter.close()


# ==================== 5. NaiveHostOffloader & StaticFissionDropper ====================


def test_naive_host_offloader_and_static_fission_dropper_lifecycle() -> None:
    torch = pytest.importorskip("torch")

    pipe, _ = _build_bagel_stub(torch, num_layers=2, hidden=8)
    spec = ExpertPartitionSpec.for_bagel(pipe, role="generation", super_handle_mib=2, wrap_vae=False)
    gen_params = [ps.param_ref for plan in spec.plans for ps in plan.params]
    und_params = [p for n, p in pipe.bagel.language_model.model.layers.named_parameters() if "moe_gen" not in n]
    assert len(gen_params) > 0 and len(und_params) > 0

    originals = [p.detach().clone() for p in gen_params]

    offloader = NaiveHostOffloader(spec, device=torch.device("cpu"))
    try:
        moved = offloader.offload_to_cpu()
        assert moved > 0
        assert offloader.is_resident is False
        assert offloader.hold_depth == 0
        assert all(p.numel() == 0 for p in gen_params)
        assert all(p.numel() > 0 for p in und_params)

        assert offloader.swap_in_sync() is True
        assert offloader.is_resident is True
        assert all(torch.equal(p.detach(), orig) for p, orig in zip(gen_params, originals))
        assert offloader.swap_in_sync() is False  # Already resident

        offloader.drop_from_gpu()
        assert offloader.is_resident is False
        assert all(p.numel() == 0 for p in gen_params)

        # Reference-counted `hold_resident()` + `acquire_resident()` / `release_resident()` prevent double H2D
        with offloader.hold_resident():
            assert offloader.is_resident is True
            assert offloader.hold_depth == 1
            # Non-forced drop must be deferred while held
            offloader.drop_from_gpu()
            assert offloader.is_resident is True
            assert offloader.acquire_resident(swap_in=True) is False
            assert offloader.hold_depth == 2
            offloader.release_resident(drop_when_zero=True)
            assert offloader.hold_depth == 1
            assert offloader.is_resident is True
        assert offloader.hold_depth == 0
        assert offloader.is_resident is False

        # Forced drop overrides active hold depth cleanly
        offloader.acquire_resident(swap_in=True)
        assert offloader.is_resident is True
        offloader.drop_from_gpu(force=True)
        assert offloader.is_resident is False
        offloader.release_resident(drop_when_zero=False)
    finally:
        offloader.close()

    # Restore weights and test StaticFissionDropper
    for p, orig in zip(gen_params, originals):
        p.data = orig.clone()
    dropper = StaticFissionDropper(spec, device=torch.device("cpu"))
    try:
        dropped = dropper.drop()
        assert dropped > 0
        assert all(p.numel() == 0 for p in gen_params)
        assert all(p.numel() > 0 for p in und_params)
    finally:
        dropper.close()


def test_bagel_update_image_ctx_uses_generation_mode_without_scoped_vae(tmp_path: Path) -> None:
    """Regression test for `no_m1_scoped_vae` `err=52`: `forward_cache_update_vae` runs `mode='gen'`
    Triton kernels and MUST enter `pager.generation_mode` even when `use_scoped_vae=False`, while
    skipping re-entrant pacer acquisition when `pacer_already_held=True`."""
    torch = pytest.importorskip("torch")
    from contextlib import contextmanager
    from PIL import Image

    pipe, cache_cls = _build_bagel_stub(torch, num_layers=2, hidden=8)
    pol = EnginePolicyConfig.from_preset(
        "morphserve",
        num_gpus=1,
        use_tripool_vmm=False,
        use_expert_pager=False,
        use_scoped_vae=False,
        preemption_mode="non_preemptive",
    )
    adapter = BagelCPTGAdapter(
        rank=0,
        policy=pol,
        bench_root=tmp_path,
        pipeline=pipe,
        naive_cache_cls=cache_cls,
        device=torch.device("cpu"),
    )
    calls: list[dict[str, Any]] = []

    class _SpyPager:
        @contextmanager
        def generation_mode(self, **kwargs: Any):
            calls.append(dict(kwargs))
            yield

    adapter.pager = _SpyPager()  # type: ignore[assignment]
    adapter.vmm = object()  # type: ignore[assignment]
    adapter.pacer = object()
    ctx = {"past_key_values": cache_cls(2), "kv_lens": [0], "ropes": [0]}
    img = Image.new("RGB", (16, 16), color=(10, 20, 30))

    try:
        adapter._update_image_ctx(ctx, img, update_vae=True, update_vit=False, pacer_already_held=False)
        adapter._update_image_ctx(ctx, img, update_vae=True, update_vit=False, pacer_already_held=True)
        assert len(calls) == 2
        assert calls[0]["prefetch_all"] is True
        assert calls[0]["pacer"] is adapter.pacer
        assert calls[1]["prefetch_all"] is True
        assert calls[1]["pacer"] is None
    finally:
        adapter.pager = None
        adapter.vmm = None
        adapter.close()


def test_internvlu_bridge_pack_inline_metadata_and_16b_alignment() -> None:
    """Small int64/bool grid metadata tensors serialize inline in JSON `meta` and bulk tensors stay 16B-aligned."""
    torch = pytest.importorskip("torch")
    from morphserve.models.internvl_u_adapter import pack_internvlu_bridge_state, unpack_internvlu_bridge_state

    diff_inp = {
        "encoder_hidden_states": [torch.randn((5, 8), dtype=torch.bfloat16)],
        "encoder_image_token_mask": [torch.tensor([True, False, True] * 25, dtype=torch.bool)],  # 75 bytes (odd!)
        "image_fhw_cond": [torch.zeros((0, 3), dtype=torch.int64)],
        "image_grid_thw_gen": torch.tensor([[1, 32, 32]], dtype=torch.int64),
        "conditional_image": [torch.randn((1, 3, 4, 4), dtype=torch.float32)],
        "image_hidden_states": None,
    }
    meta, flat = pack_internvlu_bridge_state(diff_inp, device=torch.device("cpu"))
    assert meta["tree"]["image_grid_thw_gen"]["kind"] == "inline_tensor"
    assert meta["tree"]["image_fhw_cond"]["items"][0]["kind"] == "inline_tensor"
    # Bulk tensor following odd-length 75-byte bool mask MUST be 16-byte aligned.
    cond_node = meta["tree"]["conditional_image"]["items"][0]
    assert cond_node["kind"] == "tensor"
    assert int(cond_node["offset"]) % 16 == 0

    restored = unpack_internvlu_bridge_state(meta, flat)
    assert torch.equal(restored["image_grid_thw_gen"], diff_inp["image_grid_thw_gen"])
    assert tuple(restored["image_fhw_cond"][0].shape) == (0, 3)
    assert restored["image_fhw_cond"][0].dtype == torch.int64
    assert torch.equal(restored["encoder_hidden_states"][0], diff_inp["encoder_hidden_states"][0])
    assert torch.equal(restored["encoder_image_token_mask"][0], diff_inp["encoder_image_token_mask"][0])
    assert torch.equal(restored["conditional_image"][0], diff_inp["conditional_image"][0])
    assert restored["image_hidden_states"] is None

    with pytest.raises(ValueError, match="truncated"):
        unpack_internvlu_bridge_state(meta, flat[:-1])


def test_internvlu_step_preemption_regime_a_vs_regime_b_guard(tmp_path: Path) -> None:
    """Regime A services arriving AR turns at DiT step boundaries with OOM isolation; Regime B disables mid-DiT preemption."""
    torch = pytest.importorskip("torch")
    from PIL import Image

    class _FakeScheduler:
        def __init__(self) -> None:
            self.steps_called = 0

        def step(self, noise_pred: Any, t: Any, latents: Any, return_dict: bool = False) -> tuple[Any]:
            self.steps_called += 1
            return (latents,)

    class _StepPipe:
        def __init__(self) -> None:
            self.scheduler = _FakeScheduler()
            self.orig_step = self.scheduler.step
            self.step_wrapped_seen: list[bool] = []

        def __call__(
            self,
            *,
            num_inference_steps: int = 4,
            callback_on_step_end: Any = None,
            **kwargs: Any,
        ) -> Any:
            # Must NOT use upstream's buggy `callback_on_step_end` hook.
            assert callback_on_step_end is None
            self.step_wrapped_seen.append(self.scheduler.step != self.orig_step)
            latents = torch.zeros((1, 4, 8, 8))
            for step_i in range(num_inference_steps):
                latents = self.scheduler.step(latents, step_i, latents, return_dict=False)[0]
            return SimpleNamespace(images=[Image.new("RGB", (32, 32), color=(40, 80, 120))])

    def _make_stub_pipe():
        step_pipe = _StepPipe()
        proc = SimpleNamespace(image_gen_processor=SimpleNamespace(merge_size=2, patch_size=8))
        tok = SimpleNamespace(
            pad_token_id=0,
            eos_token_id=2,
            unk_token_id=1,
            convert_tokens_to_ids=lambda _s: -1,
            decode=lambda seq, skip_special_tokens=True: "step preempt answer",
        )
        pipe = SimpleNamespace(
            device=torch.device("cpu"),
            processor=proc,
            tokenizer=tok,
            vlm=None,
            generation_decoder=None,
            image_pipeline=step_pipe,
        )
        return pipe, step_pipe

    # 1. Regime A (unconstrained): callback_on_step_end fires, serves valid AR & isolates error without killing DiT!
    pipe_a, step_pipe_a = _make_stub_pipe()
    pol_a = EnginePolicyConfig.from_preset(
        "morphserve",
        num_gpus=1,
        model_family="internvl_u",
        use_tripool_vmm=False,
        use_expert_pager=False,
        preemption_mode="layer_chunk",
    )
    ad_a = InternVLUCPTGAdapter(rank=0, policy=pol_a, bench_root=tmp_path, pipeline=pipe_a, device=torch.device("cpu"))
    try:
        ad_a._ctx_store["dit_a"] = {
            "prompt": "draw a cat",
            "diff_inp": {},
            "gh": 32,
            "gw": 32,
            "bridge_bytes": 64,
        }
        ad_a._ctx_store["ar_ok"] = {"prompt": "describe", "history": [], "images": [], "bridge_bytes": 64}
        ad_a._ctx_store["ar_bad"] = {"prompt": "bad", "history": [], "images": [], "bridge_bytes": 64}

        def _fake_decode_ar_batch(batch_cmds: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if len(batch_cmds) > 1 or any(c.get("ctx_id") == "ar_bad" for c in batch_cmds):
                if len(batch_cmds) > 1:
                    raise RuntimeError("injected batch error")
                raise MemoryError("out of memory on ar_bad")
            return [
                {
                    "text": "ok reply",
                    "tokens": 2,
                    "ttft_ms": 12.0,
                    "tpots_ms": [8.0],
                    "kv_bytes": 64,
                }
            ]

        ad_a.decode_ar_batch = _fake_decode_ar_batch  # type: ignore[method-assign]
        queued_batches = [
            [
                {"cmd_id": "c_ok", "ctx_id": "ar_ok", "prompt": "describe", "max_ar_tokens": 4},
                {"cmd_id": "c_bad", "ctx_id": "ar_bad", "prompt": "bad", "max_ar_tokens": 4},
            ],
            [],
            [],
            [],
        ]
        acks: list[dict[str, Any]] = []
        res_a = ad_a.generate_dit(
            "dit_a",
            {"height": 32, "width": 32, "dit_steps": 4, "seed": 7},
            poll_preempt_cmds=lambda: queued_batches.pop(0) if queued_batches else [],
            emit_preempt_ack=acks.append,
        )
        assert res_a["status"] == "ok"
        assert step_pipe_a.step_wrapped_seen == [True]
        assert step_pipe_a.scheduler.step == step_pipe_a.orig_step
        assert step_pipe_a.scheduler.steps_called == 4
        assert res_a["yield_count"] == 1
        by_id = {a["cmd_id"]: a for a in acks}
        assert by_id["c_ok"]["status"] == "ok" and by_id["c_ok"]["preempted_mid_dit"] is True
        assert by_id["c_bad"]["status"] == "oom"
    finally:
        ad_a.close()

    # 2. Regime B (quota-constrained): scheduler.step MUST NOT be wrapped (prevents VLM<->Decoder thrash).
    pipe_b, step_pipe_b = _make_stub_pipe()
    pol_b = EnginePolicyConfig.from_preset(
        "morphserve",
        num_gpus=1,
        model_family="internvl_u",
        use_tripool_vmm=False,
        use_expert_pager=False,
        preemption_mode="layer_chunk",
        gpu_memory_quota_mib=12288.0,
    )
    ad_b = InternVLUCPTGAdapter(rank=0, policy=pol_b, bench_root=tmp_path, pipeline=pipe_b, device=torch.device("cpu"))
    try:
        ad_b._ctx_store["dit_b"] = {
            "prompt": "draw a dog",
            "diff_inp": {},
            "gh": 32,
            "gw": 32,
            "bridge_bytes": 64,
        }
        acks_b: list[dict[str, Any]] = []
        res_b = ad_b.generate_dit(
            "dit_b",
            {"height": 32, "width": 32, "dit_steps": 4, "seed": 7},
            poll_preempt_cmds=lambda: [{"cmd_id": "c_never", "ctx_id": "x"}],
            emit_preempt_ack=acks_b.append,
        )
        assert res_b["status"] == "ok"
        assert step_pipe_b.step_wrapped_seen == [False]
        assert step_pipe_b.scheduler.step == step_pipe_b.orig_step
        assert res_b["yield_count"] == 0
        assert acks_b == []
    finally:
        ad_b.close()


def test_bagel_static_fission_gen_rank_drops_unused_ar_vit_and_uniform_oom_retry(tmp_path: Path) -> None:
    """Verify static-fission `gen` ranks drop unused `vit_model`/`connector`/untied `lm_head` while preserving
    `embed_tokens` + understanding layers, and verify uniform worker OOM retry gates `evict_idle_kv_to_host`."""
    torch = pytest.importorskip("torch")

    pipe, cache_cls = _build_bagel_stub(torch, num_layers=2, hidden=8)
    pipe.bagel.vit_model = torch.nn.Linear(8, 8, bias=False)
    pipe.bagel.connector = torch.nn.Linear(8, 8, bias=False)
    pipe.bagel.language_model.model.embed_tokens = torch.nn.Embedding(32, 8)
    assert pipe.bagel.vit_model.weight.numel() == 64
    assert pipe.bagel.connector.weight.numel() == 64
    assert pipe.bagel.language_model.lm_head.weight.numel() == 256

    pol_fis = EnginePolicyConfig.from_preset("static_nccl_fission", num_gpus=8)
    gen_rank = pol_fis.static_gen_ranks[0]
    ad_gen = BagelCPTGAdapter(
        rank=gen_rank,
        policy=pol_fis,
        bench_root=tmp_path,
        pipeline=pipe,
        naive_cache_cls=cache_cls,
        device=torch.device("cpu"),
    )
    try:
        assert ad_gen.role == "gen"
        assert pipe.bagel.vit_model.weight.numel() == 0
        assert pipe.bagel.connector.weight.numel() == 0
        assert pipe.bagel.language_model.lm_head.weight.numel() == 0
        # `embed_tokens` and understanding MoT layers MUST remain intact for tight-MoT `mode="gen"`.
        assert pipe.bagel.language_model.model.embed_tokens.weight.numel() == 256
        assert pipe.bagel.language_model.model.layers[0].mlp_und.weight.numel() == 64
    finally:
        ad_gen.close()

    # Verify colocated baselines (`monolithic_dp`, `monolithic_kv_swap`, `prism_tetriserve_coloc`) NEVER drop
    # `vit_model`/`connector`/`lm_head` even though `policy.role_of(0) == "gen"` when `use_expert_pager=False`,
    # and verify uniform `max_dit_oom_retries = 2` on `monolithic_dp` never calls `evict_idle_kv_to_host`.
    for coloc_preset in ("monolithic_dp", "monolithic_kv_swap", "prism_tetriserve_coloc"):
        pipe_c, cache_cls_c = _build_bagel_stub(torch, num_layers=2, hidden=8)
        pipe_c.bagel.vit_model = torch.nn.Linear(8, 8, bias=False)
        pipe_c.bagel.connector = torch.nn.Linear(8, 8, bias=False)
        pipe_c.bagel.language_model.model.embed_tokens = torch.nn.Embedding(32, 8)
        pol_c = EnginePolicyConfig.from_preset(coloc_preset, num_gpus=1, use_tripool_vmm=False)
        ad_c = BagelCPTGAdapter(
            rank=0,
            policy=pol_c,
            bench_root=tmp_path,
            pipeline=pipe_c,
            naive_cache_cls=cache_cls_c,
            device=torch.device("cpu"),
        )
        try:
            assert pipe_c.bagel.vit_model.weight.numel() == 64
            assert pipe_c.bagel.connector.weight.numel() == 64
            assert pipe_c.bagel.language_model.lm_head.weight.numel() == 256
            if coloc_preset == "monolithic_dp":
                evict_calls: list[int] = []
                ad_c.evict_idle_kv_to_host = lambda *a, **kw: (evict_calls.append(1) or 0)  # type: ignore[method-assign]
                pipe_c.bagel.generate_image = lambda **kw: (_ for _ in ()).throw(MemoryError("injected dit oom"))
                ad_c.prepare_turn("mono_s0", {"prompt": "draw a tree", "prepare_for_dit": True, "height": 512, "width": 512})
                with pytest.raises(MemoryError, match="injected dit oom"):
                    ad_c.generate_dit("mono_s0", {"height": 512, "width": 512, "dit_steps": 2, "prompt": "draw a tree"})
                assert evict_calls == []
        finally:
            ad_c.close()
