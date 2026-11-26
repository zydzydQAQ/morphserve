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
    finally:
        adapter.close()


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

        # Pre-encode edit turn into bridge tensors outside generation_mode.
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
        assert all(p.numel() == 0 for p in gen_params)
        assert all(p.numel() > 0 for p in und_params)

        assert offloader.swap_in_sync() is True
        assert all(torch.equal(p.detach(), orig) for p, orig in zip(gen_params, originals))
        assert offloader.swap_in_sync() is False  # Already resident

        offloader.drop_from_gpu()
        assert all(p.numel() == 0 for p in gen_params)
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
