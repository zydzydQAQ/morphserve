"""Single permanent, mode-aware execution coordinator for `BagelPipeline`."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import time
from typing import Any, Callable

from morphserve.kernels import BagelSharedPrefixPatch, install_shared_prefix_bagel_hooks


@dataclass
class _FallbackNavitOutput:
    packed_query_sequence: Any
    past_key_values: Any


def _make_navit_output(seq: Any, pkv: Any) -> Any:
    try:
        from vllm_omni.diffusion.models.bagel.bagel_transformer import BaseNavitOutputWithPast

        return BaseNavitOutputWithPast(packed_query_sequence=seq, past_key_values=pkv)
    except ImportError:
        return _FallbackNavitOutput(packed_query_sequence=seq, past_key_values=pkv)


def _sync_cuda_if_present(device: Any = None) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            if device is not None:
                torch.cuda.synchronize(device)
            else:
                torch.cuda.synchronize()
    except ImportError:
        pass


class BagelExecutionInterceptor:
    """Permanent per-pipeline execution coordinator owning generation-only (`mode == 'gen'`) hooks.

    Consolidates:
      1. `LayerExpertPager` block-level H2D wait gating via `pager.set_wait_gate(...)` (zero private pokes).
      2. Step-0 `HalfNodeH2DPacer` slot release immediately after Step-0 completes.
      3. `UnifiedPreemptiveScheduler` EDF block-chunk & step-boundary AR yields.
      4. `BagelSharedPrefixPatch` `ContextVar`-scoped `SharedPrefixCache` + `FlashPrefix-CFG` attention.
    """

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.bagel = pipeline.bagel
        self.lm_model = pipeline.bagel.language_model.model
        self.device = getattr(pipeline, "device", None)

        self.pager: Any = None
        self.cfg_patch: BagelSharedPrefixPatch | None = None
        self._active_scheduler: Any = None
        self._step0_release_cb: Callable[[], None] | None = None
        self._step0_done: bool = False
        self._gen_depth: int = 0
        self._installed: bool = False
        self._orig_lm_forward: Any = None
        self._orig_bagel_forward: Any = None

    @classmethod
    def for_pipeline(cls, pipeline: Any) -> "BagelExecutionInterceptor":
        existing = getattr(pipeline, "_morphserve_interceptor", None)
        if isinstance(existing, cls):
            existing.install()
            return existing
        inst = cls(pipeline)
        inst.install()
        setattr(pipeline, "_morphserve_interceptor", inst)
        return inst

    @property
    def is_gen_compute_active(self) -> bool:
        sched = self._active_scheduler
        in_ar_yield = bool(sched is not None and getattr(sched, "in_ar_yield", False))
        return (self._gen_depth > 0) and (not in_ar_yield)

    def bind_pager(self, pager: Any | None) -> None:
        """Bind `LayerExpertPager` so its layer waits are gated automatically on `mode == 'gen'`."""
        self.pager = pager
        if pager is not None and hasattr(pager, "set_wait_gate"):
            pager.set_wait_gate(lambda: self.is_gen_compute_active)

    def enable_shared_prefix_cfg(self, min_shared_tokens: int = 32) -> BagelSharedPrefixPatch:
        """Install and bind `BagelSharedPrefixPatch` under this execution coordinator."""
        if self.cfg_patch is None:
            self.cfg_patch = install_shared_prefix_bagel_hooks(self.pipeline, min_shared_tokens=min_shared_tokens)
        return self.cfg_patch

    def install(self) -> None:
        if self._installed:
            return
        interceptor = self
        self._orig_lm_forward = self.lm_model.forward
        self._orig_bagel_forward = self.bagel.forward

        def _intercepted_lm_forward(
            packed_query_sequence=None,
            query_lens=None,
            packed_query_position_ids=None,
            past_key_values=None,
            update_past_key_values=True,
            is_causal=True,
            mode="und",
            packed_vae_token_indexes=None,
            packed_text_indexes=None,
            packed_text_ids=None,
            return_embeddings_only=False,
        ):
            sched = interceptor._active_scheduler
            in_yield = bool(sched is not None and getattr(sched, "in_ar_yield", False))
            if mode != "gen" or return_embeddings_only or in_yield:
                return interceptor._orig_lm_forward(
                    packed_query_sequence=packed_query_sequence,
                    query_lens=query_lens,
                    packed_query_position_ids=packed_query_position_ids,
                    past_key_values=past_key_values,
                    update_past_key_values=update_past_key_values,
                    is_causal=is_causal,
                    mode=mode,
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_text_indexes=packed_text_indexes,
                    packed_text_ids=packed_text_ids,
                    return_embeddings_only=return_embeddings_only,
                )

            interceptor._gen_depth += 1
            try:
                if sched is None or sched.mode != "layer_chunk":
                    return interceptor._orig_lm_forward(
                        packed_query_sequence=packed_query_sequence,
                        query_lens=query_lens,
                        packed_query_position_ids=packed_query_position_ids,
                        past_key_values=past_key_values,
                        update_past_key_values=update_past_key_values,
                        is_causal=is_causal,
                        mode=mode,
                        packed_vae_token_indexes=packed_vae_token_indexes,
                        packed_text_indexes=packed_text_indexes,
                        packed_text_ids=packed_text_ids,
                        return_embeddings_only=return_embeddings_only,
                    )

                lm = interceptor.lm_model
                if packed_query_sequence is None:
                    packed_query_sequence = lm.embed_tokens(packed_text_ids)
                if hasattr(lm, "rotary_emb") and packed_query_position_ids is not None:
                    cos, sin = lm.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
                    packed_query_position_embeddings = (cos.squeeze(0), sin.squeeze(0))
                else:
                    packed_query_position_embeddings = None

                extra_inputs: dict[str, Any] = {}
                if getattr(lm, "use_moe", False):
                    extra_inputs = {
                        "mode": "gen",
                        "packed_vae_token_indexes": packed_vae_token_indexes,
                        "packed_text_indexes": packed_text_indexes,
                    }

                num_layers = len(lm.layers)
                for layer_idx, decoder_layer in enumerate(lm.layers):
                    if interceptor.pager is not None and not interceptor.pager.spec.install_module_hooks:
                        interceptor.pager.wait_layer_if_needed(layer_idx, "interceptor_gen")
                    t_l0 = time.perf_counter()
                    packed_query_sequence, past_key_values = decoder_layer(
                        hidden_states=packed_query_sequence,
                        encoder_hidden_states=None,
                        query_lens=query_lens,
                        packed_query_position_embeddings=packed_query_position_embeddings,
                        past_key_values=past_key_values,
                        update_past_key_values=update_past_key_values,
                        is_causal=is_causal,
                        **extra_inputs,
                    )
                    sched.on_gen_layer_complete(layer_idx=layer_idx, num_layers=num_layers, t_layer_start_s=t_l0)

                if hasattr(lm, "norm"):
                    packed_query_sequence = lm.norm(
                        packed_query_sequence, packed_text_indexes, packed_vae_token_indexes
                    )
                return _make_navit_output(packed_query_sequence, past_key_values)
            finally:
                interceptor._gen_depth -= 1

        def _intercepted_bagel_forward(*args: Any, **kwargs: Any) -> Any:
            sched = interceptor._active_scheduler
            t_s0 = time.perf_counter()
            ar_before_ms = float(sched.ar_compute_total_ms) if sched is not None else 0.0

            interceptor._gen_depth += 1
            try:
                out = interceptor._orig_bagel_forward(*args, **kwargs)
            finally:
                interceptor._gen_depth -= 1

            if interceptor._step0_release_cb is not None and not interceptor._step0_done:
                _sync_cuda_if_present(interceptor.device)
                cb = interceptor._step0_release_cb
                interceptor._step0_done = True
                interceptor._step0_release_cb = None
                cb()

            if sched is not None:
                sched.on_dit_step_complete(step_start_s=t_s0, ar_compute_before_ms=ar_before_ms)
            return out

        self.lm_model.forward = _intercepted_lm_forward
        self.bagel.forward = _intercepted_bagel_forward
        self._installed = True

    def uninstall(self) -> None:
        if self.cfg_patch is not None:
            try:
                self.cfg_patch.uninstall()
            except Exception:
                pass
            self.cfg_patch = None
        if not self._installed:
            return
        if self._orig_lm_forward is not None:
            self.lm_model.forward = self._orig_lm_forward
            self._orig_lm_forward = None
        if self._orig_bagel_forward is not None:
            self.bagel.forward = self._orig_bagel_forward
            self._orig_bagel_forward = None
        self._installed = False

    @contextmanager
    def generation_session(
        self,
        *,
        pager: Any | None = None,
        cfg_patch: Any | None = None,
        use_cfg_patch: bool = True,
        scheduler: Any | None = None,
        pacer_slot_cm: Any | None = None,
        prefetch_all: bool = True,
        reset_cfg_stats: bool = True,
    ):
        """Coordinate Pager + Pacer + CFG Patch + EDF Scheduler over DiT denoising steps."""
        self.install()
        eff_pager = pager if pager is not None else self.pager
        if eff_pager is not None:
            self.bind_pager(eff_pager)
        eff_cfg = cfg_patch if cfg_patch is not None else (self.cfg_patch if use_cfg_patch else None)

        prev_sched = self._active_scheduler
        prev_cb = self._step0_release_cb
        prev_done = self._step0_done
        pacer_active = {"held": False}

        def _release_pacer_once() -> None:
            if pacer_slot_cm is not None and pacer_active["held"]:
                pacer_active["held"] = False
                pacer_slot_cm.__exit__(None, None, None)

        with ExitStack() as stack:
            if pacer_slot_cm is not None:
                pacer_slot_cm.__enter__()
                pacer_active["held"] = True
                self._step0_release_cb = _release_pacer_once
                self._step0_done = False
            if eff_pager is not None:
                stack.enter_context(eff_pager.generation_mode(prefetch_all=prefetch_all))
            if eff_cfg is not None:
                stack.enter_context(eff_cfg.active(reset_stats=reset_cfg_stats))
            if scheduler is not None:
                scheduler.begin_session_clock()
                self._active_scheduler = scheduler
            try:
                yield self
            finally:
                self._active_scheduler = prev_sched
                self._step0_release_cb = prev_cb
                self._step0_done = prev_done
                _release_pacer_once()
