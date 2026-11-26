"""Policy configurations and presets for `MorphServeClusterEngine` & 5-baseline comparison matrix."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from morphserve.router.topology import DEFAULT_OPERATOR_PROFILE

VALID_POLICIES = (
    "morphserve",
    "monolithic_dp",
    "twostage_shm",
    "static_nccl_fission",
    "prism_tetriserve_coloc",
    "naive_host_offload",
)


@dataclass
class EnginePolicyConfig:
    """Feature-toggle matrix implementing MorphServe and all 5 comparison baselines on one unified engine."""

    policy_name: str
    num_gpus: int = 8
    model_family: str = "bagel"  # "bagel" | "internvl_u" | "synthetic"
    cptg_arch_override: str | None = None
    vendor_dir: str | None = None
    routing_mode: str = "dual_path"  # "dual_path" | "sticky_least_loaded" | "twostage_shm" | "static_nccl_fission"
    use_tripool_vmm: bool = True
    use_expert_pager: bool = True
    use_scoped_vae: bool = True
    use_shared_numa_mirror: bool = True
    use_h2d_pacer: bool = True
    use_module3_cow_loopback: bool = True
    allow_cascade_cfg_split: bool = False
    use_naive_offload: bool = False
    kv_vmm_only: bool = False
    preemption_mode: str = "layer_chunk"  # "non_preemptive" | "step_preemptive" | "layer_chunk"
    static_und_ranks: tuple[int, ...] = ()
    static_gen_ranks: tuple[int, ...] = ()
    initial_alphas: dict[int, float] = field(default_factory=dict)

    # Per-class decoupled k_AR / k_DiT normalized unloaded SLOs (`D1-D5`) & bounded OOM recovery
    slo_scale_k: float = 2.0
    dit_slo_scale_k: float = 3.0
    ar_tpot_slo_ms: float = 125.64
    ar_ttft_slo_ms: float = 1500.0
    dit_512_slo_ms: float = 18360.0
    dit_1024_slo_ms: float = 65160.0
    dit_turn_slo_override_ms: float | None = None
    max_oom_retries: int = 1
    inter_turn_think_s_range: tuple[float, float] = (2.0, 25.0)
    think_time_dist: str = "lognormal"
    think_time_median_s: float = 6.0
    think_time_sigma: float = 0.75
    ar_tpot_slo_mode: str = "turn_p95"
    steady_window_mode: str = "time"
    warmup_window_s: float = 60.0
    steady_window_s: float = 180.0
    steady_state_trim_frac: float = 0.15
    gpu_memory_fraction: float | None = None
    gpu_memory_quota_mib: float | None = None

    # TriPool-VMM physical & virtual sizing
    super_pool_mib: int = 14336
    fine_kv_pool_mib: int = 256
    super_handle_mib: int = 64
    transient_va_gib: int = 16
    kv_va_gib: int = 16
    allow_vmm_overflow: bool = True

    @property
    def cptg_architecture(self) -> str:
        if self.cptg_arch_override is not None:
            return self.cptg_arch_override
        return "bridge_mllm_dit" if self.model_family == "internvl_u" else "tight_mot"

    def dit_turn_slo_for(self, height: int, steps: int = 20, k: float | None = None) -> float:
        if self.dit_turn_slo_override_ms is not None and k is None:
            return float(self.dit_turn_slo_override_ms)
        eff_k = self.dit_slo_scale_k if k is None else float(k)
        return DEFAULT_OPERATOR_PROFILE.dit_turn_slo_ms(
            height=height,
            k=eff_k,
            model_family="internvl_u" if self.model_family == "internvl_u" else "bagel",
            steps=steps,
        )

    def role_of(self, rank: int) -> str:
        if self.static_und_ranks and rank in self.static_und_ranks:
            return "und"
        if self.static_gen_ranks and rank in self.static_gen_ranks:
            return "gen"
        alpha = self.initial_alphas.get(rank, 1.0)
        if alpha >= 0.99:
            return "gen" if not self.use_expert_pager else "hybrid"
        if alpha <= 0.01:
            return "und"
        return "hybrid"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnginePolicyConfig":
        d = dict(data)
        d["initial_alphas"] = {int(k): float(v) for k, v in (d.get("initial_alphas") or {}).items()}
        d["static_und_ranks"] = tuple(int(x) for x in (d.get("static_und_ranks") or ()))
        d["static_gen_ranks"] = tuple(int(x) for x in (d.get("static_gen_ranks") or ()))
        tr = d.get("inter_turn_think_s_range")
        if tr is None:
            d["inter_turn_think_s_range"] = (2.0, 25.0)
        else:
            d["inter_turn_think_s_range"] = (float(tr[0]), float(tr[1]))
        if "allow_cascade_cfg_split" in d:
            if d["allow_cascade_cfg_split"] is None:
                d.pop("allow_cascade_cfg_split")
            else:
                d["allow_cascade_cfg_split"] = bool(d["allow_cascade_cfg_split"])
        for str_key in ("think_time_dist", "ar_tpot_slo_mode", "steady_window_mode"):
            if str_key in d:
                if d[str_key] is None:
                    d.pop(str_key)
                else:
                    d[str_key] = str(d[str_key])
        for flt_key in ("dit_slo_scale_k", "think_time_median_s", "think_time_sigma", "warmup_window_s", "steady_window_s"):
            if flt_key in d:
                if d[flt_key] is None:
                    d.pop(flt_key)
                else:
                    d[flt_key] = float(d[flt_key])
        return cls(**d)

    @classmethod
    def from_preset(
        cls,
        policy_name: str,
        num_gpus: int = 8,
        **overrides: Any,
    ) -> "EnginePolicyConfig":
        from morphserve._core import call_json, encode_json, get_core_lib

        L = get_core_lib()
        ser_overrides: dict[str, Any] = {}
        for k, v in overrides.items():
            if k == "initial_alphas" and isinstance(v, dict):
                ser_overrides[k] = {str(rk): float(rv) for rk, rv in v.items()}
            elif isinstance(v, tuple):
                ser_overrides[k] = list(v)
            else:
                ser_overrides[k] = v
        out = call_json(
            L,
            L.ms_policy_from_preset_json,
            str(policy_name).encode("utf-8"),
            int(num_gpus),
            encode_json(ser_overrides),
        )
        return cls.from_dict(out)
