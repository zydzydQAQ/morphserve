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
    use_naive_offload: bool = False
    kv_vmm_only: bool = False
    preemption_mode: str = "layer_chunk"  # "non_preemptive" | "step_preemptive" | "layer_chunk"
    static_und_ranks: tuple[int, ...] = ()
    static_gen_ranks: tuple[int, ...] = ()
    initial_alphas: dict[int, float] = field(default_factory=dict)

    # Per-class k× normalized unloaded SLOs (`D3`) & bounded OOM recovery
    slo_scale_k: float = 2.0
    ar_tpot_slo_ms: float = 125.64
    ar_ttft_slo_ms: float = 1500.0
    dit_512_slo_ms: float = 12240.0
    dit_1024_slo_ms: float = 43440.0
    dit_turn_slo_override_ms: float | None = None
    max_oom_retries: int = 1
    inter_turn_think_s_range: tuple[float, float] = (3.0, 8.0)
    steady_state_trim_frac: float = 0.15
    gpu_memory_fraction: float | None = None

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
        eff_k = self.slo_scale_k if k is None else float(k)
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
    def from_preset(
        cls,
        policy_name: str,
        num_gpus: int = 8,
        **overrides: Any,
    ) -> "EnginePolicyConfig":
        if policy_name not in VALID_POLICIES:
            raise ValueError(f"unsupported policy_name={policy_name!r}; expected one of {VALID_POLICIES}")
        n = int(num_gpus)
        if n < 1:
            raise ValueError("num_gpus must be >= 1")
        fam = str(overrides.get("model_family", "bagel"))
        k_val = float(overrides.get("slo_scale_k", 2.0))
        slo_map = DEFAULT_OPERATOR_PROFILE.slo_thresholds_ms(
            k=k_val,
            model_family="internvl_u" if fam == "internvl_u" else "bagel",
        )
        think_default = (0.002, 0.005) if fam == "synthetic" else (3.0, 8.0)

        if policy_name == "morphserve":
            # Warm alpha=1.0 startup across all workers (`D1`); tail expert layers drop strictly on demand!
            alphas = {g: 1.0 for g in range(n)}
            cfg = cls(
                policy_name="morphserve",
                num_gpus=n,
                model_family=fam,
                routing_mode="dual_path" if n >= 2 else "sticky_least_loaded",
                use_tripool_vmm=True,
                use_expert_pager=True,
                use_scoped_vae=True,
                use_shared_numa_mirror=(n >= 2),
                use_h2d_pacer=(n >= 2),
                use_module3_cow_loopback=(fam != "internvl_u"),
                use_naive_offload=False,
                kv_vmm_only=False,
                preemption_mode="non_preemptive" if fam == "internvl_u" else "layer_chunk",
                initial_alphas=alphas,
                slo_scale_k=k_val,
                ar_tpot_slo_ms=slo_map["ar_tpot_slo_ms"],
                ar_ttft_slo_ms=slo_map["ar_ttft_slo_ms"],
                dit_512_slo_ms=slo_map["dit_512_slo_ms"],
                dit_1024_slo_ms=slo_map["dit_1024_slo_ms"],
                inter_turn_think_s_range=think_default,
                super_pool_mib=4864 if fam == "internvl_u" else 14336,
                fine_kv_pool_mib=512 if fam == "internvl_u" else 256,
                transient_va_gib=16,
                kv_va_gib=4 if fam == "internvl_u" else 16,
            )
        elif policy_name == "monolithic_dp":
            cfg = cls(
                policy_name="monolithic_dp",
                num_gpus=n,
                model_family=fam,
                routing_mode="sticky_least_loaded",
                use_tripool_vmm=False,
                use_expert_pager=False,
                use_scoped_vae=False,
                use_shared_numa_mirror=False,
                use_h2d_pacer=False,
                use_module3_cow_loopback=False,
                use_naive_offload=False,
                kv_vmm_only=False,
                preemption_mode="non_preemptive",
                initial_alphas={g: 1.0 for g in range(n)},
                slo_scale_k=k_val,
                ar_tpot_slo_ms=slo_map["ar_tpot_slo_ms"],
                ar_ttft_slo_ms=slo_map["ar_ttft_slo_ms"],
                dit_512_slo_ms=slo_map["dit_512_slo_ms"],
                dit_1024_slo_ms=slo_map["dit_1024_slo_ms"],
                inter_turn_think_s_range=think_default,
            )
        elif policy_name == "twostage_shm":
            if n < 2 or n % 2 != 0:
                raise ValueError("twostage_shm requires an even number of GPUs >= 2")
            und = tuple(range(0, n, 2))
            gen = tuple(range(1, n, 2))
            cfg = cls(
                policy_name="twostage_shm",
                num_gpus=n,
                model_family=fam,
                routing_mode="twostage_shm",
                use_tripool_vmm=False,
                use_expert_pager=False,
                use_scoped_vae=False,
                use_shared_numa_mirror=False,
                use_h2d_pacer=False,
                use_module3_cow_loopback=False,
                use_naive_offload=False,
                kv_vmm_only=False,
                preemption_mode="non_preemptive",
                static_und_ranks=und,
                static_gen_ranks=gen,
                initial_alphas={g: (0.0 if g in und else 1.0) for g in range(n)},
                slo_scale_k=k_val,
                ar_tpot_slo_ms=slo_map["ar_tpot_slo_ms"],
                ar_ttft_slo_ms=slo_map["ar_ttft_slo_ms"],
                dit_512_slo_ms=slo_map["dit_512_slo_ms"],
                dit_1024_slo_ms=slo_map["dit_1024_slo_ms"],
                inter_turn_think_s_range=think_default,
            )
        elif policy_name == "static_nccl_fission":
            if n < 2:
                raise ValueError("static_nccl_fission requires >= 2 GPUs")
            if n >= 8 and n % 8 == 0:
                und = tuple(g for g in range(n) if (g % 8) in (0, 1, 4, 5))
                gen = tuple(g for g in range(n) if g not in und)
            else:
                half = n // 2
                und, gen = tuple(range(half)), tuple(range(half, n))
            cfg = cls(
                policy_name="static_nccl_fission",
                num_gpus=n,
                model_family=fam,
                routing_mode="static_nccl_fission",
                use_tripool_vmm=False,
                use_expert_pager=False,
                use_scoped_vae=False,
                use_shared_numa_mirror=False,
                use_h2d_pacer=False,
                use_module3_cow_loopback=False,
                use_naive_offload=False,
                kv_vmm_only=False,
                preemption_mode="non_preemptive",
                static_und_ranks=und,
                static_gen_ranks=gen,
                initial_alphas={g: (0.0 if g in und else 1.0) for g in range(n)},
                slo_scale_k=k_val,
                ar_tpot_slo_ms=slo_map["ar_tpot_slo_ms"],
                ar_ttft_slo_ms=slo_map["ar_ttft_slo_ms"],
                dit_512_slo_ms=slo_map["dit_512_slo_ms"],
                dit_1024_slo_ms=slo_map["dit_1024_slo_ms"],
                inter_turn_think_s_range=think_default,
            )
        elif policy_name == "prism_tetriserve_coloc":
            # 100% fair colocation baseline (`D2`): uses real `VmmNaiveCache` (`tripool_kv_region_*`)
            # with zero shadow pages; only differs in static un-pageable expert weights + step preemption.
            cfg = cls(
                policy_name="prism_tetriserve_coloc",
                num_gpus=n,
                model_family=fam,
                routing_mode="sticky_least_loaded",
                use_tripool_vmm=True,
                use_expert_pager=False,
                use_scoped_vae=False,
                use_shared_numa_mirror=False,
                use_h2d_pacer=False,
                use_module3_cow_loopback=False,
                use_naive_offload=False,
                kv_vmm_only=True,
                preemption_mode="non_preemptive" if fam == "internvl_u" else "step_preemptive",
                initial_alphas={g: 1.0 for g in range(n)},
                slo_scale_k=k_val,
                ar_tpot_slo_ms=slo_map["ar_tpot_slo_ms"],
                ar_ttft_slo_ms=slo_map["ar_ttft_slo_ms"],
                dit_512_slo_ms=slo_map["dit_512_slo_ms"],
                dit_1024_slo_ms=slo_map["dit_1024_slo_ms"],
                inter_turn_think_s_range=think_default,
                super_pool_mib=128,
                fine_kv_pool_mib=2048,
                kv_va_gib=16,
                allow_vmm_overflow=True,
            )
        else:  # "naive_host_offload"
            cfg = cls(
                policy_name="naive_host_offload",
                num_gpus=n,
                model_family=fam,
                routing_mode="sticky_least_loaded",
                use_tripool_vmm=False,
                use_expert_pager=False,
                use_scoped_vae=False,
                use_shared_numa_mirror=False,
                use_h2d_pacer=False,
                use_module3_cow_loopback=False,
                use_naive_offload=True,
                kv_vmm_only=False,
                preemption_mode="non_preemptive",
                initial_alphas={g: 0.0 for g in range(n)},
                slo_scale_k=k_val,
                ar_tpot_slo_ms=slo_map["ar_tpot_slo_ms"],
                ar_ttft_slo_ms=slo_map["ar_ttft_slo_ms"],
                dit_512_slo_ms=slo_map["dit_512_slo_ms"],
                dit_1024_slo_ms=slo_map["dit_1024_slo_ms"],
                inter_turn_think_s_range=think_default,
            )

        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ValueError(f"unknown EnginePolicyConfig field: {k}")
            setattr(cfg, k, v)
        return cfg
