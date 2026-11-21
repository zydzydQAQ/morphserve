"""Policy configurations and presets for MorphServeClusterEngine & 5-baseline comparison matrix."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    """Feature-toggle matrix implementing MorphServe and all 5 comparison baselines on one operator stack."""

    policy_name: str
    num_gpus: int = 8
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

    # SLO & VMM pool sizing
    ar_tpot_slo_ms: float = 125.6  # 2.0x unloaded 62.82ms A100 baseline
    ar_ttft_slo_ms: float = 1500.0
    dit_turn_slo_ms: float = 15000.0
    super_pool_mib: int = 14336
    fine_kv_pool_mib: int = 128
    super_handle_mib: int = 64
    transient_va_gib: int = 16
    kv_va_gib: int = 8
    allow_vmm_overflow: bool = True
    adaptive_alpha_rebalance: bool = True

    def role_of(self, rank: int) -> str:
        if self.static_und_ranks and rank in self.static_und_ranks:
            return "und"
        if self.static_gen_ranks and rank in self.static_gen_ranks:
            return "gen"
        alpha = self.initial_alphas.get(rank, 1.0)
        if alpha >= 0.99:
            return "gen"
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

        if policy_name == "morphserve":
            # Soft-biased elastic alphas: even ranks start at 0.50 (hybrid), odd ranks at 1.00 (gen)
            alphas = {g: (1.0 if (g % 2 == 1 or n == 1) else 0.50) for g in range(n)}
            cfg = cls(
                policy_name="morphserve",
                num_gpus=n,
                routing_mode="dual_path" if n >= 2 else "sticky_least_loaded",
                use_tripool_vmm=True,
                use_expert_pager=True,
                use_scoped_vae=True,
                use_shared_numa_mirror=(n >= 2),
                use_h2d_pacer=(n >= 2),
                use_module3_cow_loopback=True,
                use_naive_offload=False,
                kv_vmm_only=False,
                preemption_mode="layer_chunk",
                initial_alphas=alphas,
                super_pool_mib=14336,
                fine_kv_pool_mib=128,
                adaptive_alpha_rebalance=True,
            )
        elif policy_name == "monolithic_dp":
            cfg = cls(
                policy_name="monolithic_dp",
                num_gpus=n,
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
                adaptive_alpha_rebalance=False,
            )
        elif policy_name == "twostage_shm":
            if n < 2 or n % 2 != 0:
                raise ValueError("twostage_shm requires an even number of GPUs >= 2")
            und = tuple(range(0, n, 2))
            gen = tuple(range(1, n, 2))
            cfg = cls(
                policy_name="twostage_shm",
                num_gpus=n,
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
                adaptive_alpha_rebalance=False,
            )
        elif policy_name == "static_nccl_fission":
            if n < 2:
                raise ValueError("static_nccl_fission requires >= 2 GPUs")
            if n == 8:
                und, gen = (0, 1, 4, 5), (2, 3, 6, 7)
            else:
                half = n // 2
                und, gen = tuple(range(half)), tuple(range(half, n))
            cfg = cls(
                policy_name="static_nccl_fission",
                num_gpus=n,
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
                adaptive_alpha_rebalance=False,
            )
        elif policy_name == "prism_tetriserve_coloc":
            # Prism/TetriServe-style colocation: KV-only VMM + step-level preemption (static weights, no scoped VAE)
            cfg = cls(
                policy_name="prism_tetriserve_coloc",
                num_gpus=n,
                routing_mode="sticky_least_loaded",
                use_tripool_vmm=True,
                use_expert_pager=False,
                use_scoped_vae=False,
                use_shared_numa_mirror=False,
                use_h2d_pacer=False,
                use_module3_cow_loopback=False,
                use_naive_offload=False,
                kv_vmm_only=True,
                preemption_mode="step_preemptive",
                initial_alphas={g: 1.0 for g in range(n)},
                super_pool_mib=512,
                fine_kv_pool_mib=2048,
                adaptive_alpha_rebalance=False,
            )
        else:  # "naive_host_offload"
            cfg = cls(
                policy_name="naive_host_offload",
                num_gpus=n,
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
                adaptive_alpha_rebalance=False,
            )

        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ValueError(f"unknown EnginePolicyConfig field: {k}")
            setattr(cfg, k, v)
        return cfg
