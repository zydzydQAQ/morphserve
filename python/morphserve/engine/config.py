"""Policy configurations, preset/ablation builders, `RunWindowOptions`, and `WorkerBudgetPlanner`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any

from morphserve.router.topology import default_operator_profile

VALID_POLICIES = (
    "morphserve",
    "monolithic_dp",
    "monolithic_kv_swap",
    "twostage_shm",
    "static_nccl_fission",
    "prism_tetriserve_coloc",
    "naive_host_offload",
)

MAIN_QPS_POLICIES = (
    "morphserve",
    "monolithic_dp",
    "monolithic_kv_swap",
    "static_nccl_fission",
    "prism_tetriserve_coloc",
)

STATIC_SPLIT_POLICIES = (
    "static_2und_6gen",
    "static_4und_4gen",
    "static_6und_2gen",
)


def _serialize_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in overrides.items():
        if k == "initial_alphas" and isinstance(v, Mapping):
            out[k] = {str(rk): float(rv) for rk, rv in v.items()}
        elif isinstance(v, tuple):
            out[k] = list(v)
        elif isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def _parse_named_policy_list(raw: Any) -> list[tuple[str, "EnginePolicyConfig"]]:
    out: list[tuple[str, EnginePolicyConfig]] = []
    if isinstance(raw, Mapping):
        for k, v in raw.items():
            out.append((str(k), EnginePolicyConfig.from_dict(v)))
        return out
    for item in raw:
        if isinstance(item, Mapping):
            name = str(item.get("name") or item.get("label") or item["policy_name"])
            cfg_d = item.get("config") or item.get("policy") or item
            out.append((name, EnginePolicyConfig.from_dict(cfg_d)))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((str(item[0]), EnginePolicyConfig.from_dict(item[1])))
    return out


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
    use_idle_kv_swap: bool = False
    kv_vmm_only: bool = False
    preemption_mode: str = "layer_chunk"  # "non_preemptive" | "step_preemptive" | "layer_chunk"
    cost_model_mode: str = "pipeline_analytical"  # "pipeline_analytical" | "atlas_3d" | "legacy_analytical"
    gpu_queue_mode: str = "memory_slack"  # "memory_slack" | "fcfs"
    gpu_queue_max_aging_ms: float = 45000.0
    gpu_queue_max_bypasses: int = 5
    kv_eviction_mode: str = "swap"  # "swap" | "recompute"
    static_und_ranks: tuple[int, ...] = ()
    static_gen_ranks: tuple[int, ...] = ()
    initial_alphas: dict[int, float] = field(default_factory=dict)

    # Per-class decoupled k_AR / k_DiT normalized unloaded SLOs (`D1-D5`) & bounded OOM recovery
    slo_scale_k: float = 2.0
    dit_slo_scale_k: float = 3.0
    ar_tpot_slo_ms: float = 101.40
    ar_ttft_slo_ms: float = 1500.0
    dit_512_slo_ms: float = 14916.30
    dit_1024_slo_ms: float = 57219.60
    dit_turn_slo_override_ms: float | None = None
    max_oom_retries: int = 1
    inter_turn_think_s_range: tuple[float, float] = (2.0, 25.0)
    think_time_dist: str = "lognormal"
    think_time_median_s: float = 6.0
    think_time_sigma: float = 0.75
    ar_tpot_slo_mode: str = "turn_p95"
    steady_window_mode: str = "time"
    warmup_window_s: float = 60.0
    steady_window_s: float = 150.0
    cooldown_s: float = 30.0
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
        return default_operator_profile().dit_turn_slo_ms(
            height=height,
            k=eff_k,
            model_family="internvl_u" if self.model_family == "internvl_u" else "bagel",
            steps=steps,
        )

    def ar_ttft_slo_for(
        self,
        has_image: bool,
        num_images: int = 0,
        image_res: int = 512,
        k: float | None = None,
    ) -> float:
        eff_k = self.slo_scale_k if k is None else float(k)
        return default_operator_profile().ar_ttft_slo_for(
            has_image=has_image,
            num_images=num_images,
            image_res=image_res,
            k=eff_k,
            model_family="internvl_u" if self.model_family == "internvl_u" else "bagel",
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the canonical C++/wire JSON dictionary schema."""
        raw = asdict(self)
        raw["initial_alphas"] = {str(k): float(v) for k, v in self.initial_alphas.items()}
        raw["static_und_ranks"] = [int(x) for x in self.static_und_ranks]
        raw["static_gen_ranks"] = [int(x) for x in self.static_gen_ranks]
        raw["inter_turn_think_s_range"] = [
            float(self.inter_turn_think_s_range[0]),
            float(self.inter_turn_think_s_range[1]),
        ]
        raw["allow_cascade_cfg_split"] = bool(self.allow_cascade_cfg_split)
        raw["use_idle_kv_swap"] = bool(self.use_idle_kv_swap)
        raw["think_time_dist"] = str(self.think_time_dist)
        raw["think_time_median_s"] = float(self.think_time_median_s)
        raw["think_time_sigma"] = float(self.think_time_sigma)
        raw["ar_tpot_slo_mode"] = str(self.ar_tpot_slo_mode)
        raw["steady_window_mode"] = str(self.steady_window_mode)
        raw["warmup_window_s"] = float(self.warmup_window_s)
        raw["steady_window_s"] = float(self.steady_window_s)
        raw["cooldown_s"] = float(self.cooldown_s)
        return raw

    def to_json(self) -> str:
        """Serialize to compact UTF-8 JSON matching C++ `PolicyConfig::from_json`."""
        return json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "EnginePolicyConfig":
        """Deserialize from UTF-8 JSON produced by `to_json()` or C++ `PolicyConfig::to_json()`."""
        return cls.from_dict(json.loads(raw))

    def with_overrides(self, **overrides: Any) -> "EnginePolicyConfig":
        """Return a new `EnginePolicyConfig` with field overrides applied without mutating `self`."""
        if not overrides:
            return self
        return replace(self, **overrides)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnginePolicyConfig":
        d = dict(data)
        d["initial_alphas"] = {int(k): float(v) for k, v in (d.get("initial_alphas") or {}).items()}
        d["static_und_ranks"] = tuple(int(x) for x in (d.get("static_und_ranks") or ()))
        d["static_gen_ranks"] = tuple(int(x) for x in (d.get("static_gen_ranks") or ()))
        tr = d.get("inter_turn_think_s_range")
        if tr is None:
            d["inter_turn_think_s_range"] = (2.0, 25.0)
        else:
            d["inter_turn_think_s_range"] = (float(tr[0]), float(tr[1]))
        for bool_key in ("allow_cascade_cfg_split", "use_idle_kv_swap"):
            if bool_key in d:
                if d[bool_key] is None:
                    d.pop(bool_key)
                else:
                    d[bool_key] = bool(d[bool_key])
        for str_key in (
            "think_time_dist",
            "ar_tpot_slo_mode",
            "steady_window_mode",
            "cost_model_mode",
            "gpu_queue_mode",
            "kv_eviction_mode",
        ):
            if str_key in d:
                if d[str_key] is None:
                    d.pop(str_key)
                else:
                    d[str_key] = str(d[str_key])
        for flt_key in (
            "dit_slo_scale_k",
            "think_time_median_s",
            "think_time_sigma",
            "warmup_window_s",
            "steady_window_s",
            "cooldown_s",
            "gpu_queue_max_aging_ms",
        ):
            if flt_key in d:
                if d[flt_key] is None:
                    d.pop(flt_key)
                else:
                    d[flt_key] = float(d[flt_key])
        if "gpu_queue_max_bypasses" in d:
            if d["gpu_queue_max_bypasses"] is None:
                d.pop("gpu_queue_max_bypasses")
            else:
                d["gpu_queue_max_bypasses"] = int(d["gpu_queue_max_bypasses"])
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
        out = call_json(
            L,
            L.ms_policy_from_preset_json,
            str(policy_name).encode("utf-8"),
            int(num_gpus),
            encode_json(_serialize_overrides(overrides)),
        )
        return cls.from_dict(out)


@dataclass(frozen=True)
class RunWindowOptions:
    """Per-run think-time, steady-state window, and SLO overrides for `MorphServeClusterEngine.run_suite`."""

    inter_turn_think_s_range: tuple[float, float] | None = None
    think_time_dist: str | None = None
    think_time_median_s: float | None = None
    think_time_sigma: float | None = None
    steady_window_mode: str | None = None
    warmup_window_s: float | None = None
    steady_window_s: float | None = None
    cooldown_s: float | None = None
    steady_state_trim_frac: float | None = None
    slo_scale_k: float | None = None
    dit_slo_scale_k: float | None = None
    ar_tpot_slo_mode: str | None = None

    def apply_to_policy(self, policy: EnginePolicyConfig) -> EnginePolicyConfig:
        """Return an updated `EnginePolicyConfig` copy applying non-None run window options."""
        updates: dict[str, Any] = {}
        if self.inter_turn_think_s_range is not None:
            updates["inter_turn_think_s_range"] = (
                float(self.inter_turn_think_s_range[0]),
                float(self.inter_turn_think_s_range[1]),
            )
        if self.think_time_dist is not None:
            updates["think_time_dist"] = str(self.think_time_dist)
        if self.think_time_median_s is not None:
            updates["think_time_median_s"] = float(self.think_time_median_s)
        if self.think_time_sigma is not None:
            updates["think_time_sigma"] = float(self.think_time_sigma)
        if self.steady_window_mode is not None:
            updates["steady_window_mode"] = str(self.steady_window_mode)
        if self.warmup_window_s is not None:
            updates["warmup_window_s"] = float(self.warmup_window_s)
        if self.steady_window_s is not None:
            updates["steady_window_s"] = float(self.steady_window_s)
        if self.cooldown_s is not None:
            updates["cooldown_s"] = float(self.cooldown_s)
        if self.steady_state_trim_frac is not None:
            updates["steady_state_trim_frac"] = float(self.steady_state_trim_frac)
        if self.ar_tpot_slo_mode is not None:
            updates["ar_tpot_slo_mode"] = str(self.ar_tpot_slo_mode)

        fam = "internvl_u" if policy.model_family == "internvl_u" else "bagel"
        if self.slo_scale_k is not None:
            k_ar = float(self.slo_scale_k)
            ar_slos = default_operator_profile().slo_thresholds_ms(k=k_ar, model_family=fam)
            updates["slo_scale_k"] = k_ar
            updates["ar_ttft_slo_ms"] = ar_slos["ar_ttft_slo_ms"]
            updates["ar_tpot_slo_ms"] = ar_slos["ar_tpot_slo_ms"]
        if self.dit_slo_scale_k is not None:
            k_dit = float(self.dit_slo_scale_k)
            dit_slos = default_operator_profile().slo_thresholds_ms(k=k_dit, model_family=fam)
            updates["dit_slo_scale_k"] = k_dit
            updates["dit_512_slo_ms"] = dit_slos["dit_512_slo_ms"]
            updates["dit_1024_slo_ms"] = dit_slos["dit_1024_slo_ms"]

        return replace(policy, **updates) if updates else policy


def build_ablation_configs(num_gpus: int = 8, **overrides: Any) -> list[tuple[str, EnginePolicyConfig]]:
    """Build Leave-One-Out M1-M4 ablation configs via C++17 `build_ablation_configs`."""
    from morphserve._core import call_json, encode_json, get_core_lib

    L = get_core_lib()
    raw = call_json(
        L,
        L.ms_policy_build_ablation_configs_json,
        int(num_gpus),
        encode_json(_serialize_overrides(overrides)),
    )
    return _parse_named_policy_list(raw)


def build_static_split_configs(num_gpus: int = 8, **overrides: Any) -> list[tuple[str, EnginePolicyConfig]]:
    """Build elastic + static split-ratio sensitivity configs (`2:6`, `4:4`, `6:2`) via C++17."""
    from morphserve._core import call_json, encode_json, get_core_lib

    L = get_core_lib()
    raw = call_json(
        L,
        L.ms_policy_build_static_split_configs_json,
        int(num_gpus),
        encode_json(_serialize_overrides(overrides)),
    )
    return _parse_named_policy_list(raw)


class WorkerBudgetPlanner:
    """Public Python adapter over C++17 `morphserve::control::WorkerBudgetPlanner` (`ms_budget_*`)."""

    @staticmethod
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

    @staticmethod
    def compute_internvlu_quota_overrides(
        policy_name: str,
        num_gpus: int,
        vendor_dir: str | Path,
        *,
        constrained_gib: float = 12.0,
        total_gpu_gib: float = 40.0,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        from morphserve._core import call_json, get_core_lib

        L = get_core_lib()
        out = call_json(
            L,
            L.ms_budget_internvlu_quota_overrides_json,
            str(policy_name).encode("utf-8"),
            int(num_gpus),
            str(vendor_dir).encode("utf-8"),
            float(constrained_gib),
            float(total_gpu_gib),
        )
        ov = dict(out["overrides"])
        tr = ov.get("inter_turn_think_s_range")
        if isinstance(tr, list) and len(tr) == 2:
            ov["inter_turn_think_s_range"] = (float(tr[0]), float(tr[1]))
        if "initial_alphas" in ov and isinstance(ov["initial_alphas"], Mapping):
            ov["initial_alphas"] = {int(k): float(v) for k, v in ov["initial_alphas"].items()}
        budget = dict(out.get("budget") or out["budget_plan"])
        return ov, budget

    @staticmethod
    def verify_internvlu_memory_quota(
        run_result: Mapping[str, Any],
        policy: EnginePolicyConfig,
        *,
        constrained_gib: float = 12.0,
    ) -> dict[str, Any]:
        from morphserve._core import call_json, encode_json, get_core_lib

        L = get_core_lib()
        return call_json(
            L,
            L.ms_budget_verify_internvlu_quota_json,
            encode_json(dict(run_result)),
            encode_json(policy.to_dict()),
            float(constrained_gib),
        )


__all__ = [
    "EnginePolicyConfig",
    "MAIN_QPS_POLICIES",
    "RunWindowOptions",
    "STATIC_SPLIT_POLICIES",
    "VALID_POLICIES",
    "WorkerBudgetPlanner",
    "build_ablation_configs",
    "build_static_split_configs",
]
