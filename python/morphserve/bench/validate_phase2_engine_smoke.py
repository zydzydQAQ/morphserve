"""Phase-2 Milestone M2.2 live 8xA100 cluster smoke validation across MorphServe and all 5 baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Callable

from morphserve.bench.harness import BenchmarkRunner, resolve_sources_dir
from morphserve.bench.interleaved_bench import (
    WorkloadSuite,
    load_burstgpt_interarrivals,
    load_gentd26_interarrivals,
)
from morphserve.engine.config import EnginePolicyConfig, VALID_POLICIES

MULTI_GPU_ONLY_POLICIES = ("twostage_shm", "static_nccl_fission")


def run_smoke_validation(
    model_dir: Path,
    bench_root: Path,
    output_dir: Path,
    qps: float = 0.8,
    num_gpus: int = 8,
    policies: tuple[str, ...] = VALID_POLICIES,
    *,
    fleet_factory: Callable[[EnginePolicyConfig], Any] | None = None,
    engine_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    num_gpus = int(num_gpus)
    if num_gpus < 1:
        raise ValueError("num_gpus must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    suite_path = bench_root / "suites/smoke_8gpu_12.json"
    if not suite_path.is_file():
        raise FileNotFoundError(f"missing Interleaved-Bench smoke suite: {suite_path}")
    suite = WorkloadSuite.load_json(suite_path)

    policies = tuple(policies)
    if num_gpus < 2 and policies == VALID_POLICIES:
        policies = tuple(p for p in policies if p not in MULTI_GPU_ONLY_POLICIES)

    sources_dir = resolve_sources_dir(bench_root)
    burst_csv = sources_dir / "burstgpt_1_sample.csv"
    gentd_csv = sources_dir / "gentd26_lora_request_trace.csv"
    burst_deltas = load_burstgpt_interarrivals(burst_csv) if burst_csv.is_file() else None
    gentd_deltas = load_gentd26_interarrivals(gentd_csv) if gentd_csv.is_file() else None

    runner = BenchmarkRunner(
        model_dir=model_dir,
        bench_root=bench_root,
        tmp_root_base=output_dir,
        engine_factory=engine_factory,
        fleet_factory=fleet_factory,
    )

    results: list[dict[str, Any]] = []
    for preset in policies:
        policy = EnginePolicyConfig.from_preset(preset, num_gpus=num_gpus)
        with runner.session(policy, tmp_tag=f"_tmp_smoke_{preset}") as sess:
            rep = sess.run_suite(
                suite,
                target_qps=qps,
                burstgpt_deltas=burst_deltas,
                gentd26_deltas=gentd_deltas,
            )
            rep_dict = rep.to_dict()
            rep_dict["startup_seconds"] = sess.startup_seconds
            rep_dict["init_summary"] = {
                "host_shmem_delta_gib": sess.init_info.get("host_shmem_delta_gib"),
                "mirror_inodes": sorted(
                    {
                        w.get("mirror_inode")
                        for w in (sess.init_info.get("workers") or {}).values()
                        if w.get("mirror_inode") is not None
                    }
                ),
            }
            results.append(rep_dict)
            print(
                f"[SMOKE] policy={preset} status={rep.status} completed={rep.completed_sessions}/{rep.total_sessions} "
                f"goodput={rep.joint_slo_goodput_rps:.3f} ar_tpot_p95={rep.ar_tpot_p95_ms}ms "
                f"ship={rep.path_a_ships} morph={rep.path_b_morphs} shmem={rep.host_shmem_delta_gib}GiB",
                flush=True,
            )

    by_policy = {r["policy_name"]: r for r in results}
    ms = by_policy.get("morphserve")
    all_pass = bool(ms is not None)
    if ms is not None:
        fission = by_policy.get("static_nccl_fission")
        nccl_ship_ok = True if fission is None else (fission["path_a_ships"] > 0)
        dual_path_ok = True if num_gpus < 2 else ((ms["path_a_ships"] + ms["path_b_morphs"]) > 0)
        all_pass = (
            ms["status"] == "ok"
            and ms["completed_sessions"] == ms["total_sessions"]
            and ms["oom_sessions"] == 0
            and ms["worker_oom_count"] == 0
            and ms["vmm_overflow_creates"] == 0
            and ms["host_shmem_delta_gib"] <= 32.0
            and dual_path_ok
            and nccl_ship_ok
            and all(r["error_sessions"] == 0 for r in results)
        )

    report = {
        "schema": "morphserve.phase2.m22_smoke.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_dir": str(model_dir),
        "bench_root": str(bench_root),
        "num_gpus": num_gpus,
        "offered_qps": qps,
        "results": results,
        "status": "ok" if all_pass else "check_failed",
    }
    out_path = output_dir / "m22-smoke-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MorphServe Phase-2 M2.2 8-GPU Engine Smoke Validator")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qps", type=float, default=0.8)
    parser.add_argument("--num-gpus", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rep = run_smoke_validation(
        model_dir=args.model_dir,
        bench_root=args.bench_root,
        output_dir=args.output_dir,
        qps=args.qps,
        num_gpus=args.num_gpus,
    )
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
