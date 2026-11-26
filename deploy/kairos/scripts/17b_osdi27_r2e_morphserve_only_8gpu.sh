#!/usr/bin/env bash
# OSDI '27 Job R2e-r3 (1x8 GPUs): Re-run `morphserve` on build f1b4b83 across Low-to-Knee QPS
# (0.05..0.25 + 0.40, 1.10 RPS) and merge unmodified 5-baseline rows from R2e-r1 + high-QPS rows from R2a-r6.
set -euo pipefail

source /shared/data/MORPHSERVE/scripts/morphserve_common.sh

kairos_check_shared_storage

MODEL_DIR="${MODELS}/BAGEL-7B-MoT"
BENCH_DIR="${DATA}/benchmarks/interleaved-bench-v1"
PREV_SUITE_A="${MORPHSERVE_PREV_SUITE_A:-/shared/output/MORPHSERVE/runs/morphserve-osdi27-r2a-suitea-20261125-r6/m23-suite-a-report.json}"
BASE_LOWQPS="${MORPHSERVE_BASELINE_LOWQPS:-/shared/output/MORPHSERVE/runs/morphserve-osdi27-r2e-lowqps-20261125/m23-lowqps-slo-report.json}"

if [ ! -f "${MODEL_DIR}/MODEL_MANIFEST.json" ] || [ ! -f "${BENCH_DIR}/BENCH_MANIFEST.json" ]; then
  echo "error: missing BAGEL-7B-MoT or Interleaved-Bench manifest" >&2
  exit 1
fi

python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
  --output "${RUN_DIR}/runtime-facts.json" \
  --model-dir "bagel=${MODEL_DIR}"

python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
  --memory-summary "${RUN_DIR}/memory-summary.json" &
SAMPLER_PID=$!

_cleanup() {
  if [ -n "${SAMPLER_PID:-}" ] && kill -0 "$SAMPLER_PID" 2>/dev/null; then
    kill -TERM "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
  fi
  SAMPLER_PID=""
  rm -rf "$MORPHSERVE_LOCAL_TMP" /tmp/ms_m23low_* /tmp/morphserve_h2d_locks /dev/shm/ms_shm_* 2>/dev/null || true
  _kairos_cleanup
}
trap _cleanup EXIT

DRIVER_PY="${MORPHSERVE_LOCAL_TMP}/ms_r2e_morphserve_driver.py"
cat <<'PY' > "${DRIVER_PY}"
import json
from pathlib import Path
import shutil
import sys

from morphserve.bench.harness import compute_sustainable_qps_summary
from morphserve.bench.run_phase2_bagel_main import run_suite_a_lowqps_and_cdf


def main() -> None:
    model_dir = Path(sys.argv[1])
    bench_dir = Path(sys.argv[2])
    run_dir = Path(sys.argv[3])
    prev_suite_a = Path(sys.argv[4])
    base_lowqps = Path(sys.argv[5])

    rep = run_suite_a_lowqps_and_cdf(
        model_dir=model_dir,
        bench_root=bench_dir,
        output_dir=run_dir,
        num_gpus=8,
        qps_points=(0.05, 0.10, 0.15, 0.20, 0.25, 0.40, 1.10),
        min_sessions=28,
        max_sessions=48,
        policies=("morphserve",),
        merge_suite_a_report=prev_suite_a if prev_suite_a.is_file() else None,
    )

    raw_path = run_dir / "m23-lowqps-slo-report.json"
    shutil.copy2(raw_path, run_dir / "m23-lowqps-morphserve-raw.json")

    merged_lowqps = list(rep["qps_sweep"])
    if base_lowqps.is_file():
        base_data = json.loads(base_lowqps.read_text(encoding="utf-8"))
        for row in base_data.get("qps_sweep", []):
            if str(row.get("policy_name")) != "morphserve":
                merged_lowqps.append(row)

    combined_for_summary = list(merged_lowqps)
    if prev_suite_a.is_file():
        sa = json.loads(prev_suite_a.read_text(encoding="utf-8"))
        seen = {(str(r["policy_name"]), round(float(r["offered_qps"]), 3)) for r in combined_for_summary}
        for old_r in sa.get("qps_sweep", []):
            key = (str(old_r["policy_name"]), round(float(old_r["offered_qps"]), 3))
            if key not in seen:
                combined_for_summary.append(old_r)

    rep["qps_sweep"] = merged_lowqps
    rep["sustainable_qps_summary"] = compute_sustainable_qps_summary(combined_for_summary)
    rep["baseline_lowqps_source"] = str(base_lowqps) if base_lowqps.is_file() else None
    rep["suite_a_merge_source"] = str(prev_suite_a) if prev_suite_a.is_file() else None
    rep["status"] = "ok" if all(int(r.get("error_sessions", 0)) == 0 for r in merged_lowqps) else "check_failed"
    raw_path.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote merged {raw_path} ({len(merged_lowqps)} rows, status={rep['status']})", flush=True)


if __name__ == "__main__":
    main()
PY

python3 "${DRIVER_PY}" "${MODEL_DIR}" "${BENCH_DIR}" "${RUN_DIR}" "${PREV_SUITE_A}" "${BASE_LOWQPS}"
rm -f "${DRIVER_PY}"

kairos_check_shared_storage
echo "== OSDI '27 Job R2e-r3 (MorphServe Low-QPS + Merged Baselines) completed cleanly: ${RUN_DIR}"
