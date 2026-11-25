#!/usr/bin/env bash
# OSDI '27 Job R2e (1x8 GPUs): Low-to-Knee QPS Sweep (0.05..0.25 + 0.40, 1.10 RPS),
# per-suite reset_peak_memory_stats / NVML peak verification, and OOM-aware empirical Latency CDFs.
set -euo pipefail

source /shared/data/MORPHSERVE/scripts/morphserve_common.sh

kairos_check_shared_storage

MODEL_DIR="${MODELS}/BAGEL-7B-MoT"
BENCH_DIR="${DATA}/benchmarks/interleaved-bench-v1"
PREV_SUITE_A="/shared/output/MORPHSERVE/runs/morphserve-osdi27-r2a-suitea-20261124-r5/m23-suite-a-report.json"

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

MERGE_ARGS=()
if [ -f "${PREV_SUITE_A}" ]; then
  MERGE_ARGS+=(--merge-suite-a-report "${PREV_SUITE_A}")
fi

python3 -m morphserve.bench.run_phase2_bagel_main \
  --mode suite_a_lowqps \
  --model-dir "${MODEL_DIR}" \
  --bench-root "${BENCH_DIR}" \
  --output-dir "${RUN_DIR}" \
  --num-gpus 8 \
  --low-sessions "${MORPHSERVE_LOW_SESSIONS:-28}" \
  --knee-sessions "${MORPHSERVE_KNEE_SESSIONS:-48}" \
  "${MERGE_ARGS[@]}" \
  ${MORPHSERVE_MAX_SESSIONS:+--max-sessions "${MORPHSERVE_MAX_SESSIONS}"}

kairos_check_shared_storage
echo "== OSDI '27 Job R2e (Low-QPS SLO Curve & OOM-Aware CDF) completed cleanly: ${RUN_DIR}"
