#!/usr/bin/env bash
# Phase-2 Wave-3 Job 2g (1x8 GPUs): Full 80-Session Main Mix + Static Split-Ratio (2:6/4:4/6:2) Sensitivity.
set -euo pipefail

source /shared/data/MORPHSERVE/scripts/morphserve_common.sh

kairos_check_shared_storage

MODEL_DIR="${MODELS}/BAGEL-7B-MoT"
BENCH_DIR="${DATA}/benchmarks/interleaved-bench-v1"
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
  rm -rf "$MORPHSERVE_LOCAL_TMP" /tmp/ms_deep_* /tmp/ms_deep80_* /tmp/ms_ratio_* /tmp/morphserve_h2d_locks /dev/shm/ms_shm_* 2>/dev/null || true
  _kairos_cleanup
}
trap _cleanup EXIT

python3 -m morphserve.bench.run_phase2_overnight_deep \
  --model-dir "${MODEL_DIR}" \
  --bench-root "${BENCH_DIR}" \
  --output-dir "${RUN_DIR}" \
  --num-gpus 8

kairos_check_shared_storage
echo "== Phase-2 Wave-3 Job 2g completed cleanly: ${RUN_DIR}"
