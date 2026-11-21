#!/usr/bin/env bash
# Phase-2 Job 2b (1x8 GPUs): Live 8xA100 smoke gate for MorphServeClusterEngine across all 6 presets.
set -euo pipefail

source /shared/data/MORPHSERVE/scripts/morphserve_common.sh

kairos_check_shared_storage

MODEL_DIR="${MODELS}/BAGEL-7B-MoT"
BENCH_DIR="${DATA}/benchmarks/interleaved-bench-v1"
if [ ! -f "${MODEL_DIR}/MODEL_MANIFEST.json" ]; then
  echo "error: verified BAGEL-7B-MoT not found at ${MODEL_DIR}" >&2
  exit 1
fi
if [ ! -f "${BENCH_DIR}/BENCH_MANIFEST.json" ]; then
  echo "error: Interleaved-Bench missing at ${BENCH_DIR}" >&2
  exit 1
fi

python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
  --output "${RUN_DIR}/runtime-facts.json" \
  --model-dir "bagel=${MODEL_DIR}"

python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
  --memory-summary "${RUN_DIR}/memory-summary.json" &
SAMPLER_PID=$!

_morphserve_job2b_cleanup() {
  if kill -0 "$SAMPLER_PID" 2>/dev/null; then
    kill -TERM "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
  fi
  rm -rf "$MORPHSERVE_LOCAL_TMP" /tmp/morphserve_h2d_locks /tmp/morphserve_smoke_* 2>/dev/null || true
  _kairos_cleanup
}
trap _morphserve_job2b_cleanup EXIT

python3 -m morphserve.bench.validate_phase2_engine_smoke \
  --model-dir "${MODEL_DIR}" \
  --bench-root "${BENCH_DIR}" \
  --output-dir "${RUN_DIR}" \
  --qps 0.8 \
  --num-gpus 8

kairos_check_shared_storage
echo "== Phase-2 Job 2b completed cleanly: ${RUN_DIR}"
