#!/usr/bin/env bash
# Phase-1 Job 1b (1x1 GPU): Live A100-40GB validation of Module 3 (Shared-Prefix CFG Attention + LECP CoW + GPU Loopback).
set -euo pipefail

source /shared/data/MORPHSERVE/scripts/morphserve_common.sh

kairos_check_shared_storage

MODEL_DIR="${MODELS}/BAGEL-7B-MoT"
if [ ! -f "${MODEL_DIR}/MODEL_MANIFEST.json" ]; then
  echo "error: verified BAGEL-7B-MoT not found at ${MODEL_DIR}" >&2
  exit 1
fi

python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
  --output "${RUN_DIR}/runtime-facts.json" \
  --model-dir "bagel=${MODEL_DIR}"

python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
  --memory-summary "${RUN_DIR}/memory-summary.json" &
SAMPLER_PID=$!

_morphserve_job1b_cleanup() {
  if kill -0 "$SAMPLER_PID" 2>/dev/null; then
    kill -TERM "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
  fi
  rm -rf "$MORPHSERVE_LOCAL_TMP" 2>/dev/null || true
  _kairos_cleanup
}
trap _morphserve_job1b_cleanup EXIT

python3 -m morphserve.bench.validate_module3_cfg_loopback \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${RUN_DIR}" \
  --steps 20

kairos_check_shared_storage
echo "== Phase-1 Job 1b (Module 3 Shared-Prefix CFG + GPU Loopback) completed cleanly: ${RUN_DIR}"
