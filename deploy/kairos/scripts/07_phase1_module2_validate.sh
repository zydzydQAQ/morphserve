#!/usr/bin/env bash
# Phase-1 Job 1d (1x8 GPUs): Live 8xA100 validation of Module 2 (Dual-Path Morphing Router + Half-Node H2D Pacer + NCCL State Shipper).
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

_morphserve_job1d_cleanup() {
  if kill -0 "$SAMPLER_PID" 2>/dev/null; then
    kill -TERM "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
  fi
  rm -rf "$MORPHSERVE_LOCAL_TMP" /tmp/morphserve_h2d_locks 2>/dev/null || true
  _kairos_cleanup
}
trap _morphserve_job1d_cleanup EXIT

python3 -m morphserve.bench.validate_module2_dualpath \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${RUN_DIR}" \
  --steps 20

kairos_check_shared_storage
echo "== Phase-1 Job 1d (Module 2 Dual-Path 8-GPU Validation) completed cleanly: ${RUN_DIR}"
