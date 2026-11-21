#!/usr/bin/env bash
# Phase-2 Job 2a (1x1 GPU): Re-validate Module 3 high-res gate + build Interleaved-Bench (<1 GiB).
set -euo pipefail

source /shared/data/MORPHSERVE/scripts/morphserve_common.sh

kairos_check_shared_storage

MODEL_DIR="${MODELS}/BAGEL-7B-MoT"
if [ ! -f "${MODEL_DIR}/MODEL_MANIFEST.json" ]; then
  echo "error: verified BAGEL-7B-MoT not found at ${MODEL_DIR}" >&2
  exit 1
fi

BENCH_DIR="${DATA}/benchmarks/interleaved-bench-v1"
SOURCES_DIR="${BENCH_DIR}/sources"
if [ ! -f "${SOURCES_DIR}/SOURCES_MANIFEST.json" ]; then
  echo "error: staged bench sources missing at ${SOURCES_DIR}" >&2
  exit 1
fi

python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
  --output "${RUN_DIR}/runtime-facts.json" \
  --model-dir "bagel=${MODEL_DIR}"

python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
  --memory-summary "${RUN_DIR}/memory-summary.json" &
SAMPLER_PID=$!

_morphserve_job2a_cleanup() {
  if kill -0 "$SAMPLER_PID" 2>/dev/null; then
    kill -TERM "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
  fi
  rm -rf "$MORPHSERVE_LOCAL_TMP" 2>/dev/null || true
  _kairos_cleanup
}
trap _morphserve_job2a_cleanup EXIT

echo "== Step 1: Module 3 verification (including 1024px_turn2_8k >= 1.00x gate)"
python3 -m morphserve.bench.validate_module3_cfg_loopback \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${RUN_DIR}" \
  --steps 20

echo "== Step 2: Build Interleaved-Bench (<1 GiB) from staged sources + hf-mirror parquets"
python3 -m morphserve.bench.interleaved_bench build \
  --sources-dir "${SOURCES_DIR}" \
  --output-dir "${BENCH_DIR}" \
  --tmp-dir "${MORPHSERVE_LOCAL_TMP}/bench_build"

cp "${BENCH_DIR}/BENCH_MANIFEST.json" "${RUN_DIR}/BENCH_MANIFEST.json"

kairos_check_shared_storage
echo "== Phase-2 Job 2a completed cleanly: ${RUN_DIR}"
