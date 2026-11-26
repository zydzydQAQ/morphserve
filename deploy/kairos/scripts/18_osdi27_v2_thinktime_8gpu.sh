#!/usr/bin/env bash
# OSDI '27 Interleaved-Bench v2 Job R2f (1x8 GPUs):
#   Think-Time Sensitivity Sweep (0.0 / 5.0 / 20.0 / 60.0 s -> Active-JCT, KV host swap-outs & SLO attainment).
set -euo pipefail

source /shared/data/MORPHSERVE/scripts/morphserve_common.sh

kairos_check_shared_storage

MODEL_DIR="${MODELS}/BAGEL-7B-MoT"
BENCH_DIR="${MORPHSERVE_BENCH_DIR:-${DATA}/benchmarks/interleaved-bench-v1}"
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
  rm -rf "$MORPHSERVE_LOCAL_TMP" /tmp/ms_m23think_* /tmp/ms_think_* /tmp/morphserve_h2d_locks /dev/shm/ms_shm_* 2>/dev/null || true
  _kairos_cleanup
}
trap _cleanup EXIT

python3 -m morphserve.bench.run_phase2_bagel_main \
  --mode think_time_sweep \
  --model-dir "${MODEL_DIR}" \
  --bench-root "${BENCH_DIR}" \
  --output-dir "${RUN_DIR}" \
  --num-gpus 8 \
  ${MORPHSERVE_THINK_QPS:+--think-qps "${MORPHSERVE_THINK_QPS}"} \
  ${MORPHSERVE_THINK_TIMES:+--think-times "${MORPHSERVE_THINK_TIMES}"} \
  ${MORPHSERVE_POLICIES:+--policies "${MORPHSERVE_POLICIES}"} \
  ${MORPHSERVE_MAX_SESSIONS:+--max-sessions "${MORPHSERVE_MAX_SESSIONS}"}

kairos_check_shared_storage
echo "== OSDI '27 Interleaved-Bench v2 Job R2f (Think-Time Sweep) completed cleanly: ${RUN_DIR}"
