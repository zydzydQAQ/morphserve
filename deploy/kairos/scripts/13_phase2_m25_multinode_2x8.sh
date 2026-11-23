#!/usr/bin/env bash
# Phase-2 Job 2f (2x8 = 16 GPUs): Milestone M2.5 Cross-Node Fabric Calibration & 16-GPU E2E Serving Benchmark.
# Preserves platform-injected $RANK, $WORLD_SIZE, $MASTER_ADDR, $MASTER_PORT per AGENTS.md!
set -euo pipefail

# Prevent non-zero ranks from concurrently tee-appending into Node 0's NFS log file.
if [ "${RANK:-0}" != "0" ]; then
  export KAIROS_DISABLE_TEE="${KAIROS_DISABLE_TEE:-1}"
fi

source /shared/data/MORPHSERVE/scripts/morphserve_common.sh

if [ "${RANK:-0}" != "0" ] && [ "${KAIROS_DISABLE_NODE_TEE:-0}" != "1" ]; then
  exec > >(tee -a "${RUN_DIR}/$(basename "$0" .sh)-node${RANK}.log") 2>&1
  echo "== $(basename "$0") node_rank=${RANK} project=${KAIROS_PROJECT} run=${KAIROS_RUN_ID} host=$(hostname) $(date -Is)"
fi

MODEL_DIR="${MODELS}/BAGEL-7B-MoT"
BENCH_DIR="${DATA}/benchmarks/interleaved-bench-v1"
MODE_2X8="${MORPHSERVE_2X8_MODE:-both}"

if [ "${MODE_2X8}" != "spotcheck" ]; then
  if [ ! -f "${MODEL_DIR}/MODEL_MANIFEST.json" ] || [ ! -f "${BENCH_DIR}/BENCH_MANIFEST.json" ]; then
    echo "error: missing BAGEL-7B-MoT or Interleaved-Bench manifest on rank ${RANK:-0}" >&2
    exit 1
  fi
fi

SAMPLER_PID=""
if [ "${RANK:-0}" = "0" ]; then
  kairos_check_shared_storage
  python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
    --output "${RUN_DIR}/runtime-facts-node0.json" \
    --model-dir "bagel=${MODEL_DIR}"
  python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
    --memory-summary "${RUN_DIR}/memory-summary-node0.json" &
  SAMPLER_PID=$!
fi

_stop_node0_sampler() {
  if [ -n "${SAMPLER_PID:-}" ] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then
    kill -TERM "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
  fi
  SAMPLER_PID=""
}

_cleanup() {
  _stop_node0_sampler
  rm -rf "$MORPHSERVE_LOCAL_TMP" /tmp/ms_m25_* /tmp/morphserve_h2d_locks /dev/shm/ms_shm_* 2>/dev/null || true
  if [ "${RANK:-0}" = "0" ]; then
    rm -rf "${RUN_DIR}"/_tmp_m25_* 2>/dev/null || true
  fi
  _kairos_cleanup
}
trap _cleanup EXIT

python3 -m morphserve.bench.validate_phase2_multinode_2x8 \
  --mode "${MODE_2X8}" \
  --model-dir "${MODEL_DIR}" \
  --bench-root "${BENCH_DIR}" \
  --num-sessions "${MORPHSERVE_2X8_SESSIONS:-160}" \
  --qps "${MORPHSERVE_2X8_QPS:-1.6}" \
  --output-dir "${RUN_DIR}" \
  --gpus-per-node 8

_stop_node0_sampler
if [ "${RANK:-0}" = "0" ]; then
  kairos_check_shared_storage
  echo "== Phase-2 Job 2f (M2.5 2x8 Multi-Node) completed cleanly: ${RUN_DIR}"
fi
