#!/usr/bin/env bash
# Phase-0 Job 0a (1x1 GPU): Toolchain/topology/vllm-omni audit + TriPool-VMM live A100 verification.
set -euo pipefail

source /shared/data/MORPHSERVE/scripts/morphserve_common.sh

kairos_check_shared_storage

# 1. Runtime facts via kairos-toolkit auditor
python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
  --output "${RUN_DIR}/runtime-facts.json"

# 2. Bounded NVML memory/utilization sampler in background
python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
  --memory-summary "${RUN_DIR}/memory-summary.json" &
SAMPLER_PID=$!

_morphserve_job0a_cleanup() {
  if kill -0 "$SAMPLER_PID" 2>/dev/null; then
    kill -TERM "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
  fi
  rm -rf "$MORPHSERVE_LOCAL_TMP" 2>/dev/null || true
  _kairos_cleanup
}
trap _morphserve_job0a_cleanup EXIT

# 3. Run comprehensive Phase-0 probe + C++ cuMemMap 2MB microbenchmark + PyTorch MemPool test
python3 -m morphserve.probe \
  --output-dir "${RUN_DIR}" \
  --bin-dir "${MORPHSERVE_BIN_DIR}" \
  --lib-dir "${MORPHSERVE_LIB_DIR}"

kairos_check_shared_storage
echo "== Phase-0 Job 0a completed cleanly: ${RUN_DIR}"
