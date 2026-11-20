#!/usr/bin/env bash
# Phase-0 Job 0c (1x8 GPU): Full 8xA100 PCIe/NUMA topology, P2P, NUMA-pinned H2D, memfd & NCCL P2P probe.
set -euo pipefail

source /shared/data/MORPHSERVE/scripts/morphserve_common.sh

kairos_check_shared_storage

python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
  --output "${RUN_DIR}/runtime-facts.json"

python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
  --memory-summary "${RUN_DIR}/memory-summary.json" &
SAMPLER_PID=$!

_morphserve_job0c_cleanup() {
  if kill -0 "$SAMPLER_PID" 2>/dev/null; then
    kill -TERM "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
  fi
  rm -rf "$MORPHSERVE_LOCAL_TMP" 2>/dev/null || true
  _kairos_cleanup
}
trap _morphserve_job0c_cleanup EXIT

python3 -m morphserve.bench.topo_probe \
  --output-dir "${RUN_DIR}" \
  --size-mib 512

kairos_check_shared_storage
echo "== Phase-0 Job 0c completed cleanly: ${RUN_DIR}"
