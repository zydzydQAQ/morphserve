#!/usr/bin/env bash
# Phase-2 Job 2f (2x8 = 16 GPUs): Milestone M2.5 Cross-Node Fabric & Dual-Path Routing Spot-Check.
# Preserves platform-injected $RANK, $WORLD_SIZE, $MASTER_ADDR, $MASTER_PORT per AGENTS.md!
set -euo pipefail

source /shared/data/MORPHSERVE/scripts/morphserve_common.sh

if [ "${RANK:-0}" = "0" ]; then
  kairos_check_shared_storage
  python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
    --output "${RUN_DIR}/runtime-facts-node0.json"
fi

_cleanup() {
  rm -rf "$MORPHSERVE_LOCAL_TMP" 2>/dev/null || true
  _kairos_cleanup
}
trap _cleanup EXIT

python3 -m morphserve.bench.validate_phase2_multinode_2x8 \
  --output-dir "${RUN_DIR}" \
  --gpus-per-node 8

if [ "${RANK:-0}" = "0" ]; then
  kairos_check_shared_storage
  echo "== Phase-2 Job 2f (M2.5 2x8 Multi-Node) completed cleanly: ${RUN_DIR}"
fi
