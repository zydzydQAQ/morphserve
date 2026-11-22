#!/usr/bin/env bash
# Phase-2 Job 2e (1x8 GPUs): Milestone M2.4 InternVL-U (~8.16 GiB) Download + SHA256 Verify + 8xA100 Generality Validation.
set -euo pipefail

source /shared/data/MORPHSERVE/scripts/morphserve_common.sh

# Strict pre-download storage gate (< 790G and < 80% used)
USED_PCT=$(df -P "${KAIROS_SHARED_ROOT:-/shared}" | awk 'NR==2 {for (i=1; i<=NF; i++) if ($i ~ /^[0-9]+%$/) {sub("%", "", $i); print $i; exit}}')
if [ -z "${USED_PCT}" ] || [ "${USED_PCT}" -ge 80 ]; then
  echo "error: /shared usage ${USED_PCT}% >= 80% pre-download gate; stopping immediately" >&2
  exit 1
fi
kairos_check_shared_storage

IVLU_DIR="${MODELS}/InternVL-U"
VENDOR_DIR="${DATA}/vendor/internvlu_pkg"
if [ ! -d "${VENDOR_DIR}/internvlu" ]; then
  echo "error: vendored internvlu package missing at ${VENDOR_DIR}" >&2
  exit 1
fi

if [ ! -f "${IVLU_DIR}/MODEL_MANIFEST.json" ]; then
  echo "== Fetching and SHA256-verifying InternVL-U/InternVL-U (~8.16 GiB) from mirror"
  python3 -m morphserve.tools.fetch_hf \
    --repo-id "InternVL-U/InternVL-U" \
    --revision "f012d760e69712bb47f7d3d09a24280f346cee01" \
    --dest "${IVLU_DIR}" \
    --endpoint "https://hf-mirror.com" \
    --max-workers 16
fi

kairos_check_shared_storage

python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
  --output "${RUN_DIR}/runtime-facts.json" \
  --model-dir "internvlu=${IVLU_DIR}"

python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
  --memory-summary "${RUN_DIR}/memory-summary.json" &
SAMPLER_PID=$!

_cleanup() {
  if [ -n "${SAMPLER_PID:-}" ] && kill -0 "$SAMPLER_PID" 2>/dev/null; then
    kill -TERM "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
  fi
  SAMPLER_PID=""
  rm -rf "$MORPHSERVE_LOCAL_TMP" /tmp/ms_ivlu_* /tmp/ivlu_parity_* /tmp/morphserve_h2d_locks /dev/shm/ms_shm_* 2>/dev/null || true
  _kairos_cleanup
}
trap _cleanup EXIT

python3 -m morphserve.bench.validate_phase2_internvl_u \
  --model-dir "${IVLU_DIR}" \
  --vendor-dir "${VENDOR_DIR}" \
  --output-dir "${RUN_DIR}" \
  --num-gpus 8

cp "${IVLU_DIR}/MODEL_MANIFEST.json" "${RUN_DIR}/INTERNVLU_MODEL_MANIFEST.json"
kairos_check_shared_storage
echo "== Phase-2 Job 2e (M2.4 InternVL-U) completed cleanly: ${RUN_DIR}"
