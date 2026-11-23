#!/usr/bin/env bash
# Phase-0 Job 0b (1x1 GPU): Download and SHA256-verify ByteDance-Seed/BAGEL-7B-MoT via hf-mirror.com.
set -euo pipefail

export KAIROS_OFFLINE_HF=0
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

source /shared/data/MORPHSERVE/scripts/morphserve_common.sh

# Ensure online mirror access inside this download job (override any offline flags).
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_ENABLE_HF_TRANSFER=0

MODEL_REPO="${MORPHSERVE_MODEL_REPO:-ByteDance-Seed/BAGEL-7B-MoT}"
MODEL_REV="${MORPHSERVE_MODEL_REVISION:-5019f57d168e5816e8f3f701b17cc816bb7cf24b}"
MODEL_DEST="${MODELS}/BAGEL-7B-MoT"

# Storage gate BEFORE download (require <= 760G / < 77% so +30G stays safely below 790G deploy gate!)
KAIROS_MAX_USED_GIB=760 KAIROS_MAX_PERCENT=77 kairos_check_shared_storage

_morphserve_job0b_cleanup() {
  chmod -R a+rX "$MODEL_DEST" 2>/dev/null || true
  rm -rf "$MORPHSERVE_LOCAL_TMP" 2>/dev/null || true
  _kairos_cleanup
}
trap _morphserve_job0b_cleanup EXIT

echo "== downloading ${MODEL_REPO}@${MODEL_REV} from ${HF_ENDPOINT} (parallel HTTP Range, 16 workers) -> ${MODEL_DEST}"
python3 -m morphserve.tools.fetch_hf \
  --repo-id "$MODEL_REPO" \
  --revision "$MODEL_REV" \
  --dest "$MODEL_DEST" \
  --endpoint "$HF_ENDPOINT" \
  --max-workers 16 \
  --chunk-mib 16

echo "== offline manifest re-verification"
python3 -m morphserve.tools.verify_model \
  --dest "$MODEL_DEST" \
  --revision "$MODEL_REV" | tee "${RUN_DIR}/verify-bagel.json"

cp "${MODEL_DEST}/MODEL_MANIFEST.json" "${RUN_DIR}/MODEL_MANIFEST.json"
kairos_check_shared_storage
df -h /shared
echo "== Phase-0 Job 0b completed cleanly: ${MODEL_DEST}"
