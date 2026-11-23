#!/usr/bin/env bash
# Phase-2 Job 2d (1x8 GPUs): Milestone M2.3 Suite B (Modality Skew 80:20/50:50/20:80 + GenEval/MMVet Lossless Parity).
set -euo pipefail

source /shared/data/MORPHSERVE/scripts/morphserve_common.sh

kairos_check_shared_storage

MODEL_DIR="${MODELS}/BAGEL-7B-MoT"
BENCH_DIR="${DATA}/benchmarks/interleaved-bench-v1"
if [ ! -f "${MODEL_DIR}/MODEL_MANIFEST.json" ] || [ ! -f "${BENCH_DIR}/BENCH_MANIFEST.json" ]; then
  echo "error: missing BAGEL-7B-MoT or Interleaved-Bench manifest" >&2
  exit 1
fi

_has_eval_weights() {
  local d="$1"
  [ -f "${d}/config.json" ] && {
    [ -f "${d}/model.safetensors" ] || [ -f "${d}/pytorch_model.bin" ] || \
    [ -f "${d}/model.safetensors.index.json" ] || [ -f "${d}/pytorch_model.bin.index.json" ]
  }
}

export MORPHSERVE_EVAL_MODELS_DIR="${DATA}/eval_models"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if ! _has_eval_weights "${MORPHSERVE_EVAL_MODELS_DIR}/owlv2-base-patch16-ensemble" || \
   ! _has_eval_weights "${MORPHSERVE_EVAL_MODELS_DIR}/clip-vit-large-patch14"; then
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
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
  rm -rf "$MORPHSERVE_LOCAL_TMP" /tmp/ms_m23_* /tmp/ms_m23b_* /tmp/morphserve_h2d_locks /dev/shm/ms_shm_* 2>/dev/null || true
  _kairos_cleanup
}
trap _cleanup EXIT

python3 -m morphserve.bench.run_phase2_bagel_main \
  --mode suite_b \
  --model-dir "${MODEL_DIR}" \
  --bench-root "${BENCH_DIR}" \
  --output-dir "${RUN_DIR}" \
  --num-gpus 8 \
  --allow-eval-download \
  ${MORPHSERVE_MAX_SESSIONS:+--max-sessions "${MORPHSERVE_MAX_SESSIONS}"}

kairos_check_shared_storage
echo "== Phase-2 Job 2d (M2.3 Suite B) completed cleanly: ${RUN_DIR}"
