#!/usr/bin/env bash
# Phase-0 Job 0d (1x2 GPU): BAGEL-7B-MoT weight layout, 1-GPU VAE spike/DiT timing, and 2-GPU vLLM-Omni KV stall profiling.
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

_morphserve_job0d_cleanup() {
  if kill -0 "$SAMPLER_PID" 2>/dev/null; then
    kill -TERM "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
  fi
  rm -rf "$MORPHSERVE_LOCAL_TMP" 2>/dev/null || true
  _kairos_cleanup
}
trap _morphserve_job0d_cleanup EXIT

# Run single_gpu and two_stage in separate Python processes so GPU 0 HBM is 100% freed before vllm serve launches!
python3 -m morphserve.bench.motivation_profile \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${RUN_DIR}/part1_single" \
  --modes weights single_gpu \
  --steps 20

python3 -m morphserve.bench.motivation_profile \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${RUN_DIR}/part2_twostage" \
  --deploy-config "${BUILD}/configs/bagel_2gpu_baseline.yaml" \
  --modes two_stage \
  --steps 20

# Merge both parts into unified motivation-report.json
python3 - "${RUN_DIR}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
p1 = json.loads((root / "part1_single/motivation-report.json").read_text(encoding="utf-8"))
p2 = json.loads((root / "part2_twostage/motivation-report.json").read_text(encoding="utf-8"))
p1["two_stage"] = p2.get("two_stage")
(root / "motivation-report.json").write_text(json.dumps(p1, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("merged unified motivation-report.json", flush=True)
PY

kairos_check_shared_storage
echo "== Phase-0 Job 0d completed cleanly: ${RUN_DIR}"
