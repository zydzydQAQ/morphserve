#!/usr/bin/env bash
# Fast 1x1 GPU pre-flight smoke: verifies fixed MorphServeClusterEngine (preemption + lazy VMM + KV guard)
# AND InternVL-U transformers>=5 processor/bridge/decoder end-to-end before launching 8-GPU Wave-2/3 jobs.
set -euo pipefail

source /shared/data/MORPHSERVE/scripts/morphserve_common.sh

kairos_check_shared_storage

BAGEL_DIR="${MODELS}/BAGEL-7B-MoT"
BENCH_DIR="${DATA}/benchmarks/interleaved-bench-v1"
IVLU_DIR="${MODELS}/InternVL-U"
VENDOR_DIR="${DATA}/vendor/internvlu_pkg"

python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
  --output "${RUN_DIR}/runtime-facts.json" \
  --model-dir "bagel=${BAGEL_DIR}" \
  --model-dir "internvlu=${IVLU_DIR}"

_cleanup() {
  rm -rf "$MORPHSERVE_LOCAL_TMP" /tmp/ms_1g_smoke_* /tmp/ivlu_parity_*.png 2>/dev/null || true
  _kairos_cleanup
}
trap _cleanup EXIT

echo "== [1/2] 1-GPU InternVL-U End-to-End Smoke (VLM_TEXT + Bridge + Decoder + Parity)"
python3 -m morphserve.bench.validate_phase2_internvl_u \
  --model-dir "${IVLU_DIR}" \
  --vendor-dir "${VENDOR_DIR}" \
  --output-dir "${RUN_DIR}/internvlu_1gpu" \
  --num-gpus 1 \
  --smoke-1gpu

echo "== [2/2] 1-GPU MorphServeClusterEngine Preemption + Lazy Alpha + Watermark Smoke"
python3 - <<'PY' "${BAGEL_DIR}" "${BENCH_DIR}" "${RUN_DIR}/bagel_1gpu"
import json
from pathlib import Path
import sys

from morphserve.bench.interleaved_bench import InterleavedSession, InterleavedTurn, WorkloadSuite
from morphserve.engine import EnginePolicyConfig, MorphServeClusterEngine

bagel_dir = Path(sys.argv[1])
bench_dir = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)

# Construct a tight 2-session 1-GPU overlap trace: 1 multi-turn edit (t=0) + 1 mid-flight AR VQA (t=0.8s)
sessions = [
    InterleavedSession(
        session_id="smk_edit_0",
        quadrant="multiturn_edit",
        source_dataset="smoke",
        arrival_offset_s=0.0,
        turns=[
            InterleavedTurn(turn_idx=0, modality_plan="dit_t2i", prompt="A serene alpine cabin by a clear lake.", height=512, width=512, dit_steps=12),
            InterleavedTurn(turn_idx=1, modality_plan="dit_edit", prompt="Add golden autumn foliage along the shore.", height=512, width=512, dit_steps=12),
        ],
    ),
    InterleavedSession(
        session_id="smk_vqa_mid_dit",
        quadrant="multi_image_vqa",
        source_dataset="smoke",
        arrival_offset_s=0.8,
        turns=[
            InterleavedTurn(turn_idx=0, modality_plan="ar_only", prompt="State two visual properties of calm mountain lakes concisely.", max_ar_tokens=20),
        ],
    ),
]
suite = WorkloadSuite(name="smoke_1gpu_preempt", trace_source="synthetic", base_qps=1.0, sessions=sessions)
cfg = EnginePolicyConfig.from_preset("morphserve", num_gpus=1, initial_alphas={0: 0.50})
eng = MorphServeClusterEngine(policy=cfg, model_dir=bagel_dir, bench_root=bench_dir, tmp_root=Path("/tmp/ms_1g_smoke_eng"))
try:
    eng.start()
    rep = eng.run_suite(suite)
    d = rep.to_dict()
    (out_dir / "bagel-1gpu-smoke.json").write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    print(
        f"[1GPU-BAGEL-SMOKE] status={rep.status} oom={rep.oom_sessions} preempted={rep.preempted_ar_count} "
        f"yields={rep.total_yields} ar_ttft_p95={rep.ar_ttft_p95_ms}ms ar_tpot_p95={rep.ar_tpot_p95_ms}ms "
        f"goodput={rep.joint_slo_goodput_rps}",
        flush=True,
    )
    assert rep.status == "ok" and rep.oom_sessions == 0 and rep.completed_sessions == 2, f"smoke failed: {d}"
    assert rep.total_yields > 0 and rep.preempted_ar_count >= 1, f"expected mid-DiT preemption to fire: {d}"
finally:
    eng.shutdown()
PY

kairos_check_shared_storage
echo "== Phase-2 1-GPU Pre-Flight Smoke PASSED: ${RUN_DIR}"
