#!/usr/bin/env bash
# OSDI '27 Stage-1 Unified 1x1 A100 Hard-Gate:
#   (a) /shared < 80% + kairos_check_shared_storage gate
#   (b) Stage google/owlv2-base-patch16-ensemble & openai/clip-vit-large-patch14 (~1.3 GiB) into ${DATA}/eval_models
#   (c) Live A100 container pytest suite -> ${RUN_DIR}/pytest-a100.txt
#   (d) Module 1 (TriPool-VMM + BAGEL) live 1xA100 validation -> ${RUN_DIR}/module1
#   (e) Module 3 (FlashPrefix-CFG + CoW + GPU Loopback + OWLv2/CLIP) -> ${RUN_DIR}/module3
#   (f) Module 4 (Layer-Chunk Preemption + Batched AR + M1+M3+M4 Stack) -> ${RUN_DIR}/module4
#   (g) 1-GPU InternVL-U smoke + 3-session (1 edit @ t=0 + 2 concurrent AR VQA @ t=0.8s) BAGEL cluster smoke
#       + consolidated ${RUN_DIR}/stage1-gate-summary.json
set -euo pipefail

# Enforce Stage-1 < 80% /shared cap even during any lazy build inside morphserve_common.sh.
export KAIROS_MAX_PERCENT="${KAIROS_MAX_PERCENT:-80}"

source /shared/data/MORPHSERVE/scripts/morphserve_common.sh

_enforce_shared_under_80pct() {
  local shared_root="${KAIROS_SHARED_ROOT:-/shared}"
  local used_pct
  used_pct=$(df -P "${shared_root}" | awk 'NR==2 {for (i=1; i<=NF; i++) if ($i ~ /^[0-9]+%$/) {sub("%", "", $i); print $i; exit}}')
  if [ -z "${used_pct}" ] || [ "${used_pct}" -ge 80 ]; then
    echo "error: ${shared_root} usage ${used_pct:-unknown}% >= 80% Stage-1 gate; stopping immediately" >&2
    exit 1
  fi
  kairos_check_shared_storage
}

# (a) Pre-flight /shared < 80% + Kairos storage stop-line gate
_enforce_shared_under_80pct

BAGEL_DIR="${MODELS}/BAGEL-7B-MoT"
IVLU_DIR="${MODELS}/InternVL-U"
VENDOR_DIR="${DATA}/vendor/internvlu_pkg"
BENCH_DIR="${DATA}/benchmarks/interleaved-bench-v1"
EVAL_MODELS_DIR="${DATA}/eval_models"

if [ ! -f "${BAGEL_DIR}/MODEL_MANIFEST.json" ]; then
  echo "error: verified BAGEL-7B-MoT not found at ${BAGEL_DIR}" >&2
  exit 1
fi
if [ ! -f "${IVLU_DIR}/MODEL_MANIFEST.json" ]; then
  echo "error: verified InternVL-U not found at ${IVLU_DIR}" >&2
  exit 1
fi
if [ ! -d "${VENDOR_DIR}/internvlu" ]; then
  echo "error: vendored internvlu package missing at ${VENDOR_DIR}" >&2
  exit 1
fi
if [ ! -f "${BENCH_DIR}/BENCH_MANIFEST.json" ]; then
  echo "error: Interleaved-Bench manifest missing at ${BENCH_DIR}" >&2
  exit 1
fi

python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
  --output "${RUN_DIR}/runtime-facts.json" \
  --model-dir "bagel=${BAGEL_DIR}" \
  --model-dir "internvlu=${IVLU_DIR}"

python3 /shared/data/MORPHSERVE/scripts/kairos_runtime_audit.py \
  --memory-summary "${RUN_DIR}/memory-summary.json" &
SAMPLER_PID=$!

_stop_sampler() {
  if [ -n "${SAMPLER_PID:-}" ] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then
    kill -TERM "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
  fi
  SAMPLER_PID=""
}

_cleanup() {
  _stop_sampler
  rm -rf "$MORPHSERVE_LOCAL_TMP" /tmp/ms_1g_smoke_* /tmp/ms_ivlu_* /tmp/ivlu_parity_* /tmp/morphserve_h2d_locks /dev/shm/ms_shm_* 2>/dev/null || true
  _kairos_cleanup
}
trap _cleanup EXIT

# (b) Stage OWLv2 & CLIP evaluator models (~1.3 GiB total) via hf-mirror.com into ${DATA}/eval_models if missing
echo "== [Stage 1 / Step b] Verifying / staging OWLv2 & CLIP evaluator weights in ${EVAL_MODELS_DIR}"
mkdir -p "${EVAL_MODELS_DIR}"
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  HF_HUB_ENABLE_HF_TRANSFER=0 \
  python3 - "${EVAL_MODELS_DIR}" "${MORPHSERVE_LOCAL_TMP}/stage_eval" <<'PY'
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

eval_root = Path(sys.argv[1])
stage_root = Path(sys.argv[2])
endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
ignore_pats = ["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"]
repos = (
    "google/owlv2-base-patch16-ensemble",
    "openai/clip-vit-large-patch14",
)
weight_files = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def has_weights(cand: Path) -> bool:
    return (
        cand.is_dir()
        and (cand / "config.json").is_file()
        and any((cand / fn).is_file() for fn in weight_files)
    )


def check_shared_pct() -> None:
    shared_root = os.environ.get("KAIROS_SHARED_ROOT", "/shared")
    lines = [ln for ln in subprocess.check_output(["df", "-P", shared_root], text=True).splitlines() if ln.strip()]
    pct_tok = next((tok for tok in lines[-1].split() if tok.endswith("%") and tok[:-1].isdigit()), None)
    if pct_tok is None:
        raise RuntimeError(f"failed to parse df -P usage percent for {shared_root}: {lines[-1] if lines else ''}")
    pct = int(pct_tok[:-1])
    if pct >= 80:
        raise RuntimeError(f"{shared_root} usage {pct}% >= 80% during eval_models staging")


eval_root.mkdir(parents=True, exist_ok=True)
for repo_id in repos:
    name = repo_id.split("/")[-1]
    dest = eval_root / name
    if has_weights(dest):
        print(f"[eval_models] already staged: {dest}", flush=True)
        continue

    check_shared_pct()
    from huggingface_hub import snapshot_download

    local_stage = stage_root / name
    shutil.rmtree(local_stage, ignore_errors=True)
    local_stage.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(1, 4):
        try:
            print(f"[eval_models] downloading {repo_id} via {endpoint} (attempt {attempt}/3)", flush=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_stage),
                endpoint=endpoint,
                ignore_patterns=ignore_pats,
            )
            shutil.rmtree(local_stage / ".cache", ignore_errors=True)
            has_proc = (local_stage / "preprocessor_config.json").is_file() or (
                local_stage / "processor_config.json"
            ).is_file()
            if not (has_weights(local_stage) and has_proc):
                raise RuntimeError(f"downloaded {repo_id} missing config.json, processor config, or weights")
            break
        except Exception as exc:
            last_err = exc
            time.sleep(2.0 * attempt)
    else:
        raise RuntimeError(f"failed to stage {repo_id}: {last_err}")

    check_shared_pct()
    nfs_tmp = eval_root / f".tmp-{name}-{os.getpid()}"
    shutil.rmtree(nfs_tmp, ignore_errors=True)
    shutil.copytree(local_stage, nfs_tmp)
    subprocess.run(["chmod", "-R", "a+rX", str(nfs_tmp)], check=True)
    if not dest.exists():
        os.rename(nfs_tmp, dest)
    else:
        shutil.rmtree(nfs_tmp, ignore_errors=True)
    shutil.rmtree(local_stage, ignore_errors=True)
    print(f"[eval_models] staged and verified: {dest}", flush=True)
PY

_enforce_shared_under_80pct
export MORPHSERVE_EVAL_MODELS_DIR="${EVAL_MODELS_DIR}"

# (c) Run full pytest suite on the live A100 container -> ${RUN_DIR}/pytest-a100.txt
# Stage a clean container-local workspace under ${MORPHSERVE_LOCAL_TMP} so pytest never mutates
# immutable/read-only ${BUILD} on /shared (e.g. build/pytest_tmp or __pycache__) and works across
# both --mode stage (.src-${KAIROS_BUILD}/tests) and --mode direct (${BUILD}/tests).
echo "== [Stage 1 / Step c] Running pytest suite on live A100 container"
if [ -z "${BUILD:-}" ] || [ ! -d "${BUILD}" ]; then
  echo "error: BUILD directory is unset or missing (${BUILD:-<empty>})" >&2
  exit 1
fi
PYTEST_WS="${MORPHSERVE_LOCAL_TMP}/pytest_ws"
SRC_STAGE="${CODE}/builds/.src-${KAIROS_BUILD:-}"
rm -rf "${PYTEST_WS}"
mkdir -p "${PYTEST_WS}/data"
for entry in "${BUILD}"/*; do
  [ -e "${entry}" ] || continue
  bname="$(basename "${entry}")"
  case "${bname}" in
    tests|data|build|src|cpp_src) continue ;;
  esac
  ln -sfn "${entry}" "${PYTEST_WS}/${bname}"
done
if [ -d "${SRC_STAGE}/src" ]; then
  ln -sfn "${SRC_STAGE}/src" "${PYTEST_WS}/src"
elif [ -d "${BUILD}/cpp_src" ]; then
  ln -sfn "${BUILD}/cpp_src" "${PYTEST_WS}/src"
fi
if [ -d "${SRC_STAGE}" ]; then
  for extra in include cmd CMakeLists.txt deploy results README.md pyproject.toml GLOSSARY.md; do
    if [ ! -e "${PYTEST_WS}/${extra}" ] && [ -e "${SRC_STAGE}/${extra}" ]; then
      ln -sfn "${SRC_STAGE}/${extra}" "${PYTEST_WS}/${extra}"
    fi
  done
fi
ln -sfn "${BENCH_DIR}" "${PYTEST_WS}/data/interleaved-bench-v1"
if [ -d "${BUILD}/tests" ]; then
  cp -R "${BUILD}/tests" "${PYTEST_WS}/tests"
elif [ -d "${SRC_STAGE}/tests" ]; then
  cp -R "${SRC_STAGE}/tests" "${PYTEST_WS}/tests"
else
  echo "error: tests directory not found in ${BUILD}/tests or ${SRC_STAGE}/tests" >&2
  exit 1
fi
chmod -R u+rwX "${PYTEST_WS}/tests"
if python3 -c "import pytest" >/dev/null 2>&1; then
  PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -v \
    --rootdir="${PYTEST_WS}" \
    -o "cache_dir=${MORPHSERVE_LOCAL_TMP}/pytest_cache" \
    "${PYTEST_WS}/tests" 2>&1 | tee "${RUN_DIR}/pytest-a100.txt"
else
  echo "== [Stage 1 / Step c] pytest not installed; falling back to python3 -m unittest discover"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PYTEST_WS}/python${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m unittest discover -s "${PYTEST_WS}/tests/python" -p "test_*.py" -v \
    2>&1 | tee "${RUN_DIR}/pytest-a100.txt"
fi

# (d) Module 1 (TriPool-VMM + BAGEL) live A100 validation
echo "== [Stage 1 / Step d] Validating Module 1 (TriPool-VMM + BAGEL)"
python3 -m morphserve.bench.validate_module1_bagel \
  --model-dir "${BAGEL_DIR}" \
  --output-dir "${RUN_DIR}/module1" \
  --steps 20

# (e) Module 3 (FlashPrefix-CFG + LECP CoW + GPU Loopback + OWLv2/CLIP) live A100 validation
echo "== [Stage 1 / Step e] Validating Module 3 (Shared-Prefix CFG + CoW + GPU Loopback)"
python3 -m morphserve.bench.validate_module3_cfg_loopback \
  --model-dir "${BAGEL_DIR}" \
  --output-dir "${RUN_DIR}/module3" \
  --steps 20 \
  --allow-eval-download

# (f) Module 4 (Layer-Chunk Preemptive AR+DiT Scheduler + Batched AR + M1+M3+M4 Stack)
echo "== [Stage 1 / Step f] Validating Module 4 (Layer-Chunk Preemption + Batched AR)"
python3 -m morphserve.bench.validate_module4_preemption \
  --model-dir "${BAGEL_DIR}" \
  --output-dir "${RUN_DIR}/module4" \
  --steps 20

# (g.1) 1-GPU InternVL-U end-to-end smoke (VLM_TEXT + Bridge + Decoder + 12 GiB quota + Parity)
echo "== [Stage 1 / Step g.1] Running 1-GPU InternVL-U end-to-end smoke"
python3 -m morphserve.bench.validate_phase2_internvl_u \
  --model-dir "${IVLU_DIR}" \
  --vendor-dir "${VENDOR_DIR}" \
  --output-dir "${RUN_DIR}/internvlu_1gpu" \
  --num-gpus 1 \
  --smoke-1gpu

# (g.2) 3-session 1-GPU MorphServeClusterEngine smoke:
#       1 multi-turn edit at t=0 + 2 concurrent AR VQA sessions at t=0.8s
echo "== [Stage 1 / Step g.2] Running 3-session 1-GPU BAGEL MorphServeClusterEngine concurrency & preemption smoke"
SMOKE_PY="${MORPHSERVE_LOCAL_TMP}/ms_stage1_1g_smoke_driver.py"
cat <<'PY' > "${SMOKE_PY}"
import json
from pathlib import Path
import sys

from morphserve.bench.interleaved_bench import InterleavedSession, InterleavedTurn, WorkloadSuite
from morphserve.engine import EnginePolicyConfig, MorphServeClusterEngine


def main() -> None:
    bagel_dir = Path(sys.argv[1])
    bench_dir = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    tmp_eng_dir = Path(sys.argv[4])
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = [
        InterleavedSession(
            session_id="smk_edit_0",
            quadrant="multiturn_edit",
            source_dataset="stage1_smoke",
            arrival_offset_s=0.0,
            turns=[
                InterleavedTurn(
                    turn_idx=0,
                    modality_plan="dit_t2i",
                    prompt="A serene alpine cabin by a clear mountain lake.",
                    height=512,
                    width=512,
                    dit_steps=14,
                ),
                InterleavedTurn(
                    turn_idx=1,
                    modality_plan="dit_edit",
                    prompt="Add golden autumn foliage along the rocky lakeshore.",
                    height=512,
                    width=512,
                    dit_steps=12,
                ),
            ],
        ),
        InterleavedSession(
            session_id="smk_vqa_mid_dit_0",
            quadrant="multi_image_vqa",
            source_dataset="stage1_smoke",
            arrival_offset_s=0.8,
            turns=[
                InterleavedTurn(
                    turn_idx=0,
                    modality_plan="ar_only",
                    prompt="State two visual properties of calm alpine lakes concisely.",
                    max_ar_tokens=20,
                ),
            ],
        ),
        InterleavedSession(
            session_id="smk_vqa_mid_dit_1",
            quadrant="multi_image_vqa",
            source_dataset="stage1_smoke",
            arrival_offset_s=0.8,
            turns=[
                InterleavedTurn(
                    turn_idx=0,
                    modality_plan="ar_only",
                    prompt="Name two seasonal colors typical of autumn mountain forests.",
                    max_ar_tokens=20,
                ),
            ],
        ),
    ]
    suite = WorkloadSuite(
        name="stage1_1gpu_3sess_preempt",
        trace_source="synthetic",
        base_qps=1.5,
        sessions=sessions,
    )
    cfg = EnginePolicyConfig.from_preset(
        "morphserve",
        num_gpus=1,
        inter_turn_think_s_range=(0.2, 0.5),
    )
    eng = MorphServeClusterEngine(
        policy=cfg,
        model_dir=bagel_dir,
        bench_root=bench_dir,
        tmp_root=tmp_eng_dir,
    )
    try:
        eng.start()
        rep = eng.run_suite(suite)
        d = rep.to_dict()
        (out_dir / "bagel-1gpu-smoke.json").write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
        print(
            f"[STAGE1-1GPU-BAGEL-SMOKE] status={rep.status} completed={rep.completed_sessions}/{rep.total_sessions} "
            f"oom={rep.oom_sessions} preempted={rep.preempted_ar_count} yields={rep.total_yields} "
            f"ar_ttft_p95={rep.ar_ttft_p95_ms}ms ar_tpot_p95={rep.ar_tpot_p95_ms}ms "
            f"peak_phys_mib={rep.peak_gpu_reserved_mib} peak_pt_mib={rep.peak_pytorch_reserved_mib}",
            flush=True,
        )
        assert rep.status == "ok", f"expected status=='ok': {d}"
        assert rep.completed_sessions == 3 and rep.error_sessions == 0, f"expected 3/3 completed: {d}"
        assert rep.oom_sessions == 0, f"expected oom_sessions==0: {d}"
        assert rep.preempted_ar_count >= 1, f"expected preempted_ar_count>=1: {d}"
        assert rep.total_yields > 0, f"expected total_yields>0: {d}"
        assert rep.peak_gpu_reserved_mib >= rep.peak_pytorch_reserved_mib, f"expected combined >= pytorch HBM: {d}"
    finally:
        eng.shutdown()


if __name__ == "__main__":
    main()
PY
python3 "${SMOKE_PY}" "${BAGEL_DIR}" "${BENCH_DIR}" "${RUN_DIR}/bagel_1gpu" "${MORPHSERVE_LOCAL_TMP}/ms_1g_smoke_eng"
rm -f "${SMOKE_PY}"

_stop_sampler
_enforce_shared_under_80pct

# Consolidate Stage-1 gate results into ${RUN_DIR}/stage1-gate-summary.json
python3 - "${RUN_DIR}" "${EVAL_MODELS_DIR}" <<'PY'
import json
import os
from pathlib import Path
import sys
import time

run_dir = Path(sys.argv[1])
eval_dir = Path(sys.argv[2])

import re

m1 = json.loads((run_dir / "module1/module1-report.json").read_text(encoding="utf-8"))
m3 = json.loads((run_dir / "module3/module3-report.json").read_text(encoding="utf-8"))
m4 = json.loads((run_dir / "module4/module4-report.json").read_text(encoding="utf-8"))
ivlu = json.loads((run_dir / "internvlu_1gpu/m24-internvlu-report.json").read_text(encoding="utf-8"))
bagel_smk = json.loads((run_dir / "bagel_1gpu/bagel-1gpu-smoke.json").read_text(encoding="utf-8"))
pytest_raw = (run_dir / "pytest-a100.txt").read_text(encoding="utf-8", errors="replace")
pytest_txt = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", pytest_raw)
nonempty_lines = [ln.strip() for ln in pytest_txt.splitlines() if ln.strip()]
pytest_tail = nonempty_lines[-5:]
tail_window = nonempty_lines[-10:]
pytest_last = pytest_tail[-1] if pytest_tail else ""
pytest_banner_re = re.compile(
    r"^(?:=+\s+)?.*\b[1-9][0-9]*\s+passed\b.*\bin\s+[0-9]+(?:\.[0-9]+)?s\b.*(?:\s+=+)?$"
)
pytest_summary_ok = bool(
    (
        re.search(r"\b[1-9][0-9]*\s+passed\b", pytest_last)
        or any(pytest_banner_re.match(ln) for ln in tail_window)
    )
    and not any(re.search(r"\b[1-9][0-9]*\s+(failed|error|errors)\b", ln) for ln in tail_window)
    and not re.search(r"(?m)^=+\s*(?:FAILURES|ERRORS)\s*=+", pytest_txt)
    and not re.search(r"(?m)^FAILED\b", pytest_txt)
)
ran_indices = [i for i, ln in enumerate(tail_window) if re.match(r"^Ran\s+[0-9]+\s+tests?\b", ln)]
last_ran_idx = ran_indices[-1] if ran_indices else -1
unittest_ok_line = tail_window[last_ran_idx + 1] if (0 <= last_ran_idx and last_ran_idx + 1 < len(tail_window)) else ""
unittest_summary_ok = bool(
    last_ran_idx >= 0
    and re.match(r"^Ran\s+[1-9][0-9]*\s+tests?\b", tail_window[last_ran_idx])
    and not re.search(r"(?m)^Ran\s+0\s+tests\b", "\n".join(tail_window[last_ran_idx:]))
    and re.match(r"^OK(\s*\([^)]*\))?\s*$", unittest_ok_line)
    and "FAILED" not in unittest_ok_line
    and not re.search(r"(?m)^FAILED\b", pytest_txt)
)
pytest_passed = bool((run_dir / "pytest-a100.txt").is_file() and (pytest_summary_ok or unittest_summary_ok))

weight_files = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def has_eval_weights(cand: Path) -> bool:
    return (
        cand.is_dir()
        and (cand / "config.json").is_file()
        and any((cand / fn).is_file() for fn in weight_files)
    )


eval_staged = {
    "owlv2-base-patch16-ensemble": has_eval_weights(eval_dir / "owlv2-base-patch16-ensemble"),
    "clip-vit-large-patch14": has_eval_weights(eval_dir / "clip-vit-large-patch14"),
}

m3_t4 = m3.get("test4_visual_cot_and_downstream")
m3_eval_ok = True
if isinstance(m3_t4, dict):
    m3_eval_ok = (
        m3_t4.get("owlv2_backend") not in (None, "unavailable_offline")
        and m3_t4.get("clip_backend") not in (None, "unavailable_offline")
    )

gates = {
    "pytest_a100": {
        "status": "ok" if pytest_passed else "check_failed",
        "log_path": str(run_dir / "pytest-a100.txt"),
        "tail": pytest_tail,
    },
    "module1_vmm_bagel": {
        "status": m1.get("status", "check_failed"),
        "report_path": str(run_dir / "module1/module1-report.json"),
    },
    "module3_cfg_loopback": {
        "status": "ok" if (m3.get("status") == "ok" and m3_eval_ok) else "check_failed",
        "owlv2_backend": m3_t4.get("owlv2_backend") if isinstance(m3_t4, dict) else None,
        "clip_backend": m3_t4.get("clip_backend") if isinstance(m3_t4, dict) else None,
        "report_path": str(run_dir / "module3/module3-report.json"),
    },
    "module4_preemption": {
        "status": m4.get("status", "check_failed"),
        "report_path": str(run_dir / "module4/module4-report.json"),
    },
    "internvlu_1gpu_smoke": {
        "status": ivlu.get("status", "check_failed"),
        "morphserve_12gib_quota_verified": ivlu.get("morphserve_12gib_quota_verified"),
        "report_path": str(run_dir / "internvlu_1gpu/m24-internvlu-report.json"),
    },
    "bagel_1gpu_cluster_smoke": {
        "status": (
            "ok"
            if (
                bagel_smk.get("status") == "ok"
                and int(bagel_smk.get("completed_sessions", 0)) == 3
                and int(bagel_smk.get("oom_sessions", -1)) == 0
                and int(bagel_smk.get("preempted_ar_count", 0)) >= 1
                and int(bagel_smk.get("total_yields", 0)) > 0
            )
            else "check_failed"
        ),
        "completed_sessions": bagel_smk.get("completed_sessions"),
        "oom_sessions": bagel_smk.get("oom_sessions"),
        "preempted_ar_count": bagel_smk.get("preempted_ar_count"),
        "total_yields": bagel_smk.get("total_yields"),
        "peak_gpu_reserved_mib": bagel_smk.get("peak_gpu_reserved_mib"),
        "peak_pytorch_reserved_mib": bagel_smk.get("peak_pytorch_reserved_mib"),
        "report_path": str(run_dir / "bagel_1gpu/bagel-1gpu-smoke.json"),
    },
}

all_ok = all(eval_staged.values()) and all(g.get("status") == "ok" for g in gates.values())
summary = {
    "schema": "morphserve.osdi27.stage1_1gpu_gate.v1",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "kairos_run_id": os.environ.get("KAIROS_RUN_ID", ""),
    "kairos_build": os.environ.get("KAIROS_BUILD", ""),
    "eval_models_dir": str(eval_dir),
    "eval_models_staged": eval_staged,
    "gates": gates,
    "status": "ok" if all_ok else "check_failed",
}
out_path = run_dir / "stage1-gate-summary.json"
out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {out_path} (status={summary['status']})", flush=True)
if not all_ok:
    raise SystemExit(2)
PY

echo "== OSDI '27 Stage-1 Unified 1x1 A100 Hard-Gate PASSED: ${RUN_DIR}/stage1-gate-summary.json"
