# MorphServe container runtime wrapper built directly atop kairos-toolkit/runtime/kairos_common.sh.
# Keeps all ephemeral compiler/ML caches on container-local /tmp to protect /shared (/data) headroom.
set -euo pipefail

export KAIROS_PROJECT="${KAIROS_PROJECT:-MORPHSERVE}"
export KAIROS_REQUIRE_BUILD="${KAIROS_REQUIRE_BUILD:-1}"
export KAIROS_EXECUTION_IMAGE="${KAIROS_EXECUTION_IMAGE:-192.168.1.114:30500/jjjkairos/mllmserve:vllm-omni0.30.0-r2}"
export KAIROS_BUILD_CMD="${KAIROS_BUILD_CMD:-bash \"\$SRC_DIR/deploy/kairos/scripts/build_stage.sh\"}"

# Source shared kairos-toolkit runtime library (handles lazy build lock, /data gate, RUN_DIR, tee log).
source /shared/data/MORPHSERVE/scripts/kairos_common.sh

# Override all ephemeral caches to container-local /tmp (Decision 3: never bloat /data).
export MORPHSERVE_LOCAL_TMP="/tmp/morphserve-${KAIROS_RUN_ID}"
export HOME="${MORPHSERVE_LOCAL_TMP}/home"
export XDG_CACHE_HOME="${MORPHSERVE_LOCAL_TMP}/cache"
export HF_HOME="${MORPHSERVE_LOCAL_TMP}/cache/huggingface"
export TORCH_HOME="${MORPHSERVE_LOCAL_TMP}/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="${MORPHSERVE_LOCAL_TMP}/cache/torchinductor"
export TRITON_CACHE_DIR="${MORPHSERVE_LOCAL_TMP}/cache/triton"
export MPLCONFIGDIR="${MORPHSERVE_LOCAL_TMP}/cache/matplotlib"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"

export PYTHONPATH="${BUILD}/python:${BUILD}/src${PYTHONPATH:+:$PYTHONPATH}"
export MORPHSERVE_BIN_DIR="${BUILD}/bin"
export MORPHSERVE_LIB_DIR="${BUILD}/lib"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# If a lightweight project venv overlay exists on /shared/code/MORPHSERVE/venv, expose its site-packages.
if [ -d "${CODE}/venv" ]; then
  for sp in "${CODE}"/venv/lib/python*/site-packages; do
    if [ -d "$sp" ]; then
      export PYTHONPATH="${PYTHONPATH}:${sp}"
    fi
  done
fi

echo "== morphserve_common build=${KAIROS_BUILD:-none} bin=${MORPHSERVE_BIN_DIR} lib=${MORPHSERVE_LIB_DIR} local_tmp=${MORPHSERVE_LOCAL_TMP}"
