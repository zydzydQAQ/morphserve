#!/usr/bin/env bash
# Build `libmorphserve_core.so`, `morphserve-sim`, and `tripool_vmm_probe`, and stage Python + artifacts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SRC="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SRC_DIR="${SRC_DIR:-${DEFAULT_SRC}}"
BUILD_TMP="${BUILD_TMP:-${SRC_DIR}/build/stage_cmake}"
INSTALL_TMP="${INSTALL_TMP:-${SRC_DIR}/build/install}"

mkdir -p "${BUILD_TMP}" "${INSTALL_TMP}/bin" "${INSTALL_TMP}/lib"

CXX_BIN="${CXX:-g++}"
if ! command -v "${CXX_BIN}" >/dev/null 2>&1; then
  echo "error: C++17 compiler '${CXX_BIN}' not found" >&2
  exit 1
fi
CXX_REAL="$(command -v "${CXX_BIN}")"

# Compile in container-local /tmp object dir to avoid NTFS/NFS sub-second clock-skew warnings from gmake
OBJ_DIR="$(mktemp -d "${TMPDIR:-/tmp}/morphserve_build_objs.XXXXXX")"
cleanup_objs() {
  rm -rf "${OBJ_DIR}"
}
trap cleanup_objs EXIT

INCLUDES=(
  -I"${SRC_DIR}/include"
  -I"${SRC_DIR}/deploy/kairos/deps/nlohmann3.11.3"
)
CXXFLAGS=(-std=c++17 -O2 -Wall -Wextra -Wpedantic -fPIC -pthread)
CORE_SRCS=(
  "runtime/cuda_driver.cpp"
  "runtime/numa_mirror.cpp"
  "runtime/h2d_pacer.cpp"
  "runtime/peer_transport.cpp"
  "vmm/page_backend.cpp"
  "vmm/tripool_vmm.cpp"
  "control/hash_util.cpp"
  "control/topology.cpp"
  "control/policy.cpp"
  "control/budget_planner.cpp"
  "control/gpu_queue_scheduler.cpp"
  "control/performance_atlas.cpp"
  "control/preempt_edf.cpp"
  "control/router.cpp"
  "control/morph_controller.cpp"
  "control/sim_memory_ledger.cpp"
  "control/sim_trace_loader.cpp"
  "control/sim_calibration.cpp"
  "control/simulator.cpp"
  "c_api.cpp"
)

if [ -z "${MAX_JOBS:-}" ]; then
  if grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
    MAX_JOBS=1
    echo "note: WSL detected; defaulting MAX_JOBS=1 to avoid WSL OOM (prefer Kairos in-container build via deploy_build.sh --mode stage; override with MAX_JOBS=N)" >&2
  else
    MAX_JOBS="$(nproc 2>/dev/null || echo 4)"
  fi
fi
OBJS=()
PIDS=()
ACTIVE=0
for rel in "${CORE_SRCS[@]}"; do
  obj_name="${rel//\//_}.o"
  obj_path="${OBJ_DIR}/${obj_name}"
  OBJS+=("${obj_path}")
  "${CXX_REAL}" "${CXXFLAGS[@]}" "${INCLUDES[@]}" -c "${SRC_DIR}/src/morphserve/${rel}" -o "${obj_path}" &
  PIDS+=("$!")
  ACTIVE=$((ACTIVE + 1))
  if [ "${ACTIVE}" -ge "${MAX_JOBS}" ]; then
    wait "${PIDS[0]}"
    PIDS=("${PIDS[@]:1}")
    ACTIVE=$((ACTIVE - 1))
  fi
done
for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

"${CXX_REAL}" "${CXXFLAGS[@]}" -shared "${OBJS[@]}" -o "${INSTALL_TMP}/lib/libmorphserve_core.so" -ldl
ln -sfn "libmorphserve_core.so" "${INSTALL_TMP}/lib/libtripool_vmm.so"
ln -sfn "libmorphserve_core.so" "${INSTALL_TMP}/lib/libtripool_pluggable.so"

"${CXX_REAL}" "${CXXFLAGS[@]}" "${INCLUDES[@]}" \
  "${SRC_DIR}/cmd/morphserve_sim.cpp" \
  -L"${INSTALL_TMP}/lib" -Wl,-rpath,'$ORIGIN/../lib' -lmorphserve_core -ldl \
  -o "${INSTALL_TMP}/bin/morphserve-sim" &
PID_SIM=$!

"${CXX_REAL}" "${CXXFLAGS[@]}" "${INCLUDES[@]}" \
  "${SRC_DIR}/cmd/tripool_vmm_probe.cpp" \
  -L"${INSTALL_TMP}/lib" -Wl,-rpath,'$ORIGIN/../lib' -lmorphserve_core -ldl \
  -o "${INSTALL_TMP}/bin/tripool_vmm_probe" &
PID_PROBE=$!

wait "${PID_SIM}"
wait "${PID_PROBE}"

# Stage Python data plane (`python/morphserve/`) + compatibility `src -> python` symlink
rm -rf "${INSTALL_TMP}/python" "${INSTALL_TMP}/src" "${INSTALL_TMP}/cpp_src"
cp -R "${SRC_DIR}/python" "${INSTALL_TMP}/python"
find "${INSTALL_TMP}/python" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "${INSTALL_TMP}/python" \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true
ln -sfn "python" "${INSTALL_TMP}/src"
cp -R "${SRC_DIR}/src" "${INSTALL_TMP}/cpp_src"

# Stage C++17 headers/entrypoints, tests, deploy scripts, helper scripts, docs, and configs
for sub in include cmd configs docs deploy scripts tests; do
  if [ -d "${SRC_DIR}/${sub}" ]; then
    rm -rf "${INSTALL_TMP}/${sub}"
    cp -R "${SRC_DIR}/${sub}" "${INSTALL_TMP}/${sub}"
    find "${INSTALL_TMP}/${sub}" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
  fi
done
chmod +x "${INSTALL_TMP}"/deploy/kairos/scripts/*.sh 2>/dev/null || true

for top_doc in CMakeLists.txt README.md pyproject.toml GLOSSARY.md; do
  if [ -f "${SRC_DIR}/${top_doc}" ]; then
    cp -f "${SRC_DIR}/${top_doc}" "${INSTALL_TMP}/${top_doc}"
  fi
done

if [ -d "${SRC_DIR}/results" ]; then
  rm -rf "${INSTALL_TMP}/results"
  mkdir -p "${INSTALL_TMP}/results"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a \
      --exclude='*.log' \
      --exclude='*.nsys-rep' \
      --exclude='*.sqlite' \
      --exclude='vllm_omni_ref' \
      --exclude='raw' \
      "${SRC_DIR}/results/" "${INSTALL_TMP}/results/"
  else
    cp -R "${SRC_DIR}/results/." "${INSTALL_TMP}/results/"
    find "${INSTALL_TMP}/results" \( -name "*.log" -o -name "*.nsys-rep" -o -name "*.sqlite" \) -delete 2>/dev/null || true
    rm -rf "${INSTALL_TMP}/results/vllm_omni_ref" "${INSTALL_TMP}/results/raw" 2>/dev/null || true
  fi
fi

cat <<EOF > "${INSTALL_TMP}/VMM_BUILD_STATUS.json"
{
  "status": "ok",
  "compiled": true,
  "cxx": "${CXX_REAL}",
  "core_lib": "lib/libmorphserve_core.so",
  "sim_bin": "bin/morphserve-sim",
  "probe_bin": "bin/tripool_vmm_probe"
}
EOF
