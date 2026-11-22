#!/usr/bin/env bash
# Lazy in-cluster / local build script for MorphServe TriPool-VMM C++17 artifacts.
# Expects SRC_DIR, BUILD_TMP, INSTALL_TMP (provided by kairos_common.sh --mode stage),
# or defaults to building inside the repo when run directly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SRC_DIR="${SRC_DIR:-$REPO_ROOT}"
BUILD_TMP="${BUILD_TMP:-$REPO_ROOT/build/vmm_tripool}"
INSTALL_TMP="${INSTALL_TMP:-$REPO_ROOT/build/install}"

mkdir -p "$BUILD_TMP" "$INSTALL_TMP/bin" "$INSTALL_TMP/lib"

CXX="${CXX:-}"
if [ -z "$CXX" ]; then
  for cand in g++ clang++ c++; do
    if command -v "$cand" >/dev/null 2>&1; then
      CXX="$cand"
      break
    fi
  done
fi

if [ -n "$CXX" ] && command -v "$CXX" >/dev/null 2>&1; then
  CUDA_INC_FLAGS=()
  HAS_CUDA_H=false
  for inc_dir in /usr/local/cuda/include /usr/include; do
    if [ -f "$inc_dir/cuda.h" ]; then
      CUDA_INC_FLAGS=("-I$inc_dir")
      HAS_CUDA_H=true
      echo "== found cuda.h in $inc_dir"
      break
    fi
  done
  if [ "${#CUDA_INC_FLAGS[@]}" -eq 0 ]; then
    echo "== cuda.h not present; compiling with built-in CUDA Driver ABI fallback (dlopen libcuda.so.1)"
  fi

  VMM_SRC="$SRC_DIR/csrc/vmm_tripool"
  COMMON_FLAGS=(-O2 -std=c++17 -Wall -Wextra -pthread "${CUDA_INC_FLAGS[@]}")

  echo "== compiling tripool_vmm_probe with $CXX"
  "$CXX" "${COMMON_FLAGS[@]}" \
    "$VMM_SRC/tripool_vmm_probe.cpp" \
    -o "$INSTALL_TMP/bin/tripool_vmm_probe" \
    -ldl

  echo "== compiling libtripool_pluggable.so with $CXX"
  "$CXX" "${COMMON_FLAGS[@]}" -shared -fPIC \
    "$VMM_SRC/tripool_pluggable_allocator.cpp" \
    -o "$INSTALL_TMP/lib/libtripool_pluggable.so" \
    -ldl

  echo "== compiling libtripool_vmm.so with $CXX"
  "$CXX" "${COMMON_FLAGS[@]}" -shared -fPIC \
    "$VMM_SRC/tripool_vmm.cpp" \
    -o "$INSTALL_TMP/lib/libtripool_vmm.so" \
    -ldl

  cat > "$INSTALL_TMP/VMM_BUILD_STATUS.json" <<EOF
{"compiled": true, "compiler": "$CXX", "has_cuda_h": $HAS_CUDA_H}
EOF
else
  echo "== WARN: no C++ compiler (g++/clang++/c++) found in PATH; installing Python sources so probe can report full toolchain" >&2
  cat > "$INSTALL_TMP/VMM_BUILD_STATUS.json" <<EOF
{"compiled": false, "compiler": null, "reason": "no_cxx_compiler_in_image"}
EOF
fi

echo "== installing Python package, configs, tests, deploy scripts, and baseline results into $INSTALL_TMP"
for subdir in src configs docs csrc tests deploy results; do
  if [ -d "$SRC_DIR/$subdir" ]; then
    rm -rf "$INSTALL_TMP/$subdir"
    tar -C "$SRC_DIR" \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      --exclude='*.pyo' \
      --exclude='*.log' \
      --exclude='*.nsys-rep' \
      --exclude='*.sqlite' \
      --exclude='vllm_omni_ref' \
      --exclude='results/raw' \
      -cf - "$subdir" | tar -C "$INSTALL_TMP" -xf -
  fi
done
for topfile in README.md pyproject.toml GLOSSARY.md; do
  if [ -f "$SRC_DIR/$topfile" ]; then
    cp "$SRC_DIR/$topfile" "$INSTALL_TMP/$topfile"
  fi
done

echo "== build complete: $(cat "$INSTALL_TMP/VMM_BUILD_STATUS.json")"
