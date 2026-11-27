"""Slow deploy, packaging, shell syntax, Kairos job schema, and CMake/build_stage parity contracts."""

from __future__ import annotations

import json
from pathlib import Path
import py_compile
import re
import subprocess

import pytest

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "deploy" / "kairos" / "scripts"
JOBS_DIR = REPO_ROOT / "deploy" / "kairos" / "jobs"
_SRC_CPP_CAND = REPO_ROOT / "src" / "morphserve"
_STAGE_CPP_CAND = REPO_ROOT / "cpp_src" / "morphserve"
CPP_SRC_DIR = _SRC_CPP_CAND if any(_SRC_CPP_CAND.rglob("*.cpp")) else _STAGE_CPP_CAND
CPP_TESTS_DIR = REPO_ROOT / "tests" / "cpp"
CMAKE_PATH = REPO_ROOT / "CMakeLists.txt"
BUILD_STAGE_PATH = SCRIPTS_DIR / "build_stage.sh"
STAGE1_GATE_PATH = SCRIPTS_DIR / "16_osdi27_stage1_1gpu_gate.sh"

EXPECTED_CPP_CTEST_TARGETS = (
    "test_tripool_vmm",
    "test_runtime_transport",
    "test_control_engine",
    "test_cluster_simulator",
    "test_c_api",
)


def test_all_kairos_scripts_pass_bash_syntax_and_clean_cli_contracts() -> None:
    scripts = sorted(SCRIPTS_DIR.glob("*.sh"))
    assert len(scripts) >= 15

    forbidden_tokens = (
        "--low-sessions",
        "--knee-sessions",
        "_compute_sustainable_qps_summary",
    )
    for script in scripts:
        proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"bash -n failed for {script.name}: {proc.stderr}"
        body = script.read_text(encoding="utf-8")
        for tok in forbidden_tokens:
            assert tok not in body, f"stale token {tok!r} still present in {script.name}"


def test_kairos_job_specs_dns1035_and_schema_compliance() -> None:
    job_files = sorted(JOBS_DIR.glob("*.json"))
    assert len(job_files) >= 10

    dns_re = re.compile(r"^[a-z][a-z0-9-]{0,50}[a-z0-9]$")
    script_re = re.compile(r"/shared/data/MORPHSERVE/scripts/([^\s\"']+\.sh)")
    available_scripts = {p.name for p in SCRIPTS_DIR.glob("*.sh")}

    for jf in job_files:
        raw = jf.read_text(encoding="utf-8")
        assert "localhost:30800" not in raw, f"use 127.0.0.1 instead of localhost in {jf.name}"
        spec = json.loads(raw)

        for req_key in ("name", "image", "workers", "gpus_per_worker", "mount_shared", "command", "env"):
            assert req_key in spec, f"{jf.name} missing required key {req_key!r}"

        name = str(spec["name"])
        assert len(name) <= 52 and dns_re.fullmatch(name), f"invalid DNS-1035 job name {name!r} in {jf.name}"
        assert spec["mount_shared"] is True, f"{jf.name} must enable mount_shared"
        assert int(spec["workers"]) >= 1
        assert int(spec["gpus_per_worker"]) >= 1

        cmd = str(spec["command"])
        m = script_re.search(cmd)
        assert m is not None, f"{jf.name} command missing /shared/data/MORPHSERVE/scripts/*.sh target: {cmd!r}"
        script_basename = Path(m.group(1)).name
        assert script_basename in available_scripts, (
            f"{jf.name} references missing script {script_basename!r} (command={cmd!r})"
        )


def test_cmake_and_build_stage_cpp_source_list_parity() -> None:
    disk_srcs = {p.relative_to(CPP_SRC_DIR).as_posix() for p in CPP_SRC_DIR.rglob("*.cpp")}
    assert disk_srcs, "expected C++17 sources under src/morphserve/"

    cmake_txt = CMAKE_PATH.read_text(encoding="utf-8")
    m_cmake = re.search(r"add_library\s*\(\s*morphserve_core\s+SHARED\s+(.*?)\)", cmake_txt, flags=re.S)
    assert m_cmake is not None, "could not locate add_library(morphserve_core SHARED ...) in CMakeLists.txt"
    cmake_srcs = {
        tok.removeprefix("src/morphserve/")
        for tok in re.findall(r"src/morphserve/[^\s)\"]+\.cpp", m_cmake.group(1))
    }

    bs_txt = BUILD_STAGE_PATH.read_text(encoding="utf-8")
    m_bs = re.search(r"CORE_SRCS=\((.*?)\)", bs_txt, flags=re.S)
    assert m_bs is not None, "could not locate CORE_SRCS=(...) in build_stage.sh"
    bs_srcs = set(re.findall(r'"([^"]+\.cpp)"', m_bs.group(1)))

    assert cmake_srcs == disk_srcs, (
        f"CMakeLists.txt out of sync with src/morphserve/: "
        f"missing={sorted(disk_srcs - cmake_srcs)} extra={sorted(cmake_srcs - disk_srcs)}"
    )
    assert bs_srcs == disk_srcs, (
        f"build_stage.sh CORE_SRCS out of sync with src/morphserve/: "
        f"missing={sorted(disk_srcs - bs_srcs)} extra={sorted(bs_srcs - disk_srcs)}"
    )


def test_stage1_gate_covers_all_5_cpp_ctest_targets() -> None:
    cmake_txt = CMAKE_PATH.read_text(encoding="utf-8")
    gate_txt = STAGE1_GATE_PATH.read_text(encoding="utf-8")
    assert "enable_testing()" in cmake_txt and "add_test(" in cmake_txt

    for target in EXPECTED_CPP_CTEST_TARGETS:
        assert (CPP_TESTS_DIR / f"{target}.cpp").is_file(), f"missing tests/cpp/{target}.cpp"
        assert re.search(rf"\b{re.escape(target)}\b", cmake_txt), (
            f"CMakeLists.txt missing CTest target {target!r}"
        )
        assert target in gate_txt, f"16_osdi27_stage1_1gpu_gate.sh missing CTest target {target!r}"
        assert f"[PASS] {target}" in gate_txt, f"16_osdi27_stage1_1gpu_gate.sh missing pass banner '[PASS] {target}'"

    for legacy_target in ("test_vmm_and_runtime", "test_control_and_sim"):
        assert legacy_target not in cmake_txt, f"legacy CTest target {legacy_target!r} still in CMakeLists.txt"
        assert legacy_target not in gate_txt, f"legacy CTest target {legacy_target!r} still referenced in Stage-1 gate"


def test_bench_and_plot_python_modules_compile_cleanly(tmp_path: Path) -> None:
    py_targets = sorted((REPO_ROOT / "python" / "morphserve" / "bench").glob("*.py"))
    py_targets.append(REPO_ROOT / "scripts" / "plot_osdi27_slo_and_oom_cdf.py")
    assert py_targets
    for idx, src in enumerate(py_targets):
        py_compile.compile(str(src), cfile=str(tmp_path / f"mod_{idx}.pyc"), doraise=True)
