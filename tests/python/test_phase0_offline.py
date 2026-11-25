"""Offline unit tests for MorphServe Phase-0 scaffolding, C++17 compilation, and Kairos job specs."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "python"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from morphserve import probe  # noqa: E402


class TestPhase0Offline(unittest.TestCase):
    def test_probe_offline_host_and_toolchain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            rc = probe.main(
                [
                    "--output-dir",
                    str(out_dir),
                    "--skip-gpu-bench",
                    "--skip-network",
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads((out_dir / "probe-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], "morphserve.phase0.probe.v1")
            self.assertIn("host_and_numa", report)
            self.assertIn("toolchain", report)
            self.assertIn("python_and_vllm", report)

    def test_cpp_vmm_builds_cleanly_if_gxx_present(self) -> None:
        if not shutil.which("g++"):
            self.skipTest("g++ not installed in local WSL environment")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            env = {
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "SRC_DIR": str(REPO_ROOT),
                "BUILD_TMP": str(tmp / "build"),
                "INSTALL_TMP": str(tmp / "install"),
            }
            proc = subprocess.run(
                ["bash", str(REPO_ROOT / "deploy/kairos/scripts/build_stage.sh")],
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(
                proc.returncode,
                0,
                msg=f"build_stage.sh failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
            )
            self.assertTrue((tmp / "install/bin/tripool_vmm_probe").is_file())
            self.assertTrue((tmp / "install/bin/morphserve-sim").is_file())
            self.assertTrue((tmp / "install/lib/libmorphserve_core.so").is_file())
            self.assertTrue((tmp / "install/lib/libtripool_pluggable.so").is_file())
            self.assertTrue((tmp / "install/python/morphserve/probe.py").is_file())
            self.assertTrue((tmp / "install/src/morphserve/probe.py").is_file())

            cxx = shutil.which("g++")
            assert cxx is not None
            inc_args = [
                f"-I{REPO_ROOT / 'include'}",
                f"-I{REPO_ROOT / 'deploy/kairos/deps/nlohmann3.11.3'}",
            ]
            lib_dir = tmp / "install/lib"
            link_args = [f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}", "-lmorphserve_core", "-ldl", "-pthread"]
            for cpp_name in ("test_vmm_and_runtime", "test_control_and_sim"):
                cpp_src = REPO_ROOT / "tests/cpp" / f"{cpp_name}.cpp"
                if not cpp_src.is_file():
                    continue
                if cpp_name == "test_control_and_sim" and not (
                    (REPO_ROOT / "results/phase0-job0c").is_dir()
                    and (REPO_ROOT / "data/interleaved-bench-v1").is_dir()
                ):
                    continue
                exe_path = tmp / cpp_name
                comp = subprocess.run(
                    [
                        cxx,
                        "-std=c++17",
                        "-O2",
                        "-Wall",
                        "-Wextra",
                        "-Wpedantic",
                        *inc_args,
                        str(cpp_src),
                        *link_args,
                        "-o",
                        str(exe_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(comp.returncode, 0, msg=f"{cpp_name} compile failed:\n{comp.stderr}")
                run_t = subprocess.run(
                    [str(exe_path)],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(
                    run_t.returncode,
                    0,
                    msg=f"{cpp_name} failed:\nSTDOUT:\n{run_t.stdout}\nSTDERR:\n{run_t.stderr}",
                )

    def test_shell_scripts_syntax(self) -> None:
        scripts = [
            REPO_ROOT / "deploy/kairos/scripts/build_stage.sh",
            REPO_ROOT / "deploy/kairos/scripts/morphserve_common.sh",
            REPO_ROOT / "deploy/kairos/scripts/00_phase0_probe_vmm.sh",
            REPO_ROOT / "deploy/kairos/scripts/01_phase0_download_bagel.sh",
            REPO_ROOT / "deploy/kairos/scripts/02_phase0_topo_8gpu.sh",
            REPO_ROOT / "deploy/kairos/scripts/03_phase0_motivation.sh",
        ]
        for script in scripts:
            proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=f"bash -n failed for {script}: {proc.stderr}")

    def test_phase0_modules_import(self) -> None:
        from morphserve.bench import motivation_profile, topo_probe
        from morphserve.models import bagel_adapter
        from morphserve.tools import fetch_hf, verify_model

        self.assertTrue(callable(topo_probe.main))
        self.assertTrue(callable(motivation_profile.main))
        self.assertTrue(callable(bagel_adapter.apply_vllm_omni_mm_limit))
        self.assertTrue(callable(fetch_hf.main))
        self.assertTrue(callable(verify_model.main))


if __name__ == "__main__":
    unittest.main()
