"""Offline unit tests for MorphServe Phase-0 scaffolding, C++ compilation, and Kairos job specs."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
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
                ["bash", str(REPO_ROOT / "csrc/vmm_tripool/build.sh")],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                proc.returncode,
                0,
                msg=f"build.sh failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
            )
            self.assertTrue((tmp / "install/bin/tripool_vmm_probe").is_file())
            self.assertTrue((tmp / "install/lib/libtripool_pluggable.so").is_file())
            self.assertTrue((tmp / "install/src/morphserve/probe.py").is_file())

    def test_shell_scripts_syntax(self) -> None:
        scripts = [
            REPO_ROOT / "csrc/vmm_tripool/build.sh",
            REPO_ROOT / "deploy/kairos/scripts/morphserve_common.sh",
            REPO_ROOT / "deploy/kairos/scripts/00_phase0_probe_vmm.sh",
        ]
        for script in scripts:
            proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=f"bash -n failed for {script}: {proc.stderr}")


if __name__ == "__main__":
    unittest.main()
