"""Phase-0 live probe for Kairos GPU containers: toolchain, topology, vllm-omni, and TriPool-VMM."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request


def run_cmd(args: list[str], timeout: int = 20) -> dict:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip()[:4000],
            "stderr": proc.stderr.strip()[:2000],
        }
    except FileNotFoundError:
        return {"ok": False, "error": "not_found"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def probe_host_and_numa() -> dict:
    meminfo = {}
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.is_file():
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            for key in ("MemTotal", "MemAvailable", "Shmem", "HugePages_Total", "Hugepagesize"):
                if line.startswith(key + ":"):
                    meminfo[key] = line.split(":", 1)[1].strip()

    shm_info = {}
    if Path("/dev/shm").exists():
        usage = shutil.disk_usage("/dev/shm")
        shm_info = {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "total_gib": round(usage.total / (1024**3), 3),
            "free_gib": round(usage.free / (1024**3), 3),
        }

    numa_online = None
    online_path = Path("/sys/devices/system/node/online")
    if online_path.is_file():
        numa_online = online_path.read_text(encoding="utf-8").strip()

    gpu_pci_numa = []
    smi = run_cmd(
        ["nvidia-smi", "--query-gpu=index,name,pci.bus_id,memory.total", "--format=csv,noheader"]
    )
    if smi.get("ok"):
        for line in smi["stdout"].splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                bdf = parts[2].lower()
                # Normalize 00000000:xx:yy.z -> 0000:xx:yy.z for sysfs
                if bdf.startswith("00000000:"):
                    bdf = bdf[4:]
                numa_file = Path(f"/sys/bus/pci/devices/{bdf}/numa_node")
                numa_node = (
                    int(numa_file.read_text(encoding="utf-8").strip())
                    if numa_file.is_file()
                    else None
                )
                gpu_pci_numa.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "pci_bus_id": parts[2],
                        "memory_total": parts[3],
                        "numa_node": numa_node,
                    }
                )

    return {
        "hostname": platform.node(),
        "kernel": platform.release(),
        "cpu_count": os.cpu_count(),
        "meminfo": meminfo,
        "dev_shm": shm_info,
        "numa_online": numa_online,
        "numactl_hardware": run_cmd(["numactl", "--hardware"]),
        "nvidia_smi_topo": run_cmd(["nvidia-smi", "topo", "-m"]),
        "gpu_pci_numa": gpu_pci_numa,
    }


def probe_toolchain() -> dict:
    tools = ("g++", "gcc", "cmake", "ninja", "nvcc", "nsys", "ncu", "numactl", "nvbandwidth")
    bins = {}
    for tool in tools:
        path = shutil.which(tool)
        if not path:
            bins[tool] = {"present": False}
            continue
        ver_flag = "--version" if tool not in ("numactl",) else "--show"
        ver = run_cmd([path, ver_flag], timeout=10)
        first_line = (ver.get("stdout") or ver.get("stderr") or "").splitlines()[:2]
        bins[tool] = {"present": True, "path": path, "version_head": first_line}

    headers_libs = {
        "/usr/local/cuda/include/cuda.h": Path("/usr/local/cuda/include/cuda.h").is_file(),
        "/usr/include/cuda.h": Path("/usr/include/cuda.h").is_file(),
        "libcuda.so.1": _can_dlopen("libcuda.so.1"),
        "libnuma.so.1": _can_dlopen("libnuma.so.1"),
    }
    return {"binaries": bins, "headers_and_libs": headers_libs}


def _can_dlopen(name: str) -> bool:
    try:
        ctypes.CDLL(name)
        return True
    except OSError:
        return False


def probe_python_and_vllm() -> dict:
    pkg_names = (
        "torch",
        "triton",
        "flashinfer",
        "flashinfer-python",
        "flash-attn",
        "vllm",
        "vllm-omni",
        "transformers",
        "diffusers",
        "accelerate",
        "safetensors",
        "huggingface-hub",
        "pillow",
        "pyzmq",
        "scipy",
        "numpy",
    )
    packages = {}
    for name in pkg_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    torch_info = {}
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "nccl_version": list(torch.cuda.nccl.version()) if torch.cuda.is_available() else None,
            "has_CUDAPluggableAllocator": hasattr(torch.cuda, "CUDAPluggableAllocator"),
            "has_MemPool": hasattr(torch.cuda, "MemPool"),
            "has_use_mem_pool": hasattr(torch.cuda, "use_mem_pool"),
            "has_change_current_allocator": hasattr(torch.cuda.memory, "change_current_allocator"),
        }
        if hasattr(torch.cuda, "MemPool"):
            torch_info["MemPool_sig"] = str(inspect.signature(torch.cuda.MemPool))
        if hasattr(torch.cuda, "use_mem_pool"):
            torch_info["use_mem_pool_sig"] = str(inspect.signature(torch.cuda.use_mem_pool))
    except Exception as exc:
        torch_info["error"] = f"{type(exc).__name__}: {exc}"

    vllm_omni_info = {}
    try:
        import vllm_omni

        root = Path(vllm_omni.__file__).resolve().parent
        bagel_files = sorted(
            str(p.relative_to(root)) for p in root.rglob("*bagel*") if p.is_file()
        )
        internvl_files = sorted(
            str(p.relative_to(root)) for p in root.rglob("*internvl*") if p.is_file()
        )
        stage_configs = sorted(
            str(p.relative_to(root)) for p in root.rglob("*bagel*.yaml") if p.is_file()
        )
        vllm_omni_info = {
            "root": str(root),
            "bagel_files": bagel_files[:60],
            "internvl_files": internvl_files[:40],
            "bagel_stage_configs": stage_configs[:20],
        }
        for mod_name, cls_name in (
            ("vllm_omni.model_executor.models.bagel.bagel", "OmniBagelForConditionalGeneration"),
            ("vllm_omni.diffusion.models.bagel.bagel_transformer", "NaiveCache"),
            ("vllm_omni.diffusion.models.bagel.pipeline_bagel", "BagelPipeline"),
        ):
            try:
                mod = importlib.import_module(mod_name)
                vllm_omni_info[f"class:{cls_name}"] = hasattr(mod, cls_name)
            except Exception as exc:
                vllm_omni_info[f"class:{cls_name}"] = f"import_error: {type(exc).__name__}: {exc}"
    except Exception as exc:
        vllm_omni_info["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "python_version": sys.version,
        "packages": packages,
        "torch": torch_info,
        "vllm_omni": vllm_omni_info,
    }


def probe_outbound_hf() -> dict:
    urls = {
        "hf_bagel": "https://huggingface.co/api/models/ByteDance-Seed/BAGEL-7B-MoT",
        "hf_mirror_bagel": "https://hf-mirror.com/api/models/ByteDance-Seed/BAGEL-7B-MoT",
        "hf_internvl_u": "https://huggingface.co/api/models/InternVL-U/InternVL-U",
        "hf_mirror_internvl_u": "https://hf-mirror.com/api/models/InternVL-U/InternVL-U",
    }
    results = {}
    for label, url in urls.items():
        t0 = time.perf_counter()
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "morphserve-probe/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                results[label] = {
                    "reachable": True,
                    "status": resp.status,
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                }
        except urllib.error.HTTPError as err:
            results[label] = {
                "reachable": True,
                "status": err.code,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            }
        except Exception as exc:
            results[label] = {
                "reachable": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            }
    return results


def run_cpp_vmm_probe(bin_path: Path, out_json: Path) -> dict:
    if not bin_path.is_file():
        return {"status": "missing_binary", "path": str(bin_path)}
    proc = subprocess.run(
        [
            str(bin_path),
            "--device",
            "0",
            "--spike-gib",
            "4",
            "--layer-mib",
            "512",
            "--json-out",
            str(out_json),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if out_json.is_file():
        data = json.loads(out_json.read_text(encoding="utf-8"))
        data["returncode"] = proc.returncode
        return data
    return {
        "status": "exec_failed",
        "returncode": proc.returncode,
        "stdout": proc.stdout[:2000],
        "stderr": proc.stderr[:2000],
    }


def _tensor_sha256(tensor) -> str:
    import torch

    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def run_pytorch_mempool_probe(so_path: Path, handle_mib: int = 128, repeats: int = 5) -> dict:
    if not so_path.is_file():
        return {"status": "missing_library", "path": str(so_path)}
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        return {"status": "missing_torch", "error": str(exc)}

    if not torch.cuda.is_available():
        return {"status": "no_cuda"}
    if not (hasattr(torch.cuda, "CUDAPluggableAllocator") and hasattr(torch.cuda, "MemPool")):
        return {"status": "unsupported_pytorch_api"}

    from morphserve.vmm import TriPoolVMM

    device = torch.device("cuda:0")
    try:
        vmm = TriPoolVMM(
            device=0,
            super_pool_mib=6144,
            fine_kv_pool_mib=64,
            super_handle_mib=handle_mib,
            transient_va_gib=16,
            kv_va_gib=2,
            allow_overflow=False,
            so_path=so_path,
            simulated=False,
            init_torch_cuda=True,
        )
    except Exception as exc:
        return {"status": "init_failed", "error": f"{type(exc).__name__}: {exc}"}

    try:
        torch.manual_seed(2026)
        torch.cuda.manual_seed_all(2026)

        conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, dtype=torch.bfloat16, device=device).eval()
        gn1 = nn.GroupNorm(32, 256, dtype=torch.bfloat16, device=device).eval()
        conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1, dtype=torch.bfloat16, device=device).eval()
        conv3 = nn.Conv2d(128, 3, kernel_size=3, padding=1, dtype=torch.bfloat16, device=device).eval()
        x_in = torch.randn(2, 256, 256, 256, dtype=torch.bfloat16, device=device)

        def vae_like_forward(x):
            h = F.interpolate(x, scale_factor=2.0, mode="nearest")  # [2, 256, 512, 512] = 256 MiB
            h = gn1(h)
            h = F.silu(h)
            h1 = conv1(h)
            h2 = h + h1
            h3 = F.silu(conv2(h2))
            # Force a 2.0 GiB transient buffer matching high-res FLUX VAE conv peaks
            scratch = torch.empty((16, 256, 512, 512), dtype=torch.bfloat16, device=device)
            scratch[:2].copy_(h2)
            out = conv3(h3) + scratch[:2, :3] * 0.0
            del h, h1, h2, h3, scratch
            return out

        with torch.inference_mode():
            # Warmup cuDNN kernels on default allocator
            _ = vae_like_forward(x_in)
            torch.cuda.synchronize()

            ref_times = []
            out_ref = None
            for _ in range(repeats):
                t0 = time.perf_counter()
                out_ref = vae_like_forward(x_in)
                torch.cuda.synchronize()
                ref_times.append((time.perf_counter() - t0) * 1000.0)
            ref_sha = _tensor_sha256(out_ref)

            st_before = vmm.stats()

            vmm_times = []
            out_vmm = None
            all_reclaimed_each = []
            for _ in range(repeats + 1):  # 1 warmup + repeats measured
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out_vmm = vmm.run_in_transient_scope(vae_like_forward, x_in)
                torch.cuda.synchronize()
                dt_ms = (time.perf_counter() - t0) * 1000.0
                st_iter = vmm.stats()
                all_reclaimed_each.append(st_iter.transient_mapped_handles_current == 0)
                vmm_times.append(dt_ms)

            vmm_cold_ms = vmm_times[0]
            vmm_steady_times = vmm_times[1:]

            gc.collect()
            torch.cuda.synchronize()

            st_after = vmm.stats()

            vmm_sha = _tensor_sha256(out_vmm)
            bit_exact = bool(torch.equal(out_ref, out_vmm) and ref_sha == vmm_sha)
            max_abs_diff = float((out_ref.float() - out_vmm.float()).abs().max().item())

        ref_p50 = sorted(ref_times)[len(ref_times) // 2]
        vmm_p50 = sorted(vmm_steady_times)[len(vmm_steady_times) // 2]
        all_reclaimed = st_after.transient_mapped_handles_current == 0 and all(all_reclaimed_each)
        return {
            "status": "ok" if (bit_exact and all_reclaimed) else "check_failed",
            "handle_mib": handle_mib,
            "bit_exact": bit_exact,
            "ref_sha256": ref_sha,
            "vmm_sha256": vmm_sha,
            "max_abs_diff": max_abs_diff,
            "ref_p50_ms": round(ref_p50, 3),
            "vmm_cold_ms": round(vmm_cold_ms, 3),
            "vmm_steady_p50_ms": round(vmm_p50, 3),
            "vmm_overhead_p50_ms": round(vmm_p50 - ref_p50, 3),
            "ref_times_ms": [round(x, 3) for x in ref_times],
            "vmm_steady_times_ms": [round(x, 3) for x in vmm_steady_times],
            "peak_transient_mib": round(
                (st_after.transient_mapped_handles_peak * st_after.super_handle_bytes) / (1024**2), 2
            ),
            "all_pages_reclaimed": all_reclaimed,
            "zero_hotpath_overflow": st_after.overflow_creates == 0,
            "stats_before": st_before.to_dict(),
            "stats_after": st_after.to_dict(),
        }
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        vmm.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MorphServe Phase-0 live environment & VMM probe")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bin-dir", type=Path, default=Path(os.environ.get("MORPHSERVE_BIN_DIR", ".")))
    parser.add_argument("--lib-dir", type=Path, default=Path(os.environ.get("MORPHSERVE_LIB_DIR", ".")))
    parser.add_argument("--skip-gpu-bench", action="store_true")
    parser.add_argument("--skip-network", action="store_true")
    args = parser.parse_args(argv)

    build_status_path = args.bin_dir.parent / "VMM_BUILD_STATUS.json"
    vmm_build_status = None
    if build_status_path.is_file():
        vmm_build_status = json.loads(build_status_path.read_text(encoding="utf-8"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "morphserve.phase0.probe.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vmm_build_status": vmm_build_status,
        "host_and_numa": probe_host_and_numa(),
        "toolchain": probe_toolchain(),
        "python_and_vllm": probe_python_and_vllm(),
    }
    if not args.skip_network:
        report["outbound_hf"] = probe_outbound_hf()
    if not args.skip_gpu_bench:
        if vmm_build_status is not None and not vmm_build_status.get("compiled", True):
            report["cpp_vmm_probe"] = {"status": "skipped_not_compiled", "detail": vmm_build_status}
            report["pytorch_mempool_probe"] = {"status": "skipped_not_compiled", "detail": vmm_build_status}
        else:
            report["cpp_vmm_probe"] = run_cpp_vmm_probe(
                args.bin_dir / "tripool_vmm_probe",
                args.output_dir / "vmm-bench.json",
            )
            report["pytorch_mempool_probe"] = run_pytorch_mempool_probe(
                args.lib_dir / "libtripool_pluggable.so"
            )

    out_path = args.output_dir / "probe-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)

    # Print concise executive summary to stdout for job logs
    sweep = report.get("cpp_vmm_probe", {}).get("granularity_sweep") or []
    sweep_by_mib = {item.get("handle_mib"): item for item in sweep}
    summary = {
        "vmm_compiled": None if vmm_build_status is None else vmm_build_status.get("compiled"),
        "vmm_cpp_status": report.get("cpp_vmm_probe", {}).get("status"),
        "single_2mb_remap_p50_us": report.get("cpp_vmm_probe", {}).get("single_2mb_remap_us", {}).get("p50"),
        "vae_4gib_2mib_p50_ms": sweep_by_mib.get(2, {}).get("vae_4gib_acquire_ms", {}).get("p50"),
        "vae_4gib_64mib_p50_ms": sweep_by_mib.get(64, {}).get("vae_4gib_acquire_ms", {}).get("p50"),
        "vae_4gib_128mib_p50_ms": sweep_by_mib.get(128, {}).get("vae_4gib_acquire_ms", {}).get("p50"),
        "vae_4gib_128mib_unmap_p50_ms": sweep_by_mib.get(128, {}).get("vae_4gib_bulk_unmap_only_ms", {}).get("p50"),
        "h2d_128mib_remap_p50_ms": report.get("cpp_vmm_probe", {}).get("h2d_super_remap_ms", {}).get("p50"),
        "h2d_stream_p50_gbps": report.get("cpp_vmm_probe", {}).get("h2d_stream_gbps", {}).get("p50"),
        "mempool_status": report.get("pytorch_mempool_probe", {}).get("status"),
        "mempool_bit_exact": report.get("pytorch_mempool_probe", {}).get("bit_exact"),
        "mempool_all_reclaimed": report.get("pytorch_mempool_probe", {}).get("all_pages_reclaimed"),
        "mempool_ref_p50_ms": report.get("pytorch_mempool_probe", {}).get("ref_p50_ms"),
        "mempool_vmm_steady_p50_ms": report.get("pytorch_mempool_probe", {}).get("vmm_steady_p50_ms"),
        "mempool_overhead_p50_ms": report.get("pytorch_mempool_probe", {}).get("vmm_overhead_p50_ms"),
        "mempool_peak_mib": report.get("pytorch_mempool_probe", {}).get("peak_transient_mib"),
    }
    print("PHASE0_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)

    if not args.skip_gpu_bench and (vmm_build_status is None or vmm_build_status.get("compiled", True)):
        if report["cpp_vmm_probe"].get("status") != "ok":
            return 2
        if report["pytorch_mempool_probe"].get("status") != "ok":
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
