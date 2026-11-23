"""Stage compact (< 30 MiB) benchmark source files for Interleaved-Bench into data/bench_sources/."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

DEFAULT_SOURCES_DIR = Path(__file__).resolve().parents[3] / "data/bench_sources"
DEFAULT_MLLMSERVE_DIR = Path(__file__).resolve().parents[4] / "mllmserve"

GENEVAL_URL = "https://raw.githubusercontent.com/djghosh13/geneval/main/prompts/evaluation_metadata.jsonl"
BURSTGPT_URL = "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_bytes(url: str, max_bytes: int | None = None, timeout: float = 30.0) -> bytes:
    headers = {"User-Agent": "MorphServe-BenchStager/1.0"}
    if max_bytes is not None:
        headers["Range"] = f"bytes=0-{int(max_bytes) - 1}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if max_bytes is not None and len(data) >= max_bytes:
        nl = data.rfind(b"\n")
        if nl > 0:
            data = data[: nl + 1]
    return data


def stage_sources(
    output_dir: Path = DEFAULT_SOURCES_DIR,
    mllmserve_dir: Path = DEFAULT_MLLMSERVE_DIR,
    burstgpt_max_bytes: int = 4 * 1024 * 1024,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy local MSE-Bench images + manifest + session templates + GenTD26 trace
    mse_src = mllmserve_dir / "data/session-benchmark/mse-bench"
    mse_dst = output_dir / "mse-bench"
    if not mse_src.is_dir():
        raise FileNotFoundError(f"missing local mse-bench directory: {mse_src}")
    if mse_dst.exists():
        shutil.rmtree(mse_dst)
    shutil.copytree(mse_src, mse_dst)

    templates_src = mllmserve_dir / "workloads/mse_editing_sessions.json"
    if not templates_src.is_file():
        raise FileNotFoundError(f"missing local mse_editing_sessions.json: {templates_src}")
    shutil.copy2(templates_src, output_dir / "mse_editing_sessions.json")

    gentd_src = mllmserve_dir / "data/session-benchmark/gentd26/lora_request_trace.csv"
    if not gentd_src.is_file():
        raise FileNotFoundError(f"missing local gentd26 trace: {gentd_src}")
    shutil.copy2(gentd_src, output_dir / "gentd26_lora_request_trace.csv")

    # 2. Download GenEval evaluation_metadata.jsonl (~85 KB)
    geneval_dst = output_dir / "geneval_evaluation_metadata.jsonl"
    if not geneval_dst.is_file() or geneval_dst.stat().st_size == 0:
        geneval_bytes = _download_bytes(GENEVAL_URL, max_bytes=None)
        geneval_dst.write_bytes(geneval_bytes)

    # 3. Download complete-line 4 MiB prefix of BurstGPT_1.csv (~80k production requests)
    burst_dst = output_dir / "burstgpt_1_sample.csv"
    if not burst_dst.is_file() or burst_dst.stat().st_size == 0:
        burst_bytes = _download_bytes(BURSTGPT_URL, max_bytes=burstgpt_max_bytes)
        burst_dst.write_bytes(burst_bytes)

    files_meta = []
    total_bytes = 0
    for p in sorted(output_dir.rglob("*")):
        if not p.is_file() or p.name == "SOURCES_MANIFEST.json":
            continue
        sz = p.stat().st_size
        total_bytes += sz
        files_meta.append(
            {
                "path": str(p.relative_to(output_dir)),
                "bytes": sz,
                "sha256": _sha256_file(p),
            }
        )

    manifest = {
        "schema": "morphserve.bench_sources.v1",
        "total_bytes": total_bytes,
        "total_mib": round(total_bytes / (1024**2), 2),
        "file_count": len(files_meta),
        "files": files_meta,
    }
    (output_dir / "SOURCES_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage compact Interleaved-Bench source files locally")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SOURCES_DIR)
    parser.add_argument("--mllmserve-dir", type=Path, default=DEFAULT_MLLMSERVE_DIR)
    parser.add_argument("--burstgpt-max-bytes", type=int, default=4 * 1024 * 1024)
    args = parser.parse_args(argv)
    manifest = stage_sources(
        output_dir=args.output_dir,
        mllmserve_dir=args.mllmserve_dir,
        burstgpt_max_bytes=args.burstgpt_max_bytes,
    )
    print(json.dumps({"status": "ok", "total_mib": manifest["total_mib"], "files": manifest["file_count"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
