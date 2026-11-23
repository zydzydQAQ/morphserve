#!/usr/bin/env python3
"""Offline cryptographic verification of a downloaded model directory against MODEL_MANIFEST.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CHUNK_SIZE = 4 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_dir(dest: Path, expected_revision: str | None = None) -> dict:
    manifest_path = dest / "MODEL_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing MODEL_MANIFEST.json in {dest}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected_revision and manifest.get("resolved_revision") != expected_revision:
        raise RuntimeError(
            f"revision mismatch: manifest={manifest.get('resolved_revision')} expected={expected_revision}"
        )
    total = 0
    for item in manifest["files"]:
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"unsafe path in manifest: {item['path']!r}")
        p = dest / rel
        if not p.is_file():
            raise RuntimeError(f"missing file: {p}")
        if p.stat().st_size != item["bytes"]:
            raise RuntimeError(f"size mismatch: {p}")
        actual_sha = sha256_file(p)
        if actual_sha != item["sha256"]:
            raise RuntimeError(f"SHA256 mismatch for {p}: {actual_sha} != {item['sha256']}")
        if item.get("lfs_sha256") and actual_sha != item["lfs_sha256"]:
            raise RuntimeError(f"LFS SHA256 mismatch for {p}")
        total += item["bytes"]
    if total != manifest["total_bytes"]:
        raise RuntimeError("total bytes mismatch")
    return {
        "status": "verified",
        "repo_id": manifest.get("repo_id"),
        "revision": manifest.get("resolved_revision"),
        "files": len(manifest["files"]),
        "total_bytes": total,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args(argv)
    res = verify_model_dir(args.dest, args.revision)
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
