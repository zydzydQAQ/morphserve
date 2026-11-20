#!/usr/bin/env python3
"""Download and SHA256-verify a pinned HuggingFace model snapshot (supports HF_ENDPOINT mirrors)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

CHUNK_SIZE = 4 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "morphserve-fetch/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def safe_destination(root: Path, filename: str) -> Path:
    rel = Path(filename)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe model filename: {filename!r}")
    dest = root.joinpath(rel)
    if dest.resolve() != root.resolve() and root.resolve() not in dest.resolve().parents:
        raise ValueError(f"model path escapes destination: {filename!r}")
    return dest


def fetch_tree(endpoint: str, repo_id: str, revision: str) -> tuple[str, list[dict]]:
    base = endpoint.rstrip("/")
    model_info = get_json(f"{base}/api/models/{repo_id}/revision/{urllib.parse.quote(revision, safe='')}")
    resolved_rev = model_info.get("sha") or revision

    files: list[dict] = []

    def walk(subpath: str = "") -> None:
        suffix = f"/{urllib.parse.quote(subpath, safe='/')}" if subpath else ""
        url = f"{base}/api/models/{repo_id}/tree/{resolved_rev}{suffix}"
        entries = get_json(url)
        for item in entries:
            if item.get("type") == "directory":
                walk(item["path"])
            elif item.get("type") == "file":
                files.append(item)

    walk()
    files.sort(key=lambda x: x["path"])
    return resolved_rev, files


def download_via_hub(endpoint: str, repo_id: str, revision: str, dest: Path, max_workers: int) -> None:
    os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from huggingface_hub import snapshot_download

    t0 = time.perf_counter()
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(dest),
        max_workers=max_workers,
    )
    elapsed = time.perf_counter() - t0
    print(f"snapshot_download finished in {elapsed:.1f}s -> {path}", flush=True)


def verify_and_write_manifest(
    endpoint: str,
    repo_id: str,
    revision: str,
    dest: Path,
) -> dict:
    resolved_rev, tree_files = fetch_tree(endpoint, repo_id, revision)
    if revision != resolved_rev and not resolved_rev.startswith(revision):
        raise RuntimeError(f"revision mismatch: requested={revision} resolved={resolved_rev}")

    manifest_files = []
    total_bytes = 0
    for item in tree_files:
        rel_path = item["path"]
        # Skip .gitattributes / hidden repo metadata if absent
        local_file = safe_destination(dest, rel_path)
        if not local_file.is_file():
            raise RuntimeError(f"missing expected model file: {local_file}")
        actual_size = local_file.stat().st_size
        expected_size = item.get("size")
        if expected_size is not None and actual_size != expected_size:
            raise RuntimeError(f"size mismatch for {rel_path}: {actual_size} != {expected_size}")
        actual_sha = sha256_file(local_file)
        lfs_sha = (item.get("lfs") or {}).get("oid") or (item.get("lfs") or {}).get("sha256")
        if lfs_sha and actual_sha != lfs_sha:
            raise RuntimeError(f"LFS SHA256 mismatch for {rel_path}: {actual_sha} != {lfs_sha}")
        manifest_files.append(
            {
                "path": rel_path,
                "bytes": actual_size,
                "sha256": actual_sha,
                "lfs_sha256": lfs_sha,
            }
        )
        total_bytes += actual_size
        print(f"verified {rel_path} ({actual_size / (1024**2):.2f} MiB) sha256={actual_sha[:12]}", flush=True)

    manifest = {
        "schema": "morphserve.model-manifest.v1",
        "repo_id": repo_id,
        "endpoint": endpoint.rstrip("/"),
        "revision": revision,
        "resolved_revision": resolved_rev,
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file_count": len(manifest_files),
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / (1024**3), 3),
        "files": manifest_files,
    }
    manifest_path = dest / "MODEL_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and SHA256-verify HF model snapshot")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    args.dest.mkdir(parents=True, exist_ok=True)
    if not args.verify_only:
        download_via_hub(args.endpoint, args.repo_id, args.revision, args.dest, args.max_workers)
    manifest = verify_and_write_manifest(args.endpoint, args.repo_id, args.revision, args.dest)
    print(
        json.dumps(
            {
                "status": "verified",
                "repo_id": manifest["repo_id"],
                "revision": manifest["resolved_revision"],
                "file_count": manifest["file_count"],
                "total_gib": manifest["total_gib"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
