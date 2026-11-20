#!/usr/bin/env python3
"""Fast parallel HTTP-Range downloader with mirror probing, safe cancellation, and strict HF LFS SHA256 verification."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

HASH_CHUNK_SIZE = 4 * 1024 * 1024
DEFAULT_RANGE_CHUNK_MIB = 16

MODELSCOPE_REPO_ALIASES: dict[str, list[str]] = {
    "ByteDance-Seed/BAGEL-7B-MoT": ["bytedance-community/BAGEL-7B-MoT", "ByteDance-Seed/BAGEL-7B-MoT"],
    "InternVL-U/InternVL-U": ["OpenGVLab/InternVL-U", "InternVL-U/InternVL-U"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def get_json(url: str, timeout: int = 60) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "morphserve-fetch/0.3"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
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


def _probe_mirror_url(url: str, expected_size: int) -> tuple[bool, float, str]:
    probe_end = min(expected_size - 1, 2 * 1024 * 1024 - 1)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "morphserve-fetch/0.3",
            "Range": f"bytes=0-{probe_end}",
        },
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            status = getattr(resp, "status", 200)
            cr = resp.headers.get("Content-Range", "")
            data = resp.read()
            elapsed = max(1e-6, time.perf_counter() - t0)
            if status != 206:
                return False, 0.0, f"HTTP {status} (no range support)"
            if "/" in cr:
                total_str = cr.rsplit("/", 1)[-1].strip()
                if total_str.isdigit() and int(total_str) != expected_size:
                    return False, 0.0, f"size mismatch {total_str} != {expected_size}"
            if len(data) != probe_end + 1:
                return False, 0.0, f"short probe {len(data)} != {probe_end + 1}"
            mbps = (len(data) / (1024**2)) / elapsed
            return True, mbps, f"ok ({mbps:.2f} MiB/s)"
    except Exception as exc:
        return False, 0.0, f"{type(exc).__name__}: {exc}"


def rank_mirror_urls(
    endpoint: str,
    repo_id: str,
    revision: str,
    rel_path: str,
    expected_size: int,
    extra_mirrors: list[str] | None = None,
) -> list[str]:
    quoted = urllib.parse.quote(rel_path, safe="/")
    ms_file = urllib.parse.quote(rel_path, safe="")
    candidates = []
    for ms_repo in MODELSCOPE_REPO_ALIASES.get(repo_id, [repo_id]):
        candidates.append(
            f"https://modelscope.cn/api/v1/models/{ms_repo}/repo?Revision=master&FilePath={ms_file}"
        )
    hf_primary = f"{endpoint.rstrip('/')}/{repo_id}/resolve/{revision}/{quoted}"
    candidates.append(hf_primary)
    if "huggingface.co" not in endpoint:
        candidates.append(f"https://huggingface.co/{repo_id}/resolve/{revision}/{quoted}")
    if extra_mirrors:
        for base in extra_mirrors:
            candidates.append(f"{base.rstrip('/')}/{quoted}")

    if expected_size < 16 * 1024 * 1024:
        return [hf_primary]

    ranked: list[tuple[float, str]] = []
    for url in candidates:
        ok, mbps, note = _probe_mirror_url(url, expected_size)
        print(f"mirror probe {rel_path}: {url[:95]} -> {note}", flush=True)
        if ok:
            ranked.append((mbps, url))
    if not ranked:
        return [hf_primary]
    ranked.sort(key=lambda x: x[0], reverse=True)
    ordered = [u for _, u in ranked]
    print(f"selected mirror for {rel_path}: {ordered[0]} ({ranked[0][0]:.2f} MiB/s probe)", flush=True)
    return ordered


def _download_single_small(url: str, dest: Path, expected_size: int) -> None:
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, 7):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "morphserve-fetch/0.3"})
            with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as out:
                shutil.copyfileobj(resp, out, length=1024 * 1024)
            if part.stat().st_size != expected_size:
                raise IOError(f"size mismatch {part.stat().st_size} != {expected_size}")
            part.replace(dest)
            return
        except Exception as exc:
            print(f"retry {attempt}/6 for {dest.name}: {exc}", flush=True)
            time.sleep(min(20, 2 * attempt))
    raise RuntimeError(f"failed to download {dest.name} after 6 attempts")


def download_file_range_parallel(
    urls: list[str],
    dest: Path,
    expected_size: int,
    workers: int = 16,
    chunk_mib: int = DEFAULT_RANGE_CHUNK_MIB,
    max_retries: int = 24,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if expected_size < 8 * 1024 * 1024:
        _download_single_small(urls[0], dest, expected_size)
        return

    chunk_bytes = chunk_mib * 1024 * 1024
    num_chunks = (expected_size + chunk_bytes - 1) // chunk_bytes
    part_path = dest.with_name(dest.name + ".part")
    state_path = dest.with_name(dest.name + ".part.chunks.json")

    done_chunks: set[int] = set()
    if part_path.is_file() and part_path.stat().st_size == expected_size and state_path.is_file():
        try:
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            if saved.get("size") == expected_size and saved.get("chunk_bytes") == chunk_bytes:
                done_chunks = set(saved.get("done", []))
                print(
                    f"resuming {dest.name}: {len(done_chunks)}/{num_chunks} chunks already complete",
                    flush=True,
                )
        except Exception:
            done_chunks = set()

    fd = os.open(str(part_path), os.O_CREAT | os.O_RDWR, 0o666)
    stop_event = threading.Event()
    try:
        os.ftruncate(fd, expected_size)
        pending = [i for i in range(num_chunks) if i not in done_chunks]
        lock = threading.Lock()
        bytes_done = sum(
            min(chunk_bytes, expected_size - i * chunk_bytes) for i in done_chunks
        )
        t_start = time.perf_counter()
        bytes_this_run = 0
        last_log = t_start
        chunks_since_save = 0

        def save_state_locked() -> None:
            tmp = state_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "size": expected_size,
                        "chunk_bytes": chunk_bytes,
                        "done": sorted(done_chunks),
                    }
                ),
                encoding="utf-8",
            )
            tmp.replace(state_path)

        def fetch_chunk(idx: int) -> int:
            start = idx * chunk_bytes
            end = min(expected_size - 1, start + chunk_bytes - 1)
            need = end - start + 1
            last_err: Exception | None = None
            for attempt in range(1, max_retries + 1):
                if stop_event.is_set():
                    raise RuntimeError("cancelled")
                # Primary fast mirror for first attempts, rotate fallbacks on repeated failures
                url = urls[0] if (attempt <= 4 or len(urls) == 1) else urls[(attempt - 4) % len(urls)]
                try:
                    req = urllib.request.Request(
                        url,
                        headers={
                            "User-Agent": "morphserve-fetch/0.3",
                            "Range": f"bytes={start}-{end}",
                        },
                    )
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        status = getattr(resp, "status", 200)
                        if status != 206:
                            raise IOError(f"expected HTTP 206, got {status}")
                        buf = bytearray()
                        while len(buf) < need:
                            if stop_event.is_set():
                                raise RuntimeError("cancelled")
                            piece = resp.read(min(1024 * 1024, need - len(buf)))
                            if not piece:
                                break
                            buf.extend(piece)
                        if len(buf) != need:
                            raise IOError(f"short chunk {idx}: got {len(buf)} != {need}")
                        written = 0
                        view = memoryview(buf)
                        while written < need:
                            n = os.pwrite(fd, view[written:], start + written)
                            if n <= 0:
                                raise IOError("os.pwrite returned 0")
                            written += n
                        return need
                except Exception as exc:
                    last_err = exc
                    if stop_event.is_set():
                        raise
                    time.sleep(min(20.0, 1.0 * attempt))
            raise RuntimeError(f"chunk {idx} ({start}-{end}) failed after {max_retries} attempts: {last_err}")

        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {pool.submit(fetch_chunk, idx): idx for idx in pending}
            for fut in as_completed(futures):
                idx = futures[fut]
                nbytes = fut.result()
                now = time.perf_counter()
                with lock:
                    done_chunks.add(idx)
                    bytes_done += nbytes
                    bytes_this_run += nbytes
                    chunks_since_save += 1
                    if chunks_since_save >= 8 or len(done_chunks) == num_chunks:
                        save_state_locked()
                        chunks_since_save = 0
                    if now - last_log >= 15.0 or len(done_chunks) == num_chunks:
                        rate_mibs = (bytes_this_run / (1024**2)) / max(1e-6, now - t_start)
                        pct = 100.0 * bytes_done / expected_size
                        rem_mib = (expected_size - bytes_done) / (1024**2)
                        eta_s = rem_mib / max(0.01, rate_mibs)
                        print(
                            f"[{dest.name}] {pct:5.1f}% ({bytes_done / (1024**3):.2f}/{expected_size / (1024**3):.2f} GiB) "
                            f"rate={rate_mibs:.2f} MiB/s eta={eta_s:.0f}s chunks={len(done_chunks)}/{num_chunks}",
                            flush=True,
                        )
                        last_log = now
        except Exception:
            stop_event.set()
            with lock:
                save_state_locked()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        os.fsync(fd)
    finally:
        os.close(fd)

    if part_path.stat().st_size != expected_size:
        raise RuntimeError(f"assembled size mismatch: {part_path.stat().st_size} != {expected_size}")
    part_path.replace(dest)
    if state_path.is_file():
        state_path.unlink()


def sync_and_verify_repo(
    endpoint: str,
    repo_id: str,
    revision: str,
    dest: Path,
    workers: int = 16,
    chunk_mib: int = DEFAULT_RANGE_CHUNK_MIB,
    verify_only: bool = False,
    only_files: set[str] | None = None,
) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    stale_cache = dest / ".cache"
    if stale_cache.is_dir():
        shutil.rmtree(stale_cache, ignore_errors=True)

    resolved_rev, tree_files = fetch_tree(endpoint, repo_id, revision)
    if revision != resolved_rev and not resolved_rev.startswith(revision):
        raise RuntimeError(f"revision mismatch: requested={revision} resolved={resolved_rev}")

    selected_files = [
        item for item in tree_files if (only_files is None or item["path"] in only_files)
    ]
    if only_files is not None and not selected_files:
        raise RuntimeError(f"no files matched --only-file {sorted(only_files)}")

    manifest_files = []
    total_bytes = 0
    for item in selected_files:
        rel_path = item["path"]
        expected_size = item.get("size")
        lfs_sha = (item.get("lfs") or {}).get("oid") or (item.get("lfs") or {}).get("sha256")
        local_file = safe_destination(dest, rel_path)

        need_download = True
        if local_file.is_file() and (expected_size is None or local_file.stat().st_size == expected_size):
            actual_sha = sha256_file(local_file)
            if not lfs_sha or actual_sha == lfs_sha:
                need_download = False
                print(
                    f"already verified {rel_path} ({local_file.stat().st_size / (1024**2):.2f} MiB) sha256={actual_sha[:12]}",
                    flush=True,
                )
            else:
                print(f"checksum mismatch on existing {rel_path}; re-downloading", flush=True)
                local_file.unlink()

        if need_download:
            if verify_only:
                raise RuntimeError(f"missing or invalid file in --verify-only mode: {local_file}")
            if expected_size is None:
                raise RuntimeError(f"missing expected size in tree metadata for {rel_path}")
            urls = rank_mirror_urls(endpoint, repo_id, resolved_rev, rel_path, expected_size)
            download_file_range_parallel(
                urls,
                local_file,
                expected_size,
                workers=workers,
                chunk_mib=chunk_mib,
            )
            actual_sha = sha256_file(local_file)
            if lfs_sha and actual_sha != lfs_sha:
                local_file.unlink(missing_ok=True)
                raise RuntimeError(
                    f"LFS SHA256 mismatch after download for {rel_path}: {actual_sha} != {lfs_sha}"
                )
            print(
                f"verified {rel_path} ({expected_size / (1024**2):.2f} MiB) sha256={actual_sha[:12]}",
                flush=True,
            )

        actual_size = local_file.stat().st_size
        manifest_files.append(
            {
                "path": rel_path,
                "bytes": actual_size,
                "sha256": actual_sha,
                "lfs_sha256": lfs_sha,
            }
        )
        total_bytes += actual_size

    manifest = {
        "schema": "morphserve.model-manifest.v1",
        "repo_id": repo_id,
        "endpoint": endpoint.rstrip("/"),
        "revision": revision,
        "resolved_revision": resolved_rev,
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "partial_selection": sorted(only_files) if only_files is not None else None,
        "file_count": len(manifest_files),
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / (1024**3), 3),
        "files": manifest_files,
    }
    manifest_name = "MODEL_MANIFEST.json" if only_files is None else "PARTIAL_MANIFEST.json"
    manifest_path = dest / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parallel HTTP-Range HF model downloader + SHA256 verifier")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=DEFAULT_RANGE_CHUNK_MIB)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--only-file", action="append", default=None)
    args = parser.parse_args(argv)

    manifest = sync_and_verify_repo(
        args.endpoint,
        args.repo_id,
        args.revision,
        args.dest,
        workers=args.max_workers,
        chunk_mib=args.chunk_mib,
        verify_only=args.verify_only,
        only_files=set(args.only_file) if args.only_file else None,
    )
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
