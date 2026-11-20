"""Phase-0 Job 0c: 8xA100 PCIe/NUMA topology, P2P, NUMA-pinned H2D, memfd, and NCCL P2P probe."""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
from pathlib import Path
import platform
import subprocess
import time


CU_MEMHOSTREGISTER_PORTABLE = 0x1


def run_cmd(args: list[str], timeout: int = 30) -> dict:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def median(vals: list[float]) -> float:
    s = sorted(vals)
    return s[len(s) // 2] if s else 0.0


class CudaHostPin:
    """Thin ctypes wrapper around libcuda.so.1 for NUMA-pinned H2D benchmarking."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.cuMemHostRegister = getattr(self.lib, "cuMemHostRegister_v2", self.lib.cuMemHostRegister)
        self.cuMemHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        self.cuMemHostRegister.restype = ctypes.c_int

        self.cuMemHostUnregister = self.lib.cuMemHostUnregister
        self.cuMemHostUnregister.argtypes = [ctypes.c_void_p]
        self.cuMemHostUnregister.restype = ctypes.c_int

        self.cuMemcpyHtoDAsync = getattr(self.lib, "cuMemcpyHtoDAsync_v2", self.lib.cuMemcpyHtoDAsync)
        self.cuMemcpyHtoDAsync.argtypes = [
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        self.cuMemcpyHtoDAsync.restype = ctypes.c_int

    def register(self, ptr: int, nbytes: int) -> None:
        rc = self.cuMemHostRegister(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), CU_MEMHOSTREGISTER_PORTABLE)
        if rc != 0:
            raise RuntimeError(f"cuMemHostRegister failed with CUresult={rc}")

    def unregister(self, ptr: int) -> None:
        rc = self.cuMemHostUnregister(ctypes.c_void_p(ptr))
        if rc != 0:
            raise RuntimeError(f"cuMemHostUnregister failed with CUresult={rc}")

    def h2d_async(self, dst_ptr: int, src_ptr: int, nbytes: int, stream_ptr: int) -> None:
        rc = self.cuMemcpyHtoDAsync(
            ctypes.c_uint64(dst_ptr),
            ctypes.c_void_p(src_ptr),
            ctypes.c_size_t(nbytes),
            ctypes.c_void_p(stream_ptr),
        )
        if rc != 0:
            raise RuntimeError(f"cuMemcpyHtoDAsync failed with CUresult={rc}")


def load_libnuma():
    lib = ctypes.CDLL("libnuma.so.1")
    lib.numa_available.argtypes = []
    lib.numa_available.restype = ctypes.c_int
    lib.numa_max_node.argtypes = []
    lib.numa_max_node.restype = ctypes.c_int
    lib.numa_alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]
    lib.numa_alloc_onnode.restype = ctypes.c_void_p
    lib.numa_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.numa_free.restype = None
    lib.numa_tonode_memory.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.numa_tonode_memory.restype = None
    return lib


def benchmark_p2p_matrix(size_mib: int = 512, iters: int = 8) -> dict:
    import torch

    n = torch.cuda.device_count()
    nbytes = size_mib * (1024**2)
    nelem = nbytes // 4

    can_access = [[bool(torch.cuda.can_device_access_peer(i, j)) if i != j else True for j in range(n)] for i in range(n)]
    bufs = [torch.empty(nelem, dtype=torch.float32, device=f"cuda:{i}") for i in range(n)]
    for i in range(n):
        bufs[i].fill_(float(i + 1))
        torch.cuda.synchronize(i)

    bw_gbps = [[0.0 for _ in range(n)] for _ in range(n)]
    for src in range(n):
        for dst in range(n):
            if src == dst:
                continue
            with torch.cuda.device(dst):
                stream = torch.cuda.current_stream(dst)
                # Warmup
                for _ in range(2):
                    bufs[dst].copy_(bufs[src], non_blocking=True)
                stream.synchronize()

                samples = []
                for _ in range(iters):
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    ev0.record(stream)
                    bufs[dst].copy_(bufs[src], non_blocking=True)
                    ev1.record(stream)
                    ev1.synchronize()
                    ms = ev0.elapsed_time(ev1)
                    samples.append((nbytes / 1e9) / (ms / 1e3))
                bw_gbps[src][dst] = round(median(samples), 3)

    # Concurrent pairs benchmark (when n >= 8)
    concurrent = {}
    if n >= 8:
        patterns = {
            "adjacent_4pairs": [(0, 1), (2, 3), (4, 5), (6, 7)],
            "adjacent_8dir_bidir": [(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4), (6, 7), (7, 6)],
            "cross_half_4pairs": [(0, 4), (1, 5), (2, 6), (3, 7)],
        }
        # Separate dst buffers for concurrent tests to avoid write conflicts
        dst_bufs = [torch.empty(nelem, dtype=torch.float32, device=f"cuda:{i}") for i in range(n)]
        streams = [torch.cuda.Stream(device=i) for i in range(n)]
        for name, pairs in patterns.items():
            # Warmup
            for s, d in pairs:
                with torch.cuda.device(d), torch.cuda.stream(streams[d]):
                    dst_bufs[d].copy_(bufs[s], non_blocking=True)
            for i in range(n):
                streams[i].synchronize()

            agg_samples = []
            for _ in range(iters):
                t0 = time.perf_counter()
                for s, d in pairs:
                    with torch.cuda.device(d), torch.cuda.stream(streams[d]):
                        dst_bufs[d].copy_(bufs[s], non_blocking=True)
                for _, d in pairs:
                    streams[d].synchronize()
                elapsed = time.perf_counter() - t0
                total_gb = (len(pairs) * nbytes) / 1e9
                agg_samples.append(total_gb / elapsed)
            agg = round(median(agg_samples), 3)
            concurrent[name] = {
                "pairs": pairs,
                "aggregate_gbps": agg,
                "per_link_gbps": round(agg / len(pairs), 3),
            }
        del dst_bufs
    del bufs
    torch.cuda.empty_cache()

    return {
        "size_mib": size_mib,
        "device_count": n,
        "can_access_peer": can_access,
        "unidirectional_gbps": bw_gbps,
        "concurrent_p2p": concurrent,
    }


def benchmark_numa_h2d(size_mib: int = 512, iters: int = 8) -> dict:
    import torch

    n = torch.cuda.device_count()
    nbytes = size_mib * (1024**2)
    nelem = nbytes // 4

    numa = load_libnuma()
    if numa.numa_available() < 0:
        return {"status": "numa_unavailable"}
    num_nodes = int(numa.numa_max_node()) + 1
    cuda_pin = CudaHostPin()

    # Initialize CUDA contexts on all GPUs before cuMemHostRegister
    gpu_bufs = []
    streams = []
    for i in range(n):
        torch.cuda.set_device(i)
        gpu_bufs.append(torch.empty(nelem, dtype=torch.float32, device=f"cuda:{i}"))
        streams.append(torch.cuda.Stream(device=i))
        torch.cuda.synchronize(i)

    # Allocate & pin 1 buffer per NUMA node (shared read-only mirror simulation!)
    node_ptrs: dict[int, int] = {}
    try:
        for node in range(num_nodes):
            ptr = numa.numa_alloc_onnode(nbytes, node)
            if not ptr:
                raise RuntimeError(f"numa_alloc_onnode({nbytes}, {node}) returned NULL")
            # Touch every page on target node before pinning
            ctypes.memset(ctypes.c_void_p(ptr), 0x3C + node, nbytes)
            cuda_pin.register(ptr, nbytes)
            node_ptrs[node] = ptr

        # 1. Single-GPU H2D from each NUMA node
        single_gpu = {}
        inferred_numa = {}
        for g in range(n):
            torch.cuda.set_device(g)
            s = streams[g]
            dst_ptr = gpu_bufs[g].data_ptr()
            per_node = {}
            for node, src_ptr in node_ptrs.items():
                # Warmup
                for _ in range(2):
                    cuda_pin.h2d_async(dst_ptr, src_ptr, nbytes, s.cuda_stream)
                s.synchronize()
                samples = []
                for _ in range(iters):
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    ev0.record(s)
                    cuda_pin.h2d_async(dst_ptr, src_ptr, nbytes, s.cuda_stream)
                    ev1.record(s)
                    ev1.synchronize()
                    ms = ev0.elapsed_time(ev1)
                    samples.append((nbytes / 1e9) / (ms / 1e3))
                per_node[f"numa_{node}"] = round(median(samples), 3)
            single_gpu[f"gpu_{g}"] = per_node
            best_node = max(range(num_nodes), key=lambda nd: per_node[f"numa_{nd}"])
            inferred_numa[f"gpu_{g}"] = best_node

        # 2. Multi-GPU concurrent H2D (shared read from NUMA pinned mirror!)
        concurrent_h2d = {}
        groups = {}
        if n >= 8 and num_nodes >= 2:
            node0_gpus = [g for g in range(n) if inferred_numa[f"gpu_{g}"] == 0]
            node1_gpus = [g for g in range(n) if inferred_numa[f"gpu_{g}"] == 1]
            # Fallback to 0..3 / 4..7 if NUMA delta is tiny
            if len(node0_gpus) != 4 or len(node1_gpus) != 4:
                node0_gpus, node1_gpus = [0, 1, 2, 3], [4, 5, 6, 7]
            groups = {
                "socket0_4gpu_local_numa0": [(g, 0) for g in node0_gpus],
                "socket0_4gpu_remote_numa1": [(g, 1) for g in node0_gpus],
                "socket1_4gpu_local_numa1": [(g, 1) for g in node1_gpus],
                "all_8gpu_local_numa": [(g, 0) for g in node0_gpus] + [(g, 1) for g in node1_gpus],
                "all_8gpu_all_from_numa0": [(g, 0) for g in range(n)],
            }
        else:
            groups = {f"all_{n}gpu_numa0": [(g, 0) for g in range(n)]}

        for label, pairs in groups.items():
            for g, nd in pairs:
                torch.cuda.set_device(g)
                cuda_pin.h2d_async(gpu_bufs[g].data_ptr(), node_ptrs[nd], nbytes, streams[g].cuda_stream)
            for g, _ in pairs:
                streams[g].synchronize()

            agg_samples = []
            for _ in range(iters):
                t0 = time.perf_counter()
                for g, nd in pairs:
                    torch.cuda.set_device(g)
                    cuda_pin.h2d_async(gpu_bufs[g].data_ptr(), node_ptrs[nd], nbytes, streams[g].cuda_stream)
                for g, _ in pairs:
                    streams[g].synchronize()
                elapsed = time.perf_counter() - t0
                agg_samples.append(((len(pairs) * nbytes) / 1e9) / elapsed)
            agg = round(median(agg_samples), 3)
            concurrent_h2d[label] = {
                "gpu_node_pairs": pairs,
                "aggregate_gbps": agg,
                "per_gpu_gbps": round(agg / len(pairs), 3),
            }

        # 3. Verify memfd_create + numa_tonode_memory + cuMemHostRegister (bypasses 32 GiB /dev/shm cap!)
        memfd_test = _verify_memfd_numa_pin(numa, cuda_pin, gpu_bufs[0], streams[0], size_mib=256)

        return {
            "status": "ok",
            "size_mib": size_mib,
            "num_numa_nodes": num_nodes,
            "single_gpu_h2d_gbps": single_gpu,
            "inferred_gpu_numa_node": inferred_numa,
            "concurrent_h2d": concurrent_h2d,
            "memfd_numa_pin_verify": memfd_test,
        }
    finally:
        for node, ptr in node_ptrs.items():
            try:
                cuda_pin.unregister(ptr)
            except Exception:
                pass
            numa.numa_free(ctypes.c_void_p(ptr), nbytes)
        del gpu_bufs
        torch.cuda.empty_cache()


def _verify_memfd_numa_pin(numa, cuda_pin, gpu_buf, stream, size_mib: int = 256) -> dict:
    import torch

    nbytes = size_mib * (1024**2)
    fd = -1
    mm = None
    arr = None
    buf_ptr = 0
    pinned = False
    try:
        fd = os.memfd_create("morphserve_numa_weight_test", 0)
        os.ftruncate(fd, nbytes)
        mm = mmap.mmap(fd, nbytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        arr = (ctypes.c_uint8 * nbytes).from_buffer(mm)
        buf_ptr = ctypes.addressof(arr)
        numa.numa_tonode_memory(ctypes.c_void_p(buf_ptr), ctypes.c_size_t(nbytes), 0)
        ctypes.memset(ctypes.c_void_p(buf_ptr), 0x7A, nbytes)
        cuda_pin.register(buf_ptr, nbytes)
        pinned = True

        torch.cuda.set_device(0)
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record(stream)
        cuda_pin.h2d_async(gpu_buf.data_ptr(), buf_ptr, nbytes, stream.cuda_stream)
        ev1.record(stream)
        ev1.synchronize()
        ms = ev0.elapsed_time(ev1)
        gbps = round((nbytes / 1e9) / (ms / 1e3), 3)
        # Verify byte integrity on GPU
        head_byte = int(gpu_buf.view(torch.uint8)[0].item())
        tail_byte = int(gpu_buf.view(torch.uint8)[nbytes - 1].item())
        return {
            "ok": head_byte == 0x7A and tail_byte == 0x7A,
            "size_mib": size_mib,
            "h2d_gbps": gbps,
            "verified_bytes": head_byte == 0x7A and tail_byte == 0x7A,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if pinned and buf_ptr:
            try:
                cuda_pin.unregister(buf_ptr)
            except Exception:
                pass
        if arr is not None:
            del arr  # release buffer view before closing mmap
        if mm is not None:
            mm.close()
        if fd >= 0:
            os.close(fd)


def _nccl_worker(rank: int, world_size: int, size_mib: int, iters: int, out_path: str) -> None:
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29561"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        nbytes = size_mib * (1024**2)
        nelem = nbytes // 4
        tensor = torch.full((nelem,), float(rank + 1), dtype=torch.float32, device=f"cuda:{rank}")
        recv = torch.empty_like(tensor)

        test_pairs = [(0, 1)]
        if world_size >= 8:
            test_pairs.extend([(0, 2), (0, 4), (4, 5)])

        results = {}
        for src, dst in test_pairs:
            dist.barrier()
            samples = []
            for it in range(-2, iters):
                torch.cuda.synchronize(rank)
                dist.barrier()
                t0 = time.perf_counter()
                ops = []
                if rank == src:
                    ops.append(dist.P2POp(dist.isend, tensor, dst))
                elif rank == dst:
                    ops.append(dist.P2POp(dist.irecv, recv, src))
                if ops:
                    reqs = dist.batch_isend_irecv(ops)
                    for r in reqs:
                        r.wait()
                torch.cuda.synchronize(rank)
                elapsed = time.perf_counter() - t0
                if it >= 0 and rank == dst:
                    samples.append((nbytes / 1e9) / elapsed)
            if rank == dst:
                results[f"{src}->{dst}"] = {
                    "gbps_p50": round(median(samples), 3),
                    "verified": bool(float(recv[0].item()) == float(src + 1)),
                }

        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, results)
        if rank == 0:
            merged = {}
            for item in gathered:
                if item:
                    merged.update(item)
            Path(out_path).write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


def benchmark_nccl_p2p(size_mib: int = 512, iters: int = 6, tmp_out: Path = Path("/tmp/nccl_p2p.json")) -> dict:
    import torch
    import torch.multiprocessing as mp

    n = torch.cuda.device_count()
    if n < 2:
        return {"status": "skipped_single_gpu"}
    try:
        mp.spawn(_nccl_worker, args=(n, size_mib, iters, str(tmp_out)), nprocs=n, join=True)
        data = json.loads(tmp_out.read_text(encoding="utf-8"))
        return {"status": "ok", "size_mib": size_mib, "pairs": data}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MorphServe 8xA100 PCIe/NUMA topology & bandwidth probe")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size-mib", type=int, default=512)
    parser.add_argument("--skip-nccl", action="store_true")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "morphserve.phase0.topo8gpu.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "nvidia_smi_topo": run_cmd(["nvidia-smi", "topo", "-m"]),
        "nvidia_smi_gpus": run_cmd(
            ["nvidia-smi", "--query-gpu=index,name,pci.bus_id,memory.total", "--format=csv,noheader"]
        ),
        "numactl_hardware": run_cmd(["numactl", "--hardware"]),
        "p2p_matrix": benchmark_p2p_matrix(size_mib=args.size_mib),
        "numa_h2d": benchmark_numa_h2d(size_mib=args.size_mib),
    }
    if not args.skip_nccl:
        report["nccl_p2p"] = benchmark_nccl_p2p(
            size_mib=args.size_mib,
            tmp_out=args.output_dir / "nccl-p2p-raw.json",
        )

    out_path = args.output_dir / "topo-8gpu-report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)

    # Print compact summary
    p2p = report["p2p_matrix"]["unidirectional_gbps"]
    nonzero = [v for row in p2p for v in row if v > 0]
    summary = {
        "gpus": report["p2p_matrix"]["device_count"],
        "p2p_min_gbps": min(nonzero) if nonzero else None,
        "p2p_max_gbps": max(nonzero) if nonzero else None,
        "concurrent_p2p": report["p2p_matrix"]["concurrent_p2p"],
        "inferred_numa": report["numa_h2d"].get("inferred_gpu_numa_node"),
        "memfd_numa_ok": report["numa_h2d"].get("memfd_numa_pin_verify", {}).get("ok"),
        "nccl_p2p": report.get("nccl_p2p", {}).get("pairs"),
    }
    print("TOPO8GPU_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
