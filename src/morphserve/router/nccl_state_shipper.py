"""Cross-process shared NUMA memfd pool & NCCL P2P KV/bridge state shipper for Module 2."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import mmap
import os
import socket
import struct
import time
from typing import Any


CU_MEMHOSTREGISTER_PORTABLE = 0x1


def _read_meminfo_kb() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if ":" not in line:
                    continue
                k, rest = line.split(":", 1)
                parts = rest.strip().split()
                if parts and parts[0].isdigit():
                    out[k] = int(parts[0])
    except OSError:
        pass
    return out


class _HostCudaPin:
    """Thin libcuda wrapper for portable host pinning & H2D transfers in multi-GPU workers."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.cuMemHostRegister = getattr(self.lib, "cuMemHostRegister_v2", self.lib.cuMemHostRegister)
        self.cuMemHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        self.cuMemHostRegister.restype = ctypes.c_int

        self.cuMemHostUnregister = self.lib.cuMemHostUnregister
        self.cuMemHostUnregister.argtypes = [ctypes.c_void_p]
        self.cuMemHostUnregister.restype = ctypes.c_int

    def register(self, ptr: int, nbytes: int) -> None:
        rc = int(self.cuMemHostRegister(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), CU_MEMHOSTREGISTER_PORTABLE))
        if rc != 0:
            raise RuntimeError(f"cuMemHostRegister(ptr=0x{ptr:x}, nbytes={nbytes}) failed with CUresult={rc}")

    def unregister(self, ptr: int) -> None:
        rc = int(self.cuMemHostUnregister(ctypes.c_void_p(ptr)))
        if rc != 0:
            raise RuntimeError(f"cuMemHostUnregister failed with CUresult={rc}")


def _bind_numa_memory(ptr: int, nbytes: int, numa_node: int) -> bool:
    try:
        libnuma = ctypes.CDLL("libnuma.so.1")
        libnuma.numa_available.restype = ctypes.c_int
        if int(libnuma.numa_available()) < 0:
            return False
        libnuma.numa_tonode_memory.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libnuma.numa_tonode_memory.restype = None
        libnuma.numa_tonode_memory(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), int(numa_node))
        return True
    except Exception:
        return False


class SharedNumaMemfdMirror:
    """Process-shareable NUMA-bound anonymous `memfd` weight mirror (bypasses `/dev/shm` 32 GiB cap).

    On a dual-socket 8xA100 node, GPUs `{0,1,2,3}` share ONE NUMA-0 memfd mirror and GPUs
    `{4,5,6,7}` share ONE NUMA-1 memfd mirror via `/proc/<leader_pid>/fd/<fd>` + `mmap(MAP_SHARED)`.
    Every attached worker receives identical backing kernel pages (`inode` match) while consuming
    only `2x` weight memory instead of `8x` (`4.0x` host RAM savings!).
    """

    def __init__(self, fd: int, mm: mmap.mmap, nbytes: int, numa_node: int, is_owner: bool) -> None:
        self.fd = int(fd)
        self.mm = mm
        self.nbytes = int(nbytes)
        self.numa_node = int(numa_node)
        self.is_owner = bool(is_owner)
        self.inode = int(os.fstat(self.fd).st_ino)
        self._buf_keepalive = (ctypes.c_uint8 * self.nbytes).from_buffer(self.mm)
        self.ptr = int(ctypes.addressof(self._buf_keepalive))
        self._pin_api: _HostCudaPin | None = None
        self._pinned_vmm_lib: Any = None
        self.pinned = False

    @classmethod
    def create(
        cls,
        nbytes: int,
        numa_node: int = 0,
        name: str = "morphserve_shared_expert",
        populate_bytes: bytes | None = None,
    ) -> "SharedNumaMemfdMirror":
        memfd_fn = getattr(os, "memfd_create", None)
        if memfd_fn is None:
            raise RuntimeError("os.memfd_create is required")
        fd = int(memfd_fn(name, 0))
        try:
            os.ftruncate(fd, int(nbytes))
            mm = mmap.mmap(fd, int(nbytes), mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        except Exception:
            os.close(fd)
            raise
        obj = cls(fd=fd, mm=mm, nbytes=int(nbytes), numa_node=numa_node, is_owner=True)
        # Touch 1 byte per 2 MiB stride so kernel allocates pages cleanly before numa_tonode_memory
        stride = 2 * 1024 * 1024
        for off in range(0, obj.nbytes, stride):
            obj.mm[off] = 0
        if populate_bytes:
            mv = memoryview(obj.mm)
            mv[: len(populate_bytes)] = populate_bytes
        _bind_numa_memory(obj.ptr, obj.nbytes, numa_node)
        return obj

    @classmethod
    def attach_proc_fd(
        cls,
        leader_pid: int,
        leader_fd: int,
        nbytes: int,
        numa_node: int = 0,
    ) -> "SharedNumaMemfdMirror":
        proc_path = f"/proc/{int(leader_pid)}/fd/{int(leader_fd)}"
        fd = os.open(proc_path, os.O_RDWR)
        try:
            mm = mmap.mmap(fd, int(nbytes), mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        except Exception:
            os.close(fd)
            raise
        return cls(fd=fd, mm=mm, nbytes=int(nbytes), numa_node=numa_node, is_owner=False)

    def pin_cuda(self, vmm: Any = None) -> None:
        if self.pinned:
            return
        if vmm is not None:
            rc = int(vmm.lib.tripool_host_register(ctypes.c_void_p(self.ptr), ctypes.c_size_t(self.nbytes)))
            if rc != 0:
                raise RuntimeError(f"tripool_host_register failed (rc={rc}) for {self.nbytes} bytes")
            self._pinned_vmm_lib = vmm.lib
            self.pinned = True
            return
        if self._pin_api is None:
            self._pin_api = _HostCudaPin()
        self._pin_api.register(self.ptr, self.nbytes)
        self.pinned = True

    def as_uint8_tensor(self) -> Any:
        import torch

        return torch.frombuffer(self._buf_keepalive, dtype=torch.uint8)

    def stream_h2d_chunks(self, dst_device_tensor: Any, chunk_bytes: int = 64 * 1024 * 1024) -> float:
        """Copy shared pinned mirror -> GPU tensor via CUDA H2D and return achieved GB/s."""
        import torch

        if not self.pinned:
            self.pin_cuda()
        host_u8 = self.as_uint8_tensor()
        dst_u8 = dst_device_tensor.view(torch.uint8).reshape(-1)
        total = min(int(host_u8.numel()), int(dst_u8.numel()))
        lib = self._pin_api.lib
        h2d_fn = getattr(lib, "cuMemcpyHtoDAsync_v2", lib.cuMemcpyHtoDAsync)
        h2d_fn.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        h2d_fn.restype = ctypes.c_int

        stream = torch.cuda.current_stream(dst_u8.device)
        torch.cuda.synchronize(dst_u8.device)
        t0 = time.perf_counter()
        off = 0
        dst_base = int(dst_u8.data_ptr())
        while off < total:
            sz = min(chunk_bytes, total - off)
            rc = int(h2d_fn(ctypes.c_uint64(dst_base + off), ctypes.c_void_p(self.ptr + off), ctypes.c_size_t(sz), ctypes.c_void_p(int(stream.cuda_stream))))
            if rc != 0:
                raise RuntimeError(f"cuMemcpyHtoDAsync failed CUresult={rc}")
            off += sz
        torch.cuda.synchronize(dst_u8.device)
        dt_s = max(time.perf_counter() - t0, 1e-6)
        return (total / 1e9) / dt_s

    def close(self) -> None:
        if self.pinned:
            if self._pinned_vmm_lib is not None:
                try:
                    self._pinned_vmm_lib.tripool_host_unregister(ctypes.c_void_p(self.ptr))
                except Exception:
                    pass
                self._pinned_vmm_lib = None
            elif self._pin_api is not None:
                try:
                    self._pin_api.unregister(self.ptr)
                except Exception:
                    pass
            self.pinned = False
        self._buf_keepalive = None
        if getattr(self, "mm", None) is not None:
            self.mm.close()
            self.mm = None
        if getattr(self, "fd", -1) >= 0:
            os.close(self.fd)
            self.fd = -1


def find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class ShipMetrics:
    bytes_transferred: int
    transfer_ms: float
    bandwidth_gbps: float
    num_layers: int


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError("peer socket closed before expected bytes")
        buf.extend(chunk)
    return bytes(buf)


def _load_libnccl() -> ctypes.CDLL:
    import ctypes.util

    try:
        import torch  # Ensure PyTorch CUDA/NCCL libs are loaded in-process if installed
        torch_dir = os.path.dirname(torch.__file__)
        cand = os.path.join(os.path.dirname(torch_dir), "nvidia", "nccl", "lib", "libnccl.so.2")
        if os.path.exists(cand):
            return ctypes.CDLL(cand)
    except Exception:
        pass
    for name in ("libnccl.so.2", "libnccl.so", ctypes.util.find_library("nccl")):
        if not name:
            continue
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RuntimeError("Could not locate libnccl.so.2 for DirectNcclP2PChannel")


class _NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


class DirectNcclP2PChannel:
    """Standalone 2-rank NCCL P2P communicator + TCP control plane (bypasses `torch.distributed`).

    Needed when worker processes also embed `vllm_omni` single-rank pipelines that claim
    `torch.distributed`'s default WORLD group.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_port: int,
        master_addr: str = "127.0.0.1",
        connect_timeout_s: float = 30.0,
    ) -> None:
        if world_size != 2:
            raise ValueError(f"DirectNcclP2PChannel supports world_size=2, got {world_size}")
        if rank not in (0, 1):
            raise ValueError(f"invalid rank={rank} for world_size=2")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.master_addr = master_addr
        self.master_port = int(master_port)
        self._nccl = _load_libnccl()
        self._bind_nccl_symbols()
        self._listen_sock: socket.socket | None = None
        self._peer_sock: socket.socket | None = None
        self.comm = ctypes.c_void_p(0)

        if self.rank == 0:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.master_addr, self.master_port))
            srv.listen(4)
            srv.settimeout(connect_timeout_s)
            self._listen_sock = srv
            conn, _ = srv.accept()
            conn.settimeout(None)
            self._peer_sock = conn
            uid = _NcclUniqueId()
            self._check(self._nccl.ncclGetUniqueId(ctypes.byref(uid)), "ncclGetUniqueId")
            self._peer_sock.sendall(bytes(uid))
        else:
            deadline = time.perf_counter() + connect_timeout_s
            conn: socket.socket | None = None
            last_err: Exception | None = None
            while time.perf_counter() < deadline:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    s.connect((self.master_addr, self.master_port))
                    conn = s
                    break
                except OSError as exc:
                    s.close()
                    last_err = exc
                    time.sleep(0.05)
            if conn is None:
                raise TimeoutError(f"rank 1 failed to connect to {self.master_addr}:{self.master_port}: {last_err}")
            self._peer_sock = conn
            raw_uid = _recv_exact(self._peer_sock, 128)
            uid = _NcclUniqueId.from_buffer_copy(raw_uid)

        self._check(
            self._nccl.ncclCommInitRank(ctypes.byref(self.comm), ctypes.c_int(2), uid, ctypes.c_int(self.rank)),
            "ncclCommInitRank",
        )
        self.barrier()

    def _bind_nccl_symbols(self) -> None:
        self._nccl.ncclGetUniqueId.argtypes = [ctypes.POINTER(_NcclUniqueId)]
        self._nccl.ncclGetUniqueId.restype = ctypes.c_int
        self._nccl.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _NcclUniqueId, ctypes.c_int]
        self._nccl.ncclCommInitRank.restype = ctypes.c_int
        self._nccl.ncclCommDestroy.argtypes = [ctypes.c_void_p]
        self._nccl.ncclCommDestroy.restype = ctypes.c_int
        self._nccl.ncclSend.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self._nccl.ncclSend.restype = ctypes.c_int
        self._nccl.ncclRecv.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self._nccl.ncclRecv.restype = ctypes.c_int

    def _check(self, rc: int, op: str) -> None:
        if int(rc) != 0:
            raise RuntimeError(f"{op} failed with ncclResult_t={rc}")

    def send_json(self, obj: Any) -> None:
        import json

        assert self._peer_sock is not None
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self._peer_sock.sendall(struct.pack("!Q", len(raw)) + raw)

    def recv_json(self) -> Any:
        import json

        assert self._peer_sock is not None
        (n,) = struct.unpack("!Q", _recv_exact(self._peer_sock, 8))
        if n == 0:
            return None
        return json.loads(_recv_exact(self._peer_sock, n).decode("utf-8"))

    def barrier(self) -> None:
        assert self._peer_sock is not None
        self._peer_sock.sendall(b"SYNC")
        ack = _recv_exact(self._peer_sock, 4)
        if ack != b"SYNC":
            raise RuntimeError(f"unexpected barrier token: {ack!r}")

    def exchange_json(self, my_obj: Any) -> list[Any]:
        if self.rank == 0:
            self.send_json(my_obj)
            peer_obj = self.recv_json()
            return [my_obj, peer_obj]
        peer_obj = self.recv_json()
        self.send_json(my_obj)
        return [peer_obj, my_obj]

    def send_tensor_u8(self, tensor_u8: Any, peer_rank: int, stream_ptr: int = 0) -> None:
        nbytes = int(tensor_u8.numel())
        if nbytes <= 0:
            return
        self._check(
            self._nccl.ncclSend(
                ctypes.c_void_p(int(tensor_u8.data_ptr())),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(0),  # ncclInt8 (1 byte)
                ctypes.c_int(int(peer_rank)),
                self.comm,
                ctypes.c_void_p(int(stream_ptr)),
            ),
            "ncclSend",
        )

    def recv_tensor_u8(self, tensor_u8: Any, peer_rank: int, stream_ptr: int = 0) -> None:
        nbytes = int(tensor_u8.numel())
        if nbytes <= 0:
            return
        self._check(
            self._nccl.ncclRecv(
                ctypes.c_void_p(int(tensor_u8.data_ptr())),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(0),  # ncclInt8 (1 byte)
                ctypes.c_int(int(peer_rank)),
                self.comm,
                ctypes.c_void_p(int(stream_ptr)),
            ),
            "ncclRecv",
        )

    def close(self) -> None:
        if getattr(self, "comm", None) and self.comm.value:
            try:
                self._nccl.ncclCommDestroy(self.comm)
            except Exception:
                pass
            self.comm = ctypes.c_void_p(0)
        if getattr(self, "_peer_sock", None) is not None:
            try:
                self._peer_sock.close()
            except Exception:
                pass
            self._peer_sock = None
        if getattr(self, "_listen_sock", None) is not None:
            try:
                self._listen_sock.close()
            except Exception:
                pass
            self._listen_sock = None


class NcclStateShipper:
    """Packs and ships cross-paradigm CPTG state (`NaiveCache` KV or bridge tensors) over NCCL P2P."""

    def __init__(self, channel: DirectNcclP2PChannel | None = None, group: Any | None = None) -> None:
        self.channel = channel
        self.group = group

    def _send_u8_header(self, payload: dict[str, Any], dst_rank: int, device: Any) -> None:
        if self.channel is not None:
            self.channel.send_json(payload)
            return
        import json
        import torch
        import torch.distributed as dist

        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        len_t = torch.tensor([len(raw)], dtype=torch.int64, device=device)
        dist.send(len_t, dst=dst_rank, group=self.group)
        if raw:
            buf = torch.tensor(list(raw), dtype=torch.uint8, device=device)
            dist.send(buf, dst=dst_rank, group=self.group)

    def _recv_u8_header(self, src_rank: int, device: Any) -> dict[str, Any]:
        if self.channel is not None:
            return dict(self.channel.recv_json() or {})
        import json
        import torch
        import torch.distributed as dist

        len_t = torch.empty((1,), dtype=torch.int64, device=device)
        dist.recv(len_t, src=src_rank, group=self.group)
        n = int(len_t.item())
        if n == 0:
            return {}
        buf = torch.empty((n,), dtype=torch.uint8, device=device)
        dist.recv(buf, src=src_rank, group=self.group)
        return json.loads(bytes(buf.cpu().tolist()).decode("utf-8"))

    def send_bagel_context(
        self,
        ctx: dict[str, Any],
        dst_rank: int,
        sync: bool = True,
    ) -> ShipMetrics:
        """Pack all `(key_cache[l], value_cache[l])` layers into a single contiguous NCCL P2P send."""
        import torch
        import torch.distributed as dist

        pkv = ctx["past_key_values"]
        num_layers = int(pkv.num_layers)
        get_k = pkv.key_cache.get if isinstance(pkv.key_cache, dict) else (lambda i: pkv.key_cache[i])
        get_v = pkv.value_cache.get if isinstance(pkv.value_cache, dict) else (lambda i: pkv.value_cache[i])
        sample_k = next((get_k(l) for l in range(num_layers) if get_k(l) is not None), None)
        device = sample_k.device if sample_k is not None else torch.device(f"cuda:{torch.cuda.current_device()}")

        layer_specs: list[list[Any] | None] = []
        chunks: list[Any] = []
        for l in range(num_layers):
            k = get_k(l)
            v = get_v(l)
            if k is None or v is None:
                layer_specs.append(None)
                continue
            layer_specs.append([list(k.shape), list(v.shape), str(k.dtype).replace("torch.", "")])
            chunks.append(k.contiguous().view(torch.uint8).reshape(-1))
            chunks.append(v.contiguous().view(torch.uint8).reshape(-1))

        flat = torch.cat(chunks, dim=0) if chunks else torch.empty((0,), dtype=torch.uint8, device=device)
        meta = {
            "num_layers": num_layers,
            "kv_lens": list(ctx["kv_lens"]),
            "ropes": list(ctx["ropes"]),
            "total_bytes": int(flat.numel()),
            "layers": layer_specs,
        }
        if sync:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        self._send_u8_header(meta, dst_rank=dst_rank, device=device)
        if flat.numel() > 0:
            if self.channel is not None:
                stream_ptr = int(torch.cuda.current_stream(device).cuda_stream)
                self.channel.send_tensor_u8(flat, peer_rank=dst_rank, stream_ptr=stream_ptr)
            else:
                dist.send(flat, dst=dst_rank, group=self.group)
        if sync:
            torch.cuda.synchronize(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        gbps = (float(flat.numel()) / 1e9) / max(dt_ms / 1000.0, 1e-6)
        return ShipMetrics(
            bytes_transferred=int(flat.numel()),
            transfer_ms=round(dt_ms, 3),
            bandwidth_gbps=round(gbps, 3),
            num_layers=num_layers,
        )

    def recv_bagel_context(
        self,
        src_rank: int,
        naive_cache_cls: Any,
        device: Any,
        sync: bool = True,
    ) -> tuple[dict[str, Any], ShipMetrics]:
        """Receive packed NCCL P2P KV payload and reconstruct exact `NaiveCache` context on `device`."""
        import torch
        import torch.distributed as dist

        if sync:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        meta = self._recv_u8_header(src_rank=src_rank, device=device)
        total_bytes = int(meta["total_bytes"])
        flat = torch.empty((total_bytes,), dtype=torch.uint8, device=device)
        if total_bytes > 0:
            if self.channel is not None:
                stream_ptr = int(torch.cuda.current_stream(device).cuda_stream)
                self.channel.recv_tensor_u8(flat, peer_rank=src_rank, stream_ptr=stream_ptr)
            else:
                dist.recv(flat, src=src_rank, group=self.group)

        num_layers = int(meta["num_layers"])
        pkv = naive_cache_cls(num_layers)
        off = 0
        for l, spec in enumerate(meta["layers"]):
            if spec is None:
                continue
            k_shape, v_shape, dt_name = spec
            dtype = getattr(torch, dt_name)
            elem_sz = int(torch.empty((), dtype=dtype).element_size())
            k_bytes = int(torch.Size(k_shape).numel() * elem_sz)
            v_bytes = int(torch.Size(v_shape).numel() * elem_sz)
            pkv.key_cache[l] = flat[off : off + k_bytes].view(dtype).reshape(k_shape)
            off += k_bytes
            pkv.value_cache[l] = flat[off : off + v_bytes].view(dtype).reshape(v_shape)
            off += v_bytes

        if sync:
            torch.cuda.synchronize(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        gbps = (float(total_bytes) / 1e9) / max(dt_ms / 1000.0, 1e-6)
        ctx = {"kv_lens": meta["kv_lens"], "ropes": meta["ropes"], "past_key_values": pkv}
        return ctx, ShipMetrics(
            bytes_transferred=total_bytes,
            transfer_ms=round(dt_ms, 3),
            bandwidth_gbps=round(gbps, 3),
            num_layers=num_layers,
        )


class DirectNcclMesh:
    """Multi-rank NCCL P2P communicator with dedicated transfer stream (bypasses `torch.distributed`)."""

    def __init__(self, rank: int, world_size: int, uid_bytes: bytes, device_index: int) -> None:
        import torch

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device_index = int(device_index)
        self.device = torch.device(f"cuda:{self.device_index}")
        torch.cuda.set_device(self.device)
        self._nccl = _load_libnccl()
        self._bind_nccl_symbols()
        self.comm = ctypes.c_void_p(0)
        uid = _NcclUniqueId.from_buffer_copy(uid_bytes)
        self._check(
            self._nccl.ncclCommInitRank(ctypes.byref(self.comm), ctypes.c_int(self.world_size), uid, ctypes.c_int(self.rank)),
            "ncclCommInitRank",
        )
        self.transfer_stream = torch.cuda.Stream(device=self.device)

    @staticmethod
    def generate_unique_id_bytes() -> bytes:
        nccl = _load_libnccl()
        nccl.ncclGetUniqueId.argtypes = [ctypes.POINTER(_NcclUniqueId)]
        nccl.ncclGetUniqueId.restype = ctypes.c_int
        uid = _NcclUniqueId()
        rc = int(nccl.ncclGetUniqueId(ctypes.byref(uid)))
        if rc != 0:
            raise RuntimeError(f"ncclGetUniqueId failed with ncclResult_t={rc}")
        return bytes(uid)

    def _bind_nccl_symbols(self) -> None:
        self._nccl.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _NcclUniqueId, ctypes.c_int]
        self._nccl.ncclCommInitRank.restype = ctypes.c_int
        self._nccl.ncclCommDestroy.argtypes = [ctypes.c_void_p]
        self._nccl.ncclCommDestroy.restype = ctypes.c_int
        self._nccl.ncclSend.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self._nccl.ncclSend.restype = ctypes.c_int
        self._nccl.ncclRecv.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self._nccl.ncclRecv.restype = ctypes.c_int

    def _check(self, rc: int, op: str) -> None:
        if int(rc) != 0:
            raise RuntimeError(f"{op} failed with ncclResult_t={rc}")

    @staticmethod
    def pack_bagel_context(ctx: dict[str, Any], device: Any = None) -> tuple[dict[str, Any], Any]:
        import torch

        pkv = ctx["past_key_values"]
        num_layers = int(pkv.num_layers)
        get_k = pkv.key_cache.get if isinstance(pkv.key_cache, dict) else (lambda i: pkv.key_cache[i])
        get_v = pkv.value_cache.get if isinstance(pkv.value_cache, dict) else (lambda i: pkv.value_cache[i])
        sample_k = next((get_k(l) for l in range(num_layers) if get_k(l) is not None), None)
        if device is None:
            device = sample_k.device if sample_k is not None else torch.device(f"cuda:{torch.cuda.current_device()}")

        layer_specs: list[list[Any] | None] = []
        chunks: list[Any] = []
        for l in range(num_layers):
            k = get_k(l)
            v = get_v(l)
            if k is None or v is None:
                layer_specs.append(None)
                continue
            layer_specs.append([list(k.shape), list(v.shape), str(k.dtype).replace("torch.", "")])
            chunks.append(k.contiguous().view(torch.uint8).reshape(-1))
            chunks.append(v.contiguous().view(torch.uint8).reshape(-1))
        flat = torch.cat(chunks, dim=0) if chunks else torch.empty((0,), dtype=torch.uint8, device=device)
        meta = {
            "num_layers": num_layers,
            "kv_lens": list(ctx["kv_lens"]),
            "ropes": list(ctx["ropes"]),
            "total_bytes": int(flat.numel()),
            "layers": layer_specs,
        }
        return meta, flat

    @staticmethod
    def unpack_bagel_context(meta: dict[str, Any], flat_u8: Any, naive_cache_cls: Any) -> dict[str, Any]:
        import torch

        num_layers = int(meta["num_layers"])
        pkv = naive_cache_cls(num_layers)
        off = 0
        for l, spec in enumerate(meta["layers"]):
            if spec is None:
                continue
            k_shape, v_shape, dt_name = spec
            dtype = getattr(torch, dt_name)
            elem_sz = int(torch.empty((), dtype=dtype).element_size())
            k_bytes = int(torch.Size(k_shape).numel() * elem_sz)
            v_bytes = int(torch.Size(v_shape).numel() * elem_sz)
            pkv.key_cache[l] = flat_u8[off : off + k_bytes].view(dtype).reshape(k_shape).clone()
            off += k_bytes
            pkv.value_cache[l] = flat_u8[off : off + v_bytes].view(dtype).reshape(v_shape).clone()
            off += v_bytes
        return {"kv_lens": list(meta["kv_lens"]), "ropes": list(meta["ropes"]), "past_key_values": pkv}

    def send_u8(self, tensor_u8: Any, dst_rank: int, sync_stream: bool = True) -> float:
        import torch

        nbytes = int(tensor_u8.numel())
        t0 = time.perf_counter()
        if nbytes > 0:
            ev = torch.cuda.current_stream(self.device).record_event()
            self.transfer_stream.wait_event(ev)
            self._check(
                self._nccl.ncclSend(
                    ctypes.c_void_p(int(tensor_u8.data_ptr())),
                    ctypes.c_size_t(nbytes),
                    ctypes.c_int(0),
                    ctypes.c_int(int(dst_rank)),
                    self.comm,
                    ctypes.c_void_p(int(self.transfer_stream.cuda_stream)),
                ),
                "ncclSend",
            )
            if sync_stream:
                self.transfer_stream.synchronize()
        return round((time.perf_counter() - t0) * 1000.0, 3)

    def recv_u8(self, nbytes: int, src_rank: int, sync_stream: bool = True) -> tuple[Any, float]:
        import torch

        t0 = time.perf_counter()
        flat = torch.empty((int(nbytes),), dtype=torch.uint8, device=self.device)
        if nbytes > 0:
            self._check(
                self._nccl.ncclRecv(
                    ctypes.c_void_p(int(flat.data_ptr())),
                    ctypes.c_size_t(int(nbytes)),
                    ctypes.c_int(0),
                    ctypes.c_int(int(src_rank)),
                    self.comm,
                    ctypes.c_void_p(int(self.transfer_stream.cuda_stream)),
                ),
                "ncclRecv",
            )
            if sync_stream:
                self.transfer_stream.synchronize()
        return flat, round((time.perf_counter() - t0) * 1000.0, 3)

    def close(self) -> None:
        if getattr(self, "comm", None) and self.comm.value:
            try:
                self._nccl.ncclCommDestroy(self.comm)
            except Exception:
                pass
            self.comm = ctypes.c_void_p(0)


__all__ = [
    "DirectNcclMesh",
    "DirectNcclP2PChannel",
    "NcclStateShipper",
    "SharedNumaMemfdMirror",
    "ShipMetrics",
    "find_free_tcp_port",
]
