"""Standalone N-rank NCCL P2P byte transport (`DirectNcclMesh`) on a dedicated CUDA transfer stream."""

from __future__ import annotations

import ctypes
import os
import socket
import threading
import time
from typing import Any


def find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _load_libnccl() -> ctypes.CDLL:
    import ctypes.util

    try:
        import torch

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
    raise RuntimeError("Could not locate libnccl.so.2 for DirectNcclMesh")


class _NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


class DirectNcclMesh:
    """Multi-rank NCCL P2P raw-byte communicator isolated from `torch.distributed` global groups."""

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
            self._nccl.ncclCommInitRank(
                ctypes.byref(self.comm), ctypes.c_int(self.world_size), uid, ctypes.c_int(self.rank)
            ),
            "ncclCommInitRank",
        )
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self._comm_lock = threading.Lock()

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
        self._nccl.ncclCommInitRank.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            _NcclUniqueId,
            ctypes.c_int,
        ]
        self._nccl.ncclCommInitRank.restype = ctypes.c_int
        self._nccl.ncclCommDestroy.argtypes = [ctypes.c_void_p]
        self._nccl.ncclCommDestroy.restype = ctypes.c_int
        self._nccl.ncclSend.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._nccl.ncclSend.restype = ctypes.c_int
        self._nccl.ncclRecv.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._nccl.ncclRecv.restype = ctypes.c_int

    def _check(self, rc: int, op: str) -> None:
        if int(rc) != 0:
            raise RuntimeError(f"{op} failed with ncclResult_t={rc}")

    def alloc_recv_u8(self, nbytes: int) -> Any:
        import torch

        torch.cuda.set_device(self.device)
        with torch.cuda.stream(self.transfer_stream):
            return torch.empty((max(0, int(nbytes)),), dtype=torch.uint8, device=self.device)

    def send_u8(
        self,
        tensor_u8: Any,
        dst_rank: int,
        sync_stream: bool = True,
        *,
        ready_event: Any = None,
    ) -> float:
        import torch

        torch.cuda.set_device(self.device)
        nbytes = int(tensor_u8.numel())
        t0 = time.perf_counter()
        if nbytes > 0:
            with self._comm_lock:
                if ready_event is not None:
                    self.transfer_stream.wait_event(ready_event)
                else:
                    ev = torch.cuda.current_stream(self.device).record_event()
                    self.transfer_stream.wait_event(ev)
                if hasattr(tensor_u8, "record_stream"):
                    tensor_u8.record_stream(self.transfer_stream)
                self._check(
                    self._nccl.ncclSend(
                        ctypes.c_void_p(int(tensor_u8.data_ptr())),
                        ctypes.c_size_t(nbytes),
                        ctypes.c_int(0),  # ncclInt8 (1 byte)
                        ctypes.c_int(int(dst_rank)),
                        self.comm,
                        ctypes.c_void_p(int(self.transfer_stream.cuda_stream)),
                    ),
                    "ncclSend",
                )
                if sync_stream:
                    self.transfer_stream.synchronize()
        return round((time.perf_counter() - t0) * 1000.0, 3)

    def recv_u8(
        self,
        nbytes: int,
        src_rank: int,
        sync_stream: bool = True,
        *,
        out_tensor: Any = None,
    ) -> tuple[Any, float]:
        import torch

        torch.cuda.set_device(self.device)
        t0 = time.perf_counter()
        nb = max(0, int(nbytes))
        if out_tensor is not None:
            if int(out_tensor.numel()) < nb:
                raise ValueError(f"out_tensor numel {int(out_tensor.numel())} < requested recv nbytes {nb}")
            flat = out_tensor
        else:
            with torch.cuda.stream(self.transfer_stream):
                flat = torch.empty((nb,), dtype=torch.uint8, device=self.device)
        if nb > 0:
            with self._comm_lock:
                if hasattr(flat, "record_stream"):
                    flat.record_stream(self.transfer_stream)
                self._check(
                    self._nccl.ncclRecv(
                        ctypes.c_void_p(int(flat.data_ptr())),
                        ctypes.c_size_t(nb),
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


__all__ = ["DirectNcclMesh", "find_free_tcp_port"]
