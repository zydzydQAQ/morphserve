"""Standalone N-rank NCCL P2P byte transport (`DirectNcclMesh`) backed by C++17 `NcclPeerAdapter`."""

from __future__ import annotations

import ctypes
import os
import threading
from typing import Any

from morphserve._core import get_core_lib, last_error


def find_free_tcp_port() -> int:
    L = get_core_lib()
    port = int(L.ms_find_free_tcp_port())
    if port <= 0:
        raise OSError(f"ms_find_free_tcp_port failed: {last_error(L)}")
    return port


def _nccl_hint_path() -> bytes | None:
    try:
        import torch

        torch_dir = os.path.dirname(torch.__file__)
        cand = os.path.join(os.path.dirname(torch_dir), "nvidia", "nccl", "lib", "libnccl.so.2")
        if os.path.exists(cand):
            return cand.encode("utf-8")
    except Exception:
        pass
    return None


class DirectNcclMesh:
    """Multi-rank NCCL P2P raw-byte communicator isolated from `torch.distributed` via C++ `NcclPeerAdapter`."""

    def __init__(self, rank: int, world_size: int, uid_bytes: bytes, device_index: int) -> None:
        import torch

        if len(uid_bytes) != 128:
            raise ValueError(f"expected 128-byte ncclUniqueId, got {len(uid_bytes)}")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device_index = int(device_index)
        self.device = torch.device(f"cuda:{self.device_index}")
        torch.cuda.set_device(self.device)
        self._lib = get_core_lib()

        uid_arr = (ctypes.c_uint8 * 128).from_buffer_copy(uid_bytes)
        h = int(
            self._lib.ms_nccl_create(
                ctypes.c_int(self.rank),
                ctypes.c_int(self.world_size),
                uid_arr,
                ctypes.c_int(self.device_index),
                _nccl_hint_path(),
            )
            or 0
        )
        if h == 0:
            raise RuntimeError(f"ms_nccl_create failed: {last_error(self._lib)}")
        self._handle = h
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self._comm_lock = threading.Lock()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def generate_unique_id_bytes() -> bytes:
        L = get_core_lib()
        uid_arr = (ctypes.c_uint8 * 128)()
        rc = int(L.ms_nccl_generate_uid(_nccl_hint_path(), uid_arr))
        if rc != 0:
            raise RuntimeError(f"ms_nccl_generate_uid failed: {last_error(L)}")
        return bytes(uid_arr)

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
        if nbytes <= 0:
            return 0.0
        with self._comm_lock:
            if ready_event is not None:
                self.transfer_stream.wait_event(ready_event)
            else:
                ev = torch.cuda.current_stream(self.device).record_event()
                self.transfer_stream.wait_event(ev)
            if hasattr(tensor_u8, "record_stream"):
                tensor_u8.record_stream(self.transfer_stream)
            elapsed = float(
                self._lib.ms_transport_send(
                    ctypes.c_void_p(self._handle),
                    ctypes.c_void_p(int(tensor_u8.data_ptr())),
                    ctypes.c_size_t(nbytes),
                    ctypes.c_int(int(dst_rank)),
                    ctypes.c_void_p(int(self.transfer_stream.cuda_stream)),
                    1 if sync_stream else 0,
                )
            )
            if elapsed < 0.0:
                raise RuntimeError(f"ms_transport_send(nccl) failed: {last_error(self._lib)}")
            return round(elapsed, 3)

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
        nb = max(0, int(nbytes))
        if out_tensor is not None:
            if int(out_tensor.numel()) < nb:
                raise ValueError(f"out_tensor numel {int(out_tensor.numel())} < requested recv nbytes {nb}")
            flat = out_tensor
        else:
            with torch.cuda.stream(self.transfer_stream):
                flat = torch.empty((nb,), dtype=torch.uint8, device=self.device)
        if nb <= 0:
            return flat, 0.0
        with self._comm_lock:
            if hasattr(flat, "record_stream"):
                flat.record_stream(self.transfer_stream)
            actual = ctypes.c_size_t(0)
            elapsed = ctypes.c_double(0.0)
            rc = int(
                self._lib.ms_transport_recv(
                    ctypes.c_void_p(self._handle),
                    ctypes.c_void_p(int(flat.data_ptr())),
                    ctypes.c_size_t(nb),
                    ctypes.c_int(int(src_rank)),
                    ctypes.c_void_p(int(self.transfer_stream.cuda_stream)),
                    1 if sync_stream else 0,
                    ctypes.byref(actual),
                    ctypes.byref(elapsed),
                )
            )
            if rc != 0:
                raise RuntimeError(f"ms_transport_recv(nccl) failed: {last_error(self._lib)}")
            return flat, round(float(elapsed.value), 3)

    def close(self) -> None:
        h = getattr(self, "_handle", 0)
        if h:
            self._handle = 0
            self._lib.ms_transport_close(ctypes.c_void_p(h))
            self._lib.ms_transport_destroy(ctypes.c_void_p(h))


__all__ = ["DirectNcclMesh", "find_free_tcp_port"]
