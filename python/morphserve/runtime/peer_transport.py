"""Unified Python `PeerByteTransport` adapters backed by C++17 `morphserve::runtime::PeerByteTransport`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
import os
from pathlib import Path
import socket
from typing import Any

from morphserve._core import get_core_lib, last_error


def find_free_tcp_port(host: str = "127.0.0.1") -> int:
    """Bind an ephemeral TCP port on `host` and return the free port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((str(host), 0))
        return int(sock.getsockname()[1])


def _nccl_hint_path(hint_lib_path: str | Path | None = None) -> bytes | None:
    if hint_lib_path is not None:
        return str(hint_lib_path).encode("utf-8")
    try:
        import torch

        torch_dir = os.path.dirname(torch.__file__)
        cand = os.path.join(os.path.dirname(torch_dir), "nvidia", "nccl", "lib", "libnccl.so.2")
        if os.path.exists(cand):
            return cand.encode("utf-8")
    except Exception:
        pass
    return None


def generate_nccl_unique_id(hint_lib_path: str | Path | None = None) -> bytes:
    """Generate a 128-byte `ncclUniqueId` via C++17 `NcclPeerAdapter::generate_unique_id`."""
    L = get_core_lib()
    uid_arr = (ctypes.c_uint8 * 128)()
    rc = int(L.ms_nccl_generate_uid(_nccl_hint_path(hint_lib_path), uid_arr))
    if rc != 0:
        raise RuntimeError(f"ms_nccl_generate_uid failed: {last_error(L)}")
    return bytes(uid_arr)


class PeerByteTransport:
    """Base RAII wrapper over C++17 `ms_transport_t` (`ms_transport_send`/`recv`/`close`/`destroy`)."""

    _transport_kind: str = "transport"
    _send_error_cls: type[Exception] = RuntimeError
    _recv_error_cls: type[Exception] = RuntimeError

    def __init__(self, rank: int, world_size: int, handle: int, lib: ctypes.CDLL) -> None:
        if handle == 0:
            raise RuntimeError(f"{self._transport_kind} handle must be non-null")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._lib = lib
        self._handle = int(handle)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "PeerByteTransport":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _ensure_open(self) -> int:
        h = getattr(self, "_handle", 0)
        if not h:
            raise self._send_error_cls(f"{self._transport_kind} transport is closed")
        return h

    def _send_ptr(
        self,
        src_ptr: int,
        nbytes: int,
        dst_rank: int,
        *,
        cuda_stream: int = 0,
        sync_stream: bool = True,
    ) -> float:
        h = self._ensure_open()
        elapsed = float(
            self._lib.ms_transport_send(
                ctypes.c_void_p(h),
                ctypes.c_void_p(int(src_ptr)),
                ctypes.c_size_t(int(nbytes)),
                ctypes.c_int(int(dst_rank)),
                ctypes.c_void_p(int(cuda_stream)),
                1 if sync_stream else 0,
            )
        )
        if elapsed < 0.0:
            raise self._send_error_cls(
                f"ms_transport_send({self._transport_kind}) failed: {last_error(self._lib)}"
            )
        return round(elapsed, 3)

    def _recv_ptr(
        self,
        dst_ptr: int,
        nbytes: int,
        src_rank: int,
        *,
        cuda_stream: int = 0,
        sync_stream: bool = True,
    ) -> tuple[int, float]:
        h = self._ensure_open()
        actual = ctypes.c_size_t(0)
        elapsed = ctypes.c_double(0.0)
        rc = int(
            self._lib.ms_transport_recv(
                ctypes.c_void_p(h),
                ctypes.c_void_p(int(dst_ptr)),
                ctypes.c_size_t(int(nbytes)),
                ctypes.c_int(int(src_rank)),
                ctypes.c_void_p(int(cuda_stream)),
                1 if sync_stream else 0,
                ctypes.byref(actual),
                ctypes.byref(elapsed),
            )
        )
        if rc != 0:
            raise self._recv_error_cls(
                f"ms_transport_recv({self._transport_kind}) failed: {last_error(self._lib)}"
            )
        return int(actual.value), round(float(elapsed.value), 3)

    def alloc_recv_u8(self, nbytes: int) -> Any:
        return bytearray(max(0, int(nbytes)))

    def send_u8(
        self,
        payload: Any,
        dst_rank: int,
        sync_stream: bool = True,
        *,
        ready_event: Any = None,
        cuda_stream: int | None = None,
    ) -> float:
        del ready_event
        raw = bytes(payload)
        buf = ctypes.create_string_buffer(raw, len(raw)) if raw else None
        src_ptr = ctypes.addressof(buf) if buf is not None else 0
        return self._send_ptr(
            src_ptr,
            len(raw),
            int(dst_rank),
            cuda_stream=int(cuda_stream or 0),
            sync_stream=sync_stream,
        )

    def recv_u8(
        self,
        nbytes: int,
        src_rank: int,
        sync_stream: bool = True,
        *,
        out_tensor: Any = None,
        ready_event: Any = None,
        cuda_stream: int | None = None,
    ) -> tuple[Any, float]:
        nb = max(0, int(nbytes))
        c_buf = (ctypes.c_uint8 * max(1, nb))()
        dst_ptr = ctypes.addressof(c_buf) if nb > 0 else 0
        actual, elapsed_ms = self._recv_ptr(
            dst_ptr,
            nb,
            int(src_rank),
            cuda_stream=int(cuda_stream or 0),
            sync_stream=sync_stream,
        )
        got = min(nb, actual)
        sliced = ctypes.string_at(ctypes.addressof(c_buf), got) if got > 0 else b""
        if isinstance(out_tensor, (bytearray, memoryview)):
            if len(out_tensor) < len(sliced):
                raise ValueError(f"out_tensor len {len(out_tensor)} < recv nbytes {len(sliced)}")
            out_tensor[: len(sliced)] = sliced
            return out_tensor, elapsed_ms
        return sliced, elapsed_ms

    def close(self) -> None:
        h = getattr(self, "_handle", 0)
        if h:
            self._handle = 0
            self._lib.ms_transport_close(ctypes.c_void_p(h))
            self._lib.ms_transport_destroy(ctypes.c_void_p(h))


class NcclPeerAdapter(PeerByteTransport):
    """Multi-rank NCCL P2P byte communicator backed by C++17 `NcclPeerAdapter` & its internal CUDA transfer stream."""

    _transport_kind = "nccl"
    _send_error_cls = RuntimeError
    _recv_error_cls = RuntimeError

    def __init__(
        self,
        rank: int,
        world_size: int,
        uid_bytes: bytes,
        device_index: int,
        *,
        hint_lib_path: str | Path | None = None,
    ) -> None:
        import torch

        if len(uid_bytes) != 128:
            raise ValueError(f"expected 128-byte ncclUniqueId, got {len(uid_bytes)}")
        self.device_index = int(device_index)
        self.device = torch.device(f"cuda:{self.device_index}")
        torch.cuda.set_device(self.device)

        L = get_core_lib()
        uid_arr = (ctypes.c_uint8 * 128).from_buffer_copy(uid_bytes)
        h = int(
            L.ms_nccl_create(
                ctypes.c_int(int(rank)),
                ctypes.c_int(int(world_size)),
                uid_arr,
                ctypes.c_int(self.device_index),
                _nccl_hint_path(hint_lib_path),
            )
            or 0
        )
        if h == 0:
            raise RuntimeError(f"ms_nccl_create failed: {last_error(L)}")
        super().__init__(rank=rank, world_size=world_size, handle=h, lib=L)

    @staticmethod
    def generate_unique_id_bytes(hint_lib_path: str | Path | None = None) -> bytes:
        return generate_nccl_unique_id(hint_lib_path=hint_lib_path)

    def alloc_recv_u8(self, nbytes: int) -> Any:
        import torch

        torch.cuda.set_device(self.device)
        return torch.empty((max(0, int(nbytes)),), dtype=torch.uint8, device=self.device)

    def send_u8(
        self,
        tensor_u8: Any,
        dst_rank: int,
        sync_stream: bool = True,
        *,
        ready_event: Any = None,
        cuda_stream: int | None = None,
    ) -> float:
        import torch

        torch.cuda.set_device(self.device)
        nbytes = int(tensor_u8.numel()) * int(tensor_u8.element_size()) if hasattr(tensor_u8, "element_size") else int(tensor_u8.numel())
        if nbytes <= 0:
            return 0.0
        stream_ptr = int(cuda_stream) if cuda_stream is not None else 0
        if ready_event is not None:
            ready_event.synchronize()
        elif stream_ptr == 0 and getattr(tensor_u8, "is_cuda", False):
            torch.cuda.current_stream(self.device).synchronize()
        # Passing `cuda_stream=0` uses C++ `NcclPeerAdapter::transfer_stream_` & `comm_mu_` directly.
        return self._send_ptr(
            int(tensor_u8.data_ptr()),
            nbytes,
            int(dst_rank),
            cuda_stream=stream_ptr,
            sync_stream=sync_stream,
        )

    def recv_u8(
        self,
        nbytes: int,
        src_rank: int,
        sync_stream: bool = True,
        *,
        out_tensor: Any = None,
        ready_event: Any = None,
        cuda_stream: int | None = None,
    ) -> tuple[Any, float]:
        import torch

        torch.cuda.set_device(self.device)
        nb = max(0, int(nbytes))
        if out_tensor is not None:
            if int(out_tensor.numel()) < nb:
                raise ValueError(f"out_tensor numel {int(out_tensor.numel())} < requested recv nbytes {nb}")
            flat = out_tensor
        else:
            flat = torch.empty((nb,), dtype=torch.uint8, device=self.device)
        if nb <= 0:
            return flat, 0.0
        stream_ptr = int(cuda_stream) if cuda_stream is not None else 0
        # Ensure any prior default-stream kernel that previously owned `flat`'s recycled
        # CUDACachingAllocator block has completed before `ncclRecv` writes on `transfer_stream_`.
        if ready_event is not None:
            ready_event.synchronize()
        elif stream_ptr == 0 and getattr(flat, "is_cuda", False):
            torch.cuda.default_stream(self.device).synchronize()
        _, elapsed_ms = self._recv_ptr(
            int(flat.data_ptr()),
            nb,
            int(src_rank),
            cuda_stream=stream_ptr,
            sync_stream=sync_stream,
        )
        return flat, elapsed_ms

    def send_tensor(
        self,
        tensor: Any,
        dst_rank: int,
        sync_stream: bool = True,
        *,
        ready_event: Any = None,
        cuda_stream: int | None = None,
    ) -> float:
        import torch

        t_c = tensor.to(self.device).contiguous()
        flat_u8 = t_c.view(torch.uint8).reshape(-1)
        return self.send_u8(
            flat_u8,
            dst_rank=dst_rank,
            sync_stream=sync_stream,
            ready_event=ready_event,
            cuda_stream=cuda_stream,
        )

    def recv_tensor(
        self,
        shape: Sequence[int],
        dtype: Any,
        src_rank: int,
        sync_stream: bool = True,
        *,
        cuda_stream: int | None = None,
    ) -> tuple[Any, float]:
        import torch

        out = torch.empty(tuple(int(d) for d in shape), dtype=dtype, device=self.device)
        nbytes = int(out.numel() * out.element_size())
        if nbytes <= 0:
            return out, 0.0
        _, elapsed_ms = self.recv_u8(
            nbytes,
            src_rank=src_rank,
            sync_stream=sync_stream,
            out_tensor=out.view(torch.uint8).reshape(-1),
            cuda_stream=cuda_stream,
        )
        return out, elapsed_ms


DirectNcclMesh = NcclPeerAdapter


class TcpPeerEndpoint(PeerByteTransport):
    """Full-duplex POSIX `TCP_NODELAY` P2P byte endpoint backed by C++17 `TcpPeerAdapter`."""

    _transport_kind = "tcp"
    _send_error_cls = ConnectionError
    _recv_error_cls = ConnectionError

    def __init__(self, rank: int, world_size: int, bind_host: str = "127.0.0.1") -> None:
        self.bind_host = str(bind_host)
        L = get_core_lib()
        h = int(
            L.ms_tcp_create(
                ctypes.c_int(int(rank)),
                ctypes.c_int(int(world_size)),
                self.bind_host.encode("utf-8"),
            )
            or 0
        )
        if h == 0:
            raise OSError(f"ms_tcp_create failed: {last_error(L)}")
        super().__init__(rank=rank, world_size=world_size, handle=h, lib=L)
        self.bound_port = int(L.ms_tcp_bound_port(ctypes.c_void_p(h)))

    def set_directory(self, directory: Mapping[int | str, Any]) -> None:
        h = self._ensure_open()
        for r, hp in directory.items():
            rk = int(r)
            if rk != self.rank:
                rc = int(
                    self._lib.ms_tcp_set_peer(
                        ctypes.c_void_p(h),
                        ctypes.c_int(rk),
                        str(hp[0]).encode("utf-8"),
                        ctypes.c_int(int(hp[1])),
                    )
                )
                if rc != 0:
                    raise OSError(f"ms_tcp_set_peer({rk}) failed: {last_error(self._lib)}")


class TcpByteMesh:
    """Convenience N-rank `TcpPeerEndpoint` full mesh for single-host TCP transport tests."""

    def __init__(self, world_size: int, bind_host: str = "127.0.0.1") -> None:
        self.world_size = int(world_size)
        self.endpoints = [TcpPeerEndpoint(r, self.world_size, bind_host=bind_host) for r in range(self.world_size)]
        directory = {r: (bind_host, ep.bound_port) for r, ep in enumerate(self.endpoints)}
        for ep in self.endpoints:
            ep.set_directory(directory)

    def __enter__(self) -> "TcpByteMesh":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def endpoint_for(self, rank: int) -> TcpPeerEndpoint:
        return self.endpoints[int(rank)]

    def close(self) -> None:
        for ep in self.endpoints:
            ep.close()


class InMemoryByteBus:
    """Thread-safe pairwise FIFO byte bus and named SHM store backed by C++17 `InMemoryTransportBus`."""

    def __init__(self, world_size: int) -> None:
        self.world_size = max(2, int(world_size))
        self._lib = get_core_lib()
        h = int(self._lib.ms_mem_bus_create(ctypes.c_int(self.world_size)) or 0)
        if h == 0:
            raise RuntimeError(f"ms_mem_bus_create failed: {last_error(self._lib)}")
        self._handle = h

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "InMemoryByteBus":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def endpoint_for(self, rank: int) -> "InMemoryPeerMesh":
        return InMemoryPeerMesh(self, rank)

    def put_shm(self, name: str, data: bytes) -> None:
        raw = bytes(data)
        buf = ctypes.create_string_buffer(raw, len(raw)) if raw else None
        rc = int(
            self._lib.ms_mem_bus_put_shm(
                ctypes.c_void_p(self._handle),
                name.encode("utf-8"),
                ctypes.cast(buf, ctypes.c_void_p) if buf is not None else ctypes.c_void_p(0),
                ctypes.c_size_t(len(raw)),
            )
        )
        if rc != 0:
            raise RuntimeError(f"ms_mem_bus_put_shm failed: {last_error(self._lib)}")

    def pop_shm(self, name: str) -> bytes:
        out_ptr = ctypes.c_void_p(0)
        sz = int(
            self._lib.ms_mem_bus_pop_shm_size(
                ctypes.c_void_p(self._handle),
                name.encode("utf-8"),
                ctypes.byref(out_ptr),
            )
        )
        if sz <= 0 or not out_ptr.value:
            return b""
        try:
            return ctypes.string_at(out_ptr.value, sz)
        finally:
            self._lib.ms_free_bytes(out_ptr)

    def close(self) -> None:
        h = getattr(self, "_handle", 0)
        if h:
            self._handle = 0
            self._lib.ms_mem_bus_destroy(ctypes.c_void_p(h))


class InMemoryPeerMesh(PeerByteTransport):
    """In-process pairwise byte endpoint backed by C++17 `InMemoryPeerAdapter`."""

    _transport_kind = "mem"
    _send_error_cls = RuntimeError
    _recv_error_cls = TimeoutError

    def __init__(self, bus: InMemoryByteBus, rank: int) -> None:
        self.bus = bus
        L = bus._lib
        h = int(L.ms_mem_bus_endpoint(ctypes.c_void_p(bus._handle), ctypes.c_int(int(rank))) or 0)
        if h == 0:
            raise RuntimeError(f"ms_mem_bus_endpoint failed: {last_error(L)}")
        super().__init__(rank=rank, world_size=bus.world_size, handle=h, lib=L)


__all__ = [
    "DirectNcclMesh",
    "InMemoryByteBus",
    "InMemoryPeerMesh",
    "NcclPeerAdapter",
    "PeerByteTransport",
    "TcpByteMesh",
    "TcpPeerEndpoint",
    "find_free_tcp_port",
    "generate_nccl_unique_id",
]
