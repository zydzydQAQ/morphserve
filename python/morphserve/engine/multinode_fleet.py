"""Distributed TCP Coordinator + Multi-Node `WorkerFleet` (`2x8 = 16` GPUs or N-node TCP synthetic)."""

from __future__ import annotations

import ctypes
from dataclasses import asdict
import json
from pathlib import Path
import queue
import socket
import struct
import threading
import time
import traceback
from typing import Any, Callable, Mapping

from morphserve._core import get_core_lib, last_error
from morphserve.models import CPTGModelAdapter, SyntheticCPTGAdapter
from morphserve.router import ClusterTopologyProfile, DirectNcclMesh, HalfNodeH2DPacer
from morphserve.telemetry import read_shmem_kib
from morphserve.vmm import NumaMemfdMirror

from .config import EnginePolicyConfig
from .worker_proc import run_worker_process
from .worker_runtime import InMemoryByteBus, WorkerFleet, WorkerRuntime, query_worker_kv_free_bytes


def serialize_policy_config(policy: EnginePolicyConfig) -> dict[str, Any]:
    raw = asdict(policy)
    raw["initial_alphas"] = {str(k): float(v) for k, v in policy.initial_alphas.items()}
    raw["static_und_ranks"] = list(policy.static_und_ranks)
    raw["static_gen_ranks"] = list(policy.static_gen_ranks)
    raw["inter_turn_think_s_range"] = [
        float(policy.inter_turn_think_s_range[0]),
        float(policy.inter_turn_think_s_range[1]),
    ]
    raw["allow_cascade_cfg_split"] = bool(policy.allow_cascade_cfg_split)
    raw["think_time_dist"] = str(policy.think_time_dist)
    raw["think_time_median_s"] = float(policy.think_time_median_s)
    raw["think_time_sigma"] = float(policy.think_time_sigma)
    raw["ar_tpot_slo_mode"] = str(policy.ar_tpot_slo_mode)
    raw["steady_window_mode"] = str(policy.steady_window_mode)
    raw["warmup_window_s"] = float(policy.warmup_window_s)
    raw["steady_window_s"] = float(policy.steady_window_s)
    return raw


def deserialize_policy_config(data: Mapping[str, Any]) -> EnginePolicyConfig:
    return EnginePolicyConfig.from_dict(dict(data))


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError("unexpected EOF on multi-node TCP stream")
        buf.extend(chunk)
    return bytes(buf)


class _FramedJsonChannel:
    """Thread-safe full-duplex length-prefixed JSON over TCP (`TCP_NODELAY=1`)."""

    def __init__(self, sock: socket.socket) -> None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(None)
        self.sock = sock
        self._write_lock = threading.Lock()
        self._closed = False

    def send_json(self, obj: Mapping[str, Any]) -> None:
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        pkt = struct.pack("!I", len(raw)) + raw
        with self._write_lock:
            if self._closed:
                raise ConnectionError("TCP JSON channel already closed")
            self.sock.sendall(pkt)

    def recv_json(self) -> dict[str, Any]:
        hdr = _recv_exact(self.sock, 4)
        (nbytes,) = struct.unpack("!I", hdr)
        body = _recv_exact(self.sock, nbytes)
        return json.loads(body.decode("utf-8"))

    def close(self) -> None:
        with self._write_lock:
            if self._closed:
                return
            self._closed = True
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.sock.close()
            except OSError:
                pass


class TcpPeerEndpoint:
    """Real TCP P2P byte transport (`alloc_recv_u8`, `send_u8`, `recv_u8`) backed by C++17 `TcpPeerAdapter`."""

    def __init__(self, rank: int, world_size: int, bind_host: str = "127.0.0.1") -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.bind_host = str(bind_host)
        self._lib = get_core_lib()
        h = int(
            self._lib.ms_tcp_create(
                ctypes.c_int(self.rank),
                ctypes.c_int(self.world_size),
                self.bind_host.encode("utf-8"),
            )
            or 0
        )
        if h == 0:
            raise OSError(f"ms_tcp_create failed: {last_error(self._lib)}")
        self._handle = h
        self.bound_port = int(self._lib.ms_tcp_bound_port(ctypes.c_void_p(h)))

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def set_directory(self, directory: Mapping[int | str, Any]) -> None:
        for r, hp in directory.items():
            rk = int(r)
            if rk != self.rank:
                self._lib.ms_tcp_set_peer(
                    ctypes.c_void_p(self._handle),
                    ctypes.c_int(rk),
                    str(hp[0]).encode("utf-8"),
                    ctypes.c_int(int(hp[1])),
                )

    def alloc_recv_u8(self, nbytes: int) -> bytearray:
        return bytearray(max(0, int(nbytes)))

    def send_u8(
        self,
        payload: Any,
        dst_rank: int,
        sync_stream: bool = True,
        *,
        ready_event: Any = None,
    ) -> float:
        del ready_event
        raw = bytes(payload)
        buf = ctypes.create_string_buffer(raw, len(raw)) if raw else None
        ms = float(
            self._lib.ms_transport_send(
                ctypes.c_void_p(self._handle),
                ctypes.cast(buf, ctypes.c_void_p) if buf is not None else ctypes.c_void_p(0),
                ctypes.c_size_t(len(raw)),
                ctypes.c_int(int(dst_rank)),
                ctypes.c_void_p(0),
                1 if sync_stream else 0,
            )
        )
        if ms < 0.0:
            raise ConnectionError(f"ms_transport_send(tcp) failed: {last_error(self._lib)}")
        return round(ms, 3)

    def recv_u8(
        self,
        nbytes: int,
        src_rank: int,
        sync_stream: bool = True,
        *,
        out_tensor: Any = None,
    ) -> tuple[Any, float]:
        nb = max(0, int(nbytes))
        c_buf = (ctypes.c_uint8 * max(1, nb))()
        actual = ctypes.c_size_t(0)
        elapsed = ctypes.c_double(0.0)
        rc = int(
            self._lib.ms_transport_recv(
                ctypes.c_void_p(self._handle),
                ctypes.cast(c_buf, ctypes.c_void_p) if nb > 0 else ctypes.c_void_p(0),
                ctypes.c_size_t(nb),
                ctypes.c_int(int(src_rank)),
                ctypes.c_void_p(0),
                1 if sync_stream else 0,
                ctypes.byref(actual),
                ctypes.byref(elapsed),
            )
        )
        if rc != 0:
            raise ConnectionError(f"ms_transport_recv(tcp) failed: {last_error(self._lib)}")
        got = min(nb, int(actual.value))
        sliced = ctypes.string_at(ctypes.addressof(c_buf), got) if got > 0 else b""
        ms = round(float(elapsed.value), 3)
        if isinstance(out_tensor, (bytearray, memoryview)):
            if len(out_tensor) < len(sliced):
                raise ValueError(f"out_tensor len {len(out_tensor)} < recv nbytes {len(sliced)}")
            out_tensor[: len(sliced)] = sliced
            return out_tensor, ms
        return sliced, ms

    def close(self) -> None:
        h = getattr(self, "_handle", 0)
        if h:
            self._handle = 0
            self._lib.ms_transport_close(ctypes.c_void_p(h))
            self._lib.ms_transport_destroy(ctypes.c_void_p(h))


class TcpByteMesh:
    """Convenience local N-rank `TcpPeerEndpoint` factory for single-host TCP transport tests."""

    def __init__(self, world_size: int, bind_host: str = "127.0.0.1") -> None:
        self.endpoints = [TcpPeerEndpoint(r, world_size, bind_host=bind_host) for r in range(world_size)]
        directory = {r: (bind_host, ep.bound_port) for r, ep in enumerate(self.endpoints)}
        for ep in self.endpoints:
            ep.set_directory(directory)

    def endpoint_for(self, rank: int) -> TcpPeerEndpoint:
        return self.endpoints[int(rank)]

    def close(self) -> None:
        for ep in self.endpoints:
            ep.close()


class _LocalNodeWorkerHost:
    """Manages `gpus_per_node` global ranks `[rank_offset .. rank_offset + gpus_per_node - 1]` on one node."""

    def __init__(
        self,
        *,
        node_rank: int,
        gpus_per_node: int,
        policy: EnginePolicyConfig,
        model_dir: Path,
        bench_root: Path,
        backend: str,
        on_ack: Callable[[dict[str, Any]], None],
        advertise_host: str = "127.0.0.1",
        adapter_factory: Callable[[int, EnginePolicyConfig], CPTGModelAdapter] | None = None,
    ) -> None:
        self.node_rank = int(node_rank)
        self.gpus_per_node = int(gpus_per_node)
        self.rank_offset = self.node_rank * self.gpus_per_node
        self.global_ranks = tuple(range(self.rank_offset, self.rank_offset + self.gpus_per_node))
        self.policy = policy
        self.model_dir = Path(model_dir)
        self.bench_root = Path(bench_root)
        self.backend = str(backend)  # "mp_gpu" | "inproc_synthetic"
        self.on_ack = on_ack
        self.advertise_host = str(advertise_host)
        self.adapter_factory = adapter_factory

        self._shmem_before_kib = 0
        # mp_gpu fields
        self._mp_ctx: Any = None
        self._procs: dict[int, Any] = {}
        self._cmd_qs: dict[int, Any] = {}
        self._ack_q: Any = None
        self._ack_thread: threading.Thread | None = None
        self._stop_ack = threading.Event()

        # inproc_synthetic fields
        self._tcp_endpoints: dict[int, TcpPeerEndpoint] = {}
        self._runtimes: dict[int, WorkerRuntime] = {}
        self._threads: list[threading.Thread] = []
        self._adapters: list[CPTGModelAdapter] = []
        self._shared_mirrors: dict[int, NumaMemfdMirror] = {}
        self._local_shm_bus: InMemoryByteBus | None = None
        self._ready_by_rank: dict[int, dict[str, Any]] = {}

    def _drain_local_mp_acks(self, expected_desc: str) -> dict[int, dict[str, Any]]:
        collected: dict[int, dict[str, Any]] = {}
        deadline = time.monotonic() + 360.0
        while len(collected) < len(self.global_ranks):
            rem = max(0.1, deadline - time.monotonic())
            if rem <= 0.1 and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"node {self.node_rank} timed out waiting for {expected_desc}; got {sorted(collected.keys())}"
                )
            try:
                msg = self._ack_q.get(timeout=min(2.0, rem))
            except queue.Empty:
                for gr, p in self._procs.items():
                    if not p.is_alive() and gr not in collected:
                        raise RuntimeError(
                            f"node {self.node_rank} worker global_rank {gr} exited prematurely (code={p.exitcode})"
                        )
                continue
            if msg.get("op") == "WORKER_FATAL":
                raise RuntimeError(
                    f"node {self.node_rank} worker {msg.get('rank')} fatal: {msg.get('error')}\n{msg.get('traceback', '')}"
                )
            collected[int(msg["rank"])] = msg
        return collected

    def prepare_until_mesh(self) -> dict[str, Any]:
        """Phase 1–2: Instantiate local workers + intra-node `NumaMemfdMirror`, stop right before mesh init."""
        self._shmem_before_kib = read_shmem_kib()
        if self.backend == "mp_gpu":
            import torch
            import torch.multiprocessing as mp

            if torch.cuda.is_available() and torch.cuda.device_count() < self.gpus_per_node:
                raise RuntimeError(
                    f"node {self.node_rank} requires {self.gpus_per_node} GPUs, saw {torch.cuda.device_count()}"
                )
            self._mp_ctx = mp.get_context("spawn")
            self._ack_q = self._mp_ctx.Queue()
            for gr in self.global_ranks:
                q = self._mp_ctx.Queue()
                self._cmd_qs[gr] = q
                p = self._mp_ctx.Process(
                    target=run_worker_process,
                    args=(
                        gr,
                        self.policy,
                        str(self.model_dir),
                        str(self.bench_root),
                        q,
                        self._ack_q,
                        self.gpus_per_node,
                    ),
                    daemon=False,
                )
                p.start()
                self._procs[gr] = p

            topo = ClusterTopologyProfile.from_multinode_measurements(
                world_size=self.policy.num_gpus,
                gpus_per_node=self.gpus_per_node,
                intra_bw=10.6,
                socket_bw=9.55,
                inter_bw=11.5,
            )
            sample_grp = topo.half_group_of(self.global_ranks[0])
            if self.policy.use_shared_numa_mirror and self.policy.use_expert_pager and len(sample_grp.gpu_ids) >= 2:
                init_msgs = self._drain_local_mp_acks("INIT_LEADER_MIRROR/FOLLOWER")
                leaders: dict[str, dict[str, Any]] = {}
                for m in init_msgs.values():
                    if m.get("op") == "INIT_LEADER_MIRROR":
                        info = {
                            "pid": m["pid"],
                            "fd": m["fd"],
                            "nbytes": m["nbytes"],
                            "inode": m["inode"],
                        }
                        leaders[str(m["group_id"])] = info
                        leaders[str(m["numa_node"])] = info
                for gr in self.global_ranks:
                    self._cmd_qs[gr].put({"op": "ATTACH_SHARED_MIRROR", "leaders": leaders})

            _ = self._drain_local_mp_acks("READY_FOR_NCCL")
            return {"node_rank": self.node_rank, "tcp_endpoints": {}}

        # Pure-Python TCP synthetic mode (`inproc_synthetic`)
        topo = ClusterTopologyProfile.from_multinode_measurements(
            world_size=self.policy.num_gpus,
            gpus_per_node=self.gpus_per_node,
            intra_bw=10.6,
            socket_bw=9.55,
            inter_bw=1.15,
        )
        pacer = HalfNodeH2DPacer(topo, max_concurrent_per_half=1) if self.policy.use_h2d_pacer else None
        self._local_shm_bus = InMemoryByteBus(max(2, self.policy.num_gpus))
        factory = self.adapter_factory or (lambda r, pol: SyntheticCPTGAdapter(r, pol))
        tcp_eps: dict[str, list[Any]] = {}

        for gr in self.global_ranks:
            adapter = factory(gr, self.policy)
            grp = topo.half_group_of(gr)
            mirror_inode = None
            if self.policy.use_expert_pager:
                if self.policy.use_shared_numa_mirror and len(grp.gpu_ids) >= 2:
                    shared_m = self._shared_mirrors.get(grp.group_id)
                    is_first = shared_m is None
                    if is_first:
                        shared_m = NumaMemfdMirror.create(
                            nbytes=max(4096, adapter.total_mirror_bytes),
                            numa_node=grp.numa_node,
                            name=f"ms_mn_grp{grp.group_id}",
                        )
                        self._shared_mirrors[grp.group_id] = shared_m
                    adapter.attach_expert_mirror(shared_m, populate=is_first)
                    mirror_inode = shared_m.inode
                else:
                    adapter.attach_expert_mirror(None, populate=True)

            info = adapter.bind_runtime_resources(pacer=pacer)
            adapter.warmup()
            ep = (
                TcpPeerEndpoint(gr, self.policy.num_gpus, bind_host=self.advertise_host)
                if self.policy.num_gpus >= 2
                else None
            )
            if ep is not None:
                self._tcp_endpoints[gr] = ep
                tcp_eps[str(gr)] = [self.advertise_host, ep.bound_port]

            rt = WorkerRuntime(
                rank=gr,
                policy=self.policy,
                adapter=adapter,
                peer_mesh=ep,
                ack_sink=self.on_ack,
                shm_bus=self._local_shm_bus,
            )
            self._adapters.append(adapter)
            self._runtimes[gr] = rt
            self._ready_by_rank[gr] = {
                "rank": gr,
                "resident_alpha": info.get("resident_alpha", self.policy.initial_alphas.get(gr, 1.0)),
                "kv_free_bytes": query_worker_kv_free_bytes(adapter, self.policy, gr),
                "reserved_mib": info.get("reserved_mib", 0.0),
                "mirror_inode": mirror_inode,
            }

        return {"node_rank": self.node_rank, "tcp_endpoints": tcp_eps}

    def finalize_mesh_and_start(
        self,
        *,
        uid_bytes: bytes | None = None,
        tcp_directory: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Phase 3–4: Complete 16-rank `DirectNcclMesh` or `TcpPeerEndpoint` wiring and start runtime loops."""
        if self.backend == "mp_gpu":
            for gr in self.global_ranks:
                self._cmd_qs[gr].put({"op": "INIT_NCCL_MESH", "uid_bytes": uid_bytes})
            ready_msgs = self._drain_local_mp_acks("WORKER_READY")
            shmem_after_kib = read_shmem_kib()
            shmem_delta_gib = round(max(0.0, (shmem_after_kib - self._shmem_before_kib) / (1024**2)), 2)

            self._stop_ack.clear()
            self._ack_thread = threading.Thread(
                target=self._mp_ack_pump,
                name=f"ms_mn_acks_n{self.node_rank}",
                daemon=True,
            )
            self._ack_thread.start()
            return {
                "node_rank": self.node_rank,
                "host_shmem_delta_gib": shmem_delta_gib,
                "workers": {str(r): ready_msgs[r] for r in sorted(ready_msgs.keys())},
            }

        if tcp_directory:
            for ep in self._tcp_endpoints.values():
                ep.set_directory(tcp_directory)
        for gr, rt in sorted(self._runtimes.items()):
            th = threading.Thread(target=rt.serve_forever, name=f"ms_mn_w{gr}", daemon=True)
            th.start()
            self._threads.append(th)
        return {
            "node_rank": self.node_rank,
            "host_shmem_delta_gib": 0.01 if self._shared_mirrors else 0.0,
            "workers": {str(r): self._ready_by_rank[r] for r in sorted(self._ready_by_rank.keys())},
        }

    def _mp_ack_pump(self) -> None:
        while not self._stop_ack.is_set():
            try:
                msg = self._ack_q.get(timeout=0.05)
            except queue.Empty:
                continue
            self.on_ack(msg)

    def send(self, global_rank: int, payload: dict[str, Any]) -> None:
        r = int(global_rank)
        if self.backend == "mp_gpu":
            self._cmd_qs[r].put(payload)
        else:
            self._runtimes[r].enqueue_command(payload)

    def shutdown(self) -> None:
        if self.backend == "mp_gpu":
            for gr, q in self._cmd_qs.items():
                try:
                    q.put({"op": "SHUTDOWN", "cmd_id": f"mn_shut_{gr}"})
                except Exception:
                    pass
            for p in self._procs.values():
                p.join(timeout=25.0)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5.0)
            self._stop_ack.set()
            if self._ack_thread is not None:
                self._ack_thread.join(timeout=2.0)
                self._ack_thread = None
            for q in self._cmd_qs.values():
                try:
                    q.close()
                except Exception:
                    pass
            if self._ack_q is not None:
                try:
                    self._ack_q.close()
                except Exception:
                    pass
            self._procs.clear()
            self._cmd_qs.clear()
            return

        for gr, rt in self._runtimes.items():
            rt.enqueue_command({"op": "SHUTDOWN", "cmd_id": f"mn_shut_{gr}"})
        for th in self._threads:
            th.join(timeout=5.0)
        for ep in self._tcp_endpoints.values():
            ep.close()
        for ad in self._adapters:
            ad.close()
        for m in self._shared_mirrors.values():
            m.close()
        self._tcp_endpoints.clear()
        self._runtimes.clear()
        self._threads.clear()
        self._adapters.clear()
        self._shared_mirrors.clear()


class MultinodeCoordinatorHub:
    """Persistent TCP control hub on Node 0 managing remote worker daemons (`node_rank = 1 .. num_nodes-1`)."""

    def __init__(
        self,
        *,
        num_nodes: int,
        gpus_per_node: int,
        bind_host: str = "0.0.0.0",
        port: int = 29521,
        advertise_host: str = "127.0.0.1",
        backend: str = "mp_gpu",
        connect_timeout_s: float = 240.0,
        adapter_factory: Callable[[int, EnginePolicyConfig], CPTGModelAdapter] | None = None,
    ) -> None:
        self.num_nodes = max(1, int(num_nodes))
        self.gpus_per_node = max(1, int(gpus_per_node))
        self.world_size = self.num_nodes * self.gpus_per_node
        self.bind_host = str(bind_host)
        self.advertise_host = str(advertise_host)
        self.backend = str(backend)
        self.connect_timeout_s = float(connect_timeout_s)
        self.adapter_factory = adapter_factory

        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.bind_host, int(port)))
        self.bound_port = int(self._srv.getsockname()[1])
        self._srv.listen(max(4, self.num_nodes * 2))

        self._remote_channels: dict[int, _FramedJsonChannel] = {}
        self._ctrl_queues: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._reader_threads: list[threading.Thread] = []
        self._on_ack_lock = threading.Lock()
        self._active_on_ack: Callable[[dict[str, Any]], None] | None = None
        self._connected = False
        self._closed = False

    def wait_for_remote_nodes(self) -> None:
        if self._connected or self.num_nodes <= 1:
            self._connected = True
            return
        self._srv.settimeout(self.connect_timeout_s)
        deadline = time.monotonic() + self.connect_timeout_s
        while len(self._remote_channels) < (self.num_nodes - 1):
            rem = max(0.5, deadline - time.monotonic())
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for remote nodes; got {sorted(self._remote_channels.keys())} of 1..{self.num_nodes - 1}"
                )
            self._srv.settimeout(rem)
            conn, _ = self._srv.accept()
            ch = _FramedJsonChannel(conn)
            hello = ch.recv_json()
            if hello.get("op") != "HELLO_NODE":
                ch.close()
                raise RuntimeError(f"expected HELLO_NODE, got {hello}")
            nr = int(hello["node_rank"])
            self._remote_channels[nr] = ch
            self._ctrl_queues[nr] = queue.Queue()
            th = threading.Thread(
                target=self._remote_reader_loop,
                args=(nr, ch),
                name=f"ms_hub_reader_n{nr}",
                daemon=True,
            )
            th.start()
            self._reader_threads.append(th)
        self._connected = True

    def _remote_reader_loop(self, node_rank: int, ch: _FramedJsonChannel) -> None:
        while not self._closed:
            try:
                msg = ch.recv_json()
            except (ConnectionError, OSError) as exc:
                if not self._closed:
                    err_msg = {
                        "op": "NODE_ERROR",
                        "node_rank": node_rank,
                        "error": f"remote TCP connection lost on node {node_rank}: {exc}",
                    }
                    self._ctrl_queues[node_rank].put(err_msg)
                    with self._on_ack_lock:
                        cb = self._active_on_ack
                    if cb is not None:
                        cb({"op": "NODE_DISCONNECTED", "node_rank": node_rank, "error": err_msg["error"]})
                break
            op = msg.get("op")
            if op == "WORKER_ACK":
                with self._on_ack_lock:
                    cb = self._active_on_ack
                if cb is not None:
                    cb(msg["ack"])
            else:
                self._ctrl_queues[node_rank].put(msg)

    def set_ack_callback(self, on_ack: Callable[[dict[str, Any]], None] | None) -> None:
        with self._on_ack_lock:
            self._active_on_ack = on_ack

    def send_to_remote(self, node_rank: int, msg: Mapping[str, Any]) -> None:
        self._remote_channels[int(node_rank)].send_json(msg)

    def recv_ctrl_from_remote(self, node_rank: int, expected_op: str, timeout_s: float = 420.0) -> dict[str, Any]:
        msg = self._ctrl_queues[int(node_rank)].get(timeout=timeout_s)
        if msg.get("op") == "NODE_ERROR":
            raise RuntimeError(
                f"remote node {node_rank} error: {msg.get('error')}\n{msg.get('traceback', '')}"
            )
        if msg.get("op") != expected_op:
            raise RuntimeError(f"remote node {node_rank} expected {expected_op}, got {msg}")
        return msg

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for ch in list(self._remote_channels.values()):
            try:
                ch.send_json({"op": "EXIT_DAEMON"})
            except Exception:
                pass
            ch.close()
        self._remote_channels.clear()
        try:
            self._srv.close()
        except OSError:
            pass


class TcpMultinodeWorkerFleet(WorkerFleet):
    """Distributed `WorkerFleet` spanning `num_nodes x gpus_per_node` over TCP + 16-rank `DirectNcclMesh`."""

    def __init__(
        self,
        policy: EnginePolicyConfig,
        model_dir: Path,
        bench_root: Path,
        *,
        hub: MultinodeCoordinatorHub,
    ) -> None:
        if policy.num_gpus != hub.world_size:
            raise ValueError(f"policy.num_gpus={policy.num_gpus} != hub.world_size={hub.world_size}")
        self.policy = policy
        self.model_dir = Path(model_dir)
        self.bench_root = Path(bench_root)
        self.hub = hub
        self._local_host: _LocalNodeWorkerHost | None = None
        self._started = False

    def start(self, on_ack: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        self.hub.wait_for_remote_nodes()
        self.hub.set_ack_callback(on_ack)

        pol_wire = serialize_policy_config(self.policy)
        for nr in range(1, self.hub.num_nodes):
            self.hub.send_to_remote(
                nr,
                {
                    "op": "START_FLEET",
                    "policy": pol_wire,
                    "model_dir": str(self.model_dir),
                    "bench_root": str(self.bench_root),
                    "gpus_per_node": self.hub.gpus_per_node,
                    "backend": self.hub.backend,
                },
            )

        self._local_host = _LocalNodeWorkerHost(
            node_rank=0,
            gpus_per_node=self.hub.gpus_per_node,
            policy=self.policy,
            model_dir=self.model_dir,
            bench_root=self.bench_root,
            backend=self.hub.backend,
            on_ack=on_ack,
            advertise_host=self.hub.advertise_host,
            adapter_factory=self.hub.adapter_factory,
        )
        local_prep = self._local_host.prepare_until_mesh()
        merged_tcp_dir: dict[str, Any] = dict(local_prep.get("tcp_endpoints") or {})

        for nr in range(1, self.hub.num_nodes):
            r_prep = self.hub.recv_ctrl_from_remote(nr, "NODE_READY_FOR_MESH")
            merged_tcp_dir.update(r_prep.get("tcp_endpoints") or {})

        uid_bytes = (
            DirectNcclMesh.generate_unique_id_bytes()
            if (self.hub.backend == "mp_gpu" and self.policy.num_gpus >= 2)
            else None
        )
        uid_hex = uid_bytes.hex() if uid_bytes is not None else None
        for nr in range(1, self.hub.num_nodes):
            self.hub.send_to_remote(
                nr,
                {
                    "op": "INIT_MESH",
                    "uid_hex": uid_hex,
                    "tcp_directory": merged_tcp_dir,
                },
            )

        local_ready = self._local_host.finalize_mesh_and_start(
            uid_bytes=uid_bytes,
            tcp_directory=merged_tcp_dir,
        )
        merged_workers: dict[int, dict[str, Any]] = {
            int(r): info for r, info in (local_ready.get("workers") or {}).items()
        }
        max_shmem = float(local_ready.get("host_shmem_delta_gib", 0.0))

        for nr in range(1, self.hub.num_nodes):
            r_ready = self.hub.recv_ctrl_from_remote(nr, "NODE_WORKERS_READY")
            for r, info in (r_ready.get("workers") or {}).items():
                merged_workers[int(r)] = info
            max_shmem = max(max_shmem, float(r_ready.get("host_shmem_delta_gib", 0.0)))

        self._started = True
        return {
            "num_gpus": self.policy.num_gpus,
            "num_nodes": self.hub.num_nodes,
            "gpus_per_node": self.hub.gpus_per_node,
            "policy": self.policy.policy_name,
            "host_shmem_delta_gib": max_shmem,
            "workers": {r: merged_workers[r] for r in sorted(merged_workers.keys())},
        }

    def send(self, rank: int, payload: dict[str, Any]) -> None:
        r = int(rank)
        target_node = r // self.hub.gpus_per_node
        if target_node == 0:
            assert self._local_host is not None
            self._local_host.send(r, payload)
        else:
            self.hub.send_to_remote(target_node, {"op": "WORKER_CMD", "rank": r, "payload": payload})

    def shutdown(self) -> None:
        if not self._started:
            return
        for nr in range(1, self.hub.num_nodes):
            try:
                self.hub.send_to_remote(nr, {"op": "STOP_FLEET"})
            except Exception:
                pass
        if self._local_host is not None:
            self._local_host.shutdown()
            self._local_host = None
        for nr in range(1, self.hub.num_nodes):
            try:
                self.hub.recv_ctrl_from_remote(nr, "STOP_FLEET_DONE", timeout_s=60.0)
            except Exception:
                pass
        self.hub.set_ack_callback(None)
        self._started = False


def run_multinode_worker_daemon(
    *,
    master_addr: str,
    port: int,
    node_rank: int,
    num_nodes: int,
    gpus_per_node: int = 8,
    advertise_host: str | None = None,
    connect_timeout_s: float = 240.0,
    adapter_factory: Callable[[int, EnginePolicyConfig], CPTGModelAdapter] | None = None,
) -> dict[str, Any]:
    """Entry point for remote worker nodes (`node_rank >= 1`) serving 1..K sequential `MorphServeClusterEngine` runs."""
    if int(node_rank) < 1:
        raise ValueError("run_multinode_worker_daemon expects node_rank >= 1")

    deadline = time.monotonic() + float(connect_timeout_s)
    sock: socket.socket | None = None
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            sock = socket.create_connection((str(master_addr), int(port)), timeout=5.0)
            break
        except OSError as exc:
            last_err = exc
            time.sleep(0.25)
    if sock is None:
        raise TimeoutError(f"node_rank {node_rank} could not connect to {master_addr}:{port}: {last_err}")

    adv_host = advertise_host or str(sock.getsockname()[0]) or "127.0.0.1"
    ch = _FramedJsonChannel(sock)
    ch.send_json({"op": "HELLO_NODE", "node_rank": int(node_rank), "num_nodes": int(num_nodes), "host": adv_host})

    host_runtime: _LocalNodeWorkerHost | None = None
    fleets_served = 0

    def _forward_ack(ack_msg: dict[str, Any]) -> None:
        try:
            ch.send_json({"op": "WORKER_ACK", "ack": ack_msg})
        except Exception:
            pass

    try:
        while True:
            try:
                msg = ch.recv_json()
            except (ConnectionError, OSError):
                break
            op = msg.get("op")
            if op == "EXIT_DAEMON":
                break
            try:
                if op == "START_FLEET":
                    if host_runtime is not None:
                        host_runtime.shutdown()
                        host_runtime = None
                    pol = deserialize_policy_config(msg["policy"])
                    gpn = int(msg.get("gpus_per_node", gpus_per_node))
                    host_runtime = _LocalNodeWorkerHost(
                        node_rank=int(node_rank),
                        gpus_per_node=gpn,
                        policy=pol,
                        model_dir=Path(str(msg["model_dir"])),
                        bench_root=Path(str(msg["bench_root"])),
                        backend=str(msg.get("backend", "mp_gpu")),
                        on_ack=_forward_ack,
                        advertise_host=adv_host,
                        adapter_factory=adapter_factory,
                    )
                    prep = host_runtime.prepare_until_mesh()
                    ch.send_json({"op": "NODE_READY_FOR_MESH", **prep})

                elif op == "INIT_MESH":
                    assert host_runtime is not None
                    uid_hex = msg.get("uid_hex")
                    uid_b = bytes.fromhex(uid_hex) if uid_hex else None
                    ready = host_runtime.finalize_mesh_and_start(
                        uid_bytes=uid_b,
                        tcp_directory=msg.get("tcp_directory"),
                    )
                    fleets_served += 1
                    ch.send_json({"op": "NODE_WORKERS_READY", **ready})

                elif op == "WORKER_CMD":
                    assert host_runtime is not None
                    host_runtime.send(int(msg["rank"]), msg["payload"])

                elif op == "STOP_FLEET":
                    if host_runtime is not None:
                        host_runtime.shutdown()
                        host_runtime = None
                    ch.send_json({"op": "STOP_FLEET_DONE", "node_rank": int(node_rank)})

                else:
                    ch.send_json(
                        {
                            "op": "NODE_ERROR",
                            "node_rank": int(node_rank),
                            "error": f"unknown daemon op: {op}",
                        }
                    )
            except BaseException as exc:
                ch.send_json(
                    {
                        "op": "NODE_ERROR",
                        "node_rank": int(node_rank),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-1500:],
                    }
                )
    finally:
        if host_runtime is not None:
            host_runtime.shutdown()
        ch.close()

    return {"status": "ok", "node_rank": int(node_rank), "fleets_served": fleets_served}


__all__ = [
    "MultinodeCoordinatorHub",
    "TcpByteMesh",
    "TcpMultinodeWorkerFleet",
    "TcpPeerEndpoint",
    "deserialize_policy_config",
    "run_multinode_worker_daemon",
    "serialize_policy_config",
]
