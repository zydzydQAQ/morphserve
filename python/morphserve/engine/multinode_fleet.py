"""Distributed TCP Coordinator + Multi-Node `WorkerFleet` (`2x8 = 16` GPUs or N-node TCP synthetic)."""

from __future__ import annotations

import json
from pathlib import Path
import queue
import socket
import struct
import threading
import time
import traceback
from typing import Any, Callable, Mapping

from morphserve.models import CPTGModelAdapter
from morphserve.runtime.peer_transport import NcclPeerAdapter, TcpByteMesh, TcpPeerEndpoint

from .config import EnginePolicyConfig
from .worker_proc import LocalNodeWorkerHost
from .worker_runtime import WorkerFleet


def serialize_policy_config(policy: EnginePolicyConfig) -> dict[str, Any]:
    return policy.to_dict()


def deserialize_policy_config(data: Mapping[str, Any]) -> EnginePolicyConfig:
    return EnginePolicyConfig.from_dict(data)


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
        self.backend = "inproc_tcp" if str(backend) == "inproc_synthetic" else str(backend)
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
        self._cb_lock = threading.Lock()
        self._active_on_ack: Callable[[dict[str, Any]], None] | None = None
        self._active_on_fault: Callable[[list[int], str], None] | None = None
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
                    reason = f"remote TCP connection lost on node {node_rank}: {exc}"
                    err_msg = {"op": "NODE_ERROR", "node_rank": node_rank, "error": reason}
                    self._ctrl_queues[node_rank].put(err_msg)
                    with self._cb_lock:
                        fault_cb = self._active_on_fault
                    if fault_cb is not None:
                        dead_ranks = list(
                            range(node_rank * self.gpus_per_node, (node_rank + 1) * self.gpus_per_node)
                        )
                        fault_cb(dead_ranks, reason)
                break
            op = msg.get("op")
            if op == "WORKER_ACK":
                with self._cb_lock:
                    cb = self._active_on_ack
                if cb is not None:
                    cb(msg["ack"])
            elif op == "WORKER_FAULT":
                with self._cb_lock:
                    fault_cb = self._active_on_fault
                if fault_cb is not None:
                    ranks = [int(r) for r in (msg.get("ranks") or ())]
                    fault_cb(ranks, str(msg.get("error", f"worker fault on node {node_rank}")))
            else:
                self._ctrl_queues[node_rank].put(msg)

    def set_ack_callback(self, on_ack: Callable[[dict[str, Any]], None] | None) -> None:
        with self._cb_lock:
            self._active_on_ack = on_ack

    def set_fault_callback(self, on_worker_fault: Callable[[list[int], str], None] | None) -> None:
        with self._cb_lock:
            self._active_on_fault = on_worker_fault

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
    """Distributed `WorkerFleet` spanning `num_nodes x gpus_per_node` over TCP + N-rank `NcclPeerAdapter`."""

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
        self._local_host: LocalNodeWorkerHost | None = None
        self._started = False

    def start(
        self,
        on_ack: Callable[[dict[str, Any]], None],
        on_worker_fault: Callable[[list[int], str], None] | None = None,
    ) -> dict[str, Any]:
        self.hub.wait_for_remote_nodes()
        self.hub.set_ack_callback(on_ack)
        self.hub.set_fault_callback(on_worker_fault)

        pol_wire = self.policy.to_dict()
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

        self._local_host = LocalNodeWorkerHost(
            node_rank=0,
            gpus_per_node=self.hub.gpus_per_node,
            policy=self.policy,
            model_dir=self.model_dir,
            bench_root=self.bench_root,
            backend=self.hub.backend,
            on_ack=on_ack,
            on_worker_fault=on_worker_fault,
            advertise_host=self.hub.advertise_host,
            adapter_factory=self.hub.adapter_factory,
        )
        local_prep = self._local_host.prepare_until_mesh()
        merged_tcp_dir: dict[str, Any] = dict(local_prep.get("tcp_endpoints") or {})

        for nr in range(1, self.hub.num_nodes):
            r_prep = self.hub.recv_ctrl_from_remote(nr, "NODE_READY_FOR_MESH")
            merged_tcp_dir.update(r_prep.get("tcp_endpoints") or {})

        uid_bytes = (
            NcclPeerAdapter.generate_unique_id_bytes()
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
        self.hub.set_fault_callback(None)
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

    host_runtime: LocalNodeWorkerHost | None = None
    fleets_served = 0

    def _forward_ack(ack_msg: dict[str, Any]) -> None:
        try:
            ch.send_json({"op": "WORKER_ACK", "ack": ack_msg})
        except Exception:
            pass

    def _forward_fault(dead_ranks: list[int], reason: str) -> None:
        try:
            ch.send_json(
                {
                    "op": "WORKER_FAULT",
                    "node_rank": int(node_rank),
                    "ranks": [int(r) for r in dead_ranks],
                    "error": reason,
                }
            )
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
                    pol = EnginePolicyConfig.from_dict(msg["policy"])
                    gpn = int(msg.get("gpus_per_node", gpus_per_node))
                    host_runtime = LocalNodeWorkerHost(
                        node_rank=int(node_rank),
                        gpus_per_node=gpn,
                        policy=pol,
                        model_dir=Path(str(msg["model_dir"])),
                        bench_root=Path(str(msg["bench_root"])),
                        backend=str(msg.get("backend", "mp_gpu")),
                        on_ack=_forward_ack,
                        on_worker_fault=_forward_fault,
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
