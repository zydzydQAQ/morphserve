"""Torch-free `WorkerRuntime` command state machine (batched AR + preemption) and `InProcessWorkerFleet`."""

from __future__ import annotations

from abc import ABC, abstractmethod
import gc
from multiprocessing import shared_memory
import queue
import threading
import time
import traceback
from typing import Any, Callable

from morphserve.models.base import CPTGModelAdapter, SyntheticCPTGAdapter, is_oom_exception
from morphserve.router import ClusterTopologyProfile, HalfNodeH2DPacer


class InMemoryByteBus:
    """Thread-safe pairwise FIFO byte channel implementing `DirectNcclMesh`'s `send_u8`/`recv_u8` seam + SHM."""

    def __init__(self, world_size: int) -> None:
        self._pipes: dict[tuple[int, int], queue.Queue] = {
            (s, d): queue.Queue() for s in range(world_size) for d in range(world_size) if s != d
        }
        self._shm_blobs: dict[str, bytes] = {}
        self._shm_lock = threading.Lock()

    def endpoint_for(self, rank: int) -> "InMemoryPeerMesh":
        return InMemoryPeerMesh(self, rank)

    def put_shm(self, name: str, data: bytes) -> None:
        with self._shm_lock:
            self._shm_blobs[name] = bytes(data)

    def pop_shm(self, name: str) -> bytes:
        with self._shm_lock:
            return self._shm_blobs.pop(name, b"")


class InMemoryPeerMesh:
    """CPU local-substitutable peer mesh matching `DirectNcclMesh` (`send_u8`, `recv_u8`, `close`)."""

    def __init__(self, bus: InMemoryByteBus, rank: int) -> None:
        self.bus = bus
        self.rank = int(rank)

    def send_u8(self, payload: Any, dst_rank: int, sync_stream: bool = True) -> float:
        raw = bytes(payload)
        self.bus._pipes[(self.rank, int(dst_rank))].put(raw)
        return 0.15

    def recv_u8(self, nbytes: int, src_rank: int, sync_stream: bool = True) -> tuple[bytes, float]:
        raw = self.bus._pipes[(int(src_rank), self.rank)].get(timeout=30.0)
        return (raw[: int(nbytes)] if nbytes > 0 else b""), 0.15

    def close(self) -> None:
        return None


def _clear_cuda_cache_if_present() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _payload_nbytes(payload: Any) -> int:
    return int(payload.numel()) if hasattr(payload, "numel") else len(payload)


class WorkerRuntime:
    """Model- and transport-agnostic worker command loop shared by GPU processes and CPU test fleets."""

    def __init__(
        self,
        rank: int,
        policy: Any,
        adapter: CPTGModelAdapter,
        peer_mesh: Any | None,
        ack_sink: Callable[[dict[str, Any]], None],
        shm_bus: InMemoryByteBus | None = None,
        max_ar_batch_size: int = 8,
    ) -> None:
        self.rank = int(rank)
        self.policy = policy
        self.adapter = adapter
        self.peer_mesh = peer_mesh
        self.ack_sink = ack_sink
        self.shm_bus = shm_bus
        self.max_ar_batch_size = max(1, int(max_ar_batch_size))

        self.main_cmd_q: queue.Queue = queue.Queue()
        self.ar_preempt_q: queue.Queue = queue.Queue()
        self._deferred_main_cmds: list[dict[str, Any]] = []
        self._packed_store: dict[str, Any] = {}
        self.oom_count = 0

    def enqueue_command(self, msg: dict[str, Any]) -> None:
        if msg.get("op") == "EXEC_AR_PREEMPT":
            self.ar_preempt_q.put(msg)
        else:
            self.main_cmd_q.put(msg)

    def _poll_preempt_cmds(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        while True:
            try:
                items.append(self.ar_preempt_q.get_nowait())
            except queue.Empty:
                break
        return items

    def _pop_next_main_cmd(self, timeout: float = 0.02) -> dict[str, Any] | None:
        if self._deferred_main_cmds:
            return self._deferred_main_cmds.pop(0)
        try:
            return self.main_cmd_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def _collect_ar_batch(self, first_cmd: dict[str, Any]) -> list[dict[str, Any]]:
        """Opportunistically coalesce queued `EXEC_AR` commands into one continuous-batch execution."""
        batch = [first_cmd]
        while len(batch) < self.max_ar_batch_size:
            try:
                nxt = self.main_cmd_q.get_nowait()
            except queue.Empty:
                break
            if nxt.get("op") == "EXEC_AR":
                batch.append(nxt)
            else:
                self._deferred_main_cmds.append(nxt)
                break
        return batch

    def _execute_ar_cmds_and_ack(self, ar_cmds: list[dict[str, Any]], *, preempted_flag: bool | None = None) -> None:
        try:
            batch_outs = self.adapter.decode_ar_batch(ar_cmds)
            for cmd, ar_res in zip(ar_cmds, batch_outs):
                ack = {
                    "cmd_id": cmd["cmd_id"],
                    "rank": self.rank,
                    "status": "ok",
                    "ctx_id": str(cmd["ctx_id"]),
                    **ar_res,
                }
                if preempted_flag is not None:
                    ack["preempted_mid_dit"] = preempted_flag
                self.ack_sink(ack)
        except BaseException as exc:
            # If a batch fails, fall back per-request so one bad/OOM request never kills innocent peers
            if len(ar_cmds) > 1:
                for single in ar_cmds:
                    self._execute_ar_cmds_and_ack([single], preempted_flag=preempted_flag)
                return
            cmd = ar_cmds[0]
            if is_oom_exception(exc):
                self.oom_count += 1
                gc.collect()
                _clear_cuda_cache_if_present()
                self.ack_sink({"cmd_id": cmd["cmd_id"], "rank": self.rank, "status": "oom", "error": str(exc)[:200]})
            else:
                self.ack_sink(
                    {
                        "cmd_id": cmd["cmd_id"],
                        "rank": self.rank,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-1000:],
                    }
                )

    def _drain_late_preempt_cmds(self) -> bool:
        late = self._poll_preempt_cmds()
        if not late:
            return False
        # Preserve `enqueue_time_s` queue-wait TTFT accounting (`e27bce5`)
        self._execute_ar_cmds_and_ack(late, preempted_flag=False)
        return True

    def _send_shm(self, ctx_id: str, shm_name: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        packed = self.adapter.pack_state(ctx_id, include_cfg_dup=True)
        payload = packed.payload
        if hasattr(payload, "cpu"):
            cpu_bytes = payload.cpu().numpy().tobytes()
        else:
            cpu_bytes = bytes(payload)
        del payload

        if self.shm_bus is not None:
            self.shm_bus.put_shm(shm_name, cpu_bytes)
        else:
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=max(1, len(cpu_bytes)))
            if cpu_bytes:
                shm.buf[: len(cpu_bytes)] = cpu_bytes
            shm.close()

        ser_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "status": "ok",
            "shm_name": shm_name,
            "meta": packed.meta,
            "serialize_ms": round(ser_ms, 2),
            "bytes": len(cpu_bytes),
        }

    def _recv_shm(self, shm_name: str, new_ctx_id: str, meta: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        n_wire = int(meta["total_bytes"])
        if self.shm_bus is not None:
            raw = self.shm_bus.pop_shm(shm_name)[:n_wire]
            self.adapter.unpack_state(new_ctx_id, meta, raw)
        else:
            import torch

            shm = shared_memory.SharedMemory(name=shm_name, create=False)
            try:
                raw = bytes(shm.buf[:n_wire])
            finally:
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            device = getattr(self.adapter, "device", f"cuda:{self.rank}")
            flat = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
            self.adapter.unpack_state(new_ctx_id, meta, flat)
            del flat

        return {
            "status": "ok",
            "recv_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "resident_alpha": self.adapter.resident_alpha(),
        }

    def serve_forever(self) -> None:
        """Process incoming coordinator commands until `SHUTDOWN`."""
        while True:
            if self._drain_late_preempt_cmds():
                continue

            cmd = self._pop_next_main_cmd(timeout=0.02)
            if cmd is None:
                continue

            op = cmd.get("op")
            cmd_id = cmd.get("cmd_id")
            if op == "SHUTDOWN":
                while self._drain_late_preempt_cmds():
                    pass
                self.ack_sink({"cmd_id": cmd_id, "rank": self.rank, "status": "ok"})
                break

            if op == "EXEC_AR":
                ar_batch = self._collect_ar_batch(cmd)
                self._execute_ar_cmds_and_ack(ar_batch)
                continue

            try:
                if op == "PREP_DIT":
                    cid = str(cmd["ctx_id"])
                    kv_b = self.adapter.prepare_turn(cid, cmd)
                    self.ack_sink(
                        {
                            "cmd_id": cmd_id,
                            "rank": self.rank,
                            "status": "ok",
                            "ctx_id": cid,
                            "kv_bytes": kv_b,
                            "resident_alpha": self.adapter.resident_alpha(),
                        }
                    )

                elif op == "EXEC_DIT":
                    cid = str(cmd["ctx_id"])
                    pending_inflight: list[dict[str, Any]] = []

                    def _tracked_poll() -> list[dict[str, Any]]:
                        polled = self._poll_preempt_cmds()
                        pending_inflight.extend(polled)
                        return polled

                    def _tracked_ack(ack_msg: dict[str, Any]) -> None:
                        done_id = ack_msg.get("cmd_id")
                        pending_inflight[:] = [m for m in pending_inflight if m.get("cmd_id") != done_id]
                        if ack_msg.get("status") == "oom":
                            self.oom_count += 1
                        self.ack_sink(ack_msg)

                    try:
                        dit_res = self.adapter.generate_dit(
                            cid,
                            cmd,
                            poll_preempt_cmds=_tracked_poll,
                            emit_preempt_ack=_tracked_ack,
                        )
                    except BaseException as dit_exc:
                        err_status = "oom" if is_oom_exception(dit_exc) else "error"
                        for orphan in pending_inflight:
                            self.ack_sink(
                                {
                                    "cmd_id": orphan["cmd_id"],
                                    "rank": self.rank,
                                    "status": err_status,
                                    "error": f"aborted mid-dit: {dit_exc}"[:200],
                                }
                            )
                        pending_inflight.clear()
                        raise
                    self.ack_sink({"cmd_id": cmd_id, "rank": self.rank, **dit_res})

                elif op == "PACK_CTX":
                    cid = str(cmd["ctx_id"])
                    packed = self.adapter.pack_state(cid, include_cfg_dup=bool(cmd.get("include_cfg_dup", False)))
                    packed_id = f"{cid}__{cmd_id}"
                    self._packed_store[packed_id] = packed.payload
                    self.ack_sink(
                        {"cmd_id": cmd_id, "rank": self.rank, "status": "ok", "packed_id": packed_id, "meta": packed.meta}
                    )

                elif op == "SEND_NCCL":
                    assert self.peer_mesh is not None
                    payload = self._packed_store.pop(cmd["packed_id"])
                    nbytes = _payload_nbytes(payload)
                    ship_ms = self.peer_mesh.send_u8(payload, dst_rank=int(cmd["dst_rank"]))
                    del payload
                    self.ack_sink(
                        {"cmd_id": cmd_id, "rank": self.rank, "status": "ok", "ship_ms": ship_ms, "bytes": nbytes}
                    )

                elif op == "RECV_NCCL":
                    assert self.peer_mesh is not None
                    meta = cmd["meta"]
                    wire_bytes = int(meta["total_bytes"])
                    raw, ship_ms = self.peer_mesh.recv_u8(wire_bytes, src_rank=int(cmd["src_rank"]))
                    logical_b = self.adapter.unpack_state(str(cmd["new_ctx_id"]), meta, raw)
                    del raw
                    self.ack_sink(
                        {
                            "cmd_id": cmd_id,
                            "rank": self.rank,
                            "status": "ok",
                            "ship_ms": ship_ms,
                            "bytes": int(meta.get("logical_bytes", logical_b)),
                            "resident_alpha": self.adapter.resident_alpha(),
                        }
                    )

                elif op == "SEND_SHM":
                    res = self._send_shm(str(cmd["ctx_id"]), str(cmd["shm_name"]))
                    self.ack_sink({"cmd_id": cmd_id, "rank": self.rank, **res})

                elif op == "RECV_SHM":
                    res = self._recv_shm(str(cmd["shm_name"]), str(cmd["new_ctx_id"]), cmd["meta"])
                    self.ack_sink({"cmd_id": cmd_id, "rank": self.rank, **res})

                elif op == "SET_ALPHA":
                    new_alpha = self.adapter.set_residency_alpha(float(cmd["alpha"]))
                    self.ack_sink(
                        {
                            "cmd_id": cmd_id,
                            "rank": self.rank,
                            "status": "ok",
                            "resident_alpha": new_alpha,
                            "resident_layers": int(round(new_alpha * self.adapter.num_expert_layers)),
                        }
                    )

                elif op == "RELEASE_CTX":
                    self.adapter.release_context(str(cmd["ctx_id"]))
                    self.ack_sink({"cmd_id": cmd_id, "rank": self.rank, "status": "ok"})

                elif op == "GET_STATS":
                    st = self.adapter.memory_and_vmm_stats()
                    self.ack_sink(
                        {
                            "cmd_id": cmd_id,
                            "rank": self.rank,
                            "status": "ok",
                            "oom_count": self.oom_count,
                            "resident_alpha": self.adapter.resident_alpha(),
                            **st,
                        }
                    )

                else:
                    self.ack_sink({"cmd_id": cmd_id, "rank": self.rank, "status": "error", "error": f"unknown op: {op}"})

            except BaseException as exc:
                if is_oom_exception(exc):
                    self.oom_count += 1
                    gc.collect()
                    _clear_cuda_cache_if_present()
                    self.ack_sink({"cmd_id": cmd_id, "rank": self.rank, "status": "oom", "error": str(exc)[:300]})
                else:
                    self.ack_sink(
                        {
                            "cmd_id": cmd_id,
                            "rank": self.rank,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()[-1200:],
                        }
                    )


class WorkerFleet(ABC):
    """Seam between `MorphServeClusterEngine` and worker instances (`mp.spawn` GPU vs in-process CPU)."""

    @abstractmethod
    def start(self, on_ack: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        """Start all worker ranks, register `on_ack` callback, and return startup summary."""

    @abstractmethod
    def send(self, rank: int, payload: dict[str, Any]) -> None:
        """Dispatch a command dictionary to `rank`."""

    @abstractmethod
    def shutdown(self) -> None:
        """Stop all worker threads/processes and clean up transport resources."""


class InProcessWorkerFleet(WorkerFleet):
    """Pure-Python threaded `WorkerFleet` running real `WorkerRuntime` state machines without PyTorch/CUDA."""

    def __init__(
        self,
        policy: Any,
        adapter_factory: Callable[[int, Any], CPTGModelAdapter] | None = None,
        topology: ClusterTopologyProfile | None = None,
    ) -> None:
        self.policy = policy
        self.topology = topology or ClusterTopologyProfile.a100_pcie_8gpu_default()
        self.adapter_factory = adapter_factory or (lambda r, pol: SyntheticCPTGAdapter(r, pol))
        self._bus: InMemoryByteBus | None = None
        self._runtimes: list[WorkerRuntime] = []
        self._threads: list[threading.Thread] = []
        self._adapters: list[CPTGModelAdapter] = []
        self._shared_mirrors: dict[int, Any] = {}

    def start(self, on_ack: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        from morphserve.vmm import NumaMemfdMirror

        n = int(self.policy.num_gpus)
        self._bus = InMemoryByteBus(max(2, n))
        pacer = HalfNodeH2DPacer(self.topology, max_concurrent_per_half=1) if self.policy.use_h2d_pacer else None

        ready_by_rank: dict[int, dict[str, Any]] = {}
        for rank in range(n):
            adapter = self.adapter_factory(rank, self.policy)
            numa_node = self.topology.numa_node_of(rank) if rank < self.topology.num_gpus else (rank % 2)
            mirror_inode = None
            if self.policy.use_expert_pager:
                if self.policy.use_shared_numa_mirror and n >= 2:
                    shared_m = self._shared_mirrors.get(numa_node)
                    is_first = shared_m is None
                    if is_first:
                        shared_m = NumaMemfdMirror.create(
                            nbytes=max(4096, adapter.total_mirror_bytes),
                            numa_node=numa_node,
                            name=f"ms_inproc_numa{numa_node}",
                        )
                        self._shared_mirrors[numa_node] = shared_m
                    adapter.attach_expert_mirror(shared_m, populate=is_first)
                    mirror_inode = shared_m.inode
                else:
                    adapter.attach_expert_mirror(None, populate=True)

            info = adapter.bind_runtime_resources(pacer=pacer)
            adapter.warmup()
            peer = self._bus.endpoint_for(rank) if n >= 2 else None
            rt = WorkerRuntime(
                rank=rank,
                policy=self.policy,
                adapter=adapter,
                peer_mesh=peer,
                ack_sink=on_ack,
                shm_bus=self._bus,
            )
            th = threading.Thread(target=rt.serve_forever, name=f"ms_inproc_w{rank}", daemon=True)
            th.start()
            self._adapters.append(adapter)
            self._runtimes.append(rt)
            self._threads.append(th)
            ready_by_rank[rank] = {
                "rank": rank,
                "resident_alpha": info.get("resident_alpha", self.policy.initial_alphas.get(rank, 1.0)),
                "reserved_mib": info.get("reserved_mib", 0.0),
                "mirror_inode": mirror_inode,
            }

        return {
            "num_gpus": n,
            "policy": self.policy.policy_name,
            "host_shmem_delta_gib": 0.01 if self._shared_mirrors else 0.0,
            "workers": ready_by_rank,
        }

    def send(self, rank: int, payload: dict[str, Any]) -> None:
        self._runtimes[int(rank)].enqueue_command(payload)

    def shutdown(self) -> None:
        for r, rt in enumerate(self._runtimes):
            rt.enqueue_command({"op": "SHUTDOWN", "cmd_id": f"inproc_shut_{r}"})
        for th in self._threads:
            th.join(timeout=5.0)
        for ad in self._adapters:
            ad.close()
        for m in self._shared_mirrors.values():
            m.close()
        self._shared_mirrors.clear()
        self._runtimes.clear()
        self._threads.clear()
        self._adapters.clear()
