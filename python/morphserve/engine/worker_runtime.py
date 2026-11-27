"""Torch-free `WorkerRuntime` command state machine (batched AR + preemption), `WorkerFleet` ABC, and `InProcessWorkerFleet`."""

from __future__ import annotations

from abc import ABC, abstractmethod
import gc
from multiprocessing import shared_memory
import queue
import threading
import time
import traceback
from typing import Any, Callable

from morphserve.models.base import CPTGModelAdapter, is_oom_exception
from morphserve.router.topology import ClusterTopologyProfile, default_topology_for
from morphserve.runtime.peer_transport import InMemoryByteBus, InMemoryPeerMesh

_XFER_OPS = frozenset({"ALLOC_RECV_BUF", "SEND_NCCL", "RECV_NCCL", "DROP_PACKED_CTX", "DROP_RECV_BUF"})


def _clear_cuda_cache_if_present() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _record_cuda_event_if_tensor(payload: Any) -> Any:
    if getattr(payload, "is_cuda", False):
        import torch

        return torch.cuda.current_stream(payload.device).record_event()
    return None


def _record_default_stream_event_if_cuda(payload: Any) -> Any:
    if getattr(payload, "is_cuda", False):
        import torch

        return torch.cuda.default_stream(payload.device).record_event()
    return None


def _record_current_stream_if_cuda_tensor(payload: Any) -> None:
    if getattr(payload, "is_cuda", False) and hasattr(payload, "record_stream"):
        import torch

        payload.record_stream(torch.cuda.current_stream(payload.device))


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
        self.xfer_cmd_q: queue.Queue = queue.Queue()
        self._deferred_main_cmds: list[dict[str, Any]] = []
        self._xfer_store_lock = threading.Lock()
        self._telemetry_lock = threading.Lock()
        self._telemetry_seq = 0
        self._packed_store: dict[str, tuple[Any, Any]] = {}
        self._recv_store: dict[str, Any] = {}
        self._shutdown_requested = False
        self._xfer_thread_running = False
        self.oom_count = 0

    def _current_kv_free_bytes(self) -> int:
        return max(0, int(self.adapter.available_kv_bytes()))

    def _telemetry_snapshot(self, *, resident_alpha: float | None = None) -> dict[str, Any]:
        with self._telemetry_lock:
            self._telemetry_seq += 1
            seq = self._telemetry_seq
            alpha = float(resident_alpha) if resident_alpha is not None else float(self.adapter.resident_alpha())
            kv_free_b = self._current_kv_free_bytes()
        return {
            "resident_alpha": alpha,
            "kv_free_bytes": kv_free_b,
            "telemetry_seq": seq,
        }

    def enqueue_command(self, msg: dict[str, Any]) -> None:
        op = msg.get("op")
        if op in ("EXEC_AR", "EXEC_AR_PREEMPT"):
            msg = dict(msg, _worker_recv_s=time.perf_counter())
        if op == "EXEC_AR_PREEMPT":
            self.ar_preempt_q.put(msg)
        elif op in _XFER_OPS:
            self.xfer_cmd_q.put(msg)
        else:
            self.main_cmd_q.put(msg)

    def _emit_ack(self, ack: dict[str, Any]) -> dict[str, Any]:
        self.ack_sink(ack)
        return ack

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

    def _execute_ar_cmds_and_ack(
        self,
        ar_cmds: list[dict[str, Any]],
        *,
        preempted_flag: bool | None = None,
    ) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        try:
            batch_outs = self.adapter.decode_ar_batch(ar_cmds)
            snap = self._telemetry_snapshot()
            for cmd, ar_res in zip(ar_cmds, batch_outs):
                ack = {
                    "cmd_id": cmd.get("cmd_id"),
                    "rank": self.rank,
                    "status": "ok",
                    "ctx_id": str(cmd["ctx_id"]),
                    **ar_res,
                    **snap,
                }
                if preempted_flag is not None:
                    ack["preempted_mid_dit"] = preempted_flag
                emitted.append(self._emit_ack(ack))
            return emitted
        except BaseException as exc:
            # If a batch fails, fall back per-request so one bad/OOM request never kills innocent peers.
            if len(ar_cmds) > 1:
                for single in ar_cmds:
                    emitted.extend(self._execute_ar_cmds_and_ack([single], preempted_flag=preempted_flag))
                return emitted
            cmd = ar_cmds[0]
            if is_oom_exception(exc):
                self.oom_count += 1
                gc.collect()
                _clear_cuda_cache_if_present()
                emitted.append(
                    self._emit_ack(
                        {
                            "cmd_id": cmd.get("cmd_id"),
                            "rank": self.rank,
                            "status": "oom",
                            **self._telemetry_snapshot(),
                            "error": str(exc)[:200],
                        }
                    )
                )
            else:
                emitted.append(
                    self._emit_ack(
                        {
                            "cmd_id": cmd.get("cmd_id"),
                            "rank": self.rank,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()[-1000:],
                        }
                    )
                )
            return emitted

    def _drain_late_preempt_cmds(self) -> bool:
        late = self._poll_preempt_cmds()
        if not late:
            return False
        self._execute_ar_cmds_and_ack(late, preempted_flag=False)
        return True

    def _send_shm(
        self,
        ctx_id: str,
        shm_name: str,
        *,
        include_cfg_dup: bool = True,
        release_after_send: bool = False,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        packed = self.adapter.pack_state(ctx_id, include_cfg_dup=include_cfg_dup)
        payload = packed.payload
        if hasattr(payload, "cpu"):
            cpu_bytes = payload.cpu().numpy().tobytes()
        else:
            cpu_bytes = bytes(payload)
        del payload
        if release_after_send:
            self.adapter.release_context(ctx_id)

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
            "bytes": int(packed.meta.get("logical_bytes", len(cpu_bytes))),
            "wire_bytes": len(cpu_bytes),
            **self._telemetry_snapshot(),
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
            **self._telemetry_snapshot(),
        }

    def _dispatch_xfer_cmd(self, cmd: dict[str, Any]) -> dict[str, Any] | None:
        op = cmd.get("op")
        cmd_id = cmd.get("cmd_id")
        try:
            if op == "ALLOC_RECV_BUF":
                assert self.peer_mesh is not None
                nbytes = max(0, int(cmd["nbytes"]))
                recv_id = f"rbuf_{self.rank}_{cmd_id}"
                buf = self.peer_mesh.alloc_recv_u8(nbytes) if hasattr(self.peer_mesh, "alloc_recv_u8") else bytearray(nbytes)
                alloc_ev = _record_default_stream_event_if_cuda(buf)
                with self._xfer_store_lock:
                    self._recv_store[recv_id] = (buf, alloc_ev)
                return self._emit_ack(
                    {
                        "cmd_id": cmd_id,
                        "rank": self.rank,
                        "status": "ok",
                        "recv_id": recv_id,
                        "nbytes": nbytes,
                    }
                )

            if op == "DROP_PACKED_CTX":
                with self._xfer_store_lock:
                    entry = self._packed_store.pop(str(cmd["packed_id"]), None)
                del entry
                return self._emit_ack({"cmd_id": cmd_id, "rank": self.rank, "status": "ok"})

            if op == "DROP_RECV_BUF":
                with self._xfer_store_lock:
                    buf = self._recv_store.pop(str(cmd["recv_id"]), None)
                del buf
                return self._emit_ack({"cmd_id": cmd_id, "rank": self.rank, "status": "ok"})

            if op == "SEND_NCCL":
                assert self.peer_mesh is not None
                with self._xfer_store_lock:
                    payload, ready_ev = self._packed_store.pop(str(cmd["packed_id"]))
                try:
                    nbytes = _payload_nbytes(payload)
                    ship_ms = self.peer_mesh.send_u8(
                        payload,
                        dst_rank=int(cmd["dst_rank"]),
                        ready_event=ready_ev,
                    )
                finally:
                    del payload
                return self._emit_ack(
                    {
                        "cmd_id": cmd_id,
                        "rank": self.rank,
                        "status": "ok",
                        "ship_ms": ship_ms,
                        "bytes": nbytes,
                    }
                )

            if op == "RECV_NCCL":
                assert self.peer_mesh is not None
                recv_id = cmd.get("recv_id")
                if recv_id is None:
                    raise ValueError("RECV_NCCL requires a pre-allocated recv_id from ALLOC_RECV_BUF")
                meta = cmd["meta"]
                wire_bytes = int(meta["total_bytes"])
                rid_s = str(recv_id)
                with self._xfer_store_lock:
                    entry = self._recv_store.pop(rid_s, None)
                if isinstance(entry, tuple) and len(entry) == 2:
                    pre_buf, alloc_ev = entry
                else:
                    pre_buf, alloc_ev = entry, None
                raw, ship_ms = self.peer_mesh.recv_u8(
                    wire_bytes,
                    src_rank=int(cmd["src_rank"]),
                    out_tensor=pre_buf,
                    ready_event=alloc_ev,
                )
                with self._xfer_store_lock:
                    self._recv_store[rid_s] = raw
                return self._emit_ack(
                    {
                        "cmd_id": cmd_id,
                        "rank": self.rank,
                        "status": "ok",
                        "recv_id": rid_s,
                        "ship_ms": ship_ms,
                        "bytes": int(meta.get("logical_bytes", wire_bytes)),
                    }
                )

            return self._emit_ack(
                {"cmd_id": cmd_id, "rank": self.rank, "status": "error", "error": f"unknown xfer op: {op}"}
            )
        except BaseException as exc:
            if is_oom_exception(exc):
                probe_only = bool(op == "ALLOC_RECV_BUF" and cmd.get("probe_only", False))
                if not probe_only:
                    self.oom_count += 1
                gc.collect()
                _clear_cuda_cache_if_present()
                return self._emit_ack(
                    {
                        "cmd_id": cmd_id,
                        "rank": self.rank,
                        "status": "oom",
                        "probe_only": probe_only,
                        "error": str(exc)[:300],
                    }
                )
            return self._emit_ack(
                {
                    "cmd_id": cmd_id,
                    "rank": self.rank,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-1200:],
                }
            )

    def _serve_xfer_forever(self) -> None:
        """Dedicated transfer thread executing `PeerByteTransport` P2P ops concurrently with `main_cmd_q` compute."""
        self._xfer_thread_running = True
        try:
            while True:
                cmd = self.xfer_cmd_q.get()
                if cmd is None:
                    continue
                if cmd.get("op") == "SHUTDOWN":
                    break
                self._dispatch_xfer_cmd(cmd)
        finally:
            self._xfer_thread_running = False

    def _dispatch_main_cmd(self, cmd: dict[str, Any]) -> dict[str, Any] | None:
        op = cmd.get("op")
        cmd_id = cmd.get("cmd_id")
        try:
            if op == "PREP_DIT":
                cid = str(cmd["ctx_id"])
                kv_b = self.adapter.prepare_turn(cid, cmd)
                return self._emit_ack(
                    {
                        "cmd_id": cmd_id,
                        "rank": self.rank,
                        "status": "ok",
                        "ctx_id": cid,
                        "kv_bytes": kv_b,
                        **self._telemetry_snapshot(),
                    }
                )

            if op == "EXEC_DIT":
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
                    snap = self._telemetry_snapshot(resident_alpha=ack_msg.get("resident_alpha"))
                    self._emit_ack(dict(ack_msg, **snap))

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
                        self._emit_ack(
                            {
                                "cmd_id": orphan["cmd_id"],
                                "rank": self.rank,
                                "status": err_status,
                                **self._telemetry_snapshot(),
                                "error": f"aborted mid-dit: {dit_exc}"[:200],
                            }
                        )
                    pending_inflight.clear()
                    raise
                if pending_inflight:
                    for leftover in pending_inflight:
                        self.ar_preempt_q.put(leftover)
                    pending_inflight.clear()
                if cmd.get("release_after_dit", False):
                    self.adapter.release_context(cid)
                return self._emit_ack(
                    {
                        "cmd_id": cmd_id,
                        "rank": self.rank,
                        **dit_res,
                        **self._telemetry_snapshot(),
                    }
                )

            if op == "PACK_CTX":
                cid = str(cmd["ctx_id"])
                packed = self.adapter.pack_state(cid, include_cfg_dup=bool(cmd.get("include_cfg_dup", False)))
                ready_ev = _record_cuda_event_if_tensor(packed.payload)
                packed_id = f"{cid}__{cmd_id}"
                with self._xfer_store_lock:
                    self._packed_store[packed_id] = (packed.payload, ready_ev)
                if cmd.get("release_after_pack", False):
                    # Synchronize `ready_ev` BEFORE `release_context` calls `cuMemUnmap` on VmmNaiveCache VA.
                    if ready_ev is not None and hasattr(ready_ev, "synchronize"):
                        ready_ev.synchronize()
                    self.adapter.release_context(cid)
                return self._emit_ack(
                    {
                        "cmd_id": cmd_id,
                        "rank": self.rank,
                        "status": "ok",
                        "packed_id": packed_id,
                        "meta": packed.meta,
                        **self._telemetry_snapshot(),
                    }
                )

            if op == "UNPACK_CTX":
                recv_id = str(cmd["recv_id"])
                meta = cmd["meta"]
                with self._xfer_store_lock:
                    raw = self._recv_store.pop(recv_id)
                try:
                    _record_current_stream_if_cuda_tensor(raw)
                    logical_b = self.adapter.unpack_state(str(cmd["new_ctx_id"]), meta, raw)
                    if getattr(raw, "is_cuda", False):
                        import torch

                        torch.cuda.current_stream(raw.device).synchronize()
                finally:
                    del raw
                return self._emit_ack(
                    {
                        "cmd_id": cmd_id,
                        "rank": self.rank,
                        "status": "ok",
                        "bytes": int(meta.get("logical_bytes", logical_b)),
                        **self._telemetry_snapshot(),
                    }
                )

            if op == "SEND_SHM":
                res = self._send_shm(
                    str(cmd["ctx_id"]),
                    str(cmd["shm_name"]),
                    include_cfg_dup=bool(cmd.get("include_cfg_dup", True)),
                    release_after_send=bool(cmd.get("release_after_send", False)),
                )
                return self._emit_ack({"cmd_id": cmd_id, "rank": self.rank, **res})

            if op == "RECV_SHM":
                res = self._recv_shm(str(cmd["shm_name"]), str(cmd["new_ctx_id"]), cmd["meta"])
                return self._emit_ack({"cmd_id": cmd_id, "rank": self.rank, **res})

            if op == "SET_ALPHA":
                new_alpha = self.adapter.set_residency_alpha(float(cmd["alpha"]))
                return self._emit_ack(
                    {
                        "cmd_id": cmd_id,
                        "rank": self.rank,
                        "status": "ok",
                        "resident_layers": int(round(new_alpha * self.adapter.num_expert_layers)),
                        **self._telemetry_snapshot(resident_alpha=new_alpha),
                    }
                )

            if op == "RELEASE_CTX":
                self.adapter.release_context(str(cmd["ctx_id"]))
                return self._emit_ack(
                    {
                        "cmd_id": cmd_id,
                        "rank": self.rank,
                        "status": "ok",
                        **self._telemetry_snapshot(),
                    }
                )

            if op == "GET_STATS":
                st = self.adapter.memory_and_vmm_stats()
                return self._emit_ack(
                    {
                        "cmd_id": cmd_id,
                        "rank": self.rank,
                        "status": "ok",
                        "oom_count": self.oom_count,
                        **st,
                        **self._telemetry_snapshot(),
                    }
                )

            if op == "RESET_SUITE":
                self.oom_count = 0
                gc.collect()
                _clear_cuda_cache_if_present()
                if bool(cmd.get("reset_alpha", True)):
                    new_alpha = self.adapter.reset_for_new_suite()
                else:
                    new_alpha = float(self.adapter.resident_alpha())
                    self.adapter.reset_peak_memory_stats()
                st = self.adapter.memory_and_vmm_stats()
                return self._emit_ack(
                    {
                        "cmd_id": cmd_id,
                        "rank": self.rank,
                        "status": "ok",
                        "oom_count": 0,
                        **st,
                        **self._telemetry_snapshot(resident_alpha=new_alpha),
                    }
                )

            return self._emit_ack(
                {"cmd_id": cmd_id, "rank": self.rank, "status": "error", "error": f"unknown op: {op}"}
            )

        except BaseException as exc:
            if is_oom_exception(exc):
                self.oom_count += 1
                gc.collect()
                _clear_cuda_cache_if_present()
                return self._emit_ack(
                    {
                        "cmd_id": cmd_id,
                        "rank": self.rank,
                        "status": "oom",
                        **self._telemetry_snapshot(),
                        "error": str(exc)[:300],
                    }
                )
            return self._emit_ack(
                {
                    "cmd_id": cmd_id,
                    "rank": self.rank,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-1200:],
                }
            )

    def _handle_shutdown(self, cmd: dict[str, Any]) -> dict[str, Any]:
        while self._drain_late_preempt_cmds():
            pass
        self._shutdown_requested = True
        with self._xfer_store_lock:
            self._packed_store.clear()
            self._recv_store.clear()
        return self._emit_ack({"cmd_id": cmd.get("cmd_id"), "rank": self.rank, "status": "ok"})

    def handle_command(self, cmd: dict[str, Any]) -> dict[str, Any] | None:
        """Synchronously execute one command without background threads, emit its ACK, and return it."""
        op = cmd.get("op")
        if op == "SHUTDOWN":
            return self._handle_shutdown(cmd)
        if op == "EXEC_AR":
            stamped = dict(cmd)
            stamped.setdefault("_worker_recv_s", time.perf_counter())
            acks = self._execute_ar_cmds_and_ack([stamped])
            return acks[-1] if acks else None
        if op == "EXEC_AR_PREEMPT":
            stamped = dict(cmd)
            stamped.setdefault("_worker_recv_s", time.perf_counter())
            acks = self._execute_ar_cmds_and_ack([stamped], preempted_flag=False)
            return acks[-1] if acks else None
        if op in _XFER_OPS:
            return self._dispatch_xfer_cmd(cmd)
        return self._dispatch_main_cmd(cmd)

    def step_once(self, timeout_s: float = 0.05) -> bool:
        """Deterministically process at most one ready command/batch from the runtime queues."""
        if self._drain_late_preempt_cmds():
            return True
        if not self._xfer_thread_running:
            try:
                xcmd = self.xfer_cmd_q.get_nowait()
            except queue.Empty:
                xcmd = None
            if xcmd is not None:
                if xcmd.get("op") != "SHUTDOWN":
                    self._dispatch_xfer_cmd(xcmd)
                return True

        cmd = self._pop_next_main_cmd(timeout=timeout_s)
        if cmd is None:
            return False
        op = cmd.get("op")
        if op == "SHUTDOWN":
            self._handle_shutdown(cmd)
            return True
        if op == "EXEC_AR":
            ar_batch = self._collect_ar_batch(cmd)
            self._execute_ar_cmds_and_ack(ar_batch)
            return True
        self._dispatch_main_cmd(cmd)
        return True

    def serve_forever(self) -> None:
        """Process incoming coordinator commands until `SHUTDOWN`."""
        self._shutdown_requested = False
        xfer_thread = threading.Thread(
            target=self._serve_xfer_forever,
            name=f"ms_xfer_r{self.rank}",
            daemon=True,
        )
        xfer_thread.start()
        try:
            while not self._shutdown_requested:
                self.step_once(timeout_s=0.02)
        finally:
            if xfer_thread.is_alive():
                self.xfer_cmd_q.put({"op": "SHUTDOWN"})
                xfer_thread.join(timeout=5.0)


class WorkerFleet(ABC):
    """Seam between `MorphServeClusterEngine` and worker instances (`mp.spawn` GPU, multi-node TCP, or in-process CPU)."""

    @abstractmethod
    def start(
        self,
        on_ack: Callable[[dict[str, Any]], None],
        on_worker_fault: Callable[[list[int], str], None] | None = None,
    ) -> dict[str, Any]:
        """Start all worker ranks, register `on_ack` & optional `on_worker_fault`, and return startup summary."""

    @abstractmethod
    def send(self, rank: int, payload: dict[str, Any]) -> None:
        """Dispatch a command dictionary to `rank`."""

    @abstractmethod
    def shutdown(self) -> None:
        """Stop all worker threads/processes and clean up transport resources."""


class InProcessWorkerFleet(WorkerFleet):
    """Thin 1-node in-process `WorkerFleet` built atop `LocalNodeWorkerHost(backend='inproc_mem')`."""

    def __init__(
        self,
        policy: Any,
        adapter_factory: Callable[[int, Any], CPTGModelAdapter] | None = None,
        topology: ClusterTopologyProfile | None = None,
    ) -> None:
        self.policy = policy
        self.topology = topology or default_topology_for(int(policy.num_gpus), gpus_per_node=min(8, max(1, int(policy.num_gpus))))
        self.adapter_factory = adapter_factory
        self._host: Any = None

    def start(
        self,
        on_ack: Callable[[dict[str, Any]], None],
        on_worker_fault: Callable[[list[int], str], None] | None = None,
    ) -> dict[str, Any]:
        from pathlib import Path
        from .worker_proc import LocalNodeWorkerHost

        n = int(self.policy.num_gpus)
        self._host = LocalNodeWorkerHost(
            node_rank=0,
            gpus_per_node=n,
            policy=self.policy,
            model_dir=Path("."),
            bench_root=Path("."),
            backend="inproc_mem",
            on_ack=on_ack,
            on_worker_fault=on_worker_fault,
            topology=self.topology,
            adapter_factory=self.adapter_factory,
        )
        self._host.prepare_until_mesh()
        ready = self._host.finalize_mesh_and_start()
        return {
            "num_gpus": n,
            "policy": self.policy.policy_name,
            "host_shmem_delta_gib": float(ready.get("host_shmem_delta_gib", 0.0)),
            "workers": {int(r): info for r, info in (ready.get("workers") or {}).items()},
        }

    def send(self, rank: int, payload: dict[str, Any]) -> None:
        assert self._host is not None, "InProcessWorkerFleet has not been started"
        self._host.send(rank, payload)

    def shutdown(self) -> None:
        if self._host is not None:
            self._host.shutdown()
            self._host = None


__all__ = [
    "InMemoryByteBus",
    "InMemoryPeerMesh",
    "InProcessWorkerFleet",
    "WorkerFleet",
    "WorkerRuntime",
]
