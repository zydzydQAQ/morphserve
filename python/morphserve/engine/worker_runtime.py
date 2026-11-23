"""Torch-free `WorkerRuntime` command state machine (batched AR + preemption) and `InProcessWorkerFleet`."""

from __future__ import annotations

from abc import ABC, abstractmethod
import ctypes
import gc
from multiprocessing import shared_memory
import queue
import threading
import time
import traceback
from typing import Any, Callable

from morphserve._core import get_core_lib, last_error
from morphserve.models.base import CPTGModelAdapter, SyntheticCPTGAdapter, is_oom_exception
from morphserve.router import ClusterTopologyProfile, DEFAULT_OPERATOR_PROFILE, HalfNodeH2DPacer

MIB = 1024**2
GIB = 1024**3
_XFER_OPS = frozenset({"ALLOC_RECV_BUF", "SEND_NCCL", "RECV_NCCL", "DROP_PACKED_CTX", "DROP_RECV_BUF"})


class InMemoryByteBus:
    """Thread-safe pairwise FIFO byte channel backed by C++17 `InMemoryTransportBus` (`ms_mem_bus_*`)."""

    def __init__(self, world_size: int) -> None:
        self._lib = get_core_lib()
        h = int(self._lib.ms_mem_bus_create(ctypes.c_int(max(2, int(world_size)))) or 0)
        if h == 0:
            raise RuntimeError(f"ms_mem_bus_create failed: {last_error(self._lib)}")
        self._handle = h

    def __del__(self) -> None:
        h = getattr(self, "_handle", 0)
        if h:
            self._handle = 0
            try:
                self._lib.ms_mem_bus_destroy(ctypes.c_void_p(h))
            except Exception:
                pass

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
                ctypes.c_void_p(self._handle), name.encode("utf-8"), ctypes.byref(out_ptr)
            )
        )
        if sz <= 0 or not out_ptr.value:
            return b""
        try:
            return ctypes.string_at(out_ptr.value, sz)
        finally:
            self._lib.ms_free_bytes(out_ptr)


class InMemoryPeerMesh:
    """CPU local-substitutable peer mesh backed by C++17 `InMemoryPeerAdapter` (`ms_transport_*`)."""

    def __init__(self, bus: InMemoryByteBus, rank: int) -> None:
        self.bus = bus
        self.rank = int(rank)
        self._lib = bus._lib
        h = int(self._lib.ms_mem_bus_endpoint(ctypes.c_void_p(bus._handle), ctypes.c_int(self.rank)) or 0)
        if h == 0:
            raise RuntimeError(f"ms_mem_bus_endpoint failed: {last_error(self._lib)}")
        self._handle = h

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

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
            raise RuntimeError(f"ms_transport_send(mem) failed: {last_error(self._lib)}")
        return ms

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
            raise TimeoutError(f"ms_transport_recv(mem) failed: {last_error(self._lib)}")
        got = min(nb, int(actual.value))
        sliced = ctypes.string_at(ctypes.addressof(c_buf), got) if got > 0 else b""
        if isinstance(out_tensor, (bytearray, memoryview)):
            if len(out_tensor) < len(sliced):
                raise ValueError(f"out_tensor len {len(out_tensor)} < recv nbytes {len(sliced)}")
            out_tensor[: len(sliced)] = sliced
            return out_tensor, float(elapsed.value)
        return sliced, float(elapsed.value)

    def close(self) -> None:
        h = getattr(self, "_handle", 0)
        if h:
            self._handle = 0
            self._lib.ms_transport_close(ctypes.c_void_p(h))
            self._lib.ms_transport_destroy(ctypes.c_void_p(h))


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


def _record_current_stream_if_cuda_tensor(payload: Any) -> None:
    if getattr(payload, "is_cuda", False) and hasattr(payload, "record_stream"):
        import torch

        payload.record_stream(torch.cuda.current_stream(payload.device))


def _payload_nbytes(payload: Any) -> int:
    return int(payload.numel()) if hasattr(payload, "numel") else len(payload)


def query_worker_kv_free_bytes(adapter: CPTGModelAdapter, policy: Any, rank: int = 0) -> int:
    """Compute instantaneous physical KV headroom in bytes on `adapter` (CUDA VMM, CUDA non-VMM, or CPU sim)."""
    avail_fn = getattr(adapter, "available_kv_bytes", None)
    if callable(avail_fn) and getattr(type(adapter), "available_kv_bytes", None) is not CPTGModelAdapter.available_kv_bytes:
        return max(0, int(avail_fn()))
    attr = getattr(adapter, "kv_free_bytes", None)
    if callable(attr):
        return max(0, int(attr()))
    if isinstance(attr, (int, float)):
        return max(0, int(attr))

    arch = str(getattr(adapter, "architecture", getattr(policy, "cptg_architecture", "tight_mot")))
    is_bridge = arch == "bridge_mllm_dit"
    vmm = getattr(adapter, "vmm", None) or getattr(adapter, "_vmm", None)

    if vmm is not None:
        st = vmm.stats_dict()
        gran = int(st.get("min_granularity_bytes", 2 * MIB))
        super_b = int(st.get("super_handle_bytes", 64 * MIB))
        free_in_vmm = int(st.get("fine_pool_free_pages", 0)) * gran + int(st.get("super_pool_free_handles", 0)) * super_b
        if bool(getattr(vmm, "is_simulated", False)):
            active_logical_b = sum(
                int(getattr(e, "logical_bytes", 0)) for e in getattr(adapter, "_contexts", {}).values()
            )
            kv_mapped_b = int(st.get("kv_mapped_pages_current", 0)) * gran
            if not bool(getattr(policy, "allow_vmm_overflow", True)):
                expert_b = (
                    int(st.get("expert_mapped_handles_current", 0)) * super_b
                    if bool(getattr(policy, "use_expert_pager", False))
                    else 0
                )
                extra_logical = max(0, active_logical_b - kv_mapped_b)
                return max(0, free_in_vmm + expert_b - extra_logical)
            frac = float(getattr(policy, "gpu_memory_fraction", None) or 1.0)
            total_b = int(40 * GIB * frac)
            base_b = 4 * GIB if is_bridge else 15 * GIB
            needs_exp = bool(getattr(policy, "use_expert_pager", False)) or (policy.role_of(rank) != "und")
            exp_b = (5 * GIB if is_bridge else 13 * GIB) if needs_exp else 0
            reserve_b = 512 * MIB if is_bridge else 2 * GIB
            budget_b = max(0, total_b - base_b - exp_b - reserve_b)
            return max(0, budget_b - max(active_logical_b, kv_mapped_b))

        overflow_free = 0
        if bool(getattr(policy, "allow_vmm_overflow", True)):
            import torch

            dev = getattr(adapter, "device", f"cuda:{rank}")
            dev_free, dev_total = torch.cuda.mem_get_info(dev)
            cached_free = max(0, int(torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)))
            raw_outside = int(dev_free) + cached_free
            mem_frac = getattr(policy, "gpu_memory_fraction", None)
            if mem_frac is not None:
                vmm_pool_b = (
                    int(st.get("super_pool_total_handles", 0)) * super_b
                    + int(st.get("fine_pool_total_pages", 0)) * gran
                )
                quota_b = int(float(mem_frac) * int(dev_total))
                raw_outside = min(raw_outside, max(0, quota_b - int(torch.cuda.memory_allocated(dev)) - vmm_pool_b))
            reserve_b = 512 * MIB if is_bridge else 2 * GIB
            overflow_free = max(0, raw_outside - reserve_b)

        cur_expert_b = int(st.get("expert_mapped_handles_current", 0)) * super_b
        tot_expert_b = (
            int(getattr(adapter, "total_mirror_bytes", 0)) if bool(getattr(policy, "use_expert_pager", False)) else 0
        )
        missing_expert_b = max(0, tot_expert_b - cur_expert_b)
        return max(0, free_in_vmm + overflow_free - missing_expert_b)

    if hasattr(adapter, "device"):
        import torch

        dev = adapter.device
        dev_free, dev_total = torch.cuda.mem_get_info(dev)
        cached_free = max(0, int(torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)))
        raw_free = int(dev_free) + cached_free
        mem_frac = getattr(policy, "gpu_memory_fraction", None)
        if mem_frac is not None:
            quota_b = int(float(mem_frac) * int(dev_total))
            raw_free = min(raw_free, max(0, quota_b - int(torch.cuda.memory_allocated(dev))))
        naive_need_b = (
            int(DEFAULT_OPERATOR_PROFILE.expert_bytes("internvl_u" if is_bridge else "bagel"))
            if bool(getattr(policy, "use_naive_offload", False))
            else 0
        )
        reserve_b = 512 * MIB if is_bridge else 2 * GIB
        return max(0, raw_free - naive_need_b - reserve_b)

    active_logical_b = sum(int(getattr(e, "logical_bytes", 0)) for e in getattr(adapter, "_contexts", {}).values())
    frac = float(getattr(policy, "gpu_memory_fraction", None) or 1.0)
    is_und_only = getattr(policy, "routing_mode", "") == "static_nccl_fission" and policy.role_of(rank) == "und"
    base_budget = int((22 * GIB if is_und_only else 10 * GIB) * frac)
    return max(0, base_budget - active_logical_b)


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
        self.oom_count = 0

    def _current_kv_free_bytes(self) -> int:
        return query_worker_kv_free_bytes(self.adapter, self.policy, self.rank)

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
        if op == "EXEC_AR_PREEMPT":
            self.ar_preempt_q.put(msg)
        elif op in _XFER_OPS:
            self.xfer_cmd_q.put(msg)
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
            snap = self._telemetry_snapshot()
            for cmd, ar_res in zip(ar_cmds, batch_outs):
                ack = {
                    "cmd_id": cmd["cmd_id"],
                    "rank": self.rank,
                    "status": "ok",
                    "ctx_id": str(cmd["ctx_id"]),
                    **ar_res,
                    **snap,
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
                self.ack_sink(
                    {
                        "cmd_id": cmd["cmd_id"],
                        "rank": self.rank,
                        "status": "oom",
                        **self._telemetry_snapshot(),
                        "error": str(exc)[:200],
                    }
                )
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

    def _serve_xfer_forever(self) -> None:
        """Dedicated transfer thread executing `DirectNcclMesh` P2P ops concurrently with `main_cmd_q` compute."""
        while True:
            cmd = self.xfer_cmd_q.get()
            if cmd is None:
                continue
            op = cmd.get("op")
            cmd_id = cmd.get("cmd_id")
            if op == "SHUTDOWN":
                break
            try:
                if op == "ALLOC_RECV_BUF":
                    assert self.peer_mesh is not None
                    nbytes = max(0, int(cmd["nbytes"]))
                    recv_id = f"rbuf_{self.rank}_{cmd_id}"
                    if hasattr(self.peer_mesh, "alloc_recv_u8"):
                        buf = self.peer_mesh.alloc_recv_u8(nbytes)
                    else:
                        buf = bytearray(nbytes)
                    with self._xfer_store_lock:
                        self._recv_store[recv_id] = buf
                    self.ack_sink(
                        {
                            "cmd_id": cmd_id,
                            "rank": self.rank,
                            "status": "ok",
                            "recv_id": recv_id,
                            "nbytes": nbytes,
                        }
                    )

                elif op == "DROP_PACKED_CTX":
                    with self._xfer_store_lock:
                        entry = self._packed_store.pop(str(cmd["packed_id"]), None)
                    del entry
                    self.ack_sink({"cmd_id": cmd_id, "rank": self.rank, "status": "ok"})

                elif op == "DROP_RECV_BUF":
                    with self._xfer_store_lock:
                        buf = self._recv_store.pop(str(cmd["recv_id"]), None)
                    del buf
                    self.ack_sink({"cmd_id": cmd_id, "rank": self.rank, "status": "ok"})

                elif op == "SEND_NCCL":
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
                    self.ack_sink(
                        {
                            "cmd_id": cmd_id,
                            "rank": self.rank,
                            "status": "ok",
                            "ship_ms": ship_ms,
                            "bytes": nbytes,
                        }
                    )

                elif op == "RECV_NCCL":
                    assert self.peer_mesh is not None
                    meta = cmd["meta"]
                    wire_bytes = int(meta["total_bytes"])
                    recv_id = cmd.get("recv_id")
                    if recv_id is not None:
                        rid_s = str(recv_id)
                        with self._xfer_store_lock:
                            pre_buf = self._recv_store.pop(rid_s, None)
                        raw, ship_ms = self.peer_mesh.recv_u8(
                            wire_bytes,
                            src_rank=int(cmd["src_rank"]),
                            out_tensor=pre_buf,
                        )
                        with self._xfer_store_lock:
                            self._recv_store[rid_s] = raw
                        self.ack_sink(
                            {
                                "cmd_id": cmd_id,
                                "rank": self.rank,
                                "status": "ok",
                                "recv_id": rid_s,
                                "ship_ms": ship_ms,
                                "bytes": int(meta.get("logical_bytes", wire_bytes)),
                            }
                        )
                    else:
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
                                **self._telemetry_snapshot(),
                            }
                        )

                else:
                    self.ack_sink(
                        {"cmd_id": cmd_id, "rank": self.rank, "status": "error", "error": f"unknown xfer op: {op}"}
                    )

            except BaseException as exc:
                if is_oom_exception(exc):
                    probe_only = bool(op == "ALLOC_RECV_BUF" and cmd.get("probe_only", False))
                    if not probe_only:
                        self.oom_count += 1
                    gc.collect()
                    _clear_cuda_cache_if_present()
                    self.ack_sink(
                        {
                            "cmd_id": cmd_id,
                            "rank": self.rank,
                            "status": "oom",
                            "probe_only": probe_only,
                            "error": str(exc)[:300],
                        }
                    )
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

    def serve_forever(self) -> None:
        """Process incoming coordinator commands until `SHUTDOWN`."""
        xfer_thread = threading.Thread(
            target=self._serve_xfer_forever,
            name=f"ms_xfer_r{self.rank}",
            daemon=True,
        )
        xfer_thread.start()
        try:
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
                    self.xfer_cmd_q.put({"op": "SHUTDOWN"})
                    xfer_thread.join(timeout=5.0)
                    with self._xfer_store_lock:
                        self._packed_store.clear()
                        self._recv_store.clear()
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
                                **self._telemetry_snapshot(),
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
                            snap = self._telemetry_snapshot(
                                resident_alpha=ack_msg.get("resident_alpha")
                            )
                            ack_msg = dict(ack_msg, **snap)
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
                                        **self._telemetry_snapshot(),
                                        "error": f"aborted mid-dit: {dit_exc}"[:200],
                                    }
                                )
                            pending_inflight.clear()
                            raise
                        if cmd.get("release_after_dit", False):
                            self.adapter.release_context(cid)
                        self.ack_sink(
                            {
                                "cmd_id": cmd_id,
                                "rank": self.rank,
                                **dit_res,
                                **self._telemetry_snapshot(),
                            }
                        )

                    elif op == "PACK_CTX":
                        cid = str(cmd["ctx_id"])
                        packed = self.adapter.pack_state(cid, include_cfg_dup=bool(cmd.get("include_cfg_dup", False)))
                        ready_ev = _record_cuda_event_if_tensor(packed.payload)
                        packed_id = f"{cid}__{cmd_id}"
                        with self._xfer_store_lock:
                            self._packed_store[packed_id] = (packed.payload, ready_ev)
                        if cmd.get("release_after_pack", False):
                            # Synchronize `ready_ev` BEFORE `release_context` calls `cuMemUnmap` on VmmNaiveCache VA!
                            if ready_ev is not None and hasattr(ready_ev, "synchronize"):
                                ready_ev.synchronize()
                            self.adapter.release_context(cid)
                        self.ack_sink(
                            {
                                "cmd_id": cmd_id,
                                "rank": self.rank,
                                "status": "ok",
                                "packed_id": packed_id,
                                "meta": packed.meta,
                                **self._telemetry_snapshot(),
                            }
                        )

                    elif op == "UNPACK_CTX":
                        recv_id = str(cmd["recv_id"])
                        meta = cmd["meta"]
                        with self._xfer_store_lock:
                            raw = self._recv_store.pop(recv_id)
                        try:
                            _record_current_stream_if_cuda_tensor(raw)
                            logical_b = self.adapter.unpack_state(str(cmd["new_ctx_id"]), meta, raw)
                        finally:
                            del raw
                        self.ack_sink(
                            {
                                "cmd_id": cmd_id,
                                "rank": self.rank,
                                "status": "ok",
                                "bytes": int(meta.get("logical_bytes", logical_b)),
                                **self._telemetry_snapshot(),
                            }
                        )

                    elif op == "SEND_SHM":
                        res = self._send_shm(
                            str(cmd["ctx_id"]),
                            str(cmd["shm_name"]),
                            include_cfg_dup=bool(cmd.get("include_cfg_dup", True)),
                            release_after_send=bool(cmd.get("release_after_send", False)),
                        )
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
                                "resident_layers": int(round(new_alpha * self.adapter.num_expert_layers)),
                                **self._telemetry_snapshot(resident_alpha=new_alpha),
                            }
                        )

                    elif op == "RELEASE_CTX":
                        self.adapter.release_context(str(cmd["ctx_id"]))
                        self.ack_sink(
                            {
                                "cmd_id": cmd_id,
                                "rank": self.rank,
                                "status": "ok",
                                **self._telemetry_snapshot(),
                            }
                        )

                    elif op == "GET_STATS":
                        st = self.adapter.memory_and_vmm_stats()
                        self.ack_sink(
                            {
                                "cmd_id": cmd_id,
                                "rank": self.rank,
                                "status": "ok",
                                "oom_count": self.oom_count,
                                **st,
                                **self._telemetry_snapshot(),
                            }
                        )

                    else:
                        self.ack_sink(
                            {"cmd_id": cmd_id, "rank": self.rank, "status": "error", "error": f"unknown op: {op}"}
                        )

                except BaseException as exc:
                    if is_oom_exception(exc):
                        self.oom_count += 1
                        gc.collect()
                        _clear_cuda_cache_if_present()
                        self.ack_sink(
                            {
                                "cmd_id": cmd_id,
                                "rank": self.rank,
                                "status": "oom",
                                **self._telemetry_snapshot(),
                                "error": str(exc)[:300],
                            }
                        )
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
        finally:
            if xfer_thread.is_alive():
                self.xfer_cmd_q.put({"op": "SHUTDOWN"})
                xfer_thread.join(timeout=2.0)


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
        if topology is not None:
            self.topology = topology
        elif int(policy.num_gpus) == 8:
            self.topology = ClusterTopologyProfile.a100_pcie_8gpu_default()
        else:
            self.topology = ClusterTopologyProfile.from_multinode_measurements(
                world_size=int(policy.num_gpus),
                gpus_per_node=min(8, max(1, int(policy.num_gpus))),
                intra_bw=10.6,
                socket_bw=9.55,
                inter_bw=11.5,
            )
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
            grp_id = self.topology.half_group_of(rank).group_id if rank < self.topology.num_gpus else (rank // 4)
            numa_node = self.topology.numa_node_of(rank) if rank < self.topology.num_gpus else (rank % 2)
            mirror_inode = None
            if self.policy.use_expert_pager:
                if self.policy.use_shared_numa_mirror and n >= 2:
                    shared_m = self._shared_mirrors.get(grp_id)
                    is_first = shared_m is None
                    if is_first:
                        shared_m = NumaMemfdMirror.create(
                            nbytes=max(4096, adapter.total_mirror_bytes),
                            numa_node=numa_node,
                            name=f"ms_inproc_grp{grp_id}",
                        )
                        self._shared_mirrors[grp_id] = shared_m
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
                "kv_free_bytes": query_worker_kv_free_bytes(adapter, self.policy, rank),
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
