"""Multi-process GPU `MpSpawnGpuWorkerFleet` adapter and worker bootstrap process."""

from __future__ import annotations

import gc
import os
from pathlib import Path
import queue
import threading
import time
import traceback
from typing import Any, Callable

from morphserve.models import BagelCPTGAdapter, CPTGModelAdapter, InternVLUCPTGAdapter, SyntheticCPTGAdapter
from morphserve.router import ClusterTopologyProfile, DirectNcclMesh, HalfNodeH2DPacer
from morphserve.telemetry import read_shmem_kib
from morphserve.vmm import NumaMemfdMirror

from .config import EnginePolicyConfig
from .worker_runtime import WorkerFleet, WorkerRuntime


def _instantiate_gpu_adapter(
    rank: int,
    policy: EnginePolicyConfig,
    model_dir: Path,
    bench_root: Path,
    numa_node: int,
) -> CPTGModelAdapter:
    if policy.model_family == "internvl_u":
        return InternVLUCPTGAdapter(
            rank=rank,
            policy=policy,
            model_dir=model_dir,
            bench_root=bench_root,
            numa_node=numa_node,
        )
    if policy.model_family == "synthetic":
        return SyntheticCPTGAdapter(rank=rank, policy=policy)
    return BagelCPTGAdapter(
        rank=rank,
        policy=policy,
        model_dir=model_dir,
        bench_root=bench_root,
        numa_node=numa_node,
    )


def run_worker_process(
    rank: int,
    policy: EnginePolicyConfig,
    model_dir_str: str,
    bench_root_str: str,
    cmd_q: Any,
    ack_q: Any,
) -> None:
    """Entry point for each persistent GPU worker process managed by `MpSpawnGpuWorkerFleet`."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch

    local_dev = rank % max(1, torch.cuda.device_count()) if torch.cuda.is_available() else rank
    if torch.cuda.is_available():
        torch.cuda.set_device(local_dev)

    topo = (
        ClusterTopologyProfile.a100_pcie_8gpu_default()
        if policy.num_gpus == 8
        else ClusterTopologyProfile.from_multinode_measurements(
            world_size=policy.num_gpus,
            gpus_per_node=min(8, max(1, policy.num_gpus)),
            intra_bw=10.6,
            socket_bw=9.55,
            inter_bw=11.5,
        )
    )
    numa_node = topo.numa_node_of(rank)
    pacer = HalfNodeH2DPacer(topo, max_concurrent_per_half=1) if policy.use_h2d_pacer else None

    adapter: CPTGModelAdapter | None = None
    shared_mirror: NumaMemfdMirror | None = None
    mesh: DirectNcclMesh | None = None
    runtime: WorkerRuntime | None = None
    stop_inbox = threading.Event()
    inbox_thread: threading.Thread | None = None

    try:
        adapter = _instantiate_gpu_adapter(
            rank=local_dev,
            policy=policy,
            model_dir=Path(model_dir_str),
            bench_root=Path(bench_root_str),
            numa_node=numa_node,
        )

        # Phase 1: Per-NUMA-socket shared `NumaMemfdMirror` handshake
        if policy.use_expert_pager:
            if policy.use_shared_numa_mirror and policy.num_gpus >= 2:
                grp = topo.half_group_of(rank)
                is_leader = rank == grp.gpu_ids[0]
                if is_leader:
                    shared_mirror = NumaMemfdMirror.create(
                        nbytes=adapter.total_mirror_bytes,
                        numa_node=numa_node,
                        name=f"ms_eng_{policy.model_family}_numa{numa_node}_{os.getpid()}",
                    )
                    adapter.attach_expert_mirror(shared_mirror, populate=True)
                    ack_q.put(
                        {
                            "op": "INIT_LEADER_MIRROR",
                            "rank": rank,
                            "numa_node": numa_node,
                            "pid": os.getpid(),
                            "fd": shared_mirror.fd,
                            "nbytes": shared_mirror.nbytes,
                            "inode": shared_mirror.inode,
                        }
                    )
                else:
                    ack_q.put({"op": "INIT_FOLLOWER_READY", "rank": rank, "numa_node": numa_node})

                attach_msg = cmd_q.get()
                if attach_msg.get("op") != "ATTACH_SHARED_MIRROR":
                    raise RuntimeError(f"unexpected init message: {attach_msg}")
                if not is_leader:
                    info = attach_msg["leaders"][str(numa_node)]
                    shared_mirror = NumaMemfdMirror.attach_proc_fd(
                        leader_pid=int(info["pid"]),
                        leader_fd=int(info["fd"]),
                        nbytes=int(info["nbytes"]),
                        numa_node=numa_node,
                    )
                    adapter.attach_expert_mirror(shared_mirror, populate=False)
            else:
                adapter.attach_expert_mirror(None, populate=True)

        # Phase 2: Bind TriPool-VMM, LayerExpertPager, execution interceptor & scheduler
        bind_info = adapter.bind_runtime_resources(pacer=pacer)

        # Phase 3: Full-mesh `DirectNcclMesh` bootstrap
        ack_q.put({"op": "READY_FOR_NCCL", "rank": rank})
        nccl_msg = cmd_q.get()
        if nccl_msg.get("op") != "INIT_NCCL_MESH":
            raise RuntimeError(f"expected INIT_NCCL_MESH, got {nccl_msg}")
        if nccl_msg.get("uid_bytes") is not None and policy.num_gpus >= 2:
            mesh = DirectNcclMesh(
                rank=rank,
                world_size=policy.num_gpus,
                uid_bytes=nccl_msg["uid_bytes"],
                device_index=local_dev,
            )

        # Phase 4: Warmup & start urgent/normal dual-inbox pump into `WorkerRuntime`
        adapter.warmup()
        st = adapter.memory_and_vmm_stats()
        runtime = WorkerRuntime(
            rank=rank,
            policy=policy,
            adapter=adapter,
            peer_mesh=mesh,
            ack_sink=ack_q.put,
        )

        def _inbox_pump() -> None:
            while not stop_inbox.is_set():
                try:
                    msg = cmd_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if msg is None:
                    continue
                runtime.enqueue_command(msg)
                if msg.get("op") == "SHUTDOWN":
                    break

        inbox_thread = threading.Thread(target=_inbox_pump, name=f"ms_inbox_r{rank}", daemon=True)
        inbox_thread.start()

        ack_q.put(
            {
                "op": "WORKER_READY",
                "rank": rank,
                "resident_alpha": adapter.resident_alpha(),
                "reserved_mib": st.get("peak_reserved_mib", bind_info.get("reserved_mib", 0.0)),
                "mirror_inode": shared_mirror.inode if shared_mirror is not None else None,
            }
        )

        runtime.serve_forever()

    except Exception as exc:
        ack_q.put(
            {
                "op": "WORKER_FATAL",
                "rank": rank,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-1500:],
            }
        )
    finally:
        stop_inbox.set()
        if inbox_thread is not None:
            try:
                inbox_thread.join(timeout=1.0)
            except Exception:
                pass
        if mesh is not None:
            mesh.close()
        if adapter is not None:
            adapter.close()
        if shared_mirror is not None:
            shared_mirror.close()
        gc.collect()
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MpSpawnGpuWorkerFleet(WorkerFleet):
    """Production `mp.spawn` `WorkerFleet` managing persistent GPU processes & `DirectNcclMesh`."""

    def __init__(self, policy: EnginePolicyConfig, model_dir: Path, bench_root: Path) -> None:
        self.policy = policy
        self.model_dir = Path(model_dir)
        self.bench_root = Path(bench_root)
        self._mp_ctx: Any = None
        self._procs: list[Any] = []
        self._cmd_qs: list[Any] = []
        self._ack_q: Any = None
        self._ack_thread: threading.Thread | None = None
        self._stop_ack = threading.Event()
        self._on_ack: Callable[[dict[str, Any]], None] | None = None

    def _drain_init_acks(self, expected_op: str) -> dict[int, dict[str, Any]]:
        collected: dict[int, dict[str, Any]] = {}
        deadline = time.monotonic() + 300.0
        while len(collected) < self.policy.num_gpus:
            rem = max(0.1, deadline - time.monotonic())
            if rem <= 0.1 and time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {expected_op}; got ranks {sorted(collected.keys())}")
            try:
                msg = self._ack_q.get(timeout=min(2.0, rem))
            except queue.Empty:
                for rank, p in enumerate(self._procs):
                    if not p.is_alive() and rank not in collected:
                        raise RuntimeError(f"worker rank {rank} exited prematurely (exitcode={p.exitcode})")
                continue
            if msg.get("op") == "WORKER_FATAL":
                raise RuntimeError(f"worker rank {msg.get('rank')} fatal: {msg.get('error')}\n{msg.get('traceback', '')}")
            collected[int(msg["rank"])] = msg
        return collected

    def start(self, on_ack: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        import torch
        import torch.multiprocessing as mp

        n = self.policy.num_gpus
        if torch.cuda.is_available() and torch.cuda.device_count() < n:
            raise RuntimeError(f"policy requires {n} GPUs, saw {torch.cuda.device_count()}")

        self._on_ack = on_ack
        shmem_before_kib = read_shmem_kib()
        self._mp_ctx = mp.get_context("spawn")
        self._cmd_qs = [self._mp_ctx.Queue() for _ in range(n)]
        self._ack_q = self._mp_ctx.Queue()
        self._procs = []

        for rank in range(n):
            p = self._mp_ctx.Process(
                target=run_worker_process,
                args=(
                    rank,
                    self.policy,
                    str(self.model_dir),
                    str(self.bench_root),
                    self._cmd_qs[rank],
                    self._ack_q,
                ),
                daemon=False,
            )
            p.start()
            self._procs.append(p)

        if self.policy.use_shared_numa_mirror and self.policy.use_expert_pager and n >= 2:
            init_msgs = self._drain_init_acks("INIT_LEADER_MIRROR/FOLLOWER")
            leaders: dict[str, dict[str, Any]] = {}
            for m in init_msgs.values():
                if m.get("op") == "INIT_LEADER_MIRROR":
                    leaders[str(m["numa_node"])] = {
                        "pid": m["pid"],
                        "fd": m["fd"],
                        "nbytes": m["nbytes"],
                        "inode": m["inode"],
                    }
            for rank in range(n):
                self._cmd_qs[rank].put({"op": "ATTACH_SHARED_MIRROR", "leaders": leaders})

        _ = self._drain_init_acks("READY_FOR_NCCL")
        uid_bytes = DirectNcclMesh.generate_unique_id_bytes() if n >= 2 else None
        for rank in range(n):
            self._cmd_qs[rank].put({"op": "INIT_NCCL_MESH", "uid_bytes": uid_bytes})

        ready_msgs = self._drain_init_acks("WORKER_READY")
        shmem_after_kib = read_shmem_kib()
        shmem_delta_gib = round(max(0.0, (shmem_after_kib - shmem_before_kib) / (1024**2)), 2)

        self._stop_ack.clear()
        self._ack_thread = threading.Thread(target=self._ack_loop, name="ms_coord_acks", daemon=True)
        self._ack_thread.start()

        return {
            "num_gpus": n,
            "policy": self.policy.policy_name,
            "host_shmem_delta_gib": shmem_delta_gib,
            "workers": {r: ready_msgs[r] for r in sorted(ready_msgs.keys())},
        }

    def _ack_loop(self) -> None:
        while not self._stop_ack.is_set():
            try:
                msg = self._ack_q.get(timeout=0.05)
            except queue.Empty:
                continue
            if self._on_ack is not None:
                self._on_ack(msg)

    def send(self, rank: int, payload: dict[str, Any]) -> None:
        self._cmd_qs[int(rank)].put(payload)

    def shutdown(self) -> None:
        for rank, q in enumerate(self._cmd_qs):
            try:
                q.put({"op": "SHUTDOWN", "cmd_id": f"shut_{rank}"})
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=20.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)
        self._stop_ack.set()
        if self._ack_thread is not None:
            self._ack_thread.join(timeout=2.0)
            self._ack_thread = None
        for q in self._cmd_qs:
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
