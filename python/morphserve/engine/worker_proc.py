"""Unified `LocalNodeWorkerHost`, `MpSpawnGpuWorkerFleet`, and persistent GPU worker bootstrap process."""

from __future__ import annotations

from dataclasses import replace
import gc
import os
from pathlib import Path
import queue
import threading
import time
import traceback
from typing import Any, Callable, Mapping

from morphserve.models import BagelCPTGAdapter, CPTGModelAdapter, InternVLUCPTGAdapter, SyntheticCPTGAdapter
from morphserve.router import ClusterTopologyProfile, HalfNodeH2DPacer, PcieHalfGroup, default_topology_for
from morphserve.runtime.peer_transport import InMemoryByteBus, NcclPeerAdapter, TcpPeerEndpoint
from morphserve.telemetry import read_shmem_kib
from morphserve.vmm import NumaMemfdMirror

from .config import EnginePolicyConfig
from .worker_runtime import WorkerFleet, WorkerRuntime


def _localize_policy_for_device(policy: EnginePolicyConfig, global_rank: int, local_dev: int) -> EnginePolicyConfig:
    """Align `local_dev` role & initial_alpha with `global_rank` when `global_rank >= gpus_per_node`."""
    if int(global_rank) == int(local_dev):
        return policy
    alphas = dict(policy.initial_alphas)
    alphas[int(local_dev)] = float(policy.initial_alphas.get(int(global_rank), 1.0))
    und_set = set(policy.static_und_ranks)
    gen_set = set(policy.static_gen_ranks)
    if int(global_rank) in und_set:
        und_set.add(int(local_dev))
        gen_set.discard(int(local_dev))
    elif int(global_rank) in gen_set:
        gen_set.add(int(local_dev))
        und_set.discard(int(local_dev))
    else:
        und_set.discard(int(local_dev))
        gen_set.discard(int(local_dev))
    return replace(
        policy,
        initial_alphas=alphas,
        static_und_ranks=tuple(sorted(und_set)),
        static_gen_ranks=tuple(sorted(gen_set)),
    )


def _build_local_pacer_topology(
    global_topo: ClusterTopologyProfile,
    global_rank: int,
    eff_gpn: int,
    dev_cnt: int,
) -> ClusterTopologyProfile:
    """Build a device-index-keyed topology mapping each local CUDA device on this node to its global PCIe half-group."""
    node_idx = int(global_rank) // max(1, int(eff_gpn))
    node_start = node_idx * max(1, int(eff_gpn))
    node_end = min(int(global_topo.num_gpus), node_start + max(1, int(eff_gpn)))

    by_gid: dict[int, tuple[int, list[int], float, float]] = {}
    for gr in range(node_start, node_end):
        grp = global_topo.half_group_of(gr)
        ldev = gr % max(1, int(dev_cnt))
        entry = by_gid.get(grp.group_id)
        if entry is None:
            by_gid[grp.group_id] = (grp.numa_node, [ldev], grp.aggregate_h2d_gbps, grp.solo_h2d_gbps)
        elif ldev not in entry[1]:
            entry[1].append(ldev)

    local_groups = [
        PcieHalfGroup(
            group_id=gid,
            numa_node=numa,
            gpu_ids=tuple(sorted(devs)),
            aggregate_h2d_gbps=agg_bw,
            solo_h2d_gbps=solo_bw,
        )
        for gid, (numa, devs, agg_bw, solo_bw) in sorted(by_gid.items())
    ]
    n_local = max(1, max((max(g.gpu_ids) + 1 for g in local_groups if g.gpu_ids), default=1))
    return ClusterTopologyProfile(
        num_gpus=n_local,
        gpu_names=[global_topo.gpu_names[0] if global_topo.gpu_names else "NVIDIA A100-PCIE-40GB"] * n_local,
        half_groups=local_groups,
        p2p_bw_gbps=[[0.0] * n_local for _ in range(n_local)],
        intra_half_p2p_cap_gbps=global_topo.intra_half_p2p_cap_gbps,
        cross_half_p2p_cap_gbps=global_topo.cross_half_p2p_cap_gbps,
        inter_node_p2p_cap_gbps=global_topo.inter_node_p2p_cap_gbps,
        gpus_per_node=n_local,
        intra_node_handoff_ms=global_topo.intra_node_handoff_ms,
        inter_node_handoff_ms=global_topo.inter_node_handoff_ms,
    )


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
    gpus_per_node: int | None = None,
) -> None:
    """Entry point for each persistent GPU worker process managed by `LocalNodeWorkerHost`."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch

    dev_cnt = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else max(1, gpus_per_node or policy.num_gpus)
    eff_gpn = int(gpus_per_node) if gpus_per_node is not None else min(8, max(1, min(policy.num_gpus, dev_cnt)))
    local_dev = rank % dev_cnt if torch.cuda.is_available() else (rank % eff_gpn)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_dev)

    topo = default_topology_for(policy.num_gpus, gpus_per_node=eff_gpn)
    grp = topo.half_group_of(rank)
    numa_node = grp.numa_node
    pacer_topo = _build_local_pacer_topology(topo, rank, eff_gpn, dev_cnt)
    pacer = HalfNodeH2DPacer(pacer_topo, max_concurrent_per_half=1) if policy.use_h2d_pacer else None
    local_policy = _localize_policy_for_device(policy, rank, local_dev)

    adapter: CPTGModelAdapter | None = None
    shared_mirror: NumaMemfdMirror | None = None
    mesh: NcclPeerAdapter | None = None
    runtime: WorkerRuntime | None = None
    stop_inbox = threading.Event()
    inbox_thread: threading.Thread | None = None

    try:
        adapter = _instantiate_gpu_adapter(
            rank=local_dev,
            policy=local_policy,
            model_dir=Path(model_dir_str),
            bench_root=Path(bench_root_str),
            numa_node=numa_node,
        )

        # Phase 1: Per-half-group (intra-node NUMA socket) shared `NumaMemfdMirror` handshake.
        if policy.use_expert_pager:
            if policy.use_shared_numa_mirror and len(grp.gpu_ids) >= 2:
                is_leader = rank == grp.gpu_ids[0]
                if is_leader:
                    shared_mirror = NumaMemfdMirror.create(
                        nbytes=adapter.total_mirror_bytes,
                        numa_node=numa_node,
                        name=f"ms_eng_{policy.model_family}_grp{grp.group_id}_{os.getpid()}",
                    )
                    adapter.attach_expert_mirror(shared_mirror, populate=True)
                    ack_q.put(
                        {
                            "op": "INIT_LEADER_MIRROR",
                            "rank": rank,
                            "group_id": grp.group_id,
                            "numa_node": numa_node,
                            "pid": os.getpid(),
                            "fd": shared_mirror.fd,
                            "nbytes": shared_mirror.nbytes,
                            "inode": shared_mirror.inode,
                        }
                    )
                else:
                    ack_q.put(
                        {
                            "op": "INIT_FOLLOWER_READY",
                            "rank": rank,
                            "group_id": grp.group_id,
                            "numa_node": numa_node,
                        }
                    )

                attach_msg = cmd_q.get()
                if attach_msg.get("op") != "ATTACH_SHARED_MIRROR":
                    raise RuntimeError(f"unexpected init message: {attach_msg}")
                if not is_leader:
                    leaders_map = attach_msg["leaders"]
                    info = leaders_map.get(str(grp.group_id)) or leaders_map[str(numa_node)]
                    shared_mirror = NumaMemfdMirror.attach_proc_fd(
                        leader_pid=int(info["pid"]),
                        leader_fd=int(info["fd"]),
                        nbytes=int(info["nbytes"]),
                        numa_node=numa_node,
                    )
                    adapter.attach_expert_mirror(shared_mirror, populate=False)
            else:
                adapter.attach_expert_mirror(None, populate=True)

        # Phase 2: Bind TriPool-VMM, LayerExpertPager, execution interceptor & scheduler.
        bind_info = adapter.bind_runtime_resources(pacer=pacer)

        # Phase 3: Full-mesh `NcclPeerAdapter` bootstrap.
        ack_q.put({"op": "READY_FOR_NCCL", "rank": rank})
        nccl_msg = cmd_q.get()
        if nccl_msg.get("op") != "INIT_NCCL_MESH":
            raise RuntimeError(f"expected INIT_NCCL_MESH, got {nccl_msg}")
        if nccl_msg.get("uid_bytes") is not None and policy.num_gpus >= 2:
            mesh = NcclPeerAdapter(
                rank=rank,
                world_size=policy.num_gpus,
                uid_bytes=nccl_msg["uid_bytes"],
                device_index=local_dev,
            )

        # Phase 4: Warmup & start urgent/normal dual-inbox pump into `WorkerRuntime`.
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
                "kv_free_bytes": max(0, int(getattr(adapter, "reported_kv_free_bytes", adapter.available_kv_bytes)())),
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
        if pacer is not None:
            pacer.close()
        gc.collect()
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LocalNodeWorkerHost:
    """Single canonical host managing `gpus_per_node` global ranks across `mp_gpu`, `inproc_mem`, and `inproc_tcp`."""

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
        on_worker_fault: Callable[[list[int], str], None] | None = None,
        advertise_host: str = "127.0.0.1",
        topology: ClusterTopologyProfile | None = None,
        adapter_factory: Callable[[int, EnginePolicyConfig], CPTGModelAdapter] | None = None,
    ) -> None:
        self.node_rank = int(node_rank)
        self.gpus_per_node = int(gpus_per_node)
        self.rank_offset = self.node_rank * self.gpus_per_node
        self.global_ranks = tuple(range(self.rank_offset, self.rank_offset + self.gpus_per_node))
        self.policy = policy
        self.model_dir = Path(model_dir)
        self.bench_root = Path(bench_root)
        # Normalize legacy alias `"inproc_synthetic"` -> `"inproc_tcp"`.
        self.backend = "inproc_tcp" if str(backend) == "inproc_synthetic" else str(backend)
        if self.backend not in ("mp_gpu", "inproc_mem", "inproc_tcp"):
            raise ValueError(f"unsupported LocalNodeWorkerHost backend: {backend!r}")
        self.on_ack = on_ack
        self.on_worker_fault = on_worker_fault
        self.advertise_host = str(advertise_host)
        self.topology = topology or default_topology_for(int(policy.num_gpus), gpus_per_node=self.gpus_per_node)
        self.adapter_factory = adapter_factory

        self._shmem_before_kib = 0
        # mp_gpu state
        self._mp_ctx: Any = None
        self._procs: dict[int, Any] = {}
        self._cmd_qs: dict[int, Any] = {}
        self._ack_q: Any = None
        self._ack_thread: threading.Thread | None = None
        self._stop_ack = threading.Event()
        self._shutting_down = threading.Event()
        self._reported_dead_ranks: set[int] = set()
        self._fault_lock = threading.Lock()

        # inproc (`inproc_mem` / `inproc_tcp`) state
        self._mem_bus: InMemoryByteBus | None = None
        self._tcp_endpoints: dict[int, TcpPeerEndpoint] = {}
        self._runtimes: dict[int, WorkerRuntime] = {}
        self._threads: list[threading.Thread] = []
        self._adapters: list[CPTGModelAdapter] = []
        self._shared_mirrors: dict[int, NumaMemfdMirror] = {}
        self._pacer: HalfNodeH2DPacer | None = None
        self._ready_by_rank: dict[int, dict[str, Any]] = {}

    def _emit_fault(self, ranks: list[int], reason: str) -> None:
        fresh: list[int] = []
        with self._fault_lock:
            for r in ranks:
                ri = int(r)
                if ri not in self._reported_dead_ranks:
                    self._reported_dead_ranks.add(ri)
                    fresh.append(ri)
        if fresh and self.on_worker_fault is not None:
            self.on_worker_fault(fresh, reason)

    def check_worker_liveness(self) -> list[int]:
        """Inspect spawned child worker processes and fire `on_worker_fault` immediately for any crashed ranks."""
        if self._shutting_down.is_set() or not self._procs:
            return []
        dead_info: list[tuple[int, int | None]] = []
        with self._fault_lock:
            for gr, proc in sorted(self._procs.items()):
                if gr not in self._reported_dead_ranks and not proc.is_alive():
                    dead_info.append((gr, proc.exitcode))
        if not dead_info:
            return []
        dead_ranks = [gr for gr, _ in dead_info]
        detail = ", ".join(f"rank {gr} (exitcode={code})" for gr, code in dead_info)
        reason = f"node {self.node_rank} worker process exited unexpectedly: {detail}"
        self._emit_fault(dead_ranks, reason)
        return dead_ranks

    def _drain_local_mp_acks(self, expected_desc: str, timeout_s: float = 360.0) -> dict[int, dict[str, Any]]:
        collected: dict[int, dict[str, Any]] = {}
        deadline = time.monotonic() + float(timeout_s)
        while len(collected) < len(self.global_ranks):
            rem = max(0.1, deadline - time.monotonic())
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"node {self.node_rank} timed out waiting for {expected_desc}; got {sorted(collected.keys())}"
                )
            try:
                msg = self._ack_q.get(timeout=min(2.0, rem))
            except queue.Empty:
                for gr, proc in self._procs.items():
                    if not proc.is_alive() and gr not in collected:
                        raise RuntimeError(
                            f"node {self.node_rank} worker global_rank {gr} exited prematurely (exitcode={proc.exitcode})"
                        )
                continue
            if msg.get("op") == "WORKER_FATAL":
                raise RuntimeError(
                    f"node {self.node_rank} worker {msg.get('rank')} fatal: {msg.get('error')}\n{msg.get('traceback', '')}"
                )
            collected[int(msg["rank"])] = msg
        return collected

    def _setup_inproc_mirror(self, adapter: CPTGModelAdapter, grp: PcieHalfGroup) -> int | None:
        if not self.policy.use_expert_pager:
            return None
        if self.policy.use_shared_numa_mirror and len(grp.gpu_ids) >= 2:
            shared_m = self._shared_mirrors.get(grp.group_id)
            is_first = shared_m is None
            if is_first:
                shared_m = NumaMemfdMirror.create(
                    nbytes=max(4096, adapter.total_mirror_bytes),
                    numa_node=grp.numa_node,
                    name=f"ms_host_n{self.node_rank}_grp{grp.group_id}",
                )
                self._shared_mirrors[grp.group_id] = shared_m
            adapter.attach_expert_mirror(shared_m, populate=is_first)
            return int(shared_m.inode)
        adapter.attach_expert_mirror(None, populate=True)
        return None

    def prepare_until_mesh(self) -> dict[str, Any]:
        """Phase 1-2: Instantiate local workers + per-half `NumaMemfdMirror`, stop right before mesh wiring."""
        self._shmem_before_kib = read_shmem_kib()
        self._shutting_down.clear()
        with self._fault_lock:
            self._reported_dead_ranks.clear()

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
                proc = self._mp_ctx.Process(
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
                proc.start()
                self._procs[gr] = proc

            sample_grp = self.topology.half_group_of(self.global_ranks[0])
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
                        if "group_id" in m:
                            leaders[str(m["group_id"])] = info
                        leaders[str(m["numa_node"])] = info
                for gr in self.global_ranks:
                    self._cmd_qs[gr].put({"op": "ATTACH_SHARED_MIRROR", "leaders": leaders})

            _ = self._drain_local_mp_acks("READY_FOR_NCCL")
            return {"node_rank": self.node_rank, "tcp_endpoints": {}}

        # In-process CPU modes (`inproc_mem` or `inproc_tcp`).
        world_n = int(self.policy.num_gpus)
        self._pacer = HalfNodeH2DPacer(self.topology, max_concurrent_per_half=1) if self.policy.use_h2d_pacer else None
        self._mem_bus = InMemoryByteBus(max(2, world_n))
        factory = self.adapter_factory or (lambda r, pol: SyntheticCPTGAdapter(r, pol))
        tcp_eps: dict[str, list[Any]] = {}

        for gr in self.global_ranks:
            adapter = factory(gr, self.policy)
            grp = self.topology.half_group_of(gr)
            mirror_inode = self._setup_inproc_mirror(adapter, grp)
            info = adapter.bind_runtime_resources(pacer=self._pacer)
            adapter.warmup()

            peer_mesh: Any = None
            if world_n >= 2:
                if self.backend == "inproc_tcp":
                    ep = TcpPeerEndpoint(gr, world_n, bind_host=self.advertise_host)
                    self._tcp_endpoints[gr] = ep
                    tcp_eps[str(gr)] = [self.advertise_host, ep.bound_port]
                    peer_mesh = ep
                else:
                    peer_mesh = self._mem_bus.endpoint_for(gr)

            rt = WorkerRuntime(
                rank=gr,
                policy=self.policy,
                adapter=adapter,
                peer_mesh=peer_mesh,
                ack_sink=self.on_ack,
                shm_bus=self._mem_bus,
            )
            self._adapters.append(adapter)
            self._runtimes[gr] = rt
            self._ready_by_rank[gr] = {
                "rank": gr,
                "resident_alpha": info.get("resident_alpha", self.policy.initial_alphas.get(gr, 1.0)),
                "kv_free_bytes": max(0, int(getattr(adapter, "reported_kv_free_bytes", adapter.available_kv_bytes)())),
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
        """Phase 3-4: Complete `NcclPeerAdapter` or `TcpPeerEndpoint` wiring and start worker command loops."""
        if self.backend == "mp_gpu":
            for gr in self.global_ranks:
                self._cmd_qs[gr].put({"op": "INIT_NCCL_MESH", "uid_bytes": uid_bytes})
            ready_msgs = self._drain_local_mp_acks("WORKER_READY")
            shmem_after_kib = read_shmem_kib()
            shmem_delta_gib = round(max(0.0, (shmem_after_kib - self._shmem_before_kib) / (1024**2)), 2)

            self._stop_ack.clear()
            self._ack_thread = threading.Thread(
                target=self._mp_ack_pump,
                name=f"ms_host_acks_n{self.node_rank}",
                daemon=True,
            )
            self._ack_thread.start()
            return {
                "node_rank": self.node_rank,
                "host_shmem_delta_gib": shmem_delta_gib,
                "workers": {str(r): ready_msgs[r] for r in sorted(ready_msgs.keys())},
            }

        if self.backend == "inproc_tcp" and tcp_directory:
            for ep in self._tcp_endpoints.values():
                ep.set_directory(tcp_directory)
        for gr, rt in sorted(self._runtimes.items()):
            th = threading.Thread(target=rt.serve_forever, name=f"ms_host_w{gr}", daemon=True)
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
                self.check_worker_liveness()
                continue
            if msg.get("op") == "WORKER_FATAL":
                rk = int(msg.get("rank", -1))
                err = f"worker rank {rk} fatal: {msg.get('error', 'unknown')}"
                self._emit_fault([rk], err)
            else:
                self.on_ack(msg)
            self.check_worker_liveness()

    def send(self, global_rank: int, payload: dict[str, Any]) -> None:
        r = int(global_rank)
        if self.backend == "mp_gpu":
            self._cmd_qs[r].put(payload)
        else:
            self._runtimes[r].enqueue_command(payload)

    def shutdown(self) -> None:
        self._shutting_down.set()
        if self.backend == "mp_gpu":
            for gr, q in self._cmd_qs.items():
                try:
                    q.put({"op": "SHUTDOWN", "cmd_id": f"shut_{gr}"})
                except Exception:
                    pass
            for proc in self._procs.values():
                proc.join(timeout=25.0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5.0)
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
            rt.enqueue_command({"op": "SHUTDOWN", "cmd_id": f"shut_{gr}"})
        for th in self._threads:
            th.join(timeout=5.0)
        for ep in self._tcp_endpoints.values():
            ep.close()
        for ad in self._adapters:
            ad.close()
        for m in self._shared_mirrors.values():
            m.close()
        if self._mem_bus is not None:
            self._mem_bus.close()
            self._mem_bus = None
        if self._pacer is not None:
            self._pacer.close()
            self._pacer = None
        self._tcp_endpoints.clear()
        self._runtimes.clear()
        self._threads.clear()
        self._adapters.clear()
        self._shared_mirrors.clear()
        self._ready_by_rank.clear()


class MpSpawnGpuWorkerFleet(WorkerFleet):
    """Thin 1-node `mp.spawn` GPU `WorkerFleet` built atop `LocalNodeWorkerHost(backend='mp_gpu')`."""

    def __init__(
        self,
        policy: EnginePolicyConfig,
        model_dir: Path,
        bench_root: Path,
        *,
        topology: ClusterTopologyProfile | None = None,
    ) -> None:
        self.policy = policy
        self.model_dir = Path(model_dir)
        self.bench_root = Path(bench_root)
        self.topology = topology
        self._host: LocalNodeWorkerHost | None = None

    def start(
        self,
        on_ack: Callable[[dict[str, Any]], None],
        on_worker_fault: Callable[[list[int], str], None] | None = None,
    ) -> dict[str, Any]:
        n = int(self.policy.num_gpus)
        self._host = LocalNodeWorkerHost(
            node_rank=0,
            gpus_per_node=n,
            policy=self.policy,
            model_dir=self.model_dir,
            bench_root=self.bench_root,
            backend="mp_gpu",
            on_ack=on_ack,
            on_worker_fault=on_worker_fault,
            topology=self.topology,
        )
        self._host.prepare_until_mesh()
        uid_bytes = NcclPeerAdapter.generate_unique_id_bytes() if n >= 2 else None
        ready = self._host.finalize_mesh_and_start(uid_bytes=uid_bytes)
        return {
            "num_gpus": n,
            "policy": self.policy.policy_name,
            "host_shmem_delta_gib": float(ready.get("host_shmem_delta_gib", 0.0)),
            "workers": {int(r): info for r, info in (ready.get("workers") or {}).items()},
        }

    def send(self, rank: int, payload: dict[str, Any]) -> None:
        assert self._host is not None, "MpSpawnGpuWorkerFleet has not been started"
        self._host.send(rank, payload)

    def shutdown(self) -> None:
        if self._host is not None:
            self._host.shutdown()
            self._host = None


__all__ = [
    "LocalNodeWorkerHost",
    "MpSpawnGpuWorkerFleet",
    "run_worker_process",
]
