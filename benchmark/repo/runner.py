"""Execute one TaskSpec on helium2 or helium_new2 against pool replicas.

Each task should run in its own process (see run_task_in_process): Helium's
background async runner is process-global, so concurrent Helium instances in
threads interfere on close(). The resident vLLM pool stays in the scheduler.

Helium control ports (server / LMCache) are allocated by the scheduler and
passed in — same pattern as vLLM BASE_PORT + index.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

from .pool import Replica
from .dataset import TaskSpec

if TYPE_CHECKING:
    from helium_new2.runtime.tail_resources import TailResourceState

Flavor = Literal["helium2", "helium_new2"]


@dataclass(frozen=True)
class HeliumPorts:
    """Scheduler-assigned Helium-side endpoints for one in-flight task."""

    server_port: int
    lmcache_monitor_port: int

    @property
    def lmcache_controller_url(self) -> str:
        return f"localhost:{self.lmcache_monitor_port}"


def _ensure_worker_paths() -> None:
    """Spawn workers do not inherit parent sys.path edits from __main__."""
    here = Path(__file__).resolve().parent
    benchmark_dir = here.parent
    helium_demo = benchmark_dir.parent
    src = helium_demo / "src"
    for path in (str(src), str(benchmark_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _apply_helium_ports(ports: HeliumPorts) -> str:
    """Install scheduler-assigned ports into env + a tiny lmcache yaml. Returns yaml path."""
    os.environ["HELIUM_SERVER_PORT"] = str(ports.server_port)
    lmcache_path = Path(f"/tmp/helium_repo_lmcache_{ports.lmcache_monitor_port}.yaml")
    lmcache_path.write_text(
        f'controller_url: "{ports.lmcache_controller_url}"\n',
        encoding="utf-8",
    )
    os.environ["LMCACHE_CONFIG_FILE"] = str(lmcache_path)
    return str(lmcache_path)


def _sync_envs_ports(lmcache_path: str) -> None:
    """helium2.envs binds PORT/LMCACHE at import time; refresh after dotenv."""
    import helium2.envs as envs

    envs.HELIUM_SERVER_PORT = int(os.environ["HELIUM_SERVER_PORT"])
    envs.LMCACHE_CONFIG_FILE = lmcache_path


def _server_config(
    replicas: Sequence[Replica],
    flavor: Flavor,
    lmcache_path: str,
    tail_resources: TailResourceState | None = None,
):
    if tail_resources is not None and flavor != "helium_new2":
        raise ValueError("Tail acceleration is only supported by helium_new2")

    from helium2.runtime.cache_manager.config import CacheManagerConfig
    from helium2.runtime.llm import LLMServiceConfig, LLMServiceInfo

    if flavor == "helium_new2":
        from helium_new2.runtime.server import HeliumServerConfig
    else:
        from helium2.runtime.server import HeliumServerConfig

    info = LLMServiceInfo(prefix_caching_enabled=True, cache_capacity=80000)
    return HeliumServerConfig(
        is_local=True,
        benchmarking=False,
        deployment_mode="prod",
        cache_manager_config=CacheManagerConfig(kv_cache_config_file=lmcache_path),
        llm_service_configs=[
            LLMServiceConfig(
                name="vllm-openai",
                args={
                    "model": replica.model,
                    "host": replica.host,
                    "port": replica.port,
                    "api_key": "EMPTY",
                },
                info=info,
            )
            for replica in replicas
        ],
        **({"tail_resources": tail_resources} if tail_resources is not None else {}),
    )


def _request_config():
    from helium2.runtime.protocol import HeliumRequestConfig

    return HeliumRequestConfig(
        enable_cache_aware_scheduling=True,
        enable_runtime_adjustment=True,
        system_profiling_config=None,
        query_profiling_config=None,
        precompute_mode="none",
    )


def _helium_server_cls(flavor: Flavor):
    if flavor == "helium_new2":
        from helium_new2.runtime.server import HeliumServer

        return HeliumServer
    from helium2.runtime.server import HeliumServer

    return HeliumServer


async def run_task(
    task: TaskSpec,
    replicas: Sequence[Replica],
    flavor: Flavor = "helium_new2",
    timeout_s: float = 900.0,
    helium_ports: HeliumPorts | None = None,
    tail_resources: TailResourceState | None = None,
) -> float:
    """Run compiled graph on the chosen flavor; returns execute_seconds only."""
    if not replicas:
        raise ValueError("replicas must be non-empty")
    if helium_ports is None:
        raise ValueError("helium_ports must be assigned by the scheduler")
    if tail_resources is not None and flavor != "helium_new2":
        raise ValueError("Tail acceleration is only supported by helium_new2")

    lmcache_path = _apply_helium_ports(helium_ports)
    _sync_envs_ports(lmcache_path)

    # helium_new2 spread; helium2 ignores this env (placement always off there).
    os.environ["HELIUM_INTRA_GRAPH_PLACEMENT"] = (
        "spread" if flavor == "helium_new2" else "off"
    )

    HeliumServer = _helium_server_cls(flavor)
    server = HeliumServer(
        _server_config(replicas, flavor, lmcache_path, tail_resources=tail_resources)
    )
    request_config = _request_config()
    try:
        server.start()
    except BaseException:
        if tail_resources is not None:
            tail_resources.mark_finished()
            tail_resources.seal()
        raise
    started = time.perf_counter()
    try:
        if tail_resources is not None:
            tail_resources.mark_started()
        await asyncio.wait_for(
            server.execute({task.agent_name: task.graph}, request_config),
            timeout=timeout_s,
        )
        return time.perf_counter() - started
    finally:
        if tail_resources is not None:
            tail_resources.mark_finished()
        try:
            if server.is_started:
                try:
                    server.close()
                except Exception:
                    # Tail grants may be reused only after confirmed cleanup.
                    # Preserve the existing tolerance when this feature is off.
                    if tail_resources is not None:
                        raise
        finally:
            if tail_resources is not None:
                tail_resources.seal()


def run_task_blocking(
    task: TaskSpec,
    replicas: Sequence[Replica],
    flavor: Flavor = "helium_new2",
    timeout_s: float = 900.0,
    helium_ports: HeliumPorts | None = None,
    tail_resources: TailResourceState | None = None,
) -> float:
    return asyncio.run(
        run_task(
            task,
            replicas,
            flavor=flavor,
            timeout_s=timeout_s,
            helium_ports=helium_ports,
            tail_resources=tail_resources,
        )
    )


def run_task_in_process(
    task_blob: bytes,
    replicas: list[Replica] | tuple[Replica, ...],
    flavor: Flavor,
    helium_ports: HeliumPorts,
    timeout_s: float = 900.0,
    tail_resources: TailResourceState | None = None,
) -> float:
    """Fork-pool entry: own Helium lifetime, shared HTTP vLLM only.

    TaskSpec is dill-serialized because CompiledGraph is not stdlib-pickle friendly.
    HeliumPorts come from the scheduler (same idea as Replica.port).

    Intended for ProcessPoolExecutor(mp_context=fork): inherit parent's helium2
    module identity (do not re-alias helium→helium2 here).
    """
    # Scheduler-assigned ports only; modules already consistent via fork inherit.
    lmcache_path = _apply_helium_ports(helium_ports)
    try:
        _sync_envs_ports(lmcache_path)
    except Exception:
        pass

    import dill

    task: TaskSpec = dill.loads(task_blob)
    return run_task_blocking(
        task,
        list(replicas),
        flavor=flavor,
        timeout_s=timeout_s,
        helium_ports=helium_ports,
        tail_resources=tail_resources,
    )


# Back-compat alias for experiment-group call sites.
def run_task_experiment_blocking(
    task: TaskSpec,
    replicas: Sequence[Replica],
    timeout_s: float = 900.0,
    helium_ports: HeliumPorts | None = None,
    tail_resources: TailResourceState | None = None,
) -> float:
    return run_task_blocking(
        task,
        replicas,
        flavor="helium_new2",
        timeout_s=timeout_s,
        helium_ports=helium_ports,
        tail_resources=tail_resources,
    )
