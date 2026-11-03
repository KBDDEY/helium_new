"""Execute one TaskSpec on helium2 or helium_new2 against pool replicas."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Literal, Sequence

from .dataset import TaskSpec
from .pool import Replica

Flavor = Literal["helium2", "helium_new2"]


def _server_config(replicas: Sequence[Replica], flavor: Flavor):
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
) -> float:
    """Run compiled graph on the chosen flavor; returns execute_seconds only."""
    if not replicas:
        raise ValueError("replicas must be non-empty")

    # helium_new2 spread; helium2 ignores this env (placement always off there).
    os.environ["HELIUM_INTRA_GRAPH_PLACEMENT"] = (
        "spread" if flavor == "helium_new2" else "off"
    )

    HeliumServer = _helium_server_cls(flavor)
    server = HeliumServer(_server_config(replicas, flavor))
    request_config = _request_config()
    server.start()
    started = time.perf_counter()
    try:
        await asyncio.wait_for(
            server.execute({task.agent_name: task.graph}, request_config),
            timeout=timeout_s,
        )
        return time.perf_counter() - started
    finally:
        if server.is_started:
            server.close()


def run_task_blocking(
    task: TaskSpec,
    replicas: Sequence[Replica],
    flavor: Flavor = "helium_new2",
    timeout_s: float = 900.0,
) -> float:
    return asyncio.run(run_task(task, replicas, flavor=flavor, timeout_s=timeout_s))


# Back-compat alias for experiment-group call sites.
def run_task_experiment_blocking(
    task: TaskSpec,
    replicas: Sequence[Replica],
    timeout_s: float = 900.0,
) -> float:
    return run_task_blocking(
        task, replicas, flavor="helium_new2", timeout_s=timeout_s
    )
