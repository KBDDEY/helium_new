"""Resident vLLM OpenAI-compatible engine pool (one process per DEVICE)."""

from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from multiprocessing import Process
from typing import Sequence

from . import config as cfg


@dataclass(frozen=True)
class Replica:
    device: str
    host: str
    port: int
    model: str

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


def _engine_main(device: str, host: str, port: int, model: str) -> None:
    """Child process entry: start one OpenAI HTTP engine on the given physical GPU.

    Same binding style as newResearch ref scripts: pass device=\"cuda:N\" and let
    helium2 start_server set CUDA_VISIBLE_DEVICES from it.
    """
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    from helium2.runtime.utils.vllm.config import BenchVLLMServerConfig
    from helium2.runtime.utils.vllm.server import start_server

    server_cfg = BenchVLLMServerConfig(
        model=model,
        host=host,
        port=port,
        device=device,
        enable_prefix_caching=cfg.ENABLE_PREFIX_CACHING,
        gpu_memory_utilization=cfg.GPU_MEMORY_UTILIZATION,
        max_model_len=cfg.MAX_MODEL_LEN,
        log_level="WARNING",
        uvicorn_log_level="warning",
        benchmarking=False,
    )
    start_server(server_cfg.compile())


def _wait_ready(replica: Replica, timeout_s: float) -> None:
    url = f"{replica.base_url}/models"
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    return
        except Exception as exc:  # noqa: BLE001 - poll until ready
            last_err = exc
        time.sleep(cfg.POOL_POLL_INTERVAL_S)
    raise TimeoutError(
        f"vLLM replica on {replica.device} port {replica.port} not ready "
        f"within {timeout_s}s; last_error={last_err!r}"
    )


class VLLMPool:
    """Starts N engines at construction time; acquire/release by replica list."""

    def __init__(
        self,
        devices: Sequence[str] | None = None,
        host: str | None = None,
        base_port: int | None = None,
        model: str | None = None,
    ) -> None:
        self.devices = list(devices if devices is not None else cfg.DEVICES)
        self.host = host if host is not None else cfg.HOST
        self.base_port = base_port if base_port is not None else cfg.BASE_PORT
        self.model = model if model is not None else cfg.MODEL
        self._procs: list[Process] = []
        self._replicas: list[Replica] = []
        self._free: list[Replica] = []
        self._started = False

    @property
    def replicas(self) -> list[Replica]:
        return list(self._replicas)

    @property
    def free_replicas(self) -> list[Replica]:
        return list(self._free)

    def start(self) -> None:
        if self._started:
            return
        if not self.devices:
            raise ValueError("DEVICES is empty")

        for index, device in enumerate(self.devices):
            port = self.base_port + index
            replica = Replica(
                device=device, host=self.host, port=port, model=self.model
            )
            # daemon=False: vLLM V1 spawns engine-core children; daemons cannot.
            proc = Process(
                target=_engine_main,
                args=(device, self.host, port, self.model),
                name=f"vllm-{device.replace(':', '')}-{port}",
                daemon=False,
            )
            proc.start()
            self._procs.append(proc)
            self._replicas.append(replica)

        for replica in self._replicas:
            _wait_ready(replica, cfg.POOL_READY_TIMEOUT_S)

        self._free = list(self._replicas)
        self._started = True

    def stop(self) -> None:
        for proc in self._procs:
            if proc.is_alive():
                proc.terminate()
        for proc in self._procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
        self._procs.clear()
        self._replicas.clear()
        self._free.clear()
        self._started = False

    def acquire(self, k: int) -> list[Replica]:
        if not self._started:
            raise RuntimeError("Pool has not been started")
        if k < 1:
            raise ValueError("k must be >= 1")
        if k > len(self._free):
            raise RuntimeError(
                f"Need {k} free replicas, only {len(self._free)} available"
            )
        taken = self._free[:k]
        self._free = self._free[k:]
        return taken

    def acquire_replicas(self, replicas: Sequence[Replica]) -> list[Replica]:
        """Take the exact replicas chosen by a strategy (must all be free)."""
        if not self._started:
            raise RuntimeError("Pool has not been started")
        if not replicas:
            raise ValueError("replicas must be non-empty")
        free_set = set(self._free)
        for replica in replicas:
            if replica not in free_set:
                raise RuntimeError(f"Replica not free: {replica}")
        wanted = set(replicas)
        self._free = [r for r in self._free if r not in wanted]
        return list(replicas)

    def release(self, replicas: Sequence[Replica]) -> None:
        for replica in replicas:
            if replica not in self._replicas:
                raise ValueError(f"Unknown replica: {replica}")
            if replica in self._free:
                raise ValueError(f"Replica already free: {replica}")
            self._free.append(replica)
