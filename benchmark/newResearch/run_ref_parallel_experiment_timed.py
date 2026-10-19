"""Temporary experiment. Same setup as run_ref_debate_experiment_timed.py, parallel graph.

helium_new, three GPUs, one parallel task, one whole graph, one execute.
The clock starts after server.start() and covers only server.execute.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
(_ROOT / ".env").write_text(
    "\n".join(
        [
            "HELIUM_DEPLOYMENT_MODE=prod",
            "HELIUM_SERVER_HOST=127.0.0.1",
            "HELIUM_SERVER_PORT=18080",
            "HELIUM_LLM_SERVICES=vllm-local",
            "LLM_BASE_URL=http://127.0.0.1:9/v1",
            "LLM_MODEL=qwen",
            "HELIUM_VLLM_MODEL=qwen",
            "HELIUM_VLLM_HOST=127.0.0.1",
            "HELIUM_VLLM_PORT=18320",
            "HELIUM_VLLM_DEVICE=cuda:0",
            "HELIUM_VLLM_LOGGING_LEVEL=WARNING",
            "",
        ]
    )
)
os.environ["HELIUM_INTRA_GRAPH_PLACEMENT"] = "spread"

from bench_programs.parallel.helium import HeliumParallelProgram
from bench_utils.datasets.amazon import AmazonReviewsDataset
from helium.common import GenerationConfig
from helium.ops import LLMOp
from helium.runtime.llm import LLMServiceConfig, LLMServiceInfo
from helium.runtime.protocol import HeliumRequestConfig, SystemProfilingConfig
from helium_new.runtime.server import HeliumServer, HeliumServerConfig
from helium.utils import iter_batch

MODEL = "/home/wanghaoyu/weizhihao/models/models/Qwen--Qwen2.5-3B-Instruct/snapshots/master"
RESULT = Path(__file__).resolve().parent / "results" / "ref_parallel_experiment_timed.txt"
DEVICES = ["cuda:0", "cuda:1", "cuda:2"]


def _server_config() -> HeliumServerConfig:
    info = LLMServiceInfo(prefix_caching_enabled=True, cache_capacity=80000)
    return HeliumServerConfig(
        is_local=True,
        benchmarking=True,
        deployment_mode="prod",
        llm_service_configs=[
            LLMServiceConfig(
                name="vllm-local",
                args={
                    "model": MODEL,
                    "device": device,
                    "port": 18320 + index,
                    "enable_prefix_caching": True,
                    "gpu_memory_utilization": 0.5,
                    "max_model_len": 8192,
                },
                info=info,
            )
            for index, device in enumerate(DEVICES)
        ],
    )


def _request_config() -> HeliumRequestConfig:
    return HeliumRequestConfig(
        enable_cache_aware_scheduling=True,
        enable_runtime_adjustment=True,
        system_profiling_config=SystemProfilingConfig(),
        query_profiling_config=None,
        precompute_mode="none",
    )


def _one_item():
    dataset = AmazonReviewsDataset(split="test")
    item, reviews = next(iter(dataset.iter_item_reviews(1, 60)))
    chunks = list(iter_batch(reviews, 10))
    return [(item, chunks)]


async def main() -> None:
    cfg = _server_config()
    request_config = _request_config()
    generation = GenerationConfig(
        model=MODEL,
        base_url="http://127.0.0.1:9/v1",
        temperature=0,
        max_tokens=512,
    )
    program = HeliumParallelProgram(server_config=cfg, request_config=request_config)
    prepared = program.prepare_kwargs(
        dataset="amazon",
        item_reviews=_one_item(),
        num_experts=7,
        num_review_chunks_per_item=6,
        generation_config=generation,
    )
    agent = program.create_agent(**prepared)
    graph = agent.get_and_reset_compiled_graph()
    print(
        f"experiment: helium_new, devices={DEVICES}, 1 Amazon item, 7 experts, 6 chunks",
        flush=True,
    )
    print(
        "entry=prepare_kwargs + create_agent + one execute; "
        "timer=execute only, after server.start(); "
        "HELIUM_INTRA_GRAPH_PLACEMENT=spread; CAS=True RA=True max_tokens=512",
        flush=True,
    )
    print(f"llm_ops={sum(1 for _ in graph.iter_ops(LLMOp))}", flush=True)
    server = HeliumServer(cfg)
    print("starting server", flush=True)
    server.start()
    print("server started", flush=True)
    started = time.perf_counter()
    try:
        await asyncio.wait_for(
            server.execute({agent.name: graph}, request_config),
            timeout=900,
        )
        elapsed = time.perf_counter() - started
        line = f"OK execute_seconds={elapsed:.2f}"
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - started
        line = f"TIMEOUT execute_seconds={elapsed:.2f}"
    finally:
        if server.is_started:
            server.close()
    print(line, flush=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(line + "\n")


if __name__ == "__main__":
    asyncio.run(main())
