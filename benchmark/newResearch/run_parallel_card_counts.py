"""Parallel, one Amazon item, whole graph, one execute.

Control: original Helium, 1 GPU.
Experiment: helium_new spread, 3 GPUs and 4 GPUs.
Same request settings as the debate reference. One run each.
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
            "HELIUM_VLLM_PORT=18330",
            "HELIUM_VLLM_DEVICE=cuda:0",
            "HELIUM_VLLM_LOGGING_LEVEL=WARNING",
            "",
        ]
    )
)

from bench_programs.parallel.helium import HeliumParallelProgram
from bench_utils.datasets.amazon import AmazonReviewsDataset
from helium.common import GenerationConfig
from helium.ops import LLMOp
from helium.runtime.llm import LLMServiceConfig, LLMServiceInfo
from helium.runtime.protocol import HeliumRequestConfig, SystemProfilingConfig
from helium.runtime.server import HeliumServer as OriginalHeliumServer
from helium.runtime.server import HeliumServerConfig
from helium.utils import iter_batch
from helium_new.runtime.server import HeliumServer as NewHeliumServer

MODEL = "/home/wanghaoyu/weizhihao/models/models/Qwen--Qwen2.5-3B-Instruct/snapshots/master"
RESULT = Path(__file__).resolve().parent / "results" / "parallel_card_counts.txt"
FREE_DEVICES = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]


def _request_config() -> HeliumRequestConfig:
    return HeliumRequestConfig(
        enable_cache_aware_scheduling=True,
        enable_runtime_adjustment=False,
        system_profiling_config=SystemProfilingConfig(),
        query_profiling_config=None,
        precompute_mode="none",
    )


def _server_config(devices: list[str], base_port: int) -> HeliumServerConfig:
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
                    "port": base_port + index,
                    "enable_prefix_caching": True,
                    "gpu_memory_utilization": 0.5,
                    "max_model_len": 8192,
                },
                info=info,
            )
            for index, device in enumerate(devices)
        ],
    )


def _one_item():
    dataset = AmazonReviewsDataset(split="test")
    item, reviews = next(iter(dataset.iter_item_reviews(1, 60)))
    chunks = list(iter_batch(reviews, 10))
    return [(item, chunks)]


async def _run(label: str, devices: list[str], base_port: int, use_new: bool) -> str:
    os.environ["HELIUM_INTRA_GRAPH_PLACEMENT"] = "spread" if use_new else "off"
    cfg = _server_config(devices, base_port)
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
    n_llm = sum(1 for _ in graph.iter_ops(LLMOp))
    system = "helium_new" if use_new else "original"
    print(
        f"=== {label} system={system} devices={devices} llm_ops={n_llm} ===",
        flush=True,
    )
    server_cls = NewHeliumServer if use_new else OriginalHeliumServer
    server = server_cls(cfg)
    server.start()
    started = time.perf_counter()
    try:
        await asyncio.wait_for(
            server.execute({agent.name: graph}, request_config),
            timeout=900,
        )
        elapsed = time.perf_counter() - started
        line = (
            f"{label} system={system} devices={devices} llm_ops={n_llm} "
            f"OK execute_seconds={elapsed:.2f}"
        )
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - started
        line = (
            f"{label} system={system} devices={devices} llm_ops={n_llm} "
            f"TIMEOUT execute_seconds={elapsed:.2f}"
        )
    finally:
        if server.is_started:
            server.close()
    print(line, flush=True)
    return line


async def main() -> None:
    lines = [
        "parallel, one amazon item, 7 experts, 6 chunks of 10 reviews",
        "one whole graph, one execute; CAS=True RA=False max_tokens=512",
        "1 card = original Helium; 3 and 4 cards = helium_new spread",
        "cuda:6 and cuda:7 were occupied and were not used",
    ]
    runs = [
        ("1gpu", FREE_DEVICES[:1], 18330, False),
        ("3gpu", FREE_DEVICES[:3], 18340, True),
        ("4gpu", FREE_DEVICES[:4], 18350, True),
    ]
    for label, devices, port, use_new in runs:
        lines.append(await _run(label, devices, port, use_new))
        RESULT.parent.mkdir(parents=True, exist_ok=True)
        RESULT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
