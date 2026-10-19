"""Test only. helium_new, 3 GPUs, one debate whole graph, one execute.

Two runs on the same cards. The only difference is runtime adjustment.
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
            "HELIUM_VLLM_PORT=18380",
            "HELIUM_VLLM_DEVICE=cuda:0",
            "HELIUM_VLLM_LOGGING_LEVEL=WARNING",
            "",
        ]
    )
)
os.environ["HELIUM_INTRA_GRAPH_PLACEMENT"] = "spread"

from bench_programs.debate.helium import HeliumDebateProgram
from bench_utils.datasets.tatqa.tatqa import TatQADataset
from helium.common import GenerationConfig
from helium.ops import LLMOp
from helium.runtime.llm import LLMServiceConfig, LLMServiceInfo
from helium.runtime.protocol import HeliumRequestConfig, SystemProfilingConfig
from helium_new.runtime.server import HeliumServer, HeliumServerConfig

MODEL = "/home/wanghaoyu/weizhihao/models/models/Qwen--Qwen2.5-3B-Instruct/snapshots/master"
RESULT = Path(__file__).resolve().parent / "results" / "debate_ra_on_off_3gpu.txt"
DEVICES = ["cuda:0", "cuda:1", "cuda:2"]


def _server_config(base_port: int) -> HeliumServerConfig:
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
            for index, device in enumerate(DEVICES)
        ],
    )


def _request_config(runtime_adjustment: bool) -> HeliumRequestConfig:
    return HeliumRequestConfig(
        enable_cache_aware_scheduling=True,
        enable_runtime_adjustment=runtime_adjustment,
        system_profiling_config=SystemProfilingConfig(),
        query_profiling_config=None,
        precompute_mode="none",
    )


async def _run(runtime_adjustment: bool, base_port: int) -> str:
    cfg = _server_config(base_port)
    request_config = _request_config(runtime_adjustment)
    generation = GenerationConfig(
        model=MODEL,
        base_url="http://127.0.0.1:9/v1",
        temperature=0,
        max_tokens=512,
    )
    dataset = TatQADataset(split="test", num_questions=6)
    context, questions, _answers = next(iter(dataset.iter_context_qa_pairs(1)))
    program = HeliumDebateProgram(server_config=cfg, request_config=request_config)
    prepared = program.prepare_kwargs(
        dataset="tatqa",
        contexts=[context],
        context_questions=([questions[0]],),
        context_choices=None,
        num_agents=3,
        num_rounds=2,
        different_roles=True,
        generation_config=generation,
    )
    agent = program.create_agent(**prepared)
    graph = agent.get_and_reset_compiled_graph()
    n_llm = sum(1 for _ in graph.iter_ops(LLMOp))
    print(
        f"=== RA={runtime_adjustment} devices={DEVICES} llm_ops={n_llm} ===",
        flush=True,
    )
    server = HeliumServer(cfg)
    server.start()
    started = time.perf_counter()
    try:
        await asyncio.wait_for(
            server.execute({agent.name: graph}, request_config),
            timeout=180,
        )
        elapsed = time.perf_counter() - started
        line = f"RA={runtime_adjustment} OK execute_seconds={elapsed:.2f} llm_ops={n_llm}"
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - started
        line = f"RA={runtime_adjustment} TIMEOUT execute_seconds={elapsed:.2f} llm_ops={n_llm}"
    finally:
        if server.is_started:
            server.close()
    print(line, flush=True)
    return line


async def main() -> None:
    lines = [
        "test only: helium_new debate, 3 GPUs, one whole graph, one execute",
        "same cards cuda:0 cuda:1 cuda:2; only runtime adjustment changes",
        "CAS=True max_tokens=512 spread on",
        "order: RA off, then RA on",
    ]
    for runtime_adjustment, port in ((False, 18380), (True, 18390)):
        lines.append(await _run(runtime_adjustment, port))
        RESULT.parent.mkdir(parents=True, exist_ok=True)
        RESULT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
