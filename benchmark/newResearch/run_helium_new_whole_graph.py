"""One Helium (helium_new), one debate graph, one task instance.

Compare 1 worker vs 3 workers with HELIUM_INTRA_GRAPH_PLACEMENT=spread.
Uses official DebateAgent build_ops graph and a single server.execute().
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
            "HELIUM_VLLM_PORT=18200",
            "HELIUM_VLLM_DEVICE=cuda:7",
            "HELIUM_VLLM_LOGGING_LEVEL=WARNING",
            "",
        ]
    )
)
os.environ.setdefault("HELIUM_DEPLOYMENT_MODE", "prod")

from bench_programs.debate.base import DebateProgram
from bench_programs.debate.helium import HeliumDebateProgram
from bench_utils.datasets.tatqa.tatqa import TatQADataset
from helium import ops
from helium.common import GenerationConfig
from helium.runtime.llm import LLMServiceConfig, LLMServiceInfo
from helium.runtime.protocol import HeliumRequestConfig

from helium_new.runtime.server import HeliumServer, HeliumServerConfig

MODEL = "/home/wanghaoyu/weizhihao/models/models/Qwen--Qwen2.5-3B-Instruct/snapshots/master"
RESULT = Path(__file__).resolve().parent / "results" / "helium_new_whole_graph.txt"


def _write_env(device: str, port: int) -> None:
    root = Path(__file__).resolve().parents[2]
    (root / ".env").write_text(
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
                f"HELIUM_VLLM_DEVICE={device}",
                f"HELIUM_VLLM_PORT={port}",
                "HELIUM_VLLM_LOGGING_LEVEL=WARNING",
                "",
            ]
        )
    )


def _server_config(workers: list[tuple[str, int]]) -> HeliumServerConfig:
    info = LLMServiceInfo(prefix_caching_enabled=True, cache_capacity=80000)
    return HeliumServerConfig(
        is_local=True,
        deployment_mode="prod",
        llm_service_configs=[
            LLMServiceConfig(
                name="vllm-local",
                args={
                    "model": MODEL,
                    "device": device,
                    "port": port,
                    "enable_prefix_caching": True,
                    "gpu_memory_utilization": 0.5,
                    "max_model_len": 8192,
                },
                info=info,
            )
            for device, port in workers
        ],
    )


def _load_one_question() -> tuple[str, str]:
    dataset = TatQADataset(split="test", num_questions=6)
    context, questions, _answers = next(iter(dataset.iter_context_qa_pairs(1)))
    return context, questions[0]


async def _run_debate(
    workers: list[tuple[str, int]], spread: bool, base_port: int
) -> float:
    os.environ["HELIUM_INTRA_GRAPH_PLACEMENT"] = "spread" if spread else "off"
    _write_env(workers[0][0], base_port)
    request_config = HeliumRequestConfig(
        enable_cache_aware_scheduling=False,
        enable_runtime_adjustment=False,
        precompute_mode="none",
        query_profiling_config=None,
    )
    generation = GenerationConfig(
        model=MODEL,
        base_url="http://127.0.0.1:9/v1",
        temperature=0,
        max_tokens=64,
    )
    context, question = _load_one_question()
    formats = DebateProgram.PROMPT_FORMATS["tatqa"]
    program = HeliumDebateProgram(
        server_config=_server_config(workers),
        request_config=request_config,
    )
    agent = program.create_agent(
        system_prompt=formats.system_prompt,
        role_prompts=formats.roles[:3],
        context_prompts=[context],
        context_question_prompts=([question],),
        revise_prompts=formats.revise_prompts,
        num_agents=3,
        num_rounds=2,
        generation_config=generation,
    )
    agent.compile(context=[context], question_0=[question])
    graph = agent.get_and_reset_compiled_graph()

    server = HeliumServer(_server_config(workers))
    server.start()
    try:
        started = time.perf_counter()
        await server.execute({agent.name: graph}, request_config)
        return time.perf_counter() - started
    finally:
        server.close()


async def main() -> None:
    lines = [
        "helium_new: one Helium, one DebateAgent compiled graph, one TAT-QA question",
        "3 agents, 2 rounds; batch size 1",
    ]
    print("Loading 1-worker baseline (spread off)...", flush=True)
    t1 = await _run_debate([("cuda:7", 18200)], spread=False, base_port=18200)
    lines.append(f"1worker seconds={t1:.2f}")
    print(lines[-1], flush=True)

    print("Loading 3-worker spread...", flush=True)
    t3 = await _run_debate(
        [("cuda:7", 18210), ("cuda:6", 18211), ("cuda:1", 18212)],
        spread=True,
        base_port=18210,
    )
    lines.append(f"3worker spread seconds={t3:.2f}")
    lines.append(f"speedup={t1 / t3:.2f}")
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
