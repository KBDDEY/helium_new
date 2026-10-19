"""Rerun the 7.25s sequence and sample the three GPUs during the 3-card request.

Order: original Helium on 1 card, then helium_new on 3 cards, same process.
Whole debate graph, one execute each. CAS on, RA on.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import threading
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
            "HELIUM_VLLM_PORT=18400",
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
from helium.runtime.llm import LLMServiceConfig, LLMServiceInfo
from helium.runtime.protocol import HeliumRequestConfig, SystemProfilingConfig
from helium.runtime.server import HeliumServer as OriginalHeliumServer
from helium.runtime.server import HeliumServerConfig as OriginalHeliumServerConfig
from helium_new.runtime.server import HeliumServer as NewHeliumServer
from helium_new.runtime.server import HeliumServerConfig as NewHeliumServerConfig

MODEL = "/home/wanghaoyu/weizhihao/models/models/Qwen--Qwen2.5-3B-Instruct/snapshots/master"
RESULT = Path(__file__).resolve().parent / "results" / "debate_rerun_1_then_3.txt"


def _info() -> LLMServiceInfo:
    return LLMServiceInfo(prefix_caching_enabled=True, cache_capacity=80000)


def _request_config() -> HeliumRequestConfig:
    return HeliumRequestConfig(
        enable_cache_aware_scheduling=True,
        enable_runtime_adjustment=True,
        system_profiling_config=SystemProfilingConfig(),
        query_profiling_config=None,
        precompute_mode="none",
    )


def _services(devices: list[str], base_port: int) -> list[LLMServiceConfig]:
    info = _info()
    return [
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
    ]


def _generation() -> GenerationConfig:
    return GenerationConfig(
        model=MODEL,
        base_url="http://127.0.0.1:9/v1",
        temperature=0,
        max_tokens=512,
    )


def _gpu_line() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,clocks.sm,memory.used",
            "--format=csv,noheader",
            "-i",
            "0,1,2",
        ],
        text=True,
    ).strip()


def _output_lengths(outputs: object) -> str:
    texts: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(outputs)
    joined = " ".join(texts)
    return f"string_count={len(texts)} total_chars={len(joined)}"


async def _control() -> str:
    cfg = OriginalHeliumServerConfig(
        is_local=True,
        benchmarking=True,
        deployment_mode="prod",
        llm_service_configs=_services(["cuda:0"], 18400),
    )
    request_config = _request_config()
    dataset = TatQADataset(split="test", num_questions=6)
    context, questions, _answers = next(iter(dataset.iter_context_qa_pairs(1)))
    program = HeliumDebateProgram(server_config=cfg, request_config=request_config)
    print("=== 1gpu original cuda:0 ===", flush=True)
    print("gpus before 1-card start:", _gpu_line().replace("\n", " | "), flush=True)
    started = time.perf_counter()
    try:
        outputs = await asyncio.wait_for(
            program.run_async(
                dataset="tatqa",
                contexts=[context],
                context_questions=([questions[0]],),
                context_choices=None,
                num_agents=3,
                num_rounds=2,
                different_roles=True,
                generation_config=_generation(),
            ),
            timeout=180,
        )
        elapsed = time.perf_counter() - started
        line = f"1gpu wall_seconds={elapsed:.2f} {_output_lengths(outputs)}"
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - started
        line = f"1gpu TIMEOUT wall_seconds={elapsed:.2f}"
    finally:
        server = OriginalHeliumServer.get_instance(cfg)
        if server.is_started:
            server.close()
    print(line, flush=True)
    print("gpus after 1-card close:", _gpu_line().replace("\n", " | "), flush=True)
    return line


async def _experiment() -> tuple[str, list[str]]:
    cfg = NewHeliumServerConfig(
        is_local=True,
        benchmarking=True,
        deployment_mode="prod",
        llm_service_configs=_services(["cuda:0", "cuda:1", "cuda:2"], 18410),
    )
    request_config = _request_config()
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
        generation_config=_generation(),
    )
    agent = program.create_agent(**prepared)
    graph = agent.get_and_reset_compiled_graph()
    print("=== 3gpu helium_new cuda:0,1,2 ===", flush=True)
    print("gpus before 3-card start:", _gpu_line().replace("\n", " | "), flush=True)
    server = NewHeliumServer(cfg)
    server.start()
    print("gpus after 3-card start:", _gpu_line().replace("\n", " | "), flush=True)
    samples: list[str] = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            samples.append(_gpu_line().replace("\n", " | "))
            stop.wait(0.5)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        response = await asyncio.wait_for(
            server.execute({agent.name: graph}, request_config),
            timeout=180,
        )
        elapsed = time.perf_counter() - started
        line = f"3gpu execute_seconds={elapsed:.2f} {_output_lengths(response.outputs)}"
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - started
        line = f"3gpu TIMEOUT execute_seconds={elapsed:.2f}"
    finally:
        stop.set()
        thread.join(timeout=2)
        if server.is_started:
            server.close()
    print(line, flush=True)
    for sample in samples:
        print("sample", sample, flush=True)
    return line, samples


async def main() -> None:
    lines = [
        "rerun of 1-card then 3-card debate in one process",
        "CAS=True RA=True max_tokens=512 one whole graph one execute",
        "cards cuda:0 then cuda:0,1,2 because cuda:6 and cuda:7 were occupied",
    ]
    lines.append(await _control())
    experiment_line, samples = await _experiment()
    lines.append(experiment_line)
    lines.extend(f"sample {sample}" for sample in samples)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
