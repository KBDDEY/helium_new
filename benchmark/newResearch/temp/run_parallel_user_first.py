"""Temp timing. 02 Parallel, user content before the expert system prompt.

Same item, graph, and engine settings as the current 02 scripts:
second Amazon item, 7 experts, 6 chunks, prefix caching, max_tokens=512.
Qwen's template inserts a default system line when the first message is not
system. The user content is therefore the first message, with role system,
and the original system prompt follows it. Text inside each message is unchanged.

Usage: python run_parallel_user_first.py control|experiment
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else ""
if MODE not in {"control", "experiment"}:
    raise SystemExit("usage: run_parallel_user_first.py control|experiment")

_ROOT = Path(__file__).resolve().parents[3]
if MODE == "control":
    DEVICES = ["cuda:3"]
    PORT = 18410
    PLACEMENT = "off"
else:
    DEVICES = ["cuda:0", "cuda:1", "cuda:5"]
    PORT = 18420
    PLACEMENT = "spread"

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
            f"HELIUM_VLLM_PORT={PORT}",
            f"HELIUM_VLLM_DEVICE={DEVICES[0]}",
            "HELIUM_VLLM_LOGGING_LEVEL=WARNING",
            "",
        ]
    )
)
os.environ["HELIUM_INTRA_GRAPH_PLACEMENT"] = PLACEMENT

from bench_programs.parallel.helium import ParallelAgent
from bench_utils.datasets.amazon import AmazonReviewsDataset
from helium import ops
from helium.common import GenerationConfig, Message
from helium.ops import LLMOp
from helium.runtime.llm import LLMServiceConfig, LLMServiceInfo
from helium.runtime.llm.utils import apply_chat_template
from helium.runtime.protocol import HeliumRequestConfig, SystemProfilingConfig
from helium.utils import iter_batch

if MODE == "control":
    from bench_programs.parallel.helium import HeliumParallelProgram
    from helium.runtime.server import HeliumServer, HeliumServerConfig
else:
    from bench_programs.parallel.helium import HeliumParallelProgram
    from helium_new.runtime.server import HeliumServer, HeliumServerConfig

MODEL = "/home/wanghaoyu/weizhihao/models/models/Qwen--Qwen2.5-3B-Instruct/snapshots/master"
RESULT = (
    Path(__file__).resolve().parent
    / "results"
    / f"parallel_user_first_{MODE}.txt"
)


def _user_then_system(user_content: ops.Op, system_prompt: str) -> list[ops.OpMessage]:
    return [
        ops.OpMessage(role="system", content=user_content),
        ops.OpMessage(role="system", content=system_prompt),
    ]


def _build_ops_user_first(
    self,
    expert_system_prompts: list[str],
    writer_system_prompt: str,
    extraction_instruction_fmt: str,
    summary_instruction_fmt: str,
    report_instruction_fmt: str,
    item_op: ops.Op,
    review_chunk_ops: Sequence[ops.Op],
    generation_config: GenerationConfig | None,
) -> list[ops.OutputOp]:
    insight_summaries: list[ops.Op] = []
    for system_prompt in expert_system_prompts:
        extracted_insights: list[ops.Op] = []
        for review_chunk in review_chunk_ops:
            extracted_insights.append(
                ops.llm_chat(
                    _user_then_system(
                        ops.format_op(
                            extraction_instruction_fmt, item_op, review_chunk
                        ),
                        system_prompt,
                    ),
                    generation_config,
                    return_history=False,
                )
            )
        summary_op = ops.llm_chat(
            _user_then_system(
                ops.format_op(
                    summary_instruction_fmt, item_op, *extracted_insights
                ),
                system_prompt,
            ),
            generation_config,
            return_history=False,
        )
        insight_summaries.append(summary_op)
    insight_report = ops.llm_chat(
        _user_then_system(
            ops.format_op(report_instruction_fmt, item_op, *insight_summaries),
            writer_system_prompt,
        ),
        generation_config,
        return_history=False,
    )
    return [ops.as_output("insight_report", insight_report)]


ParallelAgent.build_ops = _build_ops_user_first


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
                    "port": PORT + index,
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
    item, reviews = list(dataset.iter_item_reviews(2, 60))[1]
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
    rendered = apply_chat_template(
        MODEL,
        [[
            Message(role="system", content="USER_BODY"),
            Message(role="system", content="EXPERT_SYSTEM"),
        ]],
    )[0]
    print("RENDERED " + rendered.replace("\n", "\\n"), flush=True)
    agent = program.create_agent(**prepared)
    graph = agent.get_and_reset_compiled_graph()
    runtime = "original Helium" if MODE == "control" else "helium_new"
    print(
        f"{MODE}: {runtime}, devices={DEVICES}, 2nd Amazon item "
        f"{prepared['item_descriptions'][0].splitlines()[1]}, "
        "7 experts, 6 chunks, message_order=user_body,expert_system",
        flush=True,
    )
    print(
        "entry=prepare_kwargs + create_agent + one execute; "
        "timer=execute only, after server.start(); "
        f"HELIUM_INTRA_GRAPH_PLACEMENT={PLACEMENT}; "
        "CAS=True RA=True max_tokens=512",
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
