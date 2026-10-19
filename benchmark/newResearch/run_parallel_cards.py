"""Parallel, one Amazon item, 7 experts, 6 chunks. 1 GPU vs 3 GPUs.

Experts do not share a prefix, so prefix placement keeps one expert's chunks
together and spreads experts across cards. Summaries wait for extractions.
"""

import os
import time
from pathlib import Path

os.environ["HELIUM_SPREAD_PLACEMENT"] = "0"
os.environ["HELIUM_PREFIX_PLACEMENT"] = "1"
os.environ["HELIUM_DEPLOYMENT_MODE"] = "prod"

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
_ENV_FILE.write_text(
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
            "HELIUM_VLLM_PORT=18160",
            "HELIUM_VLLM_DEVICE=cuda:7",
            "HELIUM_VLLM_LOGGING_LEVEL=WARNING",
            "",
        ]
    )
)

from bench_programs.parallel.helium import HeliumParallelProgram
from bench_utils.datasets.amazon import AmazonReviewsDataset
from helium.utils import iter_batch

from helium import graphs, helium, ops
from helium.common import GenerationConfig
from helium.runtime import HeliumServer, HeliumServerConfig
from helium.runtime.llm import LLMServiceConfig, LLMServiceInfo
from helium.runtime.protocol import HeliumRequestConfig

MODEL = "/home/wanghaoyu/weizhihao/models/models/Qwen--Qwen2.5-3B-Instruct/snapshots/master"
RESULT_PATH = Path(__file__).resolve().parent / "results" / "parallel_cards.txt"


def service_config(device: str, port: int) -> LLMServiceConfig:
    return LLMServiceConfig(
        name="vllm-local",
        args={
            "model": MODEL,
            "device": device,
            "port": port,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.5,
            "max_model_len": 8192,
        },
        info=LLMServiceInfo(prefix_caching_enabled=True, cache_capacity=80000),
    )


def one_item():
    dataset = AmazonReviewsDataset(split="test")
    item, reviews = next(iter(dataset.iter_item_reviews(1, 60)))
    chunks = list(iter_batch(reviews, 10))
    program = HeliumParallelProgram.__new__(HeliumParallelProgram)
    prepared = program.prepare_kwargs(
        dataset="amazon",
        item_reviews=[(item, chunks)],
        num_experts=7,
        num_review_chunks_per_item=6,
        generation_config=None,
    )
    return prepared


def chat(system: str, user: ops.Op, config: GenerationConfig) -> ops.Op:
    messages = ops.message_data(
        [
            ops.OpMessage(role="system", content=system),
            ops.OpMessage(role="user", content=user),
        ]
    )
    return ops.llm_chat(messages, config, return_history=False)


def invoke(server, graph, request_config):
    started = time.perf_counter()
    response = helium.invoke(graph, request_config)
    elapsed = time.perf_counter() - started
    outputs = response if isinstance(response, dict) else response.outputs
    placement = getattr(server.processor, "_last_prefix_placement", [])
    return elapsed, outputs, placement


def run_once(devices, prepared) -> tuple[float, dict]:
    os.environ["HELIUM_PREFIX_PLACEMENT"] = "1"
    os.environ["HELIUM_SPREAD_PLACEMENT"] = "0"
    configs = [
        service_config(device, 18160 + index) for index, device in enumerate(devices)
    ]
    server = HeliumServer(
        HeliumServerConfig(
            is_local=True,
            llm_service_configs=configs,
            deployment_mode="prod",
        )
    )
    request_config = HeliumRequestConfig(
        enable_cache_aware_scheduling=False,
        enable_runtime_adjustment=True,
        precompute_mode="none",
        query_profiling_config=None,
    )
    generation = GenerationConfig(
        model=MODEL,
        base_url="http://127.0.0.1:9/v1",
        temperature=0,
        max_tokens=64,
    )
    item = prepared["item_descriptions"][0]
    chunks = [chunk[0] for chunk in prepared["review_chunks"]]
    experts = prepared["expert_system_prompts"]
    server.start()
    try:
        extract_ops = []
        for expert_index, system in enumerate(experts):
            for chunk_index, chunk in enumerate(chunks):
                user = ops.data(
                    prepared["extraction_instruction_fmt"]
                    .replace("Product description:\n  {}", f"Product description:\n  {item}", 1)
                    .replace("Customer reviews:\n  {}", f"Customer reviews:\n  {chunk}", 1)
                )
                extract_ops.append(
                    ops.as_output(f"e{expert_index}_c{chunk_index}", chat(system, user, generation))
                )
        t_extract, extracted, place_extract = invoke(
            server, graphs.from_ops(extract_ops).compile(), request_config
        )
        summary_ops = []
        for expert_index, system in enumerate(experts):
            insights = "\n".join(
                f"Chunk {chunk_index + 1}:\n{extracted[f'e{expert_index}_c{chunk_index}']}"
                for chunk_index in range(len(chunks))
            )
            text = (
                "Summarize the extracted insights for this product.\n"
                f"Product description:\n{item}\n\n{insights}"
            )
            summary_ops.append(
                ops.as_output(f"s{expert_index}", chat(system, ops.data(text), generation))
            )
        t_summary, summaries, place_summary = invoke(
            server, graphs.from_ops(summary_ops).compile(), request_config
        )
        report_text = "Write the product report from these expert summaries.\n" + "\n".join(
            f"Insights from expert {index}:\n{summaries[f's{index}']}"
            for index in range(len(experts))
        )
        t_report, _report, place_report = invoke(
            server,
            graphs.from_ops(
                [
                    ops.as_output(
                        "report",
                        chat(prepared["writer_system_prompt"], ops.data(report_text), generation),
                    )
                ]
            ).compile(),
            request_config,
        )
        counts = {}
        for _op, worker, _service in place_extract + place_summary + place_report:
            counts[worker] = counts.get(worker, 0) + 1
        return t_extract + t_summary + t_report, {
            "extract": t_extract,
            "summary": t_summary,
            "report": t_report,
            "workers": counts,
        }
    finally:
        if server.is_started:
            server.close()


def main() -> None:
    prepared = one_item()
    chunks = [chunk[0] for chunk in prepared["review_chunks"]]
    print(
        f"experts={len(prepared['expert_system_prompts'])} chunks={len(chunks)} "
        f"chunk0_chars={len(chunks[0])}",
        flush=True,
    )
    lines = [
        "parallel, one amazon item, 7 experts, 6 chunks of 10 reviews",
        f"chunk0_chars={len(chunks[0])}",
    ]
    totals = {}
    for label, devices in (
        ("1gpu", ["cuda:7"]),
        ("3gpu", ["cuda:7", "cuda:6", "cuda:1"]),
    ):
        print(f"=== {label} {devices} ===", flush=True)
        total, detail = run_once(devices, prepared)
        totals[label] = total
        lines.append(
            f"{label} devices={devices} extract={detail['extract']:.2f} "
            f"summary={detail['summary']:.2f} report={detail['report']:.2f} "
            f"total={total:.2f} per_worker={detail['workers']}"
        )
        print(lines[-1], flush=True)
    speedup = totals["1gpu"] / totals["3gpu"]
    lines.append(f"speedup={speedup:.2f}")
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
