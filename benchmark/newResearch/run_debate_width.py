"""One Debate question, round 1 only: 1 GPU vs prefix placement on 2 GPUs.

No question batch. Three agents read the same long question; the role sits
after it. Round 2 is omitted because, with cache-aware scheduling off, Helium
puts later-round calls in the same dispatch group and busy-waits before the
first round is sent. Timing excludes model load.
"""

import os
import statistics
import time
from pathlib import Path

os.environ["HELIUM_PREFIX_PLACEMENT"] = "0"
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
            "HELIUM_VLLM_PORT=18130",
            "HELIUM_VLLM_DEVICE=cuda:6",
            "HELIUM_VLLM_LOGGING_LEVEL=WARNING",
            "",
        ]
    )
)

from helium import graphs, helium, ops
from helium.common import GenerationConfig
from helium.runtime import HeliumServer, HeliumServerConfig
from helium.runtime.llm import LLMServiceConfig, LLMServiceInfo
from helium.runtime.protocol import HeliumRequestConfig

MODEL = "/home/wanghaoyu/weizhihao/models/models/Qwen--Qwen2.5-3B-Instruct/snapshots/master"
RESULT_PATH = Path(__file__).resolve().parent / "results" / "debate_width.txt"

ROLES = ("a professor", "a doctor", "a mathematician")


def passage(topic: str, fact: str) -> str:
    sentences = [f"{index}. {fact}" for index in range(1, 18)]
    body = " ".join(sentences)
    question = (
        f"Passage about {topic}.\n{body}\n"
        "Question: Which claim is supported by the passage?\n"
        "A) The passage describes a single isolated event with no repetition.\n"
        "B) The same claim is repeated across the numbered sentences.\n"
        "C) The passage denies the claim in every sentence.\n"
        "D) The passage is about an unrelated city.\n"
        "Explain your answer, putting the answer in the form (X) at the end."
    )
    return question


QUESTIONS = [
    passage(
        "harbor freight",
        "Harbor A moved grain, timber, and machine parts through the north dock.",
    ),
    passage(
        "alpine grid",
        "Region B replaced transformers and buried cable after the winter storm.",
    ),
    passage(
        "clinic inventory",
        "Clinic C restocked vaccines, syringes, and saline throughout the quarter.",
    ),
    passage(
        "orchard yield",
        "Orchard D picked apples, pears, and plums from the south plots.",
    ),
    passage(
        "library loans",
        "Library E circulated novels, atlases, and repair manuals to local schools.",
    ),
]


def build_graph(question: str, config: GenerationConfig):
    question_op = ops.data(question)
    answers = []
    for role in ROLES:
        user = ops.format_op(
            "{question}\nRespond as if you are {role}.",
            question=question_op,
            role=ops.data(role),
        )
        messages = ops.message_data(
            [
                ops.OpMessage(role="system", content="You are a helpful AI Assistant."),
                ops.OpMessage(role="user", content=user),
            ]
        )
        answers.append(ops.llm_chat(messages, config, return_history=False))
    return graphs.from_ops(
        [ops.as_output(f"agent_{index}", answer) for index, answer in enumerate(answers)]
    ).compile()


def service_config(device: str, port: int) -> LLMServiceConfig:
    return LLMServiceConfig(
        name="vllm-local",
        args={
            "model": MODEL,
            "device": device,
            "port": port,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.6,
            "max_model_len": 4096,
        },
        info=LLMServiceInfo(prefix_caching_enabled=True, cache_capacity=80000),
    )


def run_condition(devices: list[str], prefix_placement: bool) -> list[tuple[float, list, str]]:
    os.environ["HELIUM_PREFIX_PLACEMENT"] = "1" if prefix_placement else "0"
    configs = [service_config(device, 18130 + index) for index, device in enumerate(devices)]
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
    server.start()
    trials: list[tuple[float, list, str]] = []
    try:
        for question in QUESTIONS:
            graph = build_graph(question, generation)
            started = time.perf_counter()
            response = helium.invoke(graph, request_config)
            elapsed = time.perf_counter() - started
            placement = list(getattr(server.processor, "_last_prefix_placement", []))
            outputs = response if isinstance(response, dict) else response.outputs
            preview = str(outputs.get("agent_0", ""))[:160].replace("\n", " ")
            trials.append((elapsed, placement, preview))
            print(
                f"  seconds={elapsed:.2f} workers={sorted({item[1] for item in placement})} answer={preview}",
                flush=True,
            )
        return trials
    finally:
        if server.is_started:
            server.close()


def main() -> None:
    lines = [
        f"question_chars={[len(question) for question in QUESTIONS]}",
        "round=1 only; three agents; shared question stays on one card",
        "policy=compare 2*two_gpu median with one_gpu median",
        "cards=cuda:6 and cuda:1; model load excluded; five different questions",
    ]
    results: dict[str, list[float]] = {}
    for label, devices, prefix_placement in (
        ("1gpu", ["cuda:6"], False),
        ("2gpu", ["cuda:6", "cuda:1"], True),
    ):
        print(f"=== {label} {devices} ===", flush=True)
        trials = run_condition(devices, prefix_placement)
        times = [elapsed for elapsed, _, _ in trials]
        results[label] = times
        lines.append(f"{label} devices={devices} seconds={[round(item, 2) for item in times]} median={statistics.median(times):.2f}")
        _, placement, _ = trials[0]
        for preview, worker_idx, _service_name in placement:
            lines.append(f"  worker {worker_idx} | {preview[:140]}")
        lines.append(f"  sample={trials[0][2]}")
    one = statistics.median(results["1gpu"])
    two = statistics.median(results["2gpu"])
    lines.append(f"one_gpu_median={one:.2f} two_gpu_median={two:.2f} two_times_2={2 * two:.2f} worth_two_cards={2 * two < one}")
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
