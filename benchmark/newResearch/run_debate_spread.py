"""Debate, one question, 10 experts, 2 rounds. 1 GPU vs 2 GPUs.

Round 2 starts only after round 1 answers exist. Experts in a round are
spread across the available cards. No question batch.
"""

import os
import time
from pathlib import Path

os.environ["HELIUM_SPREAD_PLACEMENT"] = "1"
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
            "HELIUM_VLLM_PORT=18150",
            "HELIUM_VLLM_DEVICE=cuda:7",
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
RESULT_PATH = Path(__file__).resolve().parent / "results" / "debate_tatqa.txt"
ROLES = [
    "professor",
    "doctor",
    "mathematician",
    "lawyer",
    "historian",
    "engineer",
    "economist",
    "biologist",
    "physicist",
    "philosopher",
]
QUESTION = ""
REVISE = (
    "Using the reasoning from other agents as additional advice, can you give "
    "an updated answer? Examine your solution and that other agents step by "
    "step. Put your answer in the form (X) at the end of your response."
)


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


def chat(user: ops.Op, role: str, config: GenerationConfig) -> ops.Op:
    messages = ops.message_data(
        [
            ops.OpMessage(role="system", content="You are a helpful AI Assistant."),
            ops.OpMessage(
                role="user",
                content=ops.format_op(
                    "{user}\nRespond as if you are {role}.",
                    user=user,
                    role=ops.data(role),
                ),
            ),
        ]
    )
    return ops.llm_chat(messages, config, return_history=False)


def invoke(server, graph, request_config) -> tuple[float, dict, list]:
    started = time.perf_counter()
    response = helium.invoke(graph, request_config)
    elapsed = time.perf_counter() - started
    outputs = response if isinstance(response, dict) else response.outputs
    placement = getattr(server.processor, "_last_prefix_placement", [])
    return elapsed, outputs, placement


def run_once(devices: list[str]) -> tuple[float, float, list, str]:
    os.environ["HELIUM_SPREAD_PLACEMENT"] = "1"
    configs = [
        service_config(device, 18150 + index) for index, device in enumerate(devices)
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
    server.start()
    try:
        question = ops.data(QUESTION)
        round1 = graphs.from_ops(
            [
                ops.as_output(f"a{index}", chat(question, role, generation))
                for index, role in enumerate(ROLES)
            ]
        ).compile()
        t1, answers, place1 = invoke(server, round1, request_config)
        texts = [str(answers[f"a{index}"]) for index in range(len(ROLES))]
        round2_ops = []
        for index, role in enumerate(ROLES):
            others = "\n".join(
                f"One agent solution: ```{texts[other]}```"
                for other in range(len(ROLES))
                if other != index
            )
            user = ops.data(
                f"{QUESTION}\nRespond as if you are {role}.\n\n"
                f"These are the solutions to the problem from other agents:\n{others}\n\n"
                f"{REVISE}"
            )
            round2_ops.append(ops.as_output(f"b{index}", chat(user, role, generation)))
        round2 = graphs.from_ops(round2_ops).compile()
        t2, revised, place2 = invoke(server, round2, request_config)
        return t1, t2, place1 + place2, str(revised.get("b0", ""))[:160]
    finally:
        if server.is_started:
            server.close()


def load_first_tatqa() -> str:
    from bench_utils.datasets.tatqa import TatQADataset

    dataset = TatQADataset(split="test", num_questions=6)
    context, questions, _answers = next(iter(dataset.iter_context_qa_pairs(1)))
    return (
        "Please answer the given financial question based on the context.\n"
        f"Context:\n{context}\n\n"
        f"Question: {questions[0]}\n"
        "Explain your thoughts step by step and summarize your answer at the "
        'end of your response, preceding with "Answer: ". '
    )


def main() -> None:
    global QUESTION
    QUESTION = load_first_tatqa()
    print(f"prompt_chars={len(QUESTION)}", flush=True)
    lines = [
        "debate, TAT-QA first context, first of 6 questions, 10 experts, 2 rounds",
        f"prompt_chars={len(QUESTION)}",
        "round 2 waits for round 1; experts in a round are spread across cards",
    ]
    totals = {}
    for label, devices in (
        ("1gpu", ["cuda:7"]),
        ("2gpu", ["cuda:7", "cuda:6"]),
    ):
        print(f"=== {label} {devices} ===", flush=True)
        t1, t2, placement, sample = run_once(devices)
        totals[label] = t1 + t2
        counts = {}
        for _op_id, worker_idx, _service in placement:
            counts[worker_idx] = counts.get(worker_idx, 0) + 1
        lines.append(
            f"{label} devices={devices} round1={t1:.2f} round2={t2:.2f} total={t1 + t2:.2f} per_worker={counts}"
        )
        lines.append(f"  sample={sample}")
        print(lines[-2], flush=True)
    speedup = totals["1gpu"] / totals["2gpu"]
    lines.append(f"speedup={speedup:.2f}")
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
