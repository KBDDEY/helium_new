"""Original Helium. One debate question, 3 experts, 2 rounds.

One card runs experts one at a time. Three cards run one expert each at the
same time. Model load is not timed.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL = "/home/wanghaoyu/weizhihao/models/models/Qwen--Qwen2.5-3B-Instruct/snapshots/master"
RESULT_PATH = Path(__file__).resolve().parent / "results" / "serial_vs_cards.txt"
ROLES = ["a financial analyst", "a data scientist", "a statistician"]
PYTHON = str(Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python")
ROOT = Path(__file__).resolve().parents[2]


def load_prompt() -> str:
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


def serve(device: str, port: int) -> None:
    os.environ["HELIUM_DEPLOYMENT_MODE"] = "prod"
    os.environ.pop("HELIUM_PREFIX_PLACEMENT", None)
    os.environ.pop("HELIUM_SPREAD_PLACEMENT", None)
    env_file = ROOT / ".env"
    env_file.write_text(
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
                f"HELIUM_VLLM_PORT={port}",
                f"HELIUM_VLLM_DEVICE={device}",
                "HELIUM_VLLM_LOGGING_LEVEL=ERROR",
                "",
            ]
        )
    )
    from helium import graphs, helium, ops
    from helium.common import GenerationConfig
    from helium.runtime import HeliumServer, HeliumServerConfig
    from helium.runtime.llm import LLMServiceConfig, LLMServiceInfo
    from helium.runtime.protocol import HeliumRequestConfig

    server = HeliumServer(
        HeliumServerConfig(
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
                    info=LLMServiceInfo(
                        prefix_caching_enabled=True, cache_capacity=80000
                    ),
                )
            ],
        )
    )
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
    server.start()
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    try:
        for line in sys.stdin:
            message = json.loads(line)
            if message is None:
                break
            role = message["role"]
            prompt = message["prompt"]
            user = ops.data(f"{prompt}\nRespond as if you are {role}.")
            messages = ops.message_data(
                [
                    ops.OpMessage(
                        role="system", content="You are a helpful AI Assistant."
                    ),
                    ops.OpMessage(role="user", content=user),
                ]
            )
            graph = graphs.from_ops(
                [
                    ops.as_output(
                        "answer",
                        ops.llm_chat(messages, generation, return_history=False),
                    )
                ]
            ).compile()
            response = helium.invoke(graph, request_config)
            outputs = response if isinstance(response, dict) else response.outputs
            sys.stdout.write(json.dumps({"text": str(outputs["answer"])}) + "\n")
            sys.stdout.flush()
    finally:
        if server.is_started:
            server.close()


def open_worker(device: str, port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [PYTHON, __file__, "serve", device, str(port)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src:benchmark", "HELIUM_DEPLOYMENT_MODE": "prod"},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline()
    if not line.startswith("READY"):
        raise RuntimeError(f"{device} failed to start: {line}")
    return proc


def ask(proc: subprocess.Popen, role: str, prompt: str) -> str:
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(json.dumps({"role": role, "prompt": prompt}) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    return json.loads(line)["text"]


def close_worker(proc: subprocess.Popen) -> None:
    if proc.stdin is not None:
        proc.stdin.write("null\n")
        proc.stdin.flush()
    proc.wait(timeout=60)


def revise(prompt: str, role: str, texts: list[str]) -> str:
    others = "\n".join(
        f"One agent solution: ```{texts[index]}```"
        for index, other in enumerate(ROLES)
        if other != role
    )
    return (
        f"{prompt}\n\nThese are the solutions from other agents:\n{others}\n\n"
        "Using their reasoning as advice, give an updated answer."
    )


def main() -> None:
    prompt = load_prompt()
    lines = [
        "original helium, debate, one TAT-QA question, 3 experts, 2 rounds",
        "one card runs one expert at a time; three cards run one expert each",
        f"prompt_chars={len(prompt)}",
    ]
    print(lines[-1], flush=True)

    serial_proc = open_worker("cuda:7", 18170)
    serial = 0.0
    texts: list[str] = []
    try:
        for _round in range(2):
            snapshot = list(texts)
            texts = []
            for role in ROLES:
                user_prompt = prompt if not snapshot else revise(prompt, role, snapshot)
                started = time.perf_counter()
                texts.append(ask(serial_proc, role, user_prompt))
                serial += time.perf_counter() - started
    finally:
        close_worker(serial_proc)
    lines.append(f"1gpu serial seconds={serial:.2f}")
    print(lines[-1], flush=True)

    devices = ["cuda:7", "cuda:6", "cuda:1"]
    procs = [open_worker(device, 18180 + index) for index, device in enumerate(devices)]
    parallel = 0.0
    texts = []
    try:
        for _round in range(2):
            snapshot = list(texts)
            started = time.perf_counter()
            # One expert per card. Reads are sequential only to collect results;
            # the three servers were given their calls before any read.
            pending = []
            for index, role in enumerate(ROLES):
                user_prompt = prompt if not snapshot else revise(prompt, role, snapshot)
                assert procs[index].stdin is not None
                procs[index].stdin.write(
                    json.dumps({"role": role, "prompt": user_prompt}) + "\n"
                )
                procs[index].stdin.flush()
                pending.append(procs[index])
            texts = []
            for proc in pending:
                assert proc.stdout is not None
                texts.append(json.loads(proc.stdout.readline())["text"])
            parallel += time.perf_counter() - started
    finally:
        for proc in procs:
            close_worker(proc)
    speedup = serial / parallel
    lines.append(f"3gpu seconds={parallel:.2f}")
    lines.append(f"speedup={speedup:.2f}")
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        serve(sys.argv[2], int(sys.argv[3]))
    else:
        main()
