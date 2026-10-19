"""One workflow, no question batch: 1 GPU vs 2 GPUs.

Ten independent calls. Each call has its own long document, and the documents
do not share a prefix. Two GPUs therefore receive five calls each.
"""

import os
import time
from pathlib import Path

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
            "HELIUM_VLLM_PORT=18140",
            "HELIUM_VLLM_DEVICE=cuda:6",
            "HELIUM_VLLM_LOGGING_LEVEL=WARNING",
            "",
        ]
    )
)

from transformers import AutoTokenizer

from helium import graphs, helium, ops
from helium.common import GenerationConfig
from helium.runtime import HeliumServer, HeliumServerConfig
from helium.runtime.llm import LLMServiceConfig, LLMServiceInfo
from helium.runtime.protocol import HeliumRequestConfig

MODEL = "/home/wanghaoyu/weizhihao/models/models/Qwen--Qwen2.5-3B-Instruct/snapshots/master"
RESULT_PATH = Path(__file__).resolve().parent / "results" / "split_speedup_3gpu.txt"
NUM_EXPERTS = 9
TARGET_TOKENS = 3000


def make_document(index: int, tokenizer) -> str:
    topic = (
        f"Archive {index}. The {index} river basin recorded a separate flood, "
        f"a separate harvest, and a separate repair ledger. "
    )
    chunk = topic + (
        f"Entry {index} lists cargo, rainfall, wages, and bridge delays "
        f"that do not appear in any other archive. "
    )
    text = chunk
    while len(tokenizer.encode(text)) < TARGET_TOKENS:
        text += chunk
    ids = tokenizer.encode(text)[:TARGET_TOKENS]
    return tokenizer.decode(ids)


def expert(context: ops.Op, index: int, config: GenerationConfig) -> ops.Op:
    user = ops.format_op(
        "Document:\n{context}\n\nQuestion: In one sentence, what is archive {index} about?",
        context=context,
        index=ops.data(str(index)),
    )
    messages = ops.message_data(
        [
            ops.OpMessage(role="system", content=f"You read archive {index} only."),
            ops.OpMessage(role="user", content=user),
        ]
    )
    return ops.llm_chat(messages, config, return_history=False)


def build_graph(config: GenerationConfig, documents: list[str]):
    inputs = []
    outputs = []
    feed = {}
    for index, document in enumerate(documents):
        ctx = ops.InputOp(f"ctx_{index}")
        inputs.append(ctx)
        outputs.append(ops.as_output(f"e{index}", expert(ctx, index, config)))
        feed[f"ctx_{index}"] = [document]
    return graphs.from_ops(outputs).compile(**feed)


def service_config(device: str, port: int) -> LLMServiceConfig:
    return LLMServiceConfig(
        name="vllm-local",
        args={
            "model": MODEL,
            "device": device,
            "port": port,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.5,
            "max_model_len": 4096,
        },
        info=LLMServiceInfo(prefix_caching_enabled=True, cache_capacity=80000),
    )


def run_once(devices: list[str], documents: list[str]) -> tuple[float, list, dict]:
    os.environ["HELIUM_PREFIX_PLACEMENT"] = "1"
    configs = [
        service_config(device, 18140 + index) for index, device in enumerate(devices)
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
        max_tokens=16,
    )
    server.start()
    try:
        graph = build_graph(generation, documents)
        started = time.perf_counter()
        response = helium.invoke(graph, request_config)
        elapsed = time.perf_counter() - started
        placement = getattr(server.processor, "_last_prefix_placement", [])
        outputs = response if isinstance(response, dict) else response.outputs
        texts = {name: str(value)[:120] for name, value in outputs.items()}
        return elapsed, placement, texts
    finally:
        if server.is_started:
            server.close()


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    documents = [make_document(index, tokenizer) for index in range(NUM_EXPERTS)]
    token_lens = [len(tokenizer.encode(document)) for document in documents]
    lines = [
        "one workflow, ten independent calls, no shared prefix, no question batch",
        f"doc_tokens={token_lens}",
        "1gpu=all nine on cuda:7; 3gpu=spread across cuda:7, cuda:6, cuda:1",
    ]
    results = {}
    for label, devices in (
        ("1gpu", ["cuda:7"]),
        ("3gpu", ["cuda:7", "cuda:6", "cuda:1"]),
    ):
        print(f"=== {label} {devices} ===", flush=True)
        elapsed, placement, texts = run_once(devices, documents)
        results[label] = elapsed
        workers = sorted({worker for _, worker, _ in placement})
        lines.append(
            f"{label} devices={devices} seconds={elapsed:.2f} workers={workers} calls={len(placement)}"
        )
        for preview, worker_idx, _service_name in placement:
            lines.append(f"  worker {worker_idx} | {preview[:80]}")
        lines.append(f"  sample={texts.get('e0', '')}")
        print(lines[-1], flush=True)
    speedup = results["1gpu"] / results["3gpu"]
    lines.append(
        f"speedup={speedup:.2f} reached_3x={speedup >= 2.7} three_card_time_x3={3 * results['3gpu']:.2f} one_card={results['1gpu']:.2f}"
    )
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
