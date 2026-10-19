"""Single-instance prefix placement: 1 GPU vs 2 GPUs.

One workflow, no question batch. Two document groups share a long prefix
inside the group and not across groups. A follow-up call in each group
depends on that group's answers and stays on the same GPU.
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
            "HELIUM_VLLM_PORT=18120",
            "HELIUM_VLLM_DEVICE=cuda:6",
            "HELIUM_VLLM_LOGGING_LEVEL=WARNING",
            "",
        ]
    )
)
os.environ["HELIUM_PREFIX_PLACEMENT"] = "1"

from helium import graphs, helium, ops
from helium.common import GenerationConfig
from helium.runtime import HeliumServer, HeliumServerConfig
from helium.runtime.llm import LLMServiceConfig, LLMServiceInfo
from helium.runtime.protocol import HeliumRequestConfig

MODEL = "/home/wanghaoyu/weizhihao/models/models/Qwen--Qwen2.5-3B-Instruct/snapshots/master"
RESULT_PATH = Path(__file__).resolve().parent / "results" / "prefix_placement.txt"


def document(title: str, sentence: str) -> str:
    body = " ".join(f"{index}. {sentence}" for index in range(1, 31))
    return f"{title}\n{body}"


DOC_A = document(
    "Report A. Harbor freight volumes.",
    "Harbor A moved grain, timber, and machine parts through the north dock.",
)
DOC_B = document(
    "Report B. Alpine grid repairs.",
    "Region B replaced transformers and buried cable after the winter storm.",
)


def expert(context: ops.Op, role: str, config: GenerationConfig) -> ops.Op:
    user = ops.format_op(
        "Document:\n{context}\n\nRole: {role}\nQuestion: State the main claim in one sentence.",
        context=context,
        role=ops.data(role),
    )
    messages = ops.message_data(
        [
            ops.OpMessage(role="system", content="You are a careful reader."),
            ops.OpMessage(role="user", content=user),
        ]
    )
    return ops.llm_chat(messages, config, return_history=False)


def build_graph(config: GenerationConfig):
    ctx_a = ops.InputOp("ctx_a")
    ctx_b = ops.InputOp("ctx_b")
    note_a1 = expert(ctx_a, "analyst", config)
    note_a2 = expert(ctx_a, "checker", config)
    note_b1 = expert(ctx_b, "analyst", config)
    note_b2 = expert(ctx_b, "checker", config)
    graph = graphs.from_ops(
        [
            ops.as_output("a1", note_a1),
            ops.as_output("a2", note_a2),
            ops.as_output("b1", note_b1),
            ops.as_output("b2", note_b2),
        ]
    ).compile(ctx_a=[DOC_A], ctx_b=[DOC_B])
    return graph


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


def run_once(devices: list[str], prefix_placement: bool) -> tuple[float, list, dict]:
    os.environ["HELIUM_PREFIX_PLACEMENT"] = "1" if prefix_placement else "0"
    configs = [
        service_config(device, 18120 + index) for index, device in enumerate(devices)
    ]
    server_config = HeliumServerConfig(
        is_local=True,
        llm_service_configs=configs,
        deployment_mode="prod",
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
        max_tokens=32,
    )
    server = HeliumServer(server_config)
    server.start()
    try:
        graph = build_graph(generation)
        started = time.perf_counter()
        response = helium.invoke(graph, request_config)
        elapsed = time.perf_counter() - started
        placement = getattr(server.processor, "_last_prefix_placement", [])
        outputs = response if isinstance(response, dict) else response.outputs
        texts = {name: str(value)[:180] for name, value in outputs.items()}
        return elapsed, placement, texts
    finally:
        if server.is_started:
            server.close()


def main() -> None:
    lines = [
        f"doc_a_chars={len(DOC_A)} doc_b_chars={len(DOC_B)}",
        "policy=shared long prefix stays together; distinct groups use different GPUs",
    ]
    for label, devices, prefix_placement in (
        ("1gpu", ["cuda:6"], False),
        ("2gpu", ["cuda:6", "cuda:1"], True),
    ):
        print(f"=== {label} {devices} ===", flush=True)
        elapsed, placement, texts = run_once(devices, prefix_placement)
        lines.append(f"{label} devices={devices} seconds={elapsed:.2f}")
        for preview, worker_idx, service_name in placement:
            lines.append(f"  worker {worker_idx} | {preview}")
        for name, text in texts.items():
            lines.append(f"  out {name}: {text}")
        print(lines[-1 - len(texts) - len(placement)], flush=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
