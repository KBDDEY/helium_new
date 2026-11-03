"""In-memory dataset: first task of iterative / debate / parallel (newResearch-aligned).

Each TaskSpec carries:
  - label  → for the strategy (scheduling only)
  - graph + agent_name → compiled helium2 graph, runnable by helium2 or helium_new2
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Literal

from . import config as cfg


def _install_helium2_as_helium() -> None:
    """Bench programs import `helium`; point that name at helium2 for this process."""
    # Drop a previously imported stock `helium` so submodules resolve to helium2.
    for key in list(sys.modules):
        if key == "helium" or key.startswith("helium."):
            del sys.modules[key]

    import helium2

    sys.modules["helium"] = helium2
    for name, mod in list(sys.modules.items()):
        if name.startswith("helium2."):
            sys.modules["helium" + name[len("helium2") :]] = mod
        elif name == "helium2":
            continue


@dataclass
class TaskSpec:
    task_id: str
    label: Literal["iterative", "debate", "parallel"]
    agent_name: str
    graph: Any  # helium2.graphs.CompiledGraph
    generation_config: Any


def _generation_config():
    from helium2.common import GenerationConfig

    return GenerationConfig(
        model=cfg.MODEL,
        base_url="http://127.0.0.1:9/v1",  # overwritten by vllm-openai worker
        temperature=cfg.TEMPERATURE,
        max_tokens=cfg.MAX_TOKENS,
    )


def _dummy_server_config():
    """Avoid HeliumProgram default path load; runner supplies real replicas later."""
    from helium2.runtime.llm import LLMServiceConfig
    from helium2.runtime.server import HeliumServerConfig

    return HeliumServerConfig(
        is_local=True,
        llm_service_configs=[
            LLMServiceConfig(
                name="vllm-openai",
                args={
                    "model": cfg.MODEL,
                    "host": "127.0.0.1",
                    "port": 9,
                    "api_key": "EMPTY",
                },
            )
        ],
    )


def _build_iterative(generation_config) -> TaskSpec:
    from bench_programs.iterative.helium import HeliumIterativeProgram
    from bench_utils.datasets.amazon import AmazonReviewsDataset
    from helium2.utils import iter_batch

    dataset = AmazonReviewsDataset(split="test")
    _item, reviews = list(dataset.iter_item_reviews(3, 60))[0]
    document_chunks = [tuple(iter_batch(reviews, 10))]

    program = HeliumIterativeProgram(server_config=_dummy_server_config())
    prepared = program.prepare_kwargs(
        dataset="amazon",
        document_chunks=document_chunks,
        generation_config=generation_config,
    )
    agent = program.create_agent(**prepared)
    graph = agent.get_and_reset_compiled_graph()
    return TaskSpec(
        task_id="iterative-0",
        label="iterative",
        agent_name=agent.name,
        graph=graph,
        generation_config=generation_config,
    )


def _build_debate(generation_config) -> TaskSpec:
    from bench_programs.debate.helium import HeliumDebateProgram
    from bench_utils.datasets.tatqa.tatqa import TatQADataset

    dataset = TatQADataset(split="test", num_questions=6)
    context, questions, _answers = list(dataset.iter_context_qa_pairs(3))[0]

    program = HeliumDebateProgram(server_config=_dummy_server_config())
    prepared = program.prepare_kwargs(
        dataset="tatqa",
        contexts=[context],
        context_questions=tuple([question] for question in questions),
        context_choices=None,
        num_agents=3,
        num_rounds=2,
        different_roles=True,
        generation_config=generation_config,
    )
    agent = program.create_agent(**prepared)
    graph = agent.get_and_reset_compiled_graph()
    return TaskSpec(
        task_id="debate-0",
        label="debate",
        agent_name=agent.name,
        graph=graph,
        generation_config=generation_config,
    )


def _build_parallel(generation_config) -> TaskSpec:
    from bench_programs.parallel.helium import HeliumParallelProgram
    from bench_utils.datasets.amazon import AmazonReviewsDataset
    from helium2.utils import iter_batch

    dataset = AmazonReviewsDataset(split="test")
    item, reviews = list(dataset.iter_item_reviews(5, 60))[0]
    chunks = list(iter_batch(reviews, 10))
    item_reviews = [(item, chunks)]

    program = HeliumParallelProgram(server_config=_dummy_server_config())
    prepared = program.prepare_kwargs(
        dataset="amazon",
        item_reviews=item_reviews,
        num_experts=7,
        num_review_chunks_per_item=6,
        generation_config=generation_config,
    )
    agent = program.create_agent(**prepared)
    graph = agent.get_and_reset_compiled_graph()
    return TaskSpec(
        task_id="parallel-0",
        label="parallel",
        agent_name=agent.name,
        graph=graph,
        generation_config=generation_config,
    )


def build_fixed_dataset() -> list[TaskSpec]:
    """Three tasks: first sample of each newResearch scenario (in memory)."""
    _install_helium2_as_helium()
    generation = _generation_config()
    return [
        _build_iterative(generation),
        _build_debate(generation),
        _build_parallel(generation),
    ]
