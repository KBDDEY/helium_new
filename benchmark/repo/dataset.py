"""Build the configured task queue from indexed samples (newResearch-aligned).

Each TaskSpec carries:
  - label  → for the strategy (scheduling only)
  - graph + agent_name → compiled helium2 graph, runnable by helium2 or helium_new2
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from . import config as cfg

TaskLabel = Literal["iterative", "debate", "parallel"]


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
    label: TaskLabel
    agent_name: str
    graph: Any  # helium2.graphs.CompiledGraph
    generation_config: Any
    sample_index: int | None = None


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


def _load_samples(label: TaskLabel, count: int) -> list[Any]:
    """Read a prefix once per type; keep each sample's internal workload fixed."""
    if label in ("iterative", "parallel"):
        from bench_utils.datasets.amazon import AmazonReviewsDataset

        dataset = AmazonReviewsDataset(split="test")
        return list(dataset.iter_item_reviews(count, 60))
    if label == "debate":
        from bench_utils.datasets.tatqa.tatqa import TatQADataset

        dataset = TatQADataset(split="test", num_questions=6)
        return list(dataset.iter_context_qa_pairs(count))
    raise ValueError(f"Unsupported task label: {label!r}")


def _build_iterative(
    generation_config, sample_index: int = 0, *, sample=None, task_id: str | None = None
) -> TaskSpec:
    from bench_programs.iterative.helium import HeliumIterativeProgram
    from helium2.utils import iter_batch

    if sample is None:
        sample = _load_samples("iterative", sample_index + 1)[sample_index]
    _item, reviews = sample
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
        task_id=task_id if task_id is not None else f"iterative-{sample_index}",
        label="iterative",
        agent_name=agent.name,
        graph=graph,
        generation_config=generation_config,
        sample_index=sample_index,
    )


def _build_debate(
    generation_config, sample_index: int = 0, *, sample=None, task_id: str | None = None
) -> TaskSpec:
    from bench_programs.debate.helium import HeliumDebateProgram

    if sample is None:
        sample = _load_samples("debate", sample_index + 1)[sample_index]
    context, questions, _answers = sample

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
        task_id=task_id if task_id is not None else f"debate-{sample_index}",
        label="debate",
        agent_name=agent.name,
        graph=graph,
        generation_config=generation_config,
        sample_index=sample_index,
    )


def _build_parallel(
    generation_config, sample_index: int = 0, *, sample=None, task_id: str | None = None
) -> TaskSpec:
    from bench_programs.parallel.helium import HeliumParallelProgram
    from helium2.utils import iter_batch

    if sample is None:
        sample = _load_samples("parallel", sample_index + 1)[sample_index]
    item, reviews = sample
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
        task_id=task_id if task_id is not None else f"parallel-{sample_index}",
        label="parallel",
        agent_name=agent.name,
        graph=graph,
        generation_config=generation_config,
        sample_index=sample_index,
    )


def build_fixed_dataset(
    task_queue: Sequence[tuple[str, int]] | None = None,
) -> list[TaskSpec]:
    """Build independent tasks in TASK_QUEUE order, including repeated samples.

    Indices are zero-based positions in each existing loader's output order,
    not raw dataset row numbers. The historical function name is retained for
    scheduler callers; the default queue is now configured in config.py.
    """
    selected = cfg.TASK_QUEUE if task_queue is None else task_queue
    queue: list[tuple[TaskLabel, int]] = []
    for position, entry in enumerate(selected):
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(f"TASK_QUEUE[{position}] must be a (label, sample_index) pair")
        label, index = entry
        if label not in ("iterative", "debate", "parallel"):
            raise ValueError(f"TASK_QUEUE[{position}] has unsupported label {label!r}")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(
                f"TASK_QUEUE[{position}] sample_index must be a non-negative integer, "
                f"got {index!r}"
            )
        queue.append((cast(TaskLabel, label), index))
    if not queue:
        return []

    # Validate the full queue before changing module aliases or reading data.
    _install_helium2_as_helium()
    counts: dict[TaskLabel, int] = {}
    for label, index in queue:
        counts[label] = max(counts.get(label, 0), index + 1)
    samples: dict[TaskLabel, list[Any]] = {}
    for label, count in counts.items():
        try:
            loaded = _load_samples(label, count)
        except ValueError as exc:
            raise ValueError(
                f"Cannot load {label} sample_index={count - 1}: {exc}"
            ) from exc
        if len(loaded) < count:
            raise ValueError(
                f"Cannot load {label} sample_index={count - 1}: "
                f"only {len(loaded)} samples available"
            )
        samples[label] = loaded

    builders = {
        "iterative": _build_iterative,
        "debate": _build_debate,
        "parallel": _build_parallel,
    }
    occurrences: dict[tuple[TaskLabel, int], int] = {}
    tasks: list[TaskSpec] = []
    for label, index in queue:
        key = (label, index)
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        task_id = f"{label}-{index}"
        if occurrence:
            task_id += f"-repeat{occurrence}"
        tasks.append(
            builders[label](
                _generation_config(),
                sample_index=index,
                sample=samples[label][index],
                task_id=task_id,
            )
        )
    return tasks
