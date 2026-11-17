"""Scheduler pipeline (experiment group default = helium_new2).

Flow:
  1. build dataset  → TaskSpec list (label + compiled graph, flavor-agnostic)
  2. start vLLM pool
  3. while queue or running: if free cards, strategy.plan → launch; wait any finish → release
  4. write per-task k/time, Σ execute, and segment clearance to a result file

Disjoint card sets run in parallel; the strategy may hold free cards for parallel.
"""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from . import config as cfg
from .dataset import TaskSpec, build_fixed_dataset
from .pool import Replica, VLLMPool
from .runner import Flavor, HeliumPorts, run_task_blocking
from .strategy import (
    AllSingleCardStrategy,
    Assignment,
    LoadAwareStrategy,
    SchedulePlan,
    ScheduleStrategy,
    TaskRef,
)

if TYPE_CHECKING:
    from helium_new2.runtime.tail_resources import TailResourceState

# Inherited by fork workers: pass task_id only (no dill graph round-trip).
_FORK_TASKS: dict[str, TaskSpec] = {}
# All shared states must exist before the executor forks its first worker.
_FORK_TAIL_RESOURCES: dict[str, TailResourceState] = {}


def _helium_port_slots(n: int) -> list[HeliumPorts]:
    """One Helium port pair per concurrent slot (≤ #devices), like vLLM BASE_PORT+i."""
    return [
        HeliumPorts(
            server_port=cfg.HELIUM_SERVER_BASE_PORT + i,
            lmcache_monitor_port=cfg.LMCACHE_MONITOR_BASE_PORT + i * 2,
        )
        for i in range(n)
    ]


def _run_fork_task(
    task_id: str,
    replicas: tuple[Replica, ...] | list[Replica],
    flavor: Flavor,
    helium_ports: HeliumPorts,
    timeout_s: float = 900.0,
) -> float:
    """Fork-pool entry: look up inherited TaskSpec, apply scheduler ports, run."""
    task = _FORK_TASKS[task_id]
    return run_task_blocking(
        task,
        list(replicas),
        flavor=flavor,
        timeout_s=timeout_s,
        helium_ports=helium_ports,
        tail_resources=_FORK_TAIL_RESOURCES.get(task_id),
    )


@dataclass
class TaskResult:
    task_id: str
    label: str
    k: int
    devices: tuple[str, ...]
    ports: tuple[int, ...]
    execute_seconds: float
    flavor: str
    wall_start_seconds: float
    wall_end_seconds: float
    initial_k: int | None = None
    card_history: tuple[tuple[float, tuple[str, ...]], ...] = ()
    requests_by_device: tuple[tuple[str, int], ...] = ()
    sample_index: int | None = None


@dataclass
class ScheduleReport:
    plan: SchedulePlan
    results: list[TaskResult]
    total_execute_seconds: float
    clearance_seconds: float
    flavor: str
    tail_acceleration: bool = False

    def lines(self) -> list[str]:
        out: list[str] = [
            f"# report flavor={self.flavor}",
            *self.plan.lines(),
            "# results",
        ]
        for r in self.results:
            devices = ",".join(r.devices)
            out.append(
                f"{r.task_id} label={r.label} k={r.k} devices={devices} "
                f"execute_seconds={r.execute_seconds:.2f} "
                f"wall_start={r.wall_start_seconds:.2f} "
                f"wall_end={r.wall_end_seconds:.2f}"
                + (f" sample_index={r.sample_index}" if r.sample_index is not None else "")
            )
            if r.initial_k is not None:
                out.append(
                    f"tail_summary task_id={r.task_id} "
                    f"initial_k={r.initial_k} final_k={r.k}"
                )
                for at, held in r.card_history:
                    out.append(
                        f"card_allocation task_id={r.task_id} at={at:.6f} "
                        f"k={len(held)} devices={','.join(held)}"
                    )
                for device, count in r.requests_by_device:
                    out.append(
                        f"engine_requests task_id={r.task_id} device={device} count={count}"
                    )
        if self.tail_acceleration:
            out.append("tail_acceleration=enabled")
        out.append(f"SUM execute_seconds={self.total_execute_seconds:.2f}")
        out.append(f"clearance_seconds={self.clearance_seconds:.2f}")
        return out

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines()) + "\n", encoding="utf-8")
        return path


@dataclass
class _InFlight:
    assignment: Assignment
    task: TaskSpec
    replicas: list[Replica]
    helium_ports: HeliumPorts
    future: Future[float]
    wall_start_seconds: float
    tail_resources: TailResourceState | None = None
    card_history: list[tuple[float, tuple[str, ...]]] = field(default_factory=list)


def _queue_from_dataset(tasks: Sequence[TaskSpec]) -> list[TaskRef]:
    return [TaskRef(task_id=t.task_id, label=t.label) for t in tasks]


def _validate_assignment(assignment: Assignment, task: TaskSpec) -> None:
    if task.label != assignment.label:
        raise RuntimeError(
            f"plan/dataset label mismatch for {assignment.task_id}: "
            f"plan={assignment.label!r} dataset={task.label!r}"
        )


def run_experiment(
    strategy: ScheduleStrategy | None = None,
    devices: list[str] | None = None,
    flavor: Flavor = "helium_new2",
    result_path: Path | str | None = None,
    tasks: Sequence[TaskSpec] | None = None,
    enable_tail_acceleration: bool | None = None,
) -> ScheduleReport:
    """Full experiment-group run. Dataset graphs work for helium2 / helium_new2."""
    # The config toggle belongs only to the experiment group; an explicit request
    # for baseline expansion is rejected instead of silently changing the control.
    if enable_tail_acceleration is True and flavor != "helium_new2":
        raise ValueError("Tail acceleration is supported only by helium_new2")
    tail_enabled = flavor == "helium_new2" and (
        cfg.ENABLE_TAIL_ACCELERATION
        if enable_tail_acceleration is None
        else enable_tail_acceleration
    )
    if tail_enabled:
        if cfg.K_WISH_TABLE.get("parallel", 0) < 1:
            raise ValueError("K_WISH_TABLE['parallel'] must be >= 1")
        if cfg.K_MAX_TABLE.get("parallel", 0) < 1:
            raise ValueError("K_MAX_TABLE['parallel'] must be >= 1")
    if strategy is None:
        strategy = (
            LoadAwareStrategy() if flavor == "helium_new2" else AllSingleCardStrategy()
        )
    # 1) Dataset first (or reuse a prebuilt one): label + graph ready for either flavor.
    if tasks is None:
        tasks = build_fixed_dataset()
    by_id: dict[str, TaskSpec] = {t.task_id: t for t in tasks}
    queue = _queue_from_dataset(tasks)
    # Populate before creating the fork pool so workers inherit the graph objects.
    _FORK_TASKS.clear()
    _FORK_TASKS.update(by_id)
    _FORK_TAIL_RESOURCES.clear()

    plan = SchedulePlan()
    results: list[TaskResult] = []
    in_flight: dict[Future[float], _InFlight] = {}
    pool = VLLMPool(devices=devices)
    helium_free: list[HeliumPorts] = []
    try:
        # 2) Resident engines + Helium port slots (scheduler-owned, like engine ports).
        pool.start()
        context = get_context("fork")
        if tail_enabled:
            from helium_new2.runtime.tail_resources import TailResourceState

            endpoints = [(r.base_url, r.model, "EMPTY") for r in pool.replicas]
            _FORK_TAIL_RESOURCES.update(
                {
                    task.task_id: TailResourceState(endpoints, context=context)
                    for task in tasks
                    if task.label == "parallel"
                }
            )
        helium_free = _helium_port_slots(len(pool.replicas))
        # Segment start = scheduling begins (pool already warm). Multi-segment later:
        # clearance = max(finish of this segment's tasks) - segment_start, NOT "queue empty".
        segment_start = time.perf_counter()
        max_workers = max(1, len(pool.replicas))
        # fork: isolate Helium close(); inherit TaskSpec via _FORK_TASKS (no dill).
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=context,
        ) as executor:
            # 3) Plan & launch (or hold cards); any finish → release & replan.
            while queue or in_flight:
                _launch_while_free(
                    strategy=strategy,
                    queue=queue,
                    pool=pool,
                    helium_free=helium_free,
                    by_id=by_id,
                    flavor=flavor,
                    plan=plan,
                    executor=executor,
                    in_flight=in_flight,
                    arrival=segment_start,
                )
                if tail_enabled:
                    _accelerate_tail(
                        queue=queue,
                        pool=pool,
                        in_flight=in_flight,
                        arrival=segment_start,
                    )
                if not in_flight:
                    if queue:
                        raise RuntimeError(
                            "Strategy returned no assignment while queue is non-empty "
                            f"and no tasks are running (free={len(pool.free_replicas)}, "
                            f"queue_len={len(queue)}, head={queue[0].task_id})"
                        )
                    break
                done, _ = wait(
                    set(in_flight.keys()),
                    return_when=FIRST_COMPLETED,
                    # A P can become ready while another task keeps running.
                    # Only poll while tail expansion could use a free card.
                    timeout=(
                        0.1 if tail_enabled and not queue and pool.free_replicas else None
                    ),
                )
                for fut in done:
                    item = in_flight.pop(fut)
                    try:
                        elapsed = fut.result()
                    except BaseException:
                        if item.tail_resources is None:
                            pool.release(item.replicas)
                        else:
                            # Abort this experiment without recycling replicas
                            # whose failed client cleanup cannot be trusted.
                            item.tail_resources.seal()
                        helium_free.append(item.helium_ports)
                        raise
                    if item.tail_resources is not None:
                        item.tail_resources.seal()
                    tail_snapshot = (
                        item.tail_resources.snapshot()
                        if item.tail_resources is not None
                        else None
                    )
                    if tail_snapshot is not None and any(tail_snapshot["in_flight"].values()):
                        raise RuntimeError(
                            f"Tail task {item.task.task_id} finished with active HTTP requests; "
                            "stopping the experiment without recycling its replicas"
                        )
                    pool.release(item.replicas)
                    helium_free.append(item.helium_ports)
                    wall_end = time.perf_counter() - segment_start
                    results.append(
                        TaskResult(
                            task_id=item.task.task_id,
                            label=item.task.label,
                            k=len(item.replicas),
                            devices=tuple(r.device for r in item.replicas),
                            ports=tuple(r.port for r in item.replicas),
                            execute_seconds=elapsed,
                            flavor=flavor,
                            wall_start_seconds=item.wall_start_seconds,
                            wall_end_seconds=wall_end,
                            sample_index=item.task.sample_index,
                            initial_k=(
                                item.assignment.k if item.tail_resources is not None else None
                            ),
                            card_history=tuple(item.card_history),
                            requests_by_device=(
                                tuple(
                                    (r.device, tail_snapshot["request_counts"][r.base_url])
                                    for r in item.replicas
                                )
                                if tail_snapshot is not None
                                else ()
                            ),
                        )
                    )
    finally:
        pool.stop()
        _FORK_TAIL_RESOURCES.clear()

    # Per-segment clearance: last finish among this segment's tasks − segment start.
    # Single-segment MVP: all dataset tasks belong to the segment. Do not key off
    # "global queue empty" (wrong once multi-segment leftovers exist).
    clearance = max((r.wall_end_seconds for r in results), default=0.0)

    report = ScheduleReport(
        plan=plan,
        results=results,
        total_execute_seconds=sum(r.execute_seconds for r in results),
        clearance_seconds=clearance,
        flavor=flavor,
        tail_acceleration=tail_enabled,
    )
    path = Path(result_path) if result_path is not None else cfg.RESULT_PATH
    report.write(path)
    return report


def _launch_while_free(
    *,
    strategy: ScheduleStrategy,
    queue: list[TaskRef],
    pool: VLLMPool,
    helium_free: list[HeliumPorts],
    by_id: dict[str, TaskSpec],
    flavor: Flavor,
    plan: SchedulePlan,
    executor: ProcessPoolExecutor,
    in_flight: dict[Future[float], _InFlight],
    arrival: float,
) -> int:
    """Keep calling strategy while free cards remain; submit each assignment."""
    launched = 0
    while queue and pool.free_replicas and helium_free:
        batch = strategy.plan(
            queue,
            pool.free_replicas,
            cfg.K_MAX_TABLE,
            total_cards=len(pool.replicas),
            running_labels={item.task.label for item in in_flight.values()},
            k_wait_table=cfg.K_WAIT_TABLE,
        )
        if not batch:
            break
        for assignment in batch:
            if not helium_free:
                break
            if assignment.task_id not in {t.task_id for t in queue}:
                raise RuntimeError(
                    f"strategy assigned task not in queue: {assignment.task_id}"
                )
            task = by_id[assignment.task_id]
            _validate_assignment(assignment, task)
            replicas = pool.acquire_replicas(assignment.replicas)
            helium_ports = helium_free.pop()
            plan.append(assignment)
            queue[:] = [t for t in queue if t.task_id != assignment.task_id]
            wall_start = time.perf_counter() - arrival
            tail_resources = _FORK_TAIL_RESOURCES.get(task.task_id)
            if tail_resources is not None:
                tail_resources.initialize([r.base_url for r in replicas])
            future = executor.submit(
                _run_fork_task,
                task.task_id,
                tuple(replicas),
                flavor,
                helium_ports,
            )
            in_flight[future] = _InFlight(
                assignment=assignment,
                task=task,
                replicas=replicas,
                helium_ports=helium_ports,
                future=future,
                wall_start_seconds=wall_start,
                tail_resources=tail_resources,
                card_history=(
                    [(wall_start, tuple(r.device for r in replicas))]
                    if tail_resources is not None
                    else []
                ),
            )
            launched += 1
    return launched


def _accelerate_tail(
    *,
    queue: Sequence[TaskRef],
    pool: VLLMPool,
    in_flight: dict[Future[float], _InFlight],
    arrival: float,
) -> None:
    """Only append already-free replicas after every pending task is dispatched.

    The shared grant is also visible inside the HTTP client processes. A grant
    never moves a submitted request or withdraws an already assigned replica.
    Tail expansion stops at k_wish, bounded by k_max and the physical pool.
    """
    if queue or not pool.free_replicas:
        return
    target = min(
        cfg.K_WISH_TABLE["parallel"],
        cfg.K_MAX_TABLE["parallel"],
        len(pool.replicas),
    )
    candidates: list[tuple[float, _InFlight]] = []
    for item in in_flight.values():
        if item.tail_resources is None or item.future.done():
            continue
        state = item.tail_resources.snapshot()
        if state["ready"] and state["running"]:
            candidates.append((state["started_at"], item))
    # Stable sort preserves original dispatch order for equal start timestamps.
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    for _, item in candidates:
        count = min(target - len(item.replicas), len(pool.free_replicas))
        if count <= 0 or item.future.done():
            continue
        extra = pool.acquire(count)
        try:
            accepted = item.tail_resources.grant([r.base_url for r in extra])
        except BaseException:
            pool.release(extra)
            raise
        if not accepted:
            pool.release(extra)
            continue
        before = len(item.replicas)
        item.replicas.extend(extra)
        at = time.perf_counter() - arrival
        item.card_history.append((at, tuple(r.device for r in item.replicas)))
        print(
            f"tail_acceleration task_id={item.task.task_id} at={at:.6f} "
            f"k_before={before} k_after={len(item.replicas)} "
            f"added_devices={','.join(r.device for r in extra)}",
            flush=True,
        )


def dry_run_plan(
    strategy: ScheduleStrategy | None = None,
    devices: list[str] | None = None,
    tasks: Sequence[TaskSpec] | None = None,
) -> SchedulePlan:
    """No GPU: build dataset + simulate plan with parallel card occupancy.

    Offers free cards to the strategy (which may wait); then releases
    the oldest in-flight assignment (FIFO) and replans — mirrors the runtime loop.
    """
    if strategy is None:
        strategy = LoadAwareStrategy()
    if tasks is None:
        tasks = build_fixed_dataset()
    queue = _queue_from_dataset(tasks)
    device_list = list(devices if devices is not None else cfg.DEVICES)
    free = [
        Replica(
            device=d,
            host=cfg.HOST,
            port=cfg.BASE_PORT + i,
            model=cfg.MODEL,
        )
        for i, d in enumerate(device_list)
    ]
    plan = SchedulePlan()
    running: list[Assignment] = []
    while queue or running:
        while queue and free:
            batch = strategy.plan(
                queue,
                free,
                cfg.K_MAX_TABLE,
                total_cards=len(device_list),
                running_labels={assignment.label for assignment in running},
                k_wait_table=cfg.K_WAIT_TABLE,
            )
            if not batch:
                break
            for assignment in batch:
                for r in assignment.replicas:
                    free.remove(r)
                plan.append(assignment)
                running.append(assignment)
                queue = [t for t in queue if t.task_id != assignment.task_id]
        if not running:
            if queue:
                raise RuntimeError("dry_run: strategy produced empty plan")
            break
        # Simulate one completion (FIFO); release its cards and replan.
        finished = running.pop(0)
        free.extend(finished.replicas)
    return plan
