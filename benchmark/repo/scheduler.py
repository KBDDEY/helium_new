"""Scheduler pipeline (experiment group default = helium_new2).

Flow:
  1. build dataset  → TaskSpec list (label + compiled graph, flavor-agnostic)
  2. start vLLM pool
  3. while queue or running: if free cards, strategy.plan → launch; wait any finish → release
  4. write per-task k/time, Σ execute, and segment clearance to a result file

Matches outline: free cards start work immediately; disjoint card sets run in parallel.
"""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Sequence

from . import config as cfg
from .dataset import TaskSpec, build_fixed_dataset
from .pool import Replica, VLLMPool
from .runner import Flavor, HeliumPorts, run_task_blocking
from .strategy import (
    AllSingleCardStrategy,
    Assignment,
    SchedulePlan,
    ScheduleStrategy,
    TaskRef,
)

# Inherited by fork workers: pass task_id only (no dill graph round-trip).
_FORK_TASKS: dict[str, TaskSpec] = {}


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


@dataclass
class ScheduleReport:
    plan: SchedulePlan
    results: list[TaskResult]
    total_execute_seconds: float
    clearance_seconds: float
    flavor: str

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
            )
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
) -> ScheduleReport:
    """Full experiment-group run. Dataset graphs work for helium2 / helium_new2."""
    strategy = strategy or AllSingleCardStrategy()
    # 1) Dataset first (or reuse a prebuilt one): label + graph ready for either flavor.
    if tasks is None:
        tasks = build_fixed_dataset()
    by_id: dict[str, TaskSpec] = {t.task_id: t for t in tasks}
    queue = _queue_from_dataset(tasks)
    # Populate before creating the fork pool so workers inherit the graph objects.
    _FORK_TASKS.clear()
    _FORK_TASKS.update(by_id)

    plan = SchedulePlan()
    results: list[TaskResult] = []
    in_flight: dict[Future[float], _InFlight] = {}
    pool = VLLMPool(devices=devices)
    helium_free: list[HeliumPorts] = []
    try:
        # 2) Resident engines + Helium port slots (scheduler-owned, like engine ports).
        pool.start()
        helium_free = _helium_port_slots(len(pool.replicas))
        # Segment start = scheduling begins (pool already warm). Multi-segment later:
        # clearance = max(finish of this segment's tasks) - segment_start, NOT "queue empty".
        segment_start = time.perf_counter()
        max_workers = max(1, len(pool.replicas))
        # fork: isolate Helium close(); inherit TaskSpec via _FORK_TASKS (no dill).
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=get_context("fork"),
        ) as executor:
            # 3) Free cards → plan & launch; any finish → release & replan.
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
                )
                for fut in done:
                    item = in_flight.pop(fut)
                    try:
                        elapsed = fut.result()
                    except BaseException:
                        pool.release(item.replicas)
                        helium_free.append(item.helium_ports)
                        raise
                    pool.release(item.replicas)
                    helium_free.append(item.helium_ports)
                    wall_end = time.perf_counter() - segment_start
                    results.append(
                        TaskResult(
                            task_id=item.task.task_id,
                            label=item.task.label,
                            k=item.assignment.k,
                            devices=tuple(r.device for r in item.replicas),
                            ports=tuple(r.port for r in item.replicas),
                            execute_seconds=elapsed,
                            flavor=flavor,
                            wall_start_seconds=item.wall_start_seconds,
                            wall_end_seconds=wall_end,
                        )
                    )
    finally:
        pool.stop()

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
        batch = strategy.plan(queue, pool.free_replicas, cfg.K_MAX_TABLE)
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
            )
            launched += 1
    return launched


def dry_run_plan(
    strategy: ScheduleStrategy | None = None,
    devices: list[str] | None = None,
    tasks: Sequence[TaskSpec] | None = None,
) -> SchedulePlan:
    """No GPU: build dataset + simulate plan with parallel card occupancy.

    Fills free cards via strategy before any simulated completion; then releases
    the oldest in-flight assignment (FIFO) and replans — mirrors the runtime loop.
    """
    strategy = strategy or AllSingleCardStrategy()
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
            batch = strategy.plan(queue, free, cfg.K_MAX_TABLE)
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
