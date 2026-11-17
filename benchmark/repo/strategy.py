"""Pluggable scheduling strategies: dataset labels → schedule plan."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .pool import Replica


@dataclass(frozen=True)
class TaskRef:
    """Strategy view of a queue item (identity + label only; no graph)."""

    task_id: str
    label: str


@dataclass(frozen=True)
class Assignment:
    """One planned launch: which task, how many / which replicas."""

    task_id: str
    label: str
    replicas: tuple[Replica, ...]

    @property
    def k(self) -> int:
        return len(self.replicas)


@dataclass
class SchedulePlan:
    """Accumulated run plan produced by the strategy (process + structure).

    Even when every entry is k=1, this object is the explicit plan the
    scheduler executes against — not an ad-hoc side effect of acquire().
    """

    assignments: list[Assignment] = field(default_factory=list)

    def append(self, assignment: Assignment) -> None:
        self.assignments.append(assignment)

    def extend(self, items: Sequence[Assignment]) -> None:
        self.assignments.extend(items)

    def lines(self) -> list[str]:
        out = ["# schedule_plan"]
        for i, a in enumerate(self.assignments):
            devices = ",".join(r.device for r in a.replicas)
            ports = ",".join(str(r.port) for r in a.replicas)
            out.append(
                f"plan[{i}] task_id={a.task_id} label={a.label} "
                f"k={a.k} devices={devices} ports={ports}"
            )
        return out


class ScheduleStrategy(Protocol):
    def plan(
        self,
        queue: Sequence[TaskRef],
        free_replicas: Sequence[Replica],
        k_max_table: dict[str, int],
        *,
        total_cards: int,
        running_labels: Collection[str],
        k_wait_table: dict[str, int],
    ) -> list[Assignment]:
        """Read pending queue and live resource state; [] may mean wait for cards."""
        ...


class AllSingleCardStrategy:
    """MVP experiment strategy: always k=1 for the queue head when a card is free.

    Reads `label` only to require a k_max table entry; does not use k_max > 1.
    """

    def plan(
        self,
        queue: Sequence[TaskRef],
        free_replicas: Sequence[Replica],
        k_max_table: dict[str, int],
        *,
        total_cards: int,
        running_labels: Collection[str],
        k_wait_table: dict[str, int],
    ) -> list[Assignment]:
        if not queue or not free_replicas:
            return []
        head = queue[0]
        if head.label not in k_max_table:
            raise KeyError(
                f"label {head.label!r} missing from k_max_table "
                f"(keys={sorted(k_max_table)})"
            )
        if k_max_table[head.label] < 1:
            raise ValueError(f"k_max for label {head.label!r} must be >= 1")
        return [
            Assignment(
                task_id=head.task_id,
                label=head.label,
                replicas=(free_replicas[0],),
            )
        ]


class LoadAwareStrategy:
    """FIFO policy using pending queue pressure and running task types.

    n counts only pending tasks. At n >= total_cards, start the head on one
    card. Otherwise iterative uses one card, debate takes available cards up
    to its cap, and parallel waits for its capped target. If the remaining
    running tasks are all parallel, start parallel with available cards instead.

    Each call assigns at most the head, so the scheduler updates queue length,
    occupancy and running types before deciding again. Running tasks never resize.
    """

    def plan(
        self,
        queue: Sequence[TaskRef],
        free_replicas: Sequence[Replica],
        k_max_table: dict[str, int],
        *,
        total_cards: int,
        running_labels: Collection[str],
        k_wait_table: dict[str, int],
    ) -> list[Assignment]:
        if total_cards < 1:
            raise ValueError("total_cards must be >= 1")
        if not queue or not free_replicas:
            return []

        head = queue[0]
        if head.label not in ("iterative", "debate", "parallel"):
            raise ValueError(f"Unsupported task label: {head.label!r}")
        k_max = k_max_table[head.label]
        if k_max < 1:
            raise ValueError(f"k_max for label {head.label!r} must be >= 1")

        if len(queue) >= total_cards or head.label == "iterative":
            k = 1
        else:
            target = min(k_max, total_cards)
            k = min(len(free_replicas), target)
            if (
                head.label == "parallel"
                and k < target
                and running_labels
                and any(label != "parallel" for label in running_labels)
            ):
                # Keep the free cards idle; do not bypass the queue head.
                return []

        return [
            Assignment(
                task_id=head.task_id,
                label=head.label,
                replicas=tuple(free_replicas[:k]),
            )
        ]


class ParallelReserveStrategy:
    """FIFO policy preserving concurrency and accumulating cards for parallel.

    Busy (pending count >= total_cards): start the head on one card.
    Idle: iterative uses one card; debate budgets a capped allocation for each
    later parallel and one card per other later task, then uses remaining cards.
    If that budget does not fit, debate still starts on one card without waiting.
    Parallel preserves the single-card starts for other pending tasks when all
    pending tasks can start now. Otherwise, including a lone pending parallel,
    it may hold cards until k_wait_table["parallel"] cards are available.

    A started wait stays attached to the head: reaching enough free cards for
    single-card launches must not cancel accumulation. Only non-parallel running
    tasks justify waiting; if none remain, use available cards immediately.
    Waiting targets and allocations are capped by k_max_table and total_cards.
    Both tables are explicit plan inputs supplied by the scheduler from config.

    Create a fresh instance per experiment, or reset() after an aborted run.
    """

    def __init__(self) -> None:
        self._waiting_task_id: str | None = None
        self._waiting_target = 0

    def reset(self) -> None:
        """Discard any pending wait before reusing this policy for a new run."""
        self._waiting_task_id = None
        self._waiting_target = 0

    def plan(
        self,
        queue: Sequence[TaskRef],
        free_replicas: Sequence[Replica],
        k_max_table: dict[str, int],
        *,
        total_cards: int,
        running_labels: Collection[str],
        k_wait_table: dict[str, int],
    ) -> list[Assignment]:
        if total_cards < 1:
            raise ValueError("total_cards must be >= 1")
        if not queue:
            self.reset()
            return []

        head = queue[0]
        if head.task_id != self._waiting_task_id or head.label != "parallel":
            self.reset()
        if not free_replicas:
            return []
        if head.label not in ("iterative", "debate", "parallel"):
            raise ValueError(f"Unsupported task label: {head.label!r}")
        k_max = k_max_table[head.label]
        if k_max < 1:
            raise ValueError(f"k_max for label {head.label!r} must be >= 1")

        n = len(queue)
        free_count = len(free_replicas)
        cap = min(k_max, total_cards)
        # Extra cards beyond one for each of the other pending tasks.
        head_share = max(1, free_count - (n - 1))

        # Busy: start one task, then let the scheduler update n/F and replan.
        if n >= total_cards:
            k = 1
        # Idle iterative: never accumulate or allocate extra cards.
        elif head.label == "iterative":
            k = 1
        elif head.label == "debate":
            parallel_count = sum(task.label == "parallel" for task in queue[1:])
            tail_budget = n - 1
            if parallel_count:
                parallel_k_max = k_max_table["parallel"]
                if parallel_k_max < 1:
                    raise ValueError("k_max for label 'parallel' must be >= 1")
                parallel_cap = min(parallel_k_max, total_cards)
                tail_budget += parallel_count * (parallel_cap - 1)
            # Budget only, not an actual reservation in the pool. Debate never waits.
            k = min(cap, max(1, free_count - tail_budget))
        else:
            # Idle parallel: an existing wait takes priority over a fresh
            # concurrency-preserving allocation, even after F reaches n.
            k_wait = k_wait_table["parallel"]
            if k_wait < 1:
                raise ValueError("k_wait for label 'parallel' must be >= 1")

            already_waiting = self._waiting_task_id == head.task_id
            if not already_waiting and n > 1 and free_count >= n:
                # Fresh decision: leave one card per later task. This may give
                # parallel fewer than k_wait cards, but must not begin a wait.
                k = min(cap, head_share)
            else:
                # Continue a prior wait, consider accumulating scarce cards, or
                # accelerate the last pending parallel. Entering this branch
                # alone does not put the task in waiting state.
                target = min(
                    self._waiting_target if already_waiting else k_wait,
                    cap,
                )
                if free_count >= target:
                    # Start immediately. Extra cards are allowed only beyond
                    # the single-card budget for the later pending tasks.
                    k = min(free_count, cap, max(target, head_share))
                elif not any(label != "parallel" for label in running_labels):
                    # No non-parallel task to wait for, including nobody running.
                    k = min(free_count, cap)
                else:
                    # This is the only transition into (or continuation of)
                    # waiting state. FIFO holds the free cards for this task.
                    self._waiting_task_id = head.task_id
                    self._waiting_target = target
                    return []

        self.reset()
        return [
            Assignment(
                task_id=head.task_id,
                label=head.label,
                replicas=tuple(free_replicas[:k]),
            )
        ]
