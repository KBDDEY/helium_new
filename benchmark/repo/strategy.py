"""Pluggable scheduling strategies: dataset labels → schedule plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

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
    ) -> list[Assignment]:
        """Read queue labels + free cards → zero or more assignments."""


class AllSingleCardStrategy:
    """MVP experiment strategy: always k=1 for the queue head when a card is free.

    Reads `label` only to require a k_max table entry; does not use k_max > 1.
    """

    def plan(
        self,
        queue: Sequence[TaskRef],
        free_replicas: Sequence[Replica],
        k_max_table: dict[str, int],
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
