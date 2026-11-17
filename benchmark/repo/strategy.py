"""Initial adaptive scheduling and the single-card control policy.

The adaptive policy selects one of rules 1--4 once per segment. Rule 5 (tail
acceleration) remains a separate scheduler step and is not implemented here.
"""

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
    """Accumulated initial allocations, with the selected policy recorded."""

    assignments: list[Assignment] = field(default_factory=list)
    strategy_name: str | None = None
    adaptive_branch: int | None = None

    def append(self, assignment: Assignment) -> None:
        self.assignments.append(assignment)

    def extend(self, items: Sequence[Assignment]) -> None:
        self.assignments.extend(items)

    def lines(self) -> list[str]:
        out = ["# schedule_plan"]
        if self.strategy_name is not None:
            metadata = f"# strategy={self.strategy_name}"
            if self.adaptive_branch is not None:
                metadata += f" adaptive_branch={self.adaptive_branch}"
            out.append(metadata)
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
        k_wish_table: dict[str, int],
    ) -> list[Assignment]:
        """Plan pending work using only the currently free replicas."""
        ...


class AllSingleCardStrategy:
    """Control policy: original FIFO order, always one card per task.

    Reads `label` only to require a k_max table entry; ignores values above one.
    """

    def plan(
        self,
        queue: Sequence[TaskRef],
        free_replicas: Sequence[Replica],
        k_max_table: dict[str, int],
        *,
        total_cards: int,
        running_labels: Collection[str],
        k_wish_table: dict[str, int],
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


class AdaptiveStrategy:
    """Select one initial rule and keep it for the whole segment.

    1. Fewer tasks than cards: retain FIFO, preserve every pending task's
       single-card start, and allocate the remaining budget by task type.
    2. All parallel tasks fit at k_wish and the short-task budget fits: promote
       parallel tasks stably and start each on k_wish cards.
    3. A full number of parallel waves plus an admissible remainder: promote
       parallel tasks stably, give only the first remainder k_wish cards, and
       launch every later task on one card.
    4. Otherwise: original FIFO, all single-card.

    No branch predicts durations or accumulates cards. Multi-card launches in
    rules 2/3 fit in the initial card pool by admission. The caller removes each
    returned assignment from the pending queue and calls again after updating
    free replicas. Create a fresh instance per segment or call reset().
    """

    _LABELS = ("iterative", "debate", "parallel")

    def __init__(self) -> None:
        self.reset()

    @property
    def selected_policy(self) -> int | None:
        return self._selected_policy

    def reset(self) -> None:
        """Forget the previous segment before using the same policy instance."""
        self._selected_policy: int | None = None
        self._total_cards: int | None = None
        self._caps: dict[str, int] = {}
        self._wish = 0
        self._order: tuple[str, ...] = ()
        self._labels: dict[str, str] = {}
        self._wide_ids: set[str] = set()

    @staticmethod
    def _positive_int(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        return value

    def _initialize(
        self,
        queue: Sequence[TaskRef],
        total_cards: int,
        k_max_table: dict[str, int],
        k_wish_table: dict[str, int],
    ) -> None:
        labels: dict[str, str] = {}
        for task in queue:
            if task.label not in self._LABELS:
                raise ValueError(f"Unsupported task label: {task.label!r}")
            if task.task_id in labels:
                raise ValueError(f"Duplicate task_id in initial queue: {task.task_id!r}")
            labels[task.task_id] = task.label
        caps = {
            label: min(
                self._positive_int(k_max_table[label], f"k_max_table[{label!r}]"),
                total_cards,
            )
            for label in self._LABELS
        }
        wish = self._positive_int(k_wish_table["parallel"], "k_wish_table['parallel']")
        parallel = [task for task in queue if task.label == "parallel"]
        short = [task for task in queue if task.label != "parallel"]
        p, s = len(parallel), len(short)
        q, r = divmod(p, total_cards)
        valid_wish = 2 <= wish <= caps["parallel"]
        if len(queue) <= total_cards - 1:
            policy = 1
        elif (
            valid_wish
            and p >= 1
            and wish * p <= total_cards
            and wish * p + s <= 2 * total_cards
        ):
            policy = 2
        elif (
            valid_wish
            and q >= 1
            and 1 <= r < total_cards
            and wish * r <= total_cards
            and wish * r + s <= 2 * total_cards
        ):
            policy = 3
        else:
            policy = 4

        ordered = parallel + short if policy in (2, 3) else list(queue)
        wide = parallel if policy == 2 else parallel[:r] if policy == 3 else []
        # Commit all initial decisions together. Later calls use these snapshots,
        # even when the pending queue shrinks or input table objects are edited.
        self._selected_policy = policy
        self._total_cards = total_cards
        self._caps = caps
        self._wish = wish
        self._order = tuple(task.task_id for task in ordered)
        self._labels = labels
        self._wide_ids = {task.task_id for task in wide}

    def _pending_in_order(self, queue: Sequence[TaskRef]) -> list[TaskRef]:
        pending: dict[str, TaskRef] = {}
        for task in queue:
            if task.task_id in pending:
                raise ValueError(f"Duplicate task_id in pending queue: {task.task_id!r}")
            if task.task_id not in self._labels or self._labels[task.task_id] != task.label:
                raise ValueError(
                    f"Task {task.task_id!r} was not in this segment's initial queue "
                    "with the same label; call reset() for a new segment"
                )
            pending[task.task_id] = task
        return [pending[task_id] for task_id in self._order if task_id in pending]

    def plan(
        self,
        queue: Sequence[TaskRef],
        free_replicas: Sequence[Replica],
        k_max_table: dict[str, int],
        *,
        total_cards: int,
        running_labels: Collection[str],
        k_wish_table: dict[str, int],
    ) -> list[Assignment]:
        total_cards = self._positive_int(total_cards, "total_cards")
        if len(free_replicas) > total_cards or len(set(free_replicas)) != len(free_replicas):
            raise ValueError("Free replicas must be distinct and fit within total_cards")
        if self._total_cards is not None and total_cards != self._total_cards:
            raise ValueError("total_cards changed during the segment; call reset() first")
        if not queue:
            return []
        if self._selected_policy is None:
            self._initialize(queue, total_cards, k_max_table, k_wish_table)
        pending = self._pending_in_order(queue)
        if not free_replicas:
            return []

        head = pending[0]
        n, free_count = len(pending), len(free_replicas)
        if self._selected_policy == 1:
            # Admission and each prior launch preserve F >= n. Violating this
            # means the caller started this policy on an already occupied pool.
            if free_count < n:
                raise RuntimeError("Rule 1 requires one free replica per pending task")
            if head.label == "parallel":
                k = min(self._caps["parallel"], free_count - (n - 1))
            elif head.label == "debate":
                budget = sum(
                    self._caps["parallel"] if task.label == "parallel" else 1
                    for task in pending[1:]
                )
                k = min(self._caps["debate"], max(1, free_count - budget))
            else:
                k = 1
        elif head.task_id in self._wide_ids:
            k = self._wish
            if free_count < k:
                raise RuntimeError(
                    "Admitted initial parallel allocations must fit without waiting; "
                    "the policy requires the full free pool at segment start"
                )
        elif self._selected_policy == 2 and head.label == "debate":
            # All P were promoted and launched before any S reaches the head.
            # Budget one replica for every later S before giving D any surplus.
            k = (
                1
                if free_count < n
                else min(self._caps["debate"], free_count - (n - 1))
            )
        else:
            k = 1
        return [Assignment(head.task_id, head.label, tuple(free_replicas[:k]))]
