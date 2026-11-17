"""No-engine policy checks.

Run with: python -m unittest benchmark.repo.test_parallel_reserve_strategy
"""

import unittest
from unittest.mock import patch

from . import config as cfg
from .dataset import TaskSpec
from .pool import Replica
from .scheduler import dry_run_plan
from .strategy import ParallelReserveStrategy, TaskRef


class ParallelReserveStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = ParallelReserveStrategy()
        self.cards = [
            Replica(f"cuda:{i}", "127.0.0.1", 19100 + i, "model") for i in range(5)
        ]
        self.caps = {"iterative": 1, "debate": 3, "parallel": 3}
        self.waits = {"parallel": 2}

    def plan(self, labels, free_count, running=(), total=5):
        return self.strategy.plan(
            [TaskRef(str(i), label) for i, label in enumerate(labels)],
            self.cards[:free_count],
            self.caps,
            total_cards=total,
            running_labels=set(running),
            k_wait_table=self.waits,
        )

    def dry_plan(self, labels):
        tasks = [
            TaskSpec(str(i), label, "agent", None, None)
            for i, label in enumerate(labels)
        ]
        with (
            patch.object(cfg, "K_MAX_TABLE", self.caps),
            patch.object(cfg, "K_WAIT_TABLE", self.waits),
        ):
            return dry_run_plan(
                strategy=self.strategy,
                devices=[card.device for card in self.cards],
                tasks=tasks,
            ).assignments

    def test_busy_stays_single_card(self) -> None:
        for label in ("iterative", "debate", "parallel"):
            for n in (5, 6):
                with self.subTest(label=label, n=n):
                    self.assertEqual(self.plan([label] * n, 5)[0].k, 1)

    def test_all_parallel_preserves_immediate_single_card_starts(self) -> None:
        plan = self.dry_plan(["parallel"] * 4)
        self.assertEqual([a.k for a in plan], [2, 1, 1, 1])
        # All four assignments use distinct cards before any release is needed.
        self.assertEqual(len({r for a in plan for r in a.replicas}), 5)

    def test_all_debate_uses_only_spare_cards(self) -> None:
        plan = self.dry_plan(["debate"] * 4)
        self.assertEqual([a.k for a in plan], [2, 1, 1, 1])
        self.assertEqual(len({r for a in plan for r in a.replicas}), 5)

    def test_mixed_queue_keeps_extra_cards_for_parallel(self) -> None:
        plan = self.dry_plan(["debate", "parallel", "iterative"])
        self.assertEqual([a.k for a in plan], [1, 3, 1])

    def test_debate_uses_surplus_after_parallel_gets_its_cap(self) -> None:
        plan = self.dry_plan(["debate", "parallel"])
        self.assertEqual([a.k for a in plan], [2, 3])

    def test_debate_budget_tracks_configured_parallel_cap(self) -> None:
        self.caps["parallel"] = 2
        self.assertEqual(self.plan(["debate", "parallel"], 5)[0].k, 3)
        self.caps["parallel"] = 4
        self.assertEqual(self.plan(["debate", "parallel"], 5)[0].k, 1)

    def test_debate_budgets_every_later_parallel(self) -> None:
        self.cards = [
            Replica(f"cuda:{i}", "127.0.0.1", 19100 + i, "model") for i in range(8)
        ]
        plan = self.plan(["debate", "parallel", "parallel"], 8, total=8)
        self.assertEqual(plan[0].k, 2)
        # Insufficient budget must not make debate wait or bypass the head.
        plan = self.plan(["debate", "parallel", "parallel"], 5, total=8)
        self.assertEqual(plan[0].k, 1)

    def test_debate_never_waits_when_cards_are_scarce(self) -> None:
        plan = self.plan(["debate", "iterative"], 1, ["parallel", "iterative"])
        self.assertEqual(plan[0].k, 1)
        self.assertEqual(self.plan(["debate"], 5)[0].k, 3)

    def test_accumulated_card_is_not_given_back_to_other_pending_task(self) -> None:
        labels = ["parallel", "debate"]
        self.assertEqual(self.plan(labels, 1, ["debate", "iterative"]), [])
        # F now equals n: the prior wait must still give this parallel two cards.
        plan = self.plan(labels, 2, ["iterative"])
        self.assertEqual([(a.task_id, a.k) for a in plan], [("0", 2)])

    def test_fresh_parallel_does_not_wait_when_every_task_can_start(self) -> None:
        plan = self.plan(["parallel", "debate"], 2, ["iterative"])
        self.assertEqual([(a.task_id, a.k) for a in plan], [("0", 1)])

    def test_target_available_immediately_does_not_leave_waiting_state(self) -> None:
        # F < n, but the target is already available: launch rather than wait.
        plan = self.plan(["parallel", "debate", "iterative"], 2, ["iterative"])
        self.assertEqual(plan[0].k, 2)
        # Reuse the same task ID in a fresh scenario: no stale wait may survive.
        plan = self.plan(["parallel", "debate"], 2, ["iterative"])
        self.assertEqual(plan[0].k, 1)

    def test_waiting_parallel_can_take_more_than_its_wait_target(self) -> None:
        self.caps["parallel"] = 4
        labels = ["parallel", "debate"]
        self.assertEqual(self.plan(labels, 1, ["parallel", "debate"]), [])
        # A three-card debate releases cards. Use three, leave one for the tail.
        plan = self.plan(labels, 4, ["parallel"])
        self.assertEqual(plan[0].k, 3)

    def test_busy_to_idle_accumulates_a_single_card_release(self) -> None:
        queue = [
            TaskRef(str(i), label)
            for i, label in enumerate(
                ["iterative", "debate", "iterative", "debate", "parallel", "iterative"]
            )
        ]
        free = list(self.cards)
        running = []
        # Start the busy queue, then reach the parallel head with just one free card.
        for i in range(4):
            batch = self.strategy.plan(
                queue,
                free,
                self.caps,
                total_cards=5,
                running_labels={a.label for a in running},
                k_wait_table=self.waits,
            )
            self.assertEqual([(a.task_id, a.k) for a in batch], [(str(i), 1)])
            assignment = batch[0]
            for replica in assignment.replicas:
                free.remove(replica)
            running.append(assignment)
            queue.pop(0)

        self.assertEqual(len(free), 1)
        self.assertEqual(
            self.strategy.plan(
                queue,
                free,
                self.caps,
                total_cards=5,
                running_labels={a.label for a in running},
                k_wait_table=self.waits,
            ),
            [],
        )
        # A single debate finishes: its card joins the held card for parallel.
        free.extend(running.pop(1).replicas)
        batch = self.strategy.plan(
            queue,
            free,
            self.caps,
            total_cards=5,
            running_labels={a.label for a in running},
            k_wait_table=self.waits,
        )
        self.assertEqual([(a.task_id, a.k) for a in batch], [("4", 2)])

    def test_last_parallel_still_accumulates_cards(self) -> None:
        self.assertEqual(self.plan(["parallel"], 1, ["parallel", "debate"]), [])
        self.assertEqual(self.plan(["parallel"], 2, ["parallel"])[0].k, 2)

    def test_dry_run_replans_after_releasing_one_card(self) -> None:
        plan = self.dry_plan(
            ["iterative", "iterative", "iterative", "iterative", "parallel", "debate"]
        )
        self.assertEqual([a.k for a in plan], [1, 1, 1, 1, 2, 1])
        self.assertEqual([a.task_id for a in plan], [str(i) for i in range(6)])

    def test_only_parallel_running_cancels_wait(self) -> None:
        labels = ["parallel", "debate"]
        self.assertEqual(self.plan(labels, 1, ["parallel", "iterative"]), [])
        self.assertEqual(self.plan(labels, 1, ["parallel"])[0].k, 1)

    def test_empty_running_set_cannot_cause_wait(self) -> None:
        self.assertEqual(self.plan(["parallel"], 1)[0].k, 1)

    def test_empty_running_set_also_ends_an_existing_wait(self) -> None:
        self.assertEqual(self.plan(["parallel"], 1, ["debate"]), [])
        self.assertEqual(self.plan(["parallel"], 1)[0].k, 1)
        # The downgrade must clear its waiting record as well.
        self.assertEqual(self.plan(["parallel", "debate"], 2)[0].k, 1)

    def test_no_free_cards_keeps_existing_wait(self) -> None:
        labels = ["parallel", "debate"]
        self.assertEqual(self.plan(labels, 1, ["iterative"]), [])
        self.assertEqual(self.plan(labels, 0, ["iterative"]), [])
        self.assertEqual(self.plan(labels, 2, ["iterative"])[0].k, 2)

    def test_wait_clears_when_head_changes(self) -> None:
        self.assertEqual(self.plan(["parallel", "debate"], 1, ["iterative"]), [])
        plan = self.strategy.plan(
            [TaskRef("replacement", "parallel"), TaskRef("later", "debate")],
            self.cards[:2],
            self.caps,
            total_cards=5,
            running_labels={"iterative"},
            k_wait_table=self.waits,
        )
        self.assertEqual(plan[0].k, 1)

    def test_reset_and_empty_queue_clear_wait(self) -> None:
        labels = ["parallel", "debate"]
        self.assertEqual(self.plan(labels, 1, ["iterative"]), [])
        self.strategy.reset()
        self.assertEqual(self.plan(labels, 2, ["iterative"])[0].k, 1)
        self.assertEqual(self.plan(labels, 1, ["iterative"]), [])
        self.assertEqual(self.plan([], 5), [])
        self.assertEqual(self.plan(labels, 2, ["iterative"])[0].k, 1)

    def test_actual_card_count_and_cap_limit_wait_target(self) -> None:
        self.assertEqual(self.plan(["parallel"], 2, total=2)[0].k, 2)
        self.caps["parallel"] = 1
        self.assertEqual(self.plan(["parallel"], 1, ["debate"])[0].k, 1)

    def test_wait_target_can_be_configured(self) -> None:
        self.waits["parallel"] = 3
        self.assertEqual(self.plan(["parallel"], 2, ["debate"]), [])
        self.assertEqual(self.plan(["parallel"], 3, ["iterative"])[0].k, 3)

    def test_wait_target_above_cap_does_not_wait_for_unallocatable_cards(self) -> None:
        self.waits["parallel"] = 10
        self.assertEqual(self.plan(["parallel"], 2, ["iterative"]), [])
        self.assertEqual(self.plan(["parallel"], 3, ["iterative"])[0].k, 3)

    def test_scheduler_passes_configured_wait_target(self) -> None:
        self.waits["parallel"] = 3
        plan = self.dry_plan(
            ["iterative", "iterative", "iterative", "iterative", "parallel", "debate"]
        )
        self.assertEqual([a.k for a in plan], [1, 1, 1, 1, 3, 1])

    def test_invalid_wait_target(self) -> None:
        with self.assertRaises(ValueError):
            self.waits["parallel"] = 0
            self.plan(["parallel"], 1, ["debate"])


if __name__ == "__main__":
    unittest.main()
