"""Policy checks without models or engines: python -m unittest benchmark.repo.test_strategy."""

import unittest

from .dataset import TaskSpec
from .pool import Replica
from .scheduler import dry_run_plan
from .strategy import AllSingleCardStrategy, LoadAwareStrategy, TaskRef


class LoadAwareStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = LoadAwareStrategy()
        self.cards = [
            Replica(f"cuda:{i}", "127.0.0.1", 19100 + i, "model") for i in range(5)
        ]
        self.caps = {"iterative": 1, "debate": 3, "parallel": 4}
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

    def test_busy_includes_exactly_total_cards_pending(self) -> None:
        for n in (5, 6):
            with self.subTest(n=n):
                plan = self.plan(["parallel"] * n, 5)
                self.assertEqual([(a.task_id, a.k) for a in plan], [("0", 1)])

    def test_idle_iterative_and_debate_do_not_wait(self) -> None:
        self.assertEqual(self.plan(["iterative"], 5)[0].k, 1)
        self.assertEqual(self.plan(["debate"], 5)[0].k, 3)
        self.assertEqual(self.plan(["debate"], 1, ["iterative"])[0].k, 1)

    def test_parallel_waits_with_any_non_parallel_running(self) -> None:
        for free_count, running in (
            (1, ["parallel", "debate", "iterative"]),
            (2, ["parallel", "debate"]),
            (3, ["debate"]),
        ):
            with self.subTest(free_count=free_count, running=running):
                # A debate behind the head cannot consume the held cards.
                self.assertEqual(
                    self.plan(["parallel", "debate"], free_count, running), []
                )

    def test_parallel_rechecks_after_each_completion(self) -> None:
        self.assertEqual(
            self.plan(["parallel"], 2, ["parallel", "debate"]), []
        )
        # Last non-parallel task finishes: start now even below the target.
        self.assertEqual(self.plan(["parallel"], 3, ["parallel"])[0].k, 3)
        # Reaching the target also starts immediately, regardless of running types.
        self.assertEqual(self.plan(["parallel"], 4, ["iterative"])[0].k, 4)

    def test_parallel_target_is_capped_by_actual_card_count(self) -> None:
        self.assertEqual(self.plan(["parallel"], 2, total=2)[0].k, 2)
        self.assertEqual(self.plan(["parallel"], 1, total=1)[0].k, 1)

    def test_empty_queue_or_no_free_cards(self) -> None:
        self.assertEqual(self.plan([], 5), [])
        self.assertEqual(self.plan(["parallel"], 0, ["debate"]), [])

    def test_baseline_remains_single_card(self) -> None:
        plan = AllSingleCardStrategy().plan(
            [TaskRef("parallel-0", "parallel")],
            self.cards,
            self.caps,
            total_cards=5,
            running_labels=set(),
            k_wait_table=self.waits,
        )
        self.assertEqual(plan[0].k, 1)

    def dry_plan(self, labels):
        tasks = [
            TaskSpec(str(i), label, "agent", None, None)
            for i, label in enumerate(labels)
        ]
        return dry_run_plan(
            strategy=self.strategy,
            devices=[card.device for card in self.cards],
            tasks=tasks,
        ).assignments

    def test_launches_switch_from_busy_to_idle_without_waiting_for_completion(self) -> None:
        plan = self.dry_plan(
            ["iterative", "iterative", "debate", "iterative", "iterative", "iterative"]
        )
        # n=6 and n=5 start single-card; n=4 debate takes all 3 remaining cards.
        self.assertEqual([a.k for a in plan[:3]], [1, 1, 3])
        self.assertEqual([a.task_id for a in plan], [str(i) for i in range(6)])

    def test_all_parallel_queue_can_use_leftover_cards(self) -> None:
        plan = self.dry_plan(["parallel"] * 4)
        self.assertEqual([a.k for a in plan[:2]], [4, 1])
        self.assertTrue(set(plan[0].replicas).isdisjoint(plan[1].replicas))
        self.assertEqual(len(plan), 4)


if __name__ == "__main__":
    unittest.main()
