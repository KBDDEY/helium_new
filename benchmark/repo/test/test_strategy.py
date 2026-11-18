"""Server-side policy checks without models, engines or duration predictions.

Run from benchmark with PYTHONPATH=../src:. python -m unittest repo.test.test_strategy
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from .. import config as cfg
from ..dataset import TaskSpec
from ..pool import Replica
from ..scheduler import dry_run_plan
from ..strategy import AdaptiveStrategy, AllSingleCardStrategy, TaskRef

P, D, I = "parallel", "debate", "iterative"


class _Run:
    """Apply launch decisions and explicit completion events to exclusive cards."""

    def __init__(self, labels, cards=5, *, wish=2, caps=None, strategy=None):
        self.strategy = AdaptiveStrategy() if strategy is None else strategy
        self.cards = [
            Replica(f"cuda:{i}", "127.0.0.1", 19100 + i, "model")
            for i in range(cards)
        ]
        self.caps = {I: 1, D: 3, P: 3} if caps is None else caps
        self.wishes = {P: wish}
        self.pending = [TaskRef(str(i), label) for i, label in enumerate(labels)]
        self.free = list(self.cards)
        self.running = {}
        self.history = []

    def fill(self):
        while self.pending and self.free:
            batch = self.strategy.plan(
                self.pending, self.free, self.caps,
                total_cards=len(self.cards),
                running_labels={item.label for item in self.running.values()},
                k_wish_table=self.wishes,
            )
            if not batch:
                raise AssertionError("A legal adaptive run held free cards instead of launching")
            for item in batch:
                if item.task_id not in {task.task_id for task in self.pending}:
                    raise AssertionError("The strategy launched a non-pending task")
                if not item.replicas or len(set(item.replicas)) != item.k:
                    raise AssertionError("A launch must own distinct, non-empty replicas")
                if not set(item.replicas).issubset(self.free):
                    raise AssertionError("The strategy reused another running task's card")
                self.free = [card for card in self.free if card not in item.replicas]
                self.pending = [task for task in self.pending if task.task_id != item.task_id]
                self.running[item.task_id] = item
                self.history.append(item)
                self.assert_partition()

    def finish(self, task_id):
        self.free.extend(self.running.pop(task_id).replicas)
        self.assert_partition()

    def start_every_task(self):
        self.fill()
        while self.pending:
            self.finish(next(iter(self.running)))
            self.fill()

    def assert_partition(self):
        held = [card for item in self.running.values() for card in item.replicas]
        if len(held) != len(set(held)) or set(held).intersection(self.free):
            raise AssertionError("Running ownership overlaps")
        if len(held) + len(self.free) != len(self.cards):
            raise AssertionError("A replica was lost or released more than once")
        if set(held).union(self.free) != set(self.cards):
            raise AssertionError("An unknown replica entered the run")


class AdaptiveStrategyTests(unittest.TestCase):
    def test_initial_distribution_selects_one_branch_in_priority_order(self):
        cases = [
            ([P, D, I, I], 5, 2, 1),  # Also meets branch 2; branch 1 wins.
            ([P] + [I] * 4, 5, 2, 2),
            ([P] + [I] * 8, 5, 2, 2),  # wp+s equals 2K.
            ([P] + [I] * 9, 5, 2, 4),
            ([P] * 6 + [I] * 8, 5, 2, 3),  # wr+s equals 2K.
            ([P] * 6 + [I] * 9, 5, 2, 4),
            ([P] * 7, 5, 2, 3),
            ([P] * 8, 5, 2, 4),  # Tail P count needs more than K cards.
            ([P] * 5, 5, 2, 4),  # No partial P remainder.
            ([I] * 5, 5, 2, 4),
            ([P], 1, 2, 4),
            ([P, I], 2, 2, 2),
            ([P] + [I] * 4, 5, 1, 4),
            ([P] + [I] * 4, 5, 4, 4),  # Wish exceeds configured P cap.
        ]
        for labels, cards, wish, expected in cases:
            with self.subTest(labels=labels, cards=cards, wish=wish):
                run = _Run(labels, cards, wish=wish)
                run.fill()
                self.assertEqual(run.strategy.selected_policy, expected)

    def test_first_branch_preserves_fifo_and_starts_every_task_immediately(self):
        for labels, cards, expected in (
            ([D, P, I], 5, [1, 3, 1]),
            ([P, D, I], 5, [3, 1, 1]),
            ([D, P], 6, [3, 3]),
            ([P, P, P, I], 5, [2, 1, 1, 1]),
        ):
            with self.subTest(labels=labels, cards=cards):
                run = _Run(labels, cards)
                run.fill()
                self.assertEqual(run.strategy.selected_policy, 1)
                self.assertEqual([item.task_id for item in run.history],
                                 [str(i) for i in range(len(labels))])
                self.assertEqual([item.k for item in run.history], expected)
                self.assertEqual(run.pending, [])

    def test_second_branch_promotes_p_stably_and_preserves_mixed_s_order(self):
        run = _Run([D, I, P, D, P, I])
        run.start_every_task()
        self.assertEqual(run.strategy.selected_policy, 2)
        self.assertEqual([item.task_id for item in run.history],
                         ["2", "4", "0", "1", "3", "5"])
        self.assertEqual([item.k for item in run.history[:2]], [2, 2])

    def test_second_branch_scarce_debate_starts_single_without_holding_cards(self):
        run = _Run([P, D, I, I, I])
        run.fill()
        self.assertEqual(run.strategy.selected_policy, 2)
        self.assertEqual([(item.task_id, item.k) for item in run.history],
                         [("0", 2), ("1", 1), ("2", 1), ("3", 1)])
        self.assertEqual([task.task_id for task in run.pending], ["4"])

    def test_second_branch_late_debate_can_use_true_surplus(self):
        run = _Run([P, I, I, I, D])
        run.fill()
        self.assertEqual([task.task_id for task in run.pending], ["4"])
        run.finish("0")
        run.fill()
        self.assertEqual((run.history[-1].task_id, run.history[-1].k), ("4", 2))
        self.assertEqual(run.strategy.selected_policy, 2)

    def test_second_branch_debate_leaves_one_card_for_later_s(self):
        run = _Run([P, I, I, I, D, I])
        run.fill()
        run.finish("0")
        run.fill()
        self.assertEqual([(item.task_id, item.k) for item in run.history[-2:]],
                         [("4", 1), ("5", 1)])

    def test_third_branch_refills_each_free_card_before_all_p_finish(self):
        run = _Run([D, I] + [P] * 6)
        run.fill()
        self.assertEqual(run.strategy.selected_policy, 3)
        self.assertEqual([(item.task_id, item.k) for item in run.history],
                         [("2", 2), ("3", 1), ("4", 1), ("5", 1)])
        run.finish("3")
        run.fill()
        self.assertEqual((run.history[-1].task_id, run.history[-1].k), ("6", 1))
        run.finish("2")
        run.fill()
        self.assertEqual([(item.task_id, item.k) for item in run.history[-2:]],
                         [("7", 1), ("0", 1)])
        self.assertTrue(any(item.label == P for item in run.running.values()))
        self.assertEqual([task.task_id for task in run.pending], ["1"])

    def test_third_branch_only_first_remainder_p_get_multiple_cards(self):
        run = _Run([P] * 7 + [D])
        run.start_every_task()
        self.assertEqual(run.strategy.selected_policy, 3)
        self.assertEqual([item.k for item in run.history],
                         [2, 2, 1, 1, 1, 1, 1, 1])

    def test_configured_wish_is_used_in_both_wide_branches(self):
        for labels, cards, expected_branch in (
            ([P, P] + [I] * 4, 6, 2),
            ([P] * 9, 7, 3),
        ):
            with self.subTest(cards=cards, branch=expected_branch):
                run = _Run(labels, cards, wish=3, caps={I: 1, D: 2, P: 4})
                run.fill()
                self.assertEqual(run.strategy.selected_policy, expected_branch)
                self.assertEqual([item.k for item in run.history[:2]], [3, 3])

    def test_fallback_stays_fifo_single_after_the_queue_shrinks(self):
        run = _Run([I] * 5 + [D])
        run.fill()
        self.assertEqual(run.strategy.selected_policy, 4)
        run.finish("0")
        run.finish("1")
        run.fill()
        self.assertEqual((run.history[-1].task_id, run.history[-1].k), ("5", 1))
        self.assertEqual(len(run.free), 1)
        self.assertEqual(run.strategy.selected_policy, 4)

    def test_baseline_preserves_original_order_and_single_card_allocations(self):
        labels = [D, P, I, P, D, I]
        run = _Run(labels, strategy=AllSingleCardStrategy())
        run.start_every_task()
        self.assertEqual([item.task_id for item in run.history],
                         [str(i) for i in range(len(labels))])
        self.assertEqual([item.k for item in run.history], [1] * len(labels))

    def test_empty_queue_and_empty_free_pool_return_no_launch(self):
        strategy = AdaptiveStrategy()
        cards = _Run([]).cards
        self.assertEqual(
            strategy.plan([], cards, {I: 1, D: 3, P: 3}, total_cards=5,
                          running_labels=set(), k_wish_table={P: 2}),
            [],
        )
        self.assertIsNone(strategy.selected_policy)
        self.assertEqual(
            strategy.plan([TaskRef("p", P)], [], {I: 1, D: 3, P: 3},
                          total_cards=5, running_labels=set(), k_wish_table={P: 2}),
            [],
        )

    def test_dry_plan_resets_a_reused_strategy_between_experiments(self):
        strategy = AdaptiveStrategy()
        with (
            patch.object(cfg, "K_MAX_TABLE", {I: 1, D: 3, P: 3}),
            patch.object(cfg, "K_WISH_TABLE", {P: 2}),
        ):
            first = dry_run_plan(
                strategy=strategy, devices=[f"cuda:{i}" for i in range(5)],
                tasks=[TaskSpec("p", P, "agent", None, None)],
            )
            second = dry_run_plan(
                strategy=strategy, devices=[f"cuda:{i}" for i in range(5)],
                tasks=[TaskSpec(str(i), I, "agent", None, None) for i in range(5)],
            )
        self.assertEqual(first.adaptive_branch, 1)
        self.assertEqual(second.adaptive_branch, 4)
        self.assertEqual([item.k for item in second.assignments], [1] * 5)


if __name__ == "__main__":
    unittest.main()
