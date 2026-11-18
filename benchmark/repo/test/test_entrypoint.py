"""Server-side checks for the config-selected experiment/control entry point.

Run from benchmark with PYTHONPATH=../src:. python -m unittest repo.test.test_entrypoint
Execution is mocked; dry-run checks use prebuilt task stubs and start no engines.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import Mock, patch

from .. import __main__ as entrypoint
from .. import config as cfg
from .. import scheduler
from ..dataset import TaskSpec
from ..strategy import AdaptiveStrategy


class ConfiguredEntryPointTests(unittest.TestCase):
    def test_run_forwards_each_configured_flavor(self):
        for flavor in ("helium2", "helium_new2"):
            with self.subTest(flavor=flavor):
                report = Mock()
                report.lines.return_value = [f"# report flavor={flavor}"]
                with (
                    patch.object(cfg, "RUN_FLAVOR", flavor),
                    patch.object(entrypoint, "_ensure_paths"),
                    patch.object(scheduler, "run_experiment", return_value=report) as run,
                    patch.object(scheduler, "dry_run_plan") as dry,
                    patch("builtins.print"),
                ):
                    entrypoint.main([])
                run.assert_called_once_with(flavor=flavor)
                dry.assert_not_called()
                report.lines.assert_called_once_with()

    def test_dry_run_forwards_each_configured_flavor(self):
        for flavor in ("helium2", "helium_new2"):
            with self.subTest(flavor=flavor):
                plan = Mock()
                plan.lines.return_value = ["# schedule_plan"]
                with (
                    patch.object(cfg, "RUN_FLAVOR", flavor),
                    patch.object(entrypoint, "_ensure_paths"),
                    patch.object(scheduler, "run_experiment") as run,
                    patch.object(scheduler, "dry_run_plan", return_value=plan) as dry,
                    patch("builtins.print"),
                ):
                    entrypoint.main(["--dry-run"])
                dry.assert_called_once_with(flavor=flavor)
                run.assert_not_called()
                plan.lines.assert_called_once_with()

    def test_invalid_config_is_rejected_before_either_execution_path(self):
        for argv in ([], ["--dry-run"]):
            with self.subTest(argv=argv):
                error_output = io.StringIO()
                with (
                    patch.object(cfg, "RUN_FLAVOR", "helium_typo"),
                    patch.object(entrypoint, "_ensure_paths"),
                    patch.object(scheduler, "run_experiment") as run,
                    patch.object(scheduler, "dry_run_plan") as dry,
                    redirect_stderr(error_output),
                ):
                    with self.assertRaises(SystemExit) as error:
                        entrypoint.main(argv)
                self.assertEqual(error.exception.code, 2)
                self.assertIn("RUN_FLAVOR", error_output.getvalue())
                self.assertIn("helium2", error_output.getvalue())
                self.assertIn("helium_new2", error_output.getvalue())
                run.assert_not_called()
                dry.assert_not_called()


class DryRunFlavorTests(unittest.TestCase):
    def test_dry_run_uses_the_same_group_policy_as_real_execution(self):
        tasks = [TaskSpec("p", "parallel", "agent", None, None)]
        devices = ["cuda:0", "cuda:1", "cuda:2"]
        cases = (
            ("helium2", "AllSingleCardStrategy", None, 1),
            ("helium_new2", "AdaptiveStrategy", 1, 3),
        )
        with (
            patch.object(cfg, "K_MAX_TABLE", {"iterative": 1, "debate": 3, "parallel": 3}),
            patch.object(cfg, "K_WISH_TABLE", {"parallel": 2}),
            patch.object(scheduler, "build_fixed_dataset") as build,
            patch.object(scheduler, "VLLMPool") as engines,
        ):
            for flavor, strategy_name, branch, cards in cases:
                with self.subTest(flavor=flavor):
                    plan = scheduler.dry_run_plan(
                        flavor=flavor, devices=devices, tasks=tasks
                    )
                    self.assertEqual(plan.strategy_name, strategy_name)
                    self.assertEqual(plan.adaptive_branch, branch)
                    self.assertEqual([(a.task_id, a.k) for a in plan.assignments], [("p", cards)])
            build.assert_not_called()
            engines.assert_not_called()

    def test_control_dry_run_rejects_an_adaptive_override(self):
        with patch.object(scheduler, "build_fixed_dataset") as build:
            with self.assertRaisesRegex(ValueError, "helium2"):
                scheduler.dry_run_plan(flavor="helium2", strategy=AdaptiveStrategy())
            build.assert_not_called()

    def test_invalid_dry_run_flavor_is_rejected_before_dataset_loading(self):
        with patch.object(scheduler, "build_fixed_dataset") as build:
            with self.assertRaisesRegex(ValueError, "flavor"):
                scheduler.dry_run_plan(flavor="helium_typo")
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
