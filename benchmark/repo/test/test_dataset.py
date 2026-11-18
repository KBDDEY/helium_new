"""Queue construction checks for the server, without engines or graph execution.

Run from benchmark with PYTHONPATH=../src:. python -m unittest repo.test.test_dataset
Dataset reads and Helium graph construction are replaced by test doubles.
"""

from __future__ import annotations

import unittest
from unittest.mock import call, patch

from .. import config as cfg
from .. import dataset


class DatasetQueueTests(unittest.TestCase):
    def setUp(self):
        self.samples = {
            label: [object() for _ in range(8)]
            for label in ("iterative", "debate", "parallel")
        }
        self.build_calls = []
        self.generation_configs = []
        self.addCleanup(patch.stopall)
        self.install = patch.object(dataset, "_install_helium2_as_helium").start()
        self.loader = patch.object(
            dataset,
            "_load_samples",
            side_effect=lambda label, count: self.samples[label][:count],
        ).start()

        def generation_config():
            config = object()
            self.generation_configs.append(config)
            return config

        self.generation = patch.object(
            dataset, "_generation_config", side_effect=generation_config
        ).start()
        self.builders = {}
        for label in self.samples:
            self.builders[label] = patch.object(
                dataset, f"_build_{label}", side_effect=self.builder(label)
            ).start()

    def builder(self, label):
        def build(generation_config, sample_index=0, *, sample=None, task_id=None):
            self.build_calls.append((label, sample_index, sample, task_id))
            return dataset.TaskSpec(
                task_id=task_id,
                label=label,
                agent_name=f"agent-{task_id}",
                graph=object(),
                generation_config=generation_config,
                sample_index=sample_index,
            )

        return build

    def test_omitted_queue_uses_current_config(self):
        configured = [("parallel", 2), ("iterative", 1)]
        with patch.object(cfg, "TASK_QUEUE", configured):
            tasks = dataset.build_fixed_dataset()
        self.assertEqual(
            [(task.label, task.sample_index) for task in tasks], configured
        )
        self.assertEqual([task.task_id for task in tasks], ["parallel-2", "iterative-1"])

    def test_explicit_default_queue_keeps_original_task_ids(self):
        queue = [("iterative", 0), ("debate", 0), ("parallel", 0)]
        tasks = dataset.build_fixed_dataset(task_queue=queue)
        self.assertEqual(
            [task.task_id for task in tasks], ["iterative-0", "debate-0", "parallel-0"]
        )
        self.assertEqual([task.sample_index for task in tasks], [0, 0, 0])

    def test_indices_and_order_are_preserved_with_one_load_per_type(self):
        queue = [("parallel", 2), ("debate", 1), ("parallel", 0), ("iterative", 3)]
        tasks = dataset.build_fixed_dataset(task_queue=queue)
        self.assertEqual(
            [(task.label, task.sample_index) for task in tasks], queue
        )
        self.assertEqual(
            [task.task_id for task in tasks],
            ["parallel-2", "debate-1", "parallel-0", "iterative-3"],
        )
        self.loader.assert_has_calls(
            [call("parallel", 3), call("debate", 2), call("iterative", 4)],
            any_order=True,
        )
        self.assertEqual(self.loader.call_count, 3)
        self.assertEqual(
            [(label, index, sample) for label, index, sample, _ in self.build_calls],
            [(label, index, self.samples[label][index]) for label, index in queue],
        )

    def test_repeated_sample_gets_unique_task_and_execution_objects(self):
        queue = [("parallel", 0), ("debate", 0), ("parallel", 0), ("parallel", 0)]
        tasks = dataset.build_fixed_dataset(task_queue=queue)
        self.assertEqual(
            [task.task_id for task in tasks],
            ["parallel-0", "debate-0", "parallel-0-repeat1", "parallel-0-repeat2"],
        )
        self.assertEqual(len({id(task) for task in tasks}), len(tasks))
        self.assertEqual(len({id(task.graph) for task in tasks}), len(tasks))
        self.assertEqual(len({id(task.generation_config) for task in tasks}), len(tasks))
        self.assertEqual(self.builders["parallel"].call_count, 3)
        self.assertEqual(self.loader.call_count, 2)

    def test_explicit_empty_queue_reads_and_builds_nothing(self):
        self.assertEqual(dataset.build_fixed_dataset(task_queue=[]), [])
        self.loader.assert_not_called()
        self.generation.assert_not_called()
        for builder in self.builders.values():
            builder.assert_not_called()

    def test_invalid_entries_are_rejected_before_dataset_reads(self):
        invalid_queues = [
            [("unknown", 0)],
            [("parallel", -1)],
            [("parallel", True)],
            [("parallel", 1.5)],
            [("parallel", "1")],
            [("parallel",)],
            [("parallel", 0, "extra")],
            ["parallel"],
            [None],
            [(0, 0)],
            [("parallel", 0), ("debate", -1)],
        ]
        for queue in invalid_queues:
            with self.subTest(queue=queue):
                with self.assertRaises((ValueError, TypeError)):
                    dataset.build_fixed_dataset(task_queue=queue)
        self.loader.assert_not_called()
        for builder in self.builders.values():
            builder.assert_not_called()

    def test_unavailable_sample_error_identifies_type_and_index(self):
        self.loader.side_effect = lambda label, count: self.samples[label][:2]
        with self.assertRaises(ValueError) as error:
            dataset.build_fixed_dataset(task_queue=[("parallel", 4)])
        self.assertIn("parallel", str(error.exception))
        self.assertIn("4", str(error.exception))
        self.builders["parallel"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
