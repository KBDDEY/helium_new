"""No-engine tail-allocation checks, intended to run on the server.

Run with PYTHONPATH=src:. python -m unittest benchmark.repo.test_tail_acceleration
No vLLM services or Helium execution are started by these tests.
"""

from __future__ import annotations

import multiprocessing
import time
import unittest
from concurrent.futures import Future
from unittest.mock import patch

from helium_new2.runtime.tail_resources import TailResourceState

from . import config as cfg
from . import runner
from .dataset import TaskSpec
from .pool import Replica
from .runner import HeliumPorts
from .scheduler import _InFlight, _accelerate_tail
from .strategy import Assignment, TaskRef


class _Pool:
    """Tracks exclusive ownership with the same acquire/release contract."""

    def __init__(self, replicas, held=()):
        self.replicas = list(replicas)
        self._free = [replica for replica in replicas if replica not in held]

    @property
    def free_replicas(self):
        return list(self._free)

    def acquire(self, count):
        if not 0 < count <= len(self._free):
            raise RuntimeError("Invalid reservation")
        taken, self._free = self._free[:count], self._free[count:]
        return taken

    def release(self, replicas):
        for replica in replicas:
            if replica not in self.replicas or replica in self._free:
                raise RuntimeError("Invalid release")
            self._free.append(replica)


class _State:
    def __init__(self, task_id, started_at, grant_order, *, accept=True, running=True):
        self.task_id = task_id
        self.started_at = started_at
        self.grant_order = grant_order
        self.accept = accept
        self.running = running
        self.authorized = []

    def snapshot(self):
        return {"ready": True, "running": self.running, "started_at": self.started_at}

    def grant(self, urls):
        if not self.accept:
            return False
        self.grant_order.append(self.task_id)
        self.authorized.extend(urls)
        return True


def _fork_observe_grant(state, home_url, connection):
    """The grant happens after this process has inherited the state."""
    try:
        connection.send(state.snapshot()["authorized_urls"])
        connection.recv()
        index, endpoint = state.begin_request(home_url)
        try:
            connection.send(endpoint[0])
        finally:
            state.end_request(index)
        connection.send(state.snapshot())
    finally:
        connection.close()


class TailAllocationTests(unittest.TestCase):
    def setUp(self):
        self.cards = [
            Replica(f"cuda:{i}", "127.0.0.1", 19100 + i, "model")
            for i in range(5)
        ]
        self.grant_order = []
        self.addCleanup(patch.stopall)
        patch.object(cfg, "K_MAX_TABLE", {"parallel": 3}).start()
        patch.object(cfg, "K_WISH_TABLE", {"parallel": 2}).start()

    def item(self, task_id, cards, started_at, **state_kwargs):
        state = _State(task_id, started_at, self.grant_order, **state_kwargs)
        future = Future()
        return _InFlight(
            assignment=Assignment(task_id, "parallel", tuple(cards)),
            task=TaskSpec(task_id, "parallel", "agent", None, None),
            replicas=list(cards),
            helium_ports=HeliumPorts(20000, 22001),
            future=future,
            wall_start_seconds=0.0,
            tail_resources=state,
        )

    def accelerate(self, pool, items, queue=()):
        _accelerate_tail(
            queue=queue,
            pool=pool,
            in_flight={item.future: item for item in items},
            arrival=time.perf_counter(),
        )

    def test_pending_task_prevents_tail_allocation(self):
        item = self.item("p", self.cards[:1], 10.0)
        pool = _Pool(self.cards, held=item.replicas)
        before = pool.free_replicas
        self.accelerate(pool, [item], [TaskRef("pending", "iterative")])
        self.assertEqual(pool.free_replicas, before)
        self.assertEqual(len(item.replicas), 1)
        self.assertEqual(self.grant_order, [])

    def test_wish_is_covered_for_every_p_before_any_reaches_cap(self):
        early = self.item("early", self.cards[:1], 10.0)
        late = self.item("late", self.cards[1:2], 20.0)
        pool = _Pool(self.cards, held=early.replicas + late.replicas)
        self.accelerate(pool, [early, late])
        self.assertEqual(self.grant_order, ["late", "early", "late"])
        self.assertEqual((len(early.replicas), len(late.replicas)), (2, 3))
        self.assertEqual(pool.free_replicas, [])

    def test_actual_start_time_has_priority_over_dispatch_order(self):
        dispatched_first = self.item("started_late", self.cards[:1], 20.0)
        dispatched_second = self.item("started_early", self.cards[1:2], 10.0)
        pool = _Pool(self.cards[:3], held=self.cards[:2])
        self.accelerate(pool, [dispatched_first, dispatched_second])
        self.assertEqual(len(dispatched_first.replicas), 2)
        self.assertEqual(len(dispatched_second.replicas), 1)
        self.assertEqual(self.grant_order, ["started_late"])

    def test_rejected_grant_returns_the_reserved_cards(self):
        item = self.item("p", self.cards[:1], 10.0, accept=False)
        pool = _Pool(self.cards, held=item.replicas)
        before = set(pool.free_replicas)
        self.accelerate(pool, [item])
        self.assertEqual(set(pool.free_replicas), before)
        self.assertEqual(item.replicas, self.cards[:1])
        self.assertEqual(item.card_history, [])

    def test_completed_or_finishing_tasks_receive_no_grants(self):
        completed = self.item("completed", self.cards[:1], 10.0)
        completed.future.set_result(1.0)
        finishing = self.item("finishing", self.cards[1:2], 20.0, running=False)
        pool = _Pool(self.cards, held=self.cards[:2])
        self.accelerate(pool, [completed, finishing])
        self.assertEqual(len(pool.free_replicas), 3)
        self.assertEqual(self.grant_order, [])

    def test_full_current_allocation_is_retained_for_completion_release(self):
        item = self.item("p", self.cards[:1], 10.0)
        pool = _Pool(self.cards, held=item.replicas)
        self.accelerate(pool, [item])
        self.assertEqual(item.assignment.k, 1)
        self.assertEqual(len(item.replicas), 3)
        self.assertEqual(len(set(item.replicas)), 3)
        self.assertEqual(len(pool.free_replicas), 2)
        # The scheduler completion path releases this updated ownership list.
        pool.release(item.replicas)
        self.assertEqual(set(pool.free_replicas), set(self.cards))
        self.assertEqual(len(pool.free_replicas), len(self.cards))


class TailResourceAccountingTests(unittest.TestCase):
    def test_logical_worker_keeps_its_home_and_only_borrows_added_cards(self):
        home = "http://127.0.0.1:19100/v1"
        other_initial = "http://127.0.0.1:19101/v1"
        added = "http://127.0.0.1:19102/v1"
        state = TailResourceState(
            [(url, "model", "EMPTY") for url in (home, other_initial, added)],
            context=multiprocessing.get_context(),
        )
        state.initialize([home, other_initial])
        state.mark_started()
        reservations = []
        try:
            for _ in range(2):
                index, endpoint = state.begin_request(home)
                reservations.append(index)
                self.assertEqual(endpoint[0], home)
            self.assertTrue(state.grant([added]))
            index, endpoint = state.begin_request(home)
            reservations.append(index)
            self.assertEqual(endpoint[0], added)
            self.assertEqual(state.snapshot()["request_counts"][other_initial], 0)
        finally:
            for index in reservations:
                state.end_request(index)
            state.mark_finished()
            state.seal()
        self.assertTrue(all(count == 0 for count in state.snapshot()["in_flight"].values()))

    def test_equal_load_keeps_home_cache_affinity(self):
        home = "http://127.0.0.1:19100/v1"
        added = "http://127.0.0.1:19101/v1"
        state = TailResourceState(
            [(home, "model", "EMPTY"), (added, "model", "EMPTY")],
            context=multiprocessing.get_context(),
        )
        state.initialize([home])
        state.mark_started()
        self.assertTrue(state.grant([added]))
        try:
            for _ in range(2):
                index, endpoint = state.begin_request(home)
                try:
                    self.assertEqual(endpoint[0], home)
                finally:
                    state.end_request(index)
        finally:
            state.mark_finished()
            state.seal()
        self.assertEqual(state.snapshot()["request_counts"], {home: 2, added: 0})


@unittest.skipUnless(
    "fork" in multiprocessing.get_all_start_methods(),
    "The experiment harness requires the server's fork context",
)
class TailSharedStateTests(unittest.TestCase):
    def test_forked_sender_observes_later_grant_and_returns_request_counts(self):
        context = multiprocessing.get_context("fork")
        home = "http://127.0.0.1:19100/v1"
        added = "http://127.0.0.1:19101/v1"
        state = TailResourceState(
            [(home, "model", "EMPTY"), (added, "model", "EMPTY")],
            context=context,
        )
        state.initialize([home])
        state.mark_started()
        home_index, _ = state.begin_request(home)
        parent, child = context.Pipe()
        process = context.Process(target=_fork_observe_grant, args=(state, home, child))
        try:
            process.start()
            child.close()
            self.assertTrue(parent.poll(5.0), "Sender did not report inherited state")
            self.assertEqual(parent.recv(), (home,))
            self.assertTrue(state.grant([added]))
            parent.send("continue")
            self.assertTrue(parent.poll(5.0), "Sender did not see the grant")
            self.assertEqual(parent.recv(), added)
            self.assertTrue(parent.poll(5.0), "Sender did not finish accounting")
            observed = parent.recv()
            self.assertEqual(observed["request_counts"], {home: 1, added: 1})
            self.assertEqual(observed["in_flight"], {home: 1, added: 0})
            process.join(5.0)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(5.0)
            parent.close()
            child.close()
            state.end_request(home_index)
            state.mark_finished()
            state.seal()
        self.assertEqual(state.snapshot()["in_flight"], {home: 0, added: 0})
        self.assertFalse(state.grant([added]))
        with self.assertRaises(RuntimeError):
            state.begin_request(home)


class TailRunnerCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def run_with_close_failure(self, enabled):
        events = []

        class State:
            def mark_started(self):
                events.append("started")

            def mark_finished(self):
                events.append("finished")

            def seal(self):
                events.append("sealed")

        class Server:
            is_started = False

            def start(self):
                self.is_started = True

            async def execute(self, *_):
                events.append("execute")

            def close(self):
                events.append("close")
                raise RuntimeError("cleanup failed")

        server = Server()
        patches = [
            patch.dict(runner.os.environ, {}),
            patch.object(runner, "_apply_helium_ports", return_value="unused"),
            patch.object(runner, "_sync_envs_ports"),
            patch.object(runner, "_server_config", return_value=None),
            patch.object(runner, "_request_config", return_value=None),
            patch.object(runner, "_helium_server_cls", return_value=lambda _: server),
        ]
        for item in patches:
            item.start()
        try:
            try:
                await runner.run_task(
                    TaskSpec("p", "parallel", "agent", None, None),
                    [Replica("cuda:0", "127.0.0.1", 19100, "model")],
                    helium_ports=HeliumPorts(20000, 22001),
                    tail_resources=State() if enabled else None,
                )
            except RuntimeError as exc:
                return events, exc
            return events, None
        finally:
            for item in reversed(patches):
                item.stop()

    async def test_tail_cleanup_failure_propagates_and_seals_routing(self):
        events, error = await self.run_with_close_failure(enabled=True)
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(str(error), "cleanup failed")
        self.assertEqual(events, ["started", "execute", "finished", "close", "sealed"])

    async def test_disabled_feature_retains_existing_cleanup_tolerance(self):
        events, error = await self.run_with_close_failure(enabled=False)
        self.assertIsNone(error)
        self.assertEqual(events, ["execute", "close"])


if __name__ == "__main__":
    unittest.main()
