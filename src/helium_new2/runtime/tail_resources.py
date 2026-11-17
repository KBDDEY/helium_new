"""Shared task-local grants for tail acceleration (no Helium/LLM imports).

Create every task's state before the fork executor starts.  The scheduler,
Helium task process and its HTTP client processes then inherit the same shared
memory.  Only the scheduler grants replicas; request clients only account for
their use.  This module never starts, stops or reconfigures an engine.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from multiprocessing import get_context
from multiprocessing.context import BaseContext
from typing import Any


class TailResourceState:
    def __init__(
        self,
        endpoints: Sequence[tuple[str, str, str]],
        context: BaseContext | None = None,
    ) -> None:
        context = get_context("fork") if context is None else context
        self.endpoints = tuple(
            (url.rstrip("/"), model, api_key) for url, model, api_key in endpoints
        )
        if not self.endpoints:
            raise ValueError("Tail resources require at least one endpoint")
        self._indices = {endpoint[0]: i for i, endpoint in enumerate(self.endpoints)}
        if len(self._indices) != len(self.endpoints):
            raise ValueError("Tail resource endpoints must be unique")
        size = len(self.endpoints)
        self._lock = context.RLock()
        self._initial = context.RawArray("b", size)
        self._authorized = context.RawArray("b", size)
        self._in_flight = context.RawArray("q", size)
        self._request_counts = context.RawArray("q", size)
        self._ready = context.RawValue("b", False)
        self._running = context.RawValue("b", False)
        self._closed = context.RawValue("b", False)
        self._started_at = context.RawValue("d", 0.0)

    def _url_indices(self, urls: Sequence[str]) -> list[int]:
        try:
            return [self._indices[url.rstrip("/")] for url in urls]
        except KeyError as exc:
            raise ValueError(f"Unknown tail resource endpoint: {exc.args[0]}") from exc

    def initialize(self, initial_urls: Sequence[str]) -> None:
        """Set the initial allocation once, before this task is submitted."""
        indices = self._url_indices(initial_urls)
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("Initial endpoints must be non-empty and unique")
        with self._lock:
            if any(self._initial) or self._ready.value:
                raise RuntimeError("Tail task state has already been initialized")
            for i in indices:
                self._initial[i] = self._authorized[i] = True

    def mark_started(self) -> None:
        """Called after Helium is ready, immediately before graph execution."""
        with self._lock:
            if not any(self._initial):
                raise RuntimeError("Tail task state was not initialized")
            if self._closed.value:
                raise RuntimeError("Tail task state has already been sealed")
            if self._ready.value:
                raise RuntimeError("Tail task state has already been started")
            self._started_at.value = time.perf_counter()
            self._ready.value = self._running.value = True

    def mark_finished(self) -> None:
        """Stop accepting grants; ownership lasts until the task is cleaned up."""
        with self._lock:
            self._running.value = False

    def seal(self) -> None:
        """Close the final-send boundary before the scheduler reclaims cards.

        Normal close may first drain requests after mark_finished().  Once
        sealed, clients cannot reserve any new request, including events that
        were queued but not yet sent.  Existing requests can still decrement
        their counters.  The scheduler must also verify zero in-flight counts
        before releasing this task's replicas for reuse.
        """
        with self._lock:
            self._closed.value = True
            self._running.value = False

    def grant(self, base_urls: Sequence[str]) -> bool:
        """Atomically authorize additional replicas, if execution still runs.

        The caller must first reserve these replicas in the scheduler's pool.
        A rejected grant leaves this state unchanged so that reservation can be
        returned immediately.  Accepted grants are never withdrawn mid-task.
        """
        indices = self._url_indices(base_urls)
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("Granted endpoints must be non-empty and unique")
        with self._lock:
            if self._closed.value or not self._ready.value or not self._running.value:
                return False
            if any(self._authorized[i] for i in indices):
                raise ValueError("Tail endpoint is already authorized")
            initial_model = next(
                self.endpoints[i][1] for i, initial in enumerate(self._initial) if initial
            )
            if any(self.endpoints[i][1] != initial_model for i in indices):
                raise ValueError("Tail endpoints must serve the task's original model")
            for i in indices:
                self._authorized[i] = True
            return True

    def begin_request(self, home_url: str) -> tuple[int, tuple[str, str, str]]:
        """Choose home or a less busy added replica just before sending HTTP.

        Each logical worker retains its own initial home.  It can overflow only
        to newly granted replicas, never to another initial worker's replica.
        Home wins ties to retain the original placement/cache affinity wherever
        there is no observed request-count advantage to moving the call.
        Counts are outstanding HTTP requests, not a prediction of execution time
        or of actual prefix-cache contents.
        """
        home = self._url_indices([home_url])[0]
        with self._lock:
            if self._closed.value:
                raise RuntimeError("Tail task request routing has been sealed")
            if not self._initial[home] or not self._authorized[home]:
                raise RuntimeError("The logical worker's home is not authorized")
            # mark_finished only closes the grant window.  Already authorized
            # replicas remain owned until runner cleanup returns, including
            # requests being drained during cancellation/close.
            model = self.endpoints[home][1]
            candidates = [home] + [
                i
                for i, authorized in enumerate(self._authorized)
                if authorized and not self._initial[i] and self.endpoints[i][1] == model
            ]
            chosen = min(candidates, key=lambda i: self._in_flight[i])
            self._in_flight[chosen] += 1
            self._request_counts[chosen] += 1
            return chosen, self.endpoints[chosen]

    def end_request(self, index: int) -> None:
        with self._lock:
            if self._in_flight[index] <= 0:
                raise RuntimeError("Tail request accounting underflow")
            self._in_flight[index] -= 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ready": bool(self._ready.value),
                "running": bool(self._running.value),
                "closed": bool(self._closed.value),
                "started_at": self._started_at.value,
                "initial_urls": tuple(
                    endpoint[0]
                    for i, endpoint in enumerate(self.endpoints)
                    if self._initial[i]
                ),
                "authorized_urls": tuple(
                    endpoint[0]
                    for i, endpoint in enumerate(self.endpoints)
                    if self._authorized[i]
                ),
                "in_flight": {
                    endpoint[0]: self._in_flight[i]
                    for i, endpoint in enumerate(self.endpoints)
                },
                "request_counts": {
                    endpoint[0]: self._request_counts[i]
                    for i, endpoint in enumerate(self.endpoints)
                },
            }
