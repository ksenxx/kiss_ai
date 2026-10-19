# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: concurrent registry mutations never publish ``tabs_state``
snapshots out of order, a close never waits for an in-flight send, and a
sender stopped mid-send strands neither a deferred close nor a waiting
publication.

``AgentServer._broadcast_tabs_state`` takes the registry snapshot and
hands it to the printer as two steps.  Commands from different client
connections run concurrently on the executor, so two ``openTab``
commands could interleave as ``snapshot [X]`` / ``snapshot [X, Y]`` /
``broadcast [X, Y]`` / ``broadcast [X]``: every client then applied
the stale ``[X]`` last and closed the freshly opened ``Y`` although the
registry still held it (findings H2-F1).

The tests use a real :class:`JsonPrinter` subclass whose ``broadcast``
records every event and, for the FIRST ``tabs_state`` only, waits on an
event before returning; that stands in for the thread being descheduled
between the snapshot and the send.  A second thread mutates the
registry while the first is parked.  The last snapshot every client
saw must equal the registry, and the second thread must never wait for
the parked send (a close landing inside a run's publication must
return at once).

A registry mutation can run on a task thread, where a task Stop
injects ``KeyboardInterrupt`` at any bytecode.  The interruption tests
install a thread-local trace hook on the parked sender that raises
``KeyboardInterrupt`` once, right after the stale snapshot left the
printer, at every bookkeeping step of the protocol in turn.  Whatever
the step, the other thread's mutation must still reach the wire (a
deferred close by a helper thread, a waiting publication by itself),
and no stale sender may be left behind (review R5 I1, I2).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any
from unittest import TestCase

from kiss.server.json_printer import JsonPrinter
from kiss.server.server import AgentServer
from kiss.tests.server.conftest import redirect_persistence, restore_persistence

# (function, caller) pairs: the first call of ``function`` from ``caller``
# on the parked sender raises ``KeyboardInterrupt``.  Together they
# cover every bookkeeping step after the stale snapshot left the
# printer: before the send is recorded, before its waiters are woken,
# before the sender role is handed back on the normal exit, and before
# the ``finally`` hands it back.
INTERRUPT_POINTS = (
    ("_tabs_state_mark_sent", "locked_call"),
    ("notify_all", "_tabs_state_mark_sent"),
    ("notify_all", "_tabs_state_next"),
    ("_tabs_state_abandon", "locked_call"),
)


def _interrupt_at(function_name: str, caller_name: str) -> Callable[..., Any]:
    """Return a trace function raising ``KeyboardInterrupt`` at one call."""

    def trace(frame: FrameType, event: str, _arg: Any) -> Any:
        caller = frame.f_back
        if (
            event == "call"
            and frame.f_code.co_name == function_name
            and caller is not None
            and caller.f_code.co_name == caller_name
        ):
            raise KeyboardInterrupt("task stopped")
        return None

    return trace


class _GatedPrinter(JsonPrinter):
    """Records broadcasts; parks the first ``tabs_state`` sender until released."""

    def __init__(self) -> None:
        super().__init__()
        self.emitted: list[dict[str, Any]] = []
        self.first_entered = threading.Event()
        self.release_first = threading.Event()
        self._gated = False
        self._gate_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event*; the first ``tabs_state`` waits for ``release_first``.

        The gate has no timeout of its own short enough to matter: a
        caller that blocks on the parked send stays blocked until the
        test releases the gate, so the tests' own bounds decide.
        """
        if event.get("type") == "tabs_state":
            with self._gate_lock:
                first = not self._gated
                self._gated = True
            if first:
                self.first_entered.set()
                self.release_first.wait(timeout=120)
        self.emitted.append(event)


class TestTabsStateBroadcastOrder(TestCase):
    """The last ``tabs_state`` published always matches the registry."""

    def setUp(self) -> None:
        self.threads: list[threading.Thread] = []
        self._fresh_server()

    def tearDown(self) -> None:
        self._discard_server()

    def _fresh_server(self) -> None:
        """A gated printer and a server over an empty persisted registry."""
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-conc1018-tabs-"))
        self._saved = redirect_persistence(self.tmp)
        self.printer = _GatedPrinter()
        self.server = AgentServer(printer=self.printer)

    def _discard_server(self) -> None:
        self.printer.release_first.set()
        for thread in self.threads:
            thread.join(timeout=10)
        self.threads.clear()
        restore_persistence(self._saved)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _open(self, tab_id: str) -> None:
        self.server._cmd_open_tab(
            {
                "type": "openTab",
                "tabId": tab_id,
                "title": tab_id,
                "workDir": "",
            }
        )

    def _start(self, target: Callable[..., None], *args: Any, name: str) -> threading.Thread:
        thread = threading.Thread(target=target, args=args, name=name)
        self.threads.append(thread)
        thread.start()
        return thread

    def _open_and_park(self, tab_id: str) -> threading.Thread:
        """Open *tab_id* on a worker parked inside its ``tabs_state`` send."""
        worker = self._start(self._open, tab_id, name=f"open-{tab_id}")
        self.assertTrue(self.printer.first_entered.wait(timeout=10))
        return worker

    def _stopped_open(
        self, tab_id: str, hook: Callable[..., Any], stopped: threading.Event
    ) -> None:
        """Open *tab_id* with *hook* tracing this thread; record its Stop."""
        sys.settrace(hook)
        try:
            self._open(tab_id)
        except KeyboardInterrupt:
            stopped.set()
        finally:
            sys.settrace(None)

    def _wait_settled(self) -> None:
        """Wait until no sender is active and every request is on the wire."""
        with self.server._tabs_state_cond:
            for _ in range(200):
                if (
                    self.server._tabs_state_sender is None
                    and self.server._tabs_state_sent >= self.server._tabs_state_requested
                ):
                    return
                self.server._tabs_state_cond.wait(timeout=0.05)
        self.fail(
            f"publication never settled: sender={self.server._tabs_state_sender!r} "
            f"sent={self.server._tabs_state_sent} "
            f"requested={self.server._tabs_state_requested}"
        )

    def test_concurrent_open_tabs_publish_snapshots_in_registry_order(self) -> None:
        first = self._open_and_park("tab-x")
        # The registry already holds X; the second open lands while X's
        # snapshot is still unsent.
        second = self._start(self._open, "tab-y", name="open-y")
        second.join(timeout=0.5)
        self.printer.release_first.set()
        first.join(timeout=10)
        second.join(timeout=10)
        self.assertFalse(first.is_alive() or second.is_alive())

        self._assert_last_snapshot_is_the_registry(["tab-x", "tab-y"])

    def test_close_during_a_parked_publication_returns_and_is_republished(self) -> None:
        """A close never waits for an in-flight send; the sender re-sends it.

        The closer is bounded WHILE the gate is still closed, so an
        implementation that blocks the close on the parked send fails
        here instead of being rescued by a gate timeout.
        """
        first = self._open_and_park("tab-x")
        closer = self._start(self.server._close_tab, "tab-x", name="close-x")
        closer.join(timeout=5)
        try:
            self.assertFalse(closer.is_alive(), "the close waited for the parked send")
            self.assertFalse(self.server.tab_registry.has_tab("tab-x"))
            self.assertFalse(self.printer.release_first.is_set())
        finally:
            self.printer.release_first.set()
        first.join(timeout=10)
        self.assertFalse(first.is_alive())
        self._wait_settled()
        self._assert_last_snapshot_is_the_registry([])

    def test_stopped_sender_hands_a_deferred_close_to_a_helper(self) -> None:
        """R5 I1: a Stop right after the stale snapshot must not lose the close.

        The parked sender is stopped before it records its send; the
        closed tab's empty snapshot, deferred to that sender, must
        still reach the wire (by the helper thread) and no sender may
        linger.
        """
        for function_name, caller_name in INTERRUPT_POINTS:
            with self.subTest(interrupt=f"{caller_name} -> {function_name}"):
                self._reset()
                stopped = threading.Event()
                sender = self._start(
                    self._stopped_open,
                    "tab-x",
                    _interrupt_at(function_name, caller_name),
                    stopped,
                    name="open-x",
                )
                self.assertTrue(self.printer.first_entered.wait(timeout=10))
                closer = self._start(self.server._close_tab, "tab-x", name="close-x")
                closer.join(timeout=5)
                self.assertFalse(closer.is_alive(), "the close waited for the parked send")
                self.printer.release_first.set()
                sender.join(timeout=10)
                self.assertFalse(sender.is_alive())
                self.assertTrue(stopped.is_set(), "the trace hook never fired")
                self._wait_settled()
                self._assert_last_snapshot_is_the_registry([])

    def test_stopped_sender_never_strands_a_waiting_publication(self) -> None:
        """R5 I2: a Stop in the sender's bookkeeping must not wedge a publisher.

        A second publication deferred to the parked sender waits for
        its generation.  Whichever bookkeeping step the sender's Stop
        lands on, the publisher must return with its tab on the wire,
        and no stale sender may remain for later callers to wait on.
        """
        for function_name, caller_name in INTERRUPT_POINTS:
            with self.subTest(interrupt=f"{caller_name} -> {function_name}"):
                self._reset()
                stopped = threading.Event()
                sender = self._start(
                    self._stopped_open,
                    "tab-x",
                    _interrupt_at(function_name, caller_name),
                    stopped,
                    name="open-x",
                )
                self.assertTrue(self.printer.first_entered.wait(timeout=10))
                publisher = self._start(self._open, "tab-y", name="open-y")
                publisher.join(timeout=0.5)
                self.assertTrue(publisher.is_alive(), "the publisher did not wait")
                self.printer.release_first.set()
                sender.join(timeout=10)
                self.assertFalse(sender.is_alive())
                self.assertTrue(stopped.is_set(), "the trace hook never fired")
                publisher.join(timeout=10)
                self.assertFalse(publisher.is_alive(), "the publisher is stranded")
                self._wait_settled()
                self._assert_last_snapshot_is_the_registry(["tab-x", "tab-y"])

    def _reset(self) -> None:
        """Fresh printer, server and persisted registry for the next sub-test."""
        self._discard_server()
        self._fresh_server()

    def _assert_last_snapshot_is_the_registry(self, expected: list[str]) -> None:
        snapshots = [e for e in self.printer.emitted if e["type"] == "tabs_state"]
        registry_ids = [row["tabId"] for row in self.server.tab_registry.snapshot()]
        self.assertEqual(registry_ids, expected)
        self.assertEqual(
            [row["tabId"] for row in snapshots[-1]["tabs"]],
            registry_ids,
            "the last published snapshot must be the registry's current state",
        )
