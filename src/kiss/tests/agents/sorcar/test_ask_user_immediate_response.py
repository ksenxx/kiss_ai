# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests guarding the immediacy of ``ask_user_question``.

Two complementary bugs delayed (or dropped) the user's answer reaching
the agent thread after the user clicked "Submit":

1. ``_TaskRunnerMixin._await_user_response`` polled the per-tab
   ``user_answer_queue`` with ``q.get(timeout=0.5)``.  Even though
   ``Queue.put`` notifies a waiting consumer immediately, the loop's
   stop-event branch meant any failure to notify (e.g. the answer
   was routed to a different queue) gave a worst-case half-second
   wait per iteration and an unbounded total wait before the stop
   event eventually fires.

2. ``_CommandsMixin._cmd_user_answer`` only looked up the queue under
   the answer's exact ``tabId``.  When the askUser broadcast was
   fan-stamped per subscriber, a viewer tab that submitted the
   answer carried its own tab id — but only the task-owner tab held
   a live ``user_answer_queue``.  The answer was silently dropped
   and the agent thread waited until the task was eventually
   cancelled, surfacing as "the user's response is sometimes not
   responded to immediately".

These tests exercise the post-fix behaviour: a put always wakes the
waiter promptly, and a userAnswer from a co-subscriber tab still
reaches the owner's queue.
"""

from __future__ import annotations

import queue
import threading
import time
import unittest

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.server import VSCodeServer


class _StubPrinter:
    """Minimal printer stand-in exposing the surface the SUT uses."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[str, set[str]] = {}
        self._thread_local = threading.local()


class TestAwaitUserResponseImmediacy(unittest.TestCase):
    """``_await_user_response`` returns the answer with minimal latency."""

    def setUp(self) -> None:
        self.server = VSCodeServer()

    def tearDown(self) -> None:
        with _RunningAgentState._registry_lock:
            _RunningAgentState.running_agent_states.clear()

    def test_get_returns_within_50ms_of_put(self) -> None:
        """A ``put`` from another thread wakes the waiter promptly.

        The pre-fix polling loop with ``timeout=0.5`` exposed a
        worst-case 0.5 s delay when the put narrowly missed the
        ``q.get`` window.  The post-fix blocking ``q.get`` returns
        in the order of milliseconds.
        """
        q: queue.Queue[str] = queue.Queue(maxsize=1)
        self.server._get_tab("tab-A").user_answer_queue = q
        self.server.printer.subscribe_tab("task-A", "tab-A")
        stop = threading.Event()

        delivered = threading.Event()
        result_holder: dict[str, str] = {}
        entered = threading.Event()

        def wait_for_answer() -> None:
            # ``_thread_local`` state must be set on the waiter
            # thread itself — that is the thread that reads it.
            self.server.printer._thread_local.task_id = "task-A"
            self.server.printer._thread_local.stop_event = stop
            entered.set()
            result_holder["answer"] = self.server._await_user_response()
            delivered.set()

        waiter = threading.Thread(target=wait_for_answer, daemon=True)
        waiter.start()
        entered.wait(timeout=1.0)
        # Let the waiter enter ``q.get``.
        time.sleep(0.05)

        t0 = time.monotonic()
        q.put_nowait("the answer")
        delivered.wait(timeout=1.0)
        latency = time.monotonic() - t0

        waiter.join(timeout=1.0)
        stop.set()
        self.assertEqual(result_holder.get("answer"), "the answer")
        self.assertLess(
            latency, 0.05,
            f"answer delivery took {latency*1000:.1f}ms; "
            "expected near-instant wake-up after put",
        )

    def test_stop_event_unblocks_waiter(self) -> None:
        """Setting the stop event wakes the waiter via the sentinel."""
        q: queue.Queue[str] = queue.Queue(maxsize=1)
        self.server._get_tab("tab-B").user_answer_queue = q
        self.server.printer.subscribe_tab("task-B", "tab-B")
        stop = threading.Event()

        raised = threading.Event()
        entered = threading.Event()

        def wait_for_answer() -> None:
            self.server.printer._thread_local.task_id = "task-B"
            self.server.printer._thread_local.stop_event = stop
            entered.set()
            try:
                self.server._await_user_response()
            except KeyboardInterrupt:
                raised.set()

        waiter = threading.Thread(target=wait_for_answer, daemon=True)
        waiter.start()
        entered.wait(timeout=1.0)
        time.sleep(0.05)
        stop.set()
        self.assertTrue(raised.wait(timeout=1.0))
        waiter.join(timeout=1.0)


class TestEndToEndAskUserLatency(unittest.TestCase):
    """End-to-end latency: from ``_handle_command(userAnswer)`` to waiter return.

    The pre-fix combined latency could exceed 0.5 s in the worst case
    (the polling-loop iteration window plus the queue routing miss).
    Post-fix it is bounded by lock contention and ``queue.Queue``
    wake-up overhead — well under 50 ms on a quiet machine.
    """

    def setUp(self) -> None:
        self.server = VSCodeServer()

    def tearDown(self) -> None:
        with _RunningAgentState._registry_lock:
            _RunningAgentState.running_agent_states.clear()

    def test_handle_command_user_answer_unblocks_await_within_50ms(self) -> None:
        """The full dispatch path delivers the answer in <50 ms.

        Reproduces the user-visible scenario end-to-end: an agent
        thread is blocked inside ``_await_user_response``; the
        websocket layer's ``userAnswer`` command flows through
        ``VSCodeServer._handle_command`` (the same entry point used
        by ``RemoteAccessServer._run_cmd`` for both WSS and UDS
        peers); the waiter must wake immediately.
        """
        q: queue.Queue[str] = queue.Queue(maxsize=1)
        self.server._get_tab("tab-E").user_answer_queue = q
        self.server.printer.subscribe_tab("task-E", "tab-E")
        stop = threading.Event()

        delivered = threading.Event()
        result_holder: dict[str, str] = {}
        entered = threading.Event()

        def wait_for_answer() -> None:
            self.server.printer._thread_local.task_id = "task-E"
            self.server.printer._thread_local.stop_event = stop
            entered.set()
            result_holder["answer"] = self.server._await_user_response()
            delivered.set()

        waiter = threading.Thread(target=wait_for_answer, daemon=True)
        waiter.start()
        entered.wait(timeout=1.0)
        # Let the waiter enter ``q.get``.
        time.sleep(0.05)

        t0 = time.monotonic()
        # Go through the public command-dispatch entry point so the
        # test exercises the same code path the WS / UDS handlers
        # use when forwarding a ``userAnswer`` from the frontend.
        self.server._handle_command(
            {"type": "userAnswer", "tabId": "tab-E", "answer": "hello"},
        )
        delivered.wait(timeout=1.0)
        latency = time.monotonic() - t0

        waiter.join(timeout=1.0)
        stop.set()
        self.assertEqual(result_holder.get("answer"), "hello")
        self.assertLess(
            latency, 0.05,
            f"end-to-end userAnswer→waiter took {latency*1000:.1f}ms; "
            "expected <50ms",
        )

    def test_handle_command_viewer_user_answer_unblocks_within_50ms(self) -> None:
        """Viewer-tab answer reaches owner waiter end-to-end in <50 ms.

        Multi-viewer case: the viewer tab carries no
        ``user_answer_queue``; the owner does.  The full dispatch
        path must still resolve the queue via the subscriber
        fallback and wake the waiter immediately.
        """
        owner_q: queue.Queue[str] = queue.Queue(maxsize=1)
        self.server._get_tab("owner-V").user_answer_queue = owner_q
        self.server._get_tab("viewer-V")  # no queue
        self.server.printer.subscribe_tab("task-V", "owner-V")
        self.server.printer.subscribe_tab("task-V", "viewer-V")
        stop = threading.Event()

        delivered = threading.Event()
        result_holder: dict[str, str] = {}
        entered = threading.Event()

        def wait_for_answer() -> None:
            self.server.printer._thread_local.task_id = "task-V"
            self.server.printer._thread_local.stop_event = stop
            entered.set()
            result_holder["answer"] = self.server._await_user_response()
            delivered.set()

        waiter = threading.Thread(target=wait_for_answer, daemon=True)
        waiter.start()
        entered.wait(timeout=1.0)
        time.sleep(0.05)

        t0 = time.monotonic()
        self.server._handle_command(
            {"type": "userAnswer", "tabId": "viewer-V", "answer": "from-viewer"},
        )
        delivered.wait(timeout=1.0)
        latency = time.monotonic() - t0

        waiter.join(timeout=1.0)
        stop.set()
        self.assertEqual(result_holder.get("answer"), "from-viewer")
        self.assertLess(
            latency, 0.05,
            f"viewer-tab end-to-end userAnswer→waiter took "
            f"{latency*1000:.1f}ms; expected <50ms",
        )


class TestUserAnswerSubscriberFallback(unittest.TestCase):
    """``_cmd_user_answer`` routes to the owner queue via subscriber graph."""

    def setUp(self) -> None:
        self.server = VSCodeServer()

    def tearDown(self) -> None:
        with _RunningAgentState._registry_lock:
            _RunningAgentState.running_agent_states.clear()

    def test_viewer_tab_answer_reaches_owner_queue(self) -> None:
        """An answer carrying a viewer tab id reaches the owner's queue.

        Owner tab ``owner`` runs the task and holds the queue.  Viewer
        tab ``viewer`` is subscribed to the same task — the askUser
        broadcast was stamped with ``viewer`` and the frontend posts
        the userAnswer back with ``tabId="viewer"``.  Pre-fix the
        answer was dropped (viewer's ``user_answer_queue`` is None);
        post-fix it lands on the owner's queue.
        """
        owner_q: queue.Queue[str] = queue.Queue(maxsize=1)
        self.server._get_tab("owner").user_answer_queue = owner_q
        # Viewer tab exists but never ran a task — no queue.
        self.server._get_tab("viewer")
        self.server.printer.subscribe_tab("task-X", "owner")
        self.server.printer.subscribe_tab("task-X", "viewer")

        self.server._cmd_user_answer({"tabId": "viewer", "answer": "from-viewer"})
        self.assertEqual(owner_q.get_nowait(), "from-viewer")

    def test_unknown_tab_with_no_co_subscriber_is_dropped(self) -> None:
        """An answer from an unrelated tab id is dropped silently."""
        self.server._get_tab("stranger")  # exists, no queue, no subscription
        # Should not raise.
        self.server._cmd_user_answer({"tabId": "stranger", "answer": "noop"})

    def test_owner_tab_self_routing_still_works(self) -> None:
        """The common single-window case still uses the direct queue."""
        owner_q: queue.Queue[str] = queue.Queue(maxsize=1)
        self.server._get_tab("solo").user_answer_queue = owner_q
        self.server.printer.subscribe_tab("task-Y", "solo")
        self.server._cmd_user_answer({"tabId": "solo", "answer": "direct"})
        self.assertEqual(owner_q.get_nowait(), "direct")


if __name__ == "__main__":
    unittest.main()
