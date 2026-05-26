"""Integration tests for additional VS Code race conditions.

Each test deterministically forces an interleaving that exposes a real
data race in ``kiss.agents.vscode``.  No mocks of production behaviour
are used — only ``threading.Barrier``-based scheduling control on the
*queue object* the production code calls into.

Races covered here:

- R1 ``_cmd_user_answer`` clear-then-put: two concurrent userAnswer
  commands can both observe an empty ``maxsize=1`` queue, both reach
  ``q.put(answer)``, and the second blocks forever.

- R6 ``_persist_event`` reads ``_persist_agents`` without holding
  ``self._lock`` while ``cleanup_task`` and ``ChatSorcarAgent.run``
  mutate the same map concurrently.

These tests are written so that the post-fix code path (a fully
serialised ``_cmd_user_answer``, an under-``_lock`` lookup in
``_persist_event``) still passes them.
"""

from __future__ import annotations

import queue
import threading
import unittest

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.agents.vscode.server import VSCodeServer


class TestConcurrentUserAnswerWedge(unittest.TestCase):
    """Two concurrent ``_cmd_user_answer`` calls must not deadlock.

    The pre-fix code:

        while not q.empty():
            try: q.get_nowait()
            except queue.Empty: break
        q.put(cmd.get("answer", ""))

    runs the drain+put **outside** ``_state_lock``.  With
    ``maxsize=1`` and two answer threads, both can observe ``empty``,
    both reach ``q.put`` — first succeeds, second blocks forever.
    """

    def _make_tab(self, server: VSCodeServer, tab_id: str) -> _RunningAgentState:
        with server._state_lock:
            tab = _RunningAgentState(tab_id, server._default_model)
            _RunningAgentState.running_agent_states[tab_id] = tab
            tab.user_answer_queue = queue.Queue(maxsize=1)
        return tab

    def tearDown(self) -> None:
        # Wipe registry to keep tests independent.
        with _RunningAgentState._registry_lock:
            _RunningAgentState.running_agent_states.clear()

    def test_concurrent_user_answer_does_not_wedge(self) -> None:
        """Two userAnswer commands on the same tab finish quickly.

        Patches ``q.put`` with a ``Barrier(2)`` wrapper: if the race
        is present, both threads reach the (blocking) ``q.put`` at
        the same time, the barrier opens, the first put succeeds and
        the second blocks forever on the ``maxsize=1`` full queue.

        After the fix moves the drain+put **inside** ``_state_lock``
        and uses ``put_nowait``, the patched ``q.put`` is never
        called — the barrier times out without ever releasing — and
        both threads complete normally.
        """
        server = VSCodeServer()
        tab_id = "race-uans"
        tab = self._make_tab(server, tab_id)
        q = tab.user_answer_queue
        assert q is not None

        barrier = threading.Barrier(2, timeout=1.5)
        orig_put = q.put

        def synced_put(item, *args, **kwargs):  # type: ignore[no-untyped-def]
            # Wait for the peer thread so both reach the blocking put
            # before either succeeds.  If only one thread reaches
            # this (post-fix), the barrier times out (BrokenBarrier)
            # and we proceed normally.
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
            return orig_put(item, *args, **kwargs)

        q.put = synced_put  # type: ignore[method-assign]

        t1 = threading.Thread(
            target=server._cmd_user_answer,
            args=({"tabId": tab_id, "answer": "A"},),
            daemon=True,
        )
        t2 = threading.Thread(
            target=server._cmd_user_answer,
            args=({"tabId": tab_id, "answer": "B"},),
            daemon=True,
        )
        t1.start()
        t2.start()
        t1.join(timeout=3.0)
        t2.join(timeout=3.0)

        wedged = [t for t in (t1, t2) if t.is_alive()]
        # Drain the queue so wedged threads (if any) can unblock and
        # avoid leaking into other tests.
        try:
            while not q.empty():
                q.get_nowait()
        except queue.Empty:
            pass
        t1.join(timeout=1.0)
        t2.join(timeout=1.0)

        self.assertEqual(
            wedged, [],
            "concurrent _cmd_user_answer wedged on q.put with full maxsize=1 queue",
        )


class TestPersistEventLookupRace(unittest.TestCase):
    """``_persist_event`` reads ``_persist_agents`` without ``_lock``.

    Concurrent ``cleanup_task`` removes the entry from
    ``_persist_agents`` (under ``_lock``).  The read in
    ``_persist_event`` is not synchronised — it can observe a
    just-removed agent and route a persistence call against a
    finalised task.

    After the fix wraps the lookup in ``with self._lock:``,
    ``cleanup_task`` and ``_persist_event`` are mutually exclusive
    for the lookup window and ``_persist_event`` either sees the
    agent (and persists) or sees ``None`` (and silently drops) —
    but never the in-between torn state.
    """

    def test_cleanup_excludes_persist_lookup(self) -> None:
        """``cleanup_task`` and ``_persist_event`` must be mutually exclusive.

        Strategy: install a dict subclass that blocks inside ``pop``
        until a signal is set — this lets us pin ``cleanup_task`` to
        hold ``printer._lock`` while we fire ``_persist_event`` from
        another thread.  With the fix, ``_persist_event`` blocks on
        ``_lock`` and only returns after cleanup completes; without
        the fix, ``_persist_event`` reads ``_persist_agents.get(key)``
        without ``_lock`` and returns immediately, racing the pop.
        """
        printer = BaseBrowserPrinter()
        task_key = "42"

        class _StubAgent:
            _last_task_id = 42

        cleanup_entered = threading.Event()
        cleanup_can_exit = threading.Event()

        class _BlockingDict(dict):
            def pop(self, key, default=None):  # type: ignore[override]
                cleanup_entered.set()
                cleanup_can_exit.wait(timeout=2.0)
                return super().pop(key, default)

        printer._persist_agents = _BlockingDict()
        printer._persist_agents[task_key] = _StubAgent()

        cleanup_thread = threading.Thread(
            target=printer.cleanup_task, args=(task_key,), daemon=True,
        )
        cleanup_thread.start()
        cleanup_entered.wait(timeout=2.0)

        # While cleanup holds ``_lock``, fire ``_persist_event``.
        # With the fix in place, the lookup is now under ``_lock``
        # and this thread blocks until cleanup releases.  Without
        # the fix, the lookup races and returns immediately.
        b_done = threading.Event()

        def call_persist() -> None:
            printer._persist_event({"type": "result", "text": "x", "taskId": task_key})
            b_done.set()

        b_thread = threading.Thread(target=call_persist, daemon=True)
        b_thread.start()

        # Give the racing thread time to either complete (bug) or
        # block on the lock (fix).
        completed_during_cleanup = b_done.wait(timeout=0.3)

        cleanup_can_exit.set()
        cleanup_thread.join(timeout=2.0)
        b_thread.join(timeout=2.0)

        self.assertFalse(
            completed_during_cleanup,
            "_persist_event completed while cleanup_task held _lock — "
            "lookup is not serialised against cleanup (race present)",
        )


if __name__ == "__main__":
    unittest.main()
