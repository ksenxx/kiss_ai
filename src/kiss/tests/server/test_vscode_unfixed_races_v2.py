# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test for a VS Code user-answer race condition.

The test deterministically forces an interleaving that would expose a
real data race in ``kiss.server``.  No mocks of production behaviour
are used — only ``threading.Barrier``-based scheduling control on the
*queue object* the production code calls into.

Race covered here:

- R1 ``_cmd_user_answer`` clear-then-put: two concurrent userAnswer
  commands can both observe an empty ``maxsize=1`` queue, both reach
  ``q.put(answer)``, and the second blocks forever.

The test is written so that the post-fix code path (a fully
serialised ``_cmd_user_answer`` using ``put_nowait`` under
``_state_lock``) still passes it.
"""

from __future__ import annotations

import queue
import threading
import unittest

from kiss.server import agent_state
from kiss.server.server import VSCodeServer


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

    def _make_tab_state(
        self, server: VSCodeServer, tab_id: str,
    ) -> agent_state.AgentState:
        with server._state_lock:
            state = agent_state.AgentState(
                f"task-for-{tab_id}",
                tab_id=tab_id,
                server_owned=True,
                is_task_active=True,
            )
            state.user_answer_queue = queue.Queue(maxsize=1)
            agent_state.register(state)
        return state

    def tearDown(self) -> None:
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()

    def test_concurrent_user_answer_does_not_wedge(self) -> None:
        """Two userAnswer commands on the same tab finish quickly.

        Patches ``q.put`` with a ``Barrier(2)`` wrapper: if the race
        is present, both threads reach the (blocking) ``q.put`` at
        the same time, the barrier opens, the first put succeeds and
        the second blocks forever on the ``maxsize=1`` full queue.

        With the fixed code the drain+put runs **inside**
        ``_state_lock`` and uses ``put_nowait``, so the patched
        ``q.put`` is never called — the barrier times out without
        ever releasing — and both threads complete normally.
        """
        server = VSCodeServer()
        tab_id = "race-uans"
        state = self._make_tab_state(server, tab_id)
        q = state.user_answer_queue
        assert q is not None

        barrier = threading.Barrier(2, timeout=1.5)
        orig_put = q.put

        def synced_put(item, *args, **kwargs):  # type: ignore[no-untyped-def]
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


if __name__ == "__main__":
    unittest.main()
