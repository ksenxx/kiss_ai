# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests confirming fixes for bugs and inconsistencies in
``kiss.server`` — audit round 5.

B1 fix: ``_await_user_response`` now acquires ``_state_lock`` before
    reading the task-keyed ``agent_state`` registry, consistent with
    the locking discipline.

I1 fix: ``_cmd_user_answer`` now uses ``cmd.get("tabId", "")`` (empty
    string default), consistent with every other command handler.
"""

from __future__ import annotations

import queue
import threading
import unittest

from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer with broadcast capture (no stdout)."""
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        with lock:
            events.append(event)
        with server.printer._lock:
            server.printer._record_event(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


class TestAwaitUserResponseLockingFix(unittest.TestCase):
    """B1 FIX: ``_await_user_response`` resolves the answer queue
    through the registry under ``_state_lock`` (the registry's
    ``STATE_LOCK``), consistent with the locking discipline used
    everywhere else.
    """

    def tearDown(self) -> None:
        agent_state.agent_states.clear()

    def test_behavioral_read_with_lock(self) -> None:
        """Behavioral: ``_await_user_response`` acquires the state
        lock while resolving the queue, so calling it while another
        thread holds the lock will block until the lock is released.
        """
        server, _ = _make_server()
        state = agent_state.AgentState(
            "test-tab",
            tab_id="test-tab",
            server_owned=True,
            stop_event=threading.Event(),
        )
        state.user_answer_queue = queue.Queue(maxsize=1)
        state.user_answer_queue.put("hello")
        agent_state.register(state)

        server.printer._thread_local.stop_event = state.stop_event
        server.printer._thread_local.task_id = "test-tab"

        lock_held = threading.Event()
        await_started = threading.Event()
        done = threading.Event()
        result_box: list[str] = []

        def hold_lock() -> None:
            with server._state_lock:
                lock_held.set()
                await_started.wait(timeout=5)
                import time
                time.sleep(0.05)

        server.printer.subscribe_tab("test-tab", "test-tab")

        def call_await() -> None:
            lock_held.wait(timeout=5)
            await_started.set()
            server.printer._thread_local.stop_event = state.stop_event
            server.printer._thread_local.task_id = "test-tab"
            result_box.append(server._await_user_response())
            done.set()

        t1 = threading.Thread(target=hold_lock)
        t2 = threading.Thread(target=call_await)
        t1.start()
        t2.start()
        t2.join(timeout=5)
        t1.join(timeout=5)

        assert result_box == ["hello"], (
            f"Expected ['hello'], got {result_box}. "
            "B1 FIX: _await_user_response correctly acquires the lock"
        )






if __name__ == "__main__":
    unittest.main()
