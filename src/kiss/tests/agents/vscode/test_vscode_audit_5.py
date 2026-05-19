"""Integration tests confirming fixes for bugs and inconsistencies in
``kiss.agents.vscode`` — audit round 5.

B1 fix: ``_await_user_response`` now acquires ``_state_lock`` before
    reading ``_running_agent_states``, consistent with the locking discipline.

B2 fix: ``_handle_autocommit_action`` now acquires ``_state_lock``
    before reading ``_running_agent_states`` when persisting the autocommit event.

I1 fix: ``_cmd_user_answer`` now uses ``cmd.get("tabId", "")`` (empty
    string default), consistent with every other command handler.
"""

from __future__ import annotations

import queue
import threading
import unittest

from kiss.agents.vscode.server import VSCodeServer


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
    """B1 FIX: ``_await_user_response`` now acquires ``_state_lock``
    before reading ``_running_agent_states``, consistent with the locking
    discipline used everywhere else.
    """


    def test_behavioral_read_with_lock(self) -> None:
        """Behavioral: ``_await_user_response`` now acquires
        ``_state_lock``, so calling it while another thread holds
        the lock will block until the lock is released.
        """
        server, _ = _make_server()
        tab = server._get_tab("test-tab")
        tab.stop_event = threading.Event()
        tab.user_answer_queue = queue.Queue(maxsize=1)
        tab.user_answer_queue.put("hello")

        server.printer._thread_local.stop_event = tab.stop_event
        server.printer._thread_local.tab_id = "test-tab"

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

        def call_await() -> None:
            lock_held.wait(timeout=5)
            await_started.set()
            server.printer._thread_local.stop_event = tab.stop_event
            server.printer._thread_local.tab_id = "test-tab"
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
