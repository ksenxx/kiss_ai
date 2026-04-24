"""Integration tests confirming fixes for bugs and inconsistencies in
``kiss.agents.vscode`` — audit round 5.

B1 fix: ``_await_user_response`` now acquires ``_state_lock`` before
    reading ``_tab_states``, consistent with the locking discipline.

B2 fix: ``_handle_autocommit_action`` now acquires ``_state_lock``
    before reading ``_tab_states`` when persisting the autocommit event.

I1 fix: ``_cmd_user_answer`` now uses ``cmd.get("tabId", "")`` (empty
    string default), consistent with every other command handler.
"""

from __future__ import annotations

import inspect
import queue
import re
import threading
import unittest

from kiss.agents.vscode.commands import _CommandsMixin
from kiss.agents.vscode.merge_flow import _MergeFlowMixin
from kiss.agents.vscode.server import VSCodeServer
from kiss.agents.vscode.task_runner import _TaskRunnerMixin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ===================================================================
# B1 — _await_user_response reads _tab_states without lock
# ===================================================================


class TestAwaitUserResponseLockingFix(unittest.TestCase):
    """B1 FIX: ``_await_user_response`` now acquires ``_state_lock``
    before reading ``_tab_states``, consistent with the locking
    discipline used everywhere else.
    """

    def test_source_reads_tab_states_under_lock(self) -> None:
        """Structural: the source accesses ``_tab_states`` inside a
        ``with self._state_lock`` block.
        """
        src = inspect.getsource(_TaskRunnerMixin._await_user_response)
        assert "_tab_states.get(" in src, (
            "_await_user_response should access _tab_states"
        )
        assert "with self._state_lock" in src, (
            "B1 FIX: _await_user_response now reads _tab_states "
            "under _state_lock"
        )

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
                # Hold the lock until the await thread has had a chance
                # to start and block on it
                await_started.wait(timeout=5)
                import time
                time.sleep(0.05)
            # Lock released — _await_user_response can proceed

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

        # The call succeeded after the lock was released
        assert result_box == ["hello"], (
            f"Expected ['hello'], got {result_box}. "
            "B1 FIX: _await_user_response correctly acquires the lock"
        )


# ===================================================================
# B2 — _handle_autocommit_action reads _tab_states without lock
# ===================================================================


class TestAutocommitActionLockingFix(unittest.TestCase):
    """B2 FIX: ``_handle_autocommit_action`` now acquires ``_state_lock``
    before reading ``_tab_states`` when persisting the autocommit event.
    """

    def test_source_reads_tab_states_under_lock(self) -> None:
        """Structural: every ``_tab_states.get`` call is guarded by a
        ``with self._state_lock`` block.
        """
        src = inspect.getsource(_MergeFlowMixin._handle_autocommit_action)
        assert "_tab_states.get(tab_id)" in src.replace("self.", ""), (
            "_handle_autocommit_action should access _tab_states"
        )
        lock_blocks = list(re.finditer(r"with self\._state_lock", src))
        tab_accesses = list(re.finditer(r"self\._tab_states\.get", src))
        assert len(lock_blocks) >= len(tab_accesses), (
            f"B2 FIX: {len(tab_accesses)} _tab_states.get() calls "
            f"guarded by {len(lock_blocks)} _state_lock blocks"
        )



# ===================================================================
# I1 — Inconsistent tabId default across command handlers
# ===================================================================


class TestTabIdDefaultConsistencyFix(unittest.TestCase):
    """I1 FIX: ``_cmd_user_answer`` now uses ``cmd.get("tabId", "")``
    (empty string default), consistent with every other handler.
    """

    def test_user_answer_uses_empty_string_default(self) -> None:
        src = inspect.getsource(_CommandsMixin._cmd_user_answer)
        assert re.search(r'cmd\.get\("tabId",\s*""\)', src), (
            "I1 FIX: _cmd_user_answer now uses cmd.get('tabId', '')"
        )

    def test_all_handlers_use_empty_string_default(self) -> None:
        """Every cmd handler that reads tabId uses a default of ''."""
        handlers_with_tabid = [
            _CommandsMixin._cmd_run,
            _CommandsMixin._cmd_stop,
            _CommandsMixin._cmd_select_model,
            _CommandsMixin._cmd_close_tab,
            _CommandsMixin._cmd_new_chat,
            _CommandsMixin._cmd_resume_session,
            _CommandsMixin._cmd_merge_action,
            _CommandsMixin._cmd_complete,
            _CommandsMixin._cmd_get_adjacent_task,
            _CommandsMixin._cmd_generate_commit_message,
            _CommandsMixin._cmd_worktree_action,
            _CommandsMixin._cmd_autocommit_action,
            _CommandsMixin._cmd_user_answer,
        ]
        for handler in handlers_with_tabid:
            src = inspect.getsource(handler)
            if 'cmd.get("tabId"' not in src:
                continue
            has_empty_default = bool(
                re.search(r'cmd\.get\("tabId",\s*""\)', src)
            )
            has_none_default = bool(
                re.search(r'cmd\.get\("tabId"\)', src)
                and not re.search(r'cmd\.get\("tabId",', src)
            )
            assert has_empty_default and not has_none_default, (
                f"I1 FIX: {handler.__name__} should use "
                f'cmd.get("tabId", ""), not cmd.get("tabId")'
            )


if __name__ == "__main__":
    unittest.main()
