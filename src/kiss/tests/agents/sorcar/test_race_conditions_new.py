"""Tests for race condition fixes identified in PLAN.md.

Tests verify fixes for:
- P2 / X2: _user_answer uses queue.Queue instead of threading.Event (deadlock-proof)
- P6:      _file_cache lazy-init protected with _state_lock
- P9:      _extract_result_summary snapshots _recordings before iterating
- P15:     _task_thread cleared after task completes
- P16:     Recordings use explicit IDs instead of thread ident
- D3:      _await_user_response checks stop_event in a loop with timeout

No mocks — uses real server and browser_ui internals.
"""

import inspect
import threading
import unittest

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter
from kiss.agents.vscode.server import VSCodeServer

# ---------------------------------------------------------------------------
# P2 / X2 — _user_answer uses queue.Queue (no more clear-before-set race)
# ---------------------------------------------------------------------------


class TestP2UserAnswerClearBeforeSetRace(unittest.TestCase):
    """P2 fix: _await_user_response uses queue.Queue instead of Event."""

    def test_uses_queue_not_event(self) -> None:
        """Verify _await_user_response uses queue.Queue, not threading.Event."""
        source = inspect.getsource(VSCodeServer._await_user_response)
        assert "queue" in source.lower() or "_user_answer_queue" in source, (
            "P2 fix: _await_user_response should use queue.Queue"
        )
        assert "clear()" not in source, (
            "P2 fix: no .clear() call — queue-based, not Event-based"
        )

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# P6 — _file_cache protected with _state_lock
# ---------------------------------------------------------------------------


class TestP6FileCacheProtected(unittest.TestCase):
    """P6 fix: _file_cache lazy-init and writes are protected with _state_lock."""

    def test_get_files_uses_lock_around_lazy_init(self) -> None:
        """Verify _get_files reads self._file_cache under _state_lock."""
        source = inspect.getsource(VSCodeServer._get_files)
        assert "with self._state_lock" in source, (
            "P6 fix: _file_cache lazy-init should be under _state_lock"
        )

    def test_file_cache_write_in_refresh_has_lock(self) -> None:
        """Verify _do_refresh writes _file_cache under _state_lock."""
        source = inspect.getsource(VSCodeServer._refresh_file_cache)
        assert "with self._state_lock" in source, (
            "P6 fix: _do_refresh should write _file_cache under _state_lock"
        )


# ---------------------------------------------------------------------------
# P9 — _recordings snapshot before iterating
# ---------------------------------------------------------------------------


class TestP9RecordingsSnapshotBeforeIterate(unittest.TestCase):
    """B7 fix: _extract_result_summary uses peek_recording(recording_id)
    instead of directly iterating _recordings.
    """

    def test_snapshot_pattern_in_extract_result_summary(self) -> None:
        """Verify _extract_result_summary uses peek_recording."""
        source = inspect.getsource(VSCodeServer._extract_result_summary)
        assert "peek_recording" in source, (
            "B7 fix: should use peek_recording instead of raw _recordings"
        )
        assert "_recordings" not in source, (
            "B7 fix: should not access _recordings directly"
        )


# ---------------------------------------------------------------------------
# P15 — _task_thread cleared after task completion
# ---------------------------------------------------------------------------


class TestP15TaskThreadCleared(unittest.TestCase):
    """P15 fix: _task_thread is set to None in _run_task finally block."""

    def test_task_thread_cleared_in_run_task_finally(self) -> None:
        """Verify _run_task cleans up per-tab thread in finally."""
        source = inspect.getsource(VSCodeServer._run_task)
        assert "self._task_threads.pop(" in source, (
            "P15 fix: _run_task should remove tab from _task_threads in finally"
        )

# ---------------------------------------------------------------------------
# P16 — Recordings use explicit IDs instead of thread ident
# ---------------------------------------------------------------------------


class TestP16ExplicitRecordingIds(unittest.TestCase):
    """P16 fix: start_recording/stop_recording accept recording_id parameter."""

    def test_start_recording_accepts_id(self) -> None:
        """Verify start_recording accepts a recording_id parameter."""
        sig = inspect.signature(BaseBrowserPrinter.start_recording)
        assert "recording_id" in sig.parameters

    def test_stop_recording_accepts_id(self) -> None:
        """Verify stop_recording accepts a recording_id parameter."""
        sig = inspect.signature(BaseBrowserPrinter.stop_recording)
        assert "recording_id" in sig.parameters

# ---------------------------------------------------------------------------
# D3 — _await_user_response checks stop_event with timeout
# ---------------------------------------------------------------------------


class TestD3UserAnswerWaitWithTimeout(unittest.TestCase):
    """D3 fix: _await_user_response polls stop_event in a loop."""

    def test_await_user_response_has_timeout(self) -> None:
        """Verify _await_user_response uses timeout in its wait loop."""
        source = inspect.getsource(VSCodeServer._await_user_response)
        assert "timeout" in source, (
            "D3 fix: _await_user_response should use timeout"
        )

    def test_stop_event_checked_in_await_loop(self) -> None:
        """Verify _await_user_response checks stop_event in a loop."""
        source = inspect.getsource(VSCodeServer._await_user_response)
        assert "stop" in source.lower() and "while" in source, (
            "D3 fix: should check stop_event in a while loop"
        )

    def test_raises_on_stop(self) -> None:
        """Verify _await_user_response raises KeyboardInterrupt on stop."""
        server = VSCodeServer()
        stop_event = threading.Event()
        server.printer._thread_local.stop_event = stop_event
        stop_event.set()

        with self.assertRaises(KeyboardInterrupt):
            server._await_user_response()

if __name__ == "__main__":
    unittest.main()
