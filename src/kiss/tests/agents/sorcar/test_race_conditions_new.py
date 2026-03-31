"""Tests for race condition fixes identified in PLAN.md.

Tests verify fixes for:
- P2 / X2: _user_answer uses queue.Queue instead of threading.Event (deadlock-proof)
- P5:      _merging flag protected with _state_lock
- P6:      _file_cache lazy-init protected with _state_lock
- P9:      _extract_result_summary snapshots _recordings before iterating
- P15:     _task_thread cleared after task completes
- P16:     Recordings use explicit IDs instead of thread ident
- D3:      _await_user_response checks stop_event in a loop with timeout

No mocks — uses real server and browser_ui internals.
"""

import inspect
import queue
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

    def test_user_answer_queue_attribute(self) -> None:
        """Verify _user_answer_queue is a queue.Queue, not a plain str."""
        server = VSCodeServer()
        assert isinstance(server._user_answer_queue, queue.Queue), (
            "P2 fix: _user_answer_queue should be queue.Queue"
        )
        assert not hasattr(server, "_user_answer_event"), (
            "P2 fix: _user_answer_event should be removed"
        )

    def test_no_deadlock_with_early_answer(self) -> None:
        """Verify queue-based approach doesn't deadlock when answer arrives early."""
        server = VSCodeServer()
        server._stop_event = threading.Event()
        server.printer._thread_local.stop_event = server._stop_event

        # Put answer before await (simulates early arrival)
        server._user_answer_queue.put("early answer")

        result: list[str] = []

        def awaiter() -> None:
            answer = server._await_user_response()
            result.append(answer)

        t = threading.Thread(target=awaiter, daemon=True)
        t.start()
        t.join(timeout=2.0)

        assert result == ["early answer"], (
            "P2 fix: early answer should be received without deadlock"
        )


# ---------------------------------------------------------------------------
# P5 — _merging flag synchronized with _state_lock
# ---------------------------------------------------------------------------


class TestP5MergingFlagSynchronized(unittest.TestCase):
    """P5 fix: _merging is read and written under _state_lock."""

    def test_merging_read_in_run_task_inner_with_lock(self) -> None:
        """Verify _run_task_inner reads self._merging inside _state_lock."""
        source = inspect.getsource(VSCodeServer._run_task_inner)
        assert "with self._state_lock" in source
        # Find the _merging guard and verify it's inside _state_lock
        lock_idx = source.find("with self._state_lock")
        second_lock = source.find("with self._state_lock", lock_idx + 1)
        merging_idx = source.find("if self._merging:")
        assert merging_idx > 0
        # _merging check should be between second lock start and its block
        assert second_lock < merging_idx, (
            "P5 fix: self._merging read should be inside _state_lock"
        )

    def test_merging_written_in_finish_merge_with_lock(self) -> None:
        """Verify _finish_merge sets _merging = False under _state_lock."""
        source = inspect.getsource(VSCodeServer._finish_merge)
        assert "with self._state_lock" in source, (
            "P5 fix: _finish_merge should use _state_lock"
        )
        assert "self._merging = False" in source

    def test_merging_written_in_start_merge_with_lock(self) -> None:
        """Verify _start_merge_session sets _merging = True under _state_lock."""
        source = inspect.getsource(VSCodeServer._start_merge_session)
        assert "with self._state_lock" in source, (
            "P5 fix: _start_merge_session should use _state_lock"
        )
        assert "self._merging = True" in source


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
    """P9 fix: _extract_result_summary snapshots _recordings under lock,
    then iterates the snapshot outside the lock.
    """

    def test_snapshot_pattern_in_extract_result_summary(self) -> None:
        """Verify _extract_result_summary takes a snapshot inside lock."""
        source = inspect.getsource(VSCodeServer._extract_result_summary)
        assert "snapshot" in source, (
            "P9 fix: should take a snapshot of _recordings"
        )
        # Iteration should be on snapshot, not on _recordings directly
        lock_idx = source.find("with self.printer._lock")
        assert lock_idx >= 0
        # After the with block, iteration should be on snapshot
        after_lock = source[lock_idx:]
        assert "for events_list in snapshot" in after_lock, (
            "P9 fix: should iterate snapshot, not _recordings"
        )

    def test_lock_is_not_reentrant(self) -> None:
        """Verify BaseBrowserPrinter._lock is a non-reentrant Lock."""
        printer = BaseBrowserPrinter()
        assert isinstance(printer._lock, type(threading.Lock()))

    def test_no_deadlock_with_concurrent_broadcast(self) -> None:
        """Verify _extract_result_summary doesn't block broadcast."""
        server = VSCodeServer()

        # Start a recording and add a result event
        server.printer.start_recording(999)
        server.printer.broadcast({"type": "result", "summary": "test"})

        # Call _extract_result_summary in one thread while broadcasting in another
        results: list[str] = []
        errors: list[str] = []

        def extract() -> None:
            try:
                r = server._extract_result_summary()
                results.append(r)
            except Exception as e:
                errors.append(str(e))

        def broadcast_concurrent() -> None:
            try:
                for _ in range(10):
                    server.printer.broadcast({"type": "text_delta", "text": "x"})
            except Exception as e:
                errors.append(str(e))

        t1 = threading.Thread(target=extract, daemon=True)
        t2 = threading.Thread(target=broadcast_concurrent, daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

        assert not errors, f"No errors expected: {errors}"
        assert results == ["test"]
        server.printer.stop_recording(999)


# ---------------------------------------------------------------------------
# P15 — _task_thread cleared after task completion
# ---------------------------------------------------------------------------


class TestP15TaskThreadCleared(unittest.TestCase):
    """P15 fix: _task_thread is set to None in _run_task finally block."""

    def test_task_thread_cleared_in_run_task_finally(self) -> None:
        """Verify _run_task sets self._task_thread = None in finally."""
        source = inspect.getsource(VSCodeServer._run_task)
        assert "self._task_thread = None" in source, (
            "P15 fix: _run_task should clear _task_thread in finally"
        )

    def test_task_thread_cleared_after_completion(self) -> None:
        """Verify _task_thread is None after _run_task completes."""
        server = VSCodeServer()
        events: list[dict] = []
        server.printer.broadcast = lambda e: events.append(e)  # type: ignore[method-assign,assignment]

        # Set a dummy task thread
        thread = threading.Thread(target=lambda: None, daemon=True)
        thread.start()
        thread.join()
        server._task_thread = thread

        # Run _run_task — it will fail early but the finally block should clear _task_thread
        server._run_task({"prompt": "test", "model": "test"})

        assert server._task_thread is None, (
            "P15 fix: _task_thread should be None after _run_task completes"
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

    def test_recording_with_explicit_id(self) -> None:
        """Verify recordings work correctly with explicit IDs."""
        printer = BaseBrowserPrinter()
        printer.start_recording(42)
        printer.broadcast({"type": "text_delta", "text": "hello"})
        events = printer.stop_recording(42)
        assert len(events) == 1
        assert events[0]["text"] == "hello"

    def test_no_cross_contamination_between_ids(self) -> None:
        """Verify recordings with different IDs are independent."""
        printer = BaseBrowserPrinter()
        printer.start_recording(1)
        printer.broadcast({"type": "text_delta", "text": "task1"})
        events1 = printer.stop_recording(1)

        printer.start_recording(2)
        printer.broadcast({"type": "text_delta", "text": "task2"})
        events2 = printer.stop_recording(2)

        # Recording 1 has only task1 events
        assert len(events1) == 1
        assert events1[0]["text"] == "task1"
        # Recording 2 has only task2 events
        assert len(events2) == 1
        assert events2[0]["text"] == "task2"

    def test_backward_compat_without_id(self) -> None:
        """Verify start/stop_recording still work without explicit ID (uses thread ident)."""
        printer = BaseBrowserPrinter()
        printer.start_recording()
        printer.broadcast({"type": "text_delta", "text": "compat"})
        events = printer.stop_recording()
        assert len(events) == 1
        assert events[0]["text"] == "compat"


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
        server._stop_event = threading.Event()
        server.printer._thread_local.stop_event = server._stop_event
        server._stop_event.set()

        with self.assertRaises(KeyboardInterrupt):
            server._await_user_response()

    def test_returns_answer_from_queue(self) -> None:
        """Verify _await_user_response returns the answer from the queue."""
        server = VSCodeServer()
        server._stop_event = threading.Event()
        server.printer._thread_local.stop_event = server._stop_event
        server._user_answer_queue.put("test answer")

        result = server._await_user_response()
        assert result == "test answer"


if __name__ == "__main__":
    unittest.main()
