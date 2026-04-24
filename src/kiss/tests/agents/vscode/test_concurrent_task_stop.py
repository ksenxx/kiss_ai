"""Integration tests: stopping one task while another runs concurrently.

Reproduces the bug where stopping a task corrupts the concurrent task's
printer state (stream parsing and recording), which on the frontend
manifests as panels collapsing or going blank.

Covers both directions:
- Stopping a non-worktree task while a worktree task runs
- Stopping a worktree task while a non-worktree task runs

Root causes fixed:

1. Stream state (``_current_block_type``, ``_tool_name``,
   ``_tool_json_buffer``) and recording (``_recording``) were shared
   instance attributes.  Now thread-local / per-tab.

2. Bash buffering state (``_bash_state``) was a single shared instance.
   Two concurrent tabs shared ``buffer``, ``streamed``, ``generation``,
   and ``timer``, causing cross-contamination.  Now per-tab via
   ``_bash_states`` dict.

3. ``_flush_bash`` had a TOCTOU: after the generation check passed
   (inside ``_bash_lock``), the lock was released before ``broadcast()``,
   allowing ``reset()`` + ``start_recording()`` to slip in and the stale
   text to leak into the new recording.  Now ``broadcast()`` is called
   while still holding ``_bash_lock``.
"""

from __future__ import annotations

import threading
import unittest
from typing import Any

from kiss.agents.vscode.browser_ui import BaseBrowserPrinter


class TestStreamStateIsolation(unittest.TestCase):
    """Stream parsing state must be per-thread so concurrent tasks don't interfere."""

    def test_reset_does_not_corrupt_concurrent_task_block_type(self) -> None:
        """When task A calls reset(), task B's _current_block_type is preserved.

        Scenario: task B is mid-thinking-block (_current_block_type="thinking").
        Task A finishes and calls reset().  After that, task B's
        _current_block_type must still be "thinking", not "".
        """
        printer = BaseBrowserPrinter()
        barrier = threading.Barrier(2, timeout=5)
        barrier2 = threading.Barrier(2, timeout=5)
        results: dict[str, str] = {}

        def wt_task() -> None:
            printer._thread_local.tab_id = "B"
            printer._current_block_type = "thinking"
            barrier.wait()
            barrier2.wait()
            results["B_block_type"] = printer._current_block_type

        def non_wt_task() -> None:
            printer._thread_local.tab_id = "A"
            printer._current_block_type = "tool_use"
            barrier.wait()
            printer.reset()
            results["A_block_type"] = printer._current_block_type
            barrier2.wait()

        t_wt = threading.Thread(target=wt_task, daemon=True)
        t_nwt = threading.Thread(target=non_wt_task, daemon=True)
        t_wt.start()
        t_nwt.start()
        t_wt.join(timeout=5)
        t_nwt.join(timeout=5)

        assert results["A_block_type"] == "", (
            f"Task A's block type should be '' after reset, got '{results['A_block_type']}'"
        )
        assert results["B_block_type"] == "thinking", (
            f"Task B's block type should still be 'thinking', got '{results['B_block_type']}'"
        )

    def test_reset_does_not_corrupt_tool_json_buffer(self) -> None:
        """When task A calls reset(), task B's _tool_json_buffer is preserved."""
        printer = BaseBrowserPrinter()
        barrier = threading.Barrier(2, timeout=5)
        barrier2 = threading.Barrier(2, timeout=5)
        results: dict[str, str] = {}

        def wt_task() -> None:
            printer._thread_local.tab_id = "B"
            printer._tool_json_buffer = '{"file_path": "/tmp/foo.py"'
            printer._tool_name = "Read"
            barrier.wait()
            barrier2.wait()
            results["B_tool_json"] = printer._tool_json_buffer
            results["B_tool_name"] = printer._tool_name

        def non_wt_task() -> None:
            printer._thread_local.tab_id = "A"
            printer._tool_json_buffer = '{"command": "ls"}'
            printer._tool_name = "Bash"
            barrier.wait()
            printer.reset()
            results["A_tool_json"] = printer._tool_json_buffer
            results["A_tool_name"] = printer._tool_name
            barrier2.wait()

        t_wt = threading.Thread(target=wt_task, daemon=True)
        t_nwt = threading.Thread(target=non_wt_task, daemon=True)
        t_wt.start()
        t_nwt.start()
        t_wt.join(timeout=5)
        t_nwt.join(timeout=5)

        assert results["A_tool_json"] == ""
        assert results["A_tool_name"] == ""
        assert results["B_tool_json"] == '{"file_path": "/tmp/foo.py"'
        assert results["B_tool_name"] == "Read"


class TestRecordingIsolation(unittest.TestCase):
    """Recording must be per-tab so one task's stop doesn't lose another's events."""

    def test_stop_recording_does_not_destroy_concurrent_recording(self) -> None:
        """When task A calls stop_recording(), task B's recording is preserved.

        Scenario: both tasks start recording. Task A stops (stop_recording).
        Task B continues and later stops. Both should get their own events.
        """
        printer = BaseBrowserPrinter()
        barrier = threading.Barrier(2, timeout=5)
        barrier2 = threading.Barrier(2, timeout=5)
        results: dict[str, list[dict]] = {}

        def wt_task() -> None:
            printer._thread_local.tab_id = "B"
            printer.start_recording()
            with printer._lock:
                printer._record_event({"type": "thinking_start", "tabId": "B"})
            barrier.wait()
            barrier2.wait()
            with printer._lock:
                printer._record_event({"type": "thinking_delta", "text": "hi", "tabId": "B"})
            results["B_events"] = printer.stop_recording()

        def non_wt_task() -> None:
            printer._thread_local.tab_id = "A"
            printer.start_recording()
            with printer._lock:
                printer._record_event({"type": "tool_call", "name": "Read", "tabId": "A"})
            barrier.wait()
            results["A_events"] = printer.stop_recording()
            barrier2.wait()

        t_wt = threading.Thread(target=wt_task, daemon=True)
        t_nwt = threading.Thread(target=non_wt_task, daemon=True)
        t_wt.start()
        t_nwt.start()
        t_wt.join(timeout=5)
        t_nwt.join(timeout=5)

        assert len(results["A_events"]) == 1
        assert results["A_events"][0]["type"] == "tool_call"
        assert len(results["B_events"]) == 2
        assert results["B_events"][0]["type"] == "thinking_start"
        assert results["B_events"][1]["type"] == "thinking_delta"

    def test_start_recording_does_not_overwrite_concurrent_recording(self) -> None:
        """When task B calls start_recording(), task A's in-progress recording
        is not reset."""
        printer = BaseBrowserPrinter()
        barrier = threading.Barrier(2, timeout=5)
        results: dict[str, list[dict]] = {}

        def task_a() -> None:
            printer._thread_local.tab_id = "A"
            printer.start_recording()
            with printer._lock:
                printer._record_event({"type": "tool_call", "name": "Bash", "tabId": "A"})
            barrier.wait()
            with printer._lock:
                printer._record_event({"type": "tool_result", "content": "ok", "tabId": "A"})
            results["A_events"] = printer.stop_recording()

        def task_b() -> None:
            printer._thread_local.tab_id = "B"
            barrier.wait()
            printer.start_recording()
            with printer._lock:
                printer._record_event({"type": "text_delta", "text": "hi", "tabId": "B"})
            results["B_events"] = printer.stop_recording()

        t_a = threading.Thread(target=task_a, daemon=True)
        t_b = threading.Thread(target=task_b, daemon=True)
        t_a.start()
        t_b.start()
        t_a.join(timeout=5)
        t_b.join(timeout=5)

        assert len(results["A_events"]) == 2, (
            f"Task A should have 2 events, got {len(results['A_events'])}"
        )
        assert len(results["B_events"]) == 1

    def test_peek_recording_returns_per_tab_events(self) -> None:
        """peek_recording returns only the calling tab's events."""
        printer = BaseBrowserPrinter()
        printer._thread_local.tab_id = "A"
        printer.start_recording()
        with printer._lock:
            printer._record_event({"type": "tool_call", "name": "Read", "tabId": "A"})
        events = printer.peek_recording()
        assert len(events) == 1
        assert events[0]["type"] == "tool_call"
        with printer._lock:
            printer._record_event({"type": "tool_result", "content": "ok", "tabId": "A"})
        events = printer.stop_recording()
        assert len(events) == 2


class TestConcurrentStopScenario(unittest.TestCase):
    """End-to-end test of the concurrent stop scenario described in the bug."""

    def test_stop_non_wt_preserves_wt_task_integrity(self) -> None:
        """Stopping a non-worktree task while a worktree task streams thinking
        tokens must not corrupt the worktree task's event types.

        Simulates the exact interleaving:
        1. Both tasks start and begin streaming
        2. Non-wt task is stopped (reset + stop_recording called)
        3. Wt task continues streaming — its events must use correct types
        """
        printer = BaseBrowserPrinter()
        barrier = threading.Barrier(2, timeout=5)
        barrier2 = threading.Barrier(2, timeout=5)
        wt_events: list[dict] = []

        def wt_task() -> None:
            printer._thread_local.tab_id = "wt"
            printer.start_recording()
            printer._current_block_type = "thinking"
            printer.broadcast({"type": "thinking_start", "tabId": "wt"})
            barrier.wait()
            barrier2.wait()
            delta_type = (
                "thinking_delta"
                if printer._current_block_type == "thinking"
                else "text_delta"
            )
            printer.broadcast({"type": delta_type, "text": "reasoning...", "tabId": "wt"})
            printer._current_block_type = ""
            printer.broadcast({"type": "thinking_end", "tabId": "wt"})
            wt_events.extend(printer.stop_recording())

        def non_wt_task() -> None:
            printer._thread_local.tab_id = "nwt"
            printer.start_recording()
            printer._current_block_type = "tool_use"
            barrier.wait()
            printer.stop_recording()
            printer.reset()
            barrier2.wait()

        t_wt = threading.Thread(target=wt_task, daemon=True)
        t_nwt = threading.Thread(target=non_wt_task, daemon=True)
        t_wt.start()
        t_nwt.start()
        t_wt.join(timeout=5)
        t_nwt.join(timeout=5)

        types = [e["type"] for e in wt_events]
        assert types == ["thinking_start", "thinking_delta", "thinking_end"], (
            f"Expected thinking events, got {types}. "
            "Non-wt task's reset() corrupted wt task's stream state."
        )


class TestStopWtPreservesNonWtStreamState(unittest.TestCase):
    """Reverse direction: stopping a worktree task must not corrupt the
    concurrent non-worktree task's stream-parsing state."""

    def test_reset_on_wt_does_not_corrupt_non_wt_block_type(self) -> None:
        """When the wt task calls reset(), the non-wt task's _current_block_type
        is preserved.

        Scenario: non-wt task is mid-thinking-block (_current_block_type="thinking").
        Wt task finishes and calls reset().  After that, non-wt task's
        _current_block_type must still be "thinking", not "".
        """
        printer = BaseBrowserPrinter()
        barrier = threading.Barrier(2, timeout=5)
        barrier2 = threading.Barrier(2, timeout=5)
        results: dict[str, str] = {}

        def non_wt_task() -> None:
            printer._thread_local.tab_id = "nwt"
            printer._current_block_type = "thinking"
            barrier.wait()
            barrier2.wait()
            results["nwt_block_type"] = printer._current_block_type

        def wt_task() -> None:
            printer._thread_local.tab_id = "wt"
            printer._current_block_type = "tool_use"
            barrier.wait()
            printer.reset()
            results["wt_block_type"] = printer._current_block_type
            barrier2.wait()

        t_nwt = threading.Thread(target=non_wt_task, daemon=True)
        t_wt = threading.Thread(target=wt_task, daemon=True)
        t_nwt.start()
        t_wt.start()
        t_nwt.join(timeout=5)
        t_wt.join(timeout=5)

        assert results["wt_block_type"] == "", (
            f"Wt task's block type should be '' after reset, got '{results['wt_block_type']}'"
        )
        assert results["nwt_block_type"] == "thinking", (
            f"Non-wt task's block type should still be 'thinking', "
            f"got '{results['nwt_block_type']}'"
        )

    def test_reset_on_wt_does_not_corrupt_non_wt_tool_buffer(self) -> None:
        """When the wt task calls reset(), the non-wt task's tool_json_buffer
        and tool_name are preserved."""
        printer = BaseBrowserPrinter()
        barrier = threading.Barrier(2, timeout=5)
        barrier2 = threading.Barrier(2, timeout=5)
        results: dict[str, str] = {}

        def non_wt_task() -> None:
            printer._thread_local.tab_id = "nwt"
            printer._tool_json_buffer = '{"command": "uv run pytest"}'
            printer._tool_name = "Bash"
            barrier.wait()
            barrier2.wait()
            results["nwt_tool_json"] = printer._tool_json_buffer
            results["nwt_tool_name"] = printer._tool_name

        def wt_task() -> None:
            printer._thread_local.tab_id = "wt"
            printer._tool_json_buffer = '{"file_path": "/tmp/x.py"}'
            printer._tool_name = "Write"
            barrier.wait()
            printer.reset()
            results["wt_tool_json"] = printer._tool_json_buffer
            results["wt_tool_name"] = printer._tool_name
            barrier2.wait()

        t_nwt = threading.Thread(target=non_wt_task, daemon=True)
        t_wt = threading.Thread(target=wt_task, daemon=True)
        t_nwt.start()
        t_wt.start()
        t_nwt.join(timeout=5)
        t_wt.join(timeout=5)

        assert results["wt_tool_json"] == ""
        assert results["wt_tool_name"] == ""
        assert results["nwt_tool_json"] == '{"command": "uv run pytest"}'
        assert results["nwt_tool_name"] == "Bash"


class TestStopWtPreservesNonWtRecording(unittest.TestCase):
    """Reverse direction: stopping a worktree task's recording must not
    destroy the concurrent non-worktree task's recorded events."""

    def test_stop_wt_recording_does_not_destroy_non_wt_recording(self) -> None:
        """When the wt task calls stop_recording(), the non-wt task's
        recording is preserved and can be stopped later with its full events."""
        printer = BaseBrowserPrinter()
        barrier = threading.Barrier(2, timeout=5)
        barrier2 = threading.Barrier(2, timeout=5)
        results: dict[str, list[dict]] = {}

        def non_wt_task() -> None:
            printer._thread_local.tab_id = "nwt"
            printer.start_recording()
            with printer._lock:
                printer._record_event({"type": "thinking_start", "tabId": "nwt"})
            barrier.wait()
            barrier2.wait()
            with printer._lock:
                printer._record_event(
                    {"type": "thinking_delta", "text": "analyzing...", "tabId": "nwt"}
                )
                printer._record_event({"type": "thinking_end", "tabId": "nwt"})
            results["nwt_events"] = printer.stop_recording()

        def wt_task() -> None:
            printer._thread_local.tab_id = "wt"
            printer.start_recording()
            with printer._lock:
                printer._record_event({"type": "tool_call", "name": "Edit", "tabId": "wt"})
                printer._record_event(
                    {"type": "tool_result", "content": "ok", "tabId": "wt"}
                )
            barrier.wait()
            results["wt_events"] = printer.stop_recording()
            barrier2.wait()

        t_nwt = threading.Thread(target=non_wt_task, daemon=True)
        t_wt = threading.Thread(target=wt_task, daemon=True)
        t_nwt.start()
        t_wt.start()
        t_nwt.join(timeout=5)
        t_wt.join(timeout=5)

        assert len(results["wt_events"]) == 2
        assert results["wt_events"][0]["type"] == "tool_call"
        assert results["wt_events"][1]["type"] == "tool_result"
        assert len(results["nwt_events"]) == 3
        assert results["nwt_events"][0]["type"] == "thinking_start"
        assert results["nwt_events"][1]["type"] == "thinking_delta"
        assert results["nwt_events"][2]["type"] == "thinking_end"

    def test_start_wt_recording_does_not_overwrite_non_wt_recording(self) -> None:
        """When the wt task calls start_recording() after the non-wt task has
        already started recording, the non-wt task's events are not lost."""
        printer = BaseBrowserPrinter()
        barrier = threading.Barrier(2, timeout=5)
        results: dict[str, list[dict]] = {}

        def non_wt_task() -> None:
            printer._thread_local.tab_id = "nwt"
            printer.start_recording()
            with printer._lock:
                printer._record_event({"type": "tool_call", "name": "Read", "tabId": "nwt"})
            barrier.wait()
            with printer._lock:
                printer._record_event(
                    {"type": "tool_result", "content": "data", "tabId": "nwt"}
                )
            results["nwt_events"] = printer.stop_recording()

        def wt_task() -> None:
            printer._thread_local.tab_id = "wt"
            barrier.wait()
            printer.start_recording()
            with printer._lock:
                printer._record_event({"type": "text_delta", "text": "hello", "tabId": "wt"})
            results["wt_events"] = printer.stop_recording()

        t_nwt = threading.Thread(target=non_wt_task, daemon=True)
        t_wt = threading.Thread(target=wt_task, daemon=True)
        t_nwt.start()
        t_wt.start()
        t_nwt.join(timeout=5)
        t_wt.join(timeout=5)

        assert len(results["nwt_events"]) == 2, (
            f"Non-wt task should have 2 events, got {len(results['nwt_events'])}"
        )
        assert len(results["wt_events"]) == 1


class TestStopWtConcurrentScenario(unittest.TestCase):
    """End-to-end: stopping a worktree task while a non-worktree task streams."""

    def test_stop_wt_preserves_non_wt_task_integrity(self) -> None:
        """Stopping a worktree task while a non-worktree task streams thinking
        tokens must not corrupt the non-worktree task's event types.

        Simulates the exact interleaving:
        1. Both tasks start and begin streaming
        2. Wt task is stopped (reset + stop_recording called)
        3. Non-wt task continues streaming — its events must use correct types
        """
        printer = BaseBrowserPrinter()
        barrier = threading.Barrier(2, timeout=5)
        barrier2 = threading.Barrier(2, timeout=5)
        nwt_events: list[dict] = []

        def non_wt_task() -> None:
            printer._thread_local.tab_id = "nwt"
            printer.start_recording()
            printer._current_block_type = "thinking"
            printer.broadcast({"type": "thinking_start", "tabId": "nwt"})
            barrier.wait()
            barrier2.wait()
            delta_type = (
                "thinking_delta"
                if printer._current_block_type == "thinking"
                else "text_delta"
            )
            printer.broadcast({"type": delta_type, "text": "reasoning...", "tabId": "nwt"})
            printer._current_block_type = ""
            printer.broadcast({"type": "thinking_end", "tabId": "nwt"})
            nwt_events.extend(printer.stop_recording())

        def wt_task() -> None:
            printer._thread_local.tab_id = "wt"
            printer.start_recording()
            printer._current_block_type = "tool_use"
            barrier.wait()
            printer.stop_recording()
            printer.reset()
            barrier2.wait()

        t_nwt = threading.Thread(target=non_wt_task, daemon=True)
        t_wt = threading.Thread(target=wt_task, daemon=True)
        t_nwt.start()
        t_wt.start()
        t_nwt.join(timeout=5)
        t_wt.join(timeout=5)

        types = [e["type"] for e in nwt_events]
        assert types == ["thinking_start", "thinking_delta", "thinking_end"], (
            f"Expected thinking events, got {types}. "
            "Wt task's reset() corrupted non-wt task's stream state."
        )


class TestBashStreamedCrossContamination(unittest.TestCase):
    """Bash ``streamed`` flag must be per-tab.

    Without per-tab bash state, task A's ``bash_stream`` sets
    ``streamed = True`` on the shared ``_BashState``.  Task B's
    ``tool_result`` reads it as ``True`` and suppresses the result
    content (empty string instead of actual content).
    """

    def test_streamed_flag_isolated_between_tabs(self) -> None:
        """Task A streaming bash does not set task B's streamed flag."""
        printer = BaseBrowserPrinter()
        barrier = threading.Barrier(2, timeout=5)
        barrier2 = threading.Barrier(2, timeout=5)
        results: dict[str, bool] = {}

        def task_a() -> None:
            printer._thread_local.tab_id = "A"
            printer.start_recording()
            with printer._bash_lock:
                printer._bash_state.streamed = True
            barrier.wait()
            barrier2.wait()
            with printer._bash_lock:
                results["A_streamed"] = printer._bash_state.streamed

        def task_b() -> None:
            printer._thread_local.tab_id = "B"
            printer.start_recording()
            barrier.wait()
            with printer._bash_lock:
                results["B_streamed"] = printer._bash_state.streamed
            barrier2.wait()

        t_a = threading.Thread(target=task_a, daemon=True)
        t_b = threading.Thread(target=task_b, daemon=True)
        t_a.start()
        t_b.start()
        t_a.join(timeout=5)
        t_b.join(timeout=5)

        assert results["A_streamed"] is True, "Task A set streamed=True"
        assert results["B_streamed"] is False, (
            "Task B's streamed should be False (default), "
            "not contaminated by task A"
        )


class TestBashBufferCrossContamination(unittest.TestCase):
    """Bash buffer must be per-tab.

    Without per-tab bash state, two tasks' bash output gets mixed in
    the shared buffer, producing garbled system_output events.
    """

    def test_buffers_isolated_between_tabs(self) -> None:
        """Task A's bash buffer content does not appear in task B's buffer."""
        printer = BaseBrowserPrinter()
        barrier = threading.Barrier(2, timeout=5)
        results: dict[str, list[str]] = {}

        def task_a() -> None:
            printer._thread_local.tab_id = "A"
            with printer._bash_lock:
                printer._bash_state.buffer.append("task-A-output")
            barrier.wait()
            with printer._bash_lock:
                results["A_buffer"] = list(printer._bash_state.buffer)

        def task_b() -> None:
            printer._thread_local.tab_id = "B"
            with printer._bash_lock:
                printer._bash_state.buffer.append("task-B-output")
            barrier.wait()
            with printer._bash_lock:
                results["B_buffer"] = list(printer._bash_state.buffer)

        t_a = threading.Thread(target=task_a, daemon=True)
        t_b = threading.Thread(target=task_b, daemon=True)
        t_a.start()
        t_b.start()
        t_a.join(timeout=5)
        t_b.join(timeout=5)

        assert results["A_buffer"] == ["task-A-output"], (
            f"Task A buffer should only have its output, got {results['A_buffer']}"
        )
        assert results["B_buffer"] == ["task-B-output"], (
            f"Task B buffer should only have its output, got {results['B_buffer']}"
        )


class TestBashGenerationInterference(unittest.TestCase):
    """Bash generation counter must be per-tab.

    Without per-tab bash state, task B's ``reset()`` increments the
    shared generation counter, causing task A's legitimate timer flush
    to discard its data (generation mismatch).
    """

    def test_reset_on_tab_b_does_not_kill_tab_a_flush(self) -> None:
        """Task B's reset() does not increment task A's generation counter."""
        printer = BaseBrowserPrinter()
        barrier = threading.Barrier(2, timeout=5)
        barrier2 = threading.Barrier(2, timeout=5)
        results: dict[str, int] = {}

        def task_a() -> None:
            printer._thread_local.tab_id = "A"
            with printer._bash_lock:
                gen_before = printer._bash_state.generation
            results["A_gen_before"] = gen_before
            barrier.wait()
            barrier2.wait()
            with printer._bash_lock:
                results["A_gen_after"] = printer._bash_state.generation

        def task_b() -> None:
            printer._thread_local.tab_id = "B"
            barrier.wait()
            printer.reset()
            barrier2.wait()

        t_a = threading.Thread(target=task_a, daemon=True)
        t_b = threading.Thread(target=task_b, daemon=True)
        t_a.start()
        t_b.start()
        t_a.join(timeout=5)
        t_b.join(timeout=5)

        assert results["A_gen_before"] == results["A_gen_after"], (
            f"Task A's generation changed from {results['A_gen_before']} "
            f"to {results['A_gen_after']} — task B's reset() interfered"
        )


class TestBashTimerInterference(unittest.TestCase):
    """Bash timer must be per-tab.

    Without per-tab bash state, task B's ``reset()`` cancels task A's
    pending flush timer, causing task A's buffered output to be lost.
    """

    def test_reset_on_tab_b_does_not_cancel_tab_a_timer(self) -> None:
        """Task B's reset() does not cancel task A's pending timer."""
        printer = BaseBrowserPrinter()
        barrier = threading.Barrier(2, timeout=5)
        barrier2 = threading.Barrier(2, timeout=5)
        results: dict[str, bool] = {}

        def task_a() -> None:
            printer._thread_local.tab_id = "A"
            with printer._bash_lock:
                t = threading.Timer(999, lambda: None)
                printer._bash_state.timer = t
            barrier.wait()
            barrier2.wait()
            with printer._bash_lock:
                results["A_timer_alive"] = printer._bash_state.timer is not None
                if printer._bash_state.timer is not None:
                    printer._bash_state.timer.cancel()
                    printer._bash_state.timer = None

        def task_b() -> None:
            printer._thread_local.tab_id = "B"
            barrier.wait()
            printer.reset()
            barrier2.wait()

        t_a = threading.Thread(target=task_a, daemon=True)
        t_b = threading.Thread(target=task_b, daemon=True)
        t_a.start()
        t_b.start()
        t_a.join(timeout=5)
        t_b.join(timeout=5)

        assert results["A_timer_alive"] is True, (
            "Task A's timer should still be alive after task B's reset()"
        )


class TestFlushBashTOCTOU(unittest.TestCase):
    """``_flush_bash`` TOCTOU: broadcast must happen inside _bash_lock.

    Without the fix, after the generation check passes and the lock is
    released, ``reset()`` + ``start_recording()`` can slip in before
    ``broadcast()``, causing stale bash text to leak into the new
    recording.

    This test verifies the fix by simulating the exact interleaving
    with manual thread synchronization.
    """

    def test_stale_text_does_not_leak_into_new_recording(self) -> None:
        """Stale bash text from old turn does not appear in new recording.

        Interleaving:
        1. Timer captures text + generation (first lock)
        2. Timer releases first lock
        3. Timer acquires second lock, generation check passes
        4. (Without fix: timer releases lock, reset+start_recording, broadcast leaks)
        5. (With fix: broadcast inside lock, no window for reset)
        """
        printer = BaseBrowserPrinter()
        printer._thread_local.tab_id = "tab1"

        with printer._bash_lock:
            printer._bash_state.buffer.append("stale-from-old-turn")

        captured = threading.Event()
        proceed = threading.Event()

        def manual_flush() -> None:
            printer._thread_local.tab_id = "tab1"
            with printer._bash_lock:
                bs = printer._bash_state
                gen = bs.generation
                text = "".join(bs.buffer) if bs.buffer else ""
                bs.buffer.clear()
                bs.last_flush = 0.0
            captured.set()
            proceed.wait(timeout=5)
            if text:
                with printer._bash_lock:
                    if printer._bash_state.generation != gen:
                        return
                    printer.broadcast({"type": "system_output", "text": text})

        t = threading.Thread(target=manual_flush, daemon=True)
        t.start()

        captured.wait(timeout=5)

        printer.reset()
        printer.start_recording()

        proceed.set()
        t.join(timeout=5)

        events = printer.stop_recording()
        stale = [e for e in events if e.get("type") == "system_output"]
        assert len(stale) == 0, (
            f"Stale bash text leaked into new recording: {stale}"
        )

    def test_structural_broadcast_inside_lock(self) -> None:
        """Verify broadcast is called inside the _bash_lock in _flush_bash."""
        import inspect
        import textwrap

        src = inspect.getsource(BaseBrowserPrinter._flush_bash)
        src = textwrap.dedent(src)
        gen_check_idx = src.index("self._bash_state.generation != gen")
        broadcast_idx = src.index("self.broadcast(")
        second_lock_start = src.index("with self._bash_lock:", src.index("bs.last_flush"))
        lines = src[second_lock_start:].split("\n")
        gen_line = None
        broadcast_line = None
        for i, line in enumerate(lines):
            if "self._bash_state.generation != gen" in line:
                gen_line = i
            if "self.broadcast(" in line:
                broadcast_line = i
        assert gen_line is not None and broadcast_line is not None
        assert broadcast_line > gen_line, (
            "broadcast must come after generation check in the same lock scope"
        )
        assert gen_check_idx < broadcast_idx


class TestCleanupTab(unittest.TestCase):
    """``cleanup_tab`` removes per-tab state to prevent unbounded memory growth."""

    def test_cleanup_removes_bash_state(self) -> None:
        """After cleanup_tab, the tab's _BashState is removed from _bash_states."""
        printer = BaseBrowserPrinter()
        printer._thread_local.tab_id = "tab1"
        with printer._bash_lock:
            printer._bash_state.buffer.append("data")
        assert "tab1" in printer._bash_states
        printer.cleanup_tab("tab1")
        assert "tab1" not in printer._bash_states

    def test_cleanup_cancels_pending_timer(self) -> None:
        """cleanup_tab cancels any pending bash flush timer for the tab."""
        printer = BaseBrowserPrinter()
        printer._thread_local.tab_id = "tab1"
        t = threading.Timer(999, lambda: None)
        t.start()
        with printer._bash_lock:
            printer._bash_state.timer = t
        assert t.is_alive()
        printer.cleanup_tab("tab1")
        t.join(timeout=2)
        assert not t.is_alive(), "Timer should have been cancelled by cleanup_tab"

    def test_cleanup_removes_recording(self) -> None:
        """After cleanup_tab, the tab's recording is removed from _recordings."""
        printer = BaseBrowserPrinter()
        printer._thread_local.tab_id = "tab1"
        printer.start_recording()
        assert "tab1" in printer._recordings
        printer.cleanup_tab("tab1")
        assert "tab1" not in printer._recordings

    def test_cleanup_does_not_affect_other_tabs(self) -> None:
        """cleanup_tab for one tab leaves other tabs' state intact."""
        printer = BaseBrowserPrinter()

        def setup_tab(tab_id: str) -> None:
            printer._thread_local.tab_id = tab_id
            with printer._bash_lock:
                printer._bash_state.buffer.append(f"data-{tab_id}")
            printer.start_recording()
            with printer._lock:
                printer._record_event({"type": "text_delta", "text": "x", "tabId": tab_id})

        barrier = threading.Barrier(2, timeout=5)

        def thread_setup(tab_id: str) -> None:
            setup_tab(tab_id)
            barrier.wait()

        t1 = threading.Thread(target=thread_setup, args=("tab1",), daemon=True)
        t2 = threading.Thread(target=thread_setup, args=("tab2",), daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert "tab1" in printer._bash_states
        assert "tab2" in printer._bash_states
        assert "tab1" in printer._recordings
        assert "tab2" in printer._recordings

        printer.cleanup_tab("tab1")

        assert "tab1" not in printer._bash_states
        assert "tab2" in printer._bash_states
        assert "tab1" not in printer._recordings
        assert "tab2" in printer._recordings
        assert printer._bash_states["tab2"].buffer == ["data-tab2"]
        assert len(printer._recordings["tab2"]) == 1

    def test_cleanup_nonexistent_tab_is_noop(self) -> None:
        """cleanup_tab for a tab that doesn't exist does not raise."""
        printer = BaseBrowserPrinter()
        printer.cleanup_tab("nonexistent")
        assert "nonexistent" not in printer._bash_states
        assert "nonexistent" not in printer._recordings

    def test_cleanup_after_task_completes_frees_state(self) -> None:
        """Simulates a full lifecycle: task runs, completes, tab is closed.

        After cleanup, no per-tab entries remain in _bash_states or _recordings.
        """
        printer = BaseBrowserPrinter()
        printer._thread_local.tab_id = "tab1"

        printer.start_recording()
        with printer._bash_lock:
            printer._bash_state.buffer.append("output")
            printer._bash_state.streamed = True
        printer.broadcast({"type": "text_delta", "text": "hello"})
        printer.stop_recording()
        printer.reset()

        assert "tab1" in printer._bash_states

        printer.cleanup_tab("tab1")
        assert "tab1" not in printer._bash_states
        assert "tab1" not in printer._recordings


class TestCloseTabServer(unittest.TestCase):
    """_close_tab on VSCodeServer cleans up _tab_states, printer state,
    and _persist_agents."""

    @staticmethod
    def _make_server() -> Any:
        import os

        os.environ.setdefault("KISS_WORKDIR", "/tmp")
        from kiss.agents.vscode.server import VSCodeServer
        return VSCodeServer()

    def test_close_tab_removes_tab_state(self) -> None:
        """_close_tab removes the tab from _tab_states."""
        server = self._make_server()
        server._get_tab("tab1")
        assert "tab1" in server._tab_states
        server._close_tab("tab1")
        assert "tab1" not in server._tab_states

    def test_close_tab_cleans_printer_bash_states(self) -> None:
        """_close_tab removes the tab's bash state from the printer."""
        server = self._make_server()
        server._get_tab("tab1")
        server.printer._thread_local.tab_id = "tab1"
        with server.printer._bash_lock:
            server.printer._bash_state.buffer.append("data")
        assert "tab1" in server.printer._bash_states
        server._close_tab("tab1")
        assert "tab1" not in server.printer._bash_states

    def test_close_tab_cleans_printer_recordings(self) -> None:
        """_close_tab removes the tab's recording from the printer."""
        server = self._make_server()
        server._get_tab("tab1")
        server.printer._thread_local.tab_id = "tab1"
        server.printer.start_recording()
        assert "tab1" in server.printer._recordings
        server._close_tab("tab1")
        assert "tab1" not in server.printer._recordings

    def test_close_tab_cleans_persist_agents(self) -> None:
        """_close_tab removes the tab's entry from _persist_agents."""
        server = self._make_server()
        tab = server._get_tab("tab1")
        server.printer._persist_agents["tab1"] = tab.agent
        server._close_tab("tab1")
        assert "tab1" not in server.printer._persist_agents

    def test_close_tab_skips_active_task(self) -> None:
        """_close_tab does not remove a tab that has an active task."""
        server = self._make_server()
        tab = server._get_tab("tab1")
        tab.is_task_active = True
        server._close_tab("tab1")
        assert "tab1" in server._tab_states

    def test_close_tab_skips_merging_tab(self) -> None:
        """_close_tab does not remove a tab that is in a merge review."""
        server = self._make_server()
        tab = server._get_tab("tab1")
        tab.is_merging = True
        server._close_tab("tab1")
        assert "tab1" in server._tab_states

    def test_close_tab_nonexistent_is_noop(self) -> None:
        """_close_tab for an unknown tab_id does not raise."""
        server = self._make_server()
        server._close_tab("nonexistent")

    def test_cmd_close_tab_dispatches(self) -> None:
        """The closeTab command dispatches to _close_tab."""
        server = self._make_server()
        server._get_tab("tab1")
        assert "tab1" in server._tab_states
        server._handle_command({"type": "closeTab", "tabId": "tab1"})
        assert "tab1" not in server._tab_states


if __name__ == "__main__":
    unittest.main()
