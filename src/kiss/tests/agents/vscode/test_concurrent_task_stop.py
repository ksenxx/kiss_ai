"""Integration tests: stopping one task while another runs concurrently.

Reproduces the bug where stopping a task corrupts the concurrent task's
printer state (stream parsing and recording), which on the frontend
manifests as panels collapsing or going blank.

Covers both directions:
- Stopping a non-worktree task while a worktree task runs
- Stopping a worktree task while a non-worktree task runs

Root cause: ``BaseBrowserPrinter``'s stream state (``_current_block_type``,
``_tool_name``, ``_tool_json_buffer``) and recording (``_recording``) were
shared instance attributes.  When one task's cleanup called ``reset()``
or ``stop_recording()``, it destroyed the other task's in-flight state.
"""

from __future__ import annotations

import threading
import unittest

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


if __name__ == "__main__":
    unittest.main()
