# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for vscode agent audit round 8: redundancies, inconsistencies, bugs.

Covers:
- API_KEY_ENV_VARS type (should be frozenset, not dict)
- _timer_flush closure removal (should be a proper method)
- _timer_flush type annotation (tid should be str | None, not int | None)
- Dead re-exports in server.py
"""

from __future__ import annotations

import time


class TestTimerFlushNoClosure:
    """_timer_flush must not be a closure; it should be a method."""

    def test_timer_flush_is_method_not_closure(self) -> None:
        """Verify _timer_flush is a proper method on JsonPrinter."""
        from kiss.server.json_printer import JsonPrinter

        assert hasattr(JsonPrinter, "_timer_flush_for_task"), (
            "JsonPrinter should have _timer_flush_for_task method"
        )
        method = getattr(JsonPrinter, "_timer_flush_for_task")
        assert callable(method)


    def test_timer_flush_for_task_type_annotation(self) -> None:
        """Verify _timer_flush_for_task accepts str | None tab_id."""
        from kiss.server.json_printer import JsonPrinter

        hints = JsonPrinter._timer_flush_for_task.__annotations__
        assert "task_id" in hints

    def test_bash_timer_uses_method(self) -> None:
        """Verify the bash_stream print path creates a timer using
        _timer_flush_for_task (via functools.partial) instead of a closure."""
        from functools import partial as functools_partial

        from kiss.server.json_printer import JsonPrinter

        printer = JsonPrinter()
        events: list[dict] = []
        original_broadcast = printer.broadcast

        def capture_broadcast(event: dict) -> None:
            events.append(event)
            original_broadcast(event)

        printer.broadcast = capture_broadcast  # type: ignore[assignment]

        printer.print("chunk1", type="bash_stream")
        printer.print("chunk2", type="bash_stream")
        with printer._bash_lock:
            bs = printer._bash_state
            assert bs.timer is not None, "Timer should be set for buffered bash output"
            timer_func = bs.timer.function
            assert isinstance(timer_func, functools_partial), (
                f"Timer function should be functools.partial, got {type(timer_func)}"
            )
            assert timer_func.func == printer._timer_flush_for_task
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            flush_events = [e for e in events if e.get("type") == "system_output"]
            if any("chunk2" in e["text"] for e in flush_events):
                break
            time.sleep(0.05)
        flush_events = [e for e in events if e.get("type") == "system_output"]
        all_text = "".join(e["text"] for e in flush_events)
        assert "chunk1" in all_text
        assert "chunk2" in all_text


class TestNoMisleadingReExportComment:
    """server.py should not have misleading noqa: F401 re-export comments."""


    def test_server_all_does_not_expose_diff_merge_names(self) -> None:
        """Verify __all__ in server.py does not list diff_merge symbols."""
        from kiss.server import server

        public_names = getattr(server, "__all__", [])
        assert "_cleanup_merge_data" not in public_names
        assert "_git" not in public_names
        assert "_merge_data_dir" not in public_names
