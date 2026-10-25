# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for server audit round 8: redundancies, inconsistencies, bugs.

Covers:
- the bash flush timer runs ``flush_bash_output`` keyed by the task
- Dead re-exports in server.py
"""

from __future__ import annotations

import time


class TestBashFlushTimer:
    """The bash flush timer flushes the buffering task's output."""

    def test_bash_timer_flushes_buffered_output(self) -> None:
        """The timer armed by a buffered ``bash_stream`` flushes it within its delay."""
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
            assert timer_func.func == printer.flush_bash_output
            assert timer_func.args == ("",)
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
