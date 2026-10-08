# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A timer flush of buffered bash output lands BEFORE the event that ends the command.

``JsonPrinter`` buffers ``bash_stream`` fragments and flushes them from
a 0.1 s timer thread.  Before the fix, ``_flush_bash`` copied and
cleared the buffer under ``_bash_lock`` and only afterwards took the
per-task ``flush_lock`` to broadcast.  The agent thread's own flush
before a ``tool_result`` / ``tool_call`` synchronised only through
``_bash_lock``: it found the buffer already empty, did not wait, and
broadcast ``tool_result`` while the timer thread was still about to
send the command's last ``system_output`` — the panel then showed
orphaned output after an empty result.  Now capture and broadcast are
one critical section under ``flush_lock``, so the agent thread blocks
until the in-flight timer broadcast is out.

The tests gate the timer's broadcast with a real ``JsonPrinter``
subclass (no mocks) and assert both the blocking and the final order.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from kiss.server.json_printer import JsonPrinter


class _GatedPrinter(JsonPrinter):
    """Printer whose ``system_output`` broadcast waits for a release signal."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self.in_broadcast = threading.Event()
        self.release = threading.Event()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event*; hold a ``system_output`` send until released."""
        if event.get("type") == "system_output":
            self.in_broadcast.set()
            self.release.wait(timeout=5)
        self.events.append(event)


def _buffer_fragment_with_armed_timer(printer: _GatedPrinter, task: str) -> None:
    """Print one bash fragment so it is buffered and the flush timer is armed."""
    printer._thread_local.task_id = task
    with printer._bash_lock:
        printer._bash_state.last_flush = time.monotonic()
    printer.print("trailing output\n", type="bash_stream")
    with printer._bash_lock:
        assert printer._bash_states[task].timer is not None


def _emit_tool_result(printer: _GatedPrinter) -> None:
    """Print an empty Bash ``tool_result`` (its output was streamed)."""
    printer.print("", type="tool_result", tool_name="Bash")


def _emit_tool_call(printer: _GatedPrinter) -> None:
    """Print the next ``tool_call``."""
    printer.print("Bash", type="tool_call", tool_input={"command": "ls"})


def _reset_and_record(printer: _GatedPrinter) -> None:
    """Start a new turn: ``reset()`` then ``start_recording()``."""
    printer.reset()
    printer.start_recording()


def _on_task_thread(
    printer: _GatedPrinter, task: str, emit: Callable[[_GatedPrinter], None]
) -> None:
    """Bind *task* as the thread's task id and call ``emit(printer)``."""
    printer._thread_local.task_id = task
    emit(printer)


def _run_gated(
    printer: _GatedPrinter, task: str, emit: Callable[[_GatedPrinter], None]
) -> list[str]:
    """Let the timer flush start broadcasting, run *emit* on a thread, return event types.

    Asserts that *emit* is still blocked while the timer's broadcast is
    held (before the fix it completed immediately).
    """
    assert printer.in_broadcast.wait(timeout=5), "timer flush never fired"
    emitter = threading.Thread(target=_on_task_thread, args=(printer, task, emit), daemon=True)
    emitter.start()
    emitter.join(timeout=0.3)
    try:
        assert emitter.is_alive(), (
            "the agent thread's flush did not wait for the in-flight timer broadcast"
        )
    finally:
        printer.release.set()
    emitter.join(timeout=5)
    assert not emitter.is_alive()
    return [e["type"] for e in printer.events]


def test_tool_result_waits_for_inflight_timer_flush() -> None:
    printer = _GatedPrinter()
    _buffer_fragment_with_armed_timer(printer, "flush-order-result")
    types = _run_gated(printer, "flush-order-result", _emit_tool_result)
    assert types == ["system_output", "tool_result"], types
    assert printer.events[0]["text"] == "trailing output\n"


def test_tool_call_waits_for_inflight_timer_flush() -> None:
    printer = _GatedPrinter()
    _buffer_fragment_with_armed_timer(printer, "flush-order-call")
    types = _run_gated(printer, "flush-order-call", _emit_tool_call)
    assert types[:2] == ["system_output", "text_end"], types
    assert "tool_call" in types


def test_reset_waits_for_inflight_flush_and_new_turn_sees_no_stale_text() -> None:
    """``reset()`` blocks behind a broadcasting flush; the next turn records nothing stale."""
    printer = _GatedPrinter()
    _buffer_fragment_with_armed_timer(printer, "flush-order-reset")
    types = _run_gated(printer, "flush-order-reset", _reset_and_record)
    assert types == ["system_output"], types
    printer._thread_local.task_id = "flush-order-reset"
    assert printer.stop_recording() == []
