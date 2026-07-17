# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 4: backend ``warning`` events must survive recording/replay.

Iteration 3 made the live frontend render backend ``warning`` events
(``WorktreeSorcarAgent._flush_warnings`` broadcasts
``{type: 'warning', message: ...}``; main.js renders an amber banner and
types.ts declares the event).  But ``warning`` was never added to
``_DISPLAY_EVENT_TYPES`` in ``json_printer.py``, so:

* ``stop_recording()`` / ``peek_recording()`` filter warnings out of the
  recorded event list that ``ChatSorcarAgent`` saves for the task, and
* ``_persist_event()`` refuses to enqueue them for live persistence,

meaning a warning the user saw live silently vanishes when the task is
reopened in a viewer tab, replayed from chat history, or replayed in
demo mode.  The contract decided here: warnings are display events and
must be recorded/persisted like any other rendered event.
"""

from __future__ import annotations

from unittest import TestCase

from kiss.server.json_printer import JsonPrinter


class TestWarningSurvivesRecording(TestCase):
    """Warnings broadcast during a task must appear in the saved recording."""

    def _make_printer(self, task_id: str) -> JsonPrinter:
        printer = JsonPrinter()
        printer._thread_local.task_id = task_id
        return printer

    def test_stop_recording_keeps_warning(self) -> None:
        printer = self._make_printer("t-warn-1")
        printer.start_recording()
        printer.broadcast({"type": "text_delta", "text": "working..."})
        printer.broadcast(
            {"type": "warning", "message": "Could not checkout 'main'"},
        )
        printer.broadcast({"type": "text_end"})
        events = printer.stop_recording()
        warnings = [e for e in events if e.get("type") == "warning"]
        self.assertEqual(
            len(warnings), 1,
            "a 'warning' event the user saw live was dropped from the "
            f"saved recording (got event types {[e.get('type') for e in events]})",
        )
        self.assertEqual(
            warnings[0].get("message"), "Could not checkout 'main'",
        )

    def test_peek_recording_keeps_warning(self) -> None:
        """Crash-recovery snapshots must include warnings too."""
        printer = self._make_printer("t-warn-2")
        printer.start_recording()
        printer.broadcast({"type": "warning", "message": "stash pop failed"})
        snapshot = printer.peek_recording()
        self.assertEqual(
            [e.get("type") for e in snapshot], ["warning"],
            "peek_recording filtered out the warning event",
        )

    def test_warnings_not_coalesced_with_deltas(self) -> None:
        """A warning between two text deltas must not break/merge them."""
        printer = self._make_printer("t-warn-3")
        printer.start_recording()
        printer.broadcast({"type": "text_delta", "text": "a"})
        printer.broadcast({"type": "warning", "message": "w"})
        printer.broadcast({"type": "text_delta", "text": "b"})
        events = printer.stop_recording()
        self.assertEqual(
            [e.get("type") for e in events],
            ["text_delta", "warning", "text_delta"],
        )
        self.assertEqual(events[0].get("text"), "a")
        self.assertEqual(events[2].get("text"), "b")
