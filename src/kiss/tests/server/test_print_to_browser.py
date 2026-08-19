# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for JsonPrinter.

Tests verify correctness and accuracy of all browser streaming logic.
Uses real objects with duck-typed attributes (SimpleNamespace) as
message inputs and real queue subscribers.
"""

import unittest
from types import SimpleNamespace

from kiss.server.json_printer import (
    _DISPLAY_EVENT_TYPES,
    JsonPrinter,
    _coalesce_events,
)


def _start(printer: JsonPrinter) -> None:
    printer.start_recording()


def _drain(printer: JsonPrinter) -> list[dict]:
    return printer.stop_recording()


class TestCoalesceEvents(unittest.TestCase):
    def test_no_merge_missing_text_in_current(self):
        events = [
            {"type": "thinking_delta", "text": "A"},
            {"type": "thinking_delta"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 2


class TestHandleMessage(unittest.TestCase):
    def test_subtype_not_tool_output(self):
        p = JsonPrinter()
        _start(p)
        msg = SimpleNamespace(subtype="other", data={"content": "x"})
        p.print(msg, type="message")
        assert _drain(p) == []

    def test_unknown_message_type_no_crash(self):
        p = JsonPrinter()
        _start(p)
        msg = SimpleNamespace(unknown_attr="value")
        p.print(msg, type="message")
        assert _drain(p) == []


class TestDisplayEventTypes(unittest.TestCase):
    def test_expected_types_present(self):
        expected = {
            "clear", "thinking_start", "thinking_delta", "thinking_end",
            "text_delta", "text_end", "tool_call", "tool_result",
            "system_output", "result", "system_prompt", "prompt",
            "task_done", "task_error", "task_stopped", "task_interrupted",
            "followup_suggestion", "autocommit_done", "warning",
            "usage_info", "task_settings",
        }
        assert _DISPLAY_EVENT_TYPES == expected


class TestUsageInfoReplayed(unittest.TestCase):
    """``usage_info`` must survive into replayed transcripts.

    The chat header's tokens/cost metrics are populated only by
    ``usage_info`` and ``result`` events.  A transcript replayed while
    the task is still running (or after a stop/error, when no
    ``result`` event exists) would show an empty tokens/cost header if
    recordings filtered ``usage_info`` out.
    """

    def test_usage_info_in_stopped_recording(self):
        p = JsonPrinter()
        p._thread_local.task_id = "t-usage"
        _start(p)
        p.print(
            "Steps: 3/100, Total tokens: 1,234, Budget: $0.5000/$10.00, ",
            type="usage_info",
            total_tokens=1234,
            cost="$0.5000",
            total_steps=3,
        )
        events = _drain(p)
        usage = [e for e in events if e.get("type") == "usage_info"]
        assert len(usage) == 1
        assert usage[0]["total_tokens"] == 1234
        assert usage[0]["cost"] == "$0.5000"
        assert usage[0]["total_steps"] == 3

    def test_usage_info_in_peeked_recording(self):
        p = JsonPrinter()
        p._thread_local.task_id = "t-usage-peek"
        _start(p)
        p.print(
            "Steps: 7/100, Total tokens: 42, Budget: $0.0100/$10.00, ",
            type="usage_info",
            total_tokens=42,
            cost="$0.0100",
            total_steps=7,
        )
        events = p.peek_recording_for_task("t-usage-peek")
        usage = [e for e in events if e.get("type") == "usage_info"]
        assert len(usage) == 1
        assert usage[0]["total_tokens"] == 42
        p.stop_recording()


if __name__ == "__main__":
    unittest.main()
