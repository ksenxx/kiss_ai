"""Tests for BaseBrowserPrinter.

Tests verify correctness and accuracy of all browser streaming logic.
Uses real objects with duck-typed attributes (SimpleNamespace) as
message inputs and real queue subscribers.
"""

import asyncio
import queue
import threading
import time
import unittest
from types import SimpleNamespace

from kiss.agents.sorcar.browser_ui import (
    _DISPLAY_EVENT_TYPES,
    BaseBrowserPrinter,
    _coalesce_events,
    find_free_port,
)


def _subscribe(printer: BaseBrowserPrinter) -> queue.Queue:
    q: queue.Queue = queue.Queue()
    printer._clients.append(q)
    return q


def _drain(q: queue.Queue) -> list[dict]:
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            break
    return events


class TestCoalesceEvents(unittest.TestCase):
    def test_no_merge_missing_text_in_current(self):
        events = [
            {"type": "thinking_delta", "text": "A"},
            {"type": "thinking_delta"},
        ]
        result = _coalesce_events(events)
        assert len(result) == 2


class TestPrintSystemPrompt(unittest.TestCase):
    def test_system_prompt_broadcasts_event(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p.print("You are a helpful agent.", type="system_prompt")
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["type"] == "system_prompt"
        assert events[0]["text"] == "You are a helpful agent."


class TestHandleMessage(unittest.TestCase):
    def test_subtype_not_tool_output(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(subtype="other", data={"content": "x"})
        p.print(msg, type="message")
        assert _drain(q) == []

    def test_unknown_message_type_no_crash(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(unknown_attr="value")
        p.print(msg, type="message")
        assert _drain(q) == []


class TestDisplayEventTypes(unittest.TestCase):
    def test_expected_types_present(self):
        expected = {
            "clear", "thinking_start", "thinking_delta", "thinking_end",
            "text_delta", "text_end", "tool_call", "tool_result",
            "system_output", "result", "system_prompt", "prompt",
            "usage_info", "task_done", "task_error", "task_stopped",
            "followup_suggestion",
        }
        assert _DISPLAY_EVENT_TYPES == expected


if __name__ == "__main__":
    unittest.main()
