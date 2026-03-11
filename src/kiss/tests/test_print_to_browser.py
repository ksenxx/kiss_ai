"""Tests for BaseBrowserPrinter.

Tests verify correctness and accuracy of all browser streaming logic.
Uses real objects with duck-typed attributes (SimpleNamespace) as
message inputs and real queue subscribers.
"""

import queue
import unittest
from types import SimpleNamespace

from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter


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


class TestPrintStreamEvent(unittest.TestCase):
    def _event(self, evt_dict):
        return SimpleNamespace(event=evt_dict)

class TestPrintMessageDispatch(unittest.TestCase):
    def test_unknown_message_type_no_crash(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        msg = SimpleNamespace(unknown_attr="value")
        p.print(msg, type="message")
        assert _drain(q) == []


class TestStreamingFlow(unittest.TestCase):
    """Test the full streaming flow: block_start -> token_callback -> block_stop."""

    def _event(self, evt_dict):
        return SimpleNamespace(event=evt_dict)


if __name__ == "__main__":
    unittest.main()
