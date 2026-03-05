"""Tests for bash_stream deferred flush in BaseBrowserPrinter.

Verifies that buffered bash output is flushed via a deferred timer when
lines arrive faster than the 0.1s flush interval, rather than waiting
until the next tool_call/tool_result event.
"""

import queue
import time

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


class TestBashStreamDeferredFlush:

    def test_tool_call_flushes_and_cancels_timer(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._bash_last_flush = time.monotonic()
        p.print("buffered\n", type="bash_stream")
        assert p._bash_flush_timer is not None
        p.print("Bash", type="tool_call", tool_input={"command": "echo hi"})
        assert p._bash_flush_timer is None
        events = _drain(q)
        sys_events = [e for e in events if e["type"] == "system_output"]
        assert len(sys_events) == 1
        assert sys_events[0]["text"] == "buffered\n"

    def test_reset_cancels_timer(self):
        p = BaseBrowserPrinter()
        _subscribe(p)
        p._bash_last_flush = time.monotonic()
        p.print("line\n", type="bash_stream")
        assert p._bash_flush_timer is not None
        p.reset()
        assert p._bash_flush_timer is None
        assert p._bash_buffer == []
