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
    def test_immediate_flush_when_interval_elapsed(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._bash_last_flush = 0.0
        p.print("line1\n", type="bash_stream")
        events = _drain(q)
        assert len(events) == 1
        assert events[0] == {"type": "system_output", "text": "line1\n"}
        assert p._bash_flush_timer is None

    def test_deferred_flush_when_within_interval(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._bash_last_flush = time.monotonic()
        p.print("line1\n", type="bash_stream")
        events = _drain(q)
        assert len(events) == 0
        assert p._bash_flush_timer is not None
        time.sleep(0.2)
        events = _drain(q)
        assert len(events) == 1
        assert events[0] == {"type": "system_output", "text": "line1\n"}
        assert p._bash_flush_timer is None

    def test_deferred_flush_batches_multiple_lines(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._bash_last_flush = time.monotonic()
        p.print("line1\n", type="bash_stream")
        p.print("line2\n", type="bash_stream")
        p.print("line3\n", type="bash_stream")
        events = _drain(q)
        assert len(events) == 0
        time.sleep(0.2)
        events = _drain(q)
        assert len(events) == 1
        assert events[0] == {"type": "system_output", "text": "line1\nline2\nline3\n"}

    def test_timer_not_duplicated_for_multiple_buffered_lines(self):
        p = BaseBrowserPrinter()
        _subscribe(p)
        p._bash_last_flush = time.monotonic()
        p.print("a\n", type="bash_stream")
        timer1 = p._bash_flush_timer
        assert timer1 is not None
        p.print("b\n", type="bash_stream")
        assert p._bash_flush_timer is timer1

    def test_explicit_flush_cancels_timer(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._bash_last_flush = time.monotonic()
        p.print("line1\n", type="bash_stream")
        assert p._bash_flush_timer is not None
        p._flush_bash()
        assert p._bash_flush_timer is None
        events = _drain(q)
        assert len(events) == 1
        assert events[0]["text"] == "line1\n"
        time.sleep(0.2)
        extra = _drain(q)
        assert len(extra) == 0

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

    def test_tool_result_flushes_and_cancels_timer(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._bash_last_flush = time.monotonic()
        p.print("buffered\n", type="bash_stream")
        assert p._bash_flush_timer is not None
        p.print("OK", type="tool_result", is_error=False)
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

    def test_flush_empty_buffer_is_noop(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._flush_bash()
        assert _drain(q) == []

    def test_rapid_lines_then_deferred_flush(self):
        p = BaseBrowserPrinter()
        q = _subscribe(p)
        p._bash_last_flush = time.monotonic()
        for i in range(20):
            p.print(f"line{i}\n", type="bash_stream")
        events = _drain(q)
        assert len(events) == 0
        time.sleep(0.2)
        events = _drain(q)
        assert len(events) == 1
        text = events[0]["text"]
        for i in range(20):
            assert f"line{i}\n" in text
