"""Tests for StreamEventParser and its use in ConsolePrinter / BaseBrowserPrinter."""

from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace
from typing import Any

from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
from kiss.core.print_to_console import ConsolePrinter
from kiss.core.printer import StreamEventParser


def _evt(d: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(event=d)


def _block_start(block_type: str, **kw: Any) -> SimpleNamespace:
    block = {"type": block_type, **kw}
    return _evt({"type": "content_block_start", "content_block": block})


def _delta(delta_type: str, **kw: Any) -> SimpleNamespace:
    return _evt({"type": "content_block_delta", "delta": {"type": delta_type, **kw}})


def _block_stop() -> SimpleNamespace:
    return _evt({"type": "content_block_stop"})


# ── StreamEventParser base class tests ──


class _TrackingParser(StreamEventParser):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, Any]] = []

    def _on_thinking_start(self) -> None:
        self.calls.append(("thinking_start", None))

    def _on_thinking_end(self) -> None:
        self.calls.append(("thinking_end", None))

    def _on_tool_use_start(self, name: str) -> None:
        self.calls.append(("tool_use_start", name))

    def _on_tool_json_delta(self, partial: str) -> None:
        self.calls.append(("tool_json_delta", partial))

    def _on_tool_use_end(self, name: str, tool_input: dict) -> None:
        self.calls.append(("tool_use_end", (name, tool_input)))

    def _on_text_block_end(self) -> None:
        self.calls.append(("text_block_end", None))


def test_thinking_block_lifecycle() -> None:
    p = _TrackingParser()
    p.parse_stream_event(_block_start("thinking"))
    text = p.parse_stream_event(_delta("thinking_delta", thinking="hmm"))
    assert text == "hmm"
    p.parse_stream_event(_block_stop())
    assert ("thinking_start", None) in p.calls
    assert ("thinking_end", None) in p.calls


def test_text_delta_and_text_block_end() -> None:
    p = _TrackingParser()
    p.parse_stream_event(_block_start("text"))
    text = p.parse_stream_event(_delta("text_delta", text="hello"))
    assert text == "hello"
    p.parse_stream_event(_block_stop())
    assert ("text_block_end", None) in p.calls


def test_tool_use_block_lifecycle() -> None:
    p = _TrackingParser()
    p.parse_stream_event(_block_start("tool_use", name="Bash"))
    p.parse_stream_event(_delta("input_json_delta", partial_json='{"cmd":'))
    p.parse_stream_event(_delta("input_json_delta", partial_json='"ls"}'))
    p.parse_stream_event(_block_stop())
    assert ("tool_use_start", "Bash") in p.calls
    assert ("tool_json_delta", '{"cmd":') in p.calls
    assert ("tool_json_delta", '"ls"}') in p.calls
    end = [c for c in p.calls if c[0] == "tool_use_end"]
    assert len(end) == 1
    name, inp = end[0][1]
    assert name == "Bash"
    assert inp == {"cmd": "ls"}


def test_tool_use_invalid_json() -> None:
    p = _TrackingParser()
    p.parse_stream_event(_block_start("tool_use", name="X"))
    p.parse_stream_event(_delta("input_json_delta", partial_json="{bad"))
    p.parse_stream_event(_block_stop())
    end = [c for c in p.calls if c[0] == "tool_use_end"]
    assert len(end) == 1
    _, inp = end[0][1]
    assert inp == {"_raw": "{bad"}


def test_reset_stream_state() -> None:
    p = _TrackingParser()
    p.parse_stream_event(_block_start("thinking"))
    assert p._current_block_type == "thinking"
    p.reset_stream_state()
    assert p._current_block_type == ""
    assert p._tool_name == ""
    assert p._tool_json_buffer == ""


def test_unknown_event_type_returns_empty() -> None:
    p = _TrackingParser()
    text = p.parse_stream_event(_evt({"type": "unknown_thing"}))
    assert text == ""
    assert p.calls == []


# ── ConsolePrinter integration tests ──


def test_console_printer_thinking_stream() -> None:
    buf = StringIO()
    cp = ConsolePrinter(file=buf)
    cp.print(_block_start("thinking"), type="stream_event")
    text = cp.print(
        _delta("thinking_delta", thinking="idea"),
        type="stream_event",
    )
    assert text == "idea"
    cp.print(_block_stop(), type="stream_event")
    assert "Thinking" in buf.getvalue()


def test_console_printer_tool_use_stream() -> None:
    buf = StringIO()
    cp = ConsolePrinter(file=buf)
    cp.print(
        _block_start("tool_use", name="Read"),
        type="stream_event",
    )
    payload = json.dumps({"file_path": "/tmp/x"})
    cp.print(
        _delta("input_json_delta", partial_json=payload),
        type="stream_event",
    )
    cp.print(_block_stop(), type="stream_event")
    assert "Read" in buf.getvalue()


def test_console_printer_reset_clears_stream_state() -> None:
    buf = StringIO()
    cp = ConsolePrinter(file=buf)
    cp.print(_block_start("thinking"), type="stream_event")
    assert cp._current_block_type == "thinking"
    cp.reset()
    assert cp._current_block_type == ""
    assert cp._mid_line is False


# ── BaseBrowserPrinter integration tests ──


def test_browser_printer_thinking_stream() -> None:
    bp = BaseBrowserPrinter()
    cq = bp.add_client()
    bp.print(_block_start("thinking"), type="stream_event")
    bp.print(_block_stop(), type="stream_event")
    events = []
    while not cq.empty():
        events.append(cq.get_nowait())
    types = [e["type"] for e in events]
    assert "thinking_start" in types
    assert "thinking_end" in types


def test_browser_printer_tool_use_stream() -> None:
    bp = BaseBrowserPrinter()
    cq = bp.add_client()
    bp.print(
        _block_start("tool_use", name="Write"),
        type="stream_event",
    )
    bp.print(
        _delta("input_json_delta", partial_json='{"content":"hi"}'),
        type="stream_event",
    )
    bp.print(_block_stop(), type="stream_event")
    events = []
    while not cq.empty():
        events.append(cq.get_nowait())
    tc = [e for e in events if e.get("type") == "tool_call"]
    assert len(tc) == 1
    assert tc[0]["name"] == "Write"


def test_browser_printer_text_block_end() -> None:
    bp = BaseBrowserPrinter()
    cq = bp.add_client()
    bp.print(_block_start("text"), type="stream_event")
    bp.print(_block_stop(), type="stream_event")
    events = []
    while not cq.empty():
        events.append(cq.get_nowait())
    assert any(e["type"] == "text_end" for e in events)


def test_browser_printer_reset_clears_stream_state() -> None:
    bp = BaseBrowserPrinter()
    bp.print(_block_start("thinking"), type="stream_event")
    assert bp._current_block_type == "thinking"
    bp.reset()
    assert bp._current_block_type == ""
