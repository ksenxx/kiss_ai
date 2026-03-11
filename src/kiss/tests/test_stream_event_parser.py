"""Tests for StreamEventParser and its use in ConsolePrinter / BaseBrowserPrinter."""

from __future__ import annotations

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


# ── ConsolePrinter integration tests ──


def test_console_printer_reset_clears_stream_state() -> None:
    buf = StringIO()
    cp = ConsolePrinter(file=buf)
    cp.print(_block_start("thinking"), type="stream_event")
    assert cp._current_block_type == "thinking"
    cp.reset()
    assert cp._current_block_type == ""
    assert cp._mid_line is False


# ── BaseBrowserPrinter integration tests ──


def test_browser_printer_reset_clears_stream_state() -> None:
    bp = BaseBrowserPrinter()
    bp.print(_block_start("thinking"), type="stream_event")
    assert bp._current_block_type == "thinking"
    bp.reset()
    assert bp._current_block_type == ""
