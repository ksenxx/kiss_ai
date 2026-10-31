# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `server` / `service` / `tool` fixtures are imported from
# test_web_use_tool_attach_stop_wave2 and re-exported for pytest's collector)
"""Review 3, P2_sorcar item 1: a Stop right after the Browser tab is handed over closes it.

``WebUseTool._attach_live`` took the tab from
``BrowserTabService.open_for_agent`` on the line before the block that
closes it on any exception.  Once ``open_for_agent`` has returned, the
service's own clean-up is no longer on the stack, so a Stop delivered
on the next line of ``_attach_live`` left the blank tab on every surface
with no owner, and the next call opened a second one.  A REAL service
and a REAL tool; the Stop is injected into the tool's thread at the
first line ``_attach_live`` runs after ``open_for_agent`` returned a tab.
"""

from __future__ import annotations

import sys
import threading
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core.tool_interrupt import ToolCallInterrupted, inject_async_exception
from kiss.server.browser_tab import BrowserTabService
from kiss.tests.agents.sorcar.test_web_use_tool_attach_stop_wave2 import (  # noqa: F401
    _events,
    _wait,
    pytestmark,
    server,
    service,
    tool,
)
from kiss.tests.server._memory_printer import MemoryPrinter


class _StopAfterHandover:
    """``sys.settrace`` hook: Stop at ``_attach_live``'s first line after it got its tab."""

    def __init__(self) -> None:
        self.handed_over = False
        self.fired = False

    def __call__(self, frame: FrameType, event: str, arg: Any) -> _StopAfterHandover | None:
        code = frame.f_code
        if code is BrowserTabService.open_for_agent.__code__:
            # ``arg`` is the returned tab; ``None`` when the open raised.
            if event == "return" and arg is not None:
                self.handed_over = True
            return self
        if code is not WebUseTool._attach_live.__code__:
            return None
        if event == "line" and self.handed_over and not self.fired:
            self.fired = True
            inject_async_exception(threading.get_ident(), ToolCallInterrupted)
        return self


def test_stop_right_after_the_handover_closes_the_tab(
    tool: WebUseTool, service: tuple[BrowserTabService, MemoryPrinter], server: str
) -> None:
    svc, printer = service
    hook = _StopAfterHandover()
    previous = sys.gettrace()  # a coverage tracer, say: given back afterwards
    sys.settrace(hook)
    try:
        with pytest.raises(ToolCallInterrupted):
            tool.show_browser()
    finally:
        sys.settrace(previous)

    assert hook.fired
    (tab_id,) = {e["tab_id"] for e in _events(printer, "openBrowserTab")}
    _wait(lambda: _events(printer, "closeBrowserTab", tab_id=tab_id), "the blank tab closed")
    assert tab_id not in svc._pages
    assert tool._page is None and tool._live_tab is None
    # The next call opens a fresh tab and drives it.
    assert tool.go_to_url(f"{server}/inert").startswith("Page:")
    assert tool._live_tab in svc._pages and tool._live_tab != tab_id
    assert len({e["tab_id"] for e in _events(printer, "openBrowserTab")}) == 2
