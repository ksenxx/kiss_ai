# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
# ruff: noqa: F811  (the `server` / `service` / `tool` fixtures are imported from
# test_web_use_tool_attach_stop_wave2 and re-exported for pytest's collector)
"""Review 2, Q2 finding 5: a Stop while the Browser tab's page is adopted closes the tab.

``WebUseTool._attach_live`` closed the tab the daemon's service opened
when a Stop landed while it connected or looked for the page, but
adopting the page (crash handler, target id, network observers) and
recording the tab as the tool's own came after the protected block: a
Stop there left the blank tab open on every surface with no owner, and
the next call opened another one.  A REAL service and a REAL tool; the
Stop is injected into the tool's thread the way the daemon does, at the
entry of ``_adopt_page`` and of the observer set-up inside it.
"""

from __future__ import annotations

import functools
import threading
from types import CodeType

import pytest

from kiss.agents.sorcar.web_use_tool import WebUseTool
from kiss.core.tool_interrupt import ToolCallInterrupted
from kiss.server.browser_tab import BrowserTabService
from kiss.tests.agents.sorcar.test_web_use_tool_attach_stop_wave2 import (  # noqa: F401
    _events,
    _stop_once,
    _wait,
    pytestmark,
    server,
    service,
    tool,
)
from kiss.tests.server._memory_printer import MemoryPrinter
from kiss.tests.server.pause import Pause


@pytest.mark.parametrize(
    "where", [WebUseTool._adopt_page.__code__, WebUseTool._observe_page.__code__],
    ids=["adopt_page", "observe_page"],
)
def test_stop_while_adopting_the_page_closes_the_tab_it_opened(
    tool: WebUseTool, service: tuple[BrowserTabService, MemoryPrinter], server: str,
    where: CodeType,
) -> None:
    svc, printer = service
    fired = threading.Event()
    with Pause(where, functools.partial(_stop_once, fired)), pytest.raises(ToolCallInterrupted):
        tool.show_browser()

    assert fired.is_set()
    (tab_id,) = {e["tab_id"] for e in _events(printer, "openBrowserTab")}
    _wait(lambda: _events(printer, "closeBrowserTab", tab_id=tab_id), "the blank tab closed")
    assert tab_id not in svc._pages
    # Nothing of the half-adopted page is left as the tool's own.
    assert tool._page is None and tool._live_tab is None
    # The next call opens a fresh tab and drives it.
    assert tool.go_to_url(f"{server}/inert").startswith("Page:")
    assert tool._live_tab in svc._pages and tool._live_tab != tab_id
    assert len({e["tab_id"] for e in _events(printer, "openBrowserTab")}) == 2
