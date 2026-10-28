# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for: web browser toggle wiring and API key deletion
through the daemon's saveConfig handler.

The core-only methods (budget limit, custom endpoint, config persistence,
key save/overwrite lifecycle) moved to
``tests/core/test_features_integration.py``; the methods here exercise
``kiss.agents.sorcar.sorcar_agent`` or ``kiss.server.server``.  The
``_restore_default_config`` autouse fixture is shared from the core twin.

Each test uses real HTTP servers, real file I/O, and real objects —
no mocks, patches, fakes, or test doubles (except monkeypatch for env
isolation, which is not a test double).
"""

from __future__ import annotations

from kiss.agents.sorcar.sorcar_agent import BROWSER_TOOL_NAMES, SorcarAgent
from kiss.tests.core.test_features_integration import (  # noqa: F401
    _restore_default_config,
)

# The browser group as the agent registers it: the classic page tools,
# the DevTools-grade ones (evaluate_script, list_console_messages,
# list_network_requests, wait_for) and the two routes into the user's own
# Chrome (attach_to_chrome over CDP, use_my_chrome through the extension
# bridge).  Listed here so the toggle test fails when a tool is added to
# WebUseTool but left out of BROWSER_TOOL_NAMES or vice versa.
_EXPECTED_BROWSER_TOOLS = {
    "go_to_url", "click", "type_text", "press_key", "scroll", "screenshot",
    "get_page_content", "show_browser", "close_browser",
    "evaluate_script", "list_console_messages", "list_network_requests", "wait_for",
    "attach_to_chrome", "use_my_chrome",
}


class TestWebBrowserToggle:
    """web_tools parameter controls browser tool availability."""

    def test_browser_tool_names_are_the_web_use_tool_surface(self) -> None:
        assert set(BROWSER_TOOL_NAMES) == _EXPECTED_BROWSER_TOOLS

    def test_web_tools_false_no_browser_tools(self) -> None:
        """When web_tools=False, SorcarAgent._setup_tools skips web tools."""
        agent = SorcarAgent("no-web")
        agent._use_web_tools = False
        agent.web_use_tool = None
        tools = agent._get_tools()
        tool_names = [t.__name__ for t in tools]
        assert not _EXPECTED_BROWSER_TOOLS.intersection(tool_names)
        assert agent.web_use_tool is None

    def test_web_tools_true_has_browser_tools(self) -> None:
        """When web_tools=True, _setup_tools includes every browser tool."""
        agent = SorcarAgent("with-web")
        agent._use_web_tools = True
        agent.web_use_tool = None
        tools = agent._get_tools()
        tool_names = {t.__name__ for t in tools}
        assert _EXPECTED_BROWSER_TOOLS <= tool_names
        assert agent.web_use_tool is not None
        agent.web_use_tool.close()
        agent.web_use_tool = None
