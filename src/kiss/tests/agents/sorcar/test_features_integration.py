# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for: web browser toggle wiring and API key deletion
through the VS Code server's saveConfig handler.

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

from kiss.tests.core.test_features_integration import (  # noqa: F401
    _restore_default_config,
)


class TestWebBrowserToggle:
    """web_tools parameter controls browser tool availability."""

    def test_web_tools_false_no_browser_tools(self) -> None:
        """When web_tools=False, SorcarAgent._setup_tools skips web tools."""
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        agent = SorcarAgent("no-web")
        agent._use_web_tools = False
        agent.web_use_tool = None
        tools = agent._get_tools()
        tool_names = [t.__name__ for t in tools]
        browser_names = {
            "go_to_url", "click", "type_text", "press_key",
            "scroll", "screenshot", "get_page_content", "close_browser",
        }
        assert not browser_names.intersection(tool_names)
        assert agent.web_use_tool is None

    def test_web_tools_true_has_browser_tools(self) -> None:
        """When web_tools=True, _setup_tools includes web tools."""
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        agent = SorcarAgent("with-web")
        agent._use_web_tools = True
        agent.web_use_tool = None
        tools = agent._get_tools()
        tool_names = [t.__name__ for t in tools]
        assert "go_to_url" in tool_names
        assert agent.web_use_tool is not None
        agent.web_use_tool.close()
        agent.web_use_tool = None
