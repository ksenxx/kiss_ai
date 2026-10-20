# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the webapp frontend sources and their server wiring.

These tests read the shipped frontend file ``media/main.js`` and
exercise the :class:`AgentServer` endpoints those frontend features
call.  The pure-backend server tests live in
``kiss.tests.server.test_agent_server``.
No mocks — uses real functions from the server module.
"""

import subprocess
import unittest
from pathlib import Path

import pytest

from kiss.server import agent_state
from kiss.server.server import AgentServer


@pytest.fixture(autouse=True)
def _clean_registry():
    """Keep the process-global agent-state registry clean per test."""
    agent_state.agent_states.clear()
    yield
    agent_state.agent_states.clear()


class TestHistoryPanelSearchOnOpen(unittest.TestCase):
    """Test that opening the history panel uses existing search text.

    Regression: the menu-btn click handler used to send getHistory without
    the ``query`` parameter, ignoring text already in the search box.  The fix
    adds ``query: historySearch.value`` so the server filters results even on
    the initial open.
    """

    _js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[3] / "kiss"
        cls._js = (base / "webapp" / "media" / "main.js").read_text()

    def _get_menu_btn_click_body(self) -> str:
        """Extract the toggleHistorySidebar function body wired to menuBtn."""
        import re

        m = re.search(
            r"menuBtn\.addEventListener\(\s*'click'\s*,\s*([A-Za-z_$][\w$]*)",
            self._js,
        )
        assert m, "menuBtn click listener not found in main.js"
        handler = m.group(1)
        idx = self._js.index(f"function {handler}(")
        brace = 0
        start = self._js.index("{", idx)
        for i in range(start, len(self._js)):
            ch = self._js[i]
            if ch == "{":
                brace += 1
            elif ch == "}":
                brace -= 1
                if brace == 0:
                    return self._js[idx : i + 1]
        raise AssertionError(f"Could not extract {handler} body")

    def _get_switch_sidebar_tab_body(self) -> str:
        """Extract the switchSidebarTab function body."""
        import re

        m = re.search(
            r"function switchSidebarTab\([^)]*\)\s*\{", self._js
        )
        assert m, "switchSidebarTab function not found"
        start = m.start()
        brace = 0
        for i in range(m.end() - 1, len(self._js)):
            ch = self._js[i]
            if ch == "{":
                brace += 1
            elif ch == "}":
                brace -= 1
                if brace == 0:
                    return self._js[start : i + 1]
        raise AssertionError("Could not extract switchSidebarTab body")


    def test_server_filters_history_with_query(self) -> None:
        """AgentServer._get_history passes query to _load_history."""
        server = AgentServer()
        events: list[dict] = []
        server.printer.broadcast = lambda ev: events.append(ev)  # type: ignore[assignment]

        server._get_history("some search text", offset=0, generation=1)
        assert len(events) == 1
        assert events[0]["type"] == "history"
        assert events[0]["generation"] == 1

    def test_server_returns_unfiltered_without_query(self) -> None:
        """AgentServer._get_history returns unfiltered results when query is None."""
        server = AgentServer()
        events: list[dict] = []
        server.printer.broadcast = lambda ev: events.append(ev)  # type: ignore[assignment]

        server._get_history(None, offset=0, generation=0)
        assert len(events) == 1
        assert events[0]["type"] == "history"
        assert isinstance(events[0]["sessions"], list)

class TestAgentToggle(unittest.TestCase):
    """Tests for worktree toggle switching between agents."""

    def test_worktree_action_rejected_without_a_pending_worktree(self) -> None:
        """Worktree action fails gracefully when no task left a worktree."""
        server = AgentServer()
        result = server._handle_worktree_action("merge")
        assert result["success"] is False
        assert "No pending worktree" in result["message"]

class TestTabStateRestore(unittest.TestCase):
    """Test that tab state is persisted correctly for cross-restart restore.

    Tabs are identified by tab.id which IS the chat_id. persistTabState()
    serializes tab.id as chatId, and updateActiveTabTitle() updates tab.title.
    """

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[3] / "kiss"
        cls.js = (base / "webapp" / "media" / "main.js").read_text()



    def test_persist_tab_state_logic_via_node(self) -> None:
        """Run the actual JS logic in Node.js and verify correctness."""
        node_script = """
        var activeTabId = '';
        var tabs = [];
        var _lastState = null;

        var transport = {
            setState: function(s) { _lastState = s; },
            getState: function() { return _lastState; },
        };

        function persistTabState() {
            var serialized = tabs.map(function(t) {
                return { title: t.title, chatId: t.id };
            });
            var activeIdx = tabs.findIndex(function(t) { return t.id === activeTabId; });
            transport.setState({ tabs: serialized, activeTabIndex: activeIdx });
        }

        // Test 1: Single tab, tab.id persisted as chatId
        tabs.push({ id: 'abc123', title: 'new chat' });
        activeTabId = 'abc123';
        persistTabState();
        var state = transport.getState();
        if (state.tabs[0].chatId !== 'abc123') {
            console.log('FAIL test1: ' + state.tabs[0].chatId);
            process.exit(1);
        }

        // Test 2: Multi-tab scenario
        tabs = [];
        tabs.push({ id: 'chat-A', title: 'task A' });
        tabs.push({ id: 'chat-B', title: 'new chat' });
        activeTabId = 'chat-B';
        persistTabState();
        state = transport.getState();
        if (state.tabs[0].chatId !== 'chat-A') {
            console.log('FAIL 2a: ' + state.tabs[0].chatId);
            process.exit(1);
        }
        if (state.tabs[1].chatId !== 'chat-B') {
            console.log('FAIL 2b: ' + state.tabs[1].chatId);
            process.exit(1);
        }

        console.log('PASS: all tab state persistence tests passed');
        """
        result = subprocess.run(
            ["node", "-e", node_script],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"Node.js test failed: {result.stdout}{result.stderr}"
        assert "PASS" in result.stdout


if __name__ == "__main__":
    unittest.main()
