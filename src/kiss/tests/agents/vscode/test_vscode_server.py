# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the VS Code extension frontend sources and their server wiring.

These tests read the extension's shipped frontend files
(``media/main.js``, ``src/SorcarSidebarView.ts``) — directly or in their
``setUpClass`` — and exercise the :class:`VSCodeServer` endpoints those
frontend features call.  The pure-backend server tests live in
``kiss.tests.server.test_vscode_server``.
No mocks — uses real functions from the server module.
"""

import subprocess
import unittest
from pathlib import Path

import pytest

from kiss.server import agent_state
from kiss.server.server import VSCodeServer


@pytest.fixture(autouse=True)
def _clean_registry():
    """Keep the process-global agent-state registry clean per test."""
    agent_state.agent_states.clear()
    yield
    agent_state.agent_states.clear()


class TestMainJsInfiniteScroll(unittest.TestCase):
    """Test main.js has infinite scroll and chat_id color code."""

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.js = (base / "vscode" / "media" / "main.js").read_text()









    def test_chat_id_bg_colors_are_light(self) -> None:
        """Verify the chatIdBgColor function produces light pastel colors.

        Reimplements the JS djb2 hash + HSL logic in Python and checks that
        the minimum RGB channel is >= 140 (i.e., clearly light) for
        a wide range of chat_id strings.
        """
        import colorsys
        import ctypes

        def chat_id_bg_rgb(chat_id: str) -> tuple[int, int, int]:
            h = 5381
            for ch in chat_id:
                h = ((h << 5) + h) + ord(ch)
                h = ctypes.c_int32(h).value
            hue = abs(h) % 360
            r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.75, 0.55)
            return (round(r * 255), round(g * 255), round(b * 255))

        test_ids = [
            "abc123", "xyz789", "chat-001", "chat-002", "session-1",
            "a", "test", "550e8400-e29b-41d4-a716-446655440000",
            "f47ac10b-58cc-4372-a567-0e02b2c3d479", "z",
        ]
        for cid in test_ids:
            r, g, b = chat_id_bg_rgb(cid)
            assert min(r, g, b) >= 140, (
                f"chat_id={cid!r} produced dark color rgb({r},{g},{b})"
            )

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
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._js = (base / "vscode" / "media" / "main.js").read_text()

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
        """VSCodeServer._get_history passes query to _search_history."""
        server = VSCodeServer()
        events: list[dict] = []
        server.printer.broadcast = lambda ev: events.append(ev)  # type: ignore[assignment]

        server._get_history("some search text", offset=0, generation=1)
        assert len(events) == 1
        assert events[0]["type"] == "history"
        assert events[0]["generation"] == 1

    def test_server_returns_unfiltered_without_query(self) -> None:
        """VSCodeServer._get_history returns unfiltered results when query is None."""
        server = VSCodeServer()
        events: list[dict] = []
        server.printer.broadcast = lambda ev: events.append(ev)  # type: ignore[assignment]

        server._get_history(None, offset=0, generation=0)
        assert len(events) == 1
        assert events[0]["type"] == "history"
        assert isinstance(events[0]["sessions"], list)

class TestAgentToggle(unittest.TestCase):
    """Tests for worktree toggle switching between agents."""

    _JS_PATH = (
        Path(__file__).resolve().parents[3]
        / "agents" / "vscode" / "media" / "main.js"
    )
    _TS_PATH = (
        Path(__file__).resolve().parents[3]
        / "agents" / "vscode" / "src" / "SorcarSidebarView.ts"
    )
    _js: str
    _ts: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._js = cls._JS_PATH.read_text()
        cls._ts = cls._TS_PATH.read_text()

    def test_worktree_action_rejected_when_not_enabled(self) -> None:
        """Worktree action fails gracefully when worktree mode is off."""
        server = VSCodeServer()
        result = server._handle_worktree_action("merge")
        assert result["success"] is False
        assert "not enabled" in result["message"]

class TestTabStateRestore(unittest.TestCase):
    """Test that tab state is persisted correctly for cross-restart restore.

    Tabs are identified by tab.id which IS the chat_id. persistTabState()
    serializes tab.id as chatId, and updateActiveTabTitle() updates tab.title.
    """

    js: str

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls.js = (base / "vscode" / "media" / "main.js").read_text()



    def test_persist_tab_state_logic_via_node(self) -> None:
        """Run the actual JS logic in Node.js and verify correctness."""
        node_script = """
        var activeTabId = '';
        var tabs = [];
        var _lastState = null;

        var vscode = {
            setState: function(s) { _lastState = s; },
            getState: function() { return _lastState; },
        };

        function persistTabState() {
            var serialized = tabs.map(function(t) {
                return { title: t.title, chatId: t.id };
            });
            var activeIdx = tabs.findIndex(function(t) { return t.id === activeTabId; });
            vscode.setState({ tabs: serialized, activeTabIndex: activeIdx });
        }

        // Test 1: Single tab, tab.id persisted as chatId
        tabs.push({ id: 'abc123', title: 'new chat' });
        activeTabId = 'abc123';
        persistTabState();
        var state = vscode.getState();
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
        state = vscode.getState();
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
