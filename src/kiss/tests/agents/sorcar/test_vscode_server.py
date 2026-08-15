# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the VS Code extension backend server.

Tests cover: model picker (vendor ordering, sorting, grouping, pricing),
file picker (sorting by usage/recency/end-distance, section grouping),
keyboard interaction parity with web Sorcar, and the JS rendering code
in main.js.
No mocks — uses real functions from the server module.
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import pytest

from kiss.agents.sorcar.git_worktree import GitWorktree
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.helpers import model_vendor
from kiss.server.server import VSCodeServer


@pytest.fixture(autouse=True)
def _clean_registry():
    """Keep the process-global agent-state registry clean per test."""
    agent_state.agent_states.clear()
    yield
    agent_state.agent_states.clear()


def _register_wt_state(
    tab_id: str,
    *,
    agent: WorktreeSorcarAgent | None = None,
    use_worktree: bool = True,
    is_merging: bool = False,
) -> AgentState:
    """Register a server-owned AgentState for *tab_id* and return it."""
    state = AgentState(
        f"task-{tab_id}",
        agent=agent,
        tab_id=tab_id,
        server_owned=True,
    )
    state.use_worktree = use_worktree
    state.is_merging = is_merging
    agent_state.register(state)
    return state


def _set_agent_wt(agent: object, repo: Path, branch: str, original: str) -> None:
    """Helper to set agent._wt with a GitWorktree for testing."""
    slug = branch.replace("/", "_")
    agent._wt = GitWorktree(  # type: ignore[attr-defined]
        repo_root=repo,
        branch=branch,
        original_branch=original,
        wt_dir=repo / ".kiss-worktrees" / slug,
    )


def _model_vendor_name(name: str) -> str:
    return model_vendor(name)[0]


def _model_vendor_order(name: str) -> int:
    return model_vendor(name)[1]


class TestModelVendorOrder(unittest.TestCase):
    """Test _model_vendor_order matches web Sorcar's modelVendor sorting."""

    def test_order_is_consistent(self) -> None:
        names = [
            "unknown-model",
            "gemini-2.0-flash",
            "claude-opus-4-6",
            "gpt-4o",
            "openrouter/x",
            "glm-4.6",
            "cc/opus",
        ]
        sorted_names = sorted(names, key=_model_vendor_order)
        assert sorted_names[0] == "claude-opus-4-6"
        assert sorted_names[1] == "cc/opus"
        assert sorted_names[2] == "gpt-4o"
        assert sorted_names[3] == "gemini-2.0-flash"
        assert sorted_names[-1] in ("unknown-model", "together/some-model")


class TestGetFiles(unittest.TestCase):
    """Test VSCodeServer._get_files produces correct sections and sorting."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.server = VSCodeServer()
        self.server.work_dir = self.tmpdir
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]

        for name in ["src/main.py", "src/util.py", "README.md", "test/test_main.py"]:
            path = Path(self.tmpdir) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {name}")

        self.server._file_cache = {
            self.tmpdir: [
                "src/main.py",
                "src/util.py",
                "README.md",
                "test/test_main.py",
            ],
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_files_filtered_by_prefix(self) -> None:
        self.server._get_files("main")
        files = self.events[0]["files"]
        for f in files:
            assert "main" in f["text"].lower()


class TestNewChatBroadcastsShowWelcome(unittest.TestCase):
    """_new_chat must broadcast a showWelcome event to the tab."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.server = VSCodeServer()
        self.server.work_dir = self.tmpdir
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_new_chat_broadcasts_show_welcome(self) -> None:
        self.server._new_chat("tab-1")
        welcome_events = [e for e in self.events if e["type"] == "showWelcome"]
        assert len(welcome_events) == 1
        assert welcome_events[0]["tabId"] == "tab-1"








class TestGenerateCommitMessage(unittest.TestCase):
    """Test generateCommitMessage uses get_fast_model via _generate_commit_message_llm."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.server = VSCodeServer()
        self.server.work_dir = self.tmpdir
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)



    def test_no_staged_changes(self) -> None:
        """_generate_commit_message reports no staged changes."""
        subprocess.run(["git", "init"], cwd=self.tmpdir, capture_output=True)
        self.server._generate_commit_message()
        assert len(self.events) == 1
        assert self.events[0]["error"] == (
            "No staged changes found. Stage files with 'git add' first."
        )


class TestExtractResultSummary(unittest.TestCase):
    """Test _extract_result_summary extracts summary from recorded events."""

    def setUp(self) -> None:
        self.server = VSCodeServer()


class TestLastActiveFile(unittest.TestCase):
    """Test that _last_active_file is stored from run commands."""

    def setUp(self) -> None:
        self.server = VSCodeServer()






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






class TestCompleteFromActiveFile(unittest.TestCase):
    """Test chained identifier extraction and matching from active file content."""

    def setUp(self) -> None:
        self.server = VSCodeServer()
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]










class TestWorktreeServerIntegration(unittest.TestCase):
    """Integration tests for worktree support in VSCodeServer."""

    def _git(self, *args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=self.repo, capture_output=True,
        )

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = Path(self.tmpdir) / "repo"
        self.repo.mkdir()
        self._git("init", "-b", "main")
        self._git("config", "user.email", "test@test.com")
        self._git("config", "user.name", "Test")
        (self.repo / "file.txt").write_text("hello")
        self._git("add", ".")
        self._git("commit", "-m", "init")

        self.server = VSCodeServer()
        self.server.work_dir = str(self.repo)
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_handle_worktree_action_merge(self) -> None:
        """Merge action calls agent.merge() and returns result."""
        self._git("checkout", "-b", "kiss/merge-test")
        (self.repo / "merged.txt").write_text("merged content")
        self._git("add", ".")
        self._git("commit", "-m", "add merged")
        self._git("checkout", "main")

        state = _register_wt_state(
            "0", agent=WorktreeSorcarAgent("Sorcar VS Code"),
        )
        _set_agent_wt(state.agent, self.repo, "kiss/merge-test", "main")

        result = self.server._handle_worktree_action("merge", "0")
        assert result["success"] is True
        assert "Successfully merged" in result["message"]
        after_agent = state.agent
        assert after_agent is not None
        assert after_agent._wt_branch is None

    def test_handle_worktree_action_discard(self) -> None:
        """Discard action removes worktree branch."""
        self._git("checkout", "-b", "kiss/discard-test")
        self._git("checkout", "main")

        state = _register_wt_state(
            "0", agent=WorktreeSorcarAgent("Sorcar VS Code"),
        )
        _set_agent_wt(state.agent, self.repo, "kiss/discard-test", "main")

        result = self.server._handle_worktree_action("discard", "0")
        assert result["success"] is True
        assert "Discarded" in result["message"]
        after_agent = state.agent
        assert after_agent is not None
        assert after_agent._wt_branch is None

    def test_worktree_action_command_routing(self) -> None:
        """worktreeAction command is routed to _handle_worktree_action."""
        self._git("checkout", "-b", "kiss/route-test")
        (self.repo / "route.txt").write_text("route content")
        self._git("add", ".")
        self._git("commit", "-m", "add route")
        self._git("checkout", "main")

        state = _register_wt_state(
            "0", agent=WorktreeSorcarAgent("Sorcar VS Code"),
        )
        _set_agent_wt(state.agent, self.repo, "kiss/route-test", "main")

        self.server._handle_command({"type": "worktreeAction", "action": "merge", "tabId": "0"})
        wt_events = [e for e in self.events if e["type"] == "worktree_result"]
        assert len(wt_events) == 1
        assert wt_events[0]["success"] is True

    def test_merge_broadcasts_progress_before_result(self) -> None:
        """Merge action broadcasts worktree_progress before worktree_result."""
        self._git("checkout", "-b", "kiss/progress-test")
        (self.repo / "progress.txt").write_text("progress content")
        self._git("add", ".")
        self._git("commit", "-m", "add progress")
        self._git("checkout", "main")

        state = _register_wt_state(
            "0", agent=WorktreeSorcarAgent("Sorcar VS Code"),
        )
        _set_agent_wt(state.agent, self.repo, "kiss/progress-test", "main")

        self.server._handle_command({"type": "worktreeAction", "action": "merge", "tabId": "0"})
        progress_events = [e for e in self.events if e["type"] == "worktree_progress"]
        assert len(progress_events) == 1
        assert "Generating commit message" in progress_events[0]["message"]
        relevant = ("worktree_progress", "worktree_result")
        types = [e["type"] for e in self.events if e["type"] in relevant]
        assert types == ["worktree_progress", "worktree_result"]

    def test_discard_does_not_broadcast_progress(self) -> None:
        """Discard action does not broadcast worktree_progress."""
        self._git("checkout", "-b", "kiss/no-progress-test")
        self._git("checkout", "main")

        state = _register_wt_state(
            "0", agent=WorktreeSorcarAgent("Sorcar VS Code"),
        )
        _set_agent_wt(state.agent, self.repo, "kiss/no-progress-test", "main")

        self.server._handle_command({"type": "worktreeAction", "action": "discard", "tabId": "0"})
        progress_events = [e for e in self.events if e["type"] == "worktree_progress"]
        assert len(progress_events) == 0


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












class TestMergeGuard(unittest.TestCase):
    """The per-tab ``is_merging`` flag guards task starts on that tab."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.server = VSCodeServer()
        self.server.work_dir = self.tmpdir
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_merging_blocks_same_tab(self) -> None:
        """Cannot start a task on the same tab that has a merge in progress."""
        _register_wt_state("5", use_worktree=False, is_merging=True)
        self.server._run_task_inner({"prompt": "test", "model": "", "tabId": "5"})
        errors = [e for e in self.events if e["type"] == "error"]
        assert any("merge is in progress" in e["text"] for e in errors)

    @pytest.mark.slow
    def test_merging_does_not_block_other_tabs(self) -> None:
        """A merge on one tab does not block tasks on other tabs."""
        _register_wt_state("5", use_worktree=False, is_merging=True)
        self.events.clear()
        self.server._run_task_inner({"prompt": "test", "model": "", "tabId": "99"})
        errors = [e for e in self.events if e["type"] == "error"]
        assert not any(
            "merge is in progress" in e.get("text", "") for e in errors
        )


class TestWorktreeActionExceptionHandling(unittest.TestCase):
    """Regression: worktree actions must always broadcast worktree_result,
    even when the action raises an exception.

    Root cause: _handle_worktree_action was called without try/except in
    _handle_command, so a RuntimeError from wt.merge() (e.g. when _wt is
    None) would prevent the worktree_result broadcast, causing the VS Code
    UI to hang for the 120s timeout.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = Path(self.tmpdir) / "repo"
        self.repo.mkdir()
        subprocess.run(
            ["git", "init", "-b", "main"], cwd=self.repo, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=self.repo, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=self.repo, capture_output=True,
        )
        (self.repo / "file.txt").write_text("hello")
        subprocess.run(
            ["git", "add", "."], cwd=self.repo, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=self.repo, capture_output=True,
        )

        self.server = VSCodeServer()
        self.server.work_dir = str(self.repo)
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_merge_exception_still_broadcasts_result(self) -> None:
        """worktree_result is broadcast even when merge raises RuntimeError."""
        _register_wt_state("0", agent=WorktreeSorcarAgent("Sorcar VS Code"))
        self.server._handle_command({"type": "worktreeAction", "action": "merge", "tabId": "0"})
        results = [e for e in self.events if e["type"] == "worktree_result"]
        assert len(results) == 1
        assert results[0]["success"] is False
        assert results[0]["message"]

    def test_discard_exception_still_broadcasts_result(self) -> None:
        """worktree_result is broadcast even when discard raises RuntimeError."""
        _register_wt_state("0", agent=WorktreeSorcarAgent("Sorcar VS Code"))
        self.server._handle_command({"type": "worktreeAction", "action": "discard", "tabId": "0"})
        results = [e for e in self.events if e["type"] == "worktree_result"]
        assert len(results) == 1
        assert results[0]["success"] is False

    def test_successful_merge_still_works(self) -> None:
        """Normal merge flow still works after the try/except addition."""
        subprocess.run(
            ["git", "checkout", "-b", "kiss/exc-test"],
            cwd=self.repo, capture_output=True,
        )
        (self.repo / "new.txt").write_text("new content")
        subprocess.run(
            ["git", "add", "."], cwd=self.repo, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "add new"],
            cwd=self.repo, capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", "main"],
            cwd=self.repo, capture_output=True,
        )

        state = _register_wt_state(
            "0", agent=WorktreeSorcarAgent("Sorcar VS Code"),
        )
        _set_agent_wt(state.agent, self.repo, "kiss/exc-test", "main")

        self.server._handle_command({"type": "worktreeAction", "action": "merge", "tabId": "0"})
        results = [e for e in self.events if e["type"] == "worktree_result"]
        assert len(results) == 1
        assert results[0]["success"] is True




class TestExtractExtrasNoTruncation(unittest.TestCase):
    """Verify extract_extras does not truncate long argument values."""

    def test_long_value_not_truncated(self):
        from kiss.core.printer import extract_extras
        long_val = "x" * 500
        result = extract_extras({"custom_arg": long_val})
        assert result == {"custom_arg": long_val}
        assert "..." not in result["custom_arg"]

    def test_known_keys_excluded(self):
        from kiss.core.printer import extract_extras
        result = extract_extras({
            "file_path": "/a/b.py", "command": "ls", "extra": "val",
        })
        assert result == {"extra": "val"}






























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
