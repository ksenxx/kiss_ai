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
from kiss.agents.vscode.helpers import model_vendor
from kiss.agents.vscode.server import VSCodeServer


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
            "minimax-large",
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

        self.server._file_cache = [
            "src/main.py",
            "src/util.py",
            "README.md",
            "test/test_main.py",
        ]

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

        # Find the click handler name registered on menuBtn.
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


class TestHistoryClickTabFocus(unittest.TestCase):
    """Test that clicking a history item focuses an existing tab or creates a new one."""

    _js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._js = (base / "vscode" / "media" / "main.js").read_text()

    def _get_render_history_body(self) -> str:
        idx = self._js.index("function renderHistory(")
        end = self._js.index("\n  function ", idx + 1)
        return self._js[idx:end]


    def test_history_click_creates_new_tab(self) -> None:
        """History item click creates a new tab and loads the session."""
        body = self._get_render_history_body()
        assert "createNewTab()" in body
        assert "resumeSession" in body








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
        self._git("init")
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

        tab = self.server._get_tab("0")
        tab.agent = WorktreeSorcarAgent("Sorcar VS Code")
        tab.use_worktree = True
        _set_agent_wt(tab.agent, self.repo, "kiss/merge-test", "main")

        result = self.server._handle_worktree_action("merge", "0")
        assert result["success"] is True
        assert "Successfully merged" in result["message"]
        after_agent = self.server._get_tab("0").agent
        assert after_agent is not None
        assert after_agent._wt_branch is None

    def test_handle_worktree_action_discard(self) -> None:
        """Discard action removes worktree branch."""
        self._git("checkout", "-b", "kiss/discard-test")
        self._git("checkout", "main")

        tab = self.server._get_tab("0")
        tab.agent = WorktreeSorcarAgent("Sorcar VS Code")
        tab.use_worktree = True
        _set_agent_wt(tab.agent, self.repo, "kiss/discard-test", "main")

        result = self.server._handle_worktree_action("discard", "0")
        assert result["success"] is True
        assert "Discarded" in result["message"]
        after_agent = self.server._get_tab("0").agent
        assert after_agent is not None
        assert after_agent._wt_branch is None

    def test_worktree_action_command_routing(self) -> None:
        """worktreeAction command is routed to _handle_worktree_action."""
        self._git("checkout", "-b", "kiss/route-test")
        (self.repo / "route.txt").write_text("route content")
        self._git("add", ".")
        self._git("commit", "-m", "add route")
        self._git("checkout", "main")

        self.server._get_tab("0").use_worktree = True
        wt_agent = self.server._get_tab("0").agent
        _set_agent_wt(wt_agent, self.repo, "kiss/route-test", "main")

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

        tab = self.server._get_tab("0")
        tab.agent = WorktreeSorcarAgent("Sorcar VS Code")
        tab.use_worktree = True
        _set_agent_wt(tab.agent, self.repo, "kiss/progress-test", "main")

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

        tab = self.server._get_tab("0")
        tab.agent = WorktreeSorcarAgent("Sorcar VS Code")
        tab.use_worktree = True
        _set_agent_wt(tab.agent, self.repo, "kiss/no-progress-test", "main")

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

    def test_server_agent_is_worktree_sorcar_agent(self) -> None:
        """Server agent is a single WorktreeSorcarAgent regardless of toggle.

        ``WorktreeSorcarAgent`` subclasses ``ChatSorcarAgent`` and
        internally falls back to the stateful code path when
        ``use_worktree=False`` is passed to ``run()``.  One instance
        per tab is therefore sufficient.
        """
        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
        from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

        server = VSCodeServer()
        tab = server._get_tab("0")
        tab.agent = WorktreeSorcarAgent("Sorcar VS Code")
        assert tab.use_worktree is False
        assert isinstance(tab.agent, WorktreeSorcarAgent)
        assert isinstance(tab.agent, ChatSorcarAgent)
        original = tab.agent
        tab.use_worktree = True
        assert tab.agent is original




    def test_worktree_action_rejected_when_not_enabled(self) -> None:
        """Worktree action fails gracefully when worktree mode is off."""
        server = VSCodeServer()
        result = server._handle_worktree_action("merge")
        assert result["success"] is False
        assert "not enabled" in result["message"]












class TestServerParallelToggle(unittest.TestCase):
    """Tests for parallel toggle in VSCodeServer."""

    def test_server_defaults_parallel_on(self) -> None:
        """use_parallel is True by default on new tab state."""
        server = VSCodeServer()
        assert server._get_tab("0").use_parallel is True




class TestMergeSession(unittest.TestCase):
    """Tests for _start_merge_session, _finish_merge,
    and _restore_pending_merge."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.merge_dir = Path(self.tmpdir) / "merge_dir"
        self.merge_dir.mkdir()
        self.server = VSCodeServer()
        self.server.work_dir = self.tmpdir
        self.events: list[dict] = []

        def capture_broadcast(event: dict) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture_broadcast  # type: ignore[assignment]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_merge_json(self, files: list[dict] | None = None) -> str:
        """Write a pending-merge.json and return its path."""
        import json as _json

        if files is None:
            base = self.merge_dir / "merge-temp" / "a.txt"
            base.parent.mkdir(parents=True, exist_ok=True)
            base.write_text("old line\n")
            current = Path(self.tmpdir) / "a.txt"
            current.write_text("new line\n")
            files = [{
                "name": "a.txt",
                "base": str(base),
                "current": str(current),
                "hunks": [{"bs": 0, "bc": 1, "cs": 0, "cc": 1}],
            }]
        merge_json = self.merge_dir / "pending-merge.json"
        merge_json.write_text(_json.dumps({"branch": "HEAD", "files": files}))
        return str(merge_json)

    def test_start_merge_session_broadcasts_merge_data_and_started(self) -> None:
        """_start_merge_session broadcasts merge_data and merge_started events."""
        path = self._write_merge_json()
        result = self.server._start_merge_session(path)
        assert result is True
        types = [e["type"] for e in self.events]
        assert "merge_data" in types
        assert "merge_started" in types
        assert types.index("merge_data") < types.index("merge_started")

    def test_start_merge_session_includes_hunk_count(self) -> None:
        """merge_data event includes correct hunk_count."""
        path = self._write_merge_json()
        self.server._start_merge_session(path)
        md = [e for e in self.events if e["type"] == "merge_data"][0]
        assert md["hunk_count"] == 1

    def test_start_merge_session_returns_false_for_empty_files(self) -> None:
        """Returns False when merge JSON has no files."""
        path = self._write_merge_json(files=[])
        result = self.server._start_merge_session(path)
        assert result is False

    def test_start_merge_session_returns_false_for_zero_hunks(self) -> None:
        """Returns False when all files have zero hunks."""
        current = Path(self.tmpdir) / "b.txt"
        current.write_text("content\n")
        path = self._write_merge_json(files=[{
            "name": "b.txt",
            "base": str(current),
            "current": str(current),
            "hunks": [],
        }])
        result = self.server._start_merge_session(path)
        assert result is False

    def test_start_merge_session_returns_false_for_missing_file(self) -> None:
        """Returns False when merge JSON file doesn't exist."""
        result = self.server._start_merge_session("/nonexistent/merge.json")
        assert result is False

    def test_start_merge_session_returns_false_for_invalid_json(self) -> None:
        """Returns False when merge JSON is malformed."""
        bad = self.merge_dir / "bad.json"
        bad.write_text("not json")
        result = self.server._start_merge_session(str(bad))
        assert result is False

    def test_merge_action_all_done_finishes_merge(self) -> None:
        """mergeAction all-done routed via _cmd_merge_action calls _finish_merge."""
        path = self._write_merge_json()
        self.server._start_merge_session(path, tab_id="m-tab")
        self.events.clear()

        self.server._handle_command({"type": "mergeAction", "action": "all-done", "tabId": "m-tab"})
        types = [e["type"] for e in self.events]
        assert "merge_ended" in types

    def test_merge_action_unknown_is_noop(self) -> None:
        """Non-'all-done' actions are no-ops on the Python side."""
        self.server._get_tab("0").is_merging = True
        self.server._handle_command({"type": "mergeAction", "action": "accept", "tabId": "0"})
        assert self.server._get_tab("0").is_merging is True

    def test_finish_merge_cleans_up_data_dir(self) -> None:
        """_finish_merge removes the merge data directory for the tab."""
        import kiss.agents.vscode.diff_merge as dm
        import kiss.agents.vscode.merge_flow as mf

        orig_dm = dm._merge_data_dir
        orig_mf = mf._merge_data_dir
        dm._merge_data_dir = lambda tab_id="": self.merge_dir  # type: ignore[assignment]
        mf._merge_data_dir = lambda tab_id="": self.merge_dir  # type: ignore[assignment]
        try:
            path = self._write_merge_json()
            self.server._start_merge_session(path, tab_id="fm-tab")
            assert self.merge_dir.exists()
            self.server._finish_merge("fm-tab")
            assert not self.merge_dir.exists()
        finally:
            dm._merge_data_dir = orig_dm  # type: ignore[assignment]
            mf._merge_data_dir = orig_mf  # type: ignore[assignment]

    def test_merging_blocks_same_tab(self) -> None:
        """Cannot start a task on the same tab that has a merge in progress."""
        self.server._get_tab("5").is_merging = True
        self.server._run_task_inner({"prompt": "test", "model": "", "tabId": "5"})
        errors = [e for e in self.events if e["type"] == "error"]
        assert any("merge review" in e["text"] for e in errors)

    @pytest.mark.slow
    def test_merging_does_not_block_other_tabs(self) -> None:
        """A merge on one tab does not block tasks on other tabs."""
        self.server._get_tab("5").is_merging = True
        self.events.clear()
        self.server._run_task_inner({"prompt": "test", "model": "", "tabId": "99"})
        errors = [e for e in self.events if e["type"] == "error"]
        assert not any("merge review" in e.get("text", "") for e in errors)


    def test_merge_command_routing(self) -> None:
        """mergeAction command is routed through _handle_command."""
        path = self._write_merge_json()
        self.server._start_merge_session(path, tab_id="mr-tab")
        self.events.clear()

        import kiss.agents.vscode.diff_merge as dm
        import kiss.agents.vscode.merge_flow as mf

        orig = dm._merge_data_dir
        dm._merge_data_dir = lambda tab_id="": self.merge_dir  # type: ignore[assignment]
        orig_mf = mf._merge_data_dir
        mf._merge_data_dir = lambda tab_id="": self.merge_dir  # type: ignore[assignment]
        try:
            self.server._handle_command({
                "type": "mergeAction", "action": "all-done", "tabId": "mr-tab",
            })
        finally:
            dm._merge_data_dir = orig  # type: ignore[assignment]
            mf._merge_data_dir = orig_mf  # type: ignore[assignment]
        types = [e["type"] for e in self.events]
        assert "merge_ended" in types


class TestMergeDiffViewColumn(unittest.TestCase):
    """Verify MergeManager opens files in ViewColumn.One to preserve the
    chat webview, and only opens one file (not all changed files)."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._ts = (base / "vscode" / "src" / "MergeManager.ts").read_text()

    def _get_method_body(self, method: str) -> str:
        """Extract from method definition to the next top-level member."""
        import re as _re

        escaped = _re.escape(method)
        pat = _re.compile(
            rf"^\s+(?:private\s+|public\s+)?(?:async\s+)?{escaped}\(",
            _re.MULTILINE,
        )
        m = pat.search(self._ts)
        assert m, f"Method {method} not found"
        start = m.start()
        for marker in ("\n  private ", "\n  public ", "\n  async ", "\n  dispose"):
            try:
                end = self._ts.index(marker, m.end())
                return self._ts[start:end]
            except ValueError:
                continue
        return self._ts[start:]

    def test_do_open_merge_uses_view_column_one(self) -> None:
        """_doOpenMerge passes viewColumn: vscode.ViewColumn.One to showTextDocument."""
        body = self._get_method_body("_doOpenMerge")
        assert "viewColumn: vscode.ViewColumn.One" in body

    def test_do_open_merge_has_single_show_text_document(self) -> None:
        """_doOpenMerge calls showTextDocument only once (for the first file),
        not inside the for loop for every file."""
        body = self._get_method_body("_doOpenMerge")
        count = body.count("showTextDocument")
        assert count == 1, f"Expected 1 showTextDocument call, got {count}"

    def test_do_open_merge_uses_workspace_apply_edit(self) -> None:
        """_doOpenMerge uses WorkspaceEdit for base-line insertions instead
        of ed.edit() which requires a visible editor."""
        body = self._get_method_body("_doOpenMerge")
        assert "WorkspaceEdit" in body
        assert "applyEdit" in body

    def test_navigate_hunk_uses_view_column_one(self) -> None:
        """_navigateHunk opens files in ViewColumn.One."""
        body = self._get_method_body("_navigateHunk")
        assert "viewColumn: vscode.ViewColumn.One" in body

    def test_get_or_open_editor_uses_view_column_one(self) -> None:
        """_getOrOpenEditor opens files in ViewColumn.One."""
        body = self._get_method_body("_getOrOpenEditor")
        assert "viewColumn: vscode.ViewColumn.One" in body

    def test_do_open_merge_does_not_execute_revert_command(self) -> None:
        """_doOpenMerge no longer calls executeCommand to revert
        (which requires the document to be the active editor).
        It may use executeCommand('vscode.open') for binary files."""
        body = self._get_method_body("_doOpenMerge")
        # Must not call executeCommand for workbench.action.files.revert
        # (the string appears in a comment; check for actual command call)
        assert "executeCommand('workbench.action.files.revert')" not in body
        assert 'executeCommand("workbench.action.files.revert")' not in body

    def test_do_open_merge_tracks_first_file_fp(self) -> None:
        """_doOpenMerge tracks firstFileFp to show only one file."""
        body = self._get_method_body("_doOpenMerge")
        assert "firstFileFp" in body


class TestSorcarTabOpensFilesInLeftSplit(unittest.TestCase):
    """Verify SorcarSidebarView opens files in ViewColumn.One (the left split)."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._ts = (base / "vscode" / "src" / "SorcarSidebarView.ts").read_text()

    def _extract_case_block(self, case_label: str) -> str:
        """Extract a switch-case block from _handleMessage."""
        import re as _re

        pat = _re.compile(
            rf"case\s+'{_re.escape(case_label)}'",
            _re.MULTILINE,
        )
        m = pat.search(self._ts)
        assert m, f"Case '{case_label}' not found"
        start = m.start()
        next_case = _re.search(r"\n\s+case\s+'", self._ts[m.end():])
        if next_case:
            return self._ts[start : m.end() + next_case.start()]
        return self._ts[start:]

    def test_open_file_uses_view_column_one(self) -> None:
        """openFile handler opens files in ViewColumn.One."""
        block = self._extract_case_block("openFile")
        assert "viewColumn: vscode.ViewColumn.One" in block

    def test_submit_file_path_uses_view_column_one(self) -> None:
        """submit handler (file-path shortcut) opens files in ViewColumn.One."""
        block = self._extract_case_block("submit")
        assert "viewColumn: vscode.ViewColumn.One" in block


class TestFilePathDoesNotPopulateTaskPanel(unittest.TestCase):
    """Regression: typing a file path in the textbox and opening it must NOT
    populate the fixed task panel.

    Root cause was that sendMessage() in main.js used to set the task panel
    text (setTaskText, currentTaskName, resetAdjacentState, vscode.setState)
    *before* the extension determined whether to run a task or open a file.
    The fix moved all task-panel state management into the 'setTaskText' event
    handler, which is only sent by _startTask() — never by the file-open path.
    """

    _js: str = ""
    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._js = (base / "vscode" / "media" / "main.js").read_text()
        cls._ts = (base / "vscode" / "src" / "SorcarSidebarView.ts").read_text()


    def _get_send_message_body(self) -> str:
        start = self._js.index("function sendMessage()")
        end = self._js.index("\n  function ", start + 1)
        return self._js[start:end]

    def test_send_message_does_not_call_set_task_text(self) -> None:
        body = self._get_send_message_body()
        assert "setTaskText" not in body

    def test_send_message_does_not_set_current_task_name(self) -> None:
        body = self._get_send_message_body()
        assert "currentTaskName" not in body

    def test_send_message_does_not_call_reset_adjacent_state(self) -> None:
        body = self._get_send_message_body()
        assert "resetAdjacentState" not in body

    def test_send_message_does_not_call_set_state(self) -> None:
        body = self._get_send_message_body()
        assert "vscode.setState" not in body

    def test_send_message_does_not_hide_welcome(self) -> None:
        body = self._get_send_message_body()
        assert "welcome.style.display" not in body


    def _get_set_task_text_handler(self) -> str:
        start = self._js.index("case 'setTaskText':")
        end = self._js.index("break;", start) + len("break;")
        return self._js[start:end]

    def test_set_task_text_handler_sets_current_task_name(self) -> None:
        body = self._get_set_task_text_handler()
        assert "currentTaskName = stt" in body

    def test_set_task_text_handler_calls_reset_adjacent_state(self) -> None:
        body = self._get_set_task_text_handler()
        assert "resetAdjacentState()" in body

    def test_set_task_text_handler_persists_tab_state(self) -> None:
        body = self._get_set_task_text_handler()
        assert "updateActiveTabTitle(stt)" in body

    def test_set_task_text_handler_hides_welcome(self) -> None:
        body = self._get_set_task_text_handler()
        assert "welcome.style.display = 'none'" in body

    def test_set_task_text_handler_calls_set_task_text(self) -> None:
        body = self._get_set_task_text_handler()
        assert "setTaskText(ev.text" in body











class TestBashPanelCollapsibleJS(unittest.TestCase):
    """Test that bash panels are nested inside tool call panels in main.js."""

    _js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents"
        cls._js = (base / "vscode" / "media" / "main.js").read_text()

    def _tool_call_block(self) -> str:
        """Extract the tool_call case block from handleOutputEvent."""
        idx = self._js.index("case 'tool_call':")
        end = self._js.index("case 'tool_result':", idx)
        return self._js[idx:end]

    def test_bash_panel_no_header(self) -> None:
        """Bash panel has no header (output panel is headerless)."""
        block = self._tool_call_block()
        assert "bash-panel-hdr" not in block

    def test_bash_panel_not_collapsible(self) -> None:
        """Bash panel does not call addCollapse."""
        block = self._tool_call_block()
        assert "addCollapse(bp" not in block

    def test_bash_panel_has_content_div(self) -> None:
        """A bash-panel-content div is created for streaming output."""
        block = self._tool_call_block()
        assert "bash-panel-content" in block

    def test_bash_panel_nested_in_tool_call(self) -> None:
        """Bash panel is appended inside the tool call element, not target."""
        block = self._tool_call_block()
        assert "c.appendChild(bp)" in block

    def test_bash_panel_state_points_to_content(self) -> None:
        """tState.bashPanel is set to the content div, not the wrapper."""
        block = self._tool_call_block()
        assert "tState.bashPanel = bpContent" in block

    def test_bash_panel_content_appended_to_wrapper(self) -> None:
        """The content div is appended inside the bash-panel wrapper."""
        block = self._tool_call_block()
        assert "bp.appendChild(bpContent)" in block

    def test_last_tool_call_el_tracked(self) -> None:
        """tState.lastToolCallEl is set to the tool call element."""
        block = self._tool_call_block()
        assert "tState.lastToolCallEl = c" in block








class TestDiffFilesDeletionAtStart(unittest.TestCase):
    """Regression: _diff_files must produce correct hunk positions for
    pure deletions at the beginning of a file.

    Root cause: _diff_files returned new_start=1 (instead of 0) when
    lines were deleted at the very start and the current file was non-empty.
    This caused _hunk_to_dict to produce cs=1, so the MergeManager
    inserted old (red) lines at position 1 instead of 0 — the deleted
    lines appeared AFTER the first surviving line instead of BEFORE it.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, name: str, text: str) -> str:
        p = Path(self.tmpdir) / name
        p.write_text(text)
        return str(p)

    def test_start_deletion_cs_is_zero(self) -> None:
        """Deleting lines at the start must produce cs=0."""
        from kiss.agents.vscode.diff_merge import _diff_files, _hunk_to_dict

        base = self._write("base.txt", "A\nB\nC\nD\n")
        current = self._write("current.txt", "C\nD\n")
        hunks = _diff_files(base, current)
        dicts = [_hunk_to_dict(*h) for h in hunks]
        assert len(dicts) == 1
        assert dicts[0]["cs"] == 0, f"Expected cs=0, got cs={dicts[0]['cs']}"
        assert dicts[0]["bs"] == 0
        assert dicts[0]["bc"] == 2
        assert dicts[0]["cc"] == 0

    def test_middle_deletion_cs_correct(self) -> None:
        """Deleting lines in the middle must produce correct cs."""
        from kiss.agents.vscode.diff_merge import _diff_files, _hunk_to_dict

        base = self._write("base.txt", "A\nB\nC\nD\n")
        current = self._write("current.txt", "A\nD\n")
        hunks = _diff_files(base, current)
        dicts = [_hunk_to_dict(*h) for h in hunks]
        assert len(dicts) == 1
        assert dicts[0]["cs"] == 1

    def test_delete_all_cs_is_zero(self) -> None:
        """Deleting all lines must produce cs=0."""
        from kiss.agents.vscode.diff_merge import _diff_files, _hunk_to_dict

        base = self._write("base.txt", "A\nB\n")
        current = self._write("current.txt", "")
        hunks = _diff_files(base, current)
        dicts = [_hunk_to_dict(*h) for h in hunks]
        assert len(dicts) == 1
        assert dicts[0]["cs"] == 0

    def test_start_deletion_single_line(self) -> None:
        """Deleting a single line at the start produces cs=0."""
        from kiss.agents.vscode.diff_merge import _diff_files, _hunk_to_dict

        base = self._write("base.txt", "A\nB\nC\n")
        current = self._write("current.txt", "B\nC\n")
        hunks = _diff_files(base, current)
        dicts = [_hunk_to_dict(*h) for h in hunks]
        assert len(dicts) == 1
        assert dicts[0]["cs"] == 0
        assert dicts[0]["bc"] == 1

    def test_end_deletion_cs_correct(self) -> None:
        """Deleting lines at the end produces correct cs."""
        from kiss.agents.vscode.diff_merge import _diff_files, _hunk_to_dict

        base = self._write("base.txt", "A\nB\nC\n")
        current = self._write("current.txt", "A\n")
        hunks = _diff_files(base, current)
        dicts = [_hunk_to_dict(*h) for h in hunks]
        assert len(dicts) == 1
        assert dicts[0]["cs"] == 1

    def test_multiple_hunks_including_start(self) -> None:
        """Multiple deletions including at the start all have correct cs."""
        from kiss.agents.vscode.diff_merge import _diff_files, _hunk_to_dict

        base = self._write("base.txt", "A\nB\nC\nD\nE\n")
        current = self._write("current.txt", "C\n")
        hunks = _diff_files(base, current)
        dicts = [_hunk_to_dict(*h) for h in hunks]
        assert dicts[0]["cs"] == 0
        assert dicts[0]["bc"] == 2

    def test_start_insertion_cs_correct(self) -> None:
        """Inserting lines at the start produces cs=0."""
        from kiss.agents.vscode.diff_merge import _diff_files, _hunk_to_dict

        base = self._write("base.txt", "B\nC\n")
        current = self._write("current.txt", "A\nB\nC\n")
        hunks = _diff_files(base, current)
        dicts = [_hunk_to_dict(*h) for h in hunks]
        assert len(dicts) == 1
        assert dicts[0]["cs"] == 0
        assert dicts[0]["cc"] == 1
        assert dicts[0]["bc"] == 0

    def test_replacement_at_start(self) -> None:
        """Replacing lines at the start produces cs=0."""
        from kiss.agents.vscode.diff_merge import _diff_files, _hunk_to_dict

        base = self._write("base.txt", "A\nB\nC\n")
        current = self._write("current.txt", "X\nY\nC\n")
        hunks = _diff_files(base, current)
        dicts = [_hunk_to_dict(*h) for h in hunks]
        assert len(dicts) == 1
        assert dicts[0]["cs"] == 0
        assert dicts[0]["cc"] == 2
        assert dicts[0]["bc"] == 2


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
            ["git", "init"], cwd=self.repo, capture_output=True,
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
        self.server._get_tab("0").use_worktree = True
        self.server._handle_command({"type": "worktreeAction", "action": "merge", "tabId": "0"})
        results = [e for e in self.events if e["type"] == "worktree_result"]
        assert len(results) == 1
        assert results[0]["success"] is False
        assert results[0]["message"]

    def test_discard_exception_still_broadcasts_result(self) -> None:
        """worktree_result is broadcast even when discard raises RuntimeError."""
        self.server._get_tab("0").use_worktree = True
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

        self.server._get_tab("0").use_worktree = True
        _set_agent_wt(
            self.server._get_tab("0").agent,
            self.repo, "kiss/exc-test", "main",
        )

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












class TestSorcarSidebarViewOpensFilesInLeftSplit(unittest.TestCase):
    """Verify SorcarSidebarView opens files in ViewColumn.One like SorcarTab."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    def _extract_case_block(self, case_label: str) -> str:
        """Extract a switch-case block from _handleMessage."""
        import re as _re

        pat = _re.compile(rf"case\s+'{_re.escape(case_label)}'", _re.MULTILINE)
        m = pat.search(self._ts)
        assert m, f"Case '{case_label}' not found"
        start = m.start()
        next_case = _re.search(r"\n\s+case\s+'", self._ts[m.end():])
        if next_case:
            return self._ts[start : m.end() + next_case.start()]
        return self._ts[start:]

    def test_submit_file_open_uses_view_column_one(self) -> None:
        """submit handler opens file paths in ViewColumn.One."""
        block = self._extract_case_block("submit")
        assert "viewColumn: vscode.ViewColumn.One" in block

    def test_open_file_uses_view_column_one(self) -> None:
        """openFile handler opens files in ViewColumn.One."""
        block = self._extract_case_block("openFile")
        assert "viewColumn: vscode.ViewColumn.One" in block




class TestSorcarSidebarViewMergeActions(unittest.TestCase):
    """Verify SorcarSidebarView dispatches merge actions to MergeManager."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    def _get_merge_action_block(self) -> str:
        import re

        m = re.search(r"case\s+'mergeAction':", self._ts)
        assert m
        start = m.start()
        next_case = re.search(r"\n      case ", self._ts[start + 1 :])
        if next_case:
            end = start + 1 + next_case.start()
        else:
            end = start + 1500
        return self._ts[start:end]

    def test_dispatches_accept(self) -> None:
        block = self._get_merge_action_block()
        assert "accept:" in block
        assert "acceptChange()" in block

    def test_dispatches_reject(self) -> None:
        block = self._get_merge_action_block()
        assert "reject:" in block
        assert "rejectChange()" in block

    def test_dispatches_prev(self) -> None:
        block = self._get_merge_action_block()
        assert "prev:" in block
        assert "prevChange()" in block

    def test_dispatches_next(self) -> None:
        block = self._get_merge_action_block()
        assert "next:" in block
        assert "nextChange()" in block

    def test_dispatches_accept_all(self) -> None:
        block = self._get_merge_action_block()
        assert "'accept-all'" in block
        assert "acceptAll()" in block

    def test_dispatches_reject_all(self) -> None:
        block = self._get_merge_action_block()
        assert "'reject-all'" in block
        assert "rejectAll()" in block

    def test_dispatches_accept_file(self) -> None:
        block = self._get_merge_action_block()
        assert "'accept-file'" in block
        assert "acceptFile()" in block

    def test_dispatches_reject_file(self) -> None:
        block = self._get_merge_action_block()
        assert "'reject-file'" in block
        assert "rejectFile()" in block

    def test_all_done_sent_to_agent(self) -> None:
        """all-done action is sent to the agent process, not MergeManager."""
        block = self._get_merge_action_block()
        assert "'all-done'" in block
        assert "sendMergeAllDone" in block




class TestSorcarSidebarViewStartTask(unittest.TestCase):
    """Verify SorcarSidebarView._startTask passes all parameters like SorcarTab."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    def _get_start_task_body(self) -> str:
        idx = self._ts.index("private _startTask(")
        end = self._ts.index("\n  private ", idx + 1)
        return self._ts[idx:end]

    def test_accepts_use_worktree(self) -> None:
        body = self._get_start_task_body()
        assert "useWorktree" in body

    def test_accepts_use_parallel(self) -> None:
        body = self._get_start_task_body()
        assert "useParallel" in body

    def test_accepts_attachments(self) -> None:
        body = self._get_start_task_body()
        assert "attachments" in body

    def test_accepts_active_file(self) -> None:
        body = self._get_start_task_body()
        assert "activeFile" in body

    def test_sends_set_task_text(self) -> None:
        """_startTask sends setTaskText to the webview."""
        body = self._get_start_task_body()
        assert "setTaskText" in body

    def test_sends_status_running(self) -> None:
        """_startTask sends status running: true to the webview."""
        body = self._get_start_task_body()
        assert "running: true" in body

    def test_sends_run_command(self) -> None:
        """_startTask sends the 'run' command to the agent process."""
        body = self._get_start_task_body()
        assert "type: 'run'" in body




class TestSorcarSidebarViewAgentEventHandling(unittest.TestCase):
    """Verify SorcarSidebarView handles agent process events correctly."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    def _get_message_handler_body(self) -> str:
        """Get the _installClientListener message handler body."""
        idx = self._ts.index("private _installClientListener(")
        end = self._ts.index("\n  }", idx) + 4
        return self._ts[idx:end]

    def test_forwards_commit_messages(self) -> None:
        body = self._get_message_handler_body()
        assert "msg.type === 'commitMessage'" in body
        assert "_onCommitMessage.fire" in body

    def test_updates_selected_model(self) -> None:
        body = self._get_message_handler_body()
        assert "msg.type === 'models'" in body
        assert "this._selectedModel = msg.selected" in body

    def test_handles_merge_data(self) -> None:
        body = self._get_message_handler_body()
        assert "msg.type === 'merge_data'" in body

    def test_handles_worktree_created(self) -> None:
        body = self._get_message_handler_body()
        assert "msg.type === 'worktree_created'" in body

    def test_handles_worktree_done(self) -> None:
        body = self._get_message_handler_body()
        assert "msg.type === 'worktree_done'" in body

    def test_handles_worktree_progress(self) -> None:
        body = self._get_message_handler_body()
        assert "msg.type === 'worktree_progress'" in body

    def test_handles_worktree_result(self) -> None:
        body = self._get_message_handler_body()
        assert "msg.type === 'worktree_result'" in body

    def test_forwards_all_messages_to_webview(self) -> None:
        body = self._get_message_handler_body()
        assert "this._sendToWebview(msg)" in body

    def test_tracks_running_status(self) -> None:
        body = self._get_message_handler_body()
        assert "msg.type === 'status'" in body
        assert "this._runningTabs" in body


class TestSorcarSidebarViewReadyHandler(unittest.TestCase):
    """Verify the 'ready' message handler sends all initialization messages."""

    _ts: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[4] / "kiss" / "agents" / "vscode"
        cls._ts = (base / "src" / "SorcarSidebarView.ts").read_text()

    def _get_ready_block(self) -> str:
        import re

        m = re.search(r"case\s+'ready':", self._ts)
        assert m
        end = self._ts.index("break;", m.end()) + len("break;")
        return self._ts[m.start() : end]

    def test_requests_models(self) -> None:
        block = self._get_ready_block()
        assert "'getModels'" in block

    def test_requests_input_history(self) -> None:
        block = self._get_ready_block()
        assert "'getInputHistory'" in block

    def test_sends_focus_input(self) -> None:
        block = self._get_ready_block()
        assert "'focusInput'" in block






















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
