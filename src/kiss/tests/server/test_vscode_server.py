# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the VS Code extension backend server.

Tests cover: model picker (vendor ordering, sorting, grouping, pricing),
file picker (sorting by usage/recency/end-distance, section grouping),
and worktree action handling (merge/discard routing, guards, exception
handling).  The tests that read the extension's frontend sources
(``media/main.js``, ``SorcarSidebarView.ts``) live in
``kiss.tests.agents.vscode.test_vscode_server``.
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




if __name__ == "__main__":
    unittest.main()
