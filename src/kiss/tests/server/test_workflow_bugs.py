# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for worktree workflow bugs on the server side.

Confirms two bugs, each demonstrated by a test that fails before the fix
and passes after:

BUG 3 — ``_commit_worktree`` crashes when the LLM is unavailable
    ``_generate_commit_message`` calls the LLM with no fallback.  If
    no provider is configured or the API is unreachable, the exception
    propagates uncaught, preventing worktree finalization and blocking
    all subsequent tasks on that agent.

BUG 4 — ``_close_tab`` orphans pending worktrees
    Closing a tab with a pending worktree drops the in-memory
    reference without parking the worktree.  The worktree directory
    and branch persist in git with nobody owning them, and the orphan
    reclaim sweep then treats the user's unreviewed work as its own.
    The fix retires the worktree on close: the work is committed onto
    its ``kiss/wt-*`` branch, the directory is removed, the branch is
    marked preserved, and nothing is merged into the user's branch.

No mocks, patches, fakes, or test doubles; the commit-message
generator runs with no provider configured (:class:`OfflineFastModel`)
so it takes its real fallback path.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.agents.sorcar.test_workflow_bugs import (  # noqa: F401
    _make_repo,
)
from kiss.tests.conftest import seed_worktree
from kiss.tests.server.conftest import redirect_persistence, restore_persistence
from kiss.tests.server.parallel_agent_harness import OfflineFastModel


def _make_wt_with_commit(
    repo: Path, branch: str, agent: ChatSorcarAgent,
) -> GitWorktree:
    slug = branch.replace("/", "_")
    wt_dir = repo / ".kiss-worktrees" / slug
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    GitWorktreeOps.save_original_branch(repo, branch, "main")
    (wt_dir / "agent.txt").write_text("agent work\n")
    subprocess.run(
        ["git", "-C", str(wt_dir), "add", "."],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(wt_dir), "commit", "-m", "agent"],
        capture_output=True, check=True,
    )
    wt = GitWorktree(
        repo_root=repo,
        branch=branch,
        original_branch="main",
        wt_dir=wt_dir,
    )
    seed_worktree(agent, wt)
    return wt


def _server(repo: Path) -> tuple[AgentServer, list[dict[str, Any]]]:
    """Real server whose printer broadcasts are captured into a list."""
    server = AgentServer()
    server.work_dir = str(repo)
    events: list[dict[str, Any]] = []

    def capture(event: dict[str, Any]) -> None:
        events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


def _register_tab_state(
    tab_id: str, agent: ChatSorcarAgent,
) -> agent_state.AgentState:
    """Register a server-owned idle state for *tab_id* with *agent*."""
    state = agent_state.AgentState(
        uuid.uuid4().hex,
        agent=agent,
        tab_id=tab_id,
        server_owned=True,
    )
    agent_state.register(state)
    return state


class TestBug3AutoCommitNoLLMFallback:
    """_commit_worktree must not crash when the LLM is unavailable."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = redirect_persistence(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.offline = OfflineFastModel()
        self.offline.__enter__()

    def teardown_method(self) -> None:
        self.offline.__exit__(None, None, None)
        agent_state.agent_states.clear()
        restore_persistence(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_commit_fallback_without_llm(self) -> None:
        """_commit_worktree commits with a fallback message with no LLM."""
        agent = ChatSorcarAgent("test")
        branch = "kiss/wt-test-llm-fail"
        wt = _make_wt_with_commit(self.repo, branch, agent)

        (wt.wt_dir / "extra.txt").write_text("extra work\n")

        try:
            result = agent._commit_worktree(wt)
            assert result is True, (
                "BUG 3: _commit_worktree should commit with a fallback "
                "message when no LLM is available, but returned False"
            )

            assert not GitWorktreeOps.has_uncommitted_changes(wt.wt_dir), (
                "BUG 3: worktree still has uncommitted changes after "
                "_commit_worktree without an LLM"
            )
            tracked = subprocess.run(
                ["git", "-C", str(wt.wt_dir), "ls-files"],
                capture_output=True, text=True, check=True,
            ).stdout.split()
            assert "extra.txt" in tracked
            subject = subprocess.run(
                ["git", "-C", str(wt.wt_dir), "log", "-1", "--format=%s"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            assert subject, "the fallback commit has no subject"
        finally:
            GitWorktreeOps.remove(self.repo, wt.wt_dir)
            GitWorktreeOps.prune(self.repo)
            GitWorktreeOps.delete_branch(self.repo, branch)


class TestBug4CloseTabOrphansWorktree:
    """Closing a tab with a pending worktree must park it on its branch."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = redirect_persistence(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self.offline = OfflineFastModel()
        self.offline.__enter__()

    def teardown_method(self) -> None:
        self.offline.__exit__(None, None, None)
        agent_state.agent_states.clear()
        restore_persistence(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _main_head(self) -> str:
        return subprocess.run(
            ["git", "-C", str(self.repo), "rev-parse", "main"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    def test_close_tab_parks_pending_worktree(self) -> None:
        """Closing a tab commits the work to its branch, never to main."""
        server, events = _server(self.repo)
        tab_id = "tab-close-test"
        agent = ChatSorcarAgent("Sorcar")
        state = _register_tab_state(tab_id, agent)

        branch = "kiss/wt-close-test"
        wt = _make_wt_with_commit(self.repo, branch, agent)
        (wt.wt_dir / "late.txt").write_text("uncommitted work\n")
        head_before = self._main_head()

        assert agent._wt_pending

        server._close_tab(tab_id)

        assert not wt.wt_dir.exists(), (
            "BUG 4: _close_tab left orphaned worktree directory"
        )
        registered = [
            b for _d, b in GitWorktreeOps.registered_worktrees(self.repo)
        ]
        assert branch not in registered, (
            "BUG 4: _close_tab left the branch registered as a worktree "
            "with no owner"
        )
        assert GitWorktreeOps.branch_exists(self.repo, branch), (
            "BUG 4: _close_tab deleted the branch holding unreviewed work"
        )
        files = subprocess.run(
            ["git", "-C", str(self.repo), "show", "--name-only",
             "--pretty=format:", branch],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        assert "late.txt" in files, (
            "BUG 4: the uncommitted work was not committed to the branch"
        )
        assert GitWorktreeOps.load_preserve_marker(self.repo, branch), (
            "BUG 4: the parked branch has no preserve marker; a reclaim "
            "sweep could publish or delete it"
        )
        assert self._main_head() == head_before, (
            "_close_tab merged the pending worktree into main"
        )
        assert not (self.repo / "agent.txt").exists()
        assert not (self.repo / "late.txt").exists()
        assert agent_state.find_by_tab(tab_id) is None
        assert agent_state.get(state.task_id) is None
        warnings = [
            e.get("message", "") for e in events if e.get("type") == "warning"
        ]
        assert any(branch in w and "not merged" in w for w in warnings), (
            f"the user was not told where the work went: {warnings}"
        )

    def test_close_tab_no_changes_leaves_main_untouched(self) -> None:
        """Closing a tab with a no-change worktree invents no work."""
        server, _events = _server(self.repo)
        tab_id = "tab-close-empty"
        agent = ChatSorcarAgent("Sorcar")
        state = _register_tab_state(tab_id, agent)

        branch = "kiss/wt-close-empty"
        slug = branch.replace("/", "_")
        wt_dir = self.repo / ".kiss-worktrees" / slug
        assert GitWorktreeOps.create(self.repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(self.repo, branch, "main")
        seed_worktree(agent, GitWorktree(
            repo_root=self.repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
        ))
        assert agent._wt_pending
        head_before = self._main_head()

        server._close_tab(tab_id)

        assert not wt_dir.exists(), (
            "BUG 4: empty worktree directory left on disk after tab close"
        )
        registered = [
            b for _d, b in GitWorktreeOps.registered_worktrees(self.repo)
        ]
        assert branch not in registered
        assert self._main_head() == head_before
        if GitWorktreeOps.branch_exists(self.repo, branch):
            ahead = subprocess.run(
                ["git", "-C", str(self.repo), "rev-list", f"main..{branch}"],
                capture_output=True, text=True, check=True,
            ).stdout.split()
            assert ahead == [], (
                f"an empty worktree grew commits on close: {ahead}"
            )
        assert agent_state.get(state.task_id) is None

    def test_close_tab_active_task_no_cleanup(self) -> None:
        """Closing a tab with an active task must NOT clean up worktree."""
        server, _events = _server(self.repo)
        tab_id = "tab-active"
        agent = ChatSorcarAgent("Sorcar")
        state = _register_tab_state(tab_id, agent)
        state.is_task_active = True

        branch = "kiss/wt-active-test"
        _make_wt_with_commit(self.repo, branch, agent)

        server._close_tab(tab_id)

        assert agent_state.get(state.task_id) is state
        assert GitWorktreeOps.branch_exists(self.repo, branch)

        state.is_task_active = False
        GitWorktreeOps.remove(self.repo, agent._wt.wt_dir)  # type: ignore[union-attr]
        GitWorktreeOps.prune(self.repo)
        GitWorktreeOps.delete_branch(self.repo, branch)
