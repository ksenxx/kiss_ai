# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests verifying fixes for worktree bugs BUG-8 and BUG-9.

Each test verifies the CORRECT behavior after the fix was applied.  The
worktree is created the way a task does it, through the agent's own
``create_worktree`` tool; no mocks, patches or test doubles.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import _git
from kiss.server import agent_state
from kiss.tests.server.conftest import redirect_persistence, restore_persistence
from kiss.tests.server.worktree_helpers import worktree_changed_files


def _attach_worktree_state(
    agent: ChatSorcarAgent, tab_id: str = "0",
) -> agent_state.AgentState:
    """Register a server-owned worktree AgentState for *agent* on *tab_id*.

    Mirrors what the server does for a UI-launched run: the state is
    keyed by a minted task id and carries the launching tab id so
    ``agent_state.find_by_tab`` used by the merge-flow methods resolves
    the agent; the merge flow then decides by ``agent._wt_pending``.
    """
    st = agent_state.AgentState(
        uuid.uuid4().hex,
        agent=agent,
        tab_id=tab_id,
        server_owned=True,
    )
    agent_state.register(st)
    return st


def _create_worktree(agent: ChatSorcarAgent, repo: Path) -> None:
    """Create *agent*'s worktree in *repo* through its own tool."""
    agent.work_dir = str(repo)
    result = agent.create_worktree()
    assert result.startswith("Created worktree"), result


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        capture_output=True,
        check=True,
    )
    (path / "README.md").write_text("# Test\n")
    (path / "fileA.txt").write_text("original A\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True,
        check=True,
    )
    return path


def _make_server(repo: Path) -> tuple:
    from kiss.server.server import AgentServer

    server = AgentServer()
    events: list[dict] = []

    def capture(event: dict) -> None:
        events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    server.work_dir = str(repo)
    return server, events


class TestBug8Fix:
    """After fix, the changed-files listing only reports files the agent
    actually changed, even when the original branch has advanced.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = redirect_persistence(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        agent_state.agent_states.clear()
        restore_persistence(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_changed_files_excludes_unrelated_after_main_advances(self) -> None:
        """Only agent-modified files appear, not unrelated files from main."""

        server, events = _make_server(self.repo)
        agent = ChatSorcarAgent("Sorcar")
        _attach_worktree_state(agent)

        _create_worktree(agent, self.repo)
        wt_dir = agent._wt_dir
        assert wt_dir is not None and wt_dir.exists()

        (wt_dir / "fileA.txt").write_text("agent modified A\n")

        original_branch = agent.worktrees[0].original_branch
        assert original_branch is not None

        tmp_wt = self.repo / ".kiss-worktrees" / "tmp_advance"
        _git("worktree", "add", "-b", "tmp-advance", str(tmp_wt), cwd=self.repo)
        (tmp_wt / "unrelated_file.txt").write_text("unrelated content\n")
        _git("add", "-A", cwd=tmp_wt)
        _git("commit", "-m", "advance with unrelated file", cwd=tmp_wt)
        _git("worktree", "remove", str(tmp_wt), "--force", cwd=self.repo)
        _git("checkout", original_branch, cwd=self.repo)
        _git("merge", "--ff-only", "tmp-advance", cwd=self.repo)
        _git("branch", "-d", "tmp-advance", cwd=self.repo)

        changed = worktree_changed_files(server, "0")

        assert "fileA.txt" in changed
        assert "unrelated_file.txt" not in changed, (
            "BUG-8 FIX: unrelated files from main advancement "
            "should NOT appear as changed"
        )

        agent.discard()

    def test_changed_files_still_reports_agent_changes(self) -> None:
        """Sanity check: agent-modified files are still reported correctly."""

        server, events = _make_server(self.repo)
        agent = ChatSorcarAgent("Sorcar")
        _attach_worktree_state(agent)

        _create_worktree(agent, self.repo)
        wt_dir = agent._wt_dir
        assert wt_dir is not None

        (wt_dir / "fileA.txt").write_text("agent modified A\n")
        (wt_dir / "new_file.txt").write_text("brand new\n")

        changed = worktree_changed_files(server, "0")
        assert "fileA.txt" in changed
        assert "new_file.txt" in changed

        agent.discard()


class TestBug9Fix:
    """After fix, presenting a worktree does NOT commit its changes."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_saved = redirect_persistence(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        agent_state.agent_states.clear()
        restore_persistence(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_present_pending_worktree_does_not_commit(self) -> None:
        """_present_pending_worktree must not auto-commit."""

        server, events = _make_server(self.repo)
        agent = ChatSorcarAgent("Sorcar")
        _attach_worktree_state(agent)

        _create_worktree(agent, self.repo)
        wt_dir = agent._wt_dir
        branch = agent._wt_branch
        original = agent.worktrees[0].original_branch
        assert wt_dir is not None and branch is not None and original is not None

        (wt_dir / "agent_output.txt").write_text("work\n")

        server._present_pending_worktree("0")

        r = subprocess.run(
            ["git", "-C", str(self.repo), "rev-list", "--count",
             f"{original}..{branch}"],
            capture_output=True, text=True,
        )
        assert r.stdout.strip() == "0", (
            "BUG-9 FIX: _present_pending_worktree must not auto-commit"
        )

        agent.discard()
