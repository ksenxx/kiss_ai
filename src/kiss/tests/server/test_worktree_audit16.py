# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 16: Integration tests for bugs/inconsistencies found in audit 16.

BUG-68: post-task pending-worktree cleanup silently left a pending
    empty-change worktree when another tab's task was running (the
    since-removed ``_any_non_wt_running()`` main-tree claim) — the user
    saw no buttons and had no indication that a worktree existed.
    (Historically observed through the since-removed ``_finish_merge``;
    the shared logic now lives in ``_present_pending_worktree``.)

    Resolved by removing the cause rather than reporting it: an
    empty worktree is now discarded even while another tab's task runs,
    since that discard touches neither the main working tree's files
    nor its HEAD.  There is no orphaned branch left to tell the user
    about, and no meaningless merge/discard prompt.  See
    ``test_worktree_leak_when_main_tree_busy.py``.

RED-10: The post-task pending-worktree handling blocks in
    ``_run_task_inner`` and ``_emit_pending_worktree`` duplicated the
    same "auto-discard or emit worktree_done" logic with subtle
    divergences.  A single helper (``_present_pending_worktree``)
    eliminates the redundancy and prevents future drift.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import AgentServer
from kiss.tests.conftest import seed_worktree


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    agent_state.agent_states.clear()
    yield
    agent_state.agent_states.clear()


def _make_repo(path: Path) -> Path:
    """Create a minimal git repo with one initial commit."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"],
        capture_output=True,
        check=True,
    )
    (path / "init.txt").write_text("init\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "init"],
        capture_output=True, check=True,
    )
    return path


def _register_wt_tab(task_id: str, tab_id: str) -> AgentState:
    """Register a server-owned worktree-mode state with a real agent."""
    state = AgentState(
        task_id,
        agent=ChatSorcarAgent("Sorcar"),
        tab_id=tab_id,
        server_owned=True,
    )
    agent_state.register(state)
    return state


def _create_wt(
    repo: Path, branch: str, agent: ChatSorcarAgent,
) -> GitWorktree:
    """Create a real worktree + branch and assign it to *agent*."""
    slug = branch.replace("/", "_")
    wt_dir = repo / ".kiss-worktrees" / slug
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    GitWorktreeOps.save_original_branch(repo, branch, "main")
    wt = GitWorktree(
        repo_root=repo,
        branch=branch,
        original_branch="main",
        wt_dir=wt_dir,
    )
    seed_worktree(agent, wt)
    return wt


class _RecordingPrinter:
    """Concrete printer that records every broadcast call."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def print(self, *args: Any, **kwargs: Any) -> None:
        pass


class TestBug68PresentPendingNoBroadcastOnEmptyNonWtBusy:
    """``_present_pending_worktree`` with no worktree changes and a
    concurrent task on another tab must discard the empty worktree
    WITHOUT broadcasting the meaningless ``worktree_done`` prompt.

    The frontend renders ``worktree_done`` as "Auto-commit and merge
    or Discard?", which makes no sense when there are zero changed
    files.  Leaving the branch behind instead is not an option
    either: nothing ever retried the cleanup, so the worktree leaked
    permanently.  Discarding an empty worktree is safe while the main
    repo is busy — it touches neither its files nor its HEAD.
    """

    def test_present_pending_empty_wt_other_tab_busy(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")

        server = AgentServer()
        server.work_dir = str(repo)
        printer = _RecordingPrinter()
        server.printer = cast(Any, printer)

        tab_id = "tab-bug68a"
        state = _register_wt_tab("task-bug68a", tab_id)

        agent = cast(ChatSorcarAgent, state.agent)
        branch = "kiss/wt-bug68a-1"
        wt = _create_wt(repo, branch, agent)

        other = AgentState("task-other-68a", tab_id="other-bug68a")
        other.is_task_active = True
        agent_state.register(other)

        server._present_pending_worktree(tab_id)

        wt_done = [e for e in printer.events if e.get("type") == "worktree_done"]
        assert not wt_done, (
            "worktree_done must NOT be broadcast for an empty worktree — "
            "the resulting merge/discard prompt is meaningless when "
            f"there are no changes.  Events: {printer.events}"
        )
        assert agent._wt is None, (
            "BUG-68: another tab's running task blocked the discard of an empty "
            "worktree, leaking the branch and directory forever."
        )
        assert not GitWorktreeOps.branch_exists(repo, branch), (
            "branch survived the auto-discard."
        )
        assert not wt.wt_dir.exists(), (
            "worktree directory survived the auto-discard."
        )

        other.is_task_active = False

    def test_present_pending_empty_wt_idle_discards(
        self, tmp_path: Path,
    ) -> None:
        """Regression: when no other task is running, the empty
        worktree must still be auto-discarded (BUG-42 behavior)."""
        repo = _make_repo(tmp_path / "repo")
        server = AgentServer()
        server.work_dir = str(repo)
        printer = _RecordingPrinter()
        server.printer = cast(Any, printer)

        tab_id = "tab-bug68b"
        state = _register_wt_tab("task-bug68b", tab_id)

        agent = cast(ChatSorcarAgent, state.agent)
        _create_wt(repo, "kiss/wt-bug68b-1", agent)

        server._present_pending_worktree(tab_id)

        assert agent._wt is None, (
            "Regression: empty worktree was not auto-discarded when "
            "no other task was running."
        )


class TestRed10PostTaskPendingWtDuplication:
    """All three call sites should share a single helper that
    auto-discards or emits worktree_done on empty changes."""

    def test_unified_helper_exists(self) -> None:
        """After the fix, a single helper handles the post-task
        pending-worktree logic.  The helper must exist so future
        changes don't drift between the three sites."""
        assert hasattr(AgentServer, "_present_pending_worktree"), (
            "RED-10: the post-task pending-worktree logic is still "
            "duplicated across _run_task_inner and "
            "_emit_pending_worktree.  Expected a single helper "
            "`_present_pending_worktree`."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
