# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 17: Integration tests for race conditions in worktree mode.

BUG-71: ``VSCodeServer._new_chat`` does not check whether a worktree
    task is currently running on the same tab.  When the user triggers
    "new chat" while the agent is actively writing files into the
    worktree directory, the server calls ``tab.agent.new_chat()``
    which invokes ``_release_worktree() -> _finalize_worktree()``.
    The latter force-removes ``wt_dir`` mid-write, corrupting the
    agent's in-progress edits, the baseline commit, and produces a
    garbage squash-merge into the original branch.

    The existing BUG-44 / BUG-35 checks only guard against concurrent
    *non-worktree* tasks (via ``_any_non_wt_running``); they do not
    protect the tab's own running worktree task.

BUG-72: ``VSCodeServer._handle_worktree_action("merge"/"discard")``
    only guards against concurrent non-worktree tasks via
    ``_any_non_wt_running``, but does NOT check whether the tab's own
    worktree task is still running.  A misbehaving client (or a race
    between ``worktree_done`` broadcast and ``task_thread`` cleanup)
    can trigger ``agent.merge()`` or ``agent.discard()`` while the
    agent thread is still writing to ``wt_dir`` — same destruction
    pattern as BUG-71.

Both bugs have a common root cause: there is no per-tab "a task is
actively executing ``agent.run()``" flag.  The fix adds
``AgentState.is_task_active`` (set True immediately before the
``agent.run()`` loop and cleared in the post-task ``finally`` block
BEFORE ``worktree_done`` is broadcast), and wires it into the
``_handle_worktree_action`` guard (``_check_worktree_busy``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _register_state(
    task_id: str,
    tab_id: str,
    agent: WorktreeSorcarAgent,
    *,
    use_worktree: bool = True,
    is_task_active: bool = False,
) -> agent_state.AgentState:
    """Register a server-owned AgentState for *tab_id* running *agent*."""
    state = agent_state.AgentState(
        task_id,
        agent=agent,
        tab_id=tab_id,
        server_owned=True,
        is_task_active=is_task_active,
    )
    state.use_worktree = use_worktree
    agent_state.register(state)
    return state


def _make_repo(path: Path) -> Path:
    """Create a minimal git repo with one initial commit."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"],
        capture_output=True, check=True,
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


def _create_wt(
    repo: Path, branch: str, agent: WorktreeSorcarAgent,
) -> GitWorktree:
    """Create a real git worktree + branch and assign it to *agent*."""
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
    agent._wt = wt
    return wt


class _RecordingPrinter:
    """Concrete printer that records every broadcast call."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)


class TestNewChatSimpleBroadcast:
    """``_new_chat`` on a fresh tab id simply resets the agent's chat
    state and broadcasts ``showWelcome``.  (Earlier "refuse while a task
    is running" guards were removed because the frontend always mints a
    new tab id per ``newChat`` command, so the backend never observes a
    tab whose previous run is still active.)
    """

    def test_new_chat_regression_allowed_when_no_task(
        self, tmp_path: Path,
    ) -> None:
        """Regression: when no task is active the existing behavior
        (call ``tab.agent.new_chat()`` and broadcast ``showWelcome``)
        must be preserved."""
        repo = _make_repo(tmp_path / "repo")

        server = VSCodeServer()
        server.work_dir = str(repo)
        printer = _RecordingPrinter()
        server.printer = cast(Any, printer)

        tab_id = "tab-bug71-ok"
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        agent._wt = None
        state = _register_state("task-bug71-ok", tab_id, agent)
        try:
            server._new_chat(tab_id)
            welcome = [
                e for e in printer.events if e.get("type") == "showWelcome"
            ]
            assert welcome, (
                "Regression: showWelcome must be broadcast when no task "
                f"is active.  Events: {printer.events}"
            )
        finally:
            agent_state.unregister(state.task_id, state)


class TestBug72WorktreeActionDuringRunningTask:
    """``_handle_worktree_action("merge"/"discard")`` must refuse while
    the tab's own worktree task is still writing to ``wt_dir``."""

    def test_merge_refused_while_task_active(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")

        server = VSCodeServer()
        server.work_dir = str(repo)
        printer = _RecordingPrinter()
        server.printer = cast(Any, printer)

        tab_id = "tab-bug72-merge"
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        wt = _create_wt(repo, "kiss/wt-bug72m-1", agent)

        state = _register_state(
            "task-bug72-merge", tab_id, agent, is_task_active=True,
        )
        assert wt.wt_dir.exists()

        try:
            result = server._handle_worktree_action("merge", tab_id)
        finally:
            agent_state.unregister(state.task_id, state)

        assert result["success"] is False, (
            "BUG-72: merge must be refused while a worktree task is "
            f"running.  result={result}"
        )
        assert "running" in result["message"].lower() or \
               "task" in result["message"].lower(), (
            "BUG-72: merge refusal message should mention running "
            f"task.  result={result}"
        )

        assert agent._wt is not None, (
            "BUG-72: agent.merge() executed and cleared the worktree "
            "reference despite the task still running."
        )
        assert wt.wt_dir.exists(), (
            "BUG-72: wt_dir was removed by agent.merge() mid-write."
        )

    def test_discard_refused_while_task_active(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")

        server = VSCodeServer()
        server.work_dir = str(repo)
        printer = _RecordingPrinter()
        server.printer = cast(Any, printer)

        tab_id = "tab-bug72-discard"
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        wt = _create_wt(repo, "kiss/wt-bug72d-1", agent)

        state = _register_state(
            "task-bug72-discard", tab_id, agent, is_task_active=True,
        )
        assert wt.wt_dir.exists()

        try:
            result = server._handle_worktree_action("discard", tab_id)
        finally:
            agent_state.unregister(state.task_id, state)

        assert result["success"] is False, (
            "BUG-72: discard must be refused while a worktree task "
            f"is running.  result={result}"
        )
        assert "running" in result["message"].lower() or \
               "task" in result["message"].lower(), (
            "BUG-72: discard refusal message should mention running "
            f"task.  result={result}"
        )

        assert agent._wt is not None, (
            "BUG-72: agent.discard() executed and cleared the worktree."
        )
        assert wt.wt_dir.exists(), (
            "BUG-72: wt_dir was removed by agent.discard() mid-write."
        )
