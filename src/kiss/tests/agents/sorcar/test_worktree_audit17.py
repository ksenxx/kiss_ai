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
``_TabState.is_task_active`` (set True immediately before the
``agent.run()`` loop and cleared in the post-task ``finally`` block
BEFORE ``worktree_done`` is broadcast), and wires it into the
``_new_chat`` and ``_handle_worktree_action`` guards.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.server import VSCodeServer

# ---------------------------------------------------------------------------
# Helpers (mirror audit16 / audit15 test helpers)
# ---------------------------------------------------------------------------


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


# ===========================================================================
# BUG-71: _new_chat races with a running worktree task
# ===========================================================================


class TestBug71NewChatDuringRunningWorktreeTask:
    """``_new_chat`` must refuse while the tab's own worktree task is
    still running, otherwise ``tab.agent.new_chat()`` removes the
    worktree directory mid-write."""

    def test_new_chat_refused_while_task_active(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")

        server = VSCodeServer()
        server.work_dir = str(repo)
        printer = _RecordingPrinter()
        server.printer = cast(Any, printer)

        tab_id = "tab-bug71"
        tab = server._get_tab(tab_id)
        tab.use_worktree = True

        agent = cast(WorktreeSorcarAgent, tab.agent)
        wt = _create_wt(repo, "kiss/wt-bug71-1", agent)

        # Simulate an in-flight task (``agent.run`` is between the
        # start and end of the ``_run_task_inner`` for-loop body).
        tab.is_task_active = True

        # Worktree directory must exist before the race.
        assert wt.wt_dir.exists(), "Pre-condition: wt_dir must exist"

        server._new_chat(tab_id)

        # BUG-71 fix: server rejects with an error broadcast rather
        # than calling ``tab.agent.new_chat()`` -> ``_release_worktree``.
        errors = [
            e for e in printer.events
            if e.get("type") == "error"
            and "task" in e.get("text", "").lower()
            and "running" in e.get("text", "").lower()
        ]
        assert errors, (
            "BUG-71: _new_chat did not broadcast an error when a "
            "worktree task was still running. "
            f"Events: {printer.events}"
        )

        # The agent's worktree reference must NOT have been cleared.
        assert agent._wt is not None, (
            "BUG-71: agent._wt was cleared despite task still running."
        )

        # The worktree directory must still exist — a running task is
        # writing to it; removing it would corrupt the task's state.
        assert wt.wt_dir.exists(), (
            "BUG-71: wt_dir was removed while the task was still "
            "writing to it (tab.agent.new_chat -> _release_worktree "
            "-> _finalize_worktree -> GitWorktreeOps.remove)."
        )

        # showWelcome (which resets the UI) must NOT have been sent.
        welcome = [
            e for e in printer.events if e.get("type") == "showWelcome"
        ]
        assert not welcome, (
            "BUG-71: showWelcome broadcast despite the refusal. "
            f"Events: {printer.events}"
        )

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
        tab = server._get_tab(tab_id)
        tab.use_worktree = True
        tab.is_task_active = False

        agent = cast(WorktreeSorcarAgent, tab.agent)
        # No pending worktree — simplest path: new_chat just resets
        # chat state + broadcasts showWelcome.
        agent._wt = None

        server._new_chat(tab_id)
        welcome = [
            e for e in printer.events if e.get("type") == "showWelcome"
        ]
        assert welcome, (
            "Regression: showWelcome must be broadcast when no task "
            f"is active.  Events: {printer.events}"
        )


# ===========================================================================
# BUG-72: merge / discard during a running worktree task
# ===========================================================================


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
        tab = server._get_tab(tab_id)
        tab.use_worktree = True

        agent = cast(WorktreeSorcarAgent, tab.agent)
        wt = _create_wt(repo, "kiss/wt-bug72m-1", agent)

        tab.is_task_active = True
        assert wt.wt_dir.exists()

        result = server._handle_worktree_action("merge", tab_id)

        assert result["success"] is False, (
            "BUG-72: merge must be refused while a worktree task is "
            f"running.  result={result}"
        )
        assert "running" in result["message"].lower() or \
               "task" in result["message"].lower(), (
            "BUG-72: merge refusal message should mention running "
            f"task.  result={result}"
        )

        # Side effects — ``agent.merge()`` must NOT have run.
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
        tab = server._get_tab(tab_id)
        tab.use_worktree = True

        agent = cast(WorktreeSorcarAgent, tab.agent)
        wt = _create_wt(repo, "kiss/wt-bug72d-1", agent)

        tab.is_task_active = True
        assert wt.wt_dir.exists()

        result = server._handle_worktree_action("discard", tab_id)

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


# ===========================================================================
# BUG-71 / BUG-72 enabler: _TabState must have an ``is_task_active`` flag
# ===========================================================================


class TestIsTaskActiveFlagExists:
    """The fix relies on a new ``_TabState.is_task_active`` flag that
    is cleared on construction and set True while a task is active.
    """

    def test_is_task_active_present_and_false_by_default(
        self, tmp_path: Path,
    ) -> None:
        server = VSCodeServer()
        server.work_dir = str(tmp_path)
        tab = server._get_tab("tab-new")
        assert hasattr(tab, "is_task_active"), (
            "_TabState must expose ``is_task_active`` (required by "
            "the BUG-71 / BUG-72 fix)."
        )
        assert tab.is_task_active is False, (
            "``is_task_active`` must default to False on a fresh tab."
        )
