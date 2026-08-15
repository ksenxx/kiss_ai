# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 16: Integration tests for bugs/inconsistencies found in audit 16.

BUG-68: post-task pending-worktree cleanup silently left a pending
    empty-change worktree when ``_any_non_wt_running()`` was True —
    the user saw no buttons and had no indication that a worktree
    existed.  (Historically observed through the since-removed
    ``_finish_merge``; the shared logic now lives in
    ``_present_pending_worktree``.)

    Resolved by removing the cause rather than reporting it: an
    empty worktree is now discarded even while a non-wt task runs,
    since that discard touches neither the main working tree's files
    nor its HEAD.  There is no orphaned branch left to tell the user
    about, and no meaningless merge/discard prompt.  See
    ``test_worktree_leak_when_main_tree_busy.py``.

BUG-70: ``_check_merge_conflict`` only checks the unstaged and staged
    files of the main repo (historically via the since-removed
    ``unstaged_files``/``staged_files`` helpers) but not **untracked**
    files.
    When an agent creates a file in the worktree with the same path
    as an untracked file in the main repo, the auto-merge flow will
    fail:

    1. ``stash_if_dirty`` stashes the untracked file
       (``--include-untracked``) — main now has the file gone.
    2. Squash-merge applies the worktree's version of the file.
    3. ``stash_pop`` tries to restore the untracked file but it
       already exists — pop fails with a conflict.

    The user had no warning.  ``_check_merge_conflict`` should report
    the overlap so the user can resolve before merging.

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

from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import VSCodeServer


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
        agent=WorktreeSorcarAgent("Sorcar VS Code"),
        tab_id=tab_id,
        server_owned=True,
    )
    state.use_worktree = True
    agent_state.register(state)
    return state


def _create_wt(
    repo: Path, branch: str, agent: WorktreeSorcarAgent,
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
    agent._wt = wt
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
    concurrent non-wt task must discard the empty worktree WITHOUT
    broadcasting the meaningless ``worktree_done`` prompt.

    The frontend renders ``worktree_done`` as "Auto-commit and merge
    or Discard?", which makes no sense when there are zero changed
    files.  Leaving the branch behind instead is not an option
    either: nothing ever retried the cleanup, so the worktree leaked
    permanently.  Discarding an empty worktree is safe while the main
    tree is busy — it touches neither its files nor its HEAD.
    """

    def test_present_pending_empty_wt_non_wt_busy(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")

        server = VSCodeServer()
        server.work_dir = str(repo)
        printer = _RecordingPrinter()
        server.printer = cast(Any, printer)

        tab_id = "tab-bug68a"
        state = _register_wt_tab("task-bug68a", tab_id)

        agent = cast(WorktreeSorcarAgent, state.agent)
        branch = "kiss/wt-bug68a-1"
        wt = _create_wt(repo, branch, agent)

        other = AgentState("task-other-68a", tab_id="other-bug68a")
        other.is_running_non_wt = True
        agent_state.register(other)

        server._present_pending_worktree(tab_id)

        wt_done = [e for e in printer.events if e.get("type") == "worktree_done"]
        assert not wt_done, (
            "worktree_done must NOT be broadcast for an empty worktree — "
            "the resulting merge/discard prompt is meaningless when "
            f"there are no changes.  Events: {printer.events}"
        )
        assert agent._wt is None, (
            "BUG-68: a busy non-wt task blocked the discard of an empty "
            "worktree, leaking the branch and directory forever."
        )
        assert not GitWorktreeOps.branch_exists(repo, branch), (
            "branch survived the auto-discard."
        )
        assert not wt.wt_dir.exists(), (
            "worktree directory survived the auto-discard."
        )

        other.is_running_non_wt = False

    def test_present_pending_empty_wt_non_wt_idle_discards(
        self, tmp_path: Path,
    ) -> None:
        """Regression: when no non-wt task is running, the empty
        worktree must still be auto-discarded (BUG-42 behavior)."""
        repo = _make_repo(tmp_path / "repo")
        server = VSCodeServer()
        server.work_dir = str(repo)
        printer = _RecordingPrinter()
        server.printer = cast(Any, printer)

        tab_id = "tab-bug68b"
        state = _register_wt_tab("task-bug68b", tab_id)

        agent = cast(WorktreeSorcarAgent, state.agent)
        _create_wt(repo, "kiss/wt-bug68b-1", agent)

        server._present_pending_worktree(tab_id)

        assert agent._wt is None, (
            "Regression: empty worktree was not auto-discarded when "
            "no non-wt task was running."
        )


class TestBug70UntrackedFileConflict:
    """``_check_merge_conflict`` must detect untracked files in the
    main repo that overlap with worktree changes.

    Scenario:
    - Main has an untracked file ``foo.py``.
    - Agent creates ``foo.py`` (different content) in the worktree.
    - Auto-merge's ``stash --include-untracked`` + squash + ``stash
      pop`` fails because the file exists after squash.
    - ``_check_merge_conflict`` should report True so the user is
      warned before clicking merge.
    """

    def test_untracked_main_overlap_reports_conflict(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")

        server = VSCodeServer()
        server.work_dir = str(repo)
        printer = _RecordingPrinter()
        server.printer = cast(Any, printer)

        tab_id = "tab-bug70"
        state = _register_wt_tab("task-bug70", tab_id)

        agent = cast(WorktreeSorcarAgent, state.agent)
        branch = "kiss/wt-bug70-1"
        wt = _create_wt(repo, branch, agent)

        agent_file = wt.wt_dir / "foo.py"
        agent_file.write_text("agent content\n")
        subprocess.run(
            ["git", "-C", str(wt.wt_dir), "add", "foo.py"],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(wt.wt_dir), "commit", "-m", "agent adds foo"],
            capture_output=True, check=True,
        )

        (repo / "foo.py").write_text("user untracked content\n")

        changed = server._get_worktree_changed_files(tab_id)
        assert "foo.py" in changed, (
            f"Precondition failed: worktree change not detected: {changed}"
        )

        has_conflict = server._check_merge_conflict(tab_id)
        assert has_conflict, (
            "BUG-70: _check_merge_conflict returned False when an "
            "untracked file in main overlapped with a worktree change. "
            "The auto-merge will fail at stash-pop with an overwrite "
            "conflict, and the user had no warning."
        )

    def test_non_overlapping_untracked_no_conflict(
        self, tmp_path: Path,
    ) -> None:
        """Regression: untracked file in main that does NOT overlap
        with worktree changes must NOT report conflict."""
        repo = _make_repo(tmp_path / "repo")

        server = VSCodeServer()
        server.work_dir = str(repo)
        printer = _RecordingPrinter()
        server.printer = cast(Any, printer)

        tab_id = "tab-bug70b"
        state = _register_wt_tab("task-bug70b", tab_id)

        agent = cast(WorktreeSorcarAgent, state.agent)
        wt = _create_wt(repo, "kiss/wt-bug70-2", agent)

        (wt.wt_dir / "foo.py").write_text("agent content\n")
        subprocess.run(
            ["git", "-C", str(wt.wt_dir), "add", "foo.py"],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(wt.wt_dir), "commit", "-m", "agent adds foo"],
            capture_output=True, check=True,
        )

        (repo / "bar.py").write_text("unrelated\n")

        assert not server._check_merge_conflict(tab_id), (
            "Regression: non-overlapping untracked file in main "
            "triggered a false-positive conflict."
        )


class TestRed10PostTaskPendingWtDuplication:
    """All three call sites should share a single helper that
    auto-discards or emits worktree_done on empty changes."""

    def test_unified_helper_exists(self) -> None:
        """After the fix, a single helper handles the post-task
        pending-worktree logic.  The helper must exist so future
        changes don't drift between the three sites."""
        assert hasattr(VSCodeServer, "_present_pending_worktree"), (
            "RED-10: the post-task pending-worktree logic is still "
            "duplicated across _run_task_inner and "
            "_emit_pending_worktree.  Expected a single helper "
            "`_present_pending_worktree`."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
