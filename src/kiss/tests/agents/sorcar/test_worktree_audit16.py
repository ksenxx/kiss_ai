"""Audit 16: Integration tests for bugs/inconsistencies found in audit 16.

BUG-68: ``_finish_merge`` and ``_run_task_inner``'s post-task cleanup
    silently leave a pending empty-change worktree when
    ``_any_non_wt_running()`` is True — the user sees no buttons and
    has no indication that a worktree exists.  This is inconsistent
    with ``_emit_pending_worktree`` (which broadcasts
    ``worktree_done`` as a fallback — BUG-66 fix).

    All three call sites must present a consistent UX: when
    auto-discard is blocked by a concurrent non-wt task, broadcast
    ``worktree_done`` so the user knows the branch is pending and
    can take manual action.

BUG-70: ``_check_merge_conflict`` only checks ``unstaged_files`` and
    ``staged_files`` of the main repo but not **untracked** files.
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

RED-10: The three post-task pending-worktree handling blocks in
    ``_run_task_inner``, ``_finish_merge``, and
    ``_emit_pending_worktree`` duplicate the same "auto-discard or
    emit worktree_done" logic with subtle divergences.  A single
    helper would eliminate redundancy and prevent future drift.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.server import VSCodeServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo(path: Path) -> Path:
    """Create a minimal git repo with one initial commit."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
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


# ===========================================================================
# BUG-68: post-task cleanup silent when non-wt blocks auto-discard
# ===========================================================================


class TestBug68FinishMergeNoBroadcastOnEmptyNonWtBusy:
    """``_finish_merge`` with no worktree changes and a concurrent
    non-wt task must broadcast ``worktree_done`` so the user is
    notified — consistent with ``_emit_pending_worktree``."""

    def test_finish_merge_empty_wt_non_wt_busy(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")

        server = VSCodeServer()
        server.work_dir = str(repo)
        printer = _RecordingPrinter()
        server.printer = cast(Any, printer)

        tab_id = "tab-bug68a"
        tab = server._get_tab(tab_id)
        tab.use_worktree = True
        tab.is_merging = True  # in merge review

        agent = cast(WorktreeSorcarAgent, tab.agent)
        _create_wt(repo, "kiss/wt-bug68a-1", agent)
        # Worktree has no agent-changes — merge review was on baseline only

        # A non-wt task is running on another tab
        other = server._get_tab("other-bug68a")
        other.is_running_non_wt = True

        # Simulate the frontend sending all-done
        server._finish_merge(tab_id)

        # BUG-68: currently `_finish_merge` silently leaves the tab
        # with no broadcast when auto-discard is blocked.  The user
        # has no merge/discard UI for a pending branch.
        wt_done = [e for e in printer.events if e.get("type") == "worktree_done"]
        assert wt_done, (
            "BUG-68: _finish_merge did not broadcast worktree_done "
            "when auto-discard was blocked by a non-wt task.  The "
            f"user is left unaware of the pending branch.  Events: {printer.events}"
        )
        # Worktree reference must be preserved (NOT discarded) since
        # non-wt is busy — the user has to act later.
        assert agent._wt is not None, (
            "BUG-68: worktree was discarded despite non-wt being busy."
        )

        other.is_running_non_wt = False

    def test_finish_merge_empty_wt_non_wt_idle_discards(
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
        tab = server._get_tab(tab_id)
        tab.use_worktree = True
        tab.is_merging = True

        agent = cast(WorktreeSorcarAgent, tab.agent)
        _create_wt(repo, "kiss/wt-bug68b-1", agent)

        # No concurrent non-wt task → auto-discard should happen
        server._finish_merge(tab_id)

        assert agent._wt is None, (
            "Regression: empty worktree was not auto-discarded when "
            "no non-wt task was running."
        )


# ===========================================================================
# BUG-70: _check_merge_conflict misses untracked files in main
# ===========================================================================


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
        tab = server._get_tab(tab_id)
        tab.use_worktree = True

        agent = cast(WorktreeSorcarAgent, tab.agent)
        branch = "kiss/wt-bug70-1"
        wt = _create_wt(repo, branch, agent)

        # Agent creates a new file in the worktree.
        agent_file = wt.wt_dir / "foo.py"
        agent_file.write_text("agent content\n")
        # Commit so it's tracked on the branch and shows in diff.
        subprocess.run(
            ["git", "-C", str(wt.wt_dir), "add", "foo.py"],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(wt.wt_dir), "commit", "-m", "agent adds foo"],
            capture_output=True, check=True,
        )

        # Main repo has an untracked file at the same path with
        # different content.
        (repo / "foo.py").write_text("user untracked content\n")

        # Sanity: _get_worktree_changed_files sees foo.py on branch.
        changed = server._get_worktree_changed_files(tab_id)
        assert "foo.py" in changed, (
            f"Precondition failed: worktree change not detected: {changed}"
        )

        # BUG-70: _check_merge_conflict currently only checks
        # unstaged + staged files in main; it ignores untracked,
        # so the conflict is missed.
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
        tab = server._get_tab(tab_id)
        tab.use_worktree = True

        agent = cast(WorktreeSorcarAgent, tab.agent)
        wt = _create_wt(repo, "kiss/wt-bug70-2", agent)

        # Agent creates foo.py in worktree.
        (wt.wt_dir / "foo.py").write_text("agent content\n")
        subprocess.run(
            ["git", "-C", str(wt.wt_dir), "add", "foo.py"],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(wt.wt_dir), "commit", "-m", "agent adds foo"],
            capture_output=True, check=True,
        )

        # Main has an unrelated untracked file
        (repo / "bar.py").write_text("unrelated\n")

        assert not server._check_merge_conflict(tab_id), (
            "Regression: non-overlapping untracked file in main "
            "triggered a false-positive conflict."
        )


# ===========================================================================
# RED-10: post-task pending-worktree handling is duplicated
# ===========================================================================


class TestRed10PostTaskPendingWtDuplication:
    """All three call sites should share a single helper that
    auto-discards or emits worktree_done on empty changes."""

    def test_unified_helper_exists(self) -> None:
        """After the fix, a single helper handles the post-task
        pending-worktree logic.  The helper must exist so future
        changes don't drift between the three sites."""
        # The fix introduces `_present_pending_worktree` (or similar)
        # as a single source of truth.
        assert hasattr(VSCodeServer, "_present_pending_worktree"), (
            "RED-10: the post-task pending-worktree logic is still "
            "duplicated across _run_task_inner, _finish_merge, and "
            "_emit_pending_worktree.  Expected a single helper "
            "`_present_pending_worktree`."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
