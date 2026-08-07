# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 15: Integration tests for bugs/redundancies found in audit round 15.

BUG-66: ``_emit_pending_worktree`` broadcasts ``worktree_done`` for
    pending worktrees with **no changed files** instead of
    auto-discarding them.  ``_run_task_inner``'s finally block
    auto-discards empty-change worktrees, but
    ``_emit_pending_worktree`` (called on session resume via
    ``_replay_session``) does not.  This means after a server
    restart, a stale zero-change worktree persists and the user is
    shown merge/discard buttons for a worktree that has nothing to
    merge.

    The auto-discard used to be suppressed while a non-worktree task
    ran on the main tree, which leaked the worktree permanently.  It
    now runs unconditionally — see
    ``test_worktree_leak_when_main_tree_busy.py``.

BUG-67 (obsolete): covered ``_start_merge_session``, which was removed
    together with the interactive diff/merge review workflow.

RED-9: ``_restore_pending_merge`` is dead code — defined in
    ``VSCodeServer`` but never called by any production module.  Only
    test files reference it.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
)
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import VSCodeServer


def _register_state(
    task_id: str,
    tab_id: str,
    *,
    agent: WorktreeSorcarAgent | None = None,
    use_worktree: bool = False,
    auto_commit_mode: bool = True,
) -> agent_state.AgentState:
    """Register a server-owned AgentState for *tab_id*."""
    state = agent_state.AgentState(
        task_id, agent=agent, tab_id=tab_id, server_owned=True,
    )
    state.use_worktree = use_worktree
    state.auto_commit_mode = auto_commit_mode
    agent_state.register(state)
    return state


def _make_repo(path: Path) -> Path:
    """Create a minimal git repo with one commit."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True,
        check=True,
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
        ["git", "-C", str(path), "add", "."],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "init"],
        capture_output=True,
        check=True,
    )
    return path


@pytest.fixture(autouse=True)
def _clean_agent_registry() -> Iterator[None]:
    """Keep the global agent-state registry isolated per test."""
    agent_state.agent_states.clear()
    yield
    agent_state.agent_states.clear()


class _RecordingPrinter(JsonPrinter):
    """Concrete printer that records broadcasts and can optionally raise."""

    def __init__(self, *, raise_on: str | None = None) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self._raise_on = raise_on

    def broadcast(self, event: dict[str, Any]) -> None:
        if self._raise_on and event.get("type") == self._raise_on:
            raise BrokenPipeError("simulated stdout failure")
        self.events.append(event)


class TestBug66EmitPendingNoAutoDiscard:
    """``_emit_pending_worktree`` must auto-discard pending worktrees
    with no changed files, consistent with ``_run_task_inner``'s
    post-task cleanup.
    """

    def test_emit_pending_worktree_auto_discards_empty(
        self, tmp_path: Path,
    ) -> None:
        """A pending worktree with zero changed files should be
        auto-discarded on session resume — not shown to the user
        with merge/discard buttons for nothing."""
        repo = _make_repo(tmp_path / "repo")

        server = VSCodeServer()
        server.work_dir = str(repo)
        tab_id = "tab-bug66"
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        _register_state(
            "task-bug66", tab_id, agent=agent, use_worktree=True,
        )
        branch = "kiss/wt-bug66-1"
        wt_dir = repo / ".kiss-worktrees" / "kiss_wt-bug66-1"
        assert GitWorktreeOps.create(repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(repo, branch, "main")
        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
        )

        printer = _RecordingPrinter()
        server.printer = cast(Any, printer)
        server._emit_pending_worktree(tab_id)

        assert agent._wt is None, (
            "BUG-66: _emit_pending_worktree did not auto-discard the "
            "empty-change worktree.  The branch should have been cleaned up."
        )
        assert not GitWorktreeOps.branch_exists(repo, branch), (
            "BUG-66: branch still exists after auto-discard."
        )
        wt_done = [e for e in printer.events if e.get("type") == "worktree_done"]
        assert not wt_done, (
            "BUG-66: worktree_done was broadcast for an empty worktree "
            f"instead of auto-discarding.  Events: {wt_done}"
        )

    def test_emit_pending_worktree_keeps_changed(
        self, tmp_path: Path,
    ) -> None:
        """Regression: a pending worktree WITH changes must NOT be
        auto-discarded.  ``worktree_done`` is broadcast so the user
        gets the Merge / Discard buttons — but never auto-discard.

        Auto-commit is switched off because that is the mode in which
        the user is asked what to do with the branch.  With auto-commit
        on the branch is merged without a prompt instead — see
        :meth:`test_emit_pending_worktree_merges_changed_when_autocommit`.
        """
        repo = _make_repo(tmp_path / "repo")

        server = VSCodeServer()
        server.work_dir = str(repo)
        tab_id = "tab-bug66-changed"
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        _register_state(
            "task-bug66-changed", tab_id, agent=agent,
            use_worktree=True, auto_commit_mode=False,
        )
        branch = "kiss/wt-bug66-2"
        wt_dir = repo / ".kiss-worktrees" / "kiss_wt-bug66-2"
        assert GitWorktreeOps.create(repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(repo, branch, "main")
        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
        )
        (wt_dir / "new_file.txt").write_text("agent work\n")

        printer = _RecordingPrinter()
        server.printer = cast(Any, printer)
        server._emit_pending_worktree(tab_id)

        assert agent._wt is not None, (
            "Regression: pending worktree WITH changes was auto-discarded."
        )
        types = {e.get("type") for e in printer.events}
        assert "worktree_done" in types, (
            "Regression: worktree_done not broadcast for changed "
            f"worktree.  Events: {printer.events}"
        )

        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.delete_branch(repo, branch)

    def test_emit_pending_worktree_merges_changed_when_autocommit(
        self, tmp_path: Path,
    ) -> None:
        """With auto-commit ON the branch is merged, never reviewed.

        Auto-commit means "do not interrupt me".  A post-task merge
        that could not complete leaves the branch pending, and the next
        history click reaches this path; showing the hunk-by-hunk
        diff/merge UI there contradicts the toggle the user set.  The
        branch must instead be merged into the original branch and the
        outcome reported through ``worktree_result``.
        """
        repo = _make_repo(tmp_path / "repo")

        server = VSCodeServer()
        server.work_dir = str(repo)
        tab_id = "tab-bug66-autocommit"
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        state = _register_state(
            "task-bug66-autocommit", tab_id, agent=agent,
            use_worktree=True, auto_commit_mode=True,
        )
        branch = "kiss/wt-bug66-4"
        wt_dir = repo / ".kiss-worktrees" / "kiss_wt-bug66-4"
        assert GitWorktreeOps.create(repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(repo, branch, "main")
        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
        )
        (wt_dir / "new_file.txt").write_text("agent work\n")

        printer = _RecordingPrinter()
        server.printer = cast(Any, printer)
        server._emit_pending_worktree(tab_id)

        types = {e.get("type") for e in printer.events}
        assert "merge_started" not in types, (
            "auto-commit was on, yet the diff/merge UI was opened.  "
            f"Events: {printer.events}"
        )
        assert "merge_data" not in types, (
            "auto-commit was on, yet merge data was pushed to the "
            f"frontend.  Events: {printer.events}"
        )
        assert not state.is_merging, "the tab was left in merge-review mode."
        assert agent._wt is None, (
            "the pending worktree survived the silent merge."
        )
        assert (repo / "new_file.txt").exists(), (
            "the branch's work was not merged into the main tree."
        )
        results = [e for e in printer.events if e.get("type") == "worktree_result"]
        assert results, f"no worktree_result reported.  Events: {printer.events}"

    def test_emit_pending_discards_empty_even_when_non_wt_running(
        self, tmp_path: Path,
    ) -> None:
        """Auto-discard of an EMPTY worktree must still happen while a
        non-worktree task runs on the main tree.

        The main-tree guard exists to protect a *merge*, which
        stashes, checks out and merges the working tree the other
        task is writing.  Discarding an empty worktree only removes
        ``.kiss-worktrees/<slug>`` and deletes its unmerged branch, so
        it touches neither the main tree's files nor its HEAD.
        Skipping it leaked the branch, the directory and the
        ``branch.kiss/*`` config section forever, because nothing ever
        retried the cleanup.

        No ``worktree_done`` event may be broadcast either: there are
        no changes to merge, so the "Auto-commit and merge or
        Discard?" prompt would be meaningless and confusing.
        """
        repo = _make_repo(tmp_path / "repo")

        server = VSCodeServer()
        server.work_dir = str(repo)

        other_state = _register_state("task-other", "other")
        other_state.is_running_non_wt = True

        tab_id = "tab-bug66-guard"
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        _register_state(
            "task-bug66-guard", tab_id, agent=agent, use_worktree=True,
        )
        branch = "kiss/wt-bug66-3"
        wt_dir = repo / ".kiss-worktrees" / "kiss_wt-bug66-3"
        assert GitWorktreeOps.create(repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(repo, branch, "main")
        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
        )

        printer = _RecordingPrinter()
        server.printer = cast(Any, printer)
        server._emit_pending_worktree(tab_id)

        assert agent._wt is None, (
            "A busy main tree must not block the discard of an empty "
            "worktree — the branch and directory would leak forever."
        )
        assert not GitWorktreeOps.branch_exists(repo, branch), (
            "branch survived the auto-discard."
        )
        assert not wt_dir.exists(), "worktree directory survived the auto-discard."
        wt_done = [e for e in printer.events if e.get("type") == "worktree_done"]
        assert not wt_done, (
            "worktree_done must NOT be broadcast for an empty worktree — "
            "the resulting merge/discard prompt is meaningless with no "
            "changes."
        )

        other_state.is_running_non_wt = False


class TestRed9RestorePendingMergeDeadCode:
    """``_restore_pending_merge`` is not called by any production module."""

    def test_no_production_callers(self) -> None:
        """Verify no production code calls _restore_pending_merge."""
        import re

        src_root = Path(__file__).resolve().parents[4] / "agents"
        offenders: list[str] = []
        for py in src_root.rglob("*.py"):
            if "test" in py.name.lower():
                continue
            text = py.read_text()
            for match in re.finditer(r"\b_restore_pending_merge\b", text):
                line_start = text.rfind("\n", 0, match.start()) + 1
                line = text[line_start : text.find("\n", match.end())]
                if "def _restore_pending_merge" in line:
                    continue
                offenders.append(f"{py}:{line.strip()}")

        assert not offenders, (
            "RED-9 broken: found production callers of "
            f"_restore_pending_merge: {offenders}"
        )

    def test_restore_pending_merge_removed(self) -> None:
        """The method should be removed as dead code."""
        assert not hasattr(VSCodeServer, "_restore_pending_merge"), (
            "RED-9: _restore_pending_merge is dead code — no production "
            "caller.  Remove it."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
