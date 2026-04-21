"""Audit 15: Integration tests for bugs/redundancies found in audit round 15.

BUG-66: ``_emit_pending_worktree`` broadcasts ``worktree_done`` for
    pending worktrees with **no changed files** instead of
    auto-discarding them.  Both ``_run_task_inner``'s finally block
    and ``_finish_merge`` auto-discard empty-change worktrees (guarded
    by ``_any_non_wt_running``), but ``_emit_pending_worktree``
    (called on session resume via ``_replay_session``) does not.
    This means after a server restart, a stale zero-change worktree
    persists and the user is shown merge/discard buttons for a
    worktree that has nothing to merge.

BUG-67: ``_start_merge_session`` sets ``tab.is_merging = True``
    **before** calling ``self.printer.broadcast()``.  If the broadcast
    raises (e.g. ``BrokenPipeError`` when stdout pipe is closed), the
    exception propagates and ``is_merging`` is never cleared.  The tab
    becomes permanently locked:
    - ``_run_task_inner`` refuses new tasks ("merge review in progress")
    - ``_new_chat`` refuses new chats (BUG-65 guard)
    - ``_finish_merge`` is never called because the frontend never
      received the merge data.

RED-9: ``_restore_pending_merge`` is dead code — defined in
    ``VSCodeServer`` but never called by any production module.  Only
    test files reference it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
)
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.server import VSCodeServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo(path: Path) -> Path:
    """Create a minimal git repo with one commit."""
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


class _RecordingPrinter:
    """Concrete printer that records broadcasts and can optionally raise."""

    def __init__(self, *, raise_on: str | None = None) -> None:
        self.events: list[dict[str, Any]] = []
        self._raise_on = raise_on

    def broadcast(self, event: dict[str, Any]) -> None:
        if self._raise_on and event.get("type") == self._raise_on:
            raise BrokenPipeError("simulated stdout failure")
        self.events.append(event)


# ===========================================================================
# BUG-66: _emit_pending_worktree doesn't auto-discard empty worktrees
# ===========================================================================


class TestBug66EmitPendingNoAutoDiscard:
    """``_emit_pending_worktree`` must auto-discard pending worktrees
    with no changed files, consistent with ``_run_task_inner`` and
    ``_finish_merge``.
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
        tab = server._get_tab(tab_id)
        tab.use_worktree = True

        agent = cast(WorktreeSorcarAgent, tab.agent)
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
        # Worktree has no changes → changed files will be empty

        printer = _RecordingPrinter()
        server.printer = cast(Any, printer)
        server._emit_pending_worktree(tab_id)

        # The worktree should have been auto-discarded
        assert agent._wt is None, (
            "BUG-66: _emit_pending_worktree did not auto-discard the "
            "empty-change worktree.  The branch should have been cleaned up."
        )
        # The branch should be deleted
        assert not GitWorktreeOps.branch_exists(repo, branch), (
            "BUG-66: branch still exists after auto-discard."
        )
        # No worktree_done event should have been broadcast
        wt_done = [e for e in printer.events if e.get("type") == "worktree_done"]
        assert not wt_done, (
            "BUG-66: worktree_done was broadcast for an empty worktree "
            f"instead of auto-discarding.  Events: {wt_done}"
        )

    def test_emit_pending_worktree_keeps_changed(
        self, tmp_path: Path,
    ) -> None:
        """Regression: a pending worktree WITH changes must NOT be
        auto-discarded.  Either a merge review starts (merge_started)
        or worktree_done is broadcast — but never auto-discard."""
        repo = _make_repo(tmp_path / "repo")

        server = VSCodeServer()
        server.work_dir = str(repo)
        tab_id = "tab-bug66-changed"
        tab = server._get_tab(tab_id)
        tab.use_worktree = True

        agent = cast(WorktreeSorcarAgent, tab.agent)
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
        # Create a real change in the worktree
        (wt_dir / "new_file.txt").write_text("agent work\n")

        printer = _RecordingPrinter()
        server.printer = cast(Any, printer)
        server._emit_pending_worktree(tab_id)

        # The worktree should NOT be discarded — it has changes
        assert agent._wt is not None, (
            "Regression: pending worktree WITH changes was auto-discarded."
        )
        # Either merge_started or worktree_done should be broadcast
        types = {e.get("type") for e in printer.events}
        assert types & {"merge_started", "worktree_done"}, (
            "Regression: neither merge_started nor worktree_done broadcast "
            f"for changed worktree.  Events: {printer.events}"
        )

        # Cleanup
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.delete_branch(repo, branch)

    def test_emit_pending_no_discard_when_non_wt_running(
        self, tmp_path: Path,
    ) -> None:
        """Auto-discard must be skipped when a non-worktree task is
        running — consistent with ``_run_task_inner`` and
        ``_finish_merge``."""
        repo = _make_repo(tmp_path / "repo")

        server = VSCodeServer()
        server.work_dir = str(repo)

        # Set up a "running" non-wt tab
        other_tab = server._get_tab("other")
        other_tab.is_running_non_wt = True

        tab_id = "tab-bug66-guard"
        tab = server._get_tab(tab_id)
        tab.use_worktree = True

        agent = cast(WorktreeSorcarAgent, tab.agent)
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

        # Should NOT auto-discard because non-wt task is running
        assert agent._wt is not None, (
            "Auto-discard should be skipped when non-wt task is running."
        )
        # Should broadcast worktree_done as fallback
        wt_done = [e for e in printer.events if e.get("type") == "worktree_done"]
        assert wt_done, (
            "worktree_done should be broadcast when non-wt blocks discard."
        )

        # Cleanup
        other_tab.is_running_non_wt = False
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.delete_branch(repo, branch)


# ===========================================================================
# BUG-67: _start_merge_session is_merging stuck on broadcast failure
# ===========================================================================


class TestBug67IsMergingStuckOnBroadcastFailure:
    """``_start_merge_session`` must clear ``is_merging`` if
    ``broadcast()`` raises, so the tab is not permanently locked."""

    def _write_merge_json(self, data_dir: Path) -> str:
        """Write a valid pending-merge.json and return its path."""
        import json

        data_dir.mkdir(parents=True, exist_ok=True)
        merge_json = data_dir / "pending-merge.json"
        merge_json.write_text(
            json.dumps(
                {
                    "branch": "HEAD",
                    "files": [
                        {
                            "name": "a.txt",
                            "base": "/tmp/base",
                            "current": "/tmp/current",
                            "hunks": [{"bs": 0, "bc": 1, "cs": 0, "cc": 2}],
                        }
                    ],
                }
            )
        )
        return str(merge_json)

    def test_is_merging_cleared_on_broadcast_failure(
        self, tmp_path: Path,
    ) -> None:
        """If broadcast() raises after is_merging=True, the flag must
        be cleared so the tab is not permanently locked."""
        server = VSCodeServer()
        tab_id = "tab-bug67"
        tab = server._get_tab(tab_id)

        # Use a printer that raises on merge_data broadcast
        printer = _RecordingPrinter(raise_on="merge_data")
        server.printer = cast(Any, printer)

        merge_json = self._write_merge_json(tmp_path / "merge")
        # _start_merge_session should handle the broadcast failure
        # gracefully — is_merging must not be stuck True
        try:
            server._start_merge_session(merge_json, tab_id=tab_id)
        except BrokenPipeError:
            pass  # It's OK if the exception propagates

        assert not tab.is_merging, (
            "BUG-67: is_merging is stuck True after broadcast failure.  "
            "The tab is permanently locked — user can't run tasks or "
            "start new chats."
        )

    def test_is_merging_set_on_successful_broadcast(
        self, tmp_path: Path,
    ) -> None:
        """Regression: is_merging must be True after a successful
        _start_merge_session call."""
        server = VSCodeServer()
        tab_id = "tab-bug67-success"
        tab = server._get_tab(tab_id)

        printer = _RecordingPrinter()
        server.printer = cast(Any, printer)

        merge_json = self._write_merge_json(tmp_path / "merge2")
        result = server._start_merge_session(merge_json, tab_id=tab_id)

        assert result is True
        assert tab.is_merging, (
            "Regression: is_merging should be True after successful "
            "merge session start."
        )

    def test_merge_started_broadcast_failure(
        self, tmp_path: Path,
    ) -> None:
        """If the second broadcast (merge_started) fails, is_merging
        must still be cleared."""
        server = VSCodeServer()
        tab_id = "tab-bug67-second"
        tab = server._get_tab(tab_id)

        # Raise on the second broadcast (merge_started)
        printer = _RecordingPrinter(raise_on="merge_started")
        server.printer = cast(Any, printer)

        merge_json = self._write_merge_json(tmp_path / "merge3")
        try:
            server._start_merge_session(merge_json, tab_id=tab_id)
        except BrokenPipeError:
            pass

        assert not tab.is_merging, (
            "BUG-67: is_merging stuck True when merge_started "
            "broadcast fails."
        )


# ===========================================================================
# RED-9: _restore_pending_merge is dead code
# ===========================================================================


class TestRed9RestorePendingMergeDeadCode:
    """``_restore_pending_merge`` is not called by any production module."""

    def test_no_production_callers(self) -> None:
        """Verify no production code calls _restore_pending_merge."""
        import re

        src_root = Path(__file__).resolve().parents[4] / "agents"
        offenders: list[str] = []
        for py in src_root.rglob("*.py"):
            # Skip test files and the defining file's definition line
            if "test" in py.name.lower():
                continue
            text = py.read_text()
            # Find calls (not the definition itself)
            for match in re.finditer(r"\b_restore_pending_merge\b", text):
                # Check it's not the def line
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
