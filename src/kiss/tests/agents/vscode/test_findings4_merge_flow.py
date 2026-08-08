# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""FINDINGS-4 regression tests for merge_flow.py.

Covers (real :class:`VSCodeServer`, real :class:`WorktreeSorcarAgent`,
real on-disk git repos and worktrees — no mocks):

- F4-19: ``_handle_worktree_action(..., internal=True)`` must still
  refuse while a non-worktree task is running on the main tree.
- F4-20: a session replay (``_emit_pending_worktree``) during an
  active merge review must not regenerate the review.
- F4-21: when the recorded original branch no longer resolves, a
  worktree holding COMMITTED work must not be reported as "no
  changes" (which callers auto-discard).
- F4-22: committed agent changes in a clean worktree must reach the
  hunk review (fork-point base, not ``HEAD``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server.server import VSCodeServer

from ._memory_printer import MemoryPrinter


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=False,
    )


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True, check=True,
    )
    _run_git(path, "config", "user.email", "t@t.com")
    _run_git(path, "config", "user.name", "T")
    (path / "README.md").write_text("# Test\n")
    _run_git(path, "add", ".")
    _run_git(path, "commit", "-m", "initial")
    return path


def _setup_wt_tab(
    server: VSCodeServer, repo: Path, tab_id: str,
) -> tuple[Any, WorktreeSorcarAgent, Path]:
    """Create a worktree agent and wire it into a server tab."""
    wt_agent = WorktreeSorcarAgent("wt")
    wt_agent._chat_id = tab_id
    wt_work = wt_agent._try_setup_worktree(repo, str(repo))
    assert wt_work is not None
    tab = server._get_tab(tab_id)
    tab.agent = wt_agent
    tab.use_worktree = True
    return tab, wt_agent, Path(wt_work)


def _commit_in_worktree(wt_dir: Path, fname: str, content: str) -> None:
    (wt_dir / fname).write_text(content)
    add = _run_git(wt_dir, "add", ".")
    assert add.returncode == 0, add.stderr
    commit = _run_git(wt_dir, "commit", "-m", f"agent: add {fname}")
    assert commit.returncode == 0, commit.stderr
    status = _run_git(wt_dir, "status", "--porcelain")
    assert status.stdout.strip() == "", (
        f"worktree not clean after commit: {status.stdout!r}"
    )


class TestF419InternalStillGuardsMainTree:
    """internal=True must not bypass the live-main-tree-task guard."""

    def test_internal_merge_refused_when_non_wt_running(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        server = VSCodeServer(printer=MemoryPrinter())
        server.work_dir = str(repo)

        tab, wt_agent, wt_dir = _setup_wt_tab(server, repo, "wt-419")
        _commit_in_worktree(wt_dir, "work.txt", "agent work\n")

        non_wt_tab = server._get_tab("direct-419")
        non_wt_tab.is_running_non_wt = True
        # Mirror _run_task_inner: a running non-wt task records the
        # resolved main-repo root of its work_dir on its tab.
        non_wt_tab.non_wt_repo_root = repo.resolve()
        try:
            result = server._handle_worktree_action(
                "merge", "wt-419", internal=True,
            )
            assert result["success"] is False, (
                "internal=True bypassed the main-tree guard and "
                "merged while a direct task was running"
            )
            assert "main working tree" in result["message"]
            assert not (repo / "work.txt").exists(), (
                "the worktree branch was merged into the main tree "
                "despite the live direct task"
            )
        finally:
            non_wt_tab.is_running_non_wt = False
            non_wt_tab.non_wt_repo_root = None
            wt_agent.discard()


class TestF420ReplayDoesNotResetActiveReview:
    """Session replay must not regenerate an in-flight merge review."""

    def test_emit_pending_worktree_noops_while_merging(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        printer = MemoryPrinter()
        server = VSCodeServer(printer=printer)
        server.work_dir = str(repo)

        tab, wt_agent, wt_dir = _setup_wt_tab(server, repo, "wt-420")
        _commit_in_worktree(wt_dir, "work.txt", "agent work\n")
        try:
            tab.is_merging = True
            printer.emitted.clear()

            server._emit_pending_worktree("wt-420")

            replayed_types = [e.get("type") for e in printer.emitted]
            assert "merge_data" not in replayed_types, (
                "replay during an active merge review regenerated the "
                "review, erasing accepted/rejected hunk resolutions"
            )
            assert "worktree_done" not in replayed_types
        finally:
            tab.is_merging = False
            wt_agent.discard()


class TestF421GitFailureNotMistakenForClean:
    """A failed diff query must not report committed work as clean."""

    def test_committed_work_listed_when_original_branch_gone(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        server = VSCodeServer(printer=MemoryPrinter())
        server.work_dir = str(repo)

        tab, wt_agent, wt_dir = _setup_wt_tab(server, repo, "wt-421")
        _commit_in_worktree(wt_dir, "committed.txt", "agent work\n")

        # The user renames the original branch while the task runs;
        # the recorded original branch name no longer resolves.
        rename = _run_git(repo, "branch", "-m", "main", "renamed")
        assert rename.returncode == 0, rename.stderr
        try:
            changed = server._get_worktree_changed_files("wt-421")
            assert "committed.txt" in changed, (
                "a worktree with COMMITTED task work was reported as "
                f"clean ({changed!r}); automatic finalization would "
                "discard the branch and delete the work"
            )
        finally:
            _run_git(repo, "branch", "-m", "renamed", "main")
            wt_agent.discard()


class TestF422CommittedChangesReachHunkReview:
    """The review base must be the fork point, not worktree HEAD."""

    def test_present_pending_worktree_reviews_committed_changes(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        printer = MemoryPrinter()
        server = VSCodeServer(printer=printer)
        server.work_dir = str(repo)

        tab, wt_agent, wt_dir = _setup_wt_tab(server, repo, "wt-422")
        # The agent COMMITS its work: the worktree status is clean,
        # so a "HEAD" diff base sees nothing.
        _commit_in_worktree(wt_dir, "committed.txt", "agent work\n")
        try:
            printer.emitted.clear()
            server._present_pending_worktree("wt-422", try_merge_review=True)

            merge_events = [
                e for e in printer.emitted if e.get("type") == "merge_data"
            ]
            assert merge_events, (
                "no hunk review was started for committed worktree "
                "changes; the flow fell back to the coarse "
                f"merge/discard buttons (events: "
                f"{[e.get('type') for e in printer.emitted]})"
            )
            reviewed = {
                f.get("name")
                for f in merge_events[0]["data"].get("files", [])
            }
            assert "committed.txt" in reviewed
        finally:
            with server._state_lock:
                tab.is_merging = False
            wt_agent.discard()
