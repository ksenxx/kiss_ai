# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""FINDINGS-4 regression tests for merge_flow.py.

Covers (real :class:`VSCodeServer`, real :class:`ChatSorcarAgent`,
real on-disk git repos and worktrees — no mocks):

- F4-20: a session replay (``_emit_pending_worktree``) while a merge
  or discard owns the tab must not re-present the worktree.
- F4-21: when the recorded original branch no longer resolves, a
  worktree holding COMMITTED work must not be reported as "no
  changes" (which callers auto-discard).
- F4-22: committed agent changes in a clean worktree must reach the
  ``worktree_done`` prompt (fork-point base, not ``HEAD``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.tests.server.conftest import run_git

from ._memory_printer import MemoryPrinter


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True, check=True,
    )
    run_git(path, "config", "user.email", "t@t.com")
    run_git(path, "config", "user.name", "T")
    (path / "README.md").write_text("# Test\n")
    run_git(path, "add", ".")
    run_git(path, "commit", "-m", "initial")
    return path


def _setup_wt_tab(
    server: VSCodeServer, repo: Path, tab_id: str,
) -> tuple[Any, ChatSorcarAgent, Path]:
    """Create a worktree agent and register its tab's agent state."""
    wt_agent = ChatSorcarAgent("wt")
    wt_agent._chat_id = tab_id
    wt_work = wt_agent._try_setup_worktree(repo, str(repo))
    assert wt_work is not None
    state = agent_state.AgentState(
        f"{tab_id}-key",
        agent=wt_agent,
        tab_id=tab_id,
        server_owned=True,
    )
    agent_state.register(state)
    return state, wt_agent, Path(wt_work)


def _commit_in_worktree(wt_dir: Path, fname: str, content: str) -> None:
    (wt_dir / fname).write_text(content)
    add = run_git(wt_dir, "add", ".")
    assert add.returncode == 0, add.stderr
    commit = run_git(wt_dir, "commit", "-m", f"agent: add {fname}")
    assert commit.returncode == 0, commit.stderr
    status = run_git(wt_dir, "status", "--porcelain")
    assert status.stdout.strip() == "", (
        f"worktree not clean after commit: {status.stdout!r}"
    )


class TestF420ReplayDoesNotResetActiveReview:
    """Session replay must no-op while a merge/discard owns the tab."""

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
            agent_state.agent_states.clear()


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
        rename = run_git(repo, "branch", "-m", "main", "renamed")
        assert rename.returncode == 0, rename.stderr
        try:
            changed = server._get_worktree_changed_files("wt-421")
            assert "committed.txt" in changed, (
                "a worktree with COMMITTED task work was reported as "
                f"clean ({changed!r}); automatic finalization would "
                "discard the branch and delete the work"
            )
        finally:
            run_git(repo, "branch", "-m", "renamed", "main")
            wt_agent.discard()
            agent_state.agent_states.clear()


class TestF422CommittedChangesReachWorktreePrompt:
    """The changed-files base must be the fork point, not worktree HEAD."""

    def test_present_pending_worktree_lists_committed_changes(
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
            server._present_pending_worktree("wt-422")

            done_events = [
                e for e in printer.emitted
                if e.get("type") == "worktree_done"
            ]
            assert done_events, (
                "no worktree_done prompt was broadcast for committed "
                "worktree changes; an empty changed-files answer would "
                "let callers auto-discard the branch (events: "
                f"{[e.get('type') for e in printer.emitted]})"
            )
            assert "committed.txt" in done_events[0]["changedFiles"]
            assert wt_agent._wt_pending, (
                "presenting must not finalize or discard the branch"
            )
        finally:
            with server._state_lock:
                tab.is_merging = False
            wt_agent.discard()
            agent_state.agent_states.clear()
