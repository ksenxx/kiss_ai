# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A task's worktree setup must not sweep the branch of a merge in flight in its process.

``_merge_one`` removes the task's worktree directory
(``_finalize_worktree``) before it merges the branch (``_do_merge``).
For that moment no git worktree lists the branch, and a task that
changed nothing leaves a branch with no commit of its own, which
``sweep_orphaned_state`` deletes unless the branch is excluded.  A merge
in another process is protected by its owner stamp, a merge in this
process only by the live-branch set.  Worktree setup passed that set to
the orphan reclaim but not to the sweep, so a sibling tab starting a
task deleted the branch under the merge.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktreeOps, _WorktreeCleanupOutcome
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter


def _make_repo(path: Path) -> Path:
    """Create a git repository at *path* with one commit on ``main``."""
    path.mkdir()
    for args in (
        ["init", "-b", "main"],
        ["config", "user.email", "test@test.com"],
        ["config", "user.name", "Test"],
    ):
        subprocess.run(["git", "-C", str(path), *args], capture_output=True, check=True)
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "-C", str(path), "add", "."], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"], capture_output=True, check=True
    )
    return path


def test_sibling_setup_keeps_the_branch_of_a_merge_in_flight(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    merging = ChatSorcarAgent("merging")
    assert merging._try_setup_worktree(repo, str(repo)) is not None
    wt = merging.worktrees[0]
    state = agent_state.AgentState(
        "sweep-live-merge", agent=merging, server_owned=True, is_task_active=True,
    )
    agent_state.register(state)
    starting = ChatSorcarAgent("starting")
    starting.printer = JsonPrinter()
    try:
        # The merging agent's first step: commit (nothing) and remove the
        # worktree directory; its branch now has no worktree and no commit.
        assert merging._finalize_worktree(wt) is _WorktreeCleanupOutcome.COMMITTED_AND_REMOVED
        assert not wt.wt_dir.exists()

        assert starting._try_setup_worktree(repo, str(repo)) is not None

        assert GitWorktreeOps.branch_exists(repo, wt.branch)
        assert merging.merge().startswith(f"Successfully merged branch '{wt.branch}'.")
    finally:
        agent_state.unregister("sweep-live-merge", state)
        for agent in (starting, merging):
            if agent.worktrees:
                agent.discard()
