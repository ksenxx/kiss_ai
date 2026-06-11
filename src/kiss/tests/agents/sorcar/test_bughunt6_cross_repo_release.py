# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""BUG-6B-2: stale ``released_branch`` reused across DIFFERENT repos.

``_try_setup_worktree`` used the branch returned by
``_release_worktree()`` (the ORIGINAL branch of the PREVIOUS pending
worktree) as the new worktree's ``original_branch``.  When the user
changes ``work_dir`` to a DIFFERENT git repo between two runs of the
same agent, the previous repo's branch name leaks into the new repo:

* if the new repo happens to have a branch with that name (e.g.
  ``develop``), ``merge()`` silently checks it out and squash-merges
  the agent's work into the WRONG branch — the user ran the task from
  ``main`` but the changes land on ``develop`` and the repo is left
  switched to ``develop`` (wrong merge result + hijacked HEAD);
* if no such branch exists, ``merge()`` fails with a confusing
  "Cannot checkout" error even though nothing is wrong with the repo.

These integration tests use real on-disk git repos (no mocks).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps, _git
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def _make_repo(path: Path, branch: str) -> Path:
    """Create a real git repo on *branch* with one initial commit."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", branch, str(path)],
        capture_output=True,
        check=True,
    )
    for key, val in (("user.email", "t@t.com"), ("user.name", "T")):
        subprocess.run(
            ["git", "-C", str(path), "config", key, val],
            capture_output=True,
            check=True,
        )
    (path / "f.txt").write_text(f"{branch} base\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "init"],
        capture_output=True,
        check=True,
    )
    return path


def _commit_agent_work(agent: WorktreeSorcarAgent, fname: str) -> None:
    """Write and commit one agent change inside the pending worktree."""
    assert agent._wt is not None
    (agent._wt.wt_dir / fname).write_text("agent work\n")
    _git("add", "-A", cwd=agent._wt.wt_dir)
    result = _git("commit", "-m", f"agent: add {fname}", cwd=agent._wt.wt_dir)
    assert result.returncode == 0, result.stderr


def test_repo_switch_merges_into_current_branch(tmp_path: Path) -> None:
    """Work run from repoB's ``main`` must merge into ``main``, not into
    a branch that merely shares its name with repoA's original branch."""
    repo_a = _make_repo(tmp_path / "repoA", "develop")
    repo_b = _make_repo(tmp_path / "repoB", "main")
    # repoB also has an unrelated stale 'develop' branch; HEAD is 'main'.
    _git("branch", "develop", cwd=repo_b)

    agent = WorktreeSorcarAgent("t")

    # Task 1 runs in repoA (original branch 'develop').
    assert agent._try_setup_worktree(repo_a, None) is not None
    _commit_agent_work(agent, "a.txt")

    # Task 2: the user switched work_dir to repoB (currently on 'main').
    # Setting up the new worktree auto-releases (merges) the repoA one.
    assert agent._try_setup_worktree(repo_b, None) is not None
    assert agent._wt is not None
    assert agent._wt.original_branch == "main", (
        f"new worktree recorded original_branch="
        f"{agent._wt.original_branch!r}: repoA's branch name leaked "
        "into repoB"
    )

    _commit_agent_work(agent, "b.txt")
    msg = agent.merge()
    assert "Successfully merged" in msg, msg

    # The work must land on repoB's 'main' and HEAD must stay on 'main'.
    assert GitWorktreeOps.current_branch(repo_b) == "main", (
        "merge() switched the user's repoB checkout to the wrong branch"
    )
    assert _git("show", "main:b.txt", cwd=repo_b).returncode == 0, (
        "agent work missing from repoB's main"
    )
    assert _git("show", "develop:b.txt", cwd=repo_b).returncode != 0, (
        "agent work was merged into repoB's unrelated 'develop' branch"
    )

    # Task 1's work must still have been merged into repoA's develop.
    assert _git("show", "develop:a.txt", cwd=repo_a).returncode == 0


def test_repo_switch_without_matching_branch_still_merges(
    tmp_path: Path,
) -> None:
    """When repoB has no branch named like repoA's original branch,
    merge() must still succeed instead of failing with a bogus
    "Cannot checkout" error."""
    repo_a = _make_repo(tmp_path / "repoA", "develop")
    repo_b = _make_repo(tmp_path / "repoB", "main")

    agent = WorktreeSorcarAgent("t")
    assert agent._try_setup_worktree(repo_a, None) is not None
    _commit_agent_work(agent, "a.txt")

    assert agent._try_setup_worktree(repo_b, None) is not None
    _commit_agent_work(agent, "b.txt")

    msg = agent.merge()
    assert "Successfully merged" in msg, msg
    assert _git("show", "main:b.txt", cwd=repo_b).returncode == 0


def test_same_repo_release_keeps_original_branch(tmp_path: Path) -> None:
    """Regression guard: back-to-back tasks in the SAME repo must keep
    using the released worktree's original branch."""
    repo = _make_repo(tmp_path / "repo", "develop")

    agent = WorktreeSorcarAgent("t")
    assert agent._try_setup_worktree(repo, None) is not None
    _commit_agent_work(agent, "a.txt")

    assert agent._try_setup_worktree(repo, None) is not None
    assert agent._wt is not None
    assert agent._wt.original_branch == "develop"
    _commit_agent_work(agent, "b.txt")

    msg = agent.merge()
    assert "Successfully merged" in msg, msg
    assert _git("show", "develop:a.txt", cwd=repo).returncode == 0
    assert _git("show", "develop:b.txt", cwd=repo).returncode == 0
