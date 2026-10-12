# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""BUG-6B-2: the original branch must come from the CURRENT repo.

A worktree's ``original_branch`` is the branch ``merge()`` checks out
and squash-merges into.  When the user changes ``work_dir`` to a
DIFFERENT git repo between two tasks of the same agent, the previous
repo's branch name must not leak into the new worktree:

* if the new repo happens to have a branch with that name (e.g.
  ``develop``), ``merge()`` would silently check it out and squash-merge
  the agent's work into the WRONG branch — the user ran the task from
  ``main`` but the changes land on ``develop`` and the repo is left
  switched to ``develop`` (wrong merge result + hijacked HEAD);
* if no such branch exists, ``merge()`` would fail with a confusing
  "Cannot checkout" error even though nothing is wrong with the repo.

Each task isolates itself with the ``create_worktree`` tool and
publishes with ``merge()``; these integration tests use real on-disk
git repos (no mocks).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktreeOps, _git


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


def _start_task_in(agent: ChatSorcarAgent, repo: Path) -> None:
    """Point *agent* at *repo* and isolate the task with ``create_worktree``."""
    agent.work_dir = str(repo)
    message = agent.create_worktree()
    assert agent._wt is not None, message
    assert Path(agent.work_dir).resolve() == agent._wt.wt_dir.resolve(), message


def _commit_agent_work(agent: ChatSorcarAgent, fname: str) -> None:
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
    _git("branch", "develop", cwd=repo_b)

    agent = ChatSorcarAgent("t")

    _start_task_in(agent, repo_a)
    _commit_agent_work(agent, "a.txt")
    msg = agent.merge()
    assert "Successfully merged" in msg, msg
    assert agent._wt is None
    assert Path(agent.work_dir).resolve() == repo_a.resolve()

    _start_task_in(agent, repo_b)
    assert agent._wt is not None
    assert agent._wt.original_branch == "main", (
        f"new worktree recorded original_branch="
        f"{agent._wt.original_branch!r}: repoA's branch name leaked "
        "into repoB"
    )

    _commit_agent_work(agent, "b.txt")
    msg = agent.merge()
    assert "Successfully merged" in msg, msg
    assert Path(agent.work_dir).resolve() == repo_b.resolve()

    assert GitWorktreeOps.current_branch(repo_b) == "main", (
        "merge() switched the user's repoB checkout to the wrong branch"
    )
    assert _git("show", "main:b.txt", cwd=repo_b).returncode == 0, (
        "agent work missing from repoB's main"
    )
    assert _git("show", "develop:b.txt", cwd=repo_b).returncode != 0, (
        "agent work was merged into repoB's unrelated 'develop' branch"
    )

    assert _git("show", "develop:a.txt", cwd=repo_a).returncode == 0


def test_repo_switch_without_matching_branch_still_merges(
    tmp_path: Path,
) -> None:
    """When repoB has no branch named like repoA's original branch,
    merge() must still succeed instead of failing with a bogus
    "Cannot checkout" error."""
    repo_a = _make_repo(tmp_path / "repoA", "develop")
    repo_b = _make_repo(tmp_path / "repoB", "main")

    agent = ChatSorcarAgent("t")
    _start_task_in(agent, repo_a)
    _commit_agent_work(agent, "a.txt")
    assert "Successfully merged" in agent.merge()

    _start_task_in(agent, repo_b)
    _commit_agent_work(agent, "b.txt")

    msg = agent.merge()
    assert "Successfully merged" in msg, msg
    assert _git("show", "main:b.txt", cwd=repo_b).returncode == 0


def test_same_repo_release_keeps_original_branch(tmp_path: Path) -> None:
    """Regression guard: back-to-back tasks in the SAME repo must keep
    using the repo's checked-out branch."""
    repo = _make_repo(tmp_path / "repo", "develop")

    agent = ChatSorcarAgent("t")
    _start_task_in(agent, repo)
    _commit_agent_work(agent, "a.txt")
    assert "Successfully merged" in agent.merge()

    _start_task_in(agent, repo)
    assert agent._wt is not None
    assert agent._wt.original_branch == "develop"
    _commit_agent_work(agent, "b.txt")

    msg = agent.merge()
    assert "Successfully merged" in msg, msg
    assert _git("show", "develop:a.txt", cwd=repo).returncode == 0
    assert _git("show", "develop:b.txt", cwd=repo).returncode == 0


def test_pending_worktree_is_kept_across_a_repo_switch(tmp_path: Path) -> None:
    """``create_worktree`` never silently replaces an unmerged worktree:
    with one pending in repoA, a call from repoB adds a second worktree
    for repoB and leaves repoA's (and its original branch) untouched."""
    repo_a = _make_repo(tmp_path / "repoA", "develop")
    repo_b = _make_repo(tmp_path / "repoB", "main")

    agent = ChatSorcarAgent("t")
    _start_task_in(agent, repo_a)
    _commit_agent_work(agent, "a.txt")
    pending = agent._wt
    assert pending is not None

    agent.work_dir = str(repo_b)
    msg = agent.create_worktree()
    assert "Created worktree" in msg, msg
    assert len(agent.worktrees) == 2 and agent.worktrees[0] is pending
    assert agent.worktrees[1].repo_root.resolve() == repo_b.resolve()
    assert pending.original_branch == "develop"
    assert pending.repo_root.resolve() == repo_a.resolve()
    assert _git("show", "develop:a.txt", cwd=repo_a).returncode != 0, (
        "the pending worktree was merged behind the user's back"
    )
