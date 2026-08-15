# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 11: Integration tests for bugs in worktree and non-worktree workflows.

BUG-56: ``_check_merge_conflict`` uses ``baseline^`` / ``baseline``
    without validating the baseline SHA exists (unlike ``_resolve_base_ref``
    which has the BUG-51 ``git cat-file -t`` check).  An invalid baseline
    makes both ``git diff`` commands fail silently (empty file sets),
    causing the method to return ``False`` even when a real conflict
    exists.

(BUG-55 and BUG-57 covered the pre-task snapshot and the interactive
merge-review view, both removed together with the diff/merge review
workflow; ``_check_merge_conflict`` is kept and still guards the direct
worktree merge action.)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


def _register_wt_state(
    tab_id: str, agent: WorktreeSorcarAgent,
) -> agent_state.AgentState:
    """Register a worktree-task agent state for *tab_id*."""
    state = agent_state.AgentState(
        f"task-{tab_id}",
        agent=agent,
        tab_id=tab_id,
        server_owned=True,
    )
    state.use_worktree = True
    agent_state.register(state)
    return state


def _make_repo(tmp_path: Path, name: str = "repo") -> Path:
    """Create a bare-minimum git repo with one commit."""
    repo = tmp_path / name
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, capture_output=True,
    )
    (repo / "init.txt").write_text("init\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo, capture_output=True,
    )
    subprocess.run(
        ["git", "branch", "-M", "main"],
        cwd=repo, capture_output=True,
    )
    return repo


def _create_worktree_branch(
    repo: Path, branch: str, dirty_file: str | None = None,
) -> tuple[Path, str | None]:
    """Create a worktree branch with optional baseline from dirty state."""
    wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    GitWorktreeOps.save_original_branch(repo, branch, "main")

    baseline: str | None = None
    if dirty_file:
        (wt_dir / dirty_file).write_text("dirty content\n")
        GitWorktreeOps.stage_all(wt_dir)
        GitWorktreeOps.commit_staged(
            wt_dir, "kiss: baseline", no_verify=True,
        )
        baseline = GitWorktreeOps.head_sha(wt_dir)
        if baseline:
            GitWorktreeOps.save_baseline_commit(repo, branch, baseline)
    return wt_dir, baseline


def _add_agent_commit(wt_dir: Path, fname: str, content: str) -> None:
    """Add a commit in the worktree simulating agent work."""
    (wt_dir / fname).write_text(content)
    GitWorktreeOps.stage_all(wt_dir)
    GitWorktreeOps.commit_staged(wt_dir, f"agent: edit {fname}")


def _cleanup(repo: Path, branch: str, wt_dir: Path) -> None:
    """Best-effort cleanup."""
    if wt_dir.exists():
        GitWorktreeOps.remove(repo, wt_dir)
    GitWorktreeOps.prune(repo)
    if GitWorktreeOps.branch_exists(repo, branch):
        GitWorktreeOps.delete_branch(repo, branch)




class TestBug56ConflictCheckBaselineValidation:
    """BUG-56: ``_check_merge_conflict`` uses ``baseline^`` and
    ``baseline`` directly without validating the SHA exists.  An invalid
    baseline causes both ``git diff`` commands to fail silently, making
    the method return ``False`` (no conflict) even when there IS one.

    ``_resolve_base_ref`` validates with ``git cat-file -t`` (BUG-51 fix)
    but ``_check_merge_conflict`` doesn't.

    FIX: Validate baseline with ``git cat-file -t`` in
    ``_check_merge_conflict`` before using it; fall back to merge-base
    when invalid.
    """

    def test_invalid_baseline_still_detects_conflict(
        self, tmp_path: Path,
    ) -> None:
        """With an invalid baseline, conflict detection must still work."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug56a-1"
        wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
        assert GitWorktreeOps.create(repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(repo, branch, "main")

        (wt_dir / "init.txt").write_text("agent version\n")
        GitWorktreeOps.stage_all(wt_dir)
        GitWorktreeOps.commit_staged(wt_dir, "agent edit")

        (repo / "init.txt").write_text("user version\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "user edit"],
            cwd=repo, capture_output=True,
        )

        bogus = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        GitWorktreeOps.save_baseline_commit(repo, branch, bogus)

        server = VSCodeServer()
        server.work_dir = str(repo)
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=bogus,
        )
        state = _register_wt_state("bug56a-tab", agent)

        try:
            has_conflict = server._check_merge_conflict("bug56a-tab")
            assert has_conflict is True, (
                "BUG-56: _check_merge_conflict returned False with invalid "
                "baseline despite a real conflict — both sides edited init.txt"
            )
        finally:
            agent_state.unregister(state.task_id, state)

        _cleanup(repo, branch, wt_dir)

    def test_valid_baseline_still_detects_conflict(
        self, tmp_path: Path,
    ) -> None:
        """Regression: valid baseline must still detect conflicts."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug56b-1"
        wt_dir, baseline = _create_worktree_branch(
            repo, branch, dirty_file="dirty.txt",
        )

        (wt_dir / "init.txt").write_text("agent version\n")
        GitWorktreeOps.stage_all(wt_dir)
        GitWorktreeOps.commit_staged(wt_dir, "agent edit")

        (repo / "init.txt").write_text("user version\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "user edit"],
            cwd=repo, capture_output=True,
        )

        server = VSCodeServer()
        server.work_dir = str(repo)
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=baseline,
        )
        state = _register_wt_state("bug56b-tab", agent)

        try:
            has_conflict = server._check_merge_conflict("bug56b-tab")
            assert has_conflict is True, (
                "Regression: valid baseline should still detect conflicts"
            )
        finally:
            agent_state.unregister(state.task_id, state)

        _cleanup(repo, branch, wt_dir)

    def test_no_conflict_returns_false(self, tmp_path: Path) -> None:
        """No overlapping changes → no conflict."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug56c-1"
        wt_dir, baseline = _create_worktree_branch(repo, branch)

        _add_agent_commit(wt_dir, "agent.txt", "agent work\n")

        server = VSCodeServer()
        server.work_dir = str(repo)
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=baseline,
        )
        state = _register_wt_state("bug56c-tab", agent)

        try:
            has_conflict = server._check_merge_conflict("bug56c-tab")
            assert has_conflict is False
        finally:
            agent_state.unregister(state.task_id, state)

        _cleanup(repo, branch, wt_dir)
