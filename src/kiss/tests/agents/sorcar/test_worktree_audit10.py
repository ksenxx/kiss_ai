"""Audit 10: Integration tests for bugs in worktree and non-worktree workflows.

BUG-49: `_do_merge` pops stash on CONFLICT/MERGE_FAILED, making manual
    merge instructions unusable.  After stash pop the tree is dirty and
    `git cherry-pick --no-commit` (or `git merge --squash`) refuses to
    run.  The user cannot follow the printed instructions.

BUG-50: `_release_worktree` silently orphans a branch when
    `original_branch` is None — no warning, no cleanup, the branch
    lingers forever without the user knowing it exists.

BUG-51: `_get_worktree_changed_files` returns `[]` when `git diff`
    fails (bad base_ref, corrupt repo, etc.), which triggers silent
    auto-discard of valid agent work.  The agent's changes are
    permanently lost with no user notification.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps, MergeResult
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.server import VSCodeServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    (repo / "init.txt").write_text("init")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo, capture_output=True,
    )
    return repo


def _create_worktree_branch(
    repo: Path, branch: str, dirty_file: str | None = None,
) -> tuple[Path, str | None]:
    """Create a worktree branch with optional baseline from dirty state.

    Returns (wt_dir, baseline_commit_or_None).
    """
    wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    GitWorktreeOps.save_original_branch(repo, branch, "main")

    baseline: str | None = None
    if dirty_file:
        # Simulate copy_dirty_state + baseline commit
        (wt_dir / dirty_file).write_text("dirty content")
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
    """Best-effort cleanup of a worktree + branch."""
    if wt_dir.exists():
        GitWorktreeOps.remove(repo, wt_dir)
    GitWorktreeOps.prune(repo)
    if GitWorktreeOps.branch_exists(repo, branch):
        GitWorktreeOps.delete_branch(repo, branch)


# ===================================================================
# BUG-49: _do_merge pops stash on CONFLICT/MERGE_FAILED
# ===================================================================


class TestBug49StashPopOnMergeFailure:
    """BUG-49: _do_merge pops stash even when the merge fails (CONFLICT
    or MERGE_FAILED), leaving the working tree dirty.  The manual merge
    instructions printed to the user will fail because `git cherry-pick
    --no-commit` refuses to run on a dirty tree.

    FIX: On CONFLICT or MERGE_FAILED, do NOT pop the stash.  Return a
    stash_warning telling the user their uncommitted changes are safe
    in `git stash` so the manual merge instructions will work on a
    clean tree, and the user can `git stash pop` after resolving.
    """

    def test_stash_not_popped_on_conflict(self, tmp_path: Path) -> None:
        """After a CONFLICT merge, stash must NOT be popped — tree stays clean."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug49a-1"
        wt_dir, baseline = _create_worktree_branch(repo, branch)

        # Agent edits init.txt on the worktree branch
        _add_agent_commit(wt_dir, "init.txt", "agent version")
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)

        # Create a conflicting change on main
        (repo / "init.txt").write_text("main conflicting version")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "conflict on main"],
            cwd=repo, capture_output=True,
        )

        # User has uncommitted dirty state
        (repo / "user_file.txt").write_text("user wip")

        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "bug49a"
        wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=baseline,
        )
        agent._wt = wt

        result, stash_warning = agent._do_merge(wt)
        assert result == MergeResult.CONFLICT

        # FIX verification: stash must NOT have been popped.
        # The tree should be clean (no dirty user_file.txt) and
        # the stash list should contain an entry.
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo, capture_output=True, text=True,
        )
        assert "user_file.txt" not in status.stdout, (
            "BUG-49: stash was popped on CONFLICT — tree is dirty, "
            "manual instructions will fail"
        )

        stash_list = subprocess.run(
            ["git", "stash", "list"],
            cwd=repo, capture_output=True, text=True,
        )
        assert "kiss" in stash_list.stdout.lower() or stash_list.stdout.strip(), (
            "Stash should contain the user's saved changes"
        )

        # stash_warning should tell the user about their stashed changes
        assert stash_warning, (
            "BUG-49: stash_warning should be non-empty when stash was "
            "not popped due to merge failure"
        )

        # Cleanup
        subprocess.run(["git", "stash", "drop"], cwd=repo, capture_output=True)
        _cleanup(repo, branch, wt_dir)

    def test_stash_not_popped_on_merge_failed(self, tmp_path: Path) -> None:
        """After a MERGE_FAILED, stash must NOT be popped — tree stays clean."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug49b-1"
        wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")

        # Create worktree with NO baseline (so squash_merge_branch is used)
        assert GitWorktreeOps.create(repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(repo, branch, "main")

        # Agent edits a file
        _add_agent_commit(wt_dir, "agent.txt", "agent work")
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)

        # Create a pre-commit hook that rejects commits — forces MERGE_FAILED
        hooks_dir = repo / ".git" / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook = hooks_dir / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        # User has dirty state
        (repo / "user_dirty.txt").write_text("user wip")

        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "bug49b"
        wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
        )
        agent._wt = wt

        result, stash_warning = agent._do_merge(wt)
        assert result == MergeResult.MERGE_FAILED

        # FIX: tree should be clean (stash NOT popped)
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo, capture_output=True, text=True,
        )
        assert "user_dirty.txt" not in status.stdout, (
            "BUG-49: stash was popped on MERGE_FAILED — tree is dirty"
        )

        assert stash_warning, (
            "BUG-49: stash_warning should be set when stash was kept "
            "due to merge failure"
        )

        # Cleanup
        hook.unlink()
        subprocess.run(["git", "stash", "drop"], cwd=repo, capture_output=True)
        _cleanup(repo, branch, wt_dir)

    def test_stash_still_popped_on_success(self, tmp_path: Path) -> None:
        """On SUCCESS merge, stash MUST still be popped — user changes restored."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug49c-1"
        wt_dir, baseline = _create_worktree_branch(repo, branch)

        # Agent edits a non-conflicting file
        _add_agent_commit(wt_dir, "agent.txt", "agent work")
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)

        # User has dirty state on a different file
        (repo / "user_file.txt").write_text("user wip")

        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "bug49c"
        wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=baseline,
        )
        agent._wt = wt

        result, stash_warning = agent._do_merge(wt)
        assert result == MergeResult.SUCCESS

        # On success, stash MUST be popped — user changes restored
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo, capture_output=True, text=True,
        )
        assert "user_file.txt" in status.stdout, (
            "Regression: stash not popped on SUCCESS — user changes lost"
        )

        _cleanup(repo, branch, wt_dir)

    def test_release_worktree_conflict_instructions_mention_stash(
        self, tmp_path: Path,
    ) -> None:
        """_release_worktree CONFLICT warning must mention stash pop."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug49d-1"
        wt_dir, baseline = _create_worktree_branch(repo, branch)

        _add_agent_commit(wt_dir, "init.txt", "agent version")
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)

        # Conflict on main
        (repo / "init.txt").write_text("main conflict")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "conflict"], cwd=repo, capture_output=True,
        )

        # Dirty state — triggers stash
        (repo / "user.txt").write_text("wip")

        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "bug49d"
        agent._wt = GitWorktree(
            repo_root=repo, branch=branch, original_branch="main",
            wt_dir=wt_dir, baseline_commit=baseline,
        )

        released = agent._release_worktree()
        assert released is None
        assert agent._merge_conflict_warning is not None

        # FIX: instructions should mention stash pop
        warning = agent._merge_conflict_warning
        assert "stash" in warning.lower(), (
            "BUG-49: CONFLICT instructions must mention 'git stash pop' "
            "because user's uncommitted changes are in the stash"
        )

        # Cleanup
        subprocess.run(["git", "stash", "drop"], cwd=repo, capture_output=True)
        _cleanup(repo, branch, wt_dir)

    def test_merge_conflict_instructions_mention_stash(
        self, tmp_path: Path,
    ) -> None:
        """merge() CONFLICT message must mention stash pop."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug49e-1"
        wt_dir, baseline = _create_worktree_branch(repo, branch)

        _add_agent_commit(wt_dir, "init.txt", "agent version")
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)

        # Conflict on main
        (repo / "init.txt").write_text("main conflict")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "conflict"], cwd=repo, capture_output=True,
        )
        # Dirty state
        (repo / "user.txt").write_text("wip")

        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "bug49e"
        agent._wt = GitWorktree(
            repo_root=repo, branch=branch, original_branch="main",
            wt_dir=wt_dir, baseline_commit=baseline,
        )

        msg = agent.merge()
        assert "stash" in msg.lower(), (
            "BUG-49: merge() CONFLICT message must mention 'git stash pop'"
        )

        # Cleanup
        subprocess.run(["git", "stash", "drop"], cwd=repo, capture_output=True)
        _cleanup(repo, branch, wt_dir)


# ===================================================================
# BUG-50: _release_worktree silently orphans branch (original_branch=None)
# ===================================================================


class TestBug50ReleaseOrphansBranch:
    """BUG-50: When `original_branch` is None (crash recovery scenario
    where git config was not written and HEAD is detached),
    `_release_worktree` sets `_wt = None` without deleting the branch,
    without setting `_merge_conflict_warning`, and without notifying
    the user.  The branch lingers forever as an orphan.

    FIX: Set `_merge_conflict_warning` with the branch name and
    instructions for manual cleanup.
    """

    def test_warning_set_when_original_branch_is_none(
        self, tmp_path: Path,
    ) -> None:
        """_release_worktree must set _merge_conflict_warning when
        original_branch is None."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug50a-1"
        wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
        assert GitWorktreeOps.create(repo, branch, wt_dir)
        _add_agent_commit(wt_dir, "agent.txt", "work")
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)

        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "bug50a"
        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch=None,  # <-- the crash-recovery scenario
            wt_dir=wt_dir,
        )

        released = agent._release_worktree()
        assert released is None
        assert agent._wt is None

        # FIX: _merge_conflict_warning must be set
        assert agent._merge_conflict_warning is not None, (
            "BUG-50: _release_worktree must set _merge_conflict_warning "
            "when original_branch is None — branch silently orphaned"
        )
        assert branch in agent._merge_conflict_warning, (
            "Warning must mention the orphaned branch name"
        )

        # Branch should still exist for manual recovery
        assert GitWorktreeOps.branch_exists(repo, branch), (
            "Branch should be preserved for manual recovery"
        )

        # Cleanup
        _cleanup(repo, branch, wt_dir)

    def test_no_warning_when_original_branch_set_and_success(
        self, tmp_path: Path,
    ) -> None:
        """Normal release (original_branch set, merge succeeds) should NOT
        set _merge_conflict_warning."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug50b-1"
        wt_dir, baseline = _create_worktree_branch(repo, branch)
        _add_agent_commit(wt_dir, "agent.txt", "work")
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)

        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "bug50b"
        agent._wt = GitWorktree(
            repo_root=repo, branch=branch, original_branch="main",
            wt_dir=wt_dir, baseline_commit=baseline,
        )

        released = agent._release_worktree()
        assert released == "main"
        assert agent._merge_conflict_warning is None

        _cleanup(repo, branch, wt_dir)


# ===================================================================
# BUG-51: _get_worktree_changed_files returns [] on git diff failure
# ===================================================================


class TestBug51DiffFailureSilentDiscard:
    """BUG-51: When `git diff --name-only base_ref` fails inside
    `_get_worktree_changed_files` (e.g. invalid base_ref, corrupt repo),
    the function returns `[]`.  Callers treat `[]` as "no changes" and
    trigger auto-discard, permanently destroying the agent's work.

    FIX: When `git diff` fails, fall back to `git status --porcelain`
    to detect any changes (committed or uncommitted) rather than
    returning an empty list that triggers auto-discard.
    """

    def test_diff_failure_does_not_return_empty(self, tmp_path: Path) -> None:
        """When git diff fails, must still detect changes via fallback."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug51a-1"
        wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
        assert GitWorktreeOps.create(repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(repo, branch, "main")

        # Save a bogus baseline commit that doesn't exist
        bogus_baseline = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        GitWorktreeOps.save_baseline_commit(repo, branch, bogus_baseline)

        # Agent makes real changes (uncommitted)
        (wt_dir / "agent_work.txt").write_text("real agent work")
        GitWorktreeOps.stage_all(wt_dir)
        GitWorktreeOps.commit_staged(wt_dir, "agent commit")

        # Also add an uncommitted file
        (wt_dir / "uncommitted.txt").write_text("wip")

        server = VSCodeServer()
        server.work_dir = str(repo)
        tab = server._get_tab("bug51a-tab")
        tab.use_worktree = True
        tab.agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=bogus_baseline,
        )

        changed = server._get_worktree_changed_files("bug51a-tab")

        # FIX: even though git diff against bogus baseline fails,
        # the function must detect changes via fallback
        assert len(changed) > 0, (
            "BUG-51: _get_worktree_changed_files returned [] on diff "
            "failure — agent work would be silently discarded"
        )

        _cleanup(repo, branch, wt_dir)

    def test_valid_baseline_still_works(self, tmp_path: Path) -> None:
        """Normal case: valid baseline, diff works, changes detected."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug51b-1"
        wt_dir, baseline = _create_worktree_branch(repo, branch)
        _add_agent_commit(wt_dir, "agent.txt", "work")

        server = VSCodeServer()
        server.work_dir = str(repo)
        tab = server._get_tab("bug51b-tab")
        tab.use_worktree = True
        tab.agent._wt = GitWorktree(
            repo_root=repo, branch=branch, original_branch="main",
            wt_dir=wt_dir, baseline_commit=baseline,
        )

        changed = server._get_worktree_changed_files("bug51b-tab")
        assert "agent.txt" in changed

        _cleanup(repo, branch, wt_dir)

    def test_no_changes_returns_empty(self, tmp_path: Path) -> None:
        """When there are genuinely no changes, must still return []."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug51c-1"
        wt_dir, baseline = _create_worktree_branch(repo, branch)

        # No agent changes — baseline only (or no baseline, clean worktree)
        server = VSCodeServer()
        server.work_dir = str(repo)
        tab = server._get_tab("bug51c-tab")
        tab.use_worktree = True
        tab.agent._wt = GitWorktree(
            repo_root=repo, branch=branch, original_branch="main",
            wt_dir=wt_dir, baseline_commit=baseline,
        )

        changed = server._get_worktree_changed_files("bug51c-tab")
        assert changed == [], (
            "No agent changes — should return empty list"
        )

        _cleanup(repo, branch, wt_dir)

    def test_fallback_branch_diff_with_bad_baseline(
        self, tmp_path: Path,
    ) -> None:
        """Worktree already removed + bad baseline: fallback branch diff
        should still detect changes when possible."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug51d-1"
        wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
        assert GitWorktreeOps.create(repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(repo, branch, "main")

        # Agent commits a change
        _add_agent_commit(wt_dir, "agent.txt", "work")

        # Remove worktree but keep branch
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)

        bogus = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        server = VSCodeServer()
        server.work_dir = str(repo)
        tab = server._get_tab("bug51d-tab")
        tab.use_worktree = True
        tab.agent._wt = GitWorktree(
            repo_root=repo, branch=branch, original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=bogus,
        )

        changed = server._get_worktree_changed_files("bug51d-tab")
        # Even with bad baseline, should fall back to diffing against
        # original_branch or return non-empty if branch has commits
        # The key assertion: must not silently return [] when work exists
        # With the fallback, _resolve_base_ref should fall back to
        # merge-base or original_branch when baseline is invalid
        # Note: if _resolve_base_ref returns the bogus SHA and diff
        # fails, the function returns []. That's the bug.
        # After fix: should detect changes.
        assert len(changed) > 0, (
            "BUG-51: branch diff with bad baseline returned [] — "
            "agent work would be lost"
        )

        _cleanup(repo, branch, wt_dir)
