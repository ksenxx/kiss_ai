# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Characterization/lockdown tests for GitWorktreeOps (git_worktree.py).

These integration tests pin the CURRENT externally-observable behavior
of :class:`GitWorktreeOps` so that the planned simplifications
(tmp/findings-2.md sections C1-C8 — commit_all as stage_all+commit_staged,
branch-config get/set helpers, shared merge-commit tail, cleanup_partial
delegating to remove+prune, stash_if_dirty reusing has_uncommitted_changes,
flattened delete_branch, shared changed-files helper) cannot silently
change any contract.  Every test runs against a fresh real git repo.
"""

from __future__ import annotations

import stat
import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
    MergeResult,
    _git,
)


def _make_repo(path: Path) -> Path:
    """Create a git repo with one initial commit at *path*."""
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
    (path / "README.md").write_text("# Test\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True,
        check=True,
    )
    return path


def _create_worktree(repo: Path, branch: str) -> Path:
    """Create a worktree at repo/.kiss-worktrees/<slug>.

    Mirrors production setup: ``.kiss-worktrees/`` is added to the
    repo's local git exclude so it never shows up as untracked.
    """
    GitWorktreeOps.ensure_excluded(repo)
    slug = branch.replace("/", "_")
    wt_dir = repo / ".kiss-worktrees" / slug
    assert GitWorktreeOps.create(repo, branch, wt_dir)
    return wt_dir


def _commit_count(cwd: Path) -> int:
    """Number of commits reachable from HEAD in *cwd*."""
    result = _git("rev-list", "--count", "HEAD", cwd=cwd)
    assert result.returncode == 0
    return int(result.stdout.strip())


def _head_message(cwd: Path) -> str:
    """Full message (subject + body) of HEAD in *cwd*."""
    result = _git("log", "-1", "--format=%B", cwd=cwd)
    assert result.returncode == 0
    return result.stdout.rstrip()


def _install_rejecting_pre_commit_hook(repo: Path) -> None:
    """Install a pre-commit hook in *repo* that always rejects."""
    hooks_dir = repo / ".git" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook = hooks_dir / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class TestCommitAll:
    """C1: commit_all == stage_all + commit_staged, observably."""

    def test_single_commit_includes_all_change_kinds(self, tmp_path: Path) -> None:
        """One commit_all call captures staged+unstaged+untracked changes."""
        repo = _make_repo(tmp_path / "repo")
        # staged change
        (repo / "staged.txt").write_text("staged\n")
        _git("add", "staged.txt", cwd=repo)
        # unstaged change to a tracked file
        (repo / "README.md").write_text("# Modified\n")
        # untracked file
        (repo / "untracked.txt").write_text("untracked\n")

        before = _commit_count(repo)
        assert GitWorktreeOps.commit_all(repo, "lockdown: all kinds") is True
        assert _commit_count(repo) == before + 1
        assert _head_message(repo) == "lockdown: all kinds"

        # All three changes are in the new commit; tree is clean.
        show = _git(
            "show", "--name-only", "--format=", "HEAD", cwd=repo
        ).stdout.split()
        assert sorted(show) == ["README.md", "staged.txt", "untracked.txt"]
        assert not GitWorktreeOps.has_uncommitted_changes(repo)

    def test_returns_false_when_nothing_to_commit(self, tmp_path: Path) -> None:
        """commit_all on a clean tree commits nothing and returns False."""
        repo = _make_repo(tmp_path / "repo")
        before = _commit_count(repo)
        assert GitWorktreeOps.commit_all(repo, "noop") is False
        assert _commit_count(repo) == before

    def test_equivalent_to_stage_all_plus_commit_staged(self, tmp_path: Path) -> None:
        """Same inputs through both paths produce identical commits."""
        repo_a = _make_repo(tmp_path / "repo_a")
        repo_b = _make_repo(tmp_path / "repo_b")
        for repo in (repo_a, repo_b):
            (repo / "new.txt").write_text("new\n")
            (repo / "README.md").write_text("# Edited\n")

        assert GitWorktreeOps.commit_all(repo_a, "msg") is True
        GitWorktreeOps.stage_all(repo_b)
        assert GitWorktreeOps.commit_staged(repo_b, "msg") is True

        tree_a = _git("rev-parse", "HEAD^{tree}", cwd=repo_a).stdout.strip()
        tree_b = _git("rev-parse", "HEAD^{tree}", cwd=repo_b).stdout.strip()
        assert tree_a == tree_b
        assert _head_message(repo_a) == _head_message(repo_b) == "msg"


class TestCommitStaged:
    """C1/C7 lockdown: commit_staged hook and empty-index contracts."""

    def test_returns_false_when_nothing_staged(self, tmp_path: Path) -> None:
        """Unstaged-only changes are not committed; returns False."""
        repo = _make_repo(tmp_path / "repo")
        (repo / "README.md").write_text("# Unstaged only\n")
        before = _commit_count(repo)
        assert GitWorktreeOps.commit_staged(repo, "nothing staged") is False
        assert _commit_count(repo) == before
        # The unstaged change is untouched.
        assert (repo / "README.md").read_text() == "# Unstaged only\n"

    def test_rejecting_hook_returns_false_not_raise(self, tmp_path: Path) -> None:
        """A failing pre-commit hook yields False, no exception, no commit."""
        repo = _make_repo(tmp_path / "repo")
        _install_rejecting_pre_commit_hook(repo)
        (repo / "hooked.txt").write_text("x\n")
        GitWorktreeOps.stage_all(repo)
        before = _commit_count(repo)
        assert GitWorktreeOps.commit_staged(repo, "rejected") is False
        assert _commit_count(repo) == before

    def test_no_verify_bypasses_hook(self, tmp_path: Path) -> None:
        """no_verify=True commits despite a rejecting pre-commit hook."""
        repo = _make_repo(tmp_path / "repo")
        _install_rejecting_pre_commit_hook(repo)
        (repo / "hooked.txt").write_text("x\n")
        GitWorktreeOps.stage_all(repo)
        before = _commit_count(repo)
        assert GitWorktreeOps.commit_staged(repo, "bypassed", no_verify=True) is True
        assert _commit_count(repo) == before + 1
        assert _head_message(repo) == "bypassed"


class TestDeleteBranch:
    """C7: delete_branch outcomes and config-section cleanup."""

    def test_safe_delete_merged_branch_removes_config(self, tmp_path: Path) -> None:
        """Merged branch: -d succeeds AND branch.<name> config is removed."""
        repo = _make_repo(tmp_path / "repo")
        branch = "kiss/wt-merged"
        _git("branch", branch, cwd=repo)
        cfg = _git("config", f"branch.{branch}.kiss-original", "main", cwd=repo)
        assert cfg.returncode == 0
        assert (
            _git("config", "--get", f"branch.{branch}.kiss-original", cwd=repo)
            .stdout.strip()
            == "main"
        )

        assert GitWorktreeOps.delete_branch(repo, branch) is True
        assert not GitWorktreeOps.branch_exists(repo, branch)
        # The branch.<name> config section must actually be gone.
        get = _git("config", "--get", f"branch.{branch}.kiss-original", cwd=repo)
        assert get.returncode != 0
        assert get.stdout.strip() == ""

    def test_force_delete_fallback_for_unmerged_branch(self, tmp_path: Path) -> None:
        """Unmerged branch: -d fails, -D fallback deletes; config removed."""
        repo = _make_repo(tmp_path / "repo")
        branch = "kiss/wt-unmerged"
        wt_dir = _create_worktree(repo, branch)
        (wt_dir / "extra.txt").write_text("unmerged work\n")
        assert GitWorktreeOps.commit_all(wt_dir, "unmerged commit")
        _git("config", f"branch.{branch}.kiss-original", "main", cwd=repo)
        # Free the branch from the worktree so only "unmerged" blocks -d.
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)
        # Sanity: safe delete alone would refuse.
        assert _git("branch", "-d", branch, cwd=repo).returncode != 0

        assert GitWorktreeOps.delete_branch(repo, branch) is True
        assert not GitWorktreeOps.branch_exists(repo, branch)
        get = _git("config", "--get", f"branch.{branch}.kiss-original", cwd=repo)
        assert get.returncode != 0

    def test_returns_true_when_branch_never_existed(self, tmp_path: Path) -> None:
        """Deleting a nonexistent branch reports success."""
        repo = _make_repo(tmp_path / "repo")
        assert GitWorktreeOps.delete_branch(repo, "kiss/wt-ghost") is True

    def test_returns_false_when_checked_out_in_worktree(self, tmp_path: Path) -> None:
        """Branch that is HEAD of a live worktree cannot be deleted."""
        repo = _make_repo(tmp_path / "repo")
        branch = "kiss/wt-live"
        _create_worktree(repo, branch)
        assert GitWorktreeOps.delete_branch(repo, branch) is False
        assert GitWorktreeOps.branch_exists(repo, branch)


class TestStash:
    """C6: stash_if_dirty / stash_pop contracts."""

    def test_clean_tree_returns_false(self, tmp_path: Path) -> None:
        """No stash entry is created for a clean tree."""
        repo = _make_repo(tmp_path / "repo")
        assert GitWorktreeOps.stash_if_dirty(repo) is False
        assert _git("stash", "list", cwd=repo).stdout.strip() == ""

    def test_dirty_tree_stashes_and_pop_restores(self, tmp_path: Path) -> None:
        """Dirty tree -> True and clean afterwards; pop restores changes."""
        repo = _make_repo(tmp_path / "repo")
        (repo / "README.md").write_text("# Dirty\n")
        (repo / "fresh.txt").write_text("untracked\n")

        assert GitWorktreeOps.stash_if_dirty(repo) is True
        assert not GitWorktreeOps.has_uncommitted_changes(repo)
        assert not (repo / "fresh.txt").exists()
        assert (repo / "README.md").read_text() == "# Test\n"

        assert GitWorktreeOps.stash_pop(repo) is True
        assert (repo / "README.md").read_text() == "# Dirty\n"
        assert (repo / "fresh.txt").read_text() == "untracked\n"


class TestChangedFileLists:
    """C8: unstaged_files / staged_files report the right lists."""

    def test_staged_and_unstaged_reported_separately(self, tmp_path: Path) -> None:
        """One staged and one unstaged modification land in the right list."""
        repo = _make_repo(tmp_path / "repo")
        (repo / "tracked2.txt").write_text("v1\n")
        _git("add", "tracked2.txt", cwd=repo)
        _git("commit", "-m", "add tracked2", cwd=repo)

        (repo / "tracked2.txt").write_text("v2 staged\n")
        _git("add", "tracked2.txt", cwd=repo)
        (repo / "README.md").write_text("# unstaged edit\n")

        assert GitWorktreeOps.staged_files(repo) == ["tracked2.txt"]
        assert GitWorktreeOps.unstaged_files(repo) == ["README.md"]

    def test_clean_repo_returns_empty_lists(self, tmp_path: Path) -> None:
        """Clean tree yields empty lists from both helpers."""
        repo = _make_repo(tmp_path / "repo")
        assert GitWorktreeOps.staged_files(repo) == []
        assert GitWorktreeOps.unstaged_files(repo) == []


class TestCleanupPartial:
    """C5: cleanup_partial removes worktree registration and branch."""

    def test_removes_worktree_and_branch(self, tmp_path: Path) -> None:
        """Directory, git registration, and branch are all gone after."""
        repo = _make_repo(tmp_path / "repo")
        branch = "kiss/wt-partial"
        wt_dir = _create_worktree(repo, branch)
        assert wt_dir.exists()
        assert branch in _git("worktree", "list", "--porcelain", cwd=repo).stdout

        GitWorktreeOps.cleanup_partial(repo, branch, wt_dir)

        assert not wt_dir.exists()
        listing = _git("worktree", "list", "--porcelain", cwd=repo).stdout
        assert branch not in listing
        assert not GitWorktreeOps.branch_exists(repo, branch)


class TestSquashMergeRoundTrip:
    """C4: full create -> commit -> squash_merge_branch round-trip."""

    def test_merge_applies_one_commit_with_branch_head_message(
        self, tmp_path: Path
    ) -> None:
        """Worktree changes land on main as ONE commit, message preserved."""
        repo = _make_repo(tmp_path / "repo")
        branch = "kiss/wt-roundtrip"
        wt_dir = _create_worktree(repo, branch)

        (wt_dir / "feature.txt").write_text("feature work\n")
        (wt_dir / "second.txt").write_text("second file\n")
        assert GitWorktreeOps.commit_all(wt_dir, "agent: did the task\n\nDetails.")

        before = _commit_count(repo)
        assert GitWorktreeOps.squash_merge_branch(repo, branch) is MergeResult.SUCCESS
        assert _commit_count(repo) == before + 1
        assert _head_message(repo) == "agent: did the task\n\nDetails."
        assert (repo / "feature.txt").read_text() == "feature work\n"
        assert (repo / "second.txt").read_text() == "second file\n"
        assert not GitWorktreeOps.has_uncommitted_changes(repo)

    def test_conflict_returns_conflict_and_main_left_clean(
        self, tmp_path: Path
    ) -> None:
        """Conflicting edits -> CONFLICT, main tree fully reset/clean."""
        repo = _make_repo(tmp_path / "repo")
        branch = "kiss/wt-conflict"
        wt_dir = _create_worktree(repo, branch)

        (wt_dir / "README.md").write_text("# Worktree version\n")
        assert GitWorktreeOps.commit_all(wt_dir, "worktree edit")

        (repo / "README.md").write_text("# Main version\n")
        assert GitWorktreeOps.commit_all(repo, "main edit")

        before = _commit_count(repo)
        assert GitWorktreeOps.squash_merge_branch(repo, branch) is MergeResult.CONFLICT
        assert _commit_count(repo) == before
        assert not GitWorktreeOps.has_uncommitted_changes(repo)
        assert (repo / "README.md").read_text() == "# Main version\n"


class TestBranchConfigRoundTrip:
    """C3: save/load original-branch and baseline-commit config pairs."""

    def test_original_branch_round_trip(self, tmp_path: Path) -> None:
        """save_original_branch then load_original_branch returns the value."""
        repo = _make_repo(tmp_path / "repo")
        branch = "kiss/wt-cfg"
        _git("branch", branch, cwd=repo)
        assert GitWorktreeOps.save_original_branch(repo, branch, "main") is True
        assert GitWorktreeOps.load_original_branch(repo, branch) == "main"

    def test_baseline_commit_round_trip(self, tmp_path: Path) -> None:
        """save_baseline_commit then load_baseline_commit returns the SHA."""
        repo = _make_repo(tmp_path / "repo")
        branch = "kiss/wt-cfg2"
        _git("branch", branch, cwd=repo)
        assert GitWorktreeOps.save_baseline_commit(repo, branch, "deadbeef") is True
        assert GitWorktreeOps.load_baseline_commit(repo, branch) == "deadbeef"

    def test_load_returns_none_when_unset(self, tmp_path: Path) -> None:
        """Both loaders return None for branches without stored config."""
        repo = _make_repo(tmp_path / "repo")
        branch = "kiss/wt-bare"
        _git("branch", branch, cwd=repo)
        assert GitWorktreeOps.load_original_branch(repo, branch) is None
        assert GitWorktreeOps.load_baseline_commit(repo, branch) is None
        assert GitWorktreeOps.load_original_branch(repo, "no/such") is None
        assert GitWorktreeOps.load_baseline_commit(repo, "no/such") is None


class TestSquashMergeFromBaseline:
    """C4: squash_merge_from_baseline merges only post-baseline commits."""

    def test_merges_only_post_baseline_commits(self, tmp_path: Path) -> None:
        """Baseline-commit content stays out; agent commits land as one."""
        repo = _make_repo(tmp_path / "repo")
        branch = "kiss/wt-baseline"
        wt_dir = _create_worktree(repo, branch)

        # Baseline commit captures (simulated) user dirty state.
        (wt_dir / "user_dirty.txt").write_text("user dirty state\n")
        assert GitWorktreeOps.commit_all(wt_dir, "kiss: baseline")
        baseline = GitWorktreeOps.head_sha(wt_dir)
        assert baseline is not None

        # Agent work after the baseline.
        (wt_dir / "agent.txt").write_text("agent change\n")
        assert GitWorktreeOps.commit_all(wt_dir, "agent: post-baseline work")

        before = _commit_count(repo)
        result = GitWorktreeOps.squash_merge_from_baseline(repo, branch, baseline)
        assert result is MergeResult.SUCCESS
        assert _commit_count(repo) == before + 1
        assert _head_message(repo) == "agent: post-baseline work"
        assert (repo / "agent.txt").read_text() == "agent change\n"
        # The baseline (user dirty state) commit must NOT be merged.
        assert not (repo / "user_dirty.txt").exists()
        assert not GitWorktreeOps.has_uncommitted_changes(repo)

    def test_baseline_equals_head_is_success_noop(self, tmp_path: Path) -> None:
        """baseline == branch HEAD -> SUCCESS with no new commit on main."""
        repo = _make_repo(tmp_path / "repo")
        branch = "kiss/wt-noop"
        wt_dir = _create_worktree(repo, branch)

        (wt_dir / "user_dirty.txt").write_text("user dirty state\n")
        assert GitWorktreeOps.commit_all(wt_dir, "kiss: baseline")
        baseline = GitWorktreeOps.head_sha(wt_dir)
        assert baseline is not None

        before = _commit_count(repo)
        head_before = GitWorktreeOps.head_sha(repo)
        result = GitWorktreeOps.squash_merge_from_baseline(repo, branch, baseline)
        assert result is MergeResult.SUCCESS
        assert _commit_count(repo) == before
        assert GitWorktreeOps.head_sha(repo) == head_before
        assert not GitWorktreeOps.has_uncommitted_changes(repo)
