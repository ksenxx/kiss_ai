# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 9: Tests verifying fixes for bugs, inconsistencies, and
redundancies in both non-worktree and worktree workflows.

BUG-39: `is_running_non_wt` flag is now cleared at the very start of
    the finally block's try (before any risky calls) AND in the outer
    except handler, so it can never get permanently stuck.

BUG-40 / INC-4: `_do_merge` now returns `(MergeResult.CHECKOUT_FAILED, "")`
    instead of `(None, checkout_error_str)`.  `_release_worktree` checks
    `result == MergeResult.CHECKOUT_FAILED` instead of `result is None`,
    so the checkout error is never misattributed to `_stash_pop_warning`.

BUG-41 / RED-6: `_start_merge_session` now accepts a `tab_id` parameter.
    All callers pass it explicitly.  `is_merging` is always set correctly,
    even on the session-replay path.

BUG-42 / INC-5: Auto-discard in both `_run_task_inner` and `_finish_merge`
    now checks `_any_non_wt_running()` before calling `discard()`.

BUG-43: Manual merge instructions now use `git cherry-pick --no-commit
    baseline..branch` when a baseline commit exists, matching what the
    auto-merge actually does.

BUG-44: `_new_chat` guard now checks `tab.agent._wt_pending` regardless
    of `tab.use_worktree`, so a tab that switched modes still gets the
    non-wt-running guard.

INC-6: `_check_merge_conflict` now checks both `unstaged_files()` AND
    `staged_files()` for dirty-file overlap.

RED-5: The two consecutive `if not tab.use_worktree:` blocks in
    `_run_task_inner`'s finally are now a single block.

RED-6: See BUG-41.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
)
from kiss.agents.sorcar.worktree_sorcar_agent import (
    WorktreeSorcarAgent,
    _manual_merge_cmd,
)
from kiss.agents.vscode.server import VSCodeServer


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




class TestBug40Inc4Fix:
    """BUG-40/INC-4 FIX: _do_merge returns MergeResult.CHECKOUT_FAILED
    instead of (None, err), and _release_worktree never misattributes
    checkout errors to _stash_pop_warning."""




    def test_checkout_error_not_stored_as_stash_warning(self, tmp_path):
        """Checkout failure does NOT set _stash_pop_warning."""
        repo = _make_repo(tmp_path)
        agent = WorktreeSorcarAgent("test")
        agent._chat_id = "bug40"

        branch = "kiss/wt-bug40-test"
        wt_dir = repo / ".kiss-worktrees" / "wt-bug40"
        GitWorktreeOps.create(repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(repo, branch, "main")

        (wt_dir / "file.txt").write_text("agent work")
        GitWorktreeOps.commit_all(wt_dir, "agent work")

        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="nonexistent-branch",
            wt_dir=wt_dir,
        )

        agent._release_worktree()

        assert agent._stash_pop_warning is None, (
            "Checkout error must NOT be stored in _stash_pop_warning"
        )
        assert agent._merge_conflict_warning is not None
        assert "checkout" in agent._merge_conflict_warning.lower()

        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)
        if GitWorktreeOps.branch_exists(repo, branch):
            GitWorktreeOps.delete_branch(repo, branch)


class TestBug41Red6Fix:
    """BUG-41/RED-6 FIX: _start_merge_session accepts tab_id parameter
    and all callers pass it."""




    def test_is_merging_set_with_explicit_tab_id(self, tmp_path):
        """When tab_id is passed explicitly, is_merging is set correctly
        even if thread-local tab_id is None (replay path)."""
        server = VSCodeServer()
        tab = server._get_tab("replay-tab")
        tab.agent = WorktreeSorcarAgent("Sorcar VS Code")
        assert not tab.is_merging

        merge_dir = tmp_path / "merge"
        merge_dir.mkdir()
        merge_json = merge_dir / "pending-merge.json"
        merge_json.write_text(json.dumps({
            "files": [{
                "path": "test.txt",
                "hunks": [{"old_start": 1, "new_start": 1}],
            }],
        }))

        result = server._start_merge_session(
            str(merge_json), tab_id="replay-tab",
        )
        assert result is True
        assert tab.is_merging, (
            "is_merging must be True when tab_id is passed explicitly"
        )




class TestBug43Fix:
    """BUG-43 FIX: Instructions use cherry-pick when baseline exists."""

    def test_manual_merge_cmd_with_baseline(self):
        """_manual_merge_cmd returns cherry-pick when baseline exists."""
        wt = GitWorktree(
            repo_root=Path("/repo"),
            branch="kiss/wt-test",
            original_branch="main",
            wt_dir=Path("/repo/.kiss-worktrees/wt"),
            baseline_commit="abc123",
        )
        cmd = _manual_merge_cmd(wt)
        assert "cherry-pick" in cmd
        assert "abc123..kiss/wt-test" in cmd
        assert "merge --squash" not in cmd

    def test_manual_merge_cmd_without_baseline(self):
        """_manual_merge_cmd returns merge --squash when no baseline."""
        wt = GitWorktree(
            repo_root=Path("/repo"),
            branch="kiss/wt-test",
            original_branch="main",
            wt_dir=Path("/repo/.kiss-worktrees/wt"),
        )
        cmd = _manual_merge_cmd(wt)
        assert "merge --squash" in cmd
        assert "cherry-pick" not in cmd




    def test_functional_instructions_match_auto_merge(self, tmp_path):
        """Instructions produce the same result as auto-merge when baseline exists."""
        repo = _make_repo(tmp_path)

        branch = "kiss/wt-bug43-test"
        wt_dir = repo / ".kiss-worktrees" / "wt-bug43"
        GitWorktreeOps.create(repo, branch, wt_dir)

        (wt_dir / "dirty.txt").write_text("user dirty content")
        subprocess.run(["git", "add", "-A"], cwd=wt_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "baseline"],
            cwd=wt_dir, capture_output=True,
        )
        baseline = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=wt_dir, capture_output=True, text=True,
        ).stdout.strip()

        (wt_dir / "agent.txt").write_text("agent work")
        subprocess.run(["git", "add", "-A"], cwd=wt_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "agent work"],
            cwd=wt_dir, capture_output=True,
        )

        wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=baseline,
        )
        cmd = _manual_merge_cmd(wt)

        assert "cherry-pick" in cmd

        result = subprocess.run(
            cmd.split(), cwd=repo, capture_output=True, text=True,
        )
        assert result.returncode == 0
        status = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=repo, capture_output=True, text=True,
        )
        files = set(status.stdout.strip().splitlines())
        assert "agent.txt" in files
        assert "dirty.txt" not in files, (
            "Cherry-pick should NOT include baseline dirty state"
        )

        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo, capture_output=True)
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)
        if GitWorktreeOps.branch_exists(repo, branch):
            GitWorktreeOps.delete_branch(repo, branch)


class TestInc6Fix:
    """INC-6 FIX: _check_merge_conflict checks both unstaged and staged files."""


    def test_staged_overlap_detected(self, tmp_path):
        """A staged file overlapping with worktree changes IS detected."""
        repo = _make_repo(tmp_path)

        branch = "kiss/wt-inc6-test"
        wt_dir = repo / ".kiss-worktrees" / "wt-inc6"
        GitWorktreeOps.create(repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(repo, branch, "main")

        (wt_dir / "init.txt").write_text("agent changes")
        subprocess.run(["git", "add", "-A"], cwd=wt_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "agent work"],
            cwd=wt_dir, capture_output=True,
        )

        (repo / "init.txt").write_text("user staged change")
        subprocess.run(["git", "add", "init.txt"], cwd=repo, capture_output=True)

        server = VSCodeServer()
        server.work_dir = str(repo)
        tab = server._get_tab("inc6-tab")
        tab.agent = WorktreeSorcarAgent("Sorcar VS Code")
        tab.use_worktree = True
        tab.agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
        )

        has_conflict = server._check_merge_conflict("inc6-tab")
        assert has_conflict, (
            "INC-6 fix: staged file overlap must be detected"
        )

        subprocess.run(["git", "reset", "HEAD", "init.txt"], cwd=repo, capture_output=True)
        subprocess.run(["git", "checkout", "--", "init.txt"], cwd=repo, capture_output=True)
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)
        if GitWorktreeOps.branch_exists(repo, branch):
            GitWorktreeOps.delete_branch(repo, branch)




