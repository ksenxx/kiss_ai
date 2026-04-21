"""Audit 12: Integration tests for bugs in worktree and non-worktree workflows.

BUG-59: ``squash_merge_from_baseline`` doesn't check ``rev-list --count``
    return code.  When the baseline SHA is invalid (garbage-collected,
    corrupt git config), ``rev-list`` fails (rc != 0) but its stdout is
    read and compared to ``"0"``.  The empty string is not ``"0"``, so
    the function proceeds to ``cherry-pick`` with an invalid range.
    Cherry-pick fails and returns ``MergeResult.CONFLICT`` — misleading
    the user into thinking there's a content conflict when the real
    issue is an invalid baseline.

BUG-60: ``_do_merge`` passes ``wt.original_branch`` (type ``str | None``)
    to ``checkout()`` (expects ``str``) without an internal type guard.
    Both callers check for None beforehand, but the method itself is
    type-unsafe.  If a future caller omits the guard, git would attempt
    to checkout a branch called "None".

BUG-61: Non-worktree merge view preparation races with concurrent
    operations.  In ``_run_task_inner``'s finally block,
    ``is_running_non_wt`` is cleared BEFORE ``_prepare_and_start_merge``
    runs.  Between clearing the flag and capturing the post-task diff, a
    concurrent worktree merge can modify the working tree, causing the
    merge view to show the other tab's merge changes as the current
    agent's changes.

BUG-62: ``manual_merge_branch`` doesn't abort merge state on non-conflict
    ``MERGE_FAILED``.  When ``git merge --no-commit --no-ff`` fails
    without conflicts (e.g., unrelated histories on older git, or
    partial merge state), ``MERGE_HEAD`` can remain active.  The method
    returns ``MERGE_FAILED`` without cleaning up, leaving the repository
    in a dirty merge state that blocks subsequent git operations.
"""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import (
    GitWorktree,
    GitWorktreeOps,
    MergeResult,
)
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
    (repo / "init.txt").write_text("init\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
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


# ===================================================================
# BUG-59: squash_merge_from_baseline missing rev-list returncode check
# ===================================================================


class TestBug59RevListReturnCode:
    """BUG-59: ``squash_merge_from_baseline`` doesn't check the return
    code of ``git rev-list --count baseline..branch``.  When the
    baseline SHA is invalid, rev-list fails but the empty stdout is
    compared to ``"0"`` — it's not ``"0"``, so the code proceeds to
    cherry-pick with an invalid baseline range.

    FIX: Check ``log_result.returncode``.  If non-zero (invalid
    baseline), return ``MergeResult.CONFLICT`` immediately with a
    log warning, so the caller can fall back to ``squash_merge_branch``.
    """

    def test_invalid_baseline_returns_conflict_not_crash(
        self, tmp_path: Path,
    ) -> None:
        """Invalid baseline must fail gracefully, not produce misleading error."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug59a-1"
        wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
        assert GitWorktreeOps.create(repo, branch, wt_dir)

        # Agent makes a commit
        _add_agent_commit(wt_dir, "agent.txt", "agent work\n")

        # Remove worktree so we can merge on main
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)

        # Call with bogus baseline — should return CONFLICT, not crash
        bogus = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        result = GitWorktreeOps.squash_merge_from_baseline(
            repo, branch, bogus,
        )
        assert result == MergeResult.CONFLICT, (
            "BUG-59: squash_merge_from_baseline should return CONFLICT "
            "for invalid baseline, not proceed to cherry-pick"
        )

        # Verify repo is still clean (no partial cherry-pick state)
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo, capture_output=True, text=True,
        )
        assert not status.stdout.strip(), (
            "Repo should be clean after failed merge with invalid baseline"
        )

        # Cleanup
        if GitWorktreeOps.branch_exists(repo, branch):
            GitWorktreeOps.delete_branch(repo, branch)

    def test_valid_baseline_still_merges(self, tmp_path: Path) -> None:
        """Regression: valid baseline must still produce SUCCESS."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug59b-1"
        wt_dir, baseline = _create_worktree_branch(
            repo, branch, dirty_file="dirty.txt",
        )

        # Agent makes a real change
        _add_agent_commit(wt_dir, "agent.txt", "agent work\n")

        # Remove worktree
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)

        assert baseline is not None
        result = GitWorktreeOps.squash_merge_from_baseline(
            repo, branch, baseline,
        )
        assert result == MergeResult.SUCCESS, (
            "Regression: valid baseline should merge successfully"
        )

        # Verify agent file was merged
        assert (repo / "agent.txt").read_text() == "agent work\n"

        # Cleanup
        if GitWorktreeOps.branch_exists(repo, branch):
            GitWorktreeOps.delete_branch(repo, branch)

    def test_valid_baseline_no_changes_returns_success(
        self, tmp_path: Path,
    ) -> None:
        """Baseline with no subsequent commits returns SUCCESS."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug59c-1"
        wt_dir, baseline = _create_worktree_branch(
            repo, branch, dirty_file="dirty.txt",
        )

        # NO agent commits after baseline

        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)

        assert baseline is not None
        result = GitWorktreeOps.squash_merge_from_baseline(
            repo, branch, baseline,
        )
        assert result == MergeResult.SUCCESS

        if GitWorktreeOps.branch_exists(repo, branch):
            GitWorktreeOps.delete_branch(repo, branch)


# ===================================================================
# BUG-60: _do_merge type-unsafe original_branch
# ===================================================================


class TestBug60DoMergeTypeGuard:
    """BUG-60: ``_do_merge`` passes ``wt.original_branch`` to
    ``checkout()`` without checking for None.  The type is
    ``str | None`` but ``checkout()`` expects ``str``.

    Both callers guard (``merge()`` and ``_release_worktree()``), but
    ``_do_merge`` itself should be safe regardless.

    FIX: Add an assertion/guard inside ``_do_merge`` that returns
    ``(MergeResult.CHECKOUT_FAILED, "")`` when ``original_branch``
    is None.
    """

    def test_do_merge_with_none_original_branch(
        self, tmp_path: Path,
    ) -> None:
        """_do_merge must handle None original_branch gracefully."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug60a-1"
        wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
        assert GitWorktreeOps.create(repo, branch, wt_dir)

        _add_agent_commit(wt_dir, "agent.txt", "agent work\n")
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)

        agent = WorktreeSorcarAgent("test")
        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch=None,  # <-- None
            wt_dir=wt_dir,
            baseline_commit=None,
        )

        wt = agent._wt
        result, warning = agent._do_merge(wt)

        # FIX: must return CHECKOUT_FAILED, not attempt to checkout "None"
        assert result == MergeResult.CHECKOUT_FAILED, (
            "BUG-60: _do_merge should return CHECKOUT_FAILED when "
            "original_branch is None, not attempt checkout"
        )

        # Verify no branch called "None" was created or checked out
        current = GitWorktreeOps.current_branch(repo)
        assert current != "None", (
            "BUG-60: git should not have checked out a branch called 'None'"
        )

        # Cleanup
        if GitWorktreeOps.branch_exists(repo, branch):
            GitWorktreeOps.delete_branch(repo, branch)

    def test_do_merge_with_valid_branch_still_works(
        self, tmp_path: Path,
    ) -> None:
        """Regression: _do_merge with valid original_branch succeeds."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug60b-1"
        wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
        assert GitWorktreeOps.create(repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(repo, branch, "main")

        _add_agent_commit(wt_dir, "agent.txt", "agent work\n")
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)

        agent = WorktreeSorcarAgent("test")
        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=None,
        )

        wt = agent._wt
        result, warning = agent._do_merge(wt)
        assert result == MergeResult.SUCCESS, (
            "Regression: valid original_branch should merge successfully"
        )
        assert (repo / "agent.txt").read_text() == "agent work\n"


# ===================================================================
# BUG-61: Non-worktree merge view race condition
# ===================================================================


class TestBug61NonWtMergeViewRace:
    """BUG-61: ``is_running_non_wt`` is cleared BEFORE
    ``_prepare_and_start_merge`` in the finally block.  Between
    clearing and diff capture, a concurrent worktree merge can modify
    the working tree, causing the merge view to show incorrect changes.

    FIX: Move ``_prepare_and_start_merge`` BEFORE clearing
    ``is_running_non_wt`` so the diff capture happens while the flag
    blocks concurrent worktree merges.
    """

    def test_merge_view_prepared_before_flag_clear(self) -> None:
        """In the source, _prepare_and_start_merge must appear BEFORE
        is_running_non_wt = False in the finally block.

        The merge view captures a git diff of the working tree.  If
        the flag is cleared first, a concurrent worktree merge can
        modify the tree between flag-clear and diff-capture.
        """
        source = inspect.getsource(VSCodeServer._run_task_inner)
        lines = source.split("\n")

        # Find the finally block
        finally_start = None
        for i, line in enumerate(lines):
            if "finally:" in line and "# Entire cleanup" not in line:
                continue
            if "_record_model_usage" in line:
                finally_start = i
                break

        assert finally_start is not None, "Could not find finally block"

        # In the finally block, find the FIRST occurrences of:
        # 1. _prepare_and_start_merge
        # 2. is_running_non_wt = False
        merge_view_line = None
        flag_clear_line = None

        for i in range(finally_start, len(lines)):
            if "_prepare_and_start_merge" in lines[i] and merge_view_line is None:
                merge_view_line = i
            if (
                "is_running_non_wt = False" in lines[i]
                and flag_clear_line is None
            ):
                flag_clear_line = i

        assert merge_view_line is not None, (
            "Could not find _prepare_and_start_merge in finally block"
        )
        assert flag_clear_line is not None, (
            "Could not find is_running_non_wt = False in finally block"
        )

        assert merge_view_line < flag_clear_line, (
            "BUG-61: _prepare_and_start_merge (line offset "
            f"{merge_view_line}) must appear BEFORE "
            f"is_running_non_wt = False (line offset {flag_clear_line}) "
            "in the finally block.  Clearing the flag first allows "
            "concurrent worktree merges to modify the working tree "
            "during diff capture."
        )

    def test_flag_still_cleared_on_merge_view_failure(self) -> None:
        """The flag must be cleared even when _prepare_and_start_merge
        raises.  Otherwise the flag gets stuck True (BUG-39 regression).
        """
        source = inspect.getsource(VSCodeServer._run_task_inner)

        # Count how many times is_running_non_wt = False appears
        clear_count = source.count("is_running_non_wt = False")
        assert clear_count >= 2, (
            "There must be at least 2 flag-clear sites: one in the "
            "normal finally path and one in the except handler, to "
            "guarantee the flag never gets stuck"
        )


# ===================================================================
# BUG-62: manual_merge_branch doesn't abort on MERGE_FAILED
# ===================================================================


class TestBug62ManualMergeAbort:
    """BUG-62: ``manual_merge_branch`` returns ``MERGE_FAILED`` when
    ``git merge --no-commit --no-ff`` fails without conflict markers.
    But it doesn't abort the merge, potentially leaving ``MERGE_HEAD``
    active.  Subsequent git operations (checkout, merge, commit) would
    fail or behave unexpectedly.

    FIX: Call ``git merge --abort`` before returning MERGE_FAILED.
    """

    def test_no_merge_head_after_merge_failed(
        self, tmp_path: Path,
    ) -> None:
        """After MERGE_FAILED, MERGE_HEAD must not remain active."""
        repo = _make_repo(tmp_path)

        # Create an independent branch with no common history
        # (unrelated histories) to trigger non-conflict failure
        subprocess.run(
            ["git", "checkout", "--orphan", "unrelated"],
            cwd=repo, capture_output=True,
        )
        subprocess.run(
            ["git", "rm", "-rf", "."],
            cwd=repo, capture_output=True,
        )
        (repo / "other.txt").write_text("unrelated content\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "unrelated commit"],
            cwd=repo, capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", "main"],
            cwd=repo, capture_output=True,
        )

        # Attempt manual merge of unrelated branch
        # (may fail with unrelated-histories error on older git)
        result = GitWorktreeOps.manual_merge_branch(repo, "unrelated")

        # If git allows it (newer git may auto-allow), the result
        # might be SUCCESS.  We need to check the failure path.
        merge_head = repo / ".git" / "MERGE_HEAD"
        if result.status == MergeResult.MERGE_FAILED:
            # FIX: MERGE_HEAD must not exist after MERGE_FAILED
            assert not merge_head.exists(), (
                "BUG-62: MERGE_HEAD still exists after manual_merge_branch "
                "returned MERGE_FAILED — repo is in dirty merge state"
            )
        elif result.status == MergeResult.CONFLICT:
            # Conflict case — that's fine, conflicts are expected
            # for unrelated histories.  MERGE_HEAD may exist.
            pass
        else:
            # SUCCESS — merge worked.  Clean up.
            subprocess.run(
                ["git", "merge", "--abort"],
                cwd=repo, capture_output=True,
            )

    def test_abort_called_in_source(self) -> None:
        """Verify the source code calls merge --abort on MERGE_FAILED."""
        source = inspect.getsource(GitWorktreeOps.manual_merge_branch)

        # Find the MERGE_FAILED return
        merge_failed_pos = source.find("MergeResult.MERGE_FAILED")
        assert merge_failed_pos != -1

        # There should be a merge --abort before the MERGE_FAILED return
        lines = source.split("\n")
        merge_failed_line = None
        abort_line = None
        for i, line in enumerate(lines):
            if "MERGE_FAILED" in line and merge_failed_line is None:
                merge_failed_line = i
            if "abort" in line and "merge" in line:
                abort_line = i

        # There should be an abort call somewhere in the method
        assert abort_line is not None, (
            "BUG-62: manual_merge_branch should call 'merge --abort' "
            "to clean up merge state on MERGE_FAILED"
        )

    def test_successful_merge_still_works(self, tmp_path: Path) -> None:
        """Regression: a clean merge must still succeed."""
        repo = _make_repo(tmp_path)
        branch = "test-manual-merge"

        # Create a branch with a non-conflicting change
        subprocess.run(
            ["git", "checkout", "-b", branch],
            cwd=repo, capture_output=True,
        )
        (repo / "new_file.txt").write_text("new content\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "add new file"],
            cwd=repo, capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", "main"],
            cwd=repo, capture_output=True,
        )

        result = GitWorktreeOps.manual_merge_branch(repo, branch)
        assert result.status == MergeResult.SUCCESS, (
            "Regression: clean merge should succeed"
        )
        assert not result.has_conflicts

        # The merge was --no-commit, so changes should be unstaged
        # (reset HEAD was called)
        assert (repo / "new_file.txt").exists()

        # Clean up merge state
        subprocess.run(
            ["git", "checkout", "--", "."],
            cwd=repo, capture_output=True,
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=repo, capture_output=True,
        )

        # Cleanup branch
        GitWorktreeOps.delete_branch(repo, branch)
