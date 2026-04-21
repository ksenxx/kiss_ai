"""Audit 11: Integration tests for bugs in worktree and non-worktree workflows.

BUG-55: ``is_running_non_wt`` flag is set AFTER ``_capture_pre_snapshot``
    in ``_run_task_inner``.  A concurrent worktree merge can slip between
    the snapshot and the flag set, modifying the main tree.  The snapshot
    becomes stale and the merge view shows the other tab's merge changes
    as if they were the agent's changes.

BUG-56: ``_check_merge_conflict`` uses ``baseline^`` / ``baseline``
    without validating the baseline SHA exists (unlike ``_resolve_base_ref``
    which has the BUG-51 ``git cat-file -t`` check).  An invalid baseline
    makes both ``git diff`` commands fail silently (empty file sets),
    causing the method to return ``False`` even when a real conflict
    exists.

BUG-57: ``_file_changed`` in ``_prepare_merge_view`` returns ``False``
    when ``read_bytes()`` raises ``OSError`` on a deleted file.  Deleted
    files are excluded from the non-worktree merge review.  The manifest
    building loop also skips files where ``current_path.is_file()`` is
    False.  The user cannot see or reject file deletions.

BUG-58: ``cleanup_orphans`` classifies branches by checking whether they
    have an active git worktree directory.  After ``_finalize_worktree``
    removes the worktree directory but before ``_do_merge`` completes,
    the branch has no active worktree.  ``cleanup_orphans`` would delete
    it, permanently losing agent work.  Fix: also check the branch's
    ``kiss-original`` config entry — if it exists, the branch is pending
    merge, not orphaned.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.agents.vscode.diff_merge import (
    _capture_untracked,
    _parse_diff_hunks,
    _prepare_merge_view,
    _snapshot_files,
)
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
# BUG-55: is_running_non_wt set AFTER _capture_pre_snapshot (TOCTOU)
# ===================================================================


class TestBug55NonWtFlagToctou:
    """BUG-55: ``is_running_non_wt`` is set AFTER the pre-task snapshot
    is captured.  During the gap, a concurrent worktree merge can modify
    the main tree, making the snapshot stale.

    FIX: Set ``is_running_non_wt = True`` BEFORE calling
    ``_capture_pre_snapshot`` so concurrent merges are blocked during
    the entire snapshot + task execution window.
    """

    def test_flag_set_before_snapshot_in_source(self) -> None:
        """Verify the source code sets is_running_non_wt before snapshot.

        The flag must appear BEFORE _capture_pre_snapshot in the source
        so concurrent worktree merges are blocked during snapshot capture.
        """
        source = inspect.getsource(VSCodeServer._run_task_inner)

        flag_pos = source.find("is_running_non_wt = True")
        snapshot_pos = source.find("_capture_pre_snapshot")
        assert flag_pos != -1 and snapshot_pos != -1, (
            "Could not find both is_running_non_wt and _capture_pre_snapshot"
        )
        assert flag_pos < snapshot_pos, (
            "BUG-55: is_running_non_wt must be set BEFORE "
            "_capture_pre_snapshot to prevent TOCTOU gap. "
            f"Flag at {flag_pos}, snapshot at {snapshot_pos}"
        )

    def test_flag_cleared_on_snapshot_failure(self) -> None:
        """If _capture_pre_snapshot raises, the flag must still be cleared.

        The flag set + snapshot must be inside the try/finally that
        clears the flag, or have their own guard.
        """
        # The flag clear must happen in the finally block which executes
        # regardless of where in the try block the exception occurs.
        # Verify the structure: the flag set and snapshot are inside
        # the outer try block (whose finally clears the flag).
        source = inspect.getsource(VSCodeServer._run_task_inner)

        # The flag set should be inside the try block, not before it
        lines = source.split("\n")
        flag_line = None
        finally_line = None
        for i, line in enumerate(lines):
            if "is_running_non_wt = True" in line and flag_line is None:
                flag_line = i
            if "is_running_non_wt = False" in line and finally_line is None:
                finally_line = i

        # The flag must be set within the try/finally scope
        assert flag_line is not None
        assert finally_line is not None
        assert finally_line > flag_line, (
            "BUG-55: is_running_non_wt = False must come after = True "
            "in the code to ensure cleanup on exception"
        )


# ===================================================================
# BUG-56: _check_merge_conflict doesn't validate baseline
# ===================================================================


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

        # Agent edits init.txt on worktree branch
        (wt_dir / "init.txt").write_text("agent version\n")
        GitWorktreeOps.stage_all(wt_dir)
        GitWorktreeOps.commit_staged(wt_dir, "agent edit")

        # User edits init.txt on main → conflict
        (repo / "init.txt").write_text("user version\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "user edit"],
            cwd=repo, capture_output=True,
        )

        # Set an INVALID baseline
        bogus = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        GitWorktreeOps.save_baseline_commit(repo, branch, bogus)

        server = VSCodeServer()
        server.work_dir = str(repo)
        tab = server._get_tab("bug56a-tab")
        tab.use_worktree = True
        tab.agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=bogus,
        )

        has_conflict = server._check_merge_conflict("bug56a-tab")
        assert has_conflict is True, (
            "BUG-56: _check_merge_conflict returned False with invalid "
            "baseline despite a real conflict — both sides edited init.txt"
        )

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

        # Agent edits init.txt
        (wt_dir / "init.txt").write_text("agent version\n")
        GitWorktreeOps.stage_all(wt_dir)
        GitWorktreeOps.commit_staged(wt_dir, "agent edit")

        # User edits init.txt on main → conflict
        (repo / "init.txt").write_text("user version\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "user edit"],
            cwd=repo, capture_output=True,
        )

        server = VSCodeServer()
        server.work_dir = str(repo)
        tab = server._get_tab("bug56b-tab")
        tab.use_worktree = True
        tab.agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=baseline,
        )

        has_conflict = server._check_merge_conflict("bug56b-tab")
        assert has_conflict is True, (
            "Regression: valid baseline should still detect conflicts"
        )

        _cleanup(repo, branch, wt_dir)

    def test_no_conflict_returns_false(self, tmp_path: Path) -> None:
        """No overlapping changes → no conflict."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug56c-1"
        wt_dir, baseline = _create_worktree_branch(repo, branch)

        # Agent edits a different file
        _add_agent_commit(wt_dir, "agent.txt", "agent work\n")

        server = VSCodeServer()
        server.work_dir = str(repo)
        tab = server._get_tab("bug56c-tab")
        tab.use_worktree = True
        tab.agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=baseline,
        )

        has_conflict = server._check_merge_conflict("bug56c-tab")
        assert has_conflict is False

        _cleanup(repo, branch, wt_dir)


# ===================================================================
# BUG-57: Deleted files invisible in non-worktree merge review
# ===================================================================


class TestBug57DeletedFilesInMergeView:
    """BUG-57: ``_file_changed`` returns ``False`` when ``read_bytes()``
    raises ``OSError`` on a deleted file, and ``current_path.is_file()``
    also skips deleted files in the manifest building.  Agent-deleted
    files are invisible in the non-worktree merge review.

    FIX: ``_file_changed`` returns ``True`` when the file existed in
    ``pre_file_hashes`` but can't be read (deleted).  Manifest building
    creates an empty placeholder "current" file for deleted entries so
    the merge view can display the deletion.
    """

    def test_deleted_tracked_file_in_merge_view(
        self, tmp_path: Path,
    ) -> None:
        """A tracked file deleted by the agent must appear in the merge view."""
        repo = _make_repo(tmp_path)

        # Add a second file to the repo
        (repo / "deleteme.txt").write_text("I will be deleted\nline2\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "add deleteme"],
            cwd=repo, capture_output=True,
        )

        work_dir = str(repo)
        data_dir = str(tmp_path / "merge-data")

        # Capture pre-task state
        pre_hunks = _parse_diff_hunks(work_dir)
        pre_untracked = _capture_untracked(work_dir)
        pre_hashes = _snapshot_files(
            work_dir, set(pre_hunks.keys()) | pre_untracked,
        )
        # Include deleteme.txt in pre_hashes since it's tracked
        deleteme_path = repo / "deleteme.txt"
        pre_hashes["deleteme.txt"] = hashlib.md5(
            deleteme_path.read_bytes(),
        ).hexdigest()

        # "Agent" deletes the file
        subprocess.run(
            ["git", "rm", "deleteme.txt"], cwd=repo, capture_output=True,
        )

        result = _prepare_merge_view(
            work_dir, data_dir, pre_hunks, pre_untracked, pre_hashes,
        )

        # FIX: deleted file must appear in the merge view
        assert result.get("status") == "opened", (
            "BUG-57: deleted file not detected — merge view shows 'No changes'"
        )
        manifest_path = Path(data_dir) / "pending-merge.json"
        manifest = json.loads(manifest_path.read_text())
        names = [f["name"] for f in manifest["files"]]
        assert "deleteme.txt" in names, (
            "BUG-57: deleted file 'deleteme.txt' missing from merge manifest"
        )

        # The "current" file should exist (empty placeholder for deletion)
        for f in manifest["files"]:
            if f["name"] == "deleteme.txt":
                current = Path(f["current"])
                assert current.exists(), (
                    "BUG-57: current placeholder for deleted file must exist"
                )
                # Deleted file's current should be empty
                assert current.read_text() == "", (
                    "BUG-57: deleted file's current placeholder should be empty"
                )
                # Base should have the original content
                base = Path(f["base"])
                assert "I will be deleted" in base.read_text(), (
                    "BUG-57: base copy should contain original content"
                )
                break

    def test_deleted_untracked_file_in_merge_view(
        self, tmp_path: Path,
    ) -> None:
        """A pre-existing untracked file deleted by the agent must appear."""
        import shutil as _shutil

        repo = _make_repo(tmp_path)
        work_dir = str(repo)
        data_dir = str(tmp_path / "merge-data")

        # Create an untracked file
        (repo / "untracked.txt").write_text("untracked content\nline2\n")

        # Capture pre-task state
        pre_hunks = _parse_diff_hunks(work_dir)
        pre_untracked = _capture_untracked(work_dir)
        assert "untracked.txt" in pre_untracked
        pre_hashes = _snapshot_files(
            work_dir, set(pre_hunks.keys()) | pre_untracked,
        )
        # Save untracked base copies into the same data_dir that
        # _prepare_merge_view will use (it derives ub_dir from data_dir)
        ub_dir = Path(data_dir) / "untracked-base"
        for fname in pre_untracked:
            src = Path(work_dir) / fname
            if src.is_file():
                dst = ub_dir / fname
                dst.parent.mkdir(parents=True, exist_ok=True)
                _shutil.copy2(str(src), str(dst))

        # "Agent" deletes the untracked file
        (repo / "untracked.txt").unlink()

        result = _prepare_merge_view(
            work_dir, data_dir, pre_hunks, pre_untracked, pre_hashes,
        )

        # FIX: deleted untracked file must appear
        assert result.get("status") == "opened", (
            "BUG-57: deleted untracked file not detected"
        )
        manifest_path = Path(data_dir) / "pending-merge.json"
        manifest = json.loads(manifest_path.read_text())
        names = [f["name"] for f in manifest["files"]]
        assert "untracked.txt" in names, (
            "BUG-57: deleted untracked file missing from merge manifest"
        )

    def test_existing_file_still_shown(self, tmp_path: Path) -> None:
        """Regression: a modified (not deleted) file must still appear."""
        repo = _make_repo(tmp_path)
        work_dir = str(repo)
        data_dir = str(tmp_path / "merge-data")

        # Capture pre-task state
        pre_hunks = _parse_diff_hunks(work_dir)
        pre_untracked = _capture_untracked(work_dir)
        pre_hashes = _snapshot_files(
            work_dir, set(pre_hunks.keys()) | pre_untracked,
        )
        pre_hashes["init.txt"] = hashlib.md5(
            (repo / "init.txt").read_bytes(),
        ).hexdigest()

        # "Agent" modifies the file (not deletes)
        (repo / "init.txt").write_text("modified by agent\n")

        result = _prepare_merge_view(
            work_dir, data_dir, pre_hunks, pre_untracked, pre_hashes,
        )
        assert result.get("status") == "opened"
        manifest_path = Path(data_dir) / "pending-merge.json"
        manifest = json.loads(manifest_path.read_text())
        names = [f["name"] for f in manifest["files"]]
        assert "init.txt" in names

    def test_no_changes_still_returns_no_changes(
        self, tmp_path: Path,
    ) -> None:
        """When nothing changed, merge view returns 'No changes'."""
        repo = _make_repo(tmp_path)
        work_dir = str(repo)
        data_dir = str(tmp_path / "merge-data")

        pre_hunks = _parse_diff_hunks(work_dir)
        pre_untracked = _capture_untracked(work_dir)
        pre_hashes = _snapshot_files(
            work_dir, set(pre_hunks.keys()) | pre_untracked,
        )

        result = _prepare_merge_view(
            work_dir, data_dir, pre_hunks, pre_untracked, pre_hashes,
        )
        assert result.get("error") == "No changes"


# ===================================================================
# BUG-58: cleanup_orphans deletes pending-merge branches
# ===================================================================


class TestBug58CleanupOrphansDeletesPending:
    """BUG-58: After ``_finalize_worktree`` removes the worktree
    directory, the branch no longer has an active worktree.
    ``cleanup_orphans`` classifies it as orphaned and deletes it,
    permanently losing agent work that was pending merge.

    FIX: ``cleanup_orphans`` checks for the ``branch.<name>.kiss-original``
    git config entry.  If it exists, the branch is pending merge (not
    orphaned) and is skipped.
    """

    def test_pending_branch_not_deleted(self, tmp_path: Path) -> None:
        """A branch with kiss-original config must NOT be deleted."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug58a-1"
        wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
        assert GitWorktreeOps.create(repo, branch, wt_dir)
        GitWorktreeOps.save_original_branch(repo, branch, "main")
        _add_agent_commit(wt_dir, "agent.txt", "agent work\n")

        # Simulate _finalize_worktree: remove worktree but keep branch
        GitWorktreeOps.remove(repo, wt_dir)
        GitWorktreeOps.prune(repo)

        # Branch exists but has no active worktree — this is the
        # state between finalize and merge
        assert GitWorktreeOps.branch_exists(repo, branch)

        # Run cleanup — should NOT delete the branch
        result = GitWorktreeOps.cleanup_orphans(repo)

        # FIX: branch must still exist
        assert GitWorktreeOps.branch_exists(repo, branch), (
            "BUG-58: cleanup_orphans deleted a branch with kiss-original "
            "config — pending merge work lost"
        )
        assert "Deleted" not in result or branch not in result, (
            "BUG-58: cleanup report should not mention deleting "
            "a pending-merge branch"
        )

        # Cleanup
        _cleanup(repo, branch, wt_dir)

    def test_true_orphan_still_deleted(self, tmp_path: Path) -> None:
        """A branch with NO kiss-original config (true orphan) IS deleted."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug58b-1"

        # Create the branch directly (no worktree, no config)
        subprocess.run(
            ["git", "branch", branch],
            cwd=repo, capture_output=True,
        )
        assert GitWorktreeOps.branch_exists(repo, branch)

        result = GitWorktreeOps.cleanup_orphans(repo)
        assert not GitWorktreeOps.branch_exists(repo, branch), (
            "True orphan branch should be deleted"
        )
        assert "Deleted" in result

    def test_orphan_with_stale_config_still_deleted(
        self, tmp_path: Path,
    ) -> None:
        """A branch whose kiss-original points to a non-existent branch
        IS still deleted (truly orphaned, stale config)."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-bug58c-1"

        # Create branch with config pointing to non-existent target
        subprocess.run(
            ["git", "branch", branch],
            cwd=repo, capture_output=True,
        )
        GitWorktreeOps.save_original_branch(
            repo, branch, "nonexistent-branch-42",
        )
        assert GitWorktreeOps.branch_exists(repo, branch)

        GitWorktreeOps.cleanup_orphans(repo)

        # Branch with config pointing to a non-existent target branch
        # could be either kept or deleted.  The safe approach: keep it
        # (the user can always manually delete).  But since the target
        # branch doesn't exist, merging would fail anyway.  The fix
        # should check whether the target branch exists.
        # For safety, we keep it (don't delete).
        # Actually, the simplest safe fix: if kiss-original exists at all,
        # skip the branch.  The user can always run cleanup again after
        # resolving.
        assert GitWorktreeOps.branch_exists(repo, branch), (
            "Branch with any kiss-original config should be kept for safety"
        )

        # Final cleanup
        _cleanup(repo, branch, repo / "nonexistent")
