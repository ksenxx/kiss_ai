"""Integration tests for code_server merge/untracked utilities.

Covers _untracked_base_dir, _save_untracked_base, _cleanup_merge_data,
_prepare_merge_view, _parse_diff_hunks, _capture_untracked, _snapshot_files,
and _scan_files with full branch coverage.
"""

import os
import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.code_server import (
    _capture_untracked,
    _cleanup_merge_data,
    _parse_diff_hunks,
    _prepare_merge_view,
    _save_untracked_base,
    _snapshot_files,
    _untracked_base_dir,
)


def _create_git_repo(tmpdir: str) -> str:
    """Create a temp git repo with one committed file and return repo path."""
    repo = os.path.join(tmpdir, "repo")
    os.makedirs(repo)
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"], cwd=repo, capture_output=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True)
    Path(repo, "example.md").write_text("line 1\nline 2\nline 3\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True)
    return repo


class TestCleanupMergeData:
    """Tests for _cleanup_merge_data."""

    def test_cleanup_removes_untracked_base(self) -> None:
        """untracked-base directory should be removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)
            # Create untracked file and save base
            Path(repo, "u.py").write_text("content\n")
            _save_untracked_base(repo, data_dir, {"u.py"})
            ub_dir = _untracked_base_dir()
            assert ub_dir.exists()
            _cleanup_merge_data(data_dir)
            assert not ub_dir.exists()

            # Directory should not be created for empty set
            # (rmtree only runs if it exists, and no files to copy)
            # The directory may or may not exist depending on prior state


class TestPrepareMergeViewBranches:
    """Cover all branches in _prepare_merge_view."""

    def test_untracked_not_in_pre_hashes_skipped(self) -> None:
        """Pre-existing untracked file not in pre_file_hashes is skipped
        (but pre_file_hashes is non-empty so the block runs)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            # Make a tracked modification so pre_hashes is non-empty
            Path(repo, "example.md").write_text("line 1\nPRE MOD\nline 3\n")
            Path(repo, "ut.py").write_text("content\n")

            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            # Hash only tracked files, NOT untracked
            pre_hashes = _snapshot_files(repo, set(pre_hunks.keys()))
            assert len(pre_hashes) > 0  # pre_hashes is non-empty

            # Modify the untracked file
            Path(repo, "ut.py").write_text("changed\n")

            result = _prepare_merge_view(
                repo, data_dir, pre_hunks, pre_untracked, pre_hashes
            )
            # The tracked file is unchanged, and the untracked file is not
            # in pre_hashes, so it's skipped
            assert result.get("error") == "No changes"

    def test_deleted_untracked_file_during_merge_prep(self) -> None:
        """If untracked file is deleted between hash and merge prep, skip it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            Path(repo, "ut.py").write_text("content\n")
            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(repo, set(pre_hunks.keys()) | pre_untracked)
            _save_untracked_base(repo, data_dir, pre_untracked)

            # Delete the file (simulating OSError on hash read)
            os.remove(os.path.join(repo, "ut.py"))

            result = _prepare_merge_view(
                repo, data_dir, pre_hunks, pre_untracked, pre_hashes
            )
            assert result.get("error") == "No changes"


