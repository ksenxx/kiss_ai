"""Tests for merge view: deleted file exclusion and pre-existing diff exclusion."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from kiss.agents.vscode.diff_merge import (
    _capture_untracked,
    _cleanup_merge_data,
    _parse_diff_hunks,
    _prepare_merge_view,
    _save_untracked_base,
    _snapshot_files,
)


def _create_git_repo(tmpdir: str) -> str:
    repo = os.path.join(tmpdir, "repo")
    os.makedirs(repo)
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True)
    Path(repo, "example.md").write_text("line 1\nline 2\nline 3\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True)
    return repo


class TestMergeViewDeletedFiles:
    def test_deleted_file_excluded_from_merge(self) -> None:
        """When the agent deletes a tracked file, the merge view should not include it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)

            # Agent deletes the file
            os.remove(os.path.join(repo, "example.md"))

            result = _prepare_merge_view(repo, data_dir, pre_hunks, pre_untracked)
            assert result == {"error": "No changes"}


class TestMergeViewExcludesPreExistingDiffs:
    def test_pre_existing_diff_excluded_on_second_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            pre_hunks_1 = _parse_diff_hunks(repo)
            pre_untracked_1 = _capture_untracked(repo)
            pre_hashes_1 = _snapshot_files(
                repo, set(pre_hunks_1.keys()) | pre_untracked_1
            )
            _save_untracked_base(
                repo, pre_untracked_1 | set(pre_hunks_1.keys())
            )
            Path(repo, "example.md").write_text("line 1\nMODIFIED line 2\nline 3\n")
            result1 = _prepare_merge_view(
                repo, data_dir, pre_hunks_1, pre_untracked_1, pre_hashes_1
            )
            assert result1.get("status") == "opened"

            pre_hunks_2 = _parse_diff_hunks(repo)
            pre_untracked_2 = _capture_untracked(repo)
            pre_hashes_2 = _snapshot_files(
                repo, set(pre_hunks_2.keys()) | pre_untracked_2
            )
            _save_untracked_base(
                repo, pre_untracked_2 | set(pre_hunks_2.keys())
            )
            Path(repo, "example.md").write_text(
                "line 1\nMODIFIED line 2\nline 3\nline 4\n"
            )
            result2 = _prepare_merge_view(
                repo, data_dir, pre_hunks_2, pre_untracked_2, pre_hashes_2
            )
            assert result2.get("status") == "opened"
            merge_file = Path(data_dir) / "pending-merge.json"
            manifest = json.loads(merge_file.read_text())
            hunks = manifest["files"][0]["hunks"]
            assert len(hunks) == 1
            h = hunks[0]
            assert h["bc"] == 0
            assert h["cc"] == 1


class TestCleanupMergeData:

    def test_noop_when_dir_missing(self) -> None:
        """_cleanup_merge_data is a no-op when directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "nonexistent")
            _cleanup_merge_data(data_dir)  # should not raise
            assert not os.path.exists(data_dir)
