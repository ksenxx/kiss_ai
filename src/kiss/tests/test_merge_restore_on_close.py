"""Tests for restoring files to new-lines-only state when Sorcar closes during merge."""

import os
import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.code_server import (
    _capture_untracked,
    _parse_diff_hunks,
    _prepare_merge_view,
    _restore_merge_files,
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


class TestRestoreMergeFiles:
    """Verify _restore_merge_files restores files and cleans up."""

    def test_restore_handles_subdirectory_files(self) -> None:
        """Files in subdirectories should be restored correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            # Create a committed file in a subdirectory
            os.makedirs(os.path.join(repo, "sub"))
            Path(repo, "sub", "file.txt").write_text("original\n")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
            subprocess.run(["git", "commit", "-m", "add sub"], cwd=repo, capture_output=True)

            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(repo, set(pre_hunks.keys()) | pre_untracked)

            agent_content = "modified\n"
            Path(repo, "sub", "file.txt").write_text(agent_content)

            _prepare_merge_view(repo, data_dir, pre_hunks, pre_untracked, pre_hashes)

            # Simulate interleaved content
            Path(repo, "sub", "file.txt").write_text("original\nmodified\n")

            _restore_merge_files(data_dir, repo)

            assert Path(repo, "sub", "file.txt").read_text() == agent_content
