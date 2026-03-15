"""Tests for merge view showing up on second file change after accepting first."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.code_server import (
    _capture_untracked,
    _parse_diff_hunks,
    _prepare_merge_view,
    _save_untracked_base,
    _snapshot_files,
)


def _create_git_repo(tmpdir: str) -> str:
    """Create a temp git repo with one committed file and return repo path."""
    repo = os.path.join(tmpdir, "repo")
    os.makedirs(repo)
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True)
    # Create and commit a file
    Path(repo, "example.md").write_text("line 1\nline 2\nline 3\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True)
    return repo

class TestMergeViewExcludesPreExistingDiffs:
    """Verify merge view only shows diffs from the current task, not pre-existing ones."""

    def test_pre_existing_diff_excluded_on_second_task(self) -> None:
        """If task 1 modifies lines 2 and task 2 modifies line 4,
        merge view after task 2 should only show the line 4 change."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            # --- Task 1: modify line 2 ---
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

            # User accepts task 1's changes (no git commit)

            # --- Task 2: add line 4, leave line 2 as-is ---
            pre_hunks_2 = _parse_diff_hunks(repo)
            pre_untracked_2 = _capture_untracked(repo)
            pre_hashes_2 = _snapshot_files(
                repo, set(pre_hunks_2.keys()) | pre_untracked_2
            )
            _save_untracked_base(
                repo, pre_untracked_2 | set(pre_hunks_2.keys())
            )
            # Agent adds a new line but doesn't touch line 2
            Path(repo, "example.md").write_text(
                "line 1\nMODIFIED line 2\nline 3\nline 4\n"
            )
            result2 = _prepare_merge_view(
                repo, data_dir, pre_hunks_2, pre_untracked_2, pre_hashes_2
            )
            assert result2.get("status") == "opened"
            # Only the line 4 addition should show, not the line 2 change
            merge_file = Path(data_dir) / "pending-merge.json"
            manifest = json.loads(merge_file.read_text())
            hunks = manifest["files"][0]["hunks"]
            # Should have exactly 1 hunk for the added line 4
            assert len(hunks) == 1
            h = hunks[0]
            # The hunk should be an addition (bc=0) at the end
            assert h["bc"] == 0
            assert h["cc"] == 1

