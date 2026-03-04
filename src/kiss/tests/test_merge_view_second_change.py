"""Tests for merge view showing up on second file change after accepting first."""

import os
import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.code_server import (
    _capture_untracked,
    _parse_diff_hunks,
    _prepare_merge_view,
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


class TestMergeViewSecondChange:
    """Reproduce bug: merge view not showing after accepting first change."""

    def test_second_change_same_lines_detected(self) -> None:
        """After first change is accepted (not committed), a second change
        to the same file and same lines must still produce a merge view."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            # --- Simulate first agent run ---
            # Capture pre-state
            pre_hunks_1 = _parse_diff_hunks(repo)
            pre_untracked_1 = _capture_untracked(repo)
            pre_hashes_1 = _snapshot_files(repo, set(pre_hunks_1.keys()))
            assert pre_hunks_1 == {}  # No changes yet

            # Agent modifies the file
            Path(repo, "example.md").write_text("line 1\nMODIFIED line 2\nline 3\n")

            # Prepare merge view (first time)
            result1 = _prepare_merge_view(
                repo, data_dir, pre_hunks_1, pre_untracked_1, pre_hashes_1
            )
            assert result1.get("status") == "opened"
            assert result1.get("count") == 1

            # User "accepts" the change (file keeps agent's version, no git commit)
            # The file on disk already has the agent's content.

            # --- Simulate second agent run ---
            # Capture pre-state (file is still modified from first run)
            pre_hunks_2 = _parse_diff_hunks(repo)
            pre_untracked_2 = _capture_untracked(repo)
            pre_hashes_2 = _snapshot_files(repo, set(pre_hunks_2.keys()))
            assert len(pre_hunks_2) > 0  # File shows as modified vs HEAD

            # Agent modifies the same lines again
            Path(repo, "example.md").write_text("line 1\nRE-MODIFIED line 2\nline 3\n")

            # Prepare merge view (second time) -- THIS WAS THE BUG
            result2 = _prepare_merge_view(
                repo, data_dir, pre_hunks_2, pre_untracked_2, pre_hashes_2
            )
            # With the fix, merge view should appear
            assert result2.get("status") == "opened", (
                f"Merge view should appear on second change but got: {result2}"
            )
            assert result2.get("count") == 1

    def test_unchanged_file_not_shown(self) -> None:
        """If the agent does NOT change a previously modified file,
        no merge view should appear for it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            # Modify file (simulating accepted first change)
            Path(repo, "example.md").write_text("line 1\nMODIFIED line 2\nline 3\n")

            # Capture pre-state
            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(repo, set(pre_hunks.keys()))

            # Agent does NOT change the file

            # Prepare merge view
            result = _prepare_merge_view(
                repo, data_dir, pre_hunks, pre_untracked, pre_hashes
            )
            assert result.get("error") == "No changes"

    def test_new_file_still_works(self) -> None:
        """Creating a new (untracked) file should still show in merge view."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(repo, set(pre_hunks.keys()))

            # Agent creates a new file
            Path(repo, "new_file.py").write_text("print('hello')\n")

            result = _prepare_merge_view(
                repo, data_dir, pre_hunks, pre_untracked, pre_hashes
            )
            assert result.get("status") == "opened"
            assert result.get("count") == 1


class TestSnapshotFiles:
    """Tests for _snapshot_files helper."""

    def test_snapshot_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _snapshot_files(tmpdir, {"nonexistent.txt"})
            assert result == {}


class TestBackwardCompatibility:
    """Verify that the old behavior still works when pre_file_hashes is None."""

    def test_without_pre_file_hashes(self) -> None:
        """When pre_file_hashes is None, the old (bs, bc) filtering should still work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            # Pre-existing change
            Path(repo, "example.md").write_text("line 1\nMODIFIED line 2\nline 3\n")
            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)

            # No new changes by agent — old filtering should skip
            result = _prepare_merge_view(
                repo, data_dir, pre_hunks, pre_untracked
            )
            assert result.get("error") == "No changes"
