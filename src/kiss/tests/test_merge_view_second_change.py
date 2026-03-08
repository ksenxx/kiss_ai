"""Tests for merge view showing up on second file change after accepting first."""

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
    _untracked_base_dir,
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

class TestModifiedUntrackedFile:
    """Tests for merge view detecting modifications to pre-existing untracked files."""

    def test_save_untracked_base_skips_large_files(self) -> None:
        """Files > 2MB should not be saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            Path(repo, "big.bin").write_bytes(b"x" * 3_000_000)
            _save_untracked_base(repo, data_dir, {"big.bin"})
            assert not (_untracked_base_dir() / "big.bin").exists()

