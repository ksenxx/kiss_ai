"""Tests for restoring files to new-lines-only state when Sorcar closes during merge."""

import json
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

            # Simulate interleaved content (as if extension inserted old lines)
            Path(repo, "sub", "file.txt").write_text("original\nmodified\n")

            hunk_count = _restore_merge_files(data_dir, repo)

            # File restored to agent's version
            assert Path(repo, "sub", "file.txt").read_text() == agent_content
            # Returns hunk count and regenerates pending-merge.json
            assert hunk_count >= 1
            manifest = Path(data_dir) / "pending-merge.json"
            assert manifest.exists()
            data = json.loads(manifest.read_text())
            assert len(data["files"]) == 1
            assert data["files"][0]["name"] == str(Path("sub", "file.txt"))

    def test_restore_no_merge_current_returns_zero(self) -> None:
        """No merge-current dir means nothing to restore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert _restore_merge_files(tmpdir, tmpdir) == 0

    def test_restore_preserves_unresolved_hunks_after_partial_accept(self) -> None:
        """Crash after partial accept: restores agent version, regenerates manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            # Agent adds two distinct changes: modify line 1 and add line 4
            Path(repo, "example.md").write_text("MODIFIED 1\nline 2\nline 3\nnew line 4\n")

            pre_hunks: dict[str, list[tuple[int, int, int, int]]] = {}
            pre_untracked: set[str] = set()
            result = _prepare_merge_view(repo, data_dir, pre_hunks, pre_untracked)
            assert result.get("hunk_count", 0) >= 2, f"expected >=2 hunks: {result}"

            # Simulate the extension interleaving old lines into the file,
            # then the user accepting one hunk (deleting old lines for it).
            # The working directory is now in a mixed state.
            Path(repo, "example.md").write_text(
                "line 1\nMODIFIED 1\nline 2\nline 3\nnew line 4\n"
            )

            # Crash! _restore_merge_files is called
            hunk_count = _restore_merge_files(data_dir, repo)

            # File restored to agent's full version
            assert Path(repo, "example.md").read_text() == (
                "MODIFIED 1\nline 2\nline 3\nnew line 4\n"
            )
            # Manifest regenerated with all hunks for re-review
            assert hunk_count >= 2
            manifest = json.loads(Path(data_dir, "pending-merge.json").read_text())
            total = sum(len(f["hunks"]) for f in manifest["files"])
            assert total >= 2

    def test_restore_new_untracked_file(self) -> None:
        """Crash recovery handles new files (no base in merge-temp)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            # Agent creates a brand new file
            Path(repo, "newfile.txt").write_text("hello\nworld\n")

            pre_hunks: dict[str, list[tuple[int, int, int, int]]] = {}
            pre_untracked: set[str] = set()
            result = _prepare_merge_view(repo, data_dir, pre_hunks, pre_untracked)
            assert result.get("hunk_count", 0) >= 1

            # Simulate crash
            hunk_count = _restore_merge_files(data_dir, repo)

            assert Path(repo, "newfile.txt").read_text() == "hello\nworld\n"
            assert hunk_count >= 1
            manifest = json.loads(Path(data_dir, "pending-merge.json").read_text())
            assert len(manifest["files"]) >= 1
