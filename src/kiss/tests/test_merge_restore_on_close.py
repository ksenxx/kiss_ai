"""Tests for restoring files to new-lines-only state when Sorcar closes during merge."""

import os
import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.code_server import (
    _capture_untracked,
    _cleanup_merge_data,
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


class TestMergeCurrentSaved:
    """Verify _prepare_merge_view saves current file copies to merge-current/."""

    def test_merge_current_dir_created(self) -> None:
        """After _prepare_merge_view, merge-current/ should exist with file copies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(repo, set(pre_hunks.keys()) | pre_untracked)

            Path(repo, "example.md").write_text("line 1\nMODIFIED\nline 3\n")

            _prepare_merge_view(repo, data_dir, pre_hunks, pre_untracked, pre_hashes)

            current_dir = Path(data_dir) / "merge-current"
            assert current_dir.is_dir()
            saved = current_dir / "example.md"
            assert saved.is_file()
            assert saved.read_text() == "line 1\nMODIFIED\nline 3\n"

    def test_merge_current_has_new_file(self) -> None:
        """Newly created files should also be saved in merge-current/."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(repo, set(pre_hunks.keys()) | pre_untracked)

            Path(repo, "new_file.txt").write_text("new content\n")

            result = _prepare_merge_view(repo, data_dir, pre_hunks, pre_untracked, pre_hashes)
            assert result.get("status") == "opened"

            saved = Path(data_dir) / "merge-current" / "new_file.txt"
            assert saved.is_file()
            assert saved.read_text() == "new content\n"

    def test_merge_current_multiple_files(self) -> None:
        """Multiple changed files should all be saved in merge-current/."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(repo, set(pre_hunks.keys()) | pre_untracked)

            Path(repo, "example.md").write_text("CHANGED\n")
            Path(repo, "another.txt").write_text("hello\n")

            result = _prepare_merge_view(repo, data_dir, pre_hunks, pre_untracked, pre_hashes)
            assert result.get("status") == "opened"

            current_dir = Path(data_dir) / "merge-current"
            assert (current_dir / "example.md").read_text() == "CHANGED\n"
            assert (current_dir / "another.txt").read_text() == "hello\n"


class TestRestoreMergeFiles:
    """Verify _restore_merge_files restores files and cleans up."""

    def test_restore_overwrites_interleaved_file(self) -> None:
        """Simulates the extension inserting old lines, then restore reverts to new-only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(repo, set(pre_hunks.keys()) | pre_untracked)

            # Agent modifies file
            agent_content = "line 1\nMODIFIED line 2\nline 3\n"
            Path(repo, "example.md").write_text(agent_content)

            _prepare_merge_view(repo, data_dir, pre_hunks, pre_untracked, pre_hashes)

            # Simulate the VS Code extension inserting old lines (interleaved state)
            Path(repo, "example.md").write_text(
                "line 1\nline 2\nMODIFIED line 2\nline 3\n"
            )

            # Restore should bring back the new-lines-only version
            _restore_merge_files(data_dir, repo)

            assert Path(repo, "example.md").read_text() == agent_content

    def test_restore_cleans_up_merge_temp(self) -> None:
        """After restore, merge-temp/ should be removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(repo, set(pre_hunks.keys()) | pre_untracked)

            Path(repo, "example.md").write_text("CHANGED\n")
            _prepare_merge_view(repo, data_dir, pre_hunks, pre_untracked, pre_hashes)

            assert (Path(data_dir) / "merge-temp").is_dir()
            assert (Path(data_dir) / "merge-current").is_dir()
            assert (Path(data_dir) / "pending-merge.json").is_file()

            _restore_merge_files(data_dir, repo)

            assert not (Path(data_dir) / "merge-temp").exists()
            assert not (Path(data_dir) / "merge-current").exists()
            assert not (Path(data_dir) / "pending-merge.json").exists()

    def test_restore_noop_when_no_merge_current(self) -> None:
        """If merge-current/ doesn't exist, _restore_merge_files is a no-op."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            original = Path(repo, "example.md").read_text()

            # No merge-current exists, so restore should not modify files
            _restore_merge_files(data_dir, repo)

            assert Path(repo, "example.md").read_text() == original

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


class TestCleanupMergeDataIncludesMergeCurrent:
    """Verify _cleanup_merge_data also removes merge-current/."""

    def test_cleanup_removes_merge_current(self) -> None:
        """Normal merge completion via _cleanup_merge_data should remove merge-current/."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(repo, set(pre_hunks.keys()) | pre_untracked)

            Path(repo, "example.md").write_text("CHANGED\n")
            _prepare_merge_view(repo, data_dir, pre_hunks, pre_untracked, pre_hashes)

            assert (Path(data_dir) / "merge-current").is_dir()

            # This is what happens on normal all-done
            _cleanup_merge_data(data_dir)

            assert not (Path(data_dir) / "merge-current").exists()
            assert not (Path(data_dir) / "merge-temp").exists()
            assert not (Path(data_dir) / "pending-merge.json").exists()

    def test_cleanup_removes_pending_merge_json(self) -> None:
        """_cleanup_merge_data should remove pending-merge.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            manifest = Path(data_dir) / "pending-merge.json"
            manifest.write_text("{}")

            _cleanup_merge_data(data_dir)

            assert not manifest.exists()


class TestCrashRecoveryScenario:
    """End-to-end test simulating a crash during merge and recovery on restart."""

    def test_full_crash_recovery(self) -> None:
        """Simulate: agent changes → merge prepared → crash → restart restores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            # Agent task happens
            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(repo, set(pre_hunks.keys()) | pre_untracked)

            agent_content = "line 1\nAGENT CHANGE\nline 3\n"
            Path(repo, "example.md").write_text(agent_content)

            _prepare_merge_view(repo, data_dir, pre_hunks, pre_untracked, pre_hashes)

            # Extension opens merge, physically inserts old lines
            Path(repo, "example.md").write_text(
                "line 1\nline 2\nAGENT CHANGE\nline 3\n"
            )

            # Server crashes! (no cleanup runs)
            # ... some time passes ...

            # Server restarts: _restore_merge_files is called at startup
            _restore_merge_files(data_dir, repo)

            # File should have only the agent's new lines
            assert Path(repo, "example.md").read_text() == agent_content

            # All merge data should be cleaned up
            assert not (Path(data_dir) / "merge-temp").exists()
            assert not (Path(data_dir) / "merge-current").exists()
            assert not (Path(data_dir) / "pending-merge.json").exists()

    def test_no_stale_state_on_clean_startup(self) -> None:
        """If no merge was in progress, startup restore is a no-op."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            original = Path(repo, "example.md").read_text()

            # No merge-current exists, startup restore is a no-op
            _restore_merge_files(data_dir, repo)

            assert Path(repo, "example.md").read_text() == original
