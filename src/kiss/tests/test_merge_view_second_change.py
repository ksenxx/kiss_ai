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
            pre_hashes_2 = _snapshot_files(
                repo, set(pre_hunks_2.keys()) | pre_untracked_2
            )
            assert len(pre_hunks_2) > 0  # File shows as modified vs HEAD
            # Save pre-task copies (tracked files with diffs + untracked)
            _save_untracked_base(
                repo, data_dir, pre_untracked_2 | set(pre_hunks_2.keys())
            )

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
                repo, data_dir, pre_untracked_1 | set(pre_hunks_1.keys())
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
                repo, data_dir, pre_untracked_2 | set(pre_hunks_2.keys())
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
            import json
            manifest = json.loads(merge_file.read_text())
            hunks = manifest["files"][0]["hunks"]
            # Should have exactly 1 hunk for the added line 4
            assert len(hunks) == 1
            h = hunks[0]
            # The hunk should be an addition (bc=0) at the end
            assert h["bc"] == 0
            assert h["cc"] == 1

    def test_pre_existing_diff_not_shown_when_file_unchanged(self) -> None:
        """If a file has pre-existing diffs and the agent doesn't touch it,
        the merge view should not include it at all."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            # Manually modify the file (simulating a previous task's accepted change)
            Path(repo, "example.md").write_text("line 1\nCHANGED\nline 3\n")

            # --- New task starts ---
            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(
                repo, set(pre_hunks.keys()) | pre_untracked
            )
            _save_untracked_base(
                repo, data_dir, pre_untracked | set(pre_hunks.keys())
            )
            assert len(pre_hunks) > 0  # There are pre-existing diffs

            # Agent does NOT modify the file — create a new file instead
            Path(repo, "new.txt").write_text("hello\n")
            subprocess.run(["git", "add", "new.txt"], cwd=repo, capture_output=True)

            result = _prepare_merge_view(
                repo, data_dir, pre_hunks, pre_untracked, pre_hashes
            )
            # Should not include example.md (unchanged by agent)
            if result.get("status") == "opened":
                import json
                manifest = json.loads(
                    (Path(data_dir) / "pending-merge.json").read_text()
                )
                file_names = [f["name"] for f in manifest["files"]]
                assert "example.md" not in file_names

    def test_untracked_file_modified_shows_only_agent_changes(self) -> None:
        """When an untracked file existed before the task and the agent
        modifies it, only the agent's changes should appear."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            # Create an untracked file before the task
            Path(repo, "notes.txt").write_text("note 1\nnote 2\nnote 3\n")

            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(
                repo, set(pre_hunks.keys()) | pre_untracked
            )
            _save_untracked_base(
                repo, data_dir, pre_untracked | set(pre_hunks.keys())
            )
            assert "notes.txt" in pre_untracked

            # Agent modifies only line 2 of the untracked file
            Path(repo, "notes.txt").write_text("note 1\nMODIFIED note 2\nnote 3\n")

            result = _prepare_merge_view(
                repo, data_dir, pre_hunks, pre_untracked, pre_hashes
            )
            assert result.get("status") == "opened"
            import json
            manifest = json.loads(
                (Path(data_dir) / "pending-merge.json").read_text()
            )
            notes_files = [f for f in manifest["files"] if f["name"] == "notes.txt"]
            assert len(notes_files) == 1
            hunks = notes_files[0]["hunks"]
            # Should show only the modified line, not the entire file as new
            assert len(hunks) == 1
            assert hunks[0]["bc"] == 1  # 1 old line
            assert hunks[0]["cc"] == 1  # 1 new line


class TestDiffFiles:
    """Tests for _diff_files helper."""

    def test_diff_identical_files(self) -> None:
        """Identical files produce no hunks."""
        from kiss.agents.sorcar.code_server import _diff_files

        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "a.txt")
            f2 = os.path.join(tmpdir, "b.txt")
            Path(f1).write_text("hello\nworld\n")
            Path(f2).write_text("hello\nworld\n")
            assert _diff_files(f1, f2) == []

    def test_diff_one_line_changed(self) -> None:
        """Changing one line produces one hunk."""
        from kiss.agents.sorcar.code_server import _diff_files

        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "a.txt")
            f2 = os.path.join(tmpdir, "b.txt")
            Path(f1).write_text("hello\nworld\n")
            Path(f2).write_text("hello\nEARTH\n")
            hunks = _diff_files(f1, f2)
            assert len(hunks) == 1
            bs, bc, cs, cc = hunks[0]
            assert bc == 1
            assert cc == 1

    def test_diff_line_added(self) -> None:
        """Adding a line produces a hunk with bc=0."""
        from kiss.agents.sorcar.code_server import _diff_files

        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "a.txt")
            f2 = os.path.join(tmpdir, "b.txt")
            Path(f1).write_text("hello\nworld\n")
            Path(f2).write_text("hello\nworld\nnew line\n")
            hunks = _diff_files(f1, f2)
            assert len(hunks) == 1
            bs, bc, cs, cc = hunks[0]
            assert bc == 0
            assert cc == 1

    def test_diff_line_removed(self) -> None:
        """Removing a line produces a hunk with cc=0."""
        from kiss.agents.sorcar.code_server import _diff_files

        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "a.txt")
            f2 = os.path.join(tmpdir, "b.txt")
            Path(f1).write_text("hello\nworld\ngoodbye\n")
            Path(f2).write_text("hello\ngoodbye\n")
            hunks = _diff_files(f1, f2)
            assert len(hunks) == 1
            bs, bc, cs, cc = hunks[0]
            assert bc == 1
            assert cc == 0


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

