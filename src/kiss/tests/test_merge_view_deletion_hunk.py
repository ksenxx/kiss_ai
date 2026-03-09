"""Tests for correct hunk positions when cc=0 (pure deletion hunks).

When diff -U0 reports a hunk like @@ -3,3 +2,0 @@, cs=2 means the deletion
happened after line 2 in the current file. The merge view needs cs=2 (0-based
position 2, i.e. after line index 1) for correct placement, NOT cs-1=1.
"""

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
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=repo, capture_output=True
    )
    return repo


class TestDeletionHunkPosition:
    """Verify correct cs value when agent deletes lines (cc=0)."""

    def test_deletion_hunk_cs_position_tracked_file(self) -> None:
        """Deleting lines from a tracked file produces correct cs in merge view.

        File starts as 5 lines, agent deletes lines 3-4 (middle).
        diff -U0 output: @@ -3,2 +2,0 @@ → cs=2, cc=0.
        Merge view hunk should have cs=2 (not cs-1=1).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            # Commit a 5-line file
            Path(repo, "f.txt").write_text("A\nB\nC\nD\nE\n")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "init"], cwd=repo, capture_output=True
            )

            # Pre-task snapshot
            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(
                repo, set(pre_hunks.keys()) | pre_untracked
            )

            # Agent deletes lines 3-4 ("C" and "D")
            Path(repo, "f.txt").write_text("A\nB\nE\n")

            result = _prepare_merge_view(
                repo, data_dir, pre_hunks, pre_untracked, pre_hashes
            )
            assert result.get("status") == "opened"

            manifest = json.loads(
                (Path(data_dir) / "pending-merge.json").read_text()
            )
            hunks = manifest["files"][0]["hunks"]
            assert len(hunks) == 1
            h = hunks[0]
            # bs: base start (0-based), bc: base count (lines deleted)
            assert h["bc"] == 2  # "C" and "D" were in base
            assert h["cc"] == 0  # pure deletion
            # cs should be 2 (after line index 1 "B"), not 1
            assert h["cs"] == 2

    def test_deletion_at_end_tracked_file(self) -> None:
        """Deleting the last lines should have cs equal to remaining line count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            Path(repo, "f.txt").write_text("A\nB\nC\n")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "init"], cwd=repo, capture_output=True
            )

            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(
                repo, set(pre_hunks.keys()) | pre_untracked
            )

            # Agent deletes the last line "C"
            Path(repo, "f.txt").write_text("A\nB\n")

            result = _prepare_merge_view(
                repo, data_dir, pre_hunks, pre_untracked, pre_hashes
            )
            assert result.get("status") == "opened"

            manifest = json.loads(
                (Path(data_dir) / "pending-merge.json").read_text()
            )
            h = manifest["files"][0]["hunks"][0]
            assert h["bc"] == 1
            assert h["cc"] == 0
            # After deleting "C", current file has 2 lines. cs should be 2.
            assert h["cs"] == 2

    def test_deletion_at_beginning_tracked_file(self) -> None:
        """Deleting the first lines should have cs=0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            Path(repo, "f.txt").write_text("A\nB\nC\n")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "init"], cwd=repo, capture_output=True
            )

            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(
                repo, set(pre_hunks.keys()) | pre_untracked
            )

            # Agent deletes "A" (first line)
            Path(repo, "f.txt").write_text("B\nC\n")

            result = _prepare_merge_view(
                repo, data_dir, pre_hunks, pre_untracked, pre_hashes
            )
            assert result.get("status") == "opened"

            manifest = json.loads(
                (Path(data_dir) / "pending-merge.json").read_text()
            )
            h = manifest["files"][0]["hunks"][0]
            assert h["bc"] == 1
            assert h["cc"] == 0
            # diff -U0 for deleting first line: @@ -1,1 +0,0 @@ → cs=0
            assert h["cs"] == 0

    def test_deletion_hunk_untracked_file_with_saved_base(self) -> None:
        """Deletion from a pre-existing untracked file uses saved base correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            # Commit initial file
            Path(repo, "example.md").write_text("init\n")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "init"], cwd=repo, capture_output=True
            )

            # Create an untracked file before the task
            Path(repo, "notes.txt").write_text("L1\nL2\nL3\nL4\n")

            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(
                repo, set(pre_hunks.keys()) | pre_untracked
            )
            _save_untracked_base(
                repo, data_dir, pre_untracked | set(pre_hunks.keys())
            )

            # Agent deletes lines 2-3 from the untracked file
            Path(repo, "notes.txt").write_text("L1\nL4\n")

            result = _prepare_merge_view(
                repo, data_dir, pre_hunks, pre_untracked, pre_hashes
            )
            assert result.get("status") == "opened"

            manifest = json.loads(
                (Path(data_dir) / "pending-merge.json").read_text()
            )
            notes_files = [
                f for f in manifest["files"] if f["name"] == "notes.txt"
            ]
            assert len(notes_files) == 1
            h = notes_files[0]["hunks"][0]
            assert h["bc"] == 2  # L2, L3 deleted
            assert h["cc"] == 0
            # Deletion after L1 → cs should be 1
            assert h["cs"] == 1

    def test_modification_hunk_cs_still_zero_based(self) -> None:
        """For cc > 0 (modification, not pure deletion), cs should be cs-1 (0-based)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            Path(repo, "f.txt").write_text("A\nB\nC\n")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "init"], cwd=repo, capture_output=True
            )

            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(
                repo, set(pre_hunks.keys()) | pre_untracked
            )

            # Agent modifies line 2 ("B" → "X")
            Path(repo, "f.txt").write_text("A\nX\nC\n")

            result = _prepare_merge_view(
                repo, data_dir, pre_hunks, pre_untracked, pre_hashes
            )
            assert result.get("status") == "opened"

            manifest = json.loads(
                (Path(data_dir) / "pending-merge.json").read_text()
            )
            h = manifest["files"][0]["hunks"][0]
            assert h["bc"] == 1
            assert h["cc"] == 1
            # diff -U0: @@ -2,1 +2,1 @@ → cs=2, cc=1 → 0-based cs=1
            assert h["cs"] == 1

    def test_addition_hunk_cs_zero_based(self) -> None:
        """For bc=0 (pure addition), cs should be cs-1 (0-based)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            Path(repo, "f.txt").write_text("A\nB\n")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "init"], cwd=repo, capture_output=True
            )

            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(
                repo, set(pre_hunks.keys()) | pre_untracked
            )

            # Agent adds "NEW" between A and B
            Path(repo, "f.txt").write_text("A\nNEW\nB\n")

            result = _prepare_merge_view(
                repo, data_dir, pre_hunks, pre_untracked, pre_hashes
            )
            assert result.get("status") == "opened"

            manifest = json.loads(
                (Path(data_dir) / "pending-merge.json").read_text()
            )
            h = manifest["files"][0]["hunks"][0]
            assert h["bc"] == 0
            assert h["cc"] == 1
            # diff -U0: @@ -1,0 +2,1 @@ → cs=2, cc=1 → 0-based cs=1
            assert h["cs"] == 1

    def test_mixed_deletion_and_modification_hunks(self) -> None:
        """File with both deletion and modification hunks gets correct cs values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            Path(repo, "f.txt").write_text("A\nB\nC\nD\nE\nF\n")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "init"], cwd=repo, capture_output=True
            )

            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(
                repo, set(pre_hunks.keys()) | pre_untracked
            )

            # Agent: modify B→X and delete E
            Path(repo, "f.txt").write_text("A\nX\nC\nD\nF\n")

            result = _prepare_merge_view(
                repo, data_dir, pre_hunks, pre_untracked, pre_hashes
            )
            assert result.get("status") == "opened"

            manifest = json.loads(
                (Path(data_dir) / "pending-merge.json").read_text()
            )
            hunks = manifest["files"][0]["hunks"]
            assert len(hunks) == 2

            # First hunk: modification B→X
            h0 = hunks[0]
            assert h0["bc"] == 1 and h0["cc"] == 1
            assert h0["cs"] == 1  # 0-based

            # Second hunk: deletion of E
            h1 = hunks[1]
            assert h1["bc"] == 1 and h1["cc"] == 0
            # After deleting E, the current file is A,X,C,D,F
            # diff says: @@ -5,1 +4,0 @@ → cs=4
            assert h1["cs"] == 4  # kept as-is for cc=0

    def test_delete_all_lines_tracked_file(self) -> None:
        """Deleting all lines results in a single hunk with cs=0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

            Path(repo, "f.txt").write_text("A\nB\nC\n")
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "init"], cwd=repo, capture_output=True
            )

            pre_hunks = _parse_diff_hunks(repo)
            pre_untracked = _capture_untracked(repo)
            pre_hashes = _snapshot_files(
                repo, set(pre_hunks.keys()) | pre_untracked
            )

            # Agent deletes all lines
            Path(repo, "f.txt").write_text("")

            result = _prepare_merge_view(
                repo, data_dir, pre_hunks, pre_untracked, pre_hashes
            )
            assert result.get("status") == "opened"

            manifest = json.loads(
                (Path(data_dir) / "pending-merge.json").read_text()
            )
            h = manifest["files"][0]["hunks"][0]
            assert h["bc"] == 3
            assert h["cc"] == 0
            assert h["cs"] == 0
