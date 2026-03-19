"""Tests for merge view: restore on close, second-change exclusion of pre-existing diffs."""

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


class TestRestoreMergeFiles:
    def test_restore_handles_subdirectory_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = _create_git_repo(tmpdir)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)

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

            Path(repo, "sub", "file.txt").write_text("original\nmodified\n")

            hunk_count = _restore_merge_files(data_dir, repo)

            assert Path(repo, "sub", "file.txt").read_text() == agent_content
            assert hunk_count >= 1
            manifest = Path(data_dir) / "pending-merge.json"
            assert manifest.exists()
            data = json.loads(manifest.read_text())
            assert len(data["files"]) == 1
            assert data["files"][0]["name"] == str(Path("sub", "file.txt"))


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
