# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: a deleted tracked binary file must appear in the merge view.

Reproduces a real bug in ``diff_merge._prepare_merge_view``: when the
agent deletes a tracked BINARY file, ``git diff -U0`` emits
``Binary files a/<f> and /dev/null differ`` (an empty hunk list).  The
binary-detection branch required ``fpath.is_file()`` — false for a
deleted file — so the deletion fell through every branch and was
silently dropped from the manifest.  Deleted TEXT files are shown (via
a ``.deleted`` placeholder), so the user could never review — nor
reject/restore — a binary deletion, while the worktree changed-files
list still reported it.  Inconsistent and lossy.

Also verifies end-to-end that rejecting the manifest entry through the
web server's reject path restores the original bytes on disk.

No mocks, patches, fakes, or test doubles.  Real git repository.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from kiss.agents.vscode.diff_merge import _prepare_merge_view
from kiss.agents.vscode.web_server import _reject_all_hunks_in_file

BINARY_BYTES = b"\x00\x01\x02PNGish\xff\xfe binary payload \x00"


def _git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


class TestDeletedBinaryMergeView(unittest.TestCase):
    """Deleting a tracked binary file must be reviewable in the merge view."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-delbin-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True)
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.email", "test@example.com")
        _git(self.repo, "config", "user.name", "Test User")
        _git(self.repo, "config", "commit.gpgsign", "false")
        Path(self.repo, "img.bin").write_bytes(BINARY_BYTES)
        Path(self.repo, "keep.txt").write_text("hello\n")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "initial")
        self.data_dir = str(Path(self.tmpdir) / "merge-data")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_deleted_binary_in_manifest_and_restorable(self) -> None:
        """The deletion must be listed, and rejecting it must restore bytes."""
        Path(self.repo, "img.bin").unlink()

        result = _prepare_merge_view(self.repo, self.data_dir, {}, set())
        self.assertEqual(
            result.get("status"), "opened",
            f"deleted binary produced no merge view: {result}",
        )
        manifest = json.loads(
            (Path(self.data_dir) / "pending-merge.json").read_text(),
        )
        entries = {f["name"]: f for f in manifest["files"]}
        self.assertIn(
            "img.bin", entries,
            f"deleted binary missing from manifest: {sorted(entries)}",
        )
        entry = entries["img.bin"]
        self.assertTrue(entry.get("binary"), f"not flagged binary: {entry}")
        self.assertEqual(
            entry["target"], str(Path(self.repo) / "img.bin"),
            "target must be the real workspace path so a reject "
            f"restores the file there: {entry}",
        )

        # Rejecting the deletion through the web reject path must
        # restore the original bytes at the workspace location.
        _reject_all_hunks_in_file(entry)
        self.assertEqual(
            Path(self.repo, "img.bin").read_bytes(), BINARY_BYTES,
            "reject did not restore the deleted binary file",
        )


if __name__ == "__main__":
    unittest.main()
