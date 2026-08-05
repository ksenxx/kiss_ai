# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""FINDINGS-4 regression tests for diff_merge.py / web_merge.py.

Covers (all reproduced with real on-disk git repositories, no mocks):

- F4-25: rejecting an agent-CREATED file must remove the path instead
  of leaving an empty untracked file behind.
- F4-26: a tracked file replaced by a directory must appear in the
  merge review as a deletion instead of being silently omitted.
- F4-27: a pre-task untracked symlink retargeted by the agent must be
  restored AS A SYMLINK to its pre-task target on reject (not as a
  regular file holding the old target's content).
- F4-28: a newly created broken symlink must be visible to the merge
  review (git reports it untracked), and rejecting it must remove it.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from kiss.server.diff_merge import (
    _capture_untracked,
    _parse_diff_hunks,
    _prepare_merge_view,
    _save_untracked_base,
    _snapshot_files,
)
from kiss.server.web_merge import _reject_all_hunks_in_file


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True, check=True,
    )
    _git(path, "config", "user.email", "t@t.com")
    _git(path, "config", "user.name", "T")
    (path / "README.md").write_text("# Test\n")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "initial")
    return path


def _load_manifest(data_dir: Path) -> dict[str, dict]:
    manifest = json.loads(
        (data_dir / "pending-merge.json").read_text(encoding="utf-8"),
    )
    return {f["name"]: f for f in manifest["files"]}


class TestF425RejectCreatedFileRemovesIt(unittest.TestCase):
    """F4-25: full reject of an agent-created file restores absence."""

    def test_reject_created_text_file_removes_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            data_dir = Path(tmp) / "merge-data"
            (repo / "new.txt").write_text("agent\n")

            result = _prepare_merge_view(
                str(repo), str(data_dir), {}, set(), None,
            )
            self.assertEqual(result.get("status"), "opened")
            entry = _load_manifest(data_dir)["new.txt"]
            self.assertTrue(
                entry.get("created"),
                "agent-created file must be flagged 'created' in the "
                "manifest so a reject can restore absence",
            )

            _reject_all_hunks_in_file(entry)

            self.assertFalse(
                (repo / "new.txt").exists(),
                "rejecting an agent-created file left an empty file "
                "behind instead of removing it",
            )
            status = _git(repo, "status", "--porcelain").stdout
            self.assertNotIn("new.txt", status)

    def test_reject_created_binary_file_removes_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            data_dir = Path(tmp) / "merge-data"
            (repo / "blob.bin").write_bytes(b"\x00\x01\x02agent")

            result = _prepare_merge_view(
                str(repo), str(data_dir), {}, set(), None,
            )
            self.assertEqual(result.get("status"), "opened")
            entry = _load_manifest(data_dir)["blob.bin"]
            self.assertTrue(entry.get("binary"))
            self.assertTrue(entry.get("created"))

            _reject_all_hunks_in_file(entry)

            self.assertFalse((repo / "blob.bin").exists())


class TestF426TrackedFileReplacedByDirectory(unittest.TestCase):
    """F4-26: the tracked file's deletion must reach the merge view."""

    def test_file_to_directory_is_reviewed_as_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            data_dir = Path(tmp) / "merge-data"
            (repo / "node").write_text("original content\n")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "add node")

            pre_hunks = _parse_diff_hunks(str(repo))
            pre_untracked = _capture_untracked(str(repo))

            # Agent replaces the tracked file with a directory.
            (repo / "node").unlink()
            (repo / "node").mkdir()
            (repo / "node" / "child.txt").write_text("child\n")

            result = _prepare_merge_view(
                str(repo), str(data_dir), pre_hunks, pre_untracked, None,
            )
            self.assertEqual(
                result.get("status"), "opened",
                f"file->directory change was invisible: {result}",
            )
            entries = _load_manifest(data_dir)
            self.assertIn(
                "node", entries,
                "the tracked file's deletion is missing from the merge "
                "review; the user cannot see or reject it",
            )


class TestF427PreTaskSymlinkRejectRestoresSymlink(unittest.TestCase):
    """F4-27: reject restores the user's pre-task symlink identity."""

    def test_retargeted_untracked_symlink_restored_on_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "one").write_text("one\n")
            (repo / "two").write_text("two\n")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "targets")
            os.symlink("one", repo / "link")

            # Pre-task snapshot exactly as task_runner performs it.
            tab_id = "f427-tab"
            pre_hunks = _parse_diff_hunks(str(repo))
            pre_untracked = _capture_untracked(str(repo))
            self.assertIn("link", pre_untracked)
            pre_hashes = _snapshot_files(
                str(repo), pre_untracked | set(pre_hunks),
            )
            _save_untracked_base(
                str(repo), pre_untracked | set(pre_hunks), tab_id,
            )
            from kiss.server.diff_merge import _merge_data_dir
            data_dir = _merge_data_dir(tab_id)

            # Agent retargets the symlink.
            (repo / "link").unlink()
            os.symlink("two", repo / "link")

            result = _prepare_merge_view(
                str(repo), str(data_dir), pre_hunks, pre_untracked,
                pre_hashes,
            )
            self.assertEqual(result.get("status"), "opened")
            entry = _load_manifest(data_dir)["link"]
            self.assertEqual(
                entry.get("link_target"), "one",
                "the manifest must carry the user's PRE-TASK link "
                "target so reject can restore the symlink identity",
            )

            _reject_all_hunks_in_file(entry)

            link = repo / "link"
            self.assertTrue(
                link.is_symlink(),
                "reject replaced the user's symlink with a regular "
                "file containing the old target's content",
            )
            self.assertEqual(os.readlink(link), "one")

    def test_unchanged_pre_task_symlink_not_reported_changed(self) -> None:
        """An untouched symlink must not show up in the merge view."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "one").write_text("one\n")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "target")
            os.symlink("one", repo / "link")

            tab_id = "f427-tab2"
            pre_hunks = _parse_diff_hunks(str(repo))
            pre_untracked = _capture_untracked(str(repo))
            pre_hashes = _snapshot_files(
                str(repo), pre_untracked | set(pre_hunks),
            )
            _save_untracked_base(
                str(repo), pre_untracked | set(pre_hunks), tab_id,
            )
            from kiss.server.diff_merge import _merge_data_dir
            data_dir = _merge_data_dir(tab_id)

            result = _prepare_merge_view(
                str(repo), str(data_dir), pre_hunks, pre_untracked,
                pre_hashes,
            )
            self.assertEqual(
                result.get("error"), "No changes",
                f"untouched pre-task symlink misreported: {result}",
            )


class TestF428NewBrokenSymlinkVisible(unittest.TestCase):
    """F4-28: a new broken symlink must be reviewed and rejectable."""

    def test_broken_symlink_reaches_review_and_reject_removes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            data_dir = Path(tmp) / "merge-data"
            os.symlink("missing-target", repo / "broken")
            status = _git(repo, "status", "--porcelain").stdout
            self.assertIn("broken", status)

            result = _prepare_merge_view(
                str(repo), str(data_dir), {}, set(), None,
            )
            self.assertEqual(
                result.get("status"), "opened",
                f"new broken symlink invisible to review: {result}",
            )
            entry = _load_manifest(data_dir)["broken"]
            self.assertTrue(entry.get("created"))

            _reject_all_hunks_in_file(entry)

            self.assertFalse(os.path.lexists(repo / "broken"))
            status = _git(repo, "status", "--porcelain").stdout
            self.assertNotIn("broken", status)


class TestF427DeletedPreTaskSymlinkReviewed(unittest.TestCase):
    """Residual F4-27: DELETING a pre-task symlink must reach review.

    A pre-task symlink removed by the agent must appear in the merge
    view as a deletion carrying the pre-task ``link_target`` so a
    reject can restore the user's symlink identity.  Covers tracked
    symlinks, untracked file-target symlinks, and untracked BROKEN
    symlinks (whose saved base fails ``is_file()``).
    """

    def _snapshot(self, repo: Path, tab_id: str):
        pre_hunks = _parse_diff_hunks(str(repo))
        pre_untracked = _capture_untracked(str(repo))
        pre_hashes = _snapshot_files(str(repo), pre_untracked | set(pre_hunks))
        _save_untracked_base(str(repo), pre_untracked | set(pre_hunks), tab_id)
        from kiss.server.diff_merge import _merge_data_dir
        return pre_hunks, pre_untracked, pre_hashes, _merge_data_dir(tab_id)

    def _assert_deletion_reviewed_and_restorable(
        self, repo: Path, data_dir: Path, result: dict, target: str,
    ) -> None:
        self.assertEqual(
            result.get("status"), "opened",
            f"deleted pre-task symlink invisible to review: {result}",
        )
        entry = _load_manifest(data_dir)["link"]
        self.assertEqual(
            entry.get("link_target"), target,
            "manifest must carry the pre-task link target so reject "
            f"can restore the symlink identity: {entry}",
        )
        _reject_all_hunks_in_file(entry)
        link = repo / "link"
        self.assertTrue(
            link.is_symlink(),
            "reject did not restore the deleted pre-task symlink",
        )
        self.assertEqual(os.readlink(link), target)

    def test_deleted_untracked_symlink_reviewed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "one").write_text("one\n")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "target")
            os.symlink("one", repo / "link")

            pre_hunks, pre_untracked, pre_hashes, data_dir = (
                self._snapshot(repo, "f427-del-untracked")
            )
            self.assertIn("link", pre_untracked)
            (repo / "link").unlink()

            result = _prepare_merge_view(
                str(repo), str(data_dir), pre_hunks, pre_untracked,
                pre_hashes,
            )
            self._assert_deletion_reviewed_and_restorable(
                repo, data_dir, result, "one",
            )

    def test_deleted_untracked_broken_symlink_reviewed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            os.symlink("missing-target", repo / "link")

            pre_hunks, pre_untracked, pre_hashes, data_dir = (
                self._snapshot(repo, "f427-del-broken")
            )
            self.assertIn("link", pre_untracked)
            (repo / "link").unlink()

            result = _prepare_merge_view(
                str(repo), str(data_dir), pre_hunks, pre_untracked,
                pre_hashes,
            )
            self._assert_deletion_reviewed_and_restorable(
                repo, data_dir, result, "missing-target",
            )

    def test_deleted_tracked_symlink_reviewed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            (repo / "one").write_text("one\n")
            os.symlink("one", repo / "link")
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "tracked link")

            pre_hunks, pre_untracked, pre_hashes, data_dir = (
                self._snapshot(repo, "f427-del-tracked")
            )
            (repo / "link").unlink()

            result = _prepare_merge_view(
                str(repo), str(data_dir), pre_hunks, pre_untracked,
                pre_hashes,
            )
            self._assert_deletion_reviewed_and_restorable(
                repo, data_dir, result, "one",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
