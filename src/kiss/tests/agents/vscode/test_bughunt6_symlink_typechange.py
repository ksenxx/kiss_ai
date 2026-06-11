# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 6: rejecting changes to a TRACKED symlink corrupts the path.

git stores a symlink as a tiny blob (mode ``120000``) whose content is
the target string.  Three agent actions on a tracked symlink break the
line-based merge review machinery:

* **symlink → regular file** (typechange): ``git diff --no-renames``
  emits TWO ``diff --git`` entries for the SAME path (delete of the
  120000 blob + add of the 100644 file).  ``_parse_diff_hunks`` merged
  both into one hunk list whose coordinates do not compose — rejecting
  produced a concatenation like ``b"data.txtagent content\\n"``
  (neither the symlink nor the agent's file, and the base blob's
  missing trailing newline glued two lines together).

* **retarget** (link → other target): the "current" content is read
  THROUGH the link (the pointed-to file's lines) while the base is the
  one-line blob — the splice produced ``b"data.txtx2\\nx3\\n"`` style
  garbage.

* **delete**: rejecting the deletion wrote the blob content
  (``data.txt``) as a REGULAR file, leaving a typechange (`` T``)
  instead of restoring the link.

After the fix, ``_prepare_merge_view`` reviews any path whose base
blob is a symlink as a single whole-file entry carrying
``link_target``, and the reject path restores the symlink itself.
All tests use real git repositories — no mocks.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from kiss.agents.vscode.diff_merge import _prepare_merge_view
from kiss.agents.vscode.web_server import _reject_all_hunks_in_file


class TestSymlinkTypechangeReject(unittest.TestCase):
    """Reject must restore the tracked symlink, not splice blob text."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt6-symlink-")
        self.repo = Path(self.tmpdir) / "repo"
        self.repo.mkdir()
        self._git("init", "-q")
        self._git("config", "user.email", "t@t")
        self._git("config", "user.name", "t")
        (self.repo / "data.txt").write_text("hello\n")
        (self.repo / "other.txt").write_text("x1\nx2\nx3\n")
        os.symlink("data.txt", self.repo / "link.txt")
        self._git("add", "-A")
        self._git("commit", "-qm", "init")
        self.data_dir = Path(self.tmpdir) / "mergedata"
        self.data_dir.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=self.repo, capture_output=True, text=True,
            check=True,
        )
        return result.stdout

    def _review_and_reject_all(self) -> list[dict]:
        """Run _prepare_merge_view and reject every file's hunks."""
        result = _prepare_merge_view(str(self.repo), str(self.data_dir), {}, set())
        self.assertEqual(result.get("status"), "opened", result)
        manifest = json.loads(
            (self.data_dir / "pending-merge.json").read_text(),
        )
        files: list[dict] = manifest["files"]
        for f in files:
            _reject_all_hunks_in_file(f)
        return files

    def _assert_link_restored(self) -> None:
        link = self.repo / "link.txt"
        self.assertTrue(
            link.is_symlink(),
            "BUG: rejecting did not restore the tracked symlink "
            f"(on disk: {link.read_bytes() if link.exists() else 'missing'!r})",
        )
        self.assertEqual(os.readlink(link), "data.txt")
        self.assertEqual(
            self._git("status", "--porcelain").strip(), "",
            "working tree must be clean after rejecting all agent changes",
        )

    def test_reject_restores_symlink_replaced_by_file(self) -> None:
        """Agent replaced the symlink with a regular file; reject restores it."""
        (self.repo / "link.txt").unlink()
        (self.repo / "link.txt").write_text("agent content\n")
        self._review_and_reject_all()
        self._assert_link_restored()
        # The original target file is untouched.
        self.assertEqual((self.repo / "data.txt").read_text(), "hello\n")

    def test_reject_restores_retargeted_symlink(self) -> None:
        """Agent re-pointed the symlink; reject restores the old target."""
        (self.repo / "link.txt").unlink()
        os.symlink("other.txt", self.repo / "link.txt")
        self._review_and_reject_all()
        self._assert_link_restored()
        self.assertEqual((self.repo / "other.txt").read_text(), "x1\nx2\nx3\n")

    def test_reject_restores_deleted_symlink(self) -> None:
        """Agent deleted the symlink; reject restores the link, not a file."""
        (self.repo / "link.txt").unlink()
        self._review_and_reject_all()
        self._assert_link_restored()

    def test_symlink_entry_is_a_single_review_unit(self) -> None:
        """A symlink typechange must be one whole-file decision, not two
        incoherent line hunks."""
        (self.repo / "link.txt").unlink()
        (self.repo / "link.txt").write_text("agent content\n")
        result = _prepare_merge_view(str(self.repo), str(self.data_dir), {}, set())
        self.assertEqual(result.get("status"), "opened", result)
        manifest = json.loads(
            (self.data_dir / "pending-merge.json").read_text(),
        )
        entries = [f for f in manifest["files"] if f["name"] == "link.txt"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(
            len(entries[0]["hunks"]), 1,
            "symlink typechange must carry a single whole-file hunk "
            f"(got {entries[0]['hunks']})",
        )

    def test_reject_file_to_symlink_still_restores_file(self) -> None:
        """Control: agent replaced a regular file with a symlink — reject
        restores the regular file (pinned pre-existing behavior)."""
        (self.repo / "other.txt").unlink()
        os.symlink("data.txt", self.repo / "other.txt")
        self._review_and_reject_all()
        other = self.repo / "other.txt"
        self.assertFalse(other.is_symlink())
        self.assertEqual(other.read_text(), "x1\nx2\nx3\n")
        self.assertEqual((self.repo / "data.txt").read_text(), "hello\n")
        self.assertEqual(self._git("status", "--porcelain").strip(), "")


if __name__ == "__main__":
    unittest.main()
