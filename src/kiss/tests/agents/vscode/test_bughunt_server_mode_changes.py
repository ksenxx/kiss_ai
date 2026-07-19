# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt (server): mode-only changes must be reviewable & rejectable.

Two related defects in the merge review machinery:

1. ``_parse_diff_hunks`` ignored git's ``old mode NNNNNN`` header
   lines, so a mode-only change (``chmod +x`` / ``chmod -x`` with no
   content edit) produced NO manifest entry at all —
   ``_prepare_merge_view`` answered ``{"error": "No changes"}`` while
   ``git status`` showed the file modified, and the change could
   neither be reviewed nor rejected.

2. The reject path only knew how to SET exec bits (``exec: True`` for
   base mode ``100755``).  An exec bit the agent ADDED to a ``100644``
   file was never cleared on reject, leaving the tree dirty
   (``old mode 100644 / new mode 100755``) after a full reject-all.
   The manifest ``exec`` flag is now tri-state (True / False / absent)
   and the restore applies or clears the bits accordingly.

Real git repositories, real files — no mocks.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from kiss.server.diff_merge import _prepare_merge_view
from kiss.server.web_server import _reject_all_hunks_in_file


class TestModeOnlyChangeReview(unittest.TestCase):
    """chmod-only changes must appear in the review and reject cleanly."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt-mode-")
        self.repo = Path(self.tmpdir) / "repo"
        self.repo.mkdir()
        self._git("init", "-q")
        self._git("config", "user.email", "t@t")
        self._git("config", "user.name", "t")
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

    def _commit_file(self, name: str, text: str, mode: int) -> Path:
        path = self.repo / name
        path.write_text(text)
        path.chmod(mode)
        self._git("add", "-A")
        self._git("commit", "-qm", "init")
        return path

    def _open_view(self) -> dict:
        return _prepare_merge_view(str(self.repo), str(self.data_dir), {}, set())

    def _reject_all(self) -> None:
        manifest = json.loads(
            (self.data_dir / "pending-merge.json").read_text(),
        )
        for f in manifest["files"]:
            _reject_all_hunks_in_file(f)

    def test_chmod_plus_x_only_is_visible(self) -> None:
        """chmod +x with no content edit must open a review, not 'No changes'."""
        script = self._commit_file("run.sh", "#!/bin/sh\necho hi\n", 0o644)
        script.chmod(0o755)

        result = self._open_view()

        self.assertNotEqual(
            result.get("error"), "No changes",
            "BUG: a mode-only chmod +x was invisible to the merge review "
            "even though git status reports the file as modified",
        )
        self.assertEqual(result.get("status"), "opened", result)

    def test_reject_chmod_plus_x_clears_exec_bit(self) -> None:
        """Rejecting an agent-added exec bit must leave a clean 644 tree."""
        script = self._commit_file("run.sh", "#!/bin/sh\necho hi\n", 0o644)
        script.chmod(0o755)

        result = self._open_view()
        self.assertEqual(result.get("status"), "opened", result)
        self._reject_all()

        self.assertEqual(script.read_text(), "#!/bin/sh\necho hi\n")
        self.assertFalse(
            os.access(script, os.X_OK),
            "BUG: rejecting the chmod +x left the exec bit set "
            f"(mode {oct(script.stat().st_mode & 0o777)})",
        )
        self.assertEqual(
            self._git("status", "--porcelain").strip(), "",
            "tree must be clean after rejecting the mode-only change",
        )

    def test_reject_chmod_minus_x_restores_exec_bit(self) -> None:
        """Rejecting an agent-removed exec bit must restore mode 755."""
        script = self._commit_file("tool.sh", "#!/bin/sh\necho t\n", 0o755)
        script.chmod(0o644)

        result = self._open_view()
        self.assertEqual(result.get("status"), "opened", result)
        self._reject_all()

        self.assertEqual(script.read_text(), "#!/bin/sh\necho t\n")
        self.assertTrue(
            os.access(script, os.X_OK),
            "BUG: rejecting the chmod -x did not restore the exec bit "
            f"(mode {oct(script.stat().st_mode & 0o777)})",
        )
        self.assertEqual(self._git("status", "--porcelain").strip(), "")

    def test_reject_content_edit_with_added_exec_bit(self) -> None:
        """Reject-all of a content edit + chmod +x restores bytes AND mode."""
        doc = self._commit_file("notes.txt", "a\nb\n", 0o644)
        doc.write_text("a\nCHANGED\n")
        doc.chmod(0o755)

        result = self._open_view()
        self.assertEqual(result.get("status"), "opened", result)
        self._reject_all()

        self.assertEqual(doc.read_text(), "a\nb\n")
        self.assertFalse(
            os.access(doc, os.X_OK),
            "BUG: reject-all restored the content but kept the "
            f"agent-added exec bit (mode {oct(doc.stat().st_mode & 0o777)})",
        )
        self.assertEqual(self._git("status", "--porcelain").strip(), "")

    def test_legacy_manifest_without_exec_key_keeps_disk_mode(self) -> None:
        """Legacy entries (no ``exec`` key) must never have modes touched.

        Older pending-merge.json manifests never stamped ``exec`` on
        text entries with a mode-644 base.  Under the new tri-state
        semantics an absent key must map to ``None`` (leave the mode
        alone) — NOT to ``False`` (clear the bits), which would strip
        the exec bit of an executable script during a reject.
        """
        script = self._commit_file("legacy.sh", "#!/bin/sh\necho l\n", 0o755)
        script.write_text("#!/bin/sh\necho CHANGED\n")
        script.chmod(0o755)

        result = self._open_view()
        self.assertEqual(result.get("status"), "opened", result)
        manifest_path = self.data_dir / "pending-merge.json"
        manifest = json.loads(manifest_path.read_text())
        for f in manifest["files"]:
            f.pop("exec", None)  # simulate a legacy manifest
            _reject_all_hunks_in_file(f)

        self.assertEqual(script.read_text(), "#!/bin/sh\necho l\n")
        self.assertTrue(
            os.access(script, os.X_OK),
            "BUG: a legacy manifest entry without an 'exec' key had its "
            "exec bit stripped on reject "
            f"(mode {oct(script.stat().st_mode & 0o777)})",
        )

    def test_accepting_nothing_leaves_plain_edit_flow_unchanged(self) -> None:
        """Control: a plain content edit (no mode change) still rejects."""
        doc = self._commit_file("plain.txt", "x\ny\n", 0o644)
        doc.write_text("x\nZ\n")

        result = self._open_view()
        self.assertEqual(result.get("status"), "opened", result)
        self._reject_all()

        self.assertEqual(doc.read_text(), "x\ny\n")
        self.assertFalse(os.access(doc, os.X_OK))
        self.assertEqual(self._git("status", "--porcelain").strip(), "")


if __name__ == "__main__":
    unittest.main()
