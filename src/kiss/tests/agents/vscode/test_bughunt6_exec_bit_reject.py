# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 6: rejecting a deleted executable must restore its exec bit.

git tracks one executable bit per file (mode ``100755``).  When the
agent deletes an executable script and the user REJECTS the deletion,
the merge machinery re-created the file with default permissions
(``644``): after a full reject-all the working tree was still dirty —
``git status`` showed `` M script.sh`` with ``old mode 100755 / new
mode 100644`` — and the restored script was no longer runnable.

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


class TestExecBitReject(unittest.TestCase):
    """Reject of a deleted 100755 file must restore mode and content."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt6-exec-")
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

    def _reject_all(self) -> None:
        result = _prepare_merge_view(str(self.repo), str(self.data_dir), {}, set())
        self.assertEqual(result.get("status"), "opened", result)
        manifest = json.loads(
            (self.data_dir / "pending-merge.json").read_text(),
        )
        for f in manifest["files"]:
            _reject_all_hunks_in_file(f)

    def test_reject_deleted_executable_restores_exec_bit(self) -> None:
        """Deleted 100755 text script: reject restores content AND mode."""
        script = self.repo / "script.sh"
        script.write_text("#!/bin/sh\necho hi\n")
        script.chmod(0o755)
        self._git("add", "-A")
        self._git("commit", "-qm", "init")
        script.unlink()

        self._reject_all()

        self.assertEqual(script.read_text(), "#!/bin/sh\necho hi\n")
        self.assertTrue(
            os.access(script, os.X_OK),
            "BUG: rejected deletion restored the script without its "
            f"exec bit (mode {oct(script.stat().st_mode & 0o777)})",
        )
        self.assertEqual(
            self._git("status", "--porcelain").strip(), "",
            "tree must be clean after rejecting the deletion",
        )

    def test_reject_deleted_executable_binary_restores_exec_bit(self) -> None:
        """Deleted 100755 BINARY file: base-bytes restore keeps the mode."""
        tool = self.repo / "tool.bin"
        tool.write_bytes(b"\x7fELF\x00\x01\x02binary")
        tool.chmod(0o755)
        self._git("add", "-A")
        self._git("commit", "-qm", "init")
        tool.unlink()

        self._reject_all()

        self.assertEqual(tool.read_bytes(), b"\x7fELF\x00\x01\x02binary")
        self.assertTrue(
            os.access(tool, os.X_OK),
            "BUG: rejected binary deletion restored the file without "
            f"its exec bit (mode {oct(tool.stat().st_mode & 0o777)})",
        )
        self.assertEqual(self._git("status", "--porcelain").strip(), "")

    def test_reject_deleted_plain_file_stays_non_executable(self) -> None:
        """Control: a deleted 100644 file must NOT gain an exec bit."""
        doc = self.repo / "notes.txt"
        doc.write_text("a\nb\n")
        self._git("add", "-A")
        self._git("commit", "-qm", "init")
        doc.unlink()

        self._reject_all()

        self.assertEqual(doc.read_text(), "a\nb\n")
        self.assertFalse(os.access(doc, os.X_OK))
        self.assertEqual(self._git("status", "--porcelain").strip(), "")


if __name__ == "__main__":
    unittest.main()
