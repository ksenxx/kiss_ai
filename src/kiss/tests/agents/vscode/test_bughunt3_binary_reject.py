# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: rejecting a MODIFIED binary file's merge entry.

Reproduces a real bug in ``web_server._reject_hunk_in_file`` /
``_reject_all_hunks_in_file``: manifest entries for binary files carry
the whole-file pseudo-hunk ``{bs:0,bc:0,cs:0,cc:0}`` and point
``current``/``target`` at the real workspace file.  The reject path
called ``Path.read_text()`` on it, which raises ``UnicodeDecodeError``
— NOT the ``OSError`` the code caught — so the merge action crashed
and the binary change could never be rejected (and had it been caught,
the join/write logic would have truncated the file).  Rejecting must
restore the base bytes at the target path.

No mocks, patches, fakes, or test doubles.  Real git repository, real
``_prepare_merge_view`` manifest.
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

ORIGINAL = b"\x00\x89PNG original bytes \x00\x01\x02"
MODIFIED = b"\x00\x89PNG agent-modified bytes \xff\xfe"


def _git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


class TestBinaryRejectRestoresBase(unittest.TestCase):
    """Rejecting a binary file's pseudo-hunk must restore base bytes."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-binrej-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True)
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.email", "test@example.com")
        _git(self.repo, "config", "user.name", "Test User")
        _git(self.repo, "config", "commit.gpgsign", "false")
        Path(self.repo, "img.bin").write_bytes(ORIGINAL)
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "initial")
        self.data_dir = str(Path(self.tmpdir) / "merge-data")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_reject_modified_binary(self) -> None:
        """Reject must not crash and must restore the committed bytes."""
        Path(self.repo, "img.bin").write_bytes(MODIFIED)

        result = _prepare_merge_view(self.repo, self.data_dir, {}, set())
        self.assertEqual(result.get("status"), "opened", str(result))
        manifest = json.loads(
            (Path(self.data_dir) / "pending-merge.json").read_text(),
        )
        entries = {f["name"]: f for f in manifest["files"]}
        self.assertIn("img.bin", entries, str(sorted(entries)))
        entry = entries["img.bin"]
        self.assertTrue(entry.get("binary"), str(entry))

        _reject_all_hunks_in_file(entry)

        self.assertEqual(
            Path(self.repo, "img.bin").read_bytes(), ORIGINAL,
            "reject did not restore the binary file to its base bytes",
        )


if __name__ == "__main__":
    unittest.main()
