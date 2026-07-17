# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: rejecting one hunk must not rewrite a CRLF file as LF.

Reproduces a real bug in ``web_server._reject_hunk_in_file``: it read
the current and base files with ``Path.read_text()`` — which performs
universal-newline translation (``\\r\\n`` → ``\\n``) — then wrote the
joined lines back.  Rejecting a SINGLE hunk in a CRLF file therefore
silently converted EVERY line of the file to LF endings, corrupting the
file far beyond the rejected hunk.  ``diff_merge._write_base_copy``
already goes out of its way to preserve CRLF bytes in the base copy
(M5 fix), so the reject path destroying them is an inconsistency.

No mocks, patches, fakes, or test doubles.  Real git repository, real
``_prepare_merge_view`` manifest, real reject path.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from kiss.server.diff_merge import _prepare_merge_view
from kiss.server.web_server import _reject_hunk_in_file

ORIGINAL = b"alpha\r\nbravo\r\ncharlie\r\ndelta\r\n"
MODIFIED = b"alpha\r\nBRAVO-agent\r\ncharlie\r\ndelta\r\n"


def _git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


class TestCrlfHunkReject(unittest.TestCase):
    """Rejecting a hunk in a CRLF file must restore the exact CRLF bytes."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-crlfrej-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True)
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.email", "test@example.com")
        _git(self.repo, "config", "user.name", "Test User")
        _git(self.repo, "config", "commit.gpgsign", "false")
        _git(self.repo, "config", "core.autocrlf", "false")
        Path(self.repo, "win.txt").write_bytes(ORIGINAL)
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "initial")
        self.data_dir = str(Path(self.tmpdir) / "merge-data")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_reject_single_hunk_preserves_crlf(self) -> None:
        """Rejecting the only hunk must yield the byte-exact original file."""
        Path(self.repo, "win.txt").write_bytes(MODIFIED)

        result = _prepare_merge_view(self.repo, self.data_dir, {}, set())
        self.assertEqual(result.get("status"), "opened", str(result))
        manifest = json.loads(
            (Path(self.data_dir) / "pending-merge.json").read_text(),
        )
        entries = {f["name"]: f for f in manifest["files"]}
        self.assertIn("win.txt", entries, str(sorted(entries)))
        entry = entries["win.txt"]
        self.assertEqual(
            len(entry["hunks"]), 1,
            f"expected exactly one hunk for the single-line edit: {entry}",
        )

        _reject_hunk_in_file(
            entry["current"], entry["base"], entry["hunks"][0], entry["target"],
        )

        self.assertEqual(
            Path(self.repo, "win.txt").read_bytes(), ORIGINAL,
            "rejecting one hunk rewrote the CRLF file with LF endings",
        )


if __name__ == "__main__":
    unittest.main()
