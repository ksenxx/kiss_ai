# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: ``_augment_merge_data`` must not translate newlines.

Reproduces a real inconsistency in ``web_server._augment_merge_data``:
it read each file's ``base``/``current`` content with
``Path.read_text()``, which performs universal-newline translation —
``\\r\\n`` becomes ``\\n`` and a lone ``\\r`` becomes ``\\n``.

Every other part of the merge pipeline counts lines by splitting on
``\\n`` ONLY with the original bytes preserved (see
``diff_merge._read_lines_preserved`` / ``_split_lines_keepends`` and
the iteration-3 CRLF reject fix), so the browser receives:

* CRLF files whose displayed text silently differs from what the
  reject path writes back to disk, and
* lone-``\\r`` files with MORE lines than the hunk ``cs``/``cc``
  coordinates were computed against (``"x\\ry\\n"`` is ONE line for the
  hunk math but ``read_text()`` turns it into TWO), so hunk
  highlighting in the web client is misaligned.

No mocks, patches, fakes, or test doubles.  Real files on disk, real
``_prepare_merge_view`` manifest, real ``_augment_merge_data`` call.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from kiss.agents.vscode.diff_merge import (
    _prepare_merge_view,
    _split_lines_keepends,
)
from kiss.agents.vscode.web_server import _augment_merge_data

CRLF_ORIGINAL = b"alpha\r\nbravo\r\ncharlie\r\n"
CRLF_MODIFIED = b"alpha\r\nBRAVO-agent\r\ncharlie\r\n"
LONE_CR_CONTENT = b"progress 1\rprogress 2\rdone\n"


def _git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


class TestAugmentMergeDataNewlines(unittest.TestCase):
    """``base_text``/``current_text`` must preserve original line endings."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-augnl-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True)
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.email", "test@example.com")
        _git(self.repo, "config", "user.name", "Test User")
        _git(self.repo, "config", "commit.gpgsign", "false")
        _git(self.repo, "config", "core.autocrlf", "false")
        self.data_dir = str(Path(self.tmpdir) / "merge-data")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _merge_event(self) -> dict:
        manifest = json.loads(
            (Path(self.data_dir) / "pending-merge.json").read_text(),
        )
        return {"type": "merge_data", "data": manifest}

    def test_crlf_text_preserved(self) -> None:
        """A CRLF file's base/current text must keep ``\\r\\n`` endings."""
        Path(self.repo, "win.txt").write_bytes(CRLF_ORIGINAL)
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "initial")
        Path(self.repo, "win.txt").write_bytes(CRLF_MODIFIED)

        result = _prepare_merge_view(self.repo, self.data_dir, {}, set())
        self.assertEqual(result.get("status"), "opened", str(result))

        out = _augment_merge_data(self._merge_event())
        entries = {f["name"]: f for f in out["data"]["files"]}
        self.assertIn("win.txt", entries)
        entry = entries["win.txt"]
        self.assertEqual(
            entry["base_text"], CRLF_ORIGINAL.decode(),
            "base_text was newline-translated (CRLF lost)",
        )
        self.assertEqual(
            entry["current_text"], CRLF_MODIFIED.decode(),
            "current_text was newline-translated (CRLF lost)",
        )

    def test_lone_cr_line_count_matches_hunks(self) -> None:
        """Lone-``\\r`` content must keep the hunk-math line count."""
        link_path = Path(self.repo, "log.txt")
        link_path.write_bytes(b"seed\n")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "initial")
        link_path.write_bytes(LONE_CR_CONTENT)

        result = _prepare_merge_view(self.repo, self.data_dir, {}, set())
        self.assertEqual(result.get("status"), "opened", str(result))

        out = _augment_merge_data(self._merge_event())
        entries = {f["name"]: f for f in out["data"]["files"]}
        self.assertIn("log.txt", entries)
        entry = entries["log.txt"]
        self.assertEqual(
            entry["current_text"], LONE_CR_CONTENT.decode(),
            "current_text was newline-translated (lone \\r became \\n)",
        )
        # The browser highlights hunk lines by splitting the text it
        # received on "\n"; that count must agree with the hunk math,
        # which split the on-disk bytes on "\n" only (1 line here).
        self.assertEqual(
            len(_split_lines_keepends(entry["current_text"])),
            len(_split_lines_keepends(LONE_CR_CONTENT.decode())),
            "browser line count diverges from hunk-coordinate line count",
        )


if __name__ == "__main__":
    unittest.main()
