# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: rejecting a symlink entry must not clobber its target.

Reproduces a real data-loss bug in the web merge reject path.  When the
agent creates an (untracked) symlink, ``_prepare_merge_view`` lists it
as a new file whose hunks were computed from the *target's* content
(``Path.is_file()`` and reads follow symlinks).  Rejecting that entry
called ``open(write_to, "w")`` on the symlink path, which writes
THROUGH the link — truncating/overwriting the pointed-to file (which
may be a precious tracked file, or live entirely outside the repo)
while the rejected symlink itself survives untouched.

Git tracks the link itself, not its target, so a reject must never
write through a symlink: the link has to be replaced, leaving the
target byte-identical.

No mocks, patches, fakes, or test doubles.  Real git repository, real
``_prepare_merge_view`` manifest, real reject paths.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from kiss.server.diff_merge import _prepare_merge_view
from kiss.server.web_server import (
    _reject_all_hunks_in_file,
    _reject_hunk_in_file,
)

PRECIOUS = "precious content\n"


def _git(cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


class TestSymlinkReject(unittest.TestCase):
    """Rejecting a symlink merge entry must leave the link target intact."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-symrej-")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True)
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.email", "test@example.com")
        _git(self.repo, "config", "user.name", "Test User")
        _git(self.repo, "config", "commit.gpgsign", "false")
        self.precious = Path(self.repo, "data.txt")
        self.precious.write_text(PRECIOUS)
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "initial")
        self.data_dir = str(Path(self.tmpdir) / "merge-data")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _link_entry(self) -> dict:
        result = _prepare_merge_view(self.repo, self.data_dir, {}, set())
        self.assertEqual(result.get("status"), "opened", str(result))
        manifest = json.loads(
            (Path(self.data_dir) / "pending-merge.json").read_text(),
        )
        entries: dict[str, dict] = {
            f["name"]: f for f in manifest["files"]
        }
        self.assertIn("link.txt", entries, str(sorted(entries)))
        return entries["link.txt"]

    def test_reject_all_does_not_truncate_target(self) -> None:
        """reject-file/reject-all on a new symlink must not gut its target."""
        Path(self.repo, "link.txt").symlink_to("data.txt")
        entry = self._link_entry()

        _reject_all_hunks_in_file(entry)

        self.assertEqual(
            self.precious.read_text(), PRECIOUS,
            "rejecting the symlink wrote through it and clobbered data.txt",
        )
        self.assertFalse(
            Path(self.repo, "link.txt").is_symlink(),
            "rejected new symlink still points at the target",
        )

    def test_reject_hunk_does_not_write_through_link(self) -> None:
        """Per-hunk reject on a symlink entry must not modify the target."""
        Path(self.repo, "link.txt").symlink_to("data.txt")
        entry = self._link_entry()

        for hunk in entry["hunks"]:
            _reject_hunk_in_file(
                entry["current"], entry["base"], hunk, entry["target"],
            )

        self.assertEqual(
            self.precious.read_text(), PRECIOUS,
            "per-hunk reject wrote through the symlink and clobbered data.txt",
        )

    def test_reject_binary_symlink_does_not_clobber_target(self) -> None:
        """Binary-flagged reject must not restore base bytes into the target."""
        blob = Path(self.repo, "blob.bin")
        blob.write_bytes(b"\x00precious-binary\x00")
        Path(self.repo, "link.bin").symlink_to("blob.bin")
        result = _prepare_merge_view(self.repo, self.data_dir, {}, set())
        self.assertEqual(result.get("status"), "opened", str(result))
        manifest = json.loads(
            (Path(self.data_dir) / "pending-merge.json").read_text(),
        )
        entries = {f["name"]: f for f in manifest["files"]}
        self.assertIn("link.bin", entries, str(sorted(entries)))

        _reject_all_hunks_in_file(entries["link.bin"])

        self.assertEqual(
            blob.read_bytes(), b"\x00precious-binary\x00",
            "binary reject wrote base bytes through the symlink",
        )


if __name__ == "__main__":
    unittest.main()
