# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 8 (group B): submodule changes corrupt the merge review.

A git submodule is recorded in the parent repo as a *gitlink* entry
(mode ``160000``) whose "content" in ``git diff`` is the synthetic
one-line ``Subproject commit <sha>`` text.  ``_parse_diff_hunks``
treats that synthetic line like a regular text hunk, so
``_prepare_merge_view``:

* creates a review entry for the submodule path whose "current" side
  is an empty ``.deleted`` placeholder (the path is a DIRECTORY, so
  ``is_file()`` is False) and whose "base" side is empty too
  (``git show HEAD:sub`` fails with "bad object" for a gitlink) —
  a phantom whole-file change between two empty files;
* on reject, the merge machinery would write regular-file content at
  the submodule path — clobbering (or colliding with) the submodule
  working directory.

Submodule pointer changes cannot be reviewed line-by-line and must be
excluded from the merge review entirely.

No mocks, patches, fakes, or test doubles: real git repositories and
a real submodule built with subprocess.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from kiss.server.diff_merge import _prepare_merge_view


def _git(cwd: str | Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "protocol.file.allow=always", *args],
        cwd=str(cwd), capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"git {args}: {result.stderr}"
    return result.stdout


class TestSubmoduleMergeReview(unittest.TestCase):
    """Submodule (gitlink) entries must never enter the merge review."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt8-sub-")
        base = Path(self.tmpdir)

        # Build the submodule's upstream repo with one commit.
        self.sub_upstream = base / "sub-upstream"
        self.sub_upstream.mkdir()
        _git(self.sub_upstream, "init", "-q")
        _git(self.sub_upstream, "config", "user.email", "t@t")
        _git(self.sub_upstream, "config", "user.name", "t")
        (self.sub_upstream / "f.txt").write_text("one\n")
        _git(self.sub_upstream, "add", "-A")
        _git(self.sub_upstream, "commit", "-qm", "one")

        # Parent repo containing the submodule plus a normal file.
        self.repo = base / "repo"
        self.repo.mkdir()
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.email", "t@t")
        _git(self.repo, "config", "user.name", "t")
        (self.repo / "normal.txt").write_text("a\nb\nc\n")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-qm", "init")
        _git(self.repo, "submodule", "add", "-q",
             str(self.sub_upstream), "sub")
        _git(self.repo, "commit", "-qm", "add submodule")

        self.data_dir = base / "mergedata"
        self.data_dir.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _bump_submodule_commit(self) -> None:
        """Create a new commit inside the checked-out submodule."""
        sub = self.repo / "sub"
        _git(sub, "config", "user.email", "t@t")
        _git(sub, "config", "user.name", "t")
        (sub / "f.txt").write_text("one\ntwo\n")
        _git(sub, "add", "-A")
        _git(sub, "commit", "-qm", "two")

    def _manifest_names(self) -> list[str]:
        manifest = json.loads(
            (self.data_dir / "pending-merge.json").read_text(),
        )
        return [f["name"] for f in manifest["files"]]

    def test_submodule_pointer_change_alone_is_no_changes(self) -> None:
        """Only the submodule pointer moved: nothing is reviewable."""
        self._bump_submodule_commit()
        result = _prepare_merge_view(
            str(self.repo), str(self.data_dir), {}, set(),
        )
        self.assertEqual(
            result.get("error"), "No changes",
            "a submodule pointer change must not open a merge review "
            f"(got {result})",
        )

    def test_submodule_excluded_when_real_file_also_changed(self) -> None:
        """Submodule pointer change + text edit: only the text file is
        reviewable, and the submodule must not appear as a phantom
        deleted file."""
        self._bump_submodule_commit()
        (self.repo / "normal.txt").write_text("a\nB\nc\n")
        result = _prepare_merge_view(
            str(self.repo), str(self.data_dir), {}, set(),
        )
        self.assertEqual(result.get("status"), "opened", result)
        names = self._manifest_names()
        self.assertIn("normal.txt", names)
        self.assertNotIn(
            "sub", names,
            "the submodule gitlink leaked into the merge review as a "
            f"phantom file entry: {names}",
        )

    def test_deleted_submodule_dir_is_not_a_phantom_deleted_file(self) -> None:
        """Agent removed the submodule working dir: the gitlink deletion
        must not be presented as a regular deleted-file review entry
        (rejecting it would write a regular file where the submodule
        directory belongs)."""
        shutil.rmtree(self.repo / "sub")
        (self.repo / "normal.txt").write_text("a\nB\nc\n")
        result = _prepare_merge_view(
            str(self.repo), str(self.data_dir), {}, set(),
        )
        self.assertEqual(result.get("status"), "opened", result)
        names = self._manifest_names()
        self.assertNotIn(
            "sub", names,
            "the deleted submodule gitlink leaked into the merge review "
            f"as a phantom deleted file: {names}",
        )


if __name__ == "__main__":
    unittest.main()
