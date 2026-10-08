# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``GitWorktreeOps`` info-file writers refuse a directory that is not a git repo.

``_append_info_line`` resolves ``git rev-parse --git-common-dir`` to
find ``<common>/info/<file>``.  Before the fix it ignored a failed
``rev-parse`` and resolved the empty output relative to the directory,
so ``ensure_excluded(plain_dir)`` silently created
``plain_dir/info/exclude`` — a stray file in a directory that is not a
repository at all.  The writer now raises ``OSError`` (every caller
already handles it) and leaves the directory untouched.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps


class NonRepoInfoFileTest(unittest.TestCase):
    """Info-file writers on a plain directory."""

    def test_ensure_excluded_on_plain_directory_raises_and_writes_nothing(self) -> None:
        """A plain directory gets no ``info/exclude`` and the caller sees ``OSError``."""
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / "not-a-repo"
            plain.mkdir()
            with self.assertRaises(OSError):
                GitWorktreeOps.ensure_excluded(plain)
            self.assertEqual(sorted(p.name for p in plain.iterdir()), [])

    def test_ensure_excluded_on_repo_still_writes_exclude(self) -> None:
        """The happy path is unchanged: a real repo gets the exclude entry."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            GitWorktreeOps.ensure_excluded(repo)
            exclude = repo / ".git" / "info" / "exclude"
            self.assertIn(".kiss-worktrees/", exclude.read_text().splitlines())


if __name__ == "__main__":
    unittest.main()
