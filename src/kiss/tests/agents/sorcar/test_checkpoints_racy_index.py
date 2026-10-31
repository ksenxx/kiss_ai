# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a checkpoint sees a rewrite that kept a file's stat data.

``snapshot`` copies the real index so ``git add`` can trust the stat data
of the files that did not change.  Git compares mtimes at second
granularity, so a rewrite that keeps a file's size and lands in the same
second as the write the index recorded is invisible to that comparison;
git catches it anyway by re-reading every file whose recorded mtime is
in the same second as the index file's own ("racy git").  The copy used
to be stamped with the current time, which moved the index out of that
second: a same-size rewrite of a file written right after the real index
was, snapshotted a second or more later, was recorded with its indexed
content — ``rollback`` found nothing to roll back, and a ``checkpoint``
taken after the rewrite would have restored the old text.

The tests set the file's and the index's mtimes to one past second
(``os.utime``) instead of racing the clock, and take ctime out of git's
comparison (``core.trustctime``), as it is on Windows, where it is the
creation time.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from kiss.agents.sorcar.useful_tools import UsefulTools


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
         "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True,
    )


def _repo_written_in_one_second(path: Path, second: int) -> Path:
    """A repository whose ``a.txt`` and index both carry the mtime *second*."""
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "core.trustctime", "false")
    (path / "a.txt").write_text("v1\n")
    os.utime(path / "a.txt", (second, second))
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "init")
    os.utime(path / ".git" / "index", (second, second))
    return path


def _rewrite_in_place(path: Path, text: str, second: int) -> None:
    """Replace *path*'s content with the same-size *text*, keeping its mtime second."""
    assert len(text) == len(path.read_text())
    path.write_text(text)
    os.utime(path, (second, second))


def test_a_rollback_undoes_a_same_size_rewrite_in_the_index_second(tmp_path: Path) -> None:
    second = int(time.time()) - 10
    repo = _repo_written_in_one_second(tmp_path / "repo", second)
    tools = UsefulTools(work_dir=str(repo))
    assert tools.checkpoint("base").startswith("Saved")

    _rewrite_in_place(repo / "a.txt", "v2\n", second)

    out = tools.rollback("base")
    assert out.startswith("Rolled "), out
    assert (repo / "a.txt").read_text() == "v1\n"


def test_a_checkpoint_records_a_same_size_rewrite_in_the_index_second(tmp_path: Path) -> None:
    second = int(time.time()) - 10
    repo = _repo_written_in_one_second(tmp_path / "repo", second)
    tools = UsefulTools(work_dir=str(repo))
    assert tools.checkpoint("base").startswith("Saved")
    _rewrite_in_place(repo / "a.txt", "v2\n", second)
    assert tools.checkpoint("edited").startswith("Saved")

    assert tools.rollback("base").startswith("Rolled ")
    assert (repo / "a.txt").read_text() == "v1\n"
    out = tools.rollback("edited")
    assert out.startswith("Rolled "), out
    assert (repo / "a.txt").read_text() == "v2\n"
