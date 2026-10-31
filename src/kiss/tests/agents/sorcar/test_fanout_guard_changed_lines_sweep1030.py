# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests: a review's budget counts the lines git would diff.

``fanout_guard.changed_lines`` sizes a read-only review's default
budget.  It counted the newline bytes of every untracked file, so a
binary blob (zero lines in git's numstat) raised the budget with its
size; it read ``git ls-files`` with C-quoting on, so the lines of a file
with a non-ASCII name were missed; and it decoded git's output as text,
so a non-UTF-8 file name shown unquoted (``core.quotePath=false``)
raised ``UnicodeDecodeError`` out of the review dispatch.  Real git
repositories, no test doubles.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kiss.agents.sorcar import fanout_guard

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(repo: Path, *args: str) -> None:
    """Run one git command in *repo*, failing the test when it fails."""
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo, check=True, capture_output=True,
    )


def _repo(tmp_path: Path) -> Path:
    """An empty git repository with one empty commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "init")
    return repo


def test_untracked_files_count_like_git_numstat(tmp_path: Path) -> None:
    """Text lines count, binary files count none, odd names are read; git's numstat agrees."""
    repo = _repo(tmp_path)
    (repo / "é.txt").write_text("a\nb\n", encoding="utf-8")
    (repo / "blob.bin").write_bytes(b"\0" + b"\n" * 5000)
    (repo / "notes.md").write_text("one\ntwo\nthree")
    os.symlink("é.txt", repo / "link")  # git diffs a symlink as its one-line target
    assert fanout_guard.changed_lines(str(repo)) == 2 + 0 + 3 + 1
    _git(repo, "add", "--intent-to-add", ".")  # now git's own numstat counts them
    assert fanout_guard.changed_lines(str(repo)) == 6
    # An untracked nested repository is listed as ``sub/``: a directory,
    # not a file, so it counts none.
    _git(repo, "init", "-q", "sub")
    (repo / "sub" / "inner.txt").write_text("inner\n")
    assert fanout_guard.changed_lines(str(repo)) == 6


def test_a_file_past_git_big_file_threshold_counts_none(tmp_path: Path) -> None:
    """git diffs a file bigger than core.bigFileThreshold as binary: it is not read either."""
    repo = _repo(tmp_path)
    (repo / "small.txt").write_text("a\nb\n")
    # Sparse: text where git looks for a NUL byte, zeros after it.
    with open(repo / "big.log", "wb") as handle:
        handle.write(b"y\n" * 4000)
        handle.truncate(fanout_guard.GIT_BIG_FILE_BYTES + 1)
    assert fanout_guard.changed_lines(str(repo)) == 2


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs non-UTF-8 file names")
def test_non_utf8_file_names_do_not_break_the_count(tmp_path: Path) -> None:
    """A non-UTF-8 name, tracked and shown unquoted or untracked, is counted, not a crash."""
    repo = _repo(tmp_path)
    _git(repo, "config", "core.quotePath", "false")
    tracked = os.path.join(os.fsencode(repo), b"\xff-tracked.txt")
    with open(tracked, "wb") as handle:
        handle.write(b"x\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "tracked")
    with open(tracked, "ab") as handle:
        handle.write(b"y\nz\n")
    with open(os.path.join(os.fsencode(repo), b"\xfe-new.txt"), "wb") as handle:
        handle.write(b"1\n2\n3\n4\n")
    assert fanout_guard.changed_lines(str(repo)) == 2 + 4
