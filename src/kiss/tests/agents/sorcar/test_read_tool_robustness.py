# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the ``Read`` tool's robustness fallbacks.

These tests were added in response to a recurring class of model
failures observed in ~/.kiss/sorcar.db where the ``Read`` tool returned
bare or cryptic error messages (empty body, ``Errno 21``, ``FileNotFoundError``,
stale ``.kiss-worktrees/kiss_wt-*`` paths, literal ``PWD/`` prefix) and
the model interpreted the response as a failure.

Each test exercises a real filesystem fixture (no mocks).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools


@pytest.fixture
def temp_dir():
    d = Path(tempfile.mkdtemp()).resolve()
    cwd = Path.cwd()
    os.chdir(d)
    yield d
    os.chdir(cwd)
    shutil.rmtree(d, ignore_errors=True)


def test_empty_file_returns_sentinel(temp_dir: Path) -> None:
    """Empty files must surface a clear sentinel rather than ``""``."""
    p = temp_dir / "CONFIG.md"
    p.write_text("")

    out = UsefulTools().Read(str(p))

    assert out == "(file is empty)"


def test_non_empty_file_unchanged(temp_dir: Path) -> None:
    """Non-empty files still return their raw contents verbatim."""
    p = temp_dir / "a.txt"
    p.write_text("hello\nworld\n")
    assert UsefulTools().Read(str(p)) == "hello\nworld\n"


def test_directory_returns_listing(temp_dir: Path) -> None:
    """Reading a directory returns a one-per-line listing, not ``Errno 21``."""
    sub = temp_dir / "papers"
    sub.mkdir()
    (sub / "kiss_sorcar.tex").write_text("\\documentclass{article}\n")
    (sub / "notes").mkdir()

    out = UsefulTools().Read(str(sub))

    assert "is a directory" in out
    assert "Errno 21" not in out
    assert "kiss_sorcar.tex" in out
    assert "notes/" in out


def test_empty_directory_listing(temp_dir: Path) -> None:
    """A directory with no entries renders ``(empty directory)``."""
    sub = temp_dir / "empty_dir"
    sub.mkdir()

    out = UsefulTools().Read(str(sub))

    assert "is a directory" in out
    assert "(empty directory)" in out


def test_file_not_found_suggests_close_match(temp_dir: Path) -> None:
    """A typo like ``SORCR.md`` should suggest the real ``SORCAR.md``."""
    (temp_dir / "SORCAR.md").write_text("# sorcar\n")
    (temp_dir / "INJECTIONS.md").write_text("# injections\n")

    out = UsefulTools().Read(str(temp_dir / "SORCR.md"))

    assert "File not found" in out
    assert "Did you mean" in out
    assert "SORCAR.md" in out


def test_file_not_found_no_suggestion_when_nothing_close(temp_dir: Path) -> None:
    """If no nearby file matches, no suggestion is emitted."""
    (temp_dir / "completely_different.bin").write_text("x")

    out = UsefulTools().Read(str(temp_dir / "qqqqqq.md"))

    assert "File not found" in out
    assert "Did you mean" not in out


def test_file_not_found_walks_up_to_existing_parent(temp_dir: Path) -> None:
    """Suggestions still surface when the immediate parent doesn't exist."""
    (temp_dir / "SORCAR.md").write_text("# sorcar\n")

    out = UsefulTools().Read(str(temp_dir / "no_such_dir" / "SORCR.md"))

    assert "File not found" in out
    # The suggestion walks up to ``temp_dir`` and finds SORCAR.md.
    assert "SORCAR.md" in out


def test_pwd_prefix_expands_to_work_dir(temp_dir: Path) -> None:
    """``PWD/foo`` is rewritten to ``<work_dir>/foo``."""
    (temp_dir / "CONFIG.md").write_text("config body\n")
    tools = UsefulTools(work_dir=str(temp_dir))

    out = tools.Read("PWD/CONFIG.md")

    assert out == "config body\n"


def test_pwd_alone_expands_to_directory_listing(temp_dir: Path) -> None:
    """``PWD`` alone resolves to ``work_dir`` and yields a directory listing."""
    (temp_dir / "hello.txt").write_text("hi\n")
    tools = UsefulTools(work_dir=str(temp_dir))

    out = tools.Read("PWD")

    assert "is a directory" in out
    assert "hello.txt" in out


def test_pwd_prefix_uses_cwd_when_no_work_dir(temp_dir: Path) -> None:
    """When no ``work_dir`` is set, ``PWD/`` falls back to ``os.getcwd()``.

    The ``temp_dir`` fixture chdirs into the temp directory.
    """
    (temp_dir / "hi.md").write_text("hi\n")
    out = UsefulTools().Read("PWD/hi.md")
    assert out == "hi\n"


def test_stale_worktree_falls_back_to_repo(temp_dir: Path) -> None:
    """Reading a path under a now-deleted ``.kiss-worktrees/kiss_wt-*``
    transparently falls back to the equivalent in-repo path."""
    repo = temp_dir
    (repo / "src").mkdir()
    target = repo / "src" / "module.py"
    target.write_text("REAL CONTENT\n")

    # Simulate a worktree that has already been torn down: only the
    # path string survives, nothing on disk.
    stale = (
        repo
        / ".kiss-worktrees"
        / "kiss_wt-abc123-1700000000"
        / "src"
        / "module.py"
    )

    out = UsefulTools().Read(str(stale))

    assert out == "REAL CONTENT\n"


def test_stale_worktree_fallback_does_not_mask_real_misses(temp_dir: Path) -> None:
    """If neither the worktree path nor the in-repo equivalent exists,
    we still surface a ``File not found`` error (no silent success)."""
    stale = (
        temp_dir
        / ".kiss-worktrees"
        / "kiss_wt-deadbeef-1700000000"
        / "does_not_exist.py"
    )

    out = UsefulTools().Read(str(stale))

    assert "File not found" in out


def test_live_worktree_path_is_read_directly(temp_dir: Path) -> None:
    """A path under an *existing* worktree dir is read from the worktree,
    not silently rerouted to the parent repo."""
    repo = temp_dir
    wt = repo / ".kiss-worktrees" / "kiss_wt-abc-1"
    (wt / "src").mkdir(parents=True)
    (wt / "src" / "m.py").write_text("WORKTREE\n")
    (repo / "src").mkdir()
    (repo / "src" / "m.py").write_text("REPO\n")

    out = UsefulTools().Read(str(wt / "src" / "m.py"))

    assert out == "WORKTREE\n"


def test_is_a_directory_does_not_leak_errno(temp_dir: Path) -> None:
    """The legacy ``[Errno 21] Is a directory`` string must never appear."""
    sub = temp_dir / "d"
    sub.mkdir()
    out = UsefulTools().Read(str(sub))
    assert "Errno 21" not in out
    assert "[Errno" not in out


def test_text_file_with_only_newlines_is_not_empty(temp_dir: Path) -> None:
    """A file with content (even just whitespace) is not the empty sentinel."""
    p = temp_dir / "ws.txt"
    p.write_text("\n")
    assert UsefulTools().Read(str(p)) == "\n"
