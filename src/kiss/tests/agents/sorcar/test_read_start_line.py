# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the ``start_line`` parameter on ``UsefulTools.Read``.

The Read tool previously had no way to start at a specific line, so a
model that had already inspected the first ``max_lines`` of a large file
could not page forward without resorting to forbidden shell substitutes
like ``sed -n`` or ``tail -n +``. These tests pin the new ``start_line``
contract:

* ``start_line=1`` (default) reads from the top → backward compatible.
* ``start_line=N`` returns the file starting at the Nth 1-indexed line.
* Reading past EOF returns a clear sentinel, not an empty string.
* ``start_line=0`` or a negative value is rejected with an explicit error.
* The truncation footer reports lines that remain after the returned
  window (lines skipped *before* ``start_line`` are not double-counted).

All tests use real files on disk — no mocks.
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


def _make_numbered(p: Path, n: int) -> None:
    """Write a file with lines ``line-1``..``line-N`` (one per line)."""
    p.write_text("".join(f"line-{i}\n" for i in range(1, n + 1)))


def test_start_line_default_is_one(temp_dir: Path) -> None:
    """Omitting ``start_line`` is equivalent to ``start_line=1``."""
    p = temp_dir / "f.txt"
    _make_numbered(p, 5)

    assert UsefulTools().Read(str(p)) == UsefulTools().Read(str(p), start_line=1)


def test_start_line_skips_prefix(temp_dir: Path) -> None:
    """``start_line=3`` returns the file starting at the third line."""
    p = temp_dir / "f.txt"
    _make_numbered(p, 5)

    out = UsefulTools().Read(str(p), start_line=3)

    assert out == "line-3\nline-4\nline-5\n"


def test_start_line_with_max_lines_window(temp_dir: Path) -> None:
    """``start_line`` + ``max_lines`` return an interior window with a footer."""
    p = temp_dir / "f.txt"
    _make_numbered(p, 10)

    out = UsefulTools().Read(str(p), start_line=4, max_lines=3)

    # Window is lines 4..6 — 4 lines (7,8,9,10) remain after.
    assert out.startswith("line-4\nline-5\nline-6\n")
    assert "[truncated: 4 more lines]" in out
    assert "line-7" not in out


def test_start_line_at_last_line(temp_dir: Path) -> None:
    """``start_line`` equal to the line count returns just that line."""
    p = temp_dir / "f.txt"
    _make_numbered(p, 5)

    out = UsefulTools().Read(str(p), start_line=5)

    assert out == "line-5\n"


def test_start_line_past_eof_returns_sentinel(temp_dir: Path) -> None:
    """A ``start_line`` past EOF returns an explicit error sentinel."""
    p = temp_dir / "f.txt"
    _make_numbered(p, 3)

    out = UsefulTools().Read(str(p), start_line=10)

    assert "start_line" in out
    assert "3" in out  # mentions the actual line count
    assert "line-1" not in out  # must NOT leak earlier content


def test_start_line_zero_is_rejected(temp_dir: Path) -> None:
    """``start_line=0`` is an error (1-indexed contract)."""
    p = temp_dir / "f.txt"
    _make_numbered(p, 3)

    out = UsefulTools().Read(str(p), start_line=0)

    assert out.lower().startswith("error")
    assert "start_line" in out


def test_start_line_negative_is_rejected(temp_dir: Path) -> None:
    """Negative ``start_line`` is rejected (no Python-style backward indexing)."""
    p = temp_dir / "f.txt"
    _make_numbered(p, 3)

    out = UsefulTools().Read(str(p), start_line=-2)

    assert out.lower().startswith("error")
    assert "start_line" in out


def test_start_line_empty_file_still_sentinel(temp_dir: Path) -> None:
    """Empty file behaviour is unchanged regardless of ``start_line``."""
    p = temp_dir / "f.txt"
    p.write_text("")

    assert UsefulTools().Read(str(p), start_line=5) == "(file is empty)"


def test_start_line_does_not_double_truncate_footer(temp_dir: Path) -> None:
    """The footer counts only lines after the returned window, not the prefix."""
    p = temp_dir / "f.txt"
    _make_numbered(p, 100)

    # Read lines 50..59 — 41 lines remain after line 59.
    out = UsefulTools().Read(str(p), start_line=50, max_lines=10)

    assert "line-50\n" in out
    assert "line-59\n" in out
    assert "line-60" not in out
    assert "[truncated: 41 more lines]" in out


def test_start_line_preserves_trailing_newline_handling(temp_dir: Path) -> None:
    """A windowed read into a file without trailing newline keeps the last line."""
    p = temp_dir / "f.txt"
    p.write_text("a\nb\nc")  # no trailing newline

    out = UsefulTools().Read(str(p), start_line=2)

    assert out == "b\nc"
