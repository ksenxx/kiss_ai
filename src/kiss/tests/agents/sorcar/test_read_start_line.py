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
import tracemalloc
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.config import DEFAULT_CONFIG


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

    assert out.startswith("line-4\nline-5\nline-6\n")
    assert "[truncated: 4 more lines;" in out
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
    assert "3" in out
    assert "line-1" not in out


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

    out = UsefulTools().Read(str(p), start_line=50, max_lines=10)

    assert "line-50\n" in out
    assert "line-59\n" in out
    assert "line-60" not in out
    assert "[truncated: 41 more lines;" in out


def test_start_line_counts_the_lines_grep_and_sed_count(temp_dir: Path) -> None:
    """Only ``\\n`` ends a line: ``start_line`` is the number ``grep -n`` and an Edit report give.

    ``str.splitlines`` also split at a form feed (a page break in a
    source file) and at U+2028, which moved every later line a number down.
    """
    p = temp_dir / "f.py"
    p.write_text("a = 1\x0cb = 2\nc = '\u2028'\nd = 4\n", encoding="utf-8")
    tools = UsefulTools()

    assert tools.Read(str(p), start_line=2, max_lines=1).startswith("c = '\u2028'\n")
    assert tools.Read(str(p), start_line=3) == "d = 4\n"
    assert "past EOF (file has 3 lines)" in tools.Read(str(p), start_line=4)
    # Review 2026-11-11 S4-5: a lone carriage return ended a line too (the file was decoded
    # with universal newlines); it stays in its line, and ``\r\n`` is shown as ``\n``.
    p.write_bytes(b"a\rb\nc\r\nd\n")
    assert tools.Read(str(p)) == "a\rb\nc\nd\n"
    assert tools.Read(str(p), start_line=2, max_lines=1).startswith("c\n")
    assert "past EOF (file has 3 lines)" in tools.Read(str(p), start_line=4)


def test_a_window_of_a_big_file_costs_the_window_not_the_file(temp_dir: Path) -> None:
    """Read keeps what it returns, not the file: its memory follows the window asked for.

    S16/F4: the file was read whole and split into a list of all its
    lines before the window was taken: 35 times its size for a log of
    short lines, whatever ``max_lines`` said.  A file of one line was
    held whole to show the start of it, and so was (review of S16/F4)
    each line of a file long enough for an outline: one the outline
    does not list, and one it lists clipped.  Read in pieces of 65,536
    characters, a long line keeps the ends the file gives it: a
    ``\\r\\n`` two pieces share is one ``\\n``, and a ``\\r`` alone
    stays where it is, at the end of a piece, of the file or of both.
    """
    lines, budget = 1 << 18, DEFAULT_CONFIG.tool_output_default_chars
    line = "x" * 15 + "\n"
    fit = budget // len(line)
    many, one, ends = (temp_dir / name for name in ("many.log", "one.json", "ends.txt"))
    many.write_bytes(line.replace("\n", "\r\n").encode() * lines)
    one.write_bytes(b"x" * (4 << 20))
    long, defs = temp_dir / "long.log", temp_dir / "defs.py"
    long.write_bytes(b"y" * (4 << 20) + b"\n" * 2001)
    defs.write_bytes(b"def big(): " + b"x" * (4 << 20) + b"\n" + b"def e(): pass\n" * 2001)
    outline = (
        f"{defs}: 2,002 lines, {defs.stat().st_size:,} chars. Too long to send whole; outline "
        f"of its 2002 definitions/headings (line: text):\n1: def big(): {'x' * (budget - 11)}"
        f" [line clipped to {budget} chars]\n"
        + "".join(f"{number}: def e(): pass\n" for number in range(2, 401))
        + "... 1602 more entries\n\nRead a range with Read(file_path, start_line=N, max_lines=M), "
        'or locate text with Bash("grep -n PATTERN FILE").'
    )
    a, b = "a" * 65535, "b" * 65535
    ends.write_bytes(f"{a}\r\n{b}\rc\n\r".encode())
    more = "\n[truncated: {} more lines; Read(file_path, start_line={}, max_lines=N) continues]"
    clipped = f"\n[line clipped to {budget} chars; grep or cut for the rest]\n"
    tools = UsefulTools()
    for path, start, count, expected in (
        (many, 1, 1, line + more.format(lines - 1, 2)),
        (many, lines - 1, 2, line * 2),
        (many, 1, 2000, line * fit + more.format(lines - fit, fit + 1)),
        (one, 1, 2000, "x" * budget + clipped),
        (long, 1, 2000, "y" * budget + clipped + more.format(2000, 2)),
        (defs, 1, 2000, outline),
    ):
        tracemalloc.start()
        try:
            shown = tools.Read(str(path), max_lines=count, start_line=start)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert shown == expected, (path.name, start, count, shown[:200])
        assert peak < 1 << 20, f"Read({path.name}, {start}, {count}) allocated {peak} bytes"
    assert tools.Read(str(ends), max_lines=5) == f"{a}\n{b}\rc\n\r"
    ends.write_bytes(f"{a}\r".encode())
    assert tools.Read(str(ends), max_lines=5) == f"{a}\r"


def test_start_line_preserves_trailing_newline_handling(temp_dir: Path) -> None:
    """A windowed read into a file without trailing newline keeps the last line."""
    p = temp_dir / "f.txt"
    p.write_text("a\nb\nc")

    out = UsefulTools().Read(str(p), start_line=2)

    assert out == "b\nc"
