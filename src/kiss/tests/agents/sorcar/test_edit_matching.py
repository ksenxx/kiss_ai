# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The whitespace-tolerant matcher and the nearest-region finder of ``edit_matching.py``.

Pure functions over strings: the Edit tools' use of them is tested
with the tools (``test_honest_edit.py``, the container tests).
"""

from __future__ import annotations

from kiss.agents.sorcar.edit_matching import (
    TolerantMatch,
    nearest_region,
    normalize_line,
    reindent,
    split_lines,
    tolerant_matches,
)

FILE = "def f():\n    x = 1\n    y = 2\t \n    return x + y\n"


def test_split_lines_keeps_terminators_and_splits_on_newlines_only() -> None:
    assert split_lines("a\nb") == ["a\n", "b"]
    assert split_lines("a\n\n") == ["a\n", "\n"]
    assert split_lines("") == []
    # A form feed or a lone CR is part of its line, as in a numbered listing.
    assert split_lines("a\x0c\nb\rc\n") == ["a\x0c\n", "b\rc\n"]
    assert normalize_line("  x  =\t1 \t\r\n") == " x = 1"
    assert normalize_line("\n") == ""


def test_tolerant_matches_finds_the_lines_whitespace_aside() -> None:
    # Two-space indentation, a tab, trailing blanks: one match, spanning the original lines.
    (match,) = tolerant_matches(FILE, "  x = 1   \n\ty =  2\n")
    assert FILE[match.start : match.end] == "    x = 1\n    y = 2\t \n"
    assert match.indent == ""  # "    " does not end with the old line's "\t": no uniform prefix
    (match,) = tolerant_matches(FILE, "  x = 1\n  y = 2\n")
    assert match.indent == "  "  # every line carries two blanks beyond the old string's
    # Without a trailing newline the span keeps the file line's terminator, as a verbatim
    # match would; a CRLF terminator is kept whole.
    (match,) = tolerant_matches(FILE, "  return x +  y")
    assert FILE[match.start : match.end] == "    return x + y"
    crlf = "a\r\n    b\r\nc\r\n"
    (match,) = tolerant_matches(crlf, "  b")
    assert crlf[match.start : match.end] == "    b"
    (match,) = tolerant_matches(crlf, "  b\n")
    assert crlf[match.start : match.end] == "    b\r\n"
    # An unindented old line does not match an indented file line: one blank is kept.
    assert tolerant_matches(FILE, "x = 1\n") == []
    # Blank lines carry no indentation information; a file line indented less than the old
    # line, or lines disagreeing on the prefix, give no re-indentation.
    text = "    a\n\n    b\n"
    assert tolerant_matches(text, "  a\n\n  b\n") == [TolerantMatch(0, len(text), "  ")]
    assert tolerant_matches(text, "      a\n\n      b\n")[0].indent == ""
    assert tolerant_matches(text, "  a\n\n   b\n")[0].indent == ""
    assert tolerant_matches("", "a") == [] and tolerant_matches(FILE, "") == []


def test_tolerant_matches_lists_every_place() -> None:
    text = "a\n  b\nc\na\n  b\nd\n"
    matches = tolerant_matches(text, "a\n b\n")
    assert [text[m.start : m.end] for m in matches] == ["a\n  b\n", "a\n  b\n"]
    assert [m.start for m in matches] == [0, 8]


def test_reindent_prefixes_the_non_blank_lines() -> None:
    assert reindent("x = 1\n\ny = 2\n", "    ") == "    x = 1\n\n    y = 2\n"
    assert reindent("x = 1", "  ") == "  x = 1"
    assert reindent("x = 1\n", "") == "x = 1\n"


def test_nearest_region_names_the_closest_window() -> None:
    assert nearest_region(FILE, "    y = 3\n    return x + y\n") == (
        3, 4, "    y = 2\t \n    return x + y\n",
    )
    assert nearest_region(FILE, "def f():") == (1, 1, "def f():\n")
    # An old string longer than the file compares with the whole file.
    assert nearest_region("a\nb\n", "a\nb\nc\nd\n") == (1, 2, "a\nb\n")
    # Equally close windows: the first wins; a window whose quick_ratio (the same characters
    # in another order) beats the best but whose real ratio does not is passed over.
    assert nearest_region("x\ny\nx\ny\n", "x\nz\n") == (1, 2, "x\ny\n")
    assert nearest_region("axc\ncba\n", "abc") == (1, 1, "axc\n")
    assert nearest_region("", "a") is None and nearest_region(FILE, "") is None
