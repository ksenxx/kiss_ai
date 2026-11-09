"""Whitespace-tolerant matching for the ``Edit`` tools, and the nearest region of a miss.

An ``Edit`` whose ``old_string`` is not in the file verbatim usually
differs from it only in whitespace: indentation copied from a numbered
listing, trailing blanks, a tab for spaces.  :func:`tolerant_matches`
finds the lines of the file that equal the old string's lines once
runs of blanks are collapsed and trailing whitespace dropped; a unique
match is edited in place of its original text, re-indented when the
file's lines carry more indentation than the old string's
(:func:`reindent`).  When nothing matches that way either,
:func:`nearest_region` names the window of the file closest to the old
string, so the next attempt needs no ``Read``.

Pure functions over strings, standard library only, so the container
``Edit`` can ship this module's source into a container.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

_BLANKS = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class TolerantMatch:
    """Where the old string of an ``Edit`` matches a text, whitespace aside.

    Attributes:
        start: Offset in the text of the matched original text.
        end: Offset past its last character.
        indent: The leading whitespace every non-blank line of the file
            carries beyond the old string's line (uniformly), ``""`` when
            the lines do not differ that way; :func:`reindent` adds it to
            the replacement.
    """

    start: int
    end: int
    indent: str


def normalize_line(line: str) -> str:
    """Return *line* with runs of blanks collapsed to one and trailing whitespace dropped.

    The line's terminator (``\\n``, ``\\r\\n``) goes with the trailing
    whitespace, so lines are compared by their content.
    """
    return _BLANKS.sub(" ", line).rstrip()


def split_lines(text: str) -> list[str]:
    """Return the lines of *text*, each with its ``\\n`` (the last one without when it has none).

    Splits on ``\\n`` only (``str.splitlines`` would also split on a form
    feed or a lone ``\\r``, which are part of a line here), so the line
    numbers agree with those of a numbered listing; ``""`` has no lines.
    """
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def _leading_whitespace(line: str) -> str:
    """Return the blanks *line* starts with."""
    return line[: len(line) - len(line.lstrip(" \t"))]


def _extra_indent(file_lines: list[str], old_lines: list[str]) -> str:
    """Return the indentation every non-blank file line has beyond its old line, or ``""``.

    ``""`` as well when the lines do not agree on one prefix, or when an
    old line is indented deeper than its file line: the replacement is
    then applied as written.
    """
    extra: str | None = None
    for file_line, old_line in zip(file_lines, old_lines, strict=True):
        if not file_line.strip():
            continue
        file_ws, old_ws = _leading_whitespace(file_line), _leading_whitespace(old_line)
        if not file_ws.endswith(old_ws):
            return ""
        prefix = file_ws[: len(file_ws) - len(old_ws)]
        if extra is None:
            extra = prefix
        elif prefix != extra:
            return ""
    return extra or ""


def tolerant_matches(text: str, old: str) -> list[TolerantMatch]:
    """Return every place *old* matches *text* line by line, whitespace aside.

    A match is a run of as many lines of *text* as *old* has whose
    :func:`normalize_line` forms equal the old string's, in order.  Its
    span is the original text of those lines, the last line's terminator
    included only when *old* ends with one (so ``old_string="x = 1"``
    keeps the file line's newline, as the verbatim match would).

    Args:
        text: The file's text.
        old: The ``old_string`` that was not found verbatim.

    Returns:
        The matches in file order: none, one (apply it), or several (ask
        for more context).  Empty when *old* is empty.
    """
    old_lines = split_lines(old)
    if not old_lines:
        return []
    file_lines = split_lines(text)
    wanted = [normalize_line(line) for line in old_lines]
    normalized = [normalize_line(line) for line in file_lines]
    offsets = [0]
    for line in file_lines:
        offsets.append(offsets[-1] + len(line))
    matches = []
    count = len(wanted)
    for index in range(len(file_lines) - count + 1):
        if normalized[index : index + count] != wanted:
            continue
        window = file_lines[index : index + count]
        end = offsets[index + count]
        if not old.endswith("\n"):
            end -= len(window[-1]) - len(window[-1].rstrip("\r\n"))
        matches.append(TolerantMatch(offsets[index], end, _extra_indent(window, old_lines)))
    return matches


def reindent(new: str, indent: str) -> str:
    """Return *new* with *indent* before each of its non-blank lines.

    Blank lines stay blank; ``indent=""`` returns *new* as it is.
    """
    if not indent:
        return new
    return "".join(
        indent + line if line.strip() else line for line in split_lines(new)
    )


def nearest_region(text: str, old: str) -> tuple[int, int, str] | None:
    """Return the window of *text* most like *old*: its 1-based first and last line, its text.

    The windows are those of as many lines as *old* has (the whole text
    when it is shorter), compared after :func:`normalize_line` with
    :class:`difflib.SequenceMatcher`; ``quick_ratio`` (an upper bound)
    is checked before the real ratio, as ``difflib.get_close_matches``
    does, so a long file is scanned in reasonable time.  The first of
    equally close windows wins.

    Args:
        text: The file's text.
        old: The ``old_string`` that was not found.

    Returns:
        ``(first, last, region)`` or ``None`` when *text* or *old* has no
        lines.
    """
    old_lines = split_lines(old)
    file_lines = split_lines(text)
    if not old_lines or not file_lines:
        return None
    normalized = [normalize_line(line) for line in file_lines]
    count = min(len(old_lines), len(file_lines))
    wanted = "\n".join(normalize_line(line) for line in old_lines)
    matcher = difflib.SequenceMatcher(None, "", wanted)
    best, best_ratio = 0, -1.0
    for index in range(len(file_lines) - count + 1):
        matcher.set_seq1("\n".join(normalized[index : index + count]))
        if matcher.quick_ratio() <= best_ratio:
            continue
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best, best_ratio = index, ratio
    return best + 1, best + count, "".join(file_lines[best : best + count])
