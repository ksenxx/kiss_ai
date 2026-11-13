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
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import accumulate

_BLANKS = re.compile(r"[ \t]+")
_WORD = re.compile(r"\w+")


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


_BUDGET = 200_000
"""What :func:`nearest_region` spends at most, on each: the characters of the windows it
compares with the old string, and the places of shared words it counts to choose them."""


def _closest(texts: Iterable[str], wanted: str) -> int:
    """Return the index of the one of *texts* most like *wanted*, the first of equally close ones.

    Compared with :class:`difflib.SequenceMatcher`; ``quick_ratio`` (an
    upper bound) is checked before the real ratio, as
    ``difflib.get_close_matches`` does.
    """
    matcher = difflib.SequenceMatcher(None, "", wanted)
    best, best_ratio = 0, -1.0
    for index, text in enumerate(texts):
        matcher.set_seq1(text)
        if matcher.quick_ratio() <= best_ratio:
            continue
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best, best_ratio = index, ratio
    return best


def _likely_starts(normalized: list[str], wanted: list[str], last: int) -> list[int]:
    """Return the starts of the windows of *normalized* sharing words with *wanted*, line by line.

    A window's weight is the length of the words its lines hold of the
    line of *wanted* at the same place; a start before the first window
    or past the *last* one counts as that window.  The rarest words are
    counted first and the counting stops at :data:`_BUDGET` places, so
    a word on every line costs nothing.  The window sharing most comes
    first, the earlier of equals.
    """
    places: dict[str, list[int]] = {}
    for place, line in enumerate(wanted):
        for word in set(_WORD.findall(line)):
            places.setdefault(word, []).append(place)
    lines: dict[str, list[int]] = {}
    for index, line in enumerate(normalized):
        for word in places.keys() & _WORD.findall(line):
            lines.setdefault(word, []).append(index)
    shared: dict[int, int] = {}
    spent = 0
    costs = sorted((len(places[word]) * len(found), word) for word, found in lines.items())
    for cost, word in costs:
        spent += cost
        if spent > _BUDGET:
            break
        for place in places[word]:
            for index in lines[word]:
                start = min(max(index - place, 0), last)
                shared[start] = shared.get(start, 0) + len(word)
    return [start for _, start in sorted((-length, start) for start, length in shared.items())]


def _within_budget(starts: Iterable[int], ends: list[int], count: int) -> list[int]:
    """Return the first of *starts* whose windows of *count* lines hold :data:`_BUDGET` characters.

    ``ends[i]`` is the length of the lines before line ``i``.  The
    first window is in whatever its size.
    """
    chosen: list[int] = []
    spent = 0
    for start in starts:
        spent += ends[start + count] - ends[start]
        if chosen and spent > _BUDGET:
            break
        chosen.append(start)
    return chosen


def nearest_region(text: str, old: str) -> tuple[int, int, str] | None:
    """Return the window of *text* most like *old*: its 1-based first and last line, its text.

    The windows are those of as many lines as *old* has (the whole text
    when it is shorter), compared after :func:`normalize_line` with
    :class:`difflib.SequenceMatcher`: every window, while their text
    fits :data:`_BUDGET`.  Comparing every window of a long file took
    minutes (110 s for 50 lines against 20,000), so there only the
    likely ones are compared (:func:`_likely_starts`; the likeliest
    that fit the budget), or the first windows when none is likely.
    The first of equally close windows wins, and a single window is the
    answer as it is.

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
    wanted = [normalize_line(line) for line in old_lines]
    count = min(len(wanted), len(normalized))
    last = len(normalized) - count
    ends = list(accumulate((len(line) + 1 for line in normalized), initial=0))
    starts = _within_budget(range(last + 1), ends, count)
    if len(starts) <= last:
        likely = _likely_starts(normalized, wanted, last)
        starts = sorted(_within_budget(likely, ends, count)) or starts
    best = starts[0]
    if len(starts) > 1:
        windows = ("\n".join(normalized[start : start + count]) for start in starts)
        best = starts[_closest(windows, "\n".join(wanted))]
    return best + 1, best + count, "".join(file_lines[best : best + count])
