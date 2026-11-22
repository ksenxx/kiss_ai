# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The timeline of a task's sub-agents as an SVG Gantt chart (:func:`render`).

One thin bar per sub-agent, in the order they started, on a time axis
that starts when the task started; the bars are coloured by role, and
pale when the history records no end for the sub-agent.  A dashed
line marks a task update or an ``/ask`` question.  A shaded band is a
:class:`~kiss.agents.seas.timeline.history.Wait` and a solid line is
a message the user sent; each carries a number that a note under the
chart explains, and the numbers of marks that fall close together
are stacked, so no text hides another and no line crosses one.

The chart cannot measure the font of the viewer.  It computes the
width of every text (:func:`_width`) and writes it into the SVG
(:func:`_text`), which binds the viewer to it: a text in another
font or of unusual glyphs is squeezed or stretched to that width.
One of ASCII in Helvetica, Arial or Liberation Sans keeps its own
width, less what the font kerns: a pair such as ``Ta``, ``AV`` or
``11`` is set about a pixel closer than its two advances, and the
text is stretched by that much.  The layout works with the same
widths, so no text passes the right edge, whatever the history
holds.  A viewer that ignores ``textLength``, as librsvg does, draws
the widths of its own font instead.  Nothing binds the height of a
text: the lines of text are 14 or 16 px apart, which holds the 11 px
over and the 3 px under the baseline that these fonts take, and of
the marks a font stacks on one character a text keeps two
(:func:`_fit`).
"""

from __future__ import annotations

import math
from html import escape
from itertools import accumulate
from unicodedata import category, east_asian_width

from kiss.agents.seas.timeline.history import MERGE, WAIT_MINUTES, Run

PALETTE = (
    "#4e79a7", "#e15759", "#59a14f", "#f28e2b", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7",
)
"""The bar colours, given to the roles in the order their first sub-agent started."""

MERGE_COLOUR = "#9c755f"

LEFT = 70
"""Where the time axis starts; the chart is :data:`WIDTH` wide from there."""
WIDTH = 900
EDGE = LEFT + WIDTH + 26
"""No text ends right of this; the SVG is 4 wider, for the ink a last glyph hangs over."""
ROWS_HEIGHT = 460
"""The bars share this height: each is 2 to 18 high."""
LINE = 16
"""The distance between two lines of text under the chart."""
FONT_SIZE = 12
"""Of every text, in px."""
MARKS = 2
"""How many combining marks in a row a text keeps: fonts stack them, and a third would
reach the line above."""

_ASCII = (
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
    1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
    333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
    556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
)
"""The advance widths of Helvetica for the 95 printable ASCII characters, the space to ``~``,
in thousandths of an em: the values of Adobe's ``Helvetica.afm``.  Chromium measures those
of Liberation Sans, which has the metrics of Arial, within a quarter of a thousandth of them."""

_MARK = ("Mn", "Me")
"""The Unicode categories of a mark drawn on the character before it."""

_TICK_MINUTES = (1, 2, 5, 10, 15, 30, 60, 120, 180, 240, 360, 720, 1440)


def _advance(ch: str) -> int:
    """Return the advance width of the character *ch*, in thousandths of an em.

    Known for printable ASCII (:data:`_ASCII`) and for the ellipsis a
    cut text ends with, an em; stated for the rest: nothing for a mark
    drawn on the character before it, an em for a wide character (CJK,
    most emoji) and the advance of most lowercase letters for any
    other.
    """
    if " " <= ch <= "~":
        return _ASCII[ord(ch) - 32]
    if category(ch) in _MARK:
        return 0
    return 1000 if ch == "…" or east_asian_width(ch) in ("W", "F") else 556


def _width(text: str) -> float:
    """Return the width of *text* in px: the sum of its advances at :data:`FONT_SIZE`."""
    return sum(map(_advance, text)) * FONT_SIZE / 1000


def _fit(text: str, room: float) -> str:
    """Return *text* on one line, cut with an ellipsis to at most *room* px.

    Left out are the characters that do not print and, of the marks
    on one character, those after the first :data:`MARKS`.
    """
    shown = []
    marks = 0  # in a row, up to the character at hand
    for ch in text:
        if ch.isprintable() or ch.isspace():
            marks = marks + 1 if category(ch) in _MARK else 0
            if marks <= MARKS:
                shown.append(ch)
    text = " ".join("".join(shown).split())
    if _width(text) <= room:
        return text
    limit = room * 1000 / FONT_SIZE - _advance("…")
    kept = sum(units <= limit for units in accumulate(map(_advance, text)))
    return text[:kept] + "…"


def _text(x: float, y: float, text: str) -> str:
    """Return the ``<text>`` of *text* at (*x*, *y*), bound to its width.

    ``textLength`` makes the viewer draw the text as wide as the
    layout takes it to be (:func:`_width`), and never past
    :data:`EDGE`.
    """
    x = round(x, 1)  # as it is written, so that what is left before EDGE is exact
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" textLength="{min(_width(text), EDGE - x):.1f}" '
        f'lengthAdjust="spacingAndGlyphs">{escape(text)}</text>'
    )


def _x(run: Run, minute: float) -> float:
    """Return where *minute* is on the axis of *run*: at the end next to it when outside the run."""
    return LEFT + min(max(minute, 0.0), run.minutes) * WIDTH / run.minutes


def _colours(run: Run) -> dict[str, str]:
    """Return the colour of each role of *run*, in the order the roles first appear."""
    colours: dict[str, str] = {}
    for agent in run.agents:
        if agent.role not in colours:
            taken = len(colours) - (MERGE in colours)
            colours[agent.role] = MERGE_COLOUR if agent.role == MERGE else PALETTE[taken]
    return colours


def _lanes(labels: list[tuple[float, str]]) -> list[int]:
    """Return a lane for each of *labels* (x, text), given left to right.

    A label goes under every earlier one that reaches its x.  So no
    two overlap, and the line that hangs from a label, just left of
    it, crosses none.
    """
    ends: list[float] = []  # of the last label of each lane
    lanes = []
    for x, text in labels:
        lane = max((lane + 1 for lane, end in enumerate(ends) if end > x), default=0)
        ends[lane:] = [x + _width(text) + 4]
        lanes.append(lane)
    return lanes


def _ticks(minutes: float, bottom: float) -> list[str]:
    """Return the ticks of a *minutes* long axis at height *bottom*: at most a dozen."""
    step = next(
        (s for s in _TICK_MINUTES if minutes / s <= 10), 1440 * math.ceil(minutes / 14400)
    )
    ticks = []
    for minute in range(0, int(minutes) + 1, step):
        x = LEFT + minute / minutes * WIDTH
        label = f"{minute // 60} h" if step >= 60 else f"{minute} min"
        ticks.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{bottom:.1f}" y2="{bottom + 6:.1f}" '
            f'stroke="#333"/>' + _text(x - _width(label) / 2, bottom + 18, label)
        )
    return ticks


def _legend(entries: list[tuple[str, str]], y: float) -> tuple[list[str], float]:
    """Lay *entries* (swatch colour or ``""``, label) out in lines that start at height *y*.

    A label is cut to what a line holds, and an entry that would pass
    the end of the axis starts the next line.

    Returns:
        The SVG elements and the height of the last line.
    """
    body = []
    x = float(LEFT)
    for colour, label in entries:
        swatch = 14 if colour else 0
        label = _fit(label, WIDTH - swatch)
        width = swatch + _width(label)
        if x + width > LEFT + WIDTH:
            x, y = LEFT, y + LINE
        if colour:
            body.append(
                f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="10" height="10" fill="{colour}"/>'
            )
        body.append(_text(x + swatch, y, label))
        x += width + 16
    return body, y


def render(run: Run) -> str:
    """Return the SVG timeline of *run*; every agent of it has its role.

    Args:
        run: The run, with at least one agent and at most
            ``len(PALETTE)`` roles beside the merge resolver's.
    """
    scale = WIDTH / run.minutes
    colours = _colours(run)
    marks = sorted(
        [(wait.start, wait.end, f"{wait.end - wait.start:.0f} min in {wait.what}")
         for wait in run.waits]
        + [(at, at, f"user input: {text}") for at, text in run.inputs]
    )
    labels = []
    notes = []
    for number, (start, end, note) in enumerate(marks, 1):
        x = _x(run, start) + 3
        label = f"{number}: {end - start:.0f} min" if end > start else str(number)
        if x + _width(label) > EDGE:
            label = str(number)  # all that fits before the right edge
        labels.append((x, label))
        notes.append(_fit(f"{number}: {note}", WIDTH))
    lanes = _lanes(labels)
    top = 24 + 14 * (max(lanes, default=-1) + 1)  # of the first bar, under the stacked labels
    row = min(18.0, max(2.0, ROWS_HEIGHT / len(run.agents)))
    bottom = top + len(run.agents) * row + 16
    count = len(run.agents)
    title = f"Task {run.task_id[:8]}: {count} sub-agent{'s' * (count != 1)}, one row each"
    body = [_text(LEFT, 14, _fit(title + ", in start order", WIDTH))]
    over = []  # drawn over the bars, as the bands are drawn under them
    for (start, end, _), (label_x, label), lane in zip(marks, labels, lanes):
        x = _x(run, start)
        y = 30 + 14 * lane
        if end > start:
            body.append(
                f'<rect x="{x:.1f}" y="{y - 11}" width="{_x(run, end) - x:.1f}" '
                f'height="{bottom - y + 11:.1f}" fill="#f4c7c3" opacity="0.7"/>'
            )
            over.append(f'<g fill="#a33">{_text(label_x, y, label)}</g>')
        else:
            over.append(
                f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y - 11}" y2="{bottom:.1f}" stroke="#000"/>'
                + _text(label_x, y, label)
            )
    for index, agent in enumerate(run.agents):
        x = LEFT + agent.start * scale
        body.append(
            f'<rect x="{x:.1f}" y="{top + index * row:.1f}" '
            f'width="{max((agent.end - agent.start) * scale, 1):.1f}" '
            f'height="{row - 0.6:.1f}" fill="{colours[agent.role]}"'
            + ("/>" if agent.ended else ' opacity="0.4"/>')
        )
    for at in run.queries:
        x = LEFT + at * scale
        body.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 4}" y2="{bottom:.1f}" stroke="#888" '
            'stroke-dasharray="2,3"/>'
        )
    body += over + _ticks(run.minutes, bottom)
    roles = [agent.role for agent in run.agents]
    entries = [(colour, f"{role} ({roles.count(role)})") for role, colour in colours.items()]
    unended = sum(not agent.ended for agent in run.agents)
    if unended:
        entries.append(("", f"pale: no end recorded, drawn to the end of the run ({unended})"))
    if run.queries:
        entries.append(("", f"dashed: task updates and /ask questions ({len(run.queries)})"))
    if run.waits:
        entries.append((
            "#f4c7c3",
            f"over {WAIT_MINUTES} min in one tool call of the task while no sub-agent ran",
        ))
    legend, last = _legend(entries, bottom + 38)
    body += legend
    for note in notes:
        last += LINE
        body.append(_text(LEFT, last, note))
    width, height = EDGE + 4, math.ceil(last + 8)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="{FONT_SIZE}"><rect width="{width}" height="{height}" fill="white"/>'
        + "".join(body)
        + "</svg>\n"
    )
