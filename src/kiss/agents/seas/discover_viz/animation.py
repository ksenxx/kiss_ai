# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The HTML animation of a ``/discover`` story.

:func:`render` fills ``animation.html``, a page with no dependency, with
the story, the frames of :func:`kiss.agents.seas.discover_viz.story.replay`
and the tree layout of :func:`layout`.  Everything that can be wrong is
decided here, in Python; the page only draws the frame it is told to
show, so the same story always gives the same pictures.
"""

from __future__ import annotations

import html
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

from kiss.agents.seas.discover_viz.story import Round, Story

COLUMN_WIDTH = 120
"""Pixels between two neighbouring points of one generation."""

ROW_HEIGHT = 120
"""Pixels between a generation and the next."""

MARGIN = 80
"""Pixels between the outermost points and the edge of the drawing."""

RADIUS = 24
"""Pixels from the centre of a point to its rim."""

CLEARANCE = RADIUS + 8
"""Half the height of the band around a row that an edge crosses straight down, in pixels."""

type _Place = tuple[float, float]

_TEMPLATE = Path(__file__).with_name("animation.html")


def _columns(rounds: list[Round]) -> dict[str, float]:
    """Return the column of every point of the tree its first parents make.

    A childless point takes the next free column, in the order
    evaluated; a parent stands midway between its first and its last
    child.  Subtrees thus occupy disjoint column ranges, so two points
    of one row are at least one column apart.
    """
    children: dict[str, list[str]] = {}
    roots: list[str] = []
    for round_ in rounds:
        siblings = children.setdefault(round_.parents[0], []) if round_.parents else roots
        siblings.append(round_.id)
    columns: dict[str, float] = {}
    free = 0
    pending = [(root, False) for root in reversed(roots)]
    while pending:
        point, placed_children = pending.pop()
        kids = children.get(point, [])
        if not kids:
            columns[point] = free
            free += 1
        elif placed_children:
            columns[point] = (columns[kids[0]] + columns[kids[-1]]) / 2
        else:
            pending.append((point, True))
            pending.extend((kid, False) for kid in reversed(kids))
    return columns


def layout(rounds: list[Round]) -> dict[str, tuple[float, int]]:
    """Return ``id -> (column, row)`` for the points of *rounds*, a tree drawn top down.

    The row of a point is one below its lowest parent, so every edge,
    also the one from the second point of a compose, runs downwards.

    Args:
        rounds: The rounds of a story :func:`~kiss.agents.seas.discover_viz.story.replay`
            found sound: ids are unique and every parent is an earlier point.
    """
    columns = _columns(rounds)
    rows: dict[str, int] = {}
    for round_ in rounds:
        rows[round_.id] = 1 + max((rows[parent] for parent in round_.parents), default=-1)
    return {round_.id: (columns[round_.id], rows[round_.id]) for round_ in rounds}


def _crossing(row: list[float], wanted: float) -> float:
    """Return the x nearest to *wanted* where an edge can cross a row with points at *row*.

    That is *wanted* itself when it is half a column or more from every
    point of the row, else the nearest gap of the row: midway between
    two neighbours, or half a column outside the outermost point.
    """
    half = COLUMN_WIDTH / 2
    if all(abs(wanted - x) >= half for x in row):
        return wanted
    xs = sorted(row)
    gaps = [xs[0] - half, *((left + right) / 2 for left, right in pairwise(xs)), xs[-1] + half]
    return min((abs(gap - wanted), gap) for gap in gaps)[1]


def _curve(start: _Place, end: _Place) -> str:
    """Return the SVG curve from *start* to *end* that leaves and arrives straight down."""
    middle = (start[1] + end[1]) / 2
    return f"C{start[0]:g},{middle:g} {end[0]:g},{middle:g} {end[0]:g},{end[1]:g}"


def edge_path(start: _Place, end: _Place, rows: dict[float, list[float]]) -> str:
    """Return the SVG path of the edge from the point at *start* down to the point at *end*.

    The path leaves the upper point at its bottom and enters the lower
    one at its top.  It bends only between two rows, where no point is.
    An edge that spans several rows (a compose whose two points are on
    different rows) crosses each row in between straight down, through
    the gap (:func:`_crossing`) nearest to the straight line, so it
    never looks attached to a point it only passes.

    Args:
        start: The centre of the upper point.
        end: The centre of the lower point.
        rows: ``y -> [x, ...]`` of the points of every row.
    """
    (x1, y1), (x2, y2) = start, end
    at = (x1, y1 + RADIUS)
    path = [f"M{at[0]:g},{at[1]:g}"]
    for y in sorted(row for row in rows if y1 < row < y2):
        x = _crossing(rows[y], x1 + (x2 - x1) * (y - y1) / (y2 - y1))
        path += [_curve(at, (x, y - CLEARANCE)), f"L{x:g},{y + CLEARANCE:g}"]
        at = (x, y + CLEARANCE)
    path.append(_curve(at, (x2, y2 - RADIUS)))
    return " ".join(path)


def page_data(story: Story, frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Return what the page draws: the story, its *frames* and the position of every point.

    A point carries its tooltip: its ``details`` and the ``source``
    file they were copied from.  An edge carries its SVG ``path``
    (:func:`edge_path`) and whether it comes from the ``merged`` point
    of a compose.  A round's ``number`` counts the rounds after the
    baseline, whose own number is 0.
    """
    centres = {
        point: (MARGIN + column * COLUMN_WIDTH, float(MARGIN + row * ROW_HEIGHT))
        for point, (column, row) in layout(story.rounds).items()
    }
    rows: dict[float, list[float]] = {}
    for x, y in centres.values():
        rows.setdefault(y, []).append(x)
    points, edges, rounds = [], [], []
    number = 0
    for round_ in story.rounds:
        x, y = centres[round_.id]
        number += round_.operator != "baseline"
        points.append({
            "id": round_.id,
            "x": x,
            "y": y,
            "details": list(round_.details.items()),
            "source": round_.source,
        })
        edges += [
            {
                "to": round_.id,
                "merged": index > 0,
                "path": edge_path(centres[parent], (x, y), rows),
            }
            for index, parent in enumerate(round_.parents)
        ]
        rounds.append(round_.model_dump(exclude={"details", "source"}) | {
            "number": 0 if round_.operator == "baseline" else number,
        })
    return {
        "task_id": story.task_id,
        "title": story.title,
        "goal": story.goal,
        "metric": story.metric,
        "width": MARGIN + max(x for x, _y in centres.values()),
        "height": MARGIN + max(rows),
        "radius": RADIUS,
        "total": number,
        "points": points,
        "edges": edges,
        "rounds": rounds,
        "finish": story.finish.model_dump() if story.finish is not None else None,
        "frames": frames,
    }


def render(story: Story, frames: list[dict[str, Any]]) -> str:
    """Return the self-contained HTML page that animates *story* through *frames*.

    Args:
        story: A story with no error.
        frames: Its frames (:func:`kiss.agents.seas.discover_viz.story.replay`).
    """
    # "<" is escaped so that no text of the run can close the script element.
    data = json.dumps(page_data(story, frames), ensure_ascii=False).replace("<", "\\u003c")
    page = _TEMPLATE.read_text(encoding="utf-8")
    return page.replace("__TITLE__", html.escape(story.title)).replace("__DATA__", data)
