# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The story of a ``/discover`` run: what the model writes, checked and replayed.

The model reads the evidence of a run
(:mod:`kiss.agents.seas.discover_viz.trajectory`) and writes a
:class:`Story`: one :class:`Round` per evaluated point in the order
evaluated (the baseline first), each with its parents, how its idea
was picked, its status and the points it pushed off the frontier, then
how the run finished.  Three checks stand between that text and the
animation:

* :func:`parse` refuses a malformed story, naming each field;
* :func:`replay` walks the rounds as the loop's ``ROUND`` and
  ``FINISH`` do and refuses frontier bookkeeping the loop cannot
  produce (a parent that was never evaluated or never entered the
  frontier, an eviction of a point that is not on the frontier, a
  promotion the loop does not make, an empty frontier, a best point
  that is off it); what it returns for a sound story are the animation
  frames;
* :func:`quote_errors` refuses a value the run did not record: every
  tooltip detail, score and draw must stand in the evidence as written,
  and so must every decimal number of the sentences the model worded,
  none of them as a piece of a longer number.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kiss.agents.seas.discover.discover_sea import DISCOVERY_INSTRUCTIONS

_STATE_OF = {
    "KEPT": "frontier", "TIE": "tie", "FAILED": "failed",
    "DOMINATED": "dominated", "OVERFIT": "overfit",
}
"""The state a point is drawn in for each status of the loop."""

_CHANGE_RULE = {
    "DOMINATED": "the loop takes only a frontier point off the frontier as dominated",
    "OVERFIT": "the loop runs the held-out check on frontier and TIE points only",
    "KEPT": "the loop puts back the baseline and promotes a TIE point, nothing else",
}
"""Why a :class:`Change` is refused, per status it asks for."""

_NUMBER = re.compile(r"\d+(?:\.\d+)+")
_EMPHASIS = re.compile(r"[`*]")


class _Strict(BaseModel):
    """A model that refuses unknown keys, so a misspelt field is reported, not dropped."""

    model_config = ConfigDict(extra="forbid")


class Change(_Strict):
    """A status change of a point after its own round, as the loop makes them.

    ``DOMINATED`` takes a frontier point off the frontier; ``OVERFIT``
    does so too and also marks a ``TIE`` point that failed the held-out
    check; ``KEPT`` puts a point on the frontier: the baseline when it
    is off it, or a ``TIE`` point (the one the run reports as its best).
    """

    id: str
    status: Literal["DOMINATED", "OVERFIT", "KEPT"]
    why: str = Field(min_length=1)


class Round(_Strict):
    """One evaluated point: where it came from, how it was picked and what became of it."""

    id: str = Field(min_length=1)
    parents: list[str]
    idea: str = Field(min_length=1)
    operator: Literal["baseline", "initial", "variant", "compose", "fork", "fresh"]
    draw: str = ""
    picked: str = Field(min_length=1)
    score: str
    status: Literal["KEPT", "TIE", "FAILED"]
    outcome: str = Field(min_length=1)
    details: dict[str, str] = Field(min_length=1)
    source: Literal["pareto.md", "explored-ideas.md"]
    changes: list[Change] = []


class Finish(_Strict):
    """How the run ended: the point it reports and the last status changes."""

    best: str
    reason: str = Field(min_length=1)
    changes: list[Change] = []


class Story(_Strict):
    """A ``/discover`` run as ``/discover_viz`` animates it."""

    task_id: str
    title: str = Field(min_length=1, max_length=80)
    goal: str = Field(min_length=1)
    metric: str = Field(min_length=1, max_length=60)
    rounds: list[Round] = Field(min_length=1)
    finish: Finish | None = None


def _location(loc: tuple[int | str, ...]) -> str:
    """Return a pydantic error location as ``rounds[3].status``."""
    path = "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in loc)
    return path.lstrip(".") or "story"


def parse(data: object) -> Story | list[str]:
    """Return *data* (decoded JSON) as a :class:`Story`, or what is malformed in it.

    Returns:
        The story, or one ``<field>: <what is wrong>`` line per problem.
    """
    try:
        return Story.model_validate(data)
    except ValidationError as error:
        return [f"{_location(e['loc'])}: {e['msg']}" for e in error.errors()]


def _round_errors(
    round_: Round, states: dict[str, str], entered: set[str], where: str
) -> list[str]:
    """Return what makes *round_* impossible after the rounds that left *states*.

    *entered* holds the points that have been on the frontier.  A parent
    must be one of them, not on the frontier at this very round: rounds
    that ran at the same time are logged one after the other, and the
    first may have dominated the parent of the second.
    """
    errors = []
    if round_.id in states:
        errors.append(f"{where}.id: {round_.id!r} is already the id of an earlier round")
    unknown = [parent for parent in round_.parents if parent not in states]
    if unknown:
        errors.append(
            f"{where}.parents: {', '.join(unknown)} not evaluated in an earlier round "
            "(a point comes after its parents)"
        )
    never = [parent for parent in round_.parents if parent in states and parent not in entered]
    if never:
        errors.append(
            f"{where}.parents: {', '.join(never)} never entered the frontier, and the loop "
            "builds only on frontier points"
        )
    if (round_.operator == "baseline") != (not round_.parents):
        errors.append(f"{where}: the baseline has no parents and every other point has one")
    if (round_.operator == "compose") != (len(round_.parents) > 1):
        errors.append(
            f"{where}: a compose names both of its points, the parent first, "
            "and only a compose has two"
        )
    if round_.status != "KEPT" and "frontier" not in states.values():
        errors.append(f"{where}.status: {round_.status} needs a frontier point, and there is none")
    return errors


def _apply(states: dict[str, str], change: Change, where: str) -> list[str]:
    """Apply *change* to *states*; return the error when the loop does not make it."""
    state = states.get(change.id)
    if state is None:
        return [f"{where}: {change.id!r} is not a point evaluated so far"]
    baseline = change.id == next(iter(states))
    allowed = {
        "DOMINATED": state == "frontier",
        "OVERFIT": state in ("frontier", "tie"),
        "KEPT": state == "tie" or (baseline and state != "frontier"),
    }
    if not allowed[change.status]:
        return [
            f"{where}: {change.id} is {state} here, so it cannot become {change.status} "
            f"({_CHANGE_RULE[change.status]})"
        ]
    states[change.id] = _STATE_OF[change.status]
    return []


def replay(story: Story) -> tuple[list[dict[str, Any]], list[str]]:
    """Replay *story* as the loop ran it: return its animation frames and its bookkeeping errors.

    Every round gives two frames: ``pick`` (the point is a candidate
    beside the points of the rounds before) and ``result`` (it has its
    status and the round's changes are applied).  A finish gives a last
    ``finish`` frame with the best point.  A frame is ``{"round": index
    or None, "phase": ..., "states": {id: state}}`` (plus ``"best"`` in
    the finish frame), a state one of ``candidate``, ``frontier``,
    ``tie``, ``failed``, ``dominated``, ``overfit``.

    Returns:
        ``(frames, errors)``; the frames are meaningful only when there
        are no errors.
    """
    states: dict[str, str] = {}
    entered: set[str] = set()
    frames: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, round_ in enumerate(story.rounds):
        where = f"rounds[{index}]"
        errors += _round_errors(round_, states, entered, where)
        frames.append({
            "round": index, "phase": "pick", "states": states | {round_.id: "candidate"},
        })
        states[round_.id] = _STATE_OF[round_.status]
        entered.update(frontier(states))
        for number, change in enumerate(round_.changes):
            errors += _apply(states, change, f"{where}.changes[{number}]")
            entered.update(frontier(states))
        if "frontier" not in states.values():
            errors.append(
                f"{where}: the frontier is empty after this round; the loop puts the baseline "
                "back (a KEPT change)"
            )
        frames.append({"round": index, "phase": "result", "states": dict(states)})
    finish = story.finish
    if finish is not None:
        for number, change in enumerate(finish.changes):
            errors += _apply(states, change, f"finish.changes[{number}]")
        if states.get(finish.best) != "frontier":
            errors.append(
                f"finish.best: {finish.best!r} is not on the frontier at the end "
                f"({', '.join(frontier(states)) or 'empty'})"
            )
        frames.append({
            "round": None, "phase": "finish", "states": dict(states), "best": finish.best,
        })
    return frames, errors


def frontier(states: dict[str, str]) -> list[str]:
    """Return the ids on the frontier in *states* (a frame's ``states``), oldest first."""
    return [point for point, state in states.items() if state == "frontier"]


def _copied(story: Story) -> Iterator[tuple[str, str]]:
    """Yield ``(location, value)`` for every value of *story* copied from the run's files."""
    for index, round_ in enumerate(story.rounds):
        yield f"rounds[{index}].score", round_.score
        yield f"rounds[{index}].draw", round_.draw
        for key, value in round_.details.items():
            yield f"rounds[{index}].details[{key!r}]", value


def _worded(story: Story) -> Iterator[tuple[str, str]]:
    """Yield ``(location, text)`` for every sentence of *story* the model worded itself."""
    yield "title", story.title
    yield "goal", story.goal
    yield "metric", story.metric
    for index, round_ in enumerate(story.rounds):
        yield f"rounds[{index}].idea", round_.idea
        yield f"rounds[{index}].picked", round_.picked
        yield f"rounds[{index}].outcome", round_.outcome
        for number, change in enumerate(round_.changes):
            yield f"rounds[{index}].changes[{number}].why", change.why
    if story.finish is not None:
        yield "finish.reason", story.finish.reason
        for number, change in enumerate(story.finish.changes):
            yield f"finish.changes[{number}].why", change.why


def _plain(text: str) -> str:
    """Return *text* without Markdown emphasis marks and with every blank run as one space."""
    return " ".join(_EMPHASIS.sub("", text).split())


def _stands(value: str, text: str) -> bool:
    """Return whether *value* stands in *text*, and not as a piece of a longer number.

    ``9.99`` does not stand in ``19.99``, nor ``07186`` or ``2.07`` in
    ``2.07186``: a value that begins or ends with a digit must begin or
    end where the number in *text* does.
    """
    before = r"(?<![\d.])" if value[:1].isdigit() else ""
    after = r"(?!\.?\d)" if value[-1:].isdigit() else ""
    return re.search(f"{before}{re.escape(value)}{after}", text) is not None


def quote_errors(story: Story, evidence: str) -> list[str]:
    """Return an error for every value of *story* that the run did not record.

    A copied value (a tooltip detail, a score, a draw) must stand in
    *evidence* as written, Markdown emphasis and line breaks aside.  A
    decimal number in a worded sentence must stand there too, or in the
    loop's own definition (its draw thresholds).  Neither may be a piece
    of a longer number (:func:`_stands`).

    Args:
        story: The story.
        evidence: The evidence text of the story's run
            (:func:`kiss.agents.seas.discover_viz.trajectory.evidence`).
    """
    recorded = _plain(evidence)
    numbers = recorded + " " + _plain(DISCOVERY_INSTRUCTIONS)
    errors = [
        f"{where}: {value!r} is not in the evidence; copy it as the run wrote it"
        for where, value in _copied(story) if not _stands(_plain(value), recorded)
    ]
    for where, text in _worded(story):
        missing = [number for number in _NUMBER.findall(text) if not _stands(number, numbers)]
        if missing:
            errors.append(
                f"{where}: {', '.join(missing)} not in the evidence; quote numbers as the "
                "run recorded them and compute none"
            )
    return errors
