# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled ``/timeline`` agent
(:mod:`kiss.agents.seas.timeline.timeline_sea`).

Each test writes tasks, their events and their sub-agent rows to the
session's SQLite history, calls the SEA's two tools as the model would
and reads the SVG file they wrote; one test runs the agent itself
against a scripted model.  No mocks.

The toy run of :func:`toy` is a 100-minute task whose last sub-agent
ends at minute 104.  Minute by minute: two auditors (2 to 20), a
message of the user (21.5), a wait on a background job (22 to 40, with
three more messages in it), a shell here-document (41 to 60), a fixer
(62 to 70), a question to the user (71 to 83), a wait on a reviewer
that only ran from 95 to 97, a fixer with no recorded end (98), a late
reviewer (99 to 104) and the merge resolver (99.5); a task update at
10 and an ``/ask`` at 31.  A tool call and a message whose events hold
no time are not drawn.  The fixers and the reviewers are told apart
by the letters of one word with a number in it, as ``FIX2.md``.
"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from kiss.agents.seas.timeline import timeline_sea as sea
from kiss.agents.seas.timeline.timeline_sea import TimelineSea, draw_timeline, timeline_agents
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.merge_conflict_resolver import conflict_task
from kiss.agents.sorcar.persistence import _add_task
from kiss.tests.agents.seas.conftest import persist_task, registered, run_sea
from kiss.tests.agents.seas.sea_contract import (
    assert_worker_contract,
    system_message,
    tool_names,
    tool_result_text,
    tool_results,
)
from kiss.tests.agents.sorcar.local_model_server import finish_body, tool_call_body

pytestmark = pytest.mark.usefixtures("empty_history")

MINUTE = 60_000
SVG = "{http://www.w3.org/2000/svg}"
BAND = "#f4c7c3"
BLUE, RED, GREEN, ORANGE, BROWN = "#4e79a7", "#e15759", "#59a14f", "#f28e2b", "#9c755f"
_WAITS = "Waits of over 10 min in one tool call with no sub-agent running"
_LISTING = "How the task texts of the sub-agents open, every number written as #:"
_NO_TASK = "Pass a task id or the first six or more characters of one."
_QUESTION = "Which of the two designs do you prefer: " + "the one with the queue, " * 5 + "or not?"
# Narrow letters, letters under a combining accent, a letter under nine of them and a character
# that is not drawn, then wide characters, an emoji, the very wide ligature U+FDFD and a
# Cyrillic letter.  Of the nine accents, which fonts stack into a tower, two are shown.
_TOWER = "\u0301" * 9 + "\u202e"
_ANSWER = "il" * 40 + "e\u0301" * 39 + "o" + _TOWER + "キュー\U0001f642\ufdfdж" * 30
_SHOWN = _ANSWER.replace(_TOWER, "\u0301" * 2)
_WIDE = "W@M" * 60  # the widest Latin characters
_AUDIT = "Read /w/AUDIT.md first, follow it exactly and report every defect."
_ROLES = {
    "auditors": ["Read-only code audit"],
    "fixers": ["Read tmp/x/FIX2.md"],
    "reviewers": ["Read tmp/x/REVIEW"],
    "doc reviewers": ["Read tmp/x/REVIEW#.md and follow it for the docs"],
    "other": [""],
}


def _at(start: int, minute: float) -> int:
    """Return the epoch milliseconds of *minute* minutes after *start*."""
    return start + round(minute * MINUTE)


def _call(start: int, minute: float, name: str, **args: Any) -> dict[str, Any]:
    """Return the ``tool_call`` event of *name* at *minute*, *args* as the daemon stores them."""
    return {"type": "tool_call", "name": name, "ts": _at(start, minute), **args}


def _result(start: int, minute: float, content: str = "") -> dict[str, Any]:
    """Return a ``tool_result`` event at *minute*."""
    return {"type": "tool_result", "content": content, "ts": _at(start, minute)}


def _said(start: int, minute: float, text: str) -> dict[str, Any]:
    """Return the event of a message the user sent to the running task at *minute*."""
    return {"type": "prompt", "text": text, "steer": True, "ts": _at(start, minute)}


def _child(
    parent: str, start: int, text: str, begin: float, end: float | None, side: bool = False
) -> str:
    """Persist a sub-agent of *parent* that ran from *begin* to *end* (``None``: no end)."""
    link: dict[str, Any] = (
        {"subagent": {"parent_task_id": parent, "side_channel": True}}
        if side
        else {"parent_task_id": parent}
    )
    ended = 0 if end is None else _at(start, end)
    return persist_task(
        text, (), "ok", success=True, startTs=_at(start, begin), endTs=ended, **link
    )


def _parent(minutes: float | None, events: list[dict[str, Any]], start: int) -> str:
    """Persist a task that started at *start* and ran *minutes* (``None``: it still runs)."""
    ended = 0 if minutes is None else _at(start, minutes)
    return persist_task("the task", events, None, startTs=start, endTs=ended)


@pytest.fixture
def toy() -> str:
    """Persist the toy run of the module docstring and return its id."""
    start = int(time.time() * 1000) - 200 * MINUTE
    events = [
        {"type": "prompt", "text": "Audit and fix the parser", "ts": start},
        _result(start, 0.5),  # no call is open: ignored
        _call(start, 1, "Bash", description="List the files"),
        _result(start, 1.5),
        _call(start, 1.6, "Read") | {"ts": "bad"},  # no time: no wait, whenever its result came
        _result(start, 1.7),
        _call(start, 2, "run_parallel", description="Audit the parser"),
        _result(start, 20),
        _call(start, 21, "Bash", description="Run the full test suite", command="pytest"),
        _result(start, 21.1, "Started background job j1a2b3c4 (pid 7).\nLog: /tmp/j1a2b3c4.log"),
        _said(start, 21.5, "Oh, my toy fell down."),
        _call(start, 22, "bash_job", extras={"job_id": "j1a2b3c4", "action": "wait"}),
        _said(start, 24, "/askme <b>anything</b>"),
        _said(start, 25, "lost in time") | {"ts": "bad"},
        _said(start, 31, "/ask how far along are you?"),
        _said(start, 33, "/ask"),
        _result(start, 40),
        _call(start, 41, "Bash", description=_WIDE, command="cat > notes.md <<EOF"),
        _call(start, 41.1, "Write", path="notes.md", content="notes"),
        _result(start, 59.9),
        _result(start, 60),
        _call(start, 61, "run_agent", description="Fix the parser"),
        _result(start, 61.1, "Started the sorcar agent task as job agent-1."),
        _call(start, 71, "ask_user_question", extras={"question": _QUESTION}),
        _said(start, 83, _ANSWER),
        _result(start, 83),
        _call(start, 84, "agent_job", extras={"job_id": "agent-1", "action": "wait"}),
        _result(start, 99),
        _call(start, 99.9, "finish"),
    ]
    run = _parent(100, events, start)
    _child(run, start, "Read tmp/x/REVIEW2.md and follow it for section 1", 95, 97)
    _child(run, start, f"Read-only code audit,\n $SCOPE=C1. {_AUDIT}", 2, 20)
    _child(run, start, f"Read-only code audit, $SCOPE=C12. {_AUDIT}", 2, 20)
    _child(run, start, "Read tmp/x/FIX2.md and follow it for section 1", 62, 70)
    _child(run, start, "Read tmp/x/FIX2.md and follow it for section 2", 98, None)
    _child(run, start, "Read tmp/x/REVIEW3.md and follow it for the docs", 99, 104)
    _child(run, start, "What has this task done so far?", 10, 10.5, side=True)
    _child(run, start, "how far along are you?", 31, 31.2, side=True)
    _child(run, start, conflict_task(Path("/w"), "task", "main", ["a.py"]), 99.5, 99.8, side=True)
    return run


def _marks(unended: int, queries: int, waits: int, inputs: int) -> str:
    """Return the counts sentence both tools end their first line with."""
    return (
        f"Sub-agents with no recorded end: {unended}. Task updates and /ask questions: "
        f"{queries}. {_WAITS}: {waits}. User inputs: {inputs}."
    )


def _elements(path: Path, tag: str, **attributes: str) -> list[ET.Element]:
    """Return the *tag* elements of the SVG at *path* that carry *attributes*."""
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    return [
        element
        for element in root.iter(SVG + tag)
        if all(element.get(name.replace("_", "-")) == value for name, value in attributes.items())
    ]


def _texts(path: Path) -> list[str | None]:
    """Return the text of every ``<text>`` of the SVG at *path*, in document order.

    Each must be bound to a width (``textLength``, its glyphs squeezed to it) that ends 4 px
    inside the SVG; none may overlap another or be crossed by a line.  A text is 14 high, 11
    of it over its baseline.
    """
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    texts = list(root.iter(SVG + "text"))
    boxes = []
    for text in texts:
        assert text.get("lengthAdjust") == "spacingAndGlyphs", text.text
        x, y, length = (float(text.attrib[name]) for name in ("x", "y", "textLength"))
        assert length > 0 and x + length <= float(root.attrib["width"]) - 4, text.text
        boxes.append((x, x + length, y - 11, y + 3))
    for index, (left, right, top, bottom) in enumerate(boxes):
        for x0, x1, y0, y1 in boxes[index + 1 :]:
            assert not (left < x1 and x0 < right and top < y1 and y0 < bottom), texts[index].text
        for line in root.iter(SVG + "line"):
            x, y0, y1 = (float(line.attrib[name]) for name in ("x1", "y1", "y2"))
            assert not (left <= x <= right and y0 < bottom and top < y1), texts[index].text
    return [text.text for text in texts]


def test_sea_follows_the_contract_and_is_a_registered_command() -> None:
    """The SEA pins its two tools and its prompt; `/timeline` resolves to it."""
    agent = TimelineSea()
    assert_worker_contract(
        sea,
        agent,
        {"tool_profile": "none", "max_budget": 2.0, "timeout": 600, "locked": ["tool_profile"]},
    )
    assert agent.system_prompt("ASSEMBLED") == sea.SYSTEM_PROMPT
    assert agent.tools([print]) == [print, timeline_agents, draw_timeline]
    assert agent.prompt("  36b8e451 ") == f"36b8e451\n\n{sea.ADD_TO_PROMPT}"
    assert agent.prompt(" \n") == sea.ADD_TO_PROMPT
    path = Path(sea.__file__)
    assert sea_commands.get_command("timeline") == path
    assert sea_commands.slash_command("/timeline 36b8e451") == ("36b8e451", path)
    assert sea_commands.help_text_if_command("/timeline help") == agent.description()


def test_the_openings_are_listed_and_the_chart_shows_every_fact_of_the_run(
    toy: str, tmp_path: Path
) -> None:
    """The listing groups the sub-agents by masked opening; the chart has a bar per sub-agent
    in start order, a band per wait, a line per user input, a dashed line per task update or
    `/ask` question, and the notes.  The image tag names the file as a URL does: the whole
    path percent-encoded, the quotes and the space of its directory as the ``#`` of its name."""
    assert timeline_agents(f" {toy[:8].upper()} ").splitlines() == [
        f"Task {toy}, 104 min. Sub-agents: 6. {_marks(1, 2, 4, 4)}",
        _LISTING,
        "- 2 x, first at 2 min: Read-only code audit, $SCOPE=C#. Read /w/AUDIT.md first, fol",
        "- 2 x, first at 62 min: Read tmp/x/FIX#.md and follow it for section #",
        "- 1 x, first at 95 min: Read tmp/x/REVIEW#.md and follow it for section #",
        "- 1 x, first at 99 min: Read tmp/x/REVIEW#.md and follow it for the docs",
    ]
    target = tmp_path / "o'neil's out" / "toy #1.svg"
    report = draw_timeline(toy, json.dumps(_ROLES), str(target))
    assert report == (
        f"Timeline written to {target}. Bars: auditors (2), fixers (2), reviewers (1), "
        f"doc reviewers (1), merge-conflict resolver (1). {_marks(1, 2, 4, 4)}\n"
        f'Show it with: <img src="{quote(str(target))}" '
        f'alt="Timeline of task {toy[:8]}">'
    )
    assert "o%27neil%27s%20out" in report and 'toy%20%231.svg" alt=' in report
    assert _texts(target) == [
        f"Task {toy[:8]}: 7 sub-agents, one row each, in start order",
        "1", "2: 18 min", "3", "4", "5: 19 min", "6: 12 min", "7", "8: 11 min",
        "0 min", "15 min", "30 min", "45 min", "60 min", "75 min", "90 min",
        "auditors (2)", "fixers (2)", "reviewers (1)", "doc reviewers (1)",
        "merge-conflict resolver (1)",
        "pale: no end recorded, drawn to the end of the run (1)",
        "dashed: task updates and /ask questions (2)",
        "over 10 min in one tool call of the task while no sub-agent ran",
        "1: user input: Oh, my toy fell down.",
        "2: 18 min in bash_job: Run the full test suite",
        "3: user input: /askme <b>anything</b>",
        "4: user input: /ask",
        # A note is cut where its line ends: after 70 of the widest characters, 133 of prose
        # or 193 of the mixed text, in which an accent takes no room.
        "5: 19 min in Bash: " + _WIDE[:70] + "…",
        "6: 12 min in ask_user_question: " + _QUESTION[:133] + "…",
        "7: user input: " + _SHOWN[:193] + "…",
        "8: 11 min in agent_job",
    ]
    # A mark sits under every earlier label that reaches it: 3 is under the long label of 2,
    # which its line would cross from the top row; 4 and 7 find the top row free again.
    drawn = {text.text: text for text in _elements(target, "text")}
    rows = [drawn[label].get("y") for label in ("1", "2: 18 min", "3", "4", "7", "8: 11 min")]
    assert rows == ["30.0", "44.0", "58.0", "30.0", "30.0", "44.0"]
    # The width a text is bound to is the sum of the Helvetica advances of its characters.
    assert drawn["merge-conflict resolver (1)"].get("textLength") == "139.4"
    # One bar per sub-agent and a legend swatch per role; the bars start under the three rows.
    fills = [rect.get("fill") for rect in _elements(target, "rect")]
    assert [fills.count(c) for c in (BLUE, RED, GREEN, ORANGE, BROWN, BAND)] == [3, 3, 2, 2, 2, 5]
    first, second, third = (_elements(target, "rect", fill=BLUE)[i] for i in range(3))
    assert (first.get("x"), first.get("y"), first.get("width")) == ("87.3", "66.0", "155.8")
    assert (second.get("x"), second.get("y"), second.get("height")) == ("87.3", "84.0", "17.4")
    assert third.get("width") == "10"  # the legend swatch
    # The fixer with no recorded end is drawn to the end of the run, minute 104, and pale.
    fixers = _elements(target, "rect", fill=RED)[:2]
    assert [(r.get("x"), r.get("width"), r.get("opacity")) for r in fixers] == [
        ("606.5", "69.2", None), ("918.1", "51.9", "0.4"),
    ]
    band = _elements(target, "rect", fill=BAND)[0]
    assert (band.get("x"), band.get("y"), band.get("width")) == ("260.4", "33", "155.8")
    assert len(_elements(target, "line", stroke_dasharray="2,3")) == 2
    lines = _elements(target, "line", stroke="#000")
    assert [(line.get("x1"), line.get("y1")) for line in lines] == [
        ("256.1", "19"), ("277.7", "47"), ("355.6", "19"), ("788.3", "19"),
    ]


def test_a_short_run_a_long_one_and_one_that_still_runs_get_their_own_axis(
    tmp_path: Path,
) -> None:
    """A 30-second task gets a one-minute axis and no marks; nine hours are ticked in hours;
    a task that has run for 25 days is drawn up to now, ticked every three days."""
    now = int(time.time() * 1000)
    short = _parent(0.5, [], now - 10 * MINUTE)
    _child(short, now - 10 * MINUTE, "Summarise the log", 0.1, 0.3)
    target = tmp_path / "short.svg"
    assert draw_timeline(short, '{"writers": ["Summarise"]}', str(target)).startswith(
        f"Timeline written to {target}. Bars: writers (1). {_marks(0, 0, 0, 0)}\n"
    )
    assert _texts(target) == [
        f"Task {short[:8]}: 1 sub-agent, one row each, in start order",
        "0 min", "1 min", "writers (1)",
    ]
    (bar, swatch) = _elements(target, "rect", fill=BLUE)
    assert (bar.get("x"), bar.get("y"), bar.get("width")) == ("160.0", "24.0", "180.0")
    assert not _elements(target, "line", stroke_dasharray="2,3")

    start = now - 600 * MINUTE
    benchmark = [
        _said(start, -1, "early"),
        _call(start, 200, "Bash", description="Run the benchmark"),
        _result(start, 215),
        _call(start, 525, "Bash", description="Run it again"),
        _result(start, 545),
        _said(start, 546, "late"),
    ]
    many = _parent(540, benchmark, start)
    names = [f"task {chr(97 + i // 26)}{chr(97 + i % 26)} of the batch" for i in range(101)]
    for index, name in enumerate(names):
        _child(many, start, name, index, index + 30)
    listing = timeline_agents(many).splitlines()
    assert listing[0] == f"Task {many}, 540 min. Sub-agents: 101. {_marks(0, 0, 2, 2)}"
    assert listing[2] == "- 1 x, first at 0 min: task aa of the batch"
    assert listing[101:] == [
        "- 1 x, first at 99 min: task dv of the batch",
        "... and 1 more openings",
    ]
    target = tmp_path / "many.svg"
    assert draw_timeline(many, '{"workers": ["task"]}', str(target)).startswith(
        f"Timeline written to {target}. Bars: workers (101). "
    )
    # The last sub-agent ended at minute 130: the calls that ran from 200 and from 525 are
    # waits.  Next to the right edge only the number of a mark fits; its note has the rest.
    # What the history stamps outside the run is drawn at the end of the axis next to it:
    # the two messages, and the band of the second wait, which ends with the run.
    texts = _texts(target)
    assert texts[1:15] == ["1", "2: 15 min", "3", "4"] + [f"{hour} h" for hour in range(10)]
    assert texts[-4:] == [
        "1: user input: early", "2: 15 min in Bash: Run the benchmark",
        "3: 20 min in Bash: Run it again", "4: user input: late",
    ]
    labels = [label.get("x") for label in _elements(target, "text")[1:5]]
    assert labels == ["73.0", "406.3", "948.0", "973.0"]
    assert _elements(target, "rect", fill=BAND)[1].get("width") == "25.0"
    assert _elements(target, "rect", fill=BLUE)[100].get("height") == "4.0"

    running = _parent(None, [], now - 25 * 1440 * MINUTE)
    for day, name in enumerate("abcdefghi"):
        _child(running, now - 25 * 1440 * MINUTE, f"{name} work", day * 1440, None)
    long = "W" * 33  # with it a name takes half a line
    nine = {f"role {name} {long}": [f"{name} "] for name in "abcdefghi"}
    target = tmp_path / "running.svg"
    assert draw_timeline(running, json.dumps(nine), str(target)) == (
        "Error: nothing was written. Use at most 8 roles."
    )
    assert not target.exists()
    nine[f"role h {long * 3}"] = nine.pop(f"role h {long}") + nine.pop(f"role i {long}")
    assert draw_timeline(running, json.dumps(nine), str(target)).startswith(
        f"Timeline written to {target}. Bars: role a {long} (1), "
    )
    texts = _texts(target)
    assert texts[1:10] == [f"{hours} h" for hours in range(0, 577, 72)]
    assert texts[10:] == [f"role {name} {long} (1)" for name in "abcdefg"] + [
        f"role h {'W' * 74}…",
        "pale: no end recorded, drawn to the end of the run (9)",
    ]
    # Two legend entries fit a line and a name longer than a line is cut to one: seven roles
    # take four lines, the eighth a fifth and the last entry a sixth.
    assert len({text.get("y") for text in _elements(target, "text")[10:]}) == 6
    # A sub-agent with no recorded end runs up to now: the first bar spans the whole axis.
    first = _elements(target, "rect", fill=BLUE)[0]
    assert (first.get("width"), first.get("opacity")) == ("900.0", "0.4")


def test_a_task_that_cannot_be_drawn_is_refused_and_nothing_is_written(
    toy: str, tmp_path: Path
) -> None:
    """No such task, an ambiguous id, no sub-agents, a bad file name and bad roles each return
    an error that says what to fix, and leave no file."""
    for last in "12":
        _add_task("twin", task_id="abcdef01" + last * 24)
    for ref in ("zz", toy[:5], "g" * 8, "0" * 32, "abcdef01", ""):
        error = f"Error: no single task matches {ref!r}. {_NO_TASK}"
        assert timeline_agents(ref) == error
        assert draw_timeline(ref, "{}", "") == error
    alone = _parent(5, [], int(time.time() * 1000) - 9 * MINUTE)
    _child(alone, int(time.time() * 1000) - 9 * MINUTE, "What has this task done?", 1, 2, side=True)
    error = f"Error: task {alone} started no sub-agents, so it has no timeline."
    assert timeline_agents(alone) == error
    assert draw_timeline(alone, "{}", "") == error

    target = tmp_path / "toy.svg"
    assert draw_timeline(toy, json.dumps(_ROLES), str(tmp_path / "toy.png")) == (
        "Error: svg_file must end in .svg."
    )
    assert draw_timeline(toy, "{auditors}", str(target)).startswith("Error: roles is not JSON: ")
    shape = 'Error: roles must be a JSON object {"role": ["opening", ...], ...}.'
    for roles in ('["Fixer"]', '{"fixers": "Fixer"}', '{"fixers": [1]}', '{" ": ["Fixer"]}'):
        assert draw_timeline(toy, roles, str(target)) == shape
    partial = '{"fixers": ["Read tmp/x/FIX2.md"], "x": ["Read tmp/x/REVIEW"]}'
    assert draw_timeline(toy, partial, str(target)) == (
        "Error: nothing was written. No opening of a role starts the task text of these "
        "sub-agents; add one for each:\n"
        "- 2 x, first at 2 min: Read-only code audit, $SCOPE=C#. Read /w/AUDIT.md first, fol"
    )
    assert list(tmp_path.iterdir()) == []


def test_agent_run_lists_the_openings_draws_the_chart_and_shows_it(
    toy: str, tmp_path: Path
) -> None:
    """A run configured as the daemon configures it: the scripted model asks for the openings,
    draws the chart with its roles and finishes with the image.  The chart lands in the work
    dir of the run, under the default name, and the tool names it relative to that dir."""
    name = f"reports/timeline-{toy[:8]}.svg"
    image = f'<img src="{name}" alt="Timeline of task {toy[:8]}">'
    summary = f"{image}<p>{name}: 7 bars.</p>"
    script = [
        tool_call_body("timeline_agents", {"task": toy[:8]}, 500),
        tool_call_body("draw_timeline", {"task": toy[:8], "roles": json.dumps(_ROLES)}, 600),
        finish_body(summary, prompt_tokens=700),
    ]
    run = sea_commands.evaluate_sea([TimelineSea()], toy[:8])
    assert run.prompt == f"{toy[:8]}\n\n{sea.ADD_TO_PROMPT}"
    agent = ChatSorcarAgent("timeline-sea-test")
    with registered(agent, "timeline-sea-test"):
        parsed, agentic = run_sea(run, script, tmp_path, max_steps=4, agent=agent)
    assert parsed["success"] is True and parsed["summary"] == summary

    assert len(agentic) == 3
    for request in agentic:
        assert tool_names(request) == {"finish", "timeline_agents", "draw_timeline"}
        assert system_message(request).startswith(sea.SYSTEM_PROMPT)
    listing, drawn = (tool_result_text(result) for result in tool_results(agentic[2]))
    assert listing.startswith(f"Task {toy}, 104 min. Sub-agents: 6. ")
    assert drawn == (
        f"Timeline written to {name}. Bars: auditors (2), fixers (2), reviewers (1), "
        f"doc reviewers (1), merge-conflict resolver (1). {_marks(1, 2, 4, 4)}\n"
        f"Show it with: {image}"
    )
    assert len(_elements(tmp_path / name, "rect", fill=RED)) == 3
