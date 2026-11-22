# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Timeline agent: an SVG Gantt chart of the sub-agents of a task, shown in the result.

``/timeline <task id>`` reads the task and its sub-agents from the
history and writes one SVG file: a thin bar per sub-agent in the order
they started, coloured by role, with the task updates and ``/ask``
questions, the messages the user sent and the waits in which no
sub-agent ran (:mod:`kiss.agents.seas.timeline.chart`).

The history holds every fact of the chart but one, what each
sub-agent was for.  The model reads how their task texts open
(:func:`timeline_agents`) and names the roles; :func:`draw_timeline`
refuses roles that leave a sub-agent out and draws everything else
from the history (:mod:`kiss.agents.seas.timeline.history`).

Two ways to run it::

    /timeline 36b8e451                 a task by the first characters of its id
    run_agent(agent="timeline", task="<task id>, write it to docs/run.svg")
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from kiss.agents.seas.base.base_sea import WorkerSea
from kiss.agents.seas.timeline import chart, history
from kiss.server.agent_state import current_agent

ADD_TO_PROMPT = "Draw the timeline of the task named above."
"""Ends every run's prompt."""

LISTED_CHARS = 60
"""How much of an opening a tool lists: sub-agents that agree on it are listed as one."""

MAX_LISTED = 100
"""How many openings one tool result lists."""

SYSTEM_PROMPT = """\
You draw the timeline of a task: an SVG Gantt chart with one bar per sub-agent.

1. Call `timeline_agents(task)`. `task` is the task id the user gave, or its first six or
   more characters. It lists how the task texts of the sub-agents open.
2. Sort the sub-agents into at most 8 roles by the kind of work they did, such as auditors,
   fixers and reviewers; the same kind of work in a later round is the same role. Name each
   role in one or two plain words.
3. Call `draw_timeline(task, roles, svg_file)`. `roles` is a JSON object
   {"role": ["opening", ...]} of openings from the listing, whole or cut short: a sub-agent
   gets the role of the longest one its task text starts with. `svg_file` is the path the
   user named, else empty. When it returns `Error:`, fix what it names and call it again.
4. Call `finish` with success=true and this HTML: the `<img>` tag `draw_timeline` returned,
   unchanged, then one paragraph with the path of the file and the counts it reported.

When the user gave no task id, or a tool finds no task or no sub-agents, call `finish` with
success=false and say so. Report nothing the tools did not return.
"""


def _run(task: str) -> history.Run | str:
    """Return the run of the task *task* names, or an ``Error:`` line when it has no bar to draw."""
    found = history.find_task(task)
    if isinstance(found, str):
        return found
    run = history.read_run(found)
    if not run.agents:
        return f"Error: task {run.task_id} started no sub-agents, so it has no timeline."
    return run


def _marks(run: history.Run) -> str:
    """Return the counts of what the chart of *run* marks on and beside its bars."""
    return (
        f"Sub-agents with no recorded end: {sum(not agent.ended for agent in run.agents)}. "
        f"Task updates and /ask questions: {len(run.queries)}. Waits of over "
        f"{history.WAIT_MINUTES} min in one tool call with no sub-agent running: "
        f"{len(run.waits)}. User inputs: {len(run.inputs)}."
    )


def _openings(agents: list[history.Agent]) -> list[str]:
    """Return one line per distinct opening of *agents*, in the order first seen."""
    first: dict[str, float] = {}
    for agent in agents:
        first.setdefault(agent.opening[:LISTED_CHARS], agent.start)
    counts = Counter(agent.opening[:LISTED_CHARS] for agent in agents)
    lines = [
        f"- {counts[opening]} x, first at {start:.0f} min: {opening}"
        for opening, start in first.items()
    ]
    more = len(lines) - MAX_LISTED
    return lines[:MAX_LISTED] + ([f"... and {more} more openings"] if more > 0 else [])


def _assign(agents: list[history.Agent], roles: str) -> str:
    """Give every dispatched sub-agent of *agents* its role; return ``""`` or the refusal.

    Args:
        agents: The agents of a run; the merge resolver keeps its role.
        roles: The ``roles`` argument of :func:`draw_timeline`.
    """
    try:
        named = json.loads(roles)
    except ValueError as error:
        return f"Error: roles is not JSON: {error}"
    if not isinstance(named, dict) or not all(
        role.strip() and isinstance(openings, list) and all(isinstance(o, str) for o in openings)
        for role, openings in named.items()
    ):
        return 'Error: roles must be a JSON object {"role": ["opening", ...], ...}.'
    longest_first = sorted(
        (
            (len(opening), opening, role)
            for role, openings in named.items()
            for opening in map(history.mask, openings)
        ),
        reverse=True,
    )
    dispatched = [agent for agent in agents if agent.role != history.MERGE]
    for agent in dispatched:
        agent.role = next(
            (role for _, opening, role in longest_first if agent.opening.startswith(opening)), ""
        )
    missed = [agent for agent in dispatched if not agent.role]
    if missed:
        return "\n".join([
            "Error: nothing was written. No opening of a role starts the task text of these "
            "sub-agents; add one for each:",
            *_openings(missed),
        ])
    if len({agent.role for agent in dispatched}) > len(chart.PALETTE):
        return f"Error: nothing was written. Use at most {len(chart.PALETTE)} roles."
    return ""


def timeline_agents(task: str) -> str:
    """List how the task texts of a task's sub-agents open, to sort them into roles.

    Args:
        task: The task id, or the first six or more characters of it.

    Returns:
        What the run holds, then one line per opening with how many
        sub-agents start that way and when the first of them started;
        or an ``Error:`` line.
    """
    run = _run(task)
    if isinstance(run, str):
        return run
    dispatched = [agent for agent in run.agents if agent.role != history.MERGE]
    return "\n".join([
        f"Task {run.task_id}, {run.minutes:.0f} min. Sub-agents: {len(dispatched)}. {_marks(run)}",
        "How the task texts of the sub-agents open, every number written as #:",
        *_openings(dispatched),
    ])


def draw_timeline(task: str, roles: str, svg_file: str = "") -> str:
    """Write the SVG timeline of a task's sub-agents, the bars coloured by role.

    Args:
        task: The task id, or the first six or more characters of it.
        roles: A JSON object ``{"role": ["opening", ...], ...}``.  A
            sub-agent gets the role of the longest opening its task
            text starts with; a number is written as it is or as
            ``#``, and the empty opening ``""`` starts every task
            text.
        svg_file: Where to write the chart: a path relative to the
            work dir, or an absolute one; empty for
            ``reports/timeline-<first 8 characters of the task id>.svg``.

    Returns:
        The file written, what it shows and the ``<img>`` tag that
        shows it in a result, the path percent-encoded as in a URL;
        or ``Error:`` and what to fix, with nothing written.
    """
    run = _run(task)
    if isinstance(run, str):
        return run
    name = Path(svg_file or f"reports/timeline-{run.task_id[:8]}.svg").expanduser()
    if name.suffix != ".svg":
        return "Error: svg_file must end in .svg."
    if error := _assign(run.agents, roles):
        return error
    caller = current_agent()
    path = Path(caller.work_dir if caller is not None else Path.cwd()) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chart.render(run), encoding="utf-8")
    held = Counter(agent.role for agent in run.agents)
    bars = ", ".join(f"{role} ({count})" for role, count in held.items())
    # A relative path is named with forward slashes on Windows too: a result shows
    # ``reports\x.svg`` as plain text and ``reports/x.svg`` as a link to the file.
    shown = str(name) if name.is_absolute() else name.as_posix()
    return (
        f"Timeline written to {shown}. Bars: {bars}. {_marks(run)}\n"
        f'Show it with: <img src="{quote(shown)}" alt="Timeline of task {run.task_id[:8]}">'
    )


class TimelineSea(WorkerSea):
    """The ``/timeline`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/timeline help``."""
        return (
            "Draws the timeline of a task as an SVG Gantt chart and shows it in the result: one "
            "bar per sub-agent in start order, coloured by role, with the task updates and /ask "
            "questions, the messages you sent to the running task and the waits of over 10 "
            "minutes in which no sub-agent ran; use it as `/timeline <task id>` or "
            '`run_agent(agent="timeline", task="<task id>")`.'
        )

    def prompt(self, task: str) -> str:
        """Return *task* followed by :data:`ADD_TO_PROMPT`."""
        return f"{task.strip()}\n\n{ADD_TO_PROMPT}".strip()

    def system_prompt(self, system_prompt: str) -> str:
        """Return the agent's own system prompt (:data:`SYSTEM_PROMPT`)."""
        return SYSTEM_PROMPT

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """A short run on the two tools below and ``finish``; ``tool_profile`` is locked."""
        return settings | {
            "tool_profile": "none",
            "max_budget": 2.0,
            "timeout": 600,
            "locked": ["tool_profile"],
        }

    def tools(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        """Add :func:`timeline_agents` and :func:`draw_timeline` to the run's tools."""
        return [*tools, timeline_agents, draw_timeline]
