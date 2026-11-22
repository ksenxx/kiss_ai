# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""What the history says about the sub-agents of a task: the facts a timeline draws.

:func:`read_run` turns a task's history row, its events and the rows
of its children into a :class:`Run`: when each sub-agent ran, when a
side query started, what the user typed into the running task, and
where the task sat in one tool call while no sub-agent ran.  Every
time is in minutes after the task started; an event that holds no
time is left out.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.chat_summary import end_ms
from kiss.agents.sorcar.merge_conflict_resolver import CONFLICT_TASK_OPENING
from kiss.agents.sorcar.sea_commands import _split_slash_command
from kiss.agents.sorcar.task_digest import (
    _event_ms,
    _start_ms,
    _tool_call_args,
    child_tasks,
    load_task,
)

MERGE = "merge-conflict resolver"
"""The role of the agent the harness ran to resolve the conflicts of merging the task's work."""

WAIT_MINUTES = 10
"""A :class:`Wait` is longer than this."""

OPENING_CHARS = 400
"""How much of a sub-agent's task text :attr:`Agent.opening` keeps."""

_ID_PREFIX = re.compile(r"[0-9a-f]{6,32}")
_NUMBER = re.compile(r"\d+")
_JOB_STARTED = re.compile(r"Started background job (\w+)")


@dataclass
class Agent:
    """A sub-agent of the task: one bar of the chart."""

    opening: str
    """How its task text starts (:func:`mask`)."""
    start: float
    end: float
    """The end of the run when the history records no end for it (:attr:`ended` is false)."""
    ended: bool
    role: str = ""
    """:data:`MERGE` for the resolver; a dispatched sub-agent gets its role from the caller."""


@dataclass
class Wait:
    """A stretch in which no sub-agent ran while the task sat in one tool call."""

    start: float
    end: float
    what: str
    """The tool the task was in and, when the call has one, its description."""


@dataclass
class _Call:
    """A tool call of the task."""

    event: dict[str, Any]
    returned: float | None = None
    """When its result came (:func:`_minute`); ``None`` as well for a call that never returned."""


@dataclass
class Run:
    """The facts of one task's run that the timeline shows."""

    task_id: str
    minutes: float
    """How long the run is, up to now for a task that has not ended; at least one."""
    agents: list[Agent]
    """In the order they started."""
    queries: list[float]
    """When each side query (a task update, an ``/ask`` question) started."""
    waits: list[Wait]
    inputs: list[tuple[float, str]]
    """What the user sent to the running task, and when."""


def mask(text: str) -> str:
    """Return *text* on one line with every number in it written as ``#``.

    Sub-agents dispatched from one template differ in such numbers (a
    round, a section: ``Round-2 fixer G3``), so their masked task
    texts start alike.  The letters beside a number stay: they tell
    a fixer that follows ``FIX2.md`` from a reviewer that follows
    ``REVIEW2.md``.
    """
    return _NUMBER.sub("#", " ".join(text.split()))


def find_task(ref: str) -> dict[str, Any] | str:
    """Return the task *ref* names, with its events, or an ``Error:`` line.

    Args:
        ref: A task id or the first six or more characters of one.
    """
    ref = ref.strip().lower()
    rows = persistence.tasks_matching(id_prefix=ref, limit=2) if _ID_PREFIX.fullmatch(ref) else []
    task = load_task(str(rows[0]["id"])) if len(rows) == 1 else None
    if task is None:
        return (
            f"Error: no single task matches {ref!r}. Pass a task id or the first six or more "
            "characters of one."
        )
    return task


def _gaps(
    start: float, end: float, busy: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Return the stretches of ``start..end`` that no interval of *busy* (sorted) covers."""
    gaps = []
    for busy_start, busy_end in busy:
        if busy_start >= end:
            break
        if busy_start > start:
            gaps.append((start, busy_start))
        start = max(start, busy_end)
    if start < end:
        gaps.append((start, end))
    return gaps


def _minute(event: dict[str, Any], start_ms: int) -> float | None:
    """Return when *event* happened, in minutes after *start_ms*; ``None`` when it has no time."""
    at = _event_ms(event)
    return (at - start_ms) / 60000 if at else None


def _waits(events: list[dict[str, Any]], agents: list[Agent], start_ms: int) -> list[Wait]:
    """Return the waits of the task whose *events* these are, given its *agents*.

    A ``tool_result`` closes the innermost open call.  The harness runs
    a shell here-document as calls of its own inside the ``Bash`` call,
    so a stretch inside a wait already found is the same wait.  A call
    that never returned (``finish``, an interrupted one) is no wait,
    nor is one when its event or that of its result holds no time.  A
    ``bash_job`` call is described by the call that started its job.
    """
    busy = [(agent.start, agent.end) for agent in agents]
    jobs: dict[str, str] = {}
    calls: list[_Call] = []
    open_calls: list[_Call] = []
    for event in events:
        kind = event.get("type")
        if kind == "tool_call":
            open_calls.append(_Call(event))
            calls.append(open_calls[-1])
        elif kind == "tool_result" and open_calls:
            call = open_calls.pop()
            call.returned = _minute(event, start_ms)
            job = _JOB_STARTED.match(str(event.get("content") or ""))
            if job:
                jobs[job[1]] = str(call.event.get("description") or "")
    waits: list[Wait] = []
    for call in calls:
        called, returned = _minute(call.event, start_ms), call.returned
        if called is None or returned is None or returned - called <= WAIT_MINUTES:
            continue
        args = _tool_call_args(call.event)
        detail = (
            args.get("description") or args.get("question") or jobs.get(str(args.get("job_id")))
        )
        what = f"{call.event.get('name')}: {detail}" if detail else str(call.event.get("name"))
        for gap_start, gap_end in _gaps(called, returned, busy):
            inside = any(w.start <= gap_start and gap_end <= w.end for w in waits)
            if gap_end - gap_start > WAIT_MINUTES and not inside:
                waits.append(Wait(gap_start, gap_end, what))
    return waits


def read_run(task: dict[str, Any]) -> Run:
    """Return the :class:`Run` of *task*, a history row with its ``events`` (:func:`find_task`).

    A side-channel child is a side query, except the merge-conflict
    resolver, which worked on the task's result and gets a bar like a
    dispatched sub-agent.  A child with no recorded end (it still
    runs, or its process was killed) ends with the run.  An ``/ask``
    question the user sent is the side query it started, not an input
    to the task; nor is a message whose event holds no time.
    """
    start_ms = _start_ms(task)
    children = sorted(child_tasks(str(task["id"])), key=_start_ms)
    ended = [end_ms(child.get("end_ts")) for child in children]
    run_end = max([end_ms(task.get("end_ts")) or int(time.time() * 1000), *ended])
    minutes = max((run_end - start_ms) / 60000, 1.0)
    agents: list[Agent] = []
    queries: list[float] = []
    for child, child_end in zip(children, ended):
        text = str(child.get("task") or "")
        side = bool(child.get("is_side_channel"))
        began = min(max((_start_ms(child) - start_ms) / 60000, 0.0), minutes)
        if side and not text.startswith(CONFLICT_TASK_OPENING):
            queries.append(began)
            continue
        finished = (child_end - start_ms) / 60000 if child_end else minutes
        agents.append(
            Agent(
                mask(text[:OPENING_CHARS]),
                began,
                max(finished, began),
                bool(child_end),
                MERGE if side else "",
            )
        )
    events = task["events"]
    inputs = []
    for event in events:
        if event.get("type") != "prompt" or not event.get("steer"):
            continue
        at = _minute(event, start_ms)
        text = str(event.get("text") or "")
        command = _split_slash_command(text)
        if at is not None and (command is None or command[0] != "ask" or not command[1]):
            inputs.append((at, text))
    return Run(
        str(task["id"]), minutes, agents, queries, _waits(events, agents, start_ms), inputs
    )
