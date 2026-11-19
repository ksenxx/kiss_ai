# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The evidence of a ``/discover`` run, read from its persisted trajectory.

A ``/discover`` run keeps its state in four files of its scratch
directory (:data:`LOOP_FILES`): the frontier (``pareto.md``), the log
of every evaluated idea (``explored-ideas.md``), the ideas and the
split.  Their layout differs from run to run (markdown tables, bare
``a | b | c`` lines, the metrics in one cell or in several) and a run
writes them with ``Write`` / ``Edit`` or from the shell, so nothing
here parses them.  :func:`evidence` lists, in trajectory order, every
``Write`` and ``Edit`` of one of them, every shell command that names
one, every ``Read`` of the first two (:data:`FRONTIER_FILES`; a
``Read`` of the ideas or the split repeats what was written), every
random draw (:data:`DRAW_CALL`) and every ``decide`` call, then the
copies still on disk and the run's final result.  ``/discover_viz``
gives that text to the model and checks what the model then says about
the run against it (:mod:`kiss.agents.seas.discover_viz.story`).
"""

from __future__ import annotations

import html
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.task_digest import _tool_call_args, clip, load_task
from kiss.agents.sorcar.useful_tools import _edit_pairs
from kiss.core.kiss_tmp import kiss_tmp_dir

LOOP_FILES = ("pareto.md", "explored-ideas.md", "ideas.md", "split.md")
"""The files a ``/discover`` run keeps in its scratch directory."""

FRONTIER_FILES = LOOP_FILES[:2]
"""The two that hold the frontier and the status of every evaluated idea."""

DISCOVER_SEA = "discover_sea"
"""The ``sea`` column of a ``/discover`` run's history row."""

DRAW_CALL = "random.random()"
"""What the shell command of the loop's ``RANDOM()`` draw contains."""

RECENT_RUNS = 20
"""How many runs :func:`recent_runs` lists."""

NESTED_MARK = " (run by the harness as part of the command above)"
"""Ends the heading of an evidence entry whose call is :attr:`Call.nested`."""

# The most characters the evidence quotes of each thing (:func:`_ends` cuts the rest).
_FILE_CHARS = 12_000  # a Write or Read of a frontier file
_SIDE_FILE_CHARS = 3_000  # a Write of ideas.md or split.md
_DISK_CHARS = 60_000  # a copy on disk: quoted once, and the whole log of a 100-round run fits
_EDIT_CHARS = 4_000  # each side of a replacement
_COMMAND_CHARS = 4_000
_OUTPUT_CHARS = 6_000
_DECIDE_CHARS = 1_500
_RESULT_CHARS = 8_000
_PROMPT_CHARS = 3_000

_ID_PREFIX = re.compile(r"[0-9a-f]{6,32}")
_WORKTREE_SEGMENT = re.compile(r"/\.kiss-worktrees/[^/]+")
_TAG = re.compile(r"<[^>]+>")


@dataclass
class Call:
    """One tool call of a trajectory: its tool, its arguments, what it printed and returned."""

    name: str
    args: dict[str, Any]
    output: str = ""
    failed: bool = False
    nested: bool = False
    """True for a call the harness made inside a ``Bash`` call (:func:`tool_calls`)."""

    def touches(self, names: tuple[str, ...]) -> bool:
        """True when the call's path argument, or its shell command, names a file of *names*."""
        if self.name == "Bash":
            return any(name in str(self.args.get("command") or "") for name in names)
        return os.path.basename(str(self.args.get("path") or "")) in names


def tool_calls(events: list[dict[str, Any]]) -> list[Call]:
    """Return the tool calls of *events* in the order they started, each with its output.

    The harness runs one call at a time, but calls nest: it runs a
    shell here-document into an existing file as a ``Write`` of the
    whole file and the rest of the command as a ``Bash`` call of its
    own, and records both between the ``Bash`` call and its result.
    Streamed output and a result therefore belong to the innermost call
    still open.  A call that never gets a result (``finish``) stays
    open and takes nothing from the calls after it.
    """
    calls: list[Call] = []
    open_calls: list[Call] = []
    for event in events:
        kind = event.get("type")
        if kind == "tool_call":
            nested = bool(open_calls) and open_calls[-1].name == "Bash"
            call = Call(str(event.get("name") or ""), _tool_call_args(event), nested=nested)
            calls.append(call)
            open_calls.append(call)
        elif open_calls and kind == "system_output":
            open_calls[-1].output += str(event.get("text") or "")
        elif open_calls and kind == "tool_result":
            call = open_calls.pop()
            call.output += str(event.get("content") or "")
            call.failed = bool(event.get("is_error"))
    return calls


def _one_line(text: object, limit: int) -> str:
    """Return *text* on one line, clipped to *limit* characters."""
    return clip(" ".join(str(text or "").split()), limit)


def _ends(text: object, limit: int) -> str:
    """Return ``str(text)``, its middle replaced by a marker when it is longer than *limit*.

    Both ends are kept because both carry news: a loop file starts with
    its header and ends with its newest line, and a command that shows
    several files prints the log last.
    """
    s = str(text or "")
    if len(s) <= limit:
        return s
    half = limit // 2
    return f"{s[:half]}\n…[{len(s) - 2 * half} characters cut]…\n{s[-half:]}"


def _block(text: str) -> str:
    """Return *text* as a fenced block (tildes: the loop files hold backticks)."""
    return f"~~~\n{text.rstrip()}\n~~~"


def _state(task: dict[str, Any]) -> str:
    """Return how the run ended: not at all, or with which ``success``."""
    success = task.get("success")
    if success is None:
        return "not finished"
    return f"finished, success={'true' if success else 'false'}"


def run_line(task: dict[str, Any]) -> str:
    """Return one list line for the history row *task*: id, start, state, steps, request."""
    started = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(task["timestamp"])))
    return (
        f"- {task['id']} | {started} | {_state(task)} | {task['steps']} steps | "
        f"{_one_line(task['task'], 120)}"
    )


def recent_runs() -> str:
    """List the newest ``/discover`` runs of the history, one :func:`run_line` each."""
    rows = persistence.tasks_matching(sea=DISCOVER_SEA, limit=RECENT_RUNS)
    if not rows:
        return "The history holds no /discover run."
    return "\n".join(["Newest /discover runs:", *map(run_line, rows)])


def find_task(ref: str) -> dict[str, Any] | str:
    """Return the task *ref* names, with its events, or an ``Error:`` text.

    Args:
        ref: A task id, the first six or more characters of one, or
            empty / ``latest`` for the newest ``/discover`` run.

    Returns:
        The history row with its ``events`` (:func:`load_task`), or the
        error followed by :func:`recent_runs` when *ref* names no single task.
    """
    ref = ref.strip().lower()
    if ref in ("", "latest"):
        rows = persistence.tasks_matching(sea=DISCOVER_SEA, limit=1)
    elif _ID_PREFIX.fullmatch(ref):
        rows = persistence.tasks_matching(id_prefix=ref, limit=2)
    else:
        rows = []
    task = load_task(str(rows[0]["id"])) if len(rows) == 1 else None
    if task is None:
        return f"Error: no single task matches {ref or 'latest'!r}.\n{recent_runs()}"
    return task


def _edit_text(args: dict[str, Any]) -> str:
    """Return the replacements of an ``Edit`` call as ``old`` / ``new`` blocks."""
    pairs = _edit_pairs(str(args["edits"])) if args.get("edits") else [(
        str(args.get("old_string") or ""), str(args.get("new_string") or ""),
        str(args.get("replace_all")).lower() == "true",
    )]
    if isinstance(pairs, str):
        return pairs
    return "\n".join(
        f"old{' (every occurrence)' if every else ''}:\n{_block(_ends(old, _EDIT_CHARS))}\n"
        f"new:\n{_block(_ends(new, _EDIT_CHARS))}"
        for old, new, every in pairs
    )


def _entry(call: Call) -> tuple[str, str] | None:
    """Return the evidence entry ``(heading, body)`` of *call*, or ``None`` when it is none."""
    name = os.path.basename(str(call.args.get("path") or ""))
    failed = f"\nNOT APPLIED, the call failed: {clip(call.output, 300)}" if call.failed else ""
    part = NESTED_MARK if call.nested else ""
    if call.name == "Write" and call.touches(LOOP_FILES):
        limit = _FILE_CHARS if name in FRONTIER_FILES else _SIDE_FILE_CHARS
        return f"Write {name}{part}", _block(_ends(call.args.get("content"), limit)) + failed
    if call.name == "Edit" and call.touches(LOOP_FILES):
        return f"Edit {name}", _edit_text(call.args) + failed
    if call.name == "Read" and call.touches(FRONTIER_FILES):
        return f"Read {name}", _block(_ends(call.output, _FILE_CHARS))
    command = str(call.args.get("command") or "")
    if call.name == "Bash" and (call.touches(LOOP_FILES) or DRAW_CALL in command):
        return f"Bash{part}", (
            f"command:\n{_block(_ends(command, _COMMAND_CHARS))}\n"
            f"output:\n{_block(_ends(call.output, _OUTPUT_CHARS))}"
        )
    if call.name == "decide":
        return "decide", (
            f"state:\n{_block(_ends(call.args.get('state'), _DECIDE_CHARS))}\n"
            f"questions:\n{_block(_ends(call.args.get('questions'), _DECIDE_CHARS))}\n"
            f"answer:\n{_block(_ends(call.output, _DECIDE_CHARS))}"
        )
    return None


def _disk_copies(task: dict[str, Any], calls: list[Call]) -> list[tuple[str, str]]:
    """Return ``(path, text)`` of each loop file of the run that is still on disk.

    A file is looked for where the trajectory wrote or read it, then in
    the run's default scratch directory (``<task id>`` in the agents'
    temp directory, :func:`~kiss.core.kiss_tmp.kiss_tmp_dir`), each also
    in the main checkout when the path is inside a worktree (the
    worktree is gone once the run has merged).
    """
    work_dir = str(task.get("work_dir") or "")
    folders = [
        os.path.dirname(os.path.join(work_dir, str(call.args["path"])))
        for call in calls
        if call.name in ("Write", "Edit", "Read") and call.touches(LOOP_FILES)
    ]
    folders.append(str(kiss_tmp_dir() / str(task["id"])))
    folders += [_WORKTREE_SEGMENT.sub("", folder) for folder in folders]
    copies = []
    for name in LOOP_FILES:
        found = next((p for f in folders if (p := Path(f, name)).is_file()), None)
        if found is not None:
            copies.append((str(found), found.read_text(encoding="utf-8", errors="replace")))
    return copies


def evidence(task: dict[str, Any]) -> str:
    """Return the evidence of the ``/discover`` run *task* as Markdown.

    Args:
        task: The run's history row with its ``events`` (:func:`find_task`).

    Returns:
        A header (request, state, counters), the timeline (one numbered
        entry per ``Write`` / ``Edit`` of a loop file, ``Read`` of a
        frontier file, shell command that names a loop file or draws a
        random number, and ``decide`` call, each with its output), the
        loop files still on disk and the final result; ``""`` when no
        call wrote or read ``pareto.md`` or ``explored-ideas.md``, that
        is when the task ran no discovery loop.
    """
    calls = tool_calls(task["events"])
    if not any(call.name != "decide" and call.touches(FRONTIER_FILES) for call in calls):
        return ""
    entries = [entry for call in calls if (entry := _entry(call)) is not None]
    lines = [
        f"# Evidence of the /discover run {task['id']}",
        "",
        f"Request: {clip(task.get('task'), _PROMPT_CHARS)}",
        f"State: {_state(task)}",
        f"Steps: {task.get('steps')}, cost: ${float(task.get('cost') or 0):.2f}, "
        f"model: {task.get('model')}, work dir: {task.get('work_dir')}",
        "",
        f"## Timeline: {len(entries)} entries, in the order the run made the calls",
        "A shell here-document into an existing file was run by the harness as a Write of the "
        "whole file, and the rest of that command as a Bash call of its own; the heading of "
        f'such an entry ends "{NESTED_MARK.strip()}".',
    ]
    for number, (heading, body) in enumerate(entries, 1):
        lines += ["", f"### [{number}] {heading}", body]
    for path, text in _disk_copies(task, calls):
        lines += ["", f"## On disk now: {path}", _block(_ends(text, _DISK_CHARS))]
    result = html.unescape(_TAG.sub(" ", str(task.get("result") or "")))
    lines += ["", "## Final result of the run", _block(_ends(result, _RESULT_CHARS)), ""]
    return "\n".join(lines)
