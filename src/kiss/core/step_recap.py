# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The framework-written progress recap emitted every :data:`SUMMARY_EVERY_STEPS` steps.

A Sorcar task used to be told to call its ``summary`` tool on every
tenth step, and a guard refused every other tool call while the
summary was overdue.  Those summary turns were one model call each
(a full-context call late in a task) and the refusals a second one;
the recap itself is just a list of what the last ten steps did, which
the executor knows exactly.  So the executor keeps a
:class:`ToolRecord` per tool call and, every :data:`SUMMARY_EVERY_STEPS`
steps on which the model did not write a summary of its own, emits the
recap built by :func:`recap` as a ``summary`` tool-call event — the
event the chat panel already folds the preceding panels into.  The
``summary`` tool stays available for the checkpoints a model wants to
mark itself (before a long wait, before a merge, at finish).
"""

from __future__ import annotations

from dataclasses import dataclass

# A recap is emitted once this many steps passed since the last one
# (the model's own ``summary`` call counts as one).
SUMMARY_EVERY_STEPS = 10
# Characters kept of a record's argument and result lines.
RECORD_CHARS = 160
# Tool-call arguments whose value is the call's natural title.
_TITLE_ARGS = ("description", "command", "file_path", "path", "query", "task", "url", "name")


def first_line(text: object, limit: int = RECORD_CHARS) -> str:
    """Return the first non-empty line of *text*, clipped to *limit* characters.

    Args:
        text: Any value; ``None`` and empty strings give ``""``.
        limit: Maximum characters kept (an ellipsis marks a cut).
    """
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped if len(stripped) <= limit else stripped[: limit - 1] + "…"
    return ""


def describe_args(args: dict[str, object]) -> str:
    """Return the one-line title of a tool call from its arguments.

    The first present argument of :data:`_TITLE_ARGS` is the title (a
    ``Bash`` call reads by its ``description``, a ``Read`` by its
    path); a call with none of them shows its first argument.

    Args:
        args: The call's arguments.
    """
    for key in _TITLE_ARGS:
        if args.get(key):
            return first_line(args[key])
    for value in args.values():
        if value:
            return first_line(value)
    return ""


@dataclass
class ToolRecord:
    """One tool call as the recap shows it: step, tool, title and the result's first line."""

    step: int
    name: str
    args: str
    result: str
    is_error: bool = False


def recap(records: list[ToolRecord], first_step: int, last_step: int) -> str:
    """Return the Markdown recap of the records of steps *first_step*..*last_step*.

    One bullet per tool call: ``**step N** `tool` title → result``,
    an error result marked.  Steps without a tool call (a text-only
    turn) are not listed.

    Args:
        records: The executor's tool records (any range of steps).
        first_step: The first step the recap covers.
        last_step: The last step the recap covers.
    """
    lines = [f"Steps {first_step}-{last_step} (framework recap of the tool log):"]
    for rec in records:
        if rec.step < first_step or rec.step > last_step:
            continue
        line = f"- **step {rec.step}** `{rec.name}`"
        if rec.args:
            line += f" {rec.args}"
        if rec.result:
            line += f" → {'ERROR: ' if rec.is_error else ''}{rec.result}"
        lines.append(line)
    if len(lines) == 1:
        lines.append("- no tool calls")
    return "\n".join(lines)
