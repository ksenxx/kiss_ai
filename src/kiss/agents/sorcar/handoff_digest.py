# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Scripted digest of a session's trajectory for the hand-off summarizer.

When a session ends before the task does (context limit, budget, an
exception), the next session continues from a summary of what the
first one did.  That summary used to be written by a Read/Bash-equipped
model that sliced the trajectory JSON with ``python3 -c`` in 6 to 11
calls ($1.2 to $1.8 per hand-off in the 7-day audit of 2026-10-24).
The facts it was digging for are mechanical: :func:`handoff_digest`
extracts them here (the task text, the last ``summary`` texts, the
files written with their top-level symbols, the open errors and the
last tool calls with the first line of each result), and the model
writes the narrative in one tool-less call.

The trajectory is the executor's ``messages`` list (:attr:`KISSAgent.messages`):
a ``model`` message records each tool call as a ```` ```python name(k=v) ``` ````
block and the following ``user`` message carries the results as
``[name]: text`` paragraphs.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TASK_CHARS = 3000
"""Characters of the task text (the first user message) kept."""
RESULT_CHARS = 300
"""Characters of a tool result kept in the tool-call list."""
ERROR_CHARS = 600
"""Characters of a failing tool result kept in the open-errors list."""
SUMMARY_CHARS = 1500
"""Characters of each ``summary`` text kept."""
LAST_CALLS = 40
LAST_SUMMARIES = 3
LAST_ERRORS = 5
MAX_WRITTEN_FILES = 40
_CALL_BLOCK = re.compile(r"```python\n(.*?)\n```", re.S)
_ERROR_MARKS = ("Traceback (most recent call last)", "FAILED", "REFUSED")


@dataclass
class ToolCall:
    """One recorded tool call: its name, keyword arguments and result text."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: str = ""

    @property
    def is_error(self) -> bool:
        """Whether the result reads as a failure (an ``Error`` prefix, a traceback, a refusal)."""
        return self.result.startswith("Error") or any(m in self.result for m in _ERROR_MARKS)


def _parse_call(source: str) -> ToolCall | None:
    """Parse the one-line ``name(k=v, ...)`` repr the trajectory records for a call.

    A model message is the model's prose followed by the reprs of its
    calls, so a python block may also be a snippet the model wrote. The
    repr is one line: a single whitespace-free tool name (an identifier,
    dotted, or with the ``-`` an MCP tool name may carry) and a keyword-only
    argument list; anything else (several lines, positional arguments, an
    assignment or expression around the call) yields ``None``. A one-line
    block whose argument list does not parse is a call the executor refused
    for its malformed name and keeps the text before its ``(`` as name.
    """
    source = source.strip()
    name, paren, arguments = source.partition("(")
    name = name.strip()
    if "\n" in source or not paren:
        return None
    try:
        node = ast.parse("f(" + arguments, mode="eval").body
    except SyntaxError:
        return ToolCall(name or "?")
    if not isinstance(node, ast.Call) or node.args or len(name.split()) != 1:
        return None
    try:
        args = {str(kw.arg): ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
    except (ValueError, TypeError, RecursionError):
        args = {}
    return ToolCall(name, args)


def _attach_results(calls: list[ToolCall], text: str) -> None:
    """Fill ``call.result`` of *calls* from the ``[name]: text`` paragraphs of *text*."""
    position = 0
    for index, call in enumerate(calls):
        marker = f"[{call.name}]: "
        start = text.find(marker, position)
        if start < 0:
            return
        start += len(marker)
        end = len(text)
        if index + 1 < len(calls):
            next_marker = text.find(f"\n\n[{calls[index + 1].name}]: ", start)
            end = next_marker if next_marker >= 0 else end
        call.result = text[start:end]
        position = end


def parse_tool_calls(messages: list[dict[str, Any]]) -> list[ToolCall]:
    """Return every tool call of a trajectory, in order, with its result.

    Args:
        messages: The executor's ``messages`` list.
    """
    calls: list[ToolCall] = []
    pending: list[ToolCall] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if message.get("role") == "model":
            parsed = (_parse_call(block) for block in _CALL_BLOCK.findall(content))
            pending = [call for call in parsed if call is not None]
            calls.extend(pending)
        elif message.get("role") == "user" and pending:
            _attach_results(pending, content)
            pending = []
    return calls


def _clip(text: str, limit: int) -> str:
    """Return *text* cut to *limit* characters with a marker naming the full length."""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f" … [{len(text):,} chars]"


def _python_symbols(path: Path) -> list[str]:
    """Return the top-level function and class names of a Python file (``[]`` if unparsable)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return []
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.append(node.name)
    return names


def describe_written_files(paths: list[Path]) -> list[str]:
    """Return one line per written file: path, size and (for Python) its top-level symbols.

    Args:
        paths: Files the session wrote; missing ones are reported as deleted.
    """
    lines: list[str] = []
    for path in sorted(paths)[:MAX_WRITTEN_FILES]:
        if not path.is_file():
            lines.append(f"- {path} (no longer exists)")
            continue
        line = f"- {path} ({path.stat().st_size:,} bytes)"
        if path.suffix == ".py":
            symbols = _python_symbols(path)
            if symbols:
                line += ": " + ", ".join(symbols[:30])
        lines.append(line)
    if len(paths) > MAX_WRITTEN_FILES:
        lines.append(f"- (+{len(paths) - MAX_WRITTEN_FILES} more)")
    return lines


def _call_title(call: ToolCall) -> str:
    """Return ``name(arg, ...)`` with the call's arguments on one short line."""
    parts = []
    for key, value in call.args.items():
        text = " ".join(str(value).split())
        parts.append(f"{key}={text[:120]}{'…' if len(text) > 120 else ''}")
    return f"{call.name}({', '.join(parts)})"


def handoff_digest(
    messages: list[dict[str, Any]],
    written_files: list[Path],
    last_calls: int = LAST_CALLS,
    recaps: list[str] | None = None,
) -> str:
    """Return the Markdown digest of a session the summarizer writes its narrative from.

    Args:
        messages: The executor's ``messages`` list (its trajectory).
        written_files: Files the task has written so far.
        last_calls: How many of the newest tool calls are listed.
        recaps: The framework recaps the session emitted
            (``KISSAgent.recaps``); they are not in *messages*.
    """
    calls = parse_tool_calls(messages)
    task = next(
        (str(m.get("content")) for m in messages if m.get("role") == "user"), "",
    )
    sections = [f"## Task\n{_clip(task, TASK_CHARS)}"]

    # The model's own summaries come first and are never displaced by
    # framework recaps (whose tool calls the last section lists anyway);
    # recaps only fill the slots the model left empty.
    summaries = [
        str(c.args["description"])
        for c in calls
        if c.name == "summary" and c.args.get("description")
    ][-LAST_SUMMARIES:]
    spare = LAST_SUMMARIES - len(summaries)
    if recaps and spare > 0:
        summaries = list(recaps[-spare:]) + summaries
    if summaries:
        texts = [_clip(text, SUMMARY_CHARS) for text in summaries]
        sections.append("## Last progress summaries the session wrote\n" + "\n\n".join(texts))

    written = describe_written_files(written_files)
    if written:
        sections.append("## Files written by the task\n" + "\n".join(written))

    errors = [c for c in calls if c.is_error][-LAST_ERRORS:]
    if errors:
        lines = [f"- {_call_title(c)}\n  {_clip(c.result, ERROR_CHARS)}" for c in errors]
        sections.append("## Last failing tool calls\n" + "\n".join(lines))

    recent = calls[-last_calls:]
    if recent:
        first = len(calls) - len(recent) + 1
        lines = [
            f"{first + i}. {_call_title(c)} -> {_clip(c.result, RESULT_CHARS)}"
            for i, c in enumerate(recent)
        ]
        sections.append(
            f"## Last {len(recent)} of {len(calls)} tool calls (oldest first)\n" + "\n".join(lines)
        )
    return "\n\n".join(sections) + "\n"
