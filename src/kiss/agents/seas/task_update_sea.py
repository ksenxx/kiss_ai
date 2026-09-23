# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Task-update agent — reports what a running task has done so far.

Two ways to run it:

* ``/task_update <task_id>`` in the chat runs it as a sub-task through
  ``run_agent`` (see :mod:`kiss.agents.sorcar.sea_commands`).
* The daemon runs it in-process for the task shown in the visible chat
  webview (:mod:`kiss.server.task_update`): once when the panel first
  shows the task, then every 10 minutes, and whenever the panel's
  refresh button is pressed.  The result replaces the ``tmp/PROGRESS.md``
  mirror in the task-info panel.

The agent reads the task's persisted transcript (``~/.kiss/sorcar.db``)
through the :func:`task_transcript` tool defined here and answers
:data:`PROMPT_TEMPLATE` with a short markdown progress report.
"""

from __future__ import annotations

import json
import time
from typing import Any

PROMPT_TEMPLATE = (
    "What have the task with {task_id} done so far and what are the partial results?"
)

SYSTEM_PROMPT = """You report the progress of another Sorcar task.

The user's prompt names a task id (a 32-character hex string). If the prompt
is nothing but a task id, treat it as: "What have the task with <task_id>
done so far and what are the partial results?".

Procedure:
1. Call `task_transcript(task_id)` to read the task's persisted transcript
   (its prompt, tool calls, tool results, periodic `SUMMARY` entries, spend). It
   returns entries in pages; when the transcript has more entries than one
   page, page through the rest with `start` until you have seen every entry.
2. Call `finish` with `success=true` and, as `summary_in_html`, a concise
   progress report (under 400 words) in compact HTML (`<h4>` headings,
   `<ul><li>` bullets, `<p>`; no `<html>`/`<body>` wrapper) with these
   sections, each a short bullet list:
   - **Goal**: the task's request in one or two sentences.
   - **Done so far**: the concrete steps taken, in order (files read or
     changed, commands run, findings). Prefer the `SUMMARY` entries the task
     wrote itself; use the raw entries for the period after the last summary.
   - **Partial results**: results already produced (numbers, files, answers,
     decisions), or "None yet" when nothing is produced.
   - **Current activity**: what the task is doing in its latest entries.
   - **Spend**: steps, tokens and cost so far, from the transcript header.

Rules: report only what the transcript shows; never guess or embellish. Do
not quote long transcript excerpts. If the task id is missing or unknown,
finish with a one-line message saying so. If the task has already finished,
say so and report its final result.
"""

# A digest never carries more than this many characters per entry.
_PROMPT_CHARS = 3000
_SUMMARY_CHARS = 4000
_RESULT_CHARS = 4000
_TOOL_CALL_CHARS = 400
_TOOL_RESULT_CHARS = 500
_TEXT_CHARS = 800
_THOUGHT_CHARS = 400
_MAX_PAGE = 400

# Events that carry no progress information (UI signalling, streaming
# markers): the digest drops them without a trace.
_SKIPPED_EVENT_TYPES = frozenset({
    "system_prompt", "task_settings", "thinking_start", "thinking_end",
    "text_start", "text_end", "task_done", "new_tab", "tasks_updated",
    "subagentDone", "status", "model_pick", "agent_model_pick",
    "followup_suggestion", "usage_info",
    # The prompt event holds the chat-augmented prompt (previous tasks
    # first, the current task last); the header carries the task's own
    # prompt from its history row instead.
    "prompt",
})

# Tool arguments the printer lifts to the top level of a ``tool_call``
# event (``KNOWN_KEYS`` in :mod:`kiss.core.printer`, ``file_path`` and
# ``path`` both stored as ``path``); everything else sits under ``extras``.
_TOOL_CALL_TOP_LEVEL_ARGS = (
    "path", "description", "command", "content", "old_string", "new_string",
)


def build_prompt(task_id: str) -> str:
    """Return the agent's prompt for *task_id* (:data:`PROMPT_TEMPLATE`).

    Args:
        task_id: The ``task_history`` row id of the task to report on.

    Returns:
        The filled-in prompt text.
    """
    return PROMPT_TEMPLATE.format(task_id=task_id)


def _clip(text: object, limit: int) -> str:
    """Return ``str(text)`` shortened to *limit* characters with a marker."""
    s = str(text or "")
    if len(s) <= limit:
        return s
    return s[:limit] + f" …[{len(s) - limit} more chars]"


# Streamed event kinds: consecutive chunks coalesce into one entry with
# the given label, clipped to the given length.
_STREAMED_EVENT_TYPES = {
    "thinking_delta": ("THOUGHT", _THOUGHT_CHARS),
    "text_delta": ("ASSISTANT", _TEXT_CHARS),
    "system_output": ("OUTPUT", _TOOL_RESULT_CHARS),
}


class _StreamBuffer:
    """Collects consecutive streamed chunks of one kind into one entry."""

    def __init__(self) -> None:
        self.kind = ""
        self.parts: list[str] = []

    def add(self, entries: list[str], kind: str, text: str) -> None:
        """Buffer *text*; a change of *kind* first flushes the buffer."""
        if kind != self.kind:
            self.flush(entries)
            self.kind = kind
        self.parts.append(text)

    def flush(self, entries: list[str]) -> None:
        """Append the buffered chunks as one entry and clear the buffer."""
        if self.parts:
            label, limit = _STREAMED_EVENT_TYPES[self.kind]
            text = "".join(self.parts).strip()
            if text:
                entries.append(f"{label}: {_clip(text, limit)}")
        self.kind = ""
        self.parts = []


def _tool_call_args(ev: dict[str, Any]) -> dict[str, Any]:
    """Reassemble a ``tool_call`` event's arguments from its two homes."""
    # Presence, not truthiness: an Edit that deletes text carries
    # ``new_string: ""`` and that must survive.
    args: dict[str, Any] = {
        k: ev[k] for k in _TOOL_CALL_TOP_LEVEL_ARGS if ev.get(k) is not None
    }
    extras = ev.get("extras")
    if isinstance(extras, dict):
        args.update(extras)
    return args


def _digest_events(events: list[dict[str, Any]]) -> tuple[list[str], str]:
    """Turn the persisted event stream into compact digest entries.

    Consecutive ``thinking_delta`` / ``text_delta`` / ``system_output``
    chunks coalesce into one entry each; every ``tool_call`` and
    ``tool_result`` becomes one clipped entry, with the ``summary``
    tool's description (the task's own periodic progress log) kept
    nearly whole.

    Args:
        events: The task's event dicts in sequence order.

    Returns:
        ``(entries, spend)`` where *spend* is the text of the last
        ``usage_info`` event (``""`` when none was recorded).
    """
    entries: list[str] = []
    stream = _StreamBuffer()
    spend = ""
    for ev in events:
        kind = str(ev.get("type") or "")
        if kind in _STREAMED_EVENT_TYPES:
            stream.add(entries, kind, str(ev.get("text") or ""))
            continue
        stream.flush(entries)
        if kind == "usage_info":
            spend = str(ev.get("text") or spend)
            continue
        if kind in _SKIPPED_EVENT_TYPES:
            continue
        if kind == "tool_call":
            name = str(ev.get("name") or "?")
            args = _tool_call_args(ev)
            if name == "summary":
                entries.append(
                    f"SUMMARY: {_clip(args.get('description'), _SUMMARY_CHARS)}"
                )
            elif name == "finish":
                entries.append(
                    f"FINISH: {_clip(json.dumps(args, ensure_ascii=False), _RESULT_CHARS)}"
                )
            else:
                arg_text = json.dumps(args, ensure_ascii=False) if args else ""
                entries.append(f"TOOL CALL {name}({_clip(arg_text, _TOOL_CALL_CHARS)})")
        elif kind == "tool_result":
            flag = " (error)" if ev.get("is_error") else ""
            entries.append(
                f"RESULT{flag}: {_clip(ev.get('content'), _TOOL_RESULT_CHARS)}"
            )
        elif kind == "result":
            entries.append(f"TASK RESULT: {_clip(ev.get('text'), _RESULT_CHARS)}")
        else:
            text = ev.get("text") or ev.get("message") or ""
            if text:
                entries.append(f"{kind.upper()}: {_clip(text, _TOOL_RESULT_CHARS)}")
    stream.flush(entries)
    return entries, spend


def _header(task_id: str, session: dict[str, Any], spend: str, total: int) -> list[str]:
    """Return the digest header lines (task metadata and spend)."""
    try:
        extra = json.loads(str(session.get("extra") or "") or "{}")
    except ValueError:
        extra = {}
    if not isinstance(extra, dict):
        extra = {}
    start_ms = int(extra.get("startTs") or 0)
    end_ms = int(extra.get("endTs") or 0)
    started = (
        time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(start_ms / 1000))
        if start_ms else "unknown"
    )
    if end_ms:
        status = "finished"
        elapsed_s = max(0, (end_ms - start_ms) // 1000) if start_ms else 0
    else:
        status = "running"
        elapsed_s = max(0, int(time.time() - start_ms / 1000)) if start_ms else 0
    lines = [
        f"Task id: {task_id}",
        f"Task prompt: {_clip(session.get('task'), _PROMPT_CHARS)}",
        f"Status: {status}",
        f"Chat id: {session.get('chat_id') or ''}",
        f"Model: {extra.get('model') or ''}",
        f"Work dir: {extra.get('work_dir') or ''}",
        f"Started: {started}",
        f"Elapsed: {elapsed_s // 60} min {elapsed_s % 60} s",
        f"Spend: {spend or 'not recorded yet'}",
        f"Transcript entries: {total}",
    ]
    if extra.get("subagent"):
        lines.append(f"Sub-agent of task: {extra['subagent'].get('parent_task_id', '')}")
    return lines


def task_transcript(task_id: str, start: int = 0, count: int = 150) -> str:
    """Return a page of the digested transcript of a Sorcar task.

    The digest is built from the task's persisted events: every tool
    call and clipped tool result, the task's own periodic ``SUMMARY``
    entries, coalesced assistant text, thoughts and shell output, and
    the final result when the task has finished.  A header gives the
    task's prompt, status, model, work dir, start time, elapsed time
    and spend.

    Args:
        task_id: The ``task_history`` row id (32 hex characters).
        start: Index of the first digest entry to return (0-based).
        count: Number of entries to return (at most 400).

    Returns:
        The header followed by the numbered entries ``start`` to
        ``start + count - 1``, and a line telling how many entries
        remain; or an error line when no task has that id.
    """
    from kiss.agents.sorcar.persistence import (
        _flush_chat_events,
        _load_chat_events_by_task_id,
    )

    task_id = str(task_id or "").strip()
    if not task_id:
        return "Error: no task id given."
    _flush_chat_events(task_id)
    session = _load_chat_events_by_task_id(task_id)
    if session is None:
        return f"Error: no task with id {task_id!r}."
    events = session.get("events")
    if not isinstance(events, list):
        events = []
    entries, spend = _digest_events([e for e in events if isinstance(e, dict)])
    total = len(entries)
    start = max(0, int(start))
    count = max(1, min(int(count), _MAX_PAGE))
    page = entries[start:start + count]
    lines = _header(task_id, session, spend, total)
    lines.append("")
    if not entries:
        lines.append("(no transcript entries yet)")
        return "\n".join(lines)
    lines.append(f"Entries {start}..{start + len(page) - 1} of {total}:")
    for i, entry in enumerate(page, start):
        lines.append(f"[{i}] {entry}")
    remaining = total - (start + len(page))
    if remaining > 0:
        lines.append(f"... {remaining} more entries; call again with start={start + len(page)}.")
    else:
        lines.append("(end of transcript)")
    return "\n".join(lines)


def system_prompt() -> str:
    """Return the agent's base system prompt (:data:`SYSTEM_PROMPT`)."""
    return SYSTEM_PROMPT


def tools() -> list[Any]:
    """Return the agent's tools: :func:`task_transcript`."""
    return [task_transcript]


def tool_profile() -> str:
    """Return the built-in tool profile: ``"bash"`` (Bash and finish only)."""
    return "bash"


def max_budget() -> float:
    """Return the per-run budget cap in USD."""
    return 1.0


def is_parallel() -> bool:
    """Return whether the agent may fan out sub-agents: never."""
    return False


def use_web_tools() -> bool:
    """Return whether the agent gets the browser tools: never."""
    return False


def use_memory() -> bool:
    """Return whether the agent gets the memory tools: never."""
    return False


def use_worktree() -> bool:
    """Return whether the agent runs in a git worktree: never."""
    return False


def auto_commit() -> bool:
    """Return whether the agent's changes are auto-committed: never."""
    return False


def classify_tasks() -> bool:
    """Return whether the framework classifies the prompt first: never."""
    return False
