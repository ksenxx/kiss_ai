# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Compact digests of a persisted Sorcar task's transcript.

The persisted event stream of a task (``~/.kiss/sorcar.db``) is far
too large to hand to an LLM raw: thousands of streamed ``text_delta``
/ ``thinking_delta`` chunks, multi-kilobyte tool arguments and shell
output.  This module turns it into numbered *entries* (one per tool
call, tool result, coalesced thought / assistant text / shell output,
progress summary, user message, ``/ask`` answer, final result) and
renders them three ways:

* :func:`transcript_page` — a clipped, optionally filtered page of
  entries (the ``task_transcript`` tool of the task-update and
  ``/ask`` agents);
* :func:`overview` — everything needed to orient in one call: the
  header, sub-agent tasks, later user messages, previous ``/ask``
  answers, the task's own progress summaries and its latest entries;
* :func:`entry_detail` — one entry in full (a complete command
  output, diff or tool argument).

All three read the live database in-process (queued events are
flushed first), so a running task's newest steps are visible.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

# Per-kind clip limits (characters) applied when an entry is rendered.
_CLIP: dict[str, int] = {
    "THOUGHT": 400,
    "ASSISTANT": 800,
    "OUTPUT": 500,
    "TOOL CALL": 400,
    "RESULT": 500,
    "SUMMARY": 4000,
    "FINISH": 4000,
    "TASK RESULT": 4000,
    "USER": 1000,
    "ASK ANSWER": 1500,
}
_DEFAULT_CLIP = 500
_PROMPT_CHARS = 3000
_CHILD_TASK_CHARS = 160
_CHILD_RESULT_CHARS = 300
_OVERVIEW_SUMMARY_CHARS = 1500
_OVERVIEW_SUMMARIES_HEAD = 2
_OVERVIEW_SUMMARIES_TAIL = 10
_OVERVIEW_ASK_ANSWERS = 3
_OVERVIEW_TAIL_ENTRIES = 30
MAX_PAGE = 400
MAX_DETAIL_CHARS = 100_000

# Events that carry no progress information (UI signalling, streaming
# markers): the digest drops them without a trace.
_SKIPPED_EVENT_TYPES = frozenset({
    "system_prompt", "task_settings", "thinking_start", "thinking_end",
    "text_start", "text_end", "task_done", "new_tab", "tasks_updated",
    "subagentDone", "status", "model_pick", "agent_model_pick",
    "followup_suggestion", "usage_info",
})

# Tool arguments the printer lifts to the top level of a ``tool_call``
# event (``KNOWN_KEYS`` in :mod:`kiss.core.printer`, ``file_path`` and
# ``path`` both stored as ``path``); everything else sits under ``extras``.
_TOOL_CALL_TOP_LEVEL_ARGS = (
    "path", "description", "command", "content", "old_string", "new_string",
)

# Streamed event kinds: consecutive chunks coalesce into one entry.
_STREAMED_EVENT_TYPES = {
    "thinking_delta": "THOUGHT",
    "text_delta": "ASSISTANT",
    "system_output": "OUTPUT",
}


@dataclass
class Entry:
    """One digest entry: its *kind*, full *text* and, for tool calls, the tool *name*."""

    kind: str
    text: str
    name: str = ""

    def render(self, limit: int | None = None) -> str:
        """Return the entry as one clipped line block.

        Args:
            limit: Clip length in characters; ``None`` uses the kind's
                default from :data:`_CLIP`.
        """
        limit = _CLIP.get(self.kind, _DEFAULT_CLIP) if limit is None else limit
        if self.kind == "TOOL CALL":
            return f"TOOL CALL {self.name}({clip(self.text, limit)})"
        return f"{self.kind}: {clip(self.text, limit)}"


def clip(text: object, limit: int) -> str:
    """Return ``str(text)`` shortened to *limit* characters with a marker."""
    s = str(text or "")
    if len(s) <= limit:
        return s
    return s[:limit] + f" …[{len(s) - limit} more chars]"


class _StreamBuffer:
    """Collects consecutive streamed chunks of one kind into one entry."""

    def __init__(self) -> None:
        self.kind = ""
        self.parts: list[str] = []

    def add(self, entries: list[Entry], kind: str, text: str) -> None:
        """Buffer *text*; a change of *kind* first flushes the buffer."""
        if kind != self.kind:
            self.flush(entries)
            self.kind = kind
        self.parts.append(text)

    def flush(self, entries: list[Entry]) -> None:
        """Append the buffered chunks as one entry and clear the buffer."""
        if self.parts:
            text = "".join(self.parts).strip()
            if text:
                entries.append(Entry(self.kind, text))
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


def _tool_call_entry(ev: dict[str, Any]) -> Entry:
    """Return the entry of a ``tool_call`` event."""
    name = str(ev.get("name") or "?")
    args = _tool_call_args(ev)
    if name == "summary":
        return Entry("SUMMARY", str(args.get("description") or ""))
    if name == "finish":
        return Entry("FINISH", json.dumps(args, ensure_ascii=False))
    return Entry("TOOL CALL", json.dumps(args, ensure_ascii=False) if args else "", name)


def digest_events(events: list[dict[str, Any]]) -> tuple[list[Entry], str]:
    """Turn the persisted event stream into digest entries.

    Consecutive ``thinking_delta`` / ``text_delta`` / ``system_output``
    chunks coalesce into one entry each; every ``tool_call`` and
    ``tool_result`` becomes one entry.  The first ``prompt`` event
    (the chat-augmented task prompt) is dropped in favour of the
    history row's own task text; later ``prompt`` events are messages
    the user sent to the running task and become ``USER`` entries,
    except ``/ask`` questions, which surface through their
    ``ask_answer`` event as ``ASK ANSWER`` entries instead.

    Args:
        events: The task's event dicts in sequence order.

    Returns:
        ``(entries, spend)`` where *spend* is the text of the last
        ``usage_info`` event (``""`` when none was recorded).
    """
    entries: list[Entry] = []
    stream = _StreamBuffer()
    spend = ""
    seen_prompt = False
    for ev in events:
        kind = str(ev.get("type") or "")
        if kind in _STREAMED_EVENT_TYPES:
            stream.add(entries, _STREAMED_EVENT_TYPES[kind], str(ev.get("text") or ""))
            continue
        stream.flush(entries)
        if kind == "usage_info":
            spend = str(ev.get("text") or spend)
        elif kind == "prompt":
            text = str(ev.get("text") or "").strip()
            if seen_prompt and not text.startswith("/ask"):
                entries.append(Entry("USER", text))
            seen_prompt = True
        elif kind in _SKIPPED_EVENT_TYPES:
            continue
        elif kind == "tool_call":
            entries.append(_tool_call_entry(ev))
        elif kind == "tool_result":
            flag = " (error)" if ev.get("is_error") else ""
            entries.append(Entry(f"RESULT{flag}", str(ev.get("content") or "")))
        elif kind == "result":
            entries.append(Entry("TASK RESULT", str(ev.get("text") or "")))
        elif kind == "ask_answer":
            answer = str(ev.get("text") or "") or (
                "(no answer)" if ev.get("success") is False else ""
            )
            entries.append(Entry("ASK ANSWER", f"Q: {ev.get('question') or ''}\nA: {answer}"))
        else:
            text = ev.get("text") or ev.get("message") or ""
            if text:
                entries.append(Entry(kind.upper(), str(text)))
    stream.flush(entries)
    return entries, spend


def _event_ms(ev: dict[str, Any]) -> int:
    """Return an event's time in epoch milliseconds (``ts`` or the row timestamp), or 0."""
    try:
        return int(float(ev.get("ts") or 0)) or int(float(ev.get("_timestamp") or 0) * 1000)
    except (TypeError, ValueError):
        return 0


def _start_ms(task: dict[str, Any]) -> int:
    """Return the task's start time in epoch milliseconds.

    Legacy rows carry no ``start_ts``; they fall back to the row's
    insertion ``timestamp`` (seconds), the same fallback the history
    sidebar applies.
    """
    return int(task.get("start_ts") or 0) or int(float(task.get("timestamp") or 0) * 1000)


def _fmt_ts(ms: int) -> str:
    """Return *ms* (epoch milliseconds) as a UTC timestamp, or ``unknown``."""
    if not ms:
        return "unknown"
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ms / 1000))


def _fmt_duration(seconds: int) -> str:
    """Return *seconds* as ``H h M min S s`` (hours only when non-zero)."""
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes} min {secs} s"
    return f"{minutes} min {secs} s"


def load_task(task_id: str) -> dict[str, Any] | None:
    """Return the history row of *task_id* with its events, or ``None``.

    Queued events of the task are flushed first so the newest steps
    of a running task are included.  The dict carries every
    ``task_history`` column (``task``, ``result``, ``model``,
    ``work_dir``, ``start_ts``, ``end_ts``, ``cost``, ``steps``,
    ``chat_id``, ``parent_task_id``, ...) plus ``events`` (list of
    event dicts in sequence order).
    """
    from kiss.agents.sorcar import persistence

    persistence._flush_chat_events(task_id)
    with persistence._rw_lock.read_lock():
        db = persistence._get_db()
        row = db.execute(
            persistence._HISTORY_SELECT + "WHERE id = ?", (task_id,),
        ).fetchone()
        if row is None:
            return None
        task = persistence._history_row_to_dict(row)
        task["events"] = persistence._fetch_events_for_task_id(db, task_id)
    return task


def child_tasks(task_id: str) -> list[dict[str, Any]]:
    """Return the history rows of the sub-agent tasks *task_id* dispatched, oldest first."""
    from kiss.agents.sorcar import persistence

    with persistence._rw_lock.read_lock():
        rows = persistence._get_db().execute(
            persistence._HISTORY_SELECT
            + "WHERE parent_task_id = ? ORDER BY timestamp ASC, rowid ASC",
            (task_id,),
        ).fetchall()
    return [persistence._history_row_to_dict(r) for r in rows]


def _status(task: dict[str, Any]) -> str:
    """Return ``running`` / ``finished`` and, for a finished task, the clipped result."""
    end_ms = int(task.get("end_ts") or 0)
    if not end_ms:
        return "running"
    return f"finished ({clip(task.get('result'), 200)})"


def header_lines(task: dict[str, Any], spend: str, total: int) -> list[str]:
    """Return the digest header lines (task metadata, timing and spend)."""
    start_ms = _start_ms(task)
    end_ms = int(task.get("end_ts") or 0)
    now_ms = int(time.time() * 1000)
    elapsed_s = ((end_ms or now_ms) - start_ms) // 1000 if start_ms else 0
    lines = [
        f"Task id: {task.get('id')}",
        f"Task prompt: {clip(task.get('task'), _PROMPT_CHARS)}",
        f"Status: {_status(task)}",
        f"Chat id: {task.get('chat_id') or ''}",
        f"Model: {task.get('model') or ''}",
        f"Work dir: {task.get('work_dir') or ''}",
        f"Started: {_fmt_ts(start_ms)}",
        f"Elapsed: {_fmt_duration(elapsed_s)}",
    ]
    if end_ms:
        lines.append(f"Ended: {_fmt_ts(end_ms)}")
    lines.append(f"Spend: {spend or 'not recorded yet'}")
    lines.append(f"Transcript entries: {total}")
    if task.get("parent_task_id"):
        lines.append(f"Sub-agent of task: {task['parent_task_id']}")
    return lines


def _digest(task_id: str) -> tuple[dict[str, Any], list[Entry], str] | str:
    """Load and digest *task_id*; return an error line instead when it is unknown."""
    task_id = str(task_id or "").strip()
    if not task_id:
        return "Error: no task id given."
    task = load_task(task_id)
    if task is None:
        return f"Error: no task with id {task_id!r}."
    entries, spend = digest_events(task["events"])
    return task, entries, spend


def transcript_page(task_id: str, start: int = 0, count: int = 150, contains: str = "") -> str:
    """Return a page of a task's digested transcript.

    Args:
        task_id: The ``task_history`` row id (32 hex characters).
        start: Index of the first entry to consider (0-based).
        count: Maximum number of entries to return (at most :data:`MAX_PAGE`).
        contains: Optional filter: one or more literal search terms
            separated by ``|``; only entries whose full text contains
            at least one term (case-insensitive) are returned, keeping
            their original indices.  Literal matching (not regex) so a
            pathological pattern can never stall the tool.

    Returns:
        The header, the numbered entries, and a line telling how many
        entries remain; or an error line when no task has that id.
    """
    loaded = _digest(task_id)
    if isinstance(loaded, str):
        return loaded
    task, entries, spend = loaded
    terms = [t.lower() for t in contains.split("|") if t]
    start = max(0, int(start))
    count = max(1, min(int(count), MAX_PAGE))
    lines = header_lines(task, spend, len(entries))
    lines.append("")
    if not entries:
        lines.append("(no transcript entries yet)")
        return "\n".join(lines)
    candidates = [
        (i, e) for i, e in enumerate(entries)
        if i >= start and (not terms or _contains_any(e, terms))
    ]
    page = candidates[:count]
    remaining = len(candidates) - len(page)
    if terms:
        lines.append(
            f"Entries containing {contains!r} from index {start}: "
            f"{len(page)} shown, {remaining} more."
        )
    elif page:
        lines.append(f"Entries {page[0][0]}..{page[-1][0]} of {len(entries)}:")
    for i, entry in page:
        lines.append(f"[{i}] {entry.render()}")
    if remaining > 0:
        next_start = page[-1][0] + 1
        lines.append(f"... {remaining} more entries; call again with start={next_start}.")
    else:
        lines.append("(end of transcript)")
    return "\n".join(lines)


def _one_line(text: object, limit: int) -> str:
    """Return *text* with runs of whitespace collapsed, clipped to *limit*."""
    return clip(" ".join(str(text or "").split()), limit)


def _contains_any(entry: Entry, terms: list[str]) -> bool:
    """Return whether the entry's full rendered text contains one of the lower-cased *terms*."""
    text = entry.render(10**9).lower()
    return any(t in text for t in terms)


def _child_line(child: dict[str, Any]) -> str:
    """Return the two-line overview of a sub-agent task row."""
    cost = float(child.get("cost") or 0.0)
    status = "running" if not child.get("end_ts") else (
        f"finished: {_one_line(child.get('result'), _CHILD_RESULT_CHARS)}"
    )
    return (
        f"[{child.get('id')}] steps={child.get('steps') or 0}, cost=${cost:.2f}, "
        f"started {_fmt_ts(_start_ms(child))}, {status}\n"
        f"    task: {_one_line(child.get('task'), _CHILD_TASK_CHARS)}"
    )


def _summary_section(indexed: list[tuple[int, Entry]]) -> list[str]:
    """Return the overview lines for the task's own progress summaries."""
    head, tail = _OVERVIEW_SUMMARIES_HEAD, _OVERVIEW_SUMMARIES_TAIL
    if len(indexed) > head + tail:
        omitted = len(indexed) - head - tail
        shown = indexed[:head] + [(-1, Entry("", ""))] + indexed[-tail:]
    else:
        omitted, shown = 0, indexed
    lines = []
    for i, entry in shown:
        if i < 0:
            lines.append(f"... {omitted} earlier summaries omitted (see task_transcript) ...")
        else:
            lines.append(f"[{i}] {entry.render(_OVERVIEW_SUMMARY_CHARS)}")
    return lines


def overview(task_id: str) -> str:
    """Return the one-call orientation digest of a task.

    Sections: header (status, timing, spend, current time), sub-agent
    tasks, user messages sent after the initial prompt, the latest
    ``/ask`` answers, the task's own progress summaries, and the last
    :data:`_OVERVIEW_TAIL_ENTRIES` transcript entries.

    Args:
        task_id: The ``task_history`` row id (32 hex characters).

    Returns:
        The digest text, or an error line when no task has that id.
    """
    loaded = _digest(task_id)
    if isinstance(loaded, str):
        return loaded
    task, entries, spend = loaded
    events = task["events"]
    last_ms = _event_ms(events[-1]) if events else 0
    now = time.time()
    lines = ["== Task ==", *header_lines(task, spend, len(entries))]
    if last_ms:
        ago = _fmt_duration(int(now - last_ms / 1000))
        lines.append(f"Last event: {_fmt_ts(last_ms)} ({ago} ago)")
    lines.append(
        f"Now: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(now))} "
        f"(local: {time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(now))})"
    )
    # Side-channel children (earlier /ask and task-update runs) are not
    # the task's own work; their answers already appear as ASK ANSWER
    # entries, so only the count is shown.
    children = child_tasks(str(task["id"]))
    workers = [c for c in children if not c.get("is_side_channel")]
    lines += [
        "",
        f"== Sub-agent tasks ({len(workers)}; plus {len(children) - len(workers)} "
        "side-channel /ask or task-update runs not listed) ==",
    ]
    lines += [_child_line(c) for c in workers] or ["(none)"]
    indexed = list(enumerate(entries))
    users = [(i, e) for i, e in indexed if e.kind == "USER"]
    lines += ["", f"== User messages after the initial prompt ({len(users)}) =="]
    lines += [f"[{i}] {e.render()}" for i, e in users] or ["(none)"]
    asks = [(i, e) for i, e in indexed if e.kind == "ASK ANSWER"]
    shown_asks = asks[-_OVERVIEW_ASK_ANSWERS:]
    lines += ["", f"== Previous /ask answers ({len(shown_asks)} of {len(asks)}) =="]
    lines += [f"[{i}] {e.render()}" for i, e in shown_asks] or ["(none)"]
    summaries = [(i, e) for i, e in indexed if e.kind == "SUMMARY"]
    lines += ["", f"== Progress summaries written by the task ({len(summaries)}) =="]
    lines += _summary_section(summaries) or ["(none)"]
    tail = indexed[-_OVERVIEW_TAIL_ENTRIES:]
    lines += ["", f"== Last {len(tail)} of {len(entries)} transcript entries =="]
    lines += [f"[{i}] {e.render()}" for i, e in tail] or ["(no transcript entries yet)"]
    lines += [
        "",
        "Indices in [brackets] are transcript entry numbers: task_transcript(task_id, "
        "start, count, contains) pages or filters all entries; task_step(task_id, "
        "index) returns one entry in full.",
    ]
    return "\n".join(lines)


def entry_detail(task_id: str, index: int, max_chars: int = 20_000) -> str:
    """Return one transcript entry of a task in full.

    Args:
        task_id: The ``task_history`` row id (32 hex characters).
        index: The entry number shown in brackets by the other digests.
        max_chars: Clip length for the entry text (at most :data:`MAX_DETAIL_CHARS`).

    Returns:
        ``[index] KIND: full text``; or an error line when the task or
        the index does not exist.
    """
    loaded = _digest(task_id)
    if isinstance(loaded, str):
        return loaded
    _task, entries, _spend = loaded
    index = int(index)
    if not 0 <= index < len(entries):
        return f"Error: entry index {index} out of range (0..{len(entries) - 1})."
    limit = max(1, min(int(max_chars), MAX_DETAIL_CHARS))
    return f"[{index}] {entries[index].render(limit)}"
