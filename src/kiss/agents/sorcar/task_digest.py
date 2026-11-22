# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Compact digests of a persisted Sorcar task's transcript.

The persisted event stream of a task (``$KISS_HOME/history.db``) is far
too large to hand to an LLM raw: thousands of streamed ``text_delta``
/ ``thinking_delta`` chunks, multi-kilobyte tool arguments and shell
output.  This module turns it into numbered *entries* (one per tool
call, tool result, coalesced thought / assistant text / shell output,
progress summary, user message, ``/ask`` answer, final result) and
renders them four ways:

* :func:`transcript_page` — a clipped, optionally filtered page of
  entries (the task-update agent's ``task_transcript`` tool and the
  ``rsi7d`` agent's ``run_transcript``);
* :func:`overview` — everything needed to orient in one call: the
  header, sub-agent tasks, later user messages, previous ``/ask``
  answers, the task's own progress summaries and its latest entries
  (the ``rsi7d`` agent's ``run_overview``);
* :func:`entry_detail` — one entry in full (a complete command
  output, diff or tool argument; the ``rsi7d`` agent's ``run_entry``);
* :func:`context` — the one-shot context of the ``/ask`` agent: the
  header, the sub-agent tasks, the tail of the progress log the task
  wrote and as many of the newest transcript entries as fit a fixed
  character budget.

All of them read the live database in-process (queued events are
flushed first), so a running task's newest steps are visible.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kiss.agents.sorcar.chat_summary import end_ms, launch_ms
from kiss.agents.sorcar.edit_guards import HeredocCopy, _home_path, heredoc_copy
from kiss.agents.sorcar.edit_matching import reindent, tolerant_matches
from kiss.agents.sorcar.shell_scan import (
    Group,
    SimpleCommand,
    TooDeepError,
    Word,
    bash_path,
    host_path,
    inherited_cdpath,
    resolve,
    simple_commands,
)
from kiss.agents.sorcar.useful_tools import _absolutize, _crlf_edit, _edit_pairs

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
# A fan-out of hundreds of sub-agents would otherwise fill the whole
# context (:data:`MAX_CONTEXT_CHARS`) with child lines alone.
_CHILDREN_HEAD = 2
_CHILDREN_TAIL = 20
MAX_PAGE = 400
# Character budget of :func:`context` (~15-20k tokens): the whole
# context fits a single cheap model call next to the question.
MAX_CONTEXT_CHARS = 60_000
# Only the newest progress-log entries matter for a question about
# the current state of the task.
_PROGRESS_LOG_CHARS = 8_000
# File names of a task's progress log: every task keeps ``PROGRESS.md`` in
# its scratch dir; ``tmp/PROGRESS.md`` in the work dir and ``PROGRESS.md``
# / ``PROGRESS_LOG.md`` at its root are older conventions.
_PROGRESS_LOG_NAMES = frozenset({"PROGRESS.md", "PROGRESS_LOG.md"})
# What a container's batch ``Edit`` answers when it stopped part-way
# (:meth:`~kiss.agents.sorcar.docker_tools.DockerTools.Edit` writes edit by
# edit; the host one writes all or nothing): group 1 counts the edits applied.
_PARTIAL_BATCH = re.compile(r"Error: edit \d+: edits 1-(\d+) applied; ")
# How the result of a ``Write`` / ``Edit`` that changed nothing opens: an error, a refusal.
_UNMADE_RESULT = ("Error", "REFUSED")
# How the ``Bash`` tools say that a command did not run to an exit status 0.
# The host tool opens such a result with words of its own: a failed command's
# exit code, an error or a timeout, a refusal, a job started in the background
# or left to it, a rewrite that was not applied.  The container tool closes a
# failed command's output with its exit code, and only its own notes come
# after that line (a poll's verdict, a long sleep's; the note of a sleep spent
# on a job starts a line of its own, and was glued to the line before it in
# the records of older tasks).  The words are exact (output that opens "Error
# recovery plan" or "REFUSED" is a command's own), and a daemon's record of a
# call whose output was streamed holds nothing but them
# (``kiss.core.printer.bash_status_lines``).
_NOT_PASSED = re.compile(
    r"\A(?:Error: |Error \(exit code -?\d+\):|REFUSED: |Not applied: "
    r"|Started background job \S.* \(pid \d+\)"
    r"|Still running after \S+ left running as background job )"
    r"|^\[exit code: -?\d+\]"
    r"(?:\n\[until [^\n]*|\n\[note: the sleep of [^\n]*|\n?Waited on job [^\n]*)*\n?\Z",
    re.MULTILINE,
)
# How the result of a ``Bash`` call opens whose command the tool made as a
# ``Write`` / ``Edit`` (group 1) that left no events of its own (a run without
# redirects).  The rest of the command ran after it: its status is further
# down in the result, and does not open it.  A command may print the same
# words itself and exit 0, so such a result proves what both readings do;
# one that printed them and failed is a :data:`_NOT_PASSED` result, read first.
_REWRITTEN = re.compile(r"(?:Applied as|Appended through) (Write|Edit)\(")
# The keyword ``coproc`` as a command spells it: a line the shell joins counts.
_COPROC = re.compile(r"(?:\\\n)*".join("coproc"))
# The most spellings of it :func:`_runs_coproc` reads, at one scan each: no log
# names the word that often, and a text that does is not worth the time.
_COPROC_PROBES = 1000
# The most places :func:`_without_bodies` asks the scan about, at one scan each.
_BODY_PROBES = 50
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# The variables that the shell, the loader or a locale gives a meaning
# (bash(1), "Shell Variables"; ld.so(8)), but ``IFS`` and ``CDPATH``, which
# the scan follows: assigning one may change what a later word runs or names,
# and the value of one that is an integer is evaluated as an expression
# (``RANDOM=PATH=1`` assigns ``PATH``).
_SHELL_NAME = re.compile(
    r"(?:BASH|COMP|HIST|LC_|LD_|DYLD_|PS\d|POSIX|READLINE)\w*|"
    + "|".join(
        "CHILD_MAX COLUMNS COPROC DIRSTACK EMACS ENV EPOCHREALTIME EPOCHSECONDS EUID EXECIGNORE "
        "FCEDIT FIGNORE FUNCNAME FUNCNEST GLOBIGNORE GROUPS HOME HOSTFILE HOSTNAME HOSTTYPE "
        "IGNOREEOF INPUTRC INSIDE_EMACS LANG LANGUAGE LINENO LINES MACHTYPE MAIL MAILCHECK "
        "MAILPATH MAPFILE OLDPWD OPTARG OPTERR OPTIND OSTYPE PATH PIPESTATUS PPID PROMPT_COMMAND "
        "PROMPT_DIRTRIM PWD RANDOM REPLY SECONDS SHELL SHELLOPTS SHLVL SRANDOM TIMEFORMAT TMOUT "
        "TMPDIR UID _ auto_resume histchars".split()
    )
)
# Every builtin of bash (``compgen -b``): a literal command word that is none
# of them runs a program or a function, and a program cannot change the shell.
_BUILTINS = frozenset(
    ". : [ alias bg bind break builtin caller cd command compgen complete compopt continue "
    "declare dirs disown echo enable eval exec exit export false fc fg getopts hash help history "
    "jobs kill let local logout mapfile popd printf pushd pwd read readarray readonly return set "
    "shift shopt source suspend test times trap true type typeset ulimit umask unalias unset "
    "wait".split()
)
# Those that assign no variable and end nothing, whatever their operands are.
_ANY_OPERANDS = frozenset(
    ": true false echo cd pwd kill umask break continue shift unalias".split()
)
# Those that do neither with exactly these operands (no login shell to log
# out of; a shell that ``suspend`` stops returns no status).
_OPERANDS = {"hash": ["-r"], "logout": [], "suspend": []}
# The operands of a ``set`` that leaves the shell running the text as the scan
# reads it, each ended by a NUL: not ``-n`` (nothing more runs), ``-k``, ``-t``
# or ``-o posix``.
_SET = re.compile(
    r"(?:(?:[-+][euxvCf]+|[-+][euxvCf]*o\0(?:errexit|nounset|xtrace|verbose|noclobber|noglob"
    r"|pipefail))\0)*(?:--\0.*)?",
    re.DOTALL,
)
# A conversion of a ``printf`` format that assigns: ``printf 'x%n' PATH`` sets ``PATH`` to 1.
_PRINTF_ASSIGNS = re.compile(r"%[^%A-Za-z]*[hjlLtz]*n")
# An operand of ``export`` by its name: no brace makes two of it (``PA{TH=1,}``).
_EXPORTED = re.compile(rf"({_NAME.pattern})(?:\+?=[^{{}}]*)?", re.DOTALL)
# An assignment word by its name: not an element's (a subscript is an expression).
_ASSIGNED = re.compile(rf"({_NAME.pattern})\+?=")
# The variable that a loop names.
_NAMED = re.compile(rf"\b(?:for|select)\s+({_NAME.pattern})")
# Syntax in which bash may assign a variable, or evaluate an expression that
# assigns one, where the scan reads no command: arithmetic (``((PATH=1))``,
# ``$[...]``, ``[[ t -eq 1 ]]``), an array (``a=([k]=v)``), the variable of a
# descriptor (``{PATH}>f``), a backtick, and every expansion but ``$name``,
# ``${name}``, ``${name-word}`` (with ``:``, ``+`` or ``?``), a positional
# or special parameter and ``$(command)``: a subscript, an offset (``${x:t}``:
# the value of ``t`` is evaluated too), a reference (``${!x}``), ``${x@P}``.
# And the directory as a ``cd`` left it (``$PWD``, ``$OLDPWD``): one that
# failed left it as it was, and the scan reads it as changed.  And
# ``${name=word}``: in a here-document body that a builtin reads, bash assigns
# before it expands the words after it, which the scan read first.
_UNPROVEN = re.compile(
    r"\(\(|\[\[|=\(|`|\}[<>]|\$\{?(?:OLD)?PWD\b"
    rf"|\$(?![A-Za-z_0-9@*#?$!-]|\((?!\()|\{{(?:{_NAME.pattern}|\d+)(?:\}}|:?[-+?])"
    r"|[\s/.,:=%^+~\])}|&;<>]|\Z)"
)

# Events that carry no progress information (UI signalling, streaming
# markers): the digest drops them without a trace.
_SKIPPED_EVENT_TYPES = frozenset({
    "system_prompt", "task_settings", "thinking_start", "thinking_end",
    "text_end", "task_done", "followup_suggestion", "llm_call",
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
        ``usage_info`` event (``""`` when none was recorded); a textless
        one (a late charge's totals) is formatted from its fields.
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
            spend = str(ev.get("text") or "") or _totals_spend(ev) or spend
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


def _totals_spend(ev: dict[str, Any]) -> str:
    """Return the spend line of a textless ``usage_info`` from its totals, or ``""``.

    A late charge (persisted after the run by
    :func:`~kiss.agents.sorcar.sorcar_agent._append_late_usage_events`)
    carries an empty ``text`` and the authoritative ``cost_usd``,
    ``total_tokens`` and ``total_steps``.
    """
    if ev.get("cost_usd") is None:
        return ""
    return (
        f"Steps: {int(ev.get('total_steps') or 0)}, "
        f"Total tokens: {int(ev.get('total_tokens') or 0):,}, "
        f"Budget: ${float(ev['cost_usd']):.4f}"
    )


def _event_ms(ev: dict[str, Any]) -> int:
    """Return an event's time in epoch milliseconds (``ts`` or the row timestamp), or 0.

    0 as well for a value that is no number, or no finite one (a foreign writer's).
    """
    try:
        return int(float(ev.get("ts") or 0)) or int(float(ev.get("_timestamp") or 0) * 1000)
    except (TypeError, ValueError, OverflowError):
        return 0


def _start_ms(task: dict[str, Any]) -> int:
    """Return the task's start time in epoch milliseconds (:func:`launch_ms` on its row)."""
    return launch_ms(task.get("start_ts"), task.get("timestamp"))


def _fmt_ts(ms: int) -> str:
    """Return *ms* (epoch milliseconds) as a UTC timestamp, or ``unknown``.

    ``unknown`` stands for 0 (no time recorded) and for a number no calendar date holds.
    """
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ms / 1000)) if ms else "unknown"
    except (OverflowError, OSError, ValueError):
        return "unknown"


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
    # The connection is taken before the lock (a first call on this
    # thread may wait out a migration or a sidecar recovery) and the
    # transcript is decoded after it: the lock is writer-preferring,
    # so anything slow under it stalls every write in the process.
    db = persistence._get_db()
    with persistence._rw_lock.read_lock():
        row = db.execute(
            persistence._HISTORY_SELECT + "WHERE id = ?", (task_id,),
        ).fetchone()
        if row is None:
            return None
        event_rows = persistence._select_event_rows(db, task_id)
    task = persistence._history_row_to_dict(row)
    task["events"] = persistence._decode_event_rows(event_rows)
    return task


def child_tasks(task_id: str) -> list[dict[str, Any]]:
    """Return the history rows of the sub-agent tasks *task_id* dispatched, oldest first."""
    from kiss.agents.sorcar import persistence

    db = persistence._get_db()
    with persistence._rw_lock.read_lock():
        rows = db.execute(
            persistence._HISTORY_SELECT
            + "WHERE parent_task_id = ? ORDER BY timestamp ASC, rowid ASC",
            (task_id,),
        ).fetchall()
    return [persistence._history_row_to_dict(r) for r in rows]


def _status(task: dict[str, Any], end: int) -> str:
    """Return ``running`` / ``finished`` and, for a task that ended at *end* ms, its result."""
    if not end:
        return "running"
    return f"finished ({clip(task.get('result'), 200)})"


def header_lines(task: dict[str, Any], spend: str, total: int) -> list[str]:
    """Return the digest header lines (task metadata, timing and spend)."""
    start_ms = _start_ms(task)
    end = end_ms(task.get("end_ts"))
    now_ms = int(time.time() * 1000)
    elapsed_s = ((end or now_ms) - start_ms) // 1000 if start_ms else 0
    lines = [
        f"Task id: {task.get('id')}",
        f"Task prompt: {clip(task.get('task'), _PROMPT_CHARS)}",
        f"Status: {_status(task, end)}",
        f"Chat id: {task.get('chat_id') or ''}",
        f"Model: {task.get('model') or ''}",
        f"Work dir: {task.get('work_dir') or ''}",
        f"Started: {_fmt_ts(start_ms)}",
        f"Elapsed: {_fmt_duration(elapsed_s)}",
    ]
    if end:
        lines.append(f"Ended: {_fmt_ts(end)}")
    lines.append(f"Spend: {spend or 'not recorded yet'}")
    lines.append(f"Transcript entries: {total}")
    if task.get("parent_task_id"):
        lines.append(f"Sub-agent of task: {task['parent_task_id']}")
    return lines


def _timed_header(task: dict[str, Any], entries: list[Entry], spend: str) -> list[str]:
    """Return the ``== Task ==`` section: :func:`header_lines`, the last event's age and now."""
    events = task["events"]
    last_ms = _event_ms(events[-1]) if events else 0
    now = time.time()
    lines = ["== Task ==", *header_lines(task, spend, len(entries))]
    if last_ms:
        ago = _fmt_duration(int(now - last_ms / 1000))
        lines.append(f"Last event: {_fmt_ts(last_ms)} ({ago} ago)")
    lines.append(
        f"Now: {_fmt_ts(int(now * 1000))} "
        f"(local: {time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(now))})"
    )
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
    """Return the two-line overview of a sub-agent task row.

    Its ``end_ts`` and ``cost`` are read with the guards the row's own
    header has (:func:`end_ms`): a column a foreign tool filled with
    text counts as unset instead of raising out of the parent's overview.
    """
    from kiss.agents.sorcar.persistence import _safe_float

    cost = _safe_float(child.get("cost"))
    status = "running" if not end_ms(child.get("end_ts")) else (
        f"finished: {_one_line(child.get('result'), _CHILD_RESULT_CHARS)}"
    )
    return (
        f"[{child.get('id')}] steps={child.get('steps') or 0}, cost=${cost:.2f}, "
        f"started {_fmt_ts(_start_ms(child))}, {status}\n"
        f"    task: {_one_line(child.get('task'), _CHILD_TASK_CHARS)}"
    )


def _summary_line(indexed: tuple[int, Entry]) -> str:
    """Return one overview line of a progress summary entry, given with its transcript index."""
    index, entry = indexed
    return f"[{index}] {entry.render(_OVERVIEW_SUMMARY_CHARS)}"


def _head_and_tail[T](
    items: list[T], head: int, tail: int, omitted: str, line: Callable[[T], str],
) -> list[str]:
    """Return *items* rendered by *line*: all, or the first *head* and last *tail* around a count.

    The count line reads ``... <n> <omitted> ...``.  Only the items
    shown are rendered: a fan-out of hundreds of children formats the
    few it lists.
    """
    gone = len(items) - head - tail
    if gone <= 0:
        return [line(item) for item in items]
    return [
        *(line(item) for item in items[:head]),
        f"... {gone} {omitted} ...",
        *(line(item) for item in items[-tail:]),
    ]


def _children_section(task: dict[str, Any], with_side_channel_count: bool) -> list[str]:
    """Return the ``== Sub-agent tasks ==`` section: one :func:`_child_line` per worker child.

    Side-channel children (earlier /ask and task-update runs) are not
    the task's own work; their answers already appear as ASK ANSWER
    entries, so at most their count is shown (*with_side_channel_count*).
    """
    children = child_tasks(str(task["id"]))
    workers = [c for c in children if not c.get("is_side_channel")]
    title = f"== Sub-agent tasks ({len(workers)}"
    if with_side_channel_count:
        title += (
            f"; plus {len(children) - len(workers)} side-channel /ask or task-update "
            "runs not listed"
        )
    lines = _head_and_tail(
        workers, _CHILDREN_HEAD, _CHILDREN_TAIL, "earlier sub-agent tasks omitted", _child_line,
    )
    return ["", title + ") ==", *(lines or ["(none)"])]


def overview(task_id: str) -> str:
    """Return the one-call orientation digest of a task.

    Sections: header (status, timing, spend, current time), sub-agent
    tasks, user messages sent after the initial prompt, the latest
    ``/ask`` answers, the task's own progress summaries, and the last
    :data:`_OVERVIEW_TAIL_ENTRIES` transcript entries.  The closing
    hint names the paging tools of the ``rsi7d`` agent, whose
    ``run_overview`` tool this is.

    Args:
        task_id: The ``task_history`` row id (32 hex characters).

    Returns:
        The digest text, or an error line when no task has that id.
    """
    loaded = _digest(task_id)
    if isinstance(loaded, str):
        return loaded
    task, entries, spend = loaded
    lines = _timed_header(task, entries, spend)
    lines += _children_section(task, with_side_channel_count=True)
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
    lines += _head_and_tail(
        summaries, _OVERVIEW_SUMMARIES_HEAD, _OVERVIEW_SUMMARIES_TAIL,
        "earlier summaries omitted (see run_transcript)", _summary_line,
    ) or ["(none)"]
    tail = indexed[-_OVERVIEW_TAIL_ENTRIES:]
    lines += ["", f"== Last {len(tail)} of {len(entries)} transcript entries =="]
    lines += [f"[{i}] {e.render()}" for i, e in tail] or ["(no transcript entries yet)"]
    lines += [
        "",
        "Indices in [brackets] are transcript entry numbers: run_transcript(task_id, "
        "start, count, contains) pages or filters all entries; run_entry(task_id, "
        "index) returns one entry in full.",
    ]
    return "\n".join(lines)


def _tail(text: str, limit: int, marker: str) -> str:
    """Return *text* if it fits *limit* chars, else its newest lines under a *marker* line.

    The result never exceeds *limit* (which must leave room for the
    marker line): the cut moves forward to the next line start, or,
    when the kept tail has no line start, falls in the middle of the
    last line.
    """
    if len(text) <= limit:
        return text
    tail = text[max(0, len(text) - limit + len(marker) + 1):]
    cut = tail.find("\n")
    if 0 <= cut < len(tail) - 1:
        tail = tail[cut + 1:]
    return marker + "\n" + tail


def _replay(
    text: str, ev: dict[str, Any], copy: HeredocCopy | None, applied: int | None = None,
) -> str:
    """Return the log *text* after the task's successful ``Write`` / ``Edit`` call *ev*.

    A ``Write`` puts its content, unless it stands for the here-document
    append *copy* (of the ``Bash`` command it was made for, ``None``: none):
    its content then opens with the text the file held, other tasks'
    included, and only the here-document's body is added.  An edit
    replaces its old text (each occurrence with ``replace_all``) as the
    tool did, including its CRLF conversion (:func:`_crlf_edit`, under
    the tool's condition: old text that is missing or of one line, never
    for an ``exact`` edit) and, for a single replacement that is not
    ``exact``, its unique whitespace-tolerant match (:func:`tolerant_matches`,
    the new text re-indented by :func:`reindent`); when the old text is not in
    *text* either way, the file held text the task did not write there,
    so only the edit's new text is added, at the end.  A malformed
    ``edits`` list changes nothing.  Of a batch that stopped part-way
    only the first *applied* edits are replayed (``None``: all of them).
    """
    args = _tool_call_args(ev)
    if ev.get("name") != "Edit":
        if copy is not None and copy.append:
            return text + copy.body
        return str(args.get("content") or "")
    pairs = _edit_pairs(str(args["edits"])) if args.get("edits") else [(
        str(args.get("old_string") or ""), str(args.get("new_string") or ""),
        str(args.get("replace_all")).lower() == "true",
    )]
    if isinstance(pairs, str):
        return text
    exact = str(args.get("exact")).lower() == "true"
    for old, new, every in pairs[:applied]:
        if old and not exact and (old not in text or "\n" not in old):
            old, new = _crlf_edit(old, new, "\r\n" in text)
        if old and old in text:
            text = text.replace(old, new, -1 if every else 1)
            continue
        matches = tolerant_matches(text, old) if old and not every and not exact else []
        if len(matches) == 1:
            match = matches[0]
            text = text[:match.start] + reindent(new, match.indent) + text[match.end:]
        elif new:
            text += new if not text or text.endswith("\n") else "\n" + new
    return text


def _copy_target(command: SimpleCommand) -> tuple[Word, bool] | None:
    """Return the file the simple *command* copies its here-document into, and whether it appends.

    ``None`` unless an exit status 0 of *command* means the file took the
    here-document's text: ``cat`` with no operand and its standard output
    redirected to one file, or ``tee [-a] FILE`` with its standard output
    left alone or discarded.  The scan knows the text
    (:attr:`SimpleCommand.heredoc_body`: nothing in it for the shell to
    expand) and the program (the word as written: no wrapper, expansion,
    alias or function), and the command redirects nothing else but its
    standard error into ``/dev/null``.
    """
    if command.heredoc_body is None or command.unknown or command.script is not None:
        return None
    outputs: list[tuple[str, Word]] = []
    heredocs = 0
    for descriptor, operator, operand in command.redirections:
        if descriptor in ("", "1") and operator in (">", ">>", ">|"):
            outputs.append((operator, operand))
        elif descriptor in ("", "0") and operator in ("<<", "<<-"):
            heredocs += 1
        elif descriptor != "2" or operator not in (">", ">>") or operand.text != "/dev/null":
            return None
    if heredocs != 1 or len(outputs) > 1:
        return None
    words = [word.text for word in command.words]
    if words == ["cat"]:
        return (outputs[0][1], outputs[0][0] == ">>") if outputs else None
    options = words[1:-1]
    if (
        len(words) < 2 or words[0] != "tee" or words[-1].startswith("-")
        or any(option not in ("-a", "--append") for option in options)
        or any(operand.text != "/dev/null" for _, operand in outputs)
    ):
        return None
    return command.words[-1], bool(options)


def _top_level(node: SimpleCommand | Group) -> SimpleCommand | Group:
    """Return the command or group of the top level that *node* is in (itself there)."""
    while node.group is not None:
        node = node.group
    return node


def _free(name: str) -> bool:
    """Whether assigning the variable *name* changes nothing the shell or the scan goes by."""
    return _SHELL_NAME.fullmatch(name) is None


def _vetted(name: str, operands: list[Word]) -> bool:
    """Whether the builtin *name* with these *operands* assigns only free variables, ends nothing.

    A builtin that is not named here does not: ``exit``, ``exec``, ``trap``,
    ``source``, ``shopt``, ``declare``, ``let``, ``enable``, ``fc``...
    """
    texts = [word.text for word in operands]
    if name in _ANY_OPERANDS:
        return True
    if name == "printf":  # no ``-v NAME`` or ``%n`` (both assign), nor a word that may be one
        return (
            bool(operands) and operands[0].literal and not texts[0].startswith("-")
            and not _PRINTF_ASSIGNS.search(texts[0])
        )
    if not all(word.literal for word in operands):  # bash may split another into more operands
        return False
    if name == "export":
        return all((found := _EXPORTED.fullmatch(text)) and _free(found[1]) for text in texts)
    if name in ("unset", "read"):
        flags = ("-f", "-v") if name == "unset" else ("-r",)
        names = texts[1:] if texts[:1] and texts[0] in flags else texts
        return all(_NAME.fullmatch(text) and (texts[0] == "-f" or _free(text)) for text in names)
    if name == "set":
        return _SET.fullmatch("".join(f"{text}\0" for text in texts)) is not None
    if name == "eval":  # the scan reads the text's commands; this, what is no command
        return not _unproven(" ".join(texts))
    if name in ("test", "["):  # ``-v 'a[PATH=1]'`` evaluates the subscript
        return not {"-v", "-R"} & set(texts)
    if name == "wait":  # no ``-p NAME``
        return not any(text.startswith("-") for text in texts)
    return texts == _OPERANDS.get(name)


def _harmless(command: SimpleCommand) -> bool:
    """Whether the shell runs the text after the simple *command* as the scan reads it.

    True of the commands known to leave the shell alone, false of every
    other: an assignment of free variables (:func:`_free`), a program (a
    path, or a literal word that is no builtin's: wrappers, functions and
    the headers of loops read as one, and a function's commands are
    commands of the scan themselves) and a builtin with operands that
    :func:`_vetted` knows.  Not a command only the shell knows, a function
    that bears a builtin's name, a word that may be an assignment the scan
    does not read as one (``a["k"]=v``) or an option of ``command``, which
    runs the builtin after it (``command -p exec true``).
    """
    if command.unknown:
        return False
    words = list(command.words)
    while words and words[0].assignment:
        assigned = _ASSIGNED.match(words.pop(0).text)
        if assigned is None or not _free(assigned[1]):
            return False
    while words and words[0].literal and words[0].text in ("command", "builtin"):
        del words[0]
    if not words:
        return True
    name = words[0].text
    if not words[0].literal:
        return words[0].path
    if name in _BUILTINS:
        return (command.script is None or name == "eval") and _vetted(name, words[1:])
    return "=" not in name and "[" not in name and not name.startswith("-")


def _runs_coproc(command: str, cwd: str | None) -> bool:
    """Whether a ``coproc`` in the text of *command* may be the keyword.

    The scan reads ``coproc cat > f`` as the ``cat`` it starts, whose
    status the shell does not wait for.  Each spelling of the word
    (:data:`_COPROC`) is replaced, one at a time, by a word the text
    does not hold, its joined lines included: where it was the keyword,
    that word is now the name of a command.  Text (a file name, a
    delimiter, a here-document's body, a comment) stays text, and
    nothing of a body becomes that command: the new word is nowhere else.
    A text with more spellings than :data:`_COPROC_PROBES` is not read:
    it may run one.
    """
    runs = re.findall("x+", command.replace("\\\n", ""))
    word = "x" * (max(map(len, runs), default=0) + 1)
    for probed, found in enumerate(_COPROC.finditer(command)):
        other = command[:found.start()] + word + command[found.end():]
        if probed == _COPROC_PROBES or any(
            simple.words and simple.words[0].text == word for simple in simple_commands(other, cwd)
        ):
            return True
    return False


def _unproven(text: str) -> bool:
    """Whether *text* holds syntax of :data:`_UNPROVEN` or names a variable that is not free.

    As written, or with the lines joined that the shell joins: quoting is
    not read, so a spelling in a quoted string or a comment counts too.
    """
    return any(
        _UNPROVEN.search(form) or not all(_free(n[1]) for n in _NAMED.finditer(form))
        for form in (text, text.replace("\\\n", ""))
    )


def _shape(simple: SimpleCommand) -> tuple[Any, ...]:
    """Return what the scan read of the command *simple*, its here-document's text last."""
    return (
        [(word.text, word.literal, word.assignment, word.path) for word in simple.words],
        [(fd, operator, operand.text) for fd, operator, operand in simple.redirections],
        simple.cwd, simple.unknown, simple.script is None, simple.group is None, simple.piped,
        simple.conditional, simple.after_and, simple.background, simple.negated,
        simple.heredoc_body,
    )


def _without_bodies(command: str, cwd: str | None, commands: list[SimpleCommand]) -> str:
    """Return *command* with one new word for each here-document body it holds as written.

    The scan hands on a body's text, not its place, and the same lines may
    stand elsewhere (in another body, among the commands).  So the place
    is the scan's to tell: whole lines that are the body, before the line
    that is its delimiter, are replaced by a word the text does not hold,
    and they were the body where the scan then reads the same *commands*,
    with that word for the body of the same one.  A body it reads in no
    such place (an unquoted delimiter joins lines), or after more than
    :data:`_BODY_PROBES` scans, stays in the text.
    """
    lines = command.split("\n")
    word = "x" * (max(map(len, re.findall("x+", command)), default=0) + 1)
    shapes = [_shape(simple) for simple in commands]
    start = scans = 0
    for at, simple in enumerate(commands):
        heredocs = [
            (operator, operand.text) for fd, operator, operand in simple.redirections
            if fd in ("", "0") and operator in ("<<", "<<-")
        ]
        if simple.heredoc_body is None or not heredocs:
            continue
        strip = "\t" if heredocs[-1][0] == "<<-" else ""
        body = [*simple.heredoc_body.split("\n")[:-1], heredocs[-1][1]]
        expected = [*shapes[:at], (*shapes[at][:-1], f"{word}{at}\n"), *shapes[at + 1:]]
        for first in range(start, len(lines) - len(body) + 1):
            end = first + len(body) - 1
            if lines[end].lstrip(strip) != body[-1] or scans == _BODY_PROBES or (
                [line.lstrip(strip) for line in lines[first:end]] != body[:-1]
            ):
                continue
            scans += 1
            marked = [*lines[:first], f"{word}{at}", *lines[end:]]
            scanned = simple_commands("\n".join(marked), cwd, cdpath=inherited_cdpath)
            if [_shape(s) for s in scanned] == expected:
                lines, shapes, start = marked, expected, first + 2
                break
    return "\n".join(lines)


def _made_copies(command: str, cwd: str | None) -> list[tuple[str, bool, str]]:
    """Return the progress-log copies that an exit status 0 of the ``Bash`` *command* proves made.

    Each is ``(file, appended, text)``, in the order they ran: a
    here-document copy (:func:`_copy_target`) into a
    :data:`_PROGRESS_LOG_NAMES` file whose failure the status of the
    whole command shows, as
    :func:`~kiss.agents.seas.coding.gate_readers.verifier_decides` reads
    a verifier's.  So the copy stands at the top level, not negated,
    piped, in the background or after ``||``; every command before it,
    a function's at each call too, is one known to leave the shell alone
    (:func:`_harmless`); the command runs no ``coproc``
    (:func:`_runs_coproc`) and, outside the here-document bodies the scan
    places in it (:func:`_without_bodies`), holds no syntax that assigns
    behind the scan (:func:`_unproven`); and
    everything after it follows ``&&``: ``copy && echo ok`` fails when the copy
    fails, while ``copy; echo ok`` exits 0 whatever the copy did and
    proves nothing.  The file is an absolute path as the scan reads it
    (literal, or through the command's own assignments), one in the
    home directory (``~/x``), or a relative one when the command up to
    the copy is one ``&&`` list of such plain commands: a ``cd`` that
    failed then kept the copy from running elsewhere.  A ``cd`` is read
    with the ``CDPATH`` a shell of this process inherits
    (:func:`~kiss.agents.sorcar.shell_scan.inherited_cdpath`), as the
    host tool's guards read it: with one, bash looks for ``cd dir``
    there first, and a relative file after it is only the shell's to name.

    Args:
        command: The command text of a ``Bash`` call.
        cwd: The absolute POSIX directory it started in; ``None`` when unknown.

    Raises:
        TooDeepError: The command nests deeper than the scan reads.
    """
    everything = simple_commands(command, cwd, derived=True, cdpath=inherited_cdpath)
    commands = [simple for simple in everything if not simple.derived]
    if _runs_coproc(command, cwd) or _unproven(_without_bodies(command, cwd, commands)):
        return []
    plain = [
        simple.group is None and simple.reader is None
        and not (simple.piped or simple.conditional or simple.background or simple.negated)
        for simple in commands
    ]
    # The status shows a copy from the last command on that does not follow ``&&`` ...
    start = max(
        (at for at, simple in enumerate(commands)
         if not _top_level(simple).after_and or _top_level(simple).background),
        default=0,
    )
    # ... and before the first one that may keep the rest from running as the scan reads it.
    stop = next((at for at, simple in enumerate(everything) if not _harmless(simple)), None)
    end = sum(not simple.derived for simple in everything[:stop])
    home = bash_path(os.path.expanduser("~"))
    made: list[tuple[str, bool, str]] = []
    for at in range(start, end):
        simple = commands[at]
        target = _copy_target(simple) if plain[at] else None
        if target is None:
            continue
        path = resolve(None, target[0]) or _home_path(target[0], home)
        if path is None and all(plain[:at]) and all(c.after_and for c in commands[1:at + 1]):
            path = resolve(simple.cwd, target[0])
        if path is not None and os.path.basename(path) in _PROGRESS_LOG_NAMES:
            made.append((os.path.normpath(host_path(path)), target[1], simple.heredoc_body or ""))
    return made


def _progress_log_text(task: dict[str, Any]) -> str:
    """Return the progress log as the task's own successful writes left it (``""``: none).

    Replays the task's ``Write`` / ``Edit`` calls of a
    :data:`_PROGRESS_LOG_NAMES` file in order (:func:`_replay`), each
    paired with the result that follows it: a call whose result is an
    error (raised, or returned as ``Error: ...`` text) or that the user
    stopped (``interrupted``, which says nothing of what it wrote) wrote
    nothing, except that a container's batch ``Edit`` whose error names
    the edits it applied before the failed one (:data:`_PARTIAL_BATCH`)
    wrote those.  A path is taken as the file tools take it
    (:func:`_absolutize`: ``~`` is the home directory, a relative path
    is in the task's work dir), so two spellings of a file are one log;
    the log is the file of the newest successful call.  A call that
    follows a ``Bash`` call before any result is the ``Write`` / ``Edit``
    that command was applied as (a here-document, a ``sed -i``).

    A ``Bash`` call no such call follows ran in the shell, as a
    here-document copy that creates its file does.  Its result says how
    the command ended in the tool's own words, apart from the output:
    unless it is a :data:`_NOT_PASSED` one, the command exited 0, and the
    copies that proves (:func:`_made_copies`) are replayed.  The output
    is not read: a ``tee`` prints its text, which may open like anything
    and reaches the result cut and with its line ends changed.  A call
    that polls its command (``until``: an unknown number of runs) or
    hands it to a background job (whose end is no part of the result)
    proves nothing.  A result that states no such end and opens like
    the ``Write`` / ``Edit`` the tool made of the command itself
    (:data:`_REWRITTEN`: no call follows the ``Bash`` call in a run
    without redirects) is the tool's words, with the status of the rest
    further down, or the output of a command that exited 0: it proves
    the one copy both readings do, the tool's own ``Write`` (the
    here-document copy as :func:`heredoc_copy` reads it) when an exit
    status 0 proves that copy too.
    """
    work_dir = str(task.get("work_dir") or "")
    cwd = bash_path(os.path.abspath(work_dir)) if work_dir else None
    logs: dict[str, str] = {}
    newest = ""
    pending: tuple[str, dict[str, Any], HeredocCopy | None] | None = None
    # Of the ``Bash`` call awaiting its result: the copy the tool makes as a
    # ``Write`` when the file exists, and the copies an exit status 0 proves.
    copy: HeredocCopy | None = None
    made: list[tuple[str, bool, str]] = []
    for ev in task.get("events") or []:
        if ev.get("type") == "tool_call":
            path = str(ev.get("path") or "")
            pending = None
            if (
                ev.get("name") in ("Write", "Edit")
                and os.path.basename(path) in _PROGRESS_LOG_NAMES
            ):
                pending = (os.path.normpath(_absolutize(path, work_dir)), ev, copy)
            command = str(ev.get("command") or "") if ev.get("name") == "Bash" else ""
            copy, made = None, []
            if "<<" in command:  # only a command with a here-document is scanned
                try:
                    # The tool makes a ``Write`` only of a copy that no inherited ``CDPATH``
                    # moves (it refuses the others): its record is read without this process's.
                    copy = heredoc_copy(command, cwd, cdpath=str)
                    made = _made_copies(command, cwd)
                except TooDeepError:  # nested past what the scan reads: the tools refuse it
                    pass
            args = _tool_call_args(ev)
            if (
                args.get("until") or str(args.get("background")).lower() == "true"
                or args.get("on_timeout") == "background"
            ):
                made = []
        elif ev.get("type") == "tool_result":
            content = str(ev.get("content") or "")
            partial = _PARTIAL_BATCH.match(content)
            if ev.get("is_error") or ev.get("interrupted"):
                made = []
            elif pending is not None:
                if partial or not content.startswith(_UNMADE_RESULT):
                    newest = pending[0]
                    logs[newest] = _replay(
                        logs.get(newest, ""), pending[1], pending[2],
                        int(partial[1]) if partial else None,
                    )
            elif _NOT_PASSED.search(content):
                made = []
            elif rewritten := _REWRITTEN.match(content):
                own = (
                    (os.path.normpath(host_path(copy.path)), copy.append, copy.body)
                    if rewritten[1] == "Write" and copy is not None and copy.path is not None
                    else None
                )
                made = [own] if own is not None and own in made else []
            for file, appended, text in made:
                newest = file
                logs[file] = (logs.get(file, "") if appended else "") + text
            pending = copy = None
            made = []
    return logs.get(newest, "")


def progress_log_tail(task: dict[str, Any], limit: int = _PROGRESS_LOG_CHARS) -> str:
    """Return the newest *limit* chars of the progress log *task* wrote, or ``""``.

    Work dirs are shared by every task launched in them, so a log file
    may hold other tasks' text (appended after this task wrote, or there
    before it appended), and neither the file's name, its modification
    time nor a piece of text this task wrote says which bytes are this
    task's.  The log is therefore rebuilt from the task's own persisted
    tool calls (:func:`_progress_log_text`), not read from disk: it holds
    only text those calls wrote, however the file changed since
    (rewritten, merged, deleted).  :func:`load_task` flushes the task's
    queued events, so a running task's newest log write is included.

    Args:
        task: A :func:`load_task` row (its ``events`` and ``work_dir``).
        limit: Character budget of the returned tail.
    """
    text = _progress_log_text(task)
    if not text.strip():
        return ""
    return _tail(text, limit, "[older progress-log entries elided]")


def context(task_id: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Return the one-shot context of a task for answering a question about it.

    Sections: header (status, timing, spend, current time), sub-agent
    tasks, the tail of the progress log the task's own writes keep (see
    :func:`progress_log_tail`), and the transcript entries, oldest first.
    When the whole text would exceed *max_chars*, whole transcript
    entries are dropped OLDEST first and
    the elision is marked, so the newest steps (what questions usually
    target) always survive; the newest entry is kept even when it alone
    (with the marker) overflows the budget (its rendering is clipped,
    so the overflow is bounded by the per-kind clip limits).

    Args:
        task_id: The ``task_history`` row id (32 hex characters).
        max_chars: Character budget of the whole context.

    Returns:
        The context text, or an error line when no task has that id.
    """
    loaded = _digest(task_id)
    if isinstance(loaded, str):
        return loaded
    task, entries, spend = loaded
    lines = _timed_header(task, entries, spend)
    lines += _children_section(task, with_side_channel_count=False)
    log = progress_log_tail(task)
    if log:
        lines += ["", "== Progress log written by the task (newest entries) ==", log.rstrip()]
    lines += ["", f"== Transcript ({len(entries)} entries, oldest first) =="]
    head = "\n".join(lines) + "\n"
    if not entries:
        return head + "(no transcript entries yet)"
    blocks = [f"[{i}] {e.render()}" for i, e in enumerate(entries)]
    budget = max_chars - len(head)
    first = len(blocks) - 1
    used = len(blocks[first])
    while first > 0:
        older = first - 1
        # An elision marker line is still needed unless *older* is entry 0.
        reserve = len(_elision(older)) + 1 if older else 0
        if used + 1 + len(blocks[older]) + reserve > budget:
            break
        first = older
        used += 1 + len(blocks[older])
    if first:
        blocks[:first] = [_elision(first)]
    return head + "\n".join(blocks)


def _elision(count: int) -> str:
    """Return the marker line that stands in for *count* elided transcript entries."""
    return f"[... {count} older entries elided ...]"


def entry_detail(task_id: str, index: int) -> str:
    """Return one transcript entry of a task in full (clipped at 20,000 characters).

    Args:
        task_id: The ``task_history`` row id (32 hex characters).
        index: The entry number shown in brackets by the other digests.

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
    return f"[{index}] {entries[index].render(20_000)}"
