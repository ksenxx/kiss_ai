# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Hermes-style scheduled automations (cron) with delivery to any channel.

Mirrors the Hermes agent's cron design in the simplest possible form:

- Jobs live in a single JSON file (``$KISS_HOME/cron/jobs.json``) — no
  database.  Atomic writes and an ``flock`` guard make concurrent
  ticks and tool calls safe.
- The natural-language part is done by the LLM: the :func:`cron_job`
  tool accepts only four normalized schedule forms (interval, 5-field
  cron expression, one-shot duration, one-shot ISO timestamp) and the
  agent translates phrases like "every weekday at 9am" into them.
- The Sorcar agent does not carry the :func:`cron_job` tool itself:
  this module is an *SEA* (``kiss.server.sorcar.run``'s
  ``sea_path`` contract), and a scheduling request is dispatched to
  it with the ``run_agent`` tool as ``run_agent(task, agent="cron")`` — the
  dispatched session gets the :func:`cron_job` tool from
  :func:`tools` and runs in ``$KISS_HOME/cron/work`` without a
  worktree (a ``ChannelSea`` with the ``work_dir`` of :func:`settings`).
- The kiss-web daemon runs the scheduler automatically in a
  background thread (:func:`start_scheduler_thread`): every ~60
  seconds a tick finds due jobs, reschedules them *before* running
  (so the same occurrence never double-fires), and launches each one
  CONCURRENTLY in its own thread and its own scratch directory
  (``$KISS_HOME/cron/runs/<job_id>-<random>``, removed when the run
  ends) so simultaneous jobs never share a working directory.  The
  scheduler thread does not wait for the jobs: a tick that overlaps
  runs from a previous tick is not skipped — it simply leaves the
  jobs that are still running alone (they stay due and are picked up
  by the first tick after they finish) and starts the rest.  Like
  cron, missed occurrences are not made up: a repeating job found
  more than :data:`MISSED_RUN_GRACE_SECONDS` overdue (the daemon was
  down, the machine asleep) is rescheduled from now without running.
  ``kiss-cron --tick`` and ``kiss-cron --daemon`` remain available
  for running the scheduler outside the daemon; in that mode command
  jobs work standalone while prompt jobs still need a reachable
  kiss-web daemon (they are submitted through its local endpoint).
- A prompt job runs as the bundled SEA
  :mod:`kiss.agents.sorcar.cron_prompt_sea` through the same
  ``run_agent`` tool a chat task uses for any ``.py`` SEA
  (:func:`kiss.agents.sorcar.agent_dispatch.make_run_agent_tool`): the
  job's prompt is the task, and its model, budget, ``work_dir`` /
  ``use_worktree`` / ``auto_commit`` are the tool's arguments — a job
  that must work inside a specific project names that directory,
  otherwise the run's scratch directory with worktree and auto-commit
  off (:func:`_run_prompt_job`).
- An always-on channel gateway (README §19) is a scheduled COMMAND
  job — the channel CLI's poll tick, e.g. ``kiss-telegram
  --channel=-100123 --pairing`` — never a prompt job: a tick that
  finds no messages costs no tokens.  :func:`gateway_command` builds
  that command deterministically from the channel and chat names.
- Delivery targets are looked up dynamically: any module named
  ``kiss.agents.third_party_agents.<channel>.<channel>_sea`` with a
  ``_make_backend()`` factory can receive results (``telegram:123``,
  ``slack:eng``, ``ntfy``, ...).  This module works without those
  optional channel modules — an unknown channel just yields a
  delivery-error note.  Every run is also appended to a local log
  under ``$KISS_HOME/cron/output/``.  A ``[SILENT]`` summary (or empty
  command output) suppresses delivery, exactly like Hermes.
- ``command`` jobs (Hermes "no_agent" mode) run a shell command with
  no LLM involved; non-empty stdout is delivered verbatim.

Usage::

    kiss-cron --create "morning brief" --schedule "0 9 * * *" \\
        --prompt "Summarize today's HN front page" --deliver telegram:123
    kiss-web   # the daemon ticks the scheduler automatically
"""

import argparse
import contextlib
import importlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from kiss.agents.seas.base.base_sea import ChannelSea
from kiss.agents.sorcar.useful_tools import _popen_kwargs
from kiss.core.config import kiss_home
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.processes import SIGKILL, kill_process_group, popen_process_group
from kiss.core.utils import atomic_write_text, read_bytes_waiting_for_writer

logger = logging.getLogger(__name__)

COMMAND_TIMEOUT_SECONDS = 600.0
PROMPT_TIMEOUT_SECONDS = 3600.0

_running_lock = threading.Lock()
_running: dict[str, threading.Thread] = {}
"""Job id -> thread of every job run in progress in this process.

Ticks register the thread they launch, ``run_now`` the calling thread.
Process-local (the JSON store holds no lease): a tick treats a job whose
thread is still alive as not due and ``run_now`` refuses it, so a run
that outlasts the job's interval is never overlapped in the same
process.  ``kiss-cron --tick`` in another process cannot see these runs
and may still overlap them.
"""

_daemon_endpoint_file: str | None = None
"""Endpoint file of the kiss-web daemon hosting this process's scheduler.

Set by :func:`start_scheduler_thread` so prompt jobs — scheduled ticks
and ``cron_job("run_now", ...)`` tool calls executed inside the daemon
alike — are submitted back to the same daemon even when it publishes a
non-default endpoint file.  Read by
:func:`kiss.agents.sorcar.agent_dispatch._daemon_endpoint_file`, which
imports the CANONICAL ``kiss.agents.sorcar.cron_agent`` module: a
dispatched cron session gets its ``cron_job`` tool from a fresh
synthetic copy of this module whose own global is never set, and its
``run_now`` still has to find the recorded endpoint.  Cleared by
:func:`stop_scheduler_thread` when that daemon shuts down, so a later
daemon in the same process (an in-process restart, a test) is not sent
to the dead one's endpoint.
"""
CRON_SCAN_DAYS = 4 * 366 + 1  # covers the largest gap between leap days
DEFAULT_TICK_INTERVAL_SECONDS = 60.0
MISSED_RUN_GRACE_SECONDS = 10 * 60.0
"""How late a repeating job may still run; a later occurrence is missed.

Like cron, the scheduler never catches up on occurrences that passed
while no scheduler was running (the kiss-web daemon was stopped,
restarted by an install, or the machine was asleep): a tick that finds
a repeating job overdue by more than this many seconds reschedules it
from now without running it, so a daemon coming back after a day never
fires every daily job at once.  The window is wide enough for a daemon
restart and for a job that overlapped its own next occurrence by a few
minutes.  One-shot jobs are exempt: they fire once, late, rather than
never.
"""
MAX_STORED_SUMMARY_CHARS = 4000

_UNIT_SECONDS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_DURATION_RE = re.compile(r"^(\d+)\s*(s|m|h|d)$")
_INTERVAL_RE = re.compile(r"^every\s+(\d+)\s*(s|m|h|d)$")

_CRON_BOUNDS = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))

SCHEDULE_TZ = ZoneInfo("America/Los_Angeles")
"""Time zone of every schedule time: Pacific time (PDT, or PST in winter).

Cron expressions and offset-less ISO timestamps are evaluated in it, and
job listings and run logs show their times in it, whatever the time
zone of the machine running the scheduler.
"""


def format_schedule_time(timestamp: float) -> str:
    """Return *timestamp* (epoch seconds) as schedule time, e.g. ``2026-09-27 05:00:00 PDT``.

    Args:
        timestamp: The time in epoch seconds.

    Returns:
        The date and time in :data:`SCHEDULE_TZ` with the zone abbreviation.
    """
    return datetime.fromtimestamp(timestamp, SCHEDULE_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


def _cron_dir() -> Path:
    """Return the cron state directory (``$KISS_HOME/cron``)."""
    return kiss_home() / "cron"


def _jobs_path() -> Path:
    """Return the path of the JSON job store."""
    return _cron_dir() / "jobs.json"


def _output_dir() -> Path:
    """Return the directory holding per-job local output logs."""
    return _cron_dir() / "output"


def _runs_dir() -> Path:
    """Return the parent of the per-run scratch directories.

    Every job run gets its own fresh subdirectory here (created by
    :func:`_execute_job`, removed when the run ends) so concurrent
    jobs never share a working directory.
    """
    return _cron_dir() / "runs"


_SILENCE_TOKENS = frozenset({"[SILENT]", "NO_REPLY"})


def _is_silent(summary: str) -> bool:
    """Return whether *summary* is a Hermes-style silence token.

    A summary that is exactly ``[SILENT]`` or ``NO_REPLY`` (optionally
    wrapped in HTML tags by the daemon's HTML conversion) suppresses
    delivery.

    Args:
        summary: The job's deliverable summary text.

    Returns:
        ``True`` when delivery should be suppressed.
    """
    # ``[^<>]`` (not ``[^>]``) keeps this linear: a summary with many
    # ``<`` and no ``>`` otherwise rescans to the end from each ``<``.
    return re.sub(r"<[^<>]+>", "", summary).strip() in _SILENCE_TOKENS


def _jobs_lock(blocking: bool) -> Any:
    """Return a context manager holding the inter-process lock of the job store.

    The same lock serializes the scheduler's tick (non-blocking: an
    overlapping tick skips) and the tool's read-modify-write
    (blocking: the tool waits for a running tick to finish), so a job
    edit can never be overwritten by a stale in-memory save.  The lock
    is :func:`kiss.core.file_lock.exclusive_file_lock`, which owns
    the cross-platform (fcntl/msvcrt/no-op) mechanics.

    Args:
        blocking: Whether to wait for the lock (tool path) or give up
            immediately when it is held (tick path).

    Returns:
        A context manager yielding ``True`` while the lock is held, or
        ``False`` when *blocking* is ``False`` and another process
        holds it.
    """
    return exclusive_file_lock(_jobs_path().with_suffix(".lock"), blocking=blocking)


def load_jobs() -> list[dict[str, Any]]:
    """Load all cron jobs from the JSON store.

    A gateway command job stored with a channel delivery target (by a
    version without :func:`_gateway_delivery`) is returned with
    ``deliver`` ``none``, so listing, duplicate detection and delivery
    all see the normalised job; the next save persists it.

    Returns:
        The list of job dicts; an empty list when the store does not
        exist or is unreadable.
    """
    try:
        data = json.loads(read_bytes_waiting_for_writer(_jobs_path()).decode("utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    jobs = [job for job in data if isinstance(job, dict) and job.get("id")]
    for job in jobs:
        job["deliver"] = _gateway_delivery(
            str(job.get("command", "")), str(job.get("deliver", "local"))
        )
    return jobs


def save_jobs(jobs: list[dict[str, Any]]) -> None:
    """Atomically persist the full job list to the JSON store.

    Staged in a sibling temp file and renamed over the store, so readers
    never observe a partially written file.

    Args:
        jobs: The complete list of job dicts to write.
    """
    # Jobs carry prompts and commands: private on every save, as the
    # ``mkstemp``-staged file it replaced always was.
    atomic_write_text(_jobs_path(), json.dumps(jobs, indent=2), mode=0o600)


def _parse_cron_field(field: str, low: int, high: int) -> set[int] | None:
    """Parse one cron field into the set of matching integer values.

    Supports ``*``, ``*/step``, ``a``, ``a-b``, ``a-b/step``, and
    comma-separated lists of those.

    Args:
        field: The raw field text (e.g. ``"*/15"`` or ``"1,3-5"``).
        low: Smallest legal value for this field.
        high: Largest legal value for this field.

    Returns:
        The set of matching values, or ``None`` when the field is
        invalid.
    """
    values: set[int] = set()
    for item in field.split(","):
        item = item.strip()
        step = 1
        if "/" in item:
            item, _, step_text = item.partition("/")
            if not step_text.isdigit() or int(step_text) < 1:
                return None
            step = int(step_text)
        if item == "*":
            start, end = low, high
        elif "-" in item:
            a, _, b = item.partition("-")
            if not (a.isdigit() and b.isdigit()):
                return None
            start, end = int(a), int(b)
        elif item.isdigit():
            start = end = int(item)
        else:
            return None
        if start < low or end > high or start > end:
            return None
        values.update(range(start, end + 1, step))
    return values or None


def _parse_cron_expr(expr: str) -> tuple[list[set[int]], bool, bool] | None:
    """Parse a 5-field cron expression into per-field value sets.

    Args:
        expr: A standard 5-field cron expression
            (minute hour day-of-month month day-of-week).

    Returns:
        ``(fields, dom_star, dow_star)`` where *fields* is a list of
        five value sets and the flags record whether day-of-month /
        day-of-week were written as ``*`` (Vixie cron applies its
        either-matches rule based on the literal ``*``, not on the
        covered range).  ``None`` when *expr* is not a valid 5-field
        cron expression.  Day-of-week ``7`` is folded into ``0``
        (both mean Sunday).
    """
    parts = expr.split()
    if len(parts) != 5:
        return None
    fields: list[set[int]] = []
    for part, (low, high) in zip(parts, _CRON_BOUNDS, strict=True):
        values = _parse_cron_field(part, low, high)
        if values is None:
            return None
        fields.append(values)
    if 7 in fields[4]:
        fields[4].discard(7)
        fields[4].add(0)
    return fields, parts[2].startswith("*"), parts[4].startswith("*")


def _cron_date_matches(
    fields: list[set[int]], dom_star: bool, dow_star: bool, dt: datetime
) -> bool:
    """Return whether *dt*'s date matches the cron date fields.

    Uses the standard Vixie cron rule: when both day-of-month and
    day-of-week are restricted (neither written as ``*``), the date
    matches if EITHER matches; otherwise both must match.

    Args:
        fields: Fields from :func:`_parse_cron_expr`.
        dom_star: Whether the day-of-month field was written as ``*``.
        dow_star: Whether the day-of-week field was written as ``*``.
        dt: The schedule-time datetime whose date to test.

    Returns:
        ``True`` when the month and day rules all match.
    """
    _, _, dom, month, dow = fields
    if dt.month not in month:
        return False
    dom_ok = dt.day in dom
    dow_ok = (dt.isoweekday() % 7) in dow  # cron: 0 = Sunday
    if not dom_star and not dow_star:
        return dom_ok or dow_ok
    return dom_ok and dow_ok


def is_one_shot(schedule: str) -> bool:
    """Return whether *schedule* fires once (duration or ISO timestamp).

    Args:
        schedule: A normalized schedule string.

    Returns:
        ``True`` for one-shot durations (``"30m"``) and ISO timestamps;
        ``False`` for intervals (``"every 30m"``) and cron expressions.
    """
    text = schedule.strip().lower()
    return not _INTERVAL_RE.match(text) and _parse_cron_expr(schedule.strip()) is None


def compute_next_run(schedule: str, now: float) -> float | None:
    """Compute the next run time (epoch seconds) for a schedule.

    Supported forms:

    - Interval: ``"every 30m"``, ``"every 2h"`` (units ``s m h d``).
    - Cron: standard 5-field expression, e.g. ``"0 9 * * 1-5"``,
      evaluated in Pacific time (:data:`SCHEDULE_TZ`).
    - One-shot duration: ``"30m"``, ``"1d"`` (relative to *now*).
    - One-shot ISO 8601 timestamp: ``"2026-01-15T14:00:00"`` (Pacific
      time unless an offset is given).

    Args:
        schedule: The schedule string.
        now: Current time in epoch seconds.

    Returns:
        The next run time in epoch seconds, or ``None`` when a
        one-shot timestamp is already in the past or no cron match
        exists within the ~4-year scan horizon (which covers the
        largest gap between leap days).

    Raises:
        ValueError: When *schedule* matches none of the supported forms.

    Note:
        Cron times use Pacific wall-clock time: around a DST transition a run
        can shift by up to an hour (a time skipped by spring-forward
        fires an hour late; the repeated fall-back hour fires once).
    """
    text = schedule.strip()
    m = _INTERVAL_RE.match(text.lower())
    if m:
        return now + int(m.group(1)) * _UNIT_SECONDS[m.group(2)]
    m = _DURATION_RE.match(text.lower())
    if m:
        return now + int(m.group(1)) * _UNIT_SECONDS[m.group(2)]
    parsed = _parse_cron_expr(text)
    if parsed is not None:
        fields, dom_star, dow_star = parsed
        minute_set, hour_set = fields[0], fields[1]
        dt = datetime.fromtimestamp(now, SCHEDULE_TZ).replace(second=0, microsecond=0)
        dt += timedelta(minutes=1)
        for _ in range(CRON_SCAN_DAYS):
            if _cron_date_matches(fields, dom_star, dow_star, dt):
                day = dt.date()
                while dt.date() == day:
                    # Wall-clock arithmetic drops the fall-back fold, so a
                    # minute of the repeated hour can map to the past.
                    if dt.minute in minute_set and dt.hour in hour_set and dt.timestamp() > now:
                        return dt.timestamp()
                    dt += timedelta(minutes=1)
            else:
                dt = datetime(dt.year, dt.month, dt.day, tzinfo=SCHEDULE_TZ) + timedelta(days=1)
        return None
    try:
        when = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(
            f"Unsupported schedule {schedule!r}: use 'every N<s|m|h|d>', a "
            "5-field cron expression, a one-shot duration like '30m', or an "
            "ISO timestamp like '2026-01-15T14:00'"
        ) from None
    if when.tzinfo is None:
        when = when.replace(tzinfo=SCHEDULE_TZ)
    ts = when.timestamp()
    return ts if ts > now else None


def _deliver_to_channel(channel: str, chat: str, text: str) -> str:
    """Send *text* to one channel agent's backend.

    Imports ``kiss.agents.third_party_agents.<channel>.<channel>_sea``, builds
    its backend with the module's ``_make_backend()`` factory (which
    loads the credentials persisted under ``$KISS_HOME``), and calls
    ``send_message``.

    Args:
        channel: Channel agent short name (e.g. ``"telegram"``,
            ``"slack"``, ``"ntfy"``).
        chat: Chat/channel identifier on that platform; may be empty
            for single-destination channels.
        text: The message text to send.

    Returns:
        A human-readable delivery note (``"sent to ..."`` or an error).
    """
    target = f"{channel}:{chat}" if chat else channel
    try:
        module = importlib.import_module(
            f"kiss.agents.third_party_agents.{channel}.{channel}_sea"
        )
    except ImportError:
        return f"error: unknown channel {channel!r}"
    factory = getattr(module, "_make_backend", None)
    if not callable(factory):
        return f"error: channel {channel!r} does not support delivery"
    backend: Any = None
    try:
        backend = factory()
        connect = getattr(backend, "connect", None)
        if callable(connect) and connect() is False:
            return f"error: {target}: backend connect failed"
        channel_id = backend.find_channel(chat) or chat
        backend.send_message(channel_id, text)
    except SystemExit:
        return f"error: {target}: channel not authenticated"
    except Exception as e:
        logger.error("Cron delivery to %s failed: %s", target, e, exc_info=True)
        return f"error: {target}: {e}"
    finally:
        if backend is not None:
            with contextlib.suppress(Exception):
                backend.disconnect()
    return f"sent to {target}"


def _deliver(job: dict[str, Any], text: str) -> list[str]:
    """Deliver a job result to all of the job's targets.

    The result is always appended to the job's local log
    (``$KISS_HOME/cron/output/<job_id>.md``); ``local`` and ``none``
    targets add nothing further, and every other target is a
    ``<channel>[:<chat>]`` handled by :func:`_deliver_to_channel`.

    Args:
        job: The job dict (uses ``id``, ``name``, and ``deliver``).
        text: The result text to deliver.

    Returns:
        One note per non-local target describing success or failure.
    """
    _output_dir().mkdir(parents=True, exist_ok=True)
    log_path = _output_dir() / f"{job['id']}.md"
    stamp = format_schedule_time(time.time())
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(f"## {stamp} — {job.get('name', '')}\n\n{text}\n\n")
    notes: list[str] = []
    for target in str(job.get("deliver", "local")).split(","):
        target = target.strip()
        if not target or target in ("local", "none"):
            continue
        channel, _, chat = target.partition(":")
        notes.append(_deliver_to_channel(channel.strip(), chat.strip(), text))
    return notes


PROMPT_PREAMBLE = (
    "You are running as an unattended scheduled automation (cron job). "
    "Nobody can answer questions; never ask the user anything. Do not "
    "create, modify, or remove scheduled jobs during this run. Your "
    "final summary is delivered verbatim to the job's delivery targets; "
    "reply with exactly [SILENT] if there is nothing worth reporting — "
    "never a bare acknowledgement such as 'OK' or 'tick-OK' — and never "
    "post a heartbeat or status message to a messaging channel "
    "yourself.\n\n"
)
"""Hermes-style preamble prepended to every prompt job's prompt."""

UNATTENDED_MARKER = "Nobody can answer questions; never ask the user anything."
"""Sentence shared by both unattended preambles."""

UNATTENDED_CHILD_PREAMBLE = (
    "You are a sub-task of an unattended scheduled automation (cron job). "
    + UNATTENDED_MARKER
    + " Never wait for user approval: when an action is blocked, report "
    "the blocker in your final summary and finish."
)
"""Paragraph added to every sub-task (``run_agent`` / ``run_parallel``)
spawned from an unattended run, so the child inherits the no-questions rule
instead of blocking on ``ask_user_question`` until its timeout.  Appended
through ``append_to_prompt`` (:func:`unattended_child_suffix`), which the
daemon adds after an agent script's ``prompt()`` override has replaced
the prompt body."""

CHAT_TASK_HEADING = "# Task"
"""Heading ``ChatSorcarAgent.build_chat_prompt`` puts in front of the current
task, after the chat history (``# Task`` / ``# Task (work on it now)``)."""


def _current_task_text(agent: Any) -> str:
    """Return the text of the task *agent* is working on now.

    ``task_description`` on a
    :class:`~kiss.agents.sorcar.relentless_agent.RelentlessAgent` (its
    ``prompt_template`` is the per-session template), ``prompt_template``
    on a plain :class:`~kiss.core.kiss_agent.KISSAgent`.  A chat agent's
    text starts with earlier tasks and results; only the part after the
    last :data:`CHAT_TASK_HEADING` is the current task, so an old result
    that quotes a preamble cannot mark a later task as unattended.
    """
    task = getattr(agent, "task_description", "") or getattr(agent, "prompt_template", "")
    return str(task or "").rsplit(CHAT_TASK_HEADING, 1)[-1].strip()


def is_unattended(agent: Any) -> bool:
    """True when *agent* runs an unattended (cron) task or a sub-task of one.

    The current task text must start with :data:`PROMPT_PREAMBLE` (a cron
    prompt job) or end with :data:`UNATTENDED_CHILD_PREAMBLE` (a sub-task);
    a prompt that merely quotes the sentence elsewhere is not unattended.

    Args:
        agent: A running agent (see :func:`_current_task_text`).
    """
    text = _current_task_text(agent)
    return text.startswith(PROMPT_PREAMBLE.strip()) or text.endswith(UNATTENDED_CHILD_PREAMBLE)


def unattended_child_suffix(append_to_prompt: str) -> str:
    """Return *append_to_prompt* ending with :data:`UNATTENDED_CHILD_PREAMBLE` (once).

    Args:
        append_to_prompt: The sub-task caller's prompt suffix (may be empty).
    """
    if append_to_prompt.rstrip().endswith(UNATTENDED_CHILD_PREAMBLE):
        return append_to_prompt
    return append_to_prompt + "\n\n" + UNATTENDED_CHILD_PREAMBLE


PROMPT_SEA_PATH = Path(__file__).with_name("cron_prompt_sea.py")
"""The SEA every prompt job runs as (:mod:`kiss.agents.sorcar.cron_prompt_sea`)."""


def _job_work_dir(job: dict[str, Any], scratch_dir: Path) -> Path:
    """Return the directory a run of *job* works in.

    Args:
        job: The job dict (uses the optional ``work_dir`` field).
        scratch_dir: The run's private scratch directory.

    Returns:
        The job's ``work_dir`` when set, otherwise *scratch_dir*.
    """
    return Path(str(job.get("work_dir") or "") or scratch_dir)


def _job_timeout(job: dict[str, Any], default: float) -> float:
    """Return the per-run timeout of *job* in seconds.

    Args:
        job: The job dict (uses the optional ``timeout`` field; ``0`` or
            missing means *default*).
        default: :data:`COMMAND_TIMEOUT_SECONDS` or
            :data:`PROMPT_TIMEOUT_SECONDS`.

    Returns:
        The timeout in seconds.
    """
    return float(job.get("timeout") or 0.0) or default


def _run_prompt_job(
    job: dict[str, Any], scratch_dir: Path, run_dir: Path, timeout: float,
) -> tuple[str, str | None]:
    """Run an LLM cron job in a fresh daemon session.

    Launches :data:`PROMPT_SEA_PATH` with the ``run_agent`` tool
    (:func:`kiss.agents.sorcar.agent_dispatch.make_run_agent_tool`)
    exactly as a chat task launches any SEA: the task text is
    the Hermes-style preamble followed by the job's prompt, and the
    job's ``model`` and ``max_budget`` (``""`` / ``None`` mean "daemon
    default"), *run_dir* and the job's ``use_worktree`` /
    ``auto_commit`` flags (off unless the job asked for them, since a
    scratch directory is not a git repository) travel as the tool's
    arguments and ``options``.
    Mirrors Hermes: every run gets a brand-new session (no history),
    with a preamble marking the run as unattended and forbidding
    further scheduling; a ``[SILENT]`` (or empty) summary suppresses
    delivery.

    The run has no parent task (the tool is built without a parent
    agent), so the daemon treats it as a top-level task: no reviewer
    sub-tree marking, no shared budget.

    Args:
        job: The job dict (uses ``id``, ``name``, ``prompt``,
            ``model_name``, ``max_budget``, ``use_worktree``,
            ``auto_commit``).
        scratch_dir: The run's private scratch directory (created by
            :func:`_execute_job`), where the SEA lives.
        run_dir: The task's working directory: the job's ``work_dir``
            (a prompt that must run inside a specific project) or
            *scratch_dir*; see :func:`_job_work_dir`.
        timeout: The run's bound in seconds (see :func:`_job_timeout`).

    Returns:
        ``(status, summary)`` where status is ``"ok"``, ``"error"`` or
        ``"silent"`` (summary ``None`` — nothing to deliver).  Failures
        to reach the daemon, SEA errors and a confirmed
        timeout come back as ``"error"`` with the ``run_agent`` error
        text.  ``run_agent``'s ``timeout`` bounds the call only and hands
        back the still-running sub-task as a job; an unattended run has
        nobody to collect it later, so this function kills it.

    Raises:
        TimeoutError: When the run timed out but the daemon never
            confirmed the stop — the task may still be running, so the
            caller must keep *scratch_dir*; the message is the complete
            error to record.
    """
    # Lazy import: agent_dispatch imports this module lazily as well
    # (``_daemon_endpoint_file``), and importing it at module load would
    # pull the whole dispatch layer into ``kiss-cron --list``.
    from kiss.agents.sorcar.agent_dispatch import (
        agent_jobs_of,
        forget_agent_job,
        kill_agent_job,
        make_run_agent_tool,
        notice_job_id,
        unconfirmed_stop_error,
    )
    from kiss.agents.sorcar.sea_settings import script_name

    run_agent = make_run_agent_tool(str(scratch_dir))
    raw_budget = job.get("max_budget")
    reply = run_agent(
        agent=str(PROMPT_SEA_PATH),
        task=PROMPT_PREAMBLE + str(job.get("prompt", "")),
        model=str(job.get("model_name") or ""),
        max_budget=str(float(raw_budget)) if raw_budget else "",
        timeout=str(timeout),
        options=json.dumps({
            "work_dir": str(run_dir),
            "use_worktree": bool(job.get("use_worktree")),
            "auto_commit": bool(job.get("auto_commit")),
        }),
    )
    # ``run_agent``'s timeout bounds the call only; an unattended run
    # has nobody to collect the detached sub-task, so it is stopped.
    detached = agent_jobs_of(None).get(notice_job_id(reply))
    if detached is not None:
        try:
            reply = kill_agent_job(detached)
        finally:
            # Also when the kill itself is interrupted: nothing else
            # ever collects an ownerless job from the registry.
            forget_agent_job(detached)
        bound = f"the scheduled task did not finish within {timeout:g}s"
        unconfirmed = unconfirmed_stop_error(script_name(str(PROMPT_SEA_PATH)))
        if not detached.finished or detached.outcome == unconfirmed:
            raise TimeoutError(
                f"prompt job timed out after {timeout:g}s; the stop was not "
                f"confirmed, so the task may still be running in {run_dir} (its "
                f"scratch directory {scratch_dir} is kept): Error: {bound}; a stop "
                f"was requested but the daemon never confirmed it, so the task MAY "
                f"STILL BE RUNNING (and spending) on the daemon."
            )
        if isinstance(detached.outcome, str):  # stopped (a finished one falls through)
            return "error", f"Error: {bound}; {reply.removeprefix('Error: ')}"
    if reply.startswith("Error:"):
        return "error", reply
    result = yaml.safe_load(reply)
    if not isinstance(result, dict):
        return "error", f"unexpected run_agent reply: {reply}"
    success = bool(result.get("success"))
    summary = str(result.get("summary") or "") or ("" if success else "Task failed")
    if not summary or _is_silent(summary):
        return "silent", None
    return ("ok" if success else "error"), summary


def _proc_descendants(root_pid: int) -> set[int]:
    """Best-effort set of live descendant pids of *root_pid*.

    Walks ``/proc`` (Linux) building the parent→children map from each
    process's ``stat`` ppid field, then collects the transitive
    children of *root_pid*.  Returns an empty set on platforms without
    ``/proc`` (macOS/BSD) and for pids that raced away mid-scan.

    Args:
        root_pid: The pid whose descendants to collect.

    Returns:
        The pids of every currently visible descendant of *root_pid*.
    """
    children: dict[int, list[int]] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return set()
    for name in entries:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as fp:
                data = fp.read()
        except OSError:
            continue
        # The ppid is the second field after the ")" that closes the
        # (possibly space/paren-containing) command name.
        fields = data.rpartition(b")")[2].split()
        if len(fields) >= 2:
            children.setdefault(int(fields[1]), []).append(int(name))
    descendants: set[int] = set()
    stack = [root_pid]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in descendants:
                descendants.add(child)
                stack.append(child)
    return descendants


def _kill_command_tree(proc: subprocess.Popen) -> None:
    """Best-effort kill of a timed-out command's whole process tree.

    POSIX: kills the command's own process group (every ordinary
    foreground/background descendant), then — because a descendant that
    called ``setsid`` or daemonized lives in a NEW group the ``killpg``
    misses — walks ``/proc`` for surviving descendants by parent chain
    and kills each one's group and pid.  Descendants are snapshotted
    BEFORE the group kill: once the shell dies its orphans re-parent to
    init and the chain is lost.  The scan-and-kill pass is repeated a
    bounded number of times to catch processes spawned mid-kill.
    Residual limitation (documented, not guaranteed): a process that
    double-detaches faster than the bounded rescan, or any descendant
    on a POSIX system without ``/proc``, can still escape.

    Windows: :func:`kill_process_group` runs ``taskkill /T /F`` on the
    shell's pid (kills the process tree Windows tracks), then
    ``proc.kill()`` as a fallback; a descendant that detached from the
    tree is not covered.

    Args:
        proc: The timed-out command's shell process (session leader on
            POSIX, direct child on Windows).
    """
    if os.name == "nt":  # pragma: no cover — Windows CI is not available here
        with contextlib.suppress(OSError):  # group already gone
            kill_process_group(proc.pid, SIGKILL)
        proc.kill()
        return
    own_pgid = os.getpgrp()
    for _ in range(3):
        survivors = _proc_descendants(proc.pid)
        with contextlib.suppress(OSError):  # group already gone
            kill_process_group(proc.pid, SIGKILL)
        escaped = False
        for pid in survivors:
            if pid == os.getpid():  # pragma: no cover — defensive
                continue
            try:
                pgid = os.getpgid(pid)
            except OSError:
                continue  # already dead
            if pgid == proc.pid:
                continue  # covered by the group kill above
            escaped = True
            # Kill the escapee's own group first (a setsid child leads
            # a new group holding its subtree), then the pid itself.
            if pgid != own_pgid:  # pragma: no branch — never our own group
                with contextlib.suppress(OSError):
                    os.killpg(pgid, 9)
            with contextlib.suppress(OSError):
                os.kill(pid, 9)
        if not escaped:
            break


def _run_command_job(
    job: dict[str, Any], run_dir: Path, timeout: float,
) -> tuple[str, str | None]:
    """Run a no-LLM command job (Hermes "no_agent" mode).

    The command runs under the same shell as the agent's ``Bash`` tool
    (``sh`` on POSIX, Git bash on Windows; see
    :func:`~kiss.agents.sorcar.useful_tools._popen_kwargs`) in its own
    process group, and a timeout kills the WHOLE process tree — not
    just the shell; see :func:`_kill_command_tree` for the exact
    guarantees and residual limitations per platform.

    Args:
        job: The job dict (uses ``command``).
        run_dir: The command's working directory (see
            :func:`_job_work_dir`).
        timeout: Maximum runtime in seconds before the command's
            process tree is killed (see :func:`_job_timeout`).

    Returns:
        ``(status, text)``: ``("silent", None)`` when the command
        succeeds with empty output, ``("ok", stdout)`` on success, and
        ``("error", output)`` on non-zero exit or timeout.
    """
    # The command sees the state directory this daemon uses (the brand's
    # default unless KISS_HOME already overrides it), so a script it runs
    # with another interpreter reads and writes the same files.
    proc = popen_process_group(
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(run_dir),
        env={**os.environ, "KISS_HOME": str(kiss_home())},
        **_popen_kwargs(str(job["command"])),
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_command_tree(proc)
        # The tree is dead, so the pipes close and this drain returns
        # promptly; the bounded retry guards a straggler that survived
        # the best-effort tree kill holding the pipes open.
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            # Abandon the pipes to the straggler and reap the (killed)
            # shell, so the Popen does not linger as a zombie.
            for pipe in (proc.stdout, proc.stderr):
                if pipe is not None:
                    pipe.close()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)
        return "error", f"command timed out after {timeout:g}s"
    output = stdout.strip()
    if proc.returncode != 0:
        detail = (output + "\n" + stderr.strip()).strip()
        return "error", f"command exited {proc.returncode}: {detail}"
    if not output:
        return "silent", None
    return "ok", output


def _execute_job(job: dict[str, Any]) -> None:
    """Execute one job, deliver its result, and record the outcome.

    The run gets a fresh private scratch directory under
    :func:`_runs_dir` — its working directory for a command job, the
    daemon session's ``work_dir`` for a prompt job (unless the job
    names its own ``work_dir``, which then takes that role) — so jobs
    running
    concurrently never share one; the directory is removed when the
    run ends, whatever the outcome, so stale run directories never
    accumulate.  The one exception is a timed-out prompt job whose
    stop the daemon never confirmed: the task may still be running
    from the directory, so it is kept and its path reported.

    Never raises: failures are recorded in the job's ``last_status`` /
    ``last_summary`` fields.  Errors are still delivered (so the user
    learns the automation broke), silent results are not.

    Prompt jobs are submitted to the kiss-web daemon whose endpoint
    file :func:`start_scheduler_thread` recorded (standard resolution
    — ``KISS_SORCAR_LOCAL``, then ``$KISS_HOME/sorcar-local.json`` —
    when none was recorded, e.g. under ``kiss-cron --tick``).

    Args:
        job: The job dict to execute.
    """
    _runs_dir().mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{job['id']}-", dir=_runs_dir()))
    run_dir = _job_work_dir(job, work_dir)
    is_command = bool(str(job.get("command", "")).strip())
    keep_work_dir = False
    try:
        # Inside the try: a malformed stored timeout is recorded as the
        # run's error like any other failure.
        timeout = _job_timeout(
            job, COMMAND_TIMEOUT_SECONDS if is_command else PROMPT_TIMEOUT_SECONDS,
        )
        if is_command:
            status, text = _run_command_job(job, run_dir, timeout)
        else:
            status, text = _run_prompt_job(job, work_dir, run_dir, timeout)
    except TimeoutError as e:
        # Only the unconfirmed-stop timeout reaches here (_run_prompt_job
        # reports the confirmed-stop timeout itself): the task may still
        # be running, so its scratch directory is kept, as the error says.
        keep_work_dir = True
        status, text = "error", str(e)
    except Exception as e:
        logger.error("Cron job %s failed: %s", job["id"], e, exc_info=True)
        status, text = "error", f"{type(e).__name__}: {e}"
    finally:
        if not keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
    notes: list[str] = []
    if text is not None:
        try:
            notes = _deliver(job, text)
        except Exception as e:
            logger.error("Cron delivery for %s failed: %s", job["id"], e, exc_info=True)
            notes = [f"error: delivery failed: {e}"]
    # Retire an until_delivered poll only when the news actually reached
    # every requested target: a delivery error keeps the job alive so
    # the next tick tries again.
    delivered = (
        text is not None and status == "ok"
        and not any(note.startswith("error") for note in notes)
    )
    with _jobs_lock(blocking=True):
        jobs = load_jobs()
        for stored in jobs:
            if stored["id"] == job["id"]:
                stored["last_status"] = status
                stored["last_summary"] = (text or "")[:MAX_STORED_SUMMARY_CHARS]
                stored["last_delivery"] = notes
                if delivered and stored.get("until_delivered"):
                    # A "notify me when ..." poll has done its job.
                    stored["enabled"] = False
                    stored["next_run_at"] = None
        save_jobs(jobs)


def running_job_ids() -> set[str]:
    """Return the ids of the jobs this process is currently running.

    Also prunes finished threads from the registry.

    Returns:
        The ids of every job whose run thread is still alive.
    """
    with _running_lock:
        for job_id in [jid for jid, thread in _running.items() if not thread.is_alive()]:
            del _running[job_id]
        return set(_running)


def tick(now: float | None = None, wait: bool = True) -> int:
    """Run one scheduler pass: launch every due job.

    Takes a non-blocking ``flock`` on the job store (a tick that finds
    another tick or a tool edit mid-selection exits immediately —
    selection takes milliseconds, so the next tick catches up),
    selects due jobs, advances their ``next_run_at`` (disabling
    one-shots) BEFORE running so the same occurrence can never
    double-fire, starts every due job in its own thread and its own
    scratch directory (see :func:`_execute_job`) so simultaneous jobs
    run concurrently, and releases the lock.  A malformed job (bad
    ``next_run_at`` or schedule) is disabled and skipped instead of
    aborting the tick.

    A repeating job whose occurrence passed more than
    :data:`MISSED_RUN_GRACE_SECONDS` ago is missed, not late: the tick
    advances its ``next_run_at`` from now without running it and logs
    a warning, so a scheduler starting after downtime (a daemon
    restart by ``install.sh``, a machine asleep overnight) never fires
    every missed daily job at once.  A one-shot still runs, once,
    however late.

    A job still running from a previous tick of this process (see
    :data:`_running`) is not due: the tick leaves it untouched (its
    ``next_run_at`` stays in the past) and starts only the other due
    jobs, so it runs again on the first tick after it finishes — within
    the grace window — instead of overlapping itself.  The registry is
    process-local, so ``kiss-cron --tick`` in another process may still
    overlap a run; a one-shot claimed by a process that crashes
    mid-run is not retried.

    Args:
        now: Current time in epoch seconds; ``None`` uses the clock.
        wait: Whether to block until every launched job has finished
            (the ``kiss-cron --tick`` CLI).  The scheduler loop passes
            ``False`` so a long job never delays the next tick.

    Returns:
        The number of jobs launched (``0`` when another tick holds the
        lock).
    """
    now = time.time() if now is None else now
    with _jobs_lock(blocking=False) as held:
        if not held:
            return 0
        jobs = load_jobs()
        running = running_job_ids()
        due = []
        changed = False
        for job in jobs:
            if not job.get("enabled") or job["id"] in running:
                continue
            try:
                overdue = now - float(job["next_run_at"])
                if overdue < 0:
                    continue
                next_run = (
                    None
                    if job.get("one_shot")
                    else compute_next_run(str(job["schedule"]), now)
                )
                if overdue > MISSED_RUN_GRACE_SECONDS and not job.get("one_shot"):
                    # timedelta(int(inf)) raises OverflowError: handled below.
                    logger.warning(
                        "kiss-cron: skipping the run of %r (%s) due %s ago, past "
                        "the %s grace window; next run at %s",
                        job.get("name"), job["id"],
                        timedelta(seconds=int(overdue)),
                        timedelta(seconds=MISSED_RUN_GRACE_SECONDS),
                        format_schedule_time(next_run) if next_run else "never",
                    )
                    job["next_run_at"] = next_run
                    changed = True
                    continue
            except (TypeError, ValueError, KeyError, OverflowError) as e:
                logger.error("Disabling malformed cron job %s: %s", job.get("id"), e)
                job["enabled"] = False
                job["last_status"] = "error"
                job["last_summary"] = f"disabled: malformed job: {e}"
                changed = True
                continue
            job["last_run_at"] = now
            job["next_run_at"] = next_run
            if job.get("one_shot"):
                job["enabled"] = False
            due.append(job)
            changed = True
        if changed:
            save_jobs(jobs)
        # Registered and started while holding _running_lock (and the
        # store lock): a thread only counts as alive once started, so
        # no concurrent running_job_ids() call can observe — and prune
        # — a registered-but-idle entry.  A job that finishes instantly
        # merely waits for the store lock before recording its outcome.
        threads = [
            threading.Thread(
                target=_execute_job, args=(job,),
                name=f"kiss-cron-job-{job['id']}", daemon=True,
            )
            for job in due
        ]
        with _running_lock:
            for job, thread in zip(due, threads, strict=True):
                _running[job["id"]] = thread
                thread.start()
    if wait:
        for thread in threads:
            thread.join()
    return len(due)


def run_scheduler(
    stop_event: threading.Event,
    interval: float = DEFAULT_TICK_INTERVAL_SECONDS,
) -> None:
    """Run the scheduler loop until *stop_event* is set.

    Ticks immediately, then every *interval* seconds.  Ticks do not
    wait for the jobs they launch (``tick(wait=False)``): a long run
    never delays the next tick, which starts the other due jobs and
    leaves the still-running one alone.  A failing tick is logged and
    never stops the loop.

    Args:
        stop_event: Setting this event stops the loop (the wait
            between ticks returns early; jobs already launched run to
            completion in their own daemon threads).
        interval: Seconds between scheduler passes.
    """
    while not stop_event.is_set():
        try:
            ran = tick(wait=False)
            if ran:
                logger.info("kiss-cron: launched %d job(s)", ran)
        except Exception as e:
            logger.error("Scheduler tick failed: %s", e, exc_info=True)
        stop_event.wait(interval)


class SchedulerStop(threading.Event):
    """The stop event of a scheduler thread, which also remembers that thread.

    Returned by :func:`start_scheduler_thread`; :func:`stop_scheduler_thread`
    joins :attr:`thread` after setting the event.
    """

    thread: threading.Thread


def start_scheduler_thread(
    interval: float = DEFAULT_TICK_INTERVAL_SECONDS,
    endpoint_file: str | None = None,
) -> SchedulerStop:
    """Start the scheduler loop in a daemon thread.

    Called by the kiss-web daemon on startup so scheduled automations
    fire without any external cron process.  Prompt jobs are submitted
    back to the daemon through *endpoint_file*, recorded in
    :data:`_daemon_endpoint_file`.

    Args:
        interval: Seconds between scheduler passes.
        endpoint_file: The hosting daemon's endpoint file; recorded as
            the module's :data:`_daemon_endpoint_file` so scheduled
            prompt jobs and ``run_now`` tool calls in this process
            target the same daemon.  ``None`` keeps the standard
            endpoint resolution.

    Returns:
        The stop event, to pass to :func:`stop_scheduler_thread`.
    """
    global _daemon_endpoint_file
    if endpoint_file:
        _daemon_endpoint_file = endpoint_file
    stop_event = SchedulerStop()
    stop_event.thread = threading.Thread(
        target=run_scheduler,
        args=(stop_event, interval),
        name="kiss-cron-scheduler",
        daemon=True,
    )
    stop_event.thread.start()
    return stop_event


def stop_scheduler_thread(stop_event: SchedulerStop) -> None:
    """Stop a scheduler started by :func:`start_scheduler_thread`.

    Sets *stop_event*, waits for the loop to exit — a tick in flight
    finishes launching its due jobs first, with the endpoint still
    recorded — and then forgets the hosting daemon's endpoint file
    (:data:`_daemon_endpoint_file`): once that daemon is down,
    dispatched sub-tasks and ``run_now`` in this process must fall back
    to the standard endpoint resolution instead of a dead endpoint.
    Jobs the scheduler launched keep running in their own threads.

    Args:
        stop_event: The event returned by :func:`start_scheduler_thread`.
    """
    global _daemon_endpoint_file
    stop_event.set()
    stop_event.thread.join(timeout=30)
    _daemon_endpoint_file = None


def _dump(data: Any) -> str:
    """Return *data* as the YAML text the :func:`cron_job` tool answers with."""
    return str(yaml.safe_dump(data, sort_keys=False))


def _job_view(job: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, human-readable view of a job for listings.

    Args:
        job: The stored job dict.

    Returns:
        A dict with the job's key fields and its times in Pacific time.
    """
    view = {
        key: job.get(key)
        for key in (
            "id", "name", "schedule", "deliver", "enabled", "one_shot",
            "until_delivered", "prompt", "command", "last_status",
            "last_delivery",
        )
        if job.get(key) not in (None, "", [])
    }
    for key in ("work_dir", "use_worktree", "auto_commit", "timeout", "model_name"):
        if job.get(key):
            view[key] = job[key]
    for key in ("next_run_at", "last_run_at"):
        if job.get(key):
            view[key] = format_schedule_time(float(job[key]))
    return view


def _enable(job: dict[str, Any]) -> str:
    """Re-enable a paused *job* in place (its next run recomputed when it has none).

    Args:
        job: The stored job record, mutated; the caller saves the store.

    Returns:
        ``""`` on success, else the error text: a one-shot job that has
        already run cannot be resumed.
    """
    if job.get("one_shot") and job.get("next_run_at") is None:
        return f"one-shot job {job['id']!r} already ran; create a new job instead"
    job["enabled"] = True
    if job.get("next_run_at") is None:
        job["next_run_at"] = compute_next_run(str(job["schedule"]), time.time())
    return ""


def _find_duplicate(
    jobs: list[dict[str, Any]], candidate: dict[str, Any]
) -> dict[str, Any] | None:
    """Return the stored job that would do the same work as ``candidate``.

    Two jobs are duplicates when they run the same ``prompt`` or
    ``command`` in the same ``work_dir`` on the same ``schedule`` with
    the same ``deliver`` targets (the same prompt scheduled for two
    different projects is two jobs).  The name, model and budget are
    ignored: the LLM picks
    those freely, and the user asked for the same thing to be scheduled
    only once.  Jobs that have finished for good (a one-shot that
    already ran or an ``until_delivered`` poll that fired, i.e.
    ``next_run_at`` is ``None``) are not duplicates: recreating them
    is how a user schedules the same thing again.  Paused jobs still
    count, so the caller can suggest resuming instead of adding a copy.

    Args:
        jobs: The stored jobs (already loaded under the store lock).
        candidate: The job about to be created, with stripped fields.

    Returns:
        The matching stored job, or ``None`` when there is none.
    """
    for job in jobs:
        if job.get("next_run_at") is None:
            continue
        if all(
            str(job.get(key) or "").strip() == candidate[key]
            for key in ("prompt", "command", "work_dir", "schedule", "deliver")
        ):
            return job
    return None


def cron_job(
    action: str,
    job_id: str = "",
    name: str = "",
    prompt: str = "",
    command: str = "",
    schedule: str = "",
    deliver: str = "local",
    model_name: str = "",
    max_budget: str = "",
    until_delivered: bool = False,
    work_dir: str = "",
    use_worktree: bool = False,
    auto_commit: bool = False,
    timeout: str = "",
) -> str:
    """Manage scheduled automations (cron jobs) stored in a local JSON file.

    Translate the user's natural-language request (e.g. "every weekday
    at 9am", "in 30 minutes", "every 2 hours") into one of the four
    supported schedule forms yourself before calling this tool.

    Actions:

    - ``create``: register a job.  Requires ``name``, ``schedule``,
      and exactly one of ``prompt`` (an LLM task run unattended in a
      fresh session) or ``command`` (a shell command run without any
      LLM; its stdout is delivered verbatim).  Refused with an
      ``error`` (and the ``existing`` job) when a job with the same
      prompt/command, schedule and delivery targets is already
      scheduled or paused — do not retry under another name; tell the
      user it exists, or ``remove``/``resume`` the existing job.
    - ``ensure``: ``create`` unless a job with the same ``name`` (or,
      failing that, a ``create``-duplicate of the spec) already exists:
      an enabled one is left as it is (``exists``), a paused one is
      resumed (``resumed``); only when there is none is the job created.
      Atomic under the store lock, for a job that code re-registers on
      every trigger (autorouter's weekly ``/rsi7d``).
    - ``list``: list all jobs with their next/last run times.
    - ``remove`` / ``pause`` / ``resume``: manage the job named by
      ``job_id``.
    - ``run_now``: execute the job named by ``job_id`` immediately and
      deliver its result (the regular schedule is unaffected); refused
      while a run of that job is in progress in this process.

    Prefer ``command`` jobs for polls and checks ("is X released yet?",
    "did the build finish?", "is the site up?"): a shell command such as
    ``if curl -s URL | grep -q 'X'; then echo 'X is out'; fi`` costs
    nothing per run, prints only when there is news and exits 0 (so a
    quiet poll is silent, not an error), while a ``prompt`` job starts a
    full LLM session every time.  A poll that should stop once it has
    fired ("tell me WHEN X happens") gets ``until_delivered=True``: the
    job disables itself after its first successfully delivered
    non-silent result (also when triggered by ``run_now``) instead of
    repeating the same news on every tick.  An always-on gateway for a
    messaging channel is likewise a ``command`` job: build the command
    with the ``gateway_command`` tool first, then schedule it here
    (typically ``"every 2m"``) — never as a ``prompt`` job: ``create``
    refuses a prompt that describes a gateway tick, and a gateway
    command is always stored with ``deliver`` ``none`` so its ticks
    never post status into the chat.

    Jobs due at the same time run concurrently, each in its own scratch
    directory that is removed when the run ends; a job whose previous
    run is still in progress is not started again until it finishes.
    A job that must work inside a specific project ("run the tests in
    /home/me/proj every night and fix them") names that directory as
    ``work_dir``; a prompt job in a git repository may additionally ask
    for ``use_worktree`` / ``auto_commit`` so the run edits an isolated
    worktree and merges its commits back, exactly like a chat task with
    those toggles on.  Long runs need a larger ``timeout``: a run is
    stopped once it exceeds it (default 3600 s for a prompt job,
    600 s for a command).

    Schedule forms (all times are Pacific time, PDT/PST, never UTC or the
    machine's time zone):

    - Repeating interval: ``"every 30m"``, ``"every 2h"``
      (units ``s``, ``m``, ``h``, ``d``).
    - Repeating cron: standard 5-field expression, e.g. ``"0 9 * * 1-5"``
      for 9:00 PDT on weekdays.
    - One-shot delay: ``"30m"``, ``"1d"`` (runs once, that far from now).
    - One-shot timestamp: ISO 8601, e.g. ``"2026-01-15T14:00"`` (Pacific
      time unless it carries an offset).

    Delivery (``deliver``): comma-separated targets.  ``local`` (default)
    only appends to ``$KISS_HOME/cron/output/<job_id>.md``; any other
    target is ``<channel>[:<chat>]`` using an authenticated channel
    agent, e.g. ``telegram:123456``, ``slack:general``, ``ntfy``,
    ``discord:987``, ``email:user@example.com``.  A job whose result is
    exactly ``[SILENT]`` (or a command with empty output) delivers
    nothing.

    The kiss-web daemon runs the scheduler automatically; jobs fire
    while the daemon is up.  ``kiss-cron --daemon`` / ``--tick`` also
    run the scheduler standalone, where command jobs work on their own
    but prompt jobs still need a reachable kiss-web daemon.

    Args:
        action: One of ``create``, ``ensure``, ``list``, ``remove``,
            ``pause``, ``resume``, ``run_now``.
        job_id: Job identifier (required for remove/pause/resume/run_now).
        name: Short human-readable job name (create; the identity an
            ``ensure`` looks up).
        prompt: The LLM task to run on schedule (create).
        command: Shell command to run instead of an LLM task (create).
        schedule: Schedule string in one of the four forms above (create).
        deliver: Comma-separated delivery targets (create; default
            ``local``).
        model_name: LLM model override for prompt jobs (create; empty
            uses the default model).
        max_budget: Per-run USD budget override for prompt jobs, as a
            string like ``"2.5"`` (create; empty uses the default).
        until_delivered: Disable the job after its first non-silent
            delivery (create; for "notify me when ..." polls).
        work_dir: Existing directory the run works in — the command's
            working directory or the prompt session's work directory
            (create; empty uses a private scratch directory).
        use_worktree: Run the prompt job in a git worktree of
            ``work_dir`` (create; prompt jobs with ``work_dir`` only).
        auto_commit: Auto-commit (and, with ``use_worktree``, merge)
            the prompt job's changes (create; prompt jobs with
            ``work_dir`` only).
        timeout: Per-run timeout in seconds, as a string like
            ``"21600"`` (create; empty uses the default).

    Returns:
        A YAML string describing the result (created job, job list,
        confirmation, or an ``error`` key explaining what went wrong).
    """
    if action in ("create", "ensure"):
        if not name or not schedule:
            return _dump({"error": f"{action} requires name and schedule"})
        if bool(prompt.strip()) == bool(command.strip()):
            return _dump({"error": f"{action} requires exactly one of prompt or command"})
        if _describes_gateway(prompt):
            return _dump({
                "error": "a messaging gateway is never a prompt job: an LLM "
                "session on every tick costs money and posts status chatter "
                "such as 'tick-OK' into the chat.  Call gateway_command(channel, "
                "chat) and schedule its result as a command job with "
                "deliver='none'."
            })
        note = ""
        if _gateway_delivery(command, deliver) != deliver:
            note = (
                "a gateway tick never posts into a chat: delivery set to none "
                "(ticks that served messages or failed are logged in "
                f"{_output_dir()}/<job_id>.md)"
            )
            deliver = "none"
        try:
            next_run = compute_next_run(schedule, time.time())
        except ValueError as e:
            return _dump({"error": str(e)})
        if next_run is None:
            return _dump({"error": f"schedule {schedule!r} never fires (in the past?)"})
        try:
            budget = float(max_budget) if max_budget.strip() else 0.0
        except ValueError:
            return _dump({"error": f"max_budget {max_budget!r} is not a number"})
        try:
            timeout_seconds = float(timeout) if timeout.strip() else 0.0
        except ValueError:
            return _dump({"error": f"timeout {timeout!r} is not a number"})
        if not (timeout_seconds >= 0 and math.isfinite(timeout_seconds)):
            return _dump({"error": f"timeout {timeout!r} must be a positive number"})
        run_dir = ""
        if work_dir.strip():
            run_dir = str(Path(work_dir.strip()).expanduser().resolve())
            if not Path(run_dir).is_dir():
                return _dump({"error": f"work_dir {work_dir!r} is not a directory"})
        if (use_worktree or auto_commit) and not (run_dir and prompt.strip()):
            return _dump({
                "error": "use_worktree and auto_commit need a prompt job with work_dir"
            })
        job = {
            "id": uuid.uuid4().hex[:8],
            "name": name,
            "prompt": prompt.strip(),
            "command": command.strip(),
            "schedule": schedule.strip(),
            "deliver": deliver.strip() or "local",
            "model_name": model_name.strip(),
            "max_budget": budget,
            "work_dir": run_dir,
            "use_worktree": bool(use_worktree),
            "auto_commit": bool(auto_commit),
            "timeout": timeout_seconds,
            "enabled": True,
            "one_shot": is_one_shot(schedule),
            "until_delivered": bool(until_delivered),
            "created_at": time.time(),
            "next_run_at": next_run,
            "last_run_at": None,
            "last_status": "",
            "last_summary": "",
            "last_delivery": [],
        }
        with _jobs_lock(blocking=True):
            jobs = load_jobs()
            duplicate = _find_duplicate(jobs, job)
            if action == "ensure":
                named = [j for j in jobs if j.get("name") == name]
                existing = named[0] if named else duplicate
                if existing is not None:
                    if existing.get("enabled"):
                        return _dump({"exists": _job_view(existing)})
                    error = _enable(existing)
                    if error:
                        return _dump({"error": error})
                    save_jobs(jobs)
                    return _dump({"resumed": _job_view(existing)})
            if duplicate is not None:
                state = "paused" if not duplicate.get("enabled") else "scheduled"
                hint = (
                    f"resume it with cron_job('resume', job_id={duplicate['id']!r})"
                    if state == "paused"
                    else "remove it first if you want to replace it"
                )
                return _dump({
                    "error": f"duplicate: job {duplicate['id']!r} "
                    f"({duplicate.get('name')!r}) is already {state} with the "
                    f"same {'command' if job['command'] else 'prompt'}, schedule "
                    f"and delivery targets; {hint}",
                    "existing": _job_view(duplicate),
                })
            jobs.append(job)
            save_jobs(jobs)
        result: dict[str, Any] = {"created": _job_view(job)}
        if note:
            result["note"] = note
        return _dump(result)

    if action == "list":
        return _dump({"jobs": [_job_view(job) for job in load_jobs()]})

    if action in ("remove", "pause", "resume"):
        if not job_id:
            return _dump({"error": f"{action} requires job_id"})
        with _jobs_lock(blocking=True):
            jobs = load_jobs()
            match = [job for job in jobs if job["id"] == job_id]
            if not match:
                return _dump({"error": f"no job with id {job_id!r}"})
            if action == "remove":
                jobs = [job for job in jobs if job["id"] != job_id]
            elif action == "pause":
                match[0]["enabled"] = False
            else:
                error = _enable(match[0])
                if error:
                    return _dump({"error": error})
            save_jobs(jobs)
        return _dump({action: job_id})

    if action == "run_now":
        match = [job for job in load_jobs() if job["id"] == job_id]
        if not match:
            return _dump({"error": f"no job with id {job_id!r}"})
        # Registered like a tick's run thread in the CANONICAL module's
        # registry (a dispatched cron session runs a synthetic copy of
        # this module, see ``_daemon_endpoint_file``), under the store
        # lock the tick holds from its running-jobs snapshot through
        # its launches: a tick skips the job meanwhile and a second
        # run_now is refused.
        canonical = importlib.import_module("kiss.agents.sorcar.cron_agent")
        me = threading.current_thread()
        # The registration sits inside the ``try`` so a stop injected
        # between the store and the run still unregisters this thread
        # (a stale entry would refuse every later run_now while the
        # thread lives on in its dispatcher).
        try:
            with _jobs_lock(blocking=True), canonical._running_lock:
                running = canonical._running.get(job_id)
                if running is not None and running.is_alive():
                    return _dump({"error": f"job {job_id!r} is already running"})
                canonical._running[job_id] = me
            _execute_job(match[0])
        finally:
            with canonical._running_lock:
                if canonical._running.get(job_id) is me:
                    del canonical._running[job_id]
        refreshed = [job for job in load_jobs() if job["id"] == job_id]
        return _dump({"ran": _job_view(refreshed[0] if refreshed else match[0])})

    return _dump({
        "error": f"unknown action {action!r}: use create, ensure, list, remove, "
        "pause, resume, or run_now"
    })


def _channel_cli_name(channel: str) -> str:
    """Return the console-script name of a channel agent's CLI.

    Looks the channel's ``main`` up among the installed
    ``console_scripts`` entry points (``kiss-telegram`` for
    ``telegram``, but ``kiss-gchat`` for ``googlechat`` and ``kiss-ha``
    for ``homeassistant``), falling back to ``kiss-<channel>`` when
    the package is not installed as a distribution.

    Args:
        channel: The channel's module short name (``"telegram"``).

    Returns:
        The CLI command name.
    """
    target = f"kiss.agents.third_party_agents.{channel}.{channel}_sea:main"
    for entry in entry_points(group="console_scripts"):
        if entry.value == target:
            return entry.name
    return f"kiss-{channel}"


_GATEWAY_PROMPT_WORDS = (
    re.compile(r"\bgateway\b", re.IGNORECASE),
    re.compile(r"\b(tick|ticks|heartbeat|heartbeats|pairing)\b", re.IGNORECASE),
    re.compile(r"\b(chat|channel|messages?|dm|dms|room|group)\b", re.IGNORECASE),
)
"""Word groups that together identify a prompt as a messaging-gateway tick."""


def _describes_gateway(prompt: str) -> bool:
    """Return whether *prompt* asks an LLM run to act as a messaging gateway.

    A prompt that speaks of a *gateway*, of its *tick* (heartbeat,
    pairing) and of a *chat* (channel, messages, room, group) is the one
    failure mode the cron agent is told to avoid: scheduled as a prompt
    job it starts a paid session on every tick and that session posts
    status chatter (``tick-OK``, ``gateway heartbeat``) into the chat it
    is supposed to serve.  All three word groups are required so an
    ordinary job about an API gateway's health or heartbeat passes.

    Args:
        prompt: The prompt of a job to create (empty for a command job).

    Returns:
        ``True`` when the job must be refused in favour of
        :func:`gateway_command`.
    """
    return all(words.search(prompt) for words in _GATEWAY_PROMPT_WORDS)


def _is_gateway_command(command: str) -> bool:
    """Return whether *command* is a channel CLI's gateway tick.

    True when the command's program (its first shell word, bare or with
    any path prefix — also a ``$(...)`` lookup of the launcher path) is
    a ``kiss-<channel>`` launcher and a later word is ``--channel`` or
    ``--channel=...``, i.e. what :func:`gateway_command` builds, with or
    without extra flags such as ``--allow-users``.  A command that
    merely mentions a launcher as data (``printf 'kiss-slack
    --channel=X'``) is not a gateway.

    Args:
        command: The shell command of a job to create (empty for a
            prompt job).

    Returns:
        ``True`` for a gateway tick command.
    """
    # A ``"$(...)"`` launcher lookup is one shell word even when the
    # quotes nested inside it (a path with spaces) would confuse shlex.
    lookup = re.match(r'\s*"?(\$\(.*?\))"?(\s|$)', command, re.DOTALL)
    try:
        if lookup:
            words = [lookup.group(1), *shlex.split(command[lookup.end():])]
        else:
            words = shlex.split(command)
    except ValueError:
        return False
    if not words or not re.search(r"(^|/)kiss-[a-z0-9]+\b", words[0]):
        return False
    return any(word == "--channel" or word.startswith("--channel=") for word in words[1:])


def _gateway_delivery(command: str, deliver: str) -> str:
    """Return the delivery spec a job may keep given its command.

    A gateway tick never posts into a chat: when *command* is a gateway
    tick (:func:`_is_gateway_command`) and *deliver* names a channel
    target, the result is ``"none"``; otherwise *deliver* is returned
    unchanged.  Applied when a job is created and when the store is
    loaded, so gateway jobs stored before this rule existed are
    normalised the same way.

    Args:
        command: The job's shell command (empty for a prompt job).
        deliver: The requested comma-separated delivery targets.

    Returns:
        The delivery spec to store.
    """
    if _is_gateway_command(command) and _has_channel_target(deliver):
        return "none"
    return deliver


def _has_channel_target(deliver: str) -> bool:
    """Return whether a ``deliver`` spec names any messaging-channel target.

    Args:
        deliver: Comma-separated delivery targets as passed to
            :func:`cron_job`.

    Returns:
        ``True`` when at least one target is neither ``local`` nor
        ``none`` (nor empty).
    """
    return any(
        target.strip() not in ("", "local", "none") for target in deliver.split(",")
    )


def gateway_command(
    channel: str, chat: str, pairing: bool = True, workspace: str = "",
) -> str:
    """Build the shell command for one always-on gateway tick of a messaging channel.

    An always-on gateway makes a chat on a messaging channel a prompt
    surface for Sorcar: every tick fetches the chat's pending messages,
    runs a task per message, and replies in-channel.  Set one up by
    calling this tool FIRST, then scheduling its result as a
    ``command`` job with ``cron_job("create", command=<result>,
    schedule="every 2m", ...)`` — never as a ``prompt`` job, which
    would start a paid LLM session on every tick even when nobody
    wrote anything.  The command runs the CLI tick with ``--quiet``
    (replies go to the channel itself), so a tick that finds nothing
    prints nothing and is silent; one that served messages or failed
    is logged locally only — ``cron_job`` stores a gateway job with
    ``deliver`` ``none`` whatever was requested, so no tick ever posts
    status chatter into the chat it serves.

    Args:
        channel: Channel name, e.g. "telegram", "slack", "Google Chat" (case/spaces ignored).
        chat: Chat id; a chat/room NAME works only on Slack, Discord, Matrix and Google Chat.
        pairing: Whether unknown senders get a one-time approval code instead of service.
        workspace: Account identifier for multi-account channels; empty = default account.

    Returns:
        The command line to schedule (e.g.
        ``kiss-telegram --channel=-100123 --pairing --quiet``), or a
        string starting with ``error:`` when *channel* is unknown or
        has no gateway mode.
    """
    from kiss.agents.sorcar.agent_dispatch import _squash, available_channels

    matches = [name for name in available_channels() if _squash(name) == _squash(channel)]
    if not matches:
        return f"error: unknown channel {channel!r}"
    channel = matches[0]
    module = importlib.import_module(
        f"kiss.agents.third_party_agents.{channel}.{channel}_sea"
    )
    if not callable(getattr(module, "_make_backend", None)):
        return f"error: channel {matches[0]!r} has no gateway (poll) mode"
    if not chat.strip():
        return "error: chat is required (a chat id or channel name)"
    parts = [_channel_cli_name(matches[0]), f"--channel={shlex.quote(chat.strip())}"]
    if pairing:
        parts.append("--pairing")
    if workspace.strip():
        parts.append(f"--workspace={shlex.quote(workspace.strip())}")
    parts.append("--quiet")
    return " ".join(parts)


CRON_DISPATCH_PREAMBLE = (
    "You are the cron scheduling agent: this session already has the "
    "cron_job tool for managing scheduled automations — use it directly "
    "and immediately, without exploring any source code.  Translate the "
    "user's natural-language schedule into one of the tool's four "
    "supported schedule forms yourself.  All schedule times are Pacific "
    "time (PDT): write \"9am\" as hour 9 and convert times the user gives "
    "in another zone to Pacific, never to UTC.  For polls and checks (\"is X "
    "released?\", \"is the site up?\") create a no-LLM command job (curl/"
    "grep pipeline that prints only when there is news) rather than a "
    "prompt job, and pass until_delivered=True when the user wants to be "
    "told once.  For an always-on gateway on a messaging channel (\"make "
    "my Telegram group a chat surface\", \"run a gateway tick on Slack "
    "channel eng every 2 minutes\") FIRST call gateway_command(channel, "
    "chat, pairing) to convert the request into the channel CLI's tick "
    "command, THEN schedule that exact string as a command job (default "
    "schedule \"every 2m\", deliver \"none\"); never schedule a gateway "
    "as a prompt job — it would burn tokens on every tick.  A job that "
    "must work inside a specific project (\"run the tests in "
    "/home/me/proj nightly and fix them\") gets that directory as "
    "work_dir; for a git repository whose changes should land on its "
    "branch pass use_worktree=True and auto_commit=True as well, and "
    "give long runs a larger timeout (seconds; default 3600 for a "
    "prompt job).  create "
    "refuses a job whose prompt/command, schedule and delivery match an "
    "existing scheduled or paused job: report that to the user instead "
    "of retrying under a different name.  Never call run_agent here: it "
    "would just recurse into another session like this one.  Act only "
    "through cron_job and gateway_command: never edit source files, run "
    "test suites or debug the channel CLI — when the scheduled command "
    "itself is broken, report the failing command and its output in your "
    "result so it is fixed in a normal development task.\n\n"
)
"""Guidance appended to the system prompt of every cron-management session.

Appended by the SEA's ``system_prompt`` method, which the daemon applies when
``run_agent`` is called with ``"cron"`` as the agent (after the
``channel`` kind's generic preamble).
"""


class CronAgentSea(ChannelSea):
    """The ``/cron_agent`` SEA."""

    def tools(self, tools: list[Any]) -> list[Any]:
        """Return the cron tools (``kiss.server.sorcar.run`` SEA contract).

        Called by the kiss-web daemon when this module's path is passed as
        the API's ``sea_path``.

        Returns:
            The :func:`cron_job` and :func:`gateway_command` tools.
        """
        return tools + [cron_job, gateway_command]

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/cron help``."""
        return (
            "Scheduled automations: create, list, pause, resume, remove or run now "
            "the cron jobs of this KISS home (a polled messaging gateway is a cron "
            "job too)."
        )

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Configure a cron-management session: a channel in the cron work directory.

        A ``ChannelSea``: no git lifecycle (managing the JSON job store needs
        none), nothing inherited from the calling task, the channel
        preamble in the system prompt.  Classification is off: unattended
        scheduled automations should not spend a classifier round trip.
        """
        return settings | {"work_dir": cron_work_dir()}

    def system_prompt(self, system_prompt: str) -> str:
        """Return :data:`CRON_DISPATCH_PREAMBLE`, appended to the session's system prompt."""
        return system_prompt + "\n\n" + CRON_DISPATCH_PREAMBLE


def cron_work_dir() -> str:
    """Return the work directory of cron-management sessions.

    A ``run_agent(agent="cron", ...)`` session manages the job store
    under ``$KISS_HOME/cron`` and never touches the calling project, so it
    runs in ``$KISS_HOME/cron/work``.  Scheduled runs do not share it:
    each gets a private directory under ``$KISS_HOME/cron/runs`` (see
    :func:`_execute_job`).

    Returns:
        The cron work directory path (created when absent).
    """
    work_dir = _cron_dir() / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    return str(work_dir)


def main() -> None:
    """Run the ``kiss-cron`` CLI: manage jobs or run the scheduler."""
    if len(sys.argv) <= 1:
        print(
            "Usage: kiss-cron (--daemon [--interval SECONDS] | --tick | --list |\n"
            "  --create NAME --schedule S (--prompt P | --command C)\n"
            "    [--deliver TARGETS] [-m MODEL] [-b BUDGET] [--until-delivered]\n"
            "    [--work-dir DIR] [--worktree] [--auto-commit] [--timeout SECONDS] |\n"
            "  --remove ID | --pause ID | --resume ID | --run ID)"
        )
        sys.exit(1)
    parser = argparse.ArgumentParser(prog="kiss-cron")
    parser.add_argument("--daemon", action="store_true", help="Run the scheduler loop")
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_TICK_INTERVAL_SECONDS,
        help="Seconds between scheduler passes in --daemon mode",
    )
    parser.add_argument("--tick", action="store_true", help="Run one scheduler pass")
    parser.add_argument("--list", action="store_true", help="List all jobs")
    parser.add_argument("--create", default="", metavar="NAME", help="Create a job")
    parser.add_argument("--schedule", default="", help="Schedule for --create")
    parser.add_argument("--prompt", default="", help="LLM task for --create")
    parser.add_argument("--command", default="", help="Shell command for --create")
    parser.add_argument("--deliver", default="local", help="Delivery targets")
    parser.add_argument("-m", "--model", default="", help="Model for prompt jobs")
    parser.add_argument("-b", "--budget", default="", help="Per-run USD budget")
    parser.add_argument(
        "--until-delivered", action="store_true",
        help="Disable the job after its first non-silent delivery",
    )
    parser.add_argument(
        "--work-dir", default="", metavar="DIR",
        help="Directory the run works in (default: a private scratch directory)",
    )
    parser.add_argument(
        "--worktree", action="store_true",
        help="Run the prompt job in a git worktree of --work-dir",
    )
    parser.add_argument(
        "--auto-commit", action="store_true",
        help="Auto-commit (and merge) the prompt job's changes",
    )
    parser.add_argument(
        "--timeout", default="", metavar="SECONDS", help="Per-run timeout",
    )
    parser.add_argument("--remove", default="", metavar="ID", help="Remove a job")
    parser.add_argument("--pause", default="", metavar="ID", help="Pause a job")
    parser.add_argument("--resume", default="", metavar="ID", help="Resume a job")
    parser.add_argument("--run", default="", metavar="ID", help="Run a job now")
    args = parser.parse_args()

    if args.daemon:
        logging.basicConfig(level=logging.INFO)
        print(f"kiss-cron scheduler running (every {args.interval:.0f}s); Ctrl-C to stop")
        run_scheduler(threading.Event(), args.interval)
    elif args.tick:
        print(f"ran {tick()} job(s)")
    elif args.list:
        print(cron_job("list"), end="")
    elif args.create:
        print(
            cron_job(
                "create",
                name=args.create,
                prompt=args.prompt,
                command=args.command,
                schedule=args.schedule,
                deliver=args.deliver,
                model_name=args.model,
                max_budget=args.budget,
                until_delivered=args.until_delivered,
                work_dir=args.work_dir,
                use_worktree=args.worktree,
                auto_commit=args.auto_commit,
                timeout=args.timeout,
            ),
            end="",
        )
    elif args.remove:
        print(cron_job("remove", job_id=args.remove), end="")
    elif args.pause:
        print(cron_job("pause", job_id=args.pause), end="")
    elif args.resume:
        print(cron_job("resume", job_id=args.resume), end="")
    elif args.run:
        print(cron_job("run_now", job_id=args.run), end="")
    else:
        print("Nothing to do: pass --daemon, --tick, --list, --create, "
              "--remove, --pause, --resume, or --run")
        sys.exit(1)


if __name__ == "__main__":
    main()
