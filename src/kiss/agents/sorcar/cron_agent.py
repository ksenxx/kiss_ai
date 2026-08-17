# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Hermes-style scheduled automations (cron) with delivery to any channel.

Mirrors the Hermes agent's cron design in the simplest possible form:

- Jobs live in a single JSON file (``~/.kiss/cron/jobs.json``) — no
  database.  Atomic writes and an ``flock`` guard make concurrent
  ticks and tool calls safe.
- The natural-language part is done by the LLM: the :func:`cron_job`
  tool accepts only four normalized schedule forms (interval, 5-field
  cron expression, one-shot duration, one-shot ISO timestamp) and the
  agent translates phrases like "every weekday at 9am" into them.
- The Sorcar agent does not carry the :func:`cron_job` tool itself:
  this module is an *agent script* (``kiss.server.sorcar.run``'s
  ``extension_agent_path`` contract), and a scheduling request is dispatched to
  it with the ``run_agent`` tool as ``run_agent("cron", task)`` — the
  dispatched session gets the :func:`cron_job` tool from
  :func:`get_tools` and runs in ``~/.kiss/cron/work`` without a
  worktree (:func:`get_work_dir`, :func:`get_use_worktree`,
  :func:`get_auto_commit`).
- The kiss-web daemon runs the scheduler automatically in a
  background thread (:func:`start_scheduler_thread`): every ~60
  seconds a tick finds due jobs, reschedules them *before* running
  (so the same occurrence never double-fires; a job that outlasts
  its interval may still overlap its next run), runs each in a fresh
  daemon session, and delivers the result.  ``kiss-cron --tick`` and
  ``kiss-cron --daemon`` remain available for running the scheduler
  outside the daemon; in that mode command jobs work standalone while
  prompt jobs still need a reachable kiss-web daemon (they are
  submitted through its socket).
- Delivery targets are looked up dynamically: any module named
  ``kiss.agents.third_party_agents.<channel>_agent`` with a
  ``_make_backend()`` factory can receive results (``telegram:123``,
  ``slack:eng``, ``ntfy``, ...).  This module works without those
  optional channel modules — an unknown channel just yields a
  delivery-error note.  Every run is also appended to a local log
  under ``~/.kiss/cron/output/``.  A ``[SILENT]`` summary (or empty
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
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from kiss.core.config import kiss_home

logger = logging.getLogger(__name__)

COMMAND_TIMEOUT_SECONDS = 600.0
PROMPT_TIMEOUT_SECONDS = 3600.0

_daemon_sock_path: str | None = None
"""UDS path of the kiss-web daemon hosting this process's scheduler.

Set by :func:`start_scheduler_thread` so tool calls executed inside the
daemon (e.g. ``cron_job("run_now", ...)``) submit prompt jobs back to
the same daemon even when it serves a non-default socket.  Always read
through :func:`_recorded_daemon_sock_path`, which resolves the
CANONICAL module's value: dispatched cron sessions get their
``cron_job`` tool from a fresh synthetic copy of this module whose own
global is never set.
"""
CRON_SCAN_DAYS = 4 * 366 + 1  # covers the largest gap between leap days
DEFAULT_TICK_INTERVAL_SECONDS = 60.0
MAX_STORED_SUMMARY_CHARS = 4000

_UNIT_SECONDS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_DURATION_RE = re.compile(r"^(\d+)\s*(s|m|h|d)$")
_INTERVAL_RE = re.compile(r"^every\s+(\d+)\s*(s|m|h|d)$")

_CRON_BOUNDS = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))


def _cron_dir() -> Path:
    """Return the cron state directory (``$KISS_HOME/cron``)."""
    return kiss_home() / "cron"


def _jobs_path() -> Path:
    """Return the path of the JSON job store."""
    return _cron_dir() / "jobs.json"


def _output_dir() -> Path:
    """Return the directory holding per-job local output logs."""
    return _cron_dir() / "output"


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
    return re.sub(r"<[^>]+>", "", summary).strip() in _SILENCE_TOKENS


@contextlib.contextmanager
def _jobs_lock(blocking: bool) -> Iterator[Any | None]:
    """Acquire the ``flock`` guarding the job store.

    The same lock serializes the scheduler's tick (non-blocking: an
    overlapping tick skips) and the tool's read-modify-write
    (blocking: the tool waits for a running tick to finish), so a job
    edit can never be overwritten by a stale in-memory save.  On
    platforms without ``fcntl`` the lock file is opened but not
    locked.

    Args:
        blocking: Whether to wait for the lock (tool path) or give up
            immediately when it is held (tick path).

    Yields:
        The open lock file object while the lock is held, or ``None``
        when *blocking* is ``False`` and another process holds it.
    """
    lock_path = _jobs_path().with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fp = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-Unix platforms
            yield fp
            return
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(fp.fileno(), flags)
        except BlockingIOError:
            yield None
            return
        yield fp
    finally:
        fp.close()


def load_jobs() -> list[dict[str, Any]]:
    """Load all cron jobs from the JSON store.

    Returns:
        The list of job dicts; an empty list when the store does not
        exist or is unreadable.
    """
    try:
        data = json.loads(_jobs_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [job for job in data if isinstance(job, dict) and job.get("id")]


def save_jobs(jobs: list[dict[str, Any]]) -> None:
    """Atomically persist the full job list to the JSON store.

    Writes to a temporary sibling file and renames it over the store so
    readers never observe a partially written file.

    Args:
        jobs: The complete list of job dicts to write.
    """
    path = _jobs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(jobs, indent=2))
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


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
        dt: The local datetime whose date to test.

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
      evaluated in local time.
    - One-shot duration: ``"30m"``, ``"1d"`` (relative to *now*).
    - One-shot ISO 8601 timestamp: ``"2026-01-15T14:00:00"`` (local
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
        Cron times use naive local time: around a DST transition a run
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
        dt = datetime.fromtimestamp(now).replace(second=0, microsecond=0)
        dt += timedelta(minutes=1)
        for _ in range(CRON_SCAN_DAYS):
            if _cron_date_matches(fields, dom_star, dow_star, dt):
                day = dt.date()
                while dt.date() == day:
                    if dt.minute in minute_set and dt.hour in hour_set:
                        return dt.timestamp()
                    dt += timedelta(minutes=1)
            else:
                dt = datetime(dt.year, dt.month, dt.day) + timedelta(days=1)
        return None
    try:
        when = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(
            f"Unsupported schedule {schedule!r}: use 'every N<s|m|h|d>', a "
            "5-field cron expression, a one-shot duration like '30m', or an "
            "ISO timestamp like '2026-01-15T14:00'"
        ) from None
    ts = when.timestamp()
    return ts if ts > now else None


def _recorded_daemon_sock_path() -> str | None:
    """Return the daemon UDS recorded for this process, if any.

    :func:`start_scheduler_thread` records the hosting daemon's socket
    in the canonical ``kiss.agents.sorcar.cron_agent`` module.  The
    kiss-web daemon, however, re-executes this file as a fresh
    synthetic tools-file module for every dispatched cron session
    (``run_agent("cron", ...)``), whose own :data:`_daemon_sock_path`
    global is never set — so the lookup goes through
    :data:`sys.modules` to the canonical module, falling back to this
    module's own global for canonical and standalone callers.

    Returns:
        The recorded daemon socket path, or ``None`` when this process
        hosts no scheduler.
    """
    canonical = sys.modules.get("kiss.agents.sorcar.cron_agent")
    recorded = getattr(canonical, "_daemon_sock_path", None)
    return recorded or _daemon_sock_path


def _deliver_to_channel(channel: str, chat: str, text: str) -> str:
    """Send *text* to one channel agent's backend.

    Imports ``kiss.agents.third_party_agents.<channel>_agent``, builds
    its backend with the module's ``_make_backend()`` factory (which
    loads the credentials persisted under ``~/.kiss``), and calls
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
            f"kiss.agents.third_party_agents.{channel}_agent"
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
    (``~/.kiss/cron/output/<job_id>.md``); ``local`` and ``none``
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
    stamp = datetime.now().isoformat(timespec="seconds")
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


def _run_prompt_job(
    job: dict[str, Any], sock_path: str | None = None,
) -> tuple[str, str | None]:
    """Run an LLM cron job in a fresh kiss-web daemon session.

    Submits the prompt to the running kiss-web daemon through the
    public client API :func:`kiss.server.sorcar.run`.  Mirrors Hermes:
    every run gets a brand-new session (no history), with a preamble
    marking the run as unattended and forbidding further scheduling; a
    ``[SILENT]`` (or empty) summary suppresses delivery.

    Args:
        job: The job dict (uses ``prompt``, ``model_name``,
            ``max_budget``).
        sock_path: Daemon UDS path override; ``None`` uses the
            standard resolution (``KISS_SORCAR_SOCK`` environment
            variable, then ``$KISS_HOME/sorcar.sock``).

    Returns:
        ``(status, text)`` where status is ``"ok"``, ``"error"``, or
        ``"silent"``; text is the deliverable summary (``None`` when
        silent).
    """
    from kiss.agents.sorcar import daemon_client

    preamble = (
        "You are running as an unattended scheduled automation (cron job). "
        "Nobody can answer questions; never ask the user anything. Do not "
        "create, modify, or remove scheduled jobs during this run. Your "
        "final summary is delivered verbatim to the job's delivery targets; "
        "reply with exactly [SILENT] if there is nothing worth reporting.\n\n"
    )
    work_dir = _cron_dir() / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = daemon_client.run(
            preamble + str(job.get("prompt", "")),
            work_dir=str(work_dir),
            model=str(job.get("model_name", "")),
            max_budget=float(job["max_budget"]) if job.get("max_budget") else None,
            timeout=PROMPT_TIMEOUT_SECONDS,
            sock_path=sock_path,
        )
    except TimeoutError:
        return "error", (
            f"prompt job timed out after {PROMPT_TIMEOUT_SECONDS:.0f}s "
            "(the task keeps running in the daemon)"
        )
    except OSError as e:
        sock = (
            sock_path
            or os.environ.get("KISS_SORCAR_SOCK")
            or str(kiss_home() / "sorcar.sock")
        )
        return "error", (
            f"cannot reach the kiss-web daemon at {sock}: {e} "
            "(prompt jobs need a running kiss-web daemon; "
            "command jobs work without one)"
        )
    summary = result.text or ("" if result.success else "Task failed")
    if not summary or _is_silent(summary):
        return "silent", None
    return ("ok" if result.success else "error"), summary


def _run_command_job(job: dict[str, Any]) -> tuple[str, str | None]:
    """Run a no-LLM command job (Hermes "no_agent" mode).

    Args:
        job: The job dict (uses ``command``).

    Returns:
        ``(status, text)``: ``("silent", None)`` when the command
        succeeds with empty output, ``("ok", stdout)`` on success, and
        ``("error", output)`` on non-zero exit or timeout.
    """
    try:
        proc = subprocess.run(
            str(job["command"]),
            shell=True,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "error", f"command timed out after {COMMAND_TIMEOUT_SECONDS:.0f}s"
    output = proc.stdout.strip()
    if proc.returncode != 0:
        detail = (output + "\n" + proc.stderr.strip()).strip()
        return "error", f"command exited {proc.returncode}: {detail}"
    if not output:
        return "silent", None
    return "ok", output


def _execute_job(job: dict[str, Any], sock_path: str | None = None) -> None:
    """Execute one job, deliver its result, and record the outcome.

    Never raises: failures are recorded in the job's ``last_status`` /
    ``last_summary`` fields.  Errors are still delivered (so the user
    learns the automation broke), silent results are not.

    Args:
        job: The job dict to execute.
        sock_path: Daemon UDS path override for prompt jobs.
    """
    try:
        if str(job.get("command", "")).strip():
            status, text = _run_command_job(job)
        else:
            status, text = _run_prompt_job(
                job, sock_path or _recorded_daemon_sock_path(),
            )
    except Exception as e:
        logger.error("Cron job %s failed: %s", job["id"], e, exc_info=True)
        status, text = "error", f"{type(e).__name__}: {e}"
    notes: list[str] = []
    if text is not None:
        try:
            notes = _deliver(job, text)
        except Exception as e:
            logger.error("Cron delivery for %s failed: %s", job["id"], e, exc_info=True)
            notes = [f"error: delivery failed: {e}"]
    with _jobs_lock(blocking=True):
        jobs = load_jobs()
        for stored in jobs:
            if stored["id"] == job["id"]:
                stored["last_status"] = status
                stored["last_summary"] = (text or "")[:MAX_STORED_SUMMARY_CHARS]
                stored["last_delivery"] = notes
        save_jobs(jobs)


def tick(now: float | None = None, sock_path: str | None = None) -> int:
    """Run one scheduler pass: execute every due job.

    Takes a non-blocking ``flock`` on the job store (an overlapping
    tick exits immediately), selects due jobs, advances their
    ``next_run_at`` (disabling one-shots) BEFORE running so the same
    occurrence can never double-fire, then executes and delivers each
    due job.  A malformed job (bad ``next_run_at`` or schedule) is
    disabled and skipped instead of aborting the tick.

    Deliberate simplicity tradeoffs (no per-job lease is kept, so the
    store stays a plain JSON file): a job whose run outlasts its
    interval may overlap with its next scheduled run, ``run_now`` may
    overlap with a scheduled run, and a one-shot claimed by a tick
    that crashes mid-run is not retried.

    Args:
        now: Current time in epoch seconds; ``None`` uses the clock.
        sock_path: Daemon UDS path override for prompt jobs.

    Returns:
        The number of jobs executed (``0`` when another tick holds the
        lock).
    """
    now = time.time() if now is None else now
    with _jobs_lock(blocking=False) as lock_fp:
        if lock_fp is None:
            return 0
        jobs = load_jobs()
        due = []
        changed = False
        for job in jobs:
            if not job.get("enabled"):
                continue
            try:
                if float(job["next_run_at"]) > now:
                    continue
                next_run = (
                    None
                    if job.get("one_shot")
                    else compute_next_run(str(job["schedule"]), now)
                )
            except (TypeError, ValueError, KeyError) as e:
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
        if not due:
            return 0
    for job in due:
        _execute_job(job, sock_path)
    return len(due)


def run_scheduler(
    stop_event: threading.Event,
    interval: float = DEFAULT_TICK_INTERVAL_SECONDS,
    sock_path: str | None = None,
) -> None:
    """Run the scheduler loop until *stop_event* is set.

    Ticks immediately, then every *interval* seconds.  A failing tick
    is logged and never stops the loop.

    Args:
        stop_event: Setting this event stops the loop (the wait
            between ticks returns early; a tick already executing a
            job finishes it first).
        interval: Seconds between scheduler passes.
        sock_path: Daemon UDS path override for prompt jobs.
    """
    while not stop_event.is_set():
        try:
            ran = tick(sock_path=sock_path)
            if ran:
                logger.info("kiss-cron: ran %d job(s)", ran)
        except Exception as e:
            logger.error("Scheduler tick failed: %s", e, exc_info=True)
        stop_event.wait(interval)


def start_scheduler_thread(
    interval: float = DEFAULT_TICK_INTERVAL_SECONDS,
    sock_path: str | None = None,
) -> threading.Event:
    """Start the scheduler loop in a daemon thread.

    Called by the kiss-web daemon on startup so scheduled automations
    fire without any external cron process.  Prompt jobs are submitted
    back to the daemon through *sock_path*.

    Args:
        interval: Seconds between scheduler passes.
        sock_path: Daemon UDS path override for prompt jobs (the
            daemon passes its own socket); also becomes the module
            default so ``run_now`` tool calls in this process target
            the same daemon.

    Returns:
        The stop event: set it to stop the loop.
    """
    global _daemon_sock_path
    if sock_path:
        _daemon_sock_path = sock_path
    stop_event = threading.Event()
    threading.Thread(
        target=run_scheduler,
        args=(stop_event, interval, sock_path),
        name="kiss-cron-scheduler",
        daemon=True,
    ).start()
    return stop_event


def _job_view(job: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, human-readable view of a job for listings.

    Args:
        job: The stored job dict.

    Returns:
        A dict with the job's key fields and ISO-formatted times.
    """
    view = {
        key: job.get(key)
        for key in (
            "id", "name", "schedule", "deliver", "enabled", "one_shot",
            "prompt", "command", "last_status", "last_delivery",
        )
        if job.get(key) not in (None, "", [])
    }
    for key in ("next_run_at", "last_run_at"):
        if job.get(key):
            view[key] = datetime.fromtimestamp(float(job[key])).isoformat(
                timespec="seconds"
            )
    return view


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
) -> str:
    """Manage scheduled automations (cron jobs) stored in a local JSON file.

    Translate the user's natural-language request (e.g. "every weekday
    at 9am", "in 30 minutes", "every 2 hours") into one of the four
    supported schedule forms yourself before calling this tool.

    Actions:

    - ``create``: register a job.  Requires ``name``, ``schedule``,
      and exactly one of ``prompt`` (an LLM task run unattended in a
      fresh session) or ``command`` (a shell command run without any
      LLM; its stdout is delivered verbatim).
    - ``list``: list all jobs with their next/last run times.
    - ``remove`` / ``pause`` / ``resume``: manage the job named by
      ``job_id``.
    - ``run_now``: execute the job named by ``job_id`` immediately and
      deliver its result (the regular schedule is unaffected).

    Schedule forms (local time):

    - Repeating interval: ``"every 30m"``, ``"every 2h"``
      (units ``s``, ``m``, ``h``, ``d``).
    - Repeating cron: standard 5-field expression, e.g. ``"0 9 * * 1-5"``
      for 9:00 on weekdays.
    - One-shot delay: ``"30m"``, ``"1d"`` (runs once, that far from now).
    - One-shot timestamp: ISO 8601, e.g. ``"2026-01-15T14:00"``.

    Delivery (``deliver``): comma-separated targets.  ``local`` (default)
    only appends to ``~/.kiss/cron/output/<job_id>.md``; any other
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
        action: One of ``create``, ``list``, ``remove``, ``pause``,
            ``resume``, ``run_now``.
        job_id: Job identifier (required for remove/pause/resume/run_now).
        name: Short human-readable job name (create).
        prompt: The LLM task to run on schedule (create).
        command: Shell command to run instead of an LLM task (create).
        schedule: Schedule string in one of the four forms above (create).
        deliver: Comma-separated delivery targets (create; default
            ``local``).
        model_name: LLM model override for prompt jobs (create; empty
            uses the default model).
        max_budget: Per-run USD budget override for prompt jobs, as a
            string like ``"2.5"`` (create; empty uses the default).

    Returns:
        A YAML string describing the result (created job, job list,
        confirmation, or an ``error`` key explaining what went wrong).
    """
    def _dump(data: Any) -> str:
        return str(yaml.safe_dump(data, sort_keys=False))

    if action == "create":
        if not name or not schedule:
            return _dump({"error": "create requires name and schedule"})
        if bool(prompt.strip()) == bool(command.strip()):
            return _dump({"error": "create requires exactly one of prompt or command"})
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
        job = {
            "id": uuid.uuid4().hex[:8],
            "name": name,
            "prompt": prompt.strip(),
            "command": command.strip(),
            "schedule": schedule.strip(),
            "deliver": deliver.strip() or "local",
            "model_name": model_name.strip(),
            "max_budget": budget,
            "enabled": True,
            "one_shot": is_one_shot(schedule),
            "created_at": time.time(),
            "next_run_at": next_run,
            "last_run_at": None,
            "last_status": "",
            "last_summary": "",
            "last_delivery": [],
        }
        with _jobs_lock(blocking=True):
            jobs = load_jobs()
            jobs.append(job)
            save_jobs(jobs)
        return _dump({"created": _job_view(job)})

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
                if match[0].get("one_shot") and match[0].get("next_run_at") is None:
                    return _dump({
                        "error": f"one-shot job {job_id!r} already ran; "
                        "create a new job instead"
                    })
                match[0]["enabled"] = True
                if match[0].get("next_run_at") is None:
                    match[0]["next_run_at"] = compute_next_run(
                        str(match[0]["schedule"]), time.time()
                    )
            save_jobs(jobs)
        return _dump({action: job_id})

    if action == "run_now":
        match = [job for job in load_jobs() if job["id"] == job_id]
        if not match:
            return _dump({"error": f"no job with id {job_id!r}"})
        _execute_job(match[0])
        refreshed = [job for job in load_jobs() if job["id"] == job_id]
        return _dump({"ran": _job_view(refreshed[0] if refreshed else match[0])})

    return _dump({
        "error": f"unknown action {action!r}: use create, list, remove, "
        "pause, resume, or run_now"
    })


CRON_DISPATCH_PREAMBLE = (
    "You are the cron scheduling agent: this session already has the "
    "cron_job tool for managing scheduled automations — use it directly "
    "and immediately, without exploring any source code.  Translate the "
    "user's natural-language schedule into one of the tool's four "
    "supported schedule forms yourself.  Never call run_agent here: it "
    "would just recurse into another session like this one.\n\n"
)
"""Preamble prepended to every task dispatched to this agent script.

Used by ``kiss.agents.sorcar.agent_dispatch`` when the ``run_agent``
tool is called with ``"cron"`` as the agent, mirroring the channel
agents' dispatch preamble.
"""


def get_tools() -> list:
    """Return the cron tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument — including when the module is passed
    as the ``extension_agent_path``, which makes it its own tools file.

    Returns:
        The list containing the :func:`cron_job` tool.
    """
    return [cron_job]


def get_work_dir() -> str:
    """Return the work directory for dispatched cron-management sessions.

    Agent-script getter (``kiss.server.sorcar.run``'s
    ``extension_agent_path`` contract): a ``run_agent("cron", ...)`` session manages the job
    store under ``~/.kiss/cron`` and never touches the calling
    project, so it runs in the cron state directory — the same
    directory :func:`_run_prompt_job` uses for scheduled runs.

    Returns:
        The cron work directory path (created when absent).
    """
    work_dir = _cron_dir() / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    return str(work_dir)


def get_use_worktree() -> bool:
    """Return whether dispatched cron sessions use a git worktree.

    Agent-script getter: managing the JSON job store needs no git
    lifecycle.

    Returns:
        ``False``.
    """
    return False


def get_auto_commit() -> bool:
    """Return whether dispatched cron sessions auto-commit.

    Agent-script getter: managing the JSON job store needs no git
    lifecycle.

    Returns:
        ``False``.
    """
    return False


def main() -> None:
    """Run the ``kiss-cron`` CLI: manage jobs or run the scheduler."""
    if len(sys.argv) <= 1:
        print(
            "Usage: kiss-cron (--daemon [--interval SECONDS] | --tick | --list |\n"
            "  --create NAME --schedule S (--prompt P | --command C)\n"
            "    [--deliver TARGETS] [-m MODEL] [-b BUDGET] |\n"
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
