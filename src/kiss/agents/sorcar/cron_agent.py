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
  :func:`tools` and runs in ``~/.kiss/cron/work`` without a
  worktree (:func:`work_dir`, :func:`use_worktree`,
  :func:`auto_commit`).
- The kiss-web daemon runs the scheduler automatically in a
  background thread (:func:`start_scheduler_thread`): every ~60
  seconds a tick finds due jobs, reschedules them *before* running
  (so the same occurrence never double-fires), and launches each one
  CONCURRENTLY in its own thread and its own scratch directory
  (``~/.kiss/cron/runs/<job_id>-<random>``, removed when the run
  ends) so simultaneous jobs never share a working directory.  The
  scheduler thread does not wait for the jobs: a tick that overlaps
  runs from a previous tick is not skipped — it simply leaves the
  jobs that are still running alone (they stay due and are picked up
  by the first tick after they finish) and starts the rest.
  ``kiss-cron --tick`` and ``kiss-cron --daemon`` remain available
  for running the scheduler outside the daemon; in that mode command
  jobs work standalone while prompt jobs still need a reachable
  kiss-web daemon (they are submitted through its socket).
- An always-on channel gateway (README §19) is a scheduled COMMAND
  job — the channel CLI's poll tick, e.g. ``kiss-telegram
  --channel=-100123 --pairing`` — never a prompt job: a tick that
  finds no messages costs no tokens.  :func:`gateway_command` builds
  that command deterministically from the channel and chat names.
- Delivery targets are looked up dynamically: any module named
  ``kiss.agents.third_party_agents.<channel>_sea`` with a
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
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.useful_tools import _popen_kwargs
from kiss.core.config import kiss_home
from kiss.core.processes import SIGKILL, kill_process_group, popen_process_group

logger = logging.getLogger(__name__)

COMMAND_TIMEOUT_SECONDS = 600.0
PROMPT_TIMEOUT_SECONDS = 3600.0

_running_lock = threading.Lock()
_running: dict[str, threading.Thread] = {}
"""Job id -> thread of every job run launched by this process's ticks.

Process-local (the JSON store holds no lease): a tick treats a job whose
thread is still alive as not due, so a run that outlasts the job's
interval is never overlapped by the next tick in the same process.
``kiss-cron --tick`` in another process cannot see these runs and may
still overlap them, as may ``run_now``.
"""

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


@contextlib.contextmanager
def _jobs_lock(blocking: bool) -> Iterator[Any | None]:
    """Acquire the inter-process lock guarding the job store.

    The same lock serializes the scheduler's tick (non-blocking: an
    overlapping tick skips) and the tool's read-modify-write
    (blocking: the tool waits for a running tick to finish), so a job
    edit can never be overwritten by a stale in-memory save.  Thin
    wrapper over :func:`kiss.agents.sorcar.useful_tools._file_lock`,
    which owns the cross-platform (fcntl/msvcrt/no-op) mechanics.

    Args:
        blocking: Whether to wait for the lock (tool path) or give up
            immediately when it is held (tick path).

    Yields:
        A truthy value while the lock is held, or ``None`` when
        *blocking* is ``False`` and another process holds it.
    """
    from kiss.agents.sorcar.useful_tools import _file_lock

    with _file_lock(_jobs_path().with_suffix(".lock"), blocking=blocking) as held:
        yield held


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

    Imports ``kiss.agents.third_party_agents.<channel>_sea``, builds
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
            f"kiss.agents.third_party_agents.{channel}_sea"
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
    job: dict[str, Any],
    sock_path: str | None = None,
    work_dir: Path | None = None,
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
        work_dir: The run's private scratch directory (see
            :func:`_execute_job`, which every scheduled and ``run_now``
            run goes through); ``None`` (direct callers) uses the
            shared ``~/.kiss/cron/work``.

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
    if work_dir is None:
        work_dir = _cron_dir() / "work"
        work_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Scheduled runs execute in the cron scratch directory, outside
        # any project git lifecycle: no ``extension_agent_path`` is
        # passed here, so this module's ``use_worktree()`` /
        # ``auto_commit()`` getters do NOT apply and the values must be
        # pinned on the wire.  ``classify_tasks=False`` is pinned too:
        # cron is the one dispatch mode that never classifies — an
        # unattended scheduled automation runs repeatedly, and a
        # classifier round trip on every run buys nothing a one-off
        # dispatch would not already get.  (The worktree pin alone is
        # safe regardless: a classification verdict can only demote a
        # requested worktree run, never promote a pinned-off one.)
        # ``stop_on_timeout=True``: the run's scratch directory is
        # removed as soon as this function returns, so a task that
        # outlives the timeout must be stopped, not left running in a
        # directory that is about to disappear.  A stop the daemon
        # never confirmed (``StopUnconfirmedTimeoutError``) propagates
        # to :func:`_execute_job`, which then keeps the directory.
        result = daemon_client.run(
            preamble + str(job.get("prompt", "")),
            work_dir=str(work_dir),
            model=str(job.get("model_name", "")),
            use_worktree=False,
            auto_commit=False,
            classify_tasks=False,
            max_budget=float(job["max_budget"]) if job.get("max_budget") else None,
            timeout=PROMPT_TIMEOUT_SECONDS,
            stop_on_timeout=True,
            sock_path=sock_path,
        )
    except daemon_client.StopUnconfirmedTimeoutError:
        raise
    except TimeoutError:
        return "error", (
            f"prompt job timed out after {PROMPT_TIMEOUT_SECONDS:.0f}s "
            "and was stopped"
        )
    except OSError as e:
        # Name the exact socket the client resolved (same precedence
        # rules, same helper) instead of re-deriving it here.
        sock = daemon_client._resolve_sock_path(sock_path)
        return "error", (
            f"cannot reach the kiss-web daemon at {sock}: {e} "
            "(prompt jobs need a running kiss-web daemon; "
            "command jobs work without one)"
        )
    summary = result.text or ("" if result.success else "Task failed")
    if not summary or _is_silent(summary):
        return "silent", None
    return ("ok" if result.success else "error"), summary


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
    job: dict[str, Any],
    timeout_seconds: float | None = None,
    work_dir: Path | None = None,
) -> tuple[str, str | None]:
    """Run a no-LLM command job (Hermes "no_agent" mode).

    The command runs under the same shell as the agent's ``Bash`` tool
    (``sh`` on POSIX, Git bash on Windows; see
    :func:`~kiss.agents.sorcar.useful_tools._popen_kwargs`) in its own
    process group, and a timeout kills the WHOLE process tree — not
    just the shell.
    ``subprocess.run(..., shell=True, timeout=...)`` kills only the
    shell on expiry, so every descendant the command spawned survived
    the timeout and kept running (and writing) forever, with a
    repeating schedule spawning a fresh orphan tree on every tick; a
    plain ``killpg`` still missed ``setsid``/daemonizing descendants.
    See :func:`_kill_command_tree` for the exact guarantees and
    residual limitations per platform.

    Args:
        job: The job dict (uses ``command``).
        timeout_seconds: Maximum runtime before the command's process
            group is killed; ``None`` uses
            :data:`COMMAND_TIMEOUT_SECONDS`.
        work_dir: Working directory for the command (the run's private
            scratch directory, see :func:`_execute_job`); ``None``
            inherits this process's working directory.

    Returns:
        ``(status, text)``: ``("silent", None)`` when the command
        succeeds with empty output, ``("ok", stdout)`` on success, and
        ``("error", output)`` on non-zero exit or timeout.
    """
    timeout = COMMAND_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    proc = popen_process_group(
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=None if work_dir is None else str(work_dir),
        **_popen_kwargs(str(job["command"])),
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_command_tree(proc)
        # The tree is dead, so the pipes close and this drain returns
        # promptly; the bounded retry guards an exotic straggler that
        # survived the best-effort tree kill holding the pipe open.
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover — defensive
            proc.kill()
        return "error", f"command timed out after {timeout:.0f}s"
    output = stdout.strip()
    if proc.returncode != 0:
        detail = (output + "\n" + stderr.strip()).strip()
        return "error", f"command exited {proc.returncode}: {detail}"
    if not output:
        return "silent", None
    return "ok", output


def _execute_job(job: dict[str, Any], sock_path: str | None = None) -> None:
    """Execute one job, deliver its result, and record the outcome.

    The run gets a fresh private scratch directory under
    :func:`_runs_dir` — its working directory for a command job, the
    daemon session's ``work_dir`` for a prompt job — so jobs running
    concurrently never share one; the directory is removed when the
    run ends, whatever the outcome, so stale run directories never
    accumulate.  The one exception is a timed-out prompt job whose
    stop the daemon never confirmed: the task may still be running in
    the directory, so it is kept and its path reported.

    Never raises: failures are recorded in the job's ``last_status`` /
    ``last_summary`` fields.  Errors are still delivered (so the user
    learns the automation broke), silent results are not.

    Args:
        job: The job dict to execute.
        sock_path: Daemon UDS path override for prompt jobs.
    """
    _runs_dir().mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{job['id']}-", dir=_runs_dir()))
    keep_work_dir = False
    try:
        if str(job.get("command", "")).strip():
            status, text = _run_command_job(job, work_dir=work_dir)
        else:
            status, text = _run_prompt_job(
                job, sock_path or _recorded_daemon_sock_path(), work_dir,
            )
    except TimeoutError as e:
        # Only daemon_client.StopUnconfirmedTimeoutError reaches here
        # (_run_prompt_job handles the confirmed-stop timeout itself).
        keep_work_dir = True
        status, text = "error", (
            f"prompt job timed out after {PROMPT_TIMEOUT_SECONDS:.0f}s; the stop "
            f"was not confirmed, so the task may still be running in {work_dir} "
            f"(directory kept): {e}"
        )
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


def tick(
    now: float | None = None, sock_path: str | None = None, wait: bool = True,
) -> int:
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

    A job still running from a previous tick of this process (see
    :data:`_running`) is not due: the tick leaves it untouched (its
    ``next_run_at`` stays in the past) and starts only the other due
    jobs, so it runs again on the first tick after it finishes instead
    of overlapping itself.  The registry is process-local, so
    ``kiss-cron --tick`` in another process and ``run_now`` may still
    overlap a run; a one-shot claimed by a process that crashes
    mid-run is not retried.

    Args:
        now: Current time in epoch seconds; ``None`` uses the clock.
        sock_path: Daemon UDS path override for prompt jobs.
        wait: Whether to block until every launched job has finished
            (the ``kiss-cron --tick`` CLI).  The scheduler loop passes
            ``False`` so a long job never delays the next tick.

    Returns:
        The number of jobs launched (``0`` when another tick holds the
        lock).
    """
    now = time.time() if now is None else now
    with _jobs_lock(blocking=False) as lock_fp:
        if lock_fp is None:
            return 0
        jobs = load_jobs()
        running = running_job_ids()
        due = []
        changed = False
        for job in jobs:
            if not job.get("enabled") or job["id"] in running:
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
        # Registered and started while holding _running_lock (and the
        # store lock): a thread only counts as alive once started, so
        # no concurrent running_job_ids() call can observe — and prune
        # — a registered-but-idle entry.  A job that finishes instantly
        # merely waits for the store lock before recording its outcome.
        threads = []
        for job in due:
            thread = threading.Thread(
                target=_execute_job, args=(job, sock_path),
                name=f"kiss-cron-job-{job['id']}", daemon=True,
            )
            with _running_lock:
                _running[job["id"]] = thread
                thread.start()
            threads.append(thread)
    if wait:
        for thread in threads:
            thread.join()
    return len(due)


def run_scheduler(
    stop_event: threading.Event,
    interval: float = DEFAULT_TICK_INTERVAL_SECONDS,
    sock_path: str | None = None,
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
        sock_path: Daemon UDS path override for prompt jobs.
    """
    while not stop_event.is_set():
        try:
            ran = tick(sock_path=sock_path, wait=False)
            if ran:
                logger.info("kiss-cron: launched %d job(s)", ran)
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
            "until_delivered", "prompt", "command", "last_status",
            "last_delivery",
        )
        if job.get(key) not in (None, "", [])
    }
    for key in ("next_run_at", "last_run_at"):
        if job.get(key):
            view[key] = datetime.fromtimestamp(float(job[key])).isoformat(
                timespec="seconds"
            )
    return view


def _find_duplicate(
    jobs: list[dict[str, Any]], candidate: dict[str, Any]
) -> dict[str, Any] | None:
    """Return the stored job that would do the same work as ``candidate``.

    Two jobs are duplicates when they run the same ``prompt`` or
    ``command`` on the same ``schedule`` with the same ``deliver``
    targets.  The name, model and budget are ignored: the LLM picks
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
            for key in ("prompt", "command", "schedule", "deliver")
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
    - ``list``: list all jobs with their next/last run times.
    - ``remove`` / ``pause`` / ``resume``: manage the job named by
      ``job_id``.
    - ``run_now``: execute the job named by ``job_id`` immediately and
      deliver its result (the regular schedule is unaffected).

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
    (typically ``"every 2m"``) — never as a ``prompt`` job.

    Jobs due at the same time run concurrently, each in its own scratch
    directory that is removed when the run ends; a job whose previous
    run is still in progress is not started again until it finishes.

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
        until_delivered: Disable the job after its first non-silent
            delivery (create; for "notify me when ..." polls).

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
    target = f"kiss.agents.third_party_agents.{channel}_sea:main"
    for entry in entry_points(group="console_scripts"):
        if entry.value == target:
            return entry.name
    return f"kiss-{channel}"


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
    is logged and delivered.

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
    module = importlib.import_module(f"kiss.agents.third_party_agents.{matches[0]}_sea")
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
    "supported schedule forms yourself.  For polls and checks (\"is X "
    "released?\", \"is the site up?\") create a no-LLM command job (curl/"
    "grep pipeline that prints only when there is news) rather than a "
    "prompt job, and pass until_delivered=True when the user wants to be "
    "told once.  For an always-on gateway on a messaging channel (\"make "
    "my Telegram group a chat surface\", \"run a gateway tick on Slack "
    "channel eng every 2 minutes\") FIRST call gateway_command(channel, "
    "chat, pairing) to convert the request into the channel CLI's tick "
    "command, THEN schedule that exact string as a command job (default "
    "schedule \"every 2m\", deliver \"none\"); never schedule a gateway "
    "as a prompt job — it would burn tokens on every tick.  create "
    "refuses a job whose prompt/command, schedule and delivery match an "
    "existing scheduled or paused job: report that to the user instead "
    "of retrying under a different name.  Never call run_agent here: it "
    "would just recurse into another session like this one.\n\n"
)
"""Preamble prepended to every task dispatched to this agent script.

Used by ``kiss.agents.sorcar.agent_dispatch`` when the ``run_agent``
tool is called with ``"cron"`` as the agent, mirroring the channel
agents' dispatch preamble.
"""


def tools() -> list:
    """Return the cron tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument — including when the module is passed
    as the ``extension_agent_path``, which makes it its own tools file.

    Returns:
        The :func:`cron_job` and :func:`gateway_command` tools.
    """
    return [cron_job, gateway_command]


def work_dir() -> str:
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


def use_worktree() -> bool:
    """Return whether dispatched cron sessions use a git worktree.

    Agent-script getter: managing the JSON job store needs no git
    lifecycle.

    Returns:
        ``False``.
    """
    return False


def auto_commit() -> bool:
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
    parser.add_argument(
        "--until-delivered", action="store_true",
        help="Disable the job after its first non-silent delivery",
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
