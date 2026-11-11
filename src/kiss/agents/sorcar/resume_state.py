# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""On-disk state that lets a task continue after its process died.

A task's resume state lives in ``<kiss home>/resume/<task_id>/``:

* ``checkpoint-<n>.json``: the live session's conversation, written by
  :class:`~kiss.core.kiss_agent.KISSAgent` before every model call and
  every tool call (see ``KISSAgent._write_checkpoint``).
* ``session.json``: the session index, the earlier sessions' summaries
  and the spend banked before the session
  (:class:`~kiss.agents.sorcar.relentless_agent.RelentlessAgent`).
* ``launch.json``: the ``run`` command the daemon started the task with
  and how many times the task has been resumed (:func:`write_launch`).
* ``worktrees.json``: the git worktrees the task holds
  (:func:`write_worktrees`), reattached on resume.

The directory exists while the task runs or waits to be resumed and is
removed (:func:`clear_resume_state`) when the task ends for any other
reason: a result, a failure, a user stop.  Only the death of the
process (SIGKILL, OOM, a lost node) leaves it behind, and that is what
tells the daemon's orphan sweep to start the task again under the same
task id, and the worktree reclaim to leave the task's worktree alone.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from kiss.core.config import kiss_home

RESUME_VERSION = 1
"""Format version of ``launch.json``; a file of another version is not resumed."""

MAX_RESUMES = 3
"""A task is resumed at most this many times (the crash-loop guard)."""

MAX_RESUME_AGE_S = 6 * 3600
"""A task whose last activity is older than this is not resumed."""

RESUME_STAGGER_S = 2.0
"""Seconds between two relaunches at boot."""

LAUNCH_FILE = "launch.json"
WORKTREES_FILE = "worktrees.json"


def resume_dir(task_id: str) -> Path:
    """Return the resume directory of *task_id* (it may not exist)."""
    return kiss_home() / "resume" / task_id


def clear_resume_state(task_id: str) -> None:
    """Remove *task_id*'s resume directory; a missing one is fine."""
    if task_id:
        shutil.rmtree(resume_dir(task_id), ignore_errors=True)


def is_resumable(task_id: str) -> bool:
    """Whether a daemon would start *task_id* again after a restart.

    True while the task's resume directory holds its ``launch.json``,
    which only the daemon writes, for a top-level task.  A CLI run or a
    sub-agent never gets one: nothing restarts them, so they keep no
    checkpoints and their worktrees are ordinary orphans once their
    process is gone.
    """
    return bool(task_id) and (resume_dir(task_id) / LAUNCH_FILE).is_file()


def _write_json(path: Path, payload: Any) -> None:
    """Write *payload* to *path* atomically, readable by the owner only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    """Return the JSON in *path*, or ``None`` when it is missing or unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def launch_snapshot(cmd: dict[str, Any]) -> dict[str, Any]:
    """Return the part of a ``run`` command a relaunch needs.

    Private keys (``_``-prefixed, the runner's own bookkeeping and the
    hooks a SEA staged) are dropped, and so is any value that is not
    JSON: the relaunch goes through the same ``_run_task`` path, which
    applies the SEA again from ``seaPath``.

    Args:
        cmd: The command as the client submitted it.

    Returns:
        A JSON-serializable copy.
    """
    snapshot: dict[str, Any] = {}
    for key, value in cmd.items():
        if key.startswith("_"):
            continue
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        snapshot[key] = value
    return snapshot


def write_launch(task_id: str, cmd: dict[str, Any], *, resume_count: int = 0) -> None:
    """Record the command *task_id* was started with.

    Args:
        task_id: The task's ``task_history`` row id.
        cmd: The submitted ``run`` command (see :func:`launch_snapshot`).
        resume_count: How many times the task has been resumed so far.
    """
    _write_json(
        resume_dir(task_id) / LAUNCH_FILE,
        {
            "version": RESUME_VERSION,
            "task_id": task_id,
            "cmd": launch_snapshot(cmd),
            "resume_count": resume_count,
            "written": time.time(),
        },
    )


def read_launch(task_id: str) -> dict[str, Any] | None:
    """Return *task_id*'s ``launch.json``, or ``None`` when there is none."""
    launch = _read_json(resume_dir(task_id) / LAUNCH_FILE)
    return launch if isinstance(launch, dict) else None


def pending_task_ids() -> list[str]:
    """Return the ids of every task whose resume directory holds a ``launch.json``.

    Oldest launch first, so a daemon that comes back with several
    interrupted tasks restarts them in the order they were started.
    """
    root = kiss_home() / "resume"
    if not root.is_dir():
        return []
    found: list[tuple[float, str]] = []
    for entry in root.iterdir():
        launch = entry / LAUNCH_FILE
        if launch.is_file():
            found.append((launch.stat().st_mtime, entry.name))
    return [task_id for _mtime, task_id in sorted(found)]


def last_activity(task_id: str) -> float:
    """Return the newest mtime among *task_id*'s resume files (0.0 with none)."""
    directory = resume_dir(task_id)
    if not directory.is_dir():
        return 0.0
    return max((p.stat().st_mtime for p in directory.iterdir()), default=0.0)


def resume_refusal(task_id: str, launch: dict[str, Any], now: float) -> str:
    """Return why *task_id* must not be resumed, or ``""`` when it may be.

    Args:
        task_id: The task.
        launch: Its ``launch.json`` (see :func:`read_launch`).
        now: The current time, seconds since the epoch.
    """
    if launch.get("version") != RESUME_VERSION:
        return f"its resume state has format version {launch.get('version')!r}"
    if int(launch.get("resume_count", 0)) >= MAX_RESUMES:
        return f"it was stopped by {MAX_RESUMES} restarts in a row"
    idle = now - last_activity(task_id)
    if idle > MAX_RESUME_AGE_S:
        return f"its last activity was {idle / 3600:.1f} h ago"
    return ""


def write_worktrees(task_id: str, worktrees: list[dict[str, Any]]) -> None:
    """Record the git worktrees *task_id* holds, for reattachment on resume.

    Args:
        task_id: The task.
        worktrees: One ``{"repo_root", "branch", "original_branch",
            "wt_dir", "baseline_commit", "work_dir"}`` per worktree,
            every path a string.
    """
    _write_json(resume_dir(task_id) / WORKTREES_FILE, worktrees)


def read_worktrees(task_id: str) -> list[dict[str, Any]]:
    """Return the worktrees recorded by :func:`write_worktrees` (``[]`` with none)."""
    worktrees = _read_json(resume_dir(task_id) / WORKTREES_FILE)
    return worktrees if isinstance(worktrees, list) else []
