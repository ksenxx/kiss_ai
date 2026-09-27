# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Data behind the right sidebar's "Schedule", "Apps" and "Spend" subpanels.

Every chat surface (the remote webapp's task-info panel, the VS Code
sidebar chat's drawer and the editor-tabs Task Info view) stacks three
global subpanels under the per-task ones:

* **Schedule** lists the scheduled cron jobs
  (:mod:`kiss.agents.sorcar.cron_agent`), answered by
  :func:`cron_jobs_report` for the ``getCronJobs`` command;
* **Apps** lists every third-party channel agent with its
  authentication state, answered by :func:`apps_status` for the
  ``getAppsStatus`` command;
* **Spend** draws a daily cost heatmap and cost-by-model bars from the
  task history, answered by :func:`spend_report` for the
  ``getSpendReport`` command.

Probing the apps imports every channel module and builds its agent, so
it runs in a short-lived subprocess
(``python -m kiss.agents.third_party_agents.auth_status``) and the
result is cached for :data:`APPS_STATUS_TTL_SECONDS`; concurrent
requests share one probe.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from typing import Any

from kiss.agents.sorcar import cron_agent
from kiss.agents.sorcar.persistence import _spend_by_day_and_model

logger = logging.getLogger(__name__)

APPS_STATUS_TTL_SECONDS = 30.0
# Upper bound for the probe subprocess (the probe gives up on slow
# channels after 20 s by itself).
_PROBE_PROCESS_TIMEOUT_SECONDS = 45.0

_apps_lock = threading.Lock()
_apps_cache: list[dict[str, Any]] = []
# When the cached statuses were probed, and when the last probe (which
# may have failed) finished: a failed probe also counts as fresh for
# the TTL, so callers queued behind it share its outcome instead of
# each starting another probe.
_apps_checked_at = 0.0
_apps_attempted_at = 0.0


def _epoch_ms(timestamp: Any) -> int:
    """Return *timestamp* (epoch seconds) in epoch milliseconds, 0 when unset.

    Times travel as epoch numbers, never zone-less local strings: the
    browser showing them may sit in another time zone than the daemon.
    """
    return int(float(timestamp or 0) * 1000)


def cron_jobs_report() -> list[dict[str, Any]]:
    """Return the scheduled cron jobs in display order.

    Enabled jobs come first, each group sorted by next run time (jobs
    with no next run last).

    Returns:
        One dict per job: ``id``, ``name``, ``schedule``, ``kind``
        (``"prompt"`` or ``"command"``), ``what`` (the prompt or command
        text), ``enabled``, ``running``, ``nextRunAt`` / ``lastRunAt``
        (epoch milliseconds, 0 when none), ``lastStatus`` and ``workDir``.
    """
    running = cron_agent.running_job_ids()
    rows = []
    for job in cron_agent.load_jobs():
        job_id = str(job.get("id") or "")
        is_command = bool(job.get("command"))
        rows.append({
            "id": job_id,
            "name": str(job.get("name") or job_id),
            "schedule": str(job.get("schedule") or ""),
            "kind": "command" if is_command else "prompt",
            "what": str(job.get("command") or job.get("prompt") or ""),
            "enabled": bool(job.get("enabled", True)),
            "running": job_id in running,
            "nextRunAt": _epoch_ms(job.get("next_run_at")),
            "lastRunAt": _epoch_ms(job.get("last_run_at")),
            "lastStatus": str(job.get("last_status") or ""),
            "workDir": str(job.get("work_dir") or ""),
            "_next": float(job.get("next_run_at") or 0) or float("inf"),
        })
    rows.sort(key=lambda row: (not row["enabled"], row["_next"], row["name"]))
    for row in rows:
        del row["_next"]
    return rows


def _add_spend(bucket: dict[str, Any], row: dict[str, Any]) -> None:
    """Add *row*'s ``cost``, ``tokens`` and ``tasks`` into *bucket*."""
    bucket["cost"] += row["cost"]
    bucket["tokens"] += row["tokens"]
    bucket["tasks"] += row["tasks"]


def _spend_bucket(**tags: str) -> dict[str, Any]:
    """Return a zeroed ``{cost, tokens, tasks}`` sum carrying *tags* (``date``, ``model``)."""
    return {**tags, "cost": 0.0, "tokens": 0, "tasks": 0}


def spend_report() -> dict[str, Any]:
    """Return the task history's spend, by day, by model and by day and model.

    Sums the persisted ``cost`` and ``tokens`` columns over the same
    row set the History sidebar lists (sub-agent rows excluded: their
    usage is already folded into their parent's totals), so the Spend
    subpanel's all-time line, daily heatmap and cost-by-model bars
    agree with each other and with the history.

    Returns:
        ``total``: all-time ``{cost, tokens, tasks}``;
        ``days``: ``{date, cost, tokens, tasks}`` per local calendar
        day with at least one task, in ascending date order;
        ``totalByModel``: ``{model, cost, tokens, tasks}`` per model, in
        descending cost order;
        ``daysByModel``: ``{"YYYY-MM-DD": [{model, cost, tokens,
        tasks}, ...]}``, each day's models in descending cost order.
    """
    total = _spend_bucket()
    days: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}
    days_by_model: dict[str, list[dict[str, Any]]] = {}
    for row in _spend_by_day_and_model():
        date, model = str(row["date"]), str(row["model"])
        _add_spend(total, row)
        _add_spend(days.setdefault(date, _spend_bucket(date=date)), row)
        _add_spend(by_model.setdefault(model, _spend_bucket(model=model)), row)
        day_models = days_by_model.setdefault(date, [])
        day_models.append({key: row[key] for key in ("model", "cost", "tokens", "tasks")})
    return {
        "total": total,
        "days": list(days.values()),
        "totalByModel": sorted(by_model.values(), key=lambda m: -m["cost"]),
        "daysByModel": days_by_model,
    }


def _probe_apps() -> list[dict[str, Any]]:
    """Run the auth-status probe subprocess and parse its JSON answer.

    Returns:
        The probe's status list; empty when the probe failed.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "kiss.agents.third_party_agents.auth_status"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_PROBE_PROCESS_TIMEOUT_SECONDS,
            check=False,
        )
        data = json.loads(proc.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        logger.warning("apps auth-status probe failed", exc_info=True)
        return []
    return data if isinstance(data, list) else []


def apps_status(refresh: bool = False) -> tuple[list[dict[str, Any]], int]:
    """Return every channel agent's authentication status.

    Serves the cached probe result while the last probe (successful or
    not) is younger than :data:`APPS_STATUS_TTL_SECONDS`; otherwise (or
    with *refresh*) runs a new probe.  Concurrent callers wait for the
    same probe.  Blocks for the probe's duration, so async callers run
    it in a thread.

    Args:
        refresh: Probe again even when the cache is fresh.

    Returns:
        ``(apps, checked_at)``: the status dicts (see
        :func:`kiss.agents.third_party_agents.auth_status.channel_status`)
        and the epoch-millisecond time of the probe they came from (0
        before the first successful probe).
    """
    global _apps_cache, _apps_checked_at, _apps_attempted_at
    requested_at = time.time()
    with _apps_lock:
        age = time.time() - _apps_attempted_at
        # A probe that finished while this caller waited for the lock
        # already answers a refresh request.
        if age >= APPS_STATUS_TTL_SECONDS or (refresh and _apps_attempted_at < requested_at):
            apps = _probe_apps()
            _apps_attempted_at = time.time()
            if apps:
                _apps_cache = apps
                _apps_checked_at = _apps_attempted_at
        return list(_apps_cache), _epoch_ms(_apps_checked_at)
