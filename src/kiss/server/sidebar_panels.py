# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Data behind the right sidebar's "Schedule", "Apps" and "Spend" subpanels.

The chat page's task-info panel stacks three global subpanels under
the per-task ones:

* **Schedule** lists the scheduled cron jobs
  (:mod:`kiss.agents.sorcar.cron_agent`), answered by
  :func:`cron_jobs_report` for the ``getCronJobs`` command;
* **Machines** lists the local machine and every registered remote
  machine (:mod:`kiss.agents.sorcar.machines`) with its reachability,
  answered by :func:`machines_report` for the ``getMachines`` command;
  its add / delete buttons send ``machineAction``, see
  :func:`machine_action`;
* **Apps** lists every third-party channel agent with its
  authentication state, answered by :func:`apps_status` for the
  ``getAppsStatus`` command;
* **Spend** draws a daily cost heatmap and cost-by-model bars from the
  task history, answered by :func:`spend_report` for the
  ``getSpendReport`` command;
* **Recommendations** lists the five next tasks the last
  ``/recommendations`` run suggested (a 6-hourly default cron job,
  :data:`kiss.agents.sorcar.cron_agent.DEFAULT_JOBS`), answered by
  :func:`recommendations_report` for the ``getRecommendations`` command.

The Schedule subpanel's edit / run / pause / delete buttons send
``cronAction``, applied by :func:`kiss.agents.sorcar.cron_agent.apply_job_action`.

Probing the apps imports every channel module and builds its agent, so
it runs in a short-lived subprocess
(``python -m kiss.agents.third_party_agents.auth_status``) and the
result is cached for :data:`APPS_STATUS_TTL_SECONDS`; concurrent
requests share one probe.  Probing the machines pings each one through
its cached daemon link (concurrently, one thread per machine) and the
result is cached for :data:`MACHINES_STATUS_TTL_SECONDS`.
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlsplit

from kiss.agents.seas.recommendations.recommendations_sea import load_recommendations
from kiss.agents.sorcar import cron_agent, machines
from kiss.agents.sorcar.persistence import _spend_by_day_and_model

logger = logging.getLogger(__name__)

APPS_STATUS_TTL_SECONDS = 30.0
MACHINES_STATUS_TTL_SECONDS = 60.0
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


_machines_lock = threading.Lock()
# The last ping of every registered machine: (machine, its row without
# the label, which is put on when the report is served, since a history
# fetch may have taught the daemon the host name meanwhile).
_machines_cache: list[tuple[machines.Machine, dict[str, Any]]] = []
_machines_checked_at = 0.0


def _resolve_ip(host: str) -> str:
    """Return the IPv4 address *host* resolves to, or *host* itself when it does not."""
    try:
        return socket.gethostbyname(host)
    except OSError:
        return host


def _machine_row(machine: machines.Machine) -> tuple[machines.Machine, dict[str, Any]]:
    """Ping one registered machine through its cached link and describe it.

    The ``pong`` names the machine's own host name
    (:func:`~kiss.agents.sorcar.machines.note_host`).

    Returns:
        The machine and its row: ``name`` (as registered), ``address``
        (as registered), ``ip`` (the host's address), ``port`` (the
        daemon port, 0 for an SSH machine registered before enrollment
        existed), ``local`` (False), ``status`` (``"reachable"`` or
        ``"unreachable"``) and ``error`` (the reason when unreachable).
        :func:`machines_report` adds ``label``.
    """
    link = machines.link_for(machine)
    try:
        pong = link.request({"type": "ping"}, "pong")
        status, error = "reachable", ""
        machines.note_host(machine, pong.get("machine"))
    except machines.MachineError as exc:
        status, error = "unreachable", str(exc)
    host = machines.machine_host(machine)
    port = (urlsplit(machine.url).port or machines.DEFAULT_PORT) if machine.url else 0
    return machine, {
        "name": machine.name, "address": machine.address(), "ip": _resolve_ip(host),
        "port": port, "local": False, "status": status, "error": error,
    }


def machines_report(
    local_ip: str, local_port: int, refresh: bool = False
) -> tuple[list[dict[str, Any]], int]:
    """Return the local machine and every registered machine with its status.

    The local row (first) describes this daemon at *local_ip*:*local_port*
    and is always reachable.  The registered rows are pinged in
    parallel (each through its cached :class:`~kiss.agents.sorcar.machines.MachineLink`,
    whose backoff keeps an offline machine cheap) and cached for
    :data:`MACHINES_STATUS_TTL_SECONDS`; *refresh* pings again, and
    so does a registry that changed since the cached ping (a machine
    a ``/machine add`` task or the CLI registered, which never goes
    through :func:`machine_action`).  Blocks for the slowest ping, so
    async callers run it in a thread.

    Args:
        local_ip: The address the local daemon is reached at.
        local_port: The local daemon's port.
        refresh: Ping again even when the cache is fresh.

    Returns:
        ``(rows, checked_at)``: the rows of :func:`_machine_row` after the
        local one, each with ``label`` (what to call the machine,
        :func:`~kiss.agents.sorcar.machines.machine_label`, as of now:
        the machine's own host name for a machine registered by its
        address), and the epoch-millisecond time of the ping they came
        from.
    """
    global _machines_cache, _machines_checked_at
    requested_at = time.time()
    with _machines_lock:
        registered = machines.load_machines()
        if (
            time.time() - _machines_checked_at >= MACHINES_STATUS_TTL_SECONDS
            or (refresh and _machines_checked_at < requested_at)
            or [machine for machine, _row in _machines_cache] != registered
        ):
            with ThreadPoolExecutor(max_workers=max(1, len(registered))) as pool:
                _machines_cache = list(pool.map(_machine_row, registered))
            _machines_checked_at = time.time()
        name = machines.local_machine_name()
        local = {
            "name": name, "label": name, "address": f"https://{local_ip}:{local_port}",
            "ip": local_ip, "port": local_port, "local": True, "status": "reachable", "error": "",
        }
        rows = [
            {**row, "label": machines.machine_label(machine)}
            for machine, row in _machines_cache
        ]
        return [local, *rows], _epoch_ms(_machines_checked_at)


def machine_action(action: str, name: str, address: str = "", password: str = "") -> str:
    """Apply a MACHINES subpanel button: ``add`` or ``remove`` a machine.

    Either action drops the cached report so the next
    :func:`machines_report` pings the new registry.

    Args:
        action: ``"add"`` (register *name* at *address*, with the
            daemon *password* unless it is an SSH address) or
            ``"remove"``.
        name: The machine name (``add``: empty defaults to the host).
        address: ``https://host:port`` or ``user@host[:port]`` (``add``).
        password: The remote daemon's password (``add``).

    Returns:
        ``""`` on success, else the error text for the user.
    """
    global _machines_checked_at
    try:
        if action == "add":
            machines.add_machine(name.strip(), address.strip(), password)
        elif action == "remove":
            if not machines.remove_machine(name.strip()):
                return f"no machine named {name.strip()!r}"
        else:
            return f"unknown action {action!r}"
    except machines.MachineError as exc:
        return str(exc)
    with _machines_lock:
        _machines_checked_at = 0.0
    return ""


def machine_work_dirs(name: str, work_dir: str = "") -> dict[str, Any]:
    """Return the working directory of registered machine *name* and its recent ones.

    The chat header's WORKDIR drop-down of a tab bound to a machine
    lists that machine's directories, not this daemon's: this asks the
    machine's daemon for its ``configData`` (``getConfig``) -- after
    making *work_dir*, when given, its global working directory
    (``setWorkDir`` with ``reply: true``, which answers with the same
    ``configData`` once adopted).  ``recent_work_dirs`` is the
    machine's own list: the directories that still exist there, most
    recently selected first.

    Args:
        name: The registered machine (its name or host).
        work_dir: A directory of the machine to adopt first; "" to
            only read.

    Returns:
        ``{"machine": name, "workDir": ..., "recentWorkDirs": [{"path",
        "ts"}, ...], "error": ""}``, or with ``error`` set (and the
        other fields empty) when the machine is not registered, is
        unreachable or refused.
    """
    reply: dict[str, Any] = {"machine": name, "workDir": "", "recentWorkDirs": [], "error": ""}
    machine = machines.find_machine(name)
    if machine is None:
        reply["error"] = f"no machine named {name!r}"
        return reply
    cmd: dict[str, Any] = {"type": "getConfig"}
    if work_dir:
        cmd = {"type": "setWorkDir", "workDir": work_dir, "reply": True}
    try:
        config = machines.link_for(machine).request(cmd, "configData").get("config") or {}
    except machines.MachineError as exc:
        reply["error"] = str(exc)
        return reply
    reply["workDir"] = str(config.get("work_dir") or "")
    reply["recentWorkDirs"] = [
        row for row in config.get("recent_work_dirs") or []
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    ]
    return reply


MACHINE_WORKSPACE_REPLIES: dict[str, str] = {
    "listDir": "dirListing",
    "gitStatus": "gitStatus",
    "gitLog": "gitLog",
    "gitShow": "gitShow",
    "gitAction": "gitActionResult",
    "fsAction": "fsResult",
    "openFile": "fileContent",
    "saveFile": "fileSaved",
    "checkPaths": "pathsExist",
}
"""The workspace commands a registered machine answers for a tab bound to it, by reply type.

The remote webapp's Explorer, Source Control view, folder picker, file
editor and transcript file links of a tab whose tasks run on a machine
browse that machine's working directory: each of these commands, when
it carries ``machine``, is forwarded to the machine's daemon unchanged
(:func:`machine_workspace_reply`) instead of being served here.
"""


def machine_workspace_reply(name: str, cmd: dict[str, Any]) -> dict[str, Any]:
    """Forward the workspace command *cmd* to registered machine *name* and return its reply.

    *cmd* is one of :data:`MACHINE_WORKSPACE_REPLIES`; it is sent to
    the machine's daemon as is, minus ``machine`` and this daemon's
    ``connId`` (its
    ``workDir``, ``path`` and ``tabId`` name a directory and a tab of
    the machine: the tab runs there under the same id, see
    :mod:`kiss.server.remote_tabs`), and the reply comes back stamped
    with the machine so the client can tell it from this daemon's.

    Returns:
        The machine's reply, or one of the expected type carrying the
        ``error`` (and the command's ``path``, ``token`` and ``tabId``)
        when the machine is not registered or unreachable.
    """
    reply_type = MACHINE_WORKSPACE_REPLIES[str(cmd["type"])]
    forwarded = {k: v for k, v in cmd.items() if k not in ("machine", "connId")}
    machine = machines.find_machine(name)
    try:
        if machine is None:
            raise machines.MachineError(f"no machine named {name!r}")
        reply = machines.link_for(machine).request(forwarded, reply_type)
    except machines.MachineError as exc:
        reply = {k: cmd[k] for k in ("path", "token", "tabId") if k in cmd}
        reply.update({"type": reply_type, "error": str(exc)})
        if reply_type == "dirListing":
            reply["entries"] = []
    reply["machine"] = name
    return reply


def recommendations_report() -> dict[str, Any]:
    """Return the RECOMMENDATIONS subpanel's content.

    The five suggested tasks the last ``/recommendations`` run stored
    (:func:`kiss.agents.seas.recommendations.recommendations_sea.load_recommendations`).

    Returns:
        ``{"generatedAt": int, "tasks": list[str]}``: when the run
        stored them (epoch milliseconds, ``0`` when none has yet) and
        its task sentences (empty until the first run).
    """
    stored = load_recommendations()
    return {"generatedAt": _epoch_ms(stored["generatedAt"]), "tasks": stored["tasks"]}
