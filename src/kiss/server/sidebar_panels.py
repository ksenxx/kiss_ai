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
* **Recommendations** lists the ten next tasks the last
  ``/recommendations`` run suggested (a 6-hourly default cron job,
  :data:`kiss.agents.sorcar.cron_agent.DEFAULT_JOBS`), answered by
  :func:`recommendations_report` for the ``getRecommendations`` command.

The Schedule subpanel's edit / run / pause / delete buttons send
``cronAction``, applied by :func:`kiss.agents.sorcar.cron_agent.apply_job_action`.

Every subpanel shows the machine the chat header's drop-down has
picked: a command carrying ``machine`` is answered by that machine's
daemon instead (:data:`MACHINE_FORWARDERS`).

Probing the apps imports every channel module and builds its agent, so
it runs in a short-lived subprocess
(``python -m kiss.agents.third_party_agents.auth_status``) and the
result is cached for :data:`APPS_STATUS_TTL_SECONDS`; concurrent
requests share one probe.  Probing the machines pings each one through
its cached daemon link (concurrently, one thread per machine) and the
result is cached for :data:`MACHINES_STATUS_TTL_SECONDS` (for
:data:`MACHINES_DOWN_TTL_SECONDS` while a machine is unreachable).
"""

from __future__ import annotations

import logging
import math
import socket
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlsplit

from kiss.agents.seas.recommendations.recommendations_sea import load_recommendations
from kiss.agents.sorcar import cron_agent, machines
from kiss.agents.sorcar.persistence import _spend_by_day_and_model
from kiss.agents.third_party_agents.auth_status import (
    PROBE_PROCESS_TIMEOUT_SECONDS,
    probe_statuses,
)

logger = logging.getLogger(__name__)

APPS_STATUS_TTL_SECONDS = 30.0
MACHINES_STATUS_TTL_SECONDS = 60.0
# How long a report with an unreachable machine is served before the
# machines are pinged again: a laptop back from sleep has its machines
# within seconds of the network returning, and the link's back-off
# keeps a machine that is really down cheap to ask again.
MACHINES_DOWN_TTL_SECONDS = 10.0
# Longest a machine row waits for its host name to resolve (:func:`_resolve_ip`).
DNS_TIMEOUT_SECONDS = 2.0
# Seconds ``codex login --device-auth`` may take to print its code
# (:func:`kiss.core.models.codex_model.codex_device_login`'s default
# *timeout*): a forwarded ``startCodexLogin`` is waited for that long.
_CODEX_LOGIN_TIMEOUT_SECONDS = 60.0
# Seconds ``codex login status`` may take
# (:func:`kiss.core.models.codex_model.codex_auth_status`'s default
# *timeout*; ``claude auth status`` gets 20): a forwarded
# ``getCodexStatus`` / ``getClaudeCodeStatus`` is waited for that long.
_CLI_STATUS_TIMEOUT_SECONDS = 30.0
# Seconds a machine may take, beyond a request's usual budget, before
# its ``machines`` reply: it pings the machines it registered (an SSH
# one opens its port-forward first, every host name is looked up) and
# an ``add`` first enrolls the new machine.
_MACHINES_REPLY_SECONDS = (
    machines.SSH_TIMEOUT + machines.ENROLL_TIMEOUT + DNS_TIMEOUT_SECONDS
)

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
        })
    rows.sort(key=_cron_row_order)
    return rows


def _cron_row_order(row: dict[str, Any]) -> tuple[bool, float, str]:
    """Sort key of :func:`cron_jobs_report`: enabled first, soonest next run first, then by name."""
    return (not row["enabled"], row["nextRunAt"] or float("inf"), row["name"])


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
            apps = probe_statuses()
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
    """Return the IPv4 address *host* resolves to, or *host* itself when it does not.

    ``gethostbyname`` has no timeout of its own: a resolver that drops
    packets holds it for libc's retry schedule (tens of seconds), and
    :func:`_machine_row` runs under ``_machines_lock``, so every
    Machines subpanel would wait on that one name.  The lookup runs on
    a daemon thread this call waits for at most
    :data:`DNS_TIMEOUT_SECONDS`; one that overruns is abandoned and the
    host name shown instead.
    """
    found: list[str] = []
    thread = threading.Thread(
        target=_lookup_ip, args=(host, found), name="machine-dns", daemon=True,
    )
    thread.start()
    thread.join(DNS_TIMEOUT_SECONDS)
    return found[0] if found else host


def _lookup_ip(host: str, found: list[str]) -> None:
    """Append the IPv4 address of *host* to *found*; leave it empty when the lookup fails."""
    try:
        found.append(socket.gethostbyname(host))
    except OSError:
        pass


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
    :data:`MACHINES_STATUS_TTL_SECONDS`, or :data:`MACHINES_DOWN_TTL_SECONDS`
    when a machine was unreachable; *refresh* pings again, and
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
        ttl = (
            MACHINES_DOWN_TTL_SECONDS
            if any(row["status"] == "unreachable" for _machine, row in _machines_cache)
            else MACHINES_STATUS_TTL_SECONDS
        )
        if (
            time.time() - _machines_checked_at >= ttl
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
    recently selected first.  ``maxBudget`` is the machine's configured
    default task budget (``config.max_budget``): what a task submitted
    to the machine without a budget of its own runs under, which the
    task-info panel shows for a chat headed there before its task has
    settings; ``None`` when the machine's config names no finite one.

    Args:
        name: The registered machine (its name or host).
        work_dir: A directory of the machine to adopt first; "" to
            only read.

    Returns:
        ``{"machine": name, "workDir": ..., "recentWorkDirs": [{"path",
        "ts"}, ...], "maxBudget": <number or None>, "error": ""}``, or
        with ``error`` set (and the other fields empty) when the
        machine is not registered, is unreachable or refused.
    """
    reply: dict[str, Any] = {
        "machine": name, "workDir": "", "recentWorkDirs": [], "maxBudget": None, "error": "",
    }
    cmd: dict[str, Any] = {"type": "getConfig"}
    if work_dir:
        cmd = {"type": "setWorkDir", "workDir": work_dir, "reply": True}
    answer = _machine_reply(name, cmd, "configData", {})
    if "error" in answer:
        reply["error"] = answer["error"]
        return reply
    config = answer.get("config") or {}
    reply["workDir"] = str(config.get("work_dir") or "")
    reply["recentWorkDirs"] = [
        row for row in config.get("recent_work_dirs") or []
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    ]
    budget = config.get("max_budget")
    if isinstance(budget, (int, float)) and not isinstance(budget, bool) and math.isfinite(budget):
        reply["maxBudget"] = budget
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
    refused: dict[str, Any] = {k: cmd[k] for k in ("path", "token", "tabId") if k in cmd}
    if reply_type == "dirListing":
        refused["entries"] = []
    # A Git action waits there for the repository's lock (an auto-commit
    # may hold it for minutes) and then runs git and its hooks, and a
    # file action pastes, deletes or searches as much as the folder
    # holds: their replies are waited for MACHINE_COMMIT_TIMEOUT (a
    # timed-out reply would be lost, the page told "did not answer" while
    # the action completes there, and the machine backed off as down), on
    # a link lane of their own so the Explorer's other requests (and a
    # submit to the machine) do not queue behind them.
    timeout, lane = machines.REQUEST_TIMEOUT, ""
    if reply_type in ("gitActionResult", "fsResult"):
        timeout, lane = MACHINE_COMMIT_TIMEOUT, "actions"
    return _machine_reply(name, cmd, reply_type, refused, timeout=timeout, lane=lane)


MACHINE_SETTINGS_REPLIES: dict[str, str] = {
    "getConfig": "configData",
    "saveConfig": "configData",
    "getMyModels": "myModelsData",
    "getClaudeCodeStatus": "claudeCodeStatus",
    "getCodexStatus": "codexStatus",
    "startCodexLogin": "codexLogin",
    "saveMyModel": "myModelsData",
    "deleteMyModel": "myModelsData",
}
"""The settings commands a registered machine answers for the settings sheet, by reply type.

The remote webapp's settings sheet opens for the machine the chat
header shows: each of these commands, when it carries ``machine``, is
forwarded to that machine's daemon (:func:`machine_settings_reply`),
so the form shows and saves that machine's config, API keys and custom
models instead of this daemon's.
"""


def machine_settings_reply(name: str, cmd: dict[str, Any]) -> dict[str, Any]:
    """Forward the settings command *cmd* to registered machine *name* and return its reply.

    *cmd* is one of :data:`MACHINE_SETTINGS_REPLIES`; it is sent to the
    machine's daemon minus ``machine`` and this daemon's ``connId``.
    The reply comes back stamped ``machine`` and ``forwarded`` so the
    client can tell it from this daemon's own ``configData`` (whose
    ``machine`` is this daemon's host name).

    Returns:
        The machine's reply, or one of the expected type carrying the
        ``error`` when the machine is not registered, unreachable, or
        rejected a save (its daemon then answers with an ``error``
        frame instead, which the link does not wait for).
    """
    reply_type = MACHINE_SETTINGS_REPLIES[str(cmd["type"])]
    # The machine's device login answers once its CLI printed the code,
    # up to _CODEX_LOGIN_TIMEOUT_SECONDS later, and its CLI status probes
    # once the CLI answered, up to _CLI_STATUS_TIMEOUT_SECONDS later:
    # their replies are waited for that long, each kind on a link lane of
    # its own so the sheet's other requests (and a submit to the machine)
    # do not queue behind them.
    timeout, lane = machines.REQUEST_TIMEOUT, ""
    if reply_type == "codexLogin":
        timeout, lane = timeout + _CODEX_LOGIN_TIMEOUT_SECONDS, "login"
    elif reply_type in ("claudeCodeStatus", "codexStatus"):
        timeout, lane = timeout + _CLI_STATUS_TIMEOUT_SECONDS, "status"
    return _machine_reply(
        name, cmd, reply_type, {}, errors=True, timeout=timeout, lane=lane, forwarded=True,
    )


MACHINE_PANEL_REPLIES: dict[str, str] = {
    "getCronJobs": "cronJobs",
    "cronAction": "cronJobs",
    "getMachines": "machines",
    "machineAction": "machines",
    "getAppsStatus": "appsStatus",
    "getSpendReport": "spendReport",
    "getRecommendations": "recommendations",
}
"""The sidebar subpanel commands a registered machine answers, by reply type.

The remote webapp's Schedule, Machines, Apps, Spend and
Recommendations subpanels show the machine the chat header's
drop-down has picked: each of these commands, when it carries
``machine``, is forwarded to that machine's daemon
(:func:`machine_panel_reply`), so the subpanels list (and the Schedule
and Machines ones edit) that machine's cron jobs, registered machines,
app sign-ins, spend and recommendations instead of this daemon's.
"""

_PANEL_EMPTY_FIELDS: dict[str, dict[str, Any]] = {
    "cronJobs": {"jobs": []},
    "machines": {"machines": [], "checkedAt": 0},
    "appsStatus": {"apps": [], "checkedAt": 0},
    "spendReport": {},
    "recommendations": {"generatedAt": 0, "tasks": []},
}


def machine_panel_reply(name: str, cmd: dict[str, Any]) -> dict[str, Any]:
    """Forward the subpanel command *cmd* to registered machine *name* and return its reply.

    *cmd* is one of :data:`MACHINE_PANEL_REPLIES`; it is sent to the
    machine's daemon minus ``machine`` and this daemon's ``connId``.
    The reply comes back stamped ``machine`` and ``forwarded`` so the
    client can tell it from this daemon's own, and drop one for a
    machine it no longer shows.

    Returns:
        The machine's reply, or an empty one of the expected type
        (no jobs, machines, apps or tasks) carrying the ``error`` when
        the machine is not registered, unreachable, or refused the
        command with an ``error`` frame.  Either names a
        ``cronAction``'s ``action`` and ``jobId``, or a
        ``machineAction``'s ``action`` and ``name``, as this daemon's
        own replies do.
    """
    reply_type = MACHINE_PANEL_REPLIES[str(cmd["type"])]
    echo: dict[str, str] = {}
    if cmd["type"] == "cronAction":
        echo = {"action": str(cmd.get("action") or ""), "jobId": str(cmd.get("jobId") or "")}
    elif cmd["type"] == "machineAction":
        echo = {"action": str(cmd.get("action") or ""), "name": str(cmd.get("name") or "")}
    refused = {**_PANEL_EMPTY_FIELDS[reply_type], **echo}
    # The machine's app probe is a subprocess of up to
    # PROBE_PROCESS_TIMEOUT_SECONDS, and its machines reply waits for its
    # own pings: the reply is waited for that long, not the link's usual
    # few seconds (a timed-out reply would be lost, the panel shown "did
    # not answer" and the machine backed off as down).  The subpanels'
    # polls travel on a link lane of their own, so that wait holds up no
    # submit, settings or Explorer request.
    timeout = machines.REQUEST_TIMEOUT
    if reply_type == "appsStatus":
        timeout += PROBE_PROCESS_TIMEOUT_SECONDS
    elif reply_type == "machines":
        timeout += _MACHINES_REPLY_SECONDS
    reply = _machine_reply(
        name, cmd, reply_type, refused,
        errors=True, timeout=timeout, lane="panels", forwarded=True,
    )
    # The machine's reply may not name them: an older daemon's, or the
    # changed registry a machine pushes to every connection after a
    # ``machineAction`` (this link's too), which may come before the
    # action's own reply and is the action's outcome all the same.
    reply.update(echo)
    return reply


def machine_task_update_reply(name: str, cmd: dict[str, Any]) -> dict[str, Any]:
    """Forward the task-info panel's ``getTaskUpdate`` *cmd* to registered machine *name*.

    The panel's Task update subpanel polls for the update of the task
    running in the visible tab.  A tab bound to a machine
    (:mod:`kiss.server.remote_tabs`) runs its tasks there under the
    same tab id, so the machine's daemon owns the update (it runs the
    update agent, holds the report and answers the poll's
    ``knownSig``/``refresh`` as it would for its own chat page); the
    command is sent to it minus ``machine`` and this daemon's
    ``connId``, on the ``panels`` link lane the sidebar subpanels' polls
    use, so a submit to the machine never waits behind it.  The reply
    comes back stamped ``machine``.

    Returns:
        The machine's ``taskUpdate``, or an ``exists: false`` one (the
        subpanel hides, as for a tab without a running task) carrying
        the command's ``tabId`` and ``token`` and the ``error`` when the
        machine is not registered or unreachable.
    """
    refused: dict[str, Any] = {k: cmd[k] for k in ("tabId", "token") if k in cmd}
    refused.update({"taskId": "", "exists": False, "sig": "", "content": ""})
    return _machine_reply(name, cmd, "taskUpdate", refused, lane="panels")


FILES_INDEX_WAIT = 60.0
"""Seconds a forwarded ``getFiles`` keeps asking while the machine indexes the folder."""

_FILES_POLL_INTERVALS = (0.2, 2.0)
"""First and longest pause between two asks of a forwarded ``getFiles`` (doubling)."""


def machine_files_reply(name: str, cmd: dict[str, Any]) -> dict[str, Any]:
    """Forward the ``@``-mention picker's ``getFiles`` to registered machine *name*.

    A tab whose tasks run on a machine mentions that machine's files,
    so its picker lists them, not this daemon's.  The machine answers a
    folder it has never indexed with an empty ``loading`` list at once
    and sends the real list when its scan is done; a link request reads
    one reply, so the command is asked again, at growing intervals,
    until the list is ready or :data:`FILES_INDEX_WAIT` seconds have
    passed (the page asks again on the ``loading`` reply it then
    gets).  A list echoing another ``prefix`` or ``tabId`` is asked
    past as well: it answers an earlier request on the shared link
    (the list its scan sent after the request had returned
    ``loading``), and the page would drop it.

    Returns:
        The machine's ``files`` reply stamped ``machine``, or an empty
        one carrying the ``error`` (and the command's ``prefix`` and
        ``tabId``, which the page matches the reply with) when the
        machine is not registered or unreachable.
    """
    refused: dict[str, Any] = {k: cmd[k] for k in ("prefix", "tabId") if k in cmd}
    refused["files"] = []
    deadline = time.monotonic() + FILES_INDEX_WAIT
    pause, longest = _FILES_POLL_INTERVALS
    while True:
        reply = _machine_reply(name, cmd, "files", refused)
        late = any(k in reply and reply[k] != cmd.get(k) for k in ("prefix", "tabId"))
        if not (reply.get("loading") or late) or time.monotonic() >= deadline:
            return reply
        time.sleep(pause)
        pause = min(pause * 2, longest)


def machine_file_usage(name: str, cmd: dict[str, Any]) -> None:
    """Forward ``recordFileUsage`` to registered machine *name*.

    The machine counts the mention picked for it, so its picker ranks
    the file ``frequent``.  The command has no reply; the count is lost
    when the machine is not registered or unreachable.
    """
    reply = _machine_reply(name, cmd, "", {})
    if "error" in reply:
        logger.info("mention usage not counted on %s: %s", name, reply["error"])


def machine_models_reply(name: str, cmd: dict[str, Any]) -> dict[str, Any] | None:
    """Forward the model picker's ``getModels`` / ``selectModel`` *cmd* to machine *name*.

    The chat input's model picker is the picker of the machine the chat
    header's drop-down has picked: ``getModels`` brings that machine's
    own model list and default model, and ``selectModel`` makes the
    model the machine's default (and the tab's there), leaving this
    daemon's default alone.  The command is sent minus ``machine`` and
    this daemon's ``connId``.

    Returns:
        The machine's ``models`` reply for ``getModels``, ``None`` for
        ``selectModel`` (it has no reply), or an empty ``models`` reply
        carrying the ``error`` when the machine is not registered or
        unreachable; a reply is stamped ``machine`` and ``forwarded``
        and echoes the command's ``generation`` (0 when it had none).
    """
    reply_type = "models" if cmd["type"] == "getModels" else ""
    reply = _machine_reply(
        name, cmd, reply_type, {"models": [], "selected": ""}, errors=True, forwarded=True,
    )
    if not reply_type and "error" not in reply:
        return None
    reply["type"] = "models"
    # The link stamped the command with a generation of its own, which
    # the machine echoed: the page is answered with the one it sent.
    reply["generation"] = cmd.get("generation", 0)
    return reply


MACHINE_FORWARDERS: dict[str, Callable[[str, dict[str, Any]], dict[str, Any] | None]] = {
    **dict.fromkeys(MACHINE_WORKSPACE_REPLIES, machine_workspace_reply),
    **dict.fromkeys(MACHINE_SETTINGS_REPLIES, machine_settings_reply),
    **dict.fromkeys(MACHINE_PANEL_REPLIES, machine_panel_reply),
    "getTaskUpdate": machine_task_update_reply,
    "getFiles": machine_files_reply,
    "recordFileUsage": machine_file_usage,
    "getModels": machine_models_reply,
    "selectModel": machine_models_reply,
}
"""Every command a registered machine answers when it carries ``machine``, by its forwarder.

A forwarder returning ``None`` (``recordFileUsage``, a delivered
``selectModel``) sends no reply."""

MACHINE_COMMIT_TIMEOUT = 900.0
"""Seconds a machine gets to stage, write a commit message with an LLM and commit.

Beyond what one ``git`` call may take there (300 s), the LLM call and
a wait for the repository's lock: a timeout reports a failure here
while the machine may still commit, so it must not come first.  A
forwarded Git or file action (:func:`machine_workspace_reply`), which
may wait for that lock or copy a whole folder, gets as long.
"""


def machine_autocommit(name: str, cmd: dict[str, Any]) -> dict[str, Any]:
    """Run the Git Commit button's ``autocommitAction`` *cmd* on registered machine *name*.

    The machine's daemon commits the folder *cmd* names (its own
    configured folder when ``workDir`` is empty) and answers the link
    with the tab's ``autocommit_done`` (a link peer gets its own copy
    of the event its browsers get by broadcast; see
    ``ServerApi.autocommit_action``).  The exchange has its own
    ``commit`` lane so the minutes an LLM commit message may take do
    not hold up the machine's file browsing.

    Returns:
        The machine's ``autocommit_done``, or one with ``success``
        false and the ``error`` when the machine is not registered,
        unreachable or silent; either stamped ``machine``.
    """
    refused = {"tabId": cmd.get("tabId"), "success": False, "committed": False}
    return _machine_reply(
        name, cmd, "autocommit_done", refused, timeout=MACHINE_COMMIT_TIMEOUT, lane="commit",
    )


def _machine_reply(
    name: str, cmd: dict[str, Any], reply_type: str, refused: dict[str, Any],
    *,
    errors: bool = False,
    timeout: float = machines.REQUEST_TIMEOUT,
    lane: str = "",
    forwarded: bool = False,
) -> dict[str, Any]:
    """Forward *cmd* to registered machine *name* and return its *reply_type* reply.

    The command goes to the machine's daemon minus ``machine`` and this
    daemon's ``connId`` over the machine's *lane* link
    (:func:`~kiss.agents.sorcar.machines.link_for`); *errors* and
    *timeout* are passed to
    :meth:`~kiss.agents.sorcar.machines.MachineLink.request`.  A
    ``saveConfig`` reply re-registers the machine with the remote
    password it reports (:func:`_adopt_machine_password`).

    Returns:
        The machine's reply, or *refused* completed with the
        ``type`` and the ``error`` when the machine is not registered
        or unreachable; either stamped ``machine`` (and ``forwarded``
        when *forwarded*, for the panels that show where it came from).
    """
    payload = {k: v for k, v in cmd.items() if k not in ("machine", "connId")}
    machine = machines.find_machine(name)
    try:
        if machine is None:
            raise machines.MachineError(f"no machine named {name!r}")
        reply = machines.link_for(machine, lane).request(
            payload, reply_type, timeout, errors=errors,
        )
        if cmd["type"] == "saveConfig":
            _adopt_machine_password(machine, reply)
    except machines.MachineError as exc:
        reply = {**refused, "type": reply_type, "error": str(exc)}
    reply["machine"] = name
    if forwarded:
        reply["forwarded"] = True
    return reply


def _adopt_machine_password(machine: machines.Machine, reply: dict[str, Any]) -> None:
    """Re-register *machine* with the remote password its ``saveConfig`` *reply* reports.

    A password-authenticated machine whose settings sheet changed its
    remote password would reject this daemon's next frame: the registry
    entry takes the new password (which also drops the cached link, so
    the next request authenticates with it).  An SSH machine (no
    password here) and an unchanged or emptied password leave the
    registry alone.
    """
    config = reply.get("config")
    password = config.get("remote_password") if isinstance(config, dict) else None
    if not machine.password or not isinstance(password, str) or not password:
        return
    if password == machine.password:
        return
    with machines.registry_lock():
        registry = machines.load_machines()
        for entry in registry:
            if entry.name == machine.name:
                entry.password = password
        machines.save_machines(registry)


def recommendations_report() -> dict[str, Any]:
    """Return the RECOMMENDATIONS subpanel's content.

    The ten suggested tasks the last ``/recommendations`` run stored
    (:func:`kiss.agents.seas.recommendations.recommendations_sea.load_recommendations`).

    Returns:
        ``{"generatedAt": int, "tasks": list[str]}``: when the run
        stored them (epoch milliseconds, ``0`` when none has yet) and
        its task sentences (empty until the first run).
    """
    stored = load_recommendations()
    return {"generatedAt": _epoch_ms(stored["generatedAt"]), "tasks": stored["tasks"]}
