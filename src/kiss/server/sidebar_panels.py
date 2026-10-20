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

Probing the apps imports every channel module and builds its agent, so
it runs in a short-lived subprocess
(``python -m kiss.agents.third_party_agents.auth_status``) and the
result is cached for :data:`APPS_STATUS_TTL_SECONDS`; concurrent
requests share one probe.  Probing the machines pings each one through
its cached daemon link (concurrently, one thread per machine) and the
result is cached for :data:`MACHINES_STATUS_TTL_SECONDS`.
"""

from __future__ import annotations

import logging
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
# Longest a machine row waits for its host name to resolve (:func:`_resolve_ip`).
DNS_TIMEOUT_SECONDS = 2.0
# Seconds ``codex login --device-auth`` may take to print its code
# (:func:`kiss.core.models.codex_model.codex_device_login`'s default
# *timeout*): a forwarded ``startCodexLogin`` is waited for that long.
_CODEX_LOGIN_TIMEOUT_SECONDS = 60.0

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
    refused: dict[str, Any] = {k: cmd[k] for k in ("path", "token", "tabId") if k in cmd}
    if reply_type == "dirListing":
        refused["entries"] = []
    return _machine_reply(name, cmd, reply_type, refused)


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
    # up to _CODEX_LOGIN_TIMEOUT_SECONDS later: its reply is waited for
    # that long, on a link lane of its own so the sheet's other requests
    # (and a submit to the machine) do not queue behind it.
    timeout, lane = machines.REQUEST_TIMEOUT, ""
    if reply_type == "codexLogin":
        timeout, lane = timeout + _CODEX_LOGIN_TIMEOUT_SECONDS, "login"
    reply = _machine_reply(name, cmd, reply_type, {}, errors=True, timeout=timeout, lane=lane)
    reply["forwarded"] = True
    return reply


MACHINE_PANEL_REPLIES: dict[str, str] = {
    "getCronJobs": "cronJobs",
    "cronAction": "cronJobs",
    "getAppsStatus": "appsStatus",
    "getSpendReport": "spendReport",
    "getRecommendations": "recommendations",
}
"""The sidebar subpanel commands a registered machine answers, by reply type.

The remote webapp's Schedule, Apps, Spend and Recommendations
subpanels show the machine the chat header's drop-down has picked:
each of these commands, when it carries ``machine``, is forwarded to
that machine's daemon (:func:`machine_panel_reply`), so the subpanels
list (and the Schedule one edits) that machine's cron jobs, app
sign-ins, spend and recommendations instead of this daemon's.  The
Machines subpanel is this daemon's registry and is never forwarded.
"""

_PANEL_EMPTY_FIELDS: dict[str, dict[str, Any]] = {
    "cronJobs": {"jobs": []},
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
        (no jobs, apps or tasks; a ``cronAction``'s ``action`` kept)
        carrying the ``error`` when the machine is not registered,
        unreachable, or refused the command with an ``error`` frame.
    """
    reply_type = MACHINE_PANEL_REPLIES[str(cmd["type"])]
    refused = dict(_PANEL_EMPTY_FIELDS[reply_type])
    if cmd["type"] == "cronAction":
        refused["action"] = str(cmd.get("action") or "")
    # The machine's app probe is a subprocess of up to
    # PROBE_PROCESS_TIMEOUT_SECONDS: its reply is waited for that long,
    # not the link's usual few seconds (a timed-out reply would be lost,
    # and the panel shown "did not answer" for a probe that succeeded).
    # The subpanels' polls travel on a link lane of their own, so that
    # wait holds up no submit, settings or Explorer request.
    timeout = machines.REQUEST_TIMEOUT
    if reply_type == "appsStatus":
        timeout += PROBE_PROCESS_TIMEOUT_SECONDS
    reply = _machine_reply(
        name, cmd, reply_type, refused, errors=True, timeout=timeout, lane="panels",
    )
    reply["forwarded"] = True
    return reply


MACHINE_FORWARDERS: dict[str, Callable[[str, dict[str, Any]], dict[str, Any]]] = {
    **dict.fromkeys(MACHINE_WORKSPACE_REPLIES, machine_workspace_reply),
    **dict.fromkeys(MACHINE_SETTINGS_REPLIES, machine_settings_reply),
    **dict.fromkeys(MACHINE_PANEL_REPLIES, machine_panel_reply),
}
"""Every command a registered machine answers when it carries ``machine``, by its forwarder."""

MACHINE_COMMIT_TIMEOUT = 900.0
"""Seconds a machine gets to stage, write a commit message with an LLM and commit.

Beyond what one ``git`` call may take there (300 s), the LLM call and
a wait for the repository's lock: a timeout reports a failure here
while the machine may still commit, so it must not come first.
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
    *, errors: bool = False, timeout: float = machines.REQUEST_TIMEOUT, lane: str = "",
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
        or unreachable; either stamped ``machine``.
    """
    forwarded = {k: v for k, v in cmd.items() if k not in ("machine", "connId")}
    machine = machines.find_machine(name)
    try:
        if machine is None:
            raise machines.MachineError(f"no machine named {name!r}")
        reply = machines.link_for(machine, lane).request(
            forwarded, reply_type, timeout, errors=errors,
        )
        if cmd["type"] == "saveConfig":
            _adopt_machine_password(machine, reply)
    except machines.MachineError as exc:
        reply = {**refused, "type": reply_type, "error": str(exc)}
    reply["machine"] = name
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
