# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The Sorcar server API and a minimal synchronous client for it.

This module is the single source of truth for the wire API of the
``sorcar web`` daemon and hosts both of its Python ends:

**The server API** — :data:`API`, :func:`validate_command`, and
:class:`ServerApi` define every command a user interface (a VS Code
window, the remote webapp, or a CLI/Python client) may send to the
daemon.  Both transports speak the same JSON commands, dispatched on
the ``"type"`` field — framed as newline-delimited lines on the
Unix-domain socket (UDS) and as one object per WebSocket frame on
WSS.  The daemon routes every command through
:meth:`ServerApi.dispatch`, which validates it against the catalog
(answering an invalid one with an ``{"type": "error", "text": ...}``
event instead of processing it) and invokes the :class:`ServerApi`
method the command's catalog entry names.  The only exception is a
WSS connection's pre-dispatch ``auth`` handshake, serviced by
:meth:`ServerApi.authenticate`.  The user interfaces consume the
catalog through thin client facades — ``media/api.js`` (chat webview
and remote webapp) and ``src/SorcarApi.ts`` (VS Code extension host)
— whose methods map 1:1 onto the catalog's command names; the remote
webapp's bootstrap shim (``_WS_SHIM_JS`` in
:mod:`kiss.server.web_server`) additionally sends the ``auth``
handshake and the reconnect ``setWorkDir`` re-pin, both catalog
commands.

**The client API** — :func:`run` lets any process launch a task on an
already-running daemon and block until it finishes::

    from kiss.server import sorcar

    result = sorcar.run("Summarize README.md", work_dir="/path/to/repo")
    print(result.text, result.success, result.cost, result.tokens, result.steps)
    print(result.chat_id, result.task_id)  # daemon chat session / task row ids

    # Continue the same chat (the agent sees the prior task as context):
    follow_up = sorcar.run("Now fix the typos you found", chat_id=result.chat_id)

Caller-supplied tools become agent tools: pass the path of a Python
file via ``tools="/path/to/my_tools.py"`` and the daemon imports the
file and registers every top-level public function that is suitable as
a tool (plain synchronous functions with keyword-bindable,
type-annotated parameters and Google-style docstrings).  The client
never serializes Python functions — the daemon loads the file itself,
so the tools execute **in the daemon process** like native agent
tools::

    # my_tools.py
    def get_temperature(city: str) -> str:
        \"\"\"Return the current temperature of a city.

        Args:
            city: Name of the city to look up.
        \"\"\"
        return lookup_sensor(city)

    result = sorcar.run("What's the temperature in Paris?",
                        tools="my_tools.py")

The function speaks the daemon's newline-delimited JSON protocol over
its Unix-domain socket (``$KISS_SORCAR_SOCK``, defaulting to
``$KISS_HOME/sorcar.sock``) — the same transport the VS Code extension
and the CLI client use — so no HTTP server, password, or extra
dependency is involved.  POSIX file permissions (mode 0o600) on the
socket restrict access to the owning user.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import secrets
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from kiss.agents.sorcar.persistence import _default_kiss_dir
from kiss.core.vscode_config import load_config
from kiss.server.tools_file import resolve_tools_file

logger = logging.getLogger(__name__)

_MAX_LINE_BYTES = 64 * 1024 * 1024
"""Read buffer limit for a single daemon event line.

The daemon emits large single-line JSON events (e.g.
``system_prompt`` carrying the full SYSTEM.md), so this MUST match the
daemon-side transport frame limit (``web_server._MAX_LINE_BYTES``, 64
MiB).  A smaller client cap would split an oversized newline-delimited
frame; each fragment is then discarded as invalid JSON, and when the
oversized frame is the terminal ``result`` event the client would
return an empty unsuccessful :class:`TaskResult` for a task that
actually succeeded.
"""


def _job_dir_is_contained(
    job_dir: Path, discovered: dict[str, Path],
) -> bool:
    """Return whether *job_dir* resolves inside a recognized job root.

    ``discover_job_dirs`` follows directory symlinks, so a
    ``jobs/job_link`` entry pointing outside every ``.kiss.artifacts/jobs``
    root would still appear in its result.  This guard resolves *job_dir*
    and requires the resolved path to live directly beneath one of the
    genuine job roots (the parent directories of the discovered entries),
    rejecting symlinks that escape the tree.

    Args:
        job_dir: The candidate job directory from ``discover_job_dirs``.
        discovered: The full ``discover_job_dirs`` mapping, whose values'
            parents form the set of legitimate job roots.

    Returns:
        ``True`` when *job_dir* resolves to a direct child of a recognized
        job root, ``False`` otherwise.
    """
    try:
        resolved = job_dir.resolve()
    except OSError:
        return False
    allowed_roots = set()
    for entry in discovered.values():
        try:
            allowed_roots.add(entry.parent.resolve())
        except OSError:
            continue
    return resolved.parent in allowed_roots


def _trajectory_sort_key(trajectory: dict) -> float:
    """Sort key for trajectories: ascending run start timestamp."""
    value = trajectory.get("run_start_timestamp", 0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _load_trajectories_from_dir(job_dir: Path) -> list[dict]:
    """Load all trajectory YAML files directly from *job_dir*.

    Mirrors ``kiss.viz_trajectory.server.load_job_trajectories`` but
    reads from the ALREADY-authorized directory instead of re-resolving
    the job name through ``find_job_dir`` — the re-resolution prefers
    the primary root and follows symlinks, so it can select a different
    (older or symlinked out-of-tree) directory than the one the caller
    just validated against the discovery allow-list.

    Args:
        job_dir: The validated job directory to read.

    Returns:
        The parsed trajectory dicts sorted by ascending
        ``run_start_timestamp``.
    """
    from kiss.viz_trajectory.server import _parse_trajectory_yaml

    trajectories: list[dict] = []
    for file_path in sorted((job_dir / "trajectories").glob("trajectory_*.yaml")):
        try:
            trajectories.append(_parse_trajectory_yaml(file_path))
        except Exception:
            logger.debug("Error loading %s", file_path, exc_info=True)
    trajectories.sort(key=_trajectory_sort_key)
    return trajectories


@dataclass(frozen=True)
class TaskResult:
    """Final outcome of one synchronous daemon task run.

    Attributes:
        text: Human-readable result summary produced by the agent.
        success: Whether the agent reported the task as successful.
        cost: Budget consumed by the task in USD.
        tokens: Total LLM tokens consumed by the task.
        steps: Total agent steps taken by the task.
        chat_id: The daemon chat session id the task ran on.  Pass it
            back as the ``chat_id`` argument of :func:`run` to
            continue the chat, or use it to inspect the chat later;
            ``""`` when the run ended before the daemon assigned one.
        task_id: The daemon's persisted ``task_history`` row id of the
            run; ``""`` when the run ended before a row was allocated
            (e.g. the daemon had no model configured).
    """

    text: str
    success: bool
    cost: float
    tokens: int
    steps: int
    chat_id: str = ""
    task_id: str = ""


def _resolve_sock_path(sock_path: str | Path | None) -> Path:
    """Return the daemon UDS path to connect to.

    Precedence: explicit *sock_path* argument, then the
    ``KISS_SORCAR_SOCK`` environment variable, then the daemon's
    default ``$KISS_HOME/sorcar.sock``.

    Args:
        sock_path: Optional explicit socket path override.

    Returns:
        The resolved Unix-domain socket path.
    """
    if sock_path:
        return Path(sock_path)
    env = os.environ.get("KISS_SORCAR_SOCK")
    return Path(env) if env else _default_kiss_dir() / "sorcar.sock"


def _parse_cost(value: Any) -> float:
    """Parse a daemon cost field (``"$0.1234"``, ``"N/A"``, or a number).

    Args:
        value: The ``cost`` field of a daemon ``result`` event.

    Returns:
        The cost in USD; ``0.0`` when the field is absent or unparseable.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().lstrip("$"))
        except ValueError:
            return 0.0
    return 0.0


def _to_task_result(
    event: dict[str, Any] | None,
    chat_id: str = "",
    task_id: str = "",
) -> TaskResult:
    """Convert the final daemon ``result`` event into a :class:`TaskResult`.

    Args:
        event: The last ``result`` event received for the task's tab,
            or ``None`` when the task ended without one.
        chat_id: The daemon chat session id observed on the run's
            ``clear`` event (``""`` when none was seen).
        task_id: The persisted ``task_history`` row id observed on the
            run's event stream (``""`` when none was seen).

    Returns:
        The parsed :class:`TaskResult`.  The daemon enriches ``result``
        events with ``success`` / ``summary`` fields parsed from the
        agent's YAML result; ``summary`` is preferred over the raw
        ``text`` when present.
    """
    if event is None:
        return TaskResult(
            text="", success=False, cost=0.0, tokens=0, steps=0,
            chat_id=chat_id, task_id=task_id,
        )
    text = str(event.get("summary") or event.get("text") or "")
    return TaskResult(
        text=text,
        success=bool(event.get("success", False)),
        cost=_parse_cost(event.get("cost")),
        tokens=int(event.get("total_tokens", 0) or 0),
        steps=int(event.get("step_count", 0) or 0),
        chat_id=chat_id,
        task_id=task_id,
    )



@dataclass(frozen=True)
class ApiCommand:
    """One command of the Sorcar server API.

    Attributes:
        name: The wire value of the command's ``"type"`` field.
        required: Fields that must be present (and non-``None``) on
            the command for the daemon to accept it.
        handler: Name of the :class:`ServerApi` method that services
            the command — the actual code API entry point a client
            command invokes.  ``"forward"`` (the default) routes the
            command to the backend agent server unchanged; ``"drop"``
            marks a client message the daemon accepts and discards
            (consumed by the VS Code extension host or the WSS
            handshake, never by the daemon).
    """

    name: str
    required: tuple[str, ...] = ()
    handler: str = "forward"


def _catalog(*commands: ApiCommand) -> dict[str, ApiCommand]:
    """Build a name-keyed command catalog.

    Args:
        commands: The commands making up the catalog.

    Returns:
        A dict mapping each command's name to the command.
    """
    return {c.name: c for c in commands}


API: dict[str, ApiCommand] = _catalog(
    ApiCommand("run", required=("prompt",)),
    ApiCommand("submit", required=("prompt",), handler="submit"),
    ApiCommand("appendUserMessage", required=("prompt",)),
    ApiCommand("stop"),
    ApiCommand("userAnswer", required=("answer",)),
    ApiCommand("newChat"),
    ApiCommand("closeTab", required=("tabId",), handler="close_tab"),
    ApiCommand("resumeSession", handler="resume_session"),
    ApiCommand("ready", handler="ready"),
    ApiCommand("getHistory"),
    ApiCommand("getAdjacentTask", required=("direction",)),
    ApiCommand("getFrequentTasks"),
    ApiCommand("deleteFrequentTask", required=("task",)),
    ApiCommand("setFavorite", required=("taskId", "isFavorite")),
    ApiCommand("getInputHistory"),
    ApiCommand(
        "getWelcomeSuggestions", handler="get_welcome_suggestions"
    ),
    ApiCommand("activeTasksQuery", handler="active_tasks_query"),
    ApiCommand("getModels"),
    ApiCommand("selectModel", required=("model",)),
    ApiCommand("getConfig"),
    ApiCommand("saveConfig", required=("config",)),
    ApiCommand("setWorkDir", required=("workDir",)),
    ApiCommand("getFiles", required=("prefix",)),
    ApiCommand("recordFileUsage", required=("path",)),
    ApiCommand("openFile", required=("path",), handler="open_file"),
    ApiCommand("checkPaths", required=("paths",), handler="check_paths"),
    ApiCommand("complete", required=("query",)),
    ApiCommand("mergeAction", required=("action",), handler="merge_action"),
    ApiCommand("worktreeAction", required=("action",)),
    ApiCommand("autocommitAction", required=("action",)),
    ApiCommand("generateCommitMessage"),
    ApiCommand("auth", required=("password",), handler="drop"),
    ApiCommand("runUpdate", handler="run_update"),
    ApiCommand("serverReset", handler="server_reset"),
    ApiCommand(
        "voiceTranscribe", required=("audio",), handler="voice_transcribe"
    ),
    ApiCommand("voiceToggle", required=("enabled",), handler="drop"),
    ApiCommand("voiceSensitivity", required=("value",), handler="drop"),
    ApiCommand("voiceAck", handler="drop"),
    ApiCommand("voiceDropped", required=("text",), handler="drop"),
    ApiCommand("cliEvent", required=("event",), handler="cli_event"),
    ApiCommand("cliTabHello", required=("tabId",), handler="cli_tab_hello"),
    ApiCommand("cliTaskStart", required=("taskId",), handler="cli_task_start"),
    ApiCommand("cliTaskEnd", required=("taskId",), handler="cli_task_end"),
    ApiCommand("cliInfo"),
    ApiCommand("focusEditor", handler="drop"),
    ApiCommand("webviewFocusChanged", handler="drop"),
    ApiCommand("activeTabChanged", required=("tabId",), handler="drop"),
    ApiCommand("notificationAction", required=("id",), handler="drop"),
    ApiCommand("sizeReport", handler="drop"),
    ApiCommand("resolveDroppedPaths", required=("uris",), handler="drop"),
)
"""Every command the daemon accepts, keyed by wire name.

Each entry binds the wire name to the :class:`ServerApi` method
(``handler``) that services it, making the catalog the single routing
table for the server's code API.  ``auth`` is serviced by
:meth:`ServerApi.authenticate` during the WSS handshake, BEFORE the
per-connection dispatch loop starts; an ``auth`` frame that leaks
into an already-authenticated connection's dispatch is accepted and
discarded.
"""

DROPPED_COMMANDS: frozenset[str] = frozenset(
    c.name for c in API.values() if c.handler == "drop"
)
"""Client messages the daemon accepts and silently discards.

Derived from the catalog (``handler == "drop"``).  These messages are
consumed by the VS Code extension host (webview bridge, voice bridge)
or by the WSS handshake (``auth``, serviced pre-dispatch by
:meth:`ServerApi.authenticate`), so when one leaks to the daemon
transport it must be dropped BEFORE catalog validation — validating
it (e.g. a ``notificationAction`` missing its ``id``) would surface a
spurious error banner for a message the daemon was never meant to
handle.
"""

_CLI_HANDLERS: frozenset[str] = frozenset(
    {"cli_event", "cli_tab_hello", "cli_task_start", "cli_task_end"}
)
"""Handlers of the CLI → daemon bridge commands.

Commands routed to these handlers describe tasks the sorcar CLI runs
itself; :meth:`ServerApi.dispatch` exempts them from the per-window
``workDir`` stamping because they never read ``workDir`` and must not
be mutated on their way to the relay.
"""


def validate_command(cmd: Any) -> str | None:
    """Validate one client command against the server API catalog.

    Args:
        cmd: The parsed JSON value received from a client.

    Returns:
        ``None`` when *cmd* is a valid API command, otherwise a
        human-readable error string (unknown command name or missing
        required field).
    """
    if not isinstance(cmd, dict):
        return "Invalid command: expected a JSON object"
    name = cmd.get("type")
    if not isinstance(name, str) or not name:
        return "Invalid command: missing 'type'"
    spec = API.get(name)
    if spec is None:
        return f"Unknown command: {name}"
    missing = [f for f in spec.required if cmd.get(f) is None]
    if missing:
        return f"Invalid {name} command: missing {', '.join(missing)}"
    return None


def translate_webview_command(cmd: dict[str, Any]) -> dict[str, Any]:
    """Translate a webview wire command into a backend command.

    The chat webview (``media/main.js``) speaks the wire dialect of
    this API; the backend agent server expects slightly different
    field names for one command.  Translation applied:

    * ``resumeSession`` → renames the ``id`` field to ``chatId``

    (``media/main.js`` posts ``userAnswer`` directly, so no
    ``userActionDone`` rewrite is needed here.)

    Args:
        cmd: Raw command dictionary received from a client.

    Returns:
        The (possibly copied and modified) command dictionary ready
        for the backend agent server (``VSCodeServer._handle_command``).
    """
    cmd_type = cmd.get("type", "")
    if cmd_type == "resumeSession" and "id" in cmd and "chatId" not in cmd:
        out = dict(cmd)
        out["chatId"] = out.pop("id")
        return out
    return cmd


def passwords_equal(a: str, b: str) -> bool:
    """Compare two passwords in constant time to defeat timing attacks.

    Encodes both strings to UTF-8 bytes and delegates to
    :func:`secrets.compare_digest`.

    Args:
        a: First password string.
        b: Second password string.

    Returns:
        ``True`` when the two strings are equal.
    """
    return secrets.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


@dataclass(frozen=True)
class ApiContext:
    """Transport context of one in-flight server API call.

    Bundles the per-connection state a :class:`ServerApi` handler may
    need, so handler signatures stay uniform (``handler(cmd, ctx)``)
    and the API layer never depends on which transport (WSS or UDS)
    delivered the command.

    Attributes:
        endpoint: The client connection the command arrived on — a
            ``websockets`` ``ServerConnection`` (remote browser) or an
            :class:`asyncio.StreamWriter` (local VS Code extension).
            Used for direct replies.
        tabs_seen: Per-connection set of frontend tab ids, mutated in
            place; the transport's disconnect cleanup arms a deferred
            ``closeTab`` for every id recorded here.
        conn_state: Per-connection mutable state holding at least the
            connection's ``work_dir`` (announced via ``setWorkDir``)
            and unique ``conn_id``.
        is_uds: ``True`` when the command arrived over the local
            Unix-domain socket (a VS Code window or the sorcar CLI),
            ``False`` for a remote WSS browser client.
    """

    endpoint: Any
    tabs_seen: set[str]
    conn_state: dict[str, Any]
    is_uds: bool


class ServerBackend(Protocol):
    """Daemon capabilities the server API dispatches onto.

    Structural type of the object backing :class:`ServerApi` — in
    production the ``RemoteAccessServer`` of
    :mod:`kiss.server.web_server`, which owns the transports, the
    merge/CLI bookkeeping, and the backend agent server.  Only the
    members the API layer actually calls are declared; see the
    implementing methods in ``web_server.py`` for full behaviour
    documentation.
    """

    _printer: Any

    async def _endpoint_send(self, endpoint: Any, data: str) -> None: ...

    async def _run_cmd(self, cmd: dict[str, Any]) -> None: ...

    def _relay_cli_event(self, ev: dict[str, Any]) -> None: ...

    def _validated_cli_task_id(self, cmd: dict[str, Any]) -> str: ...

    def _handle_cli_task_start(
        self, task_id: str, conn_state: dict[str, Any],
    ) -> None: ...

    def _handle_cli_task_end(
        self, task_id: str, conn_state: dict[str, Any],
    ) -> None: ...

    async def _handle_open_file(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None: ...

    async def _handle_check_paths(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None: ...

    async def _handle_voice_transcribe(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None: ...

    async def _handle_active_tasks_query(self, endpoint: Any) -> None: ...

    def _sanitized_restored_tabs(
        self, cmd: dict[str, Any],
    ) -> list[dict[str, str]]: ...

    async def _handle_ready(
        self, cmd: dict[str, Any], websocket: Any, *, is_uds: bool = False,
    ) -> None: ...

    async def _handle_submit(self, cmd: dict[str, Any]) -> None: ...

    async def _send_welcome_info(self) -> None: ...

    async def _handle_run_update(self, conn_id: str = "") -> None: ...

    async def _handle_server_reset(self, conn_id: str = "") -> None: ...

    async def _handle_web_merge_action(self, cmd: dict[str, Any]) -> None: ...

    def _pop_merge_state(self, tab_id: str) -> Any: ...

    async def _finish_merge_and_close_tab(
        self, tab_id: str, merge_state: Any,
    ) -> None: ...

    def _client_ip(self, websocket: Any) -> str: ...

    def _auth_lock_remaining(self, ip: str) -> float: ...

    def _record_auth_failure(self, ip: str) -> None: ...


class ServerApi:
    """The Sorcar server's code-level API.

    The actual code API every client command invokes.  Each catalog
    entry in :data:`API` names (via ``ApiCommand.handler``) the method
    of this class that services it, so the clients — the VS Code
    extension (``src/SorcarApi.ts`` over UDS), the chat webview /
    remote webapp (``media/api.js`` over WSS), and the CLI / Python
    clients — call these methods remotely by sending the catalog's
    JSON commands.  The transport layer
    (``RemoteAccessServer._dispatch_client_command``) parses the JSON
    and hands every command to :meth:`dispatch`; it never routes a
    command itself, so this class is the single place where the wire
    API is bound to daemon behaviour.

    The heavy lifting stays in the *backend*
    (:class:`ServerBackend`); this class owns exactly the API-level
    concerns: pre-validation drops, catalog validation, per-connection
    stamping (``connId`` / ``workDir`` / tab registration), wire→
    backend field translation, and per-command routing.

    The remote webapp's non-command interactions are part of this API
    as well: a remote WSS connection must first complete the password
    handshake serviced by :meth:`authenticate` before its commands are
    dispatched, and the webapp's trajectory-viewer HTTP data endpoints
    are serviced by :meth:`trajectory_jobs` /
    :meth:`job_trajectories`.
    """

    def __init__(self, backend: ServerBackend) -> None:
        """Bind the API to its daemon backend.

        Args:
            backend: The daemon object providing the transports and
                command implementations (in production the
                ``RemoteAccessServer``).

        Raises:
            TypeError: When a catalog entry names a handler this
                class does not implement — catching a routing typo at
                daemon startup instead of on first use.
        """
        self._backend = backend
        for spec in API.values():
            if spec.handler != "drop" and not callable(
                getattr(self, spec.handler, None)
            ):
                raise TypeError(
                    f"API command {spec.name!r} names unknown handler "
                    f"{spec.handler!r}"
                )

    async def dispatch(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Route one client command to its API method.

        The single entry point of the code API.  Applies, in order:

        1. Silently drops :data:`DROPPED_COMMANDS` (host-consumed
           messages) BEFORE validation so they never surface errors.
        2. Validates *cmd* against the catalog
           (:func:`validate_command`) and answers an invalid command
           with a direct ``error`` event to the sender only.
        3. Records the command's ``tabId`` for the transport's
           deferred-close bookkeeping (:meth:`_record_tab`).
        4. Stamps the connection's ``conn_id`` as ``connId`` —
           overwriting any client-supplied value so it cannot be
           spoofed — which keys the backend's per-connection
           autocomplete state.
        5. Maintains the per-window work_dir invariant: a
           ``setWorkDir`` updates the connection's ``work_dir``; every
           other command lacking an explicit ``workDir`` is stamped
           with it, so two VS Code windows sharing the daemon can
           never observe each other's folder through the daemon-global
           fallback.  The CLI-bridge commands (:data:`_CLI_HANDLERS`)
           are exempt: they describe tasks the CLI runs itself, never
           read ``workDir``, and must not be mutated on their way to
           the relay.
        6. Invokes the :class:`ServerApi` method named by the
           command's catalog entry.

        Args:
            cmd: The parsed JSON command dictionary (the transport
                guarantees a dict).
            ctx: The transport context of this call.
        """
        if cmd.get("type") in DROPPED_COMMANDS:
            return
        error = validate_command(cmd)
        if error:
            reply: dict[str, Any] = {"type": "error", "text": error}
            raw_tab = cmd.get("tabId")
            if isinstance(raw_tab, str) and raw_tab:
                reply["tabId"] = raw_tab
            await self._backend._endpoint_send(ctx.endpoint, json.dumps(reply))
            return
        tab_id = cmd.get("tabId", "")
        if isinstance(tab_id, str) and tab_id:
            self._record_tab(tab_id, ctx)
        cmd["connId"] = ctx.conn_state["conn_id"]
        name = cmd["type"]
        handler = API[name].handler
        if name == "setWorkDir":
            new_wd = cmd.get("workDir", "")
            if isinstance(new_wd, str) and new_wd:
                ctx.conn_state["work_dir"] = new_wd
        elif (
            handler not in _CLI_HANDLERS
            and ctx.conn_state["work_dir"]
            and not cmd.get("workDir")
        ):
            cmd["workDir"] = ctx.conn_state["work_dir"]
        await getattr(self, handler)(cmd, ctx)

    def _record_tab(self, tab_id: str, ctx: ApiContext) -> None:
        """Record *tab_id* as touched by this connection.

        Adds the id to ``ctx.tabs_seen`` (arming the transport's
        deferred ``closeTab`` on disconnect) and, for local UDS peers,
        registers it with the printer's local-tab set exactly once per
        connection (talk-playback arbitration).

        Args:
            tab_id: The non-empty frontend tab identifier.
            ctx: The transport context of the current call.
        """
        if tab_id not in ctx.tabs_seen:
            ctx.tabs_seen.add(tab_id)
        if ctx.is_uds:
            local_tabs = ctx.conn_state.setdefault("local_tabs", set())
            if tab_id not in local_tabs:
                local_tabs.add(tab_id)
                self._backend._printer.register_local_uds_tab(tab_id)

    async def authenticate(self, websocket: Any) -> bool:
        """Authenticate a remote WSS client with the ``auth`` handshake.

        The remote webapp's entry point into the API: before a browser
        connection may issue any catalog command, its very first
        frames must complete this handshake (the ``_WS_SHIM_JS`` shim
        served with the webapp sends ``{"type": "auth", "password":
        ...}`` as soon as the socket opens).  Local UDS clients (the
        VS Code extension, the CLI) skip it — POSIX file permissions
        on the socket already gate access to the owning user.

        Protocol serviced here, in order:

        1. A source IP that is still rate-limited after too many
           failed logins is answered with ``auth_locked`` (carrying
           ``retry_after`` seconds) and closed — telling the client
           WHY instead of leaving its loading overlay spinning.
        2. Otherwise up to two ``auth`` attempts are read: a correct
           password (constant-time compare against the configured
           ``remote_password``, which may be empty) is answered with
           ``auth_ok``; the first wrong password elicits an
           ``auth_required`` retry prompt; the second failure is
           answered with an ``error`` event and the socket is closed.
           A first message that is not an ``auth`` at all closes the
           socket without counting a failed login.
        3. Only NON-EMPTY wrong guesses count toward the brute-force
           lockout: every fresh page load probes with the (possibly
           empty) password stored in ``localStorage``, and behind the
           shared cloudflared tunnel penalising that benign empty
           probe would let a handful of normal page loads lock the
           password prompt away from every visitor.

        Args:
            websocket: The remote client's WebSocket connection.

        Returns:
            ``True`` when the client authenticated; ``False`` when it
            failed (the socket is then already closed).
        """
        backend = self._backend
        ip = backend._client_ip(websocket)
        lock_remaining = backend._auth_lock_remaining(ip)
        if lock_remaining > 0.0:
            logger.warning("Auth rate-limit hit for %s; closing socket", ip)
            try:
                await websocket.send(json.dumps({
                    "type": "auth_locked",
                    "retry_after": math.ceil(lock_remaining),
                }))
                await websocket.close()
            except Exception:
                pass
            return False
        password = load_config().get("remote_password", "")
        try:
            for is_retry, timeout in ((False, 30), (True, 60)):
                raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                # Re-check the lockout BEFORE comparing or accepting the
                # submitted credential: a peer socket may have tripped
                # the per-IP threshold while this already-admitted
                # connection was waiting for the user's input.  Without
                # this check, any number of sockets admitted while the
                # failure count was below the limit could still redeem
                # a guessed password after the lock engaged.
                lock_remaining = backend._auth_lock_remaining(ip)
                if lock_remaining > 0.0:
                    logger.warning(
                        "Auth rate-limit engaged while %s awaited "
                        "credentials; closing socket", ip,
                    )
                    await websocket.send(json.dumps({
                        "type": "auth_locked",
                        "retry_after": math.ceil(lock_remaining),
                    }))
                    await websocket.close()
                    return False
                msg = json.loads(raw)
                client_pw = msg.get("password", "")
                if not isinstance(client_pw, str):
                    client_pw = ""
                if msg.get("type") == "auth" and passwords_equal(
                    password, client_pw,
                ):
                    await websocket.send(json.dumps({"type": "auth_ok"}))
                    return True
                if not is_retry and msg.get("type") != "auth":
                    await websocket.close()
                    return False
                if client_pw:
                    backend._record_auth_failure(ip)
                # Re-check the lockout AFTER recording this failure so a
                # wrong guess that crosses the brute-force threshold is
                # denied its remaining attempt(s) immediately — including
                # the concurrent case where several sockets were admitted
                # together while the failure count was still below the
                # limit.  Without this the single check before the loop
                # could be bypassed by racing connections or a serial
                # attempt that trips the threshold on its first guess.
                lock_remaining = backend._auth_lock_remaining(ip)
                if lock_remaining > 0.0:
                    logger.warning(
                        "Auth rate-limit tripped mid-handshake for %s; "
                        "closing socket", ip,
                    )
                    await websocket.send(json.dumps({
                        "type": "auth_locked",
                        "retry_after": math.ceil(lock_remaining),
                    }))
                    await websocket.close()
                    return False
                if not is_retry:
                    await websocket.send(json.dumps({"type": "auth_required"}))
            await websocket.send(
                json.dumps({"type": "error", "text": "Authentication failed"})
            )
            await websocket.close()
            return False
        except Exception:
            logger.debug("WS auth failed", exc_info=True)
            try:
                await websocket.close()
            except Exception:
                pass
            return False

    async def forward(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Run *cmd* on the backend agent server.

        The default handler: commands with no daemon-side special
        casing (``run``, ``stop``, ``getModels``, ``getConfig``,
        ``getHistory``, ``complete``, …) are executed by the backend
        ``VSCodeServer`` in the thread-pool executor.

        Args:
            cmd: The validated, connection-stamped command.
            ctx: The transport context of the current call (unused).
        """
        await self._backend._run_cmd(cmd)

    async def resume_session(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Resume a chat session in the issuing tab.

        Translates the webview wire field ``id`` to the backend's
        ``chatId`` (:func:`translate_webview_command`), then forwards.

        Args:
            cmd: The ``resumeSession`` command.
            ctx: The transport context of the current call.
        """
        await self.forward(translate_webview_command(cmd), ctx)

    async def ready(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Initialize a (re)loaded chat webview.

        Sanitizes the command's ``restoredTabs`` ONCE (warnings
        included) and writes the cleaned list back so the backend's
        own sanitize pass finds nothing left to reject or truncate,
        then records every restored tab id in the connection's
        bookkeeping: the deferred-close contract is "schedule a
        closeTab for every tab id this connection touched", and
        ``_handle_ready`` re-claims (cancels the pending close of, and
        resumes) every ``restoredTabs`` entry — without recording them
        a later disconnect would never re-arm their deferred close,
        leaking the restored backend state forever.  Finally fans the
        command out through the backend's ready handler (models /
        input history / config / session replay).

        Args:
            cmd: The ``ready`` command.
            ctx: The transport context of the current call.
        """
        cmd["restoredTabs"] = self._backend._sanitized_restored_tabs(cmd)
        for rt in cmd["restoredTabs"]:
            rt_id = rt["tabId"]
            if rt_id:
                self._record_tab(rt_id, ctx)
        await self._backend._handle_ready(
            cmd, ctx.endpoint, is_uds=ctx.is_uds,
        )

    async def submit(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Start a task from a webview ``submit``.

        The backend translates the webview ``submit`` into a ``run``
        (path resolution, running-tab tracking) exactly as the VS Code
        TypeScript extension would.

        Args:
            cmd: The ``submit`` command.
            ctx: The transport context of the current call (unused).
        """
        await self._backend._handle_submit(cmd)

    async def close_tab(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Dispose the backend state of a closed frontend tab.

        A WEB (WSS) client closing its chat tab destroys the only UI
        that could ever finish an in-flight (server-tracked) merge
        review for that tab, so the review is ended first (close =
        accept the remaining hunks; no disk writes) and the tab is
        disposed instead of leaking in ``is_merging`` limbo.  UDS (VS
        Code) clients are exempt: their TypeScript MergeManager owns
        the review in real editor tabs that survive the chat tab's
        closure and will still send ``all-done`` — their ``closeTab``
        forwards to the backend unchanged.

        Args:
            cmd: The ``closeTab`` command.
            ctx: The transport context of the current call.
        """
        tab_id = cmd.get("tabId", "")
        if isinstance(tab_id, str) and tab_id and not ctx.is_uds:
            merge_state = self._backend._pop_merge_state(tab_id)
            await self._backend._finish_merge_and_close_tab(
                tab_id, merge_state,
            )
            return
        await self.forward(cmd, ctx)

    async def merge_action(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Advance a merge review (accept / reject / navigate / finish).

        Non-``all-done`` actions are processed by the daemon's
        server-side merge engine (the web twin of the VS Code
        TypeScript ``MergeManager``).  An ``all-done`` arriving FROM a
        client is the extension's MergeManager finishing its
        editor-managed review (its per-hunk actions never reach the
        backend): the server-side shadow merge state registered when
        the ``merge_data`` event was broadcast is dropped — leaving it
        would replay a ZOMBIE review on the next webview reload, fire
        a spurious second all-done from the deferred-close path, and
        leak one state (with full file payloads) per finished review —
        and the command still falls through to the backend
        (``_cmd_merge_action`` → ``_finish_merge``).

        Args:
            cmd: The ``mergeAction`` command.
            ctx: The transport context of the current call.
        """
        if cmd.get("action", "") != "all-done":
            await self._backend._handle_web_merge_action(cmd)
            return
        tab_id = cmd.get("tabId", "")
        if isinstance(tab_id, str) and tab_id:
            # The finishing client may be one of the other windows
            # mirroring the review; the shadow state is the owner's.
            self._backend._pop_merge_state(
                self._backend._printer.ui_mirror_owner(tab_id, "merge_data"),
            )
        await self.forward(cmd, ctx)

    async def open_file(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Serve a file's content to a remote-web client.

        A remote-web (WSS) client clicked a file link in a chat
        webview.  The browser has no editor to open the file in, so
        the daemon reads the file and replies with its content for an
        in-page content tab.  UDS clients (VS Code windows) never take
        this path: their webview's ``openFile`` is consumed by the
        extension host, which opens the file in a real editor tab — so
        a UDS-delivered ``openFile`` is dropped as a defensive no-op.

        Args:
            cmd: The ``openFile`` command.
            ctx: The transport context of the current call.
        """
        if ctx.is_uds:
            return
        await self._backend._handle_open_file(cmd, ctx.endpoint)

    async def check_paths(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Report which file paths exist to a remote-web client.

        The chat webview linkifies file-path-looking strings in event
        panel contents lazily: a path only becomes a clickable link
        after this check confirms that clicking it (``openFile``)
        would actually serve a file.  UDS clients (VS Code windows)
        never take this path: their webview's ``checkPaths`` is
        consumed by the extension host, which checks the local
        filesystem itself — so a UDS-delivered ``checkPaths`` is
        dropped as a defensive no-op.

        Args:
            cmd: The ``checkPaths`` command.
            ctx: The transport context of the current call.
        """
        if ctx.is_uds:
            return
        await self._backend._handle_check_paths(cmd, ctx.endpoint)

    async def voice_transcribe(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Transcribe a remote-web client's post-wake utterance.

        A remote-web (browser mode) client heard the "Sorcar" wake
        word and captured the utterance that followed in the page (VS
        Code webviews never send this: their speech is captured and
        translated by the extension host's local listener).  The audio
        is translated with the same gpt-audio call the local listener
        uses and answered with the ``voiceSpeech`` message
        ``voice.js`` already handles.

        Args:
            cmd: The ``voiceTranscribe`` command carrying the audio.
            ctx: The transport context of the current call.
        """
        await self._backend._handle_voice_transcribe(cmd, ctx.endpoint)

    async def active_tasks_query(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Report in-flight agent tasks back to the requesting client.

        Args:
            cmd: The ``activeTasksQuery`` command (unused).
            ctx: The transport context of the current call.
        """
        await self._backend._handle_active_tasks_query(ctx.endpoint)

    async def get_welcome_suggestions(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Broadcast the welcome-screen suggestions.

        Args:
            cmd: The ``getWelcomeSuggestions`` command (unused).
            ctx: The transport context of the current call (unused).
        """
        await self._backend._send_welcome_info()

    async def run_update(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Run the KISS Sorcar installer to update the checkout.

        Args:
            cmd: The ``runUpdate`` command (unused).
            ctx: The transport context of the current call; supplies
                the requesting ``conn_id`` so acknowledgement
                notifications reach only the requesting window.
        """
        await self._backend._handle_run_update(ctx.conn_state["conn_id"])

    async def server_reset(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Restart the kiss-web daemon at the user's request.

        Args:
            cmd: The ``serverReset`` command (unused).
            ctx: The transport context of the current call; supplies
                the requesting ``conn_id`` so acknowledgement
                notifications reach only the requesting window.
        """
        await self._backend._handle_server_reset(ctx.conn_state["conn_id"])

    async def cli_event(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Relay one CLI display event to subscribed webview tabs.

        CLI → daemon live-stream bridge: the sorcar CLI forwards every
        display event here so any chat webview subscribed to the
        task's chat id sees the event immediately instead of having to
        reload to replay it from the events DB.

        Args:
            cmd: The ``cliEvent`` envelope carrying the event.
            ctx: The transport context of the current call (unused).
        """
        ev = cmd.get("event")
        if isinstance(ev, dict):
            self._backend._relay_cli_event(ev)

    async def cli_tab_hello(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Register a sorcar CLI REPL's tab id for talk arbitration.

        A CLI REPL announces its tab id so talk-playback arbitration
        can tell CLI terminal players apart from webview tabs.  Only
        local UDS peers are terminal players; a WSS/browser peer
        cannot suppress playback on the daemon machine.

        Args:
            cmd: The ``cliTabHello`` command.
            ctx: The transport context of the current call.
        """
        raw_tab = cmd.get("tabId")
        if ctx.is_uds and isinstance(raw_tab, str) and raw_tab:
            cli_tabs = ctx.conn_state.setdefault("cli_tabs", set())
            if raw_tab not in cli_tabs:
                cli_tabs.add(raw_tab)
                self._backend._printer.register_cli_tab(raw_tab)

    async def cli_task_start(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Record a CLI-launched task as running.

        The CLI announces a fresh running task so a webview tab that
        later resumes it from the history sidebar is subscribed to the
        live stream and shows the blinking-green-circle "running"
        indicator.

        Args:
            cmd: The ``cliTaskStart`` command.
            ctx: The transport context of the current call.
        """
        task_id = self._backend._validated_cli_task_id(cmd)
        if task_id:
            self._backend._handle_cli_task_start(task_id, ctx.conn_state)

    async def cli_task_end(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Mark a CLI-launched task as finished.

        The CLI announces the task finished; the daemon stops the
        running indicator on every subscribed webview tab.

        Args:
            cmd: The ``cliTaskEnd`` command.
            ctx: The transport context of the current call.
        """
        task_id = self._backend._validated_cli_task_id(cmd)
        if task_id:
            self._backend._handle_cli_task_end(task_id, ctx.conn_state)

    @staticmethod
    def trajectory_jobs() -> tuple[int, str, bytes]:
        """List all trajectory jobs (the ``/api/jobs`` endpoint).

        Mirrors the ``/api/jobs`` endpoint of the standalone
        trajectory visualizer (:mod:`kiss.viz_trajectory.server`,
        imported lazily so this client-importable module stays light).

        Returns:
            ``(200, "application/json", body)`` with the JSON job
            list.
        """
        from kiss.server import web_server as _ws

        body = json.dumps(_ws.list_jobs(_ws.get_jobs_root())).encode("utf-8")
        return (200, "application/json", body)

    @staticmethod
    def job_trajectories(path: str) -> tuple[int, str, bytes]:
        """Serve one job's trajectory list (``/api/jobs/<job>/trajectories``).

        Mirrors the ``/api/jobs/<job_name>/trajectories`` endpoint of
        the standalone trajectory visualizer.

        Args:
            path: Request path of the form
                ``/api/jobs/<job_name>/trajectories``.  The transport
                has already URL-decoded it exactly once; the job
                segment must NOT be unquoted again or names containing
                literal percent-escapes would spuriously 404.

        Returns:
            ``(200, "application/json", body)`` with the trajectory
            list, a 400 reply for an invalid job name, or a 404 reply
            when the job directory does not exist.
        """
        from kiss.server import web_server as _ws
        from kiss.viz_trajectory.server import discover_job_dirs

        job_name = path[len("/api/jobs/") : -len("/trajectories")]
        # Reject the empty segment and path separators/NUL; a harmless
        # ``..`` SUBSTRING (e.g. the legal name ``job_a..b``, which the
        # listing exposes) is fine because authorization below is exact
        # membership in the discovered allow-list, not path arithmetic.
        if (
            not job_name
            or "/" in job_name
            or "\\" in job_name
            or "\x00" in job_name
            or job_name in (".", "..")
        ):
            return (400, "application/json", b'{"error": "Invalid job name"}')
        jobs_root = _ws.get_jobs_root()
        # Authorize against the SAME allow-list the ``/api/jobs`` listing
        # exposes (``discover_job_dirs`` — only ``job_*`` directories under a
        # recognized ``.kiss.artifacts/jobs`` root).  Using ``find_job_dir``
        # here would additionally accept any child directory of the primary
        # root and follow directory symlinks pointing outside every job root,
        # disclosing unlisted or out-of-tree data and disagreeing with the
        # listing endpoint.
        discovered = discover_job_dirs(jobs_root)
        job_dir = discovered.get(job_name)
        if job_dir is None or not _job_dir_is_contained(job_dir, discovered):
            body = json.dumps(
                {"error": f"Job '{job_name}' not found"}
            ).encode("utf-8")
            return (404, "application/json", body)
        # Load from the ALREADY-validated directory.  Passing
        # ``(root, name)`` through ``load_job_trajectories`` would
        # re-resolve the name via ``find_job_dir`` (primary-root
        # preference, follows symlinks), discarding the containment
        # check above and reintroducing the TOCTOU/duplicate-selection
        # bypass it exists to prevent.
        body = json.dumps(_load_trajectories_from_dir(job_dir)).encode("utf-8")
        return (200, "application/json", body)


def run(
    prompt: str,
    *,
    work_dir: str = "",
    model: str = "",
    chat_id: str = "",
    tools: str | Path | None = None,
    use_worktree: bool = False,
    auto_commit: bool = False,
    max_budget: float | None = None,
    model_config: dict[str, Any] | None = None,
    web_tools: bool | None = None,
    is_parallel: bool = False,
    timeout: float = 3600.0,
    sock_path: str | Path | None = None,
) -> TaskResult:
    """Run *prompt* as a task on the local Sorcar daemon and block until done.

    Connects to the ``sorcar web`` daemon's Unix-domain socket, sends
    the same ``run`` command a chat webview would, streams the task's
    events, and returns once the daemon reports the task finished.

    Args:
        prompt: The task instruction to run.
        work_dir: Working directory for the task; the daemon's current
            default is used when empty.
        model: Model name; the daemon's selected default when empty.
        chat_id: Optional existing chat session id to continue.  Pass
            the ``chat_id`` of a previous :class:`TaskResult` to run
            this task in the same chat — the agent then sees the prior
            tasks and results of that chat as context.  A new chat is
            started when empty.
        tools: Optional path to a Python file supplying extra tools
            for the agent.  The daemon imports the file and registers
            every top-level public function that is suitable as a tool
            (plain synchronous functions whose parameters are all
            keyword-bindable; ``*args``/``**kwargs``/positional-only
            parameters and coroutine/generator functions are skipped).
            Each function's name, docstring (Google-style ``Args:``
            section for parameter descriptions), and annotated
            parameters define the tool schema the agent sees, exactly
            like a native tool.  The functions are never serialized by
            the client — they run **in the daemon process**.  The path
            is resolved against this process's working directory.
        use_worktree: Run the task in an isolated git worktree.
        auto_commit: Auto-commit the task's changes on success.
        max_budget: Per-task budget override in USD; ``None`` uses the
            daemon's configured default.
        model_config: Per-task model configuration override (custom
            endpoint / headers); ``None`` uses the daemon's configured
            model endpoint.  Must be JSON-serializable.
        web_tools: Per-task browser-tool enablement override; ``None``
            uses the daemon's configured default.
        is_parallel: Whether the agent may spawn parallel sub-agents.
        timeout: Maximum seconds to wait for the task to finish.
        sock_path: Daemon UDS path override (defaults to
            ``$KISS_SORCAR_SOCK`` or ``$KISS_HOME/sorcar.sock``).

    Returns:
        A :class:`TaskResult` with the result text, success flag, cost
        (USD), total tokens, step count, chat id, and task id of the
        task.  ``chat_id`` is the daemon chat session id and
        ``task_id`` the persisted ``task_history`` row id — both
        usable later to look up or resume the run in the daemon's
        history.

    Raises:
        ValueError: When *prompt* is empty or blank, or when *tools*
            is not the path of an existing Python (``.py``) file (see
            :func:`~kiss.server.tools_file.resolve_tools_file`).
        ConnectionError: When no daemon is listening on the socket, or
            the daemon drops the connection before the task finishes.
        TimeoutError: When the task does not finish within *timeout*
            seconds.  The client then disconnects, which asks the
            daemon to close the task's tab.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    tools_file = resolve_tools_file(tools)
    path = _resolve_sock_path(sock_path)
    tab_id = f"api-{uuid.uuid4().hex}"
    deadline = time.monotonic() + timeout
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    reader: Any = None
    try:
        sock.settimeout(min(timeout, 10.0))
        try:
            sock.connect(str(path))
        except OSError as exc:
            raise ConnectionError(
                f"Cannot connect to the sorcar daemon at {path}: {exc} "
                f"— start it with `sorcar web`."
            ) from exc
        cmd = {
            "type": "run",
            "prompt": prompt,
            "tabId": tab_id,
            "taskId": uuid.uuid4().hex,
            "chatId": chat_id,
            "workDir": work_dir,
            "model": model,
            "toolsFile": tools_file,
            "useWorktree": use_worktree,
            "autoCommit": auto_commit,
            "maxBudget": max_budget,
            "modelConfig": model_config,
            "webTools": web_tools,
            "useParallel": is_parallel,
        }
        sock.sendall(json.dumps(cmd).encode("utf-8") + b"\n")
        reader = sock.makefile("rb", buffering=_MAX_LINE_BYTES)
        result_event: dict[str, Any] | None = None
        task_id = ""
        started = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Task did not finish within {timeout} seconds"
                )
            sock.settimeout(remaining)
            try:
                line = reader.readline(_MAX_LINE_BYTES)
            except TimeoutError:
                raise TimeoutError(
                    f"Task did not finish within {timeout} seconds"
                ) from None
            if not line:
                raise ConnectionError(
                    "The sorcar daemon closed the connection before the "
                    "task finished"
                )
            if len(line) >= _MAX_LINE_BYTES and not line.endswith(b"\n"):
                # ``readline(size)`` returned a full-size chunk with no
                # terminating newline: the daemon sent a frame larger
                # than the client cap.  Silently skipping the fragments
                # would discard a possibly terminal ``result`` event
                # and misreport the task as failed — fail loudly
                # instead.
                raise ConnectionError(
                    "The sorcar daemon sent an event frame larger than "
                    f"the {_MAX_LINE_BYTES}-byte client limit"
                )
            try:
                event = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(event, dict) or event.get("tabId") != tab_id:
                continue
            etype = event.get("type")
            if etype == "clear":
                chat_id = str(event.get("chat_id", "") or "") or chat_id
            elif etype != "status" and event.get("taskId"):
                task_id = str(event["taskId"])
            if etype == "result":
                result_event = event
            elif etype == "status":
                if event.get("running"):
                    started = True
                elif started:
                    return _to_task_result(result_event, chat_id, task_id)
    finally:
        # ``sock.makefile()`` holds an independent reference to the
        # socket descriptor, so closing only the socket object leaves
        # the buffered reader (and its multi-MiB buffer) alive whenever
        # a caller retains a raised exception whose traceback pins this
        # frame.  Close the reader first so the peer promptly sees EOF.
        try:
            if reader is not None:
                reader.close()
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass
