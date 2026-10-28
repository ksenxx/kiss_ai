# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The Sorcar server API and a minimal synchronous client for it.

This module is the single source of truth for the wire API of the
``kiss-web`` daemon and hosts both of its Python ends:

**The server API** — :data:`API`, :func:`validate_command`, and
:class:`ServerApi` define every command a user interface (the
remote webapp or a Python client) may send to the daemon.  Local and
remote clients speak the same JSON commands over the daemon's WSS
listener, dispatched on the ``"type"`` field, one
object per WebSocket frame.  The daemon routes every command through
:meth:`ServerApi.dispatch`, which validates it against the catalog
(answering an invalid one with an ``{"type": "error", "text": ...}``
event instead of processing it) and invokes the :class:`ServerApi`
method the command's catalog entry names.  The only exception is a
WSS connection's pre-dispatch ``auth`` handshake, serviced by
:meth:`ServerApi.authenticate`.  The user interfaces consume the
catalog through the ``media/api.js`` facade, whose methods map 1:1
onto the catalog's command names; the page's transport additionally
sends the ``auth`` handshake, itself a catalog command.

**The client API** — :func:`run` lets any process launch a task on an
already-running daemon and block until it finishes::

    from kiss.server import sorcar

    result = sorcar.run("Summarize README.md", work_dir="/path/to/repo")
    print(result.text, result.success, result.cost, result.tokens, result.steps)
    print(result.chat_id, result.task_id)  # daemon chat session / task row ids

    # Continue the same chat (the agent sees the prior task as context):
    follow_up = sorcar.run("Now fix the typos you found", chat_id=result.chat_id)

``sea_path="/path/to/my_agent.py"`` names a Sorcar
Extension Agent (SEA): a file defining one subclass of
:class:`kiss.agents.seas.base.base_sea.BaseSea` whose ``settings``
method computes the run's parameters on the daemon — e.g. a
``"model"`` key overrides *model* — while parameters it leaves out
keep the values passed to :func:`run` (see the :func:`run` docstring
for the file format).  The SEA is also the only way to give the agent
extra tools: its ``tools(tools)`` method returns the run's toolset
(the built-in tools it was given plus plain synchronous functions with
keyword-bindable, type-annotated parameters and Google-style
docstrings); with ``"tool_profile": "none"`` the list it returns and
``finish`` are the whole toolset.  The client never serializes Python
functions — the daemon loads the file itself, so the tools execute
**in the daemon process** like native agent tools::

    # my_agent.py
    from kiss.agents.seas.base.base_sea import BaseSea

    def get_temperature(city: str) -> str:
        \"\"\"Return the current temperature of a city.

        Args:
            city: Name of the city to look up.
        \"\"\"
        return lookup_sensor(city)

    class WeatherSea(BaseSea):
        def tools(self, tools):
            \"\"\"Add the temperature tool to the built-in toolset.\"\"\"
            return tools + [get_temperature]

    result = sorcar.run("What's the temperature in Paris?",
                        sea_path="my_agent.py")

The class may also define ``system_prompt(system_prompt)``,
``llm_call_hook(new_messages)`` and ``tool_call_hook(name, args)``,
which the daemon applies to the run and passes to the underlying
:class:`kiss.core.kiss_agent.KISSAgent` (see
:meth:`~kiss.core.kiss_agent.KISSAgent.run`); like ``tools``, these
have no :func:`run` parameter, since a callable cannot travel the
wire.

The function speaks the daemon's JSON protocol over its local WSS
endpoint, found through ``$KISS_HOME/sorcar-local.json`` (or the file
``$KISS_SORCAR_LOCAL`` names).  The endpoint file's mode 0600
restricts the local token it
carries to the owning user, so no password is involved.
"""

from __future__ import annotations

import asyncio
import functools
import ipaddress
import json
import logging
import math
import secrets
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar
from urllib.parse import urlsplit

from kiss.agents.sorcar import cron_agent

# The synchronous client half of this API lives in the sorcar layer
# (``kiss.agents.sorcar.daemon_client``) so sorcar-layer code — the
# ``run_agent`` dispatch tool and the cron scheduler — can submit
# tasks without importing ``kiss.server`` (the layering invariant in
# ``kiss.tests.agents.sorcar.test_layering_invariants``).  It is
# re-exported here unchanged: ``kiss.server.sorcar.run`` and
# ``kiss.server.sorcar.TaskResult`` stay the public client API.
from kiss.agents.sorcar.daemon_client import (
    TaskResult as TaskResult,
)
from kiss.agents.sorcar.daemon_client import (
    run as run,
)
from kiss.agents.sorcar.machines import local_machine_name
from kiss.core.brand import PRODUCT_NAME
from kiss.core.config import kiss_home
from kiss.core.remote_password import password_accepted
from kiss.core.user_config import load_config
from kiss.core.utils import is_root_dir
from kiss.server import sidebar_panels
from kiss.server.commands import broadcast_to_conn
from kiss.server.remote_tabs import ROUTED_COMMANDS, RemoteTabs
from kiss.server.remote_views import VIEW_COMMANDS, RemoteViews
from kiss.server.signin import SigninLogin
from kiss.server.tips import TIPS_OPT_OUT_MARKER
from kiss.server.vscode_retired import NOTICE_ID as VSCODE_RETIRED_NOTICE_ID
from kiss.server.vscode_retired import dismiss_vscode_retired_notice

logger = logging.getLogger(__name__)

K = TypeVar("K")
"""The key of a pending-reply dict (see :meth:`ServerApi._spawn_after`)."""

def _write_tips_opt_out_marker(opt_out: bool) -> None:
    """Create (``opt_out``) or remove the tips opt-out marker file.

    Best-effort: an unwritable ``$KISS_HOME`` only means the tips
    window may show again, so no error surfaces to the user.

    Args:
        opt_out: True to record the opt-out, False to forget it.
    """
    marker = kiss_home() / TIPS_OPT_OUT_MARKER
    try:
        if opt_out:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
        else:
            marker.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not update %s: %s", marker, exc)


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
            command to the backend agent server unchanged.
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
    ApiCommand("interruptTool", required=("tabId",)),
    ApiCommand("userAnswer", required=("answer",)),
    ApiCommand("newChat"),
    ApiCommand("openTab", required=("tabId",)),
    ApiCommand("closeTab", required=("tabId",)),
    ApiCommand("resumeSession", handler="resume_session"),
    ApiCommand("ready", handler="ready"),
    ApiCommand("getHistory", handler="get_history"),
    ApiCommand("getAdjacentTask", required=("direction",)),
    ApiCommand("setFavorite", required=("taskId", "isFavorite")),
    ApiCommand("setChatSummary", required=("chatId", "summary")),
    ApiCommand("deleteTask", required=("taskId",)),
    ApiCommand("deleteChat", required=("chatId",)),
    ApiCommand("openExternal", required=("url",)),
    ApiCommand("getInputHistory"),
    ApiCommand("getSeaCommands"),
    ApiCommand("getWelcomeInfo", handler="get_welcome_info"),
    ApiCommand("activeTasksQuery", handler="active_tasks_query"),
    ApiCommand("getModels"),
    ApiCommand("selectModel", required=("model",)),
    ApiCommand("getConfig"),
    ApiCommand("saveConfig", required=("config",)),
    ApiCommand("getMyModels"),
    ApiCommand("getClaudeCodeStatus"),
    ApiCommand("getCodexStatus"),
    ApiCommand("startCodexLogin"),
    ApiCommand("saveMyModel", required=("name",)),
    ApiCommand("deleteMyModel", required=("name",)),
    ApiCommand("addTrick", required=("text",)),
    ApiCommand("deleteTrick", required=("text",)),
    ApiCommand("editTrick", required=("text", "newText")),
    ApiCommand("browserOpen"),
    ApiCommand("browserClose", required=("tab_id",)),
    ApiCommand("browserNavigate", required=("tab_id", "action")),
    ApiCommand("browserInput", required=("tab_id", "event")),
    ApiCommand("browserViewport", required=("tab_id",)),
    ApiCommand("browserList"),
    ApiCommand("terminalOpen", required=("tab_id",), handler="terminal_open"),
    ApiCommand(
        "terminalInput", required=("tab_id", "data"), handler="terminal_input"
    ),
    ApiCommand(
        "terminalResize",
        required=("tab_id", "cols", "rows"),
        handler="terminal_resize",
    ),
    ApiCommand("terminalClose", required=("tab_id",), handler="terminal_close"),
    ApiCommand("setWorkDir", required=("workDir",), handler="set_work_dir"),
    ApiCommand("setMachine", required=("machine",)),
    ApiCommand(
        "getMachineWorkDirs", required=("machine",), handler="get_machine_work_dirs"
    ),
    ApiCommand("getFiles", required=("prefix",)),
    ApiCommand("recordFileUsage", required=("path",)),
    ApiCommand("openFile", required=("path",), handler="open_file"),
    ApiCommand(
        "saveFile", required=("path", "content"), handler="save_file"
    ),
    ApiCommand("checkPaths", required=("paths",), handler="check_paths"),
    ApiCommand("getTaskUpdate", handler="get_task_update"),
    ApiCommand("getCronJobs", handler="get_cron_jobs"),
    ApiCommand("cronAction", required=("action",), handler="cron_action"),
    ApiCommand("getMachines", handler="get_machines"),
    ApiCommand("machineAction", required=("action", "name"), handler="machine_action"),
    ApiCommand("getAppsStatus", handler="get_apps_status"),
    ApiCommand("getSpendReport", handler="get_spend_report"),
    ApiCommand("getRecommendations", handler="get_recommendations"),
    ApiCommand("listDir", handler="list_dir"),
    ApiCommand("gitStatus", handler="git_status"),
    ApiCommand("gitLog", handler="git_log"),
    ApiCommand("gitShow", required=("sha",), handler="git_show"),
    ApiCommand(
        "gitAction", required=("action", "sha"), handler="git_action"
    ),
    ApiCommand("fsAction", required=("action", "path"), handler="fs_action"),
    ApiCommand("shareChat", required=("chatId", "html"), handler="share_chat"),
    ApiCommand(
        "shareChatTasks", required=("chatId",), handler="share_chat_tasks"
    ),
    ApiCommand("complete", required=("query",)),
    ApiCommand("worktreeAction", required=("action",), handler="worktree_action"),
    ApiCommand("autocommitAction", handler="autocommit_action"),
    ApiCommand("auth", handler="auth"),
    ApiCommand("runUpdate", handler="run_update"),
    ApiCommand("updateModels", handler="update_models"),
    ApiCommand("snoozeUpdate", handler="snooze_update"),
    ApiCommand("updateWhenIdle", handler="update_when_idle"),
    ApiCommand("ping", handler="ping"),
    ApiCommand("serverReset", handler="server_reset"),
    ApiCommand(
        "voiceTranscribe", required=("audio",), handler="voice_transcribe"
    ),
    ApiCommand("tipsOptOut", handler="tips_opt_out"),
    ApiCommand("dismissNotice", required=("id",), handler="dismiss_notice"),
    ApiCommand("signinLogin", required=("provider",), handler="signin_login"),
    ApiCommand("signinPaste", required=("text",), handler="signin_paste"),
    ApiCommand("signinCancel", handler="signin_cancel"),
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
    missing = [
        f for f in spec.required
        if cmd.get(f) is None
        # A tab id names a tab: blank or not a string, the command has
        # none (``""`` is a valid value of the other required fields,
        # a ``saveFile`` of an emptied file say).
        or (f == "tabId" and not (isinstance(cmd[f], str) and cmd[f].strip()))
    ]
    if missing:
        return f"Invalid {name} command: missing {', '.join(missing)}"
    return None


def translate_client_command(cmd: dict[str, Any]) -> dict[str, Any]:
    """Translate a chat-page wire command into a backend command.

    The chat page (``media/main.js``) speaks the wire dialect of
    this API; the backend agent server expects slightly different
    field names for one command.  Translation applied:

    * ``resumeSession`` → renames the ``id`` field to ``chatId``

    (``media/main.js`` posts ``userAnswer`` directly, so no
    ``userActionDone`` rewrite is needed here.)

    Args:
        cmd: Raw command dictionary received from a client.

    Returns:
        The (possibly copied and modified) command dictionary ready
        for the backend agent server (``AgentServer._handle_command``).
    """
    cmd_type = cmd.get("type", "")
    if cmd_type == "resumeSession" and "id" in cmd and "chatId" not in cmd:
        out = dict(cmd)
        out["chatId"] = out.pop("id")
        return out
    return cmd


def passwords_equal(a: str, b: str) -> bool:
    """Compare two secrets (the local token) in constant time."""
    return secrets.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def is_loopback_ip(ip: str) -> bool:
    """Return True when *ip* is an IPv4/IPv6 loopback address.

    A loopback TCP peer on the public port is the local ``cloudflared``
    tunnel relaying a remote visitor, or a process (a browser, a
    Python client) on this host.  IPv6-mapped IPv4 loopback
    (``::ffff:127.0.0.1``) counts too.  A malformed / empty string is
    not loopback.

    Args:
        ip: A textual IP address (no port).

    Returns:
        True when *ip* parses to a loopback address.
    """
    try:
        parsed = ipaddress.ip_address(ip)
    except ValueError:
        return False
    mapped = getattr(parsed, "ipv4_mapped", None)
    if mapped is not None:
        parsed = mapped
    return parsed.is_loopback


AuthKind = Literal["local", "loopback", "remote"]
"""How a connection authenticated: local token, loopback browser or remote password."""


@dataclass(frozen=True)
class Auth:
    """The outcome of a successful ``auth`` handshake.

    Attributes:
        kind: ``"local"`` for a token-authenticated loopback client,
            ``"loopback"`` for a browser on this machine itself
            (admitted without a password, with no local privileges),
            ``"remote"`` for a password-authenticated one.
        link: The peer is another daemon's
            :class:`~kiss.agents.sorcar.machines.MachineLink` request
            socket (``"link": true`` in its ``auth`` frame): it is
            answered, and told global events, but left out of the
            task-event fan-out (see
            :meth:`~kiss.server.web_server.WebPrinter.add_client`).
    """

    kind: AuthKind
    link: bool = False


def _auth_result(kind: AuthKind, msg: Any) -> Auth:
    """Return the :class:`Auth` of an accepted ``auth`` frame *msg*."""
    return Auth(kind, link=isinstance(msg, dict) and msg.get("link") is True)


def _frame_token(msg: Any) -> str:
    """Return the ``token`` of the ``auth`` frame *msg*: ``""`` unless it is a string."""
    token = msg.get("token", "") if isinstance(msg, dict) else ""
    return token if isinstance(token, str) else ""


def _is_legacy_vscode_client(websocket: Any) -> bool:
    """Whether *websocket* looks like the retired VS Code extension's client.

    The extension's hand-written WebSocket client (``AgentClient.ts``)
    authenticated with the local token and sent neither a
    ``User-Agent`` nor an ``Origin`` request header on its upgrade
    request; a browser sends both and the Python ``websockets`` client
    sends ``User-Agent: Python/...``.  So a token client whose upgrade
    request carries neither header is taken for the extension.

    Args:
        websocket: The server connection (its ``request.headers`` is
            the HTTP upgrade request's header map).

    Returns:
        True when the upgrade request has neither header.
    """
    request = getattr(websocket, "request", None)
    headers = getattr(request, "headers", None)
    if headers is None:
        return False
    return not headers.get("User-Agent") and not headers.get("Origin")


def _vscode_retired_notice(loopback_url: str) -> str:
    """Return the ``notice`` text telling a VS Code extension user to use the web app."""
    return json.dumps({
        "type": "notice",
        "text": (
            f"VS Code is no longer supported by {PRODUCT_NAME}. "
            f"Use the web app instead: {loopback_url}"
        ),
    })

_NO_PASSWORD_LOG = (
    "Refusing non-localhost auth handshake from %s: no remote password is set"
)
_NO_PASSWORD_REFUSAL: dict[str, Any] = {
    "type": "error",
    "code": "localhost_only",
    "text": "Remote access is turned off: no remote password is set, so "
            "only this computer may connect. Set one in Settings > Remote "
            f"password on the computer running {PRODUCT_NAME} (or run "
            "install.sh there).",
}
#: The private (``local_only``) daemon's answer to any peer that is not
#: one of the machine's own clients: no password gets it in.
_LOCAL_ONLY_REFUSAL: dict[str, Any] = {
    "type": "error",
    "code": "auth_failed",
    "text": "This daemon accepts local clients only.",
}
_NOT_AUTH_REFUSAL: dict[str, Any] = {
    "type": "error",
    "code": "auth_failed",
    "text": "The first frame must be an auth frame.",
}


def _auth_locked(lock_remaining: float) -> dict[str, Any]:
    """Return the ``auth_locked`` event for a rate-limited peer.

    Args:
        lock_remaining: Seconds left on the peer's brute-force lockout.

    Returns:
        The event, with ``retry_after`` rounded up to whole seconds.
    """
    return {"type": "auth_locked", "retry_after": math.ceil(lock_remaining)}


@dataclass(frozen=True)
class ApiContext:
    """Transport context of one in-flight server API call.

    Bundles the per-connection state a :class:`ServerApi` handler may
    need, so handler signatures stay uniform (``handler(cmd, ctx)``)
    and the API layer never depends on how the client authenticated.

    Attributes:
        endpoint: The client connection the command arrived on — a
            ``websockets`` ``ServerConnection``.  Used for direct
            replies.
        conn_state: Per-connection mutable state holding at least the
            connection's unique ``conn_id``.
        is_local: ``True`` when the connection authenticated with the
            daemon's local token from a loopback address (a local
            Python client such as a ``daemon_client`` run), ``False``
            for a remote browser client that supplied the password.
    """

    endpoint: Any
    conn_state: dict[str, Any]
    is_local: bool


class ServerBackend(Protocol):
    """Daemon capabilities the server API dispatches onto.

    Structural type of the object backing :class:`ServerApi` — in
    production the ``RemoteAccessServer`` of
    :mod:`kiss.server.web_server`, which owns the transports and the
    backend agent server.  Only the
    members the API layer actually calls are declared; see the
    implementing methods in ``web_server.py`` for full behaviour
    documentation.
    """

    _printer: Any
    _agent_server: Any
    port: int

    def _lan_urls(self) -> list[str]: ...

    async def _reply_direct(
        self, endpoint: Any, reply: dict[str, Any], what: str,
    ) -> None: ...

    async def _run_cmd(self, cmd: dict[str, Any]) -> None: ...

    async def _handle_open_file(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None: ...

    async def _handle_save_file(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None: ...

    async def _handle_share_chat(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None: ...

    async def _handle_share_chat_tasks(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None: ...

    async def _handle_check_paths(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None: ...

    async def _handle_get_task_update(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None: ...

    async def _handle_list_dir(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None: ...

    def _cmd_work_dir(self, cmd: dict[str, Any]) -> str: ...

    async def _handle_git_status(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None: ...

    async def _handle_git_log(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None: ...

    async def _handle_git_show(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None: ...

    async def _handle_git_action(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None: ...

    async def _handle_fs_action(
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
        self, cmd: dict[str, Any], websocket: Any,
    ) -> None: ...

    async def _handle_submit(
        self, cmd: dict[str, Any], endpoint: Any = None,
    ) -> None: ...

    async def _send_welcome_info(self, conn_id: str = "") -> None: ...

    async def _handle_run_update(self, conn_id: str = "") -> None: ...

    async def _handle_update_models(self, conn_id: str = "") -> None: ...

    async def _handle_snooze_update(self, latest: str = "") -> None: ...

    async def _handle_update_when_idle(self, cancel: bool = False) -> None: ...

    async def _handle_server_reset(self, conn_id: str = "") -> None: ...

    def _client_ip(self, websocket: Any) -> str: ...

    def _peer_is_loopback(self, connection: Any) -> bool: ...

    async def _peer_client_ca(self, connection: Any) -> str: ...

    def _auth_lock_remaining(self, ip: str) -> float: ...

    def _record_auth_failure(self, ip: str) -> None: ...

    @property
    def local_token(self) -> str: ...

    @property
    def local_only(self) -> bool: ...

    @property
    def _loopback_url(self) -> str: ...


class ServerApi:
    """The Sorcar server's code-level API.

    The actual code API every client command invokes.  Each catalog
    entry in :data:`API` names (via ``ApiCommand.handler``) the method
    of this class that services it, so the clients — the remote
    webapp's chat page (through ``media/api.js``) and the Python
    clients — call these methods remotely by sending
    the catalog's JSON commands.  The transport layer
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
        self.remote_views = RemoteViews(backend)
        self.remote_tabs = RemoteTabs(backend, self.remote_views)
        # In-flight background replies (``appsStatus``, ``machines``,
        # the subpanel forwards) and detached commands
        # (``worktreeAction``): the event loop keeps only weak
        # references to tasks.
        self._background_replies: set[asyncio.Task[None]] = set()
        # Machine registry changes and work-dir round trips run as
        # background tasks (so a slow handshake never holds up the
        # connection's other commands) but in command order: the tasks
        # start in the order the commands arrived and this lock hands
        # itself on first come, first served, so two ``setWorkDir``
        # for one machine apply, and answer, in the order they were sent.
        self._machine_lock = asyncio.Lock()
        # The subpanel forward in flight per (connection, command,
        # machine): a poll repeated while its predecessor is still
        # waiting on the machine is dropped (see ``_spawn_panel_reply``).
        self._panel_replies: dict[tuple[str, str, str], asyncio.Task[None]] = {}
        # The latest detached command per (connection, tab) (see
        # ``_dispatch_detached``): the tab's next command from that
        # connection queues behind it.
        self._pending_routes: dict[tuple[str, str], asyncio.Task[None]] = {}
        # The latest dictation per connection (see ``voice_transcribe``):
        # the connection's next one is answered after it.
        self._pending_voice: dict[str, asyncio.Task[None]] = {}
        # The latest history page asked for per connection (see
        # ``get_history``): the connection's next one is answered after it.
        self._pending_history: dict[str, asyncio.Task[None]] = {}
        # The brand's running ``signin login`` (see ``signin_login``).
        self.signin = SigninLogin()
        # Set by ``cancel_background_replies``: from then on no reply
        # is spawned (a frame can still arrive while the cancelled
        # tasks are awaited, before the listeners close).
        self._closing = False
        for spec in API.values():
            if not callable(
                getattr(self, spec.handler, None)
            ):
                raise TypeError(
                    f"API command {spec.name!r} names unknown handler "
                    f"{spec.handler!r}"
                )

    async def dispatch(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Route one client command to its API method.

        The single entry point of the code API.  Applies, in order:

        1. Canonicalises a string ``tabId`` (surrounding whitespace
           stripped) and writes it back into *cmd*, so the registry,
           the agent-state registry, the printer and the cleanup
           tail all key the tab identically.
        2. Validates *cmd* against the catalog
           (:func:`validate_command`) and answers an invalid command
           with a direct ``error`` event to the sender only.
        3. Records the command's ``tabId`` in the connection's
           bookkeeping (:meth:`_record_tab`).
        4. Stamps the connection's ``conn_id`` as ``connId`` —
           overwriting any client-supplied value so it cannot be
           spoofed — which keys the backend's per-connection
           autocomplete state.
        5. Blanks a ``workDir`` naming a filesystem root (``/``,
           ``C:\\`` — see :func:`kiss.core.utils.is_root_dir`) so it is
           treated exactly like an absent one: a client whose cwd
           degenerated to the root can never make the daemon's single
           global working directory the whole disk, nor run a task
           there.  A command without a ``workDir`` resolves to that
           global directory (``config.json`` ``work_dir``, set from any
           surface's chat-header work-dir drop-down via ``setWorkDir``).
        6. Sends a task command of a tab bound to a remote machine to
           that machine (:class:`~kiss.server.remote_tabs.RemoteTabs`),
           and a workspace command stamped ``machine`` (the Explorer,
           Source Control view, folder picker and file editor of such
           a tab: :data:`kiss.server.sidebar_panels.MACHINE_WORKSPACE_REPLIES`),
           a settings command stamped ``machine`` (the settings
           sheet opened for the machine:
           :data:`kiss.server.sidebar_panels.MACHINE_SETTINGS_REPLIES`)
           a subpanel command stamped ``machine`` (the Schedule,
           Apps, Spend and Recommendations subpanels shown for it:
           :data:`kiss.server.sidebar_panels.MACHINE_PANEL_REPLIES`)
           or the task-info panel's ``getTaskUpdate`` stamped
           ``machine`` (the Task update of the task such a tab runs
           there:
           :func:`kiss.server.sidebar_panels.machine_task_update_reply`)
           to the machine's daemon, answering the sender with its
           reply.
        7. Invokes the :class:`ServerApi` method named by the
           command's catalog entry.

        Args:
            cmd: The parsed JSON command dictionary (the transport
                guarantees a dict).
            ctx: The transport context of this call.
        """
        tab_id = cmd.get("tabId")
        if isinstance(tab_id, str):
            # Canonicalise ONCE, before validation and every handler:
            # ``TabRegistry`` strips ids it stores and broadcasts, so
            # a handler keying ``AgentState`` / ``_tab_chat_views`` /
            # local-tab counts by the raw string gave one wire tab two
            # identities, and closing the canonical id leaked the rest.
            # Only a PRESENT field is written back: writing ``""`` for
            # an absent one made ``validate_command`` see it as present,
            # so the catalog's ``required=("tabId",)`` never fired and a
            # ``tabId``-less ``closeTab`` got no reply (conc1018 H2-F2).
            tab_id = tab_id.strip()
            cmd["tabId"] = tab_id
        else:
            tab_id = ""
        error = validate_command(cmd)
        if error:
            reply: dict[str, Any] = {"type": "error", "text": error}
            if tab_id:
                reply["tabId"] = tab_id
            await self._backend._reply_direct(ctx.endpoint, reply, str(cmd.get("type")))
            return
        if tab_id:
            self._record_tab(tab_id, ctx)
        cmd["connId"] = ctx.conn_state["conn_id"]
        raw_wd = cmd.get("workDir")
        if isinstance(raw_wd, str) and is_root_dir(raw_wd):
            # A filesystem root is never a real workspace: it arrives
            # only from a client that inherited the root as its cwd or
            # from a tab whose persisted registry entry was poisoned by
            # one.
            # Blank it HERE — the one chokepoint every transport
            # shares — so a root can neither become the global working
            # directory via ``setWorkDir`` nor root a task or the
            # @-mention file scan at the whole disk.  The command then
            # falls back to the daemon's configured folder.
            cmd["workDir"] = ""
        if self._dispatch_detached(cmd, tab_id, ctx):
            return
        await self._dispatch_now(cmd, ctx)

    async def _dispatch_now(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Route, forward or handle the validated, stamped *cmd* (the body of :meth:`dispatch`).

        Args:
            cmd: The command, validated and stamped by :meth:`dispatch`.
            ctx: The transport context of the current call.
        """
        name = cmd["type"]
        # A tab bound to a remote machine (chat header drop-down) sends
        # its task commands there; see ``kiss.server.remote_tabs``.
        if name in ROUTED_COMMANDS and await asyncio.to_thread(self.remote_tabs.route, cmd):
            return
        # The Terminal and Browser buttons of such a tab open a shell,
        # or a page, on the machine; see ``kiss.server.remote_views``.
        if name in VIEW_COMMANDS and await asyncio.to_thread(
            self.remote_views.route, cmd,
        ):
            return
        # The Explorer, Source Control view, folder picker and file
        # editor of such a tab browse the machine's working directory,
        # the settings sheet opened for the machine shows and saves
        # its config, API keys and custom models, and the Schedule,
        # Apps, Spend and Recommendations subpanels show the machine
        # the chat header has picked, and the task-info panel's Task
        # update subpanel reports on the task the tab runs there: a
        # workspace, settings, subpanel or task-update command stamped
        # ``machine`` is answered by that machine's daemon.  The
        # subpanels poll, and a machine's app probe takes
        # seconds, as does its Codex device login: their replies are
        # produced in the background (and travel on link lanes of
        # their own, see
        # :func:`~kiss.server.sidebar_panels.machine_panel_reply`,
        # :func:`~kiss.server.sidebar_panels.machine_task_update_reply`
        # and :func:`~kiss.server.sidebar_panels.machine_settings_reply`)
        # so a ``submit`` or ``stop`` sent right after them is not held
        # up behind them, on this connection or on the machine's link.
        machine = str(cmd.get("machine") or "")
        forwarder = sidebar_panels.MACHINE_FORWARDERS.get(name) if machine else None
        if forwarder is not None:
            forwarded = self._forwarded_reply(ctx.endpoint, forwarder, machine, cmd)
            if name in ("cronAction", "startCodexLogin") or (
                name == "getTaskUpdate" and cmd.get("refresh")
            ):
                # A user's own action (the Task update's refresh button
                # included), never dropped; slow (the device login's
                # CLI may take a minute to print its code) but in the
                # background, so it holds up none of the sender's later
                # commands.
                self._spawn_reply(forwarded)
            elif name == "getTaskUpdate":
                # The Task update's 5 s poll, deduplicated per tab: a
                # switch to another tab running on the same machine
                # polls at once, while the left tab's pending poll
                # (its reply outdated by the panel's token) is the one
                # not repeated.
                key = (cmd["connId"], f"{name}:{cmd.get('tabId', '')}", machine)
                self._spawn_panel_reply(key, forwarded)
            elif name in sidebar_panels.MACHINE_PANEL_REPLIES:
                self._spawn_panel_reply((cmd["connId"], name, machine), forwarded)
            else:
                await forwarded
            return
        await getattr(self, API[name].handler)(cmd, ctx)
        if name == "ready":
            # The client has the tabs now: replay the ones running on
            # remote machines to it.
            await asyncio.to_thread(self.remote_tabs.replay_to, cmd["connId"])
            await asyncio.to_thread(self.remote_views.replay_to, cmd["connId"])

    def _spawn_reply(self, reply: Coroutine[Any, Any, None]) -> asyncio.Task[None] | None:
        """Run *reply* as a background task that outlives the command that asked for it.

        For replies that take seconds (app probes, machine pings): each
        connection's commands are dispatched one at a time, so producing
        them inline would hold up a ``submit`` or ``stop`` sent right after
        the poll.  :meth:`cancel_background_replies` ends them at shutdown,
        after which *reply* is closed unrun instead.

        Args:
            reply: The coroutine that sends the direct reply.

        Returns:
            The task, or ``None`` once the server is shutting down.
        """
        if self._closing:
            reply.close()
            return None
        task = asyncio.create_task(reply)
        self._background_replies.add(task)
        task.add_done_callback(self._background_replies.discard)
        return task

    def _spawn_after(
        self, pending: dict[K, asyncio.Task[None]], key: K, reply: Coroutine[Any, Any, None],
    ) -> None:
        """Run *reply* in the background once the task *pending* holds under *key* is done.

        The replies sharing a key go out in the order they were asked
        for, as they would have inline, while other keys are answered
        meanwhile.  *pending* keeps the latest task per key, so the
        next reply with the key queues behind this one; a finished
        task leaves it.  Nothing is spawned once the server is
        shutting down (*reply* is closed unrun, as by
        :meth:`_spawn_reply`).

        Args:
            pending: The latest task per key.
            key: The key *reply* queues under.
            reply: The coroutine that sends the reply.
        """
        if self._closing:
            reply.close()
            return
        task = self._spawn_reply(self._after(pending.get(key), reply))
        assert task is not None
        pending[key] = task
        task.add_done_callback(functools.partial(self._forget_pending, pending, key))
        # Cancelled before *reply* started (at shutdown: before its
        # first step, or while still waiting on its predecessor), the
        # task leaves *reply* unawaited; close it rather than let the
        # GC warn.  A reply that did run is closed already: a no-op.
        task.add_done_callback(functools.partial(self._close_reply, reply))

    @staticmethod
    async def _after(
        previous: asyncio.Task[None] | None, reply: Coroutine[Any, Any, None],
    ) -> None:
        """Await *reply* once *previous* (if any) is done, however it ended."""
        if previous is not None:
            await asyncio.wait({previous})
        await reply

    @staticmethod
    def _close_reply(reply: Coroutine[Any, Any, None], task: asyncio.Task[None]) -> None:
        """Close *reply* once *task*, the ``_after`` that was to await it, is done."""
        reply.close()

    @staticmethod
    def _forget_pending(
        pending: dict[K, asyncio.Task[None]], key: K, task: asyncio.Task[None],
    ) -> None:
        """Drop *task* from *pending* when it is still the latest under *key*."""
        if pending.get(key) is task:
            del pending[key]

    def _dispatch_detached(self, cmd: dict[str, Any], tab_id: str, ctx: ApiContext) -> bool:
        """Dispatch *cmd* in the background when its tab binds to a machine or waits on one.

        A ``submit``, ``run`` or ``resumeSession`` naming a ``machine``
        connects to it first (:meth:`RemoteTabs.route` blocks for up
        to the link's timeout when the host is black-holed), and
        the connection's commands are dispatched one after another:
        routed inline, the bind would hold up every other tab of the
        surface and the ``closeTab`` that would stand the bind down.
        So the bind runs as a background task, and every later command
        the connection sends for the tab (a ``stop``, a ``closeTab`` or
        an ``openTab`` sent while it still connects) queues behind it
        in its order, as it would have been dispatched inline: a
        ``stop`` reaches the machine after the run it refers to, a
        ``closeTab`` unbinds the tab the bind registered and closes it
        on the machine too, and an ``openTab`` after that ``closeTab``
        reopens the tab instead of being undone by the close.  Only
        that tab of that connection waits: the surface's other tabs
        are answered meanwhile, and so is another surface showing the
        same tab, whose commands never waited for this connection's
        (the registry orders the two, see :mod:`kiss.server.remote_tabs`).

        Args:
            cmd: The validated, stamped command.
            tab_id: The command's canonical tab id as :meth:`dispatch`
                computed it: ``""`` when absent or not a string (a
                ``ready`` carries whatever the page sent and names no
                tab to queue under).
            ctx: The transport context of the current call.

        Returns:
            ``True`` when *cmd* is handled in the background, ``False``
            when the caller dispatches it inline.
        """
        key = (cmd["connId"], tab_id)
        binds = cmd["type"] in ("submit", "run", "resumeSession") and cmd.get("machine")
        if not tab_id or not (binds or key in self._pending_routes):
            return False
        self._spawn_after(self._pending_routes, key, self._dispatch_logged(cmd, ctx))
        return True

    async def _dispatch_logged(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Dispatch *cmd* now, logging a failure instead of dying silently in a background task."""
        try:
            await self._dispatch_now(cmd, ctx)
        except Exception:  # noqa: BLE001 - a background task must not die silently
            logger.warning("Error handling client command %r", cmd["type"], exc_info=True)

    def _spawn_panel_reply(
        self, key: tuple[str, str, str], reply: Coroutine[Any, Any, None],
    ) -> None:
        """Run the subpanel forward *reply* in the background unless one like it is in flight.

        The subpanels poll every 30 s and a machine's answer can take
        longer (its app probe, a black-holed host's handshake); the
        forwards run on the default executor, which every client
        command shares.  A repeat of a poll whose predecessor is still
        waiting on the machine is dropped: the predecessor's reply is
        the answer the sender is waiting for, and the queue of
        forwards, and the executor threads they would hold, stay
        bounded by the number of connections.

        Args:
            key: ``(connection id, command, machine)`` of the forward.
            reply: The coroutine that sends the reply.
        """
        pending = self._panel_replies.get(key)
        if pending is not None and not pending.done():
            reply.close()
            logger.debug("%s for %s dropped: the previous one is still pending", *key[1:])
            return
        task = self._spawn_reply(reply)
        if task is not None:
            self._panel_replies[key] = task
            task.add_done_callback(
                functools.partial(self._forget_pending, self._panel_replies, key),
            )

    async def cancel_background_replies(self) -> None:
        """Cancel and await every background reply and detached command (at shutdown).

        The threads they wait on (``asyncio.to_thread``) run to their
        own bounds; only the tasks, which would otherwise hold the
        loop open for a slow machine's answer, end here.  A command
        dispatched from here on (one received while the cancelled
        tasks are awaited) spawns no reply, so none outlives this.
        The brand's running sign-in hook, a child process, is ended
        too: it must not finish a login after the daemon is gone.
        """
        self._closing = True
        self.signin.cancel()
        tasks = list(self._background_replies)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _forwarded_reply(
        self, endpoint: Any, forwarder: Callable[[str, dict[str, Any]], dict[str, Any]],
        machine: str, cmd: dict[str, Any],
    ) -> None:
        """Answer *cmd*, stamped *machine*, with the reply of *forwarder* (run in a thread).

        Args:
            endpoint: The requesting client's transport endpoint.
            forwarder: The command's :data:`~kiss.server.sidebar_panels.MACHINE_FORWARDERS` entry.
            machine: The registered machine the command names.
            cmd: The command.
        """
        try:
            reply = await asyncio.to_thread(forwarder, machine, cmd)
            await self._backend._reply_direct(endpoint, reply, str(cmd["type"]))
        except Exception:  # noqa: BLE001 - a background task must not die silently
            logger.warning("forwarding %s to %s failed", cmd.get("type"), machine, exc_info=True)

    def _record_tab(self, tab_id: str, ctx: ApiContext) -> None:
        """Record *tab_id* as touched by this connection.

        For local peers, records the connection's interest in the id
        with the printer's local-tab bookkeeping so task-event copies
        stamped with other tabs are withheld from the peer
        (``WebPrinter._local_clients_for_tab``).  The printer owns the
        per-connection set, under its own lock.

        Args:
            tab_id: The non-empty frontend tab identifier.
            ctx: The transport context of the current call.
        """
        if ctx.is_local:
            self._backend._printer.register_local_tab(ctx.conn_state["conn_id"], tab_id)

    async def _recv_auth_frame(self, websocket: Any, timeout: float) -> dict[str, Any] | None:
        """The peer's next frame as a JSON object, for the handshake.

        Args:
            websocket: The connection awaiting its credentials.
            timeout: Seconds to wait for the frame.

        Returns:
            The decoded object, or ``None`` when no frame came within
            *timeout* seconds, the socket closed, or the frame was not
            a JSON object; the caller refuses the peer then.
        """
        try:
            msg = json.loads(await asyncio.wait_for(websocket.recv(), timeout=timeout))
        except Exception:
            logger.debug(
                "Peer %s sent no usable auth frame", self._backend._client_ip(websocket),
                exc_info=True,
            )
            return None
        return msg if isinstance(msg, dict) else None

    def _is_local_auth(self, msg: Any, websocket: Any) -> bool:
        """Whether *msg* is an ``auth`` frame carrying this daemon's local token.

        Only a loopback peer can be local: the token travels in the
        endpoint file, which never leaves the machine.

        Args:
            msg: The decoded first frame.
            websocket: The connection it arrived on.

        Returns:
            True when the frame authenticates the peer as local.
        """
        if not isinstance(msg, dict) or msg.get("type") != "auth":
            return False
        token = msg.get("token", "")
        return (
            isinstance(token, str)
            and bool(token)
            and self._backend._peer_is_loopback(websocket)
            and passwords_equal(self._backend.local_token, token)
        )

    def _is_direct_loopback(self, websocket: Any) -> bool:
        """Whether *websocket* is a client on this machine reaching the daemon directly.

        True when the raw TCP peer is loopback AND the client IP the
        backend derives for it (:meth:`_client_ip`) is loopback too:
        the upgrade request carried no ``Cf-Connecting-Ip`` /
        ``X-Forwarded-For``, so no tunnel or proxy relayed it.  These
        are the only peers served while no remote password is set.

        Args:
            websocket: The connection to classify.
        """
        backend = self._backend
        return backend._peer_is_loopback(websocket) and is_loopback_ip(
            backend._client_ip(websocket),
        )

    def _is_loopback_browser(self, websocket: Any) -> bool:
        """Whether *websocket* is a browser page opened at a loopback address.

        True when the raw TCP peer is loopback, the client IP the
        backend derives for it (:meth:`_client_ip`) is loopback too
        (the upgrade request carried no forwarded client IP), AND the
        request's ``Origin`` header names a loopback host -- the page
        the browser loaded from ``http://127.0.0.1:<port>`` or
        ``http://localhost:<port>``.  A visitor relayed by the local
        cloudflared tunnel also arrives from loopback, but with
        ``Cf-Connecting-Ip`` / ``X-Forwarded-For`` set; a program on
        this machine (another daemon's machine link, the Python
        client) sends no ``Origin``; a page relayed by some other
        proxy carries that proxy's public origin.  None of them is a
        loopback browser.

        Args:
            websocket: The connection to classify.

        Returns:
            True for a browser page served at a loopback address.
        """
        if not self._is_direct_loopback(websocket):
            return False
        request = getattr(websocket, "request", None)
        headers = getattr(request, "headers", None)
        origin = headers.get("Origin", "") if headers is not None else ""
        host = urlsplit(origin).hostname or ""
        return host == "localhost" or is_loopback_ip(host)

    async def _admit_loopback_browser(
        self, msg: Any, websocket: Any, token: str
    ) -> Auth | None:
        """Admit a token-less ``auth`` frame from a loopback browser page, else ``None``.

        The web app opened at ``127.0.0.1`` on this machine itself needs
        no password, whatever its ``auth`` frame carries
        (:meth:`_is_loopback_browser`); the per-IP lockout protects the
        password, which such a page is not guessing.  A private daemon
        (``local_only``) admits tokens only, so it never qualifies
        there — on the locked path of :meth:`authenticate` as little as
        on the normal one.

        Args:
            msg: The decoded first frame.
            websocket: The connection it arrived on.
            token: The frame's ``token`` as a string (``""`` when absent
                or not a string, see :func:`_frame_token`).

        Returns:
            The loopback :class:`Auth` after ``auth_ok`` was sent, or
            ``None`` when the frame or the peer does not qualify.
        """
        if self._backend.local_only or not isinstance(msg, dict) or msg.get("type") != "auth":
            return None
        if token or not self._is_loopback_browser(websocket):
            return None
        await websocket.send(json.dumps({"type": "auth_ok", "local": False}))
        return _auth_result("loopback", msg)

    async def _admit_local(self, websocket: Any, msg: dict[str, Any]) -> Auth:
        """Answer ``auth_ok`` to a local-token client and return its :class:`Auth`.

        A non-link client that looks like the retired VS Code extension
        (:func:`_is_legacy_vscode_client`) is also sent a ``notice``
        event pointing it at the web app, which its webview displays.

        Args:
            websocket: The connection whose ``auth`` frame carried the
                local token.
            msg: That ``auth`` frame.

        Returns:
            The ``"local"`` :class:`Auth` of the client.
        """
        await websocket.send(json.dumps({"type": "auth_ok", "local": True}))
        auth = _auth_result("local", msg)
        if not auth.link and _is_legacy_vscode_client(websocket):
            await websocket.send(_vscode_retired_notice(self._backend._loopback_url))
        return auth

    async def _refuse(self, websocket: Any, payload: dict[str, Any]) -> None:
        """Send *payload* and close *websocket*, both best-effort.

        Every refusal :meth:`authenticate` issues ends the handshake
        this way: the client learns WHY (instead of a bare close that
        leaves its loading overlay spinning) and the socket is closed.
        A send or close that fails is ignored; the peer is gone
        either way.

        Args:
            websocket: The client's WebSocket connection.
            payload: The ``auth_locked`` or ``error`` event to send.
        """
        try:
            await websocket.send(json.dumps(payload))
            await websocket.close()
        except Exception:
            pass

    async def authenticate(self, websocket: Any) -> Auth | None:
        """Authenticate a WSS client with the ``auth`` handshake.

        Every client's entry point into the API: before a connection
        may issue any catalog command, its very first frames must
        complete this handshake.  The remote webapp's transport sends
        ``{"type": "auth", "password": ...}`` as soon as the socket
        opens; local Python clients send ``{"type": "auth", "token":
        ...}`` with the
        daemon's per-start local token read from the endpoint file
        (:mod:`kiss.agents.sorcar.local_endpoint`), whose 0600 mode gates the
        token to the owning user.

        Protocol serviced here, in order:

        -1. A peer that presented a client certificate issued by an
           enrolled CA (``tls/trusted-clients/*.pem``, see
           :meth:`~kiss.server.web_server.RemoteAccessServer._peer_client_ca`)
           is another daemon's machine link: its first frame must be an
           ``auth`` frame (its content is irrelevant), answered with
           ``auth_ok`` (``"local": true``); it returns ``"local"``
           whatever the password setting or the lockout.
        0. An ``auth`` frame carrying a ``token`` equal to the
           backend's ``local_token`` (constant-time compare) from a
           LOOPBACK peer is answered with ``auth_ok`` (``"local":
           true``) and returns ``"local"``.  A token from a
           non-loopback peer is never honoured, whatever the password
           setting; a wrong token counts as a wrong guess.  A token
           client that looks like the retired VS Code extension (no
           ``User-Agent`` and no ``Origin`` upgrade header, see
           :func:`_is_legacy_vscode_client`) is then also sent a
           ``notice`` event pointing it at the web app.
        0b. On a daemon that is not ``local_only``, an ``auth`` frame
           without a token from a LOOPBACK BROWSER
           (:meth:`_is_loopback_browser`: the raw TCP peer is loopback,
           :meth:`_client_ip` is loopback too, so the upgrade request
           carried no forwarded client IP, and its ``Origin`` names a
           loopback host) is answered with ``auth_ok`` (``"local":
           false``) whatever password it carries and returns
           ``"loopback"``: the web app opened at ``http://127.0.0.1``
           on the daemon's own machine needs no password, and no
           later frame of it is password-checked.  It gets no local
           privileges.  Visitors relayed by cloudflared also arrive
           from loopback but carry ``Cf-Connecting-Ip`` /
           ``X-Forwarded-For``; programs on this machine (machine
           links, the Python client) send no ``Origin``; both still
           need their password or token.  Any OTHER reverse proxy on
           loopback that neither sets a forwarded-IP header nor
           changes the page's origin would expose the daemon to
           everyone it relays.

        1. A source IP that is still rate-limited after too many
           failed logins is answered with ``auth_locked`` (carrying
           ``retry_after`` seconds) and closed — telling the client
           WHY instead of leaving its loading overlay spinning.  The
           local token (0a) and the loopback browser (0b) are still
           admitted: the lockout guards the password, which neither
           is guessing.
        2. When no ``remote_password`` is configured, a
           non-loopback TCP peer is refused (``error`` + close) before
           any credential is examined: with no password set, only
           localhost may connect.  The check runs before the handshake
           AND again for every attempt (against a freshly re-loaded
           config), so it mirrors the transport's 403 gate
           (``RemoteAccessServer._process_request``) and also covers a
           password cleared at any point mid-handshake.
        3. Otherwise up to two ``auth`` attempts are read: a correct
           password (a constant-time compare against the configured
           ``remote_password`` as re-loaded for that attempt, so a
           password change applies immediately) is answered with
           ``auth_ok``; the first wrong password elicits an
           ``auth_required`` retry prompt; the second failure is
           answered with an ``error`` event and the socket is closed.
           A first message that is not an ``auth`` at all closes the
           socket without counting a failed login.
        4. Only NON-EMPTY wrong guesses count toward the brute-force
           lockout: every fresh page load probes with the (possibly
           empty) password stored in ``localStorage``, and behind the
           shared cloudflared tunnel penalising that benign empty
           probe would let a handful of normal page loads lock the
           password prompt away from every visitor.

        Args:
            websocket: The client's WebSocket connection.

        Returns:
            The :class:`Auth` of the admitted client (``kind``
            ``"local"`` for a token-authenticated loopback client,
            ``"loopback"`` for a direct loopback peer admitted without
            a password, ``"remote"`` for a password-authenticated one;
            ``link`` when
            its ``auth`` frame declared it a daemon link), ``None`` when
            authentication failed (the socket is then already closed).
        """
        backend = self._backend
        ip = backend._client_ip(websocket)
        trusted_ca = await backend._peer_client_ca(websocket)
        if trusted_ca:
            # Another daemon's machine link, enrolled with its CA: the
            # client certificate it presented in the TLS handshake is
            # its credential, whatever the password setting or the
            # lockout (a 256-bit key, like the local token).
            msg = await self._recv_auth_frame(websocket, 30)
            if msg is None or msg.get("type") != "auth":
                await self._refuse(websocket, _NOT_AUTH_REFUSAL)
                return None
            logger.info("Machine link from %s admitted with a certificate of %s", ip, trusted_ca)
            return await self._admit_local(websocket, msg)
        lock_remaining = backend._auth_lock_remaining(ip)
        if lock_remaining > 0.0:
            # The lockout is per IP, and every peer reaching loopback
            # without a forwarded client IP shares 127.0.0.1 with the
            # machine's own clients: read one frame so a valid local
            # token or the password-free loopback browser page still
            # gets through, then refuse everything else.
            logger.warning("Auth rate-limit hit for %s; closing socket", ip)
            msg = await self._recv_auth_frame(websocket, 30)
            try:
                if msg is not None and self._is_local_auth(msg, websocket):
                    return await self._admit_local(websocket, msg)
                auth = await self._admit_loopback_browser(msg, websocket, _frame_token(msg))
                if auth is not None:
                    return auth
            except Exception:
                logger.debug("Locked peer %s sent no usable frame", ip, exc_info=True)
            await self._refuse(websocket, _auth_locked(lock_remaining))
            return None
        stored_password = (await asyncio.to_thread(load_config)).get(
            "remote_password", "",
        )
        if not stored_password and not self._is_direct_loopback(websocket):
            # Defense in depth for the empty-password localhost-only
            # lockdown (primary gate: the transport's
            # ``_process_request`` refuses non-loopback peers with 403
            # before the WS upgrade).  Re-applied per attempt below,
            # so clearing the password at ANY point before a frame is
            # examined refuses an already-admitted non-loopback peer.
            # A peer relayed to loopback (the tunnel, a proxy) is not
            # local either: ``password_accepted("", "")`` would admit
            # it with no password at all.  A private daemon says why
            # in its own words: a password would not get it in.
            logger.warning(_NO_PASSWORD_LOG, ip)
            await self._refuse(
                websocket,
                _LOCAL_ONLY_REFUSAL if backend.local_only else _NO_PASSWORD_REFUSAL,
            )
            return None
        try:
            for is_retry, timeout in ((False, 30), (True, 60)):
                msg = await self._recv_auth_frame(websocket, timeout)
                if msg is None:
                    await websocket.close()
                    return None
                # The local token first: it is a 256-bit secret the
                # lockout below exists to protect passwords from, and
                # the lockout must never shut the machine's own
                # clients out (see the pre-recv check above).
                if self._is_local_auth(msg, websocket):
                    return await self._admit_local(websocket, msg)
                if backend.local_only:
                    # The private daemon admits no password at all.
                    await self._refuse(websocket, _LOCAL_ONLY_REFUSAL)
                    return None
                token = _frame_token(msg)
                auth = await self._admit_loopback_browser(msg, websocket, token)
                if auth is not None:
                    return auth
                # Re-load the configured password before every compare
                # so a change made while this connection awaited
                # credentials takes effect NOW.  Without the reload, a
                # stale snapshot taken at handshake start would keep
                # accepting the OLD password — and, worse, clearing the
                # password would not subject an already-admitted
                # non-loopback peer to the localhost-only rule below.
                stored_password = (
                    await asyncio.to_thread(load_config)
                ).get("remote_password", "")
                if not stored_password and not self._is_direct_loopback(websocket):
                    logger.warning(_NO_PASSWORD_LOG, ip)
                    await self._refuse(websocket, _NO_PASSWORD_REFUSAL)
                    return None
                # Re-check the lockout BEFORE comparing or accepting the
                # submitted credential, with no await between this check
                # and the compare: a peer socket may have tripped the
                # per-IP threshold while this already-admitted
                # connection was waiting for the user's input or for the
                # config reload above.  Without this check, any number
                # of sockets admitted while the failure count was below
                # the limit could still redeem a guessed password after
                # the lock engaged.
                lock_remaining = backend._auth_lock_remaining(ip)
                if lock_remaining > 0.0:
                    logger.warning(
                        "Auth rate-limit engaged while %s awaited "
                        "credentials; closing socket", ip,
                    )
                    await self._refuse(websocket, _auth_locked(lock_remaining))
                    return None
                client_pw = msg.get("password", "")
                if not isinstance(client_pw, str):
                    client_pw = ""
                if msg.get("type") == "auth" and not token and password_accepted(
                    client_pw, stored_password,
                ):
                    await websocket.send(json.dumps({
                        "type": "auth_ok", "local": False,
                    }))
                    # With no password set only a direct loopback peer
                    # gets here (the gate above): it is admitted as a
                    # loopback client, never password-checked per
                    # frame, like a browser opened at 127.0.0.1.  A
                    # password-authenticated client is "remote".
                    return _auth_result(
                        "remote" if stored_password else "loopback", msg,
                    )
                if msg.get("type") != "auth":
                    # No password was tried: close without the
                    # "not correct" verdict, on either attempt.
                    await websocket.close()
                    return None
                if client_pw or token:
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
                    await self._refuse(websocket, _auth_locked(lock_remaining))
                    return None
                if not is_retry:
                    await websocket.send(json.dumps({"type": "auth_required"}))
            # ``code`` lets the webapp shim tell this apart from other
            # pre-auth errors (it shows the text inside the password
            # dialog and keeps the dialog open across the close below).
            await self._refuse(websocket, {
                "type": "error",
                "code": "auth_failed",
                "text": "That password is not correct. Try again.",
            })
            return None
        except Exception:
            logger.debug("WS auth failed", exc_info=True)
            try:
                await websocket.close()
            except Exception:
                pass
            return None

    async def forward(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Run *cmd* on the backend agent server.

        The default handler: commands with no daemon-side special
        casing (``run``, ``stop``, ``getModels``, ``getConfig``,
        ``getHistory``, ``complete``, …) are executed by the backend
        ``AgentServer`` in the thread-pool executor.

        Args:
            cmd: The validated, connection-stamped command.
            ctx: The transport context of the current call (unused).
        """
        await self._backend._run_cmd(cmd)

    async def resume_session(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Resume a chat session in the issuing tab.

        Translates the wire field ``id`` to the backend's
        ``chatId`` (:func:`translate_client_command`), then forwards.

        Args:
            cmd: The ``resumeSession`` command.
            ctx: The transport context of the current call.
        """
        await self.forward(translate_client_command(cmd), ctx)

    async def worktree_action(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Merge, discard or detach a tab's pending worktree in the background.

        The backend's ``worktreeAction`` handler reports through a
        tab-stamped ``worktree_result`` broadcast and claims the tab's
        ``is_merging`` flag atomically before it starts, so a second
        click is refused and a ``run`` on the tab is turned down while
        it works.  A merge generates its commit message with an LLM
        call and may run the merge-conflict SEA for minutes, and each
        connection's commands are dispatched one at a time, so the
        command runs as a background task: a ``stop`` of another tab's
        task, a ``submit`` or the transport's ``ping`` sent right after
        the click is not held up behind the merge.

        Args:
            cmd: The ``worktreeAction`` command.
            ctx: The transport context of the current call (unused).
        """
        self._spawn_reply(self._forward_detached(cmd))

    async def autocommit_action(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Run the Git Commit button's command on the backend.

        The backend reports through a tab-stamped ``autocommit_done``
        broadcast, which a daemon-link peer is never sent.  Another
        daemon forwards the Git Commit of a tab picked onto this
        machine over such a link and waits for that event as the
        reply, so a link's command is stamped ``replyTo`` with its
        connection id and the backend sends that connection its own
        copy of the outcome (``_commit_main_tree`` of
        :mod:`kiss.server.merge_flow`).

        Args:
            cmd: The ``autocommitAction`` command.
            ctx: The transport context of the current call.
        """
        # The stamp is this daemon's alone: a client's own ``replyTo``
        # must not aim another connection's reply.
        cmd["replyTo"] = cmd["connId"] if ctx.conn_state.get("link") else ""
        await self.forward(cmd, ctx)

    async def _forward_detached(self, cmd: dict[str, Any]) -> None:
        """Run *cmd* on the backend; log a failure instead of dying silently.

        Args:
            cmd: The validated, connection-stamped command.
        """
        try:
            await self._backend._run_cmd(cmd)
        except Exception:  # noqa: BLE001 - a background task must not die silently
            logger.warning("%s command failed", cmd.get("type", ""), exc_info=True)

    async def ready(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Initialize a (re)loaded chat page.

        Sanitizes the command's ``restoredTabs`` (warnings included)
        and writes the cleaned list back for the backend's ready
        handler, then fans the command out through it (models / input
        history / config / session replay).

        Args:
            cmd: The ``ready`` command.
            ctx: The transport context of the current call.
        """
        cmd["restoredTabs"] = self._backend._sanitized_restored_tabs(cmd)
        await self._backend._handle_ready(cmd, ctx.endpoint)

    async def submit(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Start a task from a chat page's ``submit``.

        The backend translates the ``submit`` into a ``run`` (path
        resolution, follow-up routing) including the path-only
        shortcut: a prompt that is just the path of an existing file is
        answered on the submitting connection with the file's
        ``fileContent`` and starts no task.

        Args:
            cmd: The ``submit`` command.
            ctx: The transport context of the current call; its
                endpoint receives a path-only prompt's reply.
        """
        await self._backend._handle_submit(cmd, ctx.endpoint)

    async def open_file(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Serve a clicked file link.

        A client clicked a file or directory link in a chat page.  The
        daemon resolves the path (``~``, the tab's work dir, its pending
        worktree) and answers with the file's content (or a plain-text
        directory listing) for an in-page content tab.

        Args:
            cmd: The ``openFile`` command.
            ctx: The transport context of the current call.
        """
        await self._backend._handle_open_file(cmd, ctx.endpoint)

    async def save_file(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Write a remote-web client's edits back to a file on disk.

        The remote webapp opens a file (``openFile``) in an editable
        Monaco editor inside a content tab; Ctrl/Cmd+S or the tab's
        Save button sends the editor's full text here.  The file must
        already exist (the editor never creates files) and is replaced
        atomically; the ``version`` stamp taken from the ``fileContent``
        reply lets the daemon refuse to overwrite a file that changed
        on disk since it was opened unless ``force`` is set.  The reply
        is a ``fileSaved`` event sent to the requester only.

        Args:
            cmd: The ``saveFile`` command (``path``, ``content``,
                optional ``workDir``, ``tabId``, ``token``, ``version``,
                ``force``).
            ctx: The transport context of the current call.
        """
        await self._backend._handle_save_file(cmd, ctx.endpoint)

    async def check_paths(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Report which file paths exist to the requesting client.

        The chat page linkifies file-path-looking strings in event
        panel contents lazily: a path only becomes a clickable link
        after this check confirms that clicking it (``openFile``)
        would actually open something — a file or a directory — with
        the same resolution ``openFile`` applies.

        Args:
            cmd: The ``checkPaths`` command.
            ctx: The transport context of the current call.
        """
        await self._backend._handle_check_paths(cmd, ctx.endpoint)

    async def get_task_update(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Report the ``/ask`` agent's task update to a task-info panel.

        The remote webapp's task-info panel (docked on desktop, a
        drawer on mobile) polls this command while the visible tab's
        task runs: the info subpanel shows the
        :mod:`~kiss.agents.seas.ask.ask_sea` agent's short answer to
        what that task has done so far and its partial results.  The
        first poll once the task is a minute old runs the agent (as a
        sub-agent of the task, in the task's chat; its cost counts
        towards the task), later polls re-run it every 10 minutes, and
        a poll with ``refresh: true`` (the panel's refresh button)
        re-runs it at once.  The direct ``taskUpdate`` reply goes back
        to the asking endpoint, local or remote.  A poll stamped
        ``machine`` (the tab is bound to a registered machine, where
        its task runs) never reaches this handler: :meth:`dispatch`
        forwards it to the machine's daemon and relays its reply
        (:func:`~kiss.server.sidebar_panels.machine_task_update_reply`).

        Args:
            cmd: The ``getTaskUpdate`` command (``tabId``, optional
                ``knownSig``, ``token``, ``refresh``, ``machine``).
            ctx: The transport context of the current call.
        """
        await self._backend._handle_get_task_update(cmd, ctx.endpoint)

    async def get_cron_jobs(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Send a client the scheduled cron jobs for its Schedule subpanel.

        The right sidebar's "Schedule" subpanel polls this command.
        The direct reply, to whichever endpoint (local or remote)
        asked, is ``{"type": "cronJobs", "jobs": [...]}`` with the
        rows of :func:`kiss.server.sidebar_panels.cron_jobs_report`.

        Args:
            cmd: The ``getCronJobs`` command (no fields).
            ctx: The transport context of the current call.
        """
        jobs = await asyncio.to_thread(sidebar_panels.cron_jobs_report)
        await self._backend._reply_direct(
            ctx.endpoint, {"type": "cronJobs", "jobs": jobs}, "getCronJobs"
        )

    async def get_apps_status(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Send a client every third-party agent's authentication status.

        Feeds the right sidebar's "Apps" subpanel.  The status comes
        from :func:`kiss.server.sidebar_panels.apps_status` (a cached
        probe subprocess; ``refresh: true`` — the subpanel's refresh
        button, or an app waiting for its connect task — probes again).
        The direct reply is ``{"type": "appsStatus", "apps": [...],
        "checkedAt": <epoch ms>}``.

        A probe takes seconds, and each connection's commands are
        dispatched one at a time, so the reply is produced by a
        background task: a ``submit`` or ``stop`` sent right after the
        poll is not held up behind the probe.

        Args:
            cmd: The ``getAppsStatus`` command (optional ``refresh``).
            ctx: The transport context of the current call.
        """
        self._spawn_reply(self._reply_apps_status(ctx.endpoint, bool(cmd.get("refresh"))))

    async def _reply_apps_status(self, endpoint: Any, refresh: bool) -> None:
        """Probe (or read the cache) and send the ``appsStatus`` reply.

        Args:
            endpoint: The requesting client's transport endpoint.
            refresh: Probe again even when the cache is fresh.
        """
        try:
            apps, checked_at = await asyncio.to_thread(sidebar_panels.apps_status, refresh)
            await self._backend._reply_direct(
                endpoint,
                {"type": "appsStatus", "apps": apps, "checkedAt": checked_at},
                "getAppsStatus",
            )
        except Exception:  # noqa: BLE001 - a background task must not die silently
            logger.warning("getAppsStatus reply failed", exc_info=True)

    async def cron_action(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Apply a Schedule subpanel button to a cron job and send the fresh list.

        ``action`` is ``create`` (the header's ``+`` form: ``name``,
        ``schedule``, ``what`` and ``kind``, ``prompt`` or ``command``;
        no ``jobId``), or ``remove``, ``pause``, ``resume``, ``run_now``
        or ``update`` (with ``name``, ``schedule`` and ``what``) on the
        job ``jobId``, applied by
        :func:`kiss.agents.sorcar.cron_agent.apply_job_action`.  The
        direct reply is a ``cronJobs`` message (as for ``getCronJobs``)
        carrying the ``action`` and an ``error`` text when it was
        refused, so the subpanel re-renders either way (and closes its
        in-place form only when its own create or update went through).

        Args:
            cmd: The ``cronAction`` command (``action``, optional
                ``jobId``, ``name``, ``schedule``, ``what``, ``kind``).
            ctx: The transport context of the current call.
        """
        error = await asyncio.to_thread(
            cron_agent.apply_job_action,
            str(cmd["action"]), str(cmd.get("jobId") or ""), str(cmd.get("name") or ""),
            str(cmd.get("schedule") or ""), str(cmd.get("what") or ""),
            str(cmd.get("kind") or "prompt"),
        )
        jobs = await asyncio.to_thread(sidebar_panels.cron_jobs_report)
        await self._backend._reply_direct(
            ctx.endpoint,
            {"type": "cronJobs", "jobs": jobs, "action": str(cmd["action"]), "error": error},
            "cronAction",
        )

    def _local_machine_address(self) -> tuple[str, int]:
        """Return the (ip, port) the local daemon is reached at: its first LAN IP, else loopback.

        Reads the config file (``_lan_urls`` checks for a password):
        call it off the event loop.
        """
        lan = self._backend._lan_urls()
        host = urlsplit(lan[0]).hostname if lan else "127.0.0.1"
        return host or "127.0.0.1", int(self._backend.port)

    async def get_machines(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Send a client the machines for its MACHINES subpanel.

        The rows come from :func:`kiss.server.sidebar_panels.machines_report`
        (the local machine first, then every registered one with its
        reachability; ``refresh: true`` pings again).  The direct reply
        is ``{"type": "machines", "machines": [...], "checkedAt": <epoch
        ms>, "action": "", "error": ""}``, produced by a background task like
        ``appsStatus`` so the pings never hold up the connection's other
        commands.

        Args:
            cmd: The ``getMachines`` command (optional ``refresh``).
            ctx: The transport context of the current call.
        """
        self._spawn_reply(
            self._reply_machines(ctx.endpoint, bool(cmd.get("refresh")), "", "", "getMachines")
        )

    async def _reply_machines(
        self, endpoint: Any, refresh: bool, action: str, error: str, command: str
    ) -> None:
        """Ping (or read the cache) and send the ``machines`` reply.

        Args:
            endpoint: The requesting client's transport endpoint.
            refresh: Ping again even when the cache is fresh.
            action: The ``machineAction`` the reply answers (``""`` for a poll).
            error: The error text of that action when it was refused.
            command: The command being answered (for the reply log).
        """
        try:
            ip, port = await asyncio.to_thread(self._local_machine_address)
            rows, checked_at = await asyncio.to_thread(
                sidebar_panels.machines_report, ip, port, refresh
            )
            await self._backend._reply_direct(
                endpoint,
                {
                    "type": "machines", "machines": rows, "checkedAt": checked_at,
                    "action": action, "error": error,
                },
                command,
            )
        except Exception:  # noqa: BLE001 - a background task must not die silently
            logger.warning("%s reply failed", command, exc_info=True)

    async def broadcast_machines(self) -> None:
        """Send a fresh ``machines`` reply (pinged anew) to every connection.

        Called by the backend when ``machines.json`` changed under the
        daemon (a ``/machine add`` task or the CLI registered or removed
        a machine), so every surface's MACHINES subpanel refreshes at
        once -- and unfolds for a new machine -- instead of at its next
        poll.  The reply's ``action`` is ``""`` as for a poll.
        """
        ip, port = await asyncio.to_thread(self._local_machine_address)
        rows, checked_at = await asyncio.to_thread(
            sidebar_panels.machines_report, ip, port, True
        )
        self._backend._printer.broadcast({
            "type": "machines", "machines": rows, "checkedAt": checked_at,
            "action": "", "error": "",
        })

    async def machine_action(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Add or remove a machine for the MACHINES subpanel and send the fresh list.

        Applied by :func:`kiss.server.sidebar_panels.machine_action`; the
        direct reply is a ``machines`` message (as for ``getMachines``,
        pinged anew) carrying the ``action`` and an ``error`` text when
        it was refused.  Produced by a background task like
        ``getMachines``: an ``add`` of an unreachable host runs its
        handshakes to their timeouts, which must not hold up the
        connection's ``submit``, ``stop`` or ``ping``.

        Args:
            cmd: The ``machineAction`` command (``action``: ``add`` or
                ``remove``, ``name``, optional ``address`` and
                ``machinePassword`` for ``add``; the machine's own
                password travels under that name because the frame's
                ``password`` is this daemon's credential).
            ctx: The transport context of the current call.
        """
        self._spawn_reply(self._reply_machine_action(ctx.endpoint, cmd))

    async def _reply_machine_action(self, endpoint: Any, cmd: dict[str, Any]) -> None:
        """Apply the ``machineAction`` *cmd* and send the fresh ``machines`` reply to *endpoint*.

        An exception the action lets through (``machine_action`` turns
        only ``MachineError`` into its error text) becomes the reply's
        ``error`` instead of killing this background task unanswered.
        """
        action = str(cmd["action"])
        try:
            async with self._machine_lock:
                error = await asyncio.to_thread(
                    sidebar_panels.machine_action,
                    action, str(cmd["name"]), str(cmd.get("address") or ""),
                    str(cmd.get("machinePassword") or ""),
                )
        except Exception as exc:  # noqa: BLE001 - a background task must not die silently
            logger.warning("machineAction %s failed", action, exc_info=True)
            error = str(exc) or type(exc).__name__
        await self._reply_machines(endpoint, True, action, error, "machineAction")

    async def get_machine_work_dirs(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Send a client the working directories of a registered machine.

        The chat header's WORKDIR drop-down of a tab bound to a machine
        lists that machine's directories: its daemon's working
        directory and the directories selected there so far; the
        task-info panel shows the directory and the machine's default
        task budget for a chat headed there whose task has no settings
        yet.  The direct reply is ``{"type": "machineWorkDirs",
        "machine": <name>, "workDir": <dir>, "recentWorkDirs":
        [{"path", "ts"}, ...], "maxBudget": <number or null>, "error":
        ""}`` from :func:`kiss.server.sidebar_panels.machine_work_dirs`
        (an unreachable or unknown machine answers with ``error``).
        The machine round trip runs in a background task so it never
        holds up the connection's other commands.

        Args:
            cmd: The ``getMachineWorkDirs`` command (``machine``).
            ctx: The transport context of the current call.
        """
        self._spawn_reply(self._reply_machine_work_dirs(
            ctx.endpoint, str(cmd["machine"]), None, "getMachineWorkDirs",
        ))

    async def _reply_machine_work_dirs(
        self, endpoint: Any, machine: str, work_dir: str | None, command: str,
    ) -> None:
        """Send *endpoint* the ``machineWorkDirs`` of *machine* (after adopting *work_dir* there).

        Args:
            endpoint: The requesting client's transport endpoint.
            machine: The registered machine's name.
            work_dir: The directory the machine's daemon adopts first,
                or ``None`` to only read; the reply is then stamped
                ``set: true``.
            command: The command being answered (for the reply log).
        """
        try:
            async with self._machine_lock:
                if work_dir is None:
                    reply = await asyncio.to_thread(sidebar_panels.machine_work_dirs, machine)
                else:
                    reply = await asyncio.to_thread(
                        sidebar_panels.machine_work_dirs, machine, work_dir,
                    )
                    reply["set"] = True
                await self._backend._reply_direct(
                    endpoint, {"type": "machineWorkDirs", **reply}, command,
                )
        except Exception:  # noqa: BLE001 - a background task must not die silently
            logger.warning("%s reply failed", command, exc_info=True)

    async def set_work_dir(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Make ``workDir`` the global working directory -- here or on a machine.

        Without ``machine`` the backend adopts the directory
        (``_cmd_set_work_dir``: persisted, ``workDirChanged``
        broadcast).  With ``machine`` (the chat header's pick for a tab
        bound to that machine) the directory is one of the machine's:
        its daemon adopts it, and the direct reply is the machine's
        fresh ``machineWorkDirs`` (as for ``getMachineWorkDirs``, with
        ``set: true``; ``error`` names a machine that could not be
        reached) -- this daemon's own working directory is untouched.

        Args:
            cmd: The ``setWorkDir`` command (``workDir``; optional
                ``machine``).
            ctx: The transport context of the current call.
        """
        machine = str(cmd.get("machine") or "")
        if not machine:
            await self.forward(cmd, ctx)
            return
        self._spawn_reply(self._reply_machine_work_dirs(
            ctx.endpoint, machine, str(cmd["workDir"]), "setWorkDir",
        ))

    async def get_spend_report(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Send a client the task history's spend for its Spend subpanel.

        The right sidebar's "Spend" subpanel polls this command for its
        all-time totals, daily cost heatmap and
        cost-by-model bars.  The direct reply is ``{"type":
        "spendReport", ...}`` carrying the ``total``, ``days``,
        ``totalByModel`` and ``daysByModel`` fields of
        :func:`kiss.server.sidebar_panels.spend_report`.

        Args:
            cmd: The ``getSpendReport`` command (no fields).
            ctx: The transport context of the current call.
        """
        report = await asyncio.to_thread(sidebar_panels.spend_report)
        await self._backend._reply_direct(
            ctx.endpoint, {"type": "spendReport", **report}, "getSpendReport"
        )

    async def get_recommendations(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Send a client the suggested tasks for its Recommendations subpanel.

        The right sidebar's "Recommendations" subpanel polls this
        command for the five tasks the last
        ``/recommendations`` run suggested.  The direct reply is
        ``{"type": "recommendations", "generatedAt": ms, "tasks":
        [...]}`` from :func:`kiss.server.sidebar_panels.recommendations_report`.

        Args:
            cmd: The ``getRecommendations`` command (no fields).
            ctx: The transport context of the current call.
        """
        report = await asyncio.to_thread(sidebar_panels.recommendations_report)
        await self._backend._reply_direct(
            ctx.endpoint, {"type": "recommendations", **report}, "getRecommendations"
        )

    async def list_dir(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """List a directory for the remote webapp's Explorer view.

        The remote webapp's task-history panel carries an activity bar
        whose Explorer view browses the workspace: opening
        the view lists the work dir, expanding a folder lists that
        folder, and clicking a file goes through ``openFile``.  The
        reply is a ``dirListing`` event sent to the requester only.

        With ``machine`` (the Explorer, folder picker and the pick
        check of the chat header's WORKDIR drop-down of a tab bound to
        that machine) the listing is the machine's, from its daemon:
        :meth:`dispatch` forwards it before this handler runs
        (:data:`kiss.server.sidebar_panels.MACHINE_WORKSPACE_REPLIES`).

        Args:
            cmd: The ``listDir`` command (optional ``path``,
                ``workDir``, ``tabId``, ``token``, ``machine``).
            ctx: The transport context of the current call.
        """
        await self._backend._handle_list_dir(cmd, ctx.endpoint)

    # The remote webapp's terminal tabs (``kiss.server.terminal_tab``):
    # a shell on the daemon's machine, owned by the connection that
    # opened it.  The ``tab_id`` is the terminal tab, never a chat.

    async def terminal_open(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Start (or re-attach) the shell of a terminal tab.

        Args:
            cmd: The ``terminalOpen`` command (``tab_id``; optional
                ``cols``, ``rows``, ``workDir``).
            ctx: The transport context of the current call.
        """
        # Off the event loop: opening forks the shell (a fork of the
        # whole daemon process), during which nothing else would be
        # served — no fan-out to any connection, no other dispatch.
        await asyncio.to_thread(
            self._backend._agent_server.terminals.open,
            str(cmd["tab_id"]),
            ctx.conn_state["conn_id"],
            self._backend._cmd_work_dir(cmd),
            cmd.get("cols"),
            cmd.get("rows"),
        )

    async def terminal_input(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Write the keystrokes of a terminal tab to its shell.

        Args:
            cmd: The ``terminalInput`` command (``tab_id``, ``data``).
            ctx: The transport context of the current call.
        """
        self._backend._agent_server.terminals.input(
            str(cmd["tab_id"]), ctx.conn_state["conn_id"], cmd["data"],
        )

    async def terminal_resize(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Resize the pty of a terminal tab.

        Args:
            cmd: The ``terminalResize`` command (``tab_id``, ``cols``,
                ``rows``).
            ctx: The transport context of the current call.
        """
        self._backend._agent_server.terminals.resize(
            str(cmd["tab_id"]), ctx.conn_state["conn_id"], cmd["cols"], cmd["rows"],
        )

    async def terminal_close(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Hang up the shell of a closed terminal tab.

        Args:
            cmd: The ``terminalClose`` command (``tab_id``).
            ctx: The transport context of the current call.
        """
        self._backend._agent_server.terminals.close(
            str(cmd["tab_id"]), ctx.conn_state["conn_id"],
        )

    async def git_status(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Report working-tree changes for the remote Source Control view.

        The activity bar's Source Control view lists the repository's
        staged, unstaged and untracked changes (VS Code's "Changes"
        section) from this command's ``gitStatus`` reply, sent to the
        requester only.

        Args:
            cmd: The ``gitStatus`` command (optional ``workDir``,
                ``tabId``, ``token``).
            ctx: The transport context of the current call.
        """
        await self._backend._handle_git_status(cmd, ctx.endpoint)

    async def git_log(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Report recent commits for the remote Source Control graph.

        The activity bar's Source Control view draws a commit graph
        (VS Code's "Graph" section) with each commit's modified files
        from this command's ``gitLog`` reply, sent to the requester
        only.

        Args:
            cmd: The ``gitLog`` command (optional ``workDir``,
                ``tabId``, ``token``, ``limit``).
            ctx: The transport context of the current call.
        """
        await self._backend._handle_git_log(cmd, ctx.endpoint)

    async def git_show(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Serve a commit's patch / a file at a commit / a revision diff.

        The remote Source Control graph's commit context menu ("Open
        Changes", "Open File", "Compare with...") reads its text from
        this command's ``gitShow`` reply, sent to the requester only.

        Args:
            cmd: The ``gitShow`` command (``sha``, optional ``path``,
                ``base``, ``mode``, ``workDir``, ``tabId``, ``token``).
            ctx: The transport context of the current call.
        """
        await self._backend._handle_git_show(cmd, ctx.endpoint)

    async def git_action(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Run a commit context-menu git action for the remote graph.

        "Checkout (Detached)", "Create Branch...", "Create Tag..." and
        "Cherry Pick" of the remote Source Control graph's commit menu
        each send one ``gitAction``; the outcome comes back as a
        ``gitActionResult`` to the requester only.

        Args:
            cmd: The ``gitAction`` command (``action``, ``sha``,
                optional ``name``, ``message``, ``workDir``, ``tabId``,
                ``token``).
            ctx: The transport context of the current call.
        """
        await self._backend._handle_git_action(cmd, ctx.endpoint)

    async def fs_action(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Run an Explorer context-menu file action for the remote webapp.

        New File..., New Folder..., Rename..., Delete, Paste, Find in
        Folder... and Compare Selected of the remote Explorer's context
        menu each send one ``fsAction``; the outcome comes back as an
        ``fsResult`` to the requester only.

        Args:
            cmd: The ``fsAction`` command (``action``, ``path``,
                optional ``dest``, ``name``, ``query``, ``overwrite``,
                ``workDir``, ``tabId``, ``token``).
            ctx: The transport context of the current call.
        """
        await self._backend._handle_fs_action(cmd, ctx.endpoint)

    async def share_chat(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Write a chat page's transcript as a standalone HTML page.

        The chat page serialized the highlighted tab's static task
        panel and event panels (its ``shareChat`` command carries the
        markup) and asks the daemon to save them as
        ``reports/chat-<title-slug>-<chatId>.html`` under the tab's work
        dir.  The reply is a direct ``share_done`` event to the
        requester.

        Args:
            cmd: The ``shareChat`` command (``chatId``, ``html``,
                optional ``title``, ``workDir``, ``tabId``).
            ctx: The transport context of the current call.
        """
        await self._backend._handle_share_chat(cmd, ctx.endpoint)

    async def share_chat_tasks(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Send the requester every task of a chat for a share export.

        The chat page's share button exports the whole chat, but
        after a reload its DOM holds only one task's transcript — the
        daemon's session replay repaints a single task.  This command
        returns the persisted transcripts of ALL of the chat's tasks
        (oldest first) as a direct ``share_tasks`` reply; the page
        replays them into detached containers, splices in the live DOM
        for the on-screen task, and sends the assembled page back via
        ``shareChat``.

        Args:
            cmd: The ``shareChatTasks`` command (``chatId``, optional
                ``tabId``, optional ``taskId`` — a sub-agent tab's
                share narrows the reply to that one task and its
                sub-agents).
            ctx: The transport context of the current call.
        """
        await self._backend._handle_share_chat_tasks(cmd, ctx.endpoint)

    async def get_history(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Answer a CHATS panel page in the background, in the connection's order.

        The page merges every registered machine's history
        (:func:`kiss.agents.sorcar.machines.merged_history`): a second
        or more over the machines' links, up to their timeout when one
        is down.  Each connection's commands are dispatched one at a
        time, so producing it inline would hold up a ``submit`` or
        ``stop`` sent right after it, and the page would itself wait
        behind the slow forwards in front of it (a transcript's
        ``checkPaths`` on a remote machine, a panel poll).  The
        connection's pages are answered in the order they were asked
        for: a later page of a view extends the earlier one.

        Args:
            cmd: The ``getHistory`` command.
            ctx: The transport context of the current call.
        """
        self._spawn_after(
            self._pending_history, ctx.conn_state["conn_id"], self._forward_detached(cmd),
        )

    async def voice_transcribe(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Transcribe a browser's post-wake utterance.

        The chat page heard the "Abracadabra" wake word and captured the
        utterance that followed.  The audio is translated by gpt-audio
        (:func:`kiss.server.speech.transcribe_pcm`) and answered with
        the ``voiceSpeech`` message ``voice.js`` handles.  The
        transcription is a network round-trip (and, on first use, the
        speaker model's download), so it runs in the background: a
        ``stop`` or ``submit`` the page sends meanwhile is not held up
        behind it.  The page's dictations are answered in the order
        they were sent, though: ``voice.js`` pairs each ``voiceSpeech``
        with the oldest of its outstanding rounds (the conversation
        the words were spoken into), so a short dictation answered
        before a longer earlier one would put each transcript into the
        other's conversation.

        Args:
            cmd: The ``voiceTranscribe`` command carrying the audio.
            ctx: The transport context of the current call.
        """
        self._spawn_after(
            self._pending_voice, ctx.conn_state["conn_id"],
            self._backend._handle_voice_transcribe(cmd, ctx.endpoint),
        )

    async def active_tasks_query(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Report in-flight agent tasks back to the requesting client.

        Args:
            cmd: The ``activeTasksQuery`` command (unused).
            ctx: The transport context of the current call.
        """
        await self._backend._handle_active_tasks_query(ctx.endpoint)

    async def get_welcome_info(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Send the welcome-screen info (the active remote URL) to the asking client.

        Args:
            cmd: The ``getWelcomeInfo`` command (unused).
            ctx: The transport context of the current call (its
                connection id addresses the reply).
        """
        await self._backend._send_welcome_info(ctx.conn_state["conn_id"])

    async def run_update(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Run the KISS Sorcar installer to update the checkout.

        Args:
            cmd: The ``runUpdate`` command (unused).
            ctx: The transport context of the current call; supplies
                the requesting ``conn_id`` so acknowledgement
                notifications reach only the requesting window.
        """
        await self._backend._handle_run_update(ctx.conn_state["conn_id"])

    async def update_models(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Refresh the user-local model catalog (``~/.kiss/MODEL_INFO.json``).

        Services the settings panel's "Update Models" button: the daemon
        runs ``kiss.scripts.update_models --model-info`` against the
        user-local catalog copy as a detached subprocess.

        Args:
            cmd: The ``updateModels`` command (unused).
            ctx: The transport context of the current call; supplies
                the requesting ``conn_id`` so acknowledgement
                notifications reach only the requesting window.
        """
        await self._backend._handle_update_models(ctx.conn_state["conn_id"])

    async def snooze_update(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Snooze the update notification for 24 hours.

        Services the "Remind me later" action of the update toast:
        records the snooze in the update-check cache and rebroadcasts
        the ``update_available`` state so every client's toast
        disappears.

        Args:
            cmd: The ``snoozeUpdate`` command; its optional ``latest``
                field names the release being snoozed.
            ctx: The transport context of the current call (unused —
                the resulting rebroadcast must reach every window).
        """
        latest = cmd.get("latest")
        await self._backend._handle_snooze_update(
            latest if isinstance(latest, str) else "",
        )

    async def tips_opt_out(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Persist, or forget, the "Don't show tips again" choice.

        Services the tips window's checkbox.  The choice is the marker
        file ``$KISS_HOME/TIPS_DISABLED`` that ``tips_data`` reads, so
        a choice made in one browser holds in every browser.  ``optOut``
        ``false`` (checkbox unticked again) removes the marker; absent
        or any other value opts out.

        Args:
            cmd: The ``tipsOptOut`` command with an optional boolean
                ``optOut``.
            ctx: The transport context of the current call (unused).
        """
        opt_out = cmd.get("optOut") is not False
        await asyncio.to_thread(_write_tips_opt_out_marker, opt_out)

    async def dismiss_notice(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Service a toast's "Do not show this again" checkbox.

        Only the "VS Code extension removed" notice offers the box
        today; its dismissal is a marker file under ``$KISS_HOME``, so
        the choice holds in every browser of this daemon.  Every window
        then gets a ``close`` for the toast.  An unknown id is ignored.

        Args:
            cmd: The ``dismissNotice`` command with the notification ``id``.
            ctx: The transport context of the current call (unused —
                the close must reach every window).
        """
        if cmd.get("id") != VSCODE_RETIRED_NOTICE_ID:
            return
        await asyncio.to_thread(dismiss_vscode_retired_notice)
        self._backend._printer.broadcast(
            {"type": "notification", "id": VSCODE_RETIRED_NOTICE_ID, "close": True},
        )

    def _signin_reply(self, conn_id: str) -> Callable[[dict[str, Any], bool], None]:
        """A :data:`kiss.server.signin.Reply` for the window of *conn_id*."""
        printer = self._backend._printer

        def reply(event: dict[str, Any], everyone: bool) -> None:
            broadcast_to_conn(printer, event, "" if everyone else conn_id)

        return reply

    async def signin_login(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Service a click on a provider button of the sign-in window.

        Starts the brand's ``$KISS_HOME/signin login <provider>``
        (:class:`kiss.server.signin.SigninLogin`); the window gets
        ``signinUrl`` as soon as the hook prints the page to open, every
        window ``signinDone`` when the sign-in succeeds, the clicking
        window ``signinError`` when it fails.

        Args:
            cmd: The ``signinLogin`` command with the ``provider`` id.
            ctx: The transport context of the current call; supplies
                the ``conn_id`` the URL and errors are stamped with.
        """
        reply = self._signin_reply(ctx.conn_state["conn_id"])
        await asyncio.to_thread(self.signin.start, str(cmd["provider"]), reply)

    async def signin_paste(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Forward a pasted redirect URL or code to the running sign-in hook.

        For a browser on another machine, whose redirect to the hook's
        loopback port cannot land.  Without a running login the window
        gets a ``signinError`` saying to start over.

        Args:
            cmd: The ``signinPaste`` command with the pasted ``text``.
            ctx: The transport context of the current call.
        """
        if not await asyncio.to_thread(self.signin.paste, str(cmd["text"])):
            error = "No sign-in is in progress: click a button to start one."
            self._signin_reply(ctx.conn_state["conn_id"])(
                {"type": "signinError", "error": error}, False
            )

    async def signin_cancel(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """End the running sign-in hook: the window was closed with "Not now".

        Args:
            cmd: The ``signinCancel`` command (unused).
            ctx: The transport context of the current call (unused).
        """
        await asyncio.to_thread(self.signin.cancel)

    async def update_when_idle(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Arm (or cancel) an update that runs once no task is running.

        Services the "Update when idle" action of the update toast:
        the daemon polls its agent registry and launches ``install.sh``
        the first time no task is in flight.  The ``update_available``
        state is rebroadcast with ``pendingIdle`` so every chat window's
        toast reflects the armed state.

        Args:
            cmd: The ``updateWhenIdle`` command; ``cancel: true`` disarms
                a pending idle update instead of arming one.
            ctx: The transport context of the current call (unused —
                the resulting rebroadcast must reach every window).
        """
        await self._backend._handle_update_when_idle(cancel=cmd.get("cancel") is True)

    async def ping(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Answer a client's ordering probe with a direct ``pong``.

        A connection's commands are dispatched one after another, so
        the ``pong`` reaches the sender only once every command it sent
        before the ``ping`` has been taken.  The chat page's
        transport (``media/transport.js``) relies
        on that: after a (re)connect it flushes the commands queued
        while the socket was down, sends ``ping`` and keeps the batch
        until ``pong`` arrives, re-sending it on the next connection if
        this one dies first.  Another daemon's
        :class:`~kiss.agents.sorcar.machines.MachineLink` pings this
        one the same way; the ``pong`` names this machine (its short
        host name, ``machine``) so that daemon can label this machine
        by its name rather than by the address it was registered at.

        Args:
            cmd: The ``ping`` command (no fields are used).
            ctx: The transport context of the current call; the reply
                goes to this sender only.
        """
        await self._backend._reply_direct(
            ctx.endpoint, {"type": "pong", "machine": local_machine_name()}, "ping",
        )

    async def auth(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Discard an ``auth`` frame on an already-authenticated connection.

        The real handshake is :meth:`authenticate`, run before the
        dispatch loop starts; a repeated frame (a client re-sending its
        queued batch after a reconnect) carries nothing to act on.

        Args:
            cmd: The ``auth`` command (unused).
            ctx: The transport context of the current call (unused).
        """

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
        from kiss.viz_trajectory.server import discover_job_dirs, load_job_trajectories

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
        # Load from the ALREADY-validated directory: with ``job_dir``
        # given, ``load_job_trajectories`` reads it as is and does not
        # re-resolve the name via ``find_job_dir`` (primary-root
        # preference, follows symlinks), which would discard the
        # containment check above.
        body = json.dumps(
            load_job_trajectories(jobs_root, job_name, job_dir=job_dir)
        ).encode("utf-8")
        return (200, "application/json", body)
