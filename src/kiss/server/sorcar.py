# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The Sorcar server API and a minimal synchronous client for it.

This module is the single source of truth for the wire API of the
``kiss-web`` daemon and hosts both of its Python ends:

**The server API** — :data:`API`, :func:`validate_command`, and
:class:`ServerApi` define every command a user interface (a VS Code
window, the remote webapp, or a Python client) may send to the
daemon.  Local and remote clients speak the same JSON commands over
the daemon's WSS listener, dispatched on the ``"type"`` field, one
object per WebSocket frame.  The daemon routes every command through
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
handshake, itself a catalog command.

**The client API** — :func:`run` lets any process launch a task on an
already-running daemon and block until it finishes::

    from kiss.server import sorcar

    result = sorcar.run("Summarize README.md", work_dir="/path/to/repo")
    print(result.text, result.success, result.cost, result.tokens, result.steps)
    print(result.chat_id, result.task_id)  # daemon chat session / task row ids

    # Continue the same chat (the agent sees the prior task as context):
    follow_up = sorcar.run("Now fix the typos you found", chat_id=result.chat_id)

``extension_agent_path="/path/to/my_agent.py"`` names an *agent
script* — a Sorcar Extension Agent (SEA) — whose ``settings()`` dict
computes the run's parameters on the daemon — e.g. a ``"model"`` key
overrides *model*, a ``"prompt"`` key overrides *prompt* — while
parameters it leaves out keep the values passed to :func:`run` (see
the :func:`run` docstring for the script format).  The script is also
the only way to give the agent extra tools: its ``add_to_tools()``
returns functions (plain synchronous functions with keyword-bindable,
type-annotated parameters and Google-style docstrings) that are added
to the built-in toolset; with ``"tool_profile": "none"`` they and
``finish`` become the whole toolset.  The client never serializes
Python functions — the daemon loads the script itself, so the tools
execute **in the daemon process** like native agent tools::

    # my_agent.py
    def get_temperature(city: str) -> str:
        \"\"\"Return the current temperature of a city.

        Args:
            city: Name of the city to look up.
        \"\"\"
        return lookup_sensor(city)

    def add_to_tools():
        \"\"\"Return the tools added to the built-in toolset.\"\"\"
        return [get_temperature]

    result = sorcar.run("What's the temperature in Paris?",
                        extension_agent_path="my_agent.py")

The script may additionally define ``llm_call_hook()`` /
``tool_call_hook()``, returning functions ``llm_call_hook`` and
``tool_call_hook`` that the daemon passes to the underlying
:class:`kiss.core.kiss_agent.KISSAgent` (see
:meth:`~kiss.core.kiss_agent.KISSAgent.run`); like the tool getters,
these have no :func:`run` parameter, since a callable cannot travel
the wire.

The function speaks the daemon's JSON protocol over its local WSS
endpoint, found through ``$KISS_HOME/sorcar-local.json`` (or the file
``$KISS_SORCAR_LOCAL`` names) — the same channel the VS Code extension
uses.  The endpoint file's mode 0600 restricts the local token it
carries to the owning user, so no password is involved.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

# The synchronous client half of this API lives in the sorcar layer
# (``kiss.agents.sorcar.daemon_client``) so sorcar-layer code — the
# ``run_agent`` dispatch tool and the cron scheduler — can submit
# tasks without importing ``kiss.server`` (the layering invariant in
# ``kiss.tests.agents.sorcar.test_layering_invariants``).  It is
# re-exported here unchanged: ``kiss.server.sorcar.run`` and
# ``kiss.server.sorcar.TaskResult`` stay the public client API.
from kiss.agents.sorcar.daemon_client import (
    _MAX_LINE_BYTES as _MAX_LINE_BYTES,
)
from kiss.agents.sorcar.daemon_client import (
    TaskResult as TaskResult,
)
from kiss.agents.sorcar.daemon_client import (
    _parse_cost as _parse_cost,
)
from kiss.agents.sorcar.daemon_client import (
    _resolve_endpoint_file as _resolve_endpoint_file,
)
from kiss.agents.sorcar.daemon_client import (
    _to_task_result as _to_task_result,
)
from kiss.agents.sorcar.daemon_client import (
    run as run,
)
from kiss.core.config import kiss_home
from kiss.core.utils import is_root_dir
from kiss.core.vscode_config import load_config
from kiss.server import sidebar_panels
from kiss.server.tips import TIPS_OPT_OUT_MARKER

logger = logging.getLogger(__name__)

# In-flight ``appsStatus`` replies (see ServerApi.get_apps_status): the
# event loop keeps only weak references to tasks.
_APPS_STATUS_REPLIES: set[asyncio.Task[None]] = set()

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
    ApiCommand("interruptTool", required=("tabId",)),
    ApiCommand("userAnswer", required=("answer",)),
    ApiCommand("newChat"),
    ApiCommand("openTab", required=("tabId",)),
    ApiCommand("getTabsState"),
    ApiCommand("closeTab", required=("tabId",)),
    ApiCommand("resumeSession", handler="resume_session"),
    ApiCommand("ready", handler="ready"),
    ApiCommand("getHistory"),
    ApiCommand("getAdjacentTask", required=("direction",)),
    ApiCommand("getFrequentTasks"),
    ApiCommand("deleteFrequentTask", required=("task",)),
    ApiCommand("setFavorite", required=("taskId", "isFavorite")),
    ApiCommand("getInputHistory"),
    ApiCommand("getSeaCommands"),
    ApiCommand("getWelcomeInfo", handler="get_welcome_info"),
    ApiCommand("activeTasksQuery", handler="active_tasks_query"),
    ApiCommand("getModels"),
    ApiCommand("selectModel", required=("model",)),
    ApiCommand("getConfig"),
    ApiCommand("saveConfig", required=("config",)),
    ApiCommand("getMyModels"),
    ApiCommand("saveMyModel", required=("name",)),
    ApiCommand("deleteMyModel", required=("name",)),
    ApiCommand("addTrick", required=("text",)),
    ApiCommand("deleteTrick", required=("text",)),
    ApiCommand("editTrick", required=("text", "newText")),
    ApiCommand("getDefaultModel", handler="get_default_model"),
    ApiCommand("readKissConfig", handler="read_kiss_config"),
    ApiCommand(
        "writeKissConfig", required=("config",), handler="write_kiss_config"
    ),
    ApiCommand("voiceWakeStart", handler="voice_wake_start"),
    ApiCommand("voiceWakeStop", handler="voice_wake_stop"),
    ApiCommand("browserOpen"),
    ApiCommand("browserClose", required=("tab_id",)),
    ApiCommand("browserNavigate", required=("tab_id", "action")),
    ApiCommand("browserInput", required=("tab_id", "event")),
    ApiCommand("browserViewport", required=("tab_id",)),
    ApiCommand("setWorkDir", required=("workDir",)),
    ApiCommand("getFiles", required=("prefix",)),
    ApiCommand("recordFileUsage", required=("path",)),
    ApiCommand("openFile", required=("path",), handler="open_file"),
    ApiCommand(
        "saveFile", required=("path", "content"), handler="save_file"
    ),
    ApiCommand("checkPaths", required=("paths",), handler="check_paths"),
    ApiCommand("getTaskUpdate", handler="get_task_update"),
    ApiCommand("getCronJobs", handler="get_cron_jobs"),
    ApiCommand("getAppsStatus", handler="get_apps_status"),
    ApiCommand("getSpendReport", handler="get_spend_report"),
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
    ApiCommand("worktreeAction", required=("action",)),
    ApiCommand("mainTreeAction", required=("action",)),
    ApiCommand("generateCommitMessage"),
    ApiCommand("autocommitAction"),
    ApiCommand("auth", required=("password",), handler="drop"),
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
    ApiCommand("voiceToggle", required=("enabled",), handler="drop"),
    ApiCommand("voiceSensitivity", required=("value",), handler="drop"),
    ApiCommand("voiceAck", handler="drop"),
    ApiCommand("voiceDropped", required=("text",), handler="drop"),
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


AuthKind = Literal["local", "remote"]
"""How a connection authenticated: local token or remote password."""


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
            daemon's local token from a loopback address (a VS Code
            window or a local Python client), ``False`` for a remote
            browser client that supplied the password.
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
    _vscode_server: Any

    async def _endpoint_send(self, endpoint: Any, data: str) -> None: ...

    async def _reply_direct(
        self, endpoint: Any, reply: dict[str, Any], what: str,
    ) -> None: ...

    async def _run_cmd(self, cmd: dict[str, Any]) -> None: ...

    async def _handle_open_file(
        self, cmd: dict[str, Any], endpoint: Any, is_local: bool = False,
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

    async def _handle_get_default_model(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None: ...

    async def _handle_read_kiss_config(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None: ...

    async def _handle_write_kiss_config(
        self, cmd: dict[str, Any], endpoint: Any,
    ) -> None: ...

    async def _handle_voice_wake_start(
        self, cmd: dict[str, Any], endpoint: Any, conn_id: str,
    ) -> None: ...

    async def _handle_voice_wake_stop(self, conn_id: str) -> None: ...

    async def _handle_active_tasks_query(self, endpoint: Any) -> None: ...

    def _sanitized_restored_tabs(
        self, cmd: dict[str, Any],
    ) -> list[dict[str, str]]: ...

    async def _handle_ready(
        self, cmd: dict[str, Any], websocket: Any,
    ) -> None: ...

    async def _handle_submit(
        self,
        cmd: dict[str, Any],
        endpoint: Any = None,
        is_local: bool = False,
    ) -> None: ...

    async def _send_welcome_info(self) -> None: ...

    async def _handle_run_update(self, conn_id: str = "") -> None: ...

    async def _handle_update_models(self, conn_id: str = "") -> None: ...

    async def _handle_snooze_update(self, latest: str = "") -> None: ...

    async def _handle_update_when_idle(self, cancel: bool = False) -> None: ...

    async def _handle_server_reset(self, conn_id: str = "") -> None: ...

    def _client_ip(self, websocket: Any) -> str: ...

    def _peer_is_loopback(self, connection: Any) -> bool: ...

    def _auth_lock_remaining(self, ip: str) -> float: ...

    def _record_auth_failure(self, ip: str) -> None: ...

    @property
    def local_token(self) -> str: ...

    @property
    def local_only(self) -> bool: ...


class ServerApi:
    """The Sorcar server's code-level API.

    The actual code API every client command invokes.  Each catalog
    entry in :data:`API` names (via ``ApiCommand.handler``) the method
    of this class that services it, so the clients — the VS Code
    extension (``src/SorcarApi.ts``), the chat webview / remote
    webapp (``media/api.js``), and the Python
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
        2. Canonicalises a string ``tabId`` (surrounding whitespace
           stripped) and writes it back into *cmd*, so the registry,
           the agent-state registry, the printer and the cleanup
           tail all key the tab identically.
        3. Validates *cmd* against the catalog
           (:func:`validate_command`) and answers an invalid command
           with a direct ``error`` event to the sender only.
        4. Records the command's ``tabId`` in the connection's
           bookkeeping (:meth:`_record_tab`).
        5. Stamps the connection's ``conn_id`` as ``connId`` —
           overwriting any client-supplied value so it cannot be
           spoofed — which keys the backend's per-connection
           autocomplete state.
        6. Blanks a ``workDir`` naming a filesystem root (``/``,
           ``C:\\`` — see :func:`kiss.core.utils.is_root_dir`) so it is
           treated exactly like an absent one: a client whose cwd
           degenerated to the root can never make the daemon's single
           global working directory the whole disk, nor run a task
           there.  A command without a ``workDir`` resolves to that
           global directory (``config.json`` ``work_dir``, set from any
           surface's "Working directory" panel via ``setWorkDir``).
        7. Invokes the :class:`ServerApi` method named by the
           command's catalog entry.

        Args:
            cmd: The parsed JSON command dictionary (the transport
                guarantees a dict).
            ctx: The transport context of this call.
        """
        if cmd.get("type") in DROPPED_COMMANDS:
            return
        tab_id = cmd.get("tabId", "")
        if isinstance(tab_id, str):
            # Canonicalise ONCE, before validation and every handler:
            # ``TabRegistry`` strips ids it stores and broadcasts, so
            # a handler keying ``AgentState`` / ``_tab_chat_views`` /
            # local-tab counts by the raw string gave one wire tab two
            # identities, and closing the canonical id leaked the rest.
            tab_id = tab_id.strip()
            cmd["tabId"] = tab_id
        error = validate_command(cmd)
        if error:
            reply: dict[str, Any] = {"type": "error", "text": error}
            if isinstance(tab_id, str) and tab_id:
                reply["tabId"] = tab_id
            await self._backend._endpoint_send(ctx.endpoint, json.dumps(reply))
            return
        if isinstance(tab_id, str) and tab_id:
            self._record_tab(tab_id, ctx)
        cmd["connId"] = ctx.conn_state["conn_id"]
        name = cmd["type"]
        handler = API[name].handler
        raw_wd = cmd.get("workDir")
        if isinstance(raw_wd, str) and is_root_dir(raw_wd):
            # A filesystem root is never a real workspace: it arrives
            # only from a client that inherited the root as its cwd (a
            # Dock-launched VS Code window with no folder open) or from
            # a tab whose persisted registry entry was poisoned by one.
            # Blank it HERE — the one chokepoint every transport
            # shares — so a root can neither become the global working
            # directory via ``setWorkDir`` nor root a task or the
            # @-mention file scan at the whole disk.  The command then
            # falls back to the daemon's configured folder.
            cmd["workDir"] = ""
        await getattr(self, handler)(cmd, ctx)

    def _record_tab(self, tab_id: str, ctx: ApiContext) -> None:
        """Record *tab_id* as touched by this connection.

        For local peers, records the connection's INTEREST in the
        id with the printer's local-tab bookkeeping (talk-playback
        arbitration).  Interest is recorded before the handler runs
        and regardless of whether the tab currently exists: the talk
        fan-out decides "shown" at talk time from the canonical facts
        (``VSCodeServer._local_tab_shown`` — registry membership plus
        an attached webview for registry tabs; for every other tab
        this interest plus either the tab's own non-closed agent
        state or, for a viewer without a state of its own, its live
        task subscription), so a stale record for a closed tab is
        inert and a record made for a ``run_agent`` ``api-…`` tab or
        a sub-agent viewer is what lets that tab count.  The
        connection's ``local_tabs`` set is mutated only inside the
        printer, under its own lock.

        Args:
            tab_id: The non-empty frontend tab identifier.
            ctx: The transport context of the current call.
        """
        if ctx.is_local:
            self._backend._printer.register_local_tab(
                ctx.conn_state["conn_id"],
                tab_id,
                ctx.conn_state.setdefault("local_tabs", set()),
            )

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

    async def _refuse_no_password_remote(
        self, websocket: Any, ip: str,
    ) -> None:
        """Refuse a non-loopback peer while no password is configured.

        Sends the explanatory ``error`` event and closes the socket
        (both best-effort).  Shared by :meth:`authenticate`'s
        pre-handshake gate and its per-attempt re-check.

        Args:
            websocket: The remote client's WebSocket connection.
            ip: The client's rate-limit key, for the log line only.
        """
        logger.warning(
            "Refusing non-localhost auth handshake from %s: "
            "remote_password is empty", ip,
        )
        try:
            await websocket.send(json.dumps({
                "type": "error",
                "code": "localhost_only",
                "text": "Remote access is turned off: no remote "
                        "password is set, so only this computer may "
                        "connect. On it, open Settings and set a "
                        "Remote password.",
            }))
            await websocket.close()
        except Exception:
            pass

    async def authenticate(self, websocket: Any) -> AuthKind | None:
        """Authenticate a WSS client with the ``auth`` handshake.

        Every client's entry point into the API: before a connection
        may issue any catalog command, its very first frames must
        complete this handshake.  The remote webapp's ``_WS_SHIM_JS``
        shim sends ``{"type": "auth", "password": ...}`` as soon as
        the socket opens; local clients (the VS Code extension, Python
        clients) send ``{"type": "auth", "token": ...}`` with the
        daemon's per-start local token read from the endpoint file
        (:mod:`kiss.agents.sorcar.local_endpoint`), whose 0600 mode gates the
        token to the owning user.

        Protocol serviced here, in order:

        0. An ``auth`` frame carrying a ``token`` equal to the
           backend's ``local_token`` (constant-time compare) from a
           LOOPBACK peer is answered with ``auth_ok`` (``"local":
           true``) and returns ``"local"``.  A token from a
           non-loopback peer is never honoured, whatever the password
           setting; a wrong token counts as a wrong guess.

        1. A source IP that is still rate-limited after too many
           failed logins is answered with ``auth_locked`` (carrying
           ``retry_after`` seconds) and closed — telling the client
           WHY instead of leaving its loading overlay spinning.
        2. When the configured ``remote_password`` is empty, a
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
            ``"local"`` for a token-authenticated loopback client,
            ``"remote"`` for a password-authenticated client, ``None``
            when authentication failed (the socket is then already
            closed).
        """
        backend = self._backend
        ip = backend._client_ip(websocket)
        lock_remaining = backend._auth_lock_remaining(ip)
        if lock_remaining > 0.0:
            # The lockout is per IP, and every visitor relayed by the
            # tunnel shares 127.0.0.1 with the extension and the Python
            # clients: read one frame so a valid local token still gets
            # through, then refuse everything else.
            logger.warning("Auth rate-limit hit for %s; closing socket", ip)
            try:
                msg = json.loads(await asyncio.wait_for(websocket.recv(), timeout=30))
                if self._is_local_auth(msg, websocket):
                    await websocket.send(json.dumps({"type": "auth_ok", "local": True}))
                    return "local"
            except Exception:
                logger.debug("Locked peer %s sent no usable frame", ip, exc_info=True)
            try:
                await websocket.send(json.dumps({
                    "type": "auth_locked",
                    "retry_after": math.ceil(lock_remaining),
                }))
                await websocket.close()
            except Exception:
                pass
            return None
        password = (await asyncio.to_thread(load_config)).get("remote_password", "")
        if not password and not backend._peer_is_loopback(websocket):
            # Defense in depth for the empty-password localhost-only
            # lockdown (primary gate: the transport's
            # ``_process_request`` refuses non-loopback peers with 403
            # before the WS upgrade).  Re-applied per attempt below,
            # so clearing the password at ANY point before a frame is
            # examined refuses an already-admitted non-loopback peer.
            await self._refuse_no_password_remote(websocket, ip)
            return None
        try:
            for is_retry, timeout in ((False, 30), (True, 60)):
                raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                msg = json.loads(raw)
                # The local token first: it is a 256-bit secret the
                # lockout below exists to protect passwords from, and
                # the lockout must never shut the machine's own
                # clients out (see the pre-recv check above).
                if self._is_local_auth(msg, websocket):
                    await websocket.send(json.dumps({
                        "type": "auth_ok", "local": True,
                    }))
                    return "local"
                if backend.local_only:
                    # The private daemon admits no password at all.
                    await websocket.send(json.dumps({
                        "type": "error",
                        "code": "auth_failed",
                        "text": "This daemon accepts local clients only.",
                    }))
                    await websocket.close()
                    return None
                # Re-load the configured password before every compare
                # so a change made while this connection awaited
                # credentials takes effect NOW.  Without the reload, a
                # stale snapshot taken at handshake start would keep
                # accepting the OLD password — and, worse, clearing the
                # password would not subject an already-admitted
                # non-loopback peer to the localhost-only rule below.
                password = (
                    await asyncio.to_thread(load_config)
                ).get("remote_password", "")
                if not password and not backend._peer_is_loopback(
                    websocket,
                ):
                    await self._refuse_no_password_remote(websocket, ip)
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
                    await websocket.send(json.dumps({
                        "type": "auth_locked",
                        "retry_after": math.ceil(lock_remaining),
                    }))
                    await websocket.close()
                    return None
                client_pw = msg.get("password", "")
                if not isinstance(client_pw, str):
                    client_pw = ""
                token = msg.get("token", "")
                if not isinstance(token, str):
                    token = ""
                if msg.get("type") == "auth" and not token and passwords_equal(
                    password, client_pw,
                ):
                    await websocket.send(json.dumps({
                        "type": "auth_ok", "local": False,
                    }))
                    return "remote"
                if not is_retry and msg.get("type") != "auth":
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
                    await websocket.send(json.dumps({
                        "type": "auth_locked",
                        "retry_after": math.ceil(lock_remaining),
                    }))
                    await websocket.close()
                    return None
                if not is_retry:
                    await websocket.send(json.dumps({"type": "auth_required"}))
            # ``code`` lets the webapp shim tell this apart from other
            # pre-auth errors (it shows the text inside the password
            # dialog and keeps the dialog open across the close below).
            await websocket.send(json.dumps({
                "type": "error",
                "code": "auth_failed",
                "text": "That password is not correct. Try again.",
            }))
            await websocket.close()
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
        own sanitize pass finds nothing left to reject or truncate.
        For a local connection it then (1) marks the connection as
        hosting a chat webview — every attached webview mirrors the
        whole canonical tab registry from ``tabs_state``, so this flag
        is what makes a registry tab's talk play natively on this
        machine (webviews cannot autoplay), including for background
        tabs the client adopts from the snapshot and tabs other clients
        publish later; and (2) RECONCILES the connection's per-tab
        interest to exactly the tabs the client announced (its own tab
        and the restored tabs), which bounds the interest a connection
        accumulates for tabs closed while it was attached.  No registry
        snapshot is copied into the bookkeeping: the talk fan-out reads
        the registry itself at decision time
        (``VSCodeServer._local_tab_shown``), so there is nothing that
        a close racing this ``ready`` could leave stale.  The sync
        updates the connection's ``local_tabs`` set in place, so
        disconnect cleanup is unchanged.  Finally fans the command out
        through the backend's ready handler (models / input history /
        config / session replay).

        Args:
            cmd: The ``ready`` command.
            ctx: The transport context of the current call.
        """
        cmd["restoredTabs"] = self._backend._sanitized_restored_tabs(cmd)
        if ctx.is_local:
            conn_id = ctx.conn_state["conn_id"]
            self._backend._printer.mark_local_webview(conn_id)
            shown = {rt["tabId"] for rt in cmd["restoredTabs"] if rt["tabId"]}
            own_tab = cmd.get("tabId")
            if isinstance(own_tab, str) and own_tab:
                shown.add(own_tab)
            self._backend._printer.sync_local_tabs(
                conn_id,
                shown,
                ctx.conn_state.setdefault("local_tabs", set()),
            )
        await self._backend._handle_ready(cmd, ctx.endpoint)

    async def submit(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Start a task from a webview ``submit``.

        The one submit path of every surface: the remote webapp sends
        its webview's ``submit`` over WSS, the VS Code extension host
        forwards its webview's ``submit`` over the local WSS endpoint,
        and the backend translates both into a ``run`` (path resolution,
        follow-up routing) including the path-only shortcut: a prompt
        that is just the path of an existing file is answered on the
        submitting connection — the resolved path for a VS Code window
        to open natively, the ``fileContent`` for a browser — and
        starts no task.

        Args:
            cmd: The ``submit`` command.
            ctx: The transport context of the current call; its
                endpoint receives a path-only prompt's reply.
        """
        await self._backend._handle_submit(cmd, ctx.endpoint, ctx.is_local)

    async def open_file(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Resolve, and for a browser serve, a clicked file link.

        A client clicked a file or directory link in a chat webview.
        The daemon resolves the path once for every surface (``~``, the
        tab's work dir, its pending worktree).  A browser has no editor
        to open the path in, so it gets the file's content (or a
        plain-text directory listing) for an in-page content tab; a
        local client (a VS Code window on the token-authenticated
        loopback WSS endpoint) gets the resolved path as an
        ``openResolvedFile`` action that its extension host opens in a
        real editor tab.

        Args:
            cmd: The ``openFile`` command.
            ctx: The transport context of the current call.
        """
        await self._backend._handle_open_file(cmd, ctx.endpoint, ctx.is_local)

    async def save_file(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Write a remote-web client's edits back to a file on disk.

        The remote webapp opens a file (``openFile``) in an editable
        Monaco editor inside a content tab; Ctrl/Cmd+S or the tab's
        Save button sends the editor's full text here.  The file must
        already exist (the editor never creates files) and is replaced
        atomically; the ``version`` stamp taken from the ``fileContent``
        reply lets the daemon refuse to overwrite a file that changed
        on disk since it was opened unless ``force`` is set.  The reply
        is a ``fileSaved`` event sent to the requester only.  Local
        clients (VS Code windows) edit files in real editor tabs, so
        a locally delivered ``saveFile`` is dropped as a defensive no-op.

        Args:
            cmd: The ``saveFile`` command (``path``, ``content``,
                optional ``workDir``, ``tabId``, ``token``, ``version``,
                ``force``).
            ctx: The transport context of the current call.
        """
        if ctx.is_local:
            return
        await self._backend._handle_save_file(cmd, ctx.endpoint)

    async def check_paths(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Report which file paths exist to the requesting client.

        The chat webview linkifies file-path-looking strings in event
        panel contents lazily: a path only becomes a clickable link
        after this check confirms that clicking it (``openFile``)
        would actually open something — a file or a directory.  Served
        to remote browsers and local clients (VS Code windows) alike,
        with the same resolution ``openFile`` applies.

        Args:
            cmd: The ``checkPaths`` command.
            ctx: The transport context of the current call.
        """
        await self._backend._handle_check_paths(cmd, ctx.endpoint)

    async def get_task_update(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Report the ``/ask`` agent's task update to a task-info panel.

        The remote webapp's task-info panel (docked on desktop, a
        drawer on mobile) and the VS Code extension's chat editor
        panels (editor-tabs mode, whose updates fill the secondary
        sidebar's Task Info view) poll this command while the visible
        tab's task runs: the info subpanel shows the
        :mod:`~kiss.agents.seas.ask.ask_sea` agent's short answer to
        what that task has done so far and its partial results.  The
        first poll once the task is a minute old runs the agent (as a
        sub-agent of the task, in the task's chat; its cost counts
        towards the task), later polls re-run it every 10 minutes, and
        a poll with ``refresh: true`` (the panel's refresh button)
        re-runs it at once.  The direct ``taskUpdate`` reply goes back
        to the asking endpoint, local or remote.

        Args:
            cmd: The ``getTaskUpdate`` command (``tabId``, optional
                ``knownSig``, ``token``, ``refresh``).
            ctx: The transport context of the current call.
        """
        await self._backend._handle_get_task_update(cmd, ctx.endpoint)

    async def get_cron_jobs(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Send a client the scheduled cron jobs for its Schedule subpanel.

        The right sidebar's "Schedule" subpanel (every surface: remote
        webapp, VS Code sidebar chat, editor-tabs Task Info view) polls
        this command.  The direct reply, to whichever endpoint (local
        or remote) asked, is ``{"type": "cronJobs", "jobs": [...]}`` with the
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
        task = asyncio.create_task(
            self._reply_apps_status(ctx.endpoint, bool(cmd.get("refresh")))
        )
        _APPS_STATUS_REPLIES.add(task)
        task.add_done_callback(_APPS_STATUS_REPLIES.discard)

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

    async def get_spend_report(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Send a client the task history's spend for its Spend subpanel.

        The right sidebar's "Spend" subpanel (every surface: remote
        webapp, VS Code sidebar chat, editor-tabs Task Info view) polls
        this command for its all-time totals, daily cost heatmap and
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

    async def list_dir(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """List a directory for the remote webapp's Explorer view.

        The remote webapp's task-history panel carries a VS Code-like
        activity bar whose Explorer view browses the workspace: opening
        the view lists the work dir, expanding a folder lists that
        folder, and clicking a file goes through ``openFile``.  The
        reply is a ``dirListing`` event sent to the requester only.
        Local clients (VS Code windows) have a real Explorer, so a
        locally delivered ``listDir`` is dropped as a defensive no-op,
        exactly like ``saveFile``.

        Args:
            cmd: The ``listDir`` command (optional ``path``,
                ``workDir``, ``tabId``, ``token``).
            ctx: The transport context of the current call.
        """
        if ctx.is_local:
            return
        await self._backend._handle_list_dir(cmd, ctx.endpoint)

    async def git_status(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Report working-tree changes for the remote Source Control view.

        The activity bar's Source Control view lists the repository's
        staged, unstaged and untracked changes (VS Code's "Changes"
        section) from this command's ``gitStatus`` reply, sent to the
        requester only.  A locally delivered ``gitStatus`` is dropped as a
        defensive no-op, exactly like ``saveFile``.

        Args:
            cmd: The ``gitStatus`` command (optional ``workDir``,
                ``tabId``, ``token``).
            ctx: The transport context of the current call.
        """
        if ctx.is_local:
            return
        await self._backend._handle_git_status(cmd, ctx.endpoint)

    async def git_log(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Report recent commits for the remote Source Control graph.

        The activity bar's Source Control view draws a commit graph
        (VS Code's "Graph" section) with each commit's modified files
        from this command's ``gitLog`` reply, sent to the requester
        only.  A locally delivered ``gitLog`` is dropped as a defensive
        no-op, exactly like ``saveFile``.

        Args:
            cmd: The ``gitLog`` command (optional ``workDir``,
                ``tabId``, ``token``, ``limit``).
            ctx: The transport context of the current call.
        """
        if ctx.is_local:
            return
        await self._backend._handle_git_log(cmd, ctx.endpoint)

    async def git_show(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Serve a commit's patch / a file at a commit / a revision diff.

        The remote Source Control graph's commit context menu ("Open
        Changes", "Open File", "Compare with...") reads its text from
        this command's ``gitShow`` reply, sent to the requester only.
        A locally delivered ``gitShow`` is dropped as a defensive no-op,
        exactly like ``gitLog``.

        Args:
            cmd: The ``gitShow`` command (``sha``, optional ``path``,
                ``base``, ``mode``, ``workDir``, ``tabId``, ``token``).
            ctx: The transport context of the current call.
        """
        if ctx.is_local:
            return
        await self._backend._handle_git_show(cmd, ctx.endpoint)

    async def git_action(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Run a commit context-menu git action for the remote graph.

        "Checkout (Detached)", "Create Branch...", "Create Tag..." and
        "Cherry Pick" of the remote Source Control graph's commit menu
        each send one ``gitAction``; the outcome comes back as a
        ``gitActionResult`` to the requester only.  A locally delivered
        ``gitAction`` is dropped as a defensive no-op (VS Code windows
        run the real Git extension).

        Args:
            cmd: The ``gitAction`` command (``action``, ``sha``,
                optional ``name``, ``message``, ``workDir``, ``tabId``,
                ``token``).
            ctx: The transport context of the current call.
        """
        if ctx.is_local:
            return
        await self._backend._handle_git_action(cmd, ctx.endpoint)

    async def fs_action(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Run an Explorer context-menu file action for the remote webapp.

        New File..., New Folder..., Rename..., Delete, Paste, Find in
        Folder... and Compare Selected of the remote Explorer's context
        menu each send one ``fsAction``; the outcome comes back as an
        ``fsResult`` to the requester only.  A locally delivered
        ``fsAction`` is dropped as a defensive no-op (VS Code windows
        have the real Explorer).

        Args:
            cmd: The ``fsAction`` command (``action``, ``path``,
                optional ``dest``, ``name``, ``query``, ``overwrite``,
                ``workDir``, ``tabId``, ``token``).
            ctx: The transport context of the current call.
        """
        if ctx.is_local:
            return
        await self._backend._handle_fs_action(cmd, ctx.endpoint)

    async def share_chat(self, cmd: dict[str, Any], ctx: ApiContext) -> None:
        """Write a chat webview's transcript as a standalone HTML page.

        The chat webview serialized the highlighted tab's static task
        panel and event panels (its ``shareChat`` command carries the
        markup) and asks the daemon to save them as
        ``reports/chat-<title-slug>-<chatId>.html`` under the tab's work
        dir.  Both
        transports take this path — the VS Code extension host
        forwards the webview's ``shareChat``, the remote
        webapp sends it over WSS — so the page is built in exactly one
        place.  The reply is a direct ``share_done`` event to the
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

        The chat webview's share button exports the whole chat, but
        after a reload its DOM holds only one task's transcript — the
        daemon's session replay repaints a single task.  This command
        returns the persisted transcripts of ALL of the chat's tasks
        (oldest first) as a direct ``share_tasks`` reply; the webview
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

    async def voice_transcribe(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Transcribe a remote-web client's post-wake utterance.

        A remote-web (browser mode) client heard the "Hey Sorcar" wake
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

    async def get_default_model(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Reply with the daemon's key-derived default model name.

        Services ``getDefaultModel`` so the VS Code extension host can
        obtain :func:`kiss.core.models.model_info.get_default_model`
        over the socket instead of spawning a throwaway ``uv run
        python -c ...`` interpreter (its historical out-of-band
        channel, still used as a fallback while the daemon is down).
        The reply is a direct ``defaultModel`` event to the requester.

        Args:
            cmd: The ``getDefaultModel`` command.
            ctx: The transport context of the current call.
        """
        await self._backend._handle_get_default_model(cmd, ctx.endpoint)

    async def read_kiss_config(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Serve the raw merged ``~/.kiss/config.json`` to a local client.

        Services ``readKissConfig`` so the extension host can read the
        daemon-owned config file through the socket instead of parsing
        the file itself.  The reply is a direct ``kissConfig`` event.

        LOCAL CLIENTS ONLY: unlike ``getConfig`` (whose reply is
        shaped for the settings panel), this returns the config
        verbatim — including ``remote_password`` — so a remote
        browser must never receive it.  A command from a remote,
        password-authenticated connection is dropped as a defensive
        no-op.

        Args:
            cmd: The ``readKissConfig`` command.
            ctx: The transport context of the current call.
        """
        if not ctx.is_local:
            return
        await self._backend._handle_read_kiss_config(cmd, ctx.endpoint)

    async def write_kiss_config(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Merge a local client's keys into ``~/.kiss/config.json``.

        Services ``writeKissConfig`` so the extension host can update
        daemon-owned config keys (e.g. ``remote_password``) through
        the socket — sharing the daemon's atomic, lock-guarded
        :func:`kiss.core.vscode_config.save_config` write path —
        instead of rewriting the file itself.  The reply is a direct
        ``kissConfigSaved`` acknowledgement event.

        LOCAL CLIENTS ONLY: a remote browser must not be
        able to change ``remote_password`` or any other daemon
        setting through this raw channel; a command from a remote,
        password-authenticated connection is dropped as a defensive
        no-op.

        Args:
            cmd: The ``writeKissConfig`` command carrying ``config``.
            ctx: The transport context of the current call.
        """
        if not ctx.is_local:
            return
        await self._backend._handle_write_kiss_config(cmd, ctx.endpoint)

    async def voice_wake_start(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Start the daemon-hosted wake-word listener for this client.

        Services ``voiceWakeStart`` so the extension host can run
        :mod:`kiss.server.voice_wake` as a daemon child over the
        socket — receiving its protocol as ``voiceWakeEvent`` /
        ``voiceWakeState`` events — instead of spawning the listener
        process itself and parsing its stdout (its historical
        out-of-band channel).  The optional ``sensitivity`` field
        (0..100) tunes wake-word eagerness.  The listener is bound to
        this connection and stopped on disconnect.

        LOCAL CLIENTS ONLY: the listener captures this
        machine's microphone, so a remote browser must not
        control it (browser-mode voice capture stays in-page via
        ``voiceTranscribe``); a command from a remote,
        password-authenticated connection is dropped as a defensive
        no-op.

        Args:
            cmd: The ``voiceWakeStart`` command.
            ctx: The transport context of the current call.
        """
        if not ctx.is_local:
            return
        await self._backend._handle_voice_wake_start(
            cmd, ctx.endpoint, ctx.conn_state["conn_id"],
        )

    async def voice_wake_stop(
        self, cmd: dict[str, Any], ctx: ApiContext,
    ) -> None:
        """Stop this client's daemon-hosted wake-word listener.

        Services ``voiceWakeStop``; a no-op when the connection has no
        running listener.  LOCAL CLIENTS ONLY, matching
        ``voiceWakeStart``.

        Args:
            cmd: The ``voiceWakeStop`` command (unused).
            ctx: The transport context of the current call.
        """
        if not ctx.is_local:
            return
        await self._backend._handle_voice_wake_stop(
            ctx.conn_state["conn_id"],
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
        """Broadcast the welcome-screen info (the active remote URL).

        Args:
            cmd: The ``getWelcomeInfo`` command (unused).
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

        Services the "Remind me later" action of the update toast in
        both frontends: records the snooze in the update-check cache
        shared with the VS Code extension and rebroadcasts the
        ``update_available`` state so every client's toast disappears.

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

        Services the tips window's checkbox on both surfaces (the VS
        Code host forwards the webview's ``tipsOptOut`` here).  The
        choice is the marker file ``$KISS_HOME/TIPS_DISABLED`` that
        ``tips_data`` reads, so a choice made on one surface holds on
        every surface.  ``optOut``
        ``false`` (checkbox unticked again) removes the marker; absent
        or any other value opts out.

        Args:
            cmd: The ``tipsOptOut`` command with an optional boolean
                ``optOut``.
            ctx: The transport context of the current call (unused).
        """
        opt_out = cmd.get("optOut") is not False
        await asyncio.to_thread(_write_tips_opt_out_marker, opt_out)

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
        before the ``ping`` has been taken.  The remote webapp's
        WebSocket shim (``kiss.server.web_server._WS_SHIM_JS``) relies
        on that: after a (re)connect it flushes the commands queued
        while the socket was down, sends ``ping`` and keeps the batch
        until ``pong`` arrives, re-sending it on the next connection if
        this one dies first.

        Args:
            cmd: The ``ping`` command (no fields are used).
            ctx: The transport context of the current call; the reply
                goes to this sender only.
        """
        await self._backend._endpoint_send(
            ctx.endpoint, json.dumps({"type": "pong"}),
        )

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
