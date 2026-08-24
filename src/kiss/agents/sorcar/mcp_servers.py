# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""MCP (Model Context Protocol) server support for Sorcar.

Implements MCP server management with Claude Code compatibility:

* **Configuration** — servers are stored as ``{"mcpServers": {...}}``
  JSON, the same shape Claude Code uses, and are discovered from
  (low → high precedence; later wins on a name clash):

  - ``~/.kiss/mcp.json`` (respecting ``KISS_HOME``) — user servers.
  - ``<work_dir>/.mcp.json`` — Claude Code's project file, so MCP
    servers checked into a repo for Claude Code work unchanged.
  - ``<work_dir>/.kiss/mcp.json`` — native project servers.

  Each entry is ``{"type": "stdio"|"http"|"sse", "command": ...,
  "args": [...], "env": {...}}`` for local servers or ``{"type":
  "http"|"sse", "url": ..., "headers": {...}}`` for remote ones
  (``type`` defaults to ``stdio`` when ``command`` is present, else
  ``http`` — matching Claude Code's leniency).

* **Tools** — every tool of every configured server is exposed to the
  agent as a function named ``<server>_<tool>`` whose signature and
  docstring are synthesized from the MCP ``inputSchema``, so the
  standard kiss schema builder produces a faithful OpenAI tool schema.

* **Permission wildcards** — the ``mcp_permissions`` key in
  ``~/.kiss/config.json`` maps wildcard patterns to ``"allow"`` or
  ``"deny"`` (e.g. ``{"*": "allow", "mymcp_*": "deny"}``).  Patterns
  are matched against the full ``<server>_<tool>`` name with the
  *last* matching rule winning (OpenCode semantics); denied tools are
  never registered.

* **OAuth** — remote (``http``/``sse``) servers authenticate through
  the MCP SDK's OAuth 2.1 provider using tokens persisted per server
  under ``~/.kiss/mcp_auth/`` by :class:`FileTokenStorage`.  Agent runs
  reuse and refresh those tokens; there is deliberately no interactive
  browser login (it would block a run on a human), so a server that
  needs one fails with a hint to provision its tokens by hand.

Connections are kept alive for the life of the process by a single
:class:`MCPManager` running an asyncio loop on a daemon thread; each
server's transport + session is opened and closed inside one long-lived
task (anyio cancel scopes must enter/exit in the same task).
"""

from __future__ import annotations

import asyncio
import atexit
import functools
import hashlib
import inspect
import json
import keyword
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.persistence import _default_kiss_dir
from kiss.agents.sorcar.skills import load_permission_rules, skill_permission
from kiss.agents.sorcar.useful_tools import _file_lock

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT = 60.0
_CONNECT_STRAGGLER_GRACE_S = 5.0
CALL_TIMEOUT = 300.0

#: How long a connection may sit unused before the next :meth:`connect`
#: reaps it.  Nothing in the long-lived daemon ever disconnects servers
#: between tasks, so without this every project x config revision would
#: leak one stdio child process for the life of the process.
IDLE_TIMEOUT = 600.0

#: Hard cap on live connections, so a burst of distinct configurations
#: cannot exhaust the machine's process/file-descriptor limits before the
#: idle timeout comes round.  The least recently used one goes first.
MAX_CONNECTIONS = 8

#: How often an idle connection is pinged.  A server that died (OOM kill,
#: crash, container restart) is otherwise never noticed: the connection
#: task is parked on its stop event and every later tool call would block
#: on a dead transport until ``CALL_TIMEOUT``.
HEALTH_INTERVAL = 30.0

_JSON_TO_PY: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


@dataclass(frozen=True)
class MCPServerConfig:
    """One configured MCP server.

    Attributes:
        name: Unique server name (used to namespace its tools).
        transport: ``"stdio"``, ``"http"``, or ``"sse"``.
        command: Executable for stdio servers (empty otherwise).
        args: Command-line arguments for stdio servers.
        env: Extra environment variables for stdio servers.
        url: Endpoint URL for http/sse servers (empty otherwise).
        headers: Extra HTTP headers for http/sse servers.
        source: Where the server was configured — ``"user"``,
            ``"claude-project"``, or ``"project"``.  Pure bookkeeping:
            excluded from equality so the same server re-discovered
            from a different file compares equal and its healthy
            connection is reused instead of torn down and re-opened.
    """

    name: str
    transport: str = "stdio"
    command: str = ""
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    url: str = ""
    headers: tuple[tuple[str, str], ...] = ()
    source: str = field(default="user", compare=False)

    def to_json(self) -> dict[str, Any]:
        """Return the Claude-Code-compatible JSON dict for this server."""
        out: dict[str, Any] = {"type": self.transport}
        if self.transport == "stdio":
            out["command"] = self.command
            if self.args:
                out["args"] = list(self.args)
            if self.env:
                out["env"] = dict(self.env)
        else:
            out["url"] = self.url
            if self.headers:
                out["headers"] = dict(self.headers)
        return out


def user_mcp_config_path() -> Path:
    """Return the user-level MCP config file (``~/.kiss/mcp.json``)."""
    return _default_kiss_dir() / "mcp.json"


def project_mcp_config_path(work_dir: str) -> Path:
    """Return the project-level MCP config file (``.kiss/mcp.json``)."""
    return Path(work_dir) / ".kiss" / "mcp.json"


def claude_project_mcp_config_path(work_dir: str) -> Path:
    """Return Claude Code's project MCP file (``<work_dir>/.mcp.json``)."""
    return Path(work_dir) / ".mcp.json"


def mcp_auth_dir() -> Path:
    """Return the directory holding per-server OAuth token files."""
    return _default_kiss_dir() / "mcp_auth"


def _parse_server_entry(name: str, raw: Any, source: str) -> MCPServerConfig | None:
    """Parse one ``mcpServers`` JSON entry leniently.

    Args:
        name: The server name (the JSON key).
        raw: The JSON value (must be a dict to be usable).
        source: Discovery source label (e.g. ``"user"``).

    Returns:
        The parsed config, or ``None`` when the entry is unusable.
    """
    if not isinstance(raw, dict):
        logger.debug("mcp server %s: entry is not a dict; skipping", name)
        return None
    command = str(raw.get("command", "") or "")
    url = str(raw.get("url", "") or "")
    transport = str(raw.get("type", "") or raw.get("transport", "") or "").lower()
    if transport not in ("stdio", "http", "sse"):
        transport = "stdio" if command else "http"
    if transport == "stdio" and not command:
        logger.debug("mcp server %s: stdio without command; skipping", name)
        return None
    if transport in ("http", "sse") and not url:
        logger.debug("mcp server %s: %s without url; skipping", name, transport)
        return None
    args = raw.get("args") or []
    env = raw.get("env") or {}
    headers = raw.get("headers") or {}
    return MCPServerConfig(
        name=name,
        transport=transport,
        command=command,
        args=tuple(str(a) for a in args) if isinstance(args, list) else (),
        env=tuple((str(k), str(v)) for k, v in env.items())
        if isinstance(env, dict) else (),
        url=url,
        headers=tuple((str(k), str(v)) for k, v in headers.items())
        if isinstance(headers, dict) else (),
        source=source,
    )


def _load_config_file(path: Path, source: str) -> dict[str, MCPServerConfig]:
    """Load every server from one ``{"mcpServers": {...}}`` file."""
    servers: dict[str, MCPServerConfig] = {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return servers
    except (OSError, ValueError):
        logger.warning("unreadable MCP config: %s", path, exc_info=True)
        return servers
    entries = raw.get("mcpServers") if isinstance(raw, dict) else None
    if not isinstance(entries, dict):
        return servers
    for name, entry in entries.items():
        cfg = _parse_server_entry(str(name), entry, source)
        if cfg is not None:
            servers[cfg.name] = cfg
    return servers


def load_mcp_servers(work_dir: str) -> dict[str, MCPServerConfig]:
    """Load all configured MCP servers visible from *work_dir*.

    Load order (low → high precedence; later wins on a name clash):
    user (``~/.kiss/mcp.json``), Claude Code project (``.mcp.json``),
    native project (``.kiss/mcp.json``).

    Args:
        work_dir: The project directory whose servers to include.

    Returns:
        Mapping of server name → :class:`MCPServerConfig`.
    """
    servers = _load_config_file(user_mcp_config_path(), "user")
    servers.update(
        _load_config_file(
            claude_project_mcp_config_path(work_dir), "claude-project"
        )
    )
    servers.update(_load_config_file(project_mcp_config_path(work_dir), "project"))
    return servers


def save_mcp_server(cfg: MCPServerConfig, scope: str, work_dir: str) -> Path:
    """Persist *cfg* in the user- or project-scope MCP config file.

    Args:
        cfg: The server configuration to save.
        scope: ``"user"`` (``~/.kiss/mcp.json``) or ``"project"``
            (``<work_dir>/.kiss/mcp.json``).
        work_dir: The project directory (used for project scope).

    Returns:
        The path of the config file written.
    """
    path = (
        user_mcp_config_path() if scope == "user"
        else project_mcp_config_path(work_dir)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(path.with_suffix(".lock")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        entries = raw.setdefault("mcpServers", {})
        if not isinstance(entries, dict):  # pragma: no cover - corrupt file guard
            entries = {}
            raw["mcpServers"] = entries
        entries[cfg.name] = cfg.to_json()
        _atomic_write_config(path, raw)
    return path


def remove_mcp_server(name: str, work_dir: str) -> list[Path]:
    """Remove server *name* from every writable MCP config file.

    Args:
        name: The server name to remove.
        work_dir: The project directory (for the project-scope files).

    Returns:
        The config files the server was actually removed from.
    """
    removed: list[Path] = []
    for path in (
        user_mcp_config_path(),
        claude_project_mcp_config_path(work_dir),
        project_mcp_config_path(work_dir),
    ):
        with _file_lock(path.with_suffix(".lock")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            entries = raw.get("mcpServers") if isinstance(raw, dict) else None
            if isinstance(entries, dict) and name in entries:
                del entries[name]
                _atomic_write_config(path, raw)
                removed.append(path)
    return removed


def _atomic_write_config(path: Path, raw: dict[str, Any]) -> None:
    """Atomically replace the MCP config *path* with *raw* as JSON.

    An interrupted in-place ``write_text`` could leave invalid JSON;
    write-to-temp + ``os.replace`` cannot.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_mcp_permissions() -> dict[str, str]:
    """Load the ``mcp_permissions`` rules from ``~/.kiss/config.json``.

    Returns:
        Mapping of wildcard pattern → ``"allow"``/``"deny"``, in file
        order.  Empty when the config or key is missing/malformed.
    """
    return load_permission_rules("mcp_permissions")


def mcp_tool_permission(tool_name: str, rules: dict[str, str]) -> str:
    """Resolve the permission for the full *tool_name* against *rules*.

    Rules use shell-style wildcards matched against the complete
    ``<server>_<tool>`` name (so ``mymcp_*`` covers every tool of the
    ``mymcp`` server); the **last** matching rule wins and the default
    is ``"allow"`` — identical semantics to skill permissions.

    Args:
        tool_name: The full tool name (e.g. ``"mymcp_search"``).
        rules: Mapping of pattern → ``"allow"``/``"deny"``.

    Returns:
        ``"allow"`` or ``"deny"``.
    """
    return skill_permission(tool_name, rules)


_RESERVED_BASENAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def _token_file_name(server_name: str) -> str:
    """Return an injective, filesystem-safe token file name.

    Plain ``_sanitize`` maps distinct names such as ``a/b`` and
    ``a b`` to the same ``a_b``, which would make two servers share
    (and leak) each other's OAuth credentials.  Only all-lowercase
    already-safe names keep their historical file name: macOS and
    Windows filesystems are typically case-insensitive, so ``GitHub``
    and ``github`` would otherwise alias one credential file.  Every
    other name gets a short digest of the exact original appended,
    which keeps distinct names distinct even after case folding.
    Windows reserved device basenames (``con``, ``nul``, ...) are
    prefixed as well.
    """
    safe = _sanitize(server_name)
    reserved = safe.lower() in _RESERVED_BASENAMES
    if safe == server_name and server_name == server_name.lower() and not reserved:
        return f"{safe}.json"
    digest = hashlib.sha256(server_name.encode("utf-8")).hexdigest()[:12]
    prefix = "s_" if reserved else ""
    return f"{prefix}{safe}.{digest}.json"


class FileTokenStorage:
    """MCP SDK :class:`~mcp.client.auth.TokenStorage` backed by a JSON file.

    Tokens and the dynamically registered client information for one
    server are stored together at ``~/.kiss/mcp_auth/<server>.json``
    with mode ``0600``.
    """

    def __init__(self, server_name: str) -> None:
        self.path = mcp_auth_dir() / _token_file_name(server_name)
        self._lock_path = self.path.with_suffix(".lock")

    def _read(self) -> dict[str, Any]:
        """Read the stored JSON payload (empty dict when absent/bad)."""
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, data: dict[str, Any]) -> None:
        """Atomically write *data*, owner-only from the very first byte.

        The temp file is created mode ``0600`` *before* any content is
        written, and ``os.replace`` swaps it in atomically — so the
        credentials are never observable with wider permissions and a
        crash can never leave a partial file.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            # mkdir applies the mode only on creation; repair a
            # pre-existing world-readable auth dir too.
            os.chmod(self.path.parent, 0o700)
        except OSError:  # pragma: no cover — permission error
            logger.debug("could not chmod auth dir", exc_info=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(data, indent=2) + "\n")
            os.replace(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    async def get_tokens(self) -> Any:
        """Return the stored :class:`~mcp.shared.auth.OAuthToken`, if any."""
        from mcp.shared.auth import OAuthToken

        raw = self._read().get("tokens")
        if not raw:
            return None
        try:
            return OAuthToken.model_validate(raw)
        except Exception:
            logger.debug("invalid stored tokens in %s", self.path, exc_info=True)
            return None

    def _locked_update(self, key: str, value: Any) -> None:
        """Replace *key* in the stored JSON under the inter-process lock.

        Args:
            key: Top-level key to set (``"tokens"``/``"client_info"``).
            value: Its already-serialized JSON value.
        """
        with _file_lock(self._lock_path):
            data = self._read()
            data[key] = value
            self._write(data)

    async def set_tokens(self, tokens: Any) -> None:
        """Persist *tokens* (an :class:`~mcp.shared.auth.OAuthToken`).

        The read-modify-write runs on a worker thread: it blocks on an
        inter-process lock that another kiss process can hold for
        minutes (an interactive OAuth login waits on a human), and this
        coroutine runs on the manager's single shared event loop, which
        also carries every other server's transport and every in-flight
        tool call.
        """
        await asyncio.to_thread(
            self._locked_update,
            "tokens",
            tokens.model_dump(mode="json", exclude_none=True),
        )

    async def get_client_info(self) -> Any:
        """Return the stored OAuth client registration, if any."""
        from mcp.shared.auth import OAuthClientInformationFull

        raw = self._read().get("client_info")
        if not raw:
            return None
        try:
            return OAuthClientInformationFull.model_validate(raw)
        except Exception:
            logger.debug("invalid client info in %s", self.path, exc_info=True)
            return None

    async def set_client_info(self, client_info: Any) -> None:
        """Persist the dynamically registered OAuth client information.

        Runs on a worker thread for the same reason as
        :meth:`set_tokens`: the lock is inter-process and the caller's
        event loop is shared by every MCP connection.
        """
        await asyncio.to_thread(
            self._locked_update,
            "client_info",
            client_info.model_dump(mode="json", exclude_none=True),
        )

    def clear(self) -> bool:
        """Delete the stored token file for this server.

        Returns:
            ``True`` when a file was deleted, ``False`` when absent.
        """
        try:
            self.path.unlink()
            return True
        except FileNotFoundError:
            return False


async def _noninteractive_redirect(url: str) -> None:
    """Refuse to start a browser OAuth flow during an agent run."""
    raise RuntimeError(
        "MCP server requires interactive OAuth login; provision its "
        "tokens under ~/.kiss/mcp_auth/ (or use a server that "
        "authenticates via --header) first."
    )


async def _noninteractive_callback() -> tuple[str, str | None]:
    """Refuse to wait for an OAuth callback during an agent run."""
    raise RuntimeError(
        "MCP server requires interactive OAuth login; provision its "
        "tokens under ~/.kiss/mcp_auth/ (or use a server that "
        "authenticates via --header) first."
    )


def build_oauth_provider(cfg: MCPServerConfig) -> Any:
    """Build the OAuth provider used to authenticate to a remote server.

    The provider refreshes and reuses the tokens stored by
    :class:`FileTokenStorage`.  It never starts an interactive login:
    an agent run must not block waiting for a human at a browser, so
    both handlers refuse and point at manual token provisioning.

    Args:
        cfg: The remote server configuration.

    Returns:
        An ``httpx.Auth`` instance (``OAuthClientProvider``).
    """
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata

    metadata = OAuthClientMetadata.model_validate({
        "client_name": "KISS Sorcar",
        "redirect_uris": ["http://localhost:0/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "client_secret_post",
    })
    return OAuthClientProvider(
        server_url=cfg.url,
        client_metadata=metadata,
        storage=FileTokenStorage(cfg.name),
        redirect_handler=_noninteractive_redirect,
        callback_handler=_noninteractive_callback,
    )


@dataclass
class _Connection:
    """Live state of one server connection owned by the manager."""

    config: MCPServerConfig
    ready: threading.Event = field(default_factory=threading.Event)
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    session: Any = None
    tools: list[Any] = field(default_factory=list)
    error: str = ""
    task: Any = None
    finished: threading.Event = field(default_factory=threading.Event)
    last_used: float = field(default_factory=time.monotonic)
    #: Tool calls currently executing on this connection.  Guarded by
    #: the manager's lock; a connection is never evicted while it is
    #: non-zero (see :meth:`MCPManager._evict_surplus`).
    in_flight: int = 0


def _child_errlog() -> Any:
    """Return a writable stream to use as an MCP child's stderr.

    Resolved fresh on every spawn.  Falls back to the process's real
    stderr file descriptor when ``sys.stderr`` is absent (pythonw) or has
    already been closed, and to ``os.devnull`` when even that fails, so a
    server can always be started.
    """
    stream = getattr(sys, "stderr", None)
    if stream is not None and not getattr(stream, "closed", False):
        return stream
    try:
        return open(os.dup(2), "w", closefd=True)  # noqa: SIM115
    except OSError:
        return open(os.devnull, "w")  # noqa: SIM115


async def _enter_transport(stack: Any, config: MCPServerConfig, auth: Any) -> tuple:
    """Open *config*'s transport on *stack* and return ``(read, write)``."""
    if config.transport == "stdio":
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=config.command,
            args=list(config.args),
            env={**os.environ, **dict(config.env)},
        )
        # ``stdio_client``'s ``errlog`` default is a *default argument*, so
        # the SDK binds whatever ``sys.stderr`` was at import time and hands
        # that object to the child as its stderr for the rest of the
        # process's life.  Anything that rebinds or closes ``sys.stderr``
        # afterwards (a log redirect in the daemon, pytest's per-test
        # capture) then makes the very next spawn die with
        # ``ValueError: I/O operation on closed file``.  Resolve it at call
        # time instead, and fall back to the real fd when the current
        # ``sys.stderr`` is missing or already closed.
        read, write = await stack.enter_async_context(
            stdio_client(params, errlog=_child_errlog())
        )
        return read, write
    if config.transport == "sse":
        from mcp.client.sse import sse_client

        read, write = await stack.enter_async_context(
            sse_client(config.url, headers=dict(config.headers) or None, auth=auth)
        )
        return read, write
    from mcp.client.streamable_http import streamablehttp_client

    read, write, _ = await stack.enter_async_context(
        streamablehttp_client(
            config.url, headers=dict(config.headers) or None, auth=auth,
        )
    )
    return read, write


def _cancel_if_not_done(task: Any) -> None:
    """Cancel *task* (a ``concurrent.futures.Future``) if still pending.

    Runs on the manager loop via ``call_later`` once the straggler
    grace period elapses; cancelling the future propagates to the
    wrapped asyncio task, raising ``CancelledError`` at its stuck
    ``await`` so the transport context unwinds and its child dies.
    """
    if not task.done():
        task.cancel()


async def _park_until_stopped(
    conn: _Connection, session: Any, health_interval: float,
) -> None:
    """Park until *conn* is stopped, failing fast if the server dies.

    A plain ``await conn.stop.wait()`` never notices a server that was
    killed, crashed, or restarted: the task stays parked, ``session``
    stays set, and every later tool call blocks on the dead transport
    until ``CALL_TIMEOUT``.  Pinging while idle turns that into a prompt
    error, which the manager turns into a reconnect.

    Args:
        conn: The connection being maintained.
        session: Its live MCP client session.
        health_interval: Seconds between pings (also the ping timeout).

    Raises:
        Exception: Whatever the failed ping raises, so the caller can
            record the error and unwind the transport.
    """
    while True:
        try:
            await asyncio.wait_for(conn.stop.wait(), timeout=health_interval)
            return
        except TimeoutError:
            await asyncio.wait_for(session.send_ping(), timeout=health_interval)


async def _maintain_connection(
    conn: _Connection, auth: Any, health_interval: float = HEALTH_INTERVAL,
) -> None:
    """Own one server connection for its whole lifetime in a single task.

    anyio cancel scopes must be entered and exited by the same task, so
    the transport and session contexts are opened here, the task then
    parks on the ``stop`` event, and the contexts unwind here too.

    Args:
        conn: The connection record to fill in and own.
        auth: Optional ``httpx.Auth`` for remote transports.
        health_interval: Seconds between idle health pings.
    """
    from contextlib import AsyncExitStack

    from mcp import ClientSession

    try:
        async with AsyncExitStack() as stack:
            read, write = await _enter_transport(stack, conn.config, auth)
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listed = await session.list_tools()
            conn.session = session
            conn.tools = list(listed.tools)
            conn.ready.set()
            await _park_until_stopped(conn, session, health_interval)
    except BaseException as exc:
        conn.error = f"{type(exc).__name__}: {exc}"
        logger.debug("MCP connection %s failed", conn.config.name, exc_info=True)
    finally:
        conn.session = None
        conn.ready.set()
        conn.finished.set()


class MCPManager:
    """Process-wide manager of live MCP server connections.

    Runs a private asyncio event loop on a daemon thread; every server
    gets one long-lived task that owns its transport + session.  The
    synchronous facade (:meth:`get_tools`, :meth:`call_tool`) is what
    the agent tool wrappers use.
    """

    _instance: MCPManager | None = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        idle_timeout_s: float = IDLE_TIMEOUT,
        max_connections: int = MAX_CONNECTIONS,
        health_interval_s: float = HEALTH_INTERVAL,
    ) -> None:
        """Start the manager's loop thread.

        Args:
            idle_timeout_s: Seconds a connection may sit unused before
                the next :meth:`connect` reaps it.
            max_connections: Hard cap on live connections; the least
                recently used one is evicted when the cap is exceeded.
            health_interval_s: Seconds between idle health pings.
        """
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="mcp-manager", daemon=True,
        )
        self._thread.start()
        self._connections: dict[str, _Connection] = {}
        # Every configuration ever connected, so a connection dropped by
        # eviction or a server crash can be rebuilt on demand.
        self._configs: dict[str, MCPServerConfig] = {}
        self._orphans: list[_Connection] = []
        self._lock = threading.Lock()
        self._shut_down = False
        self._idle_timeout_s = idle_timeout_s
        self._max_connections = max_connections
        self._health_interval_s = health_interval_s
        atexit.register(self.shutdown)

    @classmethod
    def instance(cls) -> MCPManager:
        """Return the process-wide singleton manager."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = MCPManager()
            return cls._instance

    def connect(
        self, config: MCPServerConfig, auth: Any = None,
    ) -> _Connection:
        """Connect to *config* (reusing a live connection when possible).

        Args:
            config: The server to connect to.
            auth: Optional ``httpx.Auth`` for remote servers; when
                ``None`` a non-interactive OAuth provider (stored
                tokens only) is built for http/sse servers.

        Returns:
            The connection record; ``error`` is non-empty on failure.
        """
        key = _connection_key(config)
        with self._lock:
            if self._shut_down:
                conn = _Connection(config=config)
                conn.error = "manager shut down"
                conn.ready.set()
                return conn
            self._configs[key] = config
            existing = self._connections.get(key)
            if existing is not None and existing.error == "":
                conn = existing
                conn.last_used = time.monotonic()
            else:
                if existing is not None:
                    self._loop.call_soon_threadsafe(existing.stop.set)
                if config.transport in ("http", "sse") and auth is None:
                    auth = build_oauth_provider(config)
                conn = _Connection(config=config)
                self._connections[key] = conn
                conn.task = asyncio.run_coroutine_threadsafe(
                    _maintain_connection(conn, auth, self._health_interval_s),
                    self._loop,
                )
        self._evict_surplus(key)
        if not conn.ready.wait(CONNECT_TIMEOUT):
            with self._lock:
                if self._connections.get(key) is conn:
                    del self._connections[key]
                conn.error = conn.error or "connection timed out"
                if conn.task is not None and not conn.task.done():
                    self._orphans.append(conn)
            self._loop.call_soon_threadsafe(conn.stop.set)
            self._reap_straggler(conn)
        return conn

    def _evict_surplus(self, keep: str) -> None:
        """Tear down connections that are idle or beyond the pool cap.

        Nothing else ever disconnects a server: ``shutdown`` runs only at
        interpreter exit, so in the ``kiss-web`` daemon a connection —
        and its stdio child process — would otherwise live forever.  A
        connection is only ever *dropped*, never invalidated: a later
        tool call rebuilds it from :attr:`_configs`, so eviction costs a
        reconnect at worst and never an error.

        A connection with a tool call in flight is never a candidate,
        however old it is: dropping it makes ``_maintain_connection``
        leave the session context underneath a live
        ``session.call_tool``, which strands that call until the
        five-minute call timeout expires.  Tool calls block for as long
        as the tool runs, so the busiest connection is regularly also
        the least recently *started* one.  Such a connection becomes an
        ordinary candidate again the moment its last call returns, so
        the cap still holds — it is enforced a little later.

        Args:
            keep: The connection key just connected, never evicted.
        """
        now = time.monotonic()
        with self._lock:
            evictable = {
                key for key, conn in self._connections.items()
                if key != keep and conn.in_flight == 0
            }
            doomed = {
                key for key in evictable
                if now - self._connections[key].last_used > self._idle_timeout_s
            }
            survivors = [key for key in self._connections if key not in doomed]
            surplus = len(survivors) - self._max_connections
            if surplus > 0:
                oldest = sorted(
                    (key for key in survivors if key in evictable),
                    key=lambda key: self._connections[key].last_used,
                )
                doomed.update(oldest[:surplus])
            conns = [self._connections.pop(key) for key in doomed]
        for conn in conns:
            conn.error = conn.error or "evicted: idle or over the connection cap"
            self._loop.call_soon_threadsafe(conn.stop.set)

    def _reap_straggler(self, conn: _Connection) -> None:
        """Arrange teardown of a connection evicted by a connect() timeout.

        Registers a done callback that drops *conn* from ``_orphans``
        once its task finishes (however it finishes), and schedules a
        cancellation on the manager loop after
        :data:`_CONNECT_STRAGGLER_GRACE_S` so a task stuck
        mid-handshake — which the already-set ``stop`` event cannot
        unwind — is cancelled and its transport child reaped.

        Args:
            conn: The timed-out connection whose task may be stuck.
        """
        task = conn.task
        if task is None:
            return
        task.add_done_callback(functools.partial(self._forget_orphan, conn))
        try:
            self._loop.call_soon_threadsafe(
                self._loop.call_later,
                _CONNECT_STRAGGLER_GRACE_S,
                _cancel_if_not_done,
                task,
            )
        except RuntimeError:
            pass

    def _forget_orphan(self, conn: _Connection, _future: Any) -> None:
        """Drop a finished straggler from ``_orphans`` (done callback)."""
        with self._lock:
            try:
                self._orphans.remove(conn)
            except ValueError:
                pass

    def _lease(self, server: str) -> tuple[_Connection | None, Any]:
        """Atomically look up *server*'s connection and lease it.

        The lookup and the ``in_flight`` increment happen under one
        acquisition of the manager lock: incrementing after releasing
        the lookup's lock left a window in which a concurrent
        :meth:`connect`'s :meth:`_evict_surplus` saw ``in_flight == 0``,
        evicted the connection, and tore down the session underneath
        the call about to run on it.  The session is captured under the
        same lock so the caller always uses exactly the session it
        leased.

        Args:
            server: A connection key, or a bare server name when it is
                unambiguous.

        Returns:
            ``(conn, session)`` — *session* is non-``None`` only when
            the connection is live and was leased (the caller must
            release the lease by decrementing ``in_flight``).
        """
        with self._lock:
            conn = _pick(self._connections, server)
            session = conn.session if conn is not None else None
            if conn is not None and session is not None:
                conn.in_flight += 1
                conn.last_used = time.monotonic()
            return conn, session

    def call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> str:
        """Call *tool* on *server* and return the textual result.

        Args:
            server: The connection key from :func:`_connection_key`
                (or a bare configured server name, matched when it is
                unambiguous).  Two agents may configure *different*
                servers under one conventional name (e.g. ``github``);
                keys are config-derived so one agent's calls can never
                be routed to the other agent's server.
            tool: The MCP tool name on that server.
            arguments: The tool arguments.

        Returns:
            The flattened result text (``Error: ...`` on tool errors).
        """
        display = _key_display_name(server)
        with self._lock:
            if self._shut_down:
                return (
                    f"Error: MCP server {display!r} is not connected "
                    f"(manager shut down)"
                )
        conn, session = self._lease(server)
        if session is None:
            # Evicted from the pool, or the server died mid-task: rebuild
            # it rather than failing every remaining call of the run.
            fresh = self._reconnect(server)
            if fresh is not None:
                leased, session = self._lease(server)
                conn = leased or fresh
        if conn is None or session is None:
            why = (conn.error if conn is not None else "") or "never connected"
            return f"Error: MCP server {display!r} is not connected ({why})"
        future = asyncio.run_coroutine_threadsafe(
            session.call_tool(
                tool, arguments,
                read_timeout_seconds=timedelta(seconds=CALL_TIMEOUT),
            ),
            self._loop,
        )
        try:
            result = future.result(timeout=CALL_TIMEOUT + 5)
        except Exception as exc:
            future.cancel()
            return f"Error: MCP tool call failed: {exc}"
        finally:
            with self._lock:
                conn.in_flight -= 1
                conn.last_used = time.monotonic()
        return _result_text(result)

    def _reconnect(self, server: str) -> _Connection | None:
        """Rebuild the connection for *server* from its remembered config.

        Args:
            server: A connection key, or a bare server name when it is
                unambiguous.

        Returns:
            The fresh connection, or ``None`` when the server was never
            configured (nothing to rebuild from).  A shut-down manager
            needs no special case: :meth:`connect` answers with an
            already-failed connection.
        """
        with self._lock:
            config = _pick(self._configs, server)
        if config is None:
            return None
        return self.connect(config)

    def disconnect_all(self) -> None:
        """Close every connection (their tasks unwind their contexts).

        A task still stuck mid-handshake (e.g. a stdio child that
        never speaks MCP) is not parked on its ``stop`` event, so
        setting the event cannot unwind it; such stragglers are
        cancelled after the grace period.  Either way ``conn.error``
        is stamped and ``conn.ready`` set here: a frozen task's
        ``finally`` may never run once the loop stops, and a thread
        blocked in :meth:`connect` must not burn the whole
        CONNECT_TIMEOUT waiting for a connection the manager already
        tore down.
        """
        with self._lock:
            conns = list(self._connections.values())
            self._connections.clear()
            conns.extend(self._orphans)
            self._orphans.clear()
        for conn in conns:
            self._loop.call_soon_threadsafe(conn.stop.set)
        for conn in conns:
            if conn.task is not None:
                try:
                    conn.task.result(timeout=10)
                except BaseException:  # noqa: BLE001 — CancelledError is BaseException
                    conn.task.cancel()
                    logger.debug("MCP disconnect error", exc_info=True)
                    conn.finished.wait(timeout=10)
            conn.session = None
            conn.error = conn.error or "disconnected"
            conn.ready.set()

    def shutdown(self) -> None:
        """Disconnect everything and stop the manager loop thread.

        Also resets the process-wide singleton when it points at this
        manager, so a later :meth:`instance` call builds a fresh manager
        with a live loop instead of scheduling work on a stopped one
        (which would silently never run and time out every connect).
        Idempotent: concurrent or repeated calls after the first are
        no-ops.
        """
        with MCPManager._instance_lock:
            if MCPManager._instance is self:
                MCPManager._instance = None
            if self._shut_down:
                return
            self._shut_down = True
        if not self._loop.is_running():
            return
        self.disconnect_all()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)


def _result_text(result: Any) -> str:
    """Flatten a ``CallToolResult`` into the string given to the model."""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
        else:
            block_type = getattr(block, "type", "content")
            parts.append(f"[{block_type} content omitted]")
    structured = getattr(result, "structuredContent", None)
    if structured and not parts:
        parts.append(json.dumps(structured))
    text_out = "\n".join(parts) or "(empty result)"
    if getattr(result, "isError", False):
        return f"Error: {text_out}"
    return text_out


def _sanitize(name: str) -> str:
    """Restrict *name* to characters safe in tool names and file names.

    ASCII-only: Unicode "alphanumerics" (``½``, combining marks, ...)
    are rejected by several providers' function-name grammars, so they
    become ``_`` like every other unsafe character.
    """
    return "".join(
        c if (c.isascii() and c.isalnum()) or c in "_-" else "_" for c in name
    )


def _connection_key(config: MCPServerConfig) -> str:
    """Return the manager registry key for *config*.

    Keys embed a digest of the full configuration, so two concurrent
    agents/projects that both configure a conventional name such as
    ``github`` — with different commands, URLs, or headers — get two
    isolated connections instead of silently sharing (and hijacking)
    one.
    """
    canonical = json.dumps(config.to_json(), sort_keys=True)
    digest = hashlib.sha256(f"{config.name}\n{canonical}".encode()).hexdigest()[:16]
    return f"{config.name}#{digest}"


def _key_display_name(server: str) -> str:
    """Return the configured server name behind a registry key.

    Server names may themselves contain ``#`` (``foo#bar``), so only a
    trailing ``#<16 hex>`` digest — the exact shape
    :func:`_connection_key` appends — is stripped; a bare name is
    returned unchanged.  A naive ``split("#", 1)`` would truncate
    ``foo#bar`` to ``foo``, breaking bare-name lookups and error
    messages for such servers.
    """
    base, sep, tail = server.rpartition("#")
    if sep and len(tail) == 16 and all(c in "0123456789abcdef" for c in tail):
        return base
    return server


def _pick[T](registry: dict[str, T], server: str) -> T | None:
    """Look up *server* in a key-indexed registry, tolerating a bare name.

    Args:
        registry: A mapping keyed by :func:`_connection_key`.
        server: A connection key, or a configured server name (matched
            only when exactly one key carries it).

    Returns:
        The matching entry, or ``None`` when absent or ambiguous.
    """
    entry = registry.get(server)
    if entry is not None:
        return entry
    named = [v for k, v in registry.items() if _key_display_name(k) == server]
    return named[0] if len(named) == 1 else None


def _json_schema_to_annotation(prop: Any) -> type:
    """Map one inputSchema property to the Python annotation for kiss."""
    if isinstance(prop, dict):
        ann = _JSON_TO_PY.get(str(prop.get("type", "")))
        if ann is not None:
            return ann
    return str


def _one_line(text: str) -> str:
    """Collapse *text* to a single line (kiss uses only the first line)."""
    return " ".join(str(text).split())


def _python_param_name(prop_name: str, used: set[str]) -> str:
    """Derive a valid, unique Python parameter name for a JSON property.

    JSON-schema property names may be hyphenated (``max-results``),
    start with a digit, or be Python keywords (``from``) — all invalid
    as Python parameter names.  Invalid characters become ``_``, a
    leading digit gets a ``p_`` prefix, keywords get a ``_`` suffix,
    and collisions get a numeric suffix.

    Args:
        prop_name: The original JSON property name.
        used: Names already taken (updated in place).

    Returns:
        A valid Python identifier not present in *used*.
    """
    name = "".join(
        c if (c.isascii() and c.isalnum()) or c == "_" else "_" for c in prop_name
    )
    if not name or name[0].isdigit():
        name = "p_" + name
    if keyword.iskeyword(name) or keyword.issoftkeyword(name):
        name += "_"
    base = name
    counter = 2
    while name in used:
        name = f"{base}_{counter}"
        counter += 1
    used.add(name)
    return name


def make_mcp_tool_wrapper(
    manager: MCPManager, server: str, tool: Any, connection_key: str | None = None,
) -> Any:
    """Wrap one MCP tool as a kiss-compatible Python function.

    The wrapper's ``__signature__`` and docstring are synthesized from
    the MCP ``inputSchema`` so kiss's docstring/signature-based schema
    builder reproduces the tool's parameters faithfully; calling it
    forwards to :meth:`MCPManager.call_tool`.

    Args:
        manager: The live connection manager.
        server: The configured server name.
        tool: The MCP ``Tool`` (name, description, inputSchema).
        connection_key: The :func:`_connection_key` of the live
            connection to route calls to; defaults to *server* (only
            unambiguous when a single connection has that name).

    Returns:
        The wrapper callable, named ``<server>_<tool>``.
    """
    tool_name = str(tool.name)
    full_name = f"{_sanitize(server)}_{_sanitize(tool_name)}"
    schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
    props = schema.get("properties")
    if not isinstance(props, dict):
        props = {}
    required_raw = schema.get("required")
    required = set(required_raw) if isinstance(required_raw, list) else set()

    entries: list[tuple[bool, inspect.Parameter, str]] = []
    param_map: dict[str, tuple[str, bool]] = {}
    used_names: set[str] = set()
    for prop_name in props:
        prop = props[prop_name]
        ann = _json_schema_to_annotation(prop)
        is_required = prop_name in required
        py_name = _python_param_name(str(prop_name), used_names)
        param_map[py_name] = (str(prop_name), is_required)
        param = inspect.Parameter(
            py_name,
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=inspect.Parameter.empty if is_required else None,
            annotation=ann,
        )
        desc = ""
        if isinstance(prop, dict):
            desc = _one_line(prop.get("description", "") or "")
        suffix = "" if is_required else " (optional)"
        doc_line = f"    {py_name}: {desc or 'See tool description.'}{suffix}"
        entries.append((is_required, param, doc_line))
    entries.sort(key=lambda e: not e[0])
    params = [e[1] for e in entries]
    doc_args = [e[2] for e in entries]

    def wrapper(**kwargs: Any) -> str:
        arguments: dict[str, Any] = {}
        for py_name, value in kwargs.items():
            original, is_required = param_map.get(py_name, (py_name, True))
            if value is None and not is_required:
                continue
            arguments[original] = value
        return manager.call_tool(connection_key or server, tool_name, arguments)

    description = _one_line(tool.description or f"MCP tool {tool_name} on server {server}.")
    doc = f"{description}\n"
    if doc_args:
        doc += "\nArgs:\n" + "\n".join(doc_args) + "\n"
    doc += "\nReturns:\n    The MCP tool's result text.\n"
    wrapper.__name__ = full_name
    wrapper.__qualname__ = full_name
    wrapper.__doc__ = doc
    wrapper.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        params, return_annotation=str,
    )
    return wrapper


# Names of the agent's built-in tools ("cron_job" is built in only to
# cron sessions dispatched via run_agent("cron", ...), whose tools file
# supplies it).  A synthesized MCP tool name colliding with any of
# these (e.g. server "run" + tool "parallel" → "run_parallel") would
# make KISSAgent._add_functions() raise and abort the whole tool loop,
# so such names are pre-reserved and the MCP tool gets a numeric suffix
# instead.
_RESERVED_TOOL_NAMES = frozenset({
    "Bash", "Read", "Edit", "Write", "finish",
    "go_to_url", "click", "type_text", "press_key", "scroll",
    "screenshot", "get_page_content", "show_browser", "close_browser",
    "skill", "ask_user_question", "talk", "set_model",
    "run_parallel", "number_of_cores", "summary",
    "cron_job", "run_agent",
})


def make_mcp_tools(work_dir: str) -> list[Any]:
    """Build the agent tools for every configured, permitted MCP server.

    Connects to each server configured for *work_dir* (errors are
    logged and the server skipped so a broken server never breaks
    agent startup), lists its tools, filters them through the
    ``mcp_permissions`` wildcard rules, and wraps the survivors as
    kiss tool functions named ``<server>_<tool>``.

    Args:
        work_dir: The project directory whose MCP servers to expose.

    Returns:
        The (possibly empty) list of tool callables.
    """
    servers = load_mcp_servers(work_dir)
    if not servers:
        return []
    rules = load_mcp_permissions()
    manager = MCPManager.instance()
    tools: list[Any] = []
    taken_names: set[str] = set(_RESERVED_TOOL_NAMES)
    for name, config in servers.items():
        conn = manager.connect(config)
        if conn.session is None:
            logger.warning(
                "MCP server %s unavailable: %s", name, conn.error or "unknown error",
            )
            continue
        key = _connection_key(config)
        for tool in conn.tools:
            try:
                wrapper = make_mcp_tool_wrapper(
                    manager, name, tool, connection_key=key,
                )
            except Exception:
                # One malformed tool schema must not suppress every
                # other MCP tool for the run.
                logger.warning(
                    "could not wrap MCP tool %s on server %s; skipping",
                    getattr(tool, "name", "?"), name, exc_info=True,
                )
                continue
            # Sanitized <server>_<tool> names are not injective (e.g.
            # server 'a' tool 'b_c' vs server 'a_b' tool 'c'); a
            # duplicate registration would abort the agent's tool loop.
            base = wrapper.__name__
            full_name = base
            counter = 2
            while full_name in taken_names:
                full_name = f"{base}_{counter}"
                counter += 1
            taken_names.add(full_name)
            wrapper.__name__ = full_name
            wrapper.__qualname__ = full_name
            if rules and mcp_tool_permission(full_name, rules) == "deny":
                continue
            tools.append(wrapper)
    return tools
