# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""MCP (Model Context Protocol) server support for the ``sorcar`` CLI.

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
  the MCP SDK's OAuth 2.1 provider (dynamic client registration +
  PKCE).  Tokens are persisted per server under ``~/.kiss/mcp_auth/``
  by :class:`FileTokenStorage`; ``sorcar mcp auth <name>`` runs the
  interactive browser flow (see :mod:`kiss.ui.cli.mcp_cli`),
  while agent runs reuse the stored tokens and fail with a hint to
  run the auth command when interactive login would be required.

Connections are kept alive for the life of the process by a single
:class:`MCPManager` running an asyncio loop on a daemon thread; each
server's transport + session is opened and closed inside one long-lived
task (anyio cancel scopes must enter/exit in the same task).
"""

from __future__ import annotations

import asyncio
import atexit
import functools
import inspect
import json
import keyword
import logging
import threading
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.persistence import _default_kiss_dir
from kiss.agents.sorcar.skills import load_permission_rules, skill_permission

logger = logging.getLogger(__name__)

# Seconds to wait for a server to connect / a tool call to return.
CONNECT_TIMEOUT = 60.0
# After a connect() timeout, how long a straggler task stuck
# mid-handshake is given to unwind gracefully (via its ``stop`` event)
# before it is cancelled outright so the transport child is reaped.
_CONNECT_STRAGGLER_GRACE_S = 5.0
CALL_TIMEOUT = 300.0

# JSON-schema type → Python annotation used when synthesizing wrapper
# signatures (so kiss's schema builder round-trips the MCP inputSchema).
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
        # Claude Code's leniency: infer from the fields present.
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
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
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
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entries = raw.get("mcpServers") if isinstance(raw, dict) else None
        if isinstance(entries, dict) and name in entries:
            del entries[name]
            path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
            removed.append(path)
    return removed


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


class FileTokenStorage:
    """MCP SDK :class:`~mcp.client.auth.TokenStorage` backed by a JSON file.

    Tokens and the dynamically registered client information for one
    server are stored together at ``~/.kiss/mcp_auth/<server>.json``
    with mode ``0600``.
    """

    def __init__(self, server_name: str) -> None:
        self.path = mcp_auth_dir() / f"{_sanitize(server_name)}.json"

    def _read(self) -> dict[str, Any]:
        """Read the stored JSON payload (empty dict when absent/bad)."""
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, data: dict[str, Any]) -> None:
        """Write *data* to the token file with owner-only permissions."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        try:
            self.path.chmod(0o600)
        except OSError:  # pragma: no cover - permission error
            logger.debug("could not chmod token file", exc_info=True)

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

    async def set_tokens(self, tokens: Any) -> None:
        """Persist *tokens* (an :class:`~mcp.shared.auth.OAuthToken`)."""
        data = self._read()
        data["tokens"] = tokens.model_dump(mode="json", exclude_none=True)
        self._write(data)

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
        """Persist the dynamically registered OAuth client information."""
        data = self._read()
        data["client_info"] = client_info.model_dump(mode="json", exclude_none=True)
        self._write(data)

    def clear(self) -> bool:
        """Delete the token file (``sorcar mcp logout``).

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
        "MCP server requires interactive OAuth login; run "
        "`sorcar mcp auth <name>` first."
    )


async def _noninteractive_callback() -> tuple[str, str | None]:
    """Refuse to wait for an OAuth callback during an agent run."""
    raise RuntimeError(
        "MCP server requires interactive OAuth login; run "
        "`sorcar mcp auth <name>` first."
    )


def build_oauth_provider(
    cfg: MCPServerConfig,
    redirect_handler: Any = None,
    callback_handler: Any = None,
    redirect_port: int = 0,
) -> Any:
    """Build the OAuth provider used to authenticate to a remote server.

    The provider performs OAuth 2.1 with dynamic client registration
    and PKCE on demand (i.e. when the server responds 401) and reuses
    or refreshes tokens stored by :class:`FileTokenStorage`.

    Args:
        cfg: The remote server configuration.
        redirect_handler: Async callable opening the authorization URL
            in a browser; defaults to a non-interactive refusal so
            agent runs never block waiting for a human.
        callback_handler: Async callable returning ``(code, state)``
            from the OAuth redirect; defaults to the same refusal.
        redirect_port: Local callback port registered in the client
            metadata (only meaningful for the interactive flow).

    Returns:
        An ``httpx.Auth`` instance (``OAuthClientProvider``).
    """
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata

    metadata = OAuthClientMetadata.model_validate({
        "client_name": "KISS Sorcar",
        "redirect_uris": [f"http://localhost:{redirect_port}/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "client_secret_post",
    })
    return OAuthClientProvider(
        server_url=cfg.url,
        client_metadata=metadata,
        storage=FileTokenStorage(cfg.name),
        redirect_handler=redirect_handler or _noninteractive_redirect,
        callback_handler=callback_handler or _noninteractive_callback,
    )


@dataclass
class _Connection:
    """Live state of one server connection owned by the manager."""

    config: MCPServerConfig
    ready: threading.Event = field(default_factory=threading.Event)
    # Created eagerly (asyncio.Event is loop-agnostic until awaited) so
    # a stop request arriving before the connection task's first line
    # runs is never lost — it is only ever set on the manager loop via
    # call_soon_threadsafe, and the task sees it when it parks.
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    session: Any = None
    tools: list[Any] = field(default_factory=list)
    error: str = ""
    task: Any = None
    # Set as the LAST act of ``_maintain_connection`` — i.e. only after
    # the transport contexts have fully unwound (child process reaped).
    # ``task`` (a ``concurrent.futures.Future``) is marked done the
    # instant it is *cancelled*, while the wrapped asyncio task is
    # still unwinding on the loop, so waiting on ``task`` alone lets
    # ``shutdown`` stop the loop mid-unwind and leak the child.
    finished: threading.Event = field(default_factory=threading.Event)


async def _enter_transport(stack: Any, config: MCPServerConfig, auth: Any) -> tuple:
    """Open *config*'s transport on *stack* and return ``(read, write)``."""
    import os

    if config.transport == "stdio":
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=config.command,
            args=list(config.args),
            env={**os.environ, **dict(config.env)},
        )
        read, write = await stack.enter_async_context(stdio_client(params))
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


async def _maintain_connection(conn: _Connection, auth: Any) -> None:
    """Own one server connection for its whole lifetime in a single task.

    anyio cancel scopes must be entered and exited by the same task, so
    the transport and session contexts are opened here, the task then
    parks on the ``stop`` event, and the contexts unwind here too.
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
            await conn.stop.wait()
    except BaseException as exc:
        conn.error = f"{type(exc).__name__}: {exc}"
        logger.debug("MCP connection %s failed", conn.config.name, exc_info=True)
    finally:
        conn.session = None
        conn.ready.set()
        # Signal full unwind (transports closed, child reaped) LAST —
        # ``disconnect_all`` waits on this after a cancel so it never
        # stops the loop while this task is still tearing down.
        conn.finished.set()


class MCPManager:
    """Process-wide manager of live MCP server connections.

    Runs a private asyncio event loop on a daemon thread; every server
    gets one long-lived task that owns its transport + session.  The
    synchronous facade (:meth:`get_tools`, :meth:`call_tool`) is what
    agent tool wrappers and the CLI use.
    """

    _instance: MCPManager | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="mcp-manager", daemon=True,
        )
        self._thread.start()
        self._connections: dict[str, _Connection] = {}
        # Connections evicted from ``_connections`` by a connect()
        # timeout whose task has not finished yet.  ``disconnect_all``
        # tears these down too — without this list a task stuck
        # mid-handshake (which ``stop.set()`` cannot unwind) would keep
        # its transport child alive forever, invisible to shutdown.
        self._orphans: list[_Connection] = []
        self._lock = threading.Lock()
        self._shut_down = False
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
        with self._lock:
            if self._shut_down:
                # Fail fast: the manager loop is stopped, so a scheduled
                # coroutine would never run and ready.wait() would burn
                # the full CONNECT_TIMEOUT on a stale manager reference.
                conn = _Connection(config=config)
                conn.error = "manager shut down"
                conn.ready.set()
                return conn
            existing = self._connections.get(config.name)
            if (
                existing is not None
                and existing.config == config
                and existing.error == ""
            ):
                conn = existing
            else:
                if existing is not None:
                    # Stop the stale connection's task so its server
                    # subprocess/stream is closed, not leaked.
                    self._loop.call_soon_threadsafe(existing.stop.set)
                if config.transport in ("http", "sse") and auth is None:
                    auth = build_oauth_provider(config)
                conn = _Connection(config=config)
                self._connections[config.name] = conn
                conn.task = asyncio.run_coroutine_threadsafe(
                    _maintain_connection(conn, auth), self._loop,
                )
        if not conn.ready.wait(CONNECT_TIMEOUT):
            # Tear the straggler down instead of leaving a poisoned
            # record: if the server finished connecting just after the
            # deadline, the record would be live (session set) yet
            # marked failed — contradictory state that a later
            # connect() would needlessly tear down and that
            # format_mcp_listing would report inconsistently.
            #
            # Setting ``stop`` only unwinds a task that reached its
            # ``stop.wait()`` park; one still stuck mid-handshake (a
            # stdio child that never speaks MCP) ignores it and holds
            # the transport child alive forever.  So the straggler is
            # (a) kept visible to ``disconnect_all`` via ``_orphans``
            # and (b) cancelled after a grace period even without an
            # explicit shutdown.
            with self._lock:
                if self._connections.get(config.name) is conn:
                    del self._connections[config.name]
                # Stamp the error under the lock: the connection task may
                # be finishing concurrently (setting conn.error in its
                # except clause) and other threads read conn.error through
                # lock-guarded paths.
                conn.error = conn.error or "connection timed out"
                if conn.task is not None and not conn.task.done():
                    self._orphans.append(conn)
            self._loop.call_soon_threadsafe(conn.stop.set)
            self._reap_straggler(conn)
        return conn

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
            # Loop already closed (shutdown raced us) — disconnect_all
            # has taken (or will take) care of the orphan list.
            pass

    def _forget_orphan(self, conn: _Connection, _future: Any) -> None:
        """Drop a finished straggler from ``_orphans`` (done callback)."""
        with self._lock:
            try:
                self._orphans.remove(conn)
            except ValueError:
                pass

    def call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> str:
        """Call *tool* on *server* and return the textual result.

        Args:
            server: The configured server name (must be connected).
            tool: The MCP tool name on that server.
            arguments: The tool arguments.

        Returns:
            The flattened result text (``Error: ...`` on tool errors).
        """
        with self._lock:
            if self._shut_down:
                # The manager loop is stopped: no connection can be live
                # and a scheduled coroutine would never run.
                return (
                    f"Error: MCP server {server!r} is not connected "
                    f"(manager shut down)"
                )
            conn = self._connections.get(server)
        # Snapshot the session exactly once: the manager-loop thread
        # nulls conn.session whenever the connection dies or is stopped
        # (_maintain_connection's finally), so re-reading the attribute
        # after the check would race an AttributeError out of the tool
        # wrapper instead of returning the friendly error string.
        session = conn.session if conn is not None else None
        if conn is None or session is None:
            why = conn.error if conn else "never connected"
            return f"Error: MCP server {server!r} is not connected ({why})"
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
        return _result_text(result)

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
            # Include stragglers evicted by connect() timeouts: their
            # tasks may still be stuck mid-handshake holding a live
            # transport child that only a cancel can reap.
            conns.extend(self._orphans)
            self._orphans.clear()
        for conn in conns:
            self._loop.call_soon_threadsafe(conn.stop.set)
        for conn in conns:
            if conn.task is not None:
                try:
                    conn.task.result(timeout=10)
                except BaseException:  # noqa: BLE001 — CancelledError is BaseException
                    # ``CancelledError`` (raised when the straggler
                    # grace timer cancelled the future) does NOT
                    # inherit ``Exception`` — a plain ``except
                    # Exception`` would let it blow up the whole
                    # teardown loop.
                    conn.task.cancel()
                    logger.debug("MCP disconnect error", exc_info=True)
                    # The future is marked done the moment it is
                    # cancelled, but the wrapped asyncio task is still
                    # unwinding on the loop.  Wait for the unwind to
                    # actually finish (transport closed, child reaped)
                    # so ``shutdown`` cannot stop the loop mid-teardown
                    # and orphan the server subprocess.
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
    """Restrict *name* to characters safe in tool names and file names."""
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)


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
    name = "".join(c if c.isalnum() or c == "_" else "_" for c in prop_name)
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
    manager: MCPManager, server: str, tool: Any,
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

    Returns:
        The wrapper callable, named ``<server>_<tool>``.
    """
    tool_name = str(tool.name)
    full_name = f"{_sanitize(server)}_{_sanitize(tool_name)}"
    schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
    # Nothing in MCP constrains the schema's inner values; be lenient
    # with malformed shapes (e.g. a list ``properties`` or a string
    # ``required``) so one bad tool never breaks agent startup.
    props = schema.get("properties")
    if not isinstance(props, dict):
        props = {}
    required_raw = schema.get("required")
    required = set(required_raw) if isinstance(required_raw, list) else set()

    # (is_required, Parameter, doc line) per property; the Python
    # parameter name may differ from the JSON property name (which can
    # be hyphenated or a keyword), so param_map maps it back.
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
    # Required parameters must precede optional ones in a Python
    # signature; JSON object properties carry no such ordering.
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
        return manager.call_tool(server, tool_name, arguments)

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
    for name, config in servers.items():
        conn = manager.connect(config)
        if conn.session is None:
            logger.warning(
                "MCP server %s unavailable: %s", name, conn.error or "unknown error",
            )
            continue
        for tool in conn.tools:
            wrapper = make_mcp_tool_wrapper(manager, name, tool)
            if rules and mcp_tool_permission(wrapper.__name__, rules) == "deny":
                continue
            tools.append(wrapper)
    return tools


def format_mcp_listing(work_dir: str, connect: bool = False) -> str:
    """Format the configured servers as the listing printed by ``/mcp``.

    Args:
        work_dir: The project directory whose servers to list.
        connect: When ``True``, connect to each server and append its
            live status (✓ + tool count, or ✗ + error).

    Returns:
        A printable multi-line listing, or a configuration hint when
        no servers are configured.
    """
    servers = load_mcp_servers(work_dir)
    if not servers:
        return (
            "No MCP servers configured.\n"
            "Add one with: sorcar mcp add <name> <command> [args...]\n"
            f"Config files: {user_mcp_config_path()} (user), "
            f"{project_mcp_config_path(work_dir)} (project), "
            f"{claude_project_mcp_config_path(work_dir)} (Claude-compatible)."
        )
    rules = load_mcp_permissions()
    width = max(len(n) for n in servers)
    lines = []
    for name in sorted(servers):
        cfg = servers[name]
        target = cfg.command + (" " + " ".join(cfg.args) if cfg.args else "") \
            if cfg.transport == "stdio" else cfg.url
        line = f"  {name:<{width}}  ({cfg.source}, {cfg.transport}) {target}"
        if connect:
            conn = MCPManager.instance().connect(cfg)
            if conn.session is not None:
                allowed = [
                    t for t in conn.tools
                    if not rules or mcp_tool_permission(
                        f"{_sanitize(name)}_{_sanitize(str(t.name))}", rules,
                    ) == "allow"
                ]
                line += f" — ✓ connected, {len(allowed)}/{len(conn.tools)} tools allowed"
            else:
                line += f" — ✗ {conn.error or 'connection failed'}"
        lines.append(line)
    return "\n".join(lines)
