# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""MCP (Model Context Protocol) server support for Sorcar.

Implements MCP server management with Claude Code compatibility:

* **Configuration** — servers are stored as ``{"mcpServers": {...}}``
  JSON, the same shape Claude Code uses, and are discovered from
  (low → high precedence; later wins on a name clash):

  - ``$KISS_HOME/mcp.json`` — user servers.
  - ``<work_dir>/.mcp.json`` — Claude Code's project file, so MCP
    servers checked into a repo for Claude Code work unchanged.
  - ``<work_dir>/.kiss/mcp.json`` — native project servers.

  Each entry is ``{"type": "stdio"|"http"|"sse", "command": ...,
  "args": [...], "env": {...}}`` for local servers or ``{"type":
  "http"|"sse", "url": ..., "headers": {...}}`` for remote ones
  (``type`` defaults to ``stdio`` when ``command`` is present, else
  ``http`` — matching Claude Code's leniency).  A local server of a
  project file runs in ``<work_dir>``, so its entry may name the
  project's files by relative path; each project gets its own.

* **Tools** — every tool of every configured server is exposed to the
  agent as a function named ``<server>_<tool>`` whose signature and
  docstring are synthesized from the MCP ``inputSchema``, so the
  standard kiss schema builder produces a faithful OpenAI tool schema.

* **Permission wildcards** — the ``mcp_permissions`` key in
  ``$KISS_HOME/config.json`` maps wildcard patterns to ``"allow"`` or
  ``"deny"`` (e.g. ``{"*": "allow", "mymcp_*": "deny"}``).  Patterns
  are matched against the full ``<server>_<tool>`` name with the
  *last* matching rule winning (OpenCode semantics); denied tools are
  never registered.

* **OAuth** — remote (``http``/``sse``) servers authenticate through
  the MCP SDK's OAuth 2.1 provider using tokens persisted per server
  under ``$KISS_HOME/mcp_auth/`` by :class:`FileTokenStorage`.  The user
  signs in once through :mod:`kiss.agents.sorcar.mcp_oauth` (the
  ``connect_mcp_server`` tool); agent runs then reuse and refresh those
  tokens without ever blocking on a browser.

Connections are kept alive for the life of the process by a single
:class:`MCPManager` running an asyncio loop on a daemon thread; each
server's transport + session is opened and closed inside one long-lived
task (anyio cancel scopes must enter/exit in the same task).
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextlib
import functools
import hashlib
import inspect
import itertools
import json
import keyword
import logging
import os
import sys
import threading
import time
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.mcp_tool_search import (
    ServerTools,
    make_deferred_tools,
    split_by_budget,
    upfront_tool_chars,
)
from kiss.agents.sorcar.skills import load_permission_rules, skill_permission
from kiss.core import tool_interrupt
from kiss.core.config import kiss_home
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.locked import locked, locked_call
from kiss.core.tool_schema import tool_schema
from kiss.core.utils import atomic_write_text

logger = logging.getLogger(__name__)

_LOCK_TIMEOUT_S = 30.0
"""Longest wait for another process's lock on an MCP config or token file.

A legitimate holder rewrites one small JSON file and is done in
milliseconds, so a wait past this bound is a wedged holder (a process
stopped in a debugger, a stuck filesystem): the ``TimeoutError`` is the
caller's error instead of a tool thread parked for good.
"""

CONNECT_TIMEOUT = 60.0
_CONNECT_STRAGGLER_GRACE_S = 5.0
CALL_TIMEOUT = 300.0

#: Seconds :meth:`MCPManager.disconnect_all` waits — once for every
#: connection together — for the tasks to unwind, and once more for the
#: cancelled stragglers to finish.
_DISCONNECT_WAIT_S = 10.0

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
        cwd: Working directory of a stdio server: the project whose
            config file names it, so a relative path in its entry means
            that project's file.  Empty for a user-level server, which
            every project shares and which runs in the daemon's
            directory.  It follows from where the entry was read, so
            it is not part of the entry (:meth:`to_json`).
        always_load: The entry's ``alwaysLoad`` key: ``True`` puts the
            server's tool definitions in every model request, ``False``
            defers them behind the tool search, ``None`` (no key) does
            as ``True`` unless the user set a budget for the upfront
            definitions, which then decides by size
            (:func:`~kiss.agents.sorcar.mcp_tool_search.split_by_budget`).
    """

    name: str
    transport: str = "stdio"
    command: str = ""
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    url: str = ""
    headers: tuple[tuple[str, str], ...] = ()
    cwd: str = ""
    always_load: bool | None = None

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
        if self.always_load is not None:
            out["alwaysLoad"] = self.always_load
        return out


def user_mcp_config_path() -> Path:
    """Return the user-level MCP config file (``$KISS_HOME/mcp.json``)."""
    return kiss_home() / "mcp.json"


def project_mcp_config_path(work_dir: str) -> Path:
    """Return the project-level MCP config file (``<work_dir>/.kiss/mcp.json``).

    The user's home directory is no project: ``~/.kiss`` is the stock
    brand's global home (a leftover under any other brand), and the
    file of a task working there is the live user-level one, as
    :func:`~kiss.agents.sorcar.skills.discover_skills` has it.
    """
    project = Path(work_dir)
    if project.resolve() == Path.home().resolve():
        return user_mcp_config_path()
    return project / ".kiss" / "mcp.json"


def claude_project_mcp_config_path(work_dir: str) -> Path:
    """Return Claude Code's project MCP file (``<work_dir>/.mcp.json``)."""
    return Path(work_dir) / ".mcp.json"


def _config_lock_path() -> Path:
    """Return the lock file every MCP config edit holds (``$KISS_HOME/mcp.lock``).

    One lock in ``$KISS_HOME`` for every config file: a lock next to
    each file left an untracked ``.mcp.lock`` beside Claude Code's
    checked-in ``<project>/.mcp.json`` in the user's project.
    """
    return kiss_home() / "mcp.lock"


def mcp_auth_dir() -> Path:
    """Return the directory holding per-server OAuth token files."""
    return kiss_home() / "mcp_auth"


def _parse_server_entry(name: str, raw: Any) -> MCPServerConfig | None:
    """Parse one ``mcpServers`` JSON entry leniently.

    Args:
        name: The server name (the JSON key).
        raw: The JSON value (must be a dict to be usable).

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
    always_load = raw.get("alwaysLoad")
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
        always_load=always_load if isinstance(always_load, bool) else None,
    )


def _load_config_file(path: Path) -> dict[str, MCPServerConfig]:
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
        cfg = _parse_server_entry(str(name), entry)
        if cfg is not None:
            servers[cfg.name] = cfg
    return servers


def load_mcp_servers(work_dir: str) -> dict[str, MCPServerConfig]:
    """Load all configured MCP servers visible from *work_dir*.

    Load order (low → high precedence; later wins on a name clash):
    user (``$KISS_HOME/mcp.json``), Claude Code project (``.mcp.json``),
    native project (:func:`project_mcp_config_path`).  A stdio server
    of a project file runs in *work_dir* (:attr:`MCPServerConfig.cwd`).

    Args:
        work_dir: The project directory whose servers to include.

    Returns:
        Mapping of server name → :class:`MCPServerConfig`.
    """
    user = user_mcp_config_path()
    servers = _load_config_file(user)
    cwd = os.path.abspath(work_dir)
    for path in (claude_project_mcp_config_path(work_dir), project_mcp_config_path(work_dir)):
        if path == user:
            continue  # the home directory's: read above, as the user's
        for name, cfg in _load_config_file(path).items():
            servers[name] = replace(cfg, cwd=cwd) if cfg.transport == "stdio" else cfg
    return servers


def save_mcp_server(cfg: MCPServerConfig, scope: str, work_dir: str) -> Path:
    """Persist *cfg* in the user- or project-scope MCP config file.

    Args:
        cfg: The server configuration to save.
        scope: ``"user"`` (``$KISS_HOME/mcp.json``) or ``"project"``
            (:func:`project_mcp_config_path`).
        work_dir: The project directory (used for project scope).

    Returns:
        The path of the config file written.

    Raises:
        ValueError: When the file exists but is not JSON whose top
            level and ``mcpServers`` are objects; it is left untouched
            (a hand-edited file with a syntax error must not lose its
            other servers).
        OSError: When the file cannot be read or written, or another
            process held the config lock for :data:`_LOCK_TIMEOUT_S`
            (``TimeoutError``).
    """
    path = (
        user_mcp_config_path() if scope == "user"
        else project_mcp_config_path(work_dir)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(_config_lock_path(), timeout=_LOCK_TIMEOUT_S):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raw = {}
        except ValueError as exc:
            raise ValueError(f"{path} is not valid JSON ({exc}); fix or remove it") from exc
        entries = raw.setdefault("mcpServers", {}) if isinstance(raw, dict) else None
        if not isinstance(entries, dict):
            raise ValueError(
                f'{path} is not an MCP config ({{"mcpServers": {{...}}}}); fix or remove it'
            )
        entries[cfg.name] = cfg.to_json()
        _atomic_write_config(path, raw)
    return path


def remove_mcp_server(name: str, work_dir: str) -> list[Path]:
    """Remove server *name* from every writable MCP config file.

    The write path's counterpart of :func:`save_mcp_server`; its caller
    is ``connectors/enable.py disable``, outside ``src/``.

    Args:
        name: The server name to remove.
        work_dir: The project directory (for the project-scope files).

    Returns:
        The config files the server was actually removed from.
    """
    removed: list[Path] = []
    with exclusive_file_lock(_config_lock_path(), timeout=_LOCK_TIMEOUT_S):
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
                _atomic_write_config(path, raw)
                removed.append(path)
    return removed


def _atomic_write_config(path: Path, raw: dict[str, Any]) -> None:
    """Atomically replace the MCP config *path* with *raw* as JSON.

    A new file is private (``0o600``: a server's ``env`` and ``headers``
    carry API keys and bearer tokens); an existing file keeps its bits.
    """
    atomic_write_text(path, json.dumps(raw, indent=2) + "\n")


_RESERVED_BASENAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def _token_file_name(server_name: str, server_url: str = "") -> str:
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

    With a *server_url* the digest is always there and covers name
    and URL together: two configurations of one name at different
    servers (the same name in two projects, a project override of a
    user configuration) keep separate credential files, so neither is
    ever handed the other's bearer token.  Only the name-only file is
    historical: it was written before credentials were kept per URL,
    and no server reads it now (its user signs in once more).
    """
    safe = _sanitize(server_name)
    reserved = safe.lower() in _RESERVED_BASENAMES
    plain = safe == server_name and server_name == server_name.lower() and not reserved
    if plain and not server_url:
        return f"{safe}.json"
    identity = f"{server_name}\n{server_url}" if server_url else server_name
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    prefix = "s_" if reserved else ""
    return f"{prefix}{safe}.{digest}.json"


class FileTokenStorage:
    """MCP SDK :class:`~mcp.client.auth.TokenStorage` backed by a JSON file.

    Tokens and the dynamically registered client information for one
    server are stored together under ``$KISS_HOME/mcp_auth/`` with mode
    ``0600``, in a file of the server's name and URL
    (:func:`_token_file_name`; the URL is recorded in the file as
    ``server_url`` too, the file name's digest being opaque).  The
    runtime provider and the sign-in both give the URL; a storage
    built from the name alone addresses the historical name-only file.
    """

    def __init__(self, server_name: str, server_url: str = "") -> None:
        self.path = mcp_auth_dir() / _token_file_name(server_name, server_url)
        self._lock_path = self.path.with_suffix(".lock")
        self.server_url = server_url
        # What a refresh by this storage's provider replaces (see put):
        # the stored tokens as it last read or wrote them, while it
        # holds the refresh lock.
        self._seen_tokens: Any = None
        self._refreshing = False

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
        if self.server_url:
            data["server_url"] = self.server_url
        atomic_write_text(self.path, json.dumps(data, indent=2) + "\n", mode=0o600)

    def _load(self, data: dict[str, Any], key: str, model: Any) -> Any:
        """Return ``data[key]`` validated as the pydantic *model*, else ``None``."""
        raw = data.get(key)
        if not raw:
            return None
        try:
            return model.model_validate(raw)
        except Exception:
            logger.debug("invalid %s in %s", key, self.path, exc_info=True)
            return None

    async def get_tokens(self) -> Any:
        """Return the stored :class:`~mcp.shared.auth.OAuthToken`, if any.

        The SDK treats a loaded token as valid until a request fails, so
        an access token past its stored ``expires_at`` is handed back
        with an empty ``access_token``: the provider then refreshes it
        (when a refresh token exists) instead of sending a dead token
        and falling into a full browser sign-in.
        """
        from mcp.shared.auth import OAuthToken

        data = self._read()
        self._seen_tokens = data.get("tokens")
        token = self._load(data, "tokens", OAuthToken)
        if token is None:
            return None
        deadline = self.refresh_deadline()
        if deadline is not None and time.time() >= deadline:
            token.access_token = ""
        return token

    def refresh_deadline(self) -> float | None:
        """When the stored access token is due for a refresh, if its expiry was recorded.

        A minute before the stored ``expires_at``, so a token is never
        sent in its last seconds.
        """
        expires_at = self._read().get("expires_at")
        return expires_at - 60 if isinstance(expires_at, (int, float)) else None

    @contextlib.asynccontextmanager
    async def refreshing(self, timeout: float = _LOCK_TIMEOUT_S) -> AsyncIterator[None]:
        """Hold the stored grant's refresh lock: one provider of it refreshes at a time.

        Every connection to the server, in this and in any other kiss
        process, and the sign-in keep a copy of the grant, and a rotating
        refresh token is good for one refresh: the holder re-reads the
        store and refreshes only what is still stale there.  A lock file
        of its own (:meth:`put` takes the write lock itself), probed
        rather than waited for: it is held across the refresh request,
        by a provider on the caller's own event loop too, and a cancel
        between two probes holds nothing.  Nothing is locked (and no
        lock file made) while no grant is stored.

        Args:
            timeout: The most seconds to wait for another holder.

        Raises:
            TimeoutError: When another holder kept the lock for *timeout*.
        """
        lock_path = self.path.with_suffix(".refresh.lock")
        deadline = time.monotonic() + timeout
        while self.path.exists():
            with exclusive_file_lock(lock_path, blocking=False) as held:
                if held:
                    self._refreshing = True
                    try:
                        yield
                    finally:
                        self._refreshing = False
                    return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{lock_path} still locked after {timeout:g}s")
            await asyncio.sleep(0.05)
        yield

    def get_oauth_metadata(self) -> Any:
        """Return the authorization server metadata saved at sign-in, if any."""
        from mcp.shared.auth import OAuthMetadata

        return self._load(self._read(), "oauth_metadata", OAuthMetadata)

    def set_oauth_metadata(self, metadata: Any) -> None:
        """Persist the authorization server metadata discovered at sign-in.

        The runtime provider seeds itself with it so a refresh goes to
        the real token endpoint even when the authorization server is
        not the MCP server itself.

        Args:
            metadata: An :class:`~mcp.shared.auth.OAuthMetadata`.
        """
        self.put("oauth_metadata", metadata.model_dump(mode="json", exclude_none=True))

    def put(self, key: str, value: Any, refreshed: bool = False) -> None:
        """Replace *key* in the stored JSON under the inter-process lock.

        Args:
            key: Top-level key to set (``"tokens"``, ``"client_info"`` or
                ``"oauth_metadata"``).
            value: Its already-serialized JSON value, or ``None`` to forget it.
            refreshed: Whether *value* is the tokens a refresh was
                answered with.  They replace only the tokens this
                storage last read or wrote: a grant removed
                (:meth:`clear`) or replaced (a new sign-in) while the
                refresh was at the server is not the one refreshed, and
                stays as it is.
        """
        with exclusive_file_lock(self._lock_path, timeout=_LOCK_TIMEOUT_S):
            data = self._read()
            if refreshed and data.get("tokens") != self._seen_tokens:
                logger.info("%s changed during a refresh: its answer is dropped", self.path)
                return
            data[key] = value
            if key == "tokens":
                self._seen_tokens = value
                expires_in = (value or {}).get("expires_in")
                data["expires_at"] = (
                    time.time() + expires_in if isinstance(expires_in, (int, float)) else None
                )
            self._write(data)

    async def set_tokens(self, tokens: Any) -> None:
        """Persist *tokens* (an :class:`~mcp.shared.auth.OAuthToken`).

        ``expires_at`` (absolute, seconds) is recorded next to them so
        :meth:`get_tokens` can tell an expired access token apart.

        The read-modify-write runs on a worker thread: it waits for an
        inter-process lock and writes a file, and this coroutine runs on
        the manager's single shared event loop, which also carries every
        other server's transport and every in-flight tool call.  A
        cancellation is delivered once the thread has ended, not before:
        the thread cannot be stopped, and a provider cancelled in its
        refresh would let go of the refresh lock (:meth:`refreshing`)
        with the store still holding the grant it has just used up.
        """
        import anyio

        write = asyncio.ensure_future(asyncio.to_thread(
            self.put, "tokens", tokens.model_dump(mode="json", exclude_none=True),
            self._refreshing,
        ))
        cancelled: BaseException | None = None
        # Shielded from the scope of the transport's tasks, which cancels
        # them again on every pass of the loop; asyncio cancels once.
        with anyio.CancelScope(shield=True):
            while not write.done():
                try:
                    await asyncio.wait([write])
                except asyncio.CancelledError as e:
                    cancelled = e
        error = write.exception()
        failure = cancelled or error
        if failure is not None:
            raise failure

    async def get_client_info(self) -> Any:
        """Return the stored OAuth client registration, if any (the SDK's coroutine form)."""
        return self.client_info()

    def client_info(self) -> Any:
        """Return the stored OAuth client registration, if any.

        A plain file read; callers outside the SDK (the sign-in flow)
        need no event loop for it.
        """
        from mcp.shared.auth import OAuthClientInformationFull

        return self._load(self._read(), "client_info", OAuthClientInformationFull)

    async def set_client_info(self, client_info: Any) -> None:
        """Persist the dynamically registered OAuth client information.

        Runs on a worker thread for the same reason as
        :meth:`set_tokens`: the lock is inter-process and the caller's
        event loop is shared by every MCP connection.
        """
        await asyncio.to_thread(
            self.put, "client_info", client_info.model_dump(mode="json", exclude_none=True),
        )

    def clear(self) -> bool:
        """Delete the stored token file for this server.

        Taken under the same inter-process lock as :meth:`put`, so a
        concurrent read-modify-write cannot land its stale copy on top
        of the deletion; a refresh answered after it stores nothing
        (see :meth:`put`).

        Returns:
            ``True`` when a file was deleted, ``False`` when absent.
        """
        with exclusive_file_lock(self._lock_path, timeout=_LOCK_TIMEOUT_S):
            try:
                self.path.unlink()
                return True
            except FileNotFoundError:
                return False


_LOGIN_HINT = (
    "MCP server requires an OAuth sign-in; ask the agent to call "
    "connect_mcp_server(<name>) (or run `python -m "
    "kiss.agents.sorcar.mcp_oauth <name>`) first."
)


async def _refuse_login(conn: _Connection, *_: Any) -> Any:
    """Refuse the browser OAuth flow *conn* needs during an agent run, and end *conn*.

    Without the sign-in the connection can send nothing more, so it is
    stopped here, the hint its error, whatever its transport makes of
    the exception: a request's POST ends the transport with it, but
    the SDK's streamable HTTP client logs and drops the one a
    notification's POST or its GET stream meets.
    """
    refusal = RuntimeError(_LOGIN_HINT)
    conn.error = describe_exception(refusal)
    conn.stop.set()
    raise refusal


def build_oauth_provider(conn: _Connection) -> Any:
    """Build the OAuth provider *conn* authenticates to its remote server with.

    The provider reuses the tokens stored by :class:`FileTokenStorage`
    and refreshes them at the token endpoint recorded at sign-in.  It
    never starts an interactive login: an agent run must not block
    waiting for a human at a browser, so both handlers refuse, end the
    connection and point at the sign-in tool
    (:mod:`kiss.agents.sorcar.mcp_oauth`).

    Args:
        conn: The connection to a remote server.

    Returns:
        An ``httpx.Auth`` instance (``OAuthClientProvider``).
    """
    from kiss.agents.sorcar.mcp_oauth import build_provider

    refuse = functools.partial(_refuse_login, conn)
    return build_provider(conn.config, refuse, refuse)


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
    #: The maintainer's asyncio task, and whether it has begun tearing
    #: its transport down.  Both are touched on the manager loop only
    #: (:func:`_cancel_unless_unwinding`, the straggler reaper's cancel).
    loop_task: asyncio.Task[Any] | None = None
    unwinding: bool = False
    #: The tool calls running on the session, each call's task with its
    #: caller's future: failed when the maintainer ends
    #: (:func:`_fail_calls`).  Touched on the manager loop only.
    calls: dict[asyncio.Task[Any], concurrent.futures.Future[Any]] = field(default_factory=dict)


def _failed_connection(config: MCPServerConfig, error: str) -> _Connection:
    """Return the record of a connection to *config* that was never started: *error* says why."""
    conn = _Connection(config=config, error=error)
    conn.ready.set()
    return conn


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
    return _fallback_errlog()


@functools.cache
def _fallback_errlog() -> Any:
    """The stream MCP children inherit as stderr while ``sys.stderr`` is unusable.

    Opened once per process: the SDK only hands the stream to the child
    and never closes it, so a fresh ``dup`` per spawn would leak one
    descriptor per reconnect of the long-lived daemon.
    """
    try:
        return open(os.dup(2), "w", closefd=True)  # noqa: SIM115
    except OSError:
        return open(os.devnull, "w")  # noqa: SIM115


class _PostFailed(BaseExceptionGroup[Exception]):
    """The failure of an SSE transport's POST, raised so that its writer cannot swallow it.

    The SDK's SSE writer catches ``Exception``; this group is not one,
    so it ends the writer's task and, through the transport's task
    group, the task that owns the transport.
    """


@functools.cache
def _sse_http_client_factory() -> Any:
    """Return the factory of an SSE transport's HTTP client: a POST that fails ends the transport.

    The client's messages go out as POSTs, sent in turn by one writer
    task.  The SDK's writer logs the first that fails (a network error,
    a refused sign-in, an error status) and stops, its reader left
    running: the connection looks alive and can send nothing more, and
    the request that POST carried waits for an answer nobody was asked
    for (a handshake until its deadline, a tool call for
    :data:`CALL_TIMEOUT`).  Raised as :class:`_PostFailed`, the failure
    ends the transport instead, as a failed request ends a streamable
    HTTP one: the owner is cancelled where it waits and gets the POST's
    own error (:func:`describe_exception`).

    Defined on first use: httpx is imported only once a remote server is used.
    """
    import httpx

    class SSEHttpClient(httpx.AsyncClient):
        """An ``httpx.AsyncClient`` whose failed POST raises :class:`_PostFailed`."""

        async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
            """Send *request* as ``httpx.AsyncClient.send`` does.

            Raises:
                _PostFailed: When *request* is a POST that failed: the
                    client raised, or the response it ended with (auth
                    and redirects done) is not a success.
            """
            if request.method != "POST":
                return await super().send(request, **kwargs)
            try:
                return (await super().send(request, **kwargs)).raise_for_status()
            except Exception as exc:
                raise _PostFailed("the SSE transport's POST failed", [exc]) from None

    # ``follow_redirects`` as in the SDK's own factory, ``create_mcp_http_client``.
    return functools.partial(SSEHttpClient, follow_redirects=True)


async def _enter_transport(stack: Any, config: MCPServerConfig, auth: Any) -> tuple:
    """Open *config*'s transport on *stack* and return ``(read, write)``."""
    if config.transport == "stdio":
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=config.command,
            args=list(config.args),
            env={**os.environ, **dict(config.env)},
            cwd=config.cwd or None,
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
            sse_client(
                config.url, headers=dict(config.headers) or None, auth=auth,
                httpx_client_factory=_sse_http_client_factory(),
            )
        )
        return read, write
    from mcp.client.streamable_http import streamablehttp_client

    read, write, _ = await stack.enter_async_context(
        streamablehttp_client(
            config.url, headers=dict(config.headers) or None, auth=auth,
        )
    )
    return read, write


async def _park_until_stopped(
    conn: _Connection, session: Any, health_interval: float,
) -> None:
    """Park until *conn* is stopped, failing fast if the server dies.

    A plain ``await conn.stop.wait()`` never notices a server that was
    killed, crashed, or restarted: the task stays parked, ``session``
    stays set, and every later tool call blocks on the dead transport
    until ``CALL_TIMEOUT``.  Pinging while idle turns that into a prompt
    error, which the manager turns into a reconnect.  No ping is sent
    while a call is in flight, and a ping that times out while a call
    is in flight (one that started after the ping went out) is
    inconclusive and ignored: a server whose loop is busy in a long
    sync tool handler cannot answer the ping, and failing it would tear
    the session down under a valid call, failing that call with
    "Connection closed" (a dead server is caught by the call's own
    timeout).  A server that answers the ping "Method not found" (one
    written outside the SDKs, without ``ping``) is alive: failing it
    would kill and respawn it every interval, losing its state.

    Args:
        conn: The connection being maintained.
        session: Its live MCP client session.
        health_interval: Seconds between pings (also the ping timeout).

    Raises:
        Exception: Whatever the failed ping raises, so the caller can
            record the error and unwind the transport.
    """
    from mcp import McpError
    from mcp.types import METHOD_NOT_FOUND

    while True:
        try:
            await asyncio.wait_for(conn.stop.wait(), timeout=health_interval)
            return
        except TimeoutError:
            if conn.in_flight:
                continue
            try:
                await asyncio.wait_for(session.send_ping(), timeout=health_interval)
            except TimeoutError:
                if not conn.in_flight:
                    raise
            except McpError as exc:
                if exc.error.code != METHOD_NOT_FOUND:
                    raise


def describe_exception(exc: BaseException) -> str:
    """Flatten an exception (or nested exception group) into one line.

    anyio task groups wrap transport failures in ``ExceptionGroup``s, whose
    own message ("unhandled errors in a TaskGroup") hides the real cause,
    such as the OAuth sign-in hint.

    Args:
        exc: The exception to describe.

    Returns:
        ``"Type: message"`` for each leaf exception, joined by ``"; "``.
    """
    inner = getattr(exc, "exceptions", None)
    if inner:
        return "; ".join(describe_exception(e) for e in inner)
    return f"{type(exc).__name__}: {exc}"


async def _list_all_tools(session: Any) -> list[Any]:
    """Return the tools of every page of *session*'s ``tools/list``.

    The SDK's ``list_tools`` returns one page and its ``nextCursor``;
    the pages are requested until a page has no cursor, or hands out
    one already followed (a server repeating a cursor would otherwise
    be paged for ever).
    """
    from mcp.types import PaginatedRequestParams

    tools: list[Any] = []
    followed: set[str] = set()
    params = None
    while True:
        listed = await session.list_tools(params=params)
        tools.extend(listed.tools)
        cursor = listed.nextCursor
        if not cursor or cursor in followed:
            return tools
        followed.add(cursor)
        params = PaginatedRequestParams(cursor=cursor)


async def _maintain_connection(
    conn: _Connection, auth: Any, health_interval: float = HEALTH_INTERVAL,
) -> None:
    """Own one server connection for its whole lifetime in a single task.

    anyio cancel scopes must be entered and exited by the same task, so
    the transport and session contexts are opened here, the task then
    parks on the ``stop`` event, and the contexts unwind here too.

    The handshake (transport, ``initialize``, ``tools/list``) runs under
    a deadline of this task's own: :data:`CONNECT_TIMEOUT` is otherwise
    enforced only by the thread waiting in :meth:`MCPManager.connect`,
    and a Stop injected into that thread takes the deadline with it,
    leaving a silent server's child in the pool for the life of the
    process — unready, so never evicted, and never cancelled.  The
    deadline is the moment the waiter's straggler reaper would have
    cancelled the task (:meth:`MCPManager._reap_straggler`), so a
    waiter that is not stopped sees no change.

    Args:
        conn: The connection record to fill in and own.
        auth: Optional ``httpx.Auth`` for remote transports.
        health_interval: Seconds between idle health pings.
    """
    from contextlib import AsyncExitStack

    from mcp import ClientSession

    handshake_timeout = CONNECT_TIMEOUT + _CONNECT_STRAGGLER_GRACE_S
    conn.loop_task = asyncio.current_task()
    try:
        async with AsyncExitStack() as stack:
            try:
                try:
                    async with asyncio.timeout(handshake_timeout):
                        read, write = await _enter_transport(stack, conn.config, auth)
                        session = await stack.enter_async_context(ClientSession(read, write))
                        await session.initialize()
                        conn.tools = await _list_all_tools(session)
                except TimeoutError:
                    raise TimeoutError(f"no MCP handshake within {handshake_timeout:g}s") from None
                conn.session = session
                conn.ready.set()
                await _park_until_stopped(conn, session, health_interval)
            finally:
                # The stack's exit now runs the transport's own shutdown
                # (stdio: close stdin, wait, then SIGTERM/SIGKILL).  A
                # second cancellation landing inside it skips the
                # signals and leaks the child; the straggler reaper,
                # due at the same instant as this task's handshake
                # deadline, leaves an unwinding task alone.
                # ``disconnect_all`` still cancels one at its deadline:
                # an HTTP transport's exit can block on the server.
                conn.unwinding = True
    except BaseException as exc:
        conn.error = describe_exception(exc)
        logger.debug("MCP connection %s failed", conn.config.name, exc_info=True)
    finally:
        conn.session = None
        # Before ``finished``: a shutdown waiting on it stops the loop.
        _fail_calls(conn)
        conn.ready.set()
        conn.finished.set()


def _fail_calls(conn: _Connection) -> None:
    """Fail the tool calls still running on *conn*, whose maintainer is ending (on the loop).

    No answer can reach them any more.  The SDK answers its pending
    requests when the server closes the transport, but not when the
    session is unwound under them (an HTTP transport ended by a refused
    OAuth sign-in, a disconnect), and a call is a task of its own,
    outside the transport's cancel scope: it waited out its read
    timeout (:data:`CALL_TIMEOUT`).  Each caller gets the connection's
    error at once.  No call is sent again: its tool may have run.
    """
    why = conn.error or "connection closed"
    for task, future in conn.calls.items():
        task.cancel()
        if future.set_running_or_notify_cancel():
            future.set_exception(ConnectionError(why))
    conn.calls.clear()


def _transport_closed(exc: Exception) -> bool:
    """Whether the tool-call failure *exc* says the session's transport has closed.

    A call sent on a closed transport fails with one of anyio's stream
    errors; one in flight as it closes is answered by the SDK itself,
    with its ``CONNECTION_CLOSED`` error.  The message is compared too:
    a server may use that code (JSON-RPC's first free one) for an error
    of its own, which must not cost it a restart.
    """
    import anyio
    from mcp import McpError
    from mcp.types import CONNECTION_CLOSED

    if isinstance(exc, McpError):
        return exc.error.code == CONNECTION_CLOSED and exc.error.message == "Connection closed"
    return isinstance(exc, (anyio.ClosedResourceError, anyio.BrokenResourceError))


def _release_lease(conn: _Connection) -> None:
    """Return the lease :meth:`MCPManager._lease` took on *conn* (under the manager lock)."""
    conn.in_flight -= 1
    conn.last_used = time.monotonic()


def _start_call(
    future: concurrent.futures.Future[Any],
    conn: _Connection,
    session: Any,
    tool: str,
    arguments: dict[str, Any],
) -> None:
    """Loop-thread side of :meth:`MCPManager.call_tool`: run the call, its outcome in *future*.

    The caller made *future* before it queued this callback, so it holds
    the call's handle wherever a Stop interrupts the submission: a
    *future* cancelled before this runs never starts the call, one
    cancelled later cancels the call's task.  The call is *conn*'s
    until it is settled: a connection that ends first fails it
    (:func:`_fail_calls`).

    Args:
        future: The caller's handle on the call.
        conn: The leased connection.
        session: Its MCP session, as leased.
        tool: The MCP tool name.
        arguments: The tool arguments.
    """
    if future.cancelled():
        return
    task = asyncio.get_running_loop().create_task(
        session.call_tool(tool, arguments, read_timeout_seconds=timedelta(seconds=CALL_TIMEOUT)),
    )
    conn.calls[task] = future
    future.add_done_callback(functools.partial(_cancel_task, task))
    task.add_done_callback(functools.partial(_settle, conn, future))


def _cancel_unless_unwinding(conn: _Connection) -> None:
    """Cancel *conn*'s maintainer unless it is already tearing its transport down.

    Runs on the manager loop, so the check and the cancel cannot be
    split by the task stepping into its teardown.  A maintainer that
    has not started yet is cancelled through its future.
    """
    if conn.unwinding:
        return
    if conn.loop_task is not None:
        conn.loop_task.cancel()
    elif conn.task is not None:
        conn.task.cancel()


def _cancel_task(task: asyncio.Task[Any], future: concurrent.futures.Future[Any]) -> None:
    """Done callback of a call's *future*: once it is cancelled, cancel *task* on its loop."""
    if future.cancelled():
        task.get_loop().call_soon_threadsafe(task.cancel)


def _settle(
    conn: _Connection, future: concurrent.futures.Future[Any], task: asyncio.Task[Any],
) -> None:
    """Done callback of a call's *task*: hand its outcome to the caller's *future*.

    Unless *conn* ended first and failed the call (:func:`_fail_calls`).
    """
    if conn.calls.pop(task, None) is None:
        return
    if task.cancelled():
        future.cancel()
    elif future.set_running_or_notify_cancel():
        error = task.exception()
        if error is None:
            future.set_result(task.result())
        else:
            future.set_exception(error)


class MCPManager:
    """Process-wide manager of live MCP server connections.

    Runs a private asyncio event loop on a daemon thread; every server
    gets one long-lived task that owns its transport + session.  The
    synchronous facade (:meth:`connect_all`, :meth:`call_tool`) is what
    :func:`make_mcp_tools` and the agent tool wrappers use.
    """

    _instance: MCPManager | None = None
    # RLock: :func:`locked_call` needs the owner check on release.  The
    # tool wrappers run on the task thread, where the runner's Stop
    # injects ``KeyboardInterrupt``; a plain ``with`` there can leak
    # the lock (CPython gh-74174) and wedge every later MCP call of
    # the daemon.
    _instance_lock = threading.RLock()

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
        self._connections: dict[str, _Connection] = {}
        # Every configuration ever connected, so a connection dropped by
        # eviction or a server crash can be rebuilt on demand.
        self._configs: dict[str, MCPServerConfig] = {}
        self._orphans: list[_Connection] = []
        self._lock = threading.RLock()
        # Held for the whole of :meth:`disconnect_all` and :meth:`shutdown`
        # (never while waiting under ``_lock``): a shutdown that found the
        # registries emptied by another thread's disconnect stopped and
        # closed the loop underneath the transports that disconnect was
        # still closing.
        self._teardown_lock = threading.RLock()
        self._shut_down = False
        self._idle_timeout_s = idle_timeout_s
        self._max_connections = max_connections
        self._health_interval_s = health_interval_s
        atexit.register(self.shutdown)
        try:
            self._thread.start()
        except BaseException:
            # A Stop landing as the thread starts (the task thread builds
            # the singleton): the construction fails, so nothing would
            # own the loop thread that may be running already.
            self.shutdown()
            raise

    @classmethod
    @locked(_instance_lock)
    def instance(cls) -> MCPManager:
        """Return the process-wide singleton manager."""
        if cls._instance is None:
            cls._instance = MCPManager()
        return cls._instance

    @classmethod
    def disconnect_existing(cls) -> None:
        """Disconnect the singleton's servers when one exists; never create one.

        The daemon's shutdown paths call this: a process that never
        used an MCP server has no manager, and building one (a loop
        thread plus an ``atexit`` hook) just to disconnect an empty
        pool left a live loop thread behind in every embedder or test
        process that stopped a server (conc1018 D1-F2).  The instance
        is read under the lock and disconnected outside it, so a
        manager busy closing its children does not stall
        :meth:`instance` callers.
        """
        with cls._instance_lock:
            manager = cls._instance
        if manager is not None:
            manager.disconnect_all()

    def connect(self, config: MCPServerConfig) -> _Connection:
        """Connect to *config* (reusing a live connection when possible).

        An http/sse server is authenticated by a non-interactive OAuth
        provider (stored tokens only).

        Args:
            config: The server to connect to.

        Returns:
            The connection record; ``error`` is non-empty on failure.
        """
        return self.connect_all((config,))[0]

    def connect_all(self, configs: Iterable[MCPServerConfig]) -> list[_Connection]:
        """Connect to every one of *configs* at once (see :meth:`connect`).

        The handshakes are independent tasks on the manager's loop, so
        they are all started first and then awaited under one shared
        :data:`CONNECT_TIMEOUT`: N servers stuck mid-handshake cost the
        caller (the agent's tool setup, on its task thread) one timeout
        instead of N.  Eviction runs once for the whole batch and
        spares every member of it: evicting per started connection let
        a batch larger than the pool cap tear down its own earliest
        members before the caller had read their tools.

        Args:
            configs: The servers to connect to.

        Returns:
            One connection record per config, in order; ``error`` is
            non-empty on the failed ones.
        """
        started = [self._start(config) for config in configs]
        self._evict_surplus({key for key, _ in started})
        deadline = time.monotonic() + CONNECT_TIMEOUT
        for key, conn in started:
            self._await_ready(key, conn, deadline)
        return [conn for _, conn in started]

    def _start(self, config: MCPServerConfig) -> tuple[str, _Connection]:
        """Start (or reuse) the connection for *config*; return its key and record."""
        key = _connection_key(config)
        return key, locked_call(self._lock, self._connection_for, key, config)

    def _await_ready(self, key: str, conn: _Connection, deadline: float) -> None:
        """Wait until *conn* is ready or *deadline* (``time.monotonic()``) passes.

        A connection that misses the deadline is dropped from the pool
        and its task told to stop (unless its handshake landed in the
        meantime; see :meth:`_orphan_timed_out`); one stuck
        mid-handshake is reaped by :meth:`_reap_straggler`.  A shut-down manager's already-failed
        record (``task is None``) never started, so there is nothing to
        wait for.  The global ``_shut_down`` flag is no evidence about
        *conn*: a shutdown that began after ``_connection_for`` returned
        a pending connection stamps it in ``disconnect_all``, and the
        wait sees that.
        """
        task = conn.task
        if task is None:
            return
        if tool_interrupt.wait_interruptible(conn.ready, max(0.0, deadline - time.monotonic())):
            return
        if locked_call(self._lock, self._orphan_timed_out, key, conn):
            self._reap_straggler(conn, task)

    def _connection_for(self, key: str, config: MCPServerConfig) -> _Connection:
        """Return the live connection under *key*, starting one when there is none (locked)."""
        if self._shut_down:
            return _failed_connection(config, "manager shut down")
        self._configs[key] = config
        existing = self._connections.get(key)
        if existing is not None and existing.error == "":
            existing.last_used = time.monotonic()
            return existing
        conn = _Connection(config=config)
        try:
            auth = build_oauth_provider(conn) if config.transport in ("http", "sse") else None
        except Exception as exc:
            # The SDK's provider refuses a $KISS_MCP_CLIENT_METADATA_URL that is
            # not HTTPS as it is built: this server's failure, not the batch's.
            return _failed_connection(config, describe_exception(exc))
        self._connections[key] = conn
        try:
            conn.task = asyncio.run_coroutine_threadsafe(
                _maintain_connection(conn, auth, self._health_interval_s),
                self._loop,
            )
        except BaseException:
            # A Stop landed before the handle was stored: unpublish the
            # record (every later connect would reuse it as a connection
            # that never becomes ready, its wait returning at once for
            # want of a task) and stop the maintainer in case it was
            # already queued.
            del self._connections[key]
            conn.error = "connection start interrupted"
            conn.ready.set()
            self._loop.call_soon_threadsafe(conn.stop.set)
            raise
        return conn

    def _orphan_timed_out(self, key: str, conn: _Connection) -> bool:
        """Drop the timed-out *conn* from the pool; its unfinished task is an orphan (locked).

        A connection whose handshake finished between the deadline and
        this lock acquisition is healthy and possibly already leased by
        another task, so it is kept: returns ``False`` for it, ``True``
        once *conn* has been orphaned.
        """
        if conn.ready.is_set() and conn.error == "":
            return False
        if self._connections.get(key) is conn:
            del self._connections[key]
        conn.error = conn.error or "connection timed out"
        if conn.task is not None and not conn.task.done():
            self._orphans.append(conn)
        # Told to stop under the lock, as the eviction does: a Stop
        # landing on the task thread once this returns must not leave a
        # connection in no pool whose handshake then lands and parks it
        # for ever, its child alive until the daemon's shutdown.
        self._loop.call_soon_threadsafe(conn.stop.set)
        return True

    def _evict_surplus(self, keep: set[str]) -> None:
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
        ``session.call_tool``, which fails that call with "Connection
        closed".  Tool calls block for as long as the tool runs, so
        the busiest connection is regularly also the least recently
        *started* one.  Nor is a connection still in its handshake:
        evicting it fails the ``connect()`` that is waiting on it with
        "evicted" instead of a session.  Both become ordinary
        candidates again (the last call returns; the handshake ends),
        so the cap still holds — it is enforced a little later.

        Each connection is told to stop before it leaves the pool, under
        the lock (:meth:`_pop_surplus`): this runs on the task thread,
        where a Stop can land between any two statements, and one that
        landed after the pop but before the stop left the connection in
        no registry — its task parked and its child alive for the life
        of the process, invisible to :meth:`disconnect_all`.  Now a Stop
        anywhere in the eviction leaves every connection either still
        in the pool or already stopping.

        Args:
            keep: The connection keys just connected (one, or a whole
                :meth:`connect_all` batch), never evicted.
        """
        locked_call(self._lock, self._pop_surplus, keep)

    def _pop_surplus(self, keep: set[str]) -> None:
        """Stop and remove the idle and over-cap connections (locked)."""
        now = time.monotonic()
        evictable = {
            key for key, conn in self._connections.items()
            if key not in keep and conn.in_flight == 0 and conn.ready.is_set()
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
        for key in doomed:
            conn = self._connections[key]
            conn.error = conn.error or "evicted: idle or over the connection cap"
            self._loop.call_soon_threadsafe(conn.stop.set)
            del self._connections[key]
            # Still owned until its transport has unwound: a
            # ``disconnect_all`` meanwhile must wait for the teardown
            # (and ``shutdown`` not stop the loop underneath it), or a
            # stdio child needing the SDK's SIGTERM outlives the manager.
            if conn.task is not None:
                self._orphans.append(conn)
                self._forget_when_done(conn, conn.task)

    def _forget_when_done(self, conn: _Connection, task: Any) -> None:
        """Drop *conn* from ``_orphans`` once *task* finishes, however it finishes.

        The done callback is registered on the manager loop, never on
        the calling task thread: ``add_done_callback`` takes the
        future's condition with a plain ``with``, and a Stop landing
        right after its ``__enter__`` (CPython gh-74174) would leave
        that condition held by a thread that then dies, so the loop's
        own completion of *task* would hang on it for good.  The loop
        thread receives no injections.

        Args:
            conn: The orphaned connection.
            task: ``conn.task``, the connection's maintainer.
        """
        self._loop.call_soon_threadsafe(
            task.add_done_callback,
            functools.partial(locked_call, self._lock, self._forget_orphan, conn),
        )

    def _reap_straggler(self, conn: _Connection, task: Any) -> None:
        """Arrange teardown of a connection evicted by a connect() timeout.

        Registers a done callback that drops *conn* from ``_orphans``
        once its task finishes (however it finishes), and schedules a
        cancellation on the manager loop after
        :data:`_CONNECT_STRAGGLER_GRACE_S` so a task stuck
        mid-handshake — which the already-set ``stop`` event cannot
        unwind — is cancelled and its transport child reaped.

        Args:
            conn: The timed-out connection whose task may be stuck.
            task: ``conn.task``, the connection's maintainer.
        """
        self._forget_when_done(conn, task)
        # The cancel raises ``CancelledError`` at the task's stuck
        # ``await`` so the transport context unwinds and its child dies;
        # a task whose own handshake deadline already started that
        # teardown is left to finish it.
        self._loop.call_soon_threadsafe(
            self._loop.call_later, _CONNECT_STRAGGLER_GRACE_S, _cancel_unless_unwinding, conn,
        )

    def _take_all(self) -> list[_Connection]:
        """Empty the pool and the orphan list, returning every connection (locked)."""
        conns = list(self._connections.values())
        self._connections.clear()
        conns.extend(self._orphans)
        self._orphans.clear()
        return conns

    def _forget_orphan(self, conn: _Connection, _future: Any) -> None:
        """Drop a finished straggler from ``_orphans`` (done callback; locked)."""
        try:
            self._orphans.remove(conn)
        except ValueError:
            pass

    def _lease(
        self, server: str, leases: list[_Connection],
    ) -> tuple[_Connection | None, Any]:
        """Atomically look up *server*'s connection and lease it.

        The lookup and the ``in_flight`` increment happen under one
        acquisition of the manager lock: incrementing after releasing
        the lookup's lock left a window in which a concurrent
        :meth:`connect`'s :meth:`_evict_surplus` saw ``in_flight == 0``,
        evicted the connection, and tore down the session underneath
        the call about to run on it.  The session is captured under the
        same lock so the caller always uses exactly the session it
        leased.  The leased connection is appended to *leases* with no
        check point between it and the increment, so a caller leasing
        inside its ``try`` returns exactly the leases it took even when
        a Stop's injected ``KeyboardInterrupt`` lands right after one.

        Args:
            server: A connection key, or a bare server name when it is
                unambiguous.
            leases: The caller's record of its leases; the caller
                returns each with :func:`_release_lease`.

        Returns:
            ``(conn, session)`` — *session* is non-``None`` only when
            the connection is live and was leased.
        """
        return locked_call(self._lock, self._lease_locked, server, leases)

    def _lease_locked(
        self, server: str, leases: list[_Connection],
    ) -> tuple[_Connection | None, Any]:
        """Body of :meth:`_lease`; the caller holds the lock."""
        conn = _pick(self._connections, server)
        session = conn.session if conn is not None else None
        if conn is not None and session is not None:
            conn.in_flight += 1
            leases.append(conn)
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
        if self._shut_down:
            return f"Error: MCP server {display!r} is not connected (manager shut down)"
        leases: list[_Connection] = []
        # Made before the call is submitted, so the ``finally`` can cancel
        # the call wherever a Stop lands (``run_coroutine_threadsafe``
        # returned its future only after queueing the call).
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        try:
            conn, session = self._lease(server, leases)
            if session is None:
                # Evicted from the pool, or the server died mid-task: rebuild
                # it rather than failing every remaining call of the run.
                fresh = self._reconnect(server)
                if fresh is not None:
                    leased, session = self._lease(server, leases)
                    conn = leased or fresh
            if conn is None or session is None:
                why = (conn.error if conn is not None else "") or "never connected"
                return f"Error: MCP server {display!r} is not connected ({why})"
            self._loop.call_soon_threadsafe(_start_call, future, conn, session, tool, arguments)
            try:
                result = tool_interrupt.result_interruptible(future, CALL_TIMEOUT + 5)
            except Exception as exc:
                if _transport_closed(exc):
                    locked_call(self._lock, self._retire_closed, conn)
                # The outer deadline's ``TimeoutError`` carries no text.
                why = str(exc) or (
                    f"no answer within {CALL_TIMEOUT + 5:g}s" if isinstance(exc, TimeoutError)
                    else type(exc).__name__
                )
                return f"Error: MCP tool call failed: {why}"
        finally:
            # Leased inside this try, so a Stop landing anywhere after a
            # lease still returns it.  Cancelling is a no-op once the
            # call completed; a timeout, an interrupt or an injected Stop
            # must not leave the call running on the loop with its lease
            # released.  Cancelled on the loop thread, which receives no
            # injections: ``Future.cancel`` takes the future's condition
            # with a plain ``with``, and a Stop landing right after its
            # ``__enter__`` (CPython gh-74174) would leave that condition
            # held by a thread that then dies, with the loop's ``_settle``
            # blocked on it for good.
            if not self._loop.is_closed():
                self._loop.call_soon_threadsafe(future.cancel)
            for leased_conn in leases:
                locked_call(self._lock, _release_lease, leased_conn)
        return _result_text(result)

    def _retire_closed(self, conn: _Connection) -> None:
        """Take *conn*, whose transport closed under a call, out of the pool (locked).

        The server died, and its session stays set until the next idle
        health ping: every call until then would be handed it.  Out of
        the pool, the next call rebuilds the server (:meth:`_reconnect`).
        The failed call is not sent again: its tool may have run.
        """
        key = _connection_key(conn.config)
        if self._connections.get(key) is not conn:
            return  # another call leased on the same dead session got here first
        del self._connections[key]
        conn.error = conn.error or "connection closed"
        self._loop.call_soon_threadsafe(conn.stop.set)
        # Owned until its transport has unwound, as an evicted one is.
        self._orphans.append(conn)
        self._forget_when_done(conn, conn.task)

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
        config = locked_call(self._lock, _pick, self._configs, server)
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
        tore down.  The two waits (for the tasks to unwind, then for
        the cancelled stragglers to finish) each share one deadline
        across every connection, so a shutdown is bounded by
        2 × :data:`_DISCONNECT_WAIT_S` however many servers hang.
        Serialized with :meth:`shutdown` by the teardown lock: a
        shutdown that ran while the connections were taken out of the
        registries but still closing stopped the loop underneath them.
        """
        locked_call(self._teardown_lock, self._disconnect_all_locked)

    def _disconnect_all_locked(self) -> None:
        """Body of :meth:`disconnect_all`; the caller holds the teardown lock."""
        conns = locked_call(self._lock, self._take_all)
        for conn in conns:
            self._loop.call_soon_threadsafe(conn.stop.set)
        deadline = time.monotonic() + _DISCONNECT_WAIT_S
        stragglers: list[_Connection] = []
        for conn in conns:
            if conn.task is None:
                continue
            try:
                conn.task.result(timeout=max(0.0, deadline - time.monotonic()))
            except BaseException:  # noqa: BLE001 — CancelledError is BaseException
                conn.task.cancel()
                logger.debug("MCP disconnect error", exc_info=True)
                stragglers.append(conn)
        deadline = time.monotonic() + _DISCONNECT_WAIT_S
        for conn in stragglers:
            conn.finished.wait(timeout=max(0.0, deadline - time.monotonic()))
        for conn in conns:
            conn.session = None
            conn.error = conn.error or "disconnected"
            conn.ready.set()

    def shutdown(self) -> None:
        """Disconnect everything and stop the manager loop thread.

        Also resets the process-wide singleton when it points at this
        manager, so a later :meth:`instance` call builds a fresh manager
        with a live loop instead of scheduling work on a stopped one
        (which would silently never run and time out every connect).
        Once the loop thread has stopped, the loop is closed (its
        selector and self-pipe descriptors released) and this manager's
        ``atexit`` hook withdrawn, so rebuilt managers leak neither
        descriptors nor the old manager object.  Idempotent: concurrent
        or repeated calls after the first are no-ops.
        """
        with MCPManager._instance_lock:
            if MCPManager._instance is self:
                MCPManager._instance = None
            if self._shut_down:
                return
            self._shut_down = True
        locked_call(self._teardown_lock, self._shutdown_locked)

    def _shutdown_locked(self) -> None:
        """Body of :meth:`shutdown` past its idempotence check (teardown lock held)."""
        # No ``is_running()`` guard: right after construction the loop
        # thread may not have entered ``run_forever`` yet, and a stop
        # scheduled on a not-yet-running loop still takes effect as soon
        # as it starts, so the thread never outlives this call.  A thread
        # never started (its ``start`` interrupted) cannot be joined.
        self.disconnect_all()
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=10)
        if not self._thread.is_alive():
            self._loop.close()
        atexit.unregister(self.shutdown)


def _block_text(block: Any) -> str:
    """Return what the model reads of one content block of a tool result.

    A text block's text, or the text of an embedded resource (how a
    server hands over a file: GitHub's ``get_file_contents``).  Any
    other block (an image, audio, a binary resource, a resource link)
    is a placeholder naming its type and, when it has one, its URI.
    """
    resource = getattr(block, "resource", None)
    for holder in (block, resource):
        text = getattr(holder, "text", None)
        if text is not None:
            return str(text)
    uri = getattr(block, "uri", None) or getattr(resource, "uri", None)
    kind = getattr(block, "type", "content")
    return f"[{kind} content omitted: {uri}]" if uri else f"[{kind} content omitted]"


def _result_text(result: Any) -> str:
    """Flatten a ``CallToolResult`` into the string given to the model."""
    parts = [_block_text(block) for block in getattr(result, "content", None) or []]
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
    one.  So do two projects whose files hold the same stdio entry:
    each runs its own child, in its own directory.  ``alwaysLoad`` is
    left out: it says how the server's tools are offered, not which
    server it is, and changing it must not restart the connection.
    """
    canonical = json.dumps(replace(config, always_load=None).to_json(), sort_keys=True)
    identity = f"{config.name}\n{canonical}\n{config.cwd}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
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


def _json_schema_to_annotation(declared: dict[str, Any]) -> type:
    """Map one reduced schema node to the Python annotation for kiss.

    Args:
        declared: A node :func:`_declared_schema` produced.

    Returns:
        The Python type of the node's ``type``, of its first non-null
        ``anyOf`` member for a union, and ``str`` for anything else.
    """
    kind = declared.get("type") or next(
        (m.get("type") for m in declared.get("anyOf", ()) if m.get("type") != "null"), None,
    )
    return _JSON_TO_PY.get(str(kind), str)


#: JSON-schema ``type`` names every provider's tool declaration accepts.
_JSON_TYPES = frozenset(_JSON_TO_PY) | {"null"}

#: The size of a tool's declaration, in nodes and references followed, at
#: which its references stop being followed: definitions shared down a
#: chain of models multiply (twenty models that each name the next one
#: twice declare the last a million times).
_MAX_DECLARED_NODES = 5000


def _local_ref_target(ref: str, root: dict[str, Any]) -> Any:
    """The node the local JSON pointer *ref* (``#/$defs/Name``) names in *root*.

    Args:
        ref: The ``$ref`` value.
        root: The tool's whole ``inputSchema``.

    Returns:
        The referenced node, or ``None`` for a remote or dangling reference.
    """
    if not ref.startswith("#/"):
        return None
    node: Any = root
    for step in ref[2:].split("/"):
        if not isinstance(node, dict):
            return None
        node = node.get(step.replace("~1", "/").replace("~0", "~"))
    return node


def _schema_parts(
    prop: Any, root: dict[str, Any], resolving: frozenset[str], size: Iterator[int],
) -> Iterator[tuple[dict[str, Any], frozenset[str]]]:
    """Yield *prop* and every node it is declared by, each with the references on its path.

    A node is everything its parts are: itself, then the target of its
    local ``$ref`` (pydantic, so FastMCP, emits a nested model as
    ``{"$ref": "#/$defs/Name"}``), then each member of its ``allOf``,
    every one of them followed by its own parts (an alias, a
    composition of references) before the next.  A reference is on the
    path of what it leads to only, not of the parts beside it: the
    same model named twice in a node is declared twice.  A reference
    already on the path (a self-referencing model, a cycle of aliases)
    gives its target's type only, as does every reference of a
    declaration that has grown to ``_MAX_DECLARED_NODES``.  Walked with
    a stack of its own: a chain of aliases may be longer than the
    interpreter's stack is deep.

    Args:
        prop: The schema node; anything but a mapping has no parts.
        root: The tool's whole ``inputSchema``, where ``$ref`` targets live.
        resolving: The references on the path to this node.
        size: Counts the tool's declaration: each ``next`` is its size
            so far, and adds one to it.
    """
    pending = [(prop, resolving)]
    while pending:
        node, path = pending.pop()
        if not isinstance(node, dict):
            continue
        yield node, path
        parts: list[tuple[Any, frozenset[str]]] = []
        ref = node.get("$ref")
        if isinstance(ref, str):
            target = _local_ref_target(ref, root)
            if isinstance(target, dict):
                followed = ref not in path and next(size) < _MAX_DECLARED_NODES
                parts.append((target if followed else {"type": target.get("type")}, path | {ref}))
        if isinstance(node.get("allOf"), list):
            parts += [(member, path) for member in node["allOf"]]
        pending += reversed(parts)


def _declared_schema(
    prop: Any, root: dict[str, Any], resolving: frozenset[str], size: Iterator[int],
) -> dict[str, Any]:
    """Reduce one MCP ``inputSchema`` node to the keywords every provider accepts.

    Keeps ``type`` (a union list becomes ``anyOf``), ``description``, a
    string ``enum`` and ``required``, and recurses into ``properties``,
    ``items`` and the union members of ``anyOf`` and ``oneOf``, both
    declared as ``anyOf`` (the one union keyword Gemini's declaration
    schema knows; the model picks a member either way).  The node is
    read through its parts (:func:`_schema_parts`: itself, its local
    ``$ref``, its ``allOf`` members): the first part to carry a keyword
    gives it (a description next to a reference wins over the
    target's), and ``properties`` and ``required`` are joined, the
    first part to name a property declaring it: a validator applies
    every part, so an argument any of them requires must be declared.
    Every other keyword (``$schema``, ``$defs``,
    ``additionalProperties``, ``title``, ``default``, ``format``,
    bounds, ...) is left out: Gemini's declaration schema rejects
    keywords it does not know, and the model needs the argument
    structure, not the validation extras.  A node left without a type
    (a dangling reference, a malformed schema) declares an object when
    it has properties, an array when it has items, and a string
    otherwise, as every provider wants a typed node.

    Args:
        prop: The schema node (the root, a property, ``items`` or a
            union member).
        root: The tool's whole ``inputSchema``, where ``$ref`` targets live.
        resolving: The references on the path to this node.
        size: Counts the tool's declaration (see :func:`_schema_parts`).

    Returns:
        The reduced JSON schema of the node.
    """
    next(size)
    # Each keyword with the path of the part it was read from: what it
    # holds (a property, the items, a union member) is reduced on that path.
    first: dict[str, tuple[Any, frozenset[str]]] = {}
    props: dict[str, tuple[Any, frozenset[str]]] | None = None
    required: list[Any] | None = None
    for part, path in _schema_parts(prop, root, resolving, size):
        for key, value in part.items():
            first.setdefault(key, (value, path))
            if key == "properties" and isinstance(value, dict):
                props = props or {}
                for name, child in value.items():
                    props.setdefault(str(name), (child, path))
            elif key == "required" and isinstance(value, list):
                required = [*(required or ()), *value]
    absent: tuple[Any, frozenset[str]] = (None, resolving)
    out: dict[str, Any] = {}
    kind = first.get("type", absent)[0]
    if isinstance(kind, list):
        members = [{"type": k} for k in kind if k in _JSON_TYPES]
        if members:
            out["anyOf"] = members
    elif kind in _JSON_TYPES:
        out["type"] = kind
    description = first.get("description", absent)[0]
    if isinstance(description, str):
        out["description"] = description
    enum = first.get("enum", absent)[0]
    if isinstance(enum, list) and enum and all(isinstance(v, str) for v in enum):
        out["enum"] = enum
    if props is not None:
        out["properties"] = {
            name: _declared_schema(child, root, via, size) for name, (child, via) in props.items()
        }
        if required is not None:
            named = (r for r in required if isinstance(r, str) and r in props)
            out["required"] = list(dict.fromkeys(named))
    if "items" in first:
        items, via = first["items"]
        out["items"] = _declared_schema(items, root, via, size)
    union = [
        (member, via)
        for alternatives, via in (first.get("anyOf", absent), first.get("oneOf", absent))
        if isinstance(alternatives, list)
        for member in alternatives
    ]
    if union:
        out["anyOf"] = [_declared_schema(member, root, via, size) for member, via in union]
    if not out.keys() & {"type", "anyOf"}:
        out["type"] = "object" if "properties" in out else "array" if "items" in out else "string"
    return out


def _admits_null(schema: dict[str, Any]) -> bool:
    """Whether the reduced *schema* accepts ``null`` (itself or through ``anyOf``)."""
    return schema.get("type") == "null" or any(
        _admits_null(member) for member in schema.get("anyOf", ())
    )


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
    manager: MCPManager, server: str, tool: Any, connection_key: str,
) -> Any:
    """Wrap one MCP tool as a kiss-compatible Python function.

    The wrapper's ``__signature__`` is synthesized from the MCP
    ``inputSchema``, its docstring is the tool's description, and the
    schema itself is declared on the wrapper
    (:func:`~kiss.core.tool_schema.tool_schema`, reduced by
    :func:`_declared_schema`, under the Python parameter names) so the
    model sees the tool's arrays, objects, enums and null unions
    instead of the strings a type round trip would leave; calling it
    forwards to :meth:`MCPManager.call_tool`.

    Args:
        manager: The live connection manager.
        server: The configured server name.
        tool: The MCP ``Tool`` (name, description, inputSchema).
        connection_key: The :func:`_connection_key` of the live
            connection to route calls to.

    Returns:
        The wrapper callable, named ``<server>_<tool>``.
    """
    tool_name = str(tool.name)
    full_name = f"{_sanitize(server)}_{_sanitize(tool_name)}"
    root = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
    # The root is reduced like any node: it may name its arguments by
    # reference or composition too.
    schema = _declared_schema(root, root, frozenset(), itertools.count())
    required = schema.get("required", ())

    params: list[inspect.Parameter] = []
    param_map: dict[str, tuple[str, bool]] = {}
    declared_props: dict[str, Any] = {}
    declared_required: list[str] = []
    used_names: set[str] = set()
    for prop_name, declared in schema.get("properties", {}).items():
        is_required = prop_name in required
        py_name = _python_param_name(prop_name, used_names)
        declared_props[py_name] = declared
        if is_required:
            declared_required.append(py_name)
        # ``None`` is forwarded as ``null`` only where the schema admits
        # it: models send ``null`` for optionals they leave unused, which
        # a server would reject for a non-nullable property.
        param_map[py_name] = (str(prop_name), is_required or _admits_null(declared))
        params.append(inspect.Parameter(
            py_name,
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=inspect.Parameter.empty if is_required else None,
            annotation=_json_schema_to_annotation(declared),
        ))
    # A signature wants its required parameters before the defaulted ones.
    params.sort(key=lambda p: p.default is not inspect.Parameter.empty)

    def wrapper(**kwargs: Any) -> str:
        arguments: dict[str, Any] = {}
        for py_name, value in kwargs.items():
            original, keep_none = param_map.get(py_name, (py_name, True))
            if value is None and not keep_none:
                continue
            arguments[original] = value
        return manager.call_tool(connection_key, tool_name, arguments)

    description = _one_line(tool.description or f"MCP tool {tool_name} on server {server}.")
    wrapper.__name__ = full_name
    wrapper.__qualname__ = full_name
    wrapper.__doc__ = description
    wrapper.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        params, return_annotation=str,
    )
    return tool_schema(
        parameters={
            "type": "object",
            "properties": declared_props,
            "required": declared_required,
        },
        description=description,
    )(wrapper)


#: The longest function name the providers' tool grammars accept.
_MAX_TOOL_NAME = 64


def _bounded_tool_name(base: str, counter: int = 0) -> str:
    """*base* (``<base>_<counter>`` for a collision) within :data:`_MAX_TOOL_NAME`.

    A longer name is cut and ends in a digest of the whole, so two long
    names sharing a prefix still differ.

    Args:
        base: The sanitized ``<server>_<tool>`` name.
        counter: The collision suffix, ``0`` for none.

    Returns:
        The name to register.
    """
    suffix = f"_{counter}" if counter else ""
    room = _MAX_TOOL_NAME - len(suffix)
    if len(base) > room:
        base = f"{base[:room - 9]}_{hashlib.sha1(base.encode()).hexdigest()[:8]}"
    return base + suffix


def _claim_tool_name(base: str, taken: set[str]) -> str:
    """Return *base*, or its first numbered variant not in *taken*, and add it to *taken*."""
    name = _bounded_tool_name(base)
    counter = 2
    while name in taken:
        name = _bounded_tool_name(base, counter)
        counter += 1
    taken.add(name)
    return name


# Names of the agent's built-in tools ("cron_job" is built in only to
# cron sessions dispatched via run_agent(agent="cron", ...), whose agent
# SEA's tools() supplies it).  A synthesized MCP tool name colliding with any of
# these (e.g. server "run" + tool "parallel" → "run_parallel") would
# make KISSAgent._add_functions() raise and abort the whole tool loop,
# so such names are pre-reserved and the MCP tool gets a numeric suffix
# instead.  The default of ``make_mcp_tools``' *taken_names*; a caller
# that has its toolset in hand passes the real names instead.
_RESERVED_TOOL_NAMES = frozenset({
    "Bash", "bash_job", "run_commands_parallel", "Read", "Edit", "Write", "finish",
    "create_worktree", "commit_and_merge", "checkpoint", "rollback",
    "go_to_url", "click", "type_text", "press_key", "scroll",
    "screenshot", "get_page_content", "evaluate_script", "list_console_messages",
    "list_network_requests", "wait_for", "attach_to_chrome", "use_my_chrome",
    "show_browser", "close_browser",
    "memory_search", "memory_pull", "memory_read", "memory_write", "memory_list",
    "memory_delete", "memory_refresh",
    "skill", "ask_user_question", "talk", "set_model", "decide",
    "run_parallel", "number_of_cores", "summary",
    "cron_job", "run_agent", "agent_job",
    "connect_mcp_server", "finish_mcp_server_connect",
})


def make_mcp_tools(
    work_dir: str, taken_names: Iterable[str] = _RESERVED_TOOL_NAMES,
) -> list[Any]:
    """Build the agent tools for every configured, permitted MCP server.

    Connects to every server configured for *work_dir* at once (errors
    are logged and the server skipped so a broken server never breaks
    agent startup), lists its tools, filters them through the
    ``mcp_permissions`` wildcard rules, and wraps the survivors as
    kiss tool functions named ``<server>_<tool>``.

    Every tool returned is part of every model request.  The user can
    have servers deferred instead, one by one (``"alwaysLoad": false``)
    or by size (the ``mcp_upfront_tool_chars`` budget;
    :func:`~kiss.agents.sorcar.mcp_tool_search.split_by_budget`): the
    wrappers of a deferred server are not returned, and the tools that
    stand in for them
    (:func:`~kiss.agents.sorcar.mcp_tool_search.make_deferred_tools`),
    ``mcp_tool_search`` and ``mcp_tool_call`` first, are returned last.

    Args:
        work_dir: The project directory whose MCP servers to expose.
        taken_names: Tool names already in use by the agent; a
            synthesized name among them gets a numeric suffix.  Defaults
            to the built-in tool names.

    Returns:
        The (possibly empty) list of tool callables.
    """
    servers = load_mcp_servers(work_dir)
    if not servers:
        return []
    rules = load_permission_rules("mcp_permissions")
    manager = MCPManager.instance()
    offered: list[ServerTools] = []
    taken = set(taken_names)
    connections = manager.connect_all(servers.values())
    for (name, config), conn in zip(servers.items(), connections, strict=True):
        # Wrapped from the tools the handshake discovered, not from the
        # session's presence now: another task's startup enforcing the
        # pool cap can evict an idle, ready connection between the batch
        # returning and this read, and a wrapper reconnects on call
        # where a skipped server is lost to the task for good.
        if conn.session is None and not conn.tools:
            logger.warning(
                "MCP server %s unavailable: %s", name, conn.error or "unknown error",
            )
            continue
        key = _connection_key(config)
        wrappers: list[Any] = []
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
            # Judged on the un-suffixed name: a denied tool is dropped
            # before it can reserve a suffix another tool would get.
            if skill_permission(base, rules) == "deny":
                continue
            wrapper.__name__ = wrapper.__qualname__ = _claim_tool_name(base, taken)
            wrappers.append(wrapper)
        if wrappers:
            offered.append(ServerTools(name, config.always_load, wrappers))
    tools, deferred = split_by_budget(offered, upfront_tool_chars())
    if deferred:
        # Named after every server tool, so a server's own
        # ``mcp_tool_search`` keeps its name and these get the suffix.
        search_name = _claim_tool_name("mcp_tool_search", taken)
        call_name = _claim_tool_name("mcp_tool_call", taken)
        tools += make_deferred_tools(deferred, search_name, call_name)
    return tools
