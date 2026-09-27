# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Interactive OAuth sign-in for remote MCP servers.

Remote MCP servers such as Notion, Linear and Asana follow the MCP
authorization spec: the server names its authorization server, the
client registers itself (Client ID Metadata Document or Dynamic Client
Registration), and the user signs in and clicks Allow.  No KISS-owned
app and no broker sit in between.

:class:`MCPLoginSession` runs that flow in the background with the MCP
SDK's ``OAuthClientProvider``: it opens the server's transport, hands
the authorization URL to the user, receives the redirect on a fixed
loopback port, and leaves the tokens (and the registered client) in
:class:`~kiss.agents.sorcar.mcp_servers.FileTokenStorage`, where later
agent runs reuse and refresh them non-interactively.

Servers whose authorization server allows neither CIMD nor DCR (Zoom)
need a pre-registered app: set ``KISS_MCP_<NAME>_CLIENT_ID`` (and
``KISS_MCP_<NAME>_CLIENT_SECRET`` for a confidential app) and register
:data:`MCP_REDIRECT_URI` with it.  ``KISS_MCP_CLIENT_METADATA_URL`` sets
the HTTPS URL of a hosted Client ID Metadata Document to prefer CIMD.

Command line: ``python -m kiss.agents.sorcar.mcp_oauth <name> [url]``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from kiss.agents.sorcar.mcp_servers import (
    FileTokenStorage,
    MCPServerConfig,
    _enter_transport,
    describe_exception,
    load_mcp_servers,
    save_mcp_server,
)
from kiss.core.brand import PRODUCT_NAME
from kiss.core.browser_handoff import open_in_default_browser

#: Well-known remote MCP servers: name -> (transport, URL).
KNOWN_MCP_SERVERS: dict[str, tuple[str, str]] = {
    "notion": ("http", "https://mcp.notion.com/mcp"),
    "linear": ("http", "https://mcp.linear.app/mcp"),
    "asana": ("sse", "https://mcp.asana.com/sse"),
    "zoom": ("http", "https://mcp.zoom.us/mcp/zoom/streamable"),
}

#: Fixed loopback redirect: a dynamically registered client keeps
#: working across sign-ins only if its redirect URI never changes.
MCP_REDIRECT_PORT = 53683
MCP_REDIRECT_URI = f"http://localhost:{MCP_REDIRECT_PORT}/callback"

_LOGIN_TIMEOUT = 600.0


def client_metadata() -> Any:
    """Return the OAuth client metadata KISS registers with MCP servers.

    KISS is a public client: it proves possession with PKCE and has no
    secret, so ``token_endpoint_auth_method`` is ``none``.

    Returns:
        An ``mcp.shared.auth.OAuthClientMetadata``.
    """
    from mcp.shared.auth import OAuthClientMetadata

    return OAuthClientMetadata.model_validate({
        "client_name": PRODUCT_NAME,
        "redirect_uris": [MCP_REDIRECT_URI],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    })


def _env_name(server: str) -> str:
    """Return the environment-variable form of a server name."""
    return "".join(c if c.isalnum() else "_" for c in server).upper()


def seed_preregistered_client(storage: FileTokenStorage, server: str) -> bool:
    """Store a pre-registered client for *server* from the environment.

    Args:
        storage: The server's token storage.
        server: Server name.

    Returns:
        True when ``KISS_MCP_<NAME>_CLIENT_ID`` was set and stored.
    """
    from mcp.shared.auth import OAuthClientInformationFull

    prefix = f"KISS_MCP_{_env_name(server)}_CLIENT"
    client_id = os.environ.get(f"{prefix}_ID", "").strip()
    if not client_id:
        return False
    secret = os.environ.get(f"{prefix}_SECRET", "").strip()
    info = OAuthClientInformationFull.model_validate({
        **client_metadata().model_dump(mode="json", exclude_none=True),
        "client_id": client_id,
        "client_secret": secret or None,
        "token_endpoint_auth_method": "client_secret_basic" if secret else "none",
    })
    storage._locked_update("client_info", info.model_dump(mode="json", exclude_none=True))
    return True


def build_provider(cfg: MCPServerConfig, redirect_handler: Any, callback_handler: Any) -> Any:
    """Build the MCP SDK OAuth provider for *cfg*.

    Args:
        cfg: Remote server configuration.
        redirect_handler: ``async (url) -> None`` shown the authorization URL.
        callback_handler: ``async () -> (code, state)`` awaiting the redirect.

    Returns:
        An ``OAuthClientProvider`` (an ``httpx.Auth``).
    """
    from mcp.client.auth import OAuthClientProvider

    storage = FileTokenStorage(cfg.name)
    provider = OAuthClientProvider(
        server_url=cfg.url,
        client_metadata=client_metadata(),
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
        timeout=_LOGIN_TIMEOUT,
        client_metadata_url=os.environ.get("KISS_MCP_CLIENT_METADATA_URL", "").strip() or None,
    )
    provider.context.oauth_metadata = storage.get_oauth_metadata()
    return provider


def drop_stale_registration(storage: FileTokenStorage) -> bool:
    """Forget a stored client registration that lacks the current redirect URI.

    The SDK skips registration whenever client information is stored and
    then authorizes with the *current* redirect URI, which the server
    rejects when the registration named another one (older KISS versions
    registered ``http://localhost:0/callback``).

    Args:
        storage: The server's token storage.

    Returns:
        True when a stale registration (and its tokens) was removed.
    """
    info = asyncio.run(storage.get_client_info())
    if info is None:
        return False
    if MCP_REDIRECT_URI in [str(uri) for uri in (info.redirect_uris or [])]:
        return False
    storage._locked_update("client_info", None)
    storage._locked_update("tokens", None)
    return True


class _RedirectHandler(BaseHTTPRequestHandler):
    """Records the OAuth redirect for the login session."""

    server: _RedirectServer  # type: ignore[assignment]
    # Socket timeout of an accepted connection: a half-open browser
    # request must not pin the serving thread (and the fixed port) forever.
    timeout = 5

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        """Record the redirect query and show a completion page."""
        parts = urlsplit(self.path)
        if parts.path != "/callback":
            self.send_error(404)
            return
        self.server.query = dict(parse_qsl(parts.query))
        body = b"Sign-in received. You can close this tab and return to the chat."
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:  # type: ignore[override]
        """Silence per-request logging."""


class _RedirectServer(HTTPServer):
    """Loopback server receiving the authorization redirect."""

    allow_reuse_address = True

    def __init__(self) -> None:
        self.query: dict[str, str] | None = None
        super().__init__(("127.0.0.1", MCP_REDIRECT_PORT), _RedirectHandler)


class MCPLoginSession:
    """One background OAuth sign-in to a remote MCP server.

    At most one session runs at a time (the redirect port is fixed);
    starting a new one cancels the previous one.

    Args:
        cfg: The remote server configuration.
    """

    _active: MCPLoginSession | None = None
    _lock = threading.Lock()

    def __init__(self, cfg: MCPServerConfig) -> None:
        self.cfg = cfg
        self.auth_url = ""
        self.url_shown = False
        self.done = False
        self.error = ""
        self._cancelled = False
        self._url_ready = threading.Event()
        self._finished = threading.Event()
        self._server: _RedirectServer | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    @classmethod
    def start(cls, cfg: MCPServerConfig, wait_seconds: float = 30.0) -> MCPLoginSession:
        """Start signing in to *cfg* and wait for the authorization URL.

        Args:
            cfg: The remote server configuration.
            wait_seconds: How long to wait for the URL (or for success
                when stored tokens already work).

        Returns:
            The session; check ``auth_url``, ``done`` and ``error``.
        """
        storage = FileTokenStorage(cfg.name)
        if not seed_preregistered_client(storage, cfg.name):
            drop_stale_registration(storage)
        with cls._lock:
            previous = cls._active
            if previous is not None:
                previous.cancel()
                previous._finished.wait(timeout=5.0)
            session = cls(cfg)
            session._server = _RedirectServer()
            session._server.timeout = 0.25
            cls._active = session
            session._thread.start()
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline and not session._finished.is_set():
            if session._url_ready.wait(timeout=0.1):
                break
        return session

    @classmethod
    def active(cls, name: str) -> MCPLoginSession | None:
        """Return the running session for server *name*, if any."""
        with cls._lock:
            session = cls._active
        return session if session is not None and session.cfg.name == name else None

    def cancel(self) -> None:
        """Stop the session; its thread exits at the next wake-up."""
        self._cancelled = True

    def wait(self, seconds: float) -> bool:
        """Wait up to *seconds* for the session to end; return True if it did."""
        return self._finished.wait(timeout=seconds)

    def _run(self) -> None:
        """Run the async sign-in on this thread's own event loop."""
        try:
            asyncio.run(self._login())
            self.done = not self._cancelled
            if self._cancelled:
                self.error = "sign-in cancelled"
        except BaseException as exc:
            self.error = describe_exception(exc)
        finally:
            if self._server is not None:
                self._server.server_close()
            self._finished.set()

    async def _login(self) -> None:
        """Open the transport with the OAuth provider and initialize."""
        from contextlib import AsyncExitStack

        from mcp import ClientSession

        auth = build_provider(self.cfg, self._on_redirect, self._await_callback)
        async with AsyncExitStack() as stack:
            read, write = await _enter_transport(stack, self.cfg, auth)
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        if auth.context.oauth_metadata is not None:
            FileTokenStorage(self.cfg.name).set_oauth_metadata(auth.context.oauth_metadata)

    async def _on_redirect(self, url: str) -> None:
        """Publish the authorization URL and open it for the user."""
        self.auth_url = url
        self._url_ready.set()
        await asyncio.to_thread(open_in_default_browser, url)

    async def _await_callback(self) -> tuple[str, str | None]:
        """Serve the loopback port until the redirect arrives."""
        server = self._server
        assert server is not None
        deadline = time.monotonic() + _LOGIN_TIMEOUT
        while server.query is None:
            if self._cancelled or time.monotonic() > deadline:
                raise RuntimeError("sign-in was not completed")
            await asyncio.to_thread(server.handle_request)
        query = server.query
        if query.get("error"):
            raise RuntimeError(f"sign-in refused ({query['error']})")
        return query.get("code", ""), query.get("state")


def resolve_server(name: str, url: str, transport: str, work_dir: str) -> MCPServerConfig:
    """Find or configure the remote MCP server to sign in to.

    Args:
        name: Configured server name, or a known one (notion, linear,
            asana, zoom).
        url: Server URL for a new server (empty for known/configured).
        transport: ``http`` or ``sse`` for a new server (default http).
        work_dir: Project directory (for project-scope configs).

    Returns:
        The server configuration (saved to ``~/.kiss/mcp.json`` when new).

    Raises:
        ValueError: When the server is unknown and no URL was given, the
            configured server is not a remote one, or a different URL is
            given for a configured name.
    """
    configured = load_mcp_servers(work_dir).get(name)
    if configured is not None:
        if url and url != configured.url:
            raise ValueError(
                f"MCP server {name!r} is already configured with {configured.url}; edit its "
                "config or choose another name"
            )
        if configured.transport not in ("http", "sse"):
            raise ValueError(f"MCP server {name!r} is a local stdio server; it needs no sign-in")
        return configured
    if not url:
        if name not in KNOWN_MCP_SERVERS:
            raise ValueError(
                f"Unknown MCP server {name!r}: pass its url (known servers: "
                f"{', '.join(sorted(KNOWN_MCP_SERVERS))})"
            )
        transport, url = KNOWN_MCP_SERVERS[name]
    cfg = MCPServerConfig(name=name, transport=transport or "http", url=url)
    save_mcp_server(cfg, "user", work_dir)
    return cfg


def _session_answer(session: MCPLoginSession) -> dict[str, Any]:
    """Build the tool answer describing a login session's state."""
    name = session.cfg.name
    if session.done:
        return {
            "ok": True,
            "message": f"MCP server {name!r} is connected; its tools load in the next task.",
        }
    if session.error:
        return {"ok": False, "error": f"MCP sign-in to {name!r} failed: {session.error}"}
    if not session.auth_url:
        return {"ok": False, "status": "pending", "error": "still contacting the server; retry"}
    session.url_shown = True
    return {
        "ok": True,
        "status": "consent_required",
        "verification_uri": session.auth_url,
        "instructions": (
            f"The USER signs in and clicks Allow for MCP server {name!r}; you only "
            "relay the link. Do NOT open it in your built-in browser and never ask "
            "for passwords or 2FA codes. 1) Call ask_user_question() with this URL "
            f"for the user to open in their OWN browser if no window appeared: "
            f"{session.auth_url} 2) Call finish_mcp_server_connect({name!r}); if it "
            "returns 'pending', wait a few seconds and call it again. If the user "
            "approved on ANOTHER device, their browser ends on an unreachable "
            "http://localhost:53683/callback?... page: ask them to paste that URL "
            "and deliver it here with Bash: curl -s '<pasted URL>' (quoted)."
        ),
    }


def make_mcp_auth_tools(work_dir: str) -> list[Any]:
    """Return the agent tools that sign in to remote MCP servers.

    Args:
        work_dir: Project directory used to resolve configured servers.

    Returns:
        ``[connect_mcp_server, finish_mcp_server_connect]``.
    """

    def connect_mcp_server(name: str, url: str = "", transport: str = "") -> str:
        """Sign in to a remote MCP server (Notion, Linear, Asana, Zoom, or any URL).

        Configures the server in ~/.kiss/mcp.json when it is new, then
        starts the OAuth sign-in: the user opens the returned URL, signs
        in, and clicks Allow.  Finish with finish_mcp_server_connect().

        Args:
            name: Server name: a configured one, or notion, linear,
                asana, zoom.
            url: MCP endpoint URL for a server that is neither
                configured nor known.
            transport: "http" (default) or "sse" for a new URL.

        Returns:
            JSON: consent_required with the URL, success when stored
            tokens already work, or an error.
        """
        try:
            cfg = resolve_server(name.strip(), url.strip(), transport.strip(), work_dir)
        except ValueError as e:
            return json.dumps({"ok": False, "error": str(e)})
        try:
            session = MCPLoginSession.start(cfg)
        except OSError as e:
            return json.dumps({"ok": False, "error": f"cannot bind the redirect port: {e}"})
        return json.dumps(_session_answer(session))

    def finish_mcp_server_connect(name: str) -> str:
        """Complete a sign-in started by connect_mcp_server().

        Args:
            name: The server name passed to connect_mcp_server().

        Returns:
            JSON: success, a pending status while the user has not
            approved yet, or an error.
        """
        session = MCPLoginSession.active(name.strip())
        if session is None:
            return json.dumps({
                "ok": False,
                "error": f"no sign-in in progress for {name!r}; call connect_mcp_server() first",
            })
        if not session.wait(5.0):
            if session.auth_url and not session.url_shown:
                # Discovery outlasted connect's wait: hand the URL over now.
                return json.dumps(_session_answer(session))
            return json.dumps({
                "ok": False,
                "status": "pending",
                "error": "The user has not approved yet; call this tool again shortly.",
            })
        return json.dumps(_session_answer(session))

    return [connect_mcp_server, finish_mcp_server_connect]


def main(argv: list[str] | None = None) -> int:
    """Sign in to a remote MCP server from the command line.

    Usage: ``python -m kiss.agents.sorcar.mcp_oauth <name> [url] [transport]``.

    Args:
        argv: Command-line arguments (default ``sys.argv[1:]``).

    Returns:
        Process exit status.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 2
    name, url, transport = (args + ["", ""])[:3]
    try:
        cfg = resolve_server(name, url, transport, os.getcwd())
        session = MCPLoginSession.start(cfg)
    except (ValueError, OSError) as e:
        print(f"error: {e}")
        return 1
    finished = False
    while not finished:
        finished = session.wait(1.0)
        if session.auth_url and not session.url_shown:
            session.url_shown = True
            print(f"Open this URL, sign in, and click Allow:\n{session.auth_url}")
    answer = _session_answer(session)
    print(answer.get("message") or answer.get("error"))
    return 0 if session.done else 1


if __name__ == "__main__":
    sys.exit(main())
