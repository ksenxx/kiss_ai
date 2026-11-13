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
import contextlib
import functools
import json
import os
import socket
import sys
import threading
import time
from collections.abc import AsyncGenerator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from kiss.agents.sorcar.mcp_servers import (
    CONNECT_TIMEOUT,
    FileTokenStorage,
    MCPServerConfig,
    _enter_transport,
    describe_exception,
    load_mcp_servers,
    save_mcp_server,
)
from kiss.core import tool_interrupt
from kiss.core.brand import PRODUCT_NAME
from kiss.core.browser_handoff import BROWSER_TAB, open_for_user
from kiss.core.locked import locked_call

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
    storage.put("client_info", info.model_dump(mode="json", exclude_none=True))
    return True


@functools.cache
def _provider_class() -> Any:
    """Return the SDK's OAuth provider, made to share its stored grant.

    Defined on first use: the SDK is imported only once a remote server is used.
    """
    from mcp.client.auth import OAuthClientProvider

    class SharedGrantProvider(OAuthClientProvider):
        """An ``OAuthClientProvider`` over a grant that other providers refresh too.

        The SDK loads the stored grant once and refreshes its own copy.
        Each request here first re-reads the store (tokens, client and
        the refresh deadline) under the store's refresh lock, and the
        SDK's refresh, if still due, runs under that lock.  A refresh
        answered without a refresh token keeps the one it used.  The
        authorization server metadata a sign-in discovers is stored
        ahead of the grant it leads to (:meth:`_keep_metadata`).
        """

        def __init__(self, storage: FileTokenStorage, **kwargs: Any) -> None:
            super().__init__(storage=storage, **kwargs)
            self._store = storage
            # One request at a time, as under the SDK's own lock, but
            # taken before the refresh lock: a request queued behind a
            # slow one must not hold up the grant's other providers.
            self._flow_lock = asyncio.Lock()
            # The stored metadata: a refresh goes to its token endpoint.
            self._metadata = self.context.oauth_metadata = storage.get_oauth_metadata()

        async def async_auth_flow(self, request: Any) -> AsyncGenerator[Any, Any]:
            """Run the SDK's flow for *request*, refreshing under the store's lock.

            A 401 for an access token the store held as good (its expiry
            was not recorded, or the server ended it early) is answered
            once like a recorded expiry: by a refresh and the request
            again.  Only a 401 after that is the SDK's to answer, with a
            new sign-in.  The request it answered was refused, not run.
            """
            context, store = self.context, self._store
            async with self._flow_lock:
                rejected = None  # the access token a 401 answered
                while True:
                    flow = super().async_auth_flow(request)
                    try:
                        async with store.refreshing():
                            context.current_tokens = stored = await store.get_tokens()
                            context.client_info = await store.get_client_info()
                            context.token_expiry_time = store.refresh_deadline()
                            if stored and stored.access_token == rejected:
                                # As good as expired: the SDK refreshes it first.
                                context.token_expiry_time = time.time() - 1
                            outgoing = await flow.__anext__()
                            refreshed = outgoing is not request
                            while outgoing is not request:  # the refresh round trip
                                outgoing = await flow.asend((yield outgoing))
                                await self._keep_refresh_token(stored)
                        response = yield outgoing
                        sent = context.current_tokens
                        if (
                            response.status_code == 401 and rejected is None
                            and not refreshed and sent and context.can_refresh_token()
                        ):
                            rejected = sent.access_token
                            continue
                        outgoing = await flow.asend(response)
                        while True:
                            await self._keep_metadata()
                            outgoing = await flow.asend((yield outgoing))
                    except StopAsyncIteration:
                        return
                    finally:
                        await flow.aclose()

        async def _keep_refresh_token(self, stored: Any) -> None:
            """Give a grant refreshed without a refresh token the one of *stored*, its predecessor.

            An authorization server that does not rotate its refresh
            tokens leaves it out of the refresh response: the one in use
            stays good (RFC 6749 section 6).  The SDK stores the response
            whole, and the next expiry would find nothing to refresh
            with and ask for a new sign-in.
            """
            fresh = self.context.current_tokens  # None when the refresh failed
            if fresh and not fresh.refresh_token and stored and stored.refresh_token:
                fresh.refresh_token = stored.refresh_token
                await self._store.set_tokens(fresh)

        async def _keep_metadata(self) -> None:
            """Store the authorization server metadata once the flow has discovered it.

            Discovery comes before the token request, so no grant is
            stored without the token endpoint its refresh goes to,
            however the connection the sign-in runs on ends afterwards
            (a refresh without it is sent to ``/token`` of the server).
            On a worker thread, as :meth:`FileTokenStorage.set_tokens`.
            """
            metadata = self.context.oauth_metadata
            if metadata is not None and metadata != self._metadata:
                self._metadata = metadata
                await asyncio.to_thread(self._store.set_oauth_metadata, metadata)

    return SharedGrantProvider


def build_provider(cfg: MCPServerConfig, redirect_handler: Any, callback_handler: Any) -> Any:
    """Build the MCP SDK OAuth provider for *cfg*.

    Args:
        cfg: Remote server configuration.
        redirect_handler: ``async (url) -> None`` shown the authorization URL.
        callback_handler: ``async () -> (code, state)`` awaiting the redirect.

    Returns:
        An ``OAuthClientProvider`` (an ``httpx.Auth``).
    """
    # Bound to the URL: a same-named configuration of another server
    # must not be handed this one's credentials (see FileTokenStorage).
    return _provider_class()(
        server_url=cfg.url,
        client_metadata=client_metadata(),
        storage=FileTokenStorage(cfg.name, cfg.url),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
        client_metadata_url=os.environ.get("KISS_MCP_CLIENT_METADATA_URL", "").strip() or None,
    )


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
    info = storage.client_info()
    if info is None:
        return False
    if MCP_REDIRECT_URI in [str(uri) for uri in (info.redirect_uris or [])]:
        return False
    storage.clear()
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
        query = dict(parse_qsl(parts.query))
        if query.get("state") != self.server.state:
            # An earlier sign-in's page (the port is fixed), or no OAuth
            # redirect at all: the session keeps waiting for its own.
            self.send_error(400, "This sign-in page is out of date")
            return
        self.server.query = query
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

    # Poll interval of ``handle_request``: how quickly the login thread
    # notices a cancel.
    timeout = 0.25

    def __init__(self) -> None:
        self.query: dict[str, str] | None = None
        # The ``state`` of the authorization URL the session opened; only
        # a redirect that carries it is recorded in ``query``.
        self.state = ""
        # The browser connection being served, so :meth:`server_close`
        # can end it: the handler's socket timeout is an inactivity
        # timeout, which a connection that keeps sending header bytes
        # never trips, and the flow cannot finish while it is served.
        self.accepted: socket.socket | None = None
        super().__init__(("127.0.0.1", MCP_REDIRECT_PORT), _RedirectHandler)

    def process_request(self, request: Any, client_address: Any) -> None:
        """Serve *request* (one accepted connection), remembering it meanwhile.

        Recorded before the listener is checked, and :meth:`server_close`
        closes the listener before it reads the record: a close that
        lands between the accept and here is seen by one side or the other.
        """
        self.accepted = request
        try:
            if self.socket.fileno() < 0:  # closed since the accept
                self.shutdown_request(request)
            else:
                super().process_request(request, client_address)
        finally:
            self.accepted = None

    def server_close(self) -> None:
        """Close the listener and end the connection being served, if any."""
        super().server_close()
        accepted = self.accepted
        if accepted is not None:
            with contextlib.suppress(OSError):
                accepted.shutdown(socket.SHUT_RDWR)


class MCPLoginSession:
    """One background OAuth sign-in to a remote MCP server.

    At most one session runs at a time (the redirect port is fixed);
    starting a new one cancels the previous one.

    Args:
        cfg: The remote server configuration.
    """

    # Written under ``_start_lock`` only, which serialises whole
    # ``start()`` calls (cancel the previous session, bind the port);
    # read without a lock, so ``active()`` answers at once.  The lock is
    # taken through :func:`locked_call` (hence an RLock): the
    # ``connect_mcp_server`` tools run on the task thread, where the
    # runner's Stop injection would leak a plain ``with`` (CPython
    # gh-74174) and refuse every later sign-in.
    _active: MCPLoginSession | None = None
    _start_lock = threading.RLock()

    def __init__(self, cfg: MCPServerConfig) -> None:
        self.cfg = cfg
        self.auth_url = ""
        self.opened_in = ""  # where open_for_user() put the sign-in page
        self.url_shown = False
        self.done = False
        self.error = ""
        self._cancelled = False
        self._url_ready = threading.Event()
        self._finished = threading.Event()
        self._server: _RedirectServer | None = None
        # The flow's loop and task, set by ``_login`` so ``cancel`` can
        # cancel the flow from another thread.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[Any] | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    @classmethod
    def start(cls, cfg: MCPServerConfig, wait_seconds: float = 30.0) -> MCPLoginSession:
        """Start signing in to *cfg* and wait for the authorization URL.

        Args:
            cfg: The remote server configuration.
            wait_seconds: How long to wait for the URL (or for success
                when stored tokens already work).

        Returns:
            The session; check ``auth_url``, ``done`` and ``error``.  A
            sign-in whose token store cannot be prepared is returned
            ended, with that error, and replaces no running session.

        Raises:
            OSError: When the redirect port cannot be bound.
            ToolCallInterrupted: When the calling tool call was stopped
                while waiting (checked between slices, as in :meth:`wait`).
        """
        storage = FileTokenStorage(cfg.name, cfg.url)
        try:
            if not seed_preregistered_client(storage, cfg.name):
                drop_stale_registration(storage)
        except OSError as e:
            failed = cls(cfg)
            failed.error = describe_exception(e)
            failed._finished.set()
            return failed
        session = locked_call(cls._start_lock, cls._replace_active, cfg)
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline and not session._finished.is_set():
            if session._url_ready.wait(timeout=0.1):
                break
            tool_interrupt.raise_if_interrupted()
        return session

    @classmethod
    def _replace_active(cls, cfg: MCPServerConfig) -> MCPLoginSession:
        """Cancel the active session, if any, and start one for *cfg* (under ``_start_lock``)."""
        previous = cls._active
        if previous is not None:
            previous.cancel()
            previous.wait(5.0)
        session = cls(cfg)
        try:
            session._server = _RedirectServer()
        except BaseException:
            # The previous session stays published until the new one
            # replaces it, so ``active()`` never answers ``None`` in the
            # middle of a replacement; should the bind fail, no caller
            # must find the cancelled session still active.
            cls._active = None
            raise
        try:
            cls._active = session
            session._thread.start()
        except BaseException:
            # A Stop landing here (the tool runs on the task thread), or
            # a thread that cannot start: nothing would ever finish the
            # session, and no later sign-in could bind its port.
            session.cancel()
            session.error = "sign-in cancelled"
            session._finished.set()
            cls._active = None
            raise
        return session

    @classmethod
    def active(cls, name: str) -> MCPLoginSession | None:
        """Return the running session for server *name*, if any."""
        session = cls._active
        return session if session is not None and session.cfg.name == name else None

    def cancel(self) -> None:
        """Stop the session and free the redirect port at once.

        The flow's task is cancelled on its own loop, so a login sitting
        in a network call to the MCP server (a first request the server
        never answers) unwinds at once instead of holding the thread
        until the transport's read timeout; closing the listener here
        lets a replacement session bind the port without waiting even
        for that.
        """
        self._cancelled = True
        loop, task = self._loop, self._task
        if loop is not None and task is not None:
            # A finished flow has closed its loop: nothing left to cancel.
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(task.cancel)
        if self._server is not None:
            self._server.server_close()

    def wait(self, seconds: float) -> bool:
        """Wait up to *seconds* for the session to end; return True if it did.

        Waits in short slices
        (:func:`~kiss.core.tool_interrupt.wait_interruptible`): the
        sign-in tools call this on the task thread, where an injected
        Stop lands only between C-level waits and the tool-panel Stop is
        honoured through
        :func:`~kiss.core.tool_interrupt.raise_if_interrupted` (a no-op
        outside a tool call, as from the command line).

        Raises:
            ToolCallInterrupted: When the calling tool call was stopped.
        """
        return tool_interrupt.wait_interruptible(self._finished, seconds)

    def _run(self) -> None:
        """Run the async sign-in on this thread's own event loop."""
        try:
            asyncio.run(self._login())
            self.done = not self._cancelled
            if self._cancelled:
                self.error = "sign-in cancelled"
        except BaseException as exc:
            # A cancel closes the listener under the running flow, so
            # whatever the flow raised, the cause is the cancel.
            self.error = "sign-in cancelled" if self._cancelled else describe_exception(exc)
        finally:
            if self._server is not None:
                self._server.server_close()
            self._finished.set()

    async def _login(self) -> None:
        """Open the transport with the OAuth provider and initialize.

        The whole flow is bounded by :data:`_LOGIN_TIMEOUT` (the user's
        consent allowance, enforced again by :meth:`_await_callback`)
        plus :data:`~kiss.agents.sorcar.mcp_servers.CONNECT_TIMEOUT`
        for the handshake itself: the SDK's session has no read
        timeout, so a server that accepts the transport and never
        answers ``initialize`` would otherwise hold the thread for ever.
        """
        from contextlib import AsyncExitStack

        from mcp import ClientSession

        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        # A cancel that landed before the task was published above
        # could not cancel it; it is honoured here instead.
        if self._cancelled:
            raise RuntimeError("sign-in cancelled")
        auth = build_provider(self.cfg, self._on_redirect, self._await_callback)
        async with asyncio.timeout(_LOGIN_TIMEOUT + CONNECT_TIMEOUT), AsyncExitStack() as stack:
            read, write = await _enter_transport(stack, self.cfg, auth)
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

    @property
    def ready(self) -> bool:
        """Whether the authorization URL is known AND its hand-off is settled.

        ``auth_url`` alone is not enough for a tool answer: where the
        page went (Browser tab, default browser, nowhere) decides how
        the agent must word the hand-off, so answers wait for this.
        """
        return self._url_ready.is_set()

    async def _on_redirect(self, url: str) -> None:
        """Publish the authorization URL and open it for the user.

        The opening runs on its own daemon thread: the loopback callback
        is only served once this returns, and a provider that approves
        at once redirects there before the page finishes loading (which
        the Browser tab waits for), so opening inline would deadlock.
        A plain thread rather than ``asyncio.to_thread``, because
        ``asyncio.run`` joins its default executor on exit: the flow
        would otherwise not count as finished until the opener returned
        (up to its page-load wait), even though the sign-in had already
        succeeded.  ``ready`` turns true when the opening outcome is
        known, so the first tool answer is worded right; ``done`` and
        ``error`` take precedence over it in the answer, so an opener
        that outlives the flow changes nothing.

        A session cancelled while its first request to the server was
        still in flight (a replacement session took over after the 5 s
        wait of ``_replace_active``) ends here instead: its redirect
        listener is already closed, so the page would be a dead sign-in
        next to the replacement's.  ``_run`` reports the cancel.
        """
        if self._cancelled:
            raise RuntimeError("sign-in cancelled")
        server = self._server
        assert server is not None
        server.state = dict(parse_qsl(urlsplit(url).query)).get("state", "")
        self.auth_url = url
        threading.Thread(target=self._open_for_user, args=(url,), daemon=True).start()

    def _open_for_user(self, url: str) -> None:
        """Open the sign-in page *url*, unless the session was cancelled before this thread ran.

        A replacement session may have its own page open by then, which
        this one would take every surface away from.
        """
        if self._cancelled:
            return
        self.opened_in = open_for_user(url)
        self._url_ready.set()

    async def _await_callback(self) -> tuple[str, str | None]:
        """Serve the loopback port until the redirect arrives.

        A flow that fails here closes the server itself (``_run`` closes
        it again, harmlessly): the deadline of :meth:`_login` cancels
        this task while ``handle_request`` may be serving a connection
        that keeps sending header bytes (its inactivity timeout never
        fires), ``to_thread`` cannot stop that worker, and ``asyncio.run``
        waits for it (five minutes at most) before ``_run`` gets to close
        the server.  Closing it here ends the connection, and the worker
        with it, before that wait.
        """
        server = self._server
        assert server is not None
        deadline = time.monotonic() + _LOGIN_TIMEOUT
        try:
            while server.query is None:
                if self._cancelled or time.monotonic() > deadline:
                    raise RuntimeError("sign-in was not completed")
                await asyncio.to_thread(server.handle_request)
        except BaseException:
            server.server_close()
            raise
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
        The server configuration (saved to ``$KISS_HOME/mcp.json`` when new).

    Raises:
        ValueError: When the server is unknown and no URL was given, the
            configured server is not a remote one, a different URL is
            given for a configured name, a new server has no name or a
            transport that is no remote one, or it cannot be saved
            because ``$KISS_HOME/mcp.json`` does not parse (see
            :func:`~kiss.agents.sorcar.mcp_servers.save_mcp_server`).
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
    elif not name or transport not in ("", "http", "sse"):
        # Saved as given, a "stdio" entry would drop the URL and launch no command.
        raise ValueError(
            "A new MCP server needs a name and transport http or sse "
            f"(got name {name!r}, transport {transport!r})"
        )
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
    if not session.ready:
        return {
            "ok": False,
            "status": "pending",
            "error": "still contacting the server or opening the sign-in page; retry",
        }
    session.url_shown = True
    if session.opened_in == BROWSER_TAB:
        step_one = (
            "The sign-in page is already open in the Browser tab that every KISS "
            "surface has just switched to, so the user is looking at it: do NOT ask "
            "them to open a URL. 1) Call ask_user_question() telling the user to sign "
            "in and click Allow in that Browser tab and reply here when done; only if "
            f"they cannot see the page, give them {session.auth_url} to open themselves."
        )
    else:
        step_one = (
            "1) Call ask_user_question() with this URL for the user to open in their "
            f"OWN browser if no window appeared: {session.auth_url}"
        )
    return {
        "ok": True,
        "status": "consent_required",
        "verification_uri": session.auth_url,
        "opened_in": session.opened_in,
        "instructions": (
            f"The USER signs in and clicks Allow for MCP server {name!r}; you only "
            "tell them what to do. Do NOT open the page in your built-in browser and "
            f"never ask for passwords or 2FA codes. {step_one} 2) Call "
            f"finish_mcp_server_connect({name!r}); if it returns 'pending', wait a "
            "few seconds and call it again. If the user approved on ANOTHER device, "
            f"their browser ends on an unreachable {MCP_REDIRECT_URI}?... "
            "page: ask them to paste that URL and deliver it here with Bash: "
            "curl -s '<pasted URL>' (quoted)."
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

        Configures the server in $KISS_HOME/mcp.json when it is new, then
        starts the OAuth sign-in and opens its page for the user (the
        Browser tab on every KISS surface, else their default browser;
        see 'opened_in'): follow the returned 'instructions' so the user
        signs in and clicks Allow.  Finish with finish_mcp_server_connect().

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
        name = name.strip()
        session = MCPLoginSession.active(name)
        # A server is its name at its URL (as its stored grant is): the
        # same name in another project may be another server, whose
        # sign-in is not this project's to finish.
        mine = load_mcp_servers(work_dir).get(name)
        if session is None or mine is None or mine.url != session.cfg.url:
            return json.dumps({
                "ok": False,
                "error": f"no sign-in in progress for {name!r}; call connect_mcp_server() first",
            })
        if not session.wait(5.0):
            if session.ready and not session.url_shown:
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
        if session.ready and not session.url_shown:
            session.url_shown = True
            if session.opened_in == BROWSER_TAB:
                print("Sign in and click Allow in the Browser tab that just opened.")
            else:
                print(f"Open this URL, sign in, and click Allow:\n{session.auth_url}")
    answer = _session_answer(session)
    print(answer.get("message") or answer.get("error"))
    return 0 if session.done else 1


if __name__ == "__main__":
    sys.exit(main())
