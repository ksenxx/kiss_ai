# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the interactive MCP OAuth sign-in.

Covers ``kiss.agents.sorcar.mcp_oauth`` and ``build_oauth_provider`` in
``kiss.agents.sorcar.mcp_servers`` against a real OAuth-protected MCP
server: the MCP SDK's own ``FastMCP`` with ``AuthSettings`` and an
in-memory ``OAuthAuthorizationServerProvider`` (dynamic client
registration, PKCE, refresh), served by uvicorn on a free local port.
The "browser" is httpx following the authorization redirect to the
loopback callback ``http://localhost:<port>/callback``.

The product's redirect port is fixed (a dynamically registered client
keeps working only if its redirect URI never changes); every test here
registers a fresh client, so ``_own_redirect_port`` moves the listener
to a free ephemeral port and concurrently running pytest processes
never collide on it.

Branches not exercised:

* ``_refuse_login`` as the callback handler in ``mcp_servers``: the SDK
  calls the redirect handler first, and that always raises, so the
  callback handler can never be reached.
* ``_fail_calls`` in ``mcp_servers`` passing over a call its caller has
  already cancelled: the cancelled call's task settles within a few
  turns of the manager loop, and the connection would have to end
  inside them.
* ``MCPLoginSession._run``'s no-exception ``error = "sign-in cancelled"``
  branch: it needs a cancel to land after the redirect was consumed but
  before ``initialize`` returns, a window no real client/server exchange
  can hit deterministically (the exception path of a cancel is covered
  by ``test_new_session_replaces_running_one``).
* ``if self._server is not None`` being false in ``_run``: ``start``
  always binds the server before starting the thread.
* The ``__main__`` guard (``sys.exit(main())``): ``main`` itself is
  tested in-process.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import json
import secrets
import socket
import socketserver
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import FrameType
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

import anyio
import httpx
import pytest
import uvicorn
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.server.fastmcp import FastMCP
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from starlette.responses import JSONResponse

from kiss.agents.sorcar import mcp_oauth
from kiss.agents.sorcar.mcp_oauth import (
    KNOWN_MCP_SERVERS,
    MCPLoginSession,
    _session_answer,
    build_provider,
    drop_stale_registration,
    make_mcp_auth_tools,
    resolve_server,
    seed_preregistered_client,
)
from kiss.agents.sorcar.mcp_servers import (
    _LOGIN_HINT,
    FileTokenStorage,
    MCPManager,
    MCPServerConfig,
    _Connection,
    _connection_key,
    build_oauth_provider,
    load_mcp_servers,
    save_mcp_server,
    user_mcp_config_path,
)
from kiss.core.file_lock import exclusive_file_lock
from kiss.tests.conftest import occupy_loopback_port, wait_for

# ---------------------------------------------------------------------------
# A real OAuth-protected MCP server
# ---------------------------------------------------------------------------


class _MemoryAuthProvider:
    """In-memory authorization server that approves (or refuses) at once.

    ``authorize`` plays the user's consent screen: it redirects straight
    back to the client's ``redirect_uri`` with a code, or with
    ``error=access_denied`` when ``refuse`` is set.
    """

    def __init__(self) -> None:
        self.refuse = False
        self.rotate = True  # a refresh issues a new refresh token and voids the old one
        self.expires_in = 3600
        self.clients: dict[str, OAuthClientInformationFull] = {}
        self.registrations = 0
        self.codes: dict[str, AuthorizationCode] = {}
        self.access: dict[str, AccessToken] = {}
        self.refresh: dict[str, RefreshToken] = {}

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self.clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self.registrations += 1
        assert client_info.client_id is not None
        self.clients[client_info.client_id] = client_info

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        if self.refuse:
            return construct_redirect_uri(
                str(params.redirect_uri), error="access_denied", state=params.state
            )
        code = secrets.token_urlsafe(16)
        assert client.client_id is not None
        self.codes[code] = AuthorizationCode(
            code=code,
            scopes=params.scopes or [],
            expires_at=time.time() + 300,
            client_id=client.client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
        )
        return construct_redirect_uri(str(params.redirect_uri), code=code, state=params.state)

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        return self.codes.get(authorization_code)

    def _issue(self, client_id: str, scopes: list[str]) -> OAuthToken:
        access, refresh = secrets.token_urlsafe(16), secrets.token_urlsafe(16)
        self.access[access] = AccessToken(
            token=access,
            client_id=client_id,
            scopes=scopes,
            expires_at=int(time.time() + self.expires_in),
        )
        self.refresh[refresh] = RefreshToken(token=refresh, client_id=client_id, scopes=scopes)
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=self.expires_in,
            refresh_token=refresh,
        )

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        del self.codes[authorization_code.code]
        return self._issue(authorization_code.client_id, authorization_code.scopes)

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        return self.refresh.get(refresh_token)

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str]
    ) -> OAuthToken:
        token = self._issue(refresh_token.client_id, scopes or refresh_token.scopes)
        if self.rotate:
            del self.refresh[refresh_token.token]
            return token
        # Not rotated (RFC 6749 section 6 leaves it to the server): the one
        # in use stays good and the response carries none.
        assert token.refresh_token is not None
        del self.refresh[token.refresh_token]
        return token.model_copy(update={"refresh_token": None})

    async def load_access_token(self, token: str) -> AccessToken | None:
        return self.access.get(token)

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        self.access.pop(token.token, None)
        self.refresh.pop(token.token, None)


class _MovedTokenEndpoint:
    """An OAuth MCP app whose token endpoint is ``/custom-token``, as its metadata says.

    ``/token``, where a client without the metadata sends its refresh,
    is a 404.  While ``failing`` is set an authorized MCP request (over
    SSE, a message POST) is answered 503: a sign-in is granted its
    tokens and the handshake that follows fails.
    """

    def __init__(self, app: Any, base: str) -> None:
        self.app, self.base = app, base
        self.failing = True
        self.paths: list[str] = []

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        path = scope.get("path", "")
        self.paths.append(path)
        if path == "/token":
            await JSONResponse({"error": "not found"}, 404)(scope, receive, send)
        elif (
            path in ("/mcp", "/messages/") and self.failing
            and b"authorization" in dict(scope["headers"])
        ):
            await JSONResponse({"error": "unavailable"}, 503)(scope, receive, send)
        elif path == "/custom-token":
            await self.app({**scope, "path": "/token", "raw_path": b"/token"}, receive, send)
        elif path == "/.well-known/oauth-authorization-server":
            body = bytearray()

            async def collect(message: Any) -> None:
                body.extend(message.get("body", b""))

            await self.app(scope, receive, collect)
            moved = {**json.loads(bytes(body)), "token_endpoint": f"{self.base}/custom-token"}
            await JSONResponse(moved)(scope, receive, send)
        else:
            await self.app(scope, receive, send)


class _AuthMCPServer:
    """A FastMCP server with OAuth, running under uvicorn on a thread.

    *wrap*, when given, is the ASGI app class its app is served through;
    *transport* is ``"http"`` (streamable HTTP) or ``"sse"``.
    """

    def __init__(self, wrap: Any = None, transport: str = "http") -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        self.transport = transport
        self.base = f"http://127.0.0.1:{port}"
        self.url = f"{self.base}/{'mcp' if transport == 'http' else 'sse'}"
        self.provider = _MemoryAuthProvider()
        mcp = FastMCP(
            "authdemo",
            host="127.0.0.1",
            port=port,
            auth_server_provider=self.provider,
            auth=AuthSettings(
                issuer_url=self.base,  # type: ignore[arg-type]
                resource_server_url=self.url,  # type: ignore[arg-type]
                client_registration_options=ClientRegistrationOptions(enabled=True),
            ),
        )
        mcp.add_tool(_echo)
        app = mcp.streamable_http_app() if transport == "http" else mcp.sse_app()
        self.app: Any = wrap(app, self.base) if wrap else app
        config = uvicorn.Config(self.app, host="127.0.0.1", port=port, log_level="error")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + 20
        while not self._server.started:
            assert time.monotonic() < deadline, "uvicorn did not start"
            time.sleep(0.02)

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


def _echo(text: str) -> str:
    """Echo the given text back."""
    return f"echo: {text}"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test KISS home (user mcp.json and mcp_auth token files)."""
    kiss_home = tmp_path / "kiss_home"
    kiss_home.mkdir()
    monkeypatch.setenv("KISS_HOME", str(kiss_home))
    monkeypatch.setenv("KISS_HEADLESS", "1")
    monkeypatch.delenv("KISS_MCP_CLIENT_METADATA_URL", raising=False)
    return kiss_home


@pytest.fixture
def work_dir(tmp_path: Path) -> str:
    path = tmp_path / "project"
    path.mkdir()
    return str(path)


@pytest.fixture
def server(home: Path) -> Iterator[_AuthMCPServer]:
    srv = _AuthMCPServer()
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


@pytest.fixture(autouse=True)
def _own_redirect_port(mcp_redirect_port: int) -> Iterator[int]:
    """Run every test on its own ephemeral redirect port.

    Cancels any login session the test left running, so no listener
    outlives the test on that port.
    """
    yield mcp_redirect_port
    session = MCPLoginSession._active
    if session is not None:
        session.cancel()
        session.wait(10)
    MCPLoginSession._active = None


def _cfg(server: _AuthMCPServer, name: str = "authdemo") -> MCPServerConfig:
    return MCPServerConfig(name=name, transport=server.transport, url=server.url)


def _twins(server: _AuthMCPServer) -> tuple[MCPServerConfig, MCPServerConfig]:
    """Two configurations of one server: separate connections sharing one stored grant."""
    cfg = _cfg(server)
    return cfg, dataclasses.replace(cfg, headers=(("X-Project", "b"),))


def _approve(auth_url: str) -> httpx.Response:
    """Play the user's browser: open the URL and follow every redirect."""
    return httpx.get(auth_url, follow_redirects=True, trust_env=False, timeout=20)


def _tokens(name: str, url: str = "") -> Any:
    """The tokens stored for *name* at *url* (credentials are kept per name and URL)."""
    return asyncio.run(FileTokenStorage(name, url).get_tokens())


def _client_info(name: str, url: str = "") -> Any:
    return asyncio.run(FileTokenStorage(name, url).get_client_info())


def _login(server: _AuthMCPServer, name: str = "authdemo") -> MCPLoginSession:
    session = MCPLoginSession.start(_cfg(server, name))
    assert session.auth_url, session.error
    _approve(session.auth_url)
    assert session.wait(30)
    assert session.done, session.error
    return session


# ---------------------------------------------------------------------------
# MCPLoginSession against the real server
# ---------------------------------------------------------------------------


def test_login_stores_tokens_and_manager_connects(server: _AuthMCPServer, home: Path) -> None:
    cfg = _cfg(server)
    session = MCPLoginSession.start(cfg)
    assert session.error == ""
    assert session.done is False
    query = {k: v[0] for k, v in parse_qs(urlsplit(session.auth_url).query).items()}
    assert session.auth_url.startswith(f"{server.base}/authorize?")
    assert query["redirect_uri"] == mcp_oauth.MCP_REDIRECT_URI
    assert query["code_challenge_method"] == "S256"
    assert query["client_id"] in server.provider.clients
    assert MCPLoginSession.active("authdemo") is session
    assert MCPLoginSession.active("other") is None

    answer = _session_answer(session)
    assert answer["status"] == "consent_required"
    assert answer["verification_uri"] == session.auth_url
    assert answer["opened_in"] == "" and session.opened_in == ""  # KISS_HEADLESS=1
    assert "OWN browser" in answer["instructions"]
    assert "finish_mcp_server_connect('authdemo')" in answer["instructions"]

    wrong = httpx.get(f"http://127.0.0.1:{mcp_oauth.MCP_REDIRECT_PORT}/elsewhere", trust_env=False)
    assert wrong.status_code == 404

    page = _approve(session.auth_url)
    assert page.status_code == 200
    assert "Sign-in received" in page.text
    assert session.wait(30)
    assert (session.done, session.error) == (True, "")
    assert _session_answer(session)["ok"] is True

    # The registration used KISS's public-client metadata.
    registered = server.provider.clients[query["client_id"]]
    assert registered.token_endpoint_auth_method == "none"
    assert [str(u) for u in registered.redirect_uris or []] == [mcp_oauth.MCP_REDIRECT_URI]
    assert registered.client_name == mcp_oauth.PRODUCT_NAME

    tokens = _tokens("authdemo", server.url)
    assert tokens is not None
    assert tokens.access_token in server.provider.access
    assert _client_info("authdemo", server.url).client_id == query["client_id"]

    # An agent run now connects without any browser step.
    manager = MCPManager()
    try:
        conn = manager.connect(cfg)
        assert conn.error == ""
        assert [t.name for t in conn.tools] == ["_echo"]
        assert "echo: hi" in manager.call_tool("authdemo", "_echo", {"text": "hi"})
    finally:
        manager.shutdown()

    # Signing in again with working tokens finishes without a redirect.
    again = MCPLoginSession.start(cfg)
    assert again.wait(30)
    assert (again.done, again.auth_url, again.error) == (True, "", "")
    assert server.provider.registrations == 1


def test_manager_without_tokens_refuses_interactive_login(
    server: _AuthMCPServer, home: Path
) -> None:
    cfg = _cfg(server)
    auth = build_oauth_provider(_Connection(config=cfg))
    assert auth.context.client_metadata.token_endpoint_auth_method == "none"
    manager = MCPManager()
    try:
        conn = manager.connect(cfg)
        assert _LOGIN_HINT in conn.error
    finally:
        manager.shutdown()
    assert server.provider.access == {}
    # No stored grant, nothing to refresh: no refresh lock file was made.
    assert not list((home / "mcp_auth").glob("*.refresh.lock"))


async def _second_refresher_gives_up(storage: FileTokenStorage) -> None:
    async with storage.refreshing():
        with pytest.raises(TimeoutError, match=r"\.refresh\.lock still locked after 0\.2s"):
            async with FileTokenStorage("authdemo", storage.server_url).refreshing(timeout=0.2):
                raise AssertionError("two holders of one grant's refresh lock")


def test_a_wedged_refresh_costs_the_next_provider_a_bounded_wait(home: Path) -> None:
    """A stored grant's refresh lock admits one holder; a waiter gives up at its deadline."""
    storage = FileTokenStorage("authdemo", "https://mcp.example/mcp")
    storage.put("tokens", {"access_token": "a", "token_type": "Bearer"})
    asyncio.run(_second_refresher_gives_up(storage))


def test_manager_refreshes_expired_tokens(server: _AuthMCPServer, home: Path) -> None:
    _login(server)
    storage = FileTokenStorage("authdemo", server.url)
    assert storage.get_oauth_metadata() is not None
    tokens = storage._read()["tokens"]
    old = tokens["access_token"]
    # The access token expires, on the server and in the stored record;
    # the refresh token stays valid.  Expired after the login: a 1 s
    # lifetime from the start let a slow (Windows, loaded) login's own
    # check see the token already expired.
    server.provider.access[old] = server.provider.access[old].model_copy(
        update={"expires_at": int(time.time()) - 1}
    )
    storage.put("tokens", {**tokens, "expires_in": 0})
    manager = MCPManager()
    try:
        # The server takes a refresh token once: the two connections
        # must not each present the one they found in the store.
        assert [conn.error for conn in manager.connect_all(_twins(server))] == ["", ""]
    finally:
        manager.shutdown()
    new = storage._read()["tokens"]["access_token"]
    assert new != old
    assert new in server.provider.access


def test_a_refresh_answered_without_a_refresh_token_keeps_the_stored_one(
    server: _AuthMCPServer, home: Path,
) -> None:
    """A grant survives an authorization server that does not rotate its refresh tokens.

    Such a server leaves ``refresh_token`` out of its refresh response:
    the one in use stays good (RFC 6749 section 6).  The SDK stored the
    response whole, so the grant lost its refresh token at the first
    refresh and the next expiry demanded a new sign-in.
    """
    _login(server)
    server.provider.rotate = False
    storage = FileTokenStorage("authdemo", server.url)
    refresh = storage._read()["tokens"]["refresh_token"]
    for _ in range(2):  # the second expiry is the one that asked for the sign-in
        tokens = storage._read()["tokens"]
        old = tokens["access_token"]
        server.provider.access[old] = server.provider.access[old].model_copy(
            update={"expires_at": int(time.time()) - 1}
        )
        storage.put("tokens", {**tokens, "expires_in": 0})
        manager = MCPManager()
        try:
            assert manager.connect(_cfg(server)).error == ""
        finally:
            manager.shutdown()
        assert storage._read()["tokens"]["access_token"] != old
        assert storage._read()["tokens"].get("refresh_token") == refresh


def _stale_grant(server: _AuthMCPServer) -> FileTokenStorage:
    """Sign in and mark the stored access token as due for a refresh."""
    _login(server)
    storage = FileTokenStorage("authdemo", server.url)
    storage.put("tokens", {**storage._read()["tokens"], "expires_in": 0})
    return storage


def _flow(server: _AuthMCPServer) -> Any:
    """A provider's flow for one request to *server*: no sign-in is expected."""
    cfg = _cfg(server)
    provider = build_provider(cfg, _no_redirect, _no_callback)
    return provider.async_auth_flow(httpx.Request("POST", cfg.url))


async def _refresh(server: _AuthMCPServer, meanwhile: Any = None) -> None:
    """Refresh the stored grant, calling *meanwhile* while the refresh is at the server.

    One task drives the flow from its first request to its close, as httpx does.
    """
    flow = _flow(server)
    try:
        refresh = await flow.__anext__()
        assert b"grant_type=refresh_token" in refresh.content
        if meanwhile is not None:
            meanwhile()
        async with httpx.AsyncClient(trust_env=False) as client:
            await flow.asend(await client.send(refresh))
    finally:
        await flow.aclose()


async def _first_request(server: _AuthMCPServer) -> httpx.Request:
    """The first request a provider sends for a request to *server*."""
    flow = _flow(server)
    try:
        return cast(httpx.Request, await flow.__anext__())
    finally:
        await flow.aclose()


def test_a_refresh_replaces_only_the_grant_it_refreshed(
    server: _AuthMCPServer, home: Path,
) -> None:
    """A refresh answered after its grant was removed or replaced stores nothing.

    A new sign-in drops a registration with another redirect URI, grant
    and all, and stores the grant it ends with; neither waits for a
    refresh a connection has at the server.  That refresh's answer
    brought the dropped grant back (its tokens, with no client to
    refresh them as) and replaced the new sign-in's tokens.
    """
    storage = _stale_grant(server)
    asyncio.run(_refresh(server, storage.clear))
    assert not storage.path.exists()
    storage = _stale_grant(server)
    signed_in = {"access_token": "a", "token_type": "Bearer", "refresh_token": "r"}
    asyncio.run(_refresh(server, functools.partial(storage.put, "tokens", signed_in)))
    assert storage._read()["tokens"] == signed_in


def _a_token_write_waits() -> bool:
    """Whether a thread is inside ``FileTokenStorage.put`` (here: waiting for the store's lock)."""
    frames: list[FrameType | None] = list(sys._current_frames().values())
    while frames:
        frame = frames.pop()
        if frame is not None:
            if frame.f_code is FileTokenStorage.put.__code__:
                return True
            frames.append(frame.f_back)
    return False


async def _refresh_in(scope: anyio.CancelScope, server: _AuthMCPServer) -> None:
    with scope:
        await _refresh(server)


async def _cancel_a_refresh_at_its_token_write(
    server: _AuthMCPServer, storage: FileTokenStorage, scoped: bool,
) -> None:
    old = storage._read()["tokens"]
    scope = anyio.CancelScope()
    with exclusive_file_lock(storage._lock_path):  # as another process writing the store
        refreshing = asyncio.ensure_future(_refresh_in(scope, server))
        async with asyncio.timeout(10):
            while not _a_token_write_waits():
                await asyncio.sleep(0.01)
        if scoped:
            scope.cancel()
        else:
            refreshing.cancel()
        next_provider = asyncio.ensure_future(_first_request(server))
        done, _ = await asyncio.wait([next_provider], timeout=0.5)
        assert not done, "a second refresh was sent with the refresh token the first had used"
    await asyncio.wait([refreshing])
    assert refreshing.cancelled() is not scoped  # a scope ends its own cancellation
    request = await next_provider
    new = storage._read()["tokens"]
    assert new["refresh_token"] != old["refresh_token"]
    assert request.headers["Authorization"] == f"Bearer {new['access_token']}"


@pytest.mark.parametrize("scoped", [False, True], ids=["by-asyncio", "by-its-anyio-scope"])
def test_a_refresh_cancelled_at_its_token_write_keeps_the_grant_to_itself(
    server: _AuthMCPServer, home: Path, scoped: bool,
) -> None:
    """The refresh lock covers the write of the refreshed grant, cancelled or not.

    The write runs on a worker thread, which a cancellation (the
    connection ending: its task cancelled, or the scope of its
    transport's tasks) leaves running: the cancelled provider let go of
    the refresh lock with the store still holding the old grant, and
    the next provider presented the refresh token the server had just
    taken, which a server that rotates them answers by ending the grant.
    """
    asyncio.run(_cancel_a_refresh_at_its_token_write(server, _stale_grant(server), scoped))


def test_manager_refreshes_a_token_that_expires_while_connected(
    server: _AuthMCPServer, home: Path,
) -> None:
    """A token valid at connect time is refreshed once its stored expiry passes.

    The SDK loads stored tokens without a deadline and treats a loaded
    token as valid until a request fails, so a long-lived connection
    kept sending an access token past its stored ``expires_at``; the
    server's 401 then took the interactive sign-in path, which an agent
    run refuses, although a refresh token was at hand.  Every request
    now starts from the stored deadline (less the refresh margin).
    """
    _login(server)
    storage = FileTokenStorage("authdemo", server.url)
    tokens = storage._read()["tokens"]
    old = tokens["access_token"]
    # Expires in 3 s on the server; the stored record (an integer
    # ``expires_in``, as the token model demands) puts it 2 s past the
    # 60 s refresh margin, so it is valid when the connection starts and
    # due for a refresh before the tool call below.
    server.provider.access[old] = server.provider.access[old].model_copy(
        update={"expires_at": int(time.time()) + 3}
    )
    storage.put("tokens", {**tokens, "expires_in": 62})
    manager = MCPManager()
    try:
        assert [conn.error for conn in manager.connect_all(_twins(server))] == ["", ""]
        time.sleep(2.5)
        # Both hold the old grant in memory: the second to refresh must
        # find the first one's in the store, not present the used token.
        for cfg in _twins(server):
            result = manager.call_tool(_connection_key(cfg), "_echo", {"text": "hi"})
            assert result == "echo: hi", result
    finally:
        manager.shutdown()
    new = storage._read()["tokens"]["access_token"]
    assert new != old
    assert new in server.provider.access


def _end_access_token(server: _AuthMCPServer, storage: FileTokenStorage) -> str:
    """End the stored access token on the server, ahead of what the store knows; return it."""
    token = str(storage._read()["tokens"]["access_token"])
    server.provider.access[token] = server.provider.access[token].model_copy(
        update={"expires_at": int(time.time()) - 1}
    )
    return token


def test_a_401_for_a_token_the_store_holds_as_good_is_answered_by_a_refresh(
    server: _AuthMCPServer, home: Path,
) -> None:
    """A grant issued without ``expires_in`` is refreshed when the server stops taking it.

    ``expires_in`` is optional (RFC 6749 section 5.1); without it the
    server's 401 is the first sign that the access token ended.  That
    401 went straight to the interactive sign-in, which an agent run
    refuses, although the refresh token was good.  Met while connecting
    and in a tool call, by both connections of the grant: the server
    takes a refresh token once.
    """
    _login(server)
    storage = FileTokenStorage("authdemo", server.url)
    tokens = storage._read()["tokens"]
    storage.put("tokens", {k: v for k, v in tokens.items() if k != "expires_in"})
    assert storage.refresh_deadline() is None
    first = _end_access_token(server, storage)
    manager = MCPManager()
    try:
        assert [conn.error for conn in manager.connect_all(_twins(server))] == ["", ""]
        second = _end_access_token(server, storage)
        assert second != first
        for cfg in _twins(server):
            result = manager.call_tool(_connection_key(cfg), "_echo", {"text": "hi"})
            assert result == "echo: hi", result
    finally:
        manager.shutdown()
    third = storage._read()["tokens"]["access_token"]
    assert third != second and third in server.provider.access


@pytest.mark.parametrize("transport", ["http", "sse"])
def test_a_call_fails_at_once_when_its_connection_ends_on_a_refused_sign_in(
    home: Path, transport: str,
) -> None:
    """A tool call in flight when its connection ends gets that connection's error.

    The server stops taking the grant, refresh token and all: the
    call's 401 needs a new sign-in, which a run refuses, and the
    connection is over.  Over streamable HTTP the refusal ends the
    transport, and the call, a task of its own, waited out the SDK's
    300 s read timeout for an answer no one was left to send.  The
    SDK's SSE client logged the refusal and carried on without its
    writer: the same wait, on a connection that looked alive.  The
    call is not sent again: the next one starts over.
    """
    server = _AuthMCPServer(transport=transport)
    server.start()
    manager = MCPManager()
    try:
        _login(server)
        cfg = _cfg(server)
        assert manager.connect(cfg).error == ""
        server.provider.access.clear()
        server.provider.refresh.clear()
        started = time.monotonic()
        result = manager.call_tool(_connection_key(cfg), "_echo", {"text": "hi"})
        assert time.monotonic() - started < 60, "the call waited for its read timeout"
        assert result.startswith("Error: MCP tool call failed") and _LOGIN_HINT in result, result
        assert _LOGIN_HINT in manager.call_tool(_connection_key(cfg), "_echo", {"text": "hi"})
    finally:
        manager.shutdown()
        server.stop()


@pytest.mark.parametrize("transport", ["http", "sse"])
def test_a_grant_keeps_its_token_endpoint_when_the_sign_in_fails_after_it(
    home: Path, transport: str,
) -> None:
    """A grant issued by a sign-in whose handshake then fails is stored with its metadata.

    The authorization server's metadata names the token endpoint
    (``/custom-token`` here; ``/token``, the client's guess without the
    metadata, is a 404).  It was stored only once ``initialize`` had
    succeeded: a sign-in granted its tokens and then answered 503 left a
    grant whose first refresh went to ``/token`` and ended in a demand
    for a new sign-in.  Over SSE the 503 answers a POST, which the
    SDK's writer logged and dropped: the sign-in ended not with it but
    at its own 660 s deadline.
    """
    srv = _AuthMCPServer(_MovedTokenEndpoint, transport)
    srv.start()
    manager = MCPManager()
    try:
        cfg = _cfg(srv)
        session = MCPLoginSession.start(cfg)
        _approve(session.auth_url)
        assert session.wait(30)
        assert not session.done and "503" in session.error, session.error
        storage = FileTokenStorage("authdemo", srv.url)
        tokens = storage._read()["tokens"]
        assert tokens["refresh_token"] in srv.provider.refresh
        metadata = storage.get_oauth_metadata()
        assert metadata is not None, "the grant was stored without its token endpoint"
        assert str(metadata.token_endpoint) == f"{srv.base}/custom-token"

        # The server recovers; the access token has expired by then.
        srv.app.failing = False
        _end_access_token(srv, storage)
        storage.put("tokens", {**tokens, "expires_in": 0})
        assert manager.connect(cfg).error == ""
        assert "/token" not in srv.app.paths
        assert storage._read()["tokens"]["access_token"] != tokens["access_token"]
    finally:
        manager.shutdown()
        srv.stop()


def test_login_refused_by_user(server: _AuthMCPServer, home: Path) -> None:
    server.provider.refuse = True
    session = MCPLoginSession.start(_cfg(server))
    _approve(session.auth_url)
    assert session.wait(30)
    assert session.done is False
    assert "sign-in refused (access_denied)" in session.error
    answer = _session_answer(session)
    assert answer["ok"] is False
    assert answer["error"].startswith("MCP sign-in to 'authdemo' failed:")
    assert _tokens("authdemo", server.url) is None


def test_new_session_replaces_running_one(server: _AuthMCPServer, home: Path) -> None:
    first = MCPLoginSession.start(_cfg(server))
    assert first.auth_url
    second = MCPLoginSession.start(_cfg(server))
    assert first.wait(0)
    assert first.done is False
    assert first.error == "sign-in cancelled"
    assert MCPLoginSession.active("authdemo") is second
    assert second.auth_url and second.auth_url != first.auth_url
    # The first sign-in's browser tab redirects to the same fixed port,
    # once approved and once refused: neither may end the second sign-in.
    for refuse in (False, True, False):
        server.provider.refuse = refuse
        stale = _approve(first.auth_url)
        assert stale.status_code == 400 and "Sign-in received" not in stale.text
    _approve(second.auth_url)
    assert second.wait(30)
    assert second.done, second.error


def test_half_open_redirect_connection_does_not_pin_the_port(
    server: _AuthMCPServer, home: Path
) -> None:
    session = MCPLoginSession.start(_cfg(server))
    assert session.auth_url
    # A browser that opens the connection but never finishes its request:
    # the handler's socket timeout must return the serving thread to the
    # cancel check instead of blocking on the missing headers forever.
    half_open = socket.create_connection(("127.0.0.1", mcp_oauth.MCP_REDIRECT_PORT), timeout=10)
    try:
        half_open.sendall(b"GET /callback HTTP/1.1\r\n")
        time.sleep(0.5)  # the redirect server has accepted the connection by now
        session.cancel()
        assert session.wait(10)
        assert session.done is False
        assert session.error == "sign-in cancelled"
        assert half_open.recv(1) == b""  # the server closed the half-open connection

        replacement = MCPLoginSession.start(_cfg(server))
        assert replacement.auth_url, replacement.error
        _approve(replacement.auth_url)
        assert replacement.wait(30)
        assert replacement.done, replacement.error
    finally:
        half_open.close()


def _callback_connection_accepted() -> bool:
    """Whether the redirect server is inside its request handler (a connection accepted)."""
    handling = BaseHTTPRequestHandler.handle_one_request.__code__
    for top in sys._current_frames().values():
        frame: FrameType | None = top
        while frame is not None:
            if frame.f_code is handling:
                return True
            frame = frame.f_back
    return False


#: What ``recv`` raises, besides returning ``b""``, once the server ended a
#: connection the feeder was still writing to: a reset, which Windows reports
#: as the connection aborted (``WSAECONNABORTED``) when the send met it.
_ENDED_BY_THE_SERVER = (ConnectionResetError, ConnectionAbortedError)


def test_cancel_ends_a_callback_connection_that_keeps_the_handler_busy(
    server: _AuthMCPServer, home: Path
) -> None:
    """A browser connection that keeps sending header bytes never trips the
    handler's inactivity timeout; a cancel must shut that connection down
    too, else the flow waits for it for good and the session never ends
    (audit A09)."""
    session = MCPLoginSession.start(_cfg(server))
    assert session.auth_url
    slow = socket.create_connection(("127.0.0.1", mcp_oauth.MCP_REDIRECT_PORT), timeout=10)
    stop = threading.Event()

    def keep_the_header_coming() -> None:
        # One header byte a second: well within the handler's 5 s timeout.
        with contextlib.suppress(OSError):
            while not stop.wait(1.0):
                slow.sendall(b"X")

    feeder = threading.Thread(target=keep_the_header_coming, daemon=True)
    try:
        slow.sendall(b"GET /callback HTTP/1.1\r\nX-Slow: ")
        assert wait_for(_callback_connection_accepted, timeout=10)
        feeder.start()
        session.cancel()
        assert session.wait(10), "the accepted connection kept the cancelled flow alive"
        assert session.done is False
        assert session.error == "sign-in cancelled"
        stop.set()
        feeder.join(5)
        with contextlib.suppress(*_ENDED_BY_THE_SERVER):
            assert slow.recv(1) == b""  # the server ended the connection
        replacement = MCPLoginSession.start(_cfg(server))
        assert replacement.auth_url, replacement.error
    finally:
        stop.set()
        slow.close()


def test_the_deadline_ends_a_callback_connection_that_keeps_the_handler_busy(
    server: _AuthMCPServer, home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The login deadline, not just a cancel, must end a busy callback connection.

    The deadline cancels the flow's task, but the ``to_thread`` worker
    serving the trickling connection cannot be cancelled, and the server
    was only closed by ``_run`` after ``asyncio.run`` returned: which it
    does only once its executor shutdown gave up on the worker (five
    minutes), so the session and its port outlived the deadline by that
    much (audit 2026-11-08, S5 F3).  The flow closes the server itself
    now, so the worker is freed before the executor is shut down.
    """
    monkeypatch.setattr(mcp_oauth, "_LOGIN_TIMEOUT", 5.0)
    monkeypatch.setattr(mcp_oauth, "CONNECT_TIMEOUT", 1.0)
    session = MCPLoginSession.start(_cfg(server))
    assert session.auth_url
    slow = socket.create_connection(("127.0.0.1", mcp_oauth.MCP_REDIRECT_PORT), timeout=10)
    stop = threading.Event()

    def keep_the_header_coming() -> None:
        with contextlib.suppress(OSError):
            while not stop.wait(0.5):
                slow.sendall(b"X")

    feeder = threading.Thread(target=keep_the_header_coming, daemon=True)
    try:
        slow.sendall(b"GET /callback HTTP/1.1\r\nX-Slow: ")
        assert wait_for(_callback_connection_accepted, timeout=10)
        assert not session._finished.is_set(), "the flow ended before the connection was served"
        feeder.start()
        assert session.wait(6.0 + 5.0), "the busy connection kept the timed-out flow alive"
        assert session.done is False
        assert "TimeoutError" in session.error, session.error
        stop.set()
        feeder.join(5)
        with contextlib.suppress(*_ENDED_BY_THE_SERVER):
            assert slow.recv(1) == b""  # the server ended the connection
        replacement = MCPLoginSession.start(_cfg(server))  # the port is free again
        assert replacement.auth_url, replacement.error
    finally:
        stop.set()
        slow.close()


def test_a_stop_between_publishing_the_session_and_starting_it_leaves_nothing(
    server: _AuthMCPServer, home: Path
) -> None:
    """A Stop (``KeyboardInterrupt`` on the task thread) that lands after the
    listener is bound and the session published, before its worker starts,
    must unpublish the session and free the port: nothing would ever finish
    that session and no later sign-in could bind the port (audit A09)."""
    start = threading.Thread.start.__code__

    def interrupt_the_start(frame: Any, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code is start:
            sys.setprofile(None)
            raise KeyboardInterrupt

    sys.setprofile(interrupt_the_start)
    try:
        with pytest.raises(KeyboardInterrupt):
            MCPLoginSession.start(_cfg(server))
    finally:
        sys.setprofile(None)
    assert MCPLoginSession.active("authdemo") is None
    replacement = MCPLoginSession.start(_cfg(server))  # the port is free again
    assert replacement.auth_url, replacement.error
    _approve(replacement.auth_url)
    assert replacement.wait(30)
    assert replacement.done, replacement.error


def test_login_opens_the_browser_tab_and_hands_off_without_a_url(
    server: _AuthMCPServer, home: Path
) -> None:
    """Under the daemon the consent page opens in the Browser tab before the URL is announced."""
    from kiss.core.browser_handoff import set_browser_tab_opener

    seen: list[str] = []

    def open_tab(url: str) -> bool:
        seen.append(url)
        return True

    set_browser_tab_opener(open_tab)
    try:
        session = MCPLoginSession.start(_cfg(server))
        assert session.error == "" and session.auth_url
        # The page was opened first, so the very first answer is worded right.
        assert seen == [session.auth_url]
        assert session.opened_in == "browser_tab"
        answer = _session_answer(session)
        assert answer["status"] == "consent_required"
        assert answer["opened_in"] == "browser_tab"
        text = answer["instructions"]
        assert "already open in the Browser tab" in text
        assert "do NOT ask them to open a URL" in text and "OWN browser" not in text
        assert f"cannot see the page, give them {session.auth_url}" in text
        _approve(session.auth_url)
        assert session.wait(30)
        assert session.done, session.error
    finally:
        set_browser_tab_opener(None)


def test_browser_tab_that_follows_an_immediate_redirect_does_not_deadlock(
    server: _AuthMCPServer, home: Path
) -> None:
    """The Browser tab waits for the page to load; a provider that approves at once redirects
    straight to the loopback callback, which must be served while the tab is still opening."""
    from kiss.core.browser_handoff import set_browser_tab_opener

    def open_tab(url: str) -> bool:
        # Like the streamed browser: return only once navigation committed,
        # here after following every redirect down to the callback page.
        assert "Sign-in received" in _approve(url).text
        return True

    set_browser_tab_opener(open_tab)
    try:
        started = time.monotonic()
        session = MCPLoginSession.start(_cfg(server))
        assert session.wait(30)
        assert session.done, session.error
        assert time.monotonic() - started < 20
        assert session.opened_in == "browser_tab"
        assert _session_answer(session)["ok"] is True
    finally:
        set_browser_tab_opener(None)


def test_a_session_cancelled_before_its_page_opens_opens_nothing(
    server: _AuthMCPServer, home: Path,
) -> None:
    """A cancelled session's hand-off, still waiting to run, opens no page.

    The page is opened on a thread of its own, which a cancel (a
    replacement session taking over) can get ahead of: the dead
    sign-in's page then opened after the replacement's, and every
    surface switched back to it.  The opener's thread is parked as it
    enters its body (a profile hook: nothing is replaced) and the
    session cancelled meanwhile.
    """
    from kiss.core.browser_handoff import set_browser_tab_opener

    seen: list[str] = []
    entered, cancelled = threading.Event(), threading.Event()
    openers: list[threading.Thread] = []

    def open_tab(url: str) -> bool:
        seen.append(url)
        return True

    def park(frame: FrameType, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code is MCPLoginSession._open_for_user.__code__:
            openers.append(threading.current_thread())
            entered.set()
            cancelled.wait(30)

    set_browser_tab_opener(open_tab)
    threading.setprofile(park)  # for the threads the session starts
    try:
        session = MCPLoginSession.start(_cfg(server), wait_seconds=0)
        assert entered.wait(30), session.error
        threading.setprofile(None)
        session.cancel()
        cancelled.set()
        openers[0].join(10)
        assert not openers[0].is_alive()
        assert seen == []
        assert session.wait(10)
        assert _session_answer(session)["error"].endswith("sign-in cancelled")
    finally:
        threading.setprofile(None)
        cancelled.set()
        set_browser_tab_opener(None)


def test_answer_waits_for_the_hand_off_outcome(server: _AuthMCPServer, home: Path) -> None:
    """While the page is still opening the answer is pending, never a URL-based hand-off."""
    from kiss.core.browser_handoff import set_browser_tab_opener

    release = threading.Event()

    def slow_tab(url: str) -> bool:
        assert release.wait(10)
        return True

    set_browser_tab_opener(slow_tab)
    session = MCPLoginSession.start(_cfg(server), wait_seconds=1.0)
    try:
        deadline = time.monotonic() + 10
        while not session.auth_url and time.monotonic() < deadline:
            time.sleep(0.05)
        assert session.auth_url and not session.ready  # the page is still opening
        pending = _session_answer(session)
        assert pending["status"] == "pending" and session.url_shown is False
        release.set()
        deadline = time.monotonic() + 10
        while not session.ready and time.monotonic() < deadline:
            time.sleep(0.05)
        answer = _session_answer(session)
        assert answer["status"] == "consent_required" and answer["opened_in"] == "browser_tab"
        assert "do NOT ask them to open a URL" in answer["instructions"]
    finally:
        set_browser_tab_opener(None)
        session.cancel()


def test_flow_finishing_before_the_page_opened_never_publishes_a_url_hand_off(
    server: _AuthMCPServer, home: Path
) -> None:
    """The opener follows the redirect (approving) but returns late: the flow ends first,
    while the opener still lingers; the answer must then be success, never an 'open this URL'."""
    from kiss.core.browser_handoff import set_browser_tab_opener

    release = threading.Event()

    def approve_then_linger(url: str) -> bool:
        assert "Sign-in received" in _approve(url).text
        assert release.wait(20)
        return True

    set_browser_tab_opener(approve_then_linger)
    try:
        session = MCPLoginSession.start(_cfg(server), wait_seconds=1.0)
        deadline = time.monotonic() + 10
        while not session.auth_url and time.monotonic() < deadline:
            time.sleep(0.05)
        assert session.auth_url
        # The flow has (or is about to have) finished while the opener still
        # lingers: nothing may be answered as consent_required now.
        for _ in range(20):
            answer = _session_answer(session)
            assert answer.get("status") != "consent_required", answer
            if answer.get("ok"):
                break
            time.sleep(0.1)
        release.set()
        assert session.wait(30) and session.done, session.error
        assert _session_answer(session)["ok"] is True
    finally:
        release.set()
        set_browser_tab_opener(None)


def test_session_answer_before_url_is_pending(server: _AuthMCPServer) -> None:
    session = MCPLoginSession(_cfg(server))
    assert _session_answer(session) == {
        "ok": False,
        "status": "pending",
        "error": "still contacting the server or opening the sign-in page; retry",
    }


def test_unreachable_server_reports_error(home: Path) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    cfg = MCPServerConfig(name="dead", transport="http", url=f"http://127.0.0.1:{port}/mcp")
    session = MCPLoginSession.start(cfg)
    assert session.wait(30)
    assert session.done is False
    assert session.error
    assert "ConnectError" in session.error


class _SilentHandler(socketserver.BaseRequestHandler):
    """Accept the connection, read the request, never answer."""

    def handle(self) -> None:
        with contextlib.suppress(OSError):
            while self.request.recv(65536):
                pass


@pytest.fixture
def silent_server() -> Iterator[str]:
    """A server that accepts the MCP transport but never answers a request."""
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _SilentHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/mcp"
    finally:
        server.shutdown()
        server.server_close()


def test_cancel_ends_a_login_stuck_on_a_silent_server(home: Path, silent_server: str) -> None:
    """``cancel()`` must end the login flow, not just flag it.

    The flow sits in its first request to a server that never answers;
    ``cancel()`` only set a flag the flow checks at the redirect and in
    the callback loop, which it never reaches, so the login thread kept
    waiting on the server (the transport's read timeout, minutes) after
    the user cancelled or a replacement session took over.
    """
    cfg = MCPServerConfig(name="silent", transport="http", url=silent_server)
    session = MCPLoginSession.start(cfg, wait_seconds=1.0)
    assert not session.auth_url and not session._finished.is_set()
    session.cancel()
    assert session.wait(10), "the cancelled login kept waiting on the silent server"
    assert session.done is False
    assert session.error == "sign-in cancelled"


def test_login_stuck_on_a_silent_server_times_out(
    home: Path, silent_server: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flow the server never answers ends at the sign-in deadline with an error."""
    monkeypatch.setattr(mcp_oauth, "_LOGIN_TIMEOUT", 0.5)
    monkeypatch.setattr(mcp_oauth, "CONNECT_TIMEOUT", 0.5)
    cfg = MCPServerConfig(name="silent", transport="http", url=silent_server)
    session = MCPLoginSession.start(cfg, wait_seconds=10.0)
    assert session.wait(10), "the login outlived its deadline"
    assert session.done is False
    assert "TimeoutError" in session.error, session.error


# ---------------------------------------------------------------------------
# Pre-registered clients
# ---------------------------------------------------------------------------


def test_seed_preregistered_client(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage = FileTokenStorage("my-server.v2")
    monkeypatch.delenv("KISS_MCP_MY_SERVER_V2_CLIENT_ID", raising=False)
    assert seed_preregistered_client(storage, "my-server.v2") is False
    assert asyncio.run(storage.get_client_info()) is None

    monkeypatch.setenv("KISS_MCP_MY_SERVER_V2_CLIENT_ID", " public-id ")
    assert seed_preregistered_client(storage, "my-server.v2") is True
    info = asyncio.run(storage.get_client_info())
    assert (info.client_id, info.client_secret) == ("public-id", None)
    assert info.token_endpoint_auth_method == "none"
    assert [str(u) for u in info.redirect_uris] == [mcp_oauth.MCP_REDIRECT_URI]

    monkeypatch.setenv("KISS_MCP_MY_SERVER_V2_CLIENT_SECRET", " s3cret ")
    assert seed_preregistered_client(storage, "my-server.v2") is True
    info = asyncio.run(storage.get_client_info())
    assert (info.client_id, info.client_secret) == ("public-id", "s3cret")
    assert info.token_endpoint_auth_method == "client_secret_basic"


def test_login_with_preregistered_client_skips_registration(
    server: _AuthMCPServer, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.provider.clients["kiss-prereg"] = OAuthClientInformationFull.model_validate(
        {
            **mcp_oauth.client_metadata().model_dump(mode="json", exclude_none=True),
            "client_id": "kiss-prereg",
        }
    )
    monkeypatch.setenv("KISS_MCP_AUTHDEMO_CLIENT_ID", "kiss-prereg")
    session = _login(server)
    assert parse_qs(urlsplit(session.auth_url).query)["client_id"] == ["kiss-prereg"]
    assert server.provider.registrations == 0
    assert _client_info("authdemo", server.url).client_id == "kiss-prereg"


# ---------------------------------------------------------------------------
# Stale registrations (older KISS registered http://localhost:0/callback)
# ---------------------------------------------------------------------------


def _store_registration(name: str, client_id: str, redirect_uri: str, url: str = "") -> None:
    """Store a client registration for *name* at *url* plus a dead access token."""
    storage = FileTokenStorage(name, url)
    info = OAuthClientInformationFull.model_validate(
        {
            **mcp_oauth.client_metadata().model_dump(mode="json", exclude_none=True),
            "client_id": client_id,
            "redirect_uris": [redirect_uri],
        }
    )
    storage.put("client_info", info.model_dump(mode="json", exclude_none=True))
    storage.put("tokens", {"access_token": f"dead-{client_id}", "token_type": "Bearer"})


def test_drop_stale_registration_only_for_other_redirect_uris(home: Path) -> None:
    storage = FileTokenStorage("stale-check")
    assert drop_stale_registration(storage) is False

    _store_registration("stale-check", "current", mcp_oauth.MCP_REDIRECT_URI)
    assert drop_stale_registration(storage) is False
    assert _client_info("stale-check").client_id == "current"
    assert _tokens("stale-check").access_token == "dead-current"

    _store_registration("stale-check", "old", "http://localhost:0/callback")
    assert drop_stale_registration(storage) is True
    assert _client_info("stale-check") is None
    assert _tokens("stale-check") is None
    assert not storage.path.exists()
    # Nothing left to drop.
    assert drop_stale_registration(storage) is False


def test_login_reregisters_a_stale_client(server: _AuthMCPServer, home: Path) -> None:
    _store_registration("authdemo", "stale-client", "http://localhost:0/callback", server.url)
    session = _login(server)
    # The dead token forced a sign-in, and the stale registration was
    # replaced instead of being sent to the authorization endpoint.
    new_id = parse_qs(urlsplit(session.auth_url).query)["client_id"][0]
    assert new_id != "stale-client"
    assert new_id in server.provider.clients
    assert server.provider.registrations == 1
    info = _client_info("authdemo", server.url)
    assert info.client_id == new_id
    assert [str(u) for u in info.redirect_uris] == [mcp_oauth.MCP_REDIRECT_URI]
    assert _tokens("authdemo", server.url).access_token in server.provider.access


def test_preregistered_client_wins_over_stale_check(
    server: _AuthMCPServer, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.provider.clients["kiss-prereg"] = OAuthClientInformationFull.model_validate(
        {
            **mcp_oauth.client_metadata().model_dump(mode="json", exclude_none=True),
            "client_id": "kiss-prereg",
        }
    )
    monkeypatch.setenv("KISS_MCP_AUTHDEMO_CLIENT_ID", "kiss-prereg")
    _store_registration("authdemo", "stale-client", "http://localhost:0/callback", server.url)
    session = _login(server)
    # The environment's client replaced the stored one; the stale check
    # never ran, so nothing was registered dynamically.
    assert parse_qs(urlsplit(session.auth_url).query)["client_id"] == ["kiss-prereg"]
    assert server.provider.registrations == 0
    assert _client_info("authdemo", server.url).client_id == "kiss-prereg"
    assert _tokens("authdemo", server.url).access_token in server.provider.access


# ---------------------------------------------------------------------------
# build_provider
# ---------------------------------------------------------------------------


async def _no_redirect(_url: str) -> None:
    raise AssertionError("no sign-in expected")


async def _no_callback() -> tuple[str, str | None]:
    raise AssertionError("no sign-in expected")


def test_build_provider_seeds_stored_oauth_metadata(server: _AuthMCPServer, home: Path) -> None:
    cfg = _cfg(server)
    assert build_provider(cfg, _no_redirect, _no_callback).context.oauth_metadata is None
    _login(server)
    stored = FileTokenStorage("authdemo", server.url).get_oauth_metadata()
    assert stored is not None
    auth = build_provider(cfg, _no_redirect, _no_callback)
    assert auth.context.oauth_metadata is not None
    assert auth.context.oauth_metadata.token_endpoint == stored.token_endpoint
    assert str(auth.context.oauth_metadata.token_endpoint) == f"{server.base}/token"


# ---------------------------------------------------------------------------
# resolve_server
# ---------------------------------------------------------------------------


def test_resolve_known_preset_is_saved(home: Path, work_dir: str) -> None:
    cfg = resolve_server("notion", "", "", work_dir)
    assert (cfg.transport, cfg.url) == KNOWN_MCP_SERVERS["notion"]
    saved = json.loads(user_mcp_config_path().read_text())["mcpServers"]["notion"]
    assert saved["url"] == "https://mcp.notion.com/mcp"
    assert resolve_server("asana", "", "", work_dir).transport == "sse"
    # Now configured: returned as-is without re-saving.
    assert resolve_server("notion", "", "", work_dir) == cfg


def test_resolve_new_url_defaults_to_http(home: Path, work_dir: str) -> None:
    cfg = resolve_server("custom", "https://mcp.example.test/mcp", "", work_dir)
    assert (cfg.transport, cfg.url) == ("http", "https://mcp.example.test/mcp")
    assert load_mcp_servers(work_dir)["custom"].url == "https://mcp.example.test/mcp"
    # The same URL is accepted again; a different one for a configured
    # name is refused (it would sign in to a config the next task never loads).
    assert resolve_server("custom", "https://mcp.example.test/mcp", "", work_dir).url == cfg.url
    with pytest.raises(ValueError, match="already configured with"):
        resolve_server("custom", "https://mcp.example.test/sse", "sse", work_dir)
    sse = resolve_server("custom2", "https://mcp.example.test/sse", "sse", work_dir)
    assert (sse.transport, sse.url) == ("sse", "https://mcp.example.test/sse")


def test_resolve_rejects_stdio_and_unknown(home: Path, work_dir: str) -> None:
    project = Path(work_dir) / ".kiss" / "mcp.json"
    project.parent.mkdir()
    project.write_text(json.dumps({"mcpServers": {"local": {"command": "echo", "args": ["x"]}}}))
    with pytest.raises(ValueError, match="local stdio server; it needs no sign-in"):
        resolve_server("local", "", "", work_dir)
    with pytest.raises(ValueError, match="Unknown MCP server 'nope': pass its url"):
        resolve_server("nope", "", "", work_dir)
    # A new server is a named remote one.  A nameless entry was saved as
    # given, and so was a "stdio" one, without the URL it was asked for.
    for name, transport in (("", ""), ("custom", "stdio")):
        with pytest.raises(ValueError, match="needs a name and transport http or sse"):
            resolve_server(name, "https://mcp.example.test/mcp", transport, work_dir)
    assert not user_mcp_config_path().exists()


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------


def test_auth_tools_connect_and_finish(server: _AuthMCPServer, home: Path, work_dir: str) -> None:
    connect, finish = make_mcp_auth_tools(work_dir)
    assert (connect.__name__, finish.__name__) == (
        "connect_mcp_server",
        "finish_mcp_server_connect",
    )

    assert json.loads(finish("authdemo")) == {
        "ok": False,
        "error": "no sign-in in progress for 'authdemo'; call connect_mcp_server() first",
    }
    answer = json.loads(connect(" authdemo ", f" {server.url} "))
    assert answer["status"] == "consent_required"
    assert load_mcp_servers(work_dir)["authdemo"].url == server.url

    pending = json.loads(finish("authdemo"))
    assert pending["status"] == "pending"

    _approve(answer["verification_uri"])
    done = json.loads(finish(" authdemo "))
    assert done == {
        "ok": True,
        "message": "MCP server 'authdemo' is connected; its tools load in the next task.",
    }
    # Configured now, and the stored tokens work: connect succeeds at once.
    assert json.loads(connect("authdemo"))["ok"] is True


def test_finish_answers_for_its_own_projects_server_only(
    server: _AuthMCPServer, home: Path, tmp_path: Path,
) -> None:
    """A sign-in is to a server, a name at a URL: not to every project that uses the name.

    Two projects configure ``shared`` at different URLs.  The sign-in of
    one was found by the other's ``finish_mcp_server_connect`` under the
    name alone and reported as its own: its consent page, then
    "connected", for a server that project never signed in to.
    """
    tools = {}
    for project, url in (("a", "https://other.example/mcp"), ("b", server.url)):
        work_dir = tmp_path / project
        work_dir.mkdir()
        cfg = MCPServerConfig(name="shared", transport="http", url=url)
        save_mcp_server(cfg, "project", str(work_dir))
        tools[project] = make_mcp_auth_tools(str(work_dir))
    (_, finish_a), (connect_b, finish_b) = tools["a"], tools["b"]
    answer = json.loads(connect_b("shared"))
    assert answer["status"] == "consent_required"
    not_mine = {
        "ok": False,
        "error": "no sign-in in progress for 'shared'; call connect_mcp_server() first",
    }
    assert json.loads(finish_a("shared")) == not_mine
    _approve(answer["verification_uri"])
    assert json.loads(finish_b("shared"))["ok"] is True
    assert json.loads(finish_a("shared")) == not_mine


def test_finish_delivers_url_that_arrived_after_connect(
    server: _AuthMCPServer, home: Path, work_dir: str
) -> None:
    # Discovery outlasts connect's wait: the session exists (and is the
    # active one) but has no URL yet, so connect would answer "pending".
    save_mcp_server(_cfg(server), "user", work_dir)  # as connect does for a new server
    session = MCPLoginSession.start(_cfg(server), wait_seconds=0)
    assert MCPLoginSession.active("authdemo") is session
    assert _session_answer(session) == {
        "ok": False,
        "status": "pending",
        "error": "still contacting the server or opening the sign-in page; retry",
    }
    assert session.url_shown is False

    _connect, finish = make_mcp_auth_tools(work_dir)
    handed_over = json.loads(finish("authdemo"))
    assert session.auth_url
    assert handed_over["status"] == "consent_required"
    assert handed_over["verification_uri"] == session.auth_url
    assert session.url_shown is True

    # The URL is handed over exactly once; then it is pending until approval.
    assert json.loads(finish("authdemo")) == {
        "ok": False,
        "status": "pending",
        "error": "The user has not approved yet; call this tool again shortly.",
    }
    _approve(session.auth_url)
    assert json.loads(finish("authdemo"))["ok"] is True
    assert session.done


def test_auth_tools_errors(home: Path, work_dir: str, monkeypatch: pytest.MonkeyPatch) -> None:
    connect, _finish = make_mcp_auth_tools(work_dir)
    answer = json.loads(connect("nope"))
    assert answer["ok"] is False
    assert answer["error"].startswith("Unknown MCP server 'nope'")

    with occupy_loopback_port(mcp_oauth.MCP_REDIRECT_PORT):
        answer = json.loads(connect("busy", "https://mcp.example.test/mcp"))
    assert answer["ok"] is False
    assert answer["error"].startswith("cannot bind the redirect port:")

    # A token store that cannot be written (here to keep the client the
    # environment registers) fails the sign-in; it was reported as a busy port.
    (home / "mcp_auth").write_text("a file where the token directory goes")
    monkeypatch.setenv("KISS_MCP_BUSY_CLIENT_ID", "preregistered")
    answer = json.loads(connect("busy"))
    assert answer["ok"] is False
    assert answer["error"].startswith("MCP sign-in to 'busy' failed: ")
    assert "mcp_auth" in answer["error"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _approve_when_ready(name: str) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        session = MCPLoginSession.active(name)
        if session is not None and session.auth_url:
            _approve(session.auth_url)
            return
        time.sleep(0.05)


def test_main_usage_and_errors(
    home: Path, work_dir: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert mcp_oauth.main([]) == 2
    assert "Interactive OAuth sign-in for remote MCP servers" in capsys.readouterr().out
    monkeypatch.chdir(work_dir)
    assert mcp_oauth.main(["nope"]) == 1
    assert capsys.readouterr().out.startswith("error: Unknown MCP server 'nope'")


def test_main_signs_in(
    server: _AuthMCPServer,
    work_dir: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(work_dir)
    browser = threading.Thread(target=_approve_when_ready, args=("authdemo",), daemon=True)
    browser.start()
    assert mcp_oauth.main(["authdemo", server.url]) == 0
    browser.join(10)
    out = capsys.readouterr().out
    assert "Open this URL, sign in, and click Allow:\nhttp://127.0.0.1:" in out
    assert "MCP server 'authdemo' is connected" in out

    # Tokens now work: no URL is printed.
    assert mcp_oauth.main(["authdemo"]) == 0
    assert "Open this URL" not in capsys.readouterr().out


def _approve_after_delay(name: str, delay: float) -> None:
    """Approve *name*'s sign-in *delay* seconds after its URL appeared."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        session = MCPLoginSession.active(name)
        if session is not None and session.auth_url:
            time.sleep(delay)
            _approve(session.auth_url)
            return
        time.sleep(0.05)


def test_main_prints_url_once_while_waiting(
    server: _AuthMCPServer,
    work_dir: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(work_dir)
    # The approval lands only after several of main's 1-second waits.
    browser = threading.Thread(target=_approve_after_delay, args=("authdemo", 2.5), daemon=True)
    browser.start()
    started = time.monotonic()
    assert mcp_oauth.main(["authdemo", server.url]) == 0
    assert time.monotonic() - started >= 2.5
    browser.join(10)
    out = capsys.readouterr().out
    assert out.count("Open this URL, sign in, and click Allow:\n") == 1
    assert out.count(f"{server.base}/authorize?") == 1
    assert out.rstrip().endswith("'authdemo' is connected; its tools load in the next task.")


def test_main_refused(
    server: _AuthMCPServer,
    work_dir: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server.provider.refuse = True
    monkeypatch.chdir(work_dir)
    browser = threading.Thread(target=_approve_when_ready, args=("authdemo",), daemon=True)
    browser.start()
    assert mcp_oauth.main(["authdemo", server.url, "http"]) == 1
    browser.join(10)
    assert "sign-in refused (access_denied)" in capsys.readouterr().out
