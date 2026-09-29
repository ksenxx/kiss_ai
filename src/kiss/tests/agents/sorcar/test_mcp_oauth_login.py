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
fixed loopback callback ``http://localhost:53683/callback``.

The redirect port is fixed by design (a dynamically registered client
keeps working only if its redirect URI never changes), so these tests
need port 53683 free: ``_own_redirect_port`` reserves it across
concurrently running pytest processes.

Branches not exercised:

* ``_noninteractive_callback`` in ``mcp_servers``: the SDK calls the
  redirect handler first, and ``_noninteractive_redirect`` always
  raises, so the callback handler can never be reached.
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
import json
import secrets
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

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

from kiss.agents.sorcar import mcp_oauth
from kiss.agents.sorcar.mcp_oauth import (
    KNOWN_MCP_SERVERS,
    MCP_REDIRECT_PORT,
    MCP_REDIRECT_URI,
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
    build_oauth_provider,
    load_mcp_servers,
    user_mcp_config_path,
)
from kiss.tests.conftest import hold_loopback_port, occupy_loopback_port

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
        del self.refresh[refresh_token.token]
        return self._issue(refresh_token.client_id, scopes or refresh_token.scopes)

    async def load_access_token(self, token: str) -> AccessToken | None:
        return self.access.get(token)

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        self.access.pop(token.token, None)
        self.refresh.pop(token.token, None)


class _AuthMCPServer:
    """A FastMCP server with OAuth, running under uvicorn on a thread."""

    def __init__(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        self.base = f"http://127.0.0.1:{port}"
        self.url = f"{self.base}/mcp"
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
        config = uvicorn.Config(
            mcp.streamable_http_app(), host="127.0.0.1", port=port, log_level="error"
        )
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
def _own_redirect_port() -> Iterator[None]:
    """Reserve port 53683 across pytest processes and free it afterwards.

    Cancels any login session the test left running before the
    reservation ends, so the next test (in any process) can bind it.
    """
    with hold_loopback_port(MCP_REDIRECT_PORT):
        yield
        session = MCPLoginSession._active
        if session is not None:
            session.cancel()
            session.wait(10)
        MCPLoginSession._active = None


def _cfg(server: _AuthMCPServer, name: str = "authdemo") -> MCPServerConfig:
    return MCPServerConfig(name=name, transport="http", url=server.url)


def _approve(auth_url: str) -> httpx.Response:
    """Play the user's browser: open the URL and follow every redirect."""
    return httpx.get(auth_url, follow_redirects=True, trust_env=False, timeout=20)


def _tokens(name: str) -> Any:
    return asyncio.run(FileTokenStorage(name).get_tokens())


def _client_info(name: str) -> Any:
    return asyncio.run(FileTokenStorage(name).get_client_info())


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
    assert query["redirect_uri"] == MCP_REDIRECT_URI
    assert query["code_challenge_method"] == "S256"
    assert query["client_id"] in server.provider.clients
    assert MCPLoginSession.active("authdemo") is session
    assert MCPLoginSession.active("other") is None

    answer = _session_answer(session)
    assert answer["status"] == "consent_required"
    assert answer["verification_uri"] == session.auth_url
    assert "finish_mcp_server_connect('authdemo')" in answer["instructions"]

    wrong = httpx.get(f"http://127.0.0.1:{MCP_REDIRECT_PORT}/elsewhere", trust_env=False)
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
    assert [str(u) for u in registered.redirect_uris or []] == [MCP_REDIRECT_URI]
    assert registered.client_name == mcp_oauth.PRODUCT_NAME

    tokens = _tokens("authdemo")
    assert tokens is not None
    assert tokens.access_token in server.provider.access
    assert _client_info("authdemo").client_id == query["client_id"]

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
    auth = build_oauth_provider(cfg)
    assert auth.context.client_metadata.token_endpoint_auth_method == "none"
    manager = MCPManager()
    try:
        conn = manager.connect(cfg)
        assert _LOGIN_HINT in conn.error
    finally:
        manager.shutdown()
    assert server.provider.access == {}


def test_manager_refreshes_expired_tokens(server: _AuthMCPServer, home: Path) -> None:
    server.provider.expires_in = 1
    _login(server)
    server.provider.expires_in = 3600
    storage = FileTokenStorage("authdemo")
    assert storage.get_oauth_metadata() is not None
    old = storage._read()["tokens"]["access_token"]
    time.sleep(1.5)  # the access token expires; the refresh token stays valid
    manager = MCPManager()
    try:
        conn = manager.connect(_cfg(server))
        assert conn.error == ""
    finally:
        manager.shutdown()
    new = storage._read()["tokens"]["access_token"]
    assert new != old
    assert new in server.provider.access


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
    assert _tokens("authdemo") is None


def test_new_session_replaces_running_one(server: _AuthMCPServer, home: Path) -> None:
    first = MCPLoginSession.start(_cfg(server))
    assert first.auth_url
    second = MCPLoginSession.start(_cfg(server))
    assert first.wait(0)
    assert first.done is False
    assert first.error == "sign-in cancelled"
    assert MCPLoginSession.active("authdemo") is second
    assert second.auth_url and second.auth_url != first.auth_url
    _approve(second.auth_url)
    assert second.wait(30)
    assert second.done


def test_half_open_redirect_connection_does_not_pin_the_port(
    server: _AuthMCPServer, home: Path
) -> None:
    session = MCPLoginSession.start(_cfg(server))
    assert session.auth_url
    # A browser that opens the connection but never finishes its request:
    # the handler's socket timeout must return the serving thread to the
    # cancel check instead of blocking on the missing headers forever.
    half_open = socket.create_connection(("127.0.0.1", MCP_REDIRECT_PORT), timeout=10)
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


def test_session_answer_before_url_is_pending(server: _AuthMCPServer) -> None:
    session = MCPLoginSession(_cfg(server))
    assert _session_answer(session) == {
        "ok": False,
        "status": "pending",
        "error": "still contacting the server; retry",
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
    assert [str(u) for u in info.redirect_uris] == [MCP_REDIRECT_URI]

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
    assert _client_info("authdemo").client_id == "kiss-prereg"


# ---------------------------------------------------------------------------
# Stale registrations (older KISS registered http://localhost:0/callback)
# ---------------------------------------------------------------------------


def _store_registration(name: str, client_id: str, redirect_uri: str) -> None:
    """Store a client registration for *name* plus a dead access token."""
    storage = FileTokenStorage(name)
    info = OAuthClientInformationFull.model_validate(
        {
            **mcp_oauth.client_metadata().model_dump(mode="json", exclude_none=True),
            "client_id": client_id,
            "redirect_uris": [redirect_uri],
        }
    )
    storage._locked_update("client_info", info.model_dump(mode="json", exclude_none=True))
    storage._locked_update("tokens", {"access_token": f"dead-{client_id}", "token_type": "Bearer"})


def test_drop_stale_registration_only_for_other_redirect_uris(home: Path) -> None:
    storage = FileTokenStorage("stale-check")
    assert drop_stale_registration(storage) is False

    _store_registration("stale-check", "current", MCP_REDIRECT_URI)
    assert drop_stale_registration(storage) is False
    assert _client_info("stale-check").client_id == "current"
    assert _tokens("stale-check").access_token == "dead-current"

    _store_registration("stale-check", "old", "http://localhost:0/callback")
    assert drop_stale_registration(storage) is True
    assert _client_info("stale-check") is None
    assert _tokens("stale-check") is None
    assert storage._read()["expires_at"] is None
    # Nothing left to drop.
    assert drop_stale_registration(storage) is False


def test_login_reregisters_a_stale_client(server: _AuthMCPServer, home: Path) -> None:
    _store_registration("authdemo", "stale-client", "http://localhost:0/callback")
    session = _login(server)
    # The dead token forced a sign-in, and the stale registration was
    # replaced instead of being sent to the authorization endpoint.
    new_id = parse_qs(urlsplit(session.auth_url).query)["client_id"][0]
    assert new_id != "stale-client"
    assert new_id in server.provider.clients
    assert server.provider.registrations == 1
    info = _client_info("authdemo")
    assert info.client_id == new_id
    assert [str(u) for u in info.redirect_uris] == [MCP_REDIRECT_URI]
    assert _tokens("authdemo").access_token in server.provider.access


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
    _store_registration("authdemo", "stale-client", "http://localhost:0/callback")
    session = _login(server)
    # The environment's client replaced the stored one; the stale check
    # never ran, so nothing was registered dynamically.
    assert parse_qs(urlsplit(session.auth_url).query)["client_id"] == ["kiss-prereg"]
    assert server.provider.registrations == 0
    assert _client_info("authdemo").client_id == "kiss-prereg"
    assert _tokens("authdemo").access_token in server.provider.access


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
    stored = FileTokenStorage("authdemo").get_oauth_metadata()
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


def test_finish_delivers_url_that_arrived_after_connect(
    server: _AuthMCPServer, home: Path, work_dir: str
) -> None:
    # Discovery outlasts connect's wait: the session exists (and is the
    # active one) but has no URL yet, so connect would answer "pending".
    session = MCPLoginSession.start(_cfg(server), wait_seconds=0)
    assert MCPLoginSession.active("authdemo") is session
    assert _session_answer(session) == {
        "ok": False,
        "status": "pending",
        "error": "still contacting the server; retry",
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


def test_auth_tools_errors(home: Path, work_dir: str) -> None:
    connect, _finish = make_mcp_auth_tools(work_dir)
    answer = json.loads(connect("nope"))
    assert answer["ok"] is False
    assert answer["error"].startswith("Unknown MCP server 'nope'")

    with occupy_loopback_port(MCP_REDIRECT_PORT):
        answer = json.loads(connect("busy", "https://mcp.example.test/mcp"))
    assert answer["ok"] is False
    assert answer["error"].startswith("cannot bind the redirect port:")


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
