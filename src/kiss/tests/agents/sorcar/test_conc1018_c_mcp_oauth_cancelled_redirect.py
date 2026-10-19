# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A superseded MCP sign-in must not open its consent page for the user.

``MCPLoginSession.start`` replaces the active session: it cancels the
previous one and waits at most five seconds for its thread.  That thread
may still sit in the first HTTP request to the MCP server; when the
request finally returns, the OAuth flow reaches ``_on_redirect`` and --
before the fix -- opened the dead session's sign-in page next to the
replacement's, although its redirect listener was already closed.

The server here is the real OAuth-protected FastMCP server of
``test_mcp_oauth_login`` behind a TCP proxy that holds the first
connection for longer than the replacement wait before passing it on,
so the first session is still in flight when the second one starts.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import uvicorn
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.server.fastmcp import FastMCP

from kiss.agents.sorcar.mcp_oauth import MCPLoginSession
from kiss.agents.sorcar.mcp_servers import MCPServerConfig
from kiss.core.browser_handoff import set_browser_tab_opener
from kiss.tests.agents.sorcar.test_mcp_oauth_login import _echo, _MemoryAuthProvider
from kiss.tests.conftest import find_free_port

# Longer than the 5 s ``_replace_active`` waits for the previous thread.
_FIRST_CONNECTION_DELAY = 6.5


def _pump(source: socket.socket, sink: socket.socket) -> None:
    """Copy bytes from *source* to *sink* until *source* closes."""
    try:
        while chunk := source.recv(65536):
            sink.sendall(chunk)
    except OSError:
        pass
    finally:
        try:
            sink.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class _DelayingProxy:
    """A TCP proxy that holds its first accepted connection before forwarding it."""

    def __init__(self, backend_port: int, delay: float) -> None:
        self.backend_port = backend_port
        self.delay = delay
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()
        self.port = int(self.listener.getsockname()[1])
        self._delayed_once = False
        self._closing = threading.Event()
        self._sockets: list[socket.socket] = []  # every accepted / upstream connection
        self._workers: list[threading.Thread] = []
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                client, _ = self.listener.accept()
            except OSError:
                return
            self._sockets.append(client)
            worker = threading.Thread(target=self._forward, args=(client,), daemon=True)
            self._workers.append(worker)
            worker.start()

    def _forward(self, client: socket.socket) -> None:
        if not self._delayed_once:
            self._delayed_once = True
            self._closing.wait(self.delay)
        if self._closing.is_set():
            client.close()
            return
        try:
            upstream = socket.create_connection(("127.0.0.1", self.backend_port), timeout=10)
        except OSError:
            client.close()
            return
        self._sockets.append(upstream)
        pump = threading.Thread(target=_pump, args=(upstream, client), daemon=True)
        self._workers.append(pump)
        pump.start()
        _pump(client, upstream)
        # Closing ``upstream`` while the reverse pump still blocks in its
        # ``recv`` would leave that thread stuck for good (on POSIX ``close``
        # does not wake a blocked ``recv``, and ``_DelayingProxy.close`` can no
        # longer ``shutdown`` a closed socket): wait for the pump first.  It ends when
        # the backend hangs up or ``close`` shuts the upstream socket down.
        pump.join()
        upstream.close()
        client.close()

    def close(self) -> None:
        """Stop accepting, cut every connection and join every thread the proxy started.

        Closing a listening socket does not wake a thread blocked in
        ``accept`` on Linux (it keeps the port until ``accept`` returns),
        nor does closing a connected socket wake a blocked ``recv``:
        ``shutdown`` does both.
        """
        self._closing.set()
        _shutdown(self.listener)
        self.listener.close()
        self._thread.join(timeout=5)  # nothing is accepted (and tracked) after this
        for sock in self._sockets:
            _shutdown(sock)
        for worker in self._workers:
            worker.join(timeout=5)
        stuck = [t for t in [self._thread, *self._workers] if t.is_alive()]
        assert not stuck, f"proxy threads still running: {stuck}"


def _shutdown(sock: socket.socket) -> None:
    """Wake whatever blocks on *sock*; a socket already closed by its pump is fine.

    ``shutdown`` wakes a ``recv`` blocked on another thread on POSIX but
    not on Windows, where only ``close`` does (the ``recv`` then raises
    ``WSAENOTSOCK``, which the pumps treat like end of stream).
    """
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    if os.name == "nt":
        sock.close()


class _SlowAuthMCPServer:
    """The OAuth FastMCP server of ``test_mcp_oauth_login`` reached through a delaying proxy.

    Every URL the server publishes (issuer, resource, authorization page)
    names the proxy port, so the whole flow travels through the proxy.
    """

    def __init__(self) -> None:
        backend_port = find_free_port()
        self.proxy = _DelayingProxy(backend_port, _FIRST_CONNECTION_DELAY)
        self.base = f"http://127.0.0.1:{self.proxy.port}"
        self.url = f"{self.base}/mcp"
        mcp = FastMCP(
            "authdemo",
            host="127.0.0.1",
            port=backend_port,
            auth_server_provider=_MemoryAuthProvider(),
            auth=AuthSettings(
                issuer_url=self.base,  # type: ignore[arg-type]
                resource_server_url=self.url,  # type: ignore[arg-type]
                client_registration_options=ClientRegistrationOptions(enabled=True),
            ),
        )
        mcp.add_tool(_echo)
        config = uvicorn.Config(
            mcp.streamable_http_app(), host="127.0.0.1", port=backend_port, log_level="error"
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
        self.proxy.close()
        self._server.should_exit = True
        self._thread.join(timeout=10)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test KISS home; headless so no default browser is ever launched."""
    kiss_home = tmp_path / "kiss_home"
    kiss_home.mkdir()
    monkeypatch.setenv("KISS_HOME", str(kiss_home))
    monkeypatch.setenv("KISS_HEADLESS", "1")
    monkeypatch.delenv("KISS_MCP_CLIENT_METADATA_URL", raising=False)
    return kiss_home


@pytest.fixture
def slow_server(home: Path) -> Iterator[_SlowAuthMCPServer]:
    srv = _SlowAuthMCPServer()
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


@pytest.fixture(autouse=True)
def _own_redirect_port(mcp_redirect_port: int) -> Iterator[int]:
    """Own ephemeral redirect port; cancel whatever session the test left running."""
    yield mcp_redirect_port
    session = MCPLoginSession._active
    if session is not None:
        session.cancel()
        session.wait(10)
    MCPLoginSession._active = None


def test_proxy_close_wakes_accept_and_ends_every_connection() -> None:
    """``_DelayingProxy.close`` leaves no thread and no listening port behind."""
    backend = socket.socket()
    backend.bind(("127.0.0.1", 0))
    backend.listen()
    proxy = _DelayingProxy(int(backend.getsockname()[1]), 0)
    client = socket.create_connection(("127.0.0.1", proxy.port), timeout=5)
    try:
        upstream, _ = backend.accept()  # the proxy connected through: both pumps are blocked
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not (
            len(proxy._workers) == 2 and all(t.is_alive() for t in proxy._workers)
        ):
            time.sleep(0.01)
        assert proxy._thread.is_alive() and all(t.is_alive() for t in proxy._workers)
        proxy.close()
        assert not proxy._thread.is_alive()
        assert all(not t.is_alive() for t in proxy._workers)
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", proxy.port), timeout=1).close()
        assert client.recv(1) == b""  # the proxy hung up on the client ...
        assert upstream.recv(1) == b""  # ... and on the backend
        upstream.close()
    finally:
        client.close()
        backend.close()


def test_superseded_session_does_not_open_its_page(slow_server: _SlowAuthMCPServer) -> None:
    opened: list[str] = []

    def open_tab(url: str) -> bool:
        opened.append(url)
        return True

    set_browser_tab_opener(open_tab)
    try:
        cfg = MCPServerConfig(name="authdemo", transport="http", url=slow_server.url)
        # The first request of this session sits in the proxy's delay ...
        first = MCPLoginSession.start(cfg, wait_seconds=0.5)
        assert first.auth_url == "" and not first.wait(0)
        # ... so the replacement finds it still running, cancels it, and
        # starts its own flow, which opens the only page the user gets.
        second = MCPLoginSession.start(cfg)
        assert second.auth_url, second.error
        assert opened == [second.auth_url]
        assert MCPLoginSession.active("authdemo") is second
        # The first session's request has now gone through; its flow ends
        # as cancelled without putting a second, dead sign-in page in
        # front of the user.
        assert first.wait(_FIRST_CONNECTION_DELAY + 10)
        time.sleep(0.5)  # an opener thread spawned anyway would have landed by now
        assert opened == [second.auth_url]
        assert first.auth_url == ""
        assert first.done is False
        assert first.error == "sign-in cancelled"
        # The surviving session still completes.
        httpx.get(second.auth_url, follow_redirects=True, trust_env=False, timeout=20)
        assert second.wait(30)
        assert second.done, second.error
    finally:
        set_browser_tab_opener(None)
