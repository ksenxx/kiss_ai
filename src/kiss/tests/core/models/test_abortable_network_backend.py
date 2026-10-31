# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""An aborted connection pool also reaches the connections still being set up.

``_blocking_request`` aborts a stopped request by shutting its socket
down, but a request whose connection is not up yet has no socket in the
pool.  :class:`AbortableNetworkBackend` keeps every stream the pool
opens, and the stream a TLS upgrade yields (the upgrade detaches the
plain socket), and after an abort shuts each new one down before
httpcore writes on it.  Its streams also read in slices, so a read whose
socket is closed under it ends (no kernel wakes it).  The model-level
case, a plain-HTTP connect, is in ``test_f1_blocking_request_stop.py``;
here a TLS connection stopped before and during its handshake, a read
closed under it, a read timing out, a Unix socket, a proxied request
(whose pool is a proxy mount's, not the default transport's) and the
backend's retry sleep go through real servers.
"""

from __future__ import annotations

import http.server
import shutil
import socketserver
import ssl
import tempfile
import threading
import time
from pathlib import Path
from types import CodeType
from typing import Any

import httpx
import pytest
from httpcore._backends.sync import SyncBackend, SyncStream

from kiss.core.models.stream_abort import (
    AbortableNetworkBackend,
    make_pool_abortable,
    shutdown_pool_sockets,
)
from kiss.tests.conftest import posix_only
from kiss.tests.local_ws import make_test_tls
from kiss.tests.server.pause import Pause

_REQUESTER = "abortable-backend-request"


class _Ok(http.server.BaseHTTPRequestHandler):
    """Answer every GET with ``ok`` and record its path on the server."""

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self.server.paths.append(self.path)  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - http.server API
        pass


class _Held(_Ok):
    """Record every GET, then answer it only once the server's ``release`` is set."""

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self.server.paths.append(self.path)  # type: ignore[attr-defined]
        self.server.release.wait(60)  # type: ignore[attr-defined]


def _drop_error(_request: Any, _client_address: Any) -> None:
    """Swallow a handler's error: a connection the abort shut down resets the server's read."""


def _serve(server: Any) -> Any:
    """Start *server* (daemon handler threads, quiet errors, an empty path log) and return it."""
    server.daemon_threads = True
    server.handle_error = _drop_error
    server.paths = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _serve_held() -> Any:
    """Start a server that holds every request until its ``release`` is set."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Held)
    server.release = threading.Event()  # type: ignore[attr-defined]
    return _serve(server)


def _get_into(client: httpx.Client, url: str, outcome: list[BaseException]) -> None:
    """GET *url*, appending the error it raised (if any) to *outcome*."""
    try:
        client.get(url)
    except BaseException as error:  # noqa: BLE001 - the error is the result
        outcome.append(error)


def _on_the_requester(_frame_locals: dict[str, Any]) -> bool:
    """Whether the thread entering the :class:`Pause` hook is the test's requester."""
    return threading.current_thread().name == _REQUESTER


@pytest.mark.parametrize(
    ("hold_at", "reaches_a_socket"),
    [
        # Connected, not yet upgraded: the plain socket the backend kept is shut down.
        pytest.param(SyncStream.start_tls.__code__, True, id="before-the-handshake"),
        # Mid-handshake the plain socket is detached and the pool has no
        # stream for the connection: the abort reaches nothing, and the
        # upgraded stream is shut down as the backend gets it.
        pytest.param(ssl.SSLSocket.do_handshake.__code__, False, id="mid-handshake"),
    ],
)
def test_a_request_stopped_while_its_tls_connection_is_set_up_is_never_sent(
    tmp_path: Path, hold_at: CodeType, reaches_a_socket: bool
) -> None:
    """Aborted while its TLS connection is set up: the request never reaches the server.

    The requester is held at *hold_at* while the pool is aborted; it
    then fails with a transport error instead of sending the request.
    """
    certfile, keyfile, ca_file = make_test_tls(tmp_path)
    plain = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Ok)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile, keyfile)
    plain.socket = context.wrap_socket(plain.socket, server_side=True)
    server = _serve(plain)
    url = f"https://127.0.0.1:{server.server_port}/"
    client = httpx.Client(verify=ssl.create_default_context(cafile=str(ca_file)))
    try:
        make_pool_abortable(client)
        backend = client._transport._pool._network_backend  # type: ignore[attr-defined]
        assert isinstance(backend, AbortableNetworkBackend)
        make_pool_abortable(client)
        assert client._transport._pool._network_backend is backend  # type: ignore[attr-defined]
        # TLS through the wrapped streams; the server closes the connection after it.
        assert client.get(url, headers={"Connection": "close"}).text == "ok"
        outcome: list[BaseException] = []
        requester = threading.Thread(
            target=_get_into, args=(client, url, outcome), name=_REQUESTER, daemon=True
        )
        with Pause(hold_at, _on_the_requester) as held:
            requester.start()
            assert held.paused.wait(10), "the request never reached its TLS upgrade"
            assert shutdown_pool_sockets(client) is reaches_a_socket
        requester.join(10)
        assert not requester.is_alive(), "the stopped request still runs"
        assert outcome and isinstance(outcome[0], httpx.TransportError), outcome
        assert server.paths == ["/"], "the stopped request reached the server"
    finally:
        client.close()
        server.shutdown()
        server.server_close()


def test_a_read_whose_client_is_closed_under_it_ends_within_a_slice() -> None:
    """Closing the client mid-read: the waiting read ends, not at the client's timeout.

    Closing a socket wakes no thread already waiting on it in the kernel
    (and the shutdown of an abort can be missed the same way), so nothing
    is shut down here: only the read's own check of its socket between
    slices can end it.  Without that check the read, which has no
    timeout, would wait forever.
    """
    server = _serve_held()
    client = httpx.Client(timeout=None)
    try:
        make_pool_abortable(client)
        outcome: list[BaseException] = []
        requester = threading.Thread(
            target=_get_into,
            args=(client, f"http://127.0.0.1:{server.server_port}/", outcome),
            name=_REQUESTER,
            daemon=True,
        )
        requester.start()
        deadline = time.monotonic() + 10
        while not server.paths and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.paths == ["/"], "the request never reached the server"
        # Let the requester settle into its wait for the reply; were it not
        # there yet, its read of the closed socket fails at once (also fine).
        time.sleep(0.2)
        client.close()
        requester.join(5)
        assert not requester.is_alive(), "the read still waits on its closed socket"
        assert outcome and isinstance(outcome[0], httpx.ReadError), outcome
    finally:
        server.release.set()  # type: ignore[attr-defined]
        server.shutdown()
        server.server_close()


def test_a_read_in_slices_still_times_out_at_the_client_timeout() -> None:
    """The slices keep the read's own deadline: a reply that never comes times out on time."""
    server = _serve_held()
    client = httpx.Client(timeout=0.8)
    try:
        make_pool_abortable(client)
        started = time.monotonic()
        with pytest.raises(httpx.ReadTimeout):
            client.get(f"http://127.0.0.1:{server.server_port}/")
        assert 0.8 <= time.monotonic() - started < 5
    finally:
        client.close()
        server.release.set()  # type: ignore[attr-defined]
        server.shutdown()
        server.server_close()


@posix_only("httpx's uds transport and AF_UNIX sockets")
def test_a_unix_socket_pool_is_aborted_too_and_its_retries_sleep() -> None:
    """Over a Unix socket: the abort shuts the kept connection down and every later one.

    httpcore's connect retries sleep through the backend as well.
    """
    # A Unix socket path is limited to about 100 bytes; pytest's tmp_path is longer.
    short_dir = tempfile.mkdtemp(dir="/tmp")
    path = str(Path(short_dir) / "http.sock")
    server = _serve(socketserver.ThreadingUnixStreamServer(path, _Ok))
    client = httpx.Client(transport=httpx.HTTPTransport(uds=path))
    retrying = httpx.Client(transport=httpx.HTTPTransport(uds=path + ".none", retries=1))
    try:
        make_pool_abortable(client)
        assert client.get("http://localhost/").text == "ok"
        assert shutdown_pool_sockets(client), "the kept-alive connection was not shut down"
        with pytest.raises(httpx.TransportError):
            client.get("http://localhost/second")
        assert server.paths == ["/"], "a request after the abort reached the server"
        make_pool_abortable(retrying)
        with pytest.raises(httpx.ConnectError):
            retrying.get("http://localhost/")
    finally:
        client.close()
        retrying.close()
        server.shutdown()
        server.server_close()
        shutil.rmtree(short_dir, ignore_errors=True)


def test_a_client_without_a_reachable_pool_is_left_alone() -> None:
    """Best-effort: an object without httpx's pool attributes is not touched (nor raises)."""
    plain = object()
    make_pool_abortable(plain)
    assert not hasattr(plain, "_transport")
    assert shutdown_pool_sockets(plain) is False


def test_a_proxied_request_stopped_while_it_connects_is_never_sent() -> None:
    """Through a proxy: the mount's pool is aborted too, so the stopped request never leaves.

    A client with a proxy (``HTTP(S)_PROXY`` or ``proxy=``) sends through
    the proxy mount's pool, not the default transport's.  The requester
    is held entering the TCP connect to the proxy while the client is
    aborted; the connection is shut down as it comes up and the request
    never reaches the proxy (here an ordinary server, which answers the
    absolute-URI request a forward proxy receives).  A ``None`` mount
    (what a ``NO_PROXY`` pattern maps to) is skipped.
    """
    proxy = _serve(http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Ok))
    proxy_url = f"http://127.0.0.1:{proxy.server_port}"
    # trust_env=False: an ambient HTTP(S)_PROXY would add mounts of its own.
    client = httpx.Client(
        mounts={"http://": httpx.HTTPTransport(proxy=proxy_url), "all://localhost": None},
        trust_env=False,
    )
    url = "http://kiss-proxied.test/"
    try:
        make_pool_abortable(client)
        proxied = [t for t in client._mounts.values() if t is not None]  # type: ignore[attr-defined]
        assert len(proxied) == 1
        pool = proxied[0]._pool  # type: ignore[attr-defined]
        assert isinstance(pool._network_backend, AbortableNetworkBackend)
        assert isinstance(client._transport._pool._network_backend, AbortableNetworkBackend)  # type: ignore[attr-defined]
        # Through the proxy and the wrapped streams; the proxy closes the connection after it.
        assert client.get(url, headers={"Connection": "close"}).text == "ok"
        assert proxy.paths == [url]
        outcome: list[BaseException] = []
        requester = threading.Thread(
            target=_get_into, args=(client, url + "stopped", outcome), name=_REQUESTER, daemon=True
        )
        with Pause(SyncBackend.connect_tcp.__code__, _on_the_requester) as held:
            requester.start()
            assert held.paused.wait(10), "the request never reached its connect"
            shutdown_pool_sockets(client)
            assert pool._network_backend._aborted
        requester.join(10)
        assert not requester.is_alive(), "the stopped request still runs"
        assert outcome and isinstance(outcome[0], httpx.TransportError), outcome
        assert proxy.paths == [url], "the stopped request reached the proxy"
    finally:
        client.close()
        proxy.shutdown()
        proxy.server_close()
