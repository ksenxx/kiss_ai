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
here a TLS connection stopped before and during its handshake, a
connect still looking its host up, a read closed under it, a read
timing out, a Unix socket, a proxied request (whose pool is a proxy
mount's, not the default transport's) and the backend's retry sleep go
through real servers.
"""

from __future__ import annotations

import _thread
import contextlib
import functools
import http.server
import shutil
import socket
import socketserver
import ssl
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import CodeType, FrameType
from typing import Any

import httpcore
import httpcore2
import httpx
import httpx2
import pytest
import truststore
from httpcore._backends.sync import SyncBackend

from kiss.core.locked import locked_call
from kiss.core.models.stream_abort import (
    AbortableNetworkBackend,
    RequestAbortWatchdog,
    _AbortableStream,
    _shutdown_network_streams,
    make_pool_abortable,
    shutdown_pool_sockets,
)
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.conftest import posix_only
from kiss.tests.local_ws import make_test_tls
from kiss.tests.server.pause import Pause

_REQUESTER = "abortable-backend-request"
_STOPPED_EXIT = "watchdog-with-a-stopped-exit"

# The SDKs send through two HTTP stacks with one API and unrelated classes,
# the exceptions included: httpx on httpcore (OpenAI, Gemini) and httpx2 on
# httpcore2 (Anthropic).  The backend speaks the one it wraps.
both_stacks = pytest.mark.parametrize("stack", [httpx, httpx2], ids=["httpx", "httpx2"])


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


def _get_into(client: Any, url: str, outcome: list[BaseException]) -> None:
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
        pytest.param(_AbortableStream.start_tls.__code__, True, id="before-the-handshake"),
        # Mid-handshake the plain socket is detached; the upgraded socket,
        # kept before its handshake, is the one shut down.
        pytest.param(ssl.SSLSocket.do_handshake.__code__, True, id="mid-handshake"),
    ],
)
@both_stacks
def test_a_request_stopped_while_its_tls_connection_is_set_up_is_never_sent(
    tmp_path: Path, hold_at: CodeType, reaches_a_socket: bool, stack: Any
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
    client = stack.Client(verify=ssl.create_default_context(cafile=str(ca_file)))
    try:
        make_pool_abortable(client)
        backend = client._transport._pool._network_backend
        assert isinstance(backend, AbortableNetworkBackend)
        make_pool_abortable(client)
        assert client._transport._pool._network_backend is backend
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
        assert outcome and isinstance(outcome[0], stack.TransportError), outcome
        assert server.paths == ["/"], "the stopped request reached the server"
    finally:
        client.close()
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("trusted", [True, False], ids=["trusted-ca", "untrusted"])
@both_stacks
def test_an_os_trust_store_context_connects_and_still_verifies(
    tmp_path: Path, trusted: bool, stack: Any
) -> None:
    """A ``truststore.SSLContext`` (httpx2's default) verifies after the deferred handshake.

    Verifying inside ``wrap_socket``, before the handshake, refused every
    connection on macOS and Windows (an empty chain), which broke every
    Anthropic call there; an untrusted server must still be refused.
    """
    certfile, keyfile, ca_file = make_test_tls(tmp_path)
    plain = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Ok)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile, keyfile)
    plain.socket = context.wrap_socket(plain.socket, server_side=True)
    server = _serve(plain)
    client_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    if trusted:
        client_context.load_verify_locations(cafile=str(ca_file))
    client = stack.Client(verify=client_context)
    try:
        make_pool_abortable(client)
        url = f"https://127.0.0.1:{server.server_port}/"
        if trusted:
            assert client.get(url).text == "ok"
        else:
            with pytest.raises(stack.ConnectError):
                client.get(url)
    finally:
        client.close()
        server.shutdown()
        server.server_close()


def _in_a_handshake(requester: threading.Thread) -> bool:
    """Whether *requester* is inside ``SSLSocket.do_handshake`` right now."""
    frame = sys._current_frames().get(requester.ident or -1)
    return frame is not None and frame.f_code is ssl.SSLSocket.do_handshake.__code__


@both_stacks
def test_a_request_stopped_in_a_handshake_the_peer_never_answers_fails_at_once(stack: Any) -> None:
    """An abort during the TLS handshake shuts down the socket doing the handshake.

    ``wrap_socket`` detaches the plain socket the backend kept and the
    upgraded one was kept only once the handshake was done, so an abort
    meanwhile found a dead descriptor: a Stop during a handshake a peer
    never answers (it accepted the connection and went silent) waited
    out the connect timeout, 5 s on the OpenAI and Anthropic clients.
    """
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    accepted: list[socket.socket] = []

    def accept_and_say_nothing() -> None:
        with contextlib.suppress(OSError):
            while True:
                accepted.append(listener.accept()[0])

    threading.Thread(target=accept_and_say_nothing, daemon=True).start()
    client = stack.Client(verify=False, timeout=stack.Timeout(10.0, connect=10.0))
    outcome: list[BaseException] = []
    requester = threading.Thread(
        target=_get_into,
        args=(client, f"https://127.0.0.1:{listener.getsockname()[1]}/", outcome),
        name=_REQUESTER,
        daemon=True,
    )
    try:
        make_pool_abortable(client)
        requester.start()
        deadline = time.monotonic() + 10
        while not _in_a_handshake(requester):
            assert time.monotonic() < deadline, "the request never reached its handshake"
            time.sleep(0.01)
        assert shutdown_pool_sockets(client), "the abort found no socket to shut down"
        requester.join(5)
        assert not requester.is_alive(), "the stopped request waited for the connect timeout"
        assert outcome and isinstance(outcome[0], stack.ConnectError), outcome
    finally:
        listener.close()
        for conn in accepted:
            conn.close()
        requester.join(15)
        client.close()


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


@both_stacks
def test_a_read_in_slices_still_times_out_at_the_client_timeout(stack: Any) -> None:
    """The slices keep the read's own deadline: a reply that never comes times out on time.

    A slice that ends without data is the wrapped stream's own timeout,
    of its own package: taken for httpcore's on an httpcore2 pool, it
    ended the read at the first slice.
    """
    server = _serve_held()
    client = stack.Client(timeout=0.8)
    try:
        make_pool_abortable(client)
        started = time.monotonic()
        with pytest.raises(stack.ReadTimeout):
            client.get(f"http://127.0.0.1:{server.server_port}/")
        assert 0.8 <= time.monotonic() - started < 5
    finally:
        client.close()
        server.release.set()  # type: ignore[attr-defined]
        server.shutdown()
        server.server_close()


def _to_port(port: int, frame_locals: dict[str, Any]) -> bool:
    """A :class:`Pause` matcher: the call (a lookup, a connect) whose ``port`` is *port*."""
    return frame_locals.get("port") == port


def _connect_into(backend: AbortableNetworkBackend, port: int, outcome: list[Any]) -> None:
    """Connect to local *port* through *backend*; append the stream, or the error, to *outcome*."""
    try:
        outcome.append(backend.connect_tcp("127.0.0.1", port, timeout=30))
    except BaseException as error:  # noqa: BLE001 - the error is the result
        outcome.append(error)


@pytest.mark.parametrize(
    ("core", "stopped"),
    [(httpcore, False), (httpcore2, False), (httpcore, True)],
    ids=["abort", "abort-on-httpcore2", "injected-stop"],
)
def test_a_connect_whose_address_lookup_has_not_returned_ends_at_the_abort(
    core: Any, stopped: bool
) -> None:
    """An abort ends a connect that is still looking its host up; the late connection is closed.

    ``getaddrinfo`` takes no timeout, and before it returns there is no
    socket an abort could shut down or a connect timeout could run on:
    the connect, and with it its request's deadline and Stop, waited for
    as long as the resolver did.  The lookup is held at its entry, as a
    resolver that does not answer holds it.  The runner's Stop, injected
    into the waiting thread (a streamed request has no watchdog yet),
    ends the wait as well.  The error is the one of the wrapped backend's
    package: its pool maps no other.
    """
    backend = AbortableNetworkBackend(core.SyncBackend())
    with socket.create_server(("127.0.0.1", 0)) as listener:
        listener.settimeout(10)
        port = listener.getsockname()[1]
        outcome: list[Any] = []
        requester = threading.Thread(
            target=_connect_into, args=(backend, port, outcome), name=_REQUESTER, daemon=True
        )
        with Pause(socket.getaddrinfo.__code__, functools.partial(_to_port, port)) as held:
            requester.start()
            assert held.paused.wait(10), "the connect never reached its address lookup"
            if stopped:
                assert inject_async_exception(requester.ident or 0, KeyboardInterrupt) == 1
            else:
                assert backend.abort() is False, "no socket exists yet"
            requester.join(5)
            assert not requester.is_alive(), "the connect waits for its address lookup"
            ended_with = KeyboardInterrupt if stopped else core.ConnectError
            assert isinstance(outcome[0], ended_with), outcome
            if not stopped:
                # Still aborted: a retry is refused, with no lookup or connection of its own.
                with pytest.raises(core.ConnectError):
                    backend.connect_tcp("127.0.0.1", port, timeout=30)
        # The lookup returns: the connection comes up with nobody waiting for it.
        late, _ = listener.accept()
        with late:
            late.settimeout(10)
            assert late.recv(1) == b"", "the connection nobody waited for is still open"
        backend.rearm()
        stream = backend.connect_tcp("127.0.0.1", port, timeout=30)
        try:
            peer, address = listener.accept()
            peer.close()
            assert address == stream.get_extra_info("socket").getsockname(), (
                "the refused retry connected after all"
            )
        finally:
            stream.close()


def _late_stop(aborting: Pause, frame: FrameType, event: str, arg: Any) -> None:
    """Profile hook: a Stop lands inside the watchdog's ``Thread.start()``, once the thread
    it spawned is held *aborting* the pool (or after 30 s, which the test then reports)."""
    if event == "c_return" and arg is _thread.start_joinable_thread:
        sys.setprofile(None)
        aborting.paused.wait(30)
        aborting.release()
        raise KeyboardInterrupt


def test_a_stop_inside_the_watchdogs_start_leaves_the_pool_usable() -> None:
    """``__exit__`` never runs for the interrupted ``__enter__``, so the entry itself re-arms
    the pool its watchdog (due at its first poll) aborted: the next request gets through.
    The watchdog thread, named as the requester, is held once it has marked the pool aborted."""
    server = _serve(http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Ok))
    client = httpx.Client(trust_env=False)
    try:
        make_pool_abortable(client)
        with Pause(_shutdown_network_streams.__code__, _on_the_requester) as aborting:
            sys.setprofile(functools.partial(_late_stop, aborting))
            with pytest.raises(KeyboardInterrupt), RequestAbortWatchdog(client, 0, _REQUESTER):
                raise AssertionError("the body must not run")
        assert aborting.paused.is_set(), "the watchdog thread never aborted the pool"
        assert client.get(f"http://127.0.0.1:{server.server_port}/").text == "ok"
    finally:
        sys.setprofile(None)
        client.close()
        server.shutdown()
        server.server_close()


def _stop_as_it_calls(caller: Any, callee: Any, frame: FrameType, event: str, arg: Any) -> None:
    """Profile hook: a Stop lands as *caller* calls *callee*, before *callee* did anything."""
    if (
        event == "call"
        and frame.f_code is callee.__code__
        and frame.f_back is not None
        and frame.f_back.f_code is caller.__code__
    ):
        sys.setprofile(None)
        raise KeyboardInterrupt


def _cut_request_with_a_stopped_exit(
    client: httpx.Client, url: str, hook: Any, stops: list[BaseException]
) -> None:
    """Send requests under a watchdog due at once until it cuts one; *hook* stops its exit."""
    sys.setprofile(hook)
    try:
        with RequestAbortWatchdog(client, 0, _STOPPED_EXIT):
            while True:
                client.get(url)
    except KeyboardInterrupt as stop:
        stops.append(stop)
    finally:
        sys.setprofile(None)


@pytest.mark.parametrize("owner_lives", [False, True], ids=["owner-dies", "owner-lives"])
@pytest.mark.parametrize(
    ("caller", "callee"),
    [
        (RequestAbortWatchdog.__exit__, locked_call),
        (RequestAbortWatchdog._disarm, AbortableNetworkBackend.rearm),
    ],
    ids=["before-the-disarm", "inside-the-disarm"],
)
def test_a_stop_inside_the_watchdogs_exit_leaves_the_pool_usable(
    caller: Any, callee: Any, owner_lives: bool
) -> None:
    """A watchdog whose ``__exit__`` a Stop cut short is disarmed all the same: by its own
    thread once the thread that armed it is dead, or by the next watchdog that thread arms.
    It was armed for good, its pool refusing every later request of the client."""
    server = _serve(http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Ok))
    client = httpx.Client(trust_env=False)
    url = f"http://127.0.0.1:{server.server_port}/"
    stops: list[BaseException] = []
    args = (client, url, functools.partial(_stop_as_it_calls, caller, callee), stops)
    try:
        make_pool_abortable(client)
        if owner_lives:
            _cut_request_with_a_stopped_exit(*args)
        else:
            owner = threading.Thread(target=_cut_request_with_a_stopped_exit, args=args)
            owner.start()
            owner.join(30)
        assert stops, "the Stop never landed in the watchdog's exit"
        stale = [thread for thread in threading.enumerate() if thread.name == _STOPPED_EXIT]
        if owner_lives:
            with RequestAbortWatchdog(client, 30):
                assert client.get(url).text == "ok"
        for thread in stale:
            thread.join(10)
        assert not any(thread.is_alive() for thread in stale), "the watchdog is armed for good"
        assert client.get(url).text == "ok"
    finally:
        client.close()
        server.shutdown()
        server.server_close()


def _stop_once_upgraded(frame: FrameType, event: str, arg: Any) -> Any:
    """Trace hook: a Stop lands on the first line ``start_tls`` runs holding the upgraded socket."""
    if frame.f_code is not _AbortableStream.start_tls.__code__:
        return None
    if event == "line" and isinstance(frame.f_locals.get("tls"), ssl.SSLSocket):
        sys.settrace(None)
        raise KeyboardInterrupt
    return _stop_once_upgraded


@pytest.mark.parametrize(
    ("install", "hook"),
    [
        pytest.param(
            sys.setprofile,
            functools.partial(
                _stop_as_it_calls, _AbortableStream.start_tls, AbortableNetworkBackend.track
            ),
            id="at-the-tracking",
        ),
        pytest.param(
            sys.setprofile,
            functools.partial(
                _stop_as_it_calls, _AbortableStream.start_tls, ssl.SSLSocket.do_handshake
            ),
            id="at-the-handshake",
        ),
        pytest.param(sys.settrace, _stop_once_upgraded, id="once-upgraded"),
    ],
)
def test_a_stop_during_the_tls_upgrade_closes_the_upgraded_socket(install: Any, hook: Any) -> None:
    """The upgrade detaches the plain socket, so only ``start_tls`` can close the new one:
    a Stop injected there, from the first line that holds it on, closes it, though the
    caller keeps the exception and its frames."""
    with socket.create_server(("127.0.0.1", 0)) as listener:
        port = listener.getsockname()[1]
        stream = AbortableNetworkBackend(SyncBackend()).connect_tcp("127.0.0.1", port)
        peer, _ = listener.accept()
        with peer:
            peer.settimeout(10)
            tracer = sys.gettrace()
            install(hook)
            try:
                with pytest.raises(KeyboardInterrupt) as stopped:
                    stream.start_tls(ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT), "localhost")
            finally:
                sys.setprofile(None)
                sys.settrace(tracer)
            assert peer.recv(1) == b"", "the upgraded socket is still open"
            assert stopped.tb is not None  # held until here: the leak lasted as long as it


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
    the proxy mount's pool, not the default transport's.  The TCP
    connect to the proxy is held at its entry while the client is
    aborted; the request fails, the connection is closed as it comes up
    and the request never reaches the proxy (here an ordinary server,
    which answers the absolute-URI request a forward proxy receives).
    A ``None`` mount (what a ``NO_PROXY`` pattern maps to) is skipped.
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
        to_the_proxy = functools.partial(_to_port, proxy.server_port)
        with Pause(SyncBackend.connect_tcp.__code__, to_the_proxy) as held:
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
