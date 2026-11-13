# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sweep 2026-10-30 (S11): ``local_endpoint`` contracts its docstrings now state.

* ``connect(max_size=None)`` was documented as "the ``websockets``
  default" (1 MiB), but ``None`` is passed through and lifts the limit:
  the docstring now says so, and this pins it (the daemon client relies on
  a cap it passes explicitly).
* ``write_endpoint`` no longer creates the parent directory itself: the
  endpoint lock and the atomic write both do, which this pins.
* ``connect`` left the proxy lookup of ``websockets`` on: with
  ``http_proxy`` / ``https_proxy`` set and no ``no_proxy`` for loopback,
  the connection to the daemon on this machine went to the proxy and the
  client reported the daemon down (audit 2026-11-11, S9).
* ``connect`` dialled on the caller's thread, the one a Stop interrupts:
  the Stop waited for the daemon's answer and then left the socket open
  (audit 2026-11-11, S9 re-fix).
* A Stop raised once the dialling thread had been told its connection
  was taken, and before ``connect`` returned it, left the connection
  open with nobody to close it (audit 2026-11-12, B4).

A real ``websockets`` server on loopback plays the daemon's side of the
local ``auth`` handshake.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, serve

from kiss.agents.sorcar import local_endpoint
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.conftest import IS_WINDOWS

_BIG_FRAME = "x" * (2 * 1024 * 1024)  # twice the websockets default limit
_TOKEN = "t" * 64


def _daemon(ws: ServerConnection) -> None:
    """Accept the local token, then send one frame larger than 1 MiB."""
    auth = json.loads(ws.recv(timeout=5))
    ws.send(json.dumps({"type": "auth_ok", "local": auth.get("token") == _TOKEN}))
    ws.send(_BIG_FRAME)
    try:
        ws.recv(timeout=5)  # hold the connection until the client is done
    except Exception:  # noqa: BLE001 - the client closed it
        pass


@contextmanager
def _serving(
    daemon: Callable[[ServerConnection], None], tmp_path: Path, **options: Any,
) -> Iterator[Path]:
    """Serve *daemon* on loopback; yield its endpoint file, in a directory not there yet."""
    with serve(daemon, "127.0.0.1", 0, max_size=None, **options) as server:
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        port = server.socket.getsockname()[1]
        path = tmp_path / "not" / "yet" / "sorcar-local.json"
        local_endpoint.write_endpoint(
            path,
            local_endpoint.LocalEndpoint(
                url=f"ws://127.0.0.1:{port}", token=_TOKEN, ca=None, pid=1,
            ),
        )
        yield path
    server_thread.join(5)


@pytest.fixture
def endpoint_file(tmp_path: Path) -> Iterator[Path]:
    """The endpoint file of a daemon that accepts the local token and sends a big frame."""
    with _serving(_daemon, tmp_path) as path:
        yield path


def test_write_endpoint_creates_the_directory_with_a_private_file(endpoint_file: Path) -> None:
    """The file lands in a fresh directory, mode 0600 (POSIX modes only), and reads back."""
    assert IS_WINDOWS or endpoint_file.stat().st_mode & 0o777 == 0o600
    endpoint = local_endpoint.read_endpoint(endpoint_file)
    assert endpoint is not None and endpoint.token == _TOKEN


def test_connect_without_max_size_accepts_any_frame_size(endpoint_file: Path) -> None:
    """``max_size=None`` (the default) receives a 2 MiB frame."""
    ws = local_endpoint.connect(endpoint_file, open_timeout=5.0)
    try:
        assert ws.recv(timeout=5) == _BIG_FRAME
    finally:
        ws.close()


def test_connect_with_max_size_caps_the_frame(endpoint_file: Path) -> None:
    """An explicit 1 MiB cap refuses the same frame (close code 1009)."""
    ws = local_endpoint.connect(endpoint_file, open_timeout=5.0, max_size=1024 * 1024)
    try:
        with pytest.raises(ConnectionClosed) as closed:
            ws.recv(timeout=5)
        assert closed.value.rcvd is None or closed.value.rcvd.code == 1009
        assert closed.value.sent is not None and closed.value.sent.code == 1009
    finally:
        ws.close()


@pytest.mark.parametrize("variable", ["http_proxy", "https_proxy"])
def test_connect_never_dials_the_environments_proxy(
    endpoint_file: Path, monkeypatch: pytest.MonkeyPatch, variable: str,
) -> None:
    """The daemon is on this machine: a proxy the environment names is not asked for it.

    Before: the proxy (here a loopback port nothing listens on) was
    dialled instead and ``connect`` raised "Cannot connect to the
    sorcar daemon ... start it with `kiss-web`" at a running daemon.
    """
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        monkeypatch.setenv(variable, f"http://127.0.0.1:{unused.getsockname()[1]}")
    for no_proxy in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(no_proxy, raising=False)
    ws = local_endpoint.connect(endpoint_file, open_timeout=5.0)
    try:
        assert ws.recv(timeout=5) == _BIG_FRAME
    finally:
        ws.close()


def test_a_stop_during_a_stalled_handshake_frees_the_caller_and_closes_the_late_connection(
    tmp_path: Path,
) -> None:
    """A Stop (the runner's injected ``KeyboardInterrupt``) ends ``connect`` at a stalled daemon.

    The dial runs on a thread no Stop reaches: the caller leaves at
    once, and the connection that dial opens afterwards is closed.

    Before: the dial ran on the caller's thread.  The Stop was raised
    only when the blocking wait for the daemon's answer returned (up to
    ``open_timeout`` later), inside ``websockets``' ``connect``, whose
    cleanup catches ``Exception`` only: the socket stayed open, nobody's.
    """
    reached = threading.Event()  # the opening handshake's request is with the daemon
    answer = threading.Event()  # the daemon may answer it
    daemon = _ClosedWitness()
    outcome: dict[str, Any] = {}

    def stall(connection: ServerConnection, request: object) -> None:
        reached.set()
        answer.wait(60)

    with _serving(daemon, tmp_path, process_request=stall) as path:
        worker = threading.Thread(target=_connect, args=(path, outcome), daemon=True)
        worker.start()
        try:
            assert reached.wait(10), "the opening handshake never reached the daemon"
            assert inject_async_exception(worker.ident or 0, KeyboardInterrupt) == 1
            worker.join(5)
            freed = not worker.is_alive()
        finally:
            answer.set()
        assert freed, "the Stop waited for the daemon's answer"
        assert outcome == {"error": "KeyboardInterrupt"}
        assert daemon.closed.wait(10), "the connection opened after the Stop was left open"


class _ClosedWitness:
    """A daemon that answers the ``auth`` and sets ``closed`` once the client hangs up."""

    def __init__(self) -> None:
        self.closed = threading.Event()

    def __call__(self, ws: ServerConnection) -> None:
        """Serve one connection."""
        try:
            ws.recv(timeout=10)  # the auth frame
            ws.send(json.dumps({"type": "auth_ok", "local": True}))
            ws.recv(timeout=10)
        except ConnectionClosed:
            self.closed.set()
        except TimeoutError:  # a client that left the socket open says nothing
            pass


def _connect(path: Path, outcome: dict[str, Any], profile: Any = None) -> None:
    """Thread body: ``connect`` to *path* under *profile*; record the socket or the error's name."""
    sys.setprofile(profile)
    try:
        outcome["ws"] = local_endpoint.connect(path, open_timeout=30.0)
    except BaseException as exc:  # noqa: BLE001 - the Stop under test
        outcome["error"] = type(exc).__name__
    finally:
        sys.setprofile(None)


def _stop_once_taken(frame: FrameType, event: str, arg: object) -> None:
    """Profile function: a Stop of its own thread as ``dial_off_thread``'s first ``put`` returns.

    That ``put`` tells the dialling thread its connection is taken; an
    injected exception is raised at the check that follows a call.
    """
    if (
        event == "c_return" and getattr(arg, "__name__", "") == "put"
        and frame.f_code is local_endpoint.dial_off_thread.__code__
    ):
        sys.setprofile(None)
        inject_async_exception(threading.get_ident(), KeyboardInterrupt)


def test_a_stop_as_the_caller_takes_the_dialled_connection_closes_it(tmp_path: Path) -> None:
    """A Stop raised once the dialling thread was told its connection is taken closes it.

    Before: the dialling thread, told so, left the connection to the
    caller, and the caller, unwound before ``connect`` returned, never
    had it: the socket stayed open, its threads answering the daemon's
    pings for as long as the process lived.
    """
    daemon = _ClosedWitness()
    outcome: dict[str, Any] = {}
    with _serving(daemon, tmp_path) as path:
        worker = threading.Thread(
            target=_connect, args=(path, outcome, _stop_once_taken), daemon=True,
        )
        worker.start()
        worker.join(20)
        try:
            assert not worker.is_alive(), "the Stop left the caller blocked"
            assert outcome == {"error": "KeyboardInterrupt"}
            assert daemon.closed.wait(10), "the connection the caller had taken was left open"
        finally:
            if "ws" in outcome:  # the Stop did not land: a failing run leaves it open
                outcome["ws"].close()
