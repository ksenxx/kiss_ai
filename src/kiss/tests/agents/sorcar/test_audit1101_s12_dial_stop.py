"""A Stop during a link's authentication closes the socket nobody owns yet (S12 F3).

``MachineLink._dial`` opens the WebSocket (receive and keepalive
threads included) and then sends the ``auth`` frame and waits for the
daemon's answer.  The cleanup of that wait caught only the connection
errors; the runner's injected ``KeyboardInterrupt`` went past it, and
as ``_dial`` never returned, the link's ``_ws`` stayed ``None`` and
``link.close()`` had nothing to close: the open socket and its
threads outlived the task, answering the daemon's pings.  The socket
is now closed on every unsuccessful exit of the handshake.

``local_endpoint.connect`` (every same-machine client, ``daemon_client``
included) had the same handshake and the same gap: a Stop landing in
its auth wait left an open, authenticated socket nobody could close.
Its handshake now runs on a thread no Stop reaches (audit 2026-11-11,
S9 re-fix): on CPython 3.14 a Stop raised as the caller's own receive
took the frame assembler's mutex left that mutex held, and the close
meant to clean up waited for it for good.

``MachineLink._dial`` runs its handshake that way too (same audit,
S10 re-fix round 2): it had the same receive on the task's thread,
where a Stop also waited for the daemon's answer, the handshake's
whole budget when none came.

A request on the socket a link keeps still receives on the task's
thread (same audit, S10 re-fix round 3).  A Stop that landed in the
request, or in the close of the socket (``close``, ``retire``), left
the socket on the link as it was at that instant: with the frame
assembler's mutex held (CPython 3.14: the next request waited for it
for good), refusing every later receive (3.13: the next request
failed and backed the machine off), or open after a close that never
ran.  And the Stop itself waited for the read's whole budget.  The
link now forgets such a socket and shuts its connection down, and
reads in short slices.  A Stop raised after the dialling thread had
handed the new socket over, and before the link kept it, left the
socket open with nobody to close it: the dial now hangs it up.
"""

from __future__ import annotations

import inspect
import socket
import ssl
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import CodeType, FrameType
from typing import Any

import pytest
from websockets.sync.client import ClientConnection

from kiss.agents.sorcar import local_endpoint
from kiss.agents.sorcar import machines as mach
from kiss.server.task_runner import inject_keyboard_interrupt
from kiss.tests.local_ws import make_test_tls
from kiss.tests.remote_daemon import start_remote_daemon
from kiss.tests.server.pause import Pause

REQUESTER = "s12-requester"
BUDGET = 4.0
"""Seconds a stalled handshake or exchange below may take: long against the time a Stop needs."""


def _parked_in(ident: int, code: CodeType) -> bool:
    """Whether thread *ident* comes to rest in *code* (blocked there, not passing through)."""
    deadline = time.monotonic() + 10
    seen = None
    while time.monotonic() < deadline:
        frame = sys._current_frames().get(ident)
        at = None if frame is None else (frame.f_code, frame.f_lasti)
        if at is not None and at[0] is code and at == seen:
            return True  # the same instruction, two looks apart
        seen = at
        time.sleep(0.05)
    return False


def test_stop_while_waiting_for_auth_ok_closes_the_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Stop lands while the auth receive waits for the frame assembler's mutex.

    The local client's test below says why there.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "local"))
    daemon = start_remote_daemon(tmp_path / "remote", password="pw", env={"HOME": str(tmp_path)})
    sockets: list[ClientConnection] = []
    readers: list[int] = []  # the thread that reads the daemon's answer
    outcome: dict[str, Any] = {}
    try:
        machine = mach.add_machine("box", daemon.url, "pw")
        link = mach.MachineLink(machine)

        def requester() -> None:
            try:
                outcome["reply"] = link.request({"type": "ping"}, "pong")
            except BaseException as exc:
                outcome["error"] = type(exc).__name__
            finally:
                link.close()

        def capture_auth_wait(frame_locals: dict[str, Any]) -> bool:
            sockets.append(frame_locals["ws"])
            readers.append(threading.get_ident())
            return True

        worker = threading.Thread(target=requester, name=REQUESTER, daemon=True)
        # Hold the handshake after the auth frame went out, while the
        # reply is awaited: the socket is open and nobody's yet.
        with Pause(mach.recv_auth_reply.__code__, capture_auth_wait) as held:
            worker.start()
            assert held.paused.wait(30), "the auth wait was not reached"
            ws = sockets[0]
            mutex = ws.recv_messages.mutex
            assert mutex.acquire(timeout=10)
        try:
            assert _parked_in(readers[0], type(ws.recv_messages).get.__code__)
            assert inject_keyboard_interrupt(worker.ident or 0) == 1
        finally:
            mutex.release()
        worker.join(10)
        assert not worker.is_alive(), "the Stop left the requester blocked"
        assert outcome == {"error": "KeyboardInterrupt"}
        assert link._ws is None
        ws.recv_events_thread.join(10)
        assert ws.state.name == "CLOSED"
        assert not ws.recv_events_thread.is_alive()
        assert not mutex.locked()
        # The link re-connects at once afterwards: a Stop is no machine
        # failure, so it started no back-off.
        assert link.request({"type": "ping"}, "pong")["type"] == "pong"
        link.close()
    finally:
        for ws in sockets:
            if ws.recv_messages.mutex.locked():  # the leak: a failing run must end, not hang
                ws.recv_messages.mutex.release()
            ws.close()
        mach.close_links()
        daemon.stop()


def test_stop_during_a_stalled_handshake_frees_the_requester_and_starts_no_back_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Stop lands while the machine's port takes the connection and answers nothing.

    On the task's own thread the handshake kept a stopped task until
    its blocking socket call returned, the handshake's whole budget
    later.  The dial the task gave up on fails when that budget ends,
    with nobody waiting for the machine: the link is not backed off.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "local"))
    _cert, _key, ca_file = make_test_tls(tmp_path)
    diallers: list[threading.Thread] = []  # the thread that opens the socket
    dialling = threading.Event()
    outcome: dict[str, Any] = {}
    with socket.create_server(("127.0.0.1", 0)) as hole:  # accepts in the kernel, never answers
        silent = mach.Machine("silent", f"https://127.0.0.1:{hole.getsockname()[1]}")
        link = mach.MachineLink(silent, cert_file=ca_file)

        def requester() -> None:
            try:
                outcome["ws"] = link.open_connection(timeout=BUDGET)
            except BaseException as exc:
                outcome["error"] = type(exc).__name__

        def note_dialler(frame_locals: dict[str, Any]) -> bool:
            diallers.append(threading.current_thread())
            dialling.set()
            return False  # nothing is held: the port stalls the handshake

        worker = threading.Thread(target=requester, name=REQUESTER, daemon=True)
        with Pause(mach._ws_connect.__code__, note_dialler):
            worker.start()
            assert dialling.wait(30), "the dial was not reached"
        dialler = diallers[0]
        assert _parked_in(dialler.ident or 0, ssl.SSLSocket.do_handshake.__code__)
        assert inject_keyboard_interrupt(worker.ident or 0) == 1
        worker.join(BUDGET / 2)
        assert not worker.is_alive(), "the Stop waited for the handshake's budget"
        assert outcome == {"error": "KeyboardInterrupt"}
        assert not link._dialing
        dialler.join(BUDGET + 10)  # the dial given up on ends with its budget
        assert not dialler.is_alive()
        assert link._health == mach.LinkHealth(), "a dial nobody waited for backed the link off"


def _stop_once_counted_in(frame: FrameType, event: str, arg: object) -> None:
    """Profile function: a Stop of its own thread as the link counts its stream handshake in."""
    if event == "return" and frame.f_code is mach.MachineLink._start_dial.__code__:
        sys.setprofile(None)
        inject_keyboard_interrupt(threading.get_ident())


def _open_stream(link: mach.MachineLink, outcome: dict[str, Any], profile: Any) -> None:
    """Thread body: ``open_connection`` on *link* under *profile*; record the error's name."""
    sys.setprofile(profile)
    try:
        outcome["ws"] = link.open_connection(timeout=BUDGET)
    except BaseException as exc:  # noqa: BLE001 - the Stop under test
        outcome["error"] = type(exc).__name__
    finally:
        sys.setprofile(None)


def test_a_stop_as_a_stream_handshake_is_counted_in_holds_no_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Stop raised once the handshake is counted, before the dial, counts it out again.

    The count was taken ahead of the ``try`` that gives it back, and a
    Stop is raised where the dial lock is released, between the two:
    the link's retirement (a registry change) then waited its whole
    bound for a handshake nobody was making (audit 2026-11-12, B4).
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "local"))
    link = mach.MachineLink(mach.Machine("nowhere", "https://127.0.0.1:1"))
    outcome: dict[str, Any] = {}
    worker = threading.Thread(
        target=_open_stream, args=(link, outcome, _stop_once_counted_in), name=REQUESTER,
        daemon=True,
    )
    worker.start()
    worker.join(BUDGET + 10)
    assert outcome == {"error": "KeyboardInterrupt"}
    started = time.monotonic()
    link.retire()
    assert time.monotonic() - started < 5, "the retirement waited for a handshake nobody makes"


def test_stop_while_a_local_client_waits_for_auth_ok_closes_the_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Stop lands while the auth receive waits for the frame assembler's mutex.

    That is where CPython 3.14 raises an injected exception between a
    ``with lock:`` acquire and the block that would release the lock.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "local"))
    daemon = start_remote_daemon(tmp_path / "remote", env={"HOME": str(tmp_path)})
    sockets: list[ClientConnection] = []
    readers: list[int] = []  # the thread that reads the daemon's answer
    outcome: dict[str, Any] = {}
    try:
        endpoint_file = daemon.home / local_endpoint.LOCAL_ENDPOINT_FILE

        def client() -> None:
            try:
                outcome["ws"] = local_endpoint.connect(endpoint_file)
            except BaseException as exc:
                outcome["error"] = type(exc).__name__

        def capture_auth_wait(frame_locals: dict[str, Any]) -> bool:
            sockets.append(frame_locals["ws"])
            readers.append(threading.get_ident())
            return True

        worker = threading.Thread(target=client, name=REQUESTER, daemon=True)
        with Pause(local_endpoint.recv_auth_reply.__code__, capture_auth_wait) as held:
            worker.start()
            assert held.paused.wait(30), "the auth wait was not reached"
            ws = sockets[0]
            mutex = ws.recv_messages.mutex
            assert mutex.acquire(timeout=10)
        try:
            assert _parked_in(readers[0], type(ws.recv_messages).get.__code__)
            assert inject_keyboard_interrupt(worker.ident or 0) == 1
        finally:
            mutex.release()
        worker.join(10)
        assert not worker.is_alive(), "the Stop left the client blocked"
        assert outcome == {"error": "KeyboardInterrupt"}
        ws.recv_events_thread.join(10)
        assert ws.state.name == "CLOSED"
        assert not ws.recv_events_thread.is_alive()
        assert not mutex.locked()
    finally:
        for ws in sockets:
            if ws.recv_messages.mutex.locked():  # the leak: a failing run must end, not hang
                ws.recv_messages.mutex.release()
            ws.close()
        daemon.stop()


def _release_held(ws: ClientConnection) -> None:
    """Release the mutexes of *ws* that a Stop left held (CPython 3.14, see the test above)."""
    for mutex in (ws.recv_messages.mutex, ws.protocol_mutex):
        if mutex.locked():
            mutex.release()


@pytest.fixture
def kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[mach.MachineLink, ClientConnection]]:
    """A link a real daemon has answered once, and the socket it keeps for its next request."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "local"))
    daemon = start_remote_daemon(tmp_path / "remote", password="pw", env={"HOME": str(tmp_path)})
    try:
        link = mach.MachineLink(mach.add_machine("box", daemon.url, "pw"))
        assert link.request({"type": "ping"}, "pong")["type"] == "pong"
        ws = link._ws
        assert ws is not None
        try:
            yield link, ws
        finally:
            _release_held(ws)  # a failing run must end, not hang
            ws.close()
            link.close()
    finally:
        mach.close_links()
        daemon.stop()


def _ping(link: mach.MachineLink, reply_type: str, timeout: float, outcome: dict[str, Any]) -> None:
    """Thread body: ping over *link* and wait *timeout* seconds for a *reply_type* answer."""
    try:
        outcome["reply"] = link.request({"type": "ping"}, reply_type, timeout=timeout)["type"]
    except BaseException as exc:
        outcome["error"] = type(exc).__name__


def _run(call: Callable[[], None], outcome: dict[str, Any]) -> None:
    """Thread body: *call*; the error that ended it is named in *outcome*."""
    try:
        call()
    except BaseException as exc:
        outcome["error"] = type(exc).__name__


def _stop_as_it_takes(mutex: threading.Lock, code: CodeType, worker: threading.Thread) -> None:
    """Run *worker* until it rests in *code* waiting for *mutex*, stop it there and let it end."""
    assert mutex.acquire(timeout=10)
    try:
        worker.start()
        assert _parked_in(worker.ident or 0, code)
        assert inject_keyboard_interrupt(worker.ident or 0) == 1
    finally:
        mutex.release()
    worker.join(10)
    assert not worker.is_alive(), "the Stop left the thread blocked"


def _assert_the_stop_cost_the_link_that_socket_only(
    link: mach.MachineLink, ws: ClientConnection, then: str = "pong",
) -> None:
    """*link* forgot *ws*, shut it down and backed nothing off; its next ping ends in *then*."""
    assert link._ws is None, "the link kept the socket the Stop cut short"
    assert link._health == mach.LinkHealth(), "a Stop is no failure of the machine"
    ws.recv_events_thread.join(1)
    if ws.recv_events_thread.is_alive():
        # The Stop left a mutex of the socket held, and its reader waits for
        # it: released, the reader shows whether its connection was shut down.
        _release_held(ws)
        ws.recv_events_thread.join(10)
    assert not ws.recv_events_thread.is_alive(), "the socket the link forgot stayed connected"
    later: dict[str, Any] = {}
    after = threading.Thread(target=_ping, args=(link, "pong", 5.0, later), daemon=True)
    after.start()
    after.join(10)
    assert list(later.values()) == [then], "the link's next request went wrong"


def test_stop_as_a_request_takes_the_frame_mutex_costs_the_link_that_socket_only(
    kept: tuple[mach.MachineLink, ClientConnection],
) -> None:
    """The Stop lands while the request's first receive waits for the frame assembler's mutex.

    It leaves the socket's own receive (``websockets``' ``Assembler.get``)
    between two of its steps: on CPython 3.14 with the mutex held, on
    3.13 with the mark of a receive in progress set.
    """
    link, ws = kept
    stopped: dict[str, Any] = {}
    worker = threading.Thread(
        target=_ping, args=(link, "pong", 5.0, stopped), name=REQUESTER, daemon=True,
    )
    _stop_as_it_takes(ws.recv_messages.mutex, type(ws.recv_messages).get.__code__, worker)
    assert stopped == {"error": "KeyboardInterrupt"}
    _assert_the_stop_cost_the_link_that_socket_only(link, ws)


def test_stop_during_a_stalled_exchange_frees_the_requester_and_costs_the_link_that_socket_only(
    kept: tuple[mach.MachineLink, ClientConnection],
) -> None:
    """The Stop lands while the request waits for an answer that does not come.

    The socket goes with the request: the answer may still arrive,
    and would be read by the link's next request.
    """
    link, ws = kept
    stopped: dict[str, Any] = {}
    worker = threading.Thread(
        target=_ping, args=(link, "never", BUDGET, stopped), name=REQUESTER, daemon=True,
    )
    worker.start()
    assert _parked_in(worker.ident or 0, type(ws.recv_messages).get_next_frame.__code__)
    assert inject_keyboard_interrupt(worker.ident or 0) == 1
    worker.join(BUDGET / 2)
    assert not worker.is_alive(), "the Stop waited for the request's budget"
    assert stopped == {"error": "KeyboardInterrupt"}
    _assert_the_stop_cost_the_link_that_socket_only(link, ws)


class _StopOnceDialled:
    """A trace function: a Stop of its own thread at ``_dial``'s first line that has the socket."""

    def __init__(self) -> None:
        self.sockets: list[ClientConnection] = []

    def __call__(self, frame: FrameType, event: str, arg: object) -> _StopOnceDialled:
        """Inject the Stop once: on the first line of ``_dial`` run with ``ws`` assigned."""
        if (
            event == "line" and frame.f_code is mach.MachineLink._dial.__code__
            and "ws" in frame.f_locals and not self.sockets
        ):
            self.sockets.append(frame.f_locals["ws"])
            inject_keyboard_interrupt(threading.get_ident())
        return self


def _ping_traced(link: mach.MachineLink, trace: _StopOnceDialled, outcome: dict[str, Any]) -> None:
    """Thread body: :func:`_ping` for a ``pong`` with *trace* following the thread."""
    sys.settrace(trace)
    try:
        _ping(link, "pong", 5.0, outcome)
    finally:
        sys.settrace(None)


def test_stop_between_the_dial_and_the_link_keeping_its_socket_hangs_the_socket_up(
    kept: tuple[mach.MachineLink, ClientConnection],
) -> None:
    """The Stop lands in ``_dial`` once the dialling thread has handed the new socket over.

    The socket was the requester's alone then, in a local variable: the
    Stop unwound the requester and the socket stayed open, its threads
    answering the daemon's pings, with ``link.close()`` knowing nothing
    of it.
    """
    link, _ = kept
    link.close()  # the next request dials
    trace = _StopOnceDialled()
    stopped: dict[str, Any] = {}
    worker = threading.Thread(
        target=_ping_traced, args=(link, trace, stopped), name=REQUESTER, daemon=True,
    )
    worker.start()
    worker.join(20)
    assert not worker.is_alive(), "the Stop left the requester blocked"
    assert stopped == {"error": "KeyboardInterrupt"}
    try:
        _assert_the_stop_cost_the_link_that_socket_only(link, trace.sockets[0], "pong")
    finally:
        trace.sockets[0].close()  # a failing run leaves it open


@pytest.mark.parametrize(("end", "then"), [("close", "pong"), ("retire", "MachineError")])
def test_stop_as_the_link_closes_its_socket_costs_the_link_that_socket_only(
    kept: tuple[mach.MachineLink, ClientConnection], end: str, then: str,
) -> None:
    """The Stop lands while the socket's close waits for the connection's protocol mutex.

    The closing handshake has not begun: the socket is open, and on
    CPython 3.14 its protocol mutex stays held.  A link closed for
    good (``retire``, a registry change) refuses its next request.
    """
    link, ws = kept
    stopped: dict[str, Any] = {}
    worker = threading.Thread(
        target=_run, args=(getattr(link, end), stopped), name=REQUESTER, daemon=True,
    )
    _stop_as_it_takes(ws.protocol_mutex, inspect.unwrap(type(ws).send_context).__code__, worker)
    assert stopped == {"error": "KeyboardInterrupt"}
    _assert_the_stop_cost_the_link_that_socket_only(link, ws, then)
