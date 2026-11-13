# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A machine link resends a command on a fresh socket only when it cannot have arrived.

``MachineLink.request`` keeps its socket between requests and, when the
kept socket turns out closed, reconnects and sends the command again.
It did so for a close seen at any point, including while waiting for
the answer, after the command had gone out: the machine may already have
run it, and a non-idempotent command (a paste, a cherry-pick, an
auto-commit) ran twice.  Now only a close seen before the command went
out is retried; a close during the wait fails the request.

A real ``wss://`` daemon stand-in (:func:`kiss.tests.local_ws.fake_daemon`)
records every command it receives; nothing is mocked.

The same stand-in, given a handler of the test's own, stages the link's
races (audit A09): a retirement during the handshake, a socket abandoned
under a request after the host slept, two stream handshakes of which the
later one succeeds.  It also plays the peer whose answer to the ``auth``
frame is none a daemon gives, and the daemon that hangs up on a
connection once it is in.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from kiss.agents.sorcar import local_endpoint
from kiss.agents.sorcar.machines import Machine, MachineError, MachineLink
from kiss.tests.local_ws import fake_daemon
from kiss.tests.server.pause import Pause


class _Daemon:
    """A stand-in daemon on its own loop thread, keeping its files in *tmp*.

    Answers ``probe`` with ``probeReply`` and closes the connection after
    the answer when ``close_after_answer`` is set; records every command
    and closes the connection without answering the first ``paste``.
    """

    def __init__(
        self, tmp: Path,
        handler: Callable[[ServerConnection], Awaitable[None]] | None = None,
        *, answer_auth: bool = True,
    ) -> None:
        """*handler* (default: the one above) serves each connection; the stand-in
        answers the ``auth`` frame first unless *answer_auth* is false."""
        self.tmp = tmp
        self.ca_file = tmp / "ca.pem"
        self.received: list[str] = []
        self.close_after_answer = False
        self._handler = handler or self._handle
        self._answer_auth = answer_auth
        self._loop = asyncio.new_event_loop()
        self._closed = asyncio.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._loop.run_until_complete, args=(self._serve(),), daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(10):
            self.close()
            raise AssertionError("the stand-in daemon never came up")

    async def _serve(self) -> None:
        async with fake_daemon(
            self.tmp, self._handler, answer_auth=self._answer_auth,
        ) as endpoint:
            served = local_endpoint.read_endpoint(endpoint)
            assert served is not None
            self.url = served.url
            self._ready.set()
            await self._closed.wait()

    async def _handle(self, ws: ServerConnection) -> None:
        try:
            async for message in ws:
                command: dict[str, Any] = json.loads(message)
                self.received.append(command["type"])
                if command["type"] == "paste" and self.received.count("paste") == 1:
                    return  # the machine has the command; the connection dies
                await ws.send(json.dumps({
                    "type": command["type"] + "Reply", "generation": command["generation"],
                }))
                if self.close_after_answer:
                    return
        except ConnectionClosed:
            return

    def close(self) -> None:
        """Stop serving, wait for the loop thread to end and close its loop (once)."""
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._closed.set)
        self._thread.join(10)
        assert not self._thread.is_alive(), "the stand-in daemon never stopped"
        self._loop.close()


def _link(daemon: _Daemon) -> MachineLink:
    port = daemon.url.split(":")[2].split("/")[0]
    return MachineLink(
        Machine(name="m", url=f"https://127.0.0.1:{port}", password="pw"),
        cert_file=daemon.ca_file,
    )


def _wait_closed(link: MachineLink) -> None:
    """Wait until the link's kept socket has processed the daemon's close."""
    deadline = time.monotonic() + 10
    while link._ws is not None and link._ws.state is not State.CLOSED:
        assert time.monotonic() < deadline, "the client never saw the close"
        time.sleep(0.01)


def test_a_close_while_waiting_for_the_answer_does_not_send_the_command_again(
    tmp_path: Path,
) -> None:
    daemon = _Daemon(tmp_path)
    link = _link(daemon)
    try:
        assert link.request({"type": "probe"}, "probeReply", timeout=10)["type"] == "probeReply"
        with pytest.raises(MachineError, match="did not answer"):
            link.request({"type": "paste"}, "pasteReply", timeout=10)
    finally:
        link.close()
        daemon.close()
    assert daemon.received == ["probe", "paste"]


def test_a_kept_socket_found_closed_before_sending_is_replaced_and_the_command_sent_once(
    tmp_path: Path,
) -> None:
    daemon = _Daemon(tmp_path)
    daemon.close_after_answer = True
    link = _link(daemon)
    try:
        assert link.request({"type": "probe"}, "probeReply", timeout=10)["type"] == "probeReply"
        _wait_closed(link)
        assert link.request({"type": "probe"}, "probeReply", timeout=10)["type"] == "probeReply"
    finally:
        link.close()
        daemon.close()
    assert daemon.received == ["probe", "probe"]


def _see_the_close_first(entered: dict[str, Any]) -> bool:
    """At the entry of ``MachineLink._exchange``: wait for the daemon's close; hold nobody."""
    _wait_closed(entered["self"])
    return False


def test_a_socket_found_closed_again_or_on_its_first_use_fails_the_request_unsent(
    tmp_path: Path,
) -> None:
    """The daemon answers its first connection only and hangs up on the others once they are in.

    A kept socket found closed is replaced once: the replacement found
    closed too fails the request, and so does a link's first socket.
    """
    served: list[ServerConnection] = []

    async def answer_the_first_connection_only(ws: ServerConnection) -> None:
        served.append(ws)
        if len(served) == 1:
            await daemon._handle(ws)

    daemon = _Daemon(tmp_path, answer_the_first_connection_only)
    daemon.close_after_answer = True
    link = _link(daemon)
    try:
        assert link.request({"type": "probe"}, "probeReply", timeout=10)["type"] == "probeReply"
        with Pause(MachineLink._exchange.__code__, _see_the_close_first):
            with pytest.raises(MachineError, match="did not answer"):
                link.request({"type": "probe"}, "probeReply", timeout=10)
            with pytest.raises(MachineError, match="did not answer"):
                _link(daemon).request({"type": "probe"}, "probeReply", timeout=10)
    finally:
        link.close()
        daemon.close()
    assert daemon.received == ["probe"]
    assert len(served) == 3


def _wait_until(condition: Callable[[], bool], what: str) -> None:
    """Poll *condition* for up to 10 s; fail naming *what* never happened."""
    deadline = time.monotonic() + 10
    while not condition():
        assert time.monotonic() < deadline, what
        time.sleep(0.01)


def _request_collecting(link: MachineLink, outcome: list[object]) -> None:
    """Thread body: one ``probe`` request on *link*, its reply or error appended to *outcome*."""
    try:
        outcome.append(link.request({"type": "probe"}, "probeReply", timeout=10))
    except MachineError as exc:
        outcome.append(exc)


def test_a_retirement_during_the_handshake_refuses_the_request_before_it_is_sent(
    tmp_path: Path,
) -> None:
    """``retire()`` while the request's handshake runs (its socket not yet
    published, so the retirement's close cannot reach it) must stop the
    request at the end of the handshake: the command used to go out and
    the retirement waited behind its whole exchange (audit A09)."""
    auth_received = threading.Event()
    retire_started = threading.Event()
    received: list[str] = []

    async def hold_auth_until_retired(ws: ServerConnection) -> None:
        await ws.recv()  # the auth frame
        auth_received.set()
        await asyncio.get_running_loop().run_in_executor(None, retire_started.wait, 10)
        await ws.send(json.dumps({"type": "auth_ok"}))
        try:
            command: dict[str, Any] = json.loads(await ws.recv())
            received.append(command["type"])
            await ws.send(json.dumps({"type": "probeReply", "generation": command["generation"]}))
        except ConnectionClosed:
            return

    daemon = _Daemon(tmp_path, hold_auth_until_retired, answer_auth=False)
    link = _link(daemon)
    outcome: list[object] = []
    requester = threading.Thread(target=_request_collecting, args=(link, outcome))
    retirer = threading.Thread(target=link.retire)
    try:
        requester.start()
        assert auth_received.wait(10), "the request never sent its auth frame"
        retirer.start()
        _wait_until(lambda: link._retired, "the retirement never started")
        retire_started.set()
        requester.join(30)
        retirer.join(30)
        assert not requester.is_alive() and not retirer.is_alive()
    finally:
        retire_started.set()
        daemon.close()
    assert received == [], "the command went out on a link retired during its handshake"
    assert isinstance(outcome[0], MachineError) and "registry changed" in str(outcome[0])
    assert link._ws is None


def test_a_request_woken_by_abandon_does_not_back_the_machine_off(tmp_path: Path) -> None:
    """``abandon()`` (the host back from sleep) shuts the socket down under a
    request waiting for its answer: the request fails, but as a sleep of
    the host, not as a failure of the machine, so the back-off ``abandon``
    just cleared is not armed again for every lane (audit A09)."""
    command_received = threading.Event()

    async def never_answer(ws: ServerConnection) -> None:
        try:
            await ws.recv()  # the command; the host sleeps before the answer
            command_received.set()
            await ws.wait_closed()
        except ConnectionClosed:
            return

    daemon = _Daemon(tmp_path, never_answer)
    link = _link(daemon)
    outcome: list[object] = []
    requester = threading.Thread(target=_request_collecting, args=(link, outcome))
    try:
        requester.start()
        assert command_received.wait(10), "the command never reached the daemon"
        link.abandon()
        requester.join(30)
        assert not requester.is_alive()
        assert isinstance(outcome[0], MachineError) and "slept" in str(outcome[0])
        assert (link._health.failures, link._health.retry_at) == (0, 0.0), (
            "the request woken by abandon() backed the machine off"
        )
        link._check_available()
    finally:
        link.close()
        daemon.close()


def test_a_handshake_that_succeeds_after_another_failed_clears_the_back_off(
    tmp_path: Path,
) -> None:
    """Two stream handshakes share the machine's back-off record: when the
    later one succeeds after the other failed, the machine is reachable
    and the failure's back-off must go, not just its failure count
    (audit A09)."""
    first_auth_received = threading.Event()
    other_failed = threading.Event()
    connections = 0

    async def answer_first_after_the_other_fails(ws: ServerConnection) -> None:
        nonlocal connections
        connections += 1
        await ws.recv()  # the auth frame
        if connections != 1:
            return  # the later handshake: closed without an answer
        first_auth_received.set()
        await asyncio.get_running_loop().run_in_executor(None, other_failed.wait, 10)
        await ws.send(json.dumps({"type": "auth_ok"}))
        try:
            await ws.wait_closed()
        except ConnectionClosed:
            return

    daemon = _Daemon(tmp_path, answer_first_after_the_other_fails, answer_auth=False)
    link = _link(daemon)
    streams: list[Any] = []
    first = threading.Thread(target=lambda: streams.append(link.open_connection(timeout=10)))
    try:
        first.start()
        assert first_auth_received.wait(10), "the first handshake never sent its auth frame"
        with pytest.raises(MachineError, match="did not complete the handshake"):
            link.open_connection(timeout=10)
        assert link._health.retry_at > time.monotonic()
        other_failed.set()
        first.join(30)
        assert streams, "the first handshake did not complete"
        link._check_available()
        assert link._health.retry_at == 0.0, "a successful handshake left the back-off"
    finally:
        other_failed.set()
        for stream in streams:
            stream.close()
        daemon.close()


def test_an_unexpected_answer_to_the_auth_frame_refuses_the_link_and_hangs_up(
    tmp_path: Path,
) -> None:
    """A peer that answers the ``auth`` frame with another frame is no daemon to link to.

    The dial closes the socket it opened, which the peer sees: nothing
    of a refused handshake stays open.
    """
    hung_up = threading.Event()

    async def answer_busy(ws: ServerConnection) -> None:
        await ws.recv()  # the auth frame
        await ws.send(json.dumps({"type": "busy"}))
        await ws.wait_closed()
        hung_up.set()

    daemon = _Daemon(tmp_path, answer_busy, answer_auth=False)
    link = _link(daemon)
    try:
        with pytest.raises(MachineError, match=r"refused the connection: \{'type': 'busy'\}"):
            link.request({"type": "ping"}, "pong")
        assert hung_up.wait(10), "the refused socket stayed open"
        assert link._ws is None
        assert link._health.retry_at > time.monotonic()
    finally:
        daemon.close()
