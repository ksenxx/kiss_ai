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
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from kiss.agents.sorcar import local_endpoint
from kiss.agents.sorcar.machines import Machine, MachineError, MachineLink
from kiss.tests.local_ws import fake_daemon


class _Daemon:
    """A stand-in daemon on its own loop thread, keeping its files in *tmp*.

    Answers ``probe`` with ``probeReply`` and closes the connection after
    the answer when ``close_after_answer`` is set; records every command
    and closes the connection without answering the first ``paste``.
    """

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.ca_file = tmp / "ca.pem"
        self.received: list[str] = []
        self.close_after_answer = False
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
        async with fake_daemon(self.tmp, self._handle) as endpoint:
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
        """Stop serving, wait for the loop thread to end and close its loop."""
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
