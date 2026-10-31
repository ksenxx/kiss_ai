# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A daemon frame over the client's cap is named whether or not the run was sent.

``daemon_client.run`` opens its connection with ``max_size=_MAX_LINE_BYTES``;
a larger frame makes ``websockets`` close the connection with code 1009,
which the client reports as "an event frame larger than the limit".  The
first fix wave gave the send of the ``run`` command its own error (the run
was never sent, so no task runs: a plain ``ConnectionError``, not the
may-still-be-running ``ConnectionLostError``) and lost the 1009 case on the
way: a too-big frame that arrived before the send read "closed the
connection before the task was sent".  ``test_dispatch_timeout.py::
test_oversize_frame_rejected``, whose stand-in sends the frame without
waiting for the run, failed whenever the frame won that race.  Both orders
are landed for sure here, against real local-WSS daemon stand-ins; nothing
is mocked.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from kiss.agents.sorcar import daemon_client, local_endpoint
from kiss.tests.agents.sorcar.test_dispatch_timeout import _LocalDaemon, _OversizeFrameDaemon
from kiss.tests.server.pause import Pause

_TOO_BIG = "The sorcar daemon sent an event frame larger than the 1024-byte client limit"


class _OversizeAnswerDaemon(_LocalDaemon):
    """A daemon stand-in that reads the ``run`` and answers it with one oversized frame."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        super().__init__("kiss_oversize_answer_")

    async def _handle(self, ws: ServerConnection) -> None:
        if await self._read_run(ws) is None:
            return
        try:
            await ws.send(self._payload.decode("ascii"))
        except ConnectionClosed:
            return
        await self._closed.wait()


def _send_once_closed(frame_locals: dict[str, Any]) -> bool:
    """Pause matcher at ``local_endpoint.send``: the client closes the connection first.

    The client's receiving thread closes the connection with code 1009
    once the stand-in's oversized frame arrives (after the ``auth``
    handshake, whose send passes); the ``run`` is then sent on the closed
    connection.  Returns False: nothing more is held.
    """
    if json.loads(frame_locals["text"]).get("type") != "run":
        return False
    ws = frame_locals["ws"]
    deadline = time.monotonic() + 10
    while ws.state is not State.CLOSED:
        assert time.monotonic() < deadline, "the client never closed the connection"
        time.sleep(0.01)
    return False


def test_a_too_big_frame_before_the_run_is_sent_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daemon_client, "_MAX_LINE_BYTES", 1024)
    daemon = _OversizeFrameDaemon(b"a" * 1500)
    try:
        with Pause(local_endpoint.send.__code__, _send_once_closed):
            with pytest.raises(ConnectionError) as caught:
                daemon_client.run(
                    "oversize probe", endpoint_file=daemon.endpoint_file, timeout=30.0,
                )
    finally:
        daemon.close()
    assert str(caught.value) == _TOO_BIG
    # The run was never sent, so no task can still be running.
    assert not isinstance(caught.value, daemon_client.ConnectionLostError)


def test_a_too_big_frame_answering_the_run_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daemon_client, "_MAX_LINE_BYTES", 1024)
    daemon = _OversizeAnswerDaemon(b"a" * 1500)
    try:
        with pytest.raises(daemon_client.ConnectionLostError) as caught:
            daemon_client.run(
                "oversize probe", endpoint_file=daemon.endpoint_file, timeout=30.0,
            )
    finally:
        daemon.close()
    assert str(caught.value) == _TOO_BIG
    assert daemon.run_cmd is not None and daemon.run_cmd["type"] == "run"
