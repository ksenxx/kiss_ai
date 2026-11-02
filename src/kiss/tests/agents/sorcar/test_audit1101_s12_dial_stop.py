"""A Stop during a link's authentication closes the socket nobody owns yet (S12 F3).

``MachineLink._dial`` opens the WebSocket (receive and keepalive
threads included) and then sends the ``auth`` frame and waits for the
daemon's answer.  The cleanup of that wait caught only the connection
errors; the runner's injected ``KeyboardInterrupt`` went past it, and
as ``_dial`` never returned, the link's ``_ws`` stayed ``None`` and
``link.close()`` had nothing to close: the open socket and its
threads outlived the task, answering the daemon's pings.  The socket
is now closed on every unsuccessful exit of the handshake.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from websockets.sync.client import ClientConnection

from kiss.agents.sorcar import machines as mach
from kiss.server.task_runner import inject_keyboard_interrupt
from kiss.tests.remote_daemon import start_remote_daemon
from kiss.tests.server.pause import Pause

REQUESTER = "s12-requester"


def test_stop_while_waiting_for_auth_ok_closes_the_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "local"))
    daemon = start_remote_daemon(tmp_path / "remote", password="pw", env={"HOME": str(tmp_path)})
    sockets: list[ClientConnection] = []
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
            if threading.current_thread().name != REQUESTER:
                return False
            sockets.append(frame_locals["ws"])
            return True

        worker = threading.Thread(target=requester, name=REQUESTER, daemon=True)
        # Hold the handshake after the auth frame went out, while the
        # reply is awaited: the socket is open and nobody's yet.
        with Pause(mach.recv_auth_reply.__code__, capture_auth_wait) as held:
            worker.start()
            assert held.paused.wait(30), "the auth wait was not reached"
            assert inject_keyboard_interrupt(worker.ident or 0) == 1
        worker.join(10)
        assert not worker.is_alive()
        assert outcome == {"error": "KeyboardInterrupt"}
        assert link._ws is None
        ws = sockets[0]
        ws.recv_events_thread.join(10)
        assert ws.state.name == "CLOSED"
        assert not ws.recv_events_thread.is_alive()
        # The link re-connects at once afterwards: a Stop is no machine
        # failure, so it started no back-off.
        assert link.request({"type": "ping"}, "pong")["type"] == "pong"
        link.close()
    finally:
        for ws in sockets:
            ws.close()
        mach.close_links()
        daemon.stop()
