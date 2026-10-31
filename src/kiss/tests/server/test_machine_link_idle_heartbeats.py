# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""An idle ``MachineLink`` outlives the daemon's heartbeats, and a dead socket is replaced.

Reproduces the hourly ``keepalive ping failed`` / ``CHATS: <machine>
did not answer: ... keepalive ping timeout`` of a daemon with a
registered machine.  The daemon's watchdog pushes a ``heartbeat``
frame at every connected socket every 15 s, the link's request socket
only reads between requests, and the websockets sync client pauses
its reader once 16 frames are queued: from then on the pongs of the
socket's own keepalive pings went unread and the keepalive closed the
socket after four idle minutes.  The next request (a history fetch,
a ping, the chat drop-down's focus) failed, the machine backed off as
unreachable and, for an SSH machine, the tunnel every running tab's
stream shares was closed under them.

Both halves of the fix run against a real in-process daemon whose
watchdog ticks every 50 ms and a link whose keepalive runs every
300 ms, so a few seconds stand in for the field's minutes:

* the request socket has no queue limit, so an idle link sees
  hundreds of heartbeats and still answers the next request;
* a socket the daemon closed meanwhile is replaced by a fresh one
  within the same request, without the back-off that would have
  closed an SSH machine's tunnel.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
import time
from pathlib import Path
from unittest import TestCase

from websockets.asyncio.server import ServerConnection

import kiss.agents.sorcar.machines as mach
import kiss.agents.sorcar.persistence as th
import kiss.core.user_config as vc
import kiss.server.web_server as web_server
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.conftest import find_free_port

WATCHDOG_INTERVAL = 0.05
KEEPALIVE = 0.3
# Longer than the library's queue limit of 16 heartbeats plus several
# keepalive rounds: ~60 heartbeats and ~10 keepalive pings.
IDLE_SECONDS = 3.0


class TestMachineLinkIdleHeartbeats(TestCase):
    """One real daemon with a fast watchdog, one ``MachineLink`` to it."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-link-idle-"))
        self._saved_persistence = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = self.tmpdir / ".kiss"
        kiss_dir.mkdir()
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        th._db_conn = None
        self._saved_cfg = (vc.CONFIG_DIR, vc.CONFIG_PATH)
        vc.CONFIG_DIR = self.tmpdir / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"
        self._saved_timing = (
            web_server.TUNNEL_CHECK_INTERVAL, mach.LINK_PING_INTERVAL, mach.LINK_PING_TIMEOUT,
        )
        # The watchdog reads the module constant on every tick.
        web_server.TUNNEL_CHECK_INTERVAL = WATCHDOG_INTERVAL  # type: ignore[assignment]
        mach.LINK_PING_INTERVAL = KEEPALIVE
        mach.LINK_PING_TIMEOUT = KEEPALIVE
        certfile, keyfile = self.tmpdir / "cert.pem", self.tmpdir / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.port = find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=self.port, certfile=str(certfile), keyfile=str(keyfile),
            url_file=self.tmpdir / "remote-url.json",
            local_endpoint_file=self.tmpdir / "sorcar-local.json",
            work_dir=str(self.tmpdir),
        )
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self.server.start_async(), self.loop).result(60)
        mach.save_machines([])
        self.machine = mach.add_machine("box", f"https://127.0.0.1:{self.port}", "")
        self.link = mach.link_for(self.machine)

    def tearDown(self) -> None:
        try:
            mach.save_machines([])
            asyncio.run_coroutine_threadsafe(self.server.stop_async(), self.loop).result(60)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join(timeout=30)
            self.loop.close()
            (
                web_server.TUNNEL_CHECK_INTERVAL, mach.LINK_PING_INTERVAL, mach.LINK_PING_TIMEOUT,
            ) = self._saved_timing
            th._close_db()
            th._DB_PATH, th._db_conn, th._KISS_DIR = self._saved_persistence
            vc.CONFIG_DIR, vc.CONFIG_PATH = self._saved_cfg
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _link_sockets_on_daemon(self) -> list[ServerConnection]:
        """The daemon's server-side connections (the link's request socket among them)."""
        servers = (self.server._ws_server, self.server._ws_loopback_server)
        return [ws for srv in servers if srv is not None for ws in srv.connections]

    def test_idle_link_survives_many_heartbeats(self) -> None:
        """Hundreds of unread heartbeats do not kill the request socket."""
        self.assertEqual(self.link.request({"type": "ping"}, "pong")["type"], "pong")
        socket_before = self.link._ws
        time.sleep(IDLE_SECONDS)
        # Still the same, living socket: the daemon kept its side open
        # and the link's keepalive never gave up on it.
        self.assertEqual(self.link.request({"type": "ping"}, "pong")["type"], "pong")
        self.assertIs(self.link._ws, socket_before)
        self.assertEqual(self.link._health.retry_at, 0.0)

    def test_request_replaces_a_socket_the_daemon_closed(self) -> None:
        """A kept socket the daemon closed is replaced, with no back-off."""
        self.assertEqual(self.link.request({"type": "ping"}, "pong")["type"], "pong")
        socket_before = self.link._ws
        # The daemon restarts (or drops the idle peer): every server-side
        # socket is closed for real.
        for ws in self._link_sockets_on_daemon():
            asyncio.run_coroutine_threadsafe(ws.close(), self.loop).result(10)
        deadline = time.monotonic() + 5
        while socket_before is not None and socket_before.protocol.state.name != "CLOSED":
            self.assertLess(time.monotonic(), deadline, "the link's socket never saw the close")
            time.sleep(0.05)
        # The same request succeeds over a fresh socket; the machine is
        # not written off (no back-off), so an SSH tunnel would have
        # stayed open under a running tab's stream.
        self.assertEqual(self.link.request({"type": "ping"}, "pong")["type"], "pong")
        self.assertIsNot(self.link._ws, socket_before)
        self.assertEqual(self.link._health.retry_at, 0.0)
        # A daemon that is really gone still counts as down.
        asyncio.run_coroutine_threadsafe(self.server.stop_async(), self.loop).result(60)
        with self.assertRaises(mach.MachineError):
            self.link.request({"type": "ping"}, "pong", timeout=3)
        self.assertGreater(self.link._health.retry_at, 0.0)
