# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A ``MachineLink`` compresses what it carries (permessage-deflate).

The daemon-to-daemon sockets used to be opened with ``compression=None``,
so a registered machine's CHATS page (``history``, 70-180 KB of JSON)
and every transcript relayed into a bound tab (``task_events``, up to
a few MB) crossed the network between the two daemons uncompressed;
on a slow link that was most of the wait after a click on a remote
machine's row.  Both the shared request socket and a stream socket
now negotiate permessage-deflate, which the daemon grants.

Real daemon (``RemoteAccessServer`` with TLS on a free port), real
link, no mocks.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
from pathlib import Path
from unittest import TestCase

from websockets.extensions.permessage_deflate import PerMessageDeflate

import kiss.agents.sorcar.machines as mach
import kiss.agents.sorcar.persistence as th
import kiss.core.user_config as vc
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import _generate_self_signed_cert
from kiss.tests.persistence_helpers import close_db

# Bigger than one deflate block and than the transport's write limit,
# so the reply is compressed and drained in several steps.
_BIG_FILE_CHARS = 300_000


def _deflate_of(ws: object) -> PerMessageDeflate | None:
    """The permessage-deflate extension negotiated on *ws* (client or server side), if any."""
    protocol = getattr(ws, "protocol", None)
    for ext in getattr(protocol, "extensions", ()):
        if isinstance(ext, PerMessageDeflate):
            return ext
    return None


class TestMachineLinkDeflate(TestCase):
    """One real daemon, one ``MachineLink`` to it."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-link-deflate-"))
        self._saved_persistence = (th._DB_PATH, th._KISS_DIR)
        kiss_dir = self.tmpdir / ".kiss"
        kiss_dir.mkdir()
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        self._saved_cfg = (vc.CONFIG_DIR, vc.CONFIG_PATH)
        vc.CONFIG_DIR = self.tmpdir / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"
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
            close_db()
            th._DB_PATH, th._KISS_DIR = self._saved_persistence
            vc.CONFIG_DIR, vc.CONFIG_PATH = self._saved_cfg
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _daemon_side_sockets(self) -> list[object]:
        """The daemon's server-side connections (the link's sockets among them)."""
        servers = (self.server._ws_server, self.server._ws_loopback_server)
        return [ws for srv in servers if srv is not None for ws in srv.connections]

    def test_request_socket_negotiates_deflate(self) -> None:
        """The shared request socket is compressed at both ends."""
        self.assertEqual(self.link.request({"type": "ping"}, "pong")["type"], "pong")
        assert self.link._ws is not None
        self.assertIsNotNone(
            _deflate_of(self.link._ws), "the link's request socket negotiated no deflate",
        )
        daemon_side = [ws for ws in self._daemon_side_sockets() if _deflate_of(ws) is not None]
        self.assertTrue(daemon_side, "the daemon granted no deflate on the link's socket")

    def test_stream_socket_negotiates_deflate(self) -> None:
        """A tab's stream socket (``open_connection``) is compressed too."""
        stream = self.link.open_connection()
        try:
            self.assertIsNotNone(_deflate_of(stream), "the stream socket negotiated no deflate")
        finally:
            stream.close()

    def test_big_reply_round_trips_compressed(self) -> None:
        """A reply far bigger than the write limit arrives whole over the compressed socket."""
        big = self.tmpdir / "big.txt"
        # Line numbers keep the text from being one repeated byte, so
        # the compressed reply is still many frames' worth of bytes.
        text = "".join(
            f"line {i}: the quick brown fox jumps over the lazy dog\n" for i in range(9000)
        )
        text = text[:_BIG_FILE_CHARS]
        # Bytes, not ``write_text``: that would store CRLF line ends on
        # Windows, and ``openFile`` returns the file as it is on disk.
        big.write_bytes(text.encode("utf-8"))
        reply = self.link.request(
            {"type": "openFile", "path": str(big), "tabId": "tab-1"}, "fileContent", timeout=30,
        )
        self.assertEqual(reply.get("content"), text)
        assert self.link._ws is not None
        self.assertIsNotNone(_deflate_of(self.link._ws))
