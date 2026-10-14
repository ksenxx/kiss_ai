# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-13: a stream lost while being bound must not stay the tab's binding.

``RemoteTabs._bind`` opens the stream (``RemoteTab.__init__`` connects and
starts the relay thread) outside the lock and registers it under the lock
afterwards.  A remote daemon that drops the socket right after ``auth_ok``
(restarting, a link that flaps) ends the relay thread, whose ``dropped``
ran BEFORE the registration: it found the stream unregistered, took it for
one already retired and did nothing.  ``_bind`` then installed the dead
stream as the tab's binding and announced it.  Every later submit on the
tab with the same machine reused that stream and was refused with "the
connection is lost"; the tab never re-bound on its own, so the user had to
pick another machine or close the tab.

Now ``dropped`` retires the stream whether or not it is registered, and
``_bind`` refuses the run instead of installing a retired stream.

The remote daemon stand-in is a loopback TLS WebSocket server that
answers ``auth`` with ``auth_ok`` and closes at once.  Which of the two
threads reaches the lock first is the scheduler's call, so the submit is
repeated on fresh tabs: before the fix a fair share of them stay bound to
a dead stream.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import shutil
import socket
import ssl
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from kiss.agents.sorcar import machines as mach
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.agents.sorcar.test_audit1012_sorcar_core_machine_link_send import (
    _AUTH_OK,
    _consume_client_frame,
)
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import make_test_tls, open_local_connection

_ROUNDS = 40
_CLOSE_FRAME = b"\x88\x02\x03\xe8"  # close, code 1000


def _auth_ok_then_close_server(tmp: Path) -> tuple[threading.Thread, int, Path, threading.Event]:
    """A daemon stand-in that authenticates the link and drops it at once."""
    certfile, keyfile, ca_file = make_test_tls(tmp)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile, keyfile)
    ctx.num_tickets = 0
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    listener.settimeout(0.2)
    port = listener.getsockname()[1]
    stop = threading.Event()

    def serve() -> None:
        try:
            while not stop.is_set():
                try:
                    raw, _ = listener.accept()
                except TimeoutError:
                    continue
                with ctx.wrap_socket(raw, server_side=True) as tls:
                    request = b""
                    while b"\r\n\r\n" not in request:
                        chunk = tls.recv(4096)
                        if not chunk:
                            break
                        request += chunk
                    key = ""
                    for line in request.decode("latin-1").split("\r\n"):
                        if line.lower().startswith("sec-websocket-key:"):
                            key = line.split(":", 1)[1].strip()
                    accept = base64.b64encode(
                        hashlib.sha1(
                            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode(),
                        ).digest()
                    ).decode()
                    tls.sendall(
                        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                        b"Connection: Upgrade\r\nSec-WebSocket-Accept: "
                        + accept.encode()
                        + b"\r\n\r\n",
                    )
                    _consume_client_frame(tls)  # the auth frame
                    # auth_ok and the close in one segment: the stream is
                    # dead by the time the client has read its welcome.
                    tls.sendall(b"\x81" + bytes([len(_AUTH_OK)]) + _AUTH_OK + _CLOSE_FRAME)
        finally:
            listener.close()

    thread = threading.Thread(target=serve, name="flapping-daemon", daemon=True)
    thread.start()
    return thread, port, ca_file, stop


class TestRemoteTabLostAtBind(IsolatedAsyncioTestCase):
    """One local daemon, one "remote daemon" that drops every stream, one client."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-flap-tab-")
        self.flap_thread, port, ca_file, self.flap_stop = _auth_ok_then_close_server(
            Path(self.tmpdir) / "flap",
        )
        flap = mach.Machine(name="flap", url=f"https://127.0.0.1:{port}", password="pw")
        flap.cert_file().parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ca_file, flap.cert_file())
        mach.save_machines([flap])
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=find_free_port(),
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        await self.server.start_async()
        self.reader, self.writer = await open_local_connection(self.server)

    async def asyncTearDown(self) -> None:
        self.writer.close()
        await self.server.stop_async()
        self.flap_stop.set()
        self.flap_thread.join(5)
        mach.close_links()
        mach.save_machines([])
        mach.Machine(name="flap").cert_file().unlink(missing_ok=True)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _send(self, cmd: dict[str, Any]) -> None:
        self.writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await self.writer.drain()

    async def _until(self, reply_type: str, tab_id: str, timeout: float) -> dict[str, Any]:
        """Return the first *reply_type* message for *tab_id*, within *timeout* seconds."""
        deadline = time.monotonic() + timeout
        while True:
            line = await asyncio.wait_for(
                self.reader.readline(),
                timeout=max(deadline - time.monotonic(), 0.1),
            )
            assert line, "local connection closed"
            msg: dict[str, Any] = json.loads(line.decode("utf-8"))
            if msg.get("type") == reply_type and msg.get("tabId") == tab_id:
                return msg

    async def test_a_stream_dropped_during_bind_never_becomes_the_binding(self) -> None:
        remote_tabs = self.server._server_api.remote_tabs
        stuck: list[str] = []
        for i in range(_ROUNDS):
            tab_id = f"tab-flap-{i}"
            await self._send(
                {"type": "submit", "tabId": tab_id, "machine": "flap", "prompt": "hi",
                 "workDir": self.tmpdir}
            )
            error = await self._until("error", tab_id, timeout=30.0)
            self.assertIn("flap", error["text"])
            # Whichever thread won: the dead stream is not the tab's binding.
            deadline = time.monotonic() + 2.0
            while remote_tabs.machine_of(tab_id) and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            if remote_tabs.machine_of(tab_id):
                stuck.append(tab_id)
        self.assertEqual(stuck, [], f"{len(stuck)}/{_ROUNDS} tabs kept a dead stream as binding")
