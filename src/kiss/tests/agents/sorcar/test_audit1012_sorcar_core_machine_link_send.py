# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-12 (sorcar-core): ``MachineLink.request`` cannot wedge on a deaf daemon.

The synchronous ``websockets`` client writes with no deadline.
``MachineLink.request`` sent under ``MachineLink._lock``, so a remote
daemon that stopped reading (frozen, suspended, half-open NAT) left the
send blocked for good with the lock held: every later request to that
machine, and the link's close at daemon exit, blocked behind it.  Both
sends now go through ``local_endpoint.send``, which closes the socket
under a write that misses its deadline.

The test runs a REAL TLS server on the loopback that completes the
WebSocket upgrade, answers the ``auth`` frame with ``auth_ok`` and then
never reads again, so a large command fills the kernel buffers.
"""

from __future__ import annotations

import base64
import hashlib
import shutil
import socket
import ssl
import tempfile
import threading
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.machines import Machine, MachineError, MachineLink
from kiss.tests.local_ws import make_test_tls

_AUTH_OK = b'{"type": "auth_ok"}'


def _read_exactly(tls: ssl.SSLSocket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = tls.recv(n - len(data))
        if not chunk:
            raise ConnectionError("client went away")
        data += chunk
    return data


def _consume_client_frame(tls: ssl.SSLSocket) -> None:
    """Read and discard one (masked) client frame."""
    header = _read_exactly(tls, 2)
    length = header[1] & 0x7F
    if length == 126:
        length = int.from_bytes(_read_exactly(tls, 2), "big")
    elif length == 127:
        length = int.from_bytes(_read_exactly(tls, 8), "big")
    _read_exactly(tls, 4 + length)  # mask key + payload


def _auth_ok_then_deaf_server(tmp: Path) -> tuple[threading.Thread, int, Path, threading.Event]:
    """A daemon stand-in that authenticates the link and then stops reading."""
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
                    tls.sendall(b"\x81" + bytes([len(_AUTH_OK)]) + _AUTH_OK)
                    # Never read again: the client's writes pile up.
                    stop.wait(60)
        finally:
            listener.close()

    thread = threading.Thread(target=serve, name="deaf-daemon", daemon=True)
    thread.start()
    return thread, port, ca_file, stop


def test_request_to_a_daemon_that_stopped_reading_fails_in_time_and_frees_the_link() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="kiss-deaf-link-"))
    thread, port, ca_file, stop = _auth_ok_then_deaf_server(tmp)
    link = MachineLink(
        Machine(name="deaf", url=f"https://127.0.0.1:{port}", password="pw"),
        cert_file=ca_file,
    )
    try:
        started = time.monotonic()
        with pytest.raises(MachineError, match="did not answer"):
            link.request(
                {"type": "history", "pad": "x" * (64 * 1024 * 1024)}, "history", timeout=2.0
            )
        assert time.monotonic() - started < 10.0
        # The lock is free again: the next request fails fast (backoff), it does not block.
        started = time.monotonic()
        with pytest.raises(MachineError):
            link.request({"type": "history"}, "history", timeout=2.0)
        assert time.monotonic() - started < 1.0
        started = time.monotonic()
        link.close()
        assert time.monotonic() - started < 1.0
    finally:
        link.close()
        stop.set()
        thread.join(5)
        shutil.rmtree(tmp, ignore_errors=True)
