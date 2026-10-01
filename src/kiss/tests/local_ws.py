# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Test-side local client for the daemon's WSS endpoint.

The daemon's local channel is a token-authenticated WebSocket (see
:mod:`kiss.agents.sorcar.local_endpoint`).  Tests that drive a
:class:`~kiss.server.web_server.RemoteAccessServer` as a local client
(a VS Code window) use :func:`open_local_connection`, which reads the
server's endpoint file, connects with the pinned CA, completes the
``auth`` handshake and returns a ``(reader, writer)`` pair with the
newline-delimited surface of ``asyncio.open_unix_connection``:
``writer.write(json_line)`` sends one text frame per line and
``reader.readline()`` returns one event per frame with a trailing
newline (``b""`` once the connection is closed).  Keeping that surface
means a test written against the former Unix socket reads the same.

:func:`open_remote_connection` opens the same listener as a browser
would (password ``auth``), for tests that need a non-local peer.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from kiss.agents.sorcar import local_endpoint

_DEFAULT_MAX_SIZE = 64 * 1024 * 1024


class LocalReader:
    """``StreamReader``-shaped reader over a WebSocket connection."""

    def __init__(self, ws: ClientConnection) -> None:
        self._ws = ws
        self._eof = False
        self._pending: bytes = b""

    async def readline(self) -> bytes:
        """Return the next event as ``json + b"\\n"``, or ``b""`` at EOF."""
        if self._pending:
            line, self._pending = self._pending, b""
            return line
        if self._eof:
            return b""
        try:
            message = await self._ws.recv()
        except ConnectionClosed:
            self._eof = True
            return b""
        data = message.encode("utf-8") if isinstance(message, str) else bytes(message)
        return data + b"\n"

    async def read(self, n: int = -1) -> bytes:
        """Return up to *n* bytes of the next event line (``-1``: the whole line)."""
        line = await self.readline()
        if n < 0 or n >= len(line):
            return line
        self._pending = line[n:]
        return line[:n]

    def at_eof(self) -> bool:
        """Whether the connection has closed and nothing is buffered."""
        return self._eof and not self._pending


class LocalWriter:
    """``StreamWriter``-shaped writer over a WebSocket connection.

    Bytes given to :meth:`write` are split on newlines and every
    complete line is sent as one text frame by :meth:`drain` (or by the
    next :meth:`write`, so callers that never await ``drain`` still get
    their frames out).
    """

    def __init__(self, ws: ClientConnection) -> None:
        self._ws = ws
        self._buffer = b""
        self._sends: list[asyncio.Task[None]] = []
        self._closing = False

    def write(self, data: bytes) -> None:
        """Buffer *data*; each complete line is queued as one frame."""
        self._buffer += data
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            if not line.strip():
                continue
            self._sends.append(asyncio.ensure_future(self._send(line)))

    async def _send(self, line: bytes) -> None:
        try:
            await self._ws.send(line.decode("utf-8"))
        except ConnectionClosed:
            pass

    async def drain(self) -> None:
        """Wait until every queued frame has been handed to the transport."""
        sends, self._sends = self._sends, []
        for task in sends:
            await task

    def close(self) -> None:
        """Start closing the connection (idempotent)."""
        if self._closing:
            return
        self._closing = True
        asyncio.ensure_future(self._ws.close())

    def is_closing(self) -> bool:
        """Whether :meth:`close` has been called."""
        return self._closing

    async def wait_closed(self) -> None:
        """Wait for the closing handshake to finish."""
        await self._ws.wait_closed()

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        """Mirror ``StreamWriter.get_extra_info`` for the underlying transport."""
        transport = getattr(self._ws, "transport", None)
        if transport is None:
            return default
        return transport.get_extra_info(name, default)

    @property
    def transport(self) -> Any:
        """The underlying asyncio transport."""
        return getattr(self._ws, "transport", None)

    @property
    def connection(self) -> ClientConnection:
        """The raw ``websockets`` connection (for protocol-level tests)."""
        return self._ws


def _endpoint_file(server_or_path: Any) -> Path:
    """Accept a server, an endpoint path, or a directory holding one."""
    if isinstance(server_or_path, (str, Path)):
        path = Path(server_or_path)
        return path / local_endpoint.LOCAL_ENDPOINT_FILE if path.is_dir() else path
    return Path(server_or_path._local_endpoint_file)


async def connect_local(
    server_or_path: Any,
    *,
    max_size: int | None = _DEFAULT_MAX_SIZE,
    open_timeout: float = 10.0,
) -> ClientConnection:
    """Connect to the daemon behind *server_or_path* as an authenticated local client.

    Args:
        server_or_path: A ``RemoteAccessServer``, its endpoint file, or
            the directory that holds one.
        max_size: Largest incoming frame accepted.
        open_timeout: Seconds allowed for the handshake and ``auth_ok``.

    Returns:
        The raw ``websockets`` connection, already past ``auth_ok``.

    Raises:
        ConnectionError: When the endpoint file is missing or the daemon
            does not accept the token as local.
    """
    path = _endpoint_file(server_or_path)
    endpoint = local_endpoint.read_endpoint(path)
    if endpoint is None:
        raise ConnectionError(f"no daemon endpoint file at {path}")
    ws = await connect(
        endpoint.url,
        ssl=local_endpoint.client_ssl_context(endpoint),
        open_timeout=open_timeout,
        close_timeout=2.0,
        compression=None,
        max_size=max_size,
    )
    await ws.send(json.dumps({"type": "auth", "token": endpoint.token}))
    reply = json.loads(await asyncio.wait_for(ws.recv(), open_timeout))
    if reply.get("type") != "auth_ok" or not reply.get("local"):
        await ws.close()
        raise ConnectionError(f"local auth rejected: {reply!r}")
    return ws


async def open_local_connection(
    server_or_path: Any,
    *,
    limit: int | None = None,
    open_timeout: float = 10.0,
) -> tuple[LocalReader, LocalWriter]:
    """Open a local connection with a ``(reader, writer)`` line surface.

    Args:
        server_or_path: A ``RemoteAccessServer``, its endpoint file, or
            the directory that holds one.
        limit: Largest incoming frame accepted (the old ``StreamReader``
            ``limit``); ``None`` keeps the 64 MiB daemon frame limit.
        open_timeout: Seconds allowed for the handshake and ``auth_ok``.

    Returns:
        ``(reader, writer)`` speaking newline-delimited JSON.
    """
    ws = await connect_local(
        server_or_path,
        max_size=limit if limit is not None else _DEFAULT_MAX_SIZE,
        open_timeout=open_timeout,
    )
    return LocalReader(ws), LocalWriter(ws)


async def open_remote_connection(
    server: Any,
    password: str = "",
    *,
    max_size: int | None = _DEFAULT_MAX_SIZE,
) -> ClientConnection:
    """Connect to *server*'s listener the way a browser does.

    Sends the password ``auth`` frame and returns the raw connection
    after ``auth_ok``.

    Args:
        server: The ``RemoteAccessServer`` under test.
        password: The remote password to present (empty by default).
        max_size: Largest incoming frame accepted.

    Returns:
        The raw ``websockets`` connection.
    """
    endpoint = local_endpoint.read_endpoint(Path(server._local_endpoint_file))
    assert endpoint is not None, "daemon endpoint file missing"
    ssl_ctx: ssl.SSLContext = local_endpoint.client_ssl_context(endpoint)
    ws = await connect(
        endpoint.url, ssl=ssl_ctx, compression=None, max_size=max_size,
        close_timeout=2.0,
    )
    await ws.send(json.dumps({"type": "auth", "password": password}))
    reply = json.loads(await asyncio.wait_for(ws.recv(), 10.0))
    assert reply.get("type") == "auth_ok", reply
    return ws


def make_test_tls(tmp_dir: Path) -> tuple[Path, Path, Path]:
    """Issue a throwaway server certificate under *tmp_dir*.

    Returns ``(certfile, keyfile, ca_file)``; the CA beside the
    certificate is what a local client pins.
    """
    from kiss.server.web_server import _generate_self_signed_cert

    certfile, keyfile = tmp_dir / "cert.pem", tmp_dir / "key.pem"
    _generate_self_signed_cert(certfile, keyfile)
    return certfile, keyfile, tmp_dir / "ca.pem"


@contextlib.asynccontextmanager
async def fake_daemon(
    tmp_dir: Path,
    handler: Callable[[ServerConnection], Awaitable[None]],
    *,
    token: str = "test-token",
    answer_auth: bool = True,
    endpoint_file: Path | None = None,
) -> AsyncIterator[Path]:
    """Serve a stand-in daemon over ``wss://`` and yield its endpoint file.

    For tests of the daemon's *clients* (``daemon_client.run``, the
    channel-agent launcher, the active-tasks probe).  Listens on an
    ephemeral loopback port with a certificate from
    :func:`make_test_tls`, writes an endpoint file and, unless
    *answer_auth* is false, consumes the client's ``auth`` frame and
    answers ``auth_ok`` (``local`` true iff the token matches) before
    *handler* runs with the connection.

    Args:
        tmp_dir: Directory for the certificate, CA and endpoint file.
        handler: Coroutine run per authenticated connection.
        token: The local token the endpoint file advertises.
        answer_auth: Whether to handle the ``auth`` frame here.
        endpoint_file: Where to write the endpoint (default
            ``tmp_dir / "sorcar-local.json"``).

    Yields:
        The endpoint file path (pass as ``endpoint_file=`` to clients).
    """
    certfile, keyfile, ca_file = make_test_tls(tmp_dir)
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(certfile, keyfile)
    ssl_ctx.num_tickets = 0  # same reason as web_server._create_ssl_context
    path = endpoint_file or tmp_dir / local_endpoint.LOCAL_ENDPOINT_FILE

    async def _serve(ws: ServerConnection) -> None:
        if answer_auth:
            try:
                first = json.loads(await ws.recv())
            except (ConnectionClosed, ValueError):
                return
            ok = isinstance(first, dict) and first.get("type") == "auth"
            local = ok and first.get("token") == token
            if not ok:
                await ws.send(json.dumps({"type": "auth_required"}))
                return
            await ws.send(json.dumps({"type": "auth_ok", "local": local}))
        await handler(ws)

    async with serve(
        _serve, "127.0.0.1", 0, ssl=ssl_ctx, compression=None,
        max_size=_DEFAULT_MAX_SIZE,
    ) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        local_endpoint.write_endpoint(
            path,
            local_endpoint.LocalEndpoint(
                url=f"wss://127.0.0.1:{port}/ws", token=token,
                ca=str(ca_file), pid=os.getpid(),
            ),
        )
        try:
            yield path
        finally:
            server.close(close_connections=True)
