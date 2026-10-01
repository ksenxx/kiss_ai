# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The (slow) SSL context build must stay off the constructor and the event loop.

Production symptom
==================
Even after the orphan-task sweep was moved off the startup critical
path (commit adf35874), users still observed a long delay between
``install.sh`` respawning ``kiss-web`` and the "KISS Sorcar Server is
starting …" overlay clearing.  Timing showed ~0.5–2 s spent inside
``RemoteAccessServer.__init__`` in ``_create_ssl_context``
(``ssl.SSLContext.load_cert_chain`` and, on near-expiry auto-generated
certs, key generation) BEFORE ``_setup_server`` bound the listener.
``install.sh`` polls for the daemon's endpoint file and gives up after
15 s, so any noticeable delay before the listener is up is user-visible.

Fix
===
``__init__`` no longer builds the SSL context at all — it stores the
certfile/keyfile paths and defers the work.  ``_bind_listeners`` builds
the context on a worker thread (``asyncio.to_thread(_create_ssl_context,
...)``), binds the WSS listener, and only then publishes the local
endpoint file that same-machine clients (the VS Code extension,
``run_agent``) read.

This is an end-to-end test: it monkey-patches
``kiss.server.web_server._create_ssl_context`` to add a deterministic
delay (mimicking a slow ``load_cert_chain`` or key keygen), starts a
real ``RemoteAccessServer``, and requires (1) a fast constructor that
never calls ``_create_ssl_context``, (2) an event loop that keeps
ticking while the slow build runs, and (3) an endpoint file that
appears only once the listener is bound, with mode 0600 and a token the
daemon accepts as local.
"""

from __future__ import annotations

import asyncio
import json
import socket
import ssl
import stat
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import kiss.server.web_server as _wsmod
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.local_ws import open_local_connection

_SSL_DELAY_SECS = 3.0

_CTOR_BUDGET_SECS = 1.5

# A loop blocked by the SSL build would stall its ticker for the whole
# _SSL_DELAY_SECS; a loop that is free ticks every few milliseconds.
_MAX_LOOP_STALL_SECS = 1.0


def _find_free_port() -> int:
    """Return an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


class SslBuildOffLoopTest(IsolatedAsyncioTestCase):
    """Slow ``_create_ssl_context`` must not run in ``__init__`` or on the loop."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-ssl-off-loop-"))
        self.endpoint_file = self.tmpdir / "sorcar-local.json"
        self.url_file = self.tmpdir / "remote-url.json"
        self.server: RemoteAccessServer | None = None

        self._original_create_ssl = _wsmod._create_ssl_context
        self._ssl_call_started_at: float | None = None
        self._ssl_call_returned_at: float | None = None
        self._endpoint_seen_during_ssl_build = False

        def slow_create_ssl_context(
            certfile: str | None = None,
            keyfile: str | None = None,
            lan_ips: Iterable[str] | None = None,
        ) -> ssl.SSLContext:
            self._ssl_call_started_at = time.monotonic()
            time.sleep(_SSL_DELAY_SECS)
            # The endpoint file must not be published before the
            # listener (which needs this context) is bound.
            self._endpoint_seen_during_ssl_build = self.endpoint_file.exists()
            ctx = self._original_create_ssl(certfile, keyfile, lan_ips)
            self._ssl_call_returned_at = time.monotonic()
            return ctx

        _wsmod._create_ssl_context = slow_create_ssl_context

    async def asyncTearDown(self) -> None:
        _wsmod._create_ssl_context = self._original_create_ssl
        if self.server is not None:
            await self.server.stop_async()

    async def test_ssl_build_deferred_and_off_loop(self) -> None:
        """Constructor is SSL-free, the loop stays live, the endpoint follows bind.

        1. Construct the server.  ``__init__`` MUST NOT invoke
           ``_create_ssl_context`` — that work is deferred to
           ``_bind_listeners`` and it must not run on the ctor's
           critical path.
        2. Start the server while a ticker task measures the longest
           gap between two loop iterations.  The ``_SSL_DELAY_SECS``
           sleep inside the patched build must not show up as a loop
           stall, proving the build runs on a worker thread.
        3. The endpoint file must be absent while the SSL build is
           still running and present (mode 0600) once ``start_async``
           returns; a local client reading it must be authenticated as
           ``local`` by the real daemon.
        """
        ctor_started = time.monotonic()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=_find_free_port(),
            use_tunnel=False,
            url_file=self.url_file,
            local_endpoint_file=self.endpoint_file,
        )
        ctor_returned = time.monotonic()
        assert self._ssl_call_started_at is None, (
            "_create_ssl_context was called during __init__; SSL work "
            "must be deferred to _bind_listeners so it does not delay "
            "the listener bind"
        )
        assert ctor_returned - ctor_started < _CTOR_BUDGET_SECS, (
            f"RemoteAccessServer.__init__ took "
            f"{ctor_returned - ctor_started:.2f}s — expected "
            f"<{_CTOR_BUDGET_SECS}s (SSL build must not run in ctor)"
        )
        assert not self.endpoint_file.exists(), (
            "endpoint file published before the listener was bound"
        )

        max_stall = 0.0
        ticking = True

        async def ticker() -> None:
            nonlocal max_stall
            last = time.monotonic()
            while ticking:
                await asyncio.sleep(0.01)
                now = time.monotonic()
                max_stall = max(max_stall, now - last)
                last = now

        ticker_task = asyncio.create_task(ticker())
        try:
            await asyncio.wait_for(
                self.server.start_async(), timeout=_SSL_DELAY_SECS + 20.0,
            )
        finally:
            ticking = False
            await ticker_task

        assert self._ssl_call_returned_at is not None, (
            "slow SSL context build never completed"
        )
        assert self._ssl_call_started_at is not None
        assert (
            self._ssl_call_returned_at - self._ssl_call_started_at
            >= _SSL_DELAY_SECS
        ), "the patched slow build did not actually run"
        assert max_stall < _MAX_LOOP_STALL_SECS, (
            f"event loop stalled for {max_stall:.2f}s during start_async; "
            f"the {_SSL_DELAY_SECS}s SSL build must run on a worker thread"
        )
        assert not self._endpoint_seen_during_ssl_build, (
            "endpoint file was published while the SSL context was still "
            "being built, i.e. before the WSS listener could be bound"
        )

        assert self.endpoint_file.exists(), (
            "start_async returned without publishing the endpoint file"
        )
        if sys.platform != "win32":  # NTFS has no POSIX mode bits
            mode = stat.S_IMODE(self.endpoint_file.stat().st_mode)
            assert mode == 0o600, f"endpoint file mode {oct(mode)}, expected 0o600"
        data = json.loads(self.endpoint_file.read_text())
        assert data["url"] == f"wss://127.0.0.1:{self.server.port}/ws"
        assert data["token"] == self.server.local_token

        reader, writer = await open_local_connection(self.server)
        try:
            writer.write(json.dumps({"type": "ping"}).encode() + b"\n")
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout=10.0)
            assert json.loads(line)["type"] == "pong", line
        finally:
            writer.close()
            await writer.wait_closed()
