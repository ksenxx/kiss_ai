# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""UDS listener must bind before the (slow) SSL context is built.

Production symptom
==================
Even after the orphan-task sweep was moved off the startup critical
path (commit adf35874), users still observed a long delay between
``install.sh`` respawning ``kiss-web`` and the "KISS Sorcar Server is
starting …" overlay clearing.  Timing showed ~0.5–2 s spent inside
``RemoteAccessServer.__init__`` in ``_create_ssl_context``
(``ssl.SSLContext.load_cert_chain`` and, on near-expiry auto-generated
certs, RSA keygen) BEFORE ``_setup_server`` bound either listener.
``install.sh`` polls for ``~/.kiss/sorcar.sock`` and gives up after
15 s, so any noticeable delay before UDS bind is user-visible.

Fix
===
``_setup_server`` binds the UDS listener FIRST (local peers
authenticate via filesystem permissions, so the SSL context is not
needed for UDS), builds the SSL context on a worker thread
(``asyncio.to_thread(_create_ssl_context, ...)``), and only then
binds the WSS listener.  ``__init__`` no longer builds the SSL
context at all — it stores the certfile/keyfile paths and defers the
work.

This is an end-to-end test: it monkey-patches
``kiss.server.web_server._create_ssl_context`` to add a
deterministic delay (mimicking a slow ``load_cert_chain`` or RSA
keygen), starts a real ``RemoteAccessServer``, and requires the UDS
listener to accept a real Unix-domain-socket connection well before
the SSL build would have finished on the old serialised code path.
"""

from __future__ import annotations

import asyncio
import socket
import ssl
import tempfile
import time
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import kiss.server.web_server as _wsmod
from kiss.server.web_server import RemoteAccessServer

# How long the artificially-slow SSL context build blocks for.  Chosen
# to be well above realistic ``load_cert_chain`` + RSA keygen latency
# (~0.5–2 s) so the test cleanly distinguishes "UDS bound before SSL
# finished" from "UDS bound alongside a fast SSL build".
_SSL_DELAY_SECS = 3.0

# UDS must bind well within this budget even while ``_create_ssl_context``
# is deliberately blocked for ``_SSL_DELAY_SECS``.  Pre-fix, UDS bind
# happened AFTER the SSL context was built inside ``__init__`` — the
# elapsed time to first UDS accept exceeded ``_SSL_DELAY_SECS``.  This
# budget is comfortably below the SSL delay so the test's assertion is
# unambiguous.
_UDS_BUDGET_SECS = 1.5


def _find_free_port() -> int:
    """Return an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


class UdsBindsBeforeSslTest(IsolatedAsyncioTestCase):
    """Slow ``_create_ssl_context`` must not delay UDS bind."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-uds-before-ssl-"))
        self.uds_path = self.tmpdir / "sorcar-test.sock"
        self.url_file = self.tmpdir / "remote-url.json"
        self.server: RemoteAccessServer | None = None

        # Snapshot and monkey-patch ``_create_ssl_context`` at the
        # ``web_server`` module so ``_setup_server`` picks up the slow
        # variant (it references the module-level name, not a bound
        # method).  Restored in ``asyncTearDown``.
        self._original_create_ssl = _wsmod._create_ssl_context
        self._ssl_call_started_at: float | None = None
        self._ssl_call_returned_at: float | None = None

        def slow_create_ssl_context(
            certfile: str | None = None,
            keyfile: str | None = None,
        ) -> ssl.SSLContext:
            self._ssl_call_started_at = time.monotonic()
            time.sleep(_SSL_DELAY_SECS)
            ctx = self._original_create_ssl(certfile, keyfile)
            self._ssl_call_returned_at = time.monotonic()
            return ctx

        _wsmod._create_ssl_context = slow_create_ssl_context

    async def asyncTearDown(self) -> None:
        _wsmod._create_ssl_context = self._original_create_ssl
        if self.server is not None:
            await self.server.stop_async()

    async def test_uds_binds_before_slow_ssl_context(self) -> None:
        """UDS accept must succeed well before the SSL build finishes.

        1. Construct the server.  ``__init__`` MUST NOT invoke
           ``_create_ssl_context`` — that work is deferred to
           ``_setup_server`` and it must not run on the ctor's
           critical path.
        2. Start the server.  ``_setup_server`` binds UDS first, then
           builds the SSL context on a worker thread, then binds WSS.
        3. Open a real UDS connection and measure the elapsed time
           from ctor start.  It must be < ``_UDS_BUDGET_SECS``, which
           is comfortably below ``_SSL_DELAY_SECS``.
        4. The slow SSL build must NOT have returned by the time the
           UDS accept happens — proving UDS is genuinely ahead of the
           SSL context creation, not just racing it.
        """
        ctor_started = time.monotonic()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=_find_free_port(),
            use_tunnel=False,
            url_file=self.url_file,
            uds_path=self.uds_path,
        )
        ctor_returned = time.monotonic()
        # 1. Ctor must not have called _create_ssl_context.
        assert self._ssl_call_started_at is None, (
            "_create_ssl_context was called during __init__; SSL work "
            "must be deferred to _setup_server so it does not delay "
            "the UDS bind"
        )
        assert ctor_returned - ctor_started < _UDS_BUDGET_SECS, (
            f"RemoteAccessServer.__init__ took "
            f"{ctor_returned - ctor_started:.2f}s — expected "
            f"<{_UDS_BUDGET_SECS}s (SSL build must not run in ctor)"
        )

        # 2. + 3. Start the server as a background task and race a
        # UDS-connect poll against it — ``start_async`` awaits the full
        # ``_setup_server`` including the slow SSL build, so blocking
        # on it would defeat the whole point of the test.  We only
        # care that the UDS file appears and accepts connections
        # BEFORE the SSL build finishes.
        start_task = asyncio.create_task(self.server.start_async())
        uds_ready_at: float | None = None
        deadline = time.monotonic() + _SSL_DELAY_SECS + 5.0
        while time.monotonic() < deadline:
            if self.uds_path.exists():
                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_unix_connection(str(self.uds_path)),
                        timeout=0.5,
                    )
                except (ConnectionError, FileNotFoundError, OSError, TimeoutError):
                    await asyncio.sleep(0.02)
                    continue
                uds_ready_at = time.monotonic()
                writer.close()
                try:
                    await writer.wait_closed()
                except (ConnectionError, OSError):
                    pass
                break
            await asyncio.sleep(0.02)
        assert uds_ready_at is not None, (
            "UDS listener never became reachable within the deadline"
        )
        elapsed = uds_ready_at - ctor_started
        assert elapsed < _UDS_BUDGET_SECS, (
            f"UDS bind took {elapsed:.2f}s — expected "
            f"<{_UDS_BUDGET_SECS}s (SSL build blocks for "
            f"{_SSL_DELAY_SECS}s so this proves UDS is on the "
            f"critical path)"
        )
        # 4. The SSL build must still be in flight — proves the UDS
        # bind is genuinely ahead of SSL, not just fast because SSL
        # happened to be quick.
        assert (
            self._ssl_call_started_at is not None
            and self._ssl_call_returned_at is None
        ), (
            "UDS bind reached before SSL context creation had even "
            "started, or SSL already finished — the test can no "
            "longer distinguish the two orderings"
        )

        # Sanity: wait for ``start_async`` (which awaits the SSL build
        # and WSS bind) to finish so ``stop_async`` in teardown has a
        # fully-initialised server.
        await asyncio.wait_for(
            start_task, timeout=_SSL_DELAY_SECS + 5.0,
        )
        assert self._ssl_call_returned_at is not None, (
            "slow SSL context build never completed"
        )
