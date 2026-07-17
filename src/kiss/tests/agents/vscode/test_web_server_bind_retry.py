# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the WSS-bind retry / clean-exit fix.

Background
----------
Before the fix, ``RemoteAccessServer._setup_server`` called
``await serve(...)`` with no error handling.  If the TCP port was
busy (the most common cause: a previous ``kiss-web`` instance had
just SIGTERM'd and its port was still in ``TIME_WAIT``, racing the
supervisor's respawn) the websockets library raised ``OSError`` with
``errno=EADDRINUSE`` straight out of ``asyncio.run`` in
:meth:`RemoteAccessServer.start`, producing a full traceback on
stderr.  The supervisor (launchd, the VS Code extension's respawn
loop) would then immediately respawn into the same OSError,
producing a visible flap loop until the port finally freed.

The fix wraps the bind in a bounded retry with backoff
(:data:`_BIND_RETRY_ATTEMPTS` attempts, :data:`_BIND_RETRY_BACKOFF`
between attempts), retrying only on transient errnos
(:data:`_BIND_RETRYABLE_ERRNOS` = ``EADDRINUSE`` / ``EADDRNOTAVAIL``)
and exiting via :class:`SystemExit` with a single-line stderr
message — no traceback — when all attempts are exhausted.

These tests drive ``_setup_server`` directly with shortened retry
constants so the loop terminates in well under a second.
"""

from __future__ import annotations

import asyncio
import socket
import tempfile
from unittest import IsolatedAsyncioTestCase

import kiss.server.web_server as ws_mod
from kiss.server.web_server import RemoteAccessServer


def _occupy_port() -> tuple[socket.socket, int]:
    """Bind a listening TCP socket on a free localhost port.

    Returns:
        ``(sock, port)`` — the caller owns *sock* and must close it
        to free the port.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return sock, int(sock.getsockname()[1])


class _BindRetryTestBase(IsolatedAsyncioTestCase):
    """Common knob-shortening setup so the retry loop terminates fast."""

    async def asyncSetUp(self) -> None:
        # Original constants, restored in teardown so a failure in
        # one test does not slow neighbouring tests.
        self._orig_attempts = ws_mod._BIND_RETRY_ATTEMPTS
        self._orig_backoff = ws_mod._BIND_RETRY_BACKOFF
        ws_mod._BIND_RETRY_ATTEMPTS = 3
        ws_mod._BIND_RETRY_BACKOFF = (0.05, 0.05, 0.05)
        self._tmpdir = tempfile.TemporaryDirectory()

    async def asyncTearDown(self) -> None:
        ws_mod._BIND_RETRY_ATTEMPTS = self._orig_attempts
        ws_mod._BIND_RETRY_BACKOFF = self._orig_backoff
        self._tmpdir.cleanup()


class TestPortBusyExitsCleanly(_BindRetryTestBase):
    """A persistently busy port produces a SystemExit, not an OSError."""

    async def test_persistently_busy_port_raises_system_exit(self) -> None:
        """All retries exhausted → ``SystemExit`` with exit code 2."""
        busy, port = _occupy_port()
        try:
            server = RemoteAccessServer(
                host="127.0.0.1",
                port=port,
                work_dir=self._tmpdir.name,
                # Use a per-test UDS path so concurrent test runs do
                # not race on the shared ``~/.kiss/sorcar.sock``.
                uds_path=f"{self._tmpdir.name}/sorcar.sock",
            )
            with self.assertRaises(SystemExit) as ctx:
                await server._setup_server()
            self.assertEqual(ctx.exception.code, 2)
            # The server must NOT have been left half-initialised.
            self.assertIsNone(server._ws_server)
        finally:
            busy.close()


class TestPortFreesDuringRetry(_BindRetryTestBase):
    """Port freed mid-retry → bind succeeds without restarting."""

    async def test_port_freed_during_retry_succeeds(self) -> None:
        """Bind that fails once and then succeeds must produce a server.

        Reproduces the realistic "previous kiss-web just SIGTERM'd,
        its port is in TIME_WAIT, the supervisor respawned, the port
        frees a couple of hundred ms later" race.
        """
        # Allow more attempts so the asynchronous "release after a
        # short sleep" task definitely lands inside the retry window.
        ws_mod._BIND_RETRY_ATTEMPTS = 6
        ws_mod._BIND_RETRY_BACKOFF = (0.05,) * 6

        busy, port = _occupy_port()

        async def _free_after_delay() -> None:
            await asyncio.sleep(0.12)
            busy.close()

        free_task = asyncio.create_task(_free_after_delay())
        server = RemoteAccessServer(
            host="127.0.0.1",
            port=port,
            work_dir=self._tmpdir.name,
            uds_path=f"{self._tmpdir.name}/sorcar.sock",
        )
        try:
            await server._setup_server()
            self.assertIsNotNone(server._ws_server)
        finally:
            await free_task
            await server.stop_async()


class TestNonRetryableErrnoFailsFast(_BindRetryTestBase):
    """A non-retryable OSError must SystemExit on the very first attempt."""

    async def test_non_retryable_errno_fails_fast(self) -> None:
        """``EACCES`` from ``serve`` must not be retried."""
        import errno as _errno

        attempts: list[int] = []
        real_serve = ws_mod.serve

        async def _fake_serve(*args: object, **kwargs: object) -> object:
            attempts.append(1)
            raise OSError(_errno.EACCES, "Permission denied")

        ws_mod.serve = _fake_serve  # type: ignore[misc,assignment]
        try:
            server = RemoteAccessServer(
                host="127.0.0.1",
                port=12345,
                work_dir=self._tmpdir.name,
                uds_path=f"{self._tmpdir.name}/sorcar.sock",
            )
            with self.assertRaises(SystemExit) as ctx:
                await server._setup_server()
            self.assertEqual(ctx.exception.code, 2)
            self.assertEqual(
                len(attempts),
                1,
                "Non-retryable errno must not be retried; got "
                f"{len(attempts)} attempts",
            )
        finally:
            ws_mod.serve = real_serve  # type: ignore[misc,assignment]


class TestRetryableErrnoIsRetried(_BindRetryTestBase):
    """Retryable OSError exhausts the configured attempt count."""

    async def test_retryable_errno_exhausts_attempts(self) -> None:
        """``EADDRINUSE`` from ``serve`` must be retried, then SystemExit."""
        import errno as _errno

        attempts: list[int] = []
        real_serve = ws_mod.serve

        async def _fake_serve(*args: object, **kwargs: object) -> object:
            attempts.append(1)
            raise OSError(_errno.EADDRINUSE, "Address already in use")

        ws_mod.serve = _fake_serve  # type: ignore[misc,assignment]
        try:
            server = RemoteAccessServer(
                host="127.0.0.1",
                port=12345,
                work_dir=self._tmpdir.name,
                uds_path=f"{self._tmpdir.name}/sorcar.sock",
            )
            with self.assertRaises(SystemExit) as ctx:
                await server._setup_server()
            self.assertEqual(ctx.exception.code, 2)
            self.assertEqual(
                len(attempts),
                ws_mod._BIND_RETRY_ATTEMPTS,
                f"Expected {ws_mod._BIND_RETRY_ATTEMPTS} bind attempts; "
                f"got {len(attempts)}",
            )
        finally:
            ws_mod.serve = real_serve  # type: ignore[misc,assignment]


if __name__ == "__main__":
    import unittest
    unittest.main()
