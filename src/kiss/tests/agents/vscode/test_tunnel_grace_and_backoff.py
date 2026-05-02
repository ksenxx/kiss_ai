"""Tests for the tunnel watchdog startup-grace period and restart backoff.

Two failure modes burned through trycloudflare.com quick-tunnels and
triggered HTTP 429 rate-limits:

1.  The watchdog killed healthy-but-slow-to-register tunnels because
    ``readyConnections=0`` during the registration window counted as
    "unhealthy".  After three ticks (90 s) the tunnel was force-restarted
    even though it would have registered shortly.
2.  When ``cloudflared`` exited with a 429, the watchdog immediately
    spawned a replacement, which also got 429, in a tight loop.

These tests verify the fixes:

*  A startup grace period (:data:`_TUNNEL_STARTUP_GRACE`) suppresses
   metrics-based unhealthy counting for the first N seconds after a
   tunnel process starts.
*  Failed restarts schedule an exponential backoff
   (:data:`_TUNNEL_BACKOFF_INITIAL` doubling up to
   :data:`_TUNNEL_BACKOFF_MAX`) so the watchdog stops hammering
   Cloudflare when rate-limited.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from kiss.agents.vscode.web_server import (
    _TUNNEL_BACKOFF_INITIAL,
    _TUNNEL_BACKOFF_MAX,
    _TUNNEL_STARTUP_GRACE,
    RemoteAccessServer,
    _tunnel_backoff_delay,
)


def _write_fake_cloudflared(
    tmpdir: Path, stderr_lines: list[str], exit_code: int = 0,
    sleep_after: float = 0.0,
) -> Path:
    """Write a fake ``cloudflared`` shell script.

    The script writes *stderr_lines* to stderr (one per line), optionally
    sleeps, then exits with *exit_code*.
    """
    body = "#!/bin/sh\n"
    for line in stderr_lines:
        body += f"printf '%s\\n' {json.dumps(line)} >&2\n"
    if sleep_after > 0:
        body += f"sleep {sleep_after}\n"
    body += f"exit {exit_code}\n"
    script = tmpdir / "cloudflared"
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


class TestBackoffDelaySchedule(unittest.TestCase):
    """The pure ``_tunnel_backoff_delay`` helper grows exponentially."""

    def test_zero_failures_returns_zero(self) -> None:
        """With no recorded failures the watchdog can retry immediately."""
        self.assertEqual(_tunnel_backoff_delay(0), 0)

    def test_first_failure_uses_initial_delay(self) -> None:
        """After a single failure the delay equals the initial constant."""
        self.assertEqual(_tunnel_backoff_delay(1), _TUNNEL_BACKOFF_INITIAL)

    def test_doubles_on_each_failure(self) -> None:
        """Each subsequent failure doubles the delay."""
        self.assertEqual(
            _tunnel_backoff_delay(2), _TUNNEL_BACKOFF_INITIAL * 2,
        )
        self.assertEqual(
            _tunnel_backoff_delay(3), _TUNNEL_BACKOFF_INITIAL * 4,
        )
        self.assertEqual(
            _tunnel_backoff_delay(4), _TUNNEL_BACKOFF_INITIAL * 8,
        )

    def test_caps_at_max(self) -> None:
        """The delay never exceeds the max-backoff constant."""
        # Pick a failure count large enough to overflow the cap.
        self.assertEqual(
            _tunnel_backoff_delay(20), _TUNNEL_BACKOFF_MAX,
        )


class TestStartupGracePeriod(unittest.IsolatedAsyncioTestCase):
    """During the startup grace window, unhealthy ticks must not advance.

    The watchdog probes ``cloudflared``'s ``/ready`` endpoint to detect
    edge deregistration (``readyConnections=0``).  During the first few
    seconds after the subprocess starts, the tunnel is still registering
    and that probe legitimately reports zero ready connections.  The
    watchdog must not treat that as a failure.
    """

    async def asyncSetUp(self) -> None:
        self._loop = asyncio.get_event_loop()
        self._proc = subprocess.Popen(
            ["sleep", "5"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.server = RemoteAccessServer(use_tunnel=False)
        self.server._loop = self._loop
        self.server._tunnel_proc = self._proc
        # Pretend cloudflared is exposing metrics on a port that has
        # nothing listening, so _probe_tunnel_ready returns False
        # whenever it actually runs.
        self.server._tunnel_metrics_port = 1  # closed port

    async def asyncTearDown(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self.server._tunnel_proc = None

    async def test_grace_period_skips_metrics_check(self) -> None:
        """Within the grace window, unhealthy_ticks stays at 0."""
        self.server._tunnel_started_at = time.monotonic()
        self.server._tunnel_unhealthy_ticks = 0
        await self.server._check_and_restart_tunnel()
        self.assertEqual(self.server._tunnel_unhealthy_ticks, 0)

    async def test_after_grace_period_metrics_check_runs(self) -> None:
        """Outside the grace window, unhealthy_ticks increments."""
        self.server._tunnel_started_at = (
            time.monotonic() - _TUNNEL_STARTUP_GRACE - 1
        )
        self.server._tunnel_unhealthy_ticks = 0
        await self.server._check_and_restart_tunnel()
        self.assertEqual(self.server._tunnel_unhealthy_ticks, 1)


class TestRestartBackoff(unittest.IsolatedAsyncioTestCase):
    """Failed restarts must schedule exponentially-growing retry delays.

    When the local ``cloudflared`` subprocess dies and the replacement
    immediately fails to start (e.g. trycloudflare.com returned 429),
    the watchdog must stop retrying for an exponentially-growing window
    so the IP is not banned indefinitely.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._tmpdir = Path(self._tmp.name)
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{self._tmpdir}{os.pathsep}{self._old_path}"

    def tearDown(self) -> None:
        os.environ["PATH"] = self._old_path
        self._tmp.cleanup()

    async def _make_server_with_dead_proc(self) -> RemoteAccessServer:
        """Build a server whose tunnel subprocess has already exited."""
        srv = RemoteAccessServer(use_tunnel=False)
        srv._loop = asyncio.get_event_loop()
        # Use /bin/true (or a one-shot cloudflared replacement) that
        # exits immediately so poll() returns a non-None rc on the
        # first watchdog tick.
        proc = subprocess.Popen(
            ["sh", "-c", "exit 1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        proc.wait()
        srv._tunnel_proc = proc
        srv._tunnel_metrics_port = 1
        srv._tunnel_started_at = time.monotonic() - 600
        return srv

    async def test_failed_restart_sets_backoff_window(self) -> None:
        """When _start_tunnel returns None, _tunnel_next_retry is set."""
        # Fake cloudflared that exits 1 with no output → returns None.
        _write_fake_cloudflared(
            self._tmpdir, ["ERR connect: 429"], exit_code=1,
        )
        srv = await self._make_server_with_dead_proc()
        before = time.monotonic()
        await srv._check_and_restart_tunnel()
        self.assertEqual(srv._tunnel_failure_count, 1)
        self.assertGreaterEqual(
            srv._tunnel_next_retry, before + _TUNNEL_BACKOFF_INITIAL - 0.5,
        )

    async def test_consecutive_failures_grow_backoff(self) -> None:
        """Repeated failures double the delay each time."""
        _write_fake_cloudflared(
            self._tmpdir, ["ERR connect: 429"], exit_code=1,
        )
        srv = await self._make_server_with_dead_proc()
        # Manually pretend two prior failures already happened.
        srv._tunnel_failure_count = 2
        # Force out of any pre-existing backoff window so the watchdog
        # actually attempts a restart this tick.
        srv._tunnel_next_retry = 0.0
        before = time.monotonic()
        await srv._check_and_restart_tunnel()
        self.assertEqual(srv._tunnel_failure_count, 3)
        self.assertGreaterEqual(
            srv._tunnel_next_retry,
            before + _TUNNEL_BACKOFF_INITIAL * 4 - 0.5,
        )

    async def test_in_backoff_window_skips_restart(self) -> None:
        """While inside the backoff window, no restart is attempted."""
        srv = RemoteAccessServer(use_tunnel=False)
        srv._loop = asyncio.get_event_loop()
        srv._tunnel_proc = None
        srv._tunnel_failure_count = 1
        srv._tunnel_next_retry = time.monotonic() + 600
        # No fake cloudflared on PATH at all — if the watchdog were to
        # ignore the backoff and call _start_tunnel, it would block
        # for 30 s on stderr readline.  The fact that this test
        # finishes quickly proves the backoff short-circuited.
        before = time.monotonic()
        await srv._check_and_restart_tunnel()
        self.assertLess(time.monotonic() - before, 1)
        # State unchanged.
        self.assertEqual(srv._tunnel_failure_count, 1)
        self.assertIsNone(srv._tunnel_proc)


class TestSuccessfulRestartResetsBackoff(unittest.IsolatedAsyncioTestCase):
    """A successful restart must reset the failure counter."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._tmpdir = Path(self._tmp.name)
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{self._tmpdir}{os.pathsep}{self._old_path}"

    def tearDown(self) -> None:
        os.environ["PATH"] = self._old_path
        self._tmp.cleanup()

    async def test_success_resets_failure_count(self) -> None:
        """After a successful tunnel start, failure count returns to 0."""
        # Fake cloudflared that prints a recognizable URL and stays alive.
        _write_fake_cloudflared(
            self._tmpdir,
            [
                "INF Starting tunnel",
                "https://test-success.trycloudflare.com",
            ],
            sleep_after=10,
        )
        srv = RemoteAccessServer(use_tunnel=False)
        srv._loop = asyncio.get_event_loop()
        srv._tunnel_proc = None
        srv._tunnel_failure_count = 3
        srv._tunnel_next_retry = 0.0
        srv._local_url = "https://localhost:8787"
        try:
            await srv._check_and_restart_tunnel()
            self.assertEqual(srv._tunnel_failure_count, 0)
            self.assertEqual(srv._tunnel_next_retry, 0.0)
            self.assertEqual(
                srv._active_url, "https://test-success.trycloudflare.com",
            )
        finally:
            srv._stop_tunnel()


class TestStartedAtSetByStartHelpers(unittest.TestCase):
    """``_start_quick_tunnel`` and ``_start_named_tunnel`` must record start time."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._tmpdir = Path(self._tmp.name)
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{self._tmpdir}{os.pathsep}{self._old_path}"

    def tearDown(self) -> None:
        os.environ["PATH"] = self._old_path
        self._tmp.cleanup()

    def test_quick_tunnel_records_start_time(self) -> None:
        """_start_quick_tunnel sets _tunnel_started_at to a recent time."""
        _write_fake_cloudflared(
            self._tmpdir,
            ["https://recorded-start.trycloudflare.com"],
            sleep_after=10,
        )
        srv = RemoteAccessServer(use_tunnel=False)
        try:
            before = time.monotonic()
            url = srv._start_quick_tunnel()
            after = time.monotonic()
            self.assertEqual(url, "https://recorded-start.trycloudflare.com")
            self.assertIsNotNone(srv._tunnel_started_at)
            assert srv._tunnel_started_at is not None
            self.assertGreaterEqual(srv._tunnel_started_at, before)
            self.assertLessEqual(srv._tunnel_started_at, after)
        finally:
            srv._stop_tunnel()

    def test_named_tunnel_records_start_time(self) -> None:
        """_start_named_tunnel sets _tunnel_started_at to a recent time."""
        _write_fake_cloudflared(
            self._tmpdir,
            [
                "INF Starting tunnel",
                "INF Registered tunnel connection connIndex=0",
            ],
            sleep_after=10,
        )
        srv = RemoteAccessServer(
            use_tunnel=False,
            tunnel_token="dummy",
            tunnel_url="https://named.example.com",
        )
        try:
            before = time.monotonic()
            url = srv._start_named_tunnel()
            after = time.monotonic()
            self.assertEqual(url, "https://named.example.com")
            self.assertIsNotNone(srv._tunnel_started_at)
            assert srv._tunnel_started_at is not None
            self.assertGreaterEqual(srv._tunnel_started_at, before)
            self.assertLessEqual(srv._tunnel_started_at, after)
        finally:
            srv._stop_tunnel()


if __name__ == "__main__":
    unittest.main()
