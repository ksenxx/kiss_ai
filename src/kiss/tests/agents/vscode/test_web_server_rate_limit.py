"""Integration tests for Cloudflare quick-tunnel rate-limit detection.

When cloudflared's stderr reports HTTP 429 / Cloudflare error code
1015 ("rate-limited"), the watchdog must apply a much longer backoff
than the regular 60s exponential schedule.  Without this, every retry
within Cloudflare's per-IP cooldown window resets the rate-limit
clock and the tunnel can stay unreachable for hours while burning
through dozens of distinct ``*.trycloudflare.com`` URLs.

These tests exercise:

* :func:`_is_rate_limit_line` recognises the documented 1015 / 429
  signal lines and ignores normal log lines.
* :func:`_rate_limit_backoff_seconds` always returns at least
  :data:`_TUNNEL_RATE_LIMIT_BACKOFF` and adds bounded jitter.
* :func:`_stderr_reader_loop` and :func:`_read_url_from_stderr` accept
  the new optional ``rate_limit_flag`` parameter and set the flag on
  matching lines without perturbing URL extraction.
* :meth:`RemoteAccessServer._start_quick_tunnel` sets
  ``_tunnel_rate_limited = True`` when cloudflared exits after writing
  a 1015 / 429 line, and leaves it ``False`` for unrelated failures.
* :meth:`RemoteAccessServer._restart_tunnel_url` schedules
  ``_tunnel_next_retry`` at least :data:`_TUNNEL_RATE_LIMIT_BACKOFF`
  seconds in the future when the flag is set, falls back to the
  regular exponential backoff otherwise, and clears the flag in both
  the success and the long-backoff branches so it cannot lock the
  watchdog out forever.

All tests use real subprocesses (``/bin/bash`` scripts) and the real
``RemoteAccessServer`` code paths — no mocks.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import subprocess
import tempfile
import time
import unittest
from unittest import IsolatedAsyncioTestCase

from kiss.agents.vscode import web_server as ws_mod
from kiss.agents.vscode.vscode_config import CONFIG_PATH, save_config
from kiss.agents.vscode.web_server import (
    RemoteAccessServer,
    _is_rate_limit_line,
    _parse_quick_tunnel_url,
    _rate_limit_backoff_seconds,
    _read_url_from_stderr,
    _stderr_reader_loop,
    _tunnel_backoff_delay,
)


def _find_free_port() -> int:
    """Return a free TCP port on localhost."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


class _ConfigSnapshot:
    """Save and restore ~/.kiss/config.json across a test."""

    def __init__(self) -> None:
        self._content: bytes | None = None

    def __enter__(self) -> _ConfigSnapshot:
        if CONFIG_PATH.exists():
            self._content = CONFIG_PATH.read_bytes()
        return self

    def __exit__(self, *_a: object) -> None:
        if self._content is not None:
            CONFIG_PATH.write_bytes(self._content)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------


class TestIsRateLimitLine(unittest.TestCase):
    """``_is_rate_limit_line`` must match documented Cloudflare signals."""

    def test_matches_error_code_1015(self) -> None:
        line = (
            'ERR Error unmarshaling QuickTunnel response: error code: '
            '1015 error="invalid character \'e\' looking for beginning '
            'of value" originalError="error code: 1015"'
        )
        self.assertTrue(_is_rate_limit_line(line))

    def test_matches_429_too_many_requests(self) -> None:
        line = 'INF status_code="429 Too Many Requests" url=...'
        self.assertTrue(_is_rate_limit_line(line))

    def test_matches_uppercase_429_phrase(self) -> None:
        self.assertTrue(_is_rate_limit_line("429 Too Many Requests"))

    def test_matches_lowercase_phrase(self) -> None:
        self.assertTrue(_is_rate_limit_line("connection rate limited"))

    def test_does_not_match_clean_log_line(self) -> None:
        clean = (
            "INF +--------------------------------------------------+\n"
        )
        self.assertFalse(_is_rate_limit_line(clean))

    def test_does_not_match_url_announcement(self) -> None:
        line = (
            "INF |  https://random-words-12345.trycloudflare.com  |"
        )
        self.assertFalse(_is_rate_limit_line(line))

    def test_does_not_match_other_4xx(self) -> None:
        # 404, 500, etc. are not rate-limit signals.
        self.assertFalse(_is_rate_limit_line("status_code=404"))
        self.assertFalse(_is_rate_limit_line("status_code=503"))

    def test_does_not_match_empty_line(self) -> None:
        self.assertFalse(_is_rate_limit_line(""))


class TestRateLimitBackoffSeconds(unittest.TestCase):
    """``_rate_limit_backoff_seconds`` returns a long, jittered delay."""

    def test_at_least_module_baseline(self) -> None:
        for _ in range(40):
            self.assertGreaterEqual(
                _rate_limit_backoff_seconds(),
                ws_mod._TUNNEL_RATE_LIMIT_BACKOFF,
            )

    def test_at_most_baseline_plus_jitter(self) -> None:
        ceiling = (
            ws_mod._TUNNEL_RATE_LIMIT_BACKOFF
            + ws_mod._TUNNEL_RATE_LIMIT_JITTER
        )
        for _ in range(40):
            self.assertLessEqual(_rate_limit_backoff_seconds(), ceiling)

    def test_baseline_meets_10_minute_minimum(self) -> None:
        # The task spec mandates a "10+ minute" backoff.  600 seconds.
        self.assertGreaterEqual(
            ws_mod._TUNNEL_RATE_LIMIT_BACKOFF,
            600,
            "rate-limit backoff must be at least 10 minutes",
        )

    def test_jitter_observed_across_calls(self) -> None:
        # With a 300s jitter window, 40 trials should not all collide.
        values = {_rate_limit_backoff_seconds() for _ in range(40)}
        self.assertGreater(
            len(values),
            1,
            "jitter must produce more than one value across 40 trials",
        )

    def test_long_backoff_dominates_normal_backoff(self) -> None:
        """The rate-limit backoff is longer than the worst normal one."""
        # _TUNNEL_BACKOFF_MAX (1800s) > baseline (900s) so this isn't
        # a strict ordering, but for the typical first-few-failure
        # case the rate-limit floor must dwarf the exponential one.
        normal_first = _tunnel_backoff_delay(1)
        normal_second = _tunnel_backoff_delay(2)
        self.assertGreater(
            ws_mod._TUNNEL_RATE_LIMIT_BACKOFF, normal_first * 5,
        )
        self.assertGreater(
            ws_mod._TUNNEL_RATE_LIMIT_BACKOFF, normal_second * 3,
        )


# ---------------------------------------------------------------------------
# Stderr reader rate_limit_flag plumbing
# ---------------------------------------------------------------------------


class TestStderrReaderLoopSignature(unittest.TestCase):
    """The optional ``rate_limit_flag`` parameter must be supported."""

    def test_stderr_reader_loop_accepts_rate_limit_flag(self) -> None:
        sig = inspect.signature(_stderr_reader_loop)
        self.assertIn("rate_limit_flag", sig.parameters)
        # Default must be None so old call-sites keep working.
        self.assertIsNone(
            sig.parameters["rate_limit_flag"].default,
            "rate_limit_flag must default to None",
        )

    def test_read_url_from_stderr_accepts_rate_limit_flag(self) -> None:
        sig = inspect.signature(_read_url_from_stderr)
        self.assertIn("rate_limit_flag", sig.parameters)
        self.assertIsNone(
            sig.parameters["rate_limit_flag"].default,
            "rate_limit_flag must default to None",
        )


class TestReadUrlFromStderrFlag(unittest.TestCase):
    """End-to-end: ``_read_url_from_stderr`` sets the flag from stderr."""

    def _spawn_bash(self, body: str) -> subprocess.Popen[str]:
        return subprocess.Popen(
            ["/bin/bash", "-c", body],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_flag_set_on_rate_limit_line_no_url(self) -> None:
        proc = self._spawn_bash(
            'echo "ERR error code: 1015 status_code=\\"429 Too Many '
            'Requests\\"" >&2; sleep 0.05',
        )
        try:
            flag = [False]
            url = _read_url_from_stderr(
                proc, _parse_quick_tunnel_url, timeout=2.0,
                rate_limit_flag=flag,
            )
            self.assertIsNone(url)
            self.assertTrue(flag[0])
        finally:
            proc.terminate()
            proc.wait(timeout=2)

    def test_flag_clear_on_clean_failure_no_url(self) -> None:
        proc = self._spawn_bash(
            'echo "INF starting tunnel" >&2; '
            'echo "ERR connection refused" >&2; '
            'sleep 0.05',
        )
        try:
            flag = [False]
            url = _read_url_from_stderr(
                proc, _parse_quick_tunnel_url, timeout=2.0,
                rate_limit_flag=flag,
            )
            self.assertIsNone(url)
            self.assertFalse(flag[0])
        finally:
            proc.terminate()
            proc.wait(timeout=2)

    def test_flag_set_then_url_found(self) -> None:
        # Even if a 1015 line arrives BEFORE a real URL (unlikely in
        # practice but the loop must not be confused by it), both the
        # URL and the flag must be reported.
        proc = self._spawn_bash(
            'echo "ERR error code: 1015 transient" >&2; '
            'echo "INF |  https://example-tunnel.trycloudflare.com  |" '
            ">&2; sleep 1",
        )
        try:
            flag = [False]
            url = _read_url_from_stderr(
                proc, _parse_quick_tunnel_url, timeout=3.0,
                rate_limit_flag=flag,
            )
            self.assertEqual(
                url, "https://example-tunnel.trycloudflare.com",
            )
            self.assertTrue(flag[0])
        finally:
            proc.terminate()
            proc.wait(timeout=2)

    def test_no_flag_arg_does_not_break_existing_callers(self) -> None:
        proc = self._spawn_bash(
            'echo "INF |  https://nf.trycloudflare.com  |" >&2; sleep 1',
        )
        try:
            url = _read_url_from_stderr(
                proc, _parse_quick_tunnel_url, timeout=3.0,
            )
            self.assertEqual(url, "https://nf.trycloudflare.com")
        finally:
            proc.terminate()
            proc.wait(timeout=2)


# ---------------------------------------------------------------------------
# _start_quick_tunnel marks RemoteAccessServer when rate-limited
# ---------------------------------------------------------------------------


class TestStartQuickTunnelMarksRateLimit(IsolatedAsyncioTestCase):
    """Rate-limit lines on cloudflared stderr propagate to the server."""

    async def asyncSetUp(self) -> None:
        self._snap = _ConfigSnapshot().__enter__()
        save_config({"remote_password": "rl-test"})
        self._tmpdir = tempfile.mkdtemp()
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = self._tmpdir + ":" + self._old_path
        self.port = _find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            use_tunnel=False,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        os.environ["PATH"] = self._old_path
        if self.server._tunnel_proc is not None:
            try:
                self.server._tunnel_proc.terminate()
                self.server._tunnel_proc.wait(timeout=5)
            except Exception:
                try:
                    self.server._tunnel_proc.kill()
                except Exception:
                    pass
        await self.server.stop_async()
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._snap.__exit__()

    def _install_fake_cloudflared(self, body: str) -> None:
        cf = os.path.join(self._tmpdir, "cloudflared")
        with open(cf, "w") as f:
            f.write("#!/bin/bash\n" + body)
        os.chmod(cf, 0o755)

    async def test_rate_limit_line_marks_server(self) -> None:
        # Fake cloudflared that reproduces the user-reported failure:
        # writes a 1015 / 429 line to stderr and exits ~1s later
        # without emitting a tunnel URL.
        self._install_fake_cloudflared(
            'echo "ERR Error unmarshaling QuickTunnel response: '
            'error code: 1015 error=\\"invalid character e\\" '
            'status_code=\\"429 Too Many Requests\\"" >&2\n'
            "exit 1\n",
        )
        self.assertFalse(self.server._tunnel_rate_limited)
        url = await asyncio.get_event_loop().run_in_executor(
            None, self.server._start_quick_tunnel,
        )
        self.assertIsNone(url)
        self.assertTrue(
            self.server._tunnel_rate_limited,
            "1015/429 in stderr must set _tunnel_rate_limited",
        )

    async def test_clean_failure_does_not_mark_server(self) -> None:
        # cloudflared writes nothing rate-limit-shaped, exits.
        self._install_fake_cloudflared(
            'echo "INF starting tunnel" >&2\n'
            'echo "ERR could not bind to port" >&2\n'
            "exit 2\n",
        )
        url = await asyncio.get_event_loop().run_in_executor(
            None, self.server._start_quick_tunnel,
        )
        self.assertIsNone(url)
        self.assertFalse(
            self.server._tunnel_rate_limited,
            "non-rate-limit failure must not mark _tunnel_rate_limited",
        )

    async def test_successful_start_does_not_mark_server(self) -> None:
        self._install_fake_cloudflared(
            'echo "INF |  https://ok-1234.trycloudflare.com  |" >&2\n'
            "sleep 60\n",
        )
        url = await asyncio.get_event_loop().run_in_executor(
            None, self.server._start_quick_tunnel,
        )
        self.assertEqual(url, "https://ok-1234.trycloudflare.com")
        self.assertFalse(self.server._tunnel_rate_limited)


# ---------------------------------------------------------------------------
# _restart_tunnel_url applies the long backoff when rate-limited
# ---------------------------------------------------------------------------


class TestRestartTunnelUrlBackoff(IsolatedAsyncioTestCase):
    """``_restart_tunnel_url`` must use a long delay on rate-limit."""

    async def asyncSetUp(self) -> None:
        self._snap = _ConfigSnapshot().__enter__()
        save_config({"remote_password": "rl-bk"})
        self._tmpdir = tempfile.mkdtemp()
        # Always-fail cloudflared so _start_tunnel returns None.
        cf = os.path.join(self._tmpdir, "cloudflared")
        with open(cf, "w") as f:
            f.write("#!/bin/bash\nexit 1\n")
        os.chmod(cf, 0o755)
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = self._tmpdir + ":" + self._old_path
        self.port = _find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            use_tunnel=False,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        os.environ["PATH"] = self._old_path
        if self.server._tunnel_proc is not None:
            try:
                self.server._tunnel_proc.terminate()
                self.server._tunnel_proc.wait(timeout=5)
            except Exception:
                try:
                    self.server._tunnel_proc.kill()
                except Exception:
                    pass
        await self.server.stop_async()
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._snap.__exit__()

    async def test_long_backoff_when_rate_limited(self) -> None:
        self.server._tunnel_rate_limited = True
        self.server._tunnel_failure_count = 0
        before = time.monotonic()
        await self.server._restart_tunnel_url()
        # Long-backoff path: at least baseline seconds, plus jitter.
        delay_floor = ws_mod._TUNNEL_RATE_LIMIT_BACKOFF
        delay_ceil = (
            ws_mod._TUNNEL_RATE_LIMIT_BACKOFF
            + ws_mod._TUNNEL_RATE_LIMIT_JITTER
        )
        scheduled = self.server._tunnel_next_retry - before
        self.assertGreaterEqual(scheduled, delay_floor - 1)
        self.assertLessEqual(scheduled, delay_ceil + 5)
        # Failure count still increments so consecutive non-rate
        # failures eventually hit the regular cap.
        self.assertEqual(self.server._tunnel_failure_count, 1)
        # Flag must be cleared so a *second* failure that is NOT a
        # rate-limit reverts to normal exponential backoff.
        self.assertFalse(self.server._tunnel_rate_limited)

    async def test_short_backoff_when_not_rate_limited(self) -> None:
        self.server._tunnel_rate_limited = False
        self.server._tunnel_failure_count = 0
        before = time.monotonic()
        await self.server._restart_tunnel_url()
        scheduled = self.server._tunnel_next_retry - before
        # First non-rate failure schedules _TUNNEL_BACKOFF_INITIAL.
        self.assertGreaterEqual(
            scheduled, ws_mod._TUNNEL_BACKOFF_INITIAL - 1,
        )
        self.assertLess(
            scheduled, ws_mod._TUNNEL_RATE_LIMIT_BACKOFF,
            "non-rate-limited failure must NOT use the long backoff",
        )

    async def test_subsequent_non_rate_failure_uses_short_path(
        self,
    ) -> None:
        # First a rate-limit failure (long backoff), flag clears.
        self.server._tunnel_rate_limited = True
        await self.server._restart_tunnel_url()
        self.assertFalse(self.server._tunnel_rate_limited)
        # Second failure (no flag): exponential schedule resumes.
        before = time.monotonic()
        await self.server._restart_tunnel_url()
        scheduled = self.server._tunnel_next_retry - before
        # _tunnel_failure_count is now 2 → 60 * 2**(2-1) = 120s.
        self.assertGreaterEqual(scheduled, 119)
        self.assertLess(scheduled, ws_mod._TUNNEL_RATE_LIMIT_BACKOFF)


if __name__ == "__main__":
    unittest.main()
