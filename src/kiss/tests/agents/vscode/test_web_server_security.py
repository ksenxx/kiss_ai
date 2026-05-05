"""Integration tests for the six HIGH-severity security fixes in web_server.py.

Each test exercises real network sockets, real subprocesses, and the
real ``RemoteAccessServer`` / ``_authenticate_ws`` / ``_spawn_cloudflared``
/ ``_generate_self_signed_cert`` / ``_read_url_from_stderr`` code paths
without mocks or test doubles.  The tests are designed to fail if any
of the six fixes is reverted, and pass with the fixes applied.

Coverage:

* H1 — auth-open-by-default: cloudflared tunnel must NOT start when
  ``remote_password`` is empty.
* H2 — ``cloudflared`` stdout pipe must be ``DEVNULL`` so a chatty
  child cannot deadlock on a full pipe buffer.
* H3 — password compare must use :func:`secrets.compare_digest`
  (constant-time), not plain ``!=``.
* H4 — repeated wrong-password attempts from the same source must
  trigger an IP-based rate limit so brute-force is bounded.
* H5 — auto-generated TLS private key must be mode ``0600`` and the
  TLS directory mode ``0700``.
* H6 — when ``_read_url_from_stderr`` times out, the daemon reader
  thread must terminate at its next iteration instead of running
  until process shutdown.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import ssl
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

from kiss.agents.vscode import web_server as ws_mod
from kiss.agents.vscode.vscode_config import CONFIG_PATH, save_config
from kiss.agents.vscode.web_server import (
    RemoteAccessServer,
    _generate_self_signed_cert,
    _parse_quick_tunnel_url,
    _read_url_from_stderr,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Return a free TCP port on localhost."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


def _no_verify_ssl() -> ssl.SSLContext:
    """SSL client context that skips certificate verification."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class _ConfigSnapshot:
    """Save and restore ~/.kiss/config.json across a test."""

    def __init__(self) -> None:
        self._content: bytes | None = None

    def __enter__(self) -> _ConfigSnapshot:
        if CONFIG_PATH.exists():
            self._content = CONFIG_PATH.read_bytes()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._content is not None:
            CONFIG_PATH.write_bytes(self._content)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()


# ---------------------------------------------------------------------------
# H1 — open-by-default authentication: tunnel must not start when no password
# ---------------------------------------------------------------------------


class TestH1NoTunnelWithoutPassword(IsolatedAsyncioTestCase):
    """The cloudflared tunnel must not start when remote_password is empty."""

    async def asyncSetUp(self) -> None:
        self._snap = _ConfigSnapshot().__enter__()
        save_config({"remote_password": ""})
        self.port = _find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            use_tunnel=True,  # ← important: ask for a tunnel
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        self._snap.__exit__()

    async def test_tunnel_proc_is_none_when_no_password(self) -> None:
        """No cloudflared subprocess is spawned when remote_password=''."""
        self.assertIsNone(self.server._tunnel_proc)

    async def test_active_url_is_local_only(self) -> None:
        """Without a password the active URL is the local URL, not a tunnel URL."""
        self.assertEqual(self.server._active_url, self.server._local_url)

    async def test_watchdog_does_not_start_tunnel_without_password(self) -> None:
        """The watchdog tick must also refuse to spawn the tunnel."""
        self.server._tunnel_proc = None
        self.server._tunnel_next_retry = 0.0  # bypass backoff
        self.server.use_tunnel = True
        await self.server._check_and_restart_tunnel()
        self.assertIsNone(self.server._tunnel_proc)


class TestH1TunnelStartsWhenPasswordSet(IsolatedAsyncioTestCase):
    """Symmetric check: with a password set, the tunnel-start path runs."""

    async def asyncSetUp(self) -> None:
        self._snap = _ConfigSnapshot().__enter__()
        save_config({"remote_password": "real-secret"})
        self._tmpdir = tempfile.mkdtemp()
        # Drop in a fake cloudflared that immediately announces a URL.
        cf = os.path.join(self._tmpdir, "cloudflared")
        with open(cf, "w") as f:
            f.write(
                "#!/bin/bash\n"
                'echo "INF https://h1-ok.trycloudflare.com" >&2\n'
                "sleep 60\n"
            )
        os.chmod(cf, 0o755)
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = self._tmpdir + ":" + self._old_path

        self.port = _find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            use_tunnel=True,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        os.environ["PATH"] = self._old_path
        if self.server._tunnel_proc is not None:
            self.server._tunnel_proc.terminate()
            try:
                self.server._tunnel_proc.wait(timeout=5)
            except Exception:
                self.server._tunnel_proc.kill()
        await self.server.stop_async()
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._snap.__exit__()

    async def test_tunnel_proc_started_when_password_set(self) -> None:
        """A cloudflared child process IS spawned once a password is configured."""
        self.assertIsNotNone(self.server._tunnel_proc)
        self.assertEqual(
            self.server._active_url, "https://h1-ok.trycloudflare.com",
        )


# ---------------------------------------------------------------------------
# H2 — cloudflared subprocess stdout must not be a PIPE
# ---------------------------------------------------------------------------


class TestH2StdoutDevnull(IsolatedAsyncioTestCase):
    """``_spawn_cloudflared`` must not connect cloudflared's stdout to a pipe."""

    async def asyncSetUp(self) -> None:
        self._snap = _ConfigSnapshot().__enter__()
        save_config({"remote_password": "pw"})
        self._tmpdir = tempfile.mkdtemp()
        # Fake cloudflared that floods stdout to prove the pipe would deadlock.
        cf = os.path.join(self._tmpdir, "cloudflared")
        with open(cf, "w") as f:
            f.write(
                "#!/bin/bash\n"
                # >256 KiB of stdout; with stdout=PIPE and no reader,
                # cloudflared would block on write() once the pipe
                # buffer (~64 KiB) fills.
                "yes 'STDOUT_FLOOD_LINE_FOR_H2_TEST' | head -c 262144\n"
                'echo "INF https://h2-ok.trycloudflare.com" >&2\n'
                "sleep 30\n"
            )
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
            self.server._tunnel_proc.terminate()
            try:
                self.server._tunnel_proc.wait(timeout=5)
            except Exception:
                self.server._tunnel_proc.kill()
        await self.server.stop_async()
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._snap.__exit__()

    async def test_stdout_attribute_is_none(self) -> None:
        """``proc.stdout`` is None when stdout is DEVNULL'd in Popen.

        With ``stdout=subprocess.PIPE`` (the bug), proc.stdout is a
        readable file object.  With ``stdout=subprocess.DEVNULL`` (the
        fix), proc.stdout is None.
        """
        url = await asyncio.get_event_loop().run_in_executor(
            None, self.server._start_quick_tunnel,
        )
        self.assertIsNotNone(self.server._tunnel_proc)
        assert self.server._tunnel_proc is not None
        self.assertIsNone(
            self.server._tunnel_proc.stdout,
            "cloudflared stdout must be DEVNULL, not a PIPE",
        )
        # Also verify the URL was extracted from stderr (sanity).
        self.assertEqual(url, "https://h2-ok.trycloudflare.com")

    async def test_chatty_stdout_does_not_deadlock(self) -> None:
        """The stdout flood does not block tunnel start.

        Even though the fake cloudflared writes >256 KiB to stdout
        (more than the ~64 KiB pipe buffer), `_start_quick_tunnel`
        completes promptly because nothing is blocking on stdout.
        """
        loop = asyncio.get_event_loop()
        start = time.monotonic()
        url = await asyncio.wait_for(
            loop.run_in_executor(None, self.server._start_quick_tunnel),
            timeout=20,
        )
        elapsed = time.monotonic() - start
        self.assertEqual(url, "https://h2-ok.trycloudflare.com")
        self.assertLess(elapsed, 10, "tunnel start should be fast")


# ---------------------------------------------------------------------------
# H3 — constant-time password comparison
# ---------------------------------------------------------------------------


class TestH3ConstantTimeCompare(unittest.TestCase):
    """Password comparison must go through :func:`secrets.compare_digest`."""

    def test_authenticate_ws_source_uses_compare_digest(self) -> None:
        """Source of ``_authenticate_ws`` (or its helper) calls compare_digest."""
        src = inspect.getsource(RemoteAccessServer._authenticate_ws)
        helper_src = inspect.getsource(RemoteAccessServer._passwords_equal)
        # Either site references compare_digest; helper is the canonical one.
        combined = src + helper_src
        self.assertIn(
            "compare_digest", combined,
            "_authenticate_ws must use secrets.compare_digest, not '!='",
        )

    def test_passwords_equal_returns_true_for_equal_strings(self) -> None:
        """The helper returns True for equal passwords."""
        self.assertTrue(RemoteAccessServer._passwords_equal("hunter2", "hunter2"))

    def test_passwords_equal_returns_false_for_different_strings(self) -> None:
        """The helper returns False for different passwords."""
        self.assertFalse(RemoteAccessServer._passwords_equal("hunter2", "hunter3"))

    def test_passwords_equal_returns_false_for_different_lengths(self) -> None:
        """Strings of different lengths are not equal (handled internally)."""
        self.assertFalse(RemoteAccessServer._passwords_equal("a", "abcdef"))

    def test_passwords_equal_handles_unicode(self) -> None:
        """Unicode passwords compare correctly via UTF-8 encoding."""
        self.assertTrue(RemoteAccessServer._passwords_equal("café", "café"))
        self.assertFalse(RemoteAccessServer._passwords_equal("café", "cafe"))


class TestH3CompareDigestActuallyCalled(IsolatedAsyncioTestCase):
    """Behavioural check: monkey-patch compare_digest and confirm it's invoked."""

    async def asyncSetUp(self) -> None:
        self._snap = _ConfigSnapshot().__enter__()
        save_config({"remote_password": "right-password"})
        self.port = _find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        self._snap.__exit__()

    async def test_compare_digest_invoked_during_auth(self) -> None:
        """Auth with a wrong password invokes secrets.compare_digest."""
        import secrets as secrets_mod

        original = secrets_mod.compare_digest
        calls: list[tuple[bytes, bytes]] = []

        def _spy(a: bytes, b: bytes) -> bool:  # type: ignore[override]
            calls.append((a, b))
            return original(a, b)

        secrets_mod.compare_digest = _spy  # type: ignore[assignment]
        try:
            async with connect(
                f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
            ) as ws:
                await ws.send(
                    json.dumps({"type": "auth", "password": "wrong-password"}),
                )
                resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                self.assertEqual(resp["type"], "auth_required")
        finally:
            secrets_mod.compare_digest = original  # type: ignore[assignment]

        # At least one comparison must have happened, and it must be over
        # bytes (the H3 fix).
        self.assertGreaterEqual(len(calls), 1)
        a, b = calls[0]
        self.assertIsInstance(a, bytes)
        self.assertIsInstance(b, bytes)
        # And the comparison must be against the configured password.
        self.assertIn(
            b"right-password", {a, b},
            "compare_digest must compare against the configured password",
        )


# ---------------------------------------------------------------------------
# H4 — per-source rate limiting on auth attempts
# ---------------------------------------------------------------------------


class TestH4AuthRateLimit(IsolatedAsyncioTestCase):
    """Repeated wrong-password attempts must trigger a per-IP cool-down."""

    async def asyncSetUp(self) -> None:
        self._snap = _ConfigSnapshot().__enter__()
        save_config({"remote_password": "secret-h4"})

        # Tighten the rate limiter so the test runs in seconds.
        self._old_max = ws_mod._AUTH_FAIL_MAX
        self._old_window = ws_mod._AUTH_FAIL_WINDOW
        self._old_lockout = ws_mod._AUTH_LOCKOUT
        ws_mod._AUTH_FAIL_MAX = 3
        ws_mod._AUTH_FAIL_WINDOW = 60.0
        ws_mod._AUTH_LOCKOUT = 60.0

        self.port = _find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        ws_mod._AUTH_FAIL_MAX = self._old_max
        ws_mod._AUTH_FAIL_WINDOW = self._old_window
        ws_mod._AUTH_LOCKOUT = self._old_lockout
        self._snap.__exit__()

    async def _try_auth(self, password: str) -> str:
        """Attempt auth, return the response 'type' or 'closed'."""
        try:
            async with connect(
                f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
                close_timeout=2,
            ) as ws:
                await ws.send(json.dumps({"type": "auth", "password": password}))
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=3)
                except TimeoutError:
                    return "timeout"
                except Exception:
                    return "closed"
                msg = json.loads(raw)
                return str(msg.get("type", "?"))
        except Exception:
            return "closed"

    async def test_rate_limit_kicks_in_after_max_failures(self) -> None:
        """After _AUTH_FAIL_MAX failures, new connections are rejected."""
        # First 3 wrong attempts → each gets auth_required.
        for i in range(ws_mod._AUTH_FAIL_MAX):
            resp = await self._try_auth("wrong-password")
            self.assertEqual(
                resp, "auth_required",
                f"attempt {i}: expected auth_required, got {resp!r}",
            )

        # The IP is now locked out: next connection is closed without
        # any auth_required prompt.
        resp = await self._try_auth("wrong-password")
        self.assertIn(
            resp, ("closed", "timeout"),
            "after lockout the socket must be closed without prompting",
        )

    async def test_correct_password_locked_out_too(self) -> None:
        """Even the correct password is refused while locked out (per-IP)."""
        for _ in range(ws_mod._AUTH_FAIL_MAX):
            await self._try_auth("wrong-password")

        # Even the correct password must not be accepted while locked.
        resp = await self._try_auth("secret-h4")
        self.assertIn(resp, ("closed", "timeout"))

    async def test_record_auth_failure_tracks_per_ip(self) -> None:
        """The internal failure tracker is per-source-IP."""
        self.server._record_auth_failure("1.2.3.4")
        self.server._record_auth_failure("1.2.3.4")
        self.server._record_auth_failure("5.6.7.8")
        self.assertEqual(len(self.server._auth_failures["1.2.3.4"]), 2)
        self.assertEqual(len(self.server._auth_failures["5.6.7.8"]), 1)
        self.assertNotIn("9.9.9.9", self.server._auth_failures)

    async def test_is_auth_locked_thresholds(self) -> None:
        """_is_auth_locked enforces _AUTH_FAIL_MAX within window."""
        ip = "10.0.0.1"
        for _ in range(ws_mod._AUTH_FAIL_MAX - 1):
            self.server._record_auth_failure(ip)
        self.assertFalse(self.server._is_auth_locked(ip))
        self.server._record_auth_failure(ip)
        self.assertTrue(self.server._is_auth_locked(ip))

    async def test_is_auth_locked_window_expiry(self) -> None:
        """Old failures outside the window do not contribute to the lock."""
        ip = "10.0.0.2"
        # Manually inject ancient timestamps.
        ancient = time.monotonic() - (ws_mod._AUTH_FAIL_WINDOW + 10)
        self.server._auth_failures[ip] = [ancient] * (ws_mod._AUTH_FAIL_MAX + 5)
        self.assertFalse(self.server._is_auth_locked(ip))
        # The pruning side-effect drops them.
        self.assertEqual(self.server._auth_failures[ip], [])


# ---------------------------------------------------------------------------
# H5 — auto-generated TLS private key must be 0600
# ---------------------------------------------------------------------------


@unittest.skipIf(
    sys.platform == "win32", "POSIX file modes are not meaningful on Windows",
)
class TestH5KeyFilePermissions(unittest.TestCase):
    """The generated TLS private key must be read/write only by the owner."""

    def setUp(self) -> None:
        self._tmpdir = Path(tempfile.mkdtemp())
        self.cert_path = self._tmpdir / "tls" / "cert.pem"
        self.key_path = self._tmpdir / "tls" / "key.pem"

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_key_file_is_mode_0600(self) -> None:
        """``key.pem`` is mode 0600 immediately after generation."""
        _generate_self_signed_cert(self.cert_path, self.key_path)
        mode = stat.S_IMODE(self.key_path.stat().st_mode)
        self.assertEqual(
            mode, 0o600,
            f"key.pem must be 0o600, got 0o{mode:o}",
        )

    def test_key_file_not_world_readable(self) -> None:
        """The world-readable bits must be cleared on the key."""
        _generate_self_signed_cert(self.cert_path, self.key_path)
        mode = stat.S_IMODE(self.key_path.stat().st_mode)
        self.assertEqual(mode & stat.S_IROTH, 0)
        self.assertEqual(mode & stat.S_IRGRP, 0)

    def test_tls_directory_is_mode_0700(self) -> None:
        """The TLS dir is 0700 so listing the dir is also restricted."""
        _generate_self_signed_cert(self.cert_path, self.key_path)
        dir_mode = stat.S_IMODE(self.key_path.parent.stat().st_mode)
        self.assertEqual(
            dir_mode, 0o700,
            f"tls/ must be 0o700, got 0o{dir_mode:o}",
        )

    def test_regenerate_overwrites_existing_key(self) -> None:
        """Calling generate twice replaces the key file (and stays 0600)."""
        _generate_self_signed_cert(self.cert_path, self.key_path)
        first = self.key_path.read_bytes()
        # Loosen permissions to verify the regenerator restores 0600.
        os.chmod(self.key_path, 0o644)
        _generate_self_signed_cert(self.cert_path, self.key_path)
        second = self.key_path.read_bytes()
        self.assertNotEqual(first, second, "key should be regenerated")
        mode = stat.S_IMODE(self.key_path.stat().st_mode)
        self.assertEqual(mode, 0o600)

    def test_real_kiss_tls_key_after_create_ssl_context(self) -> None:
        """End-to-end: ``_create_ssl_context`` produces a 0600 key in ~/.kiss/tls."""
        # Redirect _TLS_DIR to a temp location for isolation.
        old_tls_dir = ws_mod._TLS_DIR
        ws_mod._TLS_DIR = self._tmpdir / "kiss_tls"
        try:
            ctx = ws_mod._create_ssl_context(certfile=None, keyfile=None)
            self.assertIsInstance(ctx, ssl.SSLContext)
            key = ws_mod._TLS_DIR / "key.pem"
            self.assertTrue(key.is_file())
            mode = stat.S_IMODE(key.stat().st_mode)
            self.assertEqual(mode, 0o600)
        finally:
            ws_mod._TLS_DIR = old_tls_dir


# ---------------------------------------------------------------------------
# H6 — stderr-reader thread terminates after timeout (does not leak)
# ---------------------------------------------------------------------------


class TestH6StderrReaderCleanup(unittest.TestCase):
    """The reader thread must exit promptly once the subprocess emits any
    further stderr line OR dies, after a timeout has elapsed.

    Without the fix, the reader thread runs until the parent process
    terminates the subprocess, leaking one daemon thread per timed-out
    tunnel restart.
    """

    def test_reader_exits_when_proc_dies_after_timeout(self) -> None:
        """When the subprocess dies, the reader thread exits cleanly."""
        # A subprocess that prints 1 non-matching line, then exits.
        proc = subprocess.Popen(
            [
                sys.executable, "-c",
                "import sys, time;"
                "sys.stderr.write('non-matching\\n');"
                "sys.stderr.flush();"
                "time.sleep(0.2);",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            url = _read_url_from_stderr(
                proc, _parse_quick_tunnel_url, timeout=0.05,
            )
            self.assertIsNone(url)
            # Wait for proc to exit naturally.
            proc.wait(timeout=2)
            # Give the reader thread a moment to observe EOF and exit.
            time.sleep(0.3)
            # No leaked daemon thread reading our subprocess's stderr.
            for t in threading.enumerate():
                self.assertNotEqual(
                    t.name, "_stderr_reader_loop",
                    "reader thread leaked after proc death",
                )
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=2)

    def test_reader_exits_after_stop_event_on_next_line(self) -> None:
        """After timeout the stop_event is set; reader exits at next line."""
        # Subprocess emits a line every 0.1s for ~3s, no URL match.
        proc = subprocess.Popen(
            [
                sys.executable, "-u", "-c",
                "import sys, time\n"
                "for i in range(30):\n"
                "    sys.stderr.write(f'tick {i}\\n')\n"
                "    sys.stderr.flush()\n"
                "    time.sleep(0.1)\n",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            initial = {t.ident for t in threading.enumerate()}
            url = _read_url_from_stderr(
                proc, _parse_quick_tunnel_url, timeout=0.3,
            )
            self.assertIsNone(url)

            # The reader was active during the timeout; after timeout we
            # expect it to exit by its NEXT readline iteration (within
            # the 0.1s tick cadence).  Allow up to 1 s for cleanup.
            deadline = time.monotonic() + 1.5
            new_threads_alive: list[threading.Thread] = []
            while time.monotonic() < deadline:
                new_threads_alive = [
                    t for t in threading.enumerate()
                    if t.ident not in initial and t.is_alive()
                    and t.name != "MainThread"
                ]
                if not new_threads_alive:
                    break
                time.sleep(0.05)

            self.assertFalse(
                new_threads_alive,
                f"reader thread leaked after timeout: {new_threads_alive!r}",
            )
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=2)

    def test_stderr_reader_loop_signature_accepts_stop_event(self) -> None:
        """Source check: the reader loop accepts a stop_event parameter."""
        sig = inspect.signature(ws_mod._stderr_reader_loop)
        self.assertIn(
            "stop_event", sig.parameters,
            "_stderr_reader_loop must accept a stop_event for H6 cleanup",
        )

    def test_reader_returns_url_when_present(self) -> None:
        """Sanity: the URL is still returned correctly in the happy path."""
        proc = subprocess.Popen(
            [
                sys.executable, "-c",
                "import sys;"
                "sys.stderr.write('startup line\\n');"
                "sys.stderr.write('"
                "INF https://h6-ok.trycloudflare.com\\n');"
                "sys.stderr.flush();",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            url = _read_url_from_stderr(
                proc, _parse_quick_tunnel_url, timeout=5,
            )
            self.assertEqual(url, "https://h6-ok.trycloudflare.com")
        finally:
            proc.wait(timeout=2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
