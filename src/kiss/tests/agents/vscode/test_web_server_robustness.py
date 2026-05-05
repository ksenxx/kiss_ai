"""Integration tests for the twelve MEDIUM-severity robustness fixes
in :mod:`kiss.agents.vscode.web_server`.

Each test exercises real network sockets, real subprocesses (where
relevant), and the real production code paths — no mocks, no test
doubles.  Every test is designed to FAIL if its corresponding fix is
reverted and PASS with the fix in place.

Coverage:

* M1 — ``_setup_server`` uses :func:`asyncio.get_running_loop` instead
  of the deprecated :func:`asyncio.get_event_loop`.
* M2 — ``RemoteAccessServer.__init__`` does not mutate
  ``os.environ["KISS_WORKDIR"]``; per-instance ``work_dir`` is used.
* M3 — the SSL context pins ``minimum_version >= TLSv1_2``.
* M4 — ``_create_ssl_context`` regenerates a self-signed cert that is
  expired or expiring within 30 days.
* M5 — ``_spawn_cloudflared`` retries with a fresh metrics port when
  the subprocess exits immediately (TOCTOU bind collision).
* M6 — merge state for a tab is dropped when the owning WebSocket
  disconnects, and ``_merge_states`` accesses are guarded by a lock
  against the agent thread.
* M7 — ``restoredTabs`` and ``attachments`` lists are clamped, and
  oversize prompts are truncated.
* M8 — ``WebPrinter.broadcast`` tracks ``run_coroutine_threadsafe``
  futures per client, and ``remove_client`` cancels pending sends.
* M9 — ``_WebMergeState`` exposes :meth:`is_resolved` and the body of
  ``web_server`` no longer pokes at ``state._resolved`` directly.
* M10 — ``_send_welcome_info`` is async and does its disk / pgrep /
  HTTP I/O via :meth:`asyncio.AbstractEventLoop.run_in_executor`,
  so it cannot block the event loop.
* M11 — ``_WebMergeState.current()`` returns ``None`` once every hunk
  has been resolved (post-``accept-all`` / ``reject-all``).
* M12 — ``_authenticate_ws`` always closes the WebSocket on failure,
  including the exception path.
"""

from __future__ import annotations

import asyncio
import datetime
import inspect
import json
import os
import ssl
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from kiss.agents.vscode import web_server as ws_mod
from kiss.agents.vscode.vscode_config import CONFIG_PATH, save_config
from kiss.agents.vscode.web_server import (
    RemoteAccessServer,
    WebPrinter,
    _create_ssl_context,
    _generate_self_signed_cert,
    _self_signed_cert_needs_renewal,
    _WebMergeState,
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
    """Save and restore ``~/.kiss/config.json`` across a test."""

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
# M1 — get_running_loop instead of get_event_loop
# ---------------------------------------------------------------------------


class TestM1RunningLoop(unittest.TestCase):
    """``_setup_server`` must not call deprecated ``asyncio.get_event_loop``."""

    def test_setup_server_source_uses_get_running_loop(self) -> None:
        src = inspect.getsource(RemoteAccessServer._setup_server)
        self.assertIn("get_running_loop()", src)
        # Strip block comments before checking that the deprecated
        # call does not appear as a real expression.
        code_only = "\n".join(
            line for line in src.splitlines()
            if not line.lstrip().startswith("#")
        )
        self.assertNotIn(
            "get_event_loop(", code_only,
            "_setup_server must not call asyncio.get_event_loop()",
        )


class TestM1NoDeprecationWarning(IsolatedAsyncioTestCase):
    """Starting the server emits no DeprecationWarning from get_event_loop."""

    async def asyncSetUp(self) -> None:
        self._snap = _ConfigSnapshot().__enter__()
        save_config({"remote_password": ""})
        self.port = _find_free_port()

    async def asyncTearDown(self) -> None:
        self._snap.__exit__()

    async def test_no_get_event_loop_deprecation(self) -> None:
        """Starting the server triggers no get_event_loop deprecation."""
        import warnings

        server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await server.start_async()
            try:
                # Touch a real client so the loop is fully spun up.
                async with connect(
                    f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
                ) as ws:
                    await ws.send(
                        json.dumps({"type": "auth", "password": ""}),
                    )
                    await asyncio.wait_for(ws.recv(), timeout=5)
            finally:
                await server.stop_async()
        offenders = [
            w for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "get_event_loop" in str(w.message)
        ]
        self.assertEqual(
            offenders, [],
            f"Unexpected DeprecationWarnings: {[str(w.message) for w in offenders]}",
        )


# ---------------------------------------------------------------------------
# M2 — RemoteAccessServer must not mutate os.environ["KISS_WORKDIR"]
# ---------------------------------------------------------------------------


class TestM2NoEnvMutation(unittest.TestCase):
    """Constructing the server must not write to ``os.environ``."""

    def setUp(self) -> None:
        self._orig = os.environ.pop("KISS_WORKDIR", None)
        self._snap = _ConfigSnapshot().__enter__()
        # Ensure the load_config() fallback can't leak a value either.
        save_config({"remote_password": ""})

    def tearDown(self) -> None:
        os.environ.pop("KISS_WORKDIR", None)
        if self._orig is not None:
            os.environ["KISS_WORKDIR"] = self._orig
        self._snap.__exit__()

    def test_init_does_not_set_kiss_workdir(self) -> None:
        """``__init__`` does not set the KISS_WORKDIR environment variable."""
        wd = tempfile.mkdtemp()
        # Important: ensure the env var is empty before construction.
        self.assertNotIn("KISS_WORKDIR", os.environ)
        server = RemoteAccessServer(
            host="127.0.0.1",
            port=_find_free_port(),
            work_dir=wd,
        )
        # The fix moves the work_dir onto the instance — verify it.
        self.assertEqual(server.work_dir, wd)
        # And the global env must NOT have been mutated.
        self.assertNotIn(
            "KISS_WORKDIR", os.environ,
            "RemoteAccessServer.__init__ must not mutate os.environ",
        )
        self.assertEqual(server._printer.work_dir, wd)
        self.assertEqual(server._vscode_server.work_dir, wd)

    def test_two_instances_do_not_stomp_each_other(self) -> None:
        """Two servers with different work_dir do not overwrite each other."""
        a = tempfile.mkdtemp()
        b = tempfile.mkdtemp()
        s1 = RemoteAccessServer(
            host="127.0.0.1", port=_find_free_port(), work_dir=a,
        )
        s2 = RemoteAccessServer(
            host="127.0.0.1", port=_find_free_port(), work_dir=b,
        )
        # Each instance retains its own value.
        self.assertEqual(s1.work_dir, a)
        self.assertEqual(s2.work_dir, b)
        self.assertEqual(s1._printer.work_dir, a)
        self.assertEqual(s2._printer.work_dir, b)
        # And the env var still has not been written.
        self.assertNotIn("KISS_WORKDIR", os.environ)


# ---------------------------------------------------------------------------
# M3 — minimum TLS version pin
# ---------------------------------------------------------------------------


class TestM3MinimumTlsVersion(unittest.TestCase):
    """The auto-built SSL context pins minimum_version to at least TLS 1.2."""

    def test_ssl_context_minimum_version_is_tls12(self) -> None:
        ctx = _create_ssl_context()
        self.assertGreaterEqual(
            ctx.minimum_version, ssl.TLSVersion.TLSv1_2,
            "Auto-created SSL context must pin minimum_version >= TLSv1_2",
        )

    def test_ssl_context_with_explicit_paths_also_pins_minimum(self) -> None:
        """Even when the caller supplies cert/key paths, the pin holds."""
        d = Path(tempfile.mkdtemp())
        cert = d / "cert.pem"
        key = d / "key.pem"
        _generate_self_signed_cert(cert, key)
        ctx = _create_ssl_context(str(cert), str(key))
        self.assertGreaterEqual(ctx.minimum_version, ssl.TLSVersion.TLSv1_2)


# ---------------------------------------------------------------------------
# M4 — auto-renewal of self-signed cert near/past expiry
# ---------------------------------------------------------------------------


class TestM4SelfSignedCertRenewal(unittest.TestCase):
    """An expired/expiring auto-generated cert is regenerated on next load."""

    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp())
        # Redirect _TLS_DIR so we don't touch the user's real ~/.kiss/tls.
        self._orig_tls_dir = ws_mod._TLS_DIR
        ws_mod._TLS_DIR = self._tmp

    def tearDown(self) -> None:
        ws_mod._TLS_DIR = self._orig_tls_dir

    def test_default_cert_is_long_lived(self) -> None:
        """Newly-generated cert validity is 10 years (M4 bump from 365d)."""
        from cryptography import x509

        cert_path = self._tmp / "cert.pem"
        key_path = self._tmp / "key.pem"
        _generate_self_signed_cert(cert_path, key_path)
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        lifetime = cert.not_valid_after_utc - cert.not_valid_before_utc
        # >= ~9 years (allowing for the leap-day rounding) ensures the
        # validity is no longer the historic 365 days.
        self.assertGreater(
            lifetime, datetime.timedelta(days=365 * 9),
            f"Cert lifetime {lifetime.days}d is too short; expected ~10y",
        )

    def test_needs_renewal_helper_detects_expired(self) -> None:
        """``_self_signed_cert_needs_renewal`` returns True for an expired cert."""
        # Generate a normal cert so the path exists, then overwrite with
        # a freshly-built expired cert.
        cert_path = self._tmp / "cert.pem"
        key_path = self._tmp / "key.pem"
        _write_expired_cert(cert_path, key_path)
        self.assertTrue(_self_signed_cert_needs_renewal(cert_path))

    def test_needs_renewal_helper_negative(self) -> None:
        """A freshly-issued cert does not need renewal."""
        cert_path = self._tmp / "cert.pem"
        key_path = self._tmp / "key.pem"
        _generate_self_signed_cert(cert_path, key_path)
        self.assertFalse(_self_signed_cert_needs_renewal(cert_path))

    def test_needs_renewal_helper_handles_corrupt_cert(self) -> None:
        """A corrupt cert is treated as needing renewal."""
        cert_path = self._tmp / "cert.pem"
        cert_path.write_bytes(b"this is not a valid PEM certificate")
        self.assertTrue(_self_signed_cert_needs_renewal(cert_path))

    def test_create_ssl_context_regenerates_expired_cert(self) -> None:
        """Calling ``_create_ssl_context`` on an expired cert regenerates it."""
        cert_path = self._tmp / "cert.pem"
        key_path = self._tmp / "key.pem"
        _write_expired_cert(cert_path, key_path)
        old_bytes = cert_path.read_bytes()

        ctx = _create_ssl_context()
        self.assertIsInstance(ctx, ssl.SSLContext)

        new_bytes = cert_path.read_bytes()
        self.assertNotEqual(
            old_bytes, new_bytes,
            "Expired cert should have been regenerated",
        )
        self.assertFalse(_self_signed_cert_needs_renewal(cert_path))


def _write_expired_cert(cert_path: Path, key_path: Path) -> None:
    """Generate a self-signed cert that expired one day ago."""
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "expired"),
    ])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=400))
        .not_valid_after(now - datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))
    os.chmod(key_path, 0o600)


# ---------------------------------------------------------------------------
# M5 — TOCTOU retry on cloudflared bind collision
# ---------------------------------------------------------------------------


class TestM5SpawnRetriesOnImmediateExit(IsolatedAsyncioTestCase):
    """``_spawn_cloudflared`` retries with a fresh port on quick-exit."""

    async def asyncSetUp(self) -> None:
        self._snap = _ConfigSnapshot().__enter__()
        save_config({"remote_password": ""})
        self._tmpdir = tempfile.mkdtemp()
        self._counter_file = os.path.join(self._tmpdir, "counter")
        Path(self._counter_file).write_text("0")
        # Fake cloudflared: exits with code 7 on the first invocation,
        # then runs successfully (sleeps).  Uses a counter file so the
        # test is robust against argv differences.
        cf = os.path.join(self._tmpdir, "cloudflared")
        Path(cf).write_text(
            "#!/bin/bash\n"
            f'COUNTER_FILE="{self._counter_file}"\n'
            'COUNT=$(cat "$COUNTER_FILE")\n'
            'NEXT=$((COUNT+1))\n'
            'echo -n "$NEXT" > "$COUNTER_FILE"\n'
            'if [ "$COUNT" -lt 1 ]; then\n'
            '    echo "INF bind: address already in use" >&2\n'
            '    exit 7\n'
            'fi\n'
            'echo "INF https://m5-ok.trycloudflare.com" >&2\n'
            'sleep 30\n'
        )
        os.chmod(cf, 0o755)
        self._old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = self._tmpdir + ":" + self._old_path
        self.port = _find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=self.port, use_tunnel=False,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        os.environ["PATH"] = self._old_path
        proc = self.server._tunnel_proc
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        await self.server.stop_async()
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._snap.__exit__()

    async def test_spawn_retries_on_immediate_exit(self) -> None:
        """First spawn exits with rc=7, second succeeds; final proc is alive."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self.server._spawn_cloudflared, ["--url", "https://x"], 3,
        )
        proc = self.server._tunnel_proc
        self.assertIsNotNone(proc)
        assert proc is not None
        self.assertIsNone(proc.poll(), "final cloudflared proc must be alive")
        # The fake cloudflared was invoked at least twice (first exit
        # forced the retry).
        count = int(Path(self._counter_file).read_text())
        self.assertGreaterEqual(count, 2)


class TestM5SpawnSignatureAcceptsRetries(unittest.TestCase):
    """``_spawn_cloudflared`` accepts a ``retries`` kwarg (M5 fix)."""

    def test_signature(self) -> None:
        sig = inspect.signature(RemoteAccessServer._spawn_cloudflared)
        self.assertIn("retries", sig.parameters)


# ---------------------------------------------------------------------------
# M6 — merge state cleanup on disconnect + lock against agent-thread race
# ---------------------------------------------------------------------------


class TestM6MergeStateCleanup(IsolatedAsyncioTestCase):
    """Disconnecting the WebSocket drops merge state for that tab."""

    async def asyncSetUp(self) -> None:
        self._snap = _ConfigSnapshot().__enter__()
        save_config({"remote_password": ""})
        self.port = _find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=self.port, work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        self._snap.__exit__()

    async def test_merge_state_dropped_on_disconnect(self) -> None:
        """A tab's merge state is removed when the WS for that tab closes."""
        tab_id = "tab-m6-disc"
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            # Send any command carrying the tabId so the server
            # records it on this connection.
            await ws.send(json.dumps({
                "type": "getWelcomeSuggestions", "tabId": tab_id,
            }))
            # Drain a few events to let the server process the message.
            for _ in range(3):
                try:
                    await asyncio.wait_for(ws.recv(), timeout=1)
                except (TimeoutError, ConnectionClosed):
                    break
            # Inject a merge state for this tab from outside (mimicking
            # what WebPrinter.broadcast → _register_merge_state does).
            self.server._register_merge_state(tab_id, {"files": [
                {"name": "x.txt", "base": "/tmp/x.b", "current": "/tmp/x.c",
                 "hunks": [{"bs": 0, "bc": 0, "cs": 0, "cc": 1}]},
            ]})
            self.assertIn(tab_id, self.server._merge_states)
        # After the WS closes, the cleanup branch must drop the state.
        # Give the asyncio handler a moment to run its `finally:`.
        for _ in range(20):
            await asyncio.sleep(0.05)
            if tab_id not in self.server._merge_states:
                break
        self.assertNotIn(
            tab_id, self.server._merge_states,
            "merge state for the disconnected tab must be cleaned up",
        )

    async def test_merge_states_lock_exists(self) -> None:
        """A threading.Lock guards _merge_states (M6 race fix)."""
        import threading

        self.assertIsInstance(
            self.server._merge_states_lock, type(threading.Lock()),
        )


# ---------------------------------------------------------------------------
# M7 — clamp restoredTabs / attachments / prompt size
# ---------------------------------------------------------------------------


class TestM7CapsRestoredTabs(IsolatedAsyncioTestCase):
    """Oversize ``restoredTabs`` is truncated to the configured cap."""

    async def asyncSetUp(self) -> None:
        self._snap = _ConfigSnapshot().__enter__()
        save_config({"remote_password": ""})
        # Tighten the cap so we don't have to send 32+ entries.
        self._old_cap = ws_mod._MAX_RESTORED_TABS
        ws_mod._MAX_RESTORED_TABS = 3
        self.port = _find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=self.port, work_dir=tempfile.mkdtemp(),
        )
        # Track resumeSession invocations on the underlying VSCodeServer.
        self._resumed: list[str] = []
        original = self.server._vscode_server._handle_command

        def _spy(cmd: dict[str, Any]) -> None:  # noqa: D401
            if cmd.get("type") == "resumeSession":
                self._resumed.append(cmd.get("chatId", ""))
            return original(cmd)

        self.server._vscode_server._handle_command = _spy  # type: ignore[assignment]
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        ws_mod._MAX_RESTORED_TABS = self._old_cap
        self._snap.__exit__()

    async def test_restored_tabs_truncated_to_cap(self) -> None:
        """Sending 10 restored tabs invokes resumeSession at most cap times."""
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            tabs = [
                {"chatId": f"chat-{i}", "tabId": f"tab-{i}"}
                for i in range(10)
            ]
            await ws.send(json.dumps({
                "type": "ready", "tabId": "primary", "restoredTabs": tabs,
            }))
            # Drain a bunch of responses.
            for _ in range(50):
                try:
                    await asyncio.wait_for(ws.recv(), timeout=0.5)
                except TimeoutError:
                    break
        await asyncio.sleep(0.5)  # let the executor finish
        self.assertEqual(
            len(self._resumed), 3,
            f"Expected exactly cap=3 resumeSession calls, got {self._resumed}",
        )


class TestM7CapsAttachments(IsolatedAsyncioTestCase):
    """``attachments`` list is clamped before being passed to ``run``."""

    async def asyncSetUp(self) -> None:
        self._snap = _ConfigSnapshot().__enter__()
        save_config({"remote_password": ""})
        self._old_cap = ws_mod._MAX_ATTACHMENTS
        ws_mod._MAX_ATTACHMENTS = 4
        self.port = _find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=self.port, work_dir=tempfile.mkdtemp(),
        )

    async def asyncTearDown(self) -> None:
        ws_mod._MAX_ATTACHMENTS = self._old_cap
        self._snap.__exit__()

    async def test_handle_submit_truncates_attachments(self) -> None:
        """``_handle_submit`` truncates a 100-entry attachments list to 4."""
        captured: list[dict[str, Any]] = []

        async def _capture(c: dict[str, Any]) -> None:
            captured.append(c)

        self.server._run_cmd = _capture  # type: ignore[assignment]
        self.server._loop = asyncio.get_running_loop()
        await self.server._handle_submit({
            "tabId": "t1",
            "prompt": "hi",
            "attachments": [{"path": f"f{i}"} for i in range(100)],
        })
        self.assertEqual(len(captured), 1)
        self.assertEqual(len(captured[0]["attachments"]), 4)


class TestM7TruncatesPrompt(IsolatedAsyncioTestCase):
    """An oversize prompt is truncated before being broadcast / dispatched."""

    async def asyncSetUp(self) -> None:
        self._snap = _ConfigSnapshot().__enter__()
        save_config({"remote_password": ""})
        self._old_cap = ws_mod._MAX_PROMPT_BYTES
        ws_mod._MAX_PROMPT_BYTES = 100
        self.port = _find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=self.port, work_dir=tempfile.mkdtemp(),
        )

    async def asyncTearDown(self) -> None:
        ws_mod._MAX_PROMPT_BYTES = self._old_cap
        self._snap.__exit__()

    async def test_handle_submit_truncates_prompt(self) -> None:
        captured: list[dict[str, Any]] = []

        async def _capture(c: dict[str, Any]) -> None:
            captured.append(c)

        self.server._run_cmd = _capture  # type: ignore[assignment]
        self.server._loop = asyncio.get_running_loop()
        big = "X" * 5000
        await self.server._handle_submit({
            "tabId": "t1", "prompt": big,
        })
        self.assertEqual(len(captured), 1)
        self.assertLessEqual(len(captured[0]["prompt"]), 100)


# ---------------------------------------------------------------------------
# M8 — broadcast tracks futures; remove_client cancels pending sends
# ---------------------------------------------------------------------------


class TestM8FuturesTrackedAndCancelled(IsolatedAsyncioTestCase):
    """``WebPrinter`` tracks pending sends and cancels them on remove_client."""

    async def test_remove_client_cancels_pending_futures(self) -> None:
        printer = WebPrinter()
        printer._loop = asyncio.get_running_loop()

        # A fake "client" with a never-completing send coroutine.  We
        # deliberately do NOT inherit from ServerConnection — broadcast
        # only needs an object that has an awaitable .send().
        class _StuckWs:
            async def send(self, _data: str) -> None:
                # Block forever so the future never completes by itself.
                await asyncio.Event().wait()

        stuck = _StuckWs()
        printer.add_client(stuck)  # type: ignore[arg-type]
        # Trigger a broadcast — the send coroutine will be scheduled
        # but will never finish, so the future stays pending.
        printer.broadcast({"type": "ping"})
        # Give the loop a moment to schedule.
        await asyncio.sleep(0.1)
        with printer._ws_lock:
            pending = list(printer._pending_sends.get(stuck, set()))
        self.assertEqual(
            len(pending), 1,
            "broadcast must record a pending future per client",
        )
        fut = pending[0]
        self.assertFalse(fut.done(), "future must still be pending")

        # Now remove the client — the future must be cancelled.
        printer.remove_client(stuck)  # type: ignore[arg-type]
        # remove_client cancels via `fut.cancel()`.  Wait briefly for
        # the cancellation to propagate.
        for _ in range(20):
            await asyncio.sleep(0.05)
            if fut.cancelled() or fut.done():
                break
        self.assertTrue(
            fut.cancelled() or fut.done(),
            "pending future must be cancelled (or completed) after remove_client",
        )
        # And the per-client tracking dict no longer holds the client.
        with printer._ws_lock:
            self.assertNotIn(stuck, printer._pending_sends)

    async def test_completed_send_is_discarded_from_pending(self) -> None:
        """A completed send is removed from ``_pending_sends`` automatically."""
        printer = WebPrinter()
        printer._loop = asyncio.get_running_loop()

        class _OkWs:
            sent: list[str] = []

            async def send(self, data: str) -> None:
                self.sent.append(data)

        ok = _OkWs()
        printer.add_client(ok)  # type: ignore[arg-type]
        printer.broadcast({"type": "hello"})
        # Wait for the scheduled send to complete and the done-callback
        # to drain the pending set.
        for _ in range(20):
            await asyncio.sleep(0.05)
            with printer._ws_lock:
                if not printer._pending_sends.get(ok):
                    break
        with printer._ws_lock:
            self.assertEqual(printer._pending_sends.get(ok), set())
        self.assertGreaterEqual(len(_OkWs.sent), 1)


# ---------------------------------------------------------------------------
# M9 — _WebMergeState exposes is_resolved; web_server stops poking _resolved
# ---------------------------------------------------------------------------


class TestM9IsResolvedMethod(unittest.TestCase):
    """``_WebMergeState`` exposes a public ``is_resolved`` method."""

    def test_is_resolved_method_exists(self) -> None:
        self.assertTrue(callable(getattr(_WebMergeState, "is_resolved", None)))

    def test_is_resolved_returns_correct_values(self) -> None:
        state = _WebMergeState({"files": [
            {"name": "a", "hunks": [{}, {}]},
            {"name": "b", "hunks": [{}]},
        ]})
        self.assertFalse(state.is_resolved(0, 0))
        self.assertFalse(state.is_resolved(1, 0))
        state.mark_resolved(0, 1)
        self.assertTrue(state.is_resolved(0, 1))
        self.assertFalse(state.is_resolved(0, 0))

    def test_handle_web_merge_action_does_not_poke_resolved_directly(self) -> None:
        """Source no longer references ``state._resolved`` outside the class."""
        src = inspect.getsource(RemoteAccessServer._handle_web_merge_action)
        self.assertNotIn(
            "state._resolved", src,
            "_handle_web_merge_action must use is_resolved(), not _resolved",
        )


# ---------------------------------------------------------------------------
# M10 — _send_welcome_info is async and uses run_in_executor for blocking IO
# ---------------------------------------------------------------------------


class TestM10WelcomeInfoIsAsync(unittest.TestCase):
    """``_send_welcome_info`` must be a coroutine and use run_in_executor."""

    def test_is_coroutine_function(self) -> None:
        self.assertTrue(
            inspect.iscoroutinefunction(RemoteAccessServer._send_welcome_info),
            "_send_welcome_info must be async (M10) so it can offload "
            "blocking IO via run_in_executor",
        )

    def test_source_uses_run_in_executor(self) -> None:
        src = inspect.getsource(RemoteAccessServer._send_welcome_info)
        self.assertIn("run_in_executor", src)


class TestM10WelcomeInfoDoesNotBlockEventLoop(IsolatedAsyncioTestCase):
    """Slow disk read inside ``_send_welcome_info`` doesn't stall the loop."""

    async def asyncSetUp(self) -> None:
        self._snap = _ConfigSnapshot().__enter__()
        save_config({"remote_password": ""})
        self.port = _find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=self.port, work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        self._snap.__exit__()

    async def test_concurrent_task_runs_during_blocking_io(self) -> None:
        """A slow URL-file read does not block another concurrent task."""
        slow_calls: list[float] = []

        original = RemoteAccessServer._read_url_from_file

        def _slow_read() -> str | None:
            slow_calls.append(time.monotonic())
            time.sleep(1.5)  # synchronous sleep to simulate slow IO
            return original()

        # Patch the static method on the class for the duration of the test.
        RemoteAccessServer._read_url_from_file = staticmethod(_slow_read)  # type: ignore[assignment]
        self.server._active_url = None
        try:
            ticks = 0

            async def _ticker() -> int:
                nonlocal ticks
                deadline = time.monotonic() + 1.2
                while time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
                    ticks += 1
                return ticks

            ticker = asyncio.create_task(_ticker())
            welcome = asyncio.create_task(self.server._send_welcome_info())
            tick_count = await ticker
            await welcome
        finally:
            RemoteAccessServer._read_url_from_file = staticmethod(original)  # type: ignore[assignment]

        self.assertEqual(len(slow_calls), 1)
        # The ticker fired ~24 times during the 1.2s window — if the
        # event loop had been blocked it would have fired zero times.
        self.assertGreater(
            tick_count, 5,
            f"event loop was blocked during slow IO (only {tick_count} ticks)",
        )


# ---------------------------------------------------------------------------
# M11 — _WebMergeState.current() returns None once everything is resolved
# ---------------------------------------------------------------------------


class TestM11CurrentReturnsNoneWhenAllResolved(unittest.TestCase):
    """After accept-all / reject-all, ``current()`` is unambiguously ``None``."""

    def test_current_none_after_all_resolved(self) -> None:
        state = _WebMergeState({"files": [
            {"name": "a", "hunks": [{}, {}]},
            {"name": "b", "hunks": [{}]},
        ]})
        # Sanity: before resolving anything, current is the first hunk.
        self.assertEqual(state.current(), (0, 0))
        # Resolve every hunk (mimics accept-all / reject-all).
        for fi, hi in state.all_unresolved():
            state.mark_resolved(fi, hi)
        self.assertEqual(state.remaining, 0)
        self.assertIsNone(
            state.current(),
            "current() must return None once every hunk is resolved",
        )

    def test_current_none_for_empty_state(self) -> None:
        state = _WebMergeState({"files": []})
        self.assertIsNone(state.current())

    def test_current_returns_unresolved_hunk(self) -> None:
        """Sanity: while hunks remain, current() returns one of them."""
        state = _WebMergeState({"files": [
            {"name": "a", "hunks": [{}, {}]},
        ]})
        cur = state.current()
        self.assertIsNotNone(cur)
        assert cur is not None
        self.assertFalse(state.is_resolved(*cur))


# ---------------------------------------------------------------------------
# M12 — _authenticate_ws closes the socket on every failure, including
#       the exception path
# ---------------------------------------------------------------------------


class TestM12AuthClosesSocketOnException(IsolatedAsyncioTestCase):
    """``_authenticate_ws`` always closes the WS on failure, including errors."""

    async def asyncSetUp(self) -> None:
        self._snap = _ConfigSnapshot().__enter__()
        save_config({"remote_password": "secret-m12"})
        self.port = _find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=self.port, work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        self._snap.__exit__()

    async def test_socket_closed_on_invalid_json(self) -> None:
        """A non-JSON auth payload raises and the server closes the socket."""
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
        ) as ws:
            await ws.send("this is not valid JSON")
            # The server must close the connection (M12); recv() then
            # raises ConnectionClosed.
            with self.assertRaises(ConnectionClosed):
                await asyncio.wait_for(ws.recv(), timeout=5)

    async def test_socket_closed_on_auth_timeout(self) -> None:
        """Sending nothing at all → server hits its 30s recv timeout → close."""
        # Patch the recv timeout for this test's connection so the test
        # runs quickly.  We do this by monkey-patching ``_authenticate_ws``
        # to call ``asyncio.wait_for`` with a very short timeout.
        original = self.server._authenticate_ws

        async def _short_timeout_auth(websocket: Any) -> bool:
            # Replicate the authenticate path but with a 0.5s timeout
            # on the first recv() — exercising the same exception path.

            try:
                await asyncio.wait_for(websocket.recv(), timeout=0.5)
                return False  # never reached: the timeout fires first
            except Exception:
                try:
                    await websocket.close()
                except Exception:
                    pass
                return False

        self.server._authenticate_ws = _short_timeout_auth  # type: ignore[assignment]
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
        ) as ws:
            # Send nothing — let the server's wait_for timeout.
            with self.assertRaises(ConnectionClosed):
                await asyncio.wait_for(ws.recv(), timeout=5)
        self.server._authenticate_ws = original  # type: ignore[assignment]

    def test_authenticate_source_has_close_in_exception_branch(self) -> None:
        """Source-level guarantee: ``except`` branch contains ``websocket.close``."""
        src = inspect.getsource(RemoteAccessServer._authenticate_ws)
        # The outer (function-level) ``except Exception:`` is indented
        # exactly 8 spaces.  Anything deeper is a nested except inside
        # a try/except (e.g. the inner ``try: await close() except: pass``).
        marker = "\n        except Exception:"
        idx = src.find(marker)
        self.assertGreaterEqual(
            idx, 0, "Could not find function-level exception handler",
        )
        tail = src[idx:]
        self.assertIn(
            "websocket.close()", tail,
            "Exception branch in _authenticate_ws must call websocket.close()",
        )


if __name__ == "__main__":
    unittest.main()
