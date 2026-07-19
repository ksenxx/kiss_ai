# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression: the remote webapp MUST always ask for a password.

Bug being locked in ("why doesn't the remote webapp ask for a
password?"):

    Every fresh browser page-load runs the ``_WS_SHIM_JS`` shim, whose
    ``onopen`` handler unconditionally sends an ``auth`` message with
    the password read from ``localStorage`` — which is the empty string
    on a device that has never logged in (a brand-new phone, an
    incognito window, cleared storage, ...).  The server answers that
    empty-password probe with ``auth_required`` so the webapp shows its
    password modal.

    The defect: :meth:`RemoteAccessServer._authenticate_ws` counted that
    benign empty-password probe as a genuine failed login via
    :meth:`_record_auth_failure`.  Because :meth:`_client_ip` collapses
    every visitor arriving through the public cloudflared tunnel to the
    single shared loopback source IP, only :data:`_AUTH_FAIL_MAX` (5)
    normal page loads within :data:`_AUTH_FAIL_WINDOW` seconds are
    enough to rate-limit **everyone**.  Once the shared IP is locked,
    :meth:`_authenticate_ws` closes new sockets *silently* (it never
    sends ``auth_required``), so the webapp's password modal is never
    shown — the app just spins on the loading overlay.  From the user's
    point of view "the remote webapp doesn't ask for a password".

    The fix: only a NON-EMPTY wrong password guess counts toward the
    brute-force lockout.  The shim's automatic empty-password probe is
    never penalised, so legitimate visitors always receive
    ``auth_required`` (and therefore the password prompt), while genuine
    brute-force attempts (non-empty guesses) are still locked out.

These tests drive real WSS handshakes against a live
``RemoteAccessServer`` (no mocks), exactly like
``test_password_persistence.py``.
"""

from __future__ import annotations

import asyncio
import json
import socket
import ssl
import tempfile
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server.web_server import (
    _AUTH_FAIL_MAX,
    RemoteAccessServer,
)


def _pick_free_port() -> int:
    """Return an OS-assigned free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _no_verify_ssl() -> ssl.SSLContext:
    """Permissive SSL context for the dev self-signed cert."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


_PASSWORD = "correct-horse-battery-staple"


class TestRemoteWebappAlwaysPrompts(IsolatedAsyncioTestCase):
    """The password modal (``auth_required``) must always reach a visitor."""

    async def asyncSetUp(self) -> None:
        """Start a real ``RemoteAccessServer`` with a known password."""
        self._port = _pick_free_port()
        self._orig_config: str | None = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": _PASSWORD})

        self._server = RemoteAccessServer(
            host="127.0.0.1",
            port=self._port,
            work_dir=tempfile.mkdtemp(),
            use_tunnel=False,
        )
        await self._server.start_async()

    async def asyncTearDown(self) -> None:
        """Stop the server and restore the user's saved config."""
        await self._server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def _ws_connect(self) -> Any:
        """Open a fresh WSS connection to /ws on the test server."""
        return await connect(
            f"wss://127.0.0.1:{self._port}/ws",
            ssl=_no_verify_ssl(),
        )

    async def _probe(self, password: str) -> str | None:
        """Send one ``auth`` with *password*; return the reply ``type``.

        Simulates a single fresh page-load: one WS connection, one
        ``auth`` frame, read the first reply.  Returns the server's
        message ``type`` (``"auth_ok"`` / ``"auth_required"`` / ...), or
        ``None`` when the server closed the socket without replying
        (the silent-lockout path that hides the password prompt).
        """
        try:
            async with await self._ws_connect() as ws:
                await ws.send(json.dumps({"type": "auth", "password": password}))
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                msg = json.loads(raw)
                return str(msg.get("type"))
        except Exception:
            return None

    async def test_many_fresh_loads_still_get_prompt(self) -> None:
        """Repeated empty-password probes must keep yielding ``auth_required``.

        Reproduces the reported bug: on the shared cloudflared source IP,
        a handful of normal page loads (each an empty-password probe)
        must NOT rate-limit the visitor into a silent close.  Every
        probe — well past :data:`_AUTH_FAIL_MAX` — must still be answered
        with ``auth_required`` so the password modal appears.
        """
        loads = _AUTH_FAIL_MAX * 3  # comfortably past the lockout threshold
        for i in range(loads):
            reply = await self._probe("")
            self.assertEqual(
                reply,
                "auth_required",
                f"Fresh page-load #{i + 1} must be answered with "
                "auth_required so the webapp shows its password prompt; "
                f"got {reply!r} (silent close == no prompt).",
            )

    async def test_empty_probe_does_not_consume_lockout_budget(self) -> None:
        """Empty probes never lock the IP, so a real login still succeeds.

        After many empty-password probes (which used to trip the
        lockout), supplying the CORRECT password must still authenticate
        — proving the benign probes did not consume the brute-force
        budget and lock the shared IP.
        """
        for _ in range(_AUTH_FAIL_MAX * 2):
            self.assertEqual(await self._probe(""), "auth_required")
        self.assertEqual(
            await self._probe(_PASSWORD),
            "auth_ok",
            "The correct password must authenticate even after many "
            "empty-password probes from the same (shared) IP.",
        )

    async def test_correct_password_authenticates(self) -> None:
        """A matching password yields ``auth_ok`` on the first try."""
        self.assertEqual(await self._probe(_PASSWORD), "auth_ok")

    async def test_wrong_password_prompts_then_locks(self) -> None:
        """Non-empty wrong guesses still trip the brute-force lockout.

        The fix must not weaken brute-force protection: a burst of
        NON-EMPTY wrong passwords beyond :data:`_AUTH_FAIL_MAX` must
        eventually be refused with a silent close (``None``), not an
        endless supply of ``auth_required`` prompts.
        """
        # The first few wrong guesses are answered with auth_required.
        first = await self._probe("wrong-guess-0")
        self.assertEqual(
            first,
            "auth_required",
            "A wrong non-empty password should first elicit auth_required.",
        )
        got_locked = False
        for i in range(1, _AUTH_FAIL_MAX * 3):
            reply = await self._probe(f"wrong-guess-{i}")
            if reply is None:
                got_locked = True
                break
        self.assertTrue(
            got_locked,
            "A sustained burst of NON-EMPTY wrong passwords must "
            "eventually lock the IP (silent close), preserving "
            "brute-force protection.",
        )

    async def test_non_string_password_is_treated_as_empty_probe(self) -> None:
        """A non-string ``password`` is coerced to empty and not penalised.

        A malformed frame whose ``password`` is not a string (e.g. a
        number injected by a buggy client) must be coerced to the empty
        string — i.e. treated as a benign probe that yields
        ``auth_required`` without consuming the lockout budget.
        """
        try:
            async with await self._ws_connect() as ws:
                await ws.send(json.dumps({"type": "auth", "password": 12345}))
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                self.assertEqual(json.loads(raw).get("type"), "auth_required")
        except Exception:  # pragma: no cover - defensive
            self.fail("non-string password should elicit auth_required")
        # And it must not have consumed the lockout budget.
        self.assertEqual(await self._probe(_PASSWORD), "auth_ok")

    async def test_two_wrong_attempts_end_in_error_and_close(self) -> None:
        """Two wrong guesses on one socket end with an error, then close.

        Exercises the full two-attempt handshake: the first wrong
        non-empty password elicits ``auth_required``; a second wrong
        guess on the retry falls through the loop to the terminal
        ``Authentication failed`` error frame followed by a socket
        close.
        """
        async with await self._ws_connect() as ws:
            await ws.send(json.dumps({"type": "auth", "password": "nope-1"}))
            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(first.get("type"), "auth_required")
            await ws.send(json.dumps({"type": "auth", "password": "nope-2"}))
            second = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(
                second.get("type"),
                "error",
                "The second failed attempt must yield a terminal error "
                "frame before the socket is closed.",
            )
            # The server closes right after the error frame.
            with self.assertRaises(Exception):
                await asyncio.wait_for(ws.recv(), timeout=5)

    async def test_malformed_first_frame_closes_socket(self) -> None:
        """A non-JSON first frame is handled gracefully (close, no crash).

        Drives the exception path in ``_authenticate_ws`` (``json.loads``
        raises), which must close the socket rather than leave a hung
        half-open connection — and must not count as a login failure.
        """
        async with await self._ws_connect() as ws:
            await ws.send("this is not json")
            with self.assertRaises(Exception):
                await asyncio.wait_for(ws.recv(), timeout=5)
        # The malformed frame must not have locked the IP.
        self.assertEqual(await self._probe(""), "auth_required")

    async def test_connect_while_locked_is_closed_silently(self) -> None:
        """Once genuinely locked, a new connection is closed with no reply.

        After ``_AUTH_FAIL_MAX`` non-empty wrong guesses the shared IP is
        locked; the very next connection must be closed by the top-of-
        method lock check WITHOUT any ``auth_required`` (silent close).
        This is the brute-force protection the fix deliberately keeps.
        """
        for i in range(_AUTH_FAIL_MAX):
            self.assertEqual(
                await self._probe(f"brute-{i}"),
                "auth_required",
                f"Setup guess #{i} must be processed (auth_required) so the "
                "lockout is driven by genuine recorded non-empty failures.",
            )
        # The IP is now locked: a fresh connection gets no reply at all.
        self.assertIsNone(
            await self._probe("still-locked"),
            "A locked IP must be closed silently (no auth_required).",
        )

    async def test_non_auth_first_message_closes_without_penalty(self) -> None:
        """A non-``auth`` first frame closes but never counts as a failure.

        A stray non-auth frame (e.g. a buffered command replayed by a
        reconnecting shim) must close the socket without recording a
        failure, so it cannot contribute to the lockout that hides the
        prompt.  A subsequent empty probe must still get auth_required.
        """
        # Send many non-auth first frames.
        for _ in range(_AUTH_FAIL_MAX * 2):
            try:
                async with await self._ws_connect() as ws:
                    await ws.send(json.dumps({"type": "ready", "tabId": "t"}))
                    # Server closes without replying; recv raises.
                    with self.assertRaises(Exception):
                        await asyncio.wait_for(ws.recv(), timeout=5)
            except Exception:
                pass
        self.assertEqual(
            await self._probe(""),
            "auth_required",
            "Non-auth first frames must not lock the IP out of the prompt.",
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
