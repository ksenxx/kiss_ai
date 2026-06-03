# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for client-side remote password persistence.

The remote web server embeds a small JavaScript shim (`_WS_SHIM_JS`) in
the HTML body delivered to browser clients.  The shim is responsible
for:

1. Reading the cached remote-access password from browser storage on
   every WebSocket reconnect and sending it as the first ``auth``
   message, so that visiting the website does not trigger a password
   prompt as long as the cached password still matches the server's
   ``remote_password`` setting.
2. Falling back to ``window.prompt()`` when the server replies with
   ``auth_required``, and persisting the user-provided value for the
   next visit.
3. Storing the cached password in ``localStorage`` (not
   ``sessionStorage``) so the value survives tab and browser restarts.
4. Clearing the cached password when the server rejects it, so that
   the next reload re-prompts instead of silently retrying a stale
   value.

These tests pin the exact JS strings used by the shim so a future
refactor that accidentally reverts to ``sessionStorage`` (or otherwise
breaks cross-visit persistence) fails loudly.

They also drive a real WebSocket handshake against a live
``RemoteAccessServer`` to confirm that re-supplying a cached password
authenticates without a second prompt, matching the contract the JS
shim relies on.
"""

from __future__ import annotations

import asyncio
import json
import socket
import ssl
import tempfile
import unittest
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

from kiss.agents.vscode.vscode_config import CONFIG_PATH, save_config
from kiss.agents.vscode.web_server import (
    _WS_SHIM_JS,
    RemoteAccessServer,
    _build_html,
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


class TestShimUsesLocalStorageForPassword(unittest.TestCase):
    """Static assertions on the JS shim's password storage choice."""

    def test_password_read_uses_local_storage(self) -> None:
        """The auth on-open path reads the password from localStorage."""
        self.assertIn(
            "localStorage.getItem('sorcar-remote-pwd')",
            _WS_SHIM_JS,
            "Shim must read the cached remote password from localStorage so "
            "the password survives tab close / browser restart.",
        )

    def test_password_write_uses_local_storage(self) -> None:
        """The auth_required handler persists the password to localStorage."""
        self.assertIn(
            "localStorage.setItem('sorcar-remote-pwd', pwd)",
            _WS_SHIM_JS,
            "Shim must write the user-provided password to localStorage so "
            "subsequent visits skip the prompt.",
        )

    def test_password_clear_on_auth_required(self) -> None:
        """Stale cached password is removed when the server rejects it."""
        self.assertIn(
            "localStorage.removeItem('sorcar-remote-pwd')",
            _WS_SHIM_JS,
            "Shim must clear the cached password on auth_required so a stale "
            "wrong password does not silently retry forever.",
        )

    def test_password_never_uses_session_storage(self) -> None:
        """The password key must not appear in any sessionStorage call."""
        # sessionStorage is still used for sorcar-state (per-tab webview
        # state), so we just check that no sessionStorage line mentions
        # the password key.
        for line in _WS_SHIM_JS.splitlines():
            if "sorcar-remote-pwd" in line:
                self.assertNotIn(
                    "sessionStorage",
                    line,
                    "Password storage line must not use sessionStorage: "
                    f"{line!r}",
                )

    def test_state_still_uses_session_storage(self) -> None:
        """Per-tab webview state must stay in sessionStorage."""
        self.assertIn(
            "sessionStorage.getItem('sorcar-state')",
            _WS_SHIM_JS,
            "Webview state should remain per-tab (sessionStorage), not "
            "shared across tabs.",
        )
        self.assertIn(
            "sessionStorage.setItem('sorcar-state', JSON.stringify(s))",
            _WS_SHIM_JS,
        )


class TestShimEmbeddedInHtml(unittest.TestCase):
    """The HTML served to browser clients must embed the updated shim."""

    def test_html_contains_local_storage_password_read(self) -> None:
        """_build_html() output includes the localStorage-based shim."""
        html = _build_html()
        self.assertIn("localStorage.getItem('sorcar-remote-pwd')", html)
        self.assertIn("localStorage.setItem('sorcar-remote-pwd', pwd)", html)


class TestCustomAuthModal(unittest.TestCase):
    """The password dialog is a compact in-page modal, not native prompt().

    The browser-native ``prompt()`` renders an elongated dialog with
    wasted vertical space below the OK/Cancel buttons on Chrome,
    Safari and Firefox.  The auth UI must therefore be a custom HTML
    modal sized tightly around its contents.
    """

    def test_html_carries_auth_modal_container(self) -> None:
        """The HTML body must include the ``#auth-modal`` element."""
        html = _build_html()
        self.assertIn('id="auth-modal"', html)
        self.assertIn('id="auth-modal-input"', html)
        self.assertIn('id="auth-modal-ok"', html)
        self.assertIn('id="auth-modal-cancel"', html)

    def test_auth_modal_input_is_password_type(self) -> None:
        """The password input must be masked (``type="password"``)."""
        html = _build_html()
        # Locate the input declaration and confirm it is type=password.
        self.assertRegex(
            html,
            r'<input\s+type="password"\s+id="auth-modal-input"',
        )

    def test_shim_invokes_custom_auth_modal(self) -> None:
        """The auth_required handler must drive ``_showAuthModal``."""
        self.assertIn("_showAuthModal", _WS_SHIM_JS)
        # The auth_required branch should call _showAuthModal(...).then(...)
        # — not the bare prompt() that produced the elongated dialog.
        self.assertRegex(_WS_SHIM_JS, r"_showAuthModal\(\)\.then\(")

    def test_shim_keeps_prompt_only_as_fallback(self) -> None:
        """``prompt()`` remains only as the modal-absent fallback path.

        The shim is still loaded by tests that don't render the HTML
        body, so the fallback must exist; but it must NOT be the path
        used when the modal nodes are present.  We pin this by
        verifying that every occurrence of ``prompt(`` in the shim
        appears after the modal-presence guard.
        """
        idx_guard = _WS_SHIM_JS.find("if (!modal || !input")
        self.assertGreater(
            idx_guard, -1,
            "Shim must contain a modal-presence guard before falling "
            "back to prompt().",
        )
        # Every actual prompt('...') invocation (i.e. with a string
        # literal argument — not a doc-comment mention) must appear
        # after the modal-presence guard line.
        import re
        for match in re.finditer(r"""prompt\(['"]""", _WS_SHIM_JS):
            self.assertGreater(
                match.start(), idx_guard,
                "prompt() may only be invoked as the fallback path "
                "inside _showAuthModal, after the modal-presence guard.",
            )


class TestCachedPasswordReauthenticates(IsolatedAsyncioTestCase):
    """End-to-end: replaying the cached password succeeds without prompt."""

    async def asyncSetUp(self) -> None:
        """Start a real ``RemoteAccessServer`` with a known password."""
        self._port = _pick_free_port()
        self._orig_config: str | None = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": "correct-horse-battery-staple"})

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

    async def test_cached_password_authenticates_on_revisit(self) -> None:
        """Two separate sessions both auth with the same cached password.

        This simulates the user closing the tab (first session ends)
        and reopening it later (second session) — the browser's
        ``localStorage`` still contains the password, so the on-open
        path sends it and the server responds ``auth_ok`` without any
        ``auth_required`` round trip.
        """
        cached_pwd = "correct-horse-battery-staple"
        for visit in range(2):
            async with await self._ws_connect() as ws:
                await ws.send(json.dumps({"type": "auth", "password": cached_pwd}))
                resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                self.assertEqual(
                    resp["type"],
                    "auth_ok",
                    f"Visit {visit}: cached password should authenticate "
                    "without an auth_required round trip.",
                )

    async def test_stale_cached_password_triggers_auth_required(self) -> None:
        """A wrong cached password elicits auth_required (then a retry works)."""
        async with await self._ws_connect() as ws:
            await ws.send(json.dumps({"type": "auth", "password": "stale-wrong"}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(
                resp["type"],
                "auth_required",
                "Server must signal the shim to clear/replace the cached "
                "password when it no longer matches.",
            )
            # Shim would prompt the user, then resend with the new value.
            await ws.send(
                json.dumps({"type": "auth", "password": "correct-horse-battery-staple"})
            )
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "auth_ok")


if __name__ == "__main__":
    unittest.main()
