# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for the ``remote_password`` shared-config leak.

The whole pytest session shares one ``KISS_HOME`` (set by the root
``tests/conftest.py``), so ``$KISS_HOME/config.json`` is shared by every
test in the run.  A non-empty ``remote_password`` leaked into that file
by one test made every later real-server test fail its websocket
handshake with ``auth_required`` instead of ``auth_ok`` — and the
Playwright-driven tests hung behind the ``#auth-modal`` overlay.  The
failures were order-dependent: they appeared only in large
``tests/agents/vscode`` runs and vanished in isolation.

Two concrete leak paths existed, both closed by the
``_isolated_shared_config`` autouse fixture in the root conftest:

1. **Pinned path overrides.**  ``vscode_config.CONFIG_DIR`` /
   ``CONFIG_PATH`` are lazy module attributes that follow ``$KISS_HOME``
   on every access.  Many tests save them in ``setUp`` and assign them
   back in ``tearDown`` — which materializes them as permanent module
   globals.  A later test that swaps ``KISS_HOME`` and calls
   ``save_config`` (e.g. ``test_ntfy_topic_isolation.py``, which saves
   ``remote_password="test-pw"``) then writes into the session-shared
   file instead of its isolated home, and its teardown — which only
   deletes the temp home — cannot undo the damage.

2. **Direct writes.**  A test that writes the shared ``config.json``
   and fails (or simply forgets) to restore it.

The tests below run IN FILE ORDER within one pytest process and act out
both leak paths for real, then prove — with a real
:class:`RemoteAccessServer` websocket handshake — that no password
survives into a later test.  On a conftest without the fixture, steps 2
and 4 fail exactly like the original flake.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import ssl
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

from kiss.core import vscode_config as vc
from kiss.core.vscode_config import load_config, save_config
from kiss.server.web_server import RemoteAccessServer


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


def test_step1_save_restore_teardown_pattern_pins_overrides() -> None:
    """Act out the ubiquitous save/restore pattern that pins the paths.

    ``self._orig = vc.CONFIG_DIR`` in ``setUp`` followed by
    ``vc.CONFIG_DIR = self._orig`` in ``tearDown`` restores the right
    *value* but turns the lazy attribute into a permanent module
    global.  The conftest fixture must delete the override after this
    test so the attribute is lazy again for the next one.
    """
    orig_dir = vc.CONFIG_DIR
    orig_path = vc.CONFIG_PATH
    vc.CONFIG_DIR = orig_dir
    vc.CONFIG_PATH = orig_path
    assert "CONFIG_DIR" in vars(vc)
    assert "CONFIG_PATH" in vars(vc)


def test_step2_swapped_home_save_lands_in_isolated_home() -> None:
    """After step 1, a ``KISS_HOME``-swapped save must stay isolated.

    Without the conftest guard the overrides pinned by step 1 survive,
    so ``save_config`` here writes ``remote_password`` into the
    session-shared ``config.json`` — the exact
    ``test_ntfy_topic_isolation.py`` leak.  With the guard the write
    lands in the temporary home and the shared config stays untouched.
    """
    assert "CONFIG_DIR" not in vars(vc), (
        "CONFIG_DIR override pinned by an earlier test survived; "
        "save_config in a KISS_HOME-swapped test will write into the "
        "session-shared config.json"
    )
    assert "CONFIG_PATH" not in vars(vc)

    with tempfile.TemporaryDirectory(prefix="kiss-leak-home-") as home:
        old_home = os.environ.get("KISS_HOME")
        os.environ["KISS_HOME"] = home
        try:
            save_config({"remote_password": "leaked-pw"})
            assert (Path(home) / "config.json").exists(), (
                "save_config did not follow the swapped KISS_HOME"
            )
        finally:
            if old_home is None:
                os.environ.pop("KISS_HOME", None)
            else:
                os.environ["KISS_HOME"] = old_home

    assert load_config().get("remote_password", "") == "", (
        "remote_password leaked into the session-shared config.json"
    )


def test_step3_leak_password_directly_without_restore() -> None:
    """Deliberately leak a password into the shared config (path 2).

    This test intentionally does NOT restore the config — the conftest
    fixture must roll the shared file back before the next test runs.
    """
    save_config({"remote_password": "direct-leak-pw"})
    assert load_config()["remote_password"] == "direct-leak-pw"


class TestAuthHandshakeAfterLeakers(IsolatedAsyncioTestCase):
    """A later real-server test must still authenticate without password.

    This is the original flake signature: with a leaked
    ``remote_password`` the handshake returns ``auth_required`` (and
    Playwright tests hang behind ``#auth-modal``); with a clean shared
    config it returns ``auth_ok``.
    """

    async def asyncSetUp(self) -> None:
        """Start a real ``RemoteAccessServer`` on the shared config."""
        self._port = _pick_free_port()
        self._server = RemoteAccessServer(
            host="127.0.0.1",
            port=self._port,
            work_dir=tempfile.mkdtemp(prefix="kiss-leak-wd-"),
            use_tunnel=False,
        )
        await self._server.start_async()

    async def asyncTearDown(self) -> None:
        """Stop the server."""
        await self._server.stop_async()

    async def _handshake_reply(self) -> Any:
        """Send one empty-password ``auth`` frame; return the reply type."""
        async with await connect(
            f"wss://127.0.0.1:{self._port}/ws", ssl=_no_verify_ssl(),
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            return json.loads(raw).get("type")

    async def test_step4_handshake_is_auth_ok_after_leaking_tests(self) -> None:
        """The empty-password handshake must succeed after steps 1–3."""
        self.assertEqual(
            await self._handshake_reply(),
            "auth_ok",
            "a remote_password leaked by an earlier test survived into "
            "the session-shared config.json (the auth_required != "
            "auth_ok / #auth-modal flake)",
        )
