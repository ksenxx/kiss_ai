"""E2E: the Settings panel's "Rotate" button (``rotateRemotePassword``).

Against a live ``RemoteAccessServer`` with a ``remote_password``
configured:

* A password-authenticated (forwarded) session that sends
  ``rotateRemotePassword`` gets ``remotePasswordRotated`` carrying the
  new password FIRST, then a ``configData`` whose ``remote_password`` is
  the new one; ``config.json`` holds it; the same session is served
  when it repeats the new password and shut out when it falls back to
  the old one.
* Frames that session sent with the old password before the event
  reached it (a command in flight) are still served; its first frame
  with the new password ends that allowance, and a fresh session with
  the old password is turned away.
* Another password-authenticated session is not told the new password
  and is shut out at its next command; a new session is admitted with
  the new password only.
* A local-token client can rotate too and is unaffected afterwards (its
  frames never carry a password).
* When ``config.json`` cannot be read the requester gets ``error`` /
  ``rotate_remote_password_failed`` and the stored password is left as
  it was (skipped as root, which file modes do not stop).

The browser side (the transport adopting the new password, the Settings
panel revealing it) is covered by
``webapp/test/remotePasswordRotate.test.js`` and
``webapp/test/transportFramePassword.test.js``.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest import skipIf

from websockets.asyncio.client import connect

from kiss.core.user_config import CONFIG_PATH, load_config
from kiss.tests.server.conftest import no_verify_ssl
from kiss.tests.server.test_plain_http_loopback_and_frame_password import (
    _ServerCase,
)


class TestRotateRemotePassword(_ServerCase):
    """``rotateRemotePassword`` over password- and token-authenticated sessions."""

    password = "Correct-Horse-Battery-1"
    _forwarded = {"X-Forwarded-For": "203.0.113.9"}

    async def _authed(self, password: str | None = None) -> Any:
        pwd = self.password if password is None else password
        ws = await connect(
            f"wss://127.0.0.1:{self._port}/ws", ssl=no_verify_ssl(),
            additional_headers=self._forwarded,
        )
        await ws.send(json.dumps({"type": "auth", "password": pwd}))
        self.assertEqual(await self._recv(ws), {"type": "auth_ok", "local": False})
        return ws

    async def _rotate(self, ws: Any, **frame: Any) -> str:
        """Send ``rotateRemotePassword`` and return the new password after checking the replies."""
        await ws.send(json.dumps({"type": "rotateRemotePassword", **frame}))
        rotated = await self._recv(ws)
        self.assertEqual(rotated["type"], "remotePasswordRotated", rotated)
        new_password = rotated["password"]
        self.assertIsInstance(new_password, str)
        self.assertGreaterEqual(len(new_password), 12)
        self.assertNotEqual(new_password, self.password)
        config = await self._recv(ws)
        self.assertEqual(config["type"], "configData", config)
        self.assertEqual(config["config"]["remote_password"], new_password)
        self.assertEqual(load_config()["remote_password"], new_password)
        return str(new_password)

    async def test_remote_session_rotates_and_carries_on_with_the_new_password(self) -> None:
        ws = await self._authed()
        new_password = await self._rotate(ws, password=self.password)
        await ws.send(json.dumps({"type": "ping", "password": new_password}))
        self.assertEqual((await self._recv(ws))["type"], "pong")
        await ws.send(json.dumps({"type": "ping", "password": self.password}))
        self.assertEqual((await self._recv(ws))["code"], "auth_failed")
        await ws.close()

    async def test_frames_in_flight_with_the_old_password_are_still_served(self) -> None:
        """Commands the rotating browser sent before it learnt the new
        password (a request in flight, the ping after a reconnect's
        queued batch) still carry the old one; they are served, and the
        first frame with the new password ends that allowance."""
        ws = await self._authed()
        await ws.send(json.dumps({"type": "rotateRemotePassword", "password": self.password}))
        await ws.send(json.dumps({"type": "ping", "password": self.password}))
        await ws.send(json.dumps({"type": "getConfig", "password": self.password}))
        rotated = await self._recv(ws)
        self.assertEqual(rotated["type"], "remotePasswordRotated", rotated)
        new_password = rotated["password"]
        self.assertEqual((await self._recv(ws))["type"], "configData")
        self.assertEqual((await self._recv(ws))["type"], "pong")
        config = await self._recv(ws)
        self.assertEqual(config["type"], "configData", config)
        self.assertEqual(config["config"]["remote_password"], new_password)
        # Adopted: from now on only the new password is good.
        await ws.send(json.dumps({"type": "ping", "password": new_password}))
        self.assertEqual((await self._recv(ws))["type"], "pong")
        await ws.send(json.dumps({"type": "ping", "password": self.password}))
        self.assertEqual((await self._recv(ws))["code"], "auth_failed")
        await ws.close()
        # The allowance is this connection's alone: a fresh session
        # with the old password is turned away.
        async with connect(
            f"wss://127.0.0.1:{self._port}/ws", ssl=no_verify_ssl(),
            additional_headers=self._forwarded,
        ) as stale:
            await stale.send(json.dumps({"type": "auth", "password": self.password}))
            self.assertEqual((await self._recv(stale))["type"], "auth_required")

    async def test_other_sessions_are_shut_out_and_the_new_password_admits(self) -> None:
        other = await self._authed()
        await other.send(json.dumps({"type": "ping", "password": self.password}))
        self.assertEqual((await self._recv(other))["type"], "pong")
        ws = await self._authed()
        new_password = await self._rotate(ws, password=self.password)
        await ws.close()
        # The other session was not told the new password: its next
        # frame (old password) is answered by the shut-out, nothing else.
        await other.send(json.dumps({"type": "ping", "password": self.password}))
        self.assertEqual((await self._recv(other))["code"], "auth_failed")
        await other.close()
        async with connect(
            f"wss://127.0.0.1:{self._port}/ws", ssl=no_verify_ssl(),
            additional_headers=self._forwarded,
        ) as stale:
            await stale.send(json.dumps({"type": "auth", "password": self.password}))
            self.assertEqual((await self._recv(stale))["type"], "auth_required")
        fresh = await self._authed(new_password)
        await fresh.send(json.dumps({"type": "ping", "password": new_password}))
        self.assertEqual((await self._recv(fresh))["type"], "pong")
        await fresh.close()

    async def test_token_client_rotates_and_is_unaffected(self) -> None:
        async with connect(f"ws://127.0.0.1:{self._port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "token": self._token()}))
            self.assertEqual(await self._recv(ws), {"type": "auth_ok", "local": True})
            await self._rotate(ws)
            await ws.send(json.dumps({"type": "ping"}))
            self.assertEqual((await self._recv(ws))["type"], "pong")

    @skipIf(
        os.name != "posix" or os.geteuid() == 0,
        "needs a file mode root cannot bypass",
    )
    async def test_unreadable_config_fails_and_keeps_the_password(self) -> None:
        async with connect(f"ws://127.0.0.1:{self._port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "token": self._token()}))
            self.assertEqual(await self._recv(ws), {"type": "auth_ok", "local": True})
            CONFIG_PATH.chmod(0)
            try:
                await ws.send(json.dumps({"type": "rotateRemotePassword"}))
                reply = await self._recv(ws)
            finally:
                CONFIG_PATH.chmod(0o600)
            self.assertEqual(reply["type"], "error", reply)
            self.assertEqual(reply["code"], "rotate_remote_password_failed")
            self.assertIn("cannot be read", reply["text"])
        self.assertEqual(load_config()["remote_password"], self.password)
