"""E2E: the Settings panel's editable Remote password (``saveConfig``).

Against a live ``RemoteAccessServer`` with a ``remote_password``
configured:

* A password-authenticated (forwarded) session whose ``saveConfig``
  carries a different ``remote_password`` gets ``remotePasswordChanged``
  with the new password BEFORE its ``configData`` (which shows the new
  one); ``config.json`` holds it at once, with no daemon restart; the
  same session is served when it repeats the new password and shut out
  when it falls back to the old one.
* Frames that session sent with the old password before the event
  reached it (a command in flight) are still served; its first frame
  with the new password ends that allowance, and a fresh session with
  the old password is turned away.
* A ``saveConfig`` that repeats the stored password (the Settings panel
  echoing an untouched box) changes nothing and sends no
  ``remotePasswordChanged``.
* Another password-authenticated session is not told the new password
  and is shut out at its next command; a new session is admitted with
  the new password only.
* An empty password saved from a local-token client disables remote
  access: a forwarded peer is turned away, the token client carries on.
* Clearing the password signs EVERY connected password-authenticated
  session out at its next frame (old password, empty password or no
  password at all), the session that cleared it included, and no new
  remote handshake is admitted; a direct loopback browser is still
  served.

The browser side (the transport adopting the new password, the Settings
panel's box) is covered by ``webapp/test/remotePasswordEdit.test.js``
and ``webapp/test/transportFramePassword.test.js``.
"""

from __future__ import annotations

import json
from typing import Any

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from kiss.core.user_config import load_config
from kiss.tests.server.conftest import no_verify_ssl
from kiss.tests.server.test_plain_http_loopback_and_frame_password import (
    _ServerCase,
)


class TestSaveConfigRemotePassword(_ServerCase):
    """``saveConfig`` with ``remote_password`` over password- and token-authenticated sessions."""

    password = "Correct-Horse-Battery-1"
    new_password = "Staple-Battery-Horse-2"
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

    async def _recv_until(self, ws: Any, *types: str) -> dict[str, Any]:
        """Return the first reply of a type in *types* (``saveConfig`` answers ``models`` too)."""
        for _ in range(20):
            msg = await self._recv(ws)
            if msg.get("type") in types:
                return msg
        raise AssertionError(f"no {types} reply")

    async def _save(self, ws: Any, new: str, **frame: Any) -> None:
        """Save the password *new* over *ws* and check the replies and ``config.json``."""
        await ws.send(json.dumps({
            "type": "saveConfig", "config": {"remote_password": new}, **frame,
        }))
        changed = await self._recv_until(ws, "remotePasswordChanged", "configData")
        self.assertEqual(changed["type"], "remotePasswordChanged", changed)
        self.assertEqual(changed["password"], new)
        config = await self._recv_until(ws, "configData")
        self.assertEqual(config["config"]["remote_password"], new)
        self.assertEqual(load_config()["remote_password"], new)

    async def test_remote_session_saves_and_carries_on_with_the_new_password(self) -> None:
        ws = await self._authed()
        await self._save(ws, self.new_password, password=self.password)
        await ws.send(json.dumps({"type": "ping", "password": self.new_password}))
        self.assertEqual((await self._recv(ws))["type"], "pong")
        await ws.send(json.dumps({"type": "ping", "password": self.password}))
        self.assertEqual((await self._recv(ws))["code"], "auth_failed")
        await ws.close()

    async def test_frames_in_flight_with_the_old_password_are_still_served(self) -> None:
        ws = await self._authed()
        await ws.send(json.dumps({
            "type": "saveConfig", "config": {"remote_password": self.new_password},
            "password": self.password,
        }))
        await ws.send(json.dumps({"type": "ping", "password": self.password}))
        await ws.send(json.dumps({"type": "getConfig", "password": self.password}))
        changed = await self._recv_until(ws, "remotePasswordChanged", "configData", "pong")
        self.assertEqual(changed["type"], "remotePasswordChanged", changed)
        self.assertEqual(changed["password"], self.new_password)
        self.assertEqual((await self._recv_until(ws, "configData"))["type"], "configData")
        self.assertEqual((await self._recv_until(ws, "pong", "error"))["type"], "pong")
        config = await self._recv_until(ws, "configData", "error")
        self.assertEqual(config["type"], "configData", config)
        self.assertEqual(config["config"]["remote_password"], self.new_password)
        # Adopted: from now on only the new password is good.
        await ws.send(json.dumps({"type": "ping", "password": self.new_password}))
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

    async def test_saving_the_same_password_changes_nothing(self) -> None:
        ws = await self._authed()
        await ws.send(json.dumps({
            "type": "saveConfig", "config": {"remote_password": self.password, "max_budget": 7},
            "password": self.password,
        }))
        config = await self._recv_until(ws, "remotePasswordChanged", "configData")
        self.assertEqual(config["type"], "configData", config)
        self.assertEqual(config["config"]["remote_password"], self.password)
        self.assertEqual(config["config"]["max_budget"], 7)
        await ws.send(json.dumps({"type": "ping", "password": self.password}))
        self.assertEqual((await self._recv(ws))["type"], "pong")
        await ws.close()

    async def test_other_sessions_are_shut_out_and_the_new_password_admits(self) -> None:
        other = await self._authed()
        await other.send(json.dumps({"type": "ping", "password": self.password}))
        self.assertEqual((await self._recv(other))["type"], "pong")
        ws = await self._authed()
        await self._save(ws, self.new_password, password=self.password)
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
        fresh = await self._authed(self.new_password)
        await fresh.send(json.dumps({"type": "ping", "password": self.new_password}))
        self.assertEqual((await self._recv(fresh))["type"], "pong")
        await fresh.close()

    async def test_an_empty_password_disables_remote_access(self) -> None:
        async with connect(f"ws://127.0.0.1:{self._port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "token": self._token()}))
            self.assertEqual(await self._recv(ws), {"type": "auth_ok", "local": True})
            await self._save(ws, "")
            await ws.send(json.dumps({"type": "ping"}))
            self.assertEqual((await self._recv(ws))["type"], "pong")
        async with connect(
            f"wss://127.0.0.1:{self._port}/ws", ssl=no_verify_ssl(),
            additional_headers=self._forwarded,
        ) as remote:
            # Refused before any frame is read (sending one may fail).
            self.assertEqual((await self._recv(remote)).get("code"), "localhost_only")

    async def _assert_shut_out(self, ws: Any, **frame: Any) -> None:
        """The next frame of *ws* is refused with ``auth_failed`` and the socket closed.

        Args:
            ws: The admitted connection.
            frame: Extra fields of the frame, e.g. ``password=...``.
        """
        await ws.send(json.dumps({"type": "getConfig", **frame}))
        reply = await self._recv(ws)
        self.assertEqual(reply.get("code"), "auth_failed", reply)
        with self.assertRaises(ConnectionClosed):
            await ws.recv()

    async def test_clearing_the_password_from_a_remote_session_signs_every_remote_session_out(
        self,
    ) -> None:
        other = await self._authed()
        await other.send(json.dumps({"type": "ping", "password": self.password}))
        self.assertEqual((await self._recv(other))["type"], "pong")
        ws = await self._authed()
        await self._save(ws, "", password=self.password)
        # The session that cleared it keeps no allowance: its old
        # password, a bare frame and the empty password are all refused.
        await self._assert_shut_out(ws, password=self.password)
        # Already-connected sessions are refused whatever they present,
        # a bare frame included, so none of them can set a new password
        # and turn remote access back on.
        await other.send(json.dumps({
            "type": "saveConfig", "config": {"remote_password": self.new_password},
        }))
        self.assertEqual((await self._recv(other))["code"], "auth_failed")
        with self.assertRaises(ConnectionClosed):
            await other.recv()
        self.assertEqual(load_config()["remote_password"], "")
        # And no new relayed session is admitted: the handshake is
        # refused before any frame is read.
        async with connect(
            f"wss://127.0.0.1:{self._port}/ws", ssl=no_verify_ssl(),
            additional_headers=self._forwarded,
        ) as fresh:
            self.assertEqual((await self._recv(fresh)).get("code"), "localhost_only")

    async def test_clearing_the_password_from_a_token_client_signs_remote_sessions_out(
        self,
    ) -> None:
        remote_old = await self._authed()
        remote_bare = await self._authed()
        async with connect(f"ws://127.0.0.1:{self._port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "token": self._token()}))
            self.assertEqual(await self._recv(ws), {"type": "auth_ok", "local": True})
            await self._save(ws, "")
        await self._assert_shut_out(remote_old, password=self.password)
        await self._assert_shut_out(remote_bare)
        # A loopback browser without a token is still served bare.
        async with connect(f"ws://127.0.0.1:{self._port}/ws") as loopback:
            await loopback.send(json.dumps({"type": "auth"}))
            self.assertEqual(await self._recv(loopback), {"type": "auth_ok", "local": False})
            await loopback.send(json.dumps({"type": "ping"}))
            self.assertEqual((await self._recv(loopback))["type"], "pong")
