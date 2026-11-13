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
  and is shut out at once, whether it sends a frame or not: it gets
  ``auth_failed``, the close, and no event broadcast after the
  rotation.  A new session is admitted with the new password only.
* A password another process wrote into ``config.json``
  (``install.sh``) shuts a silent session out the same way.
* A big frame stops at the part that was waiting for a slow session to
  read the earlier ones when the password changed.
* An empty password saved from a local-token client disables remote
  access: a forwarded peer is turned away, the token client carries on.
* Clearing the password signs EVERY connected password-authenticated
  session out at once, the session that cleared it included (it is
  sent the shut-out, not the replies to its save), and no new remote
  handshake is admitted; a direct loopback browser is still served.

The browser side (the transport adopting the new password, the Settings
panel's box) is covered by ``webapp/test/remotePasswordEdit.test.js``
and ``webapp/test/transportFramePassword.test.js``.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
from typing import Any

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed
from websockets.typing import Origin

from kiss.agents.sorcar.local_endpoint import PART_CHARS
from kiss.core.user_config import api_keys_env_path, load_config, save_config
from kiss.server.web_server import _PART_ROOM_BYTES, _unsent_bytes
from kiss.tests.conftest import posix_only
from kiss.tests.server.conftest import no_verify_ssl
from kiss.tests.server.test_plain_http_loopback_and_frame_password import (
    AUTH_OK,
    AUTH_OK_LOCAL,
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
        self.assertEqual(await self._recv(ws), AUTH_OK)
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

    async def test_a_password_stored_before_a_failed_key_write_reaches_the_transport(self) -> None:
        """A save that stored the password and then failed still tells its sender.

        The key store cannot be locked (its lock path is a directory), so
        the save fails after ``config.json`` took the new password; the
        sender reconnects with the password it was told.  ``$HOME`` is a
        scratch directory: a key save also edits the shell RCs there.
        """
        old_home = os.environ.get("HOME")
        if old_home is None:
            self.addCleanup(os.environ.pop, "HOME", None)
        else:
            self.addCleanup(os.environ.__setitem__, "HOME", old_home)
        os.environ["HOME"] = str(self._endpoint_file.parent)
        lock = api_keys_env_path().with_name(".api_keys.env.kiss.lock")
        lock.unlink(missing_ok=True)
        lock.mkdir()
        self.addCleanup(lock.rmdir)
        ws = await self._authed()
        await ws.send(json.dumps({
            "type": "saveConfig", "config": {"remote_password": self.new_password},
            "apiKeys": {"KISS_TEST_UNSAVED_KEY": "made-up"}, "password": self.password,
        }))
        self.assertEqual((await self._recv_until(ws, "error"))["command"], "saveConfig")
        changed = await self._recv_until(ws, "remotePasswordChanged", "configData")
        self.assertEqual(changed["type"], "remotePasswordChanged")
        self.assertEqual(load_config()["remote_password"], changed["password"])
        await ws.close()
        await (await self._authed(changed["password"])).close()

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
        # The other session was not told the new password.
        await self._assert_shut_out(other)
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
            self.assertEqual(await self._recv(ws), AUTH_OK_LOCAL)
            await self._save(ws, "")
            await ws.send(json.dumps({"type": "ping"}))
            self.assertEqual((await self._recv(ws))["type"], "pong")
        async with connect(
            f"wss://127.0.0.1:{self._port}/ws", ssl=no_verify_ssl(),
            additional_headers=self._forwarded,
        ) as remote:
            # Refused before any frame is read (sending one may fail).
            self.assertEqual((await self._recv(remote)).get("code"), "localhost_only")

    async def _assert_shut_out(self, ws: Any) -> list[dict[str, Any]]:
        """The daemon shuts *ws* out without a frame from it: ``auth_failed``, then the close.

        Returns:
            Every frame *ws* was sent before the socket closed (within 5 s).
        """
        frames: list[dict[str, Any]] = []
        try:
            async with asyncio.timeout(5):
                while True:
                    frames.append(await self._recv(ws))
        except ConnectionClosed:
            pass
        self.assertEqual([frame.get("code") for frame in frames[-1:]], ["auth_failed"], frames)
        return frames

    async def test_a_rotation_shuts_a_silent_session_out_at_once(self) -> None:
        """A session that sends nothing stops receiving events when another one rotates.

        The per-frame check never sees it.  The session that rotated, a
        local-token client and the browser on this machine carry on.
        """
        silent = await self._authed()
        rotator = await self._authed()
        token = await connect(f"ws://127.0.0.1:{self._port}/ws")
        await token.send(json.dumps({"type": "auth", "token": self._token()}))
        self.assertEqual(await self._recv(token), AUTH_OK_LOCAL)
        browser = await connect(
            f"ws://127.0.0.1:{self._port}/ws", origin=Origin(f"http://127.0.0.1:{self._port}"),
        )
        await browser.send(json.dumps({"type": "auth"}))
        self.assertEqual(await self._recv(browser), AUTH_OK)

        await self._save(rotator, self.new_password, password=self.password)
        self._server._printer.broadcast({"type": "after_rotation"})

        frames = await self._assert_shut_out(silent)
        self.assertNotIn("after_rotation", [frame["type"] for frame in frames])
        for kept in (rotator, token, browser):
            await self._recv_until(kept, "after_rotation")
            await kept.close()

    async def test_a_password_written_by_another_process_shuts_a_silent_session_out(self) -> None:
        """``install.sh`` writes ``config.json`` itself: no command of the daemon announces it."""
        silent = await self._authed()
        save_config({"remote_password": self.new_password})
        await self._assert_shut_out(silent)

    # A later part is held back by the bytes the reader's kernel has
    # not taken, which the daemon can count on Linux and macOS only;
    # elsewhere the whole frame may be written before the password changes.
    @posix_only("_unsent_bytes counts the kernel's send queue on Linux and macOS only")
    async def test_a_part_waiting_for_room_is_not_sent_once_the_password_changed(self) -> None:
        """The password changes while a big frame's later part waits for a slow reader.

        The reader's socket buffer is 16 KiB and it reads nothing.  The
        frame has enough parts to fill macOS's auto-grown TCP window,
        so the daemon must hold a later part until earlier ones are read.
        """
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024)
        sock.connect(("127.0.0.1", self._port))
        reader = await connect(
            f"ws://127.0.0.1:{self._port}/ws", sock=sock, compression=None,
            additional_headers=self._forwarded,
        )
        await reader.send(json.dumps({"type": "auth", "password": self.password, "parts": True}))
        self.assertEqual(await self._recv(reader), AUTH_OK)
        reader.transport.pause_reading()
        endpoint = next(iter(self._server._password_sessions))
        # Fill even the TCP receive window that macOS grows during connect().
        self._server._printer.broadcast(
            {"type": "big", "text": "\\" * (16 * PART_CHARS)},
        )
        async with asyncio.timeout(5):
            while _unsent_bytes(endpoint.transport) <= _PART_ROOM_BYTES:
                await asyncio.sleep(0.005)

        save_config({"remote_password": self.new_password})
        reader.transport.resume_reading()

        frames = await self._assert_shut_out(reader)
        parts = [frame for frame in frames if frame["type"] == "part"]
        self.assertTrue(parts)
        self.assertEqual([frame["seq"] for frame in parts], list(range(len(parts))))
        self.assertTrue(all(frame.get("more") is True for frame in parts))

    async def test_clearing_the_password_from_a_remote_session_signs_every_remote_session_out(
        self,
    ) -> None:
        other = await self._authed()
        await other.send(json.dumps({"type": "ping", "password": self.password}))
        self.assertEqual((await self._recv(other))["type"], "pong")
        ws = await self._authed()
        await ws.send(json.dumps({
            "type": "saveConfig", "config": {"remote_password": ""}, "password": self.password,
        }))
        # Remote access is off from the save on: the session that
        # cleared it is sent the shut-out alone, not the replies to its
        # save, and none of the connected ones is left to set a new
        # password and turn remote access back on.
        self.assertEqual([frame["type"] for frame in await self._assert_shut_out(ws)], ["error"])
        await self._assert_shut_out(other)
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
        remote = await self._authed()
        async with connect(f"ws://127.0.0.1:{self._port}/ws") as ws:
            await ws.send(json.dumps({"type": "auth", "token": self._token()}))
            self.assertEqual(await self._recv(ws), AUTH_OK_LOCAL)
            await self._save(ws, "")
        await self._assert_shut_out(remote)
        # A loopback browser without a token is still served bare.
        async with connect(f"ws://127.0.0.1:{self._port}/ws") as loopback:
            await loopback.send(json.dumps({"type": "auth"}))
            self.assertEqual(await self._recv(loopback), AUTH_OK)
            await loopback.send(json.dumps({"type": "ping"}))
            self.assertEqual((await self._recv(loopback))["type"], "pong")
