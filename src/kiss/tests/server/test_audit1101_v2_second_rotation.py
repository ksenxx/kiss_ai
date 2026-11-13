# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The old-password allowance of a rotating session ends with a later rotation.

Audit 2026-11-01 (V2 F2): a remote session that changes the password
(``saveConfig`` with a new ``remote_password``) keeps being served for
frames still carrying the password it was admitted with, until its
first frame with the new one (``conn_state["rotated_from"]``,
``RemoteAccessServer._frame_password_accepted``).  That allowance
recorded only the OLD password, not which rotation it belonged to: when
another session (or ``install.sh``) then set a third password, the
first session's stale frames still passed — against a password nobody
had set from that connection — although every other session admitted
with the old password was shut out, and a ``getConfig`` so admitted
disclosed the current configuration including the replacement.

Live ``RemoteAccessServer`` over loopback with a configured password,
two forwarded (password-authenticated) sessions, real ``config.json``
writes through ``saveConfig``.  No mocks.
"""

from __future__ import annotations

import json
from typing import Any

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from kiss.tests.server.conftest import no_verify_ssl
from kiss.tests.server.test_plain_http_loopback_and_frame_password import (
    AUTH_OK,
    _ServerCase,
)


class TestSecondRotationEndsTheAllowance(_ServerCase):
    """P0 -> P1 by session A, then P1 -> P2 by session B: A, still at P0, is shut out."""

    password = "First-Password-0"
    second_password = "Second-Password-1"
    third_password = "Third-Password-2"
    _forwarded = {"X-Forwarded-For": "203.0.113.9"}

    async def _authed(self, password: str) -> Any:
        ws = await connect(
            f"wss://127.0.0.1:{self._port}/ws", ssl=no_verify_ssl(),
            additional_headers=self._forwarded,
        )
        await ws.send(json.dumps({"type": "auth", "password": password}))
        self.assertEqual(await self._recv(ws), AUTH_OK)
        return ws

    async def _recv_until(self, ws: Any, *types: str) -> dict[str, Any]:
        for _ in range(20):
            msg = await self._recv(ws)
            if msg.get("type") in types:
                return msg
        raise AssertionError(f"no {types} reply")

    async def _rotate(self, ws: Any, current: str, new: str) -> None:
        await ws.send(json.dumps({
            "type": "saveConfig", "config": {"remote_password": new}, "password": current,
        }))
        changed = await self._recv_until(ws, "remotePasswordChanged", "configData")
        self.assertEqual(changed["type"], "remotePasswordChanged", changed)
        self.assertEqual(changed["password"], new)
        self.assertEqual((await self._recv_until(ws, "configData"))["type"], "configData")

    async def test_stale_frames_are_refused_after_an_unrelated_rotation(self) -> None:
        session_a = await self._authed(self.password)
        await self._rotate(session_a, self.password, self.second_password)
        # A's own rotation: a frame still carrying the old password is served.
        await session_a.send(json.dumps({"type": "ping", "password": self.password}))
        self.assertEqual((await self._recv(session_a))["type"], "pong")

        session_b = await self._authed(self.second_password)
        await self._rotate(session_b, self.second_password, self.third_password)
        await session_b.close()

        # The password in force is one A never asked for: A's old
        # password is as stale as any other session's, so A is shut
        # out without a frame from it and is not told the configuration.
        frames: list[dict[str, Any]] = []
        with self.assertRaises(ConnectionClosed):
            while True:
                frames.append(await self._recv(session_a))
        self.assertEqual([frame.get("code") for frame in frames[-1:]], ["auth_failed"], frames)
        self.assertNotIn("configData", [frame["type"] for frame in frames])


if __name__ == "__main__":
    import unittest

    unittest.main()
