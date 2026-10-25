# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A non-``auth`` frame ends the password handshake the same way on both attempts.

Audit 2026-10-25 (server-main, simplify): after a wrong password the
handshake sends ``auth_required`` and reads one more frame.  A frame of
another type as the FIRST message closed the socket without a word,
but the same frame as the SECOND one fell out of the retry loop and
was refused with "That password is not correct" although no password
was tried.  Both attempts now close the socket on a non-``auth`` frame.
Real :class:`RemoteAccessServer` over a real TLS WebSocket, with the
password in a scratch ``KISS_HOME``.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from kiss.core.user_config import save_config
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.conftest import (
    environ,
    find_free_port,
    no_verify_ssl,
    redirect_persistence,
    restore_persistence,
)


class TestNonAuthSecondFrame(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-auth-frame-"))
        self._saved = redirect_persistence(self.tmp)
        self._env = environ(KISS_HOME=str(self.tmp / "home"))
        self._env.__enter__()
        save_config({"remote_password": "correct-horse"})
        self.port = find_free_port()
        self.server = RemoteAccessServer(host="127.0.0.1", port=self.port, work_dir=str(self.tmp))
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        self._env.__exit__(None, None, None)
        restore_persistence(self._saved)
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def _replies_to(self, frames: list[dict[str, object]]) -> list[dict[str, object]]:
        """Send *frames* in turn from a relayed (non-loopback) peer; return every reply."""
        replies: list[dict[str, object]] = []
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=no_verify_ssl(),
            additional_headers={"Cf-Connecting-Ip": "203.0.113.7"},
        ) as ws:
            try:
                for frame in frames:
                    await ws.send(json.dumps(frame))
                    replies.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=5)))
            except ConnectionClosed:
                replies.append({"type": "closed"})
        return replies

    async def test_non_auth_first_frame_closes_silently(self) -> None:
        self.assertEqual(await self._replies_to([{"type": "ping"}]), [{"type": "closed"}])

    async def test_non_auth_second_frame_closes_silently_too(self) -> None:
        replies = await self._replies_to([
            {"type": "auth", "password": "wrong"}, {"type": "ping"},
        ])
        self.assertEqual(replies, [{"type": "auth_required"}, {"type": "closed"}])

    async def test_wrong_password_twice_is_still_reported(self) -> None:
        replies = await self._replies_to([
            {"type": "auth", "password": "wrong"}, {"type": "auth", "password": "wrong again"},
        ])
        self.assertEqual(replies[0], {"type": "auth_required"})
        self.assertEqual(replies[1]["type"], "error")
        self.assertEqual(replies[1]["code"], "auth_failed")


if __name__ == "__main__":
    unittest.main()
