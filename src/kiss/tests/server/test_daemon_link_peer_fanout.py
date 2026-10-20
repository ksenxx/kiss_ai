# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""A daemon-link peer is answered but never fanned task events to.

Another daemon's ``MachineLink`` keeps one request socket open to this
daemon (``ping``, ``getHistory``, ...) and declares it with
``"link": true`` in its ``auth`` frame.  Before this change that
socket was an ordinary peer: every tab-stamped task event — streamed
tokens, status, and a whole ``task_events`` transcript replayed by
someone's history click in a UI on this machine — was pushed at it.
A transcript larger than the link's 1 MiB frame limit closed the
socket with 1009 and the other daemon showed this machine as
"unreachable" for its back-off period.

These tests run a real ``RemoteAccessServer`` and real websocket
clients: a link peer that authenticated with the local token, one that
authenticated with the password, and a plain local peer.  The link
peers get the replies to their own requests and global events; only
the plain peer gets task copies.  The link peers deliberately keep the
old 1 MiB frame limit, so if a 2 MB transcript reached them their
sockets would die — which is exactly what the later ``ping`` proves
did not happen.
"""

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import ClientConnection, connect

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar import local_endpoint
from kiss.server import agent_state
from kiss.server.web_server import RemoteAccessServer

ONE_MIB = 1024 * 1024
BIG_TEXT = "x" * (2 * ONE_MIB)


async def _recv_until(
    ws: ClientConnection, wanted: str, seconds: float,
) -> dict[str, Any] | None:
    """Return the first *wanted* event *ws* receives within *seconds*, else None."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            return None
        try:
            raw = await asyncio.wait_for(ws.recv(), remaining)
        except TimeoutError:
            return None
        event: dict[str, Any] = json.loads(raw)
        if event.get("type") == wanted:
            return event


class TestDaemonLinkPeerFanout(IsolatedAsyncioTestCase):
    """Task copies skip link peers; replies and global events reach them."""

    async def asyncSetUp(self) -> None:
        agent_state.agent_states.clear()
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-link-fanout-")
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        th._db_conn = None
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        await self.server.start_async()
        endpoint = local_endpoint.read_endpoint(Path(self.server._local_endpoint_file))
        assert endpoint is not None, "daemon endpoint file missing"
        self.endpoint = endpoint
        self.ssl_ctx = local_endpoint.client_ssl_context(endpoint)

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect(self, auth: dict[str, Any], max_size: int | None) -> ClientConnection:
        """Open a client that authenticates with *auth* and accepts frames up to *max_size*."""
        ws = await connect(
            self.endpoint.url, ssl=self.ssl_ctx, compression=None,
            max_size=max_size, close_timeout=2.0,
        )
        self.addAsyncCleanup(ws.close)
        await ws.send(json.dumps({"type": "auth", **auth}))
        reply = json.loads(await asyncio.wait_for(ws.recv(), 10.0))
        self.assertEqual(reply.get("type"), "auth_ok", reply)
        return ws

    async def _wait_for_clients(self, links: int, local: int) -> None:
        printer = self.server._printer
        for _ in range(300):
            with printer._ws_lock:
                if len(printer._link_clients) >= links and len(printer._local_clients) >= local:
                    return
            await asyncio.sleep(0.01)
        raise AssertionError(f"server never registered {links} link + {local} local clients")

    async def _ping(self, ws: ClientConnection) -> None:
        await ws.send(json.dumps({"type": "ping"}))
        self.assertIsNotNone(await _recv_until(ws, "pong", 5.0), "ping went unanswered")

    async def test_link_peers_get_replies_and_global_events_only(self) -> None:
        """Task copies reach the plain peer; link peers keep a 1 MiB limit and survive."""
        token_auth = {"token": self.endpoint.token}
        link_local = await self._connect({**token_auth, "link": True}, ONE_MIB)
        link_remote = await self._connect({"password": "", "link": True}, ONE_MIB)
        plain = await self._connect(token_auth, 64 * ONE_MIB)
        await self._wait_for_clients(links=2, local=1)
        printer = self.server._printer
        with printer._ws_lock:
            self.assertEqual(len(printer._remote_clients), 0,
                             "a password-authenticated link peer must not be a remote client")

        # Connection-scoped replies still reach a link peer.
        await self._ping(link_local)
        await self._ping(link_remote)

        # A streamed token (a fan-out copy) and a tab-stamped transcript
        # larger than the link peers' frame limit: only the plain peer
        # receives them.
        printer.subscribe_tab("task-1", "tab-a")
        await asyncio.to_thread(
            printer.broadcast,
            {"type": "text_delta", "taskId": "task-1", "text": "hello"},
        )
        await asyncio.to_thread(
            printer.broadcast,
            {"type": "task_events", "tabId": "tab-a", "taskId": "task-1",
             "events": [{"type": "text", "text": BIG_TEXT}]},
        )
        delta = await _recv_until(plain, "text_delta", 5.0)
        self.assertIsNotNone(delta, "the plain peer must get the streamed token")
        transcript = await _recv_until(plain, "task_events", 10.0)
        self.assertIsNotNone(transcript, "the plain peer must get the transcript")
        assert transcript is not None
        self.assertGreater(len(json.dumps(transcript)), ONE_MIB)

        # Global events reach every peer, link peers included: both an
        # unstamped one and one sent verbatim with an empty ``tabId``
        # stamp (``tabs_state``, the snapshot a ``ready`` is answered
        # with).
        await asyncio.to_thread(printer.broadcast, {"type": "tasks_updated"})
        await asyncio.to_thread(
            printer.broadcast, {"type": "tabs_state", "tabs": [], "tabId": ""},
        )
        for ws in (link_local, link_remote, plain):
            self.assertIsNotNone(await _recv_until(ws, "tasks_updated", 5.0))
            self.assertIsNotNone(await _recv_until(ws, "tabs_state", 5.0))

        # Both link peers are still alive and answer: neither the token
        # nor the 2 MB transcript was pushed at them (the transcript
        # would have closed their 1 MiB sockets with 1009).
        await self._ping(link_local)
        await self._ping(link_remote)

        # Closing a link peer drops it from the link set.
        await link_remote.close()
        for _ in range(300):
            with printer._ws_lock:
                if len(printer._link_clients) == 1:
                    break
            await asyncio.sleep(0.01)
        with printer._ws_lock:
            self.assertEqual(len(printer._link_clients), 1)
