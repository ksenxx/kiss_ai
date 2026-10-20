# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a remote replay answered without ``task_events`` releases its sink.

A ``resumeSession`` forwarded to a machine registers the asking
connection as the *replay sink* of its answer (``RemoteTab.replay_sinks``,
by the ``replayId`` the remote echoes on every frame of the answer).
The remote does not always reply with ``task_events``: a chat with no
history row and nothing running gets only ``status running:false``, a
finished side channel only ``subagentDone``.  Such an answer used to
leave the sink in place, so the NEXT transcript of the tab (another
connection's resume) was delivered to the stale sink and the connection
that asked never got its transcript, its page stuck in the
"waiting for status" state (audit 2026-10-20 C5-F2).  Now every answer
ends with the remote's ``replayDone``, whatever it consisted of.

Reproduction against a real remote daemon: connection A binds a tab to
the machine with a resume of a chat the remote has no row for (reply:
``status running:false``); connection B then resumes a seeded chat into
the same tab (same machine, same stream).  B must get the
``task_events`` and A must not.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from kiss.agents.sorcar import machines as mach
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import LocalReader, LocalWriter, open_local_connection

TAB = "tab-shared"


class TestRemoteReplaySinkReleased(IsolatedAsyncioTestCase):
    """One local daemon, one remote daemon seeded with two finished chats.

    ``chat-seeded`` is an ordinary task; ``chat-side`` is turned into the
    finished side channel (``/ask`` answerer) of that task by setting the
    typed columns its ``extra`` is read from, so a resume of it is
    answered with ``subagentDone``.
    """

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        mach.save_machines([])
        from kiss.tests.remote_daemon import start_remote_daemon

        remote_home = Path(self.tmpdir) / "remote-home"
        self.remote = await asyncio.to_thread(
            start_remote_daemon,
            remote_home,
            password="pw-remote",
            seed=[
                {
                    "chat_id": "chat-seeded",
                    "task": "Seeded task",
                    "result": "done",
                    "task_id": "seeded-1",
                },
                {
                    "chat_id": "chat-side",
                    "task": "Side task",
                    "result": "answered",
                    "task_id": "side-1",
                },
            ],
        )
        with sqlite3.connect(remote_home / "history.db", timeout=10) as db:
            db.execute(
                "UPDATE task_history SET parent_task_id = 'seeded-1', is_side_channel = 1"
                " WHERE id = 'side-1'",
            )
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=find_free_port(),
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        await self.server.start_async()
        self.conn_a = await open_local_connection(self.server)
        self.conn_b = await open_local_connection(self.server)
        await self._send(
            self.conn_a,
            {
                "type": "machineAction",
                "action": "add",
                "name": "box",
                "address": self.remote.url,
                "machinePassword": "pw-remote",
            },
        )
        added = await self._until(self.conn_a, "machines")
        self.assertEqual(added["error"], "")

    async def asyncTearDown(self) -> None:
        self.conn_a[1].close()
        self.conn_b[1].close()
        await self.server.stop_async()
        self.remote.stop()
        mach.save_machines([])
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _send(self, conn: tuple[LocalReader, LocalWriter], cmd: dict[str, Any]) -> None:
        conn[1].write(json.dumps(cmd).encode("utf-8") + b"\n")
        await conn[1].drain()

    async def _next(
        self,
        conn: tuple[LocalReader, LocalWriter],
        timeout: float = 60.0,
    ) -> dict[str, Any] | None:
        try:
            line = await asyncio.wait_for(conn[0].readline(), timeout=timeout)
        except TimeoutError:
            return None
        assert line, "local connection closed"
        msg: dict[str, Any] = json.loads(line.decode("utf-8"))
        return msg

    async def _until(
        self,
        conn: tuple[LocalReader, LocalWriter],
        reply_type: str,
        **fields: Any,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            msg = await self._next(conn, timeout=max(0.01, deadline - time.monotonic()))
            assert msg is not None, f"no {reply_type} reply within 45 s"
            if msg.get("type") == reply_type and all(msg.get(k) == v for k, v in fields.items()):
                return msg
        raise AssertionError(f"no {reply_type} reply within 45 s")

    async def _resume(
        self, conn: tuple[LocalReader, LocalWriter], chat_id: str, task_id: str = ""
    ) -> None:
        """Resume *chat_id* (or its *task_id*) of the machine into the shared tab, for *conn*."""
        cmd: dict[str, Any] = {
            "type": "resumeSession",
            "machine": "box",
            "chatId": chat_id,
            "tabId": TAB,
            "scoped": True,
        }
        if task_id:
            cmd["taskId"] = task_id  # a sub-agent's row is not found by chat id
        await self._send(conn, cmd)

    async def _seeded_transcript_reaches_b_only(self) -> None:
        """B resumes the seeded chat into the tab: its transcript goes to B, not A."""
        await self._resume(self.conn_b, "chat-seeded")
        events = await self._until(self.conn_b, "task_events", tabId=TAB)
        self.assertEqual(events["task"], "Seeded task")
        # A asked for nothing more: the seeded transcript must not reach
        # it.  Everything scheduled for A so far (a misrouted transcript
        # among it) precedes the reply to a request sent now.
        await self._send(self.conn_a, {"type": "getInputHistory"})
        while True:
            msg = await self._next(self.conn_a)
            assert msg is not None, "no inputHistory reply within 60 s"
            self.assertNotEqual(
                msg.get("type"),
                "task_events",
                f"B's transcript was delivered to A: {msg}",
            )
            if msg.get("type") == "inputHistory":
                break

    async def test_status_only_reply_releases_the_sink(self) -> None:
        """A's resume of a chat with no row is answered by a status alone."""
        await self._resume(self.conn_a, "chat-empty")
        status = await self._until(self.conn_a, "status", tabId=TAB)
        self.assertFalse(status["running"])
        await self._seeded_transcript_reaches_b_only()

    async def test_subagent_done_reply_releases_the_sink(self) -> None:
        """A's resume of a finished side channel is answered by ``subagentDone`` alone."""
        await self._resume(self.conn_a, "chat-side", task_id="side-1")
        await self._until(self.conn_a, "subagentDone", tab_id=TAB)
        await self._seeded_transcript_reaches_b_only()
