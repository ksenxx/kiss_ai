# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a remote replay's answer is told apart from the tab's other frames by id.

A ``resumeSession`` forwarded to a machine for one client registers
that connection as the *replay sink* of the answer.  The sink used to
be a per-tab FIFO that ANY ``task_events``, ``status running:false`` or
``subagentDone`` of the tab consumed, so a generic frame of the tab, a
task of the tab ending on the machine between the registration of
client B's sink and B's transcript, took B's sink: B's transcript then
found no sink and was broadcast, to client A too (review 2026-10-20
R6 of C5-F2).  Now the forwarded resume carries a ``replayId`` the
machine echoes on every frame of the answer and ends with
``replayDone``; the relay routes by that id alone.

The race is pinned without sleeps: the stream's lock is an ``RLock``
the relay thread takes per frame, so holding it from the test pauses
relaying while the test still routes B's resume itself; an observer
connection on the remote confirms the task-end status is out before
B's resume goes, so the stream carries the status ahead of B's
transcript, and the relay sees both only after B's sink exists.
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

from websockets.sync.client import ClientConnection

from kiss.agents.sorcar import machines as mach
from kiss.server.remote_tabs import RemoteTab
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import LocalReader, LocalWriter, open_local_connection

TAB = "tab-shared"
Conn = tuple[LocalReader, LocalWriter]


class TestRemoteReplayCorrelation(IsolatedAsyncioTestCase):
    """One local daemon with two clients, one remote daemon seeded with a chat."""

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
        await self._fence(self.conn_a)
        known = set(self.server._printer._conn_endpoints)
        self.conn_b = await open_local_connection(self.server)
        await self._fence(self.conn_b)
        (self.conn_b_id,) = set(self.server._printer._conn_endpoints) - known
        self.observer: ClientConnection | None = None
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
        if self.observer is not None:
            self.observer.close()
        self.conn_a[1].close()
        self.conn_b[1].close()
        await self.server.stop_async()
        self.remote.stop()
        mach.save_machines([])
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _send(self, conn: Conn, cmd: dict[str, Any]) -> None:
        conn[1].write(json.dumps(cmd).encode("utf-8") + b"\n")
        await conn[1].drain()

    async def _next(self, conn: Conn, timeout: float) -> dict[str, Any]:
        line = await asyncio.wait_for(conn[0].readline(), timeout=timeout)
        assert line, "local connection closed"
        msg: dict[str, Any] = json.loads(line.decode("utf-8"))
        return msg

    async def _until(
        self,
        conn: Conn,
        reply_type: str,
        forbidden: tuple[str, ...] = (),
        **fields: Any,
    ) -> dict[str, Any]:
        """Read *conn* up to its next *reply_type* frame matching *fields*.

        A frame of a *forbidden* type read on the way fails the test.
        """
        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            msg = await self._next(conn, timeout=max(0.01, deadline - time.monotonic()))
            self.assertNotIn(msg.get("type"), forbidden, f"misrouted frame: {msg}")
            if msg.get("type") == reply_type and all(msg.get(k) == v for k, v in fields.items()):
                return msg
        raise AssertionError(f"no {reply_type} reply within 45 s")

    async def _fence(self, conn: Conn, forbidden: tuple[str, ...] = ()) -> None:
        """Read *conn* up to the reply of a request sent now.

        Everything the daemon scheduled for *conn* before this call is
        read on the way (a connection's frames are sent in order), so
        no *forbidden* frame among them escapes :meth:`_until`.
        """
        await self._send(conn, {"type": "getInputHistory"})
        await self._until(conn, "inputHistory", forbidden)

    def _resume_cmd(self, chat_id: str, conn_id: str = "") -> dict[str, Any]:
        cmd: dict[str, Any] = {
            "type": "resumeSession",
            "machine": "box",
            "chatId": chat_id,
            "tabId": TAB,
            "scoped": True,
        }
        if conn_id:
            cmd["connId"] = conn_id
        return cmd

    def _bound_tab(self) -> RemoteTab:
        tab = self.server._server_api.remote_tabs._tabs[TAB]
        assert isinstance(tab, RemoteTab)
        return tab

    def _observe_task_end(self, observer: ClientConnection) -> None:
        """Run a command the remote has no SEA for on the tab; return once its status is out.

        The run fails before any model is called (``SeaError``), so
        the remote broadcasts ``status running:false`` for the tab: a
        frame of the tab that answers no replay.
        """
        observer.send(
            json.dumps(
                {
                    "type": "run",
                    "tabId": TAB,
                    "prompt": "never runs",
                    "seaCommand": "no-such-command",
                    "password": self.remote.password,
                }
            )
        )
        deadline = time.monotonic() + 45.0
        while True:
            msg = json.loads(observer.recv(timeout=max(0.01, deadline - time.monotonic())))
            if msg.get("type") == "status" and msg.get("tabId") == TAB and not msg["running"]:
                return

    async def test_a_task_end_between_sink_and_transcript_misroutes_nothing(self) -> None:
        """Client B's transcript reaches B alone when a task of the tab ends just before it."""
        await self._send(self.conn_a, self._resume_cmd("chat-seeded"))
        first = await self._until(self.conn_a, "task_events", tabId=TAB)
        self.assertEqual(first["task"], "Seeded task")
        tab = self._bound_tab()
        remote_tabs = self.server._server_api.remote_tabs
        self.observer = await asyncio.to_thread(mach.link_for(tab.machine).open_connection)
        with tab.lock:  # relaying paused: the status waits on the socket
            await asyncio.to_thread(self._observe_task_end, self.observer)
            # B's resume, registered and forwarded while the status is
            # still unrelayed: on the stream the status precedes B's
            # transcript, and the relay sees it with B's sink in place.
            remote_tabs.route(self._resume_cmd("chat-seeded", self.conn_b_id))
            (replay_id,) = tab.replay_sinks
        events = await self._until(self.conn_b, "task_events", tabId=TAB)
        self.assertEqual(events["task"], "Seeded task")
        # A got the task's end (and whatever else the tab broadcast),
        # not B's transcript.
        await self._fence(self.conn_a, forbidden=("task_events",))
        self.assertEqual(events["replayId"], replay_id)
        # The ``replayDone`` closing B's answer precedes the answer to
        # a later resume on the stream: once that one is in, the first
        # sink is gone.
        remote_tabs.route(self._resume_cmd("chat-seeded", self.conn_b_id))
        await self._until(self.conn_b, "task_events", tabId=TAB)
        self.assertNotIn(replay_id, tab.replay_sinks)

    async def test_a_resume_naming_no_connection_is_answered_to_every_client(self) -> None:
        """An in-process resume of a sub-agent task (no ``connId``) is not scoped."""
        await self._send(self.conn_a, self._resume_cmd("chat-seeded"))
        await self._until(self.conn_a, "task_events", tabId=TAB)
        sub_tab = f"{TAB}__sub_side-1"
        self.server._server_api.remote_tabs.route(
            {
                "type": "resumeSession",
                "taskId": "side-1",
                "tabId": sub_tab,
                "parentTabId": TAB,
            }
        )
        await self._until(self.conn_a, "subagentDone", tab_id=sub_tab)
        await self._until(self.conn_b, "subagentDone", tab_id=sub_tab)
        self.assertEqual(self._bound_tab().replay_sinks, {})
