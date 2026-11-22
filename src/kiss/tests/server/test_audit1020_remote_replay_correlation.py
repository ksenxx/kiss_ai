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
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.sync.client import ClientConnection

from kiss.agents.sorcar import machines as mach
from kiss.server.remote_tabs import RemoteTab, RemoteTabs
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import (
    LocalReader,
    LocalWriter,
    _generate_self_signed_cert,
    open_local_connection,
)
from kiss.tests.server.pause import Pause

TAB = "tab-shared"
Conn = tuple[LocalReader, LocalWriter]


class TestRemoteReplayCorrelation(IsolatedAsyncioTestCase):
    """One local daemon with two clients, one remote daemon seeded with a chat."""

    async def asyncSetUp(self) -> None:
        # Cleanups, each registered as its resource starts, not asyncTearDown:
        # unittest skips the tear-down when the set-up fails.
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        mach.save_machines([])
        self.addCleanup(mach.save_machines, [])
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
                    "chat_id": "chat-other",
                    "task": "Other task",
                    "result": "done",
                    "task_id": "seeded-2",
                },
            ],
        )
        self.addCleanup(self.remote.stop)
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
        # Registered before the start: one that fails part-way is stopped too.
        self.addAsyncCleanup(self.server.stop_async)
        await self.server.start_async()
        self.conn_a = await open_local_connection(self.server)
        self.addCleanup(self.conn_a[1].close)
        await self._fence(self.conn_a)
        known = set(self.server._printer._conn_endpoints)
        self.conn_b = await open_local_connection(self.server)
        self.addCleanup(self.conn_b[1].close)
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

    async def _remote(self, field: str) -> str:
        """What the machine's own registry holds as *field* (``chatId``, ``taskId``) of the tab."""
        reply = await asyncio.to_thread(
            mach.link_for(self._bound_tab().machine).request, {"type": "ready"}, "tabs_state",
        )
        return next(str(t[field]) for t in reply["tabs"] if t["tabId"] == TAB)

    _CLICK = {"type": "resumeSession", "machine": "box", "chatId": "chat-other", "tabId": TAB}
    """A CHATS-panel click: the machine's other chat is picked in the tab."""

    def _fetch_cmd(self) -> dict[str, Any]:
        """B's own fetch of the chat the tab shows (its answer to a ``replay_deferred`` offer)."""
        return {
            "type": "resumeSession", "chatId": "chat-seeded", "tabId": TAB, "scoped": True,
            "connId": self.conn_b_id,
        }

    async def _ends_on_the_chat_clicked(self, shown_to_b: list[str]) -> None:
        """Both daemons show the clicked chat in the tab; B got *shown_to_b*, in that order."""
        # Every forwarded resume is answered: the machine acted on them all.
        await self._answered()
        self.assertEqual(self._bound_tab().chat_id, "chat-other")
        self.assertEqual(await self._remote("chatId"), "chat-other")
        await self._send(self.conn_b, {"type": "getInputHistory"})
        shown: list[str] = []
        while True:
            msg = await self._next(self.conn_b, timeout=45.0)
            if msg.get("type") == "inputHistory":
                break
            if msg.get("type") == "task_events" and msg.get("tabId") == TAB:
                shown.append(msg["task"])
        self.assertEqual(shown, shown_to_b)

    async def _held_before_its_send(self, stale_resume: Callable[[], object]) -> None:
        """The machine's tab ends on the chat picked last, whatever resume was under way.

        *stale_resume*, B's resume of the chat the tab shows, is held
        right before its send while A's history click picks another
        chat of the machine in the tab.  The machine takes a forwarded
        resume as navigation: sent after the click, the held one
        switched the machine's tab back to the chat this daemon no
        longer showed, and B got that chat's transcript last.
        """
        await self._send(self.conn_a, self._resume_cmd("chat-seeded"))
        await self._until(self.conn_a, "task_events", tabId=TAB)
        tab = self._bound_tab()
        other: dict[str, Any] | None = None
        stale = Pause(
            RemoteTab.send.__code__,
            lambda local: local["self"] is tab
            and local["cmd"].get("chatId") == "chat-seeded"
            and bool(local["cmd"].get("replayId")),
        )
        with stale:
            resume = asyncio.create_task(asyncio.to_thread(stale_resume))
            self.assertTrue(await asyncio.to_thread(stale.paused.wait, 30))
            await self._send(self.conn_a, self._CLICK)
            # The click is answered in the window if anything lets it reach the machine.
            deadline = time.monotonic() + 2.0
            while other is None and time.monotonic() < deadline:
                try:
                    msg = await self._next(self.conn_a, timeout=deadline - time.monotonic())
                except TimeoutError:
                    break
                if msg.get("type") == "task_events" and msg.get("tabId") == TAB:
                    other = msg
            stale.release()
            await resume
        other = other or await self._until(self.conn_a, "task_events", tabId=TAB)
        self.assertEqual(other["task"], "Other task")
        # B is shown the tab's chats in the order they were picked.
        await self._ends_on_the_chat_clicked(["Seeded task", "Other task"])

    async def test_a_replay_held_before_its_send_does_not_undo_a_newer_history_click(self) -> None:
        """B's ``ready`` replay of the tab it shows, read before A's click."""
        remote_tabs = self.server._server_api.remote_tabs
        await self._held_before_its_send(lambda: remote_tabs.replay_to(self.conn_b_id, TAB))

    async def test_a_fetch_held_before_its_send_does_not_undo_a_newer_history_click(self) -> None:
        """B's own fetch of the chat the tab shows, checked before A's click."""
        remote_tabs = self.server._server_api.remote_tabs
        await self._held_before_its_send(lambda: remote_tabs.route(self._fetch_cmd()))

    async def test_a_fetch_checked_before_a_history_click_gets_the_chat_clicked(self) -> None:
        """A click landing between the check of B's fetch and its send is what B gets.

        The click is routed on the fetch's own thread, at the entry of
        the step that sends it: the fetch was found to name the chat
        the tab shows, and no longer does.  Sent all the same, it
        switched the machine's tab back to the chat the click left.
        """
        await self._send(self.conn_a, self._resume_cmd("chat-seeded"))
        await self._until(self.conn_a, "task_events", tabId=TAB)
        await self._fence(self.conn_b)
        remote_tabs = self.server._server_api.remote_tabs
        clicked: list[bool] = []

        def click(local: dict[str, Any]) -> bool:
            if not clicked:
                clicked.append(remote_tabs.route(dict(self._CLICK)))
            return False

        with Pause(RemoteTabs._fetch.__code__, click):
            self.assertTrue(await asyncio.to_thread(remote_tabs.route, self._fetch_cmd()))
        self.assertEqual(clicked, [True])
        other = await self._until(self.conn_a, "task_events", tabId=TAB)
        self.assertEqual(other["task"], "Other task")
        # The click's answer, sent to every client, and the fetch's own.
        await self._ends_on_the_chat_clicked(["Other task", "Other task"])

    async def _fetched_after_the_click(self, stale: dict[str, Any], shown_to_b: list[str]) -> None:
        """B's fetch *stale* of the chat the tab showed, routed after A's click, reads the tab.

        Taken for a resume of a local chat, it unbound the tab from the
        machine under A's click.
        """
        await self._send(self.conn_a, self._resume_cmd("chat-seeded"))
        await self._until(self.conn_a, "task_events", tabId=TAB)
        tab = self._bound_tab()
        await self._send(self.conn_a, self._CLICK)
        other = await self._until(self.conn_a, "task_events", tabId=TAB)
        self.assertEqual(other["task"], "Other task")
        remote_tabs = self.server._server_api.remote_tabs
        self.assertTrue(await asyncio.to_thread(remote_tabs.route, stale))
        self.assertIs(self._bound_tab(), tab)
        await self._ends_on_the_chat_clicked(shown_to_b)

    async def test_a_fetch_of_the_chat_a_history_click_left_gets_the_chat_clicked(self) -> None:
        """A page's: answered with the chat the tab shows now, after the click's own answer."""
        await self._fetched_after_the_click(self._fetch_cmd(), ["Other task", "Other task"])

    async def test_a_reconnect_to_a_run_of_the_chat_a_history_click_left_changes_nothing(
        self,
    ) -> None:
        """An API caller's (``recovery``): its run's chat left the tab, nothing is sent."""
        reconnect = {**self._fetch_cmd(), "taskId": "seeded-1", "recovery": True}
        await self._fetched_after_the_click(reconnect, ["Other task"])

    async def _answered(self) -> None:
        """Wait until the machine answered every forwarded resume (``replayDone``)."""
        tab = self._bound_tab()
        for _ in range(900):
            if not tab.replay_sinks:
                break
            await asyncio.sleep(0.05)
        self.assertEqual(tab.replay_sinks, {})

    async def test_a_fetch_at_the_task_a_history_click_left_does_not_pin_it_again(self) -> None:
        """B's fetch names the task its offer was made at; A's click shows the latest since."""
        await self._send(self.conn_a, {**self._resume_cmd("chat-seeded"), "taskId": "seeded-1"})
        await self._until(self.conn_a, "task_events", tabId=TAB)
        remote_tabs = self.server._server_api.remote_tabs
        pinned = {**self._fetch_cmd(), "taskId": "seeded-1"}
        # While the tab is pinned to the task, the fetch is one of the tab at that task.
        self.assertTrue(await asyncio.to_thread(remote_tabs.route, dict(pinned)))
        await self._answered()
        self.assertEqual(await self._remote("taskId"), "seeded-1")
        await self._send(self.conn_a, self._resume_cmd("chat-seeded"))
        await self._until(self.conn_a, "task_events", tabId=TAB)
        self.assertEqual(await self._remote("taskId"), "")
        self.assertTrue(await asyncio.to_thread(remote_tabs.route, dict(pinned)))
        await self._answered()
        tab = self._bound_tab()
        self.assertEqual((tab.chat_id, tab.shown_task_id), ("chat-seeded", ""))
        self.assertEqual(await self._remote("taskId"), "")

    async def test_a_surface_connecting_before_a_click_is_answered_is_offered_its_chat(
        self,
    ) -> None:
        """The chat a click picked is the tab's before the machine told the tab's state."""
        answer = Pause(
            RemoteTab._relay_frame.__code__, lambda local: local["msg"].get("tabId") == TAB,
        )
        with answer:
            await self._send(self.conn_a, self._resume_cmd("chat-seeded"))
            self.assertTrue(await asyncio.to_thread(answer.paused.wait, 30))
            self.assertEqual(self._bound_tab().last_status, {})
            remote_tabs = self.server._server_api.remote_tabs
            await asyncio.to_thread(remote_tabs.replay_to, self.conn_b_id, "another-tab")
            offer = await self._until(self.conn_b, "replay_deferred", tabId=TAB)
            self.assertEqual(offer["chatId"], "chat-seeded")
            answer.release()
        await self._until(self.conn_a, "task_events", tabId=TAB)

    async def test_a_resume_naming_no_connection_is_answered_to_every_client(self) -> None:
        """An in-process resume of the machine's chat (no ``connId``) is not scoped."""
        await self._send(self.conn_a, self._resume_cmd("chat-seeded"))
        await self._until(self.conn_a, "task_events", tabId=TAB)
        self.server._server_api.remote_tabs.route(self._resume_cmd("chat-seeded"))
        await self._until(self.conn_a, "task_events", tabId=TAB)
        await self._until(self.conn_b, "task_events", tabId=TAB)
        self.assertEqual(self._bound_tab().replay_sinks, {})
