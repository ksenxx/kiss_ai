# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: when a task is deleted from the task history panel, the
``taskDeleted`` broadcast MUST reach EVERY connected client so any open
tab showing the deleted task's chat can prune its chat webview.

Regression these tests reproduce: ``VSCodeServer._handle_delete_task``
broadcasts ``{"type": "taskDeleted", "taskId": <deleted id>, ...}``.
The production printer (``WebPrinter``) classifies any event that
carries a truthy top-level ``taskId`` as a *task event*: it records it,
persists it, and fans it out ONLY to tabs subscribed to that task id
(``_fanout_stamped``).  A deleted task is (almost always) a completed
task with ZERO subscribers, so the ``taskDeleted`` event was silently
swallowed and never reached any client — neither VS Code windows (UDS)
nor remote webapp browsers (WSS).  Open tabs showing the deleted task's
chat therefore never removed the task/chat from their webview.

These tests run the real ``RemoteAccessServer`` over WSS with real
``websockets`` clients (no mocks) and assert the broadcast semantics.
"""

from __future__ import annotations

import asyncio
import json
import ssl
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

import kiss.agents.sorcar.persistence as th
from kiss.server.vscode_config import CONFIG_PATH, save_config
from kiss.server.web_server import RemoteAccessServer


def _find_free_port() -> int:
    """Bind an ephemeral port, release it, and return its number."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _no_verify_ssl() -> ssl.SSLContext:
    """SSL context that skips certificate verification (self-signed)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def _drain_events(
    ws: Any, per_recv_timeout: float = 2.0, max_events: int = 30,
) -> list[dict[str, Any]]:
    """Receive events from *ws* until a timeout or *max_events*."""
    events: list[dict[str, Any]] = []
    for _ in range(max_events):
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=per_recv_timeout)
            events.append(json.loads(raw))
        except TimeoutError:
            break
    return events


class TestDeleteTaskBroadcastReachesAllClients(IsolatedAsyncioTestCase):
    """Deleting a task from the History panel must broadcast
    ``taskDeleted`` to EVERY connected client, so any open tab on any
    client showing the task's chat removes the task and its chat."""

    async def asyncSetUp(self) -> None:
        # Pin persistence to a fresh test-owned directory so these
        # tests remain independent regardless of run order.
        self._saved_persistence = (
            th._DB_PATH,
            th._db_conn,
            th._KISS_DIR,
        )
        self._persistence_dir = Path(
            tempfile.mkdtemp(prefix="kiss_del_task_test_"),
        )
        th._KISS_DIR = self._persistence_dir
        th._DB_PATH = self._persistence_dir / "sorcar.db"
        th._db_conn = None

        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

        if th._db_conn is not None:
            try:
                th._db_conn.close()
            except Exception:
                pass
            th._db_conn = None
        (
            th._DB_PATH,
            th._db_conn,
            th._KISS_DIR,
        ) = self._saved_persistence

    async def _auth(self, ws: Any) -> None:
        await ws.send(json.dumps({"type": "auth", "password": ""}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        self.assertEqual(resp["type"], "auth_ok")

    async def test_task_deleted_reaches_every_connected_client(self) -> None:
        """THE regression: a completed (unsubscribed) task is deleted
        by client 1; client 2 — a different device whose open tab shows
        the same chat — MUST also receive the ``taskDeleted`` event."""
        # Two completed tasks in the same chat, inserted directly into
        # the real DB (the same rows the History panel lists).
        id1, chat_id = await asyncio.to_thread(th._add_task, "first task")
        await asyncio.sleep(0.01)
        id2, _ = await asyncio.to_thread(
            th._add_task, "second task", chat_id,
        )

        async with (
            connect(
                f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
            ) as ws_deleter,
            connect(
                f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
            ) as ws_viewer,
        ):
            await self._auth(ws_deleter)
            await self._auth(ws_viewer)

            # Client 1 deletes task id1 from its History panel.
            await ws_deleter.send(
                json.dumps({"type": "deleteTask", "taskId": id1}),
            )

            viewer_events = await _drain_events(ws_viewer)
            deleted = [
                e for e in viewer_events if e.get("type") == "taskDeleted"
            ]
            self.assertEqual(
                len(deleted),
                1,
                "the taskDeleted broadcast must reach OTHER connected "
                "clients too (their open tabs show the deleted task's "
                f"chat); viewer got event types: "
                f"{[e.get('type') for e in viewer_events]}",
            )
            ev = deleted[0]
            self.assertEqual(ev["taskId"], id1)
            self.assertEqual(ev["chatId"], chat_id)
            self.assertTrue(ev["chatHasMoreTasks"])

            deleter_events = await _drain_events(ws_deleter)
            self.assertIn(
                "taskDeleted",
                [e.get("type") for e in deleter_events],
                "the deleting client's own open tabs must also receive "
                "taskDeleted",
            )

    async def test_deleting_last_task_signals_chat_empty(self) -> None:
        """Deleting a chat's only task must broadcast
        ``chatHasMoreTasks: false`` so open tabs close the whole chat."""
        task_id, chat_id = await asyncio.to_thread(th._add_task, "only task")

        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
        ) as ws:
            await self._auth(ws)
            await ws.send(
                json.dumps({"type": "deleteTask", "taskId": task_id}),
            )
            events = await _drain_events(ws)
            deleted = [e for e in events if e.get("type") == "taskDeleted"]
            self.assertEqual(
                len(deleted),
                1,
                "taskDeleted must be broadcast to connected clients; got "
                f"event types: {[e.get('type') for e in events]}",
            )
            ev = deleted[0]
            self.assertEqual(ev["taskId"], task_id)
            self.assertEqual(ev["chatId"], chat_id)
            self.assertFalse(ev["chatHasMoreTasks"])

    async def test_task_deleted_not_persisted_as_chat_event(self) -> None:
        """The ``taskDeleted`` broadcast must never be recorded or
        persisted under the just-deleted task id (that would re-insert
        garbage event rows for a deleted task_history row)."""
        id1, chat_id = await asyncio.to_thread(th._add_task, "first task")
        await asyncio.sleep(0.01)
        id2, _ = await asyncio.to_thread(
            th._add_task, "second task", chat_id,
        )

        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
        ) as ws:
            await self._auth(ws)
            await ws.send(json.dumps({"type": "deleteTask", "taskId": id1}))
            events = await _drain_events(ws)
            self.assertIn(
                "taskDeleted", [e.get("type") for e in events],
            )

        # Give the async persistence queue a moment to flush, then
        # verify no events were written for the deleted task id.  The
        # task_history row is gone, so _load_chat_events_by_task_id
        # must return None (no orphan event rows resurrect the task).
        await asyncio.to_thread(th._flush_chat_events)
        loaded = await asyncio.to_thread(
            th._load_chat_events_by_task_id, id1,
        )
        self.assertIsNone(
            loaded,
            "taskDeleted must not be persisted as a chat event for the "
            f"deleted task id; loaded: {loaded}",
        )
