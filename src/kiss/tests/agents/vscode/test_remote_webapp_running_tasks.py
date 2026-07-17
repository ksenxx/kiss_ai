# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a remote webapp page load opens all running tasks.

When a remote (WSS) client connects and sends ``ready``, the web
server's ``_handle_ready`` must report every task currently running in
the backend with a targeted ``openRunningTasks`` message so the page
can open one chat tab per running task and focus the tab running the
LATEST task.  The message lists one entry per running top-level chat —
``{chatId, taskId, title, startTs}`` — sorted by ``startTs`` ascending
(oldest first), so the client's natural open-in-order loop ends focused
on the newest task.

These tests drive the real ``RemoteAccessServer`` over a real WebSocket
connection, with real ``_RunningAgentState`` registry entries and real
``task_history`` rows in a test-owned sqlite database — no mocks.
"""

from __future__ import annotations

import asyncio
import json
import socket
import ssl
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server.web_server import RemoteAccessServer


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _no_verify_ssl() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class TestReadyOpensRunningTasks(IsolatedAsyncioTestCase):
    """``ready`` from a WSS client reports running tasks to open."""

    async def asyncSetUp(self) -> None:
        # Pin persistence to a fresh test-owned directory so the test
        # neither reads nor pollutes the developer's real
        # ~/.kiss/sorcar.db (same pattern as test_web_server.py).
        import kiss.agents.sorcar.persistence as _persistence

        self._saved_persistence = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        self._persistence_dir = Path(
            tempfile.mkdtemp(prefix="kiss_running_tasks_test_"),
        )
        _persistence._KISS_DIR = self._persistence_dir
        _persistence._DB_PATH = self._persistence_dir / "sorcar.db"
        _persistence._db_conn = None

        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        # The registry is process-global: remember and restore its
        # contents so this test stays independent of suite order.
        with _RunningAgentState._registry_lock:
            self._saved_registry = dict(
                _RunningAgentState.running_agent_states,
            )
            _RunningAgentState.running_agent_states.clear()

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        with _RunningAgentState._registry_lock:
            _RunningAgentState.running_agent_states.clear()
            _RunningAgentState.running_agent_states.update(
                self._saved_registry,
            )
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

        import kiss.agents.sorcar.persistence as _persistence

        if _persistence._db_conn is not None:
            try:
                _persistence._db_conn.close()
            except Exception:
                pass
            _persistence._db_conn = None
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_persistence

    def _register_running_task(
        self,
        tab_id: str,
        prompt: str,
        start_ts: int,
        *,
        is_subagent: bool = False,
        is_task_active: bool = True,
        chat_id: str = "",
    ) -> tuple[str, str]:
        """Persist a task row and register a matching live state.

        Returns:
            ``(task_id, chat_id)`` of the persisted task_history row.
        """
        import kiss.agents.sorcar.persistence as _persistence

        task_id, chat_id = _persistence._add_task(
            prompt, chat_id=chat_id, extra={"startTs": start_ts},
        )
        state = _RunningAgentState(
            tab_id,
            "test-model",
            chat_id=chat_id,
            is_subagent=is_subagent,
            is_task_active=is_task_active,
        )
        state.task_history_id = task_id
        state.last_user_prompt = prompt
        _RunningAgentState.register(tab_id, state)
        return task_id, chat_id

    async def _ready_replies(
        self, ready_cmd: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Auth, send *ready_cmd*, and drain the server's replies."""
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            self.assertEqual(resp["type"], "auth_ok")
            await ws.send(json.dumps(ready_cmd))
            events: list[dict[str, Any]] = []
            for _ in range(30):
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2)
                    events.append(json.loads(raw))
                except TimeoutError:
                    break
            return events

    async def test_ready_reports_running_tasks_latest_last(self) -> None:
        """All running tasks are reported, sorted oldest -> latest."""
        newest_task, newest_chat = self._register_running_task(
            "tab-new", "newest running task", 2_000,
        )
        oldest_task, oldest_chat = self._register_running_task(
            "tab-old", "oldest running task", 1_000,
        )

        events = await self._ready_replies({"type": "ready", "tabId": "t1"})
        opens = [e for e in events if e.get("type") == "openRunningTasks"]
        self.assertEqual(len(opens), 1)
        tasks = opens[0]["tasks"]
        self.assertEqual(
            [t["chatId"] for t in tasks], [oldest_chat, newest_chat],
        )
        self.assertEqual(
            [t["taskId"] for t in tasks], [oldest_task, newest_task],
        )
        self.assertEqual(
            [t["title"] for t in tasks],
            ["oldest running task", "newest running task"],
        )
        self.assertEqual([t["startTs"] for t in tasks], [1_000, 2_000])

    async def test_ready_without_running_tasks_sends_nothing(self) -> None:
        """No running tasks -> no ``openRunningTasks`` message at all."""
        events = await self._ready_replies({"type": "ready", "tabId": "t1"})
        types = [e.get("type") for e in events]
        self.assertNotIn("openRunningTasks", types)
        # Sanity: the normal ready fan-out still happened.
        self.assertIn("models", types)
        self.assertIn("focusInput", types)

    async def test_ready_filters_subagents_inactive_and_dupes(self) -> None:
        """Sub-agents, idle states and duplicate chats are excluded."""
        task_id, chat_id = self._register_running_task(
            "tab-main", "parent task", 1_500,
        )
        # A parallel sub-agent sharing the parent's chat id must not
        # produce a second row (the parent replay reopens its tab).
        self._register_running_task(
            "tab-sub", "sub-agent task", 1_600,
            is_subagent=True, chat_id=chat_id,
        )
        # An idle (finished) state must not be reported.
        self._register_running_task(
            "tab-idle", "finished task", 1_700, is_task_active=False,
        )
        # A second live viewer of the SAME chat collapses to one row.
        dup = _RunningAgentState(
            "tab-dup", "test-model", chat_id=chat_id, is_task_active=True,
        )
        dup.task_history_id = task_id
        dup.last_user_prompt = "parent task"
        _RunningAgentState.register("tab-dup", dup)

        events = await self._ready_replies({"type": "ready", "tabId": "t1"})
        opens = [e for e in events if e.get("type") == "openRunningTasks"]
        self.assertEqual(len(opens), 1)
        tasks = opens[0]["tasks"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["chatId"], chat_id)
        self.assertEqual(tasks[0]["taskId"], task_id)
        self.assertEqual(tasks[0]["startTs"], 1_500)
