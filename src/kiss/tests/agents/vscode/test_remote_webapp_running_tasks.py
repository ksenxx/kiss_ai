# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a webapp page load reconstructs all shared tabs.

Every running (or finished-but-open) task lives in a tab of the
daemon's shared tab registry — ``_cmd_run`` registers the tab
synchronously before its worker starts.  When a remote (WSS) client
connects and sends ``ready``, the server broadcasts the canonical
``tabs_state`` snapshot and replays every chat-bound registry tab, so
the page reconstructs the same tabs and transcripts every other
client shows.  Sub-agent tabs are derived state and are never listed
in the registry snapshot.

These tests drive the real ``RemoteAccessServer`` over a real WebSocket
connection, with real ``kiss.server.agent_state`` registry entries and
real ``task_history`` rows in a test-owned sqlite database — no mocks.
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

from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server import agent_state
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
    """``ready`` from a WSS client reconstructs the shared tab set."""

    async def asyncSetUp(self) -> None:
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

        with agent_state.STATE_LOCK:
            self._saved_registry = dict(agent_state.agent_states)
            agent_state.agent_states.clear()

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()
            agent_state.agent_states.update(self._saved_registry)
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

        extra: dict[str, object] = {"startTs": start_ts}
        if is_subagent:
            # A real sub-agent row carries its parent's task id (a
            # 32-hex task_history id — anything else is coerced away),
            # which is what the chat-id-only replay lookup filters on.
            extra["parent_task_id"] = "a" * 32
        task_id, chat_id = _persistence._add_task(
            prompt, chat_id=chat_id, extra=extra,
        )
        state = agent_state.AgentState(
            str(task_id),
            chat_id=chat_id,
            tab_id=tab_id,
            parent_task_id="parent-task" if is_subagent else None,
            server_owned=True,
            is_task_active=is_task_active,
        )
        state.last_user_prompt = prompt
        agent_state.register(state)
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

    def _register_tab(self, tab_id: str, chat_id: str, title: str) -> None:
        """Register a tab exactly like ``_cmd_run`` does."""
        self.server._vscode_server._registry_update_tab(
            tab_id, chat_id=chat_id, title=title, create=True,
        )

    async def test_ready_reports_running_tasks_latest_last(self) -> None:
        """The snapshot lists every registered tab, in registry order."""
        newest_task, newest_chat = self._register_running_task(
            "tab-new", "newest running task", 2_000,
        )
        oldest_task, oldest_chat = self._register_running_task(
            "tab-old", "oldest running task", 1_000,
        )
        self._register_tab("tab-new", newest_chat, "newest running task")
        self._register_tab("tab-old", oldest_chat, "oldest running task")

        events = await self._ready_replies({"type": "ready", "tabId": "t1"})
        snaps = [e for e in events if e.get("type") == "tabs_state"]
        self.assertTrue(snaps, "ready must deliver a tabs_state snapshot")
        tabs = snaps[-1]["tabs"]
        self.assertEqual(
            [t["tabId"] for t in tabs], ["tab-new", "tab-old"],
        )
        self.assertEqual(
            [t["chatId"] for t in tabs], [newest_chat, oldest_chat],
        )
        self.assertEqual(
            [t["title"] for t in tabs],
            ["newest running task", "oldest running task"],
        )
        replay_tabs = {
            e.get("tabId")
            for e in events
            if e.get("type") == "task_events"
        }
        self.assertLessEqual(
            {"tab-new", "tab-old"}, replay_tabs,
            "every chat-bound registry tab must be replayed so the "
            f"connecting client gets its transcript; got {replay_tabs}",
        )

    async def test_ready_without_running_tasks_sends_nothing(self) -> None:
        """No registered tabs -> an empty snapshot, no replays."""
        events = await self._ready_replies({"type": "ready", "tabId": "t1"})
        types = [e.get("type") for e in events]
        snaps = [e for e in events if e.get("type") == "tabs_state"]
        self.assertTrue(snaps)
        self.assertEqual(snaps[-1]["tabs"], [])
        self.assertNotIn("task_events", types)
        self.assertIn("models", types)
        self.assertIn("focusInput", types)

    async def test_ready_filters_subagents_inactive_and_dupes(self) -> None:
        """Sub-agent states never appear in the registry snapshot."""
        task_id, chat_id = self._register_running_task(
            "tab-main", "parent task", 1_500,
        )
        self._register_running_task(
            "tab-sub", "sub-agent task", 1_600,
            is_subagent=True, chat_id=chat_id,
        )
        self._register_tab("tab-main", chat_id, "parent task")

        events = await self._ready_replies({"type": "ready", "tabId": "t1"})
        snaps = [e for e in events if e.get("type") == "tabs_state"]
        self.assertTrue(snaps)
        tabs = snaps[-1]["tabs"]
        self.assertEqual([t["tabId"] for t in tabs], ["tab-main"])
        self.assertEqual(tabs[0]["chatId"], chat_id)
        self.assertEqual(tabs[0]["title"], "parent task")
