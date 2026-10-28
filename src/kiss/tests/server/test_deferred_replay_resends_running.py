# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a reconnect re-sends a still-running tab's ``running:true``.

A (re)connecting client's ``ready`` replays only the tab it is showing;
every other chat-bound tab is *deferred* (``AgentServer._defer_replay``):
no transcript, just the tab's running state and a ``replay_deferred``
offer.  The page settles a tab's running flag only from ``status``
events, so a background tab whose client missed the run's start or end
while disconnected keeps a stale flag unless the deferral re-sends the
state.  The previous fix settled the finished half (a finished task's
replay carries ``running:false``); these tests pin the running half:

* a tab whose own task is still running gets ``running:true`` again,
  ``attached`` (the page must not treat the re-send as a run starting),
  ahead of its ``replay_deferred``;
* two tasks of one chat running at once (the chat opened in two tabs, a
  run started in each): the tab attaches to its OWN run, not its
  sibling's, and is reported running — it used to get no status at all,
  the optimistic status of a source the tab does not own being withheld
  and the idle status skipped for a busy owner;
* a tab whose owner is busy only through a worktree merge (no run) is
  reported ``running:false``: a merge is not a run, the page shows no
  spinner for it, and a stale ``running:true`` from a missed task end
  used to survive the reconnect.

The tests drive the real ``RemoteAccessServer`` over a real WebSocket
connection, with real ``kiss.server.agent_state`` registry entries and
real ``task_history`` rows in a test-owned sqlite database — no mocks.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

from kiss.core.user_config import CONFIG_PATH, save_config
from kiss.server import agent_state
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.conftest import find_free_port, no_verify_ssl


def _tab_frames(events: list[dict[str, Any]], tab_id: str) -> list[dict[str, Any]]:
    """The ``status`` / ``replay_deferred`` / ``task_events`` frames of *tab_id*, in order."""
    return [
        e for e in events
        if e.get("tabId") == tab_id
        and e.get("type") in ("status", "replay_deferred", "task_events")
    ]


class TestDeferredReplayResendsRunning(IsolatedAsyncioTestCase):
    """A deferred reconnect replay re-sends the tab's running state."""

    async def asyncSetUp(self) -> None:
        import kiss.agents.sorcar.persistence as _persistence

        self._saved_persistence = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        self._persistence_dir = Path(
            tempfile.mkdtemp(prefix="kiss_deferred_running_test_"),
        )
        _persistence._KISS_DIR = self._persistence_dir
        _persistence._DB_PATH = self._persistence_dir / "history.db"
        _persistence._db_conn = None

        self.port = find_free_port()
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

        _persistence._close_db()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_persistence

    def _register_state(
        self,
        tab_id: str,
        prompt: str,
        *,
        chat_id: str = "",
        is_task_active: bool = True,
    ) -> tuple[str, str]:
        """Persist a task row and register a matching live state.

        Args:
            tab_id: The frontend tab the state belongs to.
            prompt: The task's prompt (its persisted row and title).
            chat_id: The chat to persist the row into (``""`` allocates
                a fresh chat).
            is_task_active: Whether the worker counts as live.

        Returns:
            ``(task_id, chat_id)`` of the persisted task_history row.
        """
        import kiss.agents.sorcar.persistence as _persistence

        task_id, chat_id = _persistence._add_task(prompt, chat_id=chat_id)
        state = agent_state.AgentState(
            str(task_id),
            chat_id=chat_id,
            tab_id=tab_id,
            server_owned=True,
            is_task_active=is_task_active,
        )
        state.last_user_prompt = prompt
        agent_state.register(state)
        self.server._agent_server._registry_update_tab(
            tab_id, chat_id=chat_id, title=prompt, create=True,
        )
        return str(task_id), chat_id

    async def _ready_replies(
        self, ready_cmd: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Auth, send *ready_cmd*, and drain the server's replies."""
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=no_verify_ssl(),
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

    async def test_own_running_task_resends_running_true(self) -> None:
        """A background tab whose task still runs gets ``running:true`` again."""
        _task, _chat = self._register_state("tab-bg", "still running")
        self._register_state("tab-fg", "shown tab", is_task_active=False)

        events = await self._ready_replies({"type": "ready", "tabId": "tab-fg"})

        frames = _tab_frames(events, "tab-bg")
        self.assertEqual(
            [f["type"] for f in frames], ["status", "replay_deferred"],
            f"the running state must precede the deferral: {frames}",
        )
        status = frames[0]
        self.assertTrue(status["running"])
        # A catch-up, not a run starting: the page keeps its window.
        self.assertTrue(status.get("attached"))
        # The shown tab is idle: it gets its transcript, flag settled.
        fg = _tab_frames(events, "tab-fg")
        self.assertEqual([f["type"] for f in fg], ["status", "task_events"])
        self.assertFalse(fg[0]["running"])

    async def test_two_runs_in_one_chat_attach_the_tab_to_its_own_run(self) -> None:
        """A tab running beside a sibling run of the same chat is reported running."""
        sibling_task, chat = self._register_state("tab-sibling", "sibling run")
        own_task, _ = self._register_state("tab-own", "own run", chat_id=chat)
        self._register_state("tab-fg", "shown tab", is_task_active=False)

        events = await self._ready_replies({"type": "ready", "tabId": "tab-fg"})

        frames = _tab_frames(events, "tab-own")
        self.assertEqual(
            [f["type"] for f in frames], ["status", "replay_deferred"],
            f"tab-own must be reported running: {frames}",
        )
        self.assertTrue(frames[0]["running"])
        self.assertTrue(frames[0].get("attached"))
        # The registry keeps one tab per chat: binding tab-own to the
        # chat displaced tab-sibling, which is therefore not replayed.
        self.assertEqual(_tab_frames(events, "tab-sibling"), [])
        printer = self.server._agent_server.printer
        # The tab follows its own run's stream, not its sibling's (the
        # sibling's state was registered first and used to win the
        # chat scan, which left the tab with no status at all).
        self.assertEqual(printer.tasks_for_tab("tab-own"), [own_task])
        self.assertNotEqual(own_task, sibling_task)

    async def test_merging_owner_is_reported_idle(self) -> None:
        """A worktree merge is not a run: the tab gets ``running:false``."""
        import kiss.agents.sorcar.persistence as _persistence

        task_id, _chat = self._register_state(
            "tab-merge", "merged task", is_task_active=False,
        )
        _persistence._save_task_result("done", task_id=task_id)
        state = agent_state.get(task_id)
        assert state is not None
        with agent_state.STATE_LOCK:
            # The merge claim ``_cmd_run``'s merge flow installs: the
            # run is over, the tab is busy, nothing is running.
            state.is_merging = True
            self.assertTrue(state.busy())
            self.assertFalse(state.running())
        self._register_state("tab-fg", "shown tab", is_task_active=False)

        events = await self._ready_replies({"type": "ready", "tabId": "tab-fg"})

        frames = _tab_frames(events, "tab-merge")
        self.assertEqual(
            [f["type"] for f in frames], ["status", "replay_deferred"],
            f"a merging tab must be settled as idle: {frames}",
        )
        self.assertFalse(frames[0]["running"])

    async def test_pinned_source_yields_to_the_tabs_own_run(self) -> None:
        """A tab pinned to a sibling's task but owning a run of its own is reported running."""
        sibling_task, chat = self._register_state("tab-sibling", "sibling run")
        _own_task, _ = self._register_state("tab-own", "own run", chat_id=chat)
        # The registry pins tab-own to the sibling's task (a history row
        # opened in it): the exact-id pass resolves the sibling, which
        # the tab's own busy run outranks.
        self.server._agent_server._registry_update_tab(
            "tab-own", task_id=sibling_task,
        )
        self._register_state("tab-fg", "shown tab", is_task_active=False)

        events = await self._ready_replies({"type": "ready", "tabId": "tab-fg"})

        frames = _tab_frames(events, "tab-own")
        self.assertEqual(
            [f["type"] for f in frames], ["status", "replay_deferred"],
            f"the tab's own run must be re-sent: {frames}",
        )
        self.assertTrue(frames[0]["running"])
        self.assertTrue(frames[0].get("attached"))

    async def test_shown_tab_owning_a_run_of_another_chat_stays_running(self) -> None:
        """The shown tab's finished-chat replay carries its own run's ``running:true``."""
        import kiss.agents.sorcar.persistence as _persistence

        # tab-x owns a live run of chat A but the registry binds it to
        # chat B, a finished chat: the replay of B attaches to nothing
        # live, and must still report the tab running.
        self._register_state("tab-x", "run of chat A")
        done_task, chat_b = _persistence._add_task("finished chat B")
        _persistence._save_task_result("done", task_id=str(done_task))
        self.server._agent_server._registry_update_tab(
            "tab-x", chat_id=chat_b, title="finished chat B",
        )

        events = await self._ready_replies({"type": "ready", "tabId": "tab-x"})

        frames = _tab_frames(events, "tab-x")
        self.assertEqual(
            [f["type"] for f in frames], ["status", "task_events"],
            f"the status must precede the transcript: {frames}",
        )
        self.assertTrue(frames[0]["running"])
        self.assertTrue(frames[0].get("attached"))
        self.assertEqual(frames[1]["task"], "finished chat B")
