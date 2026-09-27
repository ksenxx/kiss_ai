# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a conn-scoped replay must not blind other windows.

Client A shows tab T whose task is running (T is subscribed to the
task's fan-out).  Client B connects and its ``ready`` triggers a replay
of T scoped to B alone (``conn_id``).  ``_replay_session`` used to drop
ALL of T's subscriptions (``cleanup_tab``) and re-subscribe it a moment
later (``_attach_viewer_to_running_chat``).  A task event emitted in
that gap was recorded and persisted but fanned out to nobody: B still
saw it (its replay snapshot is taken after the re-subscribe), but A —
which does not receive the scoped replay — lost it until a reload.

The interleaving is forced with real locks only: the replay thread is
parked in its DB read by holding the persistence write lock, then the
test takes the server's ``_state_lock`` and releases the DB lock, so
the replay runs ``cleanup_tab`` and blocks inside
``_attach_viewer_to_running_chat`` (which needs ``_state_lock``).  The
running task emits its event exactly there.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import uuid
from types import FrameType
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.tests.server.test_web_server_tab_mirroring import (
    TabMirroringBase,
)

_GAP_DIR = "/tmp/emitted-between-unsubscribe-and-resubscribe"


def _thread_in(thread: threading.Thread, func_name: str) -> bool:
    """True when *thread*'s current stack contains a *func_name* frame."""
    frame = sys._current_frames().get(thread.ident or -1)
    while frame is not None:
        if frame.f_code.co_name == func_name:
            return True
        frame = frame.f_back
    return False


def _wait_until_in(thread: threading.Thread, func_name: str) -> None:
    """Block until *thread* is executing inside *func_name* (5 s cap)."""
    deadline = time.monotonic() + 5.0
    while not _thread_in(thread, func_name):
        assert time.monotonic() < deadline, f"replay never reached {func_name}"
        time.sleep(0.005)
    time.sleep(0.05)


class TestScopedReplayResubscribeGap(TabMirroringBase):
    """A scoped replay keeps the shared tab subscribed for other windows."""

    async def test_event_in_replay_gap_reaches_other_window(self) -> None:
        """A receives the live event emitted during B's scoped replay."""
        assert self.server is not None
        vs = self.server._vscode_server
        printer = self.server._printer
        task_id, chat_id = th._add_task("Running task")
        th._append_chat_event(
            {"type": "prompt", "text": "running prompt"}, task_id=task_id,
        )
        state = AgentState(
            str(task_id), chat_id=chat_id, tab_id="tab-t", is_task_active=True,
        )
        agent_state.register(state)

        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._send(ws_a, {
            "type": "openTab", "tabId": "tab-t", "title": "new chat",
        })
        self.assertIsNotNone(
            await self._wait_for_snapshot_with(ws_a, present={"tab-t"}),
        )
        await self._send(ws_a, {
            "type": "resumeSession", "chatId": chat_id, "tabId": "tab-t",
        })
        self.assertIsNotNone(await self._wait_for_event(
            ws_a, "task_events", pred=lambda ev: ev.get("tabId") == "tab-t",
        ))
        self.assertIn("tab-t", printer._fanout_targets(task_id))

        before = set(printer._conn_endpoints)
        ws_b = await self._connect_ok()
        conn_b = (set(printer._conn_endpoints) - before).pop()

        def emit() -> None:
            printer._thread_local.task_id = str(task_id)
            printer.broadcast(
                {"type": "worktree_created", "worktreeDir": _GAP_DIR},
            )

        def replay() -> None:
            vs._replay_session(chat_id, "tab-t", conn_id=conn_b)

        def force_interleaving() -> None:
            with th._rw_lock.write_lock():
                replayer = threading.Thread(target=replay, daemon=True)
                replayer.start()
                _wait_until_in(replayer, "_read_lock_gen")
                vs._state_lock.acquire()
            try:
                _wait_until_in(replayer, "_attach_viewer_to_running_chat")
                emitter = threading.Thread(target=emit, daemon=True)
                emitter.start()
                emitter.join(5)
                assert not emitter.is_alive(), "emitter blocked on _state_lock"
            finally:
                vs._state_lock.release()
            replayer.join(10)
            assert not replayer.is_alive()

        await asyncio.to_thread(force_interleaving)

        def is_gap_event(ev: dict[str, Any]) -> bool:
            return (
                ev.get("tabId") == "tab-t" and ev.get("worktreeDir") == _GAP_DIR
            )

        got_a = await self._wait_for_event(
            ws_a, "worktree_created", timeout=3.0, pred=is_gap_event,
        )
        self.assertIsNotNone(
            got_a,
            "window A never received the task event emitted while B's "
            "scoped replay re-subscribed the shared tab",
        )
        replay_b = await self._wait_for_event(
            ws_b, "task_events", pred=lambda ev: ev.get("tabId") == "tab-t",
        )
        assert replay_b is not None
        self.assertIn("tab-t", printer._fanout_targets(task_id))
        state.is_task_active = False

    async def test_allocation_during_replay_keeps_persisted_subscription(
        self,
    ) -> None:
        """An id allocation racing B's scoped replay must not blind A.

        The running task is still registered under its provisional id
        when B's replay resolves it as the live source.  If the task
        allocates its persisted id (re-key, then the launching tab's
        subscription to that id) after the source is resolved but
        before the replay's replacement cleanup, the cleanup — keeping
        only the obsolete provisional id — used to drop the persisted
        subscription, so A missed events until the replay's commit.

        A trace hook on the replay thread starts the real allocation
        (``agent_task_allocated`` + ``_on_run_task_id_allocated``) from
        another thread the moment the replay calls ``cleanup_tab`` and
        lets it run as far as the locks allow; an event is emitted
        under the persisted id when the replay starts publishing the
        reopen, i.e. before the commit re-subscribes the tab.
        """
        assert self.server is not None
        vs = self.server._vscode_server
        printer = self.server._printer
        _, chat_id = th._add_task("Earlier task")
        persisted_id, _ = th._add_task("Running task", chat_id=chat_id)
        th._append_chat_event(
            {"type": "prompt", "text": "running prompt"}, task_id=persisted_id,
        )
        provisional_id = uuid.uuid4().hex
        agent = WorktreeSorcarAgent("agent")
        state = AgentState(
            provisional_id,
            agent=agent,
            chat_id=chat_id,
            tab_id="tab-t",
            is_task_active=True,
        )
        agent_state.register(state)

        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._send(ws_a, {
            "type": "openTab", "tabId": "tab-t", "title": "new chat",
        })
        self.assertIsNotNone(
            await self._wait_for_snapshot_with(ws_a, present={"tab-t"}),
        )
        await self._send(ws_a, {
            "type": "resumeSession", "chatId": chat_id, "tabId": "tab-t",
        })
        self.assertIsNotNone(await self._wait_for_event(
            ws_a, "task_events", pred=lambda ev: ev.get("tabId") == "tab-t",
        ))
        self.assertIn("tab-t", printer._fanout_targets(provisional_id))

        before = set(printer._conn_endpoints)
        ws_b = await self._connect_ok()
        conn_b = (set(printer._conn_endpoints) - before).pop()

        def allocate() -> None:
            printer.agent_task_allocated(agent, persisted_id, chat_id)
            vs._on_run_task_id_allocated(
                str(persisted_id), chat_id,
                source_tab_id="tab-t", conn_id="", start_ms=0,
            )

        def emit() -> None:
            printer._thread_local.task_id = str(persisted_id)
            printer.broadcast(
                {"type": "worktree_created", "worktreeDir": _GAP_DIR},
            )

        allocator = threading.Thread(target=allocate, daemon=True)
        emitter = threading.Thread(target=emit, daemon=True)

        def hook(frame: FrameType, event: str, arg: Any) -> None:
            name = frame.f_code.co_name
            if name == "cleanup_tab" and allocator.ident is None:
                allocator.start()
                # Runs to completion unless the replay holds a lock
                # the allocation needs (it then finishes afterwards).
                allocator.join(1.0)
            elif name == "_publish_replay_reopen" and emitter.ident is None:
                allocator.join(5)
                assert not allocator.is_alive(), "allocation never finished"
                # Inside the gap: after the attach's cleanup, before the
                # publication's commit re-subscribes the tab, and with
                # no replay lock held, so the event fans out at once.
                emitter.start()
                emitter.join(1.0)

        def replay() -> None:
            sys.settrace(hook)
            try:
                vs._replay_session(chat_id, "tab-t", conn_id=conn_b)
            finally:
                sys.settrace(None)

        def run_replay() -> None:
            replayer = threading.Thread(target=replay, daemon=True)
            replayer.start()
            replayer.join(15)
            emitter.join(5)
            assert not replayer.is_alive() and not emitter.is_alive()

        await asyncio.to_thread(run_replay)

        def is_gap_event(ev: dict[str, Any]) -> bool:
            return (
                ev.get("tabId") == "tab-t" and ev.get("worktreeDir") == _GAP_DIR
            )

        got_a = await self._wait_for_event(
            ws_a, "worktree_created", timeout=3.0, pred=is_gap_event,
        )
        self.assertIsNotNone(
            got_a,
            "window A never received the event emitted under the persisted "
            "task id allocated during B's scoped replay",
        )
        self.assertIsNotNone(await self._wait_for_event(
            ws_b, "task_events", pred=lambda ev: ev.get("tabId") == "tab-t",
        ))
        self.assertIn("tab-t", printer._fanout_targets(persisted_id))
        state.is_task_active = False
