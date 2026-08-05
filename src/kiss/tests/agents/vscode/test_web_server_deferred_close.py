# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for deferred ``closeTab`` dispatch in the web server.

Browsers cannot reliably send a ``closeTab`` over the WebSocket
before the window closes (``beforeunload`` / ``pagehide`` writes are
commonly dropped), and the WS drop itself carries no per-tab
identity.  ``RemoteAccessServer`` therefore arms a grace timer for
every tab id seen on a connection when that connection drops; if a
reconnect within ``_TAB_CLOSE_GRACE`` seconds re-claims the tab id
(current ``tabId`` or any entry in ``restoredTabs``), the pending
close is cancelled.  Otherwise the timer fires a real ``closeTab``
through :class:`VSCodeServer._close_tab`, which either unregisters
the idle tab's :class:`kiss.server.agent_state.AgentState`
immediately OR flips ``frontend_closed=True`` so the existing
:meth:`VSCodeServer._dispose_if_closed` hook tears it down once the
running agent finishes — never interrupting the live task.

The tests pin this contract end-to-end against ``RemoteAccessServer``
without mocks: a real asyncio loop drives ``loop.call_later``, the
task-keyed ``kiss.server.agent_state`` registry holds the tab states,
and the pending timers are exercised through the public-on-the-class
helpers ``_schedule_tab_close`` / ``_cancel_pending_tab_close``.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.server.web_server import RemoteAccessServer


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    """Redirect the persistence DB to a temp dir; return saved state."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore_persistence(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


def _noop_broadcast(event: dict[str, Any]) -> None:
    """No-op broadcast target for quiet tests."""


def _silence_broadcasts(server: RemoteAccessServer) -> None:
    """Replace the printer's broadcast with a no-op for quiet tests."""
    server._printer.broadcast = _noop_broadcast  # type: ignore[assignment]


def _register_tab_state(
    task_id: str,
    tab_id: str,
    *,
    is_task_active: bool = False,
) -> agent_state.AgentState:
    """Register a server-owned tab state exactly like a UI-launched run."""
    state = agent_state.AgentState(
        task_id,
        tab_id=tab_id,
        server_owned=True,
        is_task_active=is_task_active,
    )
    agent_state.register(state)
    return state


class TestDeferredWebTabClose(IsolatedAsyncioTestCase):
    """Verify the WS-drop → grace-timer → ``closeTab`` flow."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)
        import kiss.server.web_server as ws

        self._orig_grace = ws._TAB_CLOSE_GRACE
        ws._TAB_CLOSE_GRACE = 0.05

        agent_state.agent_states.clear()

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        from kiss.server.web_server import _generate_self_signed_cert
        _generate_self_signed_cert(certfile, keyfile)

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
        )
        self.server._loop = asyncio.get_running_loop()
        _silence_broadcasts(self.server)

    async def asyncTearDown(self) -> None:
        with self.server._pending_tab_closes_lock:
            for h in list(self.server._pending_tab_closes.values()):
                try:
                    h.cancel()
                except Exception:
                    pass
            self.server._pending_tab_closes.clear()
        import kiss.server.web_server as ws
        ws._TAB_CLOSE_GRACE = self._orig_grace

        agent_state.agent_states.clear()

        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _wait_pending_clear(self, tab_id: str, timeout: float = 2.0) -> None:
        """Wait until ``_pending_tab_closes`` no longer contains *tab_id*."""
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            with self.server._pending_tab_closes_lock:
                if tab_id not in self.server._pending_tab_closes:
                    return
            if asyncio.get_event_loop().time() >= deadline:
                raise AssertionError(
                    f"Pending close for {tab_id!r} did not clear in {timeout}s",
                )
            await asyncio.sleep(0.01)

    async def _wait_disposed(self, tab_id: str, timeout: float = 2.0) -> None:
        """Wait until the ``agent_state`` registry no longer holds a state
        for *tab_id* (the deferred ``closeTab`` task has finished
        executing in the thread-pool executor)."""
        deadline = asyncio.get_event_loop().time() + timeout
        while agent_state.find_by_tab(tab_id) is not None:
            if asyncio.get_event_loop().time() >= deadline:
                raise AssertionError(
                    f"agent state for {tab_id!r} not disposed in {timeout}s",
                )
            await asyncio.sleep(0.01)

    async def test_schedule_arms_timer(self) -> None:
        """``_schedule_tab_close`` arms a TimerHandle for the tab id."""
        self.server._schedule_tab_close("tab-A")
        with self.server._pending_tab_closes_lock:
            self.assertIn("tab-A", self.server._pending_tab_closes)

    async def test_cancel_clears_pending(self) -> None:
        """``_cancel_pending_tab_close`` removes the pending entry and
        prevents the timer from firing.
        """
        _register_tab_state("task-B", "tab-B")
        self.server._schedule_tab_close("tab-B")
        self.server._cancel_pending_tab_close("tab-B")
        with self.server._pending_tab_closes_lock:
            self.assertNotIn("tab-B", self.server._pending_tab_closes)
        await asyncio.sleep(0.15)
        self.assertIsNotNone(agent_state.find_by_tab("tab-B"))

    async def test_unknown_and_empty_ids_are_safe(self) -> None:
        """Empty / unknown tab ids cause no errors and arm no timers."""
        self.server._schedule_tab_close("")
        self.server._cancel_pending_tab_close("")
        self.server._cancel_pending_tab_close("never-existed")
        with self.server._pending_tab_closes_lock:
            self.assertEqual(self.server._pending_tab_closes, {})

    async def test_grace_period_disposes_idle_tab(self) -> None:
        """After the grace window, an idle tab is fully disposed."""
        tab_id = "tab-idle"
        _register_tab_state("task-idle", tab_id)
        self.server._schedule_tab_close(tab_id)
        await self._wait_pending_clear(tab_id)
        await self._wait_disposed(tab_id)

    async def test_grace_period_defers_running_tab(self) -> None:
        """A tab whose task is still running has its ``AgentState`` kept
        alive but flagged ``frontend_closed=True`` for later disposal.
        """
        tab_id = "tab-running"
        state = _register_tab_state("task-running", tab_id, is_task_active=True)

        release = threading.Event()

        def fake_task() -> None:
            release.wait(timeout=5)

        thr = threading.Thread(target=fake_task, daemon=True)
        state.task_thread = thr
        thr.start()

        try:
            self.server._schedule_tab_close(tab_id)
            await self._wait_pending_clear(tab_id)
            for _ in range(100):
                if state.frontend_closed:
                    break
                await asyncio.sleep(0.01)
            self.assertIsNotNone(agent_state.find_by_tab(tab_id))
            self.assertTrue(state.frontend_closed)

            release.set()
            thr.join(timeout=5)
            with agent_state.STATE_LOCK:
                state.task_thread = None
                state.is_task_active = False
            self.server._vscode_server._dispose_if_closed(tab_id)
            self.assertIsNone(agent_state.find_by_tab(tab_id))
        finally:
            release.set()
            thr.join(timeout=5)

    async def test_reconnect_via_handle_ready_cancels_close(self) -> None:
        """A ``ready`` reconnect cancels the pending close for both the
        current ``tabId`` and every entry in ``restoredTabs``.
        """
        import kiss.server.web_server as ws

        for tab_id in ("tab-X", "tab-Y", "tab-Z"):
            _register_tab_state(f"task-{tab_id}", tab_id)
            self.server._schedule_tab_close(tab_id)
        with self.server._pending_tab_closes_lock:
            self.assertEqual(
                set(self.server._pending_tab_closes.keys()),
                {"tab-X", "tab-Y", "tab-Z"},
            )

        self.server._cancel_pending_tab_close("tab-X")
        for restored_id in ("tab-Y", "tab-Z"):
            self.server._cancel_pending_tab_close(restored_id)

        with self.server._pending_tab_closes_lock:
            self.assertEqual(self.server._pending_tab_closes, {})

        await asyncio.sleep(ws._TAB_CLOSE_GRACE + 0.1)
        for tab_id in ("tab-X", "tab-Y", "tab-Z"):
            self.assertIsNotNone(agent_state.find_by_tab(tab_id))

    async def test_dispose_if_closed_ignores_unflagged_state(self) -> None:
        """``_dispose_if_closed`` must not tear down a state whose
        ``frontend_closed`` flag was cleared by a reconnect (a
        re-claimed tab survives the post-task disposal hook).
        """
        tab_id = "tab-reload"
        state = _register_tab_state("task-reload", tab_id)
        state.frontend_closed = True

        with agent_state.STATE_LOCK:
            state.frontend_closed = False

        self.assertFalse(state.frontend_closed)
        self.server._vscode_server._dispose_if_closed(tab_id)
        self.assertIsNotNone(agent_state.find_by_tab(tab_id))

    async def test_replay_session_clears_frontend_closed(self) -> None:
        """End-to-end: a real ``_replay_session`` call clears the
        ``frontend_closed`` flag for a tab id that was flagged by an
        earlier deferred ``_close_tab``.
        """
        tab_id = "tab-replay"
        state = _register_tab_state("task-replay", tab_id)
        state.frontend_closed = True

        # The chat id is unknown in the (temp) DB, so the replay takes
        # the no-events path — which must still clear the flag for the
        # re-claimed tab.
        self.server._vscode_server._replay_session("chat-stub", tab_id)

        self.assertFalse(state.frontend_closed)
        self.server._vscode_server._dispose_if_closed(tab_id)
        self.assertIsNotNone(agent_state.find_by_tab(tab_id))

    async def test_double_schedule_replaces_existing_timer(self) -> None:
        """Re-scheduling the same tab id cancels the prior timer."""
        tab_id = "tab-respawn"
        _register_tab_state("task-respawn", tab_id)
        self.server._schedule_tab_close(tab_id)
        with self.server._pending_tab_closes_lock:
            h1 = self.server._pending_tab_closes[tab_id]
        self.server._schedule_tab_close(tab_id)
        with self.server._pending_tab_closes_lock:
            h2 = self.server._pending_tab_closes[tab_id]
        self.assertIsNot(h1, h2)
        self.assertTrue(h1.cancelled())

    async def test_stop_async_cancels_pending(self) -> None:
        """``stop_async`` cancels every armed deferred-close timer."""
        for tab_id in ("a", "b", "c"):
            _register_tab_state(f"task-{tab_id}", tab_id)
            self.server._schedule_tab_close(tab_id)
        with self.server._pending_tab_closes_lock:
            self.assertEqual(len(self.server._pending_tab_closes), 3)
        await self.server.stop_async()
        with self.server._pending_tab_closes_lock:
            self.assertEqual(self.server._pending_tab_closes, {})
