# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for the server-core audit findings.

Every test here drives a **real** :class:`RemoteAccessServer` bound to
an ephemeral TCP port (serving remote and token-bearing local WSS
clients alike), with
persistence redirected to a scratch sqlite database, and speaks the
real newline-delimited JSON wire protocol over a real client socket.
Nothing is mocked, patched or doubled: the agent states the handlers
consult are the real :mod:`kiss.server.agent_state` objects, the
worker threads are real threads.

Findings covered (audit ``08-server-web.md`` / ``09-server-rest.md``):

* **F08-1** — ``_cmd_run`` refused a run while a merge is in flight by
  broadcasting only an ``error``, leaving the optimistically raised
  ``status running:true`` set forever, so the tab's composer stayed
  disabled until the tab was closed.
* **F08-2** — the shutdown sweep and the two liveness snapshots gated
  on ``is_task_active`` alone, although a worker that has started but
  not yet raised the flag is a real, live task
  (:meth:`AgentState.thread_alive`).
* **F08-5** — ``AgentServer.__init__`` cleared the *process-global*
  agent-state registry, so merely constructing a second server
  detached the first server's live tasks.
* **F08-6** — ``AgentServer._tab_opened_task_ids`` was a mutable
  **class** attribute shadowed by an instance attribute: dead, and a
  latent cross-instance shared-state hazard.
* **F08-7** — ``_handle_submit`` rebuilt the ``run`` command without
  ``connId``, so a task launched through ``submit`` recorded ``""``
  while the identical task launched through ``run`` recorded the real
  connection id.

No test makes a paid LLM call: runs are submitted with a model name
that is not in ``get_available_models()``, which makes the real worker
return at ``task_runner``'s "No model available" guard before any
agent work.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.server.web_server import (
    RemoteAccessServer,
    _generate_self_signed_cert,
    _snapshot_active_tabs,
)
from kiss.tests.local_ws import LocalReader, LocalWriter, open_local_connection
from kiss.tests.server.conftest import (
    find_free_port,
    no_verify_ssl,
    redirect_persistence,
    restore_persistence,
)

_UNAVAILABLE_MODEL = "kiss-test-no-such-model"
"""Model name guaranteed to be absent from ``get_available_models()``.

Makes the real worker thread hit ``task_runner``'s "No model
available." guard and return before contacting any provider.
"""


class _ServerHarness(IsolatedAsyncioTestCase):
    """Real ``RemoteAccessServer`` on an ephemeral port with a local endpoint file."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.port = find_free_port()
        self.work_dir = Path(self.tmpdir) / "repo"
        self.work_dir.mkdir()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
            work_dir=str(self.work_dir),
        )
        await self.server.start_async()
        self._stopped = False
        self._writers: list[LocalWriter] = []
        self._workers: list[tuple[threading.Event, threading.Thread]] = []

    async def asyncTearDown(self) -> None:
        for writer in self._writers:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        for release, thread in self._workers:
            release.set()
            thread.join(timeout=5)
        if not self._stopped:
            await self.server.stop_async()
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()
        if th._db_conn is not None:
            th._db_conn.close()
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect(
        self,
    ) -> tuple[LocalReader, LocalWriter]:
        """Open one real local connection (simulates one window)."""
        reader, writer = await open_local_connection(
            self.server, limit=16 * 1024 * 1024,
        )
        self._writers.append(writer)
        return reader, writer

    async def _send(
        self, writer: LocalWriter, cmd: dict[str, Any],
    ) -> None:
        """Write one newline-framed JSON command."""
        writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await writer.drain()

    async def _drain_until(
        self,
        reader: LocalReader,
        predicate: Callable[[dict[str, Any]], bool],
        max_events: int = 200,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Read frames until *predicate* matches, else fail the test."""
        for _ in range(max_events):
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            assert line, "local connection closed unexpectedly"
            msg = json.loads(line.decode("utf-8"))
            assert isinstance(msg, dict)
            if predicate(msg):
                return msg
        raise AssertionError("predicate never matched")

    async def _await_dispatch(
        self, reader: LocalReader, writer: LocalWriter,
    ) -> None:
        """Block until every command sent so far has been dispatched.

        The daemon dispatches this connection's commands sequentially, so
        the reply to a trailing ``activeTasksQuery`` cannot be produced
        until the preceding commands' handlers have returned.  (It says
        nothing about *broadcast* delivery, which is scheduled onto the
        loop from the executor thread; use :meth:`_collect_frames` for
        that.)
        """
        await self._send(writer, {"type": "activeTasksQuery"})
        await self._drain_until(
            reader, lambda m: m.get("type") == "activeTasksResponse",
        )

    async def _collect_frames(
        self,
        reader: LocalReader,
        stop: Callable[[dict[str, Any]], bool],
        timeout: float = 10.0,
    ) -> list[dict[str, Any]]:
        """Read frames until *stop* matches or *timeout* elapses."""
        seen: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return seen
            try:
                line = await asyncio.wait_for(
                    reader.readline(), timeout=remaining,
                )
            except TimeoutError:
                return seen
            if not line:
                return seen
            msg = json.loads(line.decode("utf-8"))
            seen.append(msg)
            if stop(msg):
                return seen

    def _register_starting_task(
        self,
        tab_id: str,
        task_id: str,
        chat_id: str = "",
    ) -> tuple[agent_state.AgentState, threading.Event]:
        """Register a real state in the ``_cmd_run`` startup window.

        Reproduces the exact interleaving of F08-2: ``_cmd_run`` has
        registered the state and started the worker thread, but the
        worker has not yet reached ``state.is_task_active = True``.
        The worker is a real thread that honours the state's real
        ``stop_event``, so a correct shutdown sweep stops and joins it.
        """
        stop_event = threading.Event()
        exited = threading.Event()
        state = agent_state.AgentState(
            task_id,
            chat_id=chat_id or f"chat-{task_id}",
            tab_id=tab_id,
            server_owned=True,
            stop_event=stop_event,
        )
        state.last_user_prompt = f"prompt for {task_id}"

        def _worker() -> None:
            stop_event.wait(timeout=30)
            exited.set()

        thread = threading.Thread(
            target=_worker, name=f"h-worker-{task_id}", daemon=True,
        )
        state.task_thread = thread
        agent_state.register(state)
        thread.start()
        self._workers.append((stop_event, thread))
        return state, exited


class TestRunRefusedDuringMergeClearsStatus(_ServerHarness):
    """F08-1: a refused run must lower the client's running flag."""

    async def _refusal_frames(
        self, reader: LocalReader, tab_id: str,
    ) -> list[dict[str, Any]]:
        """Collect frames until the refusal is fully delivered.

        The refusal is two events — the running flag is lowered and
        the reason is shown — so reading stops only once BOTH have
        arrived (or the bound expires, which is how a missing one is
        reported).
        """
        pending = {"status", "error"}

        def _stop(msg: dict[str, Any]) -> bool:
            if msg.get("tabId") != tab_id:
                return False
            if msg.get("type") == "status" and msg.get("running") is False:
                pending.discard("status")
            elif msg.get("type") == "error":
                pending.discard("error")
            return not pending

        return await self._collect_frames(reader, _stop, timeout=5.0)

    def _claim_merge(self, tab_id: str) -> agent_state.AgentState:
        """Take the tab's merge claim exactly as ``merge_flow`` does."""
        state = agent_state.AgentState(
            "task-under-merge",
            chat_id="chat-under-merge",
            tab_id=tab_id,
            server_owned=True,
        )
        agent_state.register(state)
        with agent_state.STATE_LOCK:
            state.is_merging = True
        return state

    async def test_submit_during_merge_emits_status_false(self) -> None:
        """The browser ``submit`` dialect must un-disable the composer."""
        tab_id = "tab-merge-submit"
        state = self._claim_merge(tab_id)
        reader, writer = await self._connect()

        await self._send(writer, {
            "type": "submit",
            "prompt": "follow-up while merging",
            "tabId": tab_id,
            "model": _UNAVAILABLE_MODEL,
        })
        frames = await self._refusal_frames(reader, tab_id)

        errors = [
            f for f in frames
            if f.get("type") == "error"
            and "merge is in progress" in str(f.get("text", ""))
        ]
        self.assertTrue(errors, f"no merge refusal error in {frames}")
        self.assertTrue(
            any(
                f.get("type") == "status"
                and f.get("running") is True
                and f.get("tabId") == tab_id
                for f in frames
            ),
            "the optimistic running:true was not observed",
        )
        cleared = [
            f for f in frames
            if f.get("type") == "status"
            and f.get("running") is False
            and f.get("tabId") == tab_id
        ]
        self.assertTrue(
            cleared,
            "BUG F08-1: the refused run never broadcast "
            "status running:false, so the tab's input stays disabled "
            f"forever; frames were {[f.get('type') for f in frames]}",
        )
        self.assertTrue(state.is_merging, "the merge claim was destroyed")

    async def test_run_during_merge_emits_status_false(self) -> None:
        """The ``run`` command dialect goes through the same guard."""
        tab_id = "tab-merge-run"
        self._claim_merge(tab_id)
        reader, writer = await self._connect()

        await self._send(writer, {
            "type": "run",
            "prompt": "follow-up while merging",
            "tabId": tab_id,
            "model": _UNAVAILABLE_MODEL,
        })
        frames = await self._refusal_frames(reader, tab_id)

        self.assertTrue(
            [
                f for f in frames
                if f.get("type") == "error"
                and "merge is in progress" in str(f.get("text", ""))
            ],
            f"no merge refusal error in {frames}",
        )
        self.assertTrue(
            [
                f for f in frames
                if f.get("type") == "status"
                and f.get("running") is False
                and f.get("tabId") == tab_id
            ],
            "BUG F08-1: the chat page raised running:true before "
            "sending run, and the refusal never cleared it",
        )

    async def test_accepted_run_is_not_refused(self) -> None:
        """A tab with no merge claim still starts its task."""
        tab_id = "tab-no-merge"
        reader, writer = await self._connect()
        await self._send(writer, {
            "type": "run",
            "prompt": "a normal task",
            "tabId": tab_id,
            "model": _UNAVAILABLE_MODEL,
        })
        result = await self._drain_until(
            reader,
            lambda m: m.get("type") == "result" and m.get("tabId") == tab_id,
        )
        self.assertIn("No model available", str(result.get("text", "")))


class TestStartupWindowIsLive(_ServerHarness):
    """F08-2: a started-but-unflagged worker counts as a live task."""

    async def test_active_tasks_query_reports_starting_task(self) -> None:
        """``activeTasksQuery`` must not answer 0 during the window.

        ``install.sh`` restarts the daemon when the
        reply says ``count: 0``; answering 0 for a task whose worker
        has just been started kills it.
        """
        self._register_starting_task("tab-window", "task-window")
        reader, writer = await self._connect()

        await self._send(writer, {"type": "activeTasksQuery"})
        resp = await self._drain_until(
            reader, lambda m: m.get("type") == "activeTasksResponse",
        )
        self.assertEqual(
            resp.get("count"), 1,
            "BUG F08-2: a task whose worker started but has not yet "
            "raised is_task_active was reported as idle, so the "
            "installer would restart the daemon on top of it",
        )
        self.assertIn("tab-window(task=task-window)", resp.get("tabs", []))

    async def test_shutdown_sweep_stops_starting_task(self) -> None:
        """``stop_async`` must stop and join a task in the window.

        The embedder shutdown path sweeps exactly once, so a task
        skipped by the sweep is abandoned outright: no stop event, no
        join, no cleanup ``finally``.
        """
        state, exited = self._register_starting_task("tab-shut", "task-shut")

        await self.server.stop_async()
        self._stopped = True

        self.assertTrue(
            exited.is_set(),
            "BUG F08-2: the shutdown sweep skipped the starting task, "
            "so its worker was never signalled and its history row is "
            "stranded at the abrupt-failure sentinel",
        )
        thread = state.task_thread
        assert thread is not None
        self.assertFalse(thread.is_alive())
        self.assertTrue(state.interrupted_by_shutdown)

    async def test_shutdown_sweep_ignores_finished_states(self) -> None:
        """A state with no live worker is neither active nor swept."""
        state = agent_state.AgentState(
            "task-done", tab_id="tab-done", server_owned=True,
        )
        agent_state.register(state)
        self.assertEqual(_snapshot_active_tabs(), [])

        await self.server.stop_async()
        self._stopped = True

        self.assertFalse(state.interrupted_by_shutdown)

    async def test_ready_restores_tab_of_starting_task(self) -> None:
        """A browser connecting in the window still gets the tab.

        ``_cmd_run`` registers the tab in the shared tab registry
        synchronously, BEFORE the worker thread starts, so a client
        whose ``ready`` lands inside the startup window (worker alive,
        ``is_task_active`` not yet raised) must still receive a
        ``tabs_state`` snapshot binding the tab to its chat.
        """
        self._register_starting_task(
            "tab-restore", "task-restore", chat_id="chat-restore",
        )
        # What _cmd_run does synchronously before starting the worker:
        self.server._agent_server._registry_update_tab(
            "tab-restore",
            chat_id="chat-restore",
            title="starting task",
            create=True,
        )
        ctx = no_verify_ssl()
        async with connect(f"wss://127.0.0.1:{self.port}/ws", ssl=ctx) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            auth = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            self.assertEqual(auth.get("type"), "auth_ok")

            await ws.send(json.dumps({"type": "ready", "tabId": "tab-new"}))
            rows: list[dict[str, Any]] = []
            found = False
            # The ``tabs_state`` snapshot is broadcast from a worker
            # thread (``ready_tab_sync`` via ``asyncio.to_thread``)
            # through ``run_coroutine_threadsafe``, while direct
            # replies are awaited on the event loop; the per-endpoint
            # send lock serializes only sends already in flight, so a
            # later direct reply may legitimately hit the wire before
            # the scheduled snapshot.  Read until the snapshot with the
            # running tab arrives instead of using another reply as an
            # ordering barrier.
            for _ in range(200):
                try:
                    ev = json.loads(
                        await asyncio.wait_for(ws.recv(), timeout=15),
                    )
                except TimeoutError:
                    break
                if ev.get("type") != "tabs_state":
                    continue
                rows.extend(ev.get("tabs", []))
                if any(
                    r.get("tabId") == "tab-restore"
                    and r.get("chatId") == "chat-restore"
                    for r in rows
                ):
                    found = True
                    break

        self.assertTrue(
            found,
            "BUG F08-2: a client connecting during the startup window "
            f"silently omitted the running tab; got {rows}",
        )


class TestSecondServerPreservesRegistry(_ServerHarness):
    """F08-5: constructing a server must not detach live tasks."""

    async def test_construction_keeps_live_states_and_drops_dead(
        self,
    ) -> None:
        """Only states with no live owner may be evicted."""
        live, exited = self._register_starting_task("tab-live", "task-live")
        live.is_task_active = True
        dead = agent_state.AgentState(
            "task-dead", tab_id="tab-dead", server_owned=True,
        )
        agent_state.register(dead)

        RemoteAccessServer(
            host="127.0.0.1",
            port=find_free_port(),
            url_file=Path(self.tmpdir) / "remote-url-2.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local-2.json",
            work_dir=str(self.work_dir),
        )

        self.assertIs(
            agent_state.get("task-live"), live,
            "BUG F08-5: constructing a second server cleared the "
            "process-global registry and detached a live task",
        )
        self.assertIn("tab-live(task=task-live)", _snapshot_active_tabs())
        self.assertIsNone(
            agent_state.get("task-dead"),
            "a finished state must still be evicted by a new server",
        )

        await self.server.stop_async()
        self._stopped = True
        self.assertTrue(
            exited.is_set(),
            "BUG F08-5: the first server could no longer stop the task "
            "it owns, so the worker is killed at exit",
        )


class TestNoSharedClassLevelTabMap(unittest.TestCase):
    """F08-6: the tab→task map must not be a mutable class default."""

    def tearDown(self) -> None:
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()

    def test_tab_opened_task_ids_is_instance_only(self) -> None:
        """Two servers must never share one tab→task dict."""
        self.assertNotIn(
            "_tab_opened_task_ids", vars(AgentServer),
            "BUG F08-6: the dead mutable class attribute is back; any "
            "read before __init__ finishes would silently share one "
            "dict across every server in the process",
        )
        first = AgentServer()
        second = AgentServer()
        first._tab_opened_task_ids["tab-a"] = "task-a"
        self.assertEqual(second._tab_opened_task_ids, {})
        self.assertEqual(first._tab_opened_task_ids, {"tab-a": "task-a"})


class TestSubmitRecordsConnId(_ServerHarness):
    """F08-7: ``submit`` and ``run`` must record the same connection."""

    async def _launch(
        self, writer: LocalWriter,
        reader: LocalReader,
        cmd_type: str,
        tab_id: str,
    ) -> agent_state.AgentState:
        """Run one task to completion and return its agent state."""
        await self._send(writer, {
            "type": cmd_type,
            "prompt": "record my connection",
            "tabId": tab_id,
            "model": _UNAVAILABLE_MODEL,
        })
        await self._drain_until(
            reader,
            lambda m: m.get("type") == "status"
            and m.get("running") is False
            and m.get("tabId") == tab_id,
        )
        state = agent_state.find_by_tab(tab_id)
        assert state is not None, f"no state registered for {tab_id}"
        return state

    async def test_submit_and_run_agree_on_conn_id(self) -> None:
        """Both dialects on one connection record that connection."""
        reader, writer = await self._connect()
        await self._send(writer, {"type": "ready", "tabId": "tab-run"})
        await self._drain_until(
            reader, lambda m: m.get("type") == "configData",
        )

        via_run = await self._launch(writer, reader, "run", "tab-run")
        via_submit = await self._launch(
            writer, reader, "submit", "tab-submit",
        )

        self.assertTrue(
            via_run.conn_id, "the run dialect lost its connection id",
        )
        self.assertEqual(
            via_submit.conn_id, via_run.conn_id,
            "BUG F08-7: _handle_submit rebuilt the run command without "
            "connId, so a submit-launched task records \"\" while an "
            "identical run-launched task records the real id",
        )


if __name__ == "__main__":
    unittest.main()
