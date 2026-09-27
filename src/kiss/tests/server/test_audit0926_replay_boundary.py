# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a replay snapshot and a live event never interleave.

The webview REPLACES a tab's transcript when a ``task_events`` replay
arrives, and appends every live event.  A replay of a running task is
built from the printer's in-memory recording, while a live task event
is recorded and then fanned out to the task's subscribed tabs.  When
the two were not atomic with respect to each other:

1a. an event recorded just before the replay's snapshot but fanned out
    just after the replay was sent rendered twice (in the replay and
    again live);
1b. an event recorded and fanned out after the snapshot but before the
    replay was sent vanished (shown live, then wiped by a replay whose
    snapshot predates it);
2.  a setup-failure ``result`` recorded by the launcher's broadcast and
    then copied live to every viewer subscribed to the provisional task
    id reached a viewer that attached in between twice (replayed AND
    copied).

Everything is real: one daemon, real WSS clients A and B, B receiving a
connection-scoped replay.  The interleavings are forced with trace
hooks that park one real thread at the racy point while the other runs
as far as the printer's locks allow.  The client-side view is computed
the way ``main.js`` renders it: a replay replaces, live events append.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any

from websockets.asyncio.client import ClientConnection

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.tests.server.test_web_server_tab_mirroring import (
    TabMirroringBase,
)

_SENTINEL = "audit0926_replay_boundary_sentinel"

_FAILING_SCRIPT = textwrap.dedent(
    """
    import pathlib
    import time

    _DIR = pathlib.Path(__file__).resolve().parent


    def prompt():
        \"\"\"Block until released, then raise (the task ends in setup).\"\"\"
        deadline = time.time() + 60
        while time.time() < deadline:
            if (_DIR / "release").exists():
                raise RuntimeError("setup exploded")
            time.sleep(0.02)
        raise RuntimeError("timed out waiting for the release")
    """
)


def _rendered(messages: list[dict[str, Any]], tab_id: str) -> list[dict[str, Any]]:
    """The transcript *tab_id* shows after *messages*, as main.js renders it."""
    view: list[dict[str, Any]] = []
    for ev in messages:
        if ev.get("tabId") != tab_id:
            continue
        if ev.get("type") == "task_events":
            view = [dict(e) for e in ev.get("events") or []]
        elif ev.get("type") in ("text_delta", "result"):
            view.append(ev)
    return view


def _text(view: list[dict[str, Any]]) -> str:
    return "".join(
        str(ev.get("text", "")) for ev in view if ev.get("type") == "text_delta"
    )


def _hook(
    func_name: str,
    when: str,
    on_hit: Callable[[], None],
    pred: Callable[[FrameType], bool] | None = None,
) -> Callable[[FrameType, str, Any], Any]:
    """A trace hook calling *on_hit* once, at the first *when* event
    (``"call"`` or ``"return"``) of a *func_name* frame matching *pred*."""
    fired: list[bool] = []

    def local(frame: FrameType, event: str, arg: Any) -> Any:
        if event == "return" and not fired:
            fired.append(True)
            on_hit()
        return local

    def hook(frame: FrameType, event: str, arg: Any) -> Any:
        if (
            event != "call"
            or fired
            or frame.f_code.co_name != func_name
            or (pred is not None and not pred(frame))
        ):
            return None
        if when == "call":
            fired.append(True)
            on_hit()
            return None
        return local

    return hook


def _is_launcher_result(frame: FrameType) -> bool:
    event = frame.f_locals.get("event")
    return (
        isinstance(event, dict)
        and event.get("type") == "result"
        and event.get("tabId") == "rb-launcher"
    )


class TestReplayBoundary(TabMirroringBase):
    """Replay snapshots are atomic with respect to live task events."""

    async def _drain(self, ws: ClientConnection) -> list[dict[str, Any]]:
        """Every message *ws* receives up to (excluding) the sentinel."""
        assert self.server is not None
        self.server._printer.broadcast({"type": _SENTINEL})
        out: list[dict[str, Any]] = []
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            ev = json.loads(raw)
            if ev.get("type") == _SENTINEL:
                return out
            out.append(ev)
        raise AssertionError("sentinel never arrived")

    async def _running_task_viewed_by_a(self) -> tuple[str, str]:
        """Register a running task shown on tab-t by client A."""
        assert self.server is not None
        printer = self.server._printer
        task_id, chat_id = th._add_task("Running task")
        th._append_chat_event(
            {"type": "prompt", "text": "running prompt"}, task_id=task_id,
        )
        agent_state.register(AgentState(
            str(task_id), chat_id=chat_id, tab_id="tab-t", is_task_active=True,
        ))
        printer.ensure_recording_for_task(task_id)
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
        self.ws_a = ws_a
        return str(task_id), chat_id

    async def _connect_b(self) -> tuple[ClientConnection, str]:
        assert self.server is not None
        printer = self.server._printer
        before = set(printer._conn_endpoints)
        ws_b = await self._connect_ok()
        return ws_b, (set(printer._conn_endpoints) - before).pop()

    def _emit_delta(self, task_id: str, text: str) -> None:
        assert self.server is not None
        printer = self.server._printer
        printer._thread_local.task_id = task_id
        printer.broadcast({"type": "text_delta", "text": text})

    async def test_1a_event_recorded_before_snapshot_renders_once(self) -> None:
        """Recorded before B's snapshot, fanned out after: shown once."""
        assert self.server is not None
        vs = self.server._vscode_server
        task_id, chat_id = await self._running_task_viewed_by_a()
        ws_b, conn_b = await self._connect_b()
        paused = threading.Event()
        release = threading.Event()

        def park() -> None:
            paused.set()
            release.wait(10)

        def emit() -> None:
            # Parks the producer after its event is recorded (and the
            # recording lock released), at the start of the fan-out.
            sys.settrace(_hook("_fanout_stamped", "call", park))
            try:
                self._emit_delta(task_id, "[X]")
            finally:
                sys.settrace(None)

        def run() -> None:
            emitter = threading.Thread(target=emit, daemon=True)
            emitter.start()
            assert paused.wait(5), "producer never recorded its event"
            replayer = threading.Thread(
                target=vs._replay_session,
                args=(chat_id, "tab-t"),
                kwargs={"conn_id": conn_b},
                daemon=True,
            )
            replayer.start()
            # Runs to completion unless it must wait for the producer.
            replayer.join(1.5)
            release.set()
            emitter.join(10)
            replayer.join(10)
            assert not emitter.is_alive() and not replayer.is_alive()

        await asyncio.to_thread(run)
        view = _rendered(await self._drain(ws_b), "tab-t")
        self.assertEqual(
            _text(view).count("[X]"),
            1,
            f"B must render the delta exactly once: {_text(view)!r}",
        )
        agent_state.agent_states.clear()

    async def test_1b_event_after_snapshot_is_not_wiped(self) -> None:
        """Recorded after B's snapshot, before its replay: not lost."""
        assert self.server is not None
        vs = self.server._vscode_server
        task_id, chat_id = await self._running_task_viewed_by_a()
        await asyncio.to_thread(self._emit_delta, task_id, "HEAD ")
        ws_b, conn_b = await self._connect_b()
        emitter = threading.Thread(
            target=self._emit_delta, args=(task_id, "[Y]"), daemon=True,
        )

        def emit_after_snapshot() -> None:
            emitter.start()
            # Completes unless it must wait for the replay.
            emitter.join(1.5)

        def replay() -> None:
            # The snapshot is filtered and coalesced after it is taken
            # (and, since the fix, after ``delivery_lock`` is released).
            sys.settrace(
                _hook("_filter_and_coalesce", "call", emit_after_snapshot),
            )
            try:
                vs._replay_session(chat_id, "tab-t", conn_id=conn_b)
            finally:
                sys.settrace(None)

        def run() -> None:
            replayer = threading.Thread(target=replay, daemon=True)
            replayer.start()
            replayer.join(15)
            emitter.join(10)
            assert not replayer.is_alive() and not emitter.is_alive()

        await asyncio.to_thread(run)
        view = _rendered(await self._drain(ws_b), "tab-t")
        self.assertEqual(
            _text(view).count("[Y]"),
            1,
            f"B must still show the delta after its replay: {_text(view)!r}",
        )
        self.assertEqual(_text(view).count("HEAD "), 1, _text(view))
        agent_state.agent_states.clear()

    async def test_2_setup_failure_result_reaches_late_viewer_once(
        self,
    ) -> None:
        """A viewer attaching between record and copies sees one result."""
        assert self.server is not None
        vs = self.server._vscode_server
        tmp = Path(self.tmpdir)
        work_dir = tmp / "wd"
        work_dir.mkdir()
        script = tmp / "agent.py"
        script.write_text(_FAILING_SCRIPT, encoding="utf-8")
        vs.work_dir = str(work_dir)
        ws_b, conn_b = await self._connect_b()
        launcher, viewer, chat_id = "rb-launcher", "rb-viewer", "chat-rb"
        paused = threading.Event()
        release = threading.Event()

        def park() -> None:
            paused.set()
            release.wait(10)

        # Parks the task thread once the launcher's recorded result
        # broadcast returns, before the viewer copies are chosen.
        hook = _hook("broadcast", "return", park, _is_launcher_result)
        threading.settrace(hook)
        try:
            vs._cmd_run({
                "type": "run",
                "prompt": "setup failure viewer",
                "tabId": launcher,
                "taskId": f"tok-{launcher}",
                "chatId": chat_id,
                "workDir": str(work_dir),
                "useWorktree": False,
                "useParallel": False,
                "autoCommit": False,
                "agentPath": str(script),
            })
        finally:
            threading.settrace(None)  # type: ignore[arg-type]
        state = agent_state.find_by_tab(launcher)
        assert state is not None and state.task_thread is not None
        task_thread = state.task_thread

        def run() -> None:
            (tmp / "release").write_text("1", encoding="utf-8")
            assert paused.wait(30), "the setup failure was never broadcast"
            replayer = threading.Thread(
                target=vs._replay_session,
                args=(chat_id, viewer),
                kwargs={"conn_id": conn_b},
                daemon=True,
            )
            replayer.start()
            # Runs to completion unless it must wait for the task thread.
            replayer.join(1.5)
            release.set()
            replayer.join(10)
            task_thread.join(30)
            assert not replayer.is_alive()
            assert not task_thread.is_alive()

        await asyncio.to_thread(run)
        view = _rendered(await self._drain(ws_b), viewer)
        results = [ev for ev in view if ev.get("type") == "result"]
        self.assertEqual(
            len(results), 1, f"the viewer must show one failure panel: {view}",
        )
        self.assertIn("setup exploded", results[0].get("text", ""))

    async def test_r1_persisted_order_matches_live_order(self) -> None:
        """A task event and a tab-stamped prompt persist in live order.

        The producer of a ``text_delta`` is parked after its event was
        recorded and sent but before it was queued for persistence; a
        tab-stamped injected ``prompt`` is then recorded, sent and
        persisted.  The events table (read back ``ORDER BY seq``) must
        keep the order the clients saw: delta first, prompt second.
        """
        assert self.server is not None
        printer = self.server._printer
        task_id, chat_id = th._add_task("Persist order")
        agent = WorktreeSorcarAgent("rb-persist")
        agent._last_task_id = str(task_id)
        agent_state.register(AgentState(
            str(task_id), agent=agent, chat_id=chat_id, tab_id="tab-p",
            is_task_active=True,
        ))
        paused = threading.Event()
        release = threading.Event()

        def park() -> None:
            paused.set()
            release.wait(10)

        def emit() -> None:
            sys.settrace(_hook("_persist_event", "call", park))
            try:
                self._emit_delta(str(task_id), "[A]")
            finally:
                sys.settrace(None)

        def run() -> None:
            emitter = threading.Thread(target=emit, daemon=True)
            emitter.start()
            # Parks only where persistence runs after the fan-out.
            paused.wait(1.5)
            printer.broadcast({
                "type": "prompt", "text": "[B]", "tabId": "tab-p",
                "taskId": str(task_id),
            })
            release.set()
            emitter.join(10)
            assert not emitter.is_alive()

        await asyncio.to_thread(run)
        th._flush_chat_events(str(task_id))
        loaded = th._load_chat_events_by_task_id(task_id)
        assert loaded is not None
        events = loaded["events"]
        assert isinstance(events, list)
        order = [
            ev.get("text") for ev in events if ev.get("text") in ("[A]", "[B]")
        ]
        self.assertEqual(order, ["[A]", "[B]"], events)
        agent_state.agent_states.clear()

    async def test_r2_stop_recording_before_snapshot_keeps_final_events(
        self,
    ) -> None:
        """The task ends between B's attach and its replay snapshot.

        The final text and result are fanned out live to B's tab and
        the run then retires its recording (``stop_recording``).  B's
        replay must still carry them instead of falling back to the
        events loaded from the database before the attach.
        """
        assert self.server is not None
        printer = self.server._printer
        vs = self.server._vscode_server
        task_id, chat_id = await self._running_task_viewed_by_a()
        await asyncio.to_thread(self._emit_delta, task_id, "HEAD ")
        ws_b, conn_b = await self._connect_b()

        def finish_task() -> None:
            printer._thread_local.task_id = task_id
            printer.broadcast({"type": "text_delta", "text": "FINAL"})
            printer.broadcast({"type": "result", "text": "all done"})
            printer.stop_recording()

        def end_task_after_attach() -> None:
            ender = threading.Thread(target=finish_task, daemon=True)
            ender.start()
            ender.join(10)
            assert not ender.is_alive()

        def replay() -> None:
            # ``_broadcast_viewer_running`` runs after the attach and
            # before the replay is built.
            sys.settrace(_hook(
                "_broadcast_viewer_running", "call", end_task_after_attach,
            ))
            try:
                vs._replay_session(chat_id, "tab-t", conn_id=conn_b)
            finally:
                sys.settrace(None)

        await asyncio.to_thread(replay)
        view = _rendered(await self._drain(ws_b), "tab-t")
        self.assertEqual(_text(view), "HEAD FINAL", view)
        results = [ev for ev in view if ev.get("type") == "result"]
        self.assertEqual([ev.get("text") for ev in results], ["all done"])
        agent_state.agent_states.clear()

    async def test_r3_payload_build_does_not_stall_other_deliveries(
        self,
    ) -> None:
        """Building B's replay payload blocks no other delivery.

        B's replayer is parked inside the payload build.  A live event
        emitted meanwhile must be recorded and fanned out without
        waiting for it, and A (another connection) must receive it at
        once; B gets it after its replay, exactly once.
        """
        assert self.server is not None
        vs = self.server._vscode_server
        task_id, chat_id = await self._running_task_viewed_by_a()
        for _ in range(2000):
            await asyncio.to_thread(self._emit_delta, task_id, "h")
        ws_b, conn_b = await self._connect_b()
        paused = threading.Event()
        release = threading.Event()

        def park() -> None:
            paused.set()
            release.wait(15)

        def replay() -> None:
            sys.settrace(_hook("with_task_settings_event", "call", park))
            try:
                vs._replay_session(chat_id, "tab-t", conn_id=conn_b)
            finally:
                sys.settrace(None)

        replayer = threading.Thread(target=replay, daemon=True)
        replayer.start()
        try:
            self.assertTrue(await asyncio.to_thread(paused.wait, 10))
            emitter = threading.Thread(
                target=self._emit_delta, args=(task_id, "[Z]"), daemon=True,
            )
            emitter.start()
            await asyncio.to_thread(emitter.join, 5)
            self.assertFalse(
                emitter.is_alive(), "the live event waited for the replay build",
            )
            got = await self._wait_for_event(
                self.ws_a, "text_delta", timeout=5,
                pred=lambda ev: ev.get("text") == "[Z]",
            )
            self.assertIsNotNone(got, "A's delivery waited for B's replay")
        finally:
            release.set()
            await asyncio.to_thread(replayer.join, 15)
        self.assertFalse(replayer.is_alive())
        view = _rendered(await self._drain(ws_b), "tab-t")
        self.assertEqual(_text(view), "h" * 2000 + "[Z]")
        agent_state.agent_states.clear()

    async def _finish_after(
        self,
        when: tuple[str, str],
        after_stop: Callable[[str], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Replay tab-t to B while the task ends at trace point *when*.

        The task's final text and result are broadcast and its recording
        retired (``stop_recording``) at the first ``(function, event)``
        trace point *when* of B's replaying thread; the task still counts
        as running unless *after_stop* (called with the task id right
        after ``stop_recording``) ends it.  Returns the transcript B
        renders.
        """
        assert self.server is not None
        printer = self.server._printer
        vs = self.server._vscode_server
        task_id, chat_id = await self._running_task_viewed_by_a()
        await asyncio.to_thread(self._emit_delta, task_id, "HEAD ")
        ws_b, conn_b = await self._connect_b()

        def finish_task() -> None:
            printer._thread_local.task_id = task_id
            printer.broadcast({"type": "text_delta", "text": "FINAL"})
            printer.broadcast({"type": "result", "text": "all done"})
            printer.stop_recording()
            if after_stop is not None:
                after_stop(task_id)

        def end_task() -> None:
            ender = threading.Thread(target=finish_task, daemon=True)
            ender.start()
            ender.join(10)
            assert not ender.is_alive()

        def replay() -> None:
            sys.settrace(_hook(*when, end_task))
            try:
                vs._replay_session(chat_id, "tab-t", conn_id=conn_b)
            finally:
                sys.settrace(None)

        if when[0]:
            await asyncio.to_thread(replay)
        else:
            await asyncio.to_thread(end_task)
            await asyncio.to_thread(
                vs._replay_session, chat_id, "tab-t", conn_id=conn_b,
            )
        view = _rendered(await self._drain(ws_b), "tab-t")
        agent_state.agent_states.clear()
        return view

    async def test_i3_recording_retired_right_after_attach(self) -> None:
        """The task retires its recording just after B's attach returns.

        B is subscribed but its replay snapshot has not been taken; the
        final events must still reach B through the replay (they were
        fanned out before B's replay, which wipes them) instead of the
        stale database events.
        """
        view = await self._finish_after(
            ("_attach_viewer_to_running_chat", "return"),
        )
        self.assertEqual(_text(view), "HEAD FINAL", view)
        results = [ev for ev in view if ev.get("type") == "result"]
        self.assertEqual([ev.get("text") for ev in results], ["all done"])

    async def test_i3_recording_retired_before_attach(self) -> None:
        """The task retired its recording but still counts as running.

        B attaches after ``stop_recording`` and before ``cleanup_task``:
        its replay must carry the final events.
        """
        view = await self._finish_after(("", ""))
        self.assertEqual(_text(view), "HEAD FINAL", view)
        results = [ev for ev in view if ev.get("type") == "result"]
        self.assertEqual([ev.get("text") for ev in results], ["all done"])

    async def test_i3_retired_recording_evicted_right_after_attach(
        self,
    ) -> None:
        """Other runs stop recording between B's attach and its snapshot.

        Parallel sub-agents share the printer and stop recording
        without ever being cleaned up; however many do so, B's replay
        must still carry the final events of the task it attached to.
        """
        assert self.server is not None
        printer = self.server._printer

        def other_runs_stop(_task_id: str) -> None:
            for i in range(20):
                printer._thread_local.task_id = f"other-run-{i}"
                printer.start_recording()
                printer.broadcast({"type": "text_delta", "text": "x"})
                printer.stop_recording()

        view = await self._finish_after(
            ("_attach_viewer_to_running_chat", "return"), other_runs_stop,
        )
        self.assertEqual(_text(view), "HEAD FINAL", view)
        results = [ev for ev in view if ev.get("type") == "result"]
        self.assertEqual([ev.get("text") for ev in results], ["all done"])

    async def test_i3_task_cleaned_up_right_after_attach(self) -> None:
        """The task ends and is cleaned up between B's attach and its
        snapshot: B's replay must still carry the final events it was
        already sent live."""
        assert self.server is not None
        printer = self.server._printer

        def end_and_clean_up(task_id: str) -> None:
            state = agent_state.get(task_id)
            assert state is not None
            with agent_state.STATE_LOCK:
                state.is_task_active = False
            printer.cleanup_task(task_id)

        view = await self._finish_after(
            ("_attach_viewer_to_running_chat", "return"), end_and_clean_up,
        )
        self.assertEqual(_text(view), "HEAD FINAL", view)
        results = [ev for ev in view if ev.get("type") == "result"]
        self.assertEqual([ev.get("text") for ev in results], ["all done"])

    async def test_i3_state_rekeyed_to_next_subtask_right_after_attach(
        self,
    ) -> None:
        """A sequential run moves on to its next subtask between B's
        attach and its snapshot.

        The runner cleans up subtask A and the printer bridge re-keys
        the SAME agent state to the freshly allocated subtask B.  B's
        replay of A must still use the recording it captured at attach
        (whose final events it was already sent live), not discard it
        because the state now carries B's id.
        """
        assert self.server is not None
        printer = self.server._printer
        agent = WorktreeSorcarAgent("rb-sequential")

        def next_subtask(task_id: str) -> None:
            state = agent_state.get(task_id)
            assert state is not None
            state.agent = agent
            state.server_owned = True
            printer.agent_task_finished(agent, task_id)
            printer.cleanup_task(task_id)
            next_id, _ = th._add_task("second subtask", chat_id=state.chat_id)
            printer.agent_task_allocated(agent, next_id, state.chat_id)
            assert state.task_id == str(next_id)

        view = await self._finish_after(
            ("_attach_viewer_to_running_chat", "return"), next_subtask,
        )
        self.assertEqual(_text(view), "HEAD FINAL", view)
        results = [ev for ev in view if ev.get("type") == "result"]
        self.assertEqual([ev.get("text") for ev in results], ["all done"])

    async def test_i3_many_runs_retired_before_attach_to_oldest(
        self,
    ) -> None:
        """More than 16 still-registered runs retire their recordings
        after the task did and before B attaches to it: B's replay must
        still carry the task's final events (no retirement cap may
        evict a recording that ``cleanup_task`` has not freed)."""
        assert self.server is not None
        printer = self.server._printer

        def other_runs_retire(_task_id: str) -> None:
            for i in range(20):
                key = f"registered-run-{i}"
                agent_state.register(AgentState(key, is_task_active=True))
                printer._thread_local.task_id = key
                printer.start_recording()
                printer.broadcast({"type": "text_delta", "text": "x"})
                printer.stop_recording()
            assert all(
                printer.peek_recording_for_task(f"registered-run-{i}")
                for i in range(20)
            )

        view = await self._finish_after(("", ""), other_runs_retire)
        self.assertEqual(_text(view), "HEAD FINAL", view)
        results = [ev for ev in view if ev.get("type") == "result"]
        self.assertEqual([ev.get("text") for ev in results], ["all done"])
        for i in range(20):
            printer.cleanup_task(f"registered-run-{i}")

    async def _interrupted_reservation_keeps_endpoint(self, func: str) -> None:
        """A ``KeyboardInterrupt`` right after *func* returns in B's replay
        reservation must not leave B's connection blocked forever."""
        assert self.server is not None
        vs = self.server._vscode_server
        task_id, chat_id = await self._running_task_viewed_by_a()
        ws_b, conn_b = await self._connect_b()
        interrupted: list[bool] = []

        def raise_on_return(frame: FrameType, event: str, arg: Any) -> Any:
            if event == "return" and not interrupted:
                interrupted.append(True)
                raise KeyboardInterrupt
            return raise_on_return

        def hook(frame: FrameType, event: str, arg: Any) -> Any:
            if (
                event == "call"
                and not interrupted
                and frame.f_code.co_name == func
                and frame.f_back is not None
                and "replay" in frame.f_back.f_code.co_name
            ):
                return raise_on_return
            return None

        def replay() -> None:
            sys.settrace(hook)
            try:
                vs._replay_session(chat_id, "tab-t", conn_id=conn_b)
            except KeyboardInterrupt:
                pass
            finally:
                sys.settrace(None)

        await asyncio.to_thread(replay)
        self.assertEqual(interrupted, [True], "the trace point was never hit")
        # Raises (sentinel never arrives) when B's FIFO waits forever
        # on an abandoned replay slot.
        await self._drain(ws_b)
        agent_state.agent_states.clear()

    async def test_i1_interrupt_after_partial_reservation(self) -> None:
        """Interrupted after the send was scheduled, inside the reservation."""
        await self._interrupted_reservation_keeps_endpoint("_send_to_conn")

    async def test_i1_interrupt_after_reservation_returns(self) -> None:
        """Interrupted after the reservation returned to replay_snapshot."""
        await self._interrupted_reservation_keeps_endpoint(
            "_reserve_replay_send",
        )

    async def test_i2_no_json_encoding_under_delivery_lock(self) -> None:
        """Task events are encoded before ``delivery_lock`` is taken.

        Covers an unreserved tab-stamped ``task_events`` (a corrective
        replay), a kept and persisted tab-stamped ``prompt``, and a
        persisted ``text_delta`` fanned out to a subscribed tab.  The
        lock serializes every task's deliveries, so encoding a large
        payload under it stalls them all.
        """
        assert self.server is not None
        printer = self.server._printer
        task_id, _chat_id = await self._running_task_viewed_by_a()
        agent = WorktreeSorcarAgent("rb-encode")
        agent._last_task_id = task_id
        state = agent_state.get(task_id)
        assert state is not None
        state.agent = agent
        is_owned = getattr(printer.delivery_lock, "_is_owned")
        under_lock: list[str] = []

        def hook(frame: FrameType, event: str, arg: Any) -> Any:
            if (
                event == "call"
                and frame.f_code.co_name == "dumps"
                and frame.f_globals.get("__name__") == "json"
                # An event, not a per-tab ``json.dumps(tab_id)`` stamp.
                and isinstance(frame.f_locals.get("obj"), dict)
                and is_owned()
            ):
                caller = frame.f_back
                under_lock.append(caller.f_code.co_name if caller else "?")
            return None

        def emit() -> None:
            sys.settrace(hook)
            try:
                printer.broadcast({
                    "type": "task_events", "tabId": "tab-t",
                    "taskId": task_id,
                    "events": [{"type": "text_delta", "text": "x" * 100000}],
                })
                printer.broadcast({
                    "type": "prompt", "text": "injected", "tabId": "tab-t",
                    "taskId": task_id,
                })
                self._emit_delta(task_id, "[D]")
            finally:
                sys.settrace(None)

        await asyncio.to_thread(emit)
        self.assertEqual(under_lock, [], "json.dumps ran under delivery_lock")
        got = await self._wait_for_event(
            self.ws_a, "text_delta", timeout=5,
            pred=lambda ev: ev.get("text") == "[D]"
            and ev.get("tabId") == "tab-t",
        )
        self.assertIsNotNone(got)
        th._flush_chat_events(task_id)
        loaded = th._load_chat_events_by_task_id(task_id)
        assert loaded is not None
        events = loaded["events"]
        assert isinstance(events, list)
        texts = [ev.get("text") for ev in events]
        self.assertIn("injected", texts)
        self.assertIn("[D]", texts)
        self.assertNotIn("tabId", events[-1])
        agent_state.agent_states.clear()


def test_stop_recording_keeps_only_attachable_runs() -> None:
    """``stop_recording`` keeps a recording only while a viewer can
    still attach to its run.

    A server-owned run's state stays registered until the task runner
    cleans it up, so its final events stay readable until
    ``cleanup_task``.  A run the printer bridge already unregistered
    (a sub-agent, which is never cleaned up) cannot be attached to any
    more: its recording must be freed at once, not accumulate.
    """
    printer = JsonPrinter()
    agent_state.agent_states.clear()
    agent_state.register(AgentState("owned"))
    try:
        for key in ("owned", "child"):
            printer._thread_local.task_id = key
            printer.start_recording()
            printer.broadcast({"type": "text_delta", "text": key})
            printer.stop_recording()
        assert [ev.get("text") for ev in printer.peek_recording_for_task(
            "owned",
        )] == ["owned"]
        assert printer.peek_recording_for_task("child") == []
        printer.cleanup_task("owned")
        assert printer.peek_recording_for_task("owned") == []
    finally:
        agent_state.agent_states.clear()


def test_intermediate_subtask_cleaned_up_when_its_persistence_fails() -> None:
    """A failed intermediate-subtask persist still frees its recording.

    ``stop_recording`` keeps a still-registered run's recording until
    ``cleanup_task``.  The runner persists each intermediate subtask of
    a multi-``<task>`` prompt and then cleans it up; the persistence
    failure is logged and the loop moves on to the next subtask, whose
    id replaces the one the end-of-run cleanup frees.  The cleanup must
    therefore run even when the persist fails, or the first subtask's
    recording leaks for the life of the daemon.

    Drives the real ``_run_task`` worker and real SQLite: a trigger
    makes the first subtask's ``result`` UPDATE fail inside SQLite.
    Only the agent's LLM loop (``agent.run``) is substituted; it
    performs the printer bridge calls ``ChatSorcarAgent.run`` makes.
    """
    import os
    import queue
    import sqlite3
    import tempfile

    from kiss.server.web_server import RemoteAccessServer

    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    tmp = tempfile.mkdtemp(prefix="kiss-rb5-subtask-cleanup-")
    remote = RemoteAccessServer(
        use_tunnel=False,
        url_file=os.path.join(tmp, "url.json"),
        uds_path=os.path.join(tmp, "sorcar.sock"),
    )
    vscode = remote._vscode_server
    printer = vscode.printer
    agent_state.agent_states.clear()
    tab_id = "rb5-subtask-cleanup"
    agent = WorktreeSorcarAgent("Sorcar VS Code")
    state = AgentState(
        f"pre-{tab_id}", agent=agent, tab_id=tab_id, server_owned=True,
        stop_event=threading.Event(),
    )
    state.user_answer_queue = queue.Queue()
    agent_state.register(state)
    task_ids: list[str] = []
    db = sqlite3.connect(
        str(th._DB_PATH), timeout=30, check_same_thread=False,
    )

    def fake_run(**kwargs: Any) -> str:
        agent._chat_id = agent._chat_id or f"rb5-chat-{tab_id}"
        task_id, _ = th._add_task(
            kwargs.get("prompt_template", ""), chat_id=agent._chat_id,
        )
        task_ids.append(task_id)
        agent._last_task_id = task_id
        if len(task_ids) == 1:
            db.execute(
                "CREATE TRIGGER rb5_fail_result BEFORE UPDATE OF result "
                f"ON task_history WHEN OLD.id = '{task_id}' "
                "BEGIN SELECT RAISE(ABORT, 'induced persistence failure'); END",
            )
            db.commit()
        printer._thread_local.task_id = task_id
        printer.agent_task_allocated(agent, task_id, agent._chat_id)
        printer.start_recording()
        printer.broadcast({"type": "text_delta", "text": f"sub {task_id}"})
        printer.agent_task_finished(agent, task_id)
        printer.stop_recording()
        printer._thread_local.task_id = ""
        return f"summary: subtask {len(task_ids)} done\nsuccess: true"

    agent.run = fake_run  # type: ignore[method-assign, assignment]
    worker = threading.Thread(
        target=vscode._run_task,
        args=({
            "type": "run",
            "prompt": "<task>first part</task><task>second part</task>",
            "tabId": tab_id,
            "workDir": tempfile.mkdtemp(prefix="kiss-rb5-wd-"),
            "useParallel": False,
            "useWorktree": False,
            "autoCommit": False,
            "_state_key": state.task_id,
        },),
        daemon=True,
    )
    state.task_thread = worker
    try:
        worker.start()
        worker.join(timeout=60)
        assert not worker.is_alive(), "worker never finished"
        assert len(task_ids) == 2, task_ids
        first = db.execute(
            "SELECT result FROM task_history WHERE id = ?", (task_ids[0],),
        ).fetchone()
        assert first is not None and "subtask 1 done" not in str(first[0]), (
            "the induced persistence failure did not happen"
        )
        with printer._lock:
            leaked = [
                tid for tid in task_ids
                if tid in printer._retired_recordings
                or tid in printer._recordings
            ]
        assert leaked == [], f"recordings never cleaned up: {leaked}"
    finally:
        db.execute("DROP TRIGGER IF EXISTS rb5_fail_result")
        db.commit()
        db.close()
        agent_state.agent_states.clear()
