# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a replay snapshot and a live event never interleave.

The chat page REPLACES a tab's transcript when a ``task_events`` replay
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
    copied);
3.  a late charge of a finished task shown before the replay's place
    in the send queues was reserved, and not in the transcript the
    replay read (read before the charge, or before the charge wrote
    its event; or what the run recorded, which no late charge is in),
    was wiped by the replay.

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
from kiss.agents.sorcar import daemon_client, local_endpoint
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent, _retire_printer_task
from kiss.core.locked import locked_call
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import (
    _ending_with_row_totals,
    _extra_dict,
    _replay_events,
    _stored_replay_events,
)
from kiss.server.task_runner import _rows_of_run, _run_row_id
from kiss.server.task_update import charge_side_channel_usage
from kiss.tests.server.pause import Pause, Pauses
from kiss.tests.server.test_web_server_tab_mirroring import (
    TabMirroringBase,
)

_SENTINEL = "audit0926_replay_boundary_sentinel"

_FAILING_SCRIPT = textwrap.dedent(
    """
    import pathlib
    import time

    from kiss.agents.seas.base.base_sea import BaseSea

    _DIR = pathlib.Path(__file__).resolve().parent


    class Sea(BaseSea):
        def settings(self, settings):
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


def _shown_totals(messages: list[dict[str, Any]], tab_id: str) -> list[tuple[float, int, int]]:
    """The cost, tokens and steps *tab_id* shows after each of *messages* that sets them.

    In the order received.  A replay sets them to those of its last
    usage event (zeros when it has none), a live usage event to its own.
    """
    shown: list[tuple[float, int, int]] = []
    for ev in messages:
        if ev.get("tabId") != tab_id:
            continue
        if ev.get("type") == "task_events":
            usage = [e for e in ev.get("events") or [] if e.get("type") == "usage_info"]
            last = usage[-1] if usage else {}
        elif ev.get("type") == "usage_info":
            last = ev
        else:
            continue
        cost = float(last.get("cost", "$0").lstrip("$"))
        shown.append((cost, last.get("total_tokens", 0), last.get("total_steps", 0)))
    return shown


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

    async def _running_task_viewed_by_a(self, lazy: bool = False) -> tuple[str, str]:
        """Register a running task shown on tab-t by client A.

        With *lazy* A's page loads transcripts in pieces: the replay
        another page asks for in pieces is one frame for both, and is
        sent whole unless A's page takes pieces too.
        """
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
        await self._send(ws_a, {"type": "ready", "tabId": "", "restoredTabs": [], "lazy": lazy})
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
        vs = self.server._agent_server
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
        vs = self.server._agent_server
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
            # The snapshot is coalesced for the payload after it is
            # taken (and, since the fix, after ``delivery_lock`` is
            # released).
            sys.settrace(
                _hook("_coalesced_replay_events", "call", emit_after_snapshot),
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
        vs = self.server._agent_server
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
                "isParallel": False,
                "autoCommit": False,
                "seaPath": str(script),
            })
        finally:
            threading.settrace(None)  # type: ignore[arg-type]
        state = agent_state.find_by_tab(launcher)
        assert state is not None and state.task_thread is not None
        task_thread = state.task_thread
        # A scoped replay refreshes a tab some surface shows: the viewer
        # is the chat's registry tab (opening it displaced the launcher).
        vs._registry_update_tab(viewer, chat_id=chat_id, create=True)

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
        agent = ChatSorcarAgent("rb-persist")
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
        vs = self.server._agent_server
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
        vs = self.server._agent_server
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
            sys.settrace(_hook("with_synthesized_events", "call", park))
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
        vs = self.server._agent_server
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
        # The attach returns inside the replay's view lock section;
        # ``_race_delay`` is its first statement after the release.
        view = await self._finish_after(("_race_delay", "call"))
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

    async def test_i3_result_broadcast_after_stop_recording_before_attach(
        self,
    ) -> None:
        """A result broadcast after ``stop_recording`` joins the retired recording.

        The task runner broadcasts the ``result`` of a run that failed
        before its first step after the agent already stopped recording
        (the state still counts as running).  B attaches after that
        broadcast: its replay must carry the late result, or a surface
        that resumes the task's tab a few milliseconds after it ended
        never shows why it failed.
        """
        assert self.server is not None
        printer = self.server._printer
        vs = self.server._agent_server
        task_id, chat_id = await self._running_task_viewed_by_a()
        await asyncio.to_thread(self._emit_delta, task_id, "HEAD ")
        ws_b, conn_b = await self._connect_b()

        def stop_then_fail() -> None:
            printer._thread_local.task_id = task_id
            printer.stop_recording()
            printer.broadcast(
                {"type": "result", "text": "Task failed with error: no dir",
                 "success": False, "taskId": task_id},
            )

        await asyncio.to_thread(stop_then_fail)
        await asyncio.to_thread(
            vs._replay_session, chat_id, "tab-t", conn_id=conn_b,
        )
        view = _rendered(await self._drain(ws_b), "tab-t")
        agent_state.agent_states.clear()
        results = [ev for ev in view if ev.get("type") == "result"]
        self.assertEqual(
            [ev.get("text") for ev in results], ["Task failed with error: no dir"], view,
        )

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
            ("_race_delay", "call"), other_runs_stop,
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
            ("_race_delay", "call"), end_and_clean_up,
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
        agent = ChatSorcarAgent("rb-sequential")

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
            ("_race_delay", "call"), next_subtask,
        )
        self.assertEqual(_text(view), "HEAD FINAL", view)
        results = [ev for ev in view if ev.get("type") == "result"]
        self.assertEqual([ev.get("text") for ev in results], ["all done"])

    async def test_run_given_its_row_before_the_attach_is_replayed_under_it(self) -> None:
        """A run gets its history row between B's row read and its attach.

        The chat's latest row, read first, is the finished task before
        the run.  The replay names the run's own row: sent without it,
        the page keeps the finished task's id and reads the thread back
        from there, past that task.
        """
        assert self.server is not None
        vs = self.server._agent_server
        _, chat_id = th._add_task("finished task")
        ws_a = await self._connect_ok()
        await self._ready(ws_a)
        await self._send(ws_a, {"type": "openTab", "tabId": "tab-t", "title": "new chat"})
        await self._send(ws_a, {"type": "resumeSession", "chatId": chat_id, "tabId": "tab-t"})
        self.assertIsNotNone(await self._wait_for_event(
            ws_a, "task_events", pred=lambda ev: ev.get("tabId") == "tab-t",
        ))
        # What a run in tab-t registers on admission (``_cmd_run``).
        run = AgentState("provisional-key", chat_id=chat_id, tab_id="tab-t", is_task_active=True)
        agent_state.register(run)
        ws_b, conn_b = await self._connect_b()
        row: list[str] = []

        def allocate_row() -> None:
            task_id, _ = th._add_task("the run's task", chat_id=chat_id)
            agent_state.rekey(run, str(task_id))
            row.append(str(task_id))

        def replay() -> None:
            sys.settrace(_hook("_attach_viewer_to_running_chat", "call", allocate_row))
            try:
                vs._replay_session(chat_id, "tab-t", conn_id=conn_b)
            finally:
                sys.settrace(None)

        await asyncio.to_thread(replay)
        replays = [ev for ev in await self._drain(ws_b) if ev.get("type") == "task_events"]
        agent_state.agent_states.clear()
        self.assertEqual([(str(ev.get("task_id")), ev.get("task")) for ev in replays],
                         [(row[0], "the run's task")])

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
        vs = self.server._agent_server
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
        agent = ChatSorcarAgent("rb-encode")
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

    def _assert_cost_rises_to(
        self, messages: list[dict[str, Any]], tab_id: str, final: float, lazy: bool = False,
    ) -> None:
        """*tab_id* never shows a total below one it showed before, and ends on the cost *final*.

        Every transcript it was sent came in pieces if *lazy*, else whole.
        """
        shown = _shown_totals(messages, tab_id)
        for totals in zip(*shown):  # its costs, its tokens, its steps
            self.assertEqual(list(totals), sorted(totals), messages)
        self.assertEqual([cost for cost, _, _ in shown[-1:]], [final], messages)
        in_pieces = {
            any(e.get("type") == "lazy_steps" for e in m["events"])
            for m in messages
            if m.get("type") == "task_events" and m.get("tabId") == tab_id
        }
        self.assertEqual(in_pieces, {lazy}, messages)

    async def _through(
        self, ws: ClientConnection, wanted: str, timeout: float = 10.0,
    ) -> list[dict[str, Any]]:
        """Every message *ws* receives through the first *wanted* one, or within *timeout*."""
        out: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout
        while (left := deadline - time.monotonic()) > 0:
            try:
                out.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=left)))
            except TimeoutError:
                break
            if out[-1].get("type") == wanted:
                break
        return out

    async def _finished_task_on_tab_c(
        self, lazy: bool,
    ) -> tuple[str, dict[str, Any], ClientConnection]:
        """A finished task that cost $1.00, shown on tab-c of a client as on the tab that ran it.

        Returns:
            The task, the ``resumeSession`` that replays it into the tab
            (in pieces with *lazy*), and the client.
        """
        assert self.server is not None
        task_id, chat_id = th._add_task(
            "finished task", "",
            {"model": "finished-task-model", "cost": 1.0, "tokens": 100, "steps": 2,
             "endTs": int(time.time() * 1000)},
        )
        th._append_chat_event(th.usage_totals_event(task_id, 100, 1.0, 2), task_id=task_id)
        th._append_chat_event({"type": "result", "text": "done"}, task_id=task_id)
        ws = await self._connect_ok()
        await self._send(ws, {"type": "ready", "tabId": "", "restoredTabs": [], "lazy": lazy})
        await self._send(ws, {"type": "openTab", "tabId": "tab-c", "title": "new chat"})
        self.assertIsNotNone(await self._wait_for_snapshot_with(ws, present={"tab-c"}))
        # A task's tab stays subscribed to it after its end: its late charges reach the tab.
        self.server._printer.subscribe_tab(task_id, "tab-c")
        resume = {
            "type": "resumeSession", "chatId": chat_id, "tabId": "tab-c", "taskId": task_id,
            "replayId": "late-charge", "lazy": lazy,
        }
        return task_id, resume, ws

    def _late_charge(
        self, task_id: str, tab_id: str = "tab-c", times: int = 1, cost: float = 0.25,
    ) -> None:
        """Charge the finished task *cost*, 40 tokens and a step more, as a side channel of it
        does after its end."""
        assert self.server is not None
        for _ in range(times):
            charge_side_channel_usage(
                self.server._printer, None, task_id, cost, 40, 1, tab_id=tab_id,
            )

    def _run_broadcasts(self, task_id: str, *events: dict[str, Any]) -> None:
        """Broadcast *events* as the run of *task_id* does."""
        assert self.server is not None
        printer = self.server._printer
        printer._thread_local.task_id = task_id
        for event in events:
            printer.broadcast(event)
        printer._thread_local.task_id = None

    def _end_run(self, task_id: str, spent: float) -> None:
        """End the run of *task_id* as its worker does: its result, then its row with *spent*."""
        assert self.server is not None
        printer = self.server._printer
        printer._thread_local.task_id = task_id
        printer.broadcast({"type": "result", "text": "all done"})
        printer.stop_recording()
        printer._thread_local.task_id = None
        th._save_task_extra(
            {"cost": spent, "tokens": int(100 * spent), "steps": 2,
             "endTs": int(time.time() * 1000)},
            task_id=task_id,
        )
        state = agent_state.get(task_id)
        assert state is not None
        state.is_task_active = False

    async def _replayed_after(self, change: Callable[[str], Any]) -> list[dict[str, Any]]:
        """What tab-c gets when *change* lands between a whole replay's row read and its frame."""
        task_id, resume, ws = await self._finished_task_on_tab_c(lazy=False)
        # ``_extra_dict``: the replay's first step once it has read the row.
        with Pause(
            _extra_dict.__code__, lambda v: "finished-task-model" in str(v["extra"]),
        ) as hold:
            await self._send(ws, resume)
            self.assertTrue(await asyncio.to_thread(hold.paused.wait, 10))
            await asyncio.to_thread(change, task_id)
        return await self._through(ws, "replayDone") + await self._drain(ws)

    async def test_3_late_charge_after_the_row_was_read_is_in_the_replay(self) -> None:
        """A charge between a whole replay's reading of the row and its frame is not wiped.

        The tab is sent the charge's totals, then the replay's
        transcript, which replaces what it shows: read with the row,
        before the charge, it left the tab on the older cost.
        """
        messages = await self._replayed_after(self._late_charge)
        self._assert_cost_rises_to(messages, "tab-c", 1.25)

    async def test_3_task_deleted_after_the_row_was_read_has_no_transcript(self) -> None:
        """A task deleted in that window is replayed without the transcript it had."""
        messages = await self._replayed_after(th._delete_task)
        frames = [m["events"] for m in messages if m.get("type") == "task_events"]
        self.assertEqual(len(frames), 1, messages)
        self.assertNotIn("done", [ev.get("text") for ev in frames[0]])

    async def _charge_in_flight_is_replayed(self, lazy: bool) -> None:
        """A charge that has sent its totals and not yet written them is not wiped by a replay.

        The charge is held between the two; the replay of the task
        asked meanwhile runs as far as it can.  Its transcript replaces
        the totals the tab was sent, so it must hold them.
        """
        task_id, resume, ws = await self._finished_task_on_tab_c(lazy)
        charge = threading.Thread(target=self._late_charge, args=(task_id,), daemon=True)
        with Pause(
            th._append_chat_event.__code__,
            lambda v: v["task_id"] == task_id and v["event"].get("type") == "usage_info",
        ) as hold:
            charge.start()
            self.assertTrue(await asyncio.to_thread(hold.paused.wait, 10))
            await self._send(ws, resume)
            # Arrives unless the replay waits for the charge.
            messages = await self._through(ws, "replayDone", timeout=1.5)
        await asyncio.to_thread(charge.join, 10)
        self.assertFalse(charge.is_alive())
        if not any(m.get("type") == "replayDone" for m in messages):
            messages += await self._through(ws, "replayDone")
        messages += await self._drain(ws)
        self._assert_cost_rises_to(messages, "tab-c", 1.25, lazy)

    async def test_3_charge_in_flight_is_in_a_whole_replay(self) -> None:
        """The transcript sent whole waits for the charge's event."""
        await self._charge_in_flight_is_replayed(lazy=False)

    async def test_3_charge_in_flight_is_in_a_replay_in_pieces(self) -> None:
        """So does the first piece of a transcript loaded in pieces."""
        await self._charge_in_flight_is_replayed(lazy=True)

    async def _ended_run_charge_is_replayed(
        self, held_in: str, lazy: bool, spent: float = 1.0, charge: float = 0.25,
    ) -> None:
        """A run that ends and is charged while its replay is held in *held_in* shows the charge.

        The replay attached to the run while it ran: its transcript is
        what the run recorded (its own totals, if it *spent* anything),
        which no late charge is in, and the tab, subscribed by the
        attach, was sent the charge's totals first.  The charge costs
        *charge* and adds tokens and a step: those of a free model
        ($0) are shown, and replayed, like any other's.
        """
        assert self.server is not None
        task_id, chat_id = await self._running_task_viewed_by_a(lazy)
        ws_b, _ = await self._connect_b()

        def end_and_charge() -> None:
            self._end_run(task_id, spent)
            self._late_charge(task_id, "tab-t", cost=charge)

        if spent:
            await asyncio.to_thread(
                self._run_broadcasts, task_id, th.usage_totals_event(task_id, 100, spent, 2),
            )
        with Pause(
            getattr(type(self.server._agent_server), held_in).__code__,
            lambda v: v["tab_id"] == "tab-t" and v["source"].task_id == task_id,
        ) as hold:
            await self._send(ws_b, {
                "type": "resumeSession", "chatId": chat_id, "tabId": "tab-t", "taskId": task_id,
                "replayId": "ended-run", "lazy": lazy,
            })
            self.assertTrue(await asyncio.to_thread(hold.paused.wait, 10))
            await asyncio.to_thread(end_and_charge)
        messages = await self._through(ws_b, "replayDone") + await self._drain(ws_b)
        self._assert_cost_rises_to(messages, "tab-t", spent + charge, lazy)
        self.assertEqual(
            _shown_totals(messages, "tab-t")[-1], (spent + charge, int(100 * spent) + 40, 3),
        )

    async def test_3_run_ended_and_charged_before_its_frame(self) -> None:
        """Between the attach and the frame: the frame holds the row's totals."""
        await self._ended_run_charge_is_replayed("_broadcast_viewer_running", lazy=False)

    async def test_3_run_without_spend_ended_and_charged_before_its_frame(self) -> None:
        """So does the frame of a run that recorded no totals of its own."""
        await self._ended_run_charge_is_replayed(
            "_broadcast_viewer_running", lazy=False, spent=0.0,
        )

    async def test_3_run_ended_and_charged_before_its_frame_in_pieces(self) -> None:
        """So does the outline of a transcript loaded in pieces."""
        await self._ended_run_charge_is_replayed("_broadcast_viewer_running", lazy=True)

    async def test_3_run_ended_and_charged_before_its_terminal_frame(self) -> None:
        """After a frame without the result: the transcript sent again with it holds them."""
        await self._ended_run_charge_is_replayed("_finalize_viewer_attach", lazy=False)

    async def test_3_run_ended_and_charged_before_its_terminal_frame_in_pieces(self) -> None:
        """So does that transcript's outline."""
        await self._ended_run_charge_is_replayed("_finalize_viewer_attach", lazy=True)

    async def test_3_run_ended_and_charged_for_free_before_its_frame(self) -> None:
        """The frame holds the row's tokens and steps when the charge cost nothing."""
        await self._ended_run_charge_is_replayed(
            "_broadcast_viewer_running", lazy=False, charge=0.0,
        )

    async def test_3_run_ended_and_charged_for_free_before_its_frame_in_pieces(self) -> None:
        """So does the outline of a transcript loaded in pieces."""
        await self._ended_run_charge_is_replayed("_broadcast_viewer_running", lazy=True, charge=0.0)

    async def test_3_run_ended_and_charged_for_free_before_its_terminal_frame(self) -> None:
        """So does the transcript sent again with the result."""
        await self._ended_run_charge_is_replayed("_finalize_viewer_attach", lazy=False, charge=0.0)

    async def test_3_run_ended_and_charged_for_free_before_its_terminal_frame_in_pieces(
        self,
    ) -> None:
        """So does that transcript's outline."""
        await self._ended_run_charge_is_replayed("_finalize_viewer_attach", lazy=True, charge=0.0)

    async def _run_without_a_row_on_tab_pre(
        self, lazy: bool,
    ) -> tuple[ChatSorcarAgent, AgentState, ClientConnection, dict[str, Any]]:
        """A run that has no row yet, on tab-pre of a client.

        Returns:
            Its agent and state, the client, and the ``resumeSession``
            that names the run by its provisional key.
        """
        agent = ChatSorcarAgent("rb-pre-row")
        agent._tab_id = "tab-pre"
        state = AgentState(
            "pre-row-run", agent=agent, chat_id="pre-row-chat", tab_id="tab-pre",
            is_task_active=True,
        )
        agent_state.register(state)
        ws = await self._connect_ok()
        await self._send(ws, {"type": "ready", "tabId": "", "restoredTabs": [], "lazy": lazy})
        await self._send(ws, {"type": "openTab", "tabId": "tab-pre", "title": "new chat"})
        self.assertIsNotNone(await self._wait_for_snapshot_with(ws, present={"tab-pre"}))
        resume = {
            "type": "resumeSession", "chatId": "pre-row-chat", "tabId": "tab-pre",
            "taskId": "pre-row-run", "replayId": "pre-row", "lazy": lazy,
        }
        return agent, state, ws, resume

    def _given_its_row(self, agent: ChatSorcarAgent, cost: float) -> str:
        """Give the run of *agent* its row, as ``ChatSorcarAgent.run`` does; it then spends *cost*.

        The runner subscribes the launching tab to the row.
        """
        assert self.server is not None
        printer = self.server._printer
        task_id, _ = th._add_task("pre-row task", "pre-row-chat")
        agent._last_task_id = task_id
        printer._thread_local.task_id = task_id
        printer.agent_task_allocated(agent, task_id, "pre-row-chat")
        printer.start_recording()
        printer.subscribe_tab(task_id, "tab-pre")
        self._run_broadcasts(
            task_id,
            {"type": "task_settings", "settings": {"task_id": task_id, "model": "m"}},
            {"type": "text_delta", "text": "[X]"},
            th.usage_totals_event(task_id, int(100 * cost), cost, 2),
        )
        return task_id

    async def _run_given_its_row_ended_and_charged(
        self, held_in: str, lazy: bool, cleaned: bool = False, at_attach: bool = False,
    ) -> None:
        """A run attached to before it had a row shows the late charge of the row it got since.

        The replay names the run by its provisional key, attaches to it
        and is held in *held_in*.  The run then gets its row, records
        its totals, ends and is charged late: the tab was sent all of
        it, and the replay's transcript, what the run recorded under
        its row (what the row has stored, once the worker has *cleaned*
        the recording up), must end with that row's totals.  Read under
        the key of the attach, there was no row and the transcript put
        the tab back on the totals the run ended with.

        With *at_attach* the replay is a chat's and names no task, and
        the run gets its row as the replay attaches, which subscribes
        the tab under the row: read for the task the replay names,
        there was no row either.
        """
        assert self.server is not None
        printer = self.server._printer
        vs = type(self.server._agent_server)
        agent, state, ws, resume = await self._run_without_a_row_on_tab_pre(lazy)
        rows: list[str] = []

        def given_its_row(attach: dict[str, Any]) -> bool:
            if at_attach and attach["new_tab_id"] == "tab-pre" and not rows:
                rows.append(self._given_its_row(agent, 1.0))
            return False  # a step at the attach's entry: nothing is held there

        def end_and_charge() -> None:
            task_id = rows[0] if rows else self._given_its_row(agent, 1.0)
            self._end_run(task_id, 1.0)
            if cleaned:
                printer.cleanup_task(task_id)
            with agent_state.STATE_LOCK:
                state.task_thread = None  # the allocation named its thread the run's
            self._late_charge(task_id, "tab-pre")

        if at_attach:
            del resume["taskId"]
        hold = Pause(getattr(vs, held_in).__code__, lambda v: v["tab_id"] == "tab-pre")
        with Pauses(Pause(vs._attach_viewer_to_running_chat.__code__, given_its_row), hold):
            await self._send(ws, resume)
            self.assertTrue(await asyncio.to_thread(hold.paused.wait, 10))
            await asyncio.to_thread(end_and_charge)
        messages = await self._through(ws, "replayDone") + await self._drain(ws)
        self._assert_cost_rises_to(messages, "tab-pre", 1.25, lazy)
        self.assertEqual(_shown_totals(messages, "tab-pre")[-1], (1.25, 140, 3))

    async def test_3_run_given_its_row_ended_and_charged_before_its_frame(self) -> None:
        """The frame of a run named by its provisional key holds the totals of the run's row."""
        await self._run_given_its_row_ended_and_charged("_broadcast_viewer_running", lazy=False)

    async def test_3_run_given_its_row_ended_and_charged_before_its_frame_in_pieces(self) -> None:
        """So does the outline of a transcript loaded in pieces."""
        await self._run_given_its_row_ended_and_charged("_broadcast_viewer_running", lazy=True)

    async def test_3_run_given_its_row_ended_and_charged_before_its_terminal_frame(self) -> None:
        """So does the transcript sent again with the result."""
        await self._run_given_its_row_ended_and_charged("_finalize_viewer_attach", lazy=False)

    async def test_3_run_given_its_row_ended_and_charged_before_its_terminal_frame_in_pieces(
        self,
    ) -> None:
        """So does that transcript's outline."""
        await self._run_given_its_row_ended_and_charged("_finalize_viewer_attach", lazy=True)

    async def test_3_run_given_its_row_ended_cleaned_up_and_charged_before_its_frame(self) -> None:
        """So does the frame of what the row has stored, once the recording is freed."""
        await self._run_given_its_row_ended_and_charged(
            "_broadcast_viewer_running", lazy=False, cleaned=True,
        )

    async def test_3_run_given_its_row_ended_cleaned_up_and_charged_before_its_frame_in_pieces(
        self,
    ) -> None:
        """So does the outline of that stored transcript."""
        await self._run_given_its_row_ended_and_charged(
            "_broadcast_viewer_running", lazy=True, cleaned=True,
        )

    async def test_3_run_given_its_row_at_the_attach_ended_and_charged_before_its_frame(
        self,
    ) -> None:
        """So does the frame of a chat's replay, whose run got its row as the replay attached."""
        await self._run_given_its_row_ended_and_charged(
            "_broadcast_viewer_running", lazy=False, at_attach=True,
        )

    async def test_4_run_given_its_row_while_its_replay_reads_is_shown_once(self) -> None:
        """A run that gets its row while its replay reads for its old key is replayed as the row.

        The replay of a run without a row reads what the run's key has
        stored, where it is held while the run gets its row and records
        its first events, which the tab is sent.  The replay then reads
        again, for the row, and its transcript, sent behind those
        events, is what the run has recorded under the row: the
        recording of the key first read for is empty, and would leave
        the tab without the events and their totals.
        """
        agent, _, ws, resume = await self._run_without_a_row_on_tab_pre(lazy=False)
        with Pause(
            _stored_replay_events.__code__,
            lambda v: v["session"].get("task_id") == "pre-row-run",
        ) as hold:
            await self._send(ws, resume)
            self.assertTrue(await asyncio.to_thread(hold.paused.wait, 10))
            await asyncio.to_thread(self._given_its_row, agent, 0.5)
        messages = await self._through(ws, "replayDone") + await self._drain(ws)
        self._assert_cost_rises_to(messages, "tab-pre", 0.5)
        shown = [
            [ev.get("text") for ev in m.get("events") or [m] if ev.get("type") == "text_delta"]
            for m in messages
            if m.get("tabId") == "tab-pre" and m.get("type") in ("text_delta", "task_events")
        ]
        self.assertEqual(shown, [["[X]"], ["[X]"]], messages)  # sent live, then in the transcript
        agent_state.agent_states.clear()

    async def _run_recorded_and_retired_behind_its_stored_read(self, lazy: bool) -> None:
        """What a run records, then frees, behind its replay's stored read is in the replay.

        The replay attaches to a sub-agent's run that has recorded
        nothing, reads what the row has stored and is held before it
        takes its place in the send queues.  The run then records its
        totals and its result, which the tab is sent, and retires: its
        recording is freed, so the place finds nothing to copy.  The
        transcript read before all that, sent in its stead, put the tab
        back on no totals at all.
        """
        assert self.server is not None
        printer = self.server._printer
        task_id, chat_id = th._add_task("retiring task")
        agent = ChatSorcarAgent("rb-retiring")
        agent._last_task_id = task_id

        def record_and_retire() -> None:
            # What ``ChatSorcarAgent.run`` does, to its end, for a run the server did not start.
            printer._thread_local.task_id = task_id
            printer.start_recording()
            printer.broadcast(th.usage_totals_event(task_id, 100, 1.0, 2))
            printer.broadcast({"type": "result", "text": "retired"})
            _retire_printer_task(agent, printer, task_id)
            printer._thread_local.task_id = None

        # The run's start: the printer registers it under its row.
        await asyncio.to_thread(printer.agent_task_allocated, agent, task_id, chat_id)
        ws = await self._connect_ok()
        await self._send(ws, {"type": "ready", "tabId": "", "restoredTabs": [], "lazy": lazy})
        await self._send(ws, {"type": "openTab", "tabId": "tab-r", "title": "new chat"})
        self.assertIsNotNone(await self._wait_for_snapshot_with(ws, present={"tab-r"}))
        with Pause(
            JsonPrinter.replay_snapshot.__wrapped__.__code__,  # type: ignore[attr-defined]
            lambda v: v["task_id"] == task_id and v["tab_id"] == "tab-r",
        ) as hold:
            await self._send(ws, {
                "type": "resumeSession", "chatId": chat_id, "tabId": "tab-r", "taskId": task_id,
                "replayId": "retiring", "lazy": lazy,
            })
            self.assertTrue(await asyncio.to_thread(hold.paused.wait, 10))
            await asyncio.to_thread(record_and_retire)
        messages = await self._through(ws, "replayDone") + await self._drain(ws)
        self._assert_cost_rises_to(messages, "tab-r", 1.0, lazy)
        self.assertEqual(_shown_totals(messages, "tab-r")[-1], (1.0, 100, 2))
        frames = [m["events"] for m in messages if m.get("type") == "task_events"]
        self.assertTrue(any(ev.get("type") == "result" for ev in frames[0]), messages)

    async def test_4_run_recorded_and_retired_behind_its_stored_read_is_shown(self) -> None:
        """The frame holds what the run stored between the replay's reading and its place."""
        await self._run_recorded_and_retired_behind_its_stored_read(lazy=False)

    async def test_4_run_recorded_and_retired_behind_its_stored_read_is_shown_in_pieces(
        self,
    ) -> None:
        """So does the outline of a transcript loaded in pieces."""
        await self._run_recorded_and_retired_behind_its_stored_read(lazy=True)

    async def test_4_what_a_finished_run_sent_last_is_in_its_replay_before_it_is_written(
        self,
    ) -> None:
        """A replay of a finished task waits for the events still queued for its transcript.

        A run's events are written to its transcript by a background
        thread.  The tab was sent the last of them, and the replay's
        transcript, read while the writer was behind, replaced it with
        a transcript that lacked it.
        """
        task_id, resume, ws = await self._finished_task_on_tab_c(lazy=False)
        agent = ChatSorcarAgent("rb-finished")
        agent._last_task_id = task_id
        # The tab's state once its run is over: what is still broadcast is filed under the row.
        agent_state.register(
            AgentState(task_id, agent=agent, chat_id=resume["chatId"], tab_id="tab-c"),
        )
        with Pause(
            th._persist_batch_with_retry.__code__,
            lambda v: any(item[0] == task_id for item in v["batch"]),
        ) as hold:
            await asyncio.to_thread(
                self._run_broadcasts, task_id, {"type": "text_delta", "text": "[tail]"},
            )
            self.assertTrue(await asyncio.to_thread(hold.paused.wait, 10))
            await self._send(ws, resume)
            messages = await self._through(ws, "replayDone", timeout=1.0)
        if not any(m.get("type") == "replayDone" for m in messages):
            messages += await self._through(ws, "replayDone")  # the replay waited for the writer
        frames = [m["events"] for m in messages if m.get("type") == "task_events"]
        self.assertEqual(len(frames), 1, messages)
        self.assertIn("[tail]", _text(frames[0]))
        agent_state.agent_states.clear()

    async def test_4_recovering_client_is_not_put_back_by_the_row_of_its_running_block(
        self,
    ) -> None:
        """A recovering client gets a running block from its recording, not from its row.

        The client's replay is held as it turns to the run's row, with
        the tab attached, while the run sends totals the events table
        has not been written yet.  The row's frame, sent behind them,
        put the client back on the totals the table had.
        """
        assert self.server is not None
        task_id, chat_id = await self._running_task_viewed_by_a(lazy=False)
        agent = ChatSorcarAgent("rb-recovering")
        agent._last_task_id = task_id
        with agent_state.STATE_LOCK:
            agent_state.agent_states[task_id].agent = agent  # its events are filed under the row
        stored = th.usage_totals_event(task_id, 50, 0.5, 1)
        await asyncio.to_thread(self._run_broadcasts, task_id, stored)
        await asyncio.to_thread(th._flush_chat_events, task_id)
        ws_b, _ = await self._connect_b()
        row_spend = type(self.server._agent_server)._replay_row_spend.__code__
        writer = Pause(
            th._persist_batch_with_retry.__code__,
            lambda v: any(item[0] == task_id for item in v["batch"]),
        )
        hold = Pause(row_spend, lambda v: v["tab_id"] == "tab-t")
        with Pauses(writer, hold):
            await self._send(ws_b, {
                "type": "resumeSession", "chatId": chat_id, "tabId": "tab-t", "taskId": task_id,
                "recovery": True, "scoped": True, "replayId": "recovering", "lazy": False,
            })
            self.assertTrue(await asyncio.to_thread(hold.paused.wait, 10))
            await asyncio.to_thread(
                self._run_broadcasts, task_id, th.usage_totals_event(task_id, 100, 1.0, 2),
            )
            self.assertTrue(await asyncio.to_thread(writer.paused.wait, 10))
            hold.release()  # the replay goes on while the writer is behind
            messages = await self._through(ws_b, "replayDone")
        self._assert_cost_rises_to(messages, "tab-t", 1.0)
        self.assertEqual(_shown_totals(messages, "tab-t")[-1], (1.0, 100, 2))

    async def _block_begun_since_the_attach_is_kept(self, cleaned: bool) -> None:
        """A recovering client is not put back by the rows of a run that began its next block.

        Block 2's row is committed, and the run still has block 1's
        key, when the client's replay lists the run's rows and attaches
        the tab.  Block 2 then begins and sends its totals and its
        result.  The replay sent block 2's row as the events table had
        it when read, behind newer totals of the block (the client was
        put back on the older ones), and block 1's frames last, which
        made block 1 the client's task again and dropped block 2's
        result.  (Review L6j2 round 4; the client is the real
        ``daemon_client`` one.)

        Args:
            cleaned: Block 1's recording is freed when its row is
                finished, as the runner frees it; else it is kept, and
                is what the replay attached under block 1's key sends.
        """
        assert self.server is not None
        agent, state, _, _ = await self._run_without_a_row_on_tab_pre(lazy=False)
        token = state.task_id
        state.client_run_token = token
        state.server_owned = True
        chat_id, tab_id = state.chat_id, state.tab_id
        first, second = (_run_row_id(token, block) for block in (0, 1))
        printer, server = self.server._printer, self.server._agent_server
        began, sent, go, over = (threading.Event() for _ in range(4))
        errors: list[str] = []

        def begin(row: str, previous: str | None = None) -> None:
            """What ``ChatSorcarAgent.run`` and the runner do as a block gets its row."""
            th._add_task("a block", chat_id, task_id=row)
            agent._last_task_id = row
            printer._thread_local.task_id = row
            printer.agent_task_allocated(agent, row, chat_id)
            printer.start_recording()
            server._on_run_task_id_allocated(
                row, chat_id, source_tab_id=tab_id, start_ms=int(time.time() * 1000),
                client_task_id=token, previous_task_id=previous,
            )
            printer.broadcast({"type": "task_settings", "settings": {"task_id": row, "model": "m"}})

        def run_blocks() -> None:
            try:
                begin(first)
                printer.broadcast(th.usage_totals_event(first, 25, 0.25, 1))
                printer.broadcast({"type": "result", "text": "block 1 done"})
                _retire_printer_task(agent, printer, first)
                th._append_chat_event({"type": "task_done", "run_continues": True}, task_id=first)
                locked_call(
                    th.TASK_USAGE_LOCK, th._save_task_extra,
                    {"cost": 0.25, "tokens": 25, "steps": 1, "endTs": int(time.time() * 1000)},
                    task_id=first,
                )
                if cleaned:
                    printer.cleanup_task(first)
                begin(second, first)
                printer.broadcast(th.usage_totals_event(second, 50, 0.5, 1))
                th._flush_chat_events(second)
                began.set()
                assert go.wait(15)
                printer.broadcast(th.usage_totals_event(second, 100, 1.0, 2))
                printer.broadcast({"type": "result", "text": "block 2 done"})
                th._flush_chat_events(second)
                sent.set()
                assert over.wait(15)
            except BaseException as exc:
                errors.append(repr(exc))
                began.set()
                sent.set()
            finally:
                printer._thread_local.task_id = None

        def send_more(_entered: dict[str, Any]) -> bool:
            """Block 2 sends its last totals and its result while the entering thread waits."""
            go.set()
            sent.wait(10)
            return False

        keyed = Pause(JsonPrinter.agent_task_allocated.__code__, lambda v: v["task_id"] == second)
        attached = Pause(
            type(server)._broadcast_viewer_running.__code__, lambda v: v["tab_id"] == tab_id,
        )
        # Block 2 sends them where a replay has read block 2's row and not
        # sent it yet, or else ahead of the frame of the block it attached to.
        row_read = Pause(
            _replay_events.__code__,
            lambda v: v["task_id"] == "" and v["session"].get("task_id") == second and send_more(v),
        )
        cut = Pause(type(server)._enter_replay_cut.__code__, send_more)
        endpoint = Path(self.tmpdir) / "sorcar-local.json"
        sock = await asyncio.to_thread(local_endpoint.connect, endpoint)
        inbox = daemon_client._Inbox(sock, endpoint, tab_id)
        seen = daemon_client._Observed(chat_id=chat_id, task_id=first, run_token=token)
        charged: dict[str, dict[str, Any]] = {}
        shown: list[tuple[float, int, int]] = []  # block 2's totals, as the client has them
        worker = threading.Thread(target=run_blocks, daemon=True)
        try:
            with Pauses(keyed, attached, row_read, cut):
                worker.start()
                self.assertTrue(await asyncio.to_thread(keyed.paused.wait, 10), errors)
                self.assertEqual(_rows_of_run(th._chat_task_tree(chat_id), first), [first, second])
                await asyncio.to_thread(sock.close)
                await asyncio.to_thread(inbox.reconnect, 10.0, seen)
                self.assertTrue(await asyncio.to_thread(attached.paused.wait, 10))
                self.assertEqual(state.task_id, first)  # attached under block 1's key
                keyed.release()
                self.assertTrue(await asyncio.to_thread(began.wait, 10))
                self.assertEqual(errors, [])
                attached.release()
                event: dict[str, Any] = {}
                while event.get("type") != "replayDone":
                    received = await asyncio.to_thread(inbox.recv_event, 10.0)
                    event = received or {}
                    if event.get("tabId") != tab_id:
                        continue
                    if event.get("type") == "task_events":
                        inbox.absorb_replay(event, seen, charged)
                    else:
                        seen.absorb(event, charged)
                    if second in seen.totals:
                        spent = seen.totals[second]
                        shown.append((spent["cost"], spent["total_tokens"], spent["step_count"]))
        finally:
            go.set()
            over.set()
            await asyncio.to_thread(worker.join, 10)
            await asyncio.to_thread(inbox.ws.close)
            state.task_thread = None
            agent_state.agent_states.clear()
        self.assertEqual(errors, [])
        self.assertEqual(_rows_of_run(th._chat_task_tree(chat_id), first), [first, second])
        self.assertEqual(shown, sorted(shown), "block 2's totals went back")
        self.assertEqual(shown[-1], (1.0, 100, 2))
        self.assertEqual(seen.task_id, second)
        self.assertEqual((seen.result or {}).get("text"), "block 2 done")

    async def test_4_recovering_client_keeps_what_a_block_begun_since_its_attach_sent(
        self,
    ) -> None:
        """The next block's row is not replayed behind what that block sent since the attach."""
        await self._block_begun_since_the_attach_is_kept(cleaned=True)

    async def test_4_recovering_client_keeps_it_after_the_recording_of_the_block_before(
        self,
    ) -> None:
        """Nor is the recording of the block before, where the replay attached under its key."""
        await self._block_begun_since_the_attach_is_kept(cleaned=False)

    async def _charged_twice_behind_its_frame(self, lazy: bool) -> None:
        """Charges made once a replay's place in the send queues is reserved follow its frame.

        The replay attached to the run, which then ended, and reserved
        its frame's place.  A first charge sends its totals behind that
        place; a second has updated the row and not sent its own yet.
        A frame that took the row's totals of that moment showed the
        second charge, then the tab was put back on the first one's.
        """
        assert self.server is not None
        server = type(self.server._agent_server)
        task_id, chat_id = await self._running_task_viewed_by_a(lazy)
        ws_b, _ = await self._connect_b()
        await asyncio.to_thread(
            self._run_broadcasts, task_id, th.usage_totals_event(task_id, 100, 1.0, 2),
        )
        attached = Pause(
            server._broadcast_viewer_running.__code__,
            lambda v: v["tab_id"] == "tab-t" and v["source"].task_id == task_id,
        )
        # The frame's last step before it is sent, its place long reserved.
        reserved = Pause(_ending_with_row_totals.__code__, lambda v: True)
        sending = Pause(
            JsonPrinter.broadcast_transient.__code__,
            lambda v: v["event"].get("cost") == "$1.5000",
        )
        second = threading.Thread(target=self._late_charge, args=(task_id, "tab-t"), daemon=True)
        with Pauses(attached, reserved, sending):
            await self._send(ws_b, {
                "type": "resumeSession", "chatId": chat_id, "tabId": "tab-t", "taskId": task_id,
                "replayId": "charged-twice", "lazy": lazy,
            })
            self.assertTrue(await asyncio.to_thread(attached.paused.wait, 10))
            await asyncio.to_thread(self._end_run, task_id, 1.0)
            attached.release()
            self.assertTrue(await asyncio.to_thread(reserved.paused.wait, 10))
            await asyncio.to_thread(self._late_charge, task_id, "tab-t")
            second.start()
            self.assertTrue(await asyncio.to_thread(sending.paused.wait, 10))
            reserved.release()
            messages = await self._through(ws_b, "replayDone")
        await asyncio.to_thread(second.join, 10)
        self.assertFalse(second.is_alive())
        messages += await self._drain(ws_b)
        self._assert_cost_rises_to(messages, "tab-t", 1.5, lazy)
        agent_state.agent_states.clear()

    async def test_4_charges_behind_the_frame_of_an_ended_run_follow_it(self) -> None:
        """The frame of what the run recorded ends with the row's totals of its reservation."""
        await self._charged_twice_behind_its_frame(lazy=False)

    async def test_4_charges_behind_the_frame_of_an_ended_run_follow_it_in_pieces(self) -> None:
        """So does the outline of a transcript loaded in pieces."""
        await self._charged_twice_behind_its_frame(lazy=True)

    async def _cleaned_up_run_charged_behind_its_frame(self, lazy: bool) -> None:
        """Charges behind the frame of a run its worker has cleaned up follow that frame.

        The worker freed the run's recording and has not left yet: the
        replay attaches to it, and its frame is the stored transcript.
        Read once the frame's place was reserved, the transcript held
        the two charges made meanwhile, whose own totals then put the
        tab back on the first one's.  The replay is held where it reads
        the transcript; the charges are over within the wait unless
        they wait for the replay.
        """
        task_id, resume, ws = await self._finished_task_on_tab_c(lazy)
        agent_state.register(AgentState(
            task_id, chat_id=resume["chatId"], tab_id="tab-c", is_task_active=True,
        ))
        charges = threading.Thread(
            target=self._late_charge, args=(task_id, "tab-c", 2), daemon=True,
        )
        reads = _replay_events if lazy else th._load_chat_events_by_task_id
        with Pause(
            reads.__code__, lambda v: v["task_id"] == task_id and v.get("with_events", True),
        ) as hold:
            await self._send(ws, resume)
            self.assertTrue(await asyncio.to_thread(hold.paused.wait, 10))
            charges.start()
            await asyncio.to_thread(charges.join, 1.5)
        await asyncio.to_thread(charges.join, 10)
        self.assertFalse(charges.is_alive())
        messages = await self._through(ws, "replayDone") + await self._drain(ws)
        self._assert_cost_rises_to(messages, "tab-c", 1.5, lazy)
        agent_state.agent_states.clear()

    async def test_4_charges_behind_the_frame_of_a_cleaned_up_run_follow_it(self) -> None:
        """The stored transcript an attached replay sends is read before its reservation."""
        await self._cleaned_up_run_charged_behind_its_frame(lazy=False)

    async def test_4_charges_behind_the_frame_of_a_cleaned_up_run_follow_it_in_pieces(self) -> None:
        """So is the outline of a transcript loaded in pieces."""
        await self._cleaned_up_run_charged_behind_its_frame(lazy=True)

    async def _first_events_are_shown_once(self, lazy: bool) -> None:
        """What a run records while its replay reads the stored transcript is shown once.

        The run had recorded nothing when the replay attached to it, so
        the replay reads the stored transcript, where it is held while
        the run records its first events and they are stored.  Read
        once the frame's place was reserved, the transcript held those
        events, which the tab then got again behind the frame: its
        text twice, its older totals after its newer ones.
        """
        task_id, chat_id = await self._running_task_viewed_by_a(lazy)
        agent = ChatSorcarAgent("rb-first-events")
        agent._last_task_id = task_id
        state = agent_state.get(task_id)
        assert state is not None
        state.agent = agent  # the run's events are stored
        ws_b, _ = await self._connect_b()
        reads = _replay_events if lazy else th._load_chat_events_by_task_id
        with Pause(
            reads.__code__, lambda v: v["task_id"] == task_id and v.get("with_events", True),
        ) as hold:
            await self._send(ws_b, {
                "type": "resumeSession", "chatId": chat_id, "tabId": "tab-t", "taskId": task_id,
                "replayId": "first-events", "lazy": lazy,
            })
            self.assertTrue(await asyncio.to_thread(hold.paused.wait, 10))
            await asyncio.to_thread(
                self._run_broadcasts, task_id,
                {"type": "text_delta", "text": "[X]"},
                th.usage_totals_event(task_id, 50, 0.5, 1),
                th.usage_totals_event(task_id, 100, 1.0, 2),
            )
            await asyncio.to_thread(th._flush_chat_events, task_id)
            stored = await asyncio.to_thread(th._load_chat_events_by_task_id, task_id)
            assert stored is not None
            self.assertIn("[X]", str(stored["events"]))
        messages = await self._through(ws_b, "replayDone") + await self._drain(ws_b)
        self.assertEqual(_text(_rendered(messages, "tab-t")).count("[X]"), 1, messages)
        self._assert_cost_rises_to(messages, "tab-t", 1.0, lazy)
        agent_state.agent_states.clear()

    async def test_4_first_events_of_a_run_are_shown_once(self) -> None:
        """The stored transcript holds nothing the run recorded behind the frame."""
        await self._first_events_are_shown_once(lazy=False)

    async def test_4_first_events_of_a_run_are_shown_once_in_pieces(self) -> None:
        """Nor does the outline of a transcript loaded in pieces."""
        await self._first_events_are_shown_once(lazy=True)


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
        local_endpoint_file=os.path.join(tmp, "sorcar-local.json"),
    )
    agent_server = remote._agent_server
    printer = agent_server.printer
    agent_state.agent_states.clear()
    tab_id = "rb5-subtask-cleanup"
    agent = ChatSorcarAgent("Sorcar")
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
        target=agent_server._run_task,
        args=({
            "type": "run",
            "prompt": "<task>first part</task><task>second part</task>",
            "tabId": tab_id,
            "workDir": tempfile.mkdtemp(prefix="kiss-rb5-wd-"),
            "isParallel": False,
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
