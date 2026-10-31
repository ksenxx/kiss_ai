# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Two replay paths of ``AgentServer`` that bypassed their own rules.

1. A replay of a chat with no history row and nothing live attached
   broadcast ``status running:false`` inline, without asking the tab's
   owner and outside ``_state_lock``.  A run the user started on the
   tab while the replay sat between its attach and that broadcast (or
   the tab's own live run of another chat) then got a stale
   ``running:false`` after its ``running:true``: the surface lost the
   spinner and Stop button of a live task.  The branch now settles the
   flag through ``_broadcast_replay_status``, like every other replay
   with nothing attached.

2. When the source of a viewer attach died between the replay snapshot
   and ``_finalize_viewer_attach``, the corrective terminal snapshot
   re-sent the live recording as is: no per-delta clip, no 16 MB frame
   budget (a client drops its socket above 64 MB) and no row-derived
   events, all serialized while holding ``_state_lock``.  It is now
   built like the replay frame, outside the lock, and sent with the
   correction in one critical section that re-checks the source.

Everything is real: a real :class:`AgentServer`, real registered
:class:`AgentState` objects (the only way to hold a tab busy without an
LLM), a real printer recording, real history rows in a scratch
database.  The replay thread is parked at named calls with
``sys.settrace`` (no production code is replaced), so every
interleaving is deterministic.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import unittest
from types import FrameType
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import (
    MAX_REPLAY_EVENT_TEXT_CHARS,
    MAX_REPLAY_FRAME_BYTES,
    AgentServer,
)
from kiss.tests.server._memory_printer import MemoryPrinter
from kiss.tests.server.conftest import (
    clear_agent_states,
    redirect_persistence,
    restore_persistence,
)

_TAB = "sweep-tab"
_VIEWER = "sweep-viewer"
_LAUNCHER = "sweep-launcher"
_DELTA_TYPES = ("thinking_delta", "text_delta", "system_output")


class _ReplayParker:
    """Runs ``_replay_session`` in a thread parked at the first call of named functions."""

    def __init__(self, server: AgentServer, *names: str) -> None:
        self.server = server
        self.gates = {name: (threading.Event(), threading.Event()) for name in names}
        self.thread: threading.Thread | None = None

    def _trace(self, frame: FrameType, event: str, arg: Any) -> None:
        """Park the calling thread when it enters a gated function the first time."""
        gate = self.gates.get(frame.f_code.co_name)
        if gate is not None and not gate[0].is_set():
            gate[0].set()
            if not gate[1].wait(timeout=30):
                raise TimeoutError(f"parked too long in {frame.f_code.co_name}")

    def _replay(self, chat_id: str, tab_id: str, conn_id: str) -> None:
        sys.settrace(self._trace)
        try:
            self.server._replay_session(chat_id, tab_id, conn_id=conn_id)
        finally:
            sys.settrace(None)

    def start(self, chat_id: str, tab_id: str, conn_id: str = "") -> None:
        """Start the replay of *chat_id* into *tab_id*."""
        self.thread = threading.Thread(
            target=self._replay, args=(chat_id, tab_id, conn_id), daemon=True,
        )
        self.thread.start()

    def parked_at(self, name: str) -> bool:
        """Wait until the replay sits at the entry of *name*."""
        return self.gates[name][0].wait(timeout=30)

    def release(self, name: str) -> None:
        """Let the replay run the body of *name*."""
        self.gates[name][1].set()

    def release_all(self) -> None:
        """Release every gate (test cleanup)."""
        for _, release in self.gates.values():
            release.set()

    def finish(self) -> bool:
        """Release every gate and wait for the replay; True when it ended."""
        self.release_all()
        if self.thread is None:
            return True
        self.thread.join(timeout=30)
        return not self.thread.is_alive()


def _live_state(task_id: str, chat_id: str, tab_id: str) -> AgentState:
    """Register a running server-owned state, as ``_cmd_run`` installs one."""
    state = AgentState(task_id, chat_id=chat_id, tab_id=tab_id, server_owned=True)
    state.is_task_active = True
    agent_state.register(state)
    return state


class _ServerCase(unittest.TestCase):
    """A real server over a scratch history database."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-sweep1030-replay-")
        self.saved = redirect_persistence(self.tmpdir)
        clear_agent_states()
        self.printer = MemoryPrinter()
        self.server = AgentServer(printer=self.printer)
        self.parkers: list[_ReplayParker] = []

    def tearDown(self) -> None:
        for parker in self.parkers:
            parker.finish()
        clear_agent_states()
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _parker(self, *names: str) -> _ReplayParker:
        parker = _ReplayParker(self.server, *names)
        self.parkers.append(parker)
        return parker

    def _frames(self, tab_id: str, *types: str) -> list[dict[str, Any]]:
        return [
            ev
            for ev in list(self.printer.emitted)
            if ev.get("tabId") == tab_id and ev.get("type") in types
        ]


class TestNoRowReplayStatusAsksTheTabOwner(_ServerCase):
    """A row-less, unattached replay never tells a busy tab ``running:false``."""

    def test_run_started_inside_the_attach_window_keeps_its_spinner(self) -> None:
        """A run installed on the tab after the attach found nothing.

        A reconnecting surface replays its tab's chat (no row yet,
        nothing live); the user presses Run on the same tab on another
        surface while the replay sits right after its attach.  The
        replay's status must re-assert the live run, scoped to the
        reconnecting connection, never ``running:false``.
        """
        chat_id = "chat-sweep-norow"
        parker = self._parker("_race_delay")
        parker.start(chat_id, _TAB, "conn-sweep")
        self.assertTrue(parker.parked_at("_race_delay"))
        _live_state("sweep-live-task", chat_id, _TAB)
        self.assertTrue(parker.finish())

        statuses = self._frames(_TAB, "status")
        self.assertEqual(
            [e["running"] for e in statuses],
            [True],
            f"a stale running:false would kill the live run's spinner: {statuses}",
        )
        self.assertIs(statuses[0].get("attached"), True, "a catch-up, not a run start")
        self.assertEqual(statuses[0].get("connId"), "conn-sweep", "the replay's scope is kept")
        self.assertEqual(self._frames(_TAB, "task_events"), [], "no transcript without a row")

    def test_tabs_live_run_of_another_chat_keeps_its_spinner(self) -> None:
        """The tab's own run is bound to another chat than the one replayed."""
        _live_state("sweep-other-task", "chat-sweep-other", _TAB)
        self.server._replay_session("chat-sweep-unknown", _TAB)
        self.assertEqual([e["running"] for e in self._frames(_TAB, "status")], [True])

    def test_idle_tab_is_still_told_it_is_idle(self) -> None:
        """Nothing owned and nothing live: the page still learns the tab is idle."""
        self.server._replay_session("chat-sweep-idle", _TAB, conn_id="conn-idle")
        statuses = self._frames(_TAB, "status")
        self.assertEqual([e["running"] for e in statuses], [False])
        self.assertEqual(statuses[0].get("connId"), "conn-idle")


class TestTerminalSnapshotIsBuiltLikeTheReplay(_ServerCase):
    """The corrective snapshot of a source that died after the replay."""

    def setUp(self) -> None:
        super().setUp()
        self.task_id, self.chat_id = th._add_task("a long task", "chat-sweep-long")
        self.source = _live_state(self.task_id, self.chat_id, _LAUNCHER)
        # The test thread plays the run: its events are recorded under
        # the task id like a worker's.
        self.printer.adopt_thread_binding(self.task_id)
        self.printer.start_recording()
        self.printer.broadcast({"type": "text_delta", "text": "working"})

    def _end_run(self, with_result: bool = True) -> None:
        """End the run the way the task runner does: result first, then inactive."""
        if with_result:
            self.printer.broadcast({"type": "result", "text": "done", "success": True})
        with agent_state.STATE_LOCK:
            self.source.is_task_active = False

    def _viewer_sequence(self) -> list[tuple[Any, Any]]:
        return [
            (ev["type"], ev.get("running"))
            for ev in self._frames(_VIEWER, "task_events", "status")
        ]

    def test_snapshot_after_the_source_died_is_clipped_budgeted_and_synthesized(
        self,
    ) -> None:
        # A transcript beyond the frame budget whose newest delta is
        # beyond the per-event clip.
        chunk = "o" * (1 << 20)
        for _ in range(MAX_REPLAY_FRAME_BYTES // len(chunk) + 1):
            self.printer.broadcast({"type": "tool_result", "content": chunk})
        huge = "x" * (MAX_REPLAY_EVENT_TEXT_CHARS + 100_000)
        self.printer.broadcast({"type": "text_delta", "text": huge})

        parker = self._parker("_finalize_viewer_attach")
        parker.start(self.chat_id, _VIEWER)
        self.assertTrue(parker.parked_at("_finalize_viewer_attach"))
        self._end_run()
        self.assertTrue(parker.finish())

        frames = self._frames(_VIEWER, "task_events")
        self.assertEqual(len(frames), 2, "expected the replay and one terminal snapshot")
        replayed, terminal = frames[0]["events"], frames[1]["events"]
        self.assertFalse(
            any(ev.get("type") == "result" for ev in replayed),
            "precondition: the replay predates the run's end",
        )
        self.assertTrue(any(ev.get("type") == "result" for ev in terminal))
        for ev in terminal:
            if ev.get("type") in _DELTA_TYPES:
                self.assertLessEqual(
                    len(ev["text"]), MAX_REPLAY_EVENT_TEXT_CHARS + 100,
                    "a delta beyond the per-event clip reached the client",
                )
        size = sum(len(json.dumps(ev)) for ev in terminal)
        self.assertLessEqual(
            size, MAX_REPLAY_FRAME_BYTES + 64 * 1024,
            f"the terminal snapshot ({size} bytes) ignores the replay frame budget",
        )
        self.assertTrue(
            any("replay limit" in str(ev.get("text", "")) for ev in terminal),
            "the omitted transcript head is not announced",
        )
        for label, events in (("replay", replayed), ("terminal snapshot", terminal)):
            settings = [ev for ev in events if ev.get("type") == "task_settings"]
            self.assertEqual(
                [ev.get("taskId") for ev in settings], [self.task_id],
                f"the {label} lacks the row's task_settings",
            )
        self.assertEqual(
            self._viewer_sequence(),
            [("status", True), ("task_events", None), ("task_events", None), ("status", False)],
            "the terminal result must precede the corrective running:false",
        )

    def test_source_dead_without_a_result_gets_the_status_only(self) -> None:
        """Nothing terminal was recorded: no second transcript, just the correction."""
        parker = self._parker("_finalize_viewer_attach")
        parker.start(self.chat_id, _VIEWER)
        self.assertTrue(parker.parked_at("_finalize_viewer_attach"))
        self._end_run(with_result=False)
        self.assertTrue(parker.finish())
        self.assertEqual(
            self._viewer_sequence(),
            [("status", True), ("task_events", None), ("status", False)],
        )

    def test_viewer_owning_a_newer_run_gets_no_correction(self) -> None:
        """The viewer started its own run before the finalize step looked."""
        parker = self._parker("_finalize_viewer_attach")
        parker.start(self.chat_id, _VIEWER)
        self.assertTrue(parker.parked_at("_finalize_viewer_attach"))
        self._end_run()
        _live_state("sweep-viewer-run", "chat-sweep-viewer", _VIEWER)
        self.assertTrue(parker.finish())
        self.assertEqual(
            self._viewer_sequence(),
            [("status", True), ("task_events", None)],
            "the newer run's own broadcasts are authoritative for the tab",
        )

    def test_viewer_run_started_while_the_snapshot_is_built_gets_no_correction(
        self,
    ) -> None:
        """The snapshot is built outside the lock; the send re-checks the tab.

        The run ends before the finalize step, which therefore builds a
        terminal snapshot; the user starts a run on the viewer tab while
        the snapshot is being read.  The read must not hold
        ``_state_lock`` (every lock taker would wait for a long
        transcript), and neither the stale snapshot nor a
        ``running:false`` may reach the tab afterwards.
        """
        parker = self._parker("_finalize_viewer_attach", "peek_recording_for_task")
        parker.start(self.chat_id, _VIEWER)
        self.assertTrue(parker.parked_at("_finalize_viewer_attach"))
        self._end_run()
        parker.release("_finalize_viewer_attach")
        self.assertTrue(parker.parked_at("peek_recording_for_task"))
        self.assertTrue(
            agent_state.STATE_LOCK.acquire(timeout=5),
            "the terminal snapshot is read while holding _state_lock",
        )
        agent_state.STATE_LOCK.release()
        _live_state("sweep-viewer-run", "chat-sweep-viewer", _VIEWER)
        self.assertTrue(parker.finish())
        self.assertEqual(
            self._viewer_sequence(),
            [("status", True), ("task_events", None)],
            "a correction decided before the viewer's run started was sent after it",
        )


if __name__ == "__main__":
    unittest.main()
