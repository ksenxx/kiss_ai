# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-03 (server-core): history-resume racing the task's end.

``_replay_session`` resolves the chat's live task under
``_state_lock`` (``_reattach_running_chat``), releases the lock,
subscribes the viewer tab to the task's stream, and broadcasts
``status running=true`` to the viewer.  The task can finish inside
that window: the end-of-run fan-out
(``_TaskRunnerMixin._broadcast_status_end_to_viewers``) reads the
printer's subscriber map BEFORE the subscription lands, so nothing
ever sends the viewer tab ``running=false`` — and the replay then
broadcasts ``running=true`` for a task that is already dead.  The
viewer's spinner (and its treatment of follow-up input as
``appendUserMessage`` against a finished task) survives forever.

The window is made deterministic with a real
:class:`~kiss.tests.server._memory_printer.MemoryPrinter` subclass
whose ``subscribe_tab`` parks the replay thread on its way INTO the
subscription — exactly the interleaving in which the task's terminal
fan-out cannot see the viewer yet.  Everything else is real: a real
``VSCodeServer``, a run submitted through the real ``_cmd_run``, a
real worker thread parked in a real agent-script getter, the real
end-of-run broadcasts.  Releasing the getter makes it raise, so the
task ends in setup and no LLM is ever invoked.
"""

from __future__ import annotations

import os
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from typing import Any
from unittest import TestCase

from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.tests.server._memory_printer import MemoryPrinter

_BLOCKING_SCRIPT = textwrap.dedent(
    """
    import pathlib
    import time

    _DIR = pathlib.Path(__file__).resolve().parent


    def get_prompt():
        \"\"\"Block until released, then raise (the task ends in setup).\"\"\"
        (_DIR / "entered").write_text("1", encoding="utf-8")
        deadline = time.time() + 60
        while time.time() < deadline:
            if (_DIR / "release").exists():
                raise RuntimeError("released")
            time.sleep(0.02)
        raise RuntimeError("timed out waiting for the release")
    """
)


class _ParkingPrinter(MemoryPrinter):
    """Real test printer that parks one tab's ``subscribe_tab`` call.

    A second, independent park point holds the replaying thread right
    AFTER it broadcast a ``task_events`` for a designated tab — i.e.
    between the transcript replay and ``_finalize_viewer_attach`` —
    so a task can end (and record its terminal result) inside that
    window.
    """

    def __init__(self) -> None:
        super().__init__()
        self.park_tab = ""
        self.parked = threading.Event()
        self.release = threading.Event()
        self.park_after_events_tab = ""
        self.events_parked = threading.Event()
        self.release_events = threading.Event()

    def subscribe_tab(self, task_id: Any, tab_id: str) -> None:
        """Park the designated tab's FIRST subscription, then delegate.

        Parking BEFORE the delegation keeps the viewer out of the
        subscriber map while it waits — the state the end-of-run
        fan-out observes when the race is lost.
        """
        if tab_id == self.park_tab and not self.parked.is_set():
            self.parked.set()
            if not self.release.wait(timeout=30):
                raise TimeoutError("parked too long in subscribe_tab")
        super().subscribe_tab(task_id, tab_id)

    def broadcast(self, event: dict[str, Any]) -> None:
        """Delegate, then park after the designated ``task_events``."""
        super().broadcast(event)
        if (
            self.park_after_events_tab
            and event.get("type") == "task_events"
            and event.get("tabId") == self.park_after_events_tab
            and not self.events_parked.is_set()
        ):
            self.events_parked.set()
            if not self.release_events.wait(timeout=30):
                raise TimeoutError("parked too long after task_events")


class TestReattachStatusEndRace(TestCase):
    """A viewer attaching as the task ends must not stay running forever."""

    def setUp(self) -> None:
        os.environ.setdefault("KISS_WORKDIR", "/tmp")
        agent_state.agent_states.clear()
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-audit0903-reattach-"))
        self.work_dir = self.tmp / "wd"
        self.work_dir.mkdir()
        self.script = self.tmp / "agent.py"
        self.script.write_text(_BLOCKING_SCRIPT, encoding="utf-8")
        self.printer = _ParkingPrinter()
        self.server = VSCodeServer(printer=self.printer)
        self.server.work_dir = str(self.work_dir)

    def tearDown(self) -> None:
        (self.tmp / "release").write_text("1", encoding="utf-8")
        self.printer.release.set()
        self.printer.release_events.set()
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and any(
            s.task_thread is not None and s.task_thread.is_alive()
            for s in agent_state.agent_states.values()
        ):
            time.sleep(0.05)
        agent_state.agent_states.clear()

    def _statuses(self, tab_id: str) -> list[bool]:
        return [
            bool(ev.get("running"))
            for ev in list(self.printer.emitted)
            if ev.get("type") == "status" and ev.get("tabId") == tab_id
        ]

    def _wait(self, predicate: Any, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def _start_parked_run(self, tab_id: str, chat_id: str) -> None:
        self.server._cmd_run({
            "type": "run",
            "prompt": "reattach race",
            "tabId": tab_id,
            # A client-minted run token, as daemon_client sends: the
            # end-of-run viewer fan-out must echo it on the viewers'
            # ``status`` events.
            "taskId": f"tok-{tab_id}",
            "chatId": chat_id,
            "workDir": str(self.work_dir),
            "useWorktree": False,
            "useParallel": False,
            "autoCommit": False,
            "agentPath": str(self.script),
        })
        self.assertTrue(
            self._wait((self.tmp / "entered").exists, 30.0),
            "the run never reached the agent-script getter",
        )

    def test_viewer_attaching_as_task_ends_receives_running_false(self) -> None:
        launcher, viewer, chat_id = "launcher-tab", "viewer-tab", "chat-race"
        self._start_parked_run(launcher, chat_id)

        # A history click on the running chat, parked on its way into
        # the subscription — after the liveness check, before the
        # subscriber map knows the viewer.
        self.printer.park_tab = viewer
        replayer = threading.Thread(
            target=self.server._replay_session,
            args=(chat_id, viewer),
            daemon=True,
        )
        replayer.start()
        self.assertTrue(
            self.printer.parked.wait(timeout=30),
            "the replay never reached subscribe_tab",
        )

        # The task ends INSIDE the window: its terminal fan-out cannot
        # see the not-yet-registered viewer.
        (self.tmp / "release").write_text("1", encoding="utf-8")
        self.assertTrue(
            self._wait(
                lambda: False in self._statuses(launcher), 30.0,
            ),
            "the run never broadcast its terminal status",
        )

        self.printer.release.set()
        replayer.join(timeout=30)
        self.assertFalse(replayer.is_alive())

        self.assertIn(
            True,
            self._statuses(viewer),
            "the replay never told the viewer the task was running",
        )
        self.assertTrue(
            self._wait(lambda: self._statuses(viewer)[-1] is False, 5.0),
            "BUG: the viewer tab was left with running=true forever — "
            "the task ended between the reattach liveness check and "
            "the subscription, so the end fan-out missed it and "
            f"nothing corrected the replay's status: {self._statuses(viewer)}",
        )

        # The correction must NOT be a bare status boolean: the task's
        # terminal ``result`` (here: the setup failure) must reach the
        # viewer through the replayed/re-snapshot ``task_events``
        # BEFORE the corrective ``running=false`` — the early failure
        # result used to be addressed only to the launcher tab, never
        # recorded, so the viewer got ``[true, false]`` and an empty
        # transcript (review Finding 4).
        viewer_events = [
            ev
            for ev in list(self.printer.emitted)
            if ev.get("tabId") == viewer
        ]
        replayed_results = [
            res
            for ev in viewer_events
            if ev.get("type") == "task_events"
            for res in ev.get("events", [])
            if res.get("type") == "result"
        ]
        self.assertTrue(
            replayed_results,
            "BUG: the attached viewer never received the task's "
            "terminal result — its transcript ended with only status "
            f"booleans: {[e.get('type') for e in viewer_events]}",
        )
        self.assertFalse(replayed_results[-1].get("success"))
        self.assertIn("Task failed", str(replayed_results[-1].get("text", "")))
        last_result_idx = max(
            i
            for i, ev in enumerate(viewer_events)
            if ev.get("type") == "task_events"
            and any(r.get("type") == "result" for r in ev.get("events", []))
        )
        final_false_idx = max(
            i
            for i, ev in enumerate(viewer_events)
            if ev.get("type") == "status" and ev.get("running") is False
        )
        self.assertLess(
            last_result_idx,
            final_false_idx,
            "the terminal result must be delivered BEFORE the "
            "corrective running=false",
        )

    def test_task_ending_after_replay_snapshot_is_resnapshot(self) -> None:
        """The terminal result landing AFTER the transcript replay.

        The viewer subscribes and replays while the task still runs
        (its snapshot has no ``result``); the task then ends — and its
        early setup-failure result is recorded — before
        ``_finalize_viewer_attach`` runs.  The finalize step must
        re-snapshot the recording and deliver a ``task_events`` that
        DOES carry the terminal result, before the corrective
        ``running=false`` (review Finding 4's re-snapshot half).
        """
        launcher, viewer, chat_id = "snap-launcher", "snap-viewer", "chat-snap"
        self._start_parked_run(launcher, chat_id)

        self.printer.park_after_events_tab = viewer
        replayer = threading.Thread(
            target=self.server._replay_session,
            args=(chat_id, viewer),
            daemon=True,
        )
        replayer.start()
        self.assertTrue(
            self.printer.events_parked.wait(timeout=30),
            "the replay never broadcast the viewer's task_events",
        )

        # The task ends INSIDE the replay→finalize window: its result
        # is recorded under the run's provisional id, but the snapshot
        # the viewer just received predates it.
        (self.tmp / "release").write_text("1", encoding="utf-8")
        self.assertTrue(
            self._wait(
                lambda: False in self._statuses(launcher), 30.0,
            ),
            "the run never broadcast its terminal status",
        )

        self.printer.release_events.set()
        replayer.join(timeout=30)
        self.assertFalse(replayer.is_alive())

        viewer_events = [
            ev
            for ev in list(self.printer.emitted)
            if ev.get("tabId") == viewer
        ]
        events_batches = [
            ev for ev in viewer_events if ev.get("type") == "task_events"
        ]
        self.assertGreaterEqual(
            len(events_batches),
            2,
            "the finalize step never re-snapshot the recording after "
            f"the task ended: {[e.get('type') for e in viewer_events]}",
        )
        self.assertFalse(
            any(
                r.get("type") == "result"
                for r in events_batches[0].get("events", [])
            ),
            "precondition broken: the FIRST replay snapshot already "
            "carried the terminal result",
        )
        final_results = [
            r
            for r in events_batches[-1].get("events", [])
            if r.get("type") == "result"
        ]
        self.assertTrue(
            final_results,
            "the re-snapshot task_events carries no terminal result",
        )
        self.assertFalse(final_results[-1].get("success"))
        self.assertTrue(
            self._wait(lambda: self._statuses(viewer)[-1] is False, 5.0),
            f"no corrective running=false after the re-snapshot: "
            f"{self._statuses(viewer)}",
        )

    def test_persisted_chat_replay_attaches_to_live_task(self) -> None:
        """A history click on a chat with rows AND a live task.

        The persisted-transcript replay branch must attach the viewer
        to the chat's live run, broadcast ``running=true``, replay the
        stored transcript, and leave the still-running attach
        untouched (the finalize correction only fires for a source
        that died inside the window); the terminal ``running=false``
        arrives later through the normal end fan-out.
        """
        from uuid import uuid4

        from kiss.agents.sorcar.persistence import _add_task

        launcher, viewer = "hist-launcher", "hist-viewer"
        chat_id = f"chat-hist-{uuid4().hex}"
        _add_task("earlier finished task", chat_id)
        self._start_parked_run(launcher, chat_id)

        self.server._replay_session(chat_id, viewer)
        self.assertEqual(
            self._statuses(viewer), [True],
            "the persisted replay must flip the viewer to running "
            "exactly once while the task lives",
        )
        self.assertTrue(
            any(
                ev.get("type") == "task_events" and ev.get("tabId") == viewer
                for ev in list(self.printer.emitted)
            ),
            "the persisted transcript was never replayed to the viewer",
        )

        (self.tmp / "release").write_text("1", encoding="utf-8")
        self.assertTrue(
            self._wait(
                lambda: bool(self._statuses(viewer))
                and self._statuses(viewer)[-1] is False,
                30.0,
            ),
            "the attached viewer never received the terminal status "
            f"through the end fan-out: {self._statuses(viewer)}",
        )

    def test_replaying_unknown_idle_chat_emits_no_status(self) -> None:
        """No history row and no live task: nothing to attach or flip."""
        from uuid import uuid4

        self.server._replay_session(f"chat-none-{uuid4().hex}", "lonely-tab")
        self.assertEqual(self._statuses("lonely-tab"), [])

    def test_viewer_attaching_to_live_task_is_not_flipped_back(self) -> None:
        """The still-running re-check must not undo a valid attach.

        Covers the fix's still-running branch: a viewer attaching to a
        task that keeps running gets ``running=true`` and NO immediate
        ``running=false``; the terminal ``running=false`` arrives only
        with the task's real end, through the normal fan-out.
        """
        launcher, viewer, chat_id = "live-launcher", "live-viewer", "chat-live"
        self._start_parked_run(launcher, chat_id)

        self.server._replay_session(chat_id, viewer)
        statuses = self._statuses(viewer)
        self.assertEqual(
            statuses, [True],
            f"a live attach must broadcast exactly running=true: {statuses}",
        )

        (self.tmp / "release").write_text("1", encoding="utf-8")
        self.assertTrue(
            self._wait(
                lambda: self._statuses(viewer)
                and self._statuses(viewer)[-1] is False,
                30.0,
            ),
            "the subscribed viewer never received the terminal status "
            f"through the end fan-out: {self._statuses(viewer)}",
        )
