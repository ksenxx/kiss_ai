# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-03 (server, fix round): stale replay status vs a newer run.

``_replay_session`` attaches a viewer tab to a chat's live task and
broadcasts ``status running=true`` (with the OLD task's ``startTs``),
then — when the source died inside the attach window — an unqualified
``status running=false``, both stamped with the viewer's tab id
(:meth:`VSCodeServer._broadcast_viewer_running`).  The helper used to
re-check only the old *source* object; it never asked whether the
viewer tab meanwhile started a NEWER run of its own.  A replay delayed
in transport delivery therefore could:

* overwrite the newer run's timer with the old task's ``startTs``, and
* kill the newer run's spinner/Stop button with ``running=false``

— both actionable at the frontend, which does not generation-qualify
status events (review Finding 3).

The scenario is real end to end: the old task is launched through the
real ``_cmd_run`` (worker parked in a real agent-script getter), the
viewer attaches through the real ``_replay_session``, the newer run is
launched on the viewer tab through the real ``_cmd_run``, and the old
task really ends.  The delayed replay's status helper is then invoked
exactly as ``_replay_session`` would have — after the newer run owns
the tab — and must emit NOTHING for the viewer tab.
"""

from __future__ import annotations

import os
import tempfile
import textwrap
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
        (_DIR / "entered-{marker}").write_text("1", encoding="utf-8")
        deadline = time.time() + 60
        while time.time() < deadline:
            if (_DIR / "release-{marker}").exists():
                raise RuntimeError("released")
            time.sleep(0.02)
        raise RuntimeError("timed out waiting for the release")
    """
)


class TestViewerStatusNewerRun(TestCase):
    """A stale replay status must not clobber the viewer's newer run."""

    def setUp(self) -> None:
        os.environ.setdefault("KISS_WORKDIR", "/tmp")
        agent_state.agent_states.clear()
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-audit0903-newer-run-"))
        self.work_dir = self.tmp / "wd"
        self.work_dir.mkdir()
        self.printer = MemoryPrinter()
        self.server = VSCodeServer(printer=self.printer)
        self.server.work_dir = str(self.work_dir)

    def tearDown(self) -> None:
        for marker in ("old", "new"):
            (self.tmp / f"release-{marker}").write_text("1", encoding="utf-8")
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and any(
            s.task_thread is not None and s.task_thread.is_alive()
            for s in agent_state.agent_states.values()
        ):
            time.sleep(0.05)
        agent_state.agent_states.clear()

    def _statuses(self, tab_id: str) -> list[dict[str, Any]]:
        return [
            ev
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

    def _start_parked_run(self, tab_id: str, chat_id: str, marker: str) -> None:
        script = self.tmp / f"agent-{marker}.py"
        script.write_text(
            _BLOCKING_SCRIPT.format(marker=marker), encoding="utf-8",
        )
        self.server._cmd_run({
            "type": "run",
            "prompt": f"newer-run race {marker}",
            "tabId": tab_id,
            "chatId": chat_id,
            "workDir": str(self.work_dir),
            "useWorktree": False,
            "useParallel": False,
            "autoCommit": False,
            "agentPath": str(script),
        })
        self.assertTrue(
            self._wait((self.tmp / f"entered-{marker}").exists, 30.0),
            f"run {marker} never reached the agent-script getter",
        )

    def test_stale_replay_status_is_suppressed_for_newer_run(self) -> None:
        launcher, viewer = "old-launcher", "viewer-tab"
        self._start_parked_run(launcher, "chat-old", "old")
        with self.server._state_lock:
            old_state = agent_state.find_by_tab(launcher)
        self.assertIsNotNone(old_state)
        assert old_state is not None

        # The viewer attaches to the old task (real history resume).
        self.server._replay_session("chat-old", viewer)
        self.assertTrue(
            [e for e in self._statuses(viewer) if e.get("running")],
            "the attach never told the viewer the old task was running",
        )

        # A NEWER run starts on the very same viewer tab.
        self._start_parked_run(viewer, "chat-new", "new")

        # The old task ends while the viewer owns the newer run.
        (self.tmp / "release-old").write_text("1", encoding="utf-8")
        self.assertTrue(
            self._wait(
                lambda: any(
                    e.get("running") is False
                    for e in self._statuses(launcher)
                ),
                30.0,
            ),
            "the old run never broadcast its terminal status",
        )
        # The end fan-out must not have flipped the busy viewer either
        # (its guard predates this fix and must keep holding).
        self.assertFalse(
            any(e.get("running") is False for e in self._statuses(viewer)),
            "the old task's end fan-out killed the newer run's spinner",
        )

        # The delayed replay's status helper fires only NOW — exactly
        # the reviewer's interleaving.  It must emit nothing: neither
        # the optimistic running=true with the OLD startTs nor the
        # unqualified running=false correction.
        before = self._statuses(viewer)
        self.server._broadcast_viewer_running(viewer, old_state, 1)
        after = self._statuses(viewer)
        self.assertEqual(
            after,
            before,
            "BUG: a stale replay status reached a viewer tab that owns "
            f"a newer busy run: {after[len(before):]}",
        )

        # The delayed replay's finalize step must be suppressed the
        # same way: neither a terminal snapshot for the OLD task nor a
        # running=false may reach the busy viewer.
        before_all = len(self.printer.emitted)
        self.server._finalize_viewer_attach(
            viewer,
            old_state,
            [],
            {"type": "task_events", "task": "", "task_id": None,
             "chat_id": "chat-old", "extra": "", "tabId": viewer},
        )
        self.assertEqual(
            list(self.printer.emitted)[before_all:],
            [],
            "the finalize correction leaked onto a viewer tab that "
            "owns a newer busy run",
        )

        # The newer run keeps running untouched and ends normally.
        (self.tmp / "release-new").write_text("1", encoding="utf-8")
        self.assertTrue(
            self._wait(
                lambda: any(
                    e.get("running") is False
                    for e in self._statuses(viewer)
                ),
                30.0,
            ),
            "the newer run never broadcast its own terminal status",
        )

    def test_live_attach_still_broadcasts_running_true(self) -> None:
        """The busy-viewer guard must not suppress a NORMAL attach.

        An idle viewer attaching to a live task keeps receiving the
        optimistic ``running=true`` — the guard only bites when the
        viewer owns a DIFFERENT busy run.
        """
        launcher, viewer = "live-launcher", "idle-viewer"
        self._start_parked_run(launcher, "chat-old", "old")
        with self.server._state_lock:
            old_state = agent_state.find_by_tab(launcher)
        assert old_state is not None
        self.server._broadcast_viewer_running(viewer, old_state, 7)
        running = [e for e in self._statuses(viewer) if e.get("running")]
        self.assertEqual(len(running), 1)
        self.assertEqual(running[0].get("startTs"), 7)
        self.assertFalse(
            any(e.get("running") is False for e in self._statuses(viewer)),
            "a live attach must not be corrected back to running=false",
        )

        # Once subscribed, the idle viewer receives the terminal
        # status through the normal end fan-out (this run carries no
        # client run token, so the viewer's status has no taskId).
        self.printer.subscribe_tab(old_state.task_id, viewer)
        (self.tmp / "release-old").write_text("1", encoding="utf-8")
        self.assertTrue(
            self._wait(
                lambda: any(
                    e.get("running") is False
                    for e in self._statuses(viewer)
                ),
                30.0,
            ),
            "the subscribed idle viewer never received running=false",
        )
        final = [
            e for e in self._statuses(viewer) if e.get("running") is False
        ][-1]
        self.assertNotIn("taskId", final)

    def test_finalize_without_recorded_result_sends_status_only(self) -> None:
        """A dead source with no recorded result still gets the status.

        When the recording holds no terminal ``result`` either (e.g. a
        normally-finished task whose recording was already cleaned
        up), the finalize step cannot re-snapshot anything — it must
        still deliver the corrective ``running=false`` and nothing
        else.
        """
        launcher, viewer = "old-launcher", "idle-viewer"
        self._start_parked_run(launcher, "chat-old", "old")
        with self.server._state_lock:
            old_state = agent_state.find_by_tab(launcher)
        assert old_state is not None
        (self.tmp / "release-old").write_text("1", encoding="utf-8")
        self.assertTrue(
            self._wait(
                lambda: any(
                    e.get("running") is False
                    for e in self._statuses(launcher)
                ),
                30.0,
            ),
            "the old run never ended",
        )
        # Drop the recorded setup result so the peek finds nothing —
        # the state a cleaned-up finished task leaves behind.
        self.printer.cleanup_task(old_state.task_id)
        # An empty task id is a no-op for the recording helper (the
        # setup-failure path guards on a truthy id).
        self.printer.ensure_recording_for_task("")
        with self.printer._lock:
            self.assertNotIn("", self.printer._recordings)
        before = len(self.printer.emitted)
        self.server._finalize_viewer_attach(
            viewer,
            old_state,
            [],
            {"type": "task_events", "task": "", "task_id": None,
             "chat_id": "chat-old", "extra": "", "tabId": viewer},
        )
        emitted = list(self.printer.emitted)[before:]
        self.assertEqual(
            [(e.get("type"), e.get("running")) for e in emitted],
            [("status", False)],
            f"expected exactly one corrective running=false: {emitted}",
        )
