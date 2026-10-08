# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-26 (setupfail): a setup failure's result reaches viewers.

A run that fails during setup (before ``ChatSorcarAgent.run`` allocates
a task id) broadcasts its terminal ``result`` from the run wrapper's
setup-failure handler.  That event used to be stamped with the
launching tab's ``tabId`` only, so a viewer tab that attached to the
running chat during setup (``_replay_session`` subscribes it to the
run's provisional task id) received ``running=true`` and later
``running=false`` through the per-subscriber fan-out, but never the
live failure result: it appeared only on a later replay of the
recording.  The normal end-of-run result is fanned out to every
subscribed tab, so the setup-failure result must be too — delivered
live exactly once per watching tab, and recorded exactly once.

Everything is real: a real ``VSCodeServer``, a run submitted through
``_cmd_run`` whose SEA ``settings()`` parks on a file
and then raises (a real setup failure, no LLM call), and a real
history-click attach through ``_replay_session``.
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

    from kiss.agents.seas.base.base_sea import BaseSea

    _DIR = pathlib.Path(__file__).resolve().parent


    class Sea(BaseSea):
        def settings(self, settings):
            \"\"\"Block until released, then raise (the task ends in setup).\"\"\"
            (_DIR / "entered").write_text("1", encoding="utf-8")
            deadline = time.time() + 60
            while time.time() < deadline:
                if (_DIR / "release").exists():
                    raise RuntimeError("setup exploded")
                time.sleep(0.02)
            raise RuntimeError("timed out waiting for the release")
    """
)


class TestSetupFailureResultReachesViewer(TestCase):
    """A viewer attached during setup gets the live failure result once."""

    def setUp(self) -> None:
        os.environ.setdefault("KISS_WORKDIR", "/tmp")
        agent_state.agent_states.clear()
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-audit0926-setupfail-"))
        self.work_dir = self.tmp / "wd"
        self.work_dir.mkdir()
        self.script = self.tmp / "agent.py"
        self.script.write_text(_BLOCKING_SCRIPT, encoding="utf-8")
        self.printer = MemoryPrinter()
        self.server = VSCodeServer(printer=self.printer)
        self.server.work_dir = str(self.work_dir)

    def tearDown(self) -> None:
        (self.tmp / "release").write_text("1", encoding="utf-8")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and any(
            s.task_thread is not None and s.task_thread.is_alive()
            for s in agent_state.agent_states.values()
        ):
            time.sleep(0.05)
        agent_state.agent_states.clear()

    def _wait(self, predicate: Any, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def _events(self, tab_id: str, type_: str) -> list[dict[str, Any]]:
        return [
            ev
            for ev in list(self.printer.emitted)
            if ev.get("type") == type_ and ev.get("tabId") == tab_id
        ]

    def test_viewer_attached_during_setup_gets_live_failure_result(
        self,
    ) -> None:
        launcher, viewer, chat_id = "sf-launcher", "sf-viewer", "chat-sf"
        self.server._cmd_run({
            "type": "run",
            "prompt": "setup failure viewer",
            "tabId": launcher,
            "taskId": f"tok-{launcher}",
            "chatId": chat_id,
            "workDir": str(self.work_dir),
            "useWorktree": False,
            "isParallel": False,
            "autoCommit": False,
            "seaPath": str(self.script),
        })
        self.assertTrue(
            self._wait((self.tmp / "entered").exists, 30.0),
            "the run never reached the SEA getter",
        )
        state = agent_state.find_by_tab(launcher)
        assert state is not None
        provisional_id = state.task_id

        # A history click on the running chat, fully completed while the
        # run is still in setup: the viewer is subscribed to the run's
        # provisional task id and flipped to running.
        self.server._replay_session(chat_id, viewer)
        self.assertIn(viewer, self.printer._fanout_targets(provisional_id))
        self.assertEqual(
            [bool(ev.get("running")) for ev in self._events(viewer, "status")],
            [True],
        )

        (self.tmp / "release").write_text("1", encoding="utf-8")
        self.assertTrue(
            self._wait(
                lambda: any(
                    ev.get("running") is False
                    for ev in self._events(viewer, "status")
                ),
                30.0,
            ),
            "the viewer never received the terminal running=false",
        )

        launcher_results = self._events(launcher, "result")
        viewer_results = self._events(viewer, "result")
        self.assertEqual(len(launcher_results), 1, launcher_results)
        self.assertEqual(
            len(viewer_results),
            1,
            "BUG: the viewer attached during setup never received the "
            "live setup-failure result (or received it more than once): "
            f"{viewer_results}",
        )
        for ev in (launcher_results[0], viewer_results[0]):
            self.assertIs(ev.get("success"), False)
            self.assertIn("setup exploded", ev.get("text", ""))
        # The result reaches the viewer BEFORE its running=false, like
        # the normal end-of-run fan-out.
        viewer_seq = [
            ev.get("type")
            for ev in list(self.printer.emitted)
            if ev.get("tabId") == viewer
            and ev.get("type") in ("result", "status")
        ]
        self.assertEqual(viewer_seq[-2:], ["result", "status"], viewer_seq)
        # Recorded exactly once under the provisional id for replay.
        recorded = [
            ev
            for ev in self.printer.peek_recording_for_task(provisional_id)
            if ev.get("type") == "result"
        ]
        self.assertEqual(len(recorded), 1, recorded)
