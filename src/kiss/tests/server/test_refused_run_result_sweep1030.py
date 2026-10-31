# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A run refused for want of a model shows why, to viewers and on replay (sweep 2026-10-30, V4).

A run that fails in setup records its ``result`` under the run's
provisional task id and delivers it to the launcher and every viewer
(``_run_task``'s setup-failure path).  The "No model available" refusal
in ``_run_task_inner`` returned normally instead, broadcasting a
launcher-only, unrecorded ``result``: a viewer that attached during
setup got only ``running=False``, and a later replay of the run had
no explanation.

Reproduced end to end: a real run through ``_cmd_run`` whose SEA
parks in ``settings()`` until released (so a viewer can attach through
the real history-click path) and that names a model no provider
serves.  No mocks, patches or test doubles.

Coverage note: the ``except RuntimeError`` arm of
``_broadcast_early_failure`` (the linger timer failing to start) needs
thread exhaustion, which cannot be produced without a test double, so
it stays uncovered.
"""

from __future__ import annotations

import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any
from unittest import TestCase

from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.server._memory_printer import MemoryPrinter

_PARKING_SCRIPT = textwrap.dedent(
    """
    import pathlib
    import time

    from kiss.agents.seas.base.base_sea import BaseSea

    _DIR = pathlib.Path(__file__).resolve().parent


    class Sea(BaseSea):
        def settings(self, settings):
            \"\"\"Park until released, then keep the run's settings.\"\"\"
            (_DIR / "entered").write_text("1", encoding="utf-8")
            deadline = time.time() + 60
            while time.time() < deadline and not (_DIR / "release").exists():
                time.sleep(0.02)
            return settings
    """
)

_REFUSAL = "No model available.  Set at least one API key in the environment."


class TestNoModelRefusalReachesViewers(TestCase):
    """The refusal is delivered and recorded like a setup failure."""

    def setUp(self) -> None:
        agent_state.agent_states.clear()
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-sweep1030-refusal-"))
        self.work_dir = self.tmp / "wd"
        self.work_dir.mkdir()
        self.script = self.tmp / "agent.py"
        self.script.write_text(_PARKING_SCRIPT, encoding="utf-8")
        self.printer = MemoryPrinter()
        self.server = AgentServer(printer=self.printer)
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

    def _refusals(self, tab_id: str) -> list[dict[str, Any]]:
        return [
            ev
            for ev in list(self.printer.emitted)
            if ev.get("type") == "result"
            and ev.get("tabId") == tab_id
            and ev.get("text") == _REFUSAL
        ]

    def test_viewer_attached_during_setup_gets_the_refusal_and_it_is_recorded(self) -> None:
        launcher, viewer, chat_id = (
            "sweep1030-nm-launcher",
            "sweep1030-nm-viewer",
            "chat-sweep1030-nm",
        )
        self.server._cmd_run(
            {
                "type": "run",
                "prompt": "sweep1030 refused run",
                "tabId": launcher,
                "chatId": chat_id,
                "model": "sweep1030-no-such-model",
                "workDir": str(self.work_dir),
                "useWorktree": False,
                "isParallel": False,
                "autoCommit": False,
                "seaPath": str(self.script),
            }
        )
        self.assertTrue(
            self._wait((self.tmp / "entered").exists, 30.0), "the run never reached the SEA"
        )
        state = agent_state.find_by_tab(launcher)
        assert state is not None
        provisional_id = state.task_id
        self.server._replay_session(chat_id, viewer)
        self.assertIn(viewer, self.printer._fanout_targets(provisional_id))

        (self.tmp / "release").write_text("1", encoding="utf-8")
        self.assertTrue(
            self._wait(
                lambda: any(
                    ev.get("type") == "status"
                    and ev.get("tabId") == viewer
                    and ev.get("running") is False
                    for ev in list(self.printer.emitted)
                ),
                30.0,
            ),
            "the viewer never received the terminal running=false",
        )
        self.assertEqual(len(self._refusals(launcher)), 1)
        self.assertEqual(
            len(self._refusals(viewer)), 1, "BUG: the viewer never learned why the run ended"
        )
        recorded = [
            ev
            for ev in self.printer.peek_recording_for_task(provisional_id)
            if ev.get("type") == "result" and ev.get("text") == _REFUSAL
        ]
        self.assertEqual(len(recorded), 1, "BUG: a replay of the refused run has no explanation")
