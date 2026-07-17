# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 6: exceptions escaping ``_run_task_inner`` killed the task
thread SILENTLY (group E).

``_TaskRunnerMixin._run_task`` had no exception handler around
``_run_task_inner``: any exception raised BEFORE the inner method's
big ``try`` block (or re-raised from its pre-snapshot guard, e.g. a
git failure in ``_capture_pre_snapshot``) unwound straight through
the worker thread.  The spinner stopped (``status running=False`` is
broadcast in the ``finally``) but NO ``result`` / ``task_error`` /
``task_done`` event was ever broadcast or persisted — the user saw
the task simply vanish with no explanation.

Two deterministic triggers:

* ``attachments`` that is not a list (e.g. an int from a malformed
  client): ``for att in raw_attachments`` raised ``TypeError`` at the
  very top of ``_run_task_inner``, before even the model check.
  (Iteration 3 fixed malformed attachment *entries*; a non-iterable
  attachments *field* still killed the thread.)
* a non-string ``prompt``: ``prompt[:200]`` in the "Task started"
  log call raised ``TypeError`` after the pre-task snapshot, outside
  the inner ``try``.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.server.server as _server_module
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.server.server import VSCodeServer

_END_EVENT_TYPES = ("result", "task_done", "task_error", "task_stopped")


class TestSilentTaskDeath(unittest.TestCase):
    """A crash before the agent runs must surface a user-visible event."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt6-silent-")
        self.work_dir = str(Path(self.tmpdir) / "wd")
        Path(self.work_dir).mkdir()
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

        def capture(event: dict[str, Any]) -> None:
            with self._events_lock:
                self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

        self._orig_followup = _server_module.generate_followup_text

        def fake_followup(task: str, result: str, model: str) -> str:
            return ""

        _server_module.generate_followup_text = fake_followup  # type: ignore[assignment]

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

        def stub_run(self_agent: object, **kwargs: object) -> str:
            return "success: true\nsummary: ok\n"

        self._parent_class.run = stub_run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        _server_module.generate_followup_text = self._orig_followup
        _RunningAgentState.running_agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_and_collect(self, cmd: dict[str, Any]) -> list[dict[str, Any]]:
        """Dispatch a run command and wait for the task lifecycle to end."""
        self.server._handle_command(cmd)
        deadline = time.time() + 30
        while time.time() < deadline:
            with self._events_lock:
                done = any(
                    e.get("type") == "status" and e.get("running") is False
                    for e in self.events
                )
            if done:
                break
            time.sleep(0.02)
        # Give post-status broadcasts (none expected pre-fix) a moment.
        time.sleep(0.2)
        with self._events_lock:
            return list(self.events)

    def test_noniterable_attachments_not_silent(self) -> None:
        events = self._run_and_collect({
            "type": "run",
            "prompt": "bughunt6 silent death",
            "tabId": "bh6-att",
            "workDir": self.work_dir,
            "attachments": 5,
            "model": "",
        })
        end_events = [e for e in events if e.get("type") in _END_EVENT_TYPES]
        assert end_events, (
            "task thread died silently on a non-iterable attachments "
            f"field — no result/task_done/task_error event: {events!r}"
        )

    def test_nonstring_prompt_not_silent(self) -> None:
        available = get_available_models()
        if not available:
            self.skipTest("requires at least one available model")
        events = self._run_and_collect({
            "type": "run",
            "prompt": 5,
            "tabId": "bh6-prompt",
            "workDir": self.work_dir,
            "model": available[0],
        })
        end_events = [e for e in events if e.get("type") in _END_EVENT_TYPES]
        assert end_events, (
            "task thread died silently on a non-string prompt — no "
            f"result/task_done/task_error event: {events!r}"
        )


if __name__ == "__main__":
    unittest.main()
