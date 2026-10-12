# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt integration test for task_runner.py / server.py.

BUG-1 (server.py ``_replay_session``): when a tab opens a chat whose
task is STILL RUNNING, the resumed-running ``status`` broadcast reads
``startTs`` from the persisted ``extra`` JSON — but ``startTs`` is only
written into ``extra`` at task END (``_save_task_extra`` in
``_run_task_inner``'s cleanup finally).  The early ``extra`` payload
written at task INSERT (``ChatSorcarAgent._build_extra_payload``) has
no ``startTs``, so every live resume broadcast carried ``startTs: 0``
and the frontend (which requires ``ev.startTs > 0``) anchored the
"Running …" timer at the client's ``Date.now()`` — showing "Running 0s"
for a task that had been running for minutes.  This contradicts the
method's own docstring, which claims the broadcast echoes the agent's
true start time.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


class _BugHuntBase(unittest.TestCase):
    """Shared scaffolding: stub the inner agent run, capture broadcasts."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt-")
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []

        def capture(event: dict[str, Any]) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestResumeRunningStartTs(_BugHuntBase):
    """BUG-1: resuming a still-running task must broadcast a real startTs."""

    def test_replay_session_of_running_task_sends_nonzero_start_ts(self) -> None:
        work_dir = str(Path(self.tmpdir) / "plain")
        Path(work_dir).mkdir()
        release = threading.Event()

        def stub_run(self_agent: object, **kwargs: object) -> str:
            release.wait(timeout=30)
            return "success: true\nsummary: long task done\n"

        self._parent_class.run = stub_run

        src_tab = "bughunt-src-tab"
        self.server._cmd_run({
            "type": "run",
            "prompt": "bughunt long-running task",
            "tabId": src_tab,
            "workDir": work_dir,
            "model": "",
        })
        try:
            task_id: str | None = None
            deadline = time.time() + 30
            while time.time() < deadline:
                tab = agent_state.find_by_tab(src_tab)
                if (
                    tab is not None
                    and tab.agent is not None
                    and tab.agent._last_task_id is not None
                ):
                    task_id = tab.agent._last_task_id
                    break
                time.sleep(0.02)
            assert task_id is not None, "task row was never allocated"

            self.events.clear()
            self.server._cmd_resume_session({
                "taskId": task_id,
                "tabId": "bughunt-viewer-tab",
            })

            status_events = [
                e for e in self.events
                if e.get("type") == "status"
                and e.get("running") is True
                and e.get("tabId") == "bughunt-viewer-tab"
            ]
            assert status_events, (
                "resume of a running task must broadcast a "
                "status running=True event to the viewer tab"
            )
            start_ts = status_events[-1].get("startTs", 0)
            assert isinstance(start_ts, int) and start_ts > 0, (
                "BUG: status broadcast for a resumed RUNNING task "
                f"carried startTs={start_ts!r} — the frontend ignores "
                "startTs <= 0 and mis-anchors the 'Running …' timer at "
                "the client's Date.now()"
            )
            now_ms = int(time.time() * 1000)
            assert now_ms - 60_000 <= start_ts <= now_ms, (
                f"startTs {start_ts} is not a plausible recent ms-epoch "
                f"timestamp (now={now_ms})"
            )
        finally:
            release.set()
            tab = agent_state.find_by_tab(src_tab)
            if tab is not None and tab.task_thread is not None:
                tab.task_thread.join(timeout=30)


if __name__ == "__main__":
    unittest.main()
