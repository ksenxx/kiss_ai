# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 3: concurrent ``_cmd_run`` start race on one tab (BUG-C).

``_cmd_run``'s busy guard was ``task_thread is not None and
task_thread.is_alive()`` — but the winning submit assigns
``tab.task_thread`` under ``_state_lock`` and calls ``thread.start()``
only AFTER releasing the lock and broadcasting the ``clear`` event
(network I/O — a wide window).  A created-but-unstarted thread has
``is_alive() == False``, so a concurrent second submit for the same
tab passed the guard, clobbered ``stop_event`` /
``user_answer_queue`` / ``task_thread``, and two tasks ran
concurrently on one tab (the first becoming unstoppable, its state
then nulled by whichever finally ran first).

The test makes the race deterministic by blocking the printer's
``broadcast`` on the first ``clear`` event until a second
``_cmd_run`` has executed on another thread.
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
from kiss.server.server import VSCodeServer


class TestRunStartRace(unittest.TestCase):
    """A second submit racing the first thread.start() must be dropped."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt3-race-")
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self.first_clear_entered = threading.Event()
        self.release = threading.Event()
        self._blocked_once = False
        self._events_lock = threading.Lock()

        def blocking_broadcast(event: dict[str, Any]) -> None:
            do_block = False
            with self._events_lock:
                self.events.append(event)
                if event.get("type") == "clear" and not self._blocked_once:
                    self._blocked_once = True
                    do_block = True
            if do_block:
                self.first_clear_entered.set()
                self.release.wait(timeout=30)

        self.server.printer.broadcast = blocking_broadcast  # type: ignore[assignment]

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
        self.release.set()
        self._parent_class.run = self._original_run
        _server_module.generate_followup_text = self._orig_followup
        _RunningAgentState.running_agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_concurrent_second_submit_is_dropped(self) -> None:
        work_dir = str(Path(self.tmpdir) / "plain")
        Path(work_dir).mkdir()
        tab_id = "race-tab"
        cmd = {
            "type": "run",
            "prompt": "bughunt3 race task",
            "tabId": tab_id,
            "workDir": work_dir,
            "useWorktree": False,
            "autoCommit": False,
            "model": "",
        }
        t1 = threading.Thread(
            target=self.server._cmd_run, args=(dict(cmd),), daemon=True,
        )
        t1.start()
        assert self.first_clear_entered.wait(timeout=30), (
            "first _cmd_run never reached its clear broadcast"
        )
        # First submit is now mid-broadcast: task_thread assigned but
        # NOT yet started (is_alive() == False) — the exact race window.
        self.server._cmd_run(dict(cmd))
        self.release.set()
        t1.join(timeout=30)

        # Wait for whatever task threads ran to finish (the finally
        # resets task_thread to None).
        deadline = time.time() + 30
        while time.time() < deadline:
            tab = _RunningAgentState.running_agent_states.get(tab_id)
            if tab is not None and tab.task_thread is None:
                break
            time.sleep(0.02)

        with self._events_lock:
            clears = [e for e in self.events if e.get("type") == "clear"]
        assert len(clears) == 1, (
            f"BUG: {len(clears)} clear events — a second concurrent "
            "submit passed the busy guard during the start window and "
            "clobbered the first task's stop_event/user_answer_queue/"
            "task_thread"
        )


if __name__ == "__main__":
    unittest.main()
