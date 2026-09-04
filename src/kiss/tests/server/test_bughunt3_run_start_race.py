# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 3: concurrent ``_cmd_run`` start race on one tab (BUG-C).

The winning submit installs ``state.task_thread`` under
``_state_lock`` and calls ``thread.start()`` only AFTER releasing the
lock and broadcasting the ``clear`` event (network I/O — a wide
window).  A second concurrent submit for the same tab arriving in
that window must NOT start a second task: it must observe the
installed ``task_thread`` and queue its prompt as steering instead.

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

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server import agent_state
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

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

        def stub_run(self_agent: object, **kwargs: object) -> str:
            return "success: true\nsummary: ok\n"

        self._parent_class.run = stub_run

    def tearDown(self) -> None:
        self.release.set()
        self._parent_class.run = self._original_run
        agent_state.agent_states.clear()
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
        self.server._cmd_run(dict(cmd))
        self.release.set()
        t1.join(timeout=30)

        deadline = time.time() + 30
        while time.time() < deadline:
            st = agent_state.find_by_tab(tab_id)
            if st is None or st.task_thread is None or not st.task_thread.is_alive():
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
