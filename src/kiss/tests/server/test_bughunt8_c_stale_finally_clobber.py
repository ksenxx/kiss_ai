# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 8 (group C): a stale worker's cleanup clobbers a new task.

BUG-TR8-5 — ``_run_task``'s outer ``finally`` used to re-resolve the
tab state BY TAB ID from the global registry at cleanup time and
unconditionally null its ``task_thread`` / ``stop_event`` /
``user_answer_queue`` / ``agent`` and clear ``pending_user_messages``.
The worker can spend a long time between the agent returning and that
``finally`` (autocommit git scans, merge-view preparation, persistence
— all real I/O), during which the tab's backend state can be disposed
(``closeTab`` while the task is wedged in cleanup) and a NEW state can
be registered for the SAME tab id by a reopened frontend tab that
immediately starts a NEW task.  A tab-id lookup in the stale worker's
``finally`` would then find the FRESH state object and destroy the new
task's plumbing mid-flight: its agent slot nulled (the new worker
crashes with ``'NoneType' object has no attribute 'run'``), its answer
queue and stop event dropped (the new task becomes unanswerable and
unstoppable), and its queued follow-up messages silently discarded.

The refactored ``_run_task`` resolves its cleanup target by the run's
own registry key (``cmd["_state_key"]``), so a state registered later
for the same tab must come through completely untouched.  This test
drives that scenario with real threads and the real ``VSCodeServer``
task pipeline: the first worker blocks inside its agent ``run``, a
fresh state for the same tab id is registered and armed for a new
task, then the stale worker is allowed to finish.  No mocks or patched
methods — the agent is a real ``WorktreeSorcarAgent`` subclass whose
``run`` blocks on a real event (the established pattern of the earlier
bug-hunt tests).
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.server import VSCodeServer


def _redirect_db(tmpdir: str) -> tuple:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore_db(saved: tuple) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _init_git_repo(tmpdir: str) -> None:
    subprocess.run(["git", "init", tmpdir], capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t"], cwd=tmpdir, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "T"], cwd=tmpdir, capture_output=True,
    )
    Path(tmpdir, ".gitkeep").touch()
    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True,
    )


class _BlockingAgent(WorktreeSorcarAgent):
    """Real agent whose ``run`` blocks until released, then cancels."""

    def __init__(self) -> None:
        super().__init__("Bughunt8 blocking agent")
        self.entered = threading.Event()
        self.release = threading.Event()

    def run(self, **kwargs: Any) -> str:  # type: ignore[override]
        """Block inside the task until the test releases it."""
        self.entered.set()
        self.release.wait(timeout=30)
        raise KeyboardInterrupt("stopped by test")


class TestStaleWorkerFinallyClobbersNewTask(unittest.TestCase):
    """BUG-TR8-5: stale cleanup must not destroy a re-registered tab."""

    def setUp(self) -> None:
        models = get_available_models()
        if not models:
            self.skipTest("no model API key configured")
        self.model = models[0]
        agent_state.agent_states.clear()
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bh8c-stale-")
        self.saved = _redirect_db(self.tmpdir)
        _init_git_repo(self.tmpdir)
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

        def capture(event: dict[str, Any]) -> None:
            with self._events_lock:
                self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        agent_state.agent_states.clear()
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_stale_finally_leaves_fresh_tab_state_intact(self) -> None:
        tab_id = "bh8c-stale-tab"

        agent1 = _BlockingAgent()
        state1 = AgentState(
            "bh8c-stale-task-1",
            agent=agent1,
            tab_id=tab_id,
            server_owned=True,
            stop_event=threading.Event(),
        )
        state1.user_answer_queue = queue.Queue(maxsize=1)
        agent_state.register(state1)
        worker1 = threading.Thread(
            target=self.server._run_task,
            args=({
                "type": "run",
                "prompt": "task one",
                "model": self.model,
                "workDir": self.tmpdir,
                "tabId": tab_id,
                "_state_key": state1.task_id,
            },),
            daemon=True,
        )
        state1.task_thread = worker1
        worker1.start()
        assert agent1.entered.wait(timeout=30), "task 1 never started"

        # While worker1 is still inside its run (and therefore has its
        # whole cleanup ahead of it), the frontend closes and reopens
        # the tab: a FRESH state for the SAME tab id is registered and
        # armed for a new task.
        agent2 = WorktreeSorcarAgent("Bughunt8 fresh agent")
        queue2: queue.Queue[str] = queue.Queue(maxsize=1)
        stop2 = threading.Event()
        thread2 = threading.Thread(
            target=stop2.wait, args=(15,), daemon=True,
        )
        thread2.start()
        with self.server._state_lock:
            state2 = AgentState(
                "bh8c-fresh-task-2",
                agent=agent2,
                tab_id=tab_id,
                server_owned=True,
                stop_event=stop2,
                task_thread=thread2,
            )
            state2.user_answer_queue = queue2
            state2.is_task_active = True
            state2.pending_user_messages.append("queued follow-up")
            agent_state.register(state2)

        agent1.release.set()
        worker1.join(timeout=60)
        self.assertFalse(worker1.is_alive(), "stale worker never finished")

        try:
            self.assertIs(
                state2.agent,
                agent2,
                "BUG-TR8-5: the stale worker's finally nulled the NEW "
                "task's agent slot — the new worker would crash with "
                "'NoneType' object has no attribute 'run'",
            )
            self.assertIs(
                state2.task_thread,
                thread2,
                "BUG-TR8-5: the stale worker's finally cleared the NEW "
                "task's thread slot",
            )
            self.assertIs(
                state2.user_answer_queue,
                queue2,
                "BUG-TR8-5: the stale worker's finally dropped the NEW "
                "task's answer queue — its ask_user_question can never "
                "be answered",
            )
            self.assertIs(
                state2.stop_event,
                stop2,
                "BUG-TR8-5: the stale worker's finally dropped the NEW "
                "task's stop event — the task became unstoppable",
            )
            self.assertEqual(
                state2.pending_user_messages,
                ["queued follow-up"],
                "BUG-TR8-5: the stale worker's finally discarded the "
                "NEW task's queued follow-up messages",
            )
            self.assertTrue(
                state2.is_task_active,
                "BUG-TR8-5: the stale worker's finally flipped the NEW "
                "task's is_task_active flag while it is still running",
            )
            self.assertIs(
                agent_state.get("bh8c-fresh-task-2"),
                state2,
                "BUG-TR8-5: the stale worker's finally unregistered "
                "the NEW task's state",
            )
        finally:
            stop2.set()
            thread2.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
