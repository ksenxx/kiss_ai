# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 8 (group C): a stale worker's cleanup clobbers a new task.

BUG-TR8-5 — ``_run_task``'s outer ``finally`` re-resolves the tab state
BY ID from the global registry
(``_RunningAgentState.running_agent_states.get(tab_id)``) at cleanup
time and unconditionally nulls its ``task_thread`` / ``stop_event`` /
``user_answer_queue`` / ``agent`` and clears ``pending_user_messages``.
The worker can spend a long time between the agent returning and that
``finally`` (autocommit git scans, merge-view preparation, persistence
— all real I/O), during which the tab's backend state can be disposed
(``closeTab`` while the task is wedged in cleanup) and RE-CREATED under
the SAME tab id by a reopened frontend tab that immediately starts a
NEW task.  The stale worker's ``finally`` then looks up the FRESH state
object and destroys the new task's plumbing mid-flight: its agent slot
is nulled (the new worker crashes with ``'NoneType' object has no
attribute 'run'``), its answer queue and stop event are dropped (the
new task becomes unanswerable and unstoppable), and its queued
follow-up messages are silently discarded.

Observed in the wild as the flaky
``test_resume_running_followup_input`` failure where a lingering
worker from a previous task nulled the next task's agent between
``_run_task_inner``'s ``assert tab.agent is not None`` and
``tab.agent.run(...)``.

The test makes the race deterministic with real threads and the real
``VSCodeServer`` task pipeline: the first worker is paused inside its
post-run cleanup by holding ``_state_lock``, the tab state is swapped
(dispose + re-register, as closeTab + reopen would do) and re-armed
for a new task, then the stale worker is allowed to finish.  No mocks
or patched methods — the agent is a real ``WorktreeSorcarAgent``
subclass whose ``run`` blocks on a real event (the established
pattern of the earlier bug-hunt tests).
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
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.server import VSCodeServer
from kiss.core.models.model_info import get_available_models


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
        _RunningAgentState.running_agent_states.clear()
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
        _RunningAgentState.running_agent_states.clear()
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_stale_finally_leaves_fresh_tab_state_intact(self) -> None:
        tab_id = "bh8c-stale-tab"

        # --- Task 1: armed exactly like _cmd_run, with a real agent
        # whose run() blocks so the worker is provably in flight.
        state1 = self.server._get_tab(tab_id)
        agent1 = _BlockingAgent()
        state1.agent = agent1
        state1.stop_event = threading.Event()
        state1.user_answer_queue = queue.Queue(maxsize=1)
        worker1 = threading.Thread(
            target=self.server._run_task,
            args=({
                "type": "run",
                "prompt": "task one",
                "model": self.model,
                "workDir": self.tmpdir,
                "tabId": tab_id,
            },),
            daemon=True,
        )
        state1.task_thread = worker1
        worker1.start()
        assert agent1.entered.wait(timeout=30), "task 1 never started"

        # Pause worker1's post-run cleanup: grab _state_lock BEFORE
        # releasing the agent, so the worker blocks at its first
        # cleanup lock acquisition — i.e. before the outer finally's
        # per-tab reset runs.
        self.server._state_lock.acquire()
        try:
            agent1.release.set()

            # While worker1 is wedged in cleanup, the frontend closes
            # the tab (disposing the backend state) and reopens it:
            # a FRESH state is registered under the SAME tab id and a
            # new task is armed on it.
            _RunningAgentState.running_agent_states.pop(tab_id, None)
            state2 = _RunningAgentState(tab_id, self.model)
            _RunningAgentState.register(tab_id, state2)
            agent2 = WorktreeSorcarAgent("Bughunt8 fresh agent")
            queue2: queue.Queue[str] = queue.Queue(maxsize=1)
            stop2 = threading.Event()
            thread2 = threading.Thread(
                target=stop2.wait, args=(15,), daemon=True,
            )
            thread2.start()
            state2.agent = agent2
            state2.user_answer_queue = queue2
            state2.stop_event = stop2
            state2.task_thread = thread2
            state2.is_task_active = True
            state2.pending_user_messages.append("queued follow-up")
        finally:
            self.server._state_lock.release()

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
        finally:
            stop2.set()
            thread2.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
