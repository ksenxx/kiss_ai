# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end reproductions for round-3 task-lifecycle audit findings.

Each test reproduces a confirmed defect from the deep audit of
``src/kiss/server/{server,task_runner,commands}.py`` and passes only
because the defect is fixed (behavior preserved by the task-keyed
``kiss.server.agent_state`` registry):

S3-05  A second ``run`` (or an ``appendUserMessage``) submitted during
       the task startup window — worker thread installed and alive but
       ``is_task_active`` not yet raised — must be queued on the
       state's ``pending_user_messages``, not silently dropped.

S3-07  Mandatory end-of-task cleanup (clearing ``is_task_active`` /
       ``is_running_non_wt`` and the printer's per-task recording)
       must run in its own ``finally`` so a persistence crash cannot
       leave the task permanently flagged active.

S3-08  The follow-up suggestion must persist exactly once: while the
       run's agent is still resolvable through the registry (live
       ``_last_task_id``) a broadcast double-persists; after the
       production cleanup (``_last_task_id`` cleared +
       ``cleanup_task``) the explicit append is the single
       persistence path, and the lingering subscriber set still fans
       the broadcast out to the tab.

S3-09  ``_subagent_is_done`` must agree with
       ``_reattach_running_chat``: both consult the same live-state
       predicate on the task-keyed registry, so replay can never
       reattach a task as running while stamping its tab
       ``isDone=True``.

S3-14  (verification) ``JsonPrinter.cleanup_task`` prunes the task's
       subscriber set after a linger period instead of retaining it
       for the tab's whole lifetime.

The tests use real threads, a real ``VSCodeServer``, a real
``JsonPrinter``, and a real temporary SQLite database.  The only
harness device is the repo-standard parent-run proxy (also used by
``test_task_start_refreshes_history.py``) that lets the FULL production
task lifecycle in ``_run_task``/``_run_task_inner`` execute without a
live LLM call.
"""

from __future__ import annotations

import random
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
import kiss.server.server as _server_module
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server import agent_state
from kiss.server.server import VSCodeServer, _subagent_is_done


def _redirect_db(tmpdir: str) -> tuple:
    """Point the persistence layer at a fresh DB under *tmpdir*."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore_db(saved: tuple) -> None:
    """Restore the persistence layer redirected by :func:`_redirect_db`."""
    if th._db_conn is not None:
        try:
            th._db_conn.close()
        except Exception:
            pass
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _init_git_repo(tmpdir: str) -> None:
    """Create a minimal committed git repo in *tmpdir*."""
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


def _count_followup_rows(task_id: str) -> int:
    """Count persisted ``followup_suggestion`` event rows for *task_id*."""
    th._flush_chat_events()
    conn = sqlite3.connect(th._DB_PATH)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM events WHERE task_id = ? "
            "AND event_json LIKE '%followup_suggestion%'",
            (task_id,),
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


class TestStartupWindowInput(unittest.TestCase):
    """S3-05: input typed during the task startup window must be queued.

    ``_cmd_run`` installs and starts ``state.task_thread`` and only
    later (inside the worker) raises ``is_task_active``.  The tests
    recreate exactly that window with a real alive worker thread whose
    ``is_task_active`` flag is still False.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_db(self.tmpdir)
        self.server = VSCodeServer()
        self.tab_id = "s305-tab"
        self.release = threading.Event()
        worker = threading.Thread(target=self.release.wait, daemon=True)
        worker.start()
        with self.server._state_lock:
            state = agent_state.AgentState(
                "s305-run-key",
                tab_id=self.tab_id,
                server_owned=True,
                is_task_active=False,  # startup window: thread alive, flag down
            )
            state.task_thread = worker
            agent_state.register(state)
        self.state = state
        # Race tests: jitter the window a little before acting.
        time.sleep(random.uniform(0.001, 0.05))

    def tearDown(self) -> None:
        self.release.set()
        if self.state.task_thread is not None:
            self.state.task_thread.join(timeout=5)
        agent_state.agent_states.clear()
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_second_run_during_startup_window_is_queued(self) -> None:
        self.server._handle_command({
            "type": "run",
            "prompt": "follow-up typed during startup window",
            "tabId": self.tab_id,
        })
        self.assertIn(
            "follow-up typed during startup window",
            self.state.pending_user_messages,
            "a second `run` submitted while the worker thread was alive "
            "but is_task_active was still False must queue the prompt, "
            "not silently drop it",
        )
        self.assertIn(
            "follow-up typed during startup window",
            self.state.unattributed_prompt_echoes,
            "the queued prompt has no owning task id yet, so it must be "
            "recorded for late attribution",
        )

    def test_append_user_message_during_startup_window_is_queued(self) -> None:
        self.server._handle_command({
            "type": "appendUserMessage",
            "prompt": "typed while the task was starting",
            "tabId": self.tab_id,
        })
        self.assertIn(
            "typed while the task was starting",
            self.state.pending_user_messages,
            "appendUserMessage during the startup window must queue the "
            "prompt on the tab's own live task",
        )

    def test_truly_idle_tab_still_drops_append(self) -> None:
        idle_id = "s305-idle-tab"
        with self.server._state_lock:
            idle = agent_state.AgentState(
                "s305-idle-key",
                tab_id=idle_id,
                server_owned=True,
            )
            agent_state.register(idle)
        try:
            self.server._handle_command({
                "type": "appendUserMessage",
                "prompt": "message to an idle tab",
                "tabId": idle_id,
            })
            self.assertEqual(
                idle.pending_user_messages,
                [],
                "a tab with no thread and no active task must still "
                "reject queued input (nothing would ever drain it)",
            )
        finally:
            agent_state.unregister("s305-idle-key", idle)


class TestSubagentDoneConsistency(unittest.TestCase):
    """S3-09: ``_subagent_is_done`` must agree with reattachment."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_db(self.tmpdir)
        self.server = VSCodeServer()
        self.task_id = "s309-task-row-id"
        self.release = threading.Event()

    def tearDown(self) -> None:
        self.release.set()
        agent_state.agent_states.clear()
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _install_state(self, *, active: bool, thread_alive: bool) -> None:
        with self.server._state_lock:
            state = agent_state.AgentState(
                self.task_id,
                chat_id="s309-chat",
                tab_id="s309-sub-tab",
                parent_task_id="s309-parent-id",
                is_task_active=active,
            )
            if thread_alive:
                worker = threading.Thread(
                    target=self.release.wait, daemon=True,
                )
                worker.start()
                state.task_thread = worker
            agent_state.register(state)

    def test_active_state_is_not_done(self) -> None:
        """Startup gap: registry entry active, no thread installed yet."""
        self._install_state(active=True, thread_alive=False)
        time.sleep(random.uniform(0.001, 0.05))
        self.assertFalse(
            _subagent_is_done(self.task_id),
            "a live registered state owning the task id means the "
            "sub-agent is still running",
        )

    def test_alive_thread_without_active_flag_is_not_done(self) -> None:
        self._install_state(active=False, thread_alive=True)
        self.assertFalse(
            _subagent_is_done(self.task_id),
            "an alive worker thread owning the task id means the "
            "sub-agent is still running",
        )

    def test_reattach_and_done_never_contradict(self) -> None:
        """The exact audit contradiction: reattached AND reported done."""
        self._install_state(active=True, thread_alive=False)
        reattached = self.server._reattach_running_chat(
            "s309-chat",
            "s309-viewer-tab",
            task_id=self.task_id,
            is_subagent=True,
        )
        self.assertTrue(reattached)
        self.assertFalse(
            _subagent_is_done(self.task_id),
            "replay must not reattach a task as running while "
            "simultaneously stamping its tab isDone=True",
        )

    def test_unknown_task_is_done(self) -> None:
        self.assertTrue(_subagent_is_done("s309-no-such-task"))
        self.assertTrue(_subagent_is_done(""))
        self.assertTrue(_subagent_is_done(None))


class TestFollowupSinglePersistence(unittest.TestCase):
    """S3-08: the follow-up suggestion must persist exactly once.

    Reproduces both orderings of the race deterministically against a
    real ``JsonPrinter`` and a real SQLite DB, with the production
    persistence wiring (a real ``ChatSorcarAgent`` reachable through
    the task-keyed registry, exactly as ``ChatSorcarAgent.run``
    installs itself via ``printer.agent_task_allocated``):

    * pre-fix order (broadcast while the agent's ``_last_task_id``
      is still live in the registry, then the explicit append) →
      TWO rows;
    * post-fix order (production cleanup first — ``_last_task_id``
      cleared and ``cleanup_task`` called, as ``ChatSorcarAgent.run``
      / ``_run_task_inner`` now do — then the follow-up thread's
      broadcast + append) → ONE row, and the lingering subscriber
      set still fans the broadcast out to the tab.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_db(self.tmpdir)
        self.server = VSCodeServer()
        self.printer = self.server.printer
        task_id, _chat_id = th._add_task("s308 follow-up persistence task")
        self.task_id = str(task_id)
        self.agent = ChatSorcarAgent("s308-agent")
        with self.agent._task_id_lock:
            self.agent._last_task_id = task_id
        state = agent_state.AgentState(
            self.task_id,
            agent=cast(Any, self.agent),
            is_task_active=True,
        )
        agent_state.register(state)
        self.state = state

    def tearDown(self) -> None:
        agent_state.agent_states.clear()
        with self.printer._lock:
            self.printer._subscribers.pop(self.task_id, None)
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _followup_thread_body(self, event: dict[str, object]) -> None:
        """Exactly what ``_generate_followup_async``'s thread does."""
        self.printer._thread_local.task_id = self.task_id
        time.sleep(random.uniform(0.001, 0.05))
        self.printer.broadcast(dict(event))
        th._append_chat_event(
            dict(event),
            task_id=self.task_id,
            task="s308 follow-up persistence task",
            origin_db_path=th._current_db_path(),
        )

    def _run_followup_thread(self, text: str) -> None:
        event: dict[str, object] = {"type": "followup_suggestion", "text": text}
        worker = threading.Thread(
            target=self._followup_thread_body, args=(event,), daemon=True,
        )
        worker.start()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive())

    def test_prefix_order_duplicates_and_fixed_order_does_not(self) -> None:
        # Pre-fix ordering: the agent is still resolvable through the
        # registry with a live _last_task_id while the follow-up thread
        # broadcasts — the defect mechanism.
        self._run_followup_thread("fast follow-up (persist wiring live)")
        self.assertEqual(
            _count_followup_rows(self.task_id),
            2,
            "with the agent still registered and its _last_task_id "
            "live, broadcast + explicit append must double-persist — "
            "this is the defect mechanism the cleanup reorder "
            "eliminates",
        )

        # Post-fix ordering: production cleanup runs BEFORE the
        # follow-up thread — ChatSorcarAgent.run clears _last_task_id
        # and _run_task_inner calls cleanup_task first.
        self.printer.subscribe_tab(self.task_id, "s308-viewer-tab")
        with self.agent._task_id_lock:
            self.agent._last_task_id = None
        self.printer.cleanup_task(self.task_id)
        before = _count_followup_rows(self.task_id)
        self._run_followup_thread("follow-up after cleanup_task")
        self.assertEqual(
            _count_followup_rows(self.task_id) - before,
            1,
            "after cleanup the explicit append must be the single "
            "persistence path",
        )
        with self.printer._lock:
            self.assertIn(
                "s308-viewer-tab",
                self.printer._subscribers.get(self.task_id, set()),
                "cleanup_task must keep the subscriber set alive (linger) "
                "so the follow-up broadcast still reaches the tab",
            )


class _CompletingParentRun:
    """Repo-standard parent-run proxy: blocks, then returns success.

    Lets the FULL production ``_run_task``/``_run_task_inner`` lifecycle
    execute without a live LLM call (same harness as
    ``test_task_start_refreshes_history.py``).
    """

    def __init__(self) -> None:
        self.entered_event = threading.Event()
        self.release_event = threading.Event()

    def install(self) -> Any:
        parent = cast(Any, SorcarAgent.__mro__[1])
        original = parent.run

        def _run_proxy(self_agent: object, **kwargs: object) -> str:
            self.entered_event.set()
            self.release_event.wait(timeout=15)
            return "success: true\nsummary: done\n"

        parent.run = _run_proxy
        return original


class TestCleanupExceptionSafety(unittest.TestCase):
    """S3-07: mandatory cleanup must survive a persistence crash.

    Drives a real task through ``_run_task``, then makes the
    persistence layer genuinely unusable (the DB path becomes a
    directory, so ``_get_db`` raises a real ``sqlite3.OperationalError``)
    before the end-of-task cleanup runs.  The state must still shed its
    activity flags and the printer its per-task recording.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_db(self.tmpdir)
        _init_git_repo(self.tmpdir)
        self.server = VSCodeServer()
        self.blocker = _CompletingParentRun()
        self.original_run = self.blocker.install()
        self._orig_followup = _server_module.generate_followup_text

        def fake_followup(task: str, result: str, model: str) -> str:
            return ""

        _server_module.generate_followup_text = fake_followup  # type: ignore[assignment]
        self.tab_id = "s307-tab"

    def tearDown(self) -> None:
        self.blocker.release_event.set()
        with self.server._state_lock:
            state = agent_state.find_by_tab(self.tab_id)
        if state is not None and state.task_thread is not None:
            state.task_thread.join(timeout=15)
        cast(Any, SorcarAgent.__mro__[1]).run = self.original_run
        _server_module.generate_followup_text = self._orig_followup
        agent_state.agent_states.clear()
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_state_sheds_activity_when_persistence_crashes(self) -> None:
        self.server._handle_command({
            "type": "run",
            "prompt": "task whose cleanup persistence crashes",
            "model": "claude-opus-4-6",
            "workDir": self.tmpdir,
            "tabId": self.tab_id,
        })
        self.assertTrue(
            self.blocker.entered_event.wait(timeout=15),
            "the agent run never started",
        )
        with self.server._state_lock:
            state = agent_state.find_by_tab(self.tab_id)
        self.assertIsNotNone(state)
        assert state is not None

        rows = th._load_history(limit=1)
        self.assertTrue(rows, "the task row must exist while running")
        history_task_id = str(rows[0]["id"])

        # Sabotage the DB while the agent is still "running": close the
        # connection and turn the DB path into a directory, so every
        # subsequent persistence call raises for real.
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        bad = Path(self.tmpdir) / "db-as-directory"
        bad.mkdir()
        th._DB_PATH = bad
        time.sleep(random.uniform(0.001, 0.05))

        self.blocker.release_event.set()
        assert state.task_thread is not None
        state.task_thread.join(timeout=15)
        self.assertFalse(
            state.task_thread is not None and state.task_thread.is_alive()
        )

        self.assertFalse(
            state.is_task_active,
            "the state must not stay flagged active after cleanup crashed",
        )
        self.assertFalse(
            state.is_running_non_wt,
            "the non-worktree running flag must be lowered even when "
            "cleanup persistence crashed",
        )
        with self.server.printer._lock:
            self.assertNotIn(
                history_task_id,
                self.server.printer._recordings,
                "no per-task recording may outlive the crashed cleanup",
            )


class TestSubscriberLingerPrune(unittest.TestCase):
    """S3-14 (verification): finished tasks must not leak subscribers."""

    def test_subscribers_are_pruned_after_linger(self) -> None:
        server = VSCodeServer()
        printer = server.printer
        printer.subscribe_tab("s314-task", "s314-tab")
        with printer._lock:
            self.assertIn("s314-task", printer._subscribers)
        printer.cleanup_task("s314-task", subscriber_linger_seconds=0.05)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            # Pruning may be lazy (swept on the next subscriber-map
            # operation), so keep exercising the map like a long-lived
            # tab running follow-on tasks would.
            printer.subscribe_tab("s314-other-task", "s314-tab")
            printer.cleanup_task(
                "s314-other-task", subscriber_linger_seconds=0,
            )
            with printer._lock:
                if "s314-task" not in printer._subscribers:
                    return
            time.sleep(0.02)
        self.fail(
            "the finished task's subscriber set was never pruned — "
            "long-lived tabs would leak one entry per completed task",
        )


if __name__ == "__main__":
    unittest.main()
