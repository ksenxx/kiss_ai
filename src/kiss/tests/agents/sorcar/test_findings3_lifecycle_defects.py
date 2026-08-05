# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end reproductions for round-3 task-lifecycle audit findings.

Each test reproduces a confirmed defect from the deep audit of
``src/kiss/server/{server,task_runner,commands}.py`` and passes only
because the defect is fixed:

S3-05  ``_cmd_run`` installed and started ``tab.task_thread`` before the
       worker raised ``is_task_active``, so a second ``run`` (or an
       ``appendUserMessage``) submitted during that startup window
       failed the ``is_task_active or not thread.is_alive()`` predicate
       and the typed prompt was silently dropped.  Input must now be
       queued whenever a task thread is installed/alive.

S3-07  Mandatory end-of-task cleanup (clearing ``is_task_active``,
       ``task_history_id``, the printer persist-agent/recording and the
       worker's thread-local task id) lived in the same broad ``try`` as
       fallible persistence/broadcast/merge work, so a persistence error
       left the tab permanently carrying its finished task's identity.
       The mandatory cleanup must now run in its own ``finally``.

S3-08  The follow-up suggestion thread was started BEFORE
       ``printer.cleanup_task`` removed the task's persist-agent, so a
       fast follow-up was persisted twice (once automatically by
       ``JsonPrinter.broadcast`` while the persist-agent still existed,
       once explicitly via ``_append_chat_event``) while a slow one was
       persisted once.  ``cleanup_task`` now runs first, making the
       explicit append the single scheduling-independent persistence
       path.

S3-09  ``_subagent_is_done`` consulted only
       ``ChatSorcarAgent.running_agents`` (without its lock) while
       ``_reattach_running_chat`` scanned live ``_RunningAgentState``
       entries, so during the registration gaps replay could reattach a
       task as running and simultaneously report it done.  Both sources
       are now checked under their locks.

S3-14  (verification) ``JsonPrinter.cleanup_task`` now prunes the
       task's subscriber set after a linger period instead of retaining
       it for the tab's whole lifetime.

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
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
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


def _pop_tab(tab_id: str) -> None:
    """Remove *tab_id* from the global running-agent registry."""
    with _RunningAgentState._registry_lock:
        _RunningAgentState.running_agent_states.pop(tab_id, None)


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

    ``_cmd_run`` installs and starts ``tab.task_thread`` and only later
    (inside the worker) raises ``is_task_active``.  The tests recreate
    exactly that window with a real alive worker thread whose
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
            tab = _RunningAgentState(self.tab_id, self.server._default_model)
            tab.task_thread = worker
            tab.is_task_active = False  # startup window: thread alive, flag down
            _RunningAgentState.running_agent_states[self.tab_id] = tab
        self.tab = tab
        # Race tests: jitter the window a little before acting.
        time.sleep(random.uniform(0.001, 0.05))

    def tearDown(self) -> None:
        self.release.set()
        if self.tab.task_thread is not None:
            self.tab.task_thread.join(timeout=5)
        _pop_tab(self.tab_id)
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
            self.tab.pending_user_messages,
            "a second `run` submitted while the worker thread was alive "
            "but is_task_active was still False must queue the prompt, "
            "not silently drop it",
        )
        self.assertIn(
            "follow-up typed during startup window",
            self.tab.unattributed_prompt_echoes,
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
            self.tab.pending_user_messages,
            "appendUserMessage during the startup window must queue the "
            "prompt on the tab's own live task",
        )

    def test_truly_idle_tab_still_drops_append(self) -> None:
        idle_id = "s305-idle-tab"
        with self.server._state_lock:
            idle = _RunningAgentState(idle_id, self.server._default_model)
            _RunningAgentState.running_agent_states[idle_id] = idle
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
            _pop_tab(idle_id)


class TestSubagentDoneConsistency(unittest.TestCase):
    """S3-09: ``_subagent_is_done`` must agree with reattachment."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_db(self.tmpdir)
        self.server = VSCodeServer()
        self.tab_id = "s309-sub-tab"
        self.task_id = "s309-task-row-id"
        self.release = threading.Event()

    def tearDown(self) -> None:
        self.release.set()
        _pop_tab(self.tab_id)
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _install_tab(self, *, active: bool, thread_alive: bool) -> None:
        with self.server._state_lock:
            tab = _RunningAgentState(
                self.tab_id,
                self.server._default_model,
                is_subagent=True,
                is_task_active=active,
            )
            tab.task_history_id = self.task_id
            if thread_alive:
                worker = threading.Thread(
                    target=self.release.wait, daemon=True,
                )
                worker.start()
                tab.task_thread = worker
            _RunningAgentState.running_agent_states[self.tab_id] = tab

    def test_active_tab_missing_from_running_agents_is_not_done(self) -> None:
        """Startup gap: DB row + live tab state exist, map entry not yet."""
        self._install_tab(active=True, thread_alive=False)
        time.sleep(random.uniform(0.001, 0.05))
        with ChatSorcarAgent._running_agents_lock:
            self.assertNotIn(self.task_id, ChatSorcarAgent.running_agents)
        self.assertFalse(
            _subagent_is_done(self.task_id),
            "a live tab state owning the task id means the sub-agent is "
            "still running even before it registers in running_agents",
        )

    def test_alive_thread_without_active_flag_is_not_done(self) -> None:
        self._install_tab(active=False, thread_alive=True)
        self.assertFalse(
            _subagent_is_done(self.task_id),
            "an alive worker thread owning the task id means the "
            "sub-agent is still running",
        )

    def test_reattach_and_done_never_contradict(self) -> None:
        """The exact audit contradiction: reattached AND reported done."""
        self._install_tab(active=True, thread_alive=False)
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
    persist-agent wiring (a real ``ChatSorcarAgent`` registered in
    ``printer._persist_agents``, exactly as ``ChatSorcarAgent.run``
    registers itself):

    * pre-fix order (broadcast while the persist-agent is still
      registered, then the explicit append) → TWO rows;
    * post-fix order (``cleanup_task`` first, as ``_run_task_inner``
      now does, then the follow-up thread's broadcast + append) → ONE
      row, and the lingering subscriber set still fans the broadcast
      out to the tab.
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
        with self.printer._lock:
            self.printer._persist_agents[self.task_id] = self.agent

    def tearDown(self) -> None:
        with self.printer._lock:
            self.printer._persist_agents.pop(self.task_id, None)
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
        # Pre-fix ordering: persist-agent still registered while the
        # follow-up thread broadcasts — the defect mechanism.
        self._run_followup_thread("fast follow-up (persist-agent live)")
        self.assertEqual(
            _count_followup_rows(self.task_id),
            2,
            "with the persist-agent still registered, broadcast + "
            "explicit append must double-persist — this is the defect "
            "mechanism the reorder eliminates",
        )

        # Post-fix ordering: _run_task_inner now calls cleanup_task
        # BEFORE starting the follow-up thread.
        self.printer.subscribe_tab(self.task_id, "s308-viewer-tab")
        self.printer.cleanup_task(self.task_id)
        before = _count_followup_rows(self.task_id)
        self._run_followup_thread("follow-up after cleanup_task")
        self.assertEqual(
            _count_followup_rows(self.task_id) - before,
            1,
            "after cleanup_task removed the persist-agent, the explicit "
            "append must be the single persistence path",
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
    before the end-of-task cleanup runs.  The tab must still shed its
    task identity and activity flags.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_db(self.tmpdir)
        _init_git_repo(self.tmpdir)
        self.server = VSCodeServer()
        self.blocker = _CompletingParentRun()
        self.original_run = self.blocker.install()
        self.tab_id = "s307-tab"

    def tearDown(self) -> None:
        self.blocker.release_event.set()
        with self.server._state_lock:
            tab = _RunningAgentState.running_agent_states.get(self.tab_id)
        if tab is not None and tab.task_thread is not None:
            tab.task_thread.join(timeout=15)
        cast(Any, SorcarAgent.__mro__[1]).run = self.original_run
        _pop_tab(self.tab_id)
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_tab_sheds_task_identity_when_persistence_crashes(self) -> None:
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
            tab = _RunningAgentState.running_agent_states.get(self.tab_id)
        self.assertIsNotNone(tab)
        assert tab is not None

        rows = th._load_history(limit=1)
        self.assertTrue(rows, "the task row must exist while running")
        self.history_task_id = str(rows[0]["id"])

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
        assert tab.task_thread is not None
        tab.task_thread.join(timeout=15)
        self.assertFalse(tab.task_thread is not None and tab.task_thread.is_alive())

        self.assertFalse(
            tab.is_task_active,
            "the tab must not stay flagged active after cleanup crashed",
        )
        self.assertFalse(
            tab.is_running_non_wt,
            "the non-worktree running flag must be lowered even when "
            "cleanup persistence crashed",
        )
        self.assertIsNone(
            tab.task_history_id,
            "the tab must shed its finished task's identity even when "
            "the persistence/broadcast section of cleanup raised — "
            "pre-fix the broad except skipped this, leaving the tab "
            "permanently bound to the dead task",
        )
        with self.server.printer._lock:
            self.assertNotIn(
                self.history_task_id,
                self.server.printer._persist_agents,
                "no persist-agent entry may outlive the crashed cleanup",
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
