# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: starting a task must refresh the History sidebar.

User-visible bug
----------------
When a task is started from the VS Code chat UI, the task's panel
does not appear in the History sidebar until the task FINISHES.
Pressing Run shows the task in the active tab, but the History
sidebar list still lacks the row for the new task — only after the
task ends does the row appear.

Root cause (pre-fix)
--------------------
The frontend's ``refreshHistory()`` is triggered by either:

* a ``tasks_updated`` broadcast (handled in main.js), OR
* a ``status running=True`` broadcast (its handler also calls
  ``refreshHistory()``).

``status running=True`` is emitted by ``web_server._run_task_inner``
BEFORE the agent run actually starts — which means BEFORE
``ChatSorcarAgent.run`` calls ``persistence._add_task`` to insert
the ``task_history`` row.  ``refreshHistory`` then queries history
while the row does not yet exist in the DB.  No ``tasks_updated``
broadcast is emitted at start time; the only existing
``tasks_updated`` broadcast lives in
``vscode/task_runner.py`` and fires AT THE END of the task.  Net
effect: the sidebar gets one refresh request before the row exists
and the next refresh trigger arrives only when the task finishes.

Fix under test
--------------
``ChatSorcarAgent.run`` must emit ``{"type": "tasks_updated"}``
through its printer immediately after ``_add_task`` has committed
the row, so the frontend can refresh and pick up the new row while
the task is still running.

This test drives the full production code path:

1. Real ``VSCodeServer`` over an in-process printer that captures
   every broadcast event (and the timestamp of each broadcast
   relative to the agent run lifecycle).
2. Issue a ``run`` command exactly as the webview does.
3. Patch the underlying ``SorcarAgent.run`` so it blocks on a
   ``threading.Event`` — simulating an agent that is RUNNING but
   has not yet produced a result.  While blocked, the test asserts
   that:

   a. ``_add_task`` has already inserted the row into
      ``task_history`` (the row is visible via the public
      ``persistence._load_history`` API), and
   b. a ``tasks_updated`` broadcast has been emitted by the agent
      (covering the new sidebar refresh path).

4. Release the block, let the task finish, and tear down.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.vscode.server import VSCodeServer


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
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmpdir,
                   capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmpdir,
                   capture_output=True)
    Path(tmpdir, ".gitkeep").touch()
    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir,
                   capture_output=True)


class _BlockingFakeAgentRun:
    """Replacement for ``SorcarAgent``'s parent ``run`` that blocks on
    a release event so the test can observe the in-flight task state.

    Records when ``run`` is entered and exited so assertions can
    distinguish broadcasts emitted DURING the run from those emitted
    after the run returns.
    """

    def __init__(self) -> None:
        self.entered_event = threading.Event()
        self.release_event = threading.Event()
        self.entered_at_count: int | None = None
        self.exit_at_count: int | None = None

    def install(self, broadcast_counter: list[int]) -> Any:
        parent = cast(Any, SorcarAgent.__mro__[1])
        original = parent.run

        def _run_proxy(self_agent: object, **kwargs: object) -> str:
            self.entered_at_count = broadcast_counter[0]
            self.entered_event.set()
            # Block here so the test can verify that at this point
            # (i.e. the agent is RUNNING) a ``tasks_updated``
            # broadcast has already been emitted by
            # ``ChatSorcarAgent.run`` and the persisted row is
            # visible.
            self.release_event.wait(timeout=10)
            printer = kwargs.get("printer")
            if printer is not None and hasattr(printer, "broadcast"):
                cast(Any, printer).broadcast({
                    "type": "result", "text": "done", "success": True,
                })
            self.exit_at_count = broadcast_counter[0]
            return "success: true\nsummary: done\n"

        parent.run = _run_proxy
        return original


def _uninstall_parent_run(original: Any) -> None:
    cast(Any, SorcarAgent.__mro__[1]).run = original


class TestTaskStartRefreshesHistory(unittest.TestCase):
    """Starting a task must broadcast ``tasks_updated`` so the History
    sidebar refreshes while the task is still running."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_db(self.tmpdir)
        _init_git_repo(self.tmpdir)

        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self.events_lock = threading.Lock()
        # Sequence counter — incremented for every captured event.
        # The blocking fake-run records its entry/exit counts so the
        # test can pinpoint which broadcasts happened DURING vs. AFTER
        # the agent run.
        self.broadcast_counter = [0]

        printer = self.server.printer
        real_broadcast = printer.broadcast

        def capture(event: dict[str, Any]) -> None:
            # Run the printer's normal injection / persistence steps
            # so this remains an end-to-end exercise of the
            # broadcast pipeline (skip only the WSS transport).
            ev = printer._inject_task_id(event)
            with printer._lock:
                printer._record_event(ev)
            printer._persist_event(ev)
            with self.events_lock:
                self.broadcast_counter[0] += 1
                self.events.append({
                    **ev, "_seq": self.broadcast_counter[0],
                })
            # Also call the real broadcast so any other registered
            # side effects still fire.  This mirrors the wrapper
            # used by other end-to-end tests in this directory.
            try:
                real_broadcast(event)
            except Exception:
                pass

        printer.broadcast = capture  # type: ignore[assignment]

        self.blocker = _BlockingFakeAgentRun()
        self.original_run = self.blocker.install(self.broadcast_counter)

    def tearDown(self) -> None:
        # Always release the blocker so the task thread can exit
        # before teardown — otherwise the daemon thread would still
        # be holding references on next test setUp.
        self.blocker.release_event.set()
        try:
            tab = self.server._get_tab("tab-start")
        except Exception:
            tab = None
        if tab is not None and tab.task_thread is not None:
            tab.task_thread.join(timeout=10)
        _uninstall_parent_run(self.original_run)
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _filter(self, ev_type: str) -> list[dict[str, Any]]:
        with self.events_lock:
            return [e for e in self.events if e.get("type") == ev_type]

    def test_starting_task_broadcasts_tasks_updated_before_finish(
        self,
    ) -> None:
        """While the agent is RUNNING, ``tasks_updated`` must have been
        broadcast AND the new row must be present in ``task_history``.
        """
        tab_id = "tab-start"
        self.server._handle_command({
            "type": "run",
            "prompt": "starting task that should appear in history",
            "model": "claude-opus-4-6",
            "workDir": self.tmpdir,
            "tabId": tab_id,
        })

        # Wait until the agent's parent ``run`` is actually entered
        # (i.e. ``ChatSorcarAgent.run`` has called ``_add_task`` and
        # all the start-time setup is complete).
        assert self.blocker.entered_event.wait(timeout=10), (
            "patched parent ``SorcarAgent.run`` was never invoked; "
            "did ``_run_task_inner`` fail to start the agent?"
        )
        entered_seq = self.blocker.entered_at_count
        assert entered_seq is not None

        # Assertion A: the task row exists in the persistence layer
        # BEFORE the agent run produces any result.  This guards
        # against a regression where ``_add_task`` would be deferred
        # to the end of the run.
        rows = th._load_history(limit=10)
        assert rows, "expected at least one task_history row at start"
        assert rows[0]["task"] == (
            "starting task that should appear in history"
        ), f"unexpected task text in row: {rows[0]}"
        running_task_id = int(cast(Any, rows[0])["id"])

        # Assertion B (the fix): a ``tasks_updated`` broadcast must
        # have been emitted by ``ChatSorcarAgent.run`` BEFORE the
        # agent body started running — so the History sidebar
        # refreshes while the task is still in progress.
        with self.events_lock:
            tasks_updated_before_run = [
                e for e in self.events
                if e.get("type") == "tasks_updated"
                and int(e.get("_seq", 0)) <= int(entered_seq)
            ]
        assert tasks_updated_before_run, (
            "no ``tasks_updated`` broadcast was emitted before the "
            "agent run started.  The History sidebar therefore "
            "cannot pick up the new running task until the task "
            "ENDS, which is the user-visible bug.  Broadcasts seen "
            "so far: "
            f"{[e.get('type') for e in self.events]}"
        )

        # Assertion C: a fresh ``_get_history`` from the production
        # ``VSCodeServer`` API surfaces the running task — i.e.
        # ``refreshHistory()`` on the frontend would now see the
        # new row.  This is the user-visible effect of the fix.
        # Pre-fix this row only appears after the task ends.
        before = len(self._filter("history"))
        self.server._get_history(query=None, offset=0)
        history_events = self._filter("history")
        assert len(history_events) > before, (
            "_get_history did not emit a fresh ``history`` broadcast"
        )
        latest = history_events[-1]
        sessions = latest.get("sessions") or []
        assert any(
            isinstance(s, dict) and s.get("task_id") == running_task_id
            for s in sessions
        ), (
            f"running task {running_task_id} missing from history "
            f"sidebar response while task is running: {sessions}"
        )

        # Release the agent so the task can finish cleanly.
        self.blocker.release_event.set()
        # Give the task thread a moment to wind down so tearDown
        # doesn't race the post-task persistence path.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                tab = self.server._get_tab(tab_id)
            except Exception:
                tab = None
            if tab is None or tab.task_thread is None:
                break
            if not tab.task_thread.is_alive():
                break
            time.sleep(0.05)


if __name__ == "__main__":
    unittest.main()
