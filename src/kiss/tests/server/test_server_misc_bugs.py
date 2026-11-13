# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for assorted daemon bugs (audit findings C1-C4).

C1  the ``@``-mention scan used to keep the leading ``/`` of a
    root-anchored .gitignore entry (``/node_modules``), so it never
    matched the directory name and wrongly returned files inside it.
    ``file_index.FileIndex.scan`` must honour such entries.  The test
    uses ``/generated`` because ``node_modules`` is now skipped
    unconditionally (``JUNK_DIR_NAMES``) and would pass regardless.

C2  (obsolete) ``diff_merge._write_base_copy`` was removed together
    with the interactive diff/merge review workflow.

C3  ``server._get_history`` builds each session with
    ``failed=_is_failed_result(result)`` independently of
    ``is_running``.  While a task is running its row still holds the
    "Agent Failed Abruptly" sentinel, so the payload self-contradicts
    with ``failed=True`` AND ``is_running=True`` and the sidebar paints
    a red dot on a healthy running task.

C4  ``task_runner._run_task_inner``'s subtask-failure broadcast sends
    ``step_count=tab.agent.step_count`` which is always 0 on
    RelentlessAgent-derived agents (they accumulate into
    ``total_steps``; the persisted ``extra`` correctly uses
    ``total_steps``).
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.server import agent_state
from kiss.server.file_index import FileIndex
from kiss.server.server import AgentServer
from kiss.tests.persistence_helpers import close_db


class TestGitignoreRootAnchoredEntry(unittest.TestCase):
    """C1: ``/generated`` in .gitignore must skip ``generated/``."""

    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="kiss-c1-"))

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_root_anchored_gitignore_dir_is_skipped(self) -> None:
        (self._tmp / ".gitignore").write_text("/generated\n")
        gen = self._tmp / "generated"
        gen.mkdir()
        (gen / "x.js").write_text("ignored\n")
        (self._tmp / "keep.txt").write_text("kept\n")

        paths = FileIndex.scan(str(self._tmp)).paths

        self.assertIn("keep.txt", paths)
        self.assertNotIn("generated/x.js", paths)
        self.assertNotIn("generated/", paths)

    def test_plain_gitignore_dir_still_skipped(self) -> None:
        (self._tmp / ".gitignore").write_text("build/\n")
        b = self._tmp / "build"
        b.mkdir()
        (b / "out.o").write_text("obj\n")
        (self._tmp / "keep.txt").write_text("kept\n")

        paths = FileIndex.scan(str(self._tmp)).paths

        self.assertIn("keep.txt", paths)
        self.assertNotIn("build/out.o", paths)


class _TempDbTestCase(unittest.TestCase):
    """Shared temp-sqlite + event-capturing AgentServer fixture."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="kiss-server-misc-")
        self._orig_db_path = th._DB_PATH  # type: ignore[attr-defined]
        close_db()
        th._DB_PATH = Path(self._tmp) / "history.db"  # type: ignore[attr-defined]

        self.server = AgentServer()
        self.server.work_dir = self._tmp
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        orig = self.server.printer.broadcast

        def capture(e: dict[str, Any]) -> None:
            with self._lock:
                self.events.append(dict(e))
            orig(e)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        agent_state.agent_states.clear()
        close_db()
        th._DB_PATH = self._orig_db_path  # type: ignore[attr-defined]
        shutil.rmtree(self._tmp, ignore_errors=True)


class TestRunningTaskNotMarkedFailed(_TempDbTestCase):
    """C3: a running task must not be reported ``failed`` in history."""

    def test_running_task_has_failed_false(self) -> None:
        task_id, _ = th._add_task("running task")

        release = threading.Event()
        worker = threading.Thread(target=release.wait, daemon=True)
        worker.start()
        try:
            state = agent_state.AgentState(
                str(task_id),
                tab_id="c3-tab",
                server_owned=True,
                task_thread=worker,
                is_task_active=True,
            )
            agent_state.register(state)

            self.server._handle_command({"type": "getHistory"})
        finally:
            release.set()
            worker.join(timeout=5)

        sessions: list[dict[str, Any]] = []
        with self._lock:
            for e in reversed(self.events):
                if e["type"] == "history":
                    sessions = list(e["sessions"])
                    break
        by_task = {s["title"]: s for s in sessions}
        session = by_task["running task"]

        self.assertTrue(session["is_running"])
        self.assertFalse(
            session["failed"],
            "a running task (sentinel result) must not be marked failed",
        )

    def test_finished_sentinel_task_still_marked_failed(self) -> None:
        task_id, _ = th._add_task("crashed task")
        del task_id

        self.server._handle_command({"type": "getHistory"})

        sessions: list[dict[str, Any]] = []
        with self._lock:
            for e in reversed(self.events):
                if e["type"] == "history":
                    sessions = list(e["sessions"])
                    break
        by_task = {s["title"]: s for s in sessions}
        self.assertTrue(by_task["crashed task"]["failed"])
        self.assertFalse(by_task["crashed task"]["is_running"])

    def test_in_process_error_result_is_marked_failed(self) -> None:
        """A row whose result is a persisted in-process error (``KISS Error: ...``) is a
        failure in the history payload, as it is for the recall evaluation: the history
        classifier is the one of ``kiss.core.memoryfield.evaluate``."""
        task_id, _ = th._add_task("codex task")
        th._save_task_result("KISS Error: Codex CLI failed: usage limit reached", task_id)

        self.server._handle_command({"type": "getHistory"})

        sessions: list[dict[str, Any]] = []
        with self._lock:
            for e in reversed(self.events):
                if e["type"] == "history":
                    sessions = list(e["sessions"])
                    break
        by_task = {s["title"]: s for s in sessions}
        self.assertTrue(by_task["codex task"]["failed"], "an in-process error reads as a success")
        self.assertFalse(by_task["codex task"]["is_running"])

    def test_every_row_of_a_running_chat_says_so(self) -> None:
        """A finished task of a chat that is running is listed as one (``chat_running``).

        A click on it must open the chat on its running task; the row
        of that task may not be on the page (a search lists the tasks
        that match), so each row says it for its chat.
        """
        older, chat_id = th._add_task("older task")
        th._save_task_result("success: done", older)
        running, _ = th._add_task("running task", chat_id=chat_id)
        th._add_task("another chat")

        release = threading.Event()
        worker = threading.Thread(target=release.wait, daemon=True)
        worker.start()
        try:
            agent_state.register(agent_state.AgentState(
                str(running), chat_id=chat_id, tab_id="tab", server_owned=True,
                task_thread=worker, is_task_active=True,
            ))
            self.server._handle_command({"type": "getHistory"})
        finally:
            release.set()
            worker.join(timeout=5)

        with self._lock:
            sessions = next(e for e in reversed(self.events) if e["type"] == "history")["sessions"]
        self.assertEqual(
            {s["title"]: s["chat_running"] for s in sessions},
            {"older task": True, "running task": True, "another chat": False},
        )


class TestSubtaskFailureStepCount(_TempDbTestCase):
    """C4: the failure broadcast must report the agent's total_steps."""

    def test_failure_broadcast_uses_total_steps(self) -> None:
        if not get_available_models():
            self.skipTest("no model API key configured")

        tab_id = "c4-tab"
        agent = ChatSorcarAgent("Sorcar")
        state = agent_state.AgentState(
            "task-c4",
            agent=agent,
            tab_id=tab_id,
            server_owned=True,
            stop_event=threading.Event(),
        )
        agent_state.register(state)

        def fake_run(**kwargs: Any) -> str:
            agent.total_tokens_used = 11
            agent.budget_used = 0.02
            agent.total_steps = 7
            raise RuntimeError("boom")

        agent.run = fake_run  # type: ignore[method-assign, assignment]

        self.server._run_task_inner({
            "type": "run",
            "prompt": "c4 failing task",
            "tabId": tab_id,
            "workDir": self._tmp,
            "isParallel": False,
            "useWorktree": False,
            "autoCommit": False,
        })

        with self._lock:
            results = [
                e for e in self.events
                if e["type"] == "result" and e.get("success") is False
            ]
        self.assertTrue(results, "expected a failure result broadcast")
        self.assertEqual(
            results[-1]["step_count"], 7,
            "failure broadcast must report total_steps, not the "
            "always-zero step_count",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
