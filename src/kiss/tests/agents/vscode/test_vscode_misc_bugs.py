# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for assorted vscode-backend bugs (audit findings C1-C4).

C1  ``diff_merge._load_gitignore_dirs`` keeps the leading ``/`` of a
    root-anchored .gitignore entry (``/node_modules``), so
    ``_scan_files`` never matches the directory name and wrongly
    returns files inside it.

C2  ``diff_merge._write_base_copy`` materialises the base copy of a
    *text* file via a text-mode ``git show`` (universal newlines turn
    CRLF into LF) + ``write_text``, so for CRLF files every line of the
    base differs from the working file and the merge view shows
    spurious whole-file hunks.  The base copy must preserve the exact
    committed bytes (like the binary path already does).

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
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.agents.vscode.diff_merge import _scan_files, _write_base_copy
from kiss.agents.vscode.server import VSCodeServer
from kiss.core.models.model_info import get_available_models


class TestGitignoreRootAnchoredEntry(unittest.TestCase):
    """C1: ``/node_modules`` in .gitignore must skip ``node_modules/``."""

    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="kiss-c1-"))

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_root_anchored_gitignore_dir_is_skipped(self) -> None:
        (self._tmp / ".gitignore").write_text("/node_modules\n")
        nm = self._tmp / "node_modules"
        nm.mkdir()
        (nm / "x.js").write_text("ignored\n")
        (self._tmp / "keep.txt").write_text("kept\n")

        paths = _scan_files(str(self._tmp))

        self.assertIn("keep.txt", paths)
        self.assertNotIn("node_modules/x.js", paths)
        self.assertNotIn("node_modules/", paths)

    def test_plain_gitignore_dir_still_skipped(self) -> None:
        (self._tmp / ".gitignore").write_text("build/\n")
        b = self._tmp / "build"
        b.mkdir()
        (b / "out.o").write_text("obj\n")
        (self._tmp / "keep.txt").write_text("kept\n")

        paths = _scan_files(str(self._tmp))

        self.assertIn("keep.txt", paths)
        self.assertNotIn("build/out.o", paths)


class TestWriteBaseCopyPreservesCrlf(unittest.TestCase):
    """C2: the text-file base copy must preserve the committed bytes."""

    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="kiss-c2-"))

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _git(self, *args: str) -> None:
        subprocess.run(
            [
                "git",
                "-c", "user.email=test@test",
                "-c", "user.name=test",
                "-c", "commit.gpgsign=false",
                *args,
            ],
            cwd=self._tmp,
            check=True,
            capture_output=True,
        )

    def test_crlf_file_base_copy_keeps_crlf_bytes(self) -> None:
        committed = b"alpha\r\nbeta\r\ngamma\r\n"
        self._git("init")
        (self._tmp / "crlf.txt").write_bytes(committed)
        self._git("add", "crlf.txt")
        self._git("commit", "-m", "add crlf file")

        merge_dir = self._tmp / "merge-temp"
        ub_dir = self._tmp / "untracked-base"  # empty: no saved base copy

        base_path = _write_base_copy(
            str(self._tmp), merge_dir, ub_dir, "crlf.txt", "HEAD",
        )

        produced = base_path.read_bytes()
        self.assertIn(b"\r\n", produced)
        self.assertEqual(produced, committed)


class _TempDbTestCase(unittest.TestCase):
    """Shared temp-sqlite + event-capturing VSCodeServer fixture."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="kiss-vscode-misc-")
        self._orig_db_path = th._DB_PATH  # type: ignore[attr-defined]
        th._close_db()
        th._DB_PATH = Path(self._tmp) / "sorcar.db"  # type: ignore[attr-defined]

        self.server = VSCodeServer()
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
        th._close_db()
        th._DB_PATH = self._orig_db_path  # type: ignore[attr-defined]
        shutil.rmtree(self._tmp, ignore_errors=True)


class TestRunningTaskNotMarkedFailed(_TempDbTestCase):
    """C3: a running task must not be reported ``failed`` in history."""

    def test_running_task_has_failed_false(self) -> None:
        task_id, _ = th._add_task("running task")
        # While the task runs, its row still holds the
        # "Agent Failed Abruptly" sentinel result.

        release = threading.Event()
        worker = threading.Thread(target=release.wait, daemon=True)
        worker.start()
        try:
            tab = self.server._get_tab("c3-tab")
            tab.task_history_id = task_id
            tab.task_thread = worker

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
        by_task = {s["preview"]: s for s in sessions}
        session = by_task["running task"]

        self.assertTrue(session["is_running"])
        self.assertFalse(
            session["failed"],
            "a running task (sentinel result) must not be marked failed",
        )

    def test_finished_sentinel_task_still_marked_failed(self) -> None:
        task_id, _ = th._add_task("crashed task")
        del task_id  # no running tab: row keeps the sentinel result

        self.server._handle_command({"type": "getHistory"})

        sessions: list[dict[str, Any]] = []
        with self._lock:
            for e in reversed(self.events):
                if e["type"] == "history":
                    sessions = list(e["sessions"])
                    break
        by_task = {s["preview"]: s for s in sessions}
        self.assertTrue(by_task["crashed task"]["failed"])
        self.assertFalse(by_task["crashed task"]["is_running"])


class TestSubtaskFailureStepCount(_TempDbTestCase):
    """C4: the failure broadcast must report the agent's total_steps."""

    def test_failure_broadcast_uses_total_steps(self) -> None:
        if not get_available_models():
            self.skipTest("no model API key configured")

        tab_id = "c4-tab"
        tab = self.server._get_tab(tab_id)
        agent = WorktreeSorcarAgent("Sorcar VS Code")
        tab.agent = agent
        tab.chat_id = ""

        def fake_run(**kwargs: Any) -> str:
            agent.total_tokens_used = 11
            agent.budget_used = 0.02
            # RelentlessAgent-derived agents accumulate completed steps
            # into ``total_steps``; ``step_count`` stays 0.
            agent.total_steps = 7
            raise RuntimeError("boom")

        agent.run = fake_run  # type: ignore[method-assign, assignment]

        self.server._run_task_inner({
            "type": "run",
            "prompt": "c4 failing task",
            "tabId": tab_id,
            "workDir": self._tmp,
            "useParallel": False,
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
