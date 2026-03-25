"""Test: running a task after newChat should not hang.

Root cause: task_done was broadcast from inside the try block of _run_task,
before the finally block that sends status:running:false.  The webview
received task_done and enabled input, but the TypeScript extension's
_isRunning stayed True (only updated by 'status' events).  When the user
sent a new task, the extension's submit handler silently dropped it because
_isRunning was True.

Fix: Move task_done/task_stopped/task_error broadcasts to the end of the
finally block, right before status:running:false, so both arrive together
after all cleanup (merge view, file cache, etc.) is complete.
"""

import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.vscode.server import VSCodeServer


def _redirect_db(tmpdir: str) -> tuple:
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._db_conn = None
    return old


def _restore_db(saved: tuple) -> None:
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _init_git_repo(tmpdir: str) -> None:
    subprocess.run(["git", "init", tmpdir], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmpdir, capture_output=True)
    Path(tmpdir, ".gitkeep").touch()
    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True)


def _make_server(tmpdir: str) -> tuple[VSCodeServer, list[dict[str, Any]], threading.Lock]:
    server = VSCodeServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()

    def capture(event: dict[str, Any]) -> None:
        with lock:
            events.append(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events, lock


def _patch_run() -> Any:
    """Monkey-patch RelentlessAgent.run to avoid LLM calls. Returns original."""
    parent = cast(Any, SorcarAgent.__mro__[1])
    original = parent.run
    parent.run = lambda self, **kw: "success: true\nsummary: done\n"
    return original


def _unpatch_run(original: Any) -> None:
    cast(Any, SorcarAgent.__mro__[1]).run = original


class TestTaskEndEventOrdering(unittest.TestCase):
    """task_done/task_stopped must come after cleanup, right before status:false."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_db(self.tmpdir)
        _init_git_repo(self.tmpdir)
        self.server, self.events, self.lock = _make_server(self.tmpdir)
        self.original_run = _patch_run()

    def tearDown(self) -> None:
        _unpatch_run(self.original_run)
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_and_wait(self, prompt: str) -> None:
        self.server._handle_command({
            "type": "run", "prompt": prompt,
            "model": "claude-opus-4-6", "workDir": self.tmpdir,
        })
        t = self.server._task_thread
        assert t is not None
        t.join(timeout=10)
        assert not t.is_alive()

    def test_task_done_immediately_before_status_false(self) -> None:
        """task_done and status:false must be adjacent (no cleanup between them)."""
        self._run_and_wait("test task")

        with self.lock:
            events = list(self.events)

        task_done_idx = next(
            (i for i, e in enumerate(events) if e.get("type") == "task_done"), None
        )
        status_false_idx = next(
            (i for i, e in enumerate(events)
             if e.get("type") == "status" and e.get("running") is False), None
        )
        assert task_done_idx is not None
        assert status_false_idx is not None
        assert status_false_idx == task_done_idx + 1, (
            f"status:false (idx={status_false_idx}) must be right after "
            f"task_done (idx={task_done_idx})"
        )

    def test_tasks_updated_before_task_done(self) -> None:
        """tasks_updated (part of cleanup) must come before task_done."""
        self._run_and_wait("ordering check")

        with self.lock:
            events = list(self.events)

        task_done_idx = next(
            (i for i, e in enumerate(events) if e.get("type") == "task_done"), None
        )
        tasks_updated_idx = next(
            (i for i, e in enumerate(events) if e.get("type") == "tasks_updated"), None
        )
        assert task_done_idx is not None
        assert tasks_updated_idx is not None
        assert tasks_updated_idx < task_done_idx

    def test_second_task_after_new_chat_completes(self) -> None:
        """Running a task after newChat should not hang."""
        self._run_and_wait("task 1")
        with self.lock:
            self.events.clear()

        self.server._handle_command({"type": "newChat"})
        self._run_and_wait("task 2")

        with self.lock:
            status_false = [
                e for e in self.events
                if e.get("type") == "status" and e.get("running") is False
            ]
        assert len(status_false) >= 1

    def test_task_stopped_immediately_before_status_false(self) -> None:
        """task_stopped and status:false must be adjacent."""
        _unpatch_run(self.original_run)
        parent = cast(Any, SorcarAgent.__mro__[1])
        saved = parent.run

        def raise_ki(self_agent: object, **kwargs: object) -> str:
            raise KeyboardInterrupt("stopped")

        parent.run = raise_ki
        try:
            self.server._handle_command({
                "type": "run", "prompt": "stop me",
                "model": "claude-opus-4-6", "workDir": self.tmpdir,
            })
            t = self.server._task_thread
            assert t is not None
            t.join(timeout=10)
        finally:
            parent.run = saved
            self.original_run = _patch_run()

        with self.lock:
            events = list(self.events)

        stopped_idx = next(
            (i for i, e in enumerate(events) if e.get("type") == "task_stopped"), None
        )
        status_false_idx = next(
            (i for i, e in enumerate(events)
             if e.get("type") == "status" and e.get("running") is False), None
        )
        assert stopped_idx is not None
        assert status_false_idx is not None
        assert status_false_idx == stopped_idx + 1


class TestTypescriptIsRunningFix(unittest.TestCase):
    """Verify SorcarPanel.ts sets _isRunning=false on task end events."""

    def test_is_running_updated_on_task_done(self) -> None:
        with open("src/kiss/agents/vscode/src/SorcarPanel.ts") as f:
            source = f.read()
        assert "task_done" in source
        assert "task_stopped" in source
        assert "task_error" in source
        # Find the block that sets _isRunning = false for task end events
        assert "this._isRunning = false" in source
        # Verify the condition checks all three event types
        idx = source.find("task_done")
        block = source[max(0, idx - 200):idx + 200]
        assert "task_stopped" in block
        assert "task_error" in block


if __name__ == "__main__":
    unittest.main()
