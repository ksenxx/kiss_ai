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


class TestTaskEndEventPersistence(unittest.TestCase):
    """task_done/task_stopped/task_error events are persisted in the DB."""

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

    def test_task_done_persisted_in_db(self) -> None:
        """Successful task persists task_done event in the events table."""
        self.server._handle_command({
            "type": "run", "prompt": "test done persist",
            "model": "claude-opus-4-6", "workDir": self.tmpdir,
        })
        t = self.server._task_thread
        assert t is not None
        t.join(timeout=10)

        events = th._load_task_chat_events("test done persist")
        types = [e.get("type") for e in events]
        assert "task_done" in types

    def test_task_stopped_persisted_in_db(self) -> None:
        """Stopped task persists task_stopped event in the events table."""
        _unpatch_run(self.original_run)
        parent = cast(Any, SorcarAgent.__mro__[1])
        saved = parent.run

        def raise_ki(self_agent: object, **kwargs: object) -> str:
            raise KeyboardInterrupt("stopped")

        parent.run = raise_ki
        try:
            self.server._handle_command({
                "type": "run", "prompt": "test stop persist",
                "model": "claude-opus-4-6", "workDir": self.tmpdir,
            })
            t = self.server._task_thread
            assert t is not None
            t.join(timeout=10)
        finally:
            parent.run = saved
            self.original_run = _patch_run()

        events = th._load_task_chat_events("test stop persist")
        types = [e.get("type") for e in events]
        assert "task_stopped" in types

    def test_task_stopped_result_saved(self) -> None:
        """Stopped task saves a meaningful result string."""
        _unpatch_run(self.original_run)
        parent = cast(Any, SorcarAgent.__mro__[1])
        saved = parent.run

        def raise_ki(self_agent: object, **kwargs: object) -> str:
            raise KeyboardInterrupt("stopped")

        parent.run = raise_ki
        try:
            self.server._handle_command({
                "type": "run", "prompt": "test stop result",
                "model": "claude-opus-4-6", "workDir": self.tmpdir,
            })
            t = self.server._task_thread
            assert t is not None
            t.join(timeout=10)
        finally:
            parent.run = saved
            self.original_run = _patch_run()

        entries = th._load_history()
        entry = next(
            (e for e in entries if e["task"] == "test stop result"),
            None,
        )
        assert entry is not None
        assert "stopped" in str(entry["result"]).lower()


class TestPeriodicEventFlush(unittest.TestCase):
    """Verify events are periodically flushed to DB during task execution."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_db(self.tmpdir)
        _init_git_repo(self.tmpdir)
        self.server = VSCodeServer()
        self.server._flush_interval = 1  # speed up for testing
        # Redirect stdout so VSCodePrinter.broadcast doesn't pollute test output
        import io
        import sys
        self._real_stdout = sys.stdout
        sys.stdout = io.StringIO()

    def tearDown(self) -> None:
        import sys
        sys.stdout = self._real_stdout
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_events_flushed_before_task_completes(self) -> None:
        """Partial events are saved to DB while task is still running."""
        parent = cast(Any, SorcarAgent.__mro__[1])
        saved_run = parent.run
        resume = threading.Event()

        def slow_run(self_agent: object, **kwargs: object) -> str:
            printer = kwargs.get("printer")
            if printer:
                printer.broadcast({"type": "text_delta", "text": "partial "})  # type: ignore[union-attr,attr-defined]
                printer.broadcast({"type": "text_delta", "text": "output"})  # type: ignore[union-attr,attr-defined]
            resume.wait(timeout=15)
            return "success: true\nsummary: slow\n"

        parent.run = slow_run
        try:
            self.server._handle_command({
                "type": "run", "prompt": "test periodic flush",
                "model": "claude-opus-4-6", "workDir": self.tmpdir,
            })
            import time
            time.sleep(3)  # wait for at least 1 flush cycle

            # Events should be in DB while task is still running
            events = th._load_task_chat_events("test periodic flush")
            types = [e.get("type") for e in events]
            assert "text_delta" in types, f"Expected text_delta in {types}"

            # Result should still be "Agent Failed Abruptly" (not overwritten)
            entries = th._load_history()
            entry = next(
                (e for e in entries if e["task"] == "test periodic flush"),
                None,
            )
            assert entry is not None
            assert entry["result"] == "Agent Failed Abruptly"

            resume.set()
            t = self.server._task_thread
            assert t is not None
            t.join(timeout=10)
        finally:
            parent.run = saved_run

    def test_peek_recording_returns_snapshot(self) -> None:
        """peek_recording returns events without stopping the recording."""
        self.server.printer.start_recording(999)
        self.server.printer.broadcast({"type": "text_delta", "text": "hello"})
        events = self.server.printer.peek_recording(999)
        assert len(events) == 1
        assert events[0]["text"] == "hello"
        # Recording still active — stop_recording should also return events
        final = self.server.printer.stop_recording(999)
        assert len(final) == 1

    def test_peek_recording_empty_when_not_recording(self) -> None:
        """peek_recording returns empty list for unknown recording ID."""
        events = self.server.printer.peek_recording(12345)
        assert events == []


class TestTypescriptIsRunningFix(unittest.TestCase):
    """Verify SorcarTab.ts sets _isRunning=false on task end events."""

    def test_is_running_updated_by_status_event(self) -> None:
        """_isRunning is reset by the status event handler (sent after task_done/stopped/error)."""
        with open("src/kiss/agents/vscode/src/SorcarTab.ts") as f:
            source = f.read()
        # The status handler manages _isRunning
        idx = source.find("msg.type === 'status'")
        assert idx >= 0, "status handler not found"
        block = source[idx:idx + 200]
        assert "this._isRunning = msg.running" in block
        # Python server always sends status:running:false after task end events
        with open("src/kiss/agents/vscode/server.py") as f:
            py_source = f.read()
        assert '"type": "status", "running": False' in py_source


if __name__ == "__main__":
    unittest.main()
