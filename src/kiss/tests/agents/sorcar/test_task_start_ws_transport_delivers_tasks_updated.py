# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Transport-level end-to-end test for the task-start History refresh.

Why this test exists alongside ``test_task_start_refreshes_history``
-------------------------------------------------------------------

The sibling test ``test_task_start_refreshes_history`` wraps the
printer's ``broadcast`` method in-process and asserts a
``tasks_updated`` event is observed.  That proves the agent CALLS
``broadcast`` correctly, but it does NOT prove the broadcast
actually reaches the WS / UDS transport layer that every connected
webview client subscribes to.

This test closes that gap: it patches the printer's
``_send_to_ws_clients`` (the SINGLE chokepoint through which every
frame leaves the daemon for any connected webview) and asserts a
``tasks_updated`` JSON frame is queued for transmission to clients
BEFORE the agent body finishes.  If this assertion fails, the
History sidebar in a real browser/VS Code window would NOT receive
the refresh trigger — which is the user-visible bug.

Test flow
---------
1. Real ``VSCodeServer`` with its real ``WebPrinter``.
2. Patch ``WebPrinter._send_to_ws_clients`` to capture every frame
   it is asked to send to WS / UDS clients.
3. Patch the underlying ``SorcarAgent.run`` (one level above
   ``ChatSorcarAgent.run``) with a blocking fake so the test can
   observe the in-flight state precisely.
4. Drive ``VSCodeServer._handle_command({"type": "run", ...})``
   exactly as the production webview-submit path does.
5. Wait until the patched parent ``run`` is entered, then assert a
   ``tasks_updated`` frame is present in the captured set with
   ``taskId == ""`` (the global-broadcast convention).
6. Release the block and clean up.
"""

from __future__ import annotations

import json
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
from kiss.agents.vscode.web_server import WebPrinter


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
    """Blocking replacement for the parent ``SorcarAgent.run``.

    Lets the test pause execution while the agent is "running" so
    captured WS frames can be inspected for evidence of the
    task-start ``tasks_updated`` broadcast.
    """

    def __init__(self) -> None:
        self.entered_event = threading.Event()
        self.release_event = threading.Event()

    def install(self) -> Any:
        parent = cast(Any, SorcarAgent.__mro__[1])
        original = parent.run

        def _run_proxy(self_agent: object, **kwargs: object) -> str:
            self.entered_event.set()
            self.release_event.wait(timeout=10)
            return "success: true\nsummary: done\n"

        parent.run = _run_proxy
        return original


def _uninstall_parent_run(original: Any) -> None:
    cast(Any, SorcarAgent.__mro__[1]).run = original


class TestTaskStartWsTransportDeliversTasksUpdated(unittest.TestCase):
    """Starting a task MUST queue a ``tasks_updated`` WS frame for delivery."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_db(self.tmpdir)
        _init_git_repo(self.tmpdir)

        # Use a real ``WebPrinter`` (the same printer the production
        # ``RemoteAccessServer`` injects into ``VSCodeServer``) so the
        # broadcast under test exercises the WS / UDS transport
        # chokepoint ``_send_to_ws_clients`` — not just the base
        # ``JsonPrinter.broadcast`` recording / persistence path.
        self.printer = WebPrinter()
        self.server = VSCodeServer(printer=self.printer)
        self.captured_frames: list[str] = []
        self.captured_lock = threading.Lock()

        printer = self.printer
        original_send = printer._send_to_ws_clients

        def _capture(data: str) -> None:
            with self.captured_lock:
                self.captured_frames.append(data)
            # Still call the original so the printer's bookkeeping
            # (pending-send tracking, etc.) runs.  ``self._loop is
            # None`` in this test → ``_schedule_send`` returns early
            # without actually opening a transport, but the captured
            # JSON payload is the exact bytes a real WS / UDS peer
            # would receive.
            original_send(data)

        printer._send_to_ws_clients = _capture  # type: ignore[method-assign]

        self.blocker = _BlockingFakeAgentRun()
        self.original_run = self.blocker.install()

    def tearDown(self) -> None:
        self.blocker.release_event.set()
        try:
            tab = self.server._get_tab("tab-tx")
        except Exception:
            tab = None
        if tab is not None and tab.task_thread is not None:
            tab.task_thread.join(timeout=10)
        _uninstall_parent_run(self.original_run)
        _restore_db(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _captured_events(self) -> list[dict[str, Any]]:
        with self.captured_lock:
            frames = list(self.captured_frames)
        decoded: list[dict[str, Any]] = []
        for raw in frames:
            try:
                obj = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(obj, dict):
                decoded.append(obj)
        return decoded

    def test_tasks_updated_frame_sent_to_ws_clients_at_task_start(
        self,
    ) -> None:
        """While the agent is RUNNING, a ``tasks_updated`` JSON frame
        must have already been queued for WS / UDS delivery."""
        tab_id = "tab-tx"
        self.server._handle_command({
            "type": "run",
            "prompt": "transport-level start refresh",
            "model": "claude-opus-4-6",
            "workDir": self.tmpdir,
            "tabId": tab_id,
        })

        # Wait until the patched parent ``run`` is actually entered:
        # at this point ``ChatSorcarAgent.run`` has finished its
        # start-time setup (including the ``tasks_updated`` broadcast
        # under test).
        assert self.blocker.entered_event.wait(timeout=10), (
            "patched parent SorcarAgent.run was never invoked"
        )

        # Give any pending broadcasts a brief moment to finish
        # writing into the capture list (the broadcast is synchronous
        # but the patched run releases entered_event BEFORE returning).
        deadline = time.monotonic() + 1.0
        events: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            events = self._captured_events()
            if any(e.get("type") == "tasks_updated" for e in events):
                break
            time.sleep(0.02)

        tasks_updated = [
            e for e in events if e.get("type") == "tasks_updated"
        ]
        assert tasks_updated, (
            "No ``tasks_updated`` frame was queued for WS/UDS delivery "
            "while the task was running.  The History sidebar in real "
            "browser / VS Code windows would not refresh until the task "
            "ENDS, which is the user-visible bug.  Captured frame types "
            f"so far: {[e.get('type') for e in events]}"
        )
        # Global broadcasts carry an empty ``taskId`` (or no key) so
        # ``WebPrinter.broadcast`` routes them through the
        # ``not event.get('taskId')`` global branch.  A frame stamped
        # with a non-empty ``taskId`` would have gone through the
        # per-tab fan-out branch and would only reach subscribers of
        # that specific task — defeating the purpose of refreshing
        # every connected client's sidebar.
        assert any(
            (not e.get("taskId")) for e in tasks_updated
        ), (
            "tasks_updated frame must be a GLOBAL broadcast "
            "(taskId='' or absent) so every connected client refreshes. "
            f"Got: {tasks_updated}"
        )

        self.blocker.release_event.set()
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
