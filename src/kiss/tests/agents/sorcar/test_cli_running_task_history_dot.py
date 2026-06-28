# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: CLI-running tasks must show the pulsing dot in History.

User-visible bug
----------------
When a ``ChatSorcarAgent`` is executed by the ``sorcar`` CLI
(outside the VS Code / browser UI), the task row in the History
panel sidebar MUST display the green pulsing-circle "running"
indicator (CSS class ``sidebar-item-running``, keyframes
``sidebar-running-pulse``) for as long as the agent is running.

Pre-fix the running indicator was never set for CLI-launched tasks:
``VSCodeServer._get_history`` builds the ``is_running`` flag from
``_get_running_task_ids()`` which only scans the in-process
``_RunningAgentState`` registry.  The CLI agent runs in a separate
Python process and never has a registry entry on the daemon, so
its rows came back with ``is_running=False`` even while the agent
was actively producing events — no pulsing dot in History.

The daemon already tracks CLI-launched running task ids in
``RemoteAccessServer._cli_running_tasks`` (populated by the
``cliTaskStart`` envelope the CLI sends through
``cli_daemon_bridge``).  The fix exposes that set to
``VSCodeServer._get_running_task_ids`` via a snapshot lookup so
the merged set is used when constructing each session's
``is_running`` flag.

Fix under test
--------------
This test drives the production path end to end:

1. Persist a real ``task_history`` row to a temp sqlite DB.
2. From a ``RecordingConsolePrinter`` in the test process, broadcast
   the first event for that task id.  This is exactly what the CLI
   does; the printer's ``broadcast`` triggers the
   ``cli_daemon_bridge.send_cli_task_start`` envelope on its first
   event of a fresh ``taskId``.
3. Call ``VSCodeServer._get_history(...)`` with the printer
   capturing the broadcast event.
4. Assert the row for the persisted task id has ``is_running=True``
   (drives the green pulsing dot in the webview).
5. Broadcast a ``result`` event so the printer sends ``cliTaskEnd``;
   re-run ``_get_history`` and assert the row now has
   ``is_running=False``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import cli_daemon_bridge
from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.cli_printer import RecordingConsolePrinter
from kiss.agents.sorcar.persistence import _add_task, _append_chat_event
from kiss.agents.vscode.web_server import RemoteAccessServer


def _reset_cli_daemon_writer() -> None:
    """Drop the cached UDS writer between tests so a fresh daemon (on
    a new temp socket path) is contacted instead of a stale connection
    from a previous test."""
    with cli_daemon_bridge._LOCK:
        writer = cli_daemon_bridge._WRITER
        if writer is not None:
            try:
                writer.close()
            except OSError:
                pass
            cli_daemon_bridge._WRITER = None


class TestCliRunningTaskHistoryDot(unittest.TestCase):
    """``_get_history`` MUST flag CLI-running tasks as ``is_running=True``."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.sock_path = str(Path(self.tmpdir) / "sorcar.sock")

        # Redirect persistence to a temp sqlite DB so the
        # ``task_history`` row we create here doesn't leak into the
        # user's real ``~/.kiss`` and so ``_load_history`` returns
        # exactly the rows we control.
        self._saved_persistence = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True,
        )
        self.loop_thread.start()

        self.server = RemoteAccessServer(uds_path=self.sock_path)
        # ``_send_to_ws_clients`` needs the printer's loop set so it
        # can hand off writes from the calling thread to the
        # asyncio loop.
        self.server._printer._loop = self.loop

        self.uds_server: asyncio.Server = asyncio.run_coroutine_threadsafe(
            asyncio.start_unix_server(
                self.server._uds_handler, path=self.sock_path,
            ),
            self.loop,
        ).result(timeout=5)

        self._saved_env = os.environ.get("KISS_SORCAR_SOCK")
        os.environ["KISS_SORCAR_SOCK"] = self.sock_path
        _reset_cli_daemon_writer()

        # Capture every event broadcast by ``_get_history`` so we can
        # inspect the assembled session list.  ``WebPrinter.broadcast``
        # forwards through ``JsonPrinter.broadcast`` (records +
        # persists) and then fans out — we only need the recorded
        # payload here so a simple wrapper around the existing
        # broadcast suffices.
        self.captured: list[dict[str, Any]] = []
        real_broadcast = self.server._printer.broadcast

        def _capture(event: dict[str, Any]) -> None:
            self.captured.append(dict(event))
            real_broadcast(event)

        self.server._printer.broadcast = _capture  # type: ignore[method-assign]

    def tearDown(self) -> None:
        if self._saved_env is None:
            os.environ.pop("KISS_SORCAR_SOCK", None)
        else:
            os.environ["KISS_SORCAR_SOCK"] = self._saved_env
        _reset_cli_daemon_writer()

        async def _shutdown() -> None:
            self.uds_server.close()
            await self.uds_server.wait_closed()

        try:
            asyncio.run_coroutine_threadsafe(
                _shutdown(), self.loop,
            ).result(timeout=5)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self._saved_persistence
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _wait_for_cli_running(self, task_id: str,
                              timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.server._cli_running_lock:
                if task_id in self.server._cli_running_tasks:
                    return
            time.sleep(0.02)
        raise AssertionError(
            f"task_id={task_id} never registered in _cli_running_tasks",
        )

    def _wait_for_cli_not_running(self, task_id: str,
                                  timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.server._cli_running_lock:
                if task_id not in self.server._cli_running_tasks:
                    return
            time.sleep(0.02)
        raise AssertionError(
            f"task_id={task_id} still in _cli_running_tasks",
        )

    def _latest_history_session(
        self, task_id: str,
    ) -> dict[str, Any] | None:
        """Return the most recent ``history`` broadcast row for *task_id*."""
        for event in reversed(self.captured):
            if event.get("type") != "history":
                continue
            sessions = event.get("sessions")
            if not isinstance(sessions, list):
                continue
            for sess in sessions:
                if isinstance(sess, dict) and sess.get("task_id") == task_id:
                    return sess
        return None

    def test_cli_running_task_is_flagged_in_history_response(self) -> None:
        """``_get_history`` must mark CLI-running tasks ``is_running=True``."""
        # Persist a real row + one event so ``_load_history`` returns
        # this task in its result set.
        task_id, _chat_id = _add_task(
            task="cli-launched task", chat_id="",
        )
        _append_chat_event(
            {"type": "prompt", "text": "cli-launched task"},
            task_id=task_id,
        )

        # Sanity: before any CLI announcement the row must come back
        # NOT running.  This guards against false positives from
        # leftover state.
        self.server._vscode_server._get_history(query=None, offset=0)
        sess = self._latest_history_session(task_id)
        assert sess is not None, (
            f"persisted task {task_id} missing from history broadcast: "
            f"{self.captured}"
        )
        assert sess.get("is_running") is False, (
            f"task should NOT be marked running before CLI announce: {sess}"
        )

        # Simulate the production CLI: install the real
        # ``RecordingConsolePrinter`` the CLI installs, set the
        # task id on the thread-local context, and broadcast the
        # first event.  ``RecordingConsolePrinter`` sends
        # ``cliTaskStart`` on the first event seen for a fresh
        # integer ``taskId`` — the same envelope the CLI emits in
        # production.
        cli_printer = RecordingConsolePrinter()
        cli_printer._thread_local.task_id = str(task_id)
        cli_printer.broadcast({"type": "text_delta", "text": "hello"})
        self._wait_for_cli_running(task_id)

        # The actual fix point: a fresh ``_get_history`` call must
        # now return ``is_running=True`` for the CLI-launched task.
        self.captured.clear()
        self.server._vscode_server._get_history(query=None, offset=0)
        sess = self._latest_history_session(task_id)
        assert sess is not None, (
            f"task {task_id} missing from second history broadcast: "
            f"{[e.get('type') for e in self.captured]}"
        )
        assert sess.get("is_running") is True, (
            "CLI-launched running task must be flagged is_running=True so "
            "the History panel renders the pulsing green dot. Got: "
            f"{json.dumps(sess, sort_keys=True)}"
        )
        # While the task is running it must NOT be painted as failed
        # even though the persisted result column still holds the
        # mid-task ``"Agent Failed Abruptly"`` sentinel.
        assert sess.get("failed") is False, (
            f"running task must not be flagged failed: {sess}"
        )

        # Phase 2: CLI announces end → row must flip back to
        # ``is_running=False`` and the pulsing dot stops.
        cli_printer.broadcast({
            "type": "result", "text": "done", "summary": "ok",
        })
        self._wait_for_cli_not_running(task_id)

        self.captured.clear()
        self.server._vscode_server._get_history(query=None, offset=0)
        sess = self._latest_history_session(task_id)
        assert sess is not None
        assert sess.get("is_running") is False, (
            "CLI task must be flagged is_running=False after ``cliTaskEnd`` "
            "so the History panel stops pulsing. Got: "
            f"{json.dumps(sess, sort_keys=True)}"
        )


if __name__ == "__main__":
    unittest.main()
