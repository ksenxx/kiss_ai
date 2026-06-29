# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: clicking a running CLI task in history streams events.

User-visible bug
----------------
A previous fix already relays CLI events to webview tabs that are
ALREADY subscribed to the task's chat id (see
``test_cli_daemon_live_stream.py``).  But the real workflow is:

  1. User opens a fresh chat webview tab (no chat is yet loaded).
  2. User clicks a still-running CLI task in the History panel.
  3. The webview MUST start streaming events live AND its tab title
     MUST show the blinking-green-circle "running" indicator.

Pre-fix step (3) was broken for CLI-launched tasks: ``_replay_session``
calls ``_reattach_running_chat`` which scans the in-process
``_RunningAgentState`` registry, and CLI tasks never have an entry
there (the agent runs in the CLI process, not the daemon).  As a
result the newly-opened tab was never subscribed to the task id and
no ``status:running=true`` event was broadcast — so the tab silently
showed only the persisted history and the title was static.

Fix under test
--------------
The CLI now announces ``cliTaskStart`` / ``cliTaskEnd`` envelopes
around every task it runs.  The daemon tracks running task ids in
``RemoteAccessServer._cli_running_tasks``.  When ``_replay_session``
finds ``_reattach_running_chat`` returned False AND the task id is
in the CLI-running set, it (a) subscribes the new tab to the task's
event stream (so subsequent ``cliEvent`` relays fan out to it) and
(b) the existing rebound-running branch broadcasts
``status:running=true`` so the tab gets the blinking green circle.
A subsequent ``cliTaskEnd`` clears the running set AND fans out
``status:running=false`` to every still-subscribed tab so the
circle stops.
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


class TestCliHistoryClickResumesLiveStream(unittest.TestCase):
    """Click a running CLI task in history → live stream + green circle."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.sock_path = str(Path(self.tmpdir) / "sorcar.sock")

        # Point persistence at a temp sqlite DB so the ``task_history``
        # rows we create here don't pollute the user's real ``~/.kiss``.
        # ``_replay_session`` early-returns when the events lookup
        # returns nothing, so an actual row + at least one event must
        # be persisted before the resume click flow runs.
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
        # ``_send_to_ws_clients`` needs the printer's loop set.
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

        self._viewer_writer: asyncio.StreamWriter | None = None

    def tearDown(self) -> None:
        if self._saved_env is None:
            os.environ.pop("KISS_SORCAR_SOCK", None)
        else:
            os.environ["KISS_SORCAR_SOCK"] = self._saved_env
        _reset_cli_daemon_writer()

        async def _shutdown() -> None:
            try:
                if self._viewer_writer is not None:
                    self._viewer_writer.close()
                    await self._viewer_writer.wait_closed()
            except Exception:
                pass
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
        # Restore persistence globals before deleting the temp tree
        # so any background flushes that race teardown don't write
        # to a path we are about to remove.
        th._DB_PATH, th._db_conn, th._KISS_DIR = self._saved_persistence
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _open_viewer(self) -> tuple[list[str], threading.Event]:
        """Open a viewer UDS client and drain its inbox to *received*."""
        async def _open() -> tuple[
            asyncio.StreamReader, asyncio.StreamWriter,
        ]:
            return await asyncio.open_unix_connection(
                self.sock_path, limit=16 * 1024 * 1024,
            )

        reader, writer = asyncio.run_coroutine_threadsafe(
            _open(), self.loop,
        ).result(timeout=5)
        self._viewer_writer = writer

        received: list[str] = []
        got = threading.Event()

        async def _drain() -> None:
            while True:
                line = await reader.readline()
                if not line:
                    return
                received.append(line.decode("utf-8"))
                got.set()

        asyncio.run_coroutine_threadsafe(_drain(), self.loop)
        return received, got

    def _wait_for_writers(self, expected: int, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.server._printer._ws_lock:
                if len(self.server._printer._uds_writers) >= expected:
                    return
            time.sleep(0.02)
        raise AssertionError(
            f"only {len(self.server._printer._uds_writers)} UDS writers "
            f"registered (expected {expected})"
        )

    def _wait_for_cli_running(self, task_id: str,
                              timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.server._cli_running_lock:
                if task_id in self.server._cli_running_tasks:
                    return
            time.sleep(0.02)
        raise AssertionError(
            f"task_id={task_id} never registered in _cli_running_tasks"
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
            f"task_id={task_id} still in _cli_running_tasks after {timeout}s"
        )

    def _decoded(self, received: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for line in list(received):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def test_history_click_subscribes_tab_and_starts_indicator(
        self,
    ) -> None:
        """Clicking the running task subscribes the tab + green circle.

        Drives the exact production code path the user complained
        about: CLI announces a task as running, a fresh viewer tab
        opens and asks ``_replay_session`` to load the task (the
        history-panel click flow), and the viewer must (a) be
        subscribed to the live event stream and (b) be told the task
        is running so the tab title shows the blinking green circle.
        """
        # In production the CLI's ``ChatSorcarAgent.run`` allocates a
        # ``task_history`` row before broadcasting any events, and
        # the ``RecordingConsolePrinter`` then persists each event
        # under that row.  Mirror this exactly: ``_replay_session``
        # looks the chat events up by task_id and early-returns
        # when none exist, so we need both a real row and at least
        # one persisted event for the resume click flow to run.
        task_id, chat_id = _add_task(
            task="streamed CLI task", chat_id="",
        )
        _append_chat_event(
            {"type": "prompt", "text": "streamed CLI task"},
            task_id=task_id,
        )

        # Phase 1: CLI process announces it is running ``task_id``.
        # In production the announcement is emitted by
        # ``RecordingConsolePrinter`` on the first event of a fresh
        # taskId; we exercise that same path here so the daemon side
        # is exactly the production state when the user later clicks
        # the row in the History panel.
        cli_printer = RecordingConsolePrinter()
        cli_printer._thread_local.task_id = str(task_id)
        cli_printer.broadcast({"type": "text_delta", "text": "before-click"})
        self._wait_for_cli_running(task_id)

        # Phase 2: a fresh chat webview opens and clicks the running
        # task in the History panel.  Frontend → ``_replay_session``
        # via the WS / UDS dispatcher.  This is the line under fix:
        # without it the new tab would have NO subscription, NO live
        # stream, and a static (non-blinking) tab title.
        received, got = self._open_viewer()
        self._wait_for_writers(1)

        viewer_tab = "history-click-tab"
        self.server._vscode_server._replay_session(
            chat_id=chat_id,
            tab_id=viewer_tab,
            task_id=task_id,
        )

        # Phase 3: the viewer must now be subscribed to the live
        # stream and must have received a ``status:running=true``
        # event stamped with its tabId (the blinking green circle).
        with self.server._printer._lock:
            subscribers = self.server._printer._subscribers.get(
                str(task_id), set(),
            )
        assert viewer_tab in subscribers, (
            f"viewer tab not subscribed: {subscribers}"
        )

        assert got.wait(timeout=3.0), (
            f"viewer received NO data after history click; got: {received}"
        )
        # Give the broadcast burst a small drain window so we catch
        # both ``status`` and ``task_events`` in one pass.
        time.sleep(0.1)
        decoded = self._decoded(received)
        running_evs = [
            d for d in decoded
            if d.get("type") == "status"
            and d.get("running") is True
            and d.get("tabId") == viewer_tab
        ]
        assert running_evs, (
            "viewer did not receive status:running=true with its tabId — "
            "tab title would not blink green. "
            f"received: {decoded}"
        )

        # Phase 4: with the subscription in place, a fresh CLI event
        # for the SAME task id must now reach this viewer LIVE
        # (this is the original bug from the user — clicking the
        # history row must light up live streaming, not just history).
        received.clear()
        got.clear()
        cli_printer.broadcast({"type": "text_delta", "text": "after-click"})
        assert got.wait(timeout=3.0), (
            f"viewer received no live events after subscription; got: "
            f"{received}"
        )
        time.sleep(0.1)
        decoded = self._decoded(received)
        live_deltas = [
            d for d in decoded
            if d.get("type") == "text_delta"
            and d.get("text") == "after-click"
            and d.get("tabId") == viewer_tab
        ]
        assert live_deltas, (
            "viewer did not receive the live ``text_delta`` stamped for "
            f"its tabId; got: {decoded}"
        )

        # Phase 5: CLI announces the task is done.  The daemon must
        # both drop the task from ``_cli_running_tasks`` AND fan out
        # ``status:running=false`` so the blinking green circle
        # stops on every still-subscribed tab.
        received.clear()
        got.clear()
        cli_printer.broadcast({
            "type": "result", "text": "done", "summary": "ok",
        })
        self._wait_for_cli_not_running(task_id)
        assert got.wait(timeout=3.0), (
            f"viewer received nothing on task end; got: {received}"
        )
        time.sleep(0.1)
        decoded = self._decoded(received)
        stop_evs = [
            d for d in decoded
            if d.get("type") == "status"
            and d.get("running") is False
            and d.get("tabId") == viewer_tab
        ]
        assert stop_evs, (
            "viewer did not receive status:running=false with its tabId — "
            f"tab title would keep blinking forever. received: {decoded}"
        )

    def test_uds_disconnect_cleans_up_stale_cli_tasks(self) -> None:
        """A CLI crash without ``cliTaskEnd`` must not leak running tasks.

        If the CLI process announces ``cliTaskStart`` and then dies
        (Ctrl+C, SIGKILL, uncaught exception), the daemon must drop
        the task id from ``_cli_running_tasks`` when the UDS
        connection closes — otherwise a webview later clicking the
        row would mis-display the blinking green circle forever for
        a task that is no longer running anywhere.
        """
        task_id = "9999cccccccccccccccccccccccccccc"

        # Open a dedicated UDS connection that we will close abruptly
        # below.  Using the bridge directly is what the CLI does, but
        # we want to control the close, so use ``open_unix_connection``
        # and send the envelope ourselves.
        async def _open_and_send() -> asyncio.StreamWriter:
            reader, writer = await asyncio.open_unix_connection(
                self.sock_path,
                limit=16 * 1024 * 1024,
            )
            envelope = json.dumps({
                "type": "cliTaskStart", "taskId": task_id,
            }).encode() + b"\n"
            writer.write(envelope)
            await writer.drain()
            return writer

        writer = asyncio.run_coroutine_threadsafe(
            _open_and_send(), self.loop,
        ).result(timeout=5)
        self._wait_for_cli_running(task_id)

        # Now drop the connection without sending ``cliTaskEnd``.
        async def _abrupt_close() -> None:
            writer.close()
            await writer.wait_closed()

        asyncio.run_coroutine_threadsafe(
            _abrupt_close(), self.loop,
        ).result(timeout=5)

        # The UDS handler's ``finally`` must drop the leaked task id.
        self._wait_for_cli_not_running(task_id)


if __name__ == "__main__":
    unittest.main()
