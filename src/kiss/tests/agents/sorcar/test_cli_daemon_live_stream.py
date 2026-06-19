# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: ``sorcar`` CLI events must stream to open webviews.

User-visible bug
----------------
When a task is launched via the ``sorcar`` CLI and a chat webview is
ALREADY open on the task's chat id, the webview shows nothing — no
panels for ``tool_call``, ``tool_result``, ``text_delta`` etc. — until
the user reloads the page (which replays from the events DB).  The
chat panel does not update live as the CLI emits each event.

Root cause
----------
The CLI's :class:`~kiss.agents.sorcar.cli_printer.RecordingConsolePrinter`
inherits :meth:`JsonPrinter.broadcast`, which records + persists each
event but has NO transport: every event sits in the DB until the next
page reload re-renders it.  Meanwhile the running daemon's
:class:`~kiss.agents.vscode.web_server.WebPrinter` is the ONLY object
that fans events out over the WSS / UDS transports to connected
webviews; the CLI process has no way to feed it.

Fix under test
--------------
The CLI now opens a best-effort AF_UNIX connection to the daemon's
UDS endpoint (default ``~/.kiss/sorcar.sock``; override via
``KISS_SORCAR_SOCK``) and after each
:meth:`RecordingConsolePrinter.broadcast` forwards the event in a
``{"type": "cliEvent", "event": ...}`` envelope (see
:mod:`kiss.agents.sorcar.cli_daemon_bridge`).  The daemon dispatcher
short-circuits ``cliEvent`` to :meth:`RemoteAccessServer._relay_cli_event`
which mirrors the tail of :meth:`WebPrinter.broadcast`: look up tabs
subscribed to the event's task id and splice each ``tabId`` into the
JSON payload, then push to every WSS / UDS client.  Crucially the
daemon side does NOT re-record or re-persist the event — the CLI
already did both via the inherited :meth:`JsonPrinter.broadcast`.

This test wires up the real bridge end-to-end over a temp UDS path
and asserts a webview subscribed to the task's chat id receives the
event LIVE (while the CLI broadcast call has not yet finished joining)
instead of only after a page reload.  It also asserts the daemon's
:class:`WebPrinter` did NOT double-record the event.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
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
from kiss.agents.sorcar.cli_printer import RecordingConsolePrinter
from kiss.agents.vscode.web_server import RemoteAccessServer


class TestCliDaemonLiveStream(unittest.TestCase):
    """CLI events reach an open chat webview without a page reload."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.sock_path = str(Path(self.tmpdir) / "sorcar.sock")
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True,
        )
        self.loop_thread.start()

        # A real RemoteAccessServer: gives us the actual ``_uds_handler``
        # the production daemon uses (so the dispatcher under test runs
        # on the real code path), wired to a temp UDS so concurrent
        # tests can't race on the shared ``~/.kiss/sorcar.sock``.
        self.server = RemoteAccessServer(uds_path=self.sock_path)
        # ``_schedule_send`` (used by ``_send_to_ws_clients``) requires
        # the printer's loop to be set; production sets this inside
        # ``_setup_server`` which we don't run here.
        self.server._printer._loop = self.loop

        # Stand the UDS endpoint up on our loop using the SAME handler
        # production uses — the CLI bridge connects here, and the
        # ``cliEvent`` envelope flows through the real dispatcher.
        self.uds_server: asyncio.Server = asyncio.run_coroutine_threadsafe(
            asyncio.start_unix_server(
                self.server._uds_handler, path=self.sock_path,
            ),
            self.loop,
        ).result(timeout=5)

        # Capture this test's KISS_SORCAR_SOCK override so the CLI
        # bridge connects to OUR temp daemon, not the user's real one.
        self._saved_env = os.environ.get("KISS_SORCAR_SOCK")
        os.environ["KISS_SORCAR_SOCK"] = self.sock_path
        cli_daemon_bridge.reset_for_tests()

        self._viewer_writer: asyncio.StreamWriter | None = None
        self._reader_task: concurrent.futures.Future[None] | None = None

    def tearDown(self) -> None:
        # Restore env BEFORE shutting down the loop so any stray
        # bridge calls during teardown can't accidentally hit a
        # half-closed socket.
        if self._saved_env is None:
            os.environ.pop("KISS_SORCAR_SOCK", None)
        else:
            os.environ["KISS_SORCAR_SOCK"] = self._saved_env
        cli_daemon_bridge.reset_for_tests()

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
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _open_viewer(self) -> tuple[asyncio.StreamReader, list[str],
                                    threading.Event]:
        """Open a "chat webview" UDS connection and drain its inbox.

        Mirrors how the VS Code extension and ``RemoteAccessServer``'s
        UDS clients consume newline-JSON broadcasts.  Returns the
        reader plus a thread-safe ``received`` list and an event that
        is set on every line received so the test can ``wait`` on it.
        """
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

        self._reader_task = asyncio.run_coroutine_threadsafe(
            _drain(), self.loop,
        )
        return reader, received, got

    def _wait_for_uds_writer(self, expected_count: int,
                             timeout: float = 2.0) -> None:
        """Wait until ``add_uds_writer`` has registered *expected_count*.

        The asyncio handler runs concurrently with our test thread, so
        new connections take a tick or two to land in ``_uds_writers``.
        Without this barrier the broadcast can race ahead of the
        viewer's registration and the fan-out would miss it.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.server._printer._ws_lock:
                if len(self.server._printer._uds_writers) >= expected_count:
                    return
            time.sleep(0.02)
        raise AssertionError(
            f"only {len(self.server._printer._uds_writers)} UDS writers "
            f"registered (expected {expected_count})"
        )

    def test_cli_event_reaches_subscribed_webview_live(self) -> None:
        """Bug repro: a CLI broadcast must reach an already-open webview.

        Pre-fix the ``RecordingConsolePrinter`` only persisted the
        event to the DB; nothing fed the running daemon, so a chat
        webview subscribed to the task's chat id received NOTHING
        until the user reloaded the page.  Post-fix the bridge ships
        the event to the daemon and ``_relay_cli_event`` fans it out
        with the viewer tab's ``tabId`` stamped in.
        """
        task_id = "cli-live-task-1"
        # The viewer tab subscribes BEFORE the CLI broadcast — this
        # is the "I had the chat open when the CLI task started"
        # scenario the user reported.
        self.server._printer.subscribe_tab(task_id, "tab-viewer")
        _reader, received, got = self._open_viewer()
        # Two writers expected in ``_uds_writers`` once both peers
        # have connected: the viewer above plus the CLI bridge below.
        self._wait_for_uds_writer(1)

        # CLI side: install a real ``RecordingConsolePrinter`` and
        # broadcast a single event under the task's thread-local id.
        # This is the SAME printer the CLI installs at runtime.
        cli_printer = RecordingConsolePrinter()
        cli_printer._thread_local.task_id = task_id
        cli_printer.broadcast({"type": "text_delta", "text": "hello-cli"})

        # Within a generous window the viewer must receive a JSON
        # line carrying the event with the viewer's ``tabId``
        # stamped — exactly what ``WebPrinter.broadcast`` would have
        # produced if the daemon itself had emitted the event.
        assert got.wait(timeout=3.0), (
            "viewer received NO data — bridge or relay broken; "
            f"received so far: {received}"
        )
        # Drain a tiny extra window so we don't miss a payload that
        # is in flight at the moment ``got`` flipped to set.
        time.sleep(0.1)

        matches: list[dict[str, Any]] = []
        for line in list(received):
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (decoded.get("type") == "text_delta"
                    and decoded.get("tabId") == "tab-viewer"):
                matches.append(decoded)
        assert matches, (
            "viewer did not receive a text_delta stamped for "
            f"'tab-viewer'; raw lines: {received}"
        )
        ev = matches[0]
        assert ev.get("text") == "hello-cli", ev
        assert ev.get("taskId") == task_id, ev

    def test_daemon_does_not_double_record_cli_event(self) -> None:
        """The daemon must NOT re-record or re-persist a CLI event.

        The CLI process already did both via the inherited
        :meth:`JsonPrinter.broadcast`.  ``_relay_cli_event`` is
        deliberately a thin fan-out and skips ``_record_event`` /
        ``_persist_event`` — otherwise every CLI event would land in
        the DB twice (once from the CLI's printer, once from the
        daemon's) and ``_recording`` would contain duplicate copies
        the next replay would render twice.
        """
        task_id = "cli-live-task-2"
        self.server._printer.subscribe_tab(task_id, "tab-viewer")
        _reader, received, got = self._open_viewer()
        self._wait_for_uds_writer(1)

        # Snapshot the daemon printer's recording BEFORE the CLI emits.
        recording_before = list(
            self.server._printer._recordings.get(task_id, ()),
        )

        cli_printer = RecordingConsolePrinter()
        cli_printer._thread_local.task_id = task_id
        cli_printer.broadcast({"type": "text_delta", "text": "no-dup"})

        assert got.wait(timeout=3.0), f"viewer got nothing: {received}"

        recording_after = list(
            self.server._printer._recordings.get(task_id, ()),
        )
        # The fan-out is purely a transport — the daemon's recording
        # must be unchanged.  (The CLI's own printer's recording
        # has the event, which is the source of truth replays use.)
        assert recording_after == recording_before, (
            "daemon double-recorded the CLI event: "
            f"before={recording_before} after={recording_after}"
        )


if __name__ == "__main__":
    unittest.main()
