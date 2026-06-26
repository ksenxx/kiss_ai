# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Reproducer: ``sorcar`` interactive CLI must not show
"Daemon connection lost" right after submitting a real task prompt.

User-visible bug
----------------
Running ``sorcar`` (interactive) and typing a task (e.g. ``hi``) the
user sees::

    ✗ Daemon connection lost — type /exit to quit

That message is rendered by :func:`_submit_task` whenever
``client._closed`` becomes set during the wait — i.e. the UDS
connection to the ``sorcar web`` daemon was torn down mid-task.

Root cause
----------
:class:`CliClient` reads events with :meth:`asyncio.StreamReader.readline`.
The reader is created by :func:`asyncio.open_unix_connection` with the
default 64 KiB buffer limit.  The daemon emits a ``system_prompt``
event at the start of every task that includes the entire ``SYSTEM.md``
plus injections — well over 64 KiB on a single JSON line.  The first
``readline()`` therefore raises :class:`asyncio.LimitOverrunError`,
which is NOT caught by ``_main``'s ``except (CancelledError,
ConnectionError)`` clause, the coroutine returns, ``_run_loop``'s
``finally`` sets ``client._closed``, and the next call to
:func:`_submit_task` prints "Daemon connection lost".

What this test asserts
----------------------
1.  ``test_oversize_event_does_not_drop_uds_connection`` —
    spins a real :class:`RemoteAccessServer` on a temp UDS, connects a
    real :class:`CliClient`, then broadcasts a single event larger
    than 64 KiB and asserts the client's UDS connection is NOT torn
    down.  This is the **direct** reproducer of the bug.
2.  ``test_run_hi_does_not_drop_uds_connection`` —
    additionally submits a real ``run`` command with prompt ``hi`` and
    asserts the daemon does not drop the UDS, exercising the same
    code path through the production ``_submit_task`` entry point.
"""

from __future__ import annotations

import asyncio
import gc
import os
import shutil
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path

from kiss.agents.sorcar import cli_daemon_bridge
from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.cli_client import CliClient, _submit_task
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.web_server import RemoteAccessServer
from kiss.core.print_to_console import ConsolePrinter


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


class TestDaemonConnectionLostRepro(unittest.TestCase):
    """End-to-end reproducer for the daemon-connection-lost bug."""

    def setUp(self) -> None:
        # Every resource is registered with ``addCleanup`` the moment
        # it is acquired.  ``addCleanup`` runs in LIFO order regardless
        # of whether ``setUp`` completes — so a failure half-way
        # through still releases the loop thread, the UDS listener,
        # the asyncio event loop FDs, the devnull FD, and the tmpdir.
        # Pre-tightening, a partial setUp left every one of those
        # leaked, which compounded into ``OSError [Errno 24] Too many
        # open files`` once a few hundred CLI tests had run in the
        # same process.
        self.tmpdir = tempfile.mkdtemp(prefix="sorcar_cli_repro_")
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        # Final ``gc.collect()`` reclaims any selector / transport FDs
        # whose only references were on now-closed asyncio tasks.
        self.addCleanup(gc.collect)

        self.sock_path = str(Path(self.tmpdir) / "sorcar.sock")
        self.work_dir = str(Path(self.tmpdir) / "wd")
        os.makedirs(self.work_dir, exist_ok=True)

        # Isolate sqlite persistence under the temp kiss-home.
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self._saved_persistence = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        self.addCleanup(self._restore_persistence)
        # Drop any tabs the test may have inserted into the shared
        # process-wide registry so they cannot bleed into later tests.
        self.addCleanup(_RunningAgentState.running_agent_states.clear)

        self._saved_env = os.environ.get("KISS_SORCAR_SOCK")
        os.environ["KISS_SORCAR_SOCK"] = self.sock_path
        _reset_cli_daemon_writer()
        self.addCleanup(_reset_cli_daemon_writer)
        self.addCleanup(self._restore_sock_env)

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True,
        )
        self.loop_thread.start()
        # Stop / close the loop LAST (after every coroutine-based
        # cleanup has already run on it).  LIFO ordering means we
        # register this BEFORE we register the asyncio shutdown.
        self.addCleanup(self._stop_and_close_loop)

        self.server = RemoteAccessServer(
            uds_path=self.sock_path, work_dir=self.work_dir,
        )
        self.server._printer._loop = self.loop
        self.server._loop = self.loop
        self.uds_server: asyncio.Server = asyncio.run_coroutine_threadsafe(
            asyncio.start_unix_server(
                self.server._uds_handler, path=self.sock_path,
            ),
            self.loop,
        ).result(timeout=5)
        # Aggressive async shutdown runs before ``_stop_and_close_loop``
        # because addCleanup is LIFO.
        self.addCleanup(self._run_async_shutdown)

        self._devnull = open(os.devnull, "w")
        self.addCleanup(self._safe_close_file, self._devnull)
        self.printer = ConsolePrinter(file=self._devnull)
        self.tab_id = uuid.uuid4().hex
        self.client = CliClient(
            sock_path=Path(self.sock_path),
            work_dir=self.work_dir,
            tab_id=self.tab_id,
            printer=self.printer,
        )
        # Send a courtesy ``stop`` + ``close`` to the daemon before
        # ripping the loop down — the client's UDS writer must be
        # closed FIRST so the server-side ``_uds_handler`` exits its
        # ``readline`` await with EOF and stops referencing the
        # transport.
        self.addCleanup(self._close_client)
        self.client.start(timeout=5.0)

    def _restore_persistence(self) -> None:
        th._DB_PATH, th._db_conn, th._KISS_DIR = self._saved_persistence

    def _restore_sock_env(self) -> None:
        if self._saved_env is None:
            os.environ.pop("KISS_SORCAR_SOCK", None)
        else:
            os.environ["KISS_SORCAR_SOCK"] = self._saved_env

    @staticmethod
    def _safe_close_file(fh: object) -> None:
        try:
            fh.close()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    def _close_client(self) -> None:
        """Tell the daemon to stop the task and tear down the client UDS.

        We send ``stop`` defensively (it is a no-op if no task is
        running) and then call :meth:`CliClient.close` which closes
        the client-side StreamWriter, joins the loop thread, and
        closes the client's own asyncio loop.  Errors at any step are
        swallowed: this is best-effort teardown.
        """
        try:
            self.client.send({"type": "stop", "tabId": self.tab_id})
        except Exception:  # noqa: BLE001
            pass
        try:
            self.client.close()
        except Exception:  # noqa: BLE001
            pass

    def _run_async_shutdown(self) -> None:
        """Drive the server-side async shutdown on the harness loop.

        Closes every registered UDS writer with ``await wait_closed()``
        so the underlying socket FDs are released before the loop
        closes — a plain ``writer.close()`` only schedules the close;
        the transport's FD is not freed until the next loop tick, which
        never runs if we stop the loop too eagerly.
        """
        async def _shutdown() -> None:
            with self.server._printer._ws_lock:
                writers = list(self.server._printer._uds_writers)
                # Drop the references now — any subsequent broadcast
                # will be a no-op rather than writing into a dying
                # transport.
                self.server._printer._uds_writers.clear()
            for writer in writers:
                try:
                    writer.close()
                except Exception:  # noqa: BLE001
                    pass
            for writer in writers:
                try:
                    await writer.wait_closed()
                except Exception:  # noqa: BLE001
                    pass
            try:
                self.uds_server.close()
                await self.uds_server.wait_closed()
            except Exception:  # noqa: BLE001
                pass
            pending = [
                t for t in asyncio.all_tasks()
                if t is not asyncio.current_task()
            ]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        if not self.loop.is_closed() and self.loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(
                    _shutdown(), self.loop,
                ).result(timeout=5)
            except Exception:  # noqa: BLE001
                pass

    def _stop_and_close_loop(self) -> None:
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:  # noqa: BLE001
            pass
        self.loop_thread.join(timeout=5)
        try:
            if not self.loop.is_closed():
                self.loop.close()
        except Exception:  # noqa: BLE001
            pass

    def _wait_for_uds_writers(self, timeout: float = 5.0) -> None:
        """Block until the daemon has registered the CLI's UDS writer.

        ``RemoteAccessServer._uds_handler`` calls
        :meth:`WebPrinter.add_uds_writer` only after it has read the
        client's first command.  The CLI sends ``setWorkDir`` /
        ``ready`` from ``_main`` shortly after ``start()`` returns;
        wait briefly for the registration to complete before
        broadcasting test events.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.server._printer._ws_lock:
                if self.server._printer._uds_writers:
                    return
            time.sleep(0.02)
        raise AssertionError("daemon never registered the CLI UDS writer")

    def test_oversize_event_does_not_drop_uds_connection(self) -> None:
        """A >64 KiB broadcast must NOT tear down the CLI UDS reader."""
        self._wait_for_uds_writers()

        # Inject a single oversize event (~120 KiB) targeted at the
        # CLI's tab.  This mirrors the daemon's real ``system_prompt``
        # broadcast which is too large for asyncio's default 64 KiB
        # StreamReader buffer.
        big = "A" * 120_000
        self.server._printer.broadcast({
            "type": "text_delta",
            "tabId": self.tab_id,
            "text": big,
        })

        # Allow the asyncio reader plenty of time to consume the line.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self.client._closed.is_set():
                break
            time.sleep(0.05)

        self.assertFalse(
            self.client._closed.is_set(),
            "CliClient UDS reader closed after receiving a single "
            ">64 KiB broadcast — `Daemon connection lost` would be "
            "shown to the user.  This is the asyncio default-readline-"
            "limit bug.",
        )

    def test_run_hi_does_not_drop_uds_connection(self) -> None:
        """Submitting ``run`` with prompt ``hi`` must keep the UDS up."""
        self._wait_for_uds_writers()

        submit_thread = threading.Thread(
            target=_submit_task,
            args=(self.client, "hi"),
            kwargs={
                "use_worktree": False,
                "use_parallel": False,
                "auto_commit": False,
                "timeout_seconds": 3.0,
            },
            daemon=True,
        )
        submit_thread.start()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self.client._closed.is_set():
                break
            time.sleep(0.05)

        self.assertFalse(
            self.client._closed.is_set(),
            "Daemon closed the UDS connection in response to a "
            "plain `run` command — `Daemon connection lost` message "
            "would be shown to the user.",
        )

        submit_thread.join(timeout=10.0)
        self.assertFalse(
            submit_thread.is_alive(),
            "_submit_task did not return within its timeout budget.",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
