# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``updateModels`` daemon command.

The settings panel's "Update Models" button (5th button next to Tips /
Git Commit / Update / Reset Server) sends ``{type: 'updateModels'}``
straight to the daemon over the WebSocket.  The daemon runs
``kiss.scripts.update_models --model-info $KISS_HOME/MODEL_INFO.json``
as a detached subprocess logging to
``~/.kiss/update_models.log``, guards against concurrent runs, and
reports the exit to the clicking window only.

Every test drives the REAL :class:`RemoteAccessServer` over its local
WSS channel (same harness as
``test_audit0902_fix2_update_exit_report``) with the updater argv
pointed at real stub scripts — no mocks or doubles.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase, skipUnless

import kiss.agents.sorcar.persistence as th
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import nproc_limit_lowered_to_one, thread_start_can_be_starved
from kiss.tests.local_ws import _generate_self_signed_cert, open_local_connection
from kiss.tests.server.conftest import find_free_port
from kiss.tests.server.pause import Pause

SUCCEEDING_UPDATER = """
import sys
print("catalog refreshed")
sys.exit(0)
"""

FAILING_UPDATER = """
import sys
print("boom")
sys.exit(7)
"""

# Blocks until the test creates the release file, then succeeds.
HELD_UPDATER = """
import pathlib, sys, time
while not pathlib.Path({release!r}).exists():
    time.sleep(0.05)
sys.exit(0)
"""


def _has_type(reply_type: str) -> Callable[[dict[str, Any]], bool]:
    def _pred(msg: dict[str, Any]) -> bool:
        return msg.get("type") == reply_type

    return _pred


class TestUpdateModelsCommand(IsolatedAsyncioTestCase):
    """The daemon runs the catalog updater and reports its exit."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=find_free_port(),
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        self.log_path = Path(self.tmpdir) / "update_models.log"
        self.server._update_models_log_path = self.log_path
        self.release = Path(self.tmpdir) / "release"
        await self.server.start_async()
        self._writers: list[Any] = []

    async def asyncTearDown(self) -> None:
        self.release.write_text("")
        for writer in self._writers:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        await self.server.stop_async()
        # The updater is detached by design and may outlive stop_async
        # (whose cancelled watcher no longer reaps it); wait for the
        # released stub to exit so it does not outlive the test.
        proc = self.server._update_models_proc
        if proc is not None:
            await asyncio.to_thread(proc.wait, 10)
        th._close_thread_db()
        th._DB_PATH, th._KISS_DIR = self.saved  # type: ignore[assignment]
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _updater_stub(self, body: str) -> None:
        """Point the daemon's updater argv at a real stub script."""
        script = Path(self.tmpdir) / "updater_stub.py"
        script.write_text(body)
        self.server._update_models_argv = [sys.executable, str(script)]

    async def _connect(self) -> tuple[Any, Any]:
        reader, writer = await open_local_connection(
            self.server, limit=16 * 1024 * 1024,
        )
        self._writers.append(writer)
        return reader, writer

    async def _send(self, writer: Any, cmd: dict[str, Any]) -> None:
        writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await writer.drain()

    async def _drain_until(
        self,
        reader: Any,
        predicate: Callable[[dict[str, Any]], bool],
        max_events: int = 100,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        for _ in range(max_events):
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            assert line, "local connection closed unexpectedly"
            msg = json.loads(line.decode("utf-8"))
            assert isinstance(msg, dict)
            if predicate(msg):
                return msg
        raise AssertionError(f"predicate never matched within {max_events} events")

    async def _banners_before_probe(
        self, reader: Any, writer: Any,
    ) -> list[dict[str, Any]]:
        """Return the notice/error events queued on a window before a probe."""
        await self._send(writer, {"type": "activeTasksQuery"})
        seen: list[dict[str, Any]] = []

        def _probe(msg: dict[str, Any]) -> bool:
            seen.append(msg)
            return msg.get("type") == "activeTasksResponse"

        await self._drain_until(reader, _probe)
        return [m for m in seen if m.get("type") in ("notice", "error")]

    async def _wait_for_spawn(self) -> asyncio.Task[None]:
        for _ in range(400):
            task = self.server._update_models_watch_task
            if task is not None:
                return task
            await asyncio.sleep(0.025)
        raise AssertionError("updater never spawned")

    async def test_success_notifies_the_clicking_window_only(self) -> None:
        self._updater_stub(SUCCEEDING_UPDATER)
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()

        await self._send(writer_a, {"type": "updateModels"})
        notice = await self._drain_until(reader_a, _has_type("notice"))
        self.assertIn("Updating the model catalog", str(notice.get("text", "")))
        done = await self._drain_until(reader_a, _has_type("notice"))
        self.assertIn(
            "Model catalog update complete", str(done.get("text", ""))
        )
        self.assertIn("catalog refreshed", self.log_path.read_text())
        proc = self.server._update_models_proc
        assert proc is not None
        self.assertEqual(proc.returncode, 0)
        # The other window saw none of it.
        self.assertEqual(await self._banners_before_probe(reader_b, writer_b), [])

    async def test_failure_reports_the_exit_code_and_log_path(self) -> None:
        self._updater_stub(FAILING_UPDATER)
        reader_a, writer_a = await self._connect()

        await self._send(writer_a, {"type": "updateModels"})
        await self._drain_until(reader_a, _has_type("notice"))
        err = await self._drain_until(reader_a, _has_type("error"))
        text = str(err.get("text", ""))
        self.assertIn("Model catalog update failed (exit 7)", text)
        self.assertIn(str(self.log_path), text)
        self.assertIn("boom", self.log_path.read_text())

    async def test_second_click_while_running_is_refused(self) -> None:
        self._updater_stub(HELD_UPDATER.format(release=str(self.release)))
        reader_a, writer_a = await self._connect()

        await self._send(writer_a, {"type": "updateModels"})
        await self._drain_until(reader_a, _has_type("notice"))
        await self._wait_for_spawn()
        proc = self.server._update_models_proc
        assert proc is not None
        self.assertIsNone(proc.poll())

        await self._send(writer_a, {"type": "updateModels"})
        second = await self._drain_until(reader_a, _has_type("notice"))
        self.assertIn(
            "already running", str(second.get("text", ""))
        )
        self.release.write_text("")
        done = await self._drain_until(reader_a, _has_type("notice"))
        self.assertIn("complete", str(done.get("text", "")))

    def _is_updater_delivery(self, loc: dict[str, Any]) -> bool:
        """Pause matcher: an executor future receiving a started updater, kept in ``_delivered``."""
        if isinstance(loc["result"], subprocess.Popen):
            self._delivered = loc["result"]
            return True
        return False

    async def test_click_during_the_spawn_hand_over_is_refused(self) -> None:
        """A click landing before the loop took over a finished spawn is refused.

        Window A's updater has exited by the time its worker thread
        delivers it.  A click admitted then started a second updater,
        and window A's handler put its own finished watcher over the
        live one's handle, which ``stop_async`` then never cancelled.
        """
        self._updater_stub(SUCCEEDING_UPDATER)
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()
        # Holds window A's worker between the spawn's return and the
        # hand-over of its result to the event loop.
        with Pause(Future.set_result.__code__, self._is_updater_delivery) as pause:
            await self._send(writer_a, {"type": "updateModels"})
            self.assertTrue(await asyncio.to_thread(pause.paused.wait, 30))
            await asyncio.to_thread(self._delivered.wait, 10)
            self._updater_stub(HELD_UPDATER.format(release=str(self.release)))
            await self._send(writer_b, {"type": "updateModels"})
            second = await self._drain_until(reader_b, _has_type("notice"))
            self.assertIn("already running", str(second.get("text", "")))
        await self._drain_until(reader_a, _has_type("notice"))
        done = await self._drain_until(reader_a, _has_type("notice"))
        self.assertIn("complete", str(done.get("text", "")))
        self.assertIs(self.server._update_models_proc, self._delivered)
        self.assertFalse(self.server._update_models_starting)

    async def test_spawn_failure_is_reported_and_starts_no_watcher(self) -> None:
        self._updater_stub(SUCCEEDING_UPDATER)
        blocker = Path(self.tmpdir) / "not-a-dir"
        blocker.write_text("")
        self.server._update_models_log_path = blocker / "update_models.log"
        reader_a, writer_a = await self._connect()

        await self._send(writer_a, {"type": "updateModels"})
        await self._drain_until(reader_a, _has_type("notice"))
        err = await self._drain_until(reader_a, _has_type("error"))
        self.assertIn(
            "Failed to start the model catalog update", str(err.get("text", ""))
        )
        # The spawn thread reports the failure before the loop's
        # done-callback releases the single-flight guard, so the error
        # can be read first.  Commands on one connection are dispatched
        # sequentially: a probe round-trip proves _handle_update_models
        # has finished.
        self.assertEqual(await self._banners_before_probe(reader_a, writer_a), [])
        self.assertIsNone(self.server._update_models_watch_task)
        self.assertFalse(self.server._update_models_starting)

    async def test_a_refused_spawn_submission_releases_the_guard(self) -> None:
        """A click whose spawn the executor refused left every later one "already running"."""
        self._updater_stub(SUCCEEDING_UPDATER)
        reader_a, writer_a = await self._connect()
        loop = asyncio.get_running_loop()
        pool = ThreadPoolExecutor()
        loop.set_default_executor(pool)
        pool.shutdown()
        await self._send(writer_a, {"type": "updateModels"})
        # The handler sends this notice and submits its spawn in one loop step.
        first = await self._drain_until(reader_a, _has_type("notice"))
        self.assertIn("Updating the model catalog", str(first.get("text", "")))
        loop.set_default_executor(ThreadPoolExecutor())
        await self._send(writer_a, {"type": "updateModels"})
        second = await self._drain_until(reader_a, _has_type("notice"))
        self.assertIn("Updating the model catalog", str(second.get("text", "")))
        done = await self._drain_until(reader_a, _has_type("notice"))
        self.assertIn("complete", str(done.get("text", "")))

    def _note_updater(self, loc: dict[str, Any]) -> bool:
        """Pause matcher that holds nothing: every updater a future receives, in ``_updaters``."""
        if isinstance(loc["result"], subprocess.Popen):
            self._updaters.append(loc["result"])
        return False

    @skipUnless(thread_start_can_be_starved(), "RLIMIT_NPROC does not starve threads here")
    async def test_a_spawn_queued_by_a_refused_submission_never_runs(self) -> None:
        """A thread that could not start left the spawn queued: it ran beside the next click's."""
        fifo = Path(self.tmpdir) / "hold"
        os.mkfifo(fifo)
        self._updater_stub(f"open({str(fifo)!r}, 'rb', 0).read(1)\n")
        self._updaters: list[subprocess.Popen[bytes]] = []
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        with open(fifo, "r+b", buffering=0) as hold:
            try:
                with Pause(Future.set_result.__code__, self._note_updater):
                    with nproc_limit_lowered_to_one(), self.assertRaises(RuntimeError):
                        await self.server._handle_update_models("")
                    # The same executor can start its one thread now: the queued job runs first.
                    await self.server._handle_update_models("")
                self.assertEqual(self._updaters, [self.server._update_models_proc])
                self.assertFalse(self.server._update_models_starting)
            finally:
                hold.write(b"xx")  # a byte lets one stand-in go
                for updater in self._updaters:
                    await asyncio.to_thread(updater.wait, 10)

    async def test_stop_cancels_a_pending_watcher(self) -> None:
        self._updater_stub(HELD_UPDATER.format(release=str(self.release)))
        reader_a, writer_a = await self._connect()
        await self._send(writer_a, {"type": "updateModels"})
        await self._drain_until(reader_a, _has_type("notice"))
        task = await self._wait_for_spawn()
        self.assertFalse(task.done())
        await self.server.stop_async()
        self.assertTrue(task.cancelled())
        self.assertIsNone(self.server._update_models_watch_task)

    async def test_stop_during_the_spawn_installs_no_watcher(self) -> None:
        """A spawn still in flight when ``stop_async`` returns installs no watcher.

        The handler is awaited inline by the connection task, which the
        listener close does not cancel, so the spawn completes after the
        stop; a watcher created then would poll the detached updater on
        the stopped server's loop for the updater's whole lifetime.
        """
        self._updater_stub(HELD_UPDATER.format(release=str(self.release)))
        reader_a, writer_a = await self._connect()
        with Pause(
            RemoteAccessServer._spawn_detached.__code__,
            lambda loc: loc["self"] is self.server,
        ) as pause:
            await self._send(writer_a, {"type": "updateModels"})
            await self._drain_until(reader_a, _has_type("notice"))
            self.assertTrue(await asyncio.to_thread(pause.paused.wait, 30))
            await self.server.stop_async()
            pause.release()
        for _ in range(80):
            await asyncio.sleep(0.025)
            if self.server._update_models_watch_task is not None:
                break
        self.assertIsNotNone(self.server._update_models_proc, "the released spawn never ran")
        self.assertIsNone(
            self.server._update_models_watch_task,
            "regression: a watcher was installed after stop_async returned",
        )

    async def test_default_argv_targets_the_user_catalog(self) -> None:
        """The production argv runs update_models against $KISS_HOME."""
        argv = self.server._update_models_argv
        self.assertEqual(argv[0], sys.executable)
        self.assertEqual(argv[1:3], ["-m", "kiss.scripts.update_models"])
        self.assertEqual(argv[3], "--model-info")
        self.assertTrue(argv[4].endswith("MODEL_INFO.json"))
