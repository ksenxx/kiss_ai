# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: the daemon reports a failed ``runUpdate`` installer.

Review 2026-09-02 round 2 (review2-vscode.md #5): ``runUpdate`` told the
requesting window that an update "is getting installed", spawned
``install.sh`` detached and never looked at it again.  When the installer
lost the cross-process update lock it exited 1 with ``another KISS update
is already running (pid N); exiting.`` -- visible only in ``update.log`` --
and the window kept believing an update was under way.

The fix keeps the installer detached (``start_new_session=True``: it may
restart this very daemon) but watches its exit from an asyncio task that
polls ``Popen.poll`` without blocking the event loop or an executor
thread.  A non-zero exit is reported to the requesting connection only:
with the installer's own refusal line when this run's slice of
``update.log`` contains one, otherwise as a generic ``update failed (exit
N)`` pointing at the log.  A clean exit reports nothing more.

Every test drives the REAL :class:`RemoteAccessServer` over its UDS
protocol from two client connections (two VS Code windows) and runs a
real stub ``install.sh``.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert

REFUSAL = "another KISS update is already running (pid 123); exiting."

REFUSING_INSTALL_SH = f"""#!/bin/bash
echo "starting"
echo "{REFUSAL}" >&2
exit 1
"""

FAILING_INSTALL_SH = """#!/bin/bash
echo "ERROR: Node.js, npm, and npx are required to build the extension."
exit 3
"""

SUCCEEDING_INSTALL_SH = """#!/bin/bash
echo "=== Source bootstrap complete ==="
exit 0
"""

# Blocks until the test creates the release file, then fails.
HELD_INSTALL_SH = """#!/bin/bash
while [ ! -e "{release}" ]; do sleep 0.05; done
exit 5
"""


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


def _has_type(reply_type: str) -> Callable[[dict[str, Any]], bool]:
    def _pred(msg: dict[str, Any]) -> bool:
        return msg.get("type") == reply_type

    return _pred


class TestRunUpdateExitReport(IsolatedAsyncioTestCase):
    """The requesting window learns when the spawned installer fails."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=_find_free_port(),
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        self.install_root = Path(self.tmpdir) / "kiss_ai"
        self.install_root.mkdir()
        self.server._install_root = self.install_root
        self.log_path = Path(self.tmpdir) / "update.log"
        self.server._update_log_path = self.log_path
        self.release = Path(self.tmpdir) / "release"
        await self.server.start_async()
        self._writers: list[asyncio.StreamWriter] = []

    async def asyncTearDown(self) -> None:
        self.release.write_text("")
        for writer in self._writers:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved  # type: ignore[assignment]
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _install_stub(self, body: str) -> None:
        script = self.install_root / "install.sh"
        script.write_text(body)
        script.chmod(0o755)

    async def _connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path), limit=16 * 1024 * 1024,
        )
        self._writers.append(writer)
        return reader, writer

    async def _send(self, writer: asyncio.StreamWriter, cmd: dict[str, Any]) -> None:
        writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await writer.drain()

    async def _drain_until(
        self,
        reader: asyncio.StreamReader,
        predicate: Callable[[dict[str, Any]], bool],
        max_events: int = 100,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        for _ in range(max_events):
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            assert line, "UDS closed unexpectedly"
            msg = json.loads(line.decode("utf-8"))
            assert isinstance(msg, dict)
            if predicate(msg):
                return msg
        raise AssertionError(f"predicate never matched within {max_events} events")

    async def _banners_before_probe(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> list[dict[str, Any]]:
        """Return the notice/error events queued on a window before a probe.

        ``activeTasksQuery`` is answered directly, so everything the
        broadcast pipeline already queued for this window is read first.
        """
        await self._send(writer, {"type": "activeTasksQuery"})
        seen: list[dict[str, Any]] = []

        def _probe(msg: dict[str, Any]) -> bool:
            seen.append(msg)
            return msg.get("type") == "activeTasksResponse"

        await self._drain_until(reader, _probe)
        return [m for m in seen if m.get("type") in ("notice", "error")]

    async def _wait_for_spawn(self) -> asyncio.Task[None]:
        """Wait for the watcher task: the "getting installed" notice is
        sent BEFORE the executor spawns the installer, so the task may not
        exist yet when the notice arrives."""
        for _ in range(400):
            task = self.server._update_watch_task
            if task is not None:
                return task
            await asyncio.sleep(0.025)
        raise AssertionError("installer never spawned")

    async def _wait_for_watcher(self) -> None:
        """Wait until the installer has exited and its watcher has finished."""
        task = await self._wait_for_spawn()
        for _ in range(400):
            if task.done():
                return
            await asyncio.sleep(0.025)
        raise AssertionError("update watcher never finished")

    async def test_lock_refusal_is_reported_to_the_clicking_window_only(self) -> None:
        self._install_stub(REFUSING_INSTALL_SH)
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()

        await self._send(writer_a, {"type": "runUpdate"})
        notice = await self._drain_until(reader_a, _has_type("notice"))
        self.assertIn("is getting installed", str(notice.get("text", "")))
        err = await self._drain_until(reader_a, _has_type("error"))
        self.assertIn(REFUSAL, str(err.get("text", "")))
        self.assertNotIn("connId", err)
        # The installer's own output is still in the log.
        self.assertIn(REFUSAL, self.log_path.read_text())
        self.assertEqual(await self._banners_before_probe(reader_b, writer_b), [])

    async def test_other_failures_get_a_generic_report_pointing_at_the_log(self) -> None:
        self._install_stub(FAILING_INSTALL_SH)
        reader_a, writer_a = await self._connect()

        await self._send(writer_a, {"type": "runUpdate"})
        await self._drain_until(reader_a, _has_type("notice"))
        err = await self._drain_until(reader_a, _has_type("error"))
        text = str(err.get("text", ""))
        self.assertIn("update failed (exit 3)", text)
        self.assertIn(str(self.log_path), text)
        self.assertNotIn("already running", text)

    async def test_refusal_search_covers_only_this_run(self) -> None:
        # A refusal left in the log by an EARLIER run must not be blamed
        # on this one: only the slice appended by this installer counts.
        self.log_path.write_text(f"old run\n{REFUSAL}\n")
        self._install_stub(FAILING_INSTALL_SH)
        reader_a, writer_a = await self._connect()
        await self._send(writer_a, {"type": "runUpdate"})
        err = await self._drain_until(reader_a, _has_type("error"))
        self.assertIn("update failed (exit 3)", str(err.get("text", "")))

    async def test_clean_exit_reports_nothing_more(self) -> None:
        self._install_stub(SUCCEEDING_INSTALL_SH)
        reader_a, writer_a = await self._connect()
        await self._send(writer_a, {"type": "runUpdate"})
        await self._drain_until(reader_a, _has_type("notice"))
        await self._wait_for_watcher()
        proc = self.server._update_proc
        assert proc is not None
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(await self._banners_before_probe(reader_a, writer_a), [])
        self.assertIn("Source bootstrap complete", self.log_path.read_text())

    async def test_missing_log_still_reports_the_generic_failure(self) -> None:
        # The log vanishing under a running installer (a cleanup of
        # ~/.kiss) must not turn the report into an unhandled exception.
        self._install_stub(HELD_INSTALL_SH.format(release=self.release))
        reader_a, writer_a = await self._connect()
        await self._send(writer_a, {"type": "runUpdate"})
        await self._drain_until(reader_a, _has_type("notice"))
        await self._wait_for_spawn()
        proc = self.server._update_proc
        assert proc is not None
        self.assertIsNone(proc.poll())
        self.log_path.unlink()
        self.release.write_text("")
        err = await self._drain_until(reader_a, _has_type("error"))
        self.assertIn("update failed (exit 5)", str(err.get("text", "")))

    async def test_spawn_failure_is_reported_and_starts_no_watcher(self) -> None:
        # The update log's directory being a plain file makes the spawn
        # itself fail (mkdir raises): the error is reported as before and
        # there is no process to watch.
        self._install_stub(SUCCEEDING_INSTALL_SH)
        blocker = Path(self.tmpdir) / "not-a-dir"
        blocker.write_text("")
        self.server._update_log_path = blocker / "update.log"
        reader_a, writer_a = await self._connect()
        await self._send(writer_a, {"type": "runUpdate"})
        await self._drain_until(reader_a, _has_type("notice"))
        err = await self._drain_until(reader_a, _has_type("error"))
        self.assertIn("Failed to start KISS Sorcar update", str(err.get("text", "")))
        # Commands on one connection are dispatched sequentially, so a
        # probe round-trip proves _handle_run_update has finished.
        self.assertEqual(await self._banners_before_probe(reader_a, writer_a), [])
        self.assertIsNone(self.server._update_watch_task)
        self.assertFalse(self.server._update_starting)

    async def test_stop_cancels_a_pending_watcher(self) -> None:
        self._install_stub(HELD_INSTALL_SH.format(release=self.release))
        reader_a, writer_a = await self._connect()
        await self._send(writer_a, {"type": "runUpdate"})
        await self._drain_until(reader_a, _has_type("notice"))
        task = await self._wait_for_spawn()
        self.assertFalse(task.done())
        await self.server.stop_async()
        self.assertTrue(task.cancelled())
        self.assertIsNone(self.server._update_watch_task)
