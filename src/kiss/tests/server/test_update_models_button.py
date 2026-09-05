# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``updateModels`` daemon command.

The settings panel's "Update Models" button (5th button next to Tips /
Git Commit / Update / Reset Server) sends ``{type: 'updateModels'}``: the
VS Code webview forwards it to the daemon (``FORWARDED_COMMANDS`` in
``SorcarSidebarView.ts``) and remote browser windows send it directly.
The daemon runs ``kiss.scripts.update_models --model-info
$KISS_HOME/MODEL_INFO.json`` as a detached subprocess logging to
``~/.kiss/update_models.log``, guards against concurrent runs, and
reports the exit to the clicking window only.

Every test drives the REAL :class:`RemoteAccessServer` over its UDS
protocol (same harness as
``test_audit0902_fix2_vscode_update_exit_report``) with the updater argv
pointed at real stub scripts — no mocks or doubles.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert

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


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


def _has_type(reply_type: str) -> Callable[[dict[str, Any]], bool]:
    def _pred(msg: dict[str, Any]) -> bool:
        return msg.get("type") == reply_type

    return _pred


class TestUpdateModelsCommand(IsolatedAsyncioTestCase):
    """The daemon runs the catalog updater and reports its exit."""

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
        self.log_path = Path(self.tmpdir) / "update_models.log"
        self.server._update_models_log_path = self.log_path
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

    def _updater_stub(self, body: str) -> None:
        """Point the daemon's updater argv at a real stub script."""
        script = Path(self.tmpdir) / "updater_stub.py"
        script.write_text(body)
        self.server._update_models_argv = [sys.executable, str(script)]

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
        self.assertIsNone(self.server._update_models_watch_task)
        self.assertFalse(self.server._update_models_starting)

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

    async def test_default_argv_targets_the_user_catalog(self) -> None:
        """The production argv runs update_models against $KISS_HOME."""
        argv = self.server._update_models_argv
        self.assertEqual(argv[0], sys.executable)
        self.assertEqual(argv[1:3], ["-m", "kiss.scripts.update_models"])
        self.assertEqual(argv[3], "--model-info")
        self.assertTrue(argv[4].endswith("MODEL_INFO.json"))
