# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The daemon side of the "Working directory" panel's opened-so-far list.

Every directory a client adopts as its working directory -- a VS Code
window's ``setWorkDir`` on connect, the remote webapp's ``saveConfig``
carrying ``work_dir`` -- passes through ``_apply_new_work_dir``, which
records it in ``config.json`` (``recent_work_dirs``: ``{path, ts}``
rows).  ``getConfig`` returns the rows most recently opened first,
skipping directories that no longer exist.  Tested against a real
``RemoteAccessServer`` over its UDS socket plus the config helpers on a
real temporary ``config.json``.

The webview half (the "..." menu item, the panel, both surfaces) is
covered by ``agents/vscode/test/workDirPanel.test.js``.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
import kiss.core.vscode_config as vc
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import requires_unix_sockets


class _IsolatedConfig:
    """Point ``vscode_config`` at a throwaway ``config.json``."""

    def enter(self, tmpdir: str) -> None:
        self._orig = (vc.CONFIG_DIR, vc.CONFIG_PATH)
        vc.CONFIG_DIR = Path(tmpdir) / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"

    def exit(self) -> None:
        vc.CONFIG_DIR, vc.CONFIG_PATH = self._orig


class TestRecentWorkDirHelpers(unittest.TestCase):
    """``record_recent_work_dir`` / ``recent_work_dirs`` on a real file."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.cfg = _IsolatedConfig()
        self.cfg.enter(self.tmpdir)
        self.dir_a = Path(self.tmpdir) / "a"
        self.dir_b = Path(self.tmpdir) / "b"
        self.dir_a.mkdir()
        self.dir_b.mkdir()

    def tearDown(self) -> None:
        self.cfg.exit()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _paths(self) -> list[str]:
        return [e["path"] for e in vc.recent_work_dirs()]

    def test_empty_when_nothing_recorded(self) -> None:
        self.assertEqual(vc.recent_work_dirs(), [])

    def test_reopening_moves_to_front_with_a_newer_stamp(self) -> None:
        vc.record_recent_work_dir(str(self.dir_a))
        time.sleep(0.01)
        vc.record_recent_work_dir(str(self.dir_b))
        self.assertEqual(self._paths(), [str(self.dir_b), str(self.dir_a)])
        first_a_ts = vc.recent_work_dirs()[1]["ts"]

        time.sleep(0.01)
        vc.record_recent_work_dir(str(self.dir_a))
        rows = vc.recent_work_dirs()
        self.assertEqual(
            [e["path"] for e in rows], [str(self.dir_a), str(self.dir_b)],
        )
        self.assertGreater(rows[0]["ts"], first_a_ts)
        self.assertGreater(rows[0]["ts"], rows[1]["ts"])
        # One row per directory, persisted in config.json alongside the
        # other settings.
        stored = json.loads(vc.CONFIG_PATH.read_text())["recent_work_dirs"]
        self.assertEqual(len(stored), 2)

    def test_recording_keeps_other_settings(self) -> None:
        vc.save_config({"max_budget": 7, "work_dir": str(self.dir_a)})
        vc.record_recent_work_dir(str(self.dir_a))
        cfg = vc.load_config()
        self.assertEqual(cfg["max_budget"], 7)
        self.assertEqual(cfg["work_dir"], str(self.dir_a))
        self.assertEqual(self._paths(), [str(self.dir_a)])

    def test_non_directories_are_not_recorded(self) -> None:
        missing = Path(self.tmpdir) / "missing"
        a_file = Path(self.tmpdir) / "file.txt"
        a_file.write_text("x")
        vc.record_recent_work_dir(str(missing))
        vc.record_recent_work_dir(str(a_file))
        vc.record_recent_work_dir("")
        self.assertEqual(vc.recent_work_dirs(), [])
        self.assertFalse(vc.CONFIG_PATH.exists())

    def test_deleted_directories_disappear_from_the_list(self) -> None:
        vc.record_recent_work_dir(str(self.dir_a))
        vc.record_recent_work_dir(str(self.dir_b))
        shutil.rmtree(self.dir_a)
        self.assertEqual(self._paths(), [str(self.dir_b)])

    def test_malformed_stored_rows_are_skipped_and_order_is_by_ts(self) -> None:
        vc.CONFIG_DIR.mkdir(parents=True)
        vc.CONFIG_PATH.write_text(json.dumps({
            "recent_work_dirs": [
                {"path": str(self.dir_a), "ts": 10},
                {"path": str(self.dir_b), "ts": 20.5},
                {"path": str(self.dir_b)},
                {"path": 5, "ts": 30},
                {"ts": 40},
                "junk",
                None,
                {"path": str(self.dir_a), "ts": "31"},
            ],
        }))
        rows = vc.recent_work_dirs()
        self.assertEqual(
            rows,
            [
                {"path": str(self.dir_b), "ts": 20.5},
                {"path": str(self.dir_a), "ts": 10.0},
            ],
        )

    def test_huge_and_non_finite_timestamps_are_skipped(self) -> None:
        """``10**1000`` overflows ``float()``; ``1e309`` parses as
        ``inf`` and would leave ``configData`` unparseable in the
        browser (``json.dumps`` emits ``Infinity``); booleans are not
        stamps.  None of them may poison the list."""
        vc.CONFIG_DIR.mkdir(parents=True)
        vc.CONFIG_PATH.write_text(
            '{"recent_work_dirs": ['
            f'{{"path": {json.dumps(str(self.dir_a))}, "ts": {10**1000}}}, '
            f'{{"path": {json.dumps(str(self.dir_a))}, "ts": 1e309}}, '
            f'{{"path": {json.dumps(str(self.dir_a))}, "ts": NaN}}, '
            f'{{"path": {json.dumps(str(self.dir_a))}, "ts": true}}, '
            f'{{"path": {json.dumps(str(self.dir_b))}, "ts": 5}}'
            "]}"
        )
        rows = vc.recent_work_dirs()
        self.assertEqual(rows, [{"path": str(self.dir_b), "ts": 5.0}])
        json.loads(json.dumps(rows, allow_nan=False))

    def test_non_list_value_is_ignored(self) -> None:
        vc.CONFIG_DIR.mkdir(parents=True)
        vc.CONFIG_PATH.write_text(json.dumps({"recent_work_dirs": "nope"}))
        self.assertEqual(vc.recent_work_dirs(), [])
        # Recording repairs the key.
        vc.record_recent_work_dir(str(self.dir_a))
        self.assertEqual(self._paths(), [str(self.dir_a)])

    def test_list_is_capped(self) -> None:
        dirs = []
        for i in range(vc.MAX_RECENT_WORK_DIRS + 3):
            d = Path(self.tmpdir) / f"d{i:02d}"
            d.mkdir()
            dirs.append(str(d))
            vc.record_recent_work_dir(str(d))
        paths = self._paths()
        self.assertEqual(len(paths), vc.MAX_RECENT_WORK_DIRS)
        # The most recent survive, the oldest three are gone.
        self.assertEqual(paths[0], dirs[-1])
        for gone in dirs[:3]:
            self.assertNotIn(gone, paths)


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore_persistence(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


@requires_unix_sockets
class TestRecentWorkDirsOverUds(IsolatedAsyncioTestCase):
    """``setWorkDir`` / ``saveConfig`` record, ``getConfig`` reports."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)
        self.cfg = _IsolatedConfig()
        self.cfg.enter(self.tmpdir)

        self.dir_a = Path(self.tmpdir) / "ws_a"
        self.dir_b = Path(self.tmpdir) / "ws_b"
        self.dir_a.mkdir()
        self.dir_b.mkdir()

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        from kiss.server.web_server import _generate_self_signed_cert
        _generate_self_signed_cert(certfile, keyfile)

        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        await self.server.start_async()
        self._writers: list[asyncio.StreamWriter] = []

    async def asyncTearDown(self) -> None:
        for writer in self._writers:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        self.cfg.exit()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path), limit=16 * 1024 * 1024,
        )
        self._writers.append(writer)
        return reader, writer

    async def _send(
        self, writer: asyncio.StreamWriter, cmd: dict[str, Any],
    ) -> None:
        writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await writer.drain()

    async def _drain_until(
        self,
        reader: asyncio.StreamReader,
        predicate: Callable[[dict[str, Any]], bool],
        max_events: int = 100,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        for _ in range(max_events):
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            assert line, "UDS closed unexpectedly"
            msg = json.loads(line.decode("utf-8"))
            assert isinstance(msg, dict)
            if predicate(msg):
                return msg
        raise AssertionError(
            f"predicate never matched within {max_events} events",
        )

    async def _recent_paths(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> list[str]:
        await self._send(writer, {"type": "getConfig"})
        msg = await self._drain_until(
            reader, lambda m: m.get("type") == "configData",
        )
        rows = msg["config"]["recent_work_dirs"]
        self.assertEqual(
            rows, sorted(rows, key=lambda e: e["ts"], reverse=True),
        )
        return [e["path"] for e in rows]

    async def _wait_recorded(self, path: str) -> None:
        # setWorkDir has no reply; the record lands right after the
        # command is handled on the daemon's thread.
        for _ in range(200):
            if any(e["path"] == path for e in vc.recent_work_dirs()):
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"{path} never recorded")

    async def test_windows_and_webapp_share_one_history(self) -> None:
        """Two VS Code windows announce their folders (``setWorkDir``);
        the remote webapp then picks one of them (``saveConfig``): every
        ``getConfig`` lists the shared history, most recent first."""
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()
        await self._send(
            writer_a, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        await self._wait_recorded(str(self.dir_a))
        await asyncio.sleep(0.02)
        await self._send(
            writer_b, {"type": "setWorkDir", "workDir": str(self.dir_b)},
        )
        await self._wait_recorded(str(self.dir_b))

        self.assertEqual(
            await self._recent_paths(reader_a, writer_a),
            [str(self.dir_b), str(self.dir_a)],
        )
        # A fresh connection (the remote webapp) sees the same list.
        reader_c, writer_c = await self._connect()
        self.assertEqual(
            await self._recent_paths(reader_c, writer_c),
            [str(self.dir_b), str(self.dir_a)],
        )

        await asyncio.sleep(0.02)
        await self._send(
            writer_c,
            {"type": "saveConfig", "config": {"work_dir": str(self.dir_a)}},
        )
        await self._drain_until(
            reader_c,
            lambda m: (
                m.get("type") == "configData"
                and m.get("config", {}).get("work_dir") == str(self.dir_a)
            ),
        )
        self.assertEqual(
            await self._recent_paths(reader_c, writer_c),
            [str(self.dir_a), str(self.dir_b)],
        )

    async def test_deleted_directory_is_not_reported(self) -> None:
        reader, writer = await self._connect()
        await self._send(
            writer, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        await self._wait_recorded(str(self.dir_a))
        shutil.rmtree(self.dir_a)
        self.assertEqual(await self._recent_paths(reader, writer), [])

    async def test_filesystem_root_is_never_recorded(self) -> None:
        """``_apply_new_work_dir`` refuses a root before recording it."""
        self.server._vscode_server._apply_new_work_dir("/")
        self.assertEqual(vc.recent_work_dirs(), [])
        reader, writer = await self._connect()
        self.assertEqual(await self._recent_paths(reader, writer), [])
