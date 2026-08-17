# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The daemon side of the settings panel's "Working directory" option.

Tested against a real ``RemoteAccessServer`` over its UDS socket:

* ``getConfig`` reports each connection's (= each VS Code window's)
  own work_dir when nothing is persisted, and keeps preferring it over a
  work_dir another client persisted globally;
* a remote ``saveConfig`` carrying ``config.work_dir`` persists it to
  ``config.json`` and moves the daemon-wide fallback;
* a ``saveConfig`` that omits ``work_dir`` -- which is what a VS Code
  window sends, since its field is read-only -- leaves the persisted
  value alone.

The webview half of the field (read-only and never saved in VS Code,
editable, saved and re-pinned in the standalone web client) is covered
by the real DOM tests in
``agents/vscode/test/settingsWorkDirField.test.js``.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
import kiss.core.vscode_config as vc
from kiss.server.web_server import RemoteAccessServer


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


class TestWorkDirConfigRoundTrip(IsolatedAsyncioTestCase):
    """getConfig / saveConfig handle ``work_dir`` over real UDS."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)

        self._orig_cfg_dir = vc.CONFIG_DIR
        self._orig_cfg_path = vc.CONFIG_PATH
        vc.CONFIG_DIR = Path(self.tmpdir) / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"

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
        vc.CONFIG_DIR = self._orig_cfg_dir
        vc.CONFIG_PATH = self._orig_cfg_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path),
            limit=16 * 1024 * 1024,
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

    @staticmethod
    def _config_data_with_work_dir(
        work_dir: str,
    ) -> Callable[[dict[str, Any]], bool]:
        def _pred(msg: dict[str, Any]) -> bool:
            return (
                msg.get("type") == "configData"
                and msg.get("config", {}).get("work_dir") == work_dir
            )
        return _pred

    async def test_get_config_reports_each_windows_own_work_dir(
        self,
    ) -> None:
        """With no persisted work_dir, ``getConfig`` fills it from the
        requesting connection's own work_dir — so each VS Code window's
        settings panel can show its own workspace folder."""
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()
        await self._send(
            writer_a, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        await self._send(
            writer_b, {"type": "setWorkDir", "workDir": str(self.dir_b)},
        )

        await self._send(writer_a, {"type": "getConfig"})
        await self._drain_until(
            reader_a, self._config_data_with_work_dir(str(self.dir_a)),
        )

        await self._send(writer_b, {"type": "getConfig"})
        await self._drain_until(
            reader_b, self._config_data_with_work_dir(str(self.dir_b)),
        )

    async def test_get_config_prefers_connection_work_dir_over_persisted(
        self,
    ) -> None:
        """A connection that announced its own folder via ``setWorkDir``
        must see THAT folder in ``getConfig`` even when a different
        work_dir is persisted globally (e.g. saved by another webapp
        instance) — the stamped work_dir is what its commands actually
        run in.  A connection that never announced a folder still sees
        the persisted global value."""
        vc.save_config({"work_dir": str(self.dir_a)})

        reader_pinned, writer_pinned = await self._connect()
        await self._send(
            writer_pinned,
            {"type": "setWorkDir", "workDir": str(self.dir_b)},
        )
        await self._send(writer_pinned, {"type": "getConfig"})
        await self._drain_until(
            reader_pinned, self._config_data_with_work_dir(str(self.dir_b)),
        )

        reader_fresh, writer_fresh = await self._connect()
        await self._send(writer_fresh, {"type": "getConfig"})
        await self._drain_until(
            reader_fresh, self._config_data_with_work_dir(str(self.dir_a)),
        )

    async def test_save_config_work_dir_persists_and_updates_fallback(
        self,
    ) -> None:
        """A ``saveConfig`` carrying ``config.work_dir`` (sent by the
        standalone web client's editable settings field) persists the
        value and moves the daemon-wide fallback work_dir."""
        reader, writer = await self._connect()
        await self._send(
            writer,
            {"type": "saveConfig", "config": {"work_dir": str(self.dir_b)}},
        )
        await self._drain_until(
            reader, self._config_data_with_work_dir(str(self.dir_b)),
        )
        self.assertEqual(
            vc.load_config().get("work_dir"), str(self.dir_b),
        )
        self.assertEqual(
            self.server._vscode_server.work_dir, str(self.dir_b),
        )

    async def test_save_config_without_work_dir_keeps_persisted_value(
        self,
    ) -> None:
        """A VS Code window's ``saveConfig`` (which omits ``work_dir``
        because its field is read-only) must not clobber a previously
        persisted work_dir."""
        vc.save_config({"work_dir": str(self.dir_a)})
        reader, writer = await self._connect()
        await self._send(
            writer, {"type": "saveConfig", "config": {"max_budget": 42}},
        )
        await self._drain_until(
            reader,
            lambda m: (
                m.get("type") == "configData"
                and m.get("config", {}).get("max_budget") == 42
            ),
        )
        self.assertEqual(
            vc.load_config().get("work_dir"), str(self.dir_a),
        )
