# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: a filesystem-root ``workDir`` is treated as absent.

A Dock/Finder-launched VS Code window with no folder open runs its
extension host with cwd ``/``.  Pre-fix extensions announced that root
via ``setWorkDir`` and stamped it on commands, which (a) pinned the
connection to ``/``, (b) poisoned the daemon-global fallback
``VSCodeServer.work_dir``, (c) persisted ``workDir: "/"`` into the tab
registry (``tabs.json``), and (d) rooted tasks and the ``@``-mention
file scan at the whole disk.  The extension-side guard
(``SorcarSidebarView._getWorkDir``) stops NEW roots at the source, but
old clients and already-poisoned registry entries still deliver roots
to the daemon.

These tests verify the daemon-side guard at every layer:

* :func:`kiss.core.utils.is_root_dir` — root classification.
* :class:`kiss.server.tab_registry.TabRegistry` — refuses new root
  work dirs and HEALS poisoned entries already persisted on disk.
* ``ServerApi.dispatch`` (driven end-to-end over a real UDS
  connection) — blanks a root ``workDir`` so it can neither pin the
  connection, nor poison the daemon-global fallback, nor scope a file
  scan to the whole disk.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
import kiss.core.vscode_config as vc
from kiss.core.utils import is_root_dir
from kiss.server.tab_registry import TabRegistry
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import requires_unix_sockets


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


def _file_names(event: dict[str, Any]) -> list[str]:
    """Extract the file-name strings from a ``files`` event."""
    names: list[str] = []
    for entry in event.get("files", []):
        if isinstance(entry, dict):
            names.append(str(entry.get("text", "")))
        else:
            names.append(str(entry))
    return names


class TestIsRootDir(unittest.TestCase):
    """Classification table for :func:`is_root_dir`."""

    def test_roots(self) -> None:
        """POSIX, Windows-drive and bare-backslash roots are roots,
        including root-equivalent spellings (``/./``, ``C:\\.\\``)."""
        for path in (
            "/", "//", "///", "\\", "\\\\", " / ",
            "/.", "/./", "/..", "/../", "/./..", "/a/..",
            "C:\\", "C:/", "c:\\", "c:/", "C:", "c:", "Z:\\",
            "C:\\.\\", "C:/./",
        ):
            self.assertTrue(is_root_dir(path), path)

    def test_non_roots(self) -> None:
        """Real folders, relative paths and blanks are not roots.

        Non-ASCII drive-like names (``é:``) are ordinary filenames —
        Node's ``path.win32`` recognizes no such drive — and UNC share
        prefixes are deliberately not classified (a ``//x/y`` path is
        legal on the daemon's POSIX filesystem).
        """
        for path in (
            "", "   ",
            "/Users/ksen/work/kiss", "/a", "/a/", "/a/../b",
            "C:\\proj", "C:/proj", "c:x",
            "ab", "9:", ":c", "src", "é:", "é:\\",
            "\\\\server\\share", "//server/share",
        ):
            self.assertFalse(is_root_dir(path), path)


class TestTabRegistryRootWorkDir(unittest.TestCase):
    """The registry never adopts — and actively heals — root work dirs."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.path = self.tmpdir / "tabs.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_heals_persisted_root_work_dir(self) -> None:
        """A poisoned ``tabs.json`` entry (the real ``workDir: "/"``
        left behind by a pre-fix no-folder window) loads with the root
        blanked while every other field survives."""
        self.path.write_text(json.dumps({"tabs": [{
            "tabId": "t1", "chatId": "c1", "title": "poisoned",
            "workDir": "/", "scopeWorkDir": "C:\\", "taskId": "task9",
        }, {
            "tabId": "t2", "chatId": "c2", "title": "healthy",
            "workDir": str(self.tmpdir), "scopeWorkDir": "",
            "taskId": "",
        }]}), encoding="utf-8")
        registry = TabRegistry(self.path)
        tabs = {t["tabId"]: t for t in registry.snapshot()}
        self.assertEqual(tabs["t1"]["workDir"], "")
        self.assertEqual(tabs["t1"]["scopeWorkDir"], "")
        self.assertEqual(tabs["t1"]["chatId"], "c1")
        self.assertEqual(tabs["t1"]["title"], "poisoned")
        self.assertEqual(tabs["t1"]["taskId"], "task9")
        self.assertEqual(tabs["t2"]["workDir"], str(self.tmpdir))

    def test_healing_is_persisted_at_flush_without_mutation(self) -> None:
        """A daemon that loads a poisoned file and shuts down without
        ever touching a tab must still leave the healed state on disk
        (via the shutdown ``flush``) — but NOT write at load time:
        construction alone must never touch the file, because a
        non-owner also constructs on the canonical path (the embedded
        launcher) and a load-time write would race the owning daemon."""
        poisoned = json.dumps({"tabs": [{
            "tabId": "t1", "chatId": "c1", "title": "poisoned",
            "workDir": "/", "scopeWorkDir": "", "taskId": "",
        }]})
        self.path.write_text(poisoned, encoding="utf-8")
        registry = TabRegistry(self.path)
        self.assertEqual(
            self.path.read_text(encoding="utf-8"), poisoned,
            "loading must not write the file",
        )
        registry.flush()
        on_disk = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["tabs"][0]["workDir"], "")

    def test_healing_is_persisted_on_first_mutation(self) -> None:
        """Any mutation persists the full healed state."""
        self.path.write_text(json.dumps({"tabs": [{
            "tabId": "t1", "chatId": "c1", "title": "poisoned",
            "workDir": "/", "scopeWorkDir": "", "taskId": "",
        }]}), encoding="utf-8")
        registry = TabRegistry(self.path)
        registry.update_tab("t1", title="renamed")
        on_disk = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["tabs"][0]["workDir"], "")

    def test_clean_file_is_not_rewritten_at_load_or_flush(self) -> None:
        """A file that sanitizes to itself is left untouched (no write
        churn on daemon start or shutdown)."""
        self.path.write_text(json.dumps({"tabs": [{
            "tabId": "t1", "chatId": "c1", "title": "healthy",
            "workDir": str(self.tmpdir), "scopeWorkDir": "",
            "taskId": "",
        }]}), encoding="utf-8")
        before = self.path.stat().st_mtime_ns
        registry = TabRegistry(self.path)
        registry.flush()
        self.assertEqual(self.path.stat().st_mtime_ns, before)

    def test_open_tab_refuses_root_work_dir(self) -> None:
        """A new tab opened with a root work dir stores ``""``."""
        registry = TabRegistry(self.path)
        self.assertTrue(registry.open_tab("t1", "title", "C:\\"))
        self.assertEqual(registry.snapshot()[0]["workDir"], "")

    def test_update_tab_ignores_root_but_accepts_real_dir(self) -> None:
        """A root update leaves the stored work dir untouched; a real
        directory still updates it (the normal per-tab repin)."""
        registry = TabRegistry(self.path)
        registry.open_tab("t1", "title", str(self.tmpdir))
        changed, _, _ = registry.update_tab("t1", work_dir="/")
        self.assertFalse(changed)
        self.assertEqual(registry.snapshot()[0]["workDir"], str(self.tmpdir))
        other = self.tmpdir / "sub"
        other.mkdir()
        changed, _, _ = registry.update_tab(
            "t1", work_dir=str(other), scope_work_dir="\\\\",
        )
        self.assertTrue(changed)
        self.assertEqual(registry.snapshot()[0]["workDir"], str(other))
        self.assertEqual(registry.snapshot()[0]["scopeWorkDir"], "")


@requires_unix_sockets
class TestDispatchRootWorkDirGuard(IsolatedAsyncioTestCase):
    """Root ``workDir`` values arriving over a real UDS connection."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)

        self._orig_cfg_dir = vc.CONFIG_DIR
        self._orig_cfg_path = vc.CONFIG_PATH
        vc.CONFIG_DIR = Path(self.tmpdir) / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"

        self.dir_a = Path(self.tmpdir) / "ws_a"
        self.dir_a.mkdir()
        (self.dir_a / "alpha.txt").write_text("alpha")

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
        """Open one UDS connection (simulates one VS Code window)."""
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
        """Read events until *predicate* matches or the budget expires."""
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
    def _files_event_with(name: str) -> Callable[[dict[str, Any]], bool]:
        """Predicate: a populated ``files`` event containing *name*."""
        def _pred(msg: dict[str, Any]) -> bool:
            return (
                msg.get("type") == "files"
                and not msg.get("loading")
                and name in _file_names(msg)
            )
        return _pred

    async def test_root_set_work_dir_cannot_override_pin(self) -> None:
        """A late ``setWorkDir('/')`` (a pre-fix no-folder window's
        announce) must neither repin the connection nor redirect its
        subsequent unstamped commands to the root."""
        reader, writer = await self._connect()
        await self._send(
            writer, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        await self._send(writer, {"type": "setWorkDir", "workDir": "/"})
        await self._send(writer, {"type": "setWorkDir", "workDir": "C:\\"})
        await self._send(writer, {"type": "getFiles", "prefix": ""})
        ev = await self._drain_until(
            reader, self._files_event_with("alpha.txt"),
        )
        self.assertEqual(_file_names(ev), ["alpha.txt"])

    async def test_root_set_work_dir_does_not_poison_global(self) -> None:
        """``setWorkDir('/')`` must leave the daemon-global fallback
        (``VSCodeServer.work_dir``) untouched — pre-guard it was
        adopted verbatim by ``_apply_new_work_dir``."""
        backend = self.server._vscode_server
        before = backend.work_dir
        reader, writer = await self._connect()
        await self._send(writer, {"type": "setWorkDir", "workDir": "/"})
        # Round-trip a command on the same connection so the
        # setWorkDir above is guaranteed processed before asserting.
        await self._send(
            writer,
            {"type": "getFiles", "prefix": "", "workDir": str(self.dir_a)},
        )
        await self._drain_until(reader, self._files_event_with("alpha.txt"))
        self.assertEqual(backend.work_dir, before)
        self.assertNotEqual(backend.work_dir, "/")

    async def test_explicit_root_work_dir_falls_back_to_pin(self) -> None:
        """A command stamped ``workDir: "/"`` (a poisoned tab in an
        old webview) must scan the connection's folder, not the whole
        disk."""
        reader, writer = await self._connect()
        await self._send(
            writer, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        await self._send(
            writer, {"type": "getFiles", "prefix": "", "workDir": "/"},
        )
        ev = await self._drain_until(
            reader, self._files_event_with("alpha.txt"),
        )
        self.assertEqual(_file_names(ev), ["alpha.txt"])

    async def test_unstamped_command_uses_safe_global_after_root_pin(
        self,
    ) -> None:
        """The primary pre-fix disaster path: a no-folder window whose
        only announce is ``setWorkDir('/')``.  Its unstamped commands
        must resolve to the (safe) daemon-global fallback, not the
        root."""
        reader_a, writer_a = await self._connect()
        # Window A establishes a known-safe daemon-global fallback
        # (``_cmd_set_work_dir`` adopts it) besides its own pin.
        await self._send(
            writer_a, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        reader_b, writer_b = await self._connect()
        await self._send(writer_b, {"type": "setWorkDir", "workDir": "/"})
        await self._send(writer_b, {"type": "getFiles", "prefix": ""})
        ev = await self._drain_until(
            reader_b, self._files_event_with("alpha.txt"),
        )
        self.assertEqual(_file_names(ev), ["alpha.txt"])

    async def test_save_config_root_work_dir_keeps_fallback(self) -> None:
        """``saveConfig`` carries ``work_dir`` NESTED in its config
        object — invisible to dispatch's top-level normalization — so
        ``_apply_new_work_dir`` itself must refuse the root."""
        backend = self.server._vscode_server
        reader, writer = await self._connect()
        await self._send(
            writer,
            {"type": "saveConfig", "config": {"work_dir": str(self.dir_a)}},
        )
        await self._drain_until(
            reader,
            lambda m: (
                m.get("type") == "configData"
                and m.get("config", {}).get("work_dir") == str(self.dir_a)
            ),
        )
        self.assertEqual(backend.work_dir, str(self.dir_a))
        await self._send(
            writer, {"type": "saveConfig", "config": {"work_dir": "/"}},
        )
        await self._drain_until(
            reader, lambda m: m.get("type") == "configData",
        )
        self.assertEqual(backend.work_dir, str(self.dir_a))


class TestStartupRootFallback(IsolatedAsyncioTestCase):
    """A root inherited at daemon startup must not become the fallback.

    A launchd/`open -a`-launched daemon inherits cwd ``/`` (and a
    poisoned environment can carry a root in ``KISS_WORKDIR``); the
    daemon-global fallback must degrade to the user's home instead.
    """

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)
        self._orig_env = os.environ.get("KISS_WORKDIR")
        os.environ["KISS_WORKDIR"] = "/"

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        from kiss.server.web_server import _generate_self_signed_cert
        _generate_self_signed_cert(certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=Path(self.tmpdir) / "sorcar.sock",
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._orig_env is None:
            os.environ.pop("KISS_WORKDIR", None)
        else:
            os.environ["KISS_WORKDIR"] = self._orig_env
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_root_kiss_workdir_degrades_to_home(self) -> None:
        """``KISS_WORKDIR=/`` must not root the daemon fallback."""
        backend = self.server._vscode_server
        self.assertNotEqual(backend.work_dir, "/")
        self.assertEqual(backend.work_dir, os.path.expanduser("~"))


if __name__ == "__main__":
    unittest.main()
