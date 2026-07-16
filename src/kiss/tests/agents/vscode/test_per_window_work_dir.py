# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: each VS Code window keeps its own work_dir.

Every VS Code window owns exactly one UDS connection to the shared
``kiss-web`` daemon and announces its open workspace folder via
``setWorkDir``.  The daemon records that folder per connection
(``RemoteAccessServer._dispatch_client_command``) and stamps it onto
every command from the same connection that lacks an explicit
``workDir``.

The invariant under test: two windows sharing one daemon can NEVER
observe each other's folder.  Before the per-connection state existed,
``setWorkDir`` only mutated the daemon-global fallback
``VSCodeServer.work_dir``, so the window that synced last silently
redirected autocomplete, commit-message generation and task launches
of every other window to its own folder.

These tests bind a temporary socket under ``tmp_path`` (not the
production ``~/.kiss/sorcar.sock``) and open two real UDS client
connections that simulate two VS Code windows.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
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


def _file_names(event: dict[str, Any]) -> list[str]:
    """Extract the file-name strings from a ``files`` event."""
    names: list[str] = []
    for entry in event.get("files", []):
        if isinstance(entry, dict):
            names.append(str(entry.get("text", "")))
        else:
            names.append(str(entry))
    return names


class TestPerWindowWorkDir(IsolatedAsyncioTestCase):
    """Two UDS connections (= two VS Code windows) with distinct folders."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)

        # Two workspace folders, one per simulated VS Code window.
        self.dir_a = Path(self.tmpdir) / "ws_a"
        self.dir_b = Path(self.tmpdir) / "ws_b"
        self.dir_a.mkdir()
        self.dir_b.mkdir()
        (self.dir_a / "alpha.txt").write_text("alpha")
        (self.dir_b / "beta.txt").write_text("beta")

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
        """Read events until *predicate* matches or the budget expires.

        Broadcasts are fanned out to every connection, so a reader may
        see events triggered by the other window's commands; the
        predicate is responsible for picking out the wanted one.
        """
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

    async def test_two_windows_keep_independent_work_dirs(self) -> None:
        """The core invariant: window B's ``setWorkDir`` must never
        redirect window A's work_dir-dependent commands to folder B.

        Window A syncs folder A, window B syncs folder B afterwards
        (so the daemon-global fallback now points at B).  A ``getFiles``
        WITHOUT an explicit ``workDir`` from window A must still scan
        folder A — before the per-connection work_dir existed it
        scanned folder B.
        """
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()

        await self._send(
            writer_a, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        await self._send(
            writer_b, {"type": "setWorkDir", "workDir": str(self.dir_b)},
        )

        # Window A: autocomplete without explicit workDir → folder A.
        await self._send(writer_a, {"type": "getFiles", "prefix": ""})
        ev_a = await self._drain_until(
            reader_a, self._files_event_with("alpha.txt"),
        )
        self.assertNotIn("beta.txt", _file_names(ev_a))

        # Window B: same command → folder B, not the other window's.
        await self._send(writer_b, {"type": "getFiles", "prefix": ""})
        ev_b = await self._drain_until(
            reader_b, self._files_event_with("beta.txt"),
        )
        self.assertNotIn("alpha.txt", _file_names(ev_b))

        # Window A again, AFTER window B synced last (the daemon-global
        # fallback now points at folder B): still folder A.
        await self._send(writer_a, {"type": "getFiles", "prefix": ""})
        ev_a2 = await self._drain_until(
            reader_a, self._files_event_with("alpha.txt"),
        )
        self.assertNotIn("beta.txt", _file_names(ev_a2))

    async def test_explicit_work_dir_wins_over_connection_work_dir(
        self,
    ) -> None:
        """A command carrying its own ``workDir`` must keep it.

        Per-tab routing (the webview stamps the active tab's folder on
        ``getFiles``) takes precedence over the connection-level
        default, so the stamping must never overwrite a non-empty
        ``workDir``.
        """
        reader_a, writer_a = await self._connect()
        await self._send(
            writer_a, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        await self._send(
            writer_a,
            {"type": "getFiles", "prefix": "", "workDir": str(self.dir_b)},
        )
        ev = await self._drain_until(
            reader_a, self._files_event_with("beta.txt"),
        )
        self.assertNotIn("alpha.txt", _file_names(ev))

    async def test_empty_set_work_dir_keeps_connection_work_dir(self) -> None:
        """An empty ``setWorkDir`` must not clear the window's folder."""
        reader_a, writer_a = await self._connect()
        await self._send(
            writer_a, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        await self._send(writer_a, {"type": "setWorkDir", "workDir": ""})
        await self._send(writer_a, {"type": "getFiles", "prefix": ""})
        ev = await self._drain_until(
            reader_a, self._files_event_with("alpha.txt"),
        )
        self.assertNotIn("beta.txt", _file_names(ev))

    async def test_commit_message_uses_connection_work_dir(self) -> None:
        """``generateCommitMessage`` without ``workDir`` must run in the
        requesting window's folder, not the daemon-global fallback.

        Folder A is NOT a git repository while folder B is — and B
        synced last, so the global fallback points at the git repo.
        Window A's request must still fail with "Not a git
        repository." (folder A), proving the connection's work_dir was
        used; window B's request reaches its own repo and fails with
        the no-staged-changes message instead.
        """
        subprocess.run(
            ["git", "init", "-q"], cwd=self.dir_b, check=True, timeout=30,
        )

        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()
        await self._send(
            writer_a, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        await self._send(
            writer_b, {"type": "setWorkDir", "workDir": str(self.dir_b)},
        )

        await self._send(
            writer_a, {"type": "generateCommitMessage", "tabId": "win-a"},
        )
        msg_a = await self._drain_until(
            reader_a,
            lambda m: (
                m.get("type") == "commitMessage" and m.get("tabId") == "win-a"
            ),
        )
        self.assertEqual(msg_a.get("error"), "Not a git repository.")

        await self._send(
            writer_b, {"type": "generateCommitMessage", "tabId": "win-b"},
        )
        msg_b = await self._drain_until(
            reader_b,
            lambda m: (
                m.get("type") == "commitMessage" and m.get("tabId") == "win-b"
            ),
        )
        self.assertIn("No staged changes", str(msg_b.get("error", "")))
