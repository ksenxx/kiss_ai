# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: per-window isolation of ghost-text autocomplete.

Follow-up to the per-window ``work_dir`` isolation
(``test_per_window_work_dir.py``): the daemon's remaining
window-global autocomplete state in :class:`VSCodeServer` is now keyed
by the ``connId`` that ``RemoteAccessServer._dispatch_client_command``
stamps on every command (one connection == one VS Code window):

* ``_last_active_file`` / ``_last_active_content`` — the fallback
  snapshot of the window's active editor used when a ``complete``
  command carries no ``activeFile``.  Globally shared, one window's
  file content leaked into another window's completion context.
* ``_complete_seq_latest`` — request-staleness tracking.  A single
  global counter let whichever window typed last mark every other
  window's in-flight ghost request stale (its result was silently
  dropped).

These tests bind a temporary socket under a temp dir (not the
production ``~/.kiss/sorcar.sock``) and open two real UDS client
connections that simulate two VS Code windows.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
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


class TestPerWindowAutocomplete(IsolatedAsyncioTestCase):
    """Two UDS connections (= two VS Code windows) sharing one daemon."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)

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
    def _ghost_for(query: str) -> Callable[[dict[str, Any]], bool]:
        """Predicate: the ``ghost`` event answering *query*."""
        def _pred(msg: dict[str, Any]) -> bool:
            return msg.get("type") == "ghost" and msg.get("query") == query
        return _pred

    async def test_active_file_snapshot_is_per_window(self) -> None:
        """Window A's active editor content must never feed window B's
        ghost-text completion.

        Window A sends a ``complete`` carrying its active file content
        (which contains a unique identifier).  Window B then sends a
        ``complete`` WITHOUT an active file (e.g. no visible editor in
        that window) whose query is a prefix of A's identifier.  Before
        the per-connection snapshot existed, the daemon fell back to
        the single global ``_last_active_content`` — i.e. window A's
        file — and completed A's identifier in window B.
        """
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()

        await self._send(writer_a, {
            "type": "complete",
            "query": "zebraq_marker",
            "activeFile": "/tmp/win_a_file.py",
            "activeFileContent": "x = zebraq_marker_only_in_window_a\n",
        })
        ghost_a = await self._drain_until(
            reader_a, self._ghost_for("zebraq_marker"),
        )
        self.assertEqual(
            ghost_a.get("suggestion"), "_only_in_window_a",
            "window A must complete from its own active file",
        )

        # Window B: same identifier prefix, no active editor.
        await self._send(writer_b, {
            "type": "complete",
            "query": "zebraq_mark",
        })
        ghost_b = await self._drain_until(
            reader_b, self._ghost_for("zebraq_mark"),
        )
        self.assertEqual(
            ghost_b.get("suggestion"), "",
            "window B must NOT see completions from window A's file",
        )

        # Window B reports its own editor and gets its own completion.
        await self._send(writer_b, {
            "type": "complete",
            "query": "yonder_token",
            "activeFile": "/tmp/win_b_file.py",
            "activeFileContent": "y = yonder_token_only_in_window_b\n",
        })
        ghost_b2 = await self._drain_until(
            reader_b, self._ghost_for("yonder_token"),
        )
        self.assertEqual(ghost_b2.get("suggestion"), "_only_in_window_b")

        # And window A keeps ITS snapshot: a follow-up complete from A
        # without activeFile still completes from A's file, not B's.
        await self._send(writer_a, {
            "type": "complete",
            "query": "zebraq_marker_on",
        })
        ghost_a2 = await self._drain_until(
            reader_a, self._ghost_for("zebraq_marker_on"),
        )
        self.assertEqual(ghost_a2.get("suggestion"), "ly_in_window_a")

    async def test_concurrent_windows_do_not_cancel_each_other(self) -> None:
        """A keystroke in window B must not mark window A's in-flight
        ghost request stale.

        Before staleness became per-connection, ``_complete_seq_latest``
        was a single global counter: window B's ``complete`` bumped it,
        and window A's already-queued request — still the newest from
        window A — was discarded by the worker, so window A never
        received a ghost for its query.  Both windows must receive a
        ghost answering their own (final) query.
        """
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()

        for round_no in range(5):
            qa = f"alphaq_round{round_no}_t"
            qb = f"betaq_round{round_no}_t"
            await self._send(writer_a, {
                "type": "complete",
                "query": qa,
                "activeFile": "/tmp/a.py",
                "activeFileContent": f"alphaq_round{round_no}_token = 1\n",
            })
            await self._send(writer_b, {
                "type": "complete",
                "query": qb,
                "activeFile": "/tmp/b.py",
                "activeFileContent": f"betaq_round{round_no}_token = 2\n",
            })
            ghost_a = await self._drain_until(reader_a, self._ghost_for(qa))
            self.assertEqual(ghost_a.get("suggestion"), "oken")
            ghost_b = await self._drain_until(reader_b, self._ghost_for(qb))
            self.assertEqual(ghost_b.get("suggestion"), "oken")

    async def test_set_work_dir_clears_only_callers_snapshot(self) -> None:
        """Window A switching folders must not wipe window B's
        active-file snapshot.

        Both windows seed their per-connection snapshots, then window A
        sends ``setWorkDir`` (its folder changed).  Window B's next
        ``complete`` without an active file must still complete from
        B's own snapshot; window A's must not (its snapshot referred to
        the previous workspace and was cleared).
        """
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()

        dir_a = Path(self.tmpdir) / "ws_a"
        dir_a2 = Path(self.tmpdir) / "ws_a2"
        dir_b = Path(self.tmpdir) / "ws_b"
        for d in (dir_a, dir_a2, dir_b):
            d.mkdir()

        await self._send(
            writer_a, {"type": "setWorkDir", "workDir": str(dir_a)},
        )
        await self._send(
            writer_b, {"type": "setWorkDir", "workDir": str(dir_b)},
        )

        await self._send(writer_a, {
            "type": "complete",
            "query": "gammaq_token",
            "activeFile": "/tmp/a.py",
            "activeFileContent": "gammaq_token_from_a = 1\n",
        })
        await self._drain_until(reader_a, self._ghost_for("gammaq_token"))
        await self._send(writer_b, {
            "type": "complete",
            "query": "deltaq_token",
            "activeFile": "/tmp/b.py",
            "activeFileContent": "deltaq_token_from_b = 2\n",
        })
        await self._drain_until(reader_b, self._ghost_for("deltaq_token"))

        # Window A opens a different folder.
        await self._send(
            writer_a, {"type": "setWorkDir", "workDir": str(dir_a2)},
        )

        # Window B's snapshot must survive window A's folder change.
        await self._send(writer_b, {"type": "complete", "query": "deltaq_tok"})
        ghost_b = await self._drain_until(
            reader_b, self._ghost_for("deltaq_tok"),
        )
        self.assertEqual(ghost_b.get("suggestion"), "en_from_b")

        # Window A's own snapshot was cleared (previous workspace).
        await self._send(writer_a, {"type": "complete", "query": "gammaq_tok"})
        ghost_a = await self._drain_until(
            reader_a, self._ghost_for("gammaq_tok"),
        )
        self.assertEqual(ghost_a.get("suggestion"), "")

    async def test_disconnect_drops_per_connection_state(self) -> None:
        """Closing a window's connection must free its autocomplete
        state in the daemon (no unbounded growth across reconnects)."""
        reader_a, writer_a = await self._connect()
        await self._send(writer_a, {
            "type": "complete",
            "query": "epsilonq_tok",
            "activeFile": "/tmp/a.py",
            "activeFileContent": "epsilonq_token = 1\n",
        })
        await self._drain_until(reader_a, self._ghost_for("epsilonq_tok"))

        vss = self.server._vscode_server
        with vss._state_lock:
            self.assertEqual(len(vss._last_active_file), 1)
            self.assertEqual(len(vss._complete_seq_latest), 1)

        writer_a.close()
        await writer_a.wait_closed()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with vss._state_lock:
                if (
                    not vss._last_active_file
                    and not vss._last_active_content
                    and not vss._complete_seq_latest
                ):
                    return
            await asyncio.sleep(0.02)
        self.fail(
            "per-connection autocomplete state was not dropped on "
            f"disconnect: files={vss._last_active_file} "
            f"seq={vss._complete_seq_latest}",
        )
