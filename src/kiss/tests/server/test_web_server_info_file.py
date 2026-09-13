# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``getInfoFile`` command.

The remote webapp's docked task-info panel (desktop mode) polls
``getInfoFile`` so its info subpanel can mirror ``tmp/PROGRESS.md`` under
the active tab's work dir.  These tests drive the REAL production
paths: a live :class:`RemoteAccessServer` over WSS for the read /
signature / fallback behaviors, and a real Unix-domain-socket
connection for the UDS drop-gate (VS Code windows never show that
panel, so the daemon must ignore a UDS-delivered ``getInfoFile``).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import ssl
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server.web_server import _OPEN_FILE_MAX_BYTES, RemoteAccessServer


def _find_free_port() -> int:
    """Find an available TCP port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


def _no_verify_ssl() -> ssl.SSLContext:
    """Return an SSL client context that skips certificate verification."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class TestGetInfoFileOverWss(IsolatedAsyncioTestCase):
    """``getInfoFile`` over a live WSS connection."""

    async def asyncSetUp(self) -> None:
        import kiss.agents.sorcar.persistence as _persistence

        self._saved_persistence = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        self._persistence_dir = Path(
            tempfile.mkdtemp(prefix="kiss_infofile_test_")
        )
        _persistence._KISS_DIR = self._persistence_dir
        _persistence._DB_PATH = self._persistence_dir / "sorcar.db"
        _persistence._db_conn = None

        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

        import kiss.agents.sorcar.persistence as _persistence

        if _persistence._db_conn is not None:
            try:
                _persistence._db_conn.close()
            except Exception:
                pass
            _persistence._db_conn = None
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_persistence

    async def _get_info_file(
        self, ws: Any, cmd_fields: dict[str, Any],
    ) -> dict[str, Any]:
        """Send one ``getInfoFile`` and return its ``infoFile`` reply."""
        await ws.send(json.dumps({"type": "getInfoFile", **cmd_fields}))
        deadline = asyncio.get_event_loop().time() + 5
        while asyncio.get_event_loop().time() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            ev: dict[str, Any] = json.loads(raw)
            if ev.get("type") == "infoFile":
                return ev
        raise AssertionError("no infoFile reply received")

    async def test_missing_file_replies_empty(self) -> None:
        """A workdir without tmp/PROGRESS.md replies exists=false, no text."""
        work_dir = self.server.work_dir
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-1"}
            )
        self.assertIs(reply["exists"], False)
        self.assertEqual(reply["content"], "")
        self.assertEqual(reply["sig"], "")
        self.assertEqual(reply["workDir"], work_dir)
        self.assertEqual(reply["tabId"], "t-1")
        self.assertNotIn("unchanged", reply)

    async def test_existing_file_replies_content_and_sig(self) -> None:
        """An existing tmp/PROGRESS.md is sent back with a non-empty sig."""
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        info.parent.mkdir(parents=True)
        info.write_text("# Status\n\nAll good.\n")
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-2", "token": "tok-7"}
            )
        self.assertIs(reply["exists"], True)
        self.assertEqual(reply["content"], "# Status\n\nAll good.\n")
        self.assertNotEqual(reply["sig"], "")
        st = info.stat()
        self.assertEqual(
            reply["sig"], f"{info}:{st.st_mtime_ns}:{st.st_size}"
        )
        self.assertEqual(reply["token"], "tok-7")

    async def test_matching_known_sig_replies_unchanged(self) -> None:
        """A poll whose knownSig matches skips the content re-send."""
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        info.parent.mkdir(parents=True)
        info.write_text("stable\n")
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            first = await self._get_info_file(ws, {"workDir": work_dir})
            second = await self._get_info_file(
                ws, {"workDir": work_dir, "knownSig": first["sig"]}
            )
        self.assertEqual(first["content"], "stable\n")
        self.assertIs(second["unchanged"], True)
        self.assertIs(second["exists"], True)
        self.assertEqual(second["sig"], first["sig"])
        self.assertNotIn("content", second)

    async def test_changed_file_replies_new_content(self) -> None:
        """A stale knownSig gets the rewritten file and a new sig."""
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        info.parent.mkdir(parents=True)
        info.write_text("before\n")
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            first = await self._get_info_file(ws, {"workDir": work_dir})
            info.write_text("after: longer content\n")
            second = await self._get_info_file(
                ws, {"workDir": work_dir, "knownSig": first["sig"]}
            )
        self.assertEqual(first["content"], "before\n")
        self.assertEqual(second["content"], "after: longer content\n")
        self.assertNotEqual(second["sig"], first["sig"])
        self.assertNotIn("unchanged", second)

    async def test_deleted_file_replies_empty_again(self) -> None:
        """Deleting the file flips the reply back to exists=false."""
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        info.parent.mkdir(parents=True)
        info.write_text("soon gone\n")
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            first = await self._get_info_file(ws, {"workDir": work_dir})
            info.unlink()
            second = await self._get_info_file(
                ws, {"workDir": work_dir, "knownSig": first["sig"]}
            )
        self.assertIs(second["exists"], False)
        self.assertEqual(second["content"], "")
        self.assertEqual(second["sig"], "")

    async def test_empty_workdir_falls_back_to_daemon_dir(self) -> None:
        """No workDir resolves tmp/PROGRESS.md under the daemon work dir."""
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        info.parent.mkdir(parents=True)
        info.write_text("fallback\n")
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(ws, {})
        self.assertIs(reply["exists"], True)
        self.assertEqual(reply["content"], "fallback\n")
        self.assertEqual(reply["workDir"], "")

    async def test_worktree_copy_wins_while_task_runs_there(self) -> None:
        """A tab with a recorded worktree reads the worktree's copy.

        A worktree-mode task maintains its ``tmp/PROGRESS.md`` inside
        the worktree, not the tab's workDir, so once the tab's
        ``worktree_created`` event recorded the worktree dir the poll
        must serve the worktree's file — and switching sources must
        change the sig even for byte-identical contents (the sig is
        path-prefixed), so the client repaints.
        """
        work_dir = self.server.work_dir
        main_copy = Path(work_dir) / "tmp" / "PROGRESS.md"
        main_copy.parent.mkdir(parents=True)
        main_copy.write_text("main tree\n")
        wt_dir = tempfile.mkdtemp(prefix="kiss_wt_test_")
        wt_copy = Path(wt_dir) / "tmp" / "PROGRESS.md"
        wt_copy.parent.mkdir(parents=True)
        wt_copy.write_text("worktree progress\n")
        # The production recording path: broadcast tracking of the
        # tab's worktree_created event.
        self.server._printer._track_worktree_event(
            {"type": "worktree_created", "worktreeWorkDir": wt_dir},
            "t-wt",
            None,
        )
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            wt_reply = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-wt"}
            )
            plain_reply = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-other"}
            )
        self.assertIs(wt_reply["exists"], True)
        self.assertEqual(wt_reply["content"], "worktree progress\n")
        self.assertTrue(wt_reply["sig"].startswith(str(wt_copy) + ":"))
        self.assertIs(plain_reply["exists"], True)
        self.assertEqual(plain_reply["content"], "main tree\n")
        self.assertNotEqual(plain_reply["sig"], wt_reply["sig"])

    async def test_missing_worktree_copy_falls_back_to_workdir(self) -> None:
        """A recorded worktree without the file falls back to workDir."""
        work_dir = self.server.work_dir
        main_copy = Path(work_dir) / "tmp" / "PROGRESS.md"
        main_copy.parent.mkdir(parents=True)
        main_copy.write_text("only in main\n")
        wt_dir = tempfile.mkdtemp(prefix="kiss_wt_test_")
        self.server._printer._track_worktree_event(
            {"type": "worktree_created", "worktreeWorkDir": wt_dir},
            "t-wt2",
            None,
        )
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-wt2"}
            )
        self.assertIs(reply["exists"], True)
        self.assertEqual(reply["content"], "only in main\n")
        self.assertTrue(reply["sig"].startswith(str(main_copy) + ":"))

    async def test_directory_named_progress_md_replies_empty(self) -> None:
        """tmp/PROGRESS.md that is a directory is treated as missing."""
        work_dir = self.server.work_dir
        (Path(work_dir) / "tmp" / "PROGRESS.md").mkdir(parents=True)
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(ws, {"workDir": work_dir})
        self.assertIs(reply["exists"], False)
        self.assertEqual(reply["content"], "")
        self.assertEqual(reply["sig"], "")

    async def test_fifo_named_progress_md_replies_empty_without_hanging(
        self,
    ) -> None:
        """A FIFO planted at tmp/PROGRESS.md is rejected, not read.

        The handler opens the path with ``O_NONBLOCK`` and checks the
        descriptor's ``fstat`` for a regular file, so a FIFO can
        neither hang the worker thread on open nor be streamed as
        content.
        """
        work_dir = self.server.work_dir
        fifo = Path(work_dir) / "tmp" / "PROGRESS.md"
        fifo.parent.mkdir(parents=True)
        os.mkfifo(fifo)
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(ws, {"workDir": work_dir})
        self.assertIs(reply["exists"], False)
        self.assertEqual(reply["content"], "")
        self.assertEqual(reply["sig"], "")

    async def test_oversized_file_replies_empty(self) -> None:
        """A file above _OPEN_FILE_MAX_BYTES is treated as missing."""
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        info.parent.mkdir(parents=True)
        with info.open("wb") as f:
            f.truncate(_OPEN_FILE_MAX_BYTES + 1)
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(ws, {"workDir": work_dir})
        self.assertIs(reply["exists"], False)
        self.assertEqual(reply["content"], "")

    async def test_non_utf8_bytes_are_replaced_not_fatal(self) -> None:
        """Undecodable bytes degrade to U+FFFD instead of an error."""
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        info.parent.mkdir(parents=True)
        info.write_bytes(b"ok \xff\xfe bytes\n")
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(ws, {"workDir": work_dir})
        self.assertIs(reply["exists"], True)
        self.assertEqual(reply["content"], "ok \ufffd\ufffd bytes\n")

    async def test_malformed_fields_are_blanked(self) -> None:
        """Non-string workDir / tabId / knownSig echo back as ''."""
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        info.parent.mkdir(parents=True)
        info.write_text("typed\n")
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(
                ws, {"workDir": 123, "tabId": ["x"], "knownSig": 7, "token": 5}
            )
        # The non-string workDir falls back to the daemon work dir,
        # where the file exists; the echoes are blanked.
        self.assertIs(reply["exists"], True)
        self.assertEqual(reply["content"], "typed\n")
        self.assertEqual(reply["workDir"], "")
        self.assertEqual(reply["tabId"], "")
        self.assertEqual(reply["token"], "")


class TestGetInfoFileOverUds(unittest.TestCase):
    """A UDS-delivered ``getInfoFile`` is dropped, not answered."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.sock_path = os.path.join(self.tmp.name, "sorcar-test.sock")
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True
        )
        self.loop_thread.start()
        self.server = RemoteAccessServer(
            uds_path=self.sock_path,
            url_file=os.path.join(self.tmp.name, "remote-url.json"),
        )
        self.server._printer._loop = self.loop
        self.uds_server: asyncio.Server = asyncio.run_coroutine_threadsafe(
            asyncio.start_unix_server(
                self.server._uds_handler, path=self.sock_path
            ),
            self.loop,
        ).result(timeout=5)

    def tearDown(self) -> None:
        async def _shutdown() -> None:
            self.uds_server.close()
            await self.uds_server.wait_closed()

        concurrent.futures.wait(
            [asyncio.run_coroutine_threadsafe(_shutdown(), self.loop)],
            timeout=5,
        )
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()
        self.tmp.cleanup()

    def test_uds_get_info_file_is_dropped(self) -> None:
        """The next event after a UDS getInfoFile is the follow-up's.

        VS Code windows (UDS clients) never show the remote webapp's
        docked task-info panel, so the daemon drops their
        ``getInfoFile`` without a reply — the first event this
        connection receives is the ``activeTasksQuery`` follow-up's
        response, never an ``infoFile``.
        """

        async def _talk() -> dict[str, Any]:
            reader, writer = await asyncio.open_unix_connection(
                self.sock_path
            )
            try:
                writer.write(
                    json.dumps({"type": "getInfoFile"}).encode() + b"\n"
                )
                writer.write(
                    json.dumps({"type": "activeTasksQuery"}).encode() + b"\n"
                )
                await writer.drain()
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                event: dict[str, Any] = json.loads(line)
                return event
            finally:
                writer.close()
                await writer.wait_closed()

        event = asyncio.run_coroutine_threadsafe(_talk(), self.loop).result(
            timeout=15
        )
        self.assertEqual(event.get("type"), "activeTasksResponse")


if __name__ == "__main__":
    unittest.main()
