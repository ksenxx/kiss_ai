# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: request/reply events are per-window.

Invariant: any agent or UI activity in the chat webview of one VS Code
window must not affect the behavior or UI of the chat webview in
another window.  Each window owns exactly one UDS connection to the
shared daemon; historically the daemon *broadcast* every request/reply
event (``files``, ``ghost``, ``models``, ``history``, ``frequentTasks``,
``inputHistory``, ``configData``, unknown-command ``error``) to every
connected client, so e.g. typing ``@`` in window A popped the file
picker dropdown in window B.

Now ``RemoteAccessServer._dispatch_client_command`` stamps a per-
connection ``connId`` on every command, the command handlers echo it
onto their reply events, and ``WebPrinter.broadcast`` delivers a
``connId``-stamped event ONLY to the connection that issued the
request (stripping the stamp from the wire payload).

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
from kiss.agents.vscode.web_server import RemoteAccessServer

# Event types that are direct replies to a single window's request and
# must therefore never appear on another window's connection.
_REPLY_TYPES = frozenset({
    "files", "ghost", "models", "history", "frequentTasks",
    "inputHistory", "configData",
})


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


def _has_type(reply_type: str) -> Callable[[dict[str, Any]], bool]:
    """Return a predicate matching events whose ``type`` is *reply_type*."""

    def _pred(msg: dict[str, Any]) -> bool:
        return msg.get("type") == reply_type

    return _pred


class TestPerWindowReplyIsolation(IsolatedAsyncioTestCase):
    """Two UDS connections (= two VS Code windows) sharing one daemon."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        from kiss.agents.vscode.web_server import _generate_self_signed_cert
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
        timeout: float = 10.0,
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

    async def _assert_no_reply_leak(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        """Probe a window and assert it saw no other window's replies.

        Sends an ``activeTasksQuery`` (answered directly, bypassing the
        broadcast pipeline) and reads everything queued on *reader* up
        to that response.  Because broadcast fan-out schedules sends to
        every connection in the same call, any leaked reply event from
        the other window would already be queued ahead of this probe's
        response — so seeing the response without any ``_REPLY_TYPES``
        event proves the replies were not broadcast.
        """
        await self._send(writer, {"type": "activeTasksQuery"})
        seen: list[str] = []

        def _is_probe_response(msg: dict[str, Any]) -> bool:
            seen.append(str(msg.get("type", "")))
            return msg.get("type") == "activeTasksResponse"

        await self._drain_until(reader, _is_probe_response)
        leaked = [t for t in seen if t in _REPLY_TYPES]
        self.assertEqual(
            leaked, [],
            f"another window's reply events leaked: {leaked}",
        )

    async def test_files_picker_reply_only_to_requesting_window(self) -> None:
        """Typing ``@`` in window A must not pop window B's file picker.

        ``getFiles`` replies (the ``files`` events that the webview's
        ``renderAutocomplete`` turns into the dropdown) must reach only
        the requesting connection.
        """
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()

        ws = Path(self.tmpdir) / "ws_a"
        ws.mkdir()
        (ws / "hello_world.py").write_text("print('hi')\n")

        await self._send(writer_a, {
            "type": "getFiles", "prefix": "", "workDir": str(ws),
        })
        # First reply: the immediate loading event; second: the ranked
        # list once the background scan finishes.
        await self._drain_until(
            reader_a,
            lambda m: m.get("type") == "files" and m.get("loading") is True,
        )
        ranked = await self._drain_until(
            reader_a,
            lambda m: m.get("type") == "files" and not m.get("loading"),
        )
        names = [f.get("text", "") for f in ranked.get("files", [])]
        self.assertIn("hello_world.py", names)
        # The connId routing stamp must never reach the wire.
        self.assertNotIn("connId", ranked)

        await self._assert_no_reply_leak(reader_b, writer_b)

    async def test_data_replies_only_to_requesting_window(self) -> None:
        """models/history/frequentTasks/inputHistory/configData replies
        from window A's requests must not repaint window B's UI."""
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()

        await self._send(writer_a, {"type": "getModels"})
        await self._send(writer_a, {
            "type": "getHistory", "query": "", "offset": 0, "generation": 7,
        })
        await self._send(writer_a, {"type": "getFrequentTasks"})
        await self._send(writer_a, {"type": "getInputHistory"})
        await self._send(writer_a, {"type": "getConfig"})

        for reply_type in (
            "models", "history", "frequentTasks", "inputHistory",
            "configData",
        ):
            msg = await self._drain_until(reader_a, _has_type(reply_type))
            self.assertNotIn("connId", msg)

        await self._assert_no_reply_leak(reader_b, writer_b)

    async def test_ghost_suggestion_only_to_requesting_window(self) -> None:
        """Window A's ghost-text suggestion must never be delivered to
        window B, even when both windows hold the same input text."""
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()

        await self._send(writer_a, {
            "type": "complete",
            "query": "isolated_tok",
            "activeFile": "/tmp/win_a.py",
            "activeFileContent": "isolated_token_window_a = 1\n",
        })
        ghost = await self._drain_until(
            reader_a,
            lambda m: m.get("type") == "ghost"
            and m.get("query") == "isolated_tok",
        )
        self.assertEqual(ghost.get("suggestion"), "en_window_a")
        self.assertNotIn("connId", ghost)

        await self._assert_no_reply_leak(reader_b, writer_b)

    async def test_unknown_command_error_only_to_sender(self) -> None:
        """An unknown command from window A must raise the error banner
        only in window A, never in window B."""
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()

        await self._send(writer_a, {"type": "definitelyNotACommand"})
        err = await self._drain_until(
            reader_a,
            lambda m: m.get("type") == "error"
            and "Unknown command" in str(m.get("text", "")),
        )
        self.assertIn("definitelyNotACommand", str(err.get("text", "")))

        await self._send(writer_b, {"type": "activeTasksQuery"})
        seen: list[dict[str, Any]] = []

        def _probe(msg: dict[str, Any]) -> bool:
            seen.append(msg)
            return msg.get("type") == "activeTasksResponse"

        await self._drain_until(reader_b, _probe)
        leaked = [
            m for m in seen
            if m.get("type") == "error"
            and "Unknown command" in str(m.get("text", ""))
        ]
        self.assertEqual(leaked, [], "error reply leaked to window B")

    async def test_reply_routing_survives_other_window_disconnect(
        self,
    ) -> None:
        """A window's replies keep flowing after a sibling window closes
        (the endpoint registry must drop only the departed binding)."""
        reader_a, writer_a = await self._connect()
        _reader_b, writer_b = await self._connect()

        writer_b.close()
        try:
            await writer_b.wait_closed()
        except Exception:
            pass
        # Give the daemon a beat to run the handler's cleanup path.
        deadline = time.time() + 5
        while time.time() < deadline:
            await asyncio.sleep(0.05)
            break

        await self._send(writer_a, {"type": "getModels"})
        msg = await self._drain_until(
            reader_a, lambda m: m.get("type") == "models",
        )
        self.assertIn("models", msg)
