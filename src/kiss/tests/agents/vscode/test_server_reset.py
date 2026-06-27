# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: the settings-panel "Server reset" button.

The chat webview's "Server reset" button (next to "Update") posts a
``serverReset`` command.  ``RemoteAccessServer._dispatch_client_command``
routes it to :meth:`RemoteAccessServer._handle_server_reset`, which
broadcasts a ``notification`` acknowledgement to the requesting window
and then schedules a ``SIGTERM`` to its own process so the supervising
LaunchAgent / systemd unit respawns a fresh daemon.

These tests bind a temporary UDS socket (not the production
``~/.kiss/sorcar.sock``) and drive a real client connection.  The
actual self-``SIGTERM`` is replaced on the server instance with a
recorder so the test process is never killed — the test asserts the
acknowledgement is delivered and that the restart trigger is scheduled
and fires after the configured delay.
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
from kiss.agents.vscode.web_server import RemoteAccessServer


def _find_free_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


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


class TestServerReset(IsolatedAsyncioTestCase):
    """Drive the ``serverReset`` command over a real UDS connection."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        from kiss.agents.vscode.web_server import _generate_self_signed_cert
        _generate_self_signed_cert(certfile, keyfile)

        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.port = _find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        # Replace the self-SIGTERM trigger with a recorder so the test
        # process is never killed.  ``_handle_server_reset`` calls
        # ``self._trigger_server_reset`` — an instance attribute shadows
        # the bound method.
        self._reset_fired = asyncio.Event()

        def _record_reset() -> None:
            self._reset_fired.set()

        self.server._trigger_server_reset = _record_reset  # type: ignore[method-assign]
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
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        for _ in range(max_events):
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            assert line, "UDS closed unexpectedly"
            msg = json.loads(line.decode("utf-8"))
            assert isinstance(msg, dict)
            if predicate(msg):
                return msg
        raise AssertionError("predicate never matched within budget")

    async def test_server_reset_acks_and_triggers_restart(self) -> None:
        """``serverReset`` broadcasts a notification then fires the restart."""
        reader, writer = await self._connect()

        await self._send(writer, {"type": "serverReset"})

        notification = await self._drain_until(
            reader,
            lambda m: m.get("type") == "notification"
            and "web server" in str(m.get("message", "")).lower(),
        )
        self.assertEqual(notification.get("id"), "server-reset-restarting")
        self.assertEqual(notification.get("severity"), "info")
        self.assertIn("restart", str(notification["message"]).lower())
        # The user requested a top-right notification, not a chat-output
        # ``notice`` note — guard against accidental regressions.
        self.assertNotEqual(notification.get("type"), "notice")

        # The restart trigger is scheduled via ``call_later`` and must
        # fire shortly after the acknowledgement.
        await asyncio.wait_for(self._reset_fired.wait(), timeout=5.0)
        self.assertTrue(self._reset_fired.is_set())

    async def test_server_reset_notification_only_to_requesting_window(
        self,
    ) -> None:
        """The reset acknowledgement reaches only the clicking window."""
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()

        await self._send(writer_a, {"type": "serverReset"})

        # Window A receives the notification.
        await self._drain_until(
            reader_a,
            lambda m: m.get("type") == "notification"
            and "web server" in str(m.get("message", "")).lower(),
        )

        # Window B must NOT receive the notification.  Probe B with an
        # ``activeTasksQuery`` (answered directly, bypassing broadcast)
        # and assert no reset notification arrived ahead of the response.
        await self._send(writer_b, {"type": "activeTasksQuery"})
        seen_notification = False

        def _is_probe_response(msg: dict[str, Any]) -> bool:
            nonlocal seen_notification
            if msg.get("type") == "notification" and "web server" in str(
                msg.get("message", ""),
            ).lower():
                seen_notification = True
            return msg.get("type") == "activeTasksResponse"

        await self._drain_until(reader_b, _is_probe_response)
        self.assertFalse(
            seen_notification,
            "reset notification leaked to a sibling window",
        )
