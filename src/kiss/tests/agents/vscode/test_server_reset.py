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
import kiss.agents.vscode.web_server as web_server_mod
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

    async def test_server_reset_writes_pending_flag(self) -> None:
        """``_handle_server_reset`` drops the pending-reset flag file.

        The freshly-respawned daemon reads this flag on startup to
        decide whether to emit the paired "Server restart complete"
        notification.  Pinning the on-disk contract here guards
        against a refactor that silently drops the flag and
        therefore silently drops the post-restart toast.
        """
        flag_path = self.server._server_reset_flag_path()
        # Sanity: clean slate before the request.
        self.assertFalse(flag_path.exists())

        reader, writer = await self._connect()
        await self._send(writer, {"type": "serverReset"})
        await self._drain_until(
            reader,
            lambda m: m.get("type") == "notification"
            and m.get("id") == "server-reset-restarting",
        )

        self.assertTrue(
            flag_path.exists(),
            "_handle_server_reset must persist a pending-reset flag "
            "so the respawned daemon can broadcast 'Server restart "
            "complete'",
        )
        payload = json.loads(flag_path.read_text(encoding="utf-8"))
        self.assertIn("requested_at", payload)
        self.assertIsInstance(payload["requested_at"], (int, float))


class TestServerResetComplete(IsolatedAsyncioTestCase):
    """The post-restart "Server restart complete" broadcast.

    Simulates the freshly-respawned daemon by pre-creating the
    pending-reset flag file in the test's url-file directory
    *before* ``start_async`` runs.  The startup path then finds the
    flag, deletes it, and schedules the broadcast.  The module-level
    delay constant is shrunk so the test does not have to wait the
    full production window.
    """

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        from kiss.agents.vscode.web_server import _generate_self_signed_cert
        _generate_self_signed_cert(certfile, keyfile)

        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.port = _find_free_port()
        self.url_file = Path(self.tmpdir) / "remote-url.json"

        # Pre-create the flag file BEFORE start_async so the
        # freshly-bound server discovers it during _setup_server.
        flag_path = Path(self.tmpdir) / "server-reset-pending.json"
        flag_path.write_text(
            json.dumps({"requested_at": 0.0, "conn_id": ""}),
            encoding="utf-8",
        )
        self.flag_path = flag_path

        # Shrink the post-restart delay so the test does not wait
        # the full production window.  Stashed and restored in
        # tearDown so other tests see the production value.
        self._saved_delay = web_server_mod._SERVER_RESET_COMPLETE_DELAY
        web_server_mod._SERVER_RESET_COMPLETE_DELAY = 0.1

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=self.url_file,
            uds_path=self.uds_path,
        )
        await self.server.start_async()
        self._writers: list[asyncio.StreamWriter] = []

    async def asyncTearDown(self) -> None:
        web_server_mod._SERVER_RESET_COMPLETE_DELAY = self._saved_delay
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

    async def _drain_until(
        self,
        reader: asyncio.StreamReader,
        predicate: Callable[[dict[str, Any]], bool],
        max_events: int = 200,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        for _ in range(max_events):
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            assert line, "UDS closed unexpectedly"
            msg = json.loads(line.decode("utf-8"))
            assert isinstance(msg, dict)
            if predicate(msg):
                return msg
        raise AssertionError("predicate never matched within budget")

    async def test_pending_flag_triggers_complete_notification(
        self,
    ) -> None:
        """Flag file at startup ⇒ broadcast "Server restart complete"."""
        # The flag must be consumed eagerly at startup so a *second*
        # restart that crashes (no flag written) never replays the
        # toast.
        self.assertFalse(
            self.flag_path.exists(),
            "_setup_server must delete the pending-reset flag eagerly",
        )

        reader, _writer = await self._connect()
        notification = await self._drain_until(
            reader,
            lambda m: m.get("type") == "notification"
            and m.get("id") == "server-reset-complete",
        )
        self.assertEqual(notification.get("severity"), "info")
        self.assertIn(
            "restart complete",
            str(notification.get("message", "")).lower(),
        )
        # Top-right toast, not a chat-output banner — guard against
        # accidental regression.
        self.assertNotEqual(notification.get("type"), "notice")

    async def test_complete_notification_reaches_every_window(
        self,
    ) -> None:
        """All reconnecting clients see the "restart complete" toast.

        The requesting connection died with the previous daemon, so
        ``connId`` cannot be preserved across the restart.  The
        broadcast must therefore reach every currently-connected
        client — both the VS Code extension and any sibling browser
        webview that was disconnected by the SIGTERM.
        """
        reader_a, _writer_a = await self._connect()
        reader_b, _writer_b = await self._connect()

        notif_a = await self._drain_until(
            reader_a,
            lambda m: m.get("type") == "notification"
            and m.get("id") == "server-reset-complete",
        )
        notif_b = await self._drain_until(
            reader_b,
            lambda m: m.get("type") == "notification"
            and m.get("id") == "server-reset-complete",
        )
        # The broadcast must not be stamped with a ``connId`` —
        # otherwise WebPrinter.broadcast would route it to a single
        # connection and the other window would miss it.
        self.assertNotIn("connId", notif_a)
        self.assertNotIn("connId", notif_b)


class TestServerResetCompleteSuppressed(IsolatedAsyncioTestCase):
    """No flag at startup ⇒ no spurious "Server restart complete" toast.

    A routine launchd kick, crash respawn, or first-ever launch must
    NOT raise the post-restart notification — otherwise the user
    would see "Server restart complete" every time the daemon
    starts, which is misleading.
    """

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        from kiss.agents.vscode.web_server import _generate_self_signed_cert
        _generate_self_signed_cert(certfile, keyfile)

        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.port = _find_free_port()
        self.url_file = Path(self.tmpdir) / "remote-url.json"

        self._saved_delay = web_server_mod._SERVER_RESET_COMPLETE_DELAY
        web_server_mod._SERVER_RESET_COMPLETE_DELAY = 0.1

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=self.url_file,
            uds_path=self.uds_path,
        )
        await self.server.start_async()
        self._writers: list[asyncio.StreamWriter] = []

    async def asyncTearDown(self) -> None:
        web_server_mod._SERVER_RESET_COMPLETE_DELAY = self._saved_delay
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

    async def test_no_flag_means_no_complete_notification(self) -> None:
        """Absent flag ⇒ no "server-reset-complete" toast appears."""
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path), limit=16 * 1024 * 1024,
        )
        self._writers.append(writer)

        # Wait well past the (shrunk) delay so a missed scheduling
        # would have surfaced by now.  Use ``activeTasksQuery`` as
        # a probe: its reply is direct (bypasses broadcast) and
        # arrives after every queued startup broadcast would have.
        await asyncio.sleep(0.5)
        writer.write(
            json.dumps({"type": "activeTasksQuery"}).encode("utf-8") + b"\n",
        )
        await writer.drain()

        saw_complete = False
        for _ in range(50):
            line = await asyncio.wait_for(reader.readline(), timeout=2.0)
            assert line
            msg = json.loads(line.decode("utf-8"))
            assert isinstance(msg, dict)
            if (
                msg.get("type") == "notification"
                and msg.get("id") == "server-reset-complete"
            ):
                saw_complete = True
                break
            if msg.get("type") == "activeTasksResponse":
                break
        self.assertFalse(
            saw_complete,
            "no pending flag must NOT raise a 'Server restart "
            "complete' toast — that would fire on every cold start",
        )
