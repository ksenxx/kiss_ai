# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the Unix-domain socket listener on RemoteAccessServer.

``RemoteAccessServer`` exposes a localhost UDS at ``~/.kiss/sorcar.sock``
(mode 0o600) alongside the public WSS port so the VS Code extension
can be a second client of the same backend ``VSCodeServer`` that
browsers already talk to.  Local clients speak the SAME
newline-delimited JSON protocol as WSS clients — no password
challenge, POSIX filesystem permissions gate access instead.

These tests bind a temporary socket under ``tmp_path`` (not the
production ``~/.kiss/sorcar.sock``) by passing ``uds_path=`` so
concurrent test runs do not race on the shared default path.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, skipIf

import kiss.agents.sorcar.persistence as th
from kiss.agents.vscode.web_server import RemoteAccessServer


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


@skipIf(sys.platform == "win32", "UDS not supported on Windows")
class TestUdsListener(IsolatedAsyncioTestCase):
    """End-to-end tests for the UDS listener."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)
        import kiss.agents.vscode.web_server as ws
        self._orig_grace = ws._TAB_CLOSE_GRACE
        ws._TAB_CLOSE_GRACE = 0.05

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

    async def asyncTearDown(self) -> None:
        import kiss.agents.vscode.web_server as ws
        ws._TAB_CLOSE_GRACE = self._orig_grace
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _read_event(
        self, reader: asyncio.StreamReader, timeout: float = 1.0,
    ) -> dict[str, object]:
        """Read one newline-delimited JSON message from the UDS."""
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        assert line, "UDS closed unexpectedly"
        msg = json.loads(line.decode("utf-8"))
        assert isinstance(msg, dict)
        return msg

    async def _drain_events(
        self,
        reader: asyncio.StreamReader,
        wanted_type: str,
        max_events: int = 50,
        timeout: float = 1.0,
    ) -> dict[str, object]:
        """Read events until *wanted_type* is observed or budget expires."""
        for _ in range(max_events):
            msg = await self._read_event(reader, timeout=timeout)
            if msg.get("type") == wanted_type:
                return msg
        raise AssertionError(
            f"did not observe a {wanted_type!r} event within "
            f"{max_events} messages",
        )

    async def test_socket_exists_with_owner_only_permissions(self) -> None:
        """UDS file is bound with mode 0o600 so only the owner can connect."""
        self.assertTrue(self.uds_path.exists())
        mode = self.uds_path.stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)
        self.assertTrue(stat.S_ISSOCK(self.uds_path.stat().st_mode))

    async def test_ready_yields_focusinput_to_uds_client(self) -> None:
        """A ``ready`` command over UDS produces a ``focusInput`` reply."""
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path),
            limit=16 * 1024 * 1024,
        )
        try:
            writer.write(
                json.dumps(
                    {"type": "ready", "tabId": "tab-uds-1",
                     "restoredTabs": []},
                ).encode("utf-8") + b"\n",
            )
            await writer.drain()
            focus = await self._drain_events(reader, "focusInput", timeout=2.0)
            self.assertEqual(focus.get("tabId"), "tab-uds-1")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def test_disconnect_arms_deferred_close(self) -> None:
        """Dropping the UDS arms a deferred ``closeTab`` for every tab seen."""
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path),
            limit=16 * 1024 * 1024,
        )
        writer.write(
            json.dumps(
                {"type": "ready", "tabId": "tab-uds-2",
                 "restoredTabs": []},
            ).encode("utf-8") + b"\n",
        )
        await writer.drain()
        await self._drain_events(reader, "focusInput", timeout=2.0)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        # Allow the server's _uds_handler finally block to run.
        for _ in range(50):
            with self.server._pending_tab_closes_lock:
                if "tab-uds-2" in self.server._pending_tab_closes:
                    break
            await asyncio.sleep(0.01)
        with self.server._pending_tab_closes_lock:
            self.assertIn(
                "tab-uds-2", self.server._pending_tab_closes,
            )

    async def test_broadcast_fans_out_to_uds_client(self) -> None:
        """Backend broadcasts reach UDS clients via the WebPrinter fan-out."""
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path),
            limit=16 * 1024 * 1024,
        )
        try:
            # Register the writer with the printer by sending ready;
            # the handler calls add_uds_writer before reading commands.
            writer.write(
                json.dumps(
                    {"type": "ready", "tabId": "tab-uds-3",
                     "restoredTabs": []},
                ).encode("utf-8") + b"\n",
            )
            await writer.drain()
            await self._drain_events(reader, "focusInput", timeout=2.0)

            # Directly broadcast an event from the printer (mirrors
            # what an agent task-runner thread does).  The event must
            # arrive over the UDS connection.
            self.server._printer.broadcast(
                {"type": "ping", "tabId": "tab-uds-3", "value": 42},
            )
            ping = await self._drain_events(reader, "ping", timeout=2.0)
            self.assertEqual(ping.get("value"), 42)
            self.assertEqual(ping.get("tabId"), "tab-uds-3")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def test_submit_with_large_attachment_is_processed(self) -> None:
        """A ``submit`` whose JSON line exceeds the default 64 KiB
        ``asyncio.StreamReader`` limit (e.g. a base64-encoded image
        attachment) must still be parsed and produce a ``setTaskText``
        broadcast.  Regression test for the bug where any task with an
        attached image/PDF was silently dropped because the UDS
        ``readline()`` raised ``LimitOverrunError`` and the handler's
        outer ``except Exception`` closed the connection.
        """
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path),
            limit=16 * 1024 * 1024,
        )
        try:
            writer.write(
                json.dumps(
                    {"type": "ready", "tabId": "tab-uds-att",
                     "restoredTabs": []},
                ).encode("utf-8") + b"\n",
            )
            await writer.drain()
            await self._drain_events(reader, "focusInput", timeout=2.0)

            # Build a submit whose serialised JSON length far exceeds
            # 64 KiB — the legacy default StreamReader limit.  200 KiB
            # of base64 data is plenty to overflow the old buffer.
            big_b64 = "A" * (200 * 1024)
            submit = {
                "type": "submit",
                "tabId": "tab-uds-att",
                "prompt": "look at this image",
                "model": "",
                "workDir": self.tmpdir,
                "attachments": [
                    {
                        "name": "screenshot.png",
                        "mimeType": "image/png",
                        "data": big_b64,
                    },
                ],
                "useWorktree": False,
                "useParallel": False,
            }
            line = json.dumps(submit).encode("utf-8") + b"\n"
            self.assertGreater(len(line), 64 * 1024)
            writer.write(line)
            await writer.drain()

            # The handler echoes the prompt via ``setTaskText`` before
            # invoking the task runner — its arrival proves the line
            # was successfully read and dispatched.
            echo = await self._drain_events(
                reader, "setTaskText", timeout=5.0,
            )
            self.assertEqual(echo.get("text"), "look at this image")
            self.assertEqual(echo.get("tabId"), "tab-uds-att")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def test_active_tasks_query_idle_daemon_returns_zero(self) -> None:
        """``activeTasksQuery`` returns count=0 when no agent is running.

        Locks in the wire format the VS Code extension's dependency
        installer relies on to decide whether to SIGTERM the daemon
        before the ``ensureDependencies`` post-install step.  An idle
        daemon must return ``{type: "activeTasksResponse", count: 0,
        tabs: []}`` so the installer is allowed to restart it on a
        fingerprint change.
        """
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path),
            limit=16 * 1024 * 1024,
        )
        try:
            query = json.dumps({"type": "activeTasksQuery"}).encode("utf-8")
            writer.write(query + b"\n")
            await writer.drain()
            msg = await self._drain_events(
                reader, "activeTasksResponse", timeout=2.0,
            )
            self.assertEqual(msg.get("count"), 0)
            self.assertEqual(msg.get("tabs"), [])
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def test_active_tasks_query_reports_running_tab(self) -> None:
        """When a tab claims to be running a task, the query reports it.

        Reproduces the SIGTERM regression by injecting a fake active
        ``_RunningAgentState`` into the registry and verifying the UDS
        query returns ``count=1`` plus a ``"<tab_id>(task=<id>)"``
        descriptor — the same shape the SIGTERM log line prints.  This
        is the signal the extension uses to defer the restart.
        """
        from kiss.agents.sorcar.running_agent_state import _RunningAgentState

        class _FakeTab:
            def __init__(self, tid: str) -> None:
                self.is_task_active = True
                self.task_history_id = tid
                self.last_task_id = tid

        fake_tab_id = "ad4ecb65-2878-4c2c-9736-3bb9be18814a"
        fake = _FakeTab("74")
        # Duck-typed insertion: ``_handle_active_tasks_query`` only
        # reads ``is_task_active`` / ``task_history_id`` / ``last_task_id``.
        # pyright cannot see through the runtime structural shape so the
        # value cast is needed to satisfy the registry's declared type.
        from typing import cast

        with _RunningAgentState._registry_lock:
            _RunningAgentState.running_agent_states[fake_tab_id] = cast(
                _RunningAgentState, fake,
            )
        try:
            reader, writer = await asyncio.open_unix_connection(
                str(self.uds_path),
                limit=16 * 1024 * 1024,
            )
            try:
                writer.write(
                    json.dumps({"type": "activeTasksQuery"}).encode("utf-8")
                    + b"\n",
                )
                await writer.drain()
                msg = await self._drain_events(
                    reader, "activeTasksResponse", timeout=2.0,
                )
                self.assertEqual(msg.get("count"), 1)
                tabs = msg.get("tabs")
                self.assertIsInstance(tabs, list)
                assert isinstance(tabs, list)
                self.assertEqual(len(tabs), 1)
                self.assertEqual(tabs[0], f"{fake_tab_id}(task=74)")
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
        finally:
            with _RunningAgentState._registry_lock:
                _RunningAgentState.running_agent_states.pop(fake_tab_id, None)

    async def test_stop_async_removes_socket(self) -> None:
        """``stop_async`` unlinks the socket file on shutdown."""
        # asyncTearDown calls stop_async after this; bind a fresh
        # server here to observe the unlink without disturbing the
        # shared fixture.
        certfile = Path(self.tmpdir) / "cert2.pem"
        keyfile = Path(self.tmpdir) / "key2.pem"
        from kiss.agents.vscode.web_server import _generate_self_signed_cert
        _generate_self_signed_cert(certfile, keyfile)
        local_uds = Path(self.tmpdir) / "sorcar-extra.sock"
        srv = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url-2.json",
            uds_path=local_uds,
        )
        await srv.start_async()
        self.assertTrue(local_uds.exists())
        await srv.stop_async()
        self.assertFalse(local_uds.exists())
