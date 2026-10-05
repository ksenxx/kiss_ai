# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the daemon's local channel on RemoteAccessServer.

``RemoteAccessServer`` serves same-machine clients (the VS Code
extension, the ``sorcar`` CLI, ``run_agent`` sub-agents) over the same
WSS listener browsers use.  It publishes an endpoint file (mode 0o600)
carrying its URL, CA certificate path and a per-start token; a loopback
client that presents the token in its ``auth`` frame is admitted as a
local client and speaks the SAME newline-delimited JSON protocol as
remote WSS clients.  The endpoint file's mode gates access the way the
former Unix socket's 0o600 mode did.

These tests write a temporary endpoint file under ``tmp_path`` (not the
production ``~/.kiss/sorcar-local.json``) by passing
``local_endpoint_file=`` so concurrent test runs do not race on the
shared default path.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar import local_endpoint
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import posix_only
from kiss.tests.local_ws import LocalReader, open_local_connection


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore_persistence(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


class TestLocalListener(IsolatedAsyncioTestCase):
    """End-to-end tests for the local channel."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        from kiss.server.web_server import _generate_self_signed_cert
        _generate_self_signed_cert(certfile, keyfile)

        self.endpoint_file = Path(self.tmpdir) / "sorcar-local.json"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=self.endpoint_file,
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _read_event(
        self, reader: LocalReader, timeout: float = 1.0,
    ) -> dict[str, object]:
        """Read one newline-delimited JSON message from the local channel."""
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        assert line, "local connection closed unexpectedly"
        msg = json.loads(line.decode("utf-8"))
        assert isinstance(msg, dict)
        return msg

    async def _drain_events(
        self,
        reader: LocalReader,
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

    async def test_endpoint_file_describes_this_daemon(self) -> None:
        """The endpoint file names this daemon's port, token, CA and pid."""
        self.assertTrue(self.endpoint_file.exists())
        self.assertTrue(stat.S_ISREG(self.endpoint_file.stat().st_mode))
        endpoint = local_endpoint.read_endpoint(self.endpoint_file)
        assert endpoint is not None
        self.assertEqual(endpoint.token, self.server.local_token)
        self.assertEqual(endpoint.pid, os.getpid())
        self.assertEqual(endpoint.url, f"wss://127.0.0.1:{self.server.port}/ws")
        assert endpoint.ca is not None
        self.assertTrue(Path(endpoint.ca).is_file())

    @posix_only("chmod-based 0600 file mode")
    async def test_endpoint_file_has_owner_only_permissions(self) -> None:
        """The endpoint file is 0o600 so only the owner learns the token."""
        mode = self.endpoint_file.stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)

    async def test_wrong_token_is_not_admitted_as_local(self) -> None:
        """A loopback peer with a wrong token is not treated as local."""
        endpoint = local_endpoint.read_endpoint(self.endpoint_file)
        assert endpoint is not None
        async with connect(
            endpoint.url,
            ssl=local_endpoint.client_ssl_context(endpoint),
            open_timeout=10,
            close_timeout=2.0,
            compression=None,
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "token": "not-the-token"}))
            reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        self.assertEqual(reply.get("type"), "auth_required")

    async def test_ready_yields_focusinput_to_local_client(self) -> None:
        """A ``ready`` command over the local channel produces a ``focusInput`` reply."""
        reader, writer = await open_local_connection(self.server)
        try:
            writer.write(
                json.dumps(
                    {"type": "ready", "tabId": "tab-local-1",
                     "restoredTabs": []},
                ).encode("utf-8") + b"\n",
            )
            await writer.drain()
            focus = await self._drain_events(reader, "focusInput", timeout=2.0)
            self.assertEqual(focus.get("tabId"), "tab-local-1")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def test_broadcast_fans_out_to_local_client(self) -> None:
        """Backend broadcasts reach local clients via the WebPrinter fan-out."""
        reader, writer = await open_local_connection(self.server)
        try:
            writer.write(
                json.dumps(
                    {"type": "ready", "tabId": "tab-local-3",
                     "restoredTabs": []},
                ).encode("utf-8") + b"\n",
            )
            await writer.drain()
            await self._drain_events(reader, "focusInput", timeout=2.0)

            self.server._printer.broadcast(
                {"type": "ping", "tabId": "tab-local-3", "value": 42},
            )
            ping = await self._drain_events(reader, "ping", timeout=2.0)
            self.assertEqual(ping.get("value"), 42)
            self.assertEqual(ping.get("tabId"), "tab-local-3")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def test_submit_with_large_attachment_is_processed(self) -> None:
        """A ``submit`` whose JSON frame exceeds 64 KiB (e.g. a
        base64-encoded image attachment) must still be parsed and
        produce a ``setTaskText`` broadcast.  Regression test for the
        bug where any task with an attached image/PDF was silently
        dropped because the local channel's line reader capped a frame
        at 64 KiB and the handler's outer ``except Exception`` closed
        the connection.
        """
        reader, writer = await open_local_connection(self.server)
        try:
            writer.write(
                json.dumps(
                    {"type": "ready", "tabId": "tab-local-att",
                     "restoredTabs": []},
                ).encode("utf-8") + b"\n",
            )
            await writer.drain()
            await self._drain_events(reader, "focusInput", timeout=2.0)

            big_b64 = "A" * (200 * 1024)
            submit = {
                "type": "submit",
                "tabId": "tab-local-att",
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
                "isParallel": False,
            }
            line = json.dumps(submit).encode("utf-8") + b"\n"
            self.assertGreater(len(line), 64 * 1024)
            writer.write(line)
            await writer.drain()

            echo = await self._drain_events(
                reader, "setTaskText", timeout=5.0,
            )
            self.assertEqual(echo.get("text"), "look at this image")
            self.assertEqual(echo.get("tabId"), "tab-local-att")
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
        reader, writer = await open_local_connection(self.server)
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

        Reproduces the SIGTERM regression by registering an active
        ``AgentState`` in the registry and verifying the local query
        returns ``count=1`` plus a ``"<tab_id>(task=<id>)"``
        descriptor — the same shape the SIGTERM log line prints.  This
        is the signal the extension uses to defer the restart.
        """
        from kiss.server import agent_state
        from kiss.server.agent_state import AgentState

        fake_tab_id = "ad4ecb65-2878-4c2c-9736-3bb9be18814a"
        state = AgentState(
            "74",
            tab_id=fake_tab_id,
            server_owned=True,
            is_task_active=True,
        )
        agent_state.register(state)
        try:
            reader, writer = await open_local_connection(self.server)
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
            agent_state.unregister("74", state)

    async def test_stop_async_removes_endpoint_file(self) -> None:
        """``stop_async`` removes the endpoint file it wrote on shutdown."""
        certfile = Path(self.tmpdir) / "cert2.pem"
        keyfile = Path(self.tmpdir) / "key2.pem"
        from kiss.server.web_server import _generate_self_signed_cert
        _generate_self_signed_cert(certfile, keyfile)
        extra_endpoint = Path(self.tmpdir) / "sorcar-local-extra.json"
        srv = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url-2.json",
            local_endpoint_file=extra_endpoint,
        )
        await srv.start_async()
        self.assertTrue(extra_endpoint.exists())
        await srv.stop_async()
        self.assertFalse(extra_endpoint.exists())

    async def test_stop_async_keeps_a_successors_endpoint_file(self) -> None:
        """A daemon stopping after a successor took over must not delete
        the successor's endpoint file: the token in the file is the
        ownership witness."""
        certfile = Path(self.tmpdir) / "cert3.pem"
        keyfile = Path(self.tmpdir) / "key3.pem"
        from kiss.server.web_server import _generate_self_signed_cert
        _generate_self_signed_cert(certfile, keyfile)
        shared = Path(self.tmpdir) / "sorcar-local-shared.json"
        first = RemoteAccessServer(
            host="127.0.0.1", port=0, certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url-3.json",
            local_endpoint_file=shared,
        )
        await first.start_async()
        second = RemoteAccessServer(
            host="127.0.0.1", port=0, certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url-4.json",
            local_endpoint_file=shared,
        )
        await second.start_async()
        try:
            written = local_endpoint.read_endpoint(shared)
            assert written is not None
            self.assertEqual(written.token, second.local_token)
            await first.stop_async()
            still = local_endpoint.read_endpoint(shared)
            assert still is not None
            self.assertEqual(still.token, second.local_token)
        finally:
            await second.stop_async()
        self.assertFalse(shared.exists())
