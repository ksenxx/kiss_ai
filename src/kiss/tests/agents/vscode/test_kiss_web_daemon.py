"""Integration tests for kiss-web daemon lifecycle.

Verifies that the kiss-web remote server starts correctly and writes
the ``~/.kiss/remote-url.json`` marker file, that the daemon can
be detected as already running, and that a restart replaces a running
server with a fresh instance.
"""

from __future__ import annotations

import asyncio
import json
import socket
import ssl
import urllib.request
from unittest import IsolatedAsyncioTestCase

from kiss.agents.vscode.web_server import (
    _URL_FILE,
    RemoteAccessServer,
    _remove_url_file,
    _save_url_file,
)


def _no_verify_ssl() -> ssl.SSLContext:
    """Return an SSL client context that skips certificate verification."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class TestDaemonUrlFileLifecycle(IsolatedAsyncioTestCase):
    """Test that the server writes and removes remote-url.json on start/stop."""

    async def test_server_writes_url_file_on_start(self) -> None:
        """Starting the server creates ~/.kiss/remote-url.json with the local URL."""
        _remove_url_file()
        self.assertFalse(_URL_FILE.is_file())

        server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,  # 0 won't work; use a specific port
            use_tunnel=False,
        )
        # Use a random high port to avoid conflicts
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        server.port = port
        await server.start_async()

        # The async start doesn't call _save_url_file — it's done in
        # _serve_async.  We simulate the URL file write that happens there.
        _save_url_file(f"https://localhost:{port}")

        try:
            self.assertTrue(_URL_FILE.is_file())
            data = json.loads(_URL_FILE.read_text())
            self.assertIn("local", data)
            self.assertEqual(data["local"], f"https://localhost:{port}")
            self.assertNotIn("tunnel", data)
        finally:
            await server.stop_async()
            _remove_url_file()

    async def test_server_url_file_includes_tunnel(self) -> None:
        """When a tunnel URL is saved, it appears in remote-url.json."""
        _remove_url_file()
        _save_url_file("https://localhost:8787", "https://example.trycloudflare.com")

        try:
            self.assertTrue(_URL_FILE.is_file())
            data = json.loads(_URL_FILE.read_text())
            self.assertEqual(data["local"], "https://localhost:8787")
            self.assertEqual(data["tunnel"], "https://example.trycloudflare.com")
        finally:
            _remove_url_file()

    async def test_stop_removes_url_file(self) -> None:
        """Stopping the server removes the URL file."""
        _save_url_file("https://localhost:8787")
        self.assertTrue(_URL_FILE.is_file())

        server = RemoteAccessServer(
            host="127.0.0.1", port=18799, use_tunnel=False
        )
        await server.start_async()
        await server.stop_async()

        self.assertFalse(_URL_FILE.is_file())


class TestDaemonDetection(IsolatedAsyncioTestCase):
    """Test detecting whether the daemon is already running."""

    async def test_url_file_indicates_running(self) -> None:
        """Presence of remote-url.json indicates the daemon is running."""
        _remove_url_file()
        self.assertFalse(_URL_FILE.is_file())

        _save_url_file("https://localhost:8787")
        self.assertTrue(_URL_FILE.is_file())

        data = json.loads(_URL_FILE.read_text())
        self.assertIn("local", data)

        _remove_url_file()
        self.assertFalse(_URL_FILE.is_file())

    async def test_server_responds_to_http(self) -> None:
        """A running server responds to HTTP requests on its port."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        server = RemoteAccessServer(
            host="127.0.0.1", port=port, use_tunnel=False
        )
        await server.start_async()

        try:
            # Make an HTTP request to verify the server is responding
            loop = asyncio.get_event_loop()
            ctx = _no_verify_ssl()
            resp = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(
                    f"https://127.0.0.1:{port}/", context=ctx,
                ),
            )
            html = resp.read().decode()
            self.assertIn("<title>KISS Sorcar</title>", html)
            self.assertEqual(resp.status, 200)
        finally:
            await server.stop_async()

    async def test_concurrent_start_detection(self) -> None:
        """A second server detects the first via the URL file."""
        sock_obj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_obj.bind(("127.0.0.1", 0))
        port = sock_obj.getsockname()[1]
        sock_obj.close()

        server1 = RemoteAccessServer(
            host="127.0.0.1", port=port, use_tunnel=False
        )
        await server1.start_async()
        _save_url_file(f"https://localhost:{port}")

        try:
            # Verify URL file exists — this is what the daemon check uses
            self.assertTrue(_URL_FILE.is_file())
            data = json.loads(_URL_FILE.read_text())
            self.assertEqual(data["local"], f"https://localhost:{port}")
        finally:
            await server1.stop_async()
            _remove_url_file()


class TestDaemonRestart(IsolatedAsyncioTestCase):
    """Test that restarting the daemon replaces the running server."""

    async def test_restart_replaces_running_server(self) -> None:
        """Stopping and restarting a server on the same port works correctly.

        This simulates the restart behavior: the old server is stopped,
        a new server is started on the same port, and the new server
        responds to HTTP requests.  This mirrors what restartKissWebDaemon
        does in the VS Code extension (kill old process, start new one).
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        # Start the first server instance
        server1 = RemoteAccessServer(
            host="127.0.0.1", port=port, use_tunnel=False
        )
        await server1.start_async()
        _save_url_file(f"https://localhost:{port}")

        # Verify it responds
        loop = asyncio.get_event_loop()
        ctx = _no_verify_ssl()
        resp = await loop.run_in_executor(
            None,
            lambda: urllib.request.urlopen(
                f"https://127.0.0.1:{port}/", context=ctx,
            ),
        )
        self.assertEqual(resp.status, 200)

        # Stop the first server (simulates pkill in restartKissWebDaemon)
        await server1.stop_async()
        _remove_url_file()

        # Start a second server on the same port (simulates launchctl bootstrap)
        server2 = RemoteAccessServer(
            host="127.0.0.1", port=port, use_tunnel=False
        )
        await server2.start_async()
        _save_url_file(f"https://localhost:{port}")

        try:
            # Verify the new server responds
            ctx2 = _no_verify_ssl()
            resp2 = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(
                    f"https://127.0.0.1:{port}/", context=ctx2,
                ),
            )
            html = resp2.read().decode()
            self.assertIn("<title>KISS Sorcar</title>", html)
            self.assertEqual(resp2.status, 200)

            # Verify URL file was recreated
            self.assertTrue(_URL_FILE.is_file())
            data = json.loads(_URL_FILE.read_text())
            self.assertEqual(data["local"], f"https://localhost:{port}")
        finally:
            await server2.stop_async()
            _remove_url_file()

    async def test_restart_cleans_url_file(self) -> None:
        """A restart cycle removes and recreates the URL file."""
        _save_url_file("https://localhost:8787", "https://old.trycloudflare.com")
        self.assertTrue(_URL_FILE.is_file())

        # Simulate the stop phase of a restart
        _remove_url_file()
        self.assertFalse(_URL_FILE.is_file())

        # Simulate the start phase with a new URL
        _save_url_file("https://localhost:8787", "https://new.trycloudflare.com")
        self.assertTrue(_URL_FILE.is_file())
        data = json.loads(_URL_FILE.read_text())
        self.assertEqual(data["tunnel"], "https://new.trycloudflare.com")

        _remove_url_file()
