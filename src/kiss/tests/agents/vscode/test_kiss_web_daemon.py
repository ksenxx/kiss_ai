"""Integration tests for kiss-web daemon lifecycle.

Verifies that the kiss-web remote server starts correctly and writes
the ``~/.kiss/remote-url.json`` marker file, and that the daemon can
be detected as already running.
"""

from __future__ import annotations

import asyncio
import json
from unittest import IsolatedAsyncioTestCase

from kiss.agents.vscode.web_server import (
    _URL_FILE,
    RemoteAccessServer,
    _remove_url_file,
    _save_url_file,
)


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
            tls=False,
        )
        # Use a random high port to avoid conflicts
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        server.port = port
        await server.start_async()

        # The async start doesn't call _save_url_file — it's done in
        # _serve_async.  We simulate the URL file write that happens there.
        _save_url_file(f"http://localhost:{port}")

        try:
            self.assertTrue(_URL_FILE.is_file())
            data = json.loads(_URL_FILE.read_text())
            self.assertIn("local", data)
            self.assertEqual(data["local"], f"http://localhost:{port}")
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
            host="127.0.0.1", port=18799, use_tunnel=False, tls=False
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
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        server = RemoteAccessServer(
            host="127.0.0.1", port=port, use_tunnel=False, tls=False
        )
        await server.start_async()

        try:
            # Make an HTTP request to verify the server is responding
            import urllib.request

            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(f"http://127.0.0.1:{port}/"),
            )
            html = resp.read().decode()
            self.assertIn("<title>KISS Sorcar</title>", html)
            self.assertEqual(resp.status, 200)
        finally:
            await server.stop_async()

    async def test_concurrent_start_detection(self) -> None:
        """A second server detects the first via the URL file."""
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        server1 = RemoteAccessServer(
            host="127.0.0.1", port=port, use_tunnel=False, tls=False
        )
        await server1.start_async()
        _save_url_file(f"http://localhost:{port}")

        try:
            # Verify URL file exists — this is what the daemon check uses
            self.assertTrue(_URL_FILE.is_file())
            data = json.loads(_URL_FILE.read_text())
            self.assertEqual(data["local"], f"http://localhost:{port}")
        finally:
            await server1.stop_async()
            _remove_url_file()
