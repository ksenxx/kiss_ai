# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for ntfy.sh duplicate-URL suppression.

The web server posts the active Cloudflare tunnel URL to ntfy.sh so
remote subscribers can rediscover the URL after a restart.  When a
watchdog restart or named-tunnel re-registration produces the *same*
public hostname, re-publishing the URL would needlessly wake every
subscriber.  These tests run a real local HTTP server that emulates
ntfy.sh's poll + publish endpoints and verify that
:func:`_post_url_to_message_board` consults the topic's most recent
cached message and skips POSTs when the URL is unchanged.
"""

from __future__ import annotations

import json
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from kiss.agents.vscode.web_server import (
    _fetch_last_ntfy_message,
    _post_url_to_message_board,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


class _NtfyHandler(BaseHTTPRequestHandler):
    """Minimal ntfy.sh emulator.

    ``GET /{topic}/json?poll=1`` returns the cached messages for the
    topic as newline-delimited JSON, in chronological order.

    ``POST /{topic}`` appends the request body to the topic's cache.
    """

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        # Silence default stderr noise during tests.
        return

    def do_GET(self) -> None:  # noqa: N802
        path = self.path
        if "?" in path:
            path, query = path.split("?", 1)
        else:
            query = ""
        parts = path.strip("/").split("/")
        if len(parts) != 2 or parts[1] != "json" or "poll=1" not in query:
            self.send_response(404)
            self.end_headers()
            return
        topic = parts[0]
        store: dict[str, list[str]] = self.server.messages  # type: ignore[attr-defined]
        msgs = store.get(topic, [])
        lines: list[str] = []
        for body in msgs:
            lines.append(json.dumps({"event": "message", "message": body}))
        payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        topic = self.path.strip("/")
        if not topic or "/" in topic:
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length else ""
        store: dict[str, list[str]] = self.server.messages  # type: ignore[attr-defined]
        store.setdefault(topic, []).append(body)
        posts: list[tuple[str, str, dict[str, str]]] = (
            self.server.posts  # type: ignore[attr-defined]
        )
        posts.append((topic, body, dict(self.headers)))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"id":"x"}')


class _NtfyServerContext:
    """Spin up the local ntfy emulator on a free port."""

    def __init__(self) -> None:
        port = _find_free_port()
        self.server = ThreadingHTTPServer(("127.0.0.1", port), _NtfyHandler)
        # Shared mutable state accessed by handlers.
        self.server.messages = {}  # type: ignore[attr-defined]
        self.server.posts = []  # type: ignore[attr-defined]
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True,
        )
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{port}"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    @property
    def posts(self) -> list[tuple[str, str, dict[str, str]]]:
        return self.server.posts  # type: ignore[attr-defined,no-any-return]

    @property
    def messages(self) -> dict[str, list[str]]:
        return self.server.messages  # type: ignore[attr-defined,no-any-return]


class TestNtfyDeduplication(unittest.TestCase):
    """End-to-end verification of duplicate-URL suppression."""

    def setUp(self) -> None:
        self.ntfy = _NtfyServerContext()

    def tearDown(self) -> None:
        self.ntfy.stop()

    def test_fetch_returns_none_for_empty_topic(self) -> None:
        """An empty topic has no cached messages."""
        result = _fetch_last_ntfy_message(
            "empty-topic", base_url=self.ntfy.base_url,
        )
        self.assertIsNone(result)

    def test_first_post_succeeds_when_topic_empty(self) -> None:
        """No prior post means the URL must be published."""
        url = "https://red-fox-1234.trycloudflare.com"
        _post_url_to_message_board(url, base_url=self.ntfy.base_url)
        # Exactly one POST hit the server with the URL as body.
        self.assertEqual(len(self.ntfy.posts), 1)
        topic, body, _headers = self.ntfy.posts[0]
        self.assertTrue(topic.startswith("kiss-"))
        self.assertEqual(body, url)

    def test_duplicate_post_is_skipped(self) -> None:
        """Reposting the same URL must not hit ntfy.sh."""
        url = "https://red-fox-1234.trycloudflare.com"
        _post_url_to_message_board(url, base_url=self.ntfy.base_url)
        self.assertEqual(len(self.ntfy.posts), 1)
        # Second call with the same URL must be suppressed.
        _post_url_to_message_board(url, base_url=self.ntfy.base_url)
        self.assertEqual(len(self.ntfy.posts), 1)
        # Sanity: the cached message echoes back via fetch.
        topic = self.ntfy.posts[0][0]
        latest = _fetch_last_ntfy_message(topic, base_url=self.ntfy.base_url)
        self.assertEqual(latest, url)

    def test_different_url_is_posted(self) -> None:
        """A changed URL must be published even after a prior post."""
        first = "https://red-fox-1234.trycloudflare.com"
        second = "https://blue-bear-5678.trycloudflare.com"
        _post_url_to_message_board(first, base_url=self.ntfy.base_url)
        _post_url_to_message_board(second, base_url=self.ntfy.base_url)
        self.assertEqual(len(self.ntfy.posts), 2)
        self.assertEqual(self.ntfy.posts[0][1], first)
        self.assertEqual(self.ntfy.posts[1][1], second)
        # The latest cached message reflects the second URL.
        topic = self.ntfy.posts[0][0]
        latest = _fetch_last_ntfy_message(topic, base_url=self.ntfy.base_url)
        self.assertEqual(latest, second)

    def test_localhost_url_never_posted(self) -> None:
        """``https://localhost...`` URLs are not meant for ntfy."""
        _post_url_to_message_board(
            "https://localhost:8787", base_url=self.ntfy.base_url,
        )
        self.assertEqual(len(self.ntfy.posts), 0)

    def test_empty_url_never_posted(self) -> None:
        """Empty URLs are silently ignored."""
        _post_url_to_message_board("", base_url=self.ntfy.base_url)
        self.assertEqual(len(self.ntfy.posts), 0)

    def test_click_header_is_set_to_url(self) -> None:
        """The ``Click`` header makes the ntfy notification clickable.

        Without this header, tapping the message in the ntfy.sh web UI
        or the mobile app does nothing because the URL in the body is
        rendered as plain text.  Per
        https://docs.ntfy.sh/publish/#click-action, the ``Click``
        header is the only supported way to attach a navigation
        target to a message, so it must equal the URL we published.
        """
        url = "https://red-fox-1234.trycloudflare.com"
        _post_url_to_message_board(url, base_url=self.ntfy.base_url)
        self.assertEqual(len(self.ntfy.posts), 1)
        _topic, body, headers = self.ntfy.posts[0]
        self.assertEqual(body, url)
        # HTTP headers are case-insensitive; the BaseHTTPRequestHandler
        # preserves the on-the-wire casing, which our client sends as
        # ``Click``.  Look it up case-insensitively to stay robust.
        click = next(
            (v for k, v in headers.items() if k.lower() == "click"),
            None,
        )
        self.assertEqual(click, url)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
