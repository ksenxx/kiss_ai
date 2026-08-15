# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared local ntfy.sh emulator for web-server tests.

The web server advertises its Cloudflare tunnel URL by POSTing to a
machine-stable ntfy.sh topic.  Tests must never POST to the real
https://ntfy.sh (doing so pollutes the production discovery topic with
fixture URLs), so every test that can reach the posting path points the
server at this local emulator — or at :func:`unroutable_base_url` when
it does not care about the post itself.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def unroutable_base_url() -> str:
    """Return a ``http://127.0.0.1:<port>`` URL nothing is listening on.

    Binds an ephemeral port and immediately closes it, so a subsequent
    connection attempt is refused instantly.  Used by tests that enable
    tunneling but do not assert on the ntfy post: the server's posting
    path swallows the connection error, and the real https://ntfy.sh is
    never contacted.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


class NtfyHandler(BaseHTTPRequestHandler):
    """Minimal ntfy.sh emulator.

    ``GET /{topic}/json?poll=1`` returns the cached messages for the
    topic as newline-delimited JSON, in chronological order.

    ``POST /{topic}`` appends the request body to the topic's cache.
    """

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
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


class NtfyServerContext:
    """Spin up the local ntfy emulator on a free port."""

    def __init__(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), NtfyHandler)
        self.server.messages = {}  # type: ignore[attr-defined]
        self.server.posts = []  # type: ignore[attr-defined]
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True,
        )
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

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
