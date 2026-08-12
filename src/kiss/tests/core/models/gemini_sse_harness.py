# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A real local Gemini endpoint for driving :class:`GeminiModel` end to end.

``google-genai`` resolves its API root from ``GOOGLE_GEMINI_BASE_URL``
(``google/genai/_base_url.py``), so pointing that variable at a
``ThreadingHTTPServer`` speaking the genuine ``streamGenerateContent``
SSE wire format exercises the real SDK, the real httpx transport and the
real adapter with no mocks, patches or test doubles anywhere.

The wire format is the one the SDK parses in
``ApiClient._iter_response_stream``: one ``data: <json>`` line per
``GenerateContentResponse`` chunk, blank-line separated.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def text_part(text: str, thought: bool = False) -> dict[str, Any]:
    """Build a response part carrying text.

    Args:
        text: The part's text.
        thought: Whether Gemini marks it as summarized reasoning.

    Returns:
        The part dict as it appears on the wire.
    """
    part: dict[str, Any] = {"text": text}
    if thought:
        part["thought"] = True
    return part


def function_call_part(
    name: str, args: dict[str, Any], thought_signature: str | None = None,
) -> dict[str, Any]:
    """Build a response part carrying a function call.

    Args:
        name: The function name.
        args: The call arguments.
        thought_signature: Optional base64 signature Gemini attaches to a
            tool call so the reasoning can be replayed on the next turn.

    Returns:
        The part dict as it appears on the wire.
    """
    part: dict[str, Any] = {"functionCall": {"name": name, "args": args}}
    if thought_signature is not None:
        part["thoughtSignature"] = thought_signature
    return part


def chunk(
    parts: list[dict[str, Any]],
    usage: dict[str, int] | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """Build one ``GenerateContentResponse`` chunk.

    Args:
        parts: The candidate's content parts.
        usage: Optional ``usageMetadata`` payload.
        finish_reason: Optional candidate ``finishReason``.

    Returns:
        The chunk dict as it appears on the wire.
    """
    candidate: dict[str, Any] = {"content": {"role": "model", "parts": parts}}
    if finish_reason is not None:
        candidate["finishReason"] = finish_reason
    body: dict[str, Any] = {"candidates": [candidate]}
    if usage is not None:
        body["usageMetadata"] = usage
    return body


class GeminiScript:
    """What the local endpoint does for the next request.

    Attributes:
        chunks: Chunks written, in order, before the trailing behaviour.
        after: ``"close"`` ends the response normally, ``"silent"`` holds
            the connection open writing nothing, ``"cut"`` drops the
            connection mid-body without a terminating chunk, and
            ``"keepalive"`` keeps writing SSE blank lines — real bytes
            that reset any byte-level read deadline, and that
            ``ApiClient._iter_response_stream`` discards (``if not line:
            continue``) before yielding, so the adapter sees nothing.
        serving: Set once the server has written its scripted chunks, so
            a test can time a Stop against a stream that is genuinely
            quiet.
        release: Set by the fixture teardown to free a silent handler.
        requests: The decoded JSON bodies the endpoint received.
    """

    def __init__(self) -> None:
        self.chunks: list[dict[str, Any]] = []
        self.after = "close"
        self.serving = threading.Event()
        self.release = threading.Event()
        self.requests: list[dict[str, Any]] = []

    def play(self, chunks: list[dict[str, Any]], after: str = "close") -> None:
        """Script the next response.

        Args:
            chunks: Chunks to write before the trailing behaviour.
            after: ``"close"``, ``"silent"``, ``"cut"`` or ``"keepalive"``.
        """
        self.chunks = chunks
        self.after = after
        self.serving.clear()


class _Handler(BaseHTTPRequestHandler):
    """Serves whatever :class:`GeminiScript` currently prescribes."""

    script: GeminiScript

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        """Answer a ``streamGenerateContent`` (or ``generateContent``) call."""
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        self.script.requests.append(json.loads(raw))
        streaming = "streamGenerateContent" in self.path
        if not streaming:
            self._send_unary()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for body in self.script.chunks:
            self.wfile.write(f"data: {json.dumps(body)}\r\n\r\n".encode())
            self.wfile.flush()
        self.script.serving.set()
        if self.script.after == "silent":
            self.script.release.wait(timeout=120.0)
        elif self.script.after == "keepalive":
            self._write_keepalives()
        elif self.script.after == "cut":
            self.close_connection = True
            self.wfile.write(b"data: {")
            self.wfile.flush()
            self.connection.close()

    def _write_keepalives(self) -> None:
        """Trickle SSE blank lines until released or the client hangs up."""
        while not self.script.release.wait(timeout=0.1):
            try:
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except OSError:
                return

    def _send_unary(self) -> None:
        """Answer a non-streaming call by merging the scripted chunks."""
        parts: list[dict[str, Any]] = []
        merged: dict[str, Any] = {}
        for body in self.script.chunks:
            parts.extend(body["candidates"][0]["content"]["parts"])
            if "usageMetadata" in body:
                merged["usageMetadata"] = body["usageMetadata"]
        merged["candidates"] = [{"content": {"role": "model", "parts": parts}}]
        payload = json.dumps(merged).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.script.serving.set()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the default stderr access log."""


class _DaemonServer(ThreadingHTTPServer):
    daemon_threads = True


def serve() -> Generator[tuple[str, GeminiScript]]:
    """Run a local Gemini endpoint for the duration of one test.

    Yields:
        The base URL to feed ``GOOGLE_GEMINI_BASE_URL`` and the script
        object controlling what the endpoint does.
    """
    script = GeminiScript()
    handler = type("_ScriptedHandler", (_Handler,), {"script": script})
    server = _DaemonServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", script
    finally:
        script.release.set()
        server.shutdown()
        server.server_close()
