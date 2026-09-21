# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A scripted local OpenAI-compatible chat-completions server for agent tests.

Tests drive real ``KISSAgent`` / ``SorcarAgent`` runs against this server
instead of a paid model: each POST is answered with the next scripted
body (the last one repeats), so a test controls exactly which tool the
"model" calls at each step and what token usage it reports.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODEL = "gpt-4.1-nano-2025-04-14"  # 500k window, cheap per-token accounting


def tool_call_body(
    name: str,
    arguments: dict[str, Any],
    prompt_tokens: int,
    completion_tokens: int = 100,
    cached_tokens: int = 0,
    text: str = "",
) -> bytes:
    """Build one chat-completion response that calls a tool.

    Args:
        name: Tool to call.
        arguments: Tool arguments.
        prompt_tokens: Reported prompt (context) size.
        completion_tokens: Reported completion size.
        cached_tokens: Reported prompt-cache read tokens.
        text: Assistant text sent along with the call.

    Returns:
        The JSON body.
    """
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if cached_tokens:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return json.dumps({
        "id": "chatcmpl-scripted",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text or f"Calling {name}.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": usage,
    }).encode()


def finish_body(summary: str, prompt_tokens: int = 5000, success: bool = True) -> bytes:
    """Build a response calling ``finish`` with the structured contract."""
    return tool_call_body(
        "finish",
        {"success": success, "is_continue": False, "summary_in_html": summary},
        prompt_tokens,
    )


class ScriptedHandler(BaseHTTPRequestHandler):
    """Answer each POST with the next scripted body; the last one repeats."""

    script: list[bytes] = []
    requests: list[dict[str, Any]] = []
    lock = threading.Lock()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the access log."""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        cls = type(self)
        with cls.lock:
            try:
                cls.requests.append(json.loads(raw))
            except ValueError:
                cls.requests.append({})
            index = min(len(cls.requests), len(cls.script)) - 1
            # Unique tool-call ids per turn, as a real provider issues.
            body = cls.script[index].replace(
                b'"call_1"', f'"call_{len(cls.requests)}"'.encode(),
            )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def serve(script: list[bytes]) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Serve *script* on a local port for the block's duration.

    Args:
        script: Response bodies in order; the last repeats forever.

    Yields:
        ``(base_url, requests)`` where *requests* fills with every
        decoded request body the server received.
    """
    requests: list[dict[str, Any]] = []
    handler = type(
        "Handler", (ScriptedHandler,),
        {"script": list(script), "requests": requests, "lock": threading.Lock()},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)
